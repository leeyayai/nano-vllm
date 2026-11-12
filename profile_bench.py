"""在线推理瓶颈定位：实验驱动脚本（E0~E4）

放在项目根目录，与 nanovllm/ 同级；profiler.py 放到 nanovllm/utils/profiler.py。

与 serving_bench.py 的差异：
  1. 到达间隔改为指数分布（真正的泊松过程）。原脚本 np.random.poisson(1/rate) 产生整数间隔，
     请求在整数秒时刻成团到达，会把"突发"和"batch 过大"混在一起。
  2. TTFT/延迟以"计划到达时刻"为起点（开环），能反映 step 粒度造成的排队。
  3. 每次 run 重建 BlockManager，可用 --kv-fracs 截断 KV 池做容量扫描，无需重启进程。
  4. 同时接入 profiler，输出 steps.jsonl / preempts.jsonl / summary.json。

用法：
    python profile_bench.py --exp E1            # 到达率扫描
    python profile_bench.py --exp E2            # max_num_seqs 扫描
    python profile_bench.py --exp E3            # KV 容量扫描（截断 block 池）
    python profile_bench.py --exp E4            # block size 256 vs 512
    python profile_bench.py --exp E0            # profiler 自身开销校准（离线）
    python profile_bench.py --exp all
    python profile_bench.py --rates 8 --profile fine      # 单次运行（默认 --exp single）
然后：python analyze_profile.py prof_out
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np

from bench_fixes import setup_dynamo, warmup, compile_count, make_workload_v2
# single(): LLM(...) 之前 setup_dynamo(64)；原预热行换成 warmup(llm, SamplingParams)
# run_one(): 用 make_workload_v2 生成负载；n0 = compile_count() 开始前取，结束后断言 compile_count() == n0

# DEFAULT_MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
DEFAULT_MODEL = "/root/nano-vllm-workspace/nano-vllm/models/Qwen3-0.6B"


# ------------------------------------------------------------------ 负载
def make_workload(n, seed, max_in, max_out):
    rnd = random.Random(seed)
    prompts = [[rnd.randint(0, 10000) for _ in range(rnd.randint(100, max_in))] for _ in range(n)]
    max_toks = [rnd.randint(100, max_out) for _ in range(n)]
    return prompts, max_toks


def make_arrivals(n, rate, seed):
    """rate=inf：全部 t=0（离线）；否则指数间隔累加（泊松过程）"""
    if math.isinf(rate):
        return np.zeros(n)
    rng = np.random.default_rng(seed + 1)
    return np.cumsum(rng.exponential(1.0 / rate, n))


def _pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


# ------------------------------------------------------------------ 单次运行
def run_one(llm, SamplingParams, BlockManager, args, rate, kv_frac, tag, prof_mode):
    from nanovllm.utils.profiler import install

    sched = llm.scheduler
    assert sched.is_finished(), "上一轮请求尚未结束"
    if not hasattr(llm, "_full_kv_blocks"):
        llm._full_kv_blocks = len(sched.block_manager.blocks)
    n_blocks = max(1, int(llm._full_kv_blocks * kv_frac))
    sched.block_manager = BlockManager(n_blocks, args.block_size)      # 重置 KV 池与 prefix cache

    n = args.num_requests
    prompts, max_toks = make_workload(n, args.seed, args.max_input_len, args.max_output_len)
    arrivals = make_arrivals(n, rate, args.seed)
    print(f"[{tag}] rate={rate:g} kv_blocks={n_blocks}/{llm._full_kv_blocks} "
          f"max_num_seqs={args.max_num_seqs} block_size={args.block_size} profile={prof_mode}", flush=True)

    prof = install(llm, prof_mode)
    reqs, sent = {}, 0
    start = time.perf_counter()
    if prof:
        prof.t0 = start
    while sent < n or not llm.is_finished():
        now = time.perf_counter() - start
        while sent < n and now >= arrivals[sent]:
            llm.add_request(prompts[sent], SamplingParams(temperature=0.6, ignore_eos=True,
                                                          max_tokens=max_toks[sent]))
            sid = sched.waiting[-1].seq_id
            reqs[sid] = {"idx": sent, "input_len": len(prompts[sent]), "max_tokens": max_toks[sent],
                         "submit": start + float(arrivals[sent]), "first": None, "done": None, "out_len": None}
            sent += 1
        if sched.waiting or sched.running:
            outs, _ = llm.step()
            t = time.perf_counter()
            for sq in sched.running:                  # 本步后在 running 中 => 已产出首 token
                r = reqs.get(sq.seq_id)
                if r and r["first"] is None:
                    r["first"] = t
            for sid, ids in outs:
                r = reqs[sid]
                if r["first"] is None:
                    r["first"] = t
                r["done"], r["out_len"] = t, len(ids)
        else:
            time.sleep(0.001)
    total = time.perf_counter() - start

    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for sid, r in reqs.items():
        r = dict(r, seq_id=sid, preempts=(prof.preempt_counts.get(sid, 0) if prof else None))
        r["ttft"] = r["first"] - r["submit"]
        r["latency"] = r["done"] - r["submit"]
        r["tpot"] = (r["done"] - r["first"]) / (r["out_len"] - 1) if r["out_len"] > 1 else float("nan")
        rows.append(r)
    with open(os.path.join(out_dir, "requests.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ttft = [r["ttft"] for r in rows]
    lat = [r["latency"] for r in rows]
    tpot = [r["tpot"] for r in rows if r["tpot"] == r["tpot"]]
    total_out = sum(r["out_len"] for r in rows)
    summary = {
        "config": {"tag": tag, "rate": rate, "kv_frac": kv_frac, "kv_blocks": n_blocks,
                   "kv_blocks_full": llm._full_kv_blocks, "max_num_seqs": args.max_num_seqs,
                   "block_size": args.block_size, "num_requests": n, "profile": prof_mode, "seed": args.seed},
        "requests": {"total_time": total, "total_output": total_out, "throughput": total_out / total,
                     "ttft_p50": _pct(ttft, 50), "ttft_p90": _pct(ttft, 90), "ttft_p99": _pct(ttft, 99),
                     "tpot_mean_ms": float(np.mean(tpot)) * 1e3 if tpot else float("nan"),
                     "tpot_p99_ms": _pct(tpot, 99) * 1e3 if tpot else float("nan"),
                     "lat_p50": _pct(lat, 50), "lat_p90": _pct(lat, 90), "lat_p99": _pct(lat, 99)},
        "profiler": prof.dump(out_dir) if prof else None,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if prof:
        prof.uninstall()
    rq = summary["requests"]
    print(f"[{tag}] done: {total:.1f}s thr={rq['throughput']:.0f} tok/s "
          f"ttft_p50={rq['ttft_p50']:.2f}s tpot={rq['tpot_mean_ms']:.1f}ms "
          f"preempts={prof and summary['profiler']['preempt']['count']}", flush=True)
    return summary


def tag_of(prefix, rate, kv_frac, args):
    return f"{prefix}_r{rate:g}_kv{kv_frac:g}_mns{args.max_num_seqs}_bs{args.block_size}"


def single(args):
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.engine.sequence import Sequence

    # Sequence.block_size 在源码里写死为 256，不会随 kvcache_block_size 变化，
    # 改 block size 时必须同步，否则 num_blocks/slot_mapping 会算错。
    Sequence.block_size = args.block_size
    llm = LLM(args.model, enforce_eager=False, max_model_len=4096,
              max_num_seqs=args.max_num_seqs, kvcache_block_size=args.block_size)
    print(f"KV pool: {len(llm.scheduler.block_manager.blocks)} blocks x {args.block_size} tokens", flush=True)
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)       # 预热 compile / graph

    rates = [float(x) for x in args.rates.split(",")]
    fracs = [float(x) for x in args.kv_fracs.split(",")]
    for rate in rates:
        for frac in fracs:
            run_one(llm, SamplingParams, BlockManager, args, rate, frac,
                    tag_of(args.tag_prefix, rate, frac, args), args.profile)


# ------------------------------------------------------------------ 实验编排（每个配置一个子进程）
def orchestrate(args):
    def go(exp, tag, **kw):
        opts = dict(model=args.model, num_requests=args.num_requests, seed=args.seed,
                    max_input_len=args.max_input_len, max_output_len=args.max_output_len,
                    profile=args.profile, rates=args.rates, kv_fracs="1.0",
                    max_num_seqs=args.max_num_seqs, block_size=args.block_size)
        opts.update(kw)
        cmd = [sys.executable, os.path.abspath(__file__), "--exp", "single",
               "--out-dir", os.path.join(args.out_dir, exp), "--tag-prefix", tag]
        for k, v in opts.items():
            cmd += ["--" + k.replace("_", "-"), str(v)]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)

    exps = ["E0", "E1", "E2", "E3", "E4"] if args.exp == "all" else [args.exp]
    for exp in exps:
        if exp == "E0":      # profiler 开销：同一离线负载，三种模式
            for pm in ("off", "light", "fine"):
                go("E0", f"E0{pm}", rates="inf", profile=pm)
        elif exp == "E1":
            go("E1", "E1", rates="2,4,8,16,inf")
        elif exp == "E2":
            for mns in (512, 256, 128, 64, 32):
                go("E2", "E2", rates="8", max_num_seqs=mns)
        elif exp == "E3":
            go("E3", "E3", rates="8", kv_fracs="0.25,0.5,0.75,1.0")
        elif exp == "E4":
            for bs in (256, 512):
                go("E4", "E4", rates="8", block_size=bs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="single", choices=["single", "E0", "E1", "E2", "E3", "E4", "all"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--num-requests", type=int, default=256)
    p.add_argument("--rates", default="8", help="逗号分隔，inf 表示离线")
    p.add_argument("--kv-fracs", default="1.0", help="保留 KV block 池的比例（<=1），用于容量扫描")
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-input-len", type=int, default=1024)
    p.add_argument("--max-output-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", default=os.environ.get("NANOVLLM_PROFILE") or "light",
                   help="off | light | fine")
    p.add_argument("--out-dir", default="prof_out")
    p.add_argument("--tag-prefix", default="run")
    args = p.parse_args()
    if args.exp == "single":
        single(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()