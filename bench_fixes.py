"""profile_bench.py 的测量修复（放在项目根目录，与 profile_bench.py 同级）

用法一览：
  profile_bench.py 里：
    from bench_fixes import setup_dynamo, warmup, compile_count, make_workload_v2
  命令行：
    python bench_fixes.py prof_out/B1 --ttft-slo 2.0 --tpot-slo 0.05     # 聚合：多 seed 均值±std + SLO 容量
    python bench_fixes.py prof_out/B1 --prefill-reg                      # prefill 步长回归（dt ≈ a + b·tokens）
"""
import argparse
import json
import math
import os
import random
import re
from collections import defaultdict

import numpy as np


# ------------------------------------------------------------------ 1. dynamo / warmup
def setup_dynamo(limit=64):
    """必须在创建 LLM 之前调用：调高重编译上限，避免 rms_forward 之类函数在测量期间退回 eager"""
    import torch._dynamo as d
    for name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit"):
        if hasattr(d.config, name):
            setattr(d.config, name, max(getattr(d.config, name), limit))


def compile_count():
    """已编译图数量；测量前后各取一次，不相等说明测量期间发生了重编译（该次 run 应作废）"""
    try:
        from torch._dynamo.utils import counters
        return counters["stats"]["unique_graphs"]
    except Exception:
        return -1


def warmup(llm, SamplingParams, lens=(1, 17, 130, 513, 1024, 2048), batch=8):
    """覆盖不同 prefill 长度与不同 batch 形状，把编译全部放在计时之外"""
    rnd = random.Random(123)
    mk = lambda n: [rnd.randint(0, 10000) for _ in range(n)]
    sp = lambda: SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8)
    for n in lens:
        llm.generate([mk(n)], sp(), use_tqdm=False)
    llm.generate([mk(rnd.randint(50, 900)) for _ in range(batch)], [sp() for _ in range(batch)], use_tqdm=False)


# ------------------------------------------------------------------ 2. 负载：长尾长度 + 共享前缀
def make_workload_v2(n, seed, max_in=2048, max_out=1024, in_med=300, out_med=200, sigma=0.9,
                     shared_frac=0.5, n_prefix=4, prefix_len=512, max_len=4000):
    """ShareGPT 风格：输入/输出长度对数正态（长尾），shared_frac 的请求带 n_prefix 个系统提示之一。
    prefix_len 取 block_size 的整数倍，prefix cache 才能命中。若有真实 ShareGPT，可把长度换成真实分布。"""
    rnd = random.Random(seed)
    prefixes = [[rnd.randint(0, 10000) for _ in range(prefix_len)] for _ in range(n_prefix)]
    prompts, max_toks = [], []
    for _ in range(n):
        L = int(min(max_in, max(16, rnd.lognormvariate(math.log(in_med), sigma))))
        O = int(min(max_out, max(16, rnd.lognormvariate(math.log(out_med), sigma))))
        shared = rnd.random() < shared_frac
        L = min(L, max_len - O - (prefix_len if shared else 0))
        body = [rnd.randint(0, 10000) for _ in range(max(L, 1))]
        prompts.append(prefixes[rnd.randrange(n_prefix)] + body if shared else body)
        max_toks.append(O)
    return prompts, max_toks


# ------------------------------------------------------------------ 3. 稳态窗口 + SLO
def _jl(path):
    return [json.loads(l) for l in open(path)]


def analyze_run(run_dir, ttft_slo=2.0, tpot_slo=0.05, lo=0.2):
    """只看 [lo·T, T] 窗口（T=最后一个请求的到达时刻）：去掉冷启动与尾部排空。
    离线 rate=inf 没有稳态窗口，返回 None。"""
    from profile_bench import make_arrivals
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    c = s["config"]
    arr = make_arrivals(c["num_requests"], c["rate"], c["seed"])
    T = float(arr[-1])
    if T <= 0:
        return None
    t0, t1 = lo * T, T
    reqs = [r for r in _jl(os.path.join(run_dir, "requests.jsonl")) if t0 <= arr[r["idx"]] <= t1]
    if not reqs:
        return None
    ok = [r for r in reqs if r["ttft"] <= ttft_slo and (r["tpot"] != r["tpot"] or r["tpot"] <= tpot_slo)]
    out = {"n_win": len(reqs), "attain": len(ok) / len(reqs), "goodput_rps": len(ok) / (t1 - t0),
           "ttft_p99": float(np.percentile([r["ttft"] for r in reqs], 99)),
           "tpot_p99_ms": float(np.nanpercentile([r["tpot"] for r in reqs], 99)) * 1e3}
    sp = os.path.join(run_dir, "steps.jsonl")
    if os.path.exists(sp):  # 需要 light/fine 模式；off 模式没有 steps
        st = [x for x in _jl(sp) if t0 <= x["t"] <= t1]
        pre = [x for x in st if x["is_prefill"]]
        tot = sum(x["dt"] for x in st) or 1e-9
        out["steady_thr"] = sum(x.get("n_seqs", 0) for x in st) / (t1 - t0)   # 每步每序列产出 1 token
        out["prefill_share"] = sum(x["dt"] for x in pre) / tot
        out["prefill_steps_per_s"] = len(pre) / (t1 - t0)
        out["prefill_batch_mean"] = float(np.mean([x["n_seqs"] for x in pre])) if pre else float("nan")
    return out


def prefill_regression(run_dir):
    """prefill 步 dt ≈ a + b·新计算token：a 是与长度无关的固定开销（compile 回退/launch/H2D），b 是算力斜率"""
    st = [x for x in _jl(os.path.join(run_dir, "steps.jsonl")) if x["is_prefill"] and x.get("n_tokens")]
    if len(st) < 10:
        return None
    X = np.array([x["n_tokens"] for x in st], float)
    y = np.array([x["dt"] for x in st])
    b, a = np.polyfit(X, y, 1)
    r2 = 1 - ((y - (a + b * X)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"fixed_ms": a * 1e3, "us_per_token": b * 1e6, "r2": r2, "n": len(st),
            "single_req_ms": float(np.median([x["dt"] for x in st if x["n_seqs"] == 1] or [float("nan")])) * 1e3}


# ------------------------------------------------------------------ 4. 多 seed 聚合 + SLO 容量
ARM_RE = re.compile(r"^(.*?)_r(?:inf|[\d.]+)_kv")


def aggregate(root, **slo):
    groups = defaultdict(list)
    for d, _, files in os.walk(root):
        if "summary.json" not in files:
            continue
        c = json.load(open(os.path.join(d, "summary.json")))["config"]
        m = ARM_RE.match(c["tag"])
        r = analyze_run(d, **slo)
        if m and r:
            groups[(m.group(1), c["rate"])].append(r)
    keys = ["steady_thr", "attain", "goodput_rps", "ttft_p99", "tpot_p99_ms", "prefill_share", "prefill_batch_mean"]
    table = {}
    for k, rs in sorted(groups.items()):
        row = {"n_seeds": len(rs)}
        for key in keys:
            v = [r[key] for r in rs if key in r]
            row[key] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")) if v else None
        table[k] = row
    return table


def capacity(table, arm, thresh=0.95):
    """该 arm 下 SLO 达标率 ≥ thresh 的最大连续 rate（遇到第一个不达标就停）"""
    best = None
    for (a, rate), row in sorted(table.items(), key=lambda kv: kv[0][1]):
        if a != arm:
            continue
        if row["attain"][0] >= thresh:
            best = rate
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ttft-slo", type=float, default=2.0)
    ap.add_argument("--tpot-slo", type=float, default=0.05)
    ap.add_argument("--prefill-reg", action="store_true")
    a = ap.parse_args()
    if a.prefill_reg:
        for d, _, files in sorted(os.walk(a.root)):
            if "steps.jsonl" in files:
                r = prefill_regression(d)
                if r:
                    print(f"{os.path.basename(d)}: 固定开销 {r['fixed_ms']:.1f} ms  {r['us_per_token']:.1f} us/token  "
                          f"单请求 prefill 中位 {r['single_req_ms']:.1f} ms  R²={r['r2']:.2f}  n={r['n']}")
        return
    table = aggregate(a.root, ttft_slo=a.ttft_slo, tpot_slo=a.tpot_slo)
    f = lambda v, p=1: "-" if v is None else f"{v[0]:.{p}f}±{v[1]:.{p}f}"
    print(f"{'arm':>14} {'rate':>5} {'n':>2} {'steady thr':>14} {'attain':>12} {'goodput r/s':>13} "
          f"{'ttft p99':>12} {'tpot p99 ms':>13} {'prefill%':>10} {'pf batch':>10}")
    for (arm, rate), r in table.items():
        print(f"{arm:>14} {rate:>5g} {r['n_seeds']:>2} {f(r['steady_thr'], 0):>14} {f(r['attain'], 2):>12} "
              f"{f(r['goodput_rps'], 2):>13} {f(r['ttft_p99'], 2):>12} {f(r['tpot_p99_ms'], 1):>13} "
              f"{f(r['prefill_share'], 2):>10} {f(r['prefill_batch_mean'], 1):>10}")
    for arm in sorted({k[0] for k in table}):
        print(f"[容量] {arm}: TTFT p99≤{a.ttft_slo}s 且 TPOT≤{a.tpot_slo * 1e3:.0f}ms 的达标率≥95% 的最大 rate = {capacity(table, arm)}")


if __name__ == "__main__":
    main()