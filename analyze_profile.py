"""汇总 profile_bench.py 的输出，并对照假设 H1~H6 给出判定线索

用法：
    python analyze_profile.py prof_out              # 汇总表 + 碎片反事实
    python analyze_profile.py prof_out --plot       # 额外为每个 run 画时间线（需要 matplotlib）
"""

import argparse
import json
import os

import numpy as np


def load_runs(root):
    runs = []
    for d, _, files in os.walk(root):
        if "summary.json" in files:
            with open(os.path.join(d, "summary.json")) as f:
                s = json.load(f)
            s["dir"] = d
            runs.append(s)
    return sorted(runs, key=lambda s: (s["config"]["tag"]))


def g(d, *keys, default=float("nan")):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def print_table(headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = lambda r: "|" + "|".join(f" {str(c):>{w}} " for c, w in zip(r, widths)) + "|"
    print(line); print(fmt(headers)); print(line)
    for r in rows:
        print(fmt(r))
    print(line)


def frag_counterfactual(req_path, blocks=(64, 128, 256, 512)):
    """用请求的输入/输出长度离线估算：若 block 大小为 b，decode 期间的内部碎片率（忽略抢占）"""
    reqs = [json.loads(l) for l in open(req_path)]
    res = {}
    for b in blocks:
        waste = tot = 0
        for r in reqs:
            if r["out_len"] is None:
                continue
            l = np.arange(r["input_len"], r["input_len"] + r["out_len"])
            w = (-l) % b
            waste += int(w.sum()); tot += int((l + w).sum())
        res[b] = waste / tot if tot else float("nan")
    return res


def decode_regression(run_dir, kv_bytes_per_token):
    """decode 步长 dt ≈ a + b * Σ上下文长度。
    a = 固定开销（CPU/launch/权重读取）；b 对应读 KV 的速度 => 等效带宽 = 每token KV字节 / b。
    同时给出峰值 block 占用（距离抢占还有多少余量）。"""
    st = [json.loads(l) for l in open(os.path.join(run_dir, "steps.jsonl"))]
    dec = [x for x in st if not x["is_prefill"]]
    if len(dec) < 10:
        return None
    X = np.array([x["tokens_in_use"] for x in dec], dtype=float)
    y = np.array([x["dt"] for x in dec])
    b, a = np.polyfit(X, y, 1)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - (a + b * X)) ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return {"peak_used": max(x["used"] for x in st), "total_blocks": st[0]["total_blocks"],
            "floor_ms": a * 1e3, "ns_per_token": b * 1e9,
            "bw_gbs": kv_bytes_per_token / b / 1e9 if b > 0 else float("nan"), "r2": r2, "n": len(dec)}


def plot_timeline(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = run["dir"]
    steps = [json.loads(l) for l in open(os.path.join(d, "steps.jsonl"))]
    pre = [json.loads(l) for l in open(os.path.join(d, "preempts.jsonl"))]
    t = [x["t"] for x in steps]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t, [x["used"] / x["total_blocks"] for x in steps], label="KV pool util")
    ax[0].plot(t, [x["frag_tokens"] / max(1, x["used"] * run["profiler"]["kv"]["block_size"]) for x in steps],
               label="internal frag")
    ax[0].legend(); ax[0].set_ylim(0, 1.05)
    ax[1].plot(t, [x["running"] for x in steps], label="running")
    ax[1].plot(t, [x["waiting"] for x in steps if "waiting" in x] if all("waiting" in x for x in steps)
               else [0] * len(t), label="waiting")
    ax[1].legend()
    ax[2].plot(t, [x["dt"] * 1e3 for x in steps], lw=0.6, label="step ms")
    ax[2].scatter([x["t"] for x in steps if x["is_prefill"]], [x["dt"] * 1e3 for x in steps if x["is_prefill"]],
                  s=6, c="tab:orange", label="prefill step")
    ax[2].scatter([p["t"] for p in pre], [0] * len(pre), marker="|", c="red", label="preempt")
    ax[2].set_yscale("log"); ax[2].legend(); ax[2].set_xlabel("time (s)")
    fig.suptitle(run["config"]["tag"])
    out = os.path.join(d, "timeline.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--kv-bytes-per-token", type=int, default=114688,
                    help="每 token 的 KV 字节数 = 2*层数*kv头数*head_dim*dtype字节；默认 Qwen3-0.6B bf16")
    a = ap.parse_args()
    runs = load_runs(a.root)
    if not runs:
        print("没有找到 summary.json"); return

    headers = ["tag", "thr tok/s", "ttft p50", "ttft p99", "tpot ms", "preempt", "/1k dec", "recomp%",
               "thrash%", "util%", "frag%", "prefill%", "non-fwd%"]
    rows = []
    for r in runs:
        rq, pf = r["requests"], r.get("profiler") or {}
        rows.append([
            r["config"]["tag"], f"{rq['throughput']:.0f}", f"{rq['ttft_p50']:.2f}", f"{rq['ttft_p99']:.2f}",
            f"{rq['tpot_mean_ms']:.1f}",
            g(pf, "preempt", "count", default="-"),
            f"{g(pf, 'preempt', 'per_1k_decode_steps'):.1f}",
            f"{100 * g(pf, 'prefill_tokens', 'recompute_share'):.1f}",
            f"{100 * g(pf, 'preempt', 'thrash_ratio'):.0f}",
            f"{100 * g(pf, 'kv', 'util_mean'):.0f}",
            f"{100 * g(pf, 'kv', 'frag_mean'):.1f}",
            f"{100 * g(pf, 'steps', 'prefill_time_share'):.0f}",
            f"{100 * g(pf, 'decode', 'non_forward_share'):.0f}",
        ])
    print("\n== 汇总（util=decode 步平均 KV 池占用；frag=内部碎片/已分配；non-fwd=decode 步中非 GPU 前向占比，仅 fine 模式）==")
    print_table(headers, rows)

    print("\n== 抢占细节 ==")
    for r in runs:
        pf = r.get("profiler") or {}
        pe = pf.get("preempt")
        if not pe:
            continue
        print(f"{r['config']['tag']}: 抢占序列数={pe['distinct_seqs']}  每序列抢占次数分布={pe['per_seq_hist']}  "
              f"抢占后 running 为空={pe['running_empty_preempts']}  浪费 token={pe['wasted_tokens']}  "
              f"重算中 cache 命中={g(pf, 'prefill_tokens', 're_hit_tokens'):.0f}  "
              f"抢占发生在最近一次 prefill 后 ≤1 步/≤5 步的比例="
              f"{pe['since_prefill_le1']:.2f}/{pe['since_prefill_le5']:.2f}  "
              f"准入停止原因={g(pf, 'kv', 'stall_reasons', default={})}")

    print("\n== prefill 打断 decode 的时长分布 (s) ==")
    for r in runs:
        st = (r.get("profiler") or {}).get("prefill_stall_s")
        if st:
            print(f"{r['config']['tag']}: n={st['n']} p50={st['p50']:.3f} p95={st['p95']:.3f} max={st['max']:.3f}")

    print("\n== decode 步长回归：dt ≈ 固定开销 + 斜率 × Σ上下文token（steps.jsonl）==")
    for r in runs:
        if not r.get("profiler"):
            continue
        reg = decode_regression(r["dir"], a.kv_bytes_per_token)
        if reg:
            print(f"{r['config']['tag']}: 峰值占用 {reg['peak_used']}/{reg['total_blocks']} block "
                  f"({100 * reg['peak_used'] / reg['total_blocks']:.0f}%)  固定开销 {reg['floor_ms']:.1f} ms  "
                  f"斜率 {reg['ns_per_token']:.2f} ns/token  等效KV读带宽 {reg['bw_gbs']:.0f} GB/s  "
                  f"R²={reg['r2']:.2f}  (n={reg['n']})")

    print("\n== E4 离线反事实：不同 block size 的 decode 期内部碎片率 ==")
    done = set()
    for r in runs:
        key = (r["config"]["rate"], r["config"]["num_requests"], r["config"]["seed"])
        if key in done:
            continue
        done.add(key)
        res = frag_counterfactual(os.path.join(r["dir"], "requests.jsonl"))
        print(f"rate={r['config']['rate']}: " + "  ".join(f"b={b}: {100 * v:.1f}%" for b, v in res.items()))

    if a.plot:
        for r in runs:
            if r.get("profiler"):
                plot_timeline(r)


if __name__ == "__main__":
    main()