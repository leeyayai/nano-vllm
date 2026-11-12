"""离线/在线压测结果统一汇总

对标两个既有脚本：
- bench.py          离线压测：所有请求一次性批量提交，测引擎峰值吞吐
- serving_bench.py  在线压测：按泊松到达率加请求，模拟真实在线服务负载

本脚本依次运行两种测试，用同一套驱动逻辑收集完全一致的指标集
（吞吐、TTFT、TPOT、端到端延迟、分位延迟、prefill/decode 步速、负载分布），
最后通过纯文本表格统一并排展示，并补充"在线/离线"对比列。

用法:
    python bench_summary.py
    python bench_summary.py --model /path/to/model --request-rate 8 --num-requests 256
"""

import argparse
import os
import time
from random import randint, seed

import numpy as np

from nanovllm import LLM, SamplingParams

# DEFAULT_MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

DEFAULT_MODEL = "/root/nano-vllm-workspace/nano-vllm/models/Qwen3-0.6B"

class RequestMetrics:
    """单个请求的阶段时间记录，用于 TTFT / TPOT / Latency"""

    def __init__(self, request_id, input_len, max_output_len):
        self.request_id = request_id
        self.input_len = input_len
        self.max_output_len = max_output_len
        self.submission_time = -1
        self.first_token_time = -1
        self.completion_time = -1
        self.output_len = -1

    def record_submission(self):
        self.submission_time = time.perf_counter()

    def record_first_token(self):
        if self.first_token_time == -1:
            self.first_token_time = time.perf_counter()

    def record_completion(self, output_ids):
        self.completion_time = time.perf_counter()
        self.output_len = len(output_ids)

    @property
    def ttft(self):
        return self.first_token_time - self.submission_time

    @property
    def tpot(self):
        if self.output_len > 1:
            return (self.completion_time - self.first_token_time) / (self.output_len - 1)
        return float("nan")

    @property
    def latency(self):
        return self.completion_time - self.submission_time


def make_workload(num_requests, max_input_len, max_output_len):
    """生成与 bench.py 一致的随机负载（seed 固定，保证两种测试公平）"""
    seed(0)
    prompts = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_requests)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(100, max_output_len),
        )
        for _ in range(num_requests)
    ]
    return prompts, sampling_params


def make_arrivals(mode, num_requests, request_rate):
    """到达时刻：offline 全部在 t=0；online 按泊松间隔累积"""
    if mode == "offline":
        return np.zeros(num_requests)
    np.random.seed(0)
    intervals = np.random.poisson(1.0 / request_rate, num_requests)
    return np.cumsum(intervals)


def run_benchmark(llm, mode, num_requests, max_input_len, max_output_len,
                  request_rate):
    """统一驱动循环：add_request + step，收集指标

    mode 仅决定到达过程，其余逻辑完全相同：
    - offline: 所有请求 t=0 同时提交（等价于 llm.generate 批量）
    - online:  按 request_rate 泊松到达（等价于 serving_bench.py）
    """
    label = "离线 offline" if mode == "offline" else "在线 serving"
    print(f"[{mode}] 开始运行 {label} benchmark "
          f"({num_requests} 请求, rate={request_rate} req/s) ...")

    prompts, sampling_params = make_workload(num_requests, max_input_len, max_output_len)
    arrival_times = make_arrivals(mode, num_requests, request_rate)

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)

    metrics = {}
    requests_sent = 0
    start = time.perf_counter()
    prefill_speeds, decode_speeds = [], []

    while requests_sent < num_requests or not llm.is_finished():
        current = time.perf_counter() - start
        while requests_sent < num_requests and current >= arrival_times[requests_sent]:
            prompt, sp = prompts[requests_sent], sampling_params[requests_sent]
            llm.add_request(prompt, sp)
            seq = llm.scheduler.waiting[-1]
            rm = RequestMetrics(seq.seq_id, len(prompt), sp.max_tokens)
            rm.record_submission()
            metrics[seq.seq_id] = rm
            requests_sent += 1

        if llm.scheduler.waiting or llm.scheduler.running:
            t0 = time.perf_counter()
            finished, num_tokens = llm.step()
            step_time = time.perf_counter() - t0
            if step_time > 0:
                if num_tokens > 0:
                    prefill_speeds.append(num_tokens / step_time)
                else:
                    decode_speeds.append(-num_tokens / step_time)
            for seq in llm.scheduler.running:
                if seq.seq_id in metrics:
                    metrics[seq.seq_id].record_first_token()
            for seq_id, output_ids in finished:
                if seq_id in metrics:
                    metrics[seq_id].record_first_token()
                    metrics[seq_id].record_completion(output_ids)
        else:
            time.sleep(0.01)

    total_time = time.perf_counter() - start
    print(f"[{mode}] 完成: 总耗时 {total_time:.2f}s")
    return summarize(metrics, num_requests, total_time, prefill_speeds, decode_speeds)


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def summarize(metrics, num_requests, total_time, prefill_speeds, decode_speeds):
    total_input = sum(m.input_len for m in metrics.values())
    total_output = sum(m.output_len for m in metrics.values() if m.output_len != -1)
    budget = sum(m.max_output_len for m in metrics.values())
    latencies = [m.latency for m in metrics.values() if m.completion_time != -1]
    ttfts = [m.ttft for m in metrics.values() if m.first_token_time != -1]
    tpots = [m.tpot for m in metrics.values() if not np.isnan(m.tpot)]
    input_lens = [m.input_len for m in metrics.values()]
    output_lens = [m.output_len for m in metrics.values() if m.output_len != -1]
    return {
        "num_requests": num_requests,
        "total_time": total_time,
        "total_input": total_input,
        "total_output": total_output,
        "budget": budget,
        "throughput": total_output / total_time,
        "budget_eff": total_output / budget if budget else 0.0,
        "avg_ttft": float(np.mean(ttfts)) if ttfts else float("nan"),
        "avg_tpot": float(np.mean(tpots)) if tpots else float("nan"),
        "avg_latency": float(np.mean(latencies)) if latencies else float("nan"),
        "p50": percentile(latencies, 0.50),
        "p90": percentile(latencies, 0.90),
        "p99": percentile(latencies, 0.99),
        "avg_prefill": float(np.mean(prefill_speeds)) if prefill_speeds else float("nan"),
        "avg_decode": float(np.mean(decode_speeds)) if decode_speeds else float("nan"),
        "avg_input_len": float(np.mean(input_lens)),
        "avg_output_len": float(np.mean(output_lens)),
    }


def disp_width(text):
    """按显示宽度计算（CJK 字符占 2，保证中英文混排对齐）"""
    width = 0
    for ch in str(text):
        o = ord(ch)
        if (0x1100 <= o <= 0x115F or o in (0x2329, 0x232A) or
                0x2E80 <= o <= 0xA4CF or 0xAC00 <= o <= 0xD7A3 or
                0xF900 <= o <= 0xFAFF or 0xFE30 <= o <= 0xFE4F or
                0xFF00 <= o <= 0xFF60 or 0xFFE0 <= o <= 0xFFE6 or
                0x20000 <= o <= 0x2FFFD or 0x30000 <= o <= 0x3FFFD):
            width += 2
        else:
            width += 1
    return width


def pad(text, width, align=">"):
    text = str(text)
    pad_total = max(0, width - disp_width(text))
    if align == "^":
        left = pad_total // 2
        return " " * left + text + " " * (pad_total - left)
    if align == "<":
        return text + " " * pad_total
    return " " * pad_total + text


def print_table(title, headers, rows, aligns=None):
    if aligns is None:
        aligns = [">"] * len(headers)
    widths = [disp_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], disp_width(cell))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print("\n" + title)
    print(sep)
    print("|" + "|".join(" " + pad(h, widths[i], "^") + " "
                         for i, h in enumerate(headers)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(" " + pad(c, widths[i], aligns[i]) + " "
                             for i, c in enumerate(row)) + "|")
    print(sep)


def format_ratio(online, offline):
    if offline in (0, float("nan")) or online is float("nan") or \
            online != online or offline != offline:
        return "N/A"
    return f"{online / offline:.3f}x"


def ms(value):
    return "N/A" if value != value else f"{value * 1000:.2f}"


def render_reports(offline, online, request_rate):
    print("\n" + "=" * 78)
    print("测试配置")
    print("=" * 78)
    print_table(
        "配置项对比",
        ["配置项", "离线 Offline", "在线 Serving", "说明"],
        [
            ["请求数", f"{offline['num_requests']}", f"{online['num_requests']}", "一致，保证公平"],
            ["输入长度", "100~1024", "100~1024", "一致，两者共用同一份随机负载"],
            ["输出预算", "100~1024", "100~1024", "一致，ignore_eos=True"],
            ["到达模式", "t=0 全部提交", "泊松到达", "在线按 rate 逐批注入"],
            ["request rate", "-", f"{request_rate} req/s", "仅在线生效"],
        ],
        aligns=["<", "^", "^", "<"],
    )

    print_table(
        "核心指标对比（在线/离线 列的 1.0x 表示相等）",
        ["指标", "离线 Offline", "在线 Serving", "在线/离线"],
        [
            ["请求数 (个)", f"{offline['num_requests']}", f"{online['num_requests']}", "-"],
            ["总耗时 (s)", f"{offline['total_time']:.2f}", f"{online['total_time']:.2f}",
             format_ratio(online["total_time"], offline["total_time"])],
            ["总输出 tokens", f"{offline['total_output']:,.0f}",
             f"{online['total_output']:,.0f}", format_ratio(online["total_output"], offline["total_output"])],
            ["吞吐量 (tok/s)", f"{offline['throughput']:,.2f}",
             f"{online['throughput']:,.2f}", format_ratio(online["throughput"], offline["throughput"])],
            ["平均 TTFT (ms)", ms(offline["avg_ttft"]), ms(online["avg_ttft"]), "-"],
            ["平均 TPOT (ms/tok)", ms(offline["avg_tpot"]), ms(online["avg_tpot"]), "-"],
            ["平均端到端延迟 (s)", f"{offline['avg_latency']:.2f}",
             f"{online['avg_latency']:.2f}", format_ratio(online["avg_latency"], offline["avg_latency"])],
            ["平均 prefill 速度 (tok/s)", f"{offline['avg_prefill']:,.0f}",
             f"{online['avg_prefill']:,.0f}", format_ratio(online["avg_prefill"], offline["avg_prefill"])],
            ["平均 decode 速度 (tok/s)", f"{offline['avg_decode']:,.0f}",
             f"{online['avg_decode']:,.0f}", format_ratio(online["avg_decode"], offline["avg_decode"])],
            ["输出预算利用率", f"{offline['budget_eff']:.1%}",
             f"{online['budget_eff']:.1%}", "-"],
        ],
        aligns=["<", ">", ">", "^"],
    )

    print_table(
        "归一化/每请求指标（用于直观对比两种模式的负载特性）",
        ["指标", "离线 Offline", "在线 Serving", "在线/离线"],
        [
            ["每请求平均输入长度 (tok)", f"{offline['avg_input_len']:.0f}",
             f"{online['avg_input_len']:.0f}", format_ratio(online["avg_input_len"], offline["avg_input_len"])],
            ["每请求平均输出长度 (tok)", f"{offline['avg_output_len']:.0f}",
             f"{online['avg_output_len']:.0f}", format_ratio(online["avg_output_len"], offline["avg_output_len"])],
            ["每请求平均延迟 (s)", f"{offline['avg_latency']:.2f}",
             f"{online['avg_latency']:.2f}", format_ratio(online["avg_latency"], offline["avg_latency"])],
            ["单请求 decode 速率 (tok/s)", f"{1e3 / offline['avg_tpot']:,.1f}"
             if offline["avg_tpot"] == offline["avg_tpot"] else "N/A",
             f"{1e3 / online['avg_tpot']:,.1f}"
             if online["avg_tpot"] == online["avg_tpot"] else "N/A", "-"],
        ],
        aligns=["<", ">", ">", "^"],
    )

    print_table(
        "端到端延迟分位 (s)",
        ["分位", "离线 Offline", "在线 Serving"],
        [
            ["p50", f"{offline['p50']:.2f}", f"{online['p50']:.2f}"],
            ["p90", f"{offline['p90']:.2f}", f"{online['p90']:.2f}"],
            ["p99", f"{offline['p99']:.2f}", f"{online['p99']:.2f}"],
        ],
        aligns=["<", ">", ">"],
    )

    print("\n" + "=" * 78)
    print("解读")
    print("=" * 78)
    thr = format_ratio(online["throughput"], offline["throughput"])
    lat = format_ratio(online["avg_latency"], offline["avg_latency"])
    print(f"  1. 吞吐：在线模式峰值吞吐约为离线的 {thr}，差异来自请求按 {request_rate} req/s"
          f" 泊松到达、引擎无法始终满batch并行。")
    print(f"  2. 延迟：在线模式平均端到端延迟约为离线的 {lat}，"
          f"请求分散到达、共享算力更多集中于单请求，换取更低延迟。")
    print(f"  3. 首 token：在线平均 TTFT 反映真实排队+prefill 时延，离线批量提交时大部分请求"
          f"同时进入 prefill，FIFO 中位于队列后的请求 TTFT 显著拉高。")
    print(f"  4. TPOT 在线为 {ms(online['avg_tpot'])}ms/tok，对应单请求 decode 约 "
          f"{1e3 / online['avg_tpot']:,.0f} tok/s 的实时速率。")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="离线与在线压测统一汇总（对标 bench.py 与 serving_bench.py）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型路径")
    parser.add_argument("--num-requests", type=int, default=256, help="请求数")
    parser.add_argument("--request-rate", type=int, default=8, help="在线到达率(req/s)")
    parser.add_argument("--max-input-len", type=int, default=1024, help="最大输入长度")
    parser.add_argument("--max-output-len", type=int, default=1024, help="最大输出长度")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 78)
    print("NANO-VLLM BENCHMARK SUMMARY (离线 vs 在线)")
    print("=" * 78)
    print(f"Model           : {args.model}")
    print(f"Num requests    : {args.num_requests}")
    print(f"Request rate    : {args.request_rate} req/s (仅在线)")
    print(f"Length range    : 输入 100~{args.max_input_len}, 输出 100~{args.max_output_len}")
    print(f"Start time      : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    llm = LLM(args.model, enforce_eager=True, max_model_len=4096)

    offline = run_benchmark(llm, "offline", args.num_requests,
                            args.max_input_len, args.max_output_len, args.request_rate)
    online = run_benchmark(llm, "online", args.num_requests,
                           args.max_input_len, args.max_output_len, args.request_rate)

    render_reports(offline, online, args.request_rate)


if __name__ == "__main__":
    main()
