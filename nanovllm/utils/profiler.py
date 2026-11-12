"""nano-vllm 独立 profiling 模块（放到 nanovllm/utils/profiler.py）

设计目标：对引擎源码零侵入。通过给实例属性打补丁（monkeypatch）的方式包裹
step / schedule / postprocess / preempt / can_allocate / run 等方法，
关闭时不安装补丁，行为与原代码完全一致，并可通过 uninstall() 完整还原。

开关（环境变量 NANOVLLM_PROFILE）：
    0 / 不设置 : 关闭，install() 返回 None
    1 / light  : 轻量模式，只做 perf_counter 计时，不插入 CUDA 同步（用于吞吐对比）
    fine       : 细粒度模式，额外拆分 prepare / 模型前向 / 采样，
                 会插入 torch.cuda.synchronize()，有额外开销，仅用于瓶颈分析

用法：
    from nanovllm.utils.profiler import install
    prof = install(llm)          # llm 创建完、发请求之前调用
    ... 正常驱动 llm.step() ...
    prof.dump("prof_out/xxx")    # 写 steps.jsonl / preempts.jsonl / summary.json
    prof.uninstall()
"""

import json
import os
import time
from collections import Counter

import numpy as np


def normalize_mode(m):
    m = (m or "0").strip().lower()
    return {"1": "light", "true": "light", "light": "light", "fine": "fine"}.get(m, "off")


def mode():
    return normalize_mode(os.environ.get("NANOVLLM_PROFILE", "0"))


def _pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def _mean(v):
    return float(np.mean(v)) if len(v) else float("nan")


class Profiler:

    def __init__(self, engine, fine=False):
        self.engine = engine
        self.sched = engine.scheduler
        self.bm = self.sched.block_manager
        self.runner = engine.model_runner
        self.fine = fine
        self.t0 = time.perf_counter()
        self.steps = []                 # 每个 engine.step 一条记录
        self.preempts = []              # 每次抢占一条记录
        self.preempt_counts = {}        # seq_id -> 被抢占次数
        self.readmit_gaps = []          # 被抢占序列重新 prefill 距抢占的步数
        self._last_preempt_step = {}
        self._last_prefill_step = -1
        self._step_idx = 0
        self._cur = None
        self._patches = []

    # ---------------------------------------------------------------- 补丁管理
    def _patch(self, obj, name, fn):
        had = name in obj.__dict__
        self._patches.append((obj, name, had, obj.__dict__.get(name)))
        setattr(obj, name, fn)

    def uninstall(self):
        for obj, name, had, orig in reversed(self._patches):
            if had:
                setattr(obj, name, orig)
            else:
                obj.__dict__.pop(name, None)
        self._patches.clear()

    def _acc(self, key, dt):
        if self._cur is not None:
            self._cur[key] = self._cur.get(key, 0.0) + dt

    # ---------------------------------------------------------------- 安装
    def install(self):
        e, s, bm, r = self.engine, self.sched, self.bm, self.runner
        orig_step, orig_sched, orig_post = e.step, s.schedule, s.postprocess
        orig_preempt, orig_can, orig_run = s.preempt, bm.can_allocate, r.run
        clock = time.perf_counter

        def step():
            c = self._cur = {"i": self._step_idx, "t": clock() - self.t0,
                             "preempts": 0, "preempt_tokens": 0, "blocked_by_kv": 0}
            t = clock()
            try:
                out = orig_step()
            finally:
                c["dt"] = clock() - t
                self._finish_step(c)
                self._cur = None
                self._step_idx += 1
            return out

        def schedule():
            t = clock()
            seqs, is_prefill = orig_sched()
            c = self._cur
            if c is None:
                return seqs, is_prefill
            c["dt_sched"] = clock() - t
            c["is_prefill"] = is_prefill
            c["n_seqs"] = len(seqs)
            if is_prefill:
                new_t = re_t = re_hit = cached = 0
                for sq in seqs:
                    comp = len(sq) - sq.num_cached_tokens
                    cached += sq.num_cached_tokens
                    if sq.seq_id in self.preempt_counts:      # 被抢占后的重算
                        re_t += comp
                        re_hit += sq.num_cached_tokens
                        self.readmit_gaps.append(c["i"] - self._last_preempt_step.get(sq.seq_id, c["i"]))
                    else:
                        new_t += comp
                c.update(n_tokens=new_t + re_t, new_tokens=new_t, re_tokens=re_t,
                         re_hit_tokens=re_hit, cached_tokens=cached)
                self._last_prefill_step = c["i"]
            else:
                c["n_tokens"] = len(seqs)
            c["waiting"] = len(s.waiting)
            if s.waiting:       # 队首请求没有被准入，记录原因
                if c["blocked_by_kv"]:
                    c["stall"] = "kv"
                elif is_prefill and len(seqs) >= s.max_num_seqs:
                    c["stall"] = "max_seqs"
                else:
                    c["stall"] = "token_budget"
            return seqs, is_prefill

        def postprocess(seqs, token_ids,is_prefill):
            t = clock()
            r_ = orig_post(seqs, token_ids,is_prefill)
            self._acc("dt_post", clock() - t)
            return r_

        def preempt(seq):
            c = self._cur
            self.preempts.append({
                "step": self._step_idx, "t": clock() - self.t0, "seq_id": seq.seq_id,
                "len": len(seq), "completion": seq.num_completion_tokens,
                "blocks": len(seq.block_table),
                # 调度器先 pop 再调用 preempt，无法区分"抢占别人"与"自我抢占"；
                # running_after==0 表示 KV 池连一个序列继续增长都撑不住
                "running_after": len(s.running),
                "free": len(bm.free_block_ids), "used": len(bm.used_block_ids),
                "waiting": len(s.waiting),
                "since_prefill": self._step_idx - self._last_prefill_step,
                "prev_preempts": self.preempt_counts.get(seq.seq_id, 0),
            })
            self.preempt_counts[seq.seq_id] = self.preempt_counts.get(seq.seq_id, 0) + 1
            self._last_preempt_step[seq.seq_id] = self._step_idx
            if c is not None:
                c["preempts"] += 1
                c["preempt_tokens"] += len(seq)
            return orig_preempt(seq)

        def can_allocate(seq):
            ok = orig_can(seq)
            if not ok and self._cur is not None:
                self._cur["blocked_by_kv"] = 1
            return ok

        def run(seqs, is_prefill):
            t = clock()
            out = orig_run(seqs, is_prefill)
            self._acc("dt_run", clock() - t)
            return out

        self._patch(e, "step", step)
        self._patch(s, "schedule", schedule)
        self._patch(s, "postprocess", postprocess)
        self._patch(s, "preempt", preempt)
        self._patch(bm, "can_allocate", can_allocate)
        self._patch(r, "run", run)

        if self.fine:
            import torch
            orig_rm = r.run_model

            def wrap_prepare(orig):
                def f(seqs):
                    t = clock()
                    out = orig(seqs)
                    torch.cuda.synchronize()          # 把 H2D 拷贝也算进 prepare
                    self._acc("dt_prepare", clock() - t)
                    return out
                return f

            def run_model(input_ids, positions, is_prefill):
                torch.cuda.synchronize()
                t = clock()
                out = orig_rm(input_ids, positions, is_prefill)
                t1 = clock()
                torch.cuda.synchronize()
                t2 = clock()
                if self._cur is not None:
                    self._cur["dt_model_launch"] = t1 - t     # CPU 下发 kernel 的耗时
                    self._cur["dt_model"] = t2 - t            # 到 GPU 完成的总耗时
                    self._cur["graph"] = int(not (is_prefill or r.enforce_eager or input_ids.size(0) > 512))
                return out

            self._patch(r, "prepare_prefill", wrap_prepare(r.prepare_prefill))
            self._patch(r, "prepare_decode", wrap_prepare(r.prepare_decode))
            self._patch(r, "run_model", run_model)
        return self

    # ---------------------------------------------------------------- 每步快照
    def _finish_step(self, c):
        bm, bs = self.bm, self.bm.block_size
        running = self.sched.running
        used = len(bm.used_block_ids)
        c["running"] = len(running)
        c["used"] = used
        c["free"] = len(bm.free_block_ids)
        c["total_blocks"] = len(bm.blocks)
        c["tokens_in_use"] = sum(len(x) for x in running)
        # 内部碎片：已分配 block 容量 - 实际 token（最新 token 的 KV 尚未落盘，故按 0 截断）
        c["frag_tokens"] = sum(max(0, len(x.block_table) * bs - len(x)) for x in running)
        if self.fine:
            c["free_cached_blocks"] = sum(1 for i in bm.free_block_ids if bm.blocks[i].hash != -1)
        c.setdefault("is_prefill", False)
        self.steps.append(c)

    # ---------------------------------------------------------------- 汇总
    def summary(self):
        st = self.steps
        if not st:
            return {}
        bs = self.bm.block_size
        pre = [x for x in st if x["is_prefill"]]
        dec = [x for x in st if not x["is_prefill"]]
        wall = sum(x["dt"] for x in st) or 1e-9
        dwall = sum(x["dt"] for x in dec) or 1e-9
        S = lambda k, xs: sum(x.get(k, 0.0) for x in xs)
        out = {"mode": "fine" if self.fine else "light"}

        out["steps"] = {
            "total": len(st), "prefill": len(pre), "decode": len(dec),
            "prefill_time_share": S("dt", pre) / wall,
            "sched_share": S("dt_sched", st) / wall,
            "run_share": S("dt_run", st) / wall,
            "post_share": S("dt_post", st) / wall,
        }
        d = {
            "avg_batch": _mean([x["n_seqs"] for x in dec]),
            "step_ms_p50": _pct([x["dt"] * 1e3 for x in dec], 50),
            "step_ms_p99": _pct([x["dt"] * 1e3 for x in dec], 99),
            "sched_post_share": (S("dt_sched", dec) + S("dt_post", dec)) / dwall,
        }
        if self.fine and any("dt_model" in x for x in dec):
            d["gpu_forward_share"] = S("dt_model", dec) / dwall
            d["non_forward_share"] = 1 - d["gpu_forward_share"]      # H6: CPU/调度/采样等一切非前向部分
            d["prepare_share"] = S("dt_prepare", dec) / dwall
            d["graph_step_ratio"] = _mean([x.get("graph", 0) for x in dec])
        out["decode"] = d

        tot = S("n_tokens", pre)
        out["prefill_tokens"] = {
            "total": tot, "recompute": S("re_tokens", pre),
            "recompute_share": S("re_tokens", pre) / tot if tot else 0.0,
            "cache_hit_tokens": S("cached_tokens", pre),
            "re_hit_tokens": S("re_hit_tokens", pre),
        }

        pr = self.preempts
        lens = [r["len"] for r in pr]
        sp = [r["since_prefill"] for r in pr]
        gaps = self.readmit_gaps
        out["preempt"] = {
            "count": len(pr),
            "per_1k_decode_steps": 1000.0 * len(pr) / max(1, len(dec)),
            "distinct_seqs": len(self.preempt_counts),
            "per_seq_hist": dict(sorted(Counter(self.preempt_counts.values()).items())),
            "running_empty_preempts": sum(1 for r in pr if r["running_after"] == 0),
            "wasted_tokens": sum(lens),
            "victim_len_mean": _mean(lens), "victim_len_p95": _pct(lens, 95),
            "since_prefill_le1": _mean([x <= 1 for x in sp]),
            "since_prefill_le5": _mean([x <= 5 for x in sp]),
            "readmit_gap_median": _pct(gaps, 50),
            "thrash_ratio": _mean([g <= 2 for g in gaps]),      # H4: 抢占后 ≤2 步又被重新 prefill 的比例
        }

        util = [x["used"] / x["total_blocks"] for x in dec]
        frag = [x["frag_tokens"] / (x["used"] * bs) for x in dec if x["used"]]
        tok = [x["tokens_in_use"] / (x["total_blocks"] * bs) for x in dec]
        out["kv"] = {
            "block_size": bs,
            "util_mean": _mean(util), "util_p50": _pct(util, 50), "util_p95": _pct(util, 95),
            "tokens_in_use_mean": _mean(tok),
            "frag_mean": _mean(frag), "frag_p95": _pct(frag, 95),
            "blocked_by_kv_steps": int(S("blocked_by_kv", st)),
            "stall_reasons": dict(Counter(x["stall"] for x in st if "stall" in x)),
        }

        bursts, acc = [], 0.0
        for x in st:
            if x["is_prefill"]:
                acc += x["dt"]
            elif acc > 0:
                bursts.append(acc)
                acc = 0.0
        out["prefill_stall_s"] = {"n": len(bursts), "p50": _pct(bursts, 50),
                                  "p95": _pct(bursts, 95), "max": max(bursts) if bursts else float("nan")}
        return out

    def dump(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for name, rows in (("steps.jsonl", self.steps), ("preempts.jsonl", self.preempts)):
            with open(os.path.join(out_dir, name), "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        s = self.summary()
        with open(os.path.join(out_dir, "profiler_summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        return s


def install(engine, prof_mode=None):
    """按开关安装 profiler；关闭时返回 None，不做任何修改。"""
    m = normalize_mode(prof_mode) if prof_mode is not None else mode()
    if m == "off":
        return None
    return Profiler(engine, fine=(m == "fine")).install()