"""Parallel, resumable sweep orchestrator.

Reads configs.jsonl (one JSON object per line: {"name": str, "env": {KEY: val}}).
Runs up to K configs concurrently, each as a subprocess `uv run sweep_eval.py`
with the config's env vars set. Captures the RESULT line and appends
(name, test_corr, train_corr, secs) to sweep_results.csv. Skips configs already
present in sweep_results.csv (resumable). Exits printing ALL DONE when finished.
"""
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(HERE, "configs.jsonl")
RESULTS = os.path.join(HERE, "sweep_results.csv")
K = int(os.environ.get("SWEEP_K", "5"))
J_THREADS = os.environ.get("SWEEP_J_THREADS", "16")

_lock = threading.Lock()
_rr = 0


def done_names():
    names = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                parts = line.strip().split(",")
                if parts:
                    names.add(parts[0])
    return names


def run_one(cfg):
    name = cfg["name"]
    env = dict(os.environ)
    env["SWEEP_NAME"] = name
    for k, v in cfg.get("env", {}).items():
        env[k] = str(v)
    env.setdefault("OMP_NUM_THREADS", J_THREADS)
    env.setdefault("MKL_NUM_THREADS", J_THREADS)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Round-robin GPU assignment across SWEEP_GPUS (e.g. "2,3").
    gpus = [g for g in os.environ.get("SWEEP_GPUS", "3").split(",") if g != ""]
    if gpus:
        with _lock:
            global _rr
            g = gpus[_rr % len(gpus)]
            _rr += 1
        env["CUDA_VISIBLE_DEVICES"] = g
    try:
        out = subprocess.run(
            ["uv", "run", "sweep_eval.py"],
            cwd=HERE, env=env, capture_output=True, text=True, timeout=1200)
        line = ""
        for ln in out.stdout.splitlines():
            if ln.startswith("RESULT\t"):
                line = ln
        if line:
            _, nm, tc, tr, dt = line.split("\t")
            row = f"{nm},{tc},{tr},{dt}\n"
        else:
            tail = (out.stdout[-300:] + out.stderr[-300:]).replace("\n", " ")
            row = f"{name},CRASH,CRASH,0  # {tail}\n"
    except subprocess.TimeoutExpired:
        row = f"{name},TIMEOUT,TIMEOUT,0\n"
    with _lock:
        with open(RESULTS, "a") as f:
            f.write(row)
        print(row.strip(), flush=True)


def main():
    with open(CONFIGS) as f:
        configs = [json.loads(l) for l in f if l.strip()]
    have = done_names()
    todo = [c for c in configs if c["name"] not in have]
    print(f"{len(configs)} configs, {len(todo)} to run, K={K}", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=K) as ex:
        list(ex.map(run_one, todo))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
