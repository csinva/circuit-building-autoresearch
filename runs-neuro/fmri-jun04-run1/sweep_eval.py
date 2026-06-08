"""Single-config evaluator for the parallel sweep.

Reads config knobs from environment variables (already set by the orchestrator),
builds the embedder from interpretable_transformer (whose module-level constants
read those env vars), runs the encoding pipeline, and prints one RESULT line.
Does NOT touch results/overall_results.csv (keeps the official log clean).
"""
import os
import sys
import time

import interpretable_transformer as M
from src.eval import EncodingConfig, run_encoding


def main():
    name = os.environ.get("SWEEP_NAME", "unnamed")
    subject = os.environ.get("SWEEP_SUBJECT", "UTS03")
    num_train = int(os.environ.get("SWEEP_NUM_TRAIN", "8"))
    num_test = int(os.environ.get("SWEEP_NUM_TEST", "3"))
    t0 = time.time()
    emb = M.build_embedder(device="cuda")
    cfg = EncodingConfig(subject=subject, num_train=num_train, num_test=num_test)
    r = run_encoding(emb, cfg)
    tc = r.get("test_corr")
    tr = r.get("corrs_train_mean", r.get("train_corr"))
    dt = time.time() - t0
    print(f"RESULT\t{name}\t{tc:.4f}\t{tr:.4f}\t{dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
