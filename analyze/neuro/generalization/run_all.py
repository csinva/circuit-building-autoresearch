"""Run every selected model under each setting and append rows to results.csv.

Settings per model:
  orig        UTS03, original test stories          (re-measured baseline, x-axis)
  newsubj     UTS01 + UTS02, original test stories   (new-subject transfer)
  newstory    UTS03, held-out training-pool stories  (new-story transfer)

Idempotent/resumable: rows already present in results.csv (keyed by
model+setting) are skipped, so the script can be re-run after an interruption.

Run from this folder, choosing a GPU:
    CUDA_VISIBLE_DEVICES=1 uv run run_all.py
"""
import argparse
import csv
import os
import sys
import time
import traceback

import config as C
from harness import load_embedder, run_one

OUT = os.path.join(os.path.dirname(__file__), "results.csv")
COLS = ["run", "model", "setting", "subject", "test_stories", "test_corr",
        "train_corr", "test_median", "frac_above_0.2", "n_voxels", "n_feat",
        "reported_orig", "note", "seconds", "status"]


def load_done(out=OUT):
    done = set()
    if os.path.exists(out):
        with open(out, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["model"], row["setting"], row["subject"]))
    return done


def append_row(row, out=OUT):
    exists = os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})
        f.flush()


def jobs_for(run, model, reported, note):
    """(setting, subject, train_stories, test_stories) tuples for one model."""
    return [
        ("orig",     C.ORIG_SUBJECT, C.ORIG_TRAIN, C.ORIG_TEST),
        ("newsubj",  "UTS01",        C.ORIG_TRAIN, C.ORIG_TEST),
        ("newsubj",  "UTS02",        C.ORIG_TRAIN, C.ORIG_TEST),
        ("newstory", C.ORIG_SUBJECT, C.ORIG_TRAIN, C.NEW_STORIES),
    ]


def prewarm():
    """Populate the per-story response cache single-threaded so parallel shards
    never race on the big subject-response files. Safe to re-run."""
    from src import data
    for subj, stories in [
        ("UTS01", C.ORIG_TRAIN + C.ORIG_TEST),
        ("UTS02", C.ORIG_TRAIN + C.ORIG_TEST),
        ("UTS03", C.ORIG_TRAIN + C.ORIG_TEST + C.NEW_STORIES),
    ]:
        t0 = time.time()
        data.load_responses(stories, subject=subj)
        print(f"prewarmed {subj}: {len(stories)} stories ({time.time()-t0:.0f}s)", flush=True)
    print("PREWARM DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prewarm", action="store_true", help="just cache responses, then exit")
    ap.add_argument("--out", default=OUT, help="output csv (per-shard file)")
    ap.add_argument("--models", default="", help="comma-separated indices into SELECTED (default all)")
    args = ap.parse_args()

    if args.prewarm:
        prewarm()
        return 0

    idxs = (range(len(C.SELECTED)) if not args.models
            else [int(x) for x in args.models.split(",")])
    selected = [C.SELECTED[i] for i in idxs]
    done = load_done(args.out)
    total = sum(len(jobs_for(*s)) for s in selected)
    print(f"shard models={list(idxs)} → {args.out}; {total} jobs; {len(done)} done", flush=True)
    i = 0
    for run, model, reported, note in selected:
        mf = C.model_file(run, model)
        embedder = None
        for setting, subject, train_st, test_st in jobs_for(run, model, reported, note):
            i += 1
            key = (model, setting, subject)
            if key in done:
                print(f"[{i}/{total}] skip {model} {setting} {subject} (done)", flush=True)
                continue
            tag = f"[{i}/{total}] {model} | {setting} | {subject}"
            try:
                if embedder is None:  # load once per model, reuse across settings
                    embedder, name, desc = load_embedder(mf)
                r = run_one(embedder, subject, train_st, test_st)
                append_row({
                    "run": run, "model": model, "setting": setting, "subject": subject,
                    "test_stories": "|".join(test_st),
                    "test_corr": f"{r['test_corr']:.4f}", "train_corr": f"{r['train_corr']:.4f}",
                    "test_median": f"{r['test_median']:.4f}",
                    "frac_above_0.2": f"{r['frac_above_0.2']:.4f}",
                    "n_voxels": r["n_voxels"], "n_feat": r["n_feat"],
                    "reported_orig": reported, "note": note,
                    "seconds": f"{r['seconds']:.0f}", "status": "ok",
                }, args.out)
                print(f"{tag}  test_corr={r['test_corr']:.4f}  ({r['seconds']:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001 — record + continue so one bad model can't halt the sweep
                append_row({"run": run, "model": model, "setting": setting,
                            "subject": subject, "test_stories": "|".join(test_st),
                            "reported_orig": reported, "note": note,
                            "status": f"ERROR: {e}"}, args.out)
                print(f"{tag}  ERROR: {e}", flush=True)
                traceback.print_exc()
        embedder = None  # free GPU memory between models
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
