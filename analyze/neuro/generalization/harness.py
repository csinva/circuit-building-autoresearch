"""Generalization harness: run a historical hand-written embedder under the
fixed encoding pipeline, but with an arbitrary (subject, train_stories,
test_stories) — so we can test transfer to new subjects and new stories.

Nothing under ../../../evolve-neuro/src is modified; we import it read-only and
replicate the ~30 lines of run_encoding so we can pass explicit story lists
(src.eval.run_encoding only supports the fixed get_story_names split).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so snapshots' `from src.eval import ...` finds our symlink

from src import data, encoding, features  # noqa: E402
from src.eval import POPULAR_ROIS  # noqa: E402

NDELAYS = 4
EDGE_TRIM = 30
NGRAM = 10


# ---------------------------------------------------------------------------
# Loading a historical embedder from its snapshot .py (loaded in place, so
# may27's chain-loaded parent snapshots resolve relative to their own folder).
# ---------------------------------------------------------------------------
_MOD_CT = 0


def load_embedder(model_file: str, device: str = "cuda"):
    """Return (embedder, shorthand_name, description) for a snapshot .py file."""
    global _MOD_CT
    _MOD_CT += 1
    # Some snapshots import run-local helper modules (e.g. run3's `top_words`) that
    # live in the run root, next to the lib folder. Put both the lib dir and the run
    # root on sys.path so those imports resolve.
    lib_dir = os.path.dirname(os.path.abspath(model_file))
    run_root = os.path.dirname(lib_dir)
    for p in (run_root, lib_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    mod_name = f"_snap_{_MOD_CT}"
    spec = importlib.util.spec_from_file_location(mod_name, model_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    # construct the embedder
    if hasattr(mod, "build_embedder"):
        embedder = mod.build_embedder(device=device)
    else:
        # numpy-style snapshot (e.g. may27 WordNet*): instantiate the *Embedder class
        cls = next(
            getattr(mod, n) for n in dir(mod)
            if n.endswith("Embedder") and isinstance(getattr(mod, n), type)
        )
        embedder = cls()

    name = getattr(mod, "model_shorthand_name", None)
    desc = getattr(mod, "model_description", None)
    if name is None:  # class-based snapshot
        obj = embedder
        name = getattr(obj, "SHORTHAND_NAME", os.path.basename(model_file)[:-3])
        desc = getattr(obj, "DESCRIPTION", "")
    return embedder, name, desc


# ---------------------------------------------------------------------------
# One encoding fit/eval with explicit stories (mirrors src.eval.run_encoding).
# ---------------------------------------------------------------------------
def run_one(embedder, subject: str, train_stories: list[str], test_stories: list[str],
            compute_rois: bool = True, verbose: bool = False) -> dict:
    t0 = time.time()
    all_stories = train_stories + test_stories
    wordseqs = data.load_wordseqs(all_stories)
    resps = data.load_responses(all_stories, subject=subject)

    stim_train = features.get_features(
        wordseqs, train_stories, embedder, ngram_size=NGRAM,
        ndelays=NDELAYS, extra_trim=EDGE_TRIM)
    stim_test = features.get_features(
        wordseqs, test_stories, embedder, ngram_size=NGRAM,
        ndelays=NDELAYS, extra_trim=EDGE_TRIM)
    resps = {s: r[EDGE_TRIM:-EDGE_TRIM] for s, r in resps.items()}
    resp_train = np.vstack([resps[s] for s in train_stories])
    resp_test = np.vstack([resps[s] for s in test_stories])
    assert stim_train.shape[0] == resp_train.shape[0], "train feat/resp TR mismatch"
    assert stim_test.shape[0] == resp_test.shape[0], "test feat/resp TR mismatch"

    r = encoding.fit_encoding(stim_train, resp_train, stim_test, resp_test)

    roi_corrs = {}
    if compute_rois:
        rois = data.load_rois(subject)
        corrs_test = r["corrs_test"]
        for nm in POPULAR_ROIS:
            if nm in rois:
                idx = rois[nm]
                idx = idx[idx < len(corrs_test)]
                roi_corrs[nm] = float(np.nanmean(corrs_test[idx])) if len(idx) else float("nan")

    return {
        "subject": subject,
        "test_corr": float(r["corrs_test_mean"]),
        "train_corr": float(r["corrs_train_mean"]),
        "test_median": float(r["corrs_test_median"]),
        "frac_above_0.2": float(r["corrs_test_frac>0.2"]),
        "n_voxels": int(len(r["corrs_test"])),
        "n_feat": int(stim_train.shape[1]),
        "roi_corrs": roi_corrs,
        "seconds": time.time() - t0,
        "train_stories": train_stories,
        "test_stories": test_stories,
    }
