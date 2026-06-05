# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build a self-contained interactive HTML report from the runs-neuro results.

Each row of every run's overall_results.csv is one iteration (a hand-written
transformer tried by the coding agent). The metric is `test_corr` on subject
UTS03; GPT-2 XL rows are pretrained baselines. Model / thinking-effort per run
is read from each run folder's metadata.json (recovered from the copilot CLI
logs the first time, then persisted there).
"""
import csv
import glob
import html
import json
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS_DIR = os.path.join(REPO, "runs-neuro")
OUT = os.path.join(os.path.dirname(__file__), "report.html")

# ---------------------------------------------------------------------------
# Flagging.  The premise is that the agent hand-writes the weights of a small
# interpretable transformer — it may NOT use any model trained on data, INCLUDING
# text pretraining.  Both kinds below are explicitly disallowed and are excluded
# from each run's reported best / running-best curve:
#   "train"      — explicitly trained/back-propped on the fMRI data.
#   "pretrained" — loads an external pretrained neural encoder (Qwen/BERT/GloVe/…),
#                  i.e. a model trained on data via text pretraining.
# (The fixed GPT-2 XL baseline is also text-pretrained, but it is the reference
#  point being compared against, not a hand-written entry, so it is not flagged.)
# Detection is keyword-based, then validated against the actual descriptions.
_PRE_RE = re.compile(
    r"qwen|distilbert|deberta|roberta|\bbert\b|glove|word2vec|fasttext|spacy|"
    r"en_core_web|sentence.?transf|llama|mistral|minilm|mpnet|sbert", re.I)
_TRAIN_RE = re.compile(r"backprop|end.?to.?end.?train|epochs?\b", re.I)
# Guards against false positives: "untrained", "FASTTEXT-STYLE" hand-built bags,
# "inspired by", explicitly hand-wired/hand-built circuits.
_NEG_TRAIN_RE = re.compile(r"untrained|no training|not trained|never trained", re.I)
_NEG_PRE_RE = re.compile(r"style|inspired|hand-?built|hand-?wired|hand-?coded", re.I)


def classify(name, desc):
    """Return None | 'train' | 'pretrained' for one iteration."""
    blob = f"{name} | {desc}"
    if _TRAIN_RE.search(blob) and not _NEG_TRAIN_RE.search(blob):
        return "train"
    if _PRE_RE.search(blob) and not _NEG_PRE_RE.search(blob):
        return "pretrained"
    return None

# Discover runs from their metadata.json files (written once, then reused).
RUN_META = {}
for mpath in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "metadata.json"))):
    folder = os.path.basename(os.path.dirname(mpath))
    with open(mpath) as f:
        RUN_META[folder] = json.load(f)

# may27 (untrimmed) is shown separately; everything else is directly comparable.
UNTRIMMED = next(f for f, m in RUN_META.items() if m.get("untrimmed"))
# trimmed runs ordered by their label (chronological by construction)
TRIMMED_ORDER = sorted(
    (f for f, m in RUN_META.items() if not m.get("untrimmed")),
    key=lambda f: RUN_META[f]["label"],
)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_run(folder):
    path = os.path.join(RUNS_DIR, folder, "results", "overall_results.csv")
    rows = []
    baseline = None
    it = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            corr = fnum(r["test_corr"])
            if corr is None:
                continue
            name = r.get("model_shorthand_name", "")
            desc = (r.get("description") or "").strip()
            is_base = "gpt" in name.lower()
            rec = {
                "name": name,
                "corr": corr,
                "train": fnum(r.get("train_corr")),
                "params": fnum(r.get("n_params")),
                "desc": desc,
                "base": is_base,
                "flag": None if is_base else classify(name, desc),
            }
            if is_base:
                if baseline is None:
                    baseline = corr
                rec["it"] = None
            else:
                it += 1
                rec["it"] = it
            rows.append(rec)
    # Running max over LEGITIMATE (non-flagged) non-baseline points, in order.
    # Flagged iterations carry the line forward but never raise it.
    best = float("-inf")
    for rec in rows:
        if rec["base"]:
            rec["rmax"] = None
            continue
        if rec["flag"] is None and rec["corr"] > best:
            best = rec["corr"]
        rec["rmax"] = best if best != float("-inf") else None
    nb = [r for r in rows if not r["base"]]
    legit = [r for r in nb if r["flag"] is None]
    flagged = [r for r in nb if r["flag"] is not None]
    best_legit = max(legit, key=lambda r: r["corr"]) if legit else None
    best_any = max(nb, key=lambda r: r["corr"]) if nb else None
    meta = RUN_META[folder]
    return {
        "folder": folder,
        "label": meta["label"],
        "model": meta["model"],
        "effort": meta["effort"],
        "baseline": baseline,
        "n_iter": len(nb),
        "n_train_flag": sum(1 for r in flagged if r["flag"] == "train"),
        "n_pre_flag": sum(1 for r in flagged if r["flag"] == "pretrained"),
        # "best" is the best LEGITIMATE (hand-wired, untrained) iteration.
        "best": best_legit["corr"] if best_legit else None,
        "best_name": best_legit["name"] if best_legit else None,
        # best including flagged iterations, reported separately as a caveat.
        "best_any": best_any["corr"] if best_any else None,
        "best_any_name": best_any["name"] if best_any else None,
        "best_any_flag": best_any["flag"] if best_any else None,
        "rows": rows,
    }


DATA = {f: load_run(f) for f in RUN_META}

# ---------------------------------------------------------------------------
# Narrative writeup, grounded in the descriptions / results above.
# ---------------------------------------------------------------------------
WRITEUP = {
    "fmri-jun3-run1": {
        "headline": "Hand-wired interpretable transformers with lexical feature tokens — the strongest trimmed run.",
        "worked": "Tokenizing each word into interpretable feature tokens (function-word type, ~40 hand-curated "
                  "semantic categories, perceptual modality, concreteness, animacy, valence/arousal, "
                  "person-reference, morphology) and pooling them with multi-scale recency-weighted attention. "
                  "The <code>LexFeat*</code> family steadily climbed to <b>0.076</b>, with small fixes "
                  "(pronouns, past-tense morphology, spatial prepositions, first-word anchoring, well-formed "
                  "categories) each nudging it up.",
        "failed": "Pure bag-of-character and raw semantic-category circuits (<code>RecencyBoC</code> 0.032, "
                  "<code>SemCatBoC</code> 0.034) were far behind — character-level information alone carries "
                  "little fMRI-relevant signal. Adding hand-curated lexical/semantic structure was what mattered.",
    },
    "fmri-jun03-run2": {
        "headline": "GPT-5.5 brute-forced a large lexicon search — many iterations, modest ceiling.",
        "worked": "A 'semantic best-lexicon' approach that greedily adds/drops individual high-value content words "
                  "(love, old, little, time, face, …) with a tail/context window. With 600+ iterations it reached "
                  "<b>0.063</b> using only ~7k parameters.",
        "failed": "Char-only and multi-decay structural variants (<code>char_only_rec90</code> 0.028, "
                  "<code>structure_multidecay</code> 0.030) underperformed. The search spent enormous effort on "
                  "tiny per-word lexicon tweaks with diminishing returns, never closing the gap to the baseline.",
    },
    "fmri-jun03-run3": {
        "headline": "Gemini 3.1 Pro stalled at a 0.042 hand-wired ceiling, then went off-premise — swapping in "
                    "pretrained LLM encoders and even back-propping on data.",
        "worked": "Within the hand-wired premise, multi-timescale ensembles ('UltraTune' optimal decay scales, "
                  "staggered splits, final-LN shifts) in the <code>Deep_Ensemble_Final_LN_*</code> family and "
                  "phonetic-class character circuits plateaued at <b>0.042</b> — the weakest legitimate ceiling of "
                  "any run.",
        "failed": "Exact mechanistic tricks (math-trick character extraction, a hard-coded 4000-word lexicon matched "
                  "filter, spatial-hash smoothing) were the worst performers — token-level precision did not "
                  "translate into fMRI predictivity. Unable to break 0.042 by hand, the run then <b>abandoned the "
                  "premise</b>: 27 iterations loaded external pretrained encoders (Qwen2.5-0.5B/1.5B/3B, DistilBERT, "
                  "RoBERTa, DeBERTa, GloVe, Word2Vec, spaCy), and <code>Hybrid_Qwen1.5B_L28Mean_L14Last</code> "
                  "reached <b>0.092</b> — but those models are <b>trained on data via text pretraining</b>, which is "
                  "<b>explicitly disallowed</b>, so they are flagged and excluded. One iteration "
                  "(<code>End_to_End_Trained_10Epochs</code>) went further and back-propped through the entire ridge "
                  "pipeline — <b>training directly on the fMRI data, also disallowed</b>.",
    },
    "fmri-jun03-run4": {
        "headline": "Claude Opus 4.7 converged fast (within ~10 iterations) on a compact feature-bag transformer, "
                    "then spent 360+ more iterations expanding lexicons for tiny gains.",
        "worked": "The <code>FeatBag</code> family — a single hand-wired attention layer pooling interpretable "
                  "feature tokens over the 10-gram at four position time-scales (λ = −2, 0, 4, 16: primacy + global "
                  "mean + two recency heads), with negation flipping valence — crossed <b>0.070</b> by iteration 10 "
                  "and crept to <b>0.077</b>. Restoring extra semantic categories (motor, names, places) and adding "
                  "heads helped most; the long tail of <code>FeatBag_v3xx_*Bonus*</code> lexicon expansions added "
                  "only ~0.007 over 360+ iterations.",
        "failed": "Stripping the semantic features back (<code>FeatBag_v9_LambdaSweep</code> 0.041, "
                  "<code>FeatBag_v2_WordID</code> 0.055) or randomizing the MLP hurt. The win came from richer "
                  "hand-curated semantics, not from architectural search.",
    },
    "fmri-jun04-run1": {
        "headline": "Claude Opus 4.8 (xhigh) resumed run 4's FeatBag circuit and probed morphology / frequency / "
                    "length features — a short run that confirmed the ~0.075 ceiling.",
        "worked": "Starting from a reproduced <code>FeatBag</code> interpretable feature-bag transformer "
                  "(<code>S1_FeatBag_Repro</code> at <b>0.075</b>), the strongest tweaks added speech-act verb "
                  "inflections and possession morphology — but none beat the starting point, so the run tops out at "
                  "<b>0.075</b>, in line with run 4's FeatBag family.",
        "failed": "Stripping to content words only (<code>E24_ContentOnly</code> 0.061) hurt most; graded "
                  "word-length + finer frequency buckets (<code>S2_LenFreqBuckets</code> 0.072) and morphological "
                  "backoff for out-of-lexicon words (<code>S3_MorphBackoff</code> 0.072) did not help. In 21 "
                  "iterations the run could not push past the hand-wired FeatBag ceiling.",
    },
    "fmri-may27-run1": {
        "headline": "Claude Opus 4.7 (untrimmed) — the only run to beat GPT-2 XL, but on an easier (untrimmed) metric.",
        "worked": "Feature-engineered linguistic circuits with no transformer at all: WordNet-derived semantic "
                  "categories + morphology + perceptual-modality lexicons (vision/audition/touch/taste/smell/motor), "
                  "pooled with multi-timescale exponential windows and discourse-position / within-story novelty "
                  "signals. <code>WordNetMorphLingPerceptual</code> hit <b>0.115</b>, beating the (untrimmed) GPT-2 XL "
                  "baseline of 0.079 by ~46% relative.",
        "failed": "Plain hashed bag-of-words (<code>HashedBoW</code> 0.018–0.027) and subword-bigram bags were weak. "
                  "Structured, hand-curated linguistic features dominated raw n-gram hashing.",
    },
}

# Per-run informational banners (non-flag context worth surfacing on the card).
INFO_NOTE = {
    "fmri-jun04-run1": (
        "This run was <b>prompted to read the results of all the runs before it</b> when starting, "
        "so it began with strictly more information than any other run — it resumed from run 4's "
        "<code>FeatBag</code> circuit rather than exploring from scratch. Its ceiling should be read "
        "in that light: it is a confirmation / refinement of earlier runs, not an independent search."),
}

# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def run_card(run, idx):
    """Render one run as a card: header + plot div + collapsible methods table."""
    w = WRITEUP[run["folder"]]
    base = run["baseline"]
    best = run["best"]
    delta = (best - base) if (best is not None and base is not None) else None
    beats = delta is not None and delta > 0
    delta_str = (f"{'+' if delta >= 0 else ''}{delta:.4f}" if delta is not None else "—")
    badge_cls = "good" if beats else "below"

    # Per-run flag note (only shown when the run has flagged iterations).
    flag_note = ""
    nt, npre = run["n_train_flag"], run["n_pre_flag"]
    if nt or npre:
        parts = []
        if nt:
            parts.append(
                f'<b>{nt}</b> iteration{"s" if nt != 1 else ""} <b>trained on data</b> '
                f'(back-prop / fitting weights on the fMRI signal) — <b>explicitly disallowed</b>')
        if npre:
            parts.append(
                f'<b>{npre}</b> iteration{"s" if npre != 1 else ""} loaded an <b>external pretrained '
                f'encoder</b> (Qwen / DistilBERT / RoBERTa / GloVe / …) — a model <b>trained on data via text '
                f'pretraining</b>, also <b>explicitly disallowed</b>')
        cav = ""
        if run["best_any_flag"] is not None and run["best_any"] is not None:
            kind = ("trained directly on the fMRI data" if run["best_any_flag"] == "train"
                    else "a text-pretrained encoder")
            cav = (f' Its single highest score, <b>{run["best_any"]:.4f}</b> '
                   f'(<code>{html.escape(run["best_any_name"])}</code>), comes from {kind} '
                   f'and is <b>excluded</b> from the run\'s reported best '
                   f'(<b>{best:.4f}</b>, the top hand-wired model).')
        flag_note = (
            '<div class="flag-note"><span class="flag-ic">⚠</span><div>'
            'Flagged iterations: ' + '; '.join(parts) + '.' + cav + '</div></div>')

    info_note = ""
    if run["folder"] in INFO_NOTE:
        info_note = ('<div class="info-note"><span class="info-ic">ℹ</span><div>'
                     + INFO_NOTE[run["folder"]] + '</div></div>')

    methods = [r for r in run["rows"] if not r["base"]]
    methods_sorted = sorted(methods, key=lambda r: r["corr"], reverse=True)
    rows_html = []
    for rank, m in enumerate(methods_sorted, 1):
        above = base is not None and m["corr"] >= base and m["flag"] is None
        cls = "above" if above else ""
        if m["flag"] == "train":
            cls = "flag-train"
            tag = '<span class="ftag t">⚠ trained on data</span>'
        elif m["flag"] == "pretrained":
            cls = "flag-pre"
            tag = '<span class="ftag p">⚠ pretrained encoder (text pretraining)</span>'
        else:
            tag = ""
        params = ("%.2g" % m["params"]) if m["params"] is not None else "—"
        rows_html.append(
            f'<tr class="{cls}"><td class="rank">{rank}</td>'
            f'<td class="mono">{html.escape(m["name"])}{tag}</td>'
            f'<td class="num">{m["corr"]:.4f}</td>'
            f'<td class="num dim">{params}</td>'
            f'<td class="desc">{html.escape(m["desc"])}</td></tr>'
        )
    table = (
        '<table class="methods"><thead><tr>'
        '<th>#</th><th>model</th><th>test_corr</th><th>params</th><th>description</th>'
        '</tr></thead><tbody>' + "".join(rows_html) + '</tbody></table>'
    )

    return f'''
<div class="card">
  <div class="card-head">
    <div class="titles">
      <h3>{html.escape(run["label"])}</h3>
      <div class="meta">
        <span class="chip model">{html.escape(run["model"])}</span>
        <span class="chip effort">effort: {html.escape(run["effort"])}</span>
        <span class="chip">{run["n_iter"]} iterations</span>
      </div>
    </div>
    <div class="stats">
      <div class="stat"><span class="k">best hand-wired</span><span class="v">{best:.4f}</span></div>
      <div class="stat"><span class="k">baseline</span><span class="v">{base:.4f}</span></div>
      <div class="stat"><span class="k">Δ vs GPT-2 XL</span><span class="v badge {badge_cls}">{delta_str}</span></div>
    </div>
  </div>
  {flag_note}
  {info_note}
  <div class="plot" id="plot-{run['folder']}"></div>
  <div class="write">
    <p class="headline">{w['headline']}</p>
    <p><b class="ok">What worked:</b> {w['worked']}</p>
    <p><b class="no">What didn't:</b> {w['failed']}</p>
  </div>
  <details class="methods-wrap">
    <summary>Show all {run['n_iter']} methods tried (sorted by test_corr; green = hand-wired at/above baseline; red = trained on the fMRI data; orange = text-pretrained encoder — both disallowed &amp; excluded)</summary>
    {table}
  </details>
</div>'''


cards_trimmed = "\n".join(run_card(DATA[f], i) for i, f in enumerate(TRIMMED_ORDER))
card_untrimmed = run_card(DATA[UNTRIMMED], 0)

# JSON payload for Plotly (only the fields the front-end needs).
def js_payload(run):
    nb = [r for r in run["rows"] if not r["base"]]
    return {
        "label": run["label"],
        "model": run["model"],
        "effort": run["effort"],
        "baseline": run["baseline"],
        "it": [r["it"] for r in nb],
        "corr": [r["corr"] for r in nb],
        "rmax": [r["rmax"] for r in nb],
        "name": [r["name"] for r in nb],
        "flag": [r["flag"] for r in nb],
    }

PAYLOAD = {f: js_payload(DATA[f]) for f in RUN_META}
TRIMMED_JS = json.dumps(TRIMMED_ORDER)
UNTRIMMED_JS = json.dumps(UNTRIMMED)
PAYLOAD_JS = json.dumps(PAYLOAD)

HTML = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fMRI encoding autoresearch — run report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {{
    --ink:#1a1a1a; --dim:#6b7280; --line:#e5e7eb; --bg:#ffffff; --panel:#fafafa;
    --good:#15803d; --goodbg:#dcfce7; --below:#b45309; --belowbg:#fef3c7;
    --accent:#2563eb;
  }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased; line-height:1.55; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:48px 28px 96px; }}
  h1 {{ font-size:30px; font-weight:700; letter-spacing:-0.02em; margin:0 0 6px; }}
  h2 {{ font-size:21px; font-weight:650; letter-spacing:-0.01em; margin:56px 0 6px;
        padding-bottom:8px; border-bottom:1px solid var(--line); }}
  h3 {{ font-size:17px; font-weight:650; margin:0; }}
  p {{ margin:10px 0; }}
  a {{ color:var(--accent); text-decoration:none; }}
  .lead {{ color:var(--dim); font-size:15px; max-width:760px; }}
  code {{ background:var(--panel); border:1px solid var(--line); border-radius:5px;
    padding:1px 5px; font-size:12.5px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .note {{ background:var(--belowbg); border:1px solid #fcd34d; border-radius:10px;
    padding:14px 18px; font-size:14px; margin:16px 0 0; color:#78350f; }}
  .card {{ border:1px solid var(--line); border-radius:14px; padding:22px 22px 8px;
    margin:22px 0; background:var(--bg); }}
  .card-head {{ display:flex; justify-content:space-between; align-items:flex-start;
    gap:20px; flex-wrap:wrap; }}
  .titles h3 {{ margin-bottom:8px; }}
  .meta {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .chip {{ font-size:12px; color:var(--dim); background:var(--panel);
    border:1px solid var(--line); border-radius:999px; padding:3px 10px; white-space:nowrap; }}
  .chip.model {{ color:var(--ink); font-weight:600; }}
  .chip.effort {{ color:var(--accent); }}
  .stats {{ display:flex; gap:22px; }}
  .stat {{ display:flex; flex-direction:column; align-items:flex-end; }}
  .stat .k {{ font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:0.04em; }}
  .stat .v {{ font-size:18px; font-weight:680; font-variant-numeric:tabular-nums; }}
  .badge {{ border-radius:6px; padding:1px 8px; font-size:15px; }}
  .badge.good {{ color:var(--good); background:var(--goodbg); }}
  .badge.below {{ color:var(--below); background:var(--belowbg); }}
  .plot {{ width:100%; height:340px; margin:14px 0 4px; }}
  .write {{ font-size:14.5px; }}
  .write .headline {{ font-weight:620; }}
  b.ok {{ color:var(--good); }} b.no {{ color:var(--below); }}
  details.methods-wrap {{ margin:8px 0 14px; }}
  details.methods-wrap > summary {{ cursor:pointer; font-size:13.5px; color:var(--accent);
    padding:8px 0; user-select:none; list-style:none; }}
  details.methods-wrap > summary::-webkit-details-marker {{ display:none; }}
  details.methods-wrap > summary::before {{ content:"▸ "; }}
  details.methods-wrap[open] > summary::before {{ content:"▾ "; }}
  .methods {{ width:100%; border-collapse:collapse; font-size:12.5px;
    margin-top:8px; display:block; max-height:460px; overflow:auto;
    border:1px solid var(--line); border-radius:10px; }}
  .methods thead th {{ position:sticky; top:0; background:#fff; text-align:left;
    padding:9px 10px; border-bottom:2px solid var(--line); color:var(--dim);
    font-weight:600; z-index:1; }}
  .methods td {{ padding:7px 10px; border-bottom:1px solid #f1f1f1; vertical-align:top; }}
  .methods tr.above td {{ background:#f6fdf8; }}
  .methods tr.flag-train td {{ background:#fef2f2; }}
  .methods tr.flag-pre td {{ background:#fff7ed; }}
  .ftag {{ display:inline-block; margin-left:6px; font-size:10px; font-weight:600;
    border-radius:5px; padding:1px 6px; vertical-align:middle; white-space:nowrap;
    font-family:-apple-system,Segoe UI,Roboto,sans-serif; }}
  .ftag.t {{ color:#b91c1c; background:#fee2e2; border:1px solid #fecaca; }}
  .ftag.p {{ color:#c2410c; background:#ffedd5; border:1px solid #fed7aa; }}
  .flag-note {{ display:flex; gap:10px; align-items:flex-start;
    background:#fef2f2; border:1px solid #fecaca; border-radius:10px;
    padding:11px 14px; font-size:13px; color:#7f1d1d; margin:4px 0 6px; }}
  .flag-note .flag-ic {{ font-size:15px; line-height:1.3; }}
  .flag-note code {{ background:#fff; border-color:#fecaca; }}
  .info-note {{ display:flex; gap:10px; align-items:flex-start;
    background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px;
    padding:11px 14px; font-size:13px; color:#1e3a8a; margin:4px 0 6px; }}
  .info-note .info-ic {{ font-size:15px; line-height:1.3; }}
  .info-note code {{ background:#fff; border-color:#bfdbfe; }}
  .methods .rank {{ color:var(--dim); width:34px; }}
  .methods .num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; width:70px; }}
  .methods .num.dim {{ color:var(--dim); }}
  .methods .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    font-size:11.5px; max-width:230px; word-break:break-all; }}
  .methods .desc {{ color:#374151; min-width:280px; }}
  .summary-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:18px; }}
  @media (max-width:860px) {{ .summary-grid {{ grid-template-columns:1fr; }} }}
  .summary-box {{ border:1px solid var(--line); border-radius:14px; padding:18px; }}
  .summary-title {{ font-size:14px; font-weight:650; margin-bottom:6px; }}
  .summary-title .dimlbl {{ font-weight:400; color:var(--dim); font-size:12px; }}
  #summary-trimmed, #summary-untrimmed {{ width:100%; height:420px; }}
  .legend-note {{ font-size:13px; color:var(--dim); margin-top:10px; }}
  footer {{ margin-top:60px; color:var(--dim); font-size:12.5px;
    border-top:1px solid var(--line); padding-top:18px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Interpretable transformers for fMRI language encoding</h1>
  <p class="lead">
    Each run is an autonomous coding-agent loop that hand-writes the weights of a small,
    interpretable transformer (no gradient training) so its embeddings predict fMRI responses
    to spoken stories. Every point below is one iteration the agent tried; the metric is the
    mean encoding <b>test correlation</b> on subject <b>UTS03</b>. The dashed line is the
    pretrained <b>GPT-2 XL</b> baseline (layer-24 final-token 10-gram embeddings). Each run is
    labeled with the LLM and thinking-effort that drove it (recovered from the copilot CLI logs).
  </p>
  <div class="note" style="background:#fef2f2;border-color:#fecaca;color:#7f1d1d">
    <b>⚠ Rule &amp; flagging.</b> The premise is that the agent <b>hand-writes</b> the weights — it may
    <b>not use any model trained on data, including text pretraining</b>. Iterations that back-prop / fit
    weights on the fMRI signal are flagged <span class="ftag t">⚠ trained on data</span>; iterations that
    load a large <b>external pretrained encoder</b> (Qwen, DistilBERT, RoBERTa, GloVe, …) are flagged
    <span class="ftag p">⚠ pretrained encoder (text pretraining)</span>. <b>Both are explicitly disallowed</b>
    and are <b>excluded from each run's reported best and from the running-best curves</b>. Only one run
    (Jun-03 run 3) contains any. (The <b>GPT-2 XL baseline</b> is itself text-pretrained — that is fine, as it
    is the fixed reference point being compared against, not a hand-written entry.)
  </div>

  <h2>Summary — running-best across all runs</h2>
  <p class="lead">Each line traces one run's <b>running-best</b> test correlation as its iterations
  accumulate (points connected per run); faint markers are the raw per-iteration scores. Trimmed and
  untrimmed runs use different evaluations and baselines, so they are shown in separate panels and are
  <b>not</b> directly comparable across panels.</p>
  <div class="summary-grid">
    <div class="summary-box">
      <div class="summary-title">Trimmed runs <span class="dimlbl">(30 TRs trimmed off each story end · directly comparable)</span></div>
      <div id="summary-trimmed"></div>
    </div>
    <div class="summary-box">
      <div class="summary-title">Untrimmed run <span class="dimlbl">(story ends NOT trimmed · separate, higher metric)</span></div>
      <div id="summary-untrimmed"></div>
    </div>
  </div>

  <h2>May 27 run — untrimmed (shown separately, not directly comparable)</h2>
  <div class="note">
    <b>⚠ Different evaluation.</b> This run did <b>not</b> trim the story ends. Every other
    run trims <b>30 TRs off each end</b> of every story before fitting/scoring. Trimming removes
    the easy-to-predict onset/offset periods, so the untrimmed numbers here (and its GPT-2 XL
    baseline of 0.079) are <b>systematically higher</b> and should not be compared head-to-head
    with the trimmed runs below.
  </div>
  {card_untrimmed}

  <h2>Trimmed runs (30 TRs trimmed off each story end)</h2>
  <p class="lead">These five runs share an identical evaluation and baseline (GPT-2 XL = 0.083),
  so their curves are directly comparable.</p>
  {cards_trimmed}

  <footer>
    Generated from <code>runs-neuro/*/results/overall_results.csv</code>. Model / effort labels
    recovered from <code>~/.copilot/logs</code> (<code>selected_model</code> /
    <code>defaultReasoningEffort</code>). Subject UTS03.
  </footer>
</div>

<script>
const PAYLOAD = {PAYLOAD_JS};
const TRIMMED = {TRIMMED_JS};
const UNTRIMMED = {UNTRIMMED_JS};
const PALETTE = {{
  "fmri-jun3-run1":"#2563eb", "fmri-jun03-run2":"#db2777",
  "fmri-jun03-run3":"#059669", "fmri-jun03-run4":"#d97706",
  "fmri-jun04-run1":"#0e7490", "fmri-may27-run1":"#7c3aed"
}};
const FONT = {{ family:"-apple-system,Segoe UI,Roboto,sans-serif", color:"#1a1a1a", size:12 }};
const BASE_LAYOUT = {{
  paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:FONT,
  margin:{{l:54,r:18,t:10,b:44}},
  xaxis:{{ title:"iteration", gridcolor:"#f0f0f0", zeroline:false, showline:true,
           linecolor:"#d1d5db" }},
  yaxis:{{ title:"mean test correlation", gridcolor:"#f0f0f0", zeroline:false,
           showline:true, linecolor:"#d1d5db" }},
  legend:{{ orientation:"h", y:-0.22, x:0, font:{{size:11}} }},
  hovermode:"closest"
}};
const CFG = {{ displayModeBar:false, responsive:true }};

function wrapText(s, n) {{
  s = s || ""; const out = []; let line = "";
  for (const word of s.split(/\\s+/)) {{
    if ((line + " " + word).trim().length > n) {{ out.push(line.trim()); line = word; }}
    else line += " " + word;
  }}
  if (line.trim()) out.push(line.trim());
  return out.join("<br>");
}}

const FLAG_LABEL = {{ "train":"⚠ trained on data (disallowed)", "pretrained":"⚠ pretrained encoder — text pretraining (disallowed)" }};
function pick(d, key, want) {{
  // indices where d.flag matches the predicate `want` (null / "train" / "pretrained")
  const xs=[], ys=[], tx=[];
  d.it.forEach((it,i) => {{
    const fl = d.flag[i];
    const keep = (want==="legit") ? (fl===null) : (fl===want);
    if (keep) {{ xs.push(it); ys.push(d.corr[i]);
      tx.push("<b>"+d.name[i]+"</b><br>iter "+it+" · test_corr "+d.corr[i].toFixed(4)
        + (fl ? "<br><i>"+FLAG_LABEL[fl]+"</i>" : "")); }}
  }});
  return {{x:xs, y:ys, text:tx}};
}}
function drawRun(folder) {{
  const d = PAYLOAD[folder];
  const color = PALETTE[folder];
  const leg = pick(d, "corr", "legit");
  const tr = pick(d, "corr", "train");
  const pr = pick(d, "corr", "pretrained");
  const traces = [
    {{ x:leg.x, y:leg.y, mode:"markers", name:"hand-wired iteration",
       marker:{{size:5, color:color}}, opacity:0.5, text:leg.text, hoverinfo:"text" }},
    {{ x:d.it, y:d.rmax, mode:"lines", name:"running best (hand-wired)",
       line:{{color:color, width:2.5, shape:"hv"}}, hoverinfo:"skip" }}
  ];
  if (pr.x.length) traces.push({{ x:pr.x, y:pr.y, mode:"markers", name:"⚠ pretrained encoder",
       marker:{{size:7, color:"#c2410c", symbol:"diamond", line:{{color:"#fff", width:1}}}},
       text:pr.text, hoverinfo:"text" }});
  if (tr.x.length) traces.push({{ x:tr.x, y:tr.y, mode:"markers", name:"⚠ trained on data",
       marker:{{size:9, color:"#b91c1c", symbol:"x", line:{{color:"#fff", width:1}}}},
       text:tr.text, hoverinfo:"text" }});
  const layout = JSON.parse(JSON.stringify(BASE_LAYOUT));
  layout.shapes = [{{ type:"line", x0:0, x1:1, xref:"paper", y0:d.baseline, y1:d.baseline,
      line:{{color:"#9ca3af", width:1.5, dash:"dash"}} }}];
  layout.annotations = [{{ x:0.012, xref:"paper", y:d.baseline, xanchor:"left", yanchor:"bottom",
      text:"GPT-2 XL baseline "+d.baseline.toFixed(3), showarrow:false,
      font:{{size:11, color:"#6b7280"}} }}];
  layout.legend = {{ orientation:"h", y:1.06, x:1, xanchor:"right", font:{{size:11}} }};
  layout.margin.t = 24;
  Plotly.newPlot("plot-"+folder, traces, layout, CFG);
}}

Object.keys(PAYLOAD).forEach(drawRun);

// ---- summary overlays (trimmed & untrimmed shown separately) ----
function summaryPlot(divId, folders, baseline, baseLabel, baseColor) {{
  const traces = [];
  folders.forEach(folder => {{
    const d = PAYLOAD[folder]; const color = PALETTE[folder];
    const nm = d.label + " · " + d.model + " (" + d.effort + ")";
    // faint raw points (legitimate hand-wired iterations only)
    const leg = pick(d, "corr", "legit");
    traces.push({{ x:leg.x, y:leg.y, mode:"markers", showlegend:false,
      marker:{{size:4, color:color, opacity:0.22}}, hoverinfo:"skip", legendgroup:folder }});
    // connected running-best line
    traces.push({{ x:d.it, y:d.rmax, mode:"lines+markers", name:nm, legendgroup:folder,
      line:{{color:color, width:2.5}}, marker:{{size:3, color:color}},
      text:d.name.map((n,i)=>"<b>"+nm+"</b><br>"+n+"<br>running best "+(d.rmax[i]!=null?d.rmax[i].toFixed(4):"—")),
      hoverinfo:"text" }});
  }});
  const layout = JSON.parse(JSON.stringify(BASE_LAYOUT));
  layout.margin.t = 16;
  // extra bottom room + lower legend so the wrapped legend never collides with
  // the "iteration" axis title (the trimmed panel has up to 5 wrapping entries).
  layout.margin.b = 132;
  layout.xaxis = Object.assign({{}}, layout.xaxis, {{ title:{{text:"iteration", standoff:8}} }});
  layout.legend = {{ orientation:"h", y:-0.30, x:0, yanchor:"top", font:{{size:11}} }};
  layout.shapes = [
    {{ type:"line", x0:0, x1:1, xref:"paper", y0:baseline, y1:baseline,
       line:{{color:baseColor, width:1.4, dash:"dash"}} }}
  ];
  layout.annotations = [
    {{ x:0.012, xref:"paper", y:baseline, xanchor:"left", yanchor:"bottom",
       text:baseLabel, showarrow:false, font:{{size:10.5, color:baseColor}} }}
  ];
  Plotly.newPlot(divId, traces, layout, CFG);
}}
// trimmed panel: shared GPT-2 XL baseline (≈0.083) from any trimmed run
summaryPlot("summary-trimmed", TRIMMED, PAYLOAD[TRIMMED[0]].baseline,
  "GPT-2 XL (trimmed) " + PAYLOAD[TRIMMED[0]].baseline.toFixed(3), "#9ca3af");
// untrimmed panel: the single May-27 run against its own (higher) baseline
summaryPlot("summary-untrimmed", [UNTRIMMED], PAYLOAD[UNTRIMMED].baseline,
  "GPT-2 XL (untrimmed) " + PAYLOAD[UNTRIMMED].baseline.toFixed(3), "#d97706");

// Re-fit plots that were hidden inside <details> when they open.
document.querySelectorAll("details.methods-wrap").forEach(el => {{
  el.addEventListener("toggle", () => {{
    window.dispatchEvent(new Event("resize"));
  }});
}});
</script>
</body>
</html>'''

with open(OUT, "w") as f:
    f.write(HTML)
print(f"wrote {OUT} ({len(HTML)} bytes)")
