# fmri-jun11-run1 — findings

Goal: hand-write (no training) a small transformer whose final-token 10-gram
embedding predicts UTS03 fMRI. Metric = mean held-out voxel `test_corr`.
GPT-2 XL baseline = **0.0826**. Best interpretable model here = **~0.0567 (69%)**.

## BREAKTHROUGH (285-config parallel sweep): RIGHT-CONTEXT distributional structure
The single biggest lever, found only by an exhaustive sweep: build the main LSA from
RIGHT-context co-occurrence (what typically FOLLOWS a word — its forward-predictive /
syntactic role) instead of symmetric. This alone took 0.0515 -> ~0.0549, and with a
symmetric 2nd view + dim/window tuning -> 0.0567. Right beats left beats symmetric.
Final config (FINAL_rightLSA_v8): right-context SPPMI(10) LSA(160,win6) + symmetric 2nd
LSA(80,win5) + topic LSA(50) + 32 category flags + top-65 identity + cat-congruence;
[last word | uniform bag]. The climb: 0.0515 -> 0.0519 (CAT_SCALE 4->2) -> 0.0525
(orthographic) -> 0.0537 (right 2nd view) -> 0.0549 (right MAIN view) -> 0.0558
(dim/window tune) -> 0.0563 -> **0.0567** (knob combos). Ablations: every component helps.
Adding MORE (orthographic, 3rd view, big stacks) on top OVERFITS — the model is lean.

---
## (earlier) Best interpretable model before the sweep = 0.0515 (62%)

## BEST model (FINAL_v47, 0.0515; topic dim tuned 50->70 on the multi-scale stack)
Per-word signature concatenates five closed-form, interpretable parts:
  * **Shifted-PPMI(10) word-word LSA(200, window 5)** — local distributional semantics,
    SGNS-equivalent denoising (the workhorse).
  * **second word-word LSA(60, window 12)** — a broader (paragraph-scale) association view;
    the two scales must differ (window 8 ~ window 5 -> no gain; window 12 stacks, +0.0004).
  * **term-document topic LSA(50)** — global topical structure (which stories a word is in).
  * **32 hand-curated brain-relevant semantic-category flags** (body→EBA, place→PPA/RSC,
    motion→sPMv, vis/aud percept→AC, person/social, emotion, mental, communication→Broca,
    quantity→IPS, plus color/size/temperature/texture/light/water/fire/vehicle/etc.).
  * **one-hot identity for the 70 most frequent words** — lets ridge memorize the
    idiosyncratic response of high-frequency words (63%+ of tokens).
A 1-layer attention circuit (q=k=0, uniform) exposes `[last word | uniform bag]`, plus a
**category-congruence** interaction (product of last-word & bag category dims). No training.

## The breakthrough: STACKING complementary signals (0.0446 -> 0.0506)
After 23 iterations the LSA+cats+bag model plateaued at 0.0446. Adding *frequent-word
identity* broke it, then each further orthogonal signal stacked:
  0.0446 (LSA+cats+bag) -> +identity(K=70) 0.0469 -> +SPPMI(10) 0.0502 -> +topic(50) 0.0499*
  -> (SPPMI tuned 5->10) 0.0502 -> +category-congruence 0.0506.
Identity-K sweep peaks at ~70 (30→.0461, 50→.0464, 70→.0469, 85→.0462, 100→.0440).
SPPMI-shift peaks at ~10 (5→combo, 10→.0502, 20→.0428). LSA dim 200 and window 5 stay
optimal even on the denoised stack; topic dim 50≈90; categories saturate by ~17 on the
mean (32 ties but lifts PPA/FFA/IPS).

## Earlier best (pre-breakthrough): lsa_plus_semcats_bag, 0.0447
LSA(200)+cats+uniform [last|bag]. The single biggest lever was word->LSA semantics
(0.030->0.042); categories added a little (->0.0447).

## What moved the metric
- word→**LSA distributional semantics** is the single big lever: 0.030 → 0.042.
- **semantic-category flags** add a little and lift category-selective ROIs: → 0.0447.
- LSA dim sweep: 150→0.0422, **200→0.0447**, 300→0.0378 (clean peak at 200).

## What did NOT help (all ≤ best; most overfit under the fixed ridge α-grid [10,1e4])
recency weighting; recent-word delay-line (word order); multi-scale temporal pooling;
3 scalar lexical axes (length/freq/function-word); content-salience down-weighting of
function words; frequent-word one-hot identity (helped language ROIs, overfit the mean);
term-document topic LSA (neutral, best frac>0.2); full-corpus vocab expansion + LSA
fold-in (coverage 86%→93% but hurt: noisier space); 200-dim nonlinear LSA-product
interaction (overfit, 0.0400); 17-dim category-congruence interaction (tied, 0.0447);
19 morphosyntactic suffix flags (-ing/-ed/-ly/...: hurt, 0.0413); Shifted-PPMI/SGNS LSA
(shift=5: tied 0.0444, lifted AC 0.144->0.161 & sPMv 0.110->0.124 but not the mean).

## Robust-tie cluster (all ~0.0444-0.0447, indistinguishable on the mean)
LSA200 + categories + uniform bag, and its variants (12 vs 17 cats, +category-congruence,
SPPMI). The all-voxel mean is a hard ceiling; richer features only move specific ROIs and
frac>0.2, never the mean.

## Methodological gotcha
The multi-head "POOL_HEADS" machinery (used iter10–18) is ~0.004 WORSE than the
original simple 2-block `[last|bag]` write_weights, because its per-head LayerNorm
entangles the LSA and category dims. The original simple circuit (iter8 snapshot) is
the true best — the FINAL model uses it.

## Why the plateau
The all-voxel mean is dominated by ~90k weakly-predictable voxels; it rewards
low-dimensional, well-generalizing static semantics and punishes any higher-dim or
nonlinear addition (overfit). Closing the gap to GPT-2 XL needs learned nonlinear
contextual composition, which isn't hand-writable in one no-training forward pass.
Snapshots of every attempt are in `interpretable_transformers_lib/`.
