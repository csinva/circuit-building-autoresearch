"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sentiment lexicon — substrings indicative of positive/negative sentiment.
# We detect these as 5-char suffix windows ending at the current position.
# Each entry is (substring, score: +1 positive / -1 negative).
# Substrings shorter than 5 chars are padded with don't-care '?' on the left,
# but here we ONLY use exact-5-character substrings; word stems chosen so the
# trailing characters are common (helps catch inflected forms).
# ---------------------------------------------------------------------------

LEXICON: list[tuple[str, int]] = [
    # =============== POSITIVE 5-grams ===============
    # Each pattern is 5 chars long; word-stems chosen to cover common inflections.
    # Leading space anchors to the start of a word (avoids substring false-positives).
    (" good", +1), (" love", +1), (" best", +1), (" warm", +1),
    (" fun ", +1), (" hilr", +1), (" nice", +1), (" cool", +1),
    (" fine", +1),
    ("great", +1), ("beaut", +1), ("wonde", +1), ("brill", +1),
    ("excel", +1), ("fanta", +1), ("amazi", +1), ("perfe", +1),
    ("enjoy", +1), ("charm", +1), ("touch", +1), ("engag", +1),
    ("inspi", +1), ("delig", +1), ("sweet", +1), ("fresh", +1),
    ("hilar", +1), ("smart", +1), ("happy", +1), ("super", +1),
    ("witty", +1), ("clev",  +1),  # "?clev" -> matches "clever", "cleve"
    ("lovel", +1), ("lovin", +1), ("loved", +1), ("loves", +1),
    ("joy",   +1),  # "??joy" -> "_joy_", " joy ", "njoyx"; mostly captures "joy"
    ("vivid", +1), ("magic", +1), ("solid", +1), ("rich ", +1),
    ("noble", +1), ("merit", +1), ("shine", +1), ("thril", +1),
    ("moves", +1), ("moved", +1), ("genui", +1),  # genuine
    ("powe",  +1),  # ?powe -> 'powerful'
    ("rivet", +1),  # riveting
    ("captv", -1),  # (will not appear in normal text; dropped silently? keep)
    ("capti", +1),  # captivating
    ("graceful"[:5], +1),  # "grace"
    ("tende", +1),  # tender
    ("praise"[:5], +1),  # "prais"
    ("triumph"[:5], +1),  # "triu"... 4 chars; pad
    ("daring"[:5], +1),  # "darin"
    ("astonish"[:5], +1),  # "aston"
    ("intell"[:5], +1),  # "intel" -> intelligent
    ("memor", +1),  # memorable
    ("compel"[:5], +1),  # compelling
    ("entert"[:5], +1),  # entertaining
    ("refresh"[:5], +1),  # refreshing
    ("incred"[:5], +1),  # incredible
    ("master"[:5], +1),
    ("succe", +1),  # success / succeeds
    ("origin"[:5], +1),  # original
    # =============== NEGATIVE 5-grams ===============
    (" bad ", -1), (" bad,", -1), (" dull", -1), (" lame", -1), (" weak", -1),
    (" awfu", -1), (" hate", -1), (" mess", -1), (" not ", -1),
    (" ugly", -1), (" poor", -1), (" tire", -1), (" sill", -1),
    (" sad ", -1), (" fail", -1), (" flat", -1), (" lack", -1),
    (" so b", -1),   # "so bad", "so boring" etc.
    (" too ", -1),   # too long, too slow, etc — noisy but somewhat neg
    (" drag", -1),   # drags on
    (" yawn", -1),
    (" snor", -1),   # snore / snooze
    (" junk", -1),
    (" crap", -1),
    (" dumb", -1),
    (" sloW"[:5], -1),  # placeholder, see " slow" below
    (" slow", -1),
    (" worn", -1),
    (" anno", -1),
    ("borin", -1), ("stupi", -1), ("terri", -1), ("worst", -1),
    ("horri", -1), ("tedio", -1), ("point", -1),  # pointless
    ("annoy", -1), ("unfun", -1), ("bland", -1),
    ("waste", -1), ("ridic", -1), ("empty", -1), ("drear", -1),
    ("trite", -1), ("clich", -1), ("inept", -1), ("badly", -1),
    ("predi", -1),  # predictable
    ("medio", -1),  # mediocre
    ("disap", -1),  # disappointing
    ("hates", -1), ("hated", -1),
    ("loath", -1),
    ("fault", -1), ("vapid", -1),
    ("shame", -1), ("wreck", -1),
    ("weari", -1),
    ("stale", -1), ("messy", -1), ("phony", -1),
    ("flops", -1), ("creep", -1),
    ("dumb",  -1),
    ("idiot", -1), ("noisy", -1),
    ("dread", -1),
    ("hollo", -1),
    ("clums", -1),
    ("inane", -1),
    ("witl",  -1),
    ("forge", -1),  # forgettable -> "forge"
    ("formu", -1),  # formulaic
    ("uneve", -1),  # uneven
    ("painf", -1),  # painful
    ("offen", -1),  # offensive
    ("nothi", -1),
    ("never", -1),
    (" no ",  -1),
    ("frust", -1),
    ("haph",  -1),
    ("endle", -1),
    ("overb", -1),
    ("lumb",  -1),
    ("heavy", -1),  # heavy-handed
    ("clott", -1),
    ("scruff"[:5], -1),
    ("hideo", -1),
    ("ponde", -1),  # ponderous
    ("turgi", -1),  # turgid
    ("grim ", -1),
    ("monot", -1),  # monotonous
    ("plodd", -1),
    ("garis", -1),
    ("gross", -1),
    ("stupo", -1),
    ("absur", -1),  # absurd (could be neutral; risky)
    ("nausa", -1),  # nauseating
    ("repul", -1),  # repulsive
    ("smug ", -1),
    ("mawki", -1),  # mawkish
    ("turn ", -1),  # "stomach turning" — risky
    ("stinks"[:5], -1),
    ("desp",  -1),  # despicable / desperate (risky)
    ("hatef", -1),
    ("incoh", -1),  # incoherent
    ("scrap"[:5], -1),  # scrappy
    ("blund", -1),  # blunder
    ("garba", -1),  # garbage
    ("trash", -1),
    ("schlo", -1),  # schlock
    (" no s", -1),  # "no soul", "no story"
    (" can", -1),  # "can't" / "cannot" - risky
]


def _make_pattern(s: str) -> str:
    """Pad an n-gram pattern to length 5; left-pad with '?' (don't-care)."""
    if len(s) > 5:
        return ""  # skip overlong
    return ("?" * (5 - len(s))) + s


def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-built sentiment classifier.

    Architecture (override defaults in build_model below):
      - d_model = D = 336
      - n_heads = 4 in layer 1 (offsets 1..4): builds a 5-char window
      - n_layers = 2
      - Residual stream layout at each position i:
          dims  [0..80]    : pos one-hot at i        (81 dims)
          dims  [81..112]  : token one-hot at i      (32 dims)
          dims  [113..144] : token one-hot at i-1    (filled by layer 1 attn)
          dims  [145..176] : token one-hot at i-2
          dims  [177..208] : token one-hot at i-3
          dims  [209..240] : token one-hot at i-4
          dim   [241]      : per-position sentiment score (filled by layer 1 MLP)
          dim   [242]      : aggregated mean score      (filled by layer 2 attn)
      - Head: dim 242 -> logit for '1' (positive), -dim 242 -> '0'.
    """
    torch.manual_seed(0)
    D = model.d_model
    H = model.n_heads
    dh = D // H
    V = model.vocab_size
    L = model.max_seq_len
    assert D == 336 and H == 4 and dh == 84 and model.n_layers == 2

    # --- channel layout constants ---
    POS_BASE = 0          # 0..80
    TOK_BASE = 81         # 81..112  (token[i])
    SHIFT_BASE = [113, 145, 177, 209]  # head h fills slot for token[i-(h+1)]
    SCORE_DIM = 241       # sentiment score per position (set by layer 1 MLP)
    AGG_DIM = 242         # aggregated mean (set by layer 2 attention)

    with torch.no_grad():
        # Zero everything first.
        for p in model.parameters():
            p.zero_()
        # LayerNorm gains -> 1 (default behaviour), biases -> 0.
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
                m.bias.zero_()

        # ============== embeddings ==============
        # Token embedding: row c is one-hot at TOK_BASE + c.
        for c in range(V):
            model.token_emb.weight[c, TOK_BASE + c] = 1.0
        # Position embedding: row p is one-hot at POS_BASE + p.
        for p in range(L):
            model.pos_emb.weight[p, POS_BASE + p] = 1.0

        # ============== Layer 1: shift-by-(h+1) attention ==============
        block1 = model.blocks[0]
        # LN1 is left as identity (gain=1, bias=0).

        # W_q for head h: output dim (h*dh + k) selects input dim (k + h+1)
        #     so Q[i][h, k] = LN(x)[k + h+1] ~ 'one-hot at position i shifted by -(h+1)'
        # i.e., Q[i][h, k] is high (~+) when k+h+1 == i.
        for h in range(H):
            shift = h + 1
            for k in range(L):
                c = k + shift
                if 0 <= c < L:
                    block1.attn.W_q.weight[h * dh + k, POS_BASE + c] = 1.0

        # W_k for head h: identity on position dims -> K[j][h, k] is high when k == j.
        for h in range(H):
            for k in range(L):
                block1.attn.W_k.weight[h * dh + k, POS_BASE + k] = 1.0

        # W_v for head h: extract token one-hot dims into head's dims 0..V-1.
        for h in range(H):
            for v in range(V):
                block1.attn.W_v.weight[h * dh + v, TOK_BASE + v] = 1.0

        # Estimate post-LN1 active magnitude:
        #   2 active dims (pos & token) at value 1, D=336 -> mean=2/D, std=sqrt(2/D-(2/D)^2).
        mean1 = 2.0 / D
        var1 = 2.0 * (1 - mean1) ** 2 / D + (D - 2) * (mean1 ** 2) / D
        std1 = math.sqrt(var1)
        active1 = (1.0 - mean1) / std1  # ≈ 12.96 for D=336

        # W_o: head h's dim v (0..V-1) -> residual dim SHIFT_BASE[h] + v.
        # Scale so the residual contribution is ~1 when matching, ~0 otherwise.
        scale_o1 = 1.0 / active1
        for h in range(H):
            for v in range(V):
                block1.attn.W_o.weight[SHIFT_BASE[h] + v, h * dh + v] = scale_o1

        # ============== Layer 1 MLP: n-gram match -> sentiment score ==============
        # Each hidden unit detects one 5-char pattern. Output goes to SCORE_DIM.
        # Pattern offsets relative to current position i:
        #   pos 0 (oldest, i-4) -> SHIFT_BASE[3]   (32 dims for token one-hot)
        #   pos 1 (i-3)         -> SHIFT_BASE[2]
        #   pos 2 (i-2)         -> SHIFT_BASE[1]
        #   pos 3 (i-1)         -> SHIFT_BASE[0]
        #   pos 4 (i, current)  -> TOK_BASE

        # Compute post-LN2 active magnitude. After layer 1 attn, residual has 6 active
        # dims (pos[i], token[i], + 4 shifted token slots) at value ~1, rest ~0.
        mean2 = 6.0 / D
        var2 = 6.0 * (1 - mean2) ** 2 / D + (D - 6) * (mean2 ** 2) / D
        std2 = math.sqrt(var2)
        active2 = (1.0 - mean2) / std2  # value of a "matched" dim post-LN2
        inactive2 = (0.0 - mean2) / std2  # value of an inactive dim post-LN2

        # Compile patterns; drop invalid ones (chars not in vocab, or len>5).
        compiled: list[tuple[list[int], int]] = []
        for raw_s, score in LEXICON:
            if len(raw_s) > 5:
                continue
            s = _make_pattern(raw_s)
            token_ids: list[int] = []
            ok = True
            for ch in s:
                if ch == "?":
                    token_ids.append(-1)  # don't-care
                elif ch in task.stoi:
                    token_ids.append(task.stoi[ch])
                else:
                    ok = False
                    break
            if ok:
                compiled.append((token_ids, score))

        d_ff = block1.mlp.fc1.weight.shape[0]
        # We may need more hidden units than the default d_ff allows; we set it later.
        assert d_ff >= len(compiled), (
            f"d_ff={d_ff} too small for {len(compiled)} lexicon entries"
        )

        # Magnitude boost on score channel: write large per-match scores so the
        # signal survives LayerNorm's mean-subtraction in layer 2 / final.
        SCORE_MAGNITUDE = 15.0

        offset_bases = [SHIFT_BASE[3], SHIFT_BASE[2], SHIFT_BASE[1], SHIFT_BASE[0], TOK_BASE]
        for unit_idx, (tids, score) in enumerate(compiled):
            n_required = sum(1 for t in tids if t >= 0)
            for pos_off, t in enumerate(tids):
                if t < 0:
                    continue
                in_dim = offset_bases[pos_off] + t
                block1.mlp.fc1.weight[unit_idx, in_dim] = 1.0
            full = n_required * active2
            one_off = (n_required - 1) * active2 + inactive2
            threshold = 0.5 * (full + one_off)
            block1.mlp.fc1.bias[unit_idx] = -threshold
            margin = 0.5 * (active2 - inactive2)
            block1.mlp.fc2.weight[SCORE_DIM, unit_idx] = (SCORE_MAGNITUDE * score) / margin

        # ============== Layer 2: uniform attention averages SCORE_DIM ==============
        block2 = model.blocks[1]
        # Q, K = 0 (already zeroed) -> uniform attention.
        # W_v: extract SCORE_DIM (post-LN) into head 0, dim 0.
        block2.attn.W_v.weight[0, SCORE_DIM] = 1.0
        # W_o: head 0, dim 0 -> residual AGG_DIM.
        block2.attn.W_o.weight[AGG_DIM, 0] = 1.0
        # MLP of block2 stays zero (identity update).

        # ============== Head: AGG_DIM -> logits for '1' vs '0' ==============
        one_idx = task.stoi["1"]
        zero_idx = task.stoi["0"]
        model.head.weight[one_idx, AGG_DIM] = 10.0
        model.head.weight[zero_idx, AGG_DIM] = -10.0
        eq_tok_dim = TOK_BASE + task.stoi["="]
        model.head.weight[one_idx, eq_tok_dim] = 1.5
    return


model_shorthand_name = "LexiconNgram5_v4"
model_description = (
    "Larger lexicon (~200 patterns, expanded negative coverage) and reduced "
    "majority-class head bias (1.5 instead of 4) to combat over-prediction of '1'."
)


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def upsert_overall_results(rows: list[dict], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["task"]) for r in rows}
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_shorthand_name"), row.get("task")) not in new_keys:
                    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows])
    print(f"Overall results saved → {path}")



def build_model(task) -> SimpleTransformer:
    max_seq_len = max(task.seq_len, 16)
    if task.name == "sentiment-sst2":
        model = SimpleTransformer(
            vocab_size=task.vocab_size,
            max_seq_len=max_seq_len,
            d_model=336,
            n_heads=4,
            n_layers=2,
            d_ff=256,  # >= len(LEXICON) hidden units
        )
    else:
        model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task name (see src/task.py)", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task":        args.task,
        "accuracy":    f"{accuracy:.4f}",
        "status":      "",
        "model_shorthand_name":  model_shorthand_name,
        "n_params":    f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
