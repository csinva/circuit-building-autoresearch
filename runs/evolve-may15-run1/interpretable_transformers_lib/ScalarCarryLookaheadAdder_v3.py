"""
Interpretable transformer for character-level sequence tasks.

v3 of the hand-built 5-digit adder.

Compared to v2:
  - Digits live as **scalars** in the residual stream (a_k at dim k, b_k at
    dim 5+k) instead of 10-dim one-hots.  Block 0 attention only needs a
    1-dim value per routing head, so we collapse d_model from 256 to 128.
  - Step-function ReLUs in block 0 MLP read those two scalars directly
    (weights of just (1, 1, -t)), making the per-pair computation visibly
    a plain scalar addition+threshold rather than a 20-coefficient sum.

The downstream carry-lookahead chain (block 1 MLP) and position-gated answer
selection (block 2 MLP) are unchanged in structure from v2.

LayerNorms remain Identity so the hand-set integer weights see clean
sparse inputs.
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
from src.task import get_task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "description"]


# ---------------------------------------------------------------------------
# Architecture
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
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer sized for the scalar-digit hand-built adder."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 512,
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
        self.final_ln = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Hand-built weights — scalar-digit carry-lookahead adder
# ---------------------------------------------------------------------------
#
# Residual stream layout (d_model = 128):
#   [  0 ..   4] a_k scalar  (A digit value at pair index k, k=0..4)
#   [  5 ..   9] b_k scalar  (B digit value at pair index k)
#   [ 10 ..  59] low_k one-hot (50 dims, dim 10+10k+e = [low_k == e])
#   [ 60 ..  64] gen_k  ∈ {0,1}
#   [ 65 ..  69] prop_k ∈ {0,1}
#   [ 70 ..  73] carry-in to pair k (k=0..3)
#   [ 74      ] top_carry (carry into 10^5 place)
#   [ 75 ..  92] position one-hot (18 dims, dim 75+p)
#   [ 93 .. 102] output slot — one-hot of answer digit to emit
#   [103       ] constant 1 (set by pos_emb at every position)
#   [104 .. 109] scratch
#   [110       ] scalar digit value carried by the token embedding
#   [111 .. 127] scratch


def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        H = model.n_heads
        dh = model.d_model // H

        # --- embeddings ---
        # Token embedding: digit token t -> scalar t at dim 110.
        for t in range(10):
            model.token_emb.weight[t, 110] = float(t)
        # ('+', '=') tokens stay zero — they don't get attended to as values.
        # Position embedding: one-hot at dim 75+p AND constant 1 at dim 103.
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, 75 + p] = 1.0
            model.pos_emb.weight[p, 103] = 1.0

        ALPHA = 10.0

        # ============================================================
        # Block 0 — attention: route each input digit's scalar value
        # to its dedicated slot in the residual stream.
        # ============================================================
        blk0 = model.blocks[0]
        for k in range(10):
            if k < 5:
                src_pos = k
                dst_dim = k                # A[k] -> dim k
            else:
                src_pos = 6 + (k - 5)
                dst_dim = 5 + (k - 5)      # B[k-5] -> dim 5..9

            # Constant query realised by reading dim 103 (always 1).
            blk0.attn.W_q.weight[k * dh + 0, 103] = ALPHA
            # Key = ALPHA only at src_pos via position one-hot.
            blk0.attn.W_k.weight[k * dh + 0, 75 + src_pos] = ALPHA
            # Value = scalar digit value at dim 110.
            blk0.attn.W_v.weight[k * dh + 0, 110] = 1.0
            # Output: head k's value dim 0 -> residual dst_dim.
            blk0.attn.W_o.weight[dst_dim, k * dh + 0] = 1.0

        # ============================================================
        # Block 0 — MLP: step-function ReLUs over scalar (a_k + b_k).
        #   u_t(k) = ReLU(a_k + b_k - t),  t = 0..18,  k = 0..4
        #   [a+b >= t] = u_{t-1}(k) - u_t(k)
        # ============================================================
        fc1, fc2 = blk0.mlp.fc1, blk0.mlp.fc2
        unit_idx = 0
        step_unit: dict[tuple[int, int], int] = {}
        for k in range(5):
            for t in range(19):
                fc1.weight[unit_idx, k] = 1.0            # a_k
                fc1.weight[unit_idx, 5 + k] = 1.0        # b_k
                fc1.bias[unit_idx] = -float(t)
                step_unit[(k, t)] = unit_idx
                unit_idx += 1

        def add_step(out_dim: int, k: int, t: int, coeff: float) -> None:
            """Add coeff * [a_k+b_k >= t] to residual dim `out_dim` (t >= 1)."""
            fc2.weight[out_dim, step_unit[(k, t - 1)]] += coeff
            fc2.weight[out_dim, step_unit[(k, t)]] -= coeff

        for k in range(5):
            # one-hot low_k: bit e is 1 iff (a+b) mod 10 == e.
            for e in range(10):
                out = 10 + 10 * k + e
                if e == 0:
                    fc2.bias[out] += 1.0      # synthesises "1" part
                else:
                    add_step(out, k, e, +1.0)
                add_step(out, k, e + 1, -1.0)
                if e + 10 <= 18:
                    add_step(out, k, e + 10, +1.0)
                if e + 11 <= 18:
                    add_step(out, k, e + 11, -1.0)
            # gen_k  = [a+b >= 10]
            add_step(60 + k, k, 10, +1.0)
            # prop_k = [a+b >= 9] - [a+b >= 10]
            add_step(65 + k, k, 9, +1.0)
            add_step(65 + k, k, 10, -1.0)
        # 95 hidden units used.

        # ============================================================
        # Block 1 — attention: pass-through (all weights zero).
        # ============================================================

        # ============================================================
        # Block 1 — MLP: carry-lookahead.
        # ============================================================
        fc1, fc2 = model.blocks[1].mlp.fc1, model.blocks[1].mlp.fc2
        unit_idx = 0
        carry_targets: list[tuple[int, list[list[int]]]] = []
        for k in range(4):
            terms: list[list[int]] = []
            for m in range(k + 1, 5):
                feats = [60 + m] + [65 + j for j in range(k + 1, m)]
                terms.append(feats)
            carry_targets.append((70 + k, terms))
        top_terms: list[list[int]] = []
        for m in range(5):
            feats = [60 + m] + [65 + j for j in range(0, m)]
            top_terms.append(feats)
        carry_targets.append((74, top_terms))

        for out_dim, terms in carry_targets:
            for feats in terms:
                for fd in feats:
                    fc1.weight[unit_idx, fd] = 1.0
                fc1.bias[unit_idx] = -(len(feats) - 1)
                fc2.weight[out_dim, unit_idx] = 1.0
                unit_idx += 1
        # 15 hidden units used.

        # ============================================================
        # Block 2 — attention: pass-through.
        # ============================================================

        # ============================================================
        # Block 2 — MLP: position-gated answer selection -> output slot.
        # ============================================================
        fc1, fc2 = model.blocks[2].mlp.fc1, model.blocks[2].mlp.fc2
        unit_idx = 0

        # s = 0 (p = 11): answer[0] = top_carry
        # output slot dim 0 = [top_carry == 0]  (when p==11)
        fc1.weight[unit_idx, 75 + 11] = 1.0
        fc1.weight[unit_idx, 74] = -1.0
        fc1.bias[unit_idx] = 0.0
        fc2.weight[93 + 0, unit_idx] = 1.0
        unit_idx += 1
        # output slot dim 1 = [top_carry == 1]  (when p==11)
        fc1.weight[unit_idx, 75 + 11] = 1.0
        fc1.weight[unit_idx, 74] = 1.0
        fc1.bias[unit_idx] = -1.0
        fc2.weight[93 + 1, unit_idx] = 1.0
        unit_idx += 1

        # s = 1..4: answer[s] = (low_{s-1} + cin_{s-1}) % 10
        for s in range(1, 5):
            k = s - 1
            cin_dim = 70 + k
            p_dim = 75 + (11 + s)
            for d in range(10):
                # case A: low_k = d   AND  cin_k = 0
                fc1.weight[unit_idx, p_dim] = 1.0
                fc1.weight[unit_idx, 10 + 10 * k + d] = 1.0
                fc1.weight[unit_idx, cin_dim] = -1.0
                fc1.bias[unit_idx] = -1.0
                fc2.weight[93 + d, unit_idx] = 1.0
                unit_idx += 1
                # case B: low_k = (d-1)%10  AND  cin_k = 1
                fc1.weight[unit_idx, p_dim] = 1.0
                fc1.weight[unit_idx, 10 + 10 * k + ((d - 1) % 10)] = 1.0
                fc1.weight[unit_idx, cin_dim] = 1.0
                fc1.bias[unit_idx] = -2.0
                fc2.weight[93 + d, unit_idx] = 1.0
                unit_idx += 1

        # s = 5 (p = 16): answer[5] = low_4
        for d in range(10):
            fc1.weight[unit_idx, 75 + 16] = 1.0
            fc1.weight[unit_idx, 10 + 10 * 4 + d] = 1.0
            fc1.bias[unit_idx] = -1.0
            fc2.weight[93 + d, unit_idx] = 1.0
            unit_idx += 1

        # --- head: read the output slot ---
        for d in range(10):
            model.head.weight[d, 93 + d] = 100.0


model_shorthand_name = "ScalarCarryLookaheadAdder_v3"
model_description = (
    "Scalar-digit variant: A and B digits live as single scalars in the "
    "residual stream (not one-hot), so d_model shrinks to 128 and the "
    "block-0 MLP's step ReLUs read just two dims each. Carry-lookahead and "
    "position-gated answer selection unchanged from v2."
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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="add5", help="Task name (see src/task.py).")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = get_task(args.task)
    model = build_model(task).to(args.device)

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    upsert_overall_results([{
        "task":        args.task,
        "accuracy":    f"{accuracy:.4f}",
        "status":      "",
        "model_shorthand_name":  model_shorthand_name,
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
