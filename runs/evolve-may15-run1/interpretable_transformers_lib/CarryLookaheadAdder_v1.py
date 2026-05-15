"""
Interpretable transformer for character-level sequence tasks.

Hand-built 3-layer transformer that performs 5-digit addition by:
  - Block 0 attn: gather all 10 input digits (5 from A, 5 from B) into
    per-position-fixed slots via one head per digit position.
  - Block 0 MLP: for each digit pair k, compute one-hot low_k = (a_k+b_k)%10,
    plus scalar gen_k = [a_k+b_k>=10] and prop_k = [a_k+b_k==9].
  - Block 1 MLP: compute the 5 carries (carry-in to digit positions 0..3, and
    the top carry into the 10^5 place) by enumerating carry-lookahead terms
    AND(gen_m, prop_{k+1}, ..., prop_{m-1}).
  - Block 2 MLP: assemble (low_k + carry_in_k) % 10 for each output digit and
    use the position one-hot to gate which answer digit gets routed into the
    final "output slot" (dims 220..229).
  - Head: project the output slot to vocab logits.

LayerNorms are replaced with Identity so all computation is done with plain
linear + ReLU arithmetic on sparse one-hot signals.

Usage:
    uv run interpretable_transformer.py
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
# Architecture: LayerNorms replaced with Identity to keep activations linear-
# and-sparse so we can hand-set integer weights without LN renormalization.
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
        # Identity LNs so our designed sparse one-hot activations survive.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, sized to host the hand-built circuit."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 256,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 1024,
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
# Hand-built weights — carry-lookahead 5-digit adder
# ---------------------------------------------------------------------------
#
# Residual stream layout (d_model = 256):
#   [0   .. 49 ] A digit slots: slot k=0..4 at dims 10k..10k+9 (one-hot of A[k])
#   [50  .. 99 ] B digit slots: slot k=0..4 at dims 50+10k..50+10k+9 (one-hot B[k])
#   [100 .. 149] low_k one-hots: dims 100+10k..100+10k+9 = one-hot (a_k+b_k)%10
#   [150 .. 154] gen_k scalar (1 iff a_k+b_k >= 10) at dim 150+k
#   [155 .. 159] prop_k scalar (1 iff a_k+b_k == 9) at dim 155+k
#   [160 .. 163] carry-in to digit-pair k=0..3 at dim 160+k
#   [164       ] top_carry (carry into the 10^5 place) at dim 164
#   [165 .. 176] token-id one-hot (12 vocab tokens) at dim 165+t
#   [177 .. 194] position one-hot (18 positions) at dim 177+p
#   [220 .. 229] output slot — one-hot of the answer digit to emit at this pos.
#
# Digit-pair index conventions:
#   k=0 is the most-significant digit pair (10^4 place).
#   k=4 is the least-significant digit pair (10^0 place).
#   carry-in to pair k (k=0..3) comes from pairs k+1..4.
#   answer[s=0] = top_carry, answer[s=1..5] = (low_{s-1} + cin_{s-1}) % 10.


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate model parameters in closed form (no training)."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        D = model.d_model
        H = model.n_heads
        dh = D // H

        # --- token/pos embeddings: one-hot encoding ---
        for t in range(task.vocab_size):
            model.token_emb.weight[t, 165 + t] = 1.0
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, 177 + p] = 1.0

        ALPHA = 10.0  # softmax sharpness scaler

        # ============================================================
        # Block 0 — attention: gather A[k] and B[k] into the residual.
        # ============================================================
        blk0 = model.blocks[0]
        for k in range(10):
            if k < 5:
                src_pos = k           # A[0..4] at input positions 0..4
                dst_start = 10 * k    # → dims 0..49
            else:
                src_pos = 6 + (k - 5) # B[0..4] at input positions 6..10
                dst_start = 50 + 10 * (k - 5)  # → dims 50..99

            # Constant query (ALPHA in head k's first dim), realised by reading
            # the always-active token one-hot (dims 165..176 — exactly one is 1).
            for tok_dim in range(165, 177):
                blk0.attn.W_q.weight[k * dh + 0, tok_dim] = ALPHA
            # Key: ALPHA only at src_pos via that position's one-hot.
            blk0.attn.W_k.weight[k * dh + 0, 177 + src_pos] = ALPHA
            # Value: copy the digit-token one-hot (token ids 0..9) into the
            # first 10 dims of head k's value.
            for d in range(10):
                blk0.attn.W_v.weight[k * dh + d, 165 + d] = 1.0
            # Output: route head k's first 10 dims into the slot.
            for d in range(10):
                blk0.attn.W_o.weight[dst_start + d, k * dh + d] = 1.0

        # ============================================================
        # Block 0 — MLP: per-digit-pair lookup of low_k, gen_k, prop_k.
        # Hidden unit u_{k,a,b} fires iff A[k]=a AND B[k]=b.
        # ============================================================
        fc1, fc2 = blk0.mlp.fc1, blk0.mlp.fc2
        unit_idx = 0
        for k in range(5):
            for a in range(10):
                for b in range(10):
                    fc1.weight[unit_idx, 10 * k + a] = 1.0
                    fc1.weight[unit_idx, 50 + 10 * k + b] = 1.0
                    fc1.bias[unit_idx] = -1.0
                    fc2.weight[100 + 10 * k + ((a + b) % 10), unit_idx] = 1.0
                    if a + b >= 10:
                        fc2.weight[150 + k, unit_idx] = 1.0
                    if a + b == 9:
                        fc2.weight[155 + k, unit_idx] = 1.0
                    unit_idx += 1
        # 500 units used, fits in d_ff=1024.

        # ============================================================
        # Block 1 — attention is pass-through (all weights are zero, so
        # softmax(0)·0 = 0 → residual unchanged).
        # ============================================================

        # ============================================================
        # Block 1 — MLP: carry-lookahead.
        #   cin_k = sum_{m=k+1..4} AND(gen_m, prop_{k+1}, ..., prop_{m-1}),  k=0..3
        #   top   = sum_{m=0..4}   AND(gen_m, prop_0, ..., prop_{m-1})
        # ============================================================
        fc1, fc2 = model.blocks[1].mlp.fc1, model.blocks[1].mlp.fc2
        unit_idx = 0
        carry_targets = []
        for k in range(4):
            terms = []
            for m in range(k + 1, 5):
                feats = [150 + m] + [155 + j for j in range(k + 1, m)]
                terms.append(feats)
            carry_targets.append((160 + k, terms))
        # top_carry
        top_terms = []
        for m in range(5):
            feats = [150 + m] + [155 + j for j in range(0, m)]
            top_terms.append(feats)
        carry_targets.append((164, top_terms))

        for out_dim, terms in carry_targets:
            for feats in terms:
                for fd in feats:
                    fc1.weight[unit_idx, fd] = 1.0
                fc1.bias[unit_idx] = -(len(feats) - 1)
                fc2.weight[out_dim, unit_idx] = 1.0
                unit_idx += 1
        # 15 units used.

        # ============================================================
        # Block 2 — attention is pass-through.
        # ============================================================

        # ============================================================
        # Block 2 — MLP: select the right answer digit into the output slot
        # based on the current position.
        # ============================================================
        fc1, fc2 = model.blocks[2].mlp.fc1, model.blocks[2].mlp.fc2
        unit_idx = 0

        # s = 0 (p = 11): answer[0] = top_carry (0 or 1)
        #   (p=11 AND top_carry=0) → answer-slot dim 220
        fc1.weight[unit_idx, 177 + 11] = 1.0
        fc1.weight[unit_idx, 164] = -1.0
        fc1.bias[unit_idx] = 0.0
        fc2.weight[220 + 0, unit_idx] = 1.0
        unit_idx += 1
        #   (p=11 AND top_carry=1) → answer-slot dim 221
        fc1.weight[unit_idx, 177 + 11] = 1.0
        fc1.weight[unit_idx, 164] = 1.0
        fc1.bias[unit_idx] = -1.0
        fc2.weight[220 + 1, unit_idx] = 1.0
        unit_idx += 1

        # s = 1..4 (p = 12..15): answer[s] = (low_{s-1} + cin_{s-1}) % 10
        for s in range(1, 5):
            k = s - 1
            cin_dim = 160 + k
            p_target_dim = 177 + (11 + s)
            for d in range(10):
                # case A: low_k=d AND cin_k=0 → answer = d
                #   ReLU(p_target + low_k_d - cin_k - 1)
                fc1.weight[unit_idx, p_target_dim] = 1.0
                fc1.weight[unit_idx, 100 + 10 * k + d] = 1.0
                fc1.weight[unit_idx, cin_dim] = -1.0
                fc1.bias[unit_idx] = -1.0
                fc2.weight[220 + d, unit_idx] = 1.0
                unit_idx += 1
                # case B: low_k=(d-1)%10 AND cin_k=1 → answer = d
                #   ReLU(p_target + low_k_{d-1} + cin_k - 2)
                fc1.weight[unit_idx, p_target_dim] = 1.0
                fc1.weight[unit_idx, 100 + 10 * k + ((d - 1) % 10)] = 1.0
                fc1.weight[unit_idx, cin_dim] = 1.0
                fc1.bias[unit_idx] = -2.0
                fc2.weight[220 + d, unit_idx] = 1.0
                unit_idx += 1

        # s = 5 (p = 16): answer[5] = low_4
        for d in range(10):
            fc1.weight[unit_idx, 177 + 16] = 1.0
            fc1.weight[unit_idx, 100 + 10 * 4 + d] = 1.0
            fc1.bias[unit_idx] = -1.0
            fc2.weight[220 + d, unit_idx] = 1.0
            unit_idx += 1

        # --- head: read the output slot ---
        for d in range(10):
            model.head.weight[d, 220 + d] = 100.0


model_shorthand_name = "CarryLookaheadAdder_v1"
model_description = (
    "Hand-built 3-layer adder: attn gathers digit one-hots into per-pair slots, "
    "block 0 MLP computes (low,gen,prop) per pair, block 1 MLP runs the carry-"
    "lookahead chain, block 2 MLP selects the answer digit by position into "
    "an output slot read by the head."
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
