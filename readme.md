# circuits-evolve

Autonomous AI research on hand-writing transformer weights to solve character-level tasks.

The idea: give an AI agent a transformer architecture and a task (5-digit addition by default). The agent edits `interpretable_transformer.py`, writing weights directly into the model — no training, no gradients. It runs evaluation, checks accuracy, keeps or discards, and repeats. You wake up in the morning to a log of hand-built circuits and (hopefully) one that actually adds.

Each run lives in its own folder under `runs/evolve-<tag>/` — no git branch, no commits. The runtime files in the run folder are symlinks back to `evolve/`, except `interpretable_transformer.py` and `results/` which are local copies the agent owns.

## How it works

The pieces that matter live in `evolve/`:

- **`evolve/interpretable_transformer.py`** — defines `SimpleTransformer` and a `write_weights(model, task)` function. The agent edits this file freely, writing weight tensors directly. The default `write_weights` is a no-op (random init), so running the file as-is gives the random-init baseline. **This is the only file the agent edits.**
- **`evolve/setup_run.py`** — creates a fresh `runs/evolve-<tag>/` folder from this template (symlinks read-only files, copies `interpretable_transformer.py`).
- **`evolve/src/task.py`** — task interface and the default 5-digit addition task. Add new tasks here.
- **`evolve/src/eval.py`** — autoregressive greedy decoding + accuracy.
- **`evolve/program.md`** — instructions for the agent. Point your agent here and let it go.

## The task

The default task is **5-digit addition** (`add5`):

- prompt: `"12345+67890="` (12 chars, leading zeros allowed)
- answer: `"080235"` (6 chars, left-padded with zeros — max sum is 199998)
- vocab: `0123456789+=` (12 tokens)

The model is queried autoregressively: 12-token prompt in, 6 tokens out, greedy argmax decoding. Accuracy is exact-match over all 6 output tokens.

Other tasks can be added by subclassing `Task` in `src/task.py` and registering them in `TASK_REGISTRY`. Run with `--task <name>`.

## Metric

A single metric is tracked in `results/overall_results.csv`:

- **`accuracy`** — fraction of held-out examples where the generated answer exactly matches the target (higher is better)

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (from repo root)
uv sync

# 3. Create a fresh run folder (from repo root)
uv run evolve/setup_run.py may15-run1
cd runs/evolve-may15-run1

# 4. Run an experiment. With write_weights unmodified, this is the random-init baseline.
uv run interpretable_transformer.py
```

## Running the agent

Spin up Claude Code (or any LLM agent) in this repo and prompt:

```
Read and follow the instructions in evolve/program.md.
```

## Project structure

```
readme.md                         — this file (repo-level overview)
pyproject.toml                    — dependencies, used by uv

evolve/                           — source / template folder (read-only during a run)
  setup_run.py                    — creates a new run folder from this template
  interpretable_transformer.py    — architecture + write_weights template
  program.md                      — agent instructions
  src/
    task.py                       — task interface + 5-digit addition
    eval.py                       — autoregressive evaluation

runs/evolve-<tag>/                — one folder per experimental run
  program.md, src/                — symlinks to evolve/
  interpretable_transformer.py    — local copy (agent edits this)
  results/
    overall_results.csv           — accuracy per (model_name, task)
  interpretable_transformers_lib/
    success/, failure/            — snapshots of each attempt
```

## Design choices

- **Single file to modify.** The agent only touches `interpretable_transformer.py`. Diffs are small and reviewable.
- **No training.** The agent writes weights as closed-form constants. This forces it to actually think about a circuit instead of letting SGD do the work.
- **No git, no branches.** Each run is an isolated folder under `runs/`. The shared, read-only pieces are symlinked back to `evolve/`, so a run's diff is just `interpretable_transformer.py` + `results/` + the snapshot library.
- **Fixed eval set.** `task.generate_examples(n, seed)` is deterministic, so accuracy is directly comparable across attempts.
- **Pluggable tasks.** The same harness should work for arbitrary string-in-string-out tasks by adding a new `Task` subclass.
