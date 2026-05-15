# autoresearch — interpretable transformers

This is an experiment to have a coding agent autonomously research how to hand-write the weights of a small transformer (ideally with some human-understandable structure) so that it solves a task. The default task is 5-digit addition: prompt `"12345+67890="` → answer `"080235"`.

The model is NEVER trained. The agent must write the weights directly into `interpretable_transformer.py`.

## Setup

The work happens inside a fresh **run folder**. Each run is an isolated copy of `evolve/` under `<repo_root>/runs/evolve-<tag>/`.

1. **Pick a run tag** based on today's date and a counter, e.g. `may15-run1`.
2. **Create the run folder**:

    ```bash
    uv run evolve/setup_run.py <tag>
    ```

    This creates `runs/evolve-<tag>/` containing:
      - `program.md`, `src/` — **symlinks** back to `evolve/` (do not modify these via the symlinks)
      - `interpretable_transformer.py` — **local copy**, this is the file you edit
      - `results/` — empty, populated by your runs
      - `interpretable_transformers_lib/{success,failure}/` — empty, for snapshots

    The repo-level `readme.md` (one level above `evolve/`) describes the overall project; you can read it but should not modify it.

3. **`cd` into the run folder** and stay there for the rest of the session:

    ```bash
    cd runs/evolve-<tag>
    ```

4. **Read the in-scope files**: `../../readme.md` (repo-level overview), `interpretable_transformer.py`, `src/task.py`, `src/eval.py`, and `results/overall_results.csv` (empty at first).

## Experimentation

Run an experiment (from inside the run folder):

```bash
uv run interpretable_transformer.py
```

This builds the `SimpleTransformer`, calls `write_weights(model, task)` to populate its parameters, runs autoregressive evaluation on the task, and updates `results/overall_results.csv`.

**What you CAN do:**

- Edit `interpretable_transformer.py` in the run folder. Everything in it is fair game (except the evaluation code in the **main** block at the bottom):
  - The `SimpleTransformer`, `Block`, `CausalSelfAttention`, `MLP` architectures.
  - The `write_weights(model, task)` function — set any parameter tensors directly. Hardcoded constants, NumPy arrays, hand-built attention circuits, lookup tables — anything goes, as long as you do not train.
  - The architecture's hyperparameters (depth, width, heads, ff size).
  - The `model_shorthand_name` and `model_description` strings.
- Save snapshots of `interpretable_transformer.py` under `interpretable_transformers_lib/{success,failure}/` after each attempt.

**What you CANNOT do:**

- Run any kind of training, gradient descent, optimizer step, fitting loop, or backprop. The model parameters must be set in closed form.
- Use python functions / existing libraries to compute the task output. The model must generate the answer autoregressively using a transformer architecture with no external tools.
- Modify anything reached through a symlink: anything in `src/` or `program.md`.
- Create git branches or commits. The run folder is your workspace; persistence is via files only.
- Install new packages — only what's in the repo's `pyproject.toml`.

## Goal

Maximize **`accuracy`** on the task: the fraction of held-out examples where the autoregressively-generated answer string exactly matches the target. With the default 5-digit addition task, an untouched `write_weights` (random init) scores ~0%, and a perfect circuit scores 100%.

## Output format

Once the script finishes it prints a summary like this:

```
---
task:          add5
accuracy:      0.0050  (1/200)
total_seconds: 1.3s
```

It also updates `results/overall_results.csv` with a row keyed by `(model_name, task)`.

## Logging results

The CSV has a header row and 5 columns:

```
task,accuracy,status,model_name,description
```

1. task name (e.g. `add5`)
2. accuracy from the script output — empty for crashes
3. status: `keep`, `discard`, `crash`, or `baseline` (use `baseline` for the first random-init row)
4. shorthand name of the model attempt — must be unique within the run folder
5. brief text description of what this attempt tried

## The experiment loop

You are always inside the run folder (`runs/evolve-<tag>/`).

LOOP FOREVER:

1. Edit `interpretable_transformer.py` with one experimental idea. Update `model_shorthand_name` (must be unique within this run) and `model_description` to reflect it.
2. Run the experiment: `uv run interpretable_transformer.py > run.log 2>&1`
3. Read results: `tail -n 10 run.log` and `grep <shorthand_name> results/overall_results.csv`
4. If the run crashed, check `tail -n 50 run.log` for the stack trace and attempt a fix.
5. Update the row in `results/overall_results.csv` with the appropriate status (`keep`, `discard`, `crash`).
6. Snapshot `interpretable_transformer.py` as a new file:
   - `interpretable_transformers_lib/success/transformer_<simple_name>.py` if accuracy improved over the best prior attempt
   - `interpretable_transformers_lib/failure/transformer_<simple_name>.py` otherwise

**NEVER STOP**: once the loop has begun, do NOT pause to ask the human if you should continue. Run until manually stopped.

**Ideas to try** (not exhaustive — be creative):

- Read mech-interp papers on grokking / modular arithmetic / addition circuits and translate the proposed circuits into hand-built weights:
  - Nanda et al. 2023 "Progress measures for grokking via mechanistic interpretability" (<https://arxiv.org/abs/2301.05217>)
  - Quirke & Barez 2023 "Understanding addition in transformers" (<https://arxiv.org/abs/2310.13121>)
- Implement digit-by-digit addition with a head per digit position.
- Use position embeddings to "select" the right digit pair to add at each output step.
- Carry propagation via a second attention layer.
- Use the MLP layers as lookup tables (one-hot in, one-hot out).
- Reverse-engineer the output digits (least significant first) to make carry easier.
- Do not just generate I/O pairs and memorize them — the goal is to build a circuit that generalizes across all 5-digit pairs.

Make sure the model evaluates very quickly. Default eval is 200 examples on GPU. BE CREATIVE.
