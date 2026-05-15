"""
Tasks for evaluating transformer models on character-level sequence problems.

A Task defines:
  - a vocabulary (list of single-character tokens)
  - a prompt length (number of input tokens)
  - an answer length (number of tokens to generate)
  - a generator that yields (prompt_str, answer_str) examples
  - encode/decode helpers between strings and integer token IDs

Add a new task by subclassing `Task` and registering it in `TASK_REGISTRY`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Example:
    prompt: str   # text shown to the model (e.g. "12345+67890=")
    answer: str   # tokens the model must generate (e.g. "080235")


class Task:
    name: str
    vocab: list[str]
    prompt_len: int
    answer_len: int

    def __init__(self):
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def seq_len(self) -> int:
        return self.prompt_len + self.answer_len

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def generate_examples(self, n: int, seed: int = 0) -> list[Example]:
        raise NotImplementedError

    def is_correct(self, prediction: str, target: str) -> bool:
        return prediction == target


class FiveDigitAdditionTask(Task):
    """Add two 5-digit non-negative integers.

    prompt: "AAAAA+BBBBB="    (12 tokens; leading zeros allowed)
    answer: "SSSSSS"          (6 tokens; left-padded with zeros)

    Example: "12345+67890=" -> "080235"  (since 12345+67890 = 80235)
    """

    name = "add5"
    vocab = list("0123456789+=")
    prompt_len = 12   # 5 + 1 + 5 + 1
    answer_len = 6    # max sum is 199998 -> 6 digits

    def generate_examples(self, n: int, seed: int = 0) -> list[Example]:
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            a = rng.randint(0, 99999)
            b = rng.randint(0, 99999)
            prompt = f"{a:05d}+{b:05d}="
            answer = f"{a + b:06d}"
            out.append(Example(prompt=prompt, answer=answer))
        return out


TASK_REGISTRY: dict[str, type[Task]] = {
    FiveDigitAdditionTask.name: FiveDigitAdditionTask,
}


def get_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()
