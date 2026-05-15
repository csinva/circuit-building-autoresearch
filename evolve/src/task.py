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

    name = "addition-five-digits"
    vocab = list("0123456789+=")
    prompt_len = 12   # 5 + 1 + 5 + 1
    answer_len = 6    # max sum is 199998 -> 6 digits

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            a = rng.randint(0, 99999)
            b = rng.randint(0, 99999)
            prompt = f"{a:05d}+{b:05d}="
            answer = f"{a + b:06d}"
            out.append(Example(prompt=prompt, answer=answer))
        return out


class FiveDigitMultiplicationTask(Task):
    """Multiply two 5-digit non-negative integers.

    prompt: "AAAAA*BBBBB="    (12 tokens; leading zeros allowed)
    answer: "PPPPPPPPPP"      (10 tokens; left-padded with zeros)

    Example: "12345*67890=" -> "0838102050"  (since 12345*67890 = 838102050)
    """

    name = "multiplication-five-digits"
    vocab = list("0123456789*=")
    prompt_len = 12   # 5 + 1 + 5 + 1
    answer_len = 10   # max product is 99999*99999 = 9999800001 -> 10 digits

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            a = rng.randint(0, 99999)
            b = rng.randint(0, 99999)
            prompt = f"{a:05d}*{b:05d}="
            answer = f"{a * b:010d}"
            out.append(Example(prompt=prompt, answer=answer))
        return out


class FiveDigitSortTask(Task):
    """Sort a string of 5 digits in ascending order.

    prompt: "DDDDD="     (6 tokens)
    answer: "SSSSS"      (5 tokens; sorted ascending)

    Example: "31415=" -> "11345"
    """

    name = "sort-five-digits"
    vocab = list("0123456789=")
    prompt_len = 6    # 5 + 1
    answer_len = 5

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            digits = [rng.randint(0, 9) for _ in range(5)]
            prompt = "".join(str(d) for d in digits) + "="
            answer = "".join(str(d) for d in sorted(digits))
            out.append(Example(prompt=prompt, answer=answer))
        return out


class CountDigitTask(Task):
    """Count occurrences of a query digit in a string of 10 digits.

    The final digit before '=' is the query.

    prompt: "DDDDDDDDDDQ="  (12 tokens: 10 digits + query digit + '=')
    answer: "CC"            (2 tokens; count 00-10, left-padded)

    Example: "31415926533=" -> "02"  (digit '3' appears twice in "3141592653")
    """

    name = "digit-counting-10"
    vocab = list("0123456789=")
    prompt_len = 12   # 10 + 1 + 1
    answer_len = 2

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            digits = [rng.randint(0, 9) for _ in range(10)]
            query = rng.randint(0, 9)
            digit_str = "".join(str(d) for d in digits)
            prompt = f"{digit_str}{query}="
            count = sum(1 for d in digits if d == query)
            answer = f"{count:02d}"
            out.append(Example(prompt=prompt, answer=answer))
        return out


class ParityTask(Task):
    """Compute parity (XOR) of a variable-length bit string, length 1-10.

    Inputs are left-padded with '_' to length 10.

    prompt: "__BBBBBBBB="  (11 tokens; left-padded bits + '=')
    answer: "P"            (1 token; '0' if even number of 1s, '1' if odd)

    Example: "____1011010=" -> "0"  (four 1s -> even parity)
    """

    name = "parity-upto10-bits"
    vocab = list("01=_")
    prompt_len = 11   # 10 + 1
    answer_len = 1

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            length = rng.randint(1, 10)
            bits = [rng.randint(0, 1) for _ in range(length)]
            bit_str = "".join(str(b) for b in bits)
            prompt = bit_str.rjust(10, "_") + "="
            answer = str(sum(bits) % 2)
            out.append(Example(prompt=prompt, answer=answer))
        return out


class BooleanCircuitTask(Task):
    """Evaluate a boolean expression of 5 bits joined by 4 binary operators.

    Operators: '&' (AND), '|' (OR), '^' (XOR). Evaluated strictly left-to-right
    (no precedence), like a chain through a circuit.

    prompt: "BoBoBoBoB="   (10 tokens; 5 bits interleaved with 4 ops + '=')
    answer: "R"            (1 token; '0' or '1')

    Example: "1&0|1^1&0=" -> "0"
      ((((1&0)|1)^1)&0) = ((1|1)^1)&0 = (1^1)&0 = 0&0 = 0
    """

    name = "boolean-circuit-5-bits"
    vocab = list("01&|^=")
    prompt_len = 10   # 5 bits + 4 ops + '='
    answer_len = 1

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        ops = ["&", "|", "^"]
        out = []
        for _ in range(n):
            bits = [rng.randint(0, 1) for _ in range(5)]
            chosen_ops = [rng.choice(ops) for _ in range(4)]
            tokens = []
            for i, b in enumerate(bits):
                tokens.append(str(b))
                if i < 4:
                    tokens.append(chosen_ops[i])
            prompt = "".join(tokens) + "="
            result = bits[0]
            for i, op in enumerate(chosen_ops):
                rhs = bits[i + 1]
                if op == "&":
                    result = result & rhs
                elif op == "|":
                    result = result | rhs
                else:
                    result = result ^ rhs
            answer = str(result)
            out.append(Example(prompt=prompt, answer=answer))
        return out


class LinearInterpolationTask(Task):
    """Linear interpolation given two integer (x, y) pairs and a query x.

    The underlying line y = m*x + b uses integer slope m in [1, 5] and integer
    intercept b in [0, 9], so all y values are guaranteed integers in [0, 54].
    x values are single digits in [0, 9]; x1 != x2; query xq in [0, 9].

    prompt: "X1,Y1;X2,Y2;XQ="   (12 tokens; y values are 2 digits)
    answer: "YY"                (2 tokens)

    Example: m=2, b=3 -> "1,05;4,11;7=" -> "17"  (y = 2*7+3 = 17)
    """

    name = "linear-interpolation-two-points"
    vocab = list("0123456789,;=")
    prompt_len = 12   # 1 + 1 + 2 + 1 + 1 + 1 + 2 + 1 + 1 + 1
    answer_len = 2

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            m = rng.randint(1, 5)
            b = rng.randint(0, 9)
            x1 = rng.randint(0, 9)
            x2 = rng.randint(0, 9)
            while x2 == x1:
                x2 = rng.randint(0, 9)
            xq = rng.randint(0, 9)
            y1 = m * x1 + b
            y2 = m * x2 + b
            yq = m * xq + b
            prompt = f"{x1},{y1:02d};{x2},{y2:02d};{xq}="
            answer = f"{yq:02d}"
            out.append(Example(prompt=prompt, answer=answer))
        return out


TASK_REGISTRY: dict[str, type[Task]] = {
    FiveDigitAdditionTask.name: FiveDigitAdditionTask,
    FiveDigitMultiplicationTask.name: FiveDigitMultiplicationTask,
    FiveDigitSortTask.name: FiveDigitSortTask,
    CountDigitTask.name: CountDigitTask,
    ParityTask.name: ParityTask,
    BooleanCircuitTask.name: BooleanCircuitTask,
    LinearInterpolationTask.name: LinearInterpolationTask,
}


def get_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()

if __name__ == '__main__':
    print(TASK_REGISTRY.keys()) # ['addition-five-digits', 'multiplication-five-digits', 'sort-five-digits', 'digit-counting-10', 'parity-upto10-bits', 'boolean-circuit-5-bits', 'linear-interpolation-two-points']