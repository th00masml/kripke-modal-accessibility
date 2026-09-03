"""Apple MLX binding for judgement-set constrained decoding."""
from __future__ import annotations

from time import perf_counter_ns

import mlx.core as mx

from constraint import JudgementSetConstraint


class JudgementSetLogitsProcessor:
    def __init__(
        self,
        constraint: JudgementSetConstraint,
        prompt_length: int,
        prompt_tokens: list[int],
        eos_token_ids: list[int],
        allow_empty_output: bool,
    ):
        self.constraint = constraint
        self.prompt_length = prompt_length
        self.prompt_tokens = tuple(int(token) for token in prompt_tokens)
        self.eos_token_ids = set(eos_token_ids)
        self.allow_empty_output = allow_empty_output
        self.elapsed_ns = 0
        self.trace: list[dict[str, int]] = []

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        started = perf_counter_ns()
        token_seq = tokens.tolist()
        if token_seq and isinstance(token_seq[0], list):
            token_seq = token_seq[0]

        token_seq = [int(token) for token in token_seq]
        if len(token_seq) >= self.prompt_length and tuple(token_seq[: self.prompt_length]) == self.prompt_tokens:
            generated = token_seq[self.prompt_length :]
        else:
            generated = token_seq

        prefix = b"".join(self.constraint.token_bytes.get(token, b"") for token in generated)
        allowed = set(self.constraint.allowed(prefix))

        # If no continuation is possible, avoid producing an all -inf mask and
        # resynchronize from root frontier.
        if not allowed:
            generated = []
            prefix = b""
            allowed = set(self.constraint.allowed(prefix))

        # For in-language fixtures we do not allow immediate abstention.
        # EOS is admitted only after reaching a terminal canonical string.
        if self.allow_empty_output:
            allowed |= self.eos_token_ids
        elif len(generated) > 0 and self.constraint.accepts(prefix):
            allowed |= self.eos_token_ids

        self.trace.append(
            {
                "allowed": len(allowed),
                "vocab": logits.shape[-1],
                "generated_len": len(generated),
                "prefix_bytes": len(prefix),
            }
        )
        masked = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        idx = mx.array(sorted(allowed), dtype=mx.uint32)
        if len(logits.shape) == 1:
            masked[idx] = 0
        else:
            masked[0, idx] = 0
        self.elapsed_ns += perf_counter_ns() - started
        return logits + masked


def vocabulary_bytes(tokenizer) -> dict[int, bytes]:
    return {
        token_id: tokenizer.decode([token_id], skip_special_tokens=False).encode("utf-8")
        for token_id in range(tokenizer.vocab_size)
    }