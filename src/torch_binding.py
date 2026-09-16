"""CUDA/PyTorch (transformers) binding for judgement-set constrained decoding.

Unlike the MLX processor, transformers.LogitsProcessor receives the full
input_ids sequence (prompt + generated-so-far) on every call, so generated
token context is always available without reconstruction heuristics.
"""
from __future__ import annotations

from time import perf_counter_ns

import torch
from transformers import LogitsProcessor

from constraint import JudgementSetConstraint


class JudgementSetLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        constraint: JudgementSetConstraint,
        prompt_length: int,
        eos_token_ids: list[int],
        allow_empty_output: bool,
    ):
        self.constraint = constraint
        self.prompt_length = prompt_length
        self.eos_token_ids = set(eos_token_ids)
        self.allow_empty_output = allow_empty_output
        self.elapsed_ns = 0
        self.trace: list[dict[str, int]] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        started = perf_counter_ns()
        generated = [int(t) for t in input_ids[0, self.prompt_length :].tolist()]

        prefix = b"".join(self.constraint.token_bytes.get(token, b"") for token in generated)
        allowed = set(self.constraint.allowed(prefix))
        can_stop = self.allow_empty_output or (len(generated) > 0 and self.constraint.accepts(prefix))

        # If no continuation is possible and we have not reached a valid
        # terminal string, resynchronize from root frontier. Do NOT resync
        # once a canonical judgement is complete, or generation never stops.
        if not allowed and not can_stop:
            generated = []
            prefix = b""
            allowed = set(self.constraint.allowed(prefix))

        # For in-language fixtures we do not allow immediate abstention.
        # EOS is admitted only after reaching a terminal canonical string.
        if can_stop:
            allowed |= self.eos_token_ids

        self.trace.append(
            {
                "allowed": len(allowed),
                "vocab": scores.shape[-1],
                "generated_len": len(generated),
                "prefix_bytes": len(prefix),
            }
        )

        masked = torch.full_like(scores, float("-inf"))
        idx = torch.tensor(sorted(allowed), dtype=torch.long, device=scores.device)
        masked[0, idx] = scores[0, idx]
        self.elapsed_ns += perf_counter_ns() - started
        return masked


def vocabulary_bytes(tokenizer) -> dict[int, bytes]:
    return {
        token_id: tokenizer.decode([token_id], skip_special_tokens=False).encode("utf-8")
        for token_id in range(len(tokenizer))
    }