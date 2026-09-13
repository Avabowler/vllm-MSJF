# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output-length prediction head (PiLLM-style) for length-aware scheduling.

A small head over the serving LLM's own prefill hidden states: a per-token
FC scorer produces softmax weights, the text embedding is the weighted sum
of the token hidden states, and a shared MLP trunk feeds a bucket-classifier
head and a scalar ranking head. Only these head weights are trained; the
LLM itself is untouched.

With chunked prefill the weighted pooling is accumulated incrementally: the
softmax weights are a function of per-token scores only, so ``sum(exp(s))``
and ``sum(exp(s) * x)`` can be accumulated chunk by chunk (with a running
max for numerical stability) and normalized once at the final chunk.
"""

import json
from pathlib import Path

import torch
from torch import nn

from vllm.config import LengthPredictorConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

CHECKPOINT_CONFIG_FILE = "config.json"
CHECKPOINT_WEIGHTS_FILE = "model.safetensors"


class IncrementalWeightedPooling:
    """Numerically stable accumulator for softmax-weighted pooling across
    prefill chunks.

    Accumulates ``sum(exp(s - m) * x)`` and ``sum(exp(s - m))`` where ``m``
    is the running max score, rescaling on each update if the max grows.
    """

    def __init__(self, hidden_size: int, device: torch.device) -> None:
        self._max: torch.Tensor | None = None
        self._sum_exp = torch.zeros(1, device=device, dtype=torch.float32)
        self._wsum = torch.zeros(1, hidden_size, device=device, dtype=torch.float32)

    def update(self, hidden_states: torch.Tensor, scores: torch.Tensor) -> None:
        """Accumulate one prefill chunk.

        Args:
            hidden_states: [n, hidden_size] chunk hidden states.
            scores: [n] per-token pooling scores.
        """
        scores = scores.reshape(-1).float()
        chunk_max = scores.max()
        if self._max is None:
            self._max = chunk_max.detach().clone()
        new_max = torch.maximum(self._max, chunk_max)
        scale = torch.exp(self._max - new_max)
        exp = torch.exp(scores - new_max)
        self._sum_exp = self._sum_exp * scale + exp.sum()
        weighted = exp.unsqueeze(-1) * hidden_states.float()
        self._wsum = self._wsum * scale + weighted.sum(dim=0, keepdim=True)
        self._max = new_max.detach().clone()

    def finalize(self, dtype: torch.dtype) -> torch.Tensor:
        """Return the pooled text embedding: [hidden_size]."""
        pooled = self._wsum / self._sum_exp
        return pooled.reshape(-1).to(dtype)


class WeightedPoolingHead(nn.Module):
    """Weighted-pooling FC layer + MLP trunk + bucket classifier and ranker
    heads."""

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int,
        num_buckets: int,
    ) -> None:
        super().__init__()
        self.pool_scorer = nn.Linear(hidden_size, 1, bias=False)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(mlp_hidden_size, num_buckets)
        self.ranker = nn.Linear(mlp_hidden_size, 1)

    def score_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Per-token pooling scores: [num_tokens]."""
        return self.pool_scorer(hidden_states).squeeze(-1)

    def predict(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bucket logits [num_buckets] and rank score [] from the pooled
        text embedding."""
        trunk = self.trunk(pooled)
        return self.classifier(trunk), self.ranker(trunk).squeeze()

    def pool_all(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Weighted pooling over a full prompt in one shot: [hidden_size]."""
        scores = self.score_tokens(hidden_states)
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(-1) * hidden_states).sum(dim=0)

    def predict_request(
        self,
        hidden_states: torch.Tensor,
        bucket_width: float,
    ) -> tuple[int, int, float]:
        """One-shot prediction over full-prompt hidden states.

        Returns (bucket, predicted_output_len, rank_score)."""
        with torch.inference_mode():
            pooled = self.pool_all(hidden_states)
            bucket_logits, rank_score = self.predict(pooled)
            bucket = int(bucket_logits.argmax().item())
            predicted_len = int((bucket + 0.5) * float(bucket_width))
            return bucket, predicted_len, float(rank_score.item())


def load_length_predictor_head(
    config: LengthPredictorConfig,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> WeightedPoolingHead:
    """Build the head on device and load its checkpoint (safetensors)."""
    checkpoint_dir = Path(config.checkpoint)  # type: ignore[arg-type]
    with (checkpoint_dir / CHECKPOINT_CONFIG_FILE).open() as f:
        meta = json.load(f)
    if meta.get("hidden_size") != hidden_size:
        raise ValueError(
            f"Length predictor checkpoint was trained with "
            f"hidden_size={meta.get('hidden_size')}, but the served model "
            f"has hidden_size={hidden_size}."
        )

    from safetensors.torch import load_file

    head = WeightedPoolingHead(
        hidden_size=hidden_size,
        mlp_hidden_size=config.mlp_hidden_size,
        num_buckets=config.num_buckets,
    ).to(device=device, dtype=dtype)
    weights = load_file(str(checkpoint_dir / CHECKPOINT_WEIGHTS_FILE))
    missing, unexpected = head.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Length predictor checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    head.eval()
    logger.info(
        "Loaded output-length predictor head from %s (%d buckets).",
        checkpoint_dir,
        config.num_buckets,
    )
    return head
