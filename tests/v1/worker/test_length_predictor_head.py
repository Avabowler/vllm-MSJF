# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the output-length predictor head (CPU-only)."""

import pytest
import torch

from vllm.v1.worker.length_predictor_head import (
    IncrementalWeightedPooling,
    WeightedPoolingHead,
)

HIDDEN = 32
MLP_HIDDEN = 64
NUM_BUCKETS = 10


def test_incremental_pooling_matches_one_shot():
    """Chunked accumulation must equal pooling over the full prompt."""
    torch.manual_seed(0)
    head = WeightedPoolingHead(HIDDEN, MLP_HIDDEN, NUM_BUCKETS)
    prompt = torch.randn(37, HIDDEN)

    # One-shot reference.
    scores = head.score_tokens(prompt)
    weights = torch.softmax(scores, dim=0)
    expected = (weights.unsqueeze(-1) * prompt).sum(dim=0)

    # Incremental accumulation over uneven chunks.
    pooling = IncrementalWeightedPooling(HIDDEN, torch.device("cpu"))
    for start in range(0, 37, 8):
        chunk = prompt[start : start + 8]
        pooling.update(chunk, head.score_tokens(chunk))
    pooled = pooling.finalize(prompt.dtype)

    torch.testing.assert_close(pooled, expected, rtol=1e-4, atol=1e-5)


def test_incremental_pooling_rescales_on_larger_max():
    """A later chunk with the largest score must rescale earlier sums."""
    pooling = IncrementalWeightedPooling(4, torch.device("cpu"))
    small = torch.ones(3, 4)
    large = torch.full((2, 4), 2.0)
    pooling.update(small, torch.tensor([0.0, 0.1, 0.2]))
    pooling.update(large, torch.tensor([10.0, 20.0]))
    pooled = pooling.finalize(torch.float32)

    scores = torch.tensor([0.0, 0.1, 0.2, 10.0, 20.0])
    weights = torch.softmax(scores, dim=0)
    full = torch.cat([small, large])
    expected = (weights.unsqueeze(-1) * full).sum(dim=0)
    torch.testing.assert_close(pooled, expected, rtol=1e-4, atol=1e-6)


def test_predict_request_shapes_and_bucket_range():
    torch.manual_seed(1)
    head = WeightedPoolingHead(HIDDEN, MLP_HIDDEN, NUM_BUCKETS)
    prompt = torch.randn(50, HIDDEN)
    bucket, predicted_len, rank_score = head.predict_request(
        prompt, bucket_width=204.8
    )
    assert 0 <= bucket < NUM_BUCKETS
    assert int((bucket + 0.5) * 204.8) == predicted_len
    assert isinstance(rank_score, float)


def test_predict_request_is_deterministic():
    torch.manual_seed(2)
    head = WeightedPoolingHead(HIDDEN, MLP_HIDDEN, NUM_BUCKETS)
    prompt = torch.randn(20, HIDDEN)
    first = head.predict_request(prompt, bucket_width=100.0)
    second = head.predict_request(prompt, bucket_width=100.0)
    assert first == second


@pytest.mark.parametrize("num_chunks", [1, 3, 7])
def test_incremental_pooling_chunk_counts(num_chunks: int):
    torch.manual_seed(3)
    head = WeightedPoolingHead(HIDDEN, MLP_HIDDEN, NUM_BUCKETS)
    prompt = torch.randn(21, HIDDEN)
    pooling = IncrementalWeightedPooling(HIDDEN, torch.device("cpu"))
    bounds = [round(i * 21 / num_chunks) for i in range(num_chunks + 1)]
    for lo, hi in zip(bounds, bounds[1:]):
        chunk = prompt[lo:hi]
        pooling.update(chunk, head.score_tokens(chunk))
    expected = head.pool_all(prompt)
    torch.testing.assert_close(
        pooling.finalize(prompt.dtype), expected, rtol=1e-4, atol=1e-5
    )
