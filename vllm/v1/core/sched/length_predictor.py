# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output-length prediction backends for length-aware scheduling (MSJF).

The scheduler consults a :class:`LengthPredictor` when a request is admitted
to the waiting queue to obtain an output-length estimate (in tokens). The
"mlp" backend (the paper-faithful PiLLM predictor over prefill hidden states)
produces predictions in the model runner; they reach the request via
``predicted_output_len`` and this class simply surfaces them, falling back to
the running mean when a request has no prediction yet.
"""

from vllm.config import LengthPredictorConfig
from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)


class LengthPredictor:
    """Provides output-length estimates and tracks prediction quality.

    Estimate fallback chain: an explicit prediction attached to the request
    (client, oracle, or the mlp backend) -> exponentially weighted mean of
    observed output lengths of finished requests -> None (the caller then
    falls back to ``request.max_tokens``).
    """

    def __init__(self, config: LengthPredictorConfig) -> None:
        self.config = config
        self.backend = config.backend
        # Exponentially weighted mean of observed output lengths.
        self._ewma: float | None = None
        self._ewma_alpha = 0.1
        # Prediction quality tracking (requests with an explicit prediction).
        self._num_predictions = 0
        self._abs_error_sum = 0
        self._num_bucket_correct = 0

    def estimate(self, request: Request) -> int | None:
        """Return the predicted output length in tokens, or None."""
        if self.backend == "none":
            return None
        if request.predicted_output_len is not None:
            return request.predicted_output_len
        return self.mean_estimate()

    def mean_estimate(self) -> int | None:
        """Estimate from the running mean of observed output lengths."""
        if self._ewma is None:
            return None
        return int(round(self._ewma))

    def record(self, request: Request) -> None:
        """Update the running statistics with a finished request."""
        actual = request.num_output_tokens
        if self._ewma is None:
            self._ewma = float(actual)
        else:
            self._ewma += self._ewma_alpha * (actual - self._ewma)

        if request.predicted_output_len is not None:
            self._num_predictions += 1
            self._abs_error_sum += abs(actual - request.predicted_output_len)
            bucket = self.bucket_of(actual)
            if bucket == request.predicted_bucket:
                self._num_bucket_correct += 1

    def bucket_of(self, output_len: int) -> int:
        bucket_width = self.config.max_output_len / self.config.num_buckets
        return min(int(output_len / bucket_width), self.config.num_buckets - 1)

    def bucket_center(self, bucket: int) -> int:
        bucket_width = self.config.max_output_len / self.config.num_buckets
        return int((bucket + 0.5) * bucket_width)

    @property
    def mean_abs_error(self) -> float | None:
        if not self._num_predictions:
            return None
        return self._abs_error_sum / self._num_predictions

    @property
    def bucket_accuracy(self) -> float | None:
        if not self._num_predictions:
            return None
        return self._num_bucket_correct / self._num_predictions
