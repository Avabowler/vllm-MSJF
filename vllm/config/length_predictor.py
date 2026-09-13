# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from pydantic import Field

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

LengthPredictorBackend = Literal["none", "mlp", "client", "oracle", "mean"]


@config
class LengthPredictorConfig:
    """Configuration for output-length prediction used by length-aware
    scheduling (the ``msjf`` scheduling policy)."""

    backend: LengthPredictorBackend = "none"
    """Output-length predictor backend:

    - "none": disabled; length-aware scheduling falls back to
      ``SamplingParams.max_tokens`` as the length estimate.
    - "client": predictions supplied by the client via
      ``SamplingParams.extra_args["output_len_prediction"]``, or piggybacked
      from a prefill node through ``kv_transfer_params`` in a P/D deployment.
    - "oracle": ground-truth output lengths supplied by the benchmark harness
      via ``extra_args`` (upper bound for scheduler-gain evaluation).
    - "mean": exponentially weighted mean of the observed output lengths of
      finished requests (zero-cost runtime fallback, no model needed).
    - "mlp": the paper-faithful PiLLM predictor — a weighted-pooling FC layer
      plus MLP head over the LLM's own prefill hidden states. Requires
      ``checkpoint``."""

    checkpoint: str | None = None
    """Path to the predictor head checkpoint (safetensors) when
    backend="mlp"."""

    num_buckets: int = Field(default=10, ge=2)
    """Number of output-length buckets. Bucket i covers
    [i * max_output_len / num_buckets, (i+1) * max_output_len / num_buckets)."""

    max_output_len: int = Field(default=2048, ge=1)
    """Maximum output length covered by the bucketing."""

    mlp_hidden_size: int = Field(default=4096, ge=1)
    """Hidden size of the MLP head; must match the checkpoint."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.
        """
        # The predictor runs off the critical path and does not change the
        # computation graph; only the mlp checkpoint changes what is loaded
        # alongside the model, which is tracked by the engine via the config
        # identity itself.
        factors: list[str] = [self.backend, self.checkpoint or ""]
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.backend == "mlp" and not self.checkpoint:
            raise ValueError(
                "length_predictor backend='mlp' requires a checkpoint path"
            )
