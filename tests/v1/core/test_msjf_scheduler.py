# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MSJF (memory-aware shortest-job-first) scheduling and the
output-length prediction plumbing."""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import LengthPredictorConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.length_predictor import LengthPredictor
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import (
    MSJFRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

EOS_TOKEN_ID = 50256
BLOCK_SIZE = 16


def test_msjf_queue_orders_by_cost():
    queue = MSJFRequestQueue()
    requests = create_requests(num_requests=3, num_tokens=10)
    for request, cost in zip(requests, (300.0, 100.0, 200.0)):
        request.msjf_cost = cost
        queue.add_request(request)

    assert [req.request_id for req in queue] == ["1", "2", "0"]
    assert queue.peek_request() is requests[1]
    assert queue.pop_request() is requests[1]
    assert len(queue) == 2


def test_msjf_queue_update_request_resorts():
    queue = MSJFRequestQueue()
    requests = create_requests(num_requests=2, num_tokens=10)
    requests[0].msjf_cost = 100.0
    requests[1].msjf_cost = 50.0
    for request in requests:
        queue.add_request(request)
    assert queue.peek_request() is requests[1]

    # Underestimation correction: escalate and re-sort.
    requests[1].msjf_cost = 300.0
    queue.update_request(requests[1])
    assert queue.peek_request() is requests[0]

    assert queue.pop_request() is requests[0]
    assert queue.pop_request() is requests[1]
    assert len(queue) == 0
    with pytest.raises(IndexError):
        queue.pop_request()


def test_msjf_queue_reinsert_same_cost_is_noop():
    # add_request/update_request with an unchanged cost must not create a
    # duplicate heap entry (a request must never be popped twice).
    queue = create_request_queue(SchedulingPolicy.MSJF)
    request = create_requests(num_requests=1, num_tokens=10)[0]
    request.msjf_cost = 42.0
    queue.add_request(request)
    queue.add_request(request)
    queue.update_request(request)
    assert len(queue) == 1
    assert queue.pop_request() is request
    with pytest.raises(IndexError):
        queue.pop_request()


def test_scheduler_msjf_admission_order():
    """Waiting requests are admitted smallest predicted footprint first."""
    scheduler = create_scheduler(
        scheduling_policy="msjf",
        length_predictor_config=LengthPredictorConfig(backend="client"),
    )
    # Same prompt length, decreasing predicted output lengths; submitted in
    # reverse (longest first) so FCFS would admit in a different order.
    requests = create_requests(
        num_requests=3,
        num_tokens=10,
        max_tokens=1024,
        output_len_predictions=[900, 100, 500],
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    scheduled_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_ids == ["1", "2", "0"]


def test_scheduler_fcfs_regression():
    """FCFS admission order is unaffected by the msjf additions."""
    scheduler = create_scheduler(scheduling_policy="fcfs")
    requests = create_requests(
        num_requests=3,
        num_tokens=10,
        max_tokens=1024,
        output_len_predictions=[900, 100, 500],
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    scheduled_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_ids == ["0", "1", "2"]


def _run_one_step(scheduler, output: "SchedulerOutput"):
    """Drive scheduler.update_from_output for the given schedule() output,
    generating one token per scheduled request."""
    req_ids = list(output.num_scheduled_tokens.keys())
    model_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[100] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)


def test_msjf_reservation_gate_defers_admission():
    """The predicted future demand of running requests throttles admission."""
    scheduler = create_scheduler(
        scheduling_policy="msjf",
        num_blocks=150,
        block_size=BLOCK_SIZE,
        length_predictor_config=LengthPredictorConfig(backend="client"),
    )
    # A running request predicted to produce 3000 output tokens.
    running = create_requests(
        num_requests=1,
        num_tokens=10,
        ignore_eos=True,
        max_tokens=4000,
        output_len_predictions=[3000],
        req_ids=["running"],
    )[0]
    scheduler.add_request(running)
    assert running.effective_output_len == 3000
    _run_one_step(scheduler, scheduler.schedule())
    assert len(scheduler.running) == 1

    # A waiting request whose first chunk needs 63 blocks; without the
    # reservation it would fit in the free blocks, but the running request's
    # predicted remaining demand (cdiv(min(10+3000, max_model_len)/16) - 1
    # blocks) must stay unreserved-free.
    waiting = create_requests(
        num_requests=1,
        num_tokens=1000,
        max_tokens=16,
        req_ids=["waiting"],
    )[0]
    scheduler.add_request(waiting)
    free_blocks = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    reserved = scheduler._msjf_running_reserved_blocks()
    assert reserved > 0
    chunk_blocks = (1000 + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert free_blocks - reserved < chunk_blocks

    output = scheduler.schedule()
    assert "waiting" not in output.num_scheduled_tokens
    assert len(scheduler.waiting) + len(scheduler.skipped_waiting) == 1

    # Disabling the reservation lets the request in (blocks are free).
    scheduler.scheduler_config.msjf_reservation_factor = 0.0
    output = scheduler.schedule()
    assert "waiting" in output.num_scheduled_tokens


def test_msjf_backfill_skips_to_smaller_request():
    """When the head fails admission, a request with a smaller chunk
    footprint further down the queue is admitted (best-fit packing)."""
    scheduler = create_scheduler(
        scheduling_policy="msjf",
        num_blocks=130,
        block_size=BLOCK_SIZE,
        length_predictor_config=LengthPredictorConfig(backend="client"),
    )
    # Head of the queue: smallest predicted output (so it sorts first under
    # MSJF) but a huge prompt whose first chunk (188 blocks) cannot fit in
    # the cache (129 usable blocks).
    head = create_requests(
        num_requests=1,
        num_tokens=3000,
        max_tokens=16,
        output_len_predictions=[10],
        req_ids=["head"],
    )[0]
    # Queued second: larger predicted output but a tiny prompt that fits.
    small = create_requests(
        num_requests=1,
        num_tokens=10,
        max_tokens=6000,
        output_len_predictions=[5000],
        req_ids=["small"],
    )[0]
    scheduler.add_request(head)
    scheduler.add_request(small)
    assert scheduler.waiting.peek_request() is head

    output = scheduler.schedule()
    assert "small" in output.num_scheduled_tokens
    assert "head" not in output.num_scheduled_tokens
    # The deferred head is preserved for later steps, not dropped.
    assert len(scheduler.waiting) + len(scheduler.skipped_waiting) == 1
    deferred = (scheduler.waiting or scheduler.skipped_waiting).peek_request()
    assert deferred is head


def test_msjf_full_fit_mode():
    """In full-fit mode admission requires the whole predicted sequence to
    fit, not just the first chunk."""
    common = dict(
        scheduling_policy="msjf",
        num_blocks=120,
        block_size=BLOCK_SIZE,
        length_predictor_config=LengthPredictorConfig(backend="client"),
    )

    # Reservation mode (default): only the first chunk (7 blocks) must fit.
    scheduler = create_scheduler(**common)
    waiting = create_requests(
        num_requests=1,
        num_tokens=100,
        max_tokens=6000,
        output_len_predictions=[5000],
        req_ids=["waiting"],
    )[0]
    scheduler.add_request(waiting)
    assert scheduler._msjf_admission_gate_blocks(waiting, 100) == 7
    output = scheduler.schedule()
    assert "waiting" in output.num_scheduled_tokens

    # Full-fit mode: the whole predicted sequence (min(100 + 5000,
    # max_model_len=2048) tokens = 128 blocks) must fit in the 119 usable
    # blocks -> deferred.
    scheduler_full = create_scheduler(**common)
    scheduler_full.scheduler_config.msjf_full_fit_mode = True
    waiting_full = create_requests(
        num_requests=1,
        num_tokens=100,
        max_tokens=6000,
        output_len_predictions=[5000],
        req_ids=["waiting"],
    )[0]
    scheduler_full.add_request(waiting_full)
    assert scheduler_full._msjf_admission_gate_blocks(waiting_full, 100) == 128
    output_full = scheduler_full.schedule()
    assert "waiting" not in output_full.num_scheduled_tokens
    assert len(scheduler_full.waiting) + len(scheduler_full.skipped_waiting) == 1


def test_msjf_high_watermark_pauses_admission():
    scheduler = create_scheduler(
        scheduling_policy="msjf",
        num_blocks=1000,
        block_size=BLOCK_SIZE,
        length_predictor_config=LengthPredictorConfig(backend="client"),
    )
    scheduler.scheduler_config.msjf_high_watermark = 0.05
    running = create_requests(
        num_requests=1,
        num_tokens=1000,
        ignore_eos=True,
        max_tokens=16,
        req_ids=["running"],
    )[0]
    scheduler.add_request(running)
    _run_one_step(scheduler, scheduler.schedule())
    assert scheduler.kv_cache_manager.usage > 0.05

    waiting = create_requests(
        num_requests=1,
        num_tokens=10,
        max_tokens=16,
        req_ids=["waiting"],
    )[0]
    scheduler.add_request(waiting)
    output = scheduler.schedule()
    assert "waiting" not in output.num_scheduled_tokens
    assert len(scheduler.waiting) == 1


def test_msjf_underestimation_correction():
    scheduler = create_scheduler(
        scheduling_policy="msjf",
        length_predictor_config=LengthPredictorConfig(backend="client"),
    )
    request = create_requests(
        num_requests=1,
        num_tokens=10,
        ignore_eos=True,
        max_tokens=100,
        output_len_predictions=[10],
        req_ids=["r"],
    )[0]
    scheduler.add_request(request)
    assert request.effective_output_len == 10

    output = scheduler.schedule()
    req_ids = ["r"]
    model_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={"r": 0},
        # 20 generated tokens > the prediction of 10.
        sampled_token_ids=[[100] * 20],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    assert request.output_len_underestimated
    assert request.effective_output_len == 25  # 20 * 1.25
    assert request.msjf_cost == 10 + 25
    assert scheduler.num_msjf_underestimated == 1


class _DummyConnector:
    """Minimal scheduler-side connector stub for _connector_finished."""

    def request_finished(self, request: Request, block_ids):
        return False, None


def test_pd_prediction_relayed_via_kv_transfer_params():
    """Predictions ride the free-form kv_transfer_params channel: the
    producer-side scheduler merges them into request_finished params, and a
    consumer-side Request ingests them before admission."""
    scheduler = create_scheduler(
        scheduling_policy="msjf",
        length_predictor_config=LengthPredictorConfig(backend="client"),
    )
    request = create_requests(
        num_requests=1,
        num_tokens=10,
        ignore_eos=True,
        max_tokens=100,
        output_len_predictions=[64],
        req_ids=["p"],
    )[0]
    scheduler.add_request(request)
    assert request.effective_output_len == 64

    # Simulate the predictor head producing a prediction this step.
    scheduler._apply_output_len_prediction(request, (3, 128, 0.75))
    assert request.predicted_output_len == 128
    assert request.predicted_bucket == 3
    # Predictions only escalate, and are clamped by max_tokens (100).
    assert request.effective_output_len == 100
    assert request.msjf_cost == 10 + 100

    # The merge into kv_transfer_params happens in _connector_finished.
    scheduler.connector = _DummyConnector()
    delay, params = scheduler._connector_finished(request)
    assert not delay
    assert params["output_len_prediction"] == 128
    assert params["output_len_bucket"] == 3
    assert params["output_len_rank_score"] == 0.75

    # Consumer side: the prediction rides in kv_transfer_params.
    consumer_params = SamplingParams(
        max_tokens=16,
        extra_args={
            "kv_transfer_params": {
                "output_len_prediction": 128,
                "output_len_bucket": 3,
                "output_len_rank_score": 0.75,
            }
        },
    )
    consumer_params.update_from_generation_config({}, EOS_TOKEN_ID)
    consumer_request = Request(
        request_id="d",
        prompt_token_ids=[0] * 10,
        sampling_params=consumer_params,
        pooling_params=None,
    )
    assert consumer_request.predicted_output_len == 128
    assert consumer_request.predicted_bucket == 3
    assert consumer_request.predicted_rank_score == 0.75


def test_mean_predictor_ewma():
    predictor = LengthPredictor(LengthPredictorConfig(backend="mean"))
    request = create_requests(num_requests=1, num_tokens=10)[0]
    assert predictor.estimate(request) is None

    request._output_token_ids.extend([0] * 100)
    request.predicted_output_len = 110
    request.predicted_bucket = 0
    predictor.record(request)
    assert predictor.mean_estimate() == 100

    request._output_token_ids.clear()
    request._output_token_ids.extend([0] * 50)
    request.predicted_output_len = 40
    request.predicted_bucket = 1  # wrong bucket (actual 50 -> bucket 0)
    predictor.record(request)
    # EWMA with alpha=0.1: 100 + 0.1 * (50 - 100) = 95.
    assert predictor.mean_estimate() == 95

    # Accuracy tracking over the two predicted requests.
    assert predictor.mean_abs_error == pytest.approx(10.0)  # |100-110|, |50-40|
    assert predictor.bucket_accuracy == pytest.approx(0.5)


def test_msjf_cost_mode_output_degenerates_to_sjf():
    """'output' cost mode ranks by predicted output only (prompt ignored),
    i.e. plain SJF; default 'footprint' keeps the memory-aware prompt term."""
    scheduler = create_scheduler(
        scheduling_policy="msjf",
        length_predictor_config=LengthPredictorConfig(backend="client"),
    )
    requests = create_requests(
        num_requests=2,
        num_tokens=10,
        max_tokens=1024,
        output_len_predictions=[100, 900],
    )

    # Default footprint mode: prompt + predicted output.
    assert scheduler.scheduler_config.msjf_cost_mode == "footprint"
    scheduler._init_msjf_cost(requests[0])
    assert requests[0].msjf_cost == pytest.approx(10 + 100)

    # 'output' mode: prompt term dropped — pure output-length SJF.
    scheduler.scheduler_config.msjf_cost_mode = "output"
    scheduler._init_msjf_cost(requests[1])
    assert requests[1].msjf_cost == pytest.approx(900)
