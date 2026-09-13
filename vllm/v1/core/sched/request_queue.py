# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum
from itertools import count

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    # Memory-aware shortest job first: waiting requests are ordered by their
    # estimated KV cache footprint (Request.msjf_cost, in tokens).
    MSJF = "msjf"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class MSJFRequestQueue(RequestQueue):
    """Memory-aware shortest-job-first queue ordered by estimated KV footprint.

    Requests are ordered by ``Request.msjf_cost`` (estimated KV footprint in
    tokens: prompt length + effective predicted output length), breaking ties
    by arrival time. Heap entries carry a cost snapshot; when the scheduler
    escalates a request's estimate (underestimation correction) it calls
    :meth:`update_request`, pushing a fresh entry — outdated snapshots are
    discarded lazily when they surface at the heap head. At most one entry per
    request can match its current cost (guarded by ``_member_costs``), so
    pop/peek never see duplicates.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, float, int, Request]] = []
        self._members: dict[str, Request] = {}
        # request_id -> msjf_cost snapshot of the request's live entry.
        self._member_costs: dict[str, float] = {}
        self._seq = count()

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the MSJF policy."""
        req_id = request.request_id
        if req_id in self._member_costs and (
            self._member_costs[req_id] == request.msjf_cost
        ):
            return
        self._members[req_id] = request
        self._member_costs[req_id] = request.msjf_cost
        heapq.heappush(
            self._heap,
            (request.msjf_cost, request.arrival_time, next(self._seq), request),
        )

    def _purge_stale_head(self) -> None:
        # Any cost change is accompanied by update_request(), which pushes a
        # fresh entry, so a head whose snapshot no longer matches the
        # request's current cost is outdated and can be discarded.
        while self._heap and self._heap[0][0] != self._heap[0][3].msjf_cost:
            heapq.heappop(self._heap)

    def update_request(self, request: Request) -> None:
        """Re-sort a queued request after its cost changed."""
        if request.request_id in self._members:
            self.add_request(request)

    def pop_request(self) -> Request:
        """Pop the request with the smallest estimated KV footprint."""
        self._purge_stale_head()
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, _, request = heapq.heappop(self._heap)
        del self._members[request.request_id]
        del self._member_costs[request.request_id]
        return request

    def peek_request(self) -> Request:
        """Peek at the smallest-footprint request without removing it."""
        self._purge_stale_head()
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0][3]

    def prepend_request(self, request: Request) -> None:
        """Re-queue a request (e.g. after preemption).

        Note: A heap has no front; the request is (re-)inserted at its
        cost-based position."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Re-queue all requests from another queue according to the MSJF
        policy."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._members.pop(request.request_id, None)
        self._member_costs.pop(request.request_id, None)
        self._heap = [entry for entry in self._heap if entry[3] is not request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        removed_ids = {r.request_id for r in requests_to_remove}
        for req_id in removed_ids:
            self._members.pop(req_id, None)
            self._member_costs.pop(req_id, None)
        self._heap = [
            entry for entry in self._heap if entry[3].request_id not in removed_ids
        ]
        heapq.heapify(self._heap)

    def __contains__(self, request: object) -> bool:
        return isinstance(request, Request) and request.request_id in self._members

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._members)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._members)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue in MSJF order (by current cost)."""
        return iter(
            sorted(
                self._members.values(),
                key=lambda r: (r.msjf_cost, r.arrival_time),
            )
        )


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.MSJF:
        return MSJFRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
