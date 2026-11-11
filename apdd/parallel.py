"""Deterministic helpers for partitioning schedule-evaluation tasks."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from queue import Empty
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .schedule import Schedule

if TYPE_CHECKING:
    from .metrics import EvaluationResult


TaskPayload = tuple[int, tuple[int, ...]]
TaskCompleteCallback = Callable[[int, int, "EvaluationResult"], None]


@dataclass
class ParallelContext:
    gpu_count: int
    command_queues: tuple[Any, ...]
    result_queue: Any
    processes: tuple[Any, ...]
    next_batch_id: int = 0
    broken: bool = False
    broken_reason: str | None = None


def partition_schedule_tasks(
    tasks: Sequence[TaskPayload],
    *,
    protocol_samples: int,
    gpu_count: int,
) -> tuple[tuple[TaskPayload, ...], ...]:
    """Assign whole schedule tasks to ranks by deterministic weighted LPT."""
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    task_list = list(tasks)
    task_ids = [task_id for task_id, _nodes in task_list]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task IDs must be unique")
    if any(len(nodes) < 2 for _task_id, nodes in task_list):
        raise ValueError("every task must contain a complete Schedule")
    if gpu_count == 1:
        return (tuple(task_list),)

    partitions: list[list[TaskPayload]] = [[] for _ in range(gpu_count)]
    loads = [0 for _ in range(gpu_count)]
    ordered = sorted(
        task_list,
        key=lambda item: (-(len(item[1]) - 1) * protocol_samples, item[0]),
    )
    for task in ordered:
        rank = min(range(gpu_count), key=lambda item: (loads[item], item))
        partitions[rank].append(task)
        loads[rank] += (len(task[1]) - 1) * protocol_samples
    return tuple(tuple(group) for group in partitions)


def _resolve_evaluator(evaluate_fn: Callable[..., Any] | None) -> Callable[..., Any]:
    if evaluate_fn is not None:
        return evaluate_fn
    from .metrics import evaluate_schedule

    return evaluate_schedule


def _evaluate_tasks(
    components: Any,
    tasks: tuple[TaskPayload, ...],
    *,
    rank: int,
    seeds: tuple[int, ...],
    options: Any,
    protocol: str,
    evaluate_fn: Callable[..., Any],
    show_progress: bool,
    on_task_complete: TaskCompleteCallback | None,
) -> tuple[tuple[int, Any], ...]:
    results: list[tuple[int, Any]] = []
    for task_id, nodes in tasks:
        result = evaluate_fn(
            components.model,
            Schedule(nodes),
            components.reference,
            seeds,
            options,
            protocol,
            feature_extractor=components.feature_extractor,
            show_progress=show_progress,
        )
        results.append((task_id, result))
        if on_task_complete is not None:
            on_task_complete(rank, task_id, result)
    return tuple(results)


def _schedule_worker(
    rank: int,
    command_queue: Any,
    result_queue: Any,
    seed: int,
    deterministic: bool,
    allow_tf32: bool,
    component_builder: Callable[..., Any],
    component_kwargs: Mapping[str, Any],
    evaluate_fn: Callable[..., Any] | None,
    runtime_initializer: Callable[..., None] | None,
) -> None:
    try:
        torch.cuda.set_device(rank)
        if runtime_initializer is None:
            from .model_data import initialize_runtime

            runtime_initializer = initialize_runtime
        runtime_initializer(
            seed,
            deterministic=deterministic,
            allow_tf32=allow_tf32,
            cuda_device=rank,
        )
        build_kwargs = dict(component_kwargs)
        build_kwargs["device"] = f"cuda:{rank}"
        components = component_builder(**build_kwargs)
        worker_evaluator = _resolve_evaluator(evaluate_fn)
        result_queue.put(("ready", rank))
    except Exception as error:  # noqa: BLE001 - every worker failure needs an envelope.
        result_queue.put(
            ("error", None, rank, type(error).__name__, str(error))
        )
        return
    except BaseException as error:  # noqa: BLE001 - fatal startup is fail-stop.
        try:
            result_queue.put(
                ("fatal_error", None, rank, type(error).__name__, str(error))
            )
        except BaseException:  # noqa: BLE001 - dead-worker detection is the fallback.
            pass
        return

    while True:
        try:
            command = command_queue.get()
            if command == ("stop",):
                return
            kind, batch_id, tasks, seeds, options, protocol = command
            if kind != "batch":
                raise ValueError(f"unknown worker command: {kind!r}")
        except BaseException as error:  # noqa: BLE001 - control-plane failure is fatal.
            try:
                result_queue.put(
                    ("fatal_error", None, rank, type(error).__name__, str(error))
                )
            except BaseException:  # noqa: BLE001 - dead-worker detection is the fallback.
                pass
            return

        try:
            def send_completed(
                completed_rank: int,
                task_id: int,
                result: Any,
            ) -> None:
                result_queue.put(
                    ("task_result", batch_id, completed_rank, task_id, result)
                )

            _evaluate_tasks(
                components,
                tasks,
                rank=rank,
                seeds=seeds,
                options=options,
                protocol=protocol,
                evaluate_fn=worker_evaluator,
                show_progress=False,
                on_task_complete=send_completed,
            )
            result_queue.put(("rank_done", batch_id, rank))
        except Exception as error:  # noqa: BLE001 - report one batch error upstream.
            result_queue.put(
                ("error", batch_id, rank, type(error).__name__, str(error))
            )
        except BaseException as error:  # noqa: BLE001 - fatal batch is fail-stop.
            try:
                result_queue.put(
                    (
                        "fatal_error",
                        batch_id,
                        rank,
                        type(error).__name__,
                        str(error),
                    )
                )
            except BaseException:  # noqa: BLE001 - dead-worker detection is the fallback.
                pass
            return


def _wait_for_worker_envelope(
    context: ParallelContext,
    pending_ranks: set[int],
    batch_id: int | None,
) -> tuple[Any, ...]:
    while True:
        try:
            return context.result_queue.get(timeout=0.2)
        except Empty:
            dead_ranks = [
                rank
                for rank in pending_ranks
                if not context.processes[rank - 1].is_alive()
            ]
            if not dead_ranks:
                continue
            try:
                return context.result_queue.get(timeout=0.5)
            except Empty:
                rank = dead_ranks[0]
                exitcode = context.processes[rank - 1].exitcode
                return (
                    "fatal_error",
                    batch_id,
                    rank,
                    "WorkerProcessExit",
                    f"process exited without an envelope (exitcode={exitcode})",
                )


def _collect_worker_envelopes(
    context: ParallelContext,
    batch_id: int | None,
    on_task_complete: TaskCompleteCallback | None = None,
) -> list[tuple[int, Any]]:
    pending = set(range(1, context.gpu_count))
    gathered: list[tuple[int, Any]] = []
    errors: list[tuple[int, str, str]] = []
    fatal_errors: list[tuple[int, str, str]] = []
    while pending:
        envelope = _wait_for_worker_envelope(context, pending, batch_id)
        kind = envelope[0] if envelope else None
        worker_error: tuple[str, str] | None = None
        worker_fatal = False
        if batch_id is None and kind == "ready" and len(envelope) == 2:
            rank = envelope[1]
            received_batch_id = None
        elif batch_id is not None and kind == "task_result" and len(envelope) == 5:
            _kind, received_batch_id, rank, task_id, result = envelope
        elif batch_id is not None and kind == "rank_done" and len(envelope) == 3:
            _kind, received_batch_id, rank = envelope
        elif kind == "error" and len(envelope) == 5:
            _kind, received_batch_id, rank, error_type, message = envelope
            worker_error = (error_type, message)
        elif kind == "fatal_error" and len(envelope) == 5:
            _kind, received_batch_id, rank, error_type, message = envelope
            worker_error = (error_type, message)
            worker_fatal = True
        else:
            reason = f"invalid worker envelope: {envelope!r}"
            context.broken = True
            if context.broken_reason is None:
                context.broken_reason = reason
            raise RuntimeError(reason)
        if rank not in pending:
            reason = f"duplicate or unexpected envelope from rank {rank}"
            context.broken = True
            if context.broken_reason is None:
                context.broken_reason = reason
            raise RuntimeError(reason)
        unbound_active_fatal = (
            batch_id is not None
            and kind == "fatal_error"
            and received_batch_id is None
        )
        if received_batch_id != batch_id and not unbound_active_fatal:
            reason = (
                f"worker rank {rank} returned batch {received_batch_id}, "
                f"expected {batch_id}"
            )
            context.broken = True
            if context.broken_reason is None:
                context.broken_reason = reason
            raise RuntimeError(reason)
        if kind == "task_result":
            gathered.append((task_id, result))
            if on_task_complete is not None:
                on_task_complete(rank, task_id, result)
            continue
        pending.remove(rank)
        if worker_error is not None:
            error_type, message = worker_error
            if worker_fatal:
                stage = "startup" if batch_id is None else f"batch {batch_id}"
                reason = (
                    f"schedule worker rank {rank} encountered fatal error during "
                    f"{stage}: {error_type}: {message}"
                )
                context.broken = True
                if context.broken_reason is None:
                    context.broken_reason = reason
                if batch_id is None:
                    raise RuntimeError(reason)
                fatal_errors.append((rank, error_type, message))
                continue
            if batch_id is None:
                raise RuntimeError(
                    f"schedule worker rank {rank} failed during startup: "
                    f"{error_type}: {message}"
                )
            errors.append((rank, error_type, message))
    if fatal_errors:
        rank, error_type, message = fatal_errors[0]
        raise RuntimeError(
            f"schedule worker rank {rank} encountered fatal error during "
            f"batch {batch_id}: {error_type}: {message}"
        )
    if errors:
        rank, error_type, message = errors[0]
        stage = "startup" if batch_id is None else f"batch {batch_id}"
        raise RuntimeError(
            f"schedule worker rank {rank} failed during {stage}: "
            f"{error_type}: {message}"
        )
    return gathered


def start_schedule_workers(
    gpu_count: int,
    *,
    seed: int,
    deterministic: bool,
    allow_tf32: bool,
    component_builder: Callable[..., Any],
    component_kwargs: Mapping[str, Any] | None = None,
    evaluate_fn: Callable[..., Any] | None = None,
    runtime_initializer: Callable[..., None] | None = None,
) -> ParallelContext | None:
    """Start persistent schedule workers for logical ranks 1 through G-1."""
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    if gpu_count == 1:
        return None

    mp_context = torch.multiprocessing.get_context("spawn")
    result_queue = mp_context.Queue()
    command_queues = tuple(mp_context.Queue() for _rank in range(1, gpu_count))
    processes = tuple(
        mp_context.Process(
            target=_schedule_worker,
            name=f"apdd-schedule-rank-{rank}",
            args=(
                rank,
                command_queues[rank - 1],
                result_queue,
                seed,
                deterministic,
                allow_tf32,
                component_builder,
                {} if component_kwargs is None else dict(component_kwargs),
                evaluate_fn,
                runtime_initializer,
            ),
        )
        for rank in range(1, gpu_count)
    )
    context = ParallelContext(gpu_count, command_queues, result_queue, processes)
    try:
        for process in processes:
            process.start()
        _collect_worker_envelopes(context, None)
    except BaseException:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        close_schedule_workers(context)
        raise
    return context


def evaluate_schedule_batch(
    context: ParallelContext | None,
    tasks: Sequence[TaskPayload],
    *,
    rank0_components: Any,
    protocol_samples: int,
    seeds: tuple[int, ...],
    options: Any,
    protocol: str,
    evaluate_fn: Callable[..., Any] | None = None,
    on_task_complete: TaskCompleteCallback | None = None,
) -> list[tuple[int, EvaluationResult]]:
    """Evaluate one immutable task batch across rank 0 and persistent children."""
    gpu_count = 1 if context is None else context.gpu_count
    task_batch = tuple((task_id, tuple(nodes)) for task_id, nodes in tasks)
    partitions = partition_schedule_tasks(
        task_batch,
        protocol_samples=protocol_samples,
        gpu_count=gpu_count,
    )
    worker_evaluator = _resolve_evaluator(evaluate_fn)

    callback_lock = Lock()
    callback_errors: list[BaseException] = []

    def notify(rank: int, task_id: int, result: EvaluationResult) -> None:
        if on_task_complete is not None:
            with callback_lock:
                if callback_errors:
                    return
                try:
                    on_task_complete(rank, task_id, result)
                except BaseException as error:  # noqa: BLE001 - raise after drain.
                    callback_errors.append(error)

    batch_id = -1
    child_results: list[tuple[int, Any]] = []
    collector_errors: list[BaseException] = []
    collector: Thread | None = None
    if context is not None:
        if context.broken:
            reason = context.broken_reason or "unknown fatal worker failure"
            raise RuntimeError(f"schedule worker context is broken: {reason}")
        dead_ranks = [
            rank
            for rank, process in enumerate(context.processes, start=1)
            if not process.is_alive()
        ]
        if dead_ranks:
            details = ", ".join(
                f"rank {rank} is not alive "
                f"(exitcode={context.processes[rank - 1].exitcode})"
                for rank in dead_ranks
            )
            context.broken = True
            context.broken_reason = details
            raise RuntimeError(f"schedule worker context is broken: {details}")
        batch_id = context.next_batch_id
        context.next_batch_id += 1
        for rank, command_queue in enumerate(context.command_queues, start=1):
            command_queue.put(
                (
                    "batch",
                    batch_id,
                    partitions[rank],
                    tuple(seeds),
                    options,
                    protocol,
                )
            )

        def collect_children() -> None:
            try:
                child_results.extend(
                    _collect_worker_envelopes(
                        context,
                        batch_id,
                        on_task_complete=notify,
                    )
                )
            except BaseException as error:  # noqa: BLE001 - propagate after join.
                collector_errors.append(error)

        collector = Thread(
            target=collect_children,
            name=f"apdd-batch-{batch_id}-collector",
            daemon=False,
        )
        collector.start()

    local_error: BaseException | None = None
    results: list[tuple[int, Any]] = []
    try:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_cpu_state = torch.random.get_rng_state()
        torch_cuda_state = (
            torch.cuda.get_rng_state(device=0) if torch.cuda.is_available() else None
        )
        try:
            results.extend(
                _evaluate_tasks(
                    rank0_components,
                    partitions[0],
                    rank=0,
                    seeds=tuple(seeds),
                    options=options,
                    protocol=protocol,
                    evaluate_fn=worker_evaluator,
                    show_progress=context is None,
                    on_task_complete=notify,
                )
            )
        except BaseException as error:  # noqa: BLE001 - join before re-raising.
            local_error = error
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_cpu_state)
            if torch_cuda_state is not None:
                torch.cuda.set_rng_state(torch_cuda_state, device=0)
    finally:
        if collector is not None:
            collector.join()

    if collector is not None:
        results.extend(child_results)
    collector_error = collector_errors[0] if collector_errors else None
    if local_error is not None and not isinstance(local_error, Exception):
        if collector_error is not None:
            raise local_error from collector_error
        raise local_error
    callback_error = callback_errors[0] if callback_errors else None
    if callback_error is not None and not isinstance(callback_error, Exception):
        if collector_error is not None:
            raise callback_error from collector_error
        raise callback_error
    if collector_error is not None:
        raise collector_error
    if callback_error is not None:
        raise callback_error
    if local_error is not None:
        raise RuntimeError(
            f"schedule evaluation failed on rank 0: "
            f"{type(local_error).__name__}: {local_error}"
        ) from local_error

    expected_ids = [task_id for task_id, _nodes in task_batch]
    result_ids = [task_id for task_id, _result in results]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("schedule batch returned duplicate task IDs")
    if set(result_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(result_ids))
        unexpected = sorted(set(result_ids) - set(expected_ids))
        raise RuntimeError(
            f"schedule batch task IDs do not match input; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return sorted(results, key=lambda item: item[0])


def close_schedule_workers(context: ParallelContext | None) -> None:
    """Stop all persistent child workers and release their queues."""
    if context is None:
        return
    for command_queue, process in zip(context.command_queues, context.processes):
        if process.pid is not None and process.is_alive():
            command_queue.put(("stop",))
    for process in context.processes:
        if process.pid is not None:
            process.join(timeout=5.0)
    for process in context.processes:
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join()
    for command_queue in context.command_queues:
        command_queue.close()
        command_queue.join_thread()
    context.result_queue.close()
    context.result_queue.join_thread()
