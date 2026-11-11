"""执行 APDD 的 Warmup、Evolution 与最终 FID 评价流程。"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import diffusers
import numpy as np
import torch
from diffusers import DDIMPipeline

from .diffusion import sample_schedule
from .evolution import (
    BreedingArchives,
    Individual,
    apdd_delete,
    apdd_point_mutate,
    environmental_selection,
    non_dominated_sort,
    random_immigrant,
    segment_crossover,
    select_fixed_k_by_fid,
)
from .metrics import (
    EvaluationOptions,
    EvaluationResult,
    ReferenceStatistics,
    feature_extractor_identity,
    seed_stream,
)
from .model_data import initialize_runtime
from .parallel import ParallelContext, evaluate_schedule_batch
from .schedule import LogSNRTransform, Schedule
from .score import deletion_scores


@dataclass(frozen=True)
# 集中保存会改变 APDD 科研结果的显式搜索参数。
class SearchSettings:
    

    experiment_name: str
    results_root: str
    seed: int
    k_max: int
    k_min: int
    population_size: int
    warmup_generations: int
    max_generations: int
    search_samples: int
    archive_samples: int
    final_samples: int
    batch_size: int
    per_k_archive_size: int
    minimum_gap: int
    deletion_strategy: str
    top_r: int
    removal_temperature: float
    apdd_samples: int
    apdd_probes: int
    apdd_probe_seed: int
    deterministic: bool = True
    allow_tf32: bool = False
    resume: bool = True
    deletion_score_mode: str = "adjoint"


@dataclass
# 保存完整迭代边界可恢复的 A、E、当前批次与评价缓存。
class SearchState:
    stage: str
    generation: int
    population: list[Individual]
    pareto: list[Individual]
    breeding: dict[int, list[Individual]]
    evaluations: dict[tuple[str, tuple[int, ...]], EvaluationResult]
    compatibility: dict[str, Any] = field(default_factory=dict)
    search_pareto: list[Individual] = field(default_factory=list, repr=False)
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    historical: list[Individual] = field(default_factory=list, repr=False)
    apdd_scores_by_schedule: dict[tuple[int, ...], dict[int, float]] = field(
        default_factory=dict, repr=False
    )
    offspring_history: list[tuple[tuple[int, ...], ...]] = field(
        default_factory=list, repr=False
    )


@dataclass(frozen=True)
# 组合搜索共享的 Diffusers Pipeline、FID 统计和 log-SNR 变换。
class SearchComponents:
    model: DDIMPipeline
    reference: ReferenceStatistics
    transform: LogSNRTransform
    feature_extractor: object

    def __post_init__(self) -> None:
        if self.reference.feature_extractor is None:
            self.reference.feature_extractor = self.feature_extractor
        elif self.reference.feature_extractor is not self.feature_extractor:
            raise ValueError(
                "SearchComponents must share the reference extractor instance"
            )
        if self.reference.feature_version != self.feature_extractor.version:
            raise ValueError("reference statistics use a different feature extractor")
        extractor_identity = feature_extractor_identity(self.feature_extractor)
        if self.reference.feature_extractor_identity is None:
            self.reference.feature_extractor_identity = extractor_identity
        if self.reference.feature_extractor_identity != extractor_identity:
            raise ValueError("reference statistics use different Inception weights")
        extractor_path = getattr(self.feature_extractor, "weights_path", None)
        if (
            self.reference.inception_weights_path is not None
            and extractor_path is not None
            and Path(self.reference.inception_weights_path).resolve()
            != Path(extractor_path).resolve()
        ):
            raise ValueError(
                "reference statistics use a different Inception weights path"
            )


def _tensor_sha256(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _array_sha256(named_arrays: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, raw in sorted(named_arrays):
        value = np.ascontiguousarray(raw)
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _compatibility(
    settings: SearchSettings, components: SearchComponents
) -> dict[str, Any]:
    ignored = {"experiment_name", "results_root", "resume"}
    settings_values = {
        item.name: getattr(settings, item.name)
        for item in fields(settings)
        if item.name not in ignored
    }
    reference = components.reference
    reference_sha256 = _array_sha256(
        (
            ("mean", reference.mean),
            ("covariance", reference.covariance),
            ("count", np.asarray(reference.count, dtype=np.int64)),
            ("feature_version", np.asarray(reference.feature_version)),
        )
    )
    pipeline = components.model
    scheduler_config = {
        name: value
        for name, value in dict(pipeline.scheduler.config).items()
        if not name.startswith("_")
    }
    sample_size = pipeline.unet.config.sample_size
    spatial_shape = (
        (int(sample_size), int(sample_size))
        if isinstance(sample_size, int)
        else tuple(int(size) for size in sample_size)
    )
    components_values = {
        "algorithm_version": "apdd-evolution-framework-v2-x0-clip-warmup-a",
        "diffusers_version": diffusers.__version__,
        "scheduler_class": type(pipeline.scheduler).__name__,
        "unet_sha256": _tensor_sha256(pipeline.unet.state_dict().items()),
        "scheduler_config": scheduler_config,
        "alpha_table_sha256": _tensor_sha256(
            (("alphas_cumprod", pipeline.scheduler.alphas_cumprod.double()),)
        ),
        "reference_sha256": reference_sha256,
        "transform_sha256": _array_sha256(
            (
                ("times", components.transform.times),
                ("log_snr", components.transform.log_snr),
            )
        ),
        "prediction_type": str(pipeline.scheduler.config.prediction_type),
        "clip_sample": bool(pipeline.scheduler.config.clip_sample),
        "thresholding": bool(pipeline.scheduler.config.thresholding),
        "set_alpha_to_one": bool(pipeline.scheduler.config.set_alpha_to_one),
        "sample_shape": (int(pipeline.unet.config.in_channels),) + spatial_shape,
        "sample_dtype": str(next(pipeline.unet.parameters()).dtype),
        "inception_weights_identity": feature_extractor_identity(
            components.feature_extractor
        ),
    }
    schedule_max_time = int(round(float(components.transform.times[-1])))
    if schedule_max_time != 1000:
        components_values["schedule_max_time"] = schedule_max_time
        components_values["schedule_parameterization"] = (
            "boundary-node-minus-one-v2"
        )
    return {"settings": settings_values, "components": components_values}


def _compatibility_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and bool(np.array_equal(left, right))
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(
                _compatibility_values_equal(left[key], right[key]) for key in left
            )
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _compatibility_values_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    return bool(left == right)


def _compatibility_errors(
    checkpoint: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    for section in ("settings", "components"):
        saved = checkpoint.get(section, {})
        expected = current[section]
        for name in sorted(set(saved) | set(expected)):
            if name not in saved or name not in expected:
                errors.append(f"{section}.{name}: checkpoint schema differs")
                continue
            if section == "settings" and name == "max_generations":
                if int(expected[name]) < int(saved[name]):
                    errors.append(
                        f"{section}.{name}: checkpoint={saved[name]!r}, "
                        f"current={expected[name]!r}"
                    )
            elif not _compatibility_values_equal(expected[name], saved[name]):
                errors.append(
                    f"{section}.{name}: checkpoint={saved[name]!r}, "
                    f"current={expected[name]!r}"
                )
    return errors


def _validate_settings(settings: SearchSettings) -> None:
    name = settings.experiment_name
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("experiment_name must be one plain directory name")
    if settings.k_min < 1 or settings.k_max <= settings.k_min:
        raise ValueError("search requires 1 <= k_min < k_max")
    if settings.population_size < 1:
        raise ValueError("population_size must be positive")
    if settings.warmup_generations < 1 or settings.max_generations < 1:
        raise ValueError("Warmup and Evolution iterations must be positive")
    if settings.per_k_archive_size < 1:
        raise ValueError("per-K archive capacity must be positive")


def _rng_payload(state: SearchState) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "numpy_generator": deepcopy(state.rng.bit_generator.state),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device=0)
        if torch.cuda.is_available()
        else None,
    }


def save_checkpoint(path: str | Path, state: SearchState) -> None:
    """原子保存完整迭代边界。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    log_sizes = {
        name: (target.parent / name).stat().st_size
        if (target.parent / name).is_file()
        else 0
        for name in ("progress.txt", "schedules.txt")
    }
    torch.save(
        {"state": state, "rng": _rng_payload(state), "log_sizes": log_sizes},
        temporary,
    )
    temporary.replace(target)


def load_checkpoint(
    path: str | Path,
    expected_compatibility: dict[str, Any] | None = None,
) -> SearchState:
    """读取 checkpoint，并恢复搜索使用的全部随机数状态。"""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state: SearchState = payload["state"]
    if expected_compatibility is not None:
        errors = _compatibility_errors(state.compatibility, expected_compatibility)
        if errors:
            raise ValueError("incompatible resume: " + "; ".join(errors))
    for name, committed_size in payload["log_sizes"].items():
        log_path = Path(path).parent / name
        if log_path.exists():
            with log_path.open("r+b") as stream:
                stream.truncate(committed_size)
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng["torch_cuda"] is not None:
        cuda_rng = rng["torch_cuda"]
        if isinstance(cuda_rng, (list, tuple)):
            cuda_rng = cuda_rng[0]
        torch.cuda.set_rng_state(cuda_rng, device=0)
    generator = np.random.default_rng()
    generator.bit_generator.state = rng["numpy_generator"]
    state.rng = generator
    return state


def _output_dir(settings: SearchSettings) -> Path:
    return Path(settings.results_root) / settings.experiment_name


def _write_config(path: Path, settings: SearchSettings) -> None:
    path.write_text(
        "\n".join(
            f"{item.name}={getattr(settings, item.name)!r}" for item in fields(settings)
        )
        + "\n",
        encoding="utf-8",
    )


def load_or_initialize(
    settings: SearchSettings,
    components: SearchComponents,
    cuda_device: int | None = None,
) -> SearchState:
    """恢复完整迭代快照，或创建 Warmup 第一次迭代的 C=100。"""

    _validate_settings(settings)
    initialize_runtime(
        settings.seed,
        deterministic=settings.deterministic,
        allow_tf32=settings.allow_tf32,
        cuda_device=cuda_device,
    )
    compatibility = _compatibility(settings, components)
    output = _output_dir(settings)
    checkpoint = output / "checkpoint.pt"
    if output.exists():
        if not settings.resume:
            raise FileExistsError(
                f"result {output} already exists; choose a new experiment_name"
            )
        state = load_checkpoint(checkpoint, compatibility)
        state.compatibility = compatibility
        return state

    output.mkdir(parents=True, exist_ok=False)
    (output / "samples").mkdir()
    (output / "progress.txt").touch()
    (output / "schedules.txt").touch()
    _write_config(output / "config.txt", settings)
    generator = np.random.default_rng(settings.seed)
    schedules: list[Schedule] = []
    seen: set[tuple[int, ...]] = set()
    while len(schedules) < settings.population_size:
        schedule = random_immigrant(
            settings.k_max,
            components.transform,
            generator,
            settings.minimum_gap,
        )
        if schedule.key in seen:
            continue
        seen.add(schedule.key)
        schedules.append(schedule)
    state = SearchState(
        stage="warmup",
        generation=0,
        population=[Individual(schedule=schedule, operator="random") for schedule in schedules],
        pareto=[],
        breeding={},
        evaluations={},
        compatibility=compatibility,
        rng=generator,
    )
    save_checkpoint(checkpoint, state)
    return state


def _emit(
    settings: SearchSettings,
    line: str,
    *,
    schedule_line: bool = False,
    console: bool = True,
) -> None:
    if console:
        print(line, flush=True)
    output = _output_dir(settings)
    with (output / "progress.txt").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    if schedule_line:
        with (output / "schedules.txt").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def _protocol_options(settings: SearchSettings, protocol: str) -> EvaluationOptions:
    samples = {
        "search": settings.search_samples,
        "archive": settings.archive_samples,
        "final": settings.final_samples,
    }[protocol]
    return EvaluationOptions(
        samples=samples,
        batch_size=settings.batch_size,
        compute_kid=False,
        compute_precision_recall=False,
        compute_is=False,
    )


def _evaluate_candidates(
    state: SearchState,
    candidates: Sequence[Individual],
    settings: SearchSettings,
    components: SearchComponents,
    protocol: str,
    stage_label: str,
    parallel_context: ParallelContext | None = None,
) -> list[Individual]:
    """评价候选，并把结果写入同一个 Individual 的协议字段。"""

    options = _protocol_options(settings, protocol)
    seeds = seed_stream(settings.seed, protocol, options.samples)
    unique: list[Individual] = []
    scheduled: set[tuple[int, ...]] = set()
    for candidate in candidates:
        key = candidate.schedule.key
        if (protocol, key) not in state.evaluations and key not in scheduled:
            unique.append(candidate)
            scheduled.add(key)

    if unique:
        tasks = tuple(
            (task_id, candidate.schedule.nodes)
            for task_id, candidate in enumerate(unique)
        )
        completed_count = 0

        def report_completed(
            rank: int, task_id: int, result: EvaluationResult
        ) -> None:
            nonlocal completed_count
            completed_count += 1
            schedule = unique[task_id].schedule
            print(
                f"[{stage_label}] [Done {completed_count}/{len(unique)}] "
                f"GPU={rank} Task={task_id} K={schedule.k} "
                f"Schedule={list(schedule.nodes)} "
                f"FID={result.fid:.6f} Time={result.seconds:.2f}s",
                flush=True,
            )
        results = list(
            evaluate_schedule_batch(
                parallel_context,
                tasks,
                rank0_components=components,
                protocol_samples=options.samples,
                seeds=seeds,
                options=options,
                protocol=protocol,
                on_task_complete=report_completed,
            )
        )
        result_ids = [task_id for task_id, _result in results]
        expected_ids = set(range(len(unique)))
        if (
            len(result_ids) != len(unique)
            or len(set(result_ids)) != len(result_ids)
            or set(result_ids) != expected_ids
        ):
            raise ValueError("parallel evaluator returned invalid task IDs")
        results_by_id = dict(results)

        for task_id, candidate in enumerate(unique):
            result = results_by_id[task_id]
            schedule = candidate.schedule
            if result.schedule != schedule.nodes or result.protocol != protocol:
                raise ValueError("parallel evaluator returned a mismatched result")
            if result.samples != options.samples or result.nfe != schedule.k:
                raise ValueError("parallel evaluator returned an invalid evaluation")

        for task_id, candidate in enumerate(unique):
            result = results_by_id[task_id]
            schedule = candidate.schedule
            _emit(
                settings,
                f"[{stage_label}] [Eval {task_id + 1}/{len(unique)}] "
                f"K={schedule.k} Schedule={list(schedule.nodes)}",
                schedule_line=True,
                console=False,
            )
            state.evaluations[(protocol, schedule.key)] = result
            setattr(candidate, f"{protocol}_fid", float(result.fid))
            candidate.nfe = int(result.nfe)
            _emit(
                settings,
                f"FID={result.fid:.6f} Time={result.seconds:.2f}s",
                console=False,
            )
            if protocol == "final":
                save_checkpoint(_output_dir(settings) / "checkpoint.pt", state)

    evaluated: list[Individual] = []
    returned: set[tuple[int, ...]] = set()
    for candidate in candidates:
        key = candidate.schedule.key
        if key in returned:
            continue
        result = state.evaluations.get((protocol, key))
        if result is None:
            continue
        setattr(candidate, f"{protocol}_fid", float(result.fid))
        candidate.nfe = int(result.nfe)
        evaluated.append(candidate)
        returned.add(key)
    return evaluated


def _seen_search_schedules(state: SearchState) -> set[tuple[int, ...]]:
    return {nodes for protocol, nodes in state.evaluations if protocol == "search"}


def _apdd_scores(
    state: SearchState,
    schedule: Schedule,
    settings: SearchSettings,
    components: SearchComponents,
) -> dict[int, float]:
    stored = state.apdd_scores_by_schedule.get(schedule.key)
    if stored is not None:
        return stored
    positions = range(1, len(schedule.nodes) - 1)
    if settings.deletion_score_mode == "random":
        scores = {position: float(state.rng.random()) for position in positions}
    elif settings.deletion_score_mode == "gap":
        coordinates = components.transform.to_coordinate(
            np.asarray(schedule.nodes, dtype=np.float64)
        )
        scores = {
            position: float(
                2.0
                * abs(coordinates[position - 1] - coordinates[position])
                * abs(coordinates[position] - coordinates[position + 1])
            )
            for position in positions
        }
    else:
        seeds = seed_stream(settings.seed, "apdd-calibration", settings.apdd_samples)
        _samples, measured_nfe, _timesteps, trajectory = sample_schedule(
            components.model,
            schedule,
            seeds,
            return_trajectory=True,
            require_grad=settings.deletion_score_mode == "adjoint",
        )
        if measured_nfe != schedule.k or trajectory is None:
            raise ValueError("APDD calibration did not return the expected trajectory")
        scores = deletion_scores(
            trajectory,
            settings.deletion_score_mode,
            settings.apdd_probes,
            settings.apdd_probe_seed,
        )
    state.apdd_scores_by_schedule[schedule.key] = scores
    return scores


def _candidate(
    schedule: Schedule, operator: str, parents: Iterable[Individual]
) -> Individual:
    return Individual(
        schedule=schedule,
        parents=tuple(parent.schedule.key for parent in parents),
        operator=operator,
    )


def _add_if_new(
    children: list[Individual], candidate: Individual, seen: set[tuple[int, ...]]
) -> bool:
    if candidate.schedule.key in seen:
        return False
    seen.add(candidate.schedule.key)
    children.append(candidate)
    return True


def _point_child(
    state: SearchState,
    parent: Individual,
    settings: SearchSettings,
    components: SearchComponents,
    seen: set[tuple[int, ...]],
) -> Individual | None:
    schedule = apdd_point_mutate(
        parent.schedule,
        _apdd_scores(state, parent.schedule, settings, components),
        components.transform,
        state.rng,
        settings.minimum_gap,
        seen,
    )
    return None if schedule is None else _candidate(schedule, "point", (parent,))


def _deletion_child(
    state: SearchState,
    parent: Individual,
    settings: SearchSettings,
    components: SearchComponents,
    seen: set[tuple[int, ...]],
) -> Individual | None:
    schedule = apdd_delete(
        parent.schedule,
        _apdd_scores(state, parent.schedule, settings, components),
        settings.deletion_strategy,
        seen,
        state.rng,
        settings.k_min,
        settings.top_r,
        settings.removal_temperature,
    )
    return None if schedule is None else _candidate(schedule, "deletion", (parent,))


def _random_child(
    state: SearchState, settings: SearchSettings, components: SearchComponents
) -> Individual:
    schedule = random_immigrant(
        settings.k_max,
        components.transform,
        state.rng,
        settings.minimum_gap,
    )
    return _candidate(schedule, "random", ())


def _offspring_batch(
    state: SearchState,
    settings: SearchSettings,
    components: SearchComponents,
    *,
    warmup: bool,
) -> list[Individual]:
    """从代初冻结的 E 产生 B，并用随机 K_max 个体 C 补到100。"""

    layers = {k: list(items) for k, items in state.breeding.items() if items}
    seen = _seen_search_schedules(state)
    children: list[Individual] = []
    active_k = sorted(layers, reverse=True)
    if warmup:
        point_k = active_k
        cross_repetitions = 2
    else:
        point_count = (len(active_k) + 1) // 2
        point_k = (
            [int(value) for value in state.rng.choice(active_k, point_count, replace=False)]
            if point_count
            else []
        )
        cross_repetitions = 1

    for k in point_k:
        parent = layers[k][0]
        child = _point_child(state, parent, settings, components, seen)
        if child is not None:
            _add_if_new(children, child, seen)

    for k in active_k:
        if k <= settings.k_min:
            continue
        parent = layers[k][0]
        child = _deletion_child(state, parent, settings, components, seen)
        if child is not None:
            _add_if_new(children, child, seen)

    for k in active_k:
        layer = layers[k]
        if len(layer) < 2:
            continue
        for _ in range(cross_repetitions):
            pairs = [
                (left, right)
                for left in range(len(layer))
                for right in range(left + 1, len(layer))
            ]
            state.rng.shuffle(pairs)
            for left_index, right_index in pairs:
                first, second = layer[left_index], layer[right_index]
                crossed = segment_crossover(
                    first.schedule,
                    second.schedule,
                    state.rng,
                    settings.minimum_gap,
                )
                if len(crossed) != 2 or any(
                    schedule.key in seen for schedule in crossed
                ):
                    continue
                for schedule in crossed:
                    _add_if_new(
                        children,
                        _candidate(schedule, "crossover", (first, second)),
                        seen,
                    )
                if crossed:
                    break

    while len(children) < settings.population_size:
        _add_if_new(children, _random_child(state, settings, components), seen)
    return children[: settings.population_size]


def _flatten_breeding(state: SearchState) -> list[Individual]:
    return [item for layer in state.breeding.values() for item in layer]


def _update_breeding(
    state: SearchState,
    candidates: Sequence[Individual],
    components: SearchComponents,
    capacity: int | None,
) -> None:
    archive = BreedingArchives(
        capacity=capacity,
        transform=components.transform,
        by_k={k: list(items) for k, items in state.breeding.items()},
    )
    archive.update(candidates, "archive")
    state.breeding = archive.by_k


def run_warmup(
    state: SearchState,
    settings: SearchSettings,
    components: SearchComponents,
    parallel_context: ParallelContext | None = None,
) -> None:
    """运行5次 Warmup：Search top20、Archive top10，并维护 A 与 E。"""

    while state.generation < settings.warmup_generations:
        iteration = state.generation + 1
        if state.generation == 0:
            candidates = state.population
        else:
            candidates = _offspring_batch(
                state, settings, components, warmup=True
            )
        searched = _evaluate_candidates(
            state,
            candidates,
            settings,
            components,
            "search",
            f"Warmup {iteration}/{settings.warmup_generations} Search",
            parallel_context=parallel_context,
        )
        state.population = searched
        search_top20 = (
            select_fixed_k_by_fid(searched, 20, "search")
            if state.generation == 0
            else environmental_selection(searched, 20, "search")
        )
        archive_top20 = _evaluate_candidates(
            state,
            search_top20,
            settings,
            components,
            "archive",
            f"Warmup {iteration}/{settings.warmup_generations} Archive",
            parallel_context=parallel_context,
        )
        current_top10 = (
            select_fixed_k_by_fid(archive_top20, 10, "archive")
            if state.generation == 0
            else environmental_selection(archive_top20, 10, "archive")
        )
        previous_a = list(state.pareto)
        state.pareto = (
            current_top10
            if state.generation == 0
            else environmental_selection(
                [*previous_a, *current_top10], 10, "archive"
            )
        )
        _update_breeding(state, state.pareto, components, capacity=None)
        state.generation += 1
        save_checkpoint(_output_dir(settings) / "checkpoint.pt", state)

    # Warmup 第5次迭代更新后的 A 作为 Evolution 的初始 A；
    # E 从进入 Evolution 起再按每个 K 最多3条进行压缩。
    _update_breeding(
        state,
        [*_flatten_breeding(state), *state.pareto],
        components,
        capacity=settings.per_k_archive_size,
    )
    state.stage = "multiobjective"
    state.generation = 0
    save_checkpoint(_output_dir(settings) / "checkpoint.pt", state)


def _run_evolution_iteration(
    state: SearchState,
    settings: SearchSettings,
    components: SearchComponents,
    parallel_context: ParallelContext | None = None,
) -> None:
    offspring = _offspring_batch(state, settings, components, warmup=False)
    iteration = state.generation + 1
    searched = _evaluate_candidates(
        state,
        offspring,
        settings,
        components,
        "search",
        f"Evolution {iteration}/{settings.max_generations} Search",
        parallel_context=parallel_context,
    )
    state.population = searched
    state.offspring_history.append(
        tuple(individual.schedule.key for individual in searched)
    )
    search_fronts = non_dominated_sort(searched, "search")
    promoted = search_fronts[0] if search_fronts else []
    archive_offspring = _evaluate_candidates(
        state,
        promoted,
        settings,
        components,
        "archive",
        f"Evolution {iteration}/{settings.max_generations} Archive",
        parallel_context=parallel_context,
    )
    archive_fronts = non_dominated_sort(
        [*state.pareto, *archive_offspring], "archive"
    )
    state.pareto = archive_fronts[0] if archive_fronts else []
    _update_breeding(
        state,
        state.pareto,
        components,
        capacity=settings.per_k_archive_size,
    )
    state.generation += 1
    save_checkpoint(_output_dir(settings) / "checkpoint.pt", state)


def run_multiobjective(
    state: SearchState,
    settings: SearchSettings,
    components: SearchComponents,
    parallel_context: ParallelContext | None = None,
) -> None:
    """运行45次多K Evolution，并仅在完整迭代末提交状态。"""

    while state.generation < settings.max_generations:
        _run_evolution_iteration(
            state,
            settings,
            components,
            parallel_context=parallel_context,
        )
    state.stage = "finalize"
    save_checkpoint(_output_dir(settings) / "checkpoint.pt", state)


def finalize(
    state: SearchState,
    settings: SearchSettings,
    components: SearchComponents,
    parallel_context: ParallelContext | None = None,
) -> None:
    """冻结最终 A，只执行一次 final 协议并重建最终 Pareto。"""

    final_items = _evaluate_candidates(
        state,
        state.pareto,
        settings,
        components,
        "final",
        "Final",
        parallel_context=parallel_context,
    )
    final_fronts = non_dominated_sort(final_items, "final")
    state.pareto = final_fronts[0] if final_fronts else []
    state.stage = "completed"
    save_checkpoint(_output_dir(settings) / "checkpoint.pt", state)


def _evaluation_rows(state: SearchState) -> list[dict[str, object]]:
    rows = []
    for (protocol, nodes), result in sorted(state.evaluations.items()):
        rows.append(
            {
                "protocol": protocol,
                "schedule": json.dumps(nodes),
                "k": len(nodes) - 1,
                "samples": result.samples,
                "fid": result.fid,
                "nfe": result.nfe,
                "seconds": result.seconds,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export_results(
    state: SearchState, settings: SearchSettings, components: SearchComponents
) -> None:
    """写出配置、全部评价、最终前沿和一批可审计样本。"""

    output = _output_dir(settings)
    _write_config(output / "config.txt", settings)
    _write_csv(
        output / "evaluations.csv",
        _evaluation_rows(state),
        ["protocol", "schedule", "k", "samples", "fid", "nfe", "seconds"],
    )
    ordered = sorted(
        state.pareto,
        key=lambda item: (
            int(item.nfe),
            float(item.final_fid),
            item.schedule.key,
        ),
    )
    pareto_rows = [
        {
            "protocol": "final",
            "schedule": json.dumps(item.schedule.nodes),
            "k": item.schedule.k,
            "fid": item.final_fid,
            "nfe": item.nfe,
            "operator": item.operator,
            "parents": json.dumps(item.parents),
        }
        for item in ordered
    ]
    _write_csv(
        output / "pareto.csv",
        pareto_rows,
        ["protocol", "schedule", "k", "fid", "nfe", "operator", "parents"],
    )
    samples_dir = output / "samples"
    samples_dir.mkdir(exist_ok=True)
    for stale in samples_dir.glob("pareto_*.pt"):
        stale.unlink()
    sample_count = min(settings.final_samples, settings.batch_size)
    sample_seeds = seed_stream(settings.seed, "final", settings.final_samples)[
        :sample_count
    ]
    for index, item in enumerate(ordered, start=1):
        samples, measured_nfe, _timesteps, _trajectory = sample_schedule(
            components.model,
            item.schedule,
            sample_seeds,
        )
        if measured_nfe != item.schedule.k:
            raise ValueError("sample artifact NFE does not equal Schedule K")
        torch.save(
            {
                "schedule": item.schedule.nodes,
                "seeds": sample_seeds,
                "nfe": measured_nfe,
                "samples": samples.detach().cpu(),
            },
            samples_dir / f"pareto_{index:03d}.pt",
        )


def run_search(
    settings: SearchSettings,
    components: SearchComponents,
    parallel_context: ParallelContext | None = None,
    cuda_device: int | None = None,
) -> SearchState:
    """按5次Warmup、45次Evolution和一次Final顺序运行APDD。"""

    state = load_or_initialize(
        settings,
        components,
        cuda_device=cuda_device,
    )
    if state.stage == "completed":
        export_results(state, settings, components)
        return state
    if state.stage == "warmup":
        run_warmup(
            state,
            settings,
            components,
            parallel_context=parallel_context,
        )
    if state.stage == "multiobjective":
        run_multiobjective(
            state,
            settings,
            components,
            parallel_context=parallel_context,
        )
    if state.stage == "finalize":
        finalize(
            state,
            settings,
            components,
            parallel_context=parallel_context,
        )
    export_results(state, settings, components)
    return state
