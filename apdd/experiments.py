"""固定 Schedule 对比评价与 APDD 消融实验。"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .diffusion import sample_schedule
from .metrics import EvaluationOptions, EvaluationResult, evaluate_schedule, seed_stream
from .schedule import Schedule, make_schedule
from .score import deletion_scores
from .search import SearchComponents, SearchSettings, run_search

# 该模块只放科研对照与消融入口，主 APDD 搜索仍由 search.run_search 实现。
EvaluateFunction = Callable[
    [object, Schedule, object, tuple[int, ...], EvaluationOptions, str],
    EvaluationResult,
]

FIXED_RESULT_COLUMNS = (
    "method",
    "k",
    "schedule",
    "fid",
    "nfe",
    "samples",
    "seconds",
    "protocol",
)


def _fixed_schedule(nodes: Sequence[int]) -> Schedule:
    """Validate explicit APDD nodes without relying on ``Schedule`` asserts."""

    # 外部论文 Schedule 必须先转换为 APDD 的完整 K+1 整数节点语义。
    try:
        values = tuple(nodes)
    except TypeError as error:
        raise ValueError("fixed schedule nodes must be an integer sequence") from error
    if len(values) < 2:
        raise ValueError("fixed schedules require at least endpoints 1000 and 0")
    if any(type(node) is not int for node in values):
        raise ValueError("fixed schedule nodes must be builtin integers, not bools")
    if values[0] != 1000 or values[-1] != 0:
        raise ValueError("fixed schedules must start at 1000 and end at 0")
    if any(values[index] <= values[index + 1] for index in range(len(values) - 1)):
        raise ValueError("fixed schedule nodes must be strictly descending")
    return Schedule(values)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    columns = (
        "method",
        "candidate",
        "generation",
        "schedule",
        "k",
        "fid",
        "nfe",
        "protocol",
        "samples",
        "seconds",
        "operator",
        "union_member",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _evaluate_candidate(
    components: SearchComponents,
    schedule: Schedule,
    *,
    method: str,
    candidate: int,
    generation: int,
    operator: str,
    samples: int,
    batch_size: int,
    seed: int,
    evaluate_fn: EvaluateFunction,
    protocol: str = "search",
    progress_total: int | None = None,
) -> dict[str, object]:
    """Evaluate one greedy-ablation schedule with its ordinary protocol."""

    # 同一 protocol 下所有方法使用完全相同的公共种子流。
    options = EvaluationOptions(samples=samples, batch_size=batch_size)
    seeds = seed_stream(seed, protocol, samples)
    progress = (
        f"[Eval {candidate + 1}]"
        if progress_total is None
        else f"[Eval {candidate + 1}/{progress_total}]"
    )
    print(
        f"[{method}] [{protocol}] {progress} "
        f"K={schedule.k} Samples={samples} Schedule={list(schedule.nodes)}",
        flush=True,
    )
    if evaluate_fn is evaluate_schedule:
        result = evaluate_fn(
            components.model,
            schedule,
            components.reference,
            seeds,
            options,
            protocol,
            feature_extractor=components.feature_extractor,
        )
    else:
        result = evaluate_fn(
            components.model,
            schedule,
            components.reference,
            seeds,
            options,
            protocol,
        )
    print(f"FID={result.fid:.6f} Time={result.seconds:.2f}s", flush=True)
    if result.nfe != schedule.k:
        raise ValueError("ablation evaluation NFE does not equal Schedule K")
    return {
        "method": method,
        "candidate": candidate,
        "generation": generation,
        "schedule": json.dumps(schedule.nodes),
        "k": schedule.k,
        "fid": result.fid,
        "nfe": result.nfe,
        "protocol": result.protocol,
        "samples": result.samples,
        "seconds": result.seconds,
        "operator": operator,
        "union_member": False,
    }


def evaluate_fixed_schedules(
    schedules: Mapping[str, Sequence[int]],
    *,
    components: SearchComponents,
    output_path: str | Path,
    samples: int,
    batch_size: int,
    seed: int,
    evaluate_fn: EvaluateFunction = evaluate_schedule,
) -> Path:
    """使用统一 final 协议评价外部算法给出的固定 Schedule。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # AYS/ATS 等外部方法在这里仅提供节点序列，模型、DDIM 和 FID 管线保持统一。
    resolved = [(method, _fixed_schedule(nodes)) for method, nodes in schedules.items()]
    options = EvaluationOptions(samples=samples, batch_size=batch_size)
    # 固定 Schedule 对比直接使用论文最终保真度和共享 final 种子。
    seeds = seed_stream(seed, "final", samples)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIXED_RESULT_COLUMNS)
        writer.writeheader()
        stream.flush()
        os.fsync(stream.fileno())
        for method, schedule in resolved:
            print(
                f"[{method}] [final] K={schedule.k} Samples={samples} "
                f"Schedule={list(schedule.nodes)}",
                flush=True,
            )
            if evaluate_fn is evaluate_schedule:
                result = evaluate_fn(
                    components.model,
                    schedule,
                    components.reference,
                    seeds,
                    options,
                    "final",
                    feature_extractor=components.feature_extractor,
                )
            else:
                result = evaluate_fn(
                    components.model,
                    schedule,
                    components.reference,
                    seeds,
                    options,
                    "final",
                )
            if result.nfe != schedule.k:
                raise ValueError(
                    "fixed baseline evaluation NFE does not equal Schedule K"
                )
            if result.protocol != "final" or result.samples != samples:
                raise ValueError(
                    "fixed baseline evaluator did not return final-fidelity data"
                )
            print(
                f"FID={result.fid:.6f} NFE={result.nfe} Time={result.seconds:.2f}s",
                flush=True,
            )
            writer.writerow(
                {
                    "method": method,
                    "k": schedule.k,
                    "schedule": json.dumps(schedule.nodes),
                    "fid": result.fid,
                    "nfe": result.nfe,
                    "samples": result.samples,
                    "seconds": result.seconds,
                    "protocol": result.protocol,
                }
            )
            stream.flush()
            os.fsync(stream.fileno())
    return path


def _score_deletions(
    components: SearchComponents,
    schedule: Schedule,
    *,
    mode: str,
    seeds: tuple[int, ...],
    probes: int,
    probe_seed: int,
    rng: np.random.Generator,
) -> dict[int, float]:
    """为单路径压缩计算指定的节点删除分数。"""

    # 五种删除依据共用节点位置定义，便于构造 random/gap/local/Lipschitz/adjoint 消融。
    positions = range(1, len(schedule.nodes) - 1)
    if mode == "random":
        return {position: float(rng.random()) for position in positions}
    if mode == "gap":
        coordinates = components.transform.to_coordinate(np.asarray(schedule.nodes))
        return {
            position: float(
                2.0
                * abs(coordinates[position - 1] - coordinates[position])
                * abs(coordinates[position] - coordinates[position + 1])
            )
            for position in positions
        }
    if mode not in {"local", "lipschitz", "adjoint"}:
        raise ValueError("score_mode must be random, gap, local, lipschitz, or adjoint")
    # Diffusers Pipeline 与 FID 评价共用同一显式 source→target DDIM 路径。
    _, measured_nfe, _model_timesteps, trajectory = sample_schedule(
        components.model,
        schedule,
        seeds,
        return_trajectory=True,
        require_grad=mode == "adjoint",
    )
    if trajectory is None or measured_nfe != schedule.k:
        raise RuntimeError("greedy compression did not obtain a complete trajectory")
    return deletion_scores(trajectory, mode, probes, probe_seed)


def _run_greedy_lineage(
    method: str,
    *,
    components: SearchComponents,
    output_root: str | Path,
    k_values: Sequence[int],
    samples: int,
    batch_size: int,
    seed: int,
    apdd_samples: int,
    apdd_probes: int,
    score_mode: str,
    minimum_gap: int = 1,
    evaluate_fn: EvaluateFunction = evaluate_schedule,
) -> Path:
    values = tuple(sorted({int(value) for value in k_values}))
    if not values or values[0] < 1 or values[-1] > 1000:
        raise ValueError("k_values must contain integers in [1, 1000]")
    rng = np.random.default_rng(seed + 50_000)
    # 单谱系基线从 K_max 的 uniform-logSNR 出发，不使用种群、交叉或 NSGA-II。
    current = make_schedule(
        "uniform_logsnr",
        max(values),
        minimum_gap,
        components.transform,
    )
    calibration_seeds = seed_stream(seed, f"{method}-calibration", apdd_samples)
    rows: list[dict[str, object]] = []
    generation = 0
    while current.k >= min(values):
        if current.k in values:
            rows.append(
                _evaluate_candidate(
                    components,
                    current,
                    method=method,
                    candidate=len(rows),
                    generation=generation,
                    operator="greedy_single_lineage",
                    samples=samples,
                    batch_size=batch_size,
                    seed=seed,
                    evaluate_fn=evaluate_fn,
                    progress_total=len(values),
                )
            )
        if current.k == min(values):
            break
        print(f"[{method}] Compress step {current.k}->{current.k - 1}", flush=True)
        scores = _score_deletions(
            components,
            current,
            mode=score_mode,
            seeds=calibration_seeds,
            probes=apdd_probes,
            probe_seed=seed + 73,
            rng=rng,
        )
        if not scores:
            raise RuntimeError("greedy compression found no legal internal node")
        # 每一层只删除当前分数最低的一个节点，形成唯一 K→K-1 路径。
        removed = min(scores, key=lambda position: (scores[position], position))
        nodes = list(current.nodes)
        nodes.pop(removed)
        current = Schedule(tuple(nodes))
        generation += 1
    path = Path(output_root) / method / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_rows(path, rows)


def run_greedy_single_lineage(**kwargs) -> Path:
    """运行不使用种群的单谱系贪心压缩消融。"""

    return _run_greedy_lineage("greedy_single_lineage", **kwargs)


# 每个条目只覆盖完整 APDD 的一个设置，避免同时改变多个机制。
ABLATION_OVERRIDES: dict[str, dict[str, object]] = {
    "random_deletion": {"deletion_score_mode": "random"},
    "gap_deletion": {"deletion_score_mode": "gap"},
    "local_deletion": {"deletion_score_mode": "local"},
    "lipschitz_deletion": {"deletion_score_mode": "lipschitz"},
    "argmin_only": {"deletion_strategy": "argmin_only"},
    "top_r_softmax": {"deletion_strategy": "top_r_softmax"},
}

# 这两个名称分别是删除分数和删除策略实验中的完整方法对照，不改变 base settings。
CONTROL_ABLATIONS = {"full_adjoint_deletion", "unseen_first"}


def apply_ablation(settings: SearchSettings, name: str) -> SearchSettings:
    """仅改变一个机制并返回其余参数不变的消融配置。"""

    if name == "greedy_single_lineage":
        raise ValueError(
            "greedy_single_lineage uses its independent experiment function"
        )
    if name in CONTROL_ABLATIONS:
        return settings
    try:
        overrides = dict(ABLATION_OVERRIDES[name])
    except KeyError as error:
        raise ValueError(f"unknown ablation: {name}") from error
    return replace(settings, **overrides)


def _settings_ablation(name: str):
    return lambda settings: apply_ablation(settings, name)


ABLATIONS = {
    **{name: _settings_ablation(name) for name in ABLATION_OVERRIDES},
    **{name: _settings_ablation(name) for name in CONTROL_ABLATIONS},
    "greedy_single_lineage": run_greedy_single_lineage,
}


def run_ablation_suite(
    *,
    selected_names: Sequence[str],
    base_settings: SearchSettings,
    components: SearchComponents,
    output_root: str | Path,
    resume: bool = True,
) -> Path:
    """顺序运行选定的完整搜索消融和独立贪心单谱系。"""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    k_values = tuple(range(base_settings.k_min, base_settings.k_max + 1))
    total = len(selected_names)
    # greedy_single_lineage 走独立单路径；其余条目均运行一次完整搜索流程。
    for index, name in enumerate(selected_names, start=1):
        if name not in ABLATIONS:
            raise ValueError(f"unknown ablation: {name}")
        print(f"[Ablation {index}/{total}] START {name}", flush=True)
        if name == "greedy_single_lineage":
            path = run_greedy_single_lineage(
                components=components,
                output_root=root,
                k_values=k_values,
                samples=base_settings.search_samples,
                batch_size=base_settings.batch_size,
                seed=base_settings.seed,
                apdd_samples=base_settings.apdd_samples,
                apdd_probes=base_settings.apdd_probes,
                score_mode="adjoint",
                minimum_gap=base_settings.minimum_gap,
            )
            rows.append(
                {
                    "name": name,
                    "kind": "single_lineage",
                    "is_control": False,
                    "output": str(path),
                    "pareto_size": "",
                    "settings": "{}",
                }
            )
            print(f"[Ablation {index}/{total}] DONE {name} Results={path}", flush=True)
            continue
        # 每个消融使用独立实验目录，因此 checkpoint/resume 不会在不同机制间混用。
        settings = apply_ablation(base_settings, name)
        settings = replace(
            settings,
            experiment_name=name,
            results_root=str(root),
            resume=resume,
        )
        state = run_search(settings, components)
        rows.append(
            {
                "name": name,
                "kind": "control" if name in CONTROL_ABLATIONS else "search",
                "is_control": name in CONTROL_ABLATIONS,
                "output": str(root / name),
                "pareto_size": len(state.pareto),
                "settings": json.dumps(asdict(settings), sort_keys=True),
            }
        )
        print(
            f"[Ablation {index}/{total}] DONE {name} Pareto={len(state.pareto)}",
            flush=True,
        )
    target = root / "ablation_results.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "name",
                "kind",
                "is_control",
                "output",
                "pareto_size",
                "settings",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    return target
