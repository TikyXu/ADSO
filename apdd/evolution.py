"""APDD 演化个体、变异算子、NSGA-II 与分层繁殖档案。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from .schedule import LogSNRTransform, Schedule, repair_schedule, schedule_distance


@dataclass
class Individual:
    """保存同一 Schedule 在三种 FID 协议下的评价与谱系。"""

    schedule: Schedule
    search_fid: float | None = None
    archive_fid: float | None = None
    final_fid: float | None = None
    nfe: int | None = None
    parents: tuple[tuple[int, ...], ...] = ()
    operator: str = "initial"

    def fid_for(self, protocol: str) -> float | None:
        """返回指定协议的 FID，不在不同协议之间自动回退。"""

        return {
            "search": self.search_fid,
            "archive": self.archive_fid,
            "final": self.final_fid,
        }[protocol]


def _eligible(item: Individual, protocol: str) -> bool:
    fid = item.fid_for(protocol)
    return (
        fid is not None
        and item.nfe is not None
        and math.isfinite(float(fid))
        and int(item.nfe) >= 0
    )


def _objectives(item: Individual, protocol: str) -> tuple[float, float]:
    if not _eligible(item, protocol):
        raise ValueError(f"{protocol} comparison requires a completed evaluation")
    return float(item.nfe), float(item.fid_for(protocol))


@cache
def _schedule_tie_key(nodes: tuple[int, ...]) -> str:
    payload = {
        "context": {
            "model_version": "unspecified-model",
            "sampler_version": "ddim-eta0",
            "parameterization_version": "epsilon",
            "coordinate": "time-descending",
            "precision": 12,
        },
        "timesteps": [f"{float(node):.12f}" for node in nodes],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _eligible_items(
    items: Iterable[Individual], protocol: str
) -> list[Individual]:
    return [item for item in items if _eligible(item, protocol)]


def dominates(first: Individual, second: Individual, protocol: str) -> bool:
    """判断 first 是否在 NFE 与指定协议 FID 上支配 second。"""

    left = _objectives(first, protocol)
    right = _objectives(second, protocol)
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def _deduplicate(
    items: Iterable[Individual], protocol: str
) -> list[Individual]:
    best: dict[tuple[int, ...], Individual] = {}
    for item in items:
        if not _eligible(item, protocol):
            continue
        current = best.get(item.schedule.key)
        if current is None or _objectives(item, protocol) < _objectives(
            current, protocol
        ):
            best[item.schedule.key] = item
    return list(best.values())


def non_dominated_sort(
    candidates: Iterable[Individual], protocol: str
) -> list[list[Individual]]:
    """按指定 FID 协议执行确定性的非支配分层。"""

    population = _deduplicate(candidates, protocol)
    domination_count = [0] * len(population)
    dominated: list[list[int]] = [[] for _ in population]
    first_front: list[int] = []
    for left_index, left in enumerate(population):
        for right_index, right in enumerate(population):
            if left_index == right_index:
                continue
            if dominates(left, right, protocol):
                dominated[left_index].append(right_index)
            elif dominates(right, left, protocol):
                domination_count[left_index] += 1
        if domination_count[left_index] == 0:
            first_front.append(left_index)

    fronts: list[list[int]] = []
    current = first_front
    while current:
        fronts.append(current)
        following: list[int] = []
        for left_index in current:
            for right_index in dominated[left_index]:
                domination_count[right_index] -= 1
                if domination_count[right_index] == 0:
                    following.append(right_index)
        current = following
    return [[population[index] for index in front] for front in fronts]


def crowding_distances(
    front: Iterable[Individual], protocol: str
) -> dict[tuple[int, ...], float]:
    """返回指定协议下的 NSGA-II 归一化拥挤距离。"""

    items = list(front)
    distances = {item.schedule.key: 0.0 for item in items}
    if len(items) <= 2:
        return {item.schedule.key: math.inf for item in items}
    for objective_index in range(2):
        ordered = sorted(
            items,
            key=lambda item: (
                _objectives(item, protocol)[objective_index],
                _schedule_tie_key(item.schedule.key),
            ),
        )
        distances[ordered[0].schedule.key] = math.inf
        distances[ordered[-1].schedule.key] = math.inf
        low = _objectives(ordered[0], protocol)[objective_index]
        high = _objectives(ordered[-1], protocol)[objective_index]
        if high == low:
            continue
        for index in range(1, len(ordered) - 1):
            key = ordered[index].schedule.key
            if math.isinf(distances[key]):
                continue
            previous_value = _objectives(ordered[index - 1], protocol)[
                objective_index
            ]
            next_value = _objectives(ordered[index + 1], protocol)[objective_index]
            distances[key] += (next_value - previous_value) / (high - low)
    return distances


def environmental_selection(
    candidates: Sequence[Individual], size: int, protocol: str
) -> list[Individual]:
    """按指定协议执行 NSGA-II rank 与拥挤距离选择。"""

    unique = _deduplicate(candidates, protocol)
    selected: list[Individual] = []
    for front in non_dominated_sort(unique, protocol):
        remaining = size - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front)
            continue
        distances = crowding_distances(front, protocol)
        ordered = sorted(
            front,
            key=lambda item: (
                -distances[item.schedule.key],
                float(item.fid_for(protocol)),
                int(item.nfe),
                _schedule_tie_key(item.schedule.key),
            ),
        )
        selected.extend(ordered[:remaining])
    return selected


def select_fixed_k_by_fid(
    candidates: Sequence[Individual], size: int, protocol: str
) -> list[Individual]:
    """固定 K 时直接按指定协议 FID 选择前 size 个个体。"""

    unique = _deduplicate(candidates, protocol)
    return sorted(
        unique,
        key=lambda item: (
            float(item.fid_for(protocol)),
            _schedule_tie_key(item.schedule.key),
        ),
    )[:size]


def apdd_point_mutate(
    parent: Schedule,
    scores: Mapping[int, float],
    transform: LogSNRTransform,
    rng: np.random.Generator,
    minimum_gap: int,
    seen: set[tuple[int, ...]],
) -> Schedule | None:
    """按 APDD 低分顺序选择一个内部节点，并只改变该节点。"""

    ranked = sorted(
        (
            (int(position), float(score))
            for position, score in scores.items()
            if 0 < int(position) < len(parent.nodes) - 1 and np.isfinite(score)
        ),
        key=lambda item: (item[1], item[0]),
    )
    gap = max(1, minimum_gap)
    for position, _score in ranked:
        upper = parent.nodes[position - 1] - gap
        lower = parent.nodes[position + 1] + gap
        legal = [
            value
            for value in range(lower, upper + 1)
            if value != parent.nodes[position]
        ]
        if not legal:
            continue
        low_coordinate, high_coordinate = sorted(
            transform.to_coordinate(np.asarray([lower, upper], dtype=np.float64))
        )
        proposed = round(
            float(
                transform.from_coordinate(
                    np.asarray([rng.uniform(low_coordinate, high_coordinate)])
                )[0]
            )
        )
        proposed = min(max(proposed, lower), upper)
        order = (
            [proposed, *[value for value in legal if value != proposed]]
            if proposed in legal
            else list(legal)
        )
        if len(order) > 1:
            tail = np.asarray(order[1:], dtype=np.int64)
            rng.shuffle(tail)
            order[1:] = [int(value) for value in tail]
        for value in order:
            nodes = list(parent.nodes)
            nodes[position] = int(value)
            child = Schedule(tuple(nodes))
            if child.key not in seen:
                return child
    return None


def _negative_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = -(values - values.min()) / temperature
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def _deletion_order(
    scores: Mapping[int, float],
    strategy: str,
    rng: np.random.Generator,
    top_r: int,
    temperature: float,
) -> list[int]:
    ranked = sorted(scores, key=lambda position: (scores[position], position))
    if strategy == "unseen_first":
        return ranked
    if strategy == "argmin_only":
        return ranked[:1]
    remaining = list(ranked[: max(1, min(top_r, len(ranked)))])
    sampled: list[int] = []
    while remaining:
        values = np.asarray([scores[position] for position in remaining])
        probabilities = _negative_softmax(values, temperature)
        choice = int(rng.choice(len(remaining), p=probabilities))
        sampled.append(remaining.pop(choice))
    return sampled


def apdd_delete(
    parent: Schedule,
    scores: Mapping[int, float],
    strategy: str,
    seen: set[tuple[int, ...]],
    rng: np.random.Generator,
    k_min: int,
    top_r: int,
    temperature: float,
) -> Schedule | None:
    """按 APDD 低分顺序删除一个尚未评价的内部时间点。"""

    if parent.k <= k_min:
        return None
    legal = {
        int(position): float(score)
        for position, score in scores.items()
        if 0 < int(position) < len(parent.nodes) - 1 and np.isfinite(score)
    }
    for removed_index in _deletion_order(
        legal, strategy, rng, top_r, temperature
    ):
        nodes = list(parent.nodes)
        nodes.pop(removed_index)
        child = Schedule(tuple(nodes))
        if child.key not in seen:
            return child
    return None


def segment_crossover(
    first: Schedule,
    second: Schedule,
    rng: np.random.Generator,
    minimum_gap: int,
) -> tuple[Schedule, ...]:
    """交换同 K 父代中任意两个等长连续内部片段。"""

    if first.k != second.k or first.key == second.key or first.k < 3:
        return ()
    gap = max(1, minimum_gap)
    feasible: list[tuple[Schedule, Schedule]] = []
    internal_count = first.k - 1
    for length in range(2, internal_count + 1):
        for first_start in range(1, first.k - length + 1):
            for second_start in range(1, second.k - length + 1):
                first_nodes = (
                    first.nodes[:first_start]
                    + second.nodes[second_start : second_start + length]
                    + first.nodes[first_start + length :]
                )
                second_nodes = (
                    second.nodes[:second_start]
                    + first.nodes[first_start : first_start + length]
                    + second.nodes[second_start + length :]
                )
                if not all(
                    first_nodes[index] - first_nodes[index + 1] >= gap
                    for index in range(len(first_nodes) - 1)
                ):
                    continue
                if not all(
                    second_nodes[index] - second_nodes[index + 1] >= gap
                    for index in range(len(second_nodes) - 1)
                ):
                    continue
                child_first = Schedule(tuple(first_nodes))
                child_second = Schedule(tuple(second_nodes))
                if (
                    child_first.key in {first.key, second.key}
                    or child_second.key in {first.key, second.key}
                    or child_first.key == child_second.key
                ):
                    continue
                feasible.append((child_first, child_second))
    if not feasible:
        return ()
    return feasible[int(rng.integers(0, len(feasible)))]


def random_immigrant(
    k_max: int,
    transform: LogSNRTransform,
    rng: np.random.Generator,
    minimum_gap: int,
) -> Schedule:
    """在 K_max 层随机生成一条 log-SNR Schedule。"""

    maximum_time = int(round(float(transform.times[-1])))
    endpoints = transform.to_coordinate(
        np.asarray([maximum_time, 0], dtype=np.float64)
    )
    low, high = sorted((float(endpoints[0]), float(endpoints[1])))
    coordinates = rng.uniform(low, high, k_max - 1)
    internal = transform.from_coordinate(coordinates)
    nodes = (
        maximum_time,
        *sorted((float(value) for value in internal), reverse=True),
        0,
    )
    return repair_schedule(nodes, k_max, minimum_gap, maximum_time)


def _layer_distance(
    first: Schedule,
    second: Schedule,
    transform: LogSNRTransform | None,
) -> float:
    if transform is None:
        return schedule_distance(first, second)
    left = transform.to_coordinate(np.asarray(first.nodes[1:-1], dtype=np.float64))
    right = transform.to_coordinate(np.asarray(second.nodes[1:-1], dtype=np.float64))
    scale = abs(
        float(
            np.diff(
                transform.to_coordinate(
                    np.asarray([first.nodes[0], first.nodes[-1]], dtype=np.float64)
                )
            )[0]
        )
    )
    return float(np.sqrt(np.mean(((left - right) / scale) ** 2)))


@dataclass
class BreedingArchives:
    """按 K 保存 Archive-FID 最优和 log-SNR 多样的繁殖代表。"""

    capacity: int | None
    transform: LogSNRTransform | None = None
    by_k: dict[int, list[Individual]] = field(default_factory=dict)

    def update(
        self, candidates: Iterable[Individual], protocol: str = "archive"
    ) -> None:
        merged = [item for layer in self.by_k.values() for item in layer]
        merged.extend(candidates)
        grouped: dict[int, dict[tuple[int, ...], Individual]] = {}
        for item in _eligible_items(merged, protocol):
            layer = grouped.setdefault(item.schedule.k, {})
            current = layer.get(item.schedule.key)
            if current is None or float(item.fid_for(protocol)) < float(
                current.fid_for(protocol)
            ):
                layer[item.schedule.key] = item

        rebuilt: dict[int, list[Individual]] = {}
        for k, unique in grouped.items():
            pool = sorted(
                unique.values(),
                key=lambda item: (
                    float(item.fid_for(protocol)),
                    _schedule_tie_key(item.schedule.key),
                ),
            )
            if self.capacity is None or len(pool) <= self.capacity:
                rebuilt[k] = pool
                continue
            selected = [pool.pop(0)]
            while pool and len(selected) < self.capacity:
                chosen = max(
                    pool,
                    key=lambda item: (
                        min(
                            _layer_distance(
                                item.schedule, kept.schedule, self.transform
                            )
                            for kept in selected
                        ),
                        -float(item.fid_for(protocol)),
                    ),
                )
                selected.append(chosen)
                pool.remove(chosen)
            rebuilt[k] = sorted(
                selected,
                key=lambda item: (
                    float(item.fid_for(protocol)),
                    _schedule_tie_key(item.schedule.key),
                ),
            )
        self.by_k = rebuilt
