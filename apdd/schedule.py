"""整数 DDIM Schedule 及其可复现初始化。

科研基因型仅为不可变整数节点元组；评价、谱系和协议元数据均保存在本模块之外。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


_DEFAULT_MAX_TIME = 1000
_MIN_TIME = 0


@dataclass(frozen=True)
# Schedule 是遗传个体的唯一基因型；FID、谱系和评价协议不写入这里。
class Schedule:
    """具有整数端点的合法严格递减 DDIM Schedule。"""

    nodes: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        assert len(self.nodes) >= 2
        assert self.nodes[0] > _MIN_TIME
        assert self.nodes[-1] == _MIN_TIME
        assert all(isinstance(t, int) for t in self.nodes)
        assert all(a > b for a, b in zip(self.nodes, self.nodes[1:]))

    @property
    def k(self) -> int:
        """去噪状态转移数，也是部署 NFE。"""

        # K 统计状态转移数，因此严格等于节点数减一，也等于部署 NFE。
        return len(self.nodes) - 1

    @property
    def key(self) -> tuple[int, ...]:
        """用于缓存和去重的节点内容身份。"""

        return self.nodes


@dataclass(frozen=True)
# 可逆线性插值的时间步—log-SNR 查找表。
class LogSNRTransform:
    times: np.ndarray
    log_snr: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        log_snr = np.asarray(self.log_snr, dtype=np.float64)
        if times.ndim != 1 or log_snr.ndim != 1 or len(times) != len(log_snr):
            raise ValueError("times and log_snr must be equal one-dimensional arrays")
        if len(times) < 2 or not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing")
        differences = np.diff(log_snr)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError("log_snr must be strictly monotone")
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "log_snr", log_snr.copy())

    def to_coordinate(self, times: np.ndarray) -> np.ndarray:
        """将模型时间步插值到 log-SNR 坐标。"""

        return np.interp(np.asarray(times, dtype=np.float64), self.times, self.log_snr)

    def from_coordinate(self, coordinates: np.ndarray) -> np.ndarray:
        """通过插值反演单调 log-SNR 表得到模型时间步。"""

        coordinates = np.asarray(coordinates, dtype=np.float64)
        # np.interp 要求横坐标递增，因此下降的 log-SNR 表需要同时反转两轴。
        if self.log_snr[0] < self.log_snr[-1]:
            return np.interp(coordinates, self.log_snr, self.times)
        return np.interp(coordinates, self.log_snr[::-1], self.times[::-1])

# Apply the legacy discrete repair rule to a requested minimum gap.
def _integer_gap(minimum_gap: float) -> int:
    if not math.isfinite(minimum_gap) or minimum_gap <= 0:
        raise ValueError("minimum_gap must be positive")
    return max(1, int(math.ceil(minimum_gap)))


def _require_feasible(
    k: int,
    minimum_gap: float,
    maximum_time: int = _DEFAULT_MAX_TIME,
) -> int:
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    gap = _integer_gap(minimum_gap)
    if k * gap > maximum_time - _MIN_TIME:
        raise ValueError("node count and minimum_gap make the domain infeasible")
    return gap


def _transform_maximum_time(transform: LogSNRTransform) -> int:
    """从模型的 log-SNR 表读取扩散边界节点。"""

    maximum_time = int(round(float(transform.times[-1])))
    assert float(transform.times[0]) == float(_MIN_TIME)
    assert float(transform.times[-1]) == float(maximum_time)
    return maximum_time

# 将候选节点投影为可行的严格递减整数 Schedule。
# 依次对内部候选排序、取整，再根据前驱节点和剩余间隔边界逐点限制。
def repair_schedule(
    nodes: Iterable[float],
    k: int,
    minimum_gap: float = 1,
    maximum_time: int | None = None,
) -> Schedule:
    values = tuple(float(value) for value in nodes)
    if len(values) != k + 1:
        raise ValueError("node count must equal k + 1")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite nodes cannot be repaired")
    resolved_maximum = (
        int(round(values[0])) if maximum_time is None else int(maximum_time)
    )
    gap = _require_feasible(k, minimum_gap, resolved_maximum)

    # 先按时间降序排列，再逐点限制上下界，确保后续节点仍有足够整数间隔。
    internal = sorted(values[1:-1], reverse=True)
    projected: list[int] = [resolved_maximum]
    for index, raw_value in enumerate(internal, start=1):
        upper = projected[-1] - gap
        remaining_intervals = k - index
        lower = _MIN_TIME + remaining_intervals * gap
        candidate = int(round(raw_value))
        projected.append(min(max(candidate, lower), upper))
    projected.append(_MIN_TIME)
    return Schedule(tuple(projected))

"""
    创建修复后的 Schedule，或三种具名初始 Schedule 之一。
    ``nodes`` 可为原始候选值，或 ``"uniform"``、``"quadratic"``、
    ``"uniform_logsnr"``；具名 Schedule 需要 ``k``，并可使用真实 log-SNR 变换。
"""
def make_schedule(
    nodes: Iterable[float] | str,
    k: int | None = None,
    minimum_gap: float = 1,
    transform: LogSNRTransform | None = None,
) -> Schedule:
    if isinstance(nodes, str):
        if k is None:
            raise ValueError("named seed schedules require k")
        resolved_transform = transform or _fallback_transform()
        return repair_schedule(
            _seed_nodes(nodes, k, resolved_transform),
            k,
            minimum_gap,
            _transform_maximum_time(resolved_transform),
        )
    values = tuple(nodes)
    return repair_schedule(values, len(values) - 1 if k is None else k, minimum_gap)


# 三种确定性种子共享相同端点，只改变内部节点所在的坐标系/曲线。
def _seed_nodes(method: str, k: int, transform: LogSNRTransform) -> np.ndarray:
    maximum_time = _transform_maximum_time(transform)
    fraction = np.linspace(0.0, 1.0, k + 1)
    if method == "uniform":
        values = maximum_time - (maximum_time - _MIN_TIME) * fraction
    elif method == "quadratic":
        values = maximum_time - (maximum_time - _MIN_TIME) * fraction**2
    elif method == "uniform_logsnr":
        endpoints = transform.to_coordinate(np.asarray([maximum_time, _MIN_TIME]))
        coordinates = np.linspace(endpoints[0], endpoints[1], k + 1)
        values = transform.from_coordinate(coordinates)
    else:
        raise ValueError(f"unknown seed schedule method: {method}")
    values[0], values[-1] = maximum_time, _MIN_TIME
    return values


def uniform_schedule(
    k: int,
    minimum_gap: float = 1,
    maximum_time: int = _DEFAULT_MAX_TIME,
) -> Schedule:
    """返回确定性时间步均匀初始 Schedule。"""

    _require_feasible(k, minimum_gap, maximum_time)
    transform = _fallback_transform(maximum_time)
    return make_schedule("uniform", k, minimum_gap, transform)

# Use time as a monotone coordinate for standalone schedule checks only.
def _fallback_transform(
    maximum_time: int = _DEFAULT_MAX_TIME,
) -> LogSNRTransform:
    times = np.arange(_MIN_TIME, maximum_time + 1, dtype=np.float64)
    return LogSNRTransform(times, times)

# 随机个体在 log-SNR 坐标均匀采样，再映射回整数扩散时间。
def _random_nodes(
    k: int, rng: np.random.Generator, transform: LogSNRTransform
) -> tuple[float, ...]:    
    maximum_time = _transform_maximum_time(transform)
    endpoints = transform.to_coordinate(np.asarray([maximum_time, _MIN_TIME]))
    low, high = sorted((float(endpoints[0]), float(endpoints[1])))
    coordinates = rng.uniform(low, high, k - 1)
    internal = transform.from_coordinate(coordinates)
    return (
        maximum_time,
        *sorted((float(value) for value in internal), reverse=True),
        _MIN_TIME,
    )

"""
    使用显式随机数生成器创建唯一的确定性与随机 Schedule 种群。

    三个确定性个体分别为均匀时间、二次时间和均匀 log-SNR，
    其余个体为随机均匀 log-SNR 样本。
    传入真实 ``LogSNRTransform`` 时，相同随机状态保持旧版整数序列。
"""
def initialize_population(
    size: int,
    k: int,
    rng: np.random.Generator,
    transform: LogSNRTransform | None = None,
    minimum_gap: float = 1,
) -> list[Schedule]:
    

    if not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    resolved_transform = transform or _fallback_transform()
    maximum_time = _transform_maximum_time(resolved_transform)
    _require_feasible(k, minimum_gap, maximum_time)

    # 前三个个体提供可解释的确定性基线，其余个体提供结构多样性。
    raw_candidates = [
        _seed_nodes(method, k, resolved_transform)
        for method in ("uniform", "quadratic", "uniform_logsnr")
    ]
    population: list[Schedule] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(size, size * 100)
    attempts = 0
    while len(population) < size and attempts < max_attempts:
        candidate = (
            raw_candidates.pop(0)
            if raw_candidates
            else _random_nodes(k, rng, resolved_transform)
        )
        schedule = repair_schedule(candidate, k, minimum_gap, maximum_time)
        attempts += 1
        if schedule.key in seen:
            continue
        seen.add(schedule.key)
        population.append(schedule)
    if len(population) != size:
        raise RuntimeError(
            "could not generate the requested number of unique schedules"
        )
    return population

# 返回两条 Schedule 分位数曲线之间的归一化 L2 距离
def schedule_distance(
    first: Schedule | Iterable[float],
    second: Schedule | Iterable[float],
    *,
    quantile_points: int = 64,
) -> float:

    left_values = first.nodes if isinstance(first, Schedule) else tuple(first)
    right_values = second.nodes if isinstance(second, Schedule) else tuple(second)
    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or len(left) < 2 or len(right) < 2:
        raise ValueError("schedule_distance expects one-dimensional schedules")
    if quantile_points < 2:
        raise ValueError("quantile_points must be at least two")
    global_max = max(float(left[0]), float(right[0]))
    global_min = min(float(left[-1]), float(right[-1]))
    scale = global_max - global_min
    if scale <= 0:
        raise ValueError("schedule endpoints must span a positive interval")
    # 不同 K 的 Schedule 先插值到同一分位数网格，再计算归一化曲线距离。
    quantiles = np.linspace(0.0, 1.0, quantile_points)
    left_curve = np.interp(quantiles, np.linspace(0.0, 1.0, len(left)), left)
    right_curve = np.interp(quantiles, np.linspace(0.0, 1.0, len(right)), right)
    return float(np.sqrt(np.mean(((left_curve - right_curve) / scale) ** 2)))
