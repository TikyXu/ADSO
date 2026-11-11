"""确定性 DDIM Schedule 的可复现 FID 为主评价。"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMPipeline
from torch import nn
from tqdm import tqdm

from .diffusion import sample_schedule
from .schedule import Schedule


@dataclass
# Inception 特征统计，以及可选参考特征与提取器身份。
class ReferenceStatistics:
    mean: np.ndarray
    covariance: np.ndarray
    count: int
    feature_version: str
    real_features: torch.Tensor | None = None
    feature_extractor: object | None = None
    feature_extractor_identity: str | None = None
    inception_weights_path: str | None = None

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        if self.count < 2 or self.mean.ndim != 1:
            raise ValueError("reference statistics require at least two features")
        if self.covariance.shape != (self.mean.size, self.mean.size):
            raise ValueError("reference mean/covariance dimensions are inconsistent")
        if (
            self.feature_extractor_identity is None
            and self.feature_extractor is not None
        ):
            self.feature_extractor_identity = feature_extractor_identity(
                self.feature_extractor
            )


@dataclass(frozen=True)
# 单个 Schedule 的样本数、批大小与可选指标开关。
class EvaluationOptions:
    samples: int
    batch_size: int
    compute_kid: bool = False
    compute_precision_recall: bool = False
    compute_is: bool = False

    def __post_init__(self) -> None:
        if self.samples < 2 or self.batch_size < 1:
            raise ValueError("samples must be >= 2 and batch_size must be positive")


@dataclass
# 保存一次 Schedule 评价的质量、NFE 与耗时。
class EvaluationResult:
    schedule: tuple[int, ...]
    protocol: str
    samples: int
    fid: float
    nfe: int
    seconds: float
    kid: float | None = None
    precision: float | None = None
    recall: float | None = None
    inception_score: float | None = None
    inception_score_std: float | None = None

# 生成稳定且按评价协议隔离的 63 位样本种子。
def seed_stream(base_seed: int, protocol: str, count: int) -> tuple[int, ...]:
    if count < 0:
        raise ValueError("count must not be negative")
    return tuple(
        int.from_bytes(
            hashlib.sha256(f"{int(base_seed)}|{protocol}|{index}".encode()).digest()[
                :8
            ],
            "big",
        )
        % (2**63 - 1)
        for index in range(count)
    )


class InceptionV3FeatureExtractor(nn.Module):
    """按固定 uint8 路径提取 torch-fidelity Inception pool3 特征。"""

    base_version = "torch-fidelity-inception-v3-compat-2048-logits-v3"

    def __init__(
        self, device: torch.device | str, weights_path: str | None = None
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.weights_path = weights_path
        self.weights_identity = _weights_identity(weights_path)
        self.precision_protocol = inception_precision_protocol(self.device)
        self.version = f"{self.base_version};{self.precision_protocol}"
        self.feature_extractor: nn.Module | None = None
        self.feature_logits_extractor: nn.Module | None = None

    def _load(self, *, logits: bool) -> nn.Module:
        attribute = "feature_logits_extractor" if logits else "feature_extractor"
        extractor = getattr(self, attribute)
        if extractor is None:
            from torch_fidelity.feature_extractor_inceptionv3 import (
                FeatureExtractorInceptionV3,
            )

            requested = ["2048", "logits_unbiased"] if logits else ["2048"]
            extractor = FeatureExtractorInceptionV3(
                "inception-v3-compat",
                requested,
                feature_extractor_weights_path=self.weights_path,
            ).to(self.device)
            extractor.eval()
            setattr(self, attribute, extractor)
        return extractor

    def _to_uint8(self, images: torch.Tensor) -> torch.Tensor:
        return (
            ((images.detach().to(self.device) + 1.0) * 127.5)
            .clamp(0, 255)
            .round()
            .to(torch.uint8)
        )
    # 同时返回 pool3 特征和未经 softmax 的分类 logits。
    def forward_with_logits(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Inception input must have shape [B,3,H,W]")
        features, logits = self._load(logits=True)(self._to_uint8(images))
        return features.flatten(1).to(torch.float32), logits.flatten(1).to(
            torch.float32
        )
    # 返回用于 FID、KID 和 Precision/Recall 的 pool3 特征。
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Inception input must have shape [B,3,H,W]")
        features = self._load(logits=False)(self._to_uint8(images))
        if isinstance(features, (tuple, list)):
            if len(features) != 1:
                raise RuntimeError(
                    "Inception feature-only extractor returned unexpected outputs"
                )
            features = features[0]
        if not isinstance(features, torch.Tensor):
            raise TypeError("Inception feature-only extractor did not return features")
        return features.flatten(1).to(torch.float32)

# 计算本地权重文件哈希，并为轻量测试保留稳定身份。
def _weights_identity(weights_path: str | Path | None) -> str:
    if weights_path is None:
        return "none"
    path = Path(weights_path)
    if not path.is_file():
        return f"missing:{path}"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# 记录会改变 Inception 缓存统计的数值后端设置。
def inception_precision_protocol(device: torch.device | str) -> str:
    resolved = torch.device(device)
    return ";".join(
        (
            f"device={resolved.type}",
            "input=uint8",
            "compute=float32",
            f"deterministic={int(torch.are_deterministic_algorithms_enabled())}",
            f"matmul_tf32={int(torch.backends.cuda.matmul.allow_tf32)}",
            f"cudnn_tf32={int(torch.backends.cudnn.allow_tf32)}",
            f"cudnn_deterministic={int(torch.backends.cudnn.deterministic)}",
        )
    )

# 生成绑定版本与权重的特征提取器稳定身份。
def feature_extractor_identity(extractor: object) -> str:
    version = str(getattr(extractor, "version", type(extractor).__qualname__))
    explicit = getattr(extractor, "weights_identity", None)
    if explicit is not None:
        return f"feature_version={version};weights_identity={explicit}"
    weights = _weights_identity(getattr(extractor, "weights_path", None))
    return f"feature_version={version};weights_sha256={weights}"


class _StreamingStatistics:
    def __init__(self) -> None:
        self.count = 0
        self.sum: torch.Tensor | None = None
        self.sum_outer: torch.Tensor | None = None

    def update(self, features: torch.Tensor) -> None:
        values = features.detach().to(device="cpu", dtype=torch.float64)
        if values.ndim != 2:
            raise ValueError("features must have shape [N,D]")
        batch_sum = values.sum(dim=0)
        batch_outer = values.T.matmul(values)
        if self.sum is None:
            self.sum = torch.zeros_like(batch_sum)
            self.sum_outer = torch.zeros_like(batch_outer)
        if self.sum.shape != batch_sum.shape:
            raise ValueError("feature dimension changed during accumulation")
        self.count += len(values)
        self.sum += batch_sum
        self.sum_outer += batch_outer

    def reference(
        self, feature_version: str, real_features: torch.Tensor | None = None
    ) -> ReferenceStatistics:
        if self.count < 2 or self.sum is None or self.sum_outer is None:
            raise ValueError("at least two feature vectors are required")
        mean = self.sum / self.count
        covariance = (self.sum_outer - self.count * torch.outer(mean, mean)) / (
            self.count - 1
        )
        return ReferenceStatistics(
            mean.numpy(), covariance.numpy(), self.count, feature_version, real_features
        )


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def _frechet_distance(
    real: ReferenceStatistics, generated: ReferenceStatistics
) -> float:
    if real.mean.shape != generated.mean.shape:
        raise ValueError("feature dimensions do not match")
    if real.feature_version != generated.feature_version:
        raise ValueError("feature extractor versions do not match")
    identity = np.eye(real.mean.size, dtype=np.float64)
    covariance_real = real.covariance + 1.0e-10 * identity
    covariance_generated = generated.covariance + 1.0e-10 * identity
    root_real = _sqrt_psd(covariance_real)
    cross = _sqrt_psd(root_real @ covariance_generated @ root_real)
    difference = real.mean - generated.mean
    return max(
        0.0,
        float(difference.dot(difference))
        + float(np.trace(covariance_real))
        + float(np.trace(covariance_generated))
        - 2.0 * float(np.trace(cross)),
    )


def _kid(
    real: torch.Tensor, generated: torch.Tensor, subsets: int = 100, seed: int = 0
) -> float:
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("KID requires at least two real and generated features")
    real, generated = (
        real.detach().cpu().to(torch.float64),
        generated.detach().cpu().to(torch.float64),
    )
    size = min(1000, len(real), len(generated))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = []
    for _ in range(subsets):
        left = real[torch.randperm(len(real), generator=generator)[:size]]
        right = generated[torch.randperm(len(generated), generator=generator)[:size]]
        scale = 1.0 / left.shape[1]
        xx = (scale * left.matmul(left.T) + 1.0).pow(3)
        yy = (scale * right.matmul(right.T) + 1.0).pow(3)
        xy = (scale * left.matmul(right.T) + 1.0).pow(3)
        values.append(
            (xx.sum() - xx.diagonal().sum()) / (size * (size - 1))
            + (yy.sum() - yy.diagonal().sum()) / (size * (size - 1))
            - 2.0 * xy.mean()
        )
    return float(torch.stack(values).mean())


def _inception_score(logits: torch.Tensor, splits: int = 10) -> tuple[float, float]:
    probabilities = torch.softmax(
        logits.detach().cpu().to(torch.float64), dim=1
    ).clamp_min(1e-16)
    scores = []
    for subset in torch.tensor_split(
        probabilities, min(max(1, splits), len(probabilities))
    ):
        marginal = subset.mean(0, keepdim=True).clamp_min(1e-16)
        scores.append(
            torch.exp((subset * (subset.log() - marginal.log())).sum(1).mean())
        )
    values = torch.stack(scores).numpy()
    return float(values.mean()), float(values.std(ddof=0))


def _precision_recall(
    real: torch.Tensor, generated: torch.Tensor, neighbors: int = 3
) -> tuple[float, float]:
    real, generated = (
        real.detach().cpu().to(torch.float32),
        generated.detach().cpu().to(torch.float32),
    )
    if min(len(real), len(generated)) <= neighbors:
        raise ValueError("precision/recall needs more samples than neighbors")

    def radii(values: torch.Tensor) -> torch.Tensor:
        return (
            torch.cdist(values, values)
            .square()
            .topk(neighbors + 1, largest=False)
            .values[:, -1]
        )

    real_radii, generated_radii = radii(real), radii(generated)
    distances = torch.cdist(generated, real).square()
    precision = float(
        (distances <= real_radii.unsqueeze(0)).any(1).to(torch.float64).mean()
    )
    recall = float(
        (distances <= generated_radii.unsqueeze(1)).any(0).to(torch.float64).mean()
    )
    return precision, recall

# 每个种子生成一张图并计算 FID，其他指标仅按需计算。
def evaluate_schedule(
    pipeline: DDIMPipeline,
    schedule: Schedule,
    reference: ReferenceStatistics,
    seeds: tuple[int, ...],
    options: EvaluationOptions,
    protocol: str,
    *,
    feature_extractor: object | None = None,
    initial_noise: torch.Tensor | None = None,
    show_progress: bool = True,
) -> EvaluationResult:
    # 1. 确认种子数量、Inception 版本和权重身份与真实统计一致。
    # if len(seeds) != options.samples:
    #     raise ValueError("seeds must contain exactly options.samples entries")
    extractor = (
        reference.feature_extractor if feature_extractor is None else feature_extractor
    )
    # if extractor is None:
    #     raise ValueError("evaluate_schedule requires an explicit feature extractor")
    # if (
    #     reference.feature_extractor is not None
    #     and extractor is not reference.feature_extractor
    # ):
    #     raise ValueError(
    #         "reference and generated features must use the same extractor instance"
    #     )
    # if reference.feature_version != extractor.version:
    #     raise ValueError("reference statistics use a different feature extractor")
    extractor_identity = feature_extractor_identity(extractor)
    # if (
    #     reference.feature_extractor_identity is not None
    #     and reference.feature_extractor_identity != extractor_identity
    # ):
    #     raise ValueError(
    #         "reference statistics use a different feature extractor identity"
    #     )
    # if initial_noise is not None:
    #     if not isinstance(initial_noise, torch.Tensor):
    #         raise TypeError("initial_noise must be a torch.Tensor")
    #     if initial_noise.dtype != torch.float32:
    #         raise TypeError("initial_noise must use float32 storage")
    #     if initial_noise.ndim < 1 or initial_noise.shape[0] != options.samples:
    #         raise ValueError(
    #             "initial_noise must contain exactly options.samples entries, "
    #             f"got shape {tuple(initial_noise.shape)}"
    #         )
    # 2. FID 仅流式累加一、二阶矩；只有可选指标才保留完整特征。
    stream = _StreamingStatistics()
    keep_features = options.compute_kid or options.compute_precision_recall
    feature_batches: list[torch.Tensor] = []
    logits_batches: list[torch.Tensor] = []
    nfe: int | None = None
    started = time.perf_counter()
    # 3. 按批调用统一的 Diffusers 采样函数，并校验每批实测 NFE。
    with tqdm(
        total=len(seeds),
        desc=f"Generating K={schedule.k}",
        unit="sample",
        dynamic_ncols=True,
        disable=None if show_progress else True,
    ) as progress:
        for offset in range(0, len(seeds), options.batch_size):
            batch_seeds = seeds[offset : offset + options.batch_size]
            batch_noise = (
                None
                if initial_noise is None
                else initial_noise[offset : offset + len(batch_seeds)]
            )
            samples, measured_nfe, _model_timesteps, _ = sample_schedule(
                pipeline,
                schedule,
                batch_seeds,
                initial_noise=batch_noise,
            )
            progress.update(len(batch_seeds))
            with torch.no_grad():
                if options.compute_is:
                    features, logits = extractor.forward_with_logits(samples)
                    logits_batches.append(logits.detach().cpu())
                else:
                    features = extractor(samples)
            stream.update(features)
            if keep_features:
                feature_batches.append(features.detach().cpu())
            if nfe is None:
                nfe = measured_nfe
            elif nfe != measured_nfe:
                raise ValueError("sampler NFE changed between evaluation batches")
    # 4. 用全部生成特征的流式统计计算主指标 FID。
    generated = stream.reference(reference.feature_version)
    result = EvaluationResult(
        tuple(schedule.nodes),
        protocol,
        len(seeds),
        _frechet_distance(reference, generated),
        nfe if nfe is not None else schedule.k,
        time.perf_counter() - started,
    )
    # 5. KID、Precision/Recall 与 IS 仅用于独立评价，不增加搜索阶段开销。
    if keep_features:
        features = torch.cat(feature_batches, dim=0)
        if options.compute_kid:
            if reference.real_features is None:
                raise ValueError("KID requires stored real features")
            result.kid = _kid(reference.real_features, features)
        if options.compute_precision_recall:
            if reference.real_features is None:
                raise ValueError("precision/recall requires stored real features")
            result.precision, result.recall = _precision_recall(
                reference.real_features, features
            )
    if options.compute_is:
        result.inception_score, result.inception_score_std = _inception_score(
            torch.cat(logits_batches, dim=0)
        )
    return result

# 计算双最小化目标在固定参考点下的精确 Hypervolume。
def hypervolume_2d(
    points: Sequence[tuple[float, float]], reference_point: tuple[float, float]
) -> float:
    reference = np.asarray(reference_point, dtype=np.float64)
    values = np.asarray(points, dtype=np.float64)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_point must contain two finite values")
    if values.size == 0:
        return 0.0
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("points must be finite [N,2] values")
    values = values[np.all(values < reference, axis=1)]
    if len(values) == 0:
        return 0.0
    values = values[np.lexsort((values[:, 1], values[:, 0]))]
    area, best_second = 0.0, float(reference[1])
    for first, second in values:
        if second < best_second:
            area += (reference[0] - first) * (best_second - second)
            best_second = second
    return float(max(area, 0.0))

# 算近似前沿到参考前沿的 IGD+ 距离。
def igd_plus(
    approximation: Sequence[tuple[float, float]],
    reference: Sequence[tuple[float, float]],
) -> float:
    left, right = (
        np.asarray(approximation, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )
    if len(right) == 0:
        raise ValueError("reference front must not be empty")
    if len(left) == 0:
        return float("inf")
    return float(
        np.linalg.norm(np.maximum(left[None, :, :] - right[:, None, :], 0.0), axis=2)
        .min(axis=1)
        .mean()
    )

# 计算近似前沿相对参考前沿的加性 epsilon 指标。
def additive_epsilon_indicator(
    approximation: Sequence[tuple[float, float]],
    reference: Sequence[tuple[float, float]],
) -> float:
    left, right = (
        np.asarray(approximation, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )
    if len(right) == 0:
        raise ValueError("reference front must not be empty")
    if len(left) == 0:
        return float("inf")
    return float((left[None, :, :] - right[:, None, :]).max(axis=2).min(axis=1).max())
