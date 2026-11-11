"""本地加载 Diffusers CIFAR-10 checkpoint 与真实数据统计。"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMPipeline, DDIMScheduler, DDPMScheduler, UNet2DModel

from .metrics import (
    InceptionV3FeatureExtractor,
    ReferenceStatistics,
    feature_extractor_identity,
)

# 在加载资源前设置搜索与评价共享的可复现运行环境。
def initialize_runtime(
    seed: int,
    *,
    deterministic: bool = True,
    allow_tf32: bool = False,
    cuda_device: int | None = None,
) -> None:
    # 该函数必须在模型和 Inception 初始化前调用，保证搜索与独立评价协议一致。
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        if cuda_device is None:
            torch.cuda.manual_seed_all(seed)
        else:
            with torch.cuda.device(cuda_device):
                torch.cuda.manual_seed(seed)

# 加载本地 CIFAR-10 U-Net，并构造官方确定性 DDIM Pipeline。
def load_cifar10_pipeline(
    checkpoint: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> DDIMPipeline:
    # 1. 同时兼容 Diffusers 平铺目录和 unet/scheduler 子目录。
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"checkpoint must be a local directory: {root}")
    unet_subfolder = "unet" if (root / "unet").is_dir() else None
    scheduler_subfolder = "scheduler" if (root / "scheduler").is_dir() else None

    unet = UNet2DModel.from_pretrained(
        str(root),
        subfolder=unet_subfolder,
        local_files_only=True,
        torch_dtype=dtype,
    )
    original_scheduler = DDPMScheduler.from_pretrained(
        str(root), subfolder=scheduler_subfolder, local_files_only=True
    )
    # 2. 从 checkpoint 的训练 Scheduler 配置构造统一 DDIM 协议。
    ddim_scheduler = DDIMScheduler.from_config(
        original_scheduler.config,
        prediction_type="epsilon",
        clip_sample=False,
        thresholding=False,
        set_alpha_to_one=True,
    )
    pipe = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)

    # 3. 固定预训练去噪器：不更新参数，仅允许 APDD 对输入轨迹求梯度。
    # dtype 已在 from_pretrained(torch_dtype=...) 中设定，这里仅迁移设备，避免触发
    # diffusers 关于 `.to(dtype=...)` 的 float32 保护警告。
    resolved_device = torch.device(device)
    pipe.to(device=resolved_device)
    pipe.unet.eval()
    pipe.unet.requires_grad_(False)
    return pipe

# 读取或创建 CIFAR-10 训练集 FID 统计，仅在需要时保留 KID/PR 特征。
def prepare_cifar10_reference(
    data_root: str | Path,
    *,
    feature_extractor: torch.nn.Module | None = None,
    inception_path: str | Path | None = None,
    device: torch.device | str | None = None,
    batch_size: int,
    workers: int = 0,
    download: bool = False,
    retain_real_features: bool = False,
) -> ReferenceStatistics:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if feature_extractor is None:
        if inception_path is None or device is None:
            raise ValueError(
                "feature_extractor or both inception_path and device are required"
            )
        resolved_weights = str(Path(inception_path).expanduser().resolve())
        if not Path(resolved_weights).is_file():
            raise FileNotFoundError(
                f"Inception weights were not found locally: {resolved_weights}"
            )
        feature_extractor = InceptionV3FeatureExtractor(device, resolved_weights)
    else:
        weights_path = getattr(feature_extractor, "weights_path", None)
        resolved_weights = (
            None
            if weights_path is None
            else str(Path(weights_path).expanduser().resolve())
        )
    # 缓存身份同时绑定 Inception 版本、权重哈希和数值精度协议。
    extractor_identity = feature_extractor_identity(feature_extractor)
    statistics_path = root / "cifar10_train_fid_statistics.npz"
    real_features_path = root / "cifar10_train_real_features_10000.pt"
    feature_version = str(feature_extractor.version)
    # 只有身份完全一致的统计才可复用，避免旧权重或 TF32 设置污染 FID。
    if statistics_path.is_file() and (
        not retain_real_features or real_features_path.is_file()
    ):
        with np.load(statistics_path, allow_pickle=False) as archive:
            cached_identity = (
                str(archive["feature_extractor_identity"].item())
                if "feature_extractor_identity" in archive.files
                else None
            )
            if (
                str(archive["feature_version"].item()) == feature_version
                and cached_identity == extractor_identity
            ):
                features: torch.Tensor | None = None
                if retain_real_features:
                    try:
                        payload = torch.load(real_features_path, map_location="cpu")
                    except (OSError, RuntimeError, ValueError):
                        payload = None
                    # Real features are valid only for the exact local extractor.
                    # Legacy bare tensors omit this metadata and are cache misses.
                    if (
                        isinstance(payload, dict)
                        and payload.get("feature_version") == feature_version
                        and payload.get("feature_extractor_identity")
                        == extractor_identity
                        and isinstance(payload.get("features"), torch.Tensor)
                    ):
                        features = payload["features"]
                if not retain_real_features or features is not None:
                    return ReferenceStatistics(
                        archive["mean"],
                        archive["covariance"],
                        int(archive["count"]),
                        feature_version,
                        features,
                        feature_extractor,
                        extractor_identity,
                        resolved_weights,
                    )

    try:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError as error:  # pragma: no cover - exercised in the CUDA environment
        raise RuntimeError(
            "torchvision is required to prepare CIFAR-10 reference statistics"
        ) from error
    # CIFAR-10 图像统一映射到模型输出使用的 [-1,1] 范围。
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda image: image * 2.0 - 1.0)]
    )
    try:
        dataset = datasets.CIFAR10(
            str(root), train=True, transform=transform, download=download
        )
    except RuntimeError as error:
        if not download:
            raise FileNotFoundError(
                f"CIFAR-10 was not found locally at {root}"
            ) from error
        raise
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, workers),
        pin_memory=False,
    )
    count = 0
    total: torch.Tensor | None = None
    outer: torch.Tensor | None = None
    kept: list[torch.Tensor] = []
    kept_count = 0
    # 使用 float64 累加一阶和二阶矩，避免保存完整 50k 特征造成额外内存占用。
    for images, _ in loader:
        with torch.no_grad():
            values = feature_extractor(images).detach().cpu().to(torch.float64)
        if values.ndim != 2:
            raise ValueError("feature extractor must return [N,D] features")
        batch_sum, batch_outer = values.sum(0), values.T.matmul(values)
        if total is None:
            total, outer = torch.zeros_like(batch_sum), torch.zeros_like(batch_outer)
        count += len(values)
        total += batch_sum
        outer += batch_outer
        # KID/Precision/Recall 才按需保留 10k 真实特征；纯 FID 不生成该制品。
        if retain_real_features and kept_count < 10_000:
            take = min(len(values), 10_000 - kept_count)
            kept.append(values[:take].to(torch.float32))
            kept_count += take
    if count < 2 or total is None or outer is None:
        raise ValueError("CIFAR-10 reference needs at least two feature vectors")
    mean = total / count
    covariance = (outer - count * torch.outer(mean, mean)) / (count - 1)
    np.savez_compressed(
        statistics_path,
        count=np.asarray(count, dtype=np.int64),
        mean=mean.numpy(),
        covariance=covariance.numpy(),
        feature_version=np.asarray(feature_version),
        feature_extractor_identity=np.asarray(extractor_identity),
    )
    features = torch.cat(kept, dim=0) if retain_real_features else None
    if features is not None:
        torch.save(
            {
                "features": features,
                "feature_version": feature_version,
                "feature_extractor_identity": extractor_identity,
            },
            real_features_path,
        )
    return ReferenceStatistics(
        mean.numpy(),
        covariance.numpy(),
        count,
        feature_version,
        features,
        feature_extractor,
        extractor_identity,
        resolved_weights,
    )
