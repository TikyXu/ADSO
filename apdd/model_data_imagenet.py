"""本地加载 OpenAI improved-diffusion ImageNet-64 checkpoint 与真实数据统计。"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import DDIMPipeline, DDIMScheduler
from torch import nn

from .metrics import (
    InceptionV3FeatureExtractor,
    ReferenceStatistics,
    feature_extractor_identity,
)

# imagenet64_uncond_100M_1500K.pt 在 improved-diffusion README 中的官方训练 flags。
# MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# 其余字段取 model_and_diffusion_defaults 与官方一致的默认值。
# use_checkpoint 仅为 APDD 可微轨迹节省显存，不改变网络结构或 checkpoint 权重。
IMAGENET64_UNCOND_FLAGS = dict(
    image_size=64,
    num_channels=128,
    num_res_blocks=3,
    learn_sigma=True,
    class_cond=False,
    attention_resolutions="16,8",
    num_heads=4,
    dropout=0.0,
    use_scale_shift_norm=True,
    use_checkpoint=True,
    diffusion_steps=4000,
    noise_schedule="cosine",
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=True,
    rescale_learned_sigmas=True,
)


@dataclass
# 与 Diffusers UNet2DOutput 相同的调用约定：通过 .sample 取回模型输出。
class UNetAdapterOutput:
    sample: torch.Tensor


class ImprovedDiffusionUNet(nn.Module):
    """把 guided-diffusion UNetModel 包装为现有采样代码使用的 unet 接口。"""

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int,
        rescale_timesteps: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_timesteps = int(num_timesteps)
        self.rescale_timesteps = bool(rescale_timesteps)
        # sample_schedule 读取这两个字段确定初始噪声与采样形状。
        self.config = SimpleNamespace(sample_size=64, in_channels=3)

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor
    ) -> UNetAdapterOutput:
        # learn_sigma=True 的官方模型输出 6 通道，前 3 通道即 epsilon 预测。
        # 复刻 SpacedDiffusion._scale_timesteps：官方 U-Net 本身不执行该缩放。
        if timestep.ndim == 0:
            timestep = timestep.expand(sample.shape[0])
        model_timestep = timestep
        if self.rescale_timesteps:
            model_timestep = timestep.float() * (1000.0 / self.num_timesteps)
        output = self.model(sample, timesteps=model_timestep)
        return UNetAdapterOutput(sample=output[:, :3])

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


class ImageNet64InceptionV3FeatureExtractor(InceptionV3FeatureExtractor):
    """使用 OpenAI ImageNet64 evaluator 的 uint8 截断协议。"""

    base_version = "torch-fidelity-inception-v3-compat-2048-imagenet64-uint8-v2"

    def _to_uint8(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            return images.detach().to(self.device)
        return (
            ((images.detach().to(self.device) + 1.0) * 127.5)
            .clamp(0, 255)
            .to(torch.uint8)
        )

# 加载官方 ImageNet-64 无条件 U-Net，并构造确定性 DDIM Pipeline。
def load_imagenet64_pipeline(
    checkpoint: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> DDIMPipeline:
    from guided_diffusion.script_util import (
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    # 1. 按官方 README flags 构造架构，guided-diffusion 额外字段保持与
    # improved-diffusion 兼容（旧注意力顺序、按头数而非头维度划分）。
    config = model_and_diffusion_defaults()
    config.update(IMAGENET64_UNCOND_FLAGS)
    model, diffusion = create_model_and_diffusion(**config)
    model.load_state_dict(
        torch.load(Path(checkpoint).expanduser().resolve(), map_location="cpu")
    )
    # 2. 固定预训练去噪器：不更新参数，仅允许 APDD 对输入轨迹求梯度。
    model.eval()
    model.requires_grad_(False)
    model.to(dtype=dtype)

    # 3. 直接复用训练时的 4000 步 cosine betas，并保留官方
    # guided-diffusion 采样默认的 x_0∈[-1, 1] 裁剪语义。
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=len(diffusion.betas),
        trained_betas=np.asarray(diffusion.betas, dtype=np.float64),
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        thresholding=False,
        set_alpha_to_one=True,
    )
    pipe = DDIMPipeline(
        unet=ImprovedDiffusionUNet(
            model,
            num_timesteps=len(diffusion.betas),
            rescale_timesteps=diffusion.rescale_timesteps,
        ),
        scheduler=ddim_scheduler,
    )
    resolved_device = torch.device(device)
    pipe.to(device=resolved_device)
    return pipe


# 复刻官方 image_datasets.py 的文件枚举：排序后递归收集 jpg/jpeg/png/gif。
def list_image_files_recursively(data_dir: str | Path) -> list[str]:
    results: list[str] = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(str(data_dir), entry)
        extension = entry.split(".")[-1]
        if "." in entry and extension.lower() in ("jpg", "jpeg", "png", "gif"):
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(list_image_files_recursively(full_path))
    return results


class ImageNetImageFolderDataset(torch.utils.data.Dataset):
    """读取供官方 ImageNet64 Inception 协议使用的 RGB uint8 图像。"""

    def __init__(self, resolution: int, image_paths: Sequence[str]) -> None:
        self.resolution = resolution
        self.image_paths = list(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        from PIL import Image

        pil_image = Image.open(self.image_paths[index])
        pil_image.load()
        # PIL 版本不够新时无法使用 reducing_gap，手工先做 2 的幂 BOX 降采样。
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )
        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        # 真实图像保持原始 uint8，避免 float32 往返后再截断造成像素偏移。
        return torch.from_numpy(np.transpose(arr, [2, 0, 1]).copy()), 0


# 读取或创建 ImageNet-64 训练集 FID 统计，仅在需要时保留 KID/PR 特征。
def prepare_imagenet64_reference(
    data_root: str | Path,
    *,
    feature_extractor: torch.nn.Module | None = None,
    inception_path: str | Path | None = None,
    device: torch.device | str | None = None,
    batch_size: int,
    workers: int = 0,
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
        feature_extractor = ImageNet64InceptionV3FeatureExtractor(
            device, resolved_weights
        )
    else:
        weights_path = getattr(feature_extractor, "weights_path", None)
        resolved_weights = (
            None
            if weights_path is None
            else str(Path(weights_path).expanduser().resolve())
        )
    # 缓存身份同时绑定 Inception 版本、权重哈希和数值精度协议。
    extractor_identity = feature_extractor_identity(feature_extractor)
    statistics_path = root / "imagenet64_train_fid_statistics.npz"
    real_features_path = root / "imagenet64_train_real_features_10000.pt"
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
                    payload = torch.load(real_features_path, map_location="cpu")
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

    from torch.utils.data import DataLoader

    # 官方加载方式：递归枚举排序后的图片文件并按固定预处理确定性地遍历。
    image_paths = list_image_files_recursively(root)
    dataset = ImageNetImageFolderDataset(64, image_paths)
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
    # 使用 float64 累加一阶和二阶矩，避免保存完整特征造成额外内存占用。
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
        raise ValueError("ImageNet-64 reference needs at least two feature vectors")
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
