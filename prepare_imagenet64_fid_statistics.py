"""把 OpenAI 全训练集 ImageNet64 FID 统计转换为 APDD 缓存格式。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from apdd.metrics import feature_extractor_identity
from apdd.model_data import initialize_runtime
from apdd.model_data_imagenet import ImageNet64InceptionV3FeatureExtractor


Official_statistics_url = (
    "https://openaipublic.blob.core.windows.net/diffusion/"
    "jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz"
)
Official_statistics_sha256 = (
    "118b1020342001ce8a015b7cb5778f14089a31c7c60fb287ad8ec9706c5cdf37"
)
Official_statistics_path = Path(
    "data/imagenet64/VIRTUAL_imagenet64_labeled.npz"
)
Output_path = Path(
    "data/imagenet64_png/imagenet64_train_fid_statistics.npz"
)
Inception_path = Path(
    "pretrained_models/inception/weights-inception-2015-12-05-6726825d.pth"
)
Inception_sha256 = (
    "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
)
Training_image_count = 1_281_167
Seed = 20260801


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_official_statistics(
    source: str | Path,
    target: str | Path,
    *,
    count: int,
    feature_version: str,
    extractor_identity: str,
) -> None:
    """将官方 mu/sigma 字段写成 APDD 的参考统计 schema。"""

    with np.load(Path(source), allow_pickle=False) as archive:
        mean = np.asarray(archive["mu"], dtype=np.float64)
        covariance = np.asarray(archive["sigma"], dtype=np.float64)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_path,
        count=np.asarray(count, dtype=np.int64),
        mean=mean,
        covariance=covariance,
        feature_version=np.asarray(feature_version),
        feature_extractor_identity=np.asarray(extractor_identity),
        source=np.asarray("OpenAI VIRTUAL_imagenet64_labeled.npz"),
    )


if __name__ == "__main__":
    Official_statistics_path.parent.mkdir(parents=True, exist_ok=True)
    if not Official_statistics_path.is_file():
        print("Downloading official OpenAI ImageNet64 reference statistics...", flush=True)
        temporary_path = Official_statistics_path.with_suffix(".npz.part")
        urlretrieve(Official_statistics_url, temporary_path)
        assert _sha256(temporary_path) == Official_statistics_sha256
        temporary_path.replace(Official_statistics_path)
    assert _sha256(Official_statistics_path) == Official_statistics_sha256
    assert _sha256(Inception_path) == Inception_sha256

    initialize_runtime(
        Seed,
        deterministic=True,
        allow_tf32=False,
        cuda_device=None,
    )
    extractor = ImageNet64InceptionV3FeatureExtractor(
        "cuda:0",
        str(Inception_path.expanduser().resolve()),
    )
    convert_official_statistics(
        Official_statistics_path,
        Output_path,
        count=Training_image_count,
        feature_version=str(extractor.version),
        extractor_identity=feature_extractor_identity(extractor),
    )
    print(f"ImageNet64 FID statistics ready: {Output_path}", flush=True)
