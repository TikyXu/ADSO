"""直接点击运行的 ImageNet-64 APDD 搜索入口，所有科研参数均在文件顶部显式设置。"""

from __future__ import annotations

import numpy as np
import torch

from apdd.model_data import initialize_runtime
from apdd.model_data_imagenet import (
    load_imagenet64_pipeline,
    prepare_imagenet64_reference,
)
from apdd.parallel import close_schedule_workers, start_schedule_workers
from apdd.schedule import LogSNRTransform
from apdd.search import SearchComponents, SearchSettings, run_search

# ======================== 直接运行时可修改的实验参数 ========================
# 模型、数据、Inception 权重都只从本地读取，搜索过程中不会重新训练模型。
EXPERIMENT_NAME = "imagenet64_adso_fixed_v2_20260902"
MODEL_PATH = "pretrained_models/openai-improved_diffusion-imagenet64/imagenet64_uncond_100M_1500K.pt"
DATA_PATH = "data/imagenet64_png"
INCEPTION_PATH = "pretrained_models/inception/weights-inception-2015-12-05-6726825d.pth"
RESULTS_ROOT = "results"
SEED = 20260801
K_MAX, K_MIN = 30, 5
POPULATION_SIZE = 100
WARMUP_GENERATIONS = 5
MAX_GENERATIONS = 45
SEARCH_SAMPLES = 2000
ARCHIVE_SAMPLES = 10_000
FINAL_SAMPLES = 50_000
# 64x64 分辨率加 100M 参数模型，显存消耗约为 CIFAR-10 的数倍，可按 GPU 调整。
BATCH_SIZE = 512
RESUME = False

DTYPE = torch.float32
PER_K_ARCHIVE_SIZE = 3
MINIMUM_GAP = 1
DELETION_STRATEGY = "unseen_first"
DELETION_SCORE_MODE = "adjoint"
TOP_R = 5
REMOVAL_TEMPERATURE = 0.20
APDD_SAMPLES = 8
APDD_PROBES = 2
APDD_PROBE_SEED = 20260802

# 关闭 TF32 并启用确定性算法，使相同 Schedule 和种子的评价可复现。
DETERMINISTIC = True
ALLOW_TF32 = False


def build_settings(
    *,
    experiment_name: str,
    results_root: str,
    seed: int,
    k_max: int,
    k_min: int,
    population_size: int,
    warmup_generations: int,
    max_generations: int,
    search_samples: int,
    archive_samples: int,
    final_samples: int,
    batch_size: int,
    per_k_archive_size: int,
    minimum_gap: int,
    deletion_strategy: str,
    deletion_score_mode: str,
    top_r: int,
    removal_temperature: float,
    apdd_samples: int,
    apdd_probes: int,
    apdd_probe_seed: int,
    deterministic: bool,
    allow_tf32: bool,
    resume: bool,
) -> SearchSettings:
    """仅使用入口文件中的显式参数构造搜索配置。"""

    # 所有搜索参数都从入口显式传入，避免隐藏的 YAML 或命令行覆盖。
    return SearchSettings(
        experiment_name=experiment_name,
        results_root=results_root,
        seed=seed,
        k_max=k_max,
        k_min=k_min,
        population_size=population_size,
        warmup_generations=warmup_generations,
        max_generations=max_generations,
        search_samples=search_samples,
        archive_samples=archive_samples,
        final_samples=final_samples,
        batch_size=batch_size,
        per_k_archive_size=per_k_archive_size,
        minimum_gap=minimum_gap,
        deletion_strategy=deletion_strategy,
        top_r=top_r,
        removal_temperature=removal_temperature,
        apdd_samples=apdd_samples,
        apdd_probes=apdd_probes,
        apdd_probe_seed=apdd_probe_seed,
        deterministic=deterministic,
        allow_tf32=allow_tf32,
        resume=resume,
        deletion_score_mode=deletion_score_mode,
    )


def build_components(
    *,
    model_path: str,
    data_path: str,
    inception_path: str,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    retain_real_features: bool = False,
) -> SearchComponents:
    """加载本地 ImageNet-64 资产并构造搜索共享组件。"""

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the default entry settings")
    # 1. 所有候选共用同一 Diffusers Pipeline、真实统计和 Inception。
    pipeline = load_imagenet64_pipeline(model_path, device=device, dtype=dtype)
    reference = prepare_imagenet64_reference(
        data_path,
        inception_path=inception_path,
        device=device,
        batch_size=batch_size,
        workers=0,
        retain_real_features=retain_real_features,
    )
    # 2. 直接读取 Scheduler 的 float32 alpha 表，保持旧版数值转换顺序。
    # APDD 节点 0 是虚拟干净端点；前置 alpha_bar=1 后构造 log-SNR 映射。
    # 该模型使用 4000 步 cosine 调度，节点坐标空间为 0..4000。
    alpha_bar = pipeline.scheduler.alphas_cumprod.detach().cpu().numpy()
    coordinate_alpha_bar = np.concatenate(([1.0 - 1.0e-12], alpha_bar))
    coordinate_alpha_bar = np.clip(coordinate_alpha_bar, 1.0e-12, 1.0 - 1.0e-12)
    log_snr = np.log(coordinate_alpha_bar) - np.log1p(-coordinate_alpha_bar)
    transform = LogSNRTransform(
        np.arange(len(coordinate_alpha_bar), dtype=np.float64),
        log_snr,
    )
    return SearchComponents(
        pipeline,
        reference,
        transform,
        reference.feature_extractor,
    )


def main() -> None:
    """加载 ImageNet-64 组件并运行完整 APDD 搜索。"""

    gpu_count = torch.cuda.device_count()

    # 必须先固定数值运行环境，再加载模型或计算/读取真实 FID 统计。
    torch.cuda.set_device(0)
    initialize_runtime(
        SEED,
        deterministic=DETERMINISTIC,
        allow_tf32=ALLOW_TF32,
        cuda_device=0,
    )
    # 先把顶部常量整理为搜索配置，再构造只与 ImageNet-64 相关的薄加载层。
    settings = build_settings(
        experiment_name=EXPERIMENT_NAME,
        results_root=RESULTS_ROOT,
        seed=SEED,
        k_max=K_MAX,
        k_min=K_MIN,
        population_size=POPULATION_SIZE,
        warmup_generations=WARMUP_GENERATIONS,
        max_generations=MAX_GENERATIONS,
        search_samples=SEARCH_SAMPLES,
        archive_samples=ARCHIVE_SAMPLES,
        final_samples=FINAL_SAMPLES,
        batch_size=BATCH_SIZE,
        per_k_archive_size=PER_K_ARCHIVE_SIZE,
        minimum_gap=MINIMUM_GAP,
        deletion_strategy=DELETION_STRATEGY,
        deletion_score_mode=DELETION_SCORE_MODE,
        top_r=TOP_R,
        removal_temperature=REMOVAL_TEMPERATURE,
        apdd_samples=APDD_SAMPLES,
        apdd_probes=APDD_PROBES,
        apdd_probe_seed=APDD_PROBE_SEED,
        deterministic=DETERMINISTIC,
        allow_tf32=ALLOW_TF32,
        resume=RESUME,
    )
    components = build_components(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        inception_path=INCEPTION_PATH,
        device="cuda:0",
        dtype=DTYPE,
        batch_size=BATCH_SIZE,
    )
    parallel_context = start_schedule_workers(
        gpu_count=gpu_count,
        component_builder=build_components,
        component_kwargs={
            "model_path": MODEL_PATH,
            "data_path": DATA_PATH,
            "inception_path": INCEPTION_PATH,
            "dtype": DTYPE,
            "batch_size": BATCH_SIZE,
        },
        seed=SEED,
        deterministic=DETERMINISTIC,
        allow_tf32=ALLOW_TF32,
    )
    try:
        # run_search 按文件顶部设置执行 Warmup、Evolution 和一次 Final 评价。
        run_search(
            settings,
            components,
            parallel_context=parallel_context,
            cuda_device=0,
        )
    finally:
        close_schedule_workers(parallel_context)


if __name__ == "__main__":
    main()
