"""使用文件顶部常量直接点击运行 APDD 机制消融。"""

from __future__ import annotations

from pathlib import Path

import torch

from apdd.experiments import run_ablation_suite
from apdd.model_data import initialize_runtime
from main import build_components, build_settings

# ======================== 完整 APDD 对照配置 ========================
# 每个消融只覆盖一个机制，其余参数都继承这里的完整方法配置。
EXPERIMENT_NAME = "cifar10_apdd_ablation_base"
MODEL_PATH = "pretrained_models/google-ddpm-cifar10-32"
DATA_PATH = "data/cifar10"
INCEPTION_PATH = "pretrained_models/inception/weights-inception-2015-12-05-6726825d.pth"
RESULTS_ROOT = "results"
SEED = 20260801
SEARCH_SAMPLES = 500
ARCHIVE_SAMPLES = 10_000
FINAL_SAMPLES = 50_000
BATCH_SIZE = 128
RESUME = True

DEVICE = "cuda"
DTYPE = torch.float32
DETERMINISTIC = True
ALLOW_TF32 = False

K_MAX, K_MIN = 30, 5
POPULATION_SIZE = 100
WARMUP_GENERATIONS = 5
MAX_GENERATIONS = 45
PER_K_ARCHIVE_SIZE = 3
MINIMUM_GAP = 1
DELETION_STRATEGY = "unseen_first"
DELETION_SCORE_MODE = "adjoint"
TOP_R = 5
REMOVAL_TEMPERATURE = 0.20
APDD_SAMPLES = 8
APDD_PROBES = 2
APDD_PROBE_SEED = 20260802


# ======================== 默认运行的消融组 ========================
# 删除分数、删除策略和演化结构三组实验会依次执行；可注释不需要的条目分批运行。
SELECTED_ABLATIONS = (
    "random_deletion",
    "gap_deletion",
    "local_deletion",
    "lipschitz_deletion",
    "full_adjoint_deletion",
    "argmin_only",
    "unseen_first",
    "top_r_softmax",
    "greedy_single_lineage",
)


def main() -> None:
    """加载完整 APDD 对照配置并顺序运行选定的消融组。"""

    # 所有消融共享同一随机数和 CUDA 数值协议。
    initialize_runtime(
        SEED,
        deterministic=DETERMINISTIC,
        allow_tf32=ALLOW_TF32,
    )
    # base settings 表示未移除任何机制的完整 APDD。
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
    print("[Ablations] Loading components...", flush=True)
    components = build_components(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        inception_path=INCEPTION_PATH,
        device=DEVICE,
        dtype=DTYPE,
        batch_size=BATCH_SIZE,
    )
    # 每个消融写入独立目录，汇总表记录实际覆盖后的参数和 Pareto 数量。
    summary = run_ablation_suite(
        selected_names=SELECTED_ABLATIONS,
        base_settings=settings,
        components=components,
        output_root=Path(RESULTS_ROOT) / "ablations",
        resume=RESUME,
    )
    print(f"[Ablations] DONE Results={summary}", flush=True)


if __name__ == "__main__":
    main()
