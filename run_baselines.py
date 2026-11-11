"""使用 FID-50K 协议统一评价外部算法优化的固定 Schedule。"""

from __future__ import annotations

from pathlib import Path

import torch

from apdd.experiments import evaluate_fixed_schedules
from apdd.model_data import initialize_runtime
from main import build_components

# ======================== 模型、数据与结果路径 ========================
# 固定使用当前 CIFAR-10 模型、本地 Inception 权重和真实数据统计。
MODEL_PATH = "pretrained_models/google-ddpm-cifar10-32"
DATA_PATH = "data/cifar10"
INCEPTION_PATH = "pretrained_models/inception/weights-inception-2015-12-05-6726825d.pth"

# ======================== FID-50K 评价设置 ========================
# 所有方法共享同一 final seed stream，每个 Schedule 只评价一次。
SEED = 20260801
FINAL_SAMPLES = 50_000
BATCH_SIZE = 1024

# ======================== 运行设备与可复现性 ========================
# 在加载模型和 Inception 前固定随机数与 CUDA 数值设置。
DEVICE = "cuda"
DTYPE = torch.float32
DETERMINISTIC = True
ALLOW_TF32 = False

# ======================== 外部算法优化得到的 Schedule ========================
# 每个值是完整的严格递减整数 APDD 节点：1000 开始、0 结束。
# 节点数量必须等于 K+1；真实配置不得使用省略号。
#
# 例如，下面是完整的 DDIM 均匀 K=30 格式：
# BASELINE_SCHEDULES = {
#     "DDIM_uniform_K30": (
#         1000, 967, 933, 900, 867, 833, 800, 767, 733, 700, 667,
#         633, 600, 567, 533, 500, 467, 433, 400, 367, 333, 300,
#         267, 233, 200, 167, 133, 100, 67, 33, 0,
#     ),
# }
# 实际比较 AYS、ATS 时，将键名和完整节点替换为对应方法的结果。
BASELINE_SCHEDULES = {
    "DDIM_uniform_K30": (
        1000, 967, 933, 900, 867, 833, 800, 767, 733, 700, 667,
        633, 600, 567, 533, 500, 467, 433, 400, 367, 333, 300,
        267, 233, 200, 167, 133, 100, 67, 33, 0,
    ),
}


def main() -> None:
    """加载共享 CIFAR-10 组件并逐个评价外部基线 Schedule。"""

    if not BASELINE_SCHEDULES:
        print(
            "Set BASELINE_SCHEDULES to complete named APDD schedules before running.",
            flush=True,
        )
        return
    initialize_runtime(
        SEED,
        deterministic=DETERMINISTIC,
        allow_tf32=ALLOW_TF32,
    )
    print("[Baselines] Loading components...", flush=True)
    components = build_components(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        inception_path=INCEPTION_PATH,
        device=DEVICE,
        dtype=DTYPE,
        batch_size=BATCH_SIZE,
    )
    # 结果保存到 results/baselines/results.csv，每完成一个 Schedule 就立即刷新。
    output_path = Path("results") / "baselines" / "results.csv"
    results = evaluate_fixed_schedules(
        BASELINE_SCHEDULES,
        components=components,
        output_path=output_path,
        samples=FINAL_SAMPLES,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    print(f"[Baselines] DONE Results={results}", flush=True)


if __name__ == "__main__":
    main()
