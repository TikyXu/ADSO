"""Diffusers 组件上的显式非均匀 DDIM 采样函数。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

from .schedule import Schedule

if TYPE_CHECKING:
    from diffusers import DDIMPipeline


@dataclass
# 保存 APDD 删除分数需要的逐步状态、epsilon 和 DDIM 系数
class Trajectory:
    schedule: Schedule
    states: list[torch.Tensor]
    epsilons: list[torch.Tensor]
    coefficients: list[tuple[torch.Tensor, torch.Tensor]]
    seeds: tuple[int, ...]

# 将 APDD 节点映射为 Diffusers U-Net 时间标签。
def node_to_timestep(node: int) -> int:
    return int(node) - 1

# 按每张图的独立 seed 生成与 batch size 无关的初始噪声。
def make_initial_noise(
    seeds: tuple[int, ...],
    sample_shape: tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not seeds:
        raise ValueError("seeds cannot be empty")
    generators = [
        torch.Generator(device="cpu").manual_seed(int(seed)) for seed in seeds
    ]
    shape = (len(generators),) + tuple(int(size) for size in sample_shape)
    noise = randn_tensor(
        shape,
        generator=generators,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return noise.to(device=torch.device(device), dtype=dtype)

# 执行指定 source→target 的 eta=0 DDIM 转移。
# ImageNet-64 官方采样会先裁剪预测的 x_0，再从裁剪后的 x_0
# 反推等效 epsilon。APDD 轨迹也必须保存这个等效 epsilon，才与实际转移一致。
def _ddim_step_to_with_epsilon(
    scheduler: DDIMScheduler,
    epsilon: torch.Tensor,
    sample: torch.Tensor,
    source_timestep: int,
    target_timestep: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    calculation_device = sample.device
    alpha_bar_source = scheduler.alphas_cumprod[source_timestep].to(
        device=calculation_device,
        dtype=torch.float64,
    )
    alpha_bar_target = (
        scheduler.final_alpha_cumprod
        if target_timestep == -1
        else scheduler.alphas_cumprod[target_timestep]
    ).to(device=calculation_device, dtype=torch.float64)

    alpha_source = alpha_bar_source.sqrt()
    sigma_source = (1.0 - alpha_bar_source).clamp_min(0.0).sqrt()
    alpha_target = alpha_bar_target.sqrt()
    sigma_target = (1.0 - alpha_bar_target).clamp_min(0.0).sqrt()
    coefficient_a = alpha_target / alpha_source
    coefficient_b = sigma_target - coefficient_a * sigma_source

    effective_epsilon = epsilon
    if scheduler.config.thresholding or scheduler.config.clip_sample:
        alpha_source_sample = alpha_source.to(sample.dtype)
        sigma_source_sample = sigma_source.to(sample.dtype)
        predicted_x0 = (
            sample - sigma_source_sample * epsilon
        ) / alpha_source_sample
        if scheduler.config.thresholding:
            predicted_x0 = scheduler._threshold_sample(predicted_x0)
        else:
            clip_sample_range = float(scheduler.config.clip_sample_range)
            predicted_x0 = predicted_x0.clamp(
                -clip_sample_range,
                clip_sample_range,
            )
        effective_epsilon = (
            sample - alpha_source_sample * predicted_x0
        ) / sigma_source_sample
        # 裁剪路径直接使用官方 DDIM 形式，避免顶部时间步中
        # A*x 与 B*epsilon 两个大数在 float32 下相减造成消减误差。
        next_state = (
            alpha_target.to(sample.dtype) * predicted_x0
            + sigma_target.to(sample.dtype) * effective_epsilon
        )
    else:
        # CIFAR-10 等不裁剪路径保留原有数值运算。
        next_state = (
            coefficient_a.to(sample.dtype) * sample
            + coefficient_b.to(sample.dtype) * effective_epsilon
        )
    return next_state, coefficient_a, coefficient_b, effective_epsilon


# 保留原有公开接口：返回状态与 APDD 使用的 A/B 系数。
def ddim_step_to(
    scheduler: DDIMScheduler,
    epsilon: torch.Tensor,
    sample: torch.Tensor,
    source_timestep: int,
    target_timestep: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_state, coefficient_a, coefficient_b, _ = _ddim_step_to_with_epsilon(
        scheduler,
        epsilon,
        sample,
        source_timestep,
        target_timestep,
    )
    return next_state, coefficient_a, coefficient_b

# 按显式非均匀 Schedule 采样，并按需返回 APDD 可微轨迹。
def sample_schedule(
    pipe: DDIMPipeline,
    schedule: Schedule,
    seeds: tuple[int, ...],
    *,
    initial_noise: torch.Tensor | None = None,
    return_trajectory: bool = False,
    require_grad: bool = False,
) -> tuple[torch.Tensor, int, tuple[int, ...], Trajectory | None]:
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("seeds cannot be empty")
    parameter = next(pipe.unet.parameters())
    device, dtype = parameter.device, parameter.dtype
    sample_size = pipe.unet.config.sample_size
    spatial_shape = (
        (int(sample_size), int(sample_size))
        if isinstance(sample_size, int)
        else tuple(int(size) for size in sample_size)
    )
    sample_shape = (int(pipe.unet.config.in_channels),) + spatial_shape
    expected_shape = (len(seeds),) + sample_shape

    if initial_noise is None:
        state = make_initial_noise(seeds, sample_shape, device, dtype)
    else:
        if tuple(initial_noise.shape) != expected_shape:
            raise ValueError(
                f"initial_noise must have shape {expected_shape}, "
                f"got {tuple(initial_noise.shape)}"
            )
        state = initial_noise.to(device=device, dtype=dtype).detach().clone()
    state.requires_grad_(require_grad)

    states = [state] if return_trajectory else []
    epsilons: list[torch.Tensor] = []
    coefficients: list[tuple[torch.Tensor, torch.Tensor]] = []
    model_timesteps = tuple(node_to_timestep(node) for node in schedule.nodes[:-1])
    nfe = 0

    gradient_context = nullcontext() if require_grad else torch.no_grad()
    with gradient_context:
        for source_node, target_node in zip(schedule.nodes, schedule.nodes[1:]):
            # 1. 将 APDD 节点转换为 U-Net 的整数时间标签。
            source_timestep = node_to_timestep(source_node)
            target_timestep = node_to_timestep(target_node)
            timestep = torch.full(
                (state.shape[0],),
                source_timestep,
                device=state.device,
                dtype=torch.long,
            )

            # 2. 直接调用 Diffusers U-Net 预测 epsilon，成功后才计入 NFE。
            model_input = pipe.scheduler.scale_model_input(state, timestep)
            model_epsilon = pipe.unet(model_input, timestep).sample
            nfe += 1

            # 3. 使用 Scheduler alpha 表执行显式目标 DDIM 转移。
            state, coefficient_a, coefficient_b, effective_epsilon = (
                _ddim_step_to_with_epsilon(
                    pipe.scheduler,
                    model_epsilon,
                    state,
                    source_timestep,
                    target_timestep,
                )
            )

            # 4. 仅 full-adjoint APDD 路径保存完整可微轨迹。
            if return_trajectory:
                epsilons.append(effective_epsilon)
                coefficients.append((coefficient_a, coefficient_b))
                states.append(state)

    trajectory = None
    if return_trajectory:
        trajectory = Trajectory(
            schedule=schedule,
            states=states,
            epsilons=epsilons,
            coefficients=coefficients,
            seeds=seeds,
        )
    return state, nfe, model_timesteps, trajectory
