"""
Spectrum core: Chebyshev spectral forecasting for diffusion sampling acceleration.

Pure logic only — no ComfyUI. Used by the ComfyUI model patcher for both
image and video generation (same MODEL abstraction in ComfyUI).

Reference: CVPR 2026 "Adaptive Spectral Feature Forecasting for Diffusion
Sampling Acceleration" — https://github.com/hanjq17/Spectrum
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


# -------- Time / schedule ----------------------------------------------------


def timestep_to_float(timestep) -> float:
    """Single scalar from a possibly batched timestep tensor."""
    if torch.is_tensor(timestep):
        return float(timestep.flatten()[0].item())
    return float(timestep)


def schedule_bounds_from_cond(cond: dict) -> Optional[Tuple[float, float]]:
    """(t_min, t_max) from ComfyUI cond['transformer_options']['sample_sigmas'] or 'sigmas'."""
    try:
        opts = cond.get("transformer_options") if isinstance(cond, dict) else None
        if not isinstance(opts, dict):
            return None
        sigmas = opts.get("sample_sigmas") or opts.get("sigmas")
        if torch.is_tensor(sigmas) and sigmas.numel() > 1:
            s = sigmas.detach().flatten().float()
            return float(s.min().item()), float(s.max().item())
    except Exception:
        pass
    return None


def map_to_chebyshev_domain(t: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
    """Map t in [t_min, t_max] to x in [-1, 1] for Chebyshev T_n(x)."""
    span = t_max - t_min
    if abs(span) < 1e-12:
        return torch.zeros_like(t)
    x = ((t - t_min) / span) * 2.0 - 1.0
    return x.clamp(-1.0, 1.0)


# -------- Chebyshev basis and ridge regression ---------------------------------


def chebyshev_design(t: torch.Tensor, num_bases: int) -> torch.Tensor:
    """
    Design matrix Phi: Phi[i, k] = T_k(x_i), k = 0..num_bases-1.
    Recurrence: T_0 = 1, T_1 = x, T_k = 2*x*T_{k-1} - T_{k-2}.
    """
    if num_bases < 1:
        raise ValueError("num_bases must be >= 1")
    t = t.reshape(-1).float()
    n = t.shape[0]
    phi = torch.empty((n, num_bases), dtype=torch.float32, device=t.device)
    phi[:, 0] = 1.0
    if num_bases == 1:
        return phi
    phi[:, 1] = t
    for k in range(2, num_bases):
        phi[:, k] = 2.0 * t * phi[:, k - 1] - phi[:, k - 2]
    return phi


def ridge_solve(Phi: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Ridge regression: min_C ||Phi @ C - Y||^2 + lam ||C||^2.
    Phi: (n, p), Y: (n, d) -> C: (p, d).
    """
    n, p = Phi.shape
    if Y.dim() != 2 or Y.shape[0] != n:
        raise ValueError("Y must be (n, d) with n = Phi.shape[0]")
    Phi_f = Phi.float()
    Y_f = Y.float()
    reg = max(0.0, float(lam))
    if reg <= 0.0:
        reg = 1e-8
    A = Phi_f.T @ Phi_f + reg * torch.eye(p, device=Phi.device, dtype=torch.float32)
    B = Phi_f.T @ Y_f
    C = torch.linalg.solve(A, B)
    return C


def predict_from_coeffs(C: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """C: (p, d), x: scalar or (1,) -> prediction (d,)."""
    p = C.shape[0]
    row = chebyshev_design(x.reshape(1), p)  # (1, p)
    return (row @ C).squeeze(0)


def taylor_1_step(
    t_target: float,
    t_prev: float,
    t_last: float,
    y_prev: torch.Tensor,
    y_last: torch.Tensor,
) -> torch.Tensor:
    """First-order extrapolation: y_last + ((t_target - t_last) / (t_last - t_prev)) * (y_last - y_prev)."""
    dt = t_last - t_prev
    if abs(dt) < 1e-12:
        return y_last.clone()
    k = (t_target - t_last) / dt
    return y_last + k * (y_last - y_prev)


# -------- Per-branch state ----------------------------------------------------


@dataclass
class SpectrumState:
    """State for one sampling branch (e.g. cond vs uncond)."""

    times: List[float] = field(default_factory=list)
    outputs: List[torch.Tensor] = field(default_factory=list)
    coeffs: Optional[torch.Tensor] = None  # (p, d)
    step_index: int = 0
    last_time: Optional[float] = None
    last_shape: Optional[Tuple[int, ...]] = None
    last_dtype: Optional[torch.dtype] = None
    last_device: Optional[torch.device] = None
    curr_window: float = 2.0
    num_cached_steps: int = 0

    def reset(self, initial_window: float) -> None:
        self.times.clear()
        self.outputs.clear()
        self.coeffs = None
        self.step_index = 0
        self.last_time = None
        self.last_shape = None
        self.last_dtype = None
        self.last_device = None
        self.curr_window = initial_window
        self.num_cached_steps = 0


# -------- Decision: real step vs forecast -------------------------------------


def should_run_real_step(
    step_index: int,
    num_cached_steps: int,
    curr_window: float,
    window_size: int,
    flex_window: float,
    warmup_steps: int,
    cache_size: int,
) -> bool:
    """
    True if we must run the real denoiser this step.
    - Warmup: first `warmup_steps` steps always real.
    - Need at least 2 points to fit: if cache has < 2, run real.
    - Adaptive (flex_window > 0): real when (num_cached_steps + 1) % floor(curr_window) == 0.
    - Fixed (flex_window == 0): real when step_index % window_size == 0.
    """
    if cache_size < 2:
        return True
    if step_index < warmup_steps:
        return True
    if flex_window > 0:
        w = max(1, math.floor(curr_window))
        return (num_cached_steps + 1) % w == 0
    return step_index % max(1, window_size) == 0


# -------- Fit and forecast ----------------------------------------------------


def fit_and_forecast(
    state: SpectrumState,
    t_target: float,
    bounds: Optional[Tuple[float, float]],
    num_bases: int,
    lam: float,
    blend_w: float,
    use_taylor_blend: bool,
) -> Optional[torch.Tensor]:
    """
    If we have valid coeffs and cache, return blended prediction for t_target.
    Otherwise return None (caller should run real model).
    """
    if state.coeffs is None or len(state.outputs) == 0:
        return None

    last_out = state.outputs[-1]
    if bounds is not None:
        t_min, t_max = bounds
    else:
        t_min = min(state.times + [t_target])
        t_max = max(state.times + [t_target])

    x = map_to_chebyshev_domain(
        torch.tensor([t_target], device=last_out.device, dtype=torch.float32),
        t_min,
        t_max,
    )
    pred_flat = predict_from_coeffs(state.coeffs, x)
    pred = pred_flat.reshape(last_out.shape).to(dtype=last_out.dtype)

    if not torch.isfinite(pred).all():
        return None

    if blend_w >= 1.0:
        return pred
    if blend_w <= 0.0:
        return last_out

    if use_taylor_blend and len(state.outputs) >= 2 and len(state.times) >= 2:
        t_prev = state.times[-2]
        t_last = state.times[-1]
        base = taylor_1_step(t_target, t_prev, t_last, state.outputs[-2], last_out)
    else:
        base = last_out

    return blend_w * pred + (1.0 - blend_w) * base


def update_state_after_real(
    state: SpectrumState,
    t: float,
    y: torch.Tensor,
    bounds: Optional[Tuple[float, float]],
    num_bases: int,
    lam: float,
    flex_window: float,
    max_cache: int,
) -> None:
    """Append (t, y) to cache, optionally refit coeffs, advance window."""
    state.times.append(t)
    state.outputs.append(y.detach())
    state.num_cached_steps = 0
    if flex_window > 0:
        state.curr_window += flex_window

    if len(state.times) > max_cache:
        state.times = state.times[-max_cache:]
        state.outputs = state.outputs[-max_cache:]

    if len(state.times) < 2:
        state.coeffs = None
        return

    try:
        if bounds is not None:
            t_min, t_max = bounds
        else:
            t_min = min(state.times)
            t_max = max(state.times)
        ts = torch.tensor(state.times, device=y.device, dtype=torch.float32)
        x = map_to_chebyshev_domain(ts, t_min, t_max)
        Phi = chebyshev_design(x, num_bases)
        Y = torch.stack([o.float().reshape(-1) for o in state.outputs], dim=0)
        state.coeffs = ridge_solve(Phi, Y, lam)
    except Exception:
        state.coeffs = None
