"""
ComfyUI custom nodes: Spectrum diffusion acceleration.

Connects to native ComfyUI via MODEL patching. One patcher works for both
image and video: ComfyUI uses the same MODEL type and UNet forward contract
for both; the underlying checkpoint (SDXL, Flux, SVD, etc.) is irrelevant
to this node.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .spectrum_core import (
    SpectrumState,
    fit_and_forecast,
    schedule_bounds_from_cond,
    should_run_real_step,
    timestep_to_float,
    update_state_after_real,
)

# Defaults aligned with Spectrum paper (e.g. window_size=2, flex_window=0.75, w=0.5)
DEFAULT_W = 0.5
DEFAULT_M = 4
DEFAULT_LAM = 0.1
DEFAULT_WINDOW_SIZE = 2
DEFAULT_FLEX_WINDOW = 0.75
DEFAULT_WARMUP_STEPS = 1
EPS = 1e-7


def _cond_key(kwargs: dict) -> Tuple[int, ...]:
    """Stable key for cond/uncond branch (separate state per branch)."""
    try:
        cu = kwargs.get("cond_or_uncond", [])
        return tuple(int(x) for x in cu)
    except Exception:
        return ()


class SpectrumModelPatcher:
    """
    Patches a ComfyUI MODEL so that some diffusion steps use Chebyshev-forecasted
    denoiser output instead of running the full UNet. Drop between Load Model and
    your sampler (KSampler, etc.) for both image and video workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "w": ("FLOAT", {"default": DEFAULT_W, "min": 0.0, "max": 1.0, "step": 0.01}),
                "m": ("INT", {"default": DEFAULT_M, "min": 1, "max": 16, "step": 1}),
                "lam": ("FLOAT", {"default": DEFAULT_LAM, "min": 0.0, "max": 100.0, "step": 0.001}),
                "window_size": ("INT", {"default": DEFAULT_WINDOW_SIZE, "min": 1, "max": 32, "step": 1}),
            },
            "optional": {
                "flex_window": ("FLOAT", {"default": DEFAULT_FLEX_WINDOW, "min": 0.0, "max": 5.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": DEFAULT_WARMUP_STEPS, "min": 0, "max": 16, "step": 1}),
                "blend_mode": (["last_real", "taylor"], {"default": "last_real"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "model/patches"

    def patch_model(
        self,
        model,
        w: float,
        m: int,
        lam: float,
        window_size: int,
        flex_window: float | None = None,
        warmup_steps: int | None = None,
        blend_mode: str | None = None,
    ):
        model = model.clone()
        states: Dict[Tuple[int, ...], SpectrumState] = {}

        w_f = max(0.0, min(1.0, float(w)))
        num_bases = max(1, int(m))
        lam_f = max(0.0, float(lam))
        win = max(1, int(window_size))
        flex = max(0.0, float(flex_window if flex_window is not None else DEFAULT_FLEX_WINDOW))
        warmup = max(0, int(warmup_steps if warmup_steps is not None else DEFAULT_WARMUP_STEPS))
        use_taylor = (blend_mode or "last_real") == "taylor"
        max_cache = max(2, num_bases + 1)

        def get_state(key: Tuple[int, ...]) -> SpectrumState:
            if key not in states:
                states[key] = SpectrumState(curr_window=float(win))
            return states[key]

        def wrapper(model_fn, kwargs):
            x = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs.get("c", {})

            key = _cond_key(kwargs)
            st = get_state(key)

            t = timestep_to_float(timestep)
            dtype = x.dtype if torch.is_tensor(x) else None
            device = x.device if torch.is_tensor(x) else None
            shape = tuple(int(s) for s in x.shape) if torch.is_tensor(x) else None

            # New run or layout change -> reset
            if st.last_time is not None and t > st.last_time + EPS:
                st.reset(float(win))
            if shape is not None and st.last_shape is not None and shape != st.last_shape:
                st.reset(float(win))
            if dtype is not None and st.last_dtype is not None and dtype != st.last_dtype:
                st.reset(float(win))
            if device is not None and st.last_device is not None and device != st.last_device:
                st.reset(float(win))
            if shape is not None:
                st.last_shape = shape
            st.last_dtype = dtype
            st.last_device = device

            # Step index from schedule or counter
            opts = (c.get("transformer_options") or {}) if isinstance(c, dict) else {}
            sigmas = opts.get("sample_sigmas") or opts.get("sigmas")
            if torch.is_tensor(sigmas) and sigmas.numel() > 1:
                sigmas_flat = sigmas.detach().flatten()
                t0 = float(sigmas_flat[0].item())
                if st.last_time is not None and abs(t - t0) < 1e-8 and t > st.last_time + EPS:
                    st.reset(float(win))
                try:
                    target = torch.tensor([t], device=sigmas.device, dtype=sigmas.dtype)
                    st.step_index = int((sigmas_flat - target).abs().argmin().item())
                except Exception:
                    pass
            else:
                if st.last_time is None:
                    st.step_index = 0
                elif abs(t - st.last_time) > EPS:
                    st.step_index += 1

            st.last_time = t

            do_real = should_run_real_step(
                st.step_index,
                st.num_cached_steps,
                st.curr_window,
                win,
                flex,
                warmup,
                len(st.times),
            )

            if do_real:
                out = model_fn(x, timestep, **c)
                if not torch.is_tensor(out):
                    return out
                bounds = schedule_bounds_from_cond(c)
                update_state_after_real(st, t, out, bounds, num_bases, lam_f, flex, max_cache)
                return out

            st.num_cached_steps += 1

            bounds = schedule_bounds_from_cond(c)
            blended = fit_and_forecast(
                st, t, bounds, num_bases, lam_f, w_f, use_taylor
            )
            if blended is not None:
                return blended

            return model_fn(x, timestep, **c)

        model.set_model_unet_function_wrapper(wrapper)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "SpectrumModelPatcher": SpectrumModelPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumModelPatcher": "Spectrum Patcher",
}
