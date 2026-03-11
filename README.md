# Spectrum — ComfyUI Diffusion Acceleration

[CVPR 2026] **Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration** — training-free speedup by forecasting denoiser outputs with Chebyshev polynomials and ridge regression.

- **Paper / code:** [github.com/hanjq17/Spectrum](https://github.com/hanjq17/Spectrum)
- **One node for image and video:** same MODEL patcher works with any ComfyUI sampler (KSampler, etc.) and any backbone (SDXL, Flux, SVD, etc.).

## Install

Clone into ComfyUI’s `custom_nodes` and restart:

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-Spectrum
```

## Usage

1. **Image:** Load your model → **Spectrum Patcher** → KSampler (or any sampler) → VAE decode → Save image.
2. **Video:** Load your video model → **Spectrum Patcher** → Video sampler → VAE decode → Save video.

No extra nodes. The patcher sits between the model and the sampler; it intercepts the UNet forward and replaces some steps with Chebyshev-forecasted outputs.

### Node: **Spectrum Patcher** (`model/patches`)

| Input | Description |
|-------|-------------|
| **model** | ComfyUI MODEL (image or video). |
| **w** | Blend: `w × Chebyshev + (1−w) × base`. 0.5–1.0 recommended; 1.0 = pure forecast. |
| **m** | Number of Chebyshev bases. Default 4. |
| **lam** | Ridge regularization. Default 0.1. |
| **window_size** | Initial window for adaptive schedule; or fixed “every Nth step” when flex_window=0. |
| **flex_window** | (Optional) After each real step, window grows by this amount. 0 = fixed schedule. 0.75 ≈ 3.5×, 3.0 ≈ 5×. |
| **warmup_steps** | (Optional) First N steps always run the real model. Default 1. |
| **blend_mode** | (Optional) `last_real` or `taylor` (paper-style linear extrapolation base). |

**Suggested (paper):** `window_size=2`, `flex_window=0.75`, `w=0.5` for ~3.5× speed; or `flex_window=3.0` for ~5×.

## Implementation

- **`spectrum_core.py`** — Chebyshev design matrix, ridge solve, adaptive step rule, Taylor extrapolation, per-branch state. No ComfyUI dependency.
- **`nodes.py`** — Single ComfyUI node that patches MODEL via `set_model_unet_function_wrapper`, using the core for “real vs forecast” and blend. Same path for image and video.

State is per sampling branch (cond/uncond) and resets when the run restarts or shape/dtype/device changes.
