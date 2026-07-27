# 03 — Diffusion model

_(Rung 3 — locked after rung 2 ships.)_

The generative rite of passage. Train a small denoising diffusion model (DDPM) to
generate images, then whittle the sampling.

- **baseline:** a DDPM with a small U-Net, trained on MNIST or CIFAR-10.
- **optimized axis:** cut sampling steps (DDIM / distillation) while holding quality.
- **deliverable:** steps + wall-clock per sample before vs after, with sample grids.

Full spec written when we get here.
