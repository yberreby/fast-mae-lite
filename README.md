# `fast-mae-lite` = `fml`

A lightweight MAE-Lite port/implementation, **compatible with Python 3.11 and torch 2+**.

Written for my own use; released in case it's useful to others.

For now, this repository **has only been tested with the tiny variant**, but handling others should be very straightforward.

## Why?

I needed:

- To run and fine-tune a pretrained MAE-Lite under modern Python and Torch versions,
- To _recover color_. In the original pretrained models, there is information loss due to patch-local normalization.
  The training code in this repository fine-tunes a pretrained model to recover
  color (and also predict patches that are seen as input - a rather trivial
  operation, but convenient for some applications). Few gradient descent steps
  are needed to do so.

This is released as-is, it is likely that there will be breakages, incompatibilities, etc. Please open an issue - or better yet, a pull request - if you encounter any.

## Is it fast?

Not terrible, not insanely fast yet.

Current peak speed on a laptop with 16GB RAM, RTX 4060M (8GB VRAM), and a Ryzen 9 8945H CPU is about 2k img/s, seems to be be bottlenecked by data loading / preprocessing.

Will see how fast we can get this to train on ImageNet on a single GPU.

## Prerequisites

- Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.
- For inference/fine-tuning from pretrained: download the original [MAE-Tiny checkpoint](https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view?usp=sharing) to `ckpt/mae_tiny_400e.pth.tar`.
- To try out fine-tuning on color recovery: download `imagenette` to `~/datasets/imagenette` (or elsewhere, in which case, override the corresponding path in the config).

## Test inference from pretrained

```
uv run pytest --verbose
```

The image that will be saved _should_ look odd, but have recognizable and meaningful shapes.
The odd appearance is because the original pretrained MAE-Lite was trained to
predict _locally-normalized patches_, and the loss was masked out of patches
that are fed as input (since predicting them is trivial).

## Single training run

```
uv run -m train.main
```

Recommendations:

- Set `compile=false` for quick iteration.
- Be mindful of available system RAM when setting workers.
- Likely to be CPU- or disk-bound, not GPU-bound.
- You probably don't _need_ warmup for this task.

## Hyperparameter sweep

```
uv run python -m train.main --config-name sweep --multirun
```

## Attribution

This project contains code derived from [the original MAE-Lite repo](https://github.com/wangsr126/mae-lite), licensed under the Apache License 2.0.
We are grateful to the authors for their work, and for releasing the pretrained checkpoints that this repository is designed to work with.
