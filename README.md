# `fast-mae-lite` = `fml`

A lightweight MAE-Lite port/implementation, **compatible with Python 3.11 and torch 2+**.

Written for my own use; released in case it's useful to others.

For now, this repository **only supports the tiny variant**, but supporting others should be straightforward. I just haven't needed them so far.

## Why?

I needed it :)

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
