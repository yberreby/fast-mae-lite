# `fast-mae-lite` = `fml`

A lightweight MAE-Lite port/implementation, **compatible with Python 3.11 and torch 2+**.

Written for my own use; released in case it's useful to others.

For now, this repository **only supports the tiny variant**, but supporting others should be straightforward. I just haven't needed them so far.

## Why?

I needed it :)


## Test inference from pretrained

```
uv run pytest  --verbose
```

## Single training run

```
uv run -m train.main
```

## Hyperparameter sweep

```
uv run python -m train.main --config-name sweep --multirun
```

## Attribution

This project contains code derived from [the original MAE-Lite repo](https://github.com/wangsr126/mae-lite), licensed under the Apache License 2.0.
We are grateful to the authors for their work, and for releasing the pretrained checkpoints that this repository is designed to work with.
