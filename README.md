# `fast-mae-lite` = `fml`

A lightweight MAE-Lite implementation, compatible with Python 3.11 and torch 2+.

Tiny variant only for now, hardcoded.

This is not quite fast for now, but hopefully will become so as it gets optimized.

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
