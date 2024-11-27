from typing import List, Dict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR


import logging
from torch import nn


def create_param_groups(
    model: nn.Module,
    base_lr: float,
    decay_rate: float = 0.85,
    logger: logging.Logger | None = None,
) -> List[Dict]:
    """Create parameter groups with layer-wise learning rate decay for encoder blocks.

    All parameters default to base_lr except encoder blocks, which decay from
    deepest to shallowest layers.
    """
    used_params = set()
    param_groups = []

    # Handle encoder blocks with decay (deepest to shallowest)
    for i, block in enumerate(model.encoder.blocks):
        block_params = list(block.parameters())
        used_params.update(id(p) for p in block_params)

        # Deeper layers (early blocks) get lower LR
        lr = base_lr * (decay_rate**i)
        param_groups.append(
            {
                "params": block_params,
                "lr": lr,
                "group_name": f"encoder_block_{i}",  # For debugging
            }
        )

    # Everything else gets base LR
    remaining_params = []
    for p in model.parameters():
        if id(p) not in used_params:
            remaining_params.append(p)

    param_groups.append(
        {
            "params": remaining_params,
            "lr": base_lr,
            "group_name": "rest",  # For debugging
        }
    )

    # Log parameter group structure if logger provided
    if logger:
        total_params = 0
        for group in param_groups:
            params_in_group = sum(p.numel() for p in group["params"])
            total_params += params_in_group
            logger.info(
                f"Group {group['group_name']}: {len(group['params'])} tensors, "
                f"{params_in_group:,} params, lr={group['lr']:.2e}"
            )
        logger.info(f"Total parameters: {total_params:,}")

    return param_groups


def create_optimizer_and_scheduler(
    logger: logging.Logger, model: torch.nn.Module, cfg, warmup_steps: int
) -> tuple[Optimizer, LinearLR]:
    """Create optimizer with layer-wise decay and simple warmup scheduler."""

    if cfg.lr_layer_decay < 1.0:
        logger.info("Using layer-wise learning rate decay")
        param_groups = create_param_groups(
            model, base_lr=cfg.lr, decay_rate=cfg.lr_layer_decay, logger=logger
        )
    else:
        logger.info("Using global learning rate")
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

    scheduler = LinearLR(
        optimizer,
        start_factor=0.9,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    return optimizer, scheduler
