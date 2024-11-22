#!/usr/bin/env python3
"""Quick MAE profiling script."""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import gc
from fml.model import MAELite, MAEConfig

def profile_mae(batch_size=32):
    # Clean start
    torch.cuda.empty_cache()
    gc.collect()

    model = MAELite(MAEConfig())
    model.cuda()
    model.train()
    model = torch.compile(model, mode='reduce-overhead')

    batch = torch.randn(batch_size, 3, 224, 224).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        profile_memory=True,
        record_shapes=True,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3)
    ) as prof:
        for _ in range(5):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                with record_function("forward"):
                    loss, pred, mask, _ = model(batch)

                with record_function("backward"):
                    loss.backward()

                with record_function("optimizer"):
                    optimizer.step()

            prof.step()

    print(f"\nProfile for batch_size={batch_size}:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))

    # Save trace
    prof.export_chrome_trace(f"mae_trace_bs{batch_size}.json")

    # Memory stats
    print(f"\nPeak memory stats for batch_size={batch_size}:")
    print(f"Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

if __name__ == "__main__":
    for bs in [128]:
        profile_mae(bs)
