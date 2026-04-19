# Gradient Buffer Snapshots

## What is saved
The gradient accumulation buffer (`global_feature_grads`) from Machine 1, saved as `.pt` files via `torch.save`.

## Shape
`(128, 4096)` — (minibatch_size, cnn_feature_dim), 524,288 float32 values per snapshot (~2 MB each).

## Location
`/workspace/runs/<run_name>/grad_buffers/iteration_<N>.pt`

## Frequency
Every 3 iterations. With 244 total iterations, expect ~81 snapshots (~162 MB).

## When in the training loop
Saved at the end of each iteration (after all 320 minibatch updates), before logging metrics. Represents the buffer state after the last minibatch's accumulation and sparsification (top 10% zeroed out).

## Loading
```python
import torch
buffer = torch.load("path/to/iteration_30.pt")  # shape: (128, 4096)
abs_vals = buffer.abs()
```
