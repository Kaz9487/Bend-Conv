# Performance

[English](performance.md) | [繁體中文](performance.zh-TW.md)

YOLOv5n v7.0 forward, the bus image at 640 × 640, FP32, batch one. Two warmups and seven samples per setting; median in milliseconds.

| Backend | 1 thread | 4 threads |
|---|---:|---:|
| Bend-Conv | 245.3 | 116.2 |
| NumPy / OpenBLAS | 223.0 | 232.1 |
| PyTorch | 92.7 | 45.3 |

Measured on 2026-10-04 on a machine that was running other work; CPU affinity and frequency were not pinned. Every setting passed the layer, prediction and detection comparison with the official reference.

## Reproduce

After the [YOLO example setup](../examples/yolov5/README.md):

```sh
python run.py backends      # the table above
python run.py benchmark     # single convolution shapes
```

Samples and the environment go to `out/results/backends/`; the report is `out/results/backend_comparison.md`. [Benchmarks](../benchmarks/README.md) lists the other workflows.

## Environment

- CPU: 12th Gen Intel Core i5-12500H. Linux under WSL.
- Bend v2.0.35 (official, unmodified); Clang 19.1.7 with `-O3 -march=native -ffp-contract=off -std=c11`.
- NumPy 2.4.6 with its bundled OpenBLAS; PyTorch 2.14.0 (CPU, eager, FP32), one inter-op thread.

## What is timed

The forward computation: allocation, ownership copies, packing, kernels, gather and detection decode. Loading, compilation, preprocessing, diagnostic output and non-maximum suppression are outside the timer.

## Raw samples

| Setting | Samples (ms) |
|---|---|
| Bend-Conv, 1 thread | 231.7, 251.6, 245.1, 245.3, 245.4, 230.8, 247.1 |
| Bend-Conv, 4 threads | 116.2, 156.9, 117.7, 138.9, 112.6, 105.3, 114.3 |
| NumPy, 1 thread | 226.2, 224.3, 215.6, 205.6, 211.4, 223.0, 242.3 |
| NumPy, 4 threads | 232.3, 217.0, 232.1, 248.9, 236.0, 221.8, 217.3 |
| PyTorch, 1 thread | 97.8, 84.4, 92.7, 82.9, 93.7, 85.0, 132.3 |
| PyTorch, 4 threads | 61.1, 41.3, 45.3, 54.6, 49.7, 38.8, 39.4 |

## Other convolution shapes

Some shapes gain little from more workers: many weights with few output positions, batches of more than one image, and outputs narrower than six positions. See [design](design.md#known-limits).
