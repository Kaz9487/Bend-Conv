# Performance

[English](performance.md) | [繁體中文](performance.zh-TW.md)

YOLOv5n v7.0 forward, the bus image at 640 × 640, FP32, batch one. Two warmups and seven samples per setting; median in milliseconds.

| Backend | 1 thread | 4 threads |
|---|---:|---:|
| Bend-Conv | 271.7 | 151.4 |
| NumPy / OpenBLAS | 249.9 | 260.0 |
| PyTorch | 89.8 | 51.8 |

Measured on 2026-10-05 by the [Benchmark workflow](https://github.com/Kaz9487/Bend-Conv/actions/runs/37268568117) on a GitHub-hosted runner. Hosted runners differ from run to run, and CPU affinity and frequency are not pinned. Every setting passed the layer, prediction and detection comparison with the official reference.

## Reproduce

Start the Benchmark workflow from the Actions tab, or run it locally after the [YOLO example setup](../examples/yolov5/README.md):

```sh
python run.py backends      # the table above
python run.py benchmark     # single convolution shapes
```

Samples and the environment go to `out/results/backends/`; the report is `out/results/backend_comparison.md`. [Benchmarks](../benchmarks/README.md) lists the other workflows.

## Environment

- GitHub-hosted `ubuntu-24.04` runner: AMD EPYC 7763, 4 virtual CPUs (2 cores, 2 threads each).
- Bend v2.0.35 (official, unmodified); Clang 18.1.3 with `-O3 -march=native -ffp-contract=off -std=c11`.
- NumPy 2.4.6 with its bundled OpenBLAS; PyTorch 2.14.0 (CPU, eager, FP32), one inter-op thread.

## What is timed

The forward computation: allocation, ownership copies, packing, kernels, gather and detection decode. Loading, compilation, preprocessing, diagnostic output and non-maximum suppression are outside the timer.

## Raw samples

| Setting | Samples (ms) |
|---|---|
| Bend-Conv, 1 thread | 271.7, 272.1, 272.3, 271.7, 271.3, 271.4, 271.6 |
| Bend-Conv, 4 threads | 151.3, 151.5, 151.2, 151.2, 151.7, 151.7, 151.4 |
| NumPy, 1 thread | 248.4, 249.9, 250.6, 250.8, 248.1, 249.9, 249.9 |
| NumPy, 4 threads | 259.2, 264.2, 260.6, 260.6, 259.1, 260.0, 260.0 |
| PyTorch, 1 thread | 89.3, 89.1, 89.3, 90.3, 89.9, 90.5, 89.8 |
| PyTorch, 4 threads | 51.5, 51.8, 52.0, 54.3, 51.8, 51.4, 51.5 |

## Other convolution shapes

Some shapes gain little from more workers: many weights with few output positions, batches of more than one image, and outputs narrower than six positions. See [design](design.md#known-limits).
