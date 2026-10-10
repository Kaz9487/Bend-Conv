# Performance

[English](performance.md) | [繁體中文](performance.zh-TW.md)

YOLOv5n v7.0 forward, the bus image at 640 × 640, FP32, batch one. Two warmups and seven samples per setting; median in milliseconds.

| Backend | 1 thread | 4 threads |
|---|---:|---:|
| Stelliferous 0.0.2 | 110 | 43 |
| Stelliferous 0.0.1 | 153.7 | 54.8 |
| NumPy / OpenBLAS | 126.9 | 113.3 |
| PyTorch | 49.3 | 14.8 |

Measured on 2026-10-10 on a Google Cloud `c4d-standard-8` virtual machine. CPU affinity and frequency are not pinned. Every setting passed the layer, prediction and detection comparison with the official reference. Stelliferous 0.0.2 reports whole milliseconds. Stelliferous 0.0.1 is the released tree with its own Bend v2.0.35, measured in the same way on the same machine.

## Reproduce

After the [YOLO example setup](../examples/yolov5/README.md):

```sh
python run.py backends      # the table above
python run.py benchmark     # single convolution shapes
```

Samples and the environment go to `out/results/backends/`; the report is `out/results/backend_comparison.md`. The Benchmark workflow in the Actions tab runs the same comparison on a GitHub-hosted runner. [Benchmarks](../benchmarks/README.md) lists the other workflows.

## Environment

- Google Cloud `c4d-standard-8`, Ubuntu 24.04: AMD EPYC 9B45, 8 virtual CPUs (4 cores, 2 threads each).
- Bend v2.0.36 (official, unmodified); Clang 19.1.1 with `-O3 -march=native -ffp-contract=off -std=c11`.
- NumPy 2.4.6 with its bundled OpenBLAS; PyTorch 2.14.0 (CPU, eager, FP32), one inter-op thread.

## What is timed

The forward computation: allocation, ownership copies, packing, kernels, gather and detection decode. Loading, compilation, preprocessing, diagnostic output and non-maximum suppression are outside the timer.

## Samples

| Setting | Median (ms) | Min–max (ms) |
|---|---:|---:|
| Stelliferous 0.0.2, 1 thread | 110 | 110–111 |
| Stelliferous 0.0.2, 4 threads | 43 | 42–43 |
| Stelliferous 0.0.1, 1 thread | 153.7 | 153.4–154.4 |
| Stelliferous 0.0.1, 4 threads | 54.8 | 54.6–56.5 |
| NumPy, 1 thread | 126.9 | 126.6–127.7 |
| NumPy, 4 threads | 113.3 | 113.0–114.5 |
| PyTorch, 1 thread | 49.3 | 48.7–49.5 |
| PyTorch, 4 threads | 14.8 | 14.5–15.6 |

## Other convolution shapes

Some shapes gain little from more workers: many weights with few output positions, batches of more than one image, and outputs narrower than six positions. See [design](design.md#known-limits).
