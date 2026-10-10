# Benchmarks

[English](README.md) | [繁體中文](README.zh-TW.md)

Run from the repository root, after [getting started](../docs/getting_started.md). Every workflow writes under `out/results/`.

| Command | What it does | Guide |
|---|---|---|
| `python run.py integration` | Native convolution cases, compared bit for bit | [convolution_cases.py](checks/convolution_cases.py) |
| `python run.py api` | Programs against the public API, compared with NumPy | [public_api_cases.py](checks/public_api_cases.py) |
| `python run.py benchmark` | Time of single convolution shapes | [convolution](convolution/README.md) |
| `python run.py backends` | YOLOv5n forward time on NumPy, Stelliferous and PyTorch | [backends](backends/README.md) |
| `python -B benchmarks/model/...` | Time, allocation and worker timeline per model operation | [model](model/README.md) |
| `python -B benchmarks/capture_baseline.py LABEL` | Collects the current `benchmark` and `backends` results into `out/results/baselines/LABEL/evidence.json` | [capture_baseline.py](capture_baseline.py) |
| `python -B benchmarks/checks/math_accuracy.py` | Largest error of `sigmoid`, `silu`, `tanh`, `gelu_tanh`, `exp` and `log` against MPFR, over every FP32 input. Linux, with MPFR installed | [math_accuracy.py](checks/math_accuracy.py) |
| `python -B benchmarks/checks/measure_arithmetic_dispatch.py` | A function stored in a record against one passed as a template, in the same loop. Linux, with `CC` set | [design](../docs/design.md#functions-in-hot-loops) |

Published numbers are in [performance](../docs/performance.md).
