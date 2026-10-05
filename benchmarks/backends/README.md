# Full-model backend comparison

[English](README.md) | [繁體中文](README.zh-TW.md)

YOLOv5n forward time on NumPy, Bend-Conv and PyTorch. Set up the [YOLO example](../../examples/yolov5/README.md) first.

```sh
python run.py backends            # CPU
python run.py backends --cuda     # also the handwritten C and CUDA programs
```

Results go to `out/results/backends/`; the report is `out/results/backend_comparison.md`.

## Method

- Each backend runs two warmups and seven samples in one process.
- Timed: allocation, packing, kernels, gather and decode. Not timed: loading, compilation, preprocessing, diagnostic dumps and NMS.
- Bend reloads its input before each timed forward; its allocator and workers stay alive.
- CPU affinity and frequency are not pinned. GPU timings synchronize before reading the clock.
- Numerical validation runs separately; a missing result fails the run.

## Settings

| Variable | Meaning |
|---|---|
| `BENCH_PYTHON` | The Linux interpreter, default `python3`. It needs NumPy, PyTorch and threadpoolctl |
| `BEND_REPETITIONS` | Forwards per Bend process, default one. The benchmark sets nine |

If the interpreter's environment is read-only, install a missing package into `out/python_packages`, which the launchers add to `PYTHONPATH`:

```sh
python -m pip install --target out/python_packages threadpoolctl
```

## Files

| File | Role |
|---|---|
| [generate_graph.py](generate_graph.py) | Exports the model metadata every backend reads |
| [bench_numpy.py](bench_numpy.py), [bench_torch.py](bench_torch.py), [run_bend.sh](run_bend.sh) | Run and sample one backend |
| [run_cpu.sh](run_cpu.sh) | Selects the CPU cases and records the environment |
| [run_cuda.sh](run_cuda.sh), [probe_cuda.py](probe_cuda.py) | The CUDA capability check and backends |
| [validate.py](validate.py) | Numerical thresholds; `--backends` selects which results are required |
| [write_report.py](write_report.py) | The report, in both languages |
