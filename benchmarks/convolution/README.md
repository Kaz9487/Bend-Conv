# Convolution benchmark

[English](README.md) | [繁體中文](README.zh-TW.md)

Times single convolution shapes on Bend-Conv, a packed C reference and NumPy.

```sh
python run.py benchmark
```

Samples, the environment and the report go to `out/results/cpu_convolution/`.

## Method

- 1 and 4 threads; two warmups and seven samples per backend, in one process.
- Total latency covers packing, allocation, kernels and gather. Core latency covers the multiply-accumulate loop only. File I/O is not timed.
- The C reference follows the same packed algorithm and accumulation order. It is built twice, with and without automatic vectorization, and both builds must match Bend bit for bit.
- The C reference partitions channels and Bend partitions output positions, so the comparison isolates vectorization, not every cost of generated code.
- `BENCH_PYTHON` selects the Linux Python with NumPy and threadpoolctl, default `python3`.

## Files

| File | Role |
|---|---|
| [benchmark_cpu.py](benchmark_cpu.py) | Execution order, thread counts, samples and numerical comparison |
| [numpy_convolution.py](numpy_convolution.py) | The shapes, the seeded input and the pretrained weights |
| [packed_convolution.c](packed_convolution.c) | The C reference |
| [run_cpu.sh](run_cpu.sh) | Builds and runs the three backends |
| [report_plans.py](report_plans.py) | The planner's decisions, read from compiled Bend without timing a kernel |
| [profile_allocations.py](profile_allocations.py) | Counts allocation and copy requests, checks them against the planner's workspace and checks the output is unchanged. Its timings are discarded |
