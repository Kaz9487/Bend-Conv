#!/bin/sh
# CUDA capability probe plus handwritten, PyTorch and Bend CUDA runs.
set -eu
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
site=$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$site/nvidia/cu13/lib:/usr/lib/wsl/lib"
"$py" benchmarks/backends/probe_cuda.py
mkdir -p out/results/backends/cuda_naive
"$CC" -O3 -march=native -ffp-contract=off -Iout/backends -I"$site/nvidia/cu13/include" benchmarks/backends/cuda_host.c -L/usr/lib/wsl/lib -l:libcuda.so.1 -L"$site/nvidia/cu13/lib" -l:libnvrtc.so.13 -o out/backends/naive_cuda
out/backends/naive_cuda out/results/backends/cuda_naive
"$py" benchmarks/backends/bench_torch.py --device cuda --threads 1
sh benchmarks/backends/run_bend.sh cuda
