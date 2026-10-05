#!/bin/sh
set -eu
mode=${1:-all}
case "$mode" in all|libraries) ;; *) echo 'Use all or libraries' >&2; exit 2;; esac
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
mkdir -p out/results/backends/c_naive out/results/backends/bend_cpu
if [ "$mode" = all ]; then
    "$CC" -O3 -march=native -ffp-contract=off -Iout/backends benchmarks/backends/naive_cpu.c -lm -o out/backends/naive_c
fi
"$py" benchmarks/backends/bench_numpy.py
BENCH_THREADS=4 "$py" benchmarks/backends/bench_numpy.py
if [ "$mode" = all ]; then out/backends/naive_c out/results/backends/c_naive; fi
sh benchmarks/backends/run_bend.sh cpu
sh benchmarks/backends/run_bend.sh cpu 4
"$py" benchmarks/backends/bench_torch.py --device cpu --threads 1
"$py" benchmarks/backends/bench_torch.py --device cpu --threads 4
lscpu > out/results/backends/cpu.txt
if [ "$mode" = all ]; then /usr/lib/wsl/lib/nvidia-smi > out/results/backends/gpu.txt; fi
"$py" benchmarks/backends/record_env.py
