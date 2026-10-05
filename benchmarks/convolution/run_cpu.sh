#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
"$CC" -O3 -march=native -ffp-contract=off -std=c11 out/build/cpu_convolution/bend_convolution.c -lpthread -lm -o out/build/cpu_convolution/bend_convolution
"$CC" -O3 -march=native -ffp-contract=off -std=c11 benchmarks/convolution/packed_convolution.c -lpthread -lm -o out/build/cpu_convolution/packed_c
"$CC" -O3 -march=native -ffp-contract=off -fno-vectorize -fno-slp-vectorize -std=c11 benchmarks/convolution/packed_convolution.c -lpthread -lm -o out/build/cpu_convolution/packed_c_scalar
"$py" benchmarks/convolution/benchmark_cpu.py

"$py" benchmarks/convolution/profile_allocations.py
