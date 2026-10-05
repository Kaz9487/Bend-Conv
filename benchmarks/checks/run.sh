#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
for entry in convolution scalar_convolution array_api array_api_zero_workspace; do
    "$CC" -O3 -march=native -ffp-contract=off -std=c11 "out/build/integration/$entry.c" -lpthread -lm -o "out/build/integration/$entry"
    "$py" benchmarks/checks/check_runtime.py --entry "$entry"
done
"$CC" -O3 -march=native -ffp-contract=off -std=c11 out/build/integration/band_layout.c -lpthread -lm -o out/build/integration/band_layout
"$py" -B benchmarks/checks/band_cases.py --run
