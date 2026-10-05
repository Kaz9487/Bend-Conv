#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
case_name=${1:-bus_640}
threads=${2:-1}
stem=$case_name
result=native
if [ "$threads" -ne 1 ]; then stem="${case_name}_cpu_${threads}t"; result="native_${threads}t"; fi
case "$case_name" in bus_320|bus_640|zidane_320|zidane_640) ;; *) echo "unsupported case" >&2; exit 2;; esac
"$CC" -std=c11 -O3 -march=native -ffp-contract=off "out/models/$stem.c" -lpthread -lm -o "out/models/$stem"
export BEND_OUTDIR="$PWD/out/results/$case_name/$result"
mkdir -p "$BEND_OUTDIR"
export BEND_DUMP=1
cp "out/models/$stem.c.build.json" "$BEND_OUTDIR/bend_build.json"
"$CC" --version > "$BEND_OUTDIR/compiler.txt"
"out/models/$stem" --threads "$threads" --gpu off
