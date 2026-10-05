#!/bin/sh
# One build, validation and sampling protocol for both official native backends.
set -eu
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
backend=${1:-cpu}
threads=${2:-1}
case "$threads" in ''|*[!0-9]*) echo 'Threads must be a positive integer' >&2; exit 2;; esac
if [ "$threads" -lt 1 ]; then echo 'Threads must be positive' >&2; exit 2; fi
label=$backend
set --
case "$backend" in
  cpu)
    source=out/models/bus_640.c
    if [ "$threads" -ne 1 ]; then
      label="cpu_${threads}t"
      source="out/models/bus_640_$label.c"
    fi
    gpu=off
    ;;
  cuda)
    if [ "$threads" -ne 1 ]; then echo 'CUDA comparison uses one host thread' >&2; exit 2; fi
    source=out/backends/bend_cuda.c
    gpu=on
    site=$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$site/nvidia/cu13/lib:/usr/lib/wsl/lib"
    set -- -DBEND_CUDA=1 -I"$site/nvidia/cu13/include" -L/usr/lib/wsl/lib -l:libcuda.so.1 -L"$site/nvidia/cu13/lib" -l:libnvrtc.so.13
    ;;
  *) echo 'Use cpu or cuda' >&2; exit 2;;
esac
binary="out/backends/bend_$label"
export BEND_OUTDIR="$PWD/out/results/backends/bend_$label"
mkdir -p "$BEND_OUTDIR"
"$CC" -std=c11 -O3 -march=native -ffp-contract=off "$source" -lpthread -lm "$@" -o "$binary"
cp "$source.build.json" "$BEND_OUTDIR/bend_build.json"
rm -f "$BEND_OUTDIR/timing.json"
export BEND_DUMP=1
export BEND_REPETITIONS=1
set +e
"$binary" --threads "$threads" --gpu "$gpu" > "$BEND_OUTDIR/validation.log" 2> "$BEND_OUTDIR/stderr.log"
status=$?
set -e
printf '%s\n' "$status" > "$BEND_OUTDIR/exit_code.txt"
if [ "$status" -ne 0 ]; then
  cat "$BEND_OUTDIR/stderr.log"
  # An unavailable GPU is reported without a fallback or invented timing.
  if [ "$backend" = cuda ]; then exit 0; fi
  exit "$status"
fi
export BEND_DUMP=0
export BEND_REPETITIONS=9
"$binary" --threads "$threads" --gpu "$gpu" > "$BEND_OUTDIR/samples.log"
"$py" benchmarks/backends/collect_bend.py --backend "$backend" --threads "$threads"
