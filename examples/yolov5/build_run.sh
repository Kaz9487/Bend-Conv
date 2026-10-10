#!/bin/sh
# Builds the compiled model and runs it on one case, saving every layer output.
set -eu
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
case_name=${1:-bus_640}
threads=${2:-1}
result=native
if [ "$threads" -ne 1 ]; then result="native_${threads}t"; fi
case "$case_name" in bus_320|bus_640|zidane_320|zidane_640) ;; *) echo "unsupported case" >&2; exit 2;; esac
"$CC" -std=c11 -O3 -march=native -ffp-contract=off out/models/yolov5n.c -lpthread -lm -o out/models/yolov5n
export YOLO_SIZE="${case_name##*_}"
export YOLO_INPUT="out/results/$case_name/input.bin"
export YOLO_OUTPUT="out/results/$case_name/$result"
export YOLO_LAYERS=1
mkdir -p "$YOLO_OUTPUT"
cp out/models/yolov5n.c.build.json "$YOLO_OUTPUT/bend_build.json"
"$CC" --version > "$YOLO_OUTPUT/compiler.txt"
out/models/yolov5n --threads "$threads" --gpu off | tee "$YOLO_OUTPUT/forward.txt"
# "forward N ms", with the layer copies of YOLO_LAYERS included
milliseconds=$(sed -n 's/^forward \([0-9]*\) ms$/\1/p' "$YOLO_OUTPUT/forward.txt" | tail -1)
printf '{"seconds":%s.%03d,"backend":"Bend native C; see build metadata for platform and compiler"}\n' "$((milliseconds / 1000))" "$((milliseconds % 1000))" > "$YOLO_OUTPUT/native_run.json"
