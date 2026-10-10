# Model profiling

[English](README.md) | [繁體中文](README.zh-TW.md)

Tools that measure YOLOv5n operation by operation. They run a generated form of the [YOLO example](../../examples/yolov5/README.md) with a host timer around every operation. Run the example first, then run these on Linux from the repository root. Each run is heavy: check free memory and run one at a time. Output goes under `out/` and replaces what is there.

```sh
# the measured program, once per thread count
python -B benchmarks/model/generate_yolo_graph.py --case bus_640 --threads 4
node scripts/bend_launcher.mjs out/models/bus_640_cpu_4t.bend -o out/models/bus_640_cpu_4t.c

python -B benchmarks/model/profile_operations.py --case bus_640 --threads 4
python -B benchmarks/model/profile_allocations.py --case bus_640 --threads 4
python -B benchmarks/model/profile_timeline.py --case bus_640 --threads 4 --output out/results/timeline_4t
python -B benchmarks/model/timeline_claims.py out/results/timeline_4t
python -B benchmarks/model/report_plans.py out/models/bus_640_cpu_4t.json --output out/results/plans_4t.json
python -B benchmarks/model/compare_forward.py --case bus_640 --baseline-models out/results/baseline/models --output out/results/forward_4t --threads 4
```

| Tool | Measures |
|---|---|
| [generate_yolo_graph.py](generate_yolo_graph.py), [yolo_graph.py](yolo_graph.py), [tensor_io.c](tensor_io.c) | Write the measured program: the operation graph, its Bend form, and the host timer and buffers |
| [profile_operations.py](profile_operations.py) | Time of each graph operation |
| [profile_allocations.py](profile_allocations.py) | Heap requests and Array block allocations of each operation |
| [profile_timeline.py](profile_timeline.py) | Which workers run which operation, and when |
| [timeline_claims.py](timeline_claims.py) | From a timeline: wall time by the number of busy workers |
| [report_plans.py](report_plans.py) | Executed handwritten convolution plans (`--handwritten --threads 1 2 4 8 --output out/results/plans.json`), or generated-model metadata replay |
| [compare_forward.py](compare_forward.py) | Forward time of a saved C of the hand-written model against the current one |

Every timing tool uses two warmups and seven samples in one process.

## Operation times

Groups time by operation kind and output size, reports ownership copies separately, and computes the dependency critical path. Results go to `out/results/model_operations/`.

## Allocations

Counts calls, payload bytes and rounded heap bytes of `blk_new`, `blk_copy`, `blk_node` and `blk_half`, plus generic heap requests. The counts are cumulative and include capacity padding; they are not live bytes or RSS. Timings are discarded. `--models` and `--model-stem` select a model in another folder.

## Timeline

Records grow, work and wake phases, ring claims by worker, dispatch segments and block-copy bytes.

- A claim says which worker holds a runtime ring, not which CPU ran.
- Wake, copy and dispatch intervals overlap; do not add them.
- A dispatch segment includes the C inlined into it.
- `--event-capacity` sets the trace size (48 bytes per event); the tool fails if the trace overflows.

`timeline_claims.py --kind conv` lists every operation of one kind.

## Planner report

[plan_observation.bend](plan_observation.bend) calls the library's plan selectors on the shapes of a generated graph. It runs no tensor data. `--operations RESULTS.json` adds the measured time of each operation from `profile_operations.py`.

## Forward comparison

Measures the hand-written [yolov5n.bend](../../examples/yolov5/yolov5n.bend), not the generated form. Compiles the saved C and the current C with the same flags, requires every layer output to be bitwise equal, then runs five alternating rounds and reports round medians and interquartile ranges. The baseline folder holds a copy of `out/models/yolov5n.c` and its build record; the output folder must be new or empty.
