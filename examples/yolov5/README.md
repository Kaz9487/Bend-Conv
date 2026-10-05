# YOLOv5n example

[English](README.md) | [繁體中文](README.zh-TW.md)

Runs the official pretrained FP32 YOLOv5n on the library and compares it with PyTorch. The bundled weights and images have their own [sources and licenses](../../data/README.md).

## Set up

After the [library setup](../../docs/getting_started.md):

```sh
git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git .tools/yolov5-official
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements-lock.txt
```

On Linux, OpenCV needs libGL (`apt install libgl1` on Debian or Ubuntu).

## Run

```sh
python run.py model --case bus_640 --threads 4
```

It prepares the PyTorch reference if missing, generates the Bend program, compiles it, runs it, and compares every layer, the predictions and the detections. Results go to `out/results/<case>/`. One compiled program serves any `--threads` value.

## Files

| File | Role |
|---|---|
| [yolo_graph.py](yolo_graph.py) | The operation graph of the model |
| [generate_yolo_graph.py](generate_yolo_graph.py) | Writes the graph as a Bend program that calls [tensor_f32.bend](../../lib/tensor_f32.bend). A tensor's last use takes it with `Tensor.take`; earlier uses read it with `Tensor.get` |
| [tensor_io.c](tensor_io.c) | Loading, snapshots and timing |
| [prepare_reference.py](prepare_reference.py) | Exports the PyTorch reference and the fused weights |
| [numpy_yolo.py](numpy_yolo.py), [golden_model.py](golden_model.py), [image_pipeline.py](image_pipeline.py) | The NumPy model and image processing |
| [compare_results.py](compare_results.py) | The comparison and its thresholds |
| [build_run.sh](build_run.sh) | Compiles and runs the generated program |

Detection decoding is part of the example, not of the library. For repeated timing use the [backend benchmark](../../benchmarks/backends/README.md).
