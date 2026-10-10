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

It prepares the PyTorch reference if missing, compiles the model, runs it, and compares every layer, the predictions and the detections. Results go to `out/results/<case>/`. One compiled program serves every case and any `--threads` value.

## The model

[yolov5n.bend](yolov5n.bend) is the network, block by block as in the official model. A block takes its weights by their PyTorch name:

```python
# Convolution with its folded batch normalization, then SiLU.
def conv(x: F.Tensor(),+name: String,stride: U32,padding: U32) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    weights : F.Tensor() <- F.take(name ++ ".conv.weight")
    bias : F.Tensor() <- F.take(name ++ ".conv.bias")
    F.Model.pure(F.Tensor(),F.conv2d(x,weights,bias,stride,padding,Act.silu()))
```

```sh
node scripts/bend_launcher.mjs examples/yolov5/yolov5n.bend -o out/models/yolov5n
YOLO_OUTPUT=out ./out/models/yolov5n --threads 4      # forward ... ms, and out/pred.npy
```

| Variable | Meaning | Default |
|---|---|---|
| `YOLO_SIZE` | Side of the square image, a multiple of 32 | 640 |
| `YOLO_INPUT` | Raw FP32 image `[1,3,size,size]` | `out/results/bus_640/input.bin` |
| `YOLO_OUTPUT` | Folder for `pred.npy` | `out/results/bus_640/native` |
| `YOLO_LAYERS` | When set, every layer output is saved too | unset |
| `YOLO_EXTRA` | Forward passes before the saved one | 0 |

## Files

| File | Role |
|---|---|
| [yolov5n.bend](yolov5n.bend) | The model and its program |
| [prepare_reference.py](prepare_reference.py) | Exports the PyTorch reference and the fused weights, as `out/weights/named/<name>.npy` |
| [numpy_yolo.py](numpy_yolo.py), [golden_model.py](golden_model.py), [image_pipeline.py](image_pipeline.py) | The NumPy model and image processing |
| [compare_results.py](compare_results.py) | The comparison and its thresholds |
| [build_run.sh](build_run.sh) | Builds the compiled model and runs it on one case |

Detection decoding is part of the example, not of the library. For repeated timing use the [backend benchmark](../../benchmarks/backends/README.md).
