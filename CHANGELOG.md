# Changelog

[English](CHANGELOG.md) | [繁體中文](CHANGELOG.zh-TW.md)

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0.0 the public API may change between minor versions.

## [0.0.1] - Unreleased

First public release. Requires the official Bend v2.0.35.

### Added

- **F32 tensor API** (`lib/tensor_f32.bend`), with shapes checked at run time and failures carried by the result:
  - construction: `zeros`, `ones`, `full`, `arange`, `from_list`, `from_storage`;
  - elementwise with NumPy broadcasting: `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, and the `_scalar` forms;
  - activations: `relu`, `relu6`, `leaky_relu`, `sigmoid`, `silu`, `tanh`, `hard_sigmoid`, `hard_swish`, `gelu_tanh`, `activate`, and custom functions through `apply` and `apply_indexed`;
  - reductions: `sum`, `mean`, `max`, `min`, and `sum_axis`, `mean_axis`, `max_axis`, `min_axis`;
  - layout: `reshape`, `transpose`, `slice`, `set_slice`, `concat`, `pad`, `pad_edge`, `broadcast_to`;
  - windows: `conv2d`, `conv2d_with`, `max_pool2d`, `max_pool2d_with`, `upsample_nearest`;
  - reading: `to_list`, `item`, `to_list_or`, `item_or`, `check`, `describe`, `into_storage`.
- **Convolution in a fixed FP32 order**, without fused multiply-add or reassociation: results are bit-for-bit equal for every worker count.
- **A planner written in Bend** that chooses how many tasks a convolution uses from the shapes and the worker count.
- **Row-band storage** that lets several workers share a tensor without cloning it, kept across convolution, elementwise operations, activation, channel concat, upsampling and pooling.
- **Proofs checked by the unmodified official Bend checker**: the unconditional theorem that native convolution equals the matrix specification for every input, and value theorems for the tensor operations. See [docs/proofs.md](docs/proofs.md).
- **YOLOv5n example**: the official pretrained network generated as a Bend program, compared with PyTorch layer by layer.
- **Checks**: 79 programs against the public API compared bit for bit with NumPy, and native convolution cases on 1, 2, 4 and 8 threads.
- Documentation in English and Traditional Chinese.

### Known limitations

- Inference only: no training, no automatic differentiation.
- F32 only.
- No loader for model weights in the library; the YOLO example has its own.
- Convolution with many weights and few output positions, batches of more than one image, and outputs narrower than six positions gain little or nothing from more workers. See [docs/design.md](docs/design.md).
- Slower than PyTorch. See [docs/performance.md](docs/performance.md).
- Bend has no default arguments, named arguments or overloading, so some functions come in a short and a full form.
