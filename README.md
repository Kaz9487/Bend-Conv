<div align="center">

<img alt="Stelliferous: a star cartographer and a lamp keeper under the night sky, per aspera ad astra" src="docs/banner.jpg" width="100%">

# Stelliferous

Tensors and convolution, written in [Bend](https://github.com/bendlang/bend) and proved in Bend.

[Getting started](docs/getting_started.md) | [API reference](lib/README.md) | [Proofs](docs/proofs.md) | [Performance](docs/performance.md)

[![Bend](https://img.shields.io/badge/Bend-v2.0.36-7c3aed)](scripts/toolchain.json)
[![Proofs](https://img.shields.io/badge/proofs-official%20checker-16a34a)](docs/proofs.md)
[![Results](https://img.shields.io/badge/results-bit--exact%20across%20threads-0ea5e9)](docs/design.md)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-64748b)](#license)

[English](README.md) | [繁體中文](README.zh-TW.md)

</div>

---

Stelliferous is a tensor library built on Bend 2, with the operations a convolutional network needs. It currently supports F32.

## Get started

Requires Node.js, [Bun](https://bun.sh), Python 3 and Clang, on Linux, macOS or WSL.

```sh
git clone --depth 1 --branch v2.0.36 https://github.com/bendlang/bend.git .tools/bend
```

## Example

[examples/array_basics.bend](examples/array_basics.bend):

```python
import Base
import ../lib/tensor_f32.bend as F
import ../lib/math_activation.bend as Act

def main() -> List<&2,F32>:
  # 1.0 .. 16.0 as [batch, channels, height, width]
  image   = F.reshape(F.arange(1.0,16),[1,1,4,4])
  # [output channels, input channels, kernel height, kernel width]
  weights = F.ones([1,1,3,3])
  # one value per output channel
  bias    = F.zeros([1])
  # stride 1, padding 1, then ReLU: the size stays 4x4
  feature = F.conv2d(image,weights,bias,1,1,Act.relu())
  # 2x2 windows: 4x4 -> 2x2
  pooled  = F.max_pool2d(feature,2)
  # the values, or [] if any step above failed
  F.to_list_or(pooled,[])
```

Compile it and run it. It prints `[54.0, 63.0, 90.0, 99.0]`.

```sh
node scripts/bend_launcher.mjs examples/array_basics.bend -o out/array_basics
./out/array_basics
```

More: [a small network](examples/small_network.bend), [saving and loading](examples/save_and_load.bend), [YOLOv5n](examples/yolov5/README.md).

## Proofs

For every input, the native convolution equals its matrix specification, bit for bit and for every worker count. Checked by the official Bend checker:

```sh
python run.py proofs
```

```text
law native_inference_matches_matrix:
  for +problem: ConvSpec.Problem
  {NativeConv.infer(problem) == MatrixSpec.infer(problem) : ConvSpec.ConvOutcome}
```

## Performance

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/performance_dark.svg">
  <img alt="YOLOv5n inference time: Stelliferous 0.0.1 153.7 and 54.8 ms, Stelliferous 0.0.2 110 and 43 ms, NumPy 126.9 and 113.3 ms, PyTorch 49.3 and 14.8 ms at 1 and 4 threads" src="docs/performance_light.svg" width="720">
</picture>

\* Model computation only. Image loading, preprocessing and non-maximum suppression are not included.

Measured on a Google Cloud `c4d-standard-8` virtual machine (AMD EPYC 9B45, 8 virtual CPUs). Method and samples are in [performance](docs/performance.md).

## Documentation

[Getting started](docs/getting_started.md) · [API reference](lib/README.md) · [Design](docs/design.md) · [Proofs](docs/proofs.md) · [Performance](docs/performance.md) · [Development](docs/development.md) · [Changelog](CHANGELOG.md)

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache License 2.0](LICENSE-APACHE), at your option. The bundled YOLOv5 weights and images keep their [upstream license](data/README.md).
