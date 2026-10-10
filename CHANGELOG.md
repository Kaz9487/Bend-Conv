# Changelog

[English](CHANGELOG.md) | [繁體中文](CHANGELOG.zh-TW.md)

This file records the notable changes in each version of Stelliferous. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Before 1.0.0, the public API may still change between minor versions.

## [0.0.2] - 2026-10-11

Stelliferous 0.0.2 adds file reading and writing and model weight loading, and improves YOLOv5n inference performance. The main changes are

- NumPy `.npy` files can be read and written directly in Bend.
- A model can be defined completely as a Bend program that takes each layer's weights by name.
- YOLOv5n inference is faster. Single-thread latency drops from 153.7 ms to 110 ms, and four-thread latency from 54.8 ms to 43 ms.

From this version on, the project is renamed from **Bend-Conv** to **Stelliferous**, and moves to Bend v2.0.36.

### Upgrade notes

- **`conv2d_with` is renamed `conv2d_custom`**

  The call and its arguments are unchanged. Only the function name needs updating.

- **`activate` no longer supports custom functions**

  Passing a custom activation function to `activate` now returns a failed result. To use a custom activation, call `apply` or `conv2d_custom` instead.

- **The numerical results of some mathematical functions have changed**

  `sigmoid`, `silu`, `tanh` and `gelu_tanh` use a new algorithm that produces bit-for-bit identical results on different machines.

  The new outputs may differ from 0.0.1 in the last bits. If you used 0.0.1 outputs as a test baseline, regenerate your reference results.

### Added

- **File reading and writing**

  `load_npy(path)` and `save_npy(path, tensor)` read and write the NumPy `.npy` format.

  `load_raw(path, shape)` and `save_raw(path, tensor)` read and write raw FP32 data without a header.

- **Model weight management**

  `load_weights(folder)` loads every weight in a folder at once. While the model runs, `take(name)` takes the weights it needs by name, and `run(model, weights)` runs the whole model.

  For the other related functions, see the [API reference](lib/README.md#models).

- **Mathematical functions**

  `exp`, `log` and `softmax(tensor, axis)`.

- **Comparison and conditional selection**

  Six elementwise comparisons are added: `less`, `less_equal`, `greater`, `greater_equal`, `equal` and `not_equal`.

  The result is a tensor of 1.0 and 0.0, meaning the condition holds or does not hold at that position. All six have a `_scalar` form that compares with a single number.

  `select(mask, yes, no)` picks each element from one of two tensors according to a mask, and `clamp(tensor, low, high)` limits the elements of a tensor to a given range.

- **A complete YOLOv5n example**

  `examples/yolov5/yolov5n.bend` implements the whole YOLOv5n network as a single Bend program on the public API, for input images whose side is a multiple of 32.

  `examples/save_and_load.bend` shows how to save and load tensors.

### Performance

YOLOv5n single inference on a 640 × 640 input image, on the same test machine.

| Version | 1 thread | 4 threads |
| ----- | -------: | ------: |
| 0.0.1 | 153.7 ms | 54.8 ms |
| 0.0.2 |   110 ms |   43 ms |

For the full results, see the [performance document](docs/performance.md).

The improvement comes from the following changes.

- **Convolution runs in blocks**

  Inside a convolution, the input arrangement and the multiply-add work are cut into blocks of a fixed size that reuse one piece of working memory, instead of allocating and initializing one large temporary area for the whole layer. One thread and several threads use the same procedure.

- **Activations process eight elements at a time**

  `sigmoid`, `silu`, `tanh` and `gelu_tanh` are computed eight elements at a time, which compiles to SIMD instructions. This applies to an activation that follows a convolution and to one called on its own.

- **No calls to the system math library**

  In 0.0.1, `sigmoid`, `silu`, `tanh` and `gelu_tanh` went through the system's `exp` and `tanh`, converting to double precision and back to FP32. The new versions use only FP32 addition, subtraction, multiplication and division and 32-bit integer operations. They are faster, and their results no longer change with the version of the system library.

- **YOLOv5n detection decoding is tensor arithmetic**

  The decoding step uses the library's tensor operations and `sigmoid`, sharing one implementation with the rest of the network.

- **Updated cost estimates in the work planner**

  The planner estimates memory use from the blocked procedure, and counts the cost of each activation from its actual instructions, so that the division of work across threads follows the real cost more closely.

- **Shorter compile time for the model program**

  The YOLOv5n program generated from the official weights is now a list of operations run by a single loop. Compiling the 640 × 640 program takes about 14 seconds instead of about 6 minutes, and needs under 1 GB of memory instead of more than 12 GB.

### Formal proofs and numerical accuracy

- **New formal proofs**

  Elementwise semantics are proved for the six comparisons and for `sigmoid`, `silu`, `tanh`, `gelu_tanh`, `exp` and `log`.

  These proofs establish that every output element is the result of applying the given function to the corresponding input element. For details, see the [proofs document](docs/proofs.md).

- **Proofs for blocked convolution**

  The new blocked procedure is proved, for a single task and for parallel tasks. The unconditional theorem that native convolution matches the matrix specification for every input still holds.

- **Proofs for file reading and writing**

  Writing a value as its bits and reading it back is proved to give exactly the same bits, NaN payloads included.

- **Numerical error test**

  `sigmoid`, `silu`, `tanh`, `exp` and `log` were evaluated on all 2³² FP32 bit patterns, with the correctly rounded result computed by MPFR as the reference.

  The largest measured errors, in ULP (units in the last place), are below.

  | `sigmoid` | `silu` | `tanh` | `exp` | `log` |
  | --------: | -----: | -----: | ----: | ----: |
  |      1.81 |   1.80 |   1.06 |  0.80 |  0.52 |

  `benchmarks/checks/math_accuracy.py` reproduces the test.

### Known limitations

- **`gelu_tanh` is still inaccurate for negative inputs.** For inputs between -2 and -0.5 the largest error reaches 15 ULP. Below -2 the result is close to zero, and its reliability cannot be guaranteed yet.
- `select`, `softmax` and the other activation functions have only been tested against NumPy and have no formal proof yet.
- The numerical errors above come from measurement. There is no formal proof of an error bound.
- **`bend --verdict` cannot be used directly on this library at present.** Bend v2.0.36 stops before verification starts. The existing proofs are still verified by the official checker, with `python run.py proofs`.
- Only inference is supported at present. Model training and automatic differentiation are not implemented yet.
- Only the FP32 (F32) data type is supported.
- For convolutions with many weights and few output positions, and for a batch size above 1 or an output width below 6, more workers usually bring no clear performance gain. See the [design document](docs/design.md).
- Inference is still slower than PyTorch. For a detailed comparison, see the [performance document](docs/performance.md).

## [0.0.1] - 2026-10-05

The first public version of Stelliferous (then Bend-Conv), built on the official Bend v2.0.35.

### Added

- **FP32 tensor API**

  `lib/tensor_f32.bend` provides the basic tensor operations. Tensor shapes are checked at run time, and errors are carried by the returned result.

  The main features are

  - **Construction** — `zeros`, `ones`, `full`, `arange`, `from_list`, `from_storage`.
  - **Elementwise operations** — `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, with NumPy broadcasting rules and matching `_scalar` forms.
  - **Activation functions** — `relu`, `relu6`, `leaky_relu`, `sigmoid`, `silu`, `tanh`, `hard_sigmoid`, `hard_swish`, `gelu_tanh`, `activate`, and `apply` and `apply_indexed` for custom functions.
  - **Reductions** — `sum`, `mean`, `max`, `min`, and along a given axis `sum_axis`, `mean_axis`, `max_axis`, `min_axis`.
  - **Shape and layout operations** — `reshape`, `transpose`, `slice`, `set_slice`, `concat`, `pad`, `pad_edge`, `broadcast_to`.
  - **Convolution and window operations** — `conv2d`, `conv2d_with`, `max_pool2d`, `max_pool2d_with`, `upsample_nearest`.
  - **Reading values and checking state** — `to_list`, `item`, `to_list_or`, `item_or`, `check`, `describe`, `into_storage`.

- **FP32 convolution with deterministic results**

  The FP32 operations and the accumulation order are fixed, without FMA (fused multiply-add) or reassociation, so results are bit-for-bit identical for every worker count.

- **A work planner implemented in Bend**

  It decides how each convolution is divided into tasks from the tensor shapes and the worker count.

- **Row-band data layout**

  Row-band storage lets several workers share the work on a tensor without each copying the whole data.

  The layout is kept across convolution, elementwise operations, activation, channel concatenation (concat), upsampling and pooling, which avoids unnecessary copies.

- **Formal proofs**

  The theorems are verified by the unmodified official Bend checker. They include the unconditional theorem that native convolution matches the matrix specification for every input, and proofs of the numerical properties of the tensor operations.

  For details, see the [proofs document](docs/proofs.md).

- **YOLOv5n inference example**

  A Bend program generated from the official pretrained YOLOv5n network, compared numerically with PyTorch layer by layer.

- **Correctness tests**

  79 test programs on the public API compare their results bit for bit with NumPy.

  The convolution tests cover 1, 2, 4 and 8 threads and check native convolution for each worker count.

- **Bilingual documentation**

  Documentation in English and Traditional Chinese.

### Known limitations

- Only model inference is supported. Training and automatic differentiation are not implemented yet.
- Only the FP32 (F32) data type is supported.
- The library has no general model weight loading yet. The YOLO example uses its own file reading program.
- For convolutions with many weights and few output positions, and for a batch size above 1 or an output width below 6, more workers usually bring no clear performance gain. See the [design document](docs/design.md).
- Inference is still slower than PyTorch. For a detailed comparison, see the [performance document](docs/performance.md).
- Because Bend does not yet support default arguments, named arguments or function overloading, some APIs come in both a short and a full form.
