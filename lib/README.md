# API reference

[English](README.md) | [繁體中文](README.zh-TW.md)

```python
import stelliferous@0.0.2.0/tensor_f32.bend as F
import stelliferous@0.0.2.0/math_activation.bend as Act
```

`F` is the whole public API. A shape is a `List<&2,U32>`. Axes count from zero. Tensors are `F.Tensor()`, F32 only. [Getting started](../docs/getting_started.md) shows complete programs.

Three rules apply to every function:

- **Arguments are used up.** An operation consumes its tensors and returns the result. Binary arithmetic reuses the left storage. Keep a tensor with `copy`.
- **A failure stays on the result.** Later operations skip their work and pass the first problem on. You inspect it when you read the value.
- **Order is fixed.** Every sum and every convolution adds in a stated order, so results do not depend on the worker count.

## Create

| Function | Result |
|---|---|
| `zeros(shape)`, `ones(shape)`, `full(shape, value)` | A tensor filled with one value |
| `arange(start, count)` | Rank 1: `start`, `start + 1`, ... (`count` elements) |
| `from_list(shape, values)` | Row-major values. Fails unless the list has exactly the shape's elements |
| `from_storage(shape, storage)` | Wraps a buffer from `into_storage` or from file I/O |

## Files

| Function | Result |
|---|---|
| `load_raw(path, shape)`, `save_raw(path, tensor)` | Little-endian FP32 values in row-major order. A file that cannot be read or does not match the shape gives a failed tensor |
| `load_npy(path)`, `save_npy(path, tensor)` | NumPy `.npy` files of FP32 values. The file carries the shape |

## Models

| Function | Result |
|---|---|
| `load_weights(folder)`, `save_weights(folder, weights)` | `folder/names.txt` lists one name per line. `folder/<name>.npy` holds its tensor |
| `take(name)` | A step of a `do F.Model<T>:` block that takes a tensor from the weights |
| `keep(name, tensor)` | A step that puts a copy in the weights for a later `take` |
| `run(model, weights)` | The model's result |
| `run_keeping(model, weights)` | The result with what remains of the weights |
| `copy_weights(weights)` | Two copies of the weights |

## Read

| Function | Result |
|---|---|
| `to_list_or(tensor, fallback)` | The values in row-major order, or `fallback` if the tensor failed |
| `item_or(tensor, fallback)` | The single value of a one-element tensor, or `fallback` |
| `to_list(tensor)`, `item(tensor)` | `Result`: `Done{values}` or `Fail{problem}` |
| `check(tensor)` | `Done{tensor}`, or `Fail{(tensor, problem)}` with the status cleared |
| `describe(problem)` | Text such as `add: shapes [2] and [3] do not match` |
| `shape(tensor)` | `(tensor, extents)`. Use with `then_shape` |
| `into_storage(tensor)`, `expect(tensor)`, `expect_storage(tensor)` | The buffer as a `Result`, or the tensor or buffer inside `IO`, reporting a failure |

A problem names the `F` function that failed and is one of: shapes that do not match, a shape that is invalid or exceeds the machine capacity, or an invalid argument for a shape.

## Keep and combine values

| Function | Result |
|---|---|
| `copy(tensor)` | `(original, copy)` |
| `then(pair, body)` | Passes both tensors of a pair to `body`: `F.then(F.copy(t), kept => copied => F.add(kept, F.relu(copied)))` |
| `then_shape(pair, body)` | The same for `shape`: `F.then_shape(F.shape(t), tensor => extents => F.reshape(tensor, extents))` |
| `with_workers(tensor, count)`, `workers(tensor)` | Sets or reads the worker count. Read the pool size once with `IO.thread_count()`. Results keep the largest count of their operands |

Bend destructures parameters and fields, not call results, which is why pairs go through `then`.

## Elementwise

| Function | Result |
|---|---|
| `add`, `sub`, `mul`, `div`, `maximum`, `minimum` `(left, right)` | `operation(left, right)` per element, with NumPy broadcasting |
| `add_scalar`, `sub_scalar`, `mul_scalar`, `div_scalar` `(tensor, value)` | `operation(element, value)` |
| `less`, `less_equal`, `greater`, `greater_equal` `(left, right)` | Elementwise 0/1 comparisons with binary broadcasting. Either NaN gives 0 |
| `equal`, `not_equal` `(left, right)` | Elementwise 0/1 equality or inequality. Signed zeros are equal and NaN is unequal to every value |
| `select(mask, yes, no)` | Select yes for nonzero masks (including NaN), otherwise no, preserving selected bits with broadcasting |
| `less_scalar(tensor, value)` | 1 where an element is less than the value, otherwise 0. NaN gives 0 |
| `less_equal_scalar(tensor, value)` | 1 where an element is less than or equal to the value, otherwise 0. NaN gives 0 |
| `greater_scalar(tensor, value)` | 1 where an element is greater than the value, otherwise 0. NaN gives 0 |
| `greater_equal_scalar(tensor, value)` | 1 where an element is greater than or equal to the value, otherwise 0. NaN gives 0 |
| `equal_scalar(tensor, value)` | 1 where an element equals the value, otherwise 0. NaN gives 0 and signed zeros are equal |
| `not_equal_scalar(tensor, value)` | 1 where an element differs from the value or either is NaN, otherwise 0 |
| `clamp(tensor, low, high)` | First `maximum(element, low)`, then `minimum(result, high)`, with the same NaN and signed-zero behavior |
| `broadcast_to(tensor, shape)` | The tensor repeated to `shape` |

Broadcasting aligns extents at the last axis. A missing axis counts as 1. On each axis the extents are equal or one is 1. `maximum` and `minimum` follow Base for NaN and signed zero, in left-right order. A zero divisor gives an infinity or NaN, not a failure.

## Activations

| Function | Notes |
|---|---|
| `relu`, `relu6`, `sigmoid`, `silu`, `tanh`, `hard_sigmoid`, `hard_swish`, `gelu_tanh` `(tensor)` | As in PyTorch. `gelu_tanh` is `approximate='tanh'` |
| `leaky_relu(tensor, slope)` | |
| `activate(tensor, activation)` | With a value of `Act`: `Act.identity()`, `Act.relu()`, `Act.leaky_relu(slope)`, ... |
| `apply(~operation, operations, tensor)` | A custom `F32 -> F32` function |
| `apply_indexed(~operation, operations, tensor)` | A custom `U32 -> F32 -> F32` function that also receives the row-major index |

A custom function is a named `def` (or a lambda) passed as a template with `~`, together with its scalar operations per element: one per arithmetic operation, comparison or select, about 37 for `F32.exp` and 82 for `F32.tanh`. Use `Act.unknown_operations()` when unsure. The count only helps the planner. Results never depend on it.

```python
def clipped(value: F32) -> F32:
  F32.min(Act.relu_value(value),6.0)

# F.apply(~clipped,4.0,tensor)
# F.conv2d_custom(~clipped,4.0,input,weights,bias,1,1)
```

## Reduce

| Function | Result |
|---|---|
| `sum`, `mean`, `max`, `min` `(tensor)` | A rank-0 tensor over every element in row-major order |
| `sum_axis`, `mean_axis`, `max_axis`, `min_axis` `(tensor, axis)` | The axis is removed |

Each result takes its elements in increasing index, starting from `0`, `-inf` or `+inf`. A mean divides the sum by the count. The order is fixed, so a sum equals an explicit loop and can differ from NumPy's pairwise `sum` in the last bits. `mean` of an empty tensor is NaN.

## Layout

| Function | Result |
|---|---|
| `exp(tensor)` | Elementwise natural exponential, with gradual underflow |
| `log(tensor)` | Elementwise natural logarithm. Zero gives −∞ and negatives give NaN |
| `softmax(tensor, axis)` | Normalize exponentials along an axis after subtracting its maximum |
| `reshape(tensor, shape)` | Same elements, new shape. No copy |
| `transpose(tensor, order)` | Axes in the given order |
| `slice(tensor, axis, start, count, step)` | `count` elements along `axis` from `start`, every `step`. `step` is positive |
| `set_slice(target, axis, start, part)` | `target` with `part` written along `axis` from `start` |
| `concat(parts, axis)` | Parts joined along `axis`: `F.concat(F.cons(a, F.cons(b, F.single(c))), 1)` |
| `pad(tensor, before, after, value)` | `before[i]` and `after[i]` new elements on axis `i`, equal to `value` |
| `pad_edge(tensor, before, after)` | The same, repeating the nearest element |

## Windows

Tensors are `[channels, height, width]` or `[batch, channels, height, width]`.

| Function | Result |
|---|---|
| `conv2d(input, weights, bias, stride, padding, activation)` | Weights are `[out_channels, in_channels, kernel_height, kernel_width]`, bias `[out_channels]`. The same stride and padding on both axes. The activation is applied to the output |
| `conv2d_custom(~activation, operations, input, weights, bias, stride, padding)` | With a custom activation function |
| `max_pool2d(tensor, kernel)` | Windows of `kernel` by `kernel` that do not overlap (stride `kernel`, padding 0) |
| `max_pool2d_with(tensor, kernel, stride, padding)` | The full form. Padding counts as `-inf` |
| `upsample_nearest(tensor, scale)` | Each row and column repeated `scale` times |

Each convolution output starts at +0.0, adds products in input-channel, kernel-row, kernel-column order, then adds the bias. [Design](../docs/design.md) explains how it runs and plans. [proofs](../docs/proofs.md) says what is proved.

## Inside the library

Application code does not need these. They are listed for readers of the source.

| Module | Role |
|---|---|
| [tensor.bend](tensor.bend), [tensor_layout.bend](tensor_layout.bend), [tensor_window.bend](tensor_window.bend), [tensor_convolution.bend](tensor_convolution.bend) | The generic tensor behind `F`: any element type, dense and band storage |
| `tensor_band*.bend` | Row-band storage and its operations |
| [tensor_operations.bend](tensor_operations.bend) | Checked `copy`, `combine`, `apply`, `reduce`, `scatter`, `sweep` over views. Both `Accepted` and `Rejected` return every owner |
| [tensor_view.bend](tensor_view.bend), [tensor_shape.bend](tensor_shape.bend), [tensor_cursor.bend](tensor_cursor.bend), [tensor_access.bend](tensor_access.bend) | Views (extents, strides, offset), checked element counts, the strided cursor |
| `kernel_*.bend` | The convolution kernel: weights, packing, GEMM, gather, planning, cost |
| [storage_buffer.bend](storage_buffer.bend), [storage_allocation.bend](storage_allocation.bend), `storage_*.bend` | `Buffer`: an owned array with its logical length and a storage certificate |
| `traversal_*.bend` | The loops: copy, update, fold, indexed output, partition tree |
| [math_activation.bend](math_activation.bend), [math_fp32.bend](math_fp32.bend), [math_sequence.bend](math_sequence.bend), [ordered_matrix.bend](ordered_matrix.bend) | Activation values and scalar functions. The ordered sequences and matrix products the proofs speak about |
| [machine_limits.bend](machine_limits.bend), [kernel_cost.bend](kernel_cost.bend) and their `default_*` files | Machine bounds used by proofs. Planner cost parameters |
| [proofs/](proofs/) | The library's proofs |
