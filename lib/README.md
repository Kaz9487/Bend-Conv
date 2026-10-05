# API reference

[English](README.md) | [繁體中文](README.zh-TW.md)

```python
import lib/tensor_f32.bend as F
import lib/math_activation.bend as Act
```

`F` is the whole public API. A shape is a `List<&2,U32>`; axes count from zero. Tensors are `F.Tensor()`, F32 only. [Getting started](../docs/getting_started.md) shows complete programs.

Three rules apply to every function:

- **Arguments are used up.** An operation consumes its tensors and returns the result. Binary arithmetic reuses the left storage. Keep a tensor with `copy`.
- **A failure stays on the result.** Later operations skip their work and pass the first problem on. You inspect it when you read the value.
- **Order is fixed.** Every sum and every convolution adds in a stated order, so results do not depend on the worker count.

## Create

| Function | Result |
|---|---|
| `zeros(shape)`, `ones(shape)`, `full(shape, value)` | A tensor filled with one value |
| `arange(start, count)` | Rank 1: `start`, `start + 1`, ... (`count` elements) |
| `from_list(shape, values)` | Row-major values; fails unless the list has exactly the shape's elements |
| `from_storage(shape, storage)` | Wraps a buffer from `into_storage` or from file I/O |

## Read

| Function | Result |
|---|---|
| `to_list_or(tensor, fallback)` | The values in row-major order, or `fallback` if the tensor failed |
| `item_or(tensor, fallback)` | The single value of a one-element tensor, or `fallback` |
| `to_list(tensor)`, `item(tensor)` | `Result`: `Done{values}` or `Fail{problem}` |
| `check(tensor)` | `Done{tensor}`, or `Fail{(tensor, problem)}` with the status cleared |
| `describe(problem)` | Text such as `add: shapes [2] and [3] do not match` |
| `shape(tensor)` | `(tensor, extents)`; use with `then_shape` |
| `into_storage(tensor)`, `expect(tensor)`, `expect_storage(tensor)` | The buffer as a `Result`; or the tensor or buffer inside `IO`, reporting a failure |

A problem names the `F` function that failed and is one of: shapes that do not match, a shape that is invalid or exceeds the machine capacity, or an invalid argument for a shape.

## Keep and combine values

| Function | Result |
|---|---|
| `copy(tensor)` | `(original, copy)` |
| `then(pair, body)` | Passes both tensors of a pair to `body`: `F.then(F.copy(t), kept => copied => F.add(kept, F.relu(copied)))` |
| `then_shape(pair, body)` | The same for `shape`: `F.then_shape(F.shape(t), tensor => extents => F.reshape(tensor, extents))` |
| `with_workers(tensor, count)`, `workers(tensor)` | Sets or reads the worker count. Read the pool size once with `IO.thread_count()`; results keep the largest count of their operands |

Bend destructures parameters and fields, not call results, which is why pairs go through `then`.

## Elementwise

| Function | Result |
|---|---|
| `add`, `sub`, `mul`, `div`, `maximum`, `minimum` `(left, right)` | `operation(left, right)` per element, with NumPy broadcasting |
| `add_scalar`, `sub_scalar`, `mul_scalar`, `div_scalar` `(tensor, value)` | `operation(element, value)` |
| `broadcast_to(tensor, shape)` | The tensor repeated to `shape` |

Broadcasting aligns extents at the last axis; a missing axis counts as 1; on each axis the extents are equal or one is 1. `maximum` and `minimum` follow Base for NaN and signed zero, in left-right order. A zero divisor gives an infinity or NaN, not a failure.

## Activations

| Function | Notes |
|---|---|
| `relu`, `relu6`, `sigmoid`, `silu`, `tanh`, `hard_sigmoid`, `hard_swish`, `gelu_tanh` `(tensor)` | As in PyTorch; `gelu_tanh` is `approximate='tanh'` |
| `leaky_relu(tensor, slope)` | |
| `activate(tensor, activation)` | With a value of `Act`: `Act.identity()`, `Act.relu()`, `Act.leaky_relu(slope)`, ... |
| `apply(~operation, operations, tensor)` | A custom `F32 -> F32` function |
| `apply_indexed(~operation, operations, tensor)` | A custom `U32 -> F32 -> F32` function that also receives the row-major index |

A custom function is a named `def` (or a lambda) passed as a template with `~`, together with its scalar operations per element: one per arithmetic operation, comparison or select, about 37 for `F32.exp` and 82 for `F32.tanh`. Use `Act.unknown_operations()` when unsure. The count only helps the planner; results never depend on it.

```python
def clipped(value: F32) -> F32:
  F32.min(Act.relu_value(value),6.0)

# F.apply(~clipped,4.0,tensor)
# F.conv2d_with(~clipped,4.0,input,weights,bias,1,1)
```

## Reduce

| Function | Result |
|---|---|
| `sum`, `mean`, `max`, `min` `(tensor)` | A rank-0 tensor over every element in row-major order |
| `sum_axis`, `mean_axis`, `max_axis`, `min_axis` `(tensor, axis)` | The axis is removed |

Each result takes its elements in increasing index, starting from `0`, `-inf` or `+inf`; a mean divides the sum by the count. The order is fixed, so a sum equals an explicit loop and can differ from NumPy's pairwise `sum` in the last bits. `mean` of an empty tensor is NaN.

## Layout

| Function | Result |
|---|---|
| `reshape(tensor, shape)` | Same elements, new shape; no copy |
| `transpose(tensor, order)` | Axes in the given order |
| `slice(tensor, axis, start, count, step)` | `count` elements along `axis` from `start`, every `step`; `step` is positive |
| `set_slice(target, axis, start, part)` | `target` with `part` written along `axis` from `start` |
| `concat(parts, axis)` | Parts joined along `axis`: `F.concat(F.cons(a, F.cons(b, F.single(c))), 1)` |
| `pad(tensor, before, after, value)` | `before[i]` and `after[i]` new elements on axis `i`, equal to `value` |
| `pad_edge(tensor, before, after)` | The same, repeating the nearest element |

## Windows

Tensors are `[channels, height, width]` or `[batch, channels, height, width]`.

| Function | Result |
|---|---|
| `conv2d(input, weights, bias, stride, padding, activation)` | Weights are `[out_channels, in_channels, kernel_height, kernel_width]`, bias `[out_channels]`; the same stride and padding on both axes; the activation is applied to the output |
| `conv2d_with(~activation, operations, input, weights, bias, stride, padding)` | With a custom activation function |
| `max_pool2d(tensor, kernel)` | Windows of `kernel` by `kernel` that do not overlap (stride `kernel`, padding 0) |
| `max_pool2d_with(tensor, kernel, stride, padding)` | The full form; padding counts as `-inf` |
| `upsample_nearest(tensor, scale)` | Each row and column repeated `scale` times |

Each convolution output starts at +0.0, adds products in input-channel, kernel-row, kernel-column order, then adds the bias. [Design](../docs/design.md) explains how it runs and plans; [proofs](../docs/proofs.md) says what is proved.

## Inside the library

Application code does not need these. They are listed for readers of the source.

| Module | Role |
|---|---|
| [tensor.bend](tensor.bend), [tensor_layout.bend](tensor_layout.bend), [tensor_window.bend](tensor_window.bend), [tensor_convolution.bend](tensor_convolution.bend) | The generic tensor behind `F`: any element type, dense and band storage |
| `tensor_band*.bend` | Row-band storage and its operations |
| [tensor_operations.bend](tensor_operations.bend) | Checked `copy`, `combine`, `apply`, `reduce`, `scatter`, `sweep` over views; both `Accepted` and `Rejected` return every owner |
| [tensor_view.bend](tensor_view.bend), [tensor_shape.bend](tensor_shape.bend), [tensor_cursor.bend](tensor_cursor.bend), [tensor_access.bend](tensor_access.bend) | Views (extents, strides, offset), checked element counts, the strided cursor |
| `kernel_*.bend` | The convolution kernel: weights, packing, GEMM, gather, planning, cost |
| [storage_buffer.bend](storage_buffer.bend), [storage_allocation.bend](storage_allocation.bend), `storage_*.bend` | `Buffer`: an owned array with its logical length and a storage certificate |
| `traversal_*.bend` | The loops: copy, update, fold, indexed output, partition tree |
| [math_activation.bend](math_activation.bend), [math_fp32.bend](math_fp32.bend), [math_sequence.bend](math_sequence.bend), [ordered_matrix.bend](ordered_matrix.bend) | Activation values and scalar functions; the ordered sequences and matrix products the proofs speak about |
| [machine_limits.bend](machine_limits.bend), [kernel_cost.bend](kernel_cost.bend) and their `default_*` files | Machine bounds used by proofs; planner cost parameters |
| [proofs/](proofs/) | The library's proofs |
