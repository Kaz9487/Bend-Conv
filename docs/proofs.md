# Proofs

[English](proofs.md) | [繁體中文](proofs.zh-TW.md)

Every proof is a Bend term checked by the unmodified official Bend checker.

```sh
python run.py proofs
```

## The root theorem

[native_inference_law.bend](../convolution/proofs/native_inference_law.bend):

```text
law native_inference_matches_matrix:
  for +problem: ConvSpec.Problem
  {NativeConv.infer(problem) == MatrixSpec.infer(problem) : ConvSpec.ConvOutcome}
```

A problem holds a shape, an input, weights and a bias. The law puts no condition on it: it covers rejected input, the reference fallback and native execution, for every split depth (`inference_with_any_split_depth_matches_matrix`) and both strategies (`every_strategy_matches_matrix`).

**The numerical contract.** Each output starts at FP32 +0.0, adds the products in input-channel, kernel-row, kernel-column order (multiply, then add), and adds the bias last. Padding samples take part in that order. The theorem is about this sequence of FP32 operations, not about real numbers, and it does not allow fused multiply-add or reassociation.

**Frozen files.** [specification.bend](../convolution/specification.bend), [specification_law.bend](../convolution/specification_law.bend), [ordered_matrix.bend](../lib/ordered_matrix.bend), [matrix_specification.bend](../convolution/matrix_specification.bend) and [matrix_convolution_law.bend](../convolution/proofs/matrix_convolution_law.bend) must keep the hashes in [frozen.json](../data/contracts/frozen.json). Every proof run checks them.

## What is trusted

- The official checker and Base library.
- The compiler, the runtime and its scheduler, the C compiler and the hardware. The theorem is about the Bend source.
- Bend's file I/O under `load` and `save`. Library and convolution proofs contain no `@unsafe` definition.

Numerical checks cover what the proofs do not: generated code, model wiring, activation, I/O and detection decoding are compared with NumPy and PyTorch, and the mathematical functions with MPFR over every FP32 input ([benchmarks](../benchmarks/README.md)).

## Following the convolution proof

| Obligation | Source |
|---|---|
| Direct (scalar) execution | [scalar_inference_refinement.bend](../convolution/proofs/scalar_inference_refinement.bend) |
| Packed execution | [packed_inference_refinement.bend](../convolution/proofs/packed_inference_refinement.bend) |
| The actual pipeline: reorder, partition, pack and GEMM, gather | [microkernel_inference_refinement.bend](../convolution/proofs/microkernel_inference_refinement.bend), [microkernel_storage_execution.bend](../convolution/proofs/microkernel_storage_execution.bend) |
| Weight reorder into eight-row tiles | [weight_reorder_readback.bend](../convolution/proofs/weight_reorder_readback.bend) |
| Packing, and the input it leaves unchanged | [packing_storage_refinement.bend](../convolution/proofs/packing_storage_refinement.bend), [packing_contents.bend](../convolution/proofs/packing_contents.bend) |
| GEMM storage as a write list; bounds, values and coverage | [kernel_gemm_writes.bend](../lib/proofs/kernel_gemm_writes.bend), [gemm_output_readback.bend](../convolution/proofs/gemm_output_readback.bend) |
| The accumulator equals the matrix dot | [gemm_dot_matches_matrix.bend](../convolution/proofs/gemm_dot_matches_matrix.bend) |
| Every leaf output equals the matrix value | [microkernel_leaf_readback.bend](../convolution/proofs/microkernel_leaf_readback.bend) |
| Input row windows of every split | [window_partition.bend](../convolution/proofs/window_partition.bend), [window_leaf_readback.bend](../convolution/proofs/window_leaf_readback.bend) |
| Gather writes | [gather_write_program.bend](../convolution/proofs/gather_write_program.bend) |
| Arbitrary caller array capacities | [array_api_validation.bend](../convolution/proofs/array_api_validation.bend) |
| Shared storage contracts | [storage_refinement.bend](../lib/proofs/storage_refinement.bend) |

The pattern is the same throughout: describe what a loop stores as a list of writes, prove every write is in bounds, prove the value written, and prove that later writes do not disturb it.

## The public convolution

`F.conv2d` runs the same kernel through buffers that carry storage certificates ([kernel_pipeline.bend](../lib/kernel_pipeline.bend)). The root theorem is stated for the Array engine; the public path has its own theorems:

- **Dense path:** equal to the Array pipeline (`microkernel_storage_execution`).
- **Band path:** for the plan the code selects, the output bands cover every row and stay inside their allocations ([band_pipeline_well_formed.bend](../convolution/proofs/band_pipeline_well_formed.bend)), and every stored value is the ordered matrix value of its output point ([convolution_leaf_contract.bend](../convolution/proofs/convolution_leaf_contract.bend), [band_pipeline_execution.bend](../convolution/proofs/band_pipeline_execution.bend)).

These are separate from the unconditional root. Their premises are a Usable input in a well-formed layout with finite values, matching shapes, and the machine guards the code checks.

## Tensor operations

Each theorem has one premise about the call: it returned a Usable result.

| Operation | What a Usable result holds | Source |
|---|---|---|
| `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, with broadcasting | `operation(left, right)` at every index, in that argument order | [tensor_combine_dense_values.bend](../lib/proofs/tensor_combine_dense_values.bend) |
| `add_scalar`, `mul_scalar`, ... | `operation(x, scalar)` | [tensor_scalar_values.bend](../lib/proofs/tensor_scalar_values.bend) |
| `less`, `less_equal`, `greater`, `greater_equal`, `equal`, `not_equal` | 1.0 or 0.0 by the comparison of `left` and `right` at every index | [tensor_comparison_values.bend](../lib/proofs/tensor_comparison_values.bend) |
| `sigmoid`, `silu`, `tanh`, `gelu_tanh`, `exp`, `log` | the ordered map of the scalar function in [math_fp32.bend](../lib/math_fp32.bend) | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `transpose`, `slice`, `broadcast_to`, dense upsample | the input at the address the view gives to that index | [tensor_gather_values.bend](../lib/proofs/tensor_gather_values.bend) |
| `set_slice`, `pad`, dense `concat` | the part written through its view, the rest unchanged | [tensor_set_slice_values.bend](../lib/proofs/tensor_set_slice_values.bend), [tensor_concat_dense_values.bend](../lib/proofs/tensor_concat_dense_values.bend) |
| `sum_axis`, `max_axis`, `min_axis`, `mean_axis` | the fold of the axis in increasing index, from the initial value | [tensor_reduce_axis_values.bend](../lib/proofs/tensor_reduce_axis_values.bend) |
| `max_pool2d` | the fold of each window, rows then columns, in increasing index, from -inf | [tensor_pool_dense_values.bend](../lib/proofs/tensor_pool_dense_values.bend) |

Underneath:

| Subject | What is shown | Source |
|---|---|---|
| Bounds and non-overlap | From the guards the code checks: every read and write is in range, and source and destination do not overlap | [tensor_operation_bounds.bend](../lib/proofs/tensor_operation_bounds.bend) |
| Cursor | The native cursor visits coordinates in row-major order; copying in runs of the innermost axis equals visiting each element | [tensor_cursor_refinement.bend](../lib/proofs/tensor_cursor_refinement.bend), [tensor_cursor_order.bend](../lib/proofs/tensor_cursor_order.bend), [tensor_access_refinement.bend](../lib/proofs/tensor_access_refinement.bend) |
| Storage certificates | A buffer carries a proof that its array has the shape the model says. Certificate work is constant per operation | [storage_buffer_refinement.bend](../lib/proofs/storage_buffer_refinement.bend) |
| Failure | Which checked call each public function makes; a failed input stays failed | [tensor_public_refinement.bend](../lib/proofs/tensor_public_refinement.bend) |
| Band recursion | `Bands.each` runs every band operation and is proved once; each operation proves only its leaf | [tensor_band_each.bend](../lib/proofs/tensor_band_each.bend) |
| Band layout | When a band tensor is well formed; its values by channel, row and column | [tensor_band_shape_refinement.bend](../lib/proofs/tensor_band_shape_refinement.bend), [tensor_band_values.bend](../lib/proofs/tensor_band_values.bend) |

## What is not proved

- The planner's cost estimates, latency and workspace accounting. They cannot affect a result.
- Real-number accuracy. [math_accuracy.py](../benchmarks/checks/math_accuracy.py) measures it.
- `select` and `softmax`.
- A closed form of which indices a slice view reaches; the statements follow the cursor's coordinates.
- The YOLO example's graph, I/O and detection decoding.
- Compiler and runtime behaviour.

## Inspecting

`python run.py proofs` also writes the list of declarations with their premises, and a reference graph from the root, under `out/results/proof-audit/`. They are for navigation; the verdict is the checker's.
