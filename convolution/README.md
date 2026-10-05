# Convolution package

[English](README.md) | [繁體中文](README.zh-TW.md)

The frozen specification of convolution and the engine the root theorem is about. Programs call `F.conv2d` from the [library](../lib/README.md), which runs the same kernel.

| File | Role |
|---|---|
| [specification.bend](specification.bend), [specification_law.bend](specification_law.bend) | Frozen. The problem (shape, input, weights, bias), which problems are valid, and the outcome |
| [matrix_specification.bend](matrix_specification.bend) | Frozen. The meaning: each output is an ordered dot product of a weight row and an input patch, on the axes `(input channel, kernel row, kernel column)` |
| [reference.bend](reference.bend), [ordered_reference.bend](ordered_reference.bend) | A direct evaluation of the specification |
| [shape.bend](shape.bend) | Machine dimensions shared by the native strategies |
| [scalar_kernel.bend](scalar_kernel.bend) | Direct strategy: one output at a time, no workspace |
| [inference.bend](inference.bend) | The native entry on lists: `infer`, `infer_with_split_depth`, `infer_with_strategy` |
| [array_api.bend](array_api.bend) | The entry on owned arrays: `infer` returns `Rejected`, `Unsupported` or `Accepted` |
| [planning.bend](planning.bend) | `plan`: picks a strategy and a split depth |
| [proofs/](proofs/) | The proofs, ending in `native_inference_matches_matrix` |

## Strategies

- **Direct** computes each output with the shared reduction loop.
- **Packed** reorders the weights, packs input patches and runs the 4x8 GEMM of `lib/kernel_*.bend`.

Both use the same accumulation order and are proved equal to the matrix specification for every split depth.

`automatic(worker_limit, workspace_budget)` chooses with the cost model in [design](../docs/design.md#planning). `None{}` sets no workspace limit; `Some{0n}` selects Direct. Workspace counts packed tiles, temporary leaf outputs and what each split clones.

## Names in the specification

`ci/co` are channel counts, `ic/oc` channel indices, `oy/ox` the output row and column, `ky/kx` the kernel row and column.

The [proof guide](../docs/proofs.md) follows the proof from the root to each loop.
