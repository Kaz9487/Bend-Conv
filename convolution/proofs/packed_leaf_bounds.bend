# Real store bounds, packaged once for the two-row write/readback theorem.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../inference.bend as NativeConv
import ../../lib/kernel_tile.bend as Tile
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./prepared_packing_samples.bend as Prepared
import ./tile_partition_bounds.bend as TilePartitionBounds
import ../../lib/math_sequence.bend as Sequence

type Geometry<-limits: Machine.Limits,-problem: ConvSpec.Problem,-span: U32> is Data:
  Geometry{valid: Guards.Valid<problem>,supported: Support.FastSupported<limits,problem>,
    span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},alignment: TilePartitionBounds.BlockAligned<span>}