# Stable convolution leaf boundary. Packing proves samples; this module reuses
# the checked leaf refinement. Band scheduling does not inspect GEMM jobs,
# accumulator rows, lanes, weight blocks, or the inner reduction implementation.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./convolution_leaf_contract.bend as LeafContract
import ./packing_contents.bend as PackingContents
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./tile_partition_bounds.bend as TilePartitionBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./output_values.bend as Values
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/nat_order.bend as Order

def output(+problem: ConvSpec.Problem,+inputs: Inputs.Contract<problem>,tiles: Tree.Tree<Tile.Tile8>,+span: U32) -> Tree.Tree<F32>:
  MicroLeaf.output(problem,inputs,tiles,span)

def point(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+inputs: Inputs.Contract<problem>,
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  evidence: Evidence(problem,offset,span,tiles,output_row,output_column),span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},span_alignment: TilePartitionBounds.BlockAligned<span>,
  channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},x_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,output(problem,inputs,tiles,span),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column))) == Values.matrix_value(problem,channel,output_row,output_column) : F32}:
  MicroLeaf.every_position_from_samples(~Evidence,~samples,limits,problem,inputs,tiles,offset,span,channel,column,output_row,output_column,
    valid,supported,evidence,span_bound,span_alignment,channel_inside,column_inside,position,row_inside,x_inside)

# A band window leaf reads its own packing at any output offset.
def band_point(+limits: Machine.Limits,+problem: ConvSpec.Problem,+inputs: Inputs.Contract<problem>,+offset: U32,+span: U32,
  +channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+span_alignment: TilePartitionBounds.BlockAligned<span>,
  +range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+x_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,MicroLeaf.leaf_output(problem,inputs,offset,span),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column))) == Values.matrix_value(problem,channel,output_row,output_column) : F32}:
  TilePartitionBounds.BlockAligned{+blocks,+span_exact} = span_alignment
  +span_positive = Order.le_lt_trans(0n,column,U32.to_nat(span),Order.zero_le(column),column_inside)
  +positive = MicroLeaf.leaf_positive(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported),span,blocks,span_exact,span_bound,span_positive)
  +tiles = MicroLeaf.packed_tiles(problem,inputs,offset,span)
  +certificate = {LeafContract.BandEvidence{inputs,span_bound,range_bound,MicroLeaf.packed_matches(problem,inputs,offset,span,positive)} :
    LeafContract.band_evidence(problem,offset,span,tiles,output_row,output_column)}
  MicroLeaf.every_position_from_samples(~LeafContract.band_evidence,~LeafContract.band_samples,limits,problem,inputs,tiles,offset,span,channel,column,output_row,output_column,
    valid,supported,certificate,span_bound,TilePartitionBounds.BlockAligned{blocks,span_exact},channel_inside,column_inside,position,row_inside,x_inside)
