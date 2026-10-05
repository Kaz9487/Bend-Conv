# Connect the reindexed kernel's actual buffer reads to frozen matrix accessors.
import Base
import ./input_storage.bend as Inputs
import ../../lib/math_sequence.bend as Sequence
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../../lib/machine_limits.bend as Machine
import ../inference.bend as NativeConv
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/word_multiplication.bend as Multiplication
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_order.bend as Order
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./list_array_storage.bend as ListStorage
import ./native_weight_refinement.bend as NativeWeightRefinement
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packed_reduction_samples.bend as ReductionSamples
import ./kernel_reduction_order.bend as KernelOrder
import ./prepared_packing_samples.bend as Prepared
import ./packing_tile_bounds.bend as PackingBounds
import ./packing_readback.bend as PackingReadback

def context(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,data: Tree.Tree<Tile.Tile8>,group: Nat,output_channel: Nat) -> KernelOrder.ReindexedContext():
  KernelOrder.context(ReductionSamples.Context{data,Inputs.weight_tree(p,storage_contract),Coordinates.native_inner(p)},Guards.kernel(p),
    Nat.mul(group,Coordinates.inner(p)),Nat.mul(output_channel,Coordinates.inner(p)))

def input_matches_patch(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+data: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+group: Nat,+output_channel_index: Nat,
  +lane: Sequence.Index(8n),+input_channel_index: Nat,+output_row_index: Nat,+output_column_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,contents: Prepared.LeafSamples(p,offset,span,data),
  group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},position: {PackingReadback.logical_position(offset,group,
      lane) == Coordinates.position(p,output_row_index,output_column_index) : Nat},input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},output_row_in_bounds: {Nat.is_lt(output_row_index,
      ConvSpec.out_height(p)) == True{} : Bool},output_column_in_bounds: {Nat.is_lt(output_column_index,ConvSpec.out_width(p)) == True{} : Bool},kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,
      Guards.kernel(p)) == True{} : Bool},kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,Guards.kernel(p)) == True{} : Bool}) -> {KernelOrder.matrix_input()(context(p,storage_contract,data,
      group,output_channel_index),lane,MatrixSpec.KIndex{input_channel_index,kernel_row_index,kernel_column_index}) == MatrixSpec.patch(p,MatrixSpec.Position{output_row_index,output_column_index},MatrixSpec.KIndex{input_channel_index,kernel_row_index,kernel_column_index}) : F32}:
  %contents(lane,group,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,group_inside,position,input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds) : {KernelOrder.matrix_input()(context(p,storage_contract,data,group,output_channel_index),lane,
      MatrixSpec.KIndex{input_channel_index,kernel_row_index,kernel_column_index}) == _ : F32}
  {==}

def first_weight_matches_matrix(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+data: Tree.Tree<Tile.Tile8>,+group: Nat,+output_channel_index: Nat,
  +input_channel_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,+valid: {ConvSpec.valid(p) == True{} : Bool},+output_channel_in_bounds: {Nat.is_lt(output_channel_index,Guards.output_channel_count(p)) == True{} : Bool},+input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},
  +kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,Guards.kernel(p)) == True{} : Bool},+kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,
      Guards.kernel(p)) == True{} : Bool}) -> {KernelOrder.matrix_weight()(context(p,storage_contract,data,group,output_channel_index),False{},MatrixSpec.KIndex{input_channel_index,kernel_row_index,
      kernel_column_index}) == MatrixSpec.weight(p,output_channel_index,MatrixSpec.KIndex{input_channel_index,kernel_row_index,kernel_column_index}) : F32}:
  %Coordinates.weight_index(output_channel_index,Guards.input_channel_count(p),input_channel_index,Guards.kernel(p),kernel_row_index,kernel_column_index) :
    {Tree.read(F32,Inputs.weight_tree(p,storage_contract),U32.from_nat(_)) == ConvSpec.weight(p,output_channel_index,input_channel_index,kernel_row_index,kernel_column_index) : F32}
  Inputs.weight_matches_spec(p,storage_contract,
    output_channel_index,input_channel_index,kernel_row_index,kernel_column_index,Guards.valid(p,valid),output_channel_in_bounds,input_channel_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds)