# Every actual gather write is in bounds and agrees with its matrix coordinate.
import Base
import ./input_storage.bend as Inputs
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../inference.bend as NativeConv
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/kernel_tile.bend as Tile
import ./conditional_write_coverage.bend as Coverage
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/row_major_index.bend as RowMajor
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./prepared_packing_samples.bend as Prepared
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./output_values.bend as Values
import ./reduction_coordinate_bounds.bend as Coordinates
import ./supported_padded_coordinates.bend as Padded
import ./gather_output_index.bend as GatherOutputIndex
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ./gather_write_program.bend as GatherWrites

def output(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+offset: U32,+span: U32) -> Tree.Tree<F32>:
  MicroLeaf.leaf_output(problem,storage_contract,offset,span)

def query(+problem: ConvSpec.Problem,channel: Nat,output_row_index: Nat,output_column_index: Nat) -> Nat:
  Nat.add(Nat.mul(channel,GatherCopyCoordinates.spatial(problem)),Coordinates.position(problem,output_row_index,output_column_index))

def geometry(+limits: Machine.Limits,+problem: ConvSpec.Problem,+input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,
  valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  certificate: Prepared.LeafCertificate(problem,offset,span,input)) -> GatherCopyCoordinates.Geometry<limits,problem,offset,span>:
  Prepared.LeafEvidence{packing_contract,span_bound,offset_alignment,span_alignment,range_bound,contents} = certificate
  GatherCopyCoordinates.Geometry{Guards.valid(problem,valid),Support.fast(limits,problem,supported),span_bound,range_bound}

def position_inside(+problem: ConvSpec.Problem,+output_row_index: Nat,+output_column_index: Nat,row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Nat.is_lt(Coordinates.position(problem,output_row_index,output_column_index),GatherCopyCoordinates.spatial(problem)) == True{} : Bool}:
  RowMajor.radix(output_row_index,ConvSpec.out_height(problem),output_column_index,ConvSpec.out_width(problem),row_inside,column_inside)

def value_at_destination(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+index: U32,+channel: Nat,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificate: Prepared.LeafCertificate(problem,offset,span,input),
  +inside: {Nat.is_lt(U32.to_nat(index),Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span))) == True{} : Bool},
  +kept: {Nat.is_lt(Nat.add(U32.to_nat(offset),GatherCopyCoordinates.column(index,span)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool},
  +same_destination: {GatherOutputIndex.destination(GatherCopyCoordinates.row(index,span),GatherCopyCoordinates.column(index,span),GatherCopyCoordinates.spatial(problem),U32.to_nat(offset)) == query(problem,channel,output_row_index,output_column_index) : Nat},
  +row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,output(problem,storage_contract,offset,span),index) == Values.matrix_value(problem,channel,output_row_index,output_column_index) : F32}:
  +position = Coordinates.position(problem,output_row_index,output_column_index)
  +space = GatherCopyCoordinates.spatial(problem)
  +position_bound = position_inside(problem,output_row_index,output_column_index,row_inside,column_inside)
  +positive = Order.le_lt_trans(0n,position,space,Order.zero_le(position),position_bound)
  +same_channel = Radix.row_unique(GatherCopyCoordinates.row(index,span),channel,space,Nat.add(U32.to_nat(offset),GatherCopyCoordinates.column(index,span)),position,positive,kept,position_bound,same_destination)
  %same_channel : {Tree.read(F32,output(problem,storage_contract,offset,span),index) == Values.matrix_value(problem,_,output_row_index,output_column_index) : F32}
  %Equal.sym(U32,index,U32.from_nat(Nat.add(Nat.mul(GatherCopyCoordinates.row(index,span),U32.to_nat(span)),GatherCopyCoordinates.column(index,span))),
    Encoded.encoding(index,Nat.add(Nat.mul(GatherCopyCoordinates.row(index,span),U32.to_nat(span)),GatherCopyCoordinates.column(index,span)),Radix.division_identity(index,U32.to_nat(span),GatherCopyCoordinates.span_positive(problem,span,index,inside)))) :
    {Tree.read(F32,output(problem,storage_contract,offset,span),_) == Values.matrix_value(problem,GatherCopyCoordinates.row(index,span),output_row_index,output_column_index) : F32}
  MicroLeaf.leaf_every_position(limits,problem,storage_contract,input,offset,span,GatherCopyCoordinates.row(index,span),GatherCopyCoordinates.column(index,span),output_row_index,output_column_index,valid,supported,certificate,
    GatherCopyCoordinates.row_inside(problem,span,index,inside),Radix.remainder_inside(index,U32.to_nat(span),GatherCopyCoordinates.span_positive(problem,span,index,inside)),
    Radix.col_unique(GatherCopyCoordinates.row(index,span),channel,space,Nat.add(U32.to_nat(offset),GatherCopyCoordinates.column(index,span)),position,positive,kept,position_bound,same_destination),row_inside,column_inside)
