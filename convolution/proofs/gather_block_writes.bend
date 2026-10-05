# Bounds and matrix values for every store in the real channel-copy schedule.
import Base
import ./bounded_write_readback.bend as WriteReadback
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/kernel_gather.bend as Gather
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/u32_successor.bend as U32Successor
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./prepared_packing_samples.bend as Prepared
import ./packed_output_bounds.bend as Output
import ./supported_padded_coordinates.bend as Padded
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ./gather_output_index.bend as GatherOutputIndex
import ./gather_block_coordinates.bend as BlockCoordinates
import ./gather_block_width.bend as Width
import ./gather_blocks.bend as Blocks
import ./gather_leaf_writes.bend as GatherLeafWrites
import ./conditional_write_coverage.bend as Coverage
import ./output_values.bend as Values

def point_value(+matches: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+row: U32,+column: U32,+channel: Nat,+output_row: Nat,+output_column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificate: Prepared.LeafCertificate(problem,offset,span,input),
  geometry: GatherCopyCoordinates.Geometry<limits,problem,offset,span>,+in_row: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},+in_column: {Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool},
  +kept: {Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(column)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool},
  equal: {Nat.is_eq(GatherLeafWrites.query(problem,channel,output_row,output_column),U32.to_nat(BlockCoordinates.destination(row,Padded.spatial(problem),offset,column))) == matches : Bool},
  row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  Coverage.Matches(matches,Tree.read(F32,GatherLeafWrites.output(problem,storage,offset,span),BlockCoordinates.source(row,span,column)),Values.matrix_value(problem,channel,output_row,output_column)):
  match matches:
    case False{}: Unit{}
    case True{}:
      GatherCopyCoordinates.Geometry{+guard_valid,+guard_supported,+span_bound,range_bound} = geometry
      GatherLeafWrites.value_at_destination(limits,problem,storage,input,offset,span,BlockCoordinates.source(row,span,column),channel,output_row,output_column,valid,supported,certificate,
        BlockCoordinates.source_inside(limits,problem,span,row,column,guard_supported,span_bound,in_row,in_column),
        %Equal.sym(Nat,GatherCopyCoordinates.column(BlockCoordinates.source(row,span,column),span),U32.to_nat(column),BlockCoordinates.source_column(limits,problem,span,row,column,guard_supported,span_bound,in_row,in_column)) :
          {Nat.is_lt(Nat.add(U32.to_nat(offset),_),GatherCopyCoordinates.spatial(problem)) == True{} : Bool}
        kept,
        %Equal.sym(Nat,GatherCopyCoordinates.row(BlockCoordinates.source(row,span,column),span),U32.to_nat(row),BlockCoordinates.source_row(limits,problem,span,row,column,guard_supported,span_bound,in_row,in_column)) :
          {GatherOutputIndex.destination(_,GatherCopyCoordinates.column(BlockCoordinates.source(row,span,column),span),GatherCopyCoordinates.spatial(problem),U32.to_nat(offset)) == GatherLeafWrites.query(problem,channel,output_row,output_column) : Nat}
        %Equal.sym(Nat,GatherCopyCoordinates.column(BlockCoordinates.source(row,span,column),span),U32.to_nat(column),BlockCoordinates.source_column(limits,problem,span,row,column,guard_supported,span_bound,in_row,in_column)) :
          {GatherOutputIndex.destination(U32.to_nat(row),_,GatherCopyCoordinates.spatial(problem),U32.to_nat(offset)) == GatherLeafWrites.query(problem,channel,output_row,output_column) : Nat}
        %BlockCoordinates.destination_exact(limits,problem,offset,row,column,guard_valid,guard_supported,in_row,kept) : {_ == GatherLeafWrites.query(problem,channel,output_row,output_column) : Nat}
        Equal.sym(Nat,GatherLeafWrites.query(problem,channel,output_row,output_column),U32.to_nat(BlockCoordinates.destination(row,Padded.spatial(problem),offset,column)),
          Order.equal_of_is_eq(GatherLeafWrites.query(problem,channel,output_row,output_column),U32.to_nat(BlockCoordinates.destination(row,Padded.spatial(problem),offset,column)),equal)),row_inside,column_inside)

def prepend(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+row: U32,+column: U32,+channel: Nat,+output_row: Nat,+output_column: Nat,+capacity: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificate: Prepared.LeafCertificate(problem,offset,span,input),
  +in_row: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},in_column: {Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool},
  +kept: {Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(column)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(capacity)) == True{} : Bool},
  row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +rest: List<&2,Writes.Assignment<F32>>,tail: WriteReadback.WriteEvidence<GatherLeafWrites.query(problem,channel,output_row,output_column),Values.matrix_value(problem,channel,output_row,output_column),capacity,rest>) ->
  WriteReadback.WriteEvidence<GatherLeafWrites.query(problem,channel,output_row,output_column),Values.matrix_value(problem,channel,output_row,output_column),capacity,
    Writes.Put{BlockCoordinates.destination(row,Padded.spatial(problem),offset,column),Tree.read(F32,GatherLeafWrites.output(problem,storage,offset,span),BlockCoordinates.source(row,span,column))} <> rest>:
  WriteReadback.prepend_evidence(GatherLeafWrites.query(problem,channel,output_row,output_column),Values.matrix_value(problem,channel,output_row,output_column),capacity,
    BlockCoordinates.destination(row,Padded.spatial(problem),offset,column),Tree.read(F32,GatherLeafWrites.output(problem,storage,offset,span),BlockCoordinates.source(row,span,column)),rest,
    U32Division.nat_lt_native(BlockCoordinates.destination(row,Padded.spatial(problem),offset,column),capacity,
      %Equal.sym(Nat,U32.to_nat(BlockCoordinates.destination(row,Padded.spatial(problem),offset,column)),GatherOutputIndex.destination(U32.to_nat(row),U32.to_nat(column),GatherCopyCoordinates.spatial(problem),U32.to_nat(offset)),
        BlockCoordinates.destination_exact(limits,problem,offset,row,column,Guards.valid(problem,valid),Support.fast(limits,problem,supported),in_row,kept)) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
      Bounds.nat_trans(GatherOutputIndex.destination(U32.to_nat(row),U32.to_nat(column),GatherCopyCoordinates.spatial(problem),U32.to_nat(offset)),Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(capacity),BlockCoordinates.destination_inside(problem,offset,row,column,in_row,kept),room)),
    point_value(Nat.is_eq(GatherLeafWrites.query(problem,channel,output_row,output_column),U32.to_nat(BlockCoordinates.destination(row,Padded.spatial(problem),offset,column))),limits,problem,storage,input,offset,span,row,column,channel,output_row,output_column,valid,supported,certificate,
      GatherLeafWrites.geometry(limits,problem,input,offset,span,valid,supported,certificate),in_row,in_column,kept,{==},row_inside,column_inside),tail)

def next(+index: U32,+bound: Nat,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  inside: {Nat.is_lt(U32.to_nat(index),bound) == True{} : Bool},bounded: {Nat.is_le(bound,U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((index + 1 : U32)) == 1n+U32.to_nat(index) : Nat}:
  U32Successor.successor(index,Bounds.uint_trans(index,capacity,2147483648,
    U32Division.nat_lt_native(index,capacity,Bounds.nat_trans(U32.to_nat(index),bound,U32.to_nat(capacity),inside,bounded)),capacity_ok))

def first(-Left: Data,-Right: Data,facts: Shape.Both<Left,Right>) -> Left:
  Shape.Both{left,right} = facts
  left

def second(-Left: Data,-Right: Data,facts: Shape.Both<Left,Right>) -> Right:
  Shape.Both{left,right} = facts
  right

def width_facts(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+span: U32,+column: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  inside: {Nat.is_lt(U32.to_nat(column),U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool},{Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(column)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool}>:
  %Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported)) :
    Shape.Both<{Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool},{Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(column)),_) == True{} : Bool}>
  Width.inside(offset,Padded.spatial(problem),span,U32.to_nat(column),Machine.capacity(limits),Limits.capacity_fits(limits),
    BlockCoordinates.spatial_bound(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported)),inside)

def column_next(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+span: U32,+column: U32,
  geometry: GatherCopyCoordinates.Geometry<limits,problem,offset,span>,inside: {Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool}) ->
  {U32.to_nat((column + 1 : U32)) == 1n+U32.to_nat(column) : Nat}:
  GatherCopyCoordinates.Geometry{valid,supported,span_bound,range_bound} = geometry
  next(column,U32.to_nat(span),Machine.capacity(limits),Limits.capacity_fits(limits),inside,Output.span_in_capacity(limits,problem,span,valid,supported,span_bound))

def columns(+count: Nat,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+row: U32,+column: U32,+channel: Nat,+output_row: Nat,+output_column: Nat,+capacity: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificate: Prepared.LeafCertificate(problem,offset,span,input),
  +in_row: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(U32.to_nat(column),count),U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(capacity)) == True{} : Bool},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  WriteReadback.WriteEvidence<GatherLeafWrites.query(problem,channel,output_row,output_column),Values.matrix_value(problem,channel,output_row,output_column),capacity,
    Contiguous.writes(~F32,count,GatherLeafWrites.output(problem,storage,offset,span),column,(row * span : U32),(row * Padded.spatial(problem) + offset : U32))>:
  match count:
    case 0n: WriteReadback.WriteEvidence{Unit{},Unit{}}
    case 1n+ +rest:
      +facts = width_facts(limits,problem,offset,span,column,valid,supported,
        Order.room_index(U32.to_nat(column),rest,U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span)),remaining))
      +in_span = first({Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool},{Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(column)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool},facts)
      +kept = second({Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool},{Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(column)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool},facts)
      prepend(limits,problem,storage,input,offset,span,row,column,channel,output_row,output_column,capacity,valid,supported,certificate,in_row,in_span,kept,room,row_inside,column_inside,
        Contiguous.writes(~F32,rest,GatherLeafWrites.output(problem,storage,offset,span),(column + 1 : U32),(row * span : U32),(row * Padded.spatial(problem) + offset : U32)),
        columns(rest,limits,problem,storage,input,offset,span,row,(column + 1 : U32),channel,output_row,output_column,capacity,valid,supported,certificate,in_row,
          %Equal.sym(Nat,U32.to_nat((column + 1 : U32)),1n+U32.to_nat(column),column_next(limits,problem,offset,span,column,GatherLeafWrites.geometry(limits,problem,input,offset,span,valid,supported,certificate),in_span)) :
            {Nat.is_le(Nat.add(_,rest),U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span))) == True{} : Bool}
          Order.advance_room(U32.to_nat(column),rest,U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span)),remaining),room,row_inside,column_inside))

def channels(+count: Nat,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+row: U32,+channel: Nat,+output_row: Nat,+output_column: Nat,+capacity: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificate: Prepared.LeafCertificate(problem,offset,span,input),
  +remaining: {Nat.is_le(Nat.add(U32.to_nat(row),count),Guards.output_channel_count(problem)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(capacity)) == True{} : Bool},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  WriteReadback.WriteEvidence<GatherLeafWrites.query(problem,channel,output_row,output_column),Values.matrix_value(problem,channel,output_row,output_column),capacity,
    Blocks.writes(count,row,Gather.gather_width(offset,Padded.spatial(problem),span),offset,Padded.spatial(problem),span,GatherLeafWrites.output(problem,storage,offset,span))>:
  match count:
    case 0n: WriteReadback.WriteEvidence{Unit{},Unit{}}
    case 1n+ +rest:
      +in_row = Order.room_index(U32.to_nat(row),rest,Guards.output_channel_count(problem),remaining)
      WriteReadback.concatenated_evidence(GatherLeafWrites.query(problem,channel,output_row,output_column),Values.matrix_value(problem,channel,output_row,output_column),capacity,
        Contiguous.writes(~F32,U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span)),GatherLeafWrites.output(problem,storage,offset,span),0,(row * span : U32),(row * Padded.spatial(problem) + offset : U32)),
        Blocks.writes(rest,(row + 1 : U32),Gather.gather_width(offset,Padded.spatial(problem),span),offset,Padded.spatial(problem),span,GatherLeafWrites.output(problem,storage,offset,span)),
        columns(U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span)),limits,problem,storage,input,offset,span,row,0,channel,output_row,output_column,capacity,valid,supported,certificate,in_row,
          Order.reflexive(U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span))),room,row_inside,column_inside),
        channels(rest,limits,problem,storage,input,offset,span,(row + 1 : U32),channel,output_row,output_column,capacity,valid,supported,certificate,
          %Equal.sym(Nat,U32.to_nat((row + 1 : U32)),1n+U32.to_nat(row),next(row,Guards.output_channel_count(problem),Machine.capacity(limits),Limits.capacity_fits(limits),in_row,Output.channel_capacity_bound(limits,problem,Support.fast(limits,problem,supported)))) :
            {Nat.is_le(Nat.add(_,rest),Guards.output_channel_count(problem)) == True{} : Bool}
          Order.advance_room(U32.to_nat(row),rest,Guards.output_channel_count(problem),remaining),room,row_inside,column_inside))

def block(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+channel: Nat,+output_row: Nat,+output_column: Nat,+capacity: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificate: Prepared.LeafCertificate(problem,offset,span,input),
  room: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(capacity)) == True{} : Bool},
  row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  WriteReadback.WriteEvidence<GatherLeafWrites.query(problem,channel,output_row,output_column),Values.matrix_value(problem,channel,output_row,output_column),capacity,
    Blocks.block_writes(U32.from_nat(Guards.output_channel_count(problem)),offset,Padded.spatial(problem),span,GatherLeafWrites.output(problem,storage,offset,span))>:
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))),Guards.output_channel_count(problem),
    Bounds.roundtrip(Guards.output_channel_count(problem),Machine.capacity(limits),Limits.capacity_fits(limits),Output.channel_capacity_bound(limits,problem,Support.fast(limits,problem,supported)))) :
    WriteReadback.WriteEvidence<GatherLeafWrites.query(problem,channel,output_row,output_column),Values.matrix_value(problem,channel,output_row,output_column),capacity,
      Blocks.writes(_,0,Gather.gather_width(offset,Padded.spatial(problem),span),offset,Padded.spatial(problem),span,GatherLeafWrites.output(problem,storage,offset,span))>
  channels(Guards.output_channel_count(problem),limits,problem,storage,input,offset,span,0,channel,output_row,output_column,capacity,valid,supported,certificate,Order.reflexive(Guards.output_channel_count(problem)),room,row_inside,column_inside)
