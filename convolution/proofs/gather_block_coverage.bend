# Every valid matrix coordinate is reached by the actual nested copy loops.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/kernel_gather.bend as Gather
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/loop_projection.bend as Projection
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./prepared_packing_samples.bend as Prepared
import ./packed_output_bounds.bend as Output
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Coordinates
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ./gather_block_coordinates.bend as BlockCoordinates
import ./gather_block_width.bend as Width
import ./gather_blocks.bend as Blocks
import ./gather_leaf_writes.bend as GatherLeafWrites
import ./gather_write_program.bend as GatherWrites
import ../../lib/proofs/write_program.bend as Writes

def columns(+prefix: Nat,+suffix: Nat,+input: Tree.Tree<F32>,+position: U32,+source: U32,+destination: U32,+target: U32,+query: Nat,
  reached: {target == (position + U32.from_nat(prefix) : U32) : U32},equal: {query == U32.to_nat((destination + target : U32)) : Nat}) ->
  {GatherWrites.contains(query,Contiguous.writes(~F32,Nat.add(prefix,1n+suffix),input,position,source,destination)) == True{} : Bool}:
  match prefix:
    case 0n:
      +same = Equal.trans(U32,target,(position + 0 : U32),position,reached,Arithmetic.add_zero(position))
      +address = {%same : {query == U32.to_nat((destination + _ : U32)) : Nat} equal : {query == U32.to_nat((destination + position : U32)) : Nat}}
      %Equal.sym(Nat,query,U32.to_nat((destination + position : U32)),address) :
        {GatherWrites.contains(_,Contiguous.writes(~F32,1n+suffix,input,position,source,destination)) == True{} : Bool}
      %Equal.sym(Bool,Nat.is_eq(U32.to_nat((destination + position : U32)),U32.to_nat((destination + position : U32))),True{},Order.equal_reflexive(U32.to_nat((destination + position : U32)))) :
        {_ || GatherWrites.contains(U32.to_nat((destination + position : U32)),Contiguous.writes(~F32,suffix,input,(position + 1 : U32),source,destination)) == True{} : Bool}
      {==}
    case 1n+ +rest:
      GatherWrites.prepend_keeps_coverage(True{},query,(destination + position : U32),Tree.read(F32,input,(source + position : U32)),
        Contiguous.writes(~F32,Nat.add(rest,1n+suffix),input,(position + 1 : U32),source,destination),
        columns(rest,suffix,input,(position + 1 : U32),source,destination,target,query,
          Equal.trans(U32,target,(position + U32.from_nat(1n+rest) : U32),((position + 1 : U32) + U32.from_nat(rest) : U32),reached,Projection.indexed_successor(position,rest)),equal))

def channels(+prefix: Nat,+suffix: Nat,+input: Tree.Tree<F32>,+row: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+target: U32,+query: Nat,
  reached: {target == (row + U32.from_nat(prefix) : U32) : U32},
  covered: {GatherWrites.contains(query,Contiguous.writes(~F32,U32.to_nat(width),input,0,(target * span : U32),(target * spatial + offset : U32))) == True{} : Bool}) ->
  {GatherWrites.contains(query,Blocks.writes(Nat.add(prefix,1n+suffix),row,width,offset,spatial,span,input)) == True{} : Bool}:
  match prefix:
    case 0n:
      +same = Equal.trans(U32,target,(row + 0 : U32),row,reached,Arithmetic.add_zero(row))
      GatherWrites.concatenation_covered(query,Contiguous.writes(~F32,U32.to_nat(width),input,0,(row * span : U32),(row * spatial + offset : U32)),
        Blocks.writes(suffix,(row + 1 : U32),width,offset,spatial,span,input),
        Inl{%same : {GatherWrites.contains(query,Contiguous.writes(~F32,U32.to_nat(width),input,0,(_ * span : U32),(_ * spatial + offset : U32))) == True{} : Bool} covered})
    case 1n+ +rest:
      GatherWrites.concatenation_covered(query,Contiguous.writes(~F32,U32.to_nat(width),input,0,(row * span : U32),(row * spatial + offset : U32)),
        Blocks.writes(Nat.add(rest,1n+suffix),(row + 1 : U32),width,offset,spatial,span,input),
        Inr{channels(rest,suffix,input,(row + 1 : U32),width,offset,spatial,span,target,query,
          Equal.trans(U32,target,(row + U32.from_nat(1n+rest) : U32),((row + 1 : U32) + U32.from_nat(rest) : U32),reached,Projection.indexed_successor(row,rest)),covered)})

def target_destination(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+row: U32,+column: U32,+channel: Nat,+local_column: Nat,+output_row: Nat,+output_column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +row_exact: {U32.to_nat(row) == channel : Nat},+column_exact: {U32.to_nat(column) == local_column : Nat},
  position: {Nat.add(U32.to_nat(offset),local_column) == Coordinates.position(problem,output_row,output_column) : Nat},
  in_row: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+in_output: {Nat.is_lt(Nat.add(U32.to_nat(offset),local_column),GatherCopyCoordinates.spatial(problem)) == True{} : Bool}) ->
  {U32.to_nat(BlockCoordinates.destination(row,Padded.spatial(problem),offset,column)) == GatherLeafWrites.query(problem,channel,output_row,output_column) : Nat}:
  %position : {U32.to_nat(BlockCoordinates.destination(row,Padded.spatial(problem),offset,column)) == Nat.add(Nat.mul(channel,GatherCopyCoordinates.spatial(problem)),_) : Nat}
  %row_exact : {U32.to_nat(BlockCoordinates.destination(row,Padded.spatial(problem),offset,column)) == Nat.add(Nat.mul(_,GatherCopyCoordinates.spatial(problem)),Nat.add(U32.to_nat(offset),local_column)) : Nat}
  %column_exact : {U32.to_nat(BlockCoordinates.destination(row,Padded.spatial(problem),offset,column)) == Nat.add(Nat.mul(U32.to_nat(row),GatherCopyCoordinates.spatial(problem)),Nat.add(U32.to_nat(offset),_)) : Nat}
  BlockCoordinates.destination_exact(limits,problem,offset,row,column,Guards.valid(problem,valid),Support.fast(limits,problem,supported),
    %Equal.sym(Nat,U32.to_nat(row),channel,row_exact) : {Nat.is_lt(_,Guards.output_channel_count(problem)) == True{} : Bool} in_row,
    %Equal.sym(Nat,U32.to_nat(column),local_column,column_exact) : {Nat.is_lt(Nat.add(U32.to_nat(offset),_),GatherCopyCoordinates.spatial(problem)) == True{} : Bool} in_output)

def block(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificate: Prepared.LeafCertificate(problem,offset,span,input),
  +in_channel: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+in_column: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row,output_column) : Nat},row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {GatherWrites.contains(GatherLeafWrites.query(problem,channel,output_row,output_column),Blocks.block_writes(U32.from_nat(Guards.output_channel_count(problem)),offset,Padded.spatial(problem),span,GatherLeafWrites.output(problem,storage,offset,span))) == True{} : Bool}:
  Prepared.LeafEvidence{packing_contract,+span_bound,offset_alignment,span_alignment,range_bound,contents} = certificate
  +channel_bound = Output.channel_capacity_bound(limits,problem,Support.fast(limits,problem,supported))
  +channel_exact = Bounds.roundtrip(channel,Machine.capacity(limits),Limits.capacity_fits(limits),Bounds.strict_le(channel,U32.to_nat(Machine.capacity(limits)),Bounds.nat_trans(channel,Guards.output_channel_count(problem),U32.to_nat(Machine.capacity(limits)),in_channel,channel_bound)))
  +column_exact = Bounds.roundtrip(column,Machine.capacity(limits),Limits.capacity_fits(limits),Bounds.strict_le(column,U32.to_nat(Machine.capacity(limits)),Bounds.nat_trans(column,U32.to_nat(span),U32.to_nat(Machine.capacity(limits)),in_column,Output.span_in_capacity(limits,problem,span,Guards.valid(problem,valid),Support.fast(limits,problem,supported),span_bound))))
  +in_output = {%Equal.sym(Nat,Nat.add(U32.to_nat(offset),column),Coordinates.position(problem,output_row,output_column),position) : {Nat.is_lt(_,GatherCopyCoordinates.spatial(problem)) == True{} : Bool}
    GatherLeafWrites.position_inside(problem,output_row,output_column,row_inside,column_inside) : {Nat.is_lt(Nat.add(U32.to_nat(offset),column),GatherCopyCoordinates.spatial(problem)) == True{} : Bool}}
  +in_width = Width.covers(offset,Padded.spatial(problem),span,column,Machine.capacity(limits),Limits.capacity_fits(limits),BlockCoordinates.spatial_bound(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported)),in_column,
    %Equal.sym(Nat,U32.to_nat(Padded.spatial(problem)),GatherCopyCoordinates.spatial(problem),Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported))) : {Nat.is_lt(Nat.add(U32.to_nat(offset),column),_) == True{} : Bool} in_output)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))),Guards.output_channel_count(problem),Bounds.roundtrip(Guards.output_channel_count(problem),Machine.capacity(limits),Limits.capacity_fits(limits),channel_bound)) :
    {GatherWrites.contains(GatherLeafWrites.query(problem,channel,output_row,output_column),Blocks.writes(_,0,Gather.gather_width(offset,Padded.spatial(problem),span),offset,Padded.spatial(problem),span,GatherLeafWrites.output(problem,storage,offset,span))) == True{} : Bool}
  %Order.split_at_element(Guards.output_channel_count(problem),channel,in_channel) :
    {GatherWrites.contains(GatherLeafWrites.query(problem,channel,output_row,output_column),Blocks.writes(_,0,Gather.gather_width(offset,Padded.spatial(problem),span),offset,Padded.spatial(problem),span,GatherLeafWrites.output(problem,storage,offset,span))) == True{} : Bool}
  channels(channel,Nat.sub(Guards.output_channel_count(problem),1n+channel),GatherLeafWrites.output(problem,storage,offset,span),0,Gather.gather_width(offset,Padded.spatial(problem),span),offset,Padded.spatial(problem),span,U32.from_nat(channel),GatherLeafWrites.query(problem,channel,output_row,output_column),
    Equal.sym(U32,(0 + U32.from_nat(channel) : U32),U32.from_nat(channel),Arithmetic.add_zero_left(U32.from_nat(channel))),
    %Order.split_at_element(U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span)),column,in_width) :
      {GatherWrites.contains(GatherLeafWrites.query(problem,channel,output_row,output_column),Contiguous.writes(~F32,_,GatherLeafWrites.output(problem,storage,offset,span),0,(U32.from_nat(channel) * span : U32),(U32.from_nat(channel) * Padded.spatial(problem) + offset : U32))) == True{} : Bool}
    columns(column,Nat.sub(U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span)),1n+column),GatherLeafWrites.output(problem,storage,offset,span),0,(U32.from_nat(channel) * span : U32),(U32.from_nat(channel) * Padded.spatial(problem) + offset : U32),U32.from_nat(column),GatherLeafWrites.query(problem,channel,output_row,output_column),
      Equal.sym(U32,(0 + U32.from_nat(column) : U32),U32.from_nat(column),Arithmetic.add_zero_left(U32.from_nat(column))),
      Equal.sym(Nat,U32.to_nat(BlockCoordinates.destination(U32.from_nat(channel),Padded.spatial(problem),offset,U32.from_nat(column))),GatherLeafWrites.query(problem,channel,output_row,output_column),
        target_destination(limits,problem,offset,U32.from_nat(channel),U32.from_nat(column),channel,column,output_row,output_column,valid,supported,channel_exact,column_exact,position,in_channel,in_output))))
