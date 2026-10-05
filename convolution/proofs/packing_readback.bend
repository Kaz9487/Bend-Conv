import ../../lib/proofs/storage_shape.bend as Shape
# Read the flat tabulation of a packed buffer through its proved dense prefix.
# packing_contents proves the actual packer stores this tabulation.
import Base
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/fp32_arithmetic_laws.bend as ArithmeticLaws
import ../../lib/proofs/indexed_sequence_laws.bend as SequenceLaws
import ../../lib/math_sequence.bend as Sequence
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/row_major_index.bend as RowMajor
import ./packing_tabulation.bend as PackingTabulation
import ./packing_lane_reads.bend as LaneReads
import ./tile_array_projection.bend as ArrayProjection
import ./packed_reduction_samples.bend as ReductionSamples
import ./dense_array_initialization.bend as DenseArrayInitialization
import ../../lib/proofs/array_allocation_bounds.bend as ArrayAllocation
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./storage_capacity.bend as StorageCapacity
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../specification.bend as ConvSpec
import ../../lib/kernel_tile.bend as Tile
import ../scalar_kernel.bend as ScalarConv

def lane_values_at(+count: Nat,+index: Nat,+offset: Nat,+start: U32,+input: Tree.Tree<F32>,+height: U32,+width: U32,+kernel: U32,
  +stride: U32,+padding: U32,+output_width: U32,+inner: U32,+lane: Sequence.Index(8n),inside: {Nat.is_lt(index,
      count) == True{} : Bool}) -> {ConvSpec.at(index,PackingTabulation.lane_values(count,U32.from_nat(offset),start,input,height,width,kernel,stride,
      padding,output_width,inner,lane)) == ReductionSamples.select(lane,PackingTabulation.entry(U32.from_nat(Nat.add(index,offset)),start,input,height,width,
      kernel,stride,padding,output_width,inner)) : F32}:
  match count:
    case 0n:
      Empty.absurd({ConvSpec.at(index,PackingTabulation.lane_values(0n,U32.from_nat(offset),start,input,height,width,kernel,stride,padding,output_width,
              inner,lane)) == ReductionSamples.select(lane,PackingTabulation.entry(U32.from_nat(Nat.add(index,offset)),start,input,height,width,kernel,stride,
              padding,output_width,inner)) : F32},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,
              Nat.is_lt(index,0n),False{},Division.lt_zero(index)),inside)))
    case 1n+ +rest:
      match index:
        case 0n:
          {==}
        case 1n+ +previous:
          %Equal.sym(U32,(U32.from_nat(offset) + 1 : U32),U32.from_nat(1n+offset),
            Bounds.add_one(U32.from_nat(offset))) : {ConvSpec.at(previous,PackingTabulation.lane_values(rest,_,start,input,height,width,kernel,stride,
                padding,output_width,inner,lane)) == ReductionSamples.select(lane,PackingTabulation.entry(U32.from_nat(1n+Nat.add(previous,offset)),start,input,
                height,width,kernel,stride,padding,output_width,inner)) : F32}
          %Addition.successor_right(previous,offset) : {ConvSpec.at(previous,PackingTabulation.lane_values(rest,U32.from_nat(1n+offset),start,input,
                height,width,kernel,stride,padding,output_width,inner,lane)) == ReductionSamples.select(lane,PackingTabulation.entry(U32.from_nat(_),start,input,
                height,width,kernel,stride,padding,output_width,inner)) : F32}
          lane_values_at(rest,previous,1n+offset,start,input,height,width,kernel,stride,padding,output_width,inner,lane,inside)


def readback(+count: Nat,+start: U32,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+height: U32,+width: U32,+kernel: U32,
  +stride: U32,+padding: U32,+output_width: U32,+inner: U32,+lane: Sequence.Index(8n),+query: U32,+depth: Nat,
  +balanced: Shape.Balanced(F32,depth,ArrayProjection.project(output,lane)),+depth_ok: {Nat.is_le(depth,
      30n) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(Tree.capacity(F32,ArrayProjection.project(output,lane)))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(query),count) == True{} : Bool}) -> {ReductionSamples.select(lane,Tree.read(Tile.Tile8,PackingTabulation.fill(count,0,
        start,input,output,height,width,kernel,stride,padding,output_width,inner),query)) == ReductionSamples.select(lane,PackingTabulation.entry(query,start,
      input,height,width,kernel,stride,padding,output_width,inner)) : F32}:
  +filled = PackingTabulation.fill(count,0,start,input,output,height,width,kernel,stride,padding,output_width,inner)
  +values = PackingTabulation.lane_values(count,0,start,input,height,width,kernel,stride,padding,output_width,inner,lane)
  +length = PackingTabulation.length(count,0,start,input,height,width,kernel,stride,padding,output_width,inner,lane)
  %Arithmetic.native_roundtrip(query) : {ReductionSamples.select(lane,Tree.read(Tile.Tile8,filled,query)) == ReductionSamples.select(lane,
      PackingTabulation.entry(_,start,input,height,width,kernel,stride,padding,output_width,inner)) : F32}
  %Order.commutative(U32.to_nat(query),0n) : {ReductionSamples.select(lane,Tree.read(Tile.Tile8,filled,query)) == ReductionSamples.select(lane,
      PackingTabulation.entry(U32.from_nat(_),start,input,height,width,kernel,stride,padding,output_width,inner)) : F32}
  %lane_values_at(count,U32.to_nat(query),0n,start,input,height,width,kernel,stride,padding,output_width,inner,lane,
    inside) : {ReductionSamples.select(lane,Tree.read(Tile.Tile8,filled,query)) == _ : F32}
  %ArrayProjection.read_projected(filled,lane,query) : {_ == ConvSpec.at(U32.to_nat(query),values) : F32}
  %Equal.sym(Tree.Tree<F32>,ArrayProjection.project(filled,lane),Tree.write_sequence(~F32,values,0,ArrayProjection.project(output,lane)),
    PackingTabulation.projected(count,0,start,input,output,height,width,kernel,stride,padding,output_width,inner,lane)) : {Tree.read(F32,_,
      query) == ConvSpec.at(U32.to_nat(query),values) : F32}
  DenseArrayInitialization.read_filled_prefix(values,ArrayProjection.project(output,lane),depth,query,balanced,depth_ok,
    %Equal.sym(Nat,ConvSpec.length(values),count,length) : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,ArrayProjection.project(output,
            lane)))) == True{} : Bool}
    room,
    %Equal.sym(Nat,ConvSpec.length(values),count,length) : {Nat.is_lt(U32.to_nat(query),_) == True{} : Bool}
    inside)

def lane_position(position: U32,lane: Sequence.Index(8n)) -> U32:
  PackingTabulation.indexed_position(position,Sequence.index_value(8n,lane))

def vector_lane(+input: Tree.Tree<F32>,+position: U32,+reduction: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,
  +output_width: U32,+lane: Sequence.Index(8n)) -> {ReductionSamples.select(lane,PackingTabulation.vector(input,position,reduction,height,width,kernel,stride,
      padding,output_width)) == LaneReads.value(input,lane_position(position,lane),reduction,height,width,kernel,stride,padding,output_width) : F32}:
  %Addition.zero_right(Sequence.index_value(8n,lane)) : {ReductionSamples.select(lane,PackingTabulation.vector(input,position,reduction,height,width,kernel,
        stride,padding,output_width)) == LaneReads.value(input,PackingTabulation.indexed_position(position,_),reduction,height,width,kernel,stride,padding,
      output_width) : F32}
  SequenceLaws.get_tabulated(~F32,~PackingTabulation.PixelContext,~PackingTabulation.sample_pixel,8n,lane,0n,PackingTabulation.PixelContext{input,position,reduction,height,width,kernel,stride,padding,
      output_width})

def zero_vector() -> Tile.Tile8:
  Tile.filled(0.0)

# The tabulation over a freshly allocated buffer; allocation facts come from the actual allocator.
def allocated(+count: U32) -> Tree.Tree<Tile.Tile8>:
  Tree.fresh(Tile.Tile8,Storage.allocation_depth(count),zero_vector())

law allocated_readback:
  for +limits: Machine.Limits
  for +count: U32
  for +start: U32
  for +input: Tree.Tree<F32>
  for +height: U32
  for +width: U32
  for +kernel: U32
  for +stride: U32
  for +padding: U32
  for +output_width: U32
  for +inner: U32
  for +lane: Sequence.Index(8n)
  for +query: U32
  for +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}
  for +bounded: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  for +inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}
  {ReductionSamples.select(lane,Tree.read(Tile.Tile8,PackingTabulation.fill(U32.to_nat(count),0,start,input,allocated(count),height,width,kernel,stride,padding,output_width,inner),
      query)) == LaneReads.value(input,lane_position((start + (query / inner) * 8 : U32),lane),(query % inner : U32),height,width,kernel,stride,padding,output_width) : F32}

def allocated_readback(limits,count,start,input,height,width,kernel,stride,padding,output_width,inner,lane,query,positive,bounded,inside):
  +depth = Storage.allocation_depth(count)
  +zero = zero_vector()
  +output = allocated(count)
  +filled = PackingTabulation.fill(U32.to_nat(count),0,start,input,output,height,width,kernel,stride,padding,output_width,inner)
  ArrayAllocation.with_bounds({ReductionSamples.select(lane,Tree.read(Tile.Tile8,filled,query)) == LaneReads.value(input,lane_position((start + (query / inner) * 8 : U32),lane),
        (query % inner : U32),height,width,kernel,stride,padding,output_width) : F32},count,
    ArrayAllocation.allocated(count,Machine.capacity(limits),Limits.capacity_fits(limits),Limits.capacity_allocatable(limits),
      positive,bounded),depth_ok => room =>
    +balanced = ArrayProjection.fresh_balanced(depth,zero,lane)
    %vector_lane(input,(start + (query / inner) * 8 : U32),(query % inner : U32),height,width,kernel,stride,padding,output_width,
      lane) : {ReductionSamples.select(lane,Tree.read(Tile.Tile8,filled,query)) == _ : F32}
    readback(U32.to_nat(count),start,input,output,height,width,kernel,stride,padding,output_width,inner,lane,query,depth,balanced,depth_ok,
      %Equal.sym(U32,Tree.capacity(F32,ArrayProjection.project(output,lane)),U32.shln(1,depth),StorageCapacity.exact(depth,ArrayProjection.project(output,lane),
          balanced)) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
      room,inside))

# Logical lane offsets preserve order; no floating-point algebra is involved.
def lane_offset(lane: Sequence.Index(8n)) -> Nat:
  Sequence.index_value(8n,lane)

def indexed_position_encoded(+position: Nat,index: Nat) -> {PackingTabulation.indexed_position(U32.from_nat(position),index) == U32.from_nat(Nat.add(position,
      index)) : U32}:
  match index:
    case 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) : {U32.from_nat(position) == U32.from_nat(_) : U32}
      {==}
    case 1n+rest:
      WordArithmetic.native_add(position,1n+rest)

def lane_position_encoded(position: Nat,lane: Sequence.Index(8n)) -> {lane_position(U32.from_nat(position),lane) == U32.from_nat(Nat.add(position,
      lane_offset(lane))) : U32}:
  indexed_position_encoded(position,Sequence.index_value(8n,lane))

def logical_position(+start: U32,group: Nat,lane: Sequence.Index(8n)) -> Nat:
  Nat.add(Nat.add(U32.to_nat(start),Nat.mul(group,8n)),lane_offset(lane))

def position_exact(+start: U32,+group: Nat,+lane: Sequence.Index(8n),+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +room: {Nat.is_le(logical_position(start,group,lane),
      U32.to_nat(upper)) == True{} : Bool}) -> {U32.to_nat(lane_position((start + U32.from_nat(group) * 8 : U32),
      lane)) == logical_position(start,group,lane) : Nat}:
  +product = Nat.mul(group,8n)
  +base = Nat.add(U32.to_nat(start),product)
  +base_bound = Order.transitive(base,logical_position(start,group,lane),U32.to_nat(upper),Order.le_add(base,lane_offset(lane)),room)
  +product_bound = Order.transitive(product,base,U32.to_nat(upper),
    %Order.commutative(product,U32.to_nat(start)) : {Nat.is_le(product,_) == True{} : Bool}
    Order.le_add(product,U32.to_nat(start)),base_bound)
  +group_exact = Bounds.roundtrip(group,upper,upper_ok,Order.transitive(group,product,U32.to_nat(upper),RowMajor.product_covers(group,
        8n,{==}),product_bound))
  +encoded_product = Arithmetic.multiply(U32.from_nat(group),8,upper,upper_ok,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(group)),group,group_exact) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(upper)) == True{} : Bool}
    product_bound)
  +product_exact = Equal.trans(Nat,U32.to_nat((U32.from_nat(group) * 8 : U32)),Nat.mul(U32.to_nat(U32.from_nat(group)),8n),product,
    encoded_product,Equal.cong(Nat,Nat,n => Nat.mul(n,8n),U32.to_nat(U32.from_nat(group)),group,group_exact))
  +base_encoded = Equal.trans(U32,(start + U32.from_nat(group) * 8 : U32),U32.from_nat(Nat.add(U32.to_nat(start),
        U32.to_nat((U32.from_nat(group) * 8 : U32)))),U32.from_nat(base),Arithmetic.native_add(start,(U32.from_nat(group) * 8 : U32)),
    Equal.cong(Nat,U32,n => U32.from_nat(Nat.add(U32.to_nat(start),n)),U32.to_nat((U32.from_nat(group) * 8 : U32)),product,product_exact))
  %Equal.sym(U32,(start + U32.from_nat(group) * 8 : U32),U32.from_nat(base),base_encoded) : {U32.to_nat(lane_position(_,
        lane)) == logical_position(start,group,lane) : Nat}
  %Equal.sym(U32,lane_position(U32.from_nat(base),lane),U32.from_nat(logical_position(start,group,lane)),lane_position_encoded(base,
      lane)) : {U32.to_nat(_) == logical_position(start,group,lane) : Nat}
  Bounds.roundtrip(logical_position(start,group,lane),upper,upper_ok,room)
