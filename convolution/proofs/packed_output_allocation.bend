import ../../lib/proofs/storage_shape.bend as Shape
# Bounds for the actual output allocation. No requested buffer size or
# successful write is supplied as a premise.
import Base
import ../../lib/storage_buffer.bend as Storage
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/kernel_gather.bend as Gather
import ../scalar_kernel.bend as ScalarConv
import ../../lib/proofs/write_program.bend as Writes
import ./storage_capacity.bend as StorageCapacity
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/array_allocation_bounds.bend as ArrayAllocation
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./packed_leaf_bounds.bend as LeafBounds
import ./packed_output_bounds.bend as Output
import ./packing_tile_bounds.bend as PackingBounds

def count(problem: ConvSpec.Problem,span: U32) -> U32:
  (U32.from_nat(Guards.output_channel_count(problem)) * span : U32)

def depth(problem: ConvSpec.Problem,span: U32) -> Nat:
  Storage.allocation_depth(count(problem,span))

def channel_positive(+problem: ConvSpec.Problem,valid: Guards.Valid<problem>) -> {Nat.is_lt(0n,Guards.output_channel_count(problem)) == True{} : Bool}:
  Guards.Valid{input_channels,output_channels,height,width,kernel,stride,fits_height,fits_width,input_length,weight_length,bias_length} = valid
  output_channels

type Ready<-count: U32,-depth: Nat> is Data:
  Ready{depth_bound: {Nat.is_le(depth,30n) == True{} : Bool},room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(F32,Tree.fresh(F32,depth,0.0)))) == True{} : Bool}}

def from_allocation(+count: U32,proof: ArrayAllocation.Allocation(count)) -> Ready<count,Storage.allocation_depth(count)>:
  Shape.Both{depth_bound,room} = proof
  Ready{depth_bound,
    %Equal.sym(U32,Tree.capacity(F32,Tree.fresh(F32,Storage.allocation_depth(count),0.0)),U32.shln(1,Storage.allocation_depth(count)),StorageCapacity.exact(Storage.allocation_depth(count),Tree.fresh(F32,Storage.allocation_depth(count),0.0),Shape.fresh(F32,Storage.allocation_depth(count),0.0))) :
      {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    room}

def allocated(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,+group: Nat,geometry: LeafBounds.Geometry<limits,problem,span>,group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool}) ->
  Ready<count(problem,span),depth(problem,span)>:
  LeafBounds.Geometry{+valid,+supported,+span_bound,alignment} = geometry
  +exact = Output.count_exact(limits,problem,span,supported,span_bound)
  from_allocation(count(problem,span),ArrayAllocation.allocated(count(problem,span),Machine.capacity(limits),Limits.capacity_fits(limits),Limits.capacity_allocatable(limits),
    %Equal.sym(Nat,U32.to_nat(count(problem,span)),Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Support.positive_mul(Guards.output_channel_count(problem),U32.to_nat(span),channel_positive(problem,valid),Order.le_lt_trans(0n,Nat.mul(group,8n),U32.to_nat(span),Order.zero_le(Nat.mul(group,8n)),Output.group_start_inside(span,group,group_inside))),
    %Equal.sym(Nat,U32.to_nat(count(problem,span)),Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)),exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    Output.elements_bound(limits,problem,span,supported,span_bound)))