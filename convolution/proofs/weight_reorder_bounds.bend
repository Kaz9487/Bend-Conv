# The existing valid/supported guards suffice for the eight-row packed allocation.
# No additional shape guard or caller-provided allocation premise is introduced.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor.bend as Tensors
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/kernel_block_geometry.bend as Geometry
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Coordinates
import ./native_weight_refinement.bend as NativeWeightRefinement
import ../../lib/proofs/array_allocation_bounds.bend as ArrayAllocation

def blocks(problem: ConvSpec.Problem) -> U32:
  ((U32.from_nat(Guards.output_channel_count(problem)) + 7) / 8 : U32)

def count(+problem: ConvSpec.Problem) -> U32:
  (blocks(problem) * Coordinates.native_inner(problem) : U32)

def padded_rows_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,+supported: Support.FastSupported<limits,problem>) ->
  {Nat.is_le(Nat.add(Guards.output_channel_count(problem),7n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +capacity = U32.to_nat(Machine.capacity(limits))
  %Equal.sym(Nat,U32.to_nat(Machine.half_word(limits)),Nat.double(capacity),Limits.doubled_capacity(limits)) :
    {Nat.is_le(Nat.add(Guards.output_channel_count(problem),7n),_) == True{} : Bool}
  %Addition.double(capacity) : {Nat.is_le(Nat.add(Guards.output_channel_count(problem),7n),_) == True{} : Bool}
  Encoded.sum_bounds(Guards.output_channel_count(problem),7n,capacity,capacity,
    NativeWeightRefinement.dimension(limits,Guards.output_channel_count(problem),SupportedCoordinateBounds.co_bound(limits,problem,Padded.scalar(limits,problem,supported))),
    Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits)))

def blocks_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+supported: Support.FastSupported<limits,problem>) ->
  {U32.to_nat(blocks(problem)) == Geometry.blocks(Guards.output_channel_count(problem)) : Nat}:
  +rows = Guards.output_channel_count(problem)
  +word = U32.from_nat(rows)
  +row_exact = SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,supported))
  +added = Arithmetic.add(word,7,Machine.half_word(limits),Limits.half_word_fits(limits),
    %Equal.sym(Nat,U32.to_nat(word),rows,row_exact) : {Nat.is_le(Nat.add(_,7n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    padded_rows_bound(limits,problem,supported))
  %Order.commutative(rows,7n) : {U32.to_nat(blocks(problem)) == Nat.div(_,8n) : Nat}
  %row_exact : {U32.to_nat(blocks(problem)) == Nat.div(Nat.add(_,7n),8n) : Nat}
  %added : {U32.to_nat(blocks(problem)) == Nat.div(_,8n) : Nat}
  U32Division.divide((word + 7 : U32),8,Machine.half_word(limits),Limits.half_word_fits(limits),
    Bounds.as_nat_le(16,Machine.half_word(limits),Limits.sixteen_within_half(limits)),{==})

def elements_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+supported: Support.FastSupported<limits,problem>) ->
  {Nat.is_le(Nat.mul(Geometry.blocks(Guards.output_channel_count(problem)),Coordinates.inner(problem)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  Order.transitive(Nat.mul(Geometry.blocks(Guards.output_channel_count(problem)),Coordinates.inner(problem)),Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem)),
    U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le(Geometry.blocks(Guards.output_channel_count(problem)),Guards.output_channel_count(problem),Coordinates.inner(problem),
      Geometry.count_bound(Guards.output_channel_count(problem))),
    %Guards.reflect(ConvSpec.length(Guards.weights(problem)),Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem)),v9) :
      {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    NativeWeightRefinement.weights_bound(limits,problem,Padded.scalar(limits,problem,supported)))

def count_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+supported: Support.FastSupported<limits,problem>) ->
  {U32.to_nat(count(problem)) == Nat.mul(Geometry.blocks(Guards.output_channel_count(problem)),Coordinates.inner(problem)) : Nat}:
  +blocks_exact = blocks_exact(limits,problem,supported)
  +inner_exact = Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,supported))
  %blocks_exact : {U32.to_nat(count(problem)) == Nat.mul(_,Coordinates.inner(problem)) : Nat}
  %inner_exact : {U32.to_nat(count(problem)) == Nat.mul(U32.to_nat(blocks(problem)),_) : Nat}
  Arithmetic.multiply(blocks(problem),Coordinates.native_inner(problem),Machine.capacity(limits),Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat(blocks(problem)),Geometry.blocks(Guards.output_channel_count(problem)),blocks_exact) :
      {Nat.is_le(Nat.mul(_,U32.to_nat(Coordinates.native_inner(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(problem)),Coordinates.inner(problem),inner_exact) :
      {Nat.is_le(Nat.mul(Geometry.blocks(Guards.output_channel_count(problem)),_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    elements_bound(limits,problem,valid,supported))

def count_positive(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+supported: Support.FastSupported<limits,problem>) ->
  {Nat.is_lt(0n,U32.to_nat(count(problem))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  %Equal.sym(Nat,U32.to_nat(count(problem)),Nat.mul(Geometry.blocks(Guards.output_channel_count(problem)),Coordinates.inner(problem)),count_exact(limits,problem,valid,supported)) :
    {Nat.is_lt(0n,_) == True{} : Bool}
  Support.positive_mul(Geometry.blocks(Guards.output_channel_count(problem)),Coordinates.inner(problem),Geometry.contains_row(Guards.output_channel_count(problem),0n,v1),
    Support.positive_mul(Guards.input_channel_count(problem),Nat.mul(Guards.kernel(problem),Guards.kernel(problem)),v0,
      Support.positive_mul(Guards.kernel(problem),Guards.kernel(problem),v4,v4)))

def Ready(+problem: ConvSpec.Problem) -> Data:
  Shape.Both<{Nat.is_le(Tensors.depth_for(count(problem)),30n) == True{} : Bool},
    {Nat.is_le(U32.to_nat(count(problem)),U32.to_nat(U32.shln(1,Tensors.depth_for(count(problem))))) == True{} : Bool}>

def allocated(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+supported: Support.FastSupported<limits,problem>) -> Ready(problem):
  +positive = count_positive(limits,problem,valid,supported)
  %Equal.sym(Nat,Tensors.depth_for(count(problem)),Storage.allocation_depth(count(problem)),Depth.positive_depth(count(problem),positive)) :
    Shape.Both<{Nat.is_le(_,30n) == True{} : Bool},{Nat.is_le(U32.to_nat(count(problem)),U32.to_nat(U32.shln(1,_))) == True{} : Bool}>
  ArrayAllocation.allocated(count(problem),Machine.capacity(limits),Limits.capacity_fits(limits),Limits.capacity_allocatable(limits),positive,
    %Equal.sym(Nat,U32.to_nat(count(problem)),Nat.mul(Geometry.blocks(Guards.output_channel_count(problem)),Coordinates.inner(problem)),count_exact(limits,problem,valid,supported)) :
      {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    elements_bound(limits,problem,valid,supported))
