# All output-address bounds come from the unchanged runtime fast guard.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packing_tile_bounds.bend as PackingBounds
import ./tile_partition_bounds.bend as TilePartitionBounds
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic

def channel_capacity_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,supported: Support.FastSupported<limits,problem>) ->
  {Nat.is_le(Guards.output_channel_count(problem),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(Guards.output_channel_count(problem),U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.capacity(limits)),
    SupportedCoordinateBounds.co_bound(limits,problem,Padded.scalar(limits,problem,supported)),
    Bounds.as_nat_le(Machine.dimension(limits),Machine.capacity(limits),Limits.dimension_within_capacity(limits)))

def padded_elements_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,supported: Support.FastSupported<limits,problem>) ->
  {Nat.is_le(Nat.mul(Support.padded(problem),Guards.output_channel_count(problem)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{input_channel_count,output_channel_count,height,width,kernel,stride,padding} = shape
  Support.FastSupported{scalar,packing,output} = supported
  output

def elements_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,supported: Support.FastSupported<limits,problem>,
  span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)),Nat.mul(U32.to_nat(span),Guards.output_channel_count(problem)),RowMajor.multiply_commutative(Guards.output_channel_count(problem),U32.to_nat(span))) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Order.transitive(Nat.mul(U32.to_nat(span),Guards.output_channel_count(problem)),Nat.mul(Support.padded(problem),Guards.output_channel_count(problem)),U32.to_nat(Machine.capacity(limits)),
    RowMajor.mul_le(U32.to_nat(span),Support.padded(problem),Guards.output_channel_count(problem),span_bound),padded_elements_bound(limits,problem,supported))

def count_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,+supported: Support.FastSupported<limits,problem>,
  span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}) ->
  {U32.to_nat((U32.from_nat(Guards.output_channel_count(problem)) * span : U32)) == Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)) : Nat}:
  +channels_exact = SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,supported))
  %channels_exact : {U32.to_nat((U32.from_nat(Guards.output_channel_count(problem)) * span : U32)) == Nat.mul(_,U32.to_nat(span)) : Nat}
  Arithmetic.multiply(U32.from_nat(Guards.output_channel_count(problem)),span,Machine.capacity(limits),Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))),Guards.output_channel_count(problem),channels_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    elements_bound(limits,problem,span,supported,span_bound))

def span_in_capacity(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,valid: Guards.Valid<problem>,supported: Support.FastSupported<limits,problem>,
  span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(span),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(U32.to_nat(span),Support.padded(problem),U32.to_nat(Machine.capacity(limits)),span_bound,Padded.padded_bound(limits,problem,valid,supported))

def group_start_inside(+span: U32,+group: Nat,inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(group,8n),U32.to_nat(span)) == True{} : Bool}:
  Bounds.nat_trans(Nat.mul(group,8n),Nat.mul(PackingBounds.groups(span),8n),U32.to_nat(span),RowMajor.multiply_lt(group,PackingBounds.groups(span),8n,inside,{==}),
    TilePartitionBounds.quotient_product(span,8n,{==}))

def index_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,+channel: Nat,+column: Nat,supported: Support.FastSupported<limits,problem>,
  span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(channel,U32.to_nat(span)),column),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Bounds.strict_le(Nat.add(Nat.mul(channel,U32.to_nat(span)),column),U32.to_nat(Machine.capacity(limits)),
    Bounds.nat_trans(Nat.add(Nat.mul(channel,U32.to_nat(span)),column),Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits)),
      RowMajor.radix(channel,Guards.output_channel_count(problem),column,U32.to_nat(span),channel_inside,column_inside),elements_bound(limits,problem,span,supported,span_bound)))

