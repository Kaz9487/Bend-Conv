# The actual tile allocation count is bounded by the validated fast guard.
import Base
import ../../lib/machine_limits.bend as Machine
import ../specification.bend as ConvSpec
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Coordinates
import ./tile_partition_bounds.bend as TilePartitionBounds
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic

def groups(span: U32) -> Nat:
  Nat.div(U32.to_nat(span),8n)

def count(span: U32,p: ConvSpec.Problem) -> U32:
  (span / 8 * Coordinates.native_inner(p) : U32)

def groups_exact(+limits: Machine.Limits,+span: U32) -> {U32.to_nat((span / 8 : U32)) == groups(span) : Nat}:
  U32Division.divide(span,8,Machine.half_word(limits),Limits.half_word_fits(limits),Bounds.as_nat_le(16,
      Machine.half_word(limits),Limits.sixteen_within_half(limits)),{==})

def packing_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Support.FastSupported<limits,
  p>) -> {Nat.is_le(Nat.mul(Support.padded(p),Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = p
  ConvSpec.Shape{+channels,+outputs,+height,+width,+kernel,+stride,+padding} = shape
  Support.FastSupported{scalar,packing,output} = support
  packing

def elements_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,+span: U32,support: Support.FastSupported<limits,p>,
  span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool}) -> {Nat.is_le(Nat.mul(groups(span),Coordinates.inner(p)),
    U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +group_bound = Order.transitive(groups(span),U32.to_nat(span),Support.padded(p),Order.transitive(groups(span),Nat.mul(groups(span),8n),
      U32.to_nat(span),RowMajor.product_covers(groups(span),8n,{==}),TilePartitionBounds.quotient_product(span,8n,{==})),span_bound)
  Order.transitive(Nat.mul(groups(span),Coordinates.inner(p)),Nat.mul(Support.padded(p),Coordinates.inner(p)),
    U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le(groups(span),Support.padded(p),Coordinates.inner(p),group_bound),
    packing_bound(limits,p,support))

def count_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,+span: U32,+valid: Guards.Valid<p>,+support: Support.FastSupported<limits,p>,
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool}) -> {U32.to_nat(count(span,p)) == Nat.mul(groups(span),
    Coordinates.inner(p)) : Nat}:
  +group_exact = groups_exact(limits,span)
  +inner_exact = Coordinates.inner_exact(limits,p,valid,Padded.scalar(limits,p,support))
  %group_exact : {U32.to_nat(count(span,p)) == Nat.mul(_,Coordinates.inner(p)) : Nat}
  %inner_exact : {U32.to_nat(count(span,p)) == Nat.mul(U32.to_nat((span / 8 : U32)),_) : Nat}
  Arithmetic.multiply((span / 8 : U32),Coordinates.native_inner(p),Machine.capacity(limits),Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat((span / 8 : U32)),groups(span),group_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(Coordinates.native_inner(p))),
        U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(p)),Coordinates.inner(p),inner_exact) : {Nat.is_le(Nat.mul(groups(span),_),
        U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    elements_bound(limits,p,span,support,span_bound))

def count_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,+span: U32,+valid: Guards.Valid<p>,+support: Support.FastSupported<limits,p>,
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(count(span,p)),
    U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(count(span,p)),Nat.mul(groups(span),Coordinates.inner(p)),count_exact(limits,p,span,valid,support,
      span_bound)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  elements_bound(limits,p,span,support,span_bound)
