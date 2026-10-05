# Real compact addresses and padding decisions, derived from prepare and guards.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/row_major_index.bend as RowMajor
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packed_output_bounds.bend as Output
import ./gather_output_index.bend as GatherOutputIndex
import ./gather_write_program.bend as GatherWrites

type Geometry<-limits: Machine.Limits,-problem: ConvSpec.Problem,-offset: U32,-span: U32> is Data:
  Geometry{valid: Guards.Valid<problem>,supported: Support.FastSupported<limits,problem>,
    span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
    range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool}}

def row(index: U32,+span: U32) -> Nat:
  Nat.div(U32.to_nat(index),U32.to_nat(span))

def column(index: U32,+span: U32) -> Nat:
  Nat.mod(U32.to_nat(index),U32.to_nat(span))

def spatial(+problem: ConvSpec.Problem) -> Nat:
  Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))

def span_positive(+problem: ConvSpec.Problem,+span: U32,+index: U32,
  inside: {Nat.is_lt(U32.to_nat(index),Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span))) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool}:
  RowMajor.left_factor_positive(U32.to_nat(span),Guards.output_channel_count(problem),
    %RowMajor.multiply_commutative(Guards.output_channel_count(problem),U32.to_nat(span)) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,U32.to_nat(index),Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)),Order.zero_le(U32.to_nat(index)),inside))

def row_inside(+problem: ConvSpec.Problem,+span: U32,+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span))) == True{} : Bool}) ->
  {Nat.is_lt(row(index,span),Guards.output_channel_count(problem)) == True{} : Bool}:
  RowMajor.row_inside(row(index,span),Guards.output_channel_count(problem),U32.to_nat(span),column(index,span),
    %Radix.division_identity(index,U32.to_nat(span),span_positive(problem,span,index,inside)) : {Nat.is_lt(_,Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span))) == True{} : Bool}
    inside)
