# Coordinates of a channel copy, discharged from the existing packed guard.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_comparison.bend as Comparison
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./packed_output_bounds.bend as Output
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ./gather_output_index.bend as GatherOutputIndex

def source(row: U32,span: U32,column: U32) -> U32:
  (row * span + column : U32)

def destination(row: U32,spatial: U32,offset: U32,column: U32) -> U32:
  (row * spatial + offset + column : U32)

def spatial_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+supported: Support.FastSupported<limits,problem>) ->
  {U32.is_le(Padded.spatial(problem),Machine.capacity(limits)) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(Padded.spatial(problem),Machine.capacity(limits)),Nat.is_le(U32.to_nat(Padded.spatial(problem)),U32.to_nat(Machine.capacity(limits))),Comparison.le(Padded.spatial(problem),Machine.capacity(limits))) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Padded.spatial(problem)),GatherCopyCoordinates.spatial(problem),Padded.spatial_exact(limits,problem,valid,supported)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Padded.spatial_bound_facts(limits,problem,valid,Padded.scalar(limits,problem,supported))

def source_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,+row: U32,+column: U32,
  supported: Support.FastSupported<limits,problem>,span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  in_row: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},in_column: {Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool}) ->
  {U32.to_nat(source(row,span,column)) == Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(span)),U32.to_nat(column)) : Nat}:
  GatherOutputIndex.madd_words(row,span,column,Machine.capacity(limits),Limits.capacity_fits(limits),
    Output.index_bound(limits,problem,span,U32.to_nat(row),U32.to_nat(column),supported,span_bound,in_row,in_column))

def source_inside(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,+row: U32,+column: U32,
  supported: Support.FastSupported<limits,problem>,span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +in_row: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},+in_column: {Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(source(row,span,column)),Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(source(row,span,column)),Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(span)),U32.to_nat(column)),source_exact(limits,problem,span,row,column,supported,span_bound,in_row,in_column)) :
    {Nat.is_lt(_,Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span))) == True{} : Bool}
  RowMajor.radix(U32.to_nat(row),Guards.output_channel_count(problem),U32.to_nat(column),U32.to_nat(span),in_row,in_column)

def source_row(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,+row: U32,+column: U32,
  supported: Support.FastSupported<limits,problem>,span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  in_row: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},+in_column: {Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool}) ->
  {GatherCopyCoordinates.row(source(row,span,column),span) == U32.to_nat(row) : Nat}:
  %Equal.sym(Nat,U32.to_nat(source(row,span,column)),Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(span)),U32.to_nat(column)),source_exact(limits,problem,span,row,column,supported,span_bound,in_row,in_column)) :
    {Nat.div(_,U32.to_nat(span)) == U32.to_nat(row) : Nat}
  Radix.quotient(U32.to_nat(row),U32.to_nat(column),U32.to_nat(span),Order.le_lt_trans(0n,U32.to_nat(column),U32.to_nat(span),Order.zero_le(U32.to_nat(column)),in_column),in_column)

def source_column(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,+row: U32,+column: U32,
  supported: Support.FastSupported<limits,problem>,span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  in_row: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},+in_column: {Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool}) ->
  {GatherCopyCoordinates.column(source(row,span,column),span) == U32.to_nat(column) : Nat}:
  %Equal.sym(Nat,U32.to_nat(source(row,span,column)),Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(span)),U32.to_nat(column)),source_exact(limits,problem,span,row,column,supported,span_bound,in_row,in_column)) :
    {Nat.mod(_,U32.to_nat(span)) == U32.to_nat(column) : Nat}
  Radix.remainder(U32.to_nat(row),U32.to_nat(column),U32.to_nat(span),Order.le_lt_trans(0n,U32.to_nat(column),U32.to_nat(span),Order.zero_le(U32.to_nat(column)),in_column),in_column)

def destination_inside(+problem: ConvSpec.Problem,+offset: U32,+row: U32,+column: U32,
  in_row: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},kept: {Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(column)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool}) ->
  {Nat.is_lt(GatherOutputIndex.destination(U32.to_nat(row),U32.to_nat(column),GatherCopyCoordinates.spatial(problem),U32.to_nat(offset)),Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem))) == True{} : Bool}:
  RowMajor.radix(U32.to_nat(row),Guards.output_channel_count(problem),Nat.add(U32.to_nat(offset),U32.to_nat(column)),GatherCopyCoordinates.spatial(problem),in_row,kept)

def destination_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+row: U32,+column: U32,
  valid: Guards.Valid<problem>,+supported: Support.FastSupported<limits,problem>,
  in_row: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},kept: {Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(column)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool}) ->
  {U32.to_nat(destination(row,Padded.spatial(problem),offset,column)) == GatherOutputIndex.destination(U32.to_nat(row),U32.to_nat(column),GatherCopyCoordinates.spatial(problem),U32.to_nat(offset)) : Nat}:
  GatherOutputIndex.combine(row,column,Padded.spatial(problem),offset,U32.to_nat(row),U32.to_nat(column),GatherCopyCoordinates.spatial(problem),U32.to_nat(offset),Machine.capacity(limits),Limits.capacity_fits(limits),{==},{==},Padded.spatial_exact(limits,problem,valid,supported),{==},
    Bounds.strict_le(GatherOutputIndex.destination(U32.to_nat(row),U32.to_nat(column),GatherCopyCoordinates.spatial(problem),U32.to_nat(offset)),U32.to_nat(Machine.capacity(limits)),
      Bounds.nat_trans(GatherOutputIndex.destination(U32.to_nat(row),U32.to_nat(column),GatherCopyCoordinates.spatial(problem),U32.to_nat(offset)),Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(Machine.capacity(limits)),
        destination_inside(problem,offset,row,column,in_row,kept),Padded.scalar_output_bound(limits,problem,Padded.scalar(limits,problem,supported)))))
