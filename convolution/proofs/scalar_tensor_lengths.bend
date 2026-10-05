# Runtime tensor-length expressions equal the validated logical list lengths.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ./convolution_guard_facts.bend as Guards
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./native_sample_refinement.bend as NativeSampleRefinement
import ./native_weight_refinement.bend as NativeWeightRefinement
import ./reduction_coordinate_bounds.bend as Coordinates
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ./supported_padded_coordinates.bend as Padded
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic

def input_shape(+limits: Machine.Limits,+problem: ConvSpec.Problem,valid: Guards.Valid<problem>,
  supported: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {Nat.mul(Nat.mul(U32.to_nat(U32.from_nat(Guards.input_channel_count(problem))),U32.to_nat(U32.from_nat(Guards.height(problem)))),U32.to_nat(U32.from_nat(Guards.width(problem)))) ==
    ConvSpec.length(Guards.input(problem)) : Nat}:
  %Equal.sym(Nat,ConvSpec.length(Guards.input(problem)),
    Nat.mul(U32.to_nat(U32.from_nat(Guards.input_channel_count(problem))),Nat.mul(U32.to_nat(U32.from_nat(Guards.height(problem))),U32.to_nat(U32.from_nat(Guards.width(problem))))),
    NativeSampleRefinement.input_shape(limits,problem,valid,supported)) :
    {Nat.mul(Nat.mul(U32.to_nat(U32.from_nat(Guards.input_channel_count(problem))),U32.to_nat(U32.from_nat(Guards.height(problem)))),U32.to_nat(U32.from_nat(Guards.width(problem)))) == _ : Nat}
  RowMajor.assoc(U32.to_nat(U32.from_nat(Guards.input_channel_count(problem))),U32.to_nat(U32.from_nat(Guards.height(problem))),U32.to_nat(U32.from_nat(Guards.width(problem))))

def input_length(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,
  +supported: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {U32.to_nat((U32.from_nat(Guards.input_channel_count(problem)) * U32.from_nat(Guards.height(problem)) * U32.from_nat(Guards.width(problem)) : U32)) == ConvSpec.length(Guards.input(problem)) : Nat}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = supported
  +shape = input_shape(limits,problem,valid,supported)
  %shape : {U32.to_nat((U32.from_nat(Guards.input_channel_count(problem)) * U32.from_nat(Guards.height(problem)) * U32.from_nat(Guards.width(problem)) : U32)) == _ : Nat}
  Encoded.multiply_three(U32.from_nat(Guards.input_channel_count(problem)),U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.width(problem)),
    Machine.capacity(limits),Limits.capacity_fits(limits),NativeSampleRefinement.width_positive(limits,problem,valid,supported),
    %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(U32.from_nat(Guards.input_channel_count(problem))),U32.to_nat(U32.from_nat(Guards.height(problem)))),U32.to_nat(U32.from_nat(Guards.width(problem)))),
      ConvSpec.length(Guards.input(problem)),shape) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    h7)

def bias_length(+limits: Machine.Limits,+problem: ConvSpec.Problem,valid: Guards.Valid<problem>,
  supported: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))) == ConvSpec.length(Guards.bias(problem)) : Nat}:
  Guards.Valid{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10} = valid
  %Equal.sym(Nat,ConvSpec.length(Guards.bias(problem)),Guards.output_channel_count(problem),Guards.reflect(ConvSpec.length(Guards.bias(problem)),Guards.output_channel_count(problem),h10)) :
    {U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))) == _ : Nat}
  SupportedCoordinateBounds.co_exact(limits,problem,supported)

def output_shape(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,
  +supported: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {Nat.mul(Nat.mul(U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))),U32.to_nat(U32.from_nat(ConvSpec.out_height(problem)))),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))) ==
    Nat.mul(Guards.output_channel_count(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) : Nat}:
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))),Guards.output_channel_count(problem),SupportedCoordinateBounds.co_exact(limits,problem,supported)) :
    {Nat.mul(Nat.mul(_,U32.to_nat(U32.from_nat(ConvSpec.out_height(problem)))),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))) == Nat.mul(Guards.output_channel_count(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) : Nat}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_height(problem))),ConvSpec.out_height(problem),
    Bounds.roundtrip(ConvSpec.out_height(problem),Machine.coordinate(limits),Limits.coordinate_fits(limits),SupportedCoordinateBounds.row_limit(limits,problem,valid,supported))) :
    {Nat.mul(Nat.mul(Guards.output_channel_count(problem),_),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))) == Nat.mul(Guards.output_channel_count(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) : Nat}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ConvSpec.out_width(problem),Coordinates.width_exact(limits,problem,valid,supported)) :
    {Nat.mul(Nat.mul(Guards.output_channel_count(problem),ConvSpec.out_height(problem)),_) == Nat.mul(Guards.output_channel_count(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) : Nat}
  RowMajor.assoc(Guards.output_channel_count(problem),ConvSpec.out_height(problem),ConvSpec.out_width(problem))

def output_length(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,
  +supported: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {U32.to_nat((U32.from_nat(Guards.output_channel_count(problem)) * U32.from_nat(ConvSpec.out_height(problem)) * U32.from_nat(ConvSpec.out_width(problem)) : U32)) ==
    Nat.mul(Guards.output_channel_count(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) : Nat}:
  +shape = output_shape(limits,problem,valid,supported)
  %shape : {U32.to_nat((U32.from_nat(Guards.output_channel_count(problem)) * U32.from_nat(ConvSpec.out_height(problem)) * U32.from_nat(ConvSpec.out_width(problem)) : U32)) == _ : Nat}
  Encoded.multiply_three(U32.from_nat(Guards.output_channel_count(problem)),U32.from_nat(ConvSpec.out_height(problem)),U32.from_nat(ConvSpec.out_width(problem)),Machine.capacity(limits),Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ConvSpec.out_width(problem),Coordinates.width_exact(limits,problem,valid,supported)) : {Nat.is_lt(0n,_) == True{} : Bool}
    Coordinates.width_positive(problem),
    %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))),U32.to_nat(U32.from_nat(ConvSpec.out_height(problem)))),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))),
      Nat.mul(Guards.output_channel_count(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))),shape) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    Padded.scalar_output_bound(limits,problem,supported))
