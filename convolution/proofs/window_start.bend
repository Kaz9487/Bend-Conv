# The packing start of a window, as a natural number: the original offset
# moved back by block whole output rows.
import Base
import ../specification.bend as ConvSpec
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/u32_shift.bend as U32Shift
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/natural_addition.bend as Addition
import ./convolution_guard_facts.bend as Guards
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./window_samples.bend as Samples
import ./window_positions.bend as Positions

def stride_word(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +support: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {U32.to_nat(U32.from_nat(Guards.stride(problem))) == Guards.stride(problem) : Nat}:
  Bounds.roundtrip(Guards.stride(problem),Machine.dimension(limits),Limits.dimension_fits(limits),SupportedCoordinateBounds.stride_bound(limits,problem,support))

def stride_divisor(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +support: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {Nat.is_le(Nat.double(U32.to_nat(U32.from_nat(Guards.stride(problem)))),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  +dimension = Machine.dimension(limits)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.stride(problem))),Guards.stride(problem),stride_word(limits,problem,support)) :
    {Nat.is_le(Nat.double(_),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
  Order.transitive(Nat.double(Guards.stride(problem)),Nat.double(U32.to_nat(dimension)),U32.to_nat(Machine.coordinate(limits)),
    Order.double_le(Guards.stride(problem),U32.to_nat(dimension),SupportedCoordinateBounds.stride_bound(limits,problem,support)),
    %U32Shift.double(dimension,Limits.dimension_below_half(limits)) : {Nat.is_le(_,U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
    Bounds.as_nat_le(U32.shl(dimension),Machine.coordinate(limits),Limits.doubled_dimension_within_coordinate(limits)))

def start_when(+limits: Machine.Limits,+problem: ConvSpec.Problem,+facts: Guards.Valid<problem>,
  +support: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +offset: U32,+first: U32,+block: Nat,+zero: Bool,+zero_exact: {U32.is_zero(first) == zero : Bool},
  +first_exact: {U32.to_nat(first) == Nat.mul(block,Guards.stride(problem)) : Nat},
  +reach: {Nat.is_le(Nat.mul(block,ConvSpec.out_width(problem)),U32.to_nat(offset)) == True{} : Bool},
  +offset_room: {Nat.is_le(U32.to_nat(offset),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first)) ==
    Nat.sub(U32.to_nat(offset),Nat.mul(block,ConvSpec.out_width(problem))) : Nat}:
  Guards.Valid{ci_positive,co_positive,height_positive,width_positive,kernel_positive,+stride_positive,kernel_height,kernel_width,
    input_length,weight_length,bias_length} = facts
  match zero:
    case True{}:
      +first_zero = Order.equal_of_is_eq(U32.to_nat(first),0n,
        %Comparison.eq(first,0) : {_ == True{} : Bool}
        zero_exact)
      +block_zero = Positions.zero_factor(block,Guards.stride(problem),stride_positive,
        Equal.trans(Nat,Nat.mul(block,Guards.stride(problem)),U32.to_nat(first),0n,Equal.sym(Nat,U32.to_nat(first),Nat.mul(block,Guards.stride(problem)),first_exact),first_zero))
      %Equal.sym(Bool,U32.is_zero(first),True{},zero_exact) :
        {U32.to_nat(Bool.pick(U32,_,offset,(offset - first / U32.from_nat(Guards.stride(problem)) * U32.from_nat(ConvSpec.out_width(problem)) : U32))) ==
          Nat.sub(U32.to_nat(offset),Nat.mul(block,ConvSpec.out_width(problem))) : Nat}
      %Equal.sym(Nat,block,0n,block_zero) : {U32.to_nat(offset) == Nat.sub(U32.to_nat(offset),Nat.mul(_,ConvSpec.out_width(problem))) : Nat}
      Equal.sym(Nat,Nat.sub(U32.to_nat(offset),0n),U32.to_nat(offset),Order.subtract_zero(U32.to_nat(offset)))
    case False{}:
      +stride_u = U32.from_nat(Guards.stride(problem))
      +width_u = U32.from_nat(ConvSpec.out_width(problem))
      +capacity = Machine.capacity(limits)
      +quotient = (first / stride_u : U32)
      +product = (quotient * width_u : U32)
      +quotient_exact = {Equal.trans(Nat,U32.to_nat(quotient),Nat.div(U32.to_nat(first),U32.to_nat(stride_u)),block,
          U32Division.divide(first,stride_u,Machine.coordinate(limits),Limits.coordinate_fits(limits),stride_divisor(limits,problem,support),
            %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.stride(problem))),Guards.stride(problem),stride_word(limits,problem,support)) : {Nat.is_lt(0n,_) == True{} : Bool}
            stride_positive),
          %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.stride(problem))),Guards.stride(problem),stride_word(limits,problem,support)) : {Nat.div(U32.to_nat(first),_) == block : Nat}
          %Equal.sym(Nat,U32.to_nat(first),Nat.mul(block,Guards.stride(problem)),first_exact) : {Nat.div(_,Guards.stride(problem)) == block : Nat}
          Positions.exact_quotient(block,Guards.stride(problem),stride_positive)) : {U32.to_nat(quotient) == block : Nat}}
      +width_exact = Coordinates.width_exact(limits,problem,facts,support)
      +product_exact = {Equal.trans(Nat,U32.to_nat(product),Nat.mul(U32.to_nat(quotient),U32.to_nat(width_u)),Nat.mul(block,ConvSpec.out_width(problem)),
          Arithmetic.multiply(quotient,width_u,capacity,Limits.capacity_fits(limits),
            %Equal.sym(Nat,U32.to_nat(quotient),block,quotient_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(width_u)),U32.to_nat(capacity)) == True{} : Bool}
            %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ConvSpec.out_width(problem),width_exact) : {Nat.is_le(Nat.mul(block,_),U32.to_nat(capacity)) == True{} : Bool}
            Order.transitive(Nat.mul(block,ConvSpec.out_width(problem)),U32.to_nat(offset),U32.to_nat(capacity),reach,offset_room)),
          %Equal.sym(Nat,U32.to_nat(quotient),block,quotient_exact) : {Nat.mul(_,U32.to_nat(width_u)) == Nat.mul(block,ConvSpec.out_width(problem)) : Nat}
          %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ConvSpec.out_width(problem),width_exact) : {Nat.mul(block,_) == Nat.mul(block,ConvSpec.out_width(problem)) : Nat}
          {==}) : {U32.to_nat(product) == Nat.mul(block,ConvSpec.out_width(problem)) : Nat}}
      %Equal.sym(Bool,U32.is_zero(first),False{},zero_exact) :
        {U32.to_nat(Bool.pick(U32,_,offset,(offset - product : U32))) == Nat.sub(U32.to_nat(offset),Nat.mul(block,ConvSpec.out_width(problem))) : Nat}
      %product_exact : {U32.to_nat((offset - product : U32)) == Nat.sub(U32.to_nat(offset),_) : Nat}
      Arithmetic.subtract(offset,product,capacity,Limits.capacity_fits(limits),
        %Equal.sym(Nat,U32.to_nat(product),Nat.mul(block,ConvSpec.out_width(problem)),product_exact) : {Nat.is_le(_,U32.to_nat(offset)) == True{} : Bool}
        reach,
        Order.transitive(Nat.sub(U32.to_nat(offset),U32.to_nat(product)),U32.to_nat(offset),U32.to_nat(capacity),
          Order.sub_le(U32.to_nat(offset),U32.to_nat(product)),offset_room))

def start_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+facts: Guards.Valid<problem>,
  +support: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +offset: U32,+first: U32,+block: Nat,
  +first_exact: {U32.to_nat(first) == Nat.mul(block,Guards.stride(problem)) : Nat},
  +reach: {Nat.is_le(Nat.mul(block,ConvSpec.out_width(problem)),U32.to_nat(offset)) == True{} : Bool},
  +offset_room: {Nat.is_le(U32.to_nat(offset),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first)) ==
    Nat.sub(U32.to_nat(offset),Nat.mul(block,ConvSpec.out_width(problem))) : Nat}:
  start_when(limits,problem,facts,support,offset,first,block,U32.is_zero(first),{==},first_exact,reach,offset_room)
