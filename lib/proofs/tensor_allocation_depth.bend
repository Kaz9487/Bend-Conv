# Tail-recursive runtime depth calculation refines the original bit-length.
import Base
import ../tensor.bend as Tensors
import ../storage_buffer.bend as Storage
import ./natural_addition.bend as Addition
import ./nat_division.bend as Division
import ./u32_comparison.bend as Comparison
import ./u32_extrema.bend as U32Extrema
import ./word_encoding.bend as WordEncoding
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./array_allocation_bounds.bend as ArrayAllocation

def branch(+remaining: Nat,+value: U32,+bits: Nat,+zero: Bool,
  tail: {Tensors.significant_bits(remaining,U32.shr(value),1n+bits,U32.is_zero(U32.shr(value))) ==
    Nat.add(1n+bits,Storage.bit_length(remaining,U32.shr(value))) : Nat}) ->
  {Tensors.significant_bits(1n+remaining,value,bits,zero) ==
    Nat.add(bits,Bool.pick(Nat,zero,0n,1n+Storage.bit_length(remaining,U32.shr(value)))) : Nat}:
  match zero:
    case True{}: Equal.sym(Nat,Nat.add(bits,0n),bits,Addition.zero_right(bits))
    case False{}:
      Equal.trans(Nat,Tensors.significant_bits(remaining,U32.shr(value),1n+bits,U32.is_zero(U32.shr(value))),
        Nat.add(1n+bits,Storage.bit_length(remaining,U32.shr(value))),Nat.add(bits,1n+Storage.bit_length(remaining,U32.shr(value))),tail,
        Equal.sym(Nat,Nat.add(bits,1n+Storage.bit_length(remaining,U32.shr(value))),1n+Nat.add(bits,Storage.bit_length(remaining,U32.shr(value))),
          Addition.successor_right(bits,Storage.bit_length(remaining,U32.shr(value)))))

def countdown(+remaining: Nat,+value: U32,+bits: Nat) ->
  {Tensors.significant_bits(remaining,value,bits,U32.is_zero(value)) == Nat.add(bits,Storage.bit_length(remaining,value)) : Nat}:
  match remaining:
    case 0n: Equal.sym(Nat,Nat.add(bits,0n),bits,Addition.zero_right(bits))
    case 1n+ +rest: branch(rest,value,bits,U32.is_zero(value),countdown(rest,U32.shr(value),1n+bits))

def depth(+count: U32) ->
  {Tensors.depth_for(count) == Storage.allocation_depth(U32.max(count,1)) : Nat}:
  countdown(32n,(U32.max(count,1) - 1 : U32),0n)

def positive_not_below_one(+count: Nat,positive: {Nat.is_lt(0n,count) == True{} : Bool}) ->
  {Nat.is_lt(count,1n) == False{} : Bool}:
  match count:
    case 0n: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(positive))
    case 1n+rest: Division.lt_zero(rest)

def positive_max(+count: U32,positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) ->
  {U32.max(count,1) == count : U32}:
  %U32Extrema.pick_maximum(count,1) : {_ == count : U32}
  %Equal.sym(Bool,U32.is_lt(count,1),False{},Equal.trans(Bool,U32.is_lt(count,1),Nat.is_lt(U32.to_nat(count),1n),False{},
    Comparison.lt(count,1),positive_not_below_one(U32.to_nat(count),positive))) : {Bool.pick(U32,_,1,count) == count : U32}
  {==}

def positive_depth(+count: U32,positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) ->
  {Tensors.depth_for(count) == Storage.allocation_depth(count) : Nat}:
  %positive_max(count,positive) : {Tensors.depth_for(count) == Storage.allocation_depth(_) : Nat}
  depth(count)

def not_small_positive(+count: Nat,not_small: {Nat.is_lt(count,1n) == False{} : Bool}) -> {Nat.is_lt(0n,count) == True{} : Bool}:
  match count:
    case 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},not_small)))
    case 1n+rest: {==}

type Maximum<-count: U32,-maximum: U32,-upper: U32> is Data:
  Maximum{positive: {Nat.is_lt(0n,U32.to_nat(maximum)) == True{} : Bool},
    bounded: {Nat.is_le(U32.to_nat(maximum),U32.to_nat(upper)) == True{} : Bool},
    includes: {Nat.is_le(U32.to_nat(count),U32.to_nat(maximum)) == True{} : Bool}}

def selected_maximum(+count: U32,+upper: U32,small: Bool,compared: {U32.is_lt(count,1) == small : Bool},
  one_fits: {U32.is_le(1,upper) == True{} : Bool},bounded: {U32.is_le(count,upper) == True{} : Bool}) -> Maximum<count,Bool.pick(U32,small,1,count),upper>:
  match small:
    case True{}: Maximum{Bounds.as_nat_lt(0,1,{==}),Bounds.as_nat_le(1,upper,one_fits),Bounds.strict_le(U32.to_nat(count),1n,Bounds.as_nat_lt(count,1,compared))}
    case False{}: Maximum{not_small_positive(U32.to_nat(count),Equal.trans(Bool,Nat.is_lt(U32.to_nat(count),1n),U32.is_lt(count,1),False{},
      Equal.sym(Bool,U32.is_lt(count,1),Nat.is_lt(U32.to_nat(count),1n),Comparison.lt(count,1)),compared)),Bounds.as_nat_le(count,upper,bounded),Order.reflexive(U32.to_nat(count))}

type Sized<-count: U32> is Data:
  Sized{depth_ok: {Nat.is_le(Tensors.depth_for(count),30n) == True{} : Bool},
    room: {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,Tensors.depth_for(count)))) == True{} : Bool}}

def allocation_room(+count: U32,ready: Sized<count>) -> {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,Tensors.depth_for(count)))) == True{} : Bool}:
  Sized{depth_ok,room} = ready
  room

def ready_maximum(+count: U32,+upper: U32,upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},
  maximum: Maximum<count,U32.max(count,1),upper>) -> Sized<count>:
  Maximum{positive,bounded,+includes} = maximum
  ArrayAllocation.with_bounds(Sized<count>,U32.max(count,1),ArrayAllocation.allocated(U32.max(count,1),upper,upper_ok,upper_bound,positive,bounded),
    depth_ok => room => Sized{
      %Equal.sym(Nat,Tensors.depth_for(count),Storage.allocation_depth(U32.max(count,1)),depth(count)) : {Nat.is_le(_,30n) == True{} : Bool} depth_ok,
      %Equal.sym(Nat,Tensors.depth_for(count),Storage.allocation_depth(U32.max(count,1)),depth(count)) : {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,_))) == True{} : Bool}
      Order.transitive(U32.to_nat(count),U32.to_nat(U32.max(count,1)),U32.to_nat(U32.shln(1,Storage.allocation_depth(U32.max(count,1)))),includes,room)})

# The zero-element allocation is included: it owns one initialized cell while
# its logical length remains zero. No positive-count premise escapes to callers.
def bounded_with(+count: U32,+upper: U32,upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},
  one_fits: {U32.is_le(1,upper) == True{} : Bool},within: {U32.is_le(count,upper) == True{} : Bool}) -> Sized<count>:
  ready_maximum(count,upper,upper_ok,upper_bound,
    %U32Extrema.pick_maximum(count,1) : Maximum<count,_,upper>
    selected_maximum(count,upper,U32.is_lt(count,1),{==},one_fits,within))

def bounded(+count: U32,within: {U32.is_le(count,1073741824) == True{} : Bool}) -> Sized<count>:
  bounded_with(count,1073741824,{==},{==},{==},within)
