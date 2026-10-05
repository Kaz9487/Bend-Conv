import ./storage_shape.bend as Shape
import Base
import ../storage_buffer.bend as Storage
import ./allocation_bit_bounds.bend as AllocationBitBounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./u32_division.bend as U32Division
import ./word_encoding.bend as WordEncoding

def sub_zero(+n: Nat) -> {Nat.sub(n,0n) == n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      {==}

def positive_one(+n: Nat,h: {Nat.is_lt(0n,n) == True{} : Bool}) -> {Nat.is_le(1n,n) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(h))
    case 1n+p:
      Order.zero_le(p)

def prior_lt(+n: Nat,+cap: Nat,pos: {Nat.is_lt(0n,n) == True{} : Bool},bound: {Nat.is_le(n,cap) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(n,1n),
    cap) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({Nat.is_lt(0n,cap) == True{} : Bool},WordEncoding.false_true(pos))
    case 1n+ +p:
      %Equal.sym(Nat,Nat.sub(p,0n),p,sub_zero(p)) : {Nat.is_lt(_,cap) == True{} : Bool}
      Bounds.predecessor_strict(p,cap,bound)

def strict_follow(+a: Nat,+b: Nat,h: {Nat.is_lt(a,b) == True{} : Bool}) -> {Nat.is_le(1n+a,b) == True{} : Bool}:
  Order.strict_follow(a,b,h)

def following(+n: Nat,+cap: Nat,bound: {Nat.is_lt(Nat.sub(n,1n),cap) == True{} : Bool}) -> {Nat.is_le(n,
    cap) == True{} : Bool}:
  match n:
    case 0n:
      Order.zero_le(cap)
    case 1n+ +p:
      %sub_zero(p) : {Nat.is_le(1n+_,cap) == True{} : Bool}
      strict_follow(Nat.sub(p,0n),cap,bound)

def preceding(+n: U32,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},+pos: {Nat.is_lt(0n,U32.to_nat(n)) == True{} : Bool},
  +bound: {Nat.is_le(U32.to_nat(n),U32.to_nat(upper)) == True{} : Bool}) -> {U32.to_nat((n - 1 : U32)) == Nat.sub(U32.to_nat(n),1n) : Nat}:
  Arithmetic.subtract(n,1,upper,upper_ok,positive_one(U32.to_nat(n),pos),Order.transitive(Nat.sub(U32.to_nat(n),1n),U32.to_nat(n),
      U32.to_nat(upper),Order.sub_le(U32.to_nat(n),1n),bound))

def machine_prior(+n: U32,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},+pos: {Nat.is_lt(0n,
      U32.to_nat(n)) == True{} : Bool},+bound: {Nat.is_le(U32.to_nat(n),U32.to_nat(upper)) == True{} : Bool}) -> {U32.is_lt((n - 1 : U32),
    upper) == True{} : Bool}:
  U32Division.nat_lt_native((n - 1 : U32),upper,
    %Equal.sym(Nat,U32.to_nat((n - 1 : U32)),Nat.sub(U32.to_nat(n),1n),preceding(n,upper,upper_ok,pos,bound)) : {Nat.is_lt(_,
        U32.to_nat(upper)) == True{} : Bool}
    prior_lt(U32.to_nat(n),U32.to_nat(upper),pos,bound))

def Allocation(+n: U32) -> Data:
  Shape.Both<{Nat.is_le(Storage.allocation_depth(n),30n) == True{} : Bool},{Nat.is_le(U32.to_nat(n),U32.to_nat(U32.shln(1,
          Storage.allocation_depth(n)))) == True{} : Bool}>

def finish(+n: U32,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},+pos: {Nat.is_lt(0n,U32.to_nat(n)) == True{} : Bool},
  +bound: {Nat.is_le(U32.to_nat(n),U32.to_nat(upper)) == True{} : Bool},proof: AllocationBitBounds.LengthGoal((n - 1 : U32))) -> Allocation(n):
  Shape.Both{depth,room} = proof
  Shape.Both{depth,
    following(U32.to_nat(n),U32.to_nat(U32.shln(1,Storage.allocation_depth(n))),
      %preceding(n,upper,upper_ok,pos,bound) : {Nat.is_lt(_,U32.to_nat(U32.shln(1,Storage.allocation_depth(n)))) == True{} : Bool}
      Bounds.as_nat_lt((n - 1 : U32),U32.shln(1,Storage.allocation_depth(n)),room))}

law allocated:
  for +n: U32
  for +upper: U32
  for +upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool}
  for +upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool}
  for +positive: {Nat.is_lt(0n,U32.to_nat(n)) == True{} : Bool}
  for +bound: {Nat.is_le(U32.to_nat(n),U32.to_nat(upper)) == True{} : Bool}
  Allocation(n)

def allocated(n,upper,upper_ok,upper_bound,positive,bound):
  finish(n,upper,upper_ok,positive,bound,AllocationBitBounds.width((n - 1 : U32),Bounds.uint_trans((n - 1 : U32),upper,1073741824,
        machine_prior(n,upper,upper_ok,positive,bound),upper_bound)))

# Eliminate allocator evidence without destructuring a computed dependent value.
def with_bounds(-Result: Type,+count: U32,proof: Allocation(count),body: {Nat.is_le(Storage.allocation_depth(count),
      30n) == True{} : Bool} -> {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,
        Storage.allocation_depth(count)))) == True{} : Bool} -> Result) -> Result:
  Shape.Both{depth_ok,room} = proof
  body(depth_ok,room)
