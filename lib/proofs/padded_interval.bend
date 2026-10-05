import ./storage_shape.bend as Shape
# Wrapped native subtraction and zero-padding guards denote the same range.
import Base
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./array_allocation_bounds.bend as ArrayAllocation
import ./u32_division.bend as U32Division
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding

def add_le(+a: Nat,+b: Nat,+c: Nat,h: {Nat.is_le(a,b) == True{} : Bool}) -> {Nat.is_le(Nat.add(a,c),Nat.add(b,c)) == True{} : Bool}:
  Order.add_le(a,b,c,h)

def add_lt(+a: Nat,+b: Nat,+c: Nat,h: {Nat.is_lt(a,b) == True{} : Bool}) -> {Nat.is_lt(Nat.add(a,c),Nat.add(b,c)) == True{} : Bool}:
  Order.add_lt(a,b,c,h)

def sum_room(+r: Nat,+extent: Nat,+pad: Nat,+cap: Nat,h: {Nat.is_lt(r,extent) == True{} : Bool},bound: {Nat.is_le(Nat.add(extent,pad),
      cap) == True{} : Bool}) -> {Nat.is_le(Nat.add(r,pad),cap) == True{} : Bool}:
  Order.transitive(Nat.add(r,pad),Nat.add(extent,pad),cap,add_le(r,extent,pad,Bounds.strict_le(r,extent,h)),bound)

def recovered(+padded_row: U32,+pad: U32,+extent: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  bound: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool},h: {U32.is_lt((padded_row - pad : U32),
      extent) == True{} : Bool}) -> {Nat.add(U32.to_nat((padded_row - pad : U32)),U32.to_nat(pad)) == U32.to_nat(padded_row) : Nat}:
  %Arithmetic.add((padded_row - pad : U32),pad,cap,cap_ok,sum_room(U32.to_nat((padded_row - pad : U32)),U32.to_nat(extent),U32.to_nat(pad),
      U32.to_nat(cap),Bounds.as_nat_lt((padded_row - pad : U32),extent,h),bound)) : {_ == U32.to_nat(padded_row) : Nat}
  %Subtraction.add_subtract(padded_row,pad) : {U32.to_nat((padded_row - pad + pad : U32)) == U32.to_nat(_) : Nat}
  {==}

def forward(+padded_row: U32,+pad: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +bound: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool},+h: {U32.is_lt((padded_row - pad : U32),
      extent) == True{} : Bool}) -> Shape.Both<{Nat.is_le(U32.to_nat(pad),U32.to_nat(padded_row)) == True{} : Bool},{Nat.is_lt(U32.to_nat(padded_row),
    Nat.add(U32.to_nat(pad),U32.to_nat(extent))) == True{} : Bool}>:
  +eq = recovered(padded_row,pad,extent,cap,cap_ok,bound,h)
  Shape.Both{
    %eq : {Nat.is_le(U32.to_nat(pad),_) == True{} : Bool}
    %Order.commutative(U32.to_nat(pad),U32.to_nat((padded_row - pad : U32))) : {Nat.is_le(U32.to_nat(pad),_) == True{} : Bool}
    Order.le_add(U32.to_nat(pad),U32.to_nat((padded_row - pad : U32))),
    %eq : {Nat.is_lt(_,Nat.add(U32.to_nat(pad),U32.to_nat(extent))) == True{} : Bool}
    %Order.commutative(U32.to_nat(extent),U32.to_nat(pad)) : {Nat.is_lt(Nat.add(U32.to_nat((padded_row - pad : U32)),U32.to_nat(pad)),_) == True{} : Bool}
    add_lt(U32.to_nat((padded_row - pad : U32)),U32.to_nat(extent),U32.to_nat(pad),Bounds.as_nat_lt((padded_row - pad : U32),extent,h))}

def reverse_bound(+a: Nat,+b: Nat,+cap: Nat,h: {Nat.is_le(Nat.add(a,b),cap) == True{} : Bool}) -> {Nat.is_le(Nat.add(b,a),cap) == True{} : Bool}:
  %Order.commutative(a,b) : {Nat.is_le(_,cap) == True{} : Bool}
  h

def backward(+padded_row: U32,+pad: U32,+extent: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +bound: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool},+lo: {Nat.is_le(U32.to_nat(pad),
      U32.to_nat(padded_row)) == True{} : Bool},+hi: {Nat.is_lt(U32.to_nat(padded_row),Nat.add(U32.to_nat(pad),
        U32.to_nat(extent))) == True{} : Bool}) -> {U32.is_lt((padded_row - pad : U32),extent) == True{} : Bool}:
  +sumbound = reverse_bound(U32.to_nat(extent),U32.to_nat(pad),U32.to_nat(cap),bound)
  +pybound = Bounds.strict_le(U32.to_nat(padded_row),U32.to_nat(cap),Bounds.nat_trans(U32.to_nat(padded_row),Nat.add(U32.to_nat(pad),
        U32.to_nat(extent)),U32.to_nat(cap),hi,sumbound))
  U32Division.nat_lt_native((padded_row - pad : U32),extent,
    %Equal.sym(Nat,U32.to_nat((padded_row - pad : U32)),Nat.sub(U32.to_nat(padded_row),U32.to_nat(pad)),Arithmetic.subtract(padded_row,pad,cap,cap_ok,lo,
        Order.transitive(Nat.sub(U32.to_nat(padded_row),U32.to_nat(pad)),U32.to_nat(padded_row),U32.to_nat(cap),Order.sub_le(U32.to_nat(padded_row),
            U32.to_nat(pad)),pybound))) : {Nat.is_lt(_,U32.to_nat(extent)) == True{} : Bool}
    Order.sub_lt(U32.to_nat(padded_row),U32.to_nat(pad),U32.to_nat(extent),lo,hi))

def conjunction(+a: Bool,+b: Bool,ha: {a == True{} : Bool},hb: {b == True{} : Bool}) -> {a && b == True{} : Bool}:
  %Equal.sym(Bool,a,True{},ha) : {_ && b == True{} : Bool}
  hb

def parts(+a: Bool,+b: Bool,h: {a && b == True{} : Bool}) -> Shape.Both<{a == True{} : Bool},{b == True{} : Bool}>:
  match a b:
    case False{} False{}:
      Empty.absurd(Shape.Both<{False{} == True{} : Bool},{False{} == True{} : Bool}>,WordEncoding.false_true(h))
    case False{} True{}:
      Empty.absurd(Shape.Both<{False{} == True{} : Bool},{True{} == True{} : Bool}>,WordEncoding.false_true(h))
    case True{} False{}:
      Empty.absurd(Shape.Both<{True{} == True{} : Bool},{False{} == True{} : Bool}>,WordEncoding.false_true(h))
    case True{} True{}:
      Shape.Both{{==},{==}}

def iff(+a: Bool,+b: Bool,ab: {a == True{} : Bool} -> {b == True{} : Bool},ba: {b == True{} : Bool} -> {a == True{} : Bool}) -> {a == b : Bool}:
  match a b:
    case False{} False{}:
      {==}
    case False{} True{}:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(ba({==})))
    case True{} False{}:
      Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(ab({==})))
    case True{} True{}:
      {==}

def joined(+padded_row: U32,+pad: U32,+extent: U32,p: Shape.Both<{Nat.is_le(U32.to_nat(pad),U32.to_nat(padded_row)) == True{} : Bool},
  {Nat.is_lt(U32.to_nat(padded_row),Nat.add(U32.to_nat(pad),U32.to_nat(extent))) == True{} : Bool}>) -> {Nat.is_le(U32.to_nat(pad),
    U32.to_nat(padded_row)) && Nat.is_lt(U32.to_nat(padded_row),Nat.add(U32.to_nat(pad),U32.to_nat(extent))) == True{} : Bool}:
  Shape.Both{lo,hi} = p
  conjunction(Nat.is_le(U32.to_nat(pad),U32.to_nat(padded_row)),Nat.is_lt(U32.to_nat(padded_row),Nat.add(U32.to_nat(pad),U32.to_nat(extent))),lo,hi)

def split_backward(+padded_row: U32,+pad: U32,+extent: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  bound: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool},
  p: Shape.Both<{Nat.is_le(U32.to_nat(pad),U32.to_nat(padded_row)) == True{} : Bool},{Nat.is_lt(U32.to_nat(padded_row),Nat.add(U32.to_nat(pad),
        U32.to_nat(extent))) == True{} : Bool}>) -> {U32.is_lt((padded_row - pad : U32),extent) == True{} : Bool}:
  Shape.Both{lo,hi} = p
  backward(padded_row,pad,extent,cap,cap_ok,bound,lo,hi)

law range:
  for +padded_row: U32
  for +pad: U32
  for +extent: U32
  for +cap: U32
  for +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool}
  for +bound: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool}
  {U32.is_lt((padded_row - pad : U32),extent) == Nat.is_le(U32.to_nat(pad),U32.to_nat(padded_row)) && Nat.is_lt(U32.to_nat(padded_row),Nat.add(U32.to_nat(pad),
        U32.to_nat(extent))) : Bool}
def range(padded_row,pad,extent,cap,cap_ok,bound):
  iff(U32.is_lt((padded_row - pad : U32),extent),Nat.is_le(U32.to_nat(pad),U32.to_nat(padded_row)) && Nat.is_lt(U32.to_nat(padded_row),Nat.add(U32.to_nat(pad),
        U32.to_nat(extent))),
    h => joined(padded_row,pad,extent,forward(padded_row,pad,extent,cap,cap_ok,bound,h)),
    h => split_backward(padded_row,pad,extent,cap,cap_ok,bound,parts(Nat.is_le(U32.to_nat(pad),U32.to_nat(padded_row)),Nat.is_lt(U32.to_nat(padded_row),
          Nat.add(U32.to_nat(pad),U32.to_nat(extent))),h)))

# The scalar guard and the packed subtraction guard describe the same interval.
# The addition bound is essential: a wrapped upper endpoint is not an interval.
def native_interval(+position: U32,+padding: U32,+extent: U32,+capacity: U32,
  +capacity_bound: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +endpoint_bound: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.is_le(padding,position) && U32.is_lt(position,(padding + extent : U32)) == U32.is_lt((position - padding : U32),extent) : Bool}:
  %Equal.sym(Bool,U32.is_le(padding,position),Nat.is_le(U32.to_nat(padding),U32.to_nat(position)),Comparison.le(padding,position)) :
    {_ && U32.is_lt(position,(padding + extent : U32)) == U32.is_lt((position - padding : U32),extent) : Bool}
  %Equal.sym(Bool,U32.is_lt(position,(padding + extent : U32)),Nat.is_lt(U32.to_nat(position),U32.to_nat((padding + extent : U32))),
    Comparison.lt(position,(padding + extent : U32))) :
    {Nat.is_le(U32.to_nat(padding),U32.to_nat(position)) && _ == U32.is_lt((position - padding : U32),extent) : Bool}
  %Equal.sym(Nat,U32.to_nat((padding + extent : U32)),Nat.add(U32.to_nat(padding),U32.to_nat(extent)),
    Arithmetic.add(padding,extent,capacity,capacity_bound,
      %Order.commutative(U32.to_nat(extent),U32.to_nat(padding)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
      endpoint_bound)) :
    {Nat.is_le(U32.to_nat(padding),U32.to_nat(position)) && Nat.is_lt(U32.to_nat(position),_) == U32.is_lt((position - padding : U32),extent) : Bool}
  Equal.sym(Bool,U32.is_lt((position - padding : U32),extent),
    Nat.is_le(U32.to_nat(padding),U32.to_nat(position)) && Nat.is_lt(U32.to_nat(position),Nat.add(U32.to_nat(padding),U32.to_nat(extent))),
    range(position,padding,extent,capacity,capacity_bound,endpoint_bound))

def subtract_sum(+a: Nat,+b: Nat) -> {Nat.sub(Nat.add(a,b),b) == a : Nat}:
  match b:
    case 0n:
      %Equal.sym(Nat,Nat.add(a,0n),a,Addition.zero_right(a)) : {Nat.sub(_,0n) == a : Nat}
      ArrayAllocation.sub_zero(a)
    case 1n+ +p:
      %Equal.sym(Nat,Nat.add(a,1n+p),1n+Nat.add(a,p),Addition.successor_right(a,p)) : {Nat.sub(_,1n+p) == a : Nat}
      subtract_sum(a,p)

def offset(+padded_row: U32,+pad: U32,+extent: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  bound: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool},h: {U32.is_lt((padded_row - pad : U32),
      extent) == True{} : Bool}) -> {U32.to_nat((padded_row - pad : U32)) == Nat.sub(U32.to_nat(padded_row),U32.to_nat(pad)) : Nat}:
  %recovered(padded_row,pad,extent,cap,cap_ok,bound,h) : {U32.to_nat((padded_row - pad : U32)) == Nat.sub(_,U32.to_nat(pad)) : Nat}
  Equal.sym(Nat,Nat.sub(Nat.add(U32.to_nat((padded_row - pad : U32)),U32.to_nat(pad)),U32.to_nat(pad)),U32.to_nat((padded_row - pad : U32)),
    subtract_sum(U32.to_nat((padded_row - pad : U32)),U32.to_nat(pad)))
