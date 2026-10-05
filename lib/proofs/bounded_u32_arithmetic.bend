# Machine operations -> unbounded Nat, with explicit no-overflow obligations.
import Base
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./word_subtraction.bend as Subtraction
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./u32_comparison.bend as Comparison

def count_within(+count: U32,+length: Nat,+capacity: U32,+upper: U32,exact: {U32.to_nat(count) == length : Nat},
  room: {Nat.is_le(length,U32.to_nat(capacity)) == True{} : Bool},capacity_bound: {U32.is_le(capacity,upper) == True{} : Bool}) ->
  {U32.is_le(count,upper) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(count,upper),Nat.is_le(U32.to_nat(count),U32.to_nat(upper)),Comparison.le(count,upper)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(count),length,exact) : {Nat.is_le(_,U32.to_nat(upper)) == True{} : Bool}
  Order.transitive(length,U32.to_nat(capacity),U32.to_nat(upper),room,Bounds.as_nat_le(capacity,upper,capacity_bound))

def native_roundtrip(+a: U32) -> {U32.from_nat(U32.to_nat(a)) == a : U32}:
  match a:
    case U32{+w}:
      %Equal.sym(U32,U32.from_nat(Word.to_nat(32n,w)),U32{WordArithmetic.encode(Word.to_nat(32n,w),32n)},
        WordArithmetic.native_encoding(Word.to_nat(32n,w))) : {_ == U32{w} : U32}
      %Equal.sym(Word(32n),WordArithmetic.encode(Word.to_nat(32n,w),32n),w,Multiplication.word_roundtrip(32n,w)) : {U32{_} == U32{w} : U32}
      {==}

def native_add(+a: U32,+b: U32) -> {U32.add(a,b) == U32.from_nat(Nat.add(U32.to_nat(a),U32.to_nat(b))) : U32}:
  %native_roundtrip(a) : {U32.add(_,b) == U32.from_nat(Nat.add(U32.to_nat(a),U32.to_nat(b))) : U32}
  %native_roundtrip(b) : {U32.add(U32.from_nat(U32.to_nat(a)),_) == U32.from_nat(Nat.add(U32.to_nat(a),U32.to_nat(b))) : U32}
  WordArithmetic.native_add(U32.to_nat(a),U32.to_nat(b))

def add_zero(+value: U32) -> {(value + 0 : U32) == value : U32}:
  %native_roundtrip(value) : {(value + 0 : U32) == _ : U32}
  %Addition.zero_right(U32.to_nat(value)) : {(value + 0 : U32) == U32.from_nat(_) : U32}
  native_add(value,0)

def add_commutative(+left: U32,+right: U32) -> {(left + right : U32) == (right + left : U32) : U32}:
  %Equal.sym(U32,(left + right : U32),U32.from_nat(Nat.add(U32.to_nat(left),U32.to_nat(right))),native_add(left,right)) :
    {_ == (right + left : U32) : U32}
  %Equal.sym(U32,(right + left : U32),U32.from_nat(Nat.add(U32.to_nat(right),U32.to_nat(left))),native_add(right,left)) :
    {U32.from_nat(Nat.add(U32.to_nat(left),U32.to_nat(right))) == _ : U32}
  %Order.commutative(U32.to_nat(left),U32.to_nat(right)) :
    {U32.from_nat(Nat.add(U32.to_nat(left),U32.to_nat(right))) == U32.from_nat(_) : U32}
  {==}

def add_zero_left(+value: U32) -> {(0 + value : U32) == value : U32}:
  Equal.trans(U32,(0 + value : U32),(value + 0 : U32),value,add_commutative(0,value),add_zero(value))

def multiply_zero_left(+value: U32) -> {(0 * value : U32) == 0 : U32}:
  Multiplication.native_multiply(0,value)

# These are modular identities, so they require no no-overflow assumption.
def encoded_add_associative(+a: Nat,+b: Nat,+c: Nat) ->
  {((U32.from_nat(a) + U32.from_nat(b) : U32) + U32.from_nat(c) : U32) == (U32.from_nat(a) + (U32.from_nat(b) + U32.from_nat(c) : U32) : U32) : U32}:
  %Equal.sym(U32,(U32.from_nat(a) + U32.from_nat(b) : U32),U32.from_nat(Nat.add(a,b)),WordArithmetic.native_add(a,b)) :
    {(_ + U32.from_nat(c) : U32) == (U32.from_nat(a) + (U32.from_nat(b) + U32.from_nat(c) : U32) : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.add(a,b)) + U32.from_nat(c) : U32),U32.from_nat(Nat.add(Nat.add(a,b),c)),WordArithmetic.native_add(Nat.add(a,b),c)) :
    {_ == (U32.from_nat(a) + (U32.from_nat(b) + U32.from_nat(c) : U32) : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(b) + U32.from_nat(c) : U32),U32.from_nat(Nat.add(b,c)),WordArithmetic.native_add(b,c)) : {U32.from_nat(Nat.add(Nat.add(a,b),c)) == (U32.from_nat(a) + _ : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(a) + U32.from_nat(Nat.add(b,c)) : U32),U32.from_nat(Nat.add(a,Nat.add(b,c))),WordArithmetic.native_add(a,Nat.add(b,c))) : {U32.from_nat(Nat.add(Nat.add(a,b),c)) == _ : U32}
  %Addition.associative(a,b,c) : {U32.from_nat(Nat.add(Nat.add(a,b),c)) == U32.from_nat(_) : U32}
  {==}

def add_associative(+a: U32,+b: U32,+c: U32) -> {((a + b : U32) + c : U32) == (a + (b + c : U32) : U32) : U32}:
  %native_roundtrip(a) : {((_ + b : U32) + c : U32) == (_ + (b + c : U32) : U32) : U32}
  %native_roundtrip(b) : {((U32.from_nat(U32.to_nat(a)) + _ : U32) + c : U32) == (U32.from_nat(U32.to_nat(a)) + (_ + c : U32) : U32) : U32}
  %native_roundtrip(c) : {((U32.from_nat(U32.to_nat(a)) + U32.from_nat(U32.to_nat(b)) : U32) + _ : U32) == (U32.from_nat(U32.to_nat(a)) + (U32.from_nat(U32.to_nat(b)) + _ : U32) : U32) : U32}
  encoded_add_associative(U32.to_nat(a),U32.to_nat(b),U32.to_nat(c))

def subtract_from_sum(+a: U32,+b: U32,+c: U32) -> {((a + b : U32) - c : U32) == (a + (b - c : U32) : U32) : U32}:
  %Subtraction.add_subtract(b,c) : {((a + _ : U32) - c : U32) == (a + (b - c : U32) : U32) : U32}
  %add_associative(a,(b - c : U32),c) : {(_ - c : U32) == (a + (b - c : U32) : U32) : U32}
  Subtraction.subtract_add((a + (b - c : U32) : U32),c)

def add_after_subtract(+value: U32,+padding: U32,+offset: U32) ->
  {((value - padding : U32) + offset : U32) == ((value + offset : U32) - padding : U32) : U32}:
  %Equal.sym(U32,((value - padding : U32) + offset : U32),(offset + (value - padding : U32) : U32),add_commutative((value - padding : U32),offset)) :
    {_ == ((value + offset : U32) - padding : U32) : U32}
  %add_commutative(offset,value) : {(offset + (value - padding : U32) : U32) == (_ - padding : U32) : U32}
  Equal.sym(U32,((offset + value : U32) - padding : U32),(offset + (value - padding : U32) : U32),subtract_from_sum(offset,value,padding))

def native_subtract(+a: U32,+b: U32,h: {Nat.is_le(U32.to_nat(b),U32.to_nat(a)) == True{} : Bool}) -> {U32.sub(a,
    b) == U32.from_nat(Nat.sub(U32.to_nat(a),U32.to_nat(b))) : U32}:
  %native_roundtrip(a) : {U32.sub(_,b) == U32.from_nat(Nat.sub(U32.to_nat(a),U32.to_nat(b))) : U32}
  %native_roundtrip(b) : {U32.sub(U32.from_nat(U32.to_nat(a)),_) == U32.from_nat(Nat.sub(U32.to_nat(a),U32.to_nat(b))) : U32}
  Subtraction.encoded_subtract(U32.to_nat(a),U32.to_nat(b),h)

def add(+a: U32,+b: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},h: {Nat.is_le(Nat.add(U32.to_nat(a),U32.to_nat(b)),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(U32.add(a,b)) == Nat.add(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  %Equal.sym(U32,U32.add(a,b),U32.from_nat(Nat.add(U32.to_nat(a),U32.to_nat(b))),native_add(a,b)) : {U32.to_nat(_) == Nat.add(U32.to_nat(a),
      U32.to_nat(b)) : Nat}
  Bounds.roundtrip(Nat.add(U32.to_nat(a),U32.to_nat(b)),cap,cap_ok,h)

def subtract(+a: U32,+b: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},ordered: {Nat.is_le(U32.to_nat(b),
      U32.to_nat(a)) == True{} : Bool},h: {Nat.is_le(Nat.sub(U32.to_nat(a),U32.to_nat(b)),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(U32.sub(a,b)) == Nat.sub(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  %Equal.sym(U32,U32.sub(a,b),U32.from_nat(Nat.sub(U32.to_nat(a),U32.to_nat(b))),native_subtract(a,b,
      ordered)) : {U32.to_nat(_) == Nat.sub(U32.to_nat(a),U32.to_nat(b)) : Nat}
  Bounds.roundtrip(Nat.sub(U32.to_nat(a),U32.to_nat(b)),cap,cap_ok,h)

def remove_prefix(+value: U32,+base: U32,+local: Nat,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(value) == Nat.add(U32.to_nat(base),local) : Nat},room: {Nat.is_le(U32.to_nat(value),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((value - base : U32)) == local : Nat}:
  +local_room = Order.transitive(local,Nat.add(U32.to_nat(base),local),U32.to_nat(capacity),
    %Order.commutative(local,U32.to_nat(base)) : {Nat.is_le(local,_) == True{} : Bool} Order.le_add(local,U32.to_nat(base)),
    %exact : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} room)
  Equal.trans(Nat,U32.to_nat((value - base : U32)),Nat.sub(U32.to_nat(value),U32.to_nat(base)),local,
    subtract(value,base,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(value),Nat.add(U32.to_nat(base),local),exact) : {Nat.is_le(U32.to_nat(base),_) == True{} : Bool} Order.le_add(U32.to_nat(base),local),
      %Equal.sym(Nat,U32.to_nat(value),Nat.add(U32.to_nat(base),local),exact) : {Nat.is_le(Nat.sub(_,U32.to_nat(base)),U32.to_nat(capacity)) == True{} : Bool}
      %Equal.sym(Nat,Nat.sub(Nat.add(U32.to_nat(base),local),U32.to_nat(base)),local,Order.subtract_shift(U32.to_nat(base),local)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} local_room),
    %Equal.sym(Nat,U32.to_nat(value),Nat.add(U32.to_nat(base),local),exact) : {Nat.sub(_,U32.to_nat(base)) == local : Nat} Order.subtract_shift(U32.to_nat(base),local))

def subtract_zero(+value: U32) -> {(value - 0 : U32) == value : U32}:
  Equal.trans(U32,(value - 0 : U32),U32.from_nat(Nat.sub(U32.to_nat(value),0n)),value,native_subtract(value,0,Order.zero_le(U32.to_nat(value))),
    Equal.trans(U32,U32.from_nat(Nat.sub(U32.to_nat(value),0n)),U32.from_nat(U32.to_nat(value)),value,
      Equal.cong(Nat,U32,U32.from_nat,Nat.sub(U32.to_nat(value),0n),U32.to_nat(value),Order.subtract_zero(U32.to_nat(value))),native_roundtrip(value)))

def multiply(+a: U32,+b: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},h: {Nat.is_le(Nat.mul(U32.to_nat(a),U32.to_nat(b)),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(U32.mul(a,b)) == Nat.mul(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  Multiplication.bounded_multiply(a,b,cap,cap_ok,h)
