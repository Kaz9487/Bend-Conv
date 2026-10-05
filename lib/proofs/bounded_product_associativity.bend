# Native products associate when both intermediate products fit the supplied
# machine bound. The final encoding preserves the native wrap semantics.
import Base
import ./word_multiplication.bend as Multiplication
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./row_major_index.bend as RowMajor

def left_encoded(+left: U32,+middle: U32,+right: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(left),U32.to_nat(middle)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {((left * middle : U32) * right : U32) == U32.from_nat(Nat.mul(Nat.mul(U32.to_nat(left),U32.to_nat(middle)),U32.to_nat(right))) : U32}:
  Equal.trans(U32,((left * middle : U32) * right : U32),U32.from_nat(Nat.mul(U32.to_nat((left * middle : U32)),U32.to_nat(right))),
    U32.from_nat(Nat.mul(Nat.mul(U32.to_nat(left),U32.to_nat(middle)),U32.to_nat(right))),Multiplication.native_multiply((left * middle : U32),right),
    Equal.cong(Nat,U32,value => U32.from_nat(Nat.mul(value,U32.to_nat(right))),U32.to_nat((left * middle : U32)),Nat.mul(U32.to_nat(left),U32.to_nat(middle)),Arithmetic.multiply(left,middle,capacity,capacity_ok,room)))

def right_encoded(+left: U32,+middle: U32,+right: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(middle),U32.to_nat(right)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {(left * (middle * right : U32) : U32) == U32.from_nat(Nat.mul(U32.to_nat(left),Nat.mul(U32.to_nat(middle),U32.to_nat(right)))) : U32}:
  Equal.trans(U32,(left * (middle * right : U32) : U32),U32.from_nat(Nat.mul(U32.to_nat(left),U32.to_nat((middle * right : U32)))),
    U32.from_nat(Nat.mul(U32.to_nat(left),Nat.mul(U32.to_nat(middle),U32.to_nat(right)))),Multiplication.native_multiply(left,(middle * right : U32)),
    Equal.cong(Nat,U32,value => U32.from_nat(Nat.mul(U32.to_nat(left),value)),U32.to_nat((middle * right : U32)),Nat.mul(U32.to_nat(middle),U32.to_nat(right)),Arithmetic.multiply(middle,right,capacity,capacity_ok,room)))

def native(+left: U32,+middle: U32,+right: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +left_room: {Nat.is_le(Nat.mul(U32.to_nat(left),U32.to_nat(middle)),U32.to_nat(capacity)) == True{} : Bool},
  +right_room: {Nat.is_le(Nat.mul(U32.to_nat(middle),U32.to_nat(right)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {((left * middle : U32) * right : U32) == (left * (middle * right : U32) : U32) : U32}:
  Equal.trans(U32,((left * middle : U32) * right : U32),U32.from_nat(Nat.mul(Nat.mul(U32.to_nat(left),U32.to_nat(middle)),U32.to_nat(right))),
    (left * (middle * right : U32) : U32),left_encoded(left,middle,right,capacity,capacity_ok,left_room),
    Equal.trans(U32,U32.from_nat(Nat.mul(Nat.mul(U32.to_nat(left),U32.to_nat(middle)),U32.to_nat(right))),U32.from_nat(Nat.mul(U32.to_nat(left),Nat.mul(U32.to_nat(middle),U32.to_nat(right)))),
      (left * (middle * right : U32) : U32),Equal.cong(Nat,U32,U32.from_nat,Nat.mul(Nat.mul(U32.to_nat(left),U32.to_nat(middle)),U32.to_nat(right)),Nat.mul(U32.to_nat(left),Nat.mul(U32.to_nat(middle),U32.to_nat(right))),RowMajor.assoc(U32.to_nat(left),U32.to_nat(middle),U32.to_nat(right))),
      Equal.sym(U32,(left * (middle * right : U32) : U32),U32.from_nat(Nat.mul(U32.to_nat(left),Nat.mul(U32.to_nat(middle),U32.to_nat(right)))),right_encoded(left,middle,right,capacity,capacity_ok,right_room))))
