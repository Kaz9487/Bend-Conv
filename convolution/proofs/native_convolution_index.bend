import Base
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/word_multiplication.bend as Multiplication
import ../../lib/proofs/row_major_index.bend as RowMajor

def encoded(+x: U32) -> {x == U32.from_nat(U32.to_nat(x)) : U32}:
  Equal.sym(U32,U32.from_nat(U32.to_nat(x)),x,Arithmetic.native_roundtrip(x))

def add(+a: U32,+b: U32,+an: Nat,+bn: Nat,ea: {a == U32.from_nat(an) : U32},
  eb: {b == U32.from_nat(bn) : U32}) -> {(a + b : U32) == U32.from_nat(Nat.add(an,bn)) : U32}:
  %Equal.sym(U32,a,U32.from_nat(an),ea) : {(_ + b : U32) == U32.from_nat(Nat.add(an,bn)) : U32}
  %Equal.sym(U32,b,U32.from_nat(bn),eb) : {(U32.from_nat(an) + _ : U32) == U32.from_nat(Nat.add(an,bn)) : U32}
  WordArithmetic.native_add(an,bn)

def mul(+a: U32,+b: U32,+an: Nat,+bn: Nat,ea: {U32.to_nat(a) == an : Nat},
  eb: {U32.to_nat(b) == bn : Nat}) -> {(a * b : U32) == U32.from_nat(Nat.mul(an,bn)) : U32}:
  %ea : {(a * b : U32) == U32.from_nat(Nat.mul(_,bn)) : U32}
  %eb : {(a * b : U32) == U32.from_nat(Nat.mul(U32.to_nat(a),_)) : U32}
  Multiplication.native_multiply(a,b)

law nchw:
  for +input_channel_index: U32
  for +height: U32
  for +width: U32
  for +iy: U32
  for +ix: U32
  for +cap: U32
  for +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool}
  for row_bound: {Nat.is_le(Nat.mul(U32.to_nat(input_channel_index),U32.to_nat(height)),U32.to_nat(cap)) == True{} : Bool}
  {(input_channel_index * height * width + iy * width + ix : U32) == U32.from_nat(RowMajor.index(U32.to_nat(input_channel_index),U32.to_nat(height),U32.to_nat(width),
        U32.to_nat(iy),U32.to_nat(ix))) : U32}
def nchw(input_channel_index,height,width,iy,ix,cap,cap_ok,row_bound):
  %RowMajor.expanded(U32.to_nat(input_channel_index),U32.to_nat(height),U32.to_nat(width),U32.to_nat(iy),
    U32.to_nat(ix)) : {(input_channel_index * height * width + iy * width + ix : U32) == U32.from_nat(_) : U32}
  add((input_channel_index * height * width + iy * width : U32),ix,Nat.add(Nat.mul(Nat.mul(U32.to_nat(input_channel_index),U32.to_nat(height)),U32.to_nat(width)),
      Nat.mul(U32.to_nat(iy),U32.to_nat(width))),U32.to_nat(ix),
    add((input_channel_index * height * width : U32),(iy * width : U32),Nat.mul(Nat.mul(U32.to_nat(input_channel_index),U32.to_nat(height)),U32.to_nat(width)),
      Nat.mul(U32.to_nat(iy),U32.to_nat(width)),
      mul((input_channel_index * height : U32),width,Nat.mul(U32.to_nat(input_channel_index),U32.to_nat(height)),U32.to_nat(width),Arithmetic.multiply(input_channel_index,height,cap,
          cap_ok,row_bound),{==}),
      mul(iy,width,U32.to_nat(iy),U32.to_nat(width),{==},{==})),
    encoded(ix))
