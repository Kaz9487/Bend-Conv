# Exact checked scalar products from caller-supplied native range evidence.
# All arithmetic operands and limits remain parameters during elaboration.
import Base
import ../math_checked_word.bend as CheckedWord
import ../machine_limits.bend as Machine
import ./checked_word_refinement.bend as Checked
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic

def bounded(+left: U32,+right: U32,+limit: U32,+output: U32,
  product_exact: {Nat.mul(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(output) : Nat},
  within: {U32.is_le(output,limit) == True{} : Bool}) ->
  {CheckedWord.multiply_bounded(left,right,limit) == Some{output} : Maybe<&2,U32>}:
  %Equal.sym(Maybe<&2,U32>,CheckedWord.multiply_bounded(left,right,limit),
    CheckedWord.encoded(CheckedWord.bounded(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit))),Checked.multiply_bounded(left,right,limit)) : {_ == Some{output} : Maybe<&2,U32>}
  %Equal.sym(Nat,Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(output),product_exact) :
    {CheckedWord.encoded(CheckedWord.bounded(_,U32.to_nat(limit))) == Some{output} : Maybe<&2,U32>}
  %Equal.sym(Bool,Nat.is_le(U32.to_nat(output),U32.to_nat(limit)),True{},Bounds.as_nat_le(output,limit,within)) :
    {CheckedWord.encoded(CheckedWord.bounded_result(_,U32.to_nat(output))) == Some{output} : Maybe<&2,U32>}
  Equal.cong(U32,Maybe<&2,U32>,word => Some{word},U32.from_nat(U32.to_nat(output)),output,Arithmetic.native_roundtrip(output))

law actual:
  for +left: U32
  for +right: U32
  for +limit: U32
  for +output: U32
  for product_exact: {Nat.mul(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(output) : Nat}
  for within: {U32.is_le(output,limit) == True{} : Bool}
  for maximum: {limit == Machine.word_max() : U32}
  {CheckedWord.multiply(left,right) == Some{output} : Maybe<&2,U32>}

def actual(left,right,limit,output,product_exact,within,maximum):
  %Equal.sym(Maybe<&2,U32>,CheckedWord.multiply(left,right),
    CheckedWord.encoded(CheckedWord.bounded(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit))),Checked.multiply(left,right,limit,maximum)) : {_ == Some{output} : Maybe<&2,U32>}
  %Equal.sym(Nat,Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(output),product_exact) :
    {CheckedWord.encoded(CheckedWord.bounded(_,U32.to_nat(limit))) == Some{output} : Maybe<&2,U32>}
  %Equal.sym(Bool,Nat.is_le(U32.to_nat(output),U32.to_nat(limit)),True{},Bounds.as_nat_le(output,limit,within)) :
    {CheckedWord.encoded(CheckedWord.bounded_result(_,U32.to_nat(output))) == Some{output} : Maybe<&2,U32>}
  Equal.cong(U32,Maybe<&2,U32>,word => Some{word},U32.from_nat(U32.to_nat(output)),output,Arithmetic.native_roundtrip(output))
