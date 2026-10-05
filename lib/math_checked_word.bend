# Exact overflow-checked multiplication for metadata. Each Horner step starts
# below word_max and forms at most 3*word_max, safely below the native Nat cap.
# The loop has exactly 32 bit steps and never touches tensor payloads.
import Base
import ./machine_limits.bend as Machine

def bounded_result(valid: Bool,value: Nat) -> Maybe<&2,Nat>:
  match valid:
    case False{}: None{}
    case True{}: Some{value}

def bounded(+value: Nat,limit: Nat) -> Maybe<&2,Nat>:
  bounded_result(Nat.is_le(value,limit),value)

def multiply_step(bit: Bool,right: Nat,previous: Maybe<&2,Nat>,limit: Nat) -> Maybe<&2,Nat>:
  match previous:
    case None{}: None{}
    case Some{value}: bounded(Nat.add(Nat.double(value),Bool.pick(Nat,bit,right,0n)),limit)

def multiply_bits(width: Nat,bits: Word(width),+right: Nat,+limit: Nat) -> Maybe<&2,Nat>:
  match width bits:
    case 0n WNil{}: Some{0n}
    case 1n+rest WCon{bit,tail}: multiply_step(bit,right,multiply_bits(rest,tail,right,limit),limit)

def encoded(result: Maybe<&2,Nat>) -> Maybe<&2,U32>:
  match result:
    case None{}: None{}
    case Some{value}: Some{U32.from_nat(value)}

def multiply_bounded(left: U32,right: U32,limit: U32) -> Maybe<&2,U32>:
  U32{bits} = left
  encoded(multiply_bits(32n,bits,U32.to_nat(right),U32.to_nat(limit)))

# Unit products already fit a word. Shapes and contiguous strides frequently
# use them; returning the operand avoids the 32-step natural Horner traversal.
def multiply_right(unit: Bool,left: U32,right: U32) -> Maybe<&2,U32>:
  match unit:
    case True{}: Some{left}
    case False{}: multiply_bounded(left,right,Machine.word_max())

def multiply_left(unit: Bool,+left: U32,+right: U32) -> Maybe<&2,U32>:
  match unit:
    case True{}: Some{right}
    case False{}: multiply_right(U32.is_eq(right,1),left,right)

def multiply(+left: U32,+right: U32) -> Maybe<&2,U32>:
  multiply_left(U32.is_eq(left,1),left,right)
