import ../../lib/proofs/storage_shape.bend as Shape
# Inductive whole-write-program invariant. A covered index receives its
# specified F32 value; unrelated writes preserve it. No FP32 algebra is used.
import Base
import ./scalar_write_refinement.bend as ScalarWriteRefinement
import ../../lib/proofs/nat_division.bend as Division
import ../specification.bend as ConvSpec
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/proofs/boolean_logic.bend as Boolean

type Write is Data:
  Write{index: Nat,value: F32}

def apply(writes: List<&2,Write>,out: List<&2,F32>) -> List<&2,F32>:
  match writes:
    case Nil{}:
      out
    case Write{i,v} <> rest:
      apply(rest,ScalarWriteRefinement.set(i,v,out))

def contains(+query: Nat,writes: List<&2,Write>) -> Bool:
  match writes:
    case Nil{}:
      False{}
    case Write{i,v} <> rest:
      Nat.is_eq(query,i) || contains(query,rest)

def Matches(ok: Bool,+actual: F32,+wanted: F32) -> Data:
  Boolean.When(ok,{actual == wanted : F32})

def MatchingWritesHaveValue(+query: Nat,+value: F32,writes: List<&2,Write>) -> Data:
  match writes:
    case Nil{}:
      Unit
    case Write{i,v} <> rest:
      Shape.Both<Matches(Nat.is_eq(query,i),v,value),MatchingWritesHaveValue(query,value,rest)>

def set_length(+i: Nat,+v: F32,+xs: List<&2,F32>) -> {ConvSpec.length(ScalarWriteRefinement.set(i,v,xs)) == ConvSpec.length(xs) : Nat}:
  match i xs:
    case 0n Nil{}:
      {==}
    case 0n x <> tail:
      {==}
    case 1n+i Nil{}:
      {==}
    case 1n+ +i +x <> +tail:
      %set_length(i,v,tail) : {1n+ConvSpec.length(ScalarWriteRefinement.set(i,v,tail)) == 1n+_ : Nat}
      {==}

def set_room(+query: Nat,+i: Nat,+v: F32,+xs: List<&2,F32>,room: {Nat.is_lt(query,ConvSpec.length(xs)) == True{} : Bool}) -> {Nat.is_lt(query,
    ConvSpec.length(ScalarWriteRefinement.set(i,v,xs))) == True{} : Bool}:
  %Equal.sym(Nat,ConvSpec.length(ScalarWriteRefinement.set(i,v,xs)),ConvSpec.length(xs),set_length(i,v,xs)) : {Nat.is_lt(query,_) == True{} : Bool}
  room

def at_set(+query: Nat,+i: Nat,+v: F32,+xs: List<&2,F32>,room: {Nat.is_lt(query,ConvSpec.length(xs)) == True{} : Bool}) -> {ConvSpec.at(query,
    ScalarWriteRefinement.set(i,v,xs)) == Bool.pick(F32,Nat.is_eq(query,i),v,ConvSpec.at(query,xs)) : F32}:
  match query i xs:
    case query i Nil{}:
      Empty.absurd({ConvSpec.at(query,ScalarWriteRefinement.set(i,v,[])) == Bool.pick(F32,Nat.is_eq(query,i),v,ConvSpec.at(query,[])) : F32},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(query,0n),True{},Equal.sym(Bool,Nat.is_lt(query,0n),False{},
              Division.lt_zero(query)),room)))
    case 0n 0n x <> tail:
      {==}
    case 0n 1n+i x <> tail:
      {==}
    case 1n+query 0n x <> tail:
      {==}
    case 1n+query 1n+i x <> tail:
      at_set(query,i,v,tail,room)

law result_at:
  for +writes: List<&2,Write>
  for +query: Nat
  for +value: F32
  for +out: List<&2,F32>
  for correct: MatchingWritesHaveValue(query,value,writes)
  for +room: {Nat.is_lt(query,ConvSpec.length(out)) == True{} : Bool}
  {ConvSpec.at(query,apply(writes,out)) == Bool.pick(F32,contains(query,writes),value,ConvSpec.at(query,out)) : F32}
def result_at(writes,query,value,out,correct,room):
  match writes:
    case Nil{}:
      {==}
    case Write{+i,+v} <> +rest:
      Shape.Both{first,tail} = correct
      %Equal.sym(F32,ConvSpec.at(query,apply(rest,ScalarWriteRefinement.set(i,v,out))),Bool.pick(F32,contains(query,rest),value,
          ConvSpec.at(query,ScalarWriteRefinement.set(i,v,out))),result_at(rest,query,value,ScalarWriteRefinement.set(i,v,out),tail,
          set_room(query,i,v,out,room))) : {_ == Bool.pick(F32,Nat.is_eq(query,i) || contains(query,rest),value,ConvSpec.at(query,out)) : F32}
      %Equal.sym(F32,ConvSpec.at(query,ScalarWriteRefinement.set(i,v,out)),Bool.pick(F32,Nat.is_eq(query,i),v,ConvSpec.at(query,out)),
        at_set(query,i,v,out,room)) : {Bool.pick(F32,contains(query,rest),value,_) == Bool.pick(F32,Nat.is_eq(query,i) || contains(query,rest),
          value,ConvSpec.at(query,out)) : F32}
      Boolean.choose_matching(F32,Nat.is_eq(query,i),contains(query,rest),v,value,ConvSpec.at(query,out),first)

def covered(+writes: List<&2,Write>,+query: Nat,+value: F32,+out: List<&2,F32>,correct: MatchingWritesHaveValue(query,value,writes),
  room: {Nat.is_lt(query,ConvSpec.length(out)) == True{} : Bool},coverage: {contains(query,writes) == True{} : Bool}) -> {ConvSpec.at(query,
    apply(writes,out)) == value : F32}:
  %coverage : {ConvSpec.at(query,apply(writes,out)) == Bool.pick(F32,_,value,ConvSpec.at(query,out)) : F32}
  result_at(writes,query,value,out,correct,room)
