# Equality of two sequences is preserved by concatenation. No FP32 algebra.
import Base
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ../specification.bend as ConvSpec
import ../ordered_reference.bend as OrderedReference
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/proofs/nat_division.bend as Division

def cat(-A: Data,+xs: List<&2,A>,+ys: List<&2,A>,+us: List<&2,A>,+vs: List<&2,A>,left: {xs == us : List<&2,A>},right: {ys == vs : List<&2,
    A>}) -> {OrderedMatrix.cat(A,xs,ys) == OrderedMatrix.cat(A,us,vs) : List<&2,A>}:
  %left : {OrderedMatrix.cat(A,xs,ys) == OrderedMatrix.cat(A,_,vs) : List<&2,A>}
  %right : {OrderedMatrix.cat(A,xs,ys) == OrderedMatrix.cat(A,xs,_) : List<&2,A>}
  {==}

def at_concatenated_left(+left: List<&2,F32>,+right: List<&2,F32>,+index: Nat,inside: {Nat.is_lt(index,ConvSpec.length(left)) == True{} : Bool}) ->
  {ConvSpec.at(index,OrderedMatrix.cat(F32,left,right)) == ConvSpec.at(index,left) : F32}:
  match left:
    case Nil{}:
      Empty.absurd({ConvSpec.at(index,right) == ConvSpec.at(index,[]) : F32},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),inside)))
    case head <> +tail:
      match index:
        case 0n: {==}
        case 1n+ +previous: at_concatenated_left(tail,right,previous,inside)

def at_concatenated_right(+left: List<&2,F32>,+right: List<&2,F32>,+index: Nat) ->
  {ConvSpec.at(Nat.add(ConvSpec.length(left),index),OrderedMatrix.cat(F32,left,right)) == ConvSpec.at(index,right) : F32}:
  match left:
    case Nil{}: {==}
    case head <> +tail: at_concatenated_right(tail,right,index)

def stream_cat(+left: List<&2,F32>,+right: List<&2,F32>) -> {OrderedMatrix.cat(F32,left,right) == OrderedReference.cat(left,right) : List<&2,F32>}:
  match left:
    case Nil{}: {==}
    case +head <> +tail:
      %stream_cat(tail,right) : {head <> OrderedMatrix.cat(F32,tail,right) == head <> _ : List<&2,F32>}
      {==}
