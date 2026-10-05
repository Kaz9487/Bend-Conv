# Recover the exact count and allocation guard from the public count_of branch.
# Zero extents yield zero even if another partial product would overflow.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_shape_refinement.bend as ShapeLaws
import ./checked_word_refinement.bend as Checked
import ./word_encoding.bend as WordEncoding

type Count<-expected: Nat,-count: U32> is Data:
  Count{exact: {expected == U32.to_nat(count) : Nat},bounded: {U32.is_le(count,Machine.capacity(Default.limits())) == True{} : Bool}}

def selected(fits: Bool,+expected: Nat,+value: U32,+count: U32,exact: {expected == U32.to_nat(value) : Nat},
  bound: {U32.is_le(value,Machine.capacity(Default.limits())) == fits : Bool},accepted: {Tensors.within(fits,value) == Some{count} : Maybe<&2,U32>}) -> Count<expected,count>:
  match fits:
    case True{}:
      %Equal.cong(Maybe<&2,U32>,U32,result => Checked.value(result),Some{value},Some{count},accepted) : Count<expected,_>
      Count{exact,bound}
    case False{}:
      Empty.absurd(Count<expected,count>,WordEncoding.false_true(Equal.cong(Maybe<&2,U32>,Bool,result => Checked.present(result),None{},Some{count},accepted)))

def allocatable(+result: Maybe<&2,U32>,+expected: Nat,+count: U32,exact: ShapeLaws.Count(result,expected),
  accepted: {Tensors.allocatable(result) == Some{count} : Maybe<&2,U32>}) -> Count<expected,count>:
  match result:
    case None{}:
      Empty.absurd(Count<expected,count>,WordEncoding.false_true(Equal.cong(Maybe<&2,U32>,Bool,result => Checked.present(result),None{},Some{count},accepted)))
    case Some{+value}: selected(U32.is_le(value,Machine.capacity(Default.limits())),expected,value,count,exact,{==},accepted)

def count_of(+extents: List<&2,U32>,+count: U32,
  accepted: {Tensors.count_of(extents) == Some{count} : Maybe<&2,U32>}) -> Count<ShapeLaws.elements(extents),count>:
  allocatable(Shapes.numel(Shapes.Shape{extents}),ShapeLaws.elements(extents),count,ShapeLaws.numel(extents),accepted)
