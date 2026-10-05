# Successful contiguous construction/reshape has exactly the requested shape
# and offset. Failure observations use a fallback and assert no success.
import Base
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views

def append_empty(+values: List<&2,U32>) -> {List.append(&2,U32,values,Nil{}) == values : List<&2,U32>}:
  match values:
    case Nil{}: {==}
    case Con{+head,+tail}:
      %append_empty(tail) : {head <> List.append(&2,U32,tail,Nil{}) == head <> _ : List<&2,U32>}
      {==}

def reverse_twice_onto(+values: List<&2,U32>,+middle: List<&2,U32>,+tail: List<&2,U32>) ->
  {List.reverse.go(&2,U32,List.reverse.go(&2,U32,values,middle),tail) ==
    List.reverse.go(&2,U32,middle,List.append(&2,U32,values,tail)) : List<&2,U32>}:
  match values:
    case Nil{}: {==}
    case Con{head,rest}: reverse_twice_onto(rest,head <> middle,tail)

def reverse_twice(+values: List<&2,U32>) -> {List.reverse(&2,U32,List.reverse(&2,U32,values)) == values : List<&2,U32>}:
  %append_empty(values) : {List.reverse(&2,U32,List.reverse(&2,U32,values)) == _ : List<&2,U32>}
  reverse_twice_onto(values,Nil{},Nil{})

def observed_extents(result: Maybe<&2,Views.View>,fallback: List<&2,U32>) -> List<&2,U32>:
  match result:
    case None{}: fallback
    case Some{Views.View{axes,offset}}: Views.axis_extents(axes)

def observed_offset(result: Maybe<&2,Views.View>,fallback: U32) -> U32:
  match result:
    case None{}: fallback
    case Some{Views.View{axes,offset}}: offset

def built_shape(+reversed: List<&2,U32>,+stride: Maybe<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32) ->
  {observed_extents(Views.build_axes(reversed,stride,axes,offset),List.reverse.go(&2,U32,reversed,Views.axis_extents(axes))) ==
    List.reverse.go(&2,U32,reversed,Views.axis_extents(axes)) : List<&2,U32>}:
  match reversed stride:
    case Nil{} Some{value}: {==}
    case Con{size,tail} Some{value}: built_shape(tail,Shapes.checked_multiply(value,size),Views.Axis{size,value} <> axes,offset)
    case Nil{} None{}: {==}
    case Con{head,tail} None{}: {==}

def built_offset(+reversed: List<&2,U32>,+stride: Maybe<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32) ->
  {observed_offset(Views.build_axes(reversed,stride,axes,offset),offset) == offset : U32}:
  match reversed stride:
    case Nil{} Some{value}: {==}
    case Con{size,tail} Some{value}: built_offset(tail,Shapes.checked_multiply(value,size),Views.Axis{size,value} <> axes,offset)
    case Nil{} None{}: {==}
    case Con{head,tail} None{}: {==}

def contiguous_count_shape(+count: Maybe<&2,U32>,+extents: List<&2,U32>,+offset: U32) ->
  {observed_extents(Views.contiguous_count(count,extents,offset),extents) == extents : List<&2,U32>}:
  match count:
    case None{}: {==}
    case Some{+value}:
      %reverse_twice(extents) :
        {observed_extents(Views.build_axes(List.reverse(&2,U32,extents),Some{Bool.pick(U32,U32.is_zero(value),0,1)},Nil{},offset),_) == _ : List<&2,U32>}
      built_shape(List.reverse(&2,U32,extents),Some{Bool.pick(U32,U32.is_zero(value),0,1)},Nil{},offset)

def contiguous_shape(+extents: List<&2,U32>,+offset: U32) ->
  {observed_extents(Views.contiguous(Shapes.Shape{extents},offset),extents) == extents : List<&2,U32>}:
  contiguous_count_shape(Shapes.numel(Shapes.Shape{extents}),extents,offset)

def contiguous_count_offset(+count: Maybe<&2,U32>,+extents: List<&2,U32>,+offset: U32) ->
  {observed_offset(Views.contiguous_count(count,extents,offset),offset) == offset : U32}:
  match count:
    case None{}: {==}
    case Some{+value}: built_offset(List.reverse(&2,U32,extents),Some{Bool.pick(U32,U32.is_zero(value),0,1)},Nil{},offset)
