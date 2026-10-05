# Laws of NumPy broadcasting for extents of the same rank. A stretched view
# has exactly the target extents, and extents that Shapes.broadcast_same_rank
# accepts can be stretched from both operands. Rank alignment (with_rank) only
# adds leading axes of extent 1 and is not restated here.
import Base
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ./storage_shape.bend as Shape
import ./storage_address_bounds.bend as Address
import ./encoded_coordinates.bend as Encoded

# The extents of a stretched view.

def Extents(axes: Maybe<&2,List<&2,Views.Axis>>,target: List<&2,U32>) -> Type:
  match axes:
    case Some{axes}: {Views.axis_extents(axes) == target : List<&2,U32>}
    case None{}: Unit

def prepended_extents(+same: Bool,+single: Bool,+size: U32,+stride: U32,+extent: U32,+tail: Maybe<&2,List<&2,Views.Axis>>,+rest: List<&2,U32>,
  same_value: {U32.is_eq(size,extent) == same : Bool},tail_extents: Extents(tail,rest)) ->
  Extents(Views.stretched_prepend(Views.stretched_axis(same,single,size,stride,extent),tail),extent <> rest):
  match same single tail:
    case True{} single Some{+axes}:
      %Equal.sym(U32,size,extent,Address.equal_words(size,extent,same_value)) : {_ <> Views.axis_extents(axes) == extent <> rest : List<&2,U32>}
      %Equal.sym(List<&2,U32>,Views.axis_extents(axes),rest,tail_extents) : {extent <> _ == extent <> rest : List<&2,U32>}
      {==}
    case True{} single None{}: Unit{}
    case False{} True{} Some{+axes}:
      %Equal.sym(List<&2,U32>,Views.axis_extents(axes),rest,tail_extents) : {extent <> _ == extent <> rest : List<&2,U32>}
      {==}
    case False{} True{} None{}: Unit{}
    case False{} False{} tail: Unit{}

def stretch_extents(+axes: List<&2,Views.Axis>,+target: List<&2,U32>) -> Extents(Views.stretch_axes(axes,target),target):
  match axes target:
    case Nil{} Nil{}: {==}
    case Nil{} Con{extent,rest}: Unit{}
    case Con{axis,tail} Nil{}: Unit{}
    case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest}:
      prepended_extents(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent,Views.stretch_axes(tail,rest),rest,{==},stretch_extents(tail,rest))

# The premise form: a stretch that succeeded.
def stretched_extents(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+result: List<&2,Views.Axis>,
  stretched: {Views.stretch_axes(axes,target) == Some{result} : Maybe<&2,List<&2,Views.Axis>>}) ->
  {Views.axis_extents(result) == target : List<&2,U32>}:
  %stretched : Extents(_,target)
  stretch_extents(axes,target)

# Compatible extents can be stretched from both operands.

def Stretches(extent: Maybe<&2,U32>,size: U32,stride: U32) -> Type:
  match extent:
    case Some{extent}: {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent)) == True{} : Bool}
    case None{}: Unit

def left_stretches(+same: Bool,+left_one: Bool,+right_one: Bool,+left: U32,+stride: U32,+right: U32,
  same_value: {U32.is_eq(left,right) == same : Bool},left_value: {U32.is_eq(left,1) == left_one : Bool}) ->
  Stretches(Shapes.broadcast_extent(same,left_one,right_one,left,right),left,stride):
  match same left_one right_one:
    case True{} left_one right_one:
      %Equal.sym(Bool,U32.is_eq(left,left),True{},Encoded.equal_words(left,left,{==})) :
        {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(_,U32.is_eq(left,1),left,stride,left)) == True{} : Bool}
      {==}
    case False{} True{} right_one:
      %Equal.sym(Bool,U32.is_eq(left,right),False{},same_value) :
        {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(_,U32.is_eq(left,1),left,stride,right)) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(left,1),True{},left_value) :
        {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(False{},_,left,stride,right)) == True{} : Bool}
      {==}
    case False{} False{} True{}:
      %Equal.sym(Bool,U32.is_eq(left,left),True{},Encoded.equal_words(left,left,{==})) :
        {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(_,U32.is_eq(left,1),left,stride,left)) == True{} : Bool}
      {==}
    case False{} False{} False{}: Unit{}

def right_stretches(+same: Bool,+left_one: Bool,+right_one: Bool,+back: Bool,+left: U32,+right: U32,+stride: U32,
  same_value: {U32.is_eq(left,right) == same : Bool},right_value: {U32.is_eq(right,1) == right_one : Bool},back_value: {U32.is_eq(right,left) == back : Bool}) ->
  Stretches(Shapes.broadcast_extent(same,left_one,right_one,left,right),right,stride):
  match same left_one right_one back:
    case True{} left_one right_one back:
      %Address.equal_words(left,right,same_value) :
        {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(U32.is_eq(_,left),U32.is_eq(_,1),_,stride,left)) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(left,left),True{},Encoded.equal_words(left,left,{==})) :
        {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(_,U32.is_eq(left,1),left,stride,left)) == True{} : Bool}
      {==}
    case False{} True{} right_one back:
      %Equal.sym(Bool,U32.is_eq(right,right),True{},Encoded.equal_words(right,right,{==})) :
        {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(_,U32.is_eq(right,1),right,stride,right)) == True{} : Bool}
      {==}
    case False{} False{} True{} True{}:
      %Equal.sym(Bool,U32.is_eq(right,left),True{},back_value) :
        {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(_,U32.is_eq(right,1),right,stride,left)) == True{} : Bool}
      {==}
    case False{} False{} True{} False{}:
      %Equal.sym(Bool,U32.is_eq(right,left),False{},back_value) :
        {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(_,U32.is_eq(right,1),right,stride,left)) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(right,1),True{},right_value) :
        {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(False{},_,right,stride,left)) == True{} : Bool}
      {==}
    case False{} False{} False{} back: Unit{}

def prepended_exists(+axis: Maybe<&2,Views.Axis>,+rest: Maybe<&2,List<&2,Views.Axis>>,
  axis_exists: {Maybe.is_some(&2,Views.Axis,axis) == True{} : Bool},rest_exists: {Maybe.is_some(&2,List<&2,Views.Axis>,rest) == True{} : Bool}) ->
  {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretched_prepend(axis,rest)) == True{} : Bool}:
  match axis rest:
    case Some{axis} Some{rest}: {==}
    case Some{axis} None{}: rest_exists
    case None{} rest: axis_exists

def Compatible(left: List<&2,Views.Axis>,right: List<&2,Views.Axis>,shape: Maybe<&2,List<&2,U32>>) -> Type:
  match shape:
    case Some{shape}:
      Shape.Both<{Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(left,shape)) == True{} : Bool},
        {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(right,shape)) == True{} : Bool}>
    case None{}: Unit

def prepended_compatible(+extent: Maybe<&2,U32>,+shape: Maybe<&2,List<&2,U32>>,+left: U32,+left_stride: U32,+left_tail: List<&2,Views.Axis>,
  +right: U32,+right_stride: U32,+right_tail: List<&2,Views.Axis>,
  left_axis: Stretches(extent,left,left_stride),right_axis: Stretches(extent,right,right_stride),tail: Compatible(left_tail,right_tail,shape)) ->
  Compatible(Views.Axis{left,left_stride} <> left_tail,Views.Axis{right,right_stride} <> right_tail,Shapes.prepend_extent(extent,shape)):
  match extent shape:
    case Some{+extent} Some{+shape}:
      Shape.Both{left_tail_exists,right_tail_exists} = tail
      Shape.Both{
        prepended_exists(Views.stretched_axis(U32.is_eq(left,extent),U32.is_eq(left,1),left,left_stride,extent),Views.stretch_axes(left_tail,shape),left_axis,left_tail_exists),
        prepended_exists(Views.stretched_axis(U32.is_eq(right,extent),U32.is_eq(right,1),right,right_stride,extent),Views.stretch_axes(right_tail,shape),right_axis,right_tail_exists)}
    case Some{extent} None{}: Unit{}
    case None{} shape: Unit{}

# When the extents of two views of the same rank are compatible, both views
# stretch to the common extents.
def broadcast_stretches(+left: List<&2,Views.Axis>,+right: List<&2,Views.Axis>) ->
  Compatible(left,right,Shapes.broadcast_same_rank(Views.axis_extents(left),Views.axis_extents(right))):
  match left right:
    case Nil{} Nil{}: Shape.Both{{==},{==}}
    case Nil{} Con{Views.Axis{extent,stride},tail}: Unit{}
    case Con{Views.Axis{extent,stride},tail} Nil{}: Unit{}
    case Con{Views.Axis{+x,+left_stride},+left_tail} Con{Views.Axis{+y,+right_stride},+right_tail}:
      prepended_compatible(Shapes.broadcast_extent(U32.is_eq(x,y),U32.is_eq(x,1),U32.is_eq(y,1),x,y),
        Shapes.broadcast_same_rank(Views.axis_extents(left_tail),Views.axis_extents(right_tail)),x,left_stride,left_tail,y,right_stride,right_tail,
        left_stretches(U32.is_eq(x,y),U32.is_eq(x,1),U32.is_eq(y,1),x,left_stride,y,{==},{==}),
        right_stretches(U32.is_eq(x,y),U32.is_eq(x,1),U32.is_eq(y,1),U32.is_eq(y,x),x,y,right_stride,{==},{==},{==}),
        broadcast_stretches(left_tail,right_tail))
