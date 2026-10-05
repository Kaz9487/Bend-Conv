# Recover the exact stride relation behind the contiguous fast path.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_contiguous_bounds.bend as ContiguousBounds
import ./tensor_view_bounds.bend as ViewBounds
import ./tensor_cursor_model.bend as CursorModel
import ./boolean_logic.bend as Boolean
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./word_encoding.bend as WordEncoding

law Layout:
  for axes: List<&2,Views.Axis>
  Data

def Layout(axes):
  match axes:
    case []: Unit
    case Views.Axis{extent,stride} <> tail:
      Shape.Both<Boolean.When(Nat.is_lt(1n,U32.to_nat(extent)),{U32.to_nat(stride) == ShapeLaws.elements(Views.axis_extents(tail)) : Nat}),Layout(tail)>

law LayoutResult:
  for result: Maybe<&2,U32>
  for axes: List<&2,Views.Axis>
  Data

def LayoutResult(result,axes):
  match result:
    case None{}: Unit
    case Some{value}: Layout(axes)

def head_rule(+extent: Nat,+stride: Nat,+count: Nat,
  valid: {Nat.is_le(extent,1n) || Nat.is_eq(stride,count) == True{} : Bool}) -> Boolean.When(Nat.is_lt(1n,extent),{stride == count : Nat}):
  match extent:
    case 0n: Unit{}
    case 1n: Unit{}
    case 2n+rest: Order.equal_of_is_eq(stride,count,valid)

def output_layout(+result: Maybe<&2,U32>,+axes: List<&2,Views.Axis>,layout: Layout(axes)) -> LayoutResult(result,axes):
  match result:
    case None{}: Unit{}
    case Some{value}: layout

def selected_layout(+valid: Bool,+extent: U32,+stride: U32,+count: U32,+tail: List<&2,Views.Axis>,
  guard: {U32.is_le(extent,1) || U32.is_eq(stride,count) == valid : Bool},
  counted: {ShapeLaws.elements(Views.axis_extents(tail)) == U32.to_nat(count) : Nat},layout: Layout(tail)) ->
  LayoutResult(Views.contiguous_product(valid,extent,count),Views.Axis{extent,stride} <> tail):
  match valid:
    case False{}: Unit{}
    case True{}:
      output_layout(Shapes.checked_multiply(extent,count),Views.Axis{extent,stride} <> tail,Shape.Both{{
        %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(tail)),U32.to_nat(count),counted) : Boolean.When(Nat.is_lt(1n,U32.to_nat(extent)),{U32.to_nat(stride) == _ : Nat})
        head_rule(U32.to_nat(extent),U32.to_nat(stride),U32.to_nat(count),ContiguousBounds.stride_guard(extent,stride,count,guard)) :
          Boolean.When(Nat.is_lt(1n,U32.to_nat(extent)),{U32.to_nat(stride) == ShapeLaws.elements(Views.axis_extents(tail)) : Nat})},layout})

def axis_layout(+inner: Maybe<&2,U32>,+extent: U32,+stride: U32,+tail: List<&2,Views.Axis>,
  facts: ContiguousBounds.SpanCount(inner,ShapeLaws.elements(Views.axis_extents(tail)),ViewBounds.last_span(tail)),layout: LayoutResult(inner,tail)) ->
  LayoutResult(Views.contiguous_axis(extent,stride,inner),Views.Axis{extent,stride} <> tail):
  match inner:
    case None{}: Unit{}
    case Some{+count}:
      Shape.Both{counted,span} = facts
      selected_layout(U32.is_le(extent,1) || U32.is_eq(stride,count),extent,stride,count,tail,{==},counted,layout)

def layout_result(+axes: List<&2,Views.Axis>,nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}) -> LayoutResult(Views.contiguous_size(axes),axes):
  match axes:
    case []: Unit{}
    case Views.Axis{+extent,+stride} <> +tail:
      +tail_nonempty = Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty)
      axis_layout(Views.contiguous_size(tail),extent,stride,tail,ContiguousBounds.contiguous(tail,tail_nonempty),layout_result(tail,tail_nonempty))

def accepted_layout(+result: Maybe<&2,U32>,+axes: List<&2,Views.Axis>,
  found: {Views.has_count(result) == True{} : Bool},layout: LayoutResult(result,axes)) -> Layout(axes):
  match result:
    case None{}: Empty.absurd(Layout(axes),WordEncoding.false_true(found))
    case Some{value}: layout

def checked_layout(+axes: List<&2,Views.Axis>,+offset: U32,+nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  contiguous: {Views.is_contiguous(Views.View{axes,offset}) == True{} : Bool}) -> Layout(axes):
  accepted_layout(Views.contiguous_size(axes),axes,{
    %nonempty : {Views.contiguous_nonempty(_,axes) == True{} : Bool}
    contiguous : {Views.has_count(Views.contiguous_size(axes)) == True{} : Bool}},layout_result(axes,nonempty))

law Linear:
  for axes: List<&2,CursorModel.Coordinate>
  for stride: Nat
  Data

def Linear(axes,stride):
  match axes:
    case []: Unit
    case CursorModel.Coordinate{extent,coordinate,actual} <> tail:
      Shape.Both<{actual == stride : Nat},Linear(tail,Nat.mul(stride,extent))>

def included_linear(+extent: Nat,+stride: Nat,+tail_count: Nat,+reversed: List<&2,CursorModel.Coordinate>,
  positive: {Nat.is_lt(0n,extent) == True{} : Bool},
  head: Boolean.When(Nat.is_lt(1n,extent),{stride == tail_count : Nat}),previous: Linear(reversed,Nat.mul(extent,tail_count))) ->
  Linear(Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,extent),CursorModel.Coordinate{extent,0n,stride} <> reversed,reversed),tail_count):
  match extent:
    case 0n: Empty.absurd(Linear(reversed,tail_count),WordEncoding.false_true(positive))
    case 1n:
      %Addition.zero_right(tail_count) : Linear(reversed,_)
      previous
    case 2n+ +rest:
      Shape.Both{head,{
        %RowMajor.multiply_commutative(2n+rest,tail_count) : Linear(reversed,_)
        previous : Linear(reversed,Nat.mul(tail_count,2n+rest))}}

def prepared_linear(+axes: List<&2,Views.Axis>,+reversed: List<&2,CursorModel.Coordinate>,
  +nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},layout: Layout(axes),
  previous: Linear(reversed,ShapeLaws.elements(Views.axis_extents(axes)))) -> Linear(CursorModel.prepare_axes(axes,reversed),1n):
  match axes:
    case []: previous
    case Views.Axis{+extent,+stride} <> +tail:
      Shape.Both{head,tail_layout} = layout
      prepared_linear(tail,Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed),
        Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty),tail_layout,
        included_linear(U32.to_nat(extent),U32.to_nat(stride),ShapeLaws.elements(Views.axis_extents(tail)),reversed,
          ContiguousBounds.extent_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty)),head,previous))

def initial_linear(+axes: List<&2,Views.Axis>,+offset: U32,+nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  contiguous: {Views.is_contiguous(Views.View{axes,offset}) == True{} : Bool}) -> Linear(CursorModel.prepare_axes(axes,[]),1n):
  prepared_linear(axes,[],nonempty,checked_layout(axes,offset,nonempty,contiguous),Unit{})

def linear_address(+axes: List<&2,CursorModel.Coordinate>,+offset: Nat,+stride: Nat,linear: Linear(axes,stride)) ->
  {CursorModel.address(axes,offset) == Nat.add(offset,Nat.mul(stride,CursorModel.position(axes))) : Nat}:
  match axes:
    case []:
      %Equal.sym(Nat,Nat.mul(stride,0n),0n,RowMajor.multiply_zero_right(stride)) : {offset == Nat.add(offset,_) : Nat}
      Equal.sym(Nat,Nat.add(offset,0n),offset,Addition.zero_right(offset))
    case CursorModel.Coordinate{+extent,+coordinate,+actual} <> +tail:
      Shape.Both{equal,tail_linear} = linear
      %Equal.sym(Nat,actual,stride,equal) : {Nat.add(Nat.mul(coordinate,_),CursorModel.address(tail,offset)) == Nat.add(offset,Nat.mul(stride,CursorModel.position(CursorModel.Coordinate{extent,coordinate,actual} <> tail))) : Nat}
      %Equal.sym(Nat,CursorModel.address(tail,offset),Nat.add(offset,Nat.mul(Nat.mul(stride,extent),CursorModel.position(tail))),linear_address(tail,offset,Nat.mul(stride,extent),tail_linear)) :
        {Nat.add(Nat.mul(coordinate,stride),_) == Nat.add(offset,Nat.mul(stride,Nat.add(coordinate,Nat.mul(extent,CursorModel.position(tail))))) : Nat}
      %Equal.sym(Nat,Nat.mul(stride,Nat.add(coordinate,Nat.mul(extent,CursorModel.position(tail)))),
        Nat.add(Nat.mul(stride,coordinate),Nat.mul(stride,Nat.mul(extent,CursorModel.position(tail)))),RowMajor.right_distrib(stride,coordinate,Nat.mul(extent,CursorModel.position(tail)))) :
        {Nat.add(Nat.mul(coordinate,stride),Nat.add(offset,Nat.mul(Nat.mul(stride,extent),CursorModel.position(tail)))) == Nat.add(offset,_) : Nat}
      %RowMajor.multiply_commutative(coordinate,stride) :
        {Nat.add(Nat.mul(coordinate,stride),Nat.add(offset,Nat.mul(Nat.mul(stride,extent),CursorModel.position(tail)))) == Nat.add(offset,Nat.add(_,Nat.mul(stride,Nat.mul(extent,CursorModel.position(tail))))) : Nat}
      %RowMajor.assoc(stride,extent,CursorModel.position(tail)) :
        {Nat.add(Nat.mul(coordinate,stride),Nat.add(offset,Nat.mul(Nat.mul(stride,extent),CursorModel.position(tail)))) == Nat.add(offset,Nat.add(Nat.mul(coordinate,stride),_)) : Nat}
      RowMajor.add_swap(Nat.mul(coordinate,stride),offset,Nat.mul(Nat.mul(stride,extent),CursorModel.position(tail)))

def selected_linear(+selected: Bool,+extent: Nat,+coordinate: Nat,+actual: Nat,+tail: List<&2,CursorModel.Coordinate>,+stride: Nat,
  head: {actual == stride : Nat},tail_linear: Linear(tail,Nat.mul(stride,extent)),next_linear: Linear(CursorModel.advance(tail),Nat.mul(stride,extent))) ->
  Linear(Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,1n+coordinate,actual} <> tail,
    CursorModel.Coordinate{extent,0n,actual} <> CursorModel.advance(tail)),stride):
  match selected:
    case True{}: Shape.Both{head,tail_linear}
    case False{}: Shape.Both{head,next_linear}

def advance_linear(+axes: List<&2,CursorModel.Coordinate>,+stride: Nat,linear: Linear(axes,stride)) -> Linear(CursorModel.advance(axes),stride):
  match axes:
    case []: Unit{}
    case CursorModel.Coordinate{+extent,+coordinate,+actual} <> +tail:
      Shape.Both{head,+tail_linear} = linear
      selected_linear(Nat.is_lt(1n+coordinate,extent),extent,coordinate,actual,tail,stride,head,tail_linear,advance_linear(tail,Nat.mul(stride,extent),tail_linear))
