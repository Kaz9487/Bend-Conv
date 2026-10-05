# Row-major ordinal progress, independently of native countdown arithmetic.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_contiguous_bounds.bend as ContiguousBounds
import ./storage_shape.bend as Shape
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./nat_order.bend as Order
import ./tensor_linear_layout.bend as LinearLayout

def last_coordinate(+extent: Nat,+coordinate: Nat,
  inside: {Nat.is_lt(coordinate,extent) == True{} : Bool},last: {Nat.is_lt(1n+coordinate,extent) == False{} : Bool}) -> {extent == 1n+coordinate : Nat}:
  match extent coordinate:
    case 0n 0n: Empty.absurd({0n == 1n : Nat},WordEncoding.false_true(inside))
    case 0n 1n+coordinate: Empty.absurd({0n == 2n+coordinate : Nat},WordEncoding.false_true(inside))
    case 1n 0n: {==}
    case 2n+extent 0n: Empty.absurd({2n+extent == 1n : Nat},WordEncoding.false_true(Equal.sym(Bool,True{},False{},last)))
    case 1n+ +extent 1n+ +coordinate:
      Equal.cong(Nat,Nat,value => 1n+value,extent,1n+coordinate,last_coordinate(extent,coordinate,inside,last))

def carried_position(+extent: Nat,+coordinate: Nat,+tail_position: Nat,last: {extent == 1n+coordinate : Nat}) ->
  {1n+Nat.add(coordinate,Nat.mul(extent,tail_position)) == Nat.mul(extent,1n+tail_position) : Nat}:
  %Equal.sym(Nat,extent,1n+coordinate,last) : {1n+Nat.add(coordinate,Nat.mul(_,tail_position)) == Nat.mul(_,1n+tail_position) : Nat}
  Equal.sym(Nat,Nat.mul(1n+coordinate,1n+tail_position),Nat.add(1n+coordinate,Nat.mul(1n+coordinate,tail_position)),RowMajor.multiply_successor_right(1n+coordinate,tail_position))

def carry_room(+extent: Nat,+coordinate: Nat,+tail_position: Nat,+tail_size: Nat,last: {extent == 1n+coordinate : Nat},
  room: {Nat.is_lt(1n+Nat.add(coordinate,Nat.mul(extent,tail_position)),Nat.mul(extent,tail_size)) == True{} : Bool}) ->
  {Nat.is_lt(1n+tail_position,tail_size) == True{} : Bool}:
  RowMajor.cancel_multiply_lt(1n+tail_position,tail_size,extent,{
    %RowMajor.multiply_commutative(extent,1n+tail_position) : {Nat.is_lt(_,Nat.mul(tail_size,extent)) == True{} : Bool}
    %RowMajor.multiply_commutative(extent,tail_size) : {Nat.is_lt(Nat.mul(extent,1n+tail_position),_) == True{} : Bool}
    %carried_position(extent,coordinate,tail_position,last) : {Nat.is_lt(_,Nat.mul(extent,tail_size)) == True{} : Bool}
    room : {Nat.is_lt(Nat.mul(1n+tail_position,extent),Nat.mul(tail_size,extent)) == True{} : Bool}})

def selected_position(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>,
  inside: {Nat.is_lt(coordinate,extent) == True{} : Bool},guard: {Nat.is_lt(1n+coordinate,extent) == selected : Bool},
  room: {Nat.is_lt(1n+CursorModel.position(CursorModel.Coordinate{extent,coordinate,stride} <> tail),CursorModel.extent_product(CursorModel.Coordinate{extent,coordinate,stride} <> tail)) == True{} : Bool},
  tail_advance: {Nat.is_lt(1n+CursorModel.position(tail),CursorModel.extent_product(tail)) == True{} : Bool} -> {CursorModel.position(CursorModel.advance(tail)) == 1n+CursorModel.position(tail) : Nat}) ->
  {CursorModel.position(Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,1n+coordinate,stride} <> tail,CursorModel.Coordinate{extent,0n,stride} <> CursorModel.advance(tail))) ==
    1n+CursorModel.position(CursorModel.Coordinate{extent,coordinate,stride} <> tail) : Nat}:
  match selected:
    case True{}: {==}
    case False{}:
      +last = last_coordinate(extent,coordinate,inside,guard)
      %Equal.sym(Nat,CursorModel.position(CursorModel.advance(tail)),1n+CursorModel.position(tail),tail_advance(carry_room(extent,coordinate,CursorModel.position(tail),CursorModel.extent_product(tail),last,room))) :
        {Nat.mul(extent,_) == 1n+Nat.add(coordinate,Nat.mul(extent,CursorModel.position(tail))) : Nat}
      Equal.sym(Nat,1n+Nat.add(coordinate,Nat.mul(extent,CursorModel.position(tail))),Nat.mul(extent,1n+CursorModel.position(tail)),carried_position(extent,coordinate,CursorModel.position(tail),last))

def advance_position(+axes: List<&2,CursorModel.Coordinate>,valid: CursorLaws.Valid(axes),
  room: {Nat.is_lt(1n+CursorModel.position(axes),CursorModel.extent_product(axes)) == True{} : Bool}) ->
  {CursorModel.position(CursorModel.advance(axes)) == 1n+CursorModel.position(axes) : Nat}:
  match axes:
    case []: Empty.absurd({0n == 1n : Nat},WordEncoding.false_true(room))
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      Shape.Both{inside,tail_valid} = valid
      selected_position(Nat.is_lt(1n+coordinate,extent),extent,coordinate,stride,tail,inside,{==},room,
        next_room => advance_position(tail,tail_valid,next_room))

def selected_size(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>,
  preserved: {CursorModel.extent_product(CursorModel.advance(tail)) == CursorModel.extent_product(tail) : Nat}) ->
  {CursorModel.extent_product(Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,1n+coordinate,stride} <> tail,
      CursorModel.Coordinate{extent,0n,stride} <> CursorModel.advance(tail))) == CursorModel.extent_product(CursorModel.Coordinate{extent,coordinate,stride} <> tail) : Nat}:
  match selected:
    case True{}: {==}
    case False{}: Equal.cong(Nat,Nat,value => Nat.mul(extent,value),CursorModel.extent_product(CursorModel.advance(tail)),CursorModel.extent_product(tail),preserved)

def advance_size(+axes: List<&2,CursorModel.Coordinate>) -> {CursorModel.extent_product(CursorModel.advance(axes)) == CursorModel.extent_product(axes) : Nat}:
  match axes:
    case []: {==}
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      selected_size(Nat.is_lt(1n+coordinate,extent),extent,coordinate,stride,tail,advance_size(tail))

def included_zero(+selected: Bool,+extent: Nat,+stride: Nat,+reversed: List<&2,CursorModel.Coordinate>,zero: {CursorModel.position(reversed) == 0n : Nat}) ->
  {CursorModel.position(Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,0n,stride} <> reversed,reversed)) == 0n : Nat}:
  match selected:
    case False{}: zero
    case True{}:
      %Equal.sym(Nat,CursorModel.position(reversed),0n,zero) : {Nat.mul(extent,_) == 0n : Nat}
      RowMajor.multiply_zero_right(extent)

def prepared_zero(+axes: List<&2,Views.Axis>,+reversed: List<&2,CursorModel.Coordinate>,zero: {CursorModel.position(reversed) == 0n : Nat}) ->
  {CursorModel.position(CursorModel.prepare_axes(axes,reversed)) == 0n : Nat}:
  match axes:
    case []: zero
    case Views.Axis{+extent,+stride} <> tail:
      prepared_zero(tail,Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed),
        included_zero(Nat.is_lt(1n,U32.to_nat(extent)),U32.to_nat(extent),U32.to_nat(stride),reversed,zero))

def included_size(+extent: Nat,+stride: Nat,+reversed: List<&2,CursorModel.Coordinate>,positive: {Nat.is_lt(0n,extent) == True{} : Bool}) ->
  {CursorModel.extent_product(Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,extent),CursorModel.Coordinate{extent,0n,stride} <> reversed,reversed)) == Nat.mul(extent,CursorModel.extent_product(reversed)) : Nat}:
  match extent:
    case 0n: Empty.absurd({CursorModel.extent_product(reversed) == 0n : Nat},WordEncoding.false_true(positive))
    case 1n: Equal.sym(Nat,Nat.add(CursorModel.extent_product(reversed),0n),CursorModel.extent_product(reversed),Addition.zero_right(CursorModel.extent_product(reversed)))
    case 2n+rest: {==}

def product_reorder(+left: Nat,+right: Nat,+last: Nat) -> {Nat.mul(left,Nat.mul(right,last)) == Nat.mul(Nat.mul(right,left),last) : Nat}:
  Equal.trans(Nat,Nat.mul(left,Nat.mul(right,last)),Nat.mul(Nat.mul(left,right),last),Nat.mul(Nat.mul(right,left),last),
    Equal.sym(Nat,Nat.mul(Nat.mul(left,right),last),Nat.mul(left,Nat.mul(right,last)),RowMajor.assoc(left,right,last)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,last),Nat.mul(left,right),Nat.mul(right,left),RowMajor.multiply_commutative(left,right)))

def prepared_size(+axes: List<&2,Views.Axis>,+reversed: List<&2,CursorModel.Coordinate>,+nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}) ->
  {CursorModel.extent_product(CursorModel.prepare_axes(axes,reversed)) == Nat.mul(ShapeLaws.elements(Views.axis_extents(axes)),CursorModel.extent_product(reversed)) : Nat}:
  match axes:
    case []: Equal.sym(Nat,Nat.add(CursorModel.extent_product(reversed),0n),CursorModel.extent_product(reversed),Addition.zero_right(CursorModel.extent_product(reversed)))
    case Views.Axis{+extent,+stride} <> +tail:
      %product_reorder(ShapeLaws.elements(Views.axis_extents(tail)),U32.to_nat(extent),CursorModel.extent_product(reversed)) :
        {CursorModel.extent_product(CursorModel.prepare_axes(Views.Axis{extent,stride} <> tail,reversed)) == _ : Nat}
      %included_size(U32.to_nat(extent),U32.to_nat(stride),reversed,
        ContiguousBounds.extent_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty))) :
        {CursorModel.extent_product(CursorModel.prepare_axes(Views.Axis{extent,stride} <> tail,reversed)) == Nat.mul(ShapeLaws.elements(Views.axis_extents(tail)),_) : Nat}
      prepared_size(tail,Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed),
        Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty))

def initial_size(+axes: List<&2,Views.Axis>,nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}) ->
  {CursorModel.extent_product(CursorModel.prepare_axes(axes,[])) == ShapeLaws.elements(Views.axis_extents(axes)) : Nat}:
  %ShapeLaws.times_one(ShapeLaws.elements(Views.axis_extents(axes))) : {CursorModel.extent_product(CursorModel.prepare_axes(axes,[])) == _ : Nat}
  prepared_size(axes,[],nonempty)

def iterate(count: Nat,axes: List<&2,CursorModel.Coordinate>) -> List<&2,CursorModel.Coordinate>:
  match count:
    case 0n: axes
    case 1n+remaining: iterate(remaining,CursorModel.advance(axes))

def iterate_valid(count: Nat,+axes: List<&2,CursorModel.Coordinate>,valid: CursorLaws.Valid(axes)) -> CursorLaws.Valid(iterate(count,axes)):
  match count:
    case 0n: valid
    case 1n+remaining: iterate_valid(remaining,CursorModel.advance(axes),CursorLaws.advance_valid(axes,valid))

def iterate_linear(count: Nat,+axes: List<&2,CursorModel.Coordinate>,+stride: Nat,linear: LinearLayout.Linear(axes,stride)) -> LinearLayout.Linear(iterate(count,axes),stride):
  match count:
    case 0n: linear
    case 1n+remaining: iterate_linear(remaining,CursorModel.advance(axes),stride,LinearLayout.advance_linear(axes,stride,linear))

def first_room(+position: Nat,+remaining: Nat,+size: Nat,
  room: {Nat.is_lt(Nat.add(position,1n+remaining),size) == True{} : Bool}) -> {Nat.is_lt(1n+position,size) == True{} : Bool}:
  Order.le_lt_trans(1n+position,Nat.add(1n+position,remaining),size,Order.le_add(1n+position,remaining),{
    %Addition.successor_right(position,remaining) : {Nat.is_lt(_,size) == True{} : Bool}
    room : {Nat.is_lt(Nat.add(1n+position,remaining),size) == True{} : Bool}})

def iterate_position(+count: Nat,+axes: List<&2,CursorModel.Coordinate>,+valid: CursorLaws.Valid(axes),
  +room: {Nat.is_lt(Nat.add(CursorModel.position(axes),count),CursorModel.extent_product(axes)) == True{} : Bool}) ->
  {CursorModel.position(iterate(count,axes)) == Nat.add(CursorModel.position(axes),count) : Nat}:
  match count:
    case 0n: Equal.sym(Nat,Nat.add(CursorModel.position(axes),0n),CursorModel.position(axes),Addition.zero_right(CursorModel.position(axes)))
    case 1n+ +remaining:
      +progress = advance_position(axes,valid,first_room(CursorModel.position(axes),remaining,CursorModel.extent_product(axes),room))
      %Equal.sym(Nat,Nat.add(CursorModel.position(axes),1n+remaining),1n+Nat.add(CursorModel.position(axes),remaining),Addition.successor_right(CursorModel.position(axes),remaining)) :
        {CursorModel.position(iterate(remaining,CursorModel.advance(axes))) == _ : Nat}
      %progress : {CursorModel.position(iterate(remaining,CursorModel.advance(axes))) == Nat.add(_,remaining) : Nat}
      iterate_position(remaining,CursorModel.advance(axes),CursorLaws.advance_valid(axes,valid),{
        %Equal.sym(Nat,CursorModel.extent_product(CursorModel.advance(axes)),CursorModel.extent_product(axes),advance_size(axes)) :
          {Nat.is_lt(Nat.add(CursorModel.position(CursorModel.advance(axes)),remaining),_) == True{} : Bool}
        %Equal.sym(Nat,CursorModel.position(CursorModel.advance(axes)),1n+CursorModel.position(axes),progress) : {Nat.is_lt(Nat.add(_,remaining),CursorModel.extent_product(axes)) == True{} : Bool}
        %Addition.successor_right(CursorModel.position(axes),remaining) : {Nat.is_lt(_,CursorModel.extent_product(axes)) == True{} : Bool}
        room : {Nat.is_lt(Nat.add(CursorModel.position(CursorModel.advance(axes)),remaining),CursorModel.extent_product(CursorModel.advance(axes))) == True{} : Bool}})

def linear_reads(+index: Nat,+axes: List<&2,CursorModel.Coordinate>,+offset: Nat,+valid: CursorLaws.Valid(axes),linear: LinearLayout.Linear(axes,1n),
  room: {Nat.is_lt(Nat.add(CursorModel.position(axes),index),CursorModel.extent_product(axes)) == True{} : Bool}) ->
  {CursorModel.address(iterate(index,axes),offset) == Nat.add(offset,Nat.add(CursorModel.position(axes),index)) : Nat}:
  %iterate_position(index,axes,valid,room) : {CursorModel.address(iterate(index,axes),offset) == Nat.add(offset,_) : Nat}
  %Addition.zero_right(CursorModel.position(iterate(index,axes))) : {CursorModel.address(iterate(index,axes),offset) == Nat.add(offset,_) : Nat}
  LinearLayout.linear_address(iterate(index,axes),offset,1n,iterate_linear(index,axes,1n,linear))

def contiguous_addresses(+axes: List<&2,Views.Axis>,+offset: U32,+index: Nat,
  +nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  contiguous: {Views.is_contiguous(Views.View{axes,offset}) == True{} : Bool},
  inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {CursorModel.address(iterate(index,CursorModel.prepare_axes(axes,[])),U32.to_nat(offset)) == Nat.add(U32.to_nat(offset),index) : Nat}:
  %prepared_zero(axes,[],{==}) : {CursorModel.address(iterate(index,CursorModel.prepare_axes(axes,[])),U32.to_nat(offset)) == Nat.add(U32.to_nat(offset),Nat.add(_,index)) : Nat}
  linear_reads(index,CursorModel.prepare_axes(axes,[]),U32.to_nat(offset),CursorLaws.prepared_valid(axes,[],Unit{}),LinearLayout.initial_linear(axes,offset,nonempty,contiguous),{
    %Equal.sym(Nat,CursorModel.position(CursorModel.prepare_axes(axes,[])),0n,prepared_zero(axes,[],{==})) : {Nat.is_lt(Nat.add(_,index),CursorModel.extent_product(CursorModel.prepare_axes(axes,[]))) == True{} : Bool}
    %Equal.sym(Nat,CursorModel.extent_product(CursorModel.prepare_axes(axes,[])),ShapeLaws.elements(Views.axis_extents(axes)),initial_size(axes,nonempty)) : {Nat.is_lt(index,_) == True{} : Bool}
    inside : {Nat.is_lt(Nat.add(CursorModel.position(CursorModel.prepare_axes(axes,[])),index),CursorModel.extent_product(CursorModel.prepare_axes(axes,[]))) == True{} : Bool}})
