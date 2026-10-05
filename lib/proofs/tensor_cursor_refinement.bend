# Relate native countdowns and addresses to independent increasing coordinates.
# The eight flat levels share one generic level proof; outer axes are proved
# through a non-tail reference of the runtime carry loop.
import Base
import ../tensor_cursor.bend as Cursor
import ./tensor_cursor_model.bend as CursorModel
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./word_arithmetic.bend as WordArithmetic
import ./word_subtraction.bend as Subtraction
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ../tensor_view.bend as Views
import ./u32_comparison.bend as Comparison
import ./u32_full_range.bend as FullRange
import ./word_multiplication.bend as Multiplication
import ./bounded_u32_arithmetic.bend as Arithmetic

law Valid:
  for axes: List<&2,CursorModel.Coordinate>
  Data

def Valid(axes):
  match axes:
    case []: Unit
    case CursorModel.Coordinate{extent,coordinate,stride} <> tail:
      Shape.Both<{Nat.is_lt(coordinate,extent) == True{} : Bool},Valid(tail)>

def represent_outer(axes: List<&2,CursorModel.Coordinate>) -> List<&2,Cursor.Axis>:
  match axes:
    case []: []
    case CursorModel.Coordinate{+extent,coordinate,+stride} <> tail:
      Cursor.Axis{extent,Nat.sub(extent,coordinate),U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(extent,1n),stride))} <> represent_outer(tail)

def represent_level(~Rest: Data,~represent_rest: List<&2,CursorModel.Coordinate> -> Rest,axes: List<&2,CursorModel.Coordinate>) -> Cursor.Level<Rest>:
  match axes:
    case []: Cursor.Level{Cursor.absent(),represent_rest([])}
    case CursorModel.Coordinate{+extent,coordinate,+stride} <> tail:
      Cursor.Level{Cursor.Axis{extent,Nat.sub(extent,coordinate),U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(extent,1n),stride))},represent_rest(tail)}

def represent_axes(axes: List<&2,CursorModel.Coordinate>) -> Cursor.Levels():
  represent_level(~Cursor.Depth1(),~represent_level(~Cursor.Depth0(),~represent_outer),axes)

def represent(+axes: List<&2,CursorModel.Coordinate>,offset: Nat) -> Cursor.Cursor:
  Cursor.Cursor{represent_axes(axes),U32.from_nat(CursorModel.address(axes,offset))}

# A representation paired with the address of the same coordinates.
def placed(~Value: Data,~represent: List<&2,CursorModel.Coordinate> -> Value,+axes: List<&2,CursorModel.Coordinate>,offset: Nat) -> Value & U32:
  (represent(axes),U32.from_nat(CursorModel.address(axes,offset)))

def subtract_prefix(+prefix: Nat,+value: Nat) -> {Nat.sub(Nat.add(prefix,value),prefix) == value : Nat}:
  match prefix:
    case 0n: Order.subtract_zero(value)
    case 1n+rest: subtract_prefix(rest,value)

def subtract_suffix(+value: Nat,+suffix: Nat) -> {Nat.sub(Nat.add(value,suffix),suffix) == value : Nat}:
  %Equal.sym(Nat,Nat.add(value,suffix),Nat.add(suffix,value),Order.commutative(value,suffix)) : {Nat.sub(_,suffix) == value : Nat}
  subtract_prefix(suffix,value)

def less_reflexive(+value: Nat) -> {Nat.is_lt(value,value) == False{} : Bool}:
  match value:
    case 0n: {==}
    case 1n+rest: less_reflexive(rest)

def remaining_positive(+extent: Nat,+coordinate: Nat,inside: {Nat.is_lt(coordinate,extent) == True{} : Bool}) ->
  {Nat.is_lt(0n,Nat.sub(extent,coordinate)) == True{} : Bool}:
  match extent coordinate:
    case 0n 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(inside))
    case 0n 1n+coordinate: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(inside))
    case 1n+extent 0n: {==}
    case 1n+extent 1n+coordinate: remaining_positive(extent,coordinate,inside)

def coordinate_room(+remaining: Nat,+coordinate: Nat) ->
  {Nat.is_lt(1n+coordinate,Nat.add(2n+remaining,coordinate)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(2n+remaining,coordinate),Nat.add(coordinate,2n+remaining),Order.commutative(2n+remaining,coordinate)) :
    {Nat.is_lt(1n+coordinate,_) == True{} : Bool}
  %Addition.zero_right(coordinate) :
    {Nat.is_lt(1n+_,Nat.add(coordinate,2n+remaining)) == True{} : Bool}
  %Addition.successor_right(coordinate,0n) : {Nat.is_lt(_,Nat.add(coordinate,2n+remaining)) == True{} : Bool}
  Order.offset_lt(coordinate,1n,2n+remaining,{==})

def increment_address(+coordinate: Nat,+stride: Nat,+tail_address: Nat) ->
  {Nat.add(Nat.add(Nat.mul(coordinate,stride),tail_address),stride) == Nat.add(Nat.mul(1n+coordinate,stride),tail_address) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.add(stride,Nat.mul(coordinate,stride)),tail_address),Nat.add(stride,Nat.add(Nat.mul(coordinate,stride),tail_address)),
    Addition.associative(stride,Nat.mul(coordinate,stride),tail_address)) :
    {Nat.add(Nat.add(Nat.mul(coordinate,stride),tail_address),stride) == _ : Nat}
  Order.commutative(Nat.add(Nat.mul(coordinate,stride),tail_address),stride)

def reset_outer(+extent: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>) ->
  {represent_outer(CursorModel.Coordinate{extent,0n,stride} <> tail) ==
    Cursor.Axis{extent,extent,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(extent,1n),stride))} <> represent_outer(tail) : List<&2,Cursor.Axis>}:
  %Order.subtract_zero(extent) :
    {represent_outer(CursorModel.Coordinate{extent,0n,stride} <> tail) ==
      Cursor.Axis{extent,_,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(extent,1n),stride))} <> represent_outer(tail) : List<&2,Cursor.Axis>}
  {==}

def head_outer(+remaining: Nat,+coordinate: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>) ->
  {represent_outer(CursorModel.Coordinate{Nat.add(remaining,coordinate),coordinate,stride} <> tail) ==
    Cursor.Axis{Nat.add(remaining,coordinate),remaining,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(remaining,coordinate),1n),stride))} <> represent_outer(tail) : List<&2,Cursor.Axis>}:
  %subtract_suffix(remaining,coordinate) :
    {represent_outer(CursorModel.Coordinate{Nat.add(remaining,coordinate),coordinate,stride} <> tail) ==
      Cursor.Axis{Nat.add(remaining,coordinate),_,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(remaining,coordinate),1n),stride))} <> represent_outer(tail) : List<&2,Cursor.Axis>}
  {==}

def reset_level(~Rest: Data,~represent_rest: List<&2,CursorModel.Coordinate> -> Rest,+extent: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>) ->
  {represent_level(~Rest,~represent_rest,CursorModel.Coordinate{extent,0n,stride} <> tail) ==
    Cursor.Level{Cursor.Axis{extent,extent,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(extent,1n),stride))},represent_rest(tail)} : Cursor.Level<Rest>}:
  %Order.subtract_zero(extent) :
    {represent_level(~Rest,~represent_rest,CursorModel.Coordinate{extent,0n,stride} <> tail) ==
      Cursor.Level{Cursor.Axis{extent,_,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(extent,1n),stride))},represent_rest(tail)} : Cursor.Level<Rest>}
  {==}

def head_level(~Rest: Data,~represent_rest: List<&2,CursorModel.Coordinate> -> Rest,+remaining: Nat,+coordinate: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>) ->
  {represent_level(~Rest,~represent_rest,CursorModel.Coordinate{Nat.add(remaining,coordinate),coordinate,stride} <> tail) ==
    Cursor.Level{Cursor.Axis{Nat.add(remaining,coordinate),remaining,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(remaining,coordinate),1n),stride))},represent_rest(tail)} : Cursor.Level<Rest>}:
  %subtract_suffix(remaining,coordinate) :
    {represent_level(~Rest,~represent_rest,CursorModel.Coordinate{Nat.add(remaining,coordinate),coordinate,stride} <> tail) ==
      Cursor.Level{Cursor.Axis{Nat.add(remaining,coordinate),_,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(remaining,coordinate),1n),stride))},represent_rest(tail)} : Cursor.Level<Rest>}
  {==}

def decrement_remaining(+remaining: Nat,+coordinate: Nat) ->
  {Nat.sub(Nat.add(2n+remaining,coordinate),1n+coordinate) == 1n+remaining : Nat}:
  %Addition.successor_right(1n+remaining,coordinate) : {Nat.sub(_,1n+coordinate) == 1n+remaining : Nat}
  subtract_suffix(1n+remaining,1n+coordinate)

def subtract_front(+head: Nat,+tail: Nat) ->
  {U32.sub(U32.from_nat(Nat.add(head,tail)),U32.from_nat(head)) == U32.from_nat(tail) : U32}:
  %Equal.sym(Nat,Nat.add(head,tail),Nat.add(tail,head),Order.commutative(head,tail)) :
    {U32.sub(U32.from_nat(_),U32.from_nat(head)) == U32.from_nat(tail) : U32}
  Subtraction.encoded_subtract_add(tail,head)

def level_step(~Rest: Data,~next: Rest -> U32 -> Rest & U32,~represent_rest: List<&2,CursorModel.Coordinate> -> Rest,
  +remaining: Nat,+coordinate: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>,+offset: Nat,
  positive: {Nat.is_lt(0n,remaining) == True{} : Bool},
  after: {next(represent_rest(tail),U32.from_nat(CursorModel.address(tail,offset))) == placed(~Rest,~represent_rest,CursorModel.advance(tail),offset) : Rest & U32}) ->
  {Cursor.step(~Rest,~next,Cursor.Level{Cursor.Axis{Nat.add(remaining,coordinate),remaining,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(remaining,coordinate),1n),stride))},represent_rest(tail)},U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
    placed(~Cursor.Level<Rest>,~represent_level(~Rest,~represent_rest),CursorModel.advance(CursorModel.Coordinate{Nat.add(remaining,coordinate),coordinate,stride} <> tail),offset) : Cursor.Level<Rest> & U32}:
  match remaining:
    case 0n:
      Empty.absurd({Cursor.step(~Rest,~next,Cursor.Level{Cursor.Axis{coordinate,0n,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(coordinate,1n),stride))},represent_rest(tail)},U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
        placed(~Cursor.Level<Rest>,~represent_level(~Rest,~represent_rest),CursorModel.advance(CursorModel.Coordinate{coordinate,coordinate,stride} <> tail),offset) : Cursor.Level<Rest> & U32},WordEncoding.false_true(positive))
    case 1n:
      %Equal.sym(Bool,Nat.is_lt(1n+coordinate,1n+coordinate),False{},less_reflexive(1n+coordinate)) :
        {Cursor.step(~Rest,~next,Cursor.Level{Cursor.Axis{1n+coordinate,1n,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(coordinate,0n),stride))},represent_rest(tail)},U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
          placed(~Cursor.Level<Rest>,~represent_level(~Rest,~represent_rest),Bool.pick(List<&2,CursorModel.Coordinate>,_,CursorModel.Coordinate{1n+coordinate,1n+coordinate,stride} <> tail,
            CursorModel.Coordinate{1n+coordinate,0n,stride} <> CursorModel.advance(tail)),offset) : Cursor.Level<Rest> & U32}
      %Equal.sym(Cursor.Level<Rest>,represent_level(~Rest,~represent_rest,CursorModel.Coordinate{1n+coordinate,0n,stride} <> CursorModel.advance(tail)),
        Cursor.Level{Cursor.Axis{1n+coordinate,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(coordinate,0n),stride))},represent_rest(CursorModel.advance(tail))},
        reset_level(~Rest,~represent_rest,1n+coordinate,stride,CursorModel.advance(tail))) :
        {Cursor.step(~Rest,~next,Cursor.Level{Cursor.Axis{1n+coordinate,1n,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(coordinate,0n),stride))},represent_rest(tail)},U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
          (_,U32.from_nat(CursorModel.address(CursorModel.advance(tail),offset))) : Cursor.Level<Rest> & U32}
      %Equal.sym(Nat,Nat.sub(coordinate,0n),coordinate,Order.subtract_zero(coordinate)) :
        {Cursor.step(~Rest,~next,Cursor.Level{Cursor.Axis{1n+coordinate,1n,U32.from_nat(stride),U32.from_nat(Nat.mul(_,stride))},represent_rest(tail)},U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
          (Cursor.Level{Cursor.Axis{1n+coordinate,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(_,stride))},represent_rest(CursorModel.advance(tail))},
            U32.from_nat(CursorModel.address(CursorModel.advance(tail),offset))) : Cursor.Level<Rest> & U32}
      %Equal.sym(U32,U32.sub(U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset))),U32.from_nat(Nat.mul(coordinate,stride))),
        U32.from_nat(CursorModel.address(tail,offset)),subtract_front(Nat.mul(coordinate,stride),CursorModel.address(tail,offset))) :
        {Cursor.reset(~Rest,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(coordinate,stride)),next(represent_rest(tail),_)) ==
          (Cursor.Level{Cursor.Axis{1n+coordinate,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(coordinate,stride))},represent_rest(CursorModel.advance(tail))},
            U32.from_nat(CursorModel.address(CursorModel.advance(tail),offset))) : Cursor.Level<Rest> & U32}
      %Equal.sym(Rest & U32,next(represent_rest(tail),U32.from_nat(CursorModel.address(tail,offset))),placed(~Rest,~represent_rest,CursorModel.advance(tail),offset),after) :
        {Cursor.reset(~Rest,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(coordinate,stride)),_) ==
          (Cursor.Level{Cursor.Axis{1n+coordinate,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(coordinate,stride))},represent_rest(CursorModel.advance(tail))},
            U32.from_nat(CursorModel.address(CursorModel.advance(tail),offset))) : Cursor.Level<Rest> & U32}
      {==}
    case 2n+ +rest:
      %Equal.sym(Bool,Nat.is_lt(1n+coordinate,Nat.add(2n+rest,coordinate)),True{},coordinate_room(rest,coordinate)) :
        {Cursor.step(~Rest,~next,Cursor.Level{Cursor.Axis{Nat.add(2n+rest,coordinate),2n+rest,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))},represent_rest(tail)},U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
          placed(~Cursor.Level<Rest>,~represent_level(~Rest,~represent_rest),Bool.pick(List<&2,CursorModel.Coordinate>,_,CursorModel.Coordinate{Nat.add(2n+rest,coordinate),1n+coordinate,stride} <> tail,
            CursorModel.Coordinate{Nat.add(2n+rest,coordinate),0n,stride} <> CursorModel.advance(tail)),offset) : Cursor.Level<Rest> & U32}
      %Equal.sym(Nat,Nat.sub(Nat.add(2n+rest,coordinate),1n+coordinate),1n+rest,decrement_remaining(rest,coordinate)) :
        {(Cursor.Level{Cursor.Axis{Nat.add(2n+rest,coordinate),1n+rest,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))},represent_rest(tail)},U32.add(U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset))),U32.from_nat(stride))) ==
          (Cursor.Level{Cursor.Axis{Nat.add(2n+rest,coordinate),_,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))},represent_rest(tail)},
            U32.from_nat(Nat.add(Nat.mul(1n+coordinate,stride),CursorModel.address(tail,offset)))) : Cursor.Level<Rest> & U32}
      %increment_address(coordinate,stride,CursorModel.address(tail,offset)) :
        {(Cursor.Level{Cursor.Axis{Nat.add(2n+rest,coordinate),1n+rest,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))},represent_rest(tail)},U32.add(U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset))),U32.from_nat(stride))) ==
          (Cursor.Level{Cursor.Axis{Nat.add(2n+rest,coordinate),1n+rest,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))},represent_rest(tail)},U32.from_nat(_)) : Cursor.Level<Rest> & U32}
      Equal.cong(U32,Cursor.Level<Rest> & U32,address => (Cursor.Level{Cursor.Axis{Nat.add(2n+rest,coordinate),1n+rest,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))},represent_rest(tail)},address),
        U32.add(U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset))),U32.from_nat(stride)),
        U32.from_nat(Nat.add(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)),stride)),WordArithmetic.native_add(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)),stride))

def level(~Rest: Data,~next: Rest -> U32 -> Rest & U32,~represent_rest: List<&2,CursorModel.Coordinate> -> Rest,
  ~next_ok: @+tail: List<&2,CursorModel.Coordinate> -> @+offset: Nat -> Valid(tail) ->
    {next(represent_rest(tail),U32.from_nat(CursorModel.address(tail,offset))) == placed(~Rest,~represent_rest,CursorModel.advance(tail),offset) : Rest & U32},
  +axes: List<&2,CursorModel.Coordinate>,+offset: Nat,valid: Valid(axes)) ->
  {Cursor.step(~Rest,~next,represent_level(~Rest,~represent_rest,axes),U32.from_nat(CursorModel.address(axes,offset))) ==
    placed(~Cursor.Level<Rest>,~represent_level(~Rest,~represent_rest),CursorModel.advance(axes),offset) : Cursor.Level<Rest> & U32}:
  match axes:
    case []: {==}
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      Shape.Both{+inside,tail_valid} = valid
      %Subtraction.restore(extent,coordinate,Bounds.strict_le(coordinate,extent,inside)) :
        {Cursor.step(~Rest,~next,represent_level(~Rest,~represent_rest,CursorModel.Coordinate{_,coordinate,stride} <> tail),U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
          placed(~Cursor.Level<Rest>,~represent_level(~Rest,~represent_rest),CursorModel.advance(CursorModel.Coordinate{_,coordinate,stride} <> tail),offset) : Cursor.Level<Rest> & U32}
      %Equal.sym(Cursor.Level<Rest>,represent_level(~Rest,~represent_rest,CursorModel.Coordinate{Nat.add(Nat.sub(extent,coordinate),coordinate),coordinate,stride} <> tail),
        Cursor.Level{Cursor.Axis{Nat.add(Nat.sub(extent,coordinate),coordinate),Nat.sub(extent,coordinate),U32.from_nat(stride),
          U32.from_nat(Nat.mul(Nat.sub(Nat.add(Nat.sub(extent,coordinate),coordinate),1n),stride))},represent_rest(tail)},
        head_level(~Rest,~represent_rest,Nat.sub(extent,coordinate),coordinate,stride,tail)) :
        {Cursor.step(~Rest,~next,_,U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
          placed(~Cursor.Level<Rest>,~represent_level(~Rest,~represent_rest),CursorModel.advance(CursorModel.Coordinate{Nat.add(Nat.sub(extent,coordinate),coordinate),coordinate,stride} <> tail),offset) : Cursor.Level<Rest> & U32}
      level_step(~Rest,~next,~represent_rest,Nat.sub(extent,coordinate),coordinate,stride,tail,offset,remaining_positive(extent,coordinate,inside),
        next_ok(tail,offset,tail_valid))

# A non-tail reference for the runtime carry; the tail-recursive loop is
# proved equal to it, and the reference is related to the coordinate model.
def prepend_pair(axis: Cursor.Axis,result: List<&2,Cursor.Axis> & U32) -> List<&2,Cursor.Axis> & U32:
  (axes,address) = result
  (axis <> axes,address)

def carry_reference(axes: List<&2,Cursor.Axis>,address: U32) -> List<&2,Cursor.Axis> & U32:
  match axes:
    case []: ([],address)
    case Cursor.Axis{extent,0n,stride,rewind} <> tail: (Cursor.Axis{extent,0n,stride,rewind} <> tail,address)
    case Cursor.Axis{+extent,1n,stride,+rewind} <> tail: prepend_pair(Cursor.Axis{extent,extent,stride,rewind},carry_reference(tail,(address - rewind : U32)))
    case Cursor.Axis{extent,1n+1n+remaining,+stride,rewind} <> tail: (Cursor.Axis{extent,1n+remaining,stride,rewind} <> tail,(address + stride : U32))

def restore_pair(reset: List<&2,Cursor.Axis>,result: List<&2,Cursor.Axis> & U32) -> List<&2,Cursor.Axis> & U32:
  (axes,address) = result
  (Cursor.restore(reset,axes),address)

def restore_prepend(+reset: List<&2,Cursor.Axis>,+axis: Cursor.Axis,result: List<&2,Cursor.Axis> & U32) ->
  {restore_pair(axis <> reset,result) == restore_pair(reset,prepend_pair(axis,result)) : List<&2,Cursor.Axis> & U32}:
  (axes,address) = result
  {==}

def restore_empty(result: List<&2,Cursor.Axis> & U32) -> {restore_pair([],result) == result : List<&2,Cursor.Axis> & U32}:
  (axes,address) = result
  {==}

def carry_matches(+axes: List<&2,Cursor.Axis>,+address: U32,+reset: List<&2,Cursor.Axis>) ->
  {Cursor.carry(axes,address,reset) == restore_pair(reset,carry_reference(axes,address)) : List<&2,Cursor.Axis> & U32}:
  match axes:
    case []: {==}
    case Cursor.Axis{extent,0n,stride,rewind} <> tail: {==}
    case Cursor.Axis{+extent,1n,+stride,+rewind} <> +tail:
      Equal.trans(List<&2,Cursor.Axis> & U32,Cursor.carry(tail,(address - rewind : U32),Cursor.Axis{extent,extent,stride,rewind} <> reset),
        restore_pair(Cursor.Axis{extent,extent,stride,rewind} <> reset,carry_reference(tail,(address - rewind : U32))),
        restore_pair(reset,prepend_pair(Cursor.Axis{extent,extent,stride,rewind},carry_reference(tail,(address - rewind : U32)))),
        carry_matches(tail,(address - rewind : U32),Cursor.Axis{extent,extent,stride,rewind} <> reset),
        restore_prepend(reset,Cursor.Axis{extent,extent,stride,rewind},carry_reference(tail,(address - rewind : U32))))
    case Cursor.Axis{extent,1n+1n+remaining,stride,rewind} <> tail: {==}

def outer_step(+remaining: Nat,+coordinate: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>,+offset: Nat,
  positive: {Nat.is_lt(0n,remaining) == True{} : Bool},
  next: {carry_reference(represent_outer(tail),U32.from_nat(CursorModel.address(tail,offset))) == placed(~List<&2,Cursor.Axis>,~represent_outer,CursorModel.advance(tail),offset) : List<&2,Cursor.Axis> & U32}) ->
  {carry_reference(Cursor.Axis{Nat.add(remaining,coordinate),remaining,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(remaining,coordinate),1n),stride))} <> represent_outer(tail),
      U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
    placed(~List<&2,Cursor.Axis>,~represent_outer,CursorModel.advance(CursorModel.Coordinate{Nat.add(remaining,coordinate),coordinate,stride} <> tail),offset) : List<&2,Cursor.Axis> & U32}:
  match remaining:
    case 0n:
      Empty.absurd({carry_reference(Cursor.Axis{coordinate,0n,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(coordinate,1n),stride))} <> represent_outer(tail),
          U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
        placed(~List<&2,Cursor.Axis>,~represent_outer,CursorModel.advance(CursorModel.Coordinate{coordinate,coordinate,stride} <> tail),offset) : List<&2,Cursor.Axis> & U32},WordEncoding.false_true(positive))
    case 1n:
      %Equal.sym(Bool,Nat.is_lt(1n+coordinate,1n+coordinate),False{},less_reflexive(1n+coordinate)) :
        {carry_reference(Cursor.Axis{1n+coordinate,1n,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(coordinate,0n),stride))} <> represent_outer(tail),
            U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
          placed(~List<&2,Cursor.Axis>,~represent_outer,Bool.pick(List<&2,CursorModel.Coordinate>,_,CursorModel.Coordinate{1n+coordinate,1n+coordinate,stride} <> tail,
            CursorModel.Coordinate{1n+coordinate,0n,stride} <> CursorModel.advance(tail)),offset) : List<&2,Cursor.Axis> & U32}
      %Equal.sym(List<&2,Cursor.Axis>,represent_outer(CursorModel.Coordinate{1n+coordinate,0n,stride} <> CursorModel.advance(tail)),
        Cursor.Axis{1n+coordinate,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(coordinate,0n),stride))} <> represent_outer(CursorModel.advance(tail)),
        reset_outer(1n+coordinate,stride,CursorModel.advance(tail))) :
        {carry_reference(Cursor.Axis{1n+coordinate,1n,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(coordinate,0n),stride))} <> represent_outer(tail),
            U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) == (_,U32.from_nat(CursorModel.address(CursorModel.advance(tail),offset))) : List<&2,Cursor.Axis> & U32}
      %Equal.sym(Nat,Nat.sub(coordinate,0n),coordinate,Order.subtract_zero(coordinate)) :
        {carry_reference(Cursor.Axis{1n+coordinate,1n,U32.from_nat(stride),U32.from_nat(Nat.mul(_,stride))} <> represent_outer(tail),
            U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
          (Cursor.Axis{1n+coordinate,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(_,stride))} <> represent_outer(CursorModel.advance(tail)),
            U32.from_nat(CursorModel.address(CursorModel.advance(tail),offset))) : List<&2,Cursor.Axis> & U32}
      %Equal.sym(U32,U32.sub(U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset))),U32.from_nat(Nat.mul(coordinate,stride))),
        U32.from_nat(CursorModel.address(tail,offset)),subtract_front(Nat.mul(coordinate,stride),CursorModel.address(tail,offset))) :
        {prepend_pair(Cursor.Axis{1n+coordinate,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(coordinate,stride))},carry_reference(represent_outer(tail),_)) ==
          (Cursor.Axis{1n+coordinate,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(coordinate,stride))} <> represent_outer(CursorModel.advance(tail)),
            U32.from_nat(CursorModel.address(CursorModel.advance(tail),offset))) : List<&2,Cursor.Axis> & U32}
      %Equal.sym(List<&2,Cursor.Axis> & U32,carry_reference(represent_outer(tail),U32.from_nat(CursorModel.address(tail,offset))),placed(~List<&2,Cursor.Axis>,~represent_outer,CursorModel.advance(tail),offset),next) :
        {prepend_pair(Cursor.Axis{1n+coordinate,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(coordinate,stride))},_) ==
          (Cursor.Axis{1n+coordinate,1n+coordinate,U32.from_nat(stride),U32.from_nat(Nat.mul(coordinate,stride))} <> represent_outer(CursorModel.advance(tail)),
            U32.from_nat(CursorModel.address(CursorModel.advance(tail),offset))) : List<&2,Cursor.Axis> & U32}
      {==}
    case 2n+ +rest:
      %Equal.sym(Bool,Nat.is_lt(1n+coordinate,Nat.add(2n+rest,coordinate)),True{},coordinate_room(rest,coordinate)) :
        {carry_reference(Cursor.Axis{Nat.add(2n+rest,coordinate),2n+rest,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))} <> represent_outer(tail),
            U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
          placed(~List<&2,Cursor.Axis>,~represent_outer,Bool.pick(List<&2,CursorModel.Coordinate>,_,CursorModel.Coordinate{Nat.add(2n+rest,coordinate),1n+coordinate,stride} <> tail,
            CursorModel.Coordinate{Nat.add(2n+rest,coordinate),0n,stride} <> CursorModel.advance(tail)),offset) : List<&2,Cursor.Axis> & U32}
      %Equal.sym(Nat,Nat.sub(Nat.add(2n+rest,coordinate),1n+coordinate),1n+rest,decrement_remaining(rest,coordinate)) :
        {(Cursor.Axis{Nat.add(2n+rest,coordinate),1n+rest,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))} <> represent_outer(tail),
            U32.add(U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset))),U32.from_nat(stride))) ==
          (Cursor.Axis{Nat.add(2n+rest,coordinate),_,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))} <> represent_outer(tail),
            U32.from_nat(Nat.add(Nat.mul(1n+coordinate,stride),CursorModel.address(tail,offset)))) : List<&2,Cursor.Axis> & U32}
      %increment_address(coordinate,stride,CursorModel.address(tail,offset)) :
        {(Cursor.Axis{Nat.add(2n+rest,coordinate),1n+rest,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))} <> represent_outer(tail),
            U32.add(U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset))),U32.from_nat(stride))) ==
          (Cursor.Axis{Nat.add(2n+rest,coordinate),1n+rest,U32.from_nat(stride),U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))} <> represent_outer(tail),U32.from_nat(_)) : List<&2,Cursor.Axis> & U32}
      Equal.cong(U32,List<&2,Cursor.Axis> & U32,address => (Cursor.Axis{Nat.add(2n+rest,coordinate),1n+rest,U32.from_nat(stride),
          U32.from_nat(Nat.mul(Nat.sub(Nat.add(2n+rest,coordinate),1n),stride))} <> represent_outer(tail),address),
        U32.add(U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset))),U32.from_nat(stride)),
        U32.from_nat(Nat.add(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)),stride)),WordArithmetic.native_add(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)),stride))

def reference(+axes: List<&2,CursorModel.Coordinate>,+offset: Nat,valid: Valid(axes)) ->
  {carry_reference(represent_outer(axes),U32.from_nat(CursorModel.address(axes,offset))) == placed(~List<&2,Cursor.Axis>,~represent_outer,CursorModel.advance(axes),offset) : List<&2,Cursor.Axis> & U32}:
  match axes:
    case []: {==}
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      Shape.Both{+inside,tail_valid} = valid
      %Subtraction.restore(extent,coordinate,Bounds.strict_le(coordinate,extent,inside)) :
        {carry_reference(represent_outer(CursorModel.Coordinate{_,coordinate,stride} <> tail),U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
          placed(~List<&2,Cursor.Axis>,~represent_outer,CursorModel.advance(CursorModel.Coordinate{_,coordinate,stride} <> tail),offset) : List<&2,Cursor.Axis> & U32}
      %Equal.sym(List<&2,Cursor.Axis>,represent_outer(CursorModel.Coordinate{Nat.add(Nat.sub(extent,coordinate),coordinate),coordinate,stride} <> tail),
        Cursor.Axis{Nat.add(Nat.sub(extent,coordinate),coordinate),Nat.sub(extent,coordinate),U32.from_nat(stride),
          U32.from_nat(Nat.mul(Nat.sub(Nat.add(Nat.sub(extent,coordinate),coordinate),1n),stride))} <> represent_outer(tail),
        head_outer(Nat.sub(extent,coordinate),coordinate,stride,tail)) :
        {carry_reference(_,U32.from_nat(Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset)))) ==
          placed(~List<&2,Cursor.Axis>,~represent_outer,CursorModel.advance(CursorModel.Coordinate{Nat.add(Nat.sub(extent,coordinate),coordinate),coordinate,stride} <> tail),offset) : List<&2,Cursor.Axis> & U32}
      outer_step(Nat.sub(extent,coordinate),coordinate,stride,tail,offset,remaining_positive(extent,coordinate,inside),reference(tail,offset,tail_valid))

def outer(+axes: List<&2,CursorModel.Coordinate>,+offset: Nat,valid: Valid(axes)) ->
  {Cursor.advance_outer(represent_outer(axes),U32.from_nat(CursorModel.address(axes,offset))) == placed(~List<&2,Cursor.Axis>,~represent_outer,CursorModel.advance(axes),offset) : List<&2,Cursor.Axis> & U32}:
  Equal.trans(List<&2,Cursor.Axis> & U32,Cursor.carry(represent_outer(axes),U32.from_nat(CursorModel.address(axes,offset)),[]),
    restore_pair([],carry_reference(represent_outer(axes),U32.from_nat(CursorModel.address(axes,offset)))),
    placed(~List<&2,Cursor.Axis>,~represent_outer,CursorModel.advance(axes),offset),
    carry_matches(represent_outer(axes),U32.from_nat(CursorModel.address(axes,offset)),[]),
    Equal.trans(List<&2,Cursor.Axis> & U32,restore_pair([],carry_reference(represent_outer(axes),U32.from_nat(CursorModel.address(axes,offset)))),
      carry_reference(represent_outer(axes),U32.from_nat(CursorModel.address(axes,offset))),
      placed(~List<&2,Cursor.Axis>,~represent_outer,CursorModel.advance(axes),offset),
      restore_empty(carry_reference(represent_outer(axes),U32.from_nat(CursorModel.address(axes,offset)))),reference(axes,offset,valid)))

def split_level_matches(~Rest: Data,~split_rest: List<&2,Cursor.Axis> -> Rest,~represent_rest: List<&2,CursorModel.Coordinate> -> Rest,
  ~rest_ok: @+tail: List<&2,CursorModel.Coordinate> -> {split_rest(represent_outer(tail)) == represent_rest(tail) : Rest},+axes: List<&2,CursorModel.Coordinate>) ->
  {Cursor.split_level(~Rest,~split_rest,represent_outer(axes)) == represent_level(~Rest,~represent_rest,axes) : Cursor.Level<Rest>}:
  match axes:
    case []: Equal.cong(Rest,Cursor.Level<Rest>,rest => Cursor.Level{Cursor.absent(),rest},split_rest([]),represent_rest([]),rest_ok([]))
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      Equal.cong(Rest,Cursor.Level<Rest>,rest => Cursor.Level{Cursor.Axis{extent,Nat.sub(extent,coordinate),U32.from_nat(stride),
          U32.from_nat(Nat.mul(Nat.sub(extent,1n),stride))},rest},split_rest(represent_outer(tail)),represent_rest(tail),rest_ok(tail))

def depth1(+axes: List<&2,CursorModel.Coordinate>,+offset: Nat,valid: Valid(axes)) ->
  {Cursor.step(~Cursor.Depth0(),~Cursor.advance_outer,represent_level(~Cursor.Depth0(),~represent_outer)(axes),U32.from_nat(CursorModel.address(axes,offset))) ==
    placed(~Cursor.Depth1(),~represent_level(~Cursor.Depth0(),~represent_outer),CursorModel.advance(axes),offset) : Cursor.Depth1() & U32}:
  level(~Cursor.Depth0(),~Cursor.advance_outer,~represent_outer,~outer,axes,offset,valid)

def depth2(+axes: List<&2,CursorModel.Coordinate>,+offset: Nat,valid: Valid(axes)) ->
  {Cursor.step(~Cursor.Depth1(),~Cursor.step(~Cursor.Depth0(),~Cursor.advance_outer),represent_level(~Cursor.Depth1(),~represent_level(~Cursor.Depth0(),~represent_outer))(axes),U32.from_nat(CursorModel.address(axes,offset))) ==
    placed(~Cursor.Levels(),~represent_level(~Cursor.Depth1(),~represent_level(~Cursor.Depth0(),~represent_outer)),CursorModel.advance(axes),offset) : Cursor.Levels() & U32}:
  level(~Cursor.Depth1(),~Cursor.step(~Cursor.Depth0(),~Cursor.advance_outer),~represent_level(~Cursor.Depth0(),~represent_outer),~depth1,axes,offset,valid)

def advance(+axes: List<&2,CursorModel.Coordinate>,+offset: Nat,valid: Valid(axes)) ->
  {Cursor.advance(represent_axes(axes),U32.from_nat(CursorModel.address(axes,offset))) == represent(CursorModel.advance(axes),offset) : Cursor.Cursor}:
  Equal.cong(Cursor.Levels() & U32,Cursor.Cursor,result => Cursor.moved(result),
    Cursor.advance_levels(represent_axes(axes),U32.from_nat(CursorModel.address(axes,offset))),
    placed(~Cursor.Levels(),~represent_level(~Cursor.Depth1(),~represent_level(~Cursor.Depth0(),~represent_outer)),CursorModel.advance(axes),offset),depth2(axes,offset,valid))

def split0(+axes: List<&2,CursorModel.Coordinate>) -> {Cursor.outer(represent_outer(axes)) == represent_outer(axes) : List<&2,Cursor.Axis>}:
  {==}

def split1(+axes: List<&2,CursorModel.Coordinate>) -> {Cursor.split_level(~Cursor.Depth0(),~Cursor.outer)(represent_outer(axes)) == represent_level(~Cursor.Depth0(),~represent_outer)(axes) : Cursor.Depth1()}:
  split_level_matches(~Cursor.Depth0(),~Cursor.outer,~represent_outer,~split0,axes)

def split2(+axes: List<&2,CursorModel.Coordinate>) -> {Cursor.split_level(~Cursor.Depth1(),~Cursor.split_level(~Cursor.Depth0(),~Cursor.outer))(represent_outer(axes)) == represent_level(~Cursor.Depth1(),~represent_level(~Cursor.Depth0(),~represent_outer))(axes) : Cursor.Levels()}:
  split_level_matches(~Cursor.Depth1(),~Cursor.split_level(~Cursor.Depth0(),~Cursor.outer),~represent_level(~Cursor.Depth0(),~represent_outer),~split1,axes)

def selected_valid(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>,
  inside: {Nat.is_lt(coordinate,extent) == True{} : Bool},guard: {Nat.is_lt(1n+coordinate,extent) == selected : Bool},
  tail_valid: Valid(tail),next_valid: Valid(CursorModel.advance(tail))) ->
  Valid(Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,1n+coordinate,stride} <> tail,
    CursorModel.Coordinate{extent,0n,stride} <> CursorModel.advance(tail))):
  match selected:
    case True{}: Shape.Both{guard,tail_valid}
    case False{}: Shape.Both{Order.le_lt_trans(0n,coordinate,extent,Order.zero_le(coordinate),inside),next_valid}

def advance_valid(+axes: List<&2,CursorModel.Coordinate>,valid: Valid(axes)) -> Valid(CursorModel.advance(axes)):
  match axes:
    case []: Unit{}
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      Shape.Both{inside,+tail_valid} = valid
      selected_valid(Nat.is_lt(1n+coordinate,extent),extent,coordinate,stride,tail,inside,{==},tail_valid,advance_valid(tail,tail_valid))

def rewind(+extent: U32,+stride: U32,active: {Nat.is_lt(1n,U32.to_nat(extent)) == True{} : Bool}) ->
  {U32.mul(U32.sub(extent,1),stride) == U32.from_nat(Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride))) : U32}:
  %FullRange.subtract(extent,1,extent,Bounds.strict_le(1n,U32.to_nat(extent),active),Order.sub_le(U32.to_nat(extent),1n)) :
    {U32.mul(U32.sub(extent,1),stride) == U32.from_nat(Nat.mul(_,U32.to_nat(stride))) : U32}
  Multiplication.native_multiply(U32.sub(extent,1),stride)

def include_axis(+selected: Bool,+extent: U32,+stride: U32,+reversed: List<&2,CursorModel.Coordinate>,
  guard: {Nat.is_lt(1n,U32.to_nat(extent)) == selected : Bool}) ->
  {Cursor.include_axis(selected,extent,stride,represent_outer(reversed)) ==
    represent_outer(Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed)) : List<&2,Cursor.Axis>}:
  match selected:
    case False{}: {==}
    case True{}:
      %Equal.sym(List<&2,Cursor.Axis>,represent_outer(CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed),
        Cursor.Axis{U32.to_nat(extent),U32.to_nat(extent),U32.from_nat(U32.to_nat(stride)),U32.from_nat(Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride)))} <> represent_outer(reversed),
        reset_outer(U32.to_nat(extent),U32.to_nat(stride),reversed)) : {Cursor.include_axis(True{},extent,stride,represent_outer(reversed)) == _ : List<&2,Cursor.Axis>}
      %Equal.sym(U32,U32.from_nat(U32.to_nat(stride)),stride,Arithmetic.native_roundtrip(stride)) :
        {Cursor.include_axis(True{},extent,stride,represent_outer(reversed)) ==
          Cursor.Axis{U32.to_nat(extent),U32.to_nat(extent),_,U32.from_nat(Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride)))} <> represent_outer(reversed) : List<&2,Cursor.Axis>}
      %rewind(extent,stride,guard) :
        {Cursor.include_axis(True{},extent,stride,represent_outer(reversed)) ==
          Cursor.Axis{U32.to_nat(extent),U32.to_nat(extent),stride,_} <> represent_outer(reversed) : List<&2,Cursor.Axis>}
      {==}

def prepared_axes(+source: List<&2,Views.Axis>,+reversed: List<&2,CursorModel.Coordinate>) ->
  {Cursor.axes(source,represent_outer(reversed)) == represent_outer(CursorModel.prepare_axes(source,reversed)) : List<&2,Cursor.Axis>}:
  match source:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      %Equal.sym(Bool,U32.is_lt(1,extent),Nat.is_lt(1n,U32.to_nat(extent)),Comparison.lt(1,extent)) :
        {Cursor.axes(tail,Cursor.include_axis(_,extent,stride,represent_outer(reversed))) == represent_outer(CursorModel.prepare_axes(Views.Axis{extent,stride} <> tail,reversed)) : List<&2,Cursor.Axis>}
      %Equal.sym(List<&2,Cursor.Axis>,Cursor.include_axis(Nat.is_lt(1n,U32.to_nat(extent)),extent,stride,represent_outer(reversed)),
        represent_outer(Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed)),
        include_axis(Nat.is_lt(1n,U32.to_nat(extent)),extent,stride,reversed,{==})) :
        {Cursor.axes(tail,_) == represent_outer(CursorModel.prepare_axes(Views.Axis{extent,stride} <> tail,reversed)) : List<&2,Cursor.Axis>}
      prepared_axes(tail,Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed))

def selected_initial_valid(+selected: Bool,+extent: Nat,+stride: Nat,+reversed: List<&2,CursorModel.Coordinate>,
  guard: {Nat.is_lt(1n,extent) == selected : Bool},valid: Valid(reversed)) ->
  Valid(Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,0n,stride} <> reversed,reversed)):
  match selected:
    case False{}: valid
    case True{}: Shape.Both{Order.le_lt_trans(0n,1n,extent,{==},guard),valid}

def prepared_valid(+source: List<&2,Views.Axis>,+reversed: List<&2,CursorModel.Coordinate>,valid: Valid(reversed)) -> Valid(CursorModel.prepare_axes(source,reversed)):
  match source:
    case []: valid
    case Views.Axis{+extent,+stride} <> tail:
      prepared_valid(tail,Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed),
        selected_initial_valid(Nat.is_lt(1n,U32.to_nat(extent)),U32.to_nat(extent),U32.to_nat(stride),reversed,{==},valid))

def initial_selected_address(+selected: Bool,+extent: Nat,+stride: Nat,+reversed: List<&2,CursorModel.Coordinate>,+offset: Nat) ->
  {CursorModel.address(Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,0n,stride} <> reversed,reversed),offset) == CursorModel.address(reversed,offset) : Nat}:
  match selected:
    case False{}: {==}
    case True{}: {==}

def prepared_address(+source: List<&2,Views.Axis>,+reversed: List<&2,CursorModel.Coordinate>,+offset: Nat) ->
  {CursorModel.address(CursorModel.prepare_axes(source,reversed),offset) == CursorModel.address(reversed,offset) : Nat}:
  match source:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      Equal.trans(Nat,CursorModel.address(CursorModel.prepare_axes(Views.Axis{extent,stride} <> tail,reversed),offset),
        CursorModel.address(Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed),offset),
        CursorModel.address(reversed,offset),
        prepared_address(tail,Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed),offset),
        initial_selected_address(Nat.is_lt(1n,U32.to_nat(extent)),U32.to_nat(extent),U32.to_nat(stride),reversed,offset))

def prepared(+source: List<&2,Views.Axis>,+offset: U32) ->
  {Cursor.prepare(Views.View{source,offset}) == represent(CursorModel.prepare_axes(source,[]),U32.to_nat(offset)) : Cursor.Cursor}:
  %Equal.sym(Nat,CursorModel.address(CursorModel.prepare_axes(source,[]),U32.to_nat(offset)),U32.to_nat(offset),prepared_address(source,[],U32.to_nat(offset))) :
    {Cursor.prepare(Views.View{source,offset}) == Cursor.Cursor{represent_axes(CursorModel.prepare_axes(source,[])),U32.from_nat(_)} : Cursor.Cursor}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(offset)),offset,Arithmetic.native_roundtrip(offset)) :
    {Cursor.prepare(Views.View{source,offset}) == Cursor.Cursor{represent_axes(CursorModel.prepare_axes(source,[])),_} : Cursor.Cursor}
  Equal.cong(Cursor.Levels(),Cursor.Cursor,axes => Cursor.Cursor{axes,offset},Cursor.split(Cursor.axes(source,[])),represent_axes(CursorModel.prepare_axes(source,[])),
    Equal.trans(Cursor.Levels(),Cursor.split(Cursor.axes(source,[])),Cursor.split(represent_outer(CursorModel.prepare_axes(source,[]))),represent_axes(CursorModel.prepare_axes(source,[])),
      Equal.cong(List<&2,Cursor.Axis>,Cursor.Levels(),axes => Cursor.split(axes),Cursor.axes(source,[]),represent_outer(CursorModel.prepare_axes(source,[])),prepared_axes(source,[])),
      split2(CursorModel.prepare_axes(source,[]))))

# This type is confined to erased semantic proofs. Native cursor metadata has
# neither a Valid tree nor an increasing-coordinate representation.
type Coordinates is Data:
  Coordinates{axes: List<&2,CursorModel.Coordinate>,offset: Nat,valid: Valid(axes)}

def pack(coordinates: Coordinates) -> Cursor.Cursor:
  Coordinates{axes,offset,valid} = coordinates
  represent(axes,offset)

def next(coordinates: Coordinates) -> Coordinates:
  Coordinates{+axes,offset,valid} = coordinates
  Coordinates{CursorModel.advance(axes),offset,advance_valid(axes,valid)}

def initial(view: Views.View) -> Coordinates:
  Views.View{+axes,offset} = view
  Coordinates{CursorModel.prepare_axes(axes,[]),U32.to_nat(offset),prepared_valid(axes,[],Unit{})}

def initial_matches(+view: Views.View) -> {Cursor.prepare(view) == pack(initial(view)) : Cursor.Cursor}:
  Views.View{axes,offset} = view
  prepared(axes,offset)
