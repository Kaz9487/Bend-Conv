# The successful metadata address guard denotes an exact, in-bounds Nat sum.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ./checked_word_refinement.bend as Checked
import ./u32_full_range.bend as FullRange
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./word_encoding.bend as WordEncoding
import ./word_subtraction.bend as Subtraction
import ./storage_shape.bend as Shape
import ./boolean_logic.bend as Boolean
import ./natural_addition.bend as Addition

def less_zero_false(+value: Nat) -> {Nat.is_lt(value,0n) == False{} : Bool}:
  match value:
    case 0n: {==}
    case 1n+rest: {==}

def less_successor(+offset: Nat,+last: Nat,inside: {Nat.is_lt(offset,1n+last) == True{} : Bool}) ->
  {Nat.is_le(offset,last) == True{} : Bool}:
  match offset last:
    case 0n last: Order.zero_le(last)
    case 1n+ +offset 0n: Empty.absurd({Nat.is_le(1n+offset,0n) == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(offset,0n),True{},Equal.sym(Bool,Nat.is_lt(offset,0n),False{},less_zero_false(offset)),inside)))
    case 1n+offset 1n+last: less_successor(offset,last,inside)

def before_last(+offset: Nat,+length: Nat,inside: {Nat.is_lt(offset,length) == True{} : Bool}) ->
  {Nat.is_le(offset,Nat.sub(length,1n)) == True{} : Bool}:
  match length:
    case 0n:
      Empty.absurd({Nat.is_le(offset,0n) == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(offset,0n),True{},Equal.sym(Bool,Nat.is_lt(offset,0n),False{},less_zero_false(offset)),inside)))
    case 1n+ +last:
      %Equal.sym(Nat,Nat.sub(last,0n),last,Order.subtract_zero(last)) : {Nat.is_le(offset,_) == True{} : Bool}
      less_successor(offset,last,inside)

def positive_one(+length: Nat,positive: {Nat.is_lt(0n,length) == True{} : Bool}) -> {Nat.is_le(1n,length) == True{} : Bool}:
  Order.strict_follow(0n,length,positive)

def last_below(+length: Nat,positive: {Nat.is_lt(0n,length) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(length,1n),length) == True{} : Bool}:
  %Order.positive_has_last(length,positive) : {Nat.is_lt(Nat.sub(length,1n),_) == True{} : Bool}
  Order.lt_succ(Nat.sub(length,1n),Nat.sub(length,1n),Order.reflexive(Nat.sub(length,1n)))

def sum_within(+offset: Nat,+increment: Nat,+last: Nat,
  offset_inside: {Nat.is_le(offset,last) == True{} : Bool},
  room: {Nat.is_le(increment,Nat.sub(last,offset)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(offset,increment),last) == True{} : Bool}:
  %Subtraction.restore(last,offset,offset_inside) : {Nat.is_le(Nat.add(offset,increment),_) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(offset,increment),Nat.add(increment,offset),Order.commutative(offset,increment)) : {Nat.is_le(_,Nat.add(Nat.sub(last,offset),offset)) == True{} : Bool}
  Order.add_le(increment,Nat.sub(last,offset),offset,room)

def room_exact(+offset: U32,+length: U32,+inside: {Nat.is_lt(U32.to_nat(offset),U32.to_nat(length)) == True{} : Bool}) ->
  {U32.to_nat((length - 1 - offset : U32)) == Nat.sub(Nat.sub(U32.to_nat(length),1n),U32.to_nat(offset)) : Nat}:
  +positive = Order.le_lt_trans(0n,U32.to_nat(offset),U32.to_nat(length),Order.zero_le(U32.to_nat(offset)),inside)
  +last_exact = FullRange.subtract(length,1,length,positive_one(U32.to_nat(length),positive),Order.sub_le(U32.to_nat(length),1n))
  %last_exact : {U32.to_nat((length - 1 - offset : U32)) == Nat.sub(_,U32.to_nat(offset)) : Nat}
  FullRange.subtract((length - 1 : U32),offset,(length - 1 : U32),{
    %Equal.sym(Nat,U32.to_nat((length - 1 : U32)),Nat.sub(U32.to_nat(length),1n),last_exact) : {Nat.is_le(U32.to_nat(offset),_) == True{} : Bool}
    before_last(U32.to_nat(offset),U32.to_nat(length),inside) : {Nat.is_le(U32.to_nat(offset),U32.to_nat((length - 1 : U32))) == True{} : Bool}},Order.sub_le(U32.to_nat((length - 1 : U32)),U32.to_nat(offset)))

def addition_inside(+offset: U32,+increment: U32,+length: U32,
  inside: {U32.is_lt(offset,length) == True{} : Bool},
  room: {U32.is_le(increment,(length - 1 - offset : U32)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(increment)),U32.to_nat(length)) == True{} : Bool}:
  +inside_nat = Bounds.as_nat_lt(offset,length,inside)
  Order.le_lt_trans(Nat.add(U32.to_nat(offset),U32.to_nat(increment)),Nat.sub(U32.to_nat(length),1n),U32.to_nat(length),
    sum_within(U32.to_nat(offset),U32.to_nat(increment),Nat.sub(U32.to_nat(length),1n),before_last(U32.to_nat(offset),U32.to_nat(length),inside_nat),{
      %room_exact(offset,length,inside_nat) : {Nat.is_le(U32.to_nat(increment),_) == True{} : Bool}
      Bounds.as_nat_le(increment,(length - 1 - offset : U32),room) : {Nat.is_le(U32.to_nat(increment),Nat.sub(Nat.sub(U32.to_nat(length),1n),U32.to_nat(offset))) == True{} : Bool}}),
    last_below(U32.to_nat(length),Order.le_lt_trans(0n,U32.to_nat(offset),U32.to_nat(length),Order.zero_le(U32.to_nat(offset)),inside_nat)))

law Address:
  for result: Maybe<&2,U32>
  for expected: Nat
  for length: U32
  Data

def Address(result,expected,length):
  match result:
    case None{}: Unit
    case Some{value}: Shape.Both<{expected == U32.to_nat(value) : Nat},{Nat.is_lt(expected,U32.to_nat(length)) == True{} : Bool}>

def selected_address(+valid: Bool,+offset: U32,+increment: U32,+length: U32,
  inside: {U32.is_lt(offset,length) == True{} : Bool},guard: {U32.is_le(increment,(length - 1 - offset : U32)) == valid : Bool}) ->
  Address(Views.address_when(valid,offset,increment),Nat.add(U32.to_nat(offset),U32.to_nat(increment)),length):
  match valid:
    case False{}: Unit{}
    case True{}:
      +bounded = addition_inside(offset,increment,length,inside,guard)
      Shape.Both{Equal.sym(Nat,U32.to_nat((offset + increment : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(increment)),
        FullRange.add(offset,increment,length,Bounds.strict_le(Nat.add(U32.to_nat(offset),U32.to_nat(increment)),U32.to_nat(length),bounded))),bounded}

def product_address(+product: Maybe<&2,U32>,+offset: U32,+coordinate: U32,+stride: U32,+length: U32,
  inside: {U32.is_lt(offset,length) == True{} : Bool},computed: {Shapes.checked_multiply(coordinate,stride) == product : Maybe<&2,U32>}) ->
  Address(Views.advance_product(product,offset,(length - 1 - offset : U32)),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))),length):
  match product:
    case None{}: Unit{}
    case Some{+increment}:
      %Equal.sym(Nat,Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride)),U32.to_nat(increment),Checked.multiply_sound(coordinate,stride,increment,computed)) :
        Address(Views.advance_product(Some{increment},offset,(length - 1 - offset : U32)),Nat.add(U32.to_nat(offset),_),length)
      selected_address(U32.is_le(increment,(length - 1 - offset : U32)),offset,increment,length,inside,{==})

def selected_advance(+valid: Bool,+offset: U32,+coordinate: U32,+stride: U32,+length: U32,
  guard: {U32.is_lt(offset,length) == valid : Bool}) ->
  Address(Views.advance_inside(valid,offset,coordinate,stride,length),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))),length):
  match valid:
    case False{}: Unit{}
    case True{}: product_address(Shapes.checked_multiply(coordinate,stride),offset,coordinate,stride,length,guard,{==})

def advance(+offset: U32,+coordinate: U32,+stride: U32,+length: U32) ->
  Address(Views.advance(offset,coordinate,stride,length),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))),length):
  selected_advance(U32.is_lt(offset,length),offset,coordinate,stride,length,{==})

law Exact:
  for result: Maybe<&2,U32>
  for expected: Nat
  Data

def Exact(result,expected):
  match result:
    case None{}: Unit
    case Some{value}: {expected == U32.to_nat(value) : Nat}

def exact(+result: Maybe<&2,U32>,expected: Nat,length: U32,bound: Address(result,expected,length)) -> Exact(result,expected):
  match result:
    case None{}: Unit{}
    case Some{value}:
      Shape.Both{equation,inside} = bound
      equation

def guarded_advance(+valid: Bool,+offset: U32,+coordinate: U32,+stride: U32,+extent: U32,+length: U32,
  guard: {Bool.and(U32.is_lt(coordinate,extent),U32.is_lt(offset,length)) == valid : Bool}) ->
  Address(Views.advance_inside(valid,offset,coordinate,stride,length),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))),length):
  match valid:
    case False{}: Unit{}
    case True{}: product_address(Shapes.checked_multiply(coordinate,stride),offset,coordinate,stride,length,
      Boolean.right(U32.is_lt(coordinate,extent),U32.is_lt(offset,length),guard),{==})

def finished(+valid: Bool,+offset: U32,+length: U32,guard: {U32.is_lt(offset,length) == valid : Bool}) ->
  Address(Views.finish_address(valid,offset),U32.to_nat(offset),length):
  match valid:
    case False{}: Unit{}
    case True{}: Shape.Both{{==},Bounds.as_nat_lt(offset,length,guard)}

# Natural dot product, in source-axis order. Mismatched ranks are irrelevant:
# locate rejects them, and Address makes no assertion for a rejected result.
def location(axes: List<&2,Views.Axis>,coordinates: List<&2,U32>,offset: Nat) -> Nat:
  match axes coordinates:
    case Views.Axis{extent,stride} <> tail coordinate <> rest:
      location(tail,rest,Nat.add(offset,Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))))
    case axes coordinates: offset

def locate(+axes: List<&2,Views.Axis>,+coordinates: List<&2,U32>,+address: Maybe<&2,U32>,+expected: Nat,+length: U32,
  previous: Exact(address,expected)) -> Address(Views.locate(axes,coordinates,address,length),location(axes,coordinates,expected),length):
  match axes coordinates address:
    case Nil{} Nil{} Some{+offset}:
      %Equal.sym(Nat,expected,U32.to_nat(offset),previous) : Address(Views.locate([],[],Some{offset},length),_,length)
      finished(U32.is_lt(offset,length),offset,length,{==})
    case Views.Axis{+extent,+stride} <> +tail +coordinate <> +rest Some{+offset}:
      %Equal.sym(Nat,expected,U32.to_nat(offset),previous) :
        Address(Views.locate(Views.Axis{extent,stride} <> tail,coordinate <> rest,Some{offset},length),location(Views.Axis{extent,stride} <> tail,coordinate <> rest,_),length)
      locate(tail,rest,Views.advance_inside(Bool.and(U32.is_lt(coordinate,extent),U32.is_lt(offset,length)),offset,coordinate,stride,length),
        Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))),length,
        exact(Views.advance_inside(Bool.and(U32.is_lt(coordinate,extent),U32.is_lt(offset,length)),offset,coordinate,stride,length),
          Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))),length,
          guarded_advance(Bool.and(U32.is_lt(coordinate,extent),U32.is_lt(offset,length)),offset,coordinate,stride,extent,length,{==})))
    case Nil{} Nil{} None{}: Unit{}
    case Nil{} Con{coordinate,rest} None{}: Unit{}
    case Nil{} Con{coordinate,rest} Some{offset}: Unit{}
    case Con{Views.Axis{extent,stride},tail} Nil{} None{}: Unit{}
    case Con{Views.Axis{extent,stride},tail} Nil{} Some{offset}: Unit{}
    case Con{Views.Axis{extent,stride},tail} Con{coordinate,rest} None{}: Unit{}

def address(+axes: List<&2,Views.Axis>,+coordinates: List<&2,U32>,+offset: U32,+length: U32) ->
  Address(Views.address(Views.View{axes,offset},coordinates,length),location(axes,coordinates,U32.to_nat(offset)),length):
  locate(axes,coordinates,Some{offset},U32.to_nat(offset),length,{==})

def found_inside(+result: Maybe<&2,U32>,+expected: Nat,+length: U32,
  accepted: {Views.found(result) == True{} : Bool},bound: Address(result,expected,length)) ->
  {Nat.is_lt(expected,U32.to_nat(length)) == True{} : Bool}:
  match result:
    case None{}: Empty.absurd({Nat.is_lt(expected,U32.to_nat(length)) == True{} : Bool},WordEncoding.false_true(accepted))
    case Some{value}:
      Shape.Both{equation,inside} = bound
      inside

def last_address(+axes: List<&2,Views.Axis>,+offset: U32,+length: U32,
  fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool}) ->
  {Nat.is_lt(location(axes,Views.last_coordinates(axes),U32.to_nat(offset)),U32.to_nat(length)) == True{} : Bool}:
  found_inside(Views.address(Views.View{axes,offset},Views.last_coordinates(axes),length),location(axes,Views.last_coordinates(axes),U32.to_nat(offset)),length,
    fits,address(axes,Views.last_coordinates(axes),offset,length))

def last_span(axes: List<&2,Views.Axis>) -> Nat:
  match axes:
    case []: 0n
    case Views.Axis{extent,stride} <> tail: Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),last_span(tail))

def last_location(+axes: List<&2,Views.Axis>,+offset: Nat) ->
  {location(axes,Views.last_coordinates(axes),offset) == Nat.add(offset,last_span(axes)) : Nat}:
  match axes:
    case []: Equal.sym(Nat,Nat.add(offset,0n),offset,Addition.zero_right(offset))
    case Views.Axis{+extent,+stride} <> +tail:
      Equal.trans(Nat,location(tail,Views.last_coordinates(tail),Nat.add(offset,Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)))),
        Nat.add(Nat.add(offset,Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride))),last_span(tail)),
        Nat.add(offset,last_span(Views.Axis{extent,stride} <> tail)),
        last_location(tail,Nat.add(offset,Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)))),
        Addition.associative(offset,Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),last_span(tail)))
