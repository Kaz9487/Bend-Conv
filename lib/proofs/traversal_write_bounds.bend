# The incrementing U32 store schedule is exact, in bounds and injective.
import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./natural_addition.bend as Addition
import ./word_arithmetic.bend as WordArithmetic
import ./storage_shape.bend as Shape

law Schedule:
  for count: Nat
  for position: U32
  for length: U32
  Data

def Schedule(count,position,length):
  match count:
    case 0n: Unit
    case 1n+remaining:
      Shape.Both<{Nat.is_lt(U32.to_nat(position),U32.to_nat(length)) == True{} : Bool},
        Shape.Both<{U32.to_nat((position + 1 : U32)) == 1n+U32.to_nat(position) : Nat},Schedule(remaining,(position + 1 : U32),length)>>

def schedule(count: Nat,+position: U32,+length: U32,
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),count),U32.to_nat(length)) == True{} : Bool}) -> Schedule(count,position,length):
  match count:
    case 0n: Unit{}
    case 1n+ +remaining:
      +inside = Order.room_index(U32.to_nat(position),remaining,U32.to_nat(length),room)
      +successor = FullRange.successor(position,Bounds.uint_trans(position,length,4294967295,{
        %Equal.sym(Bool,U32.is_lt(position,length),Nat.is_lt(U32.to_nat(position),U32.to_nat(length)),Comparison.lt(position,length)) : {_ == True{} : Bool}
        inside : {U32.is_lt(position,length) == True{} : Bool}},FullRange.maximum(length)))
      Shape.Both{inside,Shape.Both{successor,schedule(remaining,(position + 1 : U32),length,{
        %Equal.sym(Nat,U32.to_nat((position + 1 : U32)),1n+U32.to_nat(position),successor) : {Nat.is_le(Nat.add(_,remaining),U32.to_nat(length)) == True{} : Bool}
        Order.advance_room(U32.to_nat(position),remaining,U32.to_nat(length),room) : {Nat.is_le(Nat.add(U32.to_nat((position + 1 : U32)),remaining),U32.to_nat(length)) == True{} : Bool}})}}

def position_after(count: Nat,position: U32) -> U32:
  match count:
    case 0n: position
    case 1n+remaining: position_after(remaining,(position + 1 : U32))

def increment_encoded(+position: Nat) -> {(U32.from_nat(position) + 1 : U32) == U32.from_nat(1n+position) : U32}:
  %Addition.successor_width(position) : {(U32.from_nat(position) + 1 : U32) == U32.from_nat(_) : U32}
  WordArithmetic.native_add(position,1n)

def positions(+count: Nat,+position: Nat) ->
  {position_after(count,U32.from_nat(position)) == U32.from_nat(Nat.add(position,count)) : U32}:
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) : {U32.from_nat(position) == U32.from_nat(_) : U32}
      {==}
    case 1n+ +remaining:
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),increment_encoded(position)) :
        {position_after(remaining,_) == U32.from_nat(Nat.add(position,1n+remaining)) : U32}
      %Equal.sym(Nat,Nat.add(position,1n+remaining),1n+Nat.add(position,remaining),Addition.successor_right(position,remaining)) :
        {position_after(remaining,U32.from_nat(1n+position)) == U32.from_nat(_) : U32}
      positions(remaining,1n+position)

def encoded_distinct(+left: Nat,+right: Nat,+length: U32,
  +ordered: {Nat.is_lt(left,right) == True{} : Bool},+right_inside: {Nat.is_le(right,U32.to_nat(length)) == True{} : Bool}) ->
  {U32.is_eq(U32.from_nat(left),U32.from_nat(right)) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(U32.from_nat(left),U32.from_nat(right)),Nat.is_eq(U32.to_nat(U32.from_nat(left)),U32.to_nat(U32.from_nat(right))),Comparison.eq(U32.from_nat(left),U32.from_nat(right))) : {_ == False{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(left)),left,FullRange.roundtrip(left,length,Bounds.strict_le(left,U32.to_nat(length),Bounds.nat_trans(left,right,U32.to_nat(length),ordered,right_inside)))) :
    {Nat.is_eq(_,U32.to_nat(U32.from_nat(right))) == False{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(right)),right,FullRange.roundtrip(right,length,right_inside)) : {Nat.is_eq(left,_) == False{} : Bool}
  Order.less_not_equal(left,right,ordered)

def distinct_positions(+left: Nat,+right: Nat,+count: Nat,+offset: Nat,+length: U32,
  ordered: {Nat.is_lt(left,right) == True{} : Bool},right_inside: {Nat.is_lt(right,count) == True{} : Bool},
  room: {Nat.is_le(Nat.add(offset,count),U32.to_nat(length)) == True{} : Bool}) ->
  {U32.is_eq(position_after(left,U32.from_nat(offset)),position_after(right,U32.from_nat(offset))) == False{} : Bool}:
  %Equal.sym(U32,position_after(left,U32.from_nat(offset)),U32.from_nat(Nat.add(offset,left)),positions(left,offset)) :
    {U32.is_eq(_,position_after(right,U32.from_nat(offset))) == False{} : Bool}
  %Equal.sym(U32,position_after(right,U32.from_nat(offset)),U32.from_nat(Nat.add(offset,right)),positions(right,offset)) :
    {U32.is_eq(U32.from_nat(Nat.add(offset,left)),_) == False{} : Bool}
  encoded_distinct(Nat.add(offset,left),Nat.add(offset,right),length,Order.offset_lt(offset,left,right,ordered),
    Bounds.strict_le(Nat.add(offset,right),U32.to_nat(length),Bounds.nat_trans(Nat.add(offset,right),Nat.add(offset,count),U32.to_nat(length),Order.offset_lt(offset,right,count,right_inside),room)))
