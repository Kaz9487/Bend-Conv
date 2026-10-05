# Natural interval bounds discharge the concrete native copy's store bounds.
# Readback is about the actual write program, including its U32 cursor.
import Base
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./contiguous_copy.bend as Contiguous
import ./write_program_readback.bend as Readback
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./u32_comparison.bend as Comparison
import ./nat_division.bend as Division
import ./word_encoding.bend as WordEncoding
import ./write_program.bend as Writes
import ./nat_interval.bend as Interval
import ./boolean_logic.bend as Boolean

def offset_exact(+base: U32,+position: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(base),position),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((base + U32.from_nat(position) : U32)) == Nat.add(U32.to_nat(base),position) : Nat}:
  +position_exact = Bounds.roundtrip(position,capacity,capacity_ok,
    Order.transitive(position,Nat.add(U32.to_nat(base),position),U32.to_nat(capacity),
      %Order.commutative(position,U32.to_nat(base)) : {Nat.is_le(position,_) == True{} : Bool}
      Order.le_add(position,U32.to_nat(base)),room))
  %position_exact : {U32.to_nat((base + U32.from_nat(position) : U32)) == Nat.add(U32.to_nat(base),_) : Nat}
  Arithmetic.add(base,U32.from_nat(position),capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,position_exact) : {Nat.is_le(Nat.add(U32.to_nat(base),_),U32.to_nat(capacity)) == True{} : Bool}
    room)

def offset_inside(+base: U32,+position: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(U32.to_nat(base),position),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.is_lt((base + U32.from_nat(position) : U32),capacity) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt((base + U32.from_nat(position) : U32),capacity),Nat.is_lt(U32.to_nat((base + U32.from_nat(position) : U32)),U32.to_nat(capacity)),
    Comparison.lt((base + U32.from_nat(position) : U32),capacity)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((base + U32.from_nat(position) : U32)),Nat.add(U32.to_nat(base),position),
    offset_exact(base,position,capacity,capacity_ok,Bounds.strict_le(Nat.add(U32.to_nat(base),position),U32.to_nat(capacity),inside))) :
    {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
  inside

def next_room(+base: U32,+position: Nat,+rest: Nat,+capacity: U32,
  room: {Nat.is_le(Nat.add(Nat.add(U32.to_nat(base),position),1n+rest),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(U32.to_nat(base),1n+position),rest),U32.to_nat(capacity)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(U32.to_nat(base),1n+position),1n+Nat.add(U32.to_nat(base),position),Addition.successor_right(U32.to_nat(base),position)) :
    {Nat.is_le(Nat.add(_,rest),U32.to_nat(capacity)) == True{} : Bool}
  Order.advance_room(Nat.add(U32.to_nat(base),position),rest,U32.to_nat(capacity),room)

def writes_inside(~Element: Data,count: Nat,+position: Nat,+input: Tree.Tree<Element>,+source: U32,+destination: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.add(U32.to_nat(destination),position),count),U32.to_nat(capacity)) == True{} : Bool}) ->
  Readback.Inside(~Element,capacity,Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{offset_inside(destination,position,capacity,capacity_ok,Order.room_index(Nat.add(U32.to_nat(destination),position),rest,U32.to_nat(capacity),room)),
        %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
          Readback.Inside(~Element,capacity,Contiguous.writes(~Element,rest,input,_,source,destination))
        writes_inside(~Element,rest,1n+position,input,source,destination,capacity,capacity_ok,next_room(destination,position,rest,capacity,room))}

def same_address(+address: U32) -> {U32.is_eq(address,address) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(address,address),Nat.is_eq(U32.to_nat(address),U32.to_nat(address)),Comparison.eq(address,address)) : {_ == True{} : Bool}
  Order.equal_reflexive(U32.to_nat(address))

def selected_outside(~Element: Data,count: Nat,+position: Nat,+query: U32,+input: Tree.Tree<Element>,+source: U32,+destination: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.add(U32.to_nat(destination),position),count),U32.to_nat(capacity)) == True{} : Bool},
  +outside: Interval.Outside<Nat.add(U32.to_nat(destination),position),count,U32.to_nat(query)>,+previous: Element) ->
  {Readback.selected(~Element,Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination),query,previous) == previous : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +different = {
        %Equal.sym(Bool,U32.is_eq((destination + U32.from_nat(position) : U32),query),Nat.is_eq(U32.to_nat((destination + U32.from_nat(position) : U32)),U32.to_nat(query)),Comparison.eq((destination + U32.from_nat(position) : U32),query)) : {_ == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat((destination + U32.from_nat(position) : U32)),Nat.add(U32.to_nat(destination),position),
          offset_exact(destination,position,capacity,capacity_ok,Bounds.strict_le(Nat.add(U32.to_nat(destination),position),U32.to_nat(capacity),Order.room_index(Nat.add(U32.to_nat(destination),position),rest,U32.to_nat(capacity),room)))) :
          {Nat.is_eq(_,U32.to_nat(query)) == False{} : Bool}
        %Order.equal_symmetric(U32.to_nat(query),Nat.add(U32.to_nat(destination),position)) : {_ == False{} : Bool}
        Interval.outside_head(Nat.add(U32.to_nat(destination),position),rest,U32.to_nat(query),outside) :
        {U32.is_eq((destination + U32.from_nat(position) : U32),query) == False{} : Bool}}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Readback.selected(~Element,Contiguous.writes(~Element,rest,input,_,source,destination),query,
          Bool.pick(Element,U32.is_eq((destination + U32.from_nat(position) : U32),query),Tree.read(Element,input,(source + U32.from_nat(position) : U32)),previous)) == previous : Element}
      %Equal.sym(Bool,U32.is_eq((destination + U32.from_nat(position) : U32),query),False{},different) :
        {Readback.selected(~Element,Contiguous.writes(~Element,rest,input,U32.from_nat(1n+position),source,destination),query,
          Bool.pick(Element,_,Tree.read(Element,input,(source + U32.from_nat(position) : U32)),previous)) == previous : Element}
      selected_outside(~Element,rest,1n+position,query,input,source,destination,capacity,capacity_ok,next_room(destination,position,rest,capacity,room),
        %Equal.sym(Nat,Nat.add(U32.to_nat(destination),1n+position),1n+Nat.add(U32.to_nat(destination),position),Addition.successor_right(U32.to_nat(destination),position)) : Interval.Outside<_,rest,U32.to_nat(query)>
        Interval.outside_tail(Nat.add(U32.to_nat(destination),position),rest,U32.to_nat(query),outside),previous)

def selected_inside(~Element: Data,count: Nat,+position: Nat,index: Nat,+input: Tree.Tree<Element>,+source: U32,+destination: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.add(U32.to_nat(destination),position),count),U32.to_nat(capacity)) == True{} : Bool},
  inside: {Nat.is_lt(index,count) == True{} : Bool},+previous: Element) ->
  {Readback.selected(~Element,Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination),(destination + U32.from_nat(Nat.add(position,index)) : U32),previous) ==
    Tree.read(Element,input,(source + U32.from_nat(Nat.add(position,index)) : U32)) : Element}:
  match count index:
    case 0n +index:
      Empty.absurd({previous == Tree.read(Element,input,(source + U32.from_nat(Nat.add(position,index)) : U32)) : Element},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),inside)))
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {Readback.selected(~Element,Contiguous.writes(~Element,1n+rest,input,U32.from_nat(position),source,destination),(destination + U32.from_nat(_) : U32),previous) ==
          Tree.read(Element,input,(source + U32.from_nat(_) : U32)) : Element}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Readback.selected(~Element,Contiguous.writes(~Element,rest,input,_,source,destination),(destination + U32.from_nat(position) : U32),
          Bool.pick(Element,U32.is_eq((destination + U32.from_nat(position) : U32),(destination + U32.from_nat(position) : U32)),Tree.read(Element,input,(source + U32.from_nat(position) : U32)),previous)) ==
          Tree.read(Element,input,(source + U32.from_nat(position) : U32)) : Element}
      %Equal.sym(Bool,U32.is_eq((destination + U32.from_nat(position) : U32),(destination + U32.from_nat(position) : U32)),True{},same_address((destination + U32.from_nat(position) : U32))) :
        {Readback.selected(~Element,Contiguous.writes(~Element,rest,input,U32.from_nat(1n+position),source,destination),(destination + U32.from_nat(position) : U32),
          Bool.pick(Element,_,Tree.read(Element,input,(source + U32.from_nat(position) : U32)),previous)) == Tree.read(Element,input,(source + U32.from_nat(position) : U32)) : Element}
      selected_outside(~Element,rest,1n+position,(destination + U32.from_nat(position) : U32),input,source,destination,capacity,capacity_ok,next_room(destination,position,rest,capacity,room),
        Interval.Before{
          %Equal.sym(Nat,U32.to_nat((destination + U32.from_nat(position) : U32)),Nat.add(U32.to_nat(destination),position),
            offset_exact(destination,position,capacity,capacity_ok,Bounds.strict_le(Nat.add(U32.to_nat(destination),position),U32.to_nat(capacity),Order.room_index(Nat.add(U32.to_nat(destination),position),rest,U32.to_nat(capacity),room)))) :
            {Nat.is_lt(_,Nat.add(U32.to_nat(destination),1n+position)) == True{} : Bool}
          %Equal.sym(Nat,Nat.add(U32.to_nat(destination),1n+position),1n+Nat.add(U32.to_nat(destination),position),Addition.successor_right(U32.to_nat(destination),position)) :
            {Nat.is_lt(Nat.add(U32.to_nat(destination),position),_) == True{} : Bool}
          Order.lt_succ(Nat.add(U32.to_nat(destination),position),Nat.add(U32.to_nat(destination),position),Order.reflexive(Nat.add(U32.to_nat(destination),position)))},
        Tree.read(Element,input,(source + U32.from_nat(position) : U32)))
    case 1n+ +rest 1n+ +index:
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Readback.selected(~Element,Contiguous.writes(~Element,rest,input,_,source,destination),(destination + U32.from_nat(Nat.add(position,1n+index)) : U32),
          Bool.pick(Element,U32.is_eq((destination + U32.from_nat(position) : U32),(destination + U32.from_nat(Nat.add(position,1n+index)) : U32)),Tree.read(Element,input,(source + U32.from_nat(position) : U32)),previous)) ==
          Tree.read(Element,input,(source + U32.from_nat(Nat.add(position,1n+index)) : U32)) : Element}
      %Equal.sym(Nat,Nat.add(position,1n+index),Nat.add(1n+position,index),Addition.successor_right(position,index)) :
        {Readback.selected(~Element,Contiguous.writes(~Element,rest,input,U32.from_nat(1n+position),source,destination),(destination + U32.from_nat(_) : U32),
          Bool.pick(Element,U32.is_eq((destination + U32.from_nat(position) : U32),(destination + U32.from_nat(_) : U32)),Tree.read(Element,input,(source + U32.from_nat(position) : U32)),previous)) ==
          Tree.read(Element,input,(source + U32.from_nat(_) : U32)) : Element}
      selected_inside(~Element,rest,1n+position,index,input,source,destination,capacity,capacity_ok,next_room(destination,position,rest,capacity,room),inside,
        Bool.pick(Element,U32.is_eq((destination + U32.from_nat(position) : U32),(destination + U32.from_nat(Nat.add(1n+position,index)) : U32)),Tree.read(Element,input,(source + U32.from_nat(position) : U32)),previous))

def read_inside(~Element: Data,+count: Nat,+position: Nat,+index: Nat,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+source: U32,+destination: U32,+depth: Nat,
  +balanced: Shape.Balanced(Element,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.add(U32.to_nat(destination),position),count),U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination),output),(destination + U32.from_nat(Nat.add(position,index)) : U32)) ==
    Tree.read(Element,input,(source + U32.from_nat(Nat.add(position,index)) : U32)) : Element}:
  +query = (destination + U32.from_nat(Nat.add(position,index)) : U32)
  +capacity = Tree.capacity(Element,output)
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination),output),query),
    Readback.selected(~Element,Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination),query,Tree.read(Element,output,query)),
    Tree.read(Element,input,(source + U32.from_nat(Nat.add(position,index)) : U32)),
    Readback.program(~Element,Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination),depth,output,query,balanced,depth_ok,
      offset_inside(destination,Nat.add(position,index),capacity,capacity_ok,
        %Addition.associative(U32.to_nat(destination),position,index) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
        Bounds.nat_trans(Nat.add(Nat.add(U32.to_nat(destination),position),index),Nat.add(Nat.add(U32.to_nat(destination),position),count),U32.to_nat(capacity),
          Order.offset_lt(Nat.add(U32.to_nat(destination),position),index,count,inside),room)),
      writes_inside(~Element,count,position,input,source,destination,capacity,capacity_ok,room)),
    selected_inside(~Element,count,position,index,input,source,destination,capacity,capacity_ok,room,inside,Tree.read(Element,output,query)))

def read_outside(~Element: Data,+count: Nat,+position: Nat,+query: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+source: U32,+destination: U32,+depth: Nat,
  +balanced: Shape.Balanced(Element,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.add(U32.to_nat(destination),position),count),U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +query_inside: {U32.is_lt(query,Tree.capacity(Element,output)) == True{} : Bool},
  +outside: Interval.Outside<Nat.add(U32.to_nat(destination),position),count,U32.to_nat(query)>) ->
  {Tree.read(Element,Writes.pure(Element,Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination),output),query) == Tree.read(Element,output,query) : Element}:
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination),output),query),
    Readback.selected(~Element,Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination),query,Tree.read(Element,output,query)),Tree.read(Element,output,query),
    Readback.program(~Element,Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination),depth,output,query,balanced,depth_ok,query_inside,
      writes_inside(~Element,count,position,input,source,destination,Tree.capacity(Element,output),capacity_ok,room)),
    selected_outside(~Element,count,position,query,input,source,destination,Tree.capacity(Element,output),capacity_ok,room,outside,Tree.read(Element,output,query)))

def covered(~Element: Data,count: Nat,+position: Nat,index: Nat,+input: Tree.Tree<Element>,+source: U32,+destination: U32,
  inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Readback.contains(~Element,(destination + U32.from_nat(Nat.add(position,index)) : U32),Contiguous.writes(~Element,count,input,U32.from_nat(position),source,destination)) == True{} : Bool}:
  match count index:
    case 0n +index:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),inside)))
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {Readback.contains(~Element,(destination + U32.from_nat(_) : U32),Contiguous.writes(~Element,1n+rest,input,U32.from_nat(position),source,destination)) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq((destination + U32.from_nat(position) : U32),(destination + U32.from_nat(position) : U32)),True{},same_address((destination + U32.from_nat(position) : U32))) :
        {_ || Readback.contains(~Element,(destination + U32.from_nat(position) : U32),Contiguous.writes(~Element,rest,input,(U32.from_nat(position) + 1 : U32),source,destination)) == True{} : Bool}
      {==}
    case 1n+ +rest 1n+ +index:
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {U32.is_eq((destination + U32.from_nat(position) : U32),(destination + U32.from_nat(Nat.add(position,1n+index)) : U32)) ||
          Readback.contains(~Element,(destination + U32.from_nat(Nat.add(position,1n+index)) : U32),Contiguous.writes(~Element,rest,input,_,source,destination)) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(position,1n+index),Nat.add(1n+position,index),Addition.successor_right(position,index)) :
        {U32.is_eq((destination + U32.from_nat(position) : U32),(destination + U32.from_nat(_) : U32)) ||
          Readback.contains(~Element,(destination + U32.from_nat(_) : U32),Contiguous.writes(~Element,rest,input,U32.from_nat(1n+position),source,destination)) == True{} : Bool}
      %Equal.sym(Bool,Readback.contains(~Element,(destination + U32.from_nat(Nat.add(1n+position,index)) : U32),Contiguous.writes(~Element,rest,input,U32.from_nat(1n+position),source,destination)),True{},
        covered(~Element,rest,1n+position,index,input,source,destination,inside)) :
        {U32.is_eq((destination + U32.from_nat(position) : U32),(destination + U32.from_nat(Nat.add(1n+position,index)) : U32)) || _ == True{} : Bool}
      Boolean.or_true_right(U32.is_eq((destination + U32.from_nat(position) : U32),(destination + U32.from_nat(Nat.add(1n+position,index)) : U32)))
