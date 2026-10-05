# Pointwise semantics of the actual ordered in-place map traversal. A bounded
# cursor visits each address once, so later stores preserve earlier results.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./traversal_array_refinement.bend as ArrayLaws
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./u32_comparison.bend as Comparison
import ./nat_interval.bend as Interval
import ./nat_division.bend as Division
import ./word_encoding.bend as WordEncoding

def output(~Element: Data,~Context: Data,state: Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>) -> Tree.Tree<Element>:
  Traversal.State{reader,output,position,context} = state
  output

def run(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,count: Nat,position: U32,context: Context,input: Tree.Tree<Element>) -> Tree.Tree<Element>:
  output(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>,~ArrayLaws.model_map_step(~Element,~Context,~operation),
    count,Traversal.State{Unit{},input,position,context}))

def capacity(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,count: Nat,+position: U32,+context: Context,+input: Tree.Tree<Element>) ->
  {Tree.capacity(Element,run(~Element,~Context,~operation,count,position,context,input)) == Tree.capacity(Element,input) : U32}:
  match count:
    case 0n: {==}
    case 1n+rest:
      +value = operation(context,Tree.read(Element,input,position))
      +written = Tree.write(Element,input,position,value)
      Equal.trans(U32,Tree.capacity(Element,run(~Element,~Context,~operation,rest,(position + 1 : U32),context,written)),Tree.capacity(Element,written),Tree.capacity(Element,input),
        capacity(~Element,~Context,~operation,rest,(position + 1 : U32),context,written),StorageReadback.write_capacity(~Element,input,position,value))

def balanced(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,count: Nat,+position: U32,+context: Context,+input: Tree.Tree<Element>,+depth: Nat,
  evidence: Shape.Balanced(Element,depth,input)) -> Shape.Balanced(Element,depth,run(~Element,~Context,~operation,count,position,context,input)):
  match count:
    case 0n: evidence
    case 1n+rest:
      +value = operation(context,Tree.read(Element,input,position))
      balanced(~Element,~Context,~operation,rest,(position + 1 : U32),context,Tree.write(Element,input,position,value),depth,Shape.write(Element,depth,input,position,value,evidence))

def address_inside(+position: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +inside: {Nat.is_lt(position,U32.to_nat(capacity)) == True{} : Bool}) -> {U32.is_lt(U32.from_nat(position),capacity) == True{} : Bool}:
  Bounds.native_bound(position,capacity,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),inside)),inside)

def unchanged(~Element: Data,+depth: Nat,+input: Tree.Tree<Element>,+position: U32,+query: U32,+value: Element,+capacity: U32,
  +capacity_equal: {Tree.capacity(Element,input) == capacity : U32},balanced: Shape.Balanced(Element,depth,input),limit: {Nat.is_le(depth,30n) == True{} : Bool},
  position_inside: {U32.is_lt(position,capacity) == True{} : Bool},query_inside: {U32.is_lt(query,capacity) == True{} : Bool},different: {U32.is_eq(position,query) == False{} : Bool}) ->
  {Tree.read(Element,Tree.write(Element,input,position,value),query) == Tree.read(Element,input,query) : Element}:
  StorageReadback.read_preserved(~Element,depth,input,position,query,value,balanced,limit,
    %Equal.sym(U32,Tree.capacity(Element,input),capacity,capacity_equal) : {U32.is_lt(position,_) == True{} : Bool} position_inside,
    %Equal.sym(U32,Tree.capacity(Element,input),capacity,capacity_equal) : {U32.is_lt(query,_) == True{} : Bool} query_inside,different)

def outside(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,count: Nat,+position: Nat,+query: U32,+context: Context,+input: Tree.Tree<Element>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Element,input) == capacity : U32},+balanced: Shape.Balanced(Element,depth,input),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool},
  +query_inside: {U32.is_lt(query,capacity) == True{} : Bool},+separate: Interval.Outside<position,count,U32.to_nat(query)>) ->
  {Tree.read(Element,run(~Element,~Context,~operation,count,U32.from_nat(position),context,input),query) == Tree.read(Element,input,query) : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +value = operation(context,Tree.read(Element,input,U32.from_nat(position)))
      +written = Tree.write(Element,input,U32.from_nat(position),value)
      +position_inside = Order.room_index(position,rest,U32.to_nat(capacity),room)
      +different = {
        %Equal.sym(Bool,U32.is_eq(U32.from_nat(position),query),Nat.is_eq(U32.to_nat(U32.from_nat(position)),U32.to_nat(query)),Comparison.eq(U32.from_nat(position),query)) : {_ == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) :
          {Nat.is_eq(_,U32.to_nat(query)) == False{} : Bool}
        %Order.equal_symmetric(U32.to_nat(query),position) : {_ == False{} : Bool}
        Interval.outside_head(position,rest,U32.to_nat(query),separate) : {U32.is_eq(U32.from_nat(position),query) == False{} : Bool}}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,run(~Element,~Context,~operation,rest,_,context,written),query) == Tree.read(Element,input,query) : Element}
      Equal.trans(Element,Tree.read(Element,run(~Element,~Context,~operation,rest,U32.from_nat(1n+position),context,written),query),Tree.read(Element,written,query),Tree.read(Element,input,query),
        outside(~Element,~Context,~operation,rest,1n+position,query,context,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,input),capacity,StorageReadback.write_capacity(~Element,input,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,input,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),query_inside,
          Interval.outside_tail(position,rest,U32.to_nat(query),separate)),
        unchanged(~Element,depth,input,U32.from_nat(position),query,value,capacity,capacity_equal,balanced,limit,address_inside(position,capacity,capacity_ok,position_inside),query_inside,different))

def inside(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,count: Nat,+position: Nat,index: Nat,+context: Context,+input: Tree.Tree<Element>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Element,input) == capacity : U32},+balanced: Shape.Balanced(Element,depth,input),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool},
  +selected: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tree.read(Element,run(~Element,~Context,~operation,count,U32.from_nat(position),context,input),U32.from_nat(Nat.add(position,index))) ==
    operation(context,Tree.read(Element,input,U32.from_nat(Nat.add(position,index)))) : Element}:
  match count index:
    case 0n +index:
      Empty.absurd({Tree.read(Element,input,U32.from_nat(Nat.add(position,index))) == operation(context,Tree.read(Element,input,U32.from_nat(Nat.add(position,index)))) : Element},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),selected)))
    case 1n+ +rest 0n:
      +value = operation(context,Tree.read(Element,input,U32.from_nat(position)))
      +written = Tree.write(Element,input,U32.from_nat(position),value)
      +position_inside = Order.room_index(position,rest,U32.to_nat(capacity),room)
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {Tree.read(Element,run(~Element,~Context,~operation,1n+rest,U32.from_nat(position),context,input),U32.from_nat(_)) == operation(context,Tree.read(Element,input,U32.from_nat(_))) : Element}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,run(~Element,~Context,~operation,rest,_,context,written),U32.from_nat(position)) == value : Element}
      Equal.trans(Element,Tree.read(Element,run(~Element,~Context,~operation,rest,U32.from_nat(1n+position),context,written),U32.from_nat(position)),Tree.read(Element,written,U32.from_nat(position)),value,
        outside(~Element,~Context,~operation,rest,1n+position,U32.from_nat(position),context,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,input),capacity,StorageReadback.write_capacity(~Element,input,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,input,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),address_inside(position,capacity,capacity_ok,position_inside),
          Interval.Before{
            %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) : {Nat.is_lt(_,1n+position) == True{} : Bool}
            Order.lt_succ(position,position,Order.reflexive(position))}),
        StorageReadback.read_written(~Element,input,U32.from_nat(position),value))
    case 1n+ +rest 1n+ +index:
      +value = operation(context,Tree.read(Element,input,U32.from_nat(position)))
      +written = Tree.write(Element,input,U32.from_nat(position),value)
      +query_position = Nat.add(1n+position,index)
      +query = U32.from_nat(query_position)
      +position_inside = Order.room_index(position,rest,U32.to_nat(capacity),room)
      +query_inside = Bounds.nat_trans(query_position,Nat.add(1n+position,rest),U32.to_nat(capacity),Order.offset_lt(1n+position,index,rest,selected),Order.advance_room(position,rest,U32.to_nat(capacity),room))
      +different = {
        %Equal.sym(Bool,U32.is_eq(U32.from_nat(position),query),Nat.is_eq(U32.to_nat(U32.from_nat(position)),U32.to_nat(query)),Comparison.eq(U32.from_nat(position),query)) : {_ == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) : {Nat.is_eq(_,U32.to_nat(query)) == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(query),query_position,Bounds.roundtrip(query_position,capacity,capacity_ok,Bounds.strict_le(query_position,U32.to_nat(capacity),query_inside))) : {Nat.is_eq(position,_) == False{} : Bool}
        Order.less_not_equal(position,query_position,Bounds.nat_trans(position,1n+position,query_position,Order.lt_succ(position,position,Order.reflexive(position)),Order.le_add(1n+position,index))) : {U32.is_eq(U32.from_nat(position),query) == False{} : Bool}}
      %Equal.sym(Nat,Nat.add(position,1n+index),query_position,Addition.successor_right(position,index)) :
        {Tree.read(Element,run(~Element,~Context,~operation,1n+rest,U32.from_nat(position),context,input),U32.from_nat(_)) == operation(context,Tree.read(Element,input,U32.from_nat(_))) : Element}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,run(~Element,~Context,~operation,rest,_,context,written),query) == operation(context,Tree.read(Element,input,query)) : Element}
      Equal.trans(Element,Tree.read(Element,run(~Element,~Context,~operation,rest,U32.from_nat(1n+position),context,written),query),operation(context,Tree.read(Element,written,query)),operation(context,Tree.read(Element,input,query)),
        inside(~Element,~Context,~operation,rest,1n+position,index,context,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,input),capacity,StorageReadback.write_capacity(~Element,input,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,input,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),selected),
        Equal.cong(Element,Element,element => operation(context,element),Tree.read(Element,written,query),Tree.read(Element,input,query),
          unchanged(~Element,depth,input,U32.from_nat(position),query,value,capacity,capacity_equal,balanced,limit,address_inside(position,capacity,capacity_ok,position_inside),address_inside(query_position,capacity,capacity_ok,query_inside),different)))
