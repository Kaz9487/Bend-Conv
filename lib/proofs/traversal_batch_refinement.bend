# A batch writes the mapped values of consecutive positions after reading all
# of them from the original tree. Inside the tree it is the ordered in-place
# map of the same positions: no step of that map reads a position an earlier
# step wrote. For every element type and every batch length.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_map_readback.bend as MapReadback
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_interval.bend as Interval

def batch(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,count: Nat,+position: U32,+context: Context,+source: Tree.Tree<Element>,
  target: Tree.Tree<Element>) -> Tree.Tree<Element>:
  match count:
    case 0n: target
    case 1n+rest:
      batch(~Element,~Context,~map,rest,(position + 1 : U32),context,source,Tree.write(Element,target,position,map(context,position,Tree.read(Element,source,position))))

def ModelState(~Element: Data,~Context: Data) -> Data:
  Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>

# A loop's last step, for every state and count.
def last_step(~State: Data,~step: State -> State,+count: Nat,+state: State) ->
  {Loop.run(~State,~step,1n+count,state) == step(Loop.run(~State,~step,count,state)) : State}:
  match count:
    case 0n: {==}
    case 1n+ +rest: last_step(~State,~step,rest,step(state))

# The whole state after count steps of the model map.
def state_after(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,+count: Nat,+start: Nat,+context: Context,+input: Tree.Tree<Element>) ->
  {Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),count,Traversal.State{Unit{},input,U32.from_nat(start),context}) ==
    Traversal.State{Unit{},MapReadback.run_indexed(~Element,~Context,~map,count,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,count)),context} :
    ModelState(~Element,~Context)}:
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) :
        {Traversal.State{Unit{},input,U32.from_nat(start),context} == Traversal.State{Unit{},input,U32.from_nat(_),context} : ModelState(~Element,~Context)}
      {==}
    case 1n+ +rest:
      +written = Tree.write(Element,input,U32.from_nat(start),map(context,U32.from_nat(start),Tree.read(Element,input,U32.from_nat(start))))
      %Equal.sym(Nat,Nat.add(start,1n+rest),1n+Nat.add(start,rest),Addition.successor_right(start,rest)) :
        {Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),rest,Traversal.State{Unit{},written,(U32.from_nat(start) + 1 : U32),context}) ==
          Traversal.State{Unit{},MapReadback.run_indexed(~Element,~Context,~map,rest,(U32.from_nat(start) + 1 : U32),context,written),U32.from_nat(_),context} : ModelState(~Element,~Context)}
      %Equal.sym(U32,(U32.from_nat(start) + 1 : U32),U32.from_nat(1n+start),Bounds.add_one(U32.from_nat(start))) :
        {Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),rest,Traversal.State{Unit{},written,_,context}) ==
          Traversal.State{Unit{},MapReadback.run_indexed(~Element,~Context,~map,rest,_,context,written),U32.from_nat(1n+Nat.add(start,rest)),context} : ModelState(~Element,~Context)}
      state_after(~Element,~Context,~map,rest,1n+start,context,written)

# One more step of the model map writes the position after the ones it mapped.
def extended(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,+done: Nat,+start: Nat,+context: Context,+input: Tree.Tree<Element>) ->
  {MapReadback.run_indexed(~Element,~Context,~map,1n+done,U32.from_nat(start),context,input) ==
    Tree.write(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done)),
      map(context,U32.from_nat(Nat.add(start,done)),Tree.read(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done))))) : Tree.Tree<Element>}:
  +initial = {Traversal.State{Unit{},input,U32.from_nat(start),context} : ModelState(~Element,~Context)}
  %Equal.sym(ModelState(~Element,~Context),Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),1n+done,initial),
    ArrayLaws.model_map_step(~Element,~Context,~map,Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),done,initial)),
    last_step(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),done,initial)) :
    {MapReadback.output(~Element,~Context,_) ==
      Tree.write(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done)),
        map(context,U32.from_nat(Nat.add(start,done)),Tree.read(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done))))) : Tree.Tree<Element>}
  %Equal.sym(ModelState(~Element,~Context),Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),done,initial),
    Traversal.State{Unit{},MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done)),context},
    state_after(~Element,~Context,~map,done,start,context,input)) :
    {MapReadback.output(~Element,~Context,ArrayLaws.model_map_step(~Element,~Context,~map,_)) ==
      Tree.write(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done)),
        map(context,U32.from_nat(Nat.add(start,done)),Tree.read(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done))))) : Tree.Tree<Element>}
  {==}

# A batch that continues done steps of the map completes the map.
def continued(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,+remaining: Nat,+done: Nat,+start: Nat,+context: Context,+input: Tree.Tree<Element>,
  +depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(Element,input) == capacity : U32},+balanced: Shape.Balanced(Element,depth,input),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.add(start,done),remaining),U32.to_nat(capacity)) == True{} : Bool}) ->
  {batch(~Element,~Context,~map,remaining,U32.from_nat(Nat.add(start,done)),context,input,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input)) ==
    MapReadback.run_indexed(~Element,~Context,~map,Nat.add(done,remaining),U32.from_nat(start),context,input) : Tree.Tree<Element>}:
  match remaining:
    case 0n:
      %Equal.sym(Nat,Nat.add(done,0n),done,Addition.zero_right(done)) :
        {MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input) == MapReadback.run_indexed(~Element,~Context,~map,_,U32.from_nat(start),context,input) : Tree.Tree<Element>}
      {==}
    case 1n+ +rest:
      +reached = Nat.add(start,done)
      +query = U32.from_nat(reached)
      +mapped = MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input)
      +inside = Order.room_index(reached,rest,U32.to_nat(capacity),room)
      +within = Bounds.strict_le(reached,U32.to_nat(capacity),inside)
      +unread = MapReadback.outside_indexed(~Element,~Context,~map,done,start,query,context,input,depth,capacity,capacity_equal,balanced,limit,capacity_ok,within,
        MapReadback.address_inside(reached,capacity,capacity_ok,inside),
        Interval.After{
          %Equal.sym(Nat,U32.to_nat(query),reached,Bounds.roundtrip(reached,capacity,capacity_ok,within)) : {Nat.is_le(reached,_) == True{} : Bool}
          Order.reflexive(reached)})
      %Equal.sym(Nat,Nat.add(done,1n+rest),1n+Nat.add(done,rest),Addition.successor_right(done,rest)) :
        {batch(~Element,~Context,~map,rest,(query + 1 : U32),context,input,Tree.write(Element,mapped,query,map(context,query,Tree.read(Element,input,query)))) ==
          MapReadback.run_indexed(~Element,~Context,~map,_,U32.from_nat(start),context,input) : Tree.Tree<Element>}
      %unread :
        {batch(~Element,~Context,~map,rest,(query + 1 : U32),context,input,Tree.write(Element,mapped,query,map(context,query,_))) ==
          MapReadback.run_indexed(~Element,~Context,~map,1n+Nat.add(done,rest),U32.from_nat(start),context,input) : Tree.Tree<Element>}
      %extended(~Element,~Context,~map,done,start,context,input) :
        {batch(~Element,~Context,~map,rest,(query + 1 : U32),context,input,_) ==
          MapReadback.run_indexed(~Element,~Context,~map,1n+Nat.add(done,rest),U32.from_nat(start),context,input) : Tree.Tree<Element>}
      %Equal.sym(U32,(query + 1 : U32),U32.from_nat(1n+reached),Bounds.add_one(query)) :
        {batch(~Element,~Context,~map,rest,_,context,input,MapReadback.run_indexed(~Element,~Context,~map,1n+done,U32.from_nat(start),context,input)) ==
          MapReadback.run_indexed(~Element,~Context,~map,1n+Nat.add(done,rest),U32.from_nat(start),context,input) : Tree.Tree<Element>}
      %Addition.successor_right(start,done) :
        {batch(~Element,~Context,~map,rest,U32.from_nat(_),context,input,MapReadback.run_indexed(~Element,~Context,~map,1n+done,U32.from_nat(start),context,input)) ==
          MapReadback.run_indexed(~Element,~Context,~map,1n+Nat.add(done,rest),U32.from_nat(start),context,input) : Tree.Tree<Element>}
      continued(~Element,~Context,~map,rest,1n+done,start,context,input,depth,capacity,capacity_equal,balanced,limit,capacity_ok,
        %Equal.sym(Nat,Nat.add(start,1n+done),1n+reached,Addition.successor_right(start,done)) : {Nat.is_le(Nat.add(_,rest),U32.to_nat(capacity)) == True{} : Bool}
        Order.advance_room(reached,rest,U32.to_nat(capacity),room))

# A batch from the original tree is the map of its positions.
def mapped(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,+count: Nat,+start: Nat,+context: Context,+input: Tree.Tree<Element>,
  +depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(Element,input) == capacity : U32},+balanced: Shape.Balanced(Element,depth,input),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,count),U32.to_nat(capacity)) == True{} : Bool}) ->
  {batch(~Element,~Context,~map,count,U32.from_nat(start),context,input,input) == MapReadback.run_indexed(~Element,~Context,~map,count,U32.from_nat(start),context,input) : Tree.Tree<Element>}:
  %Addition.zero_right(start) :
    {batch(~Element,~Context,~map,count,U32.from_nat(_),context,input,input) == MapReadback.run_indexed(~Element,~Context,~map,count,U32.from_nat(start),context,input) : Tree.Tree<Element>}
  continued(~Element,~Context,~map,count,0n,start,context,input,depth,capacity,capacity_equal,balanced,limit,capacity_ok,
    %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) : {Nat.is_le(Nat.add(_,count),U32.to_nat(capacity)) == True{} : Bool}
    room)
