# Lanes.map is Traversal.map_in_place: on a balanced array that holds its
# count positions, the rounds through tiles and the ordered scalar map leave
# the same array.
import Base
import ../traversal_lanes.bend as Lanes
import ../traversal_cells.bend as Cells
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../kernel_tile.bend as Tile
import ./traversal_cells_refinement.bend as CellLaws
import ./traversal_batch_refinement.bend as Batch
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_map_readback.bend as MapReadback
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_shape.bend as Shape
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./word_encoding.bend as WordEncoding
import ./word_arithmetic.bend as WordArithmetic
import ./word_power_of_two.bend as Power

# The tile a round gathers at position.
def cell(~Context: Data,~map: Context -> F32 -> F32,+values: Tree.Tree<F32>,+position: U32,+context: Context) -> Tile.Tile8:
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  Tile.Tile{map(context,Tree.read(F32,values,position)),map(context,Tree.read(F32,values,p1)),map(context,Tree.read(F32,values,p2)),map(context,Tree.read(F32,values,p3)),
    map(context,Tree.read(F32,values,p4)),map(context,Tree.read(F32,values,p5)),map(context,Tree.read(F32,values,p6)),map(context,Tree.read(F32,values,p7))}

def gathered(~Context: Data,~map: Context -> F32 -> F32,+values: Tree.Tree<F32>,+position: U32,+context: Context) ->
  {Lanes.gather(~Context,~map,Tree.pack(F32,values),position,context) == (Tree.pack(F32,values),cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}:
  -owner = Tree.pack(F32,values)
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  +a = Tree.read(F32,values,position)
  +b = Tree.read(F32,values,p1)
  +c = Tree.read(F32,values,p2)
  +d = Tree.read(F32,values,p3)
  +e = Tree.read(F32,values,p4)
  +f = Tree.read(F32,values,p5)
  +g = Tree.read(F32,values,p6)
  +h = Tree.read(F32,values,p7)
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,position),(owner,a),StorageLaws.read(F32,values,position)) :
    {Lanes.gather1(~Context,~map,position,context,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p1),(owner,b),StorageLaws.read(F32,values,p1)) :
    {Lanes.gather2(~Context,~map,p1,context,a,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p2),(owner,c),StorageLaws.read(F32,values,p2)) :
    {Lanes.gather3(~Context,~map,p2,context,a,b,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p3),(owner,d),StorageLaws.read(F32,values,p3)) :
    {Lanes.gather4(~Context,~map,p3,context,a,b,c,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p4),(owner,e),StorageLaws.read(F32,values,p4)) :
    {Lanes.gather5(~Context,~map,p4,context,a,b,c,d,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p5),(owner,f),StorageLaws.read(F32,values,p5)) :
    {Lanes.gather6(~Context,~map,p5,context,a,b,c,d,e,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p6),(owner,g),StorageLaws.read(F32,values,p6)) :
    {Lanes.gather7(~Context,~map,p6,context,a,b,c,d,e,f,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p7),(owner,h),StorageLaws.read(F32,values,p7)) :
    {Lanes.gathered(~Context,~map,context,a,b,c,d,e,f,g,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  {==}

# Eight ordered writes from position.
def written(+values: Tree.Tree<F32>,+position: U32,a: F32,b: F32,c: F32,d: F32,e: F32,f: F32,g: F32,h: F32) -> Tree.Tree<F32>:
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  Tree.write(F32,Tree.write(F32,Tree.write(F32,Tree.write(F32,Tree.write(F32,Tree.write(F32,Tree.write(F32,Tree.write(F32,values,position,a),p1,b),p2,c),p3,d),p4,e),p5,f),p6,g),p7,h)

def scattered(+values: Tree.Tree<F32>,+position: U32,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,+g: F32,+h: F32) ->
  {Lanes.scatter(Tree.pack(F32,values),position,Tile.Tile{a,b,c,d,e,f,g,h}) == Tree.pack(F32,written(values,position,a,b,c,d,e,f,g,h)) : Array<F32>}:
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  +w1 = Tree.write(F32,values,position,a)
  +w2 = Tree.write(F32,w1,p1,b)
  +w3 = Tree.write(F32,w2,p2,c)
  +w4 = Tree.write(F32,w3,p3,d)
  +w5 = Tree.write(F32,w4,p4,e)
  +w6 = Tree.write(F32,w5,p5,f)
  +w7 = Tree.write(F32,w6,p6,g)
  +w8 = Tree.write(F32,w7,p7,h)
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,values),position,a),Tree.pack(F32,w1),StorageLaws.write(F32,values,position,a)) :
    {Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,_,p1,b),p2,c),p3,d),p4,e),p5,f),p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w1),p1,b),Tree.pack(F32,w2),StorageLaws.write(F32,w1,p1,b)) :
    {Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,_,p2,c),p3,d),p4,e),p5,f),p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w2),p2,c),Tree.pack(F32,w3),StorageLaws.write(F32,w2,p2,c)) :
    {Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,_,p3,d),p4,e),p5,f),p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w3),p3,d),Tree.pack(F32,w4),StorageLaws.write(F32,w3,p3,d)) :
    {Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,_,p4,e),p5,f),p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w4),p4,e),Tree.pack(F32,w5),StorageLaws.write(F32,w4,p4,e)) :
    {Array.set(F32,Array.set(F32,Array.set(F32,_,p5,f),p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w5),p5,f),Tree.pack(F32,w6),StorageLaws.write(F32,w5,p5,f)) :
    {Array.set(F32,Array.set(F32,_,p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w6),p6,g),Tree.pack(F32,w7),StorageLaws.write(F32,w6,p6,g)) :
    {Array.set(F32,_,p7,h) == Tree.pack(F32,w8) : Array<F32>}
  StorageLaws.write(F32,w7,p7,h)

def cell_scattered(~Context: Data,~map: Context -> F32 -> F32,+values: Tree.Tree<F32>,+position: U32,+context: Context) ->
  {Lanes.scatter(Tree.pack(F32,values),position,cell(~Context,~map,values,position,context)) ==
    Tree.pack(F32,Batch.batch(~F32,~Context,~ArrayLaws.unindexed(~F32,~Context,~map),8n,position,context,values,values)) : Array<F32>}:
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  scattered(values,position,map(context,Tree.read(F32,values,position)),map(context,Tree.read(F32,values,p1)),map(context,Tree.read(F32,values,p2)),map(context,Tree.read(F32,values,p3)),
    map(context,Tree.read(F32,values,p4)),map(context,Tree.read(F32,values,p5)),map(context,Tree.read(F32,values,p6)),map(context,Tree.read(F32,values,p7)))

# One round maps eight positions.
def round(~Context: Data,~map: Context -> F32 -> F32,+values: Tree.Tree<F32>,+scratch: Tree.Tree<Tile.Tile8>,+start: Nat,+slot: U32,+context: Context,
  +depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(F32,values) == capacity : U32},+balanced: Shape.Balanced(F32,depth,values),
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,8n),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Cells.step(~Array<F32>,~Tile.Tile8,~Context,~Lanes.gather(~Context,~map),~Lanes.scatter,~8,
      Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),U32.from_nat(start),slot,context}) ==
    Cells.State{Tree.pack(F32,MapReadback.run(~F32,~Context,~map,8n,U32.from_nat(start),context,values)),
      Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,scratch,slot,cell(~Context,~map,values,U32.from_nat(start),context))),(U32.from_nat(start) + 8 : U32),
      U32.and((slot + 1 : U32),63),context} : Cells.State<Array<F32>,Tile.Tile8,Context>}:
  +position = U32.from_nat(start)
  %Batch.mapped(~F32,~Context,~ArrayLaws.unindexed(~F32,~Context,~map),8n,start,context,values,depth,capacity,capacity_equal,balanced,Order.successor(depth,30n,limit),capacity_ok,room) :
    {Cells.step(~Array<F32>,~Tile.Tile8,~Context,~Lanes.gather(~Context,~map),~Lanes.scatter,~8,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}) ==
      Cells.State{Tree.pack(F32,_),Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,scratch,slot,cell(~Context,~map,values,position,context))),(position + 8 : U32),
        U32.and((slot + 1 : U32),63),context} : Cells.State<Array<F32>,Tile.Tile8,Context>}
  CellLaws.round(~F32,~Tile.Tile8,~Context,~Lanes.gather(~Context,~map),~Lanes.scatter,~8,values,scratch,position,slot,context,
    cell(~Context,~map,values,position,context),Batch.batch(~F32,~Context,~ArrayLaws.unindexed(~F32,~Context,~map),8n,position,context,values,values),
    gathered(~Context,~map,values,position,context),cell_scattered(~Context,~map,values,position,context))

# The scalar map from any position, on a packed tree.
def output(~Context: Data,+state: Batch.ModelState(~F32,~Context)) ->
  {Traversal.mapped_output(~F32,~Context,ArrayLaws.represent_map(~F32,~Context,state)) == Tree.pack(F32,MapReadback.output(~F32,~Context,state)) : Array<F32>}:
  Traversal.State{Unit{},output,position,context} = state
  {==}

def scalar(~Context: Data,~map: Context -> F32 -> F32,+count: Nat,+position: U32,+context: Context,+values: Tree.Tree<F32>) ->
  {Traversal.map_in_place(~F32,~Context,~map,count,position,context,Tree.pack(F32,values)) ==
    Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,position,context,values)) : Array<F32>}:
  +initial = {Traversal.State{Unit{},values,position,context} : Batch.ModelState(~F32,~Context)}
  %Equal.sym(Traversal.State<&2,&1,Unit,Array<F32>,Context>,
    Loop.run(~Traversal.State<&2,&1,Unit,Array<F32>,Context>,~Traversal.map_step(~F32,~Context,~map),count,Traversal.State{Unit{},Tree.pack(F32,values),position,context}),
    ArrayLaws.represent_map(~F32,~Context,Loop.run(~Batch.ModelState(~F32,~Context),~ArrayLaws.model_map_step(~F32,~Context,~ArrayLaws.unindexed(~F32,~Context,~map)),count,initial)),
    ArrayLaws.map_loop_refines_model(~F32,~Context,~map,count,initial)) :
    {Traversal.mapped_output(~F32,~Context,_) == Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,position,context,values)) : Array<F32>}
  output(~Context,Loop.run(~Batch.ModelState(~F32,~Context),~ArrayLaws.model_map_step(~F32,~Context,~ArrayLaws.unindexed(~F32,~Context,~map)),count,initial))

def few(~Context: Data,~map: Context -> F32 -> F32,+count: Nat,+position: U32,+slot: U32,+context: Context,+values: Tree.Tree<F32>,+scratch: Tree.Tree<Tile.Tile8>) ->
  {Lanes.finish(~Context,~map,count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}) ==
    Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,position,context,values)) : Array<F32>}:
  scalar(~Context,~map,count,position,context,values)

# The position of the model map after k steps from start.
def stepped(steps: Nat,start: Nat) -> U32:
  match steps:
    case 0n: U32.from_nat(start)
    case 1n+rest: (stepped(rest,start) + 1 : U32)

def advanced(+steps: Nat,+start: Nat) -> {stepped(steps,start) == U32.from_nat(Nat.add(steps,start)) : U32}:
  match steps:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(U32,stepped(rest,start),U32.from_nat(Nat.add(rest,start)),advanced(rest,start)) : {(_ + 1 : U32) == U32.from_nat(1n+Nat.add(rest,start)) : U32}
      Bounds.add_one(U32.from_nat(Nat.add(rest,start)))

def after_round(+start: Nat) -> {(U32.from_nat(start) + 8 : U32) == U32.from_nat(8n+start) : U32}:
  %Order.commutative(start,8n) : {(U32.from_nat(start) + 8 : U32) == U32.from_nat(_) : U32}
  WordArithmetic.native_add(start,8n)

def round_room(+start: Nat,+rest: Nat,+capacity: Nat,room: {Nat.is_le(Nat.add(start,8n+rest),capacity) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(start,8n),capacity) == True{} : Bool}:
  Order.transitive(Nat.add(start,8n),Nat.add(start,8n+rest),capacity,Order.offset_le(start,8n,8n+rest,Order.le_add(8n,rest)),room)

def rest_room(+start: Nat,+rest: Nat,+capacity: Nat,room: {Nat.is_le(Nat.add(start,8n+rest),capacity) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(8n+start,rest),capacity) == True{} : Bool}:
  Order.advance_room(7n+start,rest,capacity,Order.advance_room(6n+start,1n+rest,capacity,Order.advance_room(5n+start,2n+rest,capacity,
    Order.advance_room(4n+start,3n+rest,capacity,Order.advance_room(3n+start,4n+rest,capacity,Order.advance_room(2n+start,5n+rest,capacity,
      Order.advance_room(1n+start,6n+rest,capacity,Order.advance_room(start,7n+rest,capacity,room))))))))

# The shared induction on eight-value rounds. The proposition may carry all
# owners and bounds, so binary and ternary instances need only local cases.
def finish_small(~Owner: Type,~Context: Data,~gather: Owner -> U32 -> Context -> Owner & Tile.Tile8,
  ~scatter: Owner -> U32 -> Tile.Tile8 -> Owner,~finish: Nat -> Cells.State<Owner,Tile.Tile8,Context> -> Owner,
  +count: Nat,-state: Cells.State<Owner,Tile.Tile8,Context>,small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  {Lanes.rounds_with(~Owner,~Context,~gather,~scatter,~finish,count,state) == finish(count,state) : Owner}:
  match count:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 1n+1n+1n+1n+1n+1n+1n+1n+ +rest:
      Empty.absurd({Lanes.rounds_with(~Owner,~Context,~gather,~scatter,~finish,8n+rest,state) == finish(8n+rest,state) : Owner},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),small)))

def by_rounds(~Property: Nat -> Type,
  ~few: @+count: Nat -> {Nat.is_lt(count,8n) == True{} : Bool} -> Property(count),
  ~more: @+rest: Nat -> Property(rest) -> Property(8n+rest),count: Nat) -> Property(count):
  match count:
    case 0n: few(0n,{==})
    case 1n: few(1n,{==})
    case 2n: few(2n,{==})
    case 3n: few(3n,{==})
    case 4n: few(4n,{==})
    case 5n: few(5n,{==})
    case 6n: few(6n,{==})
    case 7n: few(7n,{==})
    case 1n+1n+1n+1n+1n+1n+1n+1n+ +rest:
      more(rest,by_rounds(~Property,~few,~more,rest))

# Every count: whole rounds, then the last positions one by one.
def rounds_property(~Context: Data,~map: Context -> F32 -> F32,count: Nat) -> Type:
  @+start: Nat -> @+slot: U32 -> @+context: Context -> @+values: Tree.Tree<F32> -> @+scratch: Tree.Tree<Tile.Tile8> ->
  @+depth: Nat -> @+capacity: U32 -> @+capacity_equal: {Tree.capacity(F32,values) == capacity : U32} -> @+balanced: Shape.Balanced(F32,depth,values) ->
  @+limit: {Nat.is_le(depth,30n) == True{} : Bool} -> @+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool} ->
  @+room: {Nat.is_le(Nat.add(start,count),U32.to_nat(capacity)) == True{} : Bool} ->
  {Lanes.rounds(~Context,~map,count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),U32.from_nat(start),slot,context}) ==
    Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,U32.from_nat(start),context,values)) : Array<F32>}

def few_rounds(~Context: Data,~map: Context -> F32 -> F32,+count: Nat,small: {Nat.is_lt(count,8n) == True{} : Bool}) -> rounds_property(~Context,~map,count):
  match count:
    case 0n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,0n,U32.from_nat(start),slot,context,values,scratch)
    case 1n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,1n,U32.from_nat(start),slot,context,values,scratch)
    case 2n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,2n,U32.from_nat(start),slot,context,values,scratch)
    case 3n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,3n,U32.from_nat(start),slot,context,values,scratch)
    case 4n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,4n,U32.from_nat(start),slot,context,values,scratch)
    case 5n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,5n,U32.from_nat(start),slot,context,values,scratch)
    case 6n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,6n,U32.from_nat(start),slot,context,values,scratch)
    case 7n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,7n,U32.from_nat(start),slot,context,values,scratch)
    case 1n+1n+1n+1n+1n+1n+1n+1n+ +rest:
      start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      Empty.absurd({Lanes.rounds(~Context,~map,8n+rest,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),U32.from_nat(start),slot,context}) ==
    Tree.pack(F32,MapReadback.run(~F32,~Context,~map,8n+rest,U32.from_nat(start),context,values)) : Array<F32>},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),small)))

def more_rounds(~Context: Data,~map: Context -> F32 -> F32,+rest: Nat,previous: rounds_property(~Context,~map,rest)) -> rounds_property(~Context,~map,8n+rest):
  start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
  +position = U32.from_nat(start)
  +mapped = MapReadback.run(~F32,~Context,~map,8n,position,context,values)
  +next = Tree.write(Tile.Tile8,scratch,slot,cell(~Context,~map,values,position,context))
  +turn = U32.and((slot + 1 : U32),63)
  %Equal.sym(Cells.State<Array<F32>,Tile.Tile8,Context>,
    Cells.step(~Array<F32>,~Tile.Tile8,~Context,~Lanes.gather(~Context,~map),~Lanes.scatter,~8,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}),
    Cells.State{Tree.pack(F32,mapped),Tree.pack(Tile.Tile8,next),(position + 8 : U32),turn,context},
    round(~Context,~map,values,scratch,start,slot,context,depth,capacity,capacity_equal,balanced,limit,capacity_ok,round_room(start,rest,U32.to_nat(capacity),room))) :
    {Lanes.rounds(~Context,~map,rest,_) == Tree.pack(F32,MapReadback.run(~F32,~Context,~map,8n+rest,position,context,values)) : Array<F32>}
  %Equal.sym(U32,(position + 8 : U32),U32.from_nat(8n+start),after_round(start)) :
    {Lanes.rounds(~Context,~map,rest,Cells.State{Tree.pack(F32,mapped),Tree.pack(Tile.Tile8,next),_,turn,context}) ==
      Tree.pack(F32,MapReadback.run(~F32,~Context,~map,8n+rest,position,context,values)) : Array<F32>}
  %Equal.sym(U32,stepped(8n,start),U32.from_nat(8n+start),advanced(8n,start)) :
    {Lanes.rounds(~Context,~map,rest,Cells.State{Tree.pack(F32,mapped),Tree.pack(Tile.Tile8,next),U32.from_nat(8n+start),turn,context}) ==
      Tree.pack(F32,MapReadback.run(~F32,~Context,~map,rest,_,context,mapped)) : Array<F32>}
  previous(8n+start,turn,context,mapped,next,depth,capacity,
    Equal.trans(U32,Tree.capacity(F32,mapped),Tree.capacity(F32,values),capacity,MapReadback.capacity(~F32,~Context,~map,8n,position,context,values),capacity_equal),
    MapReadback.balanced(~F32,~Context,~map,8n,position,context,values,depth,balanced),limit,capacity_ok,rest_room(start,rest,U32.to_nat(capacity),room))

def rounds(~Context: Data,~map: Context -> F32 -> F32,count: Nat,+start: Nat,+slot: U32,+context: Context,+values: Tree.Tree<F32>,+scratch: Tree.Tree<Tile.Tile8>,
  +depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(F32,values) == capacity : U32},+balanced: Shape.Balanced(F32,depth,values),
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,count),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Lanes.rounds(~Context,~map,count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),U32.from_nat(start),slot,context}) ==
    Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,U32.from_nat(start),context,values)) : Array<F32>}:
  by_rounds(~rounds_property(~Context,~map),~few_rounds(~Context,~map),~more_rounds(~Context,~map),count)(start,slot,context,values,scratch,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room)

# The whole map from any position, with a fresh scratch array.
def map(~Context: Data,~map: Context -> F32 -> F32,+count: Nat,+start: Nat,+context: Context,+values: Tree.Tree<F32>,+depth: Nat,+balanced: Shape.Balanced(F32,depth,values),
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},+room: {Nat.is_le(Nat.add(start,count),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}) ->
  {Lanes.map(~Context,~map,count,U32.from_nat(start),context,Tree.pack(F32,values)) ==
    Traversal.map_in_place(~F32,~Context,~map,count,U32.from_nat(start),context,Tree.pack(F32,values)) : Array<F32>}:
  +position = U32.from_nat(start)
  %Equal.sym(Array<F32>,Traversal.map_in_place(~F32,~Context,~map,count,position,context,Tree.pack(F32,values)),Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,position,context,values)),
    scalar(~Context,~map,count,position,context,values)) :
    {Lanes.map(~Context,~map,count,position,context,Tree.pack(F32,values)) == _ : Array<F32>}
  %Equal.sym(Array<Tile.Tile8>,Array.new(Tile.Tile8,6n,Tile.filled(0.0)),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0))),StorageLaws.fresh(Tile.Tile8,6n,Tile.filled(0.0))) :
    {Lanes.rounds(~Context,~map,count,Cells.State{Tree.pack(F32,values),_,position,0,context}) == Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,position,context,values)) : Array<F32>}
  rounds(~Context,~map,count,start,0,context,values,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0)),depth,U32.shln(1,depth),Shape.capacity(F32,depth,values,balanced),balanced,limit,
    Power.native_power_bound(depth,Order.successor(depth,30n,limit)),room)
