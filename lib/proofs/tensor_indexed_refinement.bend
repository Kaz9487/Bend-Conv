# Actual indexed traversal, including its context-dependent index, refines the
# common sequential store model. No readback or caller-schedule premise.
import Base
import ../tensor_indexed.bend as Indexed
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_access_refinement.bend as AccessLaws
import ./traversal_indexed_writes.bend as TraversalIndexedWrites
import ./write_program.bend as Writes
import ./loop_projection.bend as Projection
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./boolean_logic.bend as Boolean

def sample(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  input: Tree.Tree<Element>,context: Context,position: U32) -> Element:
  +position = position
  operation(index(context,position),Tree.read(Element,input,position))

def read(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  +input: Tree.Tree<Element>,+context: Context,+position: U32) ->
  {Indexed.read(~Element,~Context,~index,~operation,Tree.pack(Element,input),context,position) ==
    (Tree.pack(Element,input),sample(~Element,~Context,~index,~operation,input,context,position)) : Array<Element> & Element}:
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),position),
    (Tree.pack(Element,input),Tree.read(Element,input,position)),StorageLaws.read(Element,input,position)) :
    {Indexed.value(~Element,~Context,~index,~operation,context,position,_) ==
      (Tree.pack(Element,input),sample(~Element,~Context,~index,~operation,input,context,position)) : Array<Element> & Element}
  {==}

def state(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  count: Nat,context: Context,input: Tree.Tree<Element>,output: Tree.Tree<Element>) ->
  Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>:
  Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,
    ~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),
    count,Traversal.State{input,output,0,context})

def run(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  +count: Nat,+context: Context,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Indexed.run(~Element,~Context,~index,~operation,Indexed.Call{count,context},Tree.pack(Element,input),Tree.pack(Element,output)) ==
    AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),
      AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,state(~Element,~Context,~index,~operation,count,context,input,output))) : Array<Element> & Array<Element>}:
  AccessLaws.finish_relation(~Array<Element>,~Tree.Tree<Element>,~Element,~Context,~Context,~ArrayLaws.pack_tree(~Element),
    ~ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Context,~ArrayLaws.pack_tree(~Element)),
    ~AccessLaws.indexed_finished(~Array<Element>,~Tree.Tree<Element>,~Element,~Context,~ArrayLaws.pack_tree(~Element)),
    Traversal.run(~Array<Element>,~Element,~Context,~Indexed.read(~Element,~Context,~index,~operation),~Indexed.destination(~Context),count,
      Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),0,context}),
    state(~Element,~Context,~index,~operation,count,context,input,output),
    ArrayLaws.traversal_refines_model(~Array<Element>,~Tree.Tree<Element>,~Element,~Context,~ArrayLaws.pack_tree(~Element),
      ~Indexed.read(~Element,~Context,~index,~operation),~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context),
      ~read(~Element,~Context,~index,~operation),count,Traversal.State{input,output,0,context}))

def writes(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,count: Nat,context: Context,input: Tree.Tree<Element>) ->
  List<&2,Writes.Assignment<Element>>:
  TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),
    count,0,TraversalIndexedWrites.Input{input,context})

def run_writes(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  +count: Nat,+context: Context,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Indexed.run(~Element,~Context,~index,~operation,Indexed.Call{count,context},Tree.pack(Element,input),Tree.pack(Element,output)) ==
    (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,~Context,~index,~operation,count,context,input),output))) : Array<Element> & Array<Element>}:
  Equal.trans(Array<Element> & Array<Element>,Indexed.run(~Element,~Context,~index,~operation,Indexed.Call{count,context},Tree.pack(Element,input),Tree.pack(Element,output)),
    AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,state(~Element,~Context,~index,~operation,count,context,input,output))),
    (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,~Context,~index,~operation,count,context,input),output))),run(~Element,~Context,~index,~operation,count,context,input,output),
    Equal.cong(AccessLaws.ModelResult<Tree.Tree<Element>,Element>,Array<Element> & Array<Element>,result => AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),result),
      AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,state(~Element,~Context,~index,~operation,count,context,input,output)),
      AccessLaws.ModelResult{input,Writes.pure(Element,writes(~Element,~Context,~index,~operation,count,context,input),output)},
      TraversalIndexedWrites.loop(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),count,0,input,context,output)))

# The blocks of run_tiled, as models. One block is the inner model loop with
# the block's own context; the next block starts from its final position.
def model_next(~Element: Data,~Context: Data,block: U32,size: U32,context: Context,state: Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Block>) -> Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>:
  Traversal.State{input,output,position,inner} = state
  Traversal.State{input,output,position,Indexed.Blocks{(block + 1 : U32),size,context}}

def model_block_step(~Element: Data,~Context: Data,~base: Context -> U32 -> U32,~operation: U32 -> Element -> Element,state: Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>) -> Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>:
  Traversal.State{input,output,+position,Indexed.Blocks{+block,+size,+context}} = state
  model_next(~Element,~Context,block,size,context,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Block>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation),~Indexed.destination(~Indexed.Block)),U32.to_nat(size),Traversal.State{input,output,position,Indexed.Block{base(context,block),position}}))

def next_represented(~Element: Data,~Context: Data,+block: U32,+size: U32,+context: Context,+state: Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Block>) ->
  {Indexed.next_block(~Element,~Context,block,size,context,ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Block,~ArrayLaws.pack_tree(~Element),state)) ==
    ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,~ArrayLaws.pack_tree(~Element),model_next(~Element,~Context,block,size,context,state)) : Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Blocks<Context>>}:
  Traversal.State{input,output,position,inner} = state
  {==}

def block_step_refines(~Element: Data,~Context: Data,~base: Context -> U32 -> U32,~operation: U32 -> Element -> Element,+state: Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>) ->
  {Indexed.block_step(~Element,~Context,~base,~operation,ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,~ArrayLaws.pack_tree(~Element),state)) ==
    ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,~ArrayLaws.pack_tree(~Element),model_block_step(~Element,~Context,~base,~operation,state)) : Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Blocks<Context>>}:
  Traversal.State{+input,+output,+position,Indexed.Blocks{+block,+size,+context}} = state
  %Equal.sym(Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Block>,
    Traversal.run(~Array<Element>,~Element,~Indexed.Block,~Indexed.read(~Element,~Indexed.Block,~Indexed.block_index,~operation),~Indexed.destination(~Indexed.Block),U32.to_nat(size),
      ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Block,~ArrayLaws.pack_tree(~Element),Traversal.State{input,output,position,Indexed.Block{base(context,block),position}})),
    ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Block,~ArrayLaws.pack_tree(~Element),Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Block>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation),~Indexed.destination(~Indexed.Block)),U32.to_nat(size),Traversal.State{input,output,position,Indexed.Block{base(context,block),position}})),
    ArrayLaws.traversal_refines_model(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Block,~ArrayLaws.pack_tree(~Element),
      ~Indexed.read(~Element,~Indexed.Block,~Indexed.block_index,~operation),~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation),~Indexed.destination(~Indexed.Block),
      ~read(~Element,~Indexed.Block,~Indexed.block_index,~operation),U32.to_nat(size),Traversal.State{input,output,position,Indexed.Block{base(context,block),position}})) :
    {Indexed.next_block(~Element,~Context,block,size,context,_) ==
      ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,~ArrayLaws.pack_tree(~Element),model_block_step(~Element,~Context,~base,~operation,Traversal.State{input,output,position,Indexed.Blocks{block,size,context}})) : Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Blocks<Context>>}
  next_represented(~Element,~Context,block,size,context,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Block>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation),~Indexed.destination(~Indexed.Block)),U32.to_nat(size),Traversal.State{input,output,position,Indexed.Block{base(context,block),position}}))

# Where count model steps from position start leave the state: the same
# reader and context, the ordered writes, and position start + count.
def run_shape(~Element: Data,~Inner: Data,~sample: Tree.Tree<Element> -> Inner -> U32 -> Element,count: Nat,+start: Nat,+input: Tree.Tree<Element>,+context: Inner,+output: Tree.Tree<Element>) ->
  {Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Inner,~sample,~Indexed.destination(~Inner)),count,Traversal.State{input,output,U32.from_nat(start),context}) ==
    Traversal.State{input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Inner>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Inner,~sample),count,U32.from_nat(start),TraversalIndexedWrites.Input{input,context}),output),U32.from_nat(Nat.add(start,count)),context} : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Inner>}:
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) :
        {Traversal.State{input,output,U32.from_nat(start),context} == Traversal.State{input,output,U32.from_nat(_),context} : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Inner>}
      {==}
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(start) + 1 : U32),U32.from_nat(1n+start),Bounds.add_one(U32.from_nat(start))) :
        {Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Inner,~sample,~Indexed.destination(~Inner)),rest,Traversal.State{input,Tree.write(Element,output,U32.from_nat(start),sample(input,context,U32.from_nat(start))),_,context}) ==
          Traversal.State{input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Inner>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Inner,~sample),rest,_,TraversalIndexedWrites.Input{input,context}),Tree.write(Element,output,U32.from_nat(start),sample(input,context,U32.from_nat(start)))),U32.from_nat(Nat.add(start,1n+rest)),context} : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Inner>}
      %Equal.sym(Nat,Nat.add(start,1n+rest),Nat.add(1n+start,rest),Addition.successor_right(start,rest)) :
        {Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Inner,~sample,~Indexed.destination(~Inner)),rest,Traversal.State{input,Tree.write(Element,output,U32.from_nat(start),sample(input,context,U32.from_nat(start))),U32.from_nat(1n+start),context}) ==
          Traversal.State{input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Inner>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Inner,~sample),rest,U32.from_nat(1n+start),TraversalIndexedWrites.Input{input,context}),Tree.write(Element,output,U32.from_nat(start),sample(input,context,U32.from_nat(start)))),U32.from_nat(_),context} : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Inner>}
      run_shape(~Element,~Inner,~sample,rest,1n+start,input,context,Tree.write(Element,output,U32.from_nat(start),sample(input,context,U32.from_nat(start))))

# Inside one block the block's index agrees with the index function, so both
# traversals write the same values at the same positions.
def block_writes(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~size: Context -> U32,~operation: U32 -> Element -> Element,
  ~agree: @+context: Context -> @+total: U32 -> @+block: Nat -> @+offset: Nat ->
    {Nat.is_lt(offset,U32.to_nat(size(context))) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool} ->
    {U32.is_le(total,Indexed.block_limit()) == True{} : Bool} ->
    {Indexed.block_index(Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) == index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) : U32},
  count: Nat,+offset: Nat,+block: Nat,+total: U32,+context: Context,+input: Tree.Tree<Element>,
  fits: {Nat.is_le(Nat.add(offset,count),U32.to_nat(size(context))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool},+bounded: {U32.is_le(total,Indexed.block_limit()) == True{} : Bool}) ->
  {TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Indexed.Block>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation)),count,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),TraversalIndexedWrites.Input{input,Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))}}) == TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),count,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),TraversalIndexedWrites.Input{input,context}) : List<&2,Writes.Assignment<Element>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +fits = fits
      %Equal.sym(U32,Indexed.block_index(Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),agree(context,total,block,offset,Order.room_index(offset,rest,U32.to_nat(size(context)),fits),room,bounded)) :
        {Writes.Put{U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),operation(_,Tree.read(Element,input,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))))} <> TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Indexed.Block>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation)),rest,(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)) + 1 : U32),TraversalIndexedWrites.Input{input,Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))}}) ==
          Writes.Put{U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),operation(index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),Tree.read(Element,input,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))))} <> TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),rest,(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)) + 1 : U32),TraversalIndexedWrites.Input{input,context}) : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(U32,(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)) + 1 : U32),U32.from_nat(1n+Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),Bounds.add_one(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)))) :
        {Writes.Put{U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),operation(index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),Tree.read(Element,input,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))))} <> TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Indexed.Block>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation)),rest,_,TraversalIndexedWrites.Input{input,Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))}}) ==
          Writes.Put{U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),operation(index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),Tree.read(Element,input,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))))} <> TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),rest,_,TraversalIndexedWrites.Input{input,context}) : List<&2,Writes.Assignment<Element>>}
      %Addition.successor_right(Nat.mul(block,U32.to_nat(size(context))),offset) :
        {Writes.Put{U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),operation(index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),Tree.read(Element,input,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))))} <> TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Indexed.Block>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation)),rest,U32.from_nat(_),TraversalIndexedWrites.Input{input,Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))}}) ==
          Writes.Put{U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),operation(index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),Tree.read(Element,input,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))))} <> TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),rest,U32.from_nat(_),TraversalIndexedWrites.Input{input,context}) : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(List<&2,Writes.Assignment<Element>>,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Indexed.Block>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation)),rest,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),1n+offset)),TraversalIndexedWrites.Input{input,Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))}}),TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),rest,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),1n+offset)),TraversalIndexedWrites.Input{input,context}),
        block_writes(~Element,~Context,~index,~base,~size,~operation,~agree,rest,1n+offset,block,total,context,input,Order.advance_room(offset,rest,U32.to_nat(size(context)),fits),room,bounded)) :
        {Writes.Put{U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),operation(index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),Tree.read(Element,input,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))))} <> _ ==
          Writes.Put{U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),operation(index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),Tree.read(Element,input,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))))} <> TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),rest,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),1n+offset)),TraversalIndexedWrites.Input{input,context}) : List<&2,Writes.Assignment<Element>>}
      {==}

def block_start_writes(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~size: Context -> U32,~operation: U32 -> Element -> Element,
  ~agree: @+context: Context -> @+total: U32 -> @+block: Nat -> @+offset: Nat ->
    {Nat.is_lt(offset,U32.to_nat(size(context))) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool} ->
    {U32.is_le(total,Indexed.block_limit()) == True{} : Bool} ->
    {Indexed.block_index(Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) == index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) : U32},
  +block: Nat,+total: U32,+context: Context,+input: Tree.Tree<Element>,room: {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool},bounded: {U32.is_le(total,Indexed.block_limit()) == True{} : Bool}) ->
  {TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Indexed.Block>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))}}) == TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,context}) : List<&2,Writes.Assignment<Element>>}:
  %Addition.zero_right(Nat.mul(block,U32.to_nat(size(context)))) :
    {TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Indexed.Block>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation)),U32.to_nat(size(context)),U32.from_nat(_),TraversalIndexedWrites.Input{input,Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))}}) == TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),U32.to_nat(size(context)),U32.from_nat(_),TraversalIndexedWrites.Input{input,context}) : List<&2,Writes.Assignment<Element>>}
  block_writes(~Element,~Context,~index,~base,~size,~operation,~agree,U32.to_nat(size(context)),0n,block,total,context,input,Order.reflexive(U32.to_nat(size(context))),room,bounded)

def flatten(~Element: Data,~Context: Data,state: Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>) -> Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>:
  Traversal.State{input,output,position,Indexed.Blocks{block,size,context}} = state
  Traversal.State{input,output,position,context}

def block_room(+first: Nat,+size: Nat,+rest: Nat,+limit: Nat,room: {Nat.is_le(Nat.add(first,Nat.add(size,rest)),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(first,size),rest),limit) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(Nat.add(first,size),rest),Nat.add(first,Nat.add(size,rest)),Addition.associative(first,size,rest)) : {Nat.is_le(_,limit) == True{} : Bool}
  room

def next_room(+first: Nat,+size: Nat,+rest: Nat,+limit: Nat,room: {Nat.is_le(Nat.add(first,Nat.add(size,rest)),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(size,first),rest),limit) == True{} : Bool}:
  %Order.commutative(first,size) : {Nat.is_le(Nat.add(_,rest),limit) == True{} : Bool}
  block_room(first,size,rest,limit,room)

# count blocks from block number block are count * size flat steps.
def blocks_flat(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~size: Context -> U32,~operation: U32 -> Element -> Element,
  ~agree: @+context: Context -> @+total: U32 -> @+block: Nat -> @+offset: Nat ->
    {Nat.is_lt(offset,U32.to_nat(size(context))) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool} ->
    {U32.is_le(total,Indexed.block_limit()) == True{} : Bool} ->
    {Indexed.block_index(Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) == index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) : U32},
  count: Nat,+block: Nat,+total: U32,+context: Context,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,
  +room: {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),Nat.mul(count,U32.to_nat(size(context)))),U32.to_nat(total)) == True{} : Bool},+bounded: {U32.is_le(total,Indexed.block_limit()) == True{} : Bool}) ->
  {flatten(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),count,
      Traversal.State{input,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),Indexed.Blocks{U32.from_nat(block),size(context),context}})) ==
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),Nat.mul(count,U32.to_nat(size(context))),Traversal.State{input,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),context}) : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Block>,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Block>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation),~Indexed.destination(~Indexed.Block)),U32.to_nat(size(context)),Traversal.State{input,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))}}),
        Traversal.State{input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Indexed.Block>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))}}),output),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context)))),Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))}},
        run_shape(~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation),U32.to_nat(size(context)),Nat.mul(block,U32.to_nat(size(context))),input,Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))},output)) :
        {flatten(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),rest,model_next(~Element,~Context,U32.from_nat(block),size(context),context,_))) == Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),Nat.add(U32.to_nat(size(context)),Nat.mul(rest,U32.to_nat(size(context)))),Traversal.State{input,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),context}) : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>}
      %Equal.sym(List<&2,Writes.Assignment<Element>>,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Indexed.Block>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Indexed.Block,~sample(~Element,~Indexed.Block,~Indexed.block_index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))}}),TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,context}),
        block_start_writes(~Element,~Context,~index,~base,~size,~operation,~agree,block,total,context,input,
          Order.transitive(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),Nat.add(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),Nat.mul(rest,U32.to_nat(size(context)))),U32.to_nat(total),Order.le_add(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),Nat.mul(rest,U32.to_nat(size(context)))),block_room(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context)),Nat.mul(rest,U32.to_nat(size(context))),U32.to_nat(total),room)),bounded)) :
        {flatten(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),rest,Traversal.State{input,Writes.pure(Element,_,output),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context)))),Indexed.Blocks{(U32.from_nat(block) + 1 : U32),size(context),context}})) == Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),Nat.add(U32.to_nat(size(context)),Nat.mul(rest,U32.to_nat(size(context)))),Traversal.State{input,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),context}) : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>}
      %Equal.sym(U32,(U32.from_nat(block) + 1 : U32),U32.from_nat(1n+block),Bounds.add_one(U32.from_nat(block))) :
        {flatten(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),rest,Traversal.State{input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,context}),output),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context)))),Indexed.Blocks{_,size(context),context}})) == Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),Nat.add(U32.to_nat(size(context)),Nat.mul(rest,U32.to_nat(size(context)))),Traversal.State{input,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),context}) : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>}
      %Order.commutative(U32.to_nat(size(context)),Nat.mul(block,U32.to_nat(size(context)))) :
        {flatten(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),rest,Traversal.State{input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,context}),output),U32.from_nat(_),Indexed.Blocks{U32.from_nat(1n+block),size(context),context}})) == Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),Nat.add(U32.to_nat(size(context)),Nat.mul(rest,U32.to_nat(size(context)))),Traversal.State{input,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),context}) : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>}
      %Equal.sym(Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),Nat.add(U32.to_nat(size(context)),Nat.mul(rest,U32.to_nat(size(context)))),Traversal.State{input,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),context}),Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),Nat.mul(rest,U32.to_nat(size(context))),Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),U32.to_nat(size(context)),Traversal.State{input,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),context})),
        AccessLaws.run_added(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),U32.to_nat(size(context)),Nat.mul(rest,U32.to_nat(size(context))),Traversal.State{input,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),context})) :
        {flatten(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),rest,Traversal.State{input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,context}),output),U32.from_nat(Nat.add(U32.to_nat(size(context)),Nat.mul(block,U32.to_nat(size(context))))),Indexed.Blocks{U32.from_nat(1n+block),size(context),context}})) == _ : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>}
      %Equal.sym(Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),U32.to_nat(size(context)),Traversal.State{input,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),context}),
        Traversal.State{input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,context}),output),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context)))),context},
        run_shape(~Element,~Context,~sample(~Element,~Context,~index,~operation),U32.to_nat(size(context)),Nat.mul(block,U32.to_nat(size(context))),input,context,output)) :
        {flatten(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),rest,Traversal.State{input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,context}),output),U32.from_nat(Nat.add(U32.to_nat(size(context)),Nat.mul(block,U32.to_nat(size(context))))),Indexed.Blocks{U32.from_nat(1n+block),size(context),context}})) == Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),Nat.mul(rest,U32.to_nat(size(context))),_) : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>}
      %Order.commutative(U32.to_nat(size(context)),Nat.mul(block,U32.to_nat(size(context)))) :
        {flatten(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),rest,Traversal.State{input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,context}),output),U32.from_nat(Nat.add(U32.to_nat(size(context)),Nat.mul(block,U32.to_nat(size(context))))),Indexed.Blocks{U32.from_nat(1n+block),size(context),context}})) ==
          Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),Nat.mul(rest,U32.to_nat(size(context))),Traversal.State{input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,context}),output),U32.from_nat(_),context}) : Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>}
      blocks_flat(~Element,~Context,~index,~base,~size,~operation,~agree,rest,1n+block,total,context,input,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Context>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation)),U32.to_nat(size(context)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),TraversalIndexedWrites.Input{input,context}),output),next_room(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context)),Nat.mul(rest,U32.to_nat(size(context))),U32.to_nat(total),room),bounded)

def flat_finish(~Element: Data,~Context: Data,+state: Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>) ->
  {AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,state) == AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,flatten(~Element,~Context,state)) : AccessLaws.ModelResult<Tree.Tree<Element>,Element>}:
  Traversal.State{input,output,position,Indexed.Blocks{block,size,context}} = state
  {==}

def whole_room(+product: Nat,+limit: Nat,exact: {product == limit : Nat}) -> {Nat.is_le(product,limit) == True{} : Bool}:
  %exact : {Nat.is_le(product,_) == True{} : Bool}
  Order.reflexive(product)

def model_equal(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~blocks: Context -> U32,~size: Context -> U32,~operation: U32 -> Element -> Element,
  ~agree: @+context: Context -> @+total: U32 -> @+block: Nat -> @+offset: Nat ->
    {Nat.is_lt(offset,U32.to_nat(size(context))) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool} ->
    {U32.is_le(total,Indexed.block_limit()) == True{} : Bool} ->
    {Indexed.block_index(Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) == index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) : U32},
  +total: U32,+context: Context,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+exact: {Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))) == U32.to_nat(total) : Nat},bounded: {U32.is_le(total,Indexed.block_limit()) == True{} : Bool}) ->
  {AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}})) == AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,state(~Element,~Context,~index,~operation,U32.to_nat(total),context,input,output)) : AccessLaws.ModelResult<Tree.Tree<Element>,Element>}:
  %exact : {AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}})) ==
    AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),_,Traversal.State{input,output,0,context})) : AccessLaws.ModelResult<Tree.Tree<Element>,Element>}
  Equal.trans(AccessLaws.ModelResult<Tree.Tree<Element>,Element>,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}})),
    AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,flatten(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}}))),
    AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))),Traversal.State{input,output,0,context})),
    flat_finish(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}})),
    Equal.cong(Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,AccessLaws.ModelResult<Tree.Tree<Element>,Element>,result => AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,result),
      flatten(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}})),Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Context,~sample(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))),Traversal.State{input,output,0,context}),
      blocks_flat(~Element,~Context,~index,~base,~size,~operation,~agree,U32.to_nat(blocks(context)),0n,total,context,input,output,whole_room(Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))),U32.to_nat(total),exact),bounded)))

def blocks_run(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~blocks: Context -> U32,~size: Context -> U32,~operation: U32 -> Element -> Element,
  ~agree: @+context: Context -> @+total: U32 -> @+block: Nat -> @+offset: Nat ->
    {Nat.is_lt(offset,U32.to_nat(size(context))) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool} ->
    {U32.is_le(total,Indexed.block_limit()) == True{} : Bool} ->
    {Indexed.block_index(Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) == index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) : U32},
  +total: U32,+context: Context,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,exact: {Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))) == U32.to_nat(total) : Nat},bounded: {U32.is_le(total,Indexed.block_limit()) == True{} : Bool}) ->
  {Indexed.run_blocks(~Element,~Context,~base,~operation,U32.to_nat(blocks(context)),size(context),context,Tree.pack(Element,input),Tree.pack(Element,output)) == Indexed.run(~Element,~Context,~index,~operation,Indexed.Call{U32.to_nat(total),context},Tree.pack(Element,input),Tree.pack(Element,output)) : Array<Element> & Array<Element>}:
  %Equal.sym(Array<Element> & Array<Element>,Indexed.run(~Element,~Context,~index,~operation,Indexed.Call{U32.to_nat(total),context},Tree.pack(Element,input),Tree.pack(Element,output)),AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,state(~Element,~Context,~index,~operation,U32.to_nat(total),context,input,output))),
    run(~Element,~Context,~index,~operation,U32.to_nat(total),context,input,output)) :
    {Indexed.run_blocks(~Element,~Context,~base,~operation,U32.to_nat(blocks(context)),size(context),context,Tree.pack(Element,input),Tree.pack(Element,output)) == _ : Array<Element> & Array<Element>}
  %Equal.sym(Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Blocks<Context>>,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Blocks<Context>>,~Indexed.block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,~ArrayLaws.pack_tree(~Element),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}})),ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,~ArrayLaws.pack_tree(~Element),Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}})),
    Projection.iteration_refines_model(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Blocks<Context>>,~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~Indexed.block_step(~Element,~Context,~base,~operation),~model_block_step(~Element,~Context,~base,~operation),
      ~ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,~ArrayLaws.pack_tree(~Element)),~block_step_refines(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}})) :
    {Traversal.finish(~Array<Element>,~Element,~Indexed.Blocks<Context>,_) == AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,state(~Element,~Context,~index,~operation,U32.to_nat(total),context,input,output))) : Array<Element> & Array<Element>}
  %Equal.sym(Array<Element> & Array<Element>,Traversal.finish(~Array<Element>,~Element,~Indexed.Blocks<Context>,ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,~ArrayLaws.pack_tree(~Element),Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}}))),AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}}))),
    AccessLaws.indexed_finished(~Array<Element>,~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,~ArrayLaws.pack_tree(~Element),Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}}))) :
    {_ == AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,state(~Element,~Context,~index,~operation,U32.to_nat(total),context,input,output))) : Array<Element> & Array<Element>}
  Equal.cong(AccessLaws.ModelResult<Tree.Tree<Element>,Element>,Array<Element> & Array<Element>,result => AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),result),
    AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Indexed.Blocks<Context>,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Indexed.Blocks<Context>>,~model_block_step(~Element,~Context,~base,~operation),U32.to_nat(blocks(context)),Traversal.State{input,output,0,Indexed.Blocks{0,size(context),context}})),AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Context,state(~Element,~Context,~index,~operation,U32.to_nat(total),context,input,output)),
    model_equal(~Element,~Context,~index,~base,~blocks,~size,~operation,~agree,total,context,input,output,exact,bounded))

def whole_run(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~blocks: Context -> U32,~size: Context -> U32,~operation: U32 -> Element -> Element,
  ~agree: @+context: Context -> @+total: U32 -> @+block: Nat -> @+offset: Nat ->
    {Nat.is_lt(offset,U32.to_nat(size(context))) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool} ->
    {U32.is_le(total,Indexed.block_limit()) == True{} : Bool} ->
    {Indexed.block_index(Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) == index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) : U32},
  whole: Bool,+total: U32,+context: Context,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+same: {U32.is_le(total,Indexed.block_limit()) && Nat.is_eq(Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))),U32.to_nat(total)) == whole : Bool}) ->
  {Indexed.run_whole(~Element,~Context,~index,~base,~operation,whole,blocks(context),size(context),total,context,Tree.pack(Element,input),Tree.pack(Element,output)) == Indexed.run(~Element,~Context,~index,~operation,Indexed.Call{U32.to_nat(total),context},Tree.pack(Element,input),Tree.pack(Element,output)) : Array<Element> & Array<Element>}:
  match whole:
    case False{}: {==}
    case True{}:
      blocks_run(~Element,~Context,~index,~base,~blocks,~size,~operation,~agree,total,context,input,output,
        Order.equal_of_is_eq(Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))),U32.to_nat(total),Boolean.right(U32.is_le(total,Indexed.block_limit()),Nat.is_eq(Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))),U32.to_nat(total)),same)),
        Boolean.left(U32.is_le(total,Indexed.block_limit()),Nat.is_eq(Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))),U32.to_nat(total)),same))

# The traversal in blocks computes exactly the flat indexed traversal, for
# every count: the blocks run only when they are the whole count.
def tiled(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~blocks: Context -> U32,~size: Context -> U32,~operation: U32 -> Element -> Element,
  ~agree: @+context: Context -> @+total: U32 -> @+block: Nat -> @+offset: Nat ->
    {Nat.is_lt(offset,U32.to_nat(size(context))) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool} ->
    {U32.is_le(total,Indexed.block_limit()) == True{} : Bool} ->
    {Indexed.block_index(Indexed.Block{base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) == index(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))) : U32},
  +total: U32,+context: Context,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Indexed.run_tiled(~Element,~Context,~index,~base,~blocks,~size,~operation,total,context,Tree.pack(Element,input),Tree.pack(Element,output)) == Indexed.run(~Element,~Context,~index,~operation,Indexed.Call{U32.to_nat(total),context},Tree.pack(Element,input),Tree.pack(Element,output)) : Array<Element> & Array<Element>}:
  whole_run(~Element,~Context,~index,~base,~blocks,~size,~operation,~agree,U32.is_le(total,Indexed.block_limit()) && Nat.is_eq(Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))),U32.to_nat(total)),total,context,input,output,{==})
