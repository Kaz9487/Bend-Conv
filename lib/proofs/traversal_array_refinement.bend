# The same ownership-parameterized state is used by implementation and model.
# One read contract and one generic Array.set refinement cover every traversal.
import Base
import ../kernel_block.bend as Block
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./loop_projection.bend as Projection
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order

def represent(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~pack_reader: Model -> Reader,
  state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) -> Traversal.State<&1,&1,Reader,Array<Element>,Context>:
  Traversal.State{reader,output,position,context} = state
  Traversal.State{pack_reader(reader),Tree.pack(Element,output),position,context}

def model_step(~Model: Data,~Element: Data,~Context: Data,
  ~sample: Model -> Context -> U32 -> Element,~destination: Context -> U32 -> U32,
  state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) -> Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>:
  Traversal.State{+reader,output,+position,+context} = state
  Traversal.State{reader,Tree.write(Element,output,destination(context,position),sample(reader,context,position)),(position + 1 : U32),context}

def step_refines_model(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Model -> Context -> U32 -> Element,~destination: Context -> U32 -> U32,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {Traversal.step(~Reader,~Element,~Context,~read,~destination,represent(~Reader,~Model,~Element,~Context,~pack_reader,state)) ==
    represent(~Reader,~Model,~Element,~Context,~pack_reader,model_step(~Model,~Element,~Context,~sample,~destination,state)) :
    Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Traversal.State{+reader,+output,+position,+context} = state
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.store(~Reader,~Element,~Context,~destination,position,context,Tree.pack(Element,output),_) ==
      represent(~Reader,~Model,~Element,~Context,~pack_reader,model_step(~Model,~Element,~Context,~sample,~destination,
        Traversal.State{reader,output,position,context})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),destination(context,position),sample(reader,context,position)),
    Tree.pack(Element,Tree.write(Element,output,destination(context,position),sample(reader,context,position))),
    StorageLaws.write(Element,output,destination(context,position),sample(reader,context,position))) :
    {Traversal.State{pack_reader(reader),_,(position + 1 : U32),context} ==
      represent(~Reader,~Model,~Element,~Context,~pack_reader,model_step(~Model,~Element,~Context,~sample,~destination,
        Traversal.State{reader,output,position,context})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  {==}

def traversal_refines_model(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Model -> Context -> U32 -> Element,~destination: Context -> U32 -> U32,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +count: Nat,+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {Traversal.run(~Reader,~Element,~Context,~read,~destination,count,represent(~Reader,~Model,~Element,~Context,~pack_reader,state)) ==
    represent(~Reader,~Model,~Element,~Context,~pack_reader,
      Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~model_step(~Model,~Element,~Context,~sample,~destination),count,state)) :
    Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Projection.iteration_refines_model(~Traversal.State<&1,&1,Reader,Array<Element>,Context>,~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
    ~Traversal.step(~Reader,~Element,~Context,~read,~destination),~model_step(~Model,~Element,~Context,~sample,~destination),
    ~represent(~Reader,~Model,~Element,~Context,~pack_reader),~step_refines_model(~Reader,~Model,~Element,~Context,~pack_reader,~read,~sample,~destination,~read_matches),count,state)
# Concrete copy/zip readers discharge the generic traversal's only read premise.

def pack_tree(~Element: Data,tree: Tree.Tree<Element>) -> Array<Element>:
  Tree.pack(Element,tree)

def sample_offset(~Element: Data,tree: Tree.Tree<Element>,context: Traversal.Offsets,position: U32) -> Element:
  Traversal.Offsets{source,destination} = context
  Tree.read(Element,tree,(source + position : U32))

def offset_read_refines_storage(~Element: Data,+tree: Tree.Tree<Element>,+context: Traversal.Offsets,+position: U32) ->
  {Traversal.source_offset(~Element,pack_tree(~Element,tree),context,position) ==
    (pack_tree(~Element,tree),sample_offset(~Element,tree,context,position)) : Array<Element> & Element}:
  Traversal.Offsets{source,destination} = context
  StorageLaws.read(Element,tree,(source + position : U32))

def copy_traversal_refines_storage(~Element: Data,+count: Nat,
  +state: Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Traversal.Offsets>) ->
  {Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.source_offset(~Element),~Traversal.destination_offset,count,
      represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Traversal.Offsets,~pack_tree(~Element),state)) ==
    represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Traversal.Offsets,~pack_tree(~Element),
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Traversal.Offsets>,
        ~model_step(~Tree.Tree<Element>,~Element,~Traversal.Offsets,~sample_offset(~Element),~Traversal.destination_offset),count,state)) :
    Traversal.State<&1,&1,Array<Element>,Array<Element>,Traversal.Offsets>}:
  traversal_refines_model(~Array<Element>,~Tree.Tree<Element>,~Element,~Traversal.Offsets,~pack_tree(~Element),
    ~Traversal.source_offset(~Element),~sample_offset(~Element),~Traversal.destination_offset,~offset_read_refines_storage(~Element),count,state)

# The accumulator is an observation, independent of its numeric interpretation.
# In particular this theorem never rewrites or reassociates a floating sum.
def represent_fold(~Reader: Type,~Model: Data,~Accumulator: Data,~Context: Data,~pack_reader: Model -> Reader,
  state: Traversal.State<&2,&2,Model,Accumulator,Context>) -> Traversal.State<&1,&2,Reader,Accumulator,Context>:
  Traversal.State{reader,accumulator,position,context} = state
  Traversal.State{pack_reader(reader),accumulator,position,context}

def model_fold_step(~Model: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~sample: Model -> Context -> U32 -> Element,~combine: Accumulator -> Element -> Accumulator,
  state: Traversal.State<&2,&2,Model,Accumulator,Context>) -> Traversal.State<&2,&2,Model,Accumulator,Context>:
  Traversal.State{+reader,accumulator,+position,+context} = state
  Traversal.State{reader,combine(accumulator,sample(reader,context,position)),(position + 1 : U32),context}

def fold_step_refines_model(~Reader: Type,~Model: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Model -> Context -> U32 -> Element,~combine: Accumulator -> Element -> Accumulator,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +state: Traversal.State<&2,&2,Model,Accumulator,Context>) ->
  {Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine,
      represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,state)) ==
    represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,
      model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine,state)) : Traversal.State<&1,&2,Reader,Accumulator,Context>}:
  Traversal.State{+reader,+accumulator,+position,+context} = state
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),
    read_matches(reader,context,position)) :
    {Traversal.fold_value(~Reader,~Element,~Accumulator,~Context,~combine,accumulator,position,context,_) ==
      represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,
        model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine,Traversal.State{reader,accumulator,position,context})) :
      Traversal.State<&1,&2,Reader,Accumulator,Context>}
  {==}

def fold_refines_ordered_model(~Reader: Type,~Model: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Model -> Context -> U32 -> Element,~combine: Accumulator -> Element -> Accumulator,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +count: Nat,+state: Traversal.State<&2,&2,Model,Accumulator,Context>) ->
  {Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine),
      count,represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,state)) ==
    represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,
      Loop.run(~Traversal.State<&2,&2,Model,Accumulator,Context>,~model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine),count,state)) :
    Traversal.State<&1,&2,Reader,Accumulator,Context>}:
  Projection.iteration_refines_model(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.State<&2,&2,Model,Accumulator,Context>,
    ~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine),~model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine),
    ~represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader),
    ~fold_step_refines_model(~Reader,~Model,~Element,~Accumulator,~Context,~pack_reader,~read,~sample,~combine,~read_matches),count,state)

# In-place transforms retain one owner and use the same generic loop lifting.
def represent_map(~Element: Data,~Context: Data,
  state: Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>) -> Traversal.State<&2,&1,Unit,Array<Element>,Context>:
  Traversal.State{Unit{},output,position,context} = state
  Traversal.State{Unit{},Tree.pack(Element,output),position,context}

def unindexed(~Element: Data,~Context: Data,~map: Context -> Element -> Element,context: Context,position: U32,value: Element) -> Element:
  map(context,value)

def model_map_step(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,
  state: Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>) -> Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>:
  Traversal.State{Unit{},+output,+position,+context} = state
  Traversal.State{Unit{},Tree.write(Element,output,position,map(context,position,Tree.read(Element,output,position))),
    (position + 1 : U32),context}

def map_step_refines_model(~Element: Data,~Context: Data,~map: Context -> Element -> Element,
  +state: Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>) ->
  {Traversal.map_step(~Element,~Context,~map,represent_map(~Element,~Context,state)) ==
    represent_map(~Element,~Context,model_map_step(~Element,~Context,~unindexed(~Element,~Context,~map),state)) : Traversal.State<&2,&1,Unit,Array<Element>,Context>}:
  Traversal.State{Unit{},+output,+position,+context} = state
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,output),position),
    (Tree.pack(Element,output),Tree.read(Element,output,position)),StorageLaws.read(Element,output,position)) :
    {Traversal.mapped_value(~Element,~Context,~map,position,context,_) ==
      represent_map(~Element,~Context,model_map_step(~Element,~Context,~unindexed(~Element,~Context,~map),state)) : Traversal.State<&2,&1,Unit,Array<Element>,Context>}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),position,map(context,Tree.read(Element,output,position))),
    Tree.pack(Element,Tree.write(Element,output,position,map(context,Tree.read(Element,output,position)))),
    StorageLaws.write(Element,output,position,map(context,Tree.read(Element,output,position)))) :
    {Traversal.State{Unit{},_,(position + 1 : U32),context} ==
      represent_map(~Element,~Context,model_map_step(~Element,~Context,~unindexed(~Element,~Context,~map),state)) : Traversal.State<&2,&1,Unit,Array<Element>,Context>}
  {==}

def map_loop_refines_model(~Element: Data,~Context: Data,~map: Context -> Element -> Element,
  +count: Nat,+state: Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>) ->
  {Loop.run(~Traversal.State<&2,&1,Unit,Array<Element>,Context>,~Traversal.map_step(~Element,~Context,~map),
      count,represent_map(~Element,~Context,state)) ==
    represent_map(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>,
      ~model_map_step(~Element,~Context,~unindexed(~Element,~Context,~map)),count,state)) : Traversal.State<&2,&1,Unit,Array<Element>,Context>}:
  Projection.iteration_refines_model(~Traversal.State<&2,&1,Unit,Array<Element>,Context>,~Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>,
    ~Traversal.map_step(~Element,~Context,~map),~model_map_step(~Element,~Context,~unindexed(~Element,~Context,~map)),~represent_map(~Element,~Context),
    ~map_step_refines_model(~Element,~Context,~map),count,state)

# One representation proof covers the entire indexed output traversal. The
# strategy supplies a single-step result, not another outer-loop induction.

# In-place binary updates use the current destination value in left position.
# This theorem models sequential writes, including modular/overlapping raw
# addresses. Valid view guards and non-overlap are separate obligations.
def model_update_step(~Model: Data,~Element: Data,~Sample: Data,~Context: Data,
  ~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,~destination: Context -> U32 -> U32,
  state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) -> Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>:
  Traversal.State{+reader,+output,+position,+context} = state
  Traversal.State{reader,Tree.write(Element,output,destination(context,position),
    combine(Tree.read(Element,output,destination(context,position)),sample(reader,context,position))),(position + 1 : U32),context}

def update_step_refines_model(~Reader: Type,~Model: Data,~Element: Data,~Sample: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Sample,
  ~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,~destination: Context -> U32 -> U32,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Sample},
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {Traversal.update_step(~Reader,~Element,~Sample,~Context,~read,~combine,~destination,
      represent(~Reader,~Model,~Element,~Context,~pack_reader,state)) ==
    represent(~Reader,~Model,~Element,~Context,~pack_reader,model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,state)) :
    Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Traversal.State{+reader,+output,+position,+context} = state
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,output),destination(context,position)),
    (Tree.pack(Element,output),Tree.read(Element,output,destination(context,position))),StorageLaws.read(Element,output,destination(context,position))) :
    {Traversal.update_current(~Reader,~Element,~Sample,~Context,~read,~combine,~destination,pack_reader(reader),position,context,_) ==
      represent(~Reader,~Model,~Element,~Context,~pack_reader,model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,
        Traversal.State{reader,output,position,context})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  %Equal.sym(Reader & Sample,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.update_sample(~Reader,~Element,~Sample,~Context,~combine,~destination,position,context,Tree.pack(Element,output),
      Tree.read(Element,output,destination(context,position)),_) ==
      represent(~Reader,~Model,~Element,~Context,~pack_reader,model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,
        Traversal.State{reader,output,position,context})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),destination(context,position),
      combine(Tree.read(Element,output,destination(context,position)),sample(reader,context,position))),
    Tree.pack(Element,Tree.write(Element,output,destination(context,position),
      combine(Tree.read(Element,output,destination(context,position)),sample(reader,context,position)))),
    StorageLaws.write(Element,output,destination(context,position),combine(Tree.read(Element,output,destination(context,position)),sample(reader,context,position)))) :
    {Traversal.State{pack_reader(reader),_,(position + 1 : U32),context} ==
      represent(~Reader,~Model,~Element,~Context,~pack_reader,model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,
        Traversal.State{reader,output,position,context})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  {==}

def update_refines_model(~Reader: Type,~Model: Data,~Element: Data,~Sample: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Sample,
  ~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,~destination: Context -> U32 -> U32,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Sample},
  +count: Nat,+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {Traversal.update(~Reader,~Element,~Sample,~Context,~read,~combine,~destination,count,
      represent(~Reader,~Model,~Element,~Context,~pack_reader,state)) ==
    represent(~Reader,~Model,~Element,~Context,~pack_reader,
      Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
        ~model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),count,state)) :
    Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Projection.iteration_refines_model(~Traversal.State<&1,&1,Reader,Array<Element>,Context>,~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
    ~Traversal.update_step(~Reader,~Element,~Sample,~Context,~read,~combine,~destination),
    ~model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),
    ~represent(~Reader,~Model,~Element,~Context,~pack_reader),
    ~update_step_refines_model(~Reader,~Model,~Element,~Sample,~Context,~pack_reader,~read,~sample,~combine,~destination,~read_matches),count,state)

def indexed_fold_refines_model(~Runtime: Type,~Model: Data,~Context: Data,
  ~update: U32 -> Context -> Runtime -> Runtime,~model_update: U32 -> Context -> Model -> Model,
  ~represent: Model -> Runtime,
  ~step_matches: @+index: U32 -> @+context: Context -> @+state: Model ->
    {update(index,context,represent(state)) == represent(model_update(index,context,state)) : Runtime},
  +count: Nat,+index: U32,+context: Context,+state: Model) ->
  {Traversal.indexed_fold(~Runtime,~Context,~update,count,index,context,represent(state)) ==
    represent(Traversal.indexed_fold(~Model,~Context,~model_update,count,index,context,state)) : Runtime}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Runtime,update(index,context,represent(state)),represent(model_update(index,context,state)),step_matches(index,context,state)) :
        {Traversal.indexed_fold(~Runtime,~Context,~update,rest,(index + 1 : U32),context,_) ==
          represent(Traversal.indexed_fold(~Model,~Context,~model_update,rest,(index + 1 : U32),context,model_update(index,context,state))) : Runtime}
      indexed_fold_refines_model(~Runtime,~Model,~Context,~update,~model_update,~represent,~step_matches,
        rest,(index + 1 : U32),context,model_update(index,context,state))

# One bounded induction carries capacity, preserved sources and completed
# output values together. A strategy proves its local update obligation;
# it does not reimplement the whole output traversal proof.
def indexed_fold_change_context(~State: Data,~Outer: Data,~Inner: Data,
  ~project: Outer -> Inner,~outer_step: U32 -> Outer -> State -> State,~inner_step: U32 -> Inner -> State -> State,
  ~same_step: @+index: U32 -> @+context: Outer -> @+state: State ->
    {outer_step(index,context,state) == inner_step(index,project(context),state) : State},
  +count: Nat,+first: U32,+context: Outer,+state: State) ->
  {Traversal.indexed_fold(~State,~Outer,~outer_step,count,first,context,state) ==
    Traversal.indexed_fold(~State,~Inner,~inner_step,count,first,project(context),state) : State}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(State,outer_step(first,context,state),inner_step(first,project(context),state),same_step(first,context,state)) :
        {Traversal.indexed_fold(~State,~Outer,~outer_step,rest,(first + 1 : U32),context,_) ==
          Traversal.indexed_fold(~State,~Inner,~inner_step,rest,(first + 1 : U32),project(context),inner_step(first,project(context),state)) : State}
      indexed_fold_change_context(~State,~Outer,~Inner,~project,~outer_step,~inner_step,~same_step,
        rest,(first + 1 : U32),context,inner_step(first,project(context),state))

def indexed_fold_preserves_invariant(~State: Data,~Context: Data,
  ~limit: Context -> Nat,~Invariant: Context -> Nat -> State -> Data,
  ~update: U32 -> Context -> State -> State,
  ~preserve: @+context: Context -> @+index: Nat -> @+state: State ->
    {Nat.is_lt(index,limit(context)) == True{} : Bool} -> Invariant(context,index,state) ->
    Invariant(context,1n+index,update(U32.from_nat(index),context,state)),
  +count: Nat,+first: Nat,+context: Context,+state: State,
  +room: {Nat.is_le(Nat.add(first,count),limit(context)) == True{} : Bool},
  evidence: Invariant(context,first,state)) ->
  Invariant(context,Nat.add(first,count),Traversal.indexed_fold(~State,~Context,~update,count,U32.from_nat(first),context,state)):
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) : Invariant(context,_,state)
      evidence
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        Invariant(context,Nat.add(first,1n+rest),Traversal.indexed_fold(~State,~Context,~update,rest,_,context,update(U32.from_nat(first),context,state)))
      %Equal.sym(Nat,Nat.add(first,1n+rest),1n+Nat.add(first,rest),Addition.successor_right(first,rest)) :
        Invariant(context,_,Traversal.indexed_fold(~State,~Context,~update,rest,U32.from_nat(1n+first),context,update(U32.from_nat(first),context,state)))
      indexed_fold_preserves_invariant(~State,~Context,~limit,~Invariant,~update,~preserve,rest,1n+first,context,update(U32.from_nat(first),context,state),
        Order.advance_room(first,rest,limit(context),room),preserve(context,first,state,Order.room_index(first,rest,limit(context),room),evidence))

# Splitting a traversal preserves every state update in the same order. The
# boundary uses modular U32 conversion; no scalar/FP32-specific induction.
def indexed_fold_split(~Value: Type,~Context: Data,~update: U32 -> Context -> Value -> Value,
  +first: Nat,+second: Nat,+start: Nat,+context: Context,-initial: Value) ->
  {Traversal.indexed_fold(~Value,~Context,~update,Nat.add(first,second),U32.from_nat(start),context,initial) ==
    Traversal.indexed_fold(~Value,~Context,~update,second,U32.from_nat(Nat.add(start,first)),context,
      Traversal.indexed_fold(~Value,~Context,~update,first,U32.from_nat(start),context,initial)) : Value}:
  match first:
    case 0n:
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) :
        {Traversal.indexed_fold(~Value,~Context,~update,second,U32.from_nat(start),context,initial) ==
          Traversal.indexed_fold(~Value,~Context,~update,second,U32.from_nat(_),context,initial) : Value}
      {==}
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(start) + 1 : U32),U32.from_nat(1n+start),Bounds.add_one(U32.from_nat(start))) :
        {Traversal.indexed_fold(~Value,~Context,~update,Nat.add(rest,second),_,context,update(U32.from_nat(start),context,initial)) ==
          Traversal.indexed_fold(~Value,~Context,~update,second,U32.from_nat(Nat.add(start,1n+rest)),context,
            Traversal.indexed_fold(~Value,~Context,~update,rest,_,context,update(U32.from_nat(start),context,initial))) : Value}
      %Equal.sym(Nat,Nat.add(start,1n+rest),1n+Nat.add(start,rest),Addition.successor_right(start,rest)) :
        {Traversal.indexed_fold(~Value,~Context,~update,Nat.add(rest,second),U32.from_nat(1n+start),context,update(U32.from_nat(start),context,initial)) ==
          Traversal.indexed_fold(~Value,~Context,~update,second,U32.from_nat(_),context,
            Traversal.indexed_fold(~Value,~Context,~update,rest,U32.from_nat(1n+start),context,update(U32.from_nat(start),context,initial))) : Value}
      indexed_fold_split(~Value,~Context,~update,rest,second,1n+start,context,update(U32.from_nat(start),context,initial))
