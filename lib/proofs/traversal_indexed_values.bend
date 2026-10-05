# An indexed store traversal writes sample(reader, context, p) at position p
# for p = 0 .. count - 1. Its output keeps capacity, balance and readiness,
# and inside the written count position p holds its own sample: the job of p
# writes it and no later job writes there. The samples never read the output.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_access_refinement.bend as AccessLaws
import ./traversal_map_readback.bend as MapReadback
import ./indexed_output_readback.bend as OutputReadback
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds

def output_tree(~Reader: Data,~Element: Data,result: AccessLaws.ModelResult<Reader,Element>) -> Tree.Tree<Element>:
  AccessLaws.ModelResult{reader,output} = result
  output

def run(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  count: Nat,position: U32,context: Context,reader: Reader,output: Tree.Tree<Element>) -> Tree.Tree<Element>:
  output_tree(~Reader,~Element,AccessLaws.finish_model(~Reader,~Element,~Context,
    Loop.run(~Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Reader,~Element,~Context,~sample,~place),
      count,Traversal.State{reader,output,position,context})))

def capacity(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  count: Nat,+position: U32,+context: Context,+reader: Reader,+output: Tree.Tree<Element>) ->
  {Tree.capacity(Element,run(~Reader,~Element,~Context,~sample,~place,count,position,context,reader,output)) == Tree.capacity(Element,output) : U32}:
  match count:
    case 0n: {==}
    case 1n+rest:
      +value = sample(reader,context,position)
      +written = Tree.write(Element,output,place(context,position),value)
      Equal.trans(U32,Tree.capacity(Element,run(~Reader,~Element,~Context,~sample,~place,rest,(position + 1 : U32),context,reader,written)),Tree.capacity(Element,written),Tree.capacity(Element,output),
        capacity(~Reader,~Element,~Context,~sample,~place,rest,(position + 1 : U32),context,reader,written),StorageReadback.write_capacity(~Element,output,place(context,position),value))

def balanced(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  count: Nat,+position: U32,+context: Context,+reader: Reader,+output: Tree.Tree<Element>,+depth: Nat,
  evidence: Shape.Balanced(Element,depth,output)) -> Shape.Balanced(Element,depth,run(~Reader,~Element,~Context,~sample,~place,count,position,context,reader,output)):
  match count:
    case 0n: evidence
    case 1n+rest:
      +value = sample(reader,context,position)
      balanced(~Reader,~Element,~Context,~sample,~place,rest,(position + 1 : U32),context,reader,Tree.write(Element,output,place(context,position),value),depth,
        Shape.write(Element,depth,output,place(context,position),value,evidence))

def ready(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  +count: Nat,+context: Context,+reader: Reader,+output: Tree.Tree<Element>,+length: Nat,
  evidence: Buffer.ContentReady<Element,Buffer.Content{output,length}>) ->
  Buffer.ContentReady<Element,Buffer.Content{run(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output),length}>:
  Buffer.ContentReady{+depth,+balance,depth_ok,room} = evidence
  Buffer.ContentReady{depth,balanced(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output,depth,balance),depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,run(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output)),Tree.capacity(Element,output),
      capacity(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output)) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
    room}

# One position of the output, asked of the whole traversal.
type Query<-Reader: Data,-Context: Data> is Data:
  Query{count: Nat,position: Nat,depth: Nat,capacity: U32,reader: Reader,context: Context}

def query_count(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> Nat:
  Query{count,position,depth,capacity,reader,context} = query
  count

def query_position(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> Nat:
  Query{count,position,depth,capacity,reader,context} = query
  position

def query_depth(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> Nat:
  Query{count,position,depth,capacity,reader,context} = query
  depth

def query_capacity(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> U32:
  Query{count,position,depth,capacity,reader,context} = query
  capacity

def query_reader(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> Reader:
  Query{count,position,depth,capacity,reader,context} = query
  reader

def query_context(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> Context:
  Query{count,position,depth,capacity,reader,context} = query
  context

def wanted(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,query: Query<Reader,Context>) -> Element:
  Query{count,position,depth,capacity,reader,context} = query
  sample(reader,context,U32.from_nat(position))

def read(~Reader: Data,~Element: Data,~Context: Data,query: Query<Reader,Context>,tree: Tree.Tree<Element>) -> Element:
  Tree.read(Element,tree,U32.from_nat(query_position(~Reader,~Context,query)))

def update(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,index: U32,query: Query<Reader,Context>,tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  Query{count,position,depth,capacity,reader,context} = query
  +index = index
  Tree.write(Element,tree,index,sample(reader,context,index))

type Frame<-Reader: Data,-Element: Data,-Context: Data,-query: Query<Reader,Context>,-tree: Tree.Tree<Element>> is Data:
  Frame{capacity_equal: {Tree.capacity(Element,tree) == query_capacity(~Reader,~Context,query) : U32},
    balanced: Shape.Balanced(Element,query_depth(~Reader,~Context,query),tree),
    depth_ok: {Nat.is_le(query_depth(~Reader,~Context,query),30n) == True{} : Bool},
    capacity_ok: {U32.is_le(query_capacity(~Reader,~Context,query),2147483648) == True{} : Bool},
    room: {Nat.is_le(query_count(~Reader,~Context,query),U32.to_nat(query_capacity(~Reader,~Context,query))) == True{} : Bool},
    selected: {Nat.is_lt(query_position(~Reader,~Context,query),query_count(~Reader,~Context,query)) == True{} : Bool}}

def frame(~Reader: Data,~Element: Data,~Context: Data,query: Query<Reader,Context>,tree: Tree.Tree<Element>) -> Data:
  Frame<Reader,Element,Context,query,tree>

def frame_step(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,+query: Query<Reader,Context>,+index: U32,+tree: Tree.Tree<Element>,
  evidence: Frame<Reader,Element,Context,query,tree>) -> Frame<Reader,Element,Context,query,update(~Reader,~Element,~Context,~sample,index,query,tree)>:
  Query{+count,+position,+depth,+capacity,+reader,+context} = query
  Frame{capacity_equal,balanced,depth_ok,capacity_ok,room,selected} = evidence
  +value = sample(reader,context,index)
  Frame{Equal.trans(U32,Tree.capacity(Element,Tree.write(Element,tree,index,value)),Tree.capacity(Element,tree),capacity,
      StorageReadback.write_capacity(~Element,tree,index,value),capacity_equal),
    Shape.write(Element,depth,tree,index,value,balanced),depth_ok,capacity_ok,room,selected}

def at_target(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,+query: Query<Reader,Context>,+tree: Tree.Tree<Element>,
  evidence: Frame<Reader,Element,Context,query,tree>) ->
  {read(~Reader,~Element,~Context,query,update(~Reader,~Element,~Context,~sample,U32.from_nat(query_position(~Reader,~Context,query)),query,tree)) ==
    wanted(~Reader,~Element,~Context,~sample,query) : Element}:
  Query{count,+position,depth,capacity,+reader,+context} = query
  StorageReadback.read_written(~Element,tree,U32.from_nat(position),sample(reader,context,U32.from_nat(position)))

def later(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,+query: Query<Reader,Context>,+index: Nat,+tree: Tree.Tree<Element>,
  evidence: Frame<Reader,Element,Context,query,tree>,after: {Nat.is_lt(query_position(~Reader,~Context,query),index) == True{} : Bool},
  inside: {Nat.is_lt(index,query_count(~Reader,~Context,query)) == True{} : Bool}) ->
  {read(~Reader,~Element,~Context,query,update(~Reader,~Element,~Context,~sample,U32.from_nat(index),query,tree)) == read(~Reader,~Element,~Context,query,tree) : Element}:
  Query{+count,+position,+depth,+capacity,+reader,+context} = query
  Frame{capacity_equal,balanced,depth_ok,+capacity_ok,+room,selected} = evidence
  +index_inside = Bounds.nat_trans(index,count,U32.to_nat(capacity),inside,room)
  +position_inside = Bounds.nat_trans(position,count,U32.to_nat(capacity),selected,room)
  +different = {
    %Equal.sym(Bool,U32.is_eq(U32.from_nat(index),U32.from_nat(position)),Nat.is_eq(U32.to_nat(U32.from_nat(index)),U32.to_nat(U32.from_nat(position))),
      Comparison.eq(U32.from_nat(index),U32.from_nat(position))) : {_ == False{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,capacity,capacity_ok,Bounds.strict_le(index,U32.to_nat(capacity),index_inside))) :
      {Nat.is_eq(_,U32.to_nat(U32.from_nat(position))) == False{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) :
      {Nat.is_eq(index,_) == False{} : Bool}
    %Order.equal_symmetric(position,index) : {_ == False{} : Bool}
    Order.less_not_equal(position,index,after) : {U32.is_eq(U32.from_nat(index),U32.from_nat(position)) == False{} : Bool}}
  MapReadback.unchanged(~Element,depth,tree,U32.from_nat(index),U32.from_nat(position),
    sample(reader,context,U32.from_nat(index)),capacity,capacity_equal,balanced,depth_ok,
    MapReadback.address_inside(index,capacity,capacity_ok,index_inside),MapReadback.address_inside(position,capacity,capacity_ok,position_inside),different)

# The traversal is the indexed fold of its stores once every job writes at
# its own position.
def folded(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  ~identity: @+context: Context -> @+position: U32 -> {place(context,position) == position : U32},
  +query: Query<Reader,Context>,remaining: Nat,+start: U32,+output: Tree.Tree<Element>) ->
  {run(~Reader,~Element,~Context,~sample,~place,remaining,start,query_context(~Reader,~Context,query),query_reader(~Reader,~Context,query),output) ==
    Traversal.indexed_fold(~Tree.Tree<Element>,~Query<Reader,Context>,~update(~Reader,~Element,~Context,~sample),remaining,start,query,output) : Tree.Tree<Element>}:
  Query{+count,+position,+depth,+capacity,+reader,+context} = query
  match remaining:
    case 0n: {==}
    case 1n+rest:
      %Equal.sym(U32,place(context,start),start,identity(context,start)) :
        {run(~Reader,~Element,~Context,~sample,~place,rest,(start + 1 : U32),context,reader,Tree.write(Element,output,_,sample(reader,context,start))) ==
          Traversal.indexed_fold(~Tree.Tree<Element>,~Query<Reader,Context>,~update(~Reader,~Element,~Context,~sample),1n+rest,start,
            Query{count,position,depth,capacity,reader,context},output) : Tree.Tree<Element>}
      folded(~Reader,~Element,~Context,~sample,~place,~identity,Query{count,position,depth,capacity,reader,context},rest,(start + 1 : U32),
        Tree.write(Element,output,start,sample(reader,context,start)))

# Inside the written count, a position of the output holds its own sample.
def stored(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  ~identity: @+context: Context -> @+position: U32 -> {place(context,position) == position : U32},
  +count: Nat,+context: Context,+reader: Reader,+output: Tree.Tree<Element>,+depth: Nat,+capacity: U32,+position: Nat,
  capacity_equal: {Tree.capacity(Element,output) == capacity : U32},balanced: Shape.Balanced(Element,depth,output),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},room: {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool},
  +selected: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {Tree.read(Element,run(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output),U32.from_nat(position)) ==
    sample(reader,context,U32.from_nat(position)) : Element}:
  +query = {Query{count,position,depth,capacity,reader,context} : Query<Reader,Context>}
  %Equal.sym(Tree.Tree<Element>,run(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output),
    Traversal.indexed_fold(~Tree.Tree<Element>,~Query<Reader,Context>,~update(~Reader,~Element,~Context,~sample),count,0,query,output),
    folded(~Reader,~Element,~Context,~sample,~place,~identity,query,count,0,output)) :
    {Tree.read(Element,_,U32.from_nat(position)) == sample(reader,context,U32.from_nat(position)) : Element}
  OutputReadback.complete_readback(~Tree.Tree<Element>,~Query<Reader,Context>,~Element,~query_count(~Reader,~Context),~query_position(~Reader,~Context),
    ~wanted(~Reader,~Element,~Context,~sample),~frame(~Reader,~Element,~Context),~read(~Reader,~Element,~Context),~update(~Reader,~Element,~Context,~sample),
    ~frame_step(~Reader,~Element,~Context,~sample),~at_target(~Reader,~Element,~Context,~sample),~later(~Reader,~Element,~Context,~sample),query,output,
    Frame{capacity_equal,balanced,depth_ok,capacity_ok,room,selected},selected)
