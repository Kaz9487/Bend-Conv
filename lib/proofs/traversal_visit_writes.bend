# Cursor visits reuse the same contiguous indexed-write readback as maps.
# The indexed value is the sample at the independently iterated cursor; no
# premise assumes a completed buffer already contains the desired values.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./traversal_visit_refinement.bend as Visit
import ./traversal_indexed_writes.bend as TraversalIndexedWrites
import ./tensor_access_refinement.bend as AccessLaws
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds

def next_after(~Coordinates: Data,~next: Coordinates -> Coordinates,count: Nat,+initial: Coordinates) ->
  {Loop.run(~Coordinates,~next,1n+count,initial) == next(Loop.run(~Coordinates,~next,count,initial)) : Coordinates}:
  match count:
    case 0n: {==}
    case 1n+rest: next_after(~Coordinates,~next,rest,next(initial))

def value(~Reader: Data,~Element: Data,~Coordinates: Data,~sample: Reader -> Coordinates -> Element,~next: Coordinates -> Coordinates,
  context: TraversalIndexedWrites.Input<Reader,Coordinates>,position: U32) -> Element:
  TraversalIndexedWrites.Input{reader,initial} = context
  sample(reader,Loop.run(~Coordinates,~next,U32.to_nat(position),initial))

def encoded_value(~Reader: Data,~Element: Data,~Coordinates: Data,~sample: Reader -> Coordinates -> Element,~next: Coordinates -> Coordinates,
  +reader: Reader,+initial: Coordinates,+position: Nat,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},room: {Nat.is_le(position,U32.to_nat(capacity)) == True{} : Bool}) ->
  {value(~Reader,~Element,~Coordinates,~sample,~next,TraversalIndexedWrites.Input{reader,initial},U32.from_nat(position)) == sample(reader,Loop.run(~Coordinates,~next,position,initial)) : Element}:
  Equal.cong(Nat,Element,index => sample(reader,Loop.run(~Coordinates,~next,index,initial)),U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,room))

def loop(~Reader: Data,~Element: Data,~Coordinates: Data,~sample: Reader -> Coordinates -> Element,~next: Coordinates -> Coordinates,
  count: Nat,+position: Nat,+reader: Reader,+initial: Coordinates,+output: Tree.Tree<Element>,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool}) ->
  {AccessLaws.finish_model(~Reader,~Element,~Coordinates,Loop.run(~Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Coordinates>,~Visit.model_step(~Reader,~Element,~Coordinates,~sample,~next),count,
      Traversal.State{reader,output,U32.from_nat(position),Loop.run(~Coordinates,~next,position,initial)})) ==
    AccessLaws.ModelResult{reader,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Reader,Coordinates>,~value(~Reader,~Element,~Coordinates,~sample,~next),count,U32.from_nat(position),TraversalIndexedWrites.Input{reader,initial}),output)} : AccessLaws.ModelResult<Reader,Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +cursor = Loop.run(~Coordinates,~next,position,initial)
      +updated = Tree.write(Element,output,U32.from_nat(position),sample(reader,cursor))
      +next_position = (U32.from_nat(position) + 1 : U32)
      +context = {TraversalIndexedWrites.Input{reader,initial} : TraversalIndexedWrites.Input<Reader,Coordinates>}
      %Equal.sym(Element,value(~Reader,~Element,~Coordinates,~sample,~next,context,U32.from_nat(position)),sample(reader,cursor),
        encoded_value(~Reader,~Element,~Coordinates,~sample,~next,reader,initial,position,capacity,capacity_ok,Order.transitive(position,Nat.add(position,1n+rest),U32.to_nat(capacity),Order.le_add(position,1n+rest),room))) :
        {AccessLaws.finish_model(~Reader,~Element,~Coordinates,Loop.run(~Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Coordinates>,~Visit.model_step(~Reader,~Element,~Coordinates,~sample,~next),rest,Traversal.State{reader,updated,next_position,next(cursor)})) ==
          AccessLaws.ModelResult{reader,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Reader,Coordinates>,~value(~Reader,~Element,~Coordinates,~sample,~next),rest,next_position,context),Tree.write(Element,output,U32.from_nat(position),_))} : AccessLaws.ModelResult<Reader,Element>}
      %Equal.sym(U32,next_position,U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {AccessLaws.finish_model(~Reader,~Element,~Coordinates,Loop.run(~Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Coordinates>,~Visit.model_step(~Reader,~Element,~Coordinates,~sample,~next),rest,Traversal.State{reader,updated,_,next(cursor)})) ==
          AccessLaws.ModelResult{reader,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Reader,Coordinates>,~value(~Reader,~Element,~Coordinates,~sample,~next),rest,_,context),updated)} : AccessLaws.ModelResult<Reader,Element>}
      %next_after(~Coordinates,~next,position,initial) :
        {AccessLaws.finish_model(~Reader,~Element,~Coordinates,Loop.run(~Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Coordinates>,~Visit.model_step(~Reader,~Element,~Coordinates,~sample,~next),rest,Traversal.State{reader,updated,U32.from_nat(1n+position),_})) ==
          AccessLaws.ModelResult{reader,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Reader,Coordinates>,~value(~Reader,~Element,~Coordinates,~sample,~next),rest,U32.from_nat(1n+position),context),updated)} : AccessLaws.ModelResult<Reader,Element>}
      loop(~Reader,~Element,~Coordinates,~sample,~next,rest,1n+position,reader,initial,updated,capacity,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room))

def output(~Reader: Data,~Element: Data,result: AccessLaws.ModelResult<Reader,Element>) -> Tree.Tree<Element>:
  AccessLaws.ModelResult{reader,output} = result
  output

def read(~Reader: Data,~Element: Data,~Coordinates: Data,~sample: Reader -> Coordinates -> Element,~next: Coordinates -> Coordinates,
  +count: Nat,+reader: Reader,+initial: Coordinates,+tree: Tree.Tree<Element>,+depth: Nat,+index: Nat,
  balanced: Shape.Balanced(Element,depth,tree),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},+inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tree.read(Element,output(~Reader,~Element,AccessLaws.finish_model(~Reader,~Element,~Coordinates,
      Loop.run(~Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Coordinates>,~Visit.model_step(~Reader,~Element,~Coordinates,~sample,~next),count,Traversal.State{reader,tree,0,initial}))),U32.from_nat(index)) ==
    sample(reader,Loop.run(~Coordinates,~next,index,initial)) : Element}:
  %Equal.sym(AccessLaws.ModelResult<Reader,Element>,AccessLaws.finish_model(~Reader,~Element,~Coordinates,
      Loop.run(~Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Coordinates>,~Visit.model_step(~Reader,~Element,~Coordinates,~sample,~next),count,Traversal.State{reader,tree,0,initial})),
    AccessLaws.ModelResult{reader,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Reader,Coordinates>,~value(~Reader,~Element,~Coordinates,~sample,~next),count,0,TraversalIndexedWrites.Input{reader,initial}),tree)},
    loop(~Reader,~Element,~Coordinates,~sample,~next,count,0n,reader,initial,tree,Tree.capacity(Element,tree),capacity_ok,room)) :
    {Tree.read(Element,output(~Reader,~Element,_),U32.from_nat(index)) == sample(reader,Loop.run(~Coordinates,~next,index,initial)) : Element}
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,TraversalIndexedWrites.writes(~Element,~TraversalIndexedWrites.Input<Reader,Coordinates>,~value(~Reader,~Element,~Coordinates,~sample,~next),count,0,TraversalIndexedWrites.Input{reader,initial}),tree),U32.from_nat(index)),
    value(~Reader,~Element,~Coordinates,~sample,~next,TraversalIndexedWrites.Input{reader,initial},U32.from_nat(index)),sample(reader,Loop.run(~Coordinates,~next,index,initial)),
    TraversalIndexedWrites.read(~Element,~TraversalIndexedWrites.Input<Reader,Coordinates>,~value(~Reader,~Element,~Coordinates,~sample,~next),count,TraversalIndexedWrites.Input{reader,initial},tree,depth,index,balanced,depth_ok,capacity_ok,room,inside),
    encoded_value(~Reader,~Element,~Coordinates,~sample,~next,reader,initial,index,Tree.capacity(Element,tree),capacity_ok,Bounds.strict_le(index,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,count,U32.to_nat(Tree.capacity(Element,tree)),inside,room))))
