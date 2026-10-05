# An immutable reader and an ordered destination cursor produce this exact
# write program. Indexed maps and other contiguous output traversals share it.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../tensor_indexed.bend as Indexed
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_access_refinement.bend as AccessLaws
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./traversal_map_readback.bend as MapReadback
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_division.bend as Division
import ./word_encoding.bend as WordEncoding
import ./boolean_logic.bend as Boolean
import ./contiguous_copy_readback.bend as CopyReadback

type Input<-Reader: Data,-Context: Data> is Data:
  Input{reader: Reader,context: Context}

def value(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,input: Input<Reader,Context>,position: U32) -> Element:
  Input{reader,context} = input
  sample(reader,context,position)

def writes(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,count: Nat,+position: U32,+context: Context) -> List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+rest: Writes.Put{position,value(context,position)} <> writes(~Element,~Context,~value,rest,(position + 1 : U32),context)

def loop(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,count: Nat,+position: U32,+reader: Reader,+context: Context,+output: Tree.Tree<Element>) ->
  {AccessLaws.finish_model(~Reader,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Context>,
    ~ArrayLaws.model_step(~Reader,~Element,~Context,~sample,~Indexed.destination(~Context)),count,Traversal.State{reader,output,position,context})) ==
    AccessLaws.ModelResult{reader,Writes.pure(Element,writes(~Element,~Input<Reader,Context>,~value(~Reader,~Element,~Context,~sample),count,position,Input{reader,context}),output)} : AccessLaws.ModelResult<Reader,Element>}:
  match count:
    case 0n: {==}
    case 1n+rest: loop(~Reader,~Element,~Context,~sample,rest,(position + 1 : U32),reader,context,Tree.write(Element,output,position,sample(reader,context,position)))

def inside(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,count: Nat,+position: Nat,+context: Context,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool}) ->
  Readback.Inside(~Element,capacity,writes(~Element,~Context,~value,count,U32.from_nat(position),context)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +room = room
      Shape.Both{MapReadback.address_inside(position,capacity,capacity_ok,Order.room_index(position,rest,U32.to_nat(capacity),room)),
        %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) : Readback.Inside(~Element,capacity,writes(~Element,~Context,~value,rest,_,context))
        inside(~Element,~Context,~value,rest,1n+position,context,capacity,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room))}

def matching(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,count: Nat,+position: U32,+context: Context,+query: U32) ->
  Readback.Matching(~Element,query,value(context,query),writes(~Element,~Context,~value,count,position,context)):
  match count:
    case 0n: Unit{}
    case 1n+rest:
      Shape.Both{Readback.value_matches(~Element,~Context,~value,context,position,query,value(context,position),U32.is_eq(position,query),{==},{==}),
        matching(~Element,~Context,~value,rest,(position + 1 : U32),context,query)}

def contains(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,count: Nat,+position: Nat,index: Nat,+context: Context,
  selected: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Readback.contains(~Element,U32.from_nat(Nat.add(position,index)),writes(~Element,~Context,~value,count,U32.from_nat(position),context)) == True{} : Bool}:
  match count index:
    case 0n +index:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),selected)))
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {Readback.contains(~Element,U32.from_nat(_),writes(~Element,~Context,~value,1n+rest,U32.from_nat(position),context)) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(U32.from_nat(position),U32.from_nat(position)),True{},CopyReadback.same_address(U32.from_nat(position))) :
        {_ || Readback.contains(~Element,U32.from_nat(position),writes(~Element,~Context,~value,rest,(U32.from_nat(position) + 1 : U32),context)) == True{} : Bool}
      {==}
    case 1n+ +rest 1n+ +index:
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {U32.is_eq(U32.from_nat(position),U32.from_nat(Nat.add(position,1n+index))) || Readback.contains(~Element,U32.from_nat(Nat.add(position,1n+index)),writes(~Element,~Context,~value,rest,_,context)) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(position,1n+index),Nat.add(1n+position,index),Addition.successor_right(position,index)) :
        {U32.is_eq(U32.from_nat(position),U32.from_nat(_)) || Readback.contains(~Element,U32.from_nat(_),writes(~Element,~Context,~value,rest,U32.from_nat(1n+position),context)) == True{} : Bool}
      %Equal.sym(Bool,Readback.contains(~Element,U32.from_nat(Nat.add(1n+position,index)),writes(~Element,~Context,~value,rest,U32.from_nat(1n+position),context)),True{},contains(~Element,~Context,~value,rest,1n+position,index,context,selected)) :
        {U32.is_eq(U32.from_nat(position),U32.from_nat(Nat.add(1n+position,index))) || _ == True{} : Bool}
      Boolean.or_true_right(U32.is_eq(U32.from_nat(position),U32.from_nat(Nat.add(1n+position,index))))

def read(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+count: Nat,+context: Context,+output: Tree.Tree<Element>,+depth: Nat,+index: Nat,
  balanced: Shape.Balanced(Element,depth,output),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +selected: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,writes(~Element,~Context,~value,count,0,context),output),U32.from_nat(index)) == value(context,U32.from_nat(index)) : Element}:
  Readback.covered(~Element,writes(~Element,~Context,~value,count,0,context),depth,output,U32.from_nat(index),value(context,U32.from_nat(index)),balanced,depth_ok,
    MapReadback.address_inside(index,Tree.capacity(Element,output),capacity_ok,Bounds.nat_trans(index,count,U32.to_nat(Tree.capacity(Element,output)),selected,room)),
    inside(~Element,~Context,~value,count,0n,context,Tree.capacity(Element,output),capacity_ok,room),matching(~Element,~Context,~value,count,0,context,U32.from_nat(index)),
    contains(~Element,~Context,~value,count,0n,index,context,selected))
