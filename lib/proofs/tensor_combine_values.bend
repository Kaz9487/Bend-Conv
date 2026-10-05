# The values of the checked in-place combination every elementwise operation
# with a view shares: the left operand is the contiguous target at offset
# zero, the right operand is read through any view of the same extents. The
# run stores, at every index inside its count, the operation applied to the
# left value at that index and the right storage at the address of that index
# in the right view. The operation keeps its argument order; no algebraic law
# is assumed.
import Base
import ../tensor_view.bend as Views
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./traversal_map_readback.bend as MapReadback
import ./traversal_visit_refinement.bend as Visit
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_band_combine_leaf.bend as BandCombineLeaf
import ../tensor_access.bend as Access
import ./storage_buffer_model.bend as Buffer
import ./tensor_copy_values.bend as CopyValues
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_visit_refinement.bend as TensorVisit
import ./tensor_shape_refinement.bend as ShapeLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_interval.bend as Interval
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding

# Ordered in-place update whose right value is read through a cursor. A store
# changes only its own address; later addresses still hold their left values.
def visit_run(~Element: Data,~Cursor: Data,~sample: Tree.Tree<Element> -> Cursor -> Element,~next: Cursor -> Cursor,~operation: Element -> Element -> Element,count: Nat,position: U32,cursor: Cursor,right: Tree.Tree<Element>,left: Tree.Tree<Element>) -> Tree.Tree<Element>:
  BandCombineLeaf.readback_output(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Cursor,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Cursor>,
      ~Visit.model_update_step(~Tree.Tree<Element>,~Element,~Cursor,~sample,~next,~operation),count,Traversal.State{right,left,position,cursor})))

def visit_outside(~Element: Data,~Cursor: Data,~sample: Tree.Tree<Element> -> Cursor -> Element,~next: Cursor -> Cursor,~operation: Element -> Element -> Element,count: Nat,+position: Nat,+query: U32,+cursor: Cursor,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Element,left) == capacity : U32},+balanced: Shape.Balanced(Element,depth,left),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool},
  +query_inside: {U32.is_lt(query,capacity) == True{} : Bool},+separate: Interval.Outside<position,count,U32.to_nat(query)>) ->
  {Tree.read(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,count,U32.from_nat(position),cursor,right,left),query) == Tree.read(Element,left,query) : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +value = operation(Tree.read(Element,left,U32.from_nat(position)),sample(right,cursor))
      +written = Tree.write(Element,left,U32.from_nat(position),value)
      +position_inside = Order.room_index(position,rest,U32.to_nat(capacity),room)
      +different = {
        %Equal.sym(Bool,U32.is_eq(U32.from_nat(position),query),Nat.is_eq(U32.to_nat(U32.from_nat(position)),U32.to_nat(query)),Comparison.eq(U32.from_nat(position),query)) : {_ == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) :
          {Nat.is_eq(_,U32.to_nat(query)) == False{} : Bool}
        %Order.equal_symmetric(U32.to_nat(query),position) : {_ == False{} : Bool}
        Interval.outside_head(position,rest,U32.to_nat(query),separate) : {U32.is_eq(U32.from_nat(position),query) == False{} : Bool}}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,rest,_,next(cursor),right,written),query) == Tree.read(Element,left,query) : Element}
      Equal.trans(Element,Tree.read(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,rest,U32.from_nat(1n+position),next(cursor),right,written),query),Tree.read(Element,written,query),Tree.read(Element,left,query),
        visit_outside(~Element,~Cursor,~sample,~next,~operation,rest,1n+position,query,next(cursor),right,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,left),capacity,StorageReadback.write_capacity(~Element,left,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,left,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),query_inside,
          Interval.outside_tail(position,rest,U32.to_nat(query),separate)),
        MapReadback.unchanged(~Element,depth,left,U32.from_nat(position),query,value,capacity,capacity_equal,balanced,limit,MapReadback.address_inside(position,capacity,capacity_ok,position_inside),query_inside,different))

def visit_inside(~Element: Data,~Cursor: Data,~sample: Tree.Tree<Element> -> Cursor -> Element,~next: Cursor -> Cursor,~operation: Element -> Element -> Element,count: Nat,+position: Nat,index: Nat,+cursor: Cursor,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Element,left) == capacity : U32},+balanced: Shape.Balanced(Element,depth,left),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool},
  +selected: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tree.read(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,count,U32.from_nat(position),cursor,right,left),U32.from_nat(Nat.add(position,index))) ==
    operation(Tree.read(Element,left,U32.from_nat(Nat.add(position,index))),sample(right,Loop.run(~Cursor,~next,index,cursor))) : Element}:
  match count index:
    case 0n +index:
      Empty.absurd({Tree.read(Element,left,U32.from_nat(Nat.add(position,index))) ==
        operation(Tree.read(Element,left,U32.from_nat(Nat.add(position,index))),sample(right,Loop.run(~Cursor,~next,index,cursor))) : Element},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),selected)))
    case 1n+ +rest 0n:
      +value = operation(Tree.read(Element,left,U32.from_nat(position)),sample(right,cursor))
      +written = Tree.write(Element,left,U32.from_nat(position),value)
      +position_inside = Order.room_index(position,rest,U32.to_nat(capacity),room)
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {Tree.read(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,1n+rest,U32.from_nat(position),cursor,right,left),U32.from_nat(_)) ==
          operation(Tree.read(Element,left,U32.from_nat(_)),sample(right,cursor)) : Element}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,rest,_,next(cursor),right,written),U32.from_nat(position)) == value : Element}
      Equal.trans(Element,Tree.read(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,rest,U32.from_nat(1n+position),next(cursor),right,written),U32.from_nat(position)),Tree.read(Element,written,U32.from_nat(position)),value,
        visit_outside(~Element,~Cursor,~sample,~next,~operation,rest,1n+position,U32.from_nat(position),next(cursor),right,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,left),capacity,StorageReadback.write_capacity(~Element,left,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,left,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),MapReadback.address_inside(position,capacity,capacity_ok,position_inside),
          Interval.Before{
            %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) : {Nat.is_lt(_,1n+position) == True{} : Bool}
            Order.lt_succ(position,position,Order.reflexive(position))}),
        StorageReadback.read_written(~Element,left,U32.from_nat(position),value))
    case 1n+ +rest 1n+ +index:
      +value = operation(Tree.read(Element,left,U32.from_nat(position)),sample(right,cursor))
      +written = Tree.write(Element,left,U32.from_nat(position),value)
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
        {Tree.read(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,1n+rest,U32.from_nat(position),cursor,right,left),U32.from_nat(_)) ==
          operation(Tree.read(Element,left,U32.from_nat(_)),sample(right,Loop.run(~Cursor,~next,index,next(cursor)))) : Element}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,rest,_,next(cursor),right,written),query) == operation(Tree.read(Element,left,query),sample(right,Loop.run(~Cursor,~next,index,next(cursor)))) : Element}
      Equal.trans(Element,Tree.read(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,rest,U32.from_nat(1n+position),next(cursor),right,written),query),operation(Tree.read(Element,written,query),sample(right,Loop.run(~Cursor,~next,index,next(cursor)))),
        operation(Tree.read(Element,left,query),sample(right,Loop.run(~Cursor,~next,index,next(cursor)))),
        visit_inside(~Element,~Cursor,~sample,~next,~operation,rest,1n+position,index,next(cursor),right,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,left),capacity,StorageReadback.write_capacity(~Element,left,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,left,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),selected),
        Equal.cong(Element,Element,element => operation(element,sample(right,Loop.run(~Cursor,~next,index,next(cursor)))),Tree.read(Element,written,query),Tree.read(Element,left,query),
          MapReadback.unchanged(~Element,depth,left,U32.from_nat(position),query,value,capacity,capacity_equal,balanced,limit,MapReadback.address_inside(position,capacity,capacity_ok,position_inside),MapReadback.address_inside(query_position,capacity,capacity_ok,query_inside),different)))

def visit_capacity(~Element: Data,~Cursor: Data,~sample: Tree.Tree<Element> -> Cursor -> Element,~next: Cursor -> Cursor,~operation: Element -> Element -> Element,count: Nat,+position: U32,+cursor: Cursor,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>) ->
  {Tree.capacity(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,count,position,cursor,right,left)) == Tree.capacity(Element,left) : U32}:
  match count:
    case 0n: {==}
    case 1n+rest:
      +value = operation(Tree.read(Element,left,position),sample(right,cursor))
      +written = Tree.write(Element,left,position,value)
      Equal.trans(U32,Tree.capacity(Element,visit_run(~Element,~Cursor,~sample,~next,~operation,rest,(position + 1 : U32),next(cursor),right,written)),Tree.capacity(Element,written),Tree.capacity(Element,left),
        visit_capacity(~Element,~Cursor,~sample,~next,~operation,rest,(position + 1 : U32),next(cursor),right,written),StorageReadback.write_capacity(~Element,left,position,value))

def visit_balanced(~Element: Data,~Cursor: Data,~sample: Tree.Tree<Element> -> Cursor -> Element,~next: Cursor -> Cursor,~operation: Element -> Element -> Element,count: Nat,+position: U32,+cursor: Cursor,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>,+depth: Nat,
  evidence: Shape.Balanced(Element,depth,left)) -> Shape.Balanced(Element,depth,visit_run(~Element,~Cursor,~sample,~next,~operation,count,position,cursor,right,left)):
  match count:
    case 0n: evidence
    case 1n+rest:
      +value = operation(Tree.read(Element,left,position),sample(right,cursor))
      visit_balanced(~Element,~Cursor,~sample,~next,~operation,rest,(position + 1 : U32),next(cursor),right,Tree.write(Element,left,position,value),depth,Shape.write(Element,depth,left,position,value,evidence))

# The value of one index of the checked combination into a left operand at
# offset zero: the fast path (a contiguous right view at offset zero) and the
# strided path give the same statement.
def combined_value(~Element: Data,~operation: Element -> Element -> Element,fast: Bool,+source: Views.View,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>,+length: Nat,+count: Nat,+index: Nat,
  +classified: {Access.zero_contiguous(source) == fast : Bool},
  ready: Buffer.ContentReady<Element,Buffer.Content{left,length}>,+capacity_ok: {U32.is_le(Tree.capacity(Element,left),2147483648) == True{} : Bool},
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,left))) == True{} : Bool},+inside: {Nat.is_lt(index,count) == True{} : Bool},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == count : Nat}) ->
  {Tree.read(Element,BandCombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,fast,count,source,0,right,left)),U32.from_nat(index)) ==
    operation(Tree.read(Element,left,U32.from_nat(index)),Tree.read(Element,right,CopyValues.address(source,index))) : Element}:
  match fast:
    case True{}:
      Buffer.ContentReady{+depth,+balanced,+depth_ok,content_room} = ready
      %Equal.sym(U32,CopyValues.address(source,index),U32.from_nat(index),CopyValues.zero_contiguous_address(source,index,classified,
        %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents_of(source)),count,exact) : {Nat.is_lt(index,_) == True{} : Bool} inside)) :
        {Tree.read(Element,BandCombineLeaf.run(~Element,~operation,count,0,right,left),U32.from_nat(index)) ==
          operation(Tree.read(Element,left,U32.from_nat(index)),Tree.read(Element,right,_)) : Element}
      BandCombineLeaf.inside(~Element,~operation,count,0n,index,right,left,depth,Tree.capacity(Element,left),{==},balanced,depth_ok,capacity_ok,room,inside)
    case False{}:
      Buffer.ContentReady{+depth,+balanced,+depth_ok,content_room} = ready
      %CopyValues.sample_address(~Element,source,right,index) :
        {Tree.read(Element,visit_run(~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,~operation,count,0,CursorLaws.initial(source),right,left),U32.from_nat(index)) ==
          operation(Tree.read(Element,left,U32.from_nat(index)),_) : Element}
      visit_inside(~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,~operation,count,0n,index,CursorLaws.initial(source),right,left,depth,Tree.capacity(Element,left),
        {==},balanced,depth_ok,capacity_ok,room,inside)

# The combination keeps the depth, the balance and the room of its left operand.
def combined_ready(~Element: Data,~operation: Element -> Element -> Element,fast: Bool,+count: Nat,+source: Views.View,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>,+length: Nat,
  ready: Buffer.ContentReady<Element,Buffer.Content{left,length}>) ->
  Buffer.ContentReady<Element,Buffer.Content{BandCombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,fast,count,source,0,right,left)),length}>:
  match fast:
    case True{}:
      Buffer.ContentReady{+depth,balanced,depth_ok,room} = ready
      Buffer.ContentReady{depth,BandCombineLeaf.balanced(~Element,~operation,count,0,right,left,depth,balanced),depth_ok,
        %Equal.sym(U32,Tree.capacity(Element,BandCombineLeaf.run(~Element,~operation,count,0,right,left)),Tree.capacity(Element,left),BandCombineLeaf.capacity(~Element,~operation,count,0,right,left)) :
          {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
        room}
    case False{}:
      Buffer.ContentReady{+depth,balanced,depth_ok,room} = ready
      Buffer.ContentReady{depth,visit_balanced(~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,~operation,count,0,CursorLaws.initial(source),right,left,depth,balanced),depth_ok,
        %Equal.sym(U32,Tree.capacity(Element,visit_run(~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,~operation,count,0,CursorLaws.initial(source),right,left)),Tree.capacity(Element,left),
          visit_capacity(~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,~operation,count,0,CursorLaws.initial(source),right,left)) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
        room}

# The right operand is only read: the model returns it unchanged.
def reader(~Element: Data,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) -> Tree.Tree<Element>:
  AccessLaws.ModelResult{reader,output} = result
  reader

def visit_reader(~Element: Data,~Cursor: Data,~sample: Tree.Tree<Element> -> Cursor -> Element,~next: Cursor -> Cursor,~operation: Element -> Element -> Element,count: Nat,+position: U32,+cursor: Cursor,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>) ->
  {reader(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Cursor,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Cursor>,
      ~Visit.model_update_step(~Tree.Tree<Element>,~Element,~Cursor,~sample,~next,~operation),count,Traversal.State{right,left,position,cursor}))) == right : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      visit_reader(~Element,~Cursor,~sample,~next,~operation,rest,(position + 1 : U32),next(cursor),right,Tree.write(Element,left,position,operation(Tree.read(Element,left,position),sample(right,cursor))))

def fast_reader(~Element: Data,~operation: Element -> Element -> Element,count: Nat,+position: U32,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>) ->
  {reader(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Unit,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Unit>,
      ~ArrayLaws.model_update_step(~Tree.Tree<Element>,~Element,~Element,~Unit,~AccessLaws.sample_index(~Element),~operation,~Traversal.index),
      count,Traversal.State{right,left,position,Unit{}}))) == right : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      fast_reader(~Element,~operation,rest,(position + 1 : U32),right,Tree.write(Element,left,position,operation(Tree.read(Element,left,position),Tree.read(Element,right,position))))

def combined_reader(~Element: Data,~operation: Element -> Element -> Element,fast: Bool,+count: Nat,+source: Views.View,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>) ->
  {reader(~Element,AccessLaws.combine_model(~Element,~operation,fast,count,source,0,right,left)) == right : Tree.Tree<Element>}:
  match fast:
    case True{}: fast_reader(~Element,~operation,count,0,right,left)
    case False{}: visit_reader(~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,~operation,count,0,CursorLaws.initial(source),right,left)
