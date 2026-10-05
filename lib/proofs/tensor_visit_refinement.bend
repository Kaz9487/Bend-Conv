# Concrete cursor reads discharge the generic visit's semantic premise.
import Base
import ../tensor_cursor.bend as Cursor
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_refinement.bend as CursorLaws
import ./traversal_visit_refinement.bend as Visit
import ./traversal_array_refinement.bend as ArrayLaws
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ../kernel_block.bend as Block

def sample(~Element: Data,tree: Tree.Tree<Element>,coordinates: CursorLaws.Coordinates) -> Element:
  CursorLaws.Coordinates{axes,offset,valid} = coordinates
  Tree.read(Element,tree,U32.from_nat(CursorModel.address(axes,offset)))

def read_matches(~Element: Data,+tree: Tree.Tree<Element>,+coordinates: CursorLaws.Coordinates) ->
  {Cursor.read(~Element,Tree.pack(Element,tree),CursorLaws.pack(coordinates)) ==
    (Tree.pack(Element,tree),Traversal.Visited{CursorLaws.pack(CursorLaws.next(coordinates)),sample(~Element,tree,coordinates)}) : Array<Element> & Traversal.Visited<Cursor.Cursor,Element>}:
  CursorLaws.Coordinates{+axes,+offset,+valid} = coordinates
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,tree),U32.from_nat(CursorModel.address(axes,offset))),
    (Tree.pack(Element,tree),Tree.read(Element,tree,U32.from_nat(CursorModel.address(axes,offset)))),StorageLaws.read(Element,tree,U32.from_nat(CursorModel.address(axes,offset)))) :
    {Cursor.read_value(~Element,CursorLaws.represent_axes(axes),U32.from_nat(CursorModel.address(axes,offset)),_) ==
      (Tree.pack(Element,tree),Traversal.Visited{CursorLaws.pack(CursorLaws.next(CursorLaws.Coordinates{axes,offset,valid})),
        sample(~Element,tree,CursorLaws.Coordinates{axes,offset,valid})}) : Array<Element> & Traversal.Visited<Cursor.Cursor,Element>}
  %Equal.sym(Cursor.Cursor,Cursor.advance(CursorLaws.represent_axes(axes),U32.from_nat(CursorModel.address(axes,offset))),CursorLaws.represent(CursorModel.advance(axes),offset),
    CursorLaws.advance(axes,offset,valid)) :
    {(Tree.pack(Element,tree),Traversal.Visited{_,Tree.read(Element,tree,U32.from_nat(CursorModel.address(axes,offset)))}) ==
      (Tree.pack(Element,tree),Traversal.Visited{CursorLaws.pack(CursorLaws.next(CursorLaws.Coordinates{axes,offset,valid})),
        sample(~Element,tree,CursorLaws.Coordinates{axes,offset,valid})}) : Array<Element> & Traversal.Visited<Cursor.Cursor,Element>}
  {==}

def visit(~Element: Data,+count: Nat,+state: Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>) ->
  {Traversal.visit(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),count,
      Visit.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack,state)) ==
    Visit.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
        ~Visit.model_step(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,~sample(~Element),~CursorLaws.next),count,state)) :
    Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>}:
  Visit.visit(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack,
    ~Cursor.read(~Element),~sample(~Element),~CursorLaws.next,~read_matches(~Element),count,state)

def address(coordinates: CursorLaws.Coordinates) -> U32:
  CursorLaws.Coordinates{axes,offset,valid} = coordinates
  U32.from_nat(CursorModel.address(axes,offset))

def target_matches(+coordinates: CursorLaws.Coordinates) ->
  {Cursor.target(CursorLaws.pack(coordinates)) == Traversal.Target{CursorLaws.pack(CursorLaws.next(coordinates)),address(coordinates)} : Traversal.Target<Cursor.Cursor>}:
  CursorLaws.Coordinates{+axes,+offset,+valid} = coordinates
  %Equal.sym(Cursor.Cursor,Cursor.advance(CursorLaws.represent_axes(axes),U32.from_nat(CursorModel.address(axes,offset))),CursorLaws.represent(CursorModel.advance(axes),offset),
    CursorLaws.advance(axes,offset,valid)) :
    {Traversal.Target{_,U32.from_nat(CursorModel.address(axes,offset))} ==
      Traversal.Target{CursorLaws.pack(CursorLaws.next(CursorLaws.Coordinates{axes,offset,valid})),address(CursorLaws.Coordinates{axes,offset,valid})} : Traversal.Target<Cursor.Cursor>}
  {==}

def spread(~Element: Data,+count: Nat,+state: Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>) ->
  {Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~Traversal.spread_step(~Element,~Cursor.Cursor,~Cursor.target),count,
      Visit.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack,state)) ==
    Visit.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
        ~Visit.model_spread_step(~Element,~CursorLaws.Coordinates,~address,~CursorLaws.next),count,state)) :
    Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>}:
  Visit.spread(~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~CursorLaws.pack,~Cursor.target,~address,~CursorLaws.next,~target_matches,count,state)

def update(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,
  +state: Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>) ->
  {Traversal.visit_update(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation,count,
      Visit.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack,state)) ==
    Visit.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
        ~Visit.model_update_step(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,~sample(~Element),~CursorLaws.next,~operation),count,state)) :
    Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>}:
  Visit.update(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack,
    ~Cursor.read(~Element),~sample(~Element),~CursorLaws.next,~operation,~read_matches(~Element),count,state)

def fold(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,+count: Nat,
  +state: Traversal.State<&2,&2,Tree.Tree<Element>,Accumulator,CursorLaws.Coordinates>) ->
  {Traversal.visit_fold(~Array<Element>,~Element,~Accumulator,~Cursor.Cursor,~Cursor.read(~Element),~operation,count,
      Visit.represent_fold(~Array<Element>,~Tree.Tree<Element>,~Accumulator,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack,state)) ==
    Visit.represent_fold(~Array<Element>,~Tree.Tree<Element>,~Accumulator,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Accumulator,CursorLaws.Coordinates>,
        ~Visit.model_fold_step(~Tree.Tree<Element>,~Element,~Accumulator,~CursorLaws.Coordinates,~sample(~Element),~CursorLaws.next,~operation),count,state)) :
    Traversal.State<&1,&2,Array<Element>,Accumulator,Cursor.Cursor>}:
  Visit.fold(~Array<Element>,~Tree.Tree<Element>,~Element,~Accumulator,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack,
    ~Cursor.read(~Element),~sample(~Element),~CursorLaws.next,~operation,~read_matches(~Element),count,state)
