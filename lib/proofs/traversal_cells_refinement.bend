# One round of Cells.map on packed trees, for every element and cell type:
# the cell passes through the scratch array unchanged, so the round is what
# gather and scatter do to the values.
import Base
import ../traversal_cells.bend as Cells
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_readback.bend as StorageReadback

# Reading the just-written scratch slot returns its cell at every slot.
# Owner projections can use this fact without identifying their certificates.
def scratch_readback(~Cell: Data,+scratch: Tree.Tree<Cell>,+slot: U32,+cell: Cell) ->
  {Array.get(Cell,Array.set(Cell,Tree.pack(Cell,scratch),slot,cell),slot) ==
    (Tree.pack(Cell,Tree.write(Cell,scratch,slot,cell)),cell) : Array<Cell> & Cell}:
  +written = Tree.write(Cell,scratch,slot,cell)
  %Equal.sym(Array<Cell>,Array.set(Cell,Tree.pack(Cell,scratch),slot,cell),Tree.pack(Cell,written),StorageLaws.write(Cell,scratch,slot,cell)) :
    {Array.get(Cell,_,slot) == (Tree.pack(Cell,written),cell) : Array<Cell> & Cell}
  %Equal.sym(Array<Cell> & Cell,Array.get(Cell,Tree.pack(Cell,written),slot),(Tree.pack(Cell,written),Tree.read(Cell,written,slot)),StorageLaws.read(Cell,written,slot)) :
    {_ == (Tree.pack(Cell,written),cell) : Array<Cell> & Cell}
  %Equal.sym(Cell,Tree.read(Cell,written,slot),cell,StorageReadback.read_written(~Cell,scratch,slot,cell)) :
    {(Tree.pack(Cell,written),_) == (Tree.pack(Cell,written),cell) : Array<Cell> & Cell}
  {==}

def round_with(~Owner: Type,~Cell: Data,~Context: Data,~gather: Owner -> U32 -> Context -> Owner & Cell,
  ~scatter: Owner -> U32 -> Cell -> Owner,~width: U32,
  -values: Owner,+scratch: Tree.Tree<Cell>,+position: U32,+slot: U32,+context: Context,+cell: Cell,-result: Owner,
  gathered: {gather(values,position,context) == (values,cell) : Owner & Cell},
  scattered: {scatter(values,position,cell) == result : Owner}) ->
  {Cells.step(~Owner,~Cell,~Context,~gather,~scatter,~width,Cells.State{values,Tree.pack(Cell,scratch),position,slot,context}) ==
    Cells.State{result,Tree.pack(Cell,Tree.write(Cell,scratch,slot,cell)),(position + width : U32),U32.and((slot + 1 : U32),63),context} :
    Cells.State<Owner,Cell,Context>}:
  +written = Tree.write(Cell,scratch,slot,cell)
  %Equal.sym(Owner & Cell,gather(values,position,context),(values,cell),gathered) :
    {Cells.stored(~Owner,~Cell,~Context,~scatter,~width,position,slot,context,Tree.pack(Cell,scratch),_) ==
      Cells.State{result,Tree.pack(Cell,written),(position + width : U32),U32.and((slot + 1 : U32),63),context} : Cells.State<Owner,Cell,Context>}
  %Equal.sym(Array<Cell> & Cell,Array.get(Cell,Array.set(Cell,Tree.pack(Cell,scratch),slot,cell),slot),(Tree.pack(Cell,written),cell),scratch_readback(~Cell,scratch,slot,cell)) :
    {Cells.scattered(~Owner,~Cell,~Context,~scatter,~width,position,slot,context,values,_) ==
      Cells.State{result,Tree.pack(Cell,written),(position + width : U32),U32.and((slot + 1 : U32),63),context} : Cells.State<Owner,Cell,Context>}
  %Equal.sym(Owner,scatter(values,position,cell),result,scattered) :
    {Cells.State{_,Tree.pack(Cell,written),(position + width : U32),U32.and((slot + 1 : U32),63),context} ==
      Cells.State{result,Tree.pack(Cell,written),(position + width : U32),U32.and((slot + 1 : U32),63),context} : Cells.State<Owner,Cell,Context>}
  {==}

def round(~Element: Data,~Cell: Data,~Context: Data,~gather: Array<Element> -> U32 -> Context -> Array<Element> & Cell,
  ~scatter: Array<Element> -> U32 -> Cell -> Array<Element>,~width: U32,
  +values: Tree.Tree<Element>,+scratch: Tree.Tree<Cell>,+position: U32,+slot: U32,+context: Context,+cell: Cell,+result: Tree.Tree<Element>,
  gathered: {gather(Tree.pack(Element,values),position,context) == (Tree.pack(Element,values),cell) : Array<Element> & Cell},
  scattered: {scatter(Tree.pack(Element,values),position,cell) == Tree.pack(Element,result) : Array<Element>}) ->
  {Cells.step(~Array<Element>,~Cell,~Context,~gather,~scatter,~width,Cells.State{Tree.pack(Element,values),Tree.pack(Cell,scratch),position,slot,context}) ==
    Cells.State{Tree.pack(Element,result),Tree.pack(Cell,Tree.write(Cell,scratch,slot,cell)),(position + width : U32),U32.and((slot + 1 : U32),63),context} :
    Cells.State<Array<Element>,Cell,Context>}:
  round_with(~Array<Element>,~Cell,~Context,~gather,~scatter,~width,Tree.pack(Element,values),scratch,position,slot,context,cell,Tree.pack(Element,result),gathered,scattered)

type Observed<-Model: Type,-Cell: Data,-Context: Data> is Type:
  Observed{values: Model,scratch: Tree.Tree<Cell>,position: U32,slot: U32,context: Context}

def observe(~Owner: Type,~Model: Type,~Cell: Data,~Context: Data,~project: Owner -> Model,state: Cells.State<Owner,Cell,Context>) -> Observed<Model,Cell,Context>:
  Cells.State{values,scratch,position,slot,context} = state
  Observed{project(values),Tree.reflect(Cell,scratch),position,slot,context}

def observe_gather(~Owner: Type,~Model: Type,~Cell: Data,~project: Owner -> Model,result: Owner & Cell) -> Model & Cell:
  (owner,cell) = result
  (project(owner),cell)

def model_stored(~Model: Type,~Cell: Data,~Context: Data,~scatter: Model -> U32 -> Cell -> Model,~width: U32,
  +position: U32,+slot: U32,context: Context,scratch: Tree.Tree<Cell>,result: Model & Cell) -> Observed<Model,Cell,Context>:
  (owner,+cell) = result
  Observed{scatter(owner,position,cell),Tree.write(Cell,scratch,slot,cell),(position + width : U32),U32.and((slot + 1 : U32),63),context}

# One shared scratch round; gather and scatter supply their numerical laws.
def stored_observed(~Owner: Type,~Model: Type,~Cell: Data,~Context: Data,~project: Owner -> Model,
  ~scatter: Owner -> U32 -> Cell -> Owner,~model_scatter: Model -> U32 -> Cell -> Model,~width: U32,
  ~scatter_matches: @owner: Owner -> @+position: U32 -> @+cell: Cell ->
    {project(scatter(owner,position,cell)) == model_scatter(project(owner),position,cell) : Model},
  +position: U32,+slot: U32,+context: Context,+scratch: Tree.Tree<Cell>,result: Owner & Cell) ->
  {observe(~Owner,~Model,~Cell,~Context,~project,Cells.stored(~Owner,~Cell,~Context,~scatter,~width,position,slot,context,Tree.pack(Cell,scratch),result)) ==
    model_stored(~Model,~Cell,~Context,~model_scatter,~width,position,slot,context,scratch,observe_gather(~Owner,~Model,~Cell,~project,result)) : Observed<Model,Cell,Context>}:
  (owner,+cell) = result
  %Equal.sym(Array<Cell> & Cell,Array.get(Cell,Array.set(Cell,Tree.pack(Cell,scratch),slot,cell),slot),
    (Tree.pack(Cell,Tree.write(Cell,scratch,slot,cell)),cell),scratch_readback(~Cell,scratch,slot,cell)) :
    {observe(~Owner,~Model,~Cell,~Context,~project,Cells.scattered(~Owner,~Cell,~Context,~scatter,~width,position,slot,context,owner,_)) ==
      model_stored(~Model,~Cell,~Context,~model_scatter,~width,position,slot,context,scratch,(project(owner),cell)) : Observed<Model,Cell,Context>}
  %Equal.sym(Tree.Tree<Cell>,Tree.reflect(Cell,Tree.pack(Cell,Tree.write(Cell,scratch,slot,cell))),Tree.write(Cell,scratch,slot,cell),StorageLaws.reflect_packed(Cell,Tree.write(Cell,scratch,slot,cell))) :
    {Observed{project(scatter(owner,position,cell)),_,(position + width : U32),U32.and((slot + 1 : U32),63),context} ==
      model_stored(~Model,~Cell,~Context,~model_scatter,~width,position,slot,context,scratch,(project(owner),cell)) : Observed<Model,Cell,Context>}
  Equal.cong(Model,Observed<Model,Cell,Context>,values => Observed{values,Tree.write(Cell,scratch,slot,cell),(position + width : U32),U32.and((slot + 1 : U32),63),context},
    project(scatter(owner,position,cell)),model_scatter(project(owner),position,cell),scatter_matches(owner,position,cell))
