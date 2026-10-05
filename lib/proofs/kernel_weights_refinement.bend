# The actual block traversal refines a row-major read / packed write model.
# Coordinate coverage is separate from this unconditional execution equation.
import Base
import ../kernel_weights.bend as Weights
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../storage_allocation.bend as Allocation
import ../tensor.bend as Tensors
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_allocation_refinement.bend as AllocationLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./kernel_weights_readback.bend as WeightsReadback

type Owners is Data:
  Owners{input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>}

def represent(owners: Owners) -> Array<F32> & Array<Tile.Tile8>:
  Owners{input,output} = owners
  (Tree.pack(F32,input),Tree.pack(Tile.Tile8,output))

def finish(state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.Context>) -> Owners:
  Traversal.State{input,output,index,context} = state
  Owners{input,output}

def finished(+state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.Context>) ->
  {Traversal.finish(~Array<F32>,~Tile.Tile8,~Weights.Context,
      ArrayLaws.represent(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Weights.Context,~ArrayLaws.pack_tree(~F32),state)) ==
    represent(finish(state)) : Array<F32> & Array<Tile.Tile8>}:
  Traversal.State{input,output,index,context} = state
  {==}

def block(block: U32,layout: Weights.Layout,owners: Owners) -> Owners:
  Weights.Layout{rows,+inner} = layout
  Owners{input,output} = owners
  +block = block
  finish(Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.Context>,
    ~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Weights.Context,~WeightsReadback.sample,~Weights.destination),U32.to_nat(inner),
    Traversal.State{input,output,0,Weights.Context{rows,inner,(block * 8 : U32),(block * inner : U32)}}))

def block_matches(+index: U32,+layout: Weights.Layout,+owners: Owners) ->
  {Weights.copy_block(index,layout,represent(owners)) == represent(block(index,layout,owners)) : Array<F32> & Array<Tile.Tile8>}:
  Weights.Layout{+rows,+inner} = layout
  Owners{+input,+output} = owners
  +state = {Traversal.State{input,output,0,Weights.Context{rows,inner,(index * 8 : U32),(index * inner : U32)}} :
    Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.Context>}
  +model = Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.Context>,
    ~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Weights.Context,~WeightsReadback.sample,~Weights.destination),U32.to_nat(inner),state)
  %Equal.sym(Traversal.State<&1,&1,Array<F32>,Array<Tile.Tile8>,Weights.Context>,
    Traversal.run(~Array<F32>,~Tile.Tile8,~Weights.Context,~Weights.read,~Weights.destination,U32.to_nat(inner),
      ArrayLaws.represent(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Weights.Context,~ArrayLaws.pack_tree(~F32),state)),
    ArrayLaws.represent(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Weights.Context,~ArrayLaws.pack_tree(~F32),model),
    ArrayLaws.traversal_refines_model(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Weights.Context,~ArrayLaws.pack_tree(~F32),
      ~Weights.read,~WeightsReadback.sample,~Weights.destination,~WeightsReadback.read,U32.to_nat(inner),state)) :
    {Traversal.finish(~Array<F32>,~Tile.Tile8,~Weights.Context,_) == represent(finish(model)) : Array<F32> & Array<Tile.Tile8>}
  finished(model)

def output(owners: Owners) -> Tree.Tree<Tile.Tile8>:
  Owners{input,output} = owners
  output

def output_matches(+owners: Owners) ->
  {Weights.reordered(represent(owners)) == Tree.pack(Tile.Tile8,output(owners)) : Array<Tile.Tile8>}:
  Owners{input,output} = owners
  {==}

def run(+rows: U32,+inner: U32,input: Tree.Tree<F32>) -> Tree.Tree<Tile.Tile8>:
  +blocks = ((rows + 7) / 8 : U32)
  output(Traversal.indexed_fold(~Owners,~Weights.Layout,~block,U32.to_nat(blocks),0,Weights.Layout{rows,inner},
    Owners{input,Tree.fresh(Tile.Tile8,Tensors.depth_for((blocks * inner : U32)),Tile.filled(0.0))}))

def allocated(+depth: Nat) ->
  {Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0)) == Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))) : Array<Tile.Tile8>}:
  Equal.trans(Array<Tile.Tile8>,Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0)),Array.new(Tile.Tile8,depth,Tile.filled(0.0)),
    Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))),AllocationLaws.array_same(Tile.Tile8,depth,Tile.filled(0.0)),
    StorageLaws.fresh(Tile.Tile8,depth,Tile.filled(0.0)))

def run_matches(+rows: U32,+inner: U32,+input: Tree.Tree<F32>) ->
  {Weights.run(rows,inner,Tree.pack(F32,input)) == Tree.pack(Tile.Tile8,run(rows,inner,input)) : Array<Tile.Tile8>}:
  +blocks = ((rows + 7) / 8 : U32)
  +depth = Tensors.depth_for((blocks * inner : U32))
  +layout = {Weights.Layout{rows,inner} : Weights.Layout}
  +owners = {Owners{input,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))} : Owners}
  +result = Traversal.indexed_fold(~Owners,~Weights.Layout,~block,U32.to_nat(blocks),0,layout,owners)
  %Equal.sym(Array<Tile.Tile8>,Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0)),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))),allocated(depth)) :
    {Weights.reordered(Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~Weights.Layout,~Weights.copy_block,U32.to_nat(blocks),0,layout,
      (Tree.pack(F32,input),_))) == Tree.pack(Tile.Tile8,output(result)) : Array<Tile.Tile8>}
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,
    Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~Weights.Layout,~Weights.copy_block,U32.to_nat(blocks),0,layout,represent(owners)),
    represent(result),ArrayLaws.indexed_fold_refines_model(~(Array<F32> & Array<Tile.Tile8>),~Owners,~Weights.Layout,
      ~Weights.copy_block,~block,~represent,~block_matches,U32.to_nat(blocks),0,layout,owners)) :
    {Weights.reordered(_) == Tree.pack(Tile.Tile8,output(result)) : Array<Tile.Tile8>}
  output_matches(result)
