# The block loop on storage trees, connected to the native array loop.
import Base
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/kernel_pipeline_buffer_shape.bend as BufferShape
import ../../lib/proofs/kernel_pipeline_activation.bend as ActivationProofs
import ../../lib/storage_buffer.bend as Storage
import ./blocked_pipeline_projection.bend as Projection
import ./microkernel_storage_execution.bend as Execution
import ./packing_storage_refinement.bend as PackingStorage
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_window.bend as Window
import ../../lib/tensor.bend as Tensors
import ../../lib/math_activation.bend as Activations
import ../../lib/traversal_array.bend as Traversal
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/kernel_gemm_writes.bend as GemmWrites
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/write_program.bend as Writes

type State is Data:
  State{input: Tree.Tree<F32>,tiles: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Tree.Tree<F32>}

def represent(state: State) -> Projection.State:
  State{input,tiles,weights,bias,output} = state
  Projection.State{Tree.pack(F32,input),Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output)}

def kept(input: Tree.Tree<F32>,state: GemmLaws.Buffers) -> State:
  GemmLaws.Buffers{tiles,weights,bias,output} = state
  State{input,tiles,weights,bias,Writes.evaluate(F32,output)}

def kept_matches(+input: Tree.Tree<F32>,+state: GemmLaws.Buffers) ->
  {Projection.kept(Tree.pack(F32,input),GemmLaws.represent(state)) == represent(kept(input,state)) : Projection.State}:
  GemmLaws.Buffers{tiles,weights,bias,output} = state
  {==}

def packed(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Tree.Tree<F32>,pair: PackingStorage.Packed) -> State:
  PackingStorage.Packed{input,tiles} = pair
  kept(input,Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step,U32.to_nat(Gemm.job_count(rows,span)),0,
    Gemm.Layout{rows,span,inner,start,positions},GemmLaws.Buffers{tiles,weights,bias,Writes.Initial{output}}))

def packed_matches(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+pair: PackingStorage.Packed) ->
  {Projection.packed(rows,positions,inner,start,span,Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),PackingStorage.pack_pair(pair)) == represent(packed(rows,positions,inner,start,span,weights,bias,output,pair)) : Projection.State}:
  PackingStorage.Packed{+input,+tiles} = pair
  +state = Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step,U32.to_nat(Gemm.job_count(rows,span)),0,
    Gemm.Layout{rows,span,inner,start,positions},GemmLaws.Buffers{tiles,weights,bias,Writes.Initial{output}})
  %Equal.sym(Gemm.Buffers,Gemm.run_keeping(rows,Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),span,inner,start,positions),GemmLaws.represent(state),
    GemmLaws.keeping_matches(rows,tiles,weights,bias,output,span,inner,start,positions)) :
    {Projection.kept(Tree.pack(F32,input),_) == represent(kept(input,state)) : Projection.State}
  kept_matches(input,state)

def kept_visible(input: Tree.Tree<F32>,state: GemmWrites.Visible) -> State:
  GemmWrites.Visible{tiles,weights,bias,output} = state
  State{input,tiles,weights,bias,output}

def kept_observation(+input: Tree.Tree<F32>,+state: GemmLaws.Buffers) ->
  {kept(input,state) == kept_visible(input,GemmWrites.visible(state)) : State}:
  GemmLaws.Buffers{tiles,weights,bias,output} = state
  {==}

# Each packed block returns its tile, weight and bias owners and applies its
# ordered GEMM writes directly to the full output tree.
def packed_writes(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,
  +weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+input: Tree.Tree<F32>,+tiles: Tree.Tree<Tile.Tile8>) ->
  {packed(rows,positions,inner,start,span,weights,bias,output,PackingStorage.Packed{input,tiles}) ==
    State{input,tiles,weights,bias,Writes.pure(F32,GemmWrites.all_writes(GemmWrites.job_count(rows,span),0,Gemm.Layout{rows,span,inner,start,positions},tiles,weights,bias),output)} : State}:
  +layout = {Gemm.Layout{rows,span,inner,start,positions} : Gemm.Layout}
  +count = GemmWrites.job_count(rows,span)
  +state = Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step,count,0,layout,GemmLaws.Buffers{tiles,weights,bias,Writes.Initial{output}})
  %Equal.sym(State,kept(input,state),kept_visible(input,GemmWrites.visible(state)),kept_observation(input,state)) :
    {_ == State{input,tiles,weights,bias,Writes.pure(F32,GemmWrites.all_writes(count,0,layout,tiles,weights,bias),output)} : State}
  Equal.cong(GemmWrites.Visible,State,state => kept_visible(input,state),GemmWrites.visible(state),
    GemmWrites.Visible{tiles,weights,bias,Writes.pure(F32,GemmWrites.all_writes(count,0,layout,tiles,weights,bias),output)},
    GemmWrites.fold_visible(count,0,layout,tiles,weights,bias,Writes.Initial{output}))

def packed_result(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,
  +weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+input: Tree.Tree<F32>,+tiles: Tree.Tree<Tile.Tile8>) ->
  {packed(rows,positions,inner,start,span,weights,bias,output,PackingStorage.Packed{input,tiles}) ==
    State{input,tiles,weights,bias,GemmLaws.run_at(rows,tiles,weights,bias,output,span,inner,start,positions)} : State}:
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run_at(rows,tiles,weights,bias,output,span,inner,start,positions),
    Writes.pure(F32,GemmWrites.all_writes(GemmWrites.job_count(rows,span),0,Gemm.Layout{rows,span,inner,start,positions},tiles,weights,bias),output),
    GemmWrites.run_at_writes(rows,tiles,weights,bias,output,span,inner,start,positions)) :
    {packed(rows,positions,inner,start,span,weights,bias,output,PackingStorage.Packed{input,tiles}) == State{input,tiles,weights,bias,_} : State}
  packed_writes(rows,positions,inner,start,span,weights,bias,output,input,tiles)

def step(work: Pipeline.Work,+pack_start: U32,+start: U32,+span: U32,state: State) -> State:
  Pipeline.Work{rows,h,w,k,s,p,ow,positions,+inner} = work
  State{input,tiles,weights,bias,output} = state
  packed(rows,positions,inner,start,span,weights,bias,output,Execution.packed(Packing.Call{U32.to_nat((span / 8 * inner : U32)),pack_start,h,w,k,s,p,ow,inner},input,tiles))

def step_matches(+work: Pipeline.Work,+pack_start: U32,+start: U32,+span: U32,+state: State) ->
  {Projection.step(work,pack_start,start,span,represent(state)) == represent(step(work,pack_start,start,span,state)) : Projection.State}:
  Pipeline.Work{+rows,+h,+w,+k,+s,+p,+ow,+positions,+inner} = work
  State{+input,+tiles,+weights,+bias,+output} = state
  +call = {Packing.Call{U32.to_nat((span / 8 * inner : U32)),pack_start,h,w,k,s,p,ow,inner} : Packing.Call}
  +pair = Execution.packed(call,input,tiles)
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,Packing.run(call,Tree.pack(F32,input),Tree.pack(Tile.Tile8,tiles)),PackingStorage.pack_pair(pair),Execution.packed_matches(call,input,tiles)) :
    {Projection.packed(rows,positions,inner,start,span,Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),_) == represent(packed(rows,positions,inner,start,span,weights,bias,output,pair)) : Projection.State}
  packed_matches(rows,positions,inner,start,span,weights,bias,output,pair)

def blocks(remaining: Nat,+work: Pipeline.Work,+pack_start: U32,+start: U32,+padded: U32,+span: U32,state: State) -> State:
  match remaining:
    case 0n: state
    case 1n+rest:
      blocks(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,step(work,pack_start,start,U32.min(span,(padded - start : U32)),state))

def blocks_match(+remaining: Nat,+work: Pipeline.Work,+pack_start: U32,+start: U32,+padded: U32,+span: U32,+state: State) ->
  {Projection.blocks(remaining,work,pack_start,start,padded,span,represent(state)) == represent(blocks(remaining,work,pack_start,start,padded,span,state)) : Projection.State}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +next = step(work,pack_start,start,U32.min(span,(padded - start : U32)),state)
      %Equal.sym(Projection.State,Projection.step(work,pack_start,start,U32.min(span,(padded - start : U32)),represent(state)),represent(next),step_matches(work,pack_start,start,U32.min(span,(padded - start : U32)),state)) :
        {Projection.blocks(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,_) == represent(blocks(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,next)) : Projection.State}
      blocks_match(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,next)

def finish(~custom: F32 -> F32,activation: Activations.Activation,count: U32,state: State) -> Tree.Tree<F32>:
  State{input,tiles,weights,bias,output} = state
  ActivationProofs.activation_tree(~custom,activation,count,output)

def finish_matches(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,+state: State) ->
  {Projection.finish(~custom,activation,count,represent(state)) == Tree.pack(F32,finish(~custom,activation,count,state)) : Array<F32>}:
  State{input,tiles,weights,bias,+output} = state
  ActivationProofs.activation_model(~custom,activation,count,output)

def allocated(~Element: Data,fill: Element,count: U32) -> Tree.Tree<Element>:
  Tree.fresh(Element,Tensors.depth_for(count),fill)

def allocation_matches(~Element: Data,+fill: Element,+count: U32) ->
  {Projection.allocated(~Element,fill,count) == Tree.pack(Element,allocated(~Element,fill,count)) : Array<Element>}:
  %Equal.sym(Storage.Buffer<Element>,Tensors.buffer(~Element,fill,count),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),ActivationProofs.allocated(~Element,fill,count)) :
    {ActivationProofs.storage(~Element,_) == Tree.pack(Element,allocated(~Element,fill,count)) : Array<Element>}
  StorageLaws.fresh(Element,Tensors.depth_for(count),fill)

def initial_with_target(+target: U32,+work: Pipeline.Work,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> State:
  Pipeline.Work{rows,h,w,k,s,p,ow,positions,inner} = work
  State{input,allocated(~Tile.Tile8,Tile.filled(0.0),U32.max(target,inner)),weights,bias,allocated(~F32,0.0,(rows * positions : U32))}

def run_with_target(~custom: F32 -> F32,+target: U32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  Pipeline.Work{+rows,h,w,k,s,p,ow,+positions,inner} = work
  +padded = Window.padded(positions)
  +span = (8 * U32.max(1,(target / inner : U32)) : U32)
  finish(~custom,activation,(rows * positions : U32),blocks(U32.to_nat(((padded + span - 1) / span : U32)),work,pack_start,0,padded,span,initial_with_target(target,work,input,weights,bias)))

def initial(+work: Pipeline.Work,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> State:
  initial_with_target(Pipeline.block_tiles(),work,input,weights,bias)

def run(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  run_with_target(~custom,Pipeline.block_tiles(),activation,work,pack_start,input,weights,bias)

def run_matches(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,+input: Tree.Tree<F32>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {Projection.run(~custom,activation,work,pack_start,Tree.pack(F32,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)) == Tree.pack(F32,run(~custom,activation,work,pack_start,input,weights,bias)) : Array<F32>}:
  Pipeline.Work{+rows,h,w,k,s,p,ow,+positions,+inner} = work
  +padded = Window.padded(positions)
  +span = Pipeline.block_positions(inner)
  +remaining = U32.to_nat(((padded + span - 1) / span : U32))
  +state = initial(work,input,weights,bias)
  %Equal.sym(Array<Tile.Tile8>,Projection.allocated(~Tile.Tile8,Tile.filled(0.0),U32.max(Pipeline.block_tiles(),inner)),Tree.pack(Tile.Tile8,allocated(~Tile.Tile8,Tile.filled(0.0),U32.max(Pipeline.block_tiles(),inner))),allocation_matches(~Tile.Tile8,Tile.filled(0.0),U32.max(Pipeline.block_tiles(),inner))) :
    {Projection.finish(~custom,activation,(rows * positions : U32),Projection.blocks(remaining,work,pack_start,0,padded,span,
      Projection.State{Tree.pack(F32,input),_,Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Projection.allocated(~F32,0.0,(rows * positions : U32))})) == Tree.pack(F32,run(~custom,activation,work,pack_start,input,weights,bias)) : Array<F32>}
  %Equal.sym(Array<F32>,Projection.allocated(~F32,0.0,(rows * positions : U32)),Tree.pack(F32,allocated(~F32,0.0,(rows * positions : U32))),allocation_matches(~F32,0.0,(rows * positions : U32))) :
    {Projection.finish(~custom,activation,(rows * positions : U32),Projection.blocks(remaining,work,pack_start,0,padded,span,
      Projection.State{Tree.pack(F32,input),Tree.pack(Tile.Tile8,allocated(~Tile.Tile8,Tile.filled(0.0),U32.max(Pipeline.block_tiles(),inner))),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),_})) == Tree.pack(F32,run(~custom,activation,work,pack_start,input,weights,bias)) : Array<F32>}
  %Equal.sym(Projection.State,Projection.blocks(remaining,work,pack_start,0,padded,span,represent(state)),represent(blocks(remaining,work,pack_start,0,padded,span,state)),blocks_match(remaining,work,pack_start,0,padded,span,state)) :
    {Projection.finish(~custom,activation,(rows * positions : U32),_) == Tree.pack(F32,run(~custom,activation,work,pack_start,input,weights,bias)) : Array<F32>}
  finish_matches(~custom,activation,(rows * positions : U32),blocks(remaining,work,pack_start,0,padded,span,state))

# Array contents and the owner footprint jointly determine Buffer contents.
def contents_of_projection(buffer: Storage.Buffer<F32>,+tree: Tree.Tree<F32>,+length: Nat,
  +values: {ActivationProofs.storage(~F32,buffer) == Tree.pack(F32,tree) : Array<F32>},
  +size: {BufferShape.length(BufferShape.footprint(~F32,buffer)) == length : Nat}) ->
  {Buffer.contents(~F32,buffer) == Buffer.Content{tree,length} : Buffer.Content<F32>}:
  Storage.Buffer{array,+stored_length,Certificate.Certificate{depth,equation}} = buffer
  %Equal.sym(Array<F32>,array,Tree.pack(F32,tree),values) : {Buffer.Content{Tree.reflect(F32,_),stored_length} == Buffer.Content{tree,length} : Buffer.Content<F32>}
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,tree)),tree,StorageLaws.reflect_packed(F32,tree)) : {Buffer.Content{_,stored_length} == Buffer.Content{tree,length} : Buffer.Content<F32>}
  %Equal.sym(Nat,stored_length,length,size) : {Buffer.Content{tree,_} == Buffer.Content{tree,length} : Buffer.Content<F32>}
  {==}

def run_contents(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,
  +input: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pipeline.blocked(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    Buffer.Content{run(~custom,activation,work,pack_start,Buffer.tree(~F32,input),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),U32.to_nat(BufferShape.output_size(work))} : Buffer.Content<F32>}:
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+weight_tree,+weight_length,+weight_certificate} = weights
  Buffer.Model{+bias_tree,+bias_length,+bias_certificate} = bias
  +input = {Buffer.Model{input_tree,input_length,input_certificate} : Buffer.Model<F32>}
  +weights = {Buffer.Model{weight_tree,weight_length,weight_certificate} : Buffer.Model<Tile.Tile8>}
  +bias = {Buffer.Model{bias_tree,bias_length,bias_certificate} : Buffer.Model<F32>}
  contents_of_projection(Pipeline.blocked(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)),
    run(~custom,activation,work,pack_start,input_tree,weight_tree,bias_tree),U32.to_nat(BufferShape.output_size(work)),
    Equal.trans(Array<F32>,ActivationProofs.storage(~F32,Pipeline.blocked(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))),
      Projection.run(~custom,activation,work,pack_start,Tree.pack(F32,input_tree),Tree.pack(Tile.Tile8,weight_tree),Tree.pack(F32,bias_tree)),
      Tree.pack(F32,run(~custom,activation,work,pack_start,input_tree,weight_tree,bias_tree)),
      Projection.run_projection(~custom,activation,work,pack_start,input,weights,bias),run_matches(~custom,activation,work,pack_start,input_tree,weight_tree,bias_tree)),
    Equal.cong(BufferShape.Footprint,Nat,value => BufferShape.length(value),BufferShape.footprint(~F32,Pipeline.blocked(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))),
      BufferShape.Footprint{Tensors.depth_for(BufferShape.output_size(work)),U32.to_nat(BufferShape.output_size(work))},BufferShape.run_footprint(~custom,activation,work,pack_start,input,weights,bias)))
