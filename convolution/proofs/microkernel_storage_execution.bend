# The complete pipelined Array execution refines a storage model. Its leaf
# obligations are discharged with packing, ordered GEMM and allocation proofs.
import Base
import ../../lib/kernel_array_pipeline.bend as ArrayPipeline
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_weights.bend as Weights
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_gather.bend as Gather
import ../../lib/kernel_tile.bend as Tile
import ../../lib/storage_allocation.bend as Allocation
import ../../lib/tensor.bend as Tensors
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_allocation_refinement.bend as AllocationLaws
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ./packing_storage_refinement.bend as PackingStorage
import ./gather_storage_model.bend as GatherModel
import ./gather_blocks.bend as Blocks
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/proofs/kernel_window_copy.bend as WindowCopy
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/write_program.bend as Writes

def allocated(+rows: U32,+span: U32) -> Tree.Tree<F32>:
  Tree.fresh(F32,Tensors.depth_for((rows * span : U32)),0.0)

def allocated_matches(+rows: U32,+span: U32) ->
  {ArrayPipeline.allocated(rows,span) == Tree.pack(F32,allocated(rows,span)) : Array<F32>}:
  AllocationLaws.fresh(~F32,Tensors.depth_for((rows * span : U32)),0.0)

def computed(+rows: U32,+span: U32,inner: U32,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,packed: PackingStorage.Packed) -> Tree.Tree<F32>:
  PackingStorage.Packed{input,tiles} = packed
  GemmLaws.run(rows,tiles,weights,bias,allocated(rows,span),span,inner)

def computed_matches(+rows: U32,+span: U32,+inner: U32,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+packed: PackingStorage.Packed) ->
  {ArrayPipeline.computed(rows,span,inner,Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),PackingStorage.pack_pair(packed)) ==
    Tree.pack(F32,computed(rows,span,inner,weights,bias,packed)) : Array<F32>}:
  PackingStorage.Packed{input,+tiles} = packed
  %Equal.sym(Array<F32>,ArrayPipeline.allocated(rows,span),Tree.pack(F32,allocated(rows,span)),allocated_matches(rows,span)) :
    {Gemm.run(rows,Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),_,span,inner) ==
      Tree.pack(F32,GemmLaws.run(rows,tiles,weights,bias,allocated(rows,span),span,inner)) : Array<F32>}
  GemmLaws.run_matches(rows,tiles,weights,bias,allocated(rows,span),span,inner)

def packed(call: Packing.Call,input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>) -> PackingStorage.Packed:
  Packing.Call{count,start,h,w,k,s,p,+ow,inner} = call
  PackingStorage.packing(count,start,input,output,h,w,k,s,p,ow,inner)

def packed_matches(+call: Packing.Call,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>) ->
  {Packing.run(call,Tree.pack(F32,input),Tree.pack(Tile.Tile8,output)) == PackingStorage.pack_pair(packed(call,input,output)) : Array<F32> & Array<Tile.Tile8>}:
  Packing.Call{count,start,h,w,k,s,p,+ow,inner} = call
  PackingStorage.packing_refines_storage(count,start,input,output,h,w,k,s,p,ow,inner)

# Rows [first, first + rows) of every channel, as a proof-only storage model.
type InputWindow is Data:
  InputWindow{tree: Tree.Tree<F32>,first: U32,rows: U32}

def represent(window: InputWindow) -> ArrayPipeline.InputWindow:
  InputWindow{tree,first,rows} = window
  ArrayPipeline.InputWindow{Tree.pack(F32,tree),first,rows}

def copied_tree(+channels: U32,+width: U32,+first: U32,+rows: U32,+left_first: U32,+left_rows: U32,+tree: Tree.Tree<F32>) -> Tree.Tree<F32>:
  Writes.pure(F32,WindowCopy.writes(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),((left_first - first) * width : U32),(left_rows * width : U32),tree),
    Tree.fresh(F32,Tensors.depth_for((channels * left_rows * width : U32)),0.0))

def copied(+channels: U32,+width: U32,+first: U32,+rows: U32,band: KernelWindow.Band,+tree: Tree.Tree<F32>) -> PartitionLaws.Split<InputWindow>:
  KernelWindow.Band{+left_first,+left_rows} = band
  PartitionLaws.Split{InputWindow{copied_tree(channels,width,first,rows,left_first,left_rows,tree),left_first,left_rows},InputWindow{tree,first,rows}}

def split_when(copy: Bool,+channels: U32,+width: U32,+first: U32,+rows: U32,+band: KernelWindow.Band,+tree: Tree.Tree<F32>) -> PartitionLaws.Split<InputWindow>:
  match copy:
    case False{}: PartitionLaws.Split{InputWindow{tree,first,rows},InputWindow{tree,first,rows}}
    case True{}: copied(channels,width,first,rows,band,tree)

def split(work: Pipeline.Work,offset: U32,half: U32,window: InputWindow) -> PartitionLaws.Split<InputWindow>:
  Pipeline.Work{channels_out,+height,+width,+kernel,+stride,+padding,+output_width,positions,+inner} = work
  InputWindow{tree,+first,+rows} = window
  +offset = offset
  +half = half
  +band = KernelWindow.band(first,rows,height,kernel,stride,padding,output_width,offset,half)
  split_when(U32.is_zero((output_width % 8 : U32)) && KernelWindow.covers(first,rows,height,kernel,stride,padding,output_width,offset,half,band),
    (inner / (kernel * kernel) : U32),width,first,rows,band,tree)

def copied_matches(+channels: U32,+width: U32,+first: U32,+rows: U32,+band: KernelWindow.Band,+tree: Tree.Tree<F32>) ->
  {ArrayPipeline.copied(channels,width,first,rows,band,Tree.pack(F32,tree)) ==
    PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,copied(channels,width,first,rows,band,tree)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  KernelWindow.Band{+left_first,+left_rows} = band
  +count = (channels * left_rows * width : U32)
  +fresh = Tree.fresh(F32,Tensors.depth_for(count),0.0)
  %Equal.sym(Array<F32>,Allocation.array(~F32,Tensors.depth_for(count),0.0),Tree.pack(F32,fresh),AllocationLaws.fresh(~F32,Tensors.depth_for(count),0.0)) :
    {ArrayPipeline.windows(first,rows,left_first,left_rows,KernelWindow.copy_planes(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),
      ((left_first - first) * width : U32),(left_rows * width : U32),(Tree.pack(F32,tree),_))) ==
      PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,copied(channels,width,first,rows,KernelWindow.Band{left_first,left_rows},tree)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}
  %Equal.sym(Array<F32> & Array<F32>,KernelWindow.copy_planes(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),
      ((left_first - first) * width : U32),(left_rows * width : U32),(Tree.pack(F32,tree),Tree.pack(F32,fresh))),
    (Tree.pack(F32,tree),Tree.pack(F32,copied_tree(channels,width,first,rows,left_first,left_rows,tree))),
    WindowCopy.planes(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),((left_first - first) * width : U32),(left_rows * width : U32),tree,fresh)) :
    {ArrayPipeline.windows(first,rows,left_first,left_rows,_) ==
      PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,copied(channels,width,first,rows,KernelWindow.Band{left_first,left_rows},tree)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}
  {==}

def split_when_matches(+copy: Bool,+channels: U32,+width: U32,+first: U32,+rows: U32,+band: KernelWindow.Band,+tree: Tree.Tree<F32>) ->
  {ArrayPipeline.split_when(copy,channels,width,first,rows,band,Tree.pack(F32,tree)) ==
    PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,split_when(copy,channels,width,first,rows,band,tree)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  match copy:
    case False{}:
      %Equal.sym(Array<F32> & Array<F32>,Array.clone(F32,Tree.pack(F32,tree)),(Tree.pack(F32,tree),Tree.pack(F32,tree)),StorageLaws.clone(F32,tree)) :
        {ArrayPipeline.cloned(first,rows,_) == (ArrayPipeline.InputWindow{Tree.pack(F32,tree),first,rows},ArrayPipeline.InputWindow{Tree.pack(F32,tree),first,rows}) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}
      {==}
    case True{}:
      copied_matches(channels,width,first,rows,band,tree)

def split_matches(+work: Pipeline.Work,+offset: U32,+half: U32,+window: InputWindow) ->
  {ArrayPipeline.split(work,offset,half,represent(window)) == PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,split(work,offset,half,window)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  Pipeline.Work{channels_out,+height,+width,+kernel,+stride,+padding,+output_width,positions,+inner} = work
  InputWindow{+tree,+first,+rows} = window
  +band = KernelWindow.band(first,rows,height,kernel,stride,padding,output_width,offset,half)
  split_when_matches(U32.is_zero((output_width % 8 : U32)) && KernelWindow.covers(first,rows,height,kernel,stride,padding,output_width,offset,half,band),
    (inner / (kernel * kernel) : U32),width,first,rows,band,tree)

def leaf(work: Pipeline.Work,offset: U32,span: U32,window: InputWindow,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> GatherModel.ConvResult():
  Pipeline.Work{+rows,h,w,k,+s,p,+ow,positions,+inner} = work
  InputWindow{input,+first,height} = window
  +offset = offset
  +span = span
  +count = (span / 8 * inner : U32)
  Partition.Leaf{computed(rows,span,inner,weights,bias,
    packed(Packing.Call{U32.to_nat(count),KernelWindow.start(offset,s,ow,first),height,w,k,s,p,ow,inner},input,
      Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))),offset,span,rows}

def leaf_matches(+work: Pipeline.Work,+offset: U32,+span: U32,+window: InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {ArrayPipeline.leaf(work,offset,span,represent(window),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)) ==
    PartitionLaws.represent(~F32,leaf(work,offset,span,window,weights,bias)) : Partition.Tree<&1,Array<F32>,U32>}:
  Pipeline.Work{+rows,+h,+w,+k,+s,+p,+ow,positions,+inner} = work
  InputWindow{+input,+first,+height} = window
  +count = (span / 8 * inner : U32)
  +call = {Packing.Call{U32.to_nat(count),KernelWindow.start(offset,s,ow,first),height,w,k,s,p,ow,inner} : Packing.Call}
  +tiles = Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))
  +prepared = packed(call,input,tiles)
  %Equal.sym(Array<Tile.Tile8>,Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)),Tree.pack(Tile.Tile8,tiles),
    AllocationLaws.fresh(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))) :
    {Partition.Leaf{ArrayPipeline.computed(rows,span,inner,Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Packing.run(call,Tree.pack(F32,input),_)),offset,span,rows} ==
      PartitionLaws.represent(~F32,Partition.Leaf{computed(rows,span,inner,weights,bias,prepared),offset,span,rows}) : Partition.Tree<&1,Array<F32>,U32>}
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,Packing.run(call,Tree.pack(F32,input),Tree.pack(Tile.Tile8,tiles)),PackingStorage.pack_pair(prepared),packed_matches(call,input,tiles)) :
    {Partition.Leaf{ArrayPipeline.computed(rows,span,inner,Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),_),offset,span,rows} ==
      PartitionLaws.represent(~F32,Partition.Leaf{computed(rows,span,inner,weights,bias,prepared),offset,span,rows}) : Partition.Tree<&1,Array<F32>,U32>}
  %Equal.sym(Array<F32>,ArrayPipeline.computed(rows,span,inner,Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),PackingStorage.pack_pair(prepared)),
    Tree.pack(F32,computed(rows,span,inner,weights,bias,prepared)),computed_matches(rows,span,inner,weights,bias,prepared)) :
    {Partition.Leaf{_,offset,span,rows} == PartitionLaws.represent(~F32,Partition.Leaf{computed(rows,span,inner,weights,bias,prepared),offset,span,rows}) : Partition.Tree<&1,Array<F32>,U32>}
  {==}

def partition(depth: Nat,work: Pipeline.Work,offset: U32,span: U32,input: InputWindow,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> GatherModel.ConvResult():
  PartitionLaws.partition(~InputWindow,~Tile.Tile8,~F32,~F32,~Pipeline.Work,~split,~leaf,depth,work,offset,span,input,weights,bias)

def partition_matches(+depth: Nat,+work: Pipeline.Work,+offset: U32,+span: U32,+input: InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {ArrayPipeline.partition(depth,work,offset,span,represent(input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)) ==
    PartitionLaws.represent(~F32,partition(depth,work,offset,span,input,weights,bias)) : Partition.Tree<&1,Array<F32>,U32>}:
  PartitionLaws.run_matches(~ArrayPipeline.InputWindow,~InputWindow,~Tile.Tile8,~F32,~F32,~Pipeline.Work,~represent,~ArrayPipeline.split,~split,~ArrayPipeline.leaf,~leaf,~split_matches,~leaf_matches,
    depth,work,offset,span,input,weights,bias)

def native_gather(partitions: Partition.Tree<&1,Array<F32>,U32>,output: Array<F32>,positions: U32) -> Array<F32>:
  Partition.gather(~Array<F32>,~(rows => offset => positions => span => input => output => Gather.gather_block(rows,offset,positions,span,input,output)),partitions,output,positions)

def gather_matches(+partitions: GatherModel.ConvResult(),+output: Tree.Tree<F32>,+positions: U32) ->
  {native_gather(PartitionLaws.represent(~F32,partitions),Tree.pack(F32,output),positions) == Tree.pack(F32,GatherModel.gather(partitions,output,positions)) : Array<F32>}:
  match partitions:
    case Partition.Leaf{input,offset,span,rows}: Blocks.block(rows,offset,positions,span,input,output)
    case Partition.Fork{+left,+right}:
      %Equal.sym(Array<F32>,native_gather(PartitionLaws.represent(~F32,left),Tree.pack(F32,output),positions),Tree.pack(F32,GatherModel.gather(left,output,positions)),
        gather_matches(left,output,positions)) :
        {native_gather(PartitionLaws.represent(~F32,right),_,positions) == Tree.pack(F32,GatherModel.gather(right,GatherModel.gather(left,output,positions),positions)) : Array<F32>}
      gather_matches(right,GatherModel.gather(left,output,positions),positions)

def finish_leaf(identity: Bool,input: Tree.Tree<F32>,+rows: U32,offset: U32,span: U32,+positions: U32) -> Tree.Tree<F32>:
  match identity:
    case True{}: input
    case False{}: GatherModel.gather(Partition.Leaf{input,offset,span,rows},allocated(rows,positions),positions)

def finish_leaf_matches(+identity: Bool,+input: Tree.Tree<F32>,+rows: U32,+offset: U32,+span: U32,+positions: U32) ->
  {ArrayPipeline.finish_leaf(identity,Tree.pack(F32,input),rows,offset,span,positions) == Tree.pack(F32,finish_leaf(identity,input,rows,offset,span,positions)) : Array<F32>}:
  match identity:
    case True{}: {==}
    case False{}:
      %Equal.sym(Array<F32>,ArrayPipeline.allocated(rows,positions),Tree.pack(F32,allocated(rows,positions)),allocated_matches(rows,positions)) :
        {Gather.gather_block(rows,offset,positions,span,Tree.pack(F32,input),_) == Tree.pack(F32,finish_leaf(False{},input,rows,offset,span,positions)) : Array<F32>}
      Blocks.block(rows,offset,positions,span,input,allocated(rows,positions))

def finish(partitions: GatherModel.ConvResult(),+rows: U32,+positions: U32) -> Tree.Tree<F32>:
  match partitions:
    case Partition.Leaf{input,+offset,+span,channels}: finish_leaf(U32.is_zero(offset) && U32.is_eq(span,positions),input,channels,offset,span,positions)
    case Partition.Fork{left,right}: GatherModel.gather(Partition.Fork{left,right},allocated(rows,positions),positions)

def finish_matches(+partitions: GatherModel.ConvResult(),+rows: U32,+positions: U32) ->
  {ArrayPipeline.finish(PartitionLaws.represent(~F32,partitions),rows,positions) == Tree.pack(F32,finish(partitions,rows,positions)) : Array<F32>}:
  match partitions:
    case Partition.Leaf{input,+offset,+span,channels}: finish_leaf_matches(U32.is_zero(offset) && U32.is_eq(span,positions),input,channels,offset,span,positions)
    case Partition.Fork{+left,+right}:
      %Equal.sym(Array<F32>,ArrayPipeline.allocated(rows,positions),Tree.pack(F32,allocated(rows,positions)),allocated_matches(rows,positions)) :
        {native_gather(Partition.Fork{PartitionLaws.represent(~F32,left),PartitionLaws.represent(~F32,right)},_,positions) ==
          Tree.pack(F32,finish(Partition.Fork{left,right},rows,positions)) : Array<F32>}
      gather_matches(Partition.Fork{left,right},allocated(rows,positions),positions)

def run(depth: U32,work: Pipeline.Work,input: Tree.Tree<F32>,weights: Tree.Tree<F32>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  Pipeline.Work{+rows,+h,w,k,s,p,ow,+positions,+inner} = work
  +padded = ((positions + 7) / 8 * 8 : U32)
  +limit = (U32.from_nat(Tensors.significant_bits(32n,(padded / 8 : U32),0n,U32.is_zero((padded / 8 : U32)))) - 1 : U32)
  finish(partition(U32.to_nat(U32.min(depth,limit)),Pipeline.Work{rows,h,w,k,s,p,ow,positions,inner},0,padded,InputWindow{input,0,h},WeightsLaws.run(rows,inner,weights),bias),rows,positions)

def run_matches(+depth: U32,+work: Pipeline.Work,+input: Tree.Tree<F32>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>) ->
  {ArrayPipeline.run(depth,work,Tree.pack(F32,input),Tree.pack(F32,weights),Tree.pack(F32,bias)) ==
    Tree.pack(F32,run(depth,work,input,weights,bias)) : Array<F32>}:
  Pipeline.Work{+rows,+h,+w,+k,+s,+p,+ow,+positions,+inner} = work
  +work = {Pipeline.Work{rows,h,w,k,s,p,ow,positions,inner} : Pipeline.Work}
  +padded = ((positions + 7) / 8 * 8 : U32)
  +limit = (U32.from_nat(Tensors.significant_bits(32n,(padded / 8 : U32),0n,U32.is_zero((padded / 8 : U32)))) - 1 : U32)
  +split = U32.to_nat(U32.min(depth,limit))
  +reordered = WeightsLaws.run(rows,inner,weights)
  +partitions = partition(split,work,0,padded,InputWindow{input,0,h},reordered,bias)
  %Equal.sym(Array<Tile.Tile8>,Weights.run(rows,inner,Tree.pack(F32,weights)),Tree.pack(Tile.Tile8,reordered),WeightsLaws.run_matches(rows,inner,weights)) :
    {ArrayPipeline.finish(ArrayPipeline.partition(split,work,0,padded,ArrayPipeline.InputWindow{Tree.pack(F32,input),0,h},_,Tree.pack(F32,bias)),rows,positions) ==
      Tree.pack(F32,finish(partitions,rows,positions)) : Array<F32>}
  %Equal.sym(Partition.Tree<&1,Array<F32>,U32>,ArrayPipeline.partition(split,work,0,padded,ArrayPipeline.InputWindow{Tree.pack(F32,input),0,h},Tree.pack(Tile.Tile8,reordered),Tree.pack(F32,bias)),
    PartitionLaws.represent(~F32,partitions),partition_matches(split,work,0,padded,InputWindow{input,0,h},reordered,bias)) :
    {ArrayPipeline.finish(_,rows,positions) == Tree.pack(F32,finish(partitions,rows,positions)) : Array<F32>}
  finish_matches(partitions,rows,positions)
