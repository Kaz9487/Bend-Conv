# Dense convolution from bands: gathered input, activation writes, output
# readback, sizes and finite values.
import Base
import ./band_leaf_values.bend as BandLeafValues
import ./window_positions.bend as Positions
import ./window_covers.bend as Covers
import ./window_band_values.bend as WindowBandValues
import ./window_problem.bend as WindowProblem
import ./convolution_leaf_semantics.bend as LeafSemantics
import ./tile_partition_bounds.bend as TilePartitionBounds
import ./gather_block_width.bend as BlockWidth
import ../../lib/proofs/u32_extrema.bend as U32Extrema
import ../../lib/proofs/tensor_band_window.bend as TensorBandWindow
import ../../lib/proofs/u32_comparison.bend as Comparison

import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/proofs/kernel_pipeline_activation.bend as ActivationProofs
import ./blocked_pipeline_initial.bend as BlockedInitial
import ./blocked_pipeline_projection.bend as BlockedProjection
import ./blocked_pipeline_model.bend as Blocked
import ../inference.bend as NativeConv
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./convolution_guard_facts.bend as Guards
import ./input_storage.bend as Inputs
import ./supported_padded_coordinates.bend as Padded
import ./concrete_support_guard.bend as Support
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./prepared_output_readback.bend as PreparedOutputReadback
import ./reduction_coordinate_bounds.bend as Coordinates
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/tensor.bend as Tensors
import ../../lib/traversal_partition.bend as Partition
import ./band_dense_pipeline.bend as DensePipeline
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ./gather_storage_model.bend as GatherModel
import ./gemm_dot_matches_matrix.bend as GemmDot
import ./microkernel_inference_refinement.bend as Inference
import ./microkernel_storage_execution.bend as Execution
import ./packed_output_bounds.bend as Output
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./tile_layout.bend as TileLayout
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/write_program.bend as Writes
import ./bounded_write_readback.bend as WriteReadback
import ./gather_block_coordinates.bend as BlockCoordinates
import ./gather_leaf_writes.bend as GatherLeafWrites
import ./gather_write_program.bend as GatherWrites
import ./output_values.bend as Values
import ./packing_storage_refinement.bend as PackingStorage
import ./prepared_packing_samples.bend as Prepared
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/storage_buffer.bend as Storage
import ./conditional_write_coverage.bend as Coverage
import ./gather_block_writes.bend as BlockWrites
import ./gather_blocks.bend as Blocks
import ./gather_tree_readback.bend as GatherTreeReadback
import ./packed_output_allocation.bend as PackedOutputAllocation
import ./tile_partition_coverage.bend as TilePartitionCoverage
import ./window_partition.bend as Windows
import ../../lib/kernel_array_pipeline.bend as ArrayPipeline
import ../../lib/kernel_gather.bend as Gather
import ../../lib/kernel_weights.bend as Weights
import ../../lib/proofs/bounded_product_associativity.bend as Products
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/tensor_dense_contract.bend as Dense

# Activation transports values through the same ordered gather writes. Address
# coverage and allocation bounds come from the original convolution proof;
# this capability never introduces another MAC reduction or rounding rule.
def mapped(~operation: F32 -> F32,writes: List<&2,Writes.Assignment<F32>>) -> List<&2,Writes.Assignment<F32>>:
  match writes:
    case Nil{}: []
    case Writes.Put{address,value} <> rest: Writes.Put{address,operation(value)} <> mapped(~operation,rest)

def mapped_matching(~operation: F32 -> F32,+matches: Bool,+actual: F32,+wanted: F32,evidence: Coverage.Matches(matches,actual,wanted)) ->
  Coverage.Matches(matches,operation(actual),operation(wanted)):
  match matches:
    case False{}: Unit{}
    case True{}: Equal.cong(F32,F32,value => operation(value),actual,wanted,evidence)

def mapped_bounds(~operation: F32 -> F32,+writes: List<&2,Writes.Assignment<F32>>,+capacity: U32,bounds: WriteReadback.WritesWithinCapacity(writes,capacity)) ->
  WriteReadback.WritesWithinCapacity(mapped(~operation,writes),capacity):
  match writes:
    case Nil{}: Unit{}
    case Writes.Put{+address,+actual} <> +rest:
      Shape.Both{head,tail} = bounds
      Shape.Both{head,mapped_bounds(~operation,rest,capacity,tail)}

def mapped_values(~operation: F32 -> F32,+writes: List<&2,Writes.Assignment<F32>>,+query: Nat,+wanted: F32,
  values: Coverage.MatchingWritesHaveValue(query,wanted,WriteReadback.natural_write_addresses(writes))) ->
  Coverage.MatchingWritesHaveValue(query,operation(wanted),WriteReadback.natural_write_addresses(mapped(~operation,writes))):
  match writes:
    case Nil{}: Unit{}
    case Writes.Put{+address,+actual} <> +rest:
      Shape.Both{head,tail} = values
      Shape.Both{mapped_matching(~operation,Nat.is_eq(query,U32.to_nat(address)),actual,wanted,head),mapped_values(~operation,rest,query,wanted,tail)}

def mapped_coverage(~operation: F32 -> F32,+writes: List<&2,Writes.Assignment<F32>>,+query: Nat) ->
  {Coverage.contains(query,WriteReadback.natural_write_addresses(mapped(~operation,writes))) == Coverage.contains(query,WriteReadback.natural_write_addresses(writes)) : Bool}:
  match writes:
    case Nil{}: {==}
    case Writes.Put{+address,+actual} <> +rest:
      %Equal.sym(Bool,Coverage.contains(query,WriteReadback.natural_write_addresses(mapped(~operation,rest))),Coverage.contains(query,WriteReadback.natural_write_addresses(rest)),mapped_coverage(~operation,rest,query)) :
        {Nat.is_eq(query,U32.to_nat(address)) || _ == Nat.is_eq(query,U32.to_nat(address)) || Coverage.contains(query,WriteReadback.natural_write_addresses(rest)) : Bool}
      {==}

def mapped_concatenation(~operation: F32 -> F32,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>) ->
  {mapped(~operation,List.append(&2,Writes.Assignment<F32>,left,right)) ==
    List.append(&2,Writes.Assignment<F32>,mapped(~operation,left),mapped(~operation,right)) : List<&2,Writes.Assignment<F32>>}:
  match left:
    case Nil{}: {==}
    case Writes.Put{+address,+actual} <> +rest:
      Equal.cong(List<&2,Writes.Assignment<F32>>,List<&2,Writes.Assignment<F32>>,tail => Writes.Put{address,operation(actual)} <> tail,
        mapped(~operation,List.append(&2,Writes.Assignment<F32>,rest,right)),List.append(&2,Writes.Assignment<F32>,mapped(~operation,rest),mapped(~operation,right)),mapped_concatenation(~operation,rest,right))

def mapped_readback(~operation: F32 -> F32,+writes: List<&2,Writes.Assignment<F32>>,+output: Tree.Tree<F32>,+depth: Nat,+query: U32,+wanted: F32,
  +balanced: Shape.Balanced(F32,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  evidence: WriteReadback.WriteEvidence<U32.to_nat(query),wanted,Tree.capacity(F32,output),writes>,
  +inside: {U32.is_lt(query,Tree.capacity(F32,output)) == True{} : Bool},
  covered: {Coverage.contains(U32.to_nat(query),WriteReadback.natural_write_addresses(writes)) == True{} : Bool}) ->
  {Tree.read(F32,Writes.pure(F32,mapped(~operation,writes),output),query) == operation(wanted) : F32}:
  WriteReadback.WriteEvidence{bounds,values} = evidence
  WriteReadback.readback_from_matching_covered_writes(mapped(~operation,writes),output,depth,query,operation(wanted),balanced,depth_ok,
    mapped_bounds(~operation,writes,Tree.capacity(F32,output),bounds),inside,
    mapped_values(~operation,writes,U32.to_nat(query),wanted,values),
    %Equal.sym(Bool,Coverage.contains(U32.to_nat(query),WriteReadback.natural_write_addresses(mapped(~operation,writes))),Coverage.contains(U32.to_nat(query),WriteReadback.natural_write_addresses(writes)),mapped_coverage(~operation,writes,U32.to_nat(query))) : {_ == True{} : Bool}
    covered)

def leaf_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,+supported: Support.FastSupported<limits,problem>,
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(MicroLeaf.rows(problem)),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(MicroLeaf.rows(problem)),Guards.output_channel_count(problem),SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,supported))) :
    {Nat.is_le(Nat.mul(_,U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Output.elements_bound(limits,problem,span,supported,span_bound)

# Every actual gather source index lies in this leaf's activation pass. The
# leaf's own established certificate supplies balance; the existing packed
# support guard supplies its product bound, including arbitrary valid padding.
def model_split(task: Pipeline.Task,offset: U32,half: U32,input: Execution.InputWindow) -> PartitionLaws.Split<Execution.InputWindow>:
  Pipeline.Task{work,activation} = task
  Execution.split(work,offset,half,input)

def model_leaf(~custom: F32 -> F32,task: Pipeline.Task,offset: U32,span: U32,input: Execution.InputWindow,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> GatherModel.ConvResult():
  Pipeline.Task{Pipeline.Work{+rows,height,+width,+kernel,+stride,+padding,+output_width,+positions,+inner},activation} = task
  Execution.InputWindow{input,+first,+window_rows} = input
  +offset = offset
  +span = span
  +count = U32.min(span,(positions - offset : U32))
  Partition.Leaf{Blocked.run(~custom,activation,Pipeline.Work{rows,window_rows,width,kernel,stride,padding,output_width,count,inner},
    KernelWindow.start(offset,stride,output_width,first),input,weights,bias),offset,count,rows}

def split_matches(+task: Pipeline.Task,+offset: U32,+half: U32,+input: Execution.InputWindow) ->
  {DensePipeline.array_split(task,offset,half,Execution.represent(input)) ==
    PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~Execution.InputWindow,~Execution.represent,model_split(task,offset,half,input)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  Pipeline.Task{work,activation} = task
  Execution.split_matches(work,offset,half,input)

def leaf_matches(~custom: F32 -> F32,+task: Pipeline.Task,+offset: U32,+span: U32,+input: Execution.InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {DensePipeline.array_leaf(~custom,task,offset,span,Execution.represent(input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)) ==
    PartitionLaws.represent(~F32,model_leaf(~custom,task,offset,span,input,weights,bias)) : Partition.Tree<&1,Array<F32>,U32>}:
  Pipeline.Task{Pipeline.Work{+rows,height,+width,+kernel,+stride,+padding,+output_width,+positions,+inner},+activation} = task
  Execution.InputWindow{+input,+first,+window_rows} = input
  +count = U32.min(span,(positions - offset : U32))
  +work = {Pipeline.Work{rows,window_rows,width,kernel,stride,padding,output_width,count,inner} : Pipeline.Work}
  +start = KernelWindow.start(offset,stride,output_width,first)
  Equal.cong(Array<F32>,Partition.Tree<&1,Array<F32>,U32>,array => Partition.Leaf{array,offset,count,rows},
    BlockedProjection.run(~custom,activation,work,start,Tree.pack(F32,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)),
    Tree.pack(F32,Blocked.run(~custom,activation,work,start,input,weights,bias)),Blocked.run_matches(~custom,activation,work,start,input,weights,bias))

def model_partition(~custom: F32 -> F32,+depth: Nat,+task: Pipeline.Task,+offset: U32,+span: U32,+input: Execution.InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) -> GatherModel.ConvResult():
  PartitionLaws.partition(~Execution.InputWindow,~Tile.Tile8,~F32,~F32,~Pipeline.Task,~model_split,~model_leaf(~custom),depth,task,offset,span,input,weights,bias)

def partition_matches(~custom: F32 -> F32,+depth: Nat,+task: Pipeline.Task,+offset: U32,+span: U32,+input: Execution.InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {DensePipeline.array_partition(~custom,depth,task,offset,span,Execution.represent(input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)) ==
    PartitionLaws.represent(~F32,model_partition(~custom,depth,task,offset,span,input,weights,bias)) : Partition.Tree<&1,Array<F32>,U32>}:
  PartitionLaws.run_matches(~ArrayPipeline.InputWindow,~Execution.InputWindow,~Tile.Tile8,~F32,~F32,~Pipeline.Task,
    ~Execution.represent,~DensePipeline.array_split,~model_split,~DensePipeline.array_leaf(~custom),~model_leaf(~custom),~split_matches,~leaf_matches(~custom),
    depth,task,offset,span,input,weights,bias)

def work_inner(work: Pipeline.Work) -> U32:
  Pipeline.Work{rows,h,w,k,s,p,ow,positions,inner} = work
  inner

def run_partitioned_model(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,input: Tree.Tree<F32>,weights: Tree.Tree<F32>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  Pipeline.Work{+rows,+height,width,kernel,stride,padding,output_width,+positions,+inner} = work
  +padded = ((positions + 7) / 8 * 8 : U32)
  +limit = (U32.from_nat(Tensors.significant_bits(32n,(padded / 8 : U32),0n,U32.is_zero((padded / 8 : U32)))) - 1 : U32)
  Execution.finish(model_partition(~custom,U32.to_nat(U32.min(depth,limit)),Pipeline.Task{Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner},activation},0,padded,
    Execution.InputWindow{input,0,height},WeightsLaws.run(rows,inner,weights),bias),rows,positions)

def array_partitioned_matches(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Tree.Tree<F32>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>) ->
  {DensePipeline.array_run_partitioned(~custom,activation,depth,work,Tree.pack(F32,input),Weights.run(DensePipeline.work_rows(work),work_inner(work),Tree.pack(F32,weights)),Tree.pack(F32,bias)) ==
    Tree.pack(F32,run_partitioned_model(~custom,activation,depth,work,input,weights,bias)) : Array<F32>}:
  Pipeline.Work{+rows,+height,+width,+kernel,+stride,+padding,+output_width,+positions,+inner} = work
  +work = {Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner} : Pipeline.Work}
  +padded = ((positions + 7) / 8 * 8 : U32)
  +limit = (U32.from_nat(Tensors.significant_bits(32n,(padded / 8 : U32),0n,U32.is_zero((padded / 8 : U32)))) - 1 : U32)
  +actual_depth = U32.to_nat(U32.min(depth,limit))
  +task = {Pipeline.Task{work,activation} : Pipeline.Task}
  +reordered = WeightsLaws.run(rows,inner,weights)
  +model = model_partition(~custom,actual_depth,task,0,padded,Execution.InputWindow{input,0,height},reordered,bias)
  %Equal.sym(Array<Tile.Tile8>,Weights.run(rows,inner,Tree.pack(F32,weights)),Tree.pack(Tile.Tile8,reordered),WeightsLaws.run_matches(rows,inner,weights)) :
    {ArrayPipeline.finish(DensePipeline.array_partition(~custom,actual_depth,task,0,padded,ArrayPipeline.InputWindow{Tree.pack(F32,input),0,height},_,Tree.pack(F32,bias)),rows,positions) ==
      Tree.pack(F32,Execution.finish(model,rows,positions)) : Array<F32>}
  %Equal.sym(Partition.Tree<&1,Array<F32>,U32>,DensePipeline.array_partition(~custom,actual_depth,task,0,padded,Execution.represent(Execution.InputWindow{input,0,height}),Tree.pack(Tile.Tile8,reordered),Tree.pack(F32,bias)),
    PartitionLaws.represent(~F32,model),partition_matches(~custom,actual_depth,task,0,padded,Execution.InputWindow{input,0,height},reordered,bias)) :
    {ArrayPipeline.finish(_,rows,positions) == Tree.pack(F32,Execution.finish(model,rows,positions)) : Array<F32>}
  Execution.finish_matches(model,rows,positions)

def run_selected_model(~custom: F32 -> F32,blocked_leaf: Bool,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,input: Tree.Tree<F32>,weights: Tree.Tree<F32>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  match blocked_leaf:
    case True{}: Blocked.run(~custom,activation,work,0,input,WeightsLaws.run(DensePipeline.work_rows(work),work_inner(work),weights),bias)
    case False{}: run_partitioned_model(~custom,activation,depth,work,input,weights,bias)

def run_model(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,input: Tree.Tree<F32>,weights: Tree.Tree<F32>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  run_selected_model(~custom,U32.is_zero(depth),activation,depth,work,input,weights,bias)

def array_selected_matches(~custom: F32 -> F32,+blocked_leaf: Bool,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Tree.Tree<F32>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>) ->
  {DensePipeline.array_run_selected(~custom,blocked_leaf,activation,depth,work,Tree.pack(F32,input),Weights.run(DensePipeline.work_rows(work),work_inner(work),Tree.pack(F32,weights)),Tree.pack(F32,bias)) ==
    Tree.pack(F32,run_selected_model(~custom,blocked_leaf,activation,depth,work,input,weights,bias)) : Array<F32>}:
  match blocked_leaf:
    case True{}:
      +rows = DensePipeline.work_rows(work)
      +inner = work_inner(work)
      %Equal.sym(Array<Tile.Tile8>,Weights.run(rows,inner,Tree.pack(F32,weights)),Tree.pack(Tile.Tile8,WeightsLaws.run(rows,inner,weights)),WeightsLaws.run_matches(rows,inner,weights)) :
        {BlockedProjection.run(~custom,activation,work,0,Tree.pack(F32,input),_,Tree.pack(F32,bias)) == Tree.pack(F32,Blocked.run(~custom,activation,work,0,input,WeightsLaws.run(rows,inner,weights),bias)) : Array<F32>}
      Blocked.run_matches(~custom,activation,work,0,input,WeightsLaws.run(rows,inner,weights),bias)
    case False{}: array_partitioned_matches(~custom,activation,depth,work,input,weights,bias)

def array_run_matches(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Tree.Tree<F32>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>) ->
  {DensePipeline.array_run(~custom,activation,depth,work,Tree.pack(F32,input),Tree.pack(F32,weights),Tree.pack(F32,bias)) == Tree.pack(F32,run_model(~custom,activation,depth,work,input,weights,bias)) : Array<F32>}:
  Pipeline.Work{rows,h,w,k,s,p,ow,positions,inner} = work
  array_selected_matches(~custom,U32.is_zero(depth),activation,depth,work,input,weights,bias)

def run_refines_model(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>) ->
  {ActivationProofs.storage(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) ==
    Tree.pack(F32,run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}:
  Equal.trans(Array<F32>,ActivationProofs.storage(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),
    DensePipeline.array_run(~custom,activation,depth,work,ActivationProofs.storage(~F32,Buffer.pack(~F32,input)),ActivationProofs.storage(~F32,Buffer.pack(~F32,weights)),ActivationProofs.storage(~F32,Buffer.pack(~F32,bias))),
    Tree.pack(F32,run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))),
    DensePipeline.run_projection(~custom,activation,depth,work,input,weights,bias),
    %Equal.sym(Array<F32>,ActivationProofs.storage(~F32,Buffer.pack(~F32,input)),Tree.pack(F32,Buffer.tree(~F32,input)),DensePipeline.storage_matches(~F32,input)) :
      {DensePipeline.array_run(~custom,activation,depth,work,_,ActivationProofs.storage(~F32,Buffer.pack(~F32,weights)),ActivationProofs.storage(~F32,Buffer.pack(~F32,bias))) ==
        Tree.pack(F32,run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}
    %Equal.sym(Array<F32>,ActivationProofs.storage(~F32,Buffer.pack(~F32,weights)),Tree.pack(F32,Buffer.tree(~F32,weights)),DensePipeline.storage_matches(~F32,weights)) :
      {DensePipeline.array_run(~custom,activation,depth,work,Tree.pack(F32,Buffer.tree(~F32,input)),_,ActivationProofs.storage(~F32,Buffer.pack(~F32,bias))) ==
        Tree.pack(F32,run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}
    %Equal.sym(Array<F32>,ActivationProofs.storage(~F32,Buffer.pack(~F32,bias)),Tree.pack(F32,Buffer.tree(~F32,bias)),DensePipeline.storage_matches(~F32,bias)) :
      {DensePipeline.array_run(~custom,activation,depth,work,Tree.pack(F32,Buffer.tree(~F32,input)),Tree.pack(F32,Buffer.tree(~F32,weights)),_) ==
        Tree.pack(F32,run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}
    array_run_matches(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)))

# The real gather reads only positions visited by each leaf's activation pass.
# Its write program therefore maps the values of the established gather while
# retaining all addresses, ordering and coverage certificates.
# A leaf's logical row width contains only real positions.
def leaf_count(+problem: ConvSpec.Problem,+offset: U32,+span: U32) -> U32:
  U32.min(span,(Padded.spatial(problem) - offset : U32))

def clipped_width_when(inside: Bool,+offset: U32,+spatial: U32,+span: U32) ->
  {Gather.gather_width_when(inside,offset,spatial,U32.min(span,(spatial - offset : U32))) == Gather.gather_width_when(inside,offset,spatial,span) : U32}:
  match inside:
    case False{}: {==}
    case True{}: U32Extrema.minimum_when_ordered(U32.min(span,(spatial - offset : U32)),(spatial - offset : U32),
      TensorBandWindow.minimum_right_below(span,(spatial - offset : U32),(spatial - offset : U32),Order.reflexive(U32.to_nat((spatial - offset : U32)))))

def clipped_width(+offset: U32,+spatial: U32,+span: U32) ->
  {Gather.gather_width(offset,spatial,U32.min(span,(spatial - offset : U32))) == Gather.gather_width(offset,spatial,span) : U32}:
  clipped_width_when(U32.is_lt(offset,spatial),offset,spatial,span)

def window_output(~custom: F32 -> F32,+activation: Activations.Activation,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+offset: U32,+count: U32,+tree: Tree.Tree<F32>,+first: U32,+rows: U32) -> Tree.Tree<F32>:
  Blocked.run(~custom,activation,Pipeline.Work{MicroLeaf.rows(problem),rows,U32.from_nat(Guards.width(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),count,Coordinates.native_inner(problem)},
    KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first),tree,GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage))

def window_output_value(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +offset: U32,+count: U32,+tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(count)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool},
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(count)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +channel: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(F32,window_output(~custom,activation,problem,storage,offset,count,tree,first,rows),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(count)),column))) ==
    ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)))) : F32}:
  Inputs.Contract{input,+weights,+bias} = storage
  +storage = {Inputs.Contract{input,weights,bias} : Inputs.Contract<problem>}
  %BandLeafValues.inner_word(limits,problem,valid,supported) :
    {Tree.read(F32,Blocked.run(~custom,activation,Pipeline.Work{MicroLeaf.rows(problem),rows,U32.from_nat(Guards.width(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),count,_},
      KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first),tree,GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(count)),column))) ==
      ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)))) : F32}
  BandLeafValues.runtime_value(~custom,activation,limits,problem,weights,bias,valid,supported,offset,count,grid,tree,first,rows,cover,contents,room,
    GemmDot.reordered(problem,storage),{==},Inputs.bias_tree(problem,storage),{==},channel,column,channel_inside,column_inside)

type Source is Data:
  Source{activation: Activations.Activation,problem: ConvSpec.Problem,storage: Inputs.Contract<problem>,offset: U32,span: U32,actual_span: U32,actual: Tree.Tree<F32>}

def SourceReads(~custom: F32 -> F32,source: Source) -> Type:
  Source{+activation,+problem,+storage,+offset,+span,+actual_span,+actual} = source
  @+row: U32 -> @+column: U32 ->
  @row_inside: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool} ->
  @column_inside: {Nat.is_lt(U32.to_nat(column),U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span))) == True{} : Bool} ->
  {Tree.read(F32,actual,BlockCoordinates.source(row,actual_span,column)) == ActivationProofs.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),BlockCoordinates.source(row,span,column))) : F32}

def window_source(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +offset: U32,+span: U32,+tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+span_alignment: TilePartitionBounds.BlockAligned<span>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +row: U32,+column: U32,+row_inside: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},
  +column_inside: {Nat.is_lt(U32.to_nat(column),U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span))) == True{} : Bool}) ->
  {Tree.read(F32,window_output(~custom,activation,problem,storage,offset,leaf_count(problem,offset,span),tree,first,rows),BlockCoordinates.source(row,leaf_count(problem,offset,span),column)) ==
    ActivationProofs.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),BlockCoordinates.source(row,span,column))) : F32}:
  +facts = BlockWrites.width_facts(limits,problem,offset,span,column,valid,supported,column_inside)
  +in_span = BlockWrites.first({Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool},{Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(column)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool},facts)
  +in_grid = BlockWrites.second({Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool},{Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(column)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool},facts)
  +fast = Support.fast(limits,problem,supported)
  +spatial = Padded.spatial(problem)
  +count = leaf_count(problem,offset,span)
  +start = U32.to_nat(offset)
  +position = Nat.add(start,U32.to_nat(column))
  +width = ConvSpec.out_width(problem)
  +width_positive = Covers.out_width_positive(problem)
  +x = Nat.mod(position,width)
  +y = Nat.div(position,width)
  +spatial_exact = Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),fast)
  +guard = {%Equal.sym(Bool,U32.is_lt(offset,spatial),Nat.is_lt(start,U32.to_nat(spatial)),Comparison.lt(offset,spatial)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(spatial),GatherCopyCoordinates.spatial(problem),spatial_exact) : {Nat.is_lt(start,_) == True{} : Bool}
    Order.le_lt_trans(start,position,GatherCopyCoordinates.spatial(problem),Order.le_add(start,U32.to_nat(column)),in_grid) : {U32.is_lt(offset,spatial) == True{} : Bool}}
  +in_count = {%guard : {Nat.is_lt(U32.to_nat(column),U32.to_nat(Gather.gather_width_when(_,offset,spatial,span))) == True{} : Bool}
    column_inside : {Nat.is_lt(U32.to_nat(column),U32.to_nat(count)) == True{} : Bool}}
  +count_le = TensorBandWindow.minimum_below(span,(spatial - offset : U32))
  +count_bound = Order.transitive(U32.to_nat(count),U32.to_nat(span),Support.padded(problem),count_le,span_bound)
  +grid = {%spatial_exact : {Nat.is_le(Nat.add(start,U32.to_nat(count)),_) == True{} : Bool}
    %BlockWidth.restored(offset,spatial,Machine.capacity(limits),Limits.capacity_fits(limits),BlockCoordinates.spatial_bound(limits,problem,Guards.valid(problem,valid),fast),guard) : {Nat.is_le(Nat.add(start,U32.to_nat(count)),_) == True{} : Bool}
    Order.offset_le(start,U32.to_nat(count),U32.to_nat((spatial - offset : U32)),TensorBandWindow.minimum_right_below(span,(spatial - offset : U32),(spatial - offset : U32),Order.reflexive(U32.to_nat((spatial - offset : U32))))) :
    {Nat.is_le(Nat.add(start,U32.to_nat(count)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool}}
  +new_value = window_output_value(~custom,activation,limits,problem,storage,offset,count,tree,first,rows,valid,supported,grid,
    Positions.cover_within(problem,U32.to_nat(first),U32.to_nat(rows),start,U32.to_nat(span),start,U32.to_nat(count),cover,width_positive,Order.reflexive(start),Order.offset_le(start,U32.to_nat(count),U32.to_nat(span),count_le)),
    contents,room,U32.to_nat(row),U32.to_nat(column),row_inside,in_count)
  +old_value = LeafSemantics.band_point(limits,problem,storage,offset,span,U32.to_nat(row),U32.to_nat(column),y,x,valid,supported,span_bound,span_alignment,
    Order.transitive(Nat.add(start,U32.to_nat(span)),Support.padded(problem),Nat.add(GatherCopyCoordinates.spatial(problem),7n),range,DivisionOrder.product_below(Nat.add(GatherCopyCoordinates.spatial(problem),7n),8n,{==})),
    row_inside,in_span,DivisionOrder.quotient_exact(position,width,width_positive),Positions.quotient_lt(position,ConvSpec.out_height(problem),width,width_positive,in_grid),DivisionOrder.remainder_small(position,width,width_positive))
  %Arithmetic.native_roundtrip(BlockCoordinates.source(row,count,column)) :
    {Tree.read(F32,window_output(~custom,activation,problem,storage,offset,count,tree,first,rows),_) == ActivationProofs.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),BlockCoordinates.source(row,span,column))) : F32}
  %Arithmetic.native_roundtrip(BlockCoordinates.source(row,span,column)) :
    {Tree.read(F32,window_output(~custom,activation,problem,storage,offset,count,tree,first,rows),U32.from_nat(U32.to_nat(BlockCoordinates.source(row,count,column)))) == ActivationProofs.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),_)) : F32}
  %Equal.sym(Nat,U32.to_nat(BlockCoordinates.source(row,count,column)),Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(count)),U32.to_nat(column)),BlockCoordinates.source_exact(limits,problem,count,row,column,fast,count_bound,row_inside,in_count)) :
    {Tree.read(F32,window_output(~custom,activation,problem,storage,offset,count,tree,first,rows),U32.from_nat(_)) == ActivationProofs.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),U32.from_nat(U32.to_nat(BlockCoordinates.source(row,span,column))))) : F32}
  %Equal.sym(Nat,U32.to_nat(BlockCoordinates.source(row,span,column)),Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(span)),U32.to_nat(column)),BlockCoordinates.source_exact(limits,problem,span,row,column,fast,span_bound,row_inside,in_span)) :
    {Tree.read(F32,window_output(~custom,activation,problem,storage,offset,count,tree,first,rows),U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(count)),U32.to_nat(column)))) == ActivationProofs.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),U32.from_nat(_))) : F32}
  Equal.trans(F32,Tree.read(F32,window_output(~custom,activation,problem,storage,offset,count,tree,first,rows),U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(count)),U32.to_nat(column)))),
    ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,U32.to_nat(row),y,x)),ActivationProofs.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(span)),U32.to_nat(column))))),new_value,
    Equal.cong(F32,F32,value => ActivationProofs.activation_value(~custom,activation,value),Values.matrix_value(problem,U32.to_nat(row),y,x),Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(span)),U32.to_nat(column)))),
      Equal.sym(F32,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(span)),U32.to_nat(column)))),Values.matrix_value(problem,U32.to_nat(row),y,x),old_value)))

type BlockedEvidence<-custom: F32 -> F32,-activation: Activations.Activation,-problem: ConvSpec.Problem,-storage: Inputs.Contract<problem>,-offset: U32,-span: U32,-actual_span: U32,-actual: Tree.Tree<F32>> is Data:
  BlockedEvidence{limits: Machine.Limits,
    valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
    span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},span_alignment: TilePartitionBounds.BlockAligned<span>,
    range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
    input: Tree.Tree<F32>,first: U32,rows: U32,
    cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
    contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),input),
    room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,input))) == True{} : Bool},
    width: {actual_span == leaf_count(problem,offset,span) : U32},
    tree: {actual == window_output(~custom,activation,problem,storage,offset,leaf_count(problem,offset,span),input,first,rows) : Tree.Tree<F32>}}

def blocked_evidence(~custom: F32 -> F32,source: Source) -> Data:
  Source{activation,problem,storage,offset,span,actual_span,actual} = source
  BlockedEvidence<custom,activation,problem,storage,offset,span,actual_span,actual>

def blocked_reads(~custom: F32 -> F32,+source: Source,evidence: blocked_evidence(~custom,source)) -> SourceReads(~custom,source):
  Source{+activation,+problem,+storage,+offset,+span,+actual_span,+actual} = source
  BlockedEvidence{+limits,+valid,+supported,+span_bound,+span_alignment,+range,+input,+first,+rows,+cover,+contents,+room,width,tree} = evidence
  row => column => row_inside => column_inside =>
    %Equal.sym(U32,actual_span,leaf_count(problem,offset,span),width) :
      {Tree.read(F32,actual,BlockCoordinates.source(row,_,column)) == ActivationProofs.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),BlockCoordinates.source(row,span,column))) : F32}
    %Equal.sym(Tree.Tree<F32>,actual,window_output(~custom,activation,problem,storage,offset,leaf_count(problem,offset,span),input,first,rows),tree) :
      {Tree.read(F32,_,BlockCoordinates.source(row,leaf_count(problem,offset,span),column)) == ActivationProofs.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),BlockCoordinates.source(row,span,column))) : F32}
    window_source(~custom,activation,limits,problem,storage,offset,span,input,first,rows,valid,supported,span_bound,span_alignment,range,cover,contents,room,row,column,row_inside,column_inside)

def columns_mapped(~custom: F32 -> F32,~Evidence: Source -> Data,~reads: @+source: Source -> @evidence: Evidence(source) -> SourceReads(~custom,source),+activation: Activations.Activation,+count: Nat,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +offset: U32,+span: U32,+actual_span: U32,+actual: Tree.Tree<F32>,+evidence: Evidence(Source{activation,problem,storage,offset,span,actual_span,actual}),+row: U32,+column: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(U32.to_nat(column),count),U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span))) == True{} : Bool}) ->
  {Contiguous.writes(~F32,count,actual,column,(row * actual_span : U32),(row * Padded.spatial(problem) + offset : U32)) ==
    mapped(~ActivationProofs.activation_value(~custom,activation),Contiguous.writes(~F32,count,MicroLeaf.leaf_output(problem,storage,offset,span),column,(row * span : U32),(row * Padded.spatial(problem) + offset : U32))) : List<&2,Writes.Assignment<F32>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +in_span = BlockWrites.first(
        {Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool},
        {Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(column)),GatherCopyCoordinates.spatial(problem)) == True{} : Bool},
        BlockWrites.width_facts(limits,problem,offset,span,column,valid,supported,
          Order.room_index(U32.to_nat(column),rest,U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span)),remaining)))
      +fast = Support.fast(limits,problem,supported)
      %Equal.sym(F32,
        Tree.read(F32,actual,BlockCoordinates.source(row,actual_span,column)),
        ActivationProofs.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),BlockCoordinates.source(row,span,column))),
        reads(Source{activation,problem,storage,offset,span,actual_span,actual},evidence,row,column,row_inside,Order.room_index(U32.to_nat(column),rest,U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span)),remaining))) :
        {Writes.Put{BlockCoordinates.destination(row,Padded.spatial(problem),offset,column),_} <>
          Contiguous.writes(~F32,rest,actual,(column + 1 : U32),(row * actual_span : U32),(row * Padded.spatial(problem) + offset : U32)) ==
          mapped(~ActivationProofs.activation_value(~custom,activation),Contiguous.writes(~F32,1n+rest,MicroLeaf.leaf_output(problem,storage,offset,span),column,(row * span : U32),(row * Padded.spatial(problem) + offset : U32))) : List<&2,Writes.Assignment<F32>>}
      Equal.cong(List<&2,Writes.Assignment<F32>>,List<&2,Writes.Assignment<F32>>,tail =>
        Writes.Put{BlockCoordinates.destination(row,Padded.spatial(problem),offset,column),ActivationProofs.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),BlockCoordinates.source(row,span,column)))} <> tail,
        Contiguous.writes(~F32,rest,actual,(column + 1 : U32),(row * actual_span : U32),(row * Padded.spatial(problem) + offset : U32)),
        mapped(~ActivationProofs.activation_value(~custom,activation),Contiguous.writes(~F32,rest,MicroLeaf.leaf_output(problem,storage,offset,span),(column + 1 : U32),(row * span : U32),(row * Padded.spatial(problem) + offset : U32))),
        columns_mapped(~custom,~Evidence,~reads,activation,rest,limits,problem,storage,offset,span,actual_span,actual,evidence,row,(column + 1 : U32),valid,supported,span_bound,row_inside,
          %Equal.sym(Nat,U32.to_nat((column + 1 : U32)),1n+U32.to_nat(column),BlockWrites.next(column,U32.to_nat(span),Machine.capacity(limits),Limits.capacity_fits(limits),in_span,
            Output.span_in_capacity(limits,problem,span,Guards.valid(problem,valid),fast,span_bound))) :
            {Nat.is_le(Nat.add(_,rest),U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span))) == True{} : Bool}
          Order.advance_room(U32.to_nat(column),rest,U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span)),remaining)))

def channels_mapped(~custom: F32 -> F32,~Evidence: Source -> Data,~reads: @+source: Source -> @evidence: Evidence(source) -> SourceReads(~custom,source),+activation: Activations.Activation,+count: Nat,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +offset: U32,+span: U32,+actual_span: U32,+actual: Tree.Tree<F32>,+evidence: Evidence(Source{activation,problem,storage,offset,span,actual_span,actual}),+row: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(U32.to_nat(row),count),Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {Blocks.writes(count,row,Gather.gather_width(offset,Padded.spatial(problem),span),offset,Padded.spatial(problem),actual_span,
      actual) ==
    mapped(~ActivationProofs.activation_value(~custom,activation),Blocks.writes(count,row,Gather.gather_width(offset,Padded.spatial(problem),span),offset,Padded.spatial(problem),span,MicroLeaf.leaf_output(problem,storage,offset,span))) : List<&2,Writes.Assignment<F32>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +width = Gather.gather_width(offset,Padded.spatial(problem),span)
      +original = MicroLeaf.leaf_output(problem,storage,offset,span)
      +activated = actual
      +in_row = Order.room_index(U32.to_nat(row),rest,Guards.output_channel_count(problem),remaining)
      +left = Contiguous.writes(~F32,U32.to_nat(width),original,0,(row * span : U32),(row * Padded.spatial(problem) + offset : U32))
      +right = Blocks.writes(rest,(row + 1 : U32),width,offset,Padded.spatial(problem),span,original)
      %Equal.sym(List<&2,Writes.Assignment<F32>>,mapped(~ActivationProofs.activation_value(~custom,activation),List.append(&2,Writes.Assignment<F32>,left,right)),
        List.append(&2,Writes.Assignment<F32>,mapped(~ActivationProofs.activation_value(~custom,activation),left),mapped(~ActivationProofs.activation_value(~custom,activation),right)),
        mapped_concatenation(~ActivationProofs.activation_value(~custom,activation),left,right)) :
        {List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(width),activated,0,(row * actual_span : U32),(row * Padded.spatial(problem) + offset : U32)),
            Blocks.writes(rest,(row + 1 : U32),width,offset,Padded.spatial(problem),actual_span,activated)) == _ : List<&2,Writes.Assignment<F32>>}
      %Equal.sym(List<&2,Writes.Assignment<F32>>,Contiguous.writes(~F32,U32.to_nat(width),activated,0,(row * actual_span : U32),(row * Padded.spatial(problem) + offset : U32)),mapped(~ActivationProofs.activation_value(~custom,activation),left),
        columns_mapped(~custom,~Evidence,~reads,activation,U32.to_nat(width),limits,problem,storage,offset,span,actual_span,actual,evidence,row,0,valid,supported,span_bound,in_row,Order.reflexive(U32.to_nat(width)))) :
        {List.append(&2,Writes.Assignment<F32>,_,Blocks.writes(rest,(row + 1 : U32),width,offset,Padded.spatial(problem),actual_span,activated)) ==
          List.append(&2,Writes.Assignment<F32>,mapped(~ActivationProofs.activation_value(~custom,activation),left),mapped(~ActivationProofs.activation_value(~custom,activation),right)) : List<&2,Writes.Assignment<F32>>}
      Equal.cong(List<&2,Writes.Assignment<F32>>,List<&2,Writes.Assignment<F32>>,right => List.append(&2,Writes.Assignment<F32>,mapped(~ActivationProofs.activation_value(~custom,activation),left),right),
        Blocks.writes(rest,(row + 1 : U32),width,offset,Padded.spatial(problem),actual_span,activated),mapped(~ActivationProofs.activation_value(~custom,activation),right),
        channels_mapped(~custom,~Evidence,~reads,activation,rest,limits,problem,storage,offset,span,actual_span,actual,evidence,(row + 1 : U32),valid,supported,span_bound,
          %Equal.sym(Nat,U32.to_nat((row + 1 : U32)),1n+U32.to_nat(row),BlockWrites.next(row,Guards.output_channel_count(problem),Machine.capacity(limits),Limits.capacity_fits(limits),in_row,
            Output.channel_capacity_bound(limits,problem,Support.fast(limits,problem,supported)))) : {Nat.is_le(Nat.add(_,rest),Guards.output_channel_count(problem)) == True{} : Bool}
          Order.advance_room(U32.to_nat(row),rest,Guards.output_channel_count(problem),remaining)))

def block_mapped(~custom: F32 -> F32,~Evidence: Source -> Data,~reads: @+source: Source -> @evidence: Evidence(source) -> SourceReads(~custom,source),+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+offset: U32,+span: U32,+actual_span: U32,+actual: Tree.Tree<F32>,+evidence: Evidence(Source{activation,problem,storage,offset,span,actual_span,actual}),
  +same_width: {Gather.gather_width(offset,Padded.spatial(problem),actual_span) == Gather.gather_width(offset,Padded.spatial(problem),span) : U32},
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}) ->
  {Blocks.block_writes(MicroLeaf.rows(problem),offset,Padded.spatial(problem),actual_span,actual) ==
    mapped(~ActivationProofs.activation_value(~custom,activation),Blocks.block_writes(MicroLeaf.rows(problem),offset,Padded.spatial(problem),span,MicroLeaf.leaf_output(problem,storage,offset,span))) : List<&2,Writes.Assignment<F32>>}:
  %Equal.sym(U32,Gather.gather_width(offset,Padded.spatial(problem),actual_span),Gather.gather_width(offset,Padded.spatial(problem),span),same_width) :
    {Blocks.writes(U32.to_nat(MicroLeaf.rows(problem)),0,_,offset,Padded.spatial(problem),actual_span,actual) == mapped(~ActivationProofs.activation_value(~custom,activation),Blocks.block_writes(MicroLeaf.rows(problem),offset,Padded.spatial(problem),span,MicroLeaf.leaf_output(problem,storage,offset,span))) : List<&2,Writes.Assignment<F32>>}
  channels_mapped(~custom,~Evidence,~reads,activation,U32.to_nat(MicroLeaf.rows(problem)),limits,problem,storage,offset,span,actual_span,actual,evidence,0,valid,supported,span_bound,
    %Equal.sym(Nat,U32.to_nat(MicroLeaf.rows(problem)),Guards.output_channel_count(problem),SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))) :
      {Nat.is_le(Nat.add(0n,_),Guards.output_channel_count(problem)) == True{} : Bool}
    Order.reflexive(Guards.output_channel_count(problem)))

def partition_writes(~custom: F32 -> F32,+activation: Activations.Activation,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,depth: Nat,offset: U32,span: U32,window: Execution.InputWindow) -> List<&2,Writes.Assignment<F32>>:
  GatherWrites.gather_writes(model_partition(~custom,depth,Pipeline.Task{Inference.work(problem),activation},offset,span,window,GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),Padded.spatial(problem))

def layout_writes(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,depth: Nat,offset: U32,span: U32) -> List<&2,Writes.Assignment<F32>>:
  GatherWrites.gather_writes(MicroLeaf.layout_result(problem,storage,TileLayout.divide(depth,offset,span)),Padded.spatial(problem))

def leaf_writes(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +offset: U32,+span: U32,+window: Execution.InputWindow,+state: Windows.State<problem,window,offset,span>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+span_alignment: TilePartitionBounds.BlockAligned<span>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool}) ->
  {partition_writes(~custom,activation,problem,storage,0n,offset,span,window) == mapped(~ActivationProofs.activation_value(~custom,activation),layout_writes(problem,storage,0n,offset,span)) : List<&2,Writes.Assignment<F32>>}:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Windows.State{+tree,+first,+rows,same,+cover,+contents,+room} = state
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  +count = leaf_count(problem,offset,span)
  +actual = window_output(~custom,activation,problem,storage,offset,count,tree,first,rows)
  %Equal.sym(Execution.InputWindow,window,Execution.InputWindow{tree,first,rows},same) :
    {partition_writes(~custom,activation,problem,storage,0n,offset,span,_) == mapped(~ActivationProofs.activation_value(~custom,activation),layout_writes(problem,storage,0n,offset,span)) : List<&2,Writes.Assignment<F32>>}
  block_mapped(~custom,~blocked_evidence(~custom),~blocked_reads(~custom),activation,limits,problem,storage,offset,span,count,actual,
    BlockedEvidence{limits,valid,supported,span_bound,span_alignment,range,tree,first,rows,cover,contents,room,{==},{==}},clipped_width(offset,Padded.spatial(problem),span),valid,supported,span_bound)

# Every split preserves its window evidence; every leaf uses the same blocked
# readback. The resulting gather has the established addresses and values.
def partition_mapped(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +depth: Nat,+offset: U32,+span: U32,+window: Execution.InputWindow,+state: Windows.State<problem,window,offset,span>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+span_alignment: TilePartitionBounds.BlockAligned<span>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool}) ->
  {partition_writes(~custom,activation,problem,storage,depth,offset,span,window) == mapped(~ActivationProofs.activation_value(~custom,activation),layout_writes(problem,storage,depth,offset,span)) : List<&2,Writes.Assignment<F32>>}:
  match depth:
    case 0n: leaf_writes(~custom,activation,limits,problem,storage,offset,span,window,state,valid,supported,span_bound,span_alignment,range)
    case 1n+ +rest:
      +capacity = Machine.capacity(limits)
      +fits = Limits.capacity_fits(limits)
      +padded = Support.padded(problem)
      +padded_room = Padded.padded_bound(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported))
      +capacity_room = Order.transitive(Nat.add(U32.to_nat(offset),U32.to_nat(span)),padded,U32.to_nat(capacity),range,padded_room)
      +bounded = Order.transitive(U32.to_nat(span),padded,U32.to_nat(capacity),span_bound,padded_room)
      +half = TilePartitionBounds.half(span)
      +pair = Execution.split(Inference.work(problem),offset,half,window)
      +kids = Windows.children(limits,problem,valid,supported,offset,span,capacity_room,range,window,state)
      +left_window = PartitionLaws.first(~Execution.InputWindow,pair)
      +right_window = PartitionLaws.second(~Execution.InputWindow,pair)
      +left_bound = Order.transitive(U32.to_nat(half),U32.to_nat(span),padded,TilePartitionBounds.half_le(span,capacity,fits,bounded),span_bound)
      +right_bound = Order.transitive(U32.to_nat((span - half : U32)),U32.to_nat(span),padded,TilePartitionBounds.remainder_le(span,capacity,fits,bounded),span_bound)
      +left = layout_writes(problem,storage,rest,offset,half)
      +right = layout_writes(problem,storage,rest,(offset + half : U32),(span - half : U32))
      %Equal.sym(List<&2,Writes.Assignment<F32>>,mapped(~ActivationProofs.activation_value(~custom,activation),List.append(&2,Writes.Assignment<F32>,left,right)),List.append(&2,Writes.Assignment<F32>,mapped(~ActivationProofs.activation_value(~custom,activation),left),mapped(~ActivationProofs.activation_value(~custom,activation),right)),mapped_concatenation(~ActivationProofs.activation_value(~custom,activation),left,right)) :
        {List.append(&2,Writes.Assignment<F32>,partition_writes(~custom,activation,problem,storage,rest,offset,half,left_window),partition_writes(~custom,activation,problem,storage,rest,(offset + half : U32),(span - half : U32),right_window)) == _ : List<&2,Writes.Assignment<F32>>}
      %Equal.sym(List<&2,Writes.Assignment<F32>>,partition_writes(~custom,activation,problem,storage,rest,offset,half,left_window),mapped(~ActivationProofs.activation_value(~custom,activation),left),
        partition_mapped(~custom,activation,limits,problem,storage,rest,offset,half,left_window,Windows.left_state(problem,pair,offset,span,kids),valid,supported,left_bound,TilePartitionBounds.half_alignment(span,capacity,fits,bounded),TilePartitionBounds.left_interval_bound(offset,span,capacity,padded,fits,capacity_room,range))) :
        {List.append(&2,Writes.Assignment<F32>,_,partition_writes(~custom,activation,problem,storage,rest,(offset + half : U32),(span - half : U32),right_window)) == List.append(&2,Writes.Assignment<F32>,mapped(~ActivationProofs.activation_value(~custom,activation),left),mapped(~ActivationProofs.activation_value(~custom,activation),right)) : List<&2,Writes.Assignment<F32>>}
      Equal.cong(List<&2,Writes.Assignment<F32>>,List<&2,Writes.Assignment<F32>>,tail => List.append(&2,Writes.Assignment<F32>,mapped(~ActivationProofs.activation_value(~custom,activation),left),tail),
        partition_writes(~custom,activation,problem,storage,rest,(offset + half : U32),(span - half : U32),right_window),mapped(~ActivationProofs.activation_value(~custom,activation),right),
        partition_mapped(~custom,activation,limits,problem,storage,rest,(offset + half : U32),(span - half : U32),right_window,Windows.right_state(problem,pair,offset,span,kids),valid,supported,right_bound,TilePartitionBounds.remainder_alignment(span,capacity,fits,bounded,span_alignment),TilePartitionBounds.right_interval_bound(offset,span,capacity,padded,fits,capacity_room,range)))

def selected_depth(+problem: ConvSpec.Problem,+depth: U32) -> Nat:
  +blocks = (Padded.padded(problem) / 8 : U32)
  U32.to_nat(U32.min(depth,(U32.from_nat(Tensors.significant_bits(32n,blocks,0n,U32.is_zero(blocks))) - 1 : U32)))

def selected_depth_exact(+problem: ConvSpec.Problem,+depth: U32) -> {selected_depth(problem,depth) == PreparedOutputReadback.selected_depth(problem,depth) : Nat}:
  +blocks = (Padded.padded(problem) / 8 : U32)
  Equal.cong(Nat,Nat,bits => U32.to_nat(U32.min(depth,(U32.from_nat(bits) - 1 : U32))),
    Tensors.significant_bits(32n,blocks,0n,U32.is_zero(blocks)),Storage.bit_length(32n,blocks),Depth.countdown(32n,blocks,0n))

def selected_partitions(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {partition_writes(~custom,activation,problem,storage,selected_depth(problem,depth),0,Padded.padded(problem),Windows.root_window(problem,storage)) ==
    mapped(~ActivationProofs.activation_value(~custom,activation),GatherTreeReadback.writes(problem,storage,PreparedOutputReadback.tiles(problem,storage,depth))) : List<&2,Writes.Assignment<F32>>}:
  +span = Padded.padded(problem)
  +span_exact = Padded.padded_exact(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported))
  +span_bound = {%Equal.sym(Nat,U32.to_nat(span),Support.padded(problem),span_exact) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    Order.reflexive(Support.padded(problem)) : {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}}
  %Equal.sym(TileLayout.Layout(),TileLayout.tiles(PreparedOutputReadback.tiles(problem,storage,depth)),TileLayout.divide(PreparedOutputReadback.selected_depth(problem,depth),0,span),PreparedOutputReadback.layout_exact(problem,storage,depth)) :
    {partition_writes(~custom,activation,problem,storage,selected_depth(problem,depth),0,span,Windows.root_window(problem,storage)) ==
      mapped(~ActivationProofs.activation_value(~custom,activation),GatherWrites.gather_writes(MicroLeaf.layout_result(problem,storage,_),Padded.spatial(problem))) : List<&2,Writes.Assignment<F32>>}
  %Equal.sym(Nat,selected_depth(problem,depth),PreparedOutputReadback.selected_depth(problem,depth),selected_depth_exact(problem,depth)) :
    {partition_writes(~custom,activation,problem,storage,_,0,span,Windows.root_window(problem,storage)) == mapped(~ActivationProofs.activation_value(~custom,activation),layout_writes(problem,storage,PreparedOutputReadback.selected_depth(problem,depth),0,span)) : List<&2,Writes.Assignment<F32>>}
  partition_mapped(~custom,activation,limits,problem,storage,PreparedOutputReadback.selected_depth(problem,depth),0,span,Windows.root_window(problem,storage),Windows.root_state(limits,problem,storage,valid,supported),valid,supported,span_bound,
    TilePartitionBounds.BlockAligned{Nat.div(Nat.add(GatherCopyCoordinates.spatial(problem),7n),8n),span_exact},span_bound)

def output_readback_query_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+channel: Nat,+row: Nat,+column: Nat,+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) == GatherLeafWrites.query(problem,channel,row,column) : Nat}:
  +query = GatherLeafWrites.query(problem,channel,row,column)
  Bounds.roundtrip(query,Machine.capacity(limits),Limits.capacity_fits(limits),Bounds.strict_le(query,U32.to_nat(Machine.capacity(limits)),
    Bounds.nat_trans(query,Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(Machine.capacity(limits)),
      PreparedOutputReadback.query_inside_count(problem,channel,row,column,channel_inside,row_inside,column_inside),Padded.scalar_output_bound(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported))))))

def gather_ready(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+tiles: PackingStorage.Tiles(),
  +outputs: GatherModel.ConvResult(),
  +writes_equal: {GatherWrites.gather_writes(outputs,Padded.spatial(problem)) == mapped(~ActivationProofs.activation_value(~custom,activation),GatherTreeReadback.writes(problem,storage,tiles)) : List<&2,Writes.Assignment<F32>>},
  +channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +certificates: Prepared.AllCertificates(problem,tiles),
  +member: Members.Member(Nat,Coordinates.position(problem,row,column),TilePartitionCoverage.positions(TileLayout.tiles(tiles))),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
  ready: PackedOutputAllocation.Sized<PreparedOutputReadback.count(problem),Storage.allocation_depth(PreparedOutputReadback.count(problem))>) ->
  {Tree.read(F32,GatherModel.gather(outputs,PreparedOutputReadback.initial(problem),Padded.spatial(problem)),
      U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  PackedOutputAllocation.Sized{+depth_ok,+allocation_room} = ready
  +output = PreparedOutputReadback.initial(problem)
  +depth = Storage.allocation_depth(PreparedOutputReadback.count(problem))
  +query = GatherLeafWrites.query(problem,channel,row,column)
  +native_query = U32.from_nat(query)
  +exact = output_readback_query_exact(limits,problem,channel,row,column,supported,channel_inside,row_inside,column_inside)
  +room = {
    %PreparedOutputReadback.count_exact(limits,problem,valid,supported) : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool}
    allocation_room : {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool}}
  +inside = Bounds.native_bound(query,Tree.capacity(F32,output),exact,
    Bounds.nat_trans(query,Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(Tree.capacity(F32,output)),
      PreparedOutputReadback.query_inside_count(problem,channel,row,column,channel_inside,row_inside,column_inside),room))
  %Equal.sym(Tree.Tree<F32>,GatherModel.gather(outputs,output,Padded.spatial(problem)),
    Writes.pure(F32,GatherWrites.gather_writes(outputs,Padded.spatial(problem)),output),
    GatherWrites.gather_matches_writes(outputs,output,Padded.spatial(problem))) :
    {Tree.read(F32,_,native_query) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
  %Equal.sym(List<&2,Writes.Assignment<F32>>,GatherWrites.gather_writes(outputs,Padded.spatial(problem)),
    mapped(~ActivationProofs.activation_value(~custom,activation),GatherTreeReadback.writes(problem,storage,tiles)),writes_equal) :
    {Tree.read(F32,Writes.pure(F32,_,output),native_query) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
  mapped_readback(~ActivationProofs.activation_value(~custom,activation),GatherTreeReadback.writes(problem,storage,tiles),output,depth,native_query,Values.matrix_value(problem,channel,row,column),Shape.fresh(F32,depth,0.0),depth_ok,
    %Equal.sym(Nat,U32.to_nat(native_query),query,exact) :
      WriteReadback.WriteEvidence<_,Values.matrix_value(problem,channel,row,column),Tree.capacity(F32,output),GatherTreeReadback.writes(problem,storage,tiles)>
    GatherTreeReadback.tree_evidence(limits,problem,storage,tiles,channel,row,column,Tree.capacity(F32,output),valid,supported,certificates,room,row_inside,column_inside),inside,
    %Equal.sym(Nat,U32.to_nat(native_query),query,exact) : {GatherWrites.contains(_,GatherTreeReadback.writes(problem,storage,tiles)) == True{} : Bool}
    GatherTreeReadback.tree_coverage(limits,problem,storage,tiles,channel,row,column,valid,supported,certificates,member,channel_inside,row_inside,column_inside))

def root_count(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {leaf_count(problem,0,Padded.padded(problem)) == Padded.spatial(problem) : U32}:
  +spatial = Padded.spatial(problem)
  +padded = Padded.padded(problem)
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +ordered = {%Equal.sym(Nat,U32.to_nat(spatial),GatherCopyCoordinates.spatial(problem),Padded.spatial_exact(limits,problem,facts,fast)) : {Nat.is_le(_,U32.to_nat(padded)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(padded),Support.padded(problem),Padded.padded_exact(limits,problem,facts,fast)) : {Nat.is_le(GatherCopyCoordinates.spatial(problem),_) == True{} : Bool}
    Padded.spatial_within_padded(limits,problem,facts,fast) : {Nat.is_le(U32.to_nat(spatial),U32.to_nat(padded)) == True{} : Bool}}
  %Equal.sym(U32,(spatial - 0 : U32),spatial,Arithmetic.subtract_zero(spatial)) : {U32.min(padded,_) == spatial : U32}
  %Equal.sym(U32,U32.min(padded,spatial),Bool.pick(U32,U32.is_lt(padded,spatial),padded,spatial),U32Extrema.minimum_pick(padded,spatial)) : {_ == spatial : U32}
  %Equal.sym(Bool,U32.is_lt(padded,spatial),Nat.is_lt(U32.to_nat(padded),U32.to_nat(spatial)),Comparison.lt(padded,spatial)) : {Bool.pick(U32,_,padded,spatial) == spatial : U32}
  %Equal.sym(Bool,Nat.is_lt(U32.to_nat(padded),U32.to_nat(spatial)),False{},Order.not_below(U32.to_nat(spatial),U32.to_nat(padded),ordered)) : {Bool.pick(U32,_,padded,spatial) == spatial : U32}
  {==}

def root_finished(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Execution.finish(model_partition(~custom,0n,Pipeline.Task{Inference.work(problem),activation},0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),MicroLeaf.rows(problem),Padded.spatial(problem)) ==
    Blocked.run(~custom,activation,Inference.work(problem),0,Inputs.input_tree(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)) : Tree.Tree<F32>}:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  +spatial = Padded.spatial(problem)
  +full = Blocked.run(~custom,activation,Inference.work(problem),0,Inputs.input_tree(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage))
  %Equal.sym(U32,leaf_count(problem,0,Padded.padded(problem)),spatial,root_count(limits,problem,valid,supported)) :
    {Execution.finish(Partition.Leaf{window_output(~custom,activation,problem,storage,0,_,Inputs.input_tree(problem,storage),0,U32.from_nat(height)),0,_,MicroLeaf.rows(problem)},MicroLeaf.rows(problem),spatial) == full : Tree.Tree<F32>}
  %Equal.sym(Bool,U32.is_eq(spatial,spatial),Nat.is_eq(U32.to_nat(spatial),U32.to_nat(spatial)),Comparison.eq(spatial,spatial)) :
    {Execution.finish_leaf(_,full,MicroLeaf.rows(problem),0,spatial,spatial) == full : Tree.Tree<F32>}
  %Equal.sym(Bool,Nat.is_eq(U32.to_nat(spatial),U32.to_nat(spatial)),True{},Order.equal_reflexive(U32.to_nat(spatial))) :
    {Execution.finish_leaf(_,full,MicroLeaf.rows(problem),0,spatial,spatial) == full : Tree.Tree<F32>}
  {==}

def full_output_readback(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,Blocked.run(~custom,activation,Inference.work(problem),0,Inputs.input_tree(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) ==
    ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  %Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported)) :
    {Tree.read(F32,Blocked.run(~custom,activation,Inference.work(problem),0,Inputs.input_tree(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),U32.from_nat(Nat.add(Nat.mul(channel,_),Coordinates.position(problem,row,column)))) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
  BlockedInitial.full_readback(~custom,activation,limits,problem,storage,channel,row,column,valid,supported,channel_inside,row_inside,column_inside)

def partition_finished_readback(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,+selected: Nat,
  +selection: {selected_depth(problem,depth) == selected : Nat},
  +channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,Execution.finish(model_partition(~custom,selected,Pipeline.Task{Inference.work(problem),activation},0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),MicroLeaf.rows(problem),Padded.spatial(problem)),U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) ==
    ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  match selected:
    case 0n:
      %Equal.sym(Tree.Tree<F32>,Execution.finish(model_partition(~custom,0n,Pipeline.Task{Inference.work(problem),activation},0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),MicroLeaf.rows(problem),Padded.spatial(problem)),Blocked.run(~custom,activation,Inference.work(problem),0,Inputs.input_tree(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),root_finished(~custom,activation,limits,problem,storage,valid,supported)) :
        {Tree.read(F32,_,U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
      full_output_readback(~custom,activation,limits,problem,storage,channel,row,column,valid,supported,channel_inside,row_inside,column_inside)
    case 1n+ +rest:
      +outputs = model_partition(~custom,1n+rest,Pipeline.Task{Inference.work(problem),activation},0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage))
      +tiles = PreparedOutputReadback.tiles(problem,storage,depth)
      +count = PreparedOutputReadback.count(problem)
      %Equal.sym(Nat,Tensors.depth_for(count),Storage.allocation_depth(count),Depth.positive_depth(count,Inference.count_positive(limits,problem,valid,supported))) :
        {Tree.read(F32,GatherModel.gather(outputs,Tree.fresh(F32,_,0.0),Padded.spatial(problem)),U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
      gather_ready(~custom,activation,limits,problem,storage,tiles,outputs,
        %selection : {partition_writes(~custom,activation,problem,storage,_,0,Padded.padded(problem),Windows.root_window(problem,storage)) == mapped(~ActivationProofs.activation_value(~custom,activation),GatherTreeReadback.writes(problem,storage,tiles)) : List<&2,Writes.Assignment<F32>>}
        selected_partitions(~custom,activation,limits,problem,storage,depth,valid,supported),
        channel,row,column,valid,supported,PreparedOutputReadback.certificates(limits,problem,storage,depth,valid,supported),PreparedOutputReadback.position_member(limits,problem,storage,depth,row,column,valid,supported,row_inside,column_inside),channel_inside,row_inside,column_inside,
        PreparedOutputReadback.allocated(limits,problem,channel,row,column,valid,supported,channel_inside,row_inside,column_inside))

def partitioned_readback(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,
  +channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,run_partitioned_model(~custom,activation,depth,Inference.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)),U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) ==
    ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  partition_finished_readback(~custom,activation,limits,problem,storage,depth,selected_depth(problem,depth),{==},channel,row,column,valid,supported,channel_inside,row_inside,column_inside)

def selected_readback(~custom: F32 -> F32,+blocked_leaf: Bool,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,
  +channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,run_selected_model(~custom,blocked_leaf,activation,depth,Inference.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)),U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) ==
    ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  match blocked_leaf:
    case False{}:
      partitioned_readback(~custom,activation,limits,problem,storage,depth,channel,row,column,valid,supported,channel_inside,row_inside,column_inside)
    case True{}:
      ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
      ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
      +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
      full_output_readback(~custom,activation,limits,problem,storage,channel,row,column,valid,supported,channel_inside,row_inside,column_inside)

def run_readback(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,
  +channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,run_model(~custom,activation,depth,Inference.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)),U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) ==
    ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  selected_readback(~custom,U32.is_zero(depth),activation,limits,problem,storage,depth,channel,row,column,valid,supported,channel_inside,row_inside,column_inside)

# Finite dense values are produced from the actual selected partition. Flat
# indices are decoded with positive spatial radices, then the established
# activated leaf readback supplies each value. No successful run is assumed.
type Logical is Data:
  Logical{problem: ConvSpec.Problem,activation: Activations.Activation}

def natural_value(~custom: F32 -> F32,+problem: ConvSpec.Problem,+activation: Activations.Activation,+index: Nat) -> F32:
  +spatial = GatherCopyCoordinates.spatial(problem)
  +position = Nat.mod(index,spatial)
  ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,Nat.div(index,spatial),Nat.div(position,ConvSpec.out_width(problem)),Nat.mod(position,ConvSpec.out_width(problem))))

def original(~custom: F32 -> F32,logical: Logical,index: U32) -> F32:
  Logical{problem,activation} = logical
  natural_value(~custom,problem,activation,U32.to_nat(index))

type Decoded<-problem: ConvSpec.Problem,-index: Nat> is Data:
  Decoded{channel: Nat,row: Nat,column: Nat,
    channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
    row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},
    column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
    query_exact: {GatherLeafWrites.query(problem,channel,row,column) == index : Nat},
    channel_exact: {channel == Nat.div(index,GatherCopyCoordinates.spatial(problem)) : Nat},
    row_exact: {row == Nat.div(Nat.mod(index,GatherCopyCoordinates.spatial(problem)),ConvSpec.out_width(problem)) : Nat},
    column_exact: {column == Nat.mod(Nat.mod(index,GatherCopyCoordinates.spatial(problem)),ConvSpec.out_width(problem)) : Nat}}

def decoded(+problem: ConvSpec.Problem,+index: Nat,
  +inside: {Nat.is_lt(index,Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem))) == True{} : Bool}) -> Decoded<problem,index>:
  +spatial = GatherCopyCoordinates.spatial(problem)
  +width = ConvSpec.out_width(problem)
  +position = Nat.mod(index,spatial)
  +channel = Nat.div(index,spatial)
  +row = Nat.div(position,width)
  +column = Nat.mod(position,width)
  +spatial_positive = Inference.spatial_positive(problem)
  +width_positive = Coordinates.width_positive(problem)
  +position_inside = DivisionOrder.remainder_small(index,spatial,spatial_positive)
  Decoded{channel,row,column,
    RowMajor.cancel_multiply_lt(channel,Guards.output_channel_count(problem),spatial,
      Order.le_lt_trans(Nat.mul(channel,spatial),index,Nat.mul(Guards.output_channel_count(problem),spatial),DivisionOrder.product_below(index,spatial,spatial_positive),inside)),
    RowMajor.cancel_multiply_lt(row,ConvSpec.out_height(problem),width,
      Order.le_lt_trans(Nat.mul(row,width),position,spatial,DivisionOrder.product_below(position,width,width_positive),position_inside)),
    DivisionOrder.remainder_small(position,width,width_positive),
    Equal.trans(Nat,GatherLeafWrites.query(problem,channel,row,column),Nat.add(Nat.mul(channel,spatial),position),index,
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(channel,spatial),value),Nat.add(Nat.mul(row,width),column),position,
        Equal.sym(Nat,position,Nat.add(Nat.mul(row,width),column),DivisionOrder.quotient_exact(position,width,width_positive))),
      Equal.sym(Nat,index,Nat.add(Nat.mul(channel,spatial),position),DivisionOrder.quotient_exact(index,spatial,spatial_positive))),
    {==},{==},{==}}

def selected_content(~custom: F32 -> F32,+activation: Activations.Activation,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32) -> Buffer.Content<F32>:
  Buffer.Content{run_model(~custom,activation,depth,Inference.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)),U32.to_nat(PreparedOutputReadback.count(problem))}

def point_decoded(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,+index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  coordinates: Decoded<problem,index>) ->
  Dense.At(~F32,~Logical,~original(~custom),Logical{problem,activation},selected_content(~custom,activation,problem,storage,depth),index):
  Decoded{+channel,+row,+column,+channel_inside,+row_inside,+column_inside,+query_exact,+channel_exact,+row_exact,+column_exact} = coordinates
  +word_exact = {
    %query_exact : {U32.to_nat(U32.from_nat(_)) == _ : Nat}
    output_readback_query_exact(limits,problem,channel,row,column,supported,channel_inside,row_inside,column_inside) : {U32.to_nat(U32.from_nat(index)) == index : Nat}}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,word_exact) :
    {Buffer.value(~F32,U32.from_nat(index),selected_content(~custom,activation,problem,storage,depth)) == natural_value(~custom,problem,activation,_) : F32}
  %channel_exact :
    {Buffer.value(~F32,U32.from_nat(index),selected_content(~custom,activation,problem,storage,depth)) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,_,Nat.div(Nat.mod(index,GatherCopyCoordinates.spatial(problem)),ConvSpec.out_width(problem)),Nat.mod(Nat.mod(index,GatherCopyCoordinates.spatial(problem)),ConvSpec.out_width(problem)))) : F32}
  %row_exact :
    {Buffer.value(~F32,U32.from_nat(index),selected_content(~custom,activation,problem,storage,depth)) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,_,Nat.mod(Nat.mod(index,GatherCopyCoordinates.spatial(problem)),ConvSpec.out_width(problem)))) : F32}
  %column_exact :
    {Buffer.value(~F32,U32.from_nat(index),selected_content(~custom,activation,problem,storage,depth)) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,_)) : F32}
  %query_exact :
    {Tree.read(F32,run_model(~custom,activation,depth,Inference.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)),U32.from_nat(_)) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
  run_readback(~custom,activation,limits,problem,storage,depth,channel,row,column,valid,supported,channel_inside,row_inside,column_inside)

def point(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,+index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +inside: {Nat.is_lt(index,U32.to_nat(PreparedOutputReadback.count(problem))) == True{} : Bool}) ->
  Dense.At(~F32,~Logical,~original(~custom),Logical{problem,activation},selected_content(~custom,activation,problem,storage,depth),index):
  point_decoded(~custom,activation,limits,problem,storage,depth,index,valid,supported,
    decoded(problem,index,
      %PreparedOutputReadback.count_exact(limits,problem,valid,supported) : {Nat.is_lt(index,_) == True{} : Bool}
      inside))

def finite_range(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,+remaining: Nat,+start: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,remaining),U32.to_nat(PreparedOutputReadback.count(problem))) == True{} : Bool}) ->
  Members.All(Nat,{index => Dense.At(~F32,~Logical,~original(~custom),Logical{problem,activation},selected_content(~custom,activation,problem,storage,depth),index) : Nat -> Data},Ranges.range(remaining,start)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{point(~custom,activation,limits,problem,storage,depth,start,valid,supported,Order.room_index(start,rest,U32.to_nat(PreparedOutputReadback.count(problem)),room)),
        finite_range(~custom,activation,limits,problem,storage,depth,rest,1n+start,valid,supported,Order.advance_room(start,rest,U32.to_nat(PreparedOutputReadback.count(problem)),room))}

def finite_values(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  Dense.Values(~F32,~Logical,~original(~custom),Logical{problem,activation},selected_content(~custom,activation,problem,storage,depth)):
  finite_range(~custom,activation,limits,problem,storage,depth,U32.to_nat(PreparedOutputReadback.count(problem)),0n,valid,supported,Order.reflexive(U32.to_nat(PreparedOutputReadback.count(problem))))

# The runtime's left-associated output product equals the established count.
# Real semantic and machine guards supply both intermediate product bounds.
def runtime_size(+problem: ConvSpec.Problem) -> U32:
  (U32.from_nat(Guards.output_channel_count(problem)) * U32.from_nat(ConvSpec.out_height(problem)) * U32.from_nat(ConvSpec.out_width(problem)) : U32)

def left_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))),U32.to_nat(U32.from_nat(ConvSpec.out_height(problem)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +facts = Guards.valid(problem,valid)
  +scalar = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +co_exact = SupportedCoordinateBounds.co_exact(limits,problem,scalar)
  +height_exact = Bounds.roundtrip(ConvSpec.out_height(problem),Machine.coordinate(limits),Limits.coordinate_fits(limits),SupportedCoordinateBounds.row_limit(limits,problem,facts,scalar))
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))),Guards.output_channel_count(problem),co_exact) :
    {Nat.is_le(Nat.mul(_,U32.to_nat(U32.from_nat(ConvSpec.out_height(problem)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_height(problem))),ConvSpec.out_height(problem),height_exact) :
    {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Order.transitive(Nat.mul(Guards.output_channel_count(problem),ConvSpec.out_height(problem)),Nat.mul(Nat.mul(Guards.output_channel_count(problem),ConvSpec.out_height(problem)),ConvSpec.out_width(problem)),U32.to_nat(Machine.capacity(limits)),
    RowMajor.product_covers(Nat.mul(Guards.output_channel_count(problem),ConvSpec.out_height(problem)),ConvSpec.out_width(problem),Coordinates.width_positive(problem)),
    %Equal.sym(Nat,Nat.mul(Nat.mul(Guards.output_channel_count(problem),ConvSpec.out_height(problem)),ConvSpec.out_width(problem)),Nat.mul(Guards.output_channel_count(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))),RowMajor.assoc(Guards.output_channel_count(problem),ConvSpec.out_height(problem),ConvSpec.out_width(problem))) :
      {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    Padded.scalar_output_bound(limits,problem,scalar))

def right_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(U32.from_nat(ConvSpec.out_height(problem))),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +facts = Guards.valid(problem,valid)
  +scalar = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +height_exact = Bounds.roundtrip(ConvSpec.out_height(problem),Machine.coordinate(limits),Limits.coordinate_fits(limits),SupportedCoordinateBounds.row_limit(limits,problem,facts,scalar))
  +width_exact = Bounds.roundtrip(ConvSpec.out_width(problem),Machine.coordinate(limits),Limits.coordinate_fits(limits),SupportedCoordinateBounds.col_limit(limits,problem,facts,scalar))
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_height(problem))),ConvSpec.out_height(problem),height_exact) :
    {Nat.is_le(Nat.mul(_,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ConvSpec.out_width(problem),width_exact) :
    {Nat.is_le(Nat.mul(ConvSpec.out_height(problem),_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Padded.spatial_bound_facts(limits,problem,facts,scalar)

def matches(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {runtime_size(problem) == PreparedOutputReadback.count(problem) : U32}:
  Products.native(U32.from_nat(Guards.output_channel_count(problem)),U32.from_nat(ConvSpec.out_height(problem)),U32.from_nat(ConvSpec.out_width(problem)),Machine.capacity(limits),Limits.capacity_fits(limits),left_room(limits,problem,valid,supported),right_room(limits,problem,valid,supported))
