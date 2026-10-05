# Dense convolution from bands: gathered input, activation writes, output
# readback, sizes and finite values.
import Base
import ./band_leaf_execution.bend as LeafExecution
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
import ../../lib/proofs/boolean_logic.bend as Boolean
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
import ../../lib/proofs/tensor_band_range.bend as BandRange
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

def activated_point(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,+values: Tree.Tree<F32>,
  ready: Depth.Ready<count>,+balanced: Shape.Balanced(F32,Tensors.depth_for(count),values),+index: Nat,
  +inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(F32,LeafExecution.activation_tree(~custom,activation,count,values),U32.from_nat(index)) ==
    LeafExecution.activation_value(~custom,activation,Tree.read(F32,values,U32.from_nat(index))) : F32}:
  Depth.Ready{+depth_ok,+room} = ready
  LeafExecution.activated_cases(~custom,activation,count,values,Tensors.depth_for(count),Shape.capacity(F32,Tensors.depth_for(count),values,balanced),
    balanced,depth_ok,LeafExecution.power_fits(Tensors.depth_for(count),depth_ok),room,index,inside)

def leaf_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,+supported: Support.FastSupported<limits,problem>,
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(MicroLeaf.rows(problem)),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(MicroLeaf.rows(problem)),Guards.output_channel_count(problem),SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,supported))) :
    {Nat.is_le(Nat.mul(_,U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Output.elements_bound(limits,problem,span,supported,span_bound)

# Every actual gather source index lies in this leaf's activation pass. The
# leaf's own established certificate supplies balance; the existing packed
# support guard supplies its product bound, including arbitrary valid padding.
def activated_source(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +offset: U32,+span: U32,+row: U32,+column: U32,
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},
  +column_inside: {Nat.is_lt(U32.to_nat(column),U32.to_nat(span)) == True{} : Bool}) ->
  {Tree.read(F32,LeafExecution.activation_tree(~custom,activation,(MicroLeaf.rows(problem) * span : U32),MicroLeaf.leaf_output(problem,storage,offset,span)),BlockCoordinates.source(row,span,column)) ==
    LeafExecution.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),BlockCoordinates.source(row,span,column))) : F32}:
  +fast = Support.fast(limits,problem,supported)
  +rows = MicroLeaf.rows(problem)
  +count = (rows * span : U32)
  +source = BlockCoordinates.source(row,span,column)
  +room = leaf_room(limits,problem,span,fast,span_bound)
  +inside = {
    %Equal.sym(Nat,U32.to_nat(count),Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)),Output.count_exact(limits,problem,span,fast,span_bound)) :
      {Nat.is_lt(U32.to_nat(source),_) == True{} : Bool}
    BlockCoordinates.source_inside(limits,problem,span,row,column,fast,span_bound,row_inside,column_inside) : {Nat.is_lt(U32.to_nat(source),U32.to_nat(count)) == True{} : Bool}}
  %Arithmetic.native_roundtrip(source) :
    {Tree.read(F32,LeafExecution.activation_tree(~custom,activation,count,MicroLeaf.leaf_output(problem,storage,offset,span)),_) ==
      LeafExecution.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),_)) : F32}
  activated_point(~custom,activation,count,MicroLeaf.leaf_output(problem,storage,offset,span),LeafExecution.product_ready(limits,rows,span,room),
    LeafExecution.gemm_balanced(rows,span,Coordinates.native_inner(problem),MicroLeaf.packed_tiles(problem,storage,offset,span),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),
    U32.to_nat(source),inside)

# Dense execution reuses the established convolution leaf and its storage
# refinement. Activation changes only leaf values, preserving the original
# input split, output coordinates and whole partition schedule.
def activate_results(~custom: F32 -> F32,+activation: Activations.Activation,outputs: GatherModel.ConvResult()) -> GatherModel.ConvResult():
  match outputs:
    case Partition.Leaf{tree,offset,+span,+rows}: Partition.Leaf{LeafExecution.activation_tree(~custom,activation,(rows * span : U32),tree),offset,span,rows}
    case Partition.Fork{left,right}: Partition.Fork{activate_results(~custom,activation,left),activate_results(~custom,activation,right)}

def activation_results_matches(~custom: F32 -> F32,+activation: Activations.Activation,+outputs: GatherModel.ConvResult()) ->
  {DensePipeline.activate_outputs(~custom,activation,PartitionLaws.represent(~F32,outputs)) == PartitionLaws.represent(~F32,activate_results(~custom,activation,outputs)) : Partition.Tree<&1,Array<F32>,U32>}:
  match outputs:
    case Partition.Leaf{+tree,+offset,+span,+rows}:
      Equal.cong(Array<F32>,Partition.Tree<&1,Array<F32>,U32>,array => Partition.Leaf{array,offset,span,rows},
        LeafExecution.activation_array(~custom,activation,(rows * span : U32),Tree.pack(F32,tree)),Tree.pack(F32,LeafExecution.activation_tree(~custom,activation,(rows * span : U32),tree)),
        LeafExecution.activation_model(~custom,activation,(rows * span : U32),tree))
    case Partition.Fork{+left,+right}:
      %Equal.sym(Partition.Tree<&1,Array<F32>,U32>,DensePipeline.activate_outputs(~custom,activation,PartitionLaws.represent(~F32,left)),PartitionLaws.represent(~F32,activate_results(~custom,activation,left)),
        activation_results_matches(~custom,activation,left)) :
        {Partition.Fork{_,DensePipeline.activate_outputs(~custom,activation,PartitionLaws.represent(~F32,right))} ==
          Partition.Fork{PartitionLaws.represent(~F32,activate_results(~custom,activation,left)),PartitionLaws.represent(~F32,activate_results(~custom,activation,right))} : Partition.Tree<&1,Array<F32>,U32>}
      Equal.cong(Partition.Tree<&1,Array<F32>,U32>,Partition.Tree<&1,Array<F32>,U32>,right => Partition.Fork{PartitionLaws.represent(~F32,activate_results(~custom,activation,left)),right},
        DensePipeline.activate_outputs(~custom,activation,PartitionLaws.represent(~F32,right)),PartitionLaws.represent(~F32,activate_results(~custom,activation,right)),activation_results_matches(~custom,activation,right))

def model_split(task: Pipeline.Task,offset: U32,half: U32,input: Execution.InputWindow) -> PartitionLaws.Split<Execution.InputWindow>:
  Pipeline.Task{work,activation} = task
  Execution.split(work,offset,half,input)

def model_leaf(~custom: F32 -> F32,task: Pipeline.Task,offset: U32,span: U32,input: Execution.InputWindow,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> GatherModel.ConvResult():
  Pipeline.Task{work,activation} = task
  activate_results(~custom,activation,Execution.leaf(work,offset,span,input,weights,bias))

def split_matches(+task: Pipeline.Task,+offset: U32,+half: U32,+input: Execution.InputWindow) ->
  {DensePipeline.array_split(task,offset,half,Execution.represent(input)) ==
    PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~Execution.InputWindow,~Execution.represent,model_split(task,offset,half,input)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  Pipeline.Task{work,activation} = task
  Execution.split_matches(work,offset,half,input)

def leaf_matches(~custom: F32 -> F32,+task: Pipeline.Task,+offset: U32,+span: U32,+input: Execution.InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {DensePipeline.array_leaf(~custom,task,offset,span,Execution.represent(input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)) ==
    PartitionLaws.represent(~F32,model_leaf(~custom,task,offset,span,input,weights,bias)) : Partition.Tree<&1,Array<F32>,U32>}:
  Pipeline.Task{+work,+activation} = task
  Equal.trans(Partition.Tree<&1,Array<F32>,U32>,
    DensePipeline.activate_outputs(~custom,activation,ArrayPipeline.leaf(work,offset,span,Execution.represent(input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias))),
    DensePipeline.activate_outputs(~custom,activation,PartitionLaws.represent(~F32,Execution.leaf(work,offset,span,input,weights,bias))),
    PartitionLaws.represent(~F32,activate_results(~custom,activation,Execution.leaf(work,offset,span,input,weights,bias))),
    Equal.cong(Partition.Tree<&1,Array<F32>,U32>,Partition.Tree<&1,Array<F32>,U32>,outputs => DensePipeline.activate_outputs(~custom,activation,outputs),
      ArrayPipeline.leaf(work,offset,span,Execution.represent(input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)),PartitionLaws.represent(~F32,Execution.leaf(work,offset,span,input,weights,bias)),
      Execution.leaf_matches(work,offset,span,input,weights,bias)),
    activation_results_matches(~custom,activation,Execution.leaf(work,offset,span,input,weights,bias)))

def model_partition(~custom: F32 -> F32,+depth: Nat,+task: Pipeline.Task,+offset: U32,+span: U32,+input: Execution.InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) -> GatherModel.ConvResult():
  PartitionLaws.partition(~Execution.InputWindow,~Tile.Tile8,~F32,~F32,~Pipeline.Task,~model_split,~model_leaf(~custom),depth,task,offset,span,input,weights,bias)

def partition_matches(~custom: F32 -> F32,+depth: Nat,+task: Pipeline.Task,+offset: U32,+span: U32,+input: Execution.InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {DensePipeline.array_partition(~custom,depth,task,offset,span,Execution.represent(input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)) ==
    PartitionLaws.represent(~F32,model_partition(~custom,depth,task,offset,span,input,weights,bias)) : Partition.Tree<&1,Array<F32>,U32>}:
  PartitionLaws.run_matches(~ArrayPipeline.InputWindow,~Execution.InputWindow,~Tile.Tile8,~F32,~F32,~Pipeline.Task,
    ~Execution.represent,~DensePipeline.array_split,~model_split,~DensePipeline.array_leaf(~custom),~model_leaf(~custom),~split_matches,~leaf_matches(~custom),
    depth,task,offset,span,input,weights,bias)

def run_model(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,input: Tree.Tree<F32>,weights: Tree.Tree<F32>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  Pipeline.Work{+rows,+height,width,kernel,stride,padding,output_width,+positions,+inner} = work
  +padded = ((positions + 7) / 8 * 8 : U32)
  +limit = (U32.from_nat(Tensors.significant_bits(32n,(padded / 8 : U32),0n,U32.is_zero((padded / 8 : U32)))) - 1 : U32)
  Execution.finish(model_partition(~custom,U32.to_nat(U32.min(depth,limit)),Pipeline.Task{Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner},activation},0,padded,
    Execution.InputWindow{input,0,height},WeightsLaws.run(rows,inner,weights),bias),rows,positions)

def array_run_matches(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Tree.Tree<F32>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>) ->
  {DensePipeline.array_run(~custom,activation,depth,work,Tree.pack(F32,input),Tree.pack(F32,weights),Tree.pack(F32,bias)) ==
    Tree.pack(F32,run_model(~custom,activation,depth,work,input,weights,bias)) : Array<F32>}:
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

def run_refines_model(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>) ->
  {LeafExecution.storage(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) ==
    Tree.pack(F32,run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}:
  Equal.trans(Array<F32>,LeafExecution.storage(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),
    DensePipeline.array_run(~custom,activation,depth,work,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),LeafExecution.storage(~F32,Buffer.pack(~F32,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))),
    Tree.pack(F32,run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))),
    DensePipeline.run_projection(~custom,activation,depth,work,input,weights,bias),
    %Equal.sym(Array<F32>,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),Tree.pack(F32,Buffer.tree(~F32,input)),DensePipeline.storage_matches(~F32,input)) :
      {DensePipeline.array_run(~custom,activation,depth,work,_,LeafExecution.storage(~F32,Buffer.pack(~F32,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) ==
        Tree.pack(F32,run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}
    %Equal.sym(Array<F32>,LeafExecution.storage(~F32,Buffer.pack(~F32,weights)),Tree.pack(F32,Buffer.tree(~F32,weights)),DensePipeline.storage_matches(~F32,weights)) :
      {DensePipeline.array_run(~custom,activation,depth,work,Tree.pack(F32,Buffer.tree(~F32,input)),_,LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) ==
        Tree.pack(F32,run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}
    %Equal.sym(Array<F32>,LeafExecution.storage(~F32,Buffer.pack(~F32,bias)),Tree.pack(F32,Buffer.tree(~F32,bias)),DensePipeline.storage_matches(~F32,bias)) :
      {DensePipeline.array_run(~custom,activation,depth,work,Tree.pack(F32,Buffer.tree(~F32,input)),Tree.pack(F32,Buffer.tree(~F32,weights)),_) ==
        Tree.pack(F32,run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}
    array_run_matches(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)))

# The real gather reads only positions visited by each leaf's activation pass.
# Its write program therefore maps the values of the established gather while
# retaining all addresses, ordering and coverage certificates.
def columns_mapped(~custom: F32 -> F32,+activation: Activations.Activation,+count: Nat,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +offset: U32,+span: U32,+row: U32,+column: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(U32.to_nat(row),Guards.output_channel_count(problem)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(U32.to_nat(column),count),U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span))) == True{} : Bool}) ->
  {Contiguous.writes(~F32,count,LeafExecution.activation_tree(~custom,activation,(MicroLeaf.rows(problem) * span : U32),MicroLeaf.leaf_output(problem,storage,offset,span)),column,(row * span : U32),(row * Padded.spatial(problem) + offset : U32)) ==
    mapped(~LeafExecution.activation_value(~custom,activation),Contiguous.writes(~F32,count,MicroLeaf.leaf_output(problem,storage,offset,span),column,(row * span : U32),(row * Padded.spatial(problem) + offset : U32))) : List<&2,Writes.Assignment<F32>>}:
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
        Tree.read(F32,LeafExecution.activation_tree(~custom,activation,(MicroLeaf.rows(problem) * span : U32),MicroLeaf.leaf_output(problem,storage,offset,span)),BlockCoordinates.source(row,span,column)),
        LeafExecution.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),BlockCoordinates.source(row,span,column))),
        activated_source(~custom,activation,limits,problem,storage,offset,span,row,column,supported,span_bound,row_inside,in_span)) :
        {Writes.Put{BlockCoordinates.destination(row,Padded.spatial(problem),offset,column),_} <>
          Contiguous.writes(~F32,rest,LeafExecution.activation_tree(~custom,activation,(MicroLeaf.rows(problem) * span : U32),MicroLeaf.leaf_output(problem,storage,offset,span)),(column + 1 : U32),(row * span : U32),(row * Padded.spatial(problem) + offset : U32)) ==
          mapped(~LeafExecution.activation_value(~custom,activation),Contiguous.writes(~F32,1n+rest,MicroLeaf.leaf_output(problem,storage,offset,span),column,(row * span : U32),(row * Padded.spatial(problem) + offset : U32))) : List<&2,Writes.Assignment<F32>>}
      Equal.cong(List<&2,Writes.Assignment<F32>>,List<&2,Writes.Assignment<F32>>,tail =>
        Writes.Put{BlockCoordinates.destination(row,Padded.spatial(problem),offset,column),LeafExecution.activation_value(~custom,activation,Tree.read(F32,MicroLeaf.leaf_output(problem,storage,offset,span),BlockCoordinates.source(row,span,column)))} <> tail,
        Contiguous.writes(~F32,rest,LeafExecution.activation_tree(~custom,activation,(MicroLeaf.rows(problem) * span : U32),MicroLeaf.leaf_output(problem,storage,offset,span)),(column + 1 : U32),(row * span : U32),(row * Padded.spatial(problem) + offset : U32)),
        mapped(~LeafExecution.activation_value(~custom,activation),Contiguous.writes(~F32,rest,MicroLeaf.leaf_output(problem,storage,offset,span),(column + 1 : U32),(row * span : U32),(row * Padded.spatial(problem) + offset : U32))),
        columns_mapped(~custom,activation,rest,limits,problem,storage,offset,span,row,(column + 1 : U32),valid,supported,span_bound,row_inside,
          %Equal.sym(Nat,U32.to_nat((column + 1 : U32)),1n+U32.to_nat(column),BlockWrites.next(column,U32.to_nat(span),Machine.capacity(limits),Limits.capacity_fits(limits),in_span,
            Output.span_in_capacity(limits,problem,span,Guards.valid(problem,valid),fast,span_bound))) :
            {Nat.is_le(Nat.add(_,rest),U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span))) == True{} : Bool}
          Order.advance_room(U32.to_nat(column),rest,U32.to_nat(Gather.gather_width(offset,Padded.spatial(problem),span)),remaining)))

def channels_mapped(~custom: F32 -> F32,+activation: Activations.Activation,+count: Nat,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +offset: U32,+span: U32,+row: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(U32.to_nat(row),count),Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {Blocks.writes(count,row,Gather.gather_width(offset,Padded.spatial(problem),span),offset,Padded.spatial(problem),span,
      LeafExecution.activation_tree(~custom,activation,(MicroLeaf.rows(problem) * span : U32),MicroLeaf.leaf_output(problem,storage,offset,span))) ==
    mapped(~LeafExecution.activation_value(~custom,activation),Blocks.writes(count,row,Gather.gather_width(offset,Padded.spatial(problem),span),offset,Padded.spatial(problem),span,MicroLeaf.leaf_output(problem,storage,offset,span))) : List<&2,Writes.Assignment<F32>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +width = Gather.gather_width(offset,Padded.spatial(problem),span)
      +original = MicroLeaf.leaf_output(problem,storage,offset,span)
      +activated = LeafExecution.activation_tree(~custom,activation,(MicroLeaf.rows(problem) * span : U32),original)
      +in_row = Order.room_index(U32.to_nat(row),rest,Guards.output_channel_count(problem),remaining)
      +left = Contiguous.writes(~F32,U32.to_nat(width),original,0,(row * span : U32),(row * Padded.spatial(problem) + offset : U32))
      +right = Blocks.writes(rest,(row + 1 : U32),width,offset,Padded.spatial(problem),span,original)
      %Equal.sym(List<&2,Writes.Assignment<F32>>,mapped(~LeafExecution.activation_value(~custom,activation),List.append(&2,Writes.Assignment<F32>,left,right)),
        List.append(&2,Writes.Assignment<F32>,mapped(~LeafExecution.activation_value(~custom,activation),left),mapped(~LeafExecution.activation_value(~custom,activation),right)),
        mapped_concatenation(~LeafExecution.activation_value(~custom,activation),left,right)) :
        {List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(width),activated,0,(row * span : U32),(row * Padded.spatial(problem) + offset : U32)),
            Blocks.writes(rest,(row + 1 : U32),width,offset,Padded.spatial(problem),span,activated)) == _ : List<&2,Writes.Assignment<F32>>}
      %Equal.sym(List<&2,Writes.Assignment<F32>>,Contiguous.writes(~F32,U32.to_nat(width),activated,0,(row * span : U32),(row * Padded.spatial(problem) + offset : U32)),mapped(~LeafExecution.activation_value(~custom,activation),left),
        columns_mapped(~custom,activation,U32.to_nat(width),limits,problem,storage,offset,span,row,0,valid,supported,span_bound,in_row,Order.reflexive(U32.to_nat(width)))) :
        {List.append(&2,Writes.Assignment<F32>,_,Blocks.writes(rest,(row + 1 : U32),width,offset,Padded.spatial(problem),span,activated)) ==
          List.append(&2,Writes.Assignment<F32>,mapped(~LeafExecution.activation_value(~custom,activation),left),mapped(~LeafExecution.activation_value(~custom,activation),right)) : List<&2,Writes.Assignment<F32>>}
      Equal.cong(List<&2,Writes.Assignment<F32>>,List<&2,Writes.Assignment<F32>>,right => List.append(&2,Writes.Assignment<F32>,mapped(~LeafExecution.activation_value(~custom,activation),left),right),
        Blocks.writes(rest,(row + 1 : U32),width,offset,Padded.spatial(problem),span,activated),mapped(~LeafExecution.activation_value(~custom,activation),right),
        channels_mapped(~custom,activation,rest,limits,problem,storage,offset,span,(row + 1 : U32),valid,supported,span_bound,
          %Equal.sym(Nat,U32.to_nat((row + 1 : U32)),1n+U32.to_nat(row),BlockWrites.next(row,Guards.output_channel_count(problem),Machine.capacity(limits),Limits.capacity_fits(limits),in_row,
            Output.channel_capacity_bound(limits,problem,Support.fast(limits,problem,supported)))) : {Nat.is_le(Nat.add(_,rest),Guards.output_channel_count(problem)) == True{} : Bool}
          Order.advance_room(U32.to_nat(row),rest,Guards.output_channel_count(problem),remaining)))

def block_mapped(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+offset: U32,+span: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}) ->
  {Blocks.block_writes(MicroLeaf.rows(problem),offset,Padded.spatial(problem),span,LeafExecution.activation_tree(~custom,activation,(MicroLeaf.rows(problem) * span : U32),MicroLeaf.leaf_output(problem,storage,offset,span))) ==
    mapped(~LeafExecution.activation_value(~custom,activation),Blocks.block_writes(MicroLeaf.rows(problem),offset,Padded.spatial(problem),span,MicroLeaf.leaf_output(problem,storage,offset,span))) : List<&2,Writes.Assignment<F32>>}:
  channels_mapped(~custom,activation,U32.to_nat(MicroLeaf.rows(problem)),limits,problem,storage,offset,span,0,valid,supported,span_bound,
    %Equal.sym(Nat,U32.to_nat(MicroLeaf.rows(problem)),Guards.output_channel_count(problem),SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))) :
      {Nat.is_le(Nat.add(0n,_),Guards.output_channel_count(problem)) == True{} : Bool}
    Order.reflexive(Guards.output_channel_count(problem)))

def tree_mapped(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+tiles: PackingStorage.Tiles(),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificates: Prepared.AllCertificates(problem,tiles)) ->
  {GatherWrites.gather_writes(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(tiles))),Padded.spatial(problem)) ==
    mapped(~LeafExecution.activation_value(~custom,activation),GatherWrites.gather_writes(MicroLeaf.layout_result(problem,storage,TileLayout.tiles(tiles)),Padded.spatial(problem))) : List<&2,Writes.Assignment<F32>>}:
  match tiles:
    case Partition.Leaf{+input,+offset,+span,Unit{}}:
      Prepared.LeafEvidence{packing,+span_bound,offset_alignment,span_alignment,range_bound,contents} = certificates
      block_mapped(~custom,activation,limits,problem,storage,offset,span,valid,supported,span_bound)
    case Partition.Fork{+left,+right}:
      Shape.Both{left_certificates,right_certificates} = certificates
      +left_writes = GatherWrites.gather_writes(MicroLeaf.layout_result(problem,storage,TileLayout.tiles(left)),Padded.spatial(problem))
      +right_writes = GatherWrites.gather_writes(MicroLeaf.layout_result(problem,storage,TileLayout.tiles(right)),Padded.spatial(problem))
      %Equal.sym(List<&2,Writes.Assignment<F32>>,mapped(~LeafExecution.activation_value(~custom,activation),List.append(&2,Writes.Assignment<F32>,left_writes,right_writes)),
        List.append(&2,Writes.Assignment<F32>,mapped(~LeafExecution.activation_value(~custom,activation),left_writes),mapped(~LeafExecution.activation_value(~custom,activation),right_writes)),
        mapped_concatenation(~LeafExecution.activation_value(~custom,activation),left_writes,right_writes)) :
        {List.append(&2,Writes.Assignment<F32>,GatherWrites.gather_writes(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(left))),Padded.spatial(problem)),
            GatherWrites.gather_writes(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(right))),Padded.spatial(problem))) == _ : List<&2,Writes.Assignment<F32>>}
      %Equal.sym(List<&2,Writes.Assignment<F32>>,GatherWrites.gather_writes(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(left))),Padded.spatial(problem)),mapped(~LeafExecution.activation_value(~custom,activation),left_writes),
        tree_mapped(~custom,activation,limits,problem,storage,left,valid,supported,left_certificates)) :
        {List.append(&2,Writes.Assignment<F32>,_,GatherWrites.gather_writes(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(right))),Padded.spatial(problem))) ==
          List.append(&2,Writes.Assignment<F32>,mapped(~LeafExecution.activation_value(~custom,activation),left_writes),mapped(~LeafExecution.activation_value(~custom,activation),right_writes)) : List<&2,Writes.Assignment<F32>>}
      Equal.cong(List<&2,Writes.Assignment<F32>>,List<&2,Writes.Assignment<F32>>,right => List.append(&2,Writes.Assignment<F32>,mapped(~LeafExecution.activation_value(~custom,activation),left_writes),right),
        GatherWrites.gather_writes(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(right))),Padded.spatial(problem)),mapped(~LeafExecution.activation_value(~custom,activation),right_writes),
        tree_mapped(~custom,activation,limits,problem,storage,right,valid,supported,right_certificates))

# Activation commutes with the established pure partition model. The selected
# depth and root-window equality come from the actual schedule and existing
# window theorem, so the activated model has the same leaf certificates.
def partition_activated(~custom: F32 -> F32,+depth: Nat,+activation: Activations.Activation,+work: Pipeline.Work,+offset: U32,+span: U32,
  +input: Execution.InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {model_partition(~custom,depth,Pipeline.Task{work,activation},offset,span,input,weights,bias) ==
    activate_results(~custom,activation,Execution.partition(depth,work,offset,span,input,weights,bias)) : GatherModel.ConvResult()}:
  match depth:
    case 0n: {==}
    case 1n+ +rest:
      +half = (span / 16 * 8 : U32)
      +inputs = Execution.split(work,offset,half,input)
      +left_input = PartitionLaws.first(~Execution.InputWindow,inputs)
      +right_input = PartitionLaws.second(~Execution.InputWindow,inputs)
      %Equal.sym(GatherModel.ConvResult(),model_partition(~custom,rest,Pipeline.Task{work,activation},offset,half,left_input,weights,bias),
        activate_results(~custom,activation,Execution.partition(rest,work,offset,half,left_input,weights,bias)),partition_activated(~custom,rest,activation,work,offset,half,left_input,weights,bias)) :
        {Partition.Fork{_,model_partition(~custom,rest,Pipeline.Task{work,activation},(offset + half : U32),(span - half : U32),right_input,weights,bias)} ==
          Partition.Fork{activate_results(~custom,activation,Execution.partition(rest,work,offset,half,left_input,weights,bias)),
            activate_results(~custom,activation,Execution.partition(rest,work,(offset + half : U32),(span - half : U32),right_input,weights,bias))} : GatherModel.ConvResult()}
      Equal.cong(GatherModel.ConvResult(),GatherModel.ConvResult(),right => Partition.Fork{activate_results(~custom,activation,Execution.partition(rest,work,offset,half,left_input,weights,bias)),right},
        model_partition(~custom,rest,Pipeline.Task{work,activation},(offset + half : U32),(span - half : U32),right_input,weights,bias),
        activate_results(~custom,activation,Execution.partition(rest,work,(offset + half : U32),(span - half : U32),right_input,weights,bias)),
        partition_activated(~custom,rest,activation,work,(offset + half : U32),(span - half : U32),right_input,weights,bias))

def selected_depth(+problem: ConvSpec.Problem,+depth: U32) -> Nat:
  +blocks = (Padded.padded(problem) / 8 : U32)
  U32.to_nat(U32.min(depth,(U32.from_nat(Tensors.significant_bits(32n,blocks,0n,U32.is_zero(blocks))) - 1 : U32)))

def selected_depth_exact(+problem: ConvSpec.Problem,+depth: U32) -> {selected_depth(problem,depth) == PreparedOutputReadback.selected_depth(problem,depth) : Nat}:
  +blocks = (Padded.padded(problem) / 8 : U32)
  Equal.cong(Nat,Nat,bits => U32.to_nat(U32.min(depth,(U32.from_nat(bits) - 1 : U32))),
    Tensors.significant_bits(32n,blocks,0n,U32.is_zero(blocks)),Storage.bit_length(32n,blocks),Depth.countdown(32n,blocks,0n))

def selected_partitions(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {model_partition(~custom,selected_depth(problem,depth),Pipeline.Task{Inference.work(problem),activation},0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)) ==
    activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(PreparedOutputReadback.tiles(problem,storage,depth)))) : GatherModel.ConvResult()}:
  %Equal.sym(Nat,selected_depth(problem,depth),PreparedOutputReadback.selected_depth(problem,depth),selected_depth_exact(problem,depth)) :
    {model_partition(~custom,_,Pipeline.Task{Inference.work(problem),activation},0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)) ==
      activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(PreparedOutputReadback.tiles(problem,storage,depth)))) : GatherModel.ConvResult()}
  Equal.trans(GatherModel.ConvResult(),
    model_partition(~custom,PreparedOutputReadback.selected_depth(problem,depth),Pipeline.Task{Inference.work(problem),activation},0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),
    activate_results(~custom,activation,Execution.partition(PreparedOutputReadback.selected_depth(problem,depth),Inference.work(problem),0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage))),
    activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(PreparedOutputReadback.tiles(problem,storage,depth)))),
    partition_activated(~custom,PreparedOutputReadback.selected_depth(problem,depth),activation,Inference.work(problem),0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),
    Equal.trans(GatherModel.ConvResult(),
      activate_results(~custom,activation,Execution.partition(PreparedOutputReadback.selected_depth(problem,depth),Inference.work(problem),0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage))),
      activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.divide(PreparedOutputReadback.selected_depth(problem,depth),0,Padded.padded(problem)))),
      activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(PreparedOutputReadback.tiles(problem,storage,depth)))),
      Equal.cong(GatherModel.ConvResult(),GatherModel.ConvResult(),outputs => activate_results(~custom,activation,outputs),
        Execution.partition(PreparedOutputReadback.selected_depth(problem,depth),Inference.work(problem),0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),
        MicroLeaf.layout_result(problem,storage,TileLayout.divide(PreparedOutputReadback.selected_depth(problem,depth),0,Padded.padded(problem))),Windows.partition_root(limits,problem,storage,valid,supported,PreparedOutputReadback.selected_depth(problem,depth))),
      Equal.cong(TileLayout.Layout(),GatherModel.ConvResult(),layout => activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,layout)),
        TileLayout.divide(PreparedOutputReadback.selected_depth(problem,depth),0,Padded.padded(problem)),TileLayout.tiles(PreparedOutputReadback.tiles(problem,storage,depth)),
        Equal.sym(TileLayout.Layout(),TileLayout.tiles(PreparedOutputReadback.tiles(problem,storage,depth)),TileLayout.divide(PreparedOutputReadback.selected_depth(problem,depth),0,Padded.padded(problem)),PreparedOutputReadback.layout_exact(problem,storage,depth)))))

def selected_run(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {run_model(~custom,activation,depth,Inference.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)) ==
    Execution.finish(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(PreparedOutputReadback.tiles(problem,storage,depth)))),MicroLeaf.rows(problem),Padded.spatial(problem)) : Tree.Tree<F32>}:
  ConvSpec.Problem{ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding},+input_values,+weight_values,+bias_values} = problem
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  Equal.cong(GatherModel.ConvResult(),Tree.Tree<F32>,outputs => Execution.finish(outputs,MicroLeaf.rows(problem),Padded.spatial(problem)),
    model_partition(~custom,selected_depth(problem,depth),Pipeline.Task{Inference.work(problem),activation},0,Padded.padded(problem),Windows.root_window(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),
    activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(PreparedOutputReadback.tiles(problem,storage,depth)))),selected_partitions(~custom,activation,limits,problem,storage,depth,valid,supported))

# Every selected output reads the activated ordered convolution value. The
# original allocation, prepared samples and spatial membership are produced by
# the existing caller theorems; activation transports their gather evidence.
def finish_same(~custom: F32 -> F32,+activation: Activations.Activation,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+tiles: PackingStorage.Tiles(),
  +positive: {Nat.is_lt(0n,U32.to_nat(PreparedOutputReadback.count(problem))) == True{} : Bool}) ->
  {Execution.finish(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(tiles))),MicroLeaf.rows(problem),Padded.spatial(problem)) ==
    GatherModel.finish_result(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(tiles))),MicroLeaf.rows(problem),Padded.spatial(problem)) : Tree.Tree<F32>}:
  match tiles:
    case Partition.Leaf{input,+offset,+span,Unit{}}:
      Inference.finish_leaf_same(problem,U32.is_zero(offset) && U32.is_eq(span,Padded.spatial(problem)),LeafExecution.activation_tree(~custom,activation,(MicroLeaf.rows(problem) * span : U32),MicroLeaf.leaf_output(problem,storage,offset,span)),offset,span,positive)
    case Partition.Fork{+left,+right}:
      +outputs = activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(Partition.Fork{left,right})))
      Equal.cong(Nat,Tree.Tree<F32>,depth => GatherModel.gather(outputs,Tree.fresh(F32,depth,0.0),Padded.spatial(problem)),
        Tensors.depth_for(PreparedOutputReadback.count(problem)),Storage.allocation_depth(PreparedOutputReadback.count(problem)),Depth.positive_depth(PreparedOutputReadback.count(problem),positive))

def output_readback_query_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+channel: Nat,+row: Nat,+column: Nat,+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) == GatherLeafWrites.query(problem,channel,row,column) : Nat}:
  +query = GatherLeafWrites.query(problem,channel,row,column)
  Bounds.roundtrip(query,Machine.capacity(limits),Limits.capacity_fits(limits),Bounds.strict_le(query,U32.to_nat(Machine.capacity(limits)),
    Bounds.nat_trans(query,Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(Machine.capacity(limits)),
      PreparedOutputReadback.query_inside_count(problem,channel,row,column,channel_inside,row_inside,column_inside),Padded.scalar_output_bound(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported))))))

def gather_ready(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+tiles: PackingStorage.Tiles(),
  +channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +certificates: Prepared.AllCertificates(problem,tiles),
  +member: Members.Member(Nat,Coordinates.position(problem,row,column),TilePartitionCoverage.positions(TileLayout.tiles(tiles))),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
  ready: PackedOutputAllocation.Ready<PreparedOutputReadback.count(problem),Storage.allocation_depth(PreparedOutputReadback.count(problem))>) ->
  {Tree.read(F32,GatherModel.gather(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(tiles))),PreparedOutputReadback.initial(problem),Padded.spatial(problem)),
      U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) == LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  PackedOutputAllocation.Ready{+depth_ok,+allocation_room} = ready
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
  %Equal.sym(Tree.Tree<F32>,GatherModel.gather(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(tiles))),output,Padded.spatial(problem)),
    Writes.pure(F32,GatherWrites.gather_writes(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(tiles))),Padded.spatial(problem)),output),
    GatherWrites.gather_matches_writes(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(tiles))),output,Padded.spatial(problem))) :
    {Tree.read(F32,_,native_query) == LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
  %Equal.sym(List<&2,Writes.Assignment<F32>>,GatherWrites.gather_writes(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(tiles))),Padded.spatial(problem)),
    mapped(~LeafExecution.activation_value(~custom,activation),GatherTreeReadback.writes(problem,storage,tiles)),tree_mapped(~custom,activation,limits,problem,storage,tiles,valid,supported,certificates)) :
    {Tree.read(F32,Writes.pure(F32,_,output),native_query) == LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
  mapped_readback(~LeafExecution.activation_value(~custom,activation),GatherTreeReadback.writes(problem,storage,tiles),output,depth,native_query,Values.matrix_value(problem,channel,row,column),Shape.fresh(F32,depth,0.0),depth_ok,
    %Equal.sym(Nat,U32.to_nat(native_query),query,exact) :
      WriteReadback.WriteEvidence<_,Values.matrix_value(problem,channel,row,column),Tree.capacity(F32,output),GatherTreeReadback.writes(problem,storage,tiles)>
    GatherTreeReadback.tree_evidence(limits,problem,storage,tiles,channel,row,column,Tree.capacity(F32,output),valid,supported,certificates,room,row_inside,column_inside),inside,
    %Equal.sym(Nat,U32.to_nat(native_query),query,exact) : {GatherWrites.contains(_,GatherTreeReadback.writes(problem,storage,tiles)) == True{} : Bool}
    GatherTreeReadback.tree_coverage(limits,problem,storage,tiles,channel,row,column,valid,supported,certificates,member,channel_inside,row_inside,column_inside))

def finish_leaf(~custom: F32 -> F32,+identity: Bool,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +certificate: Prepared.LeafCertificate(problem,offset,span,input),
  +member: Members.Member(Nat,Coordinates.position(problem,row,column),TilePartitionCoverage.positions(TileLayout.tiles(Partition.Leaf{input,offset,span,Unit{}}))),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
  +guard: {U32.is_zero(offset) && U32.is_eq(span,Padded.spatial(problem)) == identity : Bool}) ->
  {Tree.read(F32,GatherModel.finish_leaf(identity,LeafExecution.activation_tree(~custom,activation,(MicroLeaf.rows(problem) * span : U32),MicroLeaf.leaf_output(problem,storage,offset,span)),offset,span,MicroLeaf.rows(problem),Padded.spatial(problem)),
      U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) == LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  match identity:
    case False{}:
      gather_ready(~custom,activation,limits,problem,storage,Partition.Leaf{input,offset,span,Unit{}},channel,row,column,valid,supported,certificate,member,channel_inside,row_inside,column_inside,
        PreparedOutputReadback.allocated(limits,problem,channel,row,column,valid,supported,channel_inside,row_inside,column_inside))
    case True{}:
      Prepared.LeafEvidence{+packing,+span_bound,+offset_alignment,+span_alignment,+range_bound,+contents} = certificate
      +certificate = {Prepared.LeafEvidence{packing,span_bound,offset_alignment,span_alignment,range_bound,contents} : Prepared.LeafCertificate(problem,offset,span,input)}
      +fast = Support.fast(limits,problem,supported)
      +rows = MicroLeaf.rows(problem)
      +count = (rows * span : U32)
      +query = GatherLeafWrites.query(problem,channel,row,column)
      +original = MicroLeaf.leaf_output(problem,storage,offset,span)
      +span_exact = Equal.trans(Nat,U32.to_nat(span),U32.to_nat(Padded.spatial(problem)),GatherCopyCoordinates.spatial(problem),
        Equal.cong(U32,Nat,value => U32.to_nat(value),span,Padded.spatial(problem),BandRange.same_word(span,Padded.spatial(problem),Boolean.right(U32.is_zero(offset),U32.is_eq(span,Padded.spatial(problem)),guard))),
        Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),fast))
      +inside = {
        %Equal.sym(Nat,U32.to_nat(count),Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)),Output.count_exact(limits,problem,span,fast,span_bound)) :
          {Nat.is_lt(query,_) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(span),GatherCopyCoordinates.spatial(problem),span_exact) : {Nat.is_lt(query,Nat.mul(Guards.output_channel_count(problem),_)) == True{} : Bool}
        PreparedOutputReadback.query_inside_count(problem,channel,row,column,channel_inside,row_inside,column_inside) : {Nat.is_lt(query,U32.to_nat(count)) == True{} : Bool}}
      Equal.trans(F32,Tree.read(F32,LeafExecution.activation_tree(~custom,activation,count,original),U32.from_nat(query)),
        LeafExecution.activation_value(~custom,activation,Tree.read(F32,original,U32.from_nat(query))),LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)),
        activated_point(~custom,activation,count,original,LeafExecution.product_ready(limits,rows,span,leaf_room(limits,problem,span,fast,span_bound)),
          LeafExecution.gemm_balanced(rows,span,Coordinates.native_inner(problem),MicroLeaf.packed_tiles(problem,storage,offset,span),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),query,inside),
        Equal.cong(F32,F32,value => LeafExecution.activation_value(~custom,activation,value),Tree.read(F32,original,U32.from_nat(query)),Values.matrix_value(problem,channel,row,column),
          PreparedOutputReadback.finish_leaf_readback(True{},limits,problem,storage,input,offset,span,channel,row,column,valid,supported,certificate,member,channel_inside,row_inside,column_inside,guard)))

def finish_result(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+tiles: PackingStorage.Tiles(),
  +channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +certificates: Prepared.AllCertificates(problem,tiles),
  +member: Members.Member(Nat,Coordinates.position(problem,row,column),TilePartitionCoverage.positions(TileLayout.tiles(tiles))),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,GatherModel.finish_result(activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(tiles))),MicroLeaf.rows(problem),Padded.spatial(problem)),
      U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) == LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  match tiles:
    case Partition.Leaf{+input,+offset,+span,Unit{}}:
      finish_leaf(~custom,U32.is_zero(offset) && U32.is_eq(span,Padded.spatial(problem)),activation,limits,problem,storage,input,offset,span,channel,row,column,valid,supported,certificates,member,channel_inside,row_inside,column_inside,{==})
    case Partition.Fork{+left,+right}:
      gather_ready(~custom,activation,limits,problem,storage,Partition.Fork{left,right},channel,row,column,valid,supported,certificates,member,channel_inside,row_inside,column_inside,
        PreparedOutputReadback.allocated(limits,problem,channel,row,column,valid,supported,channel_inside,row_inside,column_inside))

def run_readback(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,
  +channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,run_model(~custom,activation,depth,Inference.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)),U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))) ==
    LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  +tiles = PreparedOutputReadback.tiles(problem,storage,depth)
  +outputs = activate_results(~custom,activation,MicroLeaf.layout_result(problem,storage,TileLayout.tiles(tiles)))
  +query = U32.from_nat(GatherLeafWrites.query(problem,channel,row,column))
  %Equal.sym(Tree.Tree<F32>,run_model(~custom,activation,depth,Inference.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)),
    Execution.finish(outputs,MicroLeaf.rows(problem),Padded.spatial(problem)),selected_run(~custom,activation,limits,problem,storage,depth,valid,supported)) :
    {Tree.read(F32,_,query) == LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
  %Equal.sym(Tree.Tree<F32>,Execution.finish(outputs,MicroLeaf.rows(problem),Padded.spatial(problem)),GatherModel.finish_result(outputs,MicroLeaf.rows(problem),Padded.spatial(problem)),
    finish_same(~custom,activation,problem,storage,tiles,Inference.count_positive(limits,problem,valid,supported))) :
    {Tree.read(F32,_,query) == LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
  finish_result(~custom,activation,limits,problem,storage,tiles,channel,row,column,valid,supported,
    PreparedOutputReadback.certificates(limits,problem,storage,depth,valid,supported),PreparedOutputReadback.position_member(limits,problem,storage,depth,row,column,valid,supported,row_inside,column_inside),channel_inside,row_inside,column_inside)

# Finite dense values are produced from the actual selected partition. Flat
# indices are decoded with positive spatial radices, then the established
# activated leaf readback supplies each value. No successful run is assumed.
type Logical is Data:
  Logical{problem: ConvSpec.Problem,activation: Activations.Activation}

def natural_value(~custom: F32 -> F32,+problem: ConvSpec.Problem,+activation: Activations.Activation,+index: Nat) -> F32:
  +spatial = GatherCopyCoordinates.spatial(problem)
  +position = Nat.mod(index,spatial)
  LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,Nat.div(index,spatial),Nat.div(position,ConvSpec.out_width(problem)),Nat.mod(position,ConvSpec.out_width(problem))))

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
    {Buffer.value(~F32,U32.from_nat(index),selected_content(~custom,activation,problem,storage,depth)) == LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,_,Nat.div(Nat.mod(index,GatherCopyCoordinates.spatial(problem)),ConvSpec.out_width(problem)),Nat.mod(Nat.mod(index,GatherCopyCoordinates.spatial(problem)),ConvSpec.out_width(problem)))) : F32}
  %row_exact :
    {Buffer.value(~F32,U32.from_nat(index),selected_content(~custom,activation,problem,storage,depth)) == LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,_,Nat.mod(Nat.mod(index,GatherCopyCoordinates.spatial(problem)),ConvSpec.out_width(problem)))) : F32}
  %column_exact :
    {Buffer.value(~F32,U32.from_nat(index),selected_content(~custom,activation,problem,storage,depth)) == LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,_)) : F32}
  %query_exact :
    {Tree.read(F32,run_model(~custom,activation,depth,Inference.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)),U32.from_nat(_)) == LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
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
