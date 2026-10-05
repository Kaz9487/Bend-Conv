# The actual pipelined six-row execution equals the storage model whose every
# output is read back against the frozen matrix in prepared_output_readback.
import Base
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_f32.bend as F
import ../../lib/storage_buffer.bend as Storage
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/nat_order.bend as Order
import ../specification.bend as ConvSpec
import ../shape.bend as ConvShape
import ../../lib/machine_limits.bend as Machine
import ../inference.bend as NativeConv
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./supported_padded_coordinates.bend as Padded
import ./tile_layout.bend as TileLayout
import ./gather_storage_model.bend as GatherModel
import ./gather_tree_readback.bend as GatherTreeReadback
import ./prepared_output_readback.bend as PreparedOutputReadback
import ./microkernel_storage_execution.bend as Execution
import ./window_partition.bend as Windows
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./gemm_dot_matches_matrix.bend as GemmDot
import ./gather_blocks.bend as Blocks
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ./packed_output_allocation.bend as PackedOutputAllocation
import ./concrete_support_guard.bend as Support
import ../../lib/proofs/write_program.bend as Writes

def work(+problem: ConvSpec.Problem) -> Pipeline.Work:
  NativeConv.work(ConvShape.from_problem(problem))

def finish_leaf_same(+problem: ConvSpec.Problem,+identity: Bool,+leaf: Tree.Tree<F32>,+offset: U32,+span: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(PreparedOutputReadback.count(problem))) == True{} : Bool}) ->
  {Execution.finish_leaf(identity,leaf,MicroLeaf.rows(problem),offset,span,Padded.spatial(problem)) ==
    GatherModel.finish_leaf(identity,leaf,offset,span,MicroLeaf.rows(problem),Padded.spatial(problem)) : Tree.Tree<F32>}:
  match identity:
    case True{}: {==}
    case False{}:
      %Equal.sym(Nat,Tensors.depth_for(PreparedOutputReadback.count(problem)),Storage.allocation_depth(PreparedOutputReadback.count(problem)),Depth.positive_depth(PreparedOutputReadback.count(problem),positive)) :
        {Writes.pure(F32,Blocks.block_writes(MicroLeaf.rows(problem),offset,Padded.spatial(problem),span,leaf),Tree.fresh(F32,_,0.0)) ==
          GatherModel.finish_leaf(False{},leaf,offset,span,MicroLeaf.rows(problem),Padded.spatial(problem)) : Tree.Tree<F32>}
      {==}

def finish_same(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+layout: TileLayout.Layout(),
  +positive: {Nat.is_lt(0n,U32.to_nat(PreparedOutputReadback.count(problem))) == True{} : Bool}) ->
  {Execution.finish(MicroLeaf.layout_result(problem,storage_contract,layout),MicroLeaf.rows(problem),Padded.spatial(problem)) ==
    GatherModel.finish_result(MicroLeaf.layout_result(problem,storage_contract,layout),MicroLeaf.rows(problem),Padded.spatial(problem)) : Tree.Tree<F32>}:
  match layout:
    case Partition.Leaf{unit,+offset,+span,details}:
      finish_leaf_same(problem,U32.is_zero(offset) && U32.is_eq(span,Padded.spatial(problem)),MicroLeaf.leaf_output(problem,storage_contract,offset,span),offset,span,positive)
    case Partition.Fork{+left,+right}:
      +tree = {Partition.Fork{MicroLeaf.layout_result(problem,storage_contract,left),MicroLeaf.layout_result(problem,storage_contract,right)} : GatherModel.ConvResult()}
      %Equal.sym(Nat,Tensors.depth_for(PreparedOutputReadback.count(problem)),Storage.allocation_depth(PreparedOutputReadback.count(problem)),Depth.positive_depth(PreparedOutputReadback.count(problem),positive)) :
        {GatherModel.gather(tree,Tree.fresh(F32,_,0.0),Padded.spatial(problem)) == GatherModel.finish_result(tree,MicroLeaf.rows(problem),Padded.spatial(problem)) : Tree.Tree<F32>}
      {==}

def spatial_positive(+problem: ConvSpec.Problem) -> {Nat.is_lt(0n,GatherCopyCoordinates.spatial(problem)) == True{} : Bool}:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{+input_channels,+output_channels,+height,+width,+kernel,+stride,+padding} = shape
  Support.positive_mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding),{==},{==})

def count_positive(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(PreparedOutputReadback.count(problem))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(PreparedOutputReadback.count(problem)),Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),PreparedOutputReadback.count_exact(limits,problem,valid,supported)) :
    {Nat.is_lt(0n,_) == True{} : Bool}
  Support.positive_mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem),PackedOutputAllocation.channel_positive(problem,Guards.valid(problem,valid)),spatial_positive(problem))

# The actual pipelined execution is exactly the model read back in
# prepared_output_readback; only the depth routine, windows and allocation differ.
def run_output(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Execution.run(depth,work(problem),Inputs.input_tree(problem,storage_contract),Inputs.weight_tree(problem,storage_contract),
      Inputs.bias_tree(problem,storage_contract)) == PreparedOutputReadback.output(problem,storage_contract,depth) : Tree.Tree<F32>}:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+input_channels,+output_channels,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{input_channels,output_channels,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  +positive = count_positive(limits,problem,valid,supported)
  +blocks = (Padded.padded(problem) / 8 : U32)
  +split = PreparedOutputReadback.selected_depth(problem,depth)
  +layout = TileLayout.divide(split,0,Padded.padded(problem))
  +window = Windows.root_window(problem,storage_contract)
  +reordered = GemmDot.reordered(problem,storage_contract)
  +bias = Inputs.bias_tree(problem,storage_contract)
  %Equal.sym(Nat,Tensors.significant_bits(32n,blocks,0n,U32.is_zero(blocks)),Storage.bit_length(32n,blocks),Depth.countdown(32n,blocks,0n)) :
    {Execution.finish(Execution.partition(U32.to_nat(U32.min(depth,(U32.from_nat(_) - 1 : U32))),work(problem),0,Padded.padded(problem),window,reordered,bias),
      MicroLeaf.rows(problem),Padded.spatial(problem)) == PreparedOutputReadback.output(problem,storage_contract,depth) : Tree.Tree<F32>}
  %Equal.sym(GatherModel.ConvResult(),Execution.partition(split,work(problem),0,Padded.padded(problem),window,reordered,bias),
    MicroLeaf.layout_result(problem,storage_contract,layout),Windows.partition_root(limits,problem,storage_contract,valid,supported,split)) :
    {Execution.finish(_,MicroLeaf.rows(problem),Padded.spatial(problem)) == PreparedOutputReadback.output(problem,storage_contract,depth) : Tree.Tree<F32>}
  %Equal.sym(Tree.Tree<F32>,Execution.finish(MicroLeaf.layout_result(problem,storage_contract,layout),MicroLeaf.rows(problem),Padded.spatial(problem)),
    GatherModel.finish_result(MicroLeaf.layout_result(problem,storage_contract,layout),MicroLeaf.rows(problem),Padded.spatial(problem)),
    finish_same(problem,storage_contract,layout,positive)) :
    {_ == PreparedOutputReadback.output(problem,storage_contract,depth) : Tree.Tree<F32>}
  %Equal.sym(TileLayout.Layout(),TileLayout.tiles(PreparedOutputReadback.tiles(problem,storage_contract,depth)),layout,PreparedOutputReadback.layout_exact(problem,storage_contract,depth)) :
    {GatherModel.finish_result(MicroLeaf.layout_result(problem,storage_contract,layout),MicroLeaf.rows(problem),Padded.spatial(problem)) ==
      GatherModel.finish_result(MicroLeaf.layout_result(problem,storage_contract,_),MicroLeaf.rows(problem),Padded.spatial(problem)) : Tree.Tree<F32>}
  {==}

# The Array executor of the root: storage refinement, then the model equality.
def native_run_matches(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {NativeConv.run(ConvShape.from_problem(problem),depth,Inputs.input_owner(problem,storage_contract),Inputs.weight_owner(problem,storage_contract),
      Inputs.bias_owner(problem,storage_contract)) == Tree.pack(F32,PreparedOutputReadback.output(problem,storage_contract,depth)) : Array<F32>}:
  +input = Inputs.input_tree(problem,storage_contract)
  +weights = Inputs.weight_tree(problem,storage_contract)
  +bias = Inputs.bias_tree(problem,storage_contract)
  Equal.trans(Array<F32>,
    NativeConv.run(ConvShape.from_problem(problem),depth,Inputs.input_owner(problem,storage_contract),Inputs.weight_owner(problem,storage_contract),
      Inputs.bias_owner(problem,storage_contract)),
    Tree.pack(F32,Execution.run(depth,work(problem),input,weights,bias)),
    Tree.pack(F32,PreparedOutputReadback.output(problem,storage_contract,depth)),
    Execution.run_matches(depth,work(problem),input,weights,bias),
    Equal.cong(Tree.Tree<F32>,Array<F32>,tree => Tree.pack(F32,tree),Execution.run(depth,work(problem),input,weights,bias),
      PreparedOutputReadback.output(problem,storage_contract,depth),run_output(limits,problem,storage_contract,depth,valid,supported)))
