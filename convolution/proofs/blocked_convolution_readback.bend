# Dense block states keep their sources and refine the ordered convolution.
import Base
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/proofs/natural_addition.bend as Addition
import ./blocked_pipeline_bounds.bend as BlockBounds
import ./tile_partition_bounds.bend as Partition
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_extrema.bend as Extrema
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/boolean_logic.bend as Boolean
import ./gemm_output_relocation.bend as Relocation
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/kernel_gemm_writes.bend as GemmWrites
import ../../lib/kernel_gemm.bend as Gemm
import ./blocked_pipeline_model.bend as Model
import ./input_storage.bend as Inputs
import ./prepared_packing_samples.bend as Prepared
import ./packing_storage_refinement.bend as PackingStorage
import ./packing_tabulation.bend as Tabulation
import ./packing_contents.bend as PackingContents
import ./packing_tile_bounds.bend as PackingBounds
import ./convolution_leaf_contract.bend as LeafContract
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./microkernel_inference_refinement.bend as Inference
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./gemm_dot_matches_matrix.bend as GemmDot
import ./reduction_coordinate_bounds.bend as Coordinates
import ./supported_padded_coordinates.bend as Padded
import ./output_values.bend as Values

def with_positions(work: Pipeline.Work,positions: U32) -> Pipeline.Work:
  Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,spatial,inner} = work
  Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner}

def work(+problem: ConvSpec.Problem,positions: U32) -> Pipeline.Work:
  with_positions(Inference.work(problem),positions)

def local_position(problem: ConvSpec.Problem,origin: U32,row: Nat,column: Nat) -> Nat:
  Nat.sub(Coordinates.position(problem,row,column),U32.to_nat(origin))

def state(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,workspace: Tree.Tree<Tile.Tile8>,output: Tree.Tree<F32>) -> Model.State:
  Model.State{Inputs.input_tree(problem,storage),workspace,GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage),output}

def tiles(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,workspace: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Prepared.packed_output(Prepared.workspace_pack(problem,storage,start,span,workspace))

def output(+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,workspace: Tree.Tree<Tile.Tile8>,previous: Tree.Tree<F32>) -> Tree.Tree<F32>:
  GemmLaws.run_at(MicroLeaf.rows(problem),tiles(problem,storage,(origin + start : U32),span,workspace),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage),previous,span,Coordinates.native_inner(problem),start,positions)

def step_state(+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<F32>) ->
  {Model.step(work(problem,positions),(origin + start : U32),start,span,state(problem,storage,workspace,previous)) ==
    state(problem,storage,tiles(problem,storage,(origin + start : U32),span,workspace),output(problem,origin,positions,storage,start,span,workspace,previous)) : Model.State}:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+input_channels,+output_channels,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{input_channels,output_channels,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  %Equal.sym(PackingStorage.Packed,Prepared.workspace_pack(problem,storage,(origin + start : U32),span,workspace),PackingStorage.Packed{Inputs.input_tree(problem,storage),tiles(problem,storage,(origin + start : U32),span,workspace)},Prepared.workspace_pair(problem,storage,(origin + start : U32),span,workspace)) :
    {Model.packed(MicroLeaf.rows(problem),positions,Coordinates.native_inner(problem),start,span,GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage),previous,_) ==
      state(problem,storage,tiles(problem,storage,(origin + start : U32),span,workspace),output(problem,origin,positions,storage,start,span,workspace,previous)) : Model.State}
  Model.packed_result(MicroLeaf.rows(problem),positions,Coordinates.native_inner(problem),start,span,GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage),previous,Inputs.input_tree(problem,storage),tiles(problem,storage,(origin + start : U32),span,workspace))

def tiles_balanced(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+workspace: Tree.Tree<Tile.Tile8>,+depth: Nat,
  +valid: Guards.Valid<problem>,+supported: Support.FastSupported<limits,problem>,
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(start),Nat.mul(PackingBounds.groups(span),8n)),Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool},
  balanced: Shape.Balanced(Tile.Tile8,depth,workspace)) -> Shape.Balanced(Tile.Tile8,depth,tiles(problem,storage,start,span,workspace)):
  %Equal.sym(Tree.Tree<Tile.Tile8>,tiles(problem,storage,start,span,workspace),Prepared.workspace_tabulated(problem,storage,start,span,workspace),
    Prepared.workspace_contents(limits,problem,storage,start,span,workspace,valid,supported,span_bound,range)) : Shape.Balanced(Tile.Tile8,depth,_)
  Tabulation.fill_balanced(U32.to_nat(PackingBounds.count(span,problem)),0,start,Inputs.input_tree(problem,storage),workspace,
    U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.width(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),Coordinates.native_inner(problem),depth,balanced)

def output_balanced(+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<F32>,+depth: Nat,
  balanced: Shape.Balanced(F32,depth,previous)) -> Shape.Balanced(F32,depth,output(problem,origin,positions,storage,start,span,workspace,previous)):
  +input = tiles(problem,storage,(origin + start : U32),span,workspace)
  +weights = GemmDot.reordered(problem,storage)
  +bias = Inputs.bias_tree(problem,storage)
  +writes = GemmWrites.all_writes(GemmWrites.job_count(MicroLeaf.rows(problem),span),0,Gemm.Layout{MicroLeaf.rows(problem),span,Coordinates.native_inner(problem),start,positions},input,weights,bias)
  %Equal.sym(Tree.Tree<F32>,output(problem,origin,positions,storage,start,span,workspace,previous),Writes.pure(F32,writes,previous),
    GemmWrites.run_at_writes(MicroLeaf.rows(problem),input,weights,bias,previous,span,Coordinates.native_inner(problem),start,positions)) : Shape.Balanced(F32,depth,_)
  Writes.writes_preserve_balance(F32,writes,previous,depth,balanced)

# Only the prefix packed for this block is sampled by the unchanged dot chain.
def block_readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+blocks: Nat,
  +workspace: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+previous: Tree.Tree<F32>,+output_depth: Nat,
  +channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tile_balanced: Shape.Balanced(Tile.Tile8,tile_depth,workspace),+tile_depth_ok: {Nat.is_le(tile_depth,30n) == True{} : Bool},
  +tile_room: {Nat.is_le(U32.to_nat(PackingBounds.count(span,problem)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool},
  +output_balanced: Shape.Balanced(F32,output_depth,previous),+output_depth_ok: {Nat.is_le(output_depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(MicroLeaf.rows(problem)),U32.to_nat(positions)),U32.to_nat(Tree.capacity(F32,previous))) == True{} : Bool},
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +sample_range: {Nat.is_le(Nat.add(U32.to_nat((origin + start : U32)),Nat.mul(PackingBounds.groups(span),8n)),Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(positions),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat((origin + start : U32)),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_in_width: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +in_output: {Nat.is_lt(Nat.add(U32.to_nat(start),column),U32.to_nat(positions)) == True{} : Bool}) ->
  {Tree.read(F32,output(problem,origin,positions,storage,start,span,workspace,previous),
    U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),Nat.add(U32.to_nat(start),column)))) == Values.matrix_value(problem,channel,output_row,output_column) : F32}:
  MicroLeaf.relocated_readback(~LeafContract.workspace_evidence,~LeafContract.workspace_samples,limits,problem,storage,valid,supported,tiles(problem,storage,(origin + start : U32),span,workspace),(origin + start : U32),span,blocks,channel,column,output_row,output_column,
    LeafContract.WorkspaceEvidence{storage,workspace,tile_depth,tile_balanced,tile_depth_ok,tile_room,span_bound,sample_range,{==}},
    span_exact,span_bound,channel_inside,column_inside,position,row_inside,column_in_width,start,positions,previous,output_depth,output_balanced,output_depth_ok,output_room,end_room,columns_room,in_output)

# The frame property requires no samples from the block being written.
def block_preserves_before(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+blocks: Nat,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<F32>,+output_depth: Nat,+channel: Nat,+point: Nat,
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +output_balanced: Shape.Balanced(F32,output_depth,previous),+output_depth_ok: {Nat.is_le(output_depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(MicroLeaf.rows(problem)),U32.to_nat(positions)),U32.to_nat(Tree.capacity(F32,previous))) == True{} : Bool},
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(positions),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +point_inside: {Nat.is_lt(point,U32.to_nat(positions)) == True{} : Bool},+before: {Nat.is_lt(point,U32.to_nat(start)) == True{} : Bool}) ->
  {Tree.read(F32,output(problem,origin,positions,storage,start,span,workspace,previous),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),point))) ==
    Tree.read(F32,previous,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),point))) : F32}:
  +query = MicroLeaf.leaf_query(limits,problem,storage,tiles(problem,storage,(origin + start : U32),span,workspace),span,blocks,channel,0n,span_exact,Support.fast(limits,problem,supported),span_bound,channel_inside,span_positive)
  Relocation.run_preserves_before(query,start,positions,previous,output_depth,point,output_balanced,output_depth_ok,output_room,end_room,columns_room,point_inside,before)

# The loop carries fixed geometry separately from the two changing trees.
type Geometry<-limits: Machine.Limits,-problem: ConvSpec.Problem,-origin: U32,-positions: U32,-padded: U32,-span: U32,-cap: U32> is Data:
  Geometry{origin_end: {Nat.is_le(Nat.add(U32.to_nat(origin),U32.to_nat(padded)),Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool},
    pack_room: {Nat.is_le(Nat.add(U32.to_nat(origin),U32.to_nat(padded)),U32.to_nat(cap)) == True{} : Bool},valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
    cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    padded_bound: {Nat.is_le(U32.to_nat(padded),Support.padded(problem)) == True{} : Bool},
    padded_end: {Nat.is_le(U32.to_nat(padded),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},
    columns_room: {Nat.is_le(U32.to_nat(positions),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
    sum_room: {Nat.is_le(Nat.add(U32.to_nat(padded),U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool},
    span_positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
    padded_aligned: Partition.BlockAligned<padded>,span_aligned: Partition.BlockAligned<span>}

def current(+start: U32,+padded: U32,+span: U32) -> U32:
  U32.min(span,(padded - start : U32))

type StepGeometry<-problem: ConvSpec.Problem,-origin: U32,-positions: U32,-start: U32,-padded: U32,-span: U32> is Data:
  StepGeometry{pack_exact: {U32.to_nat((origin + start : U32)) == Nat.add(U32.to_nat(origin),U32.to_nat(start)) : Nat},advance: {U32.to_nat((start + span : U32)) == Nat.add(U32.to_nat(start),U32.to_nat(span)) : Nat},
    clipped: {U32.to_nat(current(start,padded,span)) == Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))) : Nat},
    positive: {Nat.is_lt(0n,U32.to_nat(current(start,padded,span))) == True{} : Bool},
    span_bound: {Nat.is_le(U32.to_nat(current(start,padded,span)),Support.padded(problem)) == True{} : Bool},
    sample_range: {Nat.is_le(Nat.add(U32.to_nat((origin + start : U32)),Nat.mul(PackingBounds.groups(current(start,padded,span)),8n)),Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool},
    end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(current(start,padded,span))),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},
    aligned: Partition.BlockAligned<current(start,padded,span)>,next_aligned: Partition.BlockAligned<(start + span : U32)>}

def step_geometry(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +geometry: Geometry<limits,problem,origin,positions,padded,span,cap>,
  +inside: {Nat.is_lt(U32.to_nat(start),U32.to_nat(padded)) == True{} : Bool},+aligned: Partition.BlockAligned<start>) -> StepGeometry<problem,origin,positions,start,padded,span>:
  Geometry{+origin_end,+pack_room,+valid,+supported,+cap_ok,+padded_bound,+padded_end,+columns_room,+sum_room,+span_positive,+padded_aligned,+span_aligned} = geometry
  +ordered = Bounds.strict_le(U32.to_nat(start),U32.to_nat(padded),inside)
  +padded_room = Order.transitive(U32.to_nat(padded),Nat.add(U32.to_nat(padded),U32.to_nat(span)),U32.to_nat(cap),Order.le_add(U32.to_nat(padded),U32.to_nat(span)),sum_room)
  +difference = Arithmetic.subtract(padded,start,cap,cap_ok,ordered,Order.transitive(Nat.sub(U32.to_nat(padded),U32.to_nat(start)),U32.to_nat(padded),U32.to_nat(cap),Order.sub_le(U32.to_nat(padded),U32.to_nat(start)),padded_room))
  +advance = Arithmetic.add(start,span,cap,cap_ok,Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(span)),Nat.add(U32.to_nat(padded),U32.to_nat(span)),U32.to_nat(cap),Order.add_le(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),ordered),sum_room))
  +clipped = BlockBounds.clipped_exact(start,padded,span,difference)
  +pack_exact = Arithmetic.add(origin,start,cap,cap_ok,
    Order.transitive(Nat.add(U32.to_nat(origin),U32.to_nat(start)),Nat.add(U32.to_nat(origin),U32.to_nat(padded)),U32.to_nat(cap),
      Order.offset_le(U32.to_nat(origin),U32.to_nat(start),U32.to_nat(padded),ordered),pack_room))
  +end_room = {
    %Equal.sym(Nat,U32.to_nat(current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_le(Nat.add(U32.to_nat(start),_),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool}
    Order.transitive(Nat.add(U32.to_nat(start),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start)))),U32.to_nat(padded),Nat.add(U32.to_nat(positions),7n),BlockBounds.clipped_end(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),ordered),padded_end) :
      {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(current(start,padded,span))),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool}}
  StepGeometry{pack_exact,advance,clipped,
    %Equal.sym(Nat,U32.to_nat(current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_lt(0n,_) == True{} : Bool}
    Extrema.minimum_strict_lower(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start)),0n,span_positive,Order.positive_difference(U32.to_nat(padded),U32.to_nat(start),inside)),
    %Equal.sym(Nat,U32.to_nat(current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    Order.transitive(Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),U32.to_nat(padded),Support.padded(problem),BlockBounds.clipped_bound(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span)),padded_bound),
    %Equal.sym(Nat,U32.to_nat((origin + start : U32)),Nat.add(U32.to_nat(origin),U32.to_nat(start)),pack_exact) : {Nat.is_le(Nat.add(_,Nat.mul(PackingBounds.groups(current(start,padded,span)),8n)),Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(origin),U32.to_nat(start)),Nat.mul(PackingBounds.groups(current(start,padded,span)),8n)),Nat.add(U32.to_nat(origin),Nat.add(U32.to_nat(start),Nat.mul(PackingBounds.groups(current(start,padded,span)),8n))),Addition.associative(U32.to_nat(origin),U32.to_nat(start),Nat.mul(PackingBounds.groups(current(start,padded,span)),8n))) : {Nat.is_le(_,Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool}
    Order.transitive(Nat.add(U32.to_nat(origin),Nat.add(U32.to_nat(start),Nat.mul(PackingBounds.groups(current(start,padded,span)),8n))),Nat.add(U32.to_nat(origin),U32.to_nat(padded)),Nat.add(PackingContents.grid(problem),7n),
      Order.offset_le(U32.to_nat(origin),Nat.add(U32.to_nat(start),Nat.mul(PackingBounds.groups(current(start,padded,span)),8n)),U32.to_nat(padded),
        Order.transitive(Nat.add(U32.to_nat(start),Nat.mul(PackingBounds.groups(current(start,padded,span)),8n)),Nat.add(U32.to_nat(start),U32.to_nat(current(start,padded,span))),U32.to_nat(padded),
          Order.offset_le(U32.to_nat(start),Nat.mul(PackingBounds.groups(current(start,padded,span)),8n),U32.to_nat(current(start,padded,span)),Partition.quotient_product(current(start,padded,span),8n,{==})),
          %Equal.sym(Nat,U32.to_nat(current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_le(Nat.add(U32.to_nat(start),_),U32.to_nat(padded)) == True{} : Bool}
          BlockBounds.clipped_end(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),ordered))),origin_end),
    end_room,BlockBounds.minimum_aligned(span,(padded - start : U32),span_aligned,BlockBounds.difference_aligned(padded,start,difference,padded_aligned,aligned)),
    BlockBounds.sum_aligned(start,span,advance,aligned,span_aligned)}

type Memory<-problem: ConvSpec.Problem,-origin: U32,-positions: U32,-span: U32,-workspace: Tree.Tree<Tile.Tile8>,-previous: Tree.Tree<F32>,-tile_depth: Nat,-output_depth: Nat> is Data:
  Memory{tile_balanced: Shape.Balanced(Tile.Tile8,tile_depth,workspace),tile_depth_ok: {Nat.is_le(tile_depth,30n) == True{} : Bool},
    tile_room: {Nat.is_le(Nat.mul(PackingBounds.groups(span),Coordinates.inner(problem)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool},
    output_balanced: Shape.Balanced(F32,output_depth,previous),output_depth_ok: {Nat.is_le(output_depth,30n) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(U32.to_nat(MicroLeaf.rows(problem)),U32.to_nat(positions)),U32.to_nat(Tree.capacity(F32,previous))) == True{} : Bool}}

def block_tile_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,+clipped: U32,+workspace: Tree.Tree<Tile.Tile8>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(clipped),Support.padded(problem)) == True{} : Bool},
  +clipped_bound: {Nat.is_le(U32.to_nat(clipped),U32.to_nat(span)) == True{} : Bool},
  +tile_room: {Nat.is_le(Nat.mul(PackingBounds.groups(span),Coordinates.inner(problem)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(PackingBounds.count(clipped,problem)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(PackingBounds.count(clipped,problem)),Nat.mul(PackingBounds.groups(clipped),Coordinates.inner(problem)),PackingBounds.count_exact(limits,problem,clipped,Guards.valid(problem,valid),Support.fast(limits,problem,supported),span_bound)) :
    {Nat.is_le(_,U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool}
  Order.transitive(Nat.mul(PackingBounds.groups(clipped),Coordinates.inner(problem)),Nat.mul(PackingBounds.groups(span),Coordinates.inner(problem)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace)),
    RowMajor.mul_le(PackingBounds.groups(clipped),PackingBounds.groups(span),Coordinates.inner(problem),DivisionOrder.quotient_le(U32.to_nat(clipped),U32.to_nat(span),8n,{==},clipped_bound)),tile_room)

def memory_advanced(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+clipped: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(clipped),Support.padded(problem)) == True{} : Bool},
  +sample_range: {Nat.is_le(Nat.add(U32.to_nat((origin + start : U32)),Nat.mul(PackingBounds.groups(clipped),8n)),Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool},
  memory: Memory<problem,origin,positions,span,workspace,previous,tile_depth,output_depth>) ->
  Memory<problem,origin,positions,span,tiles(problem,storage,(origin + start : U32),clipped,workspace),output(problem,origin,positions,storage,start,clipped,workspace,previous),tile_depth,output_depth>:
  Memory{+tile_balanced,+tile_depth_ok,+tile_room,+previous_balanced,+output_depth_ok,+output_room} = memory
  +next_tiles = tiles(problem,storage,(origin + start : U32),clipped,workspace)
  +next_output = output(problem,origin,positions,storage,start,clipped,workspace,previous)
  +next_tiles_balanced = tiles_balanced(limits,problem,storage,(origin + start : U32),clipped,workspace,tile_depth,Guards.valid(problem,valid),Support.fast(limits,problem,supported),span_bound,sample_range,tile_balanced)
  +next_output_balanced = output_balanced(problem,origin,positions,storage,start,clipped,workspace,previous,output_depth,previous_balanced)
  Memory{next_tiles_balanced,tile_depth_ok,Shape.transfer_room(Tile.Tile8,tile_depth,workspace,next_tiles,Nat.mul(PackingBounds.groups(span),Coordinates.inner(problem)),tile_balanced,next_tiles_balanced,tile_room),
    next_output_balanced,output_depth_ok,Shape.transfer_room(F32,output_depth,previous,next_output,Nat.mul(U32.to_nat(MicroLeaf.rows(problem)),U32.to_nat(positions)),previous_balanced,next_output_balanced,output_room)}

def address(+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+channel: Nat,+row: Nat,+column: Nat) -> U32:
  U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),local_position(problem,origin,row,column)))

def Earlier(+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+channel: Nat,+row: Nat,+column: Nat,+start: U32,+previous: Tree.Tree<F32>) -> Data:
  Boolean.When(Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat(start)),{Tree.read(F32,previous,address(problem,origin,positions,channel,row,column)) == Values.matrix_value(problem,channel,row,column) : F32})

type Point<-problem: ConvSpec.Problem,-origin: U32,-positions: U32,-channel: Nat,-row: Nat,-column: Nat,-padded: U32> is Data:
  Point{restored: {Nat.add(U32.to_nat(origin),local_position(problem,origin,row,column)) == Coordinates.position(problem,row,column) : Nat},channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
    row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
    spatial_inside: {Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat(positions)) == True{} : Bool},
    padded_inside: {Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat(padded)) == True{} : Bool}}

def point_written(+before: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +geometry: Geometry<limits,problem,origin,positions,padded,span,cap>,+step: StepGeometry<problem,origin,positions,start,padded,span>,
  +memory: Memory<problem,origin,positions,span,workspace,previous,tile_depth,output_depth>,+point: Point<problem,origin,positions,channel,row,column,padded>,
  +inside: {Nat.is_lt(U32.to_nat(start),U32.to_nat(padded)) == True{} : Bool},
  +before_exact: {Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat(start)) == before : Bool},
  +before_next: {Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat((start + span : U32))) == True{} : Bool},
  earlier: Boolean.When(before,{Tree.read(F32,previous,address(problem,origin,positions,channel,row,column)) == Values.matrix_value(problem,channel,row,column) : F32})) ->
  {Tree.read(F32,output(problem,origin,positions,storage,start,current(start,padded,span),workspace,previous),address(problem,origin,positions,channel,row,column)) == Values.matrix_value(problem,channel,row,column) : F32}:
  match before:
    case True{}:
      Geometry{origin_end,pack_room,valid,+supported,cap_ok,padded_bound,padded_end,+columns_room,sum_room,span_positive,padded_aligned,span_aligned} = geometry
      StepGeometry{pack_exact,advance,clipped,+positive,+span_bound,sample_range,+end_room,+aligned,next_aligned} = step
      Partition.BlockAligned{+blocks,+span_exact} = aligned
      Memory{tile_balanced,tile_depth_ok,tile_room,+output_balanced,+output_depth_ok,+output_room} = memory
      Point{+restored_point,+channel_inside,row_inside,column_inside,+spatial_inside,padded_inside} = point
      Equal.trans(F32,Tree.read(F32,output(problem,origin,positions,storage,start,current(start,padded,span),workspace,previous),address(problem,origin,positions,channel,row,column)),Tree.read(F32,previous,address(problem,origin,positions,channel,row,column)),Values.matrix_value(problem,channel,row,column),
        block_preserves_before(limits,problem,origin,positions,storage,start,current(start,padded,span),blocks,workspace,previous,output_depth,channel,local_position(problem,origin,row,column),supported,output_balanced,output_depth_ok,output_room,span_exact,span_bound,positive,end_room,columns_room,channel_inside,spatial_inside,before_exact),earlier)
    case False{}:
      Geometry{+origin_end,+pack_room,+valid,+supported,cap_ok,padded_bound,padded_end,+columns_room,sum_room,span_positive,padded_aligned,span_aligned} = geometry
      StepGeometry{+pack_exact,+advance,+clipped,positive,+span_bound,+sample_range,+end_room,+aligned,next_aligned} = step
      Partition.BlockAligned{+blocks,+span_exact} = aligned
      Memory{+tile_balanced,+tile_depth_ok,+tile_room,+output_balanced,+output_depth_ok,+output_room} = memory
      Point{+restored_point,+channel_inside,+row_inside,+column_inside,+spatial_inside,+padded_inside} = point
      +position = local_position(problem,origin,row,column)
      +lower = Order.ge_true(position,U32.to_nat(start),Order.comparison_not_less(Nat.cmp(position,U32.to_nat(start)),before_exact))
      +local = Nat.sub(position,U32.to_nat(start))
      +restored = Order.restore_shift(U32.to_nat(start),position,lower)
      %restored : {Tree.read(F32,output(problem,origin,positions,storage,start,current(start,padded,span),workspace,previous),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),_))) == Values.matrix_value(problem,channel,row,column) : F32}
      block_readback(limits,problem,origin,positions,storage,start,current(start,padded,span),blocks,workspace,tile_depth,previous,output_depth,channel,local,row,column,valid,supported,tile_balanced,tile_depth_ok,
        block_tile_room(limits,problem,span,current(start,padded,span),workspace,valid,supported,span_bound,
          %Equal.sym(Nat,U32.to_nat(current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_le(_,U32.to_nat(span)) == True{} : Bool}
          Extrema.minimum_left(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),tile_room),
        output_balanced,output_depth_ok,output_room,span_exact,span_bound,sample_range,end_room,columns_room,channel_inside,
        %Equal.sym(Nat,U32.to_nat(current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_lt(local,_) == True{} : Bool}
        BlockBounds.clipped_contains(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),position,Bounds.strict_le(U32.to_nat(start),U32.to_nat(padded),inside),lower,padded_inside,
          %advance : {Nat.is_lt(position,_) == True{} : Bool} before_next),
        %Equal.sym(Nat,U32.to_nat((origin + start : U32)),Nat.add(U32.to_nat(origin),U32.to_nat(start)),pack_exact) : {Nat.add(_,local) == Coordinates.position(problem,row,column) : Nat}
        Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(origin),U32.to_nat(start)),local),Nat.add(U32.to_nat(origin),Nat.add(U32.to_nat(start),local)),Coordinates.position(problem,row,column),
          Addition.associative(U32.to_nat(origin),U32.to_nat(start),local),
          %Equal.sym(Nat,Nat.add(U32.to_nat(start),local),position,restored) : {Nat.add(U32.to_nat(origin),_) == Coordinates.position(problem,row,column) : Nat} restored_point),row_inside,column_inside,
        %Equal.sym(Nat,Nat.add(U32.to_nat(start),local),position,restored) : {Nat.is_lt(_,U32.to_nat(positions)) == True{} : Bool} spatial_inside)

def earlier_selected(+after: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +geometry: Geometry<limits,problem,origin,positions,padded,span,cap>,+step: StepGeometry<problem,origin,positions,start,padded,span>,
  +memory: Memory<problem,origin,positions,span,workspace,previous,tile_depth,output_depth>,+point: Point<problem,origin,positions,channel,row,column,padded>,
  +inside: {Nat.is_lt(U32.to_nat(start),U32.to_nat(padded)) == True{} : Bool},
  +after_exact: {Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat((start + span : U32))) == after : Bool},
  earlier: Earlier(problem,origin,positions,channel,row,column,start,previous)) ->
  Boolean.When(after,{Tree.read(F32,output(problem,origin,positions,storage,start,current(start,padded,span),workspace,previous),address(problem,origin,positions,channel,row,column)) == Values.matrix_value(problem,channel,row,column) : F32}):
  match after:
    case False{}: Unit{}
    case True{}:
      point_written(Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat(start)),limits,problem,origin,positions,storage,start,padded,span,cap,workspace,previous,tile_depth,output_depth,channel,row,column,geometry,step,memory,point,inside,{==},after_exact,earlier)

def step_memory(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,
  geometry: Geometry<limits,problem,origin,positions,padded,span,cap>,step: StepGeometry<problem,origin,positions,start,padded,span>,memory: Memory<problem,origin,positions,span,workspace,previous,tile_depth,output_depth>) ->
  Memory<problem,origin,positions,span,tiles(problem,storage,(origin + start : U32),current(start,padded,span),workspace),output(problem,origin,positions,storage,start,current(start,padded,span),workspace,previous),tile_depth,output_depth>:
  Geometry{+origin_end,+pack_room,+valid,+supported,cap_ok,padded_bound,padded_end,columns_room,sum_room,span_positive,padded_aligned,span_aligned} = geometry
  StepGeometry{pack_exact,advance,clipped,positive,+span_bound,+sample_range,end_room,aligned,next_aligned} = step
  memory_advanced(limits,problem,origin,positions,storage,start,span,current(start,padded,span),workspace,previous,tile_depth,output_depth,valid,supported,span_bound,sample_range,memory)

def step_schedule(+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+rest: Nat,+start: U32,+padded: U32,+span: U32,step: StepGeometry<problem,origin,positions,start,padded,span>,schedule: BlockBounds.Schedule<1n+rest,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span)>) ->
  BlockBounds.Schedule<rest,U32.to_nat((start + span : U32)),U32.to_nat(padded),U32.to_nat(span)>:
  StepGeometry{+pack_exact,+advance,clipped,positive,span_bound,sample_range,end_room,aligned,next_aligned} = step
  %Equal.sym(Nat,U32.to_nat((start + span : U32)),Nat.add(U32.to_nat(start),U32.to_nat(span)),advance) : BlockBounds.Schedule<rest,_,U32.to_nat(padded),U32.to_nat(span)>
  BlockBounds.schedule_advanced(rest,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),schedule)

def step_alignment(+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+start: U32,+padded: U32,+span: U32,step: StepGeometry<problem,origin,positions,start,padded,span>) -> Partition.BlockAligned<(start + span : U32)>:
  StepGeometry{pack_exact,advance,clipped,positive,span_bound,sample_range,end_room,aligned,next_aligned} = step
  next_aligned

def result(state: Model.State) -> Tree.Tree<F32>:
  Model.State{input,tiles,weights,bias,output} = state
  output

type LoopEvidence<-problem: ConvSpec.Problem,-origin: U32,-positions: U32,-channel: Nat,-row: Nat,-column: Nat,-output_depth: Nat,-output: Tree.Tree<F32>> is Data:
  LoopEvidence{balanced: Shape.Balanced(F32,output_depth,output),value: {Tree.read(F32,output,address(problem,origin,positions,channel,row,column)) == Values.matrix_value(problem,channel,row,column) : F32}}

# Once the schedule is exhausted every real point is in the completed prefix.
def loop_readback(+remaining: Nat,+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +geometry: Geometry<limits,problem,origin,positions,padded,span,cap>,+memory: Memory<problem,origin,positions,span,workspace,previous,tile_depth,output_depth>,
  +point: Point<problem,origin,positions,channel,row,column,padded>,+schedule: BlockBounds.Schedule<remaining,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span)>,
  +aligned: Partition.BlockAligned<start>,earlier: Earlier(problem,origin,positions,channel,row,column,start,previous)) ->
  LoopEvidence<problem,origin,positions,channel,row,column,output_depth,result(Model.blocks(remaining,work(problem,positions),(origin + start : U32),start,padded,span,state(problem,storage,workspace,previous)))>:
  match remaining:
    case 0n:
      Memory{tile_balanced,tile_depth_ok,tile_room,+output_balanced,output_depth_ok,output_room} = memory
      Point{restored_point,channel_inside,row_inside,column_inside,spatial_inside,+padded_inside} = point
      LoopEvidence{output_balanced,
      %Bounds.nat_trans(local_position(problem,origin,row,column),U32.to_nat(padded),U32.to_nat(start),padded_inside,BlockBounds.schedule_finished(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),schedule)) :
        Boolean.When(_,{Tree.read(F32,previous,address(problem,origin,positions,channel,row,column)) == Values.matrix_value(problem,channel,row,column) : F32})
      earlier}
    case 1n+ +rest:
      +inside = BlockBounds.schedule_inside(rest,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),schedule)
      +step = step_geometry(limits,problem,origin,positions,start,padded,span,cap,geometry,inside,aligned)
      +next_tiles = tiles(problem,storage,(origin + start : U32),current(start,padded,span),workspace)
      +next_output = output(problem,origin,positions,storage,start,current(start,padded,span),workspace,previous)
      %Equal.sym(Model.State,Model.step(work(problem,positions),(origin + start : U32),start,current(start,padded,span),state(problem,storage,workspace,previous)),state(problem,storage,next_tiles,next_output),step_state(problem,origin,positions,storage,start,current(start,padded,span),workspace,previous)) :
        LoopEvidence<problem,origin,positions,channel,row,column,output_depth,result(Model.blocks(rest,work(problem,positions),((origin + start : U32) + span : U32),(start + span : U32),padded,span,_))>
      %Equal.sym(U32,((origin + start : U32) + span : U32),(origin + (start + span : U32) : U32),Arithmetic.add_associative(origin,start,span)) :
        LoopEvidence<problem,origin,positions,channel,row,column,output_depth,result(Model.blocks(rest,work(problem,positions),_,(start + span : U32),padded,span,state(problem,storage,next_tiles,next_output)))>
      loop_readback(rest,limits,problem,origin,positions,storage,(start + span : U32),padded,span,cap,next_tiles,next_output,tile_depth,output_depth,channel,row,column,geometry,
        step_memory(limits,problem,origin,positions,storage,start,padded,span,cap,workspace,previous,tile_depth,output_depth,geometry,step,memory),point,
        step_schedule(problem,origin,positions,rest,start,padded,span,step,schedule),step_alignment(problem,origin,positions,start,padded,span,step),
        earlier_selected(Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat((start + span : U32))),limits,problem,origin,positions,storage,start,padded,span,cap,workspace,previous,tile_depth,output_depth,channel,row,column,geometry,step,memory,point,inside,{==},earlier))
