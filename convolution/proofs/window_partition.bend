# Every window a partition holds covers its span and holds its band, so each
# leaf stores the whole-input leaf output. Aligned rows copy bands; other rows
# keep the caller input, whose leaves pack it directly.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/u32_shift.bend as U32Shift
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/traversal_partition.bend as Partition
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./input_storage.bend as Inputs
import ./tile_partition_bounds.bend as TilePartitionBounds
import ./tile_layout.bend as TileLayout
import ./gather_storage_model.bend as GatherModel
import ./gemm_dot_matches_matrix.bend as GemmDot
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./microkernel_storage_execution.bend as Execution
import ./window_band_values.bend as WindowBandValues
import ./window_problem.bend as WindowProblem
import ./window_positions.bend as Positions
import ./window_start.bend as Start
import ./window_covers.bend as Covers
import ./window_contents.bend as Contents
import ./window_leaf_readback.bend as LeafReadback
import ../scalar_kernel.bend as ScalarConv
import ../../lib/proofs/binary_radix.bend as Radix
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ../../lib/proofs/storage_refinement.bend as StorageLaws

# The split passes the input channel count as reduction length over kernel area.
def channels_word(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem))) : U32)) ==
    Guards.input_channel_count(problem) : Nat}:
  +facts = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +channels = Guards.input_channel_count(problem)
  +kernel = Guards.kernel(problem)
  +kernel_word = U32.from_nat(kernel)
  +area = Nat.mul(kernel,kernel)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +kernel_positive = Contents.valid_condition(problem,valid,6n)
  +area_positive = Support.positive_mul(kernel,kernel,kernel_positive,kernel_positive)
  +area_le = Order.transitive(area,Nat.mul(channels,area),cap,
    %RowMajor.multiply_commutative(area,channels) : {Nat.is_le(area,_) == True{} : Bool}
    RowMajor.product_covers(area,channels,Contents.valid_condition(problem,valid,10n)),
    Coordinates.inner_bound(limits,problem,facts,support))
  +kernel_exact = SupportedCoordinateBounds.kernel_exact(limits,problem,support)
  +area_word = (kernel_word * kernel_word : U32)
  +area_exact = {%kernel_exact : {U32.to_nat(area_word) == Nat.mul(_,_) : Nat}
    Arithmetic.multiply(kernel_word,kernel_word,capacity,Limits.capacity_fits(limits),
      %Equal.sym(Nat,U32.to_nat(kernel_word),kernel,kernel_exact) : {Nat.is_le(Nat.mul(_,_),cap) == True{} : Bool}
      area_le) : {U32.to_nat(area_word) == area : Nat}}
  +divided = U32Division.divide(Coordinates.native_inner(problem),area_word,Machine.half_word(limits),Limits.half_word_fits(limits),
    %Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    %Limits.capacity_doubles_to_half(limits) : {Nat.is_le(Nat.double(area),U32.to_nat(_)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.shl(capacity)),Nat.double(cap),U32Shift.double(capacity,Limits.capacity_below_half(limits))) :
      {Nat.is_le(Nat.double(area),_) == True{} : Bool}
    Order.double_le(area,cap,area_le),
    %Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    area_positive)
  Equal.trans(Nat,U32.to_nat((Coordinates.native_inner(problem) / area_word : U32)),
    Nat.div(U32.to_nat(Coordinates.native_inner(problem)),U32.to_nat(area_word)),channels,divided,
    %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(problem)),Coordinates.inner(problem),Coordinates.inner_exact(limits,problem,facts,support)) :
      {Nat.div(_,U32.to_nat(area_word)) == channels : Nat}
    %Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.div(Coordinates.inner(problem),_) == channels : Nat}
    Positions.exact_quotient(channels,area,area_positive))

# A partition window: its rows cover the span and hold their input band.
type State<-problem: ConvSpec.Problem,-window: Execution.InputWindow,-offset: U32,-span: U32> is Data:
  State{tree: Tree.Tree<F32>,first: U32,rows: U32,
    same: {window == Execution.InputWindow{tree,first,rows} : Execution.InputWindow},
    cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
    contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
    room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}}

def leaf_window(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset: U32,+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+offset_alignment: TilePartitionBounds.BlockAligned<offset>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}) ->
  {Execution.leaf(LeafReadback.work(problem),offset,span,Execution.InputWindow{tree,first,rows},GemmDot.reordered(problem,storage_contract),
      Inputs.bias_tree(problem,storage_contract)) ==
    MicroLeaf.layout_result(problem,storage_contract,Partition.Leaf{Unit{},offset,span,Unit{}}) : GatherModel.ConvResult()}:
  match blocks:
    case 0n: LeafReadback.empty_leaf(limits,problem,storage_contract,offset,span,tree,first,rows,span_exact)
    case 1n+ +more:
      LeafReadback.band_leaf(limits,problem,storage_contract,valid,supported,offset,span,1n+more,span_exact,span_bound,offset_alignment,range,width_blocks,width,
        tree,first,rows,cover,contents,room,
        %Equal.sym(Nat,U32.to_nat(span),Nat.mul(1n+more,8n),span_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
        Support.positive_mul(1n+more,8n,{==},{==}))

# A partition leaf over aligned rows stores the whole-input leaf output.
def leaf_state(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset: U32,+span: U32,+span_alignment: TilePartitionBounds.BlockAligned<span>,
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+offset_alignment: TilePartitionBounds.BlockAligned<offset>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +window: Execution.InputWindow,+state: State<problem,window,offset,span>) ->
  {Execution.leaf(LeafReadback.work(problem),offset,span,window,GemmDot.reordered(problem,storage_contract),Inputs.bias_tree(problem,storage_contract)) ==
    MicroLeaf.layout_result(problem,storage_contract,Partition.Leaf{Unit{},offset,span,Unit{}}) : GatherModel.ConvResult()}:
  TilePartitionBounds.BlockAligned{+blocks,+span_exact} = span_alignment
  State{+tree,+first,+rows,+same,+cover,+contents,+room} = state
  %Equal.sym(Execution.InputWindow,window,Execution.InputWindow{tree,first,rows},same) :
    {Execution.leaf(LeafReadback.work(problem),offset,span,_,GemmDot.reordered(problem,storage_contract),Inputs.bias_tree(problem,storage_contract)) ==
      MicroLeaf.layout_result(problem,storage_contract,Partition.Leaf{Unit{},offset,span,Unit{}}) : GatherModel.ConvResult()}
  leaf_window(limits,problem,storage_contract,valid,supported,offset,span,blocks,span_exact,span_bound,offset_alignment,range,width_blocks,width,
    tree,first,rows,cover,contents,room)

def cover_fits(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Positions.Cover<problem,first,rows,offset,span>) ->
  {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool}:
  Positions.Cover{block,first_exact,+fits,rows_positive,kernel_fits,top,bottom} = cover
  fits

def cover_rows(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Positions.Cover<problem,first,rows,offset,span>) ->
  {Nat.is_lt(0n,rows) == True{} : Bool}:
  Positions.Cover{block,first_exact,fits,+rows_positive,kernel_fits,top,bottom} = cover
  rows_positive

# The two windows a split hands to the halves of a span.
type Children<-problem: ConvSpec.Problem,-pair: PartitionLaws.Split<Execution.InputWindow>,-offset: U32,-span: U32> is Data:
  Children{left: State<problem,PartitionLaws.first(~Execution.InputWindow,pair),offset,TilePartitionBounds.half(span)>,
    right: State<problem,PartitionLaws.second(~Execution.InputWindow,pair),(offset + TilePartitionBounds.half(span) : U32),(span - TilePartitionBounds.half(span) : U32)>}

def left_state(+problem: ConvSpec.Problem,+pair: PartitionLaws.Split<Execution.InputWindow>,+offset: U32,+span: U32,+children: Children<problem,pair,offset,span>) ->
  State<problem,PartitionLaws.first(~Execution.InputWindow,pair),offset,TilePartitionBounds.half(span)>:
  Children{+left,right} = children
  left

def right_state(+problem: ConvSpec.Problem,+pair: PartitionLaws.Split<Execution.InputWindow>,+offset: U32,+span: U32,+children: Children<problem,pair,offset,span>) ->
  State<problem,PartitionLaws.second(~Execution.InputWindow,pair),(offset + TilePartitionBounds.half(span) : U32),(span - TilePartitionBounds.half(span) : U32)>:
  Children{left,+right} = children
  right

# A window that covers a span covers its second half.
def kept_right(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+span: U32,+tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  State<problem,Execution.InputWindow{tree,first,rows},(offset + TilePartitionBounds.half(span) : U32),(span - TilePartitionBounds.half(span) : U32)>:
  +capacity = Machine.capacity(limits)
  +fits = Limits.capacity_fits(limits)
  +start = U32.to_nat(offset)
  +half = U32.to_nat(TilePartitionBounds.half(span))
  +end = Nat.add(start,U32.to_nat(span))
  State{tree,first,rows,{==},
    Positions.cover_within(problem,U32.to_nat(first),U32.to_nat(rows),start,U32.to_nat(span),U32.to_nat((offset + TilePartitionBounds.half(span) : U32)),
      U32.to_nat((span - TilePartitionBounds.half(span) : U32)),cover,Covers.out_width_positive(problem),
      %Equal.sym(Nat,U32.to_nat((offset + TilePartitionBounds.half(span) : U32)),Nat.add(start,half),TilePartitionBounds.offset_exact(offset,span,capacity,fits,capacity_room)) :
        {Nat.is_le(start,_) == True{} : Bool}
      Order.le_add(start,half),
      TilePartitionBounds.right_interval_bound(offset,span,capacity,end,fits,capacity_room,Order.reflexive(end))),
    contents,room}

# Without a copy both halves keep the parent window.
def kept(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+span: U32,+tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Children<problem,PartitionLaws.Split{Execution.InputWindow{tree,first,rows},Execution.InputWindow{tree,first,rows}},offset,span>:
  +start = U32.to_nat(offset)
  +end = Nat.add(start,U32.to_nat(span))
  Children{
    State{tree,first,rows,{==},
      Positions.cover_within(problem,U32.to_nat(first),U32.to_nat(rows),start,U32.to_nat(span),start,U32.to_nat(TilePartitionBounds.half(span)),cover,
        Covers.out_width_positive(problem),Order.reflexive(start),
        TilePartitionBounds.left_interval_bound(offset,span,Machine.capacity(limits),end,Limits.capacity_fits(limits),capacity_room,Order.reflexive(end))),
      contents,room},
    kept_right(limits,problem,offset,span,tree,first,rows,cover,contents,room,capacity_room)}

# A copied band holds its rows and covers the first half; the second keeps the parent.
def copied_with(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,
  +tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +channels: U32,+image_width: U32,+channels_exact: {U32.to_nat(channels) == Guards.input_channel_count(problem) : Nat},
  +width_exact: {U32.to_nat(image_width) == Guards.width(problem) : Nat},+left_first: U32,+left_rows: U32,
  +facts: Covers.CopyFacts<problem,first,rows,left_first,left_rows,offset,TilePartitionBounds.half(span)>) ->
  Children<problem,PartitionLaws.Split{Execution.InputWindow{Execution.copied_tree(channels,image_width,first,rows,left_first,left_rows,tree),left_first,left_rows},
    Execution.InputWindow{tree,first,rows}},offset,span>:
  Covers.CopyFacts{+left_cover,+after,+within} = facts
  +context = {Contents.CopyContext{limits,problem,valid,supported,tree,first,rows,left_first,left_rows,channels,image_width,channels_exact,width_exact,
    contents,room,after,within,cover_fits(problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span),cover),
    cover_rows(problem,U32.to_nat(left_first),U32.to_nat(left_rows),U32.to_nat(offset),U32.to_nat(TilePartitionBounds.half(span)),left_cover)} : Contents.CopyContext}
  Children{
    State{Execution.copied_tree(channels,image_width,first,rows,left_first,left_rows,tree),left_first,left_rows,{==},left_cover,
      Contents.copied_contents(context),Contents.copied_room(context)},
    kept_right(limits,problem,offset,span,tree,first,rows,cover,contents,room,capacity_room)}

def children_copy(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,
  +tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +channels: U32,+image_width: U32,+channels_exact: {U32.to_nat(channels) == Guards.input_channel_count(problem) : Nat},
  +width_exact: {U32.to_nat(image_width) == Guards.width(problem) : Nat},+band: KernelWindow.Band,
  +checked: {KernelWindow.covers(first,rows,U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),
    U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),offset,TilePartitionBounds.half(span),band) == True{} : Bool}) ->
  Children<problem,Execution.copied(channels,image_width,first,rows,band,tree),offset,span>:
  KernelWindow.Band{+left_first,+left_rows} = band
  copied_with(limits,problem,valid,supported,offset,span,tree,first,rows,cover,contents,room,capacity_room,channels,image_width,channels_exact,width_exact,
    left_first,left_rows,
    Covers.copy_facts(limits,problem,valid,supported,offset,TilePartitionBounds.half(span),first,rows,left_first,left_rows,width_blocks,width,
      TilePartitionBounds.left_interval_bound(offset,span,Machine.capacity(limits),Support.padded(problem),Limits.capacity_fits(limits),capacity_room,range),
      cover_fits(problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span),cover),checked))

# An eight-wide output row is a whole number of blocks of eight positions.
def blocks_per_row(problem: ConvSpec.Problem) -> Nat:
  U32.to_nat((U32.from_nat(ConvSpec.out_width(problem)) / 8 : U32))

def width_factor(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool},
  aligned: {U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)) == True{} : Bool}) ->
  {Nat.mul(blocks_per_row(problem),8n) == ConvSpec.out_width(problem) : Nat}:
  Equal.trans(Nat,Nat.mul(blocks_per_row(problem),8n),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ConvSpec.out_width(problem),
    Radix.divide_eight_exact_multiple(U32.from_nat(ConvSpec.out_width(problem)),aligned),Coordinates.width_exact(limits,problem,Guards.valid(problem,valid),Support.scalar(limits,problem,supported)))

def children_when(copy: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,
  +tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +channels: U32,+image_width: U32,+channels_exact: {U32.to_nat(channels) == Guards.input_channel_count(problem) : Nat},
  +width_exact: {U32.to_nat(image_width) == Guards.width(problem) : Nat},+band: KernelWindow.Band,
  +checked: {U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)) && KernelWindow.covers(first,rows,U32.from_nat(Guards.height(problem)),
    U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),
    offset,TilePartitionBounds.half(span),band) == copy : Bool}) ->
  Children<problem,Execution.split_when(copy,channels,image_width,first,rows,band,tree),offset,span>:
  match copy:
    case False{}: kept(limits,problem,offset,span,tree,first,rows,cover,contents,room,capacity_room)
    case True{}:
      children_copy(limits,problem,valid,supported,offset,span,tree,first,rows,cover,contents,room,capacity_room,range,blocks_per_row(problem),
        width_factor(limits,problem,valid,Support.scalar_guard(limits,problem,supported),Boolean.left(U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)),KernelWindow.covers(first,rows,U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),offset,TilePartitionBounds.half(span),band),checked)),
        channels,image_width,channels_exact,width_exact,band,
        Boolean.right(U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)),KernelWindow.covers(first,rows,U32.from_nat(Guards.height(problem)),
          U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),
          offset,TilePartitionBounds.half(span),band),checked))

# Both windows of a split cover their halves and hold their bands.
def children(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +window: Execution.InputWindow,+state: State<problem,window,offset,span>) ->
  Children<problem,Execution.split(LeafReadback.work(problem),offset,TilePartitionBounds.half(span),window),offset,span>:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  State{+tree,+first,+rows,+same,+cover,+contents,+room} = state
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +half = TilePartitionBounds.half(span)
  +out_width = ConvSpec.out_width(problem)
  +band = KernelWindow.band(first,rows,U32.from_nat(height),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),U32.from_nat(out_width),offset,half)
  %Equal.sym(Execution.InputWindow,window,Execution.InputWindow{tree,first,rows},same) :
    Children<problem,Execution.split(LeafReadback.work(problem),offset,TilePartitionBounds.half(span),_),offset,span>
  children_when(U32.is_zero((U32.from_nat(out_width) % 8 : U32)) && KernelWindow.covers(first,rows,U32.from_nat(height),U32.from_nat(kernel),U32.from_nat(stride),
      U32.from_nat(padding),U32.from_nat(out_width),offset,half,band),
    limits,problem,valid,supported,offset,span,tree,first,rows,cover,contents,room,capacity_room,range,
    (Coordinates.native_inner(problem) / (U32.from_nat(kernel) * U32.from_nat(kernel)) : U32),U32.from_nat(image_width),
    channels_word(limits,problem,valid,supported),SupportedCoordinateBounds.width_exact(limits,problem,support),band,{==})

# Over aligned rows every node's window covers its span and holds its band.
def partition_aligned(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  depth: Nat,+offset: U32,+span: U32,+offset_alignment: TilePartitionBounds.BlockAligned<offset>,+span_alignment: TilePartitionBounds.BlockAligned<span>,
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +window: Execution.InputWindow,+state: State<problem,window,offset,span>) ->
  {Execution.partition(depth,LeafReadback.work(problem),offset,span,window,GemmDot.reordered(problem,storage_contract),Inputs.bias_tree(problem,storage_contract)) ==
    MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(depth,offset,span)) : GatherModel.ConvResult()}:
  match depth:
    case 0n:
      leaf_state(limits,problem,storage_contract,valid,supported,offset,span,span_alignment,span_bound,offset_alignment,range,width_blocks,width,window,state)
    case 1n+ +rest:
      +capacity = Machine.capacity(limits)
      +fits = Limits.capacity_fits(limits)
      +padded = Support.padded(problem)
      +work = LeafReadback.work(problem)
      +reordered = GemmDot.reordered(problem,storage_contract)
      +bias = Inputs.bias_tree(problem,storage_contract)
      +half = TilePartitionBounds.half(span)
      +pair = Execution.split(work,offset,half,window)
      +kids = children(limits,problem,valid,supported,offset,span,capacity_room,range,window,state)
      +bounded = Order.transitive(U32.to_nat(span),padded,U32.to_nat(capacity),span_bound,
        Padded.padded_bound(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported)))
      +left_bound = Order.transitive(U32.to_nat(half),U32.to_nat(span),padded,TilePartitionBounds.half_le(span,capacity,fits,bounded),span_bound)
      +right_bound = Order.transitive(U32.to_nat((span - half : U32)),U32.to_nat(span),padded,TilePartitionBounds.remainder_le(span,capacity,fits,bounded),span_bound)
      +left_window = PartitionLaws.first(~Execution.InputWindow,pair)
      +right_window = PartitionLaws.second(~Execution.InputWindow,pair)
      %Equal.sym(GatherModel.ConvResult(),Execution.partition(rest,work,offset,half,left_window,reordered,bias),
        MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(rest,offset,half)),
        partition_aligned(limits,problem,storage_contract,valid,supported,width_blocks,width,rest,offset,half,offset_alignment,
          TilePartitionBounds.half_alignment(span,capacity,fits,bounded),TilePartitionBounds.left_room(offset,span,capacity,fits,capacity_room),left_bound,
          TilePartitionBounds.left_interval_bound(offset,span,capacity,padded,fits,capacity_room,range),left_window,left_state(problem,pair,offset,span,kids))) :
        {Partition.Fork{_,Execution.partition(rest,work,(offset + half : U32),(span - half : U32),right_window,reordered,bias)} ==
          MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(1n+rest,offset,span)) : GatherModel.ConvResult()}
      %Equal.sym(GatherModel.ConvResult(),Execution.partition(rest,work,(offset + half : U32),(span - half : U32),right_window,reordered,bias),
        MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(rest,(offset + half : U32),(span - half : U32))),
        partition_aligned(limits,problem,storage_contract,valid,supported,width_blocks,width,rest,(offset + half : U32),(span - half : U32),
          TilePartitionBounds.right_alignment(offset,span,capacity,fits,capacity_room,offset_alignment),TilePartitionBounds.remainder_alignment(span,capacity,fits,bounded,span_alignment),
          TilePartitionBounds.right_room(offset,span,capacity,fits,capacity_room),right_bound,
          TilePartitionBounds.right_interval_bound(offset,span,capacity,padded,fits,capacity_room,range),right_window,right_state(problem,pair,offset,span,kids))) :
        {Partition.Fork{MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(rest,offset,half)),_} ==
          MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(1n+rest,offset,span)) : GatherModel.ConvResult()}
      {==}

# The caller input as the first window of the partition.
def root_window(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>) -> Execution.InputWindow:
  Execution.InputWindow{Inputs.input_tree(problem,storage_contract),0,U32.from_nat(Guards.height(problem))}

# Unaligned rows never copy: both halves keep the caller input.
def root_split(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+offset: U32,+half: U32,
  +unaligned: {U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)) == False{} : Bool}) ->
  {Execution.split(LeafReadback.work(problem),offset,half,root_window(problem,storage_contract)) ==
    PartitionLaws.Split{root_window(problem,storage_contract),root_window(problem,storage_contract)} : PartitionLaws.Split<Execution.InputWindow>}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +tree = Inputs.input_tree(problem,storage_contract)
  +rows = U32.from_nat(height)
  +out_width = U32.from_nat(ConvSpec.out_width(problem))
  +band = KernelWindow.band(0,rows,rows,U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),out_width,offset,half)
  %Equal.sym(Bool,U32.is_zero((out_width % 8 : U32)),False{},unaligned) :
    {Execution.split_when(_ && KernelWindow.covers(0,rows,rows,U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),out_width,offset,half,band),
      (Coordinates.native_inner(problem) / (U32.from_nat(kernel) * U32.from_nat(kernel)) : U32),U32.from_nat(image_width),0,rows,band,tree) ==
      PartitionLaws.Split{Execution.InputWindow{tree,0,rows},Execution.InputWindow{tree,0,rows}} : PartitionLaws.Split<Execution.InputWindow>}
  {==}

def partition_unaligned(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,depth: Nat,+offset: U32,+span: U32,
  +unaligned: {U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)) == False{} : Bool}) ->
  {Execution.partition(depth,LeafReadback.work(problem),offset,span,root_window(problem,storage_contract),GemmDot.reordered(problem,storage_contract),
      Inputs.bias_tree(problem,storage_contract)) ==
    MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(depth,offset,span)) : GatherModel.ConvResult()}:
  match depth:
    case 0n: LeafReadback.root_leaf(problem,storage_contract,offset,span)
    case 1n+ +rest:
      +work = LeafReadback.work(problem)
      +reordered = GemmDot.reordered(problem,storage_contract)
      +bias = Inputs.bias_tree(problem,storage_contract)
      +half = TilePartitionBounds.half(span)
      +root = root_window(problem,storage_contract)
      %Equal.sym(PartitionLaws.Split<Execution.InputWindow>,Execution.split(work,offset,half,root),PartitionLaws.Split{root,root},
        root_split(problem,storage_contract,offset,half,unaligned)) :
        {Partition.Fork{Execution.partition(rest,work,offset,half,PartitionLaws.first(~Execution.InputWindow,_),reordered,bias),
          Execution.partition(rest,work,(offset + half : U32),(span - half : U32),PartitionLaws.second(~Execution.InputWindow,_),reordered,bias)} ==
          MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(1n+rest,offset,span)) : GatherModel.ConvResult()}
      %Equal.sym(GatherModel.ConvResult(),Execution.partition(rest,work,offset,half,root,reordered,bias),
        MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(rest,offset,half)),partition_unaligned(problem,storage_contract,rest,offset,half,unaligned)) :
        {Partition.Fork{_,Execution.partition(rest,work,(offset + half : U32),(span - half : U32),root,reordered,bias)} ==
          MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(1n+rest,offset,span)) : GatherModel.ConvResult()}
      %Equal.sym(GatherModel.ConvResult(),Execution.partition(rest,work,(offset + half : U32),(span - half : U32),root,reordered,bias),
        MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(rest,(offset + half : U32),(span - half : U32))),
        partition_unaligned(problem,storage_contract,rest,(offset + half : U32),(span - half : U32),unaligned)) :
        {Partition.Fork{MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(rest,offset,half)),_} ==
          MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(1n+rest,offset,span)) : GatherModel.ConvResult()}
      {==}

# The caller input covers the whole padded span: its rows are the image.
def root_cover(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  Positions.Cover<problem,U32.to_nat(0),U32.to_nat(U32.from_nat(Guards.height(problem))),U32.to_nat(0),U32.to_nat(Padded.padded(problem))>:
  +height = Guards.height(problem)
  +height_exact = SupportedCoordinateBounds.height_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(height)),height,height_exact) : Positions.Cover<problem,0n,_,0n,U32.to_nat(Padded.padded(problem))>
  Positions.Cover{0n,{==},Order.reflexive(height),Contents.valid_condition(problem,valid,8n),Contents.valid_condition(problem,valid,4n),Inl{{==}},Inl{{==}}}

def root_room(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+valid: {ConvSpec.valid(problem) == True{} : Bool}) ->
  {Nat.is_le(WindowProblem.count(problem,Guards.height(problem)),U32.to_nat(Tree.capacity(F32,Inputs.input_tree(problem,storage_contract)))) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence} = storage_contract
  StorageLaws.SequenceEvidence{+storage,+stored,+room} = input_evidence
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +input_length = Boolean.every_condition_holds(Guards.valid_conditions(problem),2n,valid)
  %Guards.reflect(ConvSpec.length(input),Nat.mul(ci,Nat.mul(height,width)),input_length) :
    {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,storage))) == True{} : Bool}
  room

def root_state(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  State<problem,root_window(problem,storage_contract),0,Padded.padded(problem)>:
  +height = Guards.height(problem)
  +rows = U32.from_nat(height)
  +height_exact = SupportedCoordinateBounds.height_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))
  +tree = Inputs.input_tree(problem,storage_contract)
  State{tree,0,rows,{==},root_cover(limits,problem,valid,supported),
    %Equal.sym(Nat,U32.to_nat(rows),height,height_exact) : WindowBandValues.Contents(WindowProblem.count(problem,_),WindowProblem.band(problem,0n,_),tree)
    Contents.root_contents(problem,storage_contract,valid),
    %Equal.sym(Nat,U32.to_nat(rows),height,height_exact) : {Nat.is_le(WindowProblem.count(problem,_),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}
    root_room(problem,storage_contract,valid)}

def root_when(aligned: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+depth: Nat,
  +checked: {U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)) == aligned : Bool}) ->
  {Execution.partition(depth,LeafReadback.work(problem),0,Padded.padded(problem),root_window(problem,storage_contract),GemmDot.reordered(problem,storage_contract),
      Inputs.bias_tree(problem,storage_contract)) ==
    MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(depth,0,Padded.padded(problem))) : GatherModel.ConvResult()}:
  match aligned:
    case True{}:
      +facts = Guards.valid(problem,valid)
      +fast = Support.fast(limits,problem,supported)
      +span = Padded.padded(problem)
      +span_exact = Padded.padded_exact(limits,problem,facts,fast)
      +span_bound = {%Equal.sym(Nat,U32.to_nat(span),Support.padded(problem),span_exact) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
        Order.reflexive(Support.padded(problem)) : {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}}
      partition_aligned(limits,problem,storage_contract,valid,supported,blocks_per_row(problem),
        width_factor(limits,problem,valid,Support.scalar_guard(limits,problem,supported),checked),depth,0,span,TilePartitionBounds.BlockAligned{0n,{==}},
        TilePartitionBounds.BlockAligned{Nat.div(Nat.add(GatherCopyCoordinates.spatial(problem),7n),8n),span_exact},Padded.root_room(limits,problem,facts,fast),span_bound,span_bound,
        root_window(problem,storage_contract),root_state(limits,problem,storage_contract,valid,supported))
    case False{}: partition_unaligned(problem,storage_contract,depth,0,Padded.padded(problem),checked)

# The actual partition from the caller input equals the model leaves.
def partition_root(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+depth: Nat) ->
  {Execution.partition(depth,LeafReadback.work(problem),0,Padded.padded(problem),root_window(problem,storage_contract),GemmDot.reordered(problem,storage_contract),
      Inputs.bias_tree(problem,storage_contract)) ==
    MicroLeaf.layout_result(problem,storage_contract,TileLayout.divide(depth,0,Padded.padded(problem))) : GatherModel.ConvResult()}:
  root_when(U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)),limits,problem,storage_contract,valid,supported,depth,{==})
