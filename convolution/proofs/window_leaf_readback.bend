# A window leaf stores the same output tree as the leaf that packs the whole
# input. In aligned rows every lane of the leaf is a real output position, so
# both accumulators are the frozen matrix dot of the same coordinate: the
# window through its own problem, whose samples are the original samples.
import Base
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../inference.bend as NativeConv
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./tile_partition_bounds.bend as TilePartitionBounds
import ./prepared_packing_samples.bend as Prepared
import ./convolution_leaf_contract.bend as LeafContract
import ./gemm_output_readback.bend as GemmReadback
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./output_values.bend as Values
import ./gemm_output_congruence.bend as Congruence
import ./window_band_values.bend as WindowBandValues
import ./window_samples.bend as Samples
import ./window_problem.bend as WindowProblem
import ./window_positions.bend as Positions
import ./window_start.bend as Start
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../shape.bend as ConvShape
import ./packing_storage_refinement.bend as PackingStorage
import ./gather_storage_model.bend as GatherModel
import ./gemm_dot_matches_matrix.bend as GemmDot
import ./reduction_coordinate_bounds.bend as Coordinates
import ./microkernel_storage_execution.bend as Execution
import ../../lib/proofs/binary_radix.bend as Radix
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/proofs/kernel_gemm_writes.bend as GemmWrites
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/machine_limit_bounds.bend as Limits

def stride_positive(+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool}) -> {Nat.is_lt(0n,Guards.stride(problem)) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Boolean.every_condition_holds(Guards.valid_conditions(ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias}),5n,valid)

# One lane: the accumulator over the whole-input tiles and the one over the
# window tiles are both the matrix dot of the lane's coordinate.
def lane_matches(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset: U32,+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+offset_alignment: TilePartitionBounds.BlockAligned<offset>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +channel: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool}) ->
  {GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(Nat.mod(channel,4n),GemmReadback.unit_accumulators(
      MicroLeaf.leaf_query(limits,problem,storage_contract,MicroLeaf.packed_tiles(problem,storage_contract,offset,span),span,blocks,channel,column,span_exact,
        Support.fast(limits,problem,supported),span_bound,channel_inside,column_inside),Nat.div(channel,4n),Nat.div(column,8n)))) ==
    GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(Nat.mod(channel,4n),Congruence.other(
      MicroLeaf.leaf_query(limits,problem,storage_contract,MicroLeaf.packed_tiles(problem,storage_contract,offset,span),span,blocks,channel,column,span_exact,
        Support.fast(limits,problem,supported),span_bound,channel_inside,column_inside),
      MicroLeaf.packed_tiles(Samples.window(problem,U32.to_nat(first),U32.to_nat(rows)),
        WindowProblem.contract(problem,storage_contract,U32.to_nat(first),U32.to_nat(rows),tree,contents,room),
        KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first),span),
      Nat.div(channel,4n),Nat.div(column,8n)))) : F32}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence} = storage_contract
  TilePartitionBounds.BlockAligned{+offset_blocks,+offset_exact} = offset_alignment
  Positions.Cover{+skip,+first_exact,+fits,+rows_positive,+kernel_fits,+top,+bottom} = cover
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +storage_contract = {Inputs.Contract{input_evidence,weight_evidence,bias_evidence} : Inputs.Contract<problem>}
  +first_rows = U32.to_nat(first)
  +window_rows = U32.to_nat(rows)
  +start_nat = U32.to_nat(offset)
  +span_nat = U32.to_nat(span)
  +cover = {Positions.Cover{skip,first_exact,fits,rows_positive,kernel_fits,top,bottom} : Positions.Cover<problem,first_rows,window_rows,start_nat,span_nat>}
  +offset_alignment = {TilePartitionBounds.BlockAligned{offset_blocks,offset_exact} : TilePartitionBounds.BlockAligned<offset>}
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +support = Padded.scalar(limits,problem,fast)
  +stride_ok = stride_positive(problem,valid)
  +out_height = ConvSpec.axis_size(height,kernel,stride,padding)
  +out_width = ConvSpec.axis_size(image_width,kernel,stride,padding)
  +width_ok = {{==} : {Nat.is_lt(0n,out_width) == True{} : Bool}}
  +window = Samples.window(problem,first_rows,window_rows)
  +window_contract = WindowProblem.contract(problem,storage_contract,first_rows,window_rows,tree,contents,room)
  +window_start = KernelWindow.start(offset,U32.from_nat(stride),U32.from_nat(out_width),first)
  +full_tiles = MicroLeaf.packed_tiles(problem,storage_contract,offset,span)
  +window_tiles = MicroLeaf.packed_tiles(window,window_contract,window_start,span)
  +skipped = Nat.mul(skip,out_width)
  +span_positive = Order.le_lt_trans(0n,column,span_nat,Order.zero_le(column),column_inside)
  +grid = {%Positions.aligned_padded(problem,width_blocks,width) : {Nat.is_le(Nat.add(start_nat,span_nat),_) == True{} : Bool}
    range : {Nat.is_le(Nat.add(start_nat,span_nat),Nat.mul(out_height,out_width)) == True{} : Bool}}
  +offset_room = Order.transitive(start_nat,Nat.add(start_nat,span_nat),U32.to_nat(Machine.capacity(limits)),Order.le_add(start_nat,span_nat),
    Order.transitive(Nat.add(start_nat,span_nat),Support.padded(problem),U32.to_nat(Machine.capacity(limits)),range,Padded.padded_bound(limits,problem,facts,fast)))
  +reach = Order.transitive(skipped,Nat.mul(Nat.div(start_nat,out_width),out_width),start_nat,
    RowMajor.mul_le(skip,Nat.div(start_nat,out_width),out_width,Positions.block_le_start(problem,first_rows,skip,start_nat,stride_ok,first_exact,top)),
    DivisionOrder.product_below(start_nat,out_width,width_ok))
  +start_exact = Start.start_exact(limits,problem,facts,support,offset,first,skip,first_exact,reach,offset_room)
  # The lane's position.
  +position = Nat.add(start_nat,column)
  +after = Order.le_add(start_nat,column)
  +inside = {%Equal.sym(Bool,Nat.is_lt(Nat.add(start_nat,column),Nat.add(start_nat,span_nat)),Nat.is_lt(column,span_nat),Samples.lt_offset(start_nat,column,span_nat)) :
      {_ == True{} : Bool}
    column_inside : {Nat.is_lt(position,Nat.add(start_nat,span_nat)) == True{} : Bool}}
  +row = Nat.div(position,out_width)
  +column_index = Nat.mod(position,out_width)
  +row_valid = Positions.quotient_lt(position,out_height,out_width,width_ok,Samples.lt_le(position,Nat.add(start_nat,span_nat),Nat.mul(out_height,out_width),inside,grid))
  +column_valid = DivisionOrder.remainder_small(position,out_width,width_ok)
  +ordered = Positions.block_le_row(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after)
  +virtual_row = Nat.sub(row,skip)
  # The last lane bounds the window interval.
  +last = Nat.sub(Nat.add(start_nat,span_nat),1n)
  +last_after = Order.transitive(start_nat,position,last,after,Positions.last_row(position,start_nat,span_nat,inside))
  +whole_positive = Order.le_lt_trans(0n,position,Nat.add(start_nat,span_nat),Order.zero_le(position),inside)
  +whole_exact = Order.positive_has_last(Nat.add(start_nat,span_nat),whole_positive)
  +last_inside = {%whole_exact : {Nat.is_lt(last,_) == True{} : Bool}
    Order.lt_succ(last,last,Order.reflexive(last)) : {Nat.is_lt(last,Nat.add(start_nat,span_nat)) == True{} : Bool}}
  +last_row = Nat.div(last,out_width)
  +last_valid = Positions.quotient_lt(last,out_height,out_width,width_ok,Samples.lt_le(last,Nat.add(start_nat,span_nat),Nat.mul(out_height,out_width),last_inside,grid))
  +last_ordered = Positions.block_le_row(problem,first_rows,window_rows,start_nat,span_nat,cover,last,stride_ok,width_ok,last_after)
  +last_virtual = Positions.virtual_row_inside(problem,first_rows,window_rows,start_nat,span_nat,cover,last,stride_ok,width_ok,last_after,last_inside,last_valid)
  +window_height = ConvSpec.axis_size(window_rows,kernel,stride,padding)
  +window_total = {%Equal.sym(Nat,U32.to_nat(window_start),Nat.sub(start_nat,skipped),start_exact) :
      {Nat.is_le(Nat.add(_,span_nat),Nat.mul(window_height,out_width)) == True{} : Bool}
    %Positions.sub_after(start_nat,skipped,span_nat,reach) : {Nat.is_le(_,Nat.mul(window_height,out_width)) == True{} : Bool}
    Order.transitive(Nat.sub(Nat.add(start_nat,span_nat),skipped),Nat.mul(1n+Nat.sub(last_row,skip),out_width),Nat.mul(window_height,out_width),
      %Positions.sub_succ(last_row,skip,last_ordered) : {Nat.is_le(Nat.sub(Nat.add(start_nat,span_nat),skipped),Nat.mul(_,out_width)) == True{} : Bool}
      %RowMajor.subtract_products(1n+last_row,skip,out_width) : {Nat.is_le(Nat.sub(Nat.add(start_nat,span_nat),skipped),_) == True{} : Bool}
      DivisionOrder.subtract_le(Nat.add(start_nat,span_nat),Nat.mul(1n+last_row,out_width),skipped,
        %whole_exact : {Nat.is_le(_,Nat.mul(1n+last_row,out_width)) == True{} : Bool}
        Order.strict_follow(last,Nat.mul(1n+last_row,out_width),DivisionOrder.next_above(last,out_width,width_ok))),
      RowMajor.mul_le(1n+Nat.sub(last_row,skip),window_height,out_width,Order.strict_follow(Nat.sub(last_row,skip),window_height,last_virtual))) :
    {Nat.is_le(Nat.add(U32.to_nat(window_start),span_nat),Nat.mul(window_height,out_width)) == True{} : Bool}}
  +window_range = {%Equal.sym(Nat,Support.padded(window),Nat.mul(window_height,out_width),Positions.aligned_padded(window,width_blocks,width)) :
      {Nat.is_le(Nat.add(U32.to_nat(window_start),span_nat),_) == True{} : Bool}
    window_total : {Nat.is_le(Nat.add(U32.to_nat(window_start),span_nat),Support.padded(window)) == True{} : Bool}}
  +window_span_bound = Order.transitive(span_nat,Nat.add(U32.to_nat(window_start),span_nat),Support.padded(window),
    Samples.le_after(U32.to_nat(window_start),span_nat),window_range)
  +window_alignment = {TilePartitionBounds.BlockAligned{Nat.sub(offset_blocks,Nat.mul(skip,width_blocks)),
    Equal.trans(Nat,U32.to_nat(window_start),Nat.sub(start_nat,skipped),Nat.mul(Nat.sub(offset_blocks,Nat.mul(skip,width_blocks)),8n),start_exact,
      %Equal.sym(Nat,start_nat,Nat.mul(offset_blocks,8n),offset_exact) : {Nat.sub(_,skipped) == Nat.mul(Nat.sub(offset_blocks,Nat.mul(skip,width_blocks)),8n) : Nat}
      %width : {Nat.sub(Nat.mul(offset_blocks,8n),Nat.mul(skip,_)) == Nat.mul(Nat.sub(offset_blocks,Nat.mul(skip,width_blocks)),8n) : Nat}
      %RowMajor.assoc(skip,width_blocks,8n) :
        {Nat.sub(Nat.mul(offset_blocks,8n),_) == Nat.mul(Nat.sub(offset_blocks,Nat.mul(skip,width_blocks)),8n) : Nat}
      RowMajor.subtract_products(offset_blocks,Nat.mul(skip,width_blocks),8n))} : TilePartitionBounds.BlockAligned<window_start>}
  +window_valid = WindowProblem.valid(problem,first_rows,window_rows,valid,rows_positive,kernel_fits)
  +window_supported = WindowProblem.supported(limits,problem,first_rows,window_rows,valid,supported,fits)
  +window_positive = MicroLeaf.leaf_positive(limits,window,Guards.valid(window,window_valid),Support.fast(limits,window,window_supported),span,blocks,span_exact,
    window_span_bound,span_positive)
  +window_certificate = {Prepared.LeafEvidence{window_contract,window_span_bound,window_alignment,TilePartitionBounds.BlockAligned{blocks,span_exact},window_range,
    MicroLeaf.packed_matches(window,window_contract,window_start,span,window_positive)} : Prepared.LeafCertificate(window,window_start,span,window_tiles)}
  +full_certificate = {Prepared.LeafEvidence{storage_contract,span_bound,offset_alignment,TilePartitionBounds.BlockAligned{blocks,span_exact},range,
    MicroLeaf.packed_matches(problem,storage_contract,offset,span,
      MicroLeaf.leaf_positive(limits,problem,facts,fast,span,blocks,span_exact,span_bound,span_positive))} : Prepared.LeafCertificate(problem,offset,span,full_tiles)}
  +window_position = {%Equal.sym(Nat,U32.to_nat(window_start),Nat.sub(start_nat,skipped),start_exact) :
      {Nat.add(_,column) == Nat.add(Nat.mul(virtual_row,out_width),column_index) : Nat}
    Positions.window_position(start_nat,column,out_width,skip,width_ok,reach,ordered) :
    {Nat.add(U32.to_nat(window_start),column) == Nat.add(Nat.mul(virtual_row,out_width),column_index) : Nat}}
  +window_row = Positions.virtual_row_inside(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after,inside,row_valid)
  +indices = MatrixSpec.kchannels(ci,0n,kernel)
  +full_dot = MicroLeaf.selection_matches(~LeafContract.dense_evidence,~LeafContract.dense_samples,limits,problem,storage_contract,valid,supported,full_tiles,offset,span,blocks,channel,column,row,column_index,
    full_certificate,span_exact,span_bound,channel_inside,column_inside,DivisionOrder.quotient_exact(position,out_width,width_ok),row_valid,column_valid)
  +window_dot = MicroLeaf.selection_matches(~LeafContract.dense_evidence,~LeafContract.dense_samples,limits,window,window_contract,window_valid,window_supported,window_tiles,window_start,span,blocks,channel,column,
    virtual_row,column_index,window_certificate,span_exact,window_span_bound,channel_inside,column_inside,window_position,window_row,column_valid)
  +transfer = Samples.matrix_dot(problem,first_rows,window_rows,skip,channel,virtual_row,column_index,first_exact,fits,
    Positions.virtual_top(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after),
    Positions.virtual_bottom(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after,inside))
  +restored = Equal.cong(Nat,F32,value => Values.matrix_dot(indices,problem,channel,value,column_index,0.0),Nat.add(skip,virtual_row),row,
    Positions.add_sub(skip,row,ordered))
  +query = MicroLeaf.leaf_query(limits,problem,storage_contract,full_tiles,span,blocks,channel,column,span_exact,fast,span_bound,channel_inside,column_inside)
  +left = GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(Nat.mod(channel,4n),GemmReadback.unit_accumulators(query,Nat.div(channel,4n),Nat.div(column,8n))))
  +right = GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(Nat.mod(channel,4n),Congruence.other(query,window_tiles,Nat.div(channel,4n),Nat.div(column,8n))))
  Equal.trans(F32,left,Values.matrix_dot(indices,problem,channel,row,column_index,0.0),right,full_dot,
    Equal.sym(F32,right,Values.matrix_dot(indices,problem,channel,row,column_index,0.0),
      Equal.trans(F32,right,Values.matrix_dot(indices,window,channel,virtual_row,column_index,0.0),Values.matrix_dot(indices,problem,channel,row,column_index,0.0),
        window_dot,
        Equal.trans(F32,Values.matrix_dot(indices,window,channel,virtual_row,column_index,0.0),
          Values.matrix_dot(indices,problem,channel,Nat.add(skip,virtual_row),column_index,0.0),Values.matrix_dot(indices,problem,channel,row,column_index,0.0),
          transfer,restored))))

# The data an agreement proof needs, attached to one query and one tile tree.
type Link<-query: GemmReadback.Query,-tiles: Tree.Tree<Tile.Tile8>> is Data:
  Link{limits: Machine.Limits,problem: ConvSpec.Problem,storage_contract: Inputs.Contract<problem>,
    valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
    offset: U32,span: U32,blocks: Nat,span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
    span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},offset_alignment: TilePartitionBounds.BlockAligned<offset>,
    range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
    width_blocks: Nat,width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
    tree: Tree.Tree<F32>,first: U32,rows: U32,
    cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
    contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
    room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
    channel_zero: {Nat.is_lt(0n,Guards.output_channel_count(problem)) == True{} : Bool},
    column_zero: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
    query_exact: {query == MicroLeaf.leaf_query(limits,problem,storage_contract,MicroLeaf.packed_tiles(problem,storage_contract,offset,span),span,blocks,0n,0n,
      span_exact,Support.fast(limits,problem,supported),span_bound,channel_zero,column_zero) : GemmReadback.Query},
    tiles_exact: {tiles == MicroLeaf.packed_tiles(Samples.window(problem,U32.to_nat(first),U32.to_nat(rows)),
      WindowProblem.contract(problem,storage_contract,U32.to_nat(first),U32.to_nat(rows),tree,contents,room),
      KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first),span) : Tree.Tree<Tile.Tile8>}}

def Linked(query: GemmReadback.Query,tiles: Tree.Tree<Tile.Tile8>) -> Data:
  Link<query,tiles>

def agree(+query: GemmReadback.Query,+tiles: Tree.Tree<Tile.Tile8>,+unit: Nat,+group: Nat,+row: Nat,+lane: Nat,+link: Linked(query,tiles),
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(GemmReadback.rows(query))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,GemmReadback.groups(query)) == True{} : Bool},
  +row_small: {Nat.is_lt(row,4n) == True{} : Bool},+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +row_inside: {Nat.is_lt(GemmReadback.row_number(unit,row),U32.to_nat(GemmReadback.rows(query))) == True{} : Bool}) ->
  {GemmReadback.select_lane(lane,GemmReadback.select_row(row,GemmReadback.unit_accumulators(query,unit,group))) ==
    GemmReadback.select_lane(lane,GemmReadback.select_row(row,Congruence.other(query,tiles,unit,group))) : F32}:
  Link{+limits,+problem,+storage_contract,+valid,+supported,+offset,+span,+blocks,+span_exact,+span_bound,+offset_alignment,+range,
    +width_blocks,+width,+tree,+first,+rows,+cover,+contents,+room,+channel_zero,+column_zero,+query_exact,+tiles_exact} = link
  +fast = Support.fast(limits,problem,supported)
  +base = MicroLeaf.leaf_query(limits,problem,storage_contract,MicroLeaf.packed_tiles(problem,storage_contract,offset,span),span,blocks,0n,0n,
    span_exact,fast,span_bound,channel_zero,column_zero)
  +window_tiles = MicroLeaf.packed_tiles(Samples.window(problem,U32.to_nat(first),U32.to_nat(rows)),
    WindowProblem.contract(problem,storage_contract,U32.to_nat(first),U32.to_nat(rows),tree,contents,room),
    KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first),span)
  +channel = GemmReadback.row_number(unit,row)
  +column = Nat.add(Nat.mul(group,8n),lane)
  +channel_inside = {%SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,fast)) : {Nat.is_lt(channel,_) == True{} : Bool}
    %Equal.cong(GemmReadback.Query,Nat,value => U32.to_nat(GemmReadback.rows(value)),query,base,query_exact) : {Nat.is_lt(channel,_) == True{} : Bool}
    row_inside : {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}}
  +group_bound = {%Equal.cong(GemmReadback.Query,Nat,value => GemmReadback.groups(value),query,base,query_exact) : {Nat.is_lt(group,_) == True{} : Bool}
    group_inside : {Nat.is_lt(group,blocks) == True{} : Bool}}
  +column_inside = {%Equal.sym(Nat,U32.to_nat(span),Nat.mul(blocks,8n),span_exact) : {Nat.is_lt(column,_) == True{} : Bool}
    RowMajor.radix(group,blocks,lane,8n,group_bound,lane_small) : {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool}}
  %Equal.sym(GemmReadback.Query,query,base,query_exact) :
    {GemmReadback.select_lane(lane,GemmReadback.select_row(row,GemmReadback.unit_accumulators(_,unit,group))) ==
      GemmReadback.select_lane(lane,GemmReadback.select_row(row,Congruence.other(_,tiles,unit,group))) : F32}
  %Equal.sym(Tree.Tree<Tile.Tile8>,tiles,window_tiles,tiles_exact) :
    {GemmReadback.select_lane(lane,GemmReadback.select_row(row,GemmReadback.unit_accumulators(base,unit,group))) ==
      GemmReadback.select_lane(lane,GemmReadback.select_row(row,Congruence.other(base,_,unit,group))) : F32}
  %Radix.remainder(group,lane,8n,{==},lane_small) :
    {GemmReadback.select_lane(_,GemmReadback.select_row(row,GemmReadback.unit_accumulators(base,unit,group))) ==
      GemmReadback.select_lane(_,GemmReadback.select_row(row,Congruence.other(base,window_tiles,unit,group))) : F32}
  %Radix.remainder(unit,row,4n,{==},row_small) :
    {GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(_,GemmReadback.unit_accumulators(base,unit,group))) ==
      GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(_,Congruence.other(base,window_tiles,unit,group))) : F32}
  %Radix.quotient(unit,row,4n,{==},row_small) :
    {GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(Nat.mod(channel,4n),GemmReadback.unit_accumulators(base,_,group))) ==
      GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(Nat.mod(channel,4n),Congruence.other(base,window_tiles,_,group))) : F32}
  %Radix.quotient(group,lane,8n,{==},lane_small) :
    {GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(Nat.mod(channel,4n),GemmReadback.unit_accumulators(base,Nat.div(channel,4n),_))) ==
      GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(Nat.mod(channel,4n),Congruence.other(base,window_tiles,Nat.div(channel,4n),_))) : F32}
  lane_matches(limits,problem,storage_contract,valid,supported,offset,span,blocks,span_exact,span_bound,offset_alignment,range,width_blocks,width,
    tree,first,rows,cover,contents,room,channel,column,channel_inside,column_inside)

def computed_output(+rows: U32,+span: U32,+inner: U32,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+packed: PackingStorage.Packed) ->
  {Execution.computed(rows,span,inner,weights,bias,packed) ==
    GemmLaws.run(rows,Prepared.packed_output(packed),weights,bias,MicroLeaf.allocated(rows,span),span,inner) : Tree.Tree<F32>}:
  PackingStorage.Packed{input,output} = packed
  {==}

def work(+problem: ConvSpec.Problem) -> Pipeline.Work:
  NativeConv.work(ConvShape.from_problem(problem))

# A band window leaf over aligned rows stores the whole-input leaf output.
def band_leaf(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset: U32,+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+offset_alignment: TilePartitionBounds.BlockAligned<offset>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool}) ->
  {Execution.leaf(work(problem),offset,span,Execution.InputWindow{tree,first,rows},GemmDot.reordered(problem,storage_contract),Inputs.bias_tree(problem,storage_contract)) ==
    MicroLeaf.layout_result(problem,storage_contract,Partition.Leaf{Unit{},offset,span,Unit{}}) : GatherModel.ConvResult()}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence} = storage_contract
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +storage_contract = {Inputs.Contract{input_evidence,weight_evidence,bias_evidence} : Inputs.Contract<problem>}
  +channel_zero = Boolean.every_condition_holds(Guards.valid_conditions(problem),9n,valid)
  +fast = Support.fast(limits,problem,supported)
  +base = MicroLeaf.leaf_query(limits,problem,storage_contract,MicroLeaf.packed_tiles(problem,storage_contract,offset,span),span,blocks,0n,0n,
    span_exact,fast,span_bound,channel_zero,span_positive)
  +out_width = ConvSpec.axis_size(image_width,kernel,stride,padding)
  +window = Samples.window(problem,U32.to_nat(first),U32.to_nat(rows))
  +window_contract = WindowProblem.contract(problem,storage_contract,U32.to_nat(first),U32.to_nat(rows),tree,contents,room)
  +window_start = KernelWindow.start(offset,U32.from_nat(stride),U32.from_nat(out_width),first)
  +window_tiles = MicroLeaf.packed_tiles(window,window_contract,window_start,span)
  +link = {Link{limits,problem,storage_contract,valid,supported,offset,span,blocks,span_exact,span_bound,offset_alignment,range,width_blocks,width,
    tree,first,rows,cover,contents,room,channel_zero,span_positive,{==},{==}} : Linked(base,window_tiles)}
  +channels = MicroLeaf.rows(problem)
  +inner = Coordinates.native_inner(problem)
  +reordered = GemmDot.reordered(problem,storage_contract)
  +bias_tree = Inputs.bias_tree(problem,storage_contract)
  +output = MicroLeaf.allocated(channels,span)
  +count = (span / 8 * inner : U32)
  +call = {Packing.Call{U32.to_nat(count),window_start,U32.from_nat(U32.to_nat(rows)),U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),
    U32.from_nat(padding),U32.from_nat(out_width),inner} : Packing.Call}
  +prepared = Execution.packed(call,tree,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))
  %Arithmetic.native_roundtrip(rows) :
    {Execution.leaf(work(problem),offset,span,Execution.InputWindow{tree,first,_},reordered,bias_tree) ==
      MicroLeaf.layout_result(problem,storage_contract,Partition.Leaf{Unit{},offset,span,Unit{}}) : GatherModel.ConvResult()}
  %Equal.sym(Tree.Tree<F32>,Execution.computed(channels,span,inner,reordered,bias_tree,prepared),
    GemmLaws.run(channels,Prepared.packed_output(prepared),reordered,bias_tree,output,span,inner),
    computed_output(channels,span,inner,reordered,bias_tree,prepared)) :
    {Partition.Leaf{_,offset,span,channels} == Partition.Leaf{MicroLeaf.leaf_output(problem,storage_contract,offset,span),offset,span,channels} : GatherModel.ConvResult()}
  Equal.cong(Tree.Tree<F32>,GatherModel.ConvResult(),value => Partition.Leaf{value,offset,span,channels},
    GemmLaws.run(channels,window_tiles,reordered,bias_tree,output,span,inner),MicroLeaf.leaf_output(problem,storage_contract,offset,span),
    Equal.sym(Tree.Tree<F32>,MicroLeaf.leaf_output(problem,storage_contract,offset,span),GemmLaws.run(channels,window_tiles,reordered,bias_tree,output,span,inner),
      Congruence.run_congruence(~Linked,~agree,base,window_tiles,link,output)))

# An empty span schedules no GEMM job, whatever tiles it packs.
def zero_jobs(+limits: Machine.Limits,+rows: U32,+span: U32,+empty: {U32.to_nat(span) == 0n : Nat}) -> {GemmWrites.job_count(rows,span) == 0n : Nat}:
  +units = ((rows + 3) / 4 : U32)
  +groups = (span / 8 : U32)
  +groups_zero = {Equal.trans(Nat,U32.to_nat(groups),Nat.div(U32.to_nat(span),U32.to_nat(8)),Nat.div(0n,U32.to_nat(8)),
      U32Division.divide(span,8,2147483648,{==},{==},{==}),
      Equal.cong(Nat,Nat,value => Nat.div(value,U32.to_nat(8)),U32.to_nat(span),0n,empty)) : {U32.to_nat(groups) == 0n : Nat}}
  Equal.trans(Nat,U32.to_nat((groups * units : U32)),Nat.mul(U32.to_nat(groups),U32.to_nat(units)),0n,
    Arithmetic.multiply(groups,units,Machine.capacity(limits),Limits.capacity_fits(limits),
      %Equal.sym(Nat,U32.to_nat(groups),0n,groups_zero) : {Nat.is_le(Nat.mul(_,U32.to_nat(units)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      Order.zero_le(U32.to_nat(Machine.capacity(limits)))),
    %Equal.sym(Nat,U32.to_nat(groups),0n,groups_zero) : {Nat.mul(_,U32.to_nat(units)) == 0n : Nat}
    {==})

def empty_run(+limits: Machine.Limits,+rows: U32,+tiles: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,
  +output: Tree.Tree<F32>,+span: U32,+inner: U32,+empty: {U32.to_nat(span) == 0n : Nat}) ->
  {GemmLaws.run(rows,tiles,weights,bias,output,span,inner) == output : Tree.Tree<F32>}:
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run(rows,tiles,weights,bias,output,span,inner),
    Writes.pure(F32,GemmWrites.all_writes(GemmWrites.job_count(rows,span),0,Gemm.Layout{rows,span,inner},tiles,weights,bias),output),
    GemmWrites.run_writes(rows,tiles,weights,bias,output,span,inner)) : {_ == output : Tree.Tree<F32>}
  %Equal.sym(Nat,GemmWrites.job_count(rows,span),0n,zero_jobs(limits,rows,span,empty)) :
    {Writes.pure(F32,GemmWrites.all_writes(_,0,Gemm.Layout{rows,span,inner},tiles,weights,bias),output) == output : Tree.Tree<F32>}
  {==}

# A leaf over an empty span stores its fresh output for any window.
def empty_leaf(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+offset: U32,+span: U32,
  +tree: Tree.Tree<F32>,+first: U32,+rows: U32,+empty: {U32.to_nat(span) == 0n : Nat}) ->
  {Execution.leaf(work(problem),offset,span,Execution.InputWindow{tree,first,rows},GemmDot.reordered(problem,storage_contract),Inputs.bias_tree(problem,storage_contract)) ==
    MicroLeaf.layout_result(problem,storage_contract,Partition.Leaf{Unit{},offset,span,Unit{}}) : GatherModel.ConvResult()}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence} = storage_contract
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +storage_contract = {Inputs.Contract{input_evidence,weight_evidence,bias_evidence} : Inputs.Contract<problem>}
  +out_width = ConvSpec.axis_size(image_width,kernel,stride,padding)
  +channels = MicroLeaf.rows(problem)
  +inner = Coordinates.native_inner(problem)
  +reordered = GemmDot.reordered(problem,storage_contract)
  +bias_tree = Inputs.bias_tree(problem,storage_contract)
  +output = MicroLeaf.allocated(channels,span)
  +count = (span / 8 * inner : U32)
  +call = {Packing.Call{U32.to_nat(count),KernelWindow.start(offset,U32.from_nat(stride),U32.from_nat(out_width),first),rows,U32.from_nat(image_width),
    U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),U32.from_nat(out_width),inner} : Packing.Call}
  +prepared = Execution.packed(call,tree,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))
  +full_tiles = MicroLeaf.packed_tiles(problem,storage_contract,offset,span)
  %Equal.sym(Tree.Tree<F32>,Execution.computed(channels,span,inner,reordered,bias_tree,prepared),
    GemmLaws.run(channels,Prepared.packed_output(prepared),reordered,bias_tree,output,span,inner),
    computed_output(channels,span,inner,reordered,bias_tree,prepared)) :
    {Partition.Leaf{_,offset,span,channels} == Partition.Leaf{MicroLeaf.leaf_output(problem,storage_contract,offset,span),offset,span,channels} : GatherModel.ConvResult()}
  Equal.cong(Tree.Tree<F32>,GatherModel.ConvResult(),value => Partition.Leaf{value,offset,span,channels},
    GemmLaws.run(channels,Prepared.packed_output(prepared),reordered,bias_tree,output,span,inner),MicroLeaf.leaf_output(problem,storage_contract,offset,span),
    Equal.trans(Tree.Tree<F32>,GemmLaws.run(channels,Prepared.packed_output(prepared),reordered,bias_tree,output,span,inner),output,
      MicroLeaf.leaf_output(problem,storage_contract,offset,span),
      empty_run(limits,channels,Prepared.packed_output(prepared),reordered,bias_tree,output,span,inner,empty),
      Equal.sym(Tree.Tree<F32>,MicroLeaf.leaf_output(problem,storage_contract,offset,span),output,
        empty_run(limits,channels,full_tiles,reordered,bias_tree,output,span,inner,empty))))

# The caller input as a window starts at row 0, so its leaf packs at the offset.
def root_leaf(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+offset: U32,+span: U32) ->
  {Execution.leaf(work(problem),offset,span,Execution.InputWindow{Inputs.input_tree(problem,storage_contract),0,U32.from_nat(Guards.height(problem))},
      GemmDot.reordered(problem,storage_contract),Inputs.bias_tree(problem,storage_contract)) ==
    MicroLeaf.layout_result(problem,storage_contract,Partition.Leaf{Unit{},offset,span,Unit{}}) : GatherModel.ConvResult()}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence} = storage_contract
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +storage_contract = {Inputs.Contract{input_evidence,weight_evidence,bias_evidence} : Inputs.Contract<problem>}
  +out_width = ConvSpec.axis_size(image_width,kernel,stride,padding)
  +channels = MicroLeaf.rows(problem)
  +inner = Coordinates.native_inner(problem)
  +reordered = GemmDot.reordered(problem,storage_contract)
  +bias_tree = Inputs.bias_tree(problem,storage_contract)
  +count = (span / 8 * inner : U32)
  +call = {Packing.Call{U32.to_nat(count),offset,U32.from_nat(height),U32.from_nat(image_width),
    U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),U32.from_nat(out_width),inner} : Packing.Call}
  +prepared = Execution.packed(call,Inputs.input_tree(problem,storage_contract),Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))
  %Equal.sym(Tree.Tree<F32>,Execution.computed(channels,span,inner,reordered,bias_tree,prepared),
    GemmLaws.run(channels,Prepared.packed_output(prepared),reordered,bias_tree,MicroLeaf.allocated(channels,span),span,inner),
    computed_output(channels,span,inner,reordered,bias_tree,prepared)) :
    {Partition.Leaf{_,offset,span,channels} == MicroLeaf.layout_result(problem,storage_contract,Partition.Leaf{Unit{},offset,span,Unit{}}) : GatherModel.ConvResult()}
  {==}
