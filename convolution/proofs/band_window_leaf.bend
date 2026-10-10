# The window given to a convolution leaf and its bounds.
import Base
import ./blocked_pipeline_initial.bend as BlockedInitial
import ./blocked_convolution_readback.bend as BlockedReadback
import ./blocked_pipeline_model.bend as BlockedModel
import ./blocked_pipeline_bounds.bend as BlockBounds
import ../../lib/proofs/kernel_pipeline_activation.bend as ActivationProofs
import ../../lib/proofs/tensor_band_window.bend as TensorBandWindow
import ./band_leaf_execution.bend as LeafExecution
import ./concrete_support_guard.bend as Support
import ./convolution_guard_facts.bend as Guards
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../inference.bend as NativeConv
import ../matrix_specification.bend as MatrixSpec
import ./packing_contents.bend as PackingContents
import ../../lib/proofs/natural_addition.bend as Addition
import ./convolution_leaf_semantics.bend as LeafSemantics
import ./gemm_dot_matches_matrix.bend as GemmDot
import ./input_storage.bend as Inputs
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./output_values.bend as Values
import ./packing_tile_bounds.bend as PackingBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./supported_padded_coordinates.bend as Padded
import ./tile_partition_bounds.bend as TilePartitionBounds
import ./window_band_values.bend as WindowBandValues
import ./window_leaf_readback.bend as LeafReadback
import ./window_positions.bend as Positions
import ./window_problem.bend as WindowProblem
import ./window_samples.bend as Samples
import ./window_start.bend as Start
import ../scalar_kernel.bend as ScalarConv
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_extrema.bend as Extrema
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/u32_extrema.bend as U32Extrema

# A band leaf packs the image rows its outputs read as a shorter image and
# computes its output rows at any offset. At every real output position the
# stored value is the original matrix value: the window problem's samples
# are the original samples, and the leaf itself needs no offset alignment.
def window_problem(+problem: ConvSpec.Problem,+first: U32,+rows: U32) -> ConvSpec.Problem:
  Samples.window(problem,U32.to_nat(first),U32.to_nat(rows))

def window_start(+problem: ConvSpec.Problem,+offset: U32,+first: U32) -> U32:
  KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first)

def padded(+positions: U32) -> U32:
  ((positions + 7) / 8 * 8 : U32)

# Output positions [offset, offset + positions) of a band leaf, whose window
# rows [first, first + rows) hold every row those outputs read.
# The packing call of a window leaf, stated in the window problem's terms.
def window_call(+problem: ConvSpec.Problem,+offset: U32,+first: U32,+rows: U32,+positions: U32) -> Packing.Call:
  +window = window_problem(problem,first,rows)
  Packing.Call{U32.to_nat(PackingBounds.count(padded(positions),window)),window_start(problem,offset,first),
    U32.from_nat(Guards.height(window)),U32.from_nat(Guards.width(window)),U32.from_nat(Guards.kernel(window)),U32.from_nat(Guards.stride(window)),
    U32.from_nat(Guards.pad(window)),U32.from_nat(ConvSpec.out_width(window)),Coordinates.native_inner(window)}

def output_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,fast: Support.FastSupported<limits,problem>) -> {Support.output_bound(limits,problem) == True{} : Bool}:
  Support.FastSupported{scalar,packing,output} = fast
  output

# The whole band leaf (packing, GEMM, activation and finish) at each real
# output position holds the activated original matrix value.

def leaf_value(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.weights(problem)),Guards.weights(problem)>,
  +bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.bias(problem)),Guards.bias(problem)>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset: U32,+positions: U32,
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(positions)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},
  +tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(positions)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +channel: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(positions)) == True{} : Bool}) ->
  {Tree.read(F32,LeafExecution.leaf_tree(~custom,activation,MicroLeaf.rows(window_problem(problem,first,rows)),
      Coordinates.native_inner(window_problem(problem,first,rows)),positions,window_call(problem,offset,first,rows,positions),tree,
      GemmDot.reordered(window_problem(problem,first,rows),WindowProblem.band_contract(problem,weight_evidence,bias_evidence,U32.to_nat(first),U32.to_nat(rows),tree,contents,room)),
      Inputs.bias_tree(window_problem(problem,first,rows),WindowProblem.band_contract(problem,weight_evidence,bias_evidence,U32.to_nat(first),U32.to_nat(rows),tree,contents,room))),
      U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),column))) ==
    ActivationProofs.activation_value(~custom,activation,
      Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)))) : F32}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  Positions.Cover{+skip,+first_exact,+fits,+rows_positive,+kernel_fits,+top,+bottom} = cover
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +first_rows = U32.to_nat(first)
  +window_rows = U32.to_nat(rows)
  +start_nat = U32.to_nat(offset)
  +span_nat = U32.to_nat(positions)
  +cover = {Positions.Cover{skip,first_exact,fits,rows_positive,kernel_fits,top,bottom} : Positions.Cover<problem,first_rows,window_rows,start_nat,span_nat>}
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +support = Padded.scalar(limits,problem,fast)
  +stride_ok = LeafReadback.stride_positive(problem,valid)
  +out_height = ConvSpec.axis_size(height,kernel,stride,padding)
  +out_width = ConvSpec.axis_size(image_width,kernel,stride,padding)
  +width_ok = {{==} : {Nat.is_lt(0n,out_width) == True{} : Bool}}
  +window = Samples.window(problem,first_rows,window_rows)
  +window_contract = WindowProblem.band_contract(problem,weight_evidence,bias_evidence,first_rows,window_rows,tree,contents,room)
  +window_start = KernelWindow.start(offset,U32.from_nat(stride),U32.from_nat(out_width),first)
  +span = padded(positions)
  +skipped = Nat.mul(skip,out_width)
  +grid_room = Order.transitive(Nat.add(start_nat,span_nat),Nat.mul(out_height,out_width),U32.to_nat(Machine.capacity(limits)),grid,
    Order.transitive(Nat.mul(out_height,out_width),Support.padded(problem),U32.to_nat(Machine.capacity(limits)),Padded.spatial_within_padded(limits,problem,facts,fast),
      Padded.padded_bound(limits,problem,facts,fast)))
  +offset_room = Order.transitive(start_nat,Nat.add(start_nat,span_nat),U32.to_nat(Machine.capacity(limits)),Order.le_add(start_nat,span_nat),grid_room)
  +positions_room = Order.transitive(span_nat,Nat.add(start_nat,span_nat),U32.to_nat(Machine.capacity(limits)),Samples.le_after(start_nat,span_nat),grid_room)
  +reach = Order.transitive(skipped,Nat.mul(Nat.div(start_nat,out_width),out_width),start_nat,
    RowMajor.mul_le(skip,Nat.div(start_nat,out_width),out_width,Positions.block_le_start(problem,first_rows,skip,start_nat,stride_ok,first_exact,top)),
    DivisionOrder.product_below(start_nat,out_width,width_ok))
  +start_exact = Start.start_exact(limits,problem,facts,support,offset,first,skip,first_exact,reach,offset_room)
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
  # The last real position bounds the output rows of the window.
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
  +window_valid = WindowProblem.valid(problem,first_rows,window_rows,valid,rows_positive,kernel_fits)
  +window_supported = WindowProblem.supported(limits,problem,first_rows,window_rows,valid,supported,fits)
  +window_position = {%Equal.sym(Nat,U32.to_nat(window_start),Nat.sub(start_nat,skipped),start_exact) :
      {Nat.add(_,column) == Nat.add(Nat.mul(virtual_row,out_width),column_index) : Nat}
    Positions.window_position(start_nat,column,out_width,skip,width_ok,reach,ordered) :
    {Nat.add(U32.to_nat(window_start),column) == Nat.add(Nat.mul(virtual_row,out_width),column_index) : Nat}}
  +window_row = Positions.virtual_row_inside(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after,inside,row_valid)
  +region = {BlockedInitial.Region{window_total,Order.le_lt_trans(0n,column,span_nat,Order.zero_le(column),column_inside)} : BlockedInitial.Region<window,window_start,positions>}
  +local_exact = {%window_position : {Nat.sub(_,U32.to_nat(window_start)) == column : Nat}
    Order.subtract_shift(U32.to_nat(window_start),column) : {BlockedReadback.local_position(window,window_start,virtual_row,column_index) == column : Nat}}
  +point = BlockedInitial.point(limits,window,window_start,positions,channel,virtual_row,column_index,window_valid,window_supported,region,channel_inside,window_row,column_valid,
    %window_position : {Nat.is_le(U32.to_nat(window_start),_) == True{} : Bool} Order.le_add(U32.to_nat(window_start),column),
    %Equal.sym(Nat,BlockedReadback.local_position(window,window_start,virtual_row,column_index),column,local_exact) : {Nat.is_lt(_,span_nat) == True{} : Bool} column_inside)
  +window_dot = {%local_exact : {Tree.read(F32,BlockedModel.run(~custom,activation,BlockedReadback.work(window,positions),window_start,Inputs.input_tree(window,window_contract),GemmDot.reordered(window,window_contract),Inputs.bias_tree(window,window_contract)),U32.from_nat(Nat.add(Nat.mul(channel,span_nat),_))) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(window,channel,virtual_row,column_index)) : F32}
    BlockedInitial.run_readback(~custom,activation,limits,window,window_start,positions,window_contract,channel,virtual_row,column_index,window_valid,window_supported,region,point) :
    {Tree.read(F32,BlockedModel.run(~custom,activation,BlockedReadback.work(window,positions),window_start,Inputs.input_tree(window,window_contract),GemmDot.reordered(window,window_contract),Inputs.bias_tree(window,window_contract)),U32.from_nat(Nat.add(Nat.mul(channel,span_nat),column))) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(window,channel,virtual_row,column_index)) : F32}}
  +indices = MatrixSpec.kchannels(ci,0n,kernel)
  +transfer = Samples.matrix_dot(problem,first_rows,window_rows,skip,channel,virtual_row,column_index,first_exact,fits,
    Positions.virtual_top(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after),
    Positions.virtual_bottom(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after,inside))
  +restored = Equal.cong(Nat,F32,value => Values.matrix_dot(indices,problem,channel,value,column_index,0.0),Nat.add(skip,virtual_row),row,
    Positions.add_sub(skip,row,ordered))
  Equal.trans(F32,Tree.read(F32,LeafExecution.leaf_tree(~custom,activation,MicroLeaf.rows(window),Coordinates.native_inner(window),positions,window_call(problem,offset,first,rows,positions),tree,GemmDot.reordered(window,window_contract),Inputs.bias_tree(window,window_contract)),U32.from_nat(Nat.add(Nat.mul(channel,span_nat),column))),
    ActivationProofs.activation_value(~custom,activation,Values.matrix_value(window,channel,virtual_row,column_index)),ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column_index)),window_dot,
    Equal.cong(F32,F32,value => ActivationProofs.activation_value(~custom,activation,value),Values.matrix_value(window,channel,virtual_row,column_index),Values.matrix_value(problem,channel,row,column_index),
      Equal.cong(F32,F32,dot => (dot + ConvSpec.bias_at(problem,channel) : F32),Values.matrix_dot(indices,window,channel,virtual_row,column_index,0.0),
        Values.matrix_dot(indices,problem,channel,row,column_index,0.0),
        Equal.trans(F32,Values.matrix_dot(indices,window,channel,virtual_row,column_index,0.0),Values.matrix_dot(indices,problem,channel,Nat.add(skip,virtual_row),column_index,0.0),
          Values.matrix_dot(indices,problem,channel,row,column_index,0.0),transfer,restored))))

def capacity(+limits: Machine.Limits) -> U32:
  U32.max(Machine.capacity(limits),Machine.coordinate(limits))

def selected_bound(selected: Bool,+left: U32,+right: U32,+limit: U32,left_bound: {U32.is_le(left,limit) == True{} : Bool},right_bound: {U32.is_le(right,limit) == True{} : Bool}) ->
  {U32.is_le(Bool.pick(U32,selected,right,left),limit) == True{} : Bool}:
  match selected:
    case False{}: left_bound
    case True{}: right_bound

def capacity_fits(+limits: Machine.Limits) -> {U32.is_le(capacity(limits),2147483648) == True{} : Bool}:
  %U32Extrema.pick_maximum(Machine.capacity(limits),Machine.coordinate(limits)) : {U32.is_le(_,2147483648) == True{} : Bool}
  selected_bound(U32.is_lt(Machine.capacity(limits),Machine.coordinate(limits)),Machine.capacity(limits),Machine.coordinate(limits),2147483648,Limits.capacity_fits(limits),Limits.coordinate_fits(limits))

def storage_room(+limits: Machine.Limits) -> {Nat.is_le(U32.to_nat(Machine.capacity(limits)),U32.to_nat(capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(capacity(limits)),Nat.max(U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.coordinate(limits))),U32Extrema.maximum(Machine.capacity(limits),Machine.coordinate(limits))) : {Nat.is_le(U32.to_nat(Machine.capacity(limits)),_) == True{} : Bool}
  Extrema.maximum_left(U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.coordinate(limits)))

def coordinate_room(+limits: Machine.Limits) -> {Nat.is_le(U32.to_nat(Machine.coordinate(limits)),U32.to_nat(capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(capacity(limits)),Nat.max(U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.coordinate(limits))),U32Extrema.maximum(Machine.capacity(limits),Machine.coordinate(limits))) : {Nat.is_le(U32.to_nat(Machine.coordinate(limits)),_) == True{} : Bool}
  Extrema.maximum_right(U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.coordinate(limits)))

def core(+limits: Machine.Limits,+images: U32,+height: U32,+first: U32,+rows: U32,geometry: TensorBandWindow.CoreGeometry<images,height,first,rows,Machine.capacity(limits)>) ->
  TensorBandWindow.CoreGeometry<images,height,first,rows,capacity(limits)>:
  TensorBandWindow.CoreGeometry{image_inside,core_inside,first_exact,core_exact,endpoint_room} = geometry
  TensorBandWindow.CoreGeometry{image_inside,core_inside,first_exact,core_exact,
    Order.transitive(Nat.add(U32.to_nat((first / height * height : U32)),U32.to_nat(height)),U32.to_nat(Machine.capacity(limits)),U32.to_nat(capacity(limits)),endpoint_room,storage_room(limits))}
