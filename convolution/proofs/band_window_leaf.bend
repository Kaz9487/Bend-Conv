# The window given to a convolution leaf and its bounds.
import Base
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
def ceiling_eight(+count: Nat) -> Nat:
  Nat.mul(Nat.div(Nat.add(count,7n),8n),8n)

def within_ceiling(+count: Nat) -> {Nat.is_le(count,ceiling_eight(count)) == True{} : Bool}:
  +total = Nat.add(count,7n)
  +quotient = Nat.div(total,8n)
  +product = Nat.mul(quotient,8n)
  Division.lt_succ_le(count,product,
    Order.cancel_offset_lt(7n,count,1n+product,
      %Order.commutative(count,7n) : {Nat.is_lt(_,Nat.add(7n,1n+product)) == True{} : Bool}
      %Equal.sym(Nat,total,Nat.add(product,Nat.mod(total,8n)),DivisionOrder.quotient_exact(total,8n,{==})) : {Nat.is_lt(_,Nat.add(7n,1n+product)) == True{} : Bool}
      %Order.commutative(product,8n) : {Nat.is_lt(Nat.add(product,Nat.mod(total,8n)),_) == True{} : Bool}
      Order.offset_lt(product,Nat.mod(total,8n),8n,DivisionOrder.remainder_small(total,8n,{==}))))

# The padded span of a leaf with count real positions, as a natural number.
def padded_exact(+limits: Machine.Limits,+count: U32,+bound: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(((count + 7) / 8 * 8 : U32)) == ceiling_eight(U32.to_nat(count)) : Nat}:
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +added_room = Padded.add_seven_bound(limits,U32.to_nat(count),bound)
  +added = Arithmetic.add(count,7,half,half_ok,added_room)
  +divided = U32Division.divide((count + 7 : U32),8,half,half_ok,Bounds.as_nat_le(16,half,Limits.sixteen_within_half(limits)),{==})
  +quotient = Nat.div(Nat.add(U32.to_nat(count),7n),8n)
  +quotient_exact = {%added : {U32.to_nat(((count + 7) / 8 : U32)) == Nat.div(_,8n) : Nat}
    divided : {U32.to_nat(((count + 7) / 8 : U32)) == quotient : Nat}}
  %quotient_exact : {U32.to_nat(((count + 7) / 8 * 8 : U32)) == Nat.mul(_,8n) : Nat}
  Arithmetic.multiply(((count + 7) / 8 : U32),8,half,half_ok,
    %Equal.sym(Nat,U32.to_nat(((count + 7) / 8 : U32)),quotient,quotient_exact) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(half)) == True{} : Bool}
    Order.transitive(Nat.mul(quotient,8n),Nat.add(U32.to_nat(count),7n),U32.to_nat(half),DivisionOrder.product_below(Nat.add(U32.to_nat(count),7n),8n,{==}),added_room))

def window_problem(+problem: ConvSpec.Problem,+first: U32,+rows: U32) -> ConvSpec.Problem:
  Samples.window(problem,U32.to_nat(first),U32.to_nat(rows))

def window_start(+problem: ConvSpec.Problem,+offset: U32,+first: U32) -> U32:
  KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first)

def padded(+positions: U32) -> U32:
  ((positions + 7) / 8 * 8 : U32)

# Output positions [offset, offset + positions) of a band leaf, whose window
# rows [first, first + rows) hold every row those outputs read.
def window_value(+limits: Machine.Limits,+problem: ConvSpec.Problem,
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
  {Tree.read(F32,MicroLeaf.leaf_output(window_problem(problem,first,rows),
      WindowProblem.band_contract(problem,weight_evidence,bias_evidence,U32.to_nat(first),U32.to_nat(rows),tree,contents,room),
      window_start(problem,offset,first),padded(positions)),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(padded(positions))),column))) ==
    Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem))) : F32}:
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
  +span_exact = padded_exact(limits,positions,positions_room)
  +window_span_bound = {%Equal.sym(Nat,U32.to_nat(span),ceiling_eight(span_nat),span_exact) : {Nat.is_le(_,Support.padded(window)) == True{} : Bool}
    WindowProblem.padded_le(span_nat,Nat.mul(window_height,out_width),
      Order.transitive(span_nat,Nat.add(U32.to_nat(window_start),span_nat),Nat.mul(window_height,out_width),Samples.le_after(U32.to_nat(window_start),span_nat),window_total)) :
    {Nat.is_le(U32.to_nat(span),Support.padded(window)) == True{} : Bool}}
  +span_alignment = {TilePartitionBounds.BlockAligned{Nat.div(Nat.add(span_nat,7n),8n),span_exact} : TilePartitionBounds.BlockAligned<span>}
  +padded_inside = {%Equal.sym(Nat,U32.to_nat(span),ceiling_eight(span_nat),span_exact) : {Nat.is_lt(column,_) == True{} : Bool}
    Samples.lt_le(column,span_nat,ceiling_eight(span_nat),column_inside,within_ceiling(span_nat)) : {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool}}
  +window_valid = WindowProblem.valid(problem,first_rows,window_rows,valid,rows_positive,kernel_fits)
  +window_supported = WindowProblem.supported(limits,problem,first_rows,window_rows,valid,supported,fits)
  +window_position = {%Equal.sym(Nat,U32.to_nat(window_start),Nat.sub(start_nat,skipped),start_exact) :
      {Nat.add(_,column) == Nat.add(Nat.mul(virtual_row,out_width),column_index) : Nat}
    Positions.window_position(start_nat,column,out_width,skip,width_ok,reach,ordered) :
    {Nat.add(U32.to_nat(window_start),column) == Nat.add(Nat.mul(virtual_row,out_width),column_index) : Nat}}
  +window_row = Positions.virtual_row_inside(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after,inside,row_valid)
  +window_range = {%Equal.sym(Nat,U32.to_nat(span),ceiling_eight(span_nat),span_exact) :
      {Nat.is_le(Nat.add(U32.to_nat(window_start),_),Nat.add(PackingContents.grid(window),7n)) == True{} : Bool}
    Order.transitive(Nat.add(U32.to_nat(window_start),ceiling_eight(span_nat)),Nat.add(U32.to_nat(window_start),Nat.add(span_nat,7n)),Nat.add(PackingContents.grid(window),7n),
      Order.offset_le(U32.to_nat(window_start),ceiling_eight(span_nat),Nat.add(span_nat,7n),DivisionOrder.product_below(Nat.add(span_nat,7n),8n,{==})),
      %Addition.associative(U32.to_nat(window_start),span_nat,7n) : {Nat.is_le(_,Nat.add(PackingContents.grid(window),7n)) == True{} : Bool}
      Order.add_le(Nat.add(U32.to_nat(window_start),span_nat),Nat.mul(window_height,out_width),7n,window_total)) :
    {Nat.is_le(Nat.add(U32.to_nat(window_start),U32.to_nat(span)),Nat.add(PackingContents.grid(window),7n)) == True{} : Bool}}
  +indices = MatrixSpec.kchannels(ci,0n,kernel)
  +window_dot = LeafSemantics.band_point(limits,window,window_contract,window_start,span,channel,column,virtual_row,column_index,window_valid,window_supported,
    window_span_bound,span_alignment,window_range,channel_inside,padded_inside,window_position,window_row,column_valid)
  +transfer = Samples.matrix_dot(problem,first_rows,window_rows,skip,channel,virtual_row,column_index,first_exact,fits,
    Positions.virtual_top(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after),
    Positions.virtual_bottom(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after,inside))
  +restored = Equal.cong(Nat,F32,value => Values.matrix_dot(indices,problem,channel,value,column_index,0.0),Nat.add(skip,virtual_row),row,
    Positions.add_sub(skip,row,ordered))
  Equal.trans(F32,Tree.read(F32,MicroLeaf.leaf_output(window,window_contract,window_start,span),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column))),
    Values.matrix_value(window,channel,virtual_row,column_index),Values.matrix_value(problem,channel,row,column_index),window_dot,
    Equal.cong(F32,F32,dot => (dot + ConvSpec.bias_at(problem,channel) : F32),Values.matrix_dot(indices,window,channel,virtual_row,column_index,0.0),
      Values.matrix_dot(indices,problem,channel,row,column_index,0.0),
      Equal.trans(F32,Values.matrix_dot(indices,window,channel,virtual_row,column_index,0.0),Values.matrix_dot(indices,problem,channel,Nat.add(skip,virtual_row),column_index,0.0),
        Values.matrix_dot(indices,problem,channel,row,column_index,0.0),transfer,restored)))

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
  {Tree.read(F32,LeafExecution.leaf_tree(~custom,activation,MicroLeaf.rows(window_problem(problem,first,rows)),padded(positions),
      Coordinates.native_inner(window_problem(problem,first,rows)),positions,window_call(problem,offset,first,rows,positions),
      PackingBounds.count(padded(positions),window_problem(problem,first,rows)),tree,
      GemmDot.reordered(window_problem(problem,first,rows),WindowProblem.band_contract(problem,weight_evidence,bias_evidence,U32.to_nat(first),U32.to_nat(rows),tree,contents,room)),
      Inputs.bias_tree(window_problem(problem,first,rows),WindowProblem.band_contract(problem,weight_evidence,bias_evidence,U32.to_nat(first),U32.to_nat(rows),tree,contents,room))),
      U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),column))) ==
    LeafExecution.activation_value(~custom,activation,
      Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)))) : F32}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  Positions.Cover{+skip,+first_exact,+fits,+rows_positive,+kernel_fits,+top,+bottom} = cover
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +cover = {Positions.Cover{skip,first_exact,fits,rows_positive,kernel_fits,top,bottom} : Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(positions)>}
  +window = window_problem(problem,first,rows)
  +contract = WindowProblem.band_contract(problem,weight_evidence,bias_evidence,U32.to_nat(first),U32.to_nat(rows),tree,contents,room)
  +span = padded(positions)
  +start = window_start(problem,offset,first)
  +positions_nat = U32.to_nat(positions)
  +span_nat = U32.to_nat(span)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +out_height = ConvSpec.axis_size(height,kernel,stride,padding)
  +out_width = ConvSpec.axis_size(image_width,kernel,stride,padding)
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +spatial_room = Order.transitive(Nat.mul(out_height,out_width),Support.padded(problem),capacity,Padded.spatial_within_padded(limits,problem,facts,fast),
    Padded.padded_bound(limits,problem,facts,fast))
  +positions_grid = Order.transitive(positions_nat,Nat.add(U32.to_nat(offset),positions_nat),Nat.mul(out_height,out_width),Samples.le_after(U32.to_nat(offset),positions_nat),grid)
  +positions_room = Order.transitive(positions_nat,Nat.mul(out_height,out_width),capacity,positions_grid,spatial_room)
  +span_exact = padded_exact(limits,positions,positions_room)
  +ordered = {%Equal.sym(Nat,span_nat,ceiling_eight(positions_nat),span_exact) : {Nat.is_le(positions_nat,_) == True{} : Bool}
    within_ceiling(positions_nat) : {Nat.is_le(positions_nat,span_nat) == True{} : Bool}}
  +span_bound = {%Equal.sym(Nat,span_nat,ceiling_eight(positions_nat),span_exact) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    WindowProblem.padded_le(positions_nat,Nat.mul(out_height,out_width),positions_grid) : {Nat.is_le(span_nat,Support.padded(problem)) == True{} : Bool}}
  +product_room = {%RowMajor.multiply_commutative(span_nat,co) : {Nat.is_le(_,capacity) == True{} : Bool}
    Order.transitive(Nat.mul(span_nat,co),Nat.mul(Support.padded(problem),co),capacity,RowMajor.mul_le(span_nat,Support.padded(problem),co,span_bound),
      output_room(limits,problem,fast)) : {Nat.is_le(Nat.mul(co,span_nat),capacity) == True{} : Bool}}
  +positions_positive = Order.le_lt_trans(0n,column,positions_nat,Order.zero_le(column),column_inside)
  +span_positive = Bounds.nat_trans(0n,positions_nat,span_nat,positions_positive,ordered)
  +co_room = Order.transitive(co,Nat.mul(co,span_nat),capacity,RowMajor.product_covers(co,span_nat,span_positive),product_room)
  +co_exact = Bounds.roundtrip(co,Machine.capacity(limits),Limits.capacity_fits(limits),co_room)
  +rows_room = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(co)),co,co_exact) : {Nat.is_le(Nat.mul(_,span_nat),capacity) == True{} : Bool}
    product_room : {Nat.is_le(Nat.mul(U32.to_nat(U32.from_nat(co)),span_nat),capacity) == True{} : Bool}}
  +channel_rows = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(co)),co,co_exact) : {Nat.is_lt(channel,_) == True{} : Bool}
    channel_inside : {Nat.is_lt(channel,U32.to_nat(U32.from_nat(co))) == True{} : Bool}}
  +leaf_output = MicroLeaf.leaf_output(window,contract,start,span)
  +index = U32.from_nat(Nat.add(Nat.mul(channel,span_nat),column))
  +row = Nat.div(Nat.add(U32.to_nat(offset),column),out_width)
  +column_index = Nat.mod(Nat.add(U32.to_nat(offset),column),out_width)
  Equal.trans(F32,
    Tree.read(F32,LeafExecution.leaf_tree(~custom,activation,MicroLeaf.rows(window),span,Coordinates.native_inner(window),positions,window_call(problem,offset,first,rows,positions),
      PackingBounds.count(span,window),tree,GemmDot.reordered(window,contract),Inputs.bias_tree(window,contract)),U32.from_nat(Nat.add(Nat.mul(channel,positions_nat),column))),
    LeafExecution.activation_value(~custom,activation,Tree.read(F32,leaf_output,index)),
    LeafExecution.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column_index)),
    LeafExecution.leaf_read(~custom,limits,activation,MicroLeaf.rows(window),span,Coordinates.native_inner(window),positions,MicroLeaf.packed_tiles(window,contract,start,span),
      GemmDot.reordered(window,contract),Inputs.bias_tree(window,contract),ordered,rows_room,channel,column,channel_rows,column_inside),
    Equal.cong(F32,F32,value => LeafExecution.activation_value(~custom,activation,value),Tree.read(F32,leaf_output,index),Values.matrix_value(problem,channel,row,column_index),
      window_value(limits,problem,weight_evidence,bias_evidence,valid,supported,offset,positions,grid,tree,first,rows,cover,contents,room,
        channel,column,channel_inside,column_inside)))

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
