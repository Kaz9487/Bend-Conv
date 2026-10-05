# Band contents of every window a partition holds: the caller input is the
# band of all rows, and a copied band reads its rows from the parent window.
import Base
import ../specification.bend as ConvSpec
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/boolean_logic.bend as Boolean
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./window_band_values.bend as WindowBandValues
import ./window_problem.bend as WindowProblem
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../inference.bend as NativeConv
import ./supported_padded_coordinates.bend as Padded
import ./microkernel_storage_execution.bend as Execution
import ./window_samples.bend as Samples
import ../../lib/tensor.bend as Tensors
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/kernel_window_copy.bend as WindowCopy
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/tensor_cursor_refinement.bend as CursorLaws
import ./storage_capacity.bend as StorageCapacity
import ../../lib/proofs/array_allocation_bounds.bend as ArrayAllocation
import ./window_positions.bend as Positions
import ./window_copy_readback.bend as CopyRead

def valid_condition(+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+index: Nat) ->
  {Boolean.condition_at(Guards.valid_conditions(problem),index) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Boolean.every_condition_holds(Guards.valid_conditions(ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias}),index,valid)

type RootContext is Data:
  RootContext{problem: ConvSpec.Problem,storage_contract: Inputs.Contract<problem>,valid: {ConvSpec.valid(problem) == True{} : Bool}}

def root_tree(context: RootContext) -> Tree.Tree<F32>:
  RootContext{problem,storage_contract,valid} = context
  Inputs.input_tree(problem,storage_contract)

def root_band(context: RootContext) -> WindowBandValues.Band:
  RootContext{+problem,storage_contract,valid} = context
  WindowProblem.band(problem,0n,Guards.height(problem))

def root_limit(context: RootContext) -> Nat:
  RootContext{+problem,storage_contract,valid} = context
  WindowProblem.count(problem,Guards.height(problem))

def root_read(+context: RootContext,+index: Nat,+inside: {Nat.is_lt(index,root_limit(context)) == True{} : Bool}) ->
  {Tree.read(F32,root_tree(context),U32.from_nat(index)) == WindowBandValues.value(root_band(context),index) : F32}:
  RootContext{+problem,+storage_contract,+valid} = context
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence} = storage_contract
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +height_positive = Boolean.every_condition_holds(Guards.valid_conditions(problem),8n,valid)
  +width_positive = Boolean.every_condition_holds(Guards.valid_conditions(problem),7n,valid)
  +input_length = Boolean.every_condition_holds(Guards.valid_conditions(problem),2n,valid)
  +plane = Nat.mul(height,width)
  +positive = Support.positive_mul(height,width,height_positive,width_positive)
  %DivisionOrder.quotient_exact(index,plane,positive) :
    {Tree.read(F32,StorageLaws.sequence_storage(F32,ConvSpec.length(input),input,input_evidence),U32.from_nat(index)) == ConvSpec.at(_,input) : F32}
  Inputs.stored_value(input,input_evidence,index,
    %Equal.sym(Nat,ConvSpec.length(input),Nat.mul(ci,plane),Guards.reflect(ConvSpec.length(input),Nat.mul(ci,plane),input_length)) :
      {Nat.is_lt(index,_) == True{} : Bool}
    inside)

def root_contents(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+valid: {ConvSpec.valid(problem) == True{} : Bool}) ->
  WindowBandValues.Contents(WindowProblem.count(problem,Guards.height(problem)),WindowProblem.band(problem,0n,Guards.height(problem)),
    Inputs.input_tree(problem,storage_contract)):
  +count = WindowProblem.count(problem,Guards.height(problem))
  WindowBandValues.listed(~RootContext,~root_tree,~root_band,~root_limit,~root_read,count,0n,RootContext{problem,storage_contract,valid},Order.reflexive(count))

def input_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +support: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Guards.input_channel_count(problem),Nat.mul(Guards.height(problem),Guards.width(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Guards.Supported{ci_bound,co_bound,height_bound,width_bound,kernel_bound,stride_bound,padding_bound,+input_bound,weight_bound,
    output_bound,vertical_bound,horizontal_bound} = support
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +input_length = Boolean.every_condition_holds(Guards.valid_conditions(problem),2n,valid)
  %Guards.reflect(ConvSpec.length(input),Nat.mul(ci,Nat.mul(height,width)),input_length) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  input_bound

def total_word(+channels: U32,+rows: U32,+width: U32) -> U32:
  (channels * rows * width : U32)

def area_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Guards.height(problem),Guards.width(problem)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +ci = Guards.input_channel_count(problem)
  +area = Nat.mul(Guards.height(problem),Guards.width(problem))
  Order.transitive(area,Nat.mul(ci,area),U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(area,ci) : {Nat.is_le(area,_) == True{} : Bool}
    RowMajor.product_covers(area,ci,valid_condition(problem,valid,10n)),
    input_room(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)),valid))

# The copied window allocates exactly channels * rows * width values.
def total_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+channels: U32,+rows: U32,+width: U32,
  +channels_exact: {U32.to_nat(channels) == Guards.input_channel_count(problem) : Nat},
  +width_exact: {U32.to_nat(width) == Guards.width(problem) : Nat},
  +count_le: {Nat.is_le(U32.to_nat(rows),Guards.height(problem)) == True{} : Bool}) ->
  {U32.to_nat(total_word(channels,rows,width)) == WindowProblem.count(problem,U32.to_nat(rows)) : Nat}:
  +ci = Guards.input_channel_count(problem)
  +image_width = Guards.width(problem)
  +count = U32.to_nat(rows)
  +plane = Nat.mul(count,image_width)
  +area = Nat.mul(Guards.height(problem),image_width)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +width_positive = valid_condition(problem,valid,7n)
  +window_small = Order.transitive(Nat.mul(ci,plane),Nat.mul(ci,area),cap,
    WindowProblem.multiply_left_le(ci,plane,area,RowMajor.mul_le(count,Guards.height(problem),image_width,count_le)),
    input_room(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)),valid))
  +pair_small = Order.transitive(Nat.mul(ci,count),Nat.mul(ci,plane),cap,
    WindowProblem.multiply_left_le(ci,count,plane,RowMajor.product_covers(count,image_width,width_positive)),window_small)
  +pair_exact = {%channels_exact : {U32.to_nat((channels * rows : U32)) == Nat.mul(_,count) : Nat}
    Arithmetic.multiply(channels,rows,capacity,Limits.capacity_fits(limits),
      %Equal.sym(Nat,U32.to_nat(channels),ci,channels_exact) : {Nat.is_le(Nat.mul(_,count),cap) == True{} : Bool}
      pair_small) : {U32.to_nat((channels * rows : U32)) == Nat.mul(ci,count) : Nat}}
  %RowMajor.assoc(ci,count,image_width) : {U32.to_nat(total_word(channels,rows,width)) == _ : Nat}
  %width_exact : {U32.to_nat(total_word(channels,rows,width)) == Nat.mul(Nat.mul(ci,count),_) : Nat}
  %pair_exact : {U32.to_nat(total_word(channels,rows,width)) == Nat.mul(_,U32.to_nat(width)) : Nat}
  Arithmetic.multiply((channels * rows : U32),width,capacity,Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat((channels * rows : U32)),Nat.mul(ci,count),pair_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(width)),cap) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(width),image_width,width_exact) : {Nat.is_le(Nat.mul(Nat.mul(ci,count),_),cap) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.mul(ci,count),image_width),Nat.mul(ci,plane),RowMajor.assoc(ci,count,image_width)) : {Nat.is_le(_,cap) == True{} : Bool}
    window_small)

# A band copied from its parent window: its rows lie inside the parent rows.
type CopyContext is Data:
  CopyContext{limits: Machine.Limits,problem: ConvSpec.Problem,valid: {ConvSpec.valid(problem) == True{} : Bool},
    supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
    parent: Tree.Tree<F32>,parent_first: U32,parent_rows: U32,first: U32,rows: U32,channels: U32,width: U32,
    channels_exact: {U32.to_nat(channels) == Guards.input_channel_count(problem) : Nat},
    width_exact: {U32.to_nat(width) == Guards.width(problem) : Nat},
    parent_contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(parent_rows)),WindowProblem.band(problem,U32.to_nat(parent_first),U32.to_nat(parent_rows)),parent),
    parent_room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(parent_rows)),U32.to_nat(Tree.capacity(F32,parent))) == True{} : Bool},
    after: {Nat.is_le(U32.to_nat(parent_first),U32.to_nat(first)) == True{} : Bool},
    within: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows))) == True{} : Bool},
    parent_fits: {Nat.is_le(Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows)),Guards.height(problem)) == True{} : Bool},
    rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}}

def copy_tree(context: CopyContext) -> Tree.Tree<F32>:
  CopyContext{limits,problem,valid,supported,parent,+parent_first,+parent_rows,+first,+rows,+channels,+width,channels_exact,width_exact,
    parent_contents,parent_room,after,within,parent_fits,rows_positive} = context
  Execution.copied_tree(channels,width,parent_first,parent_rows,first,rows,parent)

def copy_band(context: CopyContext) -> WindowBandValues.Band:
  CopyContext{limits,+problem,valid,supported,parent,parent_first,parent_rows,+first,+rows,channels,width,channels_exact,width_exact,
    parent_contents,parent_room,after,within,parent_fits,rows_positive} = context
  WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows))

def copy_limit(context: CopyContext) -> Nat:
  CopyContext{limits,+problem,valid,supported,parent,parent_first,parent_rows,first,+rows,channels,width,channels_exact,width_exact,
    parent_contents,parent_room,after,within,parent_fits,rows_positive} = context
  WindowProblem.count(problem,U32.to_nat(rows))

def count_le(+problem: ConvSpec.Problem,+parent_first: U32,+parent_rows: U32,+first: U32,+rows: U32,
  +within: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows))) == True{} : Bool},
  +parent_fits: {Nat.is_le(Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows)),Guards.height(problem)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(rows),Guards.height(problem)) == True{} : Bool}:
  Order.transitive(U32.to_nat(rows),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Guards.height(problem),Samples.le_after(U32.to_nat(first),U32.to_nat(rows)),
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows)),Guards.height(problem),within,parent_fits))

def copy_read_with(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +parent: Tree.Tree<F32>,+parent_first: U32,+parent_rows: U32,+first: U32,+rows: U32,+channels: U32,+width: U32,
  +channels_exact: {U32.to_nat(channels) == Guards.input_channel_count(problem) : Nat},
  +width_exact: {U32.to_nat(width) == Guards.width(problem) : Nat},
  +parent_contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(parent_rows)),WindowProblem.band(problem,U32.to_nat(parent_first),U32.to_nat(parent_rows)),parent),
  +parent_room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(parent_rows)),U32.to_nat(Tree.capacity(F32,parent))) == True{} : Bool},
  +after: {Nat.is_le(U32.to_nat(parent_first),U32.to_nat(first)) == True{} : Bool},
  +within: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows))) == True{} : Bool},
  +parent_fits: {Nat.is_le(Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows)),Guards.height(problem)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,WindowProblem.count(problem,U32.to_nat(rows))) == True{} : Bool},
  +allocation: ArrayAllocation.Allocation(total_word(channels,rows,width))) ->
  {Tree.read(F32,Execution.copied_tree(channels,width,parent_first,parent_rows,first,rows,parent),U32.from_nat(index)) ==
    WindowBandValues.value(WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),index) : F32}:
  Shape.Both{+depth_limit,+depth_room} = allocation
  +ci = Guards.input_channel_count(problem)
  +height = Guards.height(problem)
  +image_width = Guards.width(problem)
  +parent_start = U32.to_nat(parent_first)
  +parent_count = U32.to_nat(parent_rows)
  +start = U32.to_nat(first)
  +count = U32.to_nat(rows)
  +area = Nat.mul(height,image_width)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +fits = Limits.capacity_fits(limits)
  +input_bound = input_room(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)),valid)
  +area_bound = area_room(limits,problem,valid,supported)
  +ci_positive = valid_condition(problem,valid,10n)
  +height_positive = valid_condition(problem,valid,8n)
  +width_positive = valid_condition(problem,valid,7n)
  +area_positive = Support.positive_mul(height,image_width,height_positive,width_positive)
  +rows_le = count_le(problem,parent_first,parent_rows,first,rows,within,parent_fits)
  +parent_le = Order.transitive(parent_count,Nat.add(parent_start,parent_count),height,Samples.le_after(parent_start,parent_count),parent_fits)
  +shift_le = Order.transitive(Nat.sub(start,parent_start),start,height,Order.sub_le(start,parent_start),
    Order.transitive(start,Nat.add(start,count),height,Order.le_add(start,count),
      Order.transitive(Nat.add(start,count),Nat.add(parent_start,parent_count),height,within,parent_fits)))
  +height_small = Order.transitive(height,area,cap,RowMajor.product_covers(height,image_width,width_positive),area_bound)
  +plane = Nat.mul(count,image_width)
  +parent_plane = Nat.mul(parent_count,image_width)
  +shift = Nat.mul(Nat.sub(start,parent_start),image_width)
  +plane_small = Order.transitive(plane,area,cap,RowMajor.mul_le(count,height,image_width,rows_le),area_bound)
  +parent_small = Order.transitive(parent_plane,area,cap,RowMajor.mul_le(parent_count,height,image_width,parent_le),area_bound)
  +shift_small = Order.transitive(shift,area,cap,RowMajor.mul_le(Nat.sub(start,parent_start),height,image_width,shift_le),area_bound)
  +width_back = Equal.sym(Nat,U32.to_nat(width),image_width,width_exact)
  +count_word = (rows * width : U32)
  +stride_word = (parent_rows * width : U32)
  +shift_word = ((first - parent_first) * width : U32)
  +count_exact = {%width_exact : {U32.to_nat(count_word) == Nat.mul(count,_) : Nat}
    Arithmetic.multiply(rows,width,capacity,fits,%width_back : {Nat.is_le(Nat.mul(count,_),cap) == True{} : Bool} plane_small) :
    {U32.to_nat(count_word) == plane : Nat}}
  +stride_exact = {%width_exact : {U32.to_nat(stride_word) == Nat.mul(parent_count,_) : Nat}
    Arithmetic.multiply(parent_rows,width,capacity,fits,%width_back : {Nat.is_le(Nat.mul(parent_count,_),cap) == True{} : Bool} parent_small) :
    {U32.to_nat(stride_word) == parent_plane : Nat}}
  +difference_exact = Arithmetic.subtract(first,parent_first,capacity,fits,after,
    Order.transitive(Nat.sub(start,parent_start),height,cap,shift_le,height_small))
  +shift_exact = {%width_exact : {U32.to_nat(shift_word) == Nat.mul(Nat.sub(start,parent_start),_) : Nat}
    %difference_exact : {U32.to_nat(shift_word) == Nat.mul(_,U32.to_nat(width)) : Nat}
    Arithmetic.multiply((first - parent_first : U32),width,capacity,fits,
      %Equal.sym(Nat,U32.to_nat((first - parent_first : U32)),Nat.sub(start,parent_start),difference_exact) :
        {Nat.is_le(Nat.mul(_,U32.to_nat(width)),cap) == True{} : Bool}
      %width_back : {Nat.is_le(Nat.mul(Nat.sub(start,parent_start),_),cap) == True{} : Bool}
      shift_small) : {U32.to_nat(shift_word) == shift : Nat}}
  +plane_positive = Support.positive_mul(count,image_width,rows_positive,width_positive)
  +row = Nat.div(index,plane)
  +column = Nat.mod(index,plane)
  +row_inside = Positions.quotient_lt(index,ci,plane,plane_positive,inside)
  +column_inside = DivisionOrder.remainder_small(index,plane,plane_positive)
  +window_total = Nat.mul(ci,plane)
  +window_small = Order.transitive(window_total,Nat.mul(ci,area),cap,WindowProblem.multiply_left_le(ci,plane,area,RowMajor.mul_le(count,height,image_width,rows_le)),input_bound)
  +total_value = total_exact(limits,problem,valid,supported,channels,rows,width,channels_exact,width_exact,rows_le)
  +total = total_word(channels,rows,width)
  +total_positive = {%Equal.sym(Nat,U32.to_nat(total),window_total,total_value) : {Nat.is_lt(0n,_) == True{} : Bool}
    Support.positive_mul(ci,plane,ci_positive,plane_positive) : {Nat.is_lt(0n,U32.to_nat(total)) == True{} : Bool}}
  +depth = Tensors.depth_for(total)
  +depth_same = Depth.positive_depth(total,total_positive)
  +depth_ok = {%Equal.sym(Nat,Tensors.depth_for(total),Storage.allocation_depth(total),depth_same) : {Nat.is_le(_,30n) == True{} : Bool}
    depth_limit : {Nat.is_le(depth,30n) == True{} : Bool}}
  +fresh = Tree.fresh(F32,depth,0.0)
  +balanced = Shape.fresh(F32,depth,0.0)
  +capacity_exact = StorageCapacity.exact(depth,fresh,balanced)
  +capacity_room = {%Equal.sym(U32,Tree.capacity(F32,fresh),U32.shln(1,depth),capacity_exact) : {Nat.is_le(window_total,U32.to_nat(_)) == True{} : Bool}
    %Equal.sym(Nat,Tensors.depth_for(total),Storage.allocation_depth(total),depth_same) : {Nat.is_le(window_total,U32.to_nat(U32.shln(1,_))) == True{} : Bool}
    %total_value : {Nat.is_le(_,U32.to_nat(U32.shln(1,Storage.allocation_depth(total)))) == True{} : Bool}
    depth_room : {Nat.is_le(window_total,U32.to_nat(Tree.capacity(F32,fresh))) == True{} : Bool}}
  +index_small = Order.transitive(index,window_total,cap,Bounds.strict_le(index,window_total,inside),window_small)
  +index_exact = Bounds.roundtrip(index,capacity,fits,index_small)
  +rows_room = {%Positions.sub_after(start,parent_start,count,after) : {Nat.is_le(_,parent_count) == True{} : Bool}
    %CursorLaws.subtract_prefix(parent_start,parent_count) : {Nat.is_le(Nat.sub(Nat.add(start,count),parent_start),_) == True{} : Bool}
    DivisionOrder.subtract_le(Nat.add(start,count),Nat.add(parent_start,parent_count),parent_start,within) :
    {Nat.is_le(Nat.add(Nat.sub(start,parent_start),count),parent_count) == True{} : Bool}}
  +offset_room = {%Equal.sym(Bool,Nat.is_lt(Nat.add(shift,column),Nat.add(shift,plane)),Nat.is_lt(column,plane),Samples.lt_offset(shift,column,plane)) :
      {_ == True{} : Bool}
    column_inside : {Nat.is_lt(Nat.add(shift,column),Nat.add(shift,plane)) == True{} : Bool}}
  +remainder_small = Samples.lt_le(Nat.add(shift,column),Nat.add(shift,plane),parent_plane,offset_room,
    %RowMajor.distrib(Nat.sub(start,parent_start),count,image_width) : {Nat.is_le(_,parent_plane) == True{} : Bool}
    RowMajor.mul_le(Nat.add(Nat.sub(start,parent_start),count),parent_count,image_width,rows_room))
  +parent_positive = Order.le_lt_trans(0n,Nat.add(shift,column),parent_plane,Order.zero_le(Nat.add(shift,column)),remainder_small)
  +source = Nat.add(Nat.add(Nat.mul(row,parent_plane),shift),column)
  +regrouped = Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(row,parent_plane),shift),column),Nat.add(Nat.mul(row,parent_plane),Nat.add(shift,column)),
    Addition.associative(Nat.mul(row,parent_plane),shift,column))
  +source_inside = {%regrouped : {Nat.is_lt(_,Nat.mul(ci,parent_plane)) == True{} : Bool}
    Samples.lt_le(Nat.add(Nat.mul(row,parent_plane),Nat.add(shift,column)),Nat.mul(1n+row,parent_plane),Nat.mul(ci,parent_plane),
      %Order.commutative(Nat.mul(row,parent_plane),parent_plane) : {Nat.is_lt(Nat.add(Nat.mul(row,parent_plane),Nat.add(shift,column)),_) == True{} : Bool}
      %Equal.sym(Bool,Nat.is_lt(Nat.add(Nat.mul(row,parent_plane),Nat.add(shift,column)),Nat.add(Nat.mul(row,parent_plane),parent_plane)),
        Nat.is_lt(Nat.add(shift,column),parent_plane),Samples.lt_offset(Nat.mul(row,parent_plane),Nat.add(shift,column),parent_plane)) : {_ == True{} : Bool}
      remainder_small,
      RowMajor.mul_le(1n+row,ci,parent_plane,Order.strict_follow(row,ci,row_inside))) :
    {Nat.is_lt(source,Nat.mul(ci,parent_plane)) == True{} : Bool}}
  +source_small = Order.transitive(source,Nat.mul(ci,parent_plane),cap,Bounds.strict_le(source,Nat.mul(ci,parent_plane),source_inside),
    Order.transitive(Nat.mul(ci,parent_plane),Nat.mul(ci,area),cap,WindowProblem.multiply_left_le(ci,parent_plane,area,RowMajor.mul_le(parent_count,height,image_width,parent_le)),
      input_bound))
  +channels_back = Equal.sym(Nat,U32.to_nat(channels),ci,channels_exact)
  +count_back = Equal.sym(Nat,U32.to_nat(count_word),plane,count_exact)
  +planes_read = CopyRead.inside_planes(U32.to_nat(channels),Nat.is_eq(0n,row),0,count_word,stride_word,shift_word,parent,fresh,depth,U32.from_nat(index),
    capacity,fits,balanced,depth_ok,
    %channels_back : {Nat.is_le(_,cap) == True{} : Bool}
    Order.transitive(ci,Nat.mul(ci,area),cap,RowMajor.product_covers(ci,area,area_positive),input_bound),
    %channels_back : {Nat.is_le(Nat.mul(_,U32.to_nat(count_word)),U32.to_nat(Tree.capacity(F32,fresh))) == True{} : Bool}
    %count_back : {Nat.is_le(Nat.mul(ci,_),U32.to_nat(Tree.capacity(F32,fresh))) == True{} : Bool}
    capacity_room,
    %channels_back : {Nat.is_le(Nat.mul(_,U32.to_nat(count_word)),cap) == True{} : Bool}
    %count_back : {Nat.is_le(Nat.mul(ci,_),cap) == True{} : Bool}
    window_small,
    U32Division.nat_lt_native(U32.from_nat(index),Tree.capacity(F32,fresh),
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,index_exact) : {Nat.is_lt(_,U32.to_nat(Tree.capacity(F32,fresh))) == True{} : Bool}
      Samples.lt_le(index,window_total,U32.to_nat(Tree.capacity(F32,fresh)),inside,capacity_room)),
    row,column,{==},
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,index_exact) : {_ == Nat.add(Nat.mul(row,U32.to_nat(count_word)),column) : Nat}
    %count_back : {index == Nat.add(Nat.mul(row,_),column) : Nat}
    DivisionOrder.quotient_exact(index,plane,plane_positive),
    %count_back : {Nat.is_lt(column,_) == True{} : Bool}
    column_inside,
    Order.zero_le(row),
    %channels_back : {Nat.is_lt(row,_) == True{} : Bool}
    row_inside,
    %Equal.sym(Nat,U32.to_nat(stride_word),parent_plane,stride_exact) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(row,_),U32.to_nat(shift_word)),column),cap) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(shift_word),shift,shift_exact) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(row,parent_plane),_),column),cap) == True{} : Bool}
    source_small)
  +parent_band = WindowProblem.band(problem,parent_start,parent_count)
  +band = WindowProblem.band(problem,start,count)
  +parent_read = WindowBandValues.read(WindowProblem.count(problem,parent_count),parent_band,parent,parent_contents,parent_room,source,source_inside)
  +split = Division.quotient_remainder(row,Nat.add(shift,column),parent_plane,parent_positive,remainder_small)
  +same_source = {%regrouped : {WindowBandValues.source(parent_band,_) == WindowBandValues.source(band,index) : Nat}
    %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(row,parent_plane),Nat.add(shift,column)),parent_plane),(row,Nat.add(shift,column)),split) :
      {Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),area),Nat.add(Nat.mul(parent_start,image_width),Pair.snd(Nat,Nat,_))) == WindowBandValues.source(band,index) : Nat}
    %Addition.associative(Nat.mul(parent_start,image_width),shift,column) :
      {Nat.add(Nat.mul(row,area),_) == WindowBandValues.source(band,index) : Nat}
    %RowMajor.distrib(parent_start,Nat.sub(start,parent_start),image_width) :
      {Nat.add(Nat.mul(row,area),Nat.add(_,column)) == WindowBandValues.source(band,index) : Nat}
    %Equal.sym(Nat,Nat.add(parent_start,Nat.sub(start,parent_start)),start,Positions.add_sub(parent_start,start,after)) :
      {Nat.add(Nat.mul(row,area),Nat.add(Nat.mul(_,image_width),column)) == WindowBandValues.source(band,index) : Nat}
    {==} : {WindowBandValues.source(parent_band,source) == WindowBandValues.source(band,index) : Nat}}
  +planes_fresh = Writes.pure(F32,WindowCopy.writes(U32.to_nat(channels),0,count_word,stride_word,shift_word,count_word,parent),fresh)
  +laid_out = CopyRead.planes(U32.to_nat(channels),0,count_word,stride_word,shift_word,count_word,parent,fresh)
  Equal.trans(F32,Tree.read(F32,planes_fresh,U32.from_nat(index)),Tree.read(F32,parent,U32.from_nat(source)),WindowBandValues.value(band,index),
    Equal.trans(F32,Tree.read(F32,planes_fresh,U32.from_nat(index)),Tree.read(F32,laid_out,U32.from_nat(index)),Tree.read(F32,parent,U32.from_nat(source)),
      Equal.cong(Tree.Tree<F32>,F32,tree => Tree.read(F32,tree,U32.from_nat(index)),planes_fresh,laid_out,
        CopyRead.planes_exact(U32.to_nat(channels),0,count_word,stride_word,shift_word,count_word,parent,fresh)),
      %stride_exact : {Tree.read(F32,laid_out,U32.from_nat(index)) == Tree.read(F32,parent,U32.from_nat(Nat.add(Nat.add(Nat.mul(row,_),shift),column))) : F32}
      %shift_exact : {Tree.read(F32,laid_out,U32.from_nat(index)) == Tree.read(F32,parent,U32.from_nat(Nat.add(Nat.add(Nat.mul(row,U32.to_nat(stride_word)),_),column))) : F32}
      planes_read),
    Equal.trans(F32,Tree.read(F32,parent,U32.from_nat(source)),WindowBandValues.value(parent_band,source),WindowBandValues.value(band,index),parent_read,
      Equal.cong(Nat,F32,position => ConvSpec.at(position,Guards.input(problem)),WindowBandValues.source(parent_band,source),WindowBandValues.source(band,index),same_source)))

def copy_read(+context: CopyContext,+index: Nat,+inside: {Nat.is_lt(index,copy_limit(context)) == True{} : Bool}) ->
  {Tree.read(F32,copy_tree(context),U32.from_nat(index)) == WindowBandValues.value(copy_band(context),index) : F32}:
  CopyContext{+limits,+problem,+valid,+supported,+parent,+parent_first,+parent_rows,+first,+rows,+channels,+width,+channels_exact,+width_exact,
    +parent_contents,+parent_room,+after,+within,+parent_fits,+rows_positive} = context
  +ci = Guards.input_channel_count(problem)
  +plane = Nat.mul(U32.to_nat(rows),Guards.width(problem))
  +window_total = Nat.mul(ci,plane)
  +cap = U32.to_nat(Machine.capacity(limits))
  +rows_le = count_le(problem,parent_first,parent_rows,first,rows,within,parent_fits)
  +total_value = total_exact(limits,problem,valid,supported,channels,rows,width,channels_exact,width_exact,rows_le)
  +total = total_word(channels,rows,width)
  +width_positive = valid_condition(problem,valid,7n)
  +ci_positive = valid_condition(problem,valid,10n)
  +input_bound = input_room(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)),valid)
  +positive = {%Equal.sym(Nat,U32.to_nat(total),window_total,total_value) : {Nat.is_lt(0n,_) == True{} : Bool}
    Support.positive_mul(ci,plane,ci_positive,Support.positive_mul(U32.to_nat(rows),Guards.width(problem),rows_positive,width_positive)) :
    {Nat.is_lt(0n,U32.to_nat(total)) == True{} : Bool}}
  +bound = {%Equal.sym(Nat,U32.to_nat(total),window_total,total_value) : {Nat.is_le(_,cap) == True{} : Bool}
    Order.transitive(window_total,Nat.mul(ci,Nat.mul(Guards.height(problem),Guards.width(problem))),cap,
      WindowProblem.multiply_left_le(ci,plane,Nat.mul(Guards.height(problem),Guards.width(problem)),
        RowMajor.mul_le(U32.to_nat(rows),Guards.height(problem),Guards.width(problem),rows_le)),input_bound) :
    {Nat.is_le(U32.to_nat(total),cap) == True{} : Bool}}
  copy_read_with(limits,problem,valid,supported,parent,parent_first,parent_rows,first,rows,channels,width,channels_exact,width_exact,
    parent_contents,parent_room,after,within,parent_fits,rows_positive,index,inside,
    ArrayAllocation.allocated(total,Machine.capacity(limits),Limits.capacity_fits(limits),Limits.capacity_allocatable(limits),positive,bound))

def copied_contents(+context: CopyContext) -> WindowBandValues.Contents(copy_limit(context),copy_band(context),copy_tree(context)):
  WindowBandValues.listed(~CopyContext,~copy_tree,~copy_band,~copy_limit,~copy_read,copy_limit(context),0n,context,Order.reflexive(copy_limit(context)))

def copied_room_with(+problem: ConvSpec.Problem,+channels: U32,+rows: U32,+width: U32,+parent_first: U32,+parent_rows: U32,+first: U32,+parent: Tree.Tree<F32>,
  +total_value: {U32.to_nat(total_word(channels,rows,width)) == WindowProblem.count(problem,U32.to_nat(rows)) : Nat},
  +total_positive: {Nat.is_lt(0n,U32.to_nat(total_word(channels,rows,width))) == True{} : Bool},
  +allocation: ArrayAllocation.Allocation(total_word(channels,rows,width))) ->
  {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,Execution.copied_tree(channels,width,parent_first,parent_rows,first,rows,parent)))) == True{} : Bool}:
  Shape.Both{+depth_limit,+depth_room} = allocation
  +total = total_word(channels,rows,width)
  +depth = Tensors.depth_for(total)
  +depth_same = Depth.positive_depth(total,total_positive)
  +fresh = Tree.fresh(F32,depth,0.0)
  +balanced = Shape.fresh(F32,depth,0.0)
  +writes = WindowCopy.writes(U32.to_nat(channels),0,(rows * width : U32),(parent_rows * width : U32),((first - parent_first) * width : U32),(rows * width : U32),parent)
  %Equal.sym(U32,Tree.capacity(F32,Writes.pure(F32,writes,fresh)),Tree.capacity(F32,fresh),Writes.writes_preserve_capacity(F32,writes,fresh,depth,balanced)) :
    {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(U32,Tree.capacity(F32,fresh),U32.shln(1,depth),StorageCapacity.exact(depth,fresh,balanced)) :
    {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(Nat,Tensors.depth_for(total),Storage.allocation_depth(total),depth_same) :
    {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(U32.shln(1,_))) == True{} : Bool}
  %total_value : {Nat.is_le(_,U32.to_nat(U32.shln(1,Storage.allocation_depth(total)))) == True{} : Bool}
  depth_room

# The copied window has room for its band.
def copied_room(+context: CopyContext) -> {Nat.is_le(copy_limit(context),U32.to_nat(Tree.capacity(F32,copy_tree(context)))) == True{} : Bool}:
  CopyContext{+limits,+problem,+valid,+supported,+parent,+parent_first,+parent_rows,+first,+rows,+channels,+width,+channels_exact,+width_exact,
    parent_contents,parent_room,after,+within,+parent_fits,+rows_positive} = context
  +ci = Guards.input_channel_count(problem)
  +plane = Nat.mul(U32.to_nat(rows),Guards.width(problem))
  +window_total = Nat.mul(ci,plane)
  +cap = U32.to_nat(Machine.capacity(limits))
  +rows_le = count_le(problem,parent_first,parent_rows,first,rows,within,parent_fits)
  +total_value = total_exact(limits,problem,valid,supported,channels,rows,width,channels_exact,width_exact,rows_le)
  +total = total_word(channels,rows,width)
  +input_bound = input_room(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)),valid)
  +positive = {%Equal.sym(Nat,U32.to_nat(total),window_total,total_value) : {Nat.is_lt(0n,_) == True{} : Bool}
    Support.positive_mul(ci,plane,valid_condition(problem,valid,10n),Support.positive_mul(U32.to_nat(rows),Guards.width(problem),rows_positive,valid_condition(problem,valid,7n))) :
    {Nat.is_lt(0n,U32.to_nat(total)) == True{} : Bool}}
  +bound = {%Equal.sym(Nat,U32.to_nat(total),window_total,total_value) : {Nat.is_le(_,cap) == True{} : Bool}
    Order.transitive(window_total,Nat.mul(ci,Nat.mul(Guards.height(problem),Guards.width(problem))),cap,
      WindowProblem.multiply_left_le(ci,plane,Nat.mul(Guards.height(problem),Guards.width(problem)),
        RowMajor.mul_le(U32.to_nat(rows),Guards.height(problem),Guards.width(problem),rows_le)),input_bound) :
    {Nat.is_le(U32.to_nat(total),cap) == True{} : Bool}}
  copied_room_with(problem,channels,rows,width,parent_first,parent_rows,first,parent,total_value,positive,
    ArrayAllocation.allocated(total,Machine.capacity(limits),Limits.capacity_fits(limits),Limits.capacity_allocatable(limits),positive,bound))
