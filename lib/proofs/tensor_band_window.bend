# Band windows: clipping arithmetic, halo ranges and their bounds, the rows of
# an assembled window, and the mapping to output rows.
import Base
import ./nat_order.bend as Order
import ../tensor_band_window.bend as BandWindow
import ./tensor_band_gather.bend as BandGather
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_range.bend as BandRange
import ./u32_extrema.bend as U32Extrema
import ../tensor_band.bend as Bands
import ../machine_limits.bend as Machine
import ./nat_extrema.bend as Extrema
import ./storage_shape.bend as Shape
import ./tensor_band_dense.bend as BandDense
import ./tensor_band_halo.bend as BandHalo
import ./machine_limit_bounds.bend as Limits
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./tensor_band_shape_refinement.bend as BandShape
import ./u32_division.bend as U32Division
import ./natural_addition.bend as Addition
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_model.bend as Model
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding
import ./nat_ceiling.bend as Ceiling
import ./nat_interval.bend as Interval
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_repartition.bend as BandRepartition
import ./tensor_band_values.bend as BandValues
import ./boolean_logic.bend as Boolean
import ./row_major_coordinates.bend as RowCoordinates
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_tree_model.bend as Tree
import ./tensor_band_count.bend as BandCount
import ./tensor_cursor_refinement.bend as CursorLaws
import ./u32_full_range.bend as FullRange
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition

# A valid sample in an output row interval lies in the required input interval.
# Splitting that interval around an arbitrary core covers it, including empty
# cores. Native arithmetic and the reader's owner selection are separate layers.
def Contains(+first: Nat,end: Nat,+row: Nat) -> Data:
  Shape.Both<{Nat.is_le(first,row) == True{} : Bool},{Nat.is_lt(row,end) == True{} : Bool}>

type Covered<-core_first: Nat,-core_end: Nat,-needed_first: Nat,-needed_end: Nat,-row: Nat> is Data:
  Core{bounds: Contains(core_first,core_end,row)}
  Above{bounds: Contains(needed_first,Nat.min(core_first,needed_end),row)}
  Below{bounds: Contains(Nat.max(core_end,needed_first),needed_end,row)}

def partition_selected(before: Bool,within: Bool,+core_first: Nat,+core_end: Nat,+needed_first: Nat,+needed_end: Nat,+row: Nat,
  first_choice: {Nat.is_lt(row,core_first) == before : Bool},end_choice: {Nat.is_lt(row,core_end) == within : Bool},bounds: Contains(needed_first,needed_end,row)) ->
  Covered<core_first,core_end,needed_first,needed_end,row>:
  match before within bounds:
    case True{} within Shape.Both{lower,upper}: Above{Shape.Both{lower,Extrema.minimum_strict_lower(core_first,needed_end,row,first_choice,upper)}}
    case False{} True{} bounds: Core{Shape.Both{Order.ge_true(row,core_first,Order.comparison_not_less(Nat.cmp(row,core_first),first_choice)),end_choice}}
    case False{} False{} Shape.Both{lower,upper}: Below{Shape.Both{Extrema.maximum_upper(core_end,needed_first,row,Order.ge_true(row,core_end,Order.comparison_not_less(Nat.cmp(row,core_end),end_choice)),lower),upper}}

def partition(+core_first: Nat,+core_end: Nat,+needed_first: Nat,+needed_end: Nat,+row: Nat,bounds: Contains(needed_first,needed_end,row)) ->
  Covered<core_first,core_end,needed_first,needed_end,row>:
  partition_selected(Nat.is_lt(row,core_first),Nat.is_lt(row,core_end),core_first,core_end,needed_first,needed_end,row,{==},{==},bounds)

def before_last(+row: Nat,+end: Nat,inside: {Nat.is_lt(row,end) == True{} : Bool}) -> {Nat.is_le(row,Nat.sub(end,1n)) == True{} : Bool}:
  match row end:
    case 0n end: Order.zero_le(Nat.sub(end,1n))
    case 1n+previous 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(inside))
    case 1n+ +previous 1n+ +last:
      %Equal.sym(Nat,Nat.sub(last,0n),last,Order.subtract_zero(last)) : {Nat.is_le(row,_) == True{} : Bool}
      Order.strict_follow(previous,last,inside)

def needed_first(height: Nat,first: Nat,stride: Nat,padding: Nat) -> Nat:
  Nat.min(height,Nat.sub(Nat.mul(first,stride),padding))

def needed_end(height: Nat,end: Nat,stride: Nat,kernel: Nat,padding: Nat) -> Nat:
  Nat.min(height,Nat.sub(Nat.add(Nat.mul(Nat.sub(end,1n),stride),kernel),padding))

def sample(+height: Nat,+first: Nat,+end: Nat,+stride: Nat,+kernel: Nat,+padding: Nat,+output_row: Nat,+kernel_row: Nat,
  output: Contains(first,end,output_row),kernel_inside: {Nat.is_lt(kernel_row,kernel) == True{} : Bool},
  unpadded: {Nat.is_le(padding,Nat.add(Nat.mul(output_row,stride),kernel_row)) == True{} : Bool},
  image_inside: {Nat.is_lt(Nat.sub(Nat.add(Nat.mul(output_row,stride),kernel_row),padding),height) == True{} : Bool}) ->
  Contains(needed_first(height,first,stride,padding),needed_end(height,end,stride,kernel,padding),Nat.sub(Nat.add(Nat.mul(output_row,stride),kernel_row),padding)):
  Shape.Both{first_before,end_after} = output
  +coordinate = Nat.add(Nat.mul(output_row,stride),kernel_row)
  +last = Nat.sub(end,1n)
  +limit = Nat.add(Nat.mul(last,stride),kernel)
  +lower = DivisionOrder.subtract_le(Nat.mul(first,stride),coordinate,padding,
    Order.transitive(Nat.mul(first,stride),Nat.mul(output_row,stride),coordinate,RowMajor.mul_le(first,output_row,stride,first_before),Order.le_add(Nat.mul(output_row,stride),kernel_row)))
  +upper = Order.le_lt_trans(coordinate,Nat.add(Nat.mul(last,stride),kernel_row),limit,
    Order.add_le(Nat.mul(output_row,stride),Nat.mul(last,stride),kernel_row,RowMajor.mul_le(output_row,last,stride,before_last(output_row,end,end_after))),
    Order.offset_lt(Nat.mul(last,stride),kernel_row,kernel,kernel_inside))
  Shape.Both{Order.transitive(needed_first(height,first,stride,padding),Nat.sub(Nat.mul(first,stride),padding),Nat.sub(coordinate,padding),Extrema.minimum_right(height,Nat.sub(Nat.mul(first,stride),padding)),lower),
    Extrema.minimum_strict_lower(height,Nat.sub(limit,padding),Nat.sub(coordinate,padding),image_inside,Interval.shifted_inside(padding,limit,coordinate,unpadded,upper))}

# Decode the exact clipping and ceiling operations used by band windows.
def maximum_subtract(+value: Nat,+amount: Nat) -> {Nat.sub(Nat.max(value,amount),amount) == Nat.sub(value,amount) : Nat}:
  match value amount:
    case 0n 0n: {==}
    case 0n 1n+ +rest: Equal.trans(Nat,Nat.sub(rest,rest),Nat.sub(0n,rest),0n,maximum_subtract(0n,rest),Order.zero_subtract(rest))
    case 1n+value 0n: {==}
    case 1n+value 1n+amount: maximum_subtract(value,amount)

# Clipped subtraction never exceeds its value, whatever the amount.
def clipped_below(+value: U32,+amount: U32) -> {Nat.is_le(U32.to_nat(BandWindow.subtract_clipped(value,amount)),U32.to_nat(value)) == True{} : Bool}:
  +maximum = U32.max(value,amount)
  +decoded = U32Extrema.maximum(value,amount)
  +ordered = {%Equal.sym(Nat,U32.to_nat(maximum),Nat.max(U32.to_nat(value),U32.to_nat(amount)),decoded) : {Nat.is_le(U32.to_nat(amount),_) == True{} : Bool}
    Extrema.maximum_right(U32.to_nat(value),U32.to_nat(amount)) : {Nat.is_le(U32.to_nat(amount),U32.to_nat(maximum)) == True{} : Bool}}
  %Equal.sym(Nat,U32.to_nat(BandWindow.subtract_clipped(value,amount)),Nat.sub(U32.to_nat(maximum),U32.to_nat(amount)),
    FullRange.subtract(maximum,amount,maximum,ordered,Order.sub_le(U32.to_nat(maximum),U32.to_nat(amount)))) :
    {Nat.is_le(_,U32.to_nat(value)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(maximum),Nat.max(U32.to_nat(value),U32.to_nat(amount)),decoded) :
    {Nat.is_le(Nat.sub(_,U32.to_nat(amount)),U32.to_nat(value)) == True{} : Bool}
  %Equal.sym(Nat,Nat.sub(Nat.max(U32.to_nat(value),U32.to_nat(amount)),U32.to_nat(amount)),Nat.sub(U32.to_nat(value),U32.to_nat(amount)),
    maximum_subtract(U32.to_nat(value),U32.to_nat(amount))) :
    {Nat.is_le(_,U32.to_nat(value)) == True{} : Bool}
  Order.sub_le(U32.to_nat(value),U32.to_nat(amount))

def subtract_clipped(+value: U32,+amount: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  value_room: {Nat.is_le(U32.to_nat(value),U32.to_nat(capacity)) == True{} : Bool},amount_room: {Nat.is_le(U32.to_nat(amount),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(BandWindow.subtract_clipped(value,amount)) == Nat.sub(U32.to_nat(value),U32.to_nat(amount)) : Nat}:
  +maximum = U32.max(value,amount)
  +decoded = U32Extrema.maximum(value,amount)
  +room = {
    %Equal.sym(Nat,U32.to_nat(maximum),Nat.max(U32.to_nat(value),U32.to_nat(amount)),decoded) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Extrema.maximum_upper(U32.to_nat(value),U32.to_nat(amount),U32.to_nat(capacity),value_room,amount_room) : {Nat.is_le(U32.to_nat(maximum),U32.to_nat(capacity)) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat(BandWindow.subtract_clipped(value,amount)),Nat.sub(U32.to_nat(maximum),U32.to_nat(amount)),Nat.sub(U32.to_nat(value),U32.to_nat(amount)),
    Arithmetic.subtract(maximum,amount,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(maximum),Nat.max(U32.to_nat(value),U32.to_nat(amount)),decoded) : {Nat.is_le(U32.to_nat(amount),_) == True{} : Bool}
      Extrema.maximum_right(U32.to_nat(value),U32.to_nat(amount)),
      Order.transitive(Nat.sub(U32.to_nat(maximum),U32.to_nat(amount)),U32.to_nat(maximum),U32.to_nat(capacity),Order.sub_le(U32.to_nat(maximum),U32.to_nat(amount)),room)),
    %Equal.sym(Nat,U32.to_nat(maximum),Nat.max(U32.to_nat(value),U32.to_nat(amount)),decoded) : {Nat.sub(_,U32.to_nat(amount)) == Nat.sub(U32.to_nat(value),U32.to_nat(amount)) : Nat}
    maximum_subtract(U32.to_nat(value),U32.to_nat(amount)))

def rounded_native(+quotient: U32,+remainder: U32) ->
  {Nat.add(U32.to_nat(quotient),U32.to_nat(Bool.pick(U32,U32.is_zero(remainder),0,1))) == Ceiling.rounded(U32.to_nat(quotient),U32.to_nat(remainder)) : Nat}:
  %Equal.sym(Nat,U32.to_nat(Bool.pick(U32,U32.is_zero(remainder),0,1)),Bool.pick(Nat,U32.is_zero(remainder),0n,1n),U32Extrema.selected(U32.is_zero(remainder),0,1)) :
    {Nat.add(U32.to_nat(quotient),_) == Ceiling.rounded(U32.to_nat(quotient),U32.to_nat(remainder)) : Nat}
  %Equal.sym(Bool,U32.is_zero(remainder),Nat.is_eq(U32.to_nat(remainder),0n),Comparison.eq(remainder,0)) :
    {Nat.add(U32.to_nat(quotient),Bool.pick(Nat,_,0n,1n)) == Ceiling.rounded(U32.to_nat(quotient),U32.to_nat(remainder)) : Nat}
  Equal.sym(Nat,Ceiling.rounded(U32.to_nat(quotient),U32.to_nat(remainder)),Nat.add(U32.to_nat(quotient),Bool.pick(Nat,Nat.is_eq(U32.to_nat(remainder),0n),0n,1n)),Ceiling.rounded_choice(U32.to_nat(quotient),U32.to_nat(remainder)))

def rounded_division(+value: U32,+divisor: U32,+division_capacity: U32,+division_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(divisor)),U32.to_nat(division_capacity)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(divisor)) == True{} : Bool}) ->
  {Nat.add(U32.to_nat((value / divisor : U32)),U32.to_nat(Bool.pick(U32,U32.is_zero((value % divisor : U32)),0,1))) == Ceiling.ceiling(U32.to_nat(value),U32.to_nat(divisor)) : Nat}:
  Equal.trans(Nat,Nat.add(U32.to_nat((value / divisor : U32)),U32.to_nat(Bool.pick(U32,U32.is_zero((value % divisor : U32)),0,1))),
    Ceiling.rounded(U32.to_nat((value / divisor : U32)),U32.to_nat((value % divisor : U32))),Ceiling.ceiling(U32.to_nat(value),U32.to_nat(divisor)),
    rounded_native((value / divisor : U32),(value % divisor : U32)),
    %U32Division.divide(value,divisor,division_capacity,division_ok,divisor_room,positive) : {Ceiling.rounded(U32.to_nat((value / divisor : U32)),U32.to_nat((value % divisor : U32))) == Ceiling.rounded(_,Nat.mod(U32.to_nat(value),U32.to_nat(divisor))) : Nat}
    %U32Division.modulo(value,divisor,division_capacity,division_ok,divisor_room,positive) : {Ceiling.rounded(U32.to_nat((value / divisor : U32)),U32.to_nat((value % divisor : U32))) == Ceiling.rounded(U32.to_nat((value / divisor : U32)),_) : Nat}
    {==})

def ceiling(+value: U32,+divisor: U32,+capacity: U32,+division_capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +division_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},+divisor_room: {Nat.is_le(Nat.double(U32.to_nat(divisor)),U32.to_nat(division_capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(divisor)) == True{} : Bool},value_room: {Nat.is_le(U32.to_nat(value),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(BandWindow.ceiling(value,divisor)) == Ceiling.ceiling(U32.to_nat(value),U32.to_nat(divisor)) : Nat}:
  +quotient = (value / divisor : U32)
  +increment = Bool.pick(U32,U32.is_zero((value % divisor : U32)),0,1)
  +decoded = rounded_division(value,divisor,division_capacity,division_ok,divisor_room,positive)
  Equal.trans(Nat,U32.to_nat(BandWindow.ceiling(value,divisor)),Nat.add(U32.to_nat(quotient),U32.to_nat(increment)),Ceiling.ceiling(U32.to_nat(value),U32.to_nat(divisor)),
    Arithmetic.add(quotient,increment,capacity,capacity_ok,
      %Equal.sym(Nat,Nat.add(U32.to_nat(quotient),U32.to_nat(increment)),Ceiling.ceiling(U32.to_nat(value),U32.to_nat(divisor)),decoded) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
      Order.transitive(Ceiling.ceiling(U32.to_nat(value),U32.to_nat(divisor)),U32.to_nat(value),U32.to_nat(capacity),Ceiling.bounded(U32.to_nat(value),U32.to_nat(divisor),positive),value_room)),decoded)

# A clipped pair of endpoints in one image produces an allocatable halo range.
# The window calculation supplies endpoint bounds; range readback is reused.
def end_exact(+first: U32,+end: U32,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(capacity)) == True{} : Bool},end_room: {Nat.is_le(U32.to_nat(end),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(first),U32.to_nat(BandWindow.subtract_clipped(end,first))) == Nat.max(U32.to_nat(first),U32.to_nat(end)) : Nat}:
  Equal.trans(Nat,Nat.add(U32.to_nat(first),U32.to_nat(BandWindow.subtract_clipped(end,first))),Nat.add(U32.to_nat(first),Nat.sub(U32.to_nat(end),U32.to_nat(first))),Nat.max(U32.to_nat(first),U32.to_nat(end)),
    Equal.cong(Nat,Nat,count => Nat.add(U32.to_nat(first),count),U32.to_nat(BandWindow.subtract_clipped(end,first)),Nat.sub(U32.to_nat(end),U32.to_nat(first)),
      subtract_clipped(end,first,capacity,capacity_ok,end_room,first_room)),Extrema.clipped_end(U32.to_nat(first),U32.to_nat(end)))

def clipped(+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,+image: Nat,+first: U32,+end: U32,
  +domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,+image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},
  lower: {Nat.is_le(Nat.mul(image,U32.to_nat(height)),U32.to_nat(first)) == True{} : Bool},
  +first_inside: {Nat.is_le(U32.to_nat(first),Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat(height))) == True{} : Bool},
  +end_inside: {Nat.is_le(U32.to_nat(end),Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat(height))) == True{} : Bool}) ->
  BandDense.LeafFacts<Bands.Geometry{images,channels,height,width},first,BandWindow.subtract_clipped(end,first),capacity,Machine.half_word(limits)>:
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +row_total = Nat.mul(U32.to_nat(images),U32.to_nat(height))
  +image_end = Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat(height))
  +image_room = {
    %Order.commutative(U32.to_nat(height),Nat.mul(image,U32.to_nat(height))) : {Nat.is_le(_,row_total) == True{} : Bool}
    RowMajor.mul_le(1n+image,U32.to_nat(images),U32.to_nat(height),Order.strict_follow(image,U32.to_nat(images),image_inside)) : {Nat.is_le(image_end,row_total) == True{} : Bool}}
  +row_room = BandGather.rows_room(geometry,limits,capacity,domain)
  +room = Order.transitive(image_end,row_total,U32.to_nat(Machine.capacity(limits)),image_room,row_room)
  +exact = end_exact(first,end,Machine.capacity(limits),Limits.capacity_fits(limits),
    Order.transitive(U32.to_nat(first),image_end,U32.to_nat(Machine.capacity(limits)),first_inside,room),
    Order.transitive(U32.to_nat(end),image_end,U32.to_nat(Machine.capacity(limits)),end_inside,room))
  +within = {
    %Equal.sym(Nat,Nat.add(U32.to_nat(first),U32.to_nat(BandWindow.subtract_clipped(end,first))),Nat.max(U32.to_nat(first),U32.to_nat(end)),exact) : {Nat.is_le(_,image_end) == True{} : Bool}
    Extrema.maximum_upper(U32.to_nat(first),U32.to_nat(end),image_end,first_inside,end_inside) : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(BandWindow.subtract_clipped(end,first))),image_end) == True{} : Bool}}
  BandDense.bounds_leaf(images,channels,height,width,limits,first,BandWindow.subtract_clipped(end,first),capacity,
    BandPlan.ImageSpan{image,image_inside,BandPlan.LeafWithin{lower,within}},
    BandPlan.LeafWithin{Order.zero_le(U32.to_nat(first)),Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(BandWindow.subtract_clipped(end,first))),image_end,row_total,within,image_room)},domain)

# Actual halo ranges stay within their image. These allocation facts are
# separate from the proof that the ranges contain all requested samples.
type Endpoint<-image: U32,-height: U32,-position: U32> is Data:
  Endpoint{lower: {Nat.is_le(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(position)) == True{} : Bool},
    upper: {Nat.is_le(U32.to_nat(position),Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(height))) == True{} : Bool}}

def endpoint(+images: U32,+channels: U32,+height: U32,+width: U32,+image: U32,+local: U32,+limits: Machine.Limits,+capacity: U32,
  domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,image_inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},
  +local_inside: {Nat.is_le(U32.to_nat(local),U32.to_nat(height)) == True{} : Bool}) -> Endpoint<image,height,(image * height + local : U32)>:
  +base = (image * height : U32)
  +base_nat = Nat.mul(U32.to_nat(image),U32.to_nat(height))
  +end = Nat.add(base_nat,U32.to_nat(height))
  +total = Nat.mul(U32.to_nat(images),U32.to_nat(height))
  +limit = Machine.capacity(limits)
  +limit_ok = Limits.capacity_fits(limits)
  +end_room = {
    %Order.commutative(U32.to_nat(height),base_nat) : {Nat.is_le(_,total) == True{} : Bool}
    RowMajor.mul_le(1n+U32.to_nat(image),U32.to_nat(images),U32.to_nat(height),Order.strict_follow(U32.to_nat(image),U32.to_nat(images),image_inside)) : {Nat.is_le(end,total) == True{} : Bool}}
  +room = Order.transitive(end,total,U32.to_nat(limit),end_room,BandGather.rows_room(Bands.Geometry{images,channels,height,width},limits,capacity,domain))
  +base_exact = Arithmetic.multiply(image,height,limit,limit_ok,Order.transitive(base_nat,end,U32.to_nat(limit),Order.le_add(base_nat,U32.to_nat(height)),room))
  +exact = Equal.trans(Nat,U32.to_nat((base + local : U32)),Nat.add(U32.to_nat(base),U32.to_nat(local)),Nat.add(base_nat,U32.to_nat(local)),
    Arithmetic.add(base,local,limit,limit_ok,
      %Equal.sym(Nat,U32.to_nat(base),base_nat,base_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(local)),U32.to_nat(limit)) == True{} : Bool}
      Order.transitive(Nat.add(base_nat,U32.to_nat(local)),end,U32.to_nat(limit),Order.offset_le(base_nat,U32.to_nat(local),U32.to_nat(height),local_inside),room)),
    Equal.cong(Nat,Nat,start => Nat.add(start,U32.to_nat(local)),U32.to_nat(base),base_nat,base_exact))
  Endpoint{
    %Equal.sym(Nat,U32.to_nat((base + local : U32)),Nat.add(base_nat,U32.to_nat(local)),exact) : {Nat.is_le(base_nat,_) == True{} : Bool} Order.le_add(base_nat,U32.to_nat(local)),
    %Equal.sym(Nat,U32.to_nat((base + local : U32)),Nat.add(base_nat,U32.to_nat(local)),exact) : {Nat.is_le(_,end) == True{} : Bool} Order.offset_le(base_nat,U32.to_nat(local),U32.to_nat(height),local_inside)}

def range_endpoints(+images: U32,+channels: U32,+height: U32,+width: U32,+image: U32,+first: U32,+end: U32,+limits: Machine.Limits,+capacity: U32,
  domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,image_inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},
  first_bounds: Endpoint<image,height,first>,end_bounds: Endpoint<image,height,end>) ->
  BandHalo.RangeFacts(Bands.Geometry{images,channels,height,width},BandWindow.clipped(first,end),capacity,Machine.half_word(limits)):
  Endpoint{lower,first_upper} = first_bounds
  Endpoint{end_lower,end_upper} = end_bounds
  clipped(images,channels,height,width,limits,capacity,U32.to_nat(image),first,end,domain,image_inside,lower,first_upper,end_upper)

def image_range(+images: U32,+channels: U32,+height: U32,+width: U32,+image: U32,+first: U32,+end: U32,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,+image_inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},
  first_inside: {Nat.is_le(U32.to_nat(first),U32.to_nat(height)) == True{} : Bool},end_inside: {Nat.is_le(U32.to_nat(end),U32.to_nat(height)) == True{} : Bool}) ->
  BandHalo.RangeFacts(Bands.Geometry{images,channels,height,width},BandWindow.clipped((image * height + first : U32),(image * height + end : U32)),capacity,Machine.half_word(limits)):
  range_endpoints(images,channels,height,width,image,(image * height + first : U32),(image * height + end : U32),limits,capacity,domain,image_inside,
    endpoint(images,channels,height,width,image,first,limits,capacity,domain,image_inside,first_inside),endpoint(images,channels,height,width,image,end,limits,capacity,domain,image_inside,end_inside))

def minimum_below(+left: U32,+right: U32) -> {Nat.is_le(U32.to_nat(U32.min(left,right)),U32.to_nat(left)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.min(left,right)),Nat.min(U32.to_nat(left),U32.to_nat(right)),U32Extrema.minimum(left,right)) : {Nat.is_le(_,U32.to_nat(left)) == True{} : Bool}
  Extrema.minimum_left(U32.to_nat(left),U32.to_nat(right))

def minimum_right_below(+left: U32,+right: U32,+limit: U32,room: {Nat.is_le(U32.to_nat(right),U32.to_nat(limit)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(U32.min(left,right)),U32.to_nat(limit)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.min(left,right)),Nat.min(U32.to_nat(left),U32.to_nat(right)),U32Extrema.minimum(left,right)) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
  Order.transitive(Nat.min(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(right),U32.to_nat(limit),Extrema.minimum_right(U32.to_nat(left),U32.to_nat(right)),room)

def maximum_below(+left: U32,+right: U32,+limit: U32,left_room: {Nat.is_le(U32.to_nat(left),U32.to_nat(limit)) == True{} : Bool},right_room: {Nat.is_le(U32.to_nat(right),U32.to_nat(limit)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(U32.max(left,right)),U32.to_nat(limit)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.max(left,right)),Nat.max(U32.to_nat(left),U32.to_nat(right)),U32Extrema.maximum(left,right)) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
  Extrema.maximum_upper(U32.to_nat(left),U32.to_nat(right),U32.to_nat(limit),left_room,right_room)

def raw_window(+height: U32,+image: U32,core_first: U32,core_end: U32,+needed_first: U32,+needed_end: U32,output: BandWindow.Rows) -> BandWindow.RowWindow:
  BandWindow.RowWindow{output,BandWindow.clipped((image * height + needed_first : U32),(image * height + U32.min(core_first,needed_end) : U32)),
    BandWindow.clipped((image * height + U32.max(core_end,needed_first) : U32),(image * height + needed_end : U32)),
    BandWindow.clipped((image * height + needed_first : U32),(image * height + needed_end : U32))}

def raw_fits(+images: U32,+channels: U32,+height: U32,+width: U32,+image: U32,+core_first: U32,+core_end: U32,+needed_first: U32,+needed_end: U32,+output: BandWindow.Rows,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,+image_inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},
  core_inside: {Nat.is_le(U32.to_nat(core_end),U32.to_nat(height)) == True{} : Bool},+first_inside: {Nat.is_le(U32.to_nat(needed_first),U32.to_nat(height)) == True{} : Bool},+end_inside: {Nat.is_le(U32.to_nat(needed_end),U32.to_nat(height)) == True{} : Bool}) ->
  BandHalo.LeafFacts(Bands.Geometry{images,channels,height,width},raw_window(height,image,core_first,core_end,needed_first,needed_end,output),capacity,Machine.half_word(limits)):
  Shape.Both{image_range(images,channels,height,width,image,needed_first,U32.min(core_first,needed_end),limits,capacity,domain,image_inside,first_inside,minimum_right_below(core_first,needed_end,height,end_inside)),
    image_range(images,channels,height,width,image,U32.max(core_end,needed_first),needed_end,limits,capacity,domain,image_inside,maximum_below(core_end,needed_first,height,core_inside,first_inside),end_inside)}

def empty_range(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+limits: Machine.Limits,+capacity: U32,
  domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,facts: BandDense.LeafFacts<Bands.Geometry{images,channels,height,width},first,rows,capacity,Machine.half_word(limits)>) ->
  BandDense.LeafFacts<Bands.Geometry{images,channels,height,width},first,0,capacity,Machine.half_word(limits)>:
  BandDense.LeafFacts{allocation,contents,BandShape.Fields{local,rows_fit,length_exact}} = facts
  BandDense.bounds_leaf(images,channels,height,width,limits,first,0,capacity,BandPlan.EmptySpan{{==}},BandPlan.LeafWithin{Order.zero_le(U32.to_nat(first)),
    %Equal.sym(Nat,Nat.add(U32.to_nat(first),0n),U32.to_nat(first),Addition.zero_right(U32.to_nat(first))) : {Nat.is_le(_,Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool}
    Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height)),Order.le_add(U32.to_nat(first),U32.to_nat(rows)),rows_fit)},domain)

def halo_bounds_nonempty_fits(active: Bool,+images: U32,+channels: U32,+height: U32,+width: U32,+halo_bounds_window: BandWindow.RowWindow,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,facts: BandHalo.LeafFacts(Bands.Geometry{images,channels,height,width},halo_bounds_window,capacity,Machine.half_word(limits))) ->
  BandHalo.LeafFacts(Bands.Geometry{images,channels,height,width},BandWindow.nonempty(active,halo_bounds_window),capacity,Machine.half_word(limits)):
  match active halo_bounds_window:
    case True{} halo_bounds_window: facts
    case False{} BandWindow.RowWindow{output,BandWindow.Rows{+above_first,+above_rows},BandWindow.Rows{+below_first,+below_rows},BandWindow.Rows{read_first,read_rows}}:
      Shape.Both{above,below} = facts
      Shape.Both{empty_range(images,channels,height,width,above_first,above_rows,limits,capacity,domain,above),empty_range(images,channels,height,width,below_first,below_rows,limits,capacity,domain,below)}

def halo_bounds_window(+images: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_height: U32,+output_width: U32,+first: U32,+rows: U32,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,image_inside: {Nat.is_lt(U32.to_nat((first / height : U32)),U32.to_nat(images)) == True{} : Bool},
  core_inside: {Nat.is_le(U32.to_nat((first % height + rows : U32)),U32.to_nat(height)) == True{} : Bool}) ->
  BandHalo.LeafFacts(Bands.Geometry{images,channels,height,width},BandWindow.window(BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width},first,rows),capacity,Machine.half_word(limits)):
  +image = (first / height : U32)
  +core_first = (first % height : U32)
  +core_end = (core_first + rows : U32)
  +out_first = U32.min(output_height,BandWindow.ceiling(core_first,stride))
  +out_end = Bool.pick(U32,U32.is_eq(core_end,height),output_height,U32.min(output_height,BandWindow.ceiling(core_end,stride)))
  +needed = U32.min(height,BandWindow.subtract_clipped((out_first * stride : U32),padding))
  +needed_first = BandWindow.start(height,out_first,stride,padding)
  +needed_end = U32.min(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(out_end,1) * stride + kernel : U32),padding))
  +output = {BandWindow.Rows{(image * output_height + out_first : U32),BandWindow.subtract_clipped(out_end,out_first)} : BandWindow.Rows}
  halo_bounds_nonempty_fits(U32.is_lt(out_first,out_end),images,channels,height,width,raw_window(height,image,core_first,core_end,needed_first,needed_end,output),limits,capacity,domain,
    raw_fits(images,channels,height,width,image,core_first,core_end,needed_first,needed_end,output,limits,capacity,domain,image_inside,core_inside,
      Order.transitive(U32.to_nat(needed_first),U32.to_nat(needed),U32.to_nat(height),clipped_below(needed,(needed % stride : U32)),
        minimum_below(height,BandWindow.subtract_clipped((out_first * stride : U32),padding))),minimum_below(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(out_end,1) * stride + kernel : U32),padding))))

type CoreGeometry<-images: U32,-height: U32,-first: U32,-rows: U32,-capacity: U32> is Data:
  CoreGeometry{image_inside: {Nat.is_lt(U32.to_nat((first / height : U32)),U32.to_nat(images)) == True{} : Bool},
    core_inside: {Nat.is_le(U32.to_nat((first % height + rows : U32)),U32.to_nat(height)) == True{} : Bool},
    first_exact: {U32.to_nat(first) == Nat.add(U32.to_nat((first / height * height : U32)),U32.to_nat((first % height : U32))) : Nat},
    core_exact: {U32.to_nat((first % height + rows : U32)) == Nat.add(U32.to_nat((first % height : U32)),U32.to_nat(rows)) : Nat},
    endpoint_room: {Nat.is_le(Nat.add(U32.to_nat((first / height * height : U32)),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}}

def halo_bounds_coordinates(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+length: Nat,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,length),
  positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}) ->
  CoreGeometry<images,height,first,rows,Machine.capacity(limits)>:
  BandGather.Domain{capacity_ok,channels_positive,+height_positive,width_positive,machine_room,allocation_room} = domain
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  +total = Nat.mul(U32.to_nat(images),U32.to_nat(height))
  +first_inside = Bounds.nat_trans(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),total,
    %Addition.zero_right(U32.to_nat(first)) : {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}
    Order.offset_lt(U32.to_nat(first),0n,U32.to_nat(rows),positive),rows_fit)
  +images_positive = RowMajor.left_factor_positive(U32.to_nat(images),U32.to_nat(height),Order.le_lt_trans(0n,U32.to_nat(first),total,Order.zero_le(U32.to_nat(first)),first_inside))
  +height_room = Order.transitive(U32.to_nat(height),total,U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(U32.to_nat(height),U32.to_nat(images)) : {Nat.is_le(U32.to_nat(height),_) == True{} : Bool}
    RowMajor.product_covers(U32.to_nat(height),U32.to_nat(images),images_positive),BandGather.rows_room(Bands.Geometry{images,channels,height,width},limits,capacity,domain))
  +double_height = Limits.double_bound(limits,U32.to_nat(height),height_room)
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +image = Nat.div(U32.to_nat(first),U32.to_nat(height))
  +remainder = Nat.mod(U32.to_nat(first),U32.to_nat(height))
  +identity = DivisionOrder.quotient_exact(U32.to_nat(first),U32.to_nat(height),height_positive)
  +image_inside = {
    %Equal.sym(Nat,U32.to_nat((first / height : U32)),image,U32Division.divide(first,height,half,half_ok,double_height,height_positive)) : {Nat.is_lt(_,U32.to_nat(images)) == True{} : Bool}
    RowMajor.row_inside(image,U32.to_nat(images),U32.to_nat(height),remainder,
      %identity : {Nat.is_lt(_,total) == True{} : Bool} first_inside) : {Nat.is_lt(U32.to_nat((first / height : U32)),U32.to_nat(images)) == True{} : Bool}}
  +remainder_exact = U32Division.modulo(first,height,half,half_ok,double_height,height_positive)
  +local_span = {
    %Equal.sym(Nat,U32.to_nat((first % height : U32)),remainder,remainder_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool}
    Inspection.local_span(height,first,rows,half,half_ok,height_positive,double_height,local) : {Nat.is_le(Nat.add(U32.to_nat((first % height : U32)),U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool}}
  +core_exact = Arithmetic.add((first % height : U32),rows,Machine.capacity(limits),Limits.capacity_fits(limits),
    Order.transitive(Nat.add(U32.to_nat((first % height : U32)),U32.to_nat(rows)),U32.to_nat(height),U32.to_nat(Machine.capacity(limits)),local_span,height_room))
  +native_image = (first / height : U32)
  +base = (native_image * height : U32)
  +base_natural = Nat.mul(U32.to_nat(native_image),U32.to_nat(height))
  +image_exact = U32Division.divide(first,height,half,half_ok,double_height,height_positive)
  +end_room = Order.transitive(Nat.add(base_natural,U32.to_nat(height)),total,U32.to_nat(Machine.capacity(limits)),
    %Order.commutative(U32.to_nat(height),base_natural) : {Nat.is_le(_,total) == True{} : Bool}
    RowMajor.mul_le(1n+U32.to_nat(native_image),U32.to_nat(images),U32.to_nat(height),Order.strict_follow(U32.to_nat(native_image),U32.to_nat(images),image_inside)),
    BandGather.rows_room(Bands.Geometry{images,channels,height,width},limits,capacity,domain))
  +base_exact = Arithmetic.multiply(native_image,height,Machine.capacity(limits),Limits.capacity_fits(limits),
    Order.transitive(base_natural,Nat.add(base_natural,U32.to_nat(height)),U32.to_nat(Machine.capacity(limits)),Order.le_add(base_natural,U32.to_nat(height)),end_room))
  CoreGeometry{image_inside,
    %Equal.sym(Nat,U32.to_nat((first % height + rows : U32)),Nat.add(U32.to_nat((first % height : U32)),U32.to_nat(rows)),core_exact) : {Nat.is_le(_,U32.to_nat(height)) == True{} : Bool}
    local_span,
    %Equal.sym(Nat,U32.to_nat(base),base_natural,base_exact) : {U32.to_nat(first) == Nat.add(_,U32.to_nat((first % height : U32))) : Nat}
    %Equal.sym(Nat,U32.to_nat(native_image),image,image_exact) : {U32.to_nat(first) == Nat.add(Nat.mul(_,U32.to_nat(height)),U32.to_nat((first % height : U32))) : Nat}
    %Equal.sym(Nat,U32.to_nat((first % height : U32)),remainder,remainder_exact) : {U32.to_nat(first) == Nat.add(Nat.mul(image,U32.to_nat(height)),_) : Nat}
    identity,core_exact,
    %Equal.sym(Nat,U32.to_nat(base),base_natural,base_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(height)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} end_room}

def bounded_window(+images: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_height: U32,+output_width: U32,+first: U32,+rows: U32,+limits: Machine.Limits,+capacity: U32,
  domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,halo_bounds_coordinates: CoreGeometry<images,height,first,rows,Machine.capacity(limits)>) ->
  BandHalo.LeafFacts(Bands.Geometry{images,channels,height,width},BandWindow.window(BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width},first,rows),capacity,Machine.half_word(limits)):
  CoreGeometry{image_inside,core_inside,first_exact,core_exact,endpoint_room} = halo_bounds_coordinates
  halo_bounds_window(images,channels,height,width,kernel,stride,padding,output_height,output_width,first,rows,limits,capacity,domain,image_inside,core_inside)

def nonempty_window(+images: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_height: U32,+output_width: U32,+first: U32,+rows: U32,+length: Nat,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,length),
  positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}) ->
  BandHalo.LeafFacts(Bands.Geometry{images,channels,height,width},BandWindow.window(BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width},first,rows),capacity,Machine.half_word(limits)):
  bounded_window(images,channels,height,width,kernel,stride,padding,output_height,output_width,first,rows,limits,capacity,domain,halo_bounds_coordinates(images,channels,height,width,first,rows,length,limits,capacity,domain,metadata,positive))

def planned(~Element: Data,+geometry: BandWindow.Geometry,source: Model.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<BandWindow.input_geometry(geometry),limits,capacity>,metadata: BandShape.Contents(~Element,BandWindow.input_geometry(geometry),Model.content(~Element,source)),
  positive: BandPlan.Nonempty(Model.layout(~Element,source))) -> BandHalo.Fits(geometry,Model.layout(~Element,source),capacity,Machine.half_word(limits)):
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  match source:
    case Model.Leaf{Buffer.Model{tree,length,certificate},+first,+rows}: nonempty_window(images,channels,height,width,kernel,stride,padding,output_height,output_width,first,rows,length,limits,capacity,domain,metadata,positive)
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_positive,right_positive} = positive
      Shape.Both{planned(~Element,geometry,left,limits,capacity,domain,left_metadata,left_positive),planned(~Element,geometry,right,limits,capacity,domain,right_metadata,right_positive)}

def prepared(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+fill: Element,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<BandWindow.input_geometry(geometry),limits,capacity>,
  +well_formed: BandShape.WellFormed<Element,BandWindow.input_geometry(geometry),Model.content(~Element,source)>,values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.content(~Element,source)),
  positive: BandPlan.Nonempty(Model.layout(~Element,source))) ->
  BandHalo.Prepared(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandHalo.refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,Model.layout(~Element,source),Model.pack(~Element,source)))):
  BandShape.WellFormed{metadata,ready,coverage} = well_formed
  BandHalo.prepared(~Element,~Logical,~original,logical,geometry,fill,Model.layout(~Element,source),source,limits,capacity,domain,well_formed,values,
    planned(~Element,geometry,source,limits,capacity,domain,metadata,positive))

# A leaf window is the shared range copy of its rows from the upper halo, the
# core and the lower halo. Given a source for every window row, the window
# holds the original values of its rows, as a halo range does.
def source_contents(~Element: Data,above: Buffer.Content<Element>,above_range: BandWindow.Rows,core: Buffer.Content<Element>,core_range: BandWindow.Rows,
  below: Buffer.Content<Element>,below_range: BandWindow.Rows) -> Model.Content<Element>:
  BandWindow.Rows{above_first,above_rows} = above_range
  BandWindow.Rows{core_first,core_rows} = core_range
  BandWindow.Rows{below_first,below_rows} = below_range
  Model.Joined{Model.RowBand{above,above_first,above_rows},
    Model.Joined{Model.RowBand{core,core_first,core_rows},Model.RowBand{below,below_first,below_rows}}}

# A contents model independent of certificates, for induction through cloned
# owners. This states the actual range-copy computation, not its readback law.
def assembled_content(~Element: Data,exact: Bool,+geometry: Bands.Geometry,+fill: Element,assembly_window: BandWindow.Rows,+core_range: BandWindow.Rows,
  +above_range: BandWindow.Rows,+below_range: BandWindow.Rows,core: Buffer.Content<Element>,above: Buffer.Content<Element>,below: Buffer.Content<Element>) -> Buffer.Content<Element>:
  match exact:
    case True{}: core
    case False{}:
      BandWindow.Rows{first,rows} = assembly_window
      BandRepartition.leaf_buffer(~Element,geometry,fill,first,rows,source_contents(~Element,above,above_range,core,core_range,below,below_range))

def assembly_content(~Element: Data,+geometry: Bands.Geometry,+fill: Element,assembly_window: BandWindow.Rows,core_range: BandWindow.Rows,above_range: BandWindow.Rows,
  below_range: BandWindow.Rows,core: Buffer.Content<Element>,above: Buffer.Content<Element>,below: Buffer.Content<Element>) -> Buffer.Content<Element>:
  BandWindow.Rows{+window_first,+window_rows} = assembly_window
  BandWindow.Rows{+core_first,+core_rows} = core_range
  assembled_content(~Element,U32.is_eq(window_first,core_first) && U32.is_eq(window_rows,core_rows),geometry,fill,
    BandWindow.Rows{window_first,window_rows},BandWindow.Rows{core_first,core_rows},above_range,below_range,core,above,below)

def sources(~Element: Data,above: Buffer.Model<Element>,above_first: U32,above_rows: U32,core: Buffer.Model<Element>,core_first: U32,core_rows: U32,
  below: Buffer.Model<Element>,below_first: U32,below_rows: U32) -> Model.Model<Element>:
  Model.Fork{Model.Leaf{above,above_first,above_rows},Model.Fork{Model.Leaf{core,core_first,core_rows},Model.Leaf{below,below_first,below_rows}}}

def copied_contents(~Element: Data,owners: Bands.Bands(~Element) & Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,BandWindow.copied_window(~Element,owners)) == BandRange.result(~Element,BandRange.contents(~Element,owners)) : Buffer.Content<Element>}:
  (bands,assembly_window) = owners
  {==}

def Copy(~Element: Data) -> Type:
  Bands.Bands(~Element) & Storage.Buffer<Element>

# The three direct range copies of a leaf assembly_window write what one range copy
# over the three-source tree writes. Each step opens one copy result.
def lower_copy(~Element: Data,-upper: Bands.Bands(~Element),-middle: Bands.Bands(~Element),copy: Copy(~Element)) ->
  {BandWindow.copied_window(~Element,copy) ==
    BandWindow.copied_window(~Element,Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~Copy(~Element),
      Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~Copy(~Element),copy,right => output => (Partition.Fork{middle,right},output)),
      right => output => (Partition.Fork{upper,right},output))) : Storage.Buffer<Element>}:
  (right,output) = copy
  {==}

def middle_copy(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,-upper: Bands.Bands(~Element),+below_first: U32,+below_rows: U32,
  below: Storage.Buffer<Element>,copy: Copy(~Element)) ->
  {BandWindow.copied_window(~Element,Bands.range_leaf(~Element,geometry,first,rows,0,below_first,below_rows,below,BandWindow.copied_window(~Element,copy))) ==
    BandWindow.copied_window(~Element,Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~Copy(~Element),
      Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~Copy(~Element),copy,left => output =>
        Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~Copy(~Element),
          Bands.range_into(~Element,geometry,first,rows,0,Partition.Leaf{below,below_first,below_rows,Unit{}},output),right => output => (Partition.Fork{left,right},output))),
      right => output => (Partition.Fork{upper,right},output))) : Storage.Buffer<Element>}:
  (left,output) = copy
  lower_copy(~Element,upper,left,Bands.range_leaf(~Element,geometry,first,rows,0,below_first,below_rows,below,output))

def upper_copy(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+core_first: U32,+core_rows: U32,+below_first: U32,+below_rows: U32,
  core: Storage.Buffer<Element>,below: Storage.Buffer<Element>,copy: Copy(~Element)) ->
  {BandWindow.copied_window(~Element,Bands.range_leaf(~Element,geometry,first,rows,0,below_first,below_rows,below,
      BandWindow.copied_window(~Element,Bands.range_leaf(~Element,geometry,first,rows,0,core_first,core_rows,core,BandWindow.copied_window(~Element,copy))))) ==
    BandWindow.copied_window(~Element,Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~Copy(~Element),copy,left => output =>
      Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~Copy(~Element),
        Bands.range_into(~Element,geometry,first,rows,0,
          Partition.Fork{Partition.Leaf{core,core_first,core_rows,Unit{}},Partition.Leaf{below,below_first,below_rows,Unit{}}},output),
        right => output => (Partition.Fork{left,right},output)))) : Storage.Buffer<Element>}:
  (left,output) = copy
  middle_copy(~Element,geometry,first,rows,left,below_first,below_rows,below,Bands.range_leaf(~Element,geometry,first,rows,0,core_first,core_rows,core,output))

def direct(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+above_first: U32,+above_rows: U32,+core_first: U32,+core_rows: U32,
  +below_first: U32,+below_rows: U32,above: Storage.Buffer<Element>,core: Storage.Buffer<Element>,below: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  {BandWindow.copied_window(~Element,Bands.range_leaf(~Element,geometry,first,rows,0,below_first,below_rows,below,
      BandWindow.copied_window(~Element,Bands.range_leaf(~Element,geometry,first,rows,0,core_first,core_rows,core,
        BandWindow.copied_window(~Element,Bands.range_leaf(~Element,geometry,first,rows,0,above_first,above_rows,above,output)))))) ==
    BandWindow.copied_window(~Element,Bands.range_into(~Element,geometry,first,rows,0,
      Partition.Fork{Partition.Leaf{above,above_first,above_rows,Unit{}},
        Partition.Fork{Partition.Leaf{core,core_first,core_rows,Unit{}},Partition.Leaf{below,below_first,below_rows,Unit{}}}},output)) : Storage.Buffer<Element>}:
  upper_copy(~Element,geometry,first,rows,core_first,core_rows,below_first,below_rows,core,below,
    Bands.range_leaf(~Element,geometry,first,rows,0,above_first,above_rows,above,output))

# The copied assembly_window is the repartition leaf of its rows over the three sources.
def copied(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,
  +above: Buffer.Model<Element>,+above_first: U32,+above_rows: U32,+core: Buffer.Model<Element>,+core_first: U32,+core_rows: U32,
  +below: Buffer.Model<Element>,+below_first: U32,+below_rows: U32) ->
  {Buffer.contents(~Element,BandWindow.assembled(~Element,False{},fill,geometry,BandWindow.Rows{first,rows},BandWindow.Rows{core_first,core_rows},BandWindow.Rows{above_first,above_rows},
      BandWindow.Rows{below_first,below_rows},Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))) ==
    BandRepartition.leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,sources(~Element,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows))) : Buffer.Content<Element>}:
  +request = {BandRange.Request{geometry,first,rows,0} : BandRange.Request}
  +count = Bands.local_count(geometry,rows)
  +source = sources(~Element,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows)
  +fresh = {Buffer.Content{Tree.fresh(Element,Storage.depth_for(count),fill),U32.to_nat(count)} : Buffer.Content<Element>}
  Equal.trans(Buffer.Content<Element>,
    Buffer.contents(~Element,BandWindow.assembled(~Element,False{},fill,geometry,BandWindow.Rows{first,rows},BandWindow.Rows{core_first,core_rows},BandWindow.Rows{above_first,above_rows},
      BandWindow.Rows{below_first,below_rows},Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))),
    Buffer.contents(~Element,BandWindow.copied_window(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count)))),
    BandRepartition.leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source)),
    Equal.cong(Storage.Buffer<Element>,Buffer.Content<Element>,owner => Buffer.contents(~Element,owner),
      BandWindow.assembled(~Element,False{},fill,geometry,BandWindow.Rows{first,rows},BandWindow.Rows{core_first,core_rows},BandWindow.Rows{above_first,above_rows},
        BandWindow.Rows{below_first,below_rows},Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below)),
      BandWindow.copied_window(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count))),
      direct(~Element,geometry,first,rows,above_first,above_rows,core_first,core_rows,below_first,below_rows,
        Buffer.pack(~Element,above),Buffer.pack(~Element,core),Buffer.pack(~Element,below),Bands.allocate(~Element,fill,count))),
  Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,BandWindow.copied_window(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count)))),
    BandRange.result(~Element,BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count)))),
    BandRepartition.leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source)),
    copied_contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count))),
    Equal.trans(Buffer.Content<Element>,BandRange.result(~Element,BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count)))),
      BandRange.result(~Element,BandRange.written(~Element,request,Model.content(~Element,source),Buffer.contents(~Element,Bands.allocate(~Element,fill,count)))),
      BandRepartition.leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source)),
      Equal.cong(BandRange.Content<Element>,Buffer.Content<Element>,content => BandRange.result(~Element,content),
        BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count))),
        BandRange.written(~Element,request,Model.content(~Element,source),Buffer.contents(~Element,Bands.allocate(~Element,fill,count))),
        BandRange.native_range(~Element,request,source,Bands.allocate(~Element,fill,count))),
      Equal.cong(Buffer.Content<Element>,Buffer.Content<Element>,output => BandRange.result(~Element,BandRange.written(~Element,request,Model.content(~Element,source),output)),
        Buffer.contents(~Element,Bands.allocate(~Element,fill,count)),fresh,AllocationLaws.contents(~Element,Storage.depth_for(count),fill,U32.to_nat(count))))))

def assembled_contents(~Element: Data,exact: Bool,+geometry: Bands.Geometry,+fill: Element,+assembly_window: BandWindow.Rows,+core_range: BandWindow.Rows,
  +above_range: BandWindow.Rows,+below_range: BandWindow.Rows,+core: Buffer.Model<Element>,+above: Buffer.Model<Element>,+below: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,BandWindow.assembled(~Element,exact,fill,geometry,assembly_window,core_range,above_range,below_range,
      Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))) ==
    assembled_content(~Element,exact,geometry,fill,assembly_window,core_range,above_range,below_range,Buffer.content(~Element,core),Buffer.content(~Element,above),Buffer.content(~Element,below)) : Buffer.Content<Element>}:
  match exact:
    case True{}: Buffer.packed_contents(~Element,core)
    case False{}:
      BandWindow.Rows{+first,+rows} = assembly_window
      BandWindow.Rows{+core_first,+core_rows} = core_range
      BandWindow.Rows{+above_first,+above_rows} = above_range
      BandWindow.Rows{+below_first,+below_rows} = below_range
      copied(~Element,geometry,fill,first,rows,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows)

def assemble_contents(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+assembly_window: BandWindow.Rows,+core_range: BandWindow.Rows,
  +above_range: BandWindow.Rows,+below_range: BandWindow.Rows,+core: Buffer.Model<Element>,+above: Buffer.Model<Element>,+below: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,BandWindow.assemble(~Element,fill,geometry,assembly_window,core_range,above_range,below_range,
      Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))) ==
    assembly_content(~Element,geometry,fill,assembly_window,core_range,above_range,below_range,Buffer.content(~Element,core),Buffer.content(~Element,above),Buffer.content(~Element,below)) : Buffer.Content<Element>}:
  BandWindow.Rows{+window_first,+window_rows} = assembly_window
  BandWindow.Rows{+core_first,+core_rows} = core_range
  assembled_contents(~Element,U32.is_eq(window_first,core_first) && U32.is_eq(window_rows,core_rows),geometry,fill,assembly_window,core_range,above_range,below_range,core,above,below)

# A cell of the assembled window is the cell the located range copy wrote.
def assembly_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,source)),ready: Model.Ready(~Element,source),
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),within: BandPlan.Within(Model.layout(~Element,source),0n,U32.to_nat(Machine.capacity(limits))),
  locations: BandRange.RowLocations(~Element,BandRange.Request{geometry,first,rows,0},source),
  target: BandDense.LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool}) ->
  BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,
    Buffer.content_tree(~Element,BandRepartition.leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),Buffer.content_length(~Element,BandRepartition.leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),channel,row,column):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandDense.LeafFacts{allocation,contents,+target_metadata} = target
  BandShape.Fields{local,rows_fit,length_exact} = target_metadata
  +count = Bands.local_count(Bands.Geometry{images,channels,height,width},rows)
  +output = BandCount.model(~Element,count,fill)
  +output_ready = BandCount.bounded_model(~Element,count,fill,allocation)
  Shape.Both{BandValues.position_within(Bands.Geometry{images,channels,height,width},rows,U32.to_nat(count),channel,row,column,length_exact,channel_inside,row_inside,column_inside),
    BandRange.located_written_point(~Element,~Logical,~original,logical,BandRange.Request{Bands.Geometry{images,channels,height,width},first,rows,0},source,output,limits,U32.to_nat(count),capacity,domain,
      BandRepartition.target_bounds(~Element,Bands.Geometry{images,channels,height,width},first,rows,output,limits,capacity,domain,target_metadata,output_ready),metadata,ready,target_metadata,output_ready,values,within,locations,
      channel,row,column,channel_inside,row_inside,column_inside)}

# What one assembled window supplies to every cell.
type AssemblyCells<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  AssemblyCells{logical: Logical,geometry: Bands.Geometry,fill: Element,first: U32,rows: U32,source: Model.Model<Element>,limits: Machine.Limits,capacity: U32,
    domain: BandGather.Domain<geometry,limits,capacity>,metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,source)),ready: Model.Ready(~Element,source),
    values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),within: BandPlan.Within(Model.layout(~Element,source),0n,U32.to_nat(Machine.capacity(limits))),
    locations: BandRange.RowLocations(~Element,BandRange.Request{geometry,first,rows,0},source),target: BandDense.LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>}

def cells_geometry(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: AssemblyCells<Element,Logical,original>) -> Bands.Geometry:
  AssemblyCells{logical,geometry,fill,first,rows,source,limits,capacity,domain,metadata,ready,values,within,locations,target} = evidence
  geometry

def cells_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: AssemblyCells<Element,Logical,original>) -> U32:
  AssemblyCells{logical,geometry,fill,first,rows,source,limits,capacity,domain,metadata,ready,values,within,locations,target} = evidence
  rows

def AssemblyCell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: AssemblyCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  AssemblyCells{logical,+geometry,+fill,+first,+rows,+source,limits,capacity,domain,metadata,ready,values,within,locations,target} = evidence
  BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,
    Buffer.content_tree(~Element,BandRepartition.leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),Buffer.content_length(~Element,BandRepartition.leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),channel,row,column)

def assembly_leaf_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: AssemblyCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(cells_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(cells_rows(~Element,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(cells_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool}) ->
  AssemblyCell(~Element,~Logical,~original,evidence,channel,row,column):
  AssemblyCells{logical,geometry,fill,first,rows,source,limits,capacity,domain,metadata,ready,values,within,locations,target} = evidence
  assembly_cell(~Element,~Logical,~original,logical,geometry,fill,first,rows,source,limits,capacity,domain,metadata,ready,values,within,locations,target,channel,row,column,channel_inside,row_inside,column_inside)

# The copied assembly_window satisfies the halo range relation for its own rows.
def copied_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  +metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,source)),+ready: Model.Ready(~Element,source),
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),+within: BandPlan.Within(Model.layout(~Element,source),0n,U32.to_nat(Machine.capacity(limits))),
  +locations: BandRange.RowLocations(~Element,BandRange.Request{geometry,first,rows,0},source),
  +target: BandDense.LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>) ->
  BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{first,rows},BandRepartition.leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))):
  BandDense.LeafFacts{allocation,contents,+target_metadata} = target
  Shape.Both{BandRepartition.leaf_ready(~Element,geometry,fill,first,rows,Model.content(~Element,source),capacity,Machine.half_word(limits),target),
    Shape.Both{target_metadata,
      BandValues.cells(~AssemblyCells<Element,Logical,original>,~AssemblyCell(~Element,~Logical,~original),~cells_geometry(~Element,~Logical,~original),~cells_rows(~Element,~Logical,~original),~assembly_leaf_cell(~Element,~Logical,~original),
        AssemblyCells{logical,geometry,fill,first,rows,source,limits,capacity,domain,metadata,ready,values,within,locations,target})}}

def exact_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,
  +core: Buffer.Model<Element>,+core_first: U32,+core_rows: U32,+exact: {U32.is_eq(first,core_first) && U32.is_eq(rows,core_rows) == True{} : Bool},
  core_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{core_first,core_rows},Buffer.content(~Element,core))) ->
  BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{first,rows},Buffer.contents(~Element,Buffer.pack(~Element,core))):
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,core)),Buffer.content(~Element,core),Buffer.packed_contents(~Element,core)) :
    BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{first,rows},_)
  %Equal.sym(U32,first,core_first,BandRange.same_word(first,core_first,Boolean.left(U32.is_eq(first,core_first),U32.is_eq(rows,core_rows),exact))) :
    BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{_,rows},Buffer.content(~Element,core))
  %Equal.sym(U32,rows,core_rows,BandRange.same_word(rows,core_rows,Boolean.right(U32.is_eq(first,core_first),U32.is_eq(rows,core_rows),exact))) :
    BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{core_first,_},Buffer.content(~Element,core))
  core_values

def copied_assembled_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,
  +above: Buffer.Model<Element>,+above_first: U32,+above_rows: U32,+core: Buffer.Model<Element>,+core_first: U32,+core_rows: U32,
  +below: Buffer.Model<Element>,+below_first: U32,+below_rows: U32,+limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  +above_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{above_first,above_rows},Buffer.content(~Element,above)),
  +core_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{core_first,core_rows},Buffer.content(~Element,core)),
  +below_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{below_first,below_rows},Buffer.content(~Element,below)),
  +within: BandPlan.Within(Model.layout(~Element,sources(~Element,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows)),0n,U32.to_nat(Machine.capacity(limits))),
  +locations: BandRange.RowLocations(~Element,BandRange.Request{geometry,first,rows,0},sources(~Element,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows)),
  +target: BandDense.LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>) ->
  BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{first,rows},
    Buffer.contents(~Element,BandWindow.assembled(~Element,False{},fill,geometry,BandWindow.Rows{first,rows},BandWindow.Rows{core_first,core_rows},BandWindow.Rows{above_first,above_rows},
      BandWindow.Rows{below_first,below_rows},Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below)))):
  Buffer.Model{+above_tree,+above_length,+above_certificate} = above
  Buffer.Model{+core_tree,+core_length,+core_certificate} = core
  Buffer.Model{+below_tree,+below_length,+below_certificate} = below
  Shape.Both{above_ready,Shape.Both{above_metadata,above_leaf}} = above_values
  Shape.Both{core_ready,Shape.Both{core_metadata,core_leaf}} = core_values
  Shape.Both{below_ready,Shape.Both{below_metadata,below_leaf}} = below_values
  +source = sources(~Element,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows)
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,BandWindow.assembled(~Element,False{},fill,geometry,BandWindow.Rows{first,rows},BandWindow.Rows{core_first,core_rows},
      BandWindow.Rows{above_first,above_rows},BandWindow.Rows{below_first,below_rows},Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))),
    BandRepartition.leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source)),
    copied(~Element,geometry,fill,first,rows,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows)) :
    BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{first,rows},_)
  copied_values(~Element,~Logical,~original,logical,geometry,fill,first,rows,source,limits,capacity,domain,
    Shape.Both{above_metadata,Shape.Both{core_metadata,below_metadata}},
    Shape.Both{Buffer.ready_content(~Element,above,above_ready),Shape.Both{Buffer.ready_content(~Element,core,core_ready),Buffer.ready_content(~Element,below,below_ready)}},
    Shape.Both{above_leaf,Shape.Both{core_leaf,below_leaf}},within,locations,target)

def assembled_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,
  +above: Buffer.Model<Element>,+above_first: U32,+above_rows: U32,+core: Buffer.Model<Element>,+core_first: U32,+core_rows: U32,
  +below: Buffer.Model<Element>,+below_first: U32,+below_rows: U32,+limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  exact: Bool,+choice: {U32.is_eq(first,core_first) && U32.is_eq(rows,core_rows) == exact : Bool},
  +above_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{above_first,above_rows},Buffer.content(~Element,above)),
  +core_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{core_first,core_rows},Buffer.content(~Element,core)),
  +below_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{below_first,below_rows},Buffer.content(~Element,below)),
  +within: BandPlan.Within(Model.layout(~Element,sources(~Element,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows)),0n,U32.to_nat(Machine.capacity(limits))),
  +locations: BandRange.RowLocations(~Element,BandRange.Request{geometry,first,rows,0},sources(~Element,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows)),
  +target: BandDense.LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>) ->
  BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{first,rows},
    Buffer.contents(~Element,BandWindow.assembled(~Element,exact,fill,geometry,BandWindow.Rows{first,rows},BandWindow.Rows{core_first,core_rows},BandWindow.Rows{above_first,above_rows},
      BandWindow.Rows{below_first,below_rows},Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below)))):
  match exact:
    case True{}: exact_values(~Element,~Logical,~original,logical,geometry,first,rows,core,core_first,core_rows,choice,core_values)
    case False{}:
      copied_assembled_values(~Element,~Logical,~original,logical,geometry,fill,first,rows,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows,
        limits,capacity,domain,above_values,core_values,below_values,within,locations,target)

# The actual assembled assembly_window holds the original values of its rows.
def assembly_window(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,
  +above: Buffer.Model<Element>,+above_first: U32,+above_rows: U32,+core: Buffer.Model<Element>,+core_first: U32,+core_rows: U32,
  +below: Buffer.Model<Element>,+below_first: U32,+below_rows: U32,+limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  +above_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{above_first,above_rows},Buffer.content(~Element,above)),
  +core_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{core_first,core_rows},Buffer.content(~Element,core)),
  +below_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{below_first,below_rows},Buffer.content(~Element,below)),
  +within: BandPlan.Within(Model.layout(~Element,sources(~Element,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows)),0n,U32.to_nat(Machine.capacity(limits))),
  +locations: BandRange.RowLocations(~Element,BandRange.Request{geometry,first,rows,0},sources(~Element,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows)),
  +target: BandDense.LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>) ->
  BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{first,rows},
    Buffer.contents(~Element,BandWindow.assemble(~Element,fill,geometry,BandWindow.Rows{first,rows},BandWindow.Rows{core_first,core_rows},BandWindow.Rows{above_first,above_rows},
      BandWindow.Rows{below_first,below_rows},Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below)))):
  assembled_values(~Element,~Logical,~original,logical,geometry,fill,first,rows,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows,
    limits,capacity,domain,U32.is_eq(first,core_first) && U32.is_eq(rows,core_rows),{==},above_values,core_values,below_values,within,locations,target)

# Every row of an actual leaf window lies in its upper halo, its core or its
# lower halo, and the window rows form an allocatable range of one image.
def difference_positive(+end: Nat,+begin: Nat,positive: {Nat.is_lt(0n,Nat.sub(end,begin)) == True{} : Bool}) -> {Nat.is_lt(begin,end) == True{} : Bool}:
  match end begin:
    case 0n +begin: Empty.absurd({Nat.is_lt(begin,0n) == True{} : Bool},WordEncoding.false_true(%Order.zero_subtract(begin) : {Nat.is_lt(0n,_) == True{} : Bool} positive))
    case 1n+end 0n: {==}
    case 1n+ +end 1n+ +begin: difference_positive(end,begin,positive)

def common_offset(+offset: Nat,+left: Nat,+right: Nat) -> {Nat.sub(Nat.add(offset,left),Nat.add(offset,right)) == Nat.sub(left,right) : Nat}:
  match offset:
    case 0n: {==}
    case 1n+ +rest: common_offset(rest,left,right)

# A row inside [begin, end) of an absolute source range is located there.
def located_in(+first: U32,+rows: U32,+row: Nat,+begin: Nat,+end: Nat,
  +first_exact: {U32.to_nat(first) == begin : Nat},+rows_exact: {U32.to_nat(rows) == Nat.sub(end,begin) : Nat},
  +lower: {Nat.is_le(begin,row) == True{} : Bool},+upper: {Nat.is_lt(row,end) == True{} : Bool}) -> BandPlan.LeafLocation<first,rows,row>:
  BandPlan.LeafLocation{Nat.sub(row,begin),
    %Equal.sym(Nat,U32.to_nat(rows),Nat.sub(end,begin),rows_exact) : {Nat.is_lt(Nat.sub(row,begin),_) == True{} : Bool}
    Interval.shifted_inside(begin,end,row,lower,upper),
    %Equal.sym(Nat,U32.to_nat(first),begin,first_exact) : {row == Nat.add(_,Nat.sub(row,begin)) : Nat}
    Equal.sym(Nat,Nat.add(begin,Nat.sub(row,begin)),row,Order.restore_shift(begin,row,lower))}

# One image of a leaf window: its base row, the core's image rows and the
# rows its outputs need, with the room that keeps every sum exact.
type ImageWindow<-base: U32,-height: U32,-core_first: U32,-core_end: U32,-rows: U32,-needed_first: U32,-needed_end: U32,-first: U32,-capacity: U32> is Data:
  ImageWindow{capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    endpoint_room: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool},
    core_room: {Nat.is_le(U32.to_nat(core_end),U32.to_nat(height)) == True{} : Bool},
    first_room: {Nat.is_le(U32.to_nat(needed_first),U32.to_nat(height)) == True{} : Bool},
    end_room: {Nat.is_le(U32.to_nat(needed_end),U32.to_nat(height)) == True{} : Bool},
    first_exact: {U32.to_nat(first) == Nat.add(U32.to_nat(base),U32.to_nat(core_first)) : Nat},
    core_exact: {U32.to_nat(core_end) == Nat.add(U32.to_nat(core_first),U32.to_nat(rows)) : Nat}}

def shifted(+base: U32,+local: U32,+height: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +endpoint_room: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool},
  +local_room: {Nat.is_le(U32.to_nat(local),U32.to_nat(height)) == True{} : Bool}) ->
  {U32.to_nat((base + local : U32)) == Nat.add(U32.to_nat(base),U32.to_nat(local)) : Nat}:
  Arithmetic.add(base,local,capacity,capacity_ok,
    Order.transitive(Nat.add(U32.to_nat(base),U32.to_nat(local)),Nat.add(U32.to_nat(base),U32.to_nat(height)),U32.to_nat(capacity),
      Order.offset_le(U32.to_nat(base),U32.to_nat(local),U32.to_nat(height),local_room),endpoint_room))

def shifted_room(+base: U32,+local: U32,+height: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +endpoint_room: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool},
  +local_room: {Nat.is_le(U32.to_nat(local),U32.to_nat(height)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat((base + local : U32)),U32.to_nat(capacity)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((base + local : U32)),Nat.add(U32.to_nat(base),U32.to_nat(local)),shifted(base,local,height,capacity,capacity_ok,endpoint_room,local_room)) :
    {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
  Order.transitive(Nat.add(U32.to_nat(base),U32.to_nat(local)),Nat.add(U32.to_nat(base),U32.to_nat(height)),U32.to_nat(capacity),
    Order.offset_le(U32.to_nat(base),U32.to_nat(local),U32.to_nat(height),local_room),endpoint_room)

# The rows of a range [base + first, base + end) clipped at its first row.
def rows_span(+base: U32,+first: U32,+end: U32,+height: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +endpoint_room: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool},
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(height)) == True{} : Bool},+end_room: {Nat.is_le(U32.to_nat(end),U32.to_nat(height)) == True{} : Bool}) ->
  {U32.to_nat(BandWindow.subtract_clipped((base + end : U32),(base + first : U32))) ==
    Nat.sub(Nat.add(U32.to_nat(base),U32.to_nat(end)),Nat.add(U32.to_nat(base),U32.to_nat(first))) : Nat}:
  %shifted(base,end,height,capacity,capacity_ok,endpoint_room,end_room) :
    {U32.to_nat(BandWindow.subtract_clipped((base + end : U32),(base + first : U32))) == Nat.sub(_,Nat.add(U32.to_nat(base),U32.to_nat(first))) : Nat}
  %shifted(base,first,height,capacity,capacity_ok,endpoint_room,first_room) :
    {U32.to_nat(BandWindow.subtract_clipped((base + end : U32),(base + first : U32))) == Nat.sub(U32.to_nat((base + end : U32)),_) : Nat}
  subtract_clipped((base + end : U32),(base + first : U32),capacity,capacity_ok,
    shifted_room(base,end,height,capacity,capacity_ok,endpoint_room,end_room),shifted_room(base,first,height,capacity,capacity_ok,endpoint_room,first_room))

def raw_sources(~Element: Data,above: Buffer.Model<Element>,core: Buffer.Model<Element>,below: Buffer.Model<Element>,
  +base: U32,+core_first: U32,+core_end: U32,+needed_first: U32,+needed_end: U32,first: U32,rows: U32) -> Model.Model<Element>:
  sources(~Element,above,(base + needed_first : U32),BandWindow.subtract_clipped((base + U32.min(core_first,needed_end) : U32),(base + needed_first : U32)),
    core,first,rows,below,(base + U32.max(core_end,needed_first) : U32),BandWindow.subtract_clipped((base + needed_end : U32),(base + U32.max(core_end,needed_first) : U32)))

def minimum_room(+left: U32,+right: U32,+height: U32,right_room: {Nat.is_le(U32.to_nat(right),U32.to_nat(height)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(U32.min(left,right)),U32.to_nat(height)) == True{} : Bool}:
  minimum_right_below(left,right,height,right_room)

def maximum_room(+left: U32,+right: U32,+height: U32,left_room: {Nat.is_le(U32.to_nat(left),U32.to_nat(height)) == True{} : Bool},
  right_room: {Nat.is_le(U32.to_nat(right),U32.to_nat(height)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(U32.max(left,right)),U32.to_nat(height)) == True{} : Bool}:
  maximum_below(left,right,height,left_room,right_room)

def located_selected(~Element: Data,+above: Buffer.Model<Element>,+core: Buffer.Model<Element>,+below: Buffer.Model<Element>,
  +base: U32,+height: U32,+core_first: U32,+core_end: U32,+rows: U32,+needed_first: U32,+needed_end: U32,+first: U32,+capacity: U32,+row: Nat,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+endpoint_room: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool},+first_room: {Nat.is_le(U32.to_nat(needed_first),U32.to_nat(height)) == True{} : Bool},
  +end_room: {Nat.is_le(U32.to_nat(needed_end),U32.to_nat(height)) == True{} : Bool},+first_exact: {U32.to_nat(first) == Nat.add(U32.to_nat(base),U32.to_nat(core_first)) : Nat},
  +core_exact: {U32.to_nat(core_end) == Nat.add(U32.to_nat(core_first),U32.to_nat(rows)) : Nat},
  +minimum_room: {Nat.is_le(U32.to_nat(U32.min(core_first,needed_end)),U32.to_nat(height)) == True{} : Bool},
  +maximum_room: {Nat.is_le(U32.to_nat(U32.max(core_end,needed_first)),U32.to_nat(height)) == True{} : Bool},
  covered: Covered<U32.to_nat(core_first),U32.to_nat(core_end),U32.to_nat(needed_first),U32.to_nat(needed_end),row>) ->
  BandPlan.Located(Model.layout(~Element,raw_sources(~Element,above,core,below,base,core_first,core_end,needed_first,needed_end,first,rows)),Nat.add(U32.to_nat(base),row)):
  match covered:
    case Above{Shape.Both{lower,upper}}:
      Inl{located_in((base + needed_first : U32),BandWindow.subtract_clipped((base + U32.min(core_first,needed_end) : U32),(base + needed_first : U32)),Nat.add(U32.to_nat(base),row),
        Nat.add(U32.to_nat(base),U32.to_nat(needed_first)),Nat.add(U32.to_nat(base),U32.to_nat(U32.min(core_first,needed_end))),
        shifted(base,needed_first,height,capacity,capacity_ok,endpoint_room,first_room),
        rows_span(base,needed_first,U32.min(core_first,needed_end),height,capacity,capacity_ok,endpoint_room,first_room,minimum_room),
        Order.offset_le(U32.to_nat(base),U32.to_nat(needed_first),row,lower),
        Order.offset_lt(U32.to_nat(base),row,U32.to_nat(U32.min(core_first,needed_end)),
          %Equal.sym(Nat,U32.to_nat(U32.min(core_first,needed_end)),Nat.min(U32.to_nat(core_first),U32.to_nat(needed_end)),U32Extrema.minimum(core_first,needed_end)) : {Nat.is_lt(row,_) == True{} : Bool} upper))}
    case Core{Shape.Both{lower,upper}}:
      Inr{Inl{located_in(first,rows,Nat.add(U32.to_nat(base),row),Nat.add(U32.to_nat(base),U32.to_nat(core_first)),Nat.add(Nat.add(U32.to_nat(base),U32.to_nat(core_first)),U32.to_nat(rows)),first_exact,
        Equal.sym(Nat,Nat.sub(Nat.add(Nat.add(U32.to_nat(base),U32.to_nat(core_first)),U32.to_nat(rows)),Nat.add(U32.to_nat(base),U32.to_nat(core_first))),U32.to_nat(rows),
          Order.subtract_shift(Nat.add(U32.to_nat(base),U32.to_nat(core_first)),U32.to_nat(rows))),
        Order.offset_le(U32.to_nat(base),U32.to_nat(core_first),row,lower),
        %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(base),U32.to_nat(core_first)),U32.to_nat(rows)),Nat.add(U32.to_nat(base),Nat.add(U32.to_nat(core_first),U32.to_nat(rows))),Addition.associative(U32.to_nat(base),U32.to_nat(core_first),U32.to_nat(rows))) : {Nat.is_lt(Nat.add(U32.to_nat(base),row),_) == True{} : Bool}
        Order.offset_lt(U32.to_nat(base),row,Nat.add(U32.to_nat(core_first),U32.to_nat(rows)),
          %core_exact : {Nat.is_lt(row,_) == True{} : Bool} upper))}}
    case Below{Shape.Both{lower,upper}}:
      Inr{Inr{located_in((base + U32.max(core_end,needed_first) : U32),BandWindow.subtract_clipped((base + needed_end : U32),(base + U32.max(core_end,needed_first) : U32)),Nat.add(U32.to_nat(base),row),
        Nat.add(U32.to_nat(base),U32.to_nat(U32.max(core_end,needed_first))),Nat.add(U32.to_nat(base),U32.to_nat(needed_end)),
        shifted(base,U32.max(core_end,needed_first),height,capacity,capacity_ok,endpoint_room,maximum_room),
        rows_span(base,U32.max(core_end,needed_first),needed_end,height,capacity,capacity_ok,endpoint_room,maximum_room,end_room),
        Order.offset_le(U32.to_nat(base),U32.to_nat(U32.max(core_end,needed_first)),row,
          %Equal.sym(Nat,U32.to_nat(U32.max(core_end,needed_first)),Nat.max(U32.to_nat(core_end),U32.to_nat(needed_first)),U32Extrema.maximum(core_end,needed_first)) : {Nat.is_le(_,row) == True{} : Bool} lower),
        Order.offset_lt(U32.to_nat(base),row,U32.to_nat(needed_end),upper))}}

# The source of image row base + row, for a needed row inside the image window.
def located_row(~Element: Data,+above: Buffer.Model<Element>,+core: Buffer.Model<Element>,+below: Buffer.Model<Element>,
  +base: U32,+height: U32,+core_first: U32,+core_end: U32,+rows: U32,+needed_first: U32,+needed_end: U32,+first: U32,+capacity: U32,
  +image: ImageWindow<base,height,core_first,core_end,rows,needed_first,needed_end,first,capacity>,+row: Nat,
  +covered: Covered<U32.to_nat(core_first),U32.to_nat(core_end),U32.to_nat(needed_first),U32.to_nat(needed_end),row>) ->
  BandPlan.Located(Model.layout(~Element,raw_sources(~Element,above,core,below,base,core_first,core_end,needed_first,needed_end,first,rows)),Nat.add(U32.to_nat(base),row)):
  ImageWindow{+capacity_ok,+endpoint_room,+core_room,+first_room,+end_room,+first_exact,+core_exact} = image
  located_selected(~Element,above,core,below,base,height,core_first,core_end,rows,needed_first,needed_end,first,capacity,row,capacity_ok,endpoint_room,first_room,end_room,
    first_exact,core_exact,minimum_room(core_first,needed_end,height,end_room),maximum_room(core_end,needed_first,height,core_room,first_room),covered)

def raw_request(+geometry: Bands.Geometry,+base: U32,+needed_first: U32,+needed_end: U32) -> BandRange.Request:
  BandRange.Request{geometry,(base + needed_first : U32),BandWindow.subtract_clipped((base + needed_end : U32),(base + needed_first : U32)),0}

def raw_row(~Element: Data,+geometry: Bands.Geometry,+above: Buffer.Model<Element>,+core: Buffer.Model<Element>,+below: Buffer.Model<Element>,
  +base: U32,+height: U32,+core_first: U32,+core_end: U32,+rows: U32,+needed_first: U32,+needed_end: U32,+first: U32,+capacity: U32,
  +image: ImageWindow<base,height,core_first,core_end,rows,needed_first,needed_end,first,capacity>,+row: Nat,
  inside: {Nat.is_lt(row,U32.to_nat(BandWindow.subtract_clipped((base + needed_end : U32),(base + needed_first : U32)))) == True{} : Bool}) ->
  BandPlan.Located(Model.layout(~Element,raw_sources(~Element,above,core,below,base,core_first,core_end,needed_first,needed_end,first,rows)),
    BandRange.global_row(raw_request(geometry,base,needed_first,needed_end),row)):
  ImageWindow{+capacity_ok,+endpoint_room,core_room,+first_room,+end_room,first_exact,core_exact} = image
  +origin = U32.to_nat(base)
  +begin = U32.to_nat(needed_first)
  +end = U32.to_nat(needed_end)
  +difference = {
    %common_offset(origin,end,begin) : {Nat.is_lt(row,_) == True{} : Bool}
    %rows_span(base,needed_first,needed_end,height,capacity,capacity_ok,endpoint_room,first_room,end_room) : {Nat.is_lt(row,_) == True{} : Bool}
    inside : {Nat.is_lt(row,Nat.sub(end,begin)) == True{} : Bool}}
  +ordered = difference_positive(end,begin,Order.le_lt_trans(0n,row,Nat.sub(end,begin),Order.zero_le(row),difference))
  +local = Nat.add(begin,row)
  +upper = {
    %Order.restore_shift(begin,end,Bounds.strict_le(begin,end,ordered)) : {Nat.is_lt(local,_) == True{} : Bool}
    Order.offset_lt(begin,row,Nat.sub(end,begin),difference) : {Nat.is_lt(local,end) == True{} : Bool}}
  %Equal.sym(Nat,U32.to_nat((base + needed_first : U32)),Nat.add(origin,begin),shifted(base,needed_first,height,capacity,capacity_ok,endpoint_room,first_room)) :
    BandPlan.Located(Model.layout(~Element,raw_sources(~Element,above,core,below,base,core_first,core_end,needed_first,needed_end,first,rows)),Nat.add(_,row))
  %Equal.sym(Nat,Nat.add(Nat.add(origin,begin),row),Nat.add(origin,local),Addition.associative(origin,begin,row)) :
    BandPlan.Located(Model.layout(~Element,raw_sources(~Element,above,core,below,base,core_first,core_end,needed_first,needed_end,first,rows)),_)
  located_row(~Element,above,core,below,base,height,core_first,core_end,rows,needed_first,needed_end,first,capacity,image,local,
    partition(U32.to_nat(core_first),U32.to_nat(core_end),begin,end,local,Shape.Both{Order.le_add(begin,row),upper}))

def raw_rows(~Element: Data,remaining: Nat,+start: Nat,+geometry: Bands.Geometry,+above: Buffer.Model<Element>,+core: Buffer.Model<Element>,+below: Buffer.Model<Element>,
  +base: U32,+height: U32,+core_first: U32,+core_end: U32,+rows: U32,+needed_first: U32,+needed_end: U32,+first: U32,+capacity: U32,
  +image: ImageWindow<base,height,core_first,core_end,rows,needed_first,needed_end,first,capacity>,
  +room: {Nat.is_le(Nat.add(start,remaining),U32.to_nat(BandWindow.subtract_clipped((base + needed_end : U32),(base + needed_first : U32)))) == True{} : Bool}) ->
  Members.All(Nat,{row => BandPlan.Located(Model.layout(~Element,raw_sources(~Element,above,core,below,base,core_first,core_end,needed_first,needed_end,first,rows)),
    BandRange.global_row(raw_request(geometry,base,needed_first,needed_end),row)) : Nat -> Data},Ranges.range(remaining,start)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +count = U32.to_nat(BandWindow.subtract_clipped((base + needed_end : U32),(base + needed_first : U32)))
      Shape.Both{raw_row(~Element,geometry,above,core,below,base,height,core_first,core_end,rows,needed_first,needed_end,first,capacity,image,start,Order.room_index(start,rest,count,room)),
        raw_rows(~Element,rest,1n+start,geometry,above,core,below,base,height,core_first,core_end,rows,needed_first,needed_end,first,capacity,image,Order.advance_room(start,rest,count,room))}

def raw_locations(~Element: Data,+geometry: Bands.Geometry,+above: Buffer.Model<Element>,+core: Buffer.Model<Element>,+below: Buffer.Model<Element>,
  +base: U32,+height: U32,+core_first: U32,+core_end: U32,+rows: U32,+needed_first: U32,+needed_end: U32,+first: U32,+capacity: U32,
  +image: ImageWindow<base,height,core_first,core_end,rows,needed_first,needed_end,first,capacity>) ->
  BandRange.RowLocations(~Element,raw_request(geometry,base,needed_first,needed_end),raw_sources(~Element,above,core,below,base,core_first,core_end,needed_first,needed_end,first,rows)):
  +count = U32.to_nat(BandWindow.subtract_clipped((base + needed_end : U32),(base + needed_first : U32)))
  raw_rows(~Element,count,0n,geometry,above,core,below,base,height,core_first,core_end,rows,needed_first,needed_end,first,capacity,image,Order.reflexive(count))

# Every window row of a leaf has a source among its halos and core.
def Sourced(~Element: Data,+geometry: Bands.Geometry,window: BandWindow.RowWindow,above: Buffer.Model<Element>,core: Buffer.Model<Element>,below: Buffer.Model<Element>,
  first: U32,rows: U32) -> Data:
  match window:
    case BandWindow.RowWindow{output,BandWindow.Rows{above_first,above_rows},BandWindow.Rows{below_first,below_rows},BandWindow.Rows{window_first,window_rows}}:
      BandRange.RowLocations(~Element,BandRange.Request{geometry,window_first,window_rows,0},
        sources(~Element,above,above_first,above_rows,core,first,rows,below,below_first,below_rows))

def nonempty_sources(~Element: Data,active: Bool,+geometry: Bands.Geometry,+above: Buffer.Model<Element>,+core: Buffer.Model<Element>,+below: Buffer.Model<Element>,
  +height: U32,+image: U32,+core_first: U32,+core_end: U32,+needed_first: U32,+needed_end: U32,+output: BandWindow.Rows,+first: U32,+rows: U32,
  facts: Sourced(~Element,geometry,raw_window(height,image,core_first,core_end,needed_first,needed_end,output),above,core,below,first,rows)) ->
  Sourced(~Element,geometry,BandWindow.nonempty(active,raw_window(height,image,core_first,core_end,needed_first,needed_end,output)),above,core,below,first,rows):
  match active:
    case True{}: facts
    case False{}: Unit{}

def window_sources(~Element: Data,+images: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_height: U32,+output_width: U32,
  +first: U32,+rows: U32,+above: Buffer.Model<Element>,+core: Buffer.Model<Element>,+below: Buffer.Model<Element>,+limits: Machine.Limits,
  +coordinates: CoreGeometry<images,height,first,rows,Machine.capacity(limits)>) ->
  Sourced(~Element,Bands.Geometry{images,channels,height,width},
    BandWindow.window(BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width},first,rows),above,core,below,first,rows):
  CoreGeometry{image_inside,+core_inside,+first_exact,+core_exact,+endpoint_room} = coordinates
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +image = (first / height : U32)
  +core_first = (first % height : U32)
  +core_end = (core_first + rows : U32)
  +out_first = U32.min(output_height,BandWindow.ceiling(core_first,stride))
  +out_end = Bool.pick(U32,U32.is_eq(core_end,height),output_height,U32.min(output_height,BandWindow.ceiling(core_end,stride)))
  +needed = U32.min(height,BandWindow.subtract_clipped((out_first * stride : U32),padding))
  +needed_first = BandWindow.start(height,out_first,stride,padding)
  +needed_end = U32.min(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(out_end,1) * stride + kernel : U32),padding))
  +output = {BandWindow.Rows{(image * output_height + out_first : U32),BandWindow.subtract_clipped(out_end,out_first)} : BandWindow.Rows}
  +first_room = Order.transitive(U32.to_nat(needed_first),U32.to_nat(needed),U32.to_nat(height),clipped_below(needed,(needed % stride : U32)),
    minimum_below(height,BandWindow.subtract_clipped((out_first * stride : U32),padding)))
  +end_room = minimum_below(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(out_end,1) * stride + kernel : U32),padding))
  +base = (image * height : U32)
  nonempty_sources(~Element,U32.is_lt(out_first,out_end),geometry,above,core,below,height,image,core_first,core_end,needed_first,needed_end,output,first,rows,
    raw_locations(~Element,geometry,above,core,below,base,height,core_first,core_end,rows,needed_first,needed_end,first,Machine.capacity(limits),
      ImageWindow{Limits.capacity_fits(limits),endpoint_room,core_inside,first_room,end_room,first_exact,core_exact}))

# The window rows of a leaf are an allocatable range of its image.
def WindowFacts(+geometry: Bands.Geometry,window: BandWindow.RowWindow,+capacity: U32,+division_capacity: U32) -> Data:
  match window:
    case BandWindow.RowWindow{output,above,below,rows}: BandHalo.RangeFacts(geometry,rows,capacity,division_capacity)

def rows_nonempty_fits(active: Bool,+images: U32,+channels: U32,+height: U32,+width: U32,+image: U32,+core_first: U32,+core_end: U32,+needed_first: U32,+needed_end: U32,
  +output: BandWindow.Rows,+limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,
  facts: WindowFacts(Bands.Geometry{images,channels,height,width},raw_window(height,image,core_first,core_end,needed_first,needed_end,output),capacity,Machine.half_word(limits))) ->
  WindowFacts(Bands.Geometry{images,channels,height,width},BandWindow.nonempty(active,raw_window(height,image,core_first,core_end,needed_first,needed_end,output)),capacity,Machine.half_word(limits)):
  match active:
    case True{}: facts
    case False{}:
      empty_range(images,channels,height,width,(image * height + needed_first : U32),
        BandWindow.subtract_clipped((image * height + needed_end : U32),(image * height + needed_first : U32)),limits,capacity,domain,facts)

def window_fits(+images: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_height: U32,+output_width: U32,
  +first: U32,+rows: U32,+limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,
  coordinates: CoreGeometry<images,height,first,rows,Machine.capacity(limits)>) ->
  WindowFacts(Bands.Geometry{images,channels,height,width},
    BandWindow.window(BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width},first,rows),capacity,Machine.half_word(limits)):
  CoreGeometry{+image_inside,core_inside,first_exact,core_exact,endpoint_room} = coordinates
  +image = (first / height : U32)
  +core_first = (first % height : U32)
  +core_end = (core_first + rows : U32)
  +out_first = U32.min(output_height,BandWindow.ceiling(core_first,stride))
  +out_end = Bool.pick(U32,U32.is_eq(core_end,height),output_height,U32.min(output_height,BandWindow.ceiling(core_end,stride)))
  +needed = U32.min(height,BandWindow.subtract_clipped((out_first * stride : U32),padding))
  +needed_first = BandWindow.start(height,out_first,stride,padding)
  +needed_end = U32.min(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(out_end,1) * stride + kernel : U32),padding))
  +output = {BandWindow.Rows{(image * output_height + out_first : U32),BandWindow.subtract_clipped(out_end,out_first)} : BandWindow.Rows}
  rows_nonempty_fits(U32.is_lt(out_first,out_end),images,channels,height,width,image,core_first,core_end,needed_first,needed_end,output,limits,capacity,domain,
    image_range(images,channels,height,width,image,needed_first,needed_end,limits,capacity,domain,image_inside,
      Order.transitive(U32.to_nat(needed_first),U32.to_nat(needed),U32.to_nat(height),clipped_below(needed,(needed % stride : U32)),
        minimum_below(height,BandWindow.subtract_clipped((out_first * stride : U32),padding))),
      minimum_below(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(out_end,1) * stride + kernel : U32),padding))))

# Each source leaf of a window lies below the machine row capacity.
def leaf_within(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+length: Nat,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,length)) ->
  BandPlan.LeafWithin<first,rows,0n,U32.to_nat(Machine.capacity(limits))>:
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  BandPlan.LeafWithin{Order.zero_le(U32.to_nat(first)),
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height)),U32.to_nat(Machine.capacity(limits)),rows_fit,
      BandGather.rows_room(Bands.Geometry{images,channels,height,width},limits,capacity,domain))}

def sources_within(~Element: Data,+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,
  +above: Buffer.Model<Element>,+above_first: U32,+above_rows: U32,+core: Buffer.Model<Element>,+core_first: U32,+core_rows: U32,
  +below: Buffer.Model<Element>,+below_first: U32,+below_rows: U32,
  above_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},above_first,above_rows,Buffer.length(~Element,above)),
  core_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},core_first,core_rows,Buffer.length(~Element,core)),
  below_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},below_first,below_rows,Buffer.length(~Element,below))) ->
  BandPlan.Within(Model.layout(~Element,sources(~Element,above,above_first,above_rows,core,core_first,core_rows,below,below_first,below_rows)),0n,U32.to_nat(Machine.capacity(limits))):
  Shape.Both{leaf_within(images,channels,height,width,above_first,above_rows,Buffer.length(~Element,above),limits,capacity,domain,above_metadata),
    Shape.Both{leaf_within(images,channels,height,width,core_first,core_rows,Buffer.length(~Element,core),limits,capacity,domain,core_metadata),
      leaf_within(images,channels,height,width,below_first,below_rows,Buffer.length(~Element,below),limits,capacity,domain,below_metadata)}}

# Decode the concrete min/clipped-subtract/multiply halo endpoints and cover
# every valid spatial sample. Output boundaries may be arbitrary: this also
# applies to the final band, whose end includes rows introduced by padding.
def samples_first(height: U32,output_first: U32,stride: U32,padding: U32) -> U32:
  U32.min(height,BandWindow.subtract_clipped((output_first * stride : U32),padding))

def samples_end(height: U32,output_end: U32,stride: U32,kernel: U32,padding: U32) -> U32:
  U32.min(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(output_end,1) * stride + kernel : U32),padding))

# The core band's output rows, as computed by BandWindow.window.
def output_first(height: U32,output_height: U32,+stride: U32,samples_first: U32) -> U32:
  U32.min(output_height,BandWindow.ceiling((samples_first % height : U32),stride))

def output_end(+height: U32,+output_height: U32,stride: U32,samples_first: U32,rows: U32) -> U32:
  +core_end = (samples_first % height + rows : U32)
  Bool.pick(U32,U32.is_eq(core_end,height),output_height,U32.min(output_height,BandWindow.ceiling(core_end,stride)))

def clipped_decoded(+height: U32,+position: U32,+padding: U32,+coordinate: Nat,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(position) == coordinate : Nat},room: {Nat.is_le(coordinate,U32.to_nat(capacity)) == True{} : Bool},padding_room: {Nat.is_le(U32.to_nat(padding),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(U32.min(height,BandWindow.subtract_clipped(position,padding))) == Nat.min(U32.to_nat(height),Nat.sub(coordinate,U32.to_nat(padding))) : Nat}:
  Equal.trans(Nat,U32.to_nat(U32.min(height,BandWindow.subtract_clipped(position,padding))),Nat.min(U32.to_nat(height),U32.to_nat(BandWindow.subtract_clipped(position,padding))),Nat.min(U32.to_nat(height),Nat.sub(coordinate,U32.to_nat(padding))),
    U32Extrema.minimum(height,BandWindow.subtract_clipped(position,padding)),
    Equal.cong(Nat,Nat,value => Nat.min(U32.to_nat(height),value),U32.to_nat(BandWindow.subtract_clipped(position,padding)),Nat.sub(coordinate,U32.to_nat(padding)),
      Equal.trans(Nat,U32.to_nat(BandWindow.subtract_clipped(position,padding)),Nat.sub(U32.to_nat(position),U32.to_nat(padding)),Nat.sub(coordinate,U32.to_nat(padding)),
        subtract_clipped(position,padding,capacity,capacity_ok,{%Equal.sym(Nat,U32.to_nat(position),coordinate,exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} room : {Nat.is_le(U32.to_nat(position),U32.to_nat(capacity)) == True{} : Bool}},padding_room),
        Equal.cong(Nat,Nat,value => Nat.sub(value,U32.to_nat(padding)),U32.to_nat(position),coordinate,exact))))

def first_decoded(+height: U32,+output_first: U32,+stride: U32,+padding: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +product_room: {Nat.is_le(Nat.mul(U32.to_nat(output_first),U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},padding_room: {Nat.is_le(U32.to_nat(padding),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(samples_first(height,output_first,stride,padding)) == needed_first(U32.to_nat(height),U32.to_nat(output_first),U32.to_nat(stride),U32.to_nat(padding)) : Nat}:
  clipped_decoded(height,(output_first * stride : U32),padding,Nat.mul(U32.to_nat(output_first),U32.to_nat(stride)),capacity,capacity_ok,
    Arithmetic.multiply(output_first,stride,capacity,capacity_ok,product_room),product_room,padding_room)

def last_coordinate(+output_end: U32,+stride: U32,+kernel: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  end_room: {Nat.is_le(U32.to_nat(output_end),U32.to_nat(capacity)) == True{} : Bool},one_room: {Nat.is_le(1n,U32.to_nat(capacity)) == True{} : Bool},
  +window_room: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(U32.to_nat(output_end),1n),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((BandWindow.subtract_clipped(output_end,1) * stride + kernel : U32)) == Nat.add(Nat.mul(Nat.sub(U32.to_nat(output_end),1n),U32.to_nat(stride)),U32.to_nat(kernel)) : Nat}:
  +last = BandWindow.subtract_clipped(output_end,1)
  +last_exact = subtract_clipped(output_end,1,capacity,capacity_ok,end_room,one_room)
  +product = Nat.mul(Nat.sub(U32.to_nat(output_end),1n),U32.to_nat(stride))
  +product_exact = Equal.trans(Nat,U32.to_nat((last * stride : U32)),Nat.mul(U32.to_nat(last),U32.to_nat(stride)),product,
    Arithmetic.multiply(last,stride,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(last),Nat.sub(U32.to_nat(output_end),1n),last_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool}
      Order.transitive(product,Nat.add(product,U32.to_nat(kernel)),U32.to_nat(capacity),Order.le_add(product,U32.to_nat(kernel)),window_room)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(stride)),U32.to_nat(last),Nat.sub(U32.to_nat(output_end),1n),last_exact))
  Equal.trans(Nat,U32.to_nat((last * stride + kernel : U32)),Nat.add(U32.to_nat((last * stride : U32)),U32.to_nat(kernel)),Nat.add(product,U32.to_nat(kernel)),
    Arithmetic.add((last * stride : U32),kernel,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat((last * stride : U32)),product,product_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(kernel)),U32.to_nat(capacity)) == True{} : Bool} window_room),
    Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(kernel)),U32.to_nat((last * stride : U32)),product,product_exact))

def end_decoded(+height: U32,+output_end: U32,+stride: U32,+kernel: U32,+padding: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  end_room: {Nat.is_le(U32.to_nat(output_end),U32.to_nat(capacity)) == True{} : Bool},one_room: {Nat.is_le(1n,U32.to_nat(capacity)) == True{} : Bool},
  +window_room: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(U32.to_nat(output_end),1n),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(capacity)) == True{} : Bool},padding_room: {Nat.is_le(U32.to_nat(padding),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(samples_end(height,output_end,stride,kernel,padding)) == needed_end(U32.to_nat(height),U32.to_nat(output_end),U32.to_nat(stride),U32.to_nat(kernel),U32.to_nat(padding)) : Nat}:
  clipped_decoded(height,(BandWindow.subtract_clipped(output_end,1) * stride + kernel : U32),padding,Nat.add(Nat.mul(Nat.sub(U32.to_nat(output_end),1n),U32.to_nat(stride)),U32.to_nat(kernel)),capacity,capacity_ok,
    last_coordinate(output_end,stride,kernel,capacity,capacity_ok,end_room,one_room,window_room),window_room,padding_room)

def samples_point(+height: U32,+output_first: U32,+output_end: U32,+stride: U32,+kernel: U32,+padding: U32,+output_row: Nat,+kernel_row: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +end_room: {Nat.is_le(U32.to_nat(output_end),U32.to_nat(capacity)) == True{} : Bool},
  +window_room: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(U32.to_nat(output_end),1n),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(capacity)) == True{} : Bool},+padding_room: {Nat.is_le(U32.to_nat(padding),U32.to_nat(capacity)) == True{} : Bool},
  +output: Contains(U32.to_nat(output_first),U32.to_nat(output_end),output_row),kernel_inside: {Nat.is_lt(kernel_row,U32.to_nat(kernel)) == True{} : Bool},
  unpadded: {Nat.is_le(U32.to_nat(padding),Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row)) == True{} : Bool},
  image_inside: {Nat.is_lt(Nat.sub(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding)),U32.to_nat(height)) == True{} : Bool}) ->
  Contains(U32.to_nat(samples_first(height,output_first,stride,padding)),U32.to_nat(samples_end(height,output_end,stride,kernel,padding)),Nat.sub(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding))):
  Shape.Both{first_before,+end_after} = output
  +product = Nat.mul(Nat.sub(U32.to_nat(output_end),1n),U32.to_nat(stride))
  +first_room = Order.transitive(Nat.mul(U32.to_nat(output_first),U32.to_nat(stride)),product,U32.to_nat(capacity),
    RowMajor.mul_le(U32.to_nat(output_first),Nat.sub(U32.to_nat(output_end),1n),U32.to_nat(stride),Order.transitive(U32.to_nat(output_first),output_row,Nat.sub(U32.to_nat(output_end),1n),first_before,before_last(output_row,U32.to_nat(output_end),end_after))),
    Order.transitive(product,Nat.add(product,U32.to_nat(kernel)),U32.to_nat(capacity),Order.le_add(product,U32.to_nat(kernel)),window_room))
  +one_room = Order.transitive(1n,U32.to_nat(output_end),U32.to_nat(capacity),Order.strict_follow(0n,U32.to_nat(output_end),Order.le_lt_trans(0n,output_row,U32.to_nat(output_end),Order.zero_le(output_row),end_after)),end_room)
  %Equal.sym(Nat,U32.to_nat(samples_first(height,output_first,stride,padding)),needed_first(U32.to_nat(height),U32.to_nat(output_first),U32.to_nat(stride),U32.to_nat(padding)),first_decoded(height,output_first,stride,padding,capacity,capacity_ok,first_room,padding_room)) :
    Contains(_,U32.to_nat(samples_end(height,output_end,stride,kernel,padding)),Nat.sub(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding)))
  %Equal.sym(Nat,U32.to_nat(samples_end(height,output_end,stride,kernel,padding)),needed_end(U32.to_nat(height),U32.to_nat(output_end),U32.to_nat(stride),U32.to_nat(kernel),U32.to_nat(padding)),end_decoded(height,output_end,stride,kernel,padding,capacity,capacity_ok,end_room,one_room,window_room,padding_room)) :
    Contains(needed_first(U32.to_nat(height),U32.to_nat(output_first),U32.to_nat(stride),U32.to_nat(padding)),_,Nat.sub(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding)))
  sample(U32.to_nat(height),U32.to_nat(output_first),U32.to_nat(output_end),U32.to_nat(stride),U32.to_nat(kernel),U32.to_nat(padding),output_row,kernel_row,output,kernel_inside,unpadded,image_inside)

# Output row boundaries use the actual ceil/min operations. A final core ends
# at the full output height; every other boundary is shared with its neighbour.
def boundary(output_height: U32,value: U32,stride: U32) -> U32:
  U32.min(output_height,BandWindow.ceiling(value,stride))

def boundaries_decoded(+output_height: U32,+value: U32,+stride: U32,+capacity: U32,+division_capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  division_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},divisor_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(division_capacity)) == True{} : Bool},
  positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},value_room: {Nat.is_le(U32.to_nat(value),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(boundary(output_height,value,stride)) == Nat.min(U32.to_nat(output_height),Ceiling.ceiling(U32.to_nat(value),U32.to_nat(stride))) : Nat}:
  Equal.trans(Nat,U32.to_nat(boundary(output_height,value,stride)),Nat.min(U32.to_nat(output_height),U32.to_nat(BandWindow.ceiling(value,stride))),
    Nat.min(U32.to_nat(output_height),Ceiling.ceiling(U32.to_nat(value),U32.to_nat(stride))),U32Extrema.minimum(output_height,BandWindow.ceiling(value,stride)),
    Equal.cong(Nat,Nat,count => Nat.min(U32.to_nat(output_height),count),U32.to_nat(BandWindow.ceiling(value,stride)),Ceiling.ceiling(U32.to_nat(value),U32.to_nat(stride)),
      ceiling(value,stride,capacity,division_capacity,capacity_ok,division_ok,divisor_room,positive,value_room)))

def below(+output_height: U32,+value: U32,+stride: U32) -> {Nat.is_le(U32.to_nat(boundary(output_height,value,stride)),U32.to_nat(output_height)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(boundary(output_height,value,stride)),Nat.min(U32.to_nat(output_height),U32.to_nat(BandWindow.ceiling(value,stride))),U32Extrema.minimum(output_height,BandWindow.ceiling(value,stride))) : {Nat.is_le(_,U32.to_nat(output_height)) == True{} : Bool}
  Extrema.minimum_left(U32.to_nat(output_height),U32.to_nat(BandWindow.ceiling(value,stride)))

def boundaries_end(+output_height: U32,value: U32,stride: U32,last: Bool) -> U32:
  Bool.pick(U32,last,output_height,boundary(output_height,value,stride))

def end_below(+output_height: U32,+value: U32,+stride: U32,last: Bool) -> {Nat.is_le(U32.to_nat(boundaries_end(output_height,value,stride,last)),U32.to_nat(output_height)) == True{} : Bool}:
  match last:
    case True{}: Order.reflexive(U32.to_nat(output_height))
    case False{}: below(output_height,value,stride)

# An empty core never owns an output row when the image height is positive.
# This establishes the positive-core premise only on an active branch.
def activity_active(+height: U32,+output_height: U32,+stride: U32,+first: U32,rows: U32) -> Bool:
  U32.is_lt(output_first(height,output_height,stride,first),output_end(height,output_height,stride,first,rows))

def empty(+height: U32,+output_height: U32,+stride: U32,+first: U32,+division_capacity: U32,
  +capacity_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  +height_room: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(division_capacity)) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool}) -> {activity_active(height,output_height,stride,first,0) == False{} : Bool}:
  +remainder = (first % height : U32)
  +not_last = {
    %Equal.sym(Bool,U32.is_eq(remainder,height),Nat.is_eq(U32.to_nat(remainder),U32.to_nat(height)),Comparison.eq(remainder,height)) : {_ == False{} : Bool}
    %Equal.sym(Nat,U32.to_nat(remainder),Nat.mod(U32.to_nat(first),U32.to_nat(height)),U32Division.modulo(first,height,division_capacity,capacity_ok,height_room,height_positive)) :
      {Nat.is_eq(_,U32.to_nat(height)) == False{} : Bool}
    Order.less_not_equal(Nat.mod(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(height),DivisionOrder.remainder_small(U32.to_nat(first),U32.to_nat(height),height_positive)) : {U32.is_eq(remainder,height) == False{} : Bool}}
  %Equal.sym(U32,(remainder + 0 : U32),remainder,Arithmetic.add_zero(remainder)) :
    {U32.is_lt(output_first(height,output_height,stride,first),Bool.pick(U32,U32.is_eq(_,height),output_height,U32.min(output_height,BandWindow.ceiling(_,stride)))) == False{} : Bool}
  %Equal.sym(Bool,U32.is_eq(remainder,height),False{},not_last) :
    {U32.is_lt(output_first(height,output_height,stride,first),Bool.pick(U32,_,output_height,U32.min(output_height,BandWindow.ceiling(remainder,stride)))) == False{} : Bool}
  +boundary = output_first(height,output_height,stride,first)
  %Equal.sym(Bool,U32.is_lt(boundary,boundary),Nat.is_lt(U32.to_nat(boundary),U32.to_nat(boundary)),Comparison.lt(boundary,boundary)) : {_ == False{} : Bool}
  CursorLaws.less_reflexive(U32.to_nat(boundary))

def positive_decoded(+height: U32,+output_height: U32,+stride: U32,+first: U32,+rows: U32,+division_capacity: U32,
  +capacity_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  +height_room: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(division_capacity)) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+count: Nat,
  +exact: {U32.to_nat(rows) == count : Nat},guard: {activity_active(height,output_height,stride,first,rows) == True{} : Bool}) -> {Nat.is_lt(0n,count) == True{} : Bool}:
  match count:
    case 0n:
      +rows_zero = Equal.trans(U32,rows,U32.from_nat(U32.to_nat(rows)),0,Equal.sym(U32,U32.from_nat(U32.to_nat(rows)),rows,Arithmetic.native_roundtrip(rows)),
        Equal.cong(Nat,U32,value => U32.from_nat(value),U32.to_nat(rows),0n,exact))
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(
        %empty(height,output_height,stride,first,division_capacity,capacity_ok,height_room,height_positive) : {_ == True{} : Bool}
        %rows_zero : {activity_active(height,output_height,stride,first,_) == True{} : Bool}
        guard))
    case 1n+rest: {==}

def positive(+height: U32,+output_height: U32,+stride: U32,+first: U32,+rows: U32,+division_capacity: U32,
  capacity_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  height_room: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(division_capacity)) == True{} : Bool},
  height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},guard: {activity_active(height,output_height,stride,first,rows) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}:
  positive_decoded(height,output_height,stride,first,rows,division_capacity,capacity_ok,height_room,height_positive,U32.to_nat(rows),{==},guard)

# Use the actual output guard to discharge coordinates on active cores.
# Empty cores are legal inputs and need no image-inside premise.
def height_positive(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {BandGather.height_positive(geometry) == True{} : Bool}:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  height_positive

def active_bounds_active(geometry: BandWindow.Geometry,first: U32,rows: U32) -> Bool:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  activity_active(height,output_height,stride,first,rows)

def images(geometry: BandWindow.Geometry) -> U32:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  images

def height(geometry: BandWindow.Geometry) -> U32:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  height

def active_bounds_coordinates(+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+length: Nat,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<BandWindow.input_geometry(geometry),limits,capacity>,
  height_room: {Nat.is_le(U32.to_nat(height(geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length),guard: {active_bounds_active(geometry,first,rows) == True{} : Bool}) ->
  CoreGeometry<images(geometry),height(geometry),first,rows,Machine.capacity(limits)>:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  halo_bounds_coordinates(images,channels,height,width,first,rows,length,limits,capacity,domain,metadata,
    positive(height,output_height,stride,first,rows,Machine.half_word(limits),Limits.half_word_fits(limits),Limits.double_bound(limits,U32.to_nat(height),height_room),
      height_positive(Bands.Geometry{images,channels,height,width},limits,capacity,domain),guard))

def active_bounds_window(+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+limits: Machine.Limits,+capacity: U32,
  domain: BandGather.Domain<BandWindow.input_geometry(geometry),limits,capacity>,active_bounds_coordinates: CoreGeometry<images(geometry),height(geometry),first,rows,Machine.capacity(limits)>) ->
  BandHalo.LeafFacts(BandWindow.input_geometry(geometry),BandWindow.window(geometry,first,rows),capacity,Machine.half_word(limits)):
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  bounded_window(images,channels,height,width,kernel,stride,padding,output_height,output_width,first,rows,limits,capacity,domain,active_bounds_coordinates)

# Decode the actual local window endpoints into the global map's formulas.
# Machine envelopes are explicit; the shape caller must supply them.
type CoreEnd<-height: U32,-start: U32,-rows: U32> is Data:
  CoreEnd{exact: {U32.to_nat((start % height + rows : U32)) == Nat.add(Nat.mod(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(rows)) : Nat},
    inside: {Nat.is_le(U32.to_nat((start % height + rows : U32)),U32.to_nat(height)) == True{} : Bool}}

def core_end(+height: U32,+start: U32,+rows: U32,+capacity: U32,+division_capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+division_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(division_capacity)) == True{} : Bool},
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(capacity)) == True{} : Bool},
  local: {U32.is_le(rows,(height - start % height : U32)) == True{} : Bool}) -> CoreEnd<height,start,rows>:
  +remainder = (start % height : U32)
  +decoded = U32Division.modulo(start,height,division_capacity,division_ok,divisor_room,positive)
  +span = Inspection.local_span(height,start,rows,division_capacity,division_ok,positive,divisor_room,local)
  +exact = Equal.trans(Nat,U32.to_nat((remainder + rows : U32)),Nat.add(U32.to_nat(remainder),U32.to_nat(rows)),Nat.add(Nat.mod(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(rows)),
    Arithmetic.add(remainder,rows,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(remainder),Nat.mod(U32.to_nat(start),U32.to_nat(height)),decoded) : {Nat.is_le(Nat.add(_,U32.to_nat(rows)),U32.to_nat(capacity)) == True{} : Bool}
      Order.transitive(Nat.add(Nat.mod(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(rows)),U32.to_nat(height),U32.to_nat(capacity),span,height_room)),
    Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(rows)),U32.to_nat(remainder),Nat.mod(U32.to_nat(start),U32.to_nat(height)),decoded))
  CoreEnd{exact,
    %Equal.sym(Nat,U32.to_nat((remainder + rows : U32)),Nat.add(Nat.mod(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(rows)),exact) : {Nat.is_le(_,U32.to_nat(height)) == True{} : Bool}
    span}

def output_mapping_first(+height: U32,+output_height: U32,+stride: U32,+start: U32,+capacity: U32,+division_capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+division_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +height_divisor_room: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(division_capacity)) == True{} : Bool},
  +stride_divisor_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(division_capacity)) == True{} : Bool},
  height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(output_first(height,output_height,stride,start)) == BandPlan.local(U32.to_nat(output_height),Nat.mod(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(stride)) : Nat}:
  +remainder = (start % height : U32)
  +decoded = U32Division.modulo(start,height,division_capacity,division_ok,height_divisor_room,height_positive)
  +room = {
    %Equal.sym(Nat,U32.to_nat(remainder),Nat.mod(U32.to_nat(start),U32.to_nat(height)),decoded) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(Nat.mod(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(height),U32.to_nat(capacity),Bounds.strict_le(Nat.mod(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(height),DivisionOrder.remainder_small(U32.to_nat(start),U32.to_nat(height),height_positive)),height_room) :
    {Nat.is_le(U32.to_nat(remainder),U32.to_nat(capacity)) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat(output_first(height,output_height,stride,start)),BandPlan.local(U32.to_nat(output_height),U32.to_nat(remainder),U32.to_nat(stride)),
    BandPlan.local(U32.to_nat(output_height),Nat.mod(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(stride)),
    boundaries_decoded(output_height,remainder,stride,capacity,division_capacity,capacity_ok,division_ok,stride_divisor_room,stride_positive,room),
    Equal.cong(Nat,Nat,row => BandPlan.local(U32.to_nat(output_height),row,U32.to_nat(stride)),U32.to_nat(remainder),Nat.mod(U32.to_nat(start),U32.to_nat(height)),decoded))

def output_mapping_end(+height: U32,+output_height: U32,+stride: U32,+row: U32,+capacity: U32,+division_capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},division_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  divisor_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(division_capacity)) == True{} : Bool},
  positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},room: {Nat.is_le(U32.to_nat(row),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(boundaries_end(output_height,row,stride,U32.is_eq(row,height))) == BandPlan.local_end(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(row)) : Nat}:
  +last = U32.is_eq(row,height)
  %Comparison.eq(row,height) :
    {U32.to_nat(boundaries_end(output_height,row,stride,last)) == Bool.pick(Nat,_,U32.to_nat(output_height),BandPlan.local(U32.to_nat(output_height),U32.to_nat(row),U32.to_nat(stride))) : Nat}
  Equal.trans(Nat,U32.to_nat(boundaries_end(output_height,row,stride,last)),Bool.pick(Nat,last,U32.to_nat(output_height),U32.to_nat(boundary(output_height,row,stride))),
    Bool.pick(Nat,last,U32.to_nat(output_height),BandPlan.local(U32.to_nat(output_height),U32.to_nat(row),U32.to_nat(stride))),
    U32Extrema.selected(last,output_height,boundary(output_height,row,stride)),
    Equal.cong(Nat,Nat,value => Bool.pick(Nat,last,U32.to_nat(output_height),value),U32.to_nat(boundary(output_height,row,stride)),BandPlan.local(U32.to_nat(output_height),U32.to_nat(row),U32.to_nat(stride)),
      boundaries_decoded(output_height,row,stride,capacity,division_capacity,capacity_ok,division_ok,divisor_room,positive,room)))

def first_global(+height: U32,+output_height: U32,+stride: U32,+start: U32,+capacity: U32,+division_capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+division_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +height_divisor_room: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(division_capacity)) == True{} : Bool},
  +stride_divisor_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(division_capacity)) == True{} : Bool},
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(capacity)) == True{} : Bool},
  +output_room: {Nat.is_le(BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(start)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((start / height * output_height + output_first(height,output_height,stride,start) : U32)) ==
    BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(start)) : Nat}:
  +image = (start / height : U32)
  +native_base = (image * output_height : U32)
  +local = output_first(height,output_height,stride,start)
  +natural_image = Nat.div(U32.to_nat(start),U32.to_nat(height))
  +natural_base = Nat.mul(natural_image,U32.to_nat(output_height))
  +natural_local = BandPlan.local(U32.to_nat(output_height),Nat.mod(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(stride))
  +global = BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(start))
  +image_exact = U32Division.divide(start,height,division_capacity,division_ok,height_divisor_room,height_positive)
  +base_exact = Equal.trans(Nat,U32.to_nat(native_base),Nat.mul(U32.to_nat(image),U32.to_nat(output_height)),natural_base,
    Arithmetic.multiply(image,output_height,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(image),natural_image,image_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(output_height)),U32.to_nat(capacity)) == True{} : Bool}
      Order.transitive(natural_base,global,U32.to_nat(capacity),Order.le_add(natural_base,natural_local),output_room)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(output_height)),U32.to_nat(image),natural_image,image_exact))
  +local_exact = output_mapping_first(height,output_height,stride,start,capacity,division_capacity,capacity_ok,division_ok,height_positive,stride_positive,height_divisor_room,stride_divisor_room,height_room)
  +sum_exact = Equal.trans(Nat,Nat.add(U32.to_nat(native_base),U32.to_nat(local)),Nat.add(natural_base,U32.to_nat(local)),global,
    Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(local)),U32.to_nat(native_base),natural_base,base_exact),
    Equal.cong(Nat,Nat,value => Nat.add(natural_base,value),U32.to_nat(local),natural_local,local_exact))
  Equal.trans(Nat,U32.to_nat((native_base + local : U32)),Nat.add(U32.to_nat(native_base),U32.to_nat(local)),global,
    Arithmetic.add(native_base,local,capacity,capacity_ok,
      %Equal.sym(Nat,Nat.add(U32.to_nat(native_base),U32.to_nat(local)),global,sum_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
      output_room),sum_exact)

# This equality includes the last core and terminal zero-row cores. The
# source metadata supplies the local output_mapping_end equation and containment separately.
def natural_end(+height: U32,+output_height: U32,+stride: U32,+start: U32,+rows: U32,+core_end: U32,
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +core_exact: {U32.to_nat(core_end) == Nat.add(Nat.mod(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(rows)) : Nat},
  core_inside: {Nat.is_le(U32.to_nat(core_end),U32.to_nat(height)) == True{} : Bool}) ->
  {BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(start),U32.to_nat(rows))) ==
    Nat.add(Nat.mul(Nat.div(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(output_height)),BandPlan.local_end(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(core_end))) : Nat}:
  +image = Nat.div(U32.to_nat(start),U32.to_nat(height))
  +base = Nat.mul(image,U32.to_nat(height))
  +remainder = Nat.mod(U32.to_nat(start),U32.to_nat(height))
  +position = Nat.add(U32.to_nat(start),U32.to_nat(rows))
  +shifted = Equal.trans(Nat,position,Nat.add(Nat.add(base,remainder),U32.to_nat(rows)),Nat.add(base,Nat.add(remainder,U32.to_nat(rows))),
    Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(rows)),U32.to_nat(start),Nat.add(base,remainder),DivisionOrder.quotient_exact(U32.to_nat(start),U32.to_nat(height),height_positive)),
    Addition.associative(base,remainder,U32.to_nat(rows)))
  +local_shifted = Equal.trans(Nat,position,Nat.add(base,Nat.add(remainder,U32.to_nat(rows))),Nat.add(base,U32.to_nat(core_end)),shifted,
    Equal.cong(Nat,Nat,value => Nat.add(base,value),Nat.add(remainder,U32.to_nat(rows)),U32.to_nat(core_end),Equal.sym(Nat,U32.to_nat(core_end),Nat.add(remainder,U32.to_nat(rows)),core_exact)))
  %Equal.sym(Nat,position,Nat.add(base,U32.to_nat(core_end)),local_shifted) :
    {BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),_) == Nat.add(Nat.mul(image,U32.to_nat(output_height)),BandPlan.local_end(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(core_end))) : Nat}
  BandPlan.endpoint(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),image,U32.to_nat(core_end),height_positive,stride_positive,core_inside)

type LocalMapping<-height: U32,-output_height: U32,-stride: U32,-start: U32,-count: U32> is Data:
  LocalMapping{
    begin_exact: {BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(start)) ==
      Nat.add(Nat.mul(Nat.div(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(output_height)),U32.to_nat(output_first(height,output_height,stride,start))) : Nat},
    end_exact: {BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(start),U32.to_nat(count))) ==
      Nat.add(Nat.mul(Nat.div(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(output_height)),U32.to_nat(output_end(height,output_height,stride,start,count))) : Nat},
    ordered: {Nat.is_le(U32.to_nat(output_first(height,output_height,stride,start)),U32.to_nat(output_end(height,output_height,stride,start,count))) == True{} : Bool}}

def local_mapping(+height: U32,+output_height: U32,+stride: U32,+start: U32,+count: U32,+capacity: U32,+division_capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+division_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +height_divisor_room: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(division_capacity)) == True{} : Bool},
  +stride_divisor_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(division_capacity)) == True{} : Bool},
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(capacity)) == True{} : Bool},
  coordinates: CoreEnd<height,start,count>) -> LocalMapping<height,output_height,stride,start,count>:
  CoreEnd{+core_exact,+core_inside} = coordinates
  +begin = output_first(height,output_height,stride,start)
  +finish = output_end(height,output_height,stride,start,count)
  +core_end = (start % height + count : U32)
  +base = Nat.mul(Nat.div(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(output_height))
  +mapped_begin = BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(start))
  +mapped_end = BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(start),U32.to_nat(count)))
  +begin_local = BandPlan.local(U32.to_nat(output_height),Nat.mod(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(stride))
  +end_local = BandPlan.local_end(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(core_end))
  +begin_shift = Equal.cong(Nat,Nat,value => Nat.add(base,value),begin_local,U32.to_nat(begin),
    Equal.sym(Nat,U32.to_nat(begin),begin_local,output_mapping_first(height,output_height,stride,start,capacity,division_capacity,capacity_ok,division_ok,height_positive,stride_positive,height_divisor_room,stride_divisor_room,height_room)))
  +end_shift = Equal.trans(Nat,mapped_end,Nat.add(base,end_local),Nat.add(base,U32.to_nat(finish)),
    natural_end(height,output_height,stride,start,count,core_end,height_positive,stride_positive,core_exact,core_inside),
    Equal.cong(Nat,Nat,value => Nat.add(base,value),end_local,U32.to_nat(finish),
      Equal.sym(Nat,U32.to_nat(finish),end_local,output_mapping_end(height,output_height,stride,core_end,capacity,division_capacity,capacity_ok,division_ok,stride_divisor_room,stride_positive,
        Order.transitive(U32.to_nat(core_end),U32.to_nat(height),U32.to_nat(capacity),core_inside,height_room)))))
  LocalMapping{begin_shift,end_shift,Order.cancel_offset_le(base,U32.to_nat(begin),U32.to_nat(finish),
    %begin_shift : {Nat.is_le(_,Nat.add(base,U32.to_nat(finish))) == True{} : Bool}
    %end_shift : {Nat.is_le(mapped_begin,_) == True{} : Bool}
    BandPlan.monotone(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(start),Nat.add(U32.to_nat(start),U32.to_nat(count)),height_positive,stride_positive,Order.le_add(U32.to_nat(start),U32.to_nat(count))))}

def rows_mapped(+height: U32,+output_height: U32,+stride: U32,+start: U32,+count: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +output_height_room: {Nat.is_le(U32.to_nat(output_height),U32.to_nat(capacity)) == True{} : Bool},
  mapping: LocalMapping<height,output_height,stride,start,count>) ->
  {U32.to_nat(BandWindow.subtract_clipped(output_end(height,output_height,stride,start,count),output_first(height,output_height,stride,start))) ==
    Nat.sub(BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(start),U32.to_nat(count))),
      BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(start))) : Nat}:
  LocalMapping{+begin_shift,+end_shift,ordered} = mapping
  +begin = output_first(height,output_height,stride,start)
  +finish = output_end(height,output_height,stride,start,count)
  +core_end = (start % height + count : U32)
  +base = Nat.mul(Nat.div(U32.to_nat(start),U32.to_nat(height)),U32.to_nat(output_height))
  +mapped_begin = BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(start))
  +mapped_end = BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(start),U32.to_nat(count)))
  %Equal.sym(Nat,mapped_end,Nat.add(base,U32.to_nat(finish)),end_shift) :
    {U32.to_nat(BandWindow.subtract_clipped(finish,begin)) == Nat.sub(_,mapped_begin) : Nat}
  %Equal.sym(Nat,mapped_begin,Nat.add(base,U32.to_nat(begin)),begin_shift) :
    {U32.to_nat(BandWindow.subtract_clipped(finish,begin)) == Nat.sub(Nat.add(base,U32.to_nat(finish)),_) : Nat}
  Equal.trans(Nat,U32.to_nat(BandWindow.subtract_clipped(finish,begin)),Nat.sub(U32.to_nat(finish),U32.to_nat(begin)),Nat.sub(Nat.add(base,U32.to_nat(finish)),Nat.add(base,U32.to_nat(begin))),
    subtract_clipped(finish,begin,capacity,capacity_ok,
      Order.transitive(U32.to_nat(finish),U32.to_nat(output_height),U32.to_nat(capacity),end_below(output_height,core_end,stride,U32.is_eq(core_end,height)),output_height_room),
      Order.transitive(U32.to_nat(begin),U32.to_nat(output_height),U32.to_nat(capacity),below(output_height,(start % height : U32),stride),output_height_room)),
    Equal.sym(Nat,Nat.sub(Nat.add(base,U32.to_nat(finish)),Nat.add(base,U32.to_nat(begin))),Nat.sub(U32.to_nat(finish),U32.to_nat(begin)),common_offset(base,U32.to_nat(finish),U32.to_nat(begin))))

def rows(+height: U32,+output_height: U32,+stride: U32,+start: U32,+count: U32,+capacity: U32,+division_capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+division_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +height_divisor_room: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(division_capacity)) == True{} : Bool},
  +stride_divisor_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(division_capacity)) == True{} : Bool},
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(capacity)) == True{} : Bool},
  +output_height_room: {Nat.is_le(U32.to_nat(output_height),U32.to_nat(capacity)) == True{} : Bool},
  coordinates: CoreEnd<height,start,count>) ->
  {U32.to_nat(BandWindow.subtract_clipped(output_end(height,output_height,stride,start,count),output_first(height,output_height,stride,start))) ==
    Nat.sub(BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(start),U32.to_nat(count))),
      BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(start))) : Nat}:
  rows_mapped(height,output_height,stride,start,count,capacity,capacity_ok,output_height_room,
    local_mapping(height,output_height,stride,start,count,capacity,division_capacity,capacity_ok,division_ok,height_positive,stride_positive,height_divisor_room,stride_divisor_room,height_room,coordinates))

# The actual window keeps its output rows even when its halo is inactive.
# Clipped subtraction bounds every output count without a nonempty premise.
def output_rows(window: BandWindow.RowWindow) -> U32:
  BandWindow.RowWindow{BandWindow.Rows{first,count},above,below,rows} = window
  count

def output_height(geometry: BandWindow.Geometry) -> U32:
  BandWindow.Geometry{input,kernel,stride,padding,height,width} = geometry
  height

def nonempty_rows(enabled: Bool,+window: BandWindow.RowWindow) ->
  {output_rows(BandWindow.nonempty(enabled,window)) == output_rows(window) : U32}:
  match enabled window:
    case True{} window: {==}
    case False{} BandWindow.RowWindow{BandWindow.Rows{of,oc},BandWindow.Rows{af,ac},BandWindow.Rows{bf,bc},BandWindow.Rows{wf,wc}}: {==}

def bounded(+geometry: BandWindow.Geometry,+first: U32,+rows: U32) ->
  {Nat.is_le(U32.to_nat(output_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(output_height(geometry))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{batch,channels,+height,width},+kernel,+stride,+padding,+oh,ow} = geometry
  +begin = boundary(oh,(first % height : U32),stride)
  +finish = boundaries_end(oh,(first % height + rows : U32),stride,U32.is_eq((first % height + rows : U32),height))
  +raw = {BandWindow.RowWindow{BandWindow.Rows{(first / height * oh + begin : U32),BandWindow.subtract_clipped(finish,begin)},
    BandWindow.clipped((first / height * height + BandWindow.start(height,begin,stride,padding) : U32),
      (first / height * height + U32.min((first % height : U32),U32.min(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(finish,1) * stride + kernel : U32),padding))) : U32)),
    BandWindow.clipped((first / height * height + U32.max((first % height + rows : U32),BandWindow.start(height,begin,stride,padding)) : U32),
      (first / height * height + U32.min(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(finish,1) * stride + kernel : U32),padding)) : U32)),
    BandWindow.clipped((first / height * height + BandWindow.start(height,begin,stride,padding) : U32),
      (first / height * height + U32.min(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(finish,1) * stride + kernel : U32),padding)) : U32))} : BandWindow.RowWindow}
  %Equal.sym(U32,output_rows(BandWindow.nonempty(U32.is_lt(begin,finish),raw)),output_rows(raw),nonempty_rows(U32.is_lt(begin,finish),raw)) :
    {Nat.is_le(U32.to_nat(_),U32.to_nat(oh)) == True{} : Bool}
  Order.transitive(U32.to_nat(BandWindow.subtract_clipped(finish,begin)),U32.to_nat(finish),U32.to_nat(oh),
    clipped_below(finish,begin),end_below(oh,(first % height + rows : U32),stride,U32.is_eq((first % height + rows : U32),height)))
