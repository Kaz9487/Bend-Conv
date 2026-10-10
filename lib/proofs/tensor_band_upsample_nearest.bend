# Decode the actual increasing-coordinate cursor as a mixed-radix address.
# Extent-one axes may be omitted by preparation without changing that address.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../traversal_loop.bend as Loop
import ../tensor_band.bend as Bands
import ../tensor_layout.bend as Layout
import ../tensor.bend as Tensors
import ../tensor_band_schedule.bend as BandSchedule
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_cursor_order.bend as CursorOrder
import ./tensor_band_upsample_samples.bend as BandUpsampleSamples
import ./tensor_band_upsample_execution.bend as BandUpsampleExecution
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_band_shape_count.bend as BandShapeCount
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_count.bend as BandCount
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_inspection.bend as Inspection
import ./tensor_allocation_depth.bend as Depth
import ./nat_to_u32_bounds.bend as Bounds
import ./strided_range_bounds.bend as Strided
import ./storage_address_bounds.bend as Address
import ./tensor_band_values.bend as BandValues
import ./tensor_band_model.bend as Model
import ./storage_tree_model.bend as Tree
import ./write_program_readback.bend as Readback
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./tensor_count_refinement.bend as Count
import ./checked_word_refinement.bend as Checked
import ./machine_limit_bounds.bend as Limits
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_coordinates.bend as BandCoordinates
import ./tensor_shape_refinement.bend as ShapeLaws
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_coordinates.bend as RowCoordinates
import ./word_encoding.bend as WordEncoding
import ./boolean_logic.bend as Boolean
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_full_range.bend as FullRange
import ./u32_division.bend as U32Division
import ./tensor_band_map.bend as BandMap
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_public_refinement.bend as Public
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_global_shape_count.bend as Global
import ./word_multiplication.bend as Multiplication
import ./u32_comparison.bend as Comparison
import ./storage_certificate.bend as Certificate
import ./write_program.bend as Writes
import ../tensor_window.bend as Window
import ./tensor_copy_values.bend as CopyValues

def five_axes(+channels: Nat,+rows: Nat,+scale: Nat,+width: Nat) -> List<&2,CursorModel.Coordinate>:
  [CursorModel.Coordinate{scale,0n,0n},CursorModel.Coordinate{width,0n,1n},CursorModel.Coordinate{scale,0n,0n},CursorModel.Coordinate{rows,0n,width},CursorModel.Coordinate{channels,0n,Nat.mul(rows,width)}]

def five_index(+channel: Nat,+rows: Nat,+scale: Nat,+width: Nat,+row: Nat,+row_repeat: Nat,+column: Nat,+column_repeat: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(channel,rows),row),scale),row_repeat),width),column),scale),column_repeat)

def decoded(axes: List<&2,CursorModel.Coordinate>,+index: Nat) -> Nat:
  match axes:
    case []: 0n
    case CursorModel.Coordinate{+extent,coordinate,stride} <> tail:
      Nat.add(Nat.mul(Nat.mod(index,extent),stride),decoded(tail,Nat.div(index,extent)))

def position_split(+extent: Nat,+coordinate: Nat,+tail: List<&2,CursorModel.Coordinate>) ->
  {CursorModel.position(CursorModel.Coordinate{extent,coordinate,0n} <> tail) == Nat.add(Nat.mul(CursorModel.position(tail),extent),coordinate) : Nat}:
  Equal.trans(Nat,Nat.add(coordinate,Nat.mul(extent,CursorModel.position(tail))),Nat.add(Nat.mul(extent,CursorModel.position(tail)),coordinate),Nat.add(Nat.mul(CursorModel.position(tail),extent),coordinate),
    Order.commutative(coordinate,Nat.mul(extent,CursorModel.position(tail))),Equal.cong(Nat,Nat,value => Nat.add(value,coordinate),Nat.mul(extent,CursorModel.position(tail)),Nat.mul(CursorModel.position(tail),extent),RowMajor.multiply_commutative(extent,CursorModel.position(tail))))

def decoded_address(+axes: List<&2,CursorModel.Coordinate>,valid: CursorLaws.Valid(axes)) -> {CursorModel.address(axes,0n) == decoded(axes,CursorModel.position(axes)) : Nat}:
  match axes:
    case []: {==}
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      Shape.Both{+inside,tail_valid} = valid
      +positive = Order.le_lt_trans(0n,coordinate,extent,Order.zero_le(coordinate),inside)
      +split = position_split(extent,coordinate,tail)
      +quotient = Equal.trans(Nat,Nat.div(CursorModel.position(CursorModel.Coordinate{extent,coordinate,stride} <> tail),extent),
        Nat.div(Nat.add(Nat.mul(CursorModel.position(tail),extent),coordinate),extent),CursorModel.position(tail),
        Equal.cong(Nat,Nat,value => Nat.div(value,extent),CursorModel.position(CursorModel.Coordinate{extent,coordinate,stride} <> tail),Nat.add(Nat.mul(CursorModel.position(tail),extent),coordinate),split),
        BandCoordinates.quotient(CursorModel.position(tail),coordinate,extent,positive,inside))
      +remainder = Equal.trans(Nat,Nat.mod(CursorModel.position(CursorModel.Coordinate{extent,coordinate,stride} <> tail),extent),
        Nat.mod(Nat.add(Nat.mul(CursorModel.position(tail),extent),coordinate),extent),coordinate,
        Equal.cong(Nat,Nat,value => Nat.mod(value,extent),CursorModel.position(CursorModel.Coordinate{extent,coordinate,stride} <> tail),Nat.add(Nat.mul(CursorModel.position(tail),extent),coordinate),split),
        BandCoordinates.remainder(CursorModel.position(tail),coordinate,extent,positive,inside))
      %Equal.sym(Nat,Nat.mod(CursorModel.position(CursorModel.Coordinate{extent,coordinate,stride} <> tail),extent),coordinate,remainder) :
        {CursorModel.address(CursorModel.Coordinate{extent,coordinate,stride} <> tail,0n) == Nat.add(Nat.mul(_,stride),decoded(tail,Nat.div(CursorModel.position(CursorModel.Coordinate{extent,coordinate,stride} <> tail),extent))) : Nat}
      %Equal.sym(Nat,Nat.div(CursorModel.position(CursorModel.Coordinate{extent,coordinate,stride} <> tail),extent),CursorModel.position(tail),quotient) :
        {CursorModel.address(CursorModel.Coordinate{extent,coordinate,stride} <> tail,0n) == Nat.add(Nat.mul(coordinate,stride),decoded(tail,_)) : Nat}
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(coordinate,stride),value),CursorModel.address(tail,0n),decoded(tail,CursorModel.position(tail)),decoded_address(tail,tail_valid))

def selected_decoded(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>,+index: Nat,
  tail_equal: {decoded(CursorModel.advance(tail),Nat.div(index,extent)) == decoded(tail,Nat.div(index,extent)) : Nat}) ->
  {decoded(Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,1n+coordinate,stride} <> tail,CursorModel.Coordinate{extent,0n,stride} <> CursorModel.advance(tail)),index) ==
    decoded(CursorModel.Coordinate{extent,coordinate,stride} <> tail,index) : Nat}:
  match selected:
    case True{}: {==}
    case False{}:
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.mod(index,extent),stride),value),decoded(CursorModel.advance(tail),Nat.div(index,extent)),decoded(tail,Nat.div(index,extent)),tail_equal)

def advance_decoded(+axes: List<&2,CursorModel.Coordinate>,+index: Nat) -> {decoded(CursorModel.advance(axes),index) == decoded(axes,index) : Nat}:
  match axes:
    case []: {==}
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      selected_decoded(Nat.is_lt(1n+coordinate,extent),extent,coordinate,stride,tail,index,advance_decoded(tail,Nat.div(index,extent)))

def iterated_decoded(count: Nat,+axes: List<&2,CursorModel.Coordinate>,+index: Nat) -> {decoded(CursorOrder.iterate(count,axes),index) == decoded(axes,index) : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +remaining:
      Equal.trans(Nat,decoded(CursorOrder.iterate(remaining,CursorModel.advance(axes)),index),decoded(CursorModel.advance(axes),index),decoded(axes,index),iterated_decoded(remaining,CursorModel.advance(axes),index),advance_decoded(axes,index))

def full_axes(source: List<&2,Views.Axis>,reversed: List<&2,CursorModel.Coordinate>) -> List<&2,CursorModel.Coordinate>:
  match source:
    case []: reversed
    case Views.Axis{extent,stride} <> tail: full_axes(tail,CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed)

def strip_axes(axes: List<&2,CursorModel.Coordinate>) -> List<&2,CursorModel.Coordinate>:
  match axes:
    case []: []
    case CursorModel.Coordinate{+extent,coordinate,stride} <> +tail:
      Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,extent),CursorModel.Coordinate{extent,coordinate,stride} <> strip_axes(tail),strip_axes(tail))

def prepared_stripped(source: List<&2,Views.Axis>,+reversed: List<&2,CursorModel.Coordinate>) ->
  {CursorModel.prepare_axes(source,strip_axes(reversed)) == strip_axes(full_axes(source,reversed)) : List<&2,CursorModel.Coordinate>}:
  match source:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      prepared_stripped(tail,CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed)

def full_valid(source: List<&2,Views.Axis>,+reversed: List<&2,CursorModel.Coordinate>,+nonempty: {Shapes.has_zero(Views.axis_extents(source)) == False{} : Bool},+valid: CursorLaws.Valid(reversed)) -> CursorLaws.Valid(full_axes(source,reversed)):
  match source:
    case []: valid
    case Views.Axis{+extent,+stride} <> +tail:
      +head_positive = CountCompletion.positive_word(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty))
      full_valid(tail,CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,
        Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty),Shape.Both{head_positive,valid})

def divide_one(+index: Nat) -> {Nat.div(index,1n) == index : Nat}:
  %Equal.trans(Nat,Nat.add(Nat.mul(index,1n),0n),Nat.mul(index,1n),index,Addition.zero_right(Nat.mul(index,1n)),ShapeLaws.times_one(index)) :
    {Nat.div(_,1n) == index : Nat}
  BandCoordinates.quotient(index,0n,1n,{==},{==})

def remainder_one(+index: Nat) -> {Nat.mod(index,1n) == 0n : Nat}:
  %Equal.trans(Nat,Nat.add(Nat.mul(index,1n),0n),Nat.mul(index,1n),index,Addition.zero_right(Nat.mul(index,1n)),ShapeLaws.times_one(index)) :
    {Nat.mod(_,1n) == 0n : Nat}
  BandCoordinates.remainder(index,0n,1n,{==},{==})

def stripped_head(extent: Nat,+coordinate: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>,+index: Nat,
  +inside: {Nat.is_lt(coordinate,extent) == True{} : Bool},next: @position: Nat -> {decoded(strip_axes(tail),position) == decoded(tail,position) : Nat}) ->
  {decoded(strip_axes(CursorModel.Coordinate{extent,coordinate,stride} <> tail),index) == decoded(CursorModel.Coordinate{extent,coordinate,stride} <> tail,index) : Nat}:
  match extent:
    case 0n:
      Empty.absurd({decoded(strip_axes(CursorModel.Coordinate{0n,coordinate,stride} <> tail),index) == decoded(CursorModel.Coordinate{0n,coordinate,stride} <> tail,index) : Nat},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(coordinate,0n),True{},Equal.sym(Bool,Nat.is_lt(coordinate,0n),False{},Division.lt_zero(coordinate)),inside)))
    case 1n:
      %Equal.sym(Nat,Nat.mod(index,1n),0n,remainder_one(index)) :
        {decoded(strip_axes(tail),index) == Nat.add(Nat.mul(_,stride),decoded(tail,Nat.div(index,1n))) : Nat}
      %Equal.sym(Nat,Nat.div(index,1n),index,divide_one(index)) : {decoded(strip_axes(tail),index) == decoded(tail,_) : Nat}
      next(index)
    case 2n+ +rest:
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.mod(index,2n+rest),stride),value),decoded(strip_axes(tail),Nat.div(index,2n+rest)),decoded(tail,Nat.div(index,2n+rest)),next(Nat.div(index,2n+rest)))

def stripped_decoded(+axes: List<&2,CursorModel.Coordinate>,+index: Nat,valid: CursorLaws.Valid(axes)) -> {decoded(strip_axes(axes),index) == decoded(axes,index) : Nat}:
  match axes:
    case []: {==}
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      Shape.Both{inside,tail_valid} = valid
      stripped_head(extent,coordinate,stride,tail,index,inside,position => stripped_decoded(tail,position,tail_valid))

def prepared_decoded(+axes: List<&2,Views.Axis>,+index: Nat,nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}) ->
  {decoded(CursorModel.prepare_axes(axes,[]),index) == decoded(full_axes(axes,[]),index) : Nat}:
  Equal.trans(Nat,decoded(CursorModel.prepare_axes(axes,[]),index),decoded(strip_axes(full_axes(axes,[])),index),decoded(full_axes(axes,[]),index),
    Equal.cong(List<&2,CursorModel.Coordinate>,Nat,source => decoded(source,index),CursorModel.prepare_axes(axes,[]),strip_axes(full_axes(axes,[])),prepared_stripped(axes,[])),
    stripped_decoded(full_axes(axes,[]),index,full_valid(axes,[],nonempty,Unit{})))

def cursor_decoded(+axes: List<&2,Views.Axis>,+index: Nat,+nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},+inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(Views.View{axes,0}))) == U32.from_nat(decoded(full_axes(axes,[]),index)) : U32}:
  +initial_axes = CursorModel.prepare_axes(axes,[])
  +valid = CursorLaws.prepared_valid(axes,[],Unit{})
  +position = CursorOrder.iterate_position(index,initial_axes,valid,
    %Equal.sym(Nat,CursorModel.position(initial_axes),0n,CursorOrder.prepared_zero(axes,[],{==})) : {Nat.is_lt(Nat.add(_,index),CursorModel.extent_product(initial_axes)) == True{} : Bool}
    %Equal.sym(Nat,CursorModel.extent_product(initial_axes),ShapeLaws.elements(Views.axis_extents(axes)),CursorOrder.initial_size(axes,nonempty)) : {Nat.is_lt(index,_) == True{} : Bool} inside)
  +decoded_actual = Equal.trans(Nat,CursorModel.address(CursorOrder.iterate(index,initial_axes),0n),decoded(CursorOrder.iterate(index,initial_axes),CursorModel.position(CursorOrder.iterate(index,initial_axes))),decoded(CursorOrder.iterate(index,initial_axes),index),
    decoded_address(CursorOrder.iterate(index,initial_axes),CursorOrder.iterate_valid(index,initial_axes,valid)),
    Equal.cong(Nat,Nat,position => decoded(CursorOrder.iterate(index,initial_axes),position),CursorModel.position(CursorOrder.iterate(index,initial_axes)),index,
      Equal.trans(Nat,CursorModel.position(CursorOrder.iterate(index,initial_axes)),Nat.add(CursorModel.position(initial_axes),index),index,position,
        %Equal.sym(Nat,CursorModel.position(initial_axes),0n,CursorOrder.prepared_zero(axes,[],{==})) : {Nat.add(_,index) == index : Nat} {==})))
  Equal.trans(U32,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(Views.View{axes,0}))),U32.from_nat(CursorModel.address(CursorOrder.iterate(index,initial_axes),0n)),U32.from_nat(decoded(full_axes(axes,[]),index)),
    CopyValues.iterated_address(index,initial_axes,0n,valid),Equal.cong(Nat,U32,U32.from_nat,CursorModel.address(CursorOrder.iterate(index,initial_axes),0n),decoded(full_axes(axes,[]),index),
      Equal.trans(Nat,CursorModel.address(CursorOrder.iterate(index,initial_axes),0n),decoded(CursorOrder.iterate(index,initial_axes),index),decoded(full_axes(axes,[]),index),decoded_actual,
        Equal.trans(Nat,decoded(CursorOrder.iterate(index,initial_axes),index),decoded(initial_axes,index),decoded(full_axes(axes,[]),index),iterated_decoded(index,initial_axes,index),prepared_decoded(axes,index,nonempty)))))

def decoded_step(+extent: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>,+quotient: Nat,+remainder: Nat,+positive: {Nat.is_lt(0n,extent) == True{} : Bool},+inside: {Nat.is_lt(remainder,extent) == True{} : Bool}) ->
  {decoded(CursorModel.Coordinate{extent,0n,stride} <> tail,Nat.add(Nat.mul(quotient,extent),remainder)) == Nat.add(Nat.mul(remainder,stride),decoded(tail,quotient)) : Nat}:
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(quotient,extent),remainder),extent),remainder,BandCoordinates.remainder(quotient,remainder,extent,positive,inside)) :
    {Nat.add(Nat.mul(_,stride),decoded(tail,Nat.div(Nat.add(Nat.mul(quotient,extent),remainder),extent))) == Nat.add(Nat.mul(remainder,stride),decoded(tail,quotient)) : Nat}
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(quotient,extent),remainder),extent),quotient,BandCoordinates.quotient(quotient,remainder,extent,positive,inside)) :
    {Nat.add(Nat.mul(remainder,stride),decoded(tail,_)) == Nat.add(Nat.mul(remainder,stride),decoded(tail,quotient)) : Nat}
  {==}

def zero_step(+extent: Nat,+tail: List<&2,CursorModel.Coordinate>,+quotient: Nat,+remainder: Nat,+positive: {Nat.is_lt(0n,extent) == True{} : Bool},+inside: {Nat.is_lt(remainder,extent) == True{} : Bool}) ->
  {decoded(CursorModel.Coordinate{extent,0n,0n} <> tail,Nat.add(Nat.mul(quotient,extent),remainder)) == decoded(tail,quotient) : Nat}:
  Equal.trans(Nat,decoded(CursorModel.Coordinate{extent,0n,0n} <> tail,Nat.add(Nat.mul(quotient,extent),remainder)),Nat.add(Nat.mul(remainder,0n),decoded(tail,quotient)),decoded(tail,quotient),
    decoded_step(extent,0n,tail,quotient,remainder,positive,inside),
    %Equal.sym(Nat,Nat.mul(remainder,0n),0n,RowMajor.multiply_zero_right(remainder)) : {Nat.add(_,decoded(tail,quotient)) == decoded(tail,quotient) : Nat} {==})

def one_step(+extent: Nat,+tail: List<&2,CursorModel.Coordinate>,+quotient: Nat,+remainder: Nat,+positive: {Nat.is_lt(0n,extent) == True{} : Bool},+inside: {Nat.is_lt(remainder,extent) == True{} : Bool}) ->
  {decoded(CursorModel.Coordinate{extent,0n,1n} <> tail,Nat.add(Nat.mul(quotient,extent),remainder)) == Nat.add(remainder,decoded(tail,quotient)) : Nat}:
  Equal.trans(Nat,decoded(CursorModel.Coordinate{extent,0n,1n} <> tail,Nat.add(Nat.mul(quotient,extent),remainder)),Nat.add(Nat.mul(remainder,1n),decoded(tail,quotient)),Nat.add(remainder,decoded(tail,quotient)),
    decoded_step(extent,1n,tail,quotient,remainder,positive,inside),Equal.cong(Nat,Nat,value => Nat.add(value,decoded(tail,quotient)),Nat.mul(remainder,1n),remainder,ShapeLaws.times_one(remainder)))

def last_digit(+extent: Nat,+stride: Nat,+index: Nat,+inside: {Nat.is_lt(index,extent) == True{} : Bool}) ->
  {decoded([CursorModel.Coordinate{extent,0n,stride}],index) == Nat.mul(index,stride) : Nat}:
  +positive = Order.le_lt_trans(0n,index,extent,Order.zero_le(index),inside)
  Equal.trans(Nat,decoded([CursorModel.Coordinate{extent,0n,stride}],index),Nat.add(Nat.mul(index,stride),0n),Nat.mul(index,stride),decoded_step(extent,stride,[],0n,index,positive,inside),Addition.zero_right(Nat.mul(index,stride)))

def five_decoded(+channels: Nat,+rows: Nat,+scale: Nat,+width: Nat,+channel: Nat,+row: Nat,+row_repeat: Nat,+column: Nat,+column_repeat: Nat,
  +channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+row_repeat_inside: {Nat.is_lt(row_repeat,scale) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,width) == True{} : Bool},+column_repeat_inside: {Nat.is_lt(column_repeat,scale) == True{} : Bool}) ->
  {decoded(five_axes(channels,rows,scale,width),five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat)) == RowMajor.index(channel,rows,width,row,column) : Nat}:
  +row_digit = Nat.add(Nat.mul(channel,rows),row)
  +scaled_row = Nat.add(Nat.mul(row_digit,scale),row_repeat)
  +column_digit = Nat.add(Nat.mul(scaled_row,width),column)
  +tail_one = {[CursorModel.Coordinate{width,0n,1n},CursorModel.Coordinate{scale,0n,0n},CursorModel.Coordinate{rows,0n,width},CursorModel.Coordinate{channels,0n,Nat.mul(rows,width)}] : List<&2,CursorModel.Coordinate>}
  +tail_two = {[CursorModel.Coordinate{scale,0n,0n},CursorModel.Coordinate{rows,0n,width},CursorModel.Coordinate{channels,0n,Nat.mul(rows,width)}] : List<&2,CursorModel.Coordinate>}
  +tail_three = {[CursorModel.Coordinate{rows,0n,width},CursorModel.Coordinate{channels,0n,Nat.mul(rows,width)}] : List<&2,CursorModel.Coordinate>}
  +tail_four = {[CursorModel.Coordinate{channels,0n,Nat.mul(rows,width)}] : List<&2,CursorModel.Coordinate>}
  +scale_positive = Order.le_lt_trans(0n,column_repeat,scale,Order.zero_le(column_repeat),column_repeat_inside)
  +width_positive = Order.le_lt_trans(0n,column,width,Order.zero_le(column),column_inside)
  +rows_positive = Order.le_lt_trans(0n,row,rows,Order.zero_le(row),row_inside)
  %Equal.sym(Nat,decoded(five_axes(channels,rows,scale,width),five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat)),decoded(tail_one,column_digit),zero_step(scale,tail_one,column_digit,column_repeat,scale_positive,column_repeat_inside)) :
    {_ == RowMajor.index(channel,rows,width,row,column) : Nat}
  %Equal.sym(Nat,decoded(tail_one,column_digit),Nat.add(column,decoded(tail_two,scaled_row)),one_step(width,tail_two,scaled_row,column,width_positive,column_inside)) :
    {_ == RowMajor.index(channel,rows,width,row,column) : Nat}
  %Equal.sym(Nat,decoded(tail_two,scaled_row),decoded(tail_three,row_digit),zero_step(scale,tail_three,row_digit,row_repeat,scale_positive,row_repeat_inside)) :
    {Nat.add(column,_) == RowMajor.index(channel,rows,width,row,column) : Nat}
  %Equal.sym(Nat,decoded(tail_three,row_digit),Nat.add(Nat.mul(row,width),decoded(tail_four,channel)),decoded_step(rows,width,tail_four,channel,row,rows_positive,row_inside)) :
    {Nat.add(column,_) == RowMajor.index(channel,rows,width,row,column) : Nat}
  %Equal.sym(Nat,decoded(tail_four,channel),Nat.mul(channel,Nat.mul(rows,width)),last_digit(channels,Nat.mul(rows,width),channel,channel_inside)) :
    {Nat.add(column,Nat.add(Nat.mul(row,width),_)) == RowMajor.index(channel,rows,width,row,column) : Nat}
  Equal.trans(Nat,Nat.add(column,Nat.add(Nat.mul(row,width),Nat.mul(channel,Nat.mul(rows,width)))),Nat.add(Nat.mul(row,width),Nat.add(column,Nat.mul(channel,Nat.mul(rows,width)))),RowMajor.index(channel,rows,width,row,column),
    RowMajor.add_swap(column,Nat.mul(row,width),Nat.mul(channel,Nat.mul(rows,width))),
    Equal.trans(Nat,Nat.add(Nat.mul(row,width),Nat.add(column,Nat.mul(channel,Nat.mul(rows,width)))),Nat.add(Nat.mul(row,width),Nat.add(Nat.mul(channel,Nat.mul(rows,width)),column)),RowMajor.index(channel,rows,width,row,column),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(row,width),value),Nat.add(column,Nat.mul(channel,Nat.mul(rows,width))),Nat.add(Nat.mul(channel,Nat.mul(rows,width)),column),Order.commutative(column,Nat.mul(channel,Nat.mul(rows,width)))),
      RowMajor.add_swap(Nat.mul(row,width),Nat.mul(channel,Nat.mul(rows,width)),column)))

def expanded_digit(+quotient: Nat,+width: Nat,+column: Nat,+scale: Nat,+repeat: Nat) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,width),column),scale),repeat) == Nat.add(Nat.mul(quotient,Nat.mul(width,scale)),Nat.add(Nat.mul(column,scale),repeat)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,width),column),scale),repeat),Nat.add(Nat.add(Nat.mul(Nat.mul(quotient,width),scale),Nat.mul(column,scale)),repeat),Nat.add(Nat.mul(quotient,Nat.mul(width,scale)),Nat.add(Nat.mul(column,scale),repeat)),
    Equal.cong(Nat,Nat,value => Nat.add(value,repeat),Nat.mul(Nat.add(Nat.mul(quotient,width),column),scale),Nat.add(Nat.mul(Nat.mul(quotient,width),scale),Nat.mul(column,scale)),RowMajor.distrib(Nat.mul(quotient,width),column,scale)),
    Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(Nat.mul(quotient,width),scale),Nat.mul(column,scale)),repeat),Nat.add(Nat.add(Nat.mul(quotient,Nat.mul(width,scale)),Nat.mul(column,scale)),repeat),Nat.add(Nat.mul(quotient,Nat.mul(width,scale)),Nat.add(Nat.mul(column,scale),repeat)),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.add(value,Nat.mul(column,scale)),repeat),Nat.mul(Nat.mul(quotient,width),scale),Nat.mul(quotient,Nat.mul(width,scale)),RowMajor.assoc(quotient,width,scale)),
      Addition.associative(Nat.mul(quotient,Nat.mul(width,scale)),Nat.mul(column,scale),repeat)))

def five_scaled(+channel: Nat,+rows: Nat,+scale: Nat,+width: Nat,+row: Nat,+row_repeat: Nat,+column: Nat,+column_repeat: Nat) ->
  {five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat) == RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),Nat.add(Nat.mul(row,scale),row_repeat),Nat.add(Nat.mul(column,scale),column_repeat)) : Nat}:
  +scaled_row = Nat.add(Nat.mul(Nat.add(Nat.mul(channel,rows),row),scale),row_repeat)
  +scaled_column = Nat.add(Nat.mul(column,scale),column_repeat)
  Equal.trans(Nat,five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat),Nat.add(Nat.mul(scaled_row,Nat.mul(width,scale)),scaled_column),
    RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),Nat.add(Nat.mul(row,scale),row_repeat),scaled_column),expanded_digit(scaled_row,width,column,scale,column_repeat),
    Equal.trans(Nat,Nat.add(Nat.mul(scaled_row,Nat.mul(width,scale)),scaled_column),Nat.add(Nat.mul(Nat.add(Nat.mul(channel,Nat.mul(rows,scale)),Nat.add(Nat.mul(row,scale),row_repeat)),Nat.mul(width,scale)),scaled_column),
      RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),Nat.add(Nat.mul(row,scale),row_repeat),scaled_column),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,Nat.mul(width,scale)),scaled_column),scaled_row,Nat.add(Nat.mul(channel,Nat.mul(rows,scale)),Nat.add(Nat.mul(row,scale),row_repeat)),expanded_digit(channel,rows,row,scale,row_repeat)),
      RowMajor.mixed(channel,Nat.mul(rows,scale),Nat.mul(width,scale),Nat.add(Nat.mul(row,scale),row_repeat),scaled_column)))

type PairDigits<-value: Nat,-limit: Nat,-base: Nat> is Data:
  PairDigits{quotient: Nat,remainder: Nat,quotient_inside: {Nat.is_lt(quotient,limit) == True{} : Bool},remainder_inside: {Nat.is_lt(remainder,base) == True{} : Bool},
    exact: {value == Nat.add(Nat.mul(quotient,base),remainder) : Nat}}

def pair_digits(+value: Nat,+limit: Nat,+base: Nat,+positive: {Nat.is_lt(0n,base) == True{} : Bool},+inside: {Nat.is_lt(value,Nat.mul(limit,base)) == True{} : Bool}) -> PairDigits<value,limit,base>:
  +exact = DivisionOrder.quotient_exact(value,base,positive)
  PairDigits{Nat.div(value,base),Nat.mod(value,base),RowMajor.row_inside(Nat.div(value,base),limit,base,Nat.mod(value,base),
    %exact : {Nat.is_lt(_,Nat.mul(limit,base)) == True{} : Bool} inside),DivisionOrder.remainder_small(value,base,positive),exact}

type RepeatedParts<-channels: Nat,-rows: Nat,-scale: Nat,-width: Nat,-index: Nat> is Data:
  RepeatedParts{channel: Nat,row: Nat,row_repeat: Nat,column: Nat,column_repeat: Nat,
    channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},row_inside: {Nat.is_lt(row,rows) == True{} : Bool},row_repeat_inside: {Nat.is_lt(row_repeat,scale) == True{} : Bool},
    column_inside: {Nat.is_lt(column,width) == True{} : Bool},column_repeat_inside: {Nat.is_lt(column_repeat,scale) == True{} : Bool},
    exact: {index == five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat) : Nat}}

def repeated_parts_columns(+channels: Nat,+rows: Nat,+scale: Nat,+width: Nat,+index: Nat,+channel: Nat,+row: Nat,+row_repeat: Nat,+scaled_row: Nat,+scaled_column: Nat,
  +channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+repeat_inside: {Nat.is_lt(row_repeat,scale) == True{} : Bool},
  +row_exact: {scaled_row == Nat.add(Nat.mul(row,scale),row_repeat) : Nat},+index_exact: {index == RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),scaled_row,scaled_column) : Nat},
  columns: PairDigits<scaled_column,width,scale>) -> RepeatedParts<channels,rows,scale,width,index>:
  PairDigits{+column,+column_repeat,column_inside,column_repeat_inside,+column_exact} = columns
  RepeatedParts{channel,row,row_repeat,column,column_repeat,channel_inside,row_inside,repeat_inside,column_inside,column_repeat_inside,
    Equal.trans(Nat,index,RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),scaled_row,scaled_column),five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat),index_exact,
      %Equal.sym(Nat,scaled_row,Nat.add(Nat.mul(row,scale),row_repeat),row_exact) :
        {RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),_,scaled_column) == five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat) : Nat}
      %Equal.sym(Nat,scaled_column,Nat.add(Nat.mul(column,scale),column_repeat),column_exact) :
        {RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),Nat.add(Nat.mul(row,scale),row_repeat),_) == five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat) : Nat}
      Equal.sym(Nat,five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat),RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),Nat.add(Nat.mul(row,scale),row_repeat),Nat.add(Nat.mul(column,scale),column_repeat)),five_scaled(channel,rows,scale,width,row,row_repeat,column,column_repeat)))}

def repeated_parts_rows(+channels: Nat,+rows: Nat,+scale: Nat,+width: Nat,+index: Nat,+channel: Nat,+scaled_row: Nat,+scaled_column: Nat,
  +channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},+column_inside: {Nat.is_lt(scaled_column,Nat.mul(width,scale)) == True{} : Bool},+scale_positive: {Nat.is_lt(0n,scale) == True{} : Bool},
  +index_exact: {index == RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),scaled_row,scaled_column) : Nat},rows_parts: PairDigits<scaled_row,rows,scale>) -> RepeatedParts<channels,rows,scale,width,index>:
  PairDigits{+row,+row_repeat,row_inside,repeat_inside,row_exact} = rows_parts
  repeated_parts_columns(channels,rows,scale,width,index,channel,row,row_repeat,scaled_row,scaled_column,channel_inside,row_inside,repeat_inside,row_exact,index_exact,pair_digits(scaled_column,width,scale,scale_positive,column_inside))

def repeated_parts_coordinates(+channels: Nat,+rows: Nat,+scale: Nat,+width: Nat,+index: Nat,+scale_positive: {Nat.is_lt(0n,scale) == True{} : Bool},coordinates: RowCoordinates.Coordinates<channels,Nat.mul(rows,scale),Nat.mul(width,scale),index>) -> RepeatedParts<channels,rows,scale,width,index>:
  RowCoordinates.Coordinates{+channel,+row,+column,channel_inside,row_inside,column_inside,index_exact} = coordinates
  repeated_parts_rows(channels,rows,scale,width,index,channel,row,column,channel_inside,column_inside,scale_positive,index_exact,pair_digits(row,rows,scale,scale_positive,row_inside))

def repeated_parts(+channels: Nat,+rows: Nat,+scale: Nat,+width: Nat,+index: Nat,+scale_positive: {Nat.is_lt(0n,scale) == True{} : Bool},+width_positive: {Nat.is_lt(0n,width) == True{} : Bool},
  +inside: {Nat.is_lt(index,Nat.mul(channels,Nat.mul(Nat.mul(rows,scale),Nat.mul(width,scale)))) == True{} : Bool}) -> RepeatedParts<channels,rows,scale,width,index>:
  repeated_parts_coordinates(channels,rows,scale,width,index,scale_positive,RowCoordinates.locate(channels,Nat.mul(rows,scale),Nat.mul(width,scale),index,CountCompletion.positive_product(width,scale,width_positive,scale_positive),inside))

def natural_nearest(+height: Nat,+width: Nat,+scale: Nat,+index: Nat) -> Nat:
  +output_height = Nat.mul(height,scale)
  +output_width = Nat.mul(width,scale)
  +plane = Nat.mul(output_height,output_width)
  Nat.add(Nat.mul(Nat.div(index,plane),Nat.mul(height,width)),Nat.add(Nat.mul(Nat.div(Nat.div(Nat.mod(index,plane),output_width),scale),width),Nat.div(Nat.mod(index,output_width),scale)))

def nearest_flattened(+channel: Nat,+height: Nat,+width: Nat,+scale: Nat,+row: Nat,+row_repeat: Nat,+column: Nat,+column_repeat: Nat,
  +row_inside: {Nat.is_lt(row,height) == True{} : Bool},+row_repeat_inside: {Nat.is_lt(row_repeat,scale) == True{} : Bool},+column_inside: {Nat.is_lt(column,width) == True{} : Bool},+column_repeat_inside: {Nat.is_lt(column_repeat,scale) == True{} : Bool}) ->
  {natural_nearest(height,width,scale,RowMajor.index(channel,Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(row,scale),row_repeat),Nat.add(Nat.mul(column,scale),column_repeat))) == RowMajor.index(channel,height,width,row,column) : Nat}:
  +output_height = Nat.mul(height,scale)
  +output_width = Nat.mul(width,scale)
  +scaled_row = Nat.add(Nat.mul(row,scale),row_repeat)
  +scaled_column = Nat.add(Nat.mul(column,scale),column_repeat)
  +plane = Nat.mul(output_height,output_width)
  +within = Nat.add(Nat.mul(scaled_row,output_width),scaled_column)
  +index = RowMajor.index(channel,output_height,output_width,scaled_row,scaled_column)
  +height_positive = Order.le_lt_trans(0n,row,height,Order.zero_le(row),row_inside)
  +width_positive = Order.le_lt_trans(0n,column,width,Order.zero_le(column),column_inside)
  +scale_positive = Order.le_lt_trans(0n,column_repeat,scale,Order.zero_le(column_repeat),column_repeat_inside)
  +output_height_positive = CountCompletion.positive_product(height,scale,height_positive,scale_positive)
  +output_width_positive = CountCompletion.positive_product(width,scale,width_positive,scale_positive)
  +plane_positive = CountCompletion.positive_product(output_height,output_width,output_height_positive,output_width_positive)
  +scaled_row_inside = RowMajor.radix(row,height,row_repeat,scale,row_inside,row_repeat_inside)
  +scaled_column_inside = RowMajor.radix(column,width,column_repeat,scale,column_inside,column_repeat_inside)
  +within_inside = RowMajor.radix(scaled_row,output_height,scaled_column,output_width,scaled_row_inside,scaled_column_inside)
  +plane_quotient = BandCoordinates.quotient(channel,within,plane,plane_positive,within_inside)
  +plane_remainder = BandCoordinates.remainder(channel,within,plane,plane_positive,within_inside)
  +row_quotient = BandCoordinates.quotient(scaled_row,scaled_column,output_width,output_width_positive,scaled_column_inside)
  +column_remainder = Equal.trans(Nat,Nat.mod(index,output_width),Nat.mod(Nat.add(Nat.mul(Nat.add(Nat.mul(channel,output_height),scaled_row),output_width),scaled_column),output_width),scaled_column,
    Equal.cong(Nat,Nat,value => Nat.mod(value,output_width),index,Nat.add(Nat.mul(Nat.add(Nat.mul(channel,output_height),scaled_row),output_width),scaled_column),
      Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(channel,output_height),scaled_row),output_width),scaled_column),index,RowMajor.mixed(channel,output_height,output_width,scaled_row,scaled_column))),
    BandCoordinates.remainder(Nat.add(Nat.mul(channel,output_height),scaled_row),scaled_column,output_width,output_width_positive,scaled_column_inside))
  %Equal.sym(Nat,Nat.div(index,plane),channel,plane_quotient) :
    {Nat.add(Nat.mul(_,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.div(Nat.div(Nat.mod(index,plane),output_width),scale),width),Nat.div(Nat.mod(index,output_width),scale))) == RowMajor.index(channel,height,width,row,column) : Nat}
  %Equal.sym(Nat,Nat.mod(index,plane),within,plane_remainder) :
    {Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.div(Nat.div(_,output_width),scale),width),Nat.div(Nat.mod(index,output_width),scale))) == RowMajor.index(channel,height,width,row,column) : Nat}
  %Equal.sym(Nat,Nat.div(within,output_width),scaled_row,row_quotient) :
    {Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.div(_,scale),width),Nat.div(Nat.mod(index,output_width),scale))) == RowMajor.index(channel,height,width,row,column) : Nat}
  %Equal.sym(Nat,Nat.div(scaled_row,scale),row,BandCoordinates.quotient(row,row_repeat,scale,scale_positive,row_repeat_inside)) :
    {Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(_,width),Nat.div(Nat.mod(index,output_width),scale))) == RowMajor.index(channel,height,width,row,column) : Nat}
  %Equal.sym(Nat,Nat.mod(index,output_width),scaled_column,column_remainder) :
    {Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),Nat.div(_,scale))) == RowMajor.index(channel,height,width,row,column) : Nat}
  %Equal.sym(Nat,Nat.div(scaled_column,scale),column,BandCoordinates.quotient(column,column_repeat,scale,scale_positive,column_repeat_inside)) :
    {Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),_)) == RowMajor.index(channel,height,width,row,column) : Nat}
  {==}

def scaled_first(+image: Nat,+height: Nat,+first: Nat,+scale: Nat) ->
  {Nat.mul(Nat.add(Nat.mul(image,height),first),scale) == Nat.add(Nat.mul(image,Nat.mul(height,scale)),Nat.mul(first,scale)) : Nat}:
  Equal.trans(Nat,Nat.mul(Nat.add(Nat.mul(image,height),first),scale),Nat.add(Nat.mul(Nat.mul(image,height),scale),Nat.mul(first,scale)),Nat.add(Nat.mul(image,Nat.mul(height,scale)),Nat.mul(first,scale)),
    RowMajor.distrib(Nat.mul(image,height),first,scale),Equal.cong(Nat,Nat,value => Nat.add(value,Nat.mul(first,scale)),Nat.mul(Nat.mul(image,height),scale),Nat.mul(image,Nat.mul(height,scale)),RowMajor.assoc(image,height,scale)))

def scaled_row_join(+first: Nat,+row: Nat,+repeat: Nat,+scale: Nat) ->
  {Nat.add(Nat.mul(first,scale),Nat.add(Nat.mul(row,scale),repeat)) == Nat.add(Nat.mul(Nat.add(first,row),scale),repeat) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(first,scale),Nat.add(Nat.mul(row,scale),repeat)),Nat.add(Nat.add(Nat.mul(first,scale),Nat.mul(row,scale)),repeat),Nat.add(Nat.mul(Nat.add(first,row),scale),repeat),
    Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(first,scale),Nat.mul(row,scale)),repeat),Nat.add(Nat.mul(first,scale),Nat.add(Nat.mul(row,scale),repeat)),Addition.associative(Nat.mul(first,scale),Nat.mul(row,scale),repeat)),
    Equal.cong(Nat,Nat,value => Nat.add(value,repeat),Nat.add(Nat.mul(first,scale),Nat.mul(row,scale)),Nat.mul(Nat.add(first,row),scale),Equal.sym(Nat,Nat.mul(Nat.add(first,row),scale),Nat.add(Nat.mul(first,scale),Nat.mul(row,scale)),RowMajor.distrib(first,row,scale))))

def global_nearest_parts(+image: Nat,+first: Nat,+rows: Nat,+channels: Nat,+height: Nat,+width: Nat,+scale: Nat,+channel: Nat,+row: Nat,+row_repeat: Nat,+column: Nat,+column_repeat: Nat,
  +first_inside: {Nat.is_lt(first,height) == True{} : Bool},+row_fits: {Nat.is_le(Nat.add(first,rows),height) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+column_inside: {Nat.is_lt(column,width) == True{} : Bool},
  +row_repeat_inside: {Nat.is_lt(row_repeat,scale) == True{} : Bool},+column_repeat_inside: {Nat.is_lt(column_repeat,scale) == True{} : Bool}) ->
  {natural_nearest(height,width,scale,BandCoordinates.global_index(Nat.mul(Nat.add(Nat.mul(image,height),first),scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat))) ==
    BandCoordinates.global_index(Nat.add(Nat.mul(image,height),first),rows,channels,height,width,RowMajor.index(channel,rows,width,row,column)) : Nat}:
  +scaled_row = Nat.add(Nat.mul(row,scale),row_repeat)
  +scaled_column = Nat.add(Nat.mul(column,scale),column_repeat)
  +scale_positive = Order.le_lt_trans(0n,column_repeat,scale,Order.zero_le(column_repeat),column_repeat_inside)
  +scaled_row_inside = RowMajor.radix(row,rows,row_repeat,scale,row_inside,row_repeat_inside)
  +scaled_column_inside = RowMajor.radix(column,width,column_repeat,scale,column_inside,column_repeat_inside)
  +scaled_first_inside = RowMajor.multiply_lt(first,height,scale,first_inside,scale_positive)
  +global_row_inside = BandCoordinates.row_inside(first,rows,height,row,row_inside,row_fits)
  +target = Equal.trans(Nat,
    BandCoordinates.global_index(Nat.mul(Nat.add(Nat.mul(image,height),first),scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat)),
    BandCoordinates.global_index(Nat.add(Nat.mul(image,Nat.mul(height,scale)),Nat.mul(first,scale)),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),scaled_row,scaled_column)),
    RowMajor.index(Nat.add(Nat.mul(image,channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(first,scale),scaled_row),scaled_column),
    %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,height),first),scale),Nat.add(Nat.mul(image,Nat.mul(height,scale)),Nat.mul(first,scale)),scaled_first(image,height,first,scale)) :
      {BandCoordinates.global_index(_,Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat)) ==
        BandCoordinates.global_index(Nat.add(Nat.mul(image,Nat.mul(height,scale)),Nat.mul(first,scale)),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),scaled_row,scaled_column)) : Nat}
    Equal.cong(Nat,Nat,index => BandCoordinates.global_index(Nat.add(Nat.mul(image,Nat.mul(height,scale)),Nat.mul(first,scale)),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),index),
      five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat),RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),scaled_row,scaled_column),five_scaled(channel,rows,scale,width,row,row_repeat,column,column_repeat)),
    BandCoordinates.flattened(image,Nat.mul(first,scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),channel,scaled_row,scaled_column,scaled_first_inside,scaled_row_inside,scaled_column_inside))
  Equal.trans(Nat,natural_nearest(height,width,scale,BandCoordinates.global_index(Nat.mul(Nat.add(Nat.mul(image,height),first),scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat))),
    natural_nearest(height,width,scale,RowMajor.index(Nat.add(Nat.mul(image,channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(Nat.add(first,row),scale),row_repeat),scaled_column)),
    BandCoordinates.global_index(Nat.add(Nat.mul(image,height),first),rows,channels,height,width,RowMajor.index(channel,rows,width,row,column)),
    Equal.trans(Nat,natural_nearest(height,width,scale,BandCoordinates.global_index(Nat.mul(Nat.add(Nat.mul(image,height),first),scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat))),
      natural_nearest(height,width,scale,RowMajor.index(Nat.add(Nat.mul(image,channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(first,scale),scaled_row),scaled_column)),
      natural_nearest(height,width,scale,RowMajor.index(Nat.add(Nat.mul(image,channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(Nat.add(first,row),scale),row_repeat),scaled_column)),
      Equal.cong(Nat,Nat,index => natural_nearest(height,width,scale,index),BandCoordinates.global_index(Nat.mul(Nat.add(Nat.mul(image,height),first),scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat)),
        RowMajor.index(Nat.add(Nat.mul(image,channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(first,scale),scaled_row),scaled_column),target),
      Equal.cong(Nat,Nat,row => natural_nearest(height,width,scale,RowMajor.index(Nat.add(Nat.mul(image,channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),row,scaled_column)),
        Nat.add(Nat.mul(first,scale),scaled_row),Nat.add(Nat.mul(Nat.add(first,row),scale),row_repeat),scaled_row_join(first,row,row_repeat,scale))),
    Equal.trans(Nat,natural_nearest(height,width,scale,RowMajor.index(Nat.add(Nat.mul(image,channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(Nat.add(first,row),scale),row_repeat),scaled_column)),
      RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,Nat.add(first,row),column),
      BandCoordinates.global_index(Nat.add(Nat.mul(image,height),first),rows,channels,height,width,RowMajor.index(channel,rows,width,row,column)),
      nearest_flattened(Nat.add(Nat.mul(image,channels),channel),height,width,scale,Nat.add(first,row),row_repeat,column,column_repeat,global_row_inside,row_repeat_inside,column_inside,column_repeat_inside),
      Equal.sym(Nat,BandCoordinates.global_index(Nat.add(Nat.mul(image,height),first),rows,channels,height,width,RowMajor.index(channel,rows,width,row,column)),RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,Nat.add(first,row),column),
        BandCoordinates.flattened(image,first,rows,channels,height,width,channel,row,column,first_inside,row_inside,column_inside))))

def source_index(+channels: Nat,+rows: Nat,+scale: Nat,+width: Nat,+index: Nat,parts: RepeatedParts<channels,rows,scale,width,index>) -> Nat:
  RepeatedParts{channel,row,row_repeat,column,column_repeat,channel_inside,row_inside,row_repeat_inside,column_inside,column_repeat_inside,exact} = parts
  RowMajor.index(channel,rows,width,row,column)

def global_nearest(+first: Nat,+rows: Nat,+channels: Nat,+height: Nat,+width: Nat,+scale: Nat,+index: Nat,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+local_rows: {Nat.is_le(Nat.add(Nat.mod(first,height),rows),height) == True{} : Bool},parts: RepeatedParts<channels,rows,scale,width,index>) ->
  {natural_nearest(height,width,scale,BandCoordinates.global_index(Nat.mul(first,scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),index)) ==
    BandCoordinates.global_index(first,rows,channels,height,width,source_index(channels,rows,scale,width,index,parts)) : Nat}:
  RepeatedParts{+channel,+row,+row_repeat,+column,+column_repeat,channel_inside,+row_inside,+row_repeat_inside,+column_inside,+column_repeat_inside,+index_exact} = parts
  +absolute = Nat.add(Nat.mul(Nat.div(first,height),height),Nat.mod(first,height))
  %Equal.sym(Nat,first,absolute,DivisionOrder.quotient_exact(first,height,height_positive)) :
    {natural_nearest(height,width,scale,BandCoordinates.global_index(Nat.mul(_,scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),index)) ==
      BandCoordinates.global_index(_,rows,channels,height,width,RowMajor.index(channel,rows,width,row,column)) : Nat}
  %Equal.sym(Nat,index,five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat),index_exact) :
    {natural_nearest(height,width,scale,BandCoordinates.global_index(Nat.mul(absolute,scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),_)) ==
      BandCoordinates.global_index(absolute,rows,channels,height,width,RowMajor.index(channel,rows,width,row,column)) : Nat}
  global_nearest_parts(Nat.div(first,height),Nat.mod(first,height),rows,channels,height,width,scale,channel,row,row_repeat,column,column_repeat,
    DivisionOrder.remainder_small(first,height,height_positive),local_rows,row_inside,column_inside,row_repeat_inside,column_repeat_inside)

def natural_one_left(+value: Nat) -> {Nat.mul(1n,value) == value : Nat}:
  Equal.trans(Nat,Nat.mul(1n,value),Nat.mul(value,1n),value,RowMajor.multiply_commutative(1n,value),ShapeLaws.times_one(value))

def native_one_left(+value: U32) -> {(1 * value : U32) == value : U32}:
  +exact = Equal.trans(Nat,U32.to_nat((1 * value : U32)),Nat.mul(1n,U32.to_nat(value)),U32.to_nat(value),FullRange.multiply(1,value,value,
    %Equal.sym(Nat,Nat.mul(1n,U32.to_nat(value)),U32.to_nat(value),natural_one_left(U32.to_nat(value))) : {Nat.is_le(_,U32.to_nat(value)) == True{} : Bool} Order.reflexive(U32.to_nat(value))),natural_one_left(U32.to_nat(value)))
  Equal.trans(U32,(1 * value : U32),U32.from_nat(U32.to_nat((1 * value : U32))),value,Equal.sym(U32,U32.from_nat(U32.to_nat((1 * value : U32))),(1 * value : U32),Arithmetic.native_roundtrip((1 * value : U32))),
    Equal.trans(U32,U32.from_nat(U32.to_nat((1 * value : U32))),U32.from_nat(U32.to_nat(value)),value,Equal.cong(Nat,U32,U32.from_nat,U32.to_nat((1 * value : U32)),U32.to_nat(value),exact),Arithmetic.native_roundtrip(value)))

def row_stride(+channels: U32,+rows: U32,+width: U32,+count: U32,+exact: {U32.to_nat(count) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},+channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool}) ->
  {U32.to_nat((width * rows : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(width)) : Nat}:
  +plane = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +room = {
    %Equal.sym(Nat,Nat.mul(U32.to_nat(width),U32.to_nat(rows)),plane,RowMajor.multiply_commutative(U32.to_nat(width),U32.to_nat(rows))) : {Nat.is_le(_,U32.to_nat(count)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(count),Nat.mul(U32.to_nat(channels),plane),exact) : {Nat.is_le(plane,_) == True{} : Bool}
    %RowMajor.multiply_commutative(plane,U32.to_nat(channels)) : {Nat.is_le(plane,_) == True{} : Bool}
    RowMajor.product_covers(plane,U32.to_nat(channels),channels_positive) : {Nat.is_le(Nat.mul(U32.to_nat(width),U32.to_nat(rows)),U32.to_nat(count)) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat((width * rows : U32)),Nat.mul(U32.to_nat(width),U32.to_nat(rows)),plane,FullRange.multiply(width,rows,count,room),RowMajor.multiply_commutative(U32.to_nat(width),U32.to_nat(rows)))

type SourceStrides<-channels: U32,-rows: U32,-width: U32,-count: U32> is Data:
  SourceStrides{nonzero: {U32.is_zero(count) == False{} : Bool},row_exact: {U32.to_nat((width * rows : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(width)) : Nat}}

def source_strides(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,input: Buffer.Model<Element>,metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_channels(geometry))) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+width_positive: {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_width(geometry))) == True{} : Bool}) ->
  SourceStrides<BandUpsampleExecution.producer_channels(geometry),rows,BandUpsampleExecution.producer_width(geometry),Bands.local_count(geometry,rows)>:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Buffer.Model{+tree,+length,+certificate} = input
  BandShape.Fields{+local,+rows_fit,+length_exact} = metadata
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
  +metadata = {BandShape.Fields{local,rows_fit,length_exact} : BandShape.Leaf(geometry,first,rows,length)}
  +count = Bands.local_count(geometry,rows)
  +exact = Equal.trans(Nat,U32.to_nat(count),length,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),BandShapeCount.count_refinement_exact(geometry,rows,length,BandShapeCount.counted(~Element,geometry,first,rows,input,metadata,ready)),length_exact)
  SourceStrides{U32Division.nonzero(count,
    %Equal.sym(Nat,U32.to_nat(count),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    CountCompletion.positive_product(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),channels_positive,CountCompletion.positive_product(U32.to_nat(rows),U32.to_nat(width),rows_positive,width_positive))),
    row_stride(channels,rows,width,count,exact,channels_positive)}

def repeated_full_axes(geometry: Bands.Geometry,+rows: U32,+scale: U32,+count: U32,evidence: SourceStrides<BandUpsampleExecution.producer_channels(geometry),rows,BandUpsampleExecution.producer_width(geometry),count>) ->
  {full_axes(BandUpsampleSamples.axes(BandUpsampleExecution.repeated_view(geometry,rows,scale,count)),[]) == five_axes(U32.to_nat(BandUpsampleExecution.producer_channels(geometry)),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(BandUpsampleExecution.producer_width(geometry))) : List<&2,CursorModel.Coordinate>}:
  Bands.Geometry{batch,+channels,height,+width} = geometry
  SourceStrides{+nonzero,+row_exact} = evidence
  %Equal.sym(Bool,U32.is_zero(count),False{},nonzero) :
    {full_axes([Views.Axis{channels,U32.mul(U32.mul(Bool.pick(U32,_,0,1),width),rows)},Views.Axis{rows,U32.mul(Bool.pick(U32,_,0,1),width)},Views.Axis{scale,0},Views.Axis{width,Bool.pick(U32,_,0,1)},Views.Axis{scale,0}],[]) == five_axes(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width)) : List<&2,CursorModel.Coordinate>}
  %Equal.sym(U32,(1 * width : U32),width,native_one_left(width)) :
    {full_axes([Views.Axis{channels,U32.mul(_,rows)},Views.Axis{rows,_},Views.Axis{scale,0},Views.Axis{width,1},Views.Axis{scale,0}],[]) == five_axes(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width)) : List<&2,CursorModel.Coordinate>}
  %Equal.sym(Nat,U32.to_nat((width * rows : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),row_exact) :
    {[CursorModel.Coordinate{U32.to_nat(scale),0n,0n},CursorModel.Coordinate{U32.to_nat(width),0n,1n},CursorModel.Coordinate{U32.to_nat(scale),0n,0n},CursorModel.Coordinate{U32.to_nat(rows),0n,U32.to_nat(width)},CursorModel.Coordinate{U32.to_nat(channels),0n,_}] == five_axes(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width)) : List<&2,CursorModel.Coordinate>}
  {==}

type LocalSampleIndex<-geometry: Bands.Geometry,-rows: U32,-scale: U32,-index: Nat> is Data:
  LocalSampleIndex{parts: RepeatedParts<U32.to_nat(BandUpsampleExecution.producer_channels(geometry)),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(BandUpsampleExecution.producer_width(geometry)),index>,
    address: {CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))))) ==
      U32.from_nat(source_index(U32.to_nat(BandUpsampleExecution.producer_channels(geometry)),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(BandUpsampleExecution.producer_width(geometry)),index,parts)) : U32}}

def repeated_location_parts(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,+scale: U32,+input: Buffer.Model<Element>,+output_count: U32,+index: Nat,
  +descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),output_count>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  +inside: {Nat.is_lt(index,U32.to_nat(output_count)) == True{} : Bool},parts: RepeatedParts<U32.to_nat(BandUpsampleExecution.producer_channels(geometry)),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(BandUpsampleExecution.producer_width(geometry)),index>) -> LocalSampleIndex<geometry,rows,scale,index>:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  RepeatedParts{+channel,+row,+row_repeat,+column,+column_repeat,+channel_inside,+row_inside,+row_repeat_inside,+column_inside,+column_repeat_inside,+index_exact} = parts
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +parts = {RepeatedParts{channel,row,row_repeat,column,column_repeat,channel_inside,row_inside,row_repeat_inside,column_inside,column_repeat_inside,index_exact} : RepeatedParts<U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width),index>}
  +source = BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))
  +axes = BandUpsampleSamples.axes(source)
  +exact = BandUpsampleSamples.repeated_elements_exact(geometry,rows,scale,Bands.local_count(geometry,rows),output_count,descriptor)
  +nonempty = CountCompletion.positive_not_zero(Views.axis_extents(axes),
    %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(output_count),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,index,U32.to_nat(output_count),Order.zero_le(index),inside))
  +axes_equal = repeated_full_axes(geometry,rows,scale,Bands.local_count(geometry,rows),source_strides(~Element,geometry,first,rows,input,metadata,ready,
    Order.le_lt_trans(0n,channel,U32.to_nat(channels),Order.zero_le(channel),channel_inside),Order.le_lt_trans(0n,row,U32.to_nat(rows),Order.zero_le(row),row_inside),
    Order.le_lt_trans(0n,column,U32.to_nat(width),Order.zero_le(column),column_inside)))
  +cursor = cursor_decoded(axes,index,nonempty,
    %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(output_count),exact) : {Nat.is_lt(index,_) == True{} : Bool} inside)
  +decode_parts = Equal.trans(Nat,decoded(five_axes(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width)),index),
    decoded(five_axes(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width)),five_index(channel,U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width),row,row_repeat,column,column_repeat)),
    RowMajor.index(channel,U32.to_nat(rows),U32.to_nat(width),row,column),Equal.cong(Nat,Nat,position => decoded(five_axes(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width)),position),index,
      five_index(channel,U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width),row,row_repeat,column,column_repeat),index_exact),
    five_decoded(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width),channel,row,row_repeat,column,column_repeat,channel_inside,row_inside,row_repeat_inside,column_inside,column_repeat_inside))
  LocalSampleIndex{parts,
    Equal.trans(U32,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(source))),U32.from_nat(decoded(full_axes(axes,[]),index)),U32.from_nat(RowMajor.index(channel,U32.to_nat(rows),U32.to_nat(width),row,column)),cursor,
      Equal.trans(U32,U32.from_nat(decoded(full_axes(axes,[]),index)),U32.from_nat(decoded(five_axes(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width)),index)),U32.from_nat(RowMajor.index(channel,U32.to_nat(rows),U32.to_nat(width),row,column)),
        Equal.cong(List<&2,CursorModel.Coordinate>,U32,source => U32.from_nat(decoded(source,index)),full_axes(axes,[]),five_axes(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width)),axes_equal),
        Equal.cong(Nat,U32,U32.from_nat,decoded(five_axes(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(width)),index),RowMajor.index(channel,U32.to_nat(rows),U32.to_nat(width),row,column),decode_parts)))}

def repeated_location(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+scale: U32,+input: Buffer.Model<Element>,+output_count: U32,+index: Nat,
  +descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),output_count>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  +scale_positive: {Nat.is_lt(0n,U32.to_nat(scale)) == True{} : Bool},+width_positive: {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_width(geometry))) == True{} : Bool},+inside: {Nat.is_lt(index,U32.to_nat(output_count)) == True{} : Bool}) -> LocalSampleIndex<geometry,rows,scale,index>:
  +count_grouped = Equal.trans(Nat,U32.to_nat(output_count),ShapeLaws.elements(BandUpsampleExecution.repeated_shape(geometry,rows,scale)),
    Nat.mul(U32.to_nat(BandUpsampleExecution.producer_channels(geometry)),Nat.mul(Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),Nat.mul(U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale)))),
    Equal.sym(Nat,ShapeLaws.elements(BandUpsampleExecution.repeated_shape(geometry,rows,scale)),U32.to_nat(output_count),BandUpsampleExecution.count_exact(ShapeLaws.elements(BandUpsampleExecution.repeated_shape(geometry,rows,scale)),output_count,BandUpsampleExecution.counted(geometry,rows,scale,Bands.local_count(geometry,rows),output_count,descriptor))),BandUpsampleExecution.repeated_elements(geometry,rows,scale))
  repeated_location_parts(~Element,geometry,first,rows,scale,input,output_count,index,descriptor,metadata,ready,inside,
    repeated_parts(U32.to_nat(BandUpsampleExecution.producer_channels(geometry)),U32.to_nat(rows),U32.to_nat(scale),U32.to_nat(BandUpsampleExecution.producer_width(geometry)),index,scale_positive,width_positive,
      %count_grouped : {Nat.is_lt(index,_) == True{} : Bool} inside))

def flat_word(+channel: U32,+height: U32,+width: U32,+row: U32,+column: U32,+capacity: U32,
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+room: {Nat.is_le(RowMajor.index(U32.to_nat(channel),U32.to_nat(height),U32.to_nat(width),U32.to_nat(row),U32.to_nat(column)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((channel * height * width + row * width + column : U32)) == RowMajor.index(U32.to_nat(channel),U32.to_nat(height),U32.to_nat(width),U32.to_nat(row),U32.to_nat(column)) : Nat}:
  +prefix = Nat.mul(U32.to_nat(channel),U32.to_nat(height))
  +left = Nat.mul(prefix,U32.to_nat(width))
  +right = Nat.mul(U32.to_nat(row),U32.to_nat(width))
  +sum = Nat.add(left,right)
  +expanded = Nat.add(sum,U32.to_nat(column))
  +expanded_room = {
    %Equal.sym(Nat,expanded,RowMajor.index(U32.to_nat(channel),U32.to_nat(height),U32.to_nat(width),U32.to_nat(row),U32.to_nat(column)),RowMajor.expanded(U32.to_nat(channel),U32.to_nat(height),U32.to_nat(width),U32.to_nat(row),U32.to_nat(column))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room : {Nat.is_le(expanded,U32.to_nat(capacity)) == True{} : Bool}}
  +sum_room = Order.transitive(sum,expanded,U32.to_nat(capacity),Order.le_add(sum,U32.to_nat(column)),expanded_room)
  +left_room = Order.transitive(left,sum,U32.to_nat(capacity),Order.le_add(left,right),sum_room)
  +right_room = Order.transitive(right,sum,U32.to_nat(capacity),
    %Order.commutative(right,left) : {Nat.is_le(right,_) == True{} : Bool} Order.le_add(right,left),sum_room)
  +prefix_room = Order.transitive(prefix,left,U32.to_nat(capacity),RowMajor.product_covers(prefix,U32.to_nat(width),width_positive),left_room)
  +prefix_exact = FullRange.multiply(channel,height,capacity,prefix_room)
  +left_exact = {
    %prefix_exact : {U32.to_nat((channel * height * width : U32)) == Nat.mul(_,U32.to_nat(width)) : Nat}
    FullRange.multiply((channel * height : U32),width,capacity,
      %Equal.sym(Nat,U32.to_nat((channel * height : U32)),prefix,prefix_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool} left_room) :
    {U32.to_nat((channel * height * width : U32)) == left : Nat}}
  +right_exact = FullRange.multiply(row,width,capacity,right_room)
  +sum_exact = {
    %left_exact : {U32.to_nat((channel * height * width + row * width : U32)) == Nat.add(_,right) : Nat}
    %right_exact : {U32.to_nat((channel * height * width + row * width : U32)) == Nat.add(U32.to_nat((channel * height * width : U32)),_) : Nat}
    FullRange.add((channel * height * width : U32),(row * width : U32),capacity,
      %Equal.sym(Nat,U32.to_nat((channel * height * width : U32)),left,left_exact) : {Nat.is_le(Nat.add(_,U32.to_nat((row * width : U32))),U32.to_nat(capacity)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat((row * width : U32)),right,right_exact) : {Nat.is_le(Nat.add(left,_),U32.to_nat(capacity)) == True{} : Bool} sum_room) :
    {U32.to_nat((channel * height * width + row * width : U32)) == sum : Nat}}
  Equal.trans(Nat,U32.to_nat((channel * height * width + row * width + column : U32)),expanded,
    RowMajor.index(U32.to_nat(channel),U32.to_nat(height),U32.to_nat(width),U32.to_nat(row),U32.to_nat(column)),
    %sum_exact : {U32.to_nat((channel * height * width + row * width + column : U32)) == Nat.add(_,U32.to_nat(column)) : Nat}
    FullRange.add((channel * height * width + row * width : U32),column,capacity,
      %Equal.sym(Nat,U32.to_nat((channel * height * width + row * width : U32)),sum,sum_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(column)),U32.to_nat(capacity)) == True{} : Bool} expanded_room),
    RowMajor.expanded(U32.to_nat(channel),U32.to_nat(height),U32.to_nat(width),U32.to_nat(row),U32.to_nat(column)))

def nearest_word(geometry: Bands.Geometry,+scale: U32,+index: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +width_exact: {U32.to_nat((BandUpsampleExecution.producer_width(geometry) * scale : U32)) == Nat.mul(U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale)) : Nat},
  +plane_exact: {U32.to_nat(((BandUpsampleExecution.producer_height(geometry) * scale : U32) * (BandUpsampleExecution.producer_width(geometry) * scale : U32) : U32)) == Nat.mul(Nat.mul(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(scale)),Nat.mul(U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale))) : Nat},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_height(geometry))) == True{} : Bool},+width_positive: {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_width(geometry))) == True{} : Bool},+scale_positive: {Nat.is_lt(0n,U32.to_nat(scale)) == True{} : Bool},
  +plane_bound: {Nat.is_le(Nat.double(U32.to_nat(((BandUpsampleExecution.producer_height(geometry) * scale : U32) * (BandUpsampleExecution.producer_width(geometry) * scale : U32) : U32))),U32.to_nat(capacity)) == True{} : Bool},
  +width_bound: {Nat.is_le(Nat.double(U32.to_nat((BandUpsampleExecution.producer_width(geometry) * scale : U32))),U32.to_nat(capacity)) == True{} : Bool},+scale_bound: {Nat.is_le(Nat.double(U32.to_nat(scale)),U32.to_nat(capacity)) == True{} : Bool},
  +room: {Nat.is_le(natural_nearest(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale),U32.to_nat(index)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(BandUpsampleExecution.nearest_index(geometry,scale,index)) == natural_nearest(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale),U32.to_nat(index)) : Nat}:
  Bands.Geometry{batch,channels,+height,+width} = geometry
  +output_height = (height * scale : U32)
  +output_width = (width * scale : U32)
  +plane = (output_height * output_width : U32)
  +natural_plane = Nat.mul(Nat.mul(U32.to_nat(height),U32.to_nat(scale)),Nat.mul(U32.to_nat(width),U32.to_nat(scale)))
  +output_width_positive = {
    %Equal.sym(Nat,U32.to_nat(output_width),Nat.mul(U32.to_nat(width),U32.to_nat(scale)),width_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    CountCompletion.positive_product(U32.to_nat(width),U32.to_nat(scale),width_positive,scale_positive) : {Nat.is_lt(0n,U32.to_nat(output_width)) == True{} : Bool}}
  +plane_positive = {
    %Equal.sym(Nat,U32.to_nat(plane),natural_plane,plane_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    CountCompletion.positive_product(Nat.mul(U32.to_nat(height),U32.to_nat(scale)),Nat.mul(U32.to_nat(width),U32.to_nat(scale)),CountCompletion.positive_product(U32.to_nat(height),U32.to_nat(scale),height_positive,scale_positive),CountCompletion.positive_product(U32.to_nat(width),U32.to_nat(scale),width_positive,scale_positive)) : {Nat.is_lt(0n,U32.to_nat(plane)) == True{} : Bool}}
  +channel = (index / plane : U32)
  +within = (index % plane : U32)
  +scaled_row = (within / output_width : U32)
  +row = (scaled_row / scale : U32)
  +scaled_column = (index % output_width : U32)
  +column = (scaled_column / scale : U32)
  +channel_exact = {
    %plane_exact : {U32.to_nat(channel) == Nat.div(U32.to_nat(index),_) : Nat}
    U32Division.divide(index,plane,capacity,capacity_ok,plane_bound,plane_positive) : {U32.to_nat(channel) == Nat.div(U32.to_nat(index),natural_plane) : Nat}}
  +within_exact = {
    %plane_exact : {U32.to_nat(within) == Nat.mod(U32.to_nat(index),_) : Nat}
    U32Division.modulo(index,plane,capacity,capacity_ok,plane_bound,plane_positive) : {U32.to_nat(within) == Nat.mod(U32.to_nat(index),natural_plane) : Nat}}
  +scaled_row_exact = {
    %within_exact : {U32.to_nat(scaled_row) == Nat.div(_,Nat.mul(U32.to_nat(width),U32.to_nat(scale))) : Nat}
    %width_exact : {U32.to_nat(scaled_row) == Nat.div(U32.to_nat(within),_) : Nat}
    U32Division.divide(within,output_width,capacity,capacity_ok,width_bound,output_width_positive) : {U32.to_nat(scaled_row) == Nat.div(Nat.mod(U32.to_nat(index),natural_plane),Nat.mul(U32.to_nat(width),U32.to_nat(scale))) : Nat}}
  +row_exact = {
    %scaled_row_exact : {U32.to_nat(row) == Nat.div(_,U32.to_nat(scale)) : Nat}
    U32Division.divide(scaled_row,scale,capacity,capacity_ok,scale_bound,scale_positive) : {U32.to_nat(row) == Nat.div(Nat.div(Nat.mod(U32.to_nat(index),natural_plane),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),U32.to_nat(scale)) : Nat}}
  +scaled_column_exact = {
    %width_exact : {U32.to_nat(scaled_column) == Nat.mod(U32.to_nat(index),_) : Nat}
    U32Division.modulo(index,output_width,capacity,capacity_ok,width_bound,output_width_positive) : {U32.to_nat(scaled_column) == Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(width),U32.to_nat(scale))) : Nat}}
  +column_exact = {
    %scaled_column_exact : {U32.to_nat(column) == Nat.div(_,U32.to_nat(scale)) : Nat}
    U32Division.divide(scaled_column,scale,capacity,capacity_ok,scale_bound,scale_positive) : {U32.to_nat(column) == Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),U32.to_nat(scale)) : Nat}}
  %channel_exact : {U32.to_nat((channel * height * width + row * width + column : U32)) == RowMajor.index(_,U32.to_nat(height),U32.to_nat(width),Nat.div(Nat.div(Nat.mod(U32.to_nat(index),natural_plane),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),U32.to_nat(scale)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),U32.to_nat(scale))) : Nat}
  %row_exact : {U32.to_nat((channel * height * width + row * width + column : U32)) == RowMajor.index(U32.to_nat(channel),U32.to_nat(height),U32.to_nat(width),_,Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),U32.to_nat(scale))) : Nat}
  %column_exact : {U32.to_nat((channel * height * width + row * width + column : U32)) == RowMajor.index(U32.to_nat(channel),U32.to_nat(height),U32.to_nat(width),U32.to_nat(row),_) : Nat}
  flat_word(channel,height,width,row,column,capacity,width_positive,
    %Equal.sym(Nat,U32.to_nat(channel),Nat.div(U32.to_nat(index),natural_plane),channel_exact) : {Nat.is_le(RowMajor.index(_,U32.to_nat(height),U32.to_nat(width),U32.to_nat(row),U32.to_nat(column)),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(row),Nat.div(Nat.div(Nat.mod(U32.to_nat(index),natural_plane),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),U32.to_nat(scale)),row_exact) : {Nat.is_le(RowMajor.index(Nat.div(U32.to_nat(index),natural_plane),U32.to_nat(height),U32.to_nat(width),_,U32.to_nat(column)),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(column),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),U32.to_nat(scale)),column_exact) : {Nat.is_le(RowMajor.index(Nat.div(U32.to_nat(index),natural_plane),U32.to_nat(height),U32.to_nat(width),Nat.div(Nat.div(Nat.mod(U32.to_nat(index),natural_plane),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),U32.to_nat(scale)),_),U32.to_nat(capacity)) == True{} : Bool} room)

def product_le(+left: Nat,+right: Nat,+left_upper: Nat,+right_upper: Nat,left_bound: {Nat.is_le(left,left_upper) == True{} : Bool},right_bound: {Nat.is_le(right,right_upper) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(left,right),Nat.mul(left_upper,right_upper)) == True{} : Bool}:
  Order.transitive(Nat.mul(left,right),Nat.mul(left_upper,right),Nat.mul(left_upper,right_upper),RowMajor.mul_le(left,left_upper,right,left_bound),
    %Equal.sym(Nat,Nat.mul(left_upper,right),Nat.mul(right,left_upper),RowMajor.multiply_commutative(left_upper,right)) : {Nat.is_le(_,Nat.mul(left_upper,right_upper)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(left_upper,right_upper),Nat.mul(right_upper,left_upper),RowMajor.multiply_commutative(left_upper,right_upper)) : {Nat.is_le(Nat.mul(right,left_upper),_) == True{} : Bool}
    RowMajor.mul_le(right,right_upper,left_upper,right_bound))

type Domains<-geometry: Bands.Geometry,-scale: U32,-limits: Machine.Limits> is Data:
  Domains{source: BandGather.Domain<geometry,limits,Machine.capacity(limits)>,
    target: BandGather.Domain<BandUpsampleExecution.scaled_geometry(geometry,scale),limits,Machine.capacity(limits)>}

def domains_from_count(geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+global_count: U32,+limits: Machine.Limits,
  +capacity_default: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool},
  dimensions: BandUpsampleSamples.Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)>,allocation: Count.Count<ShapeLaws.elements(Layout.scaled_extents(shape,scale)),global_count>) -> Domains<geometry,scale,limits>:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  BandUpsampleSamples.Dimensions{+images_positive,+channels_positive,+height_positive,+width_positive} = dimensions
  Count.Count{+exact,default_bounded} = allocation
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +capacity = Machine.capacity(limits)
  +bounded = {
    %Equal.sym(U32,capacity,Machine.capacity(Default.limits()),capacity_default) : {U32.is_le(global_count,_) == True{} : Bool}
    default_bounded : {U32.is_le(global_count,capacity) == True{} : Bool}}
  +scale_positive = BandUpsampleExecution.scale_positive(geometry,shape,scale,truth)
  +height_scaled_exact = BandUpsampleExecution.native_exact(height,scale,BandUpsampleExecution.height_scaled(geometry,shape,scale,truth))
  +width_scaled_exact = BandUpsampleExecution.native_exact(width,scale,BandUpsampleExecution.width_scaled(geometry,shape,scale,truth))
  +scaled_expected = Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(Nat.mul(U32.to_nat(height),U32.to_nat(scale)),Nat.mul(U32.to_nat(width),U32.to_nat(scale)))))
  +scaled_room = {
    %BandUpsampleExecution.scaled_elements(geometry,shape,scale,original_shape,truth) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    BandCount.natural_room(ShapeLaws.elements(Layout.scaled_extents(shape,scale)),global_count,capacity,exact,bounded) : {Nat.is_le(scaled_expected,U32.to_nat(capacity)) == True{} : Bool}}
  +spatial_bound = product_le(U32.to_nat(height),U32.to_nat(width),Nat.mul(U32.to_nat(height),U32.to_nat(scale)),Nat.mul(U32.to_nat(width),U32.to_nat(scale)),
    RowMajor.product_covers(U32.to_nat(height),U32.to_nat(scale),scale_positive),RowMajor.product_covers(U32.to_nat(width),U32.to_nat(scale),scale_positive))
  +frame_bound = product_le(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)),U32.to_nat(channels),Nat.mul(Nat.mul(U32.to_nat(height),U32.to_nat(scale)),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),Order.reflexive(U32.to_nat(channels)),spatial_bound)
  +total_bound = product_le(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(Nat.mul(U32.to_nat(height),U32.to_nat(scale)),Nat.mul(U32.to_nat(width),U32.to_nat(scale)))),Order.reflexive(U32.to_nat(batch)),frame_bound)
  +source_room = Order.transitive(BandGather.elements(geometry),scaled_expected,U32.to_nat(capacity),total_bound,scaled_room)
  +target_geometry = BandUpsampleExecution.scaled_geometry(geometry,scale)
  +target_exact = Equal.trans(Nat,ShapeLaws.elements(BandCount.extents(target_geometry)),ShapeLaws.elements(Layout.scaled_extents(shape,scale)),U32.to_nat(global_count),
    Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,BandCount.extents(target_geometry),Layout.scaled_extents(shape,scale),Equal.sym(List<&2,U32>,Layout.scaled_extents(shape,scale),BandCount.extents(target_geometry),BandUpsampleExecution.scaled_shape(geometry,shape,scale,original_shape))),exact)
  Domains{BandGather.Domain{Limits.capacity_fits(limits),channels_positive,height_positive,width_positive,source_room,source_room},
    BandCount.bounded(target_geometry,limits,global_count,channels_positive,
      %height_scaled_exact : {Nat.is_lt(0n,_) == True{} : Bool} CountCompletion.positive_product(U32.to_nat(height),U32.to_nat(scale),height_positive,scale_positive),
      %width_scaled_exact : {Nat.is_lt(0n,_) == True{} : Bool} CountCompletion.positive_product(U32.to_nat(width),U32.to_nat(scale),width_positive,scale_positive),target_exact,bounded)}

type UpsampleLimitsObservation is Data:
  UpsampleLimitsObservation{limits: Machine.Limits,actual: {limits == Default.limits() : Machine.Limits}}

def observe_upsample_limits() -> UpsampleLimitsObservation:
  UpsampleLimitsObservation{Default.limits(),{==}}

def domains_observed(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+global_count: U32,
  +original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool},
  dimensions: BandUpsampleSamples.Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)>,allocation: Count.Count<ShapeLaws.elements(Layout.scaled_extents(shape,scale)),global_count>,
  observation: UpsampleLimitsObservation) -> Domains<geometry,scale,Default.limits()>:
  UpsampleLimitsObservation{+limits,+actual} = observation
  %actual : Domains<geometry,scale,_>
  domains_from_count(geometry,shape,scale,global_count,limits,Equal.cong(Machine.Limits,U32,Machine.capacity,limits,Default.limits(),actual),original_shape,truth,dimensions,allocation)

def enabled_domains(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) -> Domains<geometry,scale,Default.limits()>:
  domains_observed(geometry,shape,scale,Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale))),original_shape,BandUpsampleExecution.legal(geometry,shape,scale,truth),BandUpsampleSamples.enabled_dimensions(geometry,shape,scale,original_shape,truth),BandUpsampleExecution.guards_allocation(geometry,shape,scale,BandUpsampleExecution.legal(geometry,shape,scale,truth)),observe_upsample_limits())

type Envelope<-geometry: Bands.Geometry,-limits: Machine.Limits> is Data:
  Envelope{spatial_room: {Nat.is_le(Nat.mul(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(BandUpsampleExecution.producer_width(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    width_room: {Nat.is_le(U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    height_room: {Nat.is_le(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    spatial_exact: {U32.to_nat((BandUpsampleExecution.producer_height(geometry) * BandUpsampleExecution.producer_width(geometry) : U32)) == Nat.mul(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(BandUpsampleExecution.producer_width(geometry))) : Nat},
    height_bound: {Nat.is_le(Nat.double(U32.to_nat(BandUpsampleExecution.producer_height(geometry))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
    plane_bound: {Nat.is_le(Nat.double(U32.to_nat((BandUpsampleExecution.producer_height(geometry) * BandUpsampleExecution.producer_width(geometry) : U32))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
    width_bound: {Nat.is_le(Nat.double(U32.to_nat(BandUpsampleExecution.producer_width(geometry))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}

def geometry_envelope(geometry: Bands.Geometry,+limits: Machine.Limits,+images_positive: {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_batch(geometry))) == True{} : Bool},domain: BandGather.Domain<geometry,limits,Machine.capacity(limits)>) -> Envelope<geometry,limits>:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  BandGather.Domain{capacity_ok,+channels_positive,+height_positive,+width_positive,+machine_room,allocation_room} = domain
  +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +frame = Nat.mul(U32.to_nat(channels),spatial)
  +total = Nat.mul(U32.to_nat(batch),frame)
  +frame_room = Order.transitive(frame,total,U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(frame,U32.to_nat(batch)) : {Nat.is_le(frame,_) == True{} : Bool} RowMajor.product_covers(frame,U32.to_nat(batch),images_positive),machine_room)
  +spatial_room = Order.transitive(spatial,frame,U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(spatial,U32.to_nat(channels)) : {Nat.is_le(spatial,_) == True{} : Bool} RowMajor.product_covers(spatial,U32.to_nat(channels),channels_positive),frame_room)
  +height_room = Order.transitive(U32.to_nat(height),spatial,U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(height),U32.to_nat(width),width_positive),spatial_room)
  +width_room = Order.transitive(U32.to_nat(width),spatial,U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(U32.to_nat(width),U32.to_nat(height)) : {Nat.is_le(U32.to_nat(width),_) == True{} : Bool} RowMajor.product_covers(U32.to_nat(width),U32.to_nat(height),height_positive),spatial_room)
  +spatial_exact = FullRange.multiply(height,width,Machine.capacity(limits),spatial_room)
  Envelope{spatial_room,width_room,height_room,spatial_exact,Limits.double_bound(limits,U32.to_nat(height),height_room),
    %Equal.sym(Nat,U32.to_nat((height * width : U32)),spatial,spatial_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(Machine.half_word(limits))) == True{} : Bool} Limits.double_bound(limits,spatial,spatial_room),
    Limits.double_bound(limits,U32.to_nat(width),width_room)}

type Frames<-geometry: Bands.Geometry,-scale: U32,-first: U32,-rows: U32,-length: Nat,-output_count: U32,-limits: Machine.Limits> is Data:
  Frames{source_domain: BandGather.Domain<geometry,limits,Machine.capacity(limits)>,target_domain: BandGather.Domain<BandUpsampleExecution.scaled_geometry(geometry,scale),limits,Machine.capacity(limits)>,
    source_envelope: Envelope<geometry,limits>,target_envelope: Envelope<BandUpsampleExecution.scaled_geometry(geometry,scale),limits>,
    source_facts: BandGather.Facts<BandGather.Call{geometry,first,rows,length,Machine.capacity(limits),Machine.half_word(limits)}>,
    target_facts: BandGather.Facts<BandGather.Call{BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.to_nat(output_count),Machine.capacity(limits),Machine.half_word(limits)}>,
    words: BandUpsampleExecution.Words<scale,first,rows>,height_exact: {U32.to_nat((BandUpsampleExecution.producer_height(geometry) * scale : U32)) == Nat.mul(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(scale)) : Nat},
    width_exact: {U32.to_nat((BandUpsampleExecution.producer_width(geometry) * scale : U32)) == Nat.mul(U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale)) : Nat},
    scale_positive: {Nat.is_lt(0n,U32.to_nat(scale)) == True{} : Bool},scale_bound: {Nat.is_le(Nat.double(U32.to_nat(scale)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}

def frames_envelopes(+geometry: Bands.Geometry,+scale: U32,+first: U32,+rows: U32,+length: Nat,+output_count: U32,+limits: Machine.Limits,
  +source_domain: BandGather.Domain<geometry,limits,Machine.capacity(limits)>,+target_domain: BandGather.Domain<BandUpsampleExecution.scaled_geometry(geometry,scale),limits,Machine.capacity(limits)>,
  +source_facts: BandGather.Facts<BandGather.Call{geometry,first,rows,length,Machine.capacity(limits),Machine.half_word(limits)}>,
  +target_facts: BandGather.Facts<BandGather.Call{BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.to_nat(output_count),Machine.capacity(limits),Machine.half_word(limits)}>,
  +words: BandUpsampleExecution.Words<scale,first,rows>,+height_exact: {U32.to_nat((BandUpsampleExecution.producer_height(geometry) * scale : U32)) == Nat.mul(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(scale)) : Nat},
  +width_exact: {U32.to_nat((BandUpsampleExecution.producer_width(geometry) * scale : U32)) == Nat.mul(U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale)) : Nat},+width_positive: {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_width(geometry))) == True{} : Bool},
  +scale_positive: {Nat.is_lt(0n,U32.to_nat(scale)) == True{} : Bool},+source_envelope: Envelope<geometry,limits>,target_envelope: Envelope<BandUpsampleExecution.scaled_geometry(geometry,scale),limits>) -> Frames<geometry,scale,first,rows,length,output_count,limits>:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Envelope{+spatial_room,+width_room,+height_room,+spatial_exact,+height_bound,+plane_bound,+width_bound} = target_envelope
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +target_envelope = {Envelope{spatial_room,width_room,height_room,spatial_exact,height_bound,plane_bound,width_bound} : Envelope<BandUpsampleExecution.scaled_geometry(geometry,scale),limits>}
  +scale_room = Order.transitive(U32.to_nat(scale),U32.to_nat((BandUpsampleExecution.producer_width(geometry) * scale : U32)),U32.to_nat(Machine.capacity(limits)),
    %Equal.sym(Nat,U32.to_nat((BandUpsampleExecution.producer_width(geometry) * scale : U32)),Nat.mul(U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale)),width_exact) : {Nat.is_le(U32.to_nat(scale),_) == True{} : Bool}
    %RowMajor.multiply_commutative(U32.to_nat(scale),U32.to_nat(BandUpsampleExecution.producer_width(geometry))) : {Nat.is_le(U32.to_nat(scale),_) == True{} : Bool} RowMajor.product_covers(U32.to_nat(scale),U32.to_nat(BandUpsampleExecution.producer_width(geometry)),width_positive),width_room)
  Frames{source_domain,target_domain,source_envelope,target_envelope,source_facts,target_facts,words,height_exact,width_exact,scale_positive,Limits.double_bound(limits,U32.to_nat(scale),scale_room)}

def frames_from_ready(~Element: Data,geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+limits: Machine.Limits,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool},
  domains: Domains<geometry,scale,limits>,placement: BandUpsampleExecution.Placement<geometry,scale,first,rows>,allocated: Depth.Sized<BandUpsampleExecution.producer_output_count(geometry,rows,scale)>) -> Frames<geometry,scale,first,rows,Buffer.length(~Element,input),BandUpsampleExecution.producer_output_count(geometry,rows,scale),limits>:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Domains{+source_domain,+target_domain} = domains
  BandUpsampleExecution.Placement{+first_exact,+rows_exact,+target_metadata} = placement
  Depth.Sized{+depth_ok,+allocation_room} = allocated
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +capacity = Machine.capacity(limits)
  +output_count = BandUpsampleExecution.producer_output_count(geometry,rows,scale)
  +output = Buffer.fresh(~Element,Tensors.depth_for(output_count),fill,U32.to_nat(output_count))
  +output_ready = Buffer.fresh_ready(~Element,Tensors.depth_for(output_count),fill,U32.to_nat(output_count),depth_ok,allocation_room)
  +source_facts = BandGather.from_metadata(~Element,geometry,limits,first,rows,input,capacity,source_domain,metadata,rows_positive,ready)
  +source_images = BandUpsampleSamples.images_positive(geometry,BandUpsampleSamples.positive_rows(geometry,first,rows,Buffer.length(~Element,input),metadata,U32Division.nonzero(rows,rows_positive)))
  +height_exact = Equal.sym(Nat,Nat.mul(U32.to_nat(height),U32.to_nat(scale)),U32.to_nat((height * scale : U32)),BandUpsampleExecution.native_exact(height,scale,BandUpsampleExecution.height_scaled(geometry,shape,scale,BandUpsampleExecution.legal(geometry,shape,scale,truth))))
  +width_exact = Equal.sym(Nat,Nat.mul(U32.to_nat(width),U32.to_nat(scale)),U32.to_nat((width * scale : U32)),BandUpsampleExecution.native_exact(width,scale,BandUpsampleExecution.width_scaled(geometry,shape,scale,BandUpsampleExecution.legal(geometry,shape,scale,truth))))
  +scale_positive = BandUpsampleExecution.scale_positive(geometry,shape,scale,BandUpsampleExecution.legal(geometry,shape,scale,truth))
  +target_rows_positive = {
    %Equal.sym(Nat,U32.to_nat((rows * scale : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),rows_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    CountCompletion.positive_product(U32.to_nat(rows),U32.to_nat(scale),rows_positive,scale_positive) : {Nat.is_lt(0n,U32.to_nat((rows * scale : U32))) == True{} : Bool}}
  +target_facts = BandGather.from_metadata(~Element,BandUpsampleExecution.scaled_geometry(geometry,scale),limits,(first * scale : U32),(rows * scale : U32),output,capacity,target_domain,target_metadata,target_rows_positive,output_ready)
  frames_envelopes(geometry,scale,first,rows,Buffer.length(~Element,input),output_count,limits,source_domain,target_domain,source_facts,target_facts,BandUpsampleExecution.Words{first_exact,rows_exact},height_exact,width_exact,
    BandUpsampleSamples.width_positive_dimensions(geometry,BandUpsampleSamples.enabled_dimensions(geometry,shape,scale,original_shape,truth)),scale_positive,
    geometry_envelope(geometry,limits,source_images,source_domain),geometry_envelope(BandUpsampleExecution.scaled_geometry(geometry,scale),limits,source_images,target_domain))

def frames_observed(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool},
  domains: Domains<geometry,scale,Default.limits()>,placement: BandUpsampleExecution.Placement<geometry,scale,first,rows>,allocated: Depth.Sized<BandUpsampleExecution.producer_output_count(geometry,rows,scale)>,observation: UpsampleLimitsObservation) ->
  Frames<geometry,scale,first,rows,Buffer.length(~Element,input),BandUpsampleExecution.producer_output_count(geometry,rows,scale),Default.limits()>:
  UpsampleLimitsObservation{+limits,+actual} = observation
  %actual : Frames<geometry,scale,first,rows,Buffer.length(~Element,input),BandUpsampleExecution.producer_output_count(geometry,rows,scale),_>
  frames_from_ready(~Element,geometry,shape,scale,fill,first,rows,input,limits,metadata,ready,rows_positive,original_shape,truth,
    %Equal.sym(Machine.Limits,limits,Default.limits(),actual) : Domains<geometry,scale,_> domains,placement,allocated)

def enabled_frames(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale)>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  Frames<geometry,scale,first,rows,Buffer.length(~Element,input),BandUpsampleExecution.producer_output_count(geometry,rows,scale),Default.limits()>:
  +dimensions = {BandUpsampleSamples.enabled_dimensions(geometry,shape,scale,original_shape,truth) : BandUpsampleSamples.Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)>}
  frames_observed(~Element,geometry,shape,scale,fill,first,rows,input,metadata,ready,rows_positive,original_shape,truth,enabled_domains(geometry,shape,scale,original_shape,truth),
    BandUpsampleSamples.placement(~Element,geometry,shape,scale,first,rows,input,metadata,ready,
      BandUpsampleSamples.active_envelope(geometry,shape,scale,BandUpsampleSamples.channels_positive_dimensions(geometry,dimensions),BandUpsampleSamples.width_positive_dimensions(geometry,dimensions),original_shape,BandUpsampleExecution.legal(geometry,shape,scale,truth)),original_shape,BandUpsampleExecution.legal(geometry,shape,scale,truth)),
    BandUpsampleExecution.allocated(geometry,rows,scale,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),descriptor),observe_upsample_limits())

def scaled_global_flattened(+first: Nat,+rows: Nat,+channels: Nat,+height: Nat,+width: Nat,+scale: Nat,+index: Nat,+channel: Nat,+row: Nat,+row_repeat: Nat,+column: Nat,+column_repeat: Nat,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+column_inside: {Nat.is_lt(column,width) == True{} : Bool},
  +row_repeat_inside: {Nat.is_lt(row_repeat,scale) == True{} : Bool},+column_repeat_inside: {Nat.is_lt(column_repeat,scale) == True{} : Bool},
  +index_exact: {index == five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat) : Nat}) ->
  {BandCoordinates.global_index(Nat.mul(first,scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),index) ==
    RowMajor.index(Nat.add(Nat.mul(Nat.div(first,height),channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(Nat.mod(first,height),scale),Nat.add(Nat.mul(row,scale),row_repeat)),Nat.add(Nat.mul(column,scale),column_repeat)) : Nat}:
  +image = Nat.div(first,height)
  +local_first = Nat.mod(first,height)
  +absolute = Nat.add(Nat.mul(image,height),local_first)
  +scaled_row = Nat.add(Nat.mul(row,scale),row_repeat)
  +scaled_column = Nat.add(Nat.mul(column,scale),column_repeat)
  +scale_positive = Order.le_lt_trans(0n,column_repeat,scale,Order.zero_le(column_repeat),column_repeat_inside)
  %Equal.sym(Nat,first,absolute,DivisionOrder.quotient_exact(first,height,height_positive)) :
    {BandCoordinates.global_index(Nat.mul(_,scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),index) ==
      RowMajor.index(Nat.add(Nat.mul(image,channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(local_first,scale),scaled_row),scaled_column) : Nat}
  %Equal.sym(Nat,Nat.mul(absolute,scale),Nat.add(Nat.mul(image,Nat.mul(height,scale)),Nat.mul(local_first,scale)),scaled_first(image,height,local_first,scale)) :
    {BandCoordinates.global_index(_,Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),index) ==
      RowMajor.index(Nat.add(Nat.mul(image,channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(local_first,scale),scaled_row),scaled_column) : Nat}
  %Equal.sym(Nat,index,five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat),index_exact) :
    {BandCoordinates.global_index(Nat.add(Nat.mul(image,Nat.mul(height,scale)),Nat.mul(local_first,scale)),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),_) ==
      RowMajor.index(Nat.add(Nat.mul(image,channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(local_first,scale),scaled_row),scaled_column) : Nat}
  %Equal.sym(Nat,five_index(channel,rows,scale,width,row,row_repeat,column,column_repeat),RowMajor.index(channel,Nat.mul(rows,scale),Nat.mul(width,scale),scaled_row,scaled_column),five_scaled(channel,rows,scale,width,row,row_repeat,column,column_repeat)) :
    {BandCoordinates.global_index(Nat.add(Nat.mul(image,Nat.mul(height,scale)),Nat.mul(local_first,scale)),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),_) ==
      RowMajor.index(Nat.add(Nat.mul(image,channels),channel),Nat.mul(height,scale),Nat.mul(width,scale),Nat.add(Nat.mul(local_first,scale),scaled_row),scaled_column) : Nat}
  BandCoordinates.flattened(image,Nat.mul(local_first,scale),Nat.mul(rows,scale),channels,Nat.mul(height,scale),Nat.mul(width,scale),channel,scaled_row,scaled_column,
    RowMajor.multiply_lt(local_first,height,scale,DivisionOrder.remainder_small(first,height,height_positive),scale_positive),RowMajor.radix(row,rows,row_repeat,scale,row_inside,row_repeat_inside),RowMajor.radix(column,width,column_repeat,scale,column_inside,column_repeat_inside))

def target_decoded(geometry: Bands.Geometry,+scale: U32,+first: U32,+rows: U32,+index: Nat,
  +first_exact: {U32.to_nat((first * scale : U32)) == Nat.mul(U32.to_nat(first),U32.to_nat(scale)) : Nat},+rows_exact: {U32.to_nat((rows * scale : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(scale)) : Nat},
  +height_exact: {U32.to_nat((BandUpsampleExecution.producer_height(geometry) * scale : U32)) == Nat.mul(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(scale)) : Nat},+width_exact: {U32.to_nat((BandUpsampleExecution.producer_width(geometry) * scale : U32)) == Nat.mul(U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale)) : Nat},
  +index_exact: {U32.to_nat(U32.from_nat(index)) == index : Nat}) ->
  {BandCoordinates.global_index(U32.to_nat((first * scale : U32)),U32.to_nat((rows * scale : U32)),U32.to_nat(BandUpsampleExecution.producer_channels(geometry)),U32.to_nat((BandUpsampleExecution.producer_height(geometry) * scale : U32)),U32.to_nat((BandUpsampleExecution.producer_width(geometry) * scale : U32)),U32.to_nat(U32.from_nat(index))) ==
    BandCoordinates.global_index(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),U32.to_nat(BandUpsampleExecution.producer_channels(geometry)),Nat.mul(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(scale)),Nat.mul(U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale)),index) : Nat}:
  Bands.Geometry{batch,+channels,+height,+width} = geometry
  +target = BandCoordinates.global_index(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(scale)),Nat.mul(U32.to_nat(width),U32.to_nat(scale)),index)
  %Equal.sym(Nat,U32.to_nat((first * scale : U32)),Nat.mul(U32.to_nat(first),U32.to_nat(scale)),first_exact) : {BandCoordinates.global_index(_,U32.to_nat((rows * scale : U32)),U32.to_nat(channels),U32.to_nat((height * scale : U32)),U32.to_nat((width * scale : U32)),U32.to_nat(U32.from_nat(index))) == target : Nat}
  %Equal.sym(Nat,U32.to_nat((rows * scale : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),rows_exact) : {BandCoordinates.global_index(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),_,U32.to_nat(channels),U32.to_nat((height * scale : U32)),U32.to_nat((width * scale : U32)),U32.to_nat(U32.from_nat(index))) == target : Nat}
  %Equal.sym(Nat,U32.to_nat((height * scale : U32)),Nat.mul(U32.to_nat(height),U32.to_nat(scale)),height_exact) : {BandCoordinates.global_index(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),U32.to_nat(channels),_,U32.to_nat((width * scale : U32)),U32.to_nat(U32.from_nat(index))) == target : Nat}
  %Equal.sym(Nat,U32.to_nat((width * scale : U32)),Nat.mul(U32.to_nat(width),U32.to_nat(scale)),width_exact) : {BandCoordinates.global_index(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(scale)),_,U32.to_nat(U32.from_nat(index))) == target : Nat}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,index_exact) : {BandCoordinates.global_index(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(scale)),Nat.mul(U32.to_nat(width),U32.to_nat(scale)),_) == target : Nat}
  {==}

def native_sample_index(~Element: Data,geometry: Bands.Geometry,+scale: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+index: Nat,+limits: Machine.Limits,
  metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),location: LocalSampleIndex<geometry,rows,scale,index>,
  frames: Frames<geometry,scale,first,rows,Buffer.length(~Element,input),BandUpsampleExecution.producer_output_count(geometry,rows,scale),limits>,+inside: {Nat.is_lt(index,U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale))) == True{} : Bool}) ->
  {Bands.global_index(geometry,first,rows,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)))))) ==
    BandUpsampleExecution.nearest_index(geometry,scale,Bands.global_index(BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.from_nat(index))) : U32}:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  BandShape.Fields{local,+rows_fit,+length_exact} = metadata
  LocalSampleIndex{RepeatedParts{+channel,+row,+row_repeat,+column,+column_repeat,+channel_inside,+row_inside,+row_repeat_inside,+column_inside,+column_repeat_inside,+index_exact},+source_address} = location
  Frames{BandGather.Domain{source_capacity_ok,source_channels_positive,source_height_positive,source_width_positive,source_machine_room,+source_allocation_room},
    BandGather.Domain{target_capacity_ok,target_channels_positive,target_height_positive,target_width_positive,target_machine_room,+target_allocation_room},
    Envelope{source_spatial_room,source_width_room,source_height_room,source_spatial_exact,source_height_double,source_plane_double,source_width_double},
    Envelope{target_spatial_room,target_width_room,target_height_room,+target_spatial_exact,target_height_double,+target_plane_double,+target_width_double},
    BandGather.Facts{source_valid,source_stride_exact,source_counted_length,+source_length_bound,source_division_ok,+source_local_plane_bound,+source_positive_height,+source_positive_width,source_count_exact,+source_height_bound,+source_row_span},
    BandGather.Facts{target_valid,target_stride_exact,target_counted_length,+target_length_bound,target_division_ok,+target_local_plane_bound,+target_positive_height,+target_positive_width,target_count_exact,+target_height_bound,target_row_span},
    BandUpsampleExecution.Words{+first_exact,+rows_exact},+height_exact,+width_exact,+scale_positive,+scale_bound} = frames
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +capacity = Machine.capacity(limits)
  +division_capacity = Machine.half_word(limits)
  +division_ok = Limits.half_word_fits(limits)
  +source_plane_natural = {
    %source_count_exact : {Nat.is_le(Nat.double(_),U32.to_nat(division_capacity)) == True{} : Bool}
    source_local_plane_bound : {Nat.is_le(Nat.double(Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(division_capacity)) == True{} : Bool}}
  +target_plane_natural = {
    %target_count_exact : {Nat.is_le(Nat.double(_),U32.to_nat(division_capacity)) == True{} : Bool}
    target_local_plane_bound : {Nat.is_le(Nat.double(Nat.mul(U32.to_nat((rows * scale : U32)),U32.to_nat((width * scale : U32)))),U32.to_nat(division_capacity)) == True{} : Bool}}
  +height_natural = U32.to_nat(height)
  +width_natural = U32.to_nat(width)
  +rows_natural = U32.to_nat(rows)
  +scale_natural = U32.to_nat(scale)
  +first_natural = U32.to_nat(first)
  +image = Nat.div(first_natural,height_natural)
  +local_first = Nat.mod(first_natural,height_natural)
  +absolute = Nat.add(Nat.mul(image,height_natural),local_first)
  +first_split = DivisionOrder.quotient_exact(first_natural,height_natural,source_positive_height)
  +local_first_inside = DivisionOrder.remainder_small(first_natural,height_natural,source_positive_height)
  +rows_positive = Order.le_lt_trans(0n,row,rows_natural,Order.zero_le(row),row_inside)
  +first_inside_total = Bounds.nat_trans(first_natural,Nat.add(first_natural,rows_natural),Nat.mul(U32.to_nat(batch),height_natural),
    %Addition.zero_right(first_natural) : {Nat.is_lt(_,Nat.add(first_natural,rows_natural)) == True{} : Bool} Order.offset_lt(first_natural,0n,rows_natural,rows_positive),rows_fit)
  +image_inside = RowMajor.row_inside(image,U32.to_nat(batch),height_natural,local_first,
    %first_split : {Nat.is_lt(_,Nat.mul(U32.to_nat(batch),height_natural)) == True{} : Bool} first_inside_total)
  +source_local = RowMajor.index(channel,rows_natural,width_natural,row,column)
  +source_local_inside = RowMajor.in_bounds(channel,U32.to_nat(channels),rows_natural,width_natural,row,column,channel_inside,row_inside,column_inside)
  +source_local_length = {
    %Equal.sym(Nat,Buffer.length(~Element,input),Nat.mul(U32.to_nat(channels),Nat.mul(rows_natural,width_natural)),length_exact) : {Nat.is_lt(source_local,_) == True{} : Bool}
    source_local_inside : {Nat.is_lt(source_local,Buffer.length(~Element,input)) == True{} : Bool}}
  +source_roundtrip = Bounds.roundtrip(source_local,division_capacity,division_ok,Bounds.strict_le(source_local,U32.to_nat(division_capacity),Bounds.nat_trans(source_local,Buffer.length(~Element,input),U32.to_nat(division_capacity),source_local_length,source_length_bound)))
  +target_roundtrip = Bounds.roundtrip(index,division_capacity,division_ok,Bounds.strict_le(index,U32.to_nat(division_capacity),Bounds.nat_trans(index,U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale)),U32.to_nat(division_capacity),inside,target_length_bound)))
  +source_global = BandCoordinates.global_index(first_natural,rows_natural,U32.to_nat(channels),height_natural,width_natural,source_local)
  +target_global = BandCoordinates.global_index(Nat.mul(first_natural,scale_natural),Nat.mul(rows_natural,scale_natural),U32.to_nat(channels),Nat.mul(height_natural,scale_natural),Nat.mul(width_natural,scale_natural),index)
  +source_flat = {
    %Equal.sym(Nat,first_natural,absolute,first_split) : {BandCoordinates.global_index(_,rows_natural,U32.to_nat(channels),height_natural,width_natural,source_local) == RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),height_natural,width_natural,Nat.add(local_first,row),column) : Nat}
    BandCoordinates.flattened(image,local_first,rows_natural,U32.to_nat(channels),height_natural,width_natural,channel,row,column,local_first_inside,row_inside,column_inside) :
    {source_global == RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),height_natural,width_natural,Nat.add(local_first,row),column) : Nat}}
  +source_global_room = {
    %Equal.sym(Nat,source_global,RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),height_natural,width_natural,Nat.add(local_first,row),column),source_flat) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    BandCoordinates.coordinates_room(U32.to_nat(batch),U32.to_nat(channels),height_natural,width_natural,image,local_first,rows_natural,channel,row,column,U32.to_nat(capacity),image_inside,channel_inside,source_row_span,row_inside,column_inside,source_allocation_room) :
    {Nat.is_le(source_global,U32.to_nat(capacity)) == True{} : Bool}}
  +source_half_room = Order.transitive(source_global,U32.to_nat(capacity),U32.to_nat(division_capacity),source_global_room,Bounds.as_nat_le(capacity,division_capacity,Limits.capacity_within_half(limits)))
  +target_scaled_row = Nat.add(Nat.mul(row,scale_natural),row_repeat)
  +target_scaled_column = Nat.add(Nat.mul(column,scale_natural),column_repeat)
  +target_scaled_first = Nat.mul(local_first,scale_natural)
  +target_row_fit = {
    %RowMajor.distrib(local_first,rows_natural,scale_natural) : {Nat.is_le(_,Nat.mul(height_natural,scale_natural)) == True{} : Bool}
    RowMajor.mul_le(Nat.add(local_first,rows_natural),height_natural,scale_natural,source_row_span) : {Nat.is_le(Nat.add(target_scaled_first,Nat.mul(rows_natural,scale_natural)),Nat.mul(height_natural,scale_natural)) == True{} : Bool}}
  +target_total_room = {
    %height_exact : {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(_,Nat.mul(width_natural,scale_natural)))),U32.to_nat(capacity)) == True{} : Bool}
    %width_exact : {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat((height * scale : U32)),_))),U32.to_nat(capacity)) == True{} : Bool}
    target_allocation_room : {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(Nat.mul(height_natural,scale_natural),Nat.mul(width_natural,scale_natural)))),U32.to_nat(capacity)) == True{} : Bool}}
  +target_flat = scaled_global_flattened(first_natural,rows_natural,U32.to_nat(channels),height_natural,width_natural,scale_natural,index,channel,row,row_repeat,column,column_repeat,source_positive_height,row_inside,column_inside,row_repeat_inside,column_repeat_inside,index_exact)
  +target_global_room = {
    %Equal.sym(Nat,target_global,RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),Nat.mul(height_natural,scale_natural),Nat.mul(width_natural,scale_natural),Nat.add(target_scaled_first,target_scaled_row),target_scaled_column),target_flat) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    BandCoordinates.coordinates_room(U32.to_nat(batch),U32.to_nat(channels),Nat.mul(height_natural,scale_natural),Nat.mul(width_natural,scale_natural),image,target_scaled_first,Nat.mul(rows_natural,scale_natural),channel,target_scaled_row,target_scaled_column,U32.to_nat(capacity),
      image_inside,channel_inside,target_row_fit,RowMajor.radix(row,rows_natural,row_repeat,scale_natural,row_inside,row_repeat_inside),RowMajor.radix(column,width_natural,column_repeat,scale_natural,column_inside,column_repeat_inside),target_total_room) :
    {Nat.is_le(target_global,U32.to_nat(capacity)) == True{} : Bool}}
  +target_half_room = Order.transitive(target_global,U32.to_nat(capacity),U32.to_nat(division_capacity),target_global_room,Bounds.as_nat_le(capacity,division_capacity,Limits.capacity_within_half(limits)))
  +source_word = U32.from_nat(source_local)
  +source_native_global = Bands.global_index(geometry,first,rows,source_word)
  +source_native_exact = {
    %source_roundtrip : {U32.to_nat(source_native_global) == BandCoordinates.global_index(first_natural,rows_natural,U32.to_nat(channels),height_natural,width_natural,_) : Nat}
    BandCoordinates.native(batch,channels,height,width,first,rows,source_word,division_capacity,division_ok,source_positive_height,source_positive_width,rows_positive,source_height_bound,source_plane_natural,
      %Equal.sym(Nat,U32.to_nat(source_word),source_local,source_roundtrip) : {Nat.is_le(BandCoordinates.global_index(first_natural,rows_natural,U32.to_nat(channels),height_natural,width_natural,_),U32.to_nat(division_capacity)) == True{} : Bool} source_half_room) :
    {U32.to_nat(source_native_global) == source_global : Nat}}
  +target_word = Bands.global_index(BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.from_nat(index))
  +decoded_target = target_decoded(geometry,scale,first,rows,index,first_exact,rows_exact,height_exact,width_exact,target_roundtrip)
  +target_rows_positive = {
    %Equal.sym(Nat,U32.to_nat((rows * scale : U32)),Nat.mul(rows_natural,scale_natural),rows_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    CountCompletion.positive_product(rows_natural,scale_natural,rows_positive,scale_positive) : {Nat.is_lt(0n,U32.to_nat((rows * scale : U32))) == True{} : Bool}}
  +target_raw = BandCoordinates.global_index(U32.to_nat((first * scale : U32)),U32.to_nat((rows * scale : U32)),U32.to_nat(channels),U32.to_nat((height * scale : U32)),U32.to_nat((width * scale : U32)),U32.to_nat(U32.from_nat(index)))
  +target_native_exact = Equal.trans(Nat,U32.to_nat(target_word),target_raw,target_global,
    BandCoordinates.native(batch,channels,(height * scale : U32),(width * scale : U32),(first * scale : U32),(rows * scale : U32),U32.from_nat(index),division_capacity,division_ok,target_positive_height,target_positive_width,target_rows_positive,target_height_bound,target_plane_natural,
      %Equal.sym(Nat,target_raw,target_global,decoded_target) : {Nat.is_le(_,U32.to_nat(division_capacity)) == True{} : Bool} target_half_room),decoded_target)
  +parts = {RepeatedParts{channel,row,row_repeat,column,column_repeat,channel_inside,row_inside,row_repeat_inside,column_inside,column_repeat_inside,index_exact} : RepeatedParts<U32.to_nat(channels),rows_natural,scale_natural,width_natural,index>}
  +nearest_natural = global_nearest(first_natural,rows_natural,U32.to_nat(channels),height_natural,width_natural,scale_natural,index,source_positive_height,source_row_span,parts)
  +plane_exact = {
    %height_exact : {U32.to_nat(((height * scale : U32) * (width * scale : U32) : U32)) == Nat.mul(_,Nat.mul(width_natural,scale_natural)) : Nat}
    %width_exact : {U32.to_nat(((height * scale : U32) * (width * scale : U32) : U32)) == Nat.mul(U32.to_nat((height * scale : U32)),_) : Nat}
    target_spatial_exact : {U32.to_nat(((height * scale : U32) * (width * scale : U32) : U32)) == Nat.mul(Nat.mul(height_natural,scale_natural),Nat.mul(width_natural,scale_natural)) : Nat}}
  +nearest_native = nearest_word(geometry,scale,target_word,division_capacity,division_ok,width_exact,plane_exact,source_positive_height,source_positive_width,scale_positive,target_plane_double,target_width_double,scale_bound,
    %Equal.sym(Nat,U32.to_nat(target_word),target_global,target_native_exact) : {Nat.is_le(natural_nearest(height_natural,width_natural,scale_natural,_),U32.to_nat(division_capacity)) == True{} : Bool}
    %Equal.sym(Nat,natural_nearest(height_natural,width_natural,scale_natural,target_global),source_global,nearest_natural) : {Nat.is_le(_,U32.to_nat(division_capacity)) == True{} : Bool} source_half_room)
  +nearest_exact = Equal.trans(Nat,U32.to_nat(BandUpsampleExecution.nearest_index(geometry,scale,target_word)),natural_nearest(height_natural,width_natural,scale_natural,U32.to_nat(target_word)),source_global,nearest_native,
    Equal.trans(Nat,natural_nearest(height_natural,width_natural,scale_natural,U32.to_nat(target_word)),natural_nearest(height_natural,width_natural,scale_natural,target_global),source_global,
      Equal.cong(Nat,Nat,value => natural_nearest(height_natural,width_natural,scale_natural,value),U32.to_nat(target_word),target_global,target_native_exact),nearest_natural))
  %Equal.sym(U32,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))))),source_word,source_address) :
    {Bands.global_index(geometry,first,rows,_) == BandUpsampleExecution.nearest_index(geometry,scale,target_word) : U32}
  Address.equal_values(source_native_global,BandUpsampleExecution.nearest_index(geometry,scale,target_word),Equal.trans(Nat,U32.to_nat(source_native_global),source_global,U32.to_nat(BandUpsampleExecution.nearest_index(geometry,scale,target_word)),source_native_exact,
    Equal.sym(Nat,U32.to_nat(BandUpsampleExecution.nearest_index(geometry,scale,target_word)),source_global,nearest_exact)))

def located_rows_positive(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+index: Nat,location: LocalSampleIndex<geometry,rows,scale,index>) -> {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}:
  LocalSampleIndex{RepeatedParts{channel,+row,row_repeat,column,column_repeat,channel_inside,+row_inside,row_repeat_inside,column_inside,column_repeat_inside,index_exact},address} = location
  Order.le_lt_trans(0n,row,U32.to_nat(rows),Order.zero_le(row),row_inside)

def enabled_native_index(~Element: Data,geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+index: Nat,
  +descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale)>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  +inside: {Nat.is_lt(index,U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale))) == True{} : Bool},+original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  {Bands.global_index(geometry,first,rows,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)))))) ==
    BandUpsampleExecution.nearest_index(geometry,scale,Bands.global_index(BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.from_nat(index))) : U32}:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +scale_positive = BandUpsampleExecution.scale_positive(geometry,shape,scale,BandUpsampleExecution.legal(geometry,shape,scale,truth))
  +width_positive = BandUpsampleSamples.width_positive_dimensions(geometry,BandUpsampleSamples.enabled_dimensions(geometry,shape,scale,original_shape,truth))
  +location = {repeated_location(~Element,geometry,first,rows,scale,input,BandUpsampleExecution.producer_output_count(geometry,rows,scale),index,descriptor,metadata,ready,scale_positive,width_positive,inside) : LocalSampleIndex<geometry,rows,scale,index>}
  native_sample_index(~Element,geometry,scale,first,rows,input,index,Default.limits(),metadata,location,
    enabled_frames(~Element,geometry,shape,scale,fill,first,rows,input,descriptor,metadata,ready,located_rows_positive(geometry,rows,scale,index,location),original_shape,truth),inside)

def nearest_point_from_source(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+index: Nat,
  +descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale)>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  +inside: {Nat.is_lt(index,U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale))) == True{} : Bool},+original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool},
  source: Shape.Both<{U32.is_lt(U32.from_nat(index),Tree.capacity(Element,BandUpsampleExecution.fresh_tree(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale)))) == True{} : Bool},
    CopyValues.Point<Element,original(context,Bands.global_index(geometry,first,rows,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))))))),
      U32.from_nat(index),BandUpsampleExecution.leaf_writes(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),input)>>) ->
  Shape.Both<{U32.is_lt(U32.from_nat(index),Tree.capacity(Element,BandUpsampleExecution.fresh_tree(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale)))) == True{} : Bool},
    CopyValues.Point<Element,BandUpsampleExecution.value(~Element,~Context,~original,BandUpsampleExecution.Context{context,geometry,scale},Bands.global_index(BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.from_nat(index))),
      U32.from_nat(index),BandUpsampleExecution.leaf_writes(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),input)>>:
  Shape.Both{query_inside,point} = source
  Shape.Both{query_inside,
    %Equal.cong(U32,Element,address => original(context,address),Bands.global_index(geometry,first,rows,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)))))),
      BandUpsampleExecution.nearest_index(geometry,scale,Bands.global_index(BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.from_nat(index))),enabled_native_index(~Element,geometry,shape,scale,fill,first,rows,input,index,descriptor,metadata,ready,inside,original_shape,truth)) :
      CopyValues.Point<Element,_,U32.from_nat(index),BandUpsampleExecution.leaf_writes(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),input)>
    point}

def nearest_point(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+index: Nat,
  +metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,+inside: {Nat.is_lt(index,U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale))) == True{} : Bool},
  +original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool},evidence: BandUpsampleSamples.ActualLeaf<Element,Context,original,context,geometry,scale,fill,first,rows,input>) ->
  Shape.Both<{U32.is_lt(U32.from_nat(index),Tree.capacity(Element,BandUpsampleExecution.fresh_tree(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale)))) == True{} : Bool},
    CopyValues.Point<Element,BandUpsampleExecution.value(~Element,~Context,~original,BandUpsampleExecution.Context{context,geometry,scale},Bands.global_index(BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.from_nat(index))),
      U32.from_nat(index),BandUpsampleExecution.leaf_writes(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),input)>>:
  BandUpsampleSamples.ActualLeaf{+descriptor,+guarded,+stores,+source_samples} = evidence
  +evidence = {BandUpsampleSamples.ActualLeaf{descriptor,guarded,stores,source_samples} : BandUpsampleSamples.ActualLeaf<Element,Context,original,context,geometry,scale,fill,first,rows,input>}
  nearest_point_from_source(~Element,~Context,~original,context,geometry,shape,scale,fill,first,rows,input,index,descriptor,metadata,ready,inside,original_shape,truth,
    BandUpsampleSamples.actual_source_point(~Element,~Context,~original,context,geometry,scale,fill,first,rows,input,index,evidence,inside))

def domains_source(+geometry: Bands.Geometry,+scale: U32,+limits: Machine.Limits,domains: Domains<geometry,scale,limits>) -> BandGather.Domain<geometry,limits,Machine.capacity(limits)>:
  Domains{source,target} = domains
  source

def frames_target(+geometry: Bands.Geometry,+scale: U32,+first: U32,+rows: U32,+length: Nat,+output_count: U32,+limits: Machine.Limits,frames: Frames<geometry,scale,first,rows,length,output_count,limits>) ->
  BandGather.Facts<BandGather.Call{BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.to_nat(output_count),Machine.capacity(limits),Machine.half_word(limits)}>:
  Frames{source_domain,target_domain,source_envelope,target_envelope,source_facts,target_facts,words,height_exact,width_exact,scale_positive,scale_bound} = frames
  target_facts

# One cell of an upsampled leaf holds the nearest source value of its pixel.
# The copy addresses the leaf with flat positions: the position of the cell
# names its store and BandGather.indexed_pixel names its pixel.
def nearest_cell(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+channel: Nat,+row: Nat,+column: Nat,
  +metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,+original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool},
  evidence: BandUpsampleSamples.ActualLeaf<Element,Context,original,context,geometry,scale,fill,first,rows,input>,
  +channel_inside: {Nat.is_lt(channel,BandValues.channels(BandUpsampleExecution.scaled_geometry(geometry,scale))) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat((rows * scale : U32))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,BandValues.width(BandUpsampleExecution.scaled_geometry(geometry,scale))) == True{} : Bool}) ->
  BandValues.At(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),
    Writes.pure(Element,BandUpsampleExecution.leaf_writes(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),input),BandUpsampleExecution.fresh_tree(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale))),U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale)),channel,row,column):
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  BandUpsampleSamples.ActualLeaf{+descriptor,+guarded,CopyValues.Finite{+stores_inside,+points},+source_samples} = evidence
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +evidence = {BandUpsampleSamples.ActualLeaf{descriptor,guarded,CopyValues.Finite{stores_inside,points},source_samples} : BandUpsampleSamples.ActualLeaf<Element,Context,original,context,geometry,scale,fill,first,rows,input>}
  +legal = BandUpsampleExecution.legal(geometry,shape,scale,truth)
  +dimensions = {BandUpsampleSamples.enabled_dimensions(geometry,shape,scale,original_shape,truth) : BandUpsampleSamples.Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)>}
  +index = BandValues.position(BandUpsampleExecution.scaled_geometry(geometry,scale),(rows * scale : U32),channel,row,column)
  +inside = BandValues.position_within(BandUpsampleExecution.scaled_geometry(geometry,scale),(rows * scale : U32),U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale)),channel,row,column,
    BandValues.leaf_length(BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale)),
      BandUpsampleExecution.placement_metadata(geometry,scale,first,rows,
        BandUpsampleSamples.placement(~Element,geometry,shape,scale,first,rows,input,metadata,ready,
          BandUpsampleSamples.active_envelope(geometry,shape,scale,BandUpsampleSamples.channels_positive_dimensions(geometry,dimensions),BandUpsampleSamples.width_positive_dimensions(geometry,dimensions),original_shape,legal),original_shape,legal))),
    channel_inside,row_inside,column_inside)
  +location = {repeated_location(~Element,geometry,first,rows,scale,input,BandUpsampleExecution.producer_output_count(geometry,rows,scale),index,descriptor,metadata,ready,
    BandUpsampleExecution.scale_positive(geometry,shape,scale,legal),BandUpsampleSamples.width_positive_dimensions(geometry,dimensions),inside) : LocalSampleIndex<geometry,rows,scale,index>}
  +pixel_exact = BandGather.indexed_pixel(BandGather.Call{BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale)),Machine.capacity(Default.limits()),Machine.half_word(Default.limits())},channel,row,column,
    frames_target(geometry,scale,first,rows,Buffer.length(~Element,input),BandUpsampleExecution.producer_output_count(geometry,rows,scale),Default.limits(),
      enabled_frames(~Element,geometry,shape,scale,fill,first,rows,input,descriptor,metadata,ready,located_rows_positive(geometry,rows,scale,index,location),original_shape,truth)),
    channel_inside,row_inside,column_inside,inside)
  BandUpsampleExecution.program_cell(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),
    BandUpsampleExecution.fresh_tree(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale)),U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale)),BandUpsampleExecution.leaf_writes(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),input),channel,row,column,
    Buffer.content_ready(~Element,BandCount.model(~Element,BandUpsampleExecution.producer_output_count(geometry,rows,scale),fill),BandCount.model_ready(~Element,BandUpsampleExecution.producer_output_count(geometry,rows,scale),fill,BandUpsampleExecution.allocated(geometry,rows,scale,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),descriptor))),
    stores_inside,inside,
    %pixel_exact : Shape.Both<{U32.is_lt(U32.from_nat(index),Tree.capacity(Element,BandUpsampleExecution.fresh_tree(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale)))) == True{} : Bool},
      CopyValues.Point<Element,BandUpsampleExecution.value(~Element,~Context,~original,BandUpsampleExecution.Context{context,geometry,scale},_),U32.from_nat(index),BandUpsampleExecution.leaf_writes(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),input)>>
    nearest_point(~Element,~Context,~original,context,geometry,shape,scale,fill,first,rows,input,index,metadata,ready,inside,original_shape,truth,evidence))

# What an upsampled leaf supplies to every cell of its output.
type NearestCells<-Element: Data,-Context: Data,-original: Context -> U32 -> Element> is Data:
  NearestCells{context: Context,geometry: Bands.Geometry,shape: List<&2,U32>,scale: U32,fill: Element,first: U32,rows: U32,input: Buffer.Model<Element>,
    metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Sized<Element,input>,original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool},
    actual: BandUpsampleSamples.ActualLeaf<Element,Context,original,context,geometry,scale,fill,first,rows,input>}

def nearest_geometry(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,evidence: NearestCells<Element,Context,original>) -> Bands.Geometry:
  NearestCells{context,geometry,shape,scale,fill,first,rows,input,metadata,ready,original_shape,truth,actual} = evidence
  BandUpsampleExecution.scaled_geometry(geometry,scale)

def nearest_rows(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,evidence: NearestCells<Element,Context,original>) -> U32:
  NearestCells{context,geometry,shape,scale,fill,first,rows,input,metadata,ready,original_shape,truth,actual} = evidence
  (rows * scale : U32)

def NearestCell(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+evidence: NearestCells<Element,Context,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  NearestCells{+context,+geometry,+shape,+scale,+fill,+first,+rows,+input,+metadata,+ready,+original_shape,+truth,+actual} = evidence
  BandValues.At(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},BandUpsampleExecution.scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),
    Writes.pure(Element,BandUpsampleExecution.leaf_writes(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),input),BandUpsampleExecution.fresh_tree(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale))),U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale)),channel,row,column)

def nearest_leaf_cell(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+evidence: NearestCells<Element,Context,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(nearest_geometry(~Element,~Context,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(nearest_rows(~Element,~Context,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(nearest_geometry(~Element,~Context,~original,evidence))) == True{} : Bool}) ->
  NearestCell(~Element,~Context,~original,evidence,channel,row,column):
  NearestCells{+context,+geometry,+shape,+scale,+fill,+first,+rows,+input,+metadata,+ready,+original_shape,+truth,+actual} = evidence
  nearest_cell(~Element,~Context,~original,context,geometry,shape,scale,fill,first,rows,input,channel,row,column,metadata,ready,original_shape,truth,actual,channel_inside,row_inside,column_inside)

def leaf_values_from_actual(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,+original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool},
  evidence: BandUpsampleSamples.ActualLeaf<Element,Context,original,context,geometry,scale,fill,first,rows,input>) ->
  BandValues.Values(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},BandUpsampleExecution.scaled_geometry(geometry,scale),
    Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input)))):
  BandUpsampleSamples.ActualLeaf{+descriptor,+guarded,+stores,+source_samples} = evidence
  +actual = {BandUpsampleSamples.ActualLeaf{descriptor,guarded,stores,source_samples} : BandUpsampleSamples.ActualLeaf<Element,Context,original,context,geometry,scale,fill,first,rows,input>}
  BandUpsampleExecution.leaf_values(~Element,~Context,~original,context,geometry,scale,fill,first,rows,input,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),descriptor,guarded,
    BandValues.cells(~NearestCells<Element,Context,original>,~NearestCell(~Element,~Context,~original),~nearest_geometry(~Element,~Context,~original),~nearest_rows(~Element,~Context,~original),~nearest_leaf_cell(~Element,~Context,~original),
      NearestCells{context,geometry,shape,scale,fill,first,rows,input,metadata,ready,original_shape,truth,actual}))

def enabled_leaf_values(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,+values: BandValues.Leaf(~Element,~Context,~original,context,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input)),
  +original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  BandValues.Values(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},BandUpsampleExecution.scaled_geometry(geometry,scale),
    Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input)))):
  leaf_values_from_actual(~Element,~Context,~original,context,geometry,shape,scale,fill,first,rows,input,metadata,ready,original_shape,truth,BandUpsampleSamples.enabled_leaf(~Element,~Context,~original,context,geometry,shape,scale,fill,first,rows,input,
    domains_source(geometry,scale,Default.limits(),enabled_domains(geometry,shape,scale,original_shape,truth)),metadata,ready,values,original_shape,truth))

def enabled_model_values(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,bands: Model.Model<Element>,
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands),values: BandValues.Values(~Element,~Context,~original,context,geometry,Model.content(~Element,bands)),
  +original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  BandValues.Values(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},BandUpsampleExecution.scaled_geometry(geometry,scale),BandUpsampleExecution.model(~Element,geometry,scale,fill,bands)):
  match bands:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      enabled_leaf_values(~Element,~Context,~original,context,geometry,shape,scale,fill,first,rows,Buffer.Model{tree,length,certificate},metadata,ready,values,original_shape,truth)
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_values,right_values} = values
      Shape.Both{enabled_model_values(~Element,~Context,~original,context,geometry,shape,scale,fill,left,left_metadata,left_ready,left_values,original_shape,truth),
        enabled_model_values(~Element,~Context,~original,context,geometry,shape,scale,fill,right,right_metadata,right_ready,right_values,original_shape,truth)}

def enabled_scheduled_values(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  +metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),+ready: Model.Ready(~Element,bands),+values: BandValues.Values(~Element,~Context,~original,context,geometry,Model.content(~Element,bands)),
  +original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  BandValues.Values(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},BandUpsampleExecution.scaled_geometry(geometry,scale),
    Model.contents(~Element,BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands)))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands))),
    BandUpsampleExecution.model(~Element,geometry,scale,fill,bands),BandUpsampleExecution.scheduled(~Element,geometry,scale,fill,workers,bands)) :
    BandValues.Values(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},BandUpsampleExecution.scaled_geometry(geometry,scale),_)
  enabled_model_values(~Element,~Context,~original,context,geometry,shape,scale,fill,bands,metadata,ready,values,original_shape,truth)

def enabled_scheduled(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,+values: BandValues.Values(~Element,~Context,~original,context,geometry,Model.content(~Element,bands)),
  +original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  Shape.Both<BandShape.WellFormed<Element,BandUpsampleExecution.scaled_geometry(geometry,scale),Model.contents(~Element,BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands)))>,
    BandValues.Values(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},BandUpsampleExecution.scaled_geometry(geometry,scale),
      Model.contents(~Element,BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands))))>:
  BandShape.WellFormed{+metadata,+ready,+coverage} = well_formed
  +well_formed = {BandShape.WellFormed{metadata,ready,coverage} : BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>}
  Shape.Both{BandUpsampleSamples.enabled_well_formed(~Element,geometry,shape,scale,fill,workers,bands,well_formed,original_shape,truth),
    enabled_scheduled_values(~Element,~Context,~original,context,geometry,shape,scale,fill,workers,bands,metadata,Model.ready(~Element,bands,ready),values,original_shape,truth)}

# A CHW tensor has the same band geometry as a single-image NCHW tensor.
# Public labels stay CHW; only the erased geometry arithmetic is normalized.
type ShapeCoherence<-geometry: Bands.Geometry,-shape: List<&2,U32>> is Data:
  Batched{exact: {shape == BandCount.extents(geometry) : List<&2,U32>}}
  Unbatched{image: {BandUpsampleExecution.producer_batch(geometry) == 1 : U32},exact: {shape == [BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)] : List<&2,U32>}}

def leading_one_elements(+extents: List<&2,U32>) -> {ShapeLaws.elements(1 <> extents) == ShapeLaws.elements(extents) : Nat}:
  Addition.zero_right(ShapeLaws.elements(extents))

def and_true(+left: Bool,+right: Bool,+left_true: {left == True{} : Bool},right_true: {right == True{} : Bool}) -> {left && right == True{} : Bool}:
  %Equal.sym(Bool,left,True{},left_true) : {_ && right == True{} : Bool}
  right_true

def and_true_right(+left: Bool) -> {left && True{} == left : Bool}:
  match left:
    case True{}: {==}
    case False{}: {==}

def unbatched_enabled_from_count(+channels: U32,+height: U32,+width: U32,+scale: U32,+count: U32,
  +truth: {BandUpsampleExecution.enabled(Bands.Geometry{1,channels,height,width},[channels,height,width],scale) == True{} : Bool},
  evidence: Count.Count<ShapeLaws.elements(Layout.scaled_extents([channels,height,width],scale)),count>) ->
  {BandUpsampleExecution.enabled(Bands.Geometry{1,channels,height,width},[1,channels,height,width],scale) == True{} : Bool}:
  Count.Count{+exact,+bounded} = evidence
  +canonical = {[1,channels,(height * scale : U32),(width * scale : U32)] : List<&2,U32>}
  +counted = CountCompletion.count_of(canonical,count,Equal.trans(Nat,ShapeLaws.elements(canonical),ShapeLaws.elements(Layout.scaled_extents([channels,height,width],scale)),U32.to_nat(count),leading_one_elements(Layout.scaled_extents([channels,height,width],scale)),exact),bounded)
  +legal = BandUpsampleExecution.legal(Bands.Geometry{1,channels,height,width},[channels,height,width],scale,truth)
  +prefix = Bool.not(U32.is_zero(scale)) && Views.has_count(Shapes.checked_multiply(height,scale)) && Views.has_count(Shapes.checked_multiply(width,scale))
  +checks = Boolean.left(prefix,Views.has_count(Tensors.count_of(Layout.scaled_extents([channels,height,width],scale))),legal)
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(canonical),Some{count},counted) :
    {(prefix && Views.has_count(_)) && Bool.not(Shapes.has_zero(canonical)) == True{} : Bool}
  %Equal.sym(Bool,prefix && True{},prefix,and_true_right(prefix)) : {_ && Bool.not(Shapes.has_zero(canonical)) == True{} : Bool}
  and_true(prefix,Bool.not(Shapes.has_zero(canonical)),checks,
    %Equal.sym(Bool,Shapes.has_zero(canonical),False{},BandUpsampleExecution.nonempty(Bands.Geometry{1,channels,height,width},[channels,height,width],scale,truth)) : {Bool.not(_) == True{} : Bool} {==})

def canonical_enabled(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,coherence: ShapeCoherence<geometry,shape>,
  +truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) -> {BandUpsampleExecution.enabled(geometry,BandCount.extents(geometry),scale) == True{} : Bool}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  match coherence:
    case Batched{exact}:
      %exact : {BandUpsampleExecution.enabled(geometry,_,scale) == True{} : Bool} truth
    case Unbatched{+image,+exact}:
      %Equal.sym(U32,images,1,image) : {BandUpsampleExecution.enabled(Bands.Geometry{_,channels,height,width},BandCount.extents(Bands.Geometry{_,channels,height,width}),scale) == True{} : Bool}
      unbatched_enabled_from_count(channels,height,width,scale,Checked.value(Tensors.count_of(Layout.scaled_extents([channels,height,width],scale))),
        %image : {BandUpsampleExecution.enabled(Bands.Geometry{_,channels,height,width},[channels,height,width],scale) == True{} : Bool}
        %exact : {BandUpsampleExecution.enabled(geometry,_,scale) == True{} : Bool} truth,
        BandUpsampleExecution.guards_allocation(Bands.Geometry{1,channels,height,width},[channels,height,width],scale,
          BandUpsampleExecution.legal(Bands.Geometry{1,channels,height,width},[channels,height,width],scale,
            %image : {BandUpsampleExecution.enabled(Bands.Geometry{_,channels,height,width},[channels,height,width],scale) == True{} : Bool}
            %exact : {BandUpsampleExecution.enabled(geometry,_,scale) == True{} : Bool} truth)))

def enabled_public_equation(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  +truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  {Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale) ==
    Tensors.Banded{BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands)),
      BandUpsampleExecution.scaled_geometry(geometry,scale),Layout.scaled_extents(shape,scale),fill,Tensors.Usable{},workers} : Tensors.Tensor<Element>}:
  Bands.Geometry{images,channels,height,width} = geometry
  %Equal.sym(Bool,BandUpsampleExecution.enabled(geometry,shape,scale),True{},truth) :
    {Layout.upsample_supported(~Element,_,Model.pack(~Element,bands),geometry,shape,scale,fill,workers) ==
      Tensors.Banded{BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands)),
        BandUpsampleExecution.scaled_geometry(geometry,scale),Layout.scaled_extents(shape,scale),fill,Tensors.Usable{},workers} : Tensors.Tensor<Element>}
  {==}

def enabled_public(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,+values: BandValues.Values(~Element,~Context,~original,context,geometry,Model.content(~Element,bands)),
  +coherence: ShapeCoherence<geometry,shape>,+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  Shape.Both<BandMap.WellFormed(~Element,Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale)),
    BandMap.LogicalValues(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},
      Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale))>:
  %Equal.sym(Tensors.Tensor<Element>,Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale),
    Tensors.Banded{BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands)),
      BandUpsampleExecution.scaled_geometry(geometry,scale),Layout.scaled_extents(shape,scale),fill,Tensors.Usable{},workers},enabled_public_equation(~Element,geometry,shape,scale,fill,workers,bands,truth)) :
    Shape.Both<BandMap.WellFormed(~Element,_),BandMap.LogicalValues(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},_)>
  enabled_scheduled(~Element,~Context,~original,context,geometry,BandCount.extents(geometry),scale,fill,workers,bands,well_formed,values,{==},canonical_enabled(geometry,shape,scale,coherence,truth))

def matching_dimensions(+images: U32,+channels: U32,+height: U32,+width: U32) -> {U32.is_eq(images,images) && U32.is_eq(channels,channels) && U32.is_eq(height,height) && U32.is_eq(width,width) == True{} : Bool}:
  and_true(U32.is_eq(images,images) && U32.is_eq(channels,channels) && U32.is_eq(height,height),U32.is_eq(width,width),
    and_true(U32.is_eq(images,images) && U32.is_eq(channels,channels),U32.is_eq(height,height),
      and_true(U32.is_eq(images,images),U32.is_eq(channels,channels),BandShape.equal_words(images,images,{==}),BandShape.equal_words(channels,channels,{==})),BandShape.equal_words(height,height,{==})),BandShape.equal_words(width,width,{==}))

def coherent_guard(+geometry: Bands.Geometry,+shape: List<&2,U32>,coherence: ShapeCoherence<geometry,shape>) -> {Window.band_shape(shape,geometry) == True{} : Bool}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  match coherence:
    case Batched{exact}:
      %Equal.sym(List<&2,U32>,shape,BandCount.extents(geometry),exact) : {Window.band_shape(_,geometry) == True{} : Bool}
      matching_dimensions(images,channels,height,width)
    case Unbatched{+image,+exact}:
      %Equal.sym(U32,images,1,image) : {Window.band_shape(shape,Bands.Geometry{_,channels,height,width}) == True{} : Bool}
      %Equal.sym(List<&2,U32>,shape,[channels,height,width],exact) : {Window.band_shape(_,Bands.Geometry{1,channels,height,width}) == True{} : Bool}
      matching_dimensions(1,channels,height,width)

def DecodedCoherence(+shape: List<&2,U32>,result: Maybe<&2,Bands.Geometry>) -> Data:
  match result:
    case None{}: Unit
    case Some{geometry}: ShapeCoherence<geometry,shape>

def decoded_coherence(shape: List<&2,U32>) -> DecodedCoherence(shape,BandCount.decode(shape)):
  match shape:
    case Con{+channels,Con{+height,Con{+width,Nil{}}}}: Unbatched{{==},{==}}
    case Con{+images,Con{+channels,Con{+height,Con{+width,Nil{}}}}}: Batched{{==}}
    case Nil{}: Unit{}
    case Con{a,Nil{}}: Unit{}
    case Con{a,Con{b,Nil{}}}: Unit{}
    case Con{a,Con{b,Con{c,Con{d,Con{e,rest}}}}}: Unit{}

def coherence_from_decoded(shape: List<&2,U32>,+geometry: Bands.Geometry,+decoded: {BandCount.decode(shape) == Some{geometry} : Maybe<&2,Bands.Geometry>}) -> ShapeCoherence<geometry,shape>:
  %decoded : DecodedCoherence(shape,_)
  decoded_coherence(shape)

def scaled_coherence(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,coherence: ShapeCoherence<geometry,shape>) -> ShapeCoherence<BandUpsampleExecution.scaled_geometry(geometry,scale),Layout.scaled_extents(shape,scale)>:
  Bands.Geometry{images,channels,height,width} = geometry
  match coherence:
    case Batched{exact}: Batched{BandUpsampleExecution.scaled_shape(geometry,shape,scale,exact)}
    case Unbatched{image,exact}:
      Unbatched{image,Equal.cong(List<&2,U32>,List<&2,U32>,shape => Layout.scaled_extents(shape,scale),shape,[channels,height,width],exact)}

def tensor_status(~Element: Data,tensor: Tensors.Tensor<Element>) -> Tensors.Status:
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,workers}: status
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: status

def tensor_shape(~Element: Data,tensor: Tensors.Tensor<Element>) -> List<&2,U32>:
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,workers}: shape
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: shape

def logical_values(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,tensor: Tensors.Tensor<Element>) -> Data:
  match tensor:
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: BandValues.Values(~Element,~Context,~original,context,geometry,Model.contents(~Element,bands))
    case Tensors.Tensor{storage,shape,fill,status,workers}:
      Members.All(Nat,{index => {Buffer.value(~Element,U32.from_nat(index),Buffer.contents(~Element,storage)) == original(context,U32.from_nat(index)) : Element} : Nat -> Data},Ranges.range(ShapeLaws.elements(shape),0n))

def UpsampleResult(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,tensor: Tensors.Tensor<Element>) -> Data:
  Shape.Both<Global.Normal(~Element,tensor),Shape.Both<{tensor_status(~Element,tensor) == Tensors.Usable{} : Tensors.Status},
    Shape.Both<{tensor_shape(~Element,tensor) == shape : List<&2,U32>},logical_values(~Element,~Context,~original,context,tensor)>>>

def enabled_result_from_scheduled(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  +coherence: ShapeCoherence<geometry,shape>,+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool},
  evidence: Shape.Both<BandShape.WellFormed<Element,BandUpsampleExecution.scaled_geometry(geometry,scale),Model.contents(~Element,BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands)))>,
    BandValues.Values(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},BandUpsampleExecution.scaled_geometry(geometry,scale),
      Model.contents(~Element,BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands))))>) ->
  UpsampleResult(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),
    Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale)):
  Bands.Geometry{images,channels,height,width} = geometry
  Shape.Both{well_formed,values} = evidence
  %Equal.sym(Tensors.Tensor<Element>,Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale),
    Tensors.Banded{BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands)),BandUpsampleExecution.scaled_geometry(geometry,scale),Layout.scaled_extents(shape,scale),fill,Tensors.Usable{},workers},enabled_public_equation(~Element,geometry,shape,scale,fill,workers,bands,truth)) :
    UpsampleResult(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),_)
  Shape.Both{Shape.Both{well_formed,Shape.Both{Global.Counted{Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale))),
    BandUpsampleExecution.found(Tensors.count_of(Layout.scaled_extents(shape,scale)),Boolean.right(Bool.not(U32.is_zero(scale)) && Views.has_count(Shapes.checked_multiply(BandUpsampleExecution.producer_height(geometry),scale)) && Views.has_count(Shapes.checked_multiply(BandUpsampleExecution.producer_width(geometry),scale)),
      Views.has_count(Tensors.count_of(Layout.scaled_extents(shape,scale))),BandUpsampleExecution.legal(geometry,shape,scale,truth)))},
    coherent_guard(BandUpsampleExecution.scaled_geometry(geometry,scale),Layout.scaled_extents(shape,scale),scaled_coherence(geometry,shape,scale,coherence))}},Shape.Both{{==},Shape.Both{{==},values}}}

def enabled_result(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,+values: BandValues.Values(~Element,~Context,~original,context,geometry,Model.content(~Element,bands)),+coherence: ShapeCoherence<geometry,shape>,+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  UpsampleResult(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),
    Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale)):
  enabled_result_from_scheduled(~Element,~Context,~original,context,geometry,shape,scale,fill,workers,bands,coherence,truth,
    enabled_scheduled(~Element,~Context,~original,context,geometry,BandCount.extents(geometry),scale,fill,workers,bands,well_formed,values,{==},canonical_enabled(geometry,shape,scale,coherence,truth)))

def coherent_zero(+geometry: Bands.Geometry,+shape: List<&2,U32>,coherence: ShapeCoherence<geometry,shape>) -> {Shapes.has_zero(shape) == Shapes.has_zero(BandCount.extents(geometry)) : Bool}:
  Bands.Geometry{+images,channels,height,width} = geometry
  match coherence:
    case Batched{exact}: Equal.cong(List<&2,U32>,Bool,Shapes.has_zero,shape,BandCount.extents(geometry),exact)
    case Unbatched{+image,+exact}:
      %Equal.sym(U32,images,1,image) : {Shapes.has_zero(shape) == Shapes.has_zero(BandCount.extents(Bands.Geometry{_,channels,height,width})) : Bool}
      Equal.cong(List<&2,U32>,Bool,Shapes.has_zero,shape,[channels,height,width],exact)

def positive_word_not_zero(+word: U32,+positive: {Nat.is_lt(0n,U32.to_nat(word)) == True{} : Bool}) -> {U32.is_zero(word) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_zero(word),Nat.is_eq(U32.to_nat(word),0n),Comparison.eq(word,0)) : {_ == False{} : Bool}
  BandUpsampleSamples.positive_nonzero(U32.to_nat(word),positive)

def scaled_nonempty_dimensions(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool},
  dimensions: BandUpsampleSamples.Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)>) ->
  {Shapes.has_zero(BandCount.extents(BandUpsampleExecution.scaled_geometry(geometry,scale))) == False{} : Bool}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandUpsampleSamples.Dimensions{+images_positive,+channels_positive,+height_positive,+width_positive} = dimensions
  +scale_positive = BandUpsampleExecution.scale_positive(geometry,shape,scale,truth)
  +scaled_height_positive = {
    %BandUpsampleExecution.native_exact(height,scale,BandUpsampleExecution.height_scaled(geometry,shape,scale,truth)) : {Nat.is_lt(0n,_) == True{} : Bool}
    CountCompletion.positive_product(U32.to_nat(height),U32.to_nat(scale),height_positive,scale_positive) : {Nat.is_lt(0n,U32.to_nat((height * scale : U32))) == True{} : Bool}}
  +scaled_width_positive = {
    %BandUpsampleExecution.native_exact(width,scale,BandUpsampleExecution.width_scaled(geometry,shape,scale,truth)) : {Nat.is_lt(0n,_) == True{} : Bool}
    CountCompletion.positive_product(U32.to_nat(width),U32.to_nat(scale),width_positive,scale_positive) : {Nat.is_lt(0n,U32.to_nat((width * scale : U32))) == True{} : Bool}}
  %Equal.sym(Bool,U32.is_zero(images),False{},positive_word_not_zero(images,images_positive)) : {_ || Shapes.has_zero([channels,(height * scale : U32),(width * scale : U32)]) == False{} : Bool}
  %Equal.sym(Bool,U32.is_zero(channels),False{},positive_word_not_zero(channels,channels_positive)) : {_ || Shapes.has_zero([(height * scale : U32),(width * scale : U32)]) == False{} : Bool}
  %Equal.sym(Bool,U32.is_zero((height * scale : U32)),False{},positive_word_not_zero((height * scale : U32),scaled_height_positive)) : {_ || Shapes.has_zero([(width * scale : U32)]) == False{} : Bool}
  %Equal.sym(Bool,U32.is_zero((width * scale : U32)),False{},positive_word_not_zero((width * scale : U32),scaled_width_positive)) : {_ || False{} == False{} : Bool}
  {==}

def source_empty_when(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,empty: Bool,+classified: {Shapes.has_zero(shape) == empty : Bool},+coherence: ShapeCoherence<geometry,shape>,
  +truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool},+scaled_empty: {Shapes.has_zero(Layout.scaled_extents(shape,scale)) == True{} : Bool}) -> {Shapes.has_zero(shape) == True{} : Bool}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  match empty:
    case True{}: classified
    case False{}:
      +canonical_nonempty = {%coherent_zero(geometry,shape,coherence) : {_ == False{} : Bool} classified : {Shapes.has_zero(BandCount.extents(geometry)) == False{} : Bool}}
      +canonical_empty = {%coherent_zero(BandUpsampleExecution.scaled_geometry(geometry,scale),Layout.scaled_extents(shape,scale),scaled_coherence(geometry,shape,scale,coherence)) : {_ == True{} : Bool} scaled_empty : {Shapes.has_zero(BandCount.extents(BandUpsampleExecution.scaled_geometry(geometry,scale))) == True{} : Bool}}
      Empty.absurd({Shapes.has_zero(shape) == True{} : Bool},WordEncoding.false_true(
        %scaled_nonempty_dimensions(geometry,shape,scale,truth,BandUpsampleSamples.rank_four_dimensions(images,channels,height,width,canonical_nonempty)) : {_ == True{} : Bool} canonical_empty))

def source_empty(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+coherence: ShapeCoherence<geometry,shape>,+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool},
  scaled_empty: {Shapes.has_zero(Layout.scaled_extents(shape,scale)) == True{} : Bool}) -> {Shapes.has_zero(shape) == True{} : Bool}:
  source_empty_when(geometry,shape,scale,Shapes.has_zero(shape),{==},coherence,truth,scaled_empty)

def native_product(extents: List<&2,U32>,+accumulator: U32) -> U32:
  match extents:
    case []: accumulator
    case head <> tail: native_product(tail,(accumulator * head : U32))

def native_product_zero(extents: List<&2,U32>) -> {native_product(extents,0) == 0 : U32}:
  match extents:
    case []: {==}
    case +head <> +tail:
      %Equal.sym(U32,(0 * head : U32),0,Arithmetic.multiply_zero_left(head)) : {native_product(tail,_) == 0 : U32}
      native_product_zero(tail)

def native_empty_selected(+head: U32,+tail: List<&2,U32>,+accumulator: U32,zero: Bool,+classified: {U32.is_zero(head) == zero : Bool},
  +empty: {Shapes.has_zero(head <> tail) == True{} : Bool},tail_empty: {Shapes.has_zero(tail) == True{} : Bool} -> {native_product(tail,(accumulator * head : U32)) == 0 : U32}) ->
  {native_product(head <> tail,accumulator) == 0 : U32}:
  match zero:
    case True{}:
      %Equal.sym(U32,head,0,ShapeLaws.native_equal(head,0,classified)) : {native_product(tail,(accumulator * _ : U32)) == 0 : U32}
      %Equal.sym(U32,(accumulator * 0 : U32),0,Equal.trans(U32,(accumulator * 0 : U32),U32.from_nat(Nat.mul(U32.to_nat(accumulator),0n)),0,Multiplication.native_multiply(accumulator,0),Equal.cong(Nat,U32,U32.from_nat,Nat.mul(U32.to_nat(accumulator),0n),0n,RowMajor.multiply_zero_right(U32.to_nat(accumulator))))) : {native_product(tail,_) == 0 : U32}
      native_product_zero(tail)
    case False{}:
      tail_empty({%classified : {_ || Shapes.has_zero(tail) == True{} : Bool} empty : {Shapes.has_zero(tail) == True{} : Bool}})

def native_empty_product(extents: List<&2,U32>,+accumulator: U32,+empty: {Shapes.has_zero(extents) == True{} : Bool}) -> {native_product(extents,accumulator) == 0 : U32}:
  match extents:
    case []: Empty.absurd({accumulator == 0 : U32},WordEncoding.false_true(empty))
    case +head <> +tail:
      native_empty_selected(head,tail,accumulator,U32.is_zero(head),{==},empty,tail_empty => native_empty_product(tail,(accumulator * head : U32),tail_empty))

def geometry_zero(+geometry: Bands.Geometry,+shape: List<&2,U32>,+coherence: ShapeCoherence<geometry,shape>,+empty: {Shapes.has_zero(shape) == True{} : Bool}) -> {Bands.elements(geometry) == 0 : U32}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  %native_one_left(images) : {native_product([channels,height,width],_) == 0 : U32}
  native_empty_product(BandCount.extents(geometry),1,{%coherent_zero(geometry,shape,coherence) : {_ == True{} : Bool} empty : {Shapes.has_zero(BandCount.extents(geometry)) == True{} : Bool}})

def repeated_extents(extents: List<&2,U32>,+scale: U32) -> List<&2,U32>:
  match extents:
    case [height,width]: [height,scale,width,scale]
    case head <> tail: head <> repeated_extents(tail,scale)
    case []: []

def repeat_empty_pair(+height_zero: Bool,+width_zero: Bool,+scale_zero: Bool,empty: {height_zero || (width_zero || False{}) == True{} : Bool}) ->
  {height_zero || (scale_zero || (width_zero || (scale_zero || False{}))) == True{} : Bool}:
  match height_zero width_zero:
    case True{} width_zero: {==}
    case False{} True{}: Boolean.or_true_right(scale_zero)
    case False{} False{}: Empty.absurd({scale_zero || (False{} || (scale_zero || False{})) == True{} : Bool},WordEncoding.false_true(empty))

def repeat_empty_head(+head: U32,+tail: List<&2,U32>,+scale: U32,zero: Bool,+classified: {U32.is_zero(head) == zero : Bool},+empty: {Shapes.has_zero(head <> tail) == True{} : Bool},
  tail_empty: {Shapes.has_zero(tail) == True{} : Bool} -> {Shapes.has_zero(repeated_extents(tail,scale)) == True{} : Bool}) ->
  {Shapes.has_zero(head <> repeated_extents(tail,scale)) == True{} : Bool}:
  match zero:
    case True{}:
      %Equal.sym(Bool,U32.is_zero(head),True{},classified) : {_ || Shapes.has_zero(repeated_extents(tail,scale)) == True{} : Bool}
      {==}
    case False{}:
      %Equal.sym(Bool,U32.is_zero(head),False{},classified) : {_ || Shapes.has_zero(repeated_extents(tail,scale)) == True{} : Bool}
      tail_empty({%classified : {_ || Shapes.has_zero(tail) == True{} : Bool} empty : {Shapes.has_zero(tail) == True{} : Bool}})

def repeat_empty_three(+channels: U32,+height: U32,+width: U32,+scale: U32,+empty: {Shapes.has_zero([channels,height,width]) == True{} : Bool}) -> {Shapes.has_zero([channels,height,scale,width,scale]) == True{} : Bool}:
  repeat_empty_head(channels,[height,width],scale,U32.is_zero(channels),{==},empty,tail_empty => repeat_empty_pair(U32.is_zero(height),U32.is_zero(width),U32.is_zero(scale),tail_empty))

def repeat_empty_four(+images: U32,+channels: U32,+height: U32,+width: U32,+scale: U32,+empty: {Shapes.has_zero([images,channels,height,width]) == True{} : Bool}) -> {Shapes.has_zero([images,channels,height,scale,width,scale]) == True{} : Bool}:
  repeat_empty_head(images,[channels,height,width],scale,U32.is_zero(images),{==},empty,tail_empty => repeat_empty_three(channels,height,width,scale,tail_empty))

type EmptySource<-shape: List<&2,U32>,-scale: U32> is Data:
  EmptySource{source: Views.View,actual: {Layout.repeated(Tensors.view_of(shape),scale) == Some{source} : Maybe<&2,Views.View>},
    extents: {Views.axis_extents_of(source) == repeated_extents(shape,scale) : List<&2,U32>},offset: {CopyValues.view_offset(source) == 0 : U32},
    empty: {Shapes.has_zero(Views.axis_extents_of(source)) == True{} : Bool}}

def empty_source_four(+images: U32,+channels: U32,+height: U32,+width: U32,+scale: U32,+empty: {Shapes.has_zero([images,channels,height,width]) == True{} : Bool}) -> EmptySource<[images,channels,height,width],scale>:
  +shape = {[images,channels,height,width] : List<&2,U32>}
  +source = {Views.View{[Views.Axis{images,(((0 * width : U32) * height : U32) * channels : U32)},Views.Axis{channels,((0 * width : U32) * height : U32)},Views.Axis{height,(0 * width : U32)},Views.Axis{scale,0},Views.Axis{width,0},Views.Axis{scale,0}],0} : Views.View}
  EmptySource{source,
    %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(shape),Some{BandUpsampleExecution.canonical_view(shape,0)},BandUpsampleExecution.view(shape,0,ShapeLaws.empty_product(shape,empty))) : {Layout.repeated(_,scale) == Some{source} : Maybe<&2,Views.View>} {==},
    {==},{==},repeat_empty_four(images,channels,height,width,scale,empty)}

def empty_source_three(+channels: U32,+height: U32,+width: U32,+scale: U32,+empty: {Shapes.has_zero([channels,height,width]) == True{} : Bool}) -> EmptySource<[channels,height,width],scale>:
  EmptySource{BandUpsampleExecution.repeated_view(Bands.Geometry{1,channels,height,width},height,scale,0),
    BandUpsampleExecution.repeated(Bands.Geometry{1,channels,height,width},height,scale,0,ShapeLaws.empty_product([channels,height,width],empty)),{==},{==},repeat_empty_three(channels,height,width,scale,empty)}

def empty_source(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,coherence: ShapeCoherence<geometry,shape>,+empty: {Shapes.has_zero(shape) == True{} : Bool}) -> EmptySource<shape,scale>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  match coherence:
    case Batched{+exact}:
      %Equal.sym(List<&2,U32>,shape,BandCount.extents(geometry),exact) : EmptySource<_,scale>
      empty_source_four(images,channels,height,width,scale,{%exact : {Shapes.has_zero(_) == True{} : Bool} empty : {Shapes.has_zero(BandCount.extents(geometry)) == True{} : Bool}})
    case Unbatched{image,+exact}:
      %Equal.sym(List<&2,U32>,shape,[channels,height,width],exact) : EmptySource<_,scale>
      empty_source_three(channels,height,width,scale,{%exact : {Shapes.has_zero(_) == True{} : Bool} empty : {Shapes.has_zero([channels,height,width]) == True{} : Bool}})

def same_extents(+extents: List<&2,U32>) -> {Shapes.same_extents(extents,extents) == True{} : Bool}:
  match extents:
    case []: {==}
    case +head <> +tail:
      %Equal.sym(Bool,U32.is_eq(head,head),True{},Equal.trans(Bool,U32.is_eq(head,head),Nat.is_eq(U32.to_nat(head),U32.to_nat(head)),True{},Comparison.eq(head,head),Order.equal_reflexive(U32.to_nat(head)))) : {_ && Shapes.same_extents(tail,tail) == True{} : Bool}
      same_extents(tail)

def empty_view_fits(+view: Views.View,+length: U32,+empty: {Shapes.has_zero(Views.axis_extents_of(view)) == True{} : Bool},+offset_zero: {CopyValues.view_offset(view) == 0 : U32}) -> {Views.fits(view,length) == True{} : Bool}:
  Views.View{axes,+offset} = view
  %Equal.sym(Bool,Shapes.has_zero(Views.axis_extents(axes)),True{},empty) : {Views.fits_nonempty(_,view,length) == True{} : Bool}
  %Equal.sym(U32,offset,0,offset_zero) : {U32.is_le(_,length) == True{} : Bool}
  BandUpsampleSamples.native_le(0,length,Order.zero_le(U32.to_nat(length)))

def view_shapes_projection(left: Views.View,right: Views.View) ->
  {Shapes.same(Views.shape(left),Views.shape(right)) == Shapes.same_extents(Views.axis_extents_of(left),Views.axis_extents_of(right)) : Bool}:
  match left right:
    case Views.View{left_axes,left_offset} Views.View{right_axes,right_offset}: {==}

def empty_copy_guard(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+extents: List<&2,U32>,
  +input_zero: {Buffer.content_length(~Element,Buffer.contents(~Element,input)) == 0n : Nat},+output_zero: {Buffer.content_length(~Element,Buffer.contents(~Element,output)) == 0n : Nat},
  +source_shape: {Views.axis_extents_of(source) == extents : List<&2,U32>},+source_offset: {CopyValues.view_offset(source) == 0 : U32},+empty: {Shapes.has_zero(extents) == True{} : Bool}) ->
  {CopyValues.copy_guard(~Element,input,source,output,BandUpsampleExecution.canonical_view(extents,0)) == True{} : Bool}:
  Storage.Buffer{input_storage,+input_length,Certificate.Certificate{+input_depth,input_equation}} = input
  Storage.Buffer{output_storage,+output_length,Certificate.Certificate{+output_depth,output_equation}} = output
  +exact = ShapeLaws.empty_product(extents,empty)
  +target_shape = BandUpsampleExecution.canonical_shape(extents,0,exact)
  +same = {%Equal.sym(Bool,Shapes.same(Views.shape(source),Views.shape(BandUpsampleExecution.canonical_view(extents,0))),
      Shapes.same_extents(Views.axis_extents_of(source),Views.axis_extents_of(BandUpsampleExecution.canonical_view(extents,0))),view_shapes_projection(source,BandUpsampleExecution.canonical_view(extents,0))) : {_ == True{} : Bool}
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(source),extents,source_shape) : {Shapes.same_extents(_,Views.axis_extents_of(BandUpsampleExecution.canonical_view(extents,0))) == True{} : Bool}
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(BandUpsampleExecution.canonical_view(extents,0)),extents,target_shape) : {Shapes.same_extents(extents,_) == True{} : Bool}
    same_extents(extents) : {Shapes.same(Views.shape(source),Views.shape(BandUpsampleExecution.canonical_view(extents,0))) == True{} : Bool}}
  +source_fits = empty_view_fits(source,0,{%Equal.sym(List<&2,U32>,Views.axis_extents_of(source),extents,source_shape) : {Shapes.has_zero(_) == True{} : Bool} empty : {Shapes.has_zero(Views.axis_extents_of(source)) == True{} : Bool}},source_offset)
  %Equal.sym(Nat,input_length,0n,input_zero) : {Shapes.same(Views.shape(source),Views.shape(BandUpsampleExecution.canonical_view(extents,0))) && Views.is_contiguous(BandUpsampleExecution.canonical_view(extents,0)) && Operations.valid_buffer(_,U32.shln(1,input_depth),source) && Operations.valid_buffer(output_length,U32.shln(1,output_depth),BandUpsampleExecution.canonical_view(extents,0)) == True{} : Bool}
  %Equal.sym(Nat,output_length,0n,output_zero) : {Shapes.same(Views.shape(source),Views.shape(BandUpsampleExecution.canonical_view(extents,0))) && Views.is_contiguous(BandUpsampleExecution.canonical_view(extents,0)) && Operations.valid_buffer(0n,U32.shln(1,input_depth),source) && Operations.valid_buffer(_,U32.shln(1,output_depth),BandUpsampleExecution.canonical_view(extents,0)) == True{} : Bool}
  and_true(Shapes.same(Views.shape(source),Views.shape(BandUpsampleExecution.canonical_view(extents,0))) && Views.is_contiguous(BandUpsampleExecution.canonical_view(extents,0)) && Operations.valid_buffer(0n,U32.shln(1,input_depth),source),Operations.valid_buffer(0n,U32.shln(1,output_depth),BandUpsampleExecution.canonical_view(extents,0)),
    and_true(Shapes.same(Views.shape(source),Views.shape(BandUpsampleExecution.canonical_view(extents,0))) && Views.is_contiguous(BandUpsampleExecution.canonical_view(extents,0)),Operations.valid_buffer(0n,U32.shln(1,input_depth),source),
      and_true(Shapes.same(Views.shape(source),Views.shape(BandUpsampleExecution.canonical_view(extents,0))),Views.is_contiguous(BandUpsampleExecution.canonical_view(extents,0)),same,BandUpsampleSamples.canonical_contiguous(extents,0,exact)),
      and_true(Nat.is_le(0n,U32.to_nat(U32.shln(1,input_depth))),Views.fits(source,0),Order.zero_le(U32.to_nat(U32.shln(1,input_depth))),source_fits)),
    and_true(Nat.is_le(0n,U32.to_nat(U32.shln(1,output_depth))),Views.fits(BandUpsampleExecution.canonical_view(extents,0),0),Order.zero_le(U32.to_nat(U32.shln(1,output_depth))),BandUpsampleSamples.canonical_fits(extents,0,exact)))

def accepted(~Value: Type,result: Operations.Outcome<Value>) -> Bool:
  match result:
    case Operations.Accepted{value}: True{}
    case Operations.Rejected{value}: False{}

def observed_accepted(~Element: Data,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {accepted(~(Storage.Buffer<Element> & Storage.Buffer<Element>),result) == accepted(~OperationsLaws.PairOwners(Element),OperationsLaws.observe_pair_outcome(~Element,result)) : Bool}:
  match result:
    case Operations.Accepted{value}: {==}
    case Operations.Rejected{value}: {==}

def copy_denotation_accepted(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View,
  +guarded: {CopyValues.copy_guard(~Element,input,source,output,target) == True{} : Bool},+counted: {Views.numel(target) == Some{0} : Maybe<&2,U32>}) ->
  {accepted(~OperationsLaws.PairOwners(Element),OperationsLaws.copy_denotation(~Element,input,source,output,target)) == True{} : Bool}:
  Storage.Buffer{input_storage,input_length,input_certificate} = input
  Views.View{source_axes,source_offset} = source
  Storage.Buffer{output_storage,output_length,output_certificate} = output
  Views.View{target_axes,target_offset} = target
  %Equal.sym(Bool,CopyValues.copy_guard(~Element,input,source,output,target),True{},guarded) :
    {accepted(~OperationsLaws.PairOwners(Element),OperationsLaws.copy_model(~Element,_,Views.numel(target),source,target,Tree.reflect(Element,input_storage),input_length,Tree.reflect(Element,output_storage),output_length)) == True{} : Bool}
  %Equal.sym(Maybe<&2,U32>,Views.numel(target),Some{0},counted) :
    {accepted(~OperationsLaws.PairOwners(Element),OperationsLaws.copy_model(~Element,True{},_,source,target,Tree.reflect(Element,input_storage),input_length,Tree.reflect(Element,output_storage),output_length)) == True{} : Bool}
  {==}

def copy_accepted(~Element: Data,+input: Buffer.Model<Element>,+source: Views.View,+fill: Element,+target: Views.View,
  +guarded: {CopyValues.copy_guard(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target) == True{} : Bool},+counted: {Views.numel(target) == Some{0} : Maybe<&2,U32>}) ->
  {accepted(~(Storage.Buffer<Element> & Storage.Buffer<Element>),Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target)) == True{} : Bool}:
  Equal.trans(Bool,accepted(~(Storage.Buffer<Element> & Storage.Buffer<Element>),Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target)),
    accepted(~OperationsLaws.PairOwners(Element),OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target))),True{},observed_accepted(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target)),
    Equal.trans(Bool,accepted(~OperationsLaws.PairOwners(Element),OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target))),
      accepted(~OperationsLaws.PairOwners(Element),OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target)),True{},
      Equal.cong(Operations.Outcome<OperationsLaws.PairOwners(Element)>,Bool,accepted(~OperationsLaws.PairOwners(Element)),OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target)),OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target),OperationsLaws.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target)),
      copy_denotation_accepted(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target,guarded,counted)))

def program_length(~Element: Data,+source: Views.View,+target: Views.View,input: Buffer.Content<Element>,output: Buffer.Content<Element>) ->
  {Buffer.content_length(~Element,CopyValues.program_content(~Element,0,source,target,input,output)) == Buffer.content_length(~Element,output) : Nat}:
  Buffer.Content{input_tree,input_length} = input
  Buffer.Content{output_tree,output_length} = output
  {==}

def copy_zero_length(~Element: Data,+input: Buffer.Model<Element>,+source: Views.View,+fill: Element,+target: Views.View,
  +guarded: {CopyValues.copy_guard(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target) == True{} : Bool},+counted: {Views.numel(target) == Some{0} : Maybe<&2,U32>},+output_zero: {Buffer.content_length(~Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))) == 0n : Nat}) ->
  {Buffer.content_length(~Element,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target)))) == 0n : Nat}:
  Equal.trans(Nat,Buffer.content_length(~Element,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target)))),
    Buffer.content_length(~Element,CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target))),0n,
    Equal.cong(Operations.Outcome<OperationsLaws.PairOwners(Element)>,Nat,result => Buffer.content_length(~Element,CopyValues.right_content(~Element,result)),OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target)),OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target),OperationsLaws.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target)),
    Equal.trans(Nat,Buffer.content_length(~Element,CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target))),
      Buffer.content_length(~Element,CopyValues.program_content(~Element,0,source,target,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)))),0n,
      Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target)),CopyValues.program_content(~Element,0,source,target,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))),CopyValues.copy_observed_content(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),target,0,guarded,counted)),
      Equal.trans(Nat,Buffer.content_length(~Element,CopyValues.program_content(~Element,0,source,target,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)))),Buffer.content_length(~Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))),0n,program_length(~Element,source,target,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))),output_zero)))

def allocation_zero_length(~Element: Data,+fill: Element) -> {Buffer.content_length(~Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))) == 0n : Nat}:
  Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n},AllocationLaws.contents(~Element,Tensors.depth_for(0),fill,0n))

def allocation_zero_ready(~Element: Data,+fill: Element) -> Buffer.ContentReady<Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))>:
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n},AllocationLaws.contents(~Element,Tensors.depth_for(0),fill,0n)) : Buffer.ContentReady<Element,_>
  Buffer.content_ready(~Element,BandCount.model(~Element,0,fill),BandCount.bounded_model(~Element,0,fill,{==}))

def copied_empty_result(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+successful: {accepted(~(Storage.Buffer<Element> & Storage.Buffer<Element>),result) == True{} : Bool},
  +ready: Buffer.ContentReady<Element,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,result))>,
  +length_zero: {Buffer.content_length(~Element,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,result))) == 0n : Nat},+shape_zero: {ShapeLaws.elements(target_shape) == 0n : Nat}) ->
  UpsampleResult(~Element,~Context,~original,context,target_shape,Layout.result_of(~Element,Layout.copied(~Element,"upsample_nearest",source_shape,target_shape,fill,workers,result))):
  match result:
    case Operations.Accepted{(input,Storage.Buffer{storage,+length,certificate})}:
      Shape.Both{Shape.Both{ready,Shape.Both{Global.Counted{0,CountCompletion.count_of(target_shape,0,shape_zero,BandUpsampleSamples.native_le(0,Machine.capacity(Default.limits()),Order.zero_le(U32.to_nat(Machine.capacity(Default.limits())))))},
        Equal.trans(Nat,length,0n,ShapeLaws.elements(target_shape),length_zero,Equal.sym(Nat,ShapeLaws.elements(target_shape),0n,shape_zero))}},
        Shape.Both{{==},Shape.Both{{==},
          %Equal.sym(Nat,ShapeLaws.elements(target_shape),0n,shape_zero) : Members.All(Nat,{index => {Tree.read(Element,Tree.reflect(Element,storage),U32.from_nat(index)) == original(context,U32.from_nat(index)) : Element} : Nat -> Data},Ranges.range(_,0n)) Unit{}}}}
    case Operations.Rejected{value}:
      Empty.absurd(UpsampleResult(~Element,~Context,~original,context,target_shape,Layout.result_of(~Element,Layout.copied(~Element,"upsample_nearest",source_shape,target_shape,fill,workers,Operations.Rejected{value}))),WordEncoding.false_true(successful))

def gather_model_length(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,output: Buffer.Content<Element>) ->
  {Buffer.content_length(~Element,BandGather.model(~Element,geometry,bands,output)) == Buffer.content_length(~Element,output) : Nat}:
  Buffer.Content{+tree,+length} = output
  Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),BandGather.model(~Element,geometry,bands,Buffer.Content{tree,length}),Buffer.Content{Writes.pure(Element,BandGather.refinement_writes(~Element,geometry,bands),tree),length},BandGather.model_writes(~Element,geometry,bands,tree,length))

type GatherLengthCall<-Element: Data> is Data:
  GatherLengthCall{geometry: Bands.Geometry,bands: Model.Model<Element>}

def gather_length_proposition(~Element: Data,call: GatherLengthCall<Element>,-output: Storage.Buffer<Element>) -> Type:
  GatherLengthCall{geometry,bands} = call
  {Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),output))) == Buffer.content_length(~Element,Buffer.contents(~Element,output)) : Nat}

def gather_canonical_length(~Element: Data,call: GatherLengthCall<Element>,+output: Buffer.Model<Element>) -> gather_length_proposition(~Element,call,Buffer.pack(~Element,output)):
  GatherLengthCall{+geometry,+bands} = call
  Equal.trans(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,output)))),Buffer.content_length(~Element,BandGather.model(~Element,geometry,bands,Buffer.contents(~Element,Buffer.pack(~Element,output)))),Buffer.content_length(~Element,Buffer.contents(~Element,Buffer.pack(~Element,output))),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,output))),BandGather.model(~Element,geometry,bands,Buffer.contents(~Element,Buffer.pack(~Element,output))),BandGather.gather(~Element,geometry,bands,Buffer.pack(~Element,output))),gather_model_length(~Element,geometry,bands,Buffer.contents(~Element,Buffer.pack(~Element,output))))

def gather_length(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,output: Storage.Buffer<Element>) ->
  {Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),output))) == Buffer.content_length(~Element,Buffer.contents(~Element,output)) : Nat}:
  +call = {GatherLengthCall{geometry,bands} : GatherLengthCall<Element>}
  Buffer.lift(~Element,~GatherLengthCall<Element>,~gather_length_proposition(~Element),call,output,Buffer.observe(~Element,output),model => gather_canonical_length(~Element,call,model))

def gathered_zero_length(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+fill: Element,+zero: {Bands.elements(geometry) == 0 : U32}) ->
  {Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))))) == 0n : Nat}:
  %Equal.sym(U32,Bands.elements(geometry),0,zero) : {Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,_)))) == 0n : Nat}
  Equal.trans(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,0)))),Buffer.content_length(~Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))),0n,gather_length(~Element,geometry,bands,Tensors.buffer(~Element,fill,0)),allocation_zero_length(~Element,fill))

def empty_dense_model_result(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+input: Buffer.Model<Element>,
  source: EmptySource<shape,scale>,+input_zero: {Buffer.content_length(~Element,Buffer.contents(~Element,Buffer.pack(~Element,input))) == 0n : Nat},+scaled_empty: {Shapes.has_zero(Layout.scaled_extents(shape,scale)) == True{} : Bool}) ->
  UpsampleResult(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.result_of(~Element,Layout.upsample_view(~Element,Layout.repeated(Tensors.view_of(shape),scale),Tensors.Usable{},Buffer.pack(~Element,input),shape,scale,fill,workers))):
  EmptySource{+source_view,+actual,+extents,+offset,+empty} = source
  +view_shape = Views.axis_extents_of(source_view)
  +target = BandUpsampleExecution.canonical_view(view_shape,0)
  +exact = ShapeLaws.empty_product(view_shape,empty)
  +bounded = BandUpsampleSamples.native_le(0,Machine.capacity(Default.limits()),Order.zero_le(U32.to_nat(Machine.capacity(Default.limits()))))
  +counted = BandUpsampleExecution.target_counted_from_count(view_shape,0,Count.Count{exact,bounded})
  +guarded = empty_copy_guard(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),view_shape,input_zero,allocation_zero_length(~Element,fill),{==},offset,empty)
  +observed = Equal.cong(Operations.Outcome<OperationsLaws.PairOwners(Element)>,Buffer.Content<Element>,CopyValues.right_content(~Element),
    OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target)),OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target),OperationsLaws.copy(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target))
  %Equal.sym(Maybe<&2,Views.View>,Layout.repeated(Tensors.view_of(shape),scale),Some{source_view},actual) :
    UpsampleResult(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.result_of(~Element,Layout.upsample_view(~Element,_,Tensors.Usable{},Buffer.pack(~Element,input),shape,scale,fill,workers)))
  %Equal.sym(Tensors.Tensor<Element> & Tensors.Tensor<Element>,Layout.upsample_view(~Element,Some{source_view},Tensors.Usable{},Buffer.pack(~Element,input),shape,scale,fill,workers),
    Layout.copied(~Element,"upsample_nearest",shape,Layout.scaled_extents(shape,scale),fill,workers,Operations.copy(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target)),
    Public.gathered(~Element,"upsample_nearest",source_view,view_shape,Layout.scaled_extents(shape,scale),Buffer.pack(~Element,input),shape,fill,target,0,workers,BandUpsampleExecution.view(view_shape,0,exact),CountCompletion.count_of(view_shape,0,exact,bounded))) :
    UpsampleResult(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.result_of(~Element,_))
  copied_empty_result(~Element,~Context,~original,context,shape,Layout.scaled_extents(shape,scale),fill,workers,Operations.copy(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target),copy_accepted(~Element,input,source_view,fill,target,guarded,counted),
    %Equal.sym(Buffer.Content<Element>,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target))),CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target)),observed) : Buffer.ContentReady<Element,_>
    CopyValues.copy_observed_ready(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target,allocation_zero_ready(~Element,fill)),
    copy_zero_length(~Element,input,source_view,fill,target,guarded,counted,allocation_zero_length(~Element,fill)),ShapeLaws.empty_product(Layout.scaled_extents(shape,scale),scaled_empty))

type EmptyCall<-Element: Data,-Context: Data> is Data:
  EmptyCall{context: Context,shape: List<&2,U32>,scale: U32,fill: Element,workers: U32,
    source: EmptySource<shape,scale>,scaled_empty: {Shapes.has_zero(Layout.scaled_extents(shape,scale)) == True{} : Bool}}

def empty_result_proposition(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,call: EmptyCall<Element,Context>,-input: Storage.Buffer<Element>) -> Type:
  EmptyCall{context,shape,scale,fill,workers,source,scaled_empty} = call
  @+input_zero: {Buffer.content_length(~Element,Buffer.contents(~Element,input)) == 0n : Nat} ->
    UpsampleResult(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.result_of(~Element,Layout.upsample_view(~Element,Layout.repeated(Tensors.view_of(shape),scale),Tensors.Usable{},input,shape,scale,fill,workers)))

def empty_dense_canonical(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,call: EmptyCall<Element,Context>,+input: Buffer.Model<Element>) ->
  empty_result_proposition(~Element,~Context,~original,call,Buffer.pack(~Element,input)):
  EmptyCall{context,shape,scale,fill,workers,source,scaled_empty} = call
  input_zero => empty_dense_model_result(~Element,~Context,~original,context,shape,scale,fill,workers,input,source,input_zero,scaled_empty)

def empty_dense_result(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,input: Storage.Buffer<Element>,
  source: EmptySource<shape,scale>,+input_zero: {Buffer.content_length(~Element,Buffer.contents(~Element,input)) == 0n : Nat},+scaled_empty: {Shapes.has_zero(Layout.scaled_extents(shape,scale)) == True{} : Bool}) ->
  UpsampleResult(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.result_of(~Element,Layout.upsample_view(~Element,Layout.repeated(Tensors.view_of(shape),scale),Tensors.Usable{},input,shape,scale,fill,workers))):
  +call = {EmptyCall{context,shape,scale,fill,workers,source,scaled_empty} : EmptyCall<Element,Context>}
  Buffer.lift(~Element,~EmptyCall<Element,Context>,~empty_result_proposition(~Element,~Context,~original),call,input,Buffer.observe(~Element,input),
    model => empty_dense_canonical(~Element,~Context,~original,call,model))(input_zero)

def and_false_right(+left: Bool) -> {left && False{} == False{} : Bool}:
  match left:
    case True{}: {==}
    case False{}: {==}

def empty_selector(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+empty: {Shapes.has_zero(Layout.scaled_extents(shape,scale)) == True{} : Bool}) -> {BandUpsampleExecution.enabled(geometry,shape,scale) == False{} : Bool}:
  %Equal.sym(Bool,Shapes.has_zero(Layout.scaled_extents(shape,scale)),True{},empty) : {BandUpsampleExecution.guard(geometry,shape,scale) && Bool.not(_) == False{} : Bool}
  and_false_right(BandUpsampleExecution.guard(geometry,shape,scale))

def empty_public_equation(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,+empty: {Shapes.has_zero(Layout.scaled_extents(shape,scale)) == True{} : Bool}) ->
  {Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale) ==
    Layout.result_of(~Element,Layout.upsample_view(~Element,Layout.repeated(Tensors.view_of(shape),scale),Tensors.Usable{},Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))),shape,scale,fill,workers)) : Tensors.Tensor<Element>}:
  Bands.Geometry{images,channels,height,width} = geometry
  %Equal.sym(Bool,BandUpsampleExecution.enabled(geometry,shape,scale),False{},empty_selector(geometry,shape,scale,empty)) :
    {Layout.upsample_supported(~Element,_,Model.pack(~Element,bands),geometry,shape,scale,fill,workers) ==
      Layout.result_of(~Element,Layout.upsample_view(~Element,Layout.repeated(Tensors.view_of(shape),scale),Tensors.Usable{},Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))),shape,scale,fill,workers)) : Tensors.Tensor<Element>}
  {==}

def empty_public(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  +coherence: ShapeCoherence<geometry,shape>,+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool},+empty: {Shapes.has_zero(Layout.scaled_extents(shape,scale)) == True{} : Bool}) ->
  UpsampleResult(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),
    Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale)):
  +source_zero = source_empty(geometry,shape,scale,coherence,truth,empty)
  %Equal.sym(Tensors.Tensor<Element>,Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale),
    Layout.result_of(~Element,Layout.upsample_view(~Element,Layout.repeated(Tensors.view_of(shape),scale),Tensors.Usable{},Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))),shape,scale,fill,workers)),empty_public_equation(~Element,geometry,shape,scale,fill,workers,bands,empty)) :
    UpsampleResult(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),_)
  empty_dense_result(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},shape,scale,fill,workers,
    Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))),empty_source(geometry,shape,scale,coherence,source_zero),
    gathered_zero_length(~Element,geometry,bands,fill,geometry_zero(geometry,shape,coherence,source_zero)),empty)

def public_when(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,empty: Bool,
  +classified: {Shapes.has_zero(Layout.scaled_extents(shape,scale)) == empty : Bool},+well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,+values: BandValues.Values(~Element,~Context,~original,context,geometry,Model.content(~Element,bands)),
  +coherence: ShapeCoherence<geometry,shape>,+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) ->
  UpsampleResult(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),
    Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale)):
  match empty:
    case True{}: empty_public(~Element,~Context,~original,context,geometry,shape,scale,fill,workers,bands,coherence,truth,classified)
    case False{}:
      enabled_result(~Element,~Context,~original,context,geometry,shape,scale,fill,workers,bands,well_formed,values,coherence,
        and_true(BandUpsampleExecution.guard(geometry,shape,scale),Bool.not(Shapes.has_zero(Layout.scaled_extents(shape,scale))),truth,
          %Equal.sym(Bool,Shapes.has_zero(Layout.scaled_extents(shape,scale)),False{},classified) : {Bool.not(_) == True{} : Bool} {==}))

# All legal ready band calls, including every zero axis, reach this contract.
# The premises describe the input and actual guard; nearest equations are built.
def public(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,+values: BandValues.Values(~Element,~Context,~original,context,geometry,Model.content(~Element,bands)),
  +coherence: ShapeCoherence<geometry,shape>,+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) ->
  UpsampleResult(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),
    Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale)):
  public_when(~Element,~Context,~original,context,geometry,shape,scale,fill,workers,bands,Shapes.has_zero(Layout.scaled_extents(shape,scale)),{==},well_formed,values,coherence,truth)

def public_ready(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,+values: BandValues.Values(~Element,~Context,~original,context,geometry,Model.content(~Element,bands)),
  +coherent: {Window.band_shape(shape,geometry) == True{} : Bool},+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) ->
  UpsampleResult(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),
    Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale)):
  public(~Element,~Context,~original,context,geometry,shape,scale,fill,workers,bands,well_formed,values,coherence_from_decoded(shape,geometry,BandCount.decoded(shape,geometry,coherent)),truth)
