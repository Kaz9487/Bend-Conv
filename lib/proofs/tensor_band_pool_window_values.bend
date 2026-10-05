# Pool row coordinates: the padded row predicate of one axis, the origin of a
# window inside its image and the coverage of the rows a window stores.
import Base
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_values.bend as BandValues
import ./tensor_band_window.bend as TensorBandWindow
import ./tensor_band_pool_execution.bend as BandPoolExecution
import ./boolean_logic.bend as Boolean
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding
import ./word_subtraction.bend as Subtraction
import ../tensor_access.bend as Access
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ../tensor_operations.bend as Operations
import ../machine_limits.bend as Machine
import ./encoded_coordinates.bend as Encoded
import ./machine_limit_bounds.bend as Limits
import ./nat_division.bend as Division
import ./nat_extrema.bend as Extrema
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./tensor_band_halo.bend as BandHalo
import ./tensor_band_shape_refinement.bend as BandShape
import ./u32_division.bend as U32Division
import ./u32_extrema.bend as U32Extrema
import ../tensor_shape.bend as Shapes

# Producers for the native padded sample predicates. Unsigned subtraction is
# decoded only after its real bound check accepts, so an underflow cannot be
# silently interpreted as clipped Nat subtraction. All facts are about scalar
# coordinates and addresses; no completed pool or sample value is assumed.
def natural_valid(+coordinate: Nat,+padding: Nat,extent: Nat) -> Bool:
  Nat.is_le(padding,coordinate) && Nat.is_lt(Nat.sub(coordinate,padding),extent)

type AxisFacts<-coordinate: U32,-padding: U32,-extent: U32> is Data:
  AxisFacts{ordered: {Nat.is_le(U32.to_nat(padding),U32.to_nat(coordinate)) == True{} : Bool},
    inside: {Nat.is_lt(Nat.sub(U32.to_nat(coordinate),U32.to_nat(padding)),U32.to_nat(extent)) == True{} : Bool},
    exact: {U32.to_nat((coordinate - padding : U32)) == Nat.sub(U32.to_nat(coordinate),U32.to_nat(padding)) : Nat}}

def axis_ordered(+coordinate: U32,+padding: U32,+extent: U32,evidence: AxisFacts<coordinate,padding,extent>) ->
  {Nat.is_le(U32.to_nat(padding),U32.to_nat(coordinate)) == True{} : Bool}:
  AxisFacts{ordered,inside,exact} = evidence
  ordered

def axis_inside(+coordinate: U32,+padding: U32,+extent: U32,evidence: AxisFacts<coordinate,padding,extent>) ->
  {Nat.is_lt(Nat.sub(U32.to_nat(coordinate),U32.to_nat(padding)),U32.to_nat(extent)) == True{} : Bool}:
  AxisFacts{ordered,inside,exact} = evidence
  inside

def axis_exact(+coordinate: U32,+padding: U32,+extent: U32,evidence: AxisFacts<coordinate,padding,extent>) ->
  {U32.to_nat((coordinate - padding : U32)) == Nat.sub(U32.to_nat(coordinate),U32.to_nat(padding)) : Nat}:
  AxisFacts{ordered,inside,exact} = evidence
  exact

# A successful unsigned bound check itself excludes subtraction underflow.
# If the subtracted word were small, adding padding back would be an exact
# bounded sum, whose result is the original coordinate.
def axis_accept(+coordinate: U32,+padding: U32,+extent: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +coordinate_room: {Nat.is_le(U32.to_nat(coordinate),U32.to_nat(capacity)) == True{} : Bool},
  extent_padding_room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  accepted: {U32.is_lt((coordinate - padding : U32),extent) == True{} : Bool}) -> AxisFacts<coordinate,padding,extent>:
  +difference = (coordinate - padding : U32)
  +native_inside = Bounds.as_nat_lt(difference,extent,accepted)
  +sum_room = Order.transitive(Nat.add(U32.to_nat(difference),U32.to_nat(padding)),Nat.add(U32.to_nat(extent),U32.to_nat(padding)),U32.to_nat(capacity),
    Order.add_le(U32.to_nat(difference),U32.to_nat(extent),U32.to_nat(padding),Bounds.strict_le(U32.to_nat(difference),U32.to_nat(extent),native_inside)),extent_padding_room)
  +restored = Equal.trans(Nat,U32.to_nat(coordinate),U32.to_nat((difference + padding : U32)),Nat.add(U32.to_nat(difference),U32.to_nat(padding)),
    Equal.cong(U32,Nat,value => U32.to_nat(value),coordinate,(difference + padding : U32),Equal.sym(U32,(difference + padding : U32),coordinate,Subtraction.add_subtract(coordinate,padding))),
    Arithmetic.add(difference,padding,capacity,capacity_ok,sum_room))
  +ordered = {
    %Equal.sym(Nat,U32.to_nat(coordinate),Nat.add(U32.to_nat(difference),U32.to_nat(padding)),restored) : {Nat.is_le(U32.to_nat(padding),_) == True{} : Bool}
    %Order.commutative(U32.to_nat(padding),U32.to_nat(difference)) : {Nat.is_le(U32.to_nat(padding),_) == True{} : Bool}
    Order.le_add(U32.to_nat(padding),U32.to_nat(difference)) : {Nat.is_le(U32.to_nat(padding),U32.to_nat(coordinate)) == True{} : Bool}}
  +exact = Arithmetic.subtract(coordinate,padding,capacity,capacity_ok,ordered,
    Order.transitive(Nat.sub(U32.to_nat(coordinate),U32.to_nat(padding)),U32.to_nat(coordinate),U32.to_nat(capacity),Order.sub_le(U32.to_nat(coordinate),U32.to_nat(padding)),coordinate_room))
  AxisFacts{ordered,
    %exact : {Nat.is_lt(_,U32.to_nat(extent)) == True{} : Bool} native_inside,
    exact}

def axis_complete(+coordinate: U32,+padding: U32,+extent: U32,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},coordinate_room: {Nat.is_le(U32.to_nat(coordinate),U32.to_nat(capacity)) == True{} : Bool},
  ordered: {Nat.is_le(U32.to_nat(padding),U32.to_nat(coordinate)) == True{} : Bool},
  inside: {Nat.is_lt(Nat.sub(U32.to_nat(coordinate),U32.to_nat(padding)),U32.to_nat(extent)) == True{} : Bool}) ->
  {U32.is_lt((coordinate - padding : U32),extent) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt((coordinate - padding : U32),extent),Nat.is_lt(U32.to_nat((coordinate - padding : U32)),U32.to_nat(extent)),Comparison.lt((coordinate - padding : U32),extent)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((coordinate - padding : U32)),Nat.sub(U32.to_nat(coordinate),U32.to_nat(padding)),
    Arithmetic.subtract(coordinate,padding,capacity,capacity_ok,ordered,
      Order.transitive(Nat.sub(U32.to_nat(coordinate),U32.to_nat(padding)),U32.to_nat(coordinate),U32.to_nat(capacity),Order.sub_le(U32.to_nat(coordinate),U32.to_nat(padding)),coordinate_room))) :
    {Nat.is_lt(_,U32.to_nat(extent)) == True{} : Bool}
  inside

def natural_accepted(+coordinate: Nat,+padding: Nat,+extent: Nat,ordered: {Nat.is_le(padding,coordinate) == True{} : Bool},inside: {Nat.is_lt(Nat.sub(coordinate,padding),extent) == True{} : Bool}) ->
  {natural_valid(coordinate,padding,extent) == True{} : Bool}:
  %Equal.sym(Bool,Nat.is_le(padding,coordinate),True{},ordered) : {_ && Nat.is_lt(Nat.sub(coordinate,padding),extent) == True{} : Bool}
  inside

def natural_rejected(+coordinate: U32,+padding: U32,+extent: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+coordinate_room: {Nat.is_le(U32.to_nat(coordinate),U32.to_nat(capacity)) == True{} : Bool},
  +rejected: {U32.is_lt((coordinate - padding : U32),extent) == False{} : Bool},choice: Bool,
  +natural_choice: {natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(extent)) == choice : Bool}) ->
  {natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(extent)) == False{} : Bool}:
  match choice:
    case False{}: natural_choice
    case True{}:
      +ordered = Boolean.left(Nat.is_le(U32.to_nat(padding),U32.to_nat(coordinate)),Nat.is_lt(Nat.sub(U32.to_nat(coordinate),U32.to_nat(padding)),U32.to_nat(extent)),natural_choice)
      +inside = Boolean.right(Nat.is_le(U32.to_nat(padding),U32.to_nat(coordinate)),Nat.is_lt(Nat.sub(U32.to_nat(coordinate),U32.to_nat(padding)),U32.to_nat(extent)),natural_choice)
      Empty.absurd({natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(extent)) == False{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_lt((coordinate - padding : U32),extent),True{},Equal.sym(Bool,U32.is_lt((coordinate - padding : U32),extent),False{},rejected),
          axis_complete(coordinate,padding,extent,capacity,capacity_ok,coordinate_room,ordered,inside))))

def axis_decoded(+coordinate: U32,+padding: U32,+extent: U32,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},coordinate_room: {Nat.is_le(U32.to_nat(coordinate),U32.to_nat(capacity)) == True{} : Bool},
  extent_padding_room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},choice: Bool,
  +compared: {U32.is_lt((coordinate - padding : U32),extent) == choice : Bool}) ->
  {U32.is_lt((coordinate - padding : U32),extent) == natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(extent)) : Bool}:
  match choice:
    case True{}:
      +evidence = axis_accept(coordinate,padding,extent,capacity,capacity_ok,coordinate_room,extent_padding_room,compared)
      Equal.trans(Bool,U32.is_lt((coordinate - padding : U32),extent),True{},natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(extent)),compared,
        Equal.sym(Bool,natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(extent)),True{},natural_accepted(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(extent),
          axis_ordered(coordinate,padding,extent,evidence),axis_inside(coordinate,padding,extent,evidence))))
    case False{}:
      Equal.trans(Bool,U32.is_lt((coordinate - padding : U32),extent),False{},natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(extent)),compared,
        Equal.sym(Bool,natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(extent)),False{},
          natural_rejected(coordinate,padding,extent,capacity,capacity_ok,coordinate_room,compared,natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(extent)),{==})))

def axis_valid(+coordinate: U32,+padding: U32,+extent: U32,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},coordinate_room: {Nat.is_le(U32.to_nat(coordinate),U32.to_nat(capacity)) == True{} : Bool},
  extent_padding_room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.is_lt((coordinate - padding : U32),extent) == natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(extent)) : Bool}:
  axis_decoded(coordinate,padding,extent,capacity,capacity_ok,coordinate_room,extent_padding_room,U32.is_lt((coordinate - padding : U32),extent),{==})

# Adding the window's image-relative start translates a valid padded row.
def shifted_subtraction(+offset: Nat,+coordinate: Nat,+padding: Nat,ordered: {Nat.is_le(padding,coordinate) == True{} : Bool}) ->
  {Nat.sub(Nat.add(offset,coordinate),padding) == Nat.add(offset,Nat.sub(coordinate,padding)) : Nat}:
  match coordinate padding:
    case +coordinate 0n: Equal.trans(Nat,Nat.sub(Nat.add(offset,coordinate),0n),Nat.add(offset,coordinate),Nat.add(offset,Nat.sub(coordinate,0n)),Order.subtract_zero(Nat.add(offset,coordinate)),
      Equal.cong(Nat,Nat,value => Nat.add(offset,value),coordinate,Nat.sub(coordinate,0n),Equal.sym(Nat,Nat.sub(coordinate,0n),coordinate,Order.subtract_zero(coordinate))))
    case 0n 1n+padding: Empty.absurd({Nat.sub(Nat.add(offset,0n),1n+padding) == Nat.add(offset,0n) : Nat},WordEncoding.false_true(ordered))
    case 1n+ +coordinate 1n+ +padding:
      %Equal.sym(Nat,Nat.add(offset,1n+coordinate),1n+Nat.add(offset,coordinate),Addition.successor_right(offset,coordinate)) :
        {Nat.sub(_,1n+padding) == Nat.add(offset,Nat.sub(coordinate,padding)) : Nat}
      shifted_subtraction(offset,coordinate,padding,ordered)

def local_ordered(+offset: Nat,+coordinate: Nat,+padding: Nat,+global: Nat,
  +global_exact: {global == Nat.add(offset,coordinate) : Nat},global_ordered: {Nat.is_le(padding,global) == True{} : Bool},
  covered: {Nat.is_le(offset,Nat.sub(global,padding)) == True{} : Bool}) -> {Nat.is_le(padding,coordinate) == True{} : Bool}:
  Order.cancel_offset_le(offset,padding,coordinate,
    %Equal.sym(Nat,Nat.add(offset,padding),Nat.add(padding,offset),Order.commutative(offset,padding)) : {Nat.is_le(_,Nat.add(offset,coordinate)) == True{} : Bool}
    %global_exact : {Nat.is_le(Nat.add(padding,offset),_) == True{} : Bool}
    %Order.restore_shift(padding,global,global_ordered) : {Nat.is_le(Nat.add(padding,offset),_) == True{} : Bool}
    Order.offset_le(padding,offset,Nat.sub(global,padding),covered))

def global_ordered(+offset: Nat,+coordinate: Nat,+padding: Nat,+global: Nat,global_exact: {global == Nat.add(offset,coordinate) : Nat},
  ordered: {Nat.is_le(padding,coordinate) == True{} : Bool}) -> {Nat.is_le(padding,global) == True{} : Bool}:
  %Equal.sym(Nat,global,Nat.add(offset,coordinate),global_exact) : {Nat.is_le(padding,_) == True{} : Bool}
  Order.transitive(padding,coordinate,Nat.add(offset,coordinate),ordered,
    %Order.commutative(coordinate,offset) : {Nat.is_le(coordinate,_) == True{} : Bool}
    Order.le_add(coordinate,offset))

def shifted_difference(+offset: Nat,+coordinate: Nat,+padding: Nat,+global: Nat,global_exact: {global == Nat.add(offset,coordinate) : Nat},
  ordered: {Nat.is_le(padding,coordinate) == True{} : Bool}) -> {Nat.sub(global,padding) == Nat.add(offset,Nat.sub(coordinate,padding)) : Nat}:
  Equal.trans(Nat,Nat.sub(global,padding),Nat.sub(Nat.add(offset,coordinate),padding),Nat.add(offset,Nat.sub(coordinate,padding)),
    Equal.cong(Nat,Nat,value => Nat.sub(value,padding),global,Nat.add(offset,coordinate),global_exact),shifted_subtraction(offset,coordinate,padding,ordered))

def rows_forward(+offset: Nat,+coordinate: Nat,+padding: Nat,+global: Nat,+rows: Nat,+height: Nat,
  +global_exact: {global == Nat.add(offset,coordinate) : Nat},span: {Nat.is_le(Nat.add(offset,rows),height) == True{} : Bool},
  +accepted: {natural_valid(coordinate,padding,rows) == True{} : Bool}) -> {natural_valid(global,padding,height) == True{} : Bool}:
  +ordered = Boolean.left(Nat.is_le(padding,coordinate),Nat.is_lt(Nat.sub(coordinate,padding),rows),accepted)
  +inside = Boolean.right(Nat.is_le(padding,coordinate),Nat.is_lt(Nat.sub(coordinate,padding),rows),accepted)
  natural_accepted(global,padding,height,global_ordered(offset,coordinate,padding,global,global_exact,ordered),
    %Equal.sym(Nat,Nat.sub(global,padding),Nat.add(offset,Nat.sub(coordinate,padding)),shifted_difference(offset,coordinate,padding,global,global_exact,ordered)) : {Nat.is_lt(_,height) == True{} : Bool}
    Bounds.nat_trans(Nat.add(offset,Nat.sub(coordinate,padding)),Nat.add(offset,rows),height,Order.offset_lt(offset,Nat.sub(coordinate,padding),rows,inside),span))

def rows_backward(+offset: Nat,+coordinate: Nat,+padding: Nat,+global: Nat,+rows: Nat,+height: Nat,
  +global_exact: {global == Nat.add(offset,coordinate) : Nat},accepted: {natural_valid(global,padding,height) == True{} : Bool},
  coverage: TensorBandWindow.Contains(offset,Nat.add(offset,rows),Nat.sub(global,padding))) -> {natural_valid(coordinate,padding,rows) == True{} : Bool}:
  Shape.Both{lower,upper} = coverage
  +ordered = local_ordered(offset,coordinate,padding,global,global_exact,
    Boolean.left(Nat.is_le(padding,global),Nat.is_lt(Nat.sub(global,padding),height),accepted),lower)
  natural_accepted(coordinate,padding,rows,ordered,
    Order.cancel_offset_lt(offset,Nat.sub(coordinate,padding),rows,
      %shifted_difference(offset,coordinate,padding,global,global_exact,ordered) : {Nat.is_lt(_,Nat.add(offset,rows)) == True{} : Bool}
      upper))

def rows_same_selected(+offset: Nat,+coordinate: Nat,+padding: Nat,+global: Nat,+rows: Nat,+height: Nat,
  +global_exact: {global == Nat.add(offset,coordinate) : Nat},+span: {Nat.is_le(Nat.add(offset,rows),height) == True{} : Bool},
  local: Bool,original: Bool,local_choice: {natural_valid(coordinate,padding,rows) == local : Bool},+global_choice: {natural_valid(global,padding,height) == original : Bool},
  coverage: Boolean.When(original,TensorBandWindow.Contains(offset,Nat.add(offset,rows),Nat.sub(global,padding)))) ->
  {natural_valid(coordinate,padding,rows) == natural_valid(global,padding,height) : Bool}:
  match local original:
    case False{} False{}: Equal.trans(Bool,natural_valid(coordinate,padding,rows),False{},natural_valid(global,padding,height),local_choice,Equal.sym(Bool,natural_valid(global,padding,height),False{},global_choice))
    case True{} True{}: Equal.trans(Bool,natural_valid(coordinate,padding,rows),True{},natural_valid(global,padding,height),local_choice,Equal.sym(Bool,natural_valid(global,padding,height),True{},global_choice))
    case True{} False{}:
      Empty.absurd({natural_valid(coordinate,padding,rows) == natural_valid(global,padding,height) : Bool},WordEncoding.false_true(
        Equal.trans(Bool,False{},natural_valid(global,padding,height),True{},Equal.sym(Bool,natural_valid(global,padding,height),False{},global_choice),rows_forward(offset,coordinate,padding,global,rows,height,global_exact,span,local_choice))))
    case False{} True{}:
      Empty.absurd({natural_valid(coordinate,padding,rows) == natural_valid(global,padding,height) : Bool},WordEncoding.false_true(
        Equal.trans(Bool,False{},natural_valid(coordinate,padding,rows),True{},Equal.sym(Bool,natural_valid(coordinate,padding,rows),False{},local_choice),rows_backward(offset,coordinate,padding,global,rows,height,global_exact,global_choice,coverage))))

def rows_same(+offset: Nat,+coordinate: Nat,+padding: Nat,+global: Nat,+rows: Nat,+height: Nat,
  global_exact: {global == Nat.add(offset,coordinate) : Nat},span: {Nat.is_le(Nat.add(offset,rows),height) == True{} : Bool},
  coverage: Boolean.When(natural_valid(global,padding,height),TensorBandWindow.Contains(offset,Nat.add(offset,rows),Nat.sub(global,padding)))) ->
  {natural_valid(coordinate,padding,rows) == natural_valid(global,padding,height) : Bool}:
  rows_same_selected(offset,coordinate,padding,global,rows,height,global_exact,span,natural_valid(coordinate,padding,rows),natural_valid(global,padding,height),{==},{==},coverage)

def native_rows_same(+offset: Nat,+coordinate: U32,+padding: U32,+global: U32,+rows: U32,+height: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},coordinate_room: {Nat.is_le(U32.to_nat(coordinate),U32.to_nat(capacity)) == True{} : Bool},
  global_room: {Nat.is_le(U32.to_nat(global),U32.to_nat(capacity)) == True{} : Bool},rows_padding_room: {Nat.is_le(Nat.add(U32.to_nat(rows),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  height_padding_room: {Nat.is_le(Nat.add(U32.to_nat(height),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  shifted: {U32.to_nat(global) == Nat.add(offset,U32.to_nat(coordinate)) : Nat},span: {Nat.is_le(Nat.add(offset,U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool},
  coverage: Boolean.When(natural_valid(U32.to_nat(global),U32.to_nat(padding),U32.to_nat(height)),TensorBandWindow.Contains(offset,Nat.add(offset,U32.to_nat(rows)),Nat.sub(U32.to_nat(global),U32.to_nat(padding))))) ->
  {U32.is_lt((coordinate - padding : U32),rows) == U32.is_lt((global - padding : U32),height) : Bool}:
  Equal.trans(Bool,U32.is_lt((coordinate - padding : U32),rows),natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(rows)),U32.is_lt((global - padding : U32),height),
    axis_valid(coordinate,padding,rows,capacity,capacity_ok,coordinate_room,rows_padding_room),
    Equal.trans(Bool,natural_valid(U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(rows)),natural_valid(U32.to_nat(global),U32.to_nat(padding),U32.to_nat(height)),U32.is_lt((global - padding : U32),height),
      rows_same(offset,U32.to_nat(coordinate),U32.to_nat(padding),U32.to_nat(global),U32.to_nat(rows),U32.to_nat(height),shifted,span,coverage),
      Equal.sym(Bool,U32.is_lt((global - padding : U32),height),natural_valid(U32.to_nat(global),U32.to_nat(padding),U32.to_nat(height)),axis_valid(global,padding,height,capacity,capacity_ok,global_room,height_padding_room))))

def geometry_height(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  height

def geometry_images(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  images

def pool_conditions(pool: Access.Pool) -> List<&2,Bool>:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+output_height,+output_width} = pool
  [U32.is_eq(output_width,((width + 2 * padding - kernel : U32) / stride + 1 : U32)),
    U32.is_eq(output_height,((height + 2 * padding - kernel : U32) / stride + 1 : U32)),
    U32.is_le(kernel,(width + 2 * padding : U32)),U32.is_le(kernel,(height + 2 * padding : U32)),
    U32.is_lt(0,stride),U32.is_lt(0,kernel),Operations.small(padding),Operations.small(stride),Operations.small(kernel),Operations.small(width),Operations.small(height)]

def pool_condition(+pool: Access.Pool,+position: Nat,guard: {Operations.pool_geometry(pool) == True{} : Bool}) ->
  {Boolean.condition_at(pool_conditions(pool),position) == True{} : Bool}:
  Access.Pool{height,width,kernel,stride,padding,output_height,output_width} = pool
  Boolean.every_condition_holds(pool_conditions(pool),position,guard)

def pool_kernel(pool: Access.Pool) -> U32:
  Access.Pool{height,width,kernel,stride,padding,output_height,output_width} = pool
  kernel

def pool_stride(pool: Access.Pool) -> U32:
  Access.Pool{height,width,kernel,stride,padding,output_height,output_width} = pool
  stride

def pool_padding(pool: Access.Pool) -> U32:
  Access.Pool{height,width,kernel,stride,padding,output_height,output_width} = pool
  padding

def pool_height(pool: Access.Pool) -> U32:
  Access.Pool{height,width,kernel,stride,padding,output_height,output_width} = pool
  height

def pool_width(pool: Access.Pool) -> U32:
  Access.Pool{height,width,kernel,stride,padding,output_height,output_width} = pool
  width

def window_first(window: BandWindow.RowWindow) -> U32:
  BandWindow.RowWindow{output,above,below,BandWindow.Rows{first,rows}} = window
  first

def window_rows(window: BandWindow.RowWindow) -> U32:
  BandWindow.RowWindow{output,above,below,BandWindow.Rows{first,rows}} = window
  rows

def output_first(window: BandWindow.RowWindow) -> U32:
  BandWindow.RowWindow{BandWindow.Rows{first,rows},above,below,input} = window
  first

def output_rows(window: BandWindow.RowWindow) -> U32:
  BandWindow.RowWindow{BandWindow.Rows{first,rows},above,below,input} = window
  rows

def geometry_stride(geometry: BandWindow.Geometry) -> U32:
  BandWindow.Geometry{input,kernel,stride,padding,output_height,output_width} = geometry
  stride

def geometry_output_height(geometry: BandWindow.Geometry) -> U32:
  BandWindow.Geometry{input,kernel,stride,padding,output_height,output_width} = geometry
  output_height

def not_true_false(value: Bool,evidence: {Bool.not(value) == True{} : Bool}) -> {value == False{} : Bool}:
  match value:
    case False{}: {==}
    case True{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(evidence))

# This is tensor_window.band_geometry's current selector. Height zero or an
# empty spatial axis keeps the dense fallback.
def selected_pool_geometry(+pool: Access.Pool,+shape_valid: Bool,+out_shape: List<&2,U32>,
  checked: {shape_valid && Operations.pool_geometry(pool) && U32.is_lt(0,pool_height(pool)) && U32.is_lt(0,pool_width(pool)) && Bool.not(Shapes.has_zero(out_shape)) == True{} : Bool}) ->
  {Operations.pool_geometry(pool) == True{} : Bool}:
  Boolean.right(shape_valid,Operations.pool_geometry(pool),
    Boolean.left((shape_valid && Operations.pool_geometry(pool) : Bool),U32.is_lt(0,pool_height(pool)),
      Boolean.left((shape_valid && Operations.pool_geometry(pool) && U32.is_lt(0,pool_height(pool)) : Bool),U32.is_lt(0,pool_width(pool)),
        Boolean.left((shape_valid && Operations.pool_geometry(pool) && U32.is_lt(0,pool_height(pool)) && U32.is_lt(0,pool_width(pool)) : Bool),Bool.not(Shapes.has_zero(out_shape)),checked))))

def selected_input_height(+pool: Access.Pool,+shape_valid: Bool,+out_shape: List<&2,U32>,
  checked: {shape_valid && Operations.pool_geometry(pool) && U32.is_lt(0,pool_height(pool)) && U32.is_lt(0,pool_width(pool)) && Bool.not(Shapes.has_zero(out_shape)) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(pool_height(pool))) == True{} : Bool}:
  Bounds.as_nat_lt(0,pool_height(pool),
    Boolean.right((shape_valid && Operations.pool_geometry(pool) : Bool),U32.is_lt(0,pool_height(pool)),
      Boolean.left((shape_valid && Operations.pool_geometry(pool) && U32.is_lt(0,pool_height(pool)) : Bool),U32.is_lt(0,pool_width(pool)),
        Boolean.left((shape_valid && Operations.pool_geometry(pool) && U32.is_lt(0,pool_height(pool)) && U32.is_lt(0,pool_width(pool)) : Bool),Bool.not(Shapes.has_zero(out_shape)),checked))))

def selected_input_width(+pool: Access.Pool,+shape_valid: Bool,+out_shape: List<&2,U32>,
  checked: {shape_valid && Operations.pool_geometry(pool) && U32.is_lt(0,pool_height(pool)) && U32.is_lt(0,pool_width(pool)) && Bool.not(Shapes.has_zero(out_shape)) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(pool_width(pool))) == True{} : Bool}:
  Bounds.as_nat_lt(0,pool_width(pool),
    Boolean.right((shape_valid && Operations.pool_geometry(pool) && U32.is_lt(0,pool_height(pool)) : Bool),U32.is_lt(0,pool_width(pool)),
      Boolean.left((shape_valid && Operations.pool_geometry(pool) && U32.is_lt(0,pool_height(pool)) && U32.is_lt(0,pool_width(pool)) : Bool),Bool.not(Shapes.has_zero(out_shape)),checked)))

def selected_output_nonempty(+pool: Access.Pool,+shape_valid: Bool,+out_shape: List<&2,U32>,
  checked: {shape_valid && Operations.pool_geometry(pool) && U32.is_lt(0,pool_height(pool)) && U32.is_lt(0,pool_width(pool)) && Bool.not(Shapes.has_zero(out_shape)) == True{} : Bool}) ->
  {Shapes.has_zero(out_shape) == False{} : Bool}:
  not_true_false(Shapes.has_zero(out_shape),
    Boolean.right((shape_valid && Operations.pool_geometry(pool) && U32.is_lt(0,pool_height(pool)) && U32.is_lt(0,pool_width(pool)) : Bool),Bool.not(Shapes.has_zero(out_shape)),checked))

def remove_left_prefix(+base: U32,+local: U32) -> {((base + local : U32) - base : U32) == local : U32}:
  %Equal.sym(U32,(base + local : U32),(local + base : U32),Arithmetic.add_commutative(base,local)) : {(_ - base : U32) == local : U32}
  Subtraction.subtract_add(local,base)

# Cancellation is a word identity: the two image bases need no product bound.
def common_word_offset(+left: U32,+right: U32,+offset: U32) ->
  {((left + offset : U32) - (right + offset : U32) : U32) == (left - right : U32) : U32}:
  +difference = ((left + offset : U32) - (right + offset : U32) : U32)
  +restored = Equal.trans(U32,(difference + right : U32),((difference + right + offset : U32) - offset : U32),left,
    Equal.sym(U32,((difference + right + offset : U32) - offset : U32),(difference + right : U32),Subtraction.subtract_add((difference + right : U32),offset)),
    Equal.trans(U32,((difference + right + offset : U32) - offset : U32),((left + offset : U32) - offset : U32),left,
      Equal.cong(U32,U32,value => (value - offset : U32),(difference + right + offset : U32),(left + offset : U32),
        Equal.trans(U32,(difference + right + offset : U32),(difference + (right + offset : U32) : U32),(left + offset : U32),
          Arithmetic.add_associative(difference,right,offset),Subtraction.add_subtract((left + offset : U32),(right + offset : U32)))),
      Subtraction.subtract_add(left,offset)))
  Equal.trans(U32,difference,((difference + right : U32) - right : U32),(left - right : U32),
    Equal.sym(U32,((difference + right : U32) - right : U32),difference,Subtraction.subtract_add(difference,right)),
    Equal.cong(U32,U32,value => (value - right : U32),(difference + right : U32),left,restored))

def origin_nonempty(+geometry: BandWindow.Geometry,+first: U32,+window: BandWindow.RowWindow,active: Bool) ->
  {BandPoolExecution.window_origin(geometry,first,BandWindow.nonempty(active,window)) ==
    BandPoolExecution.window_origin(geometry,first,window) : U32}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{output_first,output_rows},BandWindow.Rows{above_first,above_rows},BandWindow.Rows{below_first,below_rows},BandWindow.Rows{window_first,window_rows}} = window
  match active:
    case False{}: {==}
    case True{}: {==}

# The actual caller's global output/window bases cancel before any Nat decode.
# nonempty preserves both first positions even when it sets window rows to zero.
def origin_local(+geometry: BandWindow.Geometry,+first: U32) -> U32:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},kernel,+stride,+padding,+output_height,output_width} = geometry
  +begin = TensorBandWindow.output_first(height,output_height,stride,first)
  (begin * stride - BandWindow.start(height,begin,stride,padding) : U32)

def origin_native(+geometry: BandWindow.Geometry,+first: U32,+rows: U32) ->
  {BandPoolExecution.window_origin(geometry,first,BandWindow.window(geometry,first,rows)) ==
    origin_local(geometry,first) : U32}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},+kernel,+stride,+padding,+output_height,output_width} = geometry
  +image = (first / height : U32)
  +core_first = (first % height : U32)
  +core_end = (core_first + rows : U32)
  +begin = TensorBandWindow.output_first(height,output_height,stride,first)
  +finish = TensorBandWindow.output_end(height,output_height,stride,first,rows)
  +start = BandWindow.start(height,begin,stride,padding)
  +end = TensorBandWindow.samples_end(height,finish,stride,kernel,padding)
  +active = U32.is_lt(begin,finish)
  +raw = {BandWindow.RowWindow{BandWindow.Rows{(image * output_height + begin : U32),BandWindow.subtract_clipped(finish,begin)},
    BandWindow.clipped((image * height + start : U32),(image * height + U32.min(core_first,end) : U32)),
    BandWindow.clipped((image * height + U32.max(core_end,start) : U32),(image * height + end : U32)),
    BandWindow.clipped((image * height + start : U32),(image * height + end : U32))} : BandWindow.RowWindow}
  Equal.trans(U32,BandPoolExecution.window_origin(geometry,first,BandWindow.window(geometry,first,rows)),
    BandPoolExecution.window_origin(geometry,first,raw),origin_local(geometry,first),
    origin_nonempty(geometry,first,raw,active),
    %Equal.sym(U32,((image * output_height + begin : U32) - (image * output_height : U32) : U32),begin,remove_left_prefix((image * output_height : U32),begin)) :
      {(_ * stride + image * height - (image * height + start : U32) : U32) == (begin * stride - start : U32) : U32}
    %Equal.sym(U32,(image * height + start : U32),(start + image * height : U32),Arithmetic.add_commutative((image * height : U32),start)) :
      {((begin * stride + image * height : U32) - _ : U32) == (begin * stride - start : U32) : U32}
    common_word_offset((begin * stride : U32),start,(image * height : U32)))

def start_below_first(+height: U32,+begin: U32,+stride: U32,+padding: U32) ->
  {Nat.is_le(U32.to_nat(BandWindow.start(height,begin,stride,padding)),U32.to_nat(TensorBandWindow.samples_first(height,begin,stride,padding))) == True{} : Bool}:
  TensorBandWindow.clipped_below(TensorBandWindow.samples_first(height,begin,stride,padding),(TensorBandWindow.samples_first(height,begin,stride,padding) % stride : U32))

def start_below_height(+height: U32,+begin: U32,+stride: U32,+padding: U32) ->
  {Nat.is_le(U32.to_nat(BandWindow.start(height,begin,stride,padding)),U32.to_nat(height)) == True{} : Bool}:
  Order.transitive(U32.to_nat(BandWindow.start(height,begin,stride,padding)),U32.to_nat(TensorBandWindow.samples_first(height,begin,stride,padding)),U32.to_nat(height),
    start_below_first(height,begin,stride,padding),
    %Equal.sym(Nat,U32.to_nat(TensorBandWindow.samples_first(height,begin,stride,padding)),Nat.min(U32.to_nat(height),U32.to_nat(BandWindow.subtract_clipped((begin * stride : U32),padding))),
      U32Extrema.minimum(height,BandWindow.subtract_clipped((begin * stride : U32),padding))) : {Nat.is_le(_,U32.to_nat(height)) == True{} : Bool}
    Extrema.minimum_left(U32.to_nat(height),U32.to_nat(BandWindow.subtract_clipped((begin * stride : U32),padding))))

def start_below_product(+height: U32,+begin: U32,+stride: U32,+padding: U32) ->
  {Nat.is_le(U32.to_nat(BandWindow.start(height,begin,stride,padding)),U32.to_nat((begin * stride : U32))) == True{} : Bool}:
  Order.transitive(U32.to_nat(BandWindow.start(height,begin,stride,padding)),U32.to_nat(TensorBandWindow.samples_first(height,begin,stride,padding)),U32.to_nat((begin * stride : U32)),
    start_below_first(height,begin,stride,padding),
    Order.transitive(U32.to_nat(TensorBandWindow.samples_first(height,begin,stride,padding)),U32.to_nat(BandWindow.subtract_clipped((begin * stride : U32),padding)),U32.to_nat((begin * stride : U32)),
      %Equal.sym(Nat,U32.to_nat(TensorBandWindow.samples_first(height,begin,stride,padding)),Nat.min(U32.to_nat(height),U32.to_nat(BandWindow.subtract_clipped((begin * stride : U32),padding))),
        U32Extrema.minimum(height,BandWindow.subtract_clipped((begin * stride : U32),padding))) : {Nat.is_le(_,U32.to_nat(BandWindow.subtract_clipped((begin * stride : U32),padding))) == True{} : Bool}
      Extrema.minimum_right(U32.to_nat(height),U32.to_nat(BandWindow.subtract_clipped((begin * stride : U32),padding))),
      TensorBandWindow.clipped_below((begin * stride : U32),padding)))

def origin_nat(+geometry: BandWindow.Geometry,+first: U32) -> Nat:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},kernel,+stride,+padding,+output_height,output_width} = geometry
  +begin = TensorBandWindow.output_first(height,output_height,stride,first)
  Nat.sub(Nat.mul(U32.to_nat(begin),U32.to_nat(stride)),U32.to_nat(BandWindow.start(height,begin,stride,padding)))

# The sole arithmetic envelope is the actual first output row's stride product.
# No successful native sample predicate or address equality is a premise.
def origin_decoded(+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +product_room: {Nat.is_le(Nat.mul(U32.to_nat(TensorBandWindow.output_first(geometry_height(BandWindow.input_geometry(geometry)),
    geometry_output_height(geometry),geometry_stride(geometry),first)),U32.to_nat(geometry_stride(geometry))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(BandPoolExecution.window_origin(geometry,first,BandWindow.window(geometry,first,rows))) == origin_nat(geometry,first) : Nat}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},kernel,+stride,+padding,+output_height,output_width} = geometry
  +begin = TensorBandWindow.output_first(height,output_height,stride,first)
  +start = BandWindow.start(height,begin,stride,padding)
  +product_exact = Arithmetic.multiply(begin,stride,capacity,capacity_ok,product_room)
  Equal.trans(Nat,U32.to_nat(BandPoolExecution.window_origin(geometry,first,BandWindow.window(geometry,first,rows))),U32.to_nat(origin_local(geometry,first)),origin_nat(geometry,first),
    Equal.cong(U32,Nat,value => U32.to_nat(value),BandPoolExecution.window_origin(geometry,first,BandWindow.window(geometry,first,rows)),origin_local(geometry,first),origin_native(geometry,first,rows)),
    Equal.trans(Nat,U32.to_nat((begin * stride - start : U32)),Nat.sub(U32.to_nat((begin * stride : U32)),U32.to_nat(start)),origin_nat(geometry,first),
      Arithmetic.subtract((begin * stride : U32),start,capacity,capacity_ok,start_below_product(height,begin,stride,padding),
        Order.transitive(Nat.sub(U32.to_nat((begin * stride : U32)),U32.to_nat(start)),U32.to_nat((begin * stride : U32)),U32.to_nat(capacity),Order.sub_le(U32.to_nat((begin * stride : U32)),U32.to_nat(start)),
          %Equal.sym(Nat,U32.to_nat((begin * stride : U32)),Nat.mul(U32.to_nat(begin),U32.to_nat(stride)),product_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} product_room)),
      Equal.cong(Nat,Nat,value => Nat.sub(value,U32.to_nat(start)),U32.to_nat((begin * stride : U32)),Nat.mul(U32.to_nat(begin),U32.to_nat(stride)),product_exact)))

# TensorBandWindow.samples_first is before the first valid sample, while BandWindow.start moves
# it farther down to a stride multiple. That widening preserves coverage.
def start_coverage(+height: U32,+begin: U32,+stride: U32,+padding: U32,+end: U32,+coordinate: Nat,
  evidence: TensorBandWindow.Contains(U32.to_nat(TensorBandWindow.samples_first(height,begin,stride,padding)),U32.to_nat(end),coordinate)) ->
  TensorBandWindow.Contains(U32.to_nat(BandWindow.start(height,begin,stride,padding)),U32.to_nat(end),coordinate):
  Shape.Both{lower,upper} = evidence
  Shape.Both{Order.transitive(U32.to_nat(BandWindow.start(height,begin,stride,padding)),U32.to_nat(TensorBandWindow.samples_first(height,begin,stride,padding)),coordinate,
    start_below_first(height,begin,stride,padding),lower),upper}

def clipped_sample(+height: U32,+begin: U32,+finish: U32,+stride: U32,+kernel: U32,+padding: U32,+output_row: Nat,+kernel_row: Nat,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},end_room: {Nat.is_le(U32.to_nat(finish),U32.to_nat(capacity)) == True{} : Bool},
  window_room: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(U32.to_nat(finish),1n),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(capacity)) == True{} : Bool},
  padding_room: {Nat.is_le(U32.to_nat(padding),U32.to_nat(capacity)) == True{} : Bool},
  output: TensorBandWindow.Contains(U32.to_nat(begin),U32.to_nat(finish),output_row),kernel_inside: {Nat.is_lt(kernel_row,U32.to_nat(kernel)) == True{} : Bool},
  unpadded: {Nat.is_le(U32.to_nat(padding),Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row)) == True{} : Bool},
  image_inside: {Nat.is_lt(Nat.sub(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding)),U32.to_nat(height)) == True{} : Bool}) ->
  TensorBandWindow.Contains(U32.to_nat(BandWindow.start(height,begin,stride,padding)),U32.to_nat(TensorBandWindow.samples_end(height,finish,stride,kernel,padding)),
    Nat.sub(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding))):
  start_coverage(height,begin,stride,padding,TensorBandWindow.samples_end(height,finish,stride,kernel,padding),Nat.sub(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding)),
    TensorBandWindow.samples_point(height,begin,finish,stride,kernel,padding,output_row,kernel_row,capacity,capacity_ok,end_room,window_room,padding_room,output,kernel_inside,unpadded,image_inside))

def clipped_sample_selected(+height: U32,+begin: U32,+finish: U32,+stride: U32,+kernel: U32,+padding: U32,+output_row: Nat,+kernel_row: Nat,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},end_room: {Nat.is_le(U32.to_nat(finish),U32.to_nat(capacity)) == True{} : Bool},
  window_room: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(U32.to_nat(finish),1n),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(capacity)) == True{} : Bool},
  padding_room: {Nat.is_le(U32.to_nat(padding),U32.to_nat(capacity)) == True{} : Bool},
  output: TensorBandWindow.Contains(U32.to_nat(begin),U32.to_nat(finish),output_row),kernel_inside: {Nat.is_lt(kernel_row,U32.to_nat(kernel)) == True{} : Bool},choice: Bool,
  +compared: {natural_valid(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding),U32.to_nat(height)) == choice : Bool}) ->
  Boolean.When(choice,TensorBandWindow.Contains(U32.to_nat(BandWindow.start(height,begin,stride,padding)),U32.to_nat(TensorBandWindow.samples_end(height,finish,stride,kernel,padding)),
    Nat.sub(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding)))):
  match choice:
    case False{}: Unit{}
    case True{}:
      +coordinate = Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row)
      clipped_sample(height,begin,finish,stride,kernel,padding,output_row,kernel_row,capacity,capacity_ok,end_room,window_room,padding_room,output,kernel_inside,
        Boolean.left(Nat.is_le(U32.to_nat(padding),coordinate),Nat.is_lt(Nat.sub(coordinate,U32.to_nat(padding)),U32.to_nat(height)),compared),
        Boolean.right(Nat.is_le(U32.to_nat(padding),coordinate),Nat.is_lt(Nat.sub(coordinate,U32.to_nat(padding)),U32.to_nat(height)),compared))

def clipped_sample_when(+height: U32,+begin: U32,+finish: U32,+stride: U32,+kernel: U32,+padding: U32,+output_row: Nat,+kernel_row: Nat,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},end_room: {Nat.is_le(U32.to_nat(finish),U32.to_nat(capacity)) == True{} : Bool},
  window_room: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(U32.to_nat(finish),1n),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(capacity)) == True{} : Bool},
  padding_room: {Nat.is_le(U32.to_nat(padding),U32.to_nat(capacity)) == True{} : Bool},
  output: TensorBandWindow.Contains(U32.to_nat(begin),U32.to_nat(finish),output_row),kernel_inside: {Nat.is_lt(kernel_row,U32.to_nat(kernel)) == True{} : Bool}) ->
  Boolean.When(natural_valid(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding),U32.to_nat(height)),
    TensorBandWindow.Contains(U32.to_nat(BandWindow.start(height,begin,stride,padding)),U32.to_nat(TensorBandWindow.samples_end(height,finish,stride,kernel,padding)),
      Nat.sub(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding)))):
  clipped_sample_selected(height,begin,finish,stride,kernel,padding,output_row,kernel_row,capacity,capacity_ok,end_room,window_room,padding_room,output,kernel_inside,
    natural_valid(Nat.add(Nat.mul(output_row,U32.to_nat(stride)),kernel_row),U32.to_nat(padding),U32.to_nat(height)),{==})

# Combining the caller's first-row origin with a local output row gives the
# exact global padded coordinate, in the original row/tap order.
def shifted_origin(+start: Nat,+begin: Nat,+stride: Nat,+row: Nat,+tap: Nat,
  ordered: {Nat.is_le(start,Nat.mul(begin,stride)) == True{} : Bool}) ->
  {Nat.add(Nat.mul(Nat.add(begin,row),stride),tap) ==
    Nat.add(start,Nat.add(Nat.add(Nat.mul(row,stride),Nat.sub(Nat.mul(begin,stride),start)),tap)) : Nat}:
  +base = Nat.mul(begin,stride)
  +local = Nat.mul(row,stride)
  +difference = Nat.sub(base,start)
  +restored = Order.restore_shift(start,base,ordered)
  +shifted = Equal.trans(Nat,Nat.add(start,Nat.add(Nat.add(local,difference),tap)),Nat.add(start,Nat.add(local,Nat.add(difference,tap))),Nat.add(Nat.add(base,local),tap),
    Equal.cong(Nat,Nat,value => Nat.add(start,value),Nat.add(Nat.add(local,difference),tap),Nat.add(local,Nat.add(difference,tap)),Addition.associative(local,difference,tap)),
    Equal.trans(Nat,Nat.add(start,Nat.add(local,Nat.add(difference,tap))),Nat.add(local,Nat.add(start,Nat.add(difference,tap))),Nat.add(Nat.add(base,local),tap),
      RowMajor.add_swap(start,local,Nat.add(difference,tap)),
      Equal.trans(Nat,Nat.add(local,Nat.add(start,Nat.add(difference,tap))),Nat.add(local,Nat.add(Nat.add(start,difference),tap)),Nat.add(Nat.add(base,local),tap),
        Equal.cong(Nat,Nat,value => Nat.add(local,value),Nat.add(start,Nat.add(difference,tap)),Nat.add(Nat.add(start,difference),tap),
          Equal.sym(Nat,Nat.add(Nat.add(start,difference),tap),Nat.add(start,Nat.add(difference,tap)),Addition.associative(start,difference,tap))),
        Equal.trans(Nat,Nat.add(local,Nat.add(Nat.add(start,difference),tap)),Nat.add(local,Nat.add(base,tap)),Nat.add(Nat.add(base,local),tap),
          Equal.cong(Nat,Nat,value => Nat.add(local,Nat.add(value,tap)),Nat.add(start,difference),base,restored),
          Equal.trans(Nat,Nat.add(local,Nat.add(base,tap)),Nat.add(base,Nat.add(local,tap)),Nat.add(Nat.add(base,local),tap),RowMajor.add_swap(local,base,tap),
            Equal.sym(Nat,Nat.add(Nat.add(base,local),tap),Nat.add(base,Nat.add(local,tap)),Addition.associative(base,local,tap)))))))
  Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(begin,row),stride),tap),Nat.add(Nat.add(base,local),tap),Nat.add(start,Nat.add(Nat.add(local,difference),tap)),
    Equal.cong(Nat,Nat,value => Nat.add(value,tap),Nat.mul(Nat.add(begin,row),stride),Nat.add(base,local),RowMajor.distrib(begin,row,stride)),
    Equal.sym(Nat,Nat.add(start,Nat.add(Nat.add(local,difference),tap)),Nat.add(Nat.add(base,local),tap),shifted))
