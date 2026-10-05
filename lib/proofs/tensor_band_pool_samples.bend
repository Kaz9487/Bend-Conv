# What the band pool supplies to its two sweeps, from one formal machine
# observation. The selected positive source axes and the counted input shape
# give the source address room; the counted planes and output give the room
# of both destinations. For every output row of a leaf and every tap, the
# row of the assembled window is the row of the image, or both are padding.
# Empty local halos and leaves without output rows remain valid in this path.
# No final sample or pool equality is supplied as a premise.
import Base
import ../tensor.bend as Tensors
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ../tensor_band_pool.bend as BandPool
import ../tensor_window.bend as Window
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_band_pool_execution.bend as BandPoolExecution
import ./tensor_band_pool_window_values.bend as BandPoolWindowValues
import ./tensor_band_pool_output_domain.bend as OutputDomain
import ./tensor_pool_scalar_bounds.bend as Scalar
import ./tensor_pool_scalar_envelope.bend as ScalarEnvelope
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_model.bend as Model
import ./tensor_band_values.bend as BandValues
import ./tensor_sweep_values.bend as Sweeps
import ./tensor_band_sweep_values.bend as BandSweeps
import ./encoded_coordinates.bend as Encoded
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_halo.bend as BandHalo
import ./tensor_band_window.bend as TensorBandWindow
import ./tensor_band_coordinates.bend as BandCoordinates
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./u32_comparison.bend as Comparison
import ./u32_division.bend as U32Division
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_band_dense.bend as BandDense
import ./tensor_band_inspection.bend as Inspection
import ./strided_range_bounds.bend as Strided
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_count.bend as BandCount
import ./tensor_band_range.bend as BandRange
import ./write_program.bend as Writes
import ./checked_word_refinement.bend as Checked
import ./tensor_global_shape_count.bend as Global

def positive_difference_order(+begin: Nat,+finish: Nat,positive: {Nat.is_lt(0n,Nat.sub(finish,begin)) == True{} : Bool}) ->
  {Nat.is_lt(begin,finish) == True{} : Bool}:
  match begin finish:
    case 0n 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(positive))
    case 0n 1n+finish: {==}
    case 1n+begin 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+ +begin 1n+ +finish: positive_difference_order(begin,finish,positive)

type OutputInterval<-geometry: BandWindow.Geometry,-first: U32,-rows: U32> is Data:
  OutputInterval{
    first_exact: {U32.to_nat(BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows))) ==
      Nat.add(Nat.mul(Nat.div(U32.to_nat(first),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))),U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry))),
        U32.to_nat(TensorBandWindow.output_first(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first))) : Nat},
    rows_exact: {U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))) == Nat.sub(
      U32.to_nat(TensorBandWindow.output_end(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first,rows)),
      U32.to_nat(TensorBandWindow.output_first(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first))) : Nat},
    begin_before_finish: {Nat.is_lt(U32.to_nat(TensorBandWindow.output_first(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first)),
      U32.to_nat(TensorBandWindow.output_end(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first,rows))) == True{} : Bool},
    source_image: {Nat.div(U32.to_nat(BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows))),U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry))) ==
      Nat.div(U32.to_nat(first),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))) : Nat},
    source_row_origin: {Nat.mod(U32.to_nat(BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows))),U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry))) ==
      U32.to_nat(TensorBandWindow.output_first(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first)) : Nat}}

def interval_from_mapping(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,+length: Nat,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  source: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length),
  mapping: TensorBandWindow.LocalMapping<BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first,rows>,
  positive: {Nat.is_lt(0n,U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)))) == True{} : Bool}) -> OutputInterval<geometry,first,rows>:
  BandWindow.Geometry{Bands.Geometry{+images,channels,+height,width},kernel,+stride,padding,+output_height,output_width} = geometry
  TensorBandWindow.LocalMapping{+begin_shift,+end_shift,+ordered} = mapping
  +domain = OutputDomain.from_selector(machine,shape,geometry,count,selected,counted)
  +op = OutputDomain.positive_from_selector(shape,geometry,selected)
  +hp = OutputDomain.selected_height(shape,geometry,selected)
  +sp = OutputDomain.selected_stride(shape,geometry,selected)
  +hr = OutputDomain.selected_height_room(machine,shape,geometry,selected)
  +sd = Limits.double_bound(Scalar.limits(machine),U32.to_nat(stride),OutputDomain.selected_stride_room(machine,shape,geometry,selected))
  +hd = Limits.double_bound(Scalar.limits(machine),U32.to_nat(height),hr)
  +ohr = OutputDomain.output_height_room(machine,geometry,op,domain)
  +ohp = OutputDomain.positive_height(geometry,op)
  +capacity = Machine.capacity(Scalar.limits(machine))
  +half = Machine.half_word(Scalar.limits(machine))
  +image = Nat.div(U32.to_nat(first),U32.to_nat(height))
  +begin = U32.to_nat(TensorBandWindow.output_first(height,output_height,stride,first))
  +finish = U32.to_nat(TensorBandWindow.output_end(height,output_height,stride,first,rows))
  +mapped_begin = BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(first))
  +mapped_end = BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(first),U32.to_nat(rows)))
  +mapped_ordered = BandPlan.monotone(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),hp,sp,Order.le_add(U32.to_nat(first),U32.to_nat(rows)))
  +begin_room = Order.transitive(mapped_begin,mapped_end,U32.to_nat(capacity),mapped_ordered,
    Order.transitive(mapped_end,Nat.mul(U32.to_nat(images),U32.to_nat(output_height)),U32.to_nat(capacity),OutputDomain.mapped_end_inside(geometry,first,rows,hp,sp,OutputDomain.source_fit(geometry,first,rows,length,source)),
      BandGather.rows_room(BandPoolExecution.values_output_geometry(geometry),Scalar.limits(machine),capacity,domain)))
  +out_first = BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows))
  +out_rows = BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))
  +first_mapped = Equal.trans(Nat,U32.to_nat(out_first),U32.to_nat((first / height * output_height + TensorBandWindow.output_first(height,output_height,stride,first) : U32)),mapped_begin,
    Equal.cong(U32,Nat,value => U32.to_nat(value),out_first,(first / height * output_height + TensorBandWindow.output_first(height,output_height,stride,first) : U32),OutputDomain.window_first(geometry,first,rows)),
    TensorBandWindow.first_global(height,output_height,stride,first,capacity,half,Limits.capacity_fits(Scalar.limits(machine)),Limits.half_word_fits(Scalar.limits(machine)),hp,sp,hd,sd,hr,begin_room))
  +first_exact = Equal.trans(Nat,U32.to_nat(out_first),mapped_begin,Nat.add(Nat.mul(image,U32.to_nat(output_height)),begin),first_mapped,begin_shift)
  +rows_mapped = Equal.trans(Nat,U32.to_nat(out_rows),U32.to_nat(BandWindow.subtract_clipped(TensorBandWindow.output_end(height,output_height,stride,first,rows),TensorBandWindow.output_first(height,output_height,stride,first))),Nat.sub(mapped_end,mapped_begin),
    Equal.cong(U32,Nat,value => U32.to_nat(value),out_rows,BandWindow.subtract_clipped(TensorBandWindow.output_end(height,output_height,stride,first,rows),TensorBandWindow.output_first(height,output_height,stride,first)),OutputDomain.window_rows(geometry,first,rows)),
    TensorBandWindow.rows_mapped(height,output_height,stride,first,rows,capacity,Limits.capacity_fits(Scalar.limits(machine)),ohr,TensorBandWindow.LocalMapping{begin_shift,end_shift,ordered}))
  +rows_exact = Equal.trans(Nat,U32.to_nat(out_rows),Nat.sub(mapped_end,mapped_begin),Nat.sub(finish,begin),rows_mapped,
    OutputDomain.mapped_difference_local(height,output_height,stride,first,rows,TensorBandWindow.LocalMapping{begin_shift,end_shift,ordered}))
  +before = positive_difference_order(begin,finish,
    %rows_exact : {Nat.is_lt(0n,_) == True{} : Bool} positive)
  +begin_inside = Bounds.nat_trans(begin,finish,U32.to_nat(output_height),before,
    TensorBandWindow.end_below(output_height,(first % height + rows : U32),stride,U32.is_eq((first % height + rows : U32),height)))
  OutputInterval{first_exact,rows_exact,before,
    Equal.trans(Nat,Nat.div(U32.to_nat(out_first),U32.to_nat(output_height)),Nat.div(Nat.add(Nat.mul(image,U32.to_nat(output_height)),begin),U32.to_nat(output_height)),image,
      Equal.cong(Nat,Nat,value => Nat.div(value,U32.to_nat(output_height)),U32.to_nat(out_first),Nat.add(Nat.mul(image,U32.to_nat(output_height)),begin),first_exact),BandCoordinates.quotient(image,begin,U32.to_nat(output_height),ohp,begin_inside)),
    Equal.trans(Nat,Nat.mod(U32.to_nat(out_first),U32.to_nat(output_height)),Nat.mod(Nat.add(Nat.mul(image,U32.to_nat(output_height)),begin),U32.to_nat(output_height)),begin,
      Equal.cong(Nat,Nat,value => Nat.mod(value,U32.to_nat(output_height)),U32.to_nat(out_first),Nat.add(Nat.mul(image,U32.to_nat(output_height)),begin),first_exact),BandCoordinates.remainder(image,begin,U32.to_nat(output_height),ohp,begin_inside))}

def needed_first(+geometry: BandWindow.Geometry,+first: U32) -> U32:
  BandWindow.start(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),TensorBandWindow.output_first(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first),
    BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_padding(OutputDomain.pool(geometry)))

def needed_end(+geometry: BandWindow.Geometry,+first: U32,+rows: U32) -> U32:
  TensorBandWindow.samples_end(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),TensorBandWindow.output_end(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first,rows),
    BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_kernel(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_padding(OutputDomain.pool(geometry)))

def nonempty_input_first(active: Bool,window: BandWindow.RowWindow) ->
  {BandPoolWindowValues.window_first(BandWindow.nonempty(active,window)) == BandPoolWindowValues.window_first(window) : U32}:
  match active window:
    case True{} window: {==}
    case False{} BandWindow.RowWindow{output,BandWindow.Rows{upper_first,upper_rows},BandWindow.Rows{lower_first,lower_rows},BandWindow.Rows{first,rows}}: {==}

def input_first_native(+geometry: BandWindow.Geometry,+first: U32,+rows: U32) ->
  {BandPoolWindowValues.window_first(BandWindow.window(geometry,first,rows)) == (first / BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)) * BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)) + needed_first(geometry,first) : U32) : U32}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},+kernel,+stride,+padding,+output_height,output_width} = geometry
  +image = (first / height : U32)
  +core_first = (first % height : U32)
  +core_end = (core_first + rows : U32)
  +begin = TensorBandWindow.output_first(height,output_height,stride,first)
  +finish = TensorBandWindow.output_end(height,output_height,stride,first,rows)
  +start = needed_first(geometry,first)
  +end = needed_end(geometry,first,rows)
  nonempty_input_first(U32.is_lt(begin,finish),TensorBandWindow.raw_window(height,image,core_first,core_end,start,end,BandWindow.Rows{(image * output_height + begin : U32),BandWindow.subtract_clipped(finish,begin)}))

def input_rows_active(+geometry: BandWindow.Geometry,+first: U32,+rows: U32,
  active: {U32.is_lt(TensorBandWindow.output_first(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first),
    TensorBandWindow.output_end(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first,rows)) == True{} : Bool}) ->
  {BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows)) == BandWindow.subtract_clipped(
    (first / BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)) * BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)) + needed_end(geometry,first,rows) : U32),
    (first / BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)) * BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)) + needed_first(geometry,first) : U32)) : U32}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},kernel,+stride,padding,+output_height,output_width} = geometry
  %Equal.sym(Bool,U32.is_lt(TensorBandWindow.output_first(height,output_height,stride,first),TensorBandWindow.output_end(height,output_height,stride,first,rows)),True{},active) :
    {BandPoolWindowValues.window_rows(BandWindow.nonempty(_,TensorBandWindow.raw_window(height,(first / height : U32),(first % height : U32),(first % height + rows : U32),needed_first(geometry,first),needed_end(geometry,first,rows),
      BandWindow.Rows{(first / height * output_height + TensorBandWindow.output_first(height,output_height,stride,first) : U32),BandWindow.subtract_clipped(TensorBandWindow.output_end(height,output_height,stride,first,rows),TensorBandWindow.output_first(height,output_height,stride,first))}))) ==
      BandWindow.subtract_clipped((first / height * height + needed_end(geometry,first,rows) : U32),(first / height * height + needed_first(geometry,first) : U32)) : U32}
  {==}

def endpoint_exact(+machine: Scalar.Observation,+geometry: Bands.Geometry,+image: U32,+local: U32,+domain: BandGather.Domain<geometry,Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>,
  image_inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(BandPoolWindowValues.geometry_images(geometry))) == True{} : Bool},
  +local_inside: {Nat.is_le(U32.to_nat(local),U32.to_nat(BandPoolWindowValues.geometry_height(geometry))) == True{} : Bool}) ->
  {U32.to_nat((image * BandPoolWindowValues.geometry_height(geometry) + local : U32)) == Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(BandPoolWindowValues.geometry_height(geometry))),U32.to_nat(local)) : Nat}:
  Bands.Geometry{+images,channels,+height,width} = geometry
  +base = (image * height : U32)
  +base_nat = Nat.mul(U32.to_nat(image),U32.to_nat(height))
  +end = Nat.add(base_nat,U32.to_nat(height))
  +total = Nat.mul(U32.to_nat(images),U32.to_nat(height))
  +capacity = Machine.capacity(Scalar.limits(machine))
  +end_room = Order.transitive(end,total,U32.to_nat(capacity),
    %Order.commutative(U32.to_nat(height),base_nat) : {Nat.is_le(_,total) == True{} : Bool}
    RowMajor.mul_le(1n+U32.to_nat(image),U32.to_nat(images),U32.to_nat(height),Order.strict_follow(U32.to_nat(image),U32.to_nat(images),image_inside)),BandGather.rows_room(geometry,Scalar.limits(machine),capacity,domain))
  +base_exact = Arithmetic.multiply(image,height,capacity,Limits.capacity_fits(Scalar.limits(machine)),Order.transitive(base_nat,end,U32.to_nat(capacity),Order.le_add(base_nat,U32.to_nat(height)),end_room))
  Equal.trans(Nat,U32.to_nat((base + local : U32)),Nat.add(U32.to_nat(base),U32.to_nat(local)),Nat.add(base_nat,U32.to_nat(local)),
    Arithmetic.add(base,local,capacity,Limits.capacity_fits(Scalar.limits(machine)),
      %Equal.sym(Nat,U32.to_nat(base),base_nat,base_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(local)),U32.to_nat(capacity)) == True{} : Bool}
      Order.transitive(Nat.add(base_nat,U32.to_nat(local)),end,U32.to_nat(capacity),Order.offset_le(base_nat,U32.to_nat(local),U32.to_nat(height),local_inside),end_room)),
    Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(local)),U32.to_nat(base),base_nat,base_exact))

type WindowRange<-machine: Scalar.Observation,-geometry: BandWindow.Geometry,-first: U32,-rows: U32> is Data:
  WindowRange{
    first_exact: {U32.to_nat(BandPoolWindowValues.window_first(BandWindow.window(geometry,first,rows))) == Nat.add(
      Nat.mul(Nat.div(U32.to_nat(first),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))),U32.to_nat(needed_first(geometry,first))) : Nat},
    rows_exact: {U32.to_nat(BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows))) == Nat.sub(U32.to_nat(needed_end(geometry,first,rows)),U32.to_nat(needed_first(geometry,first))) : Nat},
    target: BandDense.LeafFacts<BandWindow.input_geometry(geometry),BandPoolWindowValues.window_first(BandWindow.window(geometry,first,rows)),BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows)),Machine.capacity(Scalar.limits(machine)),Machine.half_word(Scalar.limits(machine))>}

# The stored rows of a window with output rows: its first row and its count
# in the coordinates of the image of the leaf. The image is inside the batch
# because the leaf has a source row.
def window_range(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+length: Nat,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+domain: BandGather.Domain<BandWindow.input_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>,
  source: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length),positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  interval: OutputInterval<geometry,first,rows>) -> WindowRange<machine,geometry,first,rows>:
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = source
  OutputInterval{first_exact,rows_exact,+before,source_image,source_row_origin} = interval
  +core_image = Nat.div(U32.to_nat(first),U32.to_nat(height))
  +core_image_inside = {
    %Addition.zero_right(U32.to_nat(first)) : {Nat.is_lt(Nat.div(_,U32.to_nat(height)),U32.to_nat(images)) == True{} : Bool}
    BandSweeps.image_below(U32.to_nat(first),U32.to_nat(rows),0n,U32.to_nat(images),U32.to_nat(height),OutputDomain.selected_height(shape,geometry,selected),positive,rows_fit) : {Nat.is_lt(core_image,U32.to_nat(images)) == True{} : Bool}}
  +native_image = (first / height : U32)
  +native_image_exact = U32Division.divide(first,height,Machine.half_word(Scalar.limits(machine)),Limits.half_word_fits(Scalar.limits(machine)),
    Limits.double_bound(Scalar.limits(machine),U32.to_nat(height),OutputDomain.selected_height_room(machine,shape,geometry,selected)),OutputDomain.selected_height(shape,geometry,selected))
  +native_image_inside = {
    %Equal.sym(Nat,U32.to_nat(native_image),core_image,native_image_exact) : {Nat.is_lt(_,U32.to_nat(images)) == True{} : Bool} core_image_inside : {Nat.is_lt(U32.to_nat(native_image),U32.to_nat(images)) == True{} : Bool}}
  +start = needed_first(geometry,first)
  +end = needed_end(geometry,first,rows)
  +start_inside = BandPoolWindowValues.start_below_height(height,TensorBandWindow.output_first(height,output_height,stride,first),stride,padding)
  +end_inside = TensorBandWindow.minimum_below(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(TensorBandWindow.output_end(height,output_height,stride,first,rows),1) * stride + kernel : U32),padding))
  +start_full = (native_image * height + start : U32)
  +end_full = (native_image * height + end : U32)
  +se = endpoint_exact(machine,BandWindow.input_geometry(geometry),native_image,start,domain,native_image_inside,start_inside)
  +ee = endpoint_exact(machine,BandWindow.input_geometry(geometry),native_image,end,domain,native_image_inside,end_inside)
  +base = Nat.mul(U32.to_nat(native_image),U32.to_nat(height))
  +total = Nat.mul(U32.to_nat(images),U32.to_nat(height))
  +image_room = {
    %Order.commutative(U32.to_nat(height),base) : {Nat.is_le(_,total) == True{} : Bool}
    RowMajor.mul_le(1n+U32.to_nat(native_image),U32.to_nat(images),U32.to_nat(height),Order.strict_follow(U32.to_nat(native_image),U32.to_nat(images),native_image_inside)) : {Nat.is_le(Nat.add(base,U32.to_nat(height)),total) == True{} : Bool}}
  +start_room = {
    %Equal.sym(Nat,U32.to_nat(start_full),Nat.add(base,U32.to_nat(start)),se) : {Nat.is_le(_,U32.to_nat(Machine.capacity(Scalar.limits(machine)))) == True{} : Bool}
    Order.transitive(Nat.add(base,U32.to_nat(start)),Nat.add(base,U32.to_nat(height)),U32.to_nat(Machine.capacity(Scalar.limits(machine))),Order.offset_le(base,U32.to_nat(start),U32.to_nat(height),start_inside),
      Order.transitive(Nat.add(base,U32.to_nat(height)),total,U32.to_nat(Machine.capacity(Scalar.limits(machine))),image_room,BandGather.rows_room(BandWindow.input_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),domain))) : {Nat.is_le(U32.to_nat(start_full),U32.to_nat(Machine.capacity(Scalar.limits(machine)))) == True{} : Bool}}
  +end_room = {
    %Equal.sym(Nat,U32.to_nat(end_full),Nat.add(base,U32.to_nat(end)),ee) : {Nat.is_le(_,U32.to_nat(Machine.capacity(Scalar.limits(machine)))) == True{} : Bool}
    Order.transitive(Nat.add(base,U32.to_nat(end)),Nat.add(base,U32.to_nat(height)),U32.to_nat(Machine.capacity(Scalar.limits(machine))),Order.offset_le(base,U32.to_nat(end),U32.to_nat(height),end_inside),
      Order.transitive(Nat.add(base,U32.to_nat(height)),total,U32.to_nat(Machine.capacity(Scalar.limits(machine))),image_room,BandGather.rows_room(BandWindow.input_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),domain))) : {Nat.is_le(U32.to_nat(end_full),U32.to_nat(Machine.capacity(Scalar.limits(machine)))) == True{} : Bool}}
  +active = {
    %Equal.sym(Bool,U32.is_lt(TensorBandWindow.output_first(height,output_height,stride,first),TensorBandWindow.output_end(height,output_height,stride,first,rows)),
      Nat.is_lt(U32.to_nat(TensorBandWindow.output_first(height,output_height,stride,first)),U32.to_nat(TensorBandWindow.output_end(height,output_height,stride,first,rows))),Comparison.lt(TensorBandWindow.output_first(height,output_height,stride,first),TensorBandWindow.output_end(height,output_height,stride,first,rows))) : {_ == True{} : Bool}
    before : {U32.is_lt(TensorBandWindow.output_first(height,output_height,stride,first),TensorBandWindow.output_end(height,output_height,stride,first,rows)) == True{} : Bool}}
  +window_first_exact = Equal.trans(Nat,U32.to_nat(BandPoolWindowValues.window_first(BandWindow.window(geometry,first,rows))),U32.to_nat(start_full),Nat.add(Nat.mul(core_image,U32.to_nat(height)),U32.to_nat(start)),
    Equal.cong(U32,Nat,value => U32.to_nat(value),BandPoolWindowValues.window_first(BandWindow.window(geometry,first,rows)),start_full,input_first_native(geometry,first,rows)),
    Equal.trans(Nat,U32.to_nat(start_full),Nat.add(base,U32.to_nat(start)),Nat.add(Nat.mul(core_image,U32.to_nat(height)),U32.to_nat(start)),se,
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,U32.to_nat(height)),U32.to_nat(start)),U32.to_nat(native_image),core_image,native_image_exact)))
  +window_rows_exact = Equal.trans(Nat,U32.to_nat(BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(BandWindow.subtract_clipped(end_full,start_full)),Nat.sub(U32.to_nat(end),U32.to_nat(start)),
    Equal.cong(U32,Nat,value => U32.to_nat(value),BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows)),BandWindow.subtract_clipped(end_full,start_full),input_rows_active(geometry,first,rows,active)),
    Equal.trans(Nat,U32.to_nat(BandWindow.subtract_clipped(end_full,start_full)),Nat.sub(U32.to_nat(end_full),U32.to_nat(start_full)),Nat.sub(U32.to_nat(end),U32.to_nat(start)),
      TensorBandWindow.subtract_clipped(end_full,start_full,Machine.capacity(Scalar.limits(machine)),Limits.capacity_fits(Scalar.limits(machine)),end_room,start_room),
      %Equal.sym(Nat,U32.to_nat(end_full),Nat.add(base,U32.to_nat(end)),ee) : {Nat.sub(_,U32.to_nat(start_full)) == Nat.sub(U32.to_nat(end),U32.to_nat(start)) : Nat}
      %Equal.sym(Nat,U32.to_nat(start_full),Nat.add(base,U32.to_nat(start)),se) : {Nat.sub(Nat.add(base,U32.to_nat(end)),_) == Nat.sub(U32.to_nat(end),U32.to_nat(start)) : Nat}
      TensorBandWindow.common_offset(base,U32.to_nat(end),U32.to_nat(start))))
  WindowRange{window_first_exact,window_rows_exact,
    %Equal.sym(U32,BandPoolWindowValues.window_first(BandWindow.window(geometry,first,rows)),start_full,input_first_native(geometry,first,rows)) : BandDense.LeafFacts<BandWindow.input_geometry(geometry),_,BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows)),Machine.capacity(Scalar.limits(machine)),Machine.half_word(Scalar.limits(machine))>
    %Equal.sym(U32,BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows)),BandWindow.subtract_clipped(end_full,start_full),input_rows_active(geometry,first,rows,active)) : BandDense.LeafFacts<BandWindow.input_geometry(geometry),start_full,_,Machine.capacity(Scalar.limits(machine)),Machine.half_word(Scalar.limits(machine))>
    TensorBandWindow.image_range(images,channels,height,width,native_image,start,end,Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),domain,native_image_inside,start_inside,end_inside)}

def padding_room(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(BandPoolWindowValues.pool_padding(OutputDomain.pool(geometry))),U32.to_nat(Machine.half_word(Scalar.limits(machine)))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,+padding,output_height,output_width} = geometry
  Order.transitive(U32.to_nat(BandPoolWindowValues.pool_padding(OutputDomain.pool(geometry))),U32.to_nat(Machine.capacity(Scalar.limits(machine))),U32.to_nat(Machine.half_word(Scalar.limits(machine))),
    Scalar.small_room(machine,BandPoolWindowValues.pool_padding(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_condition(OutputDomain.pool(geometry),6n,OutputDomain.selected_legal(shape,geometry,selected))),
    Bounds.as_nat_le(Machine.capacity(Scalar.limits(machine)),Machine.half_word(Scalar.limits(machine)),Limits.capacity_within_half(Scalar.limits(machine))))

def extent_padding_room(+machine: Scalar.Observation,+extent: U32,+padding: U32,
  extent_small: {Operations.small(extent) == True{} : Bool},padding_small: {Operations.small(padding) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(padding)),U32.to_nat(Machine.half_word(Scalar.limits(machine)))) == True{} : Bool}:
  +limits = Scalar.limits(machine)
  +capacity = Machine.capacity(limits)
  +half = Machine.half_word(limits)
  +extent_room = Scalar.small_room(machine,extent,extent_small)
  +padding_room = Scalar.small_room(machine,padding,padding_small)
  Order.transitive(Nat.add(U32.to_nat(extent),U32.to_nat(padding)),Nat.double(U32.to_nat(capacity)),U32.to_nat(half),
    %Addition.double(U32.to_nat(capacity)) : {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(padding)),_) == True{} : Bool}
    Order.transitive(Nat.add(U32.to_nat(extent),U32.to_nat(padding)),Nat.add(U32.to_nat(capacity),U32.to_nat(padding)),Nat.add(U32.to_nat(capacity),U32.to_nat(capacity)),
      Order.add_le(U32.to_nat(extent),U32.to_nat(capacity),U32.to_nat(padding),extent_room),
      Order.offset_le(U32.to_nat(capacity),U32.to_nat(padding),U32.to_nat(capacity),padding_room)),
    Limits.double_bound(limits,U32.to_nat(capacity),Order.reflexive(U32.to_nat(capacity))))

def encoded_next(+position: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(1n+position,U32.to_nat(capacity)) == True{} : Bool}) ->
  {(U32.from_nat(position) + 1 : U32) == U32.from_nat(1n+position) : U32}:
  +current_exact = Bounds.roundtrip(position,capacity,capacity_ok,Order.transitive(position,1n+position,U32.to_nat(capacity),Order.successor(position,position,Order.reflexive(position)),room))
  +sum_exact = Equal.trans(Nat,U32.to_nat((U32.from_nat(position) + 1 : U32)),Nat.add(position,1n),1n+position,
    Equal.trans(Nat,U32.to_nat((U32.from_nat(position) + 1 : U32)),Nat.add(U32.to_nat(U32.from_nat(position)),1n),Nat.add(position,1n),
      Arithmetic.add(U32.from_nat(position),1,capacity,capacity_ok,
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,current_exact) : {Nat.is_le(Nat.add(_,1n),U32.to_nat(capacity)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(position,1n),1n+position,Order.commutative(position,1n)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} room),
      Equal.cong(Nat,Nat,value => Nat.add(value,1n),U32.to_nat(U32.from_nat(position)),position,current_exact)),Order.commutative(position,1n))
  Equal.trans(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(U32.to_nat((U32.from_nat(position) + 1 : U32))),U32.from_nat(1n+position),
    Equal.sym(U32,U32.from_nat(U32.to_nat((U32.from_nat(position) + 1 : U32))),(U32.from_nat(position) + 1 : U32),Arithmetic.native_roundtrip((U32.from_nat(position) + 1 : U32))),
    Equal.cong(Nat,U32,U32.from_nat,U32.to_nat((U32.from_nat(position) + 1 : U32)),1n+position,sum_exact))

def kernel_room(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(BandPoolWindowValues.pool_kernel(OutputDomain.pool(geometry))),U32.to_nat(Machine.capacity(Scalar.limits(machine)))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},+kernel,stride,padding,output_height,output_width} = geometry
  Scalar.small_room(machine,BandPoolWindowValues.pool_kernel(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_condition(OutputDomain.pool(geometry),8n,OutputDomain.selected_legal(shape,geometry,selected)))

def stored_rows_bound(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},range: WindowRange<machine,geometry,first,rows>) ->
  {Nat.is_le(U32.to_nat(BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},kernel,stride,padding,output_height,output_width} = geometry
  WindowRange{first_exact,rows_exact,BandDense.LeafFacts{allocation,contents,BandShape.Fields{local,rows_fit,length_exact}}} = range
  +window_first = BandPoolWindowValues.window_first(BandWindow.window(geometry,first,rows))
  +window_rows = BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows))
  Strided.count_in_stride(Nat.mod(U32.to_nat(window_first),U32.to_nat(height)),U32.to_nat(window_rows),U32.to_nat(height),
    Inspection.local_span(height,window_first,window_rows,Machine.half_word(Scalar.limits(machine)),Limits.half_word_fits(Scalar.limits(machine)),OutputDomain.selected_height(shape,geometry,selected),
      Limits.double_bound(Scalar.limits(machine),U32.to_nat(height),OutputDomain.selected_height_room(machine,shape,geometry,selected)),local))

def words_equal(+left: U32,+right: U32,exact: {U32.to_nat(left) == U32.to_nat(right) : Nat}) -> {left == right : U32}:
  Equal.trans(U32,left,U32.from_nat(U32.to_nat(left)),right,
    Equal.sym(U32,U32.from_nat(U32.to_nat(left)),left,Arithmetic.native_roundtrip(left)),
    Equal.trans(U32,U32.from_nat(U32.to_nat(left)),U32.from_nat(U32.to_nat(right)),right,Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(left),U32.to_nat(right),exact),Arithmetic.native_roundtrip(right)))

def native_zero(+value: U32,zero: {U32.is_zero(value) == True{} : Bool}) -> {value == 0 : U32}:
  words_equal(value,0,Order.equal_of_is_eq(U32.to_nat(value),0n,
    Equal.trans(Bool,Nat.is_eq(U32.to_nat(value),0n),U32.is_zero(value),True{},Equal.sym(Bool,U32.is_zero(value),Nat.is_eq(U32.to_nat(value),0n),Comparison.eq(value,0)),zero)))

# The rows a window leaf reads. For every output row of the leaf and every
# tap, the row of the assembled window is valid exactly when the row of the
# image is, and then it is that row. Nothing here names a flat position.
def output_begin(+geometry: BandWindow.Geometry,+first: U32) -> U32:
  TensorBandWindow.output_first(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first)

def output_finish(+geometry: BandWindow.Geometry,+first: U32,+rows: U32) -> U32:
  TensorBandWindow.output_end(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),first,rows)

def TapAt(+geometry: BandWindow.Geometry,+first: U32,+window: BandWindow.RowWindow,+image: Nat,+local: U32,+global: U32,+tap: U32) -> Data:
  BandSweeps.RowTap(BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_padding(OutputDomain.pool(geometry)),BandPoolExecution.window_origin(geometry,first,window),BandPoolWindowValues.window_first(window),BandPoolWindowValues.window_rows(window),BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),image,local,global,tap)

def TapsAt(+geometry: BandWindow.Geometry,+first: U32,+window: BandWindow.RowWindow,+image: Nat,+local: U32,+global: U32,remaining: Nat,+tap: U32) -> Data:
  BandSweeps.RowTaps(BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_padding(OutputDomain.pool(geometry)),BandPoolExecution.window_origin(geometry,first,window),BandPoolWindowValues.window_first(window),BandPoolWindowValues.window_rows(window),BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),image,local,global,remaining,tap)

def RowAt(+geometry: BandWindow.Geometry,+first: U32,+window: BandWindow.RowWindow,+row: Nat) -> Data:
  BandSweeps.RowTapsAt(BandPoolWindowValues.pool_kernel(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_padding(OutputDomain.pool(geometry)),BandPoolExecution.window_origin(geometry,first,window),BandPoolWindowValues.window_first(window),BandPoolWindowValues.window_rows(window),BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.output_first(window),row)

def RowsOf(+geometry: BandWindow.Geometry,+first: U32,+window: BandWindow.RowWindow) -> Data:
  BandSweeps.RowFacts(BandPoolWindowValues.pool_kernel(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_padding(OutputDomain.pool(geometry)),BandPoolExecution.window_origin(geometry,first,window),BandPoolWindowValues.window_first(window),BandPoolWindowValues.window_rows(window),BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.output_first(window),BandPoolWindowValues.output_rows(window))

def RoomsOf(+machine: Scalar.Observation,+geometry: BandWindow.Geometry,+window: BandWindow.RowWindow) -> Data:
  BandSweeps.Rooms<BandPoolExecution.images(geometry),OutputDomain.channels(geometry),BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_width(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),OutputDomain.output_width(geometry),
    BandPoolWindowValues.window_first(window),BandPoolWindowValues.window_rows(window),BandPoolWindowValues.output_first(window),BandPoolWindowValues.output_rows(window),Scalar.limits(machine)>

def interval_first(+geometry: BandWindow.Geometry,+first: U32,+rows: U32,interval: OutputInterval<geometry,first,rows>) ->
  {U32.to_nat(BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows))) == Nat.add(Nat.mul(Nat.div(U32.to_nat(first),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))),U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry))),U32.to_nat(output_begin(geometry,first))) : Nat}:
  OutputInterval{first_exact,rows_exact,before,source_image,source_row_origin} = interval
  first_exact

def window_range_first(+machine: Scalar.Observation,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,range: WindowRange<machine,geometry,first,rows>) ->
  {U32.to_nat(BandPoolWindowValues.window_first(BandWindow.window(geometry,first,rows))) == Nat.add(Nat.mul(Nat.div(U32.to_nat(first),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))),U32.to_nat(needed_first(geometry,first))) : Nat}:
  WindowRange{first_exact,rows_exact,target} = range
  first_exact

def window_range_rows(+machine: Scalar.Observation,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,range: WindowRange<machine,geometry,first,rows>) ->
  {U32.to_nat(BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows))) == Nat.sub(U32.to_nat(needed_end(geometry,first,rows)),U32.to_nat(needed_first(geometry,first))) : Nat}:
  WindowRange{first_exact,rows_exact,target} = range
  rows_exact

def leaf_fit(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,metadata: BandShape.Leaf(geometry,first,rows,length)) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(BandPoolWindowValues.geometry_images(geometry)),U32.to_nat(BandPoolWindowValues.geometry_height(geometry)))) == True{} : Bool}:
  Bands.Geometry{images,channels,height,width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  rows_fit

def window_range_fit(+machine: Scalar.Observation,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,range: WindowRange<machine,geometry,first,rows>) ->
  {Nat.is_le(Nat.add(U32.to_nat(BandPoolWindowValues.window_first(BandWindow.window(geometry,first,rows))),U32.to_nat(BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows)))),
    Nat.mul(U32.to_nat(BandPoolExecution.images(geometry)),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry))))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  WindowRange{first_exact,rows_exact,BandDense.LeafFacts{allocation,contents,BandShape.Fields{local,rows_fit,length_exact}}} = range
  rows_fit

def finish_below(+geometry: BandWindow.Geometry,+first: U32,+rows: U32) ->
  {Nat.is_le(U32.to_nat(output_finish(geometry,first,rows)),U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},kernel,+stride,padding,+output_height,output_width} = geometry
  TensorBandWindow.end_below(output_height,(first % height + rows : U32),stride,U32.is_eq((first % height + rows : U32),height))

# An output row of the leaf is a row of the output interval of its image.
def row_below(+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+row: Nat,interval: OutputInterval<geometry,first,rows>,
  row_inside: {Nat.is_lt(row,U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(output_begin(geometry,first)),row),U32.to_nat(output_finish(geometry,first,rows))) == True{} : Bool}:
  OutputInterval{first_exact,+rows_exact,+before,source_image,source_row_origin} = interval
  +begin = U32.to_nat(output_begin(geometry,first))
  +finish = U32.to_nat(output_finish(geometry,first,rows))
  +out_rows = BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))
  +last = Equal.trans(Nat,Nat.add(begin,U32.to_nat(out_rows)),Nat.add(begin,Nat.sub(finish,begin)),finish,
    Equal.cong(Nat,Nat,value => Nat.add(begin,value),U32.to_nat(out_rows),Nat.sub(finish,begin),rows_exact),Order.restore_shift(begin,finish,Bounds.strict_le(begin,finish,before)))
  %last : {Nat.is_lt(Nat.add(begin,row),_) == True{} : Bool}
  Order.offset_lt(begin,row,U32.to_nat(out_rows),row_inside)

# start + (end - start) is start or end, so it stays below a bound of both.
def lift_clipped(+start: Nat,+end: Nat,cases: Either<&2,&2,{Nat.add(start,Nat.sub(end,start)) == start : Nat},{Nat.add(start,Nat.sub(end,start)) == end : Nat}>) ->
  Either<&2,&2,{Nat.add(1n+start,Nat.sub(1n+end,1n+start)) == 1n+start : Nat},{Nat.add(1n+start,Nat.sub(1n+end,1n+start)) == 1n+end : Nat}>:
  match cases:
    case Inl{same}: Inl{Equal.cong(Nat,Nat,value => 1n+value,Nat.add(start,Nat.sub(end,start)),start,same)}
    case Inr{same}: Inr{Equal.cong(Nat,Nat,value => 1n+value,Nat.add(start,Nat.sub(end,start)),end,same)}

def clipped_end(+start: Nat,+end: Nat) -> Either<&2,&2,{Nat.add(start,Nat.sub(end,start)) == start : Nat},{Nat.add(start,Nat.sub(end,start)) == end : Nat}>:
  match start end:
    case 0n 0n: Inr{{==}}
    case 0n 1n+end: Inr{{==}}
    case 1n+ +start 0n: Inl{Equal.cong(Nat,Nat,value => 1n+value,Nat.add(start,0n),start,Addition.zero_right(start))}
    case 1n+ +start 1n+ +end: lift_clipped(start,end,clipped_end(start,end))

def clipped_span_cases(+start: Nat,+end: Nat,+bound: Nat,cases: Either<&2,&2,{Nat.add(start,Nat.sub(end,start)) == start : Nat},{Nat.add(start,Nat.sub(end,start)) == end : Nat}>,
  start_inside: {Nat.is_le(start,bound) == True{} : Bool},end_inside: {Nat.is_le(end,bound) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(start,Nat.sub(end,start)),bound) == True{} : Bool}:
  match cases:
    case Inl{same}:
      %Equal.sym(Nat,Nat.add(start,Nat.sub(end,start)),start,same) : {Nat.is_le(_,bound) == True{} : Bool}
      start_inside
    case Inr{same}:
      %Equal.sym(Nat,Nat.add(start,Nat.sub(end,start)),end,same) : {Nat.is_le(_,bound) == True{} : Bool}
      end_inside

def clipped_span(+start: Nat,+end: Nat,+bound: Nat,start_inside: {Nat.is_le(start,bound) == True{} : Bool},end_inside: {Nat.is_le(end,bound) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(start,Nat.sub(end,start)),bound) == True{} : Bool}:
  clipped_span_cases(start,end,bound,clipped_end(start,end),start_inside,end_inside)

# A point of [start, end) is a point of [start, start + (end - start)).
def contained_span(+start: Nat,+end: Nat,+rows: Nat,+point: Nat,+rows_exact: {rows == Nat.sub(end,start) : Nat},evidence: TensorBandWindow.Contains(start,end,point)) ->
  TensorBandWindow.Contains(start,Nat.add(start,rows),point):
  Shape.Both{+lower,+upper} = evidence
  +ordered = Bounds.strict_le(start,end,Order.le_lt_trans(start,point,end,lower,upper))
  Shape.Both{lower,
    %Equal.sym(Nat,rows,Nat.sub(end,start),rows_exact) : {Nat.is_lt(point,Nat.add(start,_)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(start,Nat.sub(end,start)),end,Order.restore_shift(start,end,ordered)) : {Nat.is_lt(point,_) == True{} : Bool}
    upper}

def coverage_when(+start: Nat,+end: Nat,+rows: Nat,+point: Nat,valid: Bool,+rows_exact: {rows == Nat.sub(end,start) : Nat},
  evidence: Boolean.When(valid,TensorBandWindow.Contains(start,end,point))) -> Boolean.When(valid,TensorBandWindow.Contains(start,Nat.add(start,rows),point)):
  match valid:
    case False{}: Unit{}
    case True{}: contained_span(start,end,rows,point,rows_exact,evidence)

# The native coordinate of a tap: row * stride + origin + tap, without wrap.
def coordinate_decoded(+row: Nat,+stride: U32,+origin: U32,+tap: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +row_room: {Nat.is_le(row,U32.to_nat(capacity)) == True{} : Bool},+tap_room: {Nat.is_le(tap,U32.to_nat(capacity)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.add(Nat.mul(row,U32.to_nat(stride)),U32.to_nat(origin)),tap),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((U32.from_nat(row) * stride + origin + U32.from_nat(tap) : U32)) == Nat.add(Nat.add(Nat.mul(row,U32.to_nat(stride)),U32.to_nat(origin)),tap) : Nat}:
  +native_row = U32.from_nat(row)
  +native_tap = U32.from_nat(tap)
  +row_exact = Bounds.roundtrip(row,capacity,capacity_ok,row_room)
  +tap_exact = Bounds.roundtrip(tap,capacity,capacity_ok,tap_room)
  +product = Nat.mul(row,U32.to_nat(stride))
  +based = Nat.add(product,U32.to_nat(origin))
  +total = Nat.add(based,tap)
  +based_room = Order.transitive(based,total,U32.to_nat(capacity),Order.le_add(based,tap),room)
  +product_room = Order.transitive(product,based,U32.to_nat(capacity),Order.le_add(product,U32.to_nat(origin)),based_room)
  +product_exact = Equal.trans(Nat,U32.to_nat((native_row * stride : U32)),Nat.mul(U32.to_nat(native_row),U32.to_nat(stride)),product,
    Arithmetic.multiply(native_row,stride,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(native_row),row,row_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool} product_room),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(stride)),U32.to_nat(native_row),row,row_exact))
  +based_exact = Equal.trans(Nat,U32.to_nat((native_row * stride + origin : U32)),Nat.add(U32.to_nat((native_row * stride : U32)),U32.to_nat(origin)),based,
    Arithmetic.add((native_row * stride : U32),origin,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat((native_row * stride : U32)),product,product_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(origin)),U32.to_nat(capacity)) == True{} : Bool} based_room),
    Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(origin)),U32.to_nat((native_row * stride : U32)),product,product_exact))
  Equal.trans(Nat,U32.to_nat((native_row * stride + origin + native_tap : U32)),Nat.add(U32.to_nat((native_row * stride + origin : U32)),U32.to_nat(native_tap)),total,
    Arithmetic.add((native_row * stride + origin : U32),native_tap,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat((native_row * stride + origin : U32)),based,based_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(native_tap)),U32.to_nat(capacity)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(native_tap),tap,tap_exact) : {Nat.is_le(Nat.add(based,_),U32.to_nat(capacity)) == True{} : Bool}
      room),
    Equal.trans(Nat,Nat.add(U32.to_nat((native_row * stride + origin : U32)),U32.to_nat(native_tap)),Nat.add(based,U32.to_nat(native_tap)),total,
      Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(native_tap)),U32.to_nat((native_row * stride + origin : U32)),based,based_exact),
      Equal.cong(Nat,Nat,value => Nat.add(based,value),U32.to_nat(native_tap),tap,tap_exact)))

# A valid row of the image is the row of the assembled window at the same
# distance from the first stored row.
def row_located(+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+height: U32,+image: Nat,+local: U32,+global: U32,+tap: U32,+start: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +local_room: {Nat.is_le(U32.to_nat((local * stride + origin + tap : U32)),U32.to_nat(capacity)) == True{} : Bool},
  +global_room: {Nat.is_le(U32.to_nat((global * stride + 0 + tap : U32)),U32.to_nat(capacity)) == True{} : Bool},
  rows_padding_room: {Nat.is_le(Nat.add(U32.to_nat(window_rows),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  height_padding_room: {Nat.is_le(Nat.add(U32.to_nat(height),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  +shifted: {U32.to_nat((global * stride + 0 + tap : U32)) == Nat.add(start,U32.to_nat((local * stride + origin + tap : U32))) : Nat},
  +first_exact: {U32.to_nat(window_first) == Nat.add(Nat.mul(image,U32.to_nat(height)),start) : Nat},
  +same: {U32.is_lt((local * stride + origin + tap - padding : U32),window_rows) == U32.is_lt((global * stride + 0 + tap - padding : U32),height) : Bool},
  valid: Bool,+valid_exact: {U32.is_lt((global * stride + 0 + tap - padding : U32),height) == valid : Bool}) ->
  Boolean.When(valid,{Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat((global * stride + 0 + tap - padding : U32))) == Nat.add(U32.to_nat(window_first),U32.to_nat((local * stride + origin + tap - padding : U32))) : Nat}):
  match valid:
    case False{}: Unit{}
    case True{}:
      +local_coordinate = (local * stride + origin + tap : U32)
      +global_coordinate = (global * stride + 0 + tap : U32)
      +global_axis = BandPoolWindowValues.axis_accept(global_coordinate,padding,height,capacity,capacity_ok,global_room,height_padding_room,valid_exact)
      +local_axis = BandPoolWindowValues.axis_accept(local_coordinate,padding,window_rows,capacity,capacity_ok,local_room,rows_padding_room,
        Equal.trans(Bool,U32.is_lt((local_coordinate - padding : U32),window_rows),U32.is_lt((global_coordinate - padding : U32),height),True{},same,valid_exact))
      +global_point = Nat.sub(U32.to_nat(global_coordinate),U32.to_nat(padding))
      +local_point = Nat.sub(U32.to_nat(local_coordinate),U32.to_nat(padding))
      +global_decoded = BandPoolWindowValues.axis_exact(global_coordinate,padding,height,global_axis)
      +local_decoded = BandPoolWindowValues.axis_exact(local_coordinate,padding,window_rows,local_axis)
      +difference = BandPoolWindowValues.shifted_difference(start,U32.to_nat(local_coordinate),U32.to_nat(padding),U32.to_nat(global_coordinate),shifted,
        BandPoolWindowValues.axis_ordered(local_coordinate,padding,window_rows,local_axis))
      +base = Nat.mul(image,U32.to_nat(height))
      Equal.trans(Nat,Nat.add(base,U32.to_nat((global_coordinate - padding : U32))),Nat.add(base,Nat.add(start,local_point)),Nat.add(U32.to_nat(window_first),U32.to_nat((local_coordinate - padding : U32))),
        Equal.cong(Nat,Nat,value => Nat.add(base,value),U32.to_nat((global_coordinate - padding : U32)),Nat.add(start,local_point),
          Equal.trans(Nat,U32.to_nat((global_coordinate - padding : U32)),global_point,Nat.add(start,local_point),global_decoded,difference)),
        Equal.trans(Nat,Nat.add(base,Nat.add(start,local_point)),Nat.add(Nat.add(base,start),local_point),Nat.add(U32.to_nat(window_first),U32.to_nat((local_coordinate - padding : U32))),
          Equal.sym(Nat,Nat.add(Nat.add(base,start),local_point),Nat.add(base,Nat.add(start,local_point)),Addition.associative(base,start,local_point)),
          Equal.trans(Nat,Nat.add(Nat.add(base,start),local_point),Nat.add(U32.to_nat(window_first),local_point),Nat.add(U32.to_nat(window_first),U32.to_nat((local_coordinate - padding : U32))),
            Equal.cong(Nat,Nat,value => Nat.add(value,local_point),Nat.add(base,start),U32.to_nat(window_first),Equal.sym(Nat,U32.to_nat(window_first),Nat.add(base,start),first_exact)),
            Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(window_first),value),local_point,U32.to_nat((local_coordinate - padding : U32)),
              Equal.sym(Nat,U32.to_nat((local_coordinate - padding : U32)),local_point,local_decoded)))))

# One tap of one output row, from the scalar facts of the two coordinates:
# the global coordinate is the local one shifted by the first stored row of
# the image, and the stored rows cover every valid row the tap can name.
def shifted_row_tap(+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+height: U32,+image: Nat,+local: U32,+global: U32,+tap: U32,+start: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +local_room: {Nat.is_le(U32.to_nat((local * stride + origin + tap : U32)),U32.to_nat(capacity)) == True{} : Bool},
  +global_room: {Nat.is_le(U32.to_nat((global * stride + 0 + tap : U32)),U32.to_nat(capacity)) == True{} : Bool},
  +rows_padding_room: {Nat.is_le(Nat.add(U32.to_nat(window_rows),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  +height_padding_room: {Nat.is_le(Nat.add(U32.to_nat(height),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  +shifted: {U32.to_nat((global * stride + 0 + tap : U32)) == Nat.add(start,U32.to_nat((local * stride + origin + tap : U32))) : Nat},
  span: {Nat.is_le(Nat.add(start,U32.to_nat(window_rows)),U32.to_nat(height)) == True{} : Bool},
  coverage: Boolean.When(BandPoolWindowValues.natural_valid(U32.to_nat((global * stride + 0 + tap : U32)),U32.to_nat(padding),U32.to_nat(height)),
    TensorBandWindow.Contains(start,Nat.add(start,U32.to_nat(window_rows)),Nat.sub(U32.to_nat((global * stride + 0 + tap : U32)),U32.to_nat(padding)))),
  +first_exact: {U32.to_nat(window_first) == Nat.add(Nat.mul(image,U32.to_nat(height)),start) : Nat}) ->
  BandSweeps.RowTap(stride,padding,origin,window_first,window_rows,height,image,local,global,tap):
  +same = BandPoolWindowValues.native_rows_same(start,(local * stride + origin + tap : U32),padding,(global * stride + 0 + tap : U32),window_rows,height,capacity,capacity_ok,
    local_room,global_room,rows_padding_room,height_padding_room,shifted,span,coverage)
  Shape.Both{same,row_located(stride,padding,origin,window_first,window_rows,height,image,local,global,tap,start,capacity,capacity_ok,local_room,global_room,rows_padding_room,height_padding_room,shifted,first_exact,same,
    U32.is_lt((global * stride + 0 + tap - padding : U32),height),{==})}

# The tap of the actual caller. The selected pool guard bounds every scalar;
# the output interval and the stored range of the window place the row.
def row_tap(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,+row: Nat,+tap: Nat,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +interval: OutputInterval<geometry,first,rows>,+range: WindowRange<machine,geometry,first,rows>,
  +row_inside: {Nat.is_lt(row,U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(BandPoolWindowValues.pool_kernel(OutputDomain.pool(geometry)))) == True{} : Bool}) ->
  TapAt(geometry,first,BandWindow.window(geometry,first,rows),Nat.div(U32.to_nat(first),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))),U32.from_nat(row),U32.from_nat(Nat.add(U32.to_nat(output_begin(geometry,first)),row)),U32.from_nat(tap)):
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},+kernel,+stride,+padding,+output_height,output_width} = geometry
  +limits = Scalar.limits(machine)
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +within_half = Bounds.as_nat_le(Machine.capacity(limits),half,Limits.capacity_within_half(limits))
  +window = BandWindow.window(geometry,first,rows)
  +origin = BandPoolExecution.window_origin(geometry,first,window)
  +window_rows = BandPoolWindowValues.window_rows(window)
  +begin_native = TensorBandWindow.output_first(height,output_height,stride,first)
  +finish_native = TensorBandWindow.output_end(height,output_height,stride,first,rows)
  +begin = U32.to_nat(begin_native)
  +finish = U32.to_nat(finish_native)
  +start = U32.to_nat(BandWindow.start(height,begin_native,stride,padding))
  +end = U32.to_nat(TensorBandWindow.samples_end(height,finish_native,stride,kernel,padding))
  +stride_nat = U32.to_nat(stride)
  +origin_nat = Nat.sub(Nat.mul(begin,stride_nat),start)
  +global_row = Nat.add(begin,row)
  +global_nat = Nat.add(Nat.mul(global_row,stride_nat),tap)
  +local_nat = Nat.add(Nat.add(Nat.mul(row,stride_nat),origin_nat),tap)
  +legal = OutputDomain.selected_legal(shape,geometry,selected)
  +finish_inside = finish_below(geometry,first,rows)
  +row_below_finish = row_below(geometry,first,rows,row,interval,row_inside)
  +row_bound = Bounds.strict_le(global_row,U32.to_nat(output_height),Bounds.nat_trans(global_row,finish,U32.to_nat(output_height),row_below_finish,finish_inside))
  +tap_bound = Bounds.strict_le(tap,U32.to_nat(kernel),tap_inside)
  +global_room = OutputDomain.row_scalar_room(machine,shape,geometry,global_row,tap,selected,row_bound,tap_bound)
  +begin_product_exact = Arithmetic.multiply(begin_native,stride,half,half_ok,OutputDomain.begin_product_room(machine,shape,geometry,first,selected))
  +start_ordered = {
    %begin_product_exact : {Nat.is_le(start,_) == True{} : Bool} BandPoolWindowValues.start_below_product(height,begin_native,stride,padding) : {Nat.is_le(start,Nat.mul(begin,stride_nat)) == True{} : Bool}}
  +shift = BandPoolWindowValues.shifted_origin(start,begin,stride_nat,row,tap,start_ordered)
  +local_room = Order.transitive(local_nat,Nat.add(start,local_nat),U32.to_nat(half),
    %Order.commutative(local_nat,start) : {Nat.is_le(local_nat,_) == True{} : Bool} Order.le_add(local_nat,start),
    %shift : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool} global_room)
  +height_room = Order.transitive(U32.to_nat(output_height),U32.to_nat(Machine.capacity(limits)),U32.to_nat(half),
    OutputDomain.output_height_room(machine,geometry,OutputDomain.positive_from_selector(shape,geometry,selected),OutputDomain.from_selector(machine,shape,geometry,count,selected,counted)),within_half)
  +global_row_room = Order.transitive(global_row,U32.to_nat(output_height),U32.to_nat(half),row_bound,height_room)
  +row_room = Order.transitive(row,global_row,U32.to_nat(half),
    %Order.commutative(row,begin) : {Nat.is_le(row,_) == True{} : Bool} Order.le_add(row,begin),global_row_room)
  +tap_room = Order.transitive(tap,U32.to_nat(kernel),U32.to_nat(half),tap_bound,
    Order.transitive(U32.to_nat(kernel),U32.to_nat(Machine.capacity(limits)),U32.to_nat(half),kernel_room(machine,shape,geometry,selected),within_half))
  +origin_exact = OutputDomain.caller_origin(machine,shape,geometry,first,rows,selected)
  +local_coordinate = (U32.from_nat(row) * stride + origin + U32.from_nat(tap) : U32)
  +global_coordinate = (U32.from_nat(global_row) * stride + 0 + U32.from_nat(tap) : U32)
  +local_exact = Equal.trans(Nat,U32.to_nat(local_coordinate),Nat.add(Nat.add(Nat.mul(row,stride_nat),U32.to_nat(origin)),tap),local_nat,
    coordinate_decoded(row,stride,origin,tap,half,half_ok,row_room,tap_room,
      %Equal.sym(Nat,U32.to_nat(origin),origin_nat,origin_exact) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(row,stride_nat),_),tap),U32.to_nat(half)) == True{} : Bool} local_room),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.add(Nat.mul(row,stride_nat),value),tap),U32.to_nat(origin),origin_nat,origin_exact))
  +global_exact = Equal.trans(Nat,U32.to_nat(global_coordinate),Nat.add(Nat.add(Nat.mul(global_row,stride_nat),0n),tap),global_nat,
    coordinate_decoded(global_row,stride,0,tap,half,half_ok,global_row_room,tap_room,
      %Equal.sym(Nat,Nat.add(Nat.mul(global_row,stride_nat),0n),Nat.mul(global_row,stride_nat),Addition.zero_right(Nat.mul(global_row,stride_nat))) : {Nat.is_le(Nat.add(_,tap),U32.to_nat(half)) == True{} : Bool} global_room),
    Equal.cong(Nat,Nat,value => Nat.add(value,tap),Nat.add(Nat.mul(global_row,stride_nat),0n),Nat.mul(global_row,stride_nat),Addition.zero_right(Nat.mul(global_row,stride_nat))))
  +shifted = Equal.trans(Nat,U32.to_nat(global_coordinate),global_nat,Nat.add(start,U32.to_nat(local_coordinate)),global_exact,
    Equal.trans(Nat,global_nat,Nat.add(start,local_nat),Nat.add(start,U32.to_nat(local_coordinate)),shift,
      Equal.cong(Nat,Nat,value => Nat.add(start,value),local_nat,U32.to_nat(local_coordinate),Equal.sym(Nat,U32.to_nat(local_coordinate),local_nat,local_exact))))
  +local_native_room = {
    %Equal.sym(Nat,U32.to_nat(local_coordinate),local_nat,local_exact) : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool} local_room : {Nat.is_le(U32.to_nat(local_coordinate),U32.to_nat(half)) == True{} : Bool}}
  +global_native_room = {
    %Equal.sym(Nat,U32.to_nat(global_coordinate),global_nat,global_exact) : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool} global_room : {Nat.is_le(U32.to_nat(global_coordinate),U32.to_nat(half)) == True{} : Bool}}
  +height_padding = extent_padding_room(machine,height,padding,BandPoolWindowValues.pool_condition(OutputDomain.pool(geometry),10n,legal),BandPoolWindowValues.pool_condition(OutputDomain.pool(geometry),6n,legal))
  +rows_padding = Order.transitive(Nat.add(U32.to_nat(window_rows),U32.to_nat(padding)),Nat.add(U32.to_nat(height),U32.to_nat(padding)),U32.to_nat(half),
    Order.add_le(U32.to_nat(window_rows),U32.to_nat(height),U32.to_nat(padding),stored_rows_bound(machine,shape,geometry,first,rows,selected,range)),height_padding)
  +rows_exact = window_range_rows(machine,geometry,first,rows,range)
  +span = {
    %Equal.sym(Nat,U32.to_nat(window_rows),Nat.sub(end,start),rows_exact) : {Nat.is_le(Nat.add(start,_),U32.to_nat(height)) == True{} : Bool}
    clipped_span(start,end,U32.to_nat(height),BandPoolWindowValues.start_below_height(height,begin_native,stride,padding),
      TensorBandWindow.minimum_below(height,BandWindow.subtract_clipped((BandWindow.subtract_clipped(finish_native,1) * stride + kernel : U32),padding))) : {Nat.is_le(Nat.add(start,U32.to_nat(window_rows)),U32.to_nat(height)) == True{} : Bool}}
  +sample = BandPoolWindowValues.clipped_sample_when(height,begin_native,finish_native,stride,kernel,padding,global_row,tap,half,half_ok,
    Order.transitive(finish,U32.to_nat(output_height),U32.to_nat(half),finish_inside,height_room),
    OutputDomain.end_coordinate_room(machine,shape,geometry,first,rows,selected),padding_room(machine,shape,geometry,selected),
    Shape.Both{Order.le_add(begin,row),row_below_finish},tap_inside)
  +coverage = {
    %Equal.sym(Nat,U32.to_nat(global_coordinate),global_nat,global_exact) :
      Boolean.When(BandPoolWindowValues.natural_valid(_,U32.to_nat(padding),U32.to_nat(height)),TensorBandWindow.Contains(start,Nat.add(start,U32.to_nat(window_rows)),Nat.sub(_,U32.to_nat(padding))))
    coverage_when(start,end,U32.to_nat(window_rows),Nat.sub(global_nat,U32.to_nat(padding)),BandPoolWindowValues.natural_valid(global_nat,U32.to_nat(padding),U32.to_nat(height)),rows_exact,sample) :
    Boolean.When(BandPoolWindowValues.natural_valid(U32.to_nat(global_coordinate),U32.to_nat(padding),U32.to_nat(height)),
      TensorBandWindow.Contains(start,Nat.add(start,U32.to_nat(window_rows)),Nat.sub(U32.to_nat(global_coordinate),U32.to_nat(padding))))}
  shifted_row_tap(stride,padding,origin,BandPoolWindowValues.window_first(window),window_rows,height,Nat.div(U32.to_nat(first),U32.to_nat(height)),U32.from_nat(row),U32.from_nat(global_row),U32.from_nat(tap),start,half,half_ok,
    local_native_room,global_native_room,rows_padding,height_padding,shifted,span,coverage,window_range_first(machine,geometry,first,rows,range))

def row_taps(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,+row: Nat,remaining: Nat,+tap: Nat,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +interval: OutputInterval<geometry,first,rows>,+range: WindowRange<machine,geometry,first,rows>,
  +row_inside: {Nat.is_lt(row,U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(tap,remaining),U32.to_nat(BandPoolWindowValues.pool_kernel(OutputDomain.pool(geometry)))) == True{} : Bool}) ->
  TapsAt(geometry,first,BandWindow.window(geometry,first,rows),Nat.div(U32.to_nat(first),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))),U32.from_nat(row),U32.from_nat(Nat.add(U32.to_nat(output_begin(geometry,first)),row)),remaining,U32.from_nat(tap)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +kernel = U32.to_nat(BandPoolWindowValues.pool_kernel(OutputDomain.pool(geometry)))
      +inside = Order.room_index(tap,rest,kernel,room)
      +next_room = Order.transitive(1n+tap,kernel,U32.to_nat(Machine.capacity(Scalar.limits(machine))),Order.strict_follow(tap,kernel,inside),kernel_room(machine,shape,geometry,selected))
      Shape.Both{row_tap(machine,shape,geometry,count,first,rows,row,tap,selected,counted,interval,range,row_inside,inside),
        %Equal.sym(U32,(U32.from_nat(tap) + 1 : U32),U32.from_nat(1n+tap),encoded_next(tap,Machine.capacity(Scalar.limits(machine)),Limits.capacity_fits(Scalar.limits(machine)),next_room)) :
          TapsAt(geometry,first,BandWindow.window(geometry,first,rows),Nat.div(U32.to_nat(first),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))),U32.from_nat(row),U32.from_nat(Nat.add(U32.to_nat(output_begin(geometry,first)),row)),rest,_)
        row_taps(machine,shape,geometry,count,first,rows,row,rest,1n+tap,selected,counted,interval,range,row_inside,Order.advance_room(tap,rest,kernel,room))}

# The image and the row of an output row of the leaf, by the digits of its
# global row, and all the taps of that row.
def row_at(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,+row: Nat,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +interval: OutputInterval<geometry,first,rows>,+range: WindowRange<machine,geometry,first,rows>,
  +row_inside: {Nat.is_lt(row,U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)))) == True{} : Bool}) ->
  RowAt(geometry,first,BandWindow.window(geometry,first,rows),row):
  +window = BandWindow.window(geometry,first,rows)
  +output_height = U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry))
  +kernel = U32.to_nat(BandPoolWindowValues.pool_kernel(OutputDomain.pool(geometry)))
  +image = Nat.div(U32.to_nat(first),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry))))
  +begin = U32.to_nat(output_begin(geometry,first))
  +global_row = Nat.add(begin,row)
  +output_first = U32.to_nat(BandPoolWindowValues.output_first(window))
  +positive = OutputDomain.positive_height(geometry,OutputDomain.positive_from_selector(shape,geometry,selected))
  +inside = Bounds.nat_trans(global_row,U32.to_nat(output_finish(geometry,first,rows)),output_height,row_below(geometry,first,rows,row,interval,row_inside),finish_below(geometry,first,rows))
  +located = Equal.trans(Nat,Nat.add(output_first,row),Nat.add(Nat.add(Nat.mul(image,output_height),begin),row),Nat.add(Nat.mul(image,output_height),global_row),
    Equal.cong(Nat,Nat,value => Nat.add(value,row),output_first,Nat.add(Nat.mul(image,output_height),begin),interval_first(geometry,first,rows,interval)),
    Addition.associative(Nat.mul(image,output_height),begin,row))
  +image_exact = Equal.trans(Nat,Nat.div(Nat.add(output_first,row),output_height),Nat.div(Nat.add(Nat.mul(image,output_height),global_row),output_height),image,
    Equal.cong(Nat,Nat,value => Nat.div(value,output_height),Nat.add(output_first,row),Nat.add(Nat.mul(image,output_height),global_row),located),
    BandCoordinates.quotient(image,global_row,output_height,positive,inside))
  +row_exact = Equal.trans(Nat,Nat.mod(Nat.add(output_first,row),output_height),Nat.mod(Nat.add(Nat.mul(image,output_height),global_row),output_height),global_row,
    Equal.cong(Nat,Nat,value => Nat.mod(value,output_height),Nat.add(output_first,row),Nat.add(Nat.mul(image,output_height),global_row),located),
    BandCoordinates.remainder(image,global_row,output_height,positive,inside))
  %Equal.sym(Nat,Nat.div(Nat.add(output_first,row),output_height),image,image_exact) :
    TapsAt(geometry,first,window,_,U32.from_nat(row),U32.from_nat(Nat.mod(Nat.add(output_first,row),output_height)),kernel,0)
  %Equal.sym(Nat,Nat.mod(Nat.add(output_first,row),output_height),global_row,row_exact) :
    TapsAt(geometry,first,window,image,U32.from_nat(row),U32.from_nat(_),kernel,0)
  row_taps(machine,shape,geometry,count,first,rows,row,kernel,0n,selected,counted,interval,range,row_inside,Order.reflexive(kernel))

def window_rows_from(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,remaining: Nat,+start: Nat,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +interval: OutputInterval<geometry,first,rows>,+range: WindowRange<machine,geometry,first,rows>,
  +room: {Nat.is_le(Nat.add(start,remaining),U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)))) == True{} : Bool}) ->
  Members.All(Nat,{row => BandSweeps.RowTapsAt(BandPoolWindowValues.pool_kernel(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_stride(OutputDomain.pool(geometry)),BandPoolWindowValues.pool_padding(OutputDomain.pool(geometry)),BandPoolExecution.window_origin(geometry,first,BandWindow.window(geometry,first,rows)),BandPoolWindowValues.window_first(BandWindow.window(geometry,first,rows)),BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows)),BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows)),row) : Nat -> Data},
    Ranges.range(remaining,start)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +output_rows = U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)))
      Shape.Both{row_at(machine,shape,geometry,count,first,rows,start,selected,counted,interval,range,Order.room_index(start,rest,output_rows,room)),
        window_rows_from(machine,shape,geometry,count,first,rows,rest,1n+start,selected,counted,interval,range,Order.advance_room(start,rest,output_rows,room))}

def window_row_facts(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +interval: OutputInterval<geometry,first,rows>,+range: WindowRange<machine,geometry,first,rows>) -> RowsOf(geometry,first,BandWindow.window(geometry,first,rows)):
  window_rows_from(machine,shape,geometry,count,first,rows,U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))),0n,selected,counted,interval,range,
    Order.reflexive(U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)))))

# The planes a window sweeps along the columns: their fresh destination can
# be allocated and holds one cell per stored row, channel and output column.
def middle_projection(+geometry: BandWindow.Geometry,+window: BandWindow.RowWindow) ->
  {BandPoolExecution.middle_count(geometry,window) == (OutputDomain.channels(geometry) * BandPoolWindowValues.window_rows(window) * OutputDomain.output_width(geometry) : U32) : U32}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{output,above,below,BandWindow.Rows{first,rows}} = window
  {==}

def middle_cells_exact(+geometry: BandWindow.Geometry,+window: BandWindow.RowWindow) ->
  {BandPoolExecution.middle_cells(geometry,window) == Nat.mul(U32.to_nat(OutputDomain.channels(geometry)),Nat.mul(U32.to_nat(BandPoolWindowValues.window_rows(window)),U32.to_nat(OutputDomain.output_width(geometry)))) : Nat}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{output,above,below,BandWindow.Rows{first,rows}} = window
  {==}

def middle_ready_from(+machine: Scalar.Observation,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,positive: OutputDomain.Positive<geometry>,
  domain: BandGather.Domain<BandPoolExecution.planes_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>,
  rows_bound: {Nat.is_le(U32.to_nat(BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))) == True{} : Bool}) ->
  Depth.Ready<BandPoolExecution.middle_count(geometry,BandWindow.window(geometry,first,rows))>:
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,width},kernel,stride,padding,output_height,+output_width} = geometry
  OutputDomain.Positive{images_positive,channels_positive,height_positive,width_positive} = positive
  BandGather.Domain{capacity_ok,cp,hp,wp,planes_room,allocation_room} = domain
  +window = BandWindow.window(geometry,first,rows)
  +window_rows = BandPoolWindowValues.window_rows(window)
  %Equal.sym(U32,BandPoolExecution.middle_count(geometry,window),(channels * window_rows * output_width : U32),middle_projection(geometry,window)) : Depth.Ready<_>
  Depth.bounded((channels * window_rows * output_width : U32),BandPoolExecution.count_bounded(images,channels,height,output_width,window_rows,Scalar.limits(machine),images_positive,width_positive,rows_bound,planes_room))

def middle_ready(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+planes: U32,+first: U32,+rows: U32,+selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},
  planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},
  rows_bound: {Nat.is_le(U32.to_nat(BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))) == True{} : Bool}) ->
  Depth.Ready<BandPoolExecution.middle_count(geometry,BandWindow.window(geometry,first,rows))>:
  middle_ready_from(machine,geometry,first,rows,OutputDomain.positive_from_selector(shape,geometry,selected),OutputDomain.planes_domain(machine,shape,geometry,planes,selected,planes_counted),rows_bound)

def middle_cells_from(+machine: Scalar.Observation,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,positive: OutputDomain.Positive<geometry>,
  domain: BandGather.Domain<BandPoolExecution.planes_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>,
  rows_bound: {Nat.is_le(U32.to_nat(BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))) == True{} : Bool}) ->
  {U32.to_nat(BandPoolExecution.middle_count(geometry,BandWindow.window(geometry,first,rows))) == BandPoolExecution.middle_cells(geometry,BandWindow.window(geometry,first,rows)) : Nat}:
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,width},kernel,stride,padding,output_height,+output_width} = geometry
  OutputDomain.Positive{images_positive,channels_positive,height_positive,+width_positive} = positive
  BandGather.Domain{capacity_ok,cp,hp,wp,+planes_room,allocation_room} = domain
  +window = BandWindow.window(geometry,first,rows)
  +window_rows = BandPoolWindowValues.window_rows(window)
  +local = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(output_width)))
  +bounded = Order.transitive(local,BandGather.elements(BandPoolExecution.planes_geometry(geometry)),U32.to_nat(Machine.capacity(Scalar.limits(machine))),
    BandPoolExecution.local_elements(U32.to_nat(images),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(output_width),U32.to_nat(window_rows),images_positive,rows_bound),planes_room)
  Equal.trans(Nat,U32.to_nat(BandPoolExecution.middle_count(geometry,window)),U32.to_nat((channels * window_rows * output_width : U32)),BandPoolExecution.middle_cells(geometry,window),
    Equal.cong(U32,Nat,value => U32.to_nat(value),BandPoolExecution.middle_count(geometry,window),(channels * window_rows * output_width : U32),middle_projection(geometry,window)),
    Equal.trans(Nat,U32.to_nat((channels * window_rows * output_width : U32)),Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(window_rows)),U32.to_nat(output_width)),BandPoolExecution.middle_cells(geometry,window),
      Encoded.multiply_three(channels,window_rows,output_width,Machine.capacity(Scalar.limits(machine)),Limits.capacity_fits(Scalar.limits(machine)),width_positive,
        %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(window_rows)),U32.to_nat(output_width)),local,RowMajor.assoc(U32.to_nat(channels),U32.to_nat(window_rows),U32.to_nat(output_width))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(Scalar.limits(machine)))) == True{} : Bool} bounded),
      Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(window_rows)),U32.to_nat(output_width)),local,BandPoolExecution.middle_cells(geometry,window),
        RowMajor.assoc(U32.to_nat(channels),U32.to_nat(window_rows),U32.to_nat(output_width)),
        Equal.sym(Nat,BandPoolExecution.middle_cells(geometry,window),local,middle_cells_exact(geometry,window)))))

def middle_count_cells(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+planes: U32,+first: U32,+rows: U32,+selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},
  planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},
  rows_bound: {Nat.is_le(U32.to_nat(BandPoolWindowValues.window_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(BandPoolWindowValues.pool_height(OutputDomain.pool(geometry)))) == True{} : Bool}) ->
  {U32.to_nat(BandPoolExecution.middle_count(geometry,BandWindow.window(geometry,first,rows))) == BandPoolExecution.middle_cells(geometry,BandWindow.window(geometry,first,rows)) : Nat}:
  middle_cells_from(machine,geometry,first,rows,OutputDomain.positive_from_selector(shape,geometry,selected),OutputDomain.planes_domain(machine,shape,geometry,planes,selected,planes_counted),rows_bound)

# The bounds of a window leaf: the stored rows and the output rows lie in the
# images, and the input, the planes and the output fit the capacity, whole
# and per leaf.
def machine_room(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {Nat.is_le(BandGather.elements(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,room,allocation_room} = domain
  room

def positive_images(+geometry: BandWindow.Geometry,positive: OutputDomain.Positive<geometry>) -> {Nat.is_lt(0n,U32.to_nat(BandPoolExecution.images(geometry))) == True{} : Bool}:
  OutputDomain.Positive{images_positive,channels_positive,height_positive,width_positive} = positive
  images_positive

def window_rooms(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+planes: U32,+count: U32,+first: U32,+rows: U32,+length: Nat,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  input_domain: BandGather.Domain<BandWindow.input_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>,+source: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length),+range: WindowRange<machine,geometry,first,rows>) ->
  RoomsOf(machine,geometry,BandWindow.window(geometry,first,rows)):
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},kernel,stride,padding,+output_height,+output_width} = geometry
  +limits = Scalar.limits(machine)
  +capacity = Machine.capacity(limits)
  +window = BandWindow.window(geometry,first,rows)
  +window_rows = BandPoolWindowValues.window_rows(window)
  +output_rows = BandPoolWindowValues.output_rows(window)
  +images_positive = positive_images(geometry,OutputDomain.positive_from_selector(shape,geometry,selected))
  +rows_bound = stored_rows_bound(machine,shape,geometry,first,rows,selected,range)
  +input_room = machine_room(BandWindow.input_geometry(geometry),limits,capacity,input_domain)
  +planes_room = machine_room(BandPoolExecution.planes_geometry(geometry),limits,capacity,OutputDomain.planes_domain(machine,shape,geometry,planes,selected,planes_counted))
  +output_room = machine_room(BandPoolExecution.allocation_guards_output_geometry(geometry),limits,capacity,OutputDomain.from_selector(machine,shape,geometry,count,selected,counted))
  BandSweeps.Rooms{window_range_fit(machine,geometry,first,rows,range),
    leaf_fit(BandPoolExecution.allocation_guards_output_geometry(geometry),BandPoolWindowValues.output_first(window),output_rows,U32.to_nat(BandPoolExecution.window_count(geometry,window)),
      OutputDomain.owner_metadata(machine,shape,geometry,count,first,rows,length,selected,counted,source)),
    OutputDomain.selected_height(shape,geometry,selected),
    OutputDomain.positive_height(geometry,OutputDomain.positive_from_selector(shape,geometry,selected)),
    input_room,planes_room,output_room,
    Order.transitive(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(width))),BandGather.elements(BandWindow.input_geometry(geometry)),U32.to_nat(capacity),
      BandPoolExecution.local_elements(U32.to_nat(images),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),U32.to_nat(window_rows),images_positive,rows_bound),input_room),
    Order.transitive(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(output_width))),BandGather.elements(BandPoolExecution.planes_geometry(geometry)),U32.to_nat(capacity),
      BandPoolExecution.local_elements(U32.to_nat(images),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(output_width),U32.to_nat(window_rows),images_positive,rows_bound),planes_room),
    Order.transitive(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(output_rows),U32.to_nat(output_width))),BandGather.elements(BandPoolExecution.allocation_guards_output_geometry(geometry)),U32.to_nat(capacity),
      BandPoolExecution.local_elements(U32.to_nat(images),U32.to_nat(channels),U32.to_nat(output_height),U32.to_nat(output_width),U32.to_nat(output_rows),images_positive,OutputDomain.window_rows_bounded(geometry,first,rows)),output_room)}

# A proof-only owner retains the same tree and length. Its certificate follows
# from the actual ContentReady witness; no buffer or sample is synthesized.
type CertifiedContent<-Element: Data,-content: Buffer.Content<Element>> is Data:
  CertifiedContent{owner: Buffer.Model<Element>,exact: {Buffer.content(~Element,owner) == content : Buffer.Content<Element>},ready: Buffer.Ready<Element,owner>}

def certify(~Element: Data,+content: Buffer.Content<Element>,ready: Buffer.ContentReady<Element,content>) -> CertifiedContent<Element,content>:
  Buffer.Content{+tree,+length} = content
  Buffer.ContentReady{+depth,balanced,depth_ok,room} = ready
  +certificate = {Certificate.Certificate{depth,
    Equal.sym(Array<Element>,Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,Tree.pack(Element,tree)))),Tree.pack(Element,tree),
      Certificate.restored(Element,depth,Tree.pack(Element,tree),Witness.from_shape(Element,depth,tree,balanced)))} : Certificate.Certificate<Element,Tree.pack(Element,tree)>}
  CertifiedContent{Buffer.Model{tree,length,certificate},{==},Buffer.Ready{depth_ok,room}}

def range_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+rows: BandWindow.Rows,+content: Buffer.Content<Element>,
  values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,rows,content)) -> Buffer.ContentReady<Element,content>:
  BandWindow.Rows{first,count} = rows
  Buffer.Content{tree,length} = content
  Shape.Both{ready,Shape.Both{metadata,values}} = values
  ready

def range_first(rows: BandWindow.Rows) -> U32:
  BandWindow.Rows{first,count} = rows
  first

def range_count(rows: BandWindow.Rows) -> U32:
  BandWindow.Rows{first,count} = rows
  count

def window_above(window: BandWindow.RowWindow) -> BandWindow.Rows:
  BandWindow.RowWindow{output,above,below,input} = window
  above

def window_below(window: BandWindow.RowWindow) -> BandWindow.Rows:
  BandWindow.RowWindow{output,above,below,input} = window
  below

def range_metadata(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+rows: BandWindow.Rows,+content: Buffer.Content<Element>,
  values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,rows,content)) -> BandShape.Leaf(geometry,range_first(rows),range_count(rows),Buffer.content_length(~Element,content)):
  BandWindow.Rows{first,count} = rows
  Buffer.Content{tree,length} = content
  Shape.Both{ready,Shape.Both{metadata,values}} = values
  metadata

def modeled_range_metadata(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+rows: BandWindow.Rows,+owner: Buffer.Model<Element>,
  values: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,rows,Buffer.content(~Element,owner))) ->
  BandShape.Leaf(geometry,range_first(rows),range_count(rows),Buffer.length(~Element,owner)):
  BandWindow.Rows{first,count} = rows
  Buffer.Model{tree,length,certificate} = owner
  range_metadata(~Element,~Logical,~original,logical,geometry,rows,Buffer.content(~Element,owner),values)

def assembly_certified(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+machine: Scalar.Observation,+logical: Logical,+geometry: BandWindow.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Content<Element>,+above: Buffer.Content<Element>,+below: Buffer.Content<Element>,
  core_owner: CertifiedContent<Element,core>,above_owner: CertifiedContent<Element,above>,below_owner: CertifiedContent<Element,below>,
  +domain: BandGather.Domain<BandWindow.input_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>,
  +core_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{first,rows},core),
  +above_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),window_above(window),above),
  +below_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),window_below(window),below),
  +coordinates: TensorBandWindow.CoreGeometry<BandPoolWindowValues.geometry_images(BandWindow.input_geometry(geometry)),BandPoolWindowValues.geometry_height(BandWindow.input_geometry(geometry)),first,rows,Machine.capacity(Scalar.limits(machine))>,
  +window_exact: {BandWindow.window(geometry,first,rows) == window : BandWindow.RowWindow}) ->
  BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{BandPoolWindowValues.window_first(window),BandPoolWindowValues.window_rows(window)},BandPoolExecution.assembled_input(~Element,geometry,fill,first,rows,window,core,above,below)):
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  BandWindow.RowWindow{output,BandWindow.Rows{+above_first,+above_rows},BandWindow.Rows{+below_first,+below_rows},BandWindow.Rows{+window_first,+window_rows}} = window
  CertifiedContent{+core_model,+core_exact,core_ready} = core_owner
  CertifiedContent{+above_model,+above_exact,above_ready} = above_owner
  CertifiedContent{+below_model,+below_exact,below_ready} = below_owner
  +core_range = {
    %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,core_model),core,core_exact) : BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{first,rows},_) core_values :
    BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{first,rows},Buffer.content(~Element,core_model))}
  +above_range = {
    %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,above_model),above,above_exact) : BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{above_first,above_rows},_) above_values :
    BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{above_first,above_rows},Buffer.content(~Element,above_model))}
  +below_range = {
    %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,below_model),below,below_exact) : BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{below_first,below_rows},_) below_values :
    BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{below_first,below_rows},Buffer.content(~Element,below_model))}
  +within = TensorBandWindow.sources_within(~Element,images,channels,height,width,Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),domain,
    above_model,above_first,above_rows,core_model,first,rows,below_model,below_first,below_rows,
    modeled_range_metadata(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{above_first,above_rows},above_model,above_range),
    modeled_range_metadata(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{first,rows},core_model,core_range),
    modeled_range_metadata(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{below_first,below_rows},below_model,below_range))
  +locations = {
    %window_exact : TensorBandWindow.Sourced(~Element,BandWindow.input_geometry(geometry),_,above_model,core_model,below_model,first,rows)
    TensorBandWindow.window_sources(~Element,images,channels,height,width,kernel,stride,padding,output_height,output_width,first,rows,above_model,core_model,below_model,Scalar.limits(machine),coordinates) :
    TensorBandWindow.Sourced(~Element,BandWindow.input_geometry(geometry),window,above_model,core_model,below_model,first,rows)}
  +target = {
    %window_exact : TensorBandWindow.WindowFacts(BandWindow.input_geometry(geometry),_,Machine.capacity(Scalar.limits(machine)),Machine.half_word(Scalar.limits(machine)))
    TensorBandWindow.window_fits(images,channels,height,width,kernel,stride,padding,output_height,output_width,first,rows,Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),domain,coordinates) :
    TensorBandWindow.WindowFacts(BandWindow.input_geometry(geometry),window,Machine.capacity(Scalar.limits(machine)),Machine.half_word(Scalar.limits(machine)))}
  %core_exact : BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{window_first,window_rows},BandPoolExecution.assembled_input(~Element,geometry,fill,first,rows,window,_,above,below))
  %above_exact : BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{window_first,window_rows},BandPoolExecution.assembled_input(~Element,geometry,fill,first,rows,window,Buffer.content(~Element,core_model),_,below))
  %below_exact : BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{window_first,window_rows},BandPoolExecution.assembled_input(~Element,geometry,fill,first,rows,window,Buffer.content(~Element,core_model),Buffer.content(~Element,above_model),_))
  %TensorBandWindow.assemble_contents(~Element,BandWindow.input_geometry(geometry),fill,BandWindow.Rows{window_first,window_rows},BandWindow.Rows{first,rows},BandWindow.Rows{above_first,above_rows},BandWindow.Rows{below_first,below_rows},core_model,above_model,below_model) :
    BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{window_first,window_rows},_)
  TensorBandWindow.assembly_window(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),fill,window_first,window_rows,above_model,above_first,above_rows,core_model,first,rows,below_model,below_first,below_rows,Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),domain,above_range,core_range,below_range,within,locations,target)

def window_cells_exact(+geometry: BandWindow.Geometry,+window: BandWindow.RowWindow) ->
  {BandPoolExecution.window_cells(geometry,window) == Nat.mul(U32.to_nat(OutputDomain.channels(geometry)),Nat.mul(U32.to_nat(BandPoolWindowValues.output_rows(window)),U32.to_nat(OutputDomain.output_width(geometry)))) : Nat}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{output_first,output_rows},above,below,input} = window
  {==}

# The allocated count of a window leaf is the number of its output cells.
def window_count_cells(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,+selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},
  counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>}) ->
  {U32.to_nat(BandPoolExecution.window_count(geometry,BandWindow.window(geometry,first,rows))) == BandPoolExecution.window_cells(geometry,BandWindow.window(geometry,first,rows)) : Nat}:
  Equal.trans(Nat,U32.to_nat(BandPoolExecution.window_count(geometry,BandWindow.window(geometry,first,rows))),Nat.mul(U32.to_nat(OutputDomain.channels(geometry)),Nat.mul(U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(OutputDomain.output_width(geometry)))),BandPoolExecution.window_cells(geometry,BandWindow.window(geometry,first,rows)),
    OutputDomain.window_count_exact(machine,geometry,first,rows,OutputDomain.positive_from_selector(shape,geometry,selected),OutputDomain.from_selector(machine,shape,geometry,count,selected,counted)),
    Equal.sym(Nat,BandPoolExecution.window_cells(geometry,BandWindow.window(geometry,first,rows)),Nat.mul(U32.to_nat(OutputDomain.channels(geometry)),Nat.mul(U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(OutputDomain.output_width(geometry)))),window_cells_exact(geometry,BandWindow.window(geometry,first,rows))))

def output_mapping(geometry: BandWindow.Geometry) -> BandPlan.Mapping:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandPlan.Mapping{U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride)}

def word_from_decoded(+word: U32,+natural: Nat,exact: {U32.to_nat(word) == natural : Nat}) -> {word == U32.from_nat(natural) : U32}:
  Equal.trans(U32,word,U32.from_nat(U32.to_nat(word)),U32.from_nat(natural),Equal.sym(U32,U32.from_nat(U32.to_nat(word)),word,Arithmetic.native_roundtrip(word)),Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(word),natural,exact))

def window_begin_decoded(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,+length: Nat,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length)) ->
  {U32.to_nat(BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows))) == BandPlan.mapping(output_mapping(geometry),U32.to_nat(first)) : Nat}:
  BandWindow.Geometry{Bands.Geometry{+images,channels,+height,width},kernel,+stride,padding,+output_height,output_width} = geometry
  +capacity = Machine.capacity(Scalar.limits(machine))
  +half = Machine.half_word(Scalar.limits(machine))
  +hp = OutputDomain.selected_height(shape,geometry,selected)
  +sp = OutputDomain.selected_stride(shape,geometry,selected)
  +hr = OutputDomain.selected_height_room(machine,shape,geometry,selected)
  +output_domain = OutputDomain.from_selector(machine,shape,geometry,count,selected,counted)
  +output_room = BandGather.rows_room(BandPoolExecution.values_output_geometry(geometry),Scalar.limits(machine),capacity,output_domain)
  +end = Nat.add(U32.to_nat(first),U32.to_nat(rows))
  +mapped_end = BandPlan.mapping(output_mapping(geometry),end)
  +mapped_begin = BandPlan.mapping(output_mapping(geometry),U32.to_nat(first))
  +begin_room = Order.transitive(mapped_begin,mapped_end,U32.to_nat(capacity),BandPlan.monotone(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(first),end,hp,sp,Order.le_add(U32.to_nat(first),U32.to_nat(rows))),
    Order.transitive(mapped_end,Nat.mul(U32.to_nat(images),U32.to_nat(output_height)),U32.to_nat(capacity),OutputDomain.mapped_end_inside(geometry,first,rows,hp,sp,OutputDomain.source_fit(geometry,first,rows,length,metadata)),output_room))
  Equal.trans(Nat,U32.to_nat(BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows))),U32.to_nat((first / height * output_height + TensorBandWindow.output_first(height,output_height,stride,first) : U32)),mapped_begin,
    Equal.cong(U32,Nat,U32.to_nat,BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows)),(first / height * output_height + TensorBandWindow.output_first(height,output_height,stride,first) : U32),OutputDomain.window_first(geometry,first,rows)),
    TensorBandWindow.first_global(height,output_height,stride,first,capacity,half,Limits.capacity_fits(Scalar.limits(machine)),Limits.half_word_fits(Scalar.limits(machine)),hp,sp,Limits.double_bound(Scalar.limits(machine),U32.to_nat(height),hr),
      Limits.double_bound(Scalar.limits(machine),U32.to_nat(stride),OutputDomain.selected_stride_room(machine,shape,geometry,selected)),hr,begin_room))

def window_rows_decoded(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,+length: Nat,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length)) ->
  {U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))) == Nat.sub(BandPlan.mapping(output_mapping(geometry),Nat.add(U32.to_nat(first),U32.to_nat(rows))),BandPlan.mapping(output_mapping(geometry),U32.to_nat(first))) : Nat}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},kernel,+stride,padding,+output_height,output_width} = geometry
  Equal.trans(Nat,U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(BandWindow.subtract_clipped(TensorBandWindow.output_end(height,output_height,stride,first,rows),TensorBandWindow.output_first(height,output_height,stride,first))),
    Nat.sub(BandPlan.mapping(output_mapping(geometry),Nat.add(U32.to_nat(first),U32.to_nat(rows))),BandPlan.mapping(output_mapping(geometry),U32.to_nat(first))),
    Equal.cong(U32,Nat,U32.to_nat,BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)),BandWindow.subtract_clipped(TensorBandWindow.output_end(height,output_height,stride,first,rows),TensorBandWindow.output_first(height,output_height,stride,first)),OutputDomain.window_rows(geometry,first,rows)),
    TensorBandWindow.rows_mapped(height,output_height,stride,first,rows,Machine.capacity(Scalar.limits(machine)),Limits.capacity_fits(Scalar.limits(machine)),
      OutputDomain.output_height_room(machine,geometry,OutputDomain.positive_from_selector(shape,geometry,selected),OutputDomain.from_selector(machine,shape,geometry,count,selected,counted)),OutputDomain.mapping(machine,shape,geometry,first,rows,length,selected,metadata)))

def mapped_leaf(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,+length: Nat,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length)) ->
  {Bands.Rows{BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows)),BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))} == BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,output_mapping(geometry),Bands.Rows{first,rows}) : Bands.Plan}:
  +begin = BandPlan.mapping(output_mapping(geometry),U32.to_nat(first))
  +difference = Nat.sub(BandPlan.mapping(output_mapping(geometry),Nat.add(U32.to_nat(first),U32.to_nat(rows))),begin)
  %word_from_decoded(BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows)),begin,window_begin_decoded(machine,shape,geometry,count,first,rows,length,selected,counted,metadata)) :
    {Bands.Rows{BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows)),BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))} == Bands.Rows{_,U32.from_nat(difference)} : Bands.Plan}
  %word_from_decoded(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)),difference,window_rows_decoded(machine,shape,geometry,count,first,rows,length,selected,counted,metadata)) :
    {Bands.Rows{BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows)),BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))} == Bands.Rows{BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows)),_} : Bands.Plan}
  {==}

def row_content_metadata(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+content: Buffer.Content<Element>,
  metadata: BandShape.Leaf(geometry,first,rows,Buffer.content_length(~Element,content))) ->
  BandShape.Contents(~Element,geometry,Model.RowBand{content,first,rows}):
  Buffer.Content{tree,length} = content
  metadata

# The pooled leaf has the length of its output destination.
def pooled_length(~Element: Data,~operation: Element -> Element -> Element,+pool: Access.Pool,+pad: Element,+origin: U32,+channels: U32,
  +input: Buffer.Content<Element>,+middle: Buffer.Content<Element>,+output: Buffer.Content<Element>) ->
  {Buffer.content_length(~Element,BandPoolExecution.pooled_model(~Element,~operation,pool,pad,origin,channels,input,middle,output)) == Buffer.content_length(~Element,output) : Nat}:
  Access.Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  Buffer.Content{input_tree,input_length} = input
  Buffer.Content{middle_tree,middle_length} = middle
  Buffer.Content{output_tree,output_length} = output
  {==}

def output_leaf_metadata(~Element: Data,~operation: Element -> Element -> Element,+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+count: U32,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+core: Buffer.Content<Element>,+above: Buffer.Content<Element>,+below: Buffer.Content<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,Buffer.content_length(~Element,core)),window_exact: {BandWindow.window(geometry,first,rows) == window : BandWindow.RowWindow}) ->
  BandShape.Contents(~Element,BandPoolExecution.values_output_geometry(geometry),BandPoolExecution.refinement_window_model(~Element,~operation,geometry,pad,fill,first,rows,window,core,above,below)):
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},+above_range,+below_range,BandWindow.Rows{+window_first,+window_rows}} = window
  +middle_count = (channels * window_rows * output_width : U32)
  +local_count = (channels * output_rows * output_width : U32)
  +pool = {Access.Pool{window_rows,width,kernel,stride,padding,output_rows,output_width} : Access.Pool}
  +origin = ((output_first - first / height * output_height) * stride + first / height * height - window_first : U32)
  +input = TensorBandWindow.assembly_content(~Element,BandWindow.input_geometry(geometry),fill,BandWindow.Rows{window_first,window_rows},BandWindow.Rows{first,rows},above_range,below_range,core,above,below)
  +middle = {Buffer.Content{Tree.fresh(Element,Tensors.depth_for(middle_count),fill),U32.to_nat(middle_count)} : Buffer.Content<Element>}
  +output = {Buffer.Content{Tree.fresh(Element,Tensors.depth_for(local_count),fill),U32.to_nat(local_count)} : Buffer.Content<Element>}
  row_content_metadata(~Element,BandPoolExecution.values_output_geometry(geometry),output_first,output_rows,
    BandPoolExecution.pooled_model(~Element,~operation,pool,pad,origin,channels,input,middle,output),
    %Equal.sym(Nat,Buffer.content_length(~Element,BandPoolExecution.pooled_model(~Element,~operation,pool,pad,origin,channels,input,middle,output)),U32.to_nat(local_count),
      pooled_length(~Element,~operation,pool,pad,origin,channels,input,middle,output)) : BandShape.Leaf(BandPoolExecution.values_output_geometry(geometry),output_first,output_rows,_)
    %window_exact : BandShape.Leaf(BandPoolExecution.values_output_geometry(geometry),BandPoolWindowValues.output_first(_),BandPoolWindowValues.output_rows(_),U32.to_nat(BandPoolExecution.window_count(geometry,_)))
    OutputDomain.owner_metadata(machine,shape,geometry,count,first,rows,Buffer.content_length(~Element,core),selected,counted,metadata))

type OutputOwnership<-Element: Data,-geometry: BandWindow.Geometry,-content: Model.Content<Element>,-mapped: Bands.Plan> is Data:
  OutputOwnership{metadata: BandShape.Contents(~Element,BandPoolExecution.values_output_geometry(geometry),content),ready: Model.ContentReady(~Element,content),
    layout: {Model.content_layout(~Element,content) == mapped : Bands.Plan}}

def ownership_leaf(~Element: Data,~operation: Element -> Element -> Element,+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+count: U32,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+core: Buffer.Content<Element>,+global_source: Model.Content<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,Buffer.content_length(~Element,core)),+window_exact: {BandWindow.window(geometry,first,rows) == window : BandWindow.RowWindow}) ->
  OutputOwnership<Element,geometry,
    BandPoolExecution.execution_model(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Bands.Rows{first,rows},Model.RowBand{core,first,rows},BandHalo.leaf_model(~Element,BandWindow.input_geometry(geometry),pad,first,rows,window,global_source)),
    BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,output_mapping(geometry),Bands.Rows{first,rows})>:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{output_first,output_rows},+above_rows,+below_rows,BandWindow.Rows{window_first,window_rows}} = window
  +above = BandHalo.extract_refinement_result(~Element,BandWindow.input_geometry(geometry),pad,above_rows,global_source)
  +below = BandHalo.extract_refinement_result(~Element,BandWindow.input_geometry(geometry),pad,below_rows,global_source)
  OutputOwnership{
    output_leaf_metadata(~Element,~operation,machine,shape,geometry,pad,fill,count,first,rows,window,core,above,below,selected,counted,metadata,window_exact),
    %window_exact : Model.ContentReady(~Element,BandPoolExecution.refinement_window_model(~Element,~operation,geometry,pad,fill,first,rows,_,core,above,below))
      BandPoolExecution.modeled_ready(~Element,~operation,geometry,pad,fill,first,rows,BandWindow.window(geometry,first,rows),core,above,below,OutputDomain.allocation_ready(machine,shape,geometry,count,first,rows,selected,counted)),
    %window_exact : {Bands.Rows{BandPoolWindowValues.output_first(_),BandPoolWindowValues.output_rows(_)} == BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,output_mapping(geometry),Bands.Rows{first,rows}) : Bands.Plan}
      mapped_leaf(machine,shape,geometry,count,first,rows,Buffer.content_length(~Element,core),selected,counted,metadata)}

def ownership_join(~Element: Data,+geometry: BandWindow.Geometry,+left: Model.Content<Element>,+right: Model.Content<Element>,+left_plan: Bands.Plan,+right_plan: Bands.Plan,
  left_facts: OutputOwnership<Element,geometry,left,left_plan>,right_facts: OutputOwnership<Element,geometry,right,right_plan>) ->
  OutputOwnership<Element,geometry,Model.Joined{left,right},Bands.Split{left_plan,right_plan}>:
  OutputOwnership{left_metadata,left_ready,+left_layout} = left_facts
  OutputOwnership{right_metadata,right_ready,+right_layout} = right_facts
  OutputOwnership{Shape.Both{left_metadata,right_metadata},Shape.Both{left_ready,right_ready},
    %left_layout : {Bands.Split{Model.content_layout(~Element,left),Model.content_layout(~Element,right)} == Bands.Split{_,right_plan} : Bands.Plan}
    %right_layout : {Bands.Split{Model.content_layout(~Element,left),Model.content_layout(~Element,right)} == Bands.Split{Model.content_layout(~Element,left),_} : Bands.Plan}
    {==}}

def ownership_tree(~Element: Data,~operation: Element -> Element -> Element,+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+count: U32,+global_source: Model.Content<Element>,+cores: Model.Content<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  metadata: BandShape.Contents(~Element,BandWindow.input_geometry(geometry),cores)) ->
  OutputOwnership<Element,geometry,
    BandPoolExecution.execution_model(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Model.content_layout(~Element,cores),cores,BandHalo.model(~Element,geometry,pad,Model.content_layout(~Element,cores),global_source)),
    BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,output_mapping(geometry),Model.content_layout(~Element,cores))>:
  match cores:
    case Model.RowBand{+core,+first,+rows}:
      Buffer.Content{tree,length} = core
      ownership_leaf(~Element,~operation,machine,shape,geometry,pad,fill,workers,count,first,rows,BandWindow.window(geometry,first,rows),core,global_source,selected,counted,metadata,{==})
    case Model.Joined{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      ownership_join(~Element,geometry,
        BandPoolExecution.execution_model(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Model.content_layout(~Element,left),left,BandHalo.model(~Element,geometry,pad,Model.content_layout(~Element,left),global_source)),
        BandPoolExecution.execution_model(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Model.content_layout(~Element,right),right,BandHalo.model(~Element,geometry,pad,Model.content_layout(~Element,right),global_source)),
        BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,output_mapping(geometry),Model.content_layout(~Element,left)),BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,output_mapping(geometry),Model.content_layout(~Element,right)),
        ownership_tree(~Element,~operation,machine,shape,geometry,pad,fill,workers,count,global_source,left,selected,counted,left_metadata),ownership_tree(~Element,~operation,machine,shape,geometry,pad,fill,workers,count,global_source,right,selected,counted,right_metadata))

def ownership_well_formed(~Element: Data,+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+source: Model.Content<Element>,+output: Model.Content<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  source_coverage: BandPlan.Coverage(Model.content_layout(~Element,source),0n,Inspection.total_rows(BandWindow.input_geometry(geometry))),
  ownership: OutputOwnership<Element,geometry,output,BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,output_mapping(geometry),Model.content_layout(~Element,source))>) ->
  BandShape.WellFormed<Element,BandPoolExecution.values_output_geometry(geometry),output>:
  BandWindow.Geometry{Bands.Geometry{+images,channels,+height,width},kernel,+stride,padding,+output_height,output_width} = geometry
  OutputOwnership{metadata,ready,+layout_exact} = ownership
  +output_domain = OutputDomain.from_selector(machine,shape,geometry,count,selected,counted)
  BandShape.WellFormed{metadata,ready,
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,output),BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,output_mapping(geometry),Model.content_layout(~Element,source)),layout_exact) : BandPlan.Coverage(_,0n,Inspection.total_rows(BandPoolExecution.values_output_geometry(geometry)))
    BandPlan.output_coverage(U32.to_nat(images),U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Model.content_layout(~Element,source),Machine.capacity(Scalar.limits(machine)),Limits.capacity_fits(Scalar.limits(machine)),
      OutputDomain.selected_height(shape,geometry,selected),OutputDomain.selected_stride(shape,geometry,selected),BandGather.rows_room(BandPoolExecution.values_output_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),output_domain),source_coverage)}

def modeled_well_formed(~Element: Data,~operation: Element -> Element -> Element,+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+count: U32,+source: Model.Content<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  well_formed: BandShape.WellFormed<Element,BandWindow.input_geometry(geometry),source>) ->
  BandShape.WellFormed<Element,BandPoolExecution.values_output_geometry(geometry),BandPoolExecution.planned_model(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Model.content_layout(~Element,source),source)>:
  BandShape.WellFormed{metadata,ready,coverage} = well_formed
  ownership_well_formed(~Element,machine,shape,geometry,count,source,BandPoolExecution.planned_model(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Model.content_layout(~Element,source),source),selected,counted,coverage,
    ownership_tree(~Element,~operation,machine,shape,geometry,pad,fill,workers,count,source,source,selected,counted,metadata))

def execute_well_formed(~Element: Data,~operation: Element -> Element -> Element,+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+count: U32,+source: Model.Model<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  well_formed: BandShape.WellFormed<Element,BandWindow.input_geometry(geometry),Model.content(~Element,source)>) ->
  BandShape.WellFormed<Element,BandPoolExecution.values_output_geometry(geometry),Model.contents(~Element,BandPool.execute(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Model.pack(~Element,source)))>:
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandPool.execute(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Model.pack(~Element,source))),
    BandPoolExecution.planned_model(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Model.layout(~Element,source),Model.content(~Element,source)),BandPoolExecution.execution_execute(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},source)) : BandShape.WellFormed<Element,BandPoolExecution.values_output_geometry(geometry),_>
  %Equal.sym(Bands.Plan,Model.layout(~Element,source),Model.content_layout(~Element,Model.content(~Element,source)),Model.layout_matches_content(~Element,source)) :
    BandShape.WellFormed<Element,BandPoolExecution.values_output_geometry(geometry),BandPoolExecution.planned_model(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},_,Model.content(~Element,source))>
  modeled_well_formed(~Element,~operation,machine,shape,geometry,pad,fill,workers,count,Model.content(~Element,source),selected,counted,well_formed)

def subtract_self(+value: Nat) -> {Nat.sub(value,value) == 0n : Nat}:
  %Addition.zero_right(value) : {Nat.sub(_,value) == 0n : Nat}
  Order.subtract_shift(value,0n)

def zero_source_output_rows(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+length: Nat,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,0,length)) -> {BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,0)) == 0 : U32}:
  +begin = BandPlan.mapping(output_mapping(geometry),U32.to_nat(first))
  words_equal(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,0)),0,
    Equal.trans(Nat,U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,0))),Nat.sub(BandPlan.mapping(output_mapping(geometry),Nat.add(U32.to_nat(first),0n)),begin),0n,
      window_rows_decoded(machine,shape,geometry,count,first,0,length,selected,counted,metadata),
      %Equal.sym(Nat,Nat.add(U32.to_nat(first),0n),U32.to_nat(first),Addition.zero_right(U32.to_nat(first))) : {Nat.sub(BandPlan.mapping(output_mapping(geometry),_),begin) == 0n : Nat}
      subtract_self(begin)))

def output_cells_zero(+geometry: BandWindow.Geometry,+window: BandWindow.RowWindow,zero: {BandPoolWindowValues.output_rows(window) == 0 : U32}) -> {BandPoolExecution.window_cells(geometry,window) == 0n : Nat}:
  BandWindow.Geometry{Bands.Geometry{images,+channels,height,width},kernel,stride,padding,output_height,+output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{first,+rows},above,below,input} = window
  %Equal.sym(U32,rows,0,zero) : {Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(_),U32.to_nat(output_width))) == 0n : Nat}
  RowMajor.multiply_zero_right(U32.to_nat(channels))

def zero_leaf_receipt(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+limits: Machine.Limits,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+core: Buffer.Content<Element>,+source: Model.Content<Element>,
  zero: {BandPoolExecution.window_cells(geometry,window) == 0n : Nat}) ->
  BandPoolExecution.TreeFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},limits,Bands.Rows{first,rows},Model.RowBand{core,first,rows},BandHalo.leaf_model(~Element,BandWindow.input_geometry(geometry),pad,first,rows,window,source)):
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{output,above_range,below_range,input_range} = window
  Inl{zero}

def zero_source_leaf_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+machine: Scalar.Observation,+logical: Logical,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+count: U32,+first: U32,+core: Buffer.Content<Element>,+source: Model.Content<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,0,Buffer.content_length(~Element,core))) ->
  BandPoolExecution.TreeFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},Scalar.limits(machine),Bands.Rows{first,0},Model.RowBand{core,first,0},BandHalo.leaf_model(~Element,BandWindow.input_geometry(geometry),pad,first,0,BandWindow.window(geometry,first,0),source)):
  zero_leaf_receipt(~Element,~Logical,~original,logical,geometry,pad,fill,workers,Scalar.limits(machine),first,0,BandWindow.window(geometry,first,0),core,source,
    output_cells_zero(geometry,BandWindow.window(geometry,first,0),zero_source_output_rows(machine,shape,geometry,count,first,Buffer.content_length(~Element,core),selected,counted,metadata)))

def core_range_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,+core: Buffer.Content<Element>,
  metadata: BandShape.Leaf(geometry,first,rows,Buffer.content_length(~Element,core)),ready: Buffer.ContentReady<Element,core>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.RowBand{core,first,rows})) ->
  BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{first,rows},core):
  Buffer.Content{tree,length} = core
  Shape.Both{ready,Shape.Both{metadata,values}}

# What a window with output rows supplies to its two sweeps.
def window_leaf_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+machine: Scalar.Observation,+logical: Logical,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+planes: U32,+count: U32,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Content<Element>,+above: Buffer.Content<Element>,+below: Buffer.Content<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +domain: BandGather.Domain<BandWindow.input_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>,
  +metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,Buffer.content_length(~Element,core)),
  +assembled: BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{BandPoolWindowValues.window_first(window),BandPoolWindowValues.window_rows(window)},BandPoolExecution.assembled_input(~Element,geometry,fill,first,rows,window,core,above,below)),
  +positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +output_positive: {Nat.is_lt(0n,U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)))) == True{} : Bool},
  +window_exact: {BandWindow.window(geometry,first,rows) == window : BandWindow.RowWindow}) ->
  BandPoolExecution.WindowFacts(~Element,~Logical,~original,logical,geometry,pad,fill,first,rows,window,core,above,below,Scalar.limits(machine)):
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},+above_rows,+below_rows,BandWindow.Rows{+window_first,+window_rows}} = window
  +length = Buffer.content_length(~Element,core)
  +interval = interval_from_mapping(machine,shape,geometry,count,first,rows,length,selected,counted,metadata,OutputDomain.mapping(machine,shape,geometry,first,rows,length,selected,metadata),output_positive)
  +range = window_range(machine,shape,geometry,first,rows,length,selected,domain,metadata,positive,interval)
  +rows_bound = stored_rows_bound(machine,shape,geometry,first,rows,selected,range)
  Shape.Both{
    %window_exact : Depth.Ready<BandPoolExecution.middle_count(geometry,_)> middle_ready(machine,shape,geometry,planes,first,rows,selected,planes_counted,rows_bound),
    Shape.Both{
      %window_exact : Depth.Ready<BandPoolExecution.window_count(geometry,_)> OutputDomain.allocation_ready(machine,shape,geometry,count,first,rows,selected,counted),
      Shape.Both{
        %window_exact : {U32.to_nat(BandPoolExecution.middle_count(geometry,_)) == BandPoolExecution.middle_cells(geometry,_) : Nat}
        middle_count_cells(machine,shape,geometry,planes,first,rows,selected,planes_counted,rows_bound),
        Shape.Both{
          %window_exact : {U32.to_nat(BandPoolExecution.window_count(geometry,_)) == BandPoolExecution.window_cells(geometry,_) : Nat}
          window_count_cells(machine,shape,geometry,count,first,rows,selected,counted),
          Shape.Both{BandPoolExecution.assembled_values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandPoolWindowValues.window_first(window),BandPoolWindowValues.window_rows(window),BandPoolExecution.assembled_input(~Element,geometry,fill,first,rows,window,core,above,below),assembled),
            Shape.Both{
              %window_exact : RoomsOf(machine,geometry,_)
              window_rooms(machine,shape,geometry,planes,count,first,rows,length,selected,planes_counted,counted,domain,metadata,range),
              %window_exact : RowsOf(geometry,first,_)
              window_row_facts(machine,shape,geometry,count,first,rows,selected,counted,interval,range)}}}}}}

def nonzero_leaf(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+machine: Scalar.Observation,+logical: Logical,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+planes: U32,+count: U32,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+core: Buffer.Content<Element>,+global_source: Model.Content<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +domain: BandGather.Domain<BandWindow.input_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>,
  +core_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{first,rows},core),
  halo_values: BandHalo.TreeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandHalo.leaf_model(~Element,BandWindow.input_geometry(geometry),pad,first,rows,window,global_source)),
  +positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +output_positive: {Nat.is_lt(0n,U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)))) == True{} : Bool},
  +window_exact: {BandWindow.window(geometry,first,rows) == window : BandWindow.RowWindow}) ->
  BandPoolExecution.TreeFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},Scalar.limits(machine),Bands.Rows{first,rows},Model.RowBand{core,first,rows},
    BandHalo.leaf_model(~Element,BandWindow.input_geometry(geometry),pad,first,rows,window,global_source)):
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{output_first,output_rows},+above_rows,+below_rows,BandWindow.Rows{window_first,window_rows}} = window
  Shape.Both{+above_values,+below_values} = halo_values
  +above = BandHalo.extract_refinement_result(~Element,BandWindow.input_geometry(geometry),pad,above_rows,global_source)
  +below = BandHalo.extract_refinement_result(~Element,BandWindow.input_geometry(geometry),pad,below_rows,global_source)
  +metadata = range_metadata(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{first,rows},core,core_values)
  +coordinates = TensorBandWindow.halo_bounds_coordinates(images,channels,height,width,first,rows,Buffer.content_length(~Element,core),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),domain,metadata,positive)
  +assembled = assembly_certified(~Element,~Logical,~original,machine,logical,geometry,fill,first,rows,window,core,above,below,
    certify(~Element,core,range_ready(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{first,rows},core,core_values)),
    certify(~Element,above,range_ready(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),above_rows,above,above_values)),
    certify(~Element,below,range_ready(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),below_rows,below,below_values)),domain,core_values,above_values,below_values,coordinates,window_exact)
  Inr{window_leaf_facts(~Element,~Logical,~original,machine,logical,shape,geometry,pad,fill,planes,count,first,rows,window,core,above,below,selected,planes_counted,counted,domain,metadata,assembled,positive,output_positive,window_exact)}

# A leaf with source rows may still have no output row; the native test of
# its output rows chooses between the empty leaf and the two sweeps.
def planned_leaf(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+machine: Scalar.Observation,+logical: Logical,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+planes: U32,+count: U32,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+core: Buffer.Content<Element>,+global_source: Model.Content<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +domain: BandGather.Domain<BandWindow.input_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>,
  +core_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{first,rows},core),
  halo_values: BandHalo.TreeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandHalo.leaf_model(~Element,BandWindow.input_geometry(geometry),pad,first,rows,window,global_source)),
  +positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+window_exact: {BandWindow.window(geometry,first,rows) == window : BandWindow.RowWindow},
  choice: Bool,+choice_exact: {U32.is_zero(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))) == choice : Bool}) ->
  BandPoolExecution.TreeFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},Scalar.limits(machine),Bands.Rows{first,rows},Model.RowBand{core,first,rows},
    BandHalo.leaf_model(~Element,BandWindow.input_geometry(geometry),pad,first,rows,window,global_source)):
  match choice:
    case True{}:
      zero_leaf_receipt(~Element,~Logical,~original,logical,geometry,pad,fill,workers,Scalar.limits(machine),first,rows,window,core,global_source,
        %window_exact : {BandPoolExecution.window_cells(geometry,_) == 0n : Nat}
        output_cells_zero(geometry,BandWindow.window(geometry,first,rows),native_zero(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)),choice_exact)))
    case False{}:
      nonzero_leaf(~Element,~Logical,~original,machine,logical,shape,geometry,pad,fill,workers,planes,count,first,rows,window,core,global_source,selected,planes_counted,counted,domain,core_values,halo_values,positive,
        OutputDomain.native_nonzero_positive(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)),choice_exact),window_exact)

def normal_nonempty_leaf_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+machine: Scalar.Observation,+logical: Logical,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+planes: U32,+count: U32,+first: U32,+rows: U32,+core: Buffer.Content<Element>,+source: Model.Model<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +domain: BandGather.Domain<BandWindow.input_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>,
  well_formed: BandShape.WellFormed<Element,BandWindow.input_geometry(geometry),Model.content(~Element,source)>,global_values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.content(~Element,source)),
  +metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,Buffer.content_length(~Element,core)),ready: Buffer.ContentReady<Element,core>,
  values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.RowBand{core,first,rows}),+positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}) ->
  BandPoolExecution.TreeFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},Scalar.limits(machine),Bands.Rows{first,rows},Model.RowBand{core,first,rows},BandHalo.model(~Element,geometry,pad,Bands.Rows{first,rows},Model.content(~Element,source))):
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  +window = BandWindow.window(geometry,first,rows)
  +halo_values = BandHalo.leaf(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),pad,first,rows,window,source,Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),domain,well_formed,global_values,
    TensorBandWindow.nonempty_window(images,channels,height,width,kernel,stride,padding,output_height,output_width,first,rows,Buffer.content_length(~Element,core),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),domain,metadata,positive))
  planned_leaf(~Element,~Logical,~original,machine,logical,shape,geometry,pad,fill,workers,planes,count,first,rows,window,core,Model.content(~Element,source),selected,planes_counted,counted,domain,
    core_range_values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),first,rows,core,metadata,ready,values),halo_values,positive,{==},U32.is_zero(BandPoolWindowValues.output_rows(window)),{==})

# The actual selector makes both source axes positive. The normal global
# input count then supplies the address Domain for all active source owners.
type SourceEnvelope<-machine: Scalar.Observation,-geometry: BandWindow.Geometry> is Data:
  NormalSource{domain: BandGather.Domain<BandWindow.input_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>}

def input_channels_positive(+geometry: BandWindow.Geometry,positive: OutputDomain.Positive<geometry>) -> {BandGather.channels_positive(BandWindow.input_geometry(geometry)) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  OutputDomain.Positive{images_positive,channels_positive,height_positive,width_positive} = positive
  channels_positive

# The actual Normal state's counted input shape supplies the source capacity
# envelope. This constructor reads the input count, never the output count.
def normal_source_from_count(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,counted: Global.Counted<shape>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool}) -> SourceEnvelope<machine,geometry>:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  NormalSource{
    %Equal.sym(Machine.Limits,Scalar.limits(machine),Default.limits(),Scalar.actual_limits(machine)) : BandGather.Domain<BandWindow.input_geometry(geometry),_,Machine.capacity(_)>
    Global.domain(shape,BandWindow.input_geometry(geometry),counted,OutputDomain.selected_shape(shape,geometry,selected),
    input_channels_positive(geometry,OutputDomain.positive_from_selector(shape,geometry,selected)),OutputDomain.selected_height(shape,geometry,selected),
    OutputDomain.selected_width(shape,geometry,selected))}

def source_envelope(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,counted: Global.Counted<shape>,
  selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool}) -> SourceEnvelope<machine,geometry>:
  normal_source_from_count(machine,shape,geometry,counted,selected)

def nonempty_leaf_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+machine: Scalar.Observation,+logical: Logical,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+planes: U32,+count: U32,+first: U32,+rows: U32,+core: Buffer.Content<Element>,+source: Model.Model<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  well_formed: BandShape.WellFormed<Element,BandWindow.input_geometry(geometry),Model.content(~Element,source)>,global_values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.content(~Element,source)),
  metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,Buffer.content_length(~Element,core)),ready: Buffer.ContentReady<Element,core>,
  values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.RowBand{core,first,rows}),positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},envelope: SourceEnvelope<machine,geometry>) ->
  BandPoolExecution.TreeFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},Scalar.limits(machine),Bands.Rows{first,rows},Model.RowBand{core,first,rows},BandHalo.model(~Element,geometry,pad,Bands.Rows{first,rows},Model.content(~Element,source))):
  match envelope:
    case NormalSource{domain}:
      normal_nonempty_leaf_facts(~Element,~Logical,~original,machine,logical,shape,geometry,pad,fill,workers,planes,count,first,rows,core,source,selected,planes_counted,counted,domain,well_formed,global_values,metadata,ready,values,positive)

def leaf_facts_choice(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+machine: Scalar.Observation,+logical: Logical,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+planes: U32,+count: U32,+first: U32,+rows: U32,+core: Buffer.Content<Element>,+source: Model.Model<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  well_formed: BandShape.WellFormed<Element,BandWindow.input_geometry(geometry),Model.content(~Element,source)>,global_values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.content(~Element,source)),
  metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,Buffer.content_length(~Element,core)),ready: Buffer.ContentReady<Element,core>,
  values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.RowBand{core,first,rows}),envelope: SourceEnvelope<machine,geometry>,
  choice: Bool,+choice_exact: {U32.is_zero(rows) == choice : Bool}) ->
  BandPoolExecution.TreeFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},Scalar.limits(machine),Bands.Rows{first,rows},Model.RowBand{core,first,rows},BandHalo.model(~Element,geometry,pad,Bands.Rows{first,rows},Model.content(~Element,source))):
  match choice:
    case True{}:
      +zero = native_zero(rows,choice_exact)
      %Equal.sym(U32,rows,0,zero) : BandPoolExecution.TreeFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},Scalar.limits(machine),Bands.Rows{first,_},Model.RowBand{core,first,_},BandHalo.model(~Element,geometry,pad,Bands.Rows{first,_},Model.content(~Element,source)))
      zero_source_leaf_facts(~Element,~Logical,~original,machine,logical,shape,geometry,pad,fill,workers,count,first,core,Model.content(~Element,source),selected,counted,
        %zero : BandShape.Leaf(BandWindow.input_geometry(geometry),first,_,Buffer.content_length(~Element,core)) metadata)
    case False{}:
      nonempty_leaf_facts(~Element,~Logical,~original,machine,logical,shape,geometry,pad,fill,workers,planes,count,first,rows,core,source,selected,planes_counted,counted,well_formed,global_values,metadata,ready,values,OutputDomain.native_nonzero_positive(rows,choice_exact),envelope)

def tree_facts_all(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+machine: Scalar.Observation,+logical: Logical,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+planes: U32,+count: U32,+source: Model.Model<Element>,+cores: Model.Content<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},+planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},+counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +well_formed: BandShape.WellFormed<Element,BandWindow.input_geometry(geometry),Model.content(~Element,source)>,+global_values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.content(~Element,source)),
  metadata: BandShape.Contents(~Element,BandWindow.input_geometry(geometry),cores),ready: Model.ContentReady(~Element,cores),
  values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),cores),+envelope: SourceEnvelope<machine,geometry>) ->
  BandPoolExecution.TreeFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},Scalar.limits(machine),Model.content_layout(~Element,cores),cores,BandHalo.model(~Element,geometry,pad,Model.content_layout(~Element,cores),Model.content(~Element,source))):
  match cores:
    case Model.RowBand{+core,+first,+rows}:
      Buffer.Content{tree,length} = core
      leaf_facts_choice(~Element,~Logical,~original,machine,logical,shape,geometry,pad,fill,workers,planes,count,first,rows,core,source,selected,planes_counted,counted,well_formed,global_values,metadata,ready,values,envelope,U32.is_zero(rows),{==})
    case Model.Joined{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_values,right_values} = values
      Shape.Both{tree_facts_all(~Element,~Logical,~original,machine,logical,shape,geometry,pad,fill,workers,planes,count,source,left,selected,planes_counted,counted,well_formed,global_values,left_metadata,left_ready,left_values,envelope),
        tree_facts_all(~Element,~Logical,~original,machine,logical,shape,geometry,pad,fill,workers,planes,count,source,right,selected,planes_counted,counted,well_formed,global_values,right_metadata,right_ready,right_values,envelope)}

def planned_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+machine: Scalar.Observation,+logical: Logical,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+planes: U32,+count: U32,+source: Model.Model<Element>,
  selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +well_formed: BandShape.WellFormed<Element,BandWindow.input_geometry(geometry),Model.content(~Element,source)>,+values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.content(~Element,source)),envelope: SourceEnvelope<machine,geometry>) ->
  BandPoolExecution.PlannedFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},Scalar.limits(machine),Model.layout(~Element,source),Model.content(~Element,source)):
  BandShape.WellFormed{metadata,ready,coverage} = well_formed
  %Equal.sym(Bands.Plan,Model.layout(~Element,source),Model.content_layout(~Element,Model.content(~Element,source)),Model.layout_matches_content(~Element,source)) :
    BandPoolExecution.PlannedFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},Scalar.limits(machine),_,Model.content(~Element,source))
  tree_facts_all(~Element,~Logical,~original,machine,logical,shape,geometry,pad,fill,workers,planes,count,source,Model.content(~Element,source),selected,planes_counted,counted,well_formed,values,metadata,ready,values,envelope)

# Every output cell of the executed band pool holds the two-sweep pool of its
# pixel over the whole logical input. The planes swept along the columns and
# the output are both counted by the selector; the input by its Normal state.
def execute_values_all(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+machine: Scalar.Observation,+logical: Logical,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+workers: U32,+planes: U32,+count: U32,+source: Model.Model<Element>,
  +selected: {OutputDomain.enabled(shape,geometry) == True{} : Bool},planes_counted: {Tensors.count_of(Window.middle_shape(shape,OutputDomain.pool(geometry))) == Some{planes} : Maybe<&2,U32>},counted: {Tensors.count_of(OutputDomain.output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  well_formed: BandShape.WellFormed<Element,BandWindow.input_geometry(geometry),Model.content(~Element,source)>,values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.content(~Element,source)),input_counted: Global.Counted<shape>) ->
  BandValues.Values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),
    BandPoolExecution.global_context(~Element,~Logical,logical,geometry,pad),BandPoolExecution.values_output_geometry(geometry),
    Model.contents(~Element,BandPool.execute(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Model.pack(~Element,source)))):
  BandPoolExecution.values_execute(~Element,~Logical,~original,~operation,logical,BandPool.Context{geometry,pad,fill,workers},Scalar.limits(machine),source,
    planned_facts(~Element,~Logical,~original,machine,logical,shape,geometry,pad,fill,workers,planes,count,source,selected,planes_counted,counted,well_formed,values,source_envelope(machine,shape,geometry,input_counted,selected)))
