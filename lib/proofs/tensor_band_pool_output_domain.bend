# The actual selected band pool branch has a nonempty output shape. Decode
# its shape/count before allocating each native output window. Empty output
# shapes use tensor_window's existing dense fallback.
import Base
import ../tensor.bend as Tensors
import ../tensor_window.bend as Window
import ../tensor_shape.bend as Shapes
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_count_refinement.bend as Count
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_band_count.bend as BandCount
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_pool_execution.bend as BandPoolExecution
import ./tensor_band_pool_window_values.bend as BandPoolWindowValues
import ./tensor_allocation_depth.bend as Depth
import ./machine_limit_bounds.bend as Limits
import ./storage_buffer_model.bend as Buffer
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./encoded_coordinates.bend as Encoded
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_window.bend as TensorBandWindow
import ./tensor_band_plan.bend as BandPlan
import ./tensor_pool_scalar_bounds.bend as Scalar

def pool(geometry: BandWindow.Geometry) -> Access.Pool:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  Access.Pool{height,width,kernel,stride,padding,output_height,output_width}

def channels(geometry: BandWindow.Geometry) -> U32:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  channels

def output_width(geometry: BandWindow.Geometry) -> U32:
  BandWindow.Geometry{input,kernel,stride,padding,output_height,output_width} = geometry
  output_width

def output_shape(+shape: List<&2,U32>,geometry: BandWindow.Geometry) -> List<&2,U32>:
  BandWindow.Geometry{input,kernel,stride,padding,output_height,output_width} = geometry
  List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[output_height,output_width])

def enabled(+shape: List<&2,U32>,+geometry: BandWindow.Geometry) -> Bool:
  Window.band_shape(shape,BandWindow.input_geometry(geometry)) && Operations.pool_geometry(pool(geometry)) &&
    U32.is_lt(0,BandPoolWindowValues.pool_height(pool(geometry))) && U32.is_lt(0,BandPoolWindowValues.pool_width(pool(geometry))) && Bool.not(Shapes.has_zero(output_shape(shape,geometry)))

def selected_shape(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,selected: {enabled(shape,geometry) == True{} : Bool}) ->
  {Window.band_shape(shape,BandWindow.input_geometry(geometry)) == True{} : Bool}:
  +shape_ok = Window.band_shape(shape,BandWindow.input_geometry(geometry))
  +pool_ok = Operations.pool_geometry(pool(geometry))
  +height_ok = U32.is_lt(0,BandPoolWindowValues.pool_height(pool(geometry)))
  +width_ok = U32.is_lt(0,BandPoolWindowValues.pool_width(pool(geometry)))
  +output_ok = Bool.not(Shapes.has_zero(output_shape(shape,geometry)))
  Boolean.left(shape_ok,pool_ok,Boolean.left((shape_ok && pool_ok : Bool),height_ok,
    Boolean.left((shape_ok && pool_ok && height_ok : Bool),width_ok,
      Boolean.left((shape_ok && pool_ok && height_ok && width_ok : Bool),output_ok,selected))))

def natural_nonzero_positive(value: Nat,nonzero: {Nat.is_eq(value,0n) == False{} : Bool}) -> {Nat.is_lt(0n,value) == True{} : Bool}:
  match value:
    case 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},nonzero)))
    case 1n+rest: {==}

def native_nonzero_positive(+value: U32,nonzero: {U32.is_zero(value) == False{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(value)) == True{} : Bool}:
  natural_nonzero_positive(U32.to_nat(value),Equal.trans(Bool,Nat.is_eq(U32.to_nat(value),0n),U32.is_zero(value),False{},
    Equal.sym(Bool,U32.is_zero(value),Nat.is_eq(U32.to_nat(value),0n),Comparison.eq(value,0)),nonzero))

type Positive<-geometry: BandWindow.Geometry> is Data:
  Positive{images: {Nat.is_lt(0n,U32.to_nat(BandPoolExecution.images(geometry))) == True{} : Bool},
    channels: {BandGather.channels_positive(BandPoolExecution.allocation_guards_output_geometry(geometry)) == True{} : Bool},
    height: {BandGather.height_positive(BandPoolExecution.allocation_guards_output_geometry(geometry)) == True{} : Bool},
    width: {BandGather.width_positive(BandPoolExecution.allocation_guards_output_geometry(geometry)) == True{} : Bool}}

# band_shape admits CHW and NCHW. Decode their actual leading dimensions;
# the selector's has_zero check supplies positivity of every output extent.
def output_positive_geometry(+geometry: BandWindow.Geometry,+shape: List<&2,U32>,
  +shape_valid: {Window.band_shape(shape,BandWindow.input_geometry(geometry)) == True{} : Bool},
  +nonempty: {Shapes.has_zero(output_shape(shape,geometry)) == False{} : Bool}) -> Positive<geometry>:
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},kernel,stride,padding,+output_height,+output_width} = geometry
  match shape:
    case Con{+c,Con{+h,Con{+w,Nil{}}}}:
      +pair = Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),
        Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),shape_valid))
      +images_equal = ShapeLaws.native_equal(images,1,Boolean.left(U32.is_eq(images,1),U32.is_eq(c,channels),pair))
      +channels_equal = ShapeLaws.native_equal(c,channels,Boolean.right(U32.is_eq(images,1),U32.is_eq(c,channels),pair))
      +tail = Boolean.or_false_right(U32.is_zero(c),Shapes.has_zero([output_height,output_width]),nonempty)
      +last = Boolean.or_false_right(U32.is_zero(output_height),Shapes.has_zero([output_width]),tail)
      Positive{
        %Equal.sym(U32,images,1,images_equal) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool} {==},
        %channels_equal : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool}
        native_nonzero_positive(c,Boolean.or_false_left(U32.is_zero(c),Shapes.has_zero([output_height,output_width]),nonempty)),
        native_nonzero_positive(output_height,Boolean.or_false_left(U32.is_zero(output_height),Shapes.has_zero([output_width]),tail)),
        native_nonzero_positive(output_width,Boolean.or_false_left(U32.is_zero(output_width),Shapes.has_zero([]),last))}
    case Con{+n,Con{+c,Con{+h,Con{+w,Nil{}}}}}:
      +pair = Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),
        Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),shape_valid))
      +images_equal = ShapeLaws.native_equal(n,images,Boolean.left(U32.is_eq(n,images),U32.is_eq(c,channels),pair))
      +channels_equal = ShapeLaws.native_equal(c,channels,Boolean.right(U32.is_eq(n,images),U32.is_eq(c,channels),pair))
      +tail = Boolean.or_false_right(U32.is_zero(n),Shapes.has_zero([c,output_height,output_width]),nonempty)
      +spatial = Boolean.or_false_right(U32.is_zero(c),Shapes.has_zero([output_height,output_width]),tail)
      +last = Boolean.or_false_right(U32.is_zero(output_height),Shapes.has_zero([output_width]),spatial)
      Positive{
        %images_equal : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool}
        native_nonzero_positive(n,Boolean.or_false_left(U32.is_zero(n),Shapes.has_zero([c,output_height,output_width]),nonempty)),
        %channels_equal : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool}
        native_nonzero_positive(c,Boolean.or_false_left(U32.is_zero(c),Shapes.has_zero([output_height,output_width]),tail)),
        native_nonzero_positive(output_height,Boolean.or_false_left(U32.is_zero(output_height),Shapes.has_zero([output_width]),spatial)),
        native_nonzero_positive(output_width,Boolean.or_false_left(U32.is_zero(output_width),Shapes.has_zero([]),last))}
    case Nil{}: Empty.absurd(Positive<geometry>,WordEncoding.false_true(shape_valid))
    case Con{a,Nil{}}: Empty.absurd(Positive<geometry>,WordEncoding.false_true(shape_valid))
    case Con{a,Con{b,Nil{}}}: Empty.absurd(Positive<geometry>,WordEncoding.false_true(shape_valid))
    case Con{a,Con{b,Con{c,Con{d,Con{e,rest}}}}}: Empty.absurd(Positive<geometry>,WordEncoding.false_true(shape_valid))

def output_positive(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,
  +shape_valid: {Window.band_shape(shape,BandWindow.input_geometry(geometry)) == True{} : Bool},
  +nonempty: {Shapes.has_zero(output_shape(shape,geometry)) == False{} : Bool}) -> Positive<geometry>:
  output_positive_geometry(geometry,shape,shape_valid,nonempty)

def positive_from_selector(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+selected: {enabled(shape,geometry) == True{} : Bool}) -> Positive<geometry>:
  output_positive(shape,geometry,selected_shape(shape,geometry,selected),
    BandPoolWindowValues.selected_output_nonempty(pool(geometry),Window.band_shape(shape,BandWindow.input_geometry(geometry)),output_shape(shape,geometry),selected))

def three_elements(+channels: U32,+height: U32,+width: U32) ->
  {ShapeLaws.elements([channels,height,width]) == BandGather.elements(Bands.Geometry{1,channels,height,width}) : Nat}:
  +elements = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))
  Equal.trans(Nat,ShapeLaws.elements([channels,height,width]),elements,BandGather.elements(Bands.Geometry{1,channels,height,width}),
    Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),value)),Nat.mul(U32.to_nat(width),1n),U32.to_nat(width),ShapeLaws.times_one(U32.to_nat(width))),
    Equal.sym(Nat,BandGather.elements(Bands.Geometry{1,channels,height,width}),elements,Addition.zero_right(elements)))

# A shape whose last two axes are replaced, and the geometry it describes.
def replaced_shape(+shape: List<&2,U32>,rows: U32,columns: U32) -> List<&2,U32>:
  List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[rows,columns])

def replaced_geometry(input: Bands.Geometry,rows: U32,columns: U32) -> Bands.Geometry:
  Bands.Geometry{images,channels,height,width} = input
  Bands.Geometry{images,channels,rows,columns}

def replaced_elements(+input: Bands.Geometry,+shape: List<&2,U32>,+output_height: U32,+output_width: U32,+shape_valid: {Window.band_shape(shape,input) == True{} : Bool}) ->
  {ShapeLaws.elements(replaced_shape(shape,output_height,output_width)) == BandGather.elements(replaced_geometry(input,output_height,output_width)) : Nat}:
  Bands.Geometry{+images,+channels,+height,+width} = input
  match shape:
    case Con{+c,Con{+h,Con{+w,Nil{}}}}:
      +pair = Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),
        Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),shape_valid))
      %Equal.sym(U32,images,1,ShapeLaws.native_equal(images,1,Boolean.left(U32.is_eq(images,1),U32.is_eq(c,channels),pair))) :
        {ShapeLaws.elements([c,output_height,output_width]) == BandGather.elements(Bands.Geometry{_,channels,output_height,output_width}) : Nat}
      %Equal.sym(U32,c,channels,ShapeLaws.native_equal(c,channels,Boolean.right(U32.is_eq(images,1),U32.is_eq(c,channels),pair))) :
        {ShapeLaws.elements([_,output_height,output_width]) == BandGather.elements(Bands.Geometry{1,channels,output_height,output_width}) : Nat}
      three_elements(channels,output_height,output_width)
    case Con{+n,Con{+c,Con{+h,Con{+w,Nil{}}}}}:
      +pair = Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),
        Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),shape_valid))
      %Equal.sym(U32,n,images,ShapeLaws.native_equal(n,images,Boolean.left(U32.is_eq(n,images),U32.is_eq(c,channels),pair))) :
        {ShapeLaws.elements([_,c,output_height,output_width]) == BandGather.elements(Bands.Geometry{images,channels,output_height,output_width}) : Nat}
      %Equal.sym(U32,c,channels,ShapeLaws.native_equal(c,channels,Boolean.right(U32.is_eq(n,images),U32.is_eq(c,channels),pair))) :
        {ShapeLaws.elements([images,_,output_height,output_width]) == BandGather.elements(Bands.Geometry{images,channels,output_height,output_width}) : Nat}
      BandCount.domain_elements(Bands.Geometry{images,channels,output_height,output_width})
    case Nil{}: Empty.absurd({ShapeLaws.elements(replaced_shape(shape,output_height,output_width)) == BandGather.elements(replaced_geometry(input,output_height,output_width)) : Nat},WordEncoding.false_true(shape_valid))
    case Con{a,Nil{}}: Empty.absurd({ShapeLaws.elements(replaced_shape(shape,output_height,output_width)) == BandGather.elements(replaced_geometry(input,output_height,output_width)) : Nat},WordEncoding.false_true(shape_valid))
    case Con{a,Con{b,Nil{}}}: Empty.absurd({ShapeLaws.elements(replaced_shape(shape,output_height,output_width)) == BandGather.elements(replaced_geometry(input,output_height,output_width)) : Nat},WordEncoding.false_true(shape_valid))
    case Con{a,Con{b,Con{c,Con{d,Con{e,rest}}}}}: Empty.absurd({ShapeLaws.elements(replaced_shape(shape,output_height,output_width)) == BandGather.elements(replaced_geometry(input,output_height,output_width)) : Nat},WordEncoding.false_true(shape_valid))

def count_shape_elements_geometry(+geometry: BandWindow.Geometry,+shape: List<&2,U32>,+shape_valid: {Window.band_shape(shape,BandWindow.input_geometry(geometry)) == True{} : Bool}) ->
  {ShapeLaws.elements(output_shape(shape,geometry)) == BandGather.elements(BandPoolExecution.allocation_guards_output_geometry(geometry)) : Nat}:
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},kernel,stride,padding,+output_height,+output_width} = geometry
  replaced_elements(Bands.Geometry{images,channels,height,width},shape,output_height,output_width,shape_valid)

def count_shape_elements(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+shape_valid: {Window.band_shape(shape,BandWindow.input_geometry(geometry)) == True{} : Bool}) ->
  {ShapeLaws.elements(output_shape(shape,geometry)) == BandGather.elements(BandPoolExecution.allocation_guards_output_geometry(geometry)) : Nat}:
  count_shape_elements_geometry(geometry,shape,shape_valid)

def output_count(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,
  shape_valid: {Window.band_shape(shape,BandWindow.input_geometry(geometry)) == True{} : Bool},
  counted: {Tensors.count_of(output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>}) -> Count.Count<BandGather.elements(BandPoolExecution.allocation_guards_output_geometry(geometry)),count>:
  %count_shape_elements(shape,geometry,shape_valid) : Count.Count<_,count>
  Count.count_of(output_shape(shape,geometry),count,counted)

def domain_from_count_machine(+geometry: BandWindow.Geometry,+count: U32,+limits: Machine.Limits,+actual: {limits == Default.limits() : Machine.Limits},
  positive: Positive<geometry>,evidence: Count.Count<BandGather.elements(BandPoolExecution.allocation_guards_output_geometry(geometry)),count>) ->
  BandGather.Domain<BandPoolExecution.allocation_guards_output_geometry(geometry),limits,Machine.capacity(limits)>:
  Positive{images,channels,height,width} = positive
  Count.Count{exact,bounded} = evidence
  +room = BandCount.natural_room(BandGather.elements(BandPoolExecution.allocation_guards_output_geometry(geometry)),count,Machine.capacity(limits),exact,
    %Equal.sym(Machine.Limits,limits,Default.limits(),actual) : {U32.is_le(count,Machine.capacity(_)) == True{} : Bool} bounded)
  BandGather.Domain{Limits.capacity_fits(limits),channels,height,width,room,room}

def domain_from_count(+machine: Scalar.Observation,+geometry: BandWindow.Geometry,+count: U32,positive: Positive<geometry>,evidence: Count.Count<BandGather.elements(BandPoolExecution.allocation_guards_output_geometry(geometry)),count>) ->
  BandGather.Domain<BandPoolExecution.allocation_guards_output_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>:
  domain_from_count_machine(geometry,count,Scalar.limits(machine),Scalar.actual_limits(machine),positive,evidence)

def from_selector(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+selected: {enabled(shape,geometry) == True{} : Bool},
  counted: {Tensors.count_of(output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>}) ->
  BandGather.Domain<BandPoolExecution.allocation_guards_output_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>:
  domain_from_count(machine,geometry,count,positive_from_selector(shape,geometry,selected),output_count(shape,geometry,count,selected_shape(shape,geometry,selected),counted))

def ready_from_positive(+machine: Scalar.Observation,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,positive: Positive<geometry>,
  domain: BandGather.Domain<BandPoolExecution.allocation_guards_output_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>) ->
  Depth.Ready<BandPoolExecution.window_count(geometry,BandWindow.window(geometry,first,rows))>:
  Positive{images,channels,height,width} = positive
  BandPoolExecution.active(geometry,first,rows,Scalar.limits(machine),domain,images)

def allocation_ready(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,+selected: {enabled(shape,geometry) == True{} : Bool},
  counted: {Tensors.count_of(output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>}) ->
  Depth.Ready<BandPoolExecution.window_count(geometry,BandWindow.window(geometry,first,rows))>:
  ready_from_positive(machine,geometry,first,rows,positive_from_selector(shape,geometry,selected),from_selector(machine,shape,geometry,count,selected,counted))

# The output projection survives the inactive branch unchanged. Inactive
# windows clear their input ranges; they retain the output partition boundary.
def nonempty_first(active: Bool,+first: U32,+rows: U32,above: BandWindow.Rows,below: BandWindow.Rows,input: BandWindow.Rows) ->
  {BandPoolWindowValues.output_first(BandWindow.nonempty(active,BandWindow.RowWindow{BandWindow.Rows{first,rows},above,below,input})) == first : U32}:
  match active above below input:
    case True{} above below input: {==}
    case False{} BandWindow.Rows{above_first,above_rows} BandWindow.Rows{below_first,below_rows} BandWindow.Rows{input_first,input_rows}: {==}

def window_first(+geometry: BandWindow.Geometry,+first: U32,+rows: U32) ->
  {BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows)) ==
    (first / BandPoolWindowValues.pool_height(pool(geometry)) * BandPoolWindowValues.geometry_output_height(geometry) +
      TensorBandWindow.output_first(BandPoolWindowValues.pool_height(pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(pool(geometry)),first) : U32) : U32}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},+kernel,+stride,+padding,+output_height,output_width} = geometry
  +image = (first / height : U32)
  +core_first = (first % height : U32)
  +core_end = (core_first + rows : U32)
  +begin = TensorBandWindow.output_first(height,output_height,stride,first)
  +finish = TensorBandWindow.output_end(height,output_height,stride,first,rows)
  +start = BandWindow.start(height,begin,stride,padding)
  +end = TensorBandWindow.samples_end(height,finish,stride,kernel,padding)
  nonempty_first(U32.is_lt(begin,finish),(image * output_height + begin : U32),BandWindow.subtract_clipped(finish,begin),
    BandWindow.clipped((image * height + start : U32),(image * height + U32.min(core_first,end) : U32)),
    BandWindow.clipped((image * height + U32.max(core_end,start) : U32),(image * height + end : U32)),
    BandWindow.clipped((image * height + start : U32),(image * height + end : U32)))

def nonempty_rows(active: Bool,+first: U32,+rows: U32,above: BandWindow.Rows,below: BandWindow.Rows,input: BandWindow.Rows) ->
  {BandPoolWindowValues.output_rows(BandWindow.nonempty(active,BandWindow.RowWindow{BandWindow.Rows{first,rows},above,below,input})) == rows : U32}:
  match active above below input:
    case True{} above below input: {==}
    case False{} BandWindow.Rows{above_first,above_rows} BandWindow.Rows{below_first,below_rows} BandWindow.Rows{input_first,input_rows}: {==}

def window_rows(+geometry: BandWindow.Geometry,+first: U32,+rows: U32) ->
  {BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)) == BandWindow.subtract_clipped(
    TensorBandWindow.output_end(BandPoolWindowValues.pool_height(pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(pool(geometry)),first,rows),
    TensorBandWindow.output_first(BandPoolWindowValues.pool_height(pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(pool(geometry)),first)) : U32}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},+kernel,+stride,+padding,+output_height,output_width} = geometry
  +image = (first / height : U32)
  +core_first = (first % height : U32)
  +core_end = (core_first + rows : U32)
  +begin = TensorBandWindow.output_first(height,output_height,stride,first)
  +finish = TensorBandWindow.output_end(height,output_height,stride,first,rows)
  +start = BandWindow.start(height,begin,stride,padding)
  +end = TensorBandWindow.samples_end(height,finish,stride,kernel,padding)
  nonempty_rows(U32.is_lt(begin,finish),(image * output_height + begin : U32),BandWindow.subtract_clipped(finish,begin),
    BandWindow.clipped((image * height + start : U32),(image * height + U32.min(core_first,end) : U32)),
    BandWindow.clipped((image * height + U32.max(core_end,start) : U32),(image * height + end : U32)),
    BandWindow.clipped((image * height + start : U32),(image * height + end : U32)))

def output_height_room(+machine: Scalar.Observation,+geometry: BandWindow.Geometry,positive: Positive<geometry>,
  domain: BandGather.Domain<BandPoolExecution.allocation_guards_output_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>) ->
  {Nat.is_le(U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry)),U32.to_nat(Machine.capacity(Scalar.limits(machine)))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{+images,channels,height,width},kernel,stride,padding,+output_height,output_width} = geometry
  Positive{images_positive,channels_positive,height_positive,width_positive} = positive
  +rows_room = BandGather.rows_room(BandPoolExecution.allocation_guards_output_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),domain)
  Order.transitive(U32.to_nat(output_height),Nat.mul(U32.to_nat(images),U32.to_nat(output_height)),U32.to_nat(Machine.capacity(Scalar.limits(machine))),
    %RowMajor.multiply_commutative(U32.to_nat(output_height),U32.to_nat(images)) : {Nat.is_le(U32.to_nat(output_height),_) == True{} : Bool}
    RowMajor.product_covers(U32.to_nat(output_height),U32.to_nat(images),images_positive),rows_room)

def count_projection(+geometry: BandWindow.Geometry,window: BandWindow.RowWindow) ->
  {BandPoolExecution.window_count(geometry,window) == (channels(geometry) * BandPoolWindowValues.output_rows(window) * output_width(geometry) : U32) : U32}:
  BandWindow.Geometry{Bands.Geometry{images,+channels,height,width},kernel,stride,padding,output_height,+output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{first,+rows},above,below,input} = window
  {==}

def bounds_rows_projection(+window: BandWindow.RowWindow) ->
  {BandPoolWindowValues.output_rows(window) == TensorBandWindow.output_rows(window) : U32}:
  BandWindow.RowWindow{BandWindow.Rows{first,rows},above,below,input} = window
  {==}

def bounds_height_projection(+geometry: BandWindow.Geometry) ->
  {BandPoolWindowValues.geometry_output_height(geometry) == TensorBandWindow.output_height(geometry) : U32}:
  BandWindow.Geometry{input,kernel,stride,padding,height,width} = geometry
  {==}

# Transport the established window bound through its formal projections.
# The nonempty selector inside window remains opaque to this adapter.
def window_rows_bounded(+geometry: BandWindow.Geometry,+first: U32,+rows: U32) ->
  {Nat.is_le(U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry))) == True{} : Bool}:
  %Equal.sym(U32,BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)),TensorBandWindow.output_rows(BandWindow.window(geometry,first,rows)),bounds_rows_projection(BandWindow.window(geometry,first,rows))) :
    {Nat.is_le(U32.to_nat(_),U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry))) == True{} : Bool}
  %Equal.sym(U32,BandPoolWindowValues.geometry_output_height(geometry),TensorBandWindow.output_height(geometry),bounds_height_projection(geometry)) :
    {Nat.is_le(U32.to_nat(TensorBandWindow.output_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(_)) == True{} : Bool}
  TensorBandWindow.bounded(geometry,first,rows)

def window_count_exact(+machine: Scalar.Observation,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,positive: Positive<geometry>,
  domain: BandGather.Domain<BandPoolExecution.allocation_guards_output_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>) ->
  {U32.to_nat(BandPoolExecution.window_count(geometry,BandWindow.window(geometry,first,rows))) ==
    Nat.mul(U32.to_nat(channels(geometry)),Nat.mul(U32.to_nat(BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))),U32.to_nat(output_width(geometry)))) : Nat}:
  BandWindow.Geometry{Bands.Geometry{+images,+channels,height,width},kernel,stride,padding,+output_height,+output_width} = geometry
  Positive{images_positive,channels_positive,height_positive,+width_positive} = positive
  BandGather.Domain{capacity_ok,cp,hp,wp,+machine_room,allocation_room} = domain
  +output_rows = BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))
  +local = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(output_rows),U32.to_nat(output_width)))
  +bounded = Order.transitive(local,BandGather.elements(BandPoolExecution.allocation_guards_output_geometry(geometry)),U32.to_nat(Machine.capacity(Scalar.limits(machine))),
    BandPoolExecution.local_elements(U32.to_nat(images),U32.to_nat(channels),U32.to_nat(output_height),U32.to_nat(output_width),U32.to_nat(output_rows),images_positive,window_rows_bounded(geometry,first,rows)),machine_room)
  Equal.trans(Nat,U32.to_nat(BandPoolExecution.window_count(geometry,BandWindow.window(geometry,first,rows))),U32.to_nat((channels * output_rows * output_width : U32)),local,
    Equal.cong(U32,Nat,value => U32.to_nat(value),BandPoolExecution.window_count(geometry,BandWindow.window(geometry,first,rows)),(channels * output_rows * output_width : U32),count_projection(geometry,BandWindow.window(geometry,first,rows))),
    Equal.trans(Nat,U32.to_nat((channels * output_rows * output_width : U32)),Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(output_rows)),U32.to_nat(output_width)),local,
      Encoded.multiply_three(channels,output_rows,output_width,Machine.capacity(Scalar.limits(machine)),Limits.capacity_fits(Scalar.limits(machine)),width_positive,
        %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(output_rows)),U32.to_nat(output_width)),local,RowMajor.assoc(U32.to_nat(channels),U32.to_nat(output_rows),U32.to_nat(output_width))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(Scalar.limits(machine)))) == True{} : Bool} bounded),
      RowMajor.assoc(U32.to_nat(channels),U32.to_nat(output_rows),U32.to_nat(output_width))))

def source_local(+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+length: Nat,
  metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length)) ->
  {U32.is_le(rows,(BandPoolWindowValues.pool_height(pool(geometry)) - first % BandPoolWindowValues.pool_height(pool(geometry)) : U32)) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  local

def source_fit(+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+length: Nat,
  metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length)) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(BandPoolExecution.images(geometry)),U32.to_nat(BandPoolWindowValues.pool_height(pool(geometry))))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  rows_fit

def selected_legal(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,selected: {enabled(shape,geometry) == True{} : Bool}) ->
  {Operations.pool_geometry(pool(geometry)) == True{} : Bool}:
  BandPoolWindowValues.selected_pool_geometry(pool(geometry),Window.band_shape(shape,BandWindow.input_geometry(geometry)),output_shape(shape,geometry),selected)

def selected_height(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,selected: {enabled(shape,geometry) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(BandPoolWindowValues.pool_height(pool(geometry)))) == True{} : Bool}:
  BandPoolWindowValues.selected_input_height(pool(geometry),Window.band_shape(shape,BandWindow.input_geometry(geometry)),output_shape(shape,geometry),selected)

def selected_stride(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,selected: {enabled(shape,geometry) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(BandPoolWindowValues.pool_stride(pool(geometry)))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,+stride,padding,output_height,output_width} = geometry
  Bounds.as_nat_lt(0,BandPoolWindowValues.pool_stride(pool(geometry)),BandPoolWindowValues.pool_condition(pool(geometry),4n,selected_legal(shape,geometry,selected)))

def selected_width(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,selected: {enabled(shape,geometry) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(BandPoolWindowValues.pool_width(pool(geometry)))) == True{} : Bool}:
  BandPoolWindowValues.selected_input_width(pool(geometry),Window.band_shape(shape,BandWindow.input_geometry(geometry)),output_shape(shape,geometry),selected)

def selected_height_room(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,selected: {enabled(shape,geometry) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(BandPoolWindowValues.pool_height(pool(geometry))),U32.to_nat(Machine.capacity(Scalar.limits(machine)))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},kernel,stride,padding,output_height,output_width} = geometry
  Scalar.small_room(machine,BandPoolWindowValues.pool_height(pool(geometry)),BandPoolWindowValues.pool_condition(pool(geometry),10n,selected_legal(shape,geometry,selected)))

def selected_stride_room(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,selected: {enabled(shape,geometry) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(BandPoolWindowValues.pool_stride(pool(geometry))),U32.to_nat(Machine.capacity(Scalar.limits(machine)))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,+stride,padding,output_height,output_width} = geometry
  Scalar.small_room(machine,BandPoolWindowValues.pool_stride(pool(geometry)),BandPoolWindowValues.pool_condition(pool(geometry),7n,selected_legal(shape,geometry,selected)))

# The planes swept along the columns, [..,height,output_width], are counted
# by the selector like the output. Their count bounds every address of the
# column-swept image.
def planes_elements(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+shape_valid: {Window.band_shape(shape,BandWindow.input_geometry(geometry)) == True{} : Bool}) ->
  {ShapeLaws.elements(Window.middle_shape(shape,pool(geometry))) == BandGather.elements(BandPoolExecution.planes_geometry(geometry)) : Nat}:
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},kernel,stride,padding,output_height,+output_width} = geometry
  replaced_elements(Bands.Geometry{images,channels,height,width},shape,height,output_width,shape_valid)

def planes_count(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+planes: U32,
  shape_valid: {Window.band_shape(shape,BandWindow.input_geometry(geometry)) == True{} : Bool},
  counted: {Tensors.count_of(Window.middle_shape(shape,pool(geometry))) == Some{planes} : Maybe<&2,U32>}) -> Count.Count<BandGather.elements(BandPoolExecution.planes_geometry(geometry)),planes>:
  %planes_elements(shape,geometry,shape_valid) : Count.Count<_,planes>
  Count.count_of(Window.middle_shape(shape,pool(geometry)),planes,counted)

def planes_domain_counted(+machine: Scalar.Observation,+geometry: BandWindow.Geometry,+planes: U32,positive: Positive<geometry>,
  height_positive: {Nat.is_lt(0n,U32.to_nat(BandPoolWindowValues.pool_height(pool(geometry)))) == True{} : Bool},
  evidence: Count.Count<BandGather.elements(BandPoolExecution.planes_geometry(geometry)),planes>) ->
  BandGather.Domain<BandPoolExecution.planes_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>:
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,width},kernel,stride,padding,output_height,+output_width} = geometry
  Positive{images_positive,channels_positive,out_height_positive,width_positive} = positive
  Count.Count{exact,bounded} = evidence
  +limits = Scalar.limits(machine)
  +room = BandCount.natural_room(BandGather.elements(Bands.Geometry{images,channels,height,output_width}),planes,Machine.capacity(limits),exact,
    %Equal.sym(Machine.Limits,limits,Default.limits(),Scalar.actual_limits(machine)) : {U32.is_le(planes,Machine.capacity(_)) == True{} : Bool} bounded)
  BandGather.Domain{Limits.capacity_fits(limits),channels_positive,height_positive,width_positive,room,room}

def planes_domain(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+planes: U32,+selected: {enabled(shape,geometry) == True{} : Bool},
  counted: {Tensors.count_of(Window.middle_shape(shape,pool(geometry))) == Some{planes} : Maybe<&2,U32>}) ->
  BandGather.Domain<BandPoolExecution.planes_geometry(geometry),Scalar.limits(machine),Machine.capacity(Scalar.limits(machine))>:
  planes_domain_counted(machine,geometry,planes,positive_from_selector(shape,geometry,selected),selected_height(shape,geometry,selected),planes_count(shape,geometry,planes,selected_shape(shape,geometry,selected),counted))

def mapping(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+length: Nat,+selected: {enabled(shape,geometry) == True{} : Bool},
  metadata: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length)) ->
  TensorBandWindow.LocalMapping<BandPoolWindowValues.pool_height(pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(pool(geometry)),first,rows>:
  +height = BandPoolWindowValues.pool_height(pool(geometry))
  +stride = BandPoolWindowValues.pool_stride(pool(geometry))
  +capacity = Machine.capacity(Scalar.limits(machine))
  +division_capacity = Machine.half_word(Scalar.limits(machine))
  +hp = selected_height(shape,geometry,selected)
  +sp = selected_stride(shape,geometry,selected)
  +hr = selected_height_room(machine,shape,geometry,selected)
  +hd = Limits.double_bound(Scalar.limits(machine),U32.to_nat(height),hr)
  +sd = Limits.double_bound(Scalar.limits(machine),U32.to_nat(stride),selected_stride_room(machine,shape,geometry,selected))
  TensorBandWindow.local_mapping(height,BandPoolWindowValues.geometry_output_height(geometry),stride,first,rows,capacity,division_capacity,
    Limits.capacity_fits(Scalar.limits(machine)),Limits.half_word_fits(Scalar.limits(machine)),hp,sp,hd,sd,hr,
    TensorBandWindow.core_end(height,first,rows,capacity,division_capacity,Limits.capacity_fits(Scalar.limits(machine)),Limits.half_word_fits(Scalar.limits(machine)),hp,hd,hr,source_local(geometry,first,rows,length,metadata)))

def positive_height(+geometry: BandWindow.Geometry,positive: Positive<geometry>) ->
  {Nat.is_lt(0n,U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  Positive{images_positive,channels_positive,height_positive,width_positive} = positive
  height_positive

def mapped_end_inside(+geometry: BandWindow.Geometry,+first: U32,+rows: U32,
  +height_positive: {Nat.is_lt(0n,U32.to_nat(BandPoolWindowValues.pool_height(pool(geometry)))) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,U32.to_nat(BandPoolWindowValues.pool_stride(pool(geometry)))) == True{} : Bool},
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(BandPoolExecution.images(geometry)),U32.to_nat(BandPoolWindowValues.pool_height(pool(geometry))))) == True{} : Bool}) ->
  {Nat.is_le(BandPlan.output_boundaries_boundary(U32.to_nat(BandPoolWindowValues.pool_height(pool(geometry))),U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry)),U32.to_nat(BandPoolWindowValues.pool_stride(pool(geometry))),Nat.add(U32.to_nat(first),U32.to_nat(rows))),
    Nat.mul(U32.to_nat(BandPoolExecution.images(geometry)),U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry)))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{+images,channels,+height,width},kernel,+stride,padding,+output_height,output_width} = geometry
  %BandPlan.image_end(U32.to_nat(images),U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),height_positive,stride_positive) :
    {Nat.is_le(BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(first),U32.to_nat(rows))),_) == True{} : Bool}
  BandPlan.monotone(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height)),height_positive,stride_positive,rows_fit)

def mapped_difference_local(+height: U32,+output_height: U32,+stride: U32,+first: U32,+rows: U32,
  mapping: TensorBandWindow.LocalMapping<height,output_height,stride,first,rows>) ->
  {Nat.sub(BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(first),U32.to_nat(rows))),
    BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(first))) ==
    Nat.sub(U32.to_nat(TensorBandWindow.output_end(height,output_height,stride,first,rows)),U32.to_nat(TensorBandWindow.output_first(height,output_height,stride,first))) : Nat}:
  TensorBandWindow.LocalMapping{+begin_exact,+end_exact,ordered} = mapping
  +base = Nat.mul(Nat.div(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(output_height))
  +begin = U32.to_nat(TensorBandWindow.output_first(height,output_height,stride,first))
  +finish = U32.to_nat(TensorBandWindow.output_end(height,output_height,stride,first,rows))
  %Equal.sym(Nat,BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(first),U32.to_nat(rows))),Nat.add(base,finish),end_exact) : {Nat.sub(_,BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(first))) == Nat.sub(finish,begin) : Nat}
  %Equal.sym(Nat,BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(first)),Nat.add(base,begin),begin_exact) : {Nat.sub(Nat.add(base,finish),_) == Nat.sub(finish,begin) : Nat}
  TensorBandWindow.common_offset(base,finish,begin)

# Metadata is obtained from the source core's actual bounds, rather than from
# a positive output-row premise. This includes empty output leaves.
def metadata_from_mapping(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,+length: Nat,
  +selected: {enabled(shape,geometry) == True{} : Bool},+counted: {Tensors.count_of(output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +source: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length),
  parts: TensorBandWindow.LocalMapping<BandPoolWindowValues.pool_height(pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(pool(geometry)),first,rows>) ->
  BandShape.Leaf(BandPoolExecution.allocation_guards_output_geometry(geometry),BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows)),BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)),
    U32.to_nat(BandPoolExecution.window_count(geometry,BandWindow.window(geometry,first,rows)))):
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,width},kernel,+stride,padding,+output_height,+output_width} = geometry
  TensorBandWindow.LocalMapping{+begin_shift,+end_shift,+ordered} = parts
  +positive = positive_from_selector(shape,geometry,selected)
  +domain = from_selector(machine,shape,geometry,count,selected,counted)
  +capacity = Machine.capacity(Scalar.limits(machine))
  +division_capacity = Machine.half_word(Scalar.limits(machine))
  +hp = selected_height(shape,geometry,selected)
  +sp = selected_stride(shape,geometry,selected)
  +hr = selected_height_room(machine,shape,geometry,selected)
  +hd = Limits.double_bound(Scalar.limits(machine),U32.to_nat(height),hr)
  +sd = Limits.double_bound(Scalar.limits(machine),U32.to_nat(stride),selected_stride_room(machine,shape,geometry,selected))
  +ohr = output_height_room(machine,geometry,positive,domain)
  +ohp = positive_height(geometry,positive)
  +begin = U32.to_nat(TensorBandWindow.output_first(height,output_height,stride,first))
  +finish = U32.to_nat(TensorBandWindow.output_end(height,output_height,stride,first,rows))
  +base = Nat.mul(Nat.div(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(output_height))
  +mapped_begin = BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(first))
  +mapped_end = BandPlan.output_boundaries_boundary(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),Nat.add(U32.to_nat(first),U32.to_nat(rows)))
  +mapped_ordered = BandPlan.monotone(U32.to_nat(height),U32.to_nat(output_height),U32.to_nat(stride),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),hp,sp,Order.le_add(U32.to_nat(first),U32.to_nat(rows)))
  +end_inside = mapped_end_inside(geometry,first,rows,hp,sp,source_fit(geometry,first,rows,length,source))
  +end_room = Order.transitive(mapped_end,Nat.mul(U32.to_nat(images),U32.to_nat(output_height)),U32.to_nat(capacity),end_inside,
    BandGather.rows_room(BandPoolExecution.allocation_guards_output_geometry(geometry),Scalar.limits(machine),capacity,domain))
  +begin_room = Order.transitive(mapped_begin,mapped_end,U32.to_nat(capacity),mapped_ordered,end_room)
  +out_first = BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows))
  +out_rows = BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows))
  +first_exact = Equal.trans(Nat,U32.to_nat(out_first),U32.to_nat((first / height * output_height + TensorBandWindow.output_first(height,output_height,stride,first) : U32)),mapped_begin,
    Equal.cong(U32,Nat,value => U32.to_nat(value),out_first,(first / height * output_height + TensorBandWindow.output_first(height,output_height,stride,first) : U32),window_first(geometry,first,rows)),
    TensorBandWindow.first_global(height,output_height,stride,first,capacity,division_capacity,Limits.capacity_fits(Scalar.limits(machine)),Limits.half_word_fits(Scalar.limits(machine)),hp,sp,hd,sd,hr,begin_room))
  +rows_exact = Equal.trans(Nat,U32.to_nat(out_rows),U32.to_nat(BandWindow.subtract_clipped(TensorBandWindow.output_end(height,output_height,stride,first,rows),TensorBandWindow.output_first(height,output_height,stride,first))),Nat.sub(mapped_end,mapped_begin),
    Equal.cong(U32,Nat,value => U32.to_nat(value),out_rows,BandWindow.subtract_clipped(TensorBandWindow.output_end(height,output_height,stride,first,rows),TensorBandWindow.output_first(height,output_height,stride,first)),window_rows(geometry,first,rows)),
    TensorBandWindow.rows_mapped(height,output_height,stride,first,rows,capacity,Limits.capacity_fits(Scalar.limits(machine)),ohr,TensorBandWindow.LocalMapping{begin_shift,end_shift,ordered}))
  +rows_local = Equal.trans(Nat,U32.to_nat(out_rows),Nat.sub(mapped_end,mapped_begin),Nat.sub(finish,begin),rows_exact,
    mapped_difference_local(height,output_height,stride,first,rows,TensorBandWindow.LocalMapping{begin_shift,end_shift,ordered}))
  +span = BandPlan.span(U32.to_nat(output_height),Nat.div(U32.to_nat(first),U32.to_nat(height)),begin,finish,U32.to_nat(out_first),U32.to_nat(out_rows),ohp,
    TensorBandWindow.below(output_height,(first % height : U32),stride),ordered,
    TensorBandWindow.end_below(output_height,(first % height + rows : U32),stride,U32.is_eq((first % height + rows : U32),height)),
    Equal.trans(Nat,U32.to_nat(out_first),mapped_begin,Nat.add(base,begin),first_exact,begin_shift),rows_local)
  BandShape.Fields{
    Inspection.local_guard(output_height,out_first,out_rows,division_capacity,Limits.half_word_fits(Scalar.limits(machine)),ohp,Limits.double_bound(Scalar.limits(machine),U32.to_nat(output_height),ohr),span),
    %Equal.sym(Nat,U32.to_nat(out_first),mapped_begin,first_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(out_rows)),Nat.mul(U32.to_nat(images),U32.to_nat(output_height))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(out_rows),Nat.sub(mapped_end,mapped_begin),rows_exact) : {Nat.is_le(Nat.add(mapped_begin,_),Nat.mul(U32.to_nat(images),U32.to_nat(output_height))) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(mapped_begin,Nat.sub(mapped_end,mapped_begin)),mapped_end,Order.restore_shift(mapped_begin,mapped_end,mapped_ordered)) : {Nat.is_le(_,Nat.mul(U32.to_nat(images),U32.to_nat(output_height))) == True{} : Bool}
    end_inside,
    window_count_exact(machine,geometry,first,rows,positive,domain)}

def owner_metadata(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+count: U32,+first: U32,+rows: U32,+length: Nat,
  +selected: {enabled(shape,geometry) == True{} : Bool},counted: {Tensors.count_of(output_shape(shape,geometry)) == Some{count} : Maybe<&2,U32>},
  +source: BandShape.Leaf(BandWindow.input_geometry(geometry),first,rows,length)) ->
  BandShape.Leaf(BandPoolExecution.allocation_guards_output_geometry(geometry),BandPoolWindowValues.output_first(BandWindow.window(geometry,first,rows)),BandPoolWindowValues.output_rows(BandWindow.window(geometry,first,rows)),U32.to_nat(BandPoolExecution.window_count(geometry,BandWindow.window(geometry,first,rows)))):
  metadata_from_mapping(machine,shape,geometry,count,first,rows,length,selected,counted,source,mapping(machine,shape,geometry,first,rows,length,selected,source))

def row_scalar_room(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+coordinate: Nat,+tap: Nat,+selected: {enabled(shape,geometry) == True{} : Bool},
  coordinate_inside: {Nat.is_le(coordinate,U32.to_nat(BandPoolWindowValues.geometry_output_height(geometry))) == True{} : Bool},
  tap_inside: {Nat.is_le(tap,U32.to_nat(BandPoolWindowValues.pool_kernel(pool(geometry)))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(coordinate,U32.to_nat(BandPoolWindowValues.pool_stride(pool(geometry)))),tap),U32.to_nat(Machine.half_word(Scalar.limits(machine)))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},+kernel,+stride,+padding,+output_height,output_width} = geometry
  +legal = selected_legal(shape,geometry,selected)
  +es = BandPoolWindowValues.pool_condition(pool(geometry),10n,legal)
  +ps = BandPoolWindowValues.pool_condition(pool(geometry),6n,legal)
  +ss = BandPoolWindowValues.pool_condition(pool(geometry),7n,legal)
  +sp = selected_stride(shape,geometry,selected)
  Scalar.axis_scalar_room(machine,height,kernel,stride,padding,output_height,coordinate,tap,es,ps,BandPoolWindowValues.pool_condition(pool(geometry),8n,legal),ss,sp,
    Scalar.axis_output_exact(machine,height,kernel,stride,padding,output_height,es,ps,ss,sp,BandPoolWindowValues.pool_condition(pool(geometry),3n,legal),BandPoolWindowValues.pool_condition(pool(geometry),1n,legal)),coordinate_inside,tap_inside)

def begin_product_room(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+first: U32,+selected: {enabled(shape,geometry) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(TensorBandWindow.output_first(BandPoolWindowValues.pool_height(pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(pool(geometry)),first)),
    U32.to_nat(BandPoolWindowValues.pool_stride(pool(geometry)))),U32.to_nat(Machine.half_word(Scalar.limits(machine)))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},+kernel,+stride,padding,+output_height,output_width} = geometry
  +begin = U32.to_nat(TensorBandWindow.output_first(height,output_height,stride,first))
  +product = Nat.mul(begin,U32.to_nat(stride))
  %Addition.zero_right(product) : {Nat.is_le(_,U32.to_nat(Machine.half_word(Scalar.limits(machine)))) == True{} : Bool}
  row_scalar_room(machine,shape,geometry,begin,0n,selected,TensorBandWindow.below(output_height,(first % height : U32),stride),Order.zero_le(U32.to_nat(kernel)))

# The origin of the rows sweep of the actual caller, with its scalar envelope
# discharged from the selected pool guard.
def caller_origin(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,
  selected: {enabled(shape,geometry) == True{} : Bool}) ->
  {U32.to_nat(BandPoolExecution.window_origin(geometry,first,BandWindow.window(geometry,first,rows))) == BandPoolWindowValues.origin_nat(geometry,first) : Nat}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},kernel,+stride,padding,output_height,output_width} = geometry
  BandPoolWindowValues.origin_decoded(geometry,first,rows,Machine.half_word(Scalar.limits(machine)),Limits.half_word_fits(Scalar.limits(machine)),begin_product_room(machine,shape,geometry,first,selected))

def end_coordinate_room(+machine: Scalar.Observation,+shape: List<&2,U32>,+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+selected: {enabled(shape,geometry) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(Nat.sub(U32.to_nat(TensorBandWindow.output_end(BandPoolWindowValues.pool_height(pool(geometry)),BandPoolWindowValues.geometry_output_height(geometry),BandPoolWindowValues.pool_stride(pool(geometry)),first,rows)),1n),
    U32.to_nat(BandPoolWindowValues.pool_stride(pool(geometry)))),U32.to_nat(BandPoolWindowValues.pool_kernel(pool(geometry)))),U32.to_nat(Machine.half_word(Scalar.limits(machine)))) == True{} : Bool}:
  BandWindow.Geometry{Bands.Geometry{images,channels,+height,width},+kernel,+stride,padding,+output_height,output_width} = geometry
  +finish = U32.to_nat(TensorBandWindow.output_end(height,output_height,stride,first,rows))
  row_scalar_room(machine,shape,geometry,Nat.sub(finish,1n),U32.to_nat(kernel),selected,
    Order.transitive(Nat.sub(finish,1n),finish,U32.to_nat(output_height),Order.sub_le(finish,1n),
      TensorBandWindow.end_below(output_height,(first % height + rows : U32),stride,U32.is_eq((first % height + rows : U32),height))),Order.reflexive(U32.to_nat(kernel)))
