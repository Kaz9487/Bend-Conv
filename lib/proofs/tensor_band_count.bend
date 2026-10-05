# Local element counts of band owners and the whole-tensor machine envelope.
import Base
import ../tensor_band.bend as Bands
import ./tensor_band_gather.bend as BandGather
import ../machine_limits.bend as Machine
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./machine_limit_bounds.bend as Limits
import ./row_major_index.bend as RowMajor
import ./storage_buffer_model.bend as Buffer
import ./tensor_shape_refinement.bend as ShapeLaws
import ./u32_comparison.bend as Comparison
import ../default_machine_limits.bend as Default
import ./boolean_logic.bend as Boolean
import ./encoded_coordinates.bend as Encoded
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_refinement.bend as StorageLaws
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_count_refinement.bend as Count
import ./word_encoding.bend as WordEncoding
import ./word_multiplication.bend as Multiplication
import ../storage_allocation.bend as Allocation
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_window.bend as Window

# The actual allocator supplies the depth and logical-length bounds consumed
# by band traversal and gather semantics, including the empty-buffer case.
def fresh_capacity(~Element: Data,+depth: Nat,+fill: Element) ->
  {Tree.capacity(Element,Tree.reflect(Element,Array.new(Element,depth,fill))) == U32.shln(1,depth) : U32}:
  %Equal.sym(Array<Element>,Array.new(Element,depth,fill),Tree.pack(Element,Tree.fresh(Element,depth,fill)),StorageLaws.fresh(Element,depth,fill)) :
    {Tree.capacity(Element,Tree.reflect(Element,_)) == U32.shln(1,depth) : U32}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,Tree.fresh(Element,depth,fill))),Tree.fresh(Element,depth,fill),StorageLaws.reflect_packed(Element,Tree.fresh(Element,depth,fill))) :
    {Tree.capacity(Element,_) == U32.shln(1,depth) : U32}
  Shape.capacity(Element,depth,Tree.fresh(Element,depth,fill),Shape.fresh(Element,depth,fill))

def allocated_ready(~Element: Data,+count: U32,+fill: Element,ready: Depth.Ready<count>) ->
  Buffer.NativeReady(~Element,Bands.allocate(~Element,fill,count)):
  Depth.Ready{depth_ok,room} = ready
  %Equal.sym(Storage.Buffer<Element>,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count)),Storage.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count)),
    AllocationLaws.dispatch_same(Element,Allocation.allocated(~Element),Storage.depth_for(count),fill,U32.to_nat(count))) : Buffer.NativeReady(~Element,_)
  Shape.Both{depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,Tree.reflect(Element,Array.new(Element,Storage.depth_for(count),fill))),U32.shln(1,Storage.depth_for(count)),fresh_capacity(~Element,Storage.depth_for(count),fill)) :
      {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool} room}

def allocated(~Element: Data,+count: U32,+fill: Element,within: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Buffer.NativeReady(~Element,Bands.allocate(~Element,fill,count)):
  allocated_ready(~Element,count,fill,Depth.bounded(count,within))

def model(~Element: Data,+count: U32,fill: Element) -> Buffer.Model<Element>:
  Buffer.fresh(~Element,Storage.depth_for(count),fill,U32.to_nat(count))

def model_ready(~Element: Data,+count: U32,+fill: Element,ready: Depth.Ready<count>) -> Buffer.Ready<Element,model(~Element,count,fill)>:
  Depth.Ready{depth_ok,room} = ready
  Buffer.fresh_ready(~Element,Storage.depth_for(count),fill,U32.to_nat(count),depth_ok,room)

def bounded_model(~Element: Data,+count: U32,+fill: Element,within: {U32.is_le(count,1073741824) == True{} : Bool}) -> Buffer.Ready<Element,model(~Element,count,fill)>:
  model_ready(~Element,count,fill,Depth.bounded(count,within))

# Public shape counting discharges the whole-tensor machine envelope used by
# band semantics. Per-owner readiness remains a storage representation fact.
def extents(geometry: Bands.Geometry) -> List<&2,U32>:
  Bands.Geometry{images,channels,height,width} = geometry
  [images,channels,height,width]

def domain_elements(+geometry: Bands.Geometry) -> {ShapeLaws.elements(extents(geometry)) == BandGather.elements(geometry) : Nat}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Equal.cong(Nat,Nat,last => Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),last))),
    Nat.mul(U32.to_nat(width),1n),U32.to_nat(width),ShapeLaws.times_one(U32.to_nat(width)))

def natural_room(+expected: Nat,+count: U32,+limit: U32,exact: {expected == U32.to_nat(count) : Nat},bounded: {U32.is_le(count,limit) == True{} : Bool}) ->
  {Nat.is_le(expected,U32.to_nat(limit)) == True{} : Bool}:
  %Equal.sym(Nat,expected,U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
  %Comparison.le(count,limit) : {_ == True{} : Bool}
  bounded

def bounded(+geometry: Bands.Geometry,+limits: Machine.Limits,+count: U32,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  exact: {ShapeLaws.elements(extents(geometry)) == U32.to_nat(count) : Nat},bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) ->
  BandGather.Domain<geometry,limits,Machine.capacity(limits)>:
  +room = {
    %domain_elements(geometry) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    natural_room(ShapeLaws.elements(extents(geometry)),count,Machine.capacity(limits),exact,bounded) : {Nat.is_le(BandGather.elements(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  BandGather.Domain{Limits.capacity_fits(limits),channels_positive,height_positive,width_positive,room,room}

def evidence(+geometry: Bands.Geometry,+count: U32,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  domain_counted: Count.Count<ShapeLaws.elements(extents(geometry)),count>) ->
  BandGather.Domain<geometry,Default.limits(),Machine.capacity(Default.limits())>:
  Count.Count{exact,within} = domain_counted
  bounded(geometry,Default.limits(),count,channels_positive,height_positive,width_positive,exact,within)

def domain_counted(+geometry: Bands.Geometry,+count: U32,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  accepted: {Tensors.count_of(extents(geometry)) == Some{count} : Maybe<&2,U32>}) ->
  BandGather.Domain<geometry,Default.limits(),Machine.capacity(Default.limits())>:
  evidence(geometry,count,channels_positive,height_positive,width_positive,Count.count_of(extents(geometry),count,accepted))

# The whole-tensor envelope makes the actual native allocation count exact.
# Image count may be zero; positive spatial factors bound partial products.
def grouped(+images: U32,+channels: U32,+height: U32,+width: U32) ->
  {Nat.mul(Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height)),U32.to_nat(width)) ==
    BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat}:
  Equal.trans(Nat,Nat.mul(Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height)),U32.to_nat(width)),
    Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),Nat.mul(U32.to_nat(height),U32.to_nat(width))),BandGather.elements(Bands.Geometry{images,channels,height,width}),
    RowMajor.assoc(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height),U32.to_nat(width)),
    RowMajor.assoc(U32.to_nat(images),U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))))

def exact(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {U32.to_nat(Bands.elements(geometry)) == BandGather.elements(geometry) : Nat}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandGather.Domain{capacity_ok,channels_positive,+height_positive,+width_positive,+machine_room,allocation_room} = domain
  +prefix = Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height))
  +total = Nat.mul(prefix,U32.to_nat(width))
  +whole_room = {
    %Equal.sym(Nat,total,BandGather.elements(Bands.Geometry{images,channels,height,width}),grouped(images,channels,height,width)) :
      {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    machine_room : {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +prefix_room = Order.transitive(prefix,total,U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(prefix,U32.to_nat(width),width_positive),whole_room)
  +prefix_exact = Encoded.multiply_three(images,channels,height,Machine.capacity(limits),Limits.capacity_fits(limits),height_positive,prefix_room)
  Equal.trans(Nat,U32.to_nat(Bands.elements(Bands.Geometry{images,channels,height,width})),total,BandGather.elements(Bands.Geometry{images,channels,height,width}),
    %prefix_exact : {U32.to_nat((images * channels * height * width : U32)) == Nat.mul(_,U32.to_nat(width)) : Nat}
    Arithmetic.multiply((images * channels * height : U32),width,Machine.capacity(limits),Limits.capacity_fits(limits),
      %Equal.sym(Nat,U32.to_nat((images * channels * height : U32)),prefix,prefix_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      whole_room),
    grouped(images,channels,height,width))

def encoded(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {Bands.elements(geometry) == U32.from_nat(BandGather.elements(geometry)) : U32}:
  Encoded.encoding(Bands.elements(geometry),BandGather.elements(geometry),exact(geometry,limits,capacity,domain))

def room(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {Nat.is_le(BandGather.elements(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  machine_room

def within(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>) ->
  {U32.is_le(Bands.elements(geometry),Machine.capacity(limits)) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(Bands.elements(geometry),Machine.capacity(limits)),Nat.is_le(U32.to_nat(Bands.elements(geometry)),U32.to_nat(Machine.capacity(limits))),Comparison.le(Bands.elements(geometry),Machine.capacity(limits))) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Bands.elements(geometry)),BandGather.elements(geometry),exact(geometry,limits,capacity,domain)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  room(geometry,limits,capacity,domain)

def within_upper(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+upper: U32,domain: BandGather.Domain<geometry,limits,capacity>,
  bounded: {U32.is_le(Machine.capacity(limits),upper) == True{} : Bool}) -> {U32.is_le(Bands.elements(geometry),upper) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(Bands.elements(geometry),upper),Nat.is_le(U32.to_nat(Bands.elements(geometry)),U32.to_nat(upper)),Comparison.le(Bands.elements(geometry),upper)) : {_ == True{} : Bool}
  Order.transitive(U32.to_nat(Bands.elements(geometry)),U32.to_nat(Machine.capacity(limits)),U32.to_nat(upper),
    Bounds.as_nat_le(Bands.elements(geometry),Machine.capacity(limits),within(geometry,limits,capacity,domain)),Bounds.as_nat_le(Machine.capacity(limits),upper,bounded))

def allocatable(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {U32.is_le(Bands.elements(geometry),1073741824) == True{} : Bool}:
  within_upper(geometry,limits,capacity,1073741824,domain,Limits.capacity_allocatable(limits))

def element_count_counted(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+count: U32,domain: BandGather.Domain<geometry,limits,capacity>,
  count_exact: {BandGather.elements(geometry) == U32.to_nat(count) : Nat}) -> {Bands.elements(geometry) == count : U32}:
  Equal.trans(U32,Bands.elements(geometry),U32.from_nat(BandGather.elements(geometry)),count,encoded(geometry,limits,capacity,domain),
    Equal.trans(U32,U32.from_nat(BandGather.elements(geometry)),U32.from_nat(U32.to_nat(count)),count,
      Equal.cong(Nat,U32,U32.from_nat,BandGather.elements(geometry),U32.to_nat(count),count_exact),Arithmetic.native_roundtrip(count)))

# The actual public band-shape guard retains every CHW/NCHW axis. Equal
# element products alone cannot establish that two band geometries agree.
def decode(shape: List<&2,U32>) -> Maybe<&2,Bands.Geometry>:
  match shape:
    case [channels,height,width]: Some{Bands.Geometry{1,channels,height,width}}
    case [images,channels,height,width]: Some{Bands.Geometry{images,channels,height,width}}
    case shape: None{}

def decoded(+shape: List<&2,U32>,+geometry: Bands.Geometry,+matched: {Window.band_shape(shape,geometry) == True{} : Bool}) ->
  {decode(shape) == Some{geometry} : Maybe<&2,Bands.Geometry>}:
  match shape geometry:
    case Con{+c,Con{+h,Con{+w,Nil{}}}} Bands.Geometry{+images,+channels,+height,+width}:
      +prefix = Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched)
      +pair = Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix)
      %Equal.sym(U32,images,1,ShapeLaws.native_equal(images,1,Boolean.left(U32.is_eq(images,1),U32.is_eq(c,channels),pair))) :
        {Some{Bands.Geometry{1,c,h,w}} == Some{Bands.Geometry{_,channels,height,width}} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,c,channels,ShapeLaws.native_equal(c,channels,Boolean.right(U32.is_eq(images,1),U32.is_eq(c,channels),pair))) :
        {Some{Bands.Geometry{1,_,h,w}} == Some{Bands.Geometry{1,channels,height,width}} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,h,height,ShapeLaws.native_equal(h,height,Boolean.right((U32.is_eq(images,1) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix))) :
        {Some{Bands.Geometry{1,channels,_,w}} == Some{Bands.Geometry{1,channels,height,width}} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,w,width,ShapeLaws.native_equal(w,width,Boolean.right((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched))) :
        {Some{Bands.Geometry{1,channels,height,_}} == Some{Bands.Geometry{1,channels,height,width}} : Maybe<&2,Bands.Geometry>}
      {==}
    case Con{+n,Con{+c,Con{+h,Con{+w,Nil{}}}}} Bands.Geometry{+images,+channels,+height,+width}:
      +prefix = Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched)
      +pair = Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix)
      %Equal.sym(U32,n,images,ShapeLaws.native_equal(n,images,Boolean.left(U32.is_eq(n,images),U32.is_eq(c,channels),pair))) :
        {Some{Bands.Geometry{_,c,h,w}} == Some{geometry} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,c,channels,ShapeLaws.native_equal(c,channels,Boolean.right(U32.is_eq(n,images),U32.is_eq(c,channels),pair))) :
        {Some{Bands.Geometry{images,_,h,w}} == Some{geometry} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,h,height,ShapeLaws.native_equal(h,height,Boolean.right((U32.is_eq(n,images) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix))) :
        {Some{Bands.Geometry{images,channels,_,w}} == Some{geometry} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,w,width,ShapeLaws.native_equal(w,width,Boolean.right((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched))) :
        {Some{Bands.Geometry{images,channels,height,_}} == Some{geometry} : Maybe<&2,Bands.Geometry>}
      {==}
    case Nil{} Bands.Geometry{n,c,h,w}: Empty.absurd({decode(shape) == Some{geometry} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(matched))
    case Con{a,Nil{}} Bands.Geometry{n,c,h,w}: Empty.absurd({decode(shape) == Some{geometry} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(matched))
    case Con{a,Con{b,Nil{}}} Bands.Geometry{n,c,h,w}: Empty.absurd({decode(shape) == Some{geometry} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(matched))
    case Con{a,Con{b,Con{c,Con{d,Con{e,rest}}}}} Bands.Geometry{n,channels,h,w}: Empty.absurd({decode(shape) == Some{geometry} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(matched))

def ElementProperty(+shape: List<&2,U32>,result: Maybe<&2,Bands.Geometry>) -> Data:
  match result:
    case None{}: Unit
    case Some{geometry}: {ShapeLaws.elements(shape) == BandGather.elements(geometry) : Nat}

def decoded_elements(+shape: List<&2,U32>) -> ElementProperty(shape,decode(shape)):
  match shape:
    case Con{+c,Con{+h,Con{+w,Nil{}}}}:
      +shape_coherence_elements = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w)))
      Equal.trans(Nat,ShapeLaws.elements([c,h,w]),shape_coherence_elements,BandGather.elements(Bands.Geometry{1,c,h,w}),
        Equal.cong(Nat,Nat,last => Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),last)),Nat.mul(U32.to_nat(w),1n),U32.to_nat(w),ShapeLaws.times_one(U32.to_nat(w))),
        Equal.sym(Nat,BandGather.elements(Bands.Geometry{1,c,h,w}),shape_coherence_elements,Addition.zero_right(shape_coherence_elements)))
    case Con{+n,Con{+c,Con{+h,Con{+w,Nil{}}}}}: domain_elements(Bands.Geometry{n,c,h,w})
    case Nil{}: Unit{}
    case Con{a,Nil{}}: Unit{}
    case Con{a,Con{b,Nil{}}}: Unit{}
    case Con{a,Con{b,Con{c,Con{d,Con{e,rest}}}}}: Unit{}

def shape_coherence_elements(+shape: List<&2,U32>,+geometry: Bands.Geometry,matched: {Window.band_shape(shape,geometry) == True{} : Bool}) ->
  {ShapeLaws.elements(shape) == BandGather.elements(geometry) : Nat}:
  %decoded(shape,geometry,matched) : ElementProperty(shape,_)
  decoded_elements(shape)

def value(result: Maybe<&2,Bands.Geometry>) -> Bands.Geometry:
  match result:
    case None{}: Bands.Geometry{0,0,0,0}
    case Some{geometry}: geometry

def unique(+shape: List<&2,U32>,+left: Bands.Geometry,+right: Bands.Geometry,
  left_matched: {Window.band_shape(shape,left) == True{} : Bool},right_matched: {Window.band_shape(shape,right) == True{} : Bool}) -> {left == right : Bands.Geometry}:
  Equal.cong(Maybe<&2,Bands.Geometry>,Bands.Geometry,value,Some{left},Some{right},
    Equal.trans(Maybe<&2,Bands.Geometry>,Some{left},decode(shape),Some{right},Equal.sym(Maybe<&2,Bands.Geometry>,decode(shape),Some{left},decoded(shape,left,left_matched)),decoded(shape,right,right_matched)))

# Zero channels, rows or width make the actual modular local count zero.
# These producers need no positive-width or whole-tensor indexing envelope.
def multiply_zero_right(+value: U32) -> {(value * 0 : U32) == 0 : U32}:
  Equal.trans(U32,(value * 0 : U32),U32.from_nat(Nat.mul(U32.to_nat(value),0n)),0,
    Multiplication.native_multiply(value,0),
    Equal.cong(Nat,U32,U32.from_nat,Nat.mul(U32.to_nat(value),0n),0n,RowMajor.multiply_zero_right(U32.to_nat(value))))

def width_native(+images: U32,+channels: U32,+height: U32,+rows: U32) ->
  {Bands.local_count(Bands.Geometry{images,channels,height,0},rows) == 0 : U32}:
  multiply_zero_right((channels * rows : U32))

def channels_native(+images: U32,+height: U32,+width: U32,+rows: U32) ->
  {Bands.local_count(Bands.Geometry{images,0,height,width},rows) == 0 : U32}:
  %Equal.sym(U32,(0 * rows : U32),0,Arithmetic.multiply_zero_left(rows)) : {(_ * width : U32) == 0 : U32}
  Arithmetic.multiply_zero_left(width)

def from_zero(+geometry: Bands.Geometry,+rows: U32,+length: Nat,
  +native_zero: {Bands.local_count(geometry,rows) == 0 : U32},length_zero: {length == 0n : Nat}) -> BandShape.Count<geometry,rows,length>:
  BandShape.Count{
    Equal.trans(Nat,U32.to_nat(Bands.local_count(geometry,rows)),0n,length,
      Equal.cong(U32,Nat,U32.to_nat,Bands.local_count(geometry,rows),0,native_zero),Equal.sym(Nat,length,0n,length_zero)),
    %Equal.sym(U32,Bands.local_count(geometry,rows),0,native_zero) : {U32.is_le(_,1073741824) == True{} : Bool}
    {==}}

def width(~Element: Data,+images: U32,+channels: U32,+height: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,0},first,rows,Buffer.length(~Element,input))) ->
  BandShape.Count<Bands.Geometry{images,channels,height,0},rows,Buffer.length(~Element,input)>:
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  from_zero(Bands.Geometry{images,channels,height,0},rows,Buffer.length(~Element,input),width_native(images,channels,height,rows),
    Equal.trans(Nat,Buffer.length(~Element,input),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),0n)),0n,length_exact,
      Equal.trans(Nat,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),0n)),Nat.mul(U32.to_nat(channels),0n),0n,
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(channels),value),Nat.mul(U32.to_nat(rows),0n),0n,RowMajor.multiply_zero_right(U32.to_nat(rows))),
        RowMajor.multiply_zero_right(U32.to_nat(channels)))))

def channels(~Element: Data,+images: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  metadata: BandShape.Leaf(Bands.Geometry{images,0,height,width},first,rows,Buffer.length(~Element,input))) ->
  BandShape.Count<Bands.Geometry{images,0,height,width},rows,Buffer.length(~Element,input)>:
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  from_zero(Bands.Geometry{images,0,height,width},rows,Buffer.length(~Element,input),channels_native(images,height,width,rows),length_exact)
