# Gather from bands to dense NCHW storage: copy facts, bounds, coverage, the
# content projection and the values it reads.
import Base
import ./tensor_band_copy.bend as BandCopy
import ../tensor_band.bend as Bands
import ./nat_order.bend as Order
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_band_model.bend as Model
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./storage_shape.bend as Shape
import ./tensor_band_inspection.bend as Inspection
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./storage_certificate.bend as Certificate
import ./strided_range_bounds.bend as Strided
import ./tensor_band_coordinates.bend as BandCoordinates
import ./row_major_coordinates.bend as RowCoordinates
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./u32_division.bend as U32Division
import ./nat_division_order.bend as DivisionOrder
import ./binary_radix.bend as Radix
import ./boolean_logic.bend as Boolean
import ./contiguous_copy.bend as Contiguous
import ./natural_addition.bend as Addition
import ./storage_address_bounds.bend as Address
import ./storage_copy_refinement.bend as StorageCopyLaws
import ./tensor_band_plan.bend as BandPlan
import ./tensor_contiguous_bounds.bend as ContiguousBounds
import ./tensor_shape_refinement.bend as ShapeLaws
import ../storage_buffer.bend as Storage

# Discharge the concrete gather copy-region obligations from NCHW geometry.
def base(image: Nat,channels: Nat,height: Nat,width: Nat) -> Nat:
  Nat.mul(Nat.mul(image,channels),Nat.mul(height,width))

def bounds_region(+image: U32,+first: U32,+rows: U32,+channels: U32,+height: U32,+width: U32,capacity: U32) -> BandCopy.Region:
  BandCopy.Region{base(U32.to_nat(image),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)),Nat.mul(U32.to_nat(first),U32.to_nat(width)),
    U32.to_nat(channels),(rows * width : U32),(height * width : U32),((image * channels * height + first) * width : U32),capacity}

def image_room(+image: Nat,+images: Nat,+channels: Nat,+height: Nat,+width: Nat,+capacity: Nat,
  inside: {Nat.is_lt(image,images) == True{} : Bool},room: {Nat.is_le(Nat.mul(images,Nat.mul(channels,Nat.mul(height,width))),capacity) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(base(image,channels,height,width),Nat.mul(channels,Nat.mul(height,width))),capacity) == True{} : Bool}:
  %Equal.sym(Nat,base(image,channels,height,width),Nat.mul(image,Nat.mul(channels,Nat.mul(height,width))),RowMajor.assoc(image,channels,Nat.mul(height,width))) :
    {Nat.is_le(Nat.add(_,Nat.mul(channels,Nat.mul(height,width))),capacity) == True{} : Bool}
  Strided.end_inside(0n,0n,Nat.mul(channels,Nat.mul(height,width)),Nat.mul(channels,Nat.mul(height,width)),images,image,capacity,inside,
    Order.reflexive(Nat.mul(channels,Nat.mul(height,width))),room)

def start_natural(+image: Nat,+first: Nat,+channels: Nat,+height: Nat,+width: Nat) ->
  {BandCoordinates.global_parts(image,first,0n,0n,channels,height,width) == Nat.add(base(image,channels,height,width),Nat.mul(first,width)) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.mul(image,channels),0n),Nat.mul(image,channels),Addition.zero_right(Nat.mul(image,channels))) :
    {RowMajor.index(_,height,width,first,0n) == Nat.add(base(image,channels,height,width),Nat.mul(first,width)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.mul(first,width),0n),Nat.mul(first,width),Addition.zero_right(Nat.mul(first,width))) :
    {Nat.add(base(image,channels,height,width),_) == Nat.add(base(image,channels,height,width),Nat.mul(first,width)) : Nat}
  {==}

def start_expression(+image: U32,+first: U32,+channels: U32,+height: U32,+width: U32) ->
  {(((image * channels + 0) * height + first) * width + 0 : U32) == ((image * channels * height + first) * width : U32) : U32}:
  Equal.trans(U32,(((image * channels + 0) * height + first) * width + 0 : U32),(((image * channels + 0) * height + first) * width : U32),((image * channels * height + first) * width : U32),
    Arithmetic.add_zero((((image * channels + 0) * height + first) * width : U32)),
    Equal.cong(U32,U32,value => ((value * height + first) * width : U32),(image * channels + 0 : U32),(image * channels : U32),Arithmetic.add_zero((image * channels : U32))))

def start_exact(+image: U32,+first: U32,+channels: U32,+height: U32,+width: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(base(U32.to_nat(image),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)),Nat.mul(U32.to_nat(first),U32.to_nat(width))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(((image * channels * height + first) * width : U32)) == Nat.add(base(U32.to_nat(image),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)),Nat.mul(U32.to_nat(first),U32.to_nat(width))) : Nat}:
  +natural = start_natural(U32.to_nat(image),U32.to_nat(first),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width))
  %natural : {U32.to_nat(((image * channels * height + first) * width : U32)) == _ : Nat}
  %start_expression(image,first,channels,height,width) : {U32.to_nat(_) == BandCoordinates.global_parts(U32.to_nat(image),U32.to_nat(first),0n,0n,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) : Nat}
  BandCoordinates.native_parts(image,first,0,0,channels,height,width,capacity,capacity_ok,height_positive,width_positive,
    %Equal.sym(Nat,BandCoordinates.global_parts(U32.to_nat(image),U32.to_nat(first),0n,0n,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)),
      Nat.add(base(U32.to_nat(image),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)),Nat.mul(U32.to_nat(first),U32.to_nat(width))),natural) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} room)

def valid(+image: U32,+images: U32,+first: U32,+rows: U32,+channels: U32,+height: U32,+width: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  +image_inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},
  +row_span: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(capacity)) == True{} : Bool}) ->
  BandCopy.Valid<bounds_region(image,first,rows,channels,height,width,capacity)>:
  +plane = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +frame = Nat.mul(U32.to_nat(channels),plane)
  +base = base(U32.to_nat(image),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width))
  +offset = Nat.mul(U32.to_nat(first),U32.to_nat(width))
  +count = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +image_room = image_room(U32.to_nat(image),U32.to_nat(images),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),U32.to_nat(capacity),image_inside,total_room)
  +frame_room = Strided.count_in_stride(base,frame,U32.to_nat(capacity),image_room)
  +plane_frame = {
    %RowMajor.multiply_commutative(plane,U32.to_nat(channels)) : {Nat.is_le(plane,_) == True{} : Bool}
    RowMajor.product_covers(plane,U32.to_nat(channels),channels_positive) : {Nat.is_le(plane,frame) == True{} : Bool}}
  +plane_room = Order.transitive(plane,frame,U32.to_nat(capacity),plane_frame,frame_room)
  +span = Strided.scaled_span(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(height),U32.to_nat(width),row_span)
  +count_exact = Arithmetic.multiply(rows,width,capacity,capacity_ok,Order.transitive(count,plane,U32.to_nat(capacity),Strided.count_in_stride(offset,count,plane,span),plane_room))
  +plane_exact = Arithmetic.multiply(height,width,capacity,capacity_ok,plane_room)
  BandCopy.Valid{capacity_ok,
    Order.transitive(U32.to_nat(channels),frame,U32.to_nat(capacity),RowMajor.product_covers(U32.to_nat(channels),plane,RowMajor.multiply_lt(0n,U32.to_nat(height),U32.to_nat(width),height_positive,width_positive)),frame_room),
    start_exact(image,first,channels,height,width,capacity,capacity_ok,height_positive,width_positive,
      Order.transitive(Nat.add(base,offset),Nat.add(base,frame),U32.to_nat(capacity),Order.offset_le(base,offset,frame,
        Order.transitive(offset,plane,frame,Order.transitive(offset,Nat.add(offset,count),plane,Order.le_add(offset,count),span),plane_frame)),image_room)),
    %Equal.sym(Nat,U32.to_nat((rows * width : U32)),count,count_exact) : {Nat.is_le(Nat.add(offset,_),U32.to_nat((height * width : U32))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat((height * width : U32)),plane,plane_exact) : {Nat.is_le(Nat.add(offset,count),_) == True{} : Bool} span,
    %Equal.sym(Nat,U32.to_nat((height * width : U32)),plane,plane_exact) : {Nat.is_le(Nat.add(base,Nat.mul(U32.to_nat(channels),_)),U32.to_nat(capacity)) == True{} : Bool} image_room}

def actual_region(+first: U32,rows: U32,channels: U32,+height: U32,width: U32,capacity: U32) -> BandCopy.Region:
  bounds_region((first / height : U32),(first % height : U32),rows,channels,height,width,capacity)

# Allocation capacity and the division arithmetic envelope are separate. In a
# one-cell allocation 2*height can exceed capacity even though division is safe.
# Empty leaves are handled by the copy program's empty_writes theorem.
def bounds_actual(+images: U32,+first: U32,+rows: U32,+channels: U32,+height: U32,+width: U32,+capacity: U32,+division_capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+division_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(division_capacity)) == True{} : Bool},
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +row_span: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(capacity)) == True{} : Bool}) ->
  BandCopy.Valid<actual_region(first,rows,channels,height,width,capacity)>:
  +first_inside = Bounds.nat_trans(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height)),
    %Addition.zero_right(U32.to_nat(first)) : {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}
    Order.offset_lt(U32.to_nat(first),0n,U32.to_nat(rows),rows_positive),rows_fit)
  +image_inside = RowMajor.row_inside(Nat.div(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(images),U32.to_nat(height),Nat.mod(U32.to_nat(first),U32.to_nat(height)),
    %Radix.division_identity(first,U32.to_nat(height),height_positive) : {Nat.is_lt(_,Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool} first_inside)
  valid((first / height : U32),images,(first % height : U32),rows,channels,height,width,capacity,capacity_ok,channels_positive,height_positive,width_positive,
    %Equal.sym(Nat,U32.to_nat((first / height : U32)),Nat.div(U32.to_nat(first),U32.to_nat(height)),U32Division.divide(first,height,division_capacity,division_ok,height_bound,height_positive)) :
      {Nat.is_lt(_,U32.to_nat(images)) == True{} : Bool} image_inside,
    %Equal.sym(Nat,U32.to_nat((first % height : U32)),Nat.mod(U32.to_nat(first),U32.to_nat(height)),U32Division.modulo(first,height,division_capacity,division_ok,height_bound,height_positive)) :
      {Nat.is_le(Nat.add(_,U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool}
    Inspection.local_span(height,first,rows,division_capacity,division_ok,height_positive,height_bound,row_span),total_room)

def channel_address(+image: Nat,+first: Nat,+channels: Nat,+height: Nat,+width: Nat,+channel: Nat,+index: Nat) ->
  {Nat.add(Strided.start(base(image,channels,height,width),Nat.mul(first,width),Nat.mul(height,width),channel),index) ==
    BandCoordinates.global_parts(image,first,channel,index,channels,height,width) : Nat}:
  +base = base(image,channels,height,width)
  +plane = Nat.mul(height,width)
  +offset = Nat.mul(first,width)
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,channels),channel),plane),Nat.add(base,Nat.mul(channel,plane)),RowMajor.distrib(Nat.mul(image,channels),channel,plane)) :
    {Nat.add(Strided.start(base,offset,plane,channel),index) == Nat.add(_,Nat.add(offset,index)) : Nat}
  Equal.trans(Nat,Nat.add(Strided.start(base,offset,plane,channel),index),Nat.add(base,Nat.add(Nat.mul(channel,plane),Nat.add(offset,index))),
    Nat.add(Nat.add(base,Nat.mul(channel,plane)),Nat.add(offset,index)),Strided.end_form(base,offset,plane,channel,index),
    Equal.sym(Nat,Nat.add(Nat.add(base,Nat.mul(channel,plane)),Nat.add(offset,index)),Nat.add(base,Nat.add(Nat.mul(channel,plane),Nat.add(offset,index))),Addition.associative(base,Nat.mul(channel,plane),Nat.add(offset,index))))

def query_parts(+image: U32,+first: U32,+rows: U32,+channels: U32,+height: U32,+width: U32,+capacity: U32,+channel: Nat,+index: Nat,
  +valid: BandCopy.Valid<bounds_region(image,first,rows,channels,height,width,capacity)>,
  +stride_exact: {U32.to_nat((height * width : U32)) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+index_inside: {Nat.is_lt(index,U32.to_nat((rows * width : U32))) == True{} : Bool}) ->
  {U32.to_nat(BandCopy.readback_query(bounds_region(image,first,rows,channels,height,width,capacity),channel,index)) ==
    BandCoordinates.global_parts(U32.to_nat(image),U32.to_nat(first),channel,index,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) : Nat}:
  +bounds_region = bounds_region(image,first,rows,channels,height,width,capacity)
  Equal.trans(Nat,U32.to_nat(BandCopy.readback_query(bounds_region,channel,index)),Nat.add(BandCopy.start(bounds_region,channel),index),
    BandCoordinates.global_parts(U32.to_nat(image),U32.to_nat(first),channel,index,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)),
    BandCopy.readback_query_exact(bounds_region,channel,index,valid,channel_inside,index_inside),
    %Equal.sym(Nat,U32.to_nat((height * width : U32)),Nat.mul(U32.to_nat(height),U32.to_nat(width)),stride_exact) :
      {Nat.add(Strided.start(base(U32.to_nat(image),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)),Nat.mul(U32.to_nat(first),U32.to_nat(width)),_,channel),index) ==
        BandCoordinates.global_parts(U32.to_nat(image),U32.to_nat(first),channel,index,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) : Nat}
    channel_address(U32.to_nat(image),U32.to_nat(first),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),channel,index))

# The source address of a channel copy and the public indexed-map address are
# the same element. This connects the common value relation to gather writes.
def element_address(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+position: U32,+capacity: U32,+division_capacity: U32,+channel: Nat,+index: Nat,
  +valid: BandCopy.Valid<actual_region(first,rows,channels,height,width,capacity)>,
  +stride_exact: {U32.to_nat((height * width : U32)) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +position_exact: {U32.to_nat(position) == Nat.add(Nat.mul(channel,U32.to_nat((rows * width : U32))),index) : Nat},
  +division_ok: {U32.is_le(division_capacity,2147483648) == True{} : Bool},
  +plane_bound: {Nat.is_le(Nat.double(U32.to_nat((rows * width : U32))),U32.to_nat(division_capacity)) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+index_inside: {Nat.is_lt(index,U32.to_nat((rows * width : U32))) == True{} : Bool}) ->
  {Bands.global_index(Bands.Geometry{images,channels,height,width},first,rows,position) == BandCopy.readback_query(actual_region(first,rows,channels,height,width,capacity),channel,index) : U32}:
  +plane = (rows * width : U32)
  +image = (first / height : U32)
  +local_first = (first % height : U32)
  +quotient = (position / plane : U32)
  +remainder = (position % plane : U32)
  +bounds_region = actual_region(first,rows,channels,height,width,capacity)
  +query = BandCopy.readback_query(bounds_region,channel,index)
  +positive = Order.le_lt_trans(0n,index,U32.to_nat(plane),Order.zero_le(index),index_inside)
  +quotient_exact = Radix.quotient_word(position,plane,channel,index,U32.to_nat(plane),division_capacity,division_ok,position_exact,{==},positive,index_inside,plane_bound)
  +remainder_exact = Radix.remainder_word(position,plane,channel,index,U32.to_nat(plane),division_capacity,division_ok,position_exact,{==},positive,index_inside,plane_bound)
  +parts = BandCoordinates.global_parts(U32.to_nat(image),U32.to_nat(local_first),U32.to_nat(quotient),U32.to_nat(remainder),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width))
  +parts_query = {
    %Equal.sym(Nat,U32.to_nat(quotient),channel,quotient_exact) :
      {BandCoordinates.global_parts(U32.to_nat(image),U32.to_nat(local_first),_,U32.to_nat(remainder),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) == U32.to_nat(query) : Nat}
    %Equal.sym(Nat,U32.to_nat(remainder),index,remainder_exact) :
      {BandCoordinates.global_parts(U32.to_nat(image),U32.to_nat(local_first),channel,_,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) == U32.to_nat(query) : Nat}
    Equal.sym(Nat,U32.to_nat(query),BandCoordinates.global_parts(U32.to_nat(image),U32.to_nat(local_first),channel,index,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)),
      query_parts(image,local_first,rows,channels,height,width,capacity,channel,index,valid,stride_exact,channel_inside,index_inside)) : {parts == U32.to_nat(query) : Nat}}
  Address.equal_values(Bands.global_index(Bands.Geometry{images,channels,height,width},first,rows,position),query,
    Equal.trans(Nat,U32.to_nat(Bands.global_index(Bands.Geometry{images,channels,height,width},first,rows,position)),parts,U32.to_nat(query),
      BandCoordinates.native_parts(image,local_first,quotient,remainder,channels,height,width,capacity,BandCopy.readback_capacity_ok(bounds_region,valid),height_positive,width_positive,
        %Equal.sym(Nat,parts,U32.to_nat(query),parts_query) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        Bounds.strict_le(U32.to_nat(query),U32.to_nat(capacity),Bounds.as_nat_lt(query,capacity,BandCopy.query_word_inside(bounds_region,channel,index,valid,channel_inside,index_inside)))),parts_query))

# The public gather's exact content projection. The model is a sequence of
# channel range writes; certificates and affine ownership are erased only here.
def output(~Element: Data,owners: StorageCopyLaws.Owners<Element>) -> Buffer.Content<Element>:
  StorageCopyLaws.Owners{source,source_length,output,output_length} = owners
  Buffer.Content{output,output_length}

def owners(~Element: Data,source: Buffer.Content<Element>,output: Buffer.Content<Element>) -> StorageCopyLaws.Owners<Element>:
  Buffer.Content{source,source_length} = source
  Buffer.Content{output,output_length} = output
  StorageCopyLaws.Owners{source,source_length,output,output_length}

def destination(~Element: Data,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Bands.destination(~Element,pair)) == output(~Element,StorageCopyLaws.observe(~Element,pair)) : Buffer.Content<Element>}:
  (Storage.Buffer{source,source_length,source_certificate},Storage.Buffer{output,output_length,output_certificate}) = pair
  {==}

def leaf_model(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,input: Buffer.Content<Element>,result: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Bands.Geometry{batch,+channels,+height,+width} = geometry
  output(~Element,BandCopy.model(~Element,U32.to_nat(channels),0,(rows * width : U32),(rows * width : U32),(height * width : U32),0,
    ((first / height * channels * height + first % height) * width : U32),owners(~Element,input,result)))

def pair_contents(~Element: Data,+input: Buffer.Model<Element>,+result: Buffer.Model<Element>) ->
  {StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,input),Buffer.pack(~Element,result))) == owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,result)) : StorageCopyLaws.Owners<Element>}:
  Buffer.Model{input,input_length,input_certificate} = input
  Buffer.Model{result,result_length,result_certificate} = result
  StorageCopyLaws.represented_values(~Element,input,input_length,result,result_length)

def leaf(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+result: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Bands.gather_leaf(~Element,geometry,first,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,result))) ==
    leaf_model(~Element,geometry,first,rows,Buffer.content(~Element,input),Buffer.content(~Element,result)) : Buffer.Content<Element>}:
  Bands.Geometry{batch,+channels,+height,+width} = geometry
  +count = (rows * width : U32)
  +stride = (height * width : U32)
  +start = ((first / height * channels * height + first % height) * width : U32)
  %Equal.sym(Buffer.Content<Element>,
    Buffer.contents(~Element,Bands.destination(~Element,Bands.planes(~Element,U32.to_nat(channels),0,count,count,stride,0,start,(Buffer.pack(~Element,input),Buffer.pack(~Element,result))))),
    output(~Element,StorageCopyLaws.observe(~Element,Bands.planes(~Element,U32.to_nat(channels),0,count,count,stride,0,start,(Buffer.pack(~Element,input),Buffer.pack(~Element,result))))),
    destination(~Element,Bands.planes(~Element,U32.to_nat(channels),0,count,count,stride,0,start,(Buffer.pack(~Element,input),Buffer.pack(~Element,result))))) :
    {_ == leaf_model(~Element,Bands.Geometry{batch,channels,height,width},first,rows,Buffer.content(~Element,input),Buffer.content(~Element,result)) : Buffer.Content<Element>}
  %Equal.sym(StorageCopyLaws.Owners<Element>,
    StorageCopyLaws.observe(~Element,Bands.planes(~Element,U32.to_nat(channels),0,count,count,stride,0,start,(Buffer.pack(~Element,input),Buffer.pack(~Element,result)))),
    BandCopy.model(~Element,U32.to_nat(channels),0,count,count,stride,0,start,StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,input),Buffer.pack(~Element,result)))),
    BandCopy.planes(~Element,U32.to_nat(channels),0,count,count,stride,0,start,(Buffer.pack(~Element,input),Buffer.pack(~Element,result)))) :
    {output(~Element,_) == leaf_model(~Element,Bands.Geometry{batch,channels,height,width},first,rows,Buffer.content(~Element,input),Buffer.content(~Element,result)) : Buffer.Content<Element>}
  Equal.cong(StorageCopyLaws.Owners<Element>,Buffer.Content<Element>,pair => output(~Element,BandCopy.model(~Element,U32.to_nat(channels),0,count,count,stride,0,start,pair)),
    StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,input),Buffer.pack(~Element,result))),owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,result)),
    pair_contents(~Element,input,result))

def model(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>,result: Buffer.Content<Element>) -> Buffer.Content<Element>:
  match bands:
    case Model.Leaf{input,first,rows}: leaf_model(~Element,geometry,first,rows,Buffer.content(~Element,input),result)
    case Model.Fork{left,right}: model(~Element,geometry,right,model(~Element,geometry,left,result))

type Context<-Element: Data> is Data:
  Context{geometry: Bands.Geometry,bands: Model.Model<Element>}

def proposition(~Element: Data,context: Context<Element>,-result: Storage.Buffer<Element>) -> Type:
  Context{+geometry,+bands} = context
  {Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),result)) ==
    model(~Element,geometry,bands,Buffer.contents(~Element,result)) : Buffer.Content<Element>}

def gather(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>,result: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),result)) ==
    model(~Element,geometry,bands,Buffer.contents(~Element,result)) : Buffer.Content<Element>}:
  match bands:
    case Model.Leaf{+input,+first,+rows}:
      Buffer.lift(~Element,~Context<Element>,~proposition(~Element),Context{geometry,Model.Leaf{input,first,rows}},result,Buffer.observe(~Element,result),
        result =>
          +result = result
          Equal.trans(Buffer.Content<Element>,
            Buffer.contents(~Element,Bands.gather_leaf(~Element,geometry,first,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,result))),
            leaf_model(~Element,geometry,first,rows,Buffer.content(~Element,input),Buffer.content(~Element,result)),
            leaf_model(~Element,geometry,first,rows,Buffer.content(~Element,input),Buffer.contents(~Element,Buffer.pack(~Element,result))),
            leaf(~Element,geometry,first,rows,input,result),
            Equal.cong(Buffer.Content<Element>,Buffer.Content<Element>,output => leaf_model(~Element,geometry,first,rows,Buffer.content(~Element,input),output),
              Buffer.content(~Element,result),Buffer.contents(~Element,Buffer.pack(~Element,result)),
              Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,result)),Buffer.content(~Element,result),Buffer.packed_contents(~Element,result)))))
    case Model.Fork{+left,+right}:
      Buffer.lift(~Element,~Context<Element>,~proposition(~Element),Context{geometry,Model.Fork{left,right}},result,Buffer.observe(~Element,result),
        result =>
          +result = result
          Equal.trans(Buffer.Content<Element>,
            Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,right),Bands.gather(~Element,geometry,Model.pack(~Element,left),Buffer.pack(~Element,result)))),
            model(~Element,geometry,right,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,left),Buffer.pack(~Element,result)))),
            model(~Element,geometry,right,model(~Element,geometry,left,Buffer.contents(~Element,Buffer.pack(~Element,result)))),
            gather(~Element,geometry,right,Bands.gather(~Element,geometry,Model.pack(~Element,left),Buffer.pack(~Element,result))),
            Equal.cong(Buffer.Content<Element>,Buffer.Content<Element>,output => model(~Element,geometry,right,output),
              Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,left),Buffer.pack(~Element,result))),
              model(~Element,geometry,left,Buffer.contents(~Element,Buffer.pack(~Element,result))),
              gather(~Element,geometry,left,Buffer.pack(~Element,result)))))

# Every native tree is connected to the canonical theorem by an equality
# produced from its refinement_actual owners. No assumed layout/readback premise.
type Refinement<-Element: Data,-geometry: Bands.Geometry,-bands: Bands.Bands(~Element),-result: Storage.Buffer<Element>> is Data:
  Refinement{representation: Model.Model<Element>,
    owners: {Model.pack(~Element,representation) == bands : Bands.Bands(~Element)},
    execution: {Buffer.contents(~Element,Bands.gather(~Element,geometry,bands,result)) ==
      model(~Element,geometry,representation,Buffer.contents(~Element,result)) : Buffer.Content<Element>}}

def observed(~Element: Data,+geometry: Bands.Geometry,-bands: Bands.Bands(~Element),result: Storage.Buffer<Element>,observation: Model.Observation<Element,bands>) ->
  Refinement<Element,geometry,bands,result>:
  Model.Observation{+representation,+owners} = observation
  Refinement{representation,owners,
    %owners : {Buffer.contents(~Element,Bands.gather(~Element,geometry,_,result)) ==
      model(~Element,geometry,representation,Buffer.contents(~Element,result)) : Buffer.Content<Element>}
    gather(~Element,geometry,representation,result)}

def refinement_actual(~Element: Data,geometry: Bands.Geometry,bands: Bands.Bands(~Element),result: Storage.Buffer<Element>) -> Refinement<Element,geometry,bands,result>:
  observed(~Element,geometry,bands,result,Model.observe(~Element,bands))

def leaf_writes(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  Bands.Geometry{images,+channels,+height,+width} = geometry
  BandCopy.writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(rows * width : U32),(height * width : U32),0,
    ((first / height * channels * height + first % height) * width : U32),input)

def refinement_writes(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>) -> List<&2,Writes.Assignment<Element>>:
  match bands:
    case Model.Leaf{input,first,rows}: leaf_writes(~Element,geometry,first,rows,Buffer.tree(~Element,input))
    case Model.Fork{left,right}: List.append(&2,Writes.Assignment<Element>,refinement_writes(~Element,geometry,left),refinement_writes(~Element,geometry,right))

def leaf_model_writes(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+input_length: Nat,+result: Tree.Tree<Element>,+length: Nat) ->
  {leaf_model(~Element,geometry,first,rows,Buffer.Content{input,input_length},Buffer.Content{result,length}) ==
    Buffer.Content{Writes.pure(Element,leaf_writes(~Element,geometry,first,rows,input),result),length} : Buffer.Content<Element>}:
  Bands.Geometry{images,+channels,+height,+width} = geometry
  Equal.cong(StorageCopyLaws.Owners<Element>,Buffer.Content<Element>,pair => output(~Element,pair),
    BandCopy.model(~Element,U32.to_nat(channels),0,(rows * width : U32),(rows * width : U32),(height * width : U32),0,
      ((first / height * channels * height + first % height) * width : U32),StorageCopyLaws.Owners{input,input_length,result,length}),
    StorageCopyLaws.Owners{input,input_length,Writes.pure(Element,BandCopy.writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(rows * width : U32),(height * width : U32),0,
      ((first / height * channels * height + first % height) * width : U32),input),result),length},
    BandCopy.model_writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(rows * width : U32),(height * width : U32),0,
      ((first / height * channels * height + first % height) * width : U32),input,input_length,result,length))

def model_writes(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>,+result: Tree.Tree<Element>,+length: Nat) ->
  {model(~Element,geometry,bands,Buffer.Content{result,length}) == Buffer.Content{Writes.pure(Element,refinement_writes(~Element,geometry,bands),result),length} : Buffer.Content<Element>}:
  match bands:
    case Model.Leaf{Buffer.Model{+input,+input_length,certificate},+first,+rows}:
      leaf_model_writes(~Element,geometry,first,rows,input,input_length,result,length)
    case Model.Fork{+left,+right}:
      %Equal.sym(Buffer.Content<Element>,model(~Element,geometry,left,Buffer.Content{result,length}),Buffer.Content{Writes.pure(Element,refinement_writes(~Element,geometry,left),result),length},
        model_writes(~Element,geometry,left,result,length)) :
        {model(~Element,geometry,right,_) == Buffer.Content{Writes.pure(Element,refinement_writes(~Element,geometry,bands),result),length} : Buffer.Content<Element>}
      %Equal.sym(Tree.Tree<Element>,Writes.pure(Element,List.append(&2,Writes.Assignment<Element>,refinement_writes(~Element,geometry,left),refinement_writes(~Element,geometry,right)),result),
        Writes.pure(Element,refinement_writes(~Element,geometry,right),Writes.pure(Element,refinement_writes(~Element,geometry,left),result)),
        Writes.concatenated_writes(Element,refinement_writes(~Element,geometry,left),refinement_writes(~Element,geometry,right),result)) :
        {model(~Element,geometry,right,Buffer.Content{Writes.pure(Element,refinement_writes(~Element,geometry,left),result),length}) == Buffer.Content{_,length} : Buffer.Content<Element>}
      model_writes(~Element,geometry,right,Writes.pure(Element,refinement_writes(~Element,geometry,left),result),length)

def gathered_writes(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+result: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))) ==
    Buffer.Content{Writes.pure(Element,refinement_writes(~Element,geometry,bands),Buffer.tree(~Element,result)),Buffer.length(~Element,result)} : Buffer.Content<Element>}:
  Buffer.Model{+tree,+length,+certificate} = result
  Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,Buffer.Model{tree,length,certificate}))),
    model(~Element,geometry,bands,Buffer.contents(~Element,Buffer.pack(~Element,Buffer.Model{tree,length,certificate}))),
    Buffer.Content{Writes.pure(Element,refinement_writes(~Element,geometry,bands),tree),length},
    gather(~Element,geometry,bands,Buffer.pack(~Element,Buffer.Model{tree,length,certificate})),
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,Buffer.Model{tree,length,certificate})),Buffer.Content{tree,length},
      Buffer.packed_contents(~Element,Buffer.Model{tree,length,certificate})) :
      {model(~Element,geometry,bands,_) == Buffer.Content{Writes.pure(Element,refinement_writes(~Element,geometry,bands),tree),length} : Buffer.Content<Element>}
    model_writes(~Element,geometry,bands,tree,length))

# Gather copies preserve the shared pointwise content relation. The source
# word and destination word are derived from the actual channel-copy schedule.
type Call is Data:
  Call{values_geometry: Bands.Geometry,values_first: U32,values_rows: U32,length: Nat,capacity: U32,division_capacity: U32}

def values_geometry(call: Call) -> Bands.Geometry:
  Call{values_geometry,values_first,values_rows,length,capacity,division_capacity} = call
  values_geometry

def values_channels(call: Call) -> U32:
  Call{Bands.Geometry{images,values_channels,values_height,values_width},values_first,values_rows,length,capacity,division_capacity} = call
  values_channels

def values_height(call: Call) -> U32:
  Call{Bands.Geometry{images,values_channels,values_height,values_width},values_first,values_rows,length,capacity,division_capacity} = call
  values_height

def values_width(call: Call) -> U32:
  Call{Bands.Geometry{images,values_channels,values_height,values_width},values_first,values_rows,length,capacity,division_capacity} = call
  values_width

def values_first(call: Call) -> U32:
  Call{values_geometry,values_first,values_rows,length,capacity,division_capacity} = call
  values_first

def values_rows(call: Call) -> U32:
  Call{values_geometry,values_first,values_rows,length,capacity,division_capacity} = call
  values_rows

def length(call: Call) -> Nat:
  Call{values_geometry,values_first,values_rows,length,capacity,division_capacity} = call
  length

def capacity(call: Call) -> U32:
  Call{values_geometry,values_first,values_rows,length,capacity,division_capacity} = call
  capacity

def division_capacity(call: Call) -> U32:
  Call{values_geometry,values_first,values_rows,length,capacity,division_capacity} = call
  division_capacity

def plane(+call: Call) -> U32:
  (values_rows(call) * values_width(call) : U32)

def values_region(+call: Call) -> BandCopy.Region:
  actual_region(values_first(call),values_rows(call),values_channels(call),values_height(call),values_width(call),capacity(call))

type Facts<-call: Call> is Data:
  Facts{valid: BandCopy.Valid<values_region(call)>,
    stride_exact: {U32.to_nat((values_height(call) * values_width(call) : U32)) == Nat.mul(U32.to_nat(values_height(call)),U32.to_nat(values_width(call))) : Nat},
    length_exact: {length(call) == Nat.mul(U32.to_nat(values_channels(call)),U32.to_nat(plane(call))) : Nat},
    length_bound: {Nat.is_le(length(call),U32.to_nat(division_capacity(call))) == True{} : Bool},
    division_ok: {U32.is_le(division_capacity(call),2147483648) == True{} : Bool},
    plane_bound: {Nat.is_le(Nat.double(U32.to_nat(plane(call))),U32.to_nat(division_capacity(call))) == True{} : Bool},
    height_positive: {Nat.is_lt(0n,U32.to_nat(values_height(call))) == True{} : Bool},
    width_positive: {Nat.is_lt(0n,U32.to_nat(values_width(call))) == True{} : Bool},
    values_count_exact: {U32.to_nat(plane(call)) == Nat.mul(U32.to_nat(values_rows(call)),U32.to_nat(values_width(call))) : Nat},
    height_bound: {Nat.is_le(Nat.double(U32.to_nat(values_height(call))),U32.to_nat(division_capacity(call))) == True{} : Bool},
    row_span: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(values_first(call)),U32.to_nat(values_height(call))),U32.to_nat(values_rows(call))),U32.to_nat(values_height(call))) == True{} : Bool}}

def values_count_exact(+call: Call,facts: Facts<call>) -> {U32.to_nat(plane(call)) == Nat.mul(U32.to_nat(values_rows(call)),U32.to_nat(values_width(call))) : Nat}:
  Facts{valid,stride_exact,length_exact,length_bound,division_ok,plane_bound,height_positive,width_positive,values_count_exact,height_bound,row_span} = facts
  values_count_exact

def Leaf(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,context: Context,+call: Call,input: Tree.Tree<Element>) -> Data:
  BandValues.Leaf(~Element,~Context,~value,context,values_geometry(call),values_first(call),values_rows(call),input,length(call))

def point(geometry: Bands.Geometry,image: Nat,channel: Nat,row: Nat,column: Nat) -> Nat:
  BandValues.point(geometry,image,channel,row,column)

def global_row(geometry: Bands.Geometry,image: Nat,row: Nat) -> Nat:
  BandValues.global_row(geometry,image,row)

def leaf_address(+call: Call,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,+local: Nat,facts: Facts<call>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(values_channels(call))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(values_height(call))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(values_width(call))) == True{} : Bool},
  +local_inside: {Nat.is_lt(local,U32.to_nat(values_rows(call))) == True{} : Bool},
  located: {global_row(values_geometry(call),image,row) == Nat.add(U32.to_nat(values_first(call)),local) : Nat}) ->
  {BandCopy.readback_query(values_region(call),channel,Nat.add(Nat.mul(local,U32.to_nat(values_width(call))),column)) == U32.from_nat(point(values_geometry(call),image,channel,row,column)) : U32}:
  Call{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,length,+capacity,+division_capacity} = call
  Facts{+valid,+stride_exact,length_exact,length_bound,+division_ok,plane_bound,+height_positive,width_positive,+count_exact,+height_bound,+row_span} = facts
  +coordinates = BandCoordinates.row_coordinates(U32.to_nat(first),U32.to_nat(height),U32.to_nat(rows),local,image,row,height_positive,row_span,local_inside,row_inside,located)
  +image_equal = Shape.first({Nat.div(U32.to_nat(first),U32.to_nat(height)) == image : Nat},{Nat.add(Nat.mod(U32.to_nat(first),U32.to_nat(height)),local) == row : Nat},coordinates)
  +row_equal = Shape.second({Nat.div(U32.to_nat(first),U32.to_nat(height)) == image : Nat},{Nat.add(Nat.mod(U32.to_nat(first),U32.to_nat(height)),local) == row : Nat},coordinates)
  +native_image = (first / height : U32)
  +native_first = (first % height : U32)
  +image_exact = Equal.trans(Nat,U32.to_nat(native_image),Nat.div(U32.to_nat(first),U32.to_nat(height)),image,
    U32Division.divide(first,height,division_capacity,division_ok,height_bound,height_positive),image_equal)
  +first_exact = U32Division.modulo(first,height,division_capacity,division_ok,height_bound,height_positive)
  +index = Nat.add(Nat.mul(local,U32.to_nat(width)),column)
  +index_inside = {
    %Equal.sym(Nat,U32.to_nat((rows * width : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),count_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(local,U32.to_nat(rows),column,U32.to_nat(width),local_inside,column_inside) : {Nat.is_lt(index,U32.to_nat((rows * width : U32))) == True{} : Bool}}
  +query = BandCopy.readback_query(actual_region(first,rows,channels,height,width,capacity),channel,index)
  Encoded.encoding(query,point(Bands.Geometry{images,channels,height,width},image,channel,row,column),
    Equal.trans(Nat,U32.to_nat(query),BandCoordinates.global_parts(U32.to_nat(native_image),U32.to_nat(native_first),channel,index,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)),
      point(Bands.Geometry{images,channels,height,width},image,channel,row,column),
      query_parts(native_image,native_first,rows,channels,height,width,capacity,channel,index,valid,stride_exact,channel_inside,index_inside),
      %Equal.sym(Nat,U32.to_nat(native_image),image,image_exact) :
        {BandCoordinates.global_parts(_,U32.to_nat(native_first),channel,index,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) == point(Bands.Geometry{images,channels,height,width},image,channel,row,column) : Nat}
      %Equal.sym(Nat,U32.to_nat(native_first),Nat.mod(U32.to_nat(first),U32.to_nat(height)),first_exact) :
        {BandCoordinates.global_parts(image,_,channel,index,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) == point(Bands.Geometry{images,channels,height,width},image,channel,row,column) : Nat}
      %row_equal : {BandCoordinates.global_parts(image,Nat.mod(U32.to_nat(first),U32.to_nat(height)),channel,index,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) ==
        RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),U32.to_nat(height),U32.to_nat(width),_,column) : Nat}
      BandCoordinates.row_join(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),U32.to_nat(height),U32.to_nat(width),Nat.mod(U32.to_nat(first),U32.to_nat(height)),local,column)))

def copied_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,
  +call: Call,+input: Tree.Tree<Element>,+channel: Nat,+index: Nat,values: Leaf(~Element,~Context,~value,context,call,input),facts: Facts<call>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(values_channels(call))) == True{} : Bool},+index_inside: {Nat.is_lt(index,U32.to_nat(plane(call))) == True{} : Bool}) ->
  {BandCopy.sample(~Element,input,plane(call),0,channel,index) == value(context,BandCopy.readback_query(values_region(call),channel,index)) : Element}:
  Call{Bands.Geometry{+images,+values_channels,+values_height,+values_width},+values_first,+values_rows,+length,+capacity,+division_capacity} = call
  Facts{+valid,+stride_exact,+length_exact,+length_bound,+division_ok,+plane_bound,+height_positive,+width_positive,+values_count_exact,+height_bound,+row_span} = facts
  +plane = (values_rows * values_width : U32)
  +local = Nat.div(index,U32.to_nat(values_width))
  +column = Nat.mod(index,U32.to_nat(values_width))
  +index_exact = DivisionOrder.quotient_exact(index,U32.to_nat(values_width),width_positive)
  +index_room = {%values_count_exact : {Nat.is_lt(index,_) == True{} : Bool} index_inside : {Nat.is_lt(index,Nat.mul(U32.to_nat(values_rows),U32.to_nat(values_width))) == True{} : Bool}}
  +source_region = BandCopy.whole_region(U32.to_nat(values_channels),plane,division_capacity)
  +source_valid = BandCopy.whole_valid(U32.to_nat(values_channels),plane,division_capacity,division_ok,
    Order.le_lt_trans(0n,index,U32.to_nat(plane),Order.zero_le(index),index_inside),
    %length_exact : {Nat.is_le(_,U32.to_nat(division_capacity)) == True{} : Bool} length_bound)
  +position = BandCopy.readback_query(source_region,channel,index)
  +position_nat = Nat.add(Nat.mul(channel,U32.to_nat(plane)),index)
  +position_exact = BandCopy.readback_query_exact(source_region,channel,index,source_valid,channel_inside,index_inside)
  %Equal.sym(Nat,index,Nat.add(Nat.mul(local,U32.to_nat(values_width)),column),index_exact) :
    {Tree.read(Element,input,position) == value(context,BandCopy.readback_query(actual_region(values_first,values_rows,values_channels,values_height,values_width,capacity),channel,_)) : Element}
  %Equal.sym(U32,BandCopy.readback_query(actual_region(values_first,values_rows,values_channels,values_height,values_width,capacity),channel,Nat.add(Nat.mul(local,U32.to_nat(values_width)),column)),U32.from_nat(BandValues.pixel(Bands.Geometry{images,values_channels,values_height,values_width},channel,Nat.add(U32.to_nat(values_first),local),column)),
    leaf_address(Call{Bands.Geometry{images,values_channels,values_height,values_width},values_first,values_rows,length,capacity,division_capacity},Nat.div(Nat.add(U32.to_nat(values_first),local),U32.to_nat(values_height)),channel,Nat.mod(Nat.add(U32.to_nat(values_first),local),U32.to_nat(values_height)),column,local,
      Facts{valid,stride_exact,length_exact,length_bound,division_ok,plane_bound,height_positive,width_positive,values_count_exact,height_bound,row_span},
      channel_inside,DivisionOrder.remainder_small(Nat.add(U32.to_nat(values_first),local),U32.to_nat(values_height),height_positive),DivisionOrder.remainder_small(index,U32.to_nat(values_width),width_positive),
      RowMajor.row_inside(local,U32.to_nat(values_rows),U32.to_nat(values_width),column,%index_exact : {Nat.is_lt(_,Nat.mul(U32.to_nat(values_rows),U32.to_nat(values_width))) == True{} : Bool} index_room),
      Equal.sym(Nat,Nat.add(U32.to_nat(values_first),local),Nat.add(Nat.mul(Nat.div(Nat.add(U32.to_nat(values_first),local),U32.to_nat(values_height)),U32.to_nat(values_height)),Nat.mod(Nat.add(U32.to_nat(values_first),local),U32.to_nat(values_height))),DivisionOrder.quotient_exact(Nat.add(U32.to_nat(values_first),local),U32.to_nat(values_height),height_positive)))) :
    {Tree.read(Element,input,position) == value(context,_) : Element}
  %Equal.sym(U32,position,U32.from_nat(position_nat),Encoded.encoding(position,position_nat,position_exact)) :
    {Tree.read(Element,input,_) == value(context,U32.from_nat(BandValues.pixel(Bands.Geometry{images,values_channels,values_height,values_width},channel,Nat.add(U32.to_nat(values_first),local),column))) : Element}
  %Equal.sym(Nat,U32.to_nat(plane),Nat.mul(U32.to_nat(values_rows),U32.to_nat(values_width)),values_count_exact) :
    {Tree.read(Element,input,U32.from_nat(Nat.add(Nat.mul(channel,_),index))) == value(context,U32.from_nat(BandValues.pixel(Bands.Geometry{images,values_channels,values_height,values_width},channel,Nat.add(U32.to_nat(values_first),local),column))) : Element}
  BandValues.leaf_plane_read(~Element,~Context,~value,context,Bands.Geometry{images,values_channels,values_height,values_width},values_first,values_rows,input,length,channel,index,values,channel_inside,index_room,width_positive)

def span_writes(~Element: Data,+call: Call,remaining: Nat,+channel: Nat,position: Nat,input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  Contiguous.writes(~Element,remaining,input,U32.from_nat(position),(0 + U32.from_nat(channel) * plane(call) : U32),BandCopy.native_start(values_region(call),channel))

def matching_span(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,remaining: Nat,+context: Context,+call: Call,+input: Tree.Tree<Element>,+channel: Nat,+position: Nat,+query: U32,
  +values: Leaf(~Element,~Context,~value,context,call,input),+facts: Facts<call>,+channel_inside: {Nat.is_lt(channel,U32.to_nat(values_channels(call))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(plane(call))) == True{} : Bool}) ->
  Readback.Matching(~Element,query,value(context,query),span_writes(~Element,call,remaining,channel,position,input)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{
        Readback.value_matches(~Element,~Context,~value,context,BandCopy.readback_query(values_region(call),channel,position),query,BandCopy.sample(~Element,input,plane(call),0,channel,position),
          U32.is_eq(BandCopy.readback_query(values_region(call),channel,position),query),{==},
          copied_value(~Element,~Context,~value,context,call,input,channel,position,values,facts,channel_inside,Order.room_index(position,rest,U32.to_nat(plane(call)),room))),
        %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
          Readback.Matching(~Element,query,value(context,query),Contiguous.writes(~Element,rest,input,_,(0 + U32.from_nat(channel) * plane(call) : U32),BandCopy.native_start(values_region(call),channel)))
        matching_span(~Element,~Context,~value,rest,context,call,input,channel,1n+position,query,values,facts,channel_inside,Order.advance_room(position,rest,U32.to_nat(plane(call)),room))}

def values_writes(~Element: Data,+call: Call,remaining: Nat,channel: U32,input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  BandCopy.writes(~Element,remaining,channel,plane(call),plane(call),BandCopy.stride(values_region(call)),0,BandCopy.destination(values_region(call)),input)

def matching_planes(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,remaining: Nat,+context: Context,+call: Call,+input: Tree.Tree<Element>,+channel: Nat,+query: U32,
  +values: Leaf(~Element,~Context,~value,context,call,input),+facts: Facts<call>,+room: {Nat.is_le(Nat.add(channel,remaining),U32.to_nat(values_channels(call))) == True{} : Bool}) ->
  Readback.Matching(~Element,query,value(context,query),values_writes(~Element,call,remaining,U32.from_nat(channel),input)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Readback.matching_append(~Element,query,value(context,query),
        BandCopy.plane_writes(~Element,U32.from_nat(channel),plane(call),plane(call),BandCopy.stride(values_region(call)),0,BandCopy.destination(values_region(call)),input),
        values_writes(~Element,call,rest,(U32.from_nat(channel) + 1 : U32),input),
        matching_span(~Element,~Context,~value,U32.to_nat(plane(call)),context,call,input,channel,0n,query,values,facts,Order.room_index(channel,rest,U32.to_nat(values_channels(call)),room),Order.reflexive(U32.to_nat(plane(call)))),
        %Equal.sym(U32,(U32.from_nat(channel) + 1 : U32),U32.from_nat(1n+channel),Bounds.add_one(U32.from_nat(channel))) :
          Readback.Matching(~Element,query,value(context,query),values_writes(~Element,call,rest,_,input))
        matching_planes(~Element,~Context,~value,rest,context,call,input,1n+channel,query,values,facts,Order.advance_room(channel,rest,U32.to_nat(values_channels(call)),room)))

def TreeFacts(~Element: Data,+values_geometry: Bands.Geometry,+capacity: U32,+division_capacity: U32,bands: Model.Model<Element>) -> Data:
  match bands:
    case Model.Leaf{buffer,values_first,values_rows}:
      Either<&2,&2,{plane(Call{values_geometry,values_first,values_rows,Buffer.length(~Element,buffer),capacity,division_capacity}) == 0 : U32},
        Facts<Call{values_geometry,values_first,values_rows,Buffer.length(~Element,buffer),capacity,division_capacity}>>
    case Model.Fork{left,right}: Shape.Both<TreeFacts(~Element,values_geometry,capacity,division_capacity,left),TreeFacts(~Element,values_geometry,capacity,division_capacity,right)>

def empty_program(~Element: Data,+call: Call,+input: Tree.Tree<Element>,empty: {plane(call) == 0 : U32}) ->
  {values_writes(~Element,call,U32.to_nat(values_channels(call)),0,input) == [] : List<&2,Writes.Assignment<Element>>}:
  %Equal.sym(U32,plane(call),0,empty) :
    {BandCopy.writes(~Element,U32.to_nat(values_channels(call)),0,_,_,BandCopy.stride(values_region(call)),0,BandCopy.destination(values_region(call)),input) == [] : List<&2,Writes.Assignment<Element>>}
  BandCopy.empty_writes(~Element,U32.to_nat(values_channels(call)),0,0,BandCopy.stride(values_region(call)),0,BandCopy.destination(values_region(call)),input)

def leaf_matching(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+call: Call,+input: Tree.Tree<Element>,+query: U32,
  values: Leaf(~Element,~Context,~value,context,call,input),facts: Either<&2,&2,{plane(call) == 0 : U32},Facts<call>>) ->
  Readback.Matching(~Element,query,value(context,query),values_writes(~Element,call,U32.to_nat(values_channels(call)),0,input)):
  match facts:
    case Inl{empty}:
      %Equal.sym(List<&2,Writes.Assignment<Element>>,values_writes(~Element,call,U32.to_nat(values_channels(call)),0,input),[],empty_program(~Element,call,input,empty)) :
        Readback.Matching(~Element,query,value(context,query),_)
      Unit{}
    case Inr{facts}: matching_planes(~Element,~Context,~value,U32.to_nat(values_channels(call)),context,call,input,0n,query,values,facts,Order.reflexive(U32.to_nat(values_channels(call))))

def whole_matching(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,bands: Model.Model<Element>,+context: Context,+values_geometry: Bands.Geometry,+capacity: U32,+division_capacity: U32,+query: U32,
  values: BandValues.Values(~Element,~Context,~value,context,values_geometry,Model.content(~Element,bands)),facts: TreeFacts(~Element,values_geometry,capacity,division_capacity,bands)) ->
  Readback.Matching(~Element,query,value(context,query),refinement_writes(~Element,values_geometry,bands)):
  match bands values_geometry:
    case Model.Leaf{Buffer.Model{input,length,certificate},values_first,values_rows} Bands.Geometry{images,values_channels,values_height,values_width}:
      leaf_matching(~Element,~Context,~value,context,Call{Bands.Geometry{images,values_channels,values_height,values_width},values_first,values_rows,length,capacity,division_capacity},input,query,values,facts)
    case Model.Fork{+left,+right} +values_geometry:
      Shape.Both{left_values,right_values} = values
      Shape.Both{left_facts,right_facts} = facts
      Readback.matching_append(~Element,query,value(context,query),refinement_writes(~Element,values_geometry,left),refinement_writes(~Element,values_geometry,right),
        whole_matching(~Element,~Context,~value,left,context,values_geometry,capacity,division_capacity,query,left_values,left_facts),
        whole_matching(~Element,~Context,~value,right,context,values_geometry,capacity,division_capacity,query,right_values,right_facts))

def facts_inside(~Element: Data,+call: Call,+input: Tree.Tree<Element>,facts: Facts<call>) ->
  Readback.Inside(~Element,capacity(call),values_writes(~Element,call,U32.to_nat(values_channels(call)),0,input)):
  Call{Bands.Geometry{images,+values_channels,+values_height,+values_width},+values_first,+values_rows,length,+capacity,division_capacity} = call
  Facts{valid,stride_exact,length_exact,length_bound,division_ok,plane_bound,height_positive,width_positive,values_count_exact,height_bound,row_span} = facts
  BandCopy.writes_inside(~Element,U32.to_nat(values_channels),0n,actual_region(values_first,values_rows,values_channels,values_height,values_width,capacity),(values_rows * values_width : U32),0,input,valid,Order.reflexive(U32.to_nat(values_channels)))

def leaf_inside(~Element: Data,+call: Call,+input: Tree.Tree<Element>,facts: Either<&2,&2,{plane(call) == 0 : U32},Facts<call>>) ->
  Readback.Inside(~Element,capacity(call),values_writes(~Element,call,U32.to_nat(values_channels(call)),0,input)):
  match facts:
    case Inl{empty}:
      %Equal.sym(List<&2,Writes.Assignment<Element>>,values_writes(~Element,call,U32.to_nat(values_channels(call)),0,input),[],empty_program(~Element,call,input,empty)) :
        Readback.Inside(~Element,capacity(call),_)
      Unit{}
    case Inr{facts}: facts_inside(~Element,call,input,facts)

def whole_inside(~Element: Data,bands: Model.Model<Element>,+values_geometry: Bands.Geometry,+capacity: U32,+division_capacity: U32,
  facts: TreeFacts(~Element,values_geometry,capacity,division_capacity,bands)) -> Readback.Inside(~Element,capacity,refinement_writes(~Element,values_geometry,bands)):
  match bands values_geometry:
    case Model.Leaf{Buffer.Model{input,length,certificate},values_first,values_rows} Bands.Geometry{images,values_channels,values_height,values_width}:
      leaf_inside(~Element,Call{Bands.Geometry{images,values_channels,values_height,values_width},values_first,values_rows,length,capacity,division_capacity},input,facts)
    case Model.Fork{+left,+right} +values_geometry:
      Shape.Both{left_facts,right_facts} = facts
      Readback.inside_append(~Element,capacity,refinement_writes(~Element,values_geometry,left),refinement_writes(~Element,values_geometry,right),
        whole_inside(~Element,left,values_geometry,capacity,division_capacity,left_facts),whole_inside(~Element,right,values_geometry,capacity,division_capacity,right_facts))

# Build gather's copy facts from the actual inspected leaf and machine bounds.
# The image count is allowed to be zero; a nonempty leaf itself establishes
# that its image exists. Empty leaves use the separate zero-store branch.
def channels_positive(geometry: Bands.Geometry) -> Bool:
  Bands.Geometry{images,channels,height,width} = geometry
  Nat.is_lt(0n,U32.to_nat(channels))

def height_positive(geometry: Bands.Geometry) -> Bool:
  Bands.Geometry{images,channels,height,width} = geometry
  Nat.is_lt(0n,U32.to_nat(height))

def width_positive(geometry: Bands.Geometry) -> Bool:
  Bands.Geometry{images,channels,height,width} = geometry
  Nat.is_lt(0n,U32.to_nat(width))

def elements(geometry: Bands.Geometry) -> Nat:
  Bands.Geometry{images,channels,height,width} = geometry
  Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))))

type Domain<-geometry: Bands.Geometry,-limits: Machine.Limits,-capacity: U32> is Data:
  Domain{capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    channels_positive: {channels_positive(geometry) == True{} : Bool},height_positive: {height_positive(geometry) == True{} : Bool},
    width_positive: {width_positive(geometry) == True{} : Bool},
    machine_room: {Nat.is_le(elements(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    allocation_room: {Nat.is_le(elements(geometry),U32.to_nat(capacity)) == True{} : Bool}}

def Inspected(geometry: Bands.Geometry,first: U32,rows: U32,length: Nat,depth: Nat) -> Data:
  Bands.Geometry{images,channels,height,width} = geometry
  Inspection.LeafFacts<images,channels,height,width,first,rows,length,depth>

def inspected(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,+depth: Nat,
  accepted: {Inspection.valid(Bands.leaf_inspection(geometry,length,depth,first,rows)) == True{} : Bool}) -> Inspected(geometry,first,rows,length,depth):
  Bands.Geometry{images,channels,height,width} = geometry
  Inspection.leaf_facts(images,channels,height,width,first,rows,length,depth,accepted)

def from_leaf_facts(+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+length: Nat,+depth: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +machine_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +allocation_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(capacity)) == True{} : Bool},
  inspected: Inspection.LeafFacts<images,channels,height,width,first,rows,length,depth>) ->
  Facts<Call{Bands.Geometry{images,channels,height,width},first,rows,length,capacity,Machine.half_word(limits)}>:
  Inspection.LeafFacts{+local,+rows_fit,+length_exact,allocation_fit} = inspected
  +row_total = Nat.mul(U32.to_nat(images),U32.to_nat(height))
  +image_positive = RowMajor.left_factor_positive(U32.to_nat(images),U32.to_nat(height),
    Bounds.nat_trans(0n,U32.to_nat(rows),row_total,rows_positive,Strided.count_in_stride(U32.to_nat(first),U32.to_nat(rows),row_total,rows_fit)))
  +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +frame = Nat.mul(U32.to_nat(channels),spatial)
  +total = Nat.mul(U32.to_nat(images),frame)
  +machine_capacity = Machine.capacity(limits)
  +frame_room = Order.transitive(frame,total,U32.to_nat(machine_capacity),
    %RowMajor.multiply_commutative(frame,U32.to_nat(images)) : {Nat.is_le(frame,_) == True{} : Bool}
    RowMajor.product_covers(frame,U32.to_nat(images),image_positive),machine_room)
  +spatial_frame = {
    %RowMajor.multiply_commutative(spatial,U32.to_nat(channels)) : {Nat.is_le(spatial,_) == True{} : Bool}
    RowMajor.product_covers(spatial,U32.to_nat(channels),channels_positive) : {Nat.is_le(spatial,frame) == True{} : Bool}}
  +spatial_room = Order.transitive(spatial,frame,U32.to_nat(machine_capacity),spatial_frame,frame_room)
  +height_room = Order.transitive(U32.to_nat(height),spatial,U32.to_nat(machine_capacity),RowMajor.product_covers(U32.to_nat(height),U32.to_nat(width),width_positive),spatial_room)
  +height_double = Limits.double_bound(limits,U32.to_nat(height),height_room)
  +span = Inspection.local_span(height,first,rows,Machine.half_word(limits),Limits.half_word_fits(limits),height_positive,height_double,local)
  +rows_bound = Strided.count_in_stride(Nat.mod(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(rows),U32.to_nat(height),span)
  +count = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +count_spatial = RowMajor.mul_le(U32.to_nat(rows),U32.to_nat(height),U32.to_nat(width),rows_bound)
  +count_room = Order.transitive(count,spatial,U32.to_nat(machine_capacity),count_spatial,spatial_room)
  +count_exact = Arithmetic.multiply(rows,width,machine_capacity,Limits.capacity_fits(limits),count_room)
  +stride_exact = Arithmetic.multiply(height,width,machine_capacity,Limits.capacity_fits(limits),spatial_room)
  +length_room = {
    %Equal.sym(Nat,length,Nat.mul(U32.to_nat(channels),count),length_exact) : {Nat.is_le(_,U32.to_nat(machine_capacity)) == True{} : Bool}
    Order.transitive(Nat.mul(U32.to_nat(channels),count),frame,U32.to_nat(machine_capacity),
      %RowMajor.multiply_commutative(count,U32.to_nat(channels)) : {Nat.is_le(_,frame) == True{} : Bool}
      %RowMajor.multiply_commutative(spatial,U32.to_nat(channels)) : {Nat.is_le(Nat.mul(count,U32.to_nat(channels)),_) == True{} : Bool}
      RowMajor.mul_le(count,spatial,U32.to_nat(channels),count_spatial),frame_room) : {Nat.is_le(length,U32.to_nat(machine_capacity)) == True{} : Bool}}
  Facts{
    bounds_actual(images,first,rows,channels,height,width,capacity,Machine.half_word(limits),capacity_ok,Limits.half_word_fits(limits),height_double,
      channels_positive,height_positive,width_positive,rows_positive,local,rows_fit,allocation_room),
    stride_exact,
    %Equal.sym(Nat,U32.to_nat((rows * width : U32)),count,count_exact) : {length == Nat.mul(U32.to_nat(channels),_) : Nat} length_exact,
    Order.transitive(length,U32.to_nat(machine_capacity),U32.to_nat(Machine.half_word(limits)),length_room,
      Bounds.as_nat_le(machine_capacity,Machine.half_word(limits),Limits.capacity_within_half(limits))),
    Limits.half_word_fits(limits),
    Limits.double_bound(limits,U32.to_nat((rows * width : U32)),
      %Equal.sym(Nat,U32.to_nat((rows * width : U32)),count,count_exact) : {Nat.is_le(_,U32.to_nat(machine_capacity)) == True{} : Bool} count_room),
    height_positive,width_positive,count_exact,height_double,span}

def zero_plane(+rows: U32,+width: U32,empty: {U32.is_zero(rows) == True{} : Bool}) -> {(rows * width : U32) == 0 : U32}:
  %Equal.sym(U32,rows,0,ShapeLaws.native_equal(rows,0,empty)) : {(_ * width : U32) == 0 : U32}
  Arithmetic.multiply_zero_left(width)

def selected_leaf(empty: Bool,+limits: Machine.Limits,+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,+depth: Nat,+capacity: U32,
  compared: {U32.is_zero(rows) == empty : Bool},domain: Domain<geometry,limits,capacity>,facts: Inspected(geometry,first,rows,length,depth)) ->
  Either<&2,&2,{plane(Call{geometry,first,rows,length,capacity,Machine.half_word(limits)}) == 0 : U32},
    Facts<Call{geometry,first,rows,length,capacity,Machine.half_word(limits)}>>:
  match empty geometry domain:
    case True{} Bands.Geometry{images,channels,height,width} domain: Inl{zero_plane(rows,width,compared)}
    case False{} Bands.Geometry{images,channels,height,width} Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room}:
      Inr{from_leaf_facts(limits,images,channels,height,width,first,rows,length,depth,capacity,capacity_ok,channels_positive,height_positive,width_positive,
        ContiguousBounds.extent_positive(rows,compared),machine_room,allocation_room,facts)}

def leaf_checked(+limits: Machine.Limits,+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,+depth: Nat,+capacity: U32,
  domain: Domain<geometry,limits,capacity>,accepted: {Inspection.valid(Bands.leaf_inspection(geometry,length,depth,first,rows)) == True{} : Bool}) ->
  Either<&2,&2,{plane(Call{geometry,first,rows,length,capacity,Machine.half_word(limits)}) == 0 : U32},
    Facts<Call{geometry,first,rows,length,capacity,Machine.half_word(limits)}>>:
  selected_leaf(U32.is_zero(rows),limits,geometry,first,rows,length,depth,capacity,{==},domain,inspected(geometry,first,rows,length,depth,accepted))

def tree_checked(~Element: Data,bands: Model.Model<Element>,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,
  +domain: Domain<geometry,limits,capacity>,accepted: {Inspection.valid(Inspection.model(~Element,geometry,bands)) == True{} : Bool}) ->
  TreeFacts(~Element,geometry,capacity,Machine.half_word(limits),bands):
  match bands:
    case Model.Leaf{Buffer.Model{input,length,Certificate.Certificate{depth,equation}},first,rows}:
      leaf_checked(limits,geometry,first,rows,length,depth,capacity,domain,accepted)
    case Model.Fork{+left,+right}:
      +accepted = accepted
      Shape.Both{
        tree_checked(~Element,left,geometry,limits,capacity,domain,Inspection.joined_left(Inspection.model(~Element,geometry,left),Inspection.model(~Element,geometry,right),accepted)),
        tree_checked(~Element,right,geometry,limits,capacity,domain,Inspection.joined_right(Inspection.model(~Element,geometry,left),Inspection.model(~Element,geometry,right),accepted))}

# A row slice may occupy only part of an owner (concat uses a nonzero base).
# Its logical length is explicit and its allocation room is proved separately.
def from_slice(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+first: U32,+rows: U32,+length: Nat,+input: Buffer.Model<Element>,+capacity: U32,
  domain: Domain<geometry,limits,capacity>,metadata: BandShape.Leaf(geometry,first,rows,length),
  rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},room: {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,input)))) == True{} : Bool}) ->
  Facts<Call{geometry,first,rows,length,capacity,Machine.half_word(limits)}>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Buffer.Model{+tree,input_length,Certificate.Certificate{+depth,+equation}} = input
  Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  from_leaf_facts(limits,images,channels,height,width,first,rows,length,depth,capacity,capacity_ok,channels_positive,height_positive,width_positive,rows_positive,machine_room,allocation_room,
    Inspection.LeafFacts{local,rows_fit,length_exact,
      %Shape.capacity(Element,depth,tree,Buffer.balanced(~Element,input)) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool} room})

def from_metadata(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+capacity: U32,
  domain: Domain<geometry,limits,capacity>,metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),
  rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},ready: Buffer.Sized<Element,input>) ->
  Facts<Call{geometry,first,rows,Buffer.length(~Element,input),capacity,Machine.half_word(limits)}>:
  Buffer.Sized{depth_ok,room} = ready
  from_slice(~Element,geometry,limits,first,rows,Buffer.length(~Element,input),input,capacity,domain,metadata,rows_positive,room)

# Halo preparation exposes content and readiness, without an affine owner.
# The same inspected geometry follows directly from its balanced tree witness.
def from_content(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+first: U32,+rows: U32,+input: Buffer.Content<Element>,+capacity: U32,
  domain: Domain<geometry,limits,capacity>,metadata: BandShape.Leaf(geometry,first,rows,Buffer.content_length(~Element,input)),
  rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},ready: Buffer.ContentReady<Element,input>) ->
  Facts<Call{geometry,first,rows,Buffer.content_length(~Element,input),capacity,Machine.half_word(limits)}>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Buffer.Content{+tree,+length} = input
  Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  Buffer.ContentReady{+depth,balanced,depth_ok,room} = ready
  from_leaf_facts(limits,images,channels,height,width,first,rows,length,depth,capacity,capacity_ok,channels_positive,height_positive,width_positive,rows_positive,machine_room,allocation_room,
    Inspection.LeafFacts{local,rows_fit,length_exact,
      %Shape.capacity(Element,depth,tree,balanced) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool} room})

# A logical NCHW point located in a band names an actual gather assignment.
def leaf_covered(~Element: Data,+call: Call,+input: Tree.Tree<Element>,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,+local: Nat,+facts: Facts<call>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(values_channels(call))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(values_height(call))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(values_width(call))) == True{} : Bool},
  +local_inside: {Nat.is_lt(local,U32.to_nat(values_rows(call))) == True{} : Bool},
  +located: {global_row(values_geometry(call),image,row) == Nat.add(U32.to_nat(values_first(call)),local) : Nat}) ->
  {Readback.contains(~Element,U32.from_nat(point(values_geometry(call),image,channel,row,column)),values_writes(~Element,call,U32.to_nat(values_channels(call)),0,input)) == True{} : Bool}:
  +index = Nat.add(Nat.mul(local,U32.to_nat(values_width(call))),column)
  %leaf_address(call,image,channel,row,column,local,facts,channel_inside,row_inside,column_inside,local_inside,located) :
    {Readback.contains(~Element,_,values_writes(~Element,call,U32.to_nat(values_channels(call)),0,input)) == True{} : Bool}
  BandCopy.covered(~Element,U32.to_nat(values_channels(call)),0n,channel,index,values_region(call),plane(call),0,input,channel_inside,
    %Equal.sym(Nat,U32.to_nat(plane(call)),Nat.mul(U32.to_nat(values_rows(call)),U32.to_nat(values_width(call))),values_count_exact(call,facts)) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(local,U32.to_nat(values_rows(call)),column,U32.to_nat(values_width(call)),local_inside,column_inside))

type PointInside<-images: U32,-channels: U32,-height: U32,-width: U32,-image: Nat,-channel: Nat,-row: Nat,-column: Nat> is Data:
  PointInside{image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},
    row_inside: {Nat.is_lt(row,U32.to_nat(height)) == True{} : Bool},column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}}

def Inside(geometry: Bands.Geometry,image: Nat,channel: Nat,row: Nat,column: Nat) -> Data:
  Bands.Geometry{images,channels,height,width} = geometry
  PointInside<images,channels,height,width,image,channel,row,column>

def tree_covered(~Element: Data,bands: Model.Model<Element>,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+domain: Domain<geometry,limits,capacity>,
  +image: Nat,+channel: Nat,+row: Nat,+column: Nat,+inside: Inside(geometry,image,channel,row,column),
  +accepted: {Inspection.valid(Inspection.model(~Element,geometry,bands)) == True{} : Bool},location: BandPlan.Located(Model.layout(~Element,bands),global_row(geometry,image,row))) ->
  {Readback.contains(~Element,U32.from_nat(point(geometry,image,channel,row,column)),refinement_writes(~Element,geometry,bands)) == True{} : Bool}:
  match bands geometry domain:
    case Model.Leaf{Buffer.Model{+input,+length,Certificate.Certificate{+depth,equation}},+first,+rows} Bands.Geometry{+images,+channels,+height,+width} Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room}:
      PointInside{image_inside,+channel_inside,+row_inside,+column_inside} = inside
      BandPlan.LeafLocation{+local,+local_inside,+located} = location
      leaf_covered(~Element,Call{Bands.Geometry{images,channels,height,width},first,rows,length,capacity,Machine.half_word(limits)},input,image,channel,row,column,local,
        from_leaf_facts(limits,images,channels,height,width,first,rows,length,depth,capacity,capacity_ok,channels_positive,height_positive,width_positive,
          Order.le_lt_trans(0n,local,U32.to_nat(rows),Order.zero_le(local),local_inside),machine_room,allocation_room,
          Inspection.leaf_facts(images,channels,height,width,first,rows,length,depth,accepted)),channel_inside,row_inside,column_inside,local_inside,located)
    case Model.Fork{+left,+right} +geometry +domain:
      match location:
        case Inl{location}:
          Readback.covered_left(~Element,U32.from_nat(point(geometry,image,channel,row,column)),refinement_writes(~Element,geometry,left),refinement_writes(~Element,geometry,right),
            tree_covered(~Element,left,geometry,limits,capacity,domain,image,channel,row,column,inside,
              Inspection.joined_left(Inspection.model(~Element,geometry,left),Inspection.model(~Element,geometry,right),accepted),location))
        case Inr{location}:
          Readback.covered_right(~Element,U32.from_nat(point(geometry,image,channel,row,column)),refinement_writes(~Element,geometry,left),refinement_writes(~Element,geometry,right),
            tree_covered(~Element,right,geometry,limits,capacity,domain,image,channel,row,column,inside,
              Inspection.joined_right(Inspection.model(~Element,geometry,left),Inspection.model(~Element,geometry,right),accepted),location))

def rows_room(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: Domain<geometry,limits,capacity>) ->
  {Nat.is_le(Inspection.total_rows(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +frame = Nat.mul(U32.to_nat(channels),spatial)
  +height_frame = Order.transitive(U32.to_nat(height),spatial,frame,RowMajor.product_covers(U32.to_nat(height),U32.to_nat(width),width_positive),
    %RowMajor.multiply_commutative(spatial,U32.to_nat(channels)) : {Nat.is_le(spatial,_) == True{} : Bool}
    RowMajor.product_covers(spatial,U32.to_nat(channels),channels_positive))
  Order.transitive(Nat.mul(U32.to_nat(images),U32.to_nat(height)),Nat.mul(U32.to_nat(images),frame),U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(U32.to_nat(height),U32.to_nat(images)) : {Nat.is_le(_,Nat.mul(U32.to_nat(images),frame)) == True{} : Bool}
    %RowMajor.multiply_commutative(frame,U32.to_nat(images)) : {Nat.is_le(Nat.mul(U32.to_nat(height),U32.to_nat(images)),_) == True{} : Bool}
    RowMajor.mul_le(U32.to_nat(height),frame,U32.to_nat(images),height_frame),machine_room)

def point_row_inside(+geometry: Bands.Geometry,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,inside: Inside(geometry,image,channel,row,column)) ->
  {Nat.is_lt(global_row(geometry,image,row),Inspection.total_rows(geometry)) == True{} : Bool}:
  Bands.Geometry{images,channels,height,width} = geometry
  PointInside{image_inside,channel_inside,row_inside,column_inside} = inside
  RowMajor.radix(image,U32.to_nat(images),row,U32.to_nat(height),image_inside,row_inside)

def complete(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>) -> Bool:
  +inspection = Inspection.model(~Element,geometry,bands)
  Inspection.valid(inspection) && U32.is_zero(Inspection.first(inspection)) && Nat.is_eq(U32.to_nat(Inspection.end(inspection)),Inspection.total_rows(geometry))

def complete_valid(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,checked: {complete(~Element,geometry,bands) == True{} : Bool}) ->
  {Inspection.valid(Inspection.model(~Element,geometry,bands)) == True{} : Bool}:
  +inspection = Inspection.model(~Element,geometry,bands)
  +starts = U32.is_zero(Inspection.first(inspection))
  +ends = Nat.is_eq(U32.to_nat(Inspection.end(inspection)),Inspection.total_rows(geometry))
  Boolean.left(Inspection.valid(inspection),starts,Boolean.left((Inspection.valid(inspection) && starts : Bool),ends,checked))

def complete_coverage(~Element: Data,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+domain: Domain<geometry,limits,capacity>,
  +image: Nat,+channel: Nat,+row: Nat,+column: Nat,+inside: Inside(geometry,image,channel,row,column),+checked: {complete(~Element,geometry,bands) == True{} : Bool}) ->
  {Readback.contains(~Element,U32.from_nat(point(geometry,image,channel,row,column)),refinement_writes(~Element,geometry,bands)) == True{} : Bool}:
  tree_covered(~Element,bands,geometry,limits,capacity,domain,image,channel,row,column,inside,complete_valid(~Element,geometry,bands,checked),
    BandPlan.locate(Model.layout(~Element,bands),0n,Inspection.total_rows(geometry),global_row(geometry,image,row),
      Inspection.checked(~Element,geometry,bands,Machine.capacity(limits),Limits.capacity_fits(limits),rows_room(geometry,limits,capacity,domain),checked),
      point_row_inside(geometry,image,channel,row,column,inside)))

def query_inside(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,
  domain: Domain<geometry,limits,capacity>,inside: Inside(geometry,image,channel,row,column)) ->
  {U32.is_lt(U32.from_nat(point(geometry,image,channel,row,column)),capacity) == True{} : Bool}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Domain{+capacity_ok,channels_positive,height_positive,width_positive,machine_room,+allocation_room} = domain
  PointInside{image_inside,channel_inside,row_inside,column_inside} = inside
  +combined_channel = Nat.add(Nat.mul(image,U32.to_nat(channels)),channel)
  +index = RowMajor.index(combined_channel,U32.to_nat(height),U32.to_nat(width),row,column)
  +total = Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),Nat.mul(U32.to_nat(height),U32.to_nat(width)))
  +bound = Bounds.nat_trans(index,total,U32.to_nat(capacity),
    RowMajor.in_bounds(combined_channel,Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height),U32.to_nat(width),row,column,
      RowMajor.radix(image,U32.to_nat(images),channel,U32.to_nat(channels),image_inside,channel_inside),row_inside,column_inside),
    %Equal.sym(Nat,total,Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),RowMajor.assoc(U32.to_nat(images),U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} allocation_room)
  Bounds.native_bound(index,capacity,Bounds.roundtrip(index,capacity,capacity_ok,Bounds.strict_le(index,U32.to_nat(capacity),bound)),bound)

def indexed_leaf_address(+call: Call,+position: U32,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,+local: Nat,+facts: Facts<call>,
  position_exact: {U32.to_nat(position) == RowMajor.index(channel,U32.to_nat(values_rows(call)),U32.to_nat(values_width(call)),local,column) : Nat},
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(values_channels(call))) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(values_height(call))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(values_width(call))) == True{} : Bool},+local_inside: {Nat.is_lt(local,U32.to_nat(values_rows(call))) == True{} : Bool},
  located: {global_row(values_geometry(call),image,row) == Nat.add(U32.to_nat(values_first(call)),local) : Nat}) ->
  {Bands.global_index(values_geometry(call),values_first(call),values_rows(call),position) == U32.from_nat(point(values_geometry(call),image,channel,row,column)) : U32}:
  Call{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,+length,+capacity,+division_capacity} = call
  Facts{+valid,+stride_exact,length_exact,length_bound,+division_ok,+plane_bound,+height_positive,+width_positive,+count_exact,height_bound,row_span} = facts
  +index = Nat.add(Nat.mul(local,U32.to_nat(width)),column)
  +position_parts = {
    %Equal.sym(Nat,U32.to_nat((rows * width : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),count_exact) :
      {U32.to_nat(position) == Nat.add(Nat.mul(channel,_),index) : Nat} position_exact : {U32.to_nat(position) == Nat.add(Nat.mul(channel,U32.to_nat((rows * width : U32))),index) : Nat}}
  +index_inside = {
    %Equal.sym(Nat,U32.to_nat((rows * width : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),count_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(local,U32.to_nat(rows),column,U32.to_nat(width),local_inside,column_inside) : {Nat.is_lt(index,U32.to_nat((rows * width : U32))) == True{} : Bool}}
  Equal.trans(U32,Bands.global_index(values_geometry(call),first,rows,position),BandCopy.readback_query(values_region(call),channel,index),U32.from_nat(point(values_geometry(call),image,channel,row,column)),
    element_address(images,channels,height,width,first,rows,position,capacity,division_capacity,channel,index,valid,stride_exact,position_parts,division_ok,plane_bound,height_positive,width_positive,channel_inside,index_inside),
    leaf_address(call,image,channel,row,column,local,facts,channel_inside,row_inside,column_inside,local_inside,located))

# The domain bounds every element of the tensor by the capacity, so the index
# of a pixel of the batch is below it.
def pixel_inside(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+channel: Nat,+row: Nat,+column: Nat,domain: Domain<geometry,limits,capacity>,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},+row_inside: {Nat.is_lt(row,Inspection.total_rows(geometry)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool}) ->
  {Nat.is_lt(BandValues.pixel(geometry,channel,row,column),U32.to_nat(capacity)) == True{} : Bool}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Domain{capacity_ok,channels_positive,+height_positive,width_positive,machine_room,allocation_room} = domain
  +image = Nat.div(row,U32.to_nat(height))
  +local = Nat.mod(row,U32.to_nat(height))
  +combined_channel = Nat.add(Nat.mul(image,U32.to_nat(channels)),channel)
  +index = RowMajor.index(combined_channel,U32.to_nat(height),U32.to_nat(width),local,column)
  +total = Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),Nat.mul(U32.to_nat(height),U32.to_nat(width)))
  +image_inside = RowMajor.row_inside(image,U32.to_nat(images),U32.to_nat(height),local,
    %DivisionOrder.quotient_exact(row,U32.to_nat(height),height_positive) : {Nat.is_lt(_,Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool} row_inside)
  Bounds.nat_trans(index,total,U32.to_nat(capacity),
    RowMajor.in_bounds(combined_channel,Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height),U32.to_nat(width),local,column,
      RowMajor.radix(image,U32.to_nat(images),channel,U32.to_nat(channels),image_inside,channel_inside),DivisionOrder.remainder_small(row,U32.to_nat(height),height_positive),column_inside),
    %Equal.sym(Nat,total,Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),RowMajor.assoc(U32.to_nat(images),U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} allocation_room)

# The index of a pixel of the batch survives the machine word.
def pixel_exact(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+channel: Nat,+row: Nat,+column: Nat,+domain: Domain<geometry,limits,capacity>,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},+row_inside: {Nat.is_lt(row,Inspection.total_rows(geometry)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(BandValues.pixel(geometry,channel,row,column))) == BandValues.pixel(geometry,channel,row,column) : Nat}:
  Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  +inside = pixel_inside(geometry,limits,capacity,channel,row,column,domain,channel_inside,row_inside,column_inside)
  Bounds.roundtrip(BandValues.pixel(geometry,channel,row,column),capacity,capacity_ok,Bounds.strict_le(BandValues.pixel(geometry,channel,row,column),U32.to_nat(capacity),inside))

# The word of a pixel of the batch is below the capacity of the domain.
def pixel_word_inside(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+channel: Nat,+row: Nat,+column: Nat,+domain: Domain<geometry,limits,capacity>,
  +channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},+row_inside: {Nat.is_lt(row,Inspection.total_rows(geometry)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(BandValues.pixel(geometry,channel,row,column))),U32.to_nat(capacity)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(BandValues.pixel(geometry,channel,row,column))),BandValues.pixel(geometry,channel,row,column),pixel_exact(geometry,limits,capacity,channel,row,column,domain,channel_inside,row_inside,column_inside)) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
  pixel_inside(geometry,limits,capacity,channel,row,column,domain,channel_inside,row_inside,column_inside)

# The row of a cell of a leaf, counted over the batch, is a row of the tensor.
def cell_row_inside(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,+row: Nat,metadata: BandShape.Leaf(geometry,first,rows,length),
  row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(first),row),Inspection.total_rows(geometry)) == True{} : Bool}:
  Bands.Geometry{+images,channels,+height,width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  Bounds.nat_trans(Nat.add(U32.to_nat(first),row),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height)),
    Order.offset_lt(U32.to_nat(first),row,U32.to_nat(rows),row_inside),rows_fit)

# The native global index of the cell at channel, local row and column is the
# index of its pixel. Indexed leaves state their values on coordinates with it.
def indexed_pixel(+call: Call,+channel: Nat,+local: Nat,+column: Nat,facts: Facts<call>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(values_channels(call))) == True{} : Bool},
  +local_inside: {Nat.is_lt(local,U32.to_nat(values_rows(call))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(values_width(call))) == True{} : Bool},
  inside: {Nat.is_lt(BandValues.position(values_geometry(call),values_rows(call),channel,local,column),length(call)) == True{} : Bool}) ->
  {Bands.global_index(values_geometry(call),values_first(call),values_rows(call),U32.from_nat(BandValues.position(values_geometry(call),values_rows(call),channel,local,column))) ==
    U32.from_nat(BandValues.pixel(values_geometry(call),channel,Nat.add(U32.to_nat(values_first(call)),local),column)) : U32}:
  Call{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,+length,+capacity,+division_capacity} = call
  Facts{+valid,+stride_exact,+length_exact,+length_bound,+division_ok,+plane_bound,+height_positive,+width_positive,+count_exact,+height_bound,+row_span} = facts
  +position = BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,local,column)
  indexed_leaf_address(Call{Bands.Geometry{images,channels,height,width},first,rows,length,capacity,division_capacity},U32.from_nat(position),Nat.div(Nat.add(U32.to_nat(first),local),U32.to_nat(height)),channel,Nat.mod(Nat.add(U32.to_nat(first),local),U32.to_nat(height)),column,local,
    Facts{valid,stride_exact,length_exact,length_bound,division_ok,plane_bound,height_positive,width_positive,count_exact,height_bound,row_span},
    Bounds.roundtrip(position,division_capacity,division_ok,Bounds.strict_le(position,U32.to_nat(division_capacity),Bounds.nat_trans(position,length,U32.to_nat(division_capacity),inside,length_bound))),
    channel_inside,DivisionOrder.remainder_small(Nat.add(U32.to_nat(first),local),U32.to_nat(height),height_positive),column_inside,local_inside,
    Equal.sym(Nat,Nat.add(U32.to_nat(first),local),Nat.add(Nat.mul(Nat.div(Nat.add(U32.to_nat(first),local),U32.to_nat(height)),U32.to_nat(height)),Nat.mod(Nat.add(U32.to_nat(first),local),U32.to_nat(height))),DivisionOrder.quotient_exact(Nat.add(U32.to_nat(first),local),U32.to_nat(height),height_positive)))

# The actual whole gather reads the logical value at every valid NCHW point.
# The shape/machine domain and checked partition construct every write-bound,
# matching-value and coverage premise; no readback or schedule law is assumed.
# Producers must separately establish the shared band Values relation.
def gathered_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+limits: Machine.Limits,
  +bands: Model.Model<Element>,+result: Buffer.Model<Element>,+domain: Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>,
  +checked: {complete(~Element,geometry,bands) == True{} : Bool},values: BandValues.Values(~Element,~Context,~value,context,geometry,Model.content(~Element,bands)),
  +image: Nat,+channel: Nat,+row: Nat,+column: Nat,+inside: Inside(geometry,image,channel,row,column),
  depth_ok: {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}) ->
  {Buffer.value(~Element,U32.from_nat(point(geometry,image,channel,row,column)),
      Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))) ==
    value(context,U32.from_nat(point(geometry,image,channel,row,column))) : Element}:
  +capacity = Tree.capacity(Element,Buffer.tree(~Element,result))
  +facts = tree_checked(~Element,bands,geometry,limits,capacity,domain,complete_valid(~Element,geometry,bands,checked))
  +query = U32.from_nat(point(geometry,image,channel,row,column))
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))),
    Buffer.Content{Writes.pure(Element,refinement_writes(~Element,geometry,bands),Buffer.tree(~Element,result)),Buffer.length(~Element,result)},gathered_writes(~Element,geometry,bands,result)) :
    {Buffer.value(~Element,query,_) == value(context,query) : Element}
  Readback.covered(~Element,refinement_writes(~Element,geometry,bands),Buffer.depth(~Element,result),Buffer.tree(~Element,result),query,value(context,query),Buffer.balanced(~Element,result),depth_ok,
    query_inside(geometry,limits,capacity,image,channel,row,column,domain,inside),whole_inside(~Element,bands,geometry,capacity,Machine.half_word(limits),facts),
    whole_matching(~Element,~Context,~value,bands,context,geometry,capacity,Machine.half_word(limits),query,values,facts),
    complete_coverage(~Element,bands,geometry,limits,capacity,domain,image,channel,row,column,inside,checked))

# Observation changes only the proof representation. The actual caller owns
# the produced partition; its producer supplies content and shape evidence.
def observed_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+limits: Machine.Limits,
  -bands: Bands.Bands(~Element),observation: Model.Observation<Element,bands>,+result: Buffer.Model<Element>,
  +domain: Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>,
  well_formed: BandShape.WellFormed<Element,geometry,Model.contents(~Element,bands)>,values: BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,bands)),
  +image: Nat,+channel: Nat,+row: Nat,+column: Nat,+inside: Inside(geometry,image,channel,row,column),
  depth_ok: {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}) ->
  {Buffer.value(~Element,U32.from_nat(point(geometry,image,channel,row,column)),
      Buffer.contents(~Element,Bands.gather(~Element,geometry,bands,Buffer.pack(~Element,result)))) ==
    value(context,U32.from_nat(point(geometry,image,channel,row,column))) : Element}:
  Model.Observation{+model,+equation} = observation
  +content_equal = Equal.trans(Model.Content<Element>,Model.contents(~Element,bands),Model.contents(~Element,Model.pack(~Element,model)),Model.content(~Element,model),
    Equal.cong(Bands.Bands(~Element),Model.Content<Element>,owner => Model.contents(~Element,owner),bands,Model.pack(~Element,model),Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation)),Model.packed_contents(~Element,model))
  %equation : {Buffer.value(~Element,U32.from_nat(point(geometry,image,channel,row,column)),
    Buffer.contents(~Element,Bands.gather(~Element,geometry,_,Buffer.pack(~Element,result)))) == value(context,U32.from_nat(point(geometry,image,channel,row,column))) : Element}
  gathered_value(~Element,~Context,~value,context,geometry,limits,model,result,domain,
    BandShape.well_formed_checked(~Element,geometry,model,Machine.capacity(limits),Limits.capacity_fits(limits),rows_room(geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result)),domain),
      %content_equal : BandShape.WellFormed<Element,geometry,_> well_formed),
    %content_equal : BandValues.Values(~Element,~Context,~value,context,geometry,_) values,
    image,channel,row,column,inside,depth_ok)

def flat_read_located(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+call: Call,+input: Tree.Tree<Element>,+position: Nat,facts: Facts<call>,
  values: BandValues.Leaf(~Element,~Context,~value,context,values_geometry(call),values_first(call),values_rows(call),input,length(call)),
  inside: {Nat.is_lt(position,length(call)) == True{} : Bool},
  coordinates: RowCoordinates.Coordinates<U32.to_nat(values_channels(call)),U32.to_nat(values_rows(call)),U32.to_nat(values_width(call)),position>) ->
  {Tree.read(Element,input,U32.from_nat(position)) == value(context,Bands.global_index(values_geometry(call),values_first(call),values_rows(call),U32.from_nat(position))) : Element}:
  Call{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,+length,+capacity,+division_capacity} = call
  RowCoordinates.Coordinates{+channel,+row,+column,+channel_inside,+row_inside,+column_inside,+exact} = coordinates
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +cell = BandValues.position(geometry,rows,channel,row,column)
  %Equal.sym(Nat,position,cell,exact) : {Tree.read(Element,input,U32.from_nat(_)) == value(context,Bands.global_index(geometry,first,rows,U32.from_nat(_))) : Element}
  %Equal.sym(U32,Bands.global_index(geometry,first,rows,U32.from_nat(cell)),U32.from_nat(BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column)),
      indexed_pixel(Call{Bands.Geometry{images,channels,height,width},first,rows,length,capacity,division_capacity},channel,row,column,facts,channel_inside,row_inside,column_inside,
        %exact : {Nat.is_lt(_,length) == True{} : Bool} inside)) :
    {Tree.read(Element,input,U32.from_nat(cell)) == value(context,_) : Element}
  BandValues.leaf_read(~Element,~Context,~value,context,geometry,first,rows,input,length,channel,row,column,values,channel_inside,row_inside,column_inside)

# The flat reading of a leaf: the element at a local position is the logical
# value at the global index of that position. Operations whose schedule runs
# over flat addresses read the cells of the relation through it; the position
# is split into its channel, row and column here, once.
def flat_read(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+call: Call,+input: Tree.Tree<Element>,+position: Nat,facts: Facts<call>,
  values: BandValues.Leaf(~Element,~Context,~value,context,values_geometry(call),values_first(call),values_rows(call),input,length(call)),
  +inside: {Nat.is_lt(position,length(call)) == True{} : Bool}) ->
  {Tree.read(Element,input,U32.from_nat(position)) == value(context,Bands.global_index(values_geometry(call),values_first(call),values_rows(call),U32.from_nat(position))) : Element}:
  Call{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,+length,+capacity,+division_capacity} = call
  Facts{+valid,+stride_exact,+length_exact,+length_bound,+division_ok,+plane_bound,+height_positive,+width_positive,+count_exact,+height_bound,+row_span} = facts
  flat_read_located(~Element,~Context,~value,context,Call{Bands.Geometry{images,channels,height,width},first,rows,length,capacity,division_capacity},input,position,
    Facts{valid,stride_exact,length_exact,length_bound,division_ok,plane_bound,height_positive,width_positive,count_exact,height_bound,row_span},values,inside,
    RowCoordinates.locate(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width),position,width_positive,
      %count_exact : {Nat.is_lt(position,Nat.mul(U32.to_nat(channels),_)) == True{} : Bool}
      %length_exact : {Nat.is_lt(position,_) == True{} : Bool}
      inside))

# A position inside the content of a leaf shows that the leaf has rows.
def leaf_rows_positive(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,+position: Nat,
  metadata: BandShape.Leaf(geometry,first,rows,length),inside: {Nat.is_lt(position,length) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}:
  Bands.Geometry{images,+channels,height,+width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  RowMajor.left_factor_positive(U32.to_nat(rows),U32.to_nat(width),
    RowMajor.left_factor_positive(Nat.mul(U32.to_nat(rows),U32.to_nat(width)),U32.to_nat(channels),
      %RowMajor.multiply_commutative(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : {Nat.is_lt(0n,_) == True{} : Bool}
      %length_exact : {Nat.is_lt(0n,_) == True{} : Bool}
      Order.le_lt_trans(0n,position,length,Order.zero_le(position),inside)))

# The same flat reading for a leaf described by its metadata and readiness.
# The facts of its gather call are built here.
def flat_read_content(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+limits: Machine.Limits,+first: U32,+rows: U32,
  +input: Buffer.Content<Element>,+capacity: U32,+position: Nat,domain: Domain<geometry,limits,capacity>,
  +metadata: BandShape.Leaf(geometry,first,rows,Buffer.content_length(~Element,input)),ready: Buffer.ContentReady<Element,input>,
  values: BandValues.Leaf(~Element,~Context,~value,context,geometry,first,rows,Buffer.content_tree(~Element,input),Buffer.content_length(~Element,input)),
  +inside: {Nat.is_lt(position,Buffer.content_length(~Element,input)) == True{} : Bool}) ->
  {Tree.read(Element,Buffer.content_tree(~Element,input),U32.from_nat(position)) == value(context,Bands.global_index(geometry,first,rows,U32.from_nat(position))) : Element}:
  flat_read(~Element,~Context,~value,context,Call{geometry,first,rows,Buffer.content_length(~Element,input),capacity,Machine.half_word(limits)},Buffer.content_tree(~Element,input),position,
    from_content(~Element,geometry,limits,first,rows,input,capacity,domain,metadata,
      leaf_rows_positive(geometry,first,rows,Buffer.content_length(~Element,input),position,metadata,inside),ready),values,inside)
