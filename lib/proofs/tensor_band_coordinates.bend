# Coordinate semantics for a whole-channel row band, independent of an
# operator. Local row translation has both inverses; flattened indices use
# quotient/remainder uniqueness, never an assumption about tensor values.
import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division.bend as Division
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ../tensor_band.bend as Bands
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_division.bend as U32Division
import ./encoded_coordinates.bend as Encoded
import ./nat_division_order.bend as DivisionOrder
import ./binary_radix.bend as Radix
import ./storage_shape.bend as Shape

def row_inside(+first: Nat,+rows: Nat,+height: Nat,+row: Nat,inside: {Nat.is_lt(row,rows) == True{} : Bool},
  fits: {Nat.is_le(Nat.add(first,rows),height) == True{} : Bool}) -> {Nat.is_lt(Nat.add(first,row),height) == True{} : Bool}:
  Bounds.nat_trans(Nat.add(first,row),Nat.add(first,rows),height,Order.offset_lt(first,row,rows,inside),fits)

# A located row in an image-local band has the same image and image-relative
# row as the global NCHW point. Both coordinates follow from radix uniqueness.
def row_coordinates(+first: Nat,+height: Nat,+rows: Nat,+local: Nat,+image: Nat,+row: Nat,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+band_span: {Nat.is_le(Nat.add(Nat.mod(first,height),rows),height) == True{} : Bool},
  +local_inside: {Nat.is_lt(local,rows) == True{} : Bool},+row_bound: {Nat.is_lt(row,height) == True{} : Bool},
  +located: {Nat.add(Nat.mul(image,height),row) == Nat.add(first,local) : Nat}) ->
  Shape.Both<{Nat.div(first,height) == image : Nat},{Nat.add(Nat.mod(first,height),local) == row : Nat}>:
  +quotient = Nat.div(first,height)
  +remainder = Nat.mod(first,height)
  +row_exact = Equal.trans(Nat,Nat.add(first,local),Nat.add(Nat.add(Nat.mul(quotient,height),remainder),local),Nat.add(Nat.mul(quotient,height),Nat.add(remainder,local)),
    Equal.cong(Nat,Nat,value => Nat.add(value,local),first,Nat.add(Nat.mul(quotient,height),remainder),DivisionOrder.quotient_exact(first,height,height_positive)),
    Addition.associative(Nat.mul(quotient,height),remainder,local))
  +same = Equal.trans(Nat,Nat.add(Nat.mul(quotient,height),Nat.add(remainder,local)),Nat.add(first,local),Nat.add(Nat.mul(image,height),row),
    Equal.sym(Nat,Nat.add(first,local),Nat.add(Nat.mul(quotient,height),Nat.add(remainder,local)),row_exact),
    Equal.sym(Nat,Nat.add(Nat.mul(image,height),row),Nat.add(first,local),located))
  +inside = row_inside(remainder,rows,height,local,local_inside,band_span)
  Shape.Both{Radix.row_unique(quotient,image,height,Nat.add(remainder,local),row,height_positive,inside,row_bound,same),
    Radix.col_unique(quotient,image,height,Nat.add(remainder,local),row,height_positive,inside,row_bound,same)}

def quotient(+q: Nat,+r: Nat,+base: Nat,+positive: {Nat.is_lt(0n,base) == True{} : Bool},+bounded: {Nat.is_lt(r,base) == True{} : Bool}) ->
  {Nat.div(Nat.add(Nat.mul(q,base),r),base) == q : Nat}:
  Equal.cong(Nat & Nat,Nat,(pair => Pair.fst(Nat,Nat,pair)),Nat.divmod(Nat.add(Nat.mul(q,base),r),base),(q,r),Division.quotient_remainder(q,r,base,positive,bounded))

def remainder(+q: Nat,+r: Nat,+base: Nat,+positive: {Nat.is_lt(0n,base) == True{} : Bool},+bounded: {Nat.is_lt(r,base) == True{} : Bool}) ->
  {Nat.mod(Nat.add(Nat.mul(q,base),r),base) == r : Nat}:
  Equal.cong(Nat & Nat,Nat,(pair => Pair.snd(Nat,Nat,pair)),Nat.divmod(Nat.add(Nat.mul(q,base),r),base),(q,r),Division.quotient_remainder(q,r,base,positive,bounded))

def global_parts(image: Nat,first: Nat,channel: Nat,remainder: Nat,channels: Nat,height: Nat,width: Nat) -> Nat:
  RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,first,remainder)

def global_index(+first: Nat,+rows: Nat,channels: Nat,+height: Nat,+width: Nat,+index: Nat) -> Nat:
  global_parts(Nat.div(first,height),Nat.mod(first,height),Nat.div(index,Nat.mul(rows,width)),Nat.mod(index,Nat.mul(rows,width)),channels,height,width)

def row_join(+channel: Nat,+height: Nat,+width: Nat,+first: Nat,+row: Nat,+column: Nat) ->
  {RowMajor.index(channel,height,width,first,Nat.add(Nat.mul(row,width),column)) == RowMajor.index(channel,height,width,Nat.add(first,row),column) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(first,row),width),Nat.add(Nat.mul(first,width),Nat.mul(row,width)),RowMajor.distrib(first,row,width)) :
    {RowMajor.index(channel,height,width,first,Nat.add(Nat.mul(row,width),column)) == Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(_,column)) : Nat}
  Equal.cong(Nat,Nat,n => Nat.add(Nat.mul(channel,Nat.mul(height,width)),n),
    Nat.add(Nat.mul(first,width),Nat.add(Nat.mul(row,width),column)),Nat.add(Nat.add(Nat.mul(first,width),Nat.mul(row,width)),column),
    Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(first,width),Nat.mul(row,width)),column),Nat.add(Nat.mul(first,width),Nat.add(Nat.mul(row,width),column)),Addition.associative(Nat.mul(first,width),Nat.mul(row,width),column)))

def flattened(+image: Nat,+first: Nat,+rows: Nat,+channels: Nat,+height: Nat,+width: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +first_inside: {Nat.is_lt(first,height) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+column_inside: {Nat.is_lt(column,width) == True{} : Bool}) ->
  {global_index(Nat.add(Nat.mul(image,height),first),rows,channels,height,width,RowMajor.index(channel,rows,width,row,column)) ==
    RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,Nat.add(first,row),column) : Nat}:
  +first_absolute = Nat.add(Nat.mul(image,height),first)
  +local = RowMajor.index(channel,rows,width,row,column)
  +plane = Nat.mul(rows,width)
  +within_plane = Nat.add(Nat.mul(row,width),column)
  +height_positive = Order.le_lt_trans(0n,first,height,Order.zero_le(first),first_inside)
  +width_positive = Order.le_lt_trans(0n,column,width,Order.zero_le(column),column_inside)
  +rows_positive = Order.le_lt_trans(0n,row,rows,Order.zero_le(row),row_inside)
  +plane_positive = RowMajor.multiply_lt(0n,rows,width,rows_positive,width_positive)
  +plane_inside = RowMajor.radix(row,rows,column,width,row_inside,column_inside)
  %Equal.sym(Nat,Nat.div(first_absolute,height),image,quotient(image,first,height,height_positive,first_inside)) :
    {global_parts(_,Nat.mod(first_absolute,height),Nat.div(local,plane),Nat.mod(local,plane),channels,height,width) ==
      RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,Nat.add(first,row),column) : Nat}
  %Equal.sym(Nat,Nat.mod(first_absolute,height),first,remainder(image,first,height,height_positive,first_inside)) :
    {global_parts(image,_,Nat.div(local,plane),Nat.mod(local,plane),channels,height,width) ==
      RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,Nat.add(first,row),column) : Nat}
  %Equal.sym(Nat,Nat.div(local,plane),channel,quotient(channel,within_plane,plane,plane_positive,plane_inside)) :
    {global_parts(image,first,_,Nat.mod(local,plane),channels,height,width) ==
      RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,Nat.add(first,row),column) : Nat}
  %Equal.sym(Nat,Nat.mod(local,plane),within_plane,remainder(channel,within_plane,plane,plane_positive,plane_inside)) :
    {global_parts(image,first,channel,_,channels,height,width) ==
      RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,Nat.add(first,row),column) : Nat}
  row_join(Nat.add(Nat.mul(image,channels),channel),height,width,first,row,column)

# Each mixed-radix step is exact when its complete result fits. The product
# bound is derived here, rather than assumed for an intermediate machine word.
def digit(+a: U32,+radix: U32,+rest: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(a),U32.to_nat(radix)),U32.to_nat(rest)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((a * radix + rest : U32)) == Nat.add(Nat.mul(U32.to_nat(a),U32.to_nat(radix)),U32.to_nat(rest)) : Nat}:
  +product = Nat.mul(U32.to_nat(a),U32.to_nat(radix))
  +exact = Arithmetic.multiply(a,radix,capacity,capacity_ok,
    Order.transitive(product,Nat.add(product,U32.to_nat(rest)),U32.to_nat(capacity),Order.le_add(product,U32.to_nat(rest)),room))
  %exact : {U32.to_nat((a * radix + rest : U32)) == Nat.add(_,U32.to_nat(rest)) : Nat}
  Arithmetic.add((a * radix : U32),rest,capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat((a * radix : U32)),product,exact) : {Nat.is_le(Nat.add(_,U32.to_nat(rest)),U32.to_nat(capacity)) == True{} : Bool}
    room)

def coordinates_room(+batch: Nat,+channels: Nat,+height: Nat,+width: Nat,+image: Nat,+first: Nat,+rows: Nat,+channel: Nat,+row: Nat,+column: Nat,+capacity: Nat,
  image_inside: {Nat.is_lt(image,batch) == True{} : Bool},channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},
  row_fits: {Nat.is_le(Nat.add(first,rows),height) == True{} : Bool},row_local: {Nat.is_lt(row,rows) == True{} : Bool},column_inside: {Nat.is_lt(column,width) == True{} : Bool},
  total_room: {Nat.is_le(Nat.mul(batch,Nat.mul(channels,Nat.mul(height,width))),capacity) == True{} : Bool}) ->
  {Nat.is_le(RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,Nat.add(first,row),column),capacity) == True{} : Bool}:
  Bounds.strict_le(RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,Nat.add(first,row),column),capacity,
    Bounds.nat_trans(RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,Nat.add(first,row),column),Nat.mul(Nat.mul(batch,channels),Nat.mul(height,width)),capacity,
      RowMajor.in_bounds(Nat.add(Nat.mul(image,channels),channel),Nat.mul(batch,channels),height,width,Nat.add(first,row),column,
        RowMajor.radix(image,batch,channel,channels,image_inside,channel_inside),row_inside(first,rows,height,row,row_local,row_fits),column_inside),
      %Equal.sym(Nat,Nat.mul(Nat.mul(batch,channels),Nat.mul(height,width)),Nat.mul(batch,Nat.mul(channels,Nat.mul(height,width))),RowMajor.assoc(batch,channels,Nat.mul(height,width))) :
        {Nat.is_le(_,capacity) == True{} : Bool} total_room))

def native_parts(+image: U32,+first: U32,+channel: U32,+remainder: U32,+channels: U32,+height: U32,+width: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  +room: {Nat.is_le(global_parts(U32.to_nat(image),U32.to_nat(first),U32.to_nat(channel),U32.to_nat(remainder),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((((image * channels + channel) * height + first) * width + remainder : U32)) ==
    global_parts(U32.to_nat(image),U32.to_nat(first),U32.to_nat(channel),U32.to_nat(remainder),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) : Nat}:
  +ic = Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(channels)),U32.to_nat(channel))
  +row = Nat.add(Nat.mul(ic,U32.to_nat(height)),U32.to_nat(first))
  +mixed = RowMajor.mixed(ic,U32.to_nat(height),U32.to_nat(width),U32.to_nat(first),U32.to_nat(remainder))
  +outer_room = {
    %Equal.sym(Nat,Nat.add(Nat.mul(row,U32.to_nat(width)),U32.to_nat(remainder)),RowMajor.index(ic,U32.to_nat(height),U32.to_nat(width),U32.to_nat(first),U32.to_nat(remainder)),mixed) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room : {Nat.is_le(Nat.add(Nat.mul(row,U32.to_nat(width)),U32.to_nat(remainder)),U32.to_nat(capacity)) == True{} : Bool}}
  +row_room = Encoded.left_bound(row,U32.to_nat(width),U32.to_nat(remainder),U32.to_nat(capacity),width_positive,outer_room)
  +ic_exact = digit(image,channels,channel,capacity,capacity_ok,Encoded.left_bound(ic,U32.to_nat(height),U32.to_nat(first),U32.to_nat(capacity),height_positive,row_room))
  +row_exact = {
    %ic_exact : {U32.to_nat(((image * channels + channel) * height + first : U32)) == Nat.add(Nat.mul(_,U32.to_nat(height)),U32.to_nat(first)) : Nat}
    digit((image * channels + channel : U32),height,first,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat((image * channels + channel : U32)),ic,ic_exact) : {Nat.is_le(Nat.add(Nat.mul(_,U32.to_nat(height)),U32.to_nat(first)),U32.to_nat(capacity)) == True{} : Bool}
      row_room) : {U32.to_nat(((image * channels + channel) * height + first : U32)) == row : Nat}}
  %mixed : {U32.to_nat((((image * channels + channel) * height + first) * width + remainder : U32)) == _ : Nat}
  %row_exact : {U32.to_nat((((image * channels + channel) * height + first) * width + remainder : U32)) == Nat.add(Nat.mul(_,U32.to_nat(width)),U32.to_nat(remainder)) : Nat}
  digit(((image * channels + channel) * height + first : U32),width,remainder,capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat(((image * channels + channel) * height + first : U32)),row,row_exact) : {Nat.is_le(Nat.add(Nat.mul(_,U32.to_nat(width)),U32.to_nat(remainder)),U32.to_nat(capacity)) == True{} : Bool}
    outer_room)

def decoded_parts(+first: U32,+plane: U32,+index: U32,channels: U32,+height: U32,width: U32) -> Nat:
  global_parts(U32.to_nat((first / height : U32)),U32.to_nat((first % height : U32)),U32.to_nat((index / plane : U32)),U32.to_nat((index % plane : U32)),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width))

def decoded_exact(+first: U32,+plane: U32,+index: U32,+channels: U32,+height: U32,+width: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+plane_positive: {Nat.is_lt(0n,U32.to_nat(plane)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool},
  +plane_bound: {Nat.is_le(Nat.double(U32.to_nat(plane)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {decoded_parts(first,plane,index,channels,height,width) == global_parts(Nat.div(U32.to_nat(first),U32.to_nat(height)),Nat.mod(U32.to_nat(first),U32.to_nat(height)),
    Nat.div(U32.to_nat(index),U32.to_nat(plane)),Nat.mod(U32.to_nat(index),U32.to_nat(plane)),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) : Nat}:
  %U32Division.divide(first,height,capacity,capacity_ok,height_bound,height_positive) :
    {decoded_parts(first,plane,index,channels,height,width) == global_parts(_,Nat.mod(U32.to_nat(first),U32.to_nat(height)),Nat.div(U32.to_nat(index),U32.to_nat(plane)),Nat.mod(U32.to_nat(index),U32.to_nat(plane)),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) : Nat}
  %U32Division.modulo(first,height,capacity,capacity_ok,height_bound,height_positive) :
    {decoded_parts(first,plane,index,channels,height,width) == global_parts(U32.to_nat((first / height : U32)),_,Nat.div(U32.to_nat(index),U32.to_nat(plane)),Nat.mod(U32.to_nat(index),U32.to_nat(plane)),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) : Nat}
  %U32Division.divide(index,plane,capacity,capacity_ok,plane_bound,plane_positive) :
    {decoded_parts(first,plane,index,channels,height,width) == global_parts(U32.to_nat((first / height : U32)),U32.to_nat((first % height : U32)),_,Nat.mod(U32.to_nat(index),U32.to_nat(plane)),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) : Nat}
  %U32Division.modulo(index,plane,capacity,capacity_ok,plane_bound,plane_positive) :
    {decoded_parts(first,plane,index,channels,height,width) == global_parts(U32.to_nat((first / height : U32)),U32.to_nat((first % height : U32)),U32.to_nat((index / plane : U32)),_,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) : Nat}
  {==}

def native(+batch: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+index: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool},
  +plane_bound: {Nat.is_le(Nat.double(Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(capacity)) == True{} : Bool},
  +room: {Nat.is_le(global_index(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),U32.to_nat(index)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(Bands.global_index(Bands.Geometry{batch,channels,height,width},first,rows,index)) ==
    global_index(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),U32.to_nat(index)) : Nat}:
  +plane = (rows * width : U32)
  +plane_nat = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +plane_exact = Arithmetic.multiply(rows,width,capacity,capacity_ok,
    Order.transitive(plane_nat,Nat.double(plane_nat),U32.to_nat(capacity),Order.le_double(plane_nat),plane_bound))
  +decoded = {
    %plane_exact : {decoded_parts(first,plane,index,channels,height,width) == global_parts(Nat.div(U32.to_nat(first),U32.to_nat(height)),Nat.mod(U32.to_nat(first),U32.to_nat(height)),
      Nat.div(U32.to_nat(index),_),Nat.mod(U32.to_nat(index),_),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) : Nat}
    decoded_exact(first,plane,index,channels,height,width,capacity,capacity_ok,height_positive,
      {%Equal.sym(Nat,U32.to_nat(plane),plane_nat,plane_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
        RowMajor.multiply_lt(0n,U32.to_nat(rows),U32.to_nat(width),rows_positive,width_positive) : {Nat.is_lt(0n,U32.to_nat(plane)) == True{} : Bool}},
      height_bound,{%Equal.sym(Nat,U32.to_nat(plane),plane_nat,plane_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(capacity)) == True{} : Bool}
        plane_bound : {Nat.is_le(Nat.double(U32.to_nat(plane)),U32.to_nat(capacity)) == True{} : Bool}}) :
    {decoded_parts(first,plane,index,channels,height,width) == global_index(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),U32.to_nat(index)) : Nat}}
  %decoded : {U32.to_nat(Bands.global_index(Bands.Geometry{batch,channels,height,width},first,rows,index)) == _ : Nat}
  native_parts((first / height : U32),(first % height : U32),(index / plane : U32),(index % plane : U32),channels,height,width,capacity,capacity_ok,height_positive,width_positive,
    %Equal.sym(Nat,decoded_parts(first,plane,index,channels,height,width),global_index(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),U32.to_nat(index)),decoded) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room)
