# Contents of a convolution window and the rows it covers.
import Base
import ./concrete_support_guard.bend as Support
import ./window_positions.bend as Positions
import ./window_samples.bend as Samples
import ../specification.bend as ConvSpec
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ./convolution_guard_facts.bend as Guards
import ./input_storage.bend as Inputs
import ./supported_padded_coordinates.bend as Padded
import ./window_band_values.bend as WindowBandValues
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_extrema.bend as Extrema
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_coordinates.bend as BandCoordinates
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/tensor_band_window.bend as TensorBandWindow
import ../../lib/tensor_band.bend as Bands

# One image of a banded input is the flat list a convolution problem reads:
# its element t is the band tensor's logical value at the image offset plus t.
# A window whose band Values hold rows [first, first + rows) of that image
# therefore holds the window problem's band as its leading reads.
def image_values(~Logical: Data,~original: Logical -> U32 -> F32,count: Nat,+index: Nat,+logical: Logical) -> List<&2,F32>:
  match count:
    case 0n: []
    case 1n+rest: original(logical,U32.from_nat(index)) <> image_values(~Logical,~original,rest,1n+index,logical)

def image_length(~Logical: Data,~original: Logical -> U32 -> F32,+count: Nat,+index: Nat,+logical: Logical) ->
  {ConvSpec.length(image_values(~Logical,~original,count,index,logical)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest: Equal.cong(Nat,Nat,n => 1n+n,ConvSpec.length(image_values(~Logical,~original,rest,1n+index,logical)),rest,image_length(~Logical,~original,rest,1n+index,logical))

def image_get(~Logical: Data,~original: Logical -> U32 -> F32,+count: Nat,+index: Nat,+logical: Logical,+position: Nat,
  +inside: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {List.get(&2,F32,image_values(~Logical,~original,count,index,logical),position) == Some{original(logical,U32.from_nat(Nat.add(index,position)))} : Maybe<&2,F32>}:
  match count position:
    case 0n 0n:
      Empty.absurd({None{} == Some{original(logical,U32.from_nat(Nat.add(index,0n)))} : Maybe<&2,F32>},WordEncoding.false_true(inside))
    case 0n 1n+previous:
      Empty.absurd({None{} == Some{original(logical,U32.from_nat(Nat.add(index,1n+previous)))} : Maybe<&2,F32>},WordEncoding.false_true(inside))
    case 1n+rest 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) :
        {Some{original(logical,U32.from_nat(index))} == Some{original(logical,U32.from_nat(_))} : Maybe<&2,F32>}
      {==}
    case 1n+ +rest 1n+ +previous:
      %Equal.sym(Nat,Nat.add(index,1n+previous),1n+Nat.add(index,previous),Addition.successor_right(index,previous)) :
        {List.get(&2,F32,image_values(~Logical,~original,rest,1n+index,logical),previous) == Some{original(logical,U32.from_nat(_))} : Maybe<&2,F32>}
      image_get(~Logical,~original,rest,1n+index,logical,previous,inside)

def image_at(~Logical: Data,~original: Logical -> U32 -> F32,+count: Nat,+index: Nat,+logical: Logical,+position: Nat,
  +inside: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {ConvSpec.at(position,image_values(~Logical,~original,count,index,logical)) == original(logical,U32.from_nat(Nat.add(index,position))) : F32}:
  %Equal.sym(F32,ConvSpec.at(position,image_values(~Logical,~original,count,index,logical)),
    Maybe.default(&2,F32,List.get(&2,F32,image_values(~Logical,~original,count,index,logical),position),0.0),
    Inputs.lookup_matches_list(position,image_values(~Logical,~original,count,index,logical))) : {_ == original(logical,U32.from_nat(Nat.add(index,position))) : F32}
  %Equal.sym(Maybe<&2,F32>,List.get(&2,F32,image_values(~Logical,~original,count,index,logical),position),Some{original(logical,U32.from_nat(Nat.add(index,position)))},
    image_get(~Logical,~original,count,index,logical,position,inside)) :
    {Maybe.default(&2,F32,_,0.0) == original(logical,U32.from_nat(Nat.add(index,position))) : F32}
  {==}

def image_band(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+height: Nat,+width: Nat,+image: Nat,+first: Nat,+rows: Nat) -> WindowBandValues.Band:
  +area = Nat.mul(channels,Nat.mul(height,width))
  WindowBandValues.Band{image_values(~Logical,~original,area,Nat.mul(image,area),logical),height,width,first,rows}

# The flat index of a window element in its image, and in the whole tensor.
def image_index(+image: Nat,+channels: Nat,+area: Nat,+channel: Nat,+base: Nat,+remainder: Nat) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(image,channels),channel),area),Nat.add(base,remainder)) ==
    Nat.add(Nat.mul(image,Nat.mul(channels,area)),Nat.add(Nat.mul(channel,area),Nat.add(base,remainder))) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,channels),channel),area),Nat.add(Nat.mul(Nat.mul(image,channels),area),Nat.mul(channel,area)),RowMajor.distrib(Nat.mul(image,channels),channel,area)) :
    {Nat.add(_,Nat.add(base,remainder)) == Nat.add(Nat.mul(image,Nat.mul(channels,area)),Nat.add(Nat.mul(channel,area),Nat.add(base,remainder))) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(image,channels),area),Nat.mul(image,Nat.mul(channels,area)),RowMajor.assoc(image,channels,area)) :
    {Nat.add(Nat.add(_,Nat.mul(channel,area)),Nat.add(base,remainder)) == Nat.add(Nat.mul(image,Nat.mul(channels,area)),Nat.add(Nat.mul(channel,area),Nat.add(base,remainder))) : Nat}
  Addition.associative(Nat.mul(image,Nat.mul(channels,area)),Nat.mul(channel,area),Nat.add(base,remainder))

def positive_right(+left: Nat,+right: Nat,+positive: {Nat.is_lt(0n,Nat.mul(left,right)) == True{} : Bool}) -> {Nat.is_lt(0n,right) == True{} : Bool}:
  RowMajor.left_factor_positive(right,left,
    %RowMajor.multiply_commutative(left,right) : {Nat.is_lt(0n,_) == True{} : Bool} positive)

def multiply_left_le(+factor: Nat,+smaller: Nat,+larger: Nat,+ordered: {Nat.is_le(smaller,larger) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(factor,smaller),Nat.mul(factor,larger)) == True{} : Bool}:
  %RowMajor.multiply_commutative(smaller,factor) : {Nat.is_le(_,Nat.mul(factor,larger)) == True{} : Bool}
  %RowMajor.multiply_commutative(larger,factor) : {Nat.is_le(Nat.mul(smaller,factor),_) == True{} : Bool}
  RowMajor.mul_le(smaller,larger,factor,ordered)

# The tensor offset of a window element, in the problem's image-list terms.
def element_index(+image: Nat,+channels: Nat,+height: Nat,+width: Nat,+channel: Nat,+first: Nat,+row: Nat,+column: Nat,+local: Nat,
  +local_split: {local == Nat.add(Nat.mul(row,width),column) : Nat}) ->
  {RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,Nat.add(first,row),column) ==
    Nat.add(Nat.mul(image,Nat.mul(channels,Nat.mul(height,width))),Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(first,width),local))) : Nat}:
  +area = Nat.mul(height,width)
  +joined = Nat.add(Nat.mul(image,channels),channel)
  Equal.trans(Nat,RowMajor.index(joined,height,width,Nat.add(first,row),column),Nat.add(Nat.mul(joined,area),Nat.add(Nat.mul(first,width),local)),
    Nat.add(Nat.mul(image,Nat.mul(channels,area)),Nat.add(Nat.mul(channel,area),Nat.add(Nat.mul(first,width),local))),
    %Equal.sym(Nat,local,Nat.add(Nat.mul(row,width),column),local_split) : {RowMajor.index(joined,height,width,Nat.add(first,row),column) == Nat.add(Nat.mul(joined,area),Nat.add(Nat.mul(first,width),_)) : Nat}
    Equal.sym(Nat,RowMajor.index(joined,height,width,first,Nat.add(Nat.mul(row,width),column)),RowMajor.index(joined,height,width,Nat.add(first,row),column),
      BandCoordinates.row_join(joined,height,width,first,row,column)),
    image_index(image,channels,area,channel,Nat.mul(first,width),local))

# Every window element read back from band Values is the problem's band value.
def image_read(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32,
  +image: Nat,+local_first: Nat,+first: U32,+rows: U32,+first_exact: {U32.to_nat(first) == Nat.add(Nat.mul(image,U32.to_nat(height)),local_first) : Nat},
  +row_fits: {Nat.is_le(Nat.add(local_first,U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool},
  +tree: Tree.Tree<F32>,+length: Nat,+length_exact: {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
  +values: BandValues.Leaf(~F32,~Logical,~original,logical,Bands.Geometry{images,channels,height,width},first,rows,tree,length),
  +index: Nat,+inside: {Nat.is_lt(index,length) == True{} : Bool}) ->
  {Tree.read(F32,tree,U32.from_nat(index)) ==
    WindowBandValues.value(image_band(~Logical,~original,logical,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),image,local_first,U32.to_nat(rows)),index) : F32}:
  +ci = U32.to_nat(channels)
  +h = U32.to_nat(height)
  +w = U32.to_nat(width)
  +r = U32.to_nat(rows)
  +area = Nat.mul(h,w)
  +plane = Nat.mul(r,w)
  +total = Nat.mul(ci,area)
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +inside_total = {%length_exact : {Nat.is_lt(index,_) == True{} : Bool} inside : {Nat.is_lt(index,Nat.mul(ci,plane)) == True{} : Bool}}
  +product_positive = Order.le_lt_trans(0n,index,Nat.mul(ci,plane),Order.zero_le(index),inside_total)
  +plane_positive = positive_right(ci,plane,product_positive)
  +width_positive = positive_right(r,w,plane_positive)
  +channel = Nat.div(index,plane)
  +local = Nat.mod(index,plane)
  +row = Nat.div(local,w)
  +column = Nat.mod(local,w)
  +channel_inside = Positions.quotient_lt(index,ci,plane,plane_positive,inside_total)
  +local_inside = DivisionOrder.remainder_small(index,plane,plane_positive)
  +row_inside = Positions.quotient_lt(local,r,w,width_positive,local_inside)
  +column_inside = DivisionOrder.remainder_small(local,w,width_positive)
  +index_split = DivisionOrder.quotient_exact(index,plane,plane_positive)
  +local_split = DivisionOrder.quotient_exact(local,w,width_positive)
  +position_exact = Equal.trans(Nat,index,Nat.add(Nat.mul(channel,plane),local),RowMajor.index(channel,r,w,row,column),index_split,
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(channel,plane),value),local,Nat.add(Nat.mul(row,w),column),local_split))
  +source = Nat.add(Nat.mul(channel,area),Nat.add(Nat.mul(local_first,w),local))
  +line_inside = {
    %Equal.sym(Nat,local,Nat.add(Nat.mul(row,w),column),local_split) : {Nat.is_lt(Nat.add(Nat.mul(local_first,w),_),area) == True{} : Bool}
    %Addition.associative(Nat.mul(local_first,w),Nat.mul(row,w),column) : {Nat.is_lt(_,area) == True{} : Bool}
    %RowMajor.distrib(local_first,row,w) : {Nat.is_lt(Nat.add(_,column),area) == True{} : Bool}
    RowMajor.radix(Nat.add(local_first,row),h,column,w,
      Bounds.nat_trans(Nat.add(local_first,row),Nat.add(local_first,r),h,Order.offset_lt(local_first,row,r,row_inside),row_fits),column_inside) :
    {Nat.is_lt(Nat.add(Nat.mul(local_first,w),local),area) == True{} : Bool}}
  +source_inside = RowMajor.radix(channel,ci,Nat.add(Nat.mul(local_first,w),local),area,channel_inside,line_inside)
  +target = Nat.add(Nat.mul(image,total),source)
  +global_inside = Bounds.nat_trans(Nat.add(local_first,row),Nat.add(local_first,r),h,Order.offset_lt(local_first,row,r,row_inside),row_fits)
  +pixel = BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column)
  +pixel_exact = Equal.trans(Nat,pixel,RowMajor.index(Nat.add(Nat.mul(image,ci),channel),h,w,Nat.add(local_first,row),column),target,
    BandValues.located_pixel(geometry,image,channel,Nat.add(local_first,row),column,Nat.add(U32.to_nat(first),row),global_inside,
      %Equal.sym(Nat,U32.to_nat(first),Nat.add(Nat.mul(image,h),local_first),first_exact) : {Nat.add(Nat.mul(image,h),Nat.add(local_first,row)) == Nat.add(_,row) : Nat}
      Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(image,h),local_first),row),Nat.add(Nat.mul(image,h),Nat.add(local_first,row)),Addition.associative(Nat.mul(image,h),local_first,row))),
    element_index(image,ci,h,w,channel,local_first,row,column,local,local_split))
  Equal.trans(F32,Tree.read(F32,tree,U32.from_nat(index)),original(logical,U32.from_nat(pixel)),
    ConvSpec.at(source,image_values(~Logical,~original,total,Nat.mul(image,total),logical)),
    %Equal.sym(Nat,index,RowMajor.index(channel,r,w,row,column),position_exact) : {Tree.read(F32,tree,U32.from_nat(_)) == original(logical,U32.from_nat(pixel)) : F32}
    BandValues.leaf_read(~F32,~Logical,~original,logical,geometry,first,rows,tree,length,channel,row,column,values,channel_inside,row_inside,column_inside),
    Equal.trans(F32,original(logical,U32.from_nat(pixel)),original(logical,U32.from_nat(target)),ConvSpec.at(source,image_values(~Logical,~original,total,Nat.mul(image,total),logical)),
      Equal.cong(Nat,F32,value => original(logical,U32.from_nat(value)),pixel,target,pixel_exact),
      Equal.sym(F32,ConvSpec.at(source,image_values(~Logical,~original,total,Nat.mul(image,total),logical)),original(logical,U32.from_nat(target)),
        image_at(~Logical,~original,total,Nat.mul(image,total),logical,source,source_inside))))

def image_listed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32,
  +image: Nat,+local_first: Nat,+first: U32,+rows: U32,+first_exact: {U32.to_nat(first) == Nat.add(Nat.mul(image,U32.to_nat(height)),local_first) : Nat},
  +row_fits: {Nat.is_le(Nat.add(local_first,U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool},
  +tree: Tree.Tree<F32>,+length: Nat,+length_exact: {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
  +values: BandValues.Leaf(~F32,~Logical,~original,logical,Bands.Geometry{images,channels,height,width},first,rows,tree,length),
  count: Nat,+index: Nat,+room: {Nat.is_le(Nat.add(index,count),length) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,count,U32.from_nat(index),tree) ==
    WindowBandValues.values(count,index,image_band(~Logical,~original,logical,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),image,local_first,U32.to_nat(rows))) : List<&2,F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +band = image_band(~Logical,~original,logical,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),image,local_first,U32.to_nat(rows))
      %Equal.sym(F32,Tree.read(F32,tree,U32.from_nat(index)),WindowBandValues.value(band,index),
        image_read(~Logical,~original,logical,images,channels,height,width,image,local_first,first,rows,first_exact,row_fits,tree,length,length_exact,values,index,
          Order.room_index(index,rest,length,room))) :
        {_ <> Tree.read_sequence(~F32,rest,(U32.from_nat(index) + 1 : U32),tree) == WindowBandValues.values(1n+rest,index,band) : List<&2,F32>}
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.add_one(U32.from_nat(index))) :
        {WindowBandValues.value(band,index) <> Tree.read_sequence(~F32,rest,_,tree) == WindowBandValues.values(1n+rest,index,band) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => WindowBandValues.value(band,index) <> tail,Tree.read_sequence(~F32,rest,U32.from_nat(1n+index),tree),WindowBandValues.values(rest,1n+index,band),
        image_listed(~Logical,~original,logical,images,channels,height,width,image,local_first,first,rows,first_exact,row_fits,tree,length,length_exact,values,
          rest,1n+index,Order.advance_room(index,rest,length,room)))

# A window holding rows [first, first + rows) of image `image` holds the
# problem band those rows define.
def contents(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32,
  +image: Nat,+local_first: Nat,+first: U32,+rows: U32,+first_exact: {U32.to_nat(first) == Nat.add(Nat.mul(image,U32.to_nat(height)),local_first) : Nat},
  +row_fits: {Nat.is_le(Nat.add(local_first,U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool},
  +tree: Tree.Tree<F32>,+length: Nat,+length_exact: {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
  +values: BandValues.Leaf(~F32,~Logical,~original,logical,Bands.Geometry{images,channels,height,width},first,rows,tree,length)) ->
  WindowBandValues.Contents(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),
    image_band(~Logical,~original,logical,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),image,local_first,U32.to_nat(rows)),tree):
  %length_exact : WindowBandValues.Contents(_,image_band(~Logical,~original,logical,U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),image,local_first,U32.to_nat(rows)),tree)
  image_listed(~Logical,~original,logical,images,channels,height,width,image,local_first,first,rows,first_exact,row_fits,tree,length,length_exact,values,length,0n,Order.reflexive(length))

# A band leaf's window holds every image row its output rows read. The window
# starts at the first row read, moved down to a stride multiple, and ends at
# the last row read. Padding smaller than the kernel keeps every output row
# reading at least one image row, so the window is never empty.
def aligned(+value: Nat,+stride: Nat) -> Nat:
  Nat.sub(value,Nat.mod(value,stride))

def aligned_below(+value: Nat,+stride: Nat) -> {Nat.is_le(aligned(value,stride),value) == True{} : Bool}:
  Order.sub_le(value,Nat.mod(value,stride))

def aligned_exact(+value: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {aligned(value,stride) == Nat.mul(Nat.div(value,stride),stride) : Nat}:
  +quotient = Nat.mul(Nat.div(value,stride),stride)
  +remainder = Nat.mod(value,stride)
  %Equal.sym(Nat,value,Nat.add(quotient,remainder),DivisionOrder.quotient_exact(value,stride,positive)) :
    {Nat.sub(_,Nat.mod(value,stride)) == Nat.mul(Nat.div(value,stride),stride) : Nat}
  %Order.commutative(remainder,quotient) : {Nat.sub(_,remainder) == quotient : Nat}
  Order.subtract_shift(remainder,quotient)

def not_le(+left: Nat,+right: Nat,+outside: {Nat.is_le(left,right) == False{} : Bool}) -> {Nat.is_le(right,left) == True{} : Bool}:
  match left right:
    case 0n 0n: Empty.absurd({True{} == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},outside)))
    case 0n 1n+right: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},outside)))
    case 1n+left 0n: {==}
    case 1n+ +left 1n+ +right: not_le(left,right,outside)

def sub_below(+value: Nat,+amount: Nat,+below: {Nat.is_le(value,amount) == True{} : Bool}) -> {Nat.sub(value,amount) == 0n : Nat}:
  match value amount:
    case 0n amount: Order.zero_subtract(amount)
    case 1n+value 0n: Empty.absurd({Nat.sub(1n+value,0n) == 0n : Nat},WordEncoding.false_true(below))
    case 1n+ +value 1n+ +amount: sub_below(value,amount,below)

def zero_below(+value: Nat,+bounded: {Nat.is_le(value,0n) == True{} : Bool}) -> {value == 0n : Nat}:
  match value:
    case 0n: {==}
    case 1n+rest: Empty.absurd({1n+rest == 0n : Nat},WordEncoding.false_true(bounded))

def two_times(+value: Nat) -> {Nat.mul(2n,value) == Nat.add(value,value) : Nat}:
  %Addition.zero_right(value) : {Nat.add(value,Nat.add(value,0n)) == Nat.add(value,_) : Nat}
  {==}

def positive_amount(+amount: Nat,+extra: Nat,+small: {Nat.is_lt(amount,extra) == True{} : Bool}) -> {Nat.is_lt(0n,extra) == True{} : Bool}:
  Order.le_lt_trans(0n,amount,extra,Order.zero_le(amount),small)

def shifted_below_when(ordered: Bool,+value: Nat,+amount: Nat,+extra: Nat,+guard: {Nat.is_le(amount,value) == ordered : Bool},
  +small: {Nat.is_lt(amount,extra) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(value,amount),Nat.sub(Nat.add(value,extra),amount)) == True{} : Bool}:
  match ordered:
    case True{}:
      %Equal.sym(Nat,Nat.sub(Nat.add(value,extra),amount),Nat.add(Nat.sub(value,amount),extra),Positions.sub_after(value,amount,extra,guard)) :
        {Nat.is_lt(Nat.sub(value,amount),_) == True{} : Bool}
      %Addition.zero_right(Nat.sub(value,amount)) : {Nat.is_lt(_,Nat.add(Nat.sub(value,amount),extra)) == True{} : Bool}
      Order.offset_lt(Nat.sub(value,amount),0n,extra,positive_amount(amount,extra,small))
    case False{}:
      %Equal.sym(Nat,Nat.sub(value,amount),0n,sub_below(value,amount,not_le(amount,value,guard))) : {Nat.is_lt(_,Nat.sub(Nat.add(value,extra),amount)) == True{} : Bool}
      Order.positive_difference(Nat.add(value,extra),amount,Bounds.nat_trans(amount,extra,Nat.add(value,extra),small,Samples.le_after(value,extra)))

# Subtracting an amount smaller than the extra term keeps a strict order.
def shifted_below(+value: Nat,+amount: Nat,+extra: Nat,+small: {Nat.is_lt(amount,extra) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(value,amount),Nat.sub(Nat.add(value,extra),amount)) == True{} : Bool}:
  shifted_below_when(Nat.is_le(amount,value),value,amount,extra,{==},small)

def clipped_below_when(ordered: Bool,+value: Nat,+amount: Nat,+limit: Nat,+guard: {Nat.is_le(amount,value) == ordered : Bool},
  +positive: {Nat.is_lt(0n,limit) == True{} : Bool},+bound: {Nat.is_lt(value,Nat.add(amount,limit)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(value,amount),limit) == True{} : Bool}:
  match ordered:
    case True{}: Order.sub_lt(value,amount,limit,guard,bound)
    case False{}:
      %Equal.sym(Nat,Nat.sub(value,amount),0n,sub_below(value,amount,not_le(amount,value,guard))) : {Nat.is_lt(_,limit) == True{} : Bool}
      positive

def clipped_below(+value: Nat,+amount: Nat,+limit: Nat,+positive: {Nat.is_lt(0n,limit) == True{} : Bool},
  +bound: {Nat.is_lt(value,Nat.add(amount,limit)) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(value,amount),limit) == True{} : Bool}:
  clipped_below_when(Nat.is_le(amount,value),value,amount,limit,{==},positive,bound)

def minimum_shift_when(choice: Bool,+left: Nat,+right: Nat,+shift: Nat,+lower: Nat,
  +left_ok: {Nat.is_le(lower,Nat.add(left,shift)) == True{} : Bool},+right_ok: {Nat.is_le(lower,Nat.add(right,shift)) == True{} : Bool}) ->
  {Nat.is_le(lower,Nat.add(Bool.pick(Nat,choice,left,right),shift)) == True{} : Bool}:
  match choice:
    case True{}: left_ok
    case False{}: right_ok

def minimum_shift(+left: Nat,+right: Nat,+shift: Nat,+lower: Nat,
  +left_ok: {Nat.is_le(lower,Nat.add(left,shift)) == True{} : Bool},+right_ok: {Nat.is_le(lower,Nat.add(right,shift)) == True{} : Bool}) ->
  {Nat.is_le(lower,Nat.add(Nat.min(left,right),shift)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.min(left,right),Bool.pick(Nat,Nat.is_lt(left,right),left,right),Extrema.minimum_choice(left,right)) : {Nat.is_le(lower,Nat.add(_,shift)) == True{} : Bool}
  minimum_shift_when(Nat.is_lt(left,right),left,right,shift,lower,left_ok,right_ok)

def window_first(+height: Nat,+output_first: Nat,+stride: Nat,+padding: Nat) -> Nat:
  aligned(TensorBandWindow.needed_first(height,output_first,stride,padding),stride)

def window_end(+height: Nat,+output_end: Nat,+stride: Nat,+kernel: Nat,+padding: Nat) -> Nat:
  TensorBandWindow.needed_end(height,output_end,stride,kernel,padding)

# The Nat facts a window cover needs, from an output interval inside the image.
type Span<-height: Nat,-kernel: Nat,-stride: Nat,-padding: Nat,-output_first: Nat,-output_end: Nat> is Data:
  Span{stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
    padding_small: {Nat.is_lt(padding,kernel) == True{} : Bool},ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},
    last_room: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool}}

def first_row(+output_first: Nat,+stride: Nat) -> Nat:
  Nat.mul(output_first,stride)

def last_row(+output_end: Nat,+stride: Nat) -> Nat:
  Nat.mul(Nat.sub(output_end,1n),stride)

def rows_ordered(+output_first: Nat,+output_end: Nat,+stride: Nat,+ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool}) ->
  {Nat.is_le(first_row(output_first,stride),last_row(output_end,stride)) == True{} : Bool}:
  RowMajor.mul_le(output_first,Nat.sub(output_end,1n),stride,TensorBandWindow.before_last(output_first,output_end,ordered))

# The first row read lies inside the image.
def first_inside(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+output_first: Nat,+output_end: Nat,
  +span: Span<height,kernel,stride,padding,output_first,output_end>) ->
  {Nat.is_lt(Nat.sub(first_row(output_first,stride),padding),height) == True{} : Bool}:
  Span{stride_positive,+height_positive,+padding_small,+ordered,+last_room} = span
  +first = first_row(output_first,stride)
  +first_room = Order.transitive(Nat.add(first,kernel),Nat.add(last_row(output_end,stride),kernel),Nat.add(height,Nat.mul(2n,padding)),
    Order.add_le(first,last_row(output_end,stride),kernel,rows_ordered(output_first,output_end,stride,ordered)),last_room)
  # first + kernel <= height + padding + padding < height + padding + kernel.
  +strict = Order.le_lt_trans(Nat.add(first,kernel),Nat.add(Nat.add(height,padding),padding),Nat.add(Nat.add(height,padding),kernel),
    %Equal.sym(Nat,Nat.add(Nat.add(height,padding),padding),Nat.add(height,Nat.add(padding,padding)),Addition.associative(height,padding,padding)) :
      {Nat.is_le(Nat.add(first,kernel),_) == True{} : Bool}
    %two_times(padding) : {Nat.is_le(Nat.add(first,kernel),Nat.add(height,_)) == True{} : Bool}
    first_room,
    Order.offset_lt(Nat.add(height,padding),padding,kernel,padding_small))
  +below = {
    %Order.commutative(first,kernel) : {Nat.is_lt(_,Nat.add(kernel,Nat.add(height,padding))) == True{} : Bool}
    %Order.commutative(Nat.add(height,padding),kernel) : {Nat.is_lt(Nat.add(first,kernel),_) == True{} : Bool}
    strict : {Nat.is_lt(Nat.add(kernel,first),Nat.add(kernel,Nat.add(height,padding))) == True{} : Bool}}
  clipped_below(first,padding,height,height_positive,
    %Order.commutative(height,padding) : {Nat.is_lt(first,_) == True{} : Bool}
    Order.cancel_offset_lt(kernel,first,Nat.add(height,padding),below))

def restore_upper(+amount: Nat,+value: Nat) -> {Nat.is_le(value,Nat.add(amount,Nat.sub(value,amount))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(amount,Nat.sub(value,amount)),Nat.max(amount,value),Extrema.clipped_end(amount,value)) : {Nat.is_le(value,_) == True{} : Bool}
  Extrema.maximum_right(amount,value)

# The window starts before its end: the first row read precedes the last.
def start_before_end(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+output_first: Nat,+output_end: Nat,
  +span: Span<height,kernel,stride,padding,output_first,output_end>) ->
  {Nat.is_lt(window_first(height,output_first,stride,padding),window_end(height,output_end,stride,kernel,padding)) == True{} : Bool}:
  Span{stride_positive,height_positive,+padding_small,+ordered,last_room} = span
  +first = first_row(output_first,stride)
  +last = last_row(output_end,stride)
  +needed = TensorBandWindow.needed_first(height,output_first,stride,padding)
  +clipped = Nat.sub(first,padding)
  +strict = Extrema.minimum_strict_lower(height,Nat.sub(Nat.add(last,kernel),padding),clipped,
    first_inside(height,kernel,stride,padding,output_first,output_end,span),
    Order.le_lt_trans(clipped,Nat.sub(last,padding),Nat.sub(Nat.add(last,kernel),padding),
      DivisionOrder.subtract_le(first,last,padding,rows_ordered(output_first,output_end,stride,ordered)),shifted_below(last,padding,kernel,padding_small)))
  Order.le_lt_trans(window_first(height,output_first,stride,padding),clipped,window_end(height,output_end,stride,kernel,padding),
    Order.transitive(window_first(height,output_first,stride,padding),needed,clipped,aligned_below(needed,stride),Extrema.minimum_right(height,clipped)),strict)

def window_below_first(+height: Nat,+output_first: Nat,+stride: Nat,+padding: Nat) ->
  {Nat.is_le(window_first(height,output_first,stride,padding),Nat.sub(first_row(output_first,stride),padding)) == True{} : Bool}:
  +needed = TensorBandWindow.needed_first(height,output_first,stride,padding)
  Order.transitive(window_first(height,output_first,stride,padding),needed,Nat.sub(first_row(output_first,stride),padding),aligned_below(needed,stride),
    Extrema.minimum_right(height,Nat.sub(first_row(output_first,stride),padding)))

def padding_doubled(+padding: Nat) -> {Nat.is_le(padding,Nat.mul(2n,padding)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(2n,padding),Nat.add(padding,padding),two_times(padding)) : {Nat.is_le(padding,_) == True{} : Bool}
  Order.le_add(padding,padding)

# The window with padding on both sides fits the kernel.
def kernel_room(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+output_first: Nat,+output_end: Nat,
  +span: Span<height,kernel,stride,padding,output_first,output_end>) ->
  {Nat.is_le(Nat.add(window_first(height,output_first,stride,padding),kernel),Nat.add(window_end(height,output_end,stride,kernel,padding),Nat.mul(2n,padding))) == True{} : Bool}:
  Span{stride_positive,height_positive,+padding_small,+ordered,+last_room} = span
  +first = first_row(output_first,stride)
  +last = last_row(output_end,stride)
  +doubled = Nat.mul(2n,padding)
  +clipped_end = Nat.sub(Nat.add(last,kernel),padding)
  +start = window_first(height,output_first,stride,padding)
  +start_below = Order.transitive(start,Nat.sub(first,padding),first,window_below_first(height,output_first,stride,padding),Order.sub_le(first,padding))
  +first_last = Order.add_le(first,last,kernel,rows_ordered(output_first,output_end,stride,ordered))
  +padding_room = Bounds.strict_le(padding,Nat.add(last,kernel),Bounds.nat_trans(padding,kernel,Nat.add(last,kernel),padding_small,Samples.le_after(last,kernel)))
  +last_restored = {%Order.restore_shift(padding,Nat.add(last,kernel),padding_room) : {Nat.is_le(_,Nat.add(clipped_end,doubled)) == True{} : Bool}
    %Order.commutative(doubled,clipped_end) : {Nat.is_le(Nat.add(padding,clipped_end),_) == True{} : Bool}
    Order.add_le(padding,doubled,clipped_end,padding_doubled(padding)) : {Nat.is_le(Nat.add(last,kernel),Nat.add(clipped_end,doubled)) == True{} : Bool}}
  Order.transitive(Nat.add(start,kernel),Nat.add(first,kernel),Nat.add(window_end(height,output_end,stride,kernel,padding),doubled),Order.add_le(start,first,kernel,start_below),
    minimum_shift(height,clipped_end,doubled,Nat.add(first,kernel),Order.transitive(Nat.add(first,kernel),Nat.add(last,kernel),Nat.add(height,doubled),first_last,last_room),
      Order.transitive(Nat.add(first,kernel),Nat.add(last,kernel),Nat.add(clipped_end,doubled),first_last,last_restored)))

def top_when(ordered: Bool,+height: Nat,+stride: Nat,+padding: Nat,+output_first: Nat,+guard: {Nat.is_le(padding,first_row(output_first,stride)) == ordered : Bool}) ->
  Either<&2,&2,{window_first(height,output_first,stride,padding) == 0n : Nat},
    {Nat.is_le(Nat.add(window_first(height,output_first,stride,padding),padding),first_row(output_first,stride)) == True{} : Bool}>:
  match ordered:
    case True{}:
      +first = first_row(output_first,stride)
      +start = window_first(height,output_first,stride,padding)
      Inr{Order.transitive(Nat.add(start,padding),Nat.add(Nat.sub(first,padding),padding),first,
        Order.add_le(start,Nat.sub(first,padding),padding,window_below_first(height,output_first,stride,padding)),
        %Order.commutative(padding,Nat.sub(first,padding)) : {Nat.is_le(_,first) == True{} : Bool}
        %Order.restore_shift(padding,first,guard) : {Nat.is_le(Nat.add(padding,Nat.sub(first,padding)),_) == True{} : Bool}
        Order.reflexive(Nat.add(padding,Nat.sub(first,padding))))}
    case False{}:
      +first = first_row(output_first,stride)
      +start = window_first(height,output_first,stride,padding)
      Inl{zero_below(start,
        %sub_below(first,padding,not_le(padding,first,guard)) : {Nat.is_le(start,_) == True{} : Bool}
        window_below_first(height,output_first,stride,padding))}

def bottom_when(choice: Bool,+height: Nat,+kernel: Nat,+padding: Nat,+last: Nat,+row: Nat,
  +row_room: {Nat.is_le(row,last) == True{} : Bool}) ->
  Either<&2,&2,{height == Bool.pick(Nat,choice,height,Nat.sub(Nat.add(last,kernel),padding)) : Nat},
    {Nat.is_le(Nat.add(row,kernel),Nat.add(Bool.pick(Nat,choice,height,Nat.sub(Nat.add(last,kernel),padding)),padding)) == True{} : Bool}>:
  match choice:
    case True{}: Inl{{==}}
    case False{}:
      +clipped_end = Nat.sub(Nat.add(last,kernel),padding)
      Inr{Order.transitive(Nat.add(row,kernel),Nat.add(last,kernel),Nat.add(clipped_end,padding),Order.add_le(row,last,kernel,row_room),
        %Order.commutative(padding,clipped_end) : {Nat.is_le(Nat.add(last,kernel),_) == True{} : Bool}
        restore_upper(padding,Nat.add(last,kernel)))}

# The last output row reads inside the padded image.
def last_room(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+output_end: Nat,+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +kernel_fits: {Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool},+positive: {Nat.is_lt(0n,output_end) == True{} : Bool},
  +bounded: {Nat.is_le(output_end,ConvSpec.axis_size(height,kernel,stride,padding)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(last_row(output_end,stride),kernel),Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool}:
  +extent = Nat.add(height,Nat.mul(2n,padding))
  +reach = Nat.sub(extent,kernel)
  +rows = Nat.div(reach,stride)
  +previous = Nat.sub(output_end,1n)
  +previous_room = {%Equal.sym(Nat,1n+previous,output_end,Order.positive_has_last(output_end,positive)) : {Nat.is_le(_,1n+rows) == True{} : Bool}
    bounded : {Nat.is_le(1n+previous,1n+rows) == True{} : Bool}}
  +below = Order.transitive(Nat.mul(previous,stride),Nat.mul(rows,stride),reach,RowMajor.mul_le(previous,rows,stride,previous_room),
    DivisionOrder.product_below(reach,stride,stride_positive))
  Order.transitive(Nat.add(Nat.mul(previous,stride),kernel),Nat.add(reach,kernel),extent,Order.add_le(Nat.mul(previous,stride),reach,kernel,below),
    %Order.commutative(kernel,reach) : {Nat.is_le(_,extent) == True{} : Bool}
    %Order.restore_shift(kernel,extent,kernel_fits) : {Nat.is_le(Nat.add(kernel,reach),_) == True{} : Bool}
    Order.reflexive(Nat.add(kernel,reach)))

def before_whole(+whole: Nat,+positive: {Nat.is_lt(0n,whole) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(whole,1n),whole) == True{} : Bool}:
  %Order.positive_has_last(whole,positive) : {Nat.is_lt(Nat.sub(whole,1n),_) == True{} : Bool}
  Order.lt_succ(Nat.sub(whole,1n),Nat.sub(whole,1n),Order.reflexive(Nat.sub(whole,1n)))

# The positions of output rows [output_first, output_end) end at output_end rows.
def positions_end(+output_first: Nat,+output_end: Nat,+width: Nat,+ordered: {Nat.is_le(output_first,output_end) == True{} : Bool}) ->
  {Nat.add(Nat.mul(output_first,width),Nat.mul(Nat.sub(output_end,output_first),width)) == Nat.mul(output_end,width) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(output_first,width),Nat.mul(Nat.sub(output_end,output_first),width)),Nat.mul(Nat.add(output_first,Nat.sub(output_end,output_first)),width),
    Nat.mul(output_end,width),
    Equal.sym(Nat,Nat.mul(Nat.add(output_first,Nat.sub(output_end,output_first)),width),Nat.add(Nat.mul(output_first,width),Nat.mul(Nat.sub(output_end,output_first),width)),
      RowMajor.distrib(output_first,Nat.sub(output_end,output_first),width)),
    Equal.cong(Nat,Nat,rows => Nat.mul(rows,width),Nat.add(output_first,Nat.sub(output_end,output_first)),output_end,Order.restore_shift(output_first,output_end,ordered)))

# The last output row of the interval, in the Cover's row-major form.
def last_output(+output_first: Nat,+output_end: Nat,+width: Nat,+width_positive: {Nat.is_lt(0n,width) == True{} : Bool},
  +ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool}) ->
  {Nat.is_le(Nat.div(Nat.sub(Nat.add(Nat.mul(output_first,width),Nat.mul(Nat.sub(output_end,output_first),width)),1n),width),Nat.sub(output_end,1n)) == True{} : Bool}:
  +total = Nat.add(Nat.mul(output_first,width),Nat.mul(Nat.sub(output_end,output_first),width))
  +end_positive = Order.le_lt_trans(0n,output_first,output_end,Order.zero_le(output_first),ordered)
  +inside = {%Equal.sym(Nat,total,Nat.mul(output_end,width),positions_end(output_first,output_end,width,Bounds.strict_le(output_first,output_end,ordered))) :
      {Nat.is_lt(Nat.sub(_,1n),Nat.mul(output_end,width)) == True{} : Bool}
    before_whole(Nat.mul(output_end,width),Support.positive_mul(output_end,width,end_positive,width_positive)) :
    {Nat.is_lt(Nat.sub(total,1n),Nat.mul(output_end,width)) == True{} : Bool}}
  TensorBandWindow.before_last(Nat.div(Nat.sub(total,1n),width),output_end,Positions.quotient_lt(Nat.sub(total,1n),output_end,width,width_positive,inside))

def cover_facts(+ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,
  +input: List<&2,F32>,+weights: List<&2,F32>,+bias: List<&2,F32>,+facts: Guards.Valid<ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias}>,
  +padding_small: {Nat.is_lt(padding,kernel) == True{} : Bool},
  +output_first: Nat,+output_end: Nat,+ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},
  +bounded: {Nat.is_le(output_end,ConvSpec.axis_size(height,kernel,stride,padding)) == True{} : Bool}) ->
  Positions.Cover<ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias},window_first(height,output_first,stride,padding),
    Nat.sub(window_end(height,output_end,stride,kernel,padding),window_first(height,output_first,stride,padding)),
    Nat.mul(output_first,ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mul(Nat.sub(output_end,output_first),ConvSpec.axis_size(width,kernel,stride,padding))>:
  Guards.Valid{ci_positive,co_positive,+height_positive,width_positive,kernel_positive,+stride_positive,+kernel_fits,kernel_width,input_length,weight_length,bias_length} = facts
  +output_width = ConvSpec.axis_size(width,kernel,stride,padding)
  +width_ok = {{==} : {Nat.is_lt(0n,output_width) == True{} : Bool}}
  +end_positive = Order.le_lt_trans(0n,output_first,output_end,Order.zero_le(output_first),ordered)
  +span = {Span{stride_positive,height_positive,padding_small,ordered,last_room(height,kernel,stride,padding,output_end,stride_positive,kernel_fits,end_positive,bounded)} :
    Span<height,kernel,stride,padding,output_first,output_end>}
  +start = window_first(height,output_first,stride,padding)
  +finish = window_end(height,output_end,stride,kernel,padding)
  +clipped_end = Nat.sub(Nat.add(last_row(output_end,stride),kernel),padding)
  +rows = Nat.sub(finish,start)
  +doubled = Nat.mul(2n,padding)
  +before = start_before_end(height,kernel,stride,padding,output_first,output_end,span)
  +restored = Order.restore_shift(start,finish,Bounds.strict_le(start,finish,before))
  +needed = TensorBandWindow.needed_first(height,output_first,stride,padding)
  +fits = {%Equal.sym(Nat,Nat.add(start,rows),finish,restored) : {Nat.is_le(_,height) == True{} : Bool}
    Extrema.minimum_left(height,clipped_end) : {Nat.is_le(Nat.add(start,rows),height) == True{} : Bool}}
  +kernel_room_window = Order.cancel_offset_le(start,kernel,Nat.add(rows,doubled),
    %Addition.associative(start,rows,doubled) : {Nat.is_le(Nat.add(start,kernel),_) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(start,rows),finish,restored) : {Nat.is_le(Nat.add(start,kernel),Nat.add(_,doubled)) == True{} : Bool}
    kernel_room(height,kernel,stride,padding,output_first,output_end,span))
  +offset = Nat.mul(output_first,output_width)
  +positions = Nat.mul(Nat.sub(output_end,output_first),output_width)
  +top = {%Equal.sym(Nat,Nat.div(offset,output_width),output_first,Positions.exact_quotient(output_first,output_width,width_ok)) :
      Either<&2,&2,{start == 0n : Nat},{Nat.is_le(Nat.add(start,padding),Nat.mul(_,stride)) == True{} : Bool}>
    top_when(Nat.is_le(padding,first_row(output_first,stride)),height,stride,padding,output_first,{==}) :
    Either<&2,&2,{start == 0n : Nat},{Nat.is_le(Nat.add(start,padding),Nat.mul(Nat.div(offset,output_width),stride)) == True{} : Bool}>}
  +row = Nat.mul(Nat.div(Nat.sub(Nat.add(offset,positions),1n),output_width),stride)
  +row_room = RowMajor.mul_le(Nat.div(Nat.sub(Nat.add(offset,positions),1n),output_width),Nat.sub(output_end,1n),stride,
    last_output(output_first,output_end,output_width,width_ok,ordered))
  +bottom = {%Equal.sym(Nat,Nat.add(start,rows),finish,restored) :
      Either<&2,&2,{height == _ : Nat},{Nat.is_le(Nat.add(row,kernel),Nat.add(_,padding)) == True{} : Bool}>
    %Equal.sym(Nat,finish,Bool.pick(Nat,Nat.is_lt(height,clipped_end),height,clipped_end),Extrema.minimum_choice(height,clipped_end)) :
      Either<&2,&2,{height == _ : Nat},{Nat.is_le(Nat.add(row,kernel),Nat.add(_,padding)) == True{} : Bool}>
    bottom_when(Nat.is_lt(height,clipped_end),height,kernel,padding,last_row(output_end,stride),row,row_room) :
    Either<&2,&2,{height == Nat.add(start,rows) : Nat},{Nat.is_le(Nat.add(row,kernel),Nat.add(Nat.add(start,rows),padding)) == True{} : Bool}>}
  Positions.Cover{Nat.div(needed,stride),aligned_exact(needed,stride,stride_positive),fits,Order.positive_difference(finish,start,before),kernel_room_window,top,bottom}

# The actual window of output rows [output_first, output_end) covers them.
def cover(+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +padding_small: {Nat.is_lt(Guards.pad(problem),Guards.kernel(problem)) == True{} : Bool},
  +output_first: Nat,+output_end: Nat,+ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},
  +bounded: {Nat.is_le(output_end,ConvSpec.out_height(problem)) == True{} : Bool}) ->
  Positions.Cover<problem,window_first(Guards.height(problem),output_first,Guards.stride(problem),Guards.pad(problem)),
    Nat.sub(window_end(Guards.height(problem),output_end,Guards.stride(problem),Guards.kernel(problem),Guards.pad(problem)),
      window_first(Guards.height(problem),output_first,Guards.stride(problem),Guards.pad(problem))),
    Nat.mul(output_first,ConvSpec.out_width(problem)),Nat.mul(Nat.sub(output_end,output_first),ConvSpec.out_width(problem))>:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +facts = Guards.valid(problem,valid)
  cover_facts(ci,co,height,width,kernel,stride,padding,input,weights,bias,facts,padding_small,output_first,output_end,ordered,bounded)
