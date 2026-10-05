# One sweep over a band leaf. A leaf stores some rows of every channel of an
# image batch and names its cells by channel, row and column. A sweep along
# the columns of a leaf, or along its rows, writes a leaf of the swept image:
# every output cell holds the fold of the line of its pixel over the logical
# input. The stored rows are addressed inside the leaf and the logical input
# inside the whole batch; each sample is the same cell on both sides. The
# sweep along the columns needs only bounds. The sweep along the rows reads a
# window of rows, so its caller states, for every output row and tap, that
# the row read inside the window is the row of the image.
import Base
import ../tensor_access.bend as Access
import ../tensor_band.bend as Bands
import ../machine_limits.bend as Machine
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./tensor_band_values.bend as BandValues
import ./tensor_sweep_refinement.bend as SweepLaws
import ./tensor_sweep_values.bend as Sweeps
import ./traversal_indexed_values.bend as IndexedValues
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./tensor_shape_refinement.bend as ShapeLaws
import ./encoded_coordinates.bend as Encoded
import ./machine_limit_bounds.bend as Limits
import ./boolean_logic.bend as Boolean
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges

# A sample that is valid on both sides reads the stored cell locally and the
# logical value globally; an invalid sample is the same pad.
def moved_sample(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+input: Tree.Tree<Element>,+pad: Element,valid: Bool,+local: U32,+global: U32,
  read: Boolean.When(valid,{Tree.read(Element,input,local) == original(logical,global) : Element})) ->
  {SweepLaws.sample_when(~Element,pad,valid,input,local) == Sweeps.logical_when(~Element,~Logical,~original,pad,valid,logical,global) : Element}:
  match valid:
    case False{}: {==}
    case True{}: read

# One tap of two lines with the same step and padding: the line of the leaf
# and the line of the logical input.
def moved_tap(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+input: Tree.Tree<Element>,+pad: Element,+step: U32,+padding: U32,+tap: U32,
  +local_base: U32,+local_first: U32,+local_extent: U32,+global_base: U32,+global_first: U32,+global_extent: U32,
  +same: {U32.is_lt((local_first + tap - padding : U32),local_extent) == U32.is_lt((global_first + tap - padding : U32),global_extent) : Bool},
  read: Boolean.When(U32.is_lt((global_first + tap - padding : U32),global_extent),
    {Tree.read(Element,input,(local_base + (local_first + tap - padding : U32) * step : U32)) == original(logical,(global_base + (global_first + tap - padding : U32) * step : U32)) : Element})) ->
  {SweepLaws.sample_line(~Element,input,Access.Line{local_base,local_first,local_extent,step,padding,pad},tap) ==
    Sweeps.logical_line(~Element,~Logical,~original,logical,Access.Line{global_base,global_first,global_extent,step,padding,pad},tap) : Element}:
  %Equal.sym(Bool,U32.is_lt((local_first + tap - padding : U32),local_extent),U32.is_lt((global_first + tap - padding : U32),global_extent),same) :
    {SweepLaws.sample_when(~Element,pad,_,input,(local_base + (local_first + tap - padding : U32) * step : U32)) ==
      Sweeps.logical_when(~Element,~Logical,~original,pad,U32.is_lt((global_first + tap - padding : U32),global_extent),logical,(global_base + (global_first + tap - padding : U32) * step : U32)) : Element}
  moved_sample(~Element,~Logical,~original,logical,input,pad,U32.is_lt((global_first + tap - padding : U32),global_extent),
    (local_base + (local_first + tap - padding : U32) * step : U32),(global_base + (global_first + tap - padding : U32) * step : U32),read)

# The cell a sweep writes: inside the output, with the swept sample of its
# position, which the caller relates to the value of its pixel.
def swept_at(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,~operation: Element -> Element -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,
  +local: Access.SweepContext<Element>,+input: Tree.Tree<Element>,+output: Buffer.Model<Element>,+channel: Nat,+row: Nat,+column: Nat,
  ready: Buffer.Ready<Element,output>,
  +selected: {Nat.is_lt(BandValues.position(geometry,rows,channel,row,column),Buffer.length(~Element,output)) == True{} : Bool},
  sample: {SweepLaws.sample_swept(~Element,~operation,input,local,U32.from_nat(BandValues.position(geometry,rows,channel,row,column))) ==
    value(context,U32.from_nat(BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))) : Element}) ->
  BandValues.At(~Element,~Context,~value,context,geometry,first,rows,Sweeps.run(~Element,~operation,Buffer.length(~Element,output),local,input,Buffer.tree(~Element,output)),Buffer.length(~Element,output),channel,row,column):
  Buffer.Model{+output_tree,+length,+certificate} = output
  Buffer.Ready{+depth_ok,room} = ready
  +output = {Buffer.Model{output_tree,length,certificate} : Buffer.Model<Element>}
  +position = BandValues.position(geometry,rows,channel,row,column)
  Shape.Both{selected,
    Equal.trans(Element,Tree.read(Element,Sweeps.run(~Element,~operation,length,local,input,output_tree),U32.from_nat(position)),
      SweepLaws.sample_swept(~Element,~operation,input,local,U32.from_nat(position)),
      value(context,U32.from_nat(BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))),
      IndexedValues.stored(~Tree.Tree<Element>,~Element,~Access.SweepContext<Element>,~SweepLaws.sample_swept(~Element,~operation),~Access.swept_index(~Element),~Sweeps.identity(~Element),
        length,local,input,output_tree,Buffer.depth(~Element,output),Tree.capacity(Element,output_tree),position,
        {==},Buffer.balanced(~Element,output),depth_ok,Buffer.capacity_bound(~Element,output,depth_ok),room,selected),
      sample)}

def unit_collapse(+value: Nat) -> {Nat.add(Nat.mul(value,1n),0n) == value : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(value,1n),0n),Nat.mul(value,1n),value,Addition.zero_right(Nat.mul(value,1n)),ShapeLaws.times_one(value))

# The row-major index with step one is the index of a list of lines.
def unit_index(+block: Nat,+extent: Nat,+coordinate: Nat) -> {RowMajor.index(block,extent,1n,coordinate,0n) == Nat.add(Nat.mul(block,extent),coordinate) : Nat}:
  %Equal.sym(Nat,Nat.mul(extent,1n),extent,ShapeLaws.times_one(extent)) : {Nat.add(Nat.mul(block,_),Nat.add(Nat.mul(coordinate,1n),0n)) == Nat.add(Nat.mul(block,extent),coordinate) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.mul(coordinate,1n),0n),coordinate,unit_collapse(coordinate)) : {Nat.add(Nat.mul(block,extent),_) == Nat.add(Nat.mul(block,extent),coordinate) : Nat}
  {==}

def unit_room(+outer: Nat,+rows: Nat,+width: Nat,+limit: Nat,room: {Nat.is_le(Nat.mul(outer,Nat.mul(rows,width)),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.mul(outer,rows),Nat.mul(width,1n)),limit) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(width,1n),width,ShapeLaws.times_one(width)) : {Nat.is_le(Nat.mul(Nat.mul(outer,rows),_),limit) == True{} : Bool}
  %Equal.sym(Nat,Nat.mul(Nat.mul(outer,rows),width),Nat.mul(outer,Nat.mul(rows,width)),RowMajor.assoc(outer,rows,width)) : {Nat.is_le(_,limit) == True{} : Bool}
  room

def grouped(+batch: Nat,+channels: Nat,+rest: Nat,+limit: Nat,room: {Nat.is_le(Nat.mul(batch,Nat.mul(channels,rest)),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.mul(batch,channels),rest),limit) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(Nat.mul(batch,channels),rest),Nat.mul(batch,Nat.mul(channels,rest)),RowMajor.assoc(batch,channels,rest)) : {Nat.is_le(_,limit) == True{} : Bool}
  room

def inner_room(+outer: Nat,+middle: Nat,+inner: Nat,+limit: Nat,outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},middle_positive: {Nat.is_lt(0n,middle) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(outer,Nat.mul(middle,inner)),limit) == True{} : Bool}) -> {Nat.is_le(inner,limit) == True{} : Bool}:
  Order.transitive(inner,Nat.mul(middle,inner),limit,
    %RowMajor.multiply_commutative(inner,middle) : {Nat.is_le(inner,_) == True{} : Bool}
    RowMajor.product_covers(inner,middle,middle_positive),
    Order.transitive(Nat.mul(middle,inner),Nat.mul(outer,Nat.mul(middle,inner)),limit,
      %RowMajor.multiply_commutative(Nat.mul(middle,inner),outer) : {Nat.is_le(Nat.mul(middle,inner),_) == True{} : Bool}
      RowMajor.product_covers(Nat.mul(middle,inner),outer,outer_positive),room))

def middle_room(+outer: Nat,+middle: Nat,+inner: Nat,+limit: Nat,outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(outer,Nat.mul(middle,inner)),limit) == True{} : Bool}) -> {Nat.is_le(middle,limit) == True{} : Bool}:
  Order.transitive(middle,Nat.mul(middle,inner),limit,RowMajor.product_covers(middle,inner,inner_positive),
    Order.transitive(Nat.mul(middle,inner),Nat.mul(outer,Nat.mul(middle,inner)),limit,
      %RowMajor.multiply_commutative(Nat.mul(middle,inner),outer) : {Nat.is_le(Nat.mul(middle,inner),_) == True{} : Bool}
      RowMajor.product_covers(Nat.mul(middle,inner),outer,outer_positive),room))

def positive(+value: Nat,+limit: Nat,inside: {Nat.is_lt(value,limit) == True{} : Bool}) -> {Nat.is_lt(0n,limit) == True{} : Bool}:
  Order.le_lt_trans(0n,value,limit,Order.zero_le(value),inside)

# The native digits of the position of a cell.
def digits(+value: Nat,+step: U32,+out_extent: U32,+block: Nat,+along: Nat,+offset: Nat,+limits: Machine.Limits,
  +exact: {Nat.add(Nat.mul(Nat.add(Nat.mul(block,U32.to_nat(out_extent)),along),U32.to_nat(step)),offset) == value : Nat},
  +room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +along_inside: {Nat.is_lt(along,U32.to_nat(out_extent)) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(step)) == True{} : Bool},
  +step_room: {Nat.is_le(U32.to_nat(step),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_room: {Nat.is_le(U32.to_nat(out_extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Sweeps.Decoded<U32.from_nat(value),step,out_extent,block,along,offset>:
  Sweeps.decoded(U32.from_nat(value),step,out_extent,block,along,offset,limits,
    Equal.trans(Nat,U32.to_nat(U32.from_nat(value)),value,Nat.add(Nat.mul(Nat.add(Nat.mul(block,U32.to_nat(out_extent)),along),U32.to_nat(step)),offset),
      Bounds.roundtrip(value,Machine.capacity(limits),Limits.capacity_fits(limits),room),
      Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(block,U32.to_nat(out_extent)),along),U32.to_nat(step)),offset),value,exact)),
    along_inside,offset_inside,step_room,out_room)

# A row of a leaf that fits in the batch lies in one of its images.
def image_below(+first: Nat,+rows: Nat,+row: Nat,+batch: Nat,+height: Nat,+height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
  row_inside: {Nat.is_lt(row,rows) == True{} : Bool},rows_fit: {Nat.is_le(Nat.add(first,rows),Nat.mul(batch,height)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(Nat.add(first,row),height),batch) == True{} : Bool}:
  RowMajor.cancel_multiply_lt(Nat.div(Nat.add(first,row),height),batch,height,
    Order.le_lt_trans(Nat.mul(Nat.div(Nat.add(first,row),height),height),Nat.add(first,row),Nat.mul(batch,height),
      DivisionOrder.product_below(Nat.add(first,row),height,height_positive),
      Bounds.nat_trans(Nat.add(first,row),Nat.add(first,rows),Nat.mul(batch,height),Order.offset_lt(first,row,rows,row_inside),rows_fit)))

# The sweep along the columns. A valid sample of an output cell is a cell of
# the same stored row, and of the same row of the image.
def column_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+batch: U32,+channels: U32,+height: U32,+width: U32,+out_width: U32,+first: U32,+rows: U32,
  +input: Tree.Tree<Element>,+length: Nat,+limits: Machine.Limits,+channel: Nat,+row: Nat,+coordinate: U32,valid: Bool,+local: U32,+global: U32,
  +exact: {U32.is_lt(coordinate,width) == valid : Bool},
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},first,rows,input,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +image_inside: {Nat.is_lt(Nat.div(Nat.add(U32.to_nat(first),row),U32.to_nat(height)),U32.to_nat(batch)) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +local_block: {U32.to_nat(((local / 1 : U32) / out_width : U32)) == Nat.add(Nat.mul(channel,U32.to_nat(rows)),row) : Nat},
  +local_offset: {U32.to_nat((local % 1 : U32)) == 0n : Nat},
  +global_block: {U32.to_nat(((global / 1 : U32) / out_width : U32)) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.add(U32.to_nat(first),row),U32.to_nat(height)),U32.to_nat(channels)),channel),U32.to_nat(height)),Nat.mod(Nat.add(U32.to_nat(first),row),U32.to_nat(height))) : Nat},
  +global_offset: {U32.to_nat((global % 1 : U32)) == 0n : Nat},
  +local_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +global_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(batch),U32.to_nat(channels)),Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Boolean.When(valid,{Tree.read(Element,input,(((local / 1 : U32) / out_width : U32) * width * 1 + (local % 1 : U32) + coordinate * 1 : U32)) ==
    original(logical,(((global / 1 : U32) / out_width : U32) * width * 1 + (global % 1 : U32) + coordinate * 1 : U32)) : Element}):
  match valid:
    case False{}: Unit{}
    case True{}:
      +source = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
      +count = U32.to_nat(channels)
      +stored = U32.to_nat(rows)
      +tall = U32.to_nat(height)
      +wide = U32.to_nat(width)
      +limit = U32.to_nat(Machine.capacity(limits))
      +sample = U32.to_nat(coordinate)
      +whole = Nat.add(U32.to_nat(first),row)
      +image = Nat.div(whole,tall)
      +inner = Nat.mod(whole,tall)
      +plane = Nat.add(Nat.mul(image,count),channel)
      +local_line = Nat.add(Nat.mul(channel,stored),row)
      +global_line = Nat.add(Nat.mul(plane,tall),inner)
      +valid_nat = Bounds.as_nat_lt(coordinate,width,exact)
      +local_word = (((local / 1 : U32) / out_width : U32) * width * 1 + (local % 1 : U32) + coordinate * 1 : U32)
      +global_word = (((global / 1 : U32) / out_width : U32) * width * 1 + (global % 1 : U32) + coordinate * 1 : U32)
      +position = BandValues.position(source,rows,channel,row,sample)
      +pixel = BandValues.pixel(source,channel,whole,sample)
      +local_exact = Equal.trans(Nat,U32.to_nat(local_word),RowMajor.index(local_line,wide,1n,sample,0n),position,
        Sweeps.address_exact(((local / 1 : U32) / out_width : U32),width,1,(local % 1 : U32),coordinate,local_line,0n,Nat.mul(count,stored),limits,local_block,local_offset,
          RowMajor.radix(channel,count,row,stored,channel_inside,row_inside),{==},valid_nat,unit_room(count,stored,wide,limit,local_room)),
        Equal.trans(Nat,RowMajor.index(local_line,wide,1n,sample,0n),Nat.add(Nat.mul(local_line,wide),sample),position,unit_index(local_line,wide,sample),RowMajor.mixed(channel,stored,wide,row,sample)))
      +global_exact = Equal.trans(Nat,U32.to_nat(global_word),RowMajor.index(global_line,wide,1n,sample,0n),pixel,
        Sweeps.address_exact(((global / 1 : U32) / out_width : U32),width,1,(global % 1 : U32),coordinate,global_line,0n,Nat.mul(Nat.mul(U32.to_nat(batch),count),tall),limits,global_block,global_offset,
          RowMajor.radix(plane,Nat.mul(U32.to_nat(batch),count),inner,tall,RowMajor.radix(image,U32.to_nat(batch),channel,count,image_inside,channel_inside),DivisionOrder.remainder_small(whole,tall,height_positive)),
          {==},valid_nat,unit_room(Nat.mul(U32.to_nat(batch),count),tall,wide,limit,global_room)),
        Equal.trans(Nat,RowMajor.index(global_line,wide,1n,sample,0n),Nat.add(Nat.mul(global_line,wide),sample),pixel,unit_index(global_line,wide,sample),RowMajor.mixed(plane,tall,wide,inner,sample)))
      %Equal.sym(U32,local_word,U32.from_nat(position),Encoded.encoding(local_word,position,local_exact)) : {Tree.read(Element,input,_) == original(logical,global_word) : Element}
      %Equal.sym(U32,global_word,U32.from_nat(pixel),Encoded.encoding(global_word,pixel,global_exact)) : {Tree.read(Element,input,U32.from_nat(position)) == original(logical,_) : Element}
      BandValues.leaf_read(~Element,~Logical,~original,logical,source,first,rows,input,length,channel,row,sample,values,channel_inside,row_inside,valid_nat)

# One tap of the columns sweep, for any two positions with the digits of the
# cell: the position inside the leaf and the position inside the batch.
def column_moved(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+pad: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,+stride: U32,+padding: U32,+out_width: U32,
  +first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+limits: Machine.Limits,+channel: Nat,+row: Nat,+column: Nat,+tap: U32,+local: U32,+global: U32,
  local_digits: Sweeps.Decoded<local,1,out_width,Nat.add(Nat.mul(channel,U32.to_nat(rows)),row),column,0n>,
  global_digits: Sweeps.Decoded<global,1,out_width,Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.add(U32.to_nat(first),row),U32.to_nat(height)),U32.to_nat(channels)),channel),U32.to_nat(height)),Nat.mod(Nat.add(U32.to_nat(first),row),U32.to_nat(height))),column,0n>,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},first,rows,input,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +image_inside: {Nat.is_lt(Nat.div(Nat.add(U32.to_nat(first),row),U32.to_nat(height)),U32.to_nat(batch)) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +local_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +global_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(batch),U32.to_nat(channels)),Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {SweepLaws.sample_line(~Element,input,Access.Line{((local / 1 : U32) / out_width * width * 1 + local % 1 : U32),((local / 1 : U32) % out_width * stride + 0 : U32),width,1,padding,pad},tap) ==
    Sweeps.logical_line(~Element,~Logical,~original,logical,Access.Line{((global / 1 : U32) / out_width * width * 1 + global % 1 : U32),((global / 1 : U32) % out_width * stride + 0 : U32),width,1,padding,pad},tap) : Element}:
  Sweeps.Decoded{+local_block,+local_along,+local_offset} = local_digits
  Sweeps.Decoded{+global_block,+global_along,+global_offset} = global_digits
  +local_base = ((local / 1 : U32) / out_width * width * 1 + local % 1 : U32)
  +global_base = ((global / 1 : U32) / out_width * width * 1 + global % 1 : U32)
  +start = (U32.from_nat(column) * stride + 0 : U32)
  %Equal.sym(U32,((local / 1 : U32) % out_width : U32),U32.from_nat(column),Encoded.encoding(((local / 1 : U32) % out_width : U32),column,local_along)) :
    {SweepLaws.sample_line(~Element,input,Access.Line{local_base,(_ * stride + 0 : U32),width,1,padding,pad},tap) ==
      Sweeps.logical_line(~Element,~Logical,~original,logical,Access.Line{global_base,((global / 1 : U32) % out_width * stride + 0 : U32),width,1,padding,pad},tap) : Element}
  %Equal.sym(U32,((global / 1 : U32) % out_width : U32),U32.from_nat(column),Encoded.encoding(((global / 1 : U32) % out_width : U32),column,global_along)) :
    {SweepLaws.sample_line(~Element,input,Access.Line{local_base,start,width,1,padding,pad},tap) ==
      Sweeps.logical_line(~Element,~Logical,~original,logical,Access.Line{global_base,(_ * stride + 0 : U32),width,1,padding,pad},tap) : Element}
  moved_tap(~Element,~Logical,~original,logical,input,pad,1,padding,tap,local_base,start,width,global_base,start,width,{==},
    column_read(~Element,~Logical,~original,logical,batch,channels,height,width,out_width,first,rows,input,length,limits,channel,row,(start + tap - padding : U32),U32.is_lt((start + tap - padding : U32),width),local,global,
      {==},values,channel_inside,row_inside,image_inside,height_positive,local_block,local_offset,global_block,global_offset,local_room,global_room))

def column_tap(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+pad: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_width: U32,
  +first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+limits: Machine.Limits,+channel: Nat,+row: Nat,+column: Nat,+tap: U32,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},first,rows,input,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(out_width)) == True{} : Bool},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +planes_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(out_width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_input_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_planes_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {SweepLaws.sample_line(~Element,input,Access.line_at(~Element,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad,U32.from_nat(BandValues.position(Bands.Geometry{batch,channels,height,out_width},rows,channel,row,column))),tap) ==
    Sweeps.logical_line(~Element,~Logical,~original,logical,Access.line_at(~Element,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad,U32.from_nat(BandValues.pixel(Bands.Geometry{batch,channels,height,out_width},channel,Nat.add(U32.to_nat(first),row),column))),tap) : Element}:
  +planes = {Bands.Geometry{batch,channels,height,out_width} : Bands.Geometry}
  +images = U32.to_nat(batch)
  +count = U32.to_nat(channels)
  +stored = U32.to_nat(rows)
  +tall = U32.to_nat(height)
  +wide = U32.to_nat(out_width)
  +limit = U32.to_nat(Machine.capacity(limits))
  +whole = Nat.add(U32.to_nat(first),row)
  +image = Nat.div(whole,tall)
  +inner = Nat.mod(whole,tall)
  +plane = Nat.add(Nat.mul(image,count),channel)
  +local_line = Nat.add(Nat.mul(channel,stored),row)
  +global_line = Nat.add(Nat.mul(plane,tall),inner)
  +position = BandValues.position(planes,rows,channel,row,column)
  +pixel = BandValues.pixel(planes,channel,whole,column)
  +local = U32.from_nat(position)
  +global = U32.from_nat(pixel)
  +image_ok = image_below(U32.to_nat(first),stored,row,images,tall,height_positive,row_inside,rows_fit)
  +inner_ok = DivisionOrder.remainder_small(whole,tall,height_positive)
  +plane_ok = RowMajor.radix(image,images,channel,count,image_ok,channel_inside)
  +out_room = inner_room(count,stored,wide,limit,positive(channel,count,channel_inside),positive(row,stored,row_inside),local_planes_room)
  +one_room = Order.transitive(1n,wide,limit,Order.strict_follow(0n,wide,positive(column,wide,column_inside)),out_room)
  +grouped_planes = grouped(images,count,Nat.mul(tall,wide),limit,planes_room)
  +local_digits = digits(position,1,out_width,local_line,column,0n,limits,
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(local_line,wide),column),1n),0n),Nat.add(Nat.mul(local_line,wide),column),position,
      unit_collapse(Nat.add(Nat.mul(local_line,wide),column)),RowMajor.mixed(channel,stored,wide,row,column)),
    Bounds.strict_le(position,limit,Bounds.nat_trans(position,Nat.mul(count,Nat.mul(stored,wide)),limit,RowMajor.in_bounds(channel,count,stored,wide,row,column,channel_inside,row_inside,column_inside),local_planes_room)),
    column_inside,{==},one_room,out_room)
  +global_digits = digits(pixel,1,out_width,global_line,column,0n,limits,
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(global_line,wide),column),1n),0n),Nat.add(Nat.mul(global_line,wide),column),pixel,
      unit_collapse(Nat.add(Nat.mul(global_line,wide),column)),RowMajor.mixed(plane,tall,wide,inner,column)),
    Bounds.strict_le(pixel,limit,Bounds.nat_trans(pixel,Nat.mul(Nat.mul(images,count),Nat.mul(tall,wide)),limit,RowMajor.in_bounds(plane,Nat.mul(images,count),tall,wide,inner,column,plane_ok,inner_ok,column_inside),grouped_planes)),
    column_inside,{==},one_room,out_room)
  column_moved(~Element,~Logical,~original,logical,pad,batch,channels,height,width,stride,padding,out_width,first,rows,input,length,limits,channel,row,column,tap,local,global,local_digits,global_digits,
    values,channel_inside,row_inside,image_ok,height_positive,local_input_room,grouped(images,count,Nat.mul(tall,U32.to_nat(width)),limit,input_room))

def column_taps(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,remaining: Nat,+tap: U32,+logical: Logical,+pad: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_width: U32,
  +first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+limits: Machine.Limits,+channel: Nat,+row: Nat,+column: Nat,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},first,rows,input,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(out_width)) == True{} : Bool},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +planes_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(out_width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_input_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_planes_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Sweeps.TapFacts(~Element,~Logical,~original,logical,input,
    Access.line_at(~Element,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad,U32.from_nat(BandValues.position(Bands.Geometry{batch,channels,height,out_width},rows,channel,row,column))),
    Access.line_at(~Element,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad,U32.from_nat(BandValues.pixel(Bands.Geometry{batch,channels,height,out_width},channel,Nat.add(U32.to_nat(first),row),column))),remaining,tap):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{column_tap(~Element,~Logical,~original,logical,pad,batch,channels,height,width,kernel,stride,padding,out_width,first,rows,input,length,limits,channel,row,column,tap,
          values,channel_inside,row_inside,column_inside,rows_fit,height_positive,input_room,planes_room,local_input_room,local_planes_room),
        column_taps(~Element,~Logical,~original,rest,(tap + 1 : U32),logical,pad,batch,channels,height,width,kernel,stride,padding,out_width,first,rows,input,length,limits,channel,row,column,
          values,channel_inside,row_inside,column_inside,rows_fit,height_positive,input_room,planes_room,local_input_room,local_planes_room)}

# What the columns sweep of a leaf supplies to every cell of its output.
type ColumnCells<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  ColumnCells{logical: Logical,pad: Element,batch: U32,channels: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,out_width: U32,first: U32,rows: U32,
    input: Tree.Tree<Element>,length: Nat,output: Buffer.Model<Element>,limits: Machine.Limits,ready: Buffer.Ready<Element,output>,
    values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},first,rows,input,length),
    length_exact: {Buffer.length(~Element,output) == BandValues.cell_count(Bands.Geometry{batch,channels,height,out_width},rows) : Nat},
    rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) == True{} : Bool},
    height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
    input_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    planes_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(out_width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    local_input_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    local_planes_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def column_geometry(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: ColumnCells<Element,Logical,original>) -> Bands.Geometry:
  ColumnCells{logical,pad,batch,channels,height,width,kernel,stride,padding,out_width,first,rows,input,length,output,limits,ready,values,length_exact,rows_fit,height_positive,input_room,planes_room,local_input_room,local_planes_room} = evidence
  Bands.Geometry{batch,channels,height,out_width}

def column_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: ColumnCells<Element,Logical,original>) -> U32:
  ColumnCells{logical,pad,batch,channels,height,width,kernel,stride,padding,out_width,first,rows,input,length,output,limits,ready,values,length_exact,rows_fit,height_positive,input_room,planes_room,local_input_room,local_planes_room} = evidence
  rows

def ColumnCell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+evidence: ColumnCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  ColumnCells{+logical,+pad,+batch,+channels,+height,+width,+kernel,+stride,+padding,+out_width,+first,+rows,+input,+length,+output,+limits,+ready,+values,+length_exact,+rows_fit,+height_positive,+input_room,+planes_room,+local_input_room,+local_planes_room} = evidence
  BandValues.At(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},
    Bands.Geometry{batch,channels,height,out_width},first,rows,
    Sweeps.run(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},input,Buffer.tree(~Element,output)),Buffer.length(~Element,output),channel,row,column)

def column_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+evidence: ColumnCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(column_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(column_rows(~Element,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(column_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool}) ->
  ColumnCell(~Element,~Logical,~original,~operation,evidence,channel,row,column):
  ColumnCells{+logical,+pad,+batch,+channels,+height,+width,+kernel,+stride,+padding,+out_width,+first,+rows,+input,+length,+output,+limits,+ready,+values,+length_exact,+rows_fit,+height_positive,+input_room,+planes_room,+local_input_room,+local_planes_room} = evidence
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  +planes = {Bands.Geometry{batch,channels,height,out_width} : Bands.Geometry}
  +sweep = {Access.Sweep{width,1,kernel,stride,padding,out_width,0} : Access.Sweep}
  swept_at(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation,Sweeps.Swept{logical,sweep,pad},planes,first,rows,Access.SweepContext{sweep,pad},input,output,channel,row,column,ready,
    BandValues.position_within(planes,rows,Buffer.length(~Element,output),channel,row,column,length_exact,channel_inside,row_inside,column_inside),
    Sweeps.taps_match(~Element,~Logical,~original,~operation,U32.to_nat(kernel),0,logical,input,
      Access.line_at(~Element,sweep,pad,U32.from_nat(BandValues.position(planes,rows,channel,row,column))),
      Access.line_at(~Element,sweep,pad,U32.from_nat(BandValues.pixel(planes,channel,Nat.add(U32.to_nat(first),row),column))),pad,
      column_taps(~Element,~Logical,~original,U32.to_nat(kernel),0,logical,pad,batch,channels,height,width,kernel,stride,padding,out_width,first,rows,input,length,limits,channel,row,column,
        values,channel_inside,row_inside,column_inside,rows_fit,height_positive,input_room,planes_room,local_input_room,local_planes_room)))

# The columns sweep of a leaf of an image batch is the leaf, with the same
# rows, of the batch swept along its columns.
def columns_leaf(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +kernel: U32,+stride: U32,+padding: U32,+out_width: U32,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+output: Buffer.Model<Element>,+limits: Machine.Limits,
  +ready: Buffer.Ready<Element,output>,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},first,rows,input,length),
  +length_exact: {Buffer.length(~Element,output) == BandValues.cell_count(Bands.Geometry{batch,channels,height,out_width},rows) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +planes_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(out_width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_input_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_planes_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  BandValues.Leaf(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},
    Bands.Geometry{batch,channels,height,out_width},first,rows,
    Sweeps.run(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},input,Buffer.tree(~Element,output)),Buffer.length(~Element,output)):
  BandValues.cells(~ColumnCells<Element,Logical,original>,~ColumnCell(~Element,~Logical,~original,~operation),~column_geometry(~Element,~Logical,~original),~column_rows(~Element,~Logical,~original),~column_cell(~Element,~Logical,~original,~operation),
    ColumnCells{logical,pad,batch,channels,height,width,kernel,stride,padding,out_width,first,rows,input,length,output,limits,ready,values,length_exact,rows_fit,height_positive,input_room,planes_room,local_input_room,local_planes_room})

# The sweep along the rows. A valid sample of an output cell is a cell of the
# same column of a stored row; the caller's fact says which row of the image
# that stored row is.
def row_read(~Element: Data,~Inner: Data,~inner_value: Inner -> U32 -> Element,+inner: Inner,+batch: U32,+channels: U32,+height: U32,+width: U32,+window_first: U32,+window_rows: U32,+out_height: U32,+output_rows: U32,
  +input: Tree.Tree<Element>,+length: Nat,+limits: Machine.Limits,+channel: Nat,+column: Nat,+image: Nat,+local_coordinate: U32,+global_coordinate: U32,valid: Bool,+local: U32,+global: U32,
  +exact: {U32.is_lt(global_coordinate,height) == valid : Bool},
  +same: {U32.is_lt(local_coordinate,window_rows) == U32.is_lt(global_coordinate,height) : Bool},
  located: Boolean.When(valid,{Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat(global_coordinate)) == Nat.add(U32.to_nat(window_first),U32.to_nat(local_coordinate)) : Nat}),
  +values: BandValues.Leaf(~Element,~Inner,~inner_value,inner,Bands.Geometry{batch,channels,height,width},window_first,window_rows,input,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,U32.to_nat(batch)) == True{} : Bool},
  +local_block: {U32.to_nat(((local / width : U32) / output_rows : U32)) == channel : Nat},
  +local_offset: {U32.to_nat((local % width : U32)) == column : Nat},
  +global_block: {U32.to_nat(((global / width : U32) / out_height : U32)) == Nat.add(Nat.mul(image,U32.to_nat(channels)),channel) : Nat},
  +global_offset: {U32.to_nat((global % width : U32)) == column : Nat},
  +local_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +global_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(batch),U32.to_nat(channels)),Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Boolean.When(valid,{Tree.read(Element,input,(((local / width : U32) / output_rows : U32) * window_rows * width + (local % width : U32) + local_coordinate * width : U32)) ==
    inner_value(inner,(((global / width : U32) / out_height : U32) * height * width + (global % width : U32) + global_coordinate * width : U32)) : Element}):
  match valid:
    case False{}: Unit{}
    case True{}:
      +source = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
      +count = U32.to_nat(channels)
      +near = U32.to_nat(local_coordinate)
      +far = U32.to_nat(global_coordinate)
      +stored = Nat.add(U32.to_nat(window_first),near)
      +global_valid = Bounds.as_nat_lt(global_coordinate,height,exact)
      +local_valid = Bounds.as_nat_lt(local_coordinate,window_rows,Equal.trans(Bool,U32.is_lt(local_coordinate,window_rows),U32.is_lt(global_coordinate,height),True{},same,exact))
      +local_word = (((local / width : U32) / output_rows : U32) * window_rows * width + (local % width : U32) + local_coordinate * width : U32)
      +global_word = (((global / width : U32) / out_height : U32) * height * width + (global % width : U32) + global_coordinate * width : U32)
      +position = BandValues.position(source,window_rows,channel,near,column)
      +pixel = BandValues.pixel(source,channel,stored,column)
      +local_exact = Sweeps.address_exact(((local / width : U32) / output_rows : U32),window_rows,width,(local % width : U32),local_coordinate,channel,column,count,limits,local_block,local_offset,
        channel_inside,column_inside,local_valid,local_room)
      +global_exact = Equal.trans(Nat,U32.to_nat(global_word),BandValues.point(source,image,channel,far,column),pixel,
        Sweeps.address_exact(((global / width : U32) / out_height : U32),height,width,(global % width : U32),global_coordinate,Nat.add(Nat.mul(image,count),channel),column,Nat.mul(U32.to_nat(batch),count),limits,global_block,global_offset,
          RowMajor.radix(image,U32.to_nat(batch),channel,count,image_inside,channel_inside),column_inside,global_valid,global_room),
        Equal.sym(Nat,pixel,BandValues.point(source,image,channel,far,column),BandValues.located_pixel(source,image,channel,far,column,stored,global_valid,located)))
      %Equal.sym(U32,local_word,U32.from_nat(position),Encoded.encoding(local_word,position,local_exact)) : {Tree.read(Element,input,_) == inner_value(inner,global_word) : Element}
      %Equal.sym(U32,global_word,U32.from_nat(pixel),Encoded.encoding(global_word,pixel,global_exact)) : {Tree.read(Element,input,U32.from_nat(position)) == inner_value(inner,_) : Element}
      BandValues.leaf_read(~Element,~Inner,~inner_value,inner,source,window_first,window_rows,input,length,channel,near,column,values,channel_inside,local_valid,column_inside)

# One tap of the rows sweep, for any two positions with the digits of the
# cell.
def row_moved(~Element: Data,~Inner: Data,~inner_value: Inner -> U32 -> Element,+inner: Inner,+pad: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,+stride: U32,+padding: U32,+out_height: U32,
  +window_first: U32,+window_rows: U32,+output_rows: U32,+origin: U32,+input: Tree.Tree<Element>,+length: Nat,+limits: Machine.Limits,+channel: Nat,+row: Nat,+column: Nat,+image: Nat,+inner_row: Nat,+tap: U32,+local: U32,+global: U32,
  local_digits: Sweeps.Decoded<local,width,output_rows,channel,row,column>,
  global_digits: Sweeps.Decoded<global,width,out_height,Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),inner_row,column>,
  +same: {U32.is_lt((U32.from_nat(row) * stride + origin + tap - padding : U32),window_rows) == U32.is_lt((U32.from_nat(inner_row) * stride + 0 + tap - padding : U32),height) : Bool},
  located: Boolean.When(U32.is_lt((U32.from_nat(inner_row) * stride + 0 + tap - padding : U32),height),
    {Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat((U32.from_nat(inner_row) * stride + 0 + tap - padding : U32))) == Nat.add(U32.to_nat(window_first),U32.to_nat((U32.from_nat(row) * stride + origin + tap - padding : U32))) : Nat}),
  +values: BandValues.Leaf(~Element,~Inner,~inner_value,inner,Bands.Geometry{batch,channels,height,width},window_first,window_rows,input,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,U32.to_nat(batch)) == True{} : Bool},
  +local_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +global_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(batch),U32.to_nat(channels)),Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {SweepLaws.sample_line(~Element,input,Access.Line{((local / width : U32) / output_rows * window_rows * width + local % width : U32),((local / width : U32) % output_rows * stride + origin : U32),window_rows,width,padding,pad},tap) ==
    Sweeps.logical_line(~Element,~Inner,~inner_value,inner,Access.Line{((global / width : U32) / out_height * height * width + global % width : U32),((global / width : U32) % out_height * stride + 0 : U32),height,width,padding,pad},tap) : Element}:
  Sweeps.Decoded{+local_block,+local_along,+local_offset} = local_digits
  Sweeps.Decoded{+global_block,+global_along,+global_offset} = global_digits
  +local_base = ((local / width : U32) / output_rows * window_rows * width + local % width : U32)
  +global_base = ((global / width : U32) / out_height * height * width + global % width : U32)
  +local_first = (U32.from_nat(row) * stride + origin : U32)
  +global_first = (U32.from_nat(inner_row) * stride + 0 : U32)
  %Equal.sym(U32,((local / width : U32) % output_rows : U32),U32.from_nat(row),Encoded.encoding(((local / width : U32) % output_rows : U32),row,local_along)) :
    {SweepLaws.sample_line(~Element,input,Access.Line{local_base,(_ * stride + origin : U32),window_rows,width,padding,pad},tap) ==
      Sweeps.logical_line(~Element,~Inner,~inner_value,inner,Access.Line{global_base,((global / width : U32) % out_height * stride + 0 : U32),height,width,padding,pad},tap) : Element}
  %Equal.sym(U32,((global / width : U32) % out_height : U32),U32.from_nat(inner_row),Encoded.encoding(((global / width : U32) % out_height : U32),inner_row,global_along)) :
    {SweepLaws.sample_line(~Element,input,Access.Line{local_base,local_first,window_rows,width,padding,pad},tap) ==
      Sweeps.logical_line(~Element,~Inner,~inner_value,inner,Access.Line{global_base,(_ * stride + 0 : U32),height,width,padding,pad},tap) : Element}
  moved_tap(~Element,~Inner,~inner_value,inner,input,pad,width,padding,tap,local_base,local_first,window_rows,global_base,global_first,height,same,
    row_read(~Element,~Inner,~inner_value,inner,batch,channels,height,width,window_first,window_rows,out_height,output_rows,input,length,limits,channel,column,image,
      (local_first + tap - padding : U32),(global_first + tap - padding : U32),U32.is_lt((global_first + tap - padding : U32),height),local,global,
      {==},same,located,values,channel_inside,column_inside,image_inside,local_block,local_offset,global_block,global_offset,local_room,global_room))

# One tap of an output row: the stored row it reads is valid exactly when the
# row of the image is, and then it is that row of the image.
def RowTap(+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+height: U32,+image: Nat,+local: U32,+global: U32,+tap: U32) -> Data:
  Shape.Both<{U32.is_lt((local * stride + origin + tap - padding : U32),window_rows) == U32.is_lt((global * stride + 0 + tap - padding : U32),height) : Bool},
    Boolean.When(U32.is_lt((global * stride + 0 + tap - padding : U32),height),
      {Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat((global * stride + 0 + tap - padding : U32))) == Nat.add(U32.to_nat(window_first),U32.to_nat((local * stride + origin + tap - padding : U32))) : Nat})>

def RowTaps(+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+height: U32,+image: Nat,+local: U32,+global: U32,remaining: Nat,+tap: U32) -> Data:
  match remaining:
    case 0n: Unit
    case 1n+rest: Shape.Both<RowTap(stride,padding,origin,window_first,window_rows,height,image,local,global,tap),RowTaps(stride,padding,origin,window_first,window_rows,height,image,local,global,rest,(tap + 1 : U32))>

# The taps of one output row of a leaf: the row inside the leaf and the row
# inside its image.
def RowTapsAt(+kernel: U32,+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+height: U32,+out_height: U32,+output_first: U32,+row: Nat) -> Data:
  RowTaps(stride,padding,origin,window_first,window_rows,height,Nat.div(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height)),U32.from_nat(row),
    U32.from_nat(Nat.mod(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height))),U32.to_nat(kernel),0)

def RowFacts(+kernel: U32,+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+height: U32,+out_height: U32,+output_first: U32,+output_rows: U32) -> Data:
  Members.All(Nat,{row => RowTapsAt(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,row) : Nat -> Data},Ranges.range(U32.to_nat(output_rows),0n))

def row_tap(~Element: Data,~Inner: Data,~inner_value: Inner -> U32 -> Element,+inner: Inner,+pad: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_height: U32,
  +window_first: U32,+window_rows: U32,+output_first: U32,+output_rows: U32,+origin: U32,+input: Tree.Tree<Element>,+length: Nat,+limits: Machine.Limits,+channel: Nat,+row: Nat,+column: Nat,+tap: U32,
  +values: BandValues.Leaf(~Element,~Inner,~inner_value,inner,Bands.Geometry{batch,channels,height,width},window_first,window_rows,input,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(output_rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool},
  +output_fit: {Nat.is_le(Nat.add(U32.to_nat(output_first),U32.to_nat(output_rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(out_height))) == True{} : Bool},
  +out_height_positive: {Nat.is_lt(0n,U32.to_nat(out_height)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(out_height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_input_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_output_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(output_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  fact: RowTap(stride,padding,origin,window_first,window_rows,height,Nat.div(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height)),U32.from_nat(row),
    U32.from_nat(Nat.mod(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height))),tap)) ->
  {SweepLaws.sample_line(~Element,input,Access.line_at(~Element,Access.Sweep{window_rows,width,kernel,stride,padding,output_rows,origin},pad,U32.from_nat(BandValues.position(Bands.Geometry{batch,channels,out_height,width},output_rows,channel,row,column))),tap) ==
    Sweeps.logical_line(~Element,~Inner,~inner_value,inner,Access.line_at(~Element,Access.Sweep{height,width,kernel,stride,padding,out_height,0},pad,U32.from_nat(BandValues.pixel(Bands.Geometry{batch,channels,out_height,width},channel,Nat.add(U32.to_nat(output_first),row),column))),tap) : Element}:
  Shape.Both{+same,located} = fact
  +pooled = {Bands.Geometry{batch,channels,out_height,width} : Bands.Geometry}
  +images = U32.to_nat(batch)
  +count = U32.to_nat(channels)
  +written = U32.to_nat(output_rows)
  +tall = U32.to_nat(out_height)
  +wide = U32.to_nat(width)
  +limit = U32.to_nat(Machine.capacity(limits))
  +whole = Nat.add(U32.to_nat(output_first),row)
  +image = Nat.div(whole,tall)
  +inside = Nat.mod(whole,tall)
  +plane = Nat.add(Nat.mul(image,count),channel)
  +position = BandValues.position(pooled,output_rows,channel,row,column)
  +pixel = BandValues.pixel(pooled,channel,whole,column)
  +local = U32.from_nat(position)
  +global = U32.from_nat(pixel)
  +image_ok = image_below(U32.to_nat(output_first),written,row,images,tall,out_height_positive,row_inside,output_fit)
  +inside_ok = DivisionOrder.remainder_small(whole,tall,out_height_positive)
  +plane_ok = RowMajor.radix(image,images,channel,count,image_ok,channel_inside)
  +count_positive = positive(channel,count,channel_inside)
  +wide_positive = positive(column,wide,column_inside)
  +step_room = inner_room(count,written,wide,limit,count_positive,positive(row,written,row_inside),local_output_room)
  +grouped_output = grouped(images,count,Nat.mul(tall,wide),limit,output_room)
  +local_digits = digits(position,width,output_rows,channel,row,column,limits,RowMajor.mixed(channel,written,wide,row,column),
    Bounds.strict_le(position,limit,Bounds.nat_trans(position,Nat.mul(count,Nat.mul(written,wide)),limit,RowMajor.in_bounds(channel,count,written,wide,row,column,channel_inside,row_inside,column_inside),local_output_room)),
    row_inside,column_inside,step_room,middle_room(count,written,wide,limit,count_positive,wide_positive,local_output_room))
  +global_digits = digits(pixel,width,out_height,plane,inside,column,limits,RowMajor.mixed(plane,tall,wide,inside,column),
    Bounds.strict_le(pixel,limit,Bounds.nat_trans(pixel,Nat.mul(Nat.mul(images,count),Nat.mul(tall,wide)),limit,RowMajor.in_bounds(plane,Nat.mul(images,count),tall,wide,inside,column,plane_ok,inside_ok,column_inside),grouped_output)),
    inside_ok,column_inside,step_room,middle_room(Nat.mul(images,count),tall,wide,limit,positive(plane,Nat.mul(images,count),plane_ok),wide_positive,grouped_output))
  row_moved(~Element,~Inner,~inner_value,inner,pad,batch,channels,height,width,stride,padding,out_height,window_first,window_rows,output_rows,origin,input,length,limits,channel,row,column,image,inside,tap,local,global,
    local_digits,global_digits,same,located,values,channel_inside,column_inside,image_ok,local_input_room,grouped(images,count,Nat.mul(U32.to_nat(height),wide),limit,input_room))

def row_taps(~Element: Data,~Inner: Data,~inner_value: Inner -> U32 -> Element,remaining: Nat,+tap: U32,+inner: Inner,+pad: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_height: U32,
  +window_first: U32,+window_rows: U32,+output_first: U32,+output_rows: U32,+origin: U32,+input: Tree.Tree<Element>,+length: Nat,+limits: Machine.Limits,+channel: Nat,+row: Nat,+column: Nat,
  +values: BandValues.Leaf(~Element,~Inner,~inner_value,inner,Bands.Geometry{batch,channels,height,width},window_first,window_rows,input,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(output_rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool},
  +output_fit: {Nat.is_le(Nat.add(U32.to_nat(output_first),U32.to_nat(output_rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(out_height))) == True{} : Bool},
  +out_height_positive: {Nat.is_lt(0n,U32.to_nat(out_height)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(out_height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_input_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_output_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(output_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  facts: RowTaps(stride,padding,origin,window_first,window_rows,height,Nat.div(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height)),U32.from_nat(row),
    U32.from_nat(Nat.mod(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height))),remaining,tap)) ->
  Sweeps.TapFacts(~Element,~Inner,~inner_value,inner,input,
    Access.line_at(~Element,Access.Sweep{window_rows,width,kernel,stride,padding,output_rows,origin},pad,U32.from_nat(BandValues.position(Bands.Geometry{batch,channels,out_height,width},output_rows,channel,row,column))),
    Access.line_at(~Element,Access.Sweep{height,width,kernel,stride,padding,out_height,0},pad,U32.from_nat(BandValues.pixel(Bands.Geometry{batch,channels,out_height,width},channel,Nat.add(U32.to_nat(output_first),row),column))),remaining,tap):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{head,tail} = facts
      Shape.Both{row_tap(~Element,~Inner,~inner_value,inner,pad,batch,channels,height,width,kernel,stride,padding,out_height,window_first,window_rows,output_first,output_rows,origin,input,length,limits,channel,row,column,tap,
          values,channel_inside,row_inside,column_inside,output_fit,out_height_positive,input_room,output_room,local_input_room,local_output_room,head),
        row_taps(~Element,~Inner,~inner_value,rest,(tap + 1 : U32),inner,pad,batch,channels,height,width,kernel,stride,padding,out_height,window_first,window_rows,output_first,output_rows,origin,input,length,limits,channel,row,column,
          values,channel_inside,row_inside,column_inside,output_fit,out_height_positive,input_room,output_room,local_input_room,local_output_room,tail)}

# What the rows sweep of a leaf supplies to every cell of its output.
type RowCells<-Element: Data,-Inner: Data,-inner_value: Inner -> U32 -> Element> is Data:
  RowCells{inner: Inner,pad: Element,batch: U32,channels: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,out_height: U32,window_first: U32,window_rows: U32,output_first: U32,output_rows: U32,origin: U32,
    input: Tree.Tree<Element>,length: Nat,output: Buffer.Model<Element>,limits: Machine.Limits,ready: Buffer.Ready<Element,output>,
    values: BandValues.Leaf(~Element,~Inner,~inner_value,inner,Bands.Geometry{batch,channels,height,width},window_first,window_rows,input,length),
    length_exact: {Buffer.length(~Element,output) == BandValues.cell_count(Bands.Geometry{batch,channels,out_height,width},output_rows) : Nat},
    output_fit: {Nat.is_le(Nat.add(U32.to_nat(output_first),U32.to_nat(output_rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(out_height))) == True{} : Bool},
    out_height_positive: {Nat.is_lt(0n,U32.to_nat(out_height)) == True{} : Bool},
    input_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(out_height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    local_input_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    local_output_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(output_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    facts: RowFacts(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,output_rows)}

def row_geometry(~Element: Data,~Inner: Data,~inner_value: Inner -> U32 -> Element,evidence: RowCells<Element,Inner,inner_value>) -> Bands.Geometry:
  RowCells{inner,pad,batch,channels,height,width,kernel,stride,padding,out_height,window_first,window_rows,output_first,output_rows,origin,input,length,output,limits,ready,values,length_exact,output_fit,out_height_positive,input_room,output_room,local_input_room,local_output_room,facts} = evidence
  Bands.Geometry{batch,channels,out_height,width}

def row_rows(~Element: Data,~Inner: Data,~inner_value: Inner -> U32 -> Element,evidence: RowCells<Element,Inner,inner_value>) -> U32:
  RowCells{inner,pad,batch,channels,height,width,kernel,stride,padding,out_height,window_first,window_rows,output_first,output_rows,origin,input,length,output,limits,ready,values,length_exact,output_fit,out_height_positive,input_room,output_room,local_input_room,local_output_room,facts} = evidence
  output_rows

def RowCell(~Element: Data,~Inner: Data,~inner_value: Inner -> U32 -> Element,~operation: Element -> Element -> Element,+evidence: RowCells<Element,Inner,inner_value>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  RowCells{+inner,+pad,+batch,+channels,+height,+width,+kernel,+stride,+padding,+out_height,+window_first,+window_rows,+output_first,+output_rows,+origin,+input,+length,+output,+limits,+ready,+values,+length_exact,+output_fit,+out_height_positive,+input_room,+output_room,+local_input_room,+local_output_room,+facts} = evidence
  BandValues.At(~Element,~Sweeps.Swept<Element,Inner>,~Sweeps.value(~Element,~Inner,~inner_value,~operation),Sweeps.Swept{inner,Access.Sweep{height,width,kernel,stride,padding,out_height,0},pad},
    Bands.Geometry{batch,channels,out_height,width},output_first,output_rows,
    Sweeps.run(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{Access.Sweep{window_rows,width,kernel,stride,padding,output_rows,origin},pad},input,Buffer.tree(~Element,output)),Buffer.length(~Element,output),channel,row,column)

def row_cell(~Element: Data,~Inner: Data,~inner_value: Inner -> U32 -> Element,~operation: Element -> Element -> Element,+evidence: RowCells<Element,Inner,inner_value>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(row_geometry(~Element,~Inner,~inner_value,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(row_rows(~Element,~Inner,~inner_value,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(row_geometry(~Element,~Inner,~inner_value,evidence))) == True{} : Bool}) ->
  RowCell(~Element,~Inner,~inner_value,~operation,evidence,channel,row,column):
  RowCells{+inner,+pad,+batch,+channels,+height,+width,+kernel,+stride,+padding,+out_height,+window_first,+window_rows,+output_first,+output_rows,+origin,+input,+length,+output,+limits,+ready,+values,+length_exact,+output_fit,+out_height_positive,+input_room,+output_room,+local_input_room,+local_output_room,+facts} = evidence
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  +pooled = {Bands.Geometry{batch,channels,out_height,width} : Bands.Geometry}
  +stored = {Access.Sweep{window_rows,width,kernel,stride,padding,output_rows,origin} : Access.Sweep}
  +whole = {Access.Sweep{height,width,kernel,stride,padding,out_height,0} : Access.Sweep}
  swept_at(~Element,~Sweeps.Swept<Element,Inner>,~Sweeps.value(~Element,~Inner,~inner_value,~operation),~operation,Sweeps.Swept{inner,whole,pad},pooled,output_first,output_rows,Access.SweepContext{stored,pad},input,output,channel,row,column,ready,
    BandValues.position_within(pooled,output_rows,Buffer.length(~Element,output),channel,row,column,length_exact,channel_inside,row_inside,column_inside),
    Sweeps.taps_match(~Element,~Inner,~inner_value,~operation,U32.to_nat(kernel),0,inner,input,
      Access.line_at(~Element,stored,pad,U32.from_nat(BandValues.position(pooled,output_rows,channel,row,column))),
      Access.line_at(~Element,whole,pad,U32.from_nat(BandValues.pixel(pooled,channel,Nat.add(U32.to_nat(output_first),row),column))),pad,
      row_taps(~Element,~Inner,~inner_value,U32.to_nat(kernel),0,inner,pad,batch,channels,height,width,kernel,stride,padding,out_height,window_first,window_rows,output_first,output_rows,origin,input,length,limits,channel,row,column,
        values,channel_inside,row_inside,column_inside,output_fit,out_height_positive,input_room,output_room,local_input_room,local_output_room,
        Members.all_member(Nat,{row => RowTapsAt(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,row) : Nat -> Data},row,Ranges.range(U32.to_nat(output_rows),0n),facts,
          Members.range_member(U32.to_nat(output_rows),0n,row,row_inside)))))

# The rows sweep of a leaf holding a window of rows of an image batch is the
# leaf, with the output rows, of the batch swept along its rows.
def rows_leaf(~Element: Data,~Inner: Data,~inner_value: Inner -> U32 -> Element,~operation: Element -> Element -> Element,+inner: Inner,+pad: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+window_first: U32,+window_rows: U32,+output_first: U32,+output_rows: U32,+origin: U32,
  +input: Tree.Tree<Element>,+length: Nat,+output: Buffer.Model<Element>,+limits: Machine.Limits,+ready: Buffer.Ready<Element,output>,
  +values: BandValues.Leaf(~Element,~Inner,~inner_value,inner,Bands.Geometry{batch,channels,height,width},window_first,window_rows,input,length),
  +length_exact: {Buffer.length(~Element,output) == BandValues.cell_count(Bands.Geometry{batch,channels,out_height,width},output_rows) : Nat},
  +output_fit: {Nat.is_le(Nat.add(U32.to_nat(output_first),U32.to_nat(output_rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(out_height))) == True{} : Bool},
  +out_height_positive: {Nat.is_lt(0n,U32.to_nat(out_height)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(out_height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_input_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local_output_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(output_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +facts: RowFacts(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,output_rows)) ->
  BandValues.Leaf(~Element,~Sweeps.Swept<Element,Inner>,~Sweeps.value(~Element,~Inner,~inner_value,~operation),Sweeps.Swept{inner,Access.Sweep{height,width,kernel,stride,padding,out_height,0},pad},
    Bands.Geometry{batch,channels,out_height,width},output_first,output_rows,
    Sweeps.run(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{Access.Sweep{window_rows,width,kernel,stride,padding,output_rows,origin},pad},input,Buffer.tree(~Element,output)),Buffer.length(~Element,output)):
  BandValues.cells(~RowCells<Element,Inner,inner_value>,~RowCell(~Element,~Inner,~inner_value,~operation),~row_geometry(~Element,~Inner,~inner_value),~row_rows(~Element,~Inner,~inner_value),~row_cell(~Element,~Inner,~inner_value,~operation),
    RowCells{inner,pad,batch,channels,height,width,kernel,stride,padding,out_height,window_first,window_rows,output_first,output_rows,origin,input,length,output,limits,ready,values,length_exact,output_fit,out_height_positive,input_room,output_room,local_input_room,local_output_room,facts})

# The bounds of a pooled window: its rows and its output rows fit in the
# batch, and the input, the column-swept planes and the output fit the
# machine capacity, for the whole batch and for the rows the leaf stores.
type Rooms<-batch: U32,-channels: U32,-height: U32,-width: U32,-out_height: U32,-out_width: U32,-window_first: U32,-window_rows: U32,-output_first: U32,-output_rows: U32,-limits: Machine.Limits> is Data:
  Rooms{window_fit: {Nat.is_le(Nat.add(U32.to_nat(window_first),U32.to_nat(window_rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) == True{} : Bool},
    output_fit: {Nat.is_le(Nat.add(U32.to_nat(output_first),U32.to_nat(output_rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(out_height))) == True{} : Bool},
    height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
    out_height_positive: {Nat.is_lt(0n,U32.to_nat(out_height)) == True{} : Bool},
    input_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    planes_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(out_width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    local_input_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    local_planes_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    local_output_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(output_rows),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

# The two sweeps of a pooled window: the columns of the stored rows into
# planes of the same rows, then the rows of the planes into the output rows.
# The output is the leaf of the batch pooled as a whole.
def pooled_leaf(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+out_width: U32,+window_first: U32,+window_rows: U32,+output_first: U32,+output_rows: U32,+origin: U32,
  +input: Tree.Tree<Element>,+length: Nat,+middle: Buffer.Model<Element>,+output: Buffer.Model<Element>,+limits: Machine.Limits,
  rooms: Rooms<batch,channels,height,width,out_height,out_width,window_first,window_rows,output_first,output_rows,limits>,
  +middle_ready: Buffer.Ready<Element,middle>,+ready: Buffer.Ready<Element,output>,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},window_first,window_rows,input,length),
  +middle_exact: {Buffer.length(~Element,middle) == BandValues.cell_count(Bands.Geometry{batch,channels,height,out_width},window_rows) : Nat},
  +length_exact: {Buffer.length(~Element,output) == BandValues.cell_count(Bands.Geometry{batch,channels,out_height,out_width},output_rows) : Nat},
  +facts: RowFacts(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,output_rows)) ->
  BandValues.Leaf(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),
    Sweeps.Swept{Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},Access.Sweep{height,out_width,kernel,stride,padding,out_height,0},pad},
    Bands.Geometry{batch,channels,out_height,out_width},output_first,output_rows,
    Sweeps.run(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{Access.Sweep{window_rows,out_width,kernel,stride,padding,output_rows,origin},pad},
      Sweeps.run(~Element,~operation,Buffer.length(~Element,middle),Access.SweepContext{Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},input,Buffer.tree(~Element,middle)),Buffer.tree(~Element,output)),Buffer.length(~Element,output)):
  Rooms{+window_fit,+output_fit,+height_positive,+out_height_positive,+input_room,+planes_room,+output_room,+local_input_room,+local_planes_room,+local_output_room} = rooms
  rows_leaf(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation,Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},pad,
    batch,channels,height,out_width,kernel,stride,padding,out_height,window_first,window_rows,output_first,output_rows,origin,
    Sweeps.run(~Element,~operation,Buffer.length(~Element,middle),Access.SweepContext{Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},input,Buffer.tree(~Element,middle)),Buffer.length(~Element,middle),output,limits,ready,
    columns_leaf(~Element,~Logical,~original,~operation,logical,pad,batch,channels,height,width,kernel,stride,padding,out_width,window_first,window_rows,input,length,middle,limits,middle_ready,
      values,middle_exact,window_fit,height_positive,input_room,planes_room,local_input_room,local_planes_room),
    length_exact,output_fit,out_height_positive,planes_room,output_room,local_planes_room,local_output_room,facts)
