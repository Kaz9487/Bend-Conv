# The content relation shared by every band operation, stated on coordinates,
# and its transport between contexts with equal values.
import Base
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./tensor_band_model.bend as Model
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./nat_division.bend as Division
import ./word_encoding.bend as WordEncoding
import ./tensor_band_shape_refinement.bend as BandShape
import ./natural_addition.bend as Addition
import ../tensor_band.bend as Bands
import ./tensor_band_coordinates.bend as BandCoordinates

# The global NCHW index of a pixel: image, channel, row inside the image and
# column.
def point(geometry: Bands.Geometry,image: Nat,channel: Nat,row: Nat,column: Nat) -> Nat:
  Bands.Geometry{images,channels,height,width} = geometry
  RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),U32.to_nat(height),U32.to_nat(width),row,column)

# A row counted over the whole batch.
def global_row(geometry: Bands.Geometry,image: Nat,row: Nat) -> Nat:
  Bands.Geometry{images,channels,height,width} = geometry
  Nat.add(Nat.mul(image,U32.to_nat(height)),row)

# The same index from the channel, a row counted over the whole batch and the
# column. A band names its rows this way, so moving rows between bands only
# adds and subtracts row offsets.
def pixel(geometry: Bands.Geometry,channel: Nat,+row: Nat,column: Nat) -> Nat:
  Bands.Geometry{images,channels,+height,width} = geometry
  point(Bands.Geometry{images,channels,height,width},Nat.div(row,U32.to_nat(height)),channel,Nat.mod(row,U32.to_nat(height)),column)

def height(geometry: Bands.Geometry) -> Nat:
  Bands.Geometry{images,channels,height,width} = geometry
  U32.to_nat(height)

# A pixel whose row over the batch is a row inside one image is the point of
# that image and that row.
def located_pixel(+geometry: Bands.Geometry,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,+global: Nat,
  +row_inside: {Nat.is_lt(row,height(geometry)) == True{} : Bool},located: {global_row(geometry,image,row) == global : Nat}) ->
  {pixel(geometry,channel,global,column) == point(geometry,image,channel,row,column) : Nat}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  +base = U32.to_nat(height)
  +whole = Nat.add(Nat.mul(image,base),row)
  +positive = Order.le_lt_trans(0n,row,base,Order.zero_le(row),row_inside)
  %located : {pixel(Bands.Geometry{images,channels,height,width},channel,_,column) == point(Bands.Geometry{images,channels,height,width},image,channel,row,column) : Nat}
  %Equal.sym(Nat,Nat.div(whole,base),image,BandCoordinates.quotient(image,row,base,positive,row_inside)) :
    {point(Bands.Geometry{images,channels,height,width},_,channel,Nat.mod(whole,base),column) == point(Bands.Geometry{images,channels,height,width},image,channel,row,column) : Nat}
  %Equal.sym(Nat,Nat.mod(whole,base),row,BandCoordinates.remainder(image,row,base,positive,row_inside)) :
    {point(Bands.Geometry{images,channels,height,width},image,channel,_,column) == point(Bands.Geometry{images,channels,height,width},image,channel,row,column) : Nat}
  {==}

def channels(geometry: Bands.Geometry) -> Nat:
  Bands.Geometry{images,channels,height,width} = geometry
  U32.to_nat(channels)

def width(geometry: Bands.Geometry) -> Nat:
  Bands.Geometry{images,channels,height,width} = geometry
  U32.to_nat(width)

# Where a leaf of rows rows stores channel, row and column.
def position(geometry: Bands.Geometry,rows: U32,channel: Nat,row: Nat,column: Nat) -> Nat:
  RowMajor.index(channel,U32.to_nat(rows),width(geometry),row,column)

# The number of cells of a leaf of rows rows.
def cell_count(+geometry: Bands.Geometry,rows: U32) -> Nat:
  Nat.mul(channels(geometry),Nat.mul(U32.to_nat(rows),width(geometry)))

# Every position of a cell is below the cell count.
def position_inside(+geometry: Bands.Geometry,+rows: U32,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,channels(geometry)) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,width(geometry)) == True{} : Bool}) ->
  {Nat.is_lt(position(geometry,rows,channel,row,column),cell_count(geometry,rows)) == True{} : Bool}:
  RowMajor.in_bounds(channel,channels(geometry),U32.to_nat(rows),width(geometry),row,column,channel_inside,row_inside,column_inside)

# The same bound against a content length that equals the cell count.
def position_within(+geometry: Bands.Geometry,+rows: U32,+length: Nat,+channel: Nat,+row: Nat,+column: Nat,
  length_exact: {length == cell_count(geometry,rows) : Nat},
  channel_inside: {Nat.is_lt(channel,channels(geometry)) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,width(geometry)) == True{} : Bool}) ->
  {Nat.is_lt(position(geometry,rows,channel,row,column),length) == True{} : Bool}:
  %Equal.sym(Nat,length,cell_count(geometry,rows),length_exact) : {Nat.is_lt(position(geometry,rows,channel,row,column),_) == True{} : Bool}
  position_inside(geometry,rows,channel,row,column,channel_inside,row_inside,column_inside)

# Leaf metadata states that the content length is the cell count.
def leaf_length(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,metadata: BandShape.Leaf(geometry,first,rows,length)) ->
  {length == cell_count(geometry,rows) : Nat}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  length_exact

# A leaf whose content is empty has no cells.
def empty_count(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,metadata: BandShape.Leaf(geometry,first,rows,length),
  empty: {length == 0n : Nat}) -> {cell_count(geometry,rows) == 0n : Nat}:
  Equal.trans(Nat,cell_count(geometry,rows),length,0n,Equal.sym(Nat,length,cell_count(geometry,rows),leaf_length(geometry,first,rows,length,metadata)),empty)

# One cell of a leaf lies inside its content and holds the logical value of
# its pixel. Layout coverage, allocation bounds and the construction of this
# relation from each producer are separate obligations.
def At(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,context: Context,+geometry: Bands.Geometry,first: U32,+rows: U32,input: Tree.Tree<Element>,length: Nat,
  +channel: Nat,+row: Nat,+column: Nat) -> Data:
  Shape.Both<{Nat.is_lt(position(geometry,rows,channel,row,column),length) == True{} : Bool},
    {Tree.read(Element,input,U32.from_nat(position(geometry,rows,channel,row,column))) ==
      value(context,U32.from_nat(pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))) : Element}>

# The cells of one channel of a leaf.
def ChannelCells(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+channel: Nat) -> Data:
  Members.All(Nat,{row =>
    Members.All(Nat,{column => At(~Element,~Context,~value,context,geometry,first,rows,input,length,channel,row,column) : Nat -> Data},Ranges.range(width(geometry),0n)) : Nat -> Data},
    Ranges.range(U32.to_nat(rows),0n))

# The first count channels of a leaf. Operations that fill a leaf one channel
# group at a time state their progress with it.
def Channels(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,count: Nat) -> Data:
  Members.All(Nat,{channel => ChannelCells(~Element,~Context,~value,context,geometry,first,rows,input,length,channel) : Nat -> Data},Ranges.range(count,0n))

def Leaf(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat) -> Data:
  Channels(~Element,~Context,~value,context,geometry,first,rows,input,length,channels(geometry))

def Values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,bands: Model.Content<Element>) -> Data:
  match bands:
    case Model.RowBand{Buffer.Content{input,length},first,rows}: Leaf(~Element,~Context,~value,context,geometry,first,rows,input,length)
    case Model.Joined{left,right}: Shape.Both<Values(~Element,~Context,~value,context,geometry,left),Values(~Element,~Context,~value,context,geometry,right)>

def channels_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+count: Nat,
  +channel: Nat,+row: Nat,+column: Nat,values: Channels(~Element,~Context,~value,context,geometry,first,rows,input,length,count),
  channel_inside: {Nat.is_lt(channel,count) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,width(geometry)) == True{} : Bool}) ->
  At(~Element,~Context,~value,context,geometry,first,rows,input,length,channel,row,column):
  Members.all_member(Nat,{column => At(~Element,~Context,~value,context,geometry,first,rows,input,length,channel,row,column) : Nat -> Data},column,Ranges.range(width(geometry),0n),
    Members.all_member(Nat,{row =>
      Members.All(Nat,{column => At(~Element,~Context,~value,context,geometry,first,rows,input,length,channel,row,column) : Nat -> Data},Ranges.range(width(geometry),0n)) : Nat -> Data},
      row,Ranges.range(U32.to_nat(rows),0n),
      Members.all_member(Nat,{channel => ChannelCells(~Element,~Context,~value,context,geometry,first,rows,input,length,channel) : Nat -> Data},channel,Ranges.range(count,0n),values,Members.range_member(count,0n,channel,channel_inside)),
      Members.range_member(U32.to_nat(rows),0n,row,row_inside)),
    Members.range_member(width(geometry),0n,column,column_inside))

def leaf_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,
  +channel: Nat,+row: Nat,+column: Nat,values: Leaf(~Element,~Context,~value,context,geometry,first,rows,input,length),
  channel_inside: {Nat.is_lt(channel,channels(geometry)) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,width(geometry)) == True{} : Bool}) ->
  At(~Element,~Context,~value,context,geometry,first,rows,input,length,channel,row,column):
  channels_cell(~Element,~Context,~value,context,geometry,first,rows,input,length,channels(geometry),channel,row,column,values,channel_inside,row_inside,column_inside)

# A property of start + index for every index of a range is the property of
# every member of the range moved by start.
def all_shifted(-Property: Nat -> Data,+prefix: Nat,count: Nat,+start: Nat,
  all: Members.All(Nat,{index => Property(Nat.add(prefix,index)) : Nat -> Data},Ranges.range(count,start))) ->
  Members.All(Nat,Property,Ranges.range(count,Nat.add(prefix,start))):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{head,tail} = all
      Shape.Both{head,
        %Addition.successor_right(prefix,start) : Members.All(Nat,Property,Ranges.range(rest,_))
        all_shifted(Property,prefix,rest,1n+start,tail)}

# Channels filled so far and the next group of channels make a longer prefix.
def channels_joined(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+prefix: Nat,+count: Nat,
  previous: Channels(~Element,~Context,~value,context,geometry,first,rows,input,length,prefix),
  added: Members.All(Nat,{channel => ChannelCells(~Element,~Context,~value,context,geometry,first,rows,input,length,channel) : Nat -> Data},Ranges.range(count,prefix))) ->
  Channels(~Element,~Context,~value,context,geometry,first,rows,input,length,Nat.add(prefix,count)):
  %Ranges.join(0n,prefix,count,prefix,Nat.add(prefix,count),{==},{==}) : Members.All(Nat,{channel => ChannelCells(~Element,~Context,~value,context,geometry,first,rows,input,length,channel) : Nat -> Data},_)
  Members.all_concatenated(Nat,{channel => ChannelCells(~Element,~Context,~value,context,geometry,first,rows,input,length,channel) : Nat -> Data},Ranges.range(prefix,0n),Ranges.range(count,prefix),previous,added)

# The cell holds the logical value of its pixel.
def leaf_read(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,
  +channel: Nat,+row: Nat,+column: Nat,values: Leaf(~Element,~Context,~value,context,geometry,first,rows,input,length),
  channel_inside: {Nat.is_lt(channel,channels(geometry)) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,width(geometry)) == True{} : Bool}) ->
  {Tree.read(Element,input,U32.from_nat(position(geometry,rows,channel,row,column))) ==
    value(context,U32.from_nat(pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))) : Element}:
  Shape.second({Nat.is_lt(position(geometry,rows,channel,row,column),length) == True{} : Bool},
    {Tree.read(Element,input,U32.from_nat(position(geometry,rows,channel,row,column))) ==
      value(context,U32.from_nat(pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))) : Element},
    leaf_cell(~Element,~Context,~value,context,geometry,first,rows,input,length,channel,row,column,values,channel_inside,row_inside,column_inside))

# The same cell named by its channel and its offset inside the channel plane,
# as the plane copies address it. The offset is split by the width here, once.
def leaf_plane_read(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,
  +channel: Nat,+index: Nat,values: Leaf(~Element,~Context,~value,context,geometry,first,rows,input,length),
  channel_inside: {Nat.is_lt(channel,channels(geometry)) == True{} : Bool},+index_inside: {Nat.is_lt(index,Nat.mul(U32.to_nat(rows),width(geometry))) == True{} : Bool},
  +width_positive: {Nat.is_lt(0n,width(geometry)) == True{} : Bool}) ->
  {Tree.read(Element,input,U32.from_nat(Nat.add(Nat.mul(channel,Nat.mul(U32.to_nat(rows),width(geometry))),index))) ==
    value(context,U32.from_nat(pixel(geometry,channel,Nat.add(U32.to_nat(first),Nat.div(index,width(geometry))),Nat.mod(index,width(geometry))))) : Element}:
  +local = Nat.div(index,width(geometry))
  +column = Nat.mod(index,width(geometry))
  +exact = DivisionOrder.quotient_exact(index,width(geometry),width_positive)
  %Equal.sym(Nat,index,Nat.add(Nat.mul(local,width(geometry)),column),exact) :
    {Tree.read(Element,input,U32.from_nat(Nat.add(Nat.mul(channel,Nat.mul(U32.to_nat(rows),width(geometry))),_))) == value(context,U32.from_nat(pixel(geometry,channel,Nat.add(U32.to_nat(first),local),column))) : Element}
  leaf_read(~Element,~Context,~value,context,geometry,first,rows,input,length,channel,local,column,values,channel_inside,
    RowMajor.row_inside(local,U32.to_nat(rows),width(geometry),column,%exact : {Nat.is_lt(_,Nat.mul(U32.to_nat(rows),width(geometry))) == True{} : Bool} index_inside),
    DivisionOrder.remainder_small(index,width(geometry),width_positive))

# A property of every cell of a leaf, from a lemma for one cell with its three
# bounds. A producer names its evidence record and its cell property, says
# where the geometry and the row count are, and proves only the cell.
def cells_columns(~Evidence: Data,~Property: @+evidence: Evidence -> @+channel: Nat -> @+row: Nat -> @+column: Nat -> Data,
  ~geometry: Evidence -> Bands.Geometry,~rows: Evidence -> U32,
  ~cell: @+evidence: Evidence -> @+channel: Nat -> @+row: Nat -> @+column: Nat -> @channel_inside: {Nat.is_lt(channel,channels(geometry(evidence))) == True{} : Bool} ->
    @row_inside: {Nat.is_lt(row,U32.to_nat(rows(evidence))) == True{} : Bool} -> @column_inside: {Nat.is_lt(column,width(geometry(evidence))) == True{} : Bool} -> Property(evidence,channel,row,column),
  count: Nat,+start: Nat,+evidence: Evidence,+channel: Nat,+row: Nat,
  +channel_inside: {Nat.is_lt(channel,channels(geometry(evidence))) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows(evidence))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,count),width(geometry(evidence))) == True{} : Bool}) ->
  Members.All(Nat,{column => Property(evidence,channel,row,column) : Nat -> Data},Ranges.range(count,start)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{cell(evidence,channel,row,start,channel_inside,row_inside,Order.room_index(start,rest,width(geometry(evidence)),room)),
        cells_columns(~Evidence,~Property,~geometry,~rows,~cell,rest,1n+start,evidence,channel,row,channel_inside,row_inside,Order.advance_room(start,rest,width(geometry(evidence)),room))}

def cells_rows(~Evidence: Data,~Property: @+evidence: Evidence -> @+channel: Nat -> @+row: Nat -> @+column: Nat -> Data,
  ~geometry: Evidence -> Bands.Geometry,~rows: Evidence -> U32,
  ~cell: @+evidence: Evidence -> @+channel: Nat -> @+row: Nat -> @+column: Nat -> @channel_inside: {Nat.is_lt(channel,channels(geometry(evidence))) == True{} : Bool} ->
    @row_inside: {Nat.is_lt(row,U32.to_nat(rows(evidence))) == True{} : Bool} -> @column_inside: {Nat.is_lt(column,width(geometry(evidence))) == True{} : Bool} -> Property(evidence,channel,row,column),
  count: Nat,+start: Nat,+evidence: Evidence,+channel: Nat,+channel_inside: {Nat.is_lt(channel,channels(geometry(evidence))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,count),U32.to_nat(rows(evidence))) == True{} : Bool}) ->
  Members.All(Nat,{row => Members.All(Nat,{column => Property(evidence,channel,row,column) : Nat -> Data},Ranges.range(width(geometry(evidence)),0n)) : Nat -> Data},Ranges.range(count,start)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{cells_columns(~Evidence,~Property,~geometry,~rows,~cell,width(geometry(evidence)),0n,evidence,channel,start,channel_inside,Order.room_index(start,rest,U32.to_nat(rows(evidence)),room),Order.reflexive(width(geometry(evidence)))),
        cells_rows(~Evidence,~Property,~geometry,~rows,~cell,rest,1n+start,evidence,channel,channel_inside,Order.advance_room(start,rest,U32.to_nat(rows(evidence)),room))}

def cells_channels(~Evidence: Data,~Property: @+evidence: Evidence -> @+channel: Nat -> @+row: Nat -> @+column: Nat -> Data,
  ~geometry: Evidence -> Bands.Geometry,~rows: Evidence -> U32,
  ~cell: @+evidence: Evidence -> @+channel: Nat -> @+row: Nat -> @+column: Nat -> @channel_inside: {Nat.is_lt(channel,channels(geometry(evidence))) == True{} : Bool} ->
    @row_inside: {Nat.is_lt(row,U32.to_nat(rows(evidence))) == True{} : Bool} -> @column_inside: {Nat.is_lt(column,width(geometry(evidence))) == True{} : Bool} -> Property(evidence,channel,row,column),
  count: Nat,+start: Nat,+evidence: Evidence,+room: {Nat.is_le(Nat.add(start,count),channels(geometry(evidence))) == True{} : Bool}) ->
  Members.All(Nat,{channel => Members.All(Nat,{row => Members.All(Nat,{column => Property(evidence,channel,row,column) : Nat -> Data},Ranges.range(width(geometry(evidence)),0n)) : Nat -> Data},Ranges.range(U32.to_nat(rows(evidence)),0n)) : Nat -> Data},Ranges.range(count,start)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{cells_rows(~Evidence,~Property,~geometry,~rows,~cell,U32.to_nat(rows(evidence)),0n,evidence,start,Order.room_index(start,rest,channels(geometry(evidence)),room),Order.reflexive(U32.to_nat(rows(evidence)))),
        cells_channels(~Evidence,~Property,~geometry,~rows,~cell,rest,1n+start,evidence,Order.advance_room(start,rest,channels(geometry(evidence)),room))}

def cells(~Evidence: Data,~Property: @+evidence: Evidence -> @+channel: Nat -> @+row: Nat -> @+column: Nat -> Data,
  ~geometry: Evidence -> Bands.Geometry,~rows: Evidence -> U32,
  ~cell: @+evidence: Evidence -> @+channel: Nat -> @+row: Nat -> @+column: Nat -> @channel_inside: {Nat.is_lt(channel,channels(geometry(evidence))) == True{} : Bool} ->
    @row_inside: {Nat.is_lt(row,U32.to_nat(rows(evidence))) == True{} : Bool} -> @column_inside: {Nat.is_lt(column,width(geometry(evidence))) == True{} : Bool} -> Property(evidence,channel,row,column),
  +evidence: Evidence) ->
  Members.All(Nat,{channel => Members.All(Nat,{row => Members.All(Nat,{column => Property(evidence,channel,row,column) : Nat -> Data},Ranges.range(width(geometry(evidence)),0n)) : Nat -> Data},Ranges.range(U32.to_nat(rows(evidence)),0n)) : Nat -> Data},Ranges.range(channels(geometry(evidence)),0n)):
  cells_channels(~Evidence,~Property,~geometry,~rows,~cell,channels(geometry(evidence)),0n,evidence,Order.reflexive(channels(geometry(evidence))))

# A leaf without cells satisfies the relation with any content: no cell has a
# position below zero.
type EmptyCells<-Element: Data,-Context: Data> is Data:
  EmptyCells{context: Context,geometry: Bands.Geometry,first: U32,rows: U32,input: Tree.Tree<Element>,length: Nat,empty: {cell_count(geometry,rows) == 0n : Nat}}

def empty_geometry(~Element: Data,~Context: Data,evidence: EmptyCells<Element,Context>) -> Bands.Geometry:
  EmptyCells{context,geometry,first,rows,input,length,empty} = evidence
  geometry

def empty_rows(~Element: Data,~Context: Data,evidence: EmptyCells<Element,Context>) -> U32:
  EmptyCells{context,geometry,first,rows,input,length,empty} = evidence
  rows

def EmptyCell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+evidence: EmptyCells<Element,Context>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  EmptyCells{context,+geometry,first,+rows,input,length,empty} = evidence
  At(~Element,~Context,~value,context,geometry,first,rows,input,length,channel,row,column)

def empty_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+evidence: EmptyCells<Element,Context>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,channels(empty_geometry(~Element,~Context,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(empty_rows(~Element,~Context,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,width(empty_geometry(~Element,~Context,evidence))) == True{} : Bool}) ->
  EmptyCell(~Element,~Context,~value,evidence,channel,row,column):
  EmptyCells{+context,+geometry,+first,+rows,+input,+length,empty} = evidence
  +cell = position(geometry,rows,channel,row,column)
  Empty.absurd(At(~Element,~Context,~value,context,geometry,first,rows,input,length,channel,row,column),
    WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(cell,0n),True{},Equal.sym(Bool,Nat.is_lt(cell,0n),False{},Division.lt_zero(cell)),
      %empty : {Nat.is_lt(cell,_) == True{} : Bool} position_inside(geometry,rows,channel,row,column,channel_inside,row_inside,column_inside))))

def empty_leaf(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,
  empty: {cell_count(geometry,rows) == 0n : Nat}) -> Leaf(~Element,~Context,~value,context,geometry,first,rows,input,length):
  cells(~EmptyCells<Element,Context>,~EmptyCell(~Element,~Context,~value),~empty_geometry(~Element,~Context),~empty_rows(~Element,~Context),~empty_cell(~Element,~Context,~value),
    EmptyCells{context,geometry,first,rows,input,length,empty})

def dense_value(~Element: Data,input: Tree.Tree<Element>,index: U32) -> Element:
  Tree.read(Element,input,index)

# Change a logical context only when its pointwise values agree. The actual
# bands, the cells and their positions stay identical.
def Agreement(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data) -> Type:
  @+before: Before -> @+after: After -> @+related: Related(before,after) -> @+index: U32 -> {before_value(before,index) == after_value(after,index) : Element}

def transport_columns(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  ~agreement: Agreement(~Element,~Before,~After,~before_value,~after_value,~Related),+before: Before,+after: After,+related: Related(before,after),
  +geometry: Bands.Geometry,+first: U32,+rows: U32,+tree: Tree.Tree<Element>,+length: Nat,+channel: Nat,+row: Nat,items: List<&2,Nat>,
  values: Members.All(Nat,{column => At(~Element,~Before,~before_value,before,geometry,first,rows,tree,length,channel,row,column) : Nat -> Data},items)) ->
  Members.All(Nat,{column => At(~Element,~After,~after_value,after,geometry,first,rows,tree,length,channel,row,column) : Nat -> Data},items):
  match items:
    case Nil{}: Unit{}
    case +column <> rest:
      Shape.Both{head,tail} = values
      Shape.Both{inside,equal} = head
      +index = U32.from_nat(pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))
      Shape.Both{Shape.Both{inside,Equal.trans(Element,Tree.read(Element,tree,U32.from_nat(position(geometry,rows,channel,row,column))),before_value(before,index),after_value(after,index),
          equal,agreement(before,after,related,index))},
        transport_columns(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,first,rows,tree,length,channel,row,rest,tail)}

def transport_rows(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  ~agreement: Agreement(~Element,~Before,~After,~before_value,~after_value,~Related),+before: Before,+after: After,+related: Related(before,after),
  +geometry: Bands.Geometry,+first: U32,+rows: U32,+tree: Tree.Tree<Element>,+length: Nat,+channel: Nat,items: List<&2,Nat>,
  values: Members.All(Nat,{row => Members.All(Nat,{column => At(~Element,~Before,~before_value,before,geometry,first,rows,tree,length,channel,row,column) : Nat -> Data},Ranges.range(width(geometry),0n)) : Nat -> Data},items)) ->
  Members.All(Nat,{row => Members.All(Nat,{column => At(~Element,~After,~after_value,after,geometry,first,rows,tree,length,channel,row,column) : Nat -> Data},Ranges.range(width(geometry),0n)) : Nat -> Data},items):
  match items:
    case Nil{}: Unit{}
    case +row <> rest:
      Shape.Both{head,tail} = values
      Shape.Both{transport_columns(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,first,rows,tree,length,channel,row,Ranges.range(width(geometry),0n),head),
        transport_rows(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,first,rows,tree,length,channel,rest,tail)}

def transport_channels(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  ~agreement: Agreement(~Element,~Before,~After,~before_value,~after_value,~Related),+before: Before,+after: After,+related: Related(before,after),
  +geometry: Bands.Geometry,+first: U32,+rows: U32,+tree: Tree.Tree<Element>,+length: Nat,items: List<&2,Nat>,
  values: Members.All(Nat,{channel => Members.All(Nat,{row => Members.All(Nat,{column => At(~Element,~Before,~before_value,before,geometry,first,rows,tree,length,channel,row,column) : Nat -> Data},Ranges.range(width(geometry),0n)) : Nat -> Data},Ranges.range(U32.to_nat(rows),0n)) : Nat -> Data},items)) ->
  Members.All(Nat,{channel => Members.All(Nat,{row => Members.All(Nat,{column => At(~Element,~After,~after_value,after,geometry,first,rows,tree,length,channel,row,column) : Nat -> Data},Ranges.range(width(geometry),0n)) : Nat -> Data},Ranges.range(U32.to_nat(rows),0n)) : Nat -> Data},items):
  match items:
    case Nil{}: Unit{}
    case +channel <> rest:
      Shape.Both{head,tail} = values
      Shape.Both{transport_rows(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,first,rows,tree,length,channel,Ranges.range(U32.to_nat(rows),0n),head),
        transport_channels(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,first,rows,tree,length,rest,tail)}

def values(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  ~agreement: Agreement(~Element,~Before,~After,~before_value,~after_value,~Related),+before: Before,+after: After,+related: Related(before,after),+geometry: Bands.Geometry,
  bands: Model.Content<Element>,evidence: Values(~Element,~Before,~before_value,before,geometry,bands)) ->
  Values(~Element,~After,~after_value,after,geometry,bands):
  match bands:
    case Model.RowBand{Buffer.Content{tree,length},first,rows}:
      transport_channels(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,first,rows,tree,length,Ranges.range(channels(geometry),0n),evidence)
    case Model.Joined{left,right}:
      Shape.Both{left_values,right_values} = evidence
      Shape.Both{values(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,left,left_values),
        values(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,right,right_values)}
