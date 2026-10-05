# Band concatenation: the final layout, channel-range copies in either
# schedule, and their ordered values.
import Base
import ./tensor_band_range.bend as BandRange
import ./tensor_band_gather.bend as BandGather
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_model.bend as Model
import ../tensor_band.bend as Bands
import ../tensor_band_concat.bend as BandConcat
import ./storage_copy_refinement.bend as StorageCopyLaws
import ../tensor.bend as Tensors
import ./storage_tree_model.bend as Tree
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition
import ../machine_limits.bend as Machine
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./storage_shape.bend as Shape
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./word_arithmetic.bend as WordArithmetic
import ./contiguous_copy.bend as Contiguous
import ./contiguous_copy_readback.bend as CopyReadback
import ./nat_interval.bend as Interval
import ./nat_to_u32_bounds.bend as Bounds
import ./ordered_ranges.bend as Ranges
import ./row_major_coordinates.bend as RowCoordinates
import ./sequence_membership.bend as Members
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_band_copy.bend as BandCopy
import ./tensor_band_each.bend as Each
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_plan.bend as BandPlan
import ./word_subtraction.bend as Subtraction
import ./write_program.bend as Writes
import ../storage_copy.bend as Copy
import ../tensor_band_schedule.bend as BandSchedule

# Concatenation allocates the final layout once and writes each channel range
# directly into it. This connects allocation and the aligned leaf copy;
# coverage, channel offsets, and the Parts caller are separate obligations.
def allocation_model(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan) -> Model.Content<Element>:
  match layout:
    case Bands.Rows{first,+rows}:
      +count = Bands.local_count(geometry,rows)
      Model.RowBand{Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)},first,rows}
    case Bands.Split{left,right}:
      Model.Joined{allocation_model(~Element,geometry,fill,left),allocation_model(~Element,geometry,fill,right)}

def allocate(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan) ->
  {Model.contents(~Element,BandConcat.allocate(~Element,layout,geometry,fill)) == allocation_model(~Element,geometry,fill,layout) : Model.Content<Element>}:
  match layout:
    case Bands.Rows{+first,+rows}:
      +count = Bands.local_count(geometry,rows)
      Equal.cong(Buffer.Content<Element>,Model.Content<Element>,buffer => Model.RowBand{buffer,first,rows},
        Buffer.contents(~Element,Tensors.buffer(~Element,fill,count)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)},
        AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)))
    case Bands.Split{+left,+right}:
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandConcat.allocate(~Element,left,geometry,fill)),allocation_model(~Element,geometry,fill,left),allocate(~Element,geometry,fill,left)) :
        {Model.Joined{_,Model.contents(~Element,BandConcat.allocate(~Element,right,geometry,fill))} == Model.Joined{allocation_model(~Element,geometry,fill,left),allocation_model(~Element,geometry,fill,right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandConcat.allocate(~Element,right,geometry,fill)),allocation_model(~Element,geometry,fill,right),allocate(~Element,geometry,fill,right)) :
        {Model.Joined{allocation_model(~Element,geometry,fill,left),_} == Model.Joined{allocation_model(~Element,geometry,fill,left),allocation_model(~Element,geometry,fill,right)} : Model.Content<Element>}
      {==}

def copied_packed(~Element: Data,+count: U32,+offset: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Bands.destination(~Element,Copy.range(~Element,count,0,offset,Buffer.pack(~Element,input),Buffer.pack(~Element,output)))) ==
    BandGather.output(~Element,StorageCopyLaws.model(~Element,count,0,offset,StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,input),Buffer.pack(~Element,output))))) : Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element>,
    Buffer.contents(~Element,Bands.destination(~Element,Copy.range(~Element,count,0,offset,Buffer.pack(~Element,input),Buffer.pack(~Element,output)))),
    BandGather.output(~Element,StorageCopyLaws.observe(~Element,Copy.range(~Element,count,0,offset,Buffer.pack(~Element,input),Buffer.pack(~Element,output)))),
    BandGather.output(~Element,StorageCopyLaws.model(~Element,count,0,offset,StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,input),Buffer.pack(~Element,output))))),
    BandGather.destination(~Element,Copy.range(~Element,count,0,offset,Buffer.pack(~Element,input),Buffer.pack(~Element,output))),
    Equal.cong(StorageCopyLaws.Owners<Element>,Buffer.Content<Element>,owners => BandGather.output(~Element,owners),
      StorageCopyLaws.observe(~Element,Copy.range(~Element,count,0,offset,Buffer.pack(~Element,input),Buffer.pack(~Element,output))),
      StorageCopyLaws.model(~Element,count,0,offset,StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,input),Buffer.pack(~Element,output)))),StorageCopyLaws.range(~Element,count,0,offset,(Buffer.pack(~Element,input),Buffer.pack(~Element,output)))))

def copied_observed(~Element: Data,+count: U32,+offset: U32,-input: Storage.Buffer<Element>,-output: Storage.Buffer<Element>,
  input_observation: Buffer.Observation<Element,input>,output_observation: Buffer.Observation<Element,output>) ->
  {Buffer.contents(~Element,Bands.destination(~Element,Copy.range(~Element,count,0,offset,input,output))) ==
    BandGather.output(~Element,StorageCopyLaws.model(~Element,count,0,offset,StorageCopyLaws.observe(~Element,(input,output)))) : Buffer.Content<Element>}:
  Buffer.Observation{+input_model,input_equal} = input_observation
  Buffer.Observation{+output_model,output_equal} = output_observation
  %input_equal :
    {Buffer.contents(~Element,Bands.destination(~Element,Copy.range(~Element,count,0,offset,_,output))) ==
      BandGather.output(~Element,StorageCopyLaws.model(~Element,count,0,offset,StorageCopyLaws.observe(~Element,(_,output)))) : Buffer.Content<Element>}
  %output_equal :
    {Buffer.contents(~Element,Bands.destination(~Element,Copy.range(~Element,count,0,offset,Buffer.pack(~Element,input_model),_))) ==
      BandGather.output(~Element,StorageCopyLaws.model(~Element,count,0,offset,StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,input_model),_)))) : Buffer.Content<Element>}
  copied_packed(~Element,count,offset,input_model,output_model)

def copied(~Element: Data,+count: U32,+offset: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Bands.destination(~Element,Copy.range(~Element,count,0,offset,input,output))) ==
    BandGather.output(~Element,StorageCopyLaws.model(~Element,count,0,offset,StorageCopyLaws.observe(~Element,(input,output)))) : Buffer.Content<Element>}:
  copied_observed(~Element,count,offset,input,output,Buffer.observe(~Element,input),Buffer.observe(~Element,output))

def leaf_execution(~Element: Data,+parallel: Bool,+geometry: Bands.Geometry,+channel: U32,+first: U32,+rows: U32,+out_first: U32,+out_rows: U32,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  {BandConcat.append_aligned(~Element,parallel,geometry,channel,Partition.Leaf{input,first,rows,Unit{}},Partition.Leaf{output,out_first,out_rows,Unit{}}) ==
    Partition.Leaf{Bands.destination(~Element,Copy.range(~Element,Bands.local_count(geometry,out_rows),0,BandConcat.channel_offset(geometry,channel,out_rows),input,output)),out_first,out_rows,Unit{}} : Bands.Bands(~Element)}:
  {==}

def aligned_leaf(~Element: Data,+parallel: Bool,+geometry: Bands.Geometry,+channel: U32,+first: U32,+rows: U32,+out_first: U32,+out_rows: U32,
  +input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {Model.contents(~Element,BandConcat.append_aligned(~Element,parallel,geometry,channel,
      Model.pack(~Element,Model.Leaf{input,first,rows}),Model.pack(~Element,Model.Leaf{output,out_first,out_rows}))) ==
    Model.RowBand{BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(geometry,out_rows),0,BandConcat.channel_offset(geometry,channel,out_rows),
      BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output)))),out_first,out_rows} : Model.Content<Element>}:
  %Equal.sym(Bands.Bands(~Element),BandConcat.append_aligned(~Element,parallel,geometry,channel,
      Model.pack(~Element,Model.Leaf{input,first,rows}),Model.pack(~Element,Model.Leaf{output,out_first,out_rows})),
    Partition.Leaf{Bands.destination(~Element,Copy.range(~Element,Bands.local_count(geometry,out_rows),0,BandConcat.channel_offset(geometry,channel,out_rows),Buffer.pack(~Element,input),Buffer.pack(~Element,output))),out_first,out_rows,Unit{}},
    leaf_execution(~Element,parallel,geometry,channel,first,rows,out_first,out_rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output))) :
    {Model.contents(~Element,_) == Model.RowBand{BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(geometry,out_rows),0,BandConcat.channel_offset(geometry,channel,out_rows),
      BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output)))),out_first,out_rows} : Model.Content<Element>}
  %BandGather.pair_contents(~Element,input,output) :
    {Model.contents(~Element,Partition.Leaf{Bands.destination(~Element,Copy.range(~Element,Bands.local_count(geometry,out_rows),0,BandConcat.channel_offset(geometry,channel,out_rows),Buffer.pack(~Element,input),Buffer.pack(~Element,output))),out_first,out_rows,Unit{}}) ==
      Model.RowBand{BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(geometry,out_rows),0,BandConcat.channel_offset(geometry,channel,out_rows),_)),out_first,out_rows} : Model.Content<Element>}
  Equal.cong(Buffer.Content<Element>,Model.Content<Element>,buffer => Model.RowBand{buffer,out_first,out_rows},
    Buffer.contents(~Element,Bands.destination(~Element,Copy.range(~Element,Bands.local_count(geometry,out_rows),0,BandConcat.channel_offset(geometry,channel,out_rows),Buffer.pack(~Element,input),Buffer.pack(~Element,output)))),
    BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(geometry,out_rows),0,BandConcat.channel_offset(geometry,channel,out_rows),StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,input),Buffer.pack(~Element,output))))),
    copied(~Element,Bands.local_count(geometry,out_rows),BandConcat.channel_offset(geometry,channel,out_rows),Buffer.pack(~Element,input),Buffer.pack(~Element,output)))

# Matching band trees use the same final-owner copies in either schedule.
# Mismatched structures retain the output, as the actual fallback does.
def band_model(~Element: Data,context: BandConcat.Channel,first: U32,rows: U32,output: Buffer.Model<Element>,input: Buffer.Model<Element>) -> Model.Content<Element>:
  BandConcat.Channel{+geometry,channel} = context
  +rows = rows
  Model.RowBand{BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(geometry,rows),0,BandConcat.channel_offset(geometry,channel,rows),
    BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output)))),first,rows}

# The output bands lead: where the two trees differ in shape, the output is kept.
def pipeline_model(~Element: Data,geometry: Bands.Geometry,channel: U32,input: Model.Model<Element>,output: Model.Model<Element>) -> Model.Content<Element>:
  Each.content_zip(~Element,~BandConcat.Channel,~band_model(~Element),BandConcat.Channel{geometry,channel},output,input)

def band(~Element: Data,+context: BandConcat.Channel,+first: U32,+rows: U32,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>) ->
  {Model.contents(~Element,BandConcat.append_band(~Element,context,first,rows,Buffer.pack(~Element,output),Buffer.pack(~Element,input),Unit{})) ==
    band_model(~Element,context,first,rows,output,input) : Model.Content<Element>}:
  BandConcat.Channel{+geometry,+channel} = context
  aligned_leaf(~Element,False{},geometry,channel,first,rows,first,rows,input,output)

# The aligned append in either mode, on the plan of the output bands.
def append_aligned(~Element: Data,+parallel: Bool,+geometry: Bands.Geometry,+channel: U32,+input: Model.Model<Element>,+output: Model.Model<Element>) ->
  {Model.contents(~Element,BandConcat.append_aligned(~Element,parallel,geometry,channel,Model.pack(~Element,input),Model.pack(~Element,output))) ==
    pipeline_model(~Element,geometry,channel,input,output) : Model.Content<Element>}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,output)),(Model.pack(~Element,output),Model.layout(~Element,output)),Model.plan(~Element,output)) :
    {Model.contents(~Element,BandConcat.append_planned(~Element,parallel,geometry,channel,Model.pack(~Element,input),_)) == pipeline_model(~Element,geometry,channel,input,output) : Model.Content<Element>}
  Each.zipped(~Element,~BandConcat.Channel,~BandConcat.append_band(~Element),~band_model(~Element),~band(~Element),
    output,input,parallel && Bands.same_image(geometry,Model.layout(~Element,output)),parallel,geometry,BandConcat.Channel{geometry,channel})

# Different source boundaries copy their intersecting channel ranges directly
# into the final output owner. This is the actual Banded rows branch; no
# intermediate aligned or dense tensor is inserted by the proof model.
def rows_output_content(~Element: Data,owners: Tensors.Tensor<Element> & Storage.Buffer<Element>) -> Buffer.Content<Element>:
  (source,output) = owners
  Buffer.contents(~Element,output)

def banded_result(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  owners: Bands.Bands(~Element) & Storage.Buffer<Element>) -> Tensors.Tensor<Element> & Storage.Buffer<Element>:
  Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Tensors.Tensor<Element> & Storage.Buffer<Element>),owners,
    bands => output => (Tensors.Banded{bands,geometry,shape,fill,status,workers},output))

def projected(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  owners: Bands.Bands(~Element) & Storage.Buffer<Element>) ->
  {rows_output_content(~Element,banded_result(~Element,geometry,shape,fill,status,workers,owners)) ==
    BandRange.result(~Element,BandRange.contents(~Element,owners)) : Buffer.Content<Element>}:
  (bands,output) = owners
  {==}

def banded_rows(~Element: Data,+external_geometry: Bands.Geometry,+source_geometry: Bands.Geometry,+first: U32,+rows: U32,+channel: U32,
  +source: Model.Model<Element>,+output: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32) ->
  {rows_output_content(~Element,BandConcat.rows(~Element,external_geometry,first,rows,channel,
      Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,workers},Buffer.pack(~Element,output))) ==
    BandRange.result(~Element,BandRange.written(~Element,BandRange.Request{source_geometry,first,rows,BandConcat.channel_offset(source_geometry,channel,rows)},
      Model.content(~Element,source),Buffer.content(~Element,output))) : Buffer.Content<Element>}:
  +request = {BandRange.Request{source_geometry,first,rows,BandConcat.channel_offset(source_geometry,channel,rows)} : BandRange.Request}
  Equal.trans(Buffer.Content<Element>,
    rows_output_content(~Element,BandConcat.rows(~Element,external_geometry,first,rows,channel,
      Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,workers},Buffer.pack(~Element,output))),
    BandRange.result(~Element,BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Buffer.pack(~Element,output)))),
    BandRange.result(~Element,BandRange.written(~Element,request,Model.content(~Element,source),Buffer.content(~Element,output))),
    projected(~Element,source_geometry,shape,fill,status,workers,BandRange.run(~Element,request,Model.pack(~Element,source),Buffer.pack(~Element,output))),
    Equal.cong(BandRange.Content<Element>,Buffer.Content<Element>,content => BandRange.result(~Element,content),
      BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Buffer.pack(~Element,output))),
      BandRange.written(~Element,request,Model.content(~Element,source),Buffer.content(~Element,output)),BandRange.canonical_range(~Element,request,source,output)))

# Ordered channel-range values for the actual concat copies. The source
# Values relation supplies every copied element; no concat-result equality
# is a premise. Whole append_tree/Parts offsets and their machine bounds
# remain caller obligations, separate from this finite range contract.
def channel_length(request: BandRange.Request) -> Nat:
  BandRange.Request{Bands.Geometry{images,channels,height,width},first,rows,offset} = request
  Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))

def channel_address(request: BandRange.Request,position: Nat) -> U32:
  U32.from_nat(Nat.add(U32.to_nat(BandRange.request_offset(request)),position))

def native_address(+offset: U32,+position: Nat) ->
  {(offset + U32.from_nat(position) : U32) == U32.from_nat(Nat.add(U32.to_nat(offset),position)) : U32}:
  %Arithmetic.native_roundtrip(offset) : {(_ + U32.from_nat(position) : U32) == U32.from_nat(Nat.add(U32.to_nat(offset),position)) : U32}
  WordArithmetic.native_add(U32.to_nat(offset),position)

def zero_position_room(+offset: U32,+count: Nat,+capacity: U32,
  room: {Nat.is_le(Nat.add(U32.to_nat(offset),count),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(U32.to_nat(offset),0n),count),U32.to_nat(capacity)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(U32.to_nat(offset),0n),U32.to_nat(offset),Addition.zero_right(U32.to_nat(offset))) : {Nat.is_le(Nat.add(_,count),U32.to_nat(capacity)) == True{} : Bool}
  room

# A source cell written at the channel offset of the request: the destination
# query of the cell holds the source value of its pixel.
def ChannelAt(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+request: BandRange.Request,+output: Buffer.Content<Element>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  {Buffer.value(~Element,BandRange.coverage_query(request,channel,row,column),output) ==
    original(logical,U32.from_nat(BandValues.pixel(BandRange.request_geometry(request),channel,BandRange.global_row(request,row),column))) : Element}

def ChannelValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+request: BandRange.Request,+output: Buffer.Content<Element>) -> Data:
  Members.All(Nat,{channel => Members.All(Nat,{row => Members.All(Nat,{column => ChannelAt(~Element,~Logical,~original,logical,request,output,channel,row,column) : Nat -> Data},
    Ranges.range(BandValues.width(BandRange.request_geometry(request)),0n)) : Nat -> Data},Ranges.range(U32.to_nat(BandRange.request_rows(request)),0n)) : Nat -> Data},Ranges.range(BandValues.channels(BandRange.request_geometry(request)),0n))

def channel_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+request: BandRange.Request,+output: Buffer.Content<Element>,+channel: Nat,+row: Nat,+column: Nat,
  values: ChannelValues(~Element,~Logical,~original,logical,request,output),
  channel_inside: {Nat.is_lt(channel,BandValues.channels(BandRange.request_geometry(request))) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(BandRange.request_rows(request))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(BandRange.request_geometry(request))) == True{} : Bool}) ->
  ChannelAt(~Element,~Logical,~original,logical,request,output,channel,row,column):
  Members.all_member(Nat,{column => ChannelAt(~Element,~Logical,~original,logical,request,output,channel,row,column) : Nat -> Data},column,Ranges.range(BandValues.width(BandRange.request_geometry(request)),0n),
    Members.all_member(Nat,{row => Members.All(Nat,{column => ChannelAt(~Element,~Logical,~original,logical,request,output,channel,row,column) : Nat -> Data},Ranges.range(BandValues.width(BandRange.request_geometry(request)),0n)) : Nat -> Data},
      row,Ranges.range(U32.to_nat(BandRange.request_rows(request)),0n),
      Members.all_member(Nat,{channel => Members.All(Nat,{row => Members.All(Nat,{column => ChannelAt(~Element,~Logical,~original,logical,request,output,channel,row,column) : Nat -> Data},
    Ranges.range(BandValues.width(BandRange.request_geometry(request)),0n)) : Nat -> Data},Ranges.range(U32.to_nat(BandRange.request_rows(request)),0n)) : Nat -> Data},channel,Ranges.range(BandValues.channels(BandRange.request_geometry(request)),0n),values,Members.range_member(BandValues.channels(BandRange.request_geometry(request)),0n,channel,channel_inside)),
      Members.range_member(U32.to_nat(BandRange.request_rows(request)),0n,row,row_inside)),
    Members.range_member(BandValues.width(BandRange.request_geometry(request)),0n,column,column_inside))

# The aligned caller supplies only destination room. Exact native count is
# derived from the actual source metadata and Ready, including zero rows.
def count_exact(+geometry: Bands.Geometry,+rows: U32,+length: Nat,count: BandShape.Count<geometry,rows,length>) ->
  {U32.to_nat(Bands.local_count(geometry,rows)) == length : Nat}:
  BandShape.Count{exact,allocatable} = count
  exact

def copied_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,+offset: U32,
  +input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+channel: Nat,+row: Nat,+column: Nat,
  width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),input_ready: Buffer.Ready<Element,input>,
  output_ready: Buffer.Ready<Element,output>,room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(Bands.local_count(geometry,rows))),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool},
  cell: BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input),channel,row,column)) ->
  ChannelAt(~Element,~Logical,~original,logical,BandRange.Request{geometry,first,rows,offset},
    BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(geometry,rows),0,offset,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output)))),channel,row,column):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Buffer.Ready{+depth_ok,output_length_room} = output_ready
  Shape.Both{inside,equal} = cell
  +count = Bands.local_count(Bands.Geometry{images,channels,height,width},rows)
  +position = BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column)
  +pixel = U32.from_nat(BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column))
  +exact = count_exact(Bands.Geometry{images,channels,height,width},rows,input_length,BandShape.count(~Element,Bands.Geometry{images,channels,height,width},first,rows,input,width_positive,metadata,input_ready))
  %native_address(offset,position) :
    {Buffer.value(~Element,_,BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(Bands.Geometry{images,channels,height,width},rows),0,offset,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output))))) == original(logical,pixel) : Element}
  Equal.trans(Element,
    Buffer.value(~Element,(offset + U32.from_nat(position) : U32),BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(Bands.Geometry{images,channels,height,width},rows),0,offset,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output))))),
    Tree.read(Element,Buffer.tree(~Element,input),(0 + U32.from_nat(position) : U32)),original(logical,pixel),
    CopyReadback.read_inside(~Element,U32.to_nat(count),0n,position,Buffer.tree(~Element,input),Buffer.tree(~Element,output),0,offset,Buffer.depth(~Element,output),Buffer.balanced(~Element,output),depth_ok,Buffer.capacity_bound(~Element,output,depth_ok),
      zero_position_room(offset,U32.to_nat(count),Tree.capacity(Element,Buffer.tree(~Element,output)),room),
      %Equal.sym(Nat,U32.to_nat(count),input_length,exact) : {Nat.is_lt(position,_) == True{} : Bool} inside),
    %Equal.sym(U32,(0 + U32.from_nat(position) : U32),U32.from_nat(position),Arithmetic.add_zero_left(U32.from_nat(position))) : {Tree.read(Element,Buffer.tree(~Element,input),_) == original(logical,pixel) : Element}
    equal)

# What one aligned leaf copy supplies to every cell.
type CopiedCells<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  CopiedCells{logical: Logical,geometry: Bands.Geometry,first: U32,rows: U32,offset: U32,input: Buffer.Model<Element>,output: Buffer.Model<Element>,
    width_positive: {BandShape.width_positive(geometry) == True{} : Bool},metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),
    input_ready: Buffer.Ready<Element,input>,output_ready: Buffer.Ready<Element,output>,room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(Bands.local_count(geometry,rows))),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool},
    values: BandValues.Leaf(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input))}

def copied_geometry(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: CopiedCells<Element,Logical,original>) -> Bands.Geometry:
  CopiedCells{logical,geometry,first,rows,offset,input,output,width_positive,metadata,input_ready,output_ready,room,values} = evidence
  geometry

def copied_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: CopiedCells<Element,Logical,original>) -> U32:
  CopiedCells{logical,geometry,first,rows,offset,input,output,width_positive,metadata,input_ready,output_ready,room,values} = evidence
  rows

def CopiedCell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: CopiedCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  CopiedCells{logical,+geometry,+first,+rows,+offset,+input,+output,width_positive,metadata,input_ready,output_ready,room,values} = evidence
  ChannelAt(~Element,~Logical,~original,logical,BandRange.Request{geometry,first,rows,offset},
    BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(geometry,rows),0,offset,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output)))),channel,row,column)

def copied_leaf_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: CopiedCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(copied_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(copied_rows(~Element,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(copied_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool}) ->
  CopiedCell(~Element,~Logical,~original,evidence,channel,row,column):
  CopiedCells{+logical,+geometry,+first,+rows,+offset,+input,+output,width_positive,metadata,input_ready,output_ready,room,values} = evidence
  copied_cell(~Element,~Logical,~original,logical,geometry,first,rows,offset,input,output,channel,row,column,width_positive,metadata,input_ready,output_ready,room,
    BandValues.leaf_cell(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input),channel,row,column,values,channel_inside,row_inside,column_inside))

def copied_cells(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,+offset: U32,
  +input: Buffer.Model<Element>,+output: Buffer.Model<Element>,
  width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),input_ready: Buffer.Ready<Element,input>,output_ready: Buffer.Ready<Element,output>,
  values: BandValues.Leaf(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input)),
  room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(Bands.local_count(geometry,rows))),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool}) ->
  ChannelValues(~Element,~Logical,~original,logical,BandRange.Request{geometry,first,rows,offset},
    BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(geometry,rows),0,offset,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output))))):
  BandValues.cells(~CopiedCells<Element,Logical,original>,~CopiedCell(~Element,~Logical,~original),~copied_geometry(~Element,~Logical,~original),~copied_rows(~Element,~Logical,~original),~copied_leaf_cell(~Element,~Logical,~original),
    CopiedCells{logical,geometry,first,rows,offset,input,output,width_positive,metadata,input_ready,output_ready,room,values})

# An append starts after the previously written channels. A cell below the
# channel offset keeps its value; no per-write disjointness is supplied.
def copied_prefix_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+output_geometry: Bands.Geometry,+first: U32,+rows: U32,
  +count: U32,+offset: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+channel: Nat,+row: Nat,+column: Nat,
  output_ready: Buffer.Ready<Element,output>,
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool},
  +before: {Nat.is_lt(BandValues.position(output_geometry,rows,channel,row,column),U32.to_nat(offset)) == True{} : Bool},
  cell: BandValues.At(~Element,~Logical,~original,logical,output_geometry,first,rows,Buffer.tree(~Element,output),Buffer.length(~Element,output),channel,row,column)) ->
  BandValues.At(~Element,~Logical,~original,logical,output_geometry,first,rows,
    Buffer.content_tree(~Element,BandGather.output(~Element,StorageCopyLaws.model(~Element,count,0,offset,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output))))),
    Buffer.content_length(~Element,BandGather.output(~Element,StorageCopyLaws.model(~Element,count,0,offset,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output))))),channel,row,column):
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Buffer.Ready{+depth_ok,length_room} = output_ready
  Shape.Both{inside,equal} = cell
  +position = BandValues.position(output_geometry,rows,channel,row,column)
  +pixel = U32.from_nat(BandValues.pixel(output_geometry,channel,Nat.add(U32.to_nat(first),row),column))
  +capacity = Tree.capacity(Element,Buffer.tree(~Element,output))
  +capacity_ok = Buffer.capacity_bound(~Element,output,depth_ok)
  +position_inside = Bounds.nat_trans(position,U32.to_nat(offset),U32.to_nat(capacity),before,
    Order.transitive(U32.to_nat(offset),Nat.add(U32.to_nat(offset),U32.to_nat(count)),U32.to_nat(capacity),Order.le_add(U32.to_nat(offset),U32.to_nat(count)),room))
  +position_exact = Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))
  Shape.Both{inside,
    Equal.trans(Element,
      Buffer.value(~Element,U32.from_nat(position),BandGather.output(~Element,StorageCopyLaws.model(~Element,count,0,offset,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output))))),
      Tree.read(Element,Buffer.tree(~Element,output),U32.from_nat(position)),original(logical,pixel),
      CopyReadback.read_outside(~Element,U32.to_nat(count),0n,U32.from_nat(position),Buffer.tree(~Element,input),Buffer.tree(~Element,output),0,offset,Buffer.depth(~Element,output),Buffer.balanced(~Element,output),depth_ok,capacity_ok,
        zero_position_room(offset,U32.to_nat(count),capacity,room),Bounds.native_bound(position,capacity,position_exact,position_inside),
        Interval.Before{
          %Equal.sym(Nat,Nat.add(U32.to_nat(offset),0n),U32.to_nat(offset),Addition.zero_right(U32.to_nat(offset))) : {Nat.is_lt(U32.to_nat(U32.from_nat(position)),_) == True{} : Bool}
          %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,position_exact) : {Nat.is_lt(_,U32.to_nat(offset)) == True{} : Bool}
          before}),
      equal)}

# Different source plans use the already proved intersection/coverage
# producer; callers do not supply write matching or query coverage.
type WrittenCells<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  WrittenCells{logical: Logical,request: BandRange.Request,source: Model.Model<Element>,output: Buffer.Model<Element>,limits: Machine.Limits,target_length: Nat,capacity: U32,
    domain: BandGather.Domain<BandRange.request_geometry(request),limits,capacity>,target: BandRange.Target(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
    source_well_formed: BandShape.WellFormed<Element,BandRange.request_geometry(request),Model.content(~Element,source)>,
    target_metadata: BandShape.Leaf(BandRange.request_geometry(request),BandRange.request_first(request),BandRange.request_rows(request),target_length),
    ready: Buffer.Ready<Element,output>,values: BandValues.Values(~Element,~Logical,~original,logical,BandRange.request_geometry(request),Model.content(~Element,source))}

def written_geometry(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: WrittenCells<Element,Logical,original>) -> Bands.Geometry:
  WrittenCells{logical,request,source,output,limits,target_length,capacity,domain,target,source_well_formed,target_metadata,ready,values} = evidence
  BandRange.request_geometry(request)

def written_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: WrittenCells<Element,Logical,original>) -> U32:
  WrittenCells{logical,request,source,output,limits,target_length,capacity,domain,target,source_well_formed,target_metadata,ready,values} = evidence
  BandRange.request_rows(request)

def WrittenCell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: WrittenCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  WrittenCells{logical,+request,+source,+output,limits,target_length,capacity,domain,target,source_well_formed,target_metadata,ready,values} = evidence
  ChannelAt(~Element,~Logical,~original,logical,request,BandRange.result(~Element,BandRange.written(~Element,request,Model.content(~Element,source),Buffer.content(~Element,output))),channel,row,column)

def written_leaf_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: WrittenCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(written_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(written_rows(~Element,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(written_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool}) ->
  WrittenCell(~Element,~Logical,~original,evidence,channel,row,column):
  WrittenCells{+logical,+request,+source,+output,+limits,+target_length,+capacity,domain,target,source_well_formed,target_metadata,ready,values} = evidence
  BandRange.Request{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,+offset} = request
  BandRange.written_point(~Element,~Logical,~original,logical,BandRange.Request{Bands.Geometry{images,channels,height,width},first,rows,offset},source,output,limits,target_length,capacity,domain,target,source_well_formed,target_metadata,ready,values,
    channel,row,column,channel_inside,row_inside,column_inside)

def written_cells(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+request: BandRange.Request,+source: Model.Model<Element>,+output: Buffer.Model<Element>,
  +limits: Machine.Limits,+target_length: Nat,+capacity: U32,
  domain: BandGather.Domain<BandRange.request_geometry(request),limits,capacity>,target: BandRange.Target(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
  source_well_formed: BandShape.WellFormed<Element,BandRange.request_geometry(request),Model.content(~Element,source)>,
  target_metadata: BandShape.Leaf(BandRange.request_geometry(request),BandRange.request_first(request),BandRange.request_rows(request),target_length),
  ready: Buffer.Ready<Element,output>,values: BandValues.Values(~Element,~Logical,~original,logical,BandRange.request_geometry(request),Model.content(~Element,source))) ->
  ChannelValues(~Element,~Logical,~original,logical,request,BandRange.result(~Element,BandRange.written(~Element,request,Model.content(~Element,source),Buffer.content(~Element,output)))):
  BandValues.cells(~WrittenCells<Element,Logical,original>,~WrittenCell(~Element,~Logical,~original),~written_geometry(~Element,~Logical,~original),~written_rows(~Element,~Logical,~original),~written_leaf_cell(~Element,~Logical,~original),
    WrittenCells{logical,request,source,output,limits,target_length,capacity,domain,target,source_well_formed,target_metadata,ready,values})

# Concat follows the actual source-owner thread: rows return its observed
# source, append_tree visits output leaves left then right, and Parts update
# the channel cursor in input order. No aligned or dense temporary is added.
# Contents retain tensor metadata while erasing only the certificates.
type TensorContent<-Element: Data> is Data:
  DenseContent{buffer: Buffer.Content<Element>,shape: List<&2,U32>,fill: Element,status: Tensors.Status,workers: U32}
  BandedContent{bands: Model.Content<Element>,geometry: Bands.Geometry,shape: List<&2,U32>,fill: Element,status: Tensors.Status,workers: U32}

def tensor_contents(~Element: Data,input: Tensors.Tensor<Element>) -> TensorContent<Element>:
  match input:
    case Tensors.Tensor{buffer,shape,fill,status,workers}: DenseContent{Buffer.contents(~Element,buffer),shape,fill,status,workers}
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: BandedContent{Model.contents(~Element,bands),geometry,shape,fill,status,workers}

# Proof observations keep the existing Dense/Banded representation and its
# actual certificate. Each affine owner is observed once before proof reuse.
type TensorOwnerModel<-Element: Data> is Data:
  DenseOwner{buffer: Buffer.Model<Element>,shape: List<&2,U32>,fill: Element,status: Tensors.Status,workers: U32}
  BandedOwner{bands: Model.Model<Element>,geometry: Bands.Geometry,shape: List<&2,U32>,fill: Element,status: Tensors.Status,workers: U32}

def tensor_owner_pack(~Element: Data,owner: TensorOwnerModel<Element>) -> Tensors.Tensor<Element>:
  match owner:
    case DenseOwner{buffer,shape,fill,status,workers}: Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers}
    case BandedOwner{bands,geometry,shape,fill,status,workers}: Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers}

type TensorObservation<-Element: Data,-tensor: Tensors.Tensor<Element>> is Data:
  TensorObservation{model: TensorOwnerModel<Element>,equation: {tensor_owner_pack(~Element,model) == tensor : Tensors.Tensor<Element>}}

def tensor_observed_buffer(~Element: Data,-storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  observation: Buffer.Observation<Element,storage>) -> TensorObservation<Element,Tensors.Tensor{storage,shape,fill,status,workers}>:
  Buffer.Observation{+model,equation} = observation
  TensorObservation{DenseOwner{model,shape,fill,status,workers},
    Equal.cong(Storage.Buffer<Element>,Tensors.Tensor<Element>,owner => Tensors.Tensor{owner,shape,fill,status,workers},Buffer.pack(~Element,model),storage,equation)}

def tensor_observed_bands(~Element: Data,-bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  observation: Model.Observation<Element,bands>) -> TensorObservation<Element,Tensors.Banded{bands,geometry,shape,fill,status,workers}>:
  Model.Observation{+model,equation} = observation
  TensorObservation{BandedOwner{model,geometry,shape,fill,status,workers},
    Equal.cong(Bands.Bands(~Element),Tensors.Tensor<Element>,owner => Tensors.Banded{owner,geometry,shape,fill,status,workers},Model.pack(~Element,model),bands,equation)}

def observe_tensor(~Element: Data,tensor: Tensors.Tensor<Element>) -> TensorObservation<Element,tensor>:
  match tensor:
    case Tensors.Tensor{storage,+shape,+fill,+status,+workers}: tensor_observed_buffer(~Element,storage,shape,fill,status,workers,Buffer.observe(~Element,storage))
    case Tensors.Banded{bands,+geometry,+shape,+fill,+status,+workers}: tensor_observed_bands(~Element,bands,geometry,shape,fill,status,workers,Model.observe(~Element,bands))

type TreeOwnersObservation<-Element: Data,-owners: Tensors.Tensor<Element> & Bands.Bands(~Element)> is Data:
  TreeOwnersObservation{input: TensorOwnerModel<Element>,output: Model.Model<Element>,equation: {(tensor_owner_pack(~Element,input),Model.pack(~Element,output)) == owners : Tensors.Tensor<Element> & Bands.Bands(~Element)}}

def tree_owners_observed(~Element: Data,-input: Tensors.Tensor<Element>,-output: Bands.Bands(~Element),
  input_observation: TensorObservation<Element,input>,output_observation: Model.Observation<Element,output>) -> TreeOwnersObservation<Element,(input,output)>:
  TensorObservation{+source,source_equation} = input_observation
  Model.Observation{+target,target_equation} = output_observation
  TreeOwnersObservation{source,target,
    %source_equation : {(tensor_owner_pack(~Element,source),Model.pack(~Element,target)) == (_,output) : Tensors.Tensor<Element> & Bands.Bands(~Element)}
    %target_equation : {(tensor_owner_pack(~Element,source),Model.pack(~Element,target)) == (tensor_owner_pack(~Element,source),_) : Tensors.Tensor<Element> & Bands.Bands(~Element)}
    {==}}

def observe_tree_owners(~Element: Data,owners: Tensors.Tensor<Element> & Bands.Bands(~Element)) -> TreeOwnersObservation<Element,owners>:
  (input,output) = owners
  tree_owners_observed(~Element,input,output,observe_tensor(~Element,input),Model.observe(~Element,output))

def tensor_shape(~Element: Data,input: TensorContent<Element>) -> List<&2,U32>:
  match input:
    case DenseContent{buffer,shape,fill,status,workers}: shape
    case BandedContent{bands,geometry,shape,fill,status,workers}: shape

def dense_writes(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,channel: U32,input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  Bands.Geometry{images,+channels,+height,+width} = geometry
  BandCopy.writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),
    ((first / height * channels * height + first % height) * width : U32),(channel * rows * width : U32),input)

def row_writes(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+channel: U32,input: TensorContent<Element>) -> List<&2,Writes.Assignment<Element>>:
  match input:
    case DenseContent{Buffer.Content{tree,length},shape,fill,status,workers}: dense_writes(~Element,geometry,first,rows,channel,tree)
    case BandedContent{bands,+source_geometry,shape,fill,status,workers}:
      BandRange.content_writes(~Element,BandRange.Request{source_geometry,first,rows,BandConcat.channel_offset(source_geometry,channel,rows)},bands)

def row_model(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+channel: U32,+input: TensorContent<Element>,output: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Buffer.Content{tree,length} = output
  Buffer.Content{Writes.pure(Element,row_writes(~Element,geometry,first,rows,channel,input),tree),length}

def row_contents(~Element: Data,owners: Tensors.Tensor<Element> & Storage.Buffer<Element>) -> TensorContent<Element> & Buffer.Content<Element>:
  (source,output) = owners
  (tensor_contents(~Element,source),Buffer.contents(~Element,output))

def banded_row_content(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,content: BandRange.Content<Element>) -> TensorContent<Element> & Buffer.Content<Element>:
  BandRange.Content{source,output} = content
  (BandedContent{source,geometry,shape,fill,status,workers},output)

def banded_row_projection(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  owners: Bands.Bands(~Element) & Storage.Buffer<Element>) ->
  {row_contents(~Element,Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Tensors.Tensor<Element> & Storage.Buffer<Element>),owners,
    source => output => (Tensors.Banded{source,geometry,shape,fill,status,workers},output))) ==
    banded_row_content(~Element,geometry,shape,fill,status,workers,BandRange.contents(~Element,owners)) : TensorContent<Element> & Buffer.Content<Element>}:
  (source,output) = owners
  {==}

def banded_written_projection(~Element: Data,+external_geometry: Bands.Geometry,+geometry: Bands.Geometry,+first: U32,+rows: U32,+channel: U32,
  +source: Model.Content<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,output: Buffer.Content<Element>) ->
  {banded_row_content(~Element,geometry,shape,fill,status,workers,BandRange.written(~Element,BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,channel,rows)},source,output)) ==
    (BandedContent{source,geometry,shape,fill,status,workers},row_model(~Element,external_geometry,first,rows,channel,BandedContent{source,geometry,shape,fill,status,workers},output)) : TensorContent<Element> & Buffer.Content<Element>}:
  Buffer.Content{tree,length} = output
  {==}

def banded_rows_packed(~Element: Data,+external_geometry: Bands.Geometry,+geometry: Bands.Geometry,+first: U32,+rows: U32,+channel: U32,
  +source: Model.Model<Element>,+output: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32) ->
  {row_contents(~Element,BandConcat.rows(~Element,external_geometry,first,rows,channel,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers},Buffer.pack(~Element,output))) ==
    (tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers}),
      row_model(~Element,external_geometry,first,rows,channel,tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers}),Buffer.content(~Element,output))) : TensorContent<Element> & Buffer.Content<Element>}:
  +request = {BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,channel,rows)} : BandRange.Request}
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,source)),Model.content(~Element,source),Model.packed_contents(~Element,source)) :
    {row_contents(~Element,BandConcat.rows(~Element,external_geometry,first,rows,channel,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers},Buffer.pack(~Element,output))) ==
      (BandedContent{_,geometry,shape,fill,status,workers},row_model(~Element,external_geometry,first,rows,channel,BandedContent{_,geometry,shape,fill,status,workers},Buffer.content(~Element,output))) : TensorContent<Element> & Buffer.Content<Element>}
  %banded_written_projection(~Element,external_geometry,geometry,first,rows,channel,Model.content(~Element,source),shape,fill,status,workers,Buffer.content(~Element,output)) :
    {row_contents(~Element,BandConcat.rows(~Element,external_geometry,first,rows,channel,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers},Buffer.pack(~Element,output))) == _ : TensorContent<Element> & Buffer.Content<Element>}
  Equal.trans(TensorContent<Element> & Buffer.Content<Element>,
    row_contents(~Element,BandConcat.rows(~Element,external_geometry,first,rows,channel,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers},Buffer.pack(~Element,output))),
    banded_row_content(~Element,geometry,shape,fill,status,workers,BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Buffer.pack(~Element,output)))),
    banded_row_content(~Element,geometry,shape,fill,status,workers,BandRange.written(~Element,request,Model.content(~Element,source),Buffer.content(~Element,output))),
    banded_row_projection(~Element,geometry,shape,fill,status,workers,BandRange.run(~Element,request,Model.pack(~Element,source),Buffer.pack(~Element,output))),
    Equal.cong(BandRange.Content<Element>,TensorContent<Element> & Buffer.Content<Element>,content => banded_row_content(~Element,geometry,shape,fill,status,workers,content),
      BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Buffer.pack(~Element,output))),BandRange.written(~Element,request,Model.content(~Element,source),Buffer.content(~Element,output)),BandRange.canonical_range(~Element,request,source,output)))

def dense_row_content(~Element: Data,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,owners: StorageCopyLaws.Owners<Element>) -> TensorContent<Element> & Buffer.Content<Element>:
  StorageCopyLaws.Owners{input,input_length,output,output_length} = owners
  (DenseContent{Buffer.Content{input,input_length},shape,fill,status,workers},Buffer.Content{output,output_length})

def dense_row_projection(~Element: Data,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {row_contents(~Element,Storage.bind_pair(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~(Tensors.Tensor<Element> & Storage.Buffer<Element>),owners,
    source => output => (Tensors.Tensor{source,shape,fill,status,workers},output))) == dense_row_content(~Element,shape,fill,status,workers,StorageCopyLaws.observe(~Element,owners)) : TensorContent<Element> & Buffer.Content<Element>}:
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate}) = owners
  {==}

def dense_rows_packed(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+channel: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,
  +shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32) ->
  {row_contents(~Element,BandConcat.rows(~Element,geometry,first,rows,channel,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,status,workers},Buffer.pack(~Element,output))) ==
    (tensor_contents(~Element,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,status,workers}),row_model(~Element,geometry,first,rows,channel,tensor_contents(~Element,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,status,workers}),Buffer.contents(~Element,Buffer.pack(~Element,output)))) : TensorContent<Element> & Buffer.Content<Element>}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Buffer.Model{+source,+input_length,+input_certificate} = input
  Buffer.Model{+target,+output_length,+output_certificate} = output
  +count = (rows * width : U32)
  +source_stride = (height * width : U32)
  +destination_stride = (rows * width : U32)
  +source_start = ((first / height * channels * height + first % height) * width : U32)
  +destination_start = (channel * rows * width : U32)
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.content(~Element,input),Buffer.packed_contents(~Element,input)) :
    {row_contents(~Element,BandConcat.rows(~Element,geometry,first,rows,channel,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,status,workers},Buffer.pack(~Element,output))) ==
      (DenseContent{_,shape,fill,status,workers},row_model(~Element,geometry,first,rows,channel,DenseContent{_,shape,fill,status,workers},Buffer.contents(~Element,Buffer.pack(~Element,output)))) : TensorContent<Element> & Buffer.Content<Element>}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)) :
    {row_contents(~Element,BandConcat.rows(~Element,geometry,first,rows,channel,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,status,workers},Buffer.pack(~Element,output))) ==
      (DenseContent{Buffer.content(~Element,input),shape,fill,status,workers},row_model(~Element,geometry,first,rows,channel,DenseContent{Buffer.content(~Element,input),shape,fill,status,workers},_)) : TensorContent<Element> & Buffer.Content<Element>}
  Equal.trans(TensorContent<Element> & Buffer.Content<Element>,
    row_contents(~Element,BandConcat.rows(~Element,geometry,first,rows,channel,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,status,workers},Buffer.pack(~Element,output))),
    dense_row_content(~Element,shape,fill,status,workers,StorageCopyLaws.observe(~Element,BandConcat.dense_rows(~Element,geometry,first,rows,channel,Buffer.pack(~Element,input),Buffer.pack(~Element,output)))),
    dense_row_content(~Element,shape,fill,status,workers,StorageCopyLaws.Owners{source,input_length,Writes.pure(Element,dense_writes(~Element,geometry,first,rows,channel,source),target),output_length}),
    dense_row_projection(~Element,shape,fill,status,workers,BandConcat.dense_rows(~Element,geometry,first,rows,channel,Buffer.pack(~Element,input),Buffer.pack(~Element,output))),
    Equal.cong(StorageCopyLaws.Owners<Element>,TensorContent<Element> & Buffer.Content<Element>,owners => dense_row_content(~Element,shape,fill,status,workers,owners),
      StorageCopyLaws.observe(~Element,BandConcat.dense_rows(~Element,geometry,first,rows,channel,Buffer.pack(~Element,input),Buffer.pack(~Element,output))),StorageCopyLaws.Owners{source,input_length,Writes.pure(Element,dense_writes(~Element,geometry,first,rows,channel,source),target),output_length},
      BandCopy.packed_writes(~Element,U32.to_nat(channels),0,count,source_stride,destination_stride,source_start,destination_start,input,output)))

def dense_rows_observed(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+channel: U32,-input: Storage.Buffer<Element>,-output: Storage.Buffer<Element>,
  +shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,input_observation: Buffer.Observation<Element,input>,output_observation: Buffer.Observation<Element,output>) ->
  {row_contents(~Element,BandConcat.rows(~Element,geometry,first,rows,channel,Tensors.Tensor{input,shape,fill,status,workers},output)) ==
    (tensor_contents(~Element,Tensors.Tensor{input,shape,fill,status,workers}),row_model(~Element,geometry,first,rows,channel,tensor_contents(~Element,Tensors.Tensor{input,shape,fill,status,workers}),Buffer.contents(~Element,output))) : TensorContent<Element> & Buffer.Content<Element>}:
  Buffer.Observation{+source,source_equation} = input_observation
  Buffer.Observation{+target,target_equation} = output_observation
  %source_equation : {row_contents(~Element,BandConcat.rows(~Element,geometry,first,rows,channel,Tensors.Tensor{_,shape,fill,status,workers},output)) ==
    (tensor_contents(~Element,Tensors.Tensor{_,shape,fill,status,workers}),row_model(~Element,geometry,first,rows,channel,tensor_contents(~Element,Tensors.Tensor{_,shape,fill,status,workers}),Buffer.contents(~Element,output))) : TensorContent<Element> & Buffer.Content<Element>}
  %target_equation : {row_contents(~Element,BandConcat.rows(~Element,geometry,first,rows,channel,Tensors.Tensor{Buffer.pack(~Element,source),shape,fill,status,workers},_)) ==
    (tensor_contents(~Element,Tensors.Tensor{Buffer.pack(~Element,source),shape,fill,status,workers}),row_model(~Element,geometry,first,rows,channel,tensor_contents(~Element,Tensors.Tensor{Buffer.pack(~Element,source),shape,fill,status,workers}),Buffer.contents(~Element,_))) : TensorContent<Element> & Buffer.Content<Element>}
  dense_rows_packed(~Element,geometry,first,rows,channel,source,target,shape,fill,status,workers)

def dense_rows(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+channel: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>,
  +shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32) ->
  {row_contents(~Element,BandConcat.rows(~Element,geometry,first,rows,channel,Tensors.Tensor{input,shape,fill,status,workers},output)) ==
    (tensor_contents(~Element,Tensors.Tensor{input,shape,fill,status,workers}),row_model(~Element,geometry,first,rows,channel,tensor_contents(~Element,Tensors.Tensor{input,shape,fill,status,workers}),Buffer.contents(~Element,output))) : TensorContent<Element> & Buffer.Content<Element>}:
  dense_rows_observed(~Element,geometry,first,rows,channel,input,output,shape,fill,status,workers,Buffer.observe(~Element,input),Buffer.observe(~Element,output))

def observed_banded_rows(~Element: Data,+geometry: Bands.Geometry,+first: U32,+count: U32,+channel: U32,-bands: Bands.Bands(~Element),+source_geometry: Bands.Geometry,
  +shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,-output: Storage.Buffer<Element>,observation: Model.Observation<Element,bands>,output_observation: Buffer.Observation<Element,output>) ->
  {row_contents(~Element,BandConcat.rows(~Element,geometry,first,count,channel,Tensors.Banded{bands,source_geometry,shape,fill,status,workers},output)) ==
    (tensor_contents(~Element,Tensors.Banded{bands,source_geometry,shape,fill,status,workers}),row_model(~Element,geometry,first,count,channel,tensor_contents(~Element,Tensors.Banded{bands,source_geometry,shape,fill,status,workers}),Buffer.contents(~Element,output))) : TensorContent<Element> & Buffer.Content<Element>}:
  Model.Observation{+source,equation} = observation
  Buffer.Observation{+target,output_equation} = output_observation
  %equation : {row_contents(~Element,BandConcat.rows(~Element,geometry,first,count,channel,Tensors.Banded{_,source_geometry,shape,fill,status,workers},output)) ==
    (tensor_contents(~Element,Tensors.Banded{_,source_geometry,shape,fill,status,workers}),row_model(~Element,geometry,first,count,channel,tensor_contents(~Element,Tensors.Banded{_,source_geometry,shape,fill,status,workers}),Buffer.contents(~Element,output))) : TensorContent<Element> & Buffer.Content<Element>}
  %output_equation : {row_contents(~Element,BandConcat.rows(~Element,geometry,first,count,channel,Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,workers},_)) ==
    (tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,workers}),row_model(~Element,geometry,first,count,channel,tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,workers}),Buffer.contents(~Element,_))) : TensorContent<Element> & Buffer.Content<Element>}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,target)),Buffer.content(~Element,target),Buffer.packed_contents(~Element,target)) :
    {row_contents(~Element,BandConcat.rows(~Element,geometry,first,count,channel,Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,workers},Buffer.pack(~Element,target))) ==
      (tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,workers}),row_model(~Element,geometry,first,count,channel,tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,workers}),_)) : TensorContent<Element> & Buffer.Content<Element>}
  banded_rows_packed(~Element,geometry,source_geometry,first,count,channel,source,target,shape,fill,status,workers)

def rows(~Element: Data,+geometry: Bands.Geometry,+first: U32,+count: U32,+channel: U32,input: Tensors.Tensor<Element>,output: Storage.Buffer<Element>) ->
  {row_contents(~Element,BandConcat.rows(~Element,geometry,first,count,channel,input,output)) ==
    (tensor_contents(~Element,input),row_model(~Element,geometry,first,count,channel,tensor_contents(~Element,input),Buffer.contents(~Element,output))) : TensorContent<Element> & Buffer.Content<Element>}:
  match input:
    case Tensors.Tensor{input,shape,fill,status,workers}: dense_rows(~Element,geometry,first,count,channel,input,output,shape,fill,status,workers)
    case Tensors.Banded{bands,+source_geometry,+shape,+fill,+status,+workers}:
      observed_banded_rows(~Element,geometry,first,count,channel,bands,source_geometry,shape,fill,status,workers,output,Model.observe(~Element,bands),Buffer.observe(~Element,output))

def tree_model(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+input: TensorContent<Element>,output: Model.Content<Element>) -> Model.Content<Element>:
  match output:
    case Model.RowBand{buffer,+first,+rows}: Model.RowBand{row_model(~Element,geometry,first,rows,channel,input,buffer),first,rows}
    case Model.Joined{left,right}: Model.Joined{tree_model(~Element,geometry,channel,input,left),tree_model(~Element,geometry,channel,input,right)}

def tree_contents(~Element: Data,owners: Tensors.Tensor<Element> & Bands.Bands(~Element)) -> TensorContent<Element> & Model.Content<Element>:
  (source,output) = owners
  (tensor_contents(~Element,source),Model.contents(~Element,output))

def leaf_row_content(~Element: Data,+first: U32,+rows: U32,owners: TensorContent<Element> & Buffer.Content<Element>) -> TensorContent<Element> & Model.Content<Element>:
  (source,output) = owners
  (source,Model.RowBand{output,first,rows})

def leaf_row_projection(~Element: Data,+first: U32,+rows: U32,owners: Tensors.Tensor<Element> & Storage.Buffer<Element>) ->
  {tree_contents(~Element,Storage.bind_pair(~Tensors.Tensor<Element>,~Storage.Buffer<Element>,~(Tensors.Tensor<Element> & Bands.Bands(~Element)),owners,
    source => output => (source,Partition.Leaf{output,first,rows,Unit{}}))) == leaf_row_content(~Element,first,rows,row_contents(~Element,owners)) : TensorContent<Element> & Model.Content<Element>}:
  (source,output) = owners
  {==}

def joined_native(~Element: Data,left: Bands.Bands(~Element),owners: Tensors.Tensor<Element> & Bands.Bands(~Element)) -> Tensors.Tensor<Element> & Bands.Bands(~Element):
  Storage.bind_pair(~Tensors.Tensor<Element>,~Bands.Bands(~Element),~(Tensors.Tensor<Element> & Bands.Bands(~Element)),owners,source => right => (source,Partition.Fork{left,right}))

def joined_content(~Element: Data,+left: Model.Content<Element>,owners: TensorContent<Element> & Model.Content<Element>) -> TensorContent<Element> & Model.Content<Element>:
  (source,right) = owners
  (source,Model.Joined{left,right})

def joined_projection(~Element: Data,left: Bands.Bands(~Element),owners: Tensors.Tensor<Element> & Bands.Bands(~Element)) ->
  {tree_contents(~Element,joined_native(~Element,left,owners)) == joined_content(~Element,Model.contents(~Element,left),tree_contents(~Element,owners)) : TensorContent<Element> & Model.Content<Element>}:
  (source,right) = owners
  {==}

def continued_tree(~Element: Data,+geometry: Bands.Geometry,+channel: U32,right: Bands.Bands(~Element),owners: Tensors.Tensor<Element> & Bands.Bands(~Element)) -> Tensors.Tensor<Element> & Bands.Bands(~Element):
  Storage.bind_pair(~Tensors.Tensor<Element>,~Bands.Bands(~Element),~(Tensors.Tensor<Element> & Bands.Bands(~Element)),owners,
    source => left => joined_native(~Element,left,BandConcat.append_tree(~Element,right,geometry,channel,source)))

def continued_model(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+right: Model.Content<Element>,owners: TensorContent<Element> & Model.Content<Element>) -> TensorContent<Element> & Model.Content<Element>:
  (+source,left) = owners
  (source,Model.Joined{left,tree_model(~Element,geometry,channel,source,right)})

def continued_projection_packed(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+right: Model.Model<Element>,+source: TensorOwnerModel<Element>,+left: Model.Model<Element>,
  rule: @+source: TensorOwnerModel<Element> ->
    {tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,right),geometry,channel,tensor_owner_pack(~Element,source))) ==
      (tensor_contents(~Element,tensor_owner_pack(~Element,source)),tree_model(~Element,geometry,channel,tensor_contents(~Element,tensor_owner_pack(~Element,source)),Model.contents(~Element,Model.pack(~Element,right)))) : TensorContent<Element> & Model.Content<Element>}) ->
  {tree_contents(~Element,continued_tree(~Element,geometry,channel,Model.pack(~Element,right),(tensor_owner_pack(~Element,source),Model.pack(~Element,left)))) ==
    continued_model(~Element,geometry,channel,Model.contents(~Element,Model.pack(~Element,right)),tree_contents(~Element,(tensor_owner_pack(~Element,source),Model.pack(~Element,left)))) : TensorContent<Element> & Model.Content<Element>}:
  Equal.trans(TensorContent<Element> & Model.Content<Element>,
    tree_contents(~Element,joined_native(~Element,Model.pack(~Element,left),BandConcat.append_tree(~Element,Model.pack(~Element,right),geometry,channel,tensor_owner_pack(~Element,source)))),
    joined_content(~Element,Model.contents(~Element,Model.pack(~Element,left)),tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,right),geometry,channel,tensor_owner_pack(~Element,source)))),
    joined_content(~Element,Model.contents(~Element,Model.pack(~Element,left)),(tensor_contents(~Element,tensor_owner_pack(~Element,source)),tree_model(~Element,geometry,channel,tensor_contents(~Element,tensor_owner_pack(~Element,source)),Model.contents(~Element,Model.pack(~Element,right))))),
    joined_projection(~Element,Model.pack(~Element,left),BandConcat.append_tree(~Element,Model.pack(~Element,right),geometry,channel,tensor_owner_pack(~Element,source))),
    Equal.cong(TensorContent<Element> & Model.Content<Element>,TensorContent<Element> & Model.Content<Element>,content => joined_content(~Element,Model.contents(~Element,Model.pack(~Element,left)),content),
      tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,right),geometry,channel,tensor_owner_pack(~Element,source))),
      (tensor_contents(~Element,tensor_owner_pack(~Element,source)),tree_model(~Element,geometry,channel,tensor_contents(~Element,tensor_owner_pack(~Element,source)),Model.contents(~Element,Model.pack(~Element,right)))),rule(source)))

def continued_projection_observed(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+right: Model.Model<Element>,-owners: Tensors.Tensor<Element> & Bands.Bands(~Element),
  observation: TreeOwnersObservation<Element,owners>,
  rule: @+source: TensorOwnerModel<Element> ->
    {tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,right),geometry,channel,tensor_owner_pack(~Element,source))) ==
      (tensor_contents(~Element,tensor_owner_pack(~Element,source)),tree_model(~Element,geometry,channel,tensor_contents(~Element,tensor_owner_pack(~Element,source)),Model.contents(~Element,Model.pack(~Element,right)))) : TensorContent<Element> & Model.Content<Element>}) ->
  {tree_contents(~Element,continued_tree(~Element,geometry,channel,Model.pack(~Element,right),owners)) == continued_model(~Element,geometry,channel,Model.contents(~Element,Model.pack(~Element,right)),tree_contents(~Element,owners)) : TensorContent<Element> & Model.Content<Element>}:
  TreeOwnersObservation{+source,+left,equation} = observation
  %equation : {tree_contents(~Element,continued_tree(~Element,geometry,channel,Model.pack(~Element,right),_)) == continued_model(~Element,geometry,channel,Model.contents(~Element,Model.pack(~Element,right)),tree_contents(~Element,_)) : TensorContent<Element> & Model.Content<Element>}
  continued_projection_packed(~Element,geometry,channel,right,source,left,rule)

def continued_projection(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+right: Model.Model<Element>,owners: Tensors.Tensor<Element> & Bands.Bands(~Element),
  rule: @+source: TensorOwnerModel<Element> ->
    {tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,right),geometry,channel,tensor_owner_pack(~Element,source))) ==
      (tensor_contents(~Element,tensor_owner_pack(~Element,source)),tree_model(~Element,geometry,channel,tensor_contents(~Element,tensor_owner_pack(~Element,source)),Model.contents(~Element,Model.pack(~Element,right)))) : TensorContent<Element> & Model.Content<Element>}) ->
  {tree_contents(~Element,continued_tree(~Element,geometry,channel,Model.pack(~Element,right),owners)) == continued_model(~Element,geometry,channel,Model.contents(~Element,Model.pack(~Element,right)),tree_contents(~Element,owners)) : TensorContent<Element> & Model.Content<Element>}:
  continued_projection_observed(~Element,geometry,channel,right,owners,observe_tree_owners(~Element,owners),rule)

def append_tree_packed(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+output: Model.Model<Element>,+input: TensorOwnerModel<Element>) ->
  {tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,output),geometry,channel,tensor_owner_pack(~Element,input))) ==
    (tensor_contents(~Element,tensor_owner_pack(~Element,input)),tree_model(~Element,geometry,channel,tensor_contents(~Element,tensor_owner_pack(~Element,input)),Model.contents(~Element,Model.pack(~Element,output)))) : TensorContent<Element> & Model.Content<Element>}:
  match output:
    case Model.Leaf{+output,+first,+count}:
      Equal.trans(TensorContent<Element> & Model.Content<Element>,
        tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,Model.Leaf{output,first,count}),geometry,channel,tensor_owner_pack(~Element,input))),
        leaf_row_content(~Element,first,count,row_contents(~Element,BandConcat.rows(~Element,geometry,first,count,channel,tensor_owner_pack(~Element,input),Buffer.pack(~Element,output)))),
        leaf_row_content(~Element,first,count,(tensor_contents(~Element,tensor_owner_pack(~Element,input)),row_model(~Element,geometry,first,count,channel,tensor_contents(~Element,tensor_owner_pack(~Element,input)),Buffer.contents(~Element,Buffer.pack(~Element,output))))),
        leaf_row_projection(~Element,first,count,BandConcat.rows(~Element,geometry,first,count,channel,tensor_owner_pack(~Element,input),Buffer.pack(~Element,output))),
        Equal.cong(TensorContent<Element> & Buffer.Content<Element>,TensorContent<Element> & Model.Content<Element>,owners => leaf_row_content(~Element,first,count,owners),
          row_contents(~Element,BandConcat.rows(~Element,geometry,first,count,channel,tensor_owner_pack(~Element,input),Buffer.pack(~Element,output))),
          (tensor_contents(~Element,tensor_owner_pack(~Element,input)),row_model(~Element,geometry,first,count,channel,tensor_contents(~Element,tensor_owner_pack(~Element,input)),Buffer.contents(~Element,Buffer.pack(~Element,output)))),rows(~Element,geometry,first,count,channel,tensor_owner_pack(~Element,input),Buffer.pack(~Element,output))))
    case Model.Fork{+left,+right}:
      Equal.trans(TensorContent<Element> & Model.Content<Element>,
        tree_contents(~Element,continued_tree(~Element,geometry,channel,Model.pack(~Element,right),BandConcat.append_tree(~Element,Model.pack(~Element,left),geometry,channel,tensor_owner_pack(~Element,input)))),
        continued_model(~Element,geometry,channel,Model.contents(~Element,Model.pack(~Element,right)),tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,left),geometry,channel,tensor_owner_pack(~Element,input)))),
        continued_model(~Element,geometry,channel,Model.contents(~Element,Model.pack(~Element,right)),(tensor_contents(~Element,tensor_owner_pack(~Element,input)),tree_model(~Element,geometry,channel,tensor_contents(~Element,tensor_owner_pack(~Element,input)),Model.contents(~Element,Model.pack(~Element,left))))),
        continued_projection(~Element,geometry,channel,right,BandConcat.append_tree(~Element,Model.pack(~Element,left),geometry,channel,tensor_owner_pack(~Element,input)),source => append_tree_packed(~Element,geometry,channel,right,source)),
        Equal.cong(TensorContent<Element> & Model.Content<Element>,TensorContent<Element> & Model.Content<Element>,owners => continued_model(~Element,geometry,channel,Model.contents(~Element,Model.pack(~Element,right)),owners),
          tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,left),geometry,channel,tensor_owner_pack(~Element,input))),
          (tensor_contents(~Element,tensor_owner_pack(~Element,input)),tree_model(~Element,geometry,channel,tensor_contents(~Element,tensor_owner_pack(~Element,input)),Model.contents(~Element,Model.pack(~Element,left)))),append_tree_packed(~Element,geometry,channel,left,input)))

def append_tree_observed(~Element: Data,+geometry: Bands.Geometry,+channel: U32,-output: Bands.Bands(~Element),-input: Tensors.Tensor<Element>,
  output_observation: Model.Observation<Element,output>,input_observation: TensorObservation<Element,input>) ->
  {tree_contents(~Element,BandConcat.append_tree(~Element,output,geometry,channel,input)) ==
    (tensor_contents(~Element,input),tree_model(~Element,geometry,channel,tensor_contents(~Element,input),Model.contents(~Element,output))) : TensorContent<Element> & Model.Content<Element>}:
  Model.Observation{+target,target_equation} = output_observation
  TensorObservation{+source,source_equation} = input_observation
  %target_equation : {tree_contents(~Element,BandConcat.append_tree(~Element,_,geometry,channel,input)) ==
    (tensor_contents(~Element,input),tree_model(~Element,geometry,channel,tensor_contents(~Element,input),Model.contents(~Element,_))) : TensorContent<Element> & Model.Content<Element>}
  %source_equation : {tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,target),geometry,channel,_)) ==
    (tensor_contents(~Element,_),tree_model(~Element,geometry,channel,tensor_contents(~Element,_),Model.contents(~Element,Model.pack(~Element,target)))) : TensorContent<Element> & Model.Content<Element>}
  append_tree_packed(~Element,geometry,channel,target,source)

def append_tree(~Element: Data,+geometry: Bands.Geometry,+channel: U32,output: Bands.Bands(~Element),input: Tensors.Tensor<Element>) ->
  {tree_contents(~Element,BandConcat.append_tree(~Element,output,geometry,channel,input)) ==
    (tensor_contents(~Element,input),tree_model(~Element,geometry,channel,tensor_contents(~Element,input),Model.contents(~Element,output))) : TensorContent<Element> & Model.Content<Element>}:
  append_tree_observed(~Element,geometry,channel,output,input,Model.observe(~Element,output),observe_tensor(~Element,input))

def aligned_model(~Element: Data,+geometry: Bands.Geometry,+channel: U32,input: Model.Content<Element>,output: Model.Content<Element>) -> Model.Content<Element>:
  match input output:
    case Model.RowBand{input,first,rows} Model.RowBand{output,out_first,+out_rows}:
      Model.RowBand{BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(geometry,out_rows),0,BandConcat.channel_offset(geometry,channel,out_rows),BandGather.owners(~Element,input,output))),out_first,out_rows}
    case Model.Joined{left,right} Model.Joined{out_left,out_right}:
      Model.Joined{aligned_model(~Element,geometry,channel,left,out_left),aligned_model(~Element,geometry,channel,right,out_right)}
    case input output: output

def aligned_models(~Element: Data,+geometry: Bands.Geometry,+channel: U32,input: Model.Model<Element>,output: Model.Model<Element>) ->
  {pipeline_model(~Element,geometry,channel,input,output) == aligned_model(~Element,geometry,channel,Model.content(~Element,input),Model.content(~Element,output)) : Model.Content<Element>}:
  match input output:
    case Model.Leaf{input,first,rows} Model.Leaf{output,out_first,out_rows}: {==}
    case Model.Fork{+left,+right} Model.Fork{+out_left,+out_right}:
      %aligned_models(~Element,geometry,channel,left,out_left) :
        {Model.Joined{pipeline_model(~Element,geometry,channel,left,out_left),pipeline_model(~Element,geometry,channel,right,out_right)} == Model.Joined{_,aligned_model(~Element,geometry,channel,Model.content(~Element,right),Model.content(~Element,out_right))} : Model.Content<Element>}
      %aligned_models(~Element,geometry,channel,right,out_right) :
        {Model.Joined{pipeline_model(~Element,geometry,channel,left,out_left),pipeline_model(~Element,geometry,channel,right,out_right)} == Model.Joined{pipeline_model(~Element,geometry,channel,left,out_left),_} : Model.Content<Element>}
      {==}
    case Model.Leaf{input,first,rows} Model.Fork{left,right}: {==}
    case Model.Fork{left,right} Model.Leaf{output,first,rows}: {==}

def execution_output_content(~Element: Data,owners: TensorContent<Element> & Model.Content<Element>) -> Model.Content<Element>:
  (source,output) = owners
  output

def appended_projection(~Element: Data,owners: Tensors.Tensor<Element> & Bands.Bands(~Element)) ->
  {Model.contents(~Element,BandConcat.appended_rows(~Element,owners)) == execution_output_content(~Element,tree_contents(~Element,owners)) : Model.Content<Element>}:
  (source,output) = owners
  {==}

def appended_tree_packed(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+output: Model.Model<Element>,+input: TensorOwnerModel<Element>) ->
  {Model.contents(~Element,BandConcat.appended_rows(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,output),geometry,channel,tensor_owner_pack(~Element,input)))) ==
    tree_model(~Element,geometry,channel,tensor_contents(~Element,tensor_owner_pack(~Element,input)),Model.contents(~Element,Model.pack(~Element,output))) : Model.Content<Element>}:
  Equal.trans(Model.Content<Element>,Model.contents(~Element,BandConcat.appended_rows(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,output),geometry,channel,tensor_owner_pack(~Element,input)))),
    execution_output_content(~Element,tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,output),geometry,channel,tensor_owner_pack(~Element,input)))),
    tree_model(~Element,geometry,channel,tensor_contents(~Element,tensor_owner_pack(~Element,input)),Model.contents(~Element,Model.pack(~Element,output))),
    appended_projection(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,output),geometry,channel,tensor_owner_pack(~Element,input))),
    Equal.cong(TensorContent<Element> & Model.Content<Element>,Model.Content<Element>,owners => execution_output_content(~Element,owners),
      tree_contents(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,output),geometry,channel,tensor_owner_pack(~Element,input))),
      (tensor_contents(~Element,tensor_owner_pack(~Element,input)),tree_model(~Element,geometry,channel,tensor_contents(~Element,tensor_owner_pack(~Element,input)),Model.contents(~Element,Model.pack(~Element,output)))),
      append_tree_packed(~Element,geometry,channel,output,input)))

def appended_tree_observed(~Element: Data,+geometry: Bands.Geometry,+channel: U32,-output: Bands.Bands(~Element),-input: Tensors.Tensor<Element>,
  output_observation: Model.Observation<Element,output>,input_observation: TensorObservation<Element,input>) ->
  {Model.contents(~Element,BandConcat.appended_rows(~Element,BandConcat.append_tree(~Element,output,geometry,channel,input))) ==
    tree_model(~Element,geometry,channel,tensor_contents(~Element,input),Model.contents(~Element,output)) : Model.Content<Element>}:
  Model.Observation{+target,target_equation} = output_observation
  TensorObservation{+source,source_equation} = input_observation
  %target_equation : {Model.contents(~Element,BandConcat.appended_rows(~Element,BandConcat.append_tree(~Element,_,geometry,channel,input))) ==
    tree_model(~Element,geometry,channel,tensor_contents(~Element,input),Model.contents(~Element,_)) : Model.Content<Element>}
  %source_equation : {Model.contents(~Element,BandConcat.appended_rows(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,target),geometry,channel,_))) ==
    tree_model(~Element,geometry,channel,tensor_contents(~Element,_),Model.contents(~Element,Model.pack(~Element,target))) : Model.Content<Element>}
  appended_tree_packed(~Element,geometry,channel,target,source)

def appended_tree(~Element: Data,+geometry: Bands.Geometry,+channel: U32,output: Bands.Bands(~Element),input: Tensors.Tensor<Element>) ->
  {Model.contents(~Element,BandConcat.appended_rows(~Element,BandConcat.append_tree(~Element,output,geometry,channel,input))) == tree_model(~Element,geometry,channel,tensor_contents(~Element,input),Model.contents(~Element,output)) : Model.Content<Element>}:
  appended_tree_observed(~Element,geometry,channel,output,input,Model.observe(~Element,output),observe_tensor(~Element,input))

def selected_model(~Element: Data,same: Bool,+geometry: Bands.Geometry,+channel: U32,+input: Model.Content<Element>,+output: Model.Content<Element>,
  +shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+previous: U32) -> Model.Content<Element>:
  match same:
    case True{}: aligned_model(~Element,geometry,channel,input,output)
    case False{}: tree_model(~Element,geometry,channel,BandedContent{input,geometry,shape,fill,status,previous},output)

def selected_rows(~Element: Data,+same: Bool,+geometry: Bands.Geometry,+channel: U32,+workers: U32,+layout: Bands.Plan,+input: Model.Model<Element>,+output: Model.Model<Element>,
  +shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+previous: U32) ->
  {Model.contents(~Element,BandConcat.selected_rows(~Element,same,geometry,channel,workers,layout,Model.pack(~Element,input),Model.pack(~Element,output),shape,fill,status,previous)) ==
    selected_model(~Element,same,geometry,channel,Model.content(~Element,input),Model.content(~Element,output),shape,fill,status,previous) : Model.Content<Element>}:
  match same:
    case True{}:
      Equal.trans(Model.Content<Element>,Model.contents(~Element,BandConcat.append_aligned(~Element,BandSchedule.parallel(geometry,workers,8.0,0.0,layout),geometry,channel,Model.pack(~Element,input),Model.pack(~Element,output))),
        pipeline_model(~Element,geometry,channel,input,output),aligned_model(~Element,geometry,channel,Model.content(~Element,input),Model.content(~Element,output)),
        append_aligned(~Element,BandSchedule.parallel(geometry,workers,8.0,0.0,layout),geometry,channel,input,output),aligned_models(~Element,geometry,channel,input,output))
    case False{}:
      %Model.packed_contents(~Element,input) :
        {Model.contents(~Element,BandConcat.selected_rows(~Element,False{},geometry,channel,workers,layout,Model.pack(~Element,input),Model.pack(~Element,output),shape,fill,status,previous)) ==
          tree_model(~Element,geometry,channel,BandedContent{_,geometry,shape,fill,status,previous},Model.content(~Element,output)) : Model.Content<Element>}
      %Model.packed_contents(~Element,output) :
        {Model.contents(~Element,BandConcat.selected_rows(~Element,False{},geometry,channel,workers,layout,Model.pack(~Element,input),Model.pack(~Element,output),shape,fill,status,previous)) ==
          tree_model(~Element,geometry,channel,BandedContent{Model.contents(~Element,Model.pack(~Element,input)),geometry,shape,fill,status,previous},_) : Model.Content<Element>}
      appended_tree(~Element,geometry,channel,Model.pack(~Element,output),Tensors.Banded{Model.pack(~Element,input),geometry,shape,fill,status,previous})

def storage_model(~Element: Data,+geometry: Bands.Geometry,+channel: U32,input: TensorContent<Element>,+output: Model.Content<Element>) -> Model.Content<Element>:
  match input:
    case DenseContent{buffer,shape,fill,status,previous}: tree_model(~Element,geometry,channel,DenseContent{buffer,shape,fill,status,previous},output)
    case BandedContent{+bands,source_geometry,shape,fill,status,previous}:
      selected_model(~Element,Bands.same_plan(Model.content_layout(~Element,bands),Model.content_layout(~Element,output)),geometry,channel,bands,output,shape,fill,status,previous)

def storage_banded_packed(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+workers: U32,+input: Model.Model<Element>,+output: Model.Model<Element>,+source_geometry: Bands.Geometry,
  +shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+previous: U32) ->
  {Model.contents(~Element,BandConcat.append_storage(~Element,geometry,channel,workers,Tensors.Banded{Model.pack(~Element,input),source_geometry,shape,fill,status,previous},Model.pack(~Element,output))) ==
    storage_model(~Element,geometry,channel,tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,input),source_geometry,shape,fill,status,previous}),Model.contents(~Element,Model.pack(~Element,output))) : Model.Content<Element>}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,input)),(Model.pack(~Element,input),Model.layout(~Element,input)),Model.plan(~Element,input)) :
    {Model.contents(~Element,BandConcat.aligned_or_rows(~Element,geometry,channel,workers,_,Bands.plan(~Element,Model.pack(~Element,output)),shape,fill,status,previous)) == storage_model(~Element,geometry,channel,tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,input),source_geometry,shape,fill,status,previous}),Model.contents(~Element,Model.pack(~Element,output))) : Model.Content<Element>}
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,output)),(Model.pack(~Element,output),Model.layout(~Element,output)),Model.plan(~Element,output)) :
    {Model.contents(~Element,BandConcat.aligned_or_rows(~Element,geometry,channel,workers,(Model.pack(~Element,input),Model.layout(~Element,input)),_,shape,fill,status,previous)) == storage_model(~Element,geometry,channel,tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,input),source_geometry,shape,fill,status,previous}),Model.contents(~Element,Model.pack(~Element,output))) : Model.Content<Element>}
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,input)),Model.content(~Element,input),Model.packed_contents(~Element,input)) :
    {Model.contents(~Element,BandConcat.selected_rows(~Element,Bands.same_plan(Model.layout(~Element,input),Model.layout(~Element,output)),geometry,channel,workers,Model.layout(~Element,output),Model.pack(~Element,input),Model.pack(~Element,output),shape,fill,status,previous)) == storage_model(~Element,geometry,channel,BandedContent{_,source_geometry,shape,fill,status,previous},Model.contents(~Element,Model.pack(~Element,output))) : Model.Content<Element>}
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,output)),Model.content(~Element,output),Model.packed_contents(~Element,output)) :
    {Model.contents(~Element,BandConcat.selected_rows(~Element,Bands.same_plan(Model.layout(~Element,input),Model.layout(~Element,output)),geometry,channel,workers,Model.layout(~Element,output),Model.pack(~Element,input),Model.pack(~Element,output),shape,fill,status,previous)) == storage_model(~Element,geometry,channel,BandedContent{Model.content(~Element,input),source_geometry,shape,fill,status,previous},_) : Model.Content<Element>}
  %Model.layout_matches_content(~Element,input) :
    {Model.contents(~Element,BandConcat.selected_rows(~Element,Bands.same_plan(Model.layout(~Element,input),Model.layout(~Element,output)),geometry,channel,workers,Model.layout(~Element,output),Model.pack(~Element,input),Model.pack(~Element,output),shape,fill,status,previous)) == selected_model(~Element,Bands.same_plan(_,Model.content_layout(~Element,Model.content(~Element,output))),geometry,channel,Model.content(~Element,input),Model.content(~Element,output),shape,fill,status,previous) : Model.Content<Element>}
  %Model.layout_matches_content(~Element,output) :
    {Model.contents(~Element,BandConcat.selected_rows(~Element,Bands.same_plan(Model.layout(~Element,input),Model.layout(~Element,output)),geometry,channel,workers,Model.layout(~Element,output),Model.pack(~Element,input),Model.pack(~Element,output),shape,fill,status,previous)) == selected_model(~Element,Bands.same_plan(Model.layout(~Element,input),_),geometry,channel,Model.content(~Element,input),Model.content(~Element,output),shape,fill,status,previous) : Model.Content<Element>}
  selected_rows(~Element,Bands.same_plan(Model.layout(~Element,input),Model.layout(~Element,output)),geometry,channel,workers,Model.layout(~Element,output),input,output,shape,fill,status,previous)

def observed_banded_storage(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+workers: U32,-input: Bands.Bands(~Element),-output: Bands.Bands(~Element),+source_geometry: Bands.Geometry,
  +shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+previous: U32,source_observation: Model.Observation<Element,input>,target_observation: Model.Observation<Element,output>) ->
  {Model.contents(~Element,BandConcat.append_storage(~Element,geometry,channel,workers,Tensors.Banded{input,source_geometry,shape,fill,status,previous},output)) ==
    storage_model(~Element,geometry,channel,tensor_contents(~Element,Tensors.Banded{input,source_geometry,shape,fill,status,previous}),Model.contents(~Element,output)) : Model.Content<Element>}:
  Model.Observation{+source,source_equal} = source_observation
  Model.Observation{+target,target_equal} = target_observation
  %source_equal :
    {Model.contents(~Element,BandConcat.append_storage(~Element,geometry,channel,workers,Tensors.Banded{_,source_geometry,shape,fill,status,previous},output)) == storage_model(~Element,geometry,channel,tensor_contents(~Element,Tensors.Banded{_,source_geometry,shape,fill,status,previous}),Model.contents(~Element,output)) : Model.Content<Element>}
  %target_equal :
    {Model.contents(~Element,BandConcat.append_storage(~Element,geometry,channel,workers,Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,previous},_)) == storage_model(~Element,geometry,channel,tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,previous}),Model.contents(~Element,_)) : Model.Content<Element>}
  storage_banded_packed(~Element,geometry,channel,workers,source,target,source_geometry,shape,fill,status,previous)

def append_storage(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+workers: U32,input: Tensors.Tensor<Element>,output: Bands.Bands(~Element)) ->
  {Model.contents(~Element,BandConcat.append_storage(~Element,geometry,channel,workers,input,output)) == storage_model(~Element,geometry,channel,tensor_contents(~Element,input),Model.contents(~Element,output)) : Model.Content<Element>}:
  match input:
    case Tensors.Tensor{storage,shape,fill,status,previous}:
      appended_tree(~Element,geometry,channel,output,Tensors.Tensor{storage,shape,fill,status,previous})
    case Tensors.Banded{input,+source_geometry,+shape,+fill,+status,+previous}:
      observed_banded_storage(~Element,geometry,channel,workers,input,output,source_geometry,shape,fill,status,previous,Model.observe(~Element,input),Model.observe(~Element,output))

type PartContents<-Element: Data> is Data:
  SingleContent{input: TensorContent<Element>}
  ConsContent{input: TensorContent<Element>,rest: PartContents<Element>}

def parts_contents(~Element: Data,parts: Tensors.Parts<Element>) -> PartContents<Element>:
  match parts:
    case Tensors.Single{input}: SingleContent{tensor_contents(~Element,input)}
    case Tensors.Cons{input,rest}: ConsContent{tensor_contents(~Element,input),parts_contents(~Element,rest)}

type PartsOwnerModel<-Element: Data> is Data:
  SingleOwner{input: TensorOwnerModel<Element>}
  ConsOwner{input: TensorOwnerModel<Element>,rest: PartsOwnerModel<Element>}

def parts_owner_pack(~Element: Data,parts: PartsOwnerModel<Element>) -> Tensors.Parts<Element>:
  match parts:
    case SingleOwner{input}: Tensors.Single{tensor_owner_pack(~Element,input)}
    case ConsOwner{input,rest}: Tensors.Cons{tensor_owner_pack(~Element,input),parts_owner_pack(~Element,rest)}

type PartsObservation<-Element: Data,-parts: Tensors.Parts<Element>> is Data:
  PartsObservation{model: PartsOwnerModel<Element>,equation: {parts_owner_pack(~Element,model) == parts : Tensors.Parts<Element>}}

def parts_observed_single(~Element: Data,-input: Tensors.Tensor<Element>,observation: TensorObservation<Element,input>) -> PartsObservation<Element,Tensors.Single{input}>:
  TensorObservation{+model,equation} = observation
  PartsObservation{SingleOwner{model},Equal.cong(Tensors.Tensor<Element>,Tensors.Parts<Element>,source => Tensors.Single{source},tensor_owner_pack(~Element,model),input,equation)}

def parts_observed_cons(~Element: Data,-input: Tensors.Tensor<Element>,-rest: Tensors.Parts<Element>,
  input_observation: TensorObservation<Element,input>,rest_observation: PartsObservation<Element,rest>) -> PartsObservation<Element,Tensors.Cons{input,rest}>:
  TensorObservation{+source,source_equation} = input_observation
  PartsObservation{+remaining,rest_equation} = rest_observation
  PartsObservation{ConsOwner{source,remaining},
    %source_equation : {Tensors.Cons{tensor_owner_pack(~Element,source),parts_owner_pack(~Element,remaining)} == Tensors.Cons{_,rest} : Tensors.Parts<Element>}
    %rest_equation : {Tensors.Cons{tensor_owner_pack(~Element,source),parts_owner_pack(~Element,remaining)} == Tensors.Cons{tensor_owner_pack(~Element,source),_} : Tensors.Parts<Element>}
    {==}}

def observe_parts(~Element: Data,parts: Tensors.Parts<Element>) -> PartsObservation<Element,parts>:
  match parts:
    case Tensors.Single{input}: parts_observed_single(~Element,input,observe_tensor(~Element,input))
    case Tensors.Cons{input,rest}: parts_observed_cons(~Element,input,rest,observe_tensor(~Element,input),observe_parts(~Element,rest))

type BandStateModel<-Element: Data> is Data:
  BandStateModel{output: Model.Model<Element>,cursor: U32}

def band_state_pack(~Element: Data,state: BandStateModel<Element>) -> Bands.Bands(~Element) & U32:
  BandStateModel{output,cursor} = state
  (Model.pack(~Element,output),cursor)

type BandStateObservation<-Element: Data,-state: Bands.Bands(~Element) & U32> is Data:
  BandStateObservation{model: BandStateModel<Element>,equation: {band_state_pack(~Element,model) == state : Bands.Bands(~Element) & U32}}

def band_state_observed(~Element: Data,-output: Bands.Bands(~Element),+cursor: U32,observation: Model.Observation<Element,output>) -> BandStateObservation<Element,(output,cursor)>:
  Model.Observation{+model,equation} = observation
  BandStateObservation{BandStateModel{model,cursor},Equal.cong(Bands.Bands(~Element),Bands.Bands(~Element) & U32,owner => (owner,cursor),Model.pack(~Element,model),output,equation)}

def observe_band_state(~Element: Data,state: Bands.Bands(~Element) & U32) -> BandStateObservation<Element,state>:
  (output,+cursor) = state
  band_state_observed(~Element,output,cursor,Model.observe(~Element,output))

# The pure content/cursor state is reusable Data. The native owner pair
# remains affine and is observed once before entering this model.
def ContentState(~Element: Data) -> Data:
  Sigma<&2,&2,Model.Content<Element>,_ => U32>

def state_contents(~Element: Data,state: Bands.Bands(~Element) & U32) -> ContentState(~Element):
  (output,channel) = state
  (Model.contents(~Element,output),channel)

def state_output(~Element: Data,state: ContentState(~Element)) -> Model.Content<Element>:
  (output,channel) = state
  output

def state_channel(~Element: Data,state: ContentState(~Element)) -> U32:
  (output,channel) = state
  channel

def input_model(~Element: Data,geometry: Bands.Geometry,+channel: U32,+count: U32,input: TensorContent<Element>,output: Model.Content<Element>) -> ContentState(~Element):
  Bands.Geometry{images,channels,height,width} = geometry
  (storage_model(~Element,Bands.Geometry{images,count,height,width},channel,input,output),(channel + count : U32))

def part_model(~Element: Data,+geometry: Bands.Geometry,+input: TensorContent<Element>,state: ContentState(~Element)) -> ContentState(~Element):
  (output,channel) = state
  input_model(~Element,geometry,channel,BandConcat.channels(tensor_shape(~Element,input)),input,output)

def append_input(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+count: U32,+workers: U32,input: Tensors.Tensor<Element>,output: Bands.Bands(~Element)) ->
  {state_contents(~Element,BandConcat.append_input(~Element,geometry,channel,count,workers,input,output)) == input_model(~Element,geometry,channel,count,tensor_contents(~Element,input),Model.contents(~Element,output)) : ContentState(~Element)}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Equal.cong(Model.Content<Element>,ContentState(~Element),content => (content,(channel + count : U32)),
    Model.contents(~Element,BandConcat.append_storage(~Element,Bands.Geometry{images,count,height,width},channel,workers,input,output)),storage_model(~Element,Bands.Geometry{images,count,height,width},channel,tensor_contents(~Element,input),Model.contents(~Element,output)),
    append_storage(~Element,Bands.Geometry{images,count,height,width},channel,workers,input,output))

def append_part(~Element: Data,+geometry: Bands.Geometry,+workers: U32,input: Tensors.Tensor<Element>,state: Bands.Bands(~Element) & U32) ->
  {state_contents(~Element,BandConcat.append_part(~Element,geometry,workers,input,state)) == part_model(~Element,geometry,tensor_contents(~Element,input),state_contents(~Element,state)) : ContentState(~Element)}:
  match input:
    case Tensors.Tensor{storage,+shape,fill,status,previous}:
      (output,+channel) = state
      append_input(~Element,geometry,channel,BandConcat.channels(shape),workers,Tensors.Tensor{storage,shape,fill,status,previous},output)
    case Tensors.Banded{bands,source_geometry,+shape,fill,status,previous}:
      (output,+channel) = state
      append_input(~Element,geometry,channel,BandConcat.channels(shape),workers,Tensors.Banded{bands,source_geometry,shape,fill,status,previous},output)

def parts_model(~Element: Data,parts: PartContents<Element>,+geometry: Bands.Geometry,state: ContentState(~Element)) -> ContentState(~Element):
  match parts:
    case SingleContent{input}: part_model(~Element,geometry,input,state)
    case ConsContent{input,rest}: parts_model(~Element,rest,geometry,part_model(~Element,geometry,input,state))

def native_output(~Element: Data,state: Bands.Bands(~Element) & U32) ->
  {Model.contents(~Element,BandConcat.output(~Element,state)) == state_output(~Element,state_contents(~Element,state)) : Model.Content<Element>}:
  (output,channel) = state
  {==}

def append_parts_continued_observed(~Element: Data,+parts: PartsOwnerModel<Element>,+geometry: Bands.Geometry,+workers: U32,-state: Bands.Bands(~Element) & U32,
  observation: BandStateObservation<Element,state>,
  rule: @+state: BandStateModel<Element> ->
    {Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,parts),geometry,workers,band_state_pack(~Element,state))) ==
      state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,parts)),geometry,state_contents(~Element,band_state_pack(~Element,state)))) : Model.Content<Element>}) ->
  {Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,parts),geometry,workers,state)) ==
    state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,parts)),geometry,state_contents(~Element,state))) : Model.Content<Element>}:
  BandStateObservation{+model,equation} = observation
  %equation : {Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,parts),geometry,workers,_)) ==
    state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,parts)),geometry,state_contents(~Element,_))) : Model.Content<Element>}
  rule(model)

def append_parts_continued(~Element: Data,+parts: PartsOwnerModel<Element>,+geometry: Bands.Geometry,+workers: U32,state: Bands.Bands(~Element) & U32,
  rule: @+state: BandStateModel<Element> ->
    {Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,parts),geometry,workers,band_state_pack(~Element,state))) ==
      state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,parts)),geometry,state_contents(~Element,band_state_pack(~Element,state)))) : Model.Content<Element>}) ->
  {Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,parts),geometry,workers,state)) ==
    state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,parts)),geometry,state_contents(~Element,state))) : Model.Content<Element>}:
  append_parts_continued_observed(~Element,parts,geometry,workers,state,observe_band_state(~Element,state),rule)

def append_parts_packed(~Element: Data,parts: PartsOwnerModel<Element>,+geometry: Bands.Geometry,+workers: U32,+state: BandStateModel<Element>) ->
  {Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,parts),geometry,workers,band_state_pack(~Element,state))) ==
    state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,parts)),geometry,state_contents(~Element,band_state_pack(~Element,state)))) : Model.Content<Element>}:
  match parts:
    case SingleOwner{+input}:
      Equal.trans(Model.Content<Element>,Model.contents(~Element,BandConcat.output(~Element,BandConcat.append_part(~Element,geometry,workers,tensor_owner_pack(~Element,input),band_state_pack(~Element,state)))),
        state_output(~Element,state_contents(~Element,BandConcat.append_part(~Element,geometry,workers,tensor_owner_pack(~Element,input),band_state_pack(~Element,state)))),
        state_output(~Element,part_model(~Element,geometry,tensor_contents(~Element,tensor_owner_pack(~Element,input)),state_contents(~Element,band_state_pack(~Element,state)))),
        native_output(~Element,BandConcat.append_part(~Element,geometry,workers,tensor_owner_pack(~Element,input),band_state_pack(~Element,state))),
        Equal.cong(ContentState(~Element),Model.Content<Element>,content => state_output(~Element,content),
          state_contents(~Element,BandConcat.append_part(~Element,geometry,workers,tensor_owner_pack(~Element,input),band_state_pack(~Element,state))),
          part_model(~Element,geometry,tensor_contents(~Element,tensor_owner_pack(~Element,input)),state_contents(~Element,band_state_pack(~Element,state))),append_part(~Element,geometry,workers,tensor_owner_pack(~Element,input),band_state_pack(~Element,state))))
    case ConsOwner{+input,+rest}:
      Equal.trans(Model.Content<Element>,Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,rest),geometry,workers,BandConcat.append_part(~Element,geometry,workers,tensor_owner_pack(~Element,input),band_state_pack(~Element,state)))),
        state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,rest)),geometry,state_contents(~Element,BandConcat.append_part(~Element,geometry,workers,tensor_owner_pack(~Element,input),band_state_pack(~Element,state))))),
        state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,rest)),geometry,part_model(~Element,geometry,tensor_contents(~Element,tensor_owner_pack(~Element,input)),state_contents(~Element,band_state_pack(~Element,state))))),
        append_parts_continued(~Element,rest,geometry,workers,BandConcat.append_part(~Element,geometry,workers,tensor_owner_pack(~Element,input),band_state_pack(~Element,state)),next => append_parts_packed(~Element,rest,geometry,workers,next)),
        Equal.cong(ContentState(~Element),Model.Content<Element>,content => state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,rest)),geometry,content)),
          state_contents(~Element,BandConcat.append_part(~Element,geometry,workers,tensor_owner_pack(~Element,input),band_state_pack(~Element,state))),
          part_model(~Element,geometry,tensor_contents(~Element,tensor_owner_pack(~Element,input)),state_contents(~Element,band_state_pack(~Element,state))),append_part(~Element,geometry,workers,tensor_owner_pack(~Element,input),band_state_pack(~Element,state))))

def append_parts_observed(~Element: Data,-parts: Tensors.Parts<Element>,+geometry: Bands.Geometry,+workers: U32,-state: Bands.Bands(~Element) & U32,
  parts_observation: PartsObservation<Element,parts>,state_observation: BandStateObservation<Element,state>) ->
  {Model.contents(~Element,BandConcat.append_parts(~Element,parts,geometry,workers,state)) == state_output(~Element,parts_model(~Element,parts_contents(~Element,parts),geometry,state_contents(~Element,state))) : Model.Content<Element>}:
  PartsObservation{+sources,parts_equation} = parts_observation
  BandStateObservation{+target,state_equation} = state_observation
  %parts_equation : {Model.contents(~Element,BandConcat.append_parts(~Element,_,geometry,workers,state)) == state_output(~Element,parts_model(~Element,parts_contents(~Element,_),geometry,state_contents(~Element,state))) : Model.Content<Element>}
  %state_equation : {Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,sources),geometry,workers,_)) == state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,sources)),geometry,state_contents(~Element,_))) : Model.Content<Element>}
  append_parts_packed(~Element,sources,geometry,workers,target)

def append_parts(~Element: Data,parts: Tensors.Parts<Element>,+geometry: Bands.Geometry,+workers: U32,state: Bands.Bands(~Element) & U32) ->
  {Model.contents(~Element,BandConcat.append_parts(~Element,parts,geometry,workers,state)) == state_output(~Element,parts_model(~Element,parts_contents(~Element,parts),geometry,state_contents(~Element,state))) : Model.Content<Element>}:
  append_parts_observed(~Element,parts,geometry,workers,state,observe_parts(~Element,parts),observe_band_state(~Element,state))

# The cursor is derived by the actual left-to-right update. This identity
# is modular; recovering its unbounded total requires the public survey's
# successful no-overflow guard, which is a separate producer obligation.
def channel_cursor(~Element: Data,parts: PartContents<Element>,+channel: U32) -> U32:
  match parts:
    case SingleContent{input}: (channel + BandConcat.channels(tensor_shape(~Element,input)) : U32)
    case ConsContent{input,rest}: channel_cursor(~Element,rest,(channel + BandConcat.channels(tensor_shape(~Element,input)) : U32))

def total_channels(~Element: Data,parts: PartContents<Element>) -> Nat:
  match parts:
    case SingleContent{input}: U32.to_nat(BandConcat.channels(tensor_shape(~Element,input)))
    case ConsContent{input,rest}: Nat.add(U32.to_nat(BandConcat.channels(tensor_shape(~Element,input))),total_channels(~Element,rest))

def parts_channel(~Element: Data,parts: PartContents<Element>,+geometry: Bands.Geometry,+state: ContentState(~Element)) ->
  {state_channel(~Element,parts_model(~Element,parts,geometry,state)) == channel_cursor(~Element,parts,state_channel(~Element,state)) : U32}:
  match parts:
    case SingleContent{input}:
      Bands.Geometry{images,channels,height,width} = geometry
      (output,channel) = state
      {==}
    case ConsContent{+input,+rest}:
      Bands.Geometry{+images,+channels,+height,+width} = geometry
      (output,+channel) = state
      parts_channel(~Element,rest,geometry,part_model(~Element,geometry,input,state))

def encoded_channel_add(+position: Nat,+count: U32) ->
  {(U32.from_nat(position) + count : U32) == U32.from_nat(Nat.add(position,U32.to_nat(count))) : U32}:
  %Arithmetic.native_roundtrip(count) : {(U32.from_nat(position) + _ : U32) == U32.from_nat(Nat.add(position,U32.to_nat(count))) : U32}
  WordArithmetic.native_add(position,U32.to_nat(count))

def cursor_from_encoded(~Element: Data,parts: PartContents<Element>,+position: Nat) ->
  {channel_cursor(~Element,parts,U32.from_nat(position)) == U32.from_nat(Nat.add(position,total_channels(~Element,parts))) : U32}:
  match parts:
    case SingleContent{+input}: encoded_channel_add(position,BandConcat.channels(tensor_shape(~Element,input)))
    case ConsContent{+input,+rest}:
      +count = BandConcat.channels(tensor_shape(~Element,input))
      %Equal.sym(U32,(U32.from_nat(position) + count : U32),U32.from_nat(Nat.add(position,U32.to_nat(count))),encoded_channel_add(position,count)) :
        {channel_cursor(~Element,rest,_) == U32.from_nat(Nat.add(position,Nat.add(U32.to_nat(count),total_channels(~Element,rest)))) : U32}
      %Addition.associative(position,U32.to_nat(count),total_channels(~Element,rest)) :
        {channel_cursor(~Element,rest,U32.from_nat(Nat.add(position,U32.to_nat(count)))) == U32.from_nat(_) : U32}
      cursor_from_encoded(~Element,rest,Nat.add(position,U32.to_nat(count)))

def cursor_exact(~Element: Data,+parts: PartContents<Element>,+channel: U32) ->
  {channel_cursor(~Element,parts,channel) == U32.from_nat(Nat.add(U32.to_nat(channel),total_channels(~Element,parts))) : U32}:
  %Arithmetic.native_roundtrip(channel) : {channel_cursor(~Element,parts,_) == U32.from_nat(Nat.add(U32.to_nat(channel),total_channels(~Element,parts))) : U32}
  cursor_from_encoded(~Element,parts,U32.to_nat(channel))

def row_ready(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+channel: U32,+input: TensorContent<Element>,output: Buffer.Content<Element>,
  ready: Buffer.ContentReady<Element,output>) -> Buffer.ContentReady<Element,row_model(~Element,geometry,first,rows,channel,input,output)>:
  Buffer.Content{+tree,+length} = output
  Writes.content_ready(~Element,row_writes(~Element,geometry,first,rows,channel,input),tree,length,ready)

def tree_ready(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+input: TensorContent<Element>,output: Model.Content<Element>,
  ready: Model.ContentReady(~Element,output)) -> Model.ContentReady(~Element,tree_model(~Element,geometry,channel,input,output)):
  match output:
    case Model.RowBand{buffer,first,rows}: row_ready(~Element,geometry,first,rows,channel,input,buffer,ready)
    case Model.Joined{+left,+right}:
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{tree_ready(~Element,geometry,channel,input,left,left_ready),tree_ready(~Element,geometry,channel,input,right,right_ready)}

def aligned_ready(~Element: Data,+geometry: Bands.Geometry,+channel: U32,input: Model.Content<Element>,output: Model.Content<Element>,
  ready: Model.ContentReady(~Element,output)) -> Model.ContentReady(~Element,aligned_model(~Element,geometry,channel,input,output)):
  match input output:
    case Model.RowBand{Buffer.Content{+input_tree,input_length},first,rows} Model.RowBand{Buffer.Content{+output_tree,+output_length},out_first,+out_rows}:
      Writes.content_ready(~Element,Contiguous.writes(~Element,U32.to_nat(Bands.local_count(geometry,out_rows)),input_tree,0,0,BandConcat.channel_offset(geometry,channel,out_rows)),output_tree,output_length,ready)
    case Model.Joined{+left,+right} Model.Joined{+out_left,+out_right}:
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{aligned_ready(~Element,geometry,channel,left,out_left,left_ready),aligned_ready(~Element,geometry,channel,right,out_right,right_ready)}
    case Model.RowBand{input,first,rows} Model.Joined{left,right}: ready
    case Model.Joined{left,right} Model.RowBand{output,first,rows}: ready

def selected_ready(~Element: Data,same: Bool,+geometry: Bands.Geometry,+channel: U32,+input: Model.Content<Element>,+output: Model.Content<Element>,
  +shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+previous: U32,ready: Model.ContentReady(~Element,output)) ->
  Model.ContentReady(~Element,selected_model(~Element,same,geometry,channel,input,output,shape,fill,status,previous)):
  match same:
    case True{}: aligned_ready(~Element,geometry,channel,input,output,ready)
    case False{}: tree_ready(~Element,geometry,channel,BandedContent{input,geometry,shape,fill,status,previous},output,ready)

def storage_ready(~Element: Data,+geometry: Bands.Geometry,+channel: U32,input: TensorContent<Element>,+output: Model.Content<Element>,
  ready: Model.ContentReady(~Element,output)) -> Model.ContentReady(~Element,storage_model(~Element,geometry,channel,input,output)):
  match input:
    case DenseContent{buffer,shape,fill,status,previous}: tree_ready(~Element,geometry,channel,DenseContent{buffer,shape,fill,status,previous},output,ready)
    case BandedContent{+bands,source_geometry,+shape,+fill,+status,+previous}:
      selected_ready(~Element,Bands.same_plan(Model.content_layout(~Element,bands),Model.content_layout(~Element,output)),geometry,channel,bands,output,shape,fill,status,previous,ready)

def tree_metadata(~Element: Data,+geometry: Bands.Geometry,+output_geometry: Bands.Geometry,+channel: U32,+input: TensorContent<Element>,output: Model.Content<Element>,
  metadata: BandShape.Contents(~Element,output_geometry,output)) -> BandShape.Contents(~Element,output_geometry,tree_model(~Element,geometry,channel,input,output)):
  match output:
    case Model.RowBand{Buffer.Content{tree,length},first,rows}: metadata
    case Model.Joined{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{tree_metadata(~Element,geometry,output_geometry,channel,input,left,left_metadata),tree_metadata(~Element,geometry,output_geometry,channel,input,right,right_metadata)}

def aligned_metadata(~Element: Data,+geometry: Bands.Geometry,+output_geometry: Bands.Geometry,+channel: U32,input: Model.Content<Element>,output: Model.Content<Element>,
  metadata: BandShape.Contents(~Element,output_geometry,output)) -> BandShape.Contents(~Element,output_geometry,aligned_model(~Element,geometry,channel,input,output)):
  match input output:
    case Model.RowBand{Buffer.Content{input,input_length},first,rows} Model.RowBand{Buffer.Content{output,output_length},out_first,out_rows}: metadata
    case Model.Joined{+left,+right} Model.Joined{+out_left,+out_right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{aligned_metadata(~Element,geometry,output_geometry,channel,left,out_left,left_metadata),aligned_metadata(~Element,geometry,output_geometry,channel,right,out_right,right_metadata)}
    case Model.RowBand{input,first,rows} Model.Joined{left,right}: metadata
    case Model.Joined{left,right} Model.RowBand{output,first,rows}: metadata

def selected_metadata(~Element: Data,same: Bool,+geometry: Bands.Geometry,+output_geometry: Bands.Geometry,+channel: U32,+input: Model.Content<Element>,+output: Model.Content<Element>,
  +shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+previous: U32,metadata: BandShape.Contents(~Element,output_geometry,output)) ->
  BandShape.Contents(~Element,output_geometry,selected_model(~Element,same,geometry,channel,input,output,shape,fill,status,previous)):
  match same:
    case True{}: aligned_metadata(~Element,geometry,output_geometry,channel,input,output,metadata)
    case False{}: tree_metadata(~Element,geometry,output_geometry,channel,BandedContent{input,geometry,shape,fill,status,previous},output,metadata)

def storage_metadata(~Element: Data,+geometry: Bands.Geometry,+output_geometry: Bands.Geometry,+channel: U32,input: TensorContent<Element>,+output: Model.Content<Element>,
  metadata: BandShape.Contents(~Element,output_geometry,output)) -> BandShape.Contents(~Element,output_geometry,storage_model(~Element,geometry,channel,input,output)):
  match input:
    case DenseContent{buffer,shape,fill,status,previous}: tree_metadata(~Element,geometry,output_geometry,channel,DenseContent{buffer,shape,fill,status,previous},output,metadata)
    case BandedContent{+bands,source_geometry,+shape,+fill,+status,+previous}:
      selected_metadata(~Element,Bands.same_plan(Model.content_layout(~Element,bands),Model.content_layout(~Element,output)),geometry,output_geometry,channel,bands,output,shape,fill,status,previous,metadata)

def tree_layout(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+input: TensorContent<Element>,output: Model.Content<Element>) ->
  {Model.content_layout(~Element,tree_model(~Element,geometry,channel,input,output)) == Model.content_layout(~Element,output) : Bands.Plan}:
  match output:
    case Model.RowBand{buffer,first,rows}: {==}
    case Model.Joined{+left,+right}:
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,tree_model(~Element,geometry,channel,input,left)),Model.content_layout(~Element,left),tree_layout(~Element,geometry,channel,input,left)) :
        {Bands.Split{_,Model.content_layout(~Element,tree_model(~Element,geometry,channel,input,right))} == Bands.Split{Model.content_layout(~Element,left),Model.content_layout(~Element,right)} : Bands.Plan}
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,tree_model(~Element,geometry,channel,input,right)),Model.content_layout(~Element,right),tree_layout(~Element,geometry,channel,input,right)) :
        {Bands.Split{Model.content_layout(~Element,left),_} == Bands.Split{Model.content_layout(~Element,left),Model.content_layout(~Element,right)} : Bands.Plan}
      {==}

def aligned_layout(~Element: Data,+geometry: Bands.Geometry,+channel: U32,input: Model.Content<Element>,output: Model.Content<Element>) ->
  {Model.content_layout(~Element,aligned_model(~Element,geometry,channel,input,output)) == Model.content_layout(~Element,output) : Bands.Plan}:
  match input output:
    case Model.RowBand{input,first,rows} Model.RowBand{output,out_first,out_rows}: {==}
    case Model.Joined{+left,+right} Model.Joined{+out_left,+out_right}:
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,aligned_model(~Element,geometry,channel,left,out_left)),Model.content_layout(~Element,out_left),aligned_layout(~Element,geometry,channel,left,out_left)) :
        {Bands.Split{_,Model.content_layout(~Element,aligned_model(~Element,geometry,channel,right,out_right))} == Bands.Split{Model.content_layout(~Element,out_left),Model.content_layout(~Element,out_right)} : Bands.Plan}
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,aligned_model(~Element,geometry,channel,right,out_right)),Model.content_layout(~Element,out_right),aligned_layout(~Element,geometry,channel,right,out_right)) :
        {Bands.Split{Model.content_layout(~Element,out_left),_} == Bands.Split{Model.content_layout(~Element,out_left),Model.content_layout(~Element,out_right)} : Bands.Plan}
      {==}
    case Model.RowBand{input,first,rows} Model.Joined{left,right}: {==}
    case Model.Joined{left,right} Model.RowBand{output,first,rows}: {==}

def selected_layout(~Element: Data,same: Bool,+geometry: Bands.Geometry,+channel: U32,+input: Model.Content<Element>,+output: Model.Content<Element>,
  +shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+previous: U32) ->
  {Model.content_layout(~Element,selected_model(~Element,same,geometry,channel,input,output,shape,fill,status,previous)) == Model.content_layout(~Element,output) : Bands.Plan}:
  match same:
    case True{}: aligned_layout(~Element,geometry,channel,input,output)
    case False{}: tree_layout(~Element,geometry,channel,BandedContent{input,geometry,shape,fill,status,previous},output)

def storage_layout(~Element: Data,+geometry: Bands.Geometry,+channel: U32,input: TensorContent<Element>,+output: Model.Content<Element>) ->
  {Model.content_layout(~Element,storage_model(~Element,geometry,channel,input,output)) == Model.content_layout(~Element,output) : Bands.Plan}:
  match input:
    case DenseContent{buffer,shape,fill,status,previous}: tree_layout(~Element,geometry,channel,DenseContent{buffer,shape,fill,status,previous},output)
    case BandedContent{+bands,source_geometry,+shape,+fill,+status,+previous}:
      selected_layout(~Element,Bands.same_plan(Model.content_layout(~Element,bands),Model.content_layout(~Element,output)),geometry,channel,bands,output,shape,fill,status,previous)

def storage_well_formed(~Element: Data,+geometry: Bands.Geometry,+output_geometry: Bands.Geometry,+channel: U32,+input: TensorContent<Element>,+output: Model.Content<Element>,
  evidence: BandShape.WellFormed<Element,output_geometry,output>) -> BandShape.WellFormed<Element,output_geometry,storage_model(~Element,geometry,channel,input,output)>:
  BandShape.WellFormed{metadata,ready,coverage} = evidence
  BandShape.WellFormed{storage_metadata(~Element,geometry,output_geometry,channel,input,output,metadata),storage_ready(~Element,geometry,channel,input,output,ready),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,storage_model(~Element,geometry,channel,input,output)),Model.content_layout(~Element,output),storage_layout(~Element,geometry,channel,input,output)) : BandPlan.Coverage(_,0n,Inspection.total_rows(output_geometry))
    coverage}

def part_well_formed(~Element: Data,+geometry: Bands.Geometry,+output_geometry: Bands.Geometry,+input: TensorContent<Element>,state: ContentState(~Element),
  evidence: BandShape.WellFormed<Element,output_geometry,state_output(~Element,state)>) -> BandShape.WellFormed<Element,output_geometry,state_output(~Element,part_model(~Element,geometry,input,state))>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  (+output,+channel) = state
  storage_well_formed(~Element,Bands.Geometry{images,BandConcat.channels(tensor_shape(~Element,input)),height,width},output_geometry,channel,input,output,evidence)

def parts_well_formed(~Element: Data,parts: PartContents<Element>,+geometry: Bands.Geometry,+output_geometry: Bands.Geometry,+state: ContentState(~Element),
  evidence: BandShape.WellFormed<Element,output_geometry,state_output(~Element,state)>) -> BandShape.WellFormed<Element,output_geometry,state_output(~Element,parts_model(~Element,parts,geometry,state))>:
  match parts:
    case SingleContent{input}: part_well_formed(~Element,geometry,output_geometry,input,state,evidence)
    case ConsContent{+input,+rest}:
      parts_well_formed(~Element,rest,geometry,output_geometry,part_model(~Element,geometry,input,state),part_well_formed(~Element,geometry,output_geometry,input,state,evidence))

def append_parts_well_formed_packed(~Element: Data,+parts: PartsOwnerModel<Element>,+geometry: Bands.Geometry,+workers: U32,+state: BandStateModel<Element>,+output_geometry: Bands.Geometry,
  evidence: BandShape.WellFormed<Element,output_geometry,state_output(~Element,state_contents(~Element,band_state_pack(~Element,state)))>) ->
  BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,parts),geometry,workers,band_state_pack(~Element,state)))>:
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,parts),geometry,workers,band_state_pack(~Element,state))),state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,parts)),geometry,state_contents(~Element,band_state_pack(~Element,state)))),append_parts_packed(~Element,parts,geometry,workers,state)) : BandShape.WellFormed<Element,output_geometry,_>
  parts_well_formed(~Element,parts_contents(~Element,parts_owner_pack(~Element,parts)),geometry,output_geometry,state_contents(~Element,band_state_pack(~Element,state)),evidence)

def append_parts_well_formed_observed(~Element: Data,-parts: Tensors.Parts<Element>,+geometry: Bands.Geometry,+workers: U32,-state: Bands.Bands(~Element) & U32,+output_geometry: Bands.Geometry,
  parts_observation: PartsObservation<Element,parts>,state_observation: BandStateObservation<Element,state>,
  evidence: BandShape.WellFormed<Element,output_geometry,state_output(~Element,state_contents(~Element,state))>) ->
  BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.append_parts(~Element,parts,geometry,workers,state))>:
  PartsObservation{+sources,parts_equation} = parts_observation
  BandStateObservation{+target,+state_equation} = state_observation
  +canonical_evidence = {
    %Equal.sym(Bands.Bands(~Element) & U32,band_state_pack(~Element,target),state,state_equation) : BandShape.WellFormed<Element,output_geometry,state_output(~Element,state_contents(~Element,_))>
    evidence : BandShape.WellFormed<Element,output_geometry,state_output(~Element,state_contents(~Element,band_state_pack(~Element,target)))>}
  %parts_equation : BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.append_parts(~Element,_,geometry,workers,state))>
  %state_equation : BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,sources),geometry,workers,_))>
  append_parts_well_formed_packed(~Element,sources,geometry,workers,target,output_geometry,canonical_evidence)

def append_parts_well_formed(~Element: Data,parts: Tensors.Parts<Element>,+geometry: Bands.Geometry,+workers: U32,state: Bands.Bands(~Element) & U32,+output_geometry: Bands.Geometry,
  evidence: BandShape.WellFormed<Element,output_geometry,state_output(~Element,state_contents(~Element,state))>) ->
  BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.append_parts(~Element,parts,geometry,workers,state))>:
  append_parts_well_formed_observed(~Element,parts,geometry,workers,state,output_geometry,observe_parts(~Element,parts),observe_band_state(~Element,state),evidence)

def join_model(~Element: Data,+layout: Bands.Plan,+geometry: Bands.Geometry,parts: PartContents<Element>,shape: List<&2,U32>,+fill: Element,workers: U32) -> TensorContent<Element>:
  BandedContent{state_output(~Element,parts_model(~Element,parts,geometry,(allocation_model(~Element,geometry,fill,layout),0))),geometry,shape,fill,Tensors.Ready{},workers}

def join_packed(~Element: Data,+layout: Bands.Plan,+geometry: Bands.Geometry,+parts: PartsOwnerModel<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {tensor_contents(~Element,BandConcat.join(~Element,layout,geometry,parts_owner_pack(~Element,parts),shape,fill,workers)) == join_model(~Element,layout,geometry,parts_contents(~Element,parts_owner_pack(~Element,parts)),shape,fill,workers) : TensorContent<Element>}:
  Equal.trans(TensorContent<Element>,tensor_contents(~Element,BandConcat.join(~Element,layout,geometry,parts_owner_pack(~Element,parts),shape,fill,workers)),
    BandedContent{state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,parts)),geometry,(Model.contents(~Element,BandConcat.allocate(~Element,layout,geometry,fill)),0))),geometry,shape,fill,Tensors.Ready{},workers},
    join_model(~Element,layout,geometry,parts_contents(~Element,parts_owner_pack(~Element,parts)),shape,fill,workers),
    Equal.cong(Model.Content<Element>,TensorContent<Element>,content => BandedContent{content,geometry,shape,fill,Tensors.Ready{},workers},
      Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,parts),geometry,workers,(BandConcat.allocate(~Element,layout,geometry,fill),0))),
      state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,parts)),geometry,(Model.contents(~Element,BandConcat.allocate(~Element,layout,geometry,fill)),0))),
      append_parts(~Element,parts_owner_pack(~Element,parts),geometry,workers,(BandConcat.allocate(~Element,layout,geometry,fill),0))),
    Equal.cong(Model.Content<Element>,TensorContent<Element>,content => BandedContent{state_output(~Element,parts_model(~Element,parts_contents(~Element,parts_owner_pack(~Element,parts)),geometry,(content,0))),geometry,shape,fill,Tensors.Ready{},workers},
      Model.contents(~Element,BandConcat.allocate(~Element,layout,geometry,fill)),allocation_model(~Element,geometry,fill,layout),allocate(~Element,geometry,fill,layout)))

def join_observed(~Element: Data,+layout: Bands.Plan,+geometry: Bands.Geometry,-parts: Tensors.Parts<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  observation: PartsObservation<Element,parts>) ->
  {tensor_contents(~Element,BandConcat.join(~Element,layout,geometry,parts,shape,fill,workers)) == join_model(~Element,layout,geometry,parts_contents(~Element,parts),shape,fill,workers) : TensorContent<Element>}:
  PartsObservation{+sources,equation} = observation
  %equation : {tensor_contents(~Element,BandConcat.join(~Element,layout,geometry,_,shape,fill,workers)) == join_model(~Element,layout,geometry,parts_contents(~Element,_),shape,fill,workers) : TensorContent<Element>}
  join_packed(~Element,layout,geometry,sources,shape,fill,workers)

def join(~Element: Data,+layout: Bands.Plan,+geometry: Bands.Geometry,parts: Tensors.Parts<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {tensor_contents(~Element,BandConcat.join(~Element,layout,geometry,parts,shape,fill,workers)) == join_model(~Element,layout,geometry,parts_contents(~Element,parts),shape,fill,workers) : TensorContent<Element>}:
  join_observed(~Element,layout,geometry,parts,shape,fill,workers,observe_parts(~Element,parts))

# Fresh-allocation WF is discharged by the allocator/plan producer. Once
# supplied, every actual Parts store retains it without channel bounds.
def join_well_formed_packed(~Element: Data,+layout: Bands.Plan,+geometry: Bands.Geometry,+parts: PartsOwnerModel<Element>,+fill: Element,+workers: U32,
  allocation: BandShape.WellFormed<Element,geometry,allocation_model(~Element,geometry,fill,layout)>) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,BandConcat.append_parts(~Element,parts_owner_pack(~Element,parts),geometry,workers,(BandConcat.allocate(~Element,layout,geometry,fill),0)))>:
  append_parts_well_formed(~Element,parts_owner_pack(~Element,parts),geometry,workers,(BandConcat.allocate(~Element,layout,geometry,fill),0),geometry,
    %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandConcat.allocate(~Element,layout,geometry,fill)),allocation_model(~Element,geometry,fill,layout),allocate(~Element,geometry,fill,layout)) : BandShape.WellFormed<Element,geometry,_>
    allocation)

def join_well_formed_observed(~Element: Data,+layout: Bands.Plan,+geometry: Bands.Geometry,-parts: Tensors.Parts<Element>,+fill: Element,+workers: U32,
  observation: PartsObservation<Element,parts>,allocation: BandShape.WellFormed<Element,geometry,allocation_model(~Element,geometry,fill,layout)>) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,BandConcat.append_parts(~Element,parts,geometry,workers,(BandConcat.allocate(~Element,layout,geometry,fill),0)))>:
  PartsObservation{+sources,equation} = observation
  %equation : BandShape.WellFormed<Element,geometry,Model.contents(~Element,BandConcat.append_parts(~Element,_,geometry,workers,(BandConcat.allocate(~Element,layout,geometry,fill),0)))>
  join_well_formed_packed(~Element,layout,geometry,sources,fill,workers,allocation)

def join_well_formed(~Element: Data,+layout: Bands.Plan,+geometry: Bands.Geometry,parts: Tensors.Parts<Element>,+fill: Element,+workers: U32,
  allocation: BandShape.WellFormed<Element,geometry,allocation_model(~Element,geometry,fill,layout)>) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,BandConcat.append_parts(~Element,parts,geometry,workers,(BandConcat.allocate(~Element,layout,geometry,fill),0)))>:
  join_well_formed_observed(~Element,layout,geometry,parts,fill,workers,observe_parts(~Element,parts),allocation)

# Finite channel values are threaded through the whole output tree. The
# source's coverage and matching writes are produced by shared Range
# semantics. Target bounds remain explicit until the Parts survey and
# allocator producers discharge them; they are not concat equalities.
def TreeChannelValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+channel: U32,output: Model.Content<Element>) -> Data:
  match output:
    case Model.RowBand{+buffer,+first,+rows}:
      ChannelValues(~Element,~Logical,~original,logical,BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,channel,rows)},buffer)
    case Model.Joined{left,right}:
      Shape.Both<TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,left),TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,right)>

def RangeTargets(~Element: Data,+geometry: Bands.Geometry,+channel: U32,+limits: Machine.Limits,output: Model.Model<Element>) -> Data:
  match output:
    case Model.Leaf{buffer,+first,+rows}:
      +request = {BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,channel,rows)} : BandRange.Request}
      Shape.Both<BandRange.Target(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,buffer))),
        BandShape.Leaf(geometry,first,rows,channel_length(request))>
    case Model.Fork{left,right}:
      Shape.Both<RangeTargets(~Element,geometry,channel,limits,left),RangeTargets(~Element,geometry,channel,limits,right)>

def tree_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+external_geometry: Bands.Geometry,+geometry: Bands.Geometry,+channel: U32,
  +source: Model.Model<Element>,output: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  +source_well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,
  +source_values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  ready: Model.Ready(~Element,output),targets: RangeTargets(~Element,geometry,channel,limits,output)) ->
  TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,
    tree_model(~Element,external_geometry,channel,BandedContent{Model.content(~Element,source),geometry,shape,fill,status,workers},Model.content(~Element,output))):
  match output:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      Shape.Both{target,metadata} = targets
      +buffer = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
      +request = {BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,channel,rows)} : BandRange.Request}
      written_cells(~Element,~Logical,~original,logical,request,source,buffer,limits,channel_length(request),capacity,domain,target,source_well_formed,metadata,ready,source_values)
    case Model.Fork{+left,+right}:
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_targets,right_targets} = targets
      Shape.Both{tree_values(~Element,~Logical,~original,logical,external_geometry,geometry,channel,source,left,shape,fill,status,workers,limits,capacity,domain,source_well_formed,source_values,left_ready,left_targets),
        tree_values(~Element,~Logical,~original,logical,external_geometry,geometry,channel,source,right,shape,fill,status,workers,limits,capacity,domain,source_well_formed,source_values,right_ready,right_targets)}

def appended_tree_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+external_geometry: Bands.Geometry,+geometry: Bands.Geometry,+channel: U32,
  +source: Model.Model<Element>,+output: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  +limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>,
  source_well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,source_values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  ready: Model.Ready(~Element,output),targets: RangeTargets(~Element,geometry,channel,limits,output)) ->
  TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,
    Model.contents(~Element,BandConcat.appended_rows(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,output),external_geometry,channel,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers})))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandConcat.appended_rows(~Element,BandConcat.append_tree(~Element,Model.pack(~Element,output),external_geometry,channel,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers}))),
    tree_model(~Element,external_geometry,channel,tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers}),Model.contents(~Element,Model.pack(~Element,output))),
    appended_tree(~Element,external_geometry,channel,Model.pack(~Element,output),Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers})) : TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,_)
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,source)),Model.content(~Element,source),Model.packed_contents(~Element,source)) :
    TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,tree_model(~Element,external_geometry,channel,BandedContent{_,geometry,shape,fill,status,workers},Model.contents(~Element,Model.pack(~Element,output))))
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,output)),Model.content(~Element,output),Model.packed_contents(~Element,output)) :
    TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,tree_model(~Element,external_geometry,channel,BandedContent{Model.content(~Element,source),geometry,shape,fill,status,workers},_))
  tree_values(~Element,~Logical,~original,logical,external_geometry,geometry,channel,source,output,shape,fill,status,workers,limits,capacity,domain,source_well_formed,source_values,ready,targets)
