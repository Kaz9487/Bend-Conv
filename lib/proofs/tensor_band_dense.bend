# Dense to band conversion: copy premises, allocation, the channel copies and
# the values they establish, including empty and existing-owner cases.
import Base
import ./tensor_band_gather.bend as BandGather
import ./storage_buffer_model.bend as Buffer
import ../tensor_band.bend as Bands
import ./storage_tree_model.bend as Tree
import ./tensor_band_model.bend as Model
import ./storage_shape.bend as Shape
import ../machine_limits.bend as Machine
import ./row_major_index.bend as RowMajor
import ./tensor_band_copy.bend as BandCopy
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./machine_limit_bounds.bend as Limits
import ./nat_division_order.bend as DivisionOrder
import ./tensor_band_count.bend as BandCount
import ./tensor_band_inspection.bend as Inspection
import ./tensor_contiguous_bounds.bend as ContiguousBounds
import ./tensor_shape_refinement.bend as ShapeLaws
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding
import ./write_program.bend as Writes
import ../storage_buffer.bend as Storage
import ./natural_addition.bend as Addition
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./storage_address_bounds.bend as Address
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_copy_refinement.bend as StorageCopyLaws
import ./strided_range_bounds.bend as Strided
import ./tensor_allocation_depth.bend as Depth
import ./u32_division.bend as U32Division
import ./word_multiplication.bend as Multiplication
import ../storage_allocation.bend as Allocation
import ../traversal_partition.bend as Partition

# The inverse channel copy used by dense-to-band conversion establishes Values.
# Its source addresses reuse the gather coordinate proof in the other direction.
def write_dense(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> Tree.Tree<Element>:
  Bands.Geometry{images,+channels,+height,+width} = geometry
  Writes.pure(Element,BandCopy.writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),
    ((first / height * channels * height + first % height) * width : U32),0,input),output)

def written(~Element: Data,+call: BandGather.Call,input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> Tree.Tree<Element>:
  write_dense(~Element,BandGather.values_geometry(call),BandGather.values_first(call),BandGather.values_rows(call),input,output)

def copied_cell(~Element: Data,+call: BandGather.Call,+input: Tree.Tree<Element>,+output: Buffer.Model<Element>,+channel: Nat,+row: Nat,+column: Nat,
  facts: BandGather.Facts<call>,ready: Buffer.Ready<Element,output>,length_equal: {Buffer.length(~Element,output) == BandGather.length(call) : Nat},
  +channel_inside: {Nat.is_lt(channel,BandValues.channels(BandGather.values_geometry(call))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(BandGather.values_rows(call))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,BandValues.width(BandGather.values_geometry(call))) == True{} : Bool}) ->
  BandValues.At(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),input,BandGather.values_geometry(call),BandGather.values_first(call),BandGather.values_rows(call),
    written(~Element,call,input,Buffer.tree(~Element,output)),BandGather.length(call),channel,row,column):
  BandGather.Call{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,+length,+capacity,+division_capacity} = call
  BandGather.Facts{+valid,+stride_exact,+length_exact,+length_bound,+division_ok,+plane_bound,+height_positive,+width_positive,+count_exact,+height_bound,+row_span} = facts
  Buffer.Ready{+depth_ok,room} = ready
  +plane = (rows * width : U32)
  +index = Nat.add(Nat.mul(row,U32.to_nat(width)),column)
  +position = BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column)
  +pixel = U32.from_nat(BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column))
  +index_inside = {
    %Equal.sym(Nat,U32.to_nat(plane),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),count_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(row,U32.to_nat(rows),column,U32.to_nat(width),row_inside,column_inside) : {Nat.is_lt(index,U32.to_nat(plane)) == True{} : Bool}}
  +target_region = BandCopy.whole_region(U32.to_nat(channels),plane,Tree.capacity(Element,Buffer.tree(~Element,output)))
  +target_valid = BandCopy.whole_valid(U32.to_nat(channels),plane,Tree.capacity(Element,Buffer.tree(~Element,output)),Buffer.capacity_bound(~Element,output,depth_ok),
    Order.le_lt_trans(0n,index,U32.to_nat(plane),Order.zero_le(index),index_inside),
    %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool}
    %length_equal : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool} room)
  +query = BandCopy.readback_query(target_region,channel,index)
  +encoded = Encoded.encoding(query,position,
    %count_exact : {U32.to_nat(query) == Nat.add(Nat.mul(channel,_),index) : Nat}
    BandCopy.readback_query_exact(target_region,channel,index,target_valid,channel_inside,index_inside))
  +bound = {
    %Equal.sym(Nat,length,Nat.mul(U32.to_nat(channels),U32.to_nat(plane)),length_exact) : {Nat.is_lt(position,_) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(plane),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),count_exact) : {Nat.is_lt(position,Nat.mul(U32.to_nat(channels),_)) == True{} : Bool}
    BandValues.position_inside(Bands.Geometry{images,channels,height,width},rows,channel,row,column,channel_inside,row_inside,column_inside) : {Nat.is_lt(position,length) == True{} : Bool}}
  Shape.Both{bound,
    %encoded : {Tree.read(Element,written(~Element,call,input,Buffer.tree(~Element,output)),_) == Tree.read(Element,input,pixel) : Element}
    %BandGather.leaf_address(call,Nat.div(Nat.add(U32.to_nat(first),row),U32.to_nat(height)),channel,Nat.mod(Nat.add(U32.to_nat(first),row),U32.to_nat(height)),column,row,
      BandGather.Facts{valid,stride_exact,length_exact,length_bound,division_ok,plane_bound,height_positive,width_positive,count_exact,height_bound,row_span},
      channel_inside,DivisionOrder.remainder_small(Nat.add(U32.to_nat(first),row),U32.to_nat(height),height_positive),column_inside,row_inside,
      Equal.sym(Nat,Nat.add(U32.to_nat(first),row),Nat.add(Nat.mul(Nat.div(Nat.add(U32.to_nat(first),row),U32.to_nat(height)),U32.to_nat(height)),Nat.mod(Nat.add(U32.to_nat(first),row),U32.to_nat(height))),DivisionOrder.quotient_exact(Nat.add(U32.to_nat(first),row),U32.to_nat(height),height_positive))) :
      {Tree.read(Element,written(~Element,call,input,Buffer.tree(~Element,output)),query) == Tree.read(Element,input,_) : Element}
    BandCopy.read_inside(~Element,U32.to_nat(channels),0n,channel,index,target_region,(height * width : U32),BandCopy.destination(BandGather.values_region(call)),input,Buffer.tree(~Element,output),
      Buffer.depth(~Element,output),Buffer.balanced(~Element,output),depth_ok,{==},target_valid,Order.reflexive(U32.to_nat(channels)),channel_inside,index_inside)}

# What one copied leaf supplies to every cell.
type Cells<-Element: Data> is Data:
  Cells{call: BandGather.Call,input: Tree.Tree<Element>,output: Buffer.Model<Element>,facts: BandGather.Facts<call>,ready: Buffer.Ready<Element,output>,
    length_equal: {Buffer.length(~Element,output) == BandGather.length(call) : Nat}}

def cells_geometry(~Element: Data,evidence: Cells<Element>) -> Bands.Geometry:
  Cells{call,input,output,facts,ready,length_equal} = evidence
  BandGather.values_geometry(call)

def cells_rows(~Element: Data,evidence: Cells<Element>) -> U32:
  Cells{call,input,output,facts,ready,length_equal} = evidence
  BandGather.values_rows(call)

def Cell(~Element: Data,+evidence: Cells<Element>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  Cells{+call,+input,+output,facts,ready,length_equal} = evidence
  BandValues.At(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),input,BandGather.values_geometry(call),BandGather.values_first(call),BandGather.values_rows(call),
    written(~Element,call,input,Buffer.tree(~Element,output)),BandGather.length(call),channel,row,column)

def leaf_cell(~Element: Data,+evidence: Cells<Element>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(cells_geometry(~Element,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(cells_rows(~Element,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(cells_geometry(~Element,evidence))) == True{} : Bool}) -> Cell(~Element,evidence,channel,row,column):
  Cells{+call,+input,+output,facts,ready,length_equal} = evidence
  copied_cell(~Element,call,input,output,channel,row,column,facts,ready,length_equal,channel_inside,row_inside,column_inside)

def values(~Element: Data,+call: BandGather.Call,+input: Tree.Tree<Element>,+output: Buffer.Model<Element>,
  facts: BandGather.Facts<call>,ready: Buffer.Ready<Element,output>,length_equal: {Buffer.length(~Element,output) == BandGather.length(call) : Nat}) ->
  BandValues.Leaf(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),input,BandGather.values_geometry(call),BandGather.values_first(call),BandGather.values_rows(call),
    written(~Element,call,input,Buffer.tree(~Element,output)),BandGather.length(call)):
  BandValues.cells(~Cells<Element>,~Cell(~Element),~cells_geometry(~Element),~cells_rows(~Element),~leaf_cell(~Element),Cells{call,input,output,facts,ready,length_equal})

# Actual dense-to-band channel copies, including allocation and source owner
# content preservation. The value theorem is kept in tensor_band_dense_values.
type Content<-Element: Data> is Data:
  Content{source: Buffer.Content<Element>,bands: Model.Content<Element>}

def contents(~Element: Data,owners: Storage.Buffer<Element> & Bands.Bands(~Element)) -> Content<Element>:
  (source,bands) = owners
  Content{Buffer.contents(~Element,source),Model.contents(~Element,bands)}

def bands(~Element: Data,content: Content<Element>) -> Model.Content<Element>:
  Content{source,bands} = content
  bands

def result(~Element: Data,first: U32,rows: U32,owners: StorageCopyLaws.Owners<Element>) -> Content<Element>:
  StorageCopyLaws.Owners{input,input_length,output,output_length} = owners
  Content{Buffer.Content{input,input_length},Model.RowBand{Buffer.Content{output,output_length},first,rows}}

def finished(~Element: Data,+first: U32,+rows: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {contents(~Element,Bands.dense_result(~Element,first,rows,owners)) == result(~Element,first,rows,StorageCopyLaws.observe(~Element,owners)) : Content<Element>}:
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate}) = owners
  {==}

def run(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Bands.Bands(~Element):
  Bands.Geometry{images,+channels,+height,+width} = geometry
  Bands.dense_result(~Element,first,rows,Bands.planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),
    ((first / height * channels * height + first % height) * width : U32),0,(input,output)))

def copy_model(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,input: Buffer.Content<Element>,output: Buffer.Content<Element>) -> Content<Element>:
  Buffer.Content{+input,+input_length} = input
  Buffer.Content{output,output_length} = output
  Content{Buffer.Content{input,input_length},Model.RowBand{Buffer.Content{write_dense(~Element,geometry,first,rows,input,output),output_length},first,rows}}

def copy(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {contents(~Element,run(~Element,geometry,first,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output))) ==
    copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),Buffer.content(~Element,output)) : Content<Element>}:
  Bands.Geometry{images,+channels,+height,+width} = geometry
  Buffer.Model{+input,+input_length,+input_certificate} = input
  Buffer.Model{+output,+output_length,+output_certificate} = output
  +start = ((first / height * channels * height + first % height) * width : U32)
  %Equal.sym(Content<Element>,contents(~Element,Bands.dense_result(~Element,first,rows,Bands.planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),start,0,
      (Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate}))))),
    result(~Element,first,rows,StorageCopyLaws.observe(~Element,Bands.planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),start,0,
      (Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate}))))),
    finished(~Element,first,rows,Bands.planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),start,0,
      (Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate}))))) :
    {_ == copy_model(~Element,geometry,first,rows,Buffer.Content{input,input_length},Buffer.Content{output,output_length}) : Content<Element>}
  Equal.cong(StorageCopyLaws.Owners<Element>,Content<Element>,owners => result(~Element,first,rows,owners),
    StorageCopyLaws.observe(~Element,Bands.planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),start,0,
      (Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate})))),
    StorageCopyLaws.Owners{input,input_length,write_dense(~Element,geometry,first,rows,input,output),output_length},
    BandCopy.packed_writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),start,0,
      Buffer.Model{input,input_length,input_certificate},Buffer.Model{output,output_length,output_certificate}))

type Context<-Element: Data> is Data:
  Context{geometry: Bands.Geometry,first: U32,rows: U32,input: Buffer.Model<Element>}

def proposition(~Element: Data,context: Context<Element>,-output: Storage.Buffer<Element>) -> Type:
  Context{+geometry,+first,+rows,+input} = context
  {contents(~Element,run(~Element,geometry,first,rows,Buffer.pack(~Element,input),output)) ==
    copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),Buffer.contents(~Element,output)) : Content<Element>}

def copy_native(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,output: Storage.Buffer<Element>) ->
  {contents(~Element,run(~Element,geometry,first,rows,Buffer.pack(~Element,input),output)) ==
    copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),Buffer.contents(~Element,output)) : Content<Element>}:
  Buffer.lift(~Element,~Context<Element>,~proposition(~Element),Context{geometry,first,rows,input},output,Buffer.observe(~Element,output),output =>
    +output = output
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)) :
      {contents(~Element,run(~Element,geometry,first,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output))) == copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),_) : Content<Element>}
    copy(~Element,geometry,first,rows,input,output))

def leaf_model(~Element: Data,+geometry: Bands.Geometry,fill: Element,+first: U32,+rows: U32,input: Buffer.Content<Element>) -> Content<Element>:
  +count = Bands.local_count(geometry,rows)
  copy_model(~Element,geometry,first,rows,input,Buffer.Content{Tree.fresh(Element,Storage.depth_for(count),fill),U32.to_nat(count)})

def refinement_leaf(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>) ->
  {contents(~Element,Bands.dense_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))) == leaf_model(~Element,geometry,fill,first,rows,Buffer.content(~Element,input)) : Content<Element>}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  +count = Bands.local_count(geometry,rows)
  Equal.trans(Content<Element>,contents(~Element,Bands.dense_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))),
    copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),Buffer.contents(~Element,Bands.allocate(~Element,fill,count))),leaf_model(~Element,geometry,fill,first,rows,Buffer.content(~Element,input)),
    copy_native(~Element,geometry,first,rows,input,Bands.allocate(~Element,fill,count)),
    Equal.cong(Buffer.Content<Element>,Content<Element>,output => copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),output),
      Buffer.contents(~Element,Bands.allocate(~Element,fill,count)),Buffer.Content{Tree.fresh(Element,Storage.depth_for(count),fill),U32.to_nat(count)},AllocationLaws.contents(~Element,Storage.depth_for(count),fill,U32.to_nat(count))))

def model(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+input: Buffer.Content<Element>) -> Model.Content<Element>:
  match layout:
    case Bands.Rows{first,rows}: bands(~Element,leaf_model(~Element,geometry,fill,first,rows,input))
    case Bands.Split{left,right}: Model.Joined{model(~Element,geometry,fill,left,input),model(~Element,geometry,fill,right,input)}

def leaf_source(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Content<Element>) ->
  {leaf_model(~Element,geometry,fill,first,rows,input) == Content{input,model(~Element,geometry,fill,Bands.Rows{first,rows},input)} : Content<Element>}:
  Buffer.Content{tree,length} = input
  {==}

def joined(~Element: Data,left: Model.Content<Element>,right: Content<Element>) -> Content<Element>:
  Content{source,right} = right
  Content{source,Model.Joined{left,right}}

def join_native(~Element: Data,left: Bands.Bands(~Element),right: Storage.Buffer<Element> & Bands.Bands(~Element)) -> Storage.Buffer<Element> & Bands.Bands(~Element):
  Storage.bind_pair(~Storage.Buffer<Element>,~Bands.Bands(~Element),~(Storage.Buffer<Element> & Bands.Bands(~Element)),right,
    source => right => (source,Partition.Fork{left,right}))

def joined_contents(~Element: Data,left: Bands.Bands(~Element),right: Storage.Buffer<Element> & Bands.Bands(~Element)) ->
  {contents(~Element,join_native(~Element,left,right)) == joined(~Element,Model.contents(~Element,left),contents(~Element,right)) : Content<Element>}:
  (source,right) = right
  {==}

def continued_model(~Element: Data,geometry: Bands.Geometry,fill: Element,layout: Bands.Plan,left: Content<Element>) -> Content<Element>:
  Content{+source,left} = left
  Content{source,Model.Joined{left,model(~Element,geometry,fill,layout,source)}}

def continued(~Element: Data,geometry: Bands.Geometry,fill: Element,layout: Bands.Plan,left: Storage.Buffer<Element> & Bands.Bands(~Element)) -> Storage.Buffer<Element> & Bands.Bands(~Element):
  Storage.bind_pair(~Storage.Buffer<Element>,~Bands.Bands(~Element),~(Storage.Buffer<Element> & Bands.Bands(~Element)),left,
    source => left => join_native(~Element,left,Bands.dense_ranges(~Element,geometry,fill,layout,source)))

def continue_packed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Buffer.Model<Element>,+left: Model.Model<Element>,
  rule: @input: Storage.Buffer<Element> ->
    {contents(~Element,Bands.dense_ranges(~Element,geometry,fill,layout,input)) == Content{Buffer.contents(~Element,input),model(~Element,geometry,fill,layout,Buffer.contents(~Element,input))} : Content<Element>}) ->
  {contents(~Element,continued(~Element,geometry,fill,layout,(Buffer.pack(~Element,source),Model.pack(~Element,left)))) ==
    continued_model(~Element,geometry,fill,layout,contents(~Element,(Buffer.pack(~Element,source),Model.pack(~Element,left)))) : Content<Element>}:
  Equal.trans(Content<Element>,contents(~Element,join_native(~Element,Model.pack(~Element,left),Bands.dense_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,source)))),
    joined(~Element,Model.contents(~Element,Model.pack(~Element,left)),contents(~Element,Bands.dense_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,source)))),
    continued_model(~Element,geometry,fill,layout,contents(~Element,(Buffer.pack(~Element,source),Model.pack(~Element,left)))),
    joined_contents(~Element,Model.pack(~Element,left),Bands.dense_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,source))),
    Equal.cong(Content<Element>,Content<Element>,right => joined(~Element,Model.contents(~Element,Model.pack(~Element,left)),right),
      contents(~Element,Bands.dense_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,source))),
      Content{Buffer.contents(~Element,Buffer.pack(~Element,source)),model(~Element,geometry,fill,layout,Buffer.contents(~Element,Buffer.pack(~Element,source)))},rule(Buffer.pack(~Element,source))))

def continue_observed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,-source: Storage.Buffer<Element>,-left: Bands.Bands(~Element),
  source_observation: Buffer.Observation<Element,source>,left_observation: Model.Observation<Element,left>,
  rule: @input: Storage.Buffer<Element> ->
    {contents(~Element,Bands.dense_ranges(~Element,geometry,fill,layout,input)) == Content{Buffer.contents(~Element,input),model(~Element,geometry,fill,layout,Buffer.contents(~Element,input))} : Content<Element>}) ->
  {contents(~Element,continued(~Element,geometry,fill,layout,(source,left))) == continued_model(~Element,geometry,fill,layout,contents(~Element,(source,left))) : Content<Element>}:
  Buffer.Observation{+source_model,source_equal} = source_observation
  Model.Observation{+left_model,left_equal} = left_observation
  %source_equal : {contents(~Element,continued(~Element,geometry,fill,layout,(_,left))) == continued_model(~Element,geometry,fill,layout,contents(~Element,(_,left))) : Content<Element>}
  %left_equal : {contents(~Element,continued(~Element,geometry,fill,layout,(Buffer.pack(~Element,source_model),_))) ==
    continued_model(~Element,geometry,fill,layout,contents(~Element,(Buffer.pack(~Element,source_model),_))) : Content<Element>}
  continue_packed(~Element,geometry,fill,layout,source_model,left_model,rule)

def continuation(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,left: Storage.Buffer<Element> & Bands.Bands(~Element),
  rule: @input: Storage.Buffer<Element> ->
    {contents(~Element,Bands.dense_ranges(~Element,geometry,fill,layout,input)) == Content{Buffer.contents(~Element,input),model(~Element,geometry,fill,layout,Buffer.contents(~Element,input))} : Content<Element>}) ->
  {contents(~Element,continued(~Element,geometry,fill,layout,left)) == continued_model(~Element,geometry,fill,layout,contents(~Element,left)) : Content<Element>}:
  (source,left) = left
  continue_observed(~Element,geometry,fill,layout,source,left,Buffer.observe(~Element,source),Model.observe(~Element,left),rule)

type PlanContext<-Element: Data> is Data:
  PlanContext{geometry: Bands.Geometry,fill: Element,layout: Bands.Plan}

def plan_proposition(~Element: Data,context: PlanContext<Element>,-input: Storage.Buffer<Element>) -> Type:
  PlanContext{+geometry,+fill,+layout} = context
  {contents(~Element,Bands.dense_ranges(~Element,geometry,fill,layout,input)) == Content{Buffer.contents(~Element,input),model(~Element,geometry,fill,layout,Buffer.contents(~Element,input))} : Content<Element>}

def ranges(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,input: Storage.Buffer<Element>) ->
  {contents(~Element,Bands.dense_ranges(~Element,geometry,fill,layout,input)) == Content{Buffer.contents(~Element,input),model(~Element,geometry,fill,layout,Buffer.contents(~Element,input))} : Content<Element>}:
  match layout:
    case Bands.Rows{+first,+rows}:
      Buffer.lift(~Element,~PlanContext<Element>,~plan_proposition(~Element),PlanContext{geometry,fill,Bands.Rows{first,rows}},input,Buffer.observe(~Element,input),input =>
        +input = input
        %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.content(~Element,input),Buffer.packed_contents(~Element,input)) :
          {contents(~Element,Bands.dense_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))) == Content{_,model(~Element,geometry,fill,Bands.Rows{first,rows},_)} : Content<Element>}
        Equal.trans(Content<Element>,contents(~Element,Bands.dense_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))),leaf_model(~Element,geometry,fill,first,rows,Buffer.content(~Element,input)),
          Content{Buffer.content(~Element,input),model(~Element,geometry,fill,Bands.Rows{first,rows},Buffer.content(~Element,input))},refinement_leaf(~Element,geometry,fill,first,rows,input),leaf_source(~Element,geometry,fill,first,rows,Buffer.content(~Element,input))))
    case Bands.Split{+left,+right}:
      Buffer.lift(~Element,~PlanContext<Element>,~plan_proposition(~Element),PlanContext{geometry,fill,Bands.Split{left,right}},input,Buffer.observe(~Element,input),input =>
        +input = input
        Equal.trans(Content<Element>,contents(~Element,continued(~Element,geometry,fill,right,Bands.dense_ranges(~Element,geometry,fill,left,Buffer.pack(~Element,input)))),
          continued_model(~Element,geometry,fill,right,contents(~Element,Bands.dense_ranges(~Element,geometry,fill,left,Buffer.pack(~Element,input)))),
          Content{Buffer.contents(~Element,Buffer.pack(~Element,input)),model(~Element,geometry,fill,Bands.Split{left,right},Buffer.contents(~Element,Buffer.pack(~Element,input)))},
          continuation(~Element,geometry,fill,right,Bands.dense_ranges(~Element,geometry,fill,left,Buffer.pack(~Element,input)),source => ranges(~Element,geometry,fill,right,source)),
          Equal.cong(Content<Element>,Content<Element>,left => continued_model(~Element,geometry,fill,right,left),
            contents(~Element,Bands.dense_ranges(~Element,geometry,fill,left,Buffer.pack(~Element,input))),
            Content{Buffer.contents(~Element,Buffer.pack(~Element,input)),model(~Element,geometry,fill,left,Buffer.contents(~Element,Buffer.pack(~Element,input)))},ranges(~Element,geometry,fill,left,Buffer.pack(~Element,input)))))

def result_contents(~Element: Data,owners: Storage.Buffer<Element> & Bands.Bands(~Element)) ->
  {Model.contents(~Element,Bands.result(~Element,owners)) == bands(~Element,contents(~Element,owners)) : Model.Content<Element>}:
  (source,result) = owners
  {==}

def refinement_from_dense(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>) ->
  {Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input))) == model(~Element,geometry,fill,layout,Buffer.content(~Element,input)) : Model.Content<Element>}:
  Equal.trans(Model.Content<Element>,Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),
    bands(~Element,contents(~Element,Bands.dense_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))),model(~Element,geometry,fill,layout,Buffer.content(~Element,input)),
    result_contents(~Element,Bands.dense_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),
    %Buffer.packed_contents(~Element,input) :
      {bands(~Element,contents(~Element,Bands.dense_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))) == model(~Element,geometry,fill,layout,_) : Model.Content<Element>}
    Equal.cong(Content<Element>,Model.Content<Element>,result => bands(~Element,result),contents(~Element,Bands.dense_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),
      Content{Buffer.contents(~Element,Buffer.pack(~Element,input)),model(~Element,geometry,fill,layout,Buffer.contents(~Element,Buffer.pack(~Element,input)))},ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))))

def leaf_layout(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,input: Buffer.Content<Element>) ->
  {Model.content_layout(~Element,model(~Element,geometry,fill,Bands.Rows{first,rows},input)) == Bands.Rows{first,rows} : Bands.Plan}:
  Buffer.Content{tree,length} = input
  {==}

def model_layout(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+input: Buffer.Content<Element>) ->
  {Model.content_layout(~Element,model(~Element,geometry,fill,layout,input)) == layout : Bands.Plan}:
  match layout:
    case Bands.Rows{first,rows}: leaf_layout(~Element,geometry,fill,first,rows,input)
    case Bands.Split{+left,+right}:
      %model_layout(~Element,geometry,fill,left,input) :
        {Bands.Split{Model.content_layout(~Element,model(~Element,geometry,fill,left,input)),Model.content_layout(~Element,model(~Element,geometry,fill,right,input))} == Bands.Split{_,right} : Bands.Plan}
      %model_layout(~Element,geometry,fill,right,input) :
        {Bands.Split{Model.content_layout(~Element,model(~Element,geometry,fill,left,input)),Model.content_layout(~Element,model(~Element,geometry,fill,right,input))} == Bands.Split{Model.content_layout(~Element,model(~Element,geometry,fill,left,input)),_} : Bands.Plan}
      {==}

def from_dense_layout(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>) ->
  {Model.content_layout(~Element,Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))) == layout : Bands.Plan}:
  Equal.trans(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))),
    Model.content_layout(~Element,model(~Element,geometry,fill,layout,Buffer.content(~Element,input))),layout,
    Equal.cong(Model.Content<Element>,Bands.Plan,result => Model.content_layout(~Element,result),
      Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),model(~Element,geometry,fill,layout,Buffer.content(~Element,input)),refinement_from_dense(~Element,geometry,fill,layout,input)),
    model_layout(~Element,geometry,fill,layout,Buffer.content(~Element,input)))

# Dense conversion establishes the shared Values relation through its actual
# allocation and channel-copy traversal. Plan geometry and machine bounds are
# explicit obligations; no readback or successful output equation is assumed.
def call(+geometry: Bands.Geometry,first: U32,+rows: U32,capacity: U32,division_capacity: U32) -> BandGather.Call:
  BandGather.Call{geometry,first,rows,U32.to_nat(Bands.local_count(geometry,rows)),capacity,division_capacity}

type LeafFacts<-geometry: Bands.Geometry,-first: U32,-rows: U32,-capacity: U32,-division_capacity: U32> is Data:
  LeafFacts{allocation: {U32.is_le(Bands.local_count(geometry,rows),1073741824) == True{} : Bool},
    contents: Either<&2,&2,{U32.to_nat(Bands.local_count(geometry,rows)) == 0n : Nat},BandGather.Facts<call(geometry,first,rows,capacity,division_capacity)>>,
    metadata: BandShape.Leaf(geometry,first,rows,U32.to_nat(Bands.local_count(geometry,rows)))}

def Fits(+geometry: Bands.Geometry,layout: Bands.Plan,+capacity: U32,+division_capacity: U32) -> Data:
  match layout:
    case Bands.Rows{first,rows}: LeafFacts<geometry,first,rows,capacity,division_capacity>
    case Bands.Split{left,right}: Shape.Both<Fits(geometry,left,capacity,division_capacity),Fits(geometry,right,capacity,division_capacity)>

def leaf_empty(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,
  empty: {U32.to_nat(Bands.local_count(geometry,rows)) == 0n : Nat},metadata: BandShape.Leaf(geometry,first,rows,U32.to_nat(Bands.local_count(geometry,rows)))) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),input,geometry,
    model(~Element,geometry,fill,Bands.Rows{first,rows},Buffer.Content{input,length})):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  +count = Bands.local_count(Bands.Geometry{images,channels,height,width},rows)
  BandValues.empty_leaf(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),input,Bands.Geometry{images,channels,height,width},first,rows,
    write_dense(~Element,Bands.Geometry{images,channels,height,width},first,rows,input,Buffer.tree(~Element,BandCount.model(~Element,count,fill))),U32.to_nat(count),
    Equal.trans(Nat,BandValues.cell_count(Bands.Geometry{images,channels,height,width},rows),U32.to_nat(count),0n,Equal.sym(Nat,U32.to_nat(count),BandValues.cell_count(Bands.Geometry{images,channels,height,width},rows),length_exact),empty))

def leaf_copied(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+capacity: U32,+division_capacity: U32,
  allocation: {U32.is_le(Bands.local_count(geometry,rows),1073741824) == True{} : Bool},facts: BandGather.Facts<call(geometry,first,rows,capacity,division_capacity)>) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),input,geometry,
    model(~Element,geometry,fill,Bands.Rows{first,rows},Buffer.Content{input,length})):
  +count = Bands.local_count(geometry,rows)
  values(~Element,call(geometry,first,rows,capacity,division_capacity),input,BandCount.model(~Element,count,fill),facts,BandCount.bounded_model(~Element,count,fill,allocation),{==})

def semantics_leaf(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+capacity: U32,+division_capacity: U32,
  facts: LeafFacts<geometry,first,rows,capacity,division_capacity>) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),input,geometry,
    model(~Element,geometry,fill,Bands.Rows{first,rows},Buffer.Content{input,length})):
  LeafFacts{allocation,contents,metadata} = facts
  match contents:
    case Inl{empty}: leaf_empty(~Element,geometry,fill,first,rows,input,length,empty,metadata)
    case Inr{facts}: leaf_copied(~Element,geometry,fill,first,rows,input,length,capacity,division_capacity,allocation,facts)

def modeled(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+input: Tree.Tree<Element>,+length: Nat,+capacity: U32,+division_capacity: U32,
  fits: Fits(geometry,layout,capacity,division_capacity)) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),input,geometry,model(~Element,geometry,fill,layout,Buffer.Content{input,length})):
  match layout:
    case Bands.Rows{first,rows}: semantics_leaf(~Element,geometry,fill,first,rows,input,length,capacity,division_capacity,fits)
    case Bands.Split{+left,+right}:
      Shape.Both{left_fits,right_fits} = fits
      Shape.Both{modeled(~Element,geometry,fill,left,input,length,capacity,division_capacity,left_fits),modeled(~Element,geometry,fill,right,input,length,capacity,division_capacity,right_fits)}

def semantics_from_dense(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>,+division_capacity: U32,
  fits: Fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),Buffer.tree(~Element,input),geometry,
    Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))):
  Buffer.Model{+tree,+length,+certificate} = input
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),
    model(~Element,geometry,fill,layout,Buffer.content(~Element,input)),refinement_from_dense(~Element,geometry,fill,layout,input)) :
    BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),tree,geometry,_)
  modeled(~Element,geometry,fill,layout,tree,length,Tree.capacity(Element,tree),division_capacity,fits)

def leaf_ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,input: Buffer.Content<Element>,+capacity: U32,+division_capacity: U32,
  fits: LeafFacts<geometry,first,rows,capacity,division_capacity>) -> Model.ContentReady(~Element,model(~Element,geometry,fill,Bands.Rows{first,rows},input)):
  Bands.Geometry{images,+channels,+height,+width} = geometry
  Buffer.Content{+input,length} = input
  LeafFacts{allocation,contents,metadata} = fits
  +count = Bands.local_count(geometry,rows)
  Writes.content_ready(~Element,BandCopy.writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),
      ((first / height * channels * height + first % height) * width : U32),0,input),
    Buffer.tree(~Element,BandCount.model(~Element,count,fill)),U32.to_nat(count),
    Buffer.content_ready(~Element,BandCount.model(~Element,count,fill),BandCount.bounded_model(~Element,count,fill,allocation)))

def modeled_ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+input: Buffer.Content<Element>,+capacity: U32,+division_capacity: U32,
  fits: Fits(geometry,layout,capacity,division_capacity)) -> Model.ContentReady(~Element,model(~Element,geometry,fill,layout,input)):
  match layout:
    case Bands.Rows{first,rows}: leaf_ready(~Element,geometry,fill,first,rows,input,capacity,division_capacity,fits)
    case Bands.Split{+left,+right}:
      Shape.Both{left_fits,right_fits} = fits
      Shape.Both{modeled_ready(~Element,geometry,fill,left,input,capacity,division_capacity,left_fits),modeled_ready(~Element,geometry,fill,right,input,capacity,division_capacity,right_fits)}

def from_dense_ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>,+division_capacity: U32,
  fits: Fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  Model.ContentReady(~Element,Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),
    model(~Element,geometry,fill,layout,Buffer.content(~Element,input)),refinement_from_dense(~Element,geometry,fill,layout,input)) : Model.ContentReady(~Element,_)
  modeled_ready(~Element,geometry,fill,layout,Buffer.content(~Element,input),Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity,fits)

def modeled_shape(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+input: Buffer.Content<Element>,+capacity: U32,+division_capacity: U32,
  fits: Fits(geometry,layout,capacity,division_capacity)) -> BandShape.Contents(~Element,geometry,model(~Element,geometry,fill,layout,input)):
  match layout:
    case Bands.Rows{first,rows}:
      Buffer.Content{tree,length} = input
      LeafFacts{allocation,contents,metadata} = fits
      metadata
    case Bands.Split{+left,+right}:
      Shape.Both{left_fits,right_fits} = fits
      Shape.Both{modeled_shape(~Element,geometry,fill,left,input,capacity,division_capacity,left_fits),modeled_shape(~Element,geometry,fill,right,input,capacity,division_capacity,right_fits)}

def from_dense_shape(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>,+division_capacity: U32,
  fits: Fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  BandShape.Contents(~Element,geometry,Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),
    model(~Element,geometry,fill,layout,Buffer.content(~Element,input)),refinement_from_dense(~Element,geometry,fill,layout,input)) : BandShape.Contents(~Element,geometry,_)
  modeled_shape(~Element,geometry,fill,layout,Buffer.content(~Element,input),Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity,fits)

# Build dense conversion's copy premises from the actual image-local plan and
# the checked full tensor shape. Empty leaves do not need a source address.
def empty_count(+geometry: Bands.Geometry,+rows: U32,empty: {U32.to_nat(rows) == 0n : Nat}) -> {Bands.local_count(geometry,rows) == 0 : U32}:
  Bands.Geometry{images,+channels,height,+width} = geometry
  +prefix_zero = Equal.trans(U32,(channels * rows : U32),U32.from_nat(Nat.mul(U32.to_nat(channels),U32.to_nat(rows))),0,
    Multiplication.native_multiply(channels,rows),
    Equal.cong(Nat,U32,U32.from_nat,Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),0n,
      %Equal.sym(Nat,U32.to_nat(rows),0n,empty) : {Nat.mul(U32.to_nat(channels),_) == 0n : Nat}
      RowMajor.multiply_zero_right(U32.to_nat(channels))))
  %Equal.sym(U32,(channels * rows : U32),0,prefix_zero) : {(_ * width : U32) == 0 : U32}
  Arithmetic.multiply_zero_left(width)

def empty_leaf(+geometry: Bands.Geometry,+first: U32,+rows: U32,+capacity: U32,+division_capacity: U32,+empty: {U32.to_nat(rows) == 0n : Nat},
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry)) == True{} : Bool}) ->
  LeafFacts<geometry,first,rows,capacity,division_capacity>:
  Bands.Geometry{images,+channels,+height,+width} = geometry
  +count_zero = empty_count(geometry,rows,empty)
  +native_zero = Address.equal_values(rows,0,empty)
  +length_zero = {
    %Equal.sym(Nat,U32.to_nat(rows),0n,empty) : {Nat.mul(U32.to_nat(channels),Nat.mul(_,U32.to_nat(width))) == 0n : Nat}
    RowMajor.multiply_zero_right(U32.to_nat(channels)) : {Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) == 0n : Nat}}
  LeafFacts{
    %Equal.sym(U32,Bands.local_count(geometry,rows),0,count_zero) : {U32.is_le(_,1073741824) == True{} : Bool} {==},
    Inl{Equal.cong(U32,Nat,U32.to_nat,Bands.local_count(geometry,rows),0,count_zero)},
    BandShape.Fields{
      %Equal.sym(U32,rows,0,native_zero) : {U32.is_le(_,(height - first % height : U32)) == True{} : Bool}
      %Equal.sym(Bool,U32.is_le(0,(height - first % height : U32)),Nat.is_le(0n,U32.to_nat((height - first % height : U32))),Comparison.le(0,(height - first % height : U32))) : {_ == True{} : Bool}
      Order.zero_le(U32.to_nat((height - first % height : U32))),rows_fit,
      Equal.trans(Nat,U32.to_nat(Bands.local_count(geometry,rows)),0n,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),
        Equal.cong(U32,Nat,U32.to_nat,Bands.local_count(geometry,rows),0,count_zero),
        Equal.sym(Nat,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),0n,length_zero))}}

def positive_leaf(+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+first: U32,+rows: U32,+capacity: U32,+image: Nat,
  +inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +span: BandPlan.LeafWithin<first,rows,Nat.mul(image,U32.to_nat(height)),Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat(height))>,
  domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>) ->
  LeafFacts<Bands.Geometry{images,channels,height,width},first,rows,capacity,Machine.half_word(limits)>:
  BandGather.Domain{+capacity_ok,+channels_positive,+height_positive,+width_positive,+machine_room,+allocation_room} = domain
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +frame = Nat.mul(U32.to_nat(channels),spatial)
  +total = Nat.mul(U32.to_nat(images),frame)
  +machine_capacity = Machine.capacity(limits)
  +images_positive = Order.le_lt_trans(0n,image,U32.to_nat(images),Order.zero_le(image),inside)
  +frame_room = Order.transitive(frame,total,U32.to_nat(machine_capacity),
    %RowMajor.multiply_commutative(frame,U32.to_nat(images)) : {Nat.is_le(frame,_) == True{} : Bool}
    RowMajor.product_covers(frame,U32.to_nat(images),images_positive),machine_room)
  +spatial_room = Order.transitive(spatial,frame,U32.to_nat(machine_capacity),
    %RowMajor.multiply_commutative(spatial,U32.to_nat(channels)) : {Nat.is_le(spatial,_) == True{} : Bool}
    RowMajor.product_covers(spatial,U32.to_nat(channels),channels_positive),frame_room)
  +height_room = Order.transitive(U32.to_nat(height),spatial,U32.to_nat(machine_capacity),RowMajor.product_covers(U32.to_nat(height),U32.to_nat(width),width_positive),spatial_room)
  +rows_bound = Strided.count_in_stride(Nat.sub(U32.to_nat(first),Nat.mul(image,U32.to_nat(height))),U32.to_nat(rows),U32.to_nat(height),BandPlan.offset_span(first,rows,image,U32.to_nat(height),span))
  +plane = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +length = Nat.mul(U32.to_nat(channels),plane)
  +plane_spatial = RowMajor.mul_le(U32.to_nat(rows),U32.to_nat(height),U32.to_nat(width),rows_bound)
  +length_room = Order.transitive(length,frame,U32.to_nat(machine_capacity),
    %RowMajor.multiply_commutative(plane,U32.to_nat(channels)) : {Nat.is_le(_,frame) == True{} : Bool}
    %RowMajor.multiply_commutative(spatial,U32.to_nat(channels)) : {Nat.is_le(Nat.mul(plane,U32.to_nat(channels)),_) == True{} : Bool}
    RowMajor.mul_le(plane,spatial,U32.to_nat(channels),plane_spatial),frame_room)
  +count = Bands.local_count(geometry,rows)
  +count_exact = Equal.trans(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),U32.to_nat(width)),length,
    Encoded.multiply_three(channels,rows,width,machine_capacity,Limits.capacity_fits(limits),width_positive,
      %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),U32.to_nat(width)),length,RowMajor.assoc(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width))) :
        {Nat.is_le(_,U32.to_nat(machine_capacity)) == True{} : Bool} length_room),RowMajor.assoc(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width)))
  +allocatable = Arithmetic.count_within(count,length,machine_capacity,1073741824,count_exact,length_room,Limits.capacity_allocatable(limits))
  +local_guard = Inspection.local_guard(height,first,rows,Machine.half_word(limits),Limits.half_word_fits(limits),height_positive,Limits.double_bound(limits,U32.to_nat(height),height_room),
    BandPlan.local_span(first,rows,image,U32.to_nat(height),height_positive,rows_positive,span))
  +rows_fit = BandPlan.total_rows(first,rows,image,U32.to_nat(images),U32.to_nat(height),inside,span)
  LeafFacts{allocatable,
    Inr{BandGather.from_leaf_facts(limits,images,channels,height,width,first,rows,U32.to_nat(count),Storage.depth_for(count),capacity,capacity_ok,channels_positive,height_positive,width_positive,rows_positive,machine_room,allocation_room,
      Inspection.LeafFacts{local_guard,rows_fit,count_exact,Depth.allocation_room(count,Depth.bounded(count,allocatable))})},
    BandShape.Fields{local_guard,rows_fit,count_exact}}

def selected_leaf(empty: Bool,+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+first: U32,+rows: U32,+capacity: U32,+image: Nat,
  compared: {U32.is_zero(rows) == empty : Bool},inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},
  +span: BandPlan.LeafWithin<first,rows,Nat.mul(image,U32.to_nat(height)),Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat(height))>,
  domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>) ->
  LeafFacts<Bands.Geometry{images,channels,height,width},first,rows,capacity,Machine.half_word(limits)>:
  match empty:
    case True{}: empty_leaf(Bands.Geometry{images,channels,height,width},first,rows,capacity,Machine.half_word(limits),Equal.cong(U32,Nat,U32.to_nat,rows,0,ShapeLaws.native_equal(rows,0,compared)),BandPlan.total_rows(first,rows,image,U32.to_nat(images),U32.to_nat(height),inside,span))
    case False{}: positive_leaf(images,channels,height,width,limits,first,rows,capacity,image,inside,ContiguousBounds.extent_positive(rows,compared),span,domain)

def bounds_leaf(+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+first: U32,+rows: U32,+capacity: U32,
  local: BandPlan.ImageSpan<first,rows,U32.to_nat(images),U32.to_nat(height)>,within: BandPlan.LeafWithin<first,rows,0n,Nat.mul(U32.to_nat(images),U32.to_nat(height))>,domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>) ->
  LeafFacts<Bands.Geometry{images,channels,height,width},first,rows,capacity,Machine.half_word(limits)>:
  match local:
    case BandPlan.EmptySpan{empty}:
      BandPlan.LeafWithin{lower,upper} = within
      empty_leaf(Bands.Geometry{images,channels,height,width},first,rows,capacity,Machine.half_word(limits),empty,upper)
    case BandPlan.ImageSpan{image,inside,span}: selected_leaf(U32.is_zero(rows),images,channels,height,width,limits,first,rows,capacity,image,{==},inside,span,domain)

def fits(layout: Bands.Plan,+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,
  local: BandPlan.ImageLocal(layout,U32.to_nat(images),U32.to_nat(height)),within: BandPlan.Within(layout,0n,Nat.mul(U32.to_nat(images),U32.to_nat(height))),+domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>) ->
  Fits(Bands.Geometry{images,channels,height,width},layout,capacity,Machine.half_word(limits)):
  match layout:
    case Bands.Rows{first,rows}: bounds_leaf(images,channels,height,width,limits,first,rows,capacity,local,within,domain)
    case Bands.Split{+left,+right}:
      Shape.Both{left_local,right_local} = local
      Shape.Both{left_within,right_within} = within
      Shape.Both{fits(left,images,channels,height,width,limits,capacity,left_local,left_within,domain),fits(right,images,channels,height,width,limits,capacity,right_local,right_within,domain)}

def split_plan(geometry: Bands.Geometry,depth: Nat) -> Bands.Plan:
  Bands.Geometry{images,channels,height,width} = geometry
  Bands.image_plan(U32.to_nat(images),0,depth,height)

def image_plan(+geometry: Bands.Geometry,+limits: Machine.Limits,+depth: Nat,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>) ->
  Fits(geometry,split_plan(geometry,depth),capacity,Machine.half_word(limits)):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandGather.Domain{capacity_ok,channels_positive,+height_positive,width_positive,machine_room,allocation_room} = domain
  +row_room = BandGather.rows_room(geometry,limits,capacity,domain)
  +images_room = Order.transitive(U32.to_nat(images),Nat.mul(U32.to_nat(images),U32.to_nat(height)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(images),U32.to_nat(height),height_positive),row_room)
  fits(Bands.image_plan(U32.to_nat(images),0,depth,height),images,channels,height,width,limits,capacity,
    BandPlan.bounds_image_plan(U32.to_nat(images),0,depth,height,U32.to_nat(images),Machine.capacity(limits),Limits.capacity_fits(limits),Order.reflexive(U32.to_nat(images)),
      images_room,row_room),BandPlan.image_within(images,height,depth,Machine.capacity(limits),Limits.capacity_fits(limits),images_room,row_room),domain)

def bounds_split(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+depth: Nat,+fill: Element,+input: Buffer.Model<Element>,
  domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,input))>) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),Buffer.tree(~Element,input),geometry,
    Model.contents(~Element,Bands.split(~Element,depth,geometry,fill,Buffer.pack(~Element,input)))):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  semantics_from_dense(~Element,geometry,fill,Bands.image_plan(U32.to_nat(images),0,depth,height),input,Machine.half_word(limits),image_plan(geometry,limits,depth,Tree.capacity(Element,Buffer.tree(~Element,input)),domain))

def split_ready(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+depth: Nat,+fill: Element,+input: Buffer.Model<Element>,
  domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,input))>) ->
  Model.ContentReady(~Element,Model.contents(~Element,Bands.split(~Element,depth,geometry,fill,Buffer.pack(~Element,input)))):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  from_dense_ready(~Element,geometry,fill,Bands.image_plan(U32.to_nat(images),0,depth,height),input,Machine.half_word(limits),image_plan(geometry,limits,depth,Tree.capacity(Element,Buffer.tree(~Element,input)),domain))

def split_shape(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+depth: Nat,+fill: Element,+input: Buffer.Model<Element>,
  domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,input))>) ->
  BandShape.Contents(~Element,geometry,Model.contents(~Element,Bands.split(~Element,depth,geometry,fill,Buffer.pack(~Element,input)))):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  from_dense_shape(~Element,geometry,fill,Bands.image_plan(U32.to_nat(images),0,depth,height),input,Machine.half_word(limits),image_plan(geometry,limits,depth,Tree.capacity(Element,Buffer.tree(~Element,input)),domain))

def split_well_formed(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+depth: Nat,+fill: Element,+input: Buffer.Model<Element>,
  +domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,input))>) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.split(~Element,depth,geometry,fill,Buffer.pack(~Element,input)))>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandGather.Domain{capacity_ok,channels_positive,+height_positive,width_positive,machine_room,allocation_room} = domain
  +row_room = BandGather.rows_room(geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,input)),domain)
  +images_room = Order.transitive(U32.to_nat(images),Nat.mul(U32.to_nat(images),U32.to_nat(height)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(images),U32.to_nat(height),height_positive),row_room)
  BandShape.WellFormed{split_shape(~Element,geometry,limits,depth,fill,input,domain),split_ready(~Element,geometry,limits,depth,fill,input,domain),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Bands.split(~Element,depth,geometry,fill,Buffer.pack(~Element,input)))),
      Bands.image_plan(U32.to_nat(images),0,depth,height),from_dense_layout(~Element,geometry,fill,Bands.image_plan(U32.to_nat(images),0,depth,height),input)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    BandPlan.image_coverage(images,height,depth,Machine.capacity(limits),Limits.capacity_fits(limits),images_room,row_room)}

# Replace a whole-tensor allocation envelope with the actual dense owner's
# capacity while retaining the independently proved machine envelope.
def owned(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+input: Buffer.Model<Element>,
  expected_length: {BandGather.elements(geometry) == Buffer.length(~Element,input) : Nat},ready: Buffer.Ready<Element,input>,whole: BandGather.Domain<geometry,limits,capacity>) ->
  BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,input))>:
  match ready whole:
    case Buffer.Ready{+depth_ok,room} BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room}:
      BandGather.Domain{Buffer.capacity_bound(~Element,input,depth_ok),channels_positive,height_positive,width_positive,machine_room,
        %Equal.sym(Nat,BandGather.elements(geometry),Buffer.length(~Element,input),expected_length) :
          {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,input)))) == True{} : Bool}
        room}

# A checked existing owner supplies the target allocation and copy premises.
# Empty row leaves require no address; positive leaves use actual metadata.
def positive(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+capacity: U32,
  +domain: BandGather.Domain<geometry,limits,capacity>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Ready<Element,input>,
  rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},counted: BandShape.Count<geometry,rows,Buffer.length(~Element,input)>) ->
  LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>:
  BandShape.Count{+exact,allocatable} = counted
  LeafFacts{allocatable,
    Inr{
      %Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),Buffer.length(~Element,input),exact) :
        BandGather.Facts<BandGather.Call{geometry,first,rows,_,capacity,Machine.half_word(limits)}>
      BandGather.from_metadata(~Element,geometry,limits,first,rows,input,capacity,domain,metadata,rows_positive,ready)},
    %Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),Buffer.length(~Element,input),exact) : BandShape.Leaf(geometry,first,rows,_)
    metadata}

def selected(~Element: Data,empty: Bool,+geometry: Bands.Geometry,+limits: Machine.Limits,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+capacity: U32,
  compared: {U32.is_zero(rows) == empty : Bool},+domain: BandGather.Domain<geometry,limits,capacity>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),
  +ready: Buffer.Ready<Element,input>) -> LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>:
  match empty geometry:
    case True{} Bands.Geometry{images,channels,height,width}:
      BandShape.Fields{local,rows_fit,length_exact} = metadata
      empty_leaf(geometry,first,rows,capacity,Machine.half_word(limits),Equal.cong(U32,Nat,U32.to_nat,rows,0,ShapeLaws.native_equal(rows,0,compared)),rows_fit)
    case False{} Bands.Geometry{+images,+channels,+height,+width}:
      BandGather.Domain{capacity_ok,channels_positive,height_positive,+width_positive,machine_room,allocation_room} = domain
      +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
      positive(~Element,geometry,limits,first,rows,input,capacity,domain,metadata,ready,ContiguousBounds.extent_positive(rows,compared),
        BandShape.count(~Element,geometry,first,rows,input,width_positive,metadata,ready))

def target(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,bands: Model.Model<Element>,+domain: BandGather.Domain<geometry,limits,capacity>,
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands)) ->
  Fits(geometry,Model.layout(~Element,bands),capacity,Machine.half_word(limits)):
  match bands:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
      selected(~Element,U32.is_zero(rows),geometry,limits,first,rows,input,capacity,{==},domain,metadata,ready)
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{target(~Element,geometry,limits,capacity,left,domain,left_metadata,left_ready),target(~Element,geometry,limits,capacity,right,domain,right_metadata,right_ready)}

# Empty band conversions need no address envelope. Actual source metadata
# supplies row ownership, and each zero axis supplies the native zero count.
def empty_plan_fits_leaf(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,+capacity: U32,+division_capacity: U32,
  +native_zero: {Bands.local_count(geometry,rows) == 0 : U32},metadata: BandShape.Leaf(geometry,first,rows,length),counted: BandShape.Count<geometry,rows,length>) ->
  LeafFacts<geometry,first,rows,capacity,division_capacity>:
  Bands.Geometry{images,channels,height,width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  BandShape.Count{+exact,allocatable} = counted
  LeafFacts{allocatable,Inl{Equal.cong(U32,Nat,U32.to_nat,Bands.local_count(geometry,rows),0,native_zero)},
    %Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),length,exact) : BandShape.Leaf(geometry,first,rows,_)
    BandShape.Fields{local,rows_fit,length_exact}}

def width(~Element: Data,+images: U32,+channels: U32,+height: U32,bands: Model.Model<Element>,+capacity: U32,+division_capacity: U32,
  metadata: BandShape.Contents(~Element,Bands.Geometry{images,channels,height,0},Model.content(~Element,bands))) ->
  Fits(Bands.Geometry{images,channels,height,0},Model.layout(~Element,bands),capacity,division_capacity):
  match bands:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
      +metadata = metadata
      empty_plan_fits_leaf(Bands.Geometry{images,channels,height,0},first,rows,length,capacity,division_capacity,BandCount.width_native(images,channels,height,rows),metadata,
        BandCount.width(~Element,images,channels,height,first,rows,input,metadata))
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{width(~Element,images,channels,height,left,capacity,division_capacity,left_metadata),width(~Element,images,channels,height,right,capacity,division_capacity,right_metadata)}

def channels(~Element: Data,+images: U32,+height: U32,+width: U32,bands: Model.Model<Element>,+capacity: U32,+division_capacity: U32,
  metadata: BandShape.Contents(~Element,Bands.Geometry{images,0,height,width},Model.content(~Element,bands))) ->
  Fits(Bands.Geometry{images,0,height,width},Model.layout(~Element,bands),capacity,division_capacity):
  match bands:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
      +metadata = metadata
      empty_plan_fits_leaf(Bands.Geometry{images,0,height,width},first,rows,length,capacity,division_capacity,BandCount.channels_native(images,height,width,rows),metadata,
        BandCount.channels(~Element,images,height,width,first,rows,input,metadata))
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{channels(~Element,images,height,width,left,capacity,division_capacity,left_metadata),channels(~Element,images,height,width,right,capacity,division_capacity,right_metadata)}

def no_rows_natural(+first: Nat,+rows: Nat,bound: {Nat.is_le(Nat.add(first,rows),0n) == True{} : Bool}) -> {rows == 0n : Nat}:
  match first rows:
    case 0n 0n: {==}
    case 0n 1n+rows: Empty.absurd({1n+rows == 0n : Nat},WordEncoding.false_true(bound))
    case 1n+first rows: Empty.absurd({rows == 0n : Nat},WordEncoding.false_true(bound))

def no_rows_leaf(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,+capacity: U32,+division_capacity: U32,
  total_zero: {Inspection.total_rows(geometry) == 0n : Nat},metadata: BandShape.Leaf(geometry,first,rows,length)) ->
  LeafFacts<geometry,first,rows,capacity,division_capacity>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandShape.Fields{local,+rows_fit,length_exact} = metadata
  empty_leaf(Bands.Geometry{images,channels,height,width},first,rows,capacity,division_capacity,
    no_rows_natural(U32.to_nat(first),U32.to_nat(rows),%total_zero : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool} rows_fit),rows_fit)

def no_rows(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>,+capacity: U32,+division_capacity: U32,
  +total_zero: {Inspection.total_rows(geometry) == 0n : Nat},metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands))) ->
  Fits(geometry,Model.layout(~Element,bands),capacity,division_capacity):
  match bands:
    case Model.Leaf{Buffer.Model{tree,+length,certificate},+first,+rows}: no_rows_leaf(geometry,first,rows,length,capacity,division_capacity,total_zero,metadata)
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{no_rows(~Element,geometry,left,capacity,division_capacity,total_zero,left_metadata),no_rows(~Element,geometry,right,capacity,division_capacity,total_zero,right_metadata)}

# Real split followed by gather preserves every valid dense NCHW value.
# Planner coverage, empty leaves and buffer certificates are discharged by
# the producer. Only the tensor and machine domain remains at this boundary.
def split_gather(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+depth: Nat,+fill: Element,+input: Buffer.Model<Element>,+result: Buffer.Model<Element>,
  +input_domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,input))>,
  result_domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>,
  +image: Nat,+channel: Nat,+row: Nat,+column: Nat,inside: BandGather.Inside(geometry,image,channel,row,column),
  depth_ok: {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}) ->
  {Buffer.value(~Element,U32.from_nat(BandGather.point(geometry,image,channel,row,column)),
    Buffer.contents(~Element,Bands.gather(~Element,geometry,Bands.split(~Element,depth,geometry,fill,Buffer.pack(~Element,input)),Buffer.pack(~Element,result)))) ==
    Tree.read(Element,Buffer.tree(~Element,input),U32.from_nat(BandGather.point(geometry,image,channel,row,column))) : Element}:
  BandGather.observed_value(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),Buffer.tree(~Element,input),geometry,limits,
    Bands.split(~Element,depth,geometry,fill,Buffer.pack(~Element,input)),Model.observe(~Element,Bands.split(~Element,depth,geometry,fill,Buffer.pack(~Element,input))),result,result_domain,
    split_well_formed(~Element,geometry,limits,depth,fill,input,input_domain),bounds_split(~Element,geometry,limits,depth,fill,input,input_domain),image,channel,row,column,inside,depth_ok)

# The actual row planner stops before creating empty children. Nonempty
# images therefore provide the positive-core premise of halo preparation.
type SplitFacts<-count: U32,-capacity: U32> is Data:
  SplitFacts{left_positive: {Nat.is_lt(0n,U32.to_nat((count / 2 : U32))) == True{} : Bool},
    right_positive: {Nat.is_lt(0n,U32.to_nat((count - count / 2 : U32))) == True{} : Bool},
    left_room: {Nat.is_le(U32.to_nat((count / 2 : U32)),U32.to_nat(capacity)) == True{} : Bool},
    right_room: {Nat.is_le(U32.to_nat((count - count / 2 : U32)),U32.to_nat(capacity)) == True{} : Bool}}

def plan_nonempty_split(+count: U32,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool},
  enabled: {U32.is_lt(1,count) == True{} : Bool}) -> SplitFacts<count,capacity>:
  +half = (count / 2 : U32)
  +quotient = Nat.div(U32.to_nat(count),2n)
  +decoded = U32Division.divide(count,2,2147483648,{==},{==},{==})
  +at_least_two = Order.strict_follow(1n,U32.to_nat(count),{%Comparison.lt(1,count) : {_ == True{} : Bool} enabled : {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool}})
  +quotient_positive = Bounds.predecessor_strict(0n,quotient,DivisionOrder.quotient_le(2n,U32.to_nat(count),2n,{==},at_least_two))
  +half_inside = BandPlan.half_bound(count)
  +quotient_less = Bounds.nat_trans(quotient,Nat.mul(quotient,2n),U32.to_nat(count),
    %Equal.sym(Nat,Nat.mul(quotient,2n),Nat.mul(2n,quotient),RowMajor.multiply_commutative(quotient,2n)) : {Nat.is_lt(quotient,_) == True{} : Bool}
    %Addition.zero_right(quotient) : {Nat.is_lt(_,Nat.mul(2n,quotient)) == True{} : Bool}
    RowMajor.multiply_lt(1n,2n,quotient,{==},quotient_positive),DivisionOrder.product_below(U32.to_nat(count),2n,{==}))
  +remainder_room = Order.transitive(Nat.sub(U32.to_nat(count),U32.to_nat(half)),U32.to_nat(count),U32.to_nat(capacity),Order.sub_le(U32.to_nat(count),U32.to_nat(half)),room)
  +remainder_exact = Arithmetic.subtract(count,half,capacity,capacity_ok,half_inside,remainder_room)
  SplitFacts{
    %Equal.sym(Nat,U32.to_nat(half),quotient,decoded) : {Nat.is_lt(0n,_) == True{} : Bool} quotient_positive,
    %Equal.sym(Nat,U32.to_nat((count - half : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(half)),remainder_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.positive_difference(U32.to_nat(count),U32.to_nat(half),{%Equal.sym(Nat,U32.to_nat(half),quotient,decoded) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} quotient_less : {Nat.is_lt(U32.to_nat(half),U32.to_nat(count)) == True{} : Bool}}),
    Order.transitive(U32.to_nat(half),U32.to_nat(count),U32.to_nat(capacity),half_inside,room),
    %Equal.sym(Nat,U32.to_nat((count - half : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(half)),remainder_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} remainder_room}

def children(+depth: Nat,+first: U32,+count: U32,+capacity: U32,facts: SplitFacts<count,capacity>,
  left: {Nat.is_lt(0n,U32.to_nat((count / 2 : U32))) == True{} : Bool} -> {Nat.is_le(U32.to_nat((count / 2 : U32)),U32.to_nat(capacity)) == True{} : Bool} -> BandPlan.Nonempty(Bands.row_plan(depth,U32.is_lt(1,(count / 2 : U32)),first,(count / 2 : U32))),
  right: {Nat.is_lt(0n,U32.to_nat((count - count / 2 : U32))) == True{} : Bool} -> {Nat.is_le(U32.to_nat((count - count / 2 : U32)),U32.to_nat(capacity)) == True{} : Bool} -> BandPlan.Nonempty(Bands.row_plan(depth,U32.is_lt(1,(count - count / 2 : U32)),(first + count / 2 : U32),(count - count / 2 : U32)))) ->
  BandPlan.Nonempty(Bands.row_plan(1n+depth,True{},first,count)):
  SplitFacts{left_positive,right_positive,left_room,right_room} = facts
  Shape.Both{left(left_positive,left_room),right(right_positive,right_room)}

def rows(depth: Nat,+enabled: Bool,+first: U32,+count: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool},positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  comparison: {U32.is_lt(1,count) == enabled : Bool}) -> BandPlan.Nonempty(Bands.row_plan(depth,enabled,first,count)):
  match depth enabled:
    case 0n enabled: positive
    case 1n+rest False{}: positive
    case 1n+ +rest True{}:
      children(rest,first,count,capacity,plan_nonempty_split(count,capacity,capacity_ok,room,comparison),
        positive => room => rows(rest,U32.is_lt(1,(count / 2 : U32)),first,(count / 2 : U32),capacity,capacity_ok,room,positive,{==}),
        positive => room => rows(rest,U32.is_lt(1,(count - count / 2 : U32)),(first + count / 2 : U32),(count - count / 2 : U32),capacity,capacity_ok,room,positive,{==}))

def plan_nonempty_images(count: Nat,+image: U32,+depth: Nat,+height: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(capacity)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  positive: {Nat.is_lt(0n,count) == True{} : Bool}) -> BandPlan.Nonempty(Bands.image_plan(count,image,depth,height)):
  match count:
    case 0n: Empty.absurd(BandPlan.Nonempty(Bands.Rows{0,0}),WordEncoding.false_true(positive))
    case 1n: rows(depth,U32.is_lt(1,height),(image * height : U32),height,capacity,capacity_ok,height_room,height_positive,{==})
    case 2n+ +rest:
      Shape.Both{rows(depth,U32.is_lt(1,height),(image * height : U32),height,capacity,capacity_ok,height_room,height_positive,{==}),
        plan_nonempty_images(1n+rest,(image + 1 : U32),depth,height,capacity,capacity_ok,height_room,height_positive,{==})}

# This premise is the nonzero-batch branch of a producer, not a restriction on
# the public tensor API. The zero-batch branch remains a separate caller case.
def dense_plan(+image_count: U32,+channels: U32,+height: U32,+width: U32,+depth: Nat,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<Bands.Geometry{image_count,channels,height,width},limits,capacity>,+positive: {Nat.is_lt(0n,U32.to_nat(image_count)) == True{} : Bool}) ->
  BandPlan.Nonempty(Bands.image_plan(U32.to_nat(image_count),0,depth,height)):
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  +height_room = Order.transitive(U32.to_nat(height),Nat.mul(U32.to_nat(image_count),U32.to_nat(height)),U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(U32.to_nat(height),U32.to_nat(image_count)) : {Nat.is_le(U32.to_nat(height),_) == True{} : Bool}
    RowMajor.product_covers(U32.to_nat(height),U32.to_nat(image_count),positive),BandGather.rows_room(Bands.Geometry{image_count,channels,height,width},limits,capacity,domain))
  plan_nonempty_images(U32.to_nat(image_count),0,depth,height,Machine.capacity(limits),Limits.capacity_fits(limits),height_room,height_positive,positive)

def dense_split(~Element: Data,+image_count: U32,+channels: U32,+height: U32,+width: U32,+depth: Nat,+fill: Element,+input: Buffer.Model<Element>,+limits: Machine.Limits,
  domain: BandGather.Domain<Bands.Geometry{image_count,channels,height,width},limits,Tree.capacity(Element,Buffer.tree(~Element,input))>,positive: {Nat.is_lt(0n,U32.to_nat(image_count)) == True{} : Bool}) ->
  BandPlan.Nonempty(Model.content_layout(~Element,Model.contents(~Element,Bands.split(~Element,depth,Bands.Geometry{image_count,channels,height,width},fill,Buffer.pack(~Element,input))))):
  +geometry = {Bands.Geometry{image_count,channels,height,width} : Bands.Geometry}
  +plan = Bands.image_plan(U32.to_nat(image_count),0,depth,height)
  %Equal.sym(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Bands.split(~Element,depth,geometry,fill,Buffer.pack(~Element,input)))),plan,
    from_dense_layout(~Element,geometry,fill,plan,input)) : BandPlan.Nonempty(_)
  dense_plan(image_count,channels,height,width,depth,limits,Tree.capacity(Element,Buffer.tree(~Element,input)),domain,positive)
