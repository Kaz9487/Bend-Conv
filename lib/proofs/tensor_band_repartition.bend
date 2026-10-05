# Repartition into requested row bands through the shared range copies.
import Base
import ./tensor_band_range.bend as BandRange
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_model.bend as Model
import ../tensor_band.bend as Bands
import ../machine_limits.bend as Machine
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./ordered_ranges.bend as Ranges
import ./row_major_coordinates.bend as RowCoordinates
import ./sequence_membership.bend as Members
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_shape.bend as Shape
import ./tensor_band_count.bend as BandCount
import ./tensor_band_dense.bend as BandDense
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./write_program.bend as Writes
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition

# Repartition allocates each requested row band and uses the actual range
# reader. Source content is preserved between leaves; certificates may vary.
type Content<-Element: Data> is Data:
  Content{source: Model.Content<Element>,result: Model.Content<Element>}

def contents(~Element: Data,owners: Bands.Bands(~Element) & Bands.Bands(~Element)) -> Content<Element>:
  (source,result) = owners
  Content{Model.contents(~Element,source),Model.contents(~Element,result)}

def result(~Element: Data,content: Content<Element>) -> Model.Content<Element>:
  Content{source,result} = content
  result

def leaf_buffer(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,source: Model.Content<Element>) -> Buffer.Content<Element>:
  +count = Bands.local_count(geometry,rows)
  BandRange.result(~Element,BandRange.written(~Element,BandRange.Request{geometry,first,rows,0},source,Buffer.Content{Tree.fresh(Element,Storage.depth_for(count),fill),U32.to_nat(count)}))

def leaf_model(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,source: Model.Content<Element>) -> Model.Content<Element>:
  Model.RowBand{leaf_buffer(~Element,geometry,fill,first,rows,source),first,rows}

def model(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Content<Element>) -> Model.Content<Element>:
  match layout:
    case Bands.Rows{first,rows}: leaf_model(~Element,geometry,fill,first,rows,source)
    case Bands.Split{left,right}: Model.Joined{model(~Element,geometry,fill,left,source),model(~Element,geometry,fill,right,source)}

def ranged(~Element: Data,first: U32,rows: U32,content: BandRange.Content<Element>) -> Content<Element>:
  BandRange.Content{source,output} = content
  Content{source,Model.RowBand{output,first,rows}}

def range_result(~Element: Data,first: U32,rows: U32,owners: Bands.Bands(~Element) & Storage.Buffer<Element>) -> Bands.Bands(~Element) & Bands.Bands(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Bands.Bands(~Element) & Bands.Bands(~Element)),owners,
    source => output => (source,Partition.Leaf{output,first,rows,Unit{}}))

def range_contents(~Element: Data,+first: U32,+rows: U32,owners: Bands.Bands(~Element) & Storage.Buffer<Element>) ->
  {contents(~Element,range_result(~Element,first,rows,owners)) == ranged(~Element,first,rows,BandRange.contents(~Element,owners)) : Content<Element>}:
  (source,output) = owners
  {==}

def leaf(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+source: Model.Model<Element>) ->
  {contents(~Element,Bands.repartition(~Element,geometry,fill,Bands.Rows{first,rows},Model.pack(~Element,source))) ==
    Content{Model.content(~Element,source),leaf_model(~Element,geometry,fill,first,rows,Model.content(~Element,source))} : Content<Element>}:
  +request = {BandRange.Request{geometry,first,rows,0} : BandRange.Request}
  +count = Bands.local_count(geometry,rows)
  Equal.trans(Content<Element>,contents(~Element,Bands.repartition(~Element,geometry,fill,Bands.Rows{first,rows},Model.pack(~Element,source))),
    ranged(~Element,first,rows,BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count)))),
    Content{Model.content(~Element,source),leaf_model(~Element,geometry,fill,first,rows,Model.content(~Element,source))},
    range_contents(~Element,first,rows,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count))),
    Equal.trans(Content<Element>,ranged(~Element,first,rows,BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count)))),
      ranged(~Element,first,rows,BandRange.written(~Element,request,Model.content(~Element,source),Buffer.contents(~Element,Bands.allocate(~Element,fill,count)))),
      Content{Model.content(~Element,source),leaf_model(~Element,geometry,fill,first,rows,Model.content(~Element,source))},
      Equal.cong(BandRange.Content<Element>,Content<Element>,content => ranged(~Element,first,rows,content),
        BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count))),BandRange.written(~Element,request,Model.content(~Element,source),Buffer.contents(~Element,Bands.allocate(~Element,fill,count))),
        BandRange.native_range(~Element,request,source,Bands.allocate(~Element,fill,count))),
      Equal.cong(Buffer.Content<Element>,Content<Element>,output => ranged(~Element,first,rows,BandRange.written(~Element,request,Model.content(~Element,source),output)),
        Buffer.contents(~Element,Bands.allocate(~Element,fill,count)),Buffer.Content{Tree.fresh(Element,Storage.depth_for(count),fill),U32.to_nat(count)},AllocationLaws.contents(~Element,Storage.depth_for(count),fill,U32.to_nat(count)))))

def joined(~Element: Data,left: Model.Content<Element>,right: Content<Element>) -> Content<Element>:
  Content{source,right} = right
  Content{source,Model.Joined{left,right}}

def join_native(~Element: Data,left: Bands.Bands(~Element),right: Bands.Bands(~Element) & Bands.Bands(~Element)) -> Bands.Bands(~Element) & Bands.Bands(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~Bands.Bands(~Element),~(Bands.Bands(~Element) & Bands.Bands(~Element)),right,source => right => (source,Partition.Fork{left,right}))

def joined_contents(~Element: Data,left: Bands.Bands(~Element),right: Bands.Bands(~Element) & Bands.Bands(~Element)) ->
  {contents(~Element,join_native(~Element,left,right)) == joined(~Element,Model.contents(~Element,left),contents(~Element,right)) : Content<Element>}:
  (source,right) = right
  {==}

def continued_model(~Element: Data,geometry: Bands.Geometry,fill: Element,layout: Bands.Plan,left: Content<Element>) -> Content<Element>:
  Content{+source,left} = left
  Content{source,Model.Joined{left,model(~Element,geometry,fill,layout,source)}}

def continued(~Element: Data,geometry: Bands.Geometry,fill: Element,layout: Bands.Plan,left: Bands.Bands(~Element) & Bands.Bands(~Element)) -> Bands.Bands(~Element) & Bands.Bands(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~Bands.Bands(~Element),~(Bands.Bands(~Element) & Bands.Bands(~Element)),left,
    source => left => join_native(~Element,left,Bands.repartition(~Element,geometry,fill,layout,source)))

def continue_packed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>,+left: Model.Model<Element>,
  rule: @input: Bands.Bands(~Element) -> {contents(~Element,Bands.repartition(~Element,geometry,fill,layout,input)) ==
    Content{Model.contents(~Element,input),model(~Element,geometry,fill,layout,Model.contents(~Element,input))} : Content<Element>}) ->
  {contents(~Element,continued(~Element,geometry,fill,layout,(Model.pack(~Element,source),Model.pack(~Element,left)))) ==
    continued_model(~Element,geometry,fill,layout,contents(~Element,(Model.pack(~Element,source),Model.pack(~Element,left)))) : Content<Element>}:
  Equal.trans(Content<Element>,contents(~Element,join_native(~Element,Model.pack(~Element,left),Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),
    joined(~Element,Model.contents(~Element,Model.pack(~Element,left)),contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),
    continued_model(~Element,geometry,fill,layout,contents(~Element,(Model.pack(~Element,source),Model.pack(~Element,left)))),
    joined_contents(~Element,Model.pack(~Element,left),Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))),
    Equal.cong(Content<Element>,Content<Element>,right => joined(~Element,Model.contents(~Element,Model.pack(~Element,left)),right),
      contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))),Content{Model.contents(~Element,Model.pack(~Element,source)),model(~Element,geometry,fill,layout,Model.contents(~Element,Model.pack(~Element,source)))},rule(Model.pack(~Element,source))))

def continue_observed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,-source: Bands.Bands(~Element),-left: Bands.Bands(~Element),
  source_observation: Model.Observation<Element,source>,left_observation: Model.Observation<Element,left>,
  rule: @input: Bands.Bands(~Element) -> {contents(~Element,Bands.repartition(~Element,geometry,fill,layout,input)) == Content{Model.contents(~Element,input),model(~Element,geometry,fill,layout,Model.contents(~Element,input))} : Content<Element>}) ->
  {contents(~Element,continued(~Element,geometry,fill,layout,(source,left))) == continued_model(~Element,geometry,fill,layout,contents(~Element,(source,left))) : Content<Element>}:
  Model.Observation{+source_model,source_equal} = source_observation
  Model.Observation{+left_model,left_equal} = left_observation
  %source_equal : {contents(~Element,continued(~Element,geometry,fill,layout,(_,left))) == continued_model(~Element,geometry,fill,layout,contents(~Element,(_,left))) : Content<Element>}
  %left_equal : {contents(~Element,continued(~Element,geometry,fill,layout,(Model.pack(~Element,source_model),_))) == continued_model(~Element,geometry,fill,layout,contents(~Element,(Model.pack(~Element,source_model),_))) : Content<Element>}
  continue_packed(~Element,geometry,fill,layout,source_model,left_model,rule)

def continuation(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,left: Bands.Bands(~Element) & Bands.Bands(~Element),
  rule: @input: Bands.Bands(~Element) -> {contents(~Element,Bands.repartition(~Element,geometry,fill,layout,input)) == Content{Model.contents(~Element,input),model(~Element,geometry,fill,layout,Model.contents(~Element,input))} : Content<Element>}) ->
  {contents(~Element,continued(~Element,geometry,fill,layout,left)) == continued_model(~Element,geometry,fill,layout,contents(~Element,left)) : Content<Element>}:
  (source,left) = left
  continue_observed(~Element,geometry,fill,layout,source,left,Model.observe(~Element,source),Model.observe(~Element,left),rule)

type Context<-Element: Data> is Data:
  Context{geometry: Bands.Geometry,fill: Element,layout: Bands.Plan}

def proposition(~Element: Data,context: Context<Element>,-input: Bands.Bands(~Element)) -> Type:
  Context{+geometry,+fill,+layout} = context
  {contents(~Element,Bands.repartition(~Element,geometry,fill,layout,input)) == Content{Model.contents(~Element,input),model(~Element,geometry,fill,layout,Model.contents(~Element,input))} : Content<Element>}

def repartition(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,input: Bands.Bands(~Element)) ->
  {contents(~Element,Bands.repartition(~Element,geometry,fill,layout,input)) == Content{Model.contents(~Element,input),model(~Element,geometry,fill,layout,Model.contents(~Element,input))} : Content<Element>}:
  match layout:
    case Bands.Rows{+first,+rows}:
      Model.lift(~Element,~Context<Element>,~proposition(~Element),Context{geometry,fill,Bands.Rows{first,rows}},input,Model.observe(~Element,input),source =>
        +source = source
        %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,source)),Model.content(~Element,source),Model.packed_contents(~Element,source)) :
          {contents(~Element,Bands.repartition(~Element,geometry,fill,Bands.Rows{first,rows},Model.pack(~Element,source))) == Content{_,model(~Element,geometry,fill,Bands.Rows{first,rows},_)} : Content<Element>}
        leaf(~Element,geometry,fill,first,rows,source))
    case Bands.Split{+left,+right}:
      Model.lift(~Element,~Context<Element>,~proposition(~Element),Context{geometry,fill,Bands.Split{left,right}},input,Model.observe(~Element,input),source =>
        +source = source
        Equal.trans(Content<Element>,contents(~Element,continued(~Element,geometry,fill,right,Bands.repartition(~Element,geometry,fill,left,Model.pack(~Element,source)))),
          continued_model(~Element,geometry,fill,right,contents(~Element,Bands.repartition(~Element,geometry,fill,left,Model.pack(~Element,source)))),
          Content{Model.contents(~Element,Model.pack(~Element,source)),model(~Element,geometry,fill,Bands.Split{left,right},Model.contents(~Element,Model.pack(~Element,source)))},
          continuation(~Element,geometry,fill,right,Bands.repartition(~Element,geometry,fill,left,Model.pack(~Element,source)),input => repartition(~Element,geometry,fill,right,input)),
          Equal.cong(Content<Element>,Content<Element>,left => continued_model(~Element,geometry,fill,right,left),
            contents(~Element,Bands.repartition(~Element,geometry,fill,left,Model.pack(~Element,source))),
            Content{Model.contents(~Element,Model.pack(~Element,source)),model(~Element,geometry,fill,left,Model.contents(~Element,Model.pack(~Element,source)))},repartition(~Element,geometry,fill,left,Model.pack(~Element,source)))))

def result_contents(~Element: Data,owners: Bands.Bands(~Element) & Bands.Bands(~Element)) ->
  {Model.contents(~Element,Bands.repartitioned(~Element,owners)) == result(~Element,contents(~Element,owners)) : Model.Content<Element>}:
  (source,result) = owners
  {==}

def repartitioned(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>) ->
  {Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))) ==
    model(~Element,geometry,fill,layout,Model.content(~Element,source)) : Model.Content<Element>}:
  Equal.trans(Model.Content<Element>,Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),
    result(~Element,contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),model(~Element,geometry,fill,layout,Model.content(~Element,source)),
    result_contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))),
    %Model.packed_contents(~Element,source) : {result(~Element,contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))) == model(~Element,geometry,fill,layout,_) : Model.Content<Element>}
    Equal.cong(Content<Element>,Model.Content<Element>,owners => result(~Element,owners),contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))),
      Content{Model.contents(~Element,Model.pack(~Element,source)),model(~Element,geometry,fill,layout,Model.contents(~Element,Model.pack(~Element,source)))},repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))))

def model_layout(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Content<Element>) ->
  {Model.content_layout(~Element,model(~Element,geometry,fill,layout,source)) == layout : Bands.Plan}:
  match layout:
    case Bands.Rows{first,rows}: {==}
    case Bands.Split{+left,+right}:
      %model_layout(~Element,geometry,fill,left,source) :
        {Bands.Split{Model.content_layout(~Element,model(~Element,geometry,fill,left,source)),Model.content_layout(~Element,model(~Element,geometry,fill,right,source))} == Bands.Split{_,right} : Bands.Plan}
      %model_layout(~Element,geometry,fill,right,source) :
        {Bands.Split{Model.content_layout(~Element,model(~Element,geometry,fill,left,source)),Model.content_layout(~Element,model(~Element,geometry,fill,right,source))} == Bands.Split{Model.content_layout(~Element,model(~Element,geometry,fill,left,source)),_} : Bands.Plan}
      {==}

# Repartition outputs retain the planner's shape and allocation invariants.
# Target Fits is shared with dense conversion; range performs the real copies.
def leaf_ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+source: Model.Content<Element>,+capacity: U32,+division_capacity: U32,
  facts: BandDense.LeafFacts<geometry,first,rows,capacity,division_capacity>) -> Model.ContentReady(~Element,leaf_model(~Element,geometry,fill,first,rows,source)):
  BandDense.LeafFacts{allocation,contents,metadata} = facts
  +count = Bands.local_count(geometry,rows)
  Writes.content_ready(~Element,BandRange.content_writes(~Element,BandRange.Request{geometry,first,rows,0},source),Buffer.tree(~Element,BandCount.model(~Element,count,fill)),U32.to_nat(count),
    Buffer.content_ready(~Element,BandCount.model(~Element,count,fill),BandCount.bounded_model(~Element,count,fill,allocation)))

def modeled_ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Content<Element>,+capacity: U32,+division_capacity: U32,
  facts: BandDense.Fits(geometry,layout,capacity,division_capacity)) -> Model.ContentReady(~Element,model(~Element,geometry,fill,layout,source)):
  match layout:
    case Bands.Rows{first,rows}: leaf_ready(~Element,geometry,fill,first,rows,source,capacity,division_capacity,facts)
    case Bands.Split{+left,+right}:
      Shape.Both{left_facts,right_facts} = facts
      Shape.Both{modeled_ready(~Element,geometry,fill,left,source,capacity,division_capacity,left_facts),modeled_ready(~Element,geometry,fill,right,source,capacity,division_capacity,right_facts)}

def modeled_shape(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Content<Element>,+capacity: U32,+division_capacity: U32,
  facts: BandDense.Fits(geometry,layout,capacity,division_capacity)) -> BandShape.Contents(~Element,geometry,model(~Element,geometry,fill,layout,source)):
  match layout:
    case Bands.Rows{first,rows}:
      BandDense.LeafFacts{allocation,contents,metadata} = facts
      metadata
    case Bands.Split{+left,+right}:
      Shape.Both{left_facts,right_facts} = facts
      Shape.Both{modeled_shape(~Element,geometry,fill,left,source,capacity,division_capacity,left_facts),modeled_shape(~Element,geometry,fill,right,source,capacity,division_capacity,right_facts)}

def modeled_well_formed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Content<Element>,+capacity: U32,+division_capacity: U32,
  +fits: BandDense.Fits(geometry,layout,capacity,division_capacity),coverage: BandPlan.Coverage(layout,0n,Inspection.total_rows(geometry))) -> BandShape.WellFormed<Element,geometry,model(~Element,geometry,fill,layout,source)>:
  BandShape.WellFormed{modeled_shape(~Element,geometry,fill,layout,source,capacity,division_capacity,fits),modeled_ready(~Element,geometry,fill,layout,source,capacity,division_capacity,fits),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,model(~Element,geometry,fill,layout,source)),layout,model_layout(~Element,geometry,fill,layout,source)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry)) coverage}

def repartitioned_well_formed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>,+capacity: U32,+division_capacity: U32,
  fits: BandDense.Fits(geometry,layout,capacity,division_capacity),coverage: BandPlan.Coverage(layout,0n,Inspection.total_rows(geometry))) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))))>:
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),
    model(~Element,geometry,fill,layout,Model.content(~Element,source)),repartitioned(~Element,geometry,fill,layout,source)) : BandShape.WellFormed<Element,geometry,_>
  modeled_well_formed(~Element,geometry,fill,layout,Model.content(~Element,source),capacity,division_capacity,fits,coverage)

def target_bounds(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+output: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<geometry,limits,capacity>,metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,output)),ready: Buffer.Ready<Element,output>) ->
  BandRange.Target(BandRange.Request{geometry,first,rows,0},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  Buffer.Ready{depth_ok,room} = ready
  BandRange.TargetFields{Limits.capacity_fits(limits),Buffer.capacity_bound(~Element,output,depth_ok),channels_positive,width_positive,
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry),U32.to_nat(Machine.capacity(limits)),rows_fit,BandGather.rows_room(geometry,limits,capacity,domain)),
    %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool} room}

# A cell of a requested band is the cell the range copy wrote at its position.
def copied_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,source_values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  target: BandDense.LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool}) ->
  BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,
    Buffer.content_tree(~Element,leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),Buffer.content_length(~Element,leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),channel,row,column):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandDense.LeafFacts{allocation,contents,+metadata} = target
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  +count = Bands.local_count(Bands.Geometry{images,channels,height,width},rows)
  +output = BandCount.model(~Element,count,fill)
  +ready = BandCount.bounded_model(~Element,count,fill,allocation)
  Shape.Both{BandValues.position_within(Bands.Geometry{images,channels,height,width},rows,U32.to_nat(count),channel,row,column,length_exact,channel_inside,row_inside,column_inside),
    BandRange.written_point(~Element,~Logical,~original,logical,BandRange.Request{Bands.Geometry{images,channels,height,width},first,rows,0},source,output,limits,U32.to_nat(count),capacity,domain,
      target_bounds(~Element,Bands.Geometry{images,channels,height,width},first,rows,output,limits,capacity,domain,metadata,ready),source_shape,metadata,ready,source_values,
      channel,row,column,channel_inside,row_inside,column_inside)}

# What one requested band supplies to every cell.
type LeafCells<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  LeafCells{logical: Logical,geometry: Bands.Geometry,fill: Element,first: U32,rows: U32,source: Model.Model<Element>,limits: Machine.Limits,capacity: U32,
    domain: BandGather.Domain<geometry,limits,capacity>,source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,
    source_values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),target: BandDense.LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>}

def cells_geometry(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: LeafCells<Element,Logical,original>) -> Bands.Geometry:
  LeafCells{logical,geometry,fill,first,rows,source,limits,capacity,domain,source_shape,source_values,target} = evidence
  geometry

def cells_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: LeafCells<Element,Logical,original>) -> U32:
  LeafCells{logical,geometry,fill,first,rows,source,limits,capacity,domain,source_shape,source_values,target} = evidence
  rows

def Cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: LeafCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  LeafCells{logical,+geometry,+fill,+first,+rows,+source,limits,capacity,domain,source_shape,source_values,target} = evidence
  BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,
    Buffer.content_tree(~Element,leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),Buffer.content_length(~Element,leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),channel,row,column)

def leaf_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: LeafCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(cells_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(cells_rows(~Element,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(cells_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool}) ->
  Cell(~Element,~Logical,~original,evidence,channel,row,column):
  LeafCells{logical,geometry,fill,first,rows,source,limits,capacity,domain,source_shape,source_values,target} = evidence
  copied_cell(~Element,~Logical,~original,logical,geometry,fill,first,rows,source,limits,capacity,domain,source_shape,source_values,target,channel,row,column,channel_inside,row_inside,column_inside)

def modeled_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  +source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,+source_values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  target: BandDense.Fits(geometry,layout,capacity,Machine.half_word(limits))) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,model(~Element,geometry,fill,layout,Model.content(~Element,source))):
  match layout:
    case Bands.Rows{+first,+rows}:
      BandValues.cells(~LeafCells<Element,Logical,original>,~Cell(~Element,~Logical,~original),~cells_geometry(~Element,~Logical,~original),~cells_rows(~Element,~Logical,~original),~leaf_cell(~Element,~Logical,~original),
        LeafCells{logical,geometry,fill,first,rows,source,limits,capacity,domain,source_shape,source_values,target})
    case Bands.Split{+left,+right}:
      Shape.Both{left_target,right_target} = target
      Shape.Both{modeled_values(~Element,~Logical,~original,logical,geometry,fill,left,source,limits,capacity,domain,source_shape,source_values,left_target),
        modeled_values(~Element,~Logical,~original,logical,geometry,fill,right,source,limits,capacity,domain,source_shape,source_values,right_target)}

def repartitioned_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>,
  source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,source_values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  target: BandDense.Fits(geometry,layout,capacity,Machine.half_word(limits))) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),
    model(~Element,geometry,fill,layout,Model.content(~Element,source)),repartitioned(~Element,geometry,fill,layout,source)) : BandValues.Values(~Element,~Logical,~original,logical,geometry,_)
  modeled_values(~Element,~Logical,~original,logical,geometry,fill,layout,source,limits,capacity,domain,source_shape,source_values,target)
