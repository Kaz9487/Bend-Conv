# Every gathered flat value follows the existing finite band relation. The
# coordinate receipt is produced from the bounded flat position, and each
# sample reuses the official gather refinement rather than an output oracle.
import Base
import ../tensor_band.bend as Bands
import ../tensor.bend as Tensors
import ../tensor_window.bend as Window
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_band_model.bend as Model
import ./tensor_band_values.bend as BandValues
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_inspection.bend as Inspection
import ./tensor_dense_contract.bend as Dense
import ./tensor_normal_densification.bend as Normal
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_dense_observation.bend as DenseObservation
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_shape_count.bend as BandShapeCount
import ./tensor_band_count.bend as BandCount
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_count_refinement.bend as Count
import ./tensor_shape_refinement.bend as ShapeLaws
import ./machine_limit_bounds.bend as Limits
import ./nat_to_u32_bounds.bend as Bounds
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./row_major_coordinates.bend as RowCoordinates
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./write_program.bend as Writes
import ./word_encoding.bend as WordEncoding

type Coordinates<-geometry: Bands.Geometry,-position: Nat> is Data:
  Coordinates{image: Nat,channel: Nat,row: Nat,column: Nat,
    inside: BandGather.Inside(geometry,image,channel,row,column),
    exact: {position == BandGather.point(geometry,image,channel,row,column) : Nat}}

def FlatCoordinates(geometry: Bands.Geometry,-position: Nat) -> Data:
  Bands.Geometry{images,channels,height,width} = geometry
  RowCoordinates.Coordinates<Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height),U32.to_nat(width),position>

def split_coordinates(+geometry: Bands.Geometry,+position: Nat,
  coordinates: FlatCoordinates(geometry,position),
  +channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool}) -> Coordinates<geometry,position>:
  Bands.Geometry{+image_count,+channel_count,+row_count,+column_count} = geometry
  RowCoordinates.Coordinates{+combined,+row,+column,+combined_inside,row_inside,column_inside,+exact} = coordinates
  +image = Nat.div(combined,U32.to_nat(channel_count))
  +channel = Nat.mod(combined,U32.to_nat(channel_count))
  +split = DivisionOrder.quotient_exact(combined,U32.to_nat(channel_count),channels_positive)
  Coordinates{image,channel,row,column,
    BandGather.PointInside{RowMajor.row_inside(image,U32.to_nat(image_count),U32.to_nat(channel_count),channel,
        %split : {Nat.is_lt(_,Nat.mul(U32.to_nat(image_count),U32.to_nat(channel_count))) == True{} : Bool} combined_inside),
      DivisionOrder.remainder_small(combined,U32.to_nat(channel_count),channels_positive),row_inside,column_inside},
    Equal.trans(Nat,position,RowMajor.index(combined,U32.to_nat(row_count),U32.to_nat(column_count),row,column),BandGather.point(geometry,image,channel,row,column),exact,
      Equal.cong(Nat,Nat,value => RowMajor.index(value,U32.to_nat(row_count),U32.to_nat(column_count),row,column),combined,Nat.add(Nat.mul(image,U32.to_nat(channel_count)),channel),split))}

def coordinates(+geometry: Bands.Geometry,+position: Nat,+channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},
  +width_positive: {BandGather.width_positive(geometry) == True{} : Bool},inside: {Nat.is_lt(position,BandGather.elements(geometry)) == True{} : Bool}) -> Coordinates<geometry,position>:
  Bands.Geometry{+image_count,+channel_count,+row_count,+column_count} = geometry
  split_coordinates(geometry,position,
    RowCoordinates.locate(Nat.mul(U32.to_nat(image_count),U32.to_nat(channel_count)),U32.to_nat(row_count),U32.to_nat(column_count),position,width_positive,
      %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(image_count),U32.to_nat(channel_count)),Nat.mul(U32.to_nat(row_count),U32.to_nat(column_count))),BandGather.elements(geometry),
        RowMajor.assoc(U32.to_nat(image_count),U32.to_nat(channel_count),Nat.mul(U32.to_nat(row_count),U32.to_nat(column_count)))) : {Nat.is_lt(position,_) == True{} : Bool} inside),channels_positive)

def gathered_length(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,result: Buffer.Model<Element>) ->
  {Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))) == Buffer.length(~Element,result) : Nat}:
  Buffer.Model{+tree,+length,+certificate} = result
  Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),
    Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,Buffer.Model{tree,length,certificate}))),
    Buffer.Content{Writes.pure(Element,BandGather.refinement_writes(~Element,geometry,bands),tree),length},BandGather.gathered_writes(~Element,geometry,bands,Buffer.Model{tree,length,certificate}))

def gathered_point(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,
  +bands: Model.Model<Element>,+result: Buffer.Model<Element>,+domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>,
  +checked: {BandGather.complete(~Element,geometry,bands) == True{} : Bool},+values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),
  +position: Nat,location: Coordinates<geometry,position>,depth_ok: {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}) ->
  Dense.At(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))),position):
  Coordinates{+image,+channel,+row,+column,inside,exact} = location
  %Equal.sym(Nat,position,BandGather.point(geometry,image,channel,row,column),exact) :
    {Buffer.value(~Element,U32.from_nat(_),Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))) == original(logical,U32.from_nat(_)) : Element}
  BandGather.gathered_value(~Element,~Logical,~original,logical,geometry,limits,bands,result,domain,checked,values,image,channel,row,column,inside,depth_ok)

def domain_channels(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) -> {BandGather.channels_positive(geometry) == True{} : Bool}:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  channels_positive

def domain_width(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) -> {BandGather.width_positive(geometry) == True{} : Bool}:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  width_positive

def gathered_range(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,
  +bands: Model.Model<Element>,+result: Buffer.Model<Element>,+domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>,
  +checked: {BandGather.complete(~Element,geometry,bands) == True{} : Bool},+values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),
  remaining: Nat,+position: Nat,+room: {Nat.is_le(Nat.add(position,remaining),BandGather.elements(geometry)) == True{} : Bool},
  +depth_ok: {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}) ->
  Members.All(Nat,{index => Dense.At(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))),index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{gathered_point(~Element,~Logical,~original,logical,geometry,limits,bands,result,domain,checked,values,position,
          coordinates(geometry,position,domain_channels(geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result)),domain),domain_width(geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result)),domain),Order.room_index(position,rest,BandGather.elements(geometry),room)),depth_ok),
        gathered_range(~Element,~Logical,~original,logical,geometry,limits,bands,result,domain,checked,values,rest,1n+position,Order.advance_room(position,rest,BandGather.elements(geometry),room),depth_ok)}

def gathered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,
  +bands: Model.Model<Element>,+result: Buffer.Model<Element>,+domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>,
  +checked: {BandGather.complete(~Element,geometry,bands) == True{} : Bool},+values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),
  +length_exact: {Buffer.length(~Element,result) == BandGather.elements(geometry) : Nat},+depth_ok: {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}) ->
  Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))):
  %Equal.sym(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))),Buffer.length(~Element,result),gathered_length(~Element,geometry,bands,result)) :
    Members.All(Nat,{index => Dense.At(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))),index) : Nat -> Data},Ranges.range(_,0n))
  %Equal.sym(Nat,Buffer.length(~Element,result),BandGather.elements(geometry),length_exact) :
    Members.All(Nat,{index => Dense.At(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))),index) : Nat -> Data},Ranges.range(_,0n))
  gathered_range(~Element,~Logical,~original,logical,geometry,limits,bands,result,domain,checked,values,BandGather.elements(geometry),0n,Order.reflexive(BandGather.elements(geometry)),depth_ok)

def fresh_model(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,+result: Buffer.Model<Element>,
  allocated: {Buffer.pack(~Element,result) == Tensors.buffer(~Element,fill,Bands.elements(geometry)) : Storage.Buffer<Element>},
  coherent: {Window.band_shape(shape,geometry) == True{} : Bool}) -> Dense.WellFormed<Element,shape,Buffer.content(~Element,result)>:
  %Buffer.packed_contents(~Element,result) : Dense.WellFormed<Element,shape,_>
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,result)),Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry))),
    Equal.cong(Storage.Buffer<Element>,Buffer.Content<Element>,Buffer.contents(~Element),Buffer.pack(~Element,result),Tensors.buffer(~Element,fill,Bands.elements(geometry)),allocated)) : Dense.WellFormed<Element,shape,_>
  %Equal.sym(U32,Bands.elements(geometry),Global.count(shape,counted),BandShapeCount.global_element_count_exact(shape,geometry,counted,coherent)) : Dense.WellFormed<Element,shape,Buffer.contents(~Element,Tensors.buffer(~Element,fill,_))>
  Normal.fresh(~Element,shape,fill,counted)

def geometry_length(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,result: Buffer.Model<Element>,
  well_formed: Dense.WellFormed<Element,shape,Buffer.content(~Element,result)>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool}) ->
  {Buffer.length(~Element,result) == BandGather.elements(geometry) : Nat}:
  Buffer.Model{tree,+length,certificate} = result
  Dense.WellFormed{length_exact,ready} = well_formed
  Equal.trans(Nat,length,ShapeLaws.elements(shape),BandGather.elements(geometry),length_exact,BandCount.shape_coherence_elements(shape,geometry,coherent))

def model_ready(~Element: Data,+shape: List<&2,U32>,+result: Buffer.Model<Element>,well_formed: Dense.WellFormed<Element,shape,Buffer.content(~Element,result)>) -> Buffer.Ready<Element,result>:
  Dense.WellFormed{length_exact,ready} = well_formed
  Buffer.ready_content(~Element,result,ready)

def model_depth(~Element: Data,+result: Buffer.Model<Element>,ready: Buffer.Ready<Element,result>) -> {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}:
  Buffer.Ready{depth_ok,room} = ready
  depth_ok

def output_domain(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+counted: Global.Counted<shape>,+result: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  +capacity_limits: {capacity == Machine.capacity(limits) : U32},+capacity_default: {capacity == Machine.capacity(Default.limits()) : U32},
  evidence: Count.Count<BandGather.elements(geometry),Global.count(shape,counted)>,ready: Buffer.Ready<Element,result>,
  length_exact: {Buffer.length(~Element,result) == BandGather.elements(geometry) : Nat},
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool}) ->
  BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>:
  Count.Count{exact,bounded} = evidence
  Buffer.Ready{+depth_ok,room} = ready
  BandGather.Domain{Buffer.capacity_bound(~Element,result,depth_ok),channels_positive,height_positive,width_positive,
    %capacity_limits : {Nat.is_le(BandGather.elements(geometry),U32.to_nat(_)) == True{} : Bool}
    BandCount.natural_room(BandGather.elements(geometry),Global.count(shape,counted),capacity,exact,
      %Equal.sym(U32,capacity,Machine.capacity(Default.limits()),capacity_default) : {U32.is_le(Global.count(shape,counted),_) == True{} : Bool} bounded),
    %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,result)))) == True{} : Bool} room}

def empty_gathered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+result: Buffer.Model<Element>,
  length_exact: {Buffer.length(~Element,result) == BandGather.elements(geometry) : Nat},natural_zero: {BandGather.elements(geometry) == 0n : Nat}) ->
  Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))):
  %Equal.sym(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))),0n,
    Equal.trans(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))),Buffer.length(~Element,result),0n,
      gathered_length(~Element,geometry,bands,result),Equal.trans(Nat,Buffer.length(~Element,result),BandGather.elements(geometry),0n,length_exact,natural_zero))) :
    Members.All(Nat,{index => Dense.At(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))),index) : Nat -> Data},Ranges.range(_,0n))
  Unit{}

def banded_selected_limits(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,+result: Buffer.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+capacity_limits: {capacity == Machine.capacity(limits) : U32},+capacity_default: {capacity == Machine.capacity(Default.limits()) : U32},
  +allocated: {Buffer.pack(~Element,result) == Tensors.buffer(~Element,fill,Bands.elements(geometry)) : Storage.Buffer<Element>},
  +coherent: {Window.band_shape(shape,geometry) == True{} : Bool},+metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),classification: BandShapeCount.GeometryCase<geometry>) ->
  Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))):
  match classification:
    case BandShapeCount.ZeroGeometry{native_zero,natural_zero}:
      empty_gathered(~Element,~Logical,~original,logical,geometry,bands,result,
        geometry_length(~Element,geometry,shape,result,fresh_model(~Element,geometry,shape,fill,counted,result,allocated,coherent),coherent),natural_zero)
    case BandShapeCount.PositiveGeometry{+channels_positive,+height_positive,+width_positive}:
      +well_formed = fresh_model(~Element,geometry,shape,fill,counted,result,allocated,coherent)
      +ready = model_ready(~Element,shape,result,well_formed)
      +length_exact = geometry_length(~Element,geometry,shape,result,well_formed,coherent)
      +domain = output_domain(~Element,geometry,shape,counted,result,limits,capacity,capacity_limits,capacity_default,Global.geometry_evidence(shape,geometry,counted,coherent),ready,length_exact,channels_positive,height_positive,width_positive)
      gathered(~Element,~Logical,~original,logical,geometry,limits,bands,result,domain,
        BandShape.well_formed_checked(~Element,geometry,bands,capacity,
          %Equal.sym(U32,capacity,Machine.capacity(limits),capacity_limits) : {U32.is_le(_,2147483648) == True{} : Bool} Limits.capacity_fits(limits),
          %Equal.sym(U32,capacity,Machine.capacity(limits),capacity_limits) : {Nat.is_le(Inspection.total_rows(geometry),U32.to_nat(_)) == True{} : Bool}
          BandGather.rows_room(geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result)),domain),metadata),
        values,length_exact,model_depth(~Element,result,ready))

type MachineObservation is Data:
  MachineObservation{limits: Machine.Limits,capacity: U32,
    capacity_limits: {capacity == Machine.capacity(limits) : U32},
    capacity_default: {capacity == Machine.capacity(Default.limits()) : U32}}

def observe_machine() -> MachineObservation:
  MachineObservation{Default.limits(),Machine.capacity(Default.limits()),{==},{==}}

def banded_selected_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,+result: Buffer.Model<Element>,
  +allocated: {Buffer.pack(~Element,result) == Tensors.buffer(~Element,fill,Bands.elements(geometry)) : Storage.Buffer<Element>},
  +coherent: {Window.band_shape(shape,geometry) == True{} : Bool},+metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),+classification: BandShapeCount.GeometryCase<geometry>,observation: MachineObservation) ->
  Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))):
  MachineObservation{+limits,+capacity,+capacity_limits,+capacity_default} = observation
  banded_selected_limits(~Element,~Logical,~original,logical,geometry,bands,shape,fill,counted,result,limits,capacity,capacity_limits,capacity_default,allocated,coherent,metadata,values,classification)

def banded_selected(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,+result: Buffer.Model<Element>,
  +allocated: {Buffer.pack(~Element,result) == Tensors.buffer(~Element,fill,Bands.elements(geometry)) : Storage.Buffer<Element>},
  +coherent: {Window.band_shape(shape,geometry) == True{} : Bool},+metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),classification: BandShapeCount.GeometryCase<geometry>) ->
  Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))):
  banded_selected_observed(~Element,~Logical,~original,logical,geometry,bands,shape,fill,counted,result,allocated,coherent,metadata,values,classification,observe_machine())

def banded_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,
  +coherent: {Window.band_shape(shape,geometry) == True{} : Bool},+metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),allocation: Buffer.Observation<Element,Tensors.buffer(~Element,fill,Bands.elements(geometry))>) ->
  Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))))):
  Buffer.Observation{+result,+allocated} = allocation
  %allocated : Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),_)))
  banded_selected(~Element,~Logical,~original,logical,geometry,bands,shape,fill,counted,result,allocated,coherent,metadata,values,BandShapeCount.classify(geometry))

def banded(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,
  coherent: {Window.band_shape(shape,geometry) == True{} : Bool},metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands))) ->
  Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))))):
  banded_observed(~Element,~Logical,~original,logical,geometry,bands,shape,fill,counted,coherent,metadata,values,Buffer.observe(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry))))

def band_status(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,status: Tensors.Status,+workers: U32,
  normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers}),
  ready: {Dense.status_ready(status) == True{} : Bool},
  values: Dense.TensorValues(~Element,~Logical,~original,logical,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers})) ->
  Dense.TensorValues(~Element,~Logical,~original,logical,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers})):
  match status:
    case Tensors.Ready{}:
      Shape.Both{metadata,Shape.Both{counted,coherent}} = normal
      banded(~Element,~Logical,~original,logical,geometry,bands,shape,fill,counted,coherent,
        %Model.packed_contents(~Element,bands) : BandShape.WellFormed<Element,geometry,_> metadata,
        %Model.packed_contents(~Element,bands) : BandValues.Values(~Element,~Logical,~original,logical,geometry,_) values)
    case Tensors.Failed{problem}:
      Empty.absurd(Dense.TensorValues(~Element,~Logical,~original,logical,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers})),WordEncoding.false_true(ready))

def modeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+owner: BandConcatExecution.TensorOwnerModel<Element>,
  normal: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),
  ready: {Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == True{} : Bool},
  values: Dense.TensorValues(~Element,~Logical,~original,logical,BandConcatExecution.tensor_owner_pack(~Element,owner))) ->
  Dense.TensorValues(~Element,~Logical,~original,logical,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))):
  match owner:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}: values
    case BandConcatExecution.BandedOwner{+bands,+geometry,+shape,+fill,+status,+workers}: band_status(~Element,~Logical,~original,logical,bands,geometry,shape,fill,status,workers,normal,ready,values)

def observation_owner(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+owner: BandConcatExecution.TensorOwnerModel<Element>,+buffer: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  equation: {DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers} : Tensors.Tensor<Element>},
  normal: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),
  ready: {Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == True{} : Bool},
  values: Dense.TensorValues(~Element,~Logical,~original,logical,BandConcatExecution.tensor_owner_pack(~Element,owner))) -> Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,buffer)):
  %Buffer.packed_contents(~Element,buffer) : Dense.Values(~Element,~Logical,~original,logical,_)
  %equation : Dense.TensorValues(~Element,~Logical,~original,logical,_)
  modeled(~Element,~Logical,~original,logical,owner,normal,ready,values)

def observation_modeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+owner: BandConcatExecution.TensorOwnerModel<Element>,
  +observation: DenseObservation.DenseObservation<Element,BandConcatExecution.tensor_owner_pack(~Element,owner)>,
  normal: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),ready: {Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == True{} : Bool},
  values: Dense.TensorValues(~Element,~Logical,~original,logical,BandConcatExecution.tensor_owner_pack(~Element,owner))) ->
  Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,DenseObservation.observation_buffer(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner),observation))):
  DenseObservation.DenseObservation{+buffer,+shape,+fill,+status,+workers,equation,status_equal,shape_equal} = observation
  observation_owner(~Element,~Logical,~original,logical,owner,buffer,shape,fill,status,workers,equation,normal,ready,values)

def observation_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,-input: Tensors.Tensor<Element>,+observation: DenseObservation.DenseObservation<Element,input>,
  source_observation: BandConcatExecution.TensorObservation<Element,input>,+normal: Global.Normal(~Element,input),+ready: {Dense.tensor_ready(~Element,input) == True{} : Bool},
  +values: Dense.TensorValues(~Element,~Logical,~original,logical,input)) ->
  Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,DenseObservation.observation_buffer(~Element,input,observation))):
  DenseObservation.DenseObservation{+buffer,+shape,+fill,+status,+workers,+dense_equation,status_equal,shape_equal} = observation
  BandConcatExecution.TensorObservation{+owner,+equation} = source_observation
  observation_owner(~Element,~Logical,~original,logical,owner,buffer,shape,fill,status,workers,
    Equal.trans(Tensors.Tensor<Element>,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),DenseObservation.dense_tensor(~Element,input),Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers},
      Equal.cong(Tensors.Tensor<Element>,Tensors.Tensor<Element>,DenseObservation.dense_tensor(~Element),BandConcatExecution.tensor_owner_pack(~Element,owner),input,equation),dense_equation),
    %Equal.sym(Tensors.Tensor<Element>,BandConcatExecution.tensor_owner_pack(~Element,owner),input,equation) : Global.Normal(~Element,_) normal,
    %Equal.sym(Tensors.Tensor<Element>,BandConcatExecution.tensor_owner_pack(~Element,owner),input,equation) : {Dense.tensor_ready(~Element,_) == True{} : Bool} ready,
    %Equal.sym(Tensors.Tensor<Element>,BandConcatExecution.tensor_owner_pack(~Element,owner),input,equation) : Dense.TensorValues(~Element,~Logical,~original,logical,_) values)

def observation(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,input: Tensors.Tensor<Element>,+observed: DenseObservation.DenseObservation<Element,input>,
  normal: Global.Normal(~Element,input),ready: {Dense.tensor_ready(~Element,input) == True{} : Bool},values: Dense.TensorValues(~Element,~Logical,~original,logical,input)) ->
  Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,DenseObservation.observation_buffer(~Element,input,observed))):
  observation_observed(~Element,~Logical,~original,logical,input,observed,BandConcatExecution.observe_tensor(~Element,input),normal,ready,values)
