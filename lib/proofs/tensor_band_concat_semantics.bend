# Successful survey prefixes supply the exact native channel cursor and the
# room for each actual part. A remainder is a Nat sum, never a modular word.
import Base
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_band.bend as Bands
import ../tensor_band_concat.bend as BandConcat
import ../tensor_shape.bend as Shapes
import ../tensor_window.bend as Window
import ../machine_limits.bend as Machine
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_concat_ordered_sources.bend as ConcatOrderedSources
import ./tensor_band_concat_caller.bend as BandConcatCaller
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_gather.bend as BandGather
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./tensor_band_range.bend as BandRange
import ./tensor_band_copy.bend as BandCopy
import ./nat_interval.bend as Interval
import ./nat_to_u32_bounds.bend as Bounds
import ./encoded_coordinates.bend as Encoded
import ./tensor_band_values.bend as BandValues
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./strided_range_bounds.bend as Strided
import ./nat_division.bend as Division
import ./tensor_band_coordinates.bend as BandCoordinates
import ./word_encoding.bend as WordEncoding
import ./tensor_band_inspection.bend as Inspection
import ./boolean_logic.bend as Boolean
import ./tensor_shape_refinement.bend as ShapeLaws
import ./u32_full_range.bend as FullRange
import ./row_major_coordinates.bend as RowCoordinates
import ./nat_division_order.bend as DivisionOrder
import ./machine_limit_bounds.bend as Limits
import ./storage_copy_refinement.bend as StorageCopyLaws
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./tensor_band_dense.bend as BandDense
import ./tensor_band_plan.bend as BandPlan
import ./word_multiplication.bend as Multiplication
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_count.bend as BandCount
import ./tensor_count_completion.bend as CountCompletion
import ../default_machine_limits.bend as Default

def geometry_images(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  images

def geometry_channels(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  channels

def geometry_height(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  height

def geometry_width(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  width

def ChannelExtents(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+axis: Nat) -> Data:
  match parts:
    case BandConcatExecution.SingleContent{input}:
      {Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) == BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32}
    case BandConcatExecution.ConsContent{input,rest}:
      Shape.Both<{Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) == BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32},ChannelExtents(~Element,rest,axis)>

def channel_extents(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+summary: Layout.Survey,+limit: U32,
  fields: BandConcatCaller.Prefixes(~Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),summary,limit)) -> ChannelExtents(~Element,parts,U32.to_nat(BandConcatCaller.axis_of(rank))):
  match parts:
    case BandConcatExecution.SingleContent{+input}:
      BandConcatCaller.step_channel_extent(rank,channels,height,width,BandConcatExecution.tensor_shape(~Element,input),BandConcatCaller.survey_total(summary),limit,fields)
    case BandConcatExecution.ConsContent{+input,+rest}:
      Shape.Both{head,tail} = fields
      Shape.Both{BandConcatCaller.step_channel_extent(rank,channels,height,width,BandConcatExecution.tensor_shape(~Element,input),BandConcatCaller.survey_total(summary),limit,head),
        channel_extents(~Element,rest,rank,channels,height,width,BandConcatCaller.step(~Element,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),input,summary),limit,tail)}

type ChannelStep<-previous: U32,-source_channels: U32,-next: U32> is Data:
  ChannelStep{cursor: {next == (previous + source_channels : U32) : U32},
    exact: {U32.to_nat((previous + source_channels : U32)) == Nat.add(U32.to_nat(previous),U32.to_nat(source_channels)) : Nat}}

def channel_step(~Element: Data,+first: List<&2,U32>,+axis: Nat,+input: BandConcatExecution.TensorContent<Element>,+summary: Layout.Survey,+limit: U32,
  fields: BandConcatCaller.StepFields<first,BandConcatExecution.tensor_shape(~Element,input),axis,BandConcatCaller.survey_total(summary),limit>,
  +extent: {Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) == BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32}) ->
  ChannelStep<BandConcatCaller.survey_total(summary),BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),BandConcatCaller.survey_total(BandConcatCaller.step(~Element,first,axis,input,summary))>:
  BandConcatCaller.StepFields{rank,prefix,suffix,room,exact} = fields
  ChannelStep{
    %extent : {BandConcatCaller.survey_total(BandConcatCaller.step(~Element,first,axis,input,summary)) == (BandConcatCaller.survey_total(summary) + _ : U32) : U32}
    BandConcatCaller.step_total(~Element,first,axis,input,summary),
    %extent : {U32.to_nat((BandConcatCaller.survey_total(summary) + _ : U32)) == Nat.add(U32.to_nat(BandConcatCaller.survey_total(summary)),U32.to_nat(_)) : Nat}
    exact}

def ChannelRooms(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) -> Data:
  match parts:
    case BandConcatExecution.SingleContent{input}:
      Shape.Both<ChannelStep<BandConcatCaller.survey_total(summary),BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),BandConcatCaller.survey_total(BandConcatCaller.step(~Element,first,axis,input,summary))>,
        {Nat.is_le(Nat.add(U32.to_nat(BandConcatCaller.survey_total(summary)),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))),U32.to_nat(BandConcatCaller.survey_total(BandConcatCaller.step(~Element,first,axis,input,summary)))) == True{} : Bool}>
    case BandConcatExecution.ConsContent{input,rest}:
      Shape.Both<Shape.Both<ChannelStep<BandConcatCaller.survey_total(summary),BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),BandConcatCaller.survey_total(BandConcatCaller.step(~Element,first,axis,input,summary))>,
        {Nat.is_le(Nat.add(U32.to_nat(BandConcatCaller.survey_total(summary)),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))),U32.to_nat(BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,rest,first,axis,BandConcatCaller.step(~Element,first,axis,input,summary))))) == True{} : Bool}>,
        ChannelRooms(~Element,rest,first,axis,BandConcatCaller.step(~Element,first,axis,input,summary))>

def single_room(+previous: U32,+source_channels: U32,+next: U32,step: ChannelStep<previous,source_channels,next>) ->
  Shape.Both<ChannelStep<previous,source_channels,next>,{Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(source_channels)),U32.to_nat(next)) == True{} : Bool}>:
  ChannelStep{+cursor,+exact} = step
  Shape.Both{ChannelStep{cursor,exact},
    %Equal.sym(U32,next,(previous + source_channels : U32),cursor) : {Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(source_channels)),U32.to_nat(_)) == True{} : Bool}
    %exact : {Nat.is_le(_,U32.to_nat((previous + source_channels : U32))) == True{} : Bool}
    Order.reflexive(U32.to_nat((previous + source_channels : U32)))}

def remainder_room(+previous: U32,+source_channels: U32,+next: U32,+output_channels: U32,+remaining: Nat,+limit: U32,
  step: ChannelStep<previous,source_channels,next>,remainder: BandConcatCaller.TotalFields<output_channels,next,remaining,limit>) ->
  Shape.Both<ChannelStep<previous,source_channels,next>,{Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}>:
  ChannelStep{+cursor,+exact} = step
  Shape.Both{ChannelStep{cursor,exact},BandConcatCaller.remaining_channel_room(previous,source_channels,output_channels,remaining,limit,exact,
    %cursor : BandConcatCaller.TotalFields<output_channels,_,remaining,limit> remainder)}

def channel_rooms(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32,
  fields: BandConcatCaller.Prefixes(~Element,parts,first,axis,summary,limit),extents: ChannelExtents(~Element,parts,axis)) -> ChannelRooms(~Element,parts,first,axis,summary):
  match parts:
    case BandConcatExecution.SingleContent{+input}:
      single_room(BandConcatCaller.survey_total(summary),BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),BandConcatCaller.survey_total(BandConcatCaller.step(~Element,first,axis,input,summary)),channel_step(~Element,first,axis,input,summary,limit,fields,extents))
    case BandConcatExecution.ConsContent{+input,+rest}:
      Shape.Both{+head,+tail} = fields
      Shape.Both{+extent,+remaining_extents} = extents
      Shape.Both{remainder_room(BandConcatCaller.survey_total(summary),BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),BandConcatCaller.survey_total(BandConcatCaller.step(~Element,first,axis,input,summary)),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,rest,first,axis,BandConcatCaller.step(~Element,first,axis,input,summary))),BandConcatCaller.extent_total(~Element,rest,axis),limit,
        channel_step(~Element,first,axis,input,summary,limit,head,extent),BandConcatCaller.total_fields(~Element,rest,first,axis,BandConcatCaller.step(~Element,first,axis,input,summary),limit,tail)),
        channel_rooms(~Element,rest,first,axis,BandConcatCaller.step(~Element,first,axis,input,summary),limit,tail,remaining_extents)}

def surveyed_rooms(~Element: Data,+parts: BandConcatExecution.PartContents<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+summary: Layout.Survey,+limit: U32,+maximum: {limit == 4294967295 : U32},
  +accepted: {BandConcatCaller.survey_valid(BandConcatCaller.survey_model(~Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),summary)) == True{} : Bool}) -> ChannelRooms(~Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),summary):
  +fields = BandConcatCaller.prefixes(~Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),summary,limit,maximum,accepted)
  channel_rooms(~Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),summary,limit,fields,channel_extents(~Element,parts,rank,channels,height,width,summary,limit,fields))

# The global output room implies each source's machine/allocation room.
# Zero-channel parts use BandConcatCaller.append_zero_channels before this active branch.
def source_domain(+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+limits: Machine.Limits,+capacity: U32,
  domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,
  +source_positive: {Nat.is_lt(0n,U32.to_nat(source_channels)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}) ->
  BandGather.Domain<Bands.Geometry{images,source_channels,height,width},limits,capacity>:
  BandGather.Domain{+capacity_ok,output_positive,+height_positive,+width_positive,+machine_room,+allocation_room} = domain
  +channels_room = Order.transitive(U32.to_nat(source_channels),Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels),
    %Order.commutative(U32.to_nat(source_channels),U32.to_nat(channel)) : {Nat.is_le(U32.to_nat(source_channels),_) == True{} : Bool}
    Order.le_add(U32.to_nat(source_channels),U32.to_nat(channel)),room)
  +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +source_frame = Nat.mul(U32.to_nat(source_channels),spatial)
  +output_frame = Nat.mul(U32.to_nat(output_channels),spatial)
  +frame_room = RowMajor.mul_le(U32.to_nat(source_channels),U32.to_nat(output_channels),spatial,channels_room)
  +total_room = {
    %RowMajor.multiply_commutative(source_frame,U32.to_nat(images)) : {Nat.is_le(_,Nat.mul(U32.to_nat(images),output_frame)) == True{} : Bool}
    %RowMajor.multiply_commutative(output_frame,U32.to_nat(images)) : {Nat.is_le(Nat.mul(source_frame,U32.to_nat(images)),_) == True{} : Bool}
    RowMajor.mul_le(source_frame,output_frame,U32.to_nat(images),frame_room)
    : {Nat.is_le(BandGather.elements(Bands.Geometry{images,source_channels,height,width}),BandGather.elements(Bands.Geometry{images,output_channels,height,width})) == True{} : Bool}}
  BandGather.Domain{capacity_ok,source_positive,height_positive,width_positive,
    Order.transitive(BandGather.elements(Bands.Geometry{images,source_channels,height,width}),BandGather.elements(Bands.Geometry{images,output_channels,height,width}),U32.to_nat(Machine.capacity(limits)),total_room,machine_room),
    Order.transitive(BandGather.elements(Bands.Geometry{images,source_channels,height,width}),BandGather.elements(Bands.Geometry{images,output_channels,height,width}),U32.to_nat(capacity),total_room,allocation_room)}

def native_channel(~Element: Data,state: Bands.Bands(~Element) & U32) -> U32:
  (bands,channel) = state
  channel

def part_model_cursor(~Element: Data,geometry: Bands.Geometry,+input: BandConcatExecution.TensorContent<Element>,state: BandConcatExecution.ContentState(~Element)) ->
  {BandConcatExecution.state_channel(~Element,BandConcatExecution.part_model(~Element,geometry,input,state)) == (BandConcatExecution.state_channel(~Element,state) + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32) : U32}:
  Bands.Geometry{images,channels,height,width} = geometry
  (output,channel) = state
  {==}

def appended_cursor(~Element: Data,+geometry: Bands.Geometry,+workers: U32,input: Tensors.Tensor<Element>,state: Bands.Bands(~Element) & U32) ->
  {native_channel(~Element,BandConcat.append_part(~Element,geometry,workers,input,state)) == (native_channel(~Element,state) + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,input))) : U32) : U32}:
  match geometry input:
    case Bands.Geometry{images,channels,height,width} Tensors.Tensor{buffer,shape,fill,status,previous}:
      (output,channel) = state
      {==}
    case Bands.Geometry{images,channels,height,width} Tensors.Banded{bands,source_geometry,shape,fill,status,previous}:
      (output,channel) = state
      {==}

def appended_well_formed_packed(~Element: Data,+geometry: Bands.Geometry,+workers: U32,+input: BandConcatExecution.TensorOwnerModel<Element>,+state: BandConcatExecution.BandStateModel<Element>,+output_geometry: Bands.Geometry,
  evidence: BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.output(~Element,BandConcatExecution.band_state_pack(~Element,state)))>) ->
  BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.output(~Element,BandConcat.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.band_state_pack(~Element,state))))>:
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandConcat.output(~Element,BandConcat.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.band_state_pack(~Element,state)))),BandConcatExecution.state_output(~Element,BandConcatExecution.state_contents(~Element,BandConcat.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.band_state_pack(~Element,state)))),BandConcatExecution.native_output(~Element,BandConcat.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.band_state_pack(~Element,state)))) : BandShape.WellFormed<Element,output_geometry,_>
  %Equal.sym(BandConcatExecution.ContentState(~Element),BandConcatExecution.state_contents(~Element,BandConcat.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.band_state_pack(~Element,state))),BandConcatExecution.part_model(~Element,geometry,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.state_contents(~Element,BandConcatExecution.band_state_pack(~Element,state))),BandConcatExecution.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.band_state_pack(~Element,state))) : BandShape.WellFormed<Element,output_geometry,BandConcatExecution.state_output(~Element,_)>
  BandConcatExecution.part_well_formed(~Element,geometry,output_geometry,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.state_contents(~Element,BandConcatExecution.band_state_pack(~Element,state)),
    %BandConcatExecution.native_output(~Element,BandConcatExecution.band_state_pack(~Element,state)) : BandShape.WellFormed<Element,output_geometry,_> evidence)

def appended_well_formed_observed(~Element: Data,+geometry: Bands.Geometry,+workers: U32,-input: Tensors.Tensor<Element>,-state: Bands.Bands(~Element) & U32,+output_geometry: Bands.Geometry,
  input_observation: BandConcatExecution.TensorObservation<Element,input>,state_observation: BandConcatExecution.BandStateObservation<Element,state>,evidence: BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.output(~Element,state))>) ->
  BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.output(~Element,BandConcat.append_part(~Element,geometry,workers,input,state)))>:
  BandConcatExecution.TensorObservation{+source,source_equal} = input_observation
  BandConcatExecution.BandStateObservation{+target,+target_equal} = state_observation
  %source_equal : BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.output(~Element,BandConcat.append_part(~Element,geometry,workers,_,state)))>
  %target_equal : BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.output(~Element,BandConcat.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,source),_)))>
  appended_well_formed_packed(~Element,geometry,workers,source,target,output_geometry,
    %Equal.sym(Bands.Bands(~Element) & U32,BandConcatExecution.band_state_pack(~Element,target),state,target_equal) : BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.output(~Element,_))> evidence)

def appended_well_formed(~Element: Data,+geometry: Bands.Geometry,+workers: U32,input: Tensors.Tensor<Element>,state: Bands.Bands(~Element) & U32,+output_geometry: Bands.Geometry,
  evidence: BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.output(~Element,state))>) ->
  BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.output(~Element,BandConcat.append_part(~Element,geometry,workers,input,state)))>:
  appended_well_formed_observed(~Element,geometry,workers,input,state,output_geometry,BandConcatExecution.observe_tensor(~Element,input),BandConcatExecution.observe_band_state(~Element,state),evidence)

type TargetChoice<-Element: Data,-images: U32,-source_channels: U32,-output_channels: U32,-height: U32,-width: U32,-channel: U32,-limits: Machine.Limits,-capacity: U32,-output: Bands.Bands(~Element)> is Data:
  EmptyTarget{zero: {U32.to_nat(source_channels) == 0n : Nat}}
  ActiveTarget{source_domain: BandGather.Domain<Bands.Geometry{images,source_channels,height,width},limits,capacity>,model: Model.Model<Element>,
    owner: {Model.pack(~Element,model) == output : Bands.Bands(~Element)},targets: BandConcatExecution.RangeTargets(~Element,Bands.Geometry{images,source_channels,height,width},channel,limits,model)}

def active_model_targets(~Element: Data,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+limits: Machine.Limits,+capacity: U32,-output: Bands.Bands(~Element),+model: Model.Model<Element>,+owner: {Model.pack(~Element,model) == output : Bands.Bands(~Element)},
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,
  evidence: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,model)>,
  +positive: {Nat.is_lt(0n,U32.to_nat(source_channels)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}) ->
  TargetChoice<Element,images,source_channels,output_channels,height,width,channel,limits,capacity,output>:
  BandShape.WellFormed{metadata,ready,coverage} = evidence
  ActiveTarget{source_domain(images,source_channels,output_channels,height,width,channel,limits,capacity,domain,positive,room),model,owner,
    BandConcatCaller.range_targets(~Element,images,source_channels,output_channels,height,width,channel,limits,capacity,model,domain,metadata,Model.ready(~Element,model,ready),positive,room)}

def observed_targets(~Element: Data,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+limits: Machine.Limits,+capacity: U32,-output: Bands.Bands(~Element),
  observation: Model.Observation<Element,output>,
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,
  evidence: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,output)>,
  +positive: {Nat.is_lt(0n,U32.to_nat(source_channels)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}) ->
  TargetChoice<Element,images,source_channels,output_channels,height,width,channel,limits,capacity,output>:
  Model.Observation{+model,+owner} = observation
  +model_evidence = {
    %Model.packed_contents(~Element,model) : BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},_>
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),output,owner) : BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,_)>
    evidence : BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,model)>}
  active_model_targets(~Element,images,source_channels,output_channels,height,width,channel,limits,capacity,output,model,owner,domain,model_evidence,positive,room)

def target_cases_observed(~Element: Data,channels: Nat,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+limits: Machine.Limits,+capacity: U32,-output: Bands.Bands(~Element),
  +count: {U32.to_nat(source_channels) == channels : Nat},observation: Model.Observation<Element,output>,+domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,
  +evidence: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,output)>,
  +room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}) ->
  TargetChoice<Element,images,source_channels,output_channels,height,width,channel,limits,capacity,output>:
  match channels:
    case 0n: EmptyTarget{count}
    case 1n+ +rest:
      observed_targets(~Element,images,source_channels,output_channels,height,width,channel,limits,capacity,output,observation,domain,evidence,
        %Equal.sym(Nat,U32.to_nat(source_channels),1n+rest,count) : {Nat.is_lt(0n,_) == True{} : Bool} Order.lt_succ(0n,rest,Order.zero_le(rest)),room)

def target_cases(~Element: Data,+channels: Nat,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+limits: Machine.Limits,+capacity: U32,output: Bands.Bands(~Element),
  +count: {U32.to_nat(source_channels) == channels : Nat},+domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,
  +evidence: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,output)>,
  +room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}) ->
  TargetChoice<Element,images,source_channels,output_channels,height,width,channel,limits,capacity,output>:
  target_cases_observed(~Element,channels,images,source_channels,output_channels,height,width,channel,limits,capacity,output,count,Model.observe(~Element,output),domain,evidence,room)

def head_targets_packed(~Element: Data,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+previous: U32,+next: U32,+limits: Machine.Limits,+capacity: U32,+state: BandConcatExecution.BandStateModel<Element>,
  fields: Shape.Both<ChannelStep<previous,source_channels,next>,{Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}>,
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,
  +evidence: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,BandConcat.output(~Element,BandConcatExecution.band_state_pack(~Element,state)))>,
  +cursor: {native_channel(~Element,BandConcatExecution.band_state_pack(~Element,state)) == previous : U32}) ->
  TargetChoice<Element,images,source_channels,output_channels,height,width,native_channel(~Element,BandConcatExecution.band_state_pack(~Element,state)),limits,capacity,BandConcat.output(~Element,BandConcatExecution.band_state_pack(~Element,state))>:
  Shape.Both{step,room} = fields
  target_cases(~Element,U32.to_nat(source_channels),images,source_channels,output_channels,height,width,native_channel(~Element,BandConcatExecution.band_state_pack(~Element,state)),limits,capacity,BandConcat.output(~Element,BandConcatExecution.band_state_pack(~Element,state)),{==},domain,evidence,
    %Equal.sym(U32,native_channel(~Element,BandConcatExecution.band_state_pack(~Element,state)),previous,cursor) : {Nat.is_le(Nat.add(U32.to_nat(_),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool} room)

def head_targets_observed(~Element: Data,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+previous: U32,+next: U32,+limits: Machine.Limits,+capacity: U32,-state: Bands.Bands(~Element) & U32,
  observation: BandConcatExecution.BandStateObservation<Element,state>,fields: Shape.Both<ChannelStep<previous,source_channels,next>,{Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}>,
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+evidence: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,BandConcat.output(~Element,state))>,+cursor: {native_channel(~Element,state) == previous : U32}) ->
  TargetChoice<Element,images,source_channels,output_channels,height,width,native_channel(~Element,state),limits,capacity,BandConcat.output(~Element,state)>:
  BandConcatExecution.BandStateObservation{+target,+equation} = observation
  %equation : TargetChoice<Element,images,source_channels,output_channels,height,width,native_channel(~Element,_),limits,capacity,BandConcat.output(~Element,_)>
  head_targets_packed(~Element,images,source_channels,output_channels,height,width,previous,next,limits,capacity,target,fields,domain,
    %Equal.sym(Bands.Bands(~Element) & U32,BandConcatExecution.band_state_pack(~Element,target),state,equation) : BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,BandConcat.output(~Element,_))> evidence,
    %Equal.sym(Bands.Bands(~Element) & U32,BandConcatExecution.band_state_pack(~Element,target),state,equation) : {native_channel(~Element,_) == previous : U32} cursor)

def head_targets(~Element: Data,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+previous: U32,+next: U32,+limits: Machine.Limits,+capacity: U32,state: Bands.Bands(~Element) & U32,
  fields: Shape.Both<ChannelStep<previous,source_channels,next>,{Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}>,
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+evidence: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,BandConcat.output(~Element,state))>,+cursor: {native_channel(~Element,state) == previous : U32}) ->
  TargetChoice<Element,images,source_channels,output_channels,height,width,native_channel(~Element,state),limits,capacity,BandConcat.output(~Element,state)>:
  head_targets_observed(~Element,images,source_channels,output_channels,height,width,previous,next,limits,capacity,state,BandConcatExecution.observe_band_state(~Element,state),fields,domain,evidence,cursor)

def head_cursor_packed(~Element: Data,+geometry: Bands.Geometry,+workers: U32,+input: BandConcatExecution.TensorOwnerModel<Element>,+state: BandConcatExecution.BandStateModel<Element>,+previous: U32,+next: U32,+output_channels: U32,
  fields: Shape.Both<ChannelStep<previous,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),next>,
    {Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))))),U32.to_nat(output_channels)) == True{} : Bool}>,
  +initial: {native_channel(~Element,BandConcatExecution.band_state_pack(~Element,state)) == previous : U32}) -> {native_channel(~Element,BandConcat.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.band_state_pack(~Element,state))) == next : U32}:
  Shape.Both{step,room} = fields
  ChannelStep{+cursor,exact} = step
  Equal.trans(U32,native_channel(~Element,BandConcat.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.band_state_pack(~Element,state))),
    (previous + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))) : U32),next,
    %initial : {native_channel(~Element,BandConcat.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.band_state_pack(~Element,state))) == (_ + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))) : U32) : U32}
    appended_cursor(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.band_state_pack(~Element,state)),
    Equal.sym(U32,next,(previous + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))) : U32),cursor))

def head_cursor_observed(~Element: Data,+geometry: Bands.Geometry,+workers: U32,-input: Tensors.Tensor<Element>,-state: Bands.Bands(~Element) & U32,+previous: U32,+next: U32,+output_channels: U32,
  input_observation: BandConcatExecution.TensorObservation<Element,input>,state_observation: BandConcatExecution.BandStateObservation<Element,state>,
  fields: Shape.Both<ChannelStep<previous,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,input))),next>,{Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,input))))),U32.to_nat(output_channels)) == True{} : Bool}>,+initial: {native_channel(~Element,state) == previous : U32}) ->
  {native_channel(~Element,BandConcat.append_part(~Element,geometry,workers,input,state)) == next : U32}:
  BandConcatExecution.TensorObservation{+source,+source_equal} = input_observation
  BandConcatExecution.BandStateObservation{+target,+target_equal} = state_observation
  %source_equal : {native_channel(~Element,BandConcat.append_part(~Element,geometry,workers,_,state)) == next : U32}
  %target_equal : {native_channel(~Element,BandConcat.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,source),_)) == next : U32}
  head_cursor_packed(~Element,geometry,workers,source,target,previous,next,output_channels,
    %Equal.sym(Tensors.Tensor<Element>,BandConcatExecution.tensor_owner_pack(~Element,source),input,source_equal) : Shape.Both<ChannelStep<previous,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,_))),next>,{Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,_))))),U32.to_nat(output_channels)) == True{} : Bool}> fields,
    %Equal.sym(Bands.Bands(~Element) & U32,BandConcatExecution.band_state_pack(~Element,target),state,target_equal) : {native_channel(~Element,_) == previous : U32} initial)

def head_cursor(~Element: Data,+geometry: Bands.Geometry,+workers: U32,input: Tensors.Tensor<Element>,state: Bands.Bands(~Element) & U32,+previous: U32,+next: U32,+output_channels: U32,
  fields: Shape.Both<ChannelStep<previous,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,input))),next>,{Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,input))))),U32.to_nat(output_channels)) == True{} : Bool}>,+initial: {native_channel(~Element,state) == previous : U32}) ->
  {native_channel(~Element,BandConcat.append_part(~Element,geometry,workers,input,state)) == next : U32}:
  head_cursor_observed(~Element,geometry,workers,input,state,previous,next,output_channels,BandConcatExecution.observe_tensor(~Element,input),BandConcatExecution.observe_band_state(~Element,state),fields,initial)

def TargetsSchedule(~Element: Data,parts: Tensors.Parts<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+workers: U32,+limits: Machine.Limits,+capacity: U32,-state: Bands.Bands(~Element) & U32) -> Data:
  match parts:
    case Tensors.Single{input}:
      TargetChoice<Element,images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,input))),output_channels,height,width,native_channel(~Element,state),limits,capacity,BandConcat.output(~Element,state)>
    case Tensors.Cons{input,rest}:
      Shape.Both<TargetChoice<Element,images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,input))),output_channels,height,width,native_channel(~Element,state),limits,capacity,BandConcat.output(~Element,state)>,
        TargetsSchedule(~Element,rest,images,output_channels,height,width,workers,limits,capacity,BandConcat.append_part(~Element,Bands.Geometry{images,output_channels,height,width},workers,input,state))>

def targets_schedule_continued_observed(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+workers: U32,+limits: Machine.Limits,+capacity: U32,+previous: U32,-state: Bands.Bands(~Element) & U32,
  observation: BandConcatExecution.BandStateObservation<Element,state>,+evidence: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,BandConcat.output(~Element,state))>,+cursor: {native_channel(~Element,state) == previous : U32},
  rule: @+state: BandConcatExecution.BandStateModel<Element> -> @+evidence: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,BandConcat.output(~Element,BandConcatExecution.band_state_pack(~Element,state)))> -> @+cursor: {native_channel(~Element,BandConcatExecution.band_state_pack(~Element,state)) == previous : U32} ->
    TargetsSchedule(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),images,output_channels,height,width,workers,limits,capacity,BandConcatExecution.band_state_pack(~Element,state))) ->
  TargetsSchedule(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),images,output_channels,height,width,workers,limits,capacity,state):
  BandConcatExecution.BandStateObservation{+target,+equation} = observation
  %equation : TargetsSchedule(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),images,output_channels,height,width,workers,limits,capacity,_)
  rule(target,
    %Equal.sym(Bands.Bands(~Element) & U32,BandConcatExecution.band_state_pack(~Element,target),state,equation) : BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,BandConcat.output(~Element,_))> evidence,
    %Equal.sym(Bands.Bands(~Element) & U32,BandConcatExecution.band_state_pack(~Element,target),state,equation) : {native_channel(~Element,_) == previous : U32} cursor)

def targets_schedule_continued(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+workers: U32,+limits: Machine.Limits,+capacity: U32,+previous: U32,state: Bands.Bands(~Element) & U32,
  +evidence: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,BandConcat.output(~Element,state))>,+cursor: {native_channel(~Element,state) == previous : U32},
  rule: @+state: BandConcatExecution.BandStateModel<Element> -> @+evidence: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.contents(~Element,BandConcat.output(~Element,BandConcatExecution.band_state_pack(~Element,state)))> -> @+cursor: {native_channel(~Element,BandConcatExecution.band_state_pack(~Element,state)) == previous : U32} ->
    TargetsSchedule(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),images,output_channels,height,width,workers,limits,capacity,BandConcatExecution.band_state_pack(~Element,state))) ->
  TargetsSchedule(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),images,output_channels,height,width,workers,limits,capacity,state):
  targets_schedule_continued_observed(~Element,parts,images,output_channels,height,width,workers,limits,capacity,previous,state,BandConcatExecution.observe_band_state(~Element,state),evidence,cursor,rule)

def targets_schedule_packed(~Element: Data,parts: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+images: U32,+height: U32,+width: U32,+workers: U32,+limits: Machine.Limits,+capacity: U32,+state: BandConcatExecution.BandStateModel<Element>,
  rooms: ChannelRooms(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary),
  +domain: BandGather.Domain<Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary)),height,width},limits,capacity>,
  +evidence: BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary)),height,width},Model.contents(~Element,BandConcat.output(~Element,BandConcatExecution.band_state_pack(~Element,state)))>,
  +cursor: {native_channel(~Element,BandConcatExecution.band_state_pack(~Element,state)) == BandConcatCaller.survey_total(summary) : U32}) ->
  TargetsSchedule(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary)),height,width,workers,limits,capacity,BandConcatExecution.band_state_pack(~Element,state)):
  match parts:
    case BandConcatExecution.SingleOwner{+input}:
      head_targets_packed(~Element,images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),BandConcatCaller.survey_total(BandConcatCaller.step(~Element,first,axis,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),summary)),height,width,
        BandConcatCaller.survey_total(summary),BandConcatCaller.survey_total(BandConcatCaller.step(~Element,first,axis,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),summary)),limits,capacity,state,rooms,domain,evidence,cursor)
    case BandConcatExecution.ConsOwner{+input,+rest}:
      Shape.Both{+head,+tail} = rooms
      +next_summary = BandConcatCaller.step(~Element,first,axis,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),summary)
      +output_channels = BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,next_summary))
      +geometry = {Bands.Geometry{images,output_channels,height,width} : Bands.Geometry}
      Shape.Both{head_targets_packed(~Element,images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),output_channels,height,width,BandConcatCaller.survey_total(summary),BandConcatCaller.survey_total(next_summary),limits,capacity,state,head,domain,evidence,cursor),
        targets_schedule_continued(~Element,rest,images,output_channels,height,width,workers,limits,capacity,BandConcatCaller.survey_total(next_summary),BandConcat.append_part(~Element,geometry,workers,BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.band_state_pack(~Element,state)),
          appended_well_formed_packed(~Element,geometry,workers,input,state,geometry,evidence),head_cursor_packed(~Element,geometry,workers,input,state,BandConcatCaller.survey_total(summary),BandConcatCaller.survey_total(next_summary),output_channels,head,cursor),
          target => target_evidence => target_cursor => targets_schedule_packed(~Element,rest,first,axis,next_summary,images,height,width,workers,limits,capacity,target,tail,domain,target_evidence,target_cursor))}

def targets_schedule_observed(~Element: Data,-parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+images: U32,+height: U32,+width: U32,+workers: U32,+limits: Machine.Limits,+capacity: U32,-state: Bands.Bands(~Element) & U32,
  parts_observation: BandConcatExecution.PartsObservation<Element,parts>,state_observation: BandConcatExecution.BandStateObservation<Element,state>,rooms: ChannelRooms(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary),
  +domain: BandGather.Domain<Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},limits,capacity>,
  +evidence: BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},Model.contents(~Element,BandConcat.output(~Element,state))>,+cursor: {native_channel(~Element,state) == BandConcatCaller.survey_total(summary) : U32}) ->
  TargetsSchedule(~Element,parts,images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width,workers,limits,capacity,state):
  BandConcatExecution.PartsObservation{+sources,+parts_equal} = parts_observation
  BandConcatExecution.BandStateObservation{+target,+state_equal} = state_observation
  %parts_equal : TargetsSchedule(~Element,_,images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary)),height,width,workers,limits,capacity,state)
  %state_equal : TargetsSchedule(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,axis,summary)),height,width,workers,limits,capacity,_)
  targets_schedule_packed(~Element,sources,first,axis,summary,images,height,width,workers,limits,capacity,target,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,parts_equal) : ChannelRooms(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary) rooms,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,parts_equal) : BandGather.Domain<Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary)),height,width},limits,capacity> domain,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,parts_equal) : BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary)),height,width},Model.contents(~Element,BandConcat.output(~Element,BandConcatExecution.band_state_pack(~Element,target)))>
    %Equal.sym(Bands.Bands(~Element) & U32,BandConcatExecution.band_state_pack(~Element,target),state,state_equal) : BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},Model.contents(~Element,BandConcat.output(~Element,_))> evidence,
    %Equal.sym(Bands.Bands(~Element) & U32,BandConcatExecution.band_state_pack(~Element,target),state,state_equal) : {native_channel(~Element,_) == BandConcatCaller.survey_total(summary) : U32} cursor)

def targets_schedule(~Element: Data,parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+images: U32,+height: U32,+width: U32,+workers: U32,+limits: Machine.Limits,+capacity: U32,state: Bands.Bands(~Element) & U32,
  rooms: ChannelRooms(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary),+domain: BandGather.Domain<Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},limits,capacity>,
  +evidence: BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},Model.contents(~Element,BandConcat.output(~Element,state))>,+cursor: {native_channel(~Element,state) == BandConcatCaller.survey_total(summary) : U32}) ->
  TargetsSchedule(~Element,parts,images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width,workers,limits,capacity,state):
  targets_schedule_observed(~Element,parts,first,axis,summary,images,height,width,workers,limits,capacity,state,BandConcatExecution.observe_parts(~Element,parts),BandConcatExecution.observe_band_state(~Element,state),rooms,domain,evidence,cursor)

def surveyed_targets_observed(~Element: Data,-parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+images: U32,+height: U32,+width: U32,+workers: U32,+limits: Machine.Limits,+capacity: U32,-state: Bands.Bands(~Element) & U32,
  parts_observation: BandConcatExecution.PartsObservation<Element,parts>,state_observation: BandConcatExecution.BandStateObservation<Element,state>,
  +accepted: {BandConcatCaller.survey_valid(BandConcatCaller.observed_summary(~Element,Layout.survey(~Element,parts,first,axis,summary))) == True{} : Bool},
  +domain: BandGather.Domain<Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},limits,capacity>,
  +evidence: BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},Model.contents(~Element,BandConcat.output(~Element,state))>,+cursor: {native_channel(~Element,state) == BandConcatCaller.survey_total(summary) : U32},
  rooms_rule: @+parts: BandConcatExecution.PartContents<Element> -> @+accepted: {BandConcatCaller.survey_valid(BandConcatCaller.survey_model(~Element,parts,first,axis,summary)) == True{} : Bool} -> ChannelRooms(~Element,parts,first,axis,summary)) ->
  TargetsSchedule(~Element,parts,images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width,workers,limits,capacity,state):
  BandConcatExecution.PartsObservation{+sources,+parts_equal} = parts_observation
  BandConcatExecution.BandStateObservation{+target,+state_equal} = state_observation
  +contents = {BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)) : BandConcatExecution.PartContents<Element>}
  +model_accepted = {
    %BandConcatCaller.native_summary_packed(~Element,sources,first,axis,summary) : {BandConcatCaller.survey_valid(_) == True{} : Bool}
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,parts_equal) : {BandConcatCaller.survey_valid(BandConcatCaller.observed_summary(~Element,Layout.survey(~Element,_,first,axis,summary))) == True{} : Bool}
    accepted : {BandConcatCaller.survey_valid(BandConcatCaller.survey_model(~Element,contents,first,axis,summary)) == True{} : Bool}}
  %parts_equal : TargetsSchedule(~Element,_,images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary)),height,width,workers,limits,capacity,state)
  %state_equal : TargetsSchedule(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,contents,first,axis,summary)),height,width,workers,limits,capacity,_)
  targets_schedule_packed(~Element,sources,first,axis,summary,images,height,width,workers,limits,capacity,target,rooms_rule(contents,model_accepted),
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,parts_equal) : BandGather.Domain<Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary)),height,width},limits,capacity> domain,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,parts_equal) : BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary)),height,width},Model.contents(~Element,BandConcat.output(~Element,BandConcatExecution.band_state_pack(~Element,target)))>
    %Equal.sym(Bands.Bands(~Element) & U32,BandConcatExecution.band_state_pack(~Element,target),state,state_equal) : BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},Model.contents(~Element,BandConcat.output(~Element,_))> evidence,
    %Equal.sym(Bands.Bands(~Element) & U32,BandConcatExecution.band_state_pack(~Element,target),state,state_equal) : {native_channel(~Element,_) == BandConcatCaller.survey_total(summary) : U32} cursor)

def surveyed_targets(~Element: Data,parts: Tensors.Parts<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+summary: Layout.Survey,+workers: U32,+limits: Machine.Limits,+capacity: U32,state: Bands.Bands(~Element) & U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  +accepted: {BandConcatCaller.survey_valid(BandConcatCaller.observed_summary(~Element,Layout.survey(~Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),summary))) == True{} : Bool},
  +domain: BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),summary)),height,width},limits,capacity>,
  +evidence: BandShape.WellFormed<Element,Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),summary)),height,width},Model.contents(~Element,BandConcat.output(~Element,state))>,
  +cursor: {native_channel(~Element,state) == BandConcatCaller.survey_total(summary) : U32}) ->
  TargetsSchedule(~Element,parts,BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),summary)),height,width,workers,limits,capacity,state):
  surveyed_targets_observed(~Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),summary,BandConcatCaller.images_of(rank),height,width,workers,limits,capacity,state,BandConcatExecution.observe_parts(~Element,parts),BandConcatExecution.observe_band_state(~Element,state),accepted,domain,evidence,cursor,
    contents => model_accepted => surveyed_rooms(~Element,contents,rank,channels,height,width,summary,limit,maximum,model_accepted))

# Every write of a later channel range starts at or after its allocation base.
# This creates the whole disjointness witness from one prefix inequality.
def start_after_base(region: BandCopy.Region,+channel: Nat) -> {Nat.is_le(BandCopy.base(region),BandCopy.start(region,channel)) == True{} : Bool}:
  BandCopy.Region{+base,+offset,channels,count,+stride,destination,capacity} = region
  Order.transitive(base,Nat.add(base,offset),Nat.add(Nat.add(base,offset),Nat.mul(channel,U32.to_nat(stride))),Order.le_add(base,offset),Order.le_add(Nat.add(base,offset),Nat.mul(channel,U32.to_nat(stride))))

def planes_before(remaining: Nat,+channel: Nat,+region: BandCopy.Region,+query: U32,+before: {Nat.is_lt(U32.to_nat(query),BandCopy.base(region)) == True{} : Bool}) -> BandCopy.Outside(remaining,channel,region,query):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{Interval.Before{Bounds.nat_trans(U32.to_nat(query),BandCopy.base(region),BandCopy.start(region,channel),before,start_after_base(region,channel))},planes_before(rest,1n+channel,region,query,before)}

def range_leaf_selected_before(~Element: Data,+request: BandRange.Request,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+source_capacity: U32,+output_capacity: U32,+query: U32,+previous: Element,
  facts: BandRange.Facts<request,first,rows,source_capacity,output_capacity>,+before: {Nat.is_lt(U32.to_nat(query),U32.to_nat(BandRange.request_offset(request))) == True{} : Bool}) ->
  {Readback.selected(~Element,BandRange.leaf_writes(~Element,request,first,rows,input),query,previous) == previous : Element}:
  BandRange.Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  match facts:
    case BandRange.BoundsEmpty{empty}:
      %Equal.sym(List<&2,Writes.Assignment<Element>>,BandRange.leaf_writes(~Element,request,first,rows,input),[],BandRange.empty_writes(~Element,request,first,rows,input,empty)) : {Readback.selected(~Element,_,query,previous) == previous : Element} {==}
    case BandRange.Copied{source,target,source_span,target_span,positive}:
      BandCopy.Facts{+valid,stride_exact,count_exact,offset_exact} = target
      BandCopy.selected_outside(~Element,U32.to_nat(channels),0n,BandCopy.region(offset,(BandRange.bounds_begin(request,first) - target_first : U32),BandRange.copied_rows(request,first,rows),channels,target_rows,width,output_capacity),
        (rows * width : U32),((BandRange.bounds_begin(request,first) - first) * width : U32),input,query,previous,valid,Order.reflexive(U32.to_nat(channels)),
        planes_before(U32.to_nat(channels),0n,BandCopy.region(offset,(BandRange.bounds_begin(request,first) - target_first : U32),BandRange.copied_rows(request,first,rows),channels,target_rows,width,output_capacity),query,before))

def range_selected_before(~Element: Data,+request: BandRange.Request,source: Model.Model<Element>,+output_capacity: U32,+query: U32,+previous: Element,
  facts: BandRange.BoundsTreeFacts(~Element,request,output_capacity,source),+before: {Nat.is_lt(U32.to_nat(query),U32.to_nat(BandRange.request_offset(request))) == True{} : Bool}) ->
  {Readback.selected(~Element,BandRange.writes(~Element,request,source),query,previous) == previous : Element}:
  match source:
    case Model.Leaf{Buffer.Model{+input,length,certificate},+first,+rows}:
      range_leaf_selected_before(~Element,request,first,rows,input,Tree.capacity(Element,input),output_capacity,query,previous,facts,before)
    case Model.Fork{+left,+right}:
      Shape.Both{+left_facts,+right_facts} = facts
      %Equal.sym(Element,Readback.selected(~Element,List.append(&2,Writes.Assignment<Element>,BandRange.writes(~Element,request,left),BandRange.writes(~Element,request,right)),query,previous),Readback.selected(~Element,BandRange.writes(~Element,request,right),query,Readback.selected(~Element,BandRange.writes(~Element,request,left),query,previous)),Readback.selected_append(~Element,BandRange.writes(~Element,request,left),BandRange.writes(~Element,request,right),query,previous)) : {_ == previous : Element}
      %Equal.sym(Element,Readback.selected(~Element,BandRange.writes(~Element,request,left),query,previous),previous,range_selected_before(~Element,request,left,output_capacity,query,previous,left_facts,before)) : {Readback.selected(~Element,BandRange.writes(~Element,request,right),query,_) == previous : Element}
      range_selected_before(~Element,request,right,output_capacity,query,previous,right_facts,before)

def range_read_before(~Element: Data,+request: BandRange.Request,+source: Model.Model<Element>,+output: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,+query: U32,
  +domain: BandGather.Domain<BandRange.request_geometry(request),limits,capacity>,
  +target: BandRange.Target(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
  source_well_formed: BandShape.WellFormed<Element,BandRange.request_geometry(request),Model.content(~Element,source)>,
  ready: Buffer.Ready<Element,output>,
  +inside: {U32.is_lt(query,Tree.capacity(Element,Buffer.tree(~Element,output))) == True{} : Bool},
  +before: {Nat.is_lt(U32.to_nat(query),U32.to_nat(BandRange.request_offset(request))) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,BandRange.content_writes(~Element,request,Model.content(~Element,source)),Buffer.tree(~Element,output)),query) == Tree.read(Element,Buffer.tree(~Element,output),query) : Element}:
  BandShape.WellFormed{+metadata,+source_ready,coverage} = source_well_formed
  Buffer.Ready{+depth_ok,room} = ready
  +facts = BandRange.tree(~Element,request,source,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output)),target,metadata,Model.ready(~Element,source,source_ready),BandGather.rows_room(BandRange.request_geometry(request),limits,capacity,domain))
  %BandRange.content_program(~Element,request,source) : {Tree.read(Element,Writes.pure(Element,_,Buffer.tree(~Element,output)),query) == Tree.read(Element,Buffer.tree(~Element,output),query) : Element}
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,BandRange.writes(~Element,request,source),Buffer.tree(~Element,output)),query),Readback.selected(~Element,BandRange.writes(~Element,request,source),query,Tree.read(Element,Buffer.tree(~Element,output),query)),Tree.read(Element,Buffer.tree(~Element,output),query),
    Readback.program(~Element,BandRange.writes(~Element,request,source),Buffer.depth(~Element,output),Buffer.tree(~Element,output),query,Buffer.balanced(~Element,output),depth_ok,inside,BandRange.writes_inside(~Element,request,source,Tree.capacity(Element,Buffer.tree(~Element,output)),facts)),
    range_selected_before(~Element,request,source,Tree.capacity(Element,Buffer.tree(~Element,output)),query,Tree.read(Element,Buffer.tree(~Element,output),query),facts,before))

def prefix_offset(~Element: Data,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+first: U32,+rows: U32,+output: Buffer.Model<Element>,
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  metadata: BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.length(~Element,output)),ready: Buffer.Ready<Element,output>,
  +channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}) ->
  {U32.to_nat(BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},channel,rows)) == Nat.mul(U32.to_nat(channel),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat}:
  BandShape.Fields{local,rows_fit,+length_exact} = metadata
  Buffer.Ready{+depth_ok,+length_room} = ready
  +capacity = Tree.capacity(Element,Buffer.tree(~Element,output))
  +plane = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +channels_room = Order.transitive(U32.to_nat(channel),Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels),Order.le_add(U32.to_nat(channel),U32.to_nat(source_channels)),channel_room)
  +offset_room = Order.transitive(Nat.mul(U32.to_nat(channel),plane),Nat.mul(U32.to_nat(output_channels),plane),U32.to_nat(capacity),RowMajor.mul_le(U32.to_nat(channel),U32.to_nat(output_channels),plane,channels_room),
    %length_exact : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} length_room)
  Equal.trans(Nat,U32.to_nat(BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},channel,rows)),Nat.mul(Nat.mul(U32.to_nat(channel),U32.to_nat(rows)),U32.to_nat(width)),Nat.mul(U32.to_nat(channel),plane),
    Encoded.multiply_three(channel,rows,width,capacity,Buffer.capacity_bound(~Element,output,depth_ok),width_positive,
      %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(channel),U32.to_nat(rows)),U32.to_nat(width)),Nat.mul(U32.to_nat(channel),plane),RowMajor.assoc(U32.to_nat(channel),U32.to_nat(rows),U32.to_nat(width))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} offset_room),
    RowMajor.assoc(U32.to_nat(channel),U32.to_nat(rows),U32.to_nat(width)))

def target_offset_room(request: BandRange.Request,+row_capacity: U32,+output_capacity: U32,target: BandRange.Target(request,row_capacity,output_capacity)) ->
  {Nat.is_le(U32.to_nat(BandRange.request_offset(request)),U32.to_nat(output_capacity)) == True{} : Bool}:
  BandRange.Request{Bands.Geometry{images,+channels,height,+width},first,+rows,+offset} = request
  BandRange.TargetFields{row_capacity_ok,output_capacity_ok,channels_positive,width_positive,rows_room,storage_room} = target
  Order.transitive(U32.to_nat(offset),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))),U32.to_nat(output_capacity),Order.le_add(U32.to_nat(offset),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))),storage_room)

# A cell of a channel below the cursor lies before the channel offset of the
# next append.
def prefix_cell_before(~Element: Data,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+prefix_channel: U32,+first: U32,+rows: U32,+output: Buffer.Model<Element>,+channel: Nat,+row: Nat,+column: Nat,
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},metadata: BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.length(~Element,output)),ready: Buffer.Ready<Element,output>,+channel_room: {Nat.is_le(Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool},
  channel_inside: {Nat.is_lt(channel,U32.to_nat(prefix_channel)) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  {Nat.is_lt(BandValues.position(Bands.Geometry{images,output_channels,height,width},rows,channel,row,column),U32.to_nat(BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)),Nat.mul(U32.to_nat(prefix_channel),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),
    prefix_offset(~Element,images,source_channels,output_channels,height,width,prefix_channel,first,rows,output,width_positive,metadata,ready,channel_room)) :
    {Nat.is_lt(BandValues.position(Bands.Geometry{images,output_channels,height,width},rows,channel,row,column),_) == True{} : Bool}
  RowMajor.in_bounds(channel,U32.to_nat(prefix_channel),U32.to_nat(rows),U32.to_nat(width),row,column,channel_inside,row_inside,column_inside)

# A range append leaves every cell before its channel offset unchanged.
def range_prefix_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+output_geometry: Bands.Geometry,+request: BandRange.Request,+source: Model.Model<Element>,+output: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,+channel: Nat,+row: Nat,+column: Nat,
  domain: BandGather.Domain<BandRange.request_geometry(request),limits,capacity>,+target: BandRange.Target(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
  source_well_formed: BandShape.WellFormed<Element,BandRange.request_geometry(request),Model.content(~Element,source)>,ready: Buffer.Ready<Element,output>,
  +before: {Nat.is_lt(BandValues.position(output_geometry,BandRange.request_rows(request),channel,row,column),U32.to_nat(BandRange.request_offset(request))) == True{} : Bool},
  cell: BandValues.At(~Element,~Logical,~original,logical,output_geometry,BandRange.request_first(request),BandRange.request_rows(request),Buffer.tree(~Element,output),Buffer.length(~Element,output),channel,row,column)) ->
  BandValues.At(~Element,~Logical,~original,logical,output_geometry,BandRange.request_first(request),BandRange.request_rows(request),
    Writes.pure(Element,BandRange.content_writes(~Element,request,Model.content(~Element,source)),Buffer.tree(~Element,output)),Buffer.length(~Element,output),channel,row,column):
  Buffer.Ready{+depth_ok,length_room} = ready
  Shape.Both{inside,equal} = cell
  +position = BandValues.position(output_geometry,BandRange.request_rows(request),channel,row,column)
  +pixel = U32.from_nat(BandValues.pixel(output_geometry,channel,Nat.add(U32.to_nat(BandRange.request_first(request)),row),column))
  +buffer_capacity = Tree.capacity(Element,Buffer.tree(~Element,output))
  +position_room = Bounds.nat_trans(position,U32.to_nat(BandRange.request_offset(request)),U32.to_nat(buffer_capacity),before,target_offset_room(request,Machine.capacity(limits),buffer_capacity,target))
  +position_exact = Bounds.roundtrip(position,buffer_capacity,Buffer.capacity_bound(~Element,output,depth_ok),Bounds.strict_le(position,U32.to_nat(buffer_capacity),position_room))
  Shape.Both{inside,
    Equal.trans(Element,Tree.read(Element,Writes.pure(Element,BandRange.content_writes(~Element,request,Model.content(~Element,source)),Buffer.tree(~Element,output)),U32.from_nat(position)),Tree.read(Element,Buffer.tree(~Element,output),U32.from_nat(position)),original(logical,pixel),
      range_read_before(~Element,request,source,output,limits,capacity,U32.from_nat(position),domain,target,source_well_formed,ready,Bounds.native_bound(position,buffer_capacity,position_exact,position_room),
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,position_exact) : {Nat.is_lt(_,U32.to_nat(BandRange.request_offset(request))) == True{} : Bool} before),
      equal)}

# The channels filled so far, per leaf of the output.
def PrefixValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+channel: U32,output: Model.Content<Element>) -> Data:
  match output:
    case Model.RowBand{+content,+first,+rows}:
      BandValues.Channels(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.content_tree(~Element,content),Buffer.content_length(~Element,content),U32.to_nat(channel))
    case Model.Joined{left,right}: Shape.Both<PrefixValues(~Element,~Logical,~original,logical,geometry,channel,left),PrefixValues(~Element,~Logical,~original,logical,geometry,channel,right)>

# What a range append supplies to every cell below the cursor.
type RangePrefix<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  RangePrefix{logical: Logical,images: U32,source_channels: U32,output_channels: U32,height: U32,width: U32,prefix_channel: U32,first: U32,rows: U32,source: Model.Model<Element>,output: Buffer.Model<Element>,limits: Machine.Limits,capacity: U32,
    domain: BandGather.Domain<Bands.Geometry{images,source_channels,height,width},limits,capacity>,width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
    source_well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,source_channels,height,width},Model.content(~Element,source)>,metadata: BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.length(~Element,output)),ready: Buffer.Ready<Element,output>,
    target: BandRange.Target(BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
    values: BandValues.Channels(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.tree(~Element,output),Buffer.length(~Element,output),U32.to_nat(prefix_channel)),channel_room: {Nat.is_le(Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}}

def range_prefix_geometry(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: RangePrefix<Element,Logical,original>) -> Bands.Geometry:
  RangePrefix{logical,images,source_channels,output_channels,height,width,prefix_channel,first,rows,source,output,limits,capacity,domain,width_positive,source_well_formed,metadata,ready,target,values,channel_room} = evidence
  Bands.Geometry{images,prefix_channel,height,width}

def range_prefix_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: RangePrefix<Element,Logical,original>) -> U32:
  RangePrefix{logical,images,source_channels,output_channels,height,width,prefix_channel,first,rows,source,output,limits,capacity,domain,width_positive,source_well_formed,metadata,ready,target,values,channel_room} = evidence
  rows

def RangePrefixCell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: RangePrefix<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  RangePrefix{+logical,+images,+source_channels,+output_channels,+height,+width,+prefix_channel,+first,+rows,+source,+output,+limits,+capacity,+domain,+width_positive,+source_well_formed,+metadata,+ready,+target,+values,+channel_room} = evidence
  BandValues.At(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,
    Writes.pure(Element,BandRange.content_writes(~Element,BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)},Model.content(~Element,source)),Buffer.tree(~Element,output)),
    Buffer.length(~Element,output),channel,row,column)

def range_prefix_leaf_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: RangePrefix<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(range_prefix_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(range_prefix_rows(~Element,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(range_prefix_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool}) ->
  RangePrefixCell(~Element,~Logical,~original,evidence,channel,row,column):
  RangePrefix{+logical,+images,+source_channels,+output_channels,+height,+width,+prefix_channel,+first,+rows,+source,+output,+limits,+capacity,+domain,+width_positive,+source_well_formed,+metadata,+ready,+target,+values,+channel_room} = evidence
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  range_prefix_cell(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)},source,output,limits,capacity,channel,row,column,domain,target,source_well_formed,ready,
    prefix_cell_before(~Element,images,source_channels,output_channels,height,width,prefix_channel,first,rows,output,channel,row,column,width_positive,metadata,ready,channel_room,channel_inside,row_inside,column_inside),
    BandValues.channels_cell(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.tree(~Element,output),Buffer.length(~Element,output),U32.to_nat(prefix_channel),channel,row,column,values,channel_inside,row_inside,column_inside))

def tree_prefix(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+source: Model.Model<Element>,output: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<Bands.Geometry{images,source_channels,height,width},limits,capacity>,+width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  +source_well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,source_channels,height,width},Model.content(~Element,source)>,
  metadata: BandShape.Contents(~Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,output)),+ready: Model.Ready(~Element,output),
  values: PrefixValues(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},channel,Model.content(~Element,output)),targets: BandConcatExecution.RangeTargets(~Element,Bands.Geometry{images,source_channels,height,width},channel,limits,output),
  +channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}) ->
  PrefixValues(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},channel,
    BandConcatExecution.tree_model(~Element,Bands.Geometry{images,source_channels,height,width},channel,BandConcatExecution.BandedContent{Model.content(~Element,source),Bands.Geometry{images,source_channels,height,width},shape,fill,status,workers},Model.content(~Element,output))):
  match output:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      Shape.Both{target,target_metadata} = targets
      +buffer = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
      BandValues.cells(~RangePrefix<Element,Logical,original>,~RangePrefixCell(~Element,~Logical,~original),~range_prefix_geometry(~Element,~Logical,~original),~range_prefix_rows(~Element,~Logical,~original),~range_prefix_leaf_cell(~Element,~Logical,~original),
        RangePrefix{logical,images,source_channels,output_channels,height,width,channel,first,rows,source,buffer,limits,capacity,domain,width_positive,source_well_formed,metadata,ready,target,values,channel_room})
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_values,right_values} = values
      Shape.Both{left_targets,right_targets} = targets
      Shape.Both{tree_prefix(~Element,~Logical,~original,logical,images,source_channels,output_channels,height,width,channel,source,left,shape,fill,status,workers,limits,capacity,domain,width_positive,source_well_formed,left_metadata,left_ready,left_values,left_targets,channel_room),
        tree_prefix(~Element,~Logical,~original,logical,images,source_channels,output_channels,height,width,channel,source,right,shape,fill,status,workers,limits,capacity,domain,width_positive,source_well_formed,right_metadata,right_ready,right_values,right_targets,channel_room)}

def dense_region(request: BandRange.Request,+capacity: U32) -> BandCopy.Region:
  BandRange.Request{+geometry,first,+rows,+offset} = request
  +plane = (rows * geometry_width(geometry) : U32)
  BandCopy.Region{U32.to_nat(offset),0n,U32.to_nat(geometry_channels(geometry)),plane,plane,offset,capacity}

def dense_source_start(request: BandRange.Request) -> U32:
  BandRange.Request{Bands.Geometry{images,+channels,+height,+width},+first,rows,offset} = request
  ((first / height * channels * height + first % height) * width : U32)

def dense_source_stride(request: BandRange.Request) -> U32:
  BandRange.Request{Bands.Geometry{images,channels,+height,+width},first,rows,offset} = request
  (height * width : U32)

def dense_requested_writes(~Element: Data,request: BandRange.Request,+input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  BandRange.Request{Bands.Geometry{images,+channels,+height,+width},+first,+rows,+offset} = request
  BandCopy.writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),((first / height * channels * height + first % height) * width : U32),offset,input)

def dense_schedule(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,+channel: U32,+input: Tree.Tree<Element>) ->
  {BandConcatExecution.dense_writes(~Element,geometry,first,rows,channel,input) == dense_requested_writes(~Element,BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,channel,rows)},input) : List<&2,Writes.Assignment<Element>>}:
  Bands.Geometry{images,channels,height,width} = geometry
  {==}

def dense_plane_exact(request: BandRange.Request,+row_capacity: U32,+output_capacity: U32,target: BandRange.Target(request,row_capacity,output_capacity)) ->
  {U32.to_nat(BandCopy.count(dense_region(request,output_capacity))) == Nat.mul(U32.to_nat(BandRange.request_rows(request)),U32.to_nat(BandRange.row_width(request))) : Nat}:
  BandRange.Request{Bands.Geometry{images,+channels,height,+width},first,+rows,+offset} = request
  BandRange.TargetFields{row_capacity_ok,+capacity_ok,+channels_positive,width_positive,rows_room,+storage_room} = target
  +plane = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +plane_room = Order.transitive(plane,Nat.mul(U32.to_nat(channels),plane),U32.to_nat(output_capacity),
    %RowMajor.multiply_commutative(plane,U32.to_nat(channels)) : {Nat.is_le(plane,_) == True{} : Bool} RowMajor.product_covers(plane,U32.to_nat(channels),channels_positive),
    Strided.count_in_stride(U32.to_nat(offset),Nat.mul(U32.to_nat(channels),plane),U32.to_nat(output_capacity),storage_room))
  Arithmetic.multiply(rows,width,output_capacity,capacity_ok,plane_room)

def dense_target_valid(request: BandRange.Request,+row_capacity: U32,+output_capacity: U32,
  target: BandRange.Target(request,row_capacity,output_capacity),+index: Nat,+index_inside: {Nat.is_lt(index,U32.to_nat(BandCopy.count(dense_region(request,output_capacity)))) == True{} : Bool}) -> BandCopy.Valid<dense_region(request,output_capacity)>:
  BandRange.Request{Bands.Geometry{images,+channels,height,+width},first,+rows,+offset} = request
  BandRange.TargetFields{row_capacity_ok,+capacity_ok,channels_positive,width_positive,rows_room,+storage_room} = target
  +plane = (rows * width : U32)
  +plane_exact = dense_plane_exact(request,row_capacity,output_capacity,target)
  +plane_positive = Order.le_lt_trans(0n,index,U32.to_nat(plane),Order.zero_le(index),index_inside)
  +frame_room = {
    %Equal.sym(Nat,U32.to_nat(plane),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),plane_exact) : {Nat.is_le(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(channels),_)),U32.to_nat(output_capacity)) == True{} : Bool}
    storage_room : {Nat.is_le(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(channels),U32.to_nat(plane))),U32.to_nat(output_capacity)) == True{} : Bool}}
  BandCopy.Valid{capacity_ok,Order.transitive(U32.to_nat(channels),Nat.mul(U32.to_nat(channels),U32.to_nat(plane)),U32.to_nat(output_capacity),RowMajor.product_covers(U32.to_nat(channels),U32.to_nat(plane),plane_positive),Strided.count_in_stride(U32.to_nat(offset),Nat.mul(U32.to_nat(channels),U32.to_nat(plane)),U32.to_nat(output_capacity),frame_room)),
    Equal.sym(Nat,Nat.add(U32.to_nat(offset),0n),U32.to_nat(offset),Addition.zero_right(U32.to_nat(offset))),Order.reflexive(U32.to_nat(plane)),frame_room}

# Dense rows use the shared copy schedule, including its real image start.
# No source samples or final concat equality are supplied by the caller.
def dense_at_offset(~Element: Data,+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+offset: U32,+input: Tree.Tree<Element>,+output: Buffer.Model<Element>,+row_capacity: U32,
  +target: BandRange.Target(BandRange.Request{Bands.Geometry{images,channels,height,width},first,rows,offset},row_capacity,Tree.capacity(Element,Buffer.tree(~Element,output))),ready: Buffer.Ready<Element,output>,
  +channel: Nat,+index: Nat,+channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+index_inside: {Nat.is_lt(index,U32.to_nat((rows * width : U32))) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,BandCopy.writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),((first / height * channels * height + first % height) * width : U32),offset,input),Buffer.tree(~Element,output)),BandCopy.readback_query(dense_region(BandRange.Request{Bands.Geometry{images,channels,height,width},first,rows,offset},Tree.capacity(Element,Buffer.tree(~Element,output))),channel,index)) == BandCopy.sample(~Element,input,(height * width : U32),((first / height * channels * height + first % height) * width : U32),channel,index) : Element}:
  Buffer.Ready{+depth_ok,room} = ready
  +request = {BandRange.Request{Bands.Geometry{images,channels,height,width},first,rows,offset} : BandRange.Request}
  +region = dense_region(request,Tree.capacity(Element,Buffer.tree(~Element,output)))
  BandCopy.read_inside(~Element,U32.to_nat(channels),0n,channel,index,region,(height * width : U32),dense_source_start(request),input,Buffer.tree(~Element,output),Buffer.depth(~Element,output),Buffer.balanced(~Element,output),depth_ok,{==},dense_target_valid(request,row_capacity,Tree.capacity(Element,Buffer.tree(~Element,output)),target,index,index_inside),Order.reflexive(U32.to_nat(channels)),channel_inside,index_inside)

def dense_channel_point(~Element: Data,request: BandRange.Request,+input: Tree.Tree<Element>,+output: Buffer.Model<Element>,+row_capacity: U32,
  +target: BandRange.Target(request,row_capacity,Tree.capacity(Element,Buffer.tree(~Element,output))),ready: Buffer.Ready<Element,output>,
  +channel: Nat,+index: Nat,+channel_inside: {Nat.is_lt(channel,U32.to_nat(BandRange.channel_count(request))) == True{} : Bool},
  +index_inside: {Nat.is_lt(index,U32.to_nat(BandCopy.count(dense_region(request,Tree.capacity(Element,Buffer.tree(~Element,output)))))) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,dense_requested_writes(~Element,request,input),Buffer.tree(~Element,output)),BandCopy.readback_query(dense_region(request,Tree.capacity(Element,Buffer.tree(~Element,output))),channel,index)) == BandCopy.sample(~Element,input,dense_source_stride(request),dense_source_start(request),channel,index) : Element}:
  BandRange.Request{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,+offset} = request
  dense_at_offset(~Element,images,channels,height,width,first,rows,offset,input,output,row_capacity,target,ready,channel,index,channel_inside,index_inside)

def dense_native_channel_point(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+prefix_channel: U32,+input: Tree.Tree<Element>,+output: Buffer.Model<Element>,+row_capacity: U32,
  +target: BandRange.Target(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)},row_capacity,Tree.capacity(Element,Buffer.tree(~Element,output))),+ready: Buffer.Ready<Element,output>,
  +channel: Nat,+index: Nat,+channel_inside: {Nat.is_lt(channel,U32.to_nat(BandRange.channel_count(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)}))) == True{} : Bool},
  +index_inside: {Nat.is_lt(index,U32.to_nat(BandCopy.count(dense_region(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)},Tree.capacity(Element,Buffer.tree(~Element,output)))))) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,BandConcatExecution.dense_writes(~Element,geometry,first,rows,prefix_channel,input),Buffer.tree(~Element,output)),BandCopy.readback_query(dense_region(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)},Tree.capacity(Element,Buffer.tree(~Element,output))),channel,index)) == BandCopy.sample(~Element,input,dense_source_stride(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)}),dense_source_start(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)}),channel,index) : Element}:
  %Equal.sym(List<&2,Writes.Assignment<Element>>,BandConcatExecution.dense_writes(~Element,geometry,first,rows,prefix_channel,input),dense_requested_writes(~Element,BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)},input),dense_schedule(~Element,geometry,first,rows,prefix_channel,input)) : {Tree.read(Element,Writes.pure(Element,_,Buffer.tree(~Element,output)),BandCopy.readback_query(dense_region(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)},Tree.capacity(Element,Buffer.tree(~Element,output))),channel,index)) == BandCopy.sample(~Element,input,dense_source_stride(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)}),dense_source_start(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)}),channel,index) : Element}
  dense_channel_point(~Element,BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)},input,output,row_capacity,target,ready,channel,index,channel_inside,index_inside)

def target_slice_room(request: BandRange.Request,+row_capacity: U32,+output_capacity: U32,target: BandRange.Target(request,row_capacity,output_capacity)) ->
  {Nat.is_le(BandConcatExecution.channel_length(request),U32.to_nat(output_capacity)) == True{} : Bool}:
  BandRange.Request{Bands.Geometry{images,+channels,height,+width},first,+rows,+offset} = request
  BandRange.TargetFields{row_capacity_ok,output_capacity_ok,channels_positive,width_positive,rows_room,storage_room} = target
  Strided.count_in_stride(U32.to_nat(offset),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(output_capacity),storage_room)

def dense_slice_facts(~Element: Data,+request: BandRange.Request,+output: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<BandRange.request_geometry(request),limits,capacity>,+target: BandRange.Target(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
  +metadata: BandShape.Leaf(BandRange.request_geometry(request),BandRange.request_first(request),BandRange.request_rows(request),BandConcatExecution.channel_length(request)),
  +index: Nat,+index_inside: {Nat.is_lt(index,U32.to_nat(BandCopy.count(dense_region(request,Tree.capacity(Element,Buffer.tree(~Element,output)))))) == True{} : Bool}) ->
  BandGather.Facts<BandGather.Call{BandRange.request_geometry(request),BandRange.request_first(request),BandRange.request_rows(request),BandConcatExecution.channel_length(request),capacity,Machine.half_word(limits)}>:
  +plane_positive = {
    %dense_plane_exact(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output)),target) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,index,U32.to_nat(BandCopy.count(dense_region(request,Tree.capacity(Element,Buffer.tree(~Element,output))))),Order.zero_le(index),index_inside)
    : {Nat.is_lt(0n,Nat.mul(U32.to_nat(BandRange.request_rows(request)),U32.to_nat(BandRange.row_width(request)))) == True{} : Bool}}
  BandGather.from_slice(~Element,BandRange.request_geometry(request),limits,BandRange.request_first(request),BandRange.request_rows(request),BandConcatExecution.channel_length(request),output,capacity,domain,metadata,
    RowMajor.left_factor_positive(U32.to_nat(BandRange.request_rows(request)),U32.to_nat(BandRange.row_width(request)),plane_positive),
    target_slice_room(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output)),target))

# The logical concat oracle chooses an original part before encoding its NCHW
# address. Its channel total is the unbounded ordered sum from the Survey.
def source_pixel(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,+input: BandConcatExecution.TensorContent<Element>,+height: U32,+width: U32,image: Nat,channel: Nat,row: Nat,column: Nat) -> Element:
  original(input,U32.from_nat(BandGather.point(Bands.Geometry{0,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),height,width},image,channel,row,column)))

def channel_extent(~Element: Data,context: Unit,input: BandConcatExecution.TensorContent<Element>) -> Nat:
  U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))

def channel_sample(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,context: Sigma<&2,&2,U32,_ => U32>,input: BandConcatExecution.TensorContent<Element>,image: Nat,channel: Nat,row: Nat,column: Nat) -> Element:
  (+height,+width) = context
  +input = input
  +image = image
  +channel = channel
  +row = row
  +column = column
  source_pixel(~Element,~original,input,height,width,image,channel,row,column)

def ordered_sample(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,parts: BandConcatExecution.PartContents<Element>,+height: U32,+width: U32,+image: Nat,+channel: Nat,+row: Nat,+column: Nat) -> Element:
  ConcatOrderedSources.ordered_value(~Element,~Unit,~channel_extent(~Element),~Sigma<&2,&2,U32,_ => U32>,~channel_sample(~Element,~original),Unit{},(height,width),parts,image,channel,row,column)

type LogicalConcat<-Element: Data> is Data:
  LogicalConcat{parts: BandConcatExecution.PartContents<Element>,height: U32,width: U32}

def global_concat_value(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,context: LogicalConcat<Element>,index: U32) -> Element:
  LogicalConcat{+parts,+height,+width} = context
  +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +channels = BandConcatExecution.total_channels(~Element,parts)
  +position = U32.to_nat(index)
  ordered_sample(~Element,~original,parts,height,width,Nat.div(position,Nat.mul(channels,spatial)),Nat.mod(Nat.div(position,spatial),channels),Nat.div(Nat.mod(position,spatial),U32.to_nat(width)),Nat.mod(position,U32.to_nat(width)))

def image_frame(+image: Nat,+channels: Nat,+channel: Nat,+height: Nat,+width: Nat,+row: Nat,+column: Nat) ->
  {RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,row,column) == Nat.add(Nat.mul(image,Nat.mul(channels,Nat.mul(height,width))),RowMajor.index(channel,height,width,row,column)) : Nat}:
  +spatial = Nat.mul(height,width)
  +pixel = Nat.add(Nat.mul(row,width),column)
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,channels),channel),spatial),Nat.add(Nat.mul(Nat.mul(image,channels),spatial),Nat.mul(channel,spatial)),RowMajor.distrib(Nat.mul(image,channels),channel,spatial)) :
    {Nat.add(_,pixel) == Nat.add(Nat.mul(image,Nat.mul(channels,spatial)),RowMajor.index(channel,height,width,row,column)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(image,channels),spatial),Nat.mul(image,Nat.mul(channels,spatial)),RowMajor.assoc(image,channels,spatial)) :
    {Nat.add(Nat.add(_,Nat.mul(channel,spatial)),pixel) == Nat.add(Nat.mul(image,Nat.mul(channels,spatial)),RowMajor.index(channel,height,width,row,column)) : Nat}
  Addition.associative(Nat.mul(image,Nat.mul(channels,spatial)),Nat.mul(channel,spatial),pixel)

def global_concat_pixel(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,+parts: BandConcatExecution.PartContents<Element>,+height: U32,+width: U32,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,+index: U32,
  +encoded: {U32.to_nat(index) == RowMajor.index(Nat.add(Nat.mul(image,BandConcatExecution.total_channels(~Element,parts)),channel),U32.to_nat(height),U32.to_nat(width),row,column) : Nat},
  +channel_inside: {Nat.is_lt(channel,BandConcatExecution.total_channels(~Element,parts)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(height)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  {global_concat_value(~Element,~original,LogicalConcat{parts,height,width},index) == ordered_sample(~Element,~original,parts,height,width,image,channel,row,column) : Element}:
  +channels = BandConcatExecution.total_channels(~Element,parts)
  +height_nat = U32.to_nat(height)
  +width_nat = U32.to_nat(width)
  +spatial = Nat.mul(height_nat,width_nat)
  +frame = Nat.mul(channels,spatial)
  +pixel = Nat.add(Nat.mul(row,width_nat),column)
  +batch_channel = Nat.add(Nat.mul(image,channels),channel)
  +height_positive = Order.le_lt_trans(0n,row,height_nat,Order.zero_le(row),row_inside)
  +width_positive = Order.le_lt_trans(0n,column,width_nat,Order.zero_le(column),column_inside)
  +channels_positive = Order.le_lt_trans(0n,channel,channels,Order.zero_le(channel),channel_inside)
  +spatial_positive = RowMajor.multiply_lt(0n,height_nat,width_nat,height_positive,width_positive)
  +frame_positive = RowMajor.multiply_lt(0n,channels,spatial,channels_positive,spatial_positive)
  +pixel_inside = RowMajor.radix(row,height_nat,column,width_nat,row_inside,column_inside)
  +frame_inside = RowMajor.radix(channel,channels,pixel,spatial,channel_inside,pixel_inside)
  +image_exact = Equal.trans(Nat,Nat.div(U32.to_nat(index),frame),Nat.div(RowMajor.index(batch_channel,height_nat,width_nat,row,column),frame),image,
    Equal.cong(Nat,Nat,value => Nat.div(value,frame),U32.to_nat(index),RowMajor.index(batch_channel,height_nat,width_nat,row,column),encoded),
    %Equal.sym(Nat,RowMajor.index(batch_channel,height_nat,width_nat,row,column),Nat.add(Nat.mul(image,frame),RowMajor.index(channel,height_nat,width_nat,row,column)),image_frame(image,channels,channel,height_nat,width_nat,row,column)) : {Nat.div(_,frame) == image : Nat}
    BandCoordinates.quotient(image,RowMajor.index(channel,height_nat,width_nat,row,column),frame,frame_positive,frame_inside))
  +channel_exact = Equal.trans(Nat,Nat.mod(Nat.div(U32.to_nat(index),spatial),channels),Nat.mod(batch_channel,channels),channel,
    %Equal.sym(Nat,U32.to_nat(index),RowMajor.index(batch_channel,height_nat,width_nat,row,column),encoded) : {Nat.mod(Nat.div(_,spatial),channels) == Nat.mod(batch_channel,channels) : Nat}
    Equal.cong(Nat,Nat,value => Nat.mod(value,channels),Nat.div(RowMajor.index(batch_channel,height_nat,width_nat,row,column),spatial),batch_channel,BandCoordinates.quotient(batch_channel,pixel,spatial,spatial_positive,pixel_inside)),
    BandCoordinates.remainder(image,channel,channels,channels_positive,channel_inside))
  +pixel_exact = {
    %Equal.sym(Nat,U32.to_nat(index),RowMajor.index(batch_channel,height_nat,width_nat,row,column),encoded) : {Nat.mod(_,spatial) == pixel : Nat}
    BandCoordinates.remainder(batch_channel,pixel,spatial,spatial_positive,pixel_inside)
    : {Nat.mod(U32.to_nat(index),spatial) == pixel : Nat}}
  +row_exact = Equal.trans(Nat,Nat.div(Nat.mod(U32.to_nat(index),spatial),width_nat),Nat.div(pixel,width_nat),row,Equal.cong(Nat,Nat,value => Nat.div(value,width_nat),Nat.mod(U32.to_nat(index),spatial),pixel,pixel_exact),BandCoordinates.quotient(row,column,width_nat,width_positive,column_inside))
  +column_exact = {
    %Equal.sym(Nat,U32.to_nat(index),RowMajor.index(batch_channel,height_nat,width_nat,row,column),encoded) : {Nat.mod(_,width_nat) == column : Nat}
    %RowMajor.mixed(batch_channel,height_nat,width_nat,row,column) : {Nat.mod(_,width_nat) == column : Nat}
    BandCoordinates.remainder(Nat.add(Nat.mul(batch_channel,height_nat),row),column,width_nat,width_positive,column_inside)
    : {Nat.mod(U32.to_nat(index),width_nat) == column : Nat}}
  %Equal.sym(Nat,Nat.div(U32.to_nat(index),frame),image,image_exact) : {ordered_sample(~Element,~original,parts,height,width,_,Nat.mod(Nat.div(U32.to_nat(index),spatial),channels),Nat.div(Nat.mod(U32.to_nat(index),spatial),width_nat),Nat.mod(U32.to_nat(index),width_nat)) == ordered_sample(~Element,~original,parts,height,width,image,channel,row,column) : Element}
  %Equal.sym(Nat,Nat.mod(Nat.div(U32.to_nat(index),spatial),channels),channel,channel_exact) : {ordered_sample(~Element,~original,parts,height,width,image,_,Nat.div(Nat.mod(U32.to_nat(index),spatial),width_nat),Nat.mod(U32.to_nat(index),width_nat)) == ordered_sample(~Element,~original,parts,height,width,image,channel,row,column) : Element}
  %Equal.sym(Nat,Nat.div(Nat.mod(U32.to_nat(index),spatial),width_nat),row,row_exact) : {ordered_sample(~Element,~original,parts,height,width,image,channel,_,Nat.mod(U32.to_nat(index),width_nat)) == ordered_sample(~Element,~original,parts,height,width,image,channel,row,column) : Element}
  %Equal.sym(Nat,Nat.mod(U32.to_nat(index),width_nat),column,column_exact) : {ordered_sample(~Element,~original,parts,height,width,image,channel,row,_) == ordered_sample(~Element,~original,parts,height,width,image,channel,row,column) : Element}
  {==}

def count_projection(+geometry: Bands.Geometry,+rows: U32,+length: Nat,count: BandShape.Count<geometry,rows,length>) -> {U32.to_nat(Bands.local_count(geometry,rows)) == length : Nat}:
  BandShape.Count{exact,allocatable} = count
  exact

def metadata_length(geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,metadata: BandShape.Leaf(geometry,first,rows,length)) -> {length == BandConcatExecution.channel_length(BandRange.Request{geometry,first,rows,0}) : Nat}:
  Bands.Geometry{images,channels,height,width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  length_exact

def local_count_exact(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  +metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Ready<Element,input>) ->
  {U32.to_nat(Bands.local_count(geometry,rows)) == BandConcatExecution.channel_length(BandRange.Request{geometry,first,rows,0}) : Nat}:
  Equal.trans(Nat,U32.to_nat(Bands.local_count(geometry,rows)),Buffer.length(~Element,input),BandConcatExecution.channel_length(BandRange.Request{geometry,first,rows,0}),
    count_projection(geometry,rows,Buffer.length(~Element,input),BandShape.count(~Element,geometry,first,rows,input,width_positive,metadata,ready)),metadata_length(geometry,first,rows,Buffer.length(~Element,input),metadata))

def exact_copy_room(request: BandRange.Request,+row_capacity: U32,+output_capacity: U32,target: BandRange.Target(request,row_capacity,output_capacity),
  +exact: {U32.to_nat(Bands.local_count(BandRange.request_geometry(request),BandRange.request_rows(request))) == BandConcatExecution.channel_length(request) : Nat}) ->
  {Nat.is_le(Nat.add(U32.to_nat(BandRange.request_offset(request)),U32.to_nat(Bands.local_count(BandRange.request_geometry(request),BandRange.request_rows(request)))),U32.to_nat(output_capacity)) == True{} : Bool}:
  BandRange.Request{Bands.Geometry{images,+channels,height,+width},first,+rows,+offset} = request
  BandRange.TargetFields{row_capacity_ok,output_capacity_ok,channels_positive,width_positive,rows_room,storage_room} = target
  %Equal.sym(Nat,U32.to_nat(Bands.local_count(BandRange.request_geometry(request),rows)),BandConcatExecution.channel_length(request),exact) : {Nat.is_le(Nat.add(U32.to_nat(offset),_),U32.to_nat(output_capacity)) == True{} : Bool} storage_room

def channel_length_offset(geometry: Bands.Geometry,+first: U32,+rows: U32,+offset: U32) ->
  {BandConcatExecution.channel_length(BandRange.Request{geometry,first,rows,offset}) == BandConcatExecution.channel_length(BandRange.Request{geometry,first,rows,0}) : Nat}:
  Bands.Geometry{images,channels,height,width} = geometry
  {==}

def aligned_leaf_channels(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+channel: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+limits: Machine.Limits,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+input_ready: Buffer.Ready<Element,input>,+output_ready: Buffer.Ready<Element,output>,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input)),
  +target: BandRange.Target(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,channel,rows)},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output)))) ->
  BandConcatExecution.TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,BandConcatExecution.aligned_model(~Element,geometry,channel,Model.content(~Element,Model.Leaf{input,first,rows}),Model.content(~Element,Model.Leaf{output,first,rows}))):
  +request = {BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,channel,rows)} : BandRange.Request}
  +exact = Equal.trans(Nat,U32.to_nat(Bands.local_count(geometry,rows)),BandConcatExecution.channel_length(BandRange.Request{geometry,first,rows,0}),BandConcatExecution.channel_length(request),
    local_count_exact(~Element,geometry,first,rows,input,width_positive,metadata,input_ready),
    Equal.sym(Nat,BandConcatExecution.channel_length(request),BandConcatExecution.channel_length(BandRange.Request{geometry,first,rows,0}),channel_length_offset(geometry,first,rows,BandConcat.channel_offset(geometry,channel,rows))))
  BandConcatExecution.copied_cells(~Element,~Logical,~original,logical,geometry,first,rows,BandConcat.channel_offset(geometry,channel,rows),input,output,width_positive,metadata,input_ready,output_ready,values,
    exact_copy_room(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output)),target,exact))

def aligned_channels(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+channel: U32,source: Model.Model<Element>,output: Model.Model<Element>,+limits: Machine.Limits,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,source)),source_ready: Model.Ready(~Element,source),ready: Model.Ready(~Element,output),
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),targets: BandConcatExecution.RangeTargets(~Element,geometry,channel,limits,output),
  +same: {Bands.same_plan(Model.layout(~Element,source),Model.layout(~Element,output)) == True{} : Bool}) ->
  BandConcatExecution.TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,BandConcatExecution.aligned_model(~Element,geometry,channel,Model.content(~Element,source),Model.content(~Element,output))):
  match source output:
    case Model.Leaf{Buffer.Model{+input,+input_length,+input_certificate},+first,+rows} Model.Leaf{Buffer.Model{+out,+output_length,+output_certificate},+out_first,+out_rows}:
      Shape.Both{+target,target_metadata} = targets
      +first_equal = ShapeLaws.native_equal(first,out_first,Boolean.left(U32.is_eq(first,out_first),U32.is_eq(rows,out_rows),same))
      +rows_equal = ShapeLaws.native_equal(rows,out_rows,Boolean.right(U32.is_eq(first,out_first),U32.is_eq(rows,out_rows),same))
      %first_equal : BandConcatExecution.TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,BandConcatExecution.aligned_model(~Element,geometry,channel,Model.content(~Element,Model.Leaf{Buffer.Model{input,input_length,input_certificate},first,rows}),Model.content(~Element,Model.Leaf{Buffer.Model{out,output_length,output_certificate},_,out_rows})))
      %rows_equal : BandConcatExecution.TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,BandConcatExecution.aligned_model(~Element,geometry,channel,Model.content(~Element,Model.Leaf{Buffer.Model{input,input_length,input_certificate},first,rows}),Model.content(~Element,Model.Leaf{Buffer.Model{out,output_length,output_certificate},first,_})))
      aligned_leaf_channels(~Element,~Logical,~original,logical,geometry,channel,first,rows,Buffer.Model{input,input_length,input_certificate},Buffer.Model{out,output_length,output_certificate},limits,width_positive,metadata,source_ready,ready,values,
        %Equal.sym(U32,first,out_first,first_equal) : BandRange.Target(BandRange.Request{geometry,_,rows,BandConcat.channel_offset(geometry,channel,rows)},Machine.capacity(limits),Tree.capacity(Element,out))
        %Equal.sym(U32,rows,out_rows,rows_equal) : BandRange.Target(BandRange.Request{geometry,out_first,_,BandConcat.channel_offset(geometry,channel,_)},Machine.capacity(limits),Tree.capacity(Element,out)) target)
    case Model.Fork{+left,+right} Model.Fork{+out_left,+out_right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_source_ready,right_source_ready} = source_ready
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_values,right_values} = values
      Shape.Both{left_targets,right_targets} = targets
      Shape.Both{aligned_channels(~Element,~Logical,~original,logical,geometry,channel,left,out_left,limits,width_positive,left_metadata,left_source_ready,left_ready,left_values,left_targets,Boolean.left(Bands.same_plan(Model.layout(~Element,left),Model.layout(~Element,out_left)),Bands.same_plan(Model.layout(~Element,right),Model.layout(~Element,out_right)),same)),
        aligned_channels(~Element,~Logical,~original,logical,geometry,channel,right,out_right,limits,width_positive,right_metadata,right_source_ready,right_ready,right_values,right_targets,Boolean.right(Bands.same_plan(Model.layout(~Element,left),Model.layout(~Element,out_left)),Bands.same_plan(Model.layout(~Element,right),Model.layout(~Element,out_right)),same))}
    case Model.Leaf{buffer,first,rows} Model.Fork{left,right}:
      Empty.absurd(BandConcatExecution.TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,Model.content(~Element,Model.Fork{left,right})),WordEncoding.false_true(same))
    case Model.Fork{left,right} Model.Leaf{buffer,first,rows}:
      Empty.absurd(BandConcatExecution.TreeChannelValues(~Element,~Logical,~original,logical,geometry,channel,Model.content(~Element,Model.Leaf{buffer,first,rows})),WordEncoding.false_true(same))

def image_inside_band(+images: U32,+height: U32,+first: U32,+rows: U32,+image: Nat,+local_first: Nat,+row: Nat,
  +first_exact: {U32.to_nat(first) == Nat.add(Nat.mul(image,U32.to_nat(height)),local_first) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool}) -> {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool}:
  RowMajor.row_inside(image,U32.to_nat(images),U32.to_nat(height),local_first,
    %first_exact : {Nat.is_lt(_,Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool}
    Bounds.nat_trans(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height)),
      %Addition.zero_right(U32.to_nat(first)) : {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}
      Order.offset_lt(U32.to_nat(first),0n,U32.to_nat(rows),Order.le_lt_trans(0n,row,U32.to_nat(rows),Order.zero_le(row),row_inside)),rows_fit))

def shifted_local_index(+prefix: Nat,+channel: Nat,+rows: Nat,+width: Nat,+row: Nat,+column: Nat) ->
  {Nat.add(Nat.mul(prefix,Nat.mul(rows,width)),RowMajor.index(channel,rows,width,row,column)) == RowMajor.index(Nat.add(prefix,channel),rows,width,row,column) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(prefix,channel),Nat.mul(rows,width)),Nat.add(Nat.mul(prefix,Nat.mul(rows,width)),Nat.mul(channel,Nat.mul(rows,width))),RowMajor.distrib(prefix,channel,Nat.mul(rows,width))) : {Nat.add(Nat.mul(prefix,Nat.mul(rows,width)),RowMajor.index(channel,rows,width,row,column)) == Nat.add(_,Nat.add(Nat.mul(row,width),column)) : Nat}
  Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(prefix,Nat.mul(rows,width)),Nat.mul(channel,Nat.mul(rows,width))),Nat.add(Nat.mul(row,width),column)),Nat.add(Nat.mul(prefix,Nat.mul(rows,width)),RowMajor.index(channel,rows,width,row,column)),Addition.associative(Nat.mul(prefix,Nat.mul(rows,width)),Nat.mul(channel,Nat.mul(rows,width)),Nat.add(Nat.mul(row,width),column)))

# The pixel of a part and the same pixel of the concatenation, moved by the
# channels of the earlier parts, have the same value. The premises are the
# output domain and the ordered logical lookup; no equality about the storage
# is a premise.
def oracle_pixel(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,+input: BandConcatExecution.TensorContent<Element>,+root: BandConcatExecution.PartContents<Element>,
  +images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+prefix_channel: U32,+limits: Machine.Limits,+capacity: U32,+channel: Nat,+global: Nat,+column: Nat,
  +output_domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,
  +source_count: {BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) == source_channels : U32},+total_exact: {U32.to_nat(output_channels) == BandConcatExecution.total_channels(~Element,root) : Nat},
  +channel_room: {Nat.is_le(Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool},
  source_position: ConcatOrderedSources.SourcePosition<Element,Unit,channel_extent(~Element),Unit{},input,root,U32.to_nat(prefix_channel)>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(source_channels)) == True{} : Bool},global_inside: {Nat.is_lt(global,Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  {original(input,U32.from_nat(BandValues.pixel(Bands.Geometry{images,source_channels,height,width},channel,global,column))) ==
    global_concat_value(~Element,~original,LogicalConcat{root,height,width},U32.from_nat(BandValues.pixel(Bands.Geometry{images,output_channels,height,width},Nat.add(U32.to_nat(prefix_channel),channel),global,column))) : Element}:
  BandGather.Domain{capacity_ok,channels_positive,+height_positive,width_positive,machine_room,allocation_room} = output_domain
  +height_nat = U32.to_nat(height)
  +width_nat = U32.to_nat(width)
  +image = Nat.div(global,height_nat)
  +row = Nat.mod(global,height_nat)
  +output_channel = Nat.add(U32.to_nat(prefix_channel),channel)
  +output_channel_inside = Bounds.nat_trans(output_channel,Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels),Order.offset_lt(U32.to_nat(prefix_channel),channel,U32.to_nat(source_channels),channel_inside),channel_room)
  +source_word = U32.from_nat(BandValues.pixel(Bands.Geometry{images,source_channels,height,width},channel,global,column))
  +output_word = U32.from_nat(BandValues.pixel(Bands.Geometry{images,output_channels,height,width},output_channel,global,column))
  +output_encoded = {
    %total_exact : {U32.to_nat(output_word) == RowMajor.index(Nat.add(Nat.mul(image,_),output_channel),height_nat,width_nat,row,column) : Nat}
    BandGather.pixel_exact(Bands.Geometry{images,output_channels,height,width},limits,capacity,output_channel,global,column,output_domain,output_channel_inside,global_inside,column_inside)
    : {U32.to_nat(output_word) == RowMajor.index(Nat.add(Nat.mul(image,BandConcatExecution.total_channels(~Element,root)),output_channel),height_nat,width_nat,row,column) : Nat}}
  Equal.trans(Element,original(input,source_word),source_pixel(~Element,~original,input,height,width,image,channel,row,column),global_concat_value(~Element,~original,LogicalConcat{root,height,width},output_word),
    %Equal.sym(U32,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),source_channels,source_count) : {original(input,source_word) == original(input,U32.from_nat(RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(_)),channel),height_nat,width_nat,row,column))) : Element}
    {==},
    Equal.trans(Element,source_pixel(~Element,~original,input,height,width,image,channel,row,column),ordered_sample(~Element,~original,root,height,width,image,output_channel,row,column),global_concat_value(~Element,~original,LogicalConcat{root,height,width},output_word),
      ConcatOrderedSources.source_value(~Element,~Unit,~channel_extent(~Element),~Sigma<&2,&2,U32,_ => U32>,~channel_sample(~Element,~original),Unit{},(height,width),input,root,U32.to_nat(prefix_channel),image,channel,row,column,
        %Equal.sym(U32,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),source_channels,source_count) : {Nat.is_lt(channel,U32.to_nat(_)) == True{} : Bool} channel_inside,source_position),
      Equal.sym(Element,global_concat_value(~Element,~original,LogicalConcat{root,height,width},output_word),ordered_sample(~Element,~original,root,height,width,image,output_channel,row,column),
        global_concat_pixel(~Element,~original,root,height,width,image,output_channel,row,column,output_word,output_encoded,%total_exact : {Nat.is_lt(output_channel,_) == True{} : Bool} output_channel_inside,
          DivisionOrder.remainder_small(global,height_nat,height_positive),column_inside))))

def tensor_original(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,input: BandConcatExecution.TensorContent<Element>,index: U32) -> Element:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}: Buffer.value(~Element,index,buffer)
    case BandConcatExecution.BandedContent{bands,geometry,shape,fill,status,workers}: band_value(bands,geometry,index)

def SourceContract(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+images: U32,+height: U32,+width: U32,input: BandConcatExecution.TensorContent<Element>) -> Data:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}: Unit
    case BandConcatExecution.BandedContent{+bands,+geometry,+shape,+fill,+status,+workers}:
      Shape.Both<BandShape.WellFormed<Element,Bands.Geometry{images,BandConcat.channels(shape),height,width},bands>,
        BandValues.Values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),BandConcatExecution.BandedContent{bands,geometry,shape,fill,status,workers},Bands.Geometry{images,BandConcat.channels(shape),height,width},bands)>

def SourceContracts(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+images: U32,+height: U32,+width: U32,parts: BandConcatExecution.PartContents<Element>) -> Data:
  match parts:
    case BandConcatExecution.SingleContent{input}: SourceContract(~Element,~band_value,images,height,width,input)
    case BandConcatExecution.ConsContent{input,rest}: Shape.Both<SourceContract(~Element,~band_value,images,height,width,input),SourceContracts(~Element,~band_value,images,height,width,rest)>

def source_width(geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) -> {BandShape.width_positive(geometry) == True{} : Bool}:
  Bands.Geometry{images,channels,height,width} = geometry
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  width_positive

# The cell an append wrote at its channel offset is the cell of the output
# leaf at the channel moved by the prefix, and holds the concat oracle there.
def extended_cell(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,+input: BandConcatExecution.TensorContent<Element>,+root: BandConcatExecution.PartContents<Element>,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+prefix_channel: U32,+first: U32,+rows: U32,+output: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,+channel: Nat,+row: Nat,+column: Nat,
  +output_domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+metadata: BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.length(~Element,output)),ready: Buffer.Ready<Element,output>,
  +source_count: {BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) == source_channels : U32},+total_exact: {U32.to_nat(output_channels) == BandConcatExecution.total_channels(~Element,root) : Nat},
  +channel_room: {Nat.is_le(Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool},
  source_position: ConcatOrderedSources.SourcePosition<Element,Unit,channel_extent(~Element),Unit{},input,root,U32.to_nat(prefix_channel)>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(source_channels)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool},
  cell: BandConcatExecution.ChannelAt(~Element,~BandConcatExecution.TensorContent<Element>,~original,input,BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)},Buffer.content(~Element,output),channel,row,column)) ->
  BandValues.At(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~original),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.tree(~Element,output),Buffer.length(~Element,output),Nat.add(U32.to_nat(prefix_channel),channel),row,column):
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  +plane = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +output_channel = Nat.add(U32.to_nat(prefix_channel),channel)
  +output_channel_inside = Bounds.nat_trans(output_channel,Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels),Order.offset_lt(U32.to_nat(prefix_channel),channel,U32.to_nat(source_channels),channel_inside),channel_room)
  +global = Nat.add(U32.to_nat(first),row)
  +local_position = RowMajor.index(channel,U32.to_nat(rows),U32.to_nat(width),row,column)
  +position = RowMajor.index(output_channel,U32.to_nat(rows),U32.to_nat(width),row,column)
  +offset = U32.to_nat(BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows))
  +source_word = U32.from_nat(BandValues.pixel(Bands.Geometry{images,source_channels,height,width},channel,global,column))
  +address_exact = Equal.trans(Nat,Nat.add(offset,local_position),Nat.add(Nat.mul(U32.to_nat(prefix_channel),plane),local_position),position,
    Equal.cong(Nat,Nat,value => Nat.add(value,local_position),offset,Nat.mul(U32.to_nat(prefix_channel),plane),
      prefix_offset(~Element,images,source_channels,output_channels,height,width,prefix_channel,first,rows,output,width_positive,metadata,ready,channel_room)),
    shifted_local_index(U32.to_nat(prefix_channel),channel,U32.to_nat(rows),U32.to_nat(width),row,column))
  Shape.Both{
    BandValues.position_within(Bands.Geometry{images,output_channels,height,width},rows,Buffer.length(~Element,output),output_channel,row,column,length_exact,output_channel_inside,row_inside,column_inside),
    Equal.trans(Element,Tree.read(Element,Buffer.tree(~Element,output),U32.from_nat(position)),original(input,source_word),global_concat_value(~Element,~original,LogicalConcat{root,height,width},U32.from_nat(BandValues.pixel(Bands.Geometry{images,output_channels,height,width},output_channel,global,column))),
      %address_exact : {Tree.read(Element,Buffer.tree(~Element,output),U32.from_nat(_)) == original(input,source_word) : Element} cell,
      oracle_pixel(~Element,~original,input,root,images,source_channels,output_channels,height,width,prefix_channel,limits,capacity,channel,global,column,output_domain,source_count,total_exact,channel_room,source_position,channel_inside,
        Bounds.nat_trans(global,Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height)),Order.offset_lt(U32.to_nat(first),row,U32.to_nat(rows),row_inside),rows_fit),column_inside))}

# What an append supplies to every cell of the channels it adds.
type Extended<-Element: Data,-original: BandConcatExecution.TensorContent<Element> -> U32 -> Element> is Data:
  Extended{input: BandConcatExecution.TensorContent<Element>,root: BandConcatExecution.PartContents<Element>,images: U32,source_channels: U32,output_channels: U32,height: U32,width: U32,prefix_channel: U32,first: U32,rows: U32,output: Buffer.Model<Element>,limits: Machine.Limits,capacity: U32,
    output_domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},metadata: BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.length(~Element,output)),ready: Buffer.Ready<Element,output>,
    added: BandConcatExecution.ChannelValues(~Element,~BandConcatExecution.TensorContent<Element>,~original,input,BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)},Buffer.content(~Element,output)),
    source_count: {BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) == source_channels : U32},total_exact: {U32.to_nat(output_channels) == BandConcatExecution.total_channels(~Element,root) : Nat},
    channel_room: {Nat.is_le(Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool},
    source_position: ConcatOrderedSources.SourcePosition<Element,Unit,channel_extent(~Element),Unit{},input,root,U32.to_nat(prefix_channel)>}

def extended_geometry(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,evidence: Extended<Element,original>) -> Bands.Geometry:
  Extended{input,root,images,source_channels,output_channels,height,width,prefix_channel,first,rows,output,limits,capacity,output_domain,width_positive,metadata,ready,added,source_count,total_exact,channel_room,source_position} = evidence
  Bands.Geometry{images,source_channels,height,width}

def extended_rows(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,evidence: Extended<Element,original>) -> U32:
  Extended{input,root,images,source_channels,output_channels,height,width,prefix_channel,first,rows,output,limits,capacity,output_domain,width_positive,metadata,ready,added,source_count,total_exact,channel_room,source_position} = evidence
  rows

def ExtendedCell(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,+evidence: Extended<Element,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  Extended{+input,+root,+images,+source_channels,+output_channels,+height,+width,+prefix_channel,+first,+rows,+output,+limits,+capacity,+output_domain,+width_positive,+metadata,+ready,+added,+source_count,+total_exact,+channel_room,+source_position} = evidence
  BandValues.At(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~original),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.tree(~Element,output),Buffer.length(~Element,output),Nat.add(U32.to_nat(prefix_channel),channel),row,column)

def extended_leaf_cell(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,+evidence: Extended<Element,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(extended_geometry(~Element,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(extended_rows(~Element,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(extended_geometry(~Element,~original,evidence))) == True{} : Bool}) ->
  ExtendedCell(~Element,~original,evidence,channel,row,column):
  Extended{+input,+root,+images,+source_channels,+output_channels,+height,+width,+prefix_channel,+first,+rows,+output,+limits,+capacity,+output_domain,+width_positive,+metadata,+ready,+added,+source_count,+total_exact,+channel_room,+source_position} = evidence
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  extended_cell(~Element,~original,input,root,images,source_channels,output_channels,height,width,prefix_channel,first,rows,output,limits,capacity,channel,row,column,output_domain,width_positive,metadata,ready,source_count,total_exact,channel_room,source_position,channel_inside,row_inside,column_inside,
    BandConcatExecution.channel_cell(~Element,~BandConcatExecution.TensorContent<Element>,~original,input,BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)},Buffer.content(~Element,output),channel,row,column,added,channel_inside,row_inside,column_inside))

def extend_channels(~Element: Data,~original: BandConcatExecution.TensorContent<Element> -> U32 -> Element,+input: BandConcatExecution.TensorContent<Element>,+root: BandConcatExecution.PartContents<Element>,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+prefix_channel: U32,+next_channel: U32,output: Model.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  +output_domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,
  +metadata: BandShape.Contents(~Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,output)),+ready: Model.Ready(~Element,output),
  previous: PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~original),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},prefix_channel,Model.content(~Element,output)),
  added: BandConcatExecution.TreeChannelValues(~Element,~BandConcatExecution.TensorContent<Element>,~original,input,Bands.Geometry{images,source_channels,height,width},prefix_channel,Model.content(~Element,output)),
  +source_count: {BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) == source_channels : U32},+total_exact: {U32.to_nat(output_channels) == BandConcatExecution.total_channels(~Element,root) : Nat},+channel_room: {Nat.is_le(Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool},
  +step_exact: {U32.to_nat(next_channel) == Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)) : Nat},
  +source_position: ConcatOrderedSources.SourcePosition<Element,Unit,channel_extent(~Element),Unit{},input,root,U32.to_nat(prefix_channel)>) ->
  PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~original),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},next_channel,Model.content(~Element,output)):
  match output:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      +buffer = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
      %Equal.sym(Nat,U32.to_nat(next_channel),Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),step_exact) : BandValues.Channels(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~original),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},first,rows,tree,length,_)
      BandValues.channels_joined(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~original),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},first,rows,tree,length,U32.to_nat(prefix_channel),U32.to_nat(source_channels),previous,
        %Addition.zero_right(U32.to_nat(prefix_channel)) : Members.All(Nat,{channel => BandValues.ChannelCells(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~original),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},first,rows,tree,length,channel) : Nat -> Data},Ranges.range(U32.to_nat(source_channels),_))
        BandValues.all_shifted({channel => BandValues.ChannelCells(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~original),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},first,rows,tree,length,channel) : Nat -> Data},U32.to_nat(prefix_channel),U32.to_nat(source_channels),0n,
          BandValues.cells(~Extended<Element,original>,~ExtendedCell(~Element,~original),~extended_geometry(~Element,~original),~extended_rows(~Element,~original),~extended_leaf_cell(~Element,~original),
            Extended{input,root,images,source_channels,output_channels,height,width,prefix_channel,first,rows,buffer,limits,capacity,output_domain,source_width(Bands.Geometry{images,output_channels,height,width},limits,capacity,output_domain),metadata,ready,added,source_count,total_exact,channel_room,source_position})))
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_previous,right_previous} = previous
      Shape.Both{left_added,right_added} = added
      Shape.Both{extend_channels(~Element,~original,input,root,images,source_channels,output_channels,height,width,prefix_channel,next_channel,left,limits,capacity,output_domain,left_metadata,left_ready,left_previous,left_added,source_count,total_exact,channel_room,step_exact,source_position),
        extend_channels(~Element,~original,input,root,images,source_channels,output_channels,height,width,prefix_channel,next_channel,right,limits,capacity,output_domain,right_metadata,right_ready,right_previous,right_added,source_count,total_exact,channel_room,step_exact,source_position)}

# What an aligned append supplies to every cell below the cursor.
type AlignedPrefix<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  AlignedPrefix{logical: Logical,images: U32,source_channels: U32,output_channels: U32,height: U32,width: U32,prefix_channel: U32,first: U32,rows: U32,input: Buffer.Model<Element>,output: Buffer.Model<Element>,limits: Machine.Limits,width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
    source_metadata: BandShape.Leaf(Bands.Geometry{images,source_channels,height,width},first,rows,Buffer.length(~Element,input)),metadata: BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.length(~Element,output)),source_ready: Buffer.Ready<Element,input>,ready: Buffer.Ready<Element,output>,
    target: BandRange.Target(BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
    values: BandValues.Channels(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.tree(~Element,output),Buffer.length(~Element,output),U32.to_nat(prefix_channel)),channel_room: {Nat.is_le(Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}}

def aligned_prefix_cells_geometry(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: AlignedPrefix<Element,Logical,original>) -> Bands.Geometry:
  AlignedPrefix{logical,images,source_channels,output_channels,height,width,prefix_channel,first,rows,input,output,limits,width_positive,source_metadata,metadata,source_ready,ready,target,values,channel_room} = evidence
  Bands.Geometry{images,prefix_channel,height,width}

def aligned_prefix_cells_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: AlignedPrefix<Element,Logical,original>) -> U32:
  AlignedPrefix{logical,images,source_channels,output_channels,height,width,prefix_channel,first,rows,input,output,limits,width_positive,source_metadata,metadata,source_ready,ready,target,values,channel_room} = evidence
  rows

def AlignedPrefixCell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: AlignedPrefix<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  AlignedPrefix{+logical,+images,+source_channels,+output_channels,+height,+width,+prefix_channel,+first,+rows,+input,+output,+limits,+width_positive,+source_metadata,+metadata,+source_ready,+ready,+target,+values,+channel_room} = evidence
  BandValues.At(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,
    Buffer.content_tree(~Element,BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(Bands.Geometry{images,source_channels,height,width},rows),0,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows),BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output))))),
    Buffer.content_length(~Element,BandGather.output(~Element,StorageCopyLaws.model(~Element,Bands.local_count(Bands.Geometry{images,source_channels,height,width},rows),0,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows),BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,output))))),channel,row,column)

def aligned_prefix_cells_leaf_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: AlignedPrefix<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(aligned_prefix_cells_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(aligned_prefix_cells_rows(~Element,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(aligned_prefix_cells_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool}) ->
  AlignedPrefixCell(~Element,~Logical,~original,evidence,channel,row,column):
  AlignedPrefix{+logical,+images,+source_channels,+output_channels,+height,+width,+prefix_channel,+first,+rows,+input,+output,+limits,+width_positive,+source_metadata,+metadata,+source_ready,+ready,+target,+values,+channel_room} = evidence
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  BandConcatExecution.copied_prefix_cell(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,Bands.local_count(Bands.Geometry{images,source_channels,height,width},rows),BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows),input,output,channel,row,column,ready,
    exact_copy_room(BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output)),target,local_count_exact(~Element,Bands.Geometry{images,source_channels,height,width},first,rows,input,width_positive,source_metadata,source_ready)),
    prefix_cell_before(~Element,images,source_channels,output_channels,height,width,prefix_channel,first,rows,output,channel,row,column,width_positive,metadata,ready,channel_room,channel_inside,row_inside,column_inside),
    BandValues.channels_cell(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.tree(~Element,output),Buffer.length(~Element,output),U32.to_nat(prefix_channel),channel,row,column,values,channel_inside,row_inside,column_inside))

def aligned_prefix_leaf(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+prefix_channel: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+limits: Machine.Limits,
  width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},source_metadata: BandShape.Leaf(Bands.Geometry{images,source_channels,height,width},first,rows,Buffer.length(~Element,input)),metadata: BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.length(~Element,output)),source_ready: Buffer.Ready<Element,input>,ready: Buffer.Ready<Element,output>,
  values: BandValues.Channels(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.tree(~Element,output),Buffer.length(~Element,output),U32.to_nat(prefix_channel)),
  target: BandRange.Target(BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
  channel_room: {Nat.is_le(Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}) ->
  PrefixValues(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},prefix_channel,BandConcatExecution.aligned_model(~Element,Bands.Geometry{images,source_channels,height,width},prefix_channel,Model.content(~Element,Model.Leaf{input,first,rows}),Model.content(~Element,Model.Leaf{output,first,rows}))):
  BandValues.cells(~AlignedPrefix<Element,Logical,original>,~AlignedPrefixCell(~Element,~Logical,~original),~aligned_prefix_cells_geometry(~Element,~Logical,~original),~aligned_prefix_cells_rows(~Element,~Logical,~original),~aligned_prefix_cells_leaf_cell(~Element,~Logical,~original),
    AlignedPrefix{logical,images,source_channels,output_channels,height,width,prefix_channel,first,rows,input,output,limits,width_positive,source_metadata,metadata,source_ready,ready,target,values,channel_room})

def aligned_prefix(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,source: Model.Model<Element>,output: Model.Model<Element>,+limits: Machine.Limits,
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},source_metadata: BandShape.Contents(~Element,Bands.Geometry{images,source_channels,height,width},Model.content(~Element,source)),metadata: BandShape.Contents(~Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,output)),source_ready: Model.Ready(~Element,source),ready: Model.Ready(~Element,output),
  values: PrefixValues(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},channel,Model.content(~Element,output)),targets: BandConcatExecution.RangeTargets(~Element,Bands.Geometry{images,source_channels,height,width},channel,limits,output),
  +channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool},+same: {Bands.same_plan(Model.layout(~Element,source),Model.layout(~Element,output)) == True{} : Bool}) ->
  PrefixValues(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},channel,BandConcatExecution.aligned_model(~Element,Bands.Geometry{images,source_channels,height,width},channel,Model.content(~Element,source),Model.content(~Element,output))):
  match source output:
    case Model.Leaf{Buffer.Model{+input,+input_length,+input_certificate},+first,+rows} Model.Leaf{Buffer.Model{+out,+output_length,+output_certificate},+out_first,+out_rows}:
      Shape.Both{+target,target_metadata} = targets
      +first_equal = ShapeLaws.native_equal(first,out_first,Boolean.left(U32.is_eq(first,out_first),U32.is_eq(rows,out_rows),same))
      +rows_equal = ShapeLaws.native_equal(rows,out_rows,Boolean.right(U32.is_eq(first,out_first),U32.is_eq(rows,out_rows),same))
      %first_equal : PrefixValues(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},channel,BandConcatExecution.aligned_model(~Element,Bands.Geometry{images,source_channels,height,width},channel,Model.content(~Element,Model.Leaf{Buffer.Model{input,input_length,input_certificate},first,rows}),Model.content(~Element,Model.Leaf{Buffer.Model{out,output_length,output_certificate},_,out_rows})))
      %rows_equal : PrefixValues(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},channel,BandConcatExecution.aligned_model(~Element,Bands.Geometry{images,source_channels,height,width},channel,Model.content(~Element,Model.Leaf{Buffer.Model{input,input_length,input_certificate},first,rows}),Model.content(~Element,Model.Leaf{Buffer.Model{out,output_length,output_certificate},first,_})))
      aligned_prefix_leaf(~Element,~Logical,~original,logical,images,source_channels,output_channels,height,width,channel,first,rows,Buffer.Model{input,input_length,input_certificate},Buffer.Model{out,output_length,output_certificate},limits,width_positive,source_metadata,
        %Equal.sym(U32,first,out_first,first_equal) : BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},_,rows,output_length)
        %Equal.sym(U32,rows,out_rows,rows_equal) : BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},out_first,_,output_length) metadata,source_ready,ready,
        %Equal.sym(U32,first,out_first,first_equal) : BandValues.Channels(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},_,rows,out,output_length,U32.to_nat(channel))
        %Equal.sym(U32,rows,out_rows,rows_equal) : BandValues.Channels(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},out_first,_,out,output_length,U32.to_nat(channel)) values,
        %Equal.sym(U32,first,out_first,first_equal) : BandRange.Target(BandRange.Request{Bands.Geometry{images,source_channels,height,width},_,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},channel,rows)},Machine.capacity(limits),Tree.capacity(Element,out))
        %Equal.sym(U32,rows,out_rows,rows_equal) : BandRange.Target(BandRange.Request{Bands.Geometry{images,source_channels,height,width},out_first,_,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},channel,_)},Machine.capacity(limits),Tree.capacity(Element,out)) target,channel_room)
    case Model.Fork{+left,+right} Model.Fork{+out_left,+out_right}:
      Shape.Both{left_source_metadata,right_source_metadata} = source_metadata
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_source_ready,right_source_ready} = source_ready
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_values,right_values} = values
      Shape.Both{left_targets,right_targets} = targets
      Shape.Both{aligned_prefix(~Element,~Logical,~original,logical,images,source_channels,output_channels,height,width,channel,left,out_left,limits,width_positive,left_source_metadata,left_metadata,left_source_ready,left_ready,left_values,left_targets,channel_room,Boolean.left(Bands.same_plan(Model.layout(~Element,left),Model.layout(~Element,out_left)),Bands.same_plan(Model.layout(~Element,right),Model.layout(~Element,out_right)),same)),
        aligned_prefix(~Element,~Logical,~original,logical,images,source_channels,output_channels,height,width,channel,right,out_right,limits,width_positive,right_source_metadata,right_metadata,right_source_ready,right_ready,right_values,right_targets,channel_room,Boolean.right(Bands.same_plan(Model.layout(~Element,left),Model.layout(~Element,out_left)),Bands.same_plan(Model.layout(~Element,right),Model.layout(~Element,out_right)),same))}
    case Model.Leaf{buffer,first,rows} Model.Fork{left,right}: Empty.absurd(PrefixValues(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},channel,Model.content(~Element,Model.Fork{left,right})),WordEncoding.false_true(same))
    case Model.Fork{left,right} Model.Leaf{buffer,first,rows}: Empty.absurd(PrefixValues(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},channel,Model.content(~Element,Model.Leaf{buffer,first,rows})),WordEncoding.false_true(same))

def dense_query_encoding(+request: BandRange.Request,+capacity: U32,+channel: Nat,+index: Nat,+position: Nat,+valid: BandCopy.Valid<dense_region(request,capacity)>,+channel_inside: {Nat.is_lt(channel,U32.to_nat(BandRange.channel_count(request))) == True{} : Bool},
  +index_inside: {Nat.is_lt(index,U32.to_nat(BandCopy.count(dense_region(request,capacity)))) == True{} : Bool},+position_exact: {position == Nat.add(Nat.mul(channel,U32.to_nat(BandCopy.count(dense_region(request,capacity)))),index) : Nat}) ->
  {BandCopy.readback_query(dense_region(request,capacity),channel,index) == BandConcatExecution.channel_address(request,position) : U32}:
  BandRange.Request{Bands.Geometry{images,+channels,height,+width},first,+rows,+offset} = request
  Encoded.encoding(BandCopy.readback_query(dense_region(request,capacity),channel,index),Nat.add(U32.to_nat(BandRange.request_offset(request)),position),
    %Equal.sym(Nat,position,Nat.add(Nat.mul(channel,U32.to_nat(BandCopy.count(dense_region(request,capacity)))),index),position_exact) : {U32.to_nat(BandCopy.readback_query(dense_region(request,capacity),channel,index)) == Nat.add(U32.to_nat(BandRange.request_offset(request)),_) : Nat}
    Equal.trans(Nat,U32.to_nat(BandCopy.readback_query(dense_region(request,capacity),channel,index)),Nat.add(BandCopy.start(dense_region(request,capacity),channel),index),Nat.add(U32.to_nat(BandRange.request_offset(request)),Nat.add(Nat.mul(channel,U32.to_nat(BandCopy.count(dense_region(request,capacity)))),index)),
      BandCopy.readback_query_exact(dense_region(request,capacity),channel,index,valid,channel_inside,index_inside),
      %Equal.sym(Nat,Nat.add(U32.to_nat(offset),0n),U32.to_nat(offset),Addition.zero_right(U32.to_nat(offset))) : {Nat.add(Nat.add(_,Nat.mul(channel,U32.to_nat((rows * width : U32)))),index) == Nat.add(U32.to_nat(offset),Nat.add(Nat.mul(channel,U32.to_nat((rows * width : U32))),index)) : Nat}
      Addition.associative(U32.to_nat(offset),Nat.mul(channel,U32.to_nat((rows * width : U32))),index)))

# The destination query of a cell of a dense append holds the source value of
# its pixel: the shared copy schedule reads the source at the pixel index.
def dense_cell(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+geometry: Bands.Geometry,+first: U32,+rows: U32,+prefix_channel: U32,+input: Tree.Tree<Element>,+input_length: Nat,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,+output: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,+channel: Nat,+row: Nat,+column: Nat,
  +domain: BandGather.Domain<geometry,limits,capacity>,+target: BandRange.Target(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
  metadata: BandShape.Leaf(geometry,first,rows,BandConcatExecution.channel_length(BandRange.Request{geometry,first,rows,0})),ready: Buffer.Ready<Element,output>,
  +channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool}) ->
  BandConcatExecution.ChannelAt(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),BandConcatExecution.DenseContent{Buffer.Content{input,input_length},shape,fill,status,workers},BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)},
    Buffer.Content{Writes.pure(Element,BandConcatExecution.dense_writes(~Element,geometry,first,rows,prefix_channel,input),Buffer.tree(~Element,output)),Buffer.length(~Element,output)},channel,row,column):
  Bands.Geometry{+images,+channel_count,+height,+row_width} = geometry
  BandGather.Domain{capacity_ok,channels_positive,+height_positive,width_positive,machine_room,allocation_room} = domain
  +request = {BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)} : BandRange.Request}
  +call = {BandGather.Call{geometry,first,rows,BandConcatExecution.channel_length(request),capacity,Machine.half_word(limits)} : BandGather.Call}
  +buffer_capacity = Tree.capacity(Element,Buffer.tree(~Element,output))
  +width = U32.to_nat(row_width)
  +plane = BandCopy.count(dense_region(request,buffer_capacity))
  +index = Nat.add(Nat.mul(row,width),column)
  +position = RowMajor.index(channel,U32.to_nat(rows),width,row,column)
  +global = Nat.add(U32.to_nat(first),row)
  +pixel = U32.from_nat(BandValues.pixel(geometry,channel,global,column))
  +plane_exact = dense_plane_exact(request,Machine.capacity(limits),buffer_capacity,target)
  +index_inside = {
    %Equal.sym(Nat,U32.to_nat(plane),Nat.mul(U32.to_nat(rows),width),plane_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(row,U32.to_nat(rows),column,width,row_inside,column_inside) : {Nat.is_lt(index,U32.to_nat(plane)) == True{} : Bool}}
  +position_exact = {
    %Equal.sym(Nat,U32.to_nat(plane),Nat.mul(U32.to_nat(rows),width),plane_exact) : {position == Nat.add(Nat.mul(channel,_),index) : Nat}
    {==} : {position == Nat.add(Nat.mul(channel,U32.to_nat(plane)),index) : Nat}}
  %dense_query_encoding(request,buffer_capacity,channel,index,position,dense_target_valid(request,Machine.capacity(limits),buffer_capacity,target,index,index_inside),channel_inside,index_inside,position_exact) :
    {Tree.read(Element,Writes.pure(Element,BandConcatExecution.dense_writes(~Element,geometry,first,rows,prefix_channel,input),Buffer.tree(~Element,output)),_) == Tree.read(Element,input,pixel) : Element}
  %BandGather.leaf_address(call,Nat.div(global,U32.to_nat(height)),channel,Nat.mod(global,U32.to_nat(height)),column,row,
      dense_slice_facts(~Element,request,output,limits,capacity,domain,target,metadata,index,index_inside),
      channel_inside,DivisionOrder.remainder_small(global,U32.to_nat(height),height_positive),column_inside,row_inside,
      Equal.sym(Nat,global,Nat.add(Nat.mul(Nat.div(global,U32.to_nat(height)),U32.to_nat(height)),Nat.mod(global,U32.to_nat(height))),DivisionOrder.quotient_exact(global,U32.to_nat(height),height_positive))) :
    {Tree.read(Element,Writes.pure(Element,BandConcatExecution.dense_writes(~Element,geometry,first,rows,prefix_channel,input),Buffer.tree(~Element,output)),BandCopy.readback_query(dense_region(request,buffer_capacity),channel,index)) == Tree.read(Element,input,_) : Element}
  dense_native_channel_point(~Element,geometry,first,rows,prefix_channel,input,output,Machine.capacity(limits),target,ready,channel,index,channel_inside,index_inside)

# What a dense append supplies to every cell of the channels it adds.
type DenseCells<-Element: Data,-band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element> is Data:
  DenseCells{geometry: Bands.Geometry,first: U32,rows: U32,prefix_channel: U32,input: Tree.Tree<Element>,input_length: Nat,shape: List<&2,U32>,fill: Element,status: Tensors.Status,workers: U32,output: Buffer.Model<Element>,limits: Machine.Limits,capacity: U32,
    domain: BandGather.Domain<geometry,limits,capacity>,target: BandRange.Target(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
    metadata: BandShape.Leaf(geometry,first,rows,BandConcatExecution.channel_length(BandRange.Request{geometry,first,rows,0})),ready: Buffer.Ready<Element,output>}

def dense_cells_geometry(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,evidence: DenseCells<Element,band_value>) -> Bands.Geometry:
  DenseCells{geometry,first,rows,prefix_channel,input,input_length,shape,fill,status,workers,output,limits,capacity,domain,target,metadata,ready} = evidence
  geometry

def dense_cells_rows(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,evidence: DenseCells<Element,band_value>) -> U32:
  DenseCells{geometry,first,rows,prefix_channel,input,input_length,shape,fill,status,workers,output,limits,capacity,domain,target,metadata,ready} = evidence
  rows

def DenseCell(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+evidence: DenseCells<Element,band_value>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  DenseCells{+geometry,+first,+rows,+prefix_channel,+input,+input_length,+shape,+fill,+status,+workers,+output,+limits,+capacity,+domain,+target,+metadata,+ready} = evidence
  BandConcatExecution.ChannelAt(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),BandConcatExecution.DenseContent{Buffer.Content{input,input_length},shape,fill,status,workers},BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)},
    Buffer.Content{Writes.pure(Element,BandConcatExecution.dense_writes(~Element,geometry,first,rows,prefix_channel,input),Buffer.tree(~Element,output)),Buffer.length(~Element,output)},channel,row,column)

def dense_leaf_cell(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+evidence: DenseCells<Element,band_value>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(dense_cells_geometry(~Element,~band_value,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(dense_cells_rows(~Element,~band_value,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(dense_cells_geometry(~Element,~band_value,evidence))) == True{} : Bool}) ->
  DenseCell(~Element,~band_value,evidence,channel,row,column):
  DenseCells{+geometry,+first,+rows,+prefix_channel,+input,+input_length,+shape,+fill,+status,+workers,+output,+limits,+capacity,+domain,+target,+metadata,+ready} = evidence
  dense_cell(~Element,~band_value,geometry,first,rows,prefix_channel,input,input_length,shape,fill,status,workers,output,limits,capacity,channel,row,column,domain,target,metadata,ready,channel_inside,row_inside,column_inside)

def dense_tree_channels(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+geometry: Bands.Geometry,+prefix_channel: U32,+input: Buffer.Content<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,output: Model.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<geometry,limits,capacity>,ready: Model.Ready(~Element,output),targets: BandConcatExecution.RangeTargets(~Element,geometry,prefix_channel,limits,output)) ->
  BandConcatExecution.TreeChannelValues(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),BandConcatExecution.DenseContent{input,shape,fill,status,workers},geometry,prefix_channel,BandConcatExecution.tree_model(~Element,geometry,prefix_channel,BandConcatExecution.DenseContent{input,shape,fill,status,workers},Model.content(~Element,output))):
  Buffer.Content{+input_tree,+input_length} = input
  match output:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      Shape.Both{target,metadata} = targets
      BandValues.cells(~DenseCells<Element,band_value>,~DenseCell(~Element,~band_value),~dense_cells_geometry(~Element,~band_value),~dense_cells_rows(~Element,~band_value),~dense_leaf_cell(~Element,~band_value),
        DenseCells{geometry,first,rows,prefix_channel,input_tree,input_length,shape,fill,status,workers,Buffer.Model{tree,length,certificate},limits,capacity,domain,target,
          %channel_length_offset(geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)) : BandShape.Leaf(geometry,first,rows,_) metadata,ready})
    case Model.Fork{+left,+right}:
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_targets,right_targets} = targets
      Shape.Both{dense_tree_channels(~Element,~band_value,geometry,prefix_channel,input,shape,fill,status,workers,left,limits,capacity,domain,left_ready,left_targets),dense_tree_channels(~Element,~band_value,geometry,prefix_channel,input,shape,fill,status,workers,right,limits,capacity,domain,right_ready,right_targets)}

def dense_requested_read_before(~Element: Data,request: BandRange.Request,+input: Tree.Tree<Element>,+output: Buffer.Model<Element>,+row_capacity: U32,+query: U32,+target: BandRange.Target(request,row_capacity,Tree.capacity(Element,Buffer.tree(~Element,output))),ready: Buffer.Ready<Element,output>,
  +inside: {U32.is_lt(query,Tree.capacity(Element,Buffer.tree(~Element,output))) == True{} : Bool},+before: {Nat.is_lt(U32.to_nat(query),U32.to_nat(BandRange.request_offset(request))) == True{} : Bool},+plane_positive: {Nat.is_lt(0n,U32.to_nat(BandCopy.count(dense_region(request,Tree.capacity(Element,Buffer.tree(~Element,output)))))) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,dense_requested_writes(~Element,request,input),Buffer.tree(~Element,output)),query) == Tree.read(Element,Buffer.tree(~Element,output),query) : Element}:
  BandRange.Request{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,+offset} = request
  Buffer.Ready{+depth_ok,room} = ready
  +region = dense_region(request,Tree.capacity(Element,Buffer.tree(~Element,output)))
  BandCopy.read_outside(~Element,U32.to_nat(channels),0n,region,dense_source_stride(request),dense_source_start(request),input,Buffer.tree(~Element,output),query,Buffer.depth(~Element,output),Buffer.balanced(~Element,output),depth_ok,{==},dense_target_valid(request,row_capacity,Tree.capacity(Element,Buffer.tree(~Element,output)),target,0n,plane_positive),inside,Order.reflexive(U32.to_nat(channels)),planes_before(U32.to_nat(channels),0n,region,query,before))

def dense_read_before(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+prefix_channel: U32,+input: Tree.Tree<Element>,+output: Buffer.Model<Element>,+row_capacity: U32,+position: Nat,
  +target: BandRange.Target(BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)},row_capacity,Tree.capacity(Element,Buffer.tree(~Element,output))),ready: Buffer.Ready<Element,output>,
  +before: {Nat.is_lt(position,U32.to_nat(BandConcat.channel_offset(geometry,prefix_channel,rows))) == True{} : Bool},
  +plane_positive: {Nat.is_lt(0n,U32.to_nat(BandCopy.count(dense_region(BandRange.Request{geometry,first,rows,0},Tree.capacity(Element,Buffer.tree(~Element,output)))))) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,BandConcatExecution.dense_writes(~Element,geometry,first,rows,prefix_channel,input),Buffer.tree(~Element,output)),U32.from_nat(position)) == Tree.read(Element,Buffer.tree(~Element,output),U32.from_nat(position)) : Element}:
  Buffer.Ready{+depth_ok,room} = ready
  +request = {BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)} : BandRange.Request}
  +buffer_capacity = Tree.capacity(Element,Buffer.tree(~Element,output))
  +position_room = Bounds.nat_trans(position,U32.to_nat(BandRange.request_offset(request)),U32.to_nat(buffer_capacity),before,target_offset_room(request,row_capacity,buffer_capacity,target))
  +position_exact = Bounds.roundtrip(position,buffer_capacity,Buffer.capacity_bound(~Element,output,depth_ok),Bounds.strict_le(position,U32.to_nat(buffer_capacity),position_room))
  %Equal.sym(List<&2,Writes.Assignment<Element>>,BandConcatExecution.dense_writes(~Element,geometry,first,rows,prefix_channel,input),dense_requested_writes(~Element,request,input),dense_schedule(~Element,geometry,first,rows,prefix_channel,input)) : {Tree.read(Element,Writes.pure(Element,_,Buffer.tree(~Element,output)),U32.from_nat(position)) == Tree.read(Element,Buffer.tree(~Element,output),U32.from_nat(position)) : Element}
  dense_requested_read_before(~Element,request,input,output,row_capacity,U32.from_nat(position),target,ready,Bounds.native_bound(position,buffer_capacity,position_exact,position_room),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,position_exact) : {Nat.is_lt(_,U32.to_nat(BandRange.request_offset(request))) == True{} : Bool} before,plane_positive)

# A dense append leaves every cell before its channel offset unchanged.
def dense_prefix_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+prefix_channel: U32,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+output: Buffer.Model<Element>,+limits: Machine.Limits,+channel: Nat,+row: Nat,+column: Nat,
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},metadata: BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.length(~Element,output)),+ready: Buffer.Ready<Element,output>,
  +target: BandRange.Target(BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
  +channel_room: {Nat.is_le(Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(prefix_channel)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool},
  cell: BandValues.At(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.tree(~Element,output),Buffer.length(~Element,output),channel,row,column)) ->
  BandValues.At(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,
    Writes.pure(Element,BandConcatExecution.dense_writes(~Element,Bands.Geometry{images,source_channels,height,width},first,rows,prefix_channel,input),Buffer.tree(~Element,output)),Buffer.length(~Element,output),channel,row,column):
  Shape.Both{inside,equal} = cell
  +geometry = {Bands.Geometry{images,source_channels,height,width} : Bands.Geometry}
  +request = {BandRange.Request{geometry,first,rows,BandConcat.channel_offset(geometry,prefix_channel,rows)} : BandRange.Request}
  +position = BandValues.position(Bands.Geometry{images,output_channels,height,width},rows,channel,row,column)
  +pixel = U32.from_nat(BandValues.pixel(Bands.Geometry{images,output_channels,height,width},channel,Nat.add(U32.to_nat(first),row),column))
  +plane = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +index = Nat.add(Nat.mul(row,U32.to_nat(width)),column)
  +plane_positive = Order.le_lt_trans(0n,index,plane,Order.zero_le(index),RowMajor.radix(row,U32.to_nat(rows),column,U32.to_nat(width),row_inside,column_inside))
  Shape.Both{inside,
    Equal.trans(Element,Tree.read(Element,Writes.pure(Element,BandConcatExecution.dense_writes(~Element,geometry,first,rows,prefix_channel,input),Buffer.tree(~Element,output)),U32.from_nat(position)),Tree.read(Element,Buffer.tree(~Element,output),U32.from_nat(position)),original(logical,pixel),
      dense_read_before(~Element,geometry,first,rows,prefix_channel,input,output,Machine.capacity(limits),position,target,ready,
        prefix_cell_before(~Element,images,source_channels,output_channels,height,width,prefix_channel,first,rows,output,channel,row,column,width_positive,metadata,ready,channel_room,channel_inside,row_inside,column_inside),
        %Equal.sym(Nat,U32.to_nat(BandCopy.count(dense_region(request,Tree.capacity(Element,Buffer.tree(~Element,output))))),plane,dense_plane_exact(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output)),target)) : {Nat.is_lt(0n,_) == True{} : Bool} plane_positive),
      equal)}

# What a dense append supplies to every cell below the cursor.
type DensePrefix<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  DensePrefix{logical: Logical,images: U32,source_channels: U32,output_channels: U32,height: U32,width: U32,prefix_channel: U32,first: U32,rows: U32,input: Tree.Tree<Element>,output: Buffer.Model<Element>,limits: Machine.Limits,width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},metadata: BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.length(~Element,output)),ready: Buffer.Ready<Element,output>,
    target: BandRange.Target(BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},prefix_channel,rows)},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
    values: BandValues.Channels(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.tree(~Element,output),Buffer.length(~Element,output),U32.to_nat(prefix_channel)),channel_room: {Nat.is_le(Nat.add(U32.to_nat(prefix_channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}}

def dense_prefix_geometry(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: DensePrefix<Element,Logical,original>) -> Bands.Geometry:
  DensePrefix{logical,images,source_channels,output_channels,height,width,prefix_channel,first,rows,input,output,limits,width_positive,metadata,ready,target,values,channel_room} = evidence
  Bands.Geometry{images,prefix_channel,height,width}

def dense_prefix_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: DensePrefix<Element,Logical,original>) -> U32:
  DensePrefix{logical,images,source_channels,output_channels,height,width,prefix_channel,first,rows,input,output,limits,width_positive,metadata,ready,target,values,channel_room} = evidence
  rows

def DensePrefixCell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: DensePrefix<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  DensePrefix{+logical,+images,+source_channels,+output_channels,+height,+width,+prefix_channel,+first,+rows,+input,+output,+limits,+width_positive,+metadata,+ready,+target,+values,+channel_room} = evidence
  BandValues.At(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,
    Writes.pure(Element,BandConcatExecution.dense_writes(~Element,Bands.Geometry{images,source_channels,height,width},first,rows,prefix_channel,input),Buffer.tree(~Element,output)),
    Buffer.length(~Element,output),channel,row,column)

def dense_prefix_leaf_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: DensePrefix<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(dense_prefix_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(dense_prefix_rows(~Element,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(dense_prefix_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool}) ->
  DensePrefixCell(~Element,~Logical,~original,evidence,channel,row,column):
  DensePrefix{+logical,+images,+source_channels,+output_channels,+height,+width,+prefix_channel,+first,+rows,+input,+output,+limits,+width_positive,+metadata,+ready,+target,+values,+channel_room} = evidence
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  dense_prefix_cell(~Element,~Logical,~original,logical,images,source_channels,output_channels,height,width,prefix_channel,first,rows,input,output,limits,channel,row,column,width_positive,metadata,ready,target,channel_room,channel_inside,row_inside,column_inside,
    BandValues.channels_cell(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.tree(~Element,output),Buffer.length(~Element,output),U32.to_nat(prefix_channel),channel,row,column,values,channel_inside,row_inside,column_inside))

def dense_tree_prefix(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+input: Buffer.Content<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,output: Model.Model<Element>,+limits: Machine.Limits,
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},metadata: BandShape.Contents(~Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,output)),ready: Model.Ready(~Element,output),
  values: PrefixValues(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},channel,Model.content(~Element,output)),targets: BandConcatExecution.RangeTargets(~Element,Bands.Geometry{images,source_channels,height,width},channel,limits,output),
  +channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}) ->
  PrefixValues(~Element,~Logical,~original,logical,Bands.Geometry{images,output_channels,height,width},channel,BandConcatExecution.tree_model(~Element,Bands.Geometry{images,source_channels,height,width},channel,BandConcatExecution.DenseContent{input,shape,fill,status,workers},Model.content(~Element,output))):
  Buffer.Content{+input_tree,+input_length} = input
  match output:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      Shape.Both{target,target_metadata} = targets
      BandValues.cells(~DensePrefix<Element,Logical,original>,~DensePrefixCell(~Element,~Logical,~original),~dense_prefix_geometry(~Element,~Logical,~original),~dense_prefix_rows(~Element,~Logical,~original),~dense_prefix_leaf_cell(~Element,~Logical,~original),
        DensePrefix{logical,images,source_channels,output_channels,height,width,channel,first,rows,input_tree,Buffer.Model{tree,length,certificate},limits,width_positive,metadata,ready,target,values,channel_room})
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_values,right_values} = values
      Shape.Both{left_targets,right_targets} = targets
      Shape.Both{dense_tree_prefix(~Element,~Logical,~original,logical,images,source_channels,output_channels,height,width,channel,input,shape,fill,status,workers,left,limits,width_positive,left_metadata,left_ready,left_values,left_targets,channel_room),
        dense_tree_prefix(~Element,~Logical,~original,logical,images,source_channels,output_channels,height,width,channel,input,shape,fill,status,workers,right,limits,width_positive,right_metadata,right_ready,right_values,right_targets,channel_room)}

# Reification retains the existing tree and length and reconstructs only an
# erased certificate from ContentReady. It claims no new native owner.
type ContentModel<-Element: Data,-content: Model.Content<Element>> is Data:
  ContentModel{model: Model.Model<Element>,equal: {Model.content(~Element,model) == content : Model.Content<Element>}}

def content_fork(~Element: Data,+left: Model.Content<Element>,+right: Model.Content<Element>,left_model: ContentModel<Element,left>,right_model: ContentModel<Element,right>) -> ContentModel<Element,Model.Joined{left,right}>:
  ContentModel{+left_model,+left_equal} = left_model
  ContentModel{+right_model,+right_equal} = right_model
  ContentModel{Model.Fork{left_model,right_model},
    %Equal.sym(Model.Content<Element>,Model.content(~Element,left_model),left,left_equal) : {Model.Joined{_,Model.content(~Element,right_model)} == Model.Joined{left,right} : Model.Content<Element>}
    %Equal.sym(Model.Content<Element>,Model.content(~Element,right_model),right,right_equal) : {Model.Joined{left,_} == Model.Joined{left,right} : Model.Content<Element>} {==}}

def content_model(~Element: Data,content: Model.Content<Element>,ready: Model.ContentReady(~Element,content)) -> ContentModel<Element,content>:
  match content:
    case Model.RowBand{Buffer.Content{+tree,+length},+first,+rows}:
      Buffer.ContentReady{+depth,+balanced,depth_ok,room} = ready
      ContentModel{Model.Leaf{Buffer.Model{tree,length,Certificate.Certificate{depth,
        Equal.sym(Array<Element>,Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,Tree.pack(Element,tree)))),Tree.pack(Element,tree),Certificate.restored(Element,depth,Tree.pack(Element,tree),Witness.from_shape(Element,depth,tree,balanced)))}},first,rows},{==}}
    case Model.Joined{+left,+right}:
      Shape.Both{left_ready,right_ready} = ready
      content_fork(~Element,left,right,content_model(~Element,left,left_ready),content_model(~Element,right,right_ready))

def well_formed_ready(~Element: Data,+geometry: Bands.Geometry,+content: Model.Content<Element>,evidence: BandShape.WellFormed<Element,geometry,content>) -> Model.ContentReady(~Element,content):
  BandShape.WellFormed{metadata,ready,coverage} = evidence
  ready

def StoredValues(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+input: BandConcatExecution.TensorContent<Element>,+root: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,output: Model.Content<Element>) -> Data:
  Shape.Both<PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},channel,output),
    BandConcatExecution.TreeChannelValues(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),input,Bands.Geometry{images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),height,width},channel,output)>

def selected_banded_values(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,same: Bool,+input: BandConcatExecution.TensorContent<Element>,+root: BandConcatExecution.PartContents<Element>,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+source: Model.Model<Element>,+output: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,+limits: Machine.Limits,+capacity: U32,
  +source_domain: BandGather.Domain<Bands.Geometry{images,source_channels,height,width},limits,capacity>,+source_metadata: BandShape.Contents(~Element,Bands.Geometry{images,source_channels,height,width},Model.content(~Element,source)),+source_ready: Model.Ready(~Element,source),+source_well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,source_channels,height,width},Model.content(~Element,source)>,
  +metadata: BandShape.Contents(~Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,output)),+ready: Model.Ready(~Element,output),
  +source_values: BandValues.Values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),input,Bands.Geometry{images,source_channels,height,width},Model.content(~Element,source)),
  +previous: PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},channel,Model.content(~Element,output)),+targets: BandConcatExecution.RangeTargets(~Element,Bands.Geometry{images,source_channels,height,width},channel,limits,output),
  +source_count: {BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) == source_channels : U32},+channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool},
  +compared: {Bands.same_plan(Model.layout(~Element,source),Model.layout(~Element,output)) == same : Bool}) ->
  StoredValues(~Element,~band_value,input,root,images,output_channels,height,width,channel,BandConcatExecution.selected_model(~Element,same,Bands.Geometry{images,source_channels,height,width},channel,Model.content(~Element,source),Model.content(~Element,output),shape,fill,status,workers)):
  match same:
    case True{}:
      %Equal.sym(U32,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),source_channels,source_count) : Shape.Both<PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},channel,BandConcatExecution.aligned_model(~Element,Bands.Geometry{images,source_channels,height,width},channel,Model.content(~Element,source),Model.content(~Element,output))),
        BandConcatExecution.TreeChannelValues(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),input,Bands.Geometry{images,_,height,width},channel,BandConcatExecution.aligned_model(~Element,Bands.Geometry{images,source_channels,height,width},channel,Model.content(~Element,source),Model.content(~Element,output)))>
      Shape.Both{aligned_prefix(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},images,source_channels,output_channels,height,width,channel,source,output,limits,source_width(Bands.Geometry{images,source_channels,height,width},limits,capacity,source_domain),source_metadata,metadata,source_ready,ready,previous,targets,channel_room,compared),
        aligned_channels(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),input,Bands.Geometry{images,source_channels,height,width},channel,source,output,limits,source_width(Bands.Geometry{images,source_channels,height,width},limits,capacity,source_domain),source_metadata,source_ready,ready,source_values,targets,compared)}
    case False{}:
      %Equal.sym(U32,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),source_channels,source_count) : Shape.Both<PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},channel,BandConcatExecution.tree_model(~Element,Bands.Geometry{images,source_channels,height,width},channel,BandConcatExecution.BandedContent{Model.content(~Element,source),Bands.Geometry{images,source_channels,height,width},shape,fill,status,workers},Model.content(~Element,output))),
        BandConcatExecution.TreeChannelValues(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),input,Bands.Geometry{images,_,height,width},channel,BandConcatExecution.tree_model(~Element,Bands.Geometry{images,source_channels,height,width},channel,BandConcatExecution.BandedContent{Model.content(~Element,source),Bands.Geometry{images,source_channels,height,width},shape,fill,status,workers},Model.content(~Element,output)))>
      Shape.Both{tree_prefix(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},images,source_channels,output_channels,height,width,channel,source,output,shape,fill,status,workers,limits,capacity,source_domain,source_width(Bands.Geometry{images,source_channels,height,width},limits,capacity,source_domain),source_well_formed,metadata,ready,previous,targets,channel_room),
        BandConcatExecution.tree_values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),input,Bands.Geometry{images,source_channels,height,width},Bands.Geometry{images,source_channels,height,width},channel,source,output,shape,fill,status,workers,limits,capacity,source_domain,source_well_formed,source_values,ready,targets)}

def canonical_banded_values(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+bands: Model.Content<Element>,+original_geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,+root: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+output: Model.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  source_well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,BandConcat.channels(shape),height,width},bands>,source_model: ContentModel<Element,bands>,
  output_well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,output)>,
  +source_values: BandValues.Values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),BandConcatExecution.BandedContent{bands,original_geometry,shape,fill,status,workers},Bands.Geometry{images,BandConcat.channels(shape),height,width},bands),
  +previous: PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},channel,Model.content(~Element,output)),
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+positive: {Nat.is_lt(0n,U32.to_nat(BandConcat.channels(shape))) == True{} : Bool},+channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(BandConcat.channels(shape))),U32.to_nat(output_channels)) == True{} : Bool}) ->
  StoredValues(~Element,~band_value,BandConcatExecution.BandedContent{bands,original_geometry,shape,fill,status,workers},root,images,output_channels,height,width,channel,BandConcatExecution.storage_model(~Element,Bands.Geometry{images,BandConcat.channels(shape),height,width},channel,BandConcatExecution.BandedContent{bands,original_geometry,shape,fill,status,workers},Model.content(~Element,output))):
  BandShape.WellFormed{+source_metadata,+source_ready,+coverage} = source_well_formed
  ContentModel{+source,+source_equal} = source_model
  BandShape.WellFormed{+metadata,+ready,output_coverage} = output_well_formed
  +geometry = {Bands.Geometry{images,BandConcat.channels(shape),height,width} : Bands.Geometry}
  +input = {BandConcatExecution.BandedContent{bands,original_geometry,shape,fill,status,workers} : BandConcatExecution.TensorContent<Element>}
  +source_domain = source_domain(images,BandConcat.channels(shape),output_channels,height,width,channel,limits,capacity,domain,positive,channel_room)
  +targets = BandConcatCaller.range_targets(~Element,images,BandConcat.channels(shape),output_channels,height,width,channel,limits,capacity,output,domain,metadata,Model.ready(~Element,output,ready),positive,channel_room)
  %source_equal : StoredValues(~Element,~band_value,input,root,images,output_channels,height,width,channel,BandConcatExecution.storage_model(~Element,geometry,channel,BandConcatExecution.BandedContent{_,original_geometry,shape,fill,status,workers},Model.content(~Element,output)))
  %Model.layout_matches_content(~Element,source) : StoredValues(~Element,~band_value,input,root,images,output_channels,height,width,channel,BandConcatExecution.selected_model(~Element,Bands.same_plan(_,Model.content_layout(~Element,Model.content(~Element,output))),geometry,channel,Model.content(~Element,source),Model.content(~Element,output),shape,fill,status,workers))
  %Model.layout_matches_content(~Element,output) : StoredValues(~Element,~band_value,input,root,images,output_channels,height,width,channel,BandConcatExecution.selected_model(~Element,Bands.same_plan(Model.layout(~Element,source),_),geometry,channel,Model.content(~Element,source),Model.content(~Element,output),shape,fill,status,workers))
  selected_banded_values(~Element,~band_value,Bands.same_plan(Model.layout(~Element,source),Model.layout(~Element,output)),input,root,images,BandConcat.channels(shape),output_channels,height,width,channel,source,output,shape,fill,status,workers,limits,capacity,source_domain,
    %Equal.sym(Model.Content<Element>,Model.content(~Element,source),bands,source_equal) : BandShape.Contents(~Element,geometry,_) source_metadata,
    Model.ready(~Element,source,%Equal.sym(Model.Content<Element>,Model.content(~Element,source),bands,source_equal) : Model.ContentReady(~Element,_) source_ready),
    %Equal.sym(Model.Content<Element>,Model.content(~Element,source),bands,source_equal) : BandShape.WellFormed<Element,geometry,_> source_well_formed,metadata,Model.ready(~Element,output,ready),
    %Equal.sym(Model.Content<Element>,Model.content(~Element,source),bands,source_equal) : BandValues.Values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),input,geometry,_) source_values,previous,targets,{==},channel_room,{==})

def storage_values(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,input: BandConcatExecution.TensorContent<Element>,+root: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+output: Model.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  contract: SourceContract(~Element,~band_value,images,height,width,input),output_well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,output)>,
  +previous: PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},channel,Model.content(~Element,output)),
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+positive: {Nat.is_lt(0n,U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))) == True{} : Bool},+channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))),U32.to_nat(output_channels)) == True{} : Bool}) ->
  StoredValues(~Element,~band_value,input,root,images,output_channels,height,width,channel,BandConcatExecution.storage_model(~Element,Bands.Geometry{images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),height,width},channel,input,Model.content(~Element,output))):
  match input:
    case BandConcatExecution.DenseContent{+buffer,+shape,+fill,+status,+workers}:
      BandShape.WellFormed{+metadata,+ready,coverage} = output_well_formed
      +source_domain = source_domain(images,BandConcat.channels(shape),output_channels,height,width,channel,limits,capacity,domain,positive,channel_room)
      +targets = BandConcatCaller.range_targets(~Element,images,BandConcat.channels(shape),output_channels,height,width,channel,limits,capacity,output,domain,metadata,Model.ready(~Element,output,ready),positive,channel_room)
      Shape.Both{dense_tree_prefix(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},images,BandConcat.channels(shape),output_channels,height,width,channel,buffer,shape,fill,status,workers,output,limits,source_width(Bands.Geometry{images,BandConcat.channels(shape),height,width},limits,capacity,source_domain),metadata,Model.ready(~Element,output,ready),previous,targets,channel_room),
        dense_tree_channels(~Element,~band_value,Bands.Geometry{images,BandConcat.channels(shape),height,width},channel,buffer,shape,fill,status,workers,output,limits,capacity,source_domain,Model.ready(~Element,output,ready),targets)}
    case BandConcatExecution.BandedContent{+bands,+geometry,+shape,+fill,+status,+workers}:
      Shape.Both{+source_well_formed,+source_values} = contract
      canonical_banded_values(~Element,~band_value,bands,geometry,shape,fill,status,workers,root,images,output_channels,height,width,channel,output,limits,capacity,source_well_formed,content_model(~Element,bands,well_formed_ready(~Element,Bands.Geometry{images,BandConcat.channels(shape),height,width},bands,source_well_formed)),output_well_formed,source_values,previous,domain,positive,channel_room)

def finish_modeled_prefix(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+input: BandConcatExecution.TensorContent<Element>,+root: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+next_channel: U32,+output: Model.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  values: StoredValues(~Element,~band_value,input,root,images,output_channels,height,width,channel,Model.content(~Element,output)),
  well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,output)>,
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))),U32.to_nat(output_channels)) == True{} : Bool},
  +total_exact: {U32.to_nat(output_channels) == BandConcatExecution.total_channels(~Element,root) : Nat},+step_exact: {U32.to_nat(next_channel) == Nat.add(U32.to_nat(channel),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))) : Nat},
  +source_position: ConcatOrderedSources.SourcePosition<Element,Unit,channel_extent(~Element),Unit{},input,root,U32.to_nat(channel)>) ->
  PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},next_channel,Model.content(~Element,output)):
  Shape.Both{previous,added} = values
  BandShape.WellFormed{+metadata,+ready,coverage} = well_formed
  +source_channels = BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input))
  extend_channels(~Element,~tensor_original(~Element,~band_value),input,root,images,source_channels,output_channels,height,width,channel,next_channel,output,limits,capacity,domain,metadata,Model.ready(~Element,output,ready),previous,added,{==},total_exact,channel_room,step_exact,source_position)

def finish_content_prefix(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+input: BandConcatExecution.TensorContent<Element>,+root: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+next_channel: U32,+output: Model.Content<Element>,+limits: Machine.Limits,+capacity: U32,
  modeled: ContentModel<Element,output>,+values: StoredValues(~Element,~band_value,input,root,images,output_channels,height,width,channel,output),+well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},output>,
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))),U32.to_nat(output_channels)) == True{} : Bool},
  +total_exact: {U32.to_nat(output_channels) == BandConcatExecution.total_channels(~Element,root) : Nat},+step_exact: {U32.to_nat(next_channel) == Nat.add(U32.to_nat(channel),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))) : Nat},
  +source_position: ConcatOrderedSources.SourcePosition<Element,Unit,channel_extent(~Element),Unit{},input,root,U32.to_nat(channel)>) ->
  PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},next_channel,output):
  ContentModel{+model,+equal} = modeled
  %equal : PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},next_channel,_)
  finish_modeled_prefix(~Element,~band_value,input,root,images,output_channels,height,width,channel,next_channel,model,limits,capacity,
    %Equal.sym(Model.Content<Element>,Model.content(~Element,model),output,equal) : StoredValues(~Element,~band_value,input,root,images,output_channels,height,width,channel,_) values,
    %Equal.sym(Model.Content<Element>,Model.content(~Element,model),output,equal) : BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},_> well_formed,domain,channel_room,total_exact,step_exact,source_position)

def modeled_step_prefix(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+input: BandConcatExecution.TensorContent<Element>,+root: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+output: Model.Content<Element>,+limits: Machine.Limits,+capacity: U32,
  modeled: ContentModel<Element,output>,+contract: SourceContract(~Element,~band_value,images,height,width,input),+well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},output>,
  +previous: PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},channel,output),
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+positive: {Nat.is_lt(0n,U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))) == True{} : Bool},+channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))),U32.to_nat(output_channels)) == True{} : Bool},
  +total_exact: {U32.to_nat(output_channels) == BandConcatExecution.total_channels(~Element,root) : Nat},+step_exact: {U32.to_nat((channel + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32)) == Nat.add(U32.to_nat(channel),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))) : Nat},
  +source_position: ConcatOrderedSources.SourcePosition<Element,Unit,channel_extent(~Element),Unit{},input,root,U32.to_nat(channel)>) ->
  PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},(channel + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32),BandConcatExecution.storage_model(~Element,Bands.Geometry{images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),height,width},channel,input,output)):
  ContentModel{+model,+equal} = modeled
  +geometry = {Bands.Geometry{images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),height,width} : Bands.Geometry}
  +new_output = BandConcatExecution.storage_model(~Element,geometry,channel,input,Model.content(~Element,model))
  +model_well_formed = {
    %Equal.sym(Model.Content<Element>,Model.content(~Element,model),output,equal) : BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},_> well_formed
    : BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,model)>}
  +new_well_formed = BandConcatExecution.storage_well_formed(~Element,geometry,Bands.Geometry{images,output_channels,height,width},channel,input,Model.content(~Element,model),model_well_formed)
  %equal : PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},(channel + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32),BandConcatExecution.storage_model(~Element,geometry,channel,input,_))
  finish_content_prefix(~Element,~band_value,input,root,images,output_channels,height,width,channel,(channel + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32),new_output,limits,capacity,
    content_model(~Element,new_output,well_formed_ready(~Element,Bands.Geometry{images,output_channels,height,width},new_output,new_well_formed)),
    storage_values(~Element,~band_value,input,root,images,output_channels,height,width,channel,model,limits,capacity,contract,model_well_formed,
      %Equal.sym(Model.Content<Element>,Model.content(~Element,model),output,equal) : PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},channel,_) previous,domain,positive,channel_room),
    new_well_formed,domain,channel_room,total_exact,step_exact,source_position)

type RoomFields<-channel: U32,-source_channels: U32,-output_channels: U32> is Data:
  RoomFields{room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool},
    exact: {U32.to_nat((channel + source_channels : U32)) == Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)) : Nat}}

def PartsRooms(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+output_channels: U32,+channel: U32) -> Data:
  match parts:
    case BandConcatExecution.SingleContent{input}: RoomFields<channel,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),output_channels>
    case BandConcatExecution.ConsContent{input,rest}: Shape.Both<RoomFields<channel,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),output_channels>,PartsRooms(~Element,rest,output_channels,(channel + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32))>

def operational_head(+previous: U32,+source_channels: U32,+next: U32,+output_channels: U32,+channel: U32,
  fields: Shape.Both<ChannelStep<previous,source_channels,next>,{Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}>,+cursor: {channel == previous : U32}) -> RoomFields<channel,source_channels,output_channels>:
  Shape.Both{step,room} = fields
  ChannelStep{next_cursor,exact} = step
  RoomFields{
    %Equal.sym(U32,channel,previous,cursor) : {Nat.is_le(Nat.add(U32.to_nat(_),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool} room,
    %Equal.sym(U32,channel,previous,cursor) : {U32.to_nat((_ + source_channels : U32)) == Nat.add(U32.to_nat(_),U32.to_nat(source_channels)) : Nat} exact}

def operational_next(+previous: U32,+source_channels: U32,+next: U32,+output_channels: U32,+channel: U32,
  fields: Shape.Both<ChannelStep<previous,source_channels,next>,{Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}>,+cursor: {channel == previous : U32}) -> {(channel + source_channels : U32) == next : U32}:
  Shape.Both{step,room} = fields
  ChannelStep{+next_cursor,exact} = step
  Equal.trans(U32,(channel + source_channels : U32),(previous + source_channels : U32),next,Equal.cong(U32,U32,value => (value + source_channels : U32),channel,previous,cursor),Equal.sym(U32,next,(previous + source_channels : U32),next_cursor))

def operational_rooms(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+channel: U32,
  fields: ChannelRooms(~Element,parts,first,axis,summary),+cursor: {channel == BandConcatCaller.survey_total(summary) : U32}) ->
  PartsRooms(~Element,parts,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,parts,first,axis,summary)),channel):
  match parts:
    case BandConcatExecution.SingleContent{+input}:
      operational_head(BandConcatCaller.survey_total(summary),BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),BandConcatCaller.survey_total(BandConcatCaller.step(~Element,first,axis,input,summary)),BandConcatCaller.survey_total(BandConcatCaller.step(~Element,first,axis,input,summary)),channel,fields,cursor)
    case BandConcatExecution.ConsContent{+input,+rest}:
      Shape.Both{+head,+tail} = fields
      +next = BandConcatCaller.step(~Element,first,axis,input,summary)
      +output_channels = BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,rest,first,axis,next))
      Shape.Both{operational_head(BandConcatCaller.survey_total(summary),BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),BandConcatCaller.survey_total(next),output_channels,channel,head,cursor),
        operational_rooms(~Element,rest,first,axis,next,(channel + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32),tail,operational_next(BandConcatCaller.survey_total(summary),BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),BandConcatCaller.survey_total(next),output_channels,channel,head,cursor))}

def word_zero(+value: U32,+zero: {U32.to_nat(value) == 0n : Nat}) -> {value == 0 : U32}:
  Equal.trans(U32,value,U32.from_nat(U32.to_nat(value)),0,Equal.sym(U32,U32.from_nat(U32.to_nat(value)),value,Arithmetic.native_roundtrip(value)),Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(value),0n,zero))

def step_prefix_cases(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,channels: Nat,+input: BandConcatExecution.TensorContent<Element>,+root: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+output: Model.Content<Element>,+limits: Machine.Limits,+capacity: U32,
  +count: {U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input))) == channels : Nat},+contract: SourceContract(~Element,~band_value,images,height,width,input),+well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},output>,
  +previous: PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},channel,output),
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))),U32.to_nat(output_channels)) == True{} : Bool},+total_exact: {U32.to_nat(output_channels) == BandConcatExecution.total_channels(~Element,root) : Nat},
  +step_exact: {U32.to_nat((channel + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32)) == Nat.add(U32.to_nat(channel),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))) : Nat},
  +source_position: ConcatOrderedSources.SourcePosition<Element,Unit,channel_extent(~Element),Unit{},input,root,U32.to_nat(channel)>) ->
  PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},(channel + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32),BandConcatExecution.storage_model(~Element,Bands.Geometry{images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),height,width},channel,input,output)):
  match channels:
    case 0n:
      %Equal.sym(U32,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),0,word_zero(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),count)) : PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},(channel + _ : U32),BandConcatExecution.storage_model(~Element,Bands.Geometry{images,_,height,width},channel,input,output))
      %Equal.sym(U32,(channel + 0 : U32),channel,Arithmetic.add_zero(channel)) : PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},_,BandConcatExecution.storage_model(~Element,Bands.Geometry{images,0,height,width},channel,input,output))
      %Equal.sym(Model.Content<Element>,BandConcatExecution.storage_model(~Element,Bands.Geometry{images,0,height,width},channel,input,output),output,BandConcatCaller.empty_storage(~Element,images,height,width,channel,input,output)) : PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},channel,_) previous
    case 1n+ +rest:
      modeled_step_prefix(~Element,~band_value,input,root,images,output_channels,height,width,channel,output,limits,capacity,content_model(~Element,output,well_formed_ready(~Element,Bands.Geometry{images,output_channels,height,width},output,well_formed)),contract,well_formed,previous,domain,
        %Equal.sym(Nat,U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input))),1n+rest,count) : {Nat.is_lt(0n,_) == True{} : Bool} Order.lt_succ(0n,rest,Order.zero_le(rest)),channel_room,total_exact,step_exact,source_position)

def part_prefix_values(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+input: BandConcatExecution.TensorContent<Element>,+root: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,state: BandConcatExecution.ContentState(~Element),+limits: Machine.Limits,+capacity: U32,
  +contract: SourceContract(~Element,~band_value,images,height,width,input),+well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},BandConcatExecution.state_output(~Element,state)>,
  +previous: PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},BandConcatExecution.state_channel(~Element,state),BandConcatExecution.state_output(~Element,state)),
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,fields: RoomFields<BandConcatExecution.state_channel(~Element,state),BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),output_channels>,+total_exact: {U32.to_nat(output_channels) == BandConcatExecution.total_channels(~Element,root) : Nat},
  +source_position: ConcatOrderedSources.SourcePosition<Element,Unit,channel_extent(~Element),Unit{},input,root,U32.to_nat(BandConcatExecution.state_channel(~Element,state))>) ->
  PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},BandConcatExecution.state_channel(~Element,BandConcatExecution.part_model(~Element,Bands.Geometry{images,output_channels,height,width},input,state)),BandConcatExecution.state_output(~Element,BandConcatExecution.part_model(~Element,Bands.Geometry{images,output_channels,height,width},input,state))):
  (+output,+channel) = state
  RoomFields{+room,+exact} = fields
  step_prefix_cases(~Element,~band_value,U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input))),input,root,images,output_channels,height,width,channel,output,limits,capacity,{==},contract,well_formed,previous,domain,room,total_exact,exact,source_position)

def next_parts_position(~Element: Data,+input: BandConcatExecution.TensorContent<Element>,+rest: BandConcatExecution.PartContents<Element>,-root: BandConcatExecution.PartContents<Element>,+prefix: U32,+next: U32,+output_channels: U32,
  fields: RoomFields<prefix,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),output_channels>,
  +cursor: {next == (prefix + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32) : U32},
  position: ConcatOrderedSources.PartsPosition<Element,Unit,channel_extent(~Element),Unit{},BandConcatExecution.ConsContent{input,rest},root,U32.to_nat(prefix)>) ->
  ConcatOrderedSources.PartsPosition<Element,Unit,channel_extent(~Element),Unit{},rest,root,U32.to_nat(next)>:
  RoomFields{room,+exact} = fields
  %Equal.sym(U32,next,(prefix + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32),cursor) : ConcatOrderedSources.PartsPosition<Element,Unit,channel_extent(~Element),Unit{},rest,root,U32.to_nat(_)>
  %Equal.sym(Nat,U32.to_nat((prefix + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32)),Nat.add(U32.to_nat(prefix),U32.to_nat(BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)))),exact) : ConcatOrderedSources.PartsPosition<Element,Unit,channel_extent(~Element),Unit{},rest,root,_>
  ConcatOrderedSources.tail_position(~Element,~Unit,~channel_extent(~Element),Unit{},input,rest,root,U32.to_nat(prefix),position)

def parts_prefix_values(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,parts: BandConcatExecution.PartContents<Element>,+root: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+state: BandConcatExecution.ContentState(~Element),+limits: Machine.Limits,+capacity: U32,
  contracts: SourceContracts(~Element,~band_value,images,height,width,parts),+well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},BandConcatExecution.state_output(~Element,state)>,
  +previous: PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},BandConcatExecution.state_channel(~Element,state),BandConcatExecution.state_output(~Element,state)),
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,rooms: PartsRooms(~Element,parts,output_channels,BandConcatExecution.state_channel(~Element,state)),+total_exact: {U32.to_nat(output_channels) == BandConcatExecution.total_channels(~Element,root) : Nat},
  +position: ConcatOrderedSources.PartsPosition<Element,Unit,channel_extent(~Element),Unit{},parts,root,U32.to_nat(BandConcatExecution.state_channel(~Element,state))>) ->
  PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{root,height,width},Bands.Geometry{images,output_channels,height,width},BandConcatExecution.state_channel(~Element,BandConcatExecution.parts_model(~Element,parts,Bands.Geometry{images,output_channels,height,width},state)),BandConcatExecution.state_output(~Element,BandConcatExecution.parts_model(~Element,parts,Bands.Geometry{images,output_channels,height,width},state))):
  match parts:
    case BandConcatExecution.SingleContent{+input}:
      part_prefix_values(~Element,~band_value,input,root,images,output_channels,height,width,state,limits,capacity,contracts,well_formed,previous,domain,rooms,total_exact,
        ConcatOrderedSources.head_position(~Element,~Unit,~channel_extent(~Element),Unit{},BandConcatExecution.SingleContent{input},root,U32.to_nat(BandConcatExecution.state_channel(~Element,state)),position))
    case BandConcatExecution.ConsContent{+input,+rest}:
      Shape.Both{+head_contract,+tail_contracts} = contracts
      Shape.Both{+head_room,+tail_rooms} = rooms
      +geometry = {Bands.Geometry{images,output_channels,height,width} : Bands.Geometry}
      +next_state = BandConcatExecution.part_model(~Element,geometry,input,state)
      parts_prefix_values(~Element,~band_value,rest,root,images,output_channels,height,width,next_state,limits,capacity,tail_contracts,
        BandConcatExecution.part_well_formed(~Element,geometry,geometry,input,state,well_formed),
        part_prefix_values(~Element,~band_value,input,root,images,output_channels,height,width,state,limits,capacity,head_contract,well_formed,previous,domain,head_room,total_exact,
          ConcatOrderedSources.head_position(~Element,~Unit,~channel_extent(~Element),Unit{},BandConcatExecution.ConsContent{input,rest},root,U32.to_nat(BandConcatExecution.state_channel(~Element,state)),position)),domain,
        %Equal.sym(U32,BandConcatExecution.state_channel(~Element,next_state),(BandConcatExecution.state_channel(~Element,state) + BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)) : U32),part_model_cursor(~Element,geometry,input,state)) : PartsRooms(~Element,rest,output_channels,_) tail_rooms,
        total_exact,
        next_parts_position(~Element,input,rest,root,BandConcatExecution.state_channel(~Element,state),BandConcatExecution.state_channel(~Element,next_state),output_channels,head_room,part_model_cursor(~Element,geometry,input,state),position))

def zero_prefix(~Element: Data,~Logical: Data,~value: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,output: Model.Content<Element>) -> PrefixValues(~Element,~Logical,~value,logical,geometry,0,output):
  match output:
    case Model.RowBand{Buffer.Content{tree,length},first,rows}: Unit{}
    case Model.Joined{left,right}: Shape.Both{zero_prefix(~Element,~Logical,~value,logical,geometry,left),zero_prefix(~Element,~Logical,~value,logical,geometry,right)}

# When the cursor has reached the channel count, the prefix is the whole leaf.
def whole_prefix(~Element: Data,~Logical: Data,~value: Logical -> U32 -> Element,+logical: Logical,geometry: Bands.Geometry,output: Model.Content<Element>,prefix: PrefixValues(~Element,~Logical,~value,logical,geometry,BandRange.channel_count(BandRange.Request{geometry,0,0,0}),output)) -> BandValues.Values(~Element,~Logical,~value,logical,geometry,output):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  match output:
    case Model.RowBand{Buffer.Content{tree,length},first,rows}: prefix
    case Model.Joined{+left,+right}:
      Shape.Both{left_prefix,right_prefix} = prefix
      Shape.Both{whole_prefix(~Element,~Logical,~value,logical,Bands.Geometry{images,channels,height,width},left,left_prefix),whole_prefix(~Element,~Logical,~value,logical,Bands.Geometry{images,channels,height,width},right,right_prefix)}

def final_cursor(~Element: Data,+parts: BandConcatExecution.PartContents<Element>,+geometry: Bands.Geometry,+output: Model.Content<Element>,+count: U32,+total_exact: {U32.to_nat(count) == BandConcatExecution.total_channels(~Element,parts) : Nat}) ->
  {BandConcatExecution.state_channel(~Element,BandConcatExecution.parts_model(~Element,parts,geometry,(output,0))) == count : U32}:
  Equal.trans(U32,BandConcatExecution.state_channel(~Element,BandConcatExecution.parts_model(~Element,parts,geometry,(output,0))),BandConcatExecution.channel_cursor(~Element,parts,0),count,BandConcatExecution.parts_channel(~Element,parts,geometry,(output,0)),
    Equal.trans(U32,BandConcatExecution.channel_cursor(~Element,parts,0),U32.from_nat(BandConcatExecution.total_channels(~Element,parts)),count,BandConcatExecution.cursor_exact(~Element,parts,0),Equal.sym(U32,count,U32.from_nat(BandConcatExecution.total_channels(~Element,parts)),Encoded.encoding(count,BandConcatExecution.total_channels(~Element,parts),total_exact))))

def parts_values(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+parts: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+output: Model.Content<Element>,+limits: Machine.Limits,+capacity: U32,
  +contracts: SourceContracts(~Element,~band_value,images,height,width,parts),+well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,output_channels,height,width},output>,+domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,
  +rooms: PartsRooms(~Element,parts,output_channels,0),+total_exact: {U32.to_nat(output_channels) == BandConcatExecution.total_channels(~Element,parts) : Nat}) ->
  BandValues.Values(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{parts,height,width},Bands.Geometry{images,output_channels,height,width},BandConcatExecution.state_output(~Element,BandConcatExecution.parts_model(~Element,parts,Bands.Geometry{images,output_channels,height,width},(output,0)))):
  +geometry = {Bands.Geometry{images,output_channels,height,width} : Bands.Geometry}
  +state = {(output,0) : BandConcatExecution.ContentState(~Element)}
  +next_state = BandConcatExecution.parts_model(~Element,parts,geometry,state)
  whole_prefix(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{parts,height,width},geometry,BandConcatExecution.state_output(~Element,next_state),
    %final_cursor(~Element,parts,geometry,output,output_channels,total_exact) : PrefixValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{parts,height,width},geometry,_,BandConcatExecution.state_output(~Element,next_state))
    parts_prefix_values(~Element,~band_value,parts,parts,images,output_channels,height,width,state,limits,capacity,contracts,well_formed,zero_prefix(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{parts,height,width},geometry,output),domain,rooms,total_exact,ConcatOrderedSources.root_position(~Element,~Unit,~channel_extent(~Element),Unit{},parts)))

def total_projection(+value: U32,+previous: U32,+extent: Nat,+limit: U32,fields: BandConcatCaller.TotalFields<value,previous,extent,limit>) -> {U32.to_nat(value) == Nat.add(U32.to_nat(previous),extent) : Nat}:
  BandConcatCaller.TotalFields{exact,room} = fields
  exact

def surveyed_total(~Element: Data,+parts: BandConcatExecution.PartContents<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  +accepted: {BandConcatCaller.survey_valid(BandConcatCaller.survey_model(~Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))) == True{} : Bool}) ->
  {U32.to_nat(BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)))))) == BandConcatExecution.total_channels(~Element,parts) : Nat}:
  +first = {BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>}
  +summary = BandConcatCaller.initial_summary(first,U32.to_nat(BandConcatCaller.axis_of(rank)))
  +fields = BandConcatCaller.prefixes(~Element,parts,first,U32.to_nat(BandConcatCaller.axis_of(rank)),summary,limit,maximum,accepted)
  %BandConcatCaller.extent_channels(~Element,parts,rank,channels,height,width,summary,limit,fields) : {U32.to_nat(BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,parts,first,U32.to_nat(BandConcatCaller.axis_of(rank)),summary))) == _ : Nat}
  total_projection(BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,parts,first,U32.to_nat(BandConcatCaller.axis_of(rank)),summary)),0,BandConcatCaller.extent_total(~Element,parts,U32.to_nat(BandConcatCaller.axis_of(rank))),limit,BandConcatCaller.total_fields(~Element,parts,first,U32.to_nat(BandConcatCaller.axis_of(rank)),summary,limit,fields))

def surveyed_values_packed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+sources: BandConcatExecution.PartsOwnerModel<Element>,+target: Model.Model<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+images: U32,+height: U32,+width: U32,+workers: U32,+limits: Machine.Limits,+capacity: U32,+initial_cursor: {0 == BandConcatCaller.survey_total(summary) : U32},
  +contracts: SourceContracts(~Element,~band_value,images,height,width,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),
  +domain: BandGather.Domain<Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,axis,summary)),height,width},limits,capacity>,
  +well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,axis,summary)),height,width},Model.contents(~Element,Model.pack(~Element,target))>,
  fields: Shape.Both<ChannelRooms(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,axis,summary),
    {U32.to_nat(BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,axis,summary))) == BandConcatExecution.total_channels(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))) : Nat}>) ->
  BandValues.Values(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width},Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,axis,summary)),height,width},
    Model.contents(~Element,BandConcat.append_parts(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,axis,summary)),height,width},workers,(Model.pack(~Element,target),0)))):
  Shape.Both{+rooms,+total_exact} = fields
  +contents = {BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)) : BandConcatExecution.PartContents<Element>}
  +channels = BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,contents,first,axis,summary))
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +output = Model.contents(~Element,Model.pack(~Element,target))
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandConcat.append_parts(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),geometry,workers,(Model.pack(~Element,target),0))),BandConcatExecution.state_output(~Element,BandConcatExecution.parts_model(~Element,contents,geometry,(output,0))),
    BandConcatExecution.append_parts_packed(~Element,sources,geometry,workers,BandConcatExecution.BandStateModel{target,0})) : BandValues.Values(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{contents,height,width},geometry,_)
  parts_values(~Element,~band_value,contents,images,channels,height,width,output,limits,capacity,contracts,well_formed,domain,operational_rooms(~Element,contents,first,axis,summary,0,rooms,initial_cursor),total_exact)

def surveyed_values_observed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,-parts: Tensors.Parts<Element>,-output: Bands.Bands(~Element),+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+images: U32,+height: U32,+width: U32,+workers: U32,+limits: Machine.Limits,+capacity: U32,+initial_cursor: {0 == BandConcatCaller.survey_total(summary) : U32},
  parts_observation: BandConcatExecution.PartsObservation<Element,parts>,output_observation: Model.Observation<Element,output>,
  +accepted: {BandConcatCaller.survey_valid(BandConcatCaller.observed_summary(~Element,Layout.survey(~Element,parts,first,axis,summary))) == True{} : Bool},
  +contracts: SourceContracts(~Element,~band_value,images,height,width,BandConcatExecution.parts_contents(~Element,parts)),
  +domain: BandGather.Domain<Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},limits,capacity>,
  +well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},Model.contents(~Element,output)>,
  fields_rule: @+contents: BandConcatExecution.PartContents<Element> -> @+accepted: {BandConcatCaller.survey_valid(BandConcatCaller.survey_model(~Element,contents,first,axis,summary)) == True{} : Bool} ->
    Shape.Both<ChannelRooms(~Element,contents,first,axis,summary),{U32.to_nat(BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,contents,first,axis,summary))) == BandConcatExecution.total_channels(~Element,contents) : Nat}>) ->
  BandValues.Values(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,parts),height,width},Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},
    Model.contents(~Element,BandConcat.append_parts(~Element,parts,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},workers,(output,0)))):
  BandConcatExecution.PartsObservation{+sources,+parts_equal} = parts_observation
  Model.Observation{+target,+output_equal} = output_observation
  +contents = {BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)) : BandConcatExecution.PartContents<Element>}
  +model_accepted = {
    %BandConcatCaller.native_summary_packed(~Element,sources,first,axis,summary) : {BandConcatCaller.survey_valid(_) == True{} : Bool}
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,parts_equal) : {BandConcatCaller.survey_valid(BandConcatCaller.observed_summary(~Element,Layout.survey(~Element,_,first,axis,summary))) == True{} : Bool}
    accepted : {BandConcatCaller.survey_valid(BandConcatCaller.survey_model(~Element,contents,first,axis,summary)) == True{} : Bool}}
  %parts_equal : BandValues.Values(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,_),height,width},Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary)),height,width},Model.contents(~Element,BandConcat.append_parts(~Element,_,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary)),height,width},workers,(output,0))))
  %output_equal : BandValues.Values(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{contents,height,width},Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,contents,first,axis,summary)),height,width},Model.contents(~Element,BandConcat.append_parts(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,contents,first,axis,summary)),height,width},workers,(_,0))))
  surveyed_values_packed(~Element,~band_value,sources,target,first,axis,summary,images,height,width,workers,limits,capacity,initial_cursor,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,parts_equal) : SourceContracts(~Element,~band_value,images,height,width,BandConcatExecution.parts_contents(~Element,_)) contracts,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,parts_equal) : BandGather.Domain<Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary)),height,width},limits,capacity> domain,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,parts_equal) : BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary)),height,width},Model.contents(~Element,Model.pack(~Element,target))>
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,target),output,output_equal) : BandShape.WellFormed<Element,Bands.Geometry{images,BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)),height,width},Model.contents(~Element,_)> well_formed,
    fields_rule(contents,model_accepted))

def surveyed_parts_values(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,parts: Tensors.Parts<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,output: Bands.Bands(~Element),+workers: U32,+limits: Machine.Limits,+capacity: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  +accepted: {BandConcatCaller.survey_valid(BandConcatCaller.observed_summary(~Element,Layout.survey(~Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)))))) == True{} : Bool},
  +contracts: SourceContracts(~Element,~band_value,BandConcatCaller.images_of(rank),height,width,BandConcatExecution.parts_contents(~Element,parts)),
  +domain: BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},limits,capacity>,
  +well_formed: BandShape.WellFormed<Element,Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},Model.contents(~Element,output)>) ->
  BandValues.Values(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,parts),height,width},Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},Model.contents(~Element,BandConcat.append_parts(~Element,parts,Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},workers,(output,0)))):
  surveyed_values_observed(~Element,~band_value,parts,output,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))),BandConcatCaller.images_of(rank),height,width,workers,limits,capacity,{==},BandConcatExecution.observe_parts(~Element,parts),Model.observe(~Element,output),accepted,contracts,domain,well_formed,
    contents => model_accepted => Shape.Both{surveyed_rooms(~Element,contents,rank,channels,height,width,BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))),limit,maximum,model_accepted),surveyed_total(~Element,contents,rank,channels,height,width,limit,maximum,model_accepted)})

def TensorValues(~Element: Data,~Logical: Data,~value: Logical -> U32 -> Element,+logical: Logical,input: BandConcatExecution.TensorContent<Element>) -> Data:
  # Dense fallback has a separate contract; this selected band branch
  # cannot be inhabited by a DenseContent result.
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}: Empty
    case BandConcatExecution.BandedContent{bands,geometry,shape,fill,status,workers}: BandValues.Values(~Element,~Logical,~value,logical,geometry,bands)

def logical_owner(~Element: Data,geometry: Bands.Geometry,owners: BandConcatExecution.PartContents<Element> & Layout.Survey) -> LogicalConcat<Element>:
  Bands.Geometry{images,channels,height,width} = geometry
  (parts,summary) = owners
  LogicalConcat{parts,height,width}

def JoinedValues(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,geometry: Bands.Geometry,+layout: Bands.Plan,+fill: Element,owners: BandConcatExecution.PartContents<Element> & Layout.Survey) -> Data:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  (+parts,summary) = owners
  BandValues.Values(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{parts,height,width},geometry,BandConcatExecution.state_output(~Element,BandConcatExecution.parts_model(~Element,parts,geometry,(BandConcatExecution.allocation_model(~Element,geometry,fill,layout),0))))

def joined_expected_values(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+first: List<&2,U32>,+axis: U32,+fill: Element,geometry: Bands.Geometry,+layout: Bands.Plan,owners: BandConcatExecution.PartContents<Element> & Layout.Survey,
  values: JoinedValues(~Element,~band_value,geometry,layout,fill,owners)) -> TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),logical_owner(~Element,geometry,owners),BandConcatCaller.joined_expected(~Element,first,axis,fill,geometry,layout,owners)):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  (+parts,summary) = owners
  Layout.Survey{valid,total,status,workers} = summary
  values

type LayoutFields<-layout: Bands.Plan,-images: U32,-height: U32> is Data:
  LayoutFields{local: BandPlan.ImageLocal(layout,U32.to_nat(images),U32.to_nat(height)),
    within: BandPlan.Within(layout,0n,Nat.mul(U32.to_nat(images),U32.to_nat(height))),coverage: BandPlan.Coverage(layout,0n,Nat.mul(U32.to_nat(images),U32.to_nat(height)))}

def positive_height_room(+images: U32,+output_channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+images_positive: {Nat.is_lt(0n,U32.to_nat(images)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(height),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  BandGather.Domain{capacity_ok,+channels_positive,height_positive,+width_positive,+machine_room,allocation_room} = domain
  +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +frame = Nat.mul(U32.to_nat(output_channels),spatial)
  +height_frame = Order.transitive(U32.to_nat(height),spatial,frame,RowMajor.product_covers(U32.to_nat(height),U32.to_nat(width),width_positive),
    %RowMajor.multiply_commutative(spatial,U32.to_nat(output_channels)) : {Nat.is_le(spatial,_) == True{} : Bool} RowMajor.product_covers(spatial,U32.to_nat(output_channels),channels_positive))
  Order.transitive(U32.to_nat(height),frame,U32.to_nat(Machine.capacity(limits)),height_frame,
    Order.transitive(frame,Nat.mul(U32.to_nat(images),frame),U32.to_nat(Machine.capacity(limits)),
      %RowMajor.multiply_commutative(frame,U32.to_nat(images)) : {Nat.is_le(frame,_) == True{} : Bool} RowMajor.product_covers(frame,U32.to_nat(images),images_positive),machine_room))

def nonempty_height_room(+images: U32,+output_channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+limits: Machine.Limits,+capacity: U32,
  domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(height),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +rows_room = Order.transitive(U32.to_nat(rows),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height)),
    %Order.commutative(U32.to_nat(rows),U32.to_nat(first)) : {Nat.is_le(U32.to_nat(rows),_) == True{} : Bool} Order.le_add(U32.to_nat(rows),U32.to_nat(first)),rows_fit)
  +images_positive = RowMajor.left_factor_positive(U32.to_nat(images),U32.to_nat(height),Bounds.nat_trans(0n,U32.to_nat(rows),Nat.mul(U32.to_nat(images),U32.to_nat(height)),rows_positive,rows_room))
  positive_height_room(images,output_channels,height,width,limits,capacity,domain,images_positive)

def positive_image_span(+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+length: Nat,+limits: Machine.Limits,+capacity: U32,
  metadata: BandShape.Leaf(Bands.Geometry{images,source_channels,height,width},first,rows,length),domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}) -> BandPlan.ImageSpan<first,rows,U32.to_nat(images),U32.to_nat(height)>:
  BandShape.Fields{+local,+rows_fit,length_exact} = metadata
  BandGather.Domain{capacity_ok,channels_positive,+height_positive,width_positive,machine_room,allocation_room} = domain
  +height_nat = U32.to_nat(height)
  +image = Nat.div(U32.to_nat(first),height_nat)
  +local_first = Nat.mod(U32.to_nat(first),height_nat)
  +first_exact = DivisionOrder.quotient_exact(U32.to_nat(first),height_nat,height_positive)
  +span = Inspection.local_span(height,first,rows,Machine.half_word(limits),Limits.half_word_fits(limits),height_positive,Limits.double_bound(limits,height_nat,nonempty_height_room(images,output_channels,height,width,first,rows,limits,capacity,domain,rows_fit,positive)),local)
  +base = Nat.mul(image,height_nat)
  BandPlan.ImageSpan{image,image_inside_band(images,height,first,rows,image,local_first,0n,first_exact,rows_fit,positive),BandPlan.LeafWithin{
    %Equal.sym(Nat,U32.to_nat(first),Nat.add(base,local_first),first_exact) : {Nat.is_le(base,_) == True{} : Bool} Order.le_add(base,local_first),
    %Equal.sym(Nat,U32.to_nat(first),Nat.add(base,local_first),first_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(rows)),Nat.add(base,height_nat)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.add(base,local_first),U32.to_nat(rows)),Nat.add(base,Nat.add(local_first,U32.to_nat(rows))),Addition.associative(base,local_first,U32.to_nat(rows))) : {Nat.is_le(_,Nat.add(base,height_nat)) == True{} : Bool}
    Order.offset_le(base,Nat.add(local_first,U32.to_nat(rows)),height_nat,span)}}

def leaf_image_span(rows_count: Nat,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+length: Nat,+limits: Machine.Limits,+capacity: U32,+count: {U32.to_nat(rows) == rows_count : Nat},
  metadata: BandShape.Leaf(Bands.Geometry{images,source_channels,height,width},first,rows,length),+domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>) -> BandPlan.ImageSpan<first,rows,U32.to_nat(images),U32.to_nat(height)>:
  match rows_count:
    case 0n: BandPlan.EmptySpan{count}
    case 1n+ +rest:
      positive_image_span(images,source_channels,output_channels,height,width,first,rows,length,limits,capacity,metadata,domain,
        %Equal.sym(Nat,U32.to_nat(rows),1n+rest,count) : {Nat.is_lt(0n,_) == True{} : Bool} Order.lt_succ(0n,rest,Order.zero_le(rest)))

def content_image_local(~Element: Data,source: Model.Content<Element>,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,
  metadata: BandShape.Contents(~Element,Bands.Geometry{images,source_channels,height,width},source),+domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>) -> BandPlan.ImageLocal(Model.content_layout(~Element,source),U32.to_nat(images),U32.to_nat(height)):
  match source:
    case Model.RowBand{Buffer.Content{tree,+length},+first,+rows}: leaf_image_span(U32.to_nat(rows),images,source_channels,output_channels,height,width,first,rows,length,limits,capacity,{==},metadata,domain)
    case Model.Joined{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{content_image_local(~Element,left,images,source_channels,output_channels,height,width,limits,capacity,left_metadata,domain),content_image_local(~Element,right,images,source_channels,output_channels,height,width,limits,capacity,right_metadata,domain)}

def source_layout_fields(~Element: Data,+source: Model.Content<Element>,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,
  well_formed: BandShape.WellFormed<Element,Bands.Geometry{images,source_channels,height,width},source>,+domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>) -> LayoutFields<Model.content_layout(~Element,source),images,height>:
  BandShape.WellFormed{+metadata,ready,+coverage} = well_formed
  LayoutFields{content_image_local(~Element,source,images,source_channels,output_channels,height,width,limits,capacity,metadata,domain),BandPlan.covered(Model.content_layout(~Element,source),0n,Nat.mul(U32.to_nat(images),U32.to_nat(height)),0n,Nat.mul(U32.to_nat(images),U32.to_nat(height)),coverage,{==},Order.reflexive(Nat.mul(U32.to_nat(images),U32.to_nat(height)))),coverage}

def ChoiceFields(choice: Maybe<&2,Bands.Plan>,+images: U32,+height: U32) -> Data:
  match choice:
    case None{}: Unit
    case Some{layout}: LayoutFields<layout,images,height>

def tensor_layout_fields(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,input: BandConcatExecution.TensorContent<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,
  contract: SourceContract(~Element,~band_value,images,height,width,input),+domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>) -> ChoiceFields(BandConcatCaller.tensor_choice(~Element,input),images,height):
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}: Unit{}
    case BandConcatExecution.BandedContent{+bands,geometry,+shape,fill,status,workers}:
      Shape.Both{well_formed,values} = contract
      source_layout_fields(~Element,bands,images,BandConcat.channels(shape),output_channels,height,width,limits,capacity,well_formed,domain)

def discovery_fields(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,parts: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,
  contracts: SourceContracts(~Element,~band_value,images,height,width,parts),+domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>) -> ChoiceFields(BandConcatCaller.discovery_model(~Element,parts),images,height):
  match parts:
    case BandConcatExecution.SingleContent{input}: tensor_layout_fields(~Element,~band_value,input,images,output_channels,height,width,limits,capacity,contracts,domain)
    case BandConcatExecution.ConsContent{input,+rest}:
      match input:
        case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}:
          Shape.Both{head,tail} = contracts
          discovery_fields(~Element,~band_value,rest,images,output_channels,height,width,limits,capacity,tail,domain)
        case BandConcatExecution.BandedContent{+bands,geometry,+shape,fill,status,workers}:
          Shape.Both{head,tail} = contracts
          Shape.Both{well_formed,values} = head
          source_layout_fields(~Element,bands,images,BandConcat.channels(shape),output_channels,height,width,limits,capacity,well_formed,domain)

def discovered_fields(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+parts: BandConcatExecution.PartContents<Element>,+images: U32,+output_channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,+layout: Bands.Plan,
  +contracts: SourceContracts(~Element,~band_value,images,height,width,parts),+domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,capacity>,+selected: {BandConcatCaller.discovery_model(~Element,parts) == Some{layout} : Maybe<&2,Bands.Plan>}) -> LayoutFields<layout,images,height>:
  %selected : ChoiceFields(_,images,height)
  discovery_fields(~Element,~band_value,parts,images,output_channels,height,width,limits,capacity,contracts,domain)

def layout_fitting(+layout: Bands.Plan,+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,fields: LayoutFields<layout,images,height>,+domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>) -> BandDense.Fits(Bands.Geometry{images,channels,height,width},layout,capacity,Machine.half_word(limits)):
  LayoutFields{local,within,coverage} = fields
  BandDense.fits(layout,images,channels,height,width,limits,capacity,local,within,domain)

def layout_coverage(+layout: Bands.Plan,+images: U32,+height: U32,fields: LayoutFields<layout,images,height>) -> BandPlan.Coverage(layout,0n,Nat.mul(U32.to_nat(images),U32.to_nat(height))):
  LayoutFields{local,within,coverage} = fields
  coverage

def concat_first_values_packed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+sources: BandConcatExecution.PartsOwnerModel<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+fill: Element,+count: U32,+layout: Bands.Plan,+limits: Machine.Limits,+capacity: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  guards: BandConcatCaller.BandSurveyFields<Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)))),Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},count,layout>,
  +contracts: SourceContracts(~Element,~band_value,BandConcatCaller.images_of(rank),height,width,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),
  +domain: BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},limits,capacity>) ->
  TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width},BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,BandConcatCaller.axis_of(rank),BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.shape_of(rank,channels,height,width),fill))):
  BandConcatCaller.BandSurveyFields{+accepted,+status_guard,+geometry_guard,+count_guard,+layout_guard,+compatible_guard} = guards
  +observed_parts = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))
  +first = {BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>}
  +summary = BandConcatCaller.initial_summary(first,U32.to_nat(BandConcatCaller.axis_of(rank)))
  +survey = BandConcatCaller.survey_model(~Element,observed_parts,first,U32.to_nat(BandConcatCaller.axis_of(rank)),summary)
  +output_channels = BandConcatCaller.survey_total(survey)
  +geometry = {Bands.Geometry{BandConcatCaller.images_of(rank),output_channels,height,width} : Bands.Geometry}
  +output = BandConcatExecution.allocation_model(~Element,geometry,fill,layout)
  +fields = discovered_fields(~Element,~band_value,observed_parts,BandConcatCaller.images_of(rank),output_channels,height,width,limits,capacity,layout,contracts,domain,layout_guard)
  +fitting = layout_fitting(layout,BandConcatCaller.images_of(rank),output_channels,height,width,limits,capacity,fields,domain)
  +coverage = layout_coverage(layout,BandConcatCaller.images_of(rank),height,fields)
  %Equal.sym(BandConcatExecution.TensorContent<Element>,BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,BandConcatCaller.axis_of(rank),BandConcatExecution.parts_owner_pack(~Element,sources),first,fill)),BandConcatCaller.joined_expected(~Element,first,BandConcatCaller.axis_of(rank),fill,geometry,layout,(observed_parts,survey)),BandConcatCaller.concat_first_band_packed(~Element,sources,first,BandConcatCaller.axis_of(rank),fill,geometry,count,layout,BandConcatCaller.BandSurveyFields{accepted,status_guard,geometry_guard,count_guard,layout_guard,compatible_guard})) : TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{observed_parts,height,width},_)
  joined_expected_values(~Element,~band_value,first,BandConcatCaller.axis_of(rank),fill,geometry,layout,(observed_parts,survey),
    parts_values(~Element,~band_value,observed_parts,BandConcatCaller.images_of(rank),output_channels,height,width,output,limits,capacity,contracts,BandConcatCaller.allocation_well_formed(~Element,layout,geometry,fill,capacity,Machine.half_word(limits),fitting,coverage),domain,
      operational_rooms(~Element,observed_parts,first,U32.to_nat(BandConcatCaller.axis_of(rank)),summary,0,surveyed_rooms(~Element,observed_parts,rank,channels,height,width,summary,limit,maximum,accepted),{==}),surveyed_total(~Element,observed_parts,rank,channels,height,width,limit,maximum,accepted)))

def concat_first_values_observed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,-parts: Tensors.Parts<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+fill: Element,+count: U32,+layout: Bands.Plan,+limits: Machine.Limits,+capacity: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  observation: BandConcatExecution.PartsObservation<Element,parts>,
  guards: BandConcatCaller.BandSurveyFields<Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)))),Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},count,layout>,
  +contracts: SourceContracts(~Element,~band_value,BandConcatCaller.images_of(rank),height,width,BandConcatExecution.parts_contents(~Element,parts)),
  +domain: BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},limits,capacity>) ->
  TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,parts),height,width},BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,BandConcatCaller.axis_of(rank),parts,BandConcatCaller.shape_of(rank,channels,height,width),fill))):
  BandConcatExecution.PartsObservation{+sources,+equal} = observation
  %equal : TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,_),height,width},BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,BandConcatCaller.axis_of(rank),_,BandConcatCaller.shape_of(rank,channels,height,width),fill)))
  concat_first_values_packed(~Element,~band_value,sources,rank,channels,height,width,fill,count,layout,limits,capacity,limit,maximum,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equal) : BandConcatCaller.BandSurveyFields<Element,BandConcatExecution.parts_contents(~Element,_),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)))),Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},count,layout> guards,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equal) : SourceContracts(~Element,~band_value,BandConcatCaller.images_of(rank),height,width,BandConcatExecution.parts_contents(~Element,_)) contracts,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equal) : BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},limits,capacity> domain)

def concat_first_values(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,parts: Tensors.Parts<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+fill: Element,+count: U32,+layout: Bands.Plan,+limits: Machine.Limits,+capacity: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  guards: BandConcatCaller.BandSurveyFields<Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)))),Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},count,layout>,
  +contracts: SourceContracts(~Element,~band_value,BandConcatCaller.images_of(rank),height,width,BandConcatExecution.parts_contents(~Element,parts)),
  +domain: BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},limits,capacity>) ->
  TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,parts),height,width},BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,BandConcatCaller.axis_of(rank),parts,BandConcatCaller.shape_of(rank,channels,height,width),fill))):
  concat_first_values_observed(~Element,~band_value,parts,rank,channels,height,width,fill,count,layout,limits,capacity,limit,maximum,BandConcatExecution.observe_parts(~Element,parts),guards,contracts,domain)

# These are the actual successful public branch tests. Geometry, layout
# fitting, channel totals, cursor bounds and output Values are produced.
type NativeBandFields<-Element: Data,-parts: Tensors.Parts<Element>,-first: List<&2,U32>,-axis: U32,-count: U32,-layout: Bands.Plan> is Data:
  NativeBandFields{
    valid: {BandConcatCaller.survey_valid(BandConcatCaller.observed_summary(~Element,Layout.survey(~Element,parts,first,U32.to_nat(axis),BandConcatCaller.initial_summary(first,U32.to_nat(axis))))) == True{} : Bool},
    status: {BandConcatCaller.status_of(BandConcatCaller.observed_summary(~Element,Layout.survey(~Element,parts,first,U32.to_nat(axis),BandConcatCaller.initial_summary(first,U32.to_nat(axis))))) == Tensors.Ready{} : Tensors.Status},
    counted: {Tensors.count_of(BandConcatCaller.target_shape(first,axis,BandConcatCaller.observed_summary(~Element,Layout.survey(~Element,parts,first,U32.to_nat(axis),BandConcatCaller.initial_summary(first,U32.to_nat(axis)))))) == Some{count} : Maybe<&2,U32>},
    layout: {BandConcatCaller.discovered_plan(~Element,BandConcat.discover(~Element,parts)) == Some{layout} : Maybe<&2,Bands.Plan>},
    compatible: {BandConcatCaller.compatible_flag(~Element,BandConcatCaller.compatible_model(~Element,BandConcatExecution.parts_contents(~Element,parts))) && Bool.not(Shapes.has_zero(BandConcatCaller.target_shape(first,axis,BandConcatCaller.observed_summary(~Element,Layout.survey(~Element,parts,first,U32.to_nat(axis),BandConcatCaller.initial_summary(first,U32.to_nat(axis))))))) == True{} : Bool}}

def survey_fields(~Element: Data,+parts: BandConcatExecution.PartContents<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+summary: Layout.Survey,+count: U32,+layout: Bands.Plan,
  valid: {BandConcatCaller.survey_valid(summary) == True{} : Bool},status: {BandConcatCaller.status_of(summary) == Tensors.Ready{} : Tensors.Status},
  counted: {Tensors.count_of(BandConcatCaller.target_shape(BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),summary)) == Some{count} : Maybe<&2,U32>},selected: {BandConcatCaller.discovery_model(~Element,parts) == Some{layout} : Maybe<&2,Bands.Plan>},
  compatible: {BandConcatCaller.compatible_flag(~Element,BandConcatCaller.compatible_model(~Element,parts)) && Bool.not(Shapes.has_zero(BandConcatCaller.target_shape(BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),summary))) == True{} : Bool}) ->
  BandConcatCaller.BandSurveyFields<Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),summary,Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(summary),height,width},count,layout>:
  BandConcatCaller.BandSurveyFields{valid,status,BandConcatCaller.output_geometry(rank,channels,height,width,BandConcatCaller.survey_total(summary)),counted,selected,compatible}

def native_fields_packed(~Element: Data,+sources: BandConcatExecution.PartsOwnerModel<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+count: U32,+layout: Bands.Plan,
  guards: NativeBandFields<Element,BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),count,layout>) ->
  BandConcatCaller.BandSurveyFields<Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)))),Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},count,layout>:
  NativeBandFields{valid,status,counted,selected,compatible} = guards
  +first = {BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>}
  +summary_equal = BandConcatCaller.native_summary_packed(~Element,sources,first,U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(first,U32.to_nat(BandConcatCaller.axis_of(rank))))
  survey_fields(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),rank,channels,height,width,BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(first,U32.to_nat(BandConcatCaller.axis_of(rank)))),count,layout,
    %summary_equal : {BandConcatCaller.survey_valid(_) == True{} : Bool} valid,
    %summary_equal : {BandConcatCaller.status_of(_) == Tensors.Ready{} : Tensors.Status} status,
    %summary_equal : {Tensors.count_of(BandConcatCaller.target_shape(first,BandConcatCaller.axis_of(rank),_)) == Some{count} : Maybe<&2,U32>} counted,
    %BandConcatCaller.discover_plan_packed(~Element,sources) : {_ == Some{layout} : Maybe<&2,Bands.Plan>} selected,
    %summary_equal : {BandConcatCaller.compatible_flag(~Element,BandConcatCaller.compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)))) && Bool.not(Shapes.has_zero(BandConcatCaller.target_shape(first,BandConcatCaller.axis_of(rank),_))) == True{} : Bool} compatible)

def native_fields_observed(~Element: Data,-parts: Tensors.Parts<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+count: U32,+layout: Bands.Plan,
  observation: BandConcatExecution.PartsObservation<Element,parts>,guards: NativeBandFields<Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),count,layout>) ->
  BandConcatCaller.BandSurveyFields<Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)))),Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},count,layout>:
  BandConcatExecution.PartsObservation{+sources,+equal} = observation
  %equal : BandConcatCaller.BandSurveyFields<Element,BandConcatExecution.parts_contents(~Element,_),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)))),Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},count,layout>
  native_fields_packed(~Element,sources,rank,channels,height,width,count,layout,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equal) : NativeBandFields<Element,_,BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),count,layout> guards)

def native_fields(~Element: Data,parts: Tensors.Parts<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+count: U32,+layout: Bands.Plan,
  guards: NativeBandFields<Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),count,layout>) ->
  BandConcatCaller.BandSurveyFields<Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)))),Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},count,layout>:
  native_fields_observed(~Element,parts,rank,channels,height,width,count,layout,BandConcatExecution.observe_parts(~Element,parts),guards)

def native_concat_first_observed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,-parts: Tensors.Parts<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+fill: Element,+count: U32,+layout: Bands.Plan,+limits: Machine.Limits,+capacity: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  +observation: BandConcatExecution.PartsObservation<Element,parts>,
  guards: NativeBandFields<Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),count,layout>,
  contracts: SourceContracts(~Element,~band_value,BandConcatCaller.images_of(rank),height,width,BandConcatExecution.parts_contents(~Element,parts)),
  domain: BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},limits,capacity>) ->
  TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,parts),height,width},BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,BandConcatCaller.axis_of(rank),parts,BandConcatCaller.shape_of(rank,channels,height,width),fill))):
  concat_first_values_observed(~Element,~band_value,parts,rank,channels,height,width,fill,count,layout,limits,capacity,limit,maximum,observation,native_fields_observed(~Element,parts,rank,channels,height,width,count,layout,observation,guards),contracts,domain)

def native_concat_first(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,parts: Tensors.Parts<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+fill: Element,+count: U32,+layout: Bands.Plan,+limits: Machine.Limits,+capacity: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  guards: NativeBandFields<Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),count,layout>,
  contracts: SourceContracts(~Element,~band_value,BandConcatCaller.images_of(rank),height,width,BandConcatExecution.parts_contents(~Element,parts)),
  domain: BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},limits,capacity>) ->
  TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,parts),height,width},BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,BandConcatCaller.axis_of(rank),parts,BandConcatCaller.shape_of(rank,channels,height,width),fill))):
  native_concat_first_observed(~Element,~band_value,parts,rank,channels,height,width,fill,count,layout,limits,capacity,limit,maximum,BandConcatExecution.observe_parts(~Element,parts),guards,contracts,domain)

def first_shape(~Element: Data,parts: BandConcatExecution.PartContents<Element>) -> List<&2,U32>:
  match parts:
    case BandConcatExecution.SingleContent{input}: BandConcatExecution.tensor_shape(~Element,input)
    case BandConcatExecution.ConsContent{input,rest}: BandConcatExecution.tensor_shape(~Element,input)

def first_fill(~Element: Data,parts: BandConcatExecution.PartContents<Element>) -> Element:
  match parts:
    case BandConcatExecution.SingleContent{input}: BandConcatCaller.tensor_fill(~Element,input)
    case BandConcatExecution.ConsContent{input,rest}: BandConcatCaller.tensor_fill(~Element,input)

def native_concat_lead(~Element: Data,+axis: U32,parts: Tensors.Parts<Element>) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,parts,axis)) == BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,parts,first_shape(~Element,BandConcatExecution.parts_contents(~Element,parts)),first_fill(~Element,BandConcatExecution.parts_contents(~Element,parts)))) : BandConcatExecution.TensorContent<Element>}:
  match parts:
    case Tensors.Single{input}:
      match input:
        case Tensors.Tensor{buffer,shape,fill,status,workers}: {==}
        case Tensors.Banded{bands,geometry,shape,fill,status,workers}: {==}
    case Tensors.Cons{input,rest}:
      match input:
        case Tensors.Tensor{buffer,shape,fill,status,workers}: {==}
        case Tensors.Banded{bands,geometry,shape,fill,status,workers}: {==}

def native_concat_shape(~Element: Data,parts: Tensors.Parts<Element>,+axis: U32,+shape: List<&2,U32>,
  +shape_selected: {first_shape(~Element,BandConcatExecution.parts_contents(~Element,parts)) == shape : List<&2,U32>}) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,parts,axis)) == BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,parts,shape,first_fill(~Element,BandConcatExecution.parts_contents(~Element,parts)))) : BandConcatExecution.TensorContent<Element>}:
  %shape_selected : {BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,parts,axis)) == BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,parts,_,first_fill(~Element,BandConcatExecution.parts_contents(~Element,parts)))) : BandConcatExecution.TensorContent<Element>}
  native_concat_lead(~Element,axis,parts)

def native_concat_packed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+sources: BandConcatExecution.PartsOwnerModel<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+count: U32,+layout: Bands.Plan,+limits: Machine.Limits,+capacity: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  +first: {first_shape(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))) == BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>},
  guards: NativeBandFields<Element,BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),count,layout>,
  contracts: SourceContracts(~Element,~band_value,BandConcatCaller.images_of(rank),height,width,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),
  domain: BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},limits,capacity>) ->
  TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width},BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.axis_of(rank)))):
  %Equal.sym(BandConcatExecution.TensorContent<Element>,BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.axis_of(rank))),BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,BandConcatCaller.axis_of(rank),BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.shape_of(rank,channels,height,width),first_fill(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))))),native_concat_shape(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.axis_of(rank),BandConcatCaller.shape_of(rank,channels,height,width),first)) : TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width},_)
  native_concat_first(~Element,~band_value,BandConcatExecution.parts_owner_pack(~Element,sources),rank,channels,height,width,first_fill(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),count,layout,limits,capacity,limit,maximum,guards,contracts,domain)

def native_concat_observed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,-parts: Tensors.Parts<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+count: U32,+layout: Bands.Plan,+limits: Machine.Limits,+capacity: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  observation: BandConcatExecution.PartsObservation<Element,parts>,
  first: {first_shape(~Element,BandConcatExecution.parts_contents(~Element,parts)) == BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>},
  guards: NativeBandFields<Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),count,layout>,
  contracts: SourceContracts(~Element,~band_value,BandConcatCaller.images_of(rank),height,width,BandConcatExecution.parts_contents(~Element,parts)),
  domain: BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},limits,capacity>) ->
  TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,parts),height,width},BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,parts,BandConcatCaller.axis_of(rank)))):
  BandConcatExecution.PartsObservation{+sources,+equal} = observation
  %equal : TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,_),height,width},BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,_,BandConcatCaller.axis_of(rank))))
  native_concat_packed(~Element,~band_value,sources,rank,channels,height,width,count,layout,limits,capacity,limit,maximum,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equal) : {first_shape(~Element,BandConcatExecution.parts_contents(~Element,_)) == BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>} first,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equal) : NativeBandFields<Element,_,BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),count,layout> guards,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equal) : SourceContracts(~Element,~band_value,BandConcatCaller.images_of(rank),height,width,BandConcatExecution.parts_contents(~Element,_)) contracts,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equal) : BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},limits,capacity> domain)

def native_concat(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,parts: Tensors.Parts<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+count: U32,+layout: Bands.Plan,+limits: Machine.Limits,+capacity: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  first: {first_shape(~Element,BandConcatExecution.parts_contents(~Element,parts)) == BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>},
  guards: NativeBandFields<Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),count,layout>,
  contracts: SourceContracts(~Element,~band_value,BandConcatCaller.images_of(rank),height,width,BandConcatExecution.parts_contents(~Element,parts)),
  domain: BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},limits,capacity>) ->
  TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,parts),height,width},BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,parts,BandConcatCaller.axis_of(rank)))):
  native_concat_observed(~Element,~band_value,parts,rank,channels,height,width,count,layout,limits,capacity,limit,maximum,BandConcatExecution.observe_parts(~Element,parts),first,guards,contracts,domain)

def status_ready(status: Tensors.Status) -> Bool:
  match status:
    case Tensors.Ready{}: True{}
    case Tensors.Failed{problem}: False{}

def merge_previous_ready(previous: Tensors.Status,current: Tensors.Status,
  accepted: {status_ready(Tensors.first_failure(previous,current)) == True{} : Bool}) -> {status_ready(previous) == True{} : Bool}:
  match previous current:
    case Tensors.Ready{} current: {==}
    case Tensors.Failed{problem} current: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(accepted))

def merge_current_ready(previous: Tensors.Status,current: Tensors.Status,
  accepted: {status_ready(Tensors.first_failure(previous,current)) == True{} : Bool}) -> {status_ready(current) == True{} : Bool}:
  match previous current:
    case Tensors.Ready{} current: accepted
    case Tensors.Failed{problem} current: Empty.absurd({status_ready(current) == True{} : Bool},WordEncoding.false_true(accepted))

def survey_status_step(~Element: Data,+input: BandConcatExecution.TensorContent<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {BandConcatCaller.status_of(BandConcatCaller.step(~Element,first,axis,input,summary)) == Tensors.first_failure(BandConcatCaller.status_of(summary),BandConcatCaller.tensor_status(~Element,input)) : Tensors.Status}:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}:
      Layout.Survey{valid,total,previous,most} = summary
      {==}
    case BandConcatExecution.BandedContent{bands,geometry,shape,fill,status,workers}:
      Layout.Survey{valid,total,previous,most} = summary
      {==}

def step_previous_ready(~Element: Data,+input: BandConcatExecution.TensorContent<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,
  accepted: {status_ready(BandConcatCaller.status_of(BandConcatCaller.step(~Element,first,axis,input,summary))) == True{} : Bool}) -> {status_ready(BandConcatCaller.status_of(summary)) == True{} : Bool}:
  merge_previous_ready(BandConcatCaller.status_of(summary),BandConcatCaller.tensor_status(~Element,input),
    %survey_status_step(~Element,input,first,axis,summary) : {status_ready(_) == True{} : Bool} accepted)

def step_input_ready(~Element: Data,+input: BandConcatExecution.TensorContent<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,
  accepted: {status_ready(BandConcatCaller.status_of(BandConcatCaller.step(~Element,first,axis,input,summary))) == True{} : Bool}) -> {status_ready(BandConcatCaller.tensor_status(~Element,input)) == True{} : Bool}:
  merge_current_ready(BandConcatCaller.status_of(summary),BandConcatCaller.tensor_status(~Element,input),
    %survey_status_step(~Element,input,first,axis,summary) : {status_ready(_) == True{} : Bool} accepted)

def survey_previous_ready(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,
  accepted: {status_ready(BandConcatCaller.status_of(BandConcatCaller.survey_model(~Element,parts,first,axis,summary))) == True{} : Bool}) -> {status_ready(BandConcatCaller.status_of(summary)) == True{} : Bool}:
  match parts:
    case BandConcatExecution.SingleContent{input}: step_previous_ready(~Element,input,first,axis,summary,accepted)
    case BandConcatExecution.ConsContent{+input,rest}:
      step_previous_ready(~Element,input,first,axis,summary,survey_previous_ready(~Element,rest,first,axis,BandConcatCaller.step(~Element,first,axis,input,summary),accepted))

def SourceStatuses(~Element: Data,parts: BandConcatExecution.PartContents<Element>) -> Data:
  match parts:
    case BandConcatExecution.SingleContent{input}: {status_ready(BandConcatCaller.tensor_status(~Element,input)) == True{} : Bool}
    case BandConcatExecution.ConsContent{input,rest}: Shape.Both<{status_ready(BandConcatCaller.tensor_status(~Element,input)) == True{} : Bool},SourceStatuses(~Element,rest)>

def surveyed_statuses(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,
  +accepted: {status_ready(BandConcatCaller.status_of(BandConcatCaller.survey_model(~Element,parts,first,axis,summary))) == True{} : Bool}) -> SourceStatuses(~Element,parts):
  match parts:
    case BandConcatExecution.SingleContent{input}: step_input_ready(~Element,input,first,axis,summary,accepted)
    case BandConcatExecution.ConsContent{+input,+rest}:
      Shape.Both{step_input_ready(~Element,input,first,axis,summary,survey_previous_ready(~Element,rest,first,axis,BandConcatCaller.step(~Element,first,axis,input,summary),accepted)),
        surveyed_statuses(~Element,rest,first,axis,BandConcatCaller.step(~Element,first,axis,input,summary),accepted)}

def decoded_source_three(+shape: List<&2,U32>,+height: U32,+width: U32,
  suffix: {Shapes.same_extents([height,width],List.drop(&2,U32,shape,1n)) == True{} : Bool}) ->
  {BandCount.decode(shape) == Some{Bands.Geometry{1,BandConcat.channels(shape),height,width}} : Maybe<&2,Bands.Geometry>}:
  match shape:
    case []: Empty.absurd({None{} == Some{Bands.Geometry{1,0,height,width}} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(suffix))
    case +channels <> +tail:
      %ShapeLaws.same_extents([height,width],tail,BandConcatCaller.suffix_tail(height,width,tail,suffix)) : {BandCount.decode(channels <> _) == Some{Bands.Geometry{1,BandConcat.channels(channels <> _),height,width}} : Maybe<&2,Bands.Geometry>}
      {==}

def decoded_source_four(+shape: List<&2,U32>,+images: U32,+height: U32,+width: U32,
  prefix: {Shapes.same_extents([images],List.take(&2,U32,shape,1n)) == True{} : Bool},
  suffix: {Shapes.same_extents([height,width],List.drop(&2,U32,shape,2n)) == True{} : Bool}) ->
  {BandCount.decode(shape) == Some{Bands.Geometry{images,BandConcat.channels(shape),height,width}} : Maybe<&2,Bands.Geometry>}:
  match shape:
    case []: Empty.absurd({None{} == Some{Bands.Geometry{images,0,height,width}} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(suffix))
    case +image <> +rest:
      match rest:
        case []: Empty.absurd({None{} == Some{Bands.Geometry{images,0,height,width}} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(suffix))
        case +channels <> +tail:
          +same_image = ShapeLaws.native_equal(images,image,Boolean.left(U32.is_eq(images,image),Shapes.same_extents([],[]),prefix))
          %same_image : {BandCount.decode(_ <> channels <> tail) == Some{Bands.Geometry{images,BandConcat.channels(_ <> channels <> tail),height,width}} : Maybe<&2,Bands.Geometry>}
          %ShapeLaws.same_extents([height,width],tail,BandConcatCaller.suffix_tail(height,width,tail,suffix)) : {BandCount.decode(images <> channels <> _) == Some{Bands.Geometry{images,BandConcat.channels(images <> channels <> _),height,width}} : Maybe<&2,Bands.Geometry>}
          {==}

def step_decoded_source(+rank: BandConcatCaller.Rank,+first_channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+previous: U32,+limit: U32,
  fields: BandConcatCaller.StepFields<BandConcatCaller.shape_of(rank,first_channels,height,width),shape,U32.to_nat(BandConcatCaller.axis_of(rank)),previous,limit>) ->
  {BandCount.decode(shape) == Some{Bands.Geometry{BandConcatCaller.images_of(rank),BandConcat.channels(shape),height,width}} : Maybe<&2,Bands.Geometry>}:
  match rank:
    case BandConcatCaller.RankThree{}:
      BandConcatCaller.StepFields{length,prefix,suffix,room,exact} = fields
      decoded_source_three(shape,height,width,suffix)
    case BandConcatCaller.RankFour{+images}:
      BandConcatCaller.StepFields{length,prefix,suffix,room,exact} = fields
      decoded_source_four(shape,images,height,width,prefix,suffix)

def SourceGeometries(~Element: Data,+images: U32,+height: U32,+width: U32,parts: BandConcatExecution.PartContents<Element>) -> Data:
  match parts:
    case BandConcatExecution.SingleContent{input}: {BandCount.decode(BandConcatExecution.tensor_shape(~Element,input)) == Some{Bands.Geometry{images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),height,width}} : Maybe<&2,Bands.Geometry>}
    case BandConcatExecution.ConsContent{input,rest}: Shape.Both<{BandCount.decode(BandConcatExecution.tensor_shape(~Element,input)) == Some{Bands.Geometry{images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),height,width}} : Maybe<&2,Bands.Geometry>},SourceGeometries(~Element,images,height,width,rest)>

def surveyed_geometries(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+summary: Layout.Survey,+limit: U32,
  fields: BandConcatCaller.Prefixes(~Element,parts,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),summary,limit)) -> SourceGeometries(~Element,BandConcatCaller.images_of(rank),height,width,parts):
  match parts:
    case BandConcatExecution.SingleContent{+input}:
      step_decoded_source(rank,channels,height,width,BandConcatExecution.tensor_shape(~Element,input),BandConcatCaller.survey_total(summary),limit,fields)
    case BandConcatExecution.ConsContent{+input,+rest}:
      Shape.Both{head,tail} = fields
      Shape.Both{step_decoded_source(rank,channels,height,width,BandConcatExecution.tensor_shape(~Element,input),BandConcatCaller.survey_total(summary),limit,head),
        surveyed_geometries(~Element,rest,rank,channels,height,width,BandConcatCaller.step(~Element,BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),input,summary),limit,tail)}

def IncomingNormals(~Element: Data,sources: BandConcatExecution.PartsOwnerModel<Element>) -> Data:
  match sources:
    case BandConcatExecution.SingleOwner{input}: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))
    case BandConcatExecution.ConsOwner{input,rest}: Shape.Both<Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),IncomingNormals(~Element,rest)>

def IncomingValues(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,input: BandConcatExecution.TensorOwnerModel<Element>) -> Data:
  match input:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}: Unit
    case BandConcatExecution.BandedOwner{+bands,+geometry,+shape,+fill,Tensors.Ready{},+workers}:
      BandValues.Values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),geometry,Model.contents(~Element,Model.pack(~Element,bands)))
    case BandConcatExecution.BandedOwner{bands,geometry,shape,fill,Tensors.Failed{problem},workers}: Unit

def IncomingPartsValues(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,sources: BandConcatExecution.PartsOwnerModel<Element>) -> Data:
  match sources:
    case BandConcatExecution.SingleOwner{input}: IncomingValues(~Element,~band_value,input)
    case BandConcatExecution.ConsOwner{input,rest}: Shape.Both<IncomingValues(~Element,~band_value,input),IncomingPartsValues(~Element,~band_value,rest)>

def incoming_source_contract(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,input: BandConcatExecution.TensorOwnerModel<Element>,+images: U32,+height: U32,+width: U32,
  decoded: {BandCount.decode(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))) == Some{Bands.Geometry{images,BandConcat.channels(BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),height,width}} : Maybe<&2,Bands.Geometry>},
  ready: {status_ready(BandConcatCaller.tensor_status(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))) == True{} : Bool},
  normal: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),values: IncomingValues(~Element,~band_value,input)) ->
  SourceContract(~Element,~band_value,images,height,width,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))):
  match input:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}: Unit{}
    case BandConcatExecution.BandedOwner{+bands,+geometry,+shape,+fill,Tensors.Ready{},+workers}:
      Shape.Both{well_formed,Shape.Both{counted,+coherent}} = normal
      +expected = {Bands.Geometry{images,BandConcat.channels(shape),height,width} : Bands.Geometry}
      +same_geometry = Equal.cong(Maybe<&2,Bands.Geometry>,Bands.Geometry,BandCount.value,Some{geometry},Some{expected},
        Equal.trans(Maybe<&2,Bands.Geometry>,Some{geometry},BandCount.decode(shape),Some{expected},Equal.sym(Maybe<&2,Bands.Geometry>,BandCount.decode(shape),Some{geometry},BandCount.decoded(shape,geometry,coherent)),decoded))
      Shape.Both{
        %same_geometry : BandShape.WellFormed<Element,_,Model.contents(~Element,Model.pack(~Element,bands))> well_formed,
        %same_geometry : BandValues.Values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),_,Model.contents(~Element,Model.pack(~Element,bands))) values}
    case BandConcatExecution.BandedOwner{bands,geometry,shape,fill,Tensors.Failed{problem},workers}:
      Empty.absurd(SourceContract(~Element,~band_value,images,height,width,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),WordEncoding.false_true(ready))

def incoming_source_contracts(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,sources: BandConcatExecution.PartsOwnerModel<Element>,+images: U32,+height: U32,+width: U32,
  geometries: SourceGeometries(~Element,images,height,width,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),
  statuses: SourceStatuses(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),
  normals: IncomingNormals(~Element,sources),values: IncomingPartsValues(~Element,~band_value,sources)) ->
  SourceContracts(~Element,~band_value,images,height,width,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))):
  match sources:
    case BandConcatExecution.SingleOwner{input}: incoming_source_contract(~Element,~band_value,input,images,height,width,geometries,statuses,normals,values)
    case BandConcatExecution.ConsOwner{+input,+rest}:
      Shape.Both{head_geometry,tail_geometries} = geometries
      Shape.Both{head_status,tail_statuses} = statuses
      Shape.Both{head_normal,tail_normals} = normals
      Shape.Both{head_values,tail_values} = values
      Shape.Both{incoming_source_contract(~Element,~band_value,input,images,height,width,head_geometry,head_status,head_normal,head_values),
        incoming_source_contracts(~Element,~band_value,rest,images,height,width,tail_geometries,tail_statuses,tail_normals,tail_values)}

def output_guard(+images: U32,+channels: U32,+height: U32,+width: U32) ->
  {Window.band_shape([images,channels,height,width],Bands.Geometry{images,channels,height,width}) == True{} : Bool}:
  Boolean.all_hold([U32.is_eq(width,width),U32.is_eq(height,height),U32.is_eq(channels,channels),U32.is_eq(images,images)],
    Shape.Both{BandShape.equal_words(width,width,{==}),Shape.Both{BandShape.equal_words(height,height,{==}),
      Shape.Both{BandShape.equal_words(channels,channels,{==}),Shape.Both{BandShape.equal_words(images,images,{==}),Unit{}}}}})

def output_guard_target(+rank: BandConcatCaller.Rank,+first_channels: U32,+height: U32,+width: U32,summary: Layout.Survey) ->
  {Window.band_shape(BandConcatCaller.target_shape(BandConcatCaller.shape_of(rank,first_channels,height,width),BandConcatCaller.axis_of(rank),summary),Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(summary),height,width}) == True{} : Bool}:
  match rank:
    case BandConcatCaller.RankThree{}:
      Layout.Survey{valid,channels,status,workers} = summary
      output_guard(1,channels,height,width)
    case BandConcatCaller.RankFour{+images}:
      Layout.Survey{valid,channels,status,workers} = summary
      output_guard(images,channels,height,width)

# This predicate is Global.Normal after the affine tensor has been observed.
# It retains both storage validity and the actual ready shape/count evidence.
def ContentNormal(~Element: Data,input: BandConcatExecution.TensorContent<Element>) -> Data:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}:
      Shape.Both<Buffer.ContentReady<Element,buffer>,Global.DenseShape(~Element,status,shape,buffer)>
    case BandConcatExecution.BandedContent{bands,geometry,shape,fill,status,workers}:
      Shape.Both<BandShape.WellFormed<Element,geometry,bands>,Global.BandShape(status,shape,geometry)>

def normal_contents(~Element: Data,input: Tensors.Tensor<Element>,evidence: ContentNormal(~Element,BandConcatExecution.tensor_contents(~Element,input))) -> Global.Normal(~Element,input):
  match input:
    case Tensors.Tensor{buffer,shape,fill,status,workers}: evidence
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: evidence

def ContentContract(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+parts: BandConcatExecution.PartContents<Element>,+height: U32,+width: U32,+output: BandConcatExecution.TensorContent<Element>) -> Data:
  Shape.Both<ContentNormal(~Element,output),
    TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{parts,height,width},output)>

def NativeContract(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+parts: BandConcatExecution.PartContents<Element>,+height: U32,+width: U32,-output: Tensors.Tensor<Element>) -> Data:
  Shape.Both<Global.Normal(~Element,output),
    TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{parts,height,width},BandConcatExecution.tensor_contents(~Element,output))>

def native_contract(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+parts: BandConcatExecution.PartContents<Element>,+height: U32,+width: U32,output: Tensors.Tensor<Element>,
  receipt: ContentContract(~Element,~band_value,parts,height,width,BandConcatExecution.tensor_contents(~Element,output))) -> NativeContract(~Element,~band_value,parts,height,width,output):
  Shape.Both{normal,values} = receipt
  Shape.Both{normal_contents(~Element,output,normal),values}

def joined_expected_normal(~Element: Data,+first: List<&2,U32>,+axis: U32,+fill: Element,+geometry: Bands.Geometry,+layout: Bands.Plan,owners: BandConcatExecution.PartContents<Element> & Layout.Survey,
  allocation: BandShape.WellFormed<Element,geometry,BandConcatExecution.allocation_model(~Element,geometry,fill,layout)>,
  counted: Global.Counted<BandConcatCaller.target_shape(first,axis,BandConcatCaller.summary_projection(~Element,owners))>,
  coherent: {Window.band_shape(BandConcatCaller.target_shape(first,axis,BandConcatCaller.summary_projection(~Element,owners)),geometry) == True{} : Bool}) ->
  ContentNormal(~Element,BandConcatCaller.joined_expected(~Element,first,axis,fill,geometry,layout,owners)):
  (+parts,summary) = owners
  Layout.Survey{valid,+total,status,workers} = summary
  Shape.Both{BandConcatExecution.parts_well_formed(~Element,parts,geometry,geometry,(BandConcatExecution.allocation_model(~Element,geometry,fill,layout),0),allocation),
    Shape.Both{counted,coherent}}

def concat_first_normal_packed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+sources: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: U32,+fill: Element,geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,+limits: Machine.Limits,+capacity: U32,
  guards: BandConcatCaller.BandSurveyFields<Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,axis,BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,U32.to_nat(axis),BandConcatCaller.initial_summary(first,U32.to_nat(axis))),geometry,count,layout>,
  +contracts: SourceContracts(~Element,~band_value,geometry_images(geometry),geometry_height(geometry),geometry_width(geometry),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),
  +domain: BandGather.Domain<geometry,limits,capacity>,
  coherent: {Window.band_shape(BandConcatCaller.target_shape(first,axis,BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,U32.to_nat(axis),BandConcatCaller.initial_summary(first,U32.to_nat(axis)))),geometry) == True{} : Bool}) ->
  ContentNormal(~Element,BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,BandConcatExecution.parts_owner_pack(~Element,sources),first,fill))):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandConcatCaller.BandSurveyFields{+accepted,+status_guard,+geometry_guard,+count_guard,+selected,+compatible} = guards
  +parts = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))
  +summary = BandConcatCaller.survey_model(~Element,parts,first,U32.to_nat(axis),BandConcatCaller.initial_summary(first,U32.to_nat(axis)))
  +fields = discovered_fields(~Element,~band_value,parts,images,channels,height,width,limits,capacity,layout,contracts,domain,selected)
  +fitting = layout_fitting(layout,images,channels,height,width,limits,capacity,fields,domain)
  +coverage = layout_coverage(layout,images,height,fields)
  %Equal.sym(BandConcatExecution.TensorContent<Element>,BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,BandConcatExecution.parts_owner_pack(~Element,sources),first,fill)),BandConcatCaller.joined_expected(~Element,first,axis,fill,Bands.Geometry{images,channels,height,width},layout,(parts,summary)),
    BandConcatCaller.concat_first_band_packed(~Element,sources,first,axis,fill,Bands.Geometry{images,channels,height,width},count,layout,BandConcatCaller.BandSurveyFields{accepted,status_guard,geometry_guard,count_guard,selected,compatible})) : ContentNormal(~Element,_)
  joined_expected_normal(~Element,first,axis,fill,Bands.Geometry{images,channels,height,width},layout,(parts,summary),
    BandConcatCaller.allocation_well_formed(~Element,layout,Bands.Geometry{images,channels,height,width},fill,capacity,Machine.half_word(limits),fitting,coverage),Global.Counted{count,count_guard},coherent)

def not_zero_shape(empty: Bool,nonempty: {Bool.not(empty) == True{} : Bool}) -> {empty == False{} : Bool}:
  match empty:
    case False{}: {==}
    case True{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(nonempty))

type PositiveSpatialChannels<-channels: U32,-height: U32,-width: U32> is Data:
  PositiveSpatialChannels{channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
    height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool}}

def positive_spatial_channels(+channels: U32,+height: U32,+width: U32,+nonempty: {Shapes.has_zero([channels,height,width]) == False{} : Bool}) -> PositiveSpatialChannels<channels,height,width>:
  +spatial = Boolean.or_false_right(U32.is_zero(channels),Shapes.has_zero([height,width]),nonempty)
  +last = Boolean.or_false_right(U32.is_zero(height),Shapes.has_zero([width]),spatial)
  PositiveSpatialChannels{
    CountCompletion.positive_word(channels,Boolean.or_false_left(U32.is_zero(channels),Shapes.has_zero([height,width]),nonempty)),
    CountCompletion.positive_word(height,Boolean.or_false_left(U32.is_zero(height),Shapes.has_zero([width]),spatial)),
    CountCompletion.positive_word(width,Boolean.or_false_left(U32.is_zero(width),Shapes.has_zero([]),last))}

def counted_spatial_domain(+shape: List<&2,U32>,geometry: Bands.Geometry,counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  positive: PositiveSpatialChannels<geometry_channels(geometry),geometry_height(geometry),geometry_width(geometry)>) -> BandGather.Domain<geometry,Default.limits(),Machine.capacity(Default.limits())>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  PositiveSpatialChannels{channels_positive,height_positive,width_positive} = positive
  Global.domain(shape,Bands.Geometry{images,channels,height,width},counted,coherent,channels_positive,height_positive,width_positive)

def counted_output_domain(+rank: BandConcatCaller.Rank,+first_channels: U32,+height: U32,+width: U32,summary: Layout.Survey,+count: U32,+compatible: Bool,
  counted: {Tensors.count_of(BandConcatCaller.target_shape(BandConcatCaller.shape_of(rank,first_channels,height,width),BandConcatCaller.axis_of(rank),summary)) == Some{count} : Maybe<&2,U32>},
  enabled: {compatible && Bool.not(Shapes.has_zero(BandConcatCaller.target_shape(BandConcatCaller.shape_of(rank,first_channels,height,width),BandConcatCaller.axis_of(rank),summary))) == True{} : Bool}) ->
  BandGather.Domain<Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(summary),height,width},Default.limits(),Machine.capacity(Default.limits())>:
  match rank:
    case BandConcatCaller.RankThree{}:
      Layout.Survey{valid,+channels,status,workers} = summary
      counted_spatial_domain([channels,height,width],Bands.Geometry{1,channels,height,width},Global.Counted{count,counted},output_guard(1,channels,height,width),
        positive_spatial_channels(channels,height,width,not_zero_shape(Shapes.has_zero([channels,height,width]),Boolean.right(compatible,Bool.not(Shapes.has_zero([channels,height,width])),enabled))))
    case BandConcatCaller.RankFour{+images}:
      Layout.Survey{valid,+channels,status,workers} = summary
      counted_spatial_domain([images,channels,height,width],Bands.Geometry{images,channels,height,width},Global.Counted{count,counted},output_guard(images,channels,height,width),
        positive_spatial_channels(channels,height,width,Boolean.or_false_right(U32.is_zero(images),Shapes.has_zero([channels,height,width]),
          not_zero_shape(Shapes.has_zero([images,channels,height,width]),Boolean.right(compatible,Bool.not(Shapes.has_zero([images,channels,height,width])),enabled)))))

def concat_first_incoming(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+sources: BandConcatExecution.PartsOwnerModel<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+fill: Element,+count: U32,+layout: Bands.Plan,+limit: U32,+maximum: {limit == 4294967295 : U32},
  guards: BandConcatCaller.BandSurveyFields<Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)))),Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))),height,width},count,layout>,
  normals: IncomingNormals(~Element,sources),values: IncomingPartsValues(~Element,~band_value,sources)) ->
  Shape.Both<ContentNormal(~Element,BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,BandConcatCaller.axis_of(rank),BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.shape_of(rank,channels,height,width),fill))),
    TensorValues(~Element,~LogicalConcat<Element>,~global_concat_value(~Element,~tensor_original(~Element,~band_value)),LogicalConcat{BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width},BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,BandConcatCaller.axis_of(rank),BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.shape_of(rank,channels,height,width),fill)))>:
  BandConcatCaller.BandSurveyFields{+accepted,+status,+geometry,+counted,+selected,+compatible} = guards
  +parts = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))
  +first = {BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>}
  +initial = BandConcatCaller.initial_summary(first,U32.to_nat(BandConcatCaller.axis_of(rank)))
  +summary = BandConcatCaller.survey_model(~Element,parts,first,U32.to_nat(BandConcatCaller.axis_of(rank)),initial)
  +statuses = surveyed_statuses(~Element,parts,first,U32.to_nat(BandConcatCaller.axis_of(rank)),initial,
    %Equal.sym(Tensors.Status,BandConcatCaller.status_of(summary),Tensors.Ready{},status) : {status_ready(_) == True{} : Bool} {==})
  +contracts = incoming_source_contracts(~Element,~band_value,sources,BandConcatCaller.images_of(rank),height,width,
    surveyed_geometries(~Element,parts,rank,channels,height,width,initial,limit,BandConcatCaller.prefixes(~Element,parts,first,U32.to_nat(BandConcatCaller.axis_of(rank)),initial,limit,maximum,accepted)),statuses,normals,values)
  +domain = counted_output_domain(rank,channels,height,width,summary,count,BandConcatCaller.compatible_flag(~Element,BandConcatCaller.compatible_model(~Element,parts)),counted,compatible)
  +guards = {BandConcatCaller.BandSurveyFields{accepted,status,geometry,counted,selected,compatible} : BandConcatCaller.BandSurveyFields<Element,parts,first,BandConcatCaller.axis_of(rank),summary,Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(summary),height,width},count,layout>}
  Shape.Both{concat_first_normal_packed(~Element,~band_value,sources,first,BandConcatCaller.axis_of(rank),fill,Bands.Geometry{BandConcatCaller.images_of(rank),BandConcatCaller.survey_total(summary),height,width},count,layout,Default.limits(),Machine.capacity(Default.limits()),guards,contracts,domain,output_guard_target(rank,channels,height,width,summary)),
    concat_first_values_packed(~Element,~band_value,sources,rank,channels,height,width,fill,count,layout,Default.limits(),Machine.capacity(Default.limits()),limit,maximum,guards,contracts,domain)}

def concat_from_first_contract(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+sources: BandConcatExecution.PartsOwnerModel<Element>,+shape: List<&2,U32>,+axis: U32,+height: U32,+width: U32,
  +shape_selected: {first_shape(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))) == shape : List<&2,U32>},
  receipt: ContentContract(~Element,~band_value,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width,
    BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,BandConcatExecution.parts_owner_pack(~Element,sources),shape,first_fill(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))))))) ->
  NativeContract(~Element,~band_value,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width,Layout.concat(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),axis)):
  native_contract(~Element,~band_value,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width,Layout.concat(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),axis),
    %Equal.sym(BandConcatExecution.TensorContent<Element>,BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),axis)),
      BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,BandConcatExecution.parts_owner_pack(~Element,sources),shape,first_fill(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))))),
      native_concat_shape(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),axis,shape,shape_selected)) :
      ContentContract(~Element,~band_value,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width,_) receipt)

def concat_incoming_packed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+sources: BandConcatExecution.PartsOwnerModel<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,+count: U32,+layout: Bands.Plan,+limit: U32,+maximum: {limit == 4294967295 : U32},
  +first: {first_shape(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))) == BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>},
  guards: NativeBandFields<Element,BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),count,layout>,
  normals: IncomingNormals(~Element,sources),values: IncomingPartsValues(~Element,~band_value,sources)) ->
  NativeContract(~Element,~band_value,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width,Layout.concat(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.axis_of(rank))):
  concat_from_first_contract(~Element,~band_value,sources,BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),height,width,first,
    concat_first_incoming(~Element,~band_value,sources,rank,channels,height,width,first_fill(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),count,layout,limit,maximum,
      native_fields_packed(~Element,sources,rank,channels,height,width,count,layout,guards),normals,values))

# One metadata classifier supplies every actual successful band receipt.
# The other constructor proves that this same band predicate is false.
type ConcatSelectionInputs is Data:
  ConcatSelectionInputs{status: Tensors.Status,valid: Bool,geometry: Maybe<&2,Bands.Geometry>,counted: Maybe<&2,U32>,layout: Maybe<&2,Bands.Plan>,compatible: Bool}

def selection_inputs(~Element: Data,+parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: U32,+summary: Layout.Survey) -> ConcatSelectionInputs:
  ConcatSelectionInputs{BandConcatCaller.status_of(summary),BandConcatCaller.survey_valid(summary),BandConcat.geometry(BandConcatCaller.target_shape(first,axis,summary),axis),
    Tensors.count_of(BandConcatCaller.target_shape(first,axis,summary)),BandConcatCaller.discovery_model(~Element,parts),
    BandConcatCaller.compatible_flag(~Element,BandConcatCaller.compatible_model(~Element,parts)) && Bool.not(Shapes.has_zero(BandConcatCaller.target_shape(first,axis,summary)))}

def selection_status(inputs: ConcatSelectionInputs) -> Tensors.Status:
  ConcatSelectionInputs{status,valid,geometry,counted,layout,compatible} = inputs
  status

def selection_valid(inputs: ConcatSelectionInputs) -> Bool:
  ConcatSelectionInputs{status,valid,geometry,counted,layout,compatible} = inputs
  valid

def selection_geometry(inputs: ConcatSelectionInputs) -> Maybe<&2,Bands.Geometry>:
  ConcatSelectionInputs{status,valid,geometry,counted,layout,compatible} = inputs
  geometry

def selection_count(inputs: ConcatSelectionInputs) -> Maybe<&2,U32>:
  ConcatSelectionInputs{status,valid,geometry,counted,layout,compatible} = inputs
  counted

def selection_layout(inputs: ConcatSelectionInputs) -> Maybe<&2,Bands.Plan>:
  ConcatSelectionInputs{status,valid,geometry,counted,layout,compatible} = inputs
  layout

def selection_compatible(inputs: ConcatSelectionInputs) -> Bool:
  ConcatSelectionInputs{status,valid,geometry,counted,layout,compatible} = inputs
  compatible

def selection_is_band(inputs: ConcatSelectionInputs) -> Bool:
  ConcatSelectionInputs{status,valid,geometry,counted,layout,compatible} = inputs
  match status:
    case Tensors.Failed{problem}: False{}
    case Tensors.Ready{}:
      match valid:
        case False{}: False{}
        case True{}:
          match geometry:
            case None{}: False{}
            case Some{geometry}:
              match counted:
                case None{}: False{}
                case Some{count}:
                  match layout:
                    case None{}: False{}
                    case Some{layout}: compatible

type ConcatSelection<-Element: Data,-parts: BandConcatExecution.PartContents<Element>,-first: List<&2,U32>,-axis: U32,-summary: Layout.Survey> is Data:
  ConcatBandSelected{geometry: Bands.Geometry,count: U32,layout: Bands.Plan,
    enabled: {selection_is_band(selection_inputs(~Element,parts,first,axis,summary)) == True{} : Bool},
    fields: BandConcatCaller.BandSurveyFields<Element,parts,first,axis,summary,geometry,count,layout>}
  ConcatDenseSelected{disabled: {selection_is_band(selection_inputs(~Element,parts,first,axis,summary)) == False{} : Bool}}

def selection_disabled_received(~Element: Data,+parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: U32,+summary: Layout.Survey,+inputs: ConcatSelectionInputs,
  actual: {selection_inputs(~Element,parts,first,axis,summary) == inputs : ConcatSelectionInputs},
  disabled: {selection_is_band(inputs) == False{} : Bool}) -> ConcatSelection<Element,parts,first,axis,summary>:
  ConcatDenseSelected{Equal.trans(Bool,selection_is_band(selection_inputs(~Element,parts,first,axis,summary)),selection_is_band(inputs),False{},
    Equal.cong(ConcatSelectionInputs,Bool,selection_is_band,selection_inputs(~Element,parts,first,axis,summary),inputs,actual),disabled)}

def selection_received(~Element: Data,+parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: U32,+summary: Layout.Survey,inputs: ConcatSelectionInputs,
  +actual: {selection_inputs(~Element,parts,first,axis,summary) == inputs : ConcatSelectionInputs}) -> ConcatSelection<Element,parts,first,axis,summary>:
  ConcatSelectionInputs{status,+valid,+geometry,+counted,+layout,compatible} = inputs
  match status:
    case Tensors.Failed{+problem}:
      selection_disabled_received(~Element,parts,first,axis,summary,ConcatSelectionInputs{Tensors.Failed{problem},valid,geometry,counted,layout,compatible},actual,{==})
    case Tensors.Ready{}:
      match valid:
        case False{}:
          selection_disabled_received(~Element,parts,first,axis,summary,ConcatSelectionInputs{Tensors.Ready{},False{},geometry,counted,layout,compatible},actual,{==})
        case True{}:
          match geometry:
            case None{}:
              selection_disabled_received(~Element,parts,first,axis,summary,ConcatSelectionInputs{Tensors.Ready{},True{},None{},counted,layout,compatible},actual,{==})
            case Some{+geometry}:
              match counted:
                case None{}:
                  selection_disabled_received(~Element,parts,first,axis,summary,ConcatSelectionInputs{Tensors.Ready{},True{},Some{geometry},None{},layout,compatible},actual,{==})
                case Some{+count}:
                  match layout:
                    case None{}:
                      selection_disabled_received(~Element,parts,first,axis,summary,ConcatSelectionInputs{Tensors.Ready{},True{},Some{geometry},Some{count},None{},compatible},actual,{==})
                    case Some{+layout}:
                      match compatible:
                        case False{}:
                          selection_disabled_received(~Element,parts,first,axis,summary,ConcatSelectionInputs{Tensors.Ready{},True{},Some{geometry},Some{count},Some{layout},False{}},actual,{==})
                        case True{}:
                          +observed = selection_inputs(~Element,parts,first,axis,summary)
                          +selected = {ConcatSelectionInputs{Tensors.Ready{},True{},Some{geometry},Some{count},Some{layout},True{}} : ConcatSelectionInputs}
                          ConcatBandSelected{geometry,count,layout,Equal.cong(ConcatSelectionInputs,Bool,selection_is_band,observed,selected,actual),BandConcatCaller.BandSurveyFields{
                            Equal.cong(ConcatSelectionInputs,Bool,selection_valid,observed,selected,actual),
                            Equal.cong(ConcatSelectionInputs,Tensors.Status,selection_status,observed,selected,actual),
                            Equal.cong(ConcatSelectionInputs,Maybe<&2,Bands.Geometry>,selection_geometry,observed,selected,actual),
                            Equal.cong(ConcatSelectionInputs,Maybe<&2,U32>,selection_count,observed,selected,actual),
                            Equal.cong(ConcatSelectionInputs,Maybe<&2,Bands.Plan>,selection_layout,observed,selected,actual),
                            Equal.cong(ConcatSelectionInputs,Bool,selection_compatible,observed,selected,actual)}}

def concat_selection(~Element: Data,+sources: BandConcatExecution.PartsOwnerModel<Element>,+axis: U32) ->
  ConcatSelection<Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),
    first_shape(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),axis,
    BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first_shape(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),U32.to_nat(axis),
      BandConcatCaller.initial_summary(first_shape(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),U32.to_nat(axis)))>:
  +parts = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))
  +first = first_shape(~Element,parts)
  +summary = BandConcatCaller.survey_model(~Element,parts,first,U32.to_nat(axis),BandConcatCaller.initial_summary(first,U32.to_nat(axis)))
  selection_received(~Element,parts,first,axis,summary,selection_inputs(~Element,parts,first,axis,summary),{==})

# Every non-band ready route enters this actual dense selector. This equation
# needs no Normal, positive shape, layout pairing or downstream success guard.
def discovered_native_parts(~Element: Data,discovered: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) -> Tensors.Parts<Element>:
  (parts,layout) = discovered
  parts

def concat_dense_selected(~Element: Data,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,discovered: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) ->
  {Layout.join_compatible_selected(~Element,False{},shape,axis,fill,workers,discovered) ==
    Layout.append_parts(~Element,discovered_native_parts(~Element,discovered),axis,(Tensors.with_workers(~Element,Tensors.full(~Element,shape,fill),workers),0)) : Tensors.Tensor<Element>}:
  (parts,layout) = discovered
  {==}

# The selector produces the successful contract itself. A disabled branch
# retains its exact predicate; it does not claim dense output values.
type SelectedContract<-Contract: Data,-enabled: Bool> is Data:
  BandChecked{selected: {enabled == True{} : Bool},contract: Contract}
  DenseSelected{disabled: {enabled == False{} : Bool}}

def selected_received(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+sources: BandConcatExecution.PartsOwnerModel<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,
  +first: {first_shape(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))) == BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>},
  selection: ConcatSelection<Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),
    BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))>,
  normals: IncomingNormals(~Element,sources),values: IncomingPartsValues(~Element,~band_value,sources)) ->
  SelectedContract<NativeContract(~Element,~band_value,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width,Layout.concat(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.axis_of(rank))),
    selection_is_band(selection_inputs(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),
      BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))))>:
  match selection:
    case ConcatBandSelected{geometry,+count,+layout,enabled,fields}:
      BandConcatCaller.BandSurveyFields{valid,status,geometry_equal,counted,selected,compatible} = fields
      +parts = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))
      +shape = {BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>}
      +summary = BandConcatCaller.survey_model(~Element,parts,shape,U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(shape,U32.to_nat(BandConcatCaller.axis_of(rank))))
      BandChecked{enabled,concat_from_first_contract(~Element,~band_value,sources,shape,BandConcatCaller.axis_of(rank),height,width,first,
        concat_first_incoming(~Element,~band_value,sources,rank,channels,height,width,first_fill(~Element,parts),count,layout,4294967295,{==},
          survey_fields(~Element,parts,rank,channels,height,width,summary,count,layout,valid,status,counted,selected,compatible),normals,values))}
    case ConcatDenseSelected{disabled}: DenseSelected{disabled}

def concat_selected(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+sources: BandConcatExecution.PartsOwnerModel<Element>,+rank: BandConcatCaller.Rank,+channels: U32,+height: U32,+width: U32,
  +first: {first_shape(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))) == BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>},
  normals: IncomingNormals(~Element,sources),values: IncomingPartsValues(~Element,~band_value,sources)) ->
  SelectedContract<NativeContract(~Element,~band_value,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),height,width,Layout.concat(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),BandConcatCaller.axis_of(rank))),
    selection_is_band(selection_inputs(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),BandConcatCaller.axis_of(rank),
      BandConcatCaller.survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(BandConcatCaller.shape_of(rank,channels,height,width),U32.to_nat(BandConcatCaller.axis_of(rank))))))>:
  +parts = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))
  +shape = {BandConcatCaller.shape_of(rank,channels,height,width) : List<&2,U32>}
  +summary = BandConcatCaller.survey_model(~Element,parts,shape,U32.to_nat(BandConcatCaller.axis_of(rank)),BandConcatCaller.initial_summary(shape,U32.to_nat(BandConcatCaller.axis_of(rank))))
  selected_received(~Element,~band_value,sources,rank,channels,height,width,first,
    selection_received(~Element,parts,shape,BandConcatCaller.axis_of(rank),summary,selection_inputs(~Element,parts,shape,BandConcatCaller.axis_of(rank),summary),{==}),normals,values)
