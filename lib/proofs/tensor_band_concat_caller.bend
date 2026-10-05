# The public concat caller: survey of the parts, readiness of the final owner
# and its layout.
import Base
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_band_range.bend as BandRange
import ../tensor_band.bend as Bands
import ../tensor_band_concat.bend as BandConcat
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_model.bend as Model
import ./write_program.bend as Writes
import ../tensor.bend as Tensors
import ./nat_order.bend as Order
import ./storage_copy_refinement.bend as StorageCopyLaws
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_shape_refinement.bend as BandShape
import ./u32_full_range.bend as FullRange
import ../tensor_layout.bend as Layout
import ../tensor_shape.bend as Shapes
import ../machine_limits.bend as Machine
import ./boolean_logic.bend as Boolean
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./contiguous_copy.bend as Contiguous
import ./encoded_coordinates.bend as Encoded
import ./machine_limit_bounds.bend as Limits
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./storage_tree_model.bend as Tree
import ./tensor_band_count.bend as BandCount
import ./tensor_band_dense.bend as BandDense
import ./tensor_shape_refinement.bend as ShapeLaws
import ./word_encoding.bend as WordEncoding

# The public survey visits Parts in the same order as band concatenation.
# Its successful overflow and shape guards produce unbounded prefix totals;
# the modular concat cursor is not used as a substitute for those facts.
def survey_valid(summary: Layout.Survey) -> Bool:
  Layout.Survey{survey_valid,total,status,workers} = summary
  survey_valid

def survey_total(summary: Layout.Survey) -> U32:
  Layout.Survey{survey_valid,total,status,workers} = summary
  total

def tensor_status(~Element: Data,input: BandConcatExecution.TensorContent<Element>) -> Tensors.Status:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}: status
    case BandConcatExecution.BandedContent{bands,geometry,shape,fill,status,workers}: status

def tensor_workers(~Element: Data,input: BandConcatExecution.TensorContent<Element>) -> U32:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}: workers
    case BandConcatExecution.BandedContent{bands,geometry,shape,fill,status,workers}: workers

def step(~Element: Data,+first: List<&2,U32>,+axis: Nat,+input: BandConcatExecution.TensorContent<Element>,summary: Layout.Survey) -> Layout.Survey:
  Layout.surveyed(summary,first,BandConcatExecution.tensor_shape(~Element,input),axis,tensor_status(~Element,input),tensor_workers(~Element,input))

def step_total(~Element: Data,+first: List<&2,U32>,+axis: Nat,input: BandConcatExecution.TensorContent<Element>,summary: Layout.Survey) ->
  {survey_total(step(~Element,first,axis,input,summary)) == (survey_total(summary) + Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) : U32) : U32}:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}:
      Layout.Survey{survey_valid,total,previous_status,previous_workers} = summary
      {==}
    case BandConcatExecution.BandedContent{bands,geometry,shape,fill,status,workers}:
      Layout.Survey{survey_valid,total,previous_status,previous_workers} = summary
      {==}

def survey_model(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,summary: Layout.Survey) -> Layout.Survey:
  match parts:
    case BandConcatExecution.SingleContent{input}: step(~Element,first,axis,input,summary)
    case BandConcatExecution.ConsContent{input,rest}: survey_model(~Element,rest,first,axis,step(~Element,first,axis,input,summary))

def extent_total(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+axis: Nat) -> Nat:
  match parts:
    case BandConcatExecution.SingleContent{input}: U32.to_nat(Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis))
    case BandConcatExecution.ConsContent{input,rest}: Nat.add(U32.to_nat(Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis)),extent_total(~Element,rest,axis))

def conditions(+first: List<&2,U32>,+shape: List<&2,U32>,+axis: Nat,summary: Layout.Survey) -> List<&2,Bool>:
  Layout.Survey{previous,+total,status,workers} = summary
  [U32.is_le(Layout.extent_at(shape,axis),(4294967295 - total : U32)),
    Shapes.same_extents(List.drop(&2,U32,first,1n+axis),List.drop(&2,U32,shape,1n+axis)),
    Shapes.same_extents(List.take(&2,U32,first,axis),List.take(&2,U32,shape,axis)),
    Nat.is_eq(List.length(&2,U32,first),List.length(&2,U32,shape)),previous]

def step_conditions(~Element: Data,+first: List<&2,U32>,+axis: Nat,input: BandConcatExecution.TensorContent<Element>,summary: Layout.Survey) ->
  {survey_valid(step(~Element,first,axis,input,summary)) == Boolean.all(conditions(first,BandConcatExecution.tensor_shape(~Element,input),axis,summary)) : Bool}:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}:
      Layout.Survey{previous,total,old_status,old_workers} = summary
      {==}
    case BandConcatExecution.BandedContent{bands,geometry,shape,fill,status,workers}:
      Layout.Survey{previous,total,old_status,old_workers} = summary
      {==}

def step_previous(~Element: Data,+first: List<&2,U32>,+axis: Nat,+input: BandConcatExecution.TensorContent<Element>,+summary: Layout.Survey,
  accepted: {survey_valid(step(~Element,first,axis,input,summary)) == True{} : Bool}) -> {survey_valid(summary) == True{} : Bool}:
  Layout.Survey{+previous,+total,+status,+workers} = summary
  Boolean.every_condition_holds(conditions(first,BandConcatExecution.tensor_shape(~Element,input),axis,summary),4n,
    %step_conditions(~Element,first,axis,input,summary) : {_ == True{} : Bool} accepted)

def previous_valid(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,
  accepted: {survey_valid(survey_model(~Element,parts,first,axis,summary)) == True{} : Bool}) -> {survey_valid(summary) == True{} : Bool}:
  match parts:
    case BandConcatExecution.SingleContent{input}: step_previous(~Element,first,axis,input,summary,accepted)
    case BandConcatExecution.ConsContent{+input,+rest}:
      step_previous(~Element,first,axis,input,summary,previous_valid(~Element,rest,first,axis,step(~Element,first,axis,input,summary),accepted))

# Abstract limit keeps the general addition proof independent of a concrete
# enormous unary Nat. The actual public guard is transported by maximum.
def sum_room(+previous: U32,+extent: U32,+limit: U32,
  previous_room: {U32.is_le(previous,limit) == True{} : Bool},
  accepted: {U32.is_le(extent,(limit - previous : U32)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(extent)),U32.to_nat(limit)) == True{} : Bool}:
  +ordered = Bounds.as_nat_le(previous,limit,previous_room)
  +difference = FullRange.subtract(limit,previous,limit,ordered,Order.sub_le(U32.to_nat(limit),U32.to_nat(previous)))
  +extent_room = {
    %difference : {Nat.is_le(U32.to_nat(extent),_) == True{} : Bool}
    Bounds.as_nat_le(extent,(limit - previous : U32),accepted) : {Nat.is_le(U32.to_nat(extent),Nat.sub(U32.to_nat(limit),U32.to_nat(previous))) == True{} : Bool}}
  %Order.restore_shift(U32.to_nat(previous),U32.to_nat(limit),ordered) : {Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(extent)),_) == True{} : Bool}
  Order.offset_le(U32.to_nat(previous),U32.to_nat(extent),Nat.sub(U32.to_nat(limit),U32.to_nat(previous)),extent_room)

type StepFields<-first: List<&2,U32>,-shape: List<&2,U32>,-axis: Nat,-previous: U32,-limit: U32> is Data:
  StepFields{rank: {Nat.is_eq(List.length(&2,U32,first),List.length(&2,U32,shape)) == True{} : Bool},
    prefix: {Shapes.same_extents(List.take(&2,U32,first,axis),List.take(&2,U32,shape,axis)) == True{} : Bool},
    suffix: {Shapes.same_extents(List.drop(&2,U32,first,1n+axis),List.drop(&2,U32,shape,1n+axis)) == True{} : Bool},
    sum_room: {Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(Layout.extent_at(shape,axis))),U32.to_nat(limit)) == True{} : Bool},
    exact: {U32.to_nat((previous + Layout.extent_at(shape,axis) : U32)) == Nat.add(U32.to_nat(previous),U32.to_nat(Layout.extent_at(shape,axis))) : Nat}}

def step_fields(~Element: Data,+first: List<&2,U32>,+axis: Nat,+input: BandConcatExecution.TensorContent<Element>,+summary: Layout.Survey,+limit: U32,
  +maximum: {limit == 4294967295 : U32},accepted: {survey_valid(step(~Element,first,axis,input,summary)) == True{} : Bool}) ->
  StepFields<first,BandConcatExecution.tensor_shape(~Element,input),axis,survey_total(summary),limit>:
  Layout.Survey{+previous,+total,+status,+workers} = summary
  +shape = BandConcatExecution.tensor_shape(~Element,input)
  +holds = {
    %step_conditions(~Element,first,axis,input,summary) : {_ == True{} : Bool}
    accepted : {Boolean.all(conditions(first,shape,axis,summary)) == True{} : Bool}}
  +guard = Boolean.every_condition_holds(conditions(first,shape,axis,summary),0n,holds)
  +room = sum_room(total,Layout.extent_at(shape,axis),limit,
    %Equal.sym(U32,limit,4294967295,maximum) : {U32.is_le(total,_) == True{} : Bool} FullRange.maximum(total),
    %Equal.sym(U32,limit,4294967295,maximum) : {U32.is_le(Layout.extent_at(shape,axis),(_ - total : U32)) == True{} : Bool} guard)
  StepFields{Boolean.every_condition_holds(conditions(first,shape,axis,summary),3n,holds),Boolean.every_condition_holds(conditions(first,shape,axis,summary),2n,holds),
    Boolean.every_condition_holds(conditions(first,shape,axis,summary),1n,holds),room,FullRange.add(total,Layout.extent_at(shape,axis),limit,room)}

def Prefixes(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32) -> Data:
  match parts:
    case BandConcatExecution.SingleContent{input}: StepFields<first,BandConcatExecution.tensor_shape(~Element,input),axis,survey_total(summary),limit>
    case BandConcatExecution.ConsContent{+input,rest}:
      Shape.Both<StepFields<first,BandConcatExecution.tensor_shape(~Element,input),axis,survey_total(summary),limit>,Prefixes(~Element,rest,first,axis,step(~Element,first,axis,input,summary),limit)>

def prefixes(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32,+maximum: {limit == 4294967295 : U32},
  +accepted: {survey_valid(survey_model(~Element,parts,first,axis,summary)) == True{} : Bool}) -> Prefixes(~Element,parts,first,axis,summary,limit):
  match parts:
    case BandConcatExecution.SingleContent{input}: step_fields(~Element,first,axis,input,summary,limit,maximum,accepted)
    case BandConcatExecution.ConsContent{+input,+rest}:
      Shape.Both{step_fields(~Element,first,axis,input,summary,limit,maximum,previous_valid(~Element,rest,first,axis,step(~Element,first,axis,input,summary),accepted)),
        prefixes(~Element,rest,first,axis,step(~Element,first,axis,input,summary),limit,maximum,accepted)}

type TotalFields<-value: U32,-previous: U32,-extent: Nat,-limit: U32> is Data:
  TotalFields{exact: {U32.to_nat(value) == Nat.add(U32.to_nat(previous),extent) : Nat},
    room: {Nat.is_le(Nat.add(U32.to_nat(previous),extent),U32.to_nat(limit)) == True{} : Bool}}

def total_cons(~Element: Data,+input: BandConcatExecution.TensorContent<Element>,+rest: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32,
  head: StepFields<first,BandConcatExecution.tensor_shape(~Element,input),axis,survey_total(summary),limit>,
  remainder: TotalFields<survey_total(survey_model(~Element,rest,first,axis,step(~Element,first,axis,input,summary))),survey_total(step(~Element,first,axis,input,summary)),extent_total(~Element,rest,axis),limit>) ->
  TotalFields<survey_total(survey_model(~Element,BandConcatExecution.ConsContent{input,rest},first,axis,summary)),survey_total(summary),extent_total(~Element,BandConcatExecution.ConsContent{input,rest},axis),limit>:
  Layout.Survey{+previous,+total,+status,+workers} = summary
  StepFields{rank,prefix,suffix,head_room,+head_exact} = head
  TotalFields{+tail_exact,+tail_room} = remainder
  +extent = U32.to_nat(Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis))
  +next_exact = Equal.trans(Nat,U32.to_nat(survey_total(step(~Element,first,axis,input,summary))),U32.to_nat((total + Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) : U32)),Nat.add(U32.to_nat(total),extent),
    Equal.cong(U32,Nat,value => U32.to_nat(value),survey_total(step(~Element,first,axis,input,summary)),(total + Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) : U32),step_total(~Element,first,axis,input,summary)),head_exact)
  TotalFields{
    Equal.trans(Nat,U32.to_nat(survey_total(survey_model(~Element,rest,first,axis,step(~Element,first,axis,input,summary)))),
      Nat.add(U32.to_nat(survey_total(step(~Element,first,axis,input,summary))),extent_total(~Element,rest,axis)),Nat.add(U32.to_nat(total),Nat.add(extent,extent_total(~Element,rest,axis))),tail_exact,
      Equal.trans(Nat,Nat.add(U32.to_nat(survey_total(step(~Element,first,axis,input,summary))),extent_total(~Element,rest,axis)),Nat.add(Nat.add(U32.to_nat(total),extent),extent_total(~Element,rest,axis)),Nat.add(U32.to_nat(total),Nat.add(extent,extent_total(~Element,rest,axis))),
        Equal.cong(Nat,Nat,value => Nat.add(value,extent_total(~Element,rest,axis)),U32.to_nat(survey_total(step(~Element,first,axis,input,summary))),Nat.add(U32.to_nat(total),extent),next_exact),Addition.associative(U32.to_nat(total),extent,extent_total(~Element,rest,axis)))),
    %Addition.associative(U32.to_nat(total),extent,extent_total(~Element,rest,axis)) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
    %next_exact : {Nat.is_le(Nat.add(_,extent_total(~Element,rest,axis)),U32.to_nat(limit)) == True{} : Bool}
    tail_room}

def total_fields(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32,
  fields: Prefixes(~Element,parts,first,axis,summary,limit)) -> TotalFields<survey_total(survey_model(~Element,parts,first,axis,summary)),survey_total(summary),extent_total(~Element,parts,axis),limit>:
  match parts:
    case BandConcatExecution.SingleContent{+input}:
      Layout.Survey{+previous,+total,+status,+workers} = summary
      StepFields{rank,prefix,suffix,room,exact} = fields
      TotalFields{
        Equal.trans(Nat,U32.to_nat(survey_total(step(~Element,first,axis,input,summary))),U32.to_nat((total + Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) : U32)),
          Nat.add(U32.to_nat(total),U32.to_nat(Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis))),
          Equal.cong(U32,Nat,value => U32.to_nat(value),survey_total(step(~Element,first,axis,input,summary)),(total + Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) : U32),step_total(~Element,first,axis,input,summary)),exact),room}
    case BandConcatExecution.ConsContent{+input,+rest}:
      Shape.Both{head,tail} = fields
      total_cons(~Element,input,rest,first,axis,summary,limit,head,total_fields(~Element,rest,first,axis,step(~Element,first,axis,input,summary),limit,tail))

def drop_zero(shape: List<&2,U32>) -> {List.drop(&2,U32,shape,0n) == shape : List<&2,U32>}:
  match shape:
    case []: {==}
    case head <> tail: {==}

def suffix_tail(+height: U32,+width: U32,+tail: List<&2,U32>,suffix: {Shapes.same_extents([height,width],List.drop(&2,U32,tail,0n)) == True{} : Bool}) ->
  {Shapes.same_extents([height,width],tail) == True{} : Bool}:
  %drop_zero(tail) : {Shapes.same_extents([height,width],_) == True{} : Bool}
  suffix

def geometry_three(+shape: List<&2,U32>,+height: U32,+width: U32,
  suffix: {Shapes.same_extents([height,width],List.drop(&2,U32,shape,1n)) == True{} : Bool}) ->
  {BandConcat.geometry(shape,0) == Some{Bands.Geometry{1,BandConcat.channels(shape),height,width}} : Maybe<&2,Bands.Geometry>}:
  match shape:
    case []: Empty.absurd({None{} == Some{Bands.Geometry{1,0,height,width}} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(suffix))
    case +channels <> +tail:
      %ShapeLaws.same_extents([height,width],tail,suffix_tail(height,width,tail,suffix)) : {BandConcat.geometry(channels <> _,0) == Some{Bands.Geometry{1,BandConcat.channels(channels <> _),height,width}} : Maybe<&2,Bands.Geometry>}
      {==}

def geometry_four(+shape: List<&2,U32>,+images: U32,+height: U32,+width: U32,
  prefix: {Shapes.same_extents([images],List.take(&2,U32,shape,1n)) == True{} : Bool},
  suffix: {Shapes.same_extents([height,width],List.drop(&2,U32,shape,2n)) == True{} : Bool}) ->
  {BandConcat.geometry(shape,1) == Some{Bands.Geometry{images,BandConcat.channels(shape),height,width}} : Maybe<&2,Bands.Geometry>}:
  match shape:
    case []: Empty.absurd({None{} == Some{Bands.Geometry{images,0,height,width}} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(suffix))
    case +image <> +rest:
      match rest:
        case []: Empty.absurd({None{} == Some{Bands.Geometry{images,0,height,width}} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(suffix))
        case +channels <> +tail:
          +same_image = ShapeLaws.native_equal(images,image,Boolean.left(U32.is_eq(images,image),Shapes.same_extents([],[]),prefix))
          %Equal.sym(U32,images,image,same_image) : {BandConcat.geometry(image <> channels <> tail,1) == Some{Bands.Geometry{_,BandConcat.channels(image <> channels <> tail),height,width}} : Maybe<&2,Bands.Geometry>}
          %ShapeLaws.same_extents([height,width],tail,suffix_tail(height,width,tail,suffix)) : {BandConcat.geometry(image <> channels <> _,1) == Some{Bands.Geometry{image,BandConcat.channels(image <> channels <> _),height,width}} : Maybe<&2,Bands.Geometry>}
          {==}

def survey_contents(~Element: Data,owners: Tensors.Parts<Element> & Layout.Survey) -> BandConcatExecution.PartContents<Element> & Layout.Survey:
  (parts,summary) = owners
  (BandConcatExecution.parts_contents(~Element,parts),summary)

def observed_summary(~Element: Data,owners: Tensors.Parts<Element> & Layout.Survey) -> Layout.Survey:
  (parts,summary) = owners
  summary

def summary_projection(~Element: Data,owners: BandConcatExecution.PartContents<Element> & Layout.Survey) -> Layout.Survey:
  (parts,summary) = owners
  summary

def summary_contents(~Element: Data,owners: Tensors.Parts<Element> & Layout.Survey) ->
  {observed_summary(~Element,owners) == summary_projection(~Element,survey_contents(~Element,owners)) : Layout.Survey}:
  (parts,summary) = owners
  {==}

def prepend_model(~Element: Data,+input: BandConcatExecution.TensorContent<Element>,owners: BandConcatExecution.PartContents<Element> & Layout.Survey) -> BandConcatExecution.PartContents<Element> & Layout.Survey:
  (rest,summary) = owners
  (BandConcatExecution.ConsContent{input,rest},summary)

def prepend_projection(~Element: Data,input: Tensors.Tensor<Element>,owners: Tensors.Parts<Element> & Layout.Survey) ->
  {survey_contents(~Element,Layout.prepend_survey(~Element,input,owners)) == prepend_model(~Element,BandConcatExecution.tensor_contents(~Element,input),survey_contents(~Element,owners)) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  (rest,summary) = owners
  {==}

def single_survey(~Element: Data,input: Tensors.Tensor<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {survey_contents(~Element,Layout.survey(~Element,Tensors.Single{input},first,axis,summary)) ==
    (BandConcatExecution.SingleContent{BandConcatExecution.tensor_contents(~Element,input)},step(~Element,first,axis,BandConcatExecution.tensor_contents(~Element,input),summary)) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  match input:
    case Tensors.Tensor{storage,shape,fill,status,workers}: {==}
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: {==}

def continued_survey(~Element: Data,+input: BandConcatExecution.TensorOwnerModel<Element>,+rest: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,
  rule: @+next: Layout.Survey -> {survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,next)) ==
    (BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,next)) : BandConcatExecution.PartContents<Element> & Layout.Survey}) ->
  {survey_contents(~Element,Layout.prepend_survey(~Element,BandConcatExecution.tensor_owner_pack(~Element,input),Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,summary))) ==
    (BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))},survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,summary)) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  Equal.trans(BandConcatExecution.PartContents<Element> & Layout.Survey,
    survey_contents(~Element,Layout.prepend_survey(~Element,BandConcatExecution.tensor_owner_pack(~Element,input),Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,summary))),
    prepend_model(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,summary))),
    prepend_model(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,summary))),
    prepend_projection(~Element,BandConcatExecution.tensor_owner_pack(~Element,input),Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,summary)),
    Equal.cong(BandConcatExecution.PartContents<Element> & Layout.Survey,BandConcatExecution.PartContents<Element> & Layout.Survey,owners => prepend_model(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),owners),
      survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,summary)),(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,summary)),rule(summary)))

def cons_survey(~Element: Data,input: BandConcatExecution.TensorOwnerModel<Element>,+rest: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,
  rule: @+next: Layout.Survey -> {survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,next)) ==
    (BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,next)) : BandConcatExecution.PartContents<Element> & Layout.Survey}) ->
  {survey_contents(~Element,Layout.survey(~Element,Tensors.Cons{BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.parts_owner_pack(~Element,rest)},first,axis,summary)) ==
    (BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))},survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,step(~Element,first,axis,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),summary))) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  match input:
    case BandConcatExecution.DenseOwner{buffer,+shape,fill,+status,+workers}:
      continued_survey(~Element,BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers},rest,first,axis,Layout.surveyed(summary,first,shape,axis,status,workers),rule)
    case BandConcatExecution.BandedOwner{bands,geometry,+shape,fill,+status,+workers}:
      continued_survey(~Element,BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers},rest,first,axis,Layout.surveyed(summary,first,shape,axis,status,workers),rule)

def native_survey_packed(~Element: Data,parts: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary)) ==
    (BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary)) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  match parts:
    case BandConcatExecution.SingleOwner{input}: single_survey(~Element,BandConcatExecution.tensor_owner_pack(~Element,input),first,axis,summary)
    case BandConcatExecution.ConsOwner{input,+rest}: cons_survey(~Element,input,rest,first,axis,summary,next => native_survey_packed(~Element,rest,first,axis,next))

def native_survey_observed(~Element: Data,-parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,observation: BandConcatExecution.PartsObservation<Element,parts>) ->
  {survey_contents(~Element,Layout.survey(~Element,parts,first,axis,summary)) == (BandConcatExecution.parts_contents(~Element,parts),survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  BandConcatExecution.PartsObservation{+sources,equation} = observation
  %equation : {survey_contents(~Element,Layout.survey(~Element,_,first,axis,summary)) == (BandConcatExecution.parts_contents(~Element,_),survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary)) : BandConcatExecution.PartContents<Element> & Layout.Survey}
  native_survey_packed(~Element,sources,first,axis,summary)

def native_survey(~Element: Data,parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {survey_contents(~Element,Layout.survey(~Element,parts,first,axis,summary)) == (BandConcatExecution.parts_contents(~Element,parts),survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary)) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  native_survey_observed(~Element,parts,first,axis,summary,BandConcatExecution.observe_parts(~Element,parts))

def native_summary_packed(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary)) == survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary) : Layout.Survey}:
  Equal.trans(Layout.Survey,observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary)),summary_projection(~Element,survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary))),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary),
    summary_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary)),
    Equal.cong(BandConcatExecution.PartContents<Element> & Layout.Survey,Layout.Survey,owners => summary_projection(~Element,owners),
      survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary)),(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary)),native_survey_packed(~Element,parts,first,axis,summary)))

def native_summary_observed(~Element: Data,-parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,observation: BandConcatExecution.PartsObservation<Element,parts>) ->
  {observed_summary(~Element,Layout.survey(~Element,parts,first,axis,summary)) == survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary) : Layout.Survey}:
  BandConcatExecution.PartsObservation{+sources,equation} = observation
  %equation : {observed_summary(~Element,Layout.survey(~Element,_,first,axis,summary)) == survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,axis,summary) : Layout.Survey}
  native_summary_packed(~Element,sources,first,axis,summary)

def native_summary(~Element: Data,parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {observed_summary(~Element,Layout.survey(~Element,parts,first,axis,summary)) == survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,axis,summary) : Layout.Survey}:
  native_summary_observed(~Element,parts,first,axis,summary,BandConcatExecution.observe_parts(~Element,parts))

def native_total_fields_packed(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32,+maximum: {limit == 4294967295 : U32},
  accepted: {survey_valid(observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary))) == True{} : Bool}) ->
  TotalFields<survey_total(observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary))),survey_total(summary),extent_total(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),axis),limit>:
  +contents = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))
  +summary_equal = native_summary_packed(~Element,parts,first,axis,summary)
  +model_accepted = {%summary_equal : {survey_valid(_) == True{} : Bool} accepted : {survey_valid(survey_model(~Element,contents,first,axis,summary)) == True{} : Bool}}
  %Equal.sym(Layout.Survey,observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary)),survey_model(~Element,contents,first,axis,summary),summary_equal) :
    TotalFields<survey_total(_),survey_total(summary),extent_total(~Element,contents,axis),limit>
  total_fields(~Element,contents,first,axis,summary,limit,prefixes(~Element,contents,first,axis,summary,limit,maximum,model_accepted))

def native_total_fields_observed(~Element: Data,-parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32,+maximum: {limit == 4294967295 : U32},
  observation: BandConcatExecution.PartsObservation<Element,parts>,accepted: {survey_valid(observed_summary(~Element,Layout.survey(~Element,parts,first,axis,summary))) == True{} : Bool}) ->
  TotalFields<survey_total(observed_summary(~Element,Layout.survey(~Element,parts,first,axis,summary))),survey_total(summary),extent_total(~Element,BandConcatExecution.parts_contents(~Element,parts),axis),limit>:
  BandConcatExecution.PartsObservation{+sources,+equation} = observation
  +canonical_accepted = {
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equation) : {survey_valid(observed_summary(~Element,Layout.survey(~Element,_,first,axis,summary))) == True{} : Bool}
    accepted : {survey_valid(observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),first,axis,summary))) == True{} : Bool}}
  %equation : TotalFields<survey_total(observed_summary(~Element,Layout.survey(~Element,_,first,axis,summary))),survey_total(summary),extent_total(~Element,BandConcatExecution.parts_contents(~Element,_),axis),limit>
  native_total_fields_packed(~Element,sources,first,axis,summary,limit,maximum,canonical_accepted)

def native_total_fields(~Element: Data,parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32,+maximum: {limit == 4294967295 : U32},
  accepted: {survey_valid(observed_summary(~Element,Layout.survey(~Element,parts,first,axis,summary))) == True{} : Bool}) ->
  TotalFields<survey_total(observed_summary(~Element,Layout.survey(~Element,parts,first,axis,summary))),survey_total(summary),extent_total(~Element,BandConcatExecution.parts_contents(~Element,parts),axis),limit>:
  native_total_fields_observed(~Element,parts,first,axis,summary,limit,maximum,BandConcatExecution.observe_parts(~Element,parts),accepted)

# The two shapes the band route joins: one image at axis 0, or a batch of images at axis 1.
type Rank is Data:
  RankThree{}
  RankFour{images: U32}

def shape_of(rank: Rank,+channels: U32,+height: U32,+width: U32) -> List<&2,U32>:
  match rank:
    case RankThree{}: [channels,height,width]
    case RankFour{images}: [images,channels,height,width]

def axis_of(rank: Rank) -> U32:
  match rank:
    case RankThree{}: 0
    case RankFour{images}: 1

def images_of(rank: Rank) -> U32:
  match rank:
    case RankThree{}: 1
    case RankFour{images}: images

def source_geometry(+rank: Rank,+first_channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+previous: U32,+limit: U32,
  fields: StepFields<shape_of(rank,first_channels,height,width),shape,U32.to_nat(axis_of(rank)),previous,limit>) ->
  {BandConcat.geometry(shape,axis_of(rank)) == Some{Bands.Geometry{images_of(rank),BandConcat.channels(shape),height,width}} : Maybe<&2,Bands.Geometry>}:
  match rank:
    case RankThree{}:
      StepFields{length,prefix,suffix,room,exact} = fields
      geometry_three(shape,height,width,suffix)
    case RankFour{+images}:
      StepFields{length,prefix,suffix,room,exact} = fields
      geometry_four(shape,images,height,width,prefix,suffix)

def output_geometry(+rank: Rank,+channels: U32,+height: U32,+width: U32,+count: U32) ->
  {BandConcat.geometry(Layout.with_extent(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),count),axis_of(rank)) == Some{Bands.Geometry{images_of(rank),count,height,width}} : Maybe<&2,Bands.Geometry>}:
  match rank:
    case RankThree{}: {==}
    case RankFour{images}: {==}

def channel_extent_three(+shape: List<&2,U32>,+height: U32,+width: U32,
  suffix: {Shapes.same_extents([height,width],List.drop(&2,U32,shape,1n)) == True{} : Bool}) ->
  {Layout.extent_at(shape,0n) == BandConcat.channels(shape) : U32}:
  match shape:
    case []: {==}
    case +channels <> +tail:
      %ShapeLaws.same_extents([height,width],tail,suffix_tail(height,width,tail,suffix)) : {Layout.extent_at(channels <> _,0n) == BandConcat.channels(channels <> _) : U32}
      {==}

def channel_extent_four(+shape: List<&2,U32>,+height: U32,+width: U32,
  suffix: {Shapes.same_extents([height,width],List.drop(&2,U32,shape,2n)) == True{} : Bool}) ->
  {Layout.extent_at(shape,1n) == BandConcat.channels(shape) : U32}:
  match shape:
    case []: {==}
    case +images <> +rest:
      match rest:
        case []: {==}
        case +channels <> +tail:
          %ShapeLaws.same_extents([height,width],tail,suffix_tail(height,width,tail,suffix)) : {Layout.extent_at(images <> channels <> _,1n) == BandConcat.channels(images <> channels <> _) : U32}
          {==}

def step_channel_extent(+rank: Rank,+first_channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+previous: U32,+limit: U32,
  fields: StepFields<shape_of(rank,first_channels,height,width),shape,U32.to_nat(axis_of(rank)),previous,limit>) ->
  {Layout.extent_at(shape,U32.to_nat(axis_of(rank))) == BandConcat.channels(shape) : U32}:
  match rank:
    case RankThree{}:
      StepFields{length,prefix,suffix,room,exact} = fields
      channel_extent_three(shape,height,width,suffix)
    case RankFour{+images}:
      StepFields{length,prefix,suffix,room,exact} = fields
      channel_extent_four(shape,height,width,suffix)

def extent_channels(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+rank: Rank,+channels: U32,+height: U32,+width: U32,+summary: Layout.Survey,+limit: U32,
  fields: Prefixes(~Element,parts,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary,limit)) ->
  {extent_total(~Element,parts,U32.to_nat(axis_of(rank))) == BandConcatExecution.total_channels(~Element,parts) : Nat}:
  match parts:
    case BandConcatExecution.SingleContent{+input}:
      Equal.cong(U32,Nat,value => U32.to_nat(value),Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),U32.to_nat(axis_of(rank))),BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),step_channel_extent(rank,channels,height,width,BandConcatExecution.tensor_shape(~Element,input),survey_total(summary),limit,fields))
    case BandConcatExecution.ConsContent{+input,+rest}:
      Shape.Both{head,tail} = fields
      %Equal.cong(U32,Nat,value => U32.to_nat(value),Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),U32.to_nat(axis_of(rank))),BandConcat.channels(BandConcatExecution.tensor_shape(~Element,input)),step_channel_extent(rank,channels,height,width,BandConcatExecution.tensor_shape(~Element,input),survey_total(summary),limit,head)) :
        {Nat.add(U32.to_nat(Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),U32.to_nat(axis_of(rank)))),extent_total(~Element,rest,U32.to_nat(axis_of(rank)))) == Nat.add(_,BandConcatExecution.total_channels(~Element,rest)) : Nat}
      %extent_channels(~Element,rest,rank,channels,height,width,step(~Element,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),input,summary),limit,tail) :
        {Nat.add(U32.to_nat(Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),U32.to_nat(axis_of(rank)))),extent_total(~Element,rest,U32.to_nat(axis_of(rank)))) == Nat.add(U32.to_nat(Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),U32.to_nat(axis_of(rank)))),_) : Nat}
      {==}

def native_channel_total_packed(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>,+rank: Rank,+channels: U32,+height: U32,+width: U32,+summary: Layout.Survey,+limit: U32,+maximum: {limit == 4294967295 : U32},
  +accepted: {survey_valid(observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary))) == True{} : Bool}) ->
  TotalFields<survey_total(observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary))),survey_total(summary),BandConcatExecution.total_channels(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),limit>:
  +contents = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))
  +summary_equal = native_summary_packed(~Element,parts,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary)
  +model_accepted = {%summary_equal : {survey_valid(_) == True{} : Bool} accepted : {survey_valid(survey_model(~Element,contents,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary)) == True{} : Bool}}
  +fields = prefixes(~Element,contents,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary,limit,maximum,model_accepted)
  %extent_channels(~Element,contents,rank,channels,height,width,summary,limit,fields) :
    TotalFields<survey_total(observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary))),survey_total(summary),_,limit>
  native_total_fields_packed(~Element,parts,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary,limit,maximum,accepted)

def native_channel_total_observed(~Element: Data,-parts: Tensors.Parts<Element>,+rank: Rank,+channels: U32,+height: U32,+width: U32,+summary: Layout.Survey,+limit: U32,+maximum: {limit == 4294967295 : U32},
  observation: BandConcatExecution.PartsObservation<Element,parts>,accepted: {survey_valid(observed_summary(~Element,Layout.survey(~Element,parts,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary))) == True{} : Bool}) ->
  TotalFields<survey_total(observed_summary(~Element,Layout.survey(~Element,parts,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary))),survey_total(summary),BandConcatExecution.total_channels(~Element,BandConcatExecution.parts_contents(~Element,parts)),limit>:
  BandConcatExecution.PartsObservation{+sources,+equation} = observation
  +canonical_accepted = {
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equation) : {survey_valid(observed_summary(~Element,Layout.survey(~Element,_,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary))) == True{} : Bool}
    accepted : {survey_valid(observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary))) == True{} : Bool}}
  %equation : TotalFields<survey_total(observed_summary(~Element,Layout.survey(~Element,_,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary))),survey_total(summary),BandConcatExecution.total_channels(~Element,BandConcatExecution.parts_contents(~Element,_)),limit>
  native_channel_total_packed(~Element,sources,rank,channels,height,width,summary,limit,maximum,canonical_accepted)

def native_channel_total(~Element: Data,parts: Tensors.Parts<Element>,+rank: Rank,+channels: U32,+height: U32,+width: U32,+summary: Layout.Survey,+limit: U32,+maximum: {limit == 4294967295 : U32},
  accepted: {survey_valid(observed_summary(~Element,Layout.survey(~Element,parts,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary))) == True{} : Bool}) ->
  TotalFields<survey_total(observed_summary(~Element,Layout.survey(~Element,parts,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),summary))),survey_total(summary),BandConcatExecution.total_channels(~Element,BandConcatExecution.parts_contents(~Element,parts)),limit>:
  native_channel_total_observed(~Element,parts,rank,channels,height,width,summary,limit,maximum,BandConcatExecution.observe_parts(~Element,parts),accepted)

# The public caller keeps the actual Parts through lead, survey and discovery.
# The successful band branch enters the same join model as native execution.
# Values still need the ordered prefix and destination-bound producers below.
def tensor_choice(~Element: Data,input: BandConcatExecution.TensorContent<Element>) -> Maybe<&2,Bands.Plan>:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}: None{}
    case BandConcatExecution.BandedContent{bands,geometry,shape,fill,status,workers}: Some{Model.content_layout(~Element,bands)}

def discovery_model(~Element: Data,parts: BandConcatExecution.PartContents<Element>) -> Maybe<&2,Bands.Plan>:
  match parts:
    case BandConcatExecution.SingleContent{input}: tensor_choice(~Element,input)
    case BandConcatExecution.ConsContent{input,rest}: BandConcat.first_plan(tensor_choice(~Element,input),discovery_model(~Element,rest))

def tensor_plan_contents(~Element: Data,owners: Tensors.Tensor<Element> & Maybe<&2,Bands.Plan>) -> BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>:
  (input,layout) = owners
  (BandConcatExecution.tensor_contents(~Element,input),layout)

def tensor_plan_banded(~Element: Data,+source: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32) ->
  {tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers})) ==
    (BandConcatExecution.tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers}),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers}))) : BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,source)),(Model.pack(~Element,source),Model.layout(~Element,source)),Model.plan(~Element,source)) :
    {tensor_plan_contents(~Element,BandConcat.tensor_planned(~Element,geometry,shape,fill,status,workers,_)) == (BandConcatExecution.tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers}),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers}))) : BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>}
  Equal.cong(Bands.Plan,BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>,layout => (BandConcatExecution.tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,status,workers}),Some{layout}),
    Model.layout(~Element,source),Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,source))),
    Equal.trans(Bands.Plan,Model.layout(~Element,source),Model.content_layout(~Element,Model.content(~Element,source)),Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,source))),
      Model.layout_matches_content(~Element,source),Equal.cong(Model.Content<Element>,Bands.Plan,content => Model.content_layout(~Element,content),Model.content(~Element,source),Model.contents(~Element,Model.pack(~Element,source)),Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,source)),Model.content(~Element,source),Model.packed_contents(~Element,source)))))

def observed_tensor_plan(~Element: Data,-bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  observation: Model.Observation<Element,bands>) ->
  {tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,Tensors.Banded{bands,geometry,shape,fill,status,workers})) ==
    (BandConcatExecution.tensor_contents(~Element,Tensors.Banded{bands,geometry,shape,fill,status,workers}),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,Tensors.Banded{bands,geometry,shape,fill,status,workers}))) : BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>}:
  Model.Observation{+source,equation} = observation
  %equation : {tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,Tensors.Banded{_,geometry,shape,fill,status,workers})) ==
    (BandConcatExecution.tensor_contents(~Element,Tensors.Banded{_,geometry,shape,fill,status,workers}),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,Tensors.Banded{_,geometry,shape,fill,status,workers}))) : BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>}
  tensor_plan_banded(~Element,source,geometry,shape,fill,status,workers)

def tensor_plan(~Element: Data,input: Tensors.Tensor<Element>) ->
  {tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,input)) == (BandConcatExecution.tensor_contents(~Element,input),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,input))) : BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>}:
  match input:
    case Tensors.Tensor{buffer,shape,fill,status,workers}: {==}
    case Tensors.Banded{bands,+geometry,+shape,+fill,+status,+workers}:
      observed_tensor_plan(~Element,bands,geometry,shape,fill,status,workers,Model.observe(~Element,bands))

def discovered_contents(~Element: Data,owners: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) -> BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>:
  (parts,layout) = owners
  (BandConcatExecution.parts_contents(~Element,parts),layout)

def single_model(~Element: Data,owners: BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>) -> BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>:
  (input,layout) = owners
  (BandConcatExecution.SingleContent{input},layout)

def single_projection(~Element: Data,owners: Tensors.Tensor<Element> & Maybe<&2,Bands.Plan>) ->
  {discovered_contents(~Element,BandConcat.single_plan(~Element,owners)) == single_model(~Element,tensor_plan_contents(~Element,owners)) : BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>}:
  (input,layout) = owners
  {==}

def cons_model(~Element: Data,first: BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>,rest: BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>) -> BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>:
  (input,left) = first
  (rest,right) = rest
  (BandConcatExecution.ConsContent{input,rest},BandConcat.first_plan(left,right))

def cons_projection(~Element: Data,first: Tensors.Tensor<Element> & Maybe<&2,Bands.Plan>,rest: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) ->
  {discovered_contents(~Element,BandConcat.cons_plan(~Element,first,rest)) == cons_model(~Element,tensor_plan_contents(~Element,first),discovered_contents(~Element,rest)) : BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>}:
  (input,left) = first
  (rest,right) = rest
  {==}

def discover_packed(~Element: Data,parts: BandConcatExecution.PartsOwnerModel<Element>) ->
  {discovered_contents(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))) == (BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)))) : BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>}:
  match parts:
    case BandConcatExecution.SingleOwner{+input}:
      Equal.trans(BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>,discovered_contents(~Element,BandConcat.discover(~Element,Tensors.Single{BandConcatExecution.tensor_owner_pack(~Element,input)})),
        single_model(~Element,tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),single_model(~Element,(BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))))),
        single_projection(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),Equal.cong(BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>,BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>,owners => single_model(~Element,owners),
          tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),(BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))))
    case BandConcatExecution.ConsOwner{+input,+rest}:
      %Equal.sym(BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>,discovered_contents(~Element,BandConcat.cons_plan(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))),
        cons_model(~Element,tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),discovered_contents(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))),cons_projection(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))) :
        {_ == (BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))},BandConcat.first_plan(tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))))) : BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>}
      %Equal.sym(BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>,tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),(BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))) :
        {cons_model(~Element,_,discovered_contents(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))) == (BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))},BandConcat.first_plan(tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))))) : BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>}
      %Equal.sym(BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>,discovered_contents(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))),(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))),discover_packed(~Element,rest)) :
        {cons_model(~Element,(BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),_) == (BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))},BandConcat.first_plan(tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))))) : BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>}
      {==}

def discover_observed(~Element: Data,-parts: Tensors.Parts<Element>,observation: BandConcatExecution.PartsObservation<Element,parts>) ->
  {discovered_contents(~Element,BandConcat.discover(~Element,parts)) == (BandConcatExecution.parts_contents(~Element,parts),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,parts))) : BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>}:
  BandConcatExecution.PartsObservation{+sources,equation} = observation
  %equation : {discovered_contents(~Element,BandConcat.discover(~Element,_)) == (BandConcatExecution.parts_contents(~Element,_),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,_))) : BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>}
  discover_packed(~Element,sources)

def discover(~Element: Data,parts: Tensors.Parts<Element>) ->
  {discovered_contents(~Element,BandConcat.discover(~Element,parts)) == (BandConcatExecution.parts_contents(~Element,parts),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,parts))) : BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>}:
  discover_observed(~Element,parts,BandConcatExecution.observe_parts(~Element,parts))

def discovered_parts(~Element: Data,owners: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) -> BandConcatExecution.PartContents<Element>:
  (parts,layout) = owners
  BandConcatExecution.parts_contents(~Element,parts)

def discovered_plan(~Element: Data,owners: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) -> Maybe<&2,Bands.Plan>:
  (parts,layout) = owners
  layout

def content_parts(~Element: Data,owners: BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>) -> BandConcatExecution.PartContents<Element>:
  (parts,layout) = owners
  parts

def content_plan(~Element: Data,owners: BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>) -> Maybe<&2,Bands.Plan>:
  (parts,layout) = owners
  layout

def discovery_parts_projection(~Element: Data,owners: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) ->
  {discovered_parts(~Element,owners) == content_parts(~Element,discovered_contents(~Element,owners)) : BandConcatExecution.PartContents<Element>}:
  (parts,layout) = owners
  {==}

def discovery_plan_projection(~Element: Data,owners: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) ->
  {discovered_plan(~Element,owners) == content_plan(~Element,discovered_contents(~Element,owners)) : Maybe<&2,Bands.Plan>}:
  (parts,layout) = owners
  {==}

def discover_parts_packed(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>) ->
  {discovered_parts(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))) == BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)) : BandConcatExecution.PartContents<Element>}:
  Equal.trans(BandConcatExecution.PartContents<Element>,discovered_parts(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),content_parts(~Element,discovered_contents(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)))),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),
    discovery_parts_projection(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),Equal.cong(BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>,BandConcatExecution.PartContents<Element>,owners => content_parts(~Element,owners),
      discovered_contents(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)))),discover_packed(~Element,parts)))

def discover_parts_observed(~Element: Data,-parts: Tensors.Parts<Element>,observation: BandConcatExecution.PartsObservation<Element,parts>) ->
  {discovered_parts(~Element,BandConcat.discover(~Element,parts)) == BandConcatExecution.parts_contents(~Element,parts) : BandConcatExecution.PartContents<Element>}:
  BandConcatExecution.PartsObservation{+sources,equation} = observation
  %equation : {discovered_parts(~Element,BandConcat.discover(~Element,_)) == BandConcatExecution.parts_contents(~Element,_) : BandConcatExecution.PartContents<Element>}
  discover_parts_packed(~Element,sources)

def discover_parts(~Element: Data,parts: Tensors.Parts<Element>) ->
  {discovered_parts(~Element,BandConcat.discover(~Element,parts)) == BandConcatExecution.parts_contents(~Element,parts) : BandConcatExecution.PartContents<Element>}:
  discover_parts_observed(~Element,parts,BandConcatExecution.observe_parts(~Element,parts))

def discover_plan_packed(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>) ->
  {discovered_plan(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))) == discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))) : Maybe<&2,Bands.Plan>}:
  Equal.trans(Maybe<&2,Bands.Plan>,discovered_plan(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),content_plan(~Element,discovered_contents(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)))),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),
    discovery_plan_projection(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),Equal.cong(BandConcatExecution.PartContents<Element> & Maybe<&2,Bands.Plan>,Maybe<&2,Bands.Plan>,owners => content_plan(~Element,owners),
      discovered_contents(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)))),discover_packed(~Element,parts)))

def discover_plan_observed(~Element: Data,-parts: Tensors.Parts<Element>,observation: BandConcatExecution.PartsObservation<Element,parts>) ->
  {discovered_plan(~Element,BandConcat.discover(~Element,parts)) == discovery_model(~Element,BandConcatExecution.parts_contents(~Element,parts)) : Maybe<&2,Bands.Plan>}:
  BandConcatExecution.PartsObservation{+sources,equation} = observation
  %equation : {discovered_plan(~Element,BandConcat.discover(~Element,_)) == discovery_model(~Element,BandConcatExecution.parts_contents(~Element,_)) : Maybe<&2,Bands.Plan>}
  discover_plan_packed(~Element,sources)

def discover_plan(~Element: Data,parts: Tensors.Parts<Element>) ->
  {discovered_plan(~Element,BandConcat.discover(~Element,parts)) == discovery_model(~Element,BandConcatExecution.parts_contents(~Element,parts)) : Maybe<&2,Bands.Plan>}:
  discover_plan_observed(~Element,parts,BandConcatExecution.observe_parts(~Element,parts))

def join_discovered(~Element: Data,owners: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  geometry_selected: {BandConcat.geometry(shape,axis) == Some{geometry} : Maybe<&2,Bands.Geometry>},count_selected: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  layout_selected: {discovered_plan(~Element,owners) == Some{layout} : Maybe<&2,Bands.Plan>}) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.join_discovered(~Element,shape,axis,fill,workers,owners)) == BandConcatExecution.join_model(~Element,layout,geometry,discovered_parts(~Element,owners),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}:
  (parts,+selected) = owners
  %Equal.sym(Maybe<&2,Bands.Plan>,selected,Some{layout},layout_selected) :
    {BandConcatExecution.tensor_contents(~Element,Layout.join_selected(~Element,BandConcat.geometry(shape,axis),Tensors.count_of(shape),shape,axis,fill,workers,_,parts)) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,parts),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}
  %Equal.sym(Maybe<&2,Bands.Geometry>,BandConcat.geometry(shape,axis),Some{geometry},geometry_selected) :
    {BandConcatExecution.tensor_contents(~Element,Layout.join_selected(~Element,_,Tensors.count_of(shape),shape,axis,fill,workers,Some{layout},parts)) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,parts),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},count_selected) :
    {BandConcatExecution.tensor_contents(~Element,Layout.join_selected(~Element,Some{geometry},_,shape,axis,fill,workers,Some{layout},parts)) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,parts),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}
  BandConcatExecution.join(~Element,layout,geometry,parts,shape,fill,workers)

def join_from_discovery_packed(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  geometry_selected: {BandConcat.geometry(shape,axis) == Some{geometry} : Maybe<&2,Bands.Geometry>},count_selected: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  layout_selected: {discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))) == Some{layout} : Maybe<&2,Bands.Plan>}) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.join_discovered(~Element,shape,axis,fill,workers,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)))) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}:
  %discover_parts_packed(~Element,parts) :
    {BandConcatExecution.tensor_contents(~Element,Layout.join_discovered(~Element,shape,axis,fill,workers,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)))) == BandConcatExecution.join_model(~Element,layout,geometry,_,shape,fill,workers) : BandConcatExecution.TensorContent<Element>}
  join_discovered(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),shape,axis,fill,workers,geometry,count,layout,geometry_selected,count_selected,
    Equal.trans(Maybe<&2,Bands.Plan>,discovered_plan(~Element,BandConcat.discover(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),Some{layout},discover_plan_packed(~Element,parts),layout_selected))

def join_from_discovery_observed(~Element: Data,-parts: Tensors.Parts<Element>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  observation: BandConcatExecution.PartsObservation<Element,parts>,geometry_selected: {BandConcat.geometry(shape,axis) == Some{geometry} : Maybe<&2,Bands.Geometry>},count_selected: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  layout_selected: {discovery_model(~Element,BandConcatExecution.parts_contents(~Element,parts)) == Some{layout} : Maybe<&2,Bands.Plan>}) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.join_discovered(~Element,shape,axis,fill,workers,BandConcat.discover(~Element,parts))) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,parts),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}:
  BandConcatExecution.PartsObservation{+sources,+equation} = observation
  %equation : {BandConcatExecution.tensor_contents(~Element,Layout.join_discovered(~Element,shape,axis,fill,workers,BandConcat.discover(~Element,_))) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,_),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}
  join_from_discovery_packed(~Element,sources,shape,axis,fill,workers,geometry,count,layout,geometry_selected,count_selected,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equation) : {discovery_model(~Element,BandConcatExecution.parts_contents(~Element,_)) == Some{layout} : Maybe<&2,Bands.Plan>} layout_selected)

def join_from_discovery(~Element: Data,parts: Tensors.Parts<Element>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  geometry_selected: {BandConcat.geometry(shape,axis) == Some{geometry} : Maybe<&2,Bands.Geometry>},count_selected: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  layout_selected: {discovery_model(~Element,BandConcatExecution.parts_contents(~Element,parts)) == Some{layout} : Maybe<&2,Bands.Plan>}) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.join_discovered(~Element,shape,axis,fill,workers,BandConcat.discover(~Element,parts))) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,parts),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}:
  join_from_discovery_observed(~Element,parts,shape,axis,fill,workers,geometry,count,layout,BandConcatExecution.observe_parts(~Element,parts),geometry_selected,count_selected,layout_selected)

def allocation_metadata(~Element: Data,layout: Bands.Plan,+geometry: Bands.Geometry,+fill: Element,+capacity: U32,+division_capacity: U32,
  fitting: BandDense.Fits(geometry,layout,capacity,division_capacity)) -> BandShape.Contents(~Element,geometry,BandConcatExecution.allocation_model(~Element,geometry,fill,layout)):
  match layout:
    case Bands.Rows{first,rows}:
      BandDense.LeafFacts{allocatable,address,metadata} = fitting
      metadata
    case Bands.Split{+left,+right}:
      Shape.Both{left_fitting,right_fitting} = fitting
      Shape.Both{allocation_metadata(~Element,left,geometry,fill,capacity,division_capacity,left_fitting),allocation_metadata(~Element,right,geometry,fill,capacity,division_capacity,right_fitting)}

def allocation_ready(~Element: Data,layout: Bands.Plan,+geometry: Bands.Geometry,+fill: Element,+capacity: U32,+division_capacity: U32,
  fitting: BandDense.Fits(geometry,layout,capacity,division_capacity)) -> Model.ContentReady(~Element,BandConcatExecution.allocation_model(~Element,geometry,fill,layout)):
  match layout:
    case Bands.Rows{first,+rows}:
      BandDense.LeafFacts{allocatable,address,metadata} = fitting
      Buffer.content_ready(~Element,BandCount.model(~Element,Bands.local_count(geometry,rows),fill),BandCount.bounded_model(~Element,Bands.local_count(geometry,rows),fill,allocatable))
    case Bands.Split{+left,+right}:
      Shape.Both{left_fitting,right_fitting} = fitting
      Shape.Both{allocation_ready(~Element,left,geometry,fill,capacity,division_capacity,left_fitting),allocation_ready(~Element,right,geometry,fill,capacity,division_capacity,right_fitting)}

def allocation_layout(~Element: Data,layout: Bands.Plan,+geometry: Bands.Geometry,+fill: Element) ->
  {Model.content_layout(~Element,BandConcatExecution.allocation_model(~Element,geometry,fill,layout)) == layout : Bands.Plan}:
  match layout:
    case Bands.Rows{first,rows}: {==}
    case Bands.Split{+left,+right}:
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,BandConcatExecution.allocation_model(~Element,geometry,fill,left)),left,allocation_layout(~Element,left,geometry,fill)) :
        {Bands.Split{_,Model.content_layout(~Element,BandConcatExecution.allocation_model(~Element,geometry,fill,right))} == Bands.Split{left,right} : Bands.Plan}
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,BandConcatExecution.allocation_model(~Element,geometry,fill,right)),right,allocation_layout(~Element,right,geometry,fill)) :
        {Bands.Split{left,_} == Bands.Split{left,right} : Bands.Plan}
      {==}

def allocation_well_formed(~Element: Data,+layout: Bands.Plan,+geometry: Bands.Geometry,+fill: Element,+capacity: U32,+division_capacity: U32,
  +fitting: BandDense.Fits(geometry,layout,capacity,division_capacity),coverage: BandPlan.Coverage(layout,0n,Inspection.total_rows(geometry))) ->
  BandShape.WellFormed<Element,geometry,BandConcatExecution.allocation_model(~Element,geometry,fill,layout)>:
  BandShape.WellFormed{allocation_metadata(~Element,layout,geometry,fill,capacity,division_capacity,fitting),allocation_ready(~Element,layout,geometry,fill,capacity,division_capacity,fitting),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,BandConcatExecution.allocation_model(~Element,geometry,fill,layout)),layout,allocation_layout(~Element,layout,geometry,fill)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry)) coverage}

def cursor_natural(~Element: Data,+parts: BandConcatExecution.PartContents<Element>,+channel: U32,+limit: U32,+survey_count: U32,
  counted: TotalFields<survey_count,channel,BandConcatExecution.total_channels(~Element,parts),limit>) ->
  {U32.to_nat(BandConcatExecution.channel_cursor(~Element,parts,channel)) == U32.to_nat(survey_count) : Nat}:
  TotalFields{+exact,+room} = counted
  Equal.trans(Nat,U32.to_nat(BandConcatExecution.channel_cursor(~Element,parts,channel)),Nat.add(U32.to_nat(channel),BandConcatExecution.total_channels(~Element,parts)),U32.to_nat(survey_count),
    Equal.trans(Nat,U32.to_nat(BandConcatExecution.channel_cursor(~Element,parts,channel)),U32.to_nat(U32.from_nat(Nat.add(U32.to_nat(channel),BandConcatExecution.total_channels(~Element,parts)))),Nat.add(U32.to_nat(channel),BandConcatExecution.total_channels(~Element,parts)),
      Equal.cong(U32,Nat,value => U32.to_nat(value),BandConcatExecution.channel_cursor(~Element,parts,channel),U32.from_nat(Nat.add(U32.to_nat(channel),BandConcatExecution.total_channels(~Element,parts))),BandConcatExecution.cursor_exact(~Element,parts,channel)),
      FullRange.roundtrip(Nat.add(U32.to_nat(channel),BandConcatExecution.total_channels(~Element,parts)),limit,room)),
    Equal.sym(Nat,U32.to_nat(survey_count),Nat.add(U32.to_nat(channel),BandConcatExecution.total_channels(~Element,parts)),exact))

def cursor_word(~Element: Data,+parts: BandConcatExecution.PartContents<Element>,+channel: U32,+limit: U32,+survey_count: U32,
  counted: TotalFields<survey_count,channel,BandConcatExecution.total_channels(~Element,parts),limit>) ->
  {BandConcatExecution.channel_cursor(~Element,parts,channel) == survey_count : U32}:
  %Arithmetic.native_roundtrip(BandConcatExecution.channel_cursor(~Element,parts,channel)) : {_ == survey_count : U32}
  %Arithmetic.native_roundtrip(survey_count) : {U32.from_nat(U32.to_nat(BandConcatExecution.channel_cursor(~Element,parts,channel))) == _ : U32}
  Equal.cong(Nat,U32,value => U32.from_nat(value),U32.to_nat(BandConcatExecution.channel_cursor(~Element,parts,channel)),U32.to_nat(survey_count),cursor_natural(~Element,parts,channel,limit,survey_count,counted))

type CompatibleDiscovery<-Element: Data> is Data:
  CompatibleDiscovery{parts: BandConcatExecution.PartContents<Element>,layout: Maybe<&2,Bands.Plan>,compatible: Bool}

def compatible_parts(~Element: Data,receipt: CompatibleDiscovery<Element>) -> BandConcatExecution.PartContents<Element>:
  CompatibleDiscovery{parts,layout,compatible} = receipt
  parts

def compatible_layout(~Element: Data,receipt: CompatibleDiscovery<Element>) -> Maybe<&2,Bands.Plan>:
  CompatibleDiscovery{parts,layout,compatible} = receipt
  layout

def compatible_flag(~Element: Data,receipt: CompatibleDiscovery<Element>) -> Bool:
  CompatibleDiscovery{parts,layout,compatible} = receipt
  compatible

def plan_present(layout: Maybe<&2,Bands.Plan>) -> Bool:
  match layout:
    case Some{layout}: True{}
    case None{}: False{}

def compatible_model(~Element: Data,parts: BandConcatExecution.PartContents<Element>) -> CompatibleDiscovery<Element>:
  match parts:
    case BandConcatExecution.SingleContent{+input}:
      CompatibleDiscovery{BandConcatExecution.SingleContent{input},tensor_choice(~Element,input),plan_present(tensor_choice(~Element,input))}
    case BandConcatExecution.ConsContent{+input,+rest}:
      CompatibleDiscovery{BandConcatExecution.ConsContent{input,rest},discovery_model(~Element,BandConcatExecution.ConsContent{input,rest}),
        compatible_flag(~Element,compatible_model(~Element,rest)) && BandConcat.compatible_plans(tensor_choice(~Element,input),discovery_model(~Element,rest))}

def compatible_layout_model(~Element: Data,parts: BandConcatExecution.PartContents<Element>) ->
  {compatible_layout(~Element,compatible_model(~Element,parts)) == discovery_model(~Element,parts) : Maybe<&2,Bands.Plan>}:
  match parts:
    case BandConcatExecution.SingleContent{input}: {==}
    case BandConcatExecution.ConsContent{input,rest}: {==}

def compatible_parts_model(~Element: Data,parts: BandConcatExecution.PartContents<Element>) ->
  {compatible_parts(~Element,compatible_model(~Element,parts)) == parts : BandConcatExecution.PartContents<Element>}:
  match parts:
    case BandConcatExecution.SingleContent{input}: {==}
    case BandConcatExecution.ConsContent{input,rest}: {==}

def compatible_contents(~Element: Data,owners: (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool) -> CompatibleDiscovery<Element>:
  ((parts,layout),compatible) = owners
  CompatibleDiscovery{BandConcatExecution.parts_contents(~Element,parts),layout,compatible}

def compatible_single_model(~Element: Data,first: BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>) -> CompatibleDiscovery<Element>:
  (+input,+layout) = first
  CompatibleDiscovery{BandConcatExecution.SingleContent{input},layout,plan_present(layout)}

def compatible_single_projection(~Element: Data,first: Tensors.Tensor<Element> & Maybe<&2,Bands.Plan>) ->
  {compatible_contents(~Element,BandConcat.single_compatible(~Element,first)) == compatible_single_model(~Element,tensor_plan_contents(~Element,first)) : CompatibleDiscovery<Element>}:
  (input,layout) = first
  match layout:
    case Some{layout}: {==}
    case None{}: {==}

def compatible_cons_model(~Element: Data,first: BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>,tail: CompatibleDiscovery<Element>) -> CompatibleDiscovery<Element>:
  (+input,+first_layout) = first
  CompatibleDiscovery{+rest,+other,+compatible} = tail
  CompatibleDiscovery{BandConcatExecution.ConsContent{input,rest},BandConcat.first_plan(first_layout,other),compatible && BandConcat.compatible_plans(first_layout,other)}

def compatible_cons_fields(~Element: Data,+input: BandConcatExecution.TensorContent<Element>,+candidate: Maybe<&2,Bands.Plan>,+tail: CompatibleDiscovery<Element>) ->
  {compatible_cons_model(~Element,(input,candidate),tail) == CompatibleDiscovery{BandConcatExecution.ConsContent{input,compatible_parts(~Element,tail)},BandConcat.first_plan(candidate,compatible_layout(~Element,tail)),compatible_flag(~Element,tail) && BandConcat.compatible_plans(candidate,compatible_layout(~Element,tail))} : CompatibleDiscovery<Element>}:
  CompatibleDiscovery{+rest,+other,+compatible} = tail
  {==}

def compatible_cons_projection(~Element: Data,first: Tensors.Tensor<Element> & Maybe<&2,Bands.Plan>,rest: (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool) ->
  {compatible_contents(~Element,BandConcat.cons_compatible(~Element,first,rest)) == compatible_cons_model(~Element,tensor_plan_contents(~Element,first),compatible_contents(~Element,rest)) : CompatibleDiscovery<Element>}:
  (input,first_layout) = first
  ((rest,other),compatible) = rest
  {==}

def compatible_packed(~Element: Data,sources: BandConcatExecution.PartsOwnerModel<Element>) ->
  {compatible_contents(~Element,BandConcat.discover_compatible(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))) == compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))) : CompatibleDiscovery<Element>}:
  match sources:
    case BandConcatExecution.SingleOwner{+input}:
      %Equal.sym(CompatibleDiscovery<Element>,compatible_contents(~Element,BandConcat.single_compatible(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),
        compatible_single_model(~Element,tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),compatible_single_projection(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))) :
        {_ == compatible_model(~Element,BandConcatExecution.SingleContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))}) : CompatibleDiscovery<Element>}
      %Equal.sym(BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>,tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),
        (BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))) :
        {compatible_single_model(~Element,_) == compatible_model(~Element,BandConcatExecution.SingleContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))}) : CompatibleDiscovery<Element>}
      {==}
    case BandConcatExecution.ConsOwner{+input,+rest}:
      %Equal.sym(CompatibleDiscovery<Element>,compatible_contents(~Element,BandConcat.cons_compatible(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcat.discover_compatible(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))),
        compatible_cons_model(~Element,tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),compatible_contents(~Element,BandConcat.discover_compatible(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))),
        compatible_cons_projection(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcat.discover_compatible(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))) :
        {_ == compatible_model(~Element,BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))}) : CompatibleDiscovery<Element>}
      %Equal.sym(BandConcatExecution.TensorContent<Element> & Maybe<&2,Bands.Plan>,tensor_plan_contents(~Element,BandConcat.tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),
        (BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),tensor_plan(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))) :
        {compatible_cons_model(~Element,_,compatible_contents(~Element,BandConcat.discover_compatible(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))) == compatible_model(~Element,BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))}) : CompatibleDiscovery<Element>}
      %Equal.sym(CompatibleDiscovery<Element>,compatible_contents(~Element,BandConcat.discover_compatible(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))),compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))),compatible_packed(~Element,rest)) :
        {compatible_cons_model(~Element,(BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),_) == compatible_model(~Element,BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))}) : CompatibleDiscovery<Element>}
      %Equal.sym(CompatibleDiscovery<Element>,compatible_cons_model(~Element,(BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)))),compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))),
        CompatibleDiscovery{BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),compatible_parts(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))))},
          BandConcat.first_plan(tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),compatible_layout(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))))),
          compatible_flag(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))) && BandConcat.compatible_plans(tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),compatible_layout(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))))},
        compatible_cons_fields(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))))) :
        {_ == compatible_model(~Element,BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))}) : CompatibleDiscovery<Element>}
      %Equal.sym(BandConcatExecution.PartContents<Element>,compatible_parts(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),compatible_parts_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))) :
        {CompatibleDiscovery{BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),_},
          BandConcat.first_plan(tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),compatible_layout(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))))),
          compatible_flag(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))) && BandConcat.compatible_plans(tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),compatible_layout(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))))} ==
          compatible_model(~Element,BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))}) : CompatibleDiscovery<Element>}
      %Equal.sym(Maybe<&2,Bands.Plan>,compatible_layout(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))),discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))),compatible_layout_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))) :
        {CompatibleDiscovery{BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))},
          BandConcat.first_plan(tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),_),
          compatible_flag(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)))) && BandConcat.compatible_plans(tensor_choice(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input))),_)} ==
          compatible_model(~Element,BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))}) : CompatibleDiscovery<Element>}
      {==}

def compatible_observed(~Element: Data,-parts: Tensors.Parts<Element>,observation: BandConcatExecution.PartsObservation<Element,parts>) ->
  {compatible_contents(~Element,BandConcat.discover_compatible(~Element,parts)) == compatible_model(~Element,BandConcatExecution.parts_contents(~Element,parts)) : CompatibleDiscovery<Element>}:
  BandConcatExecution.PartsObservation{+sources,equation} = observation
  %equation : {compatible_contents(~Element,BandConcat.discover_compatible(~Element,_)) == compatible_model(~Element,BandConcatExecution.parts_contents(~Element,_)) : CompatibleDiscovery<Element>}
  compatible_packed(~Element,sources)

def compatible_native(~Element: Data,parts: Tensors.Parts<Element>) ->
  {compatible_contents(~Element,BandConcat.discover_compatible(~Element,parts)) == compatible_model(~Element,BandConcatExecution.parts_contents(~Element,parts)) : CompatibleDiscovery<Element>}:
  compatible_observed(~Element,parts,BandConcatExecution.observe_parts(~Element,parts))

def join_compatible_received(~Element: Data,owners: (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  enabled: {compatible_flag(~Element,compatible_contents(~Element,owners)) && Bool.not(Shapes.has_zero(shape)) == True{} : Bool},
  geometry_selected: {BandConcat.geometry(shape,axis) == Some{geometry} : Maybe<&2,Bands.Geometry>},count_selected: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  layout_selected: {compatible_layout(~Element,compatible_contents(~Element,owners)) == Some{layout} : Maybe<&2,Bands.Plan>}) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.join_compatible(~Element,shape,axis,fill,workers,owners)) == BandConcatExecution.join_model(~Element,layout,geometry,compatible_parts(~Element,compatible_contents(~Element,owners)),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}:
  (parts_and_layout,+compatible) = owners
  (parts,candidate) = parts_and_layout
  +candidate = {candidate : Maybe<&2,Bands.Plan>}
  %Equal.sym(Bool,compatible && Bool.not(Shapes.has_zero(shape)),True{},enabled) :
    {BandConcatExecution.tensor_contents(~Element,Layout.join_compatible_selected(~Element,_,shape,axis,fill,workers,{(parts,candidate) : Tensors.Parts<Element> & Maybe<&2,Bands.Plan>})) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,parts),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}
  join_discovered(~Element,{(parts,candidate) : Tensors.Parts<Element> & Maybe<&2,Bands.Plan>},shape,axis,fill,workers,geometry,count,layout,geometry_selected,count_selected,layout_selected)

def join_from_compatible_packed(~Element: Data,+sources: BandConcatExecution.PartsOwnerModel<Element>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  enabled: {compatible_flag(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)))) && Bool.not(Shapes.has_zero(shape)) == True{} : Bool},
  geometry_selected: {BandConcat.geometry(shape,axis) == Some{geometry} : Maybe<&2,Bands.Geometry>},count_selected: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  layout_selected: {discovery_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))) == Some{layout} : Maybe<&2,Bands.Plan>}) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.join_compatible(~Element,shape,axis,fill,workers,BandConcat.discover_compatible(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)))) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}:
  +parts = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))
  +native = compatible_contents(~Element,BandConcat.discover_compatible(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)))
  +model = compatible_model(~Element,parts)
  +equation = compatible_packed(~Element,sources)
  %Equal.trans(BandConcatExecution.PartContents<Element>,compatible_parts(~Element,native),compatible_parts(~Element,model),parts,
    Equal.cong(CompatibleDiscovery<Element>,BandConcatExecution.PartContents<Element>,compatible_parts(~Element),native,model,equation),compatible_parts_model(~Element,parts)) :
    {BandConcatExecution.tensor_contents(~Element,Layout.join_compatible(~Element,shape,axis,fill,workers,BandConcat.discover_compatible(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)))) == BandConcatExecution.join_model(~Element,layout,geometry,_,shape,fill,workers) : BandConcatExecution.TensorContent<Element>}
  join_compatible_received(~Element,BandConcat.discover_compatible(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),shape,axis,fill,workers,geometry,count,layout,
    %Equal.sym(Bool,compatible_flag(~Element,native) && Bool.not(Shapes.has_zero(shape)),compatible_flag(~Element,model) && Bool.not(Shapes.has_zero(shape)),
      Equal.cong(CompatibleDiscovery<Element>,Bool,receipt => compatible_flag(~Element,receipt) && Bool.not(Shapes.has_zero(shape)),native,model,equation)) : {_ == True{} : Bool} enabled,
    geometry_selected,count_selected,
    %Equal.sym(Maybe<&2,Bands.Plan>,compatible_layout(~Element,native),discovery_model(~Element,parts),
      Equal.trans(Maybe<&2,Bands.Plan>,compatible_layout(~Element,native),compatible_layout(~Element,model),discovery_model(~Element,parts),
        Equal.cong(CompatibleDiscovery<Element>,Maybe<&2,Bands.Plan>,compatible_layout(~Element),native,model,equation),compatible_layout_model(~Element,parts))) : {_ == Some{layout} : Maybe<&2,Bands.Plan>} layout_selected)

def join_from_compatible_observed(~Element: Data,-parts: Tensors.Parts<Element>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  observation: BandConcatExecution.PartsObservation<Element,parts>,
  enabled: {compatible_flag(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,parts))) && Bool.not(Shapes.has_zero(shape)) == True{} : Bool},
  geometry_selected: {BandConcat.geometry(shape,axis) == Some{geometry} : Maybe<&2,Bands.Geometry>},count_selected: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  layout_selected: {discovery_model(~Element,BandConcatExecution.parts_contents(~Element,parts)) == Some{layout} : Maybe<&2,Bands.Plan>}) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.join_compatible(~Element,shape,axis,fill,workers,BandConcat.discover_compatible(~Element,parts))) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,parts),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}:
  BandConcatExecution.PartsObservation{+sources,+equation} = observation
  %equation : {BandConcatExecution.tensor_contents(~Element,Layout.join_compatible(~Element,shape,axis,fill,workers,BandConcat.discover_compatible(~Element,_))) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,_),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}
  join_from_compatible_packed(~Element,sources,shape,axis,fill,workers,geometry,count,layout,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equation) : {compatible_flag(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,_))) && Bool.not(Shapes.has_zero(shape)) == True{} : Bool} enabled,
    geometry_selected,count_selected,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equation) : {discovery_model(~Element,BandConcatExecution.parts_contents(~Element,_)) == Some{layout} : Maybe<&2,Bands.Plan>} layout_selected)

def join_from_compatible(~Element: Data,parts: Tensors.Parts<Element>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  enabled: {compatible_flag(~Element,compatible_model(~Element,BandConcatExecution.parts_contents(~Element,parts))) && Bool.not(Shapes.has_zero(shape)) == True{} : Bool},
  geometry_selected: {BandConcat.geometry(shape,axis) == Some{geometry} : Maybe<&2,Bands.Geometry>},count_selected: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  layout_selected: {discovery_model(~Element,BandConcatExecution.parts_contents(~Element,parts)) == Some{layout} : Maybe<&2,Bands.Plan>}) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.join_compatible(~Element,shape,axis,fill,workers,BandConcat.discover_compatible(~Element,parts))) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,parts),shape,fill,workers) : BandConcatExecution.TensorContent<Element>}:
  join_from_compatible_observed(~Element,parts,shape,axis,fill,workers,geometry,count,layout,BandConcatExecution.observe_parts(~Element,parts),enabled,geometry_selected,count_selected,layout_selected)

def initial_summary(+first: List<&2,U32>,axis: Nat) -> Layout.Survey:
  Layout.Survey{Nat.is_lt(axis,List.length(&2,U32,first)),0,Tensors.Ready{},1}

def native_cursor_packed(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>,+geometry: Bands.Geometry,+output: Model.Content<Element>,+rank: Rank,+channels: U32,+height: U32,+width: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  accepted: {survey_valid(observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),initial_summary(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)))))) == True{} : Bool}) ->
  {BandConcatExecution.state_channel(~Element,BandConcatExecution.parts_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),geometry,(output,0))) ==
    survey_total(observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),initial_summary(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)))))) : U32}:
  Equal.trans(U32,BandConcatExecution.state_channel(~Element,BandConcatExecution.parts_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),geometry,(output,0))),BandConcatExecution.channel_cursor(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),0),
    survey_total(observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),initial_summary(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)))))),BandConcatExecution.parts_channel(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),geometry,(output,0)),
    cursor_word(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),0,limit,survey_total(observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),initial_summary(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)))))),
      native_channel_total_packed(~Element,parts,rank,channels,height,width,initial_summary(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank))),limit,maximum,accepted)))

def native_cursor_observed(~Element: Data,-parts: Tensors.Parts<Element>,+geometry: Bands.Geometry,+output: Model.Content<Element>,+rank: Rank,+channels: U32,+height: U32,+width: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  observation: BandConcatExecution.PartsObservation<Element,parts>,accepted: {survey_valid(observed_summary(~Element,Layout.survey(~Element,parts,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),initial_summary(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)))))) == True{} : Bool}) ->
  {BandConcatExecution.state_channel(~Element,BandConcatExecution.parts_model(~Element,BandConcatExecution.parts_contents(~Element,parts),geometry,(output,0))) == survey_total(observed_summary(~Element,Layout.survey(~Element,parts,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),initial_summary(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)))))) : U32}:
  BandConcatExecution.PartsObservation{+sources,+equation} = observation
  %equation : {BandConcatExecution.state_channel(~Element,BandConcatExecution.parts_model(~Element,BandConcatExecution.parts_contents(~Element,_),geometry,(output,0))) == survey_total(observed_summary(~Element,Layout.survey(~Element,_,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),initial_summary(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)))))) : U32}
  native_cursor_packed(~Element,sources,geometry,output,rank,channels,height,width,limit,maximum,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equation) : {survey_valid(observed_summary(~Element,Layout.survey(~Element,_,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),initial_summary(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)))))) == True{} : Bool} accepted)

def native_cursor(~Element: Data,parts: Tensors.Parts<Element>,+geometry: Bands.Geometry,+output: Model.Content<Element>,+rank: Rank,+channels: U32,+height: U32,+width: U32,+limit: U32,+maximum: {limit == 4294967295 : U32},
  accepted: {survey_valid(observed_summary(~Element,Layout.survey(~Element,parts,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),initial_summary(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)))))) == True{} : Bool}) ->
  {BandConcatExecution.state_channel(~Element,BandConcatExecution.parts_model(~Element,BandConcatExecution.parts_contents(~Element,parts),geometry,(output,0))) == survey_total(observed_summary(~Element,Layout.survey(~Element,parts,shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)),initial_summary(shape_of(rank,channels,height,width),U32.to_nat(axis_of(rank)))))) : U32}:
  native_cursor_observed(~Element,parts,geometry,output,rank,channels,height,width,limit,maximum,BandConcatExecution.observe_parts(~Element,parts),accepted)

def status_of(summary: Layout.Survey) -> Tensors.Status:
  Layout.Survey{valid,total,status,workers} = summary
  status

def target_shape(+first: List<&2,U32>,+axis: U32,summary: Layout.Survey) -> List<&2,U32>:
  Layout.with_extent(first,U32.to_nat(axis),survey_total(summary))

type BandSurveyFields<-Element: Data,-parts: BandConcatExecution.PartContents<Element>,-first: List<&2,U32>,-axis: U32,-summary: Layout.Survey,-geometry: Bands.Geometry,-count: U32,-layout: Bands.Plan> is Data:
  BandSurveyFields{valid: {survey_valid(summary) == True{} : Bool},status: {status_of(summary) == Tensors.Ready{} : Tensors.Status},
    geometry: {BandConcat.geometry(target_shape(first,axis,summary),axis) == Some{geometry} : Maybe<&2,Bands.Geometry>},
    counted: {Tensors.count_of(target_shape(first,axis,summary)) == Some{count} : Maybe<&2,U32>},
    layout: {discovery_model(~Element,parts) == Some{layout} : Maybe<&2,Bands.Plan>},
    compatible: {compatible_flag(~Element,compatible_model(~Element,parts)) && Bool.not(Shapes.has_zero(target_shape(first,axis,summary))) == True{} : Bool}}

def BandSurvey(~Element: Data,+first: List<&2,U32>,+axis: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,owners: BandConcatExecution.PartContents<Element> & Layout.Survey) -> Data:
  (parts,summary) = owners
  BandSurveyFields<Element,parts,first,axis,summary,geometry,count,layout>

def joined_expected(~Element: Data,+first: List<&2,U32>,+axis: U32,+fill: Element,+geometry: Bands.Geometry,+layout: Bands.Plan,owners: BandConcatExecution.PartContents<Element> & Layout.Survey) -> BandConcatExecution.TensorContent<Element>:
  (parts,summary) = owners
  Layout.Survey{valid,total,status,workers} = summary
  BandConcatExecution.join_model(~Element,layout,geometry,parts,Layout.with_extent(first,U32.to_nat(axis),total),fill,workers)

def joined_band(~Element: Data,owners: Tensors.Parts<Element> & Layout.Survey,+first: List<&2,U32>,+axis: U32,+fill: Element,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  evidence: BandSurvey(~Element,first,axis,geometry,count,layout,survey_contents(~Element,owners))) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.joined(~Element,first,axis,fill,owners)) == joined_expected(~Element,first,axis,fill,geometry,layout,survey_contents(~Element,owners)) : BandConcatExecution.TensorContent<Element>}:
  (parts,+summary) = owners
  Layout.Survey{+valid,+total,+status,+workers} = summary
  BandSurveyFields{valid_guard,status_guard,geometry_guard,count_guard,layout_guard,compatible_guard} = evidence
  %Equal.sym(Tensors.Status,status,Tensors.Ready{},status_guard) :
    {BandConcatExecution.tensor_contents(~Element,Layout.join_status(~Element,_,valid,first,Layout.with_extent(first,U32.to_nat(axis),total),axis,fill,workers,parts)) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,parts),Layout.with_extent(first,U32.to_nat(axis),total),fill,workers) : BandConcatExecution.TensorContent<Element>}
  %Equal.sym(Bool,valid,True{},valid_guard) :
    {BandConcatExecution.tensor_contents(~Element,Layout.join_status(~Element,Tensors.Ready{},_,first,Layout.with_extent(first,U32.to_nat(axis),total),axis,fill,workers,parts)) == BandConcatExecution.join_model(~Element,layout,geometry,BandConcatExecution.parts_contents(~Element,parts),Layout.with_extent(first,U32.to_nat(axis),total),fill,workers) : BandConcatExecution.TensorContent<Element>}
  join_from_compatible(~Element,parts,Layout.with_extent(first,U32.to_nat(axis),total),axis,fill,workers,geometry,count,layout,compatible_guard,geometry_guard,count_guard,layout_guard)

def concat_first_band_packed(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: U32,+fill: Element,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  evidence: BandSurvey(~Element,first,axis,geometry,count,layout,(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis)))))) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,BandConcatExecution.parts_owner_pack(~Element,parts),first,fill)) ==
    joined_expected(~Element,first,axis,fill,geometry,layout,(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis))))) : BandConcatExecution.TensorContent<Element>}:
  +survey_equal = native_survey_packed(~Element,parts,first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis)))
  Equal.trans(BandConcatExecution.TensorContent<Element>,BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,BandConcatExecution.parts_owner_pack(~Element,parts),first,fill)),
    joined_expected(~Element,first,axis,fill,geometry,layout,survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis))))),
    joined_expected(~Element,first,axis,fill,geometry,layout,(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis))))),
    joined_band(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis))),first,axis,fill,geometry,count,layout,
      %Equal.sym(BandConcatExecution.PartContents<Element> & Layout.Survey,survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis)))),
        (BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis)))),survey_equal) : BandSurvey(~Element,first,axis,geometry,count,layout,_) evidence),
    Equal.cong(BandConcatExecution.PartContents<Element> & Layout.Survey,BandConcatExecution.TensorContent<Element>,owners => joined_expected(~Element,first,axis,fill,geometry,layout,owners),
      survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis)))),
      (BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis)))),survey_equal))

def concat_first_band_observed(~Element: Data,-parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: U32,+fill: Element,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  observation: BandConcatExecution.PartsObservation<Element,parts>,evidence: BandSurvey(~Element,first,axis,geometry,count,layout,(BandConcatExecution.parts_contents(~Element,parts),survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis)))))) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,parts,first,fill)) == joined_expected(~Element,first,axis,fill,geometry,layout,(BandConcatExecution.parts_contents(~Element,parts),survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis))))) : BandConcatExecution.TensorContent<Element>}:
  BandConcatExecution.PartsObservation{+sources,+equation} = observation
  %equation : {BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,_,first,fill)) == joined_expected(~Element,first,axis,fill,geometry,layout,(BandConcatExecution.parts_contents(~Element,_),survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis))))) : BandConcatExecution.TensorContent<Element>}
  concat_first_band_packed(~Element,sources,first,axis,fill,geometry,count,layout,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equation) : BandSurvey(~Element,first,axis,geometry,count,layout,(BandConcatExecution.parts_contents(~Element,_),survey_model(~Element,BandConcatExecution.parts_contents(~Element,_),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis))))) evidence)

def concat_first_band(~Element: Data,parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: U32,+fill: Element,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  evidence: BandSurvey(~Element,first,axis,geometry,count,layout,(BandConcatExecution.parts_contents(~Element,parts),survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis)))))) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.concat_first(~Element,axis,parts,first,fill)) == joined_expected(~Element,first,axis,fill,geometry,layout,(BandConcatExecution.parts_contents(~Element,parts),survey_model(~Element,BandConcatExecution.parts_contents(~Element,parts),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis))))) : BandConcatExecution.TensorContent<Element>}:
  concat_first_band_observed(~Element,parts,first,axis,fill,geometry,count,layout,BandConcatExecution.observe_parts(~Element,parts),evidence)

def tensor_fill(~Element: Data,input: BandConcatExecution.TensorContent<Element>) -> Element:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}: fill
    case BandConcatExecution.BandedContent{bands,geometry,shape,fill,status,workers}: fill

def lead_model(~Element: Data,parts: BandConcatExecution.PartContents<Element>) -> BandConcatExecution.PartContents<Element> & (List<&2,U32> & Element):
  match parts:
    case BandConcatExecution.SingleContent{+input}: (BandConcatExecution.SingleContent{input},{(BandConcatExecution.tensor_shape(~Element,input),tensor_fill(~Element,input)) : List<&2,U32> & Element})
    case BandConcatExecution.ConsContent{+input,rest}: (BandConcatExecution.ConsContent{input,rest},{(BandConcatExecution.tensor_shape(~Element,input),tensor_fill(~Element,input)) : List<&2,U32> & Element})

def lead_contents(~Element: Data,owners: Tensors.Parts<Element> & (List<&2,U32> & Element)) -> BandConcatExecution.PartContents<Element> & (List<&2,U32> & Element):
  (parts,first) = owners
  (BandConcatExecution.parts_contents(~Element,parts),{first : List<&2,U32> & Element})

def lead(~Element: Data,parts: Tensors.Parts<Element>) ->
  {lead_contents(~Element,Layout.lead(~Element,parts)) == lead_model(~Element,BandConcatExecution.parts_contents(~Element,parts)) : BandConcatExecution.PartContents<Element> & (List<&2,U32> & Element)}:
  match parts:
    case Tensors.Single{input}:
      match input:
        case Tensors.Tensor{buffer,shape,fill,status,workers}: {==}
        case Tensors.Banded{bands,geometry,shape,fill,status,workers}: {==}
    case Tensors.Cons{input,rest}:
      match input:
        case Tensors.Tensor{buffer,shape,fill,status,workers}: {==}
        case Tensors.Banded{bands,geometry,shape,fill,status,workers}: {==}

def led_guards(~Element: Data,+axis: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,owners: BandConcatExecution.PartContents<Element> & (List<&2,U32> & Element)) -> Data:
  (+parts,first) = owners
  (+shape,fill) = first
  BandSurvey(~Element,shape,axis,geometry,count,layout,(parts,survey_model(~Element,parts,shape,U32.to_nat(axis),initial_summary(shape,U32.to_nat(axis)))))

def led_expected(~Element: Data,+axis: U32,+geometry: Bands.Geometry,+layout: Bands.Plan,owners: BandConcatExecution.PartContents<Element> & (List<&2,U32> & Element)) -> BandConcatExecution.TensorContent<Element>:
  (+parts,first) = owners
  (+shape,fill) = first
  joined_expected(~Element,shape,axis,fill,geometry,layout,(parts,survey_model(~Element,parts,shape,U32.to_nat(axis),initial_summary(shape,U32.to_nat(axis)))))

def concat_led_band(~Element: Data,owners: Tensors.Parts<Element> & (List<&2,U32> & Element),+axis: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  evidence: led_guards(~Element,axis,geometry,count,layout,lead_contents(~Element,{owners : Tensors.Parts<Element> & (List<&2,U32> & Element)}))) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.concat_led(~Element,axis,{owners : Tensors.Parts<Element> & (List<&2,U32> & Element)})) == led_expected(~Element,axis,geometry,layout,lead_contents(~Element,{owners : Tensors.Parts<Element> & (List<&2,U32> & Element)})) : BandConcatExecution.TensorContent<Element>}:
  (parts,first) = owners
  (+shape,fill) = first
  +fill = {fill : Element}
  concat_first_band(~Element,parts,shape,axis,fill,geometry,count,layout,evidence)

def concat_band_packed(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>,+axis: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  evidence: led_guards(~Element,axis,geometry,count,layout,lead_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))))) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),axis)) == led_expected(~Element,axis,geometry,layout,lead_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)))) : BandConcatExecution.TensorContent<Element>}:
  +lead_equal = lead(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))
  Equal.trans(BandConcatExecution.TensorContent<Element>,BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),axis)),led_expected(~Element,axis,geometry,layout,lead_contents(~Element,Layout.lead(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)))),led_expected(~Element,axis,geometry,layout,lead_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)))),
    concat_led_band(~Element,Layout.lead(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),axis,geometry,count,layout,
      %Equal.sym(BandConcatExecution.PartContents<Element> & (List<&2,U32> & Element),lead_contents(~Element,Layout.lead(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),lead_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),lead_equal) : led_guards(~Element,axis,geometry,count,layout,_) evidence),
    Equal.cong(BandConcatExecution.PartContents<Element> & (List<&2,U32> & Element),BandConcatExecution.TensorContent<Element>,owners => led_expected(~Element,axis,geometry,layout,owners),lead_contents(~Element,Layout.lead(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),lead_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts))),lead_equal))

def concat_band_observed(~Element: Data,-parts: Tensors.Parts<Element>,+axis: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  observation: BandConcatExecution.PartsObservation<Element,parts>,evidence: led_guards(~Element,axis,geometry,count,layout,lead_model(~Element,BandConcatExecution.parts_contents(~Element,parts)))) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,parts,axis)) == led_expected(~Element,axis,geometry,layout,lead_model(~Element,BandConcatExecution.parts_contents(~Element,parts))) : BandConcatExecution.TensorContent<Element>}:
  BandConcatExecution.PartsObservation{+sources,+equation} = observation
  %equation : {BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,_,axis)) == led_expected(~Element,axis,geometry,layout,lead_model(~Element,BandConcatExecution.parts_contents(~Element,_))) : BandConcatExecution.TensorContent<Element>}
  concat_band_packed(~Element,sources,axis,geometry,count,layout,
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,sources),parts,equation) : led_guards(~Element,axis,geometry,count,layout,lead_model(~Element,BandConcatExecution.parts_contents(~Element,_))) evidence)

def concat_band(~Element: Data,parts: Tensors.Parts<Element>,+axis: U32,+geometry: Bands.Geometry,+count: U32,+layout: Bands.Plan,
  evidence: led_guards(~Element,axis,geometry,count,layout,lead_model(~Element,BandConcatExecution.parts_contents(~Element,parts)))) ->
  {BandConcatExecution.tensor_contents(~Element,Layout.concat(~Element,parts,axis)) == led_expected(~Element,axis,geometry,layout,lead_model(~Element,BandConcatExecution.parts_contents(~Element,parts))) : BandConcatExecution.TensorContent<Element>}:
  concat_band_observed(~Element,parts,axis,geometry,count,layout,BandConcatExecution.observe_parts(~Element,parts),evidence)

# Exact checked remainder totals put each preceding cursor plus its source
# channels below the final output channels. The public survey supplies these
# fields; no write-address or concat-value conclusion appears as a premise.
def remaining_channel_room(+channel: U32,+source_channels: U32,+output_channels: U32,+remaining: Nat,+limit: U32,
  step_exact: {U32.to_nat((channel + source_channels : U32)) == Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)) : Nat},
  remainder: TotalFields<output_channels,(channel + source_channels : U32),remaining,limit>) ->
  {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}:
  TotalFields{+exact,room} = remainder
  %Equal.sym(Nat,U32.to_nat(output_channels),Nat.add(U32.to_nat((channel + source_channels : U32)),remaining),exact) : {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),_) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((channel + source_channels : U32)),Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),step_exact) : {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),Nat.add(_,remaining)) == True{} : Bool}
  Order.le_add(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),remaining)

def range_target_leaf(~Element: Data,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+first: U32,+rows: U32,
  +limits: Machine.Limits,+global_capacity: U32,+output: Buffer.Model<Element>,
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,global_capacity>,
  metadata: BandShape.Leaf(Bands.Geometry{images,output_channels,height,width},first,rows,Buffer.length(~Element,output)),ready: Buffer.Ready<Element,output>,
  source_positive: {Nat.is_lt(0n,U32.to_nat(source_channels)) == True{} : Bool},
  channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}) ->
  Shape.Both<BandRange.Target(BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},channel,rows)},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
    BandShape.Leaf(Bands.Geometry{images,source_channels,height,width},first,rows,BandConcatExecution.channel_length(BandRange.Request{Bands.Geometry{images,source_channels,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},channel,rows)}))>:
  Buffer.Model{+tree,+length,+certificate} = output
  BandGather.Domain{capacity_ok,output_positive,height_positive,+width_positive,machine_room,allocation_room} = domain
  BandShape.Fields{+local,+rows_fit,+length_exact} = metadata
  Buffer.Ready{+depth_ok,+room} = ready
  +capacity = Tree.capacity(Element,tree)
  +capacity_bound = Buffer.capacity_bound(~Element,output,depth_ok)
  +plane = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +source_count = Nat.mul(U32.to_nat(source_channels),plane)
  +offset_count = Nat.mul(U32.to_nat(channel),plane)
  +output_room = {
    %length_exact : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room : {Nat.is_le(Nat.mul(U32.to_nat(output_channels),plane),U32.to_nat(capacity)) == True{} : Bool}}
  +storage_room = {
    %RowMajor.distrib(U32.to_nat(channel),U32.to_nat(source_channels),plane) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(Nat.mul(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),plane),Nat.mul(U32.to_nat(output_channels),plane),U32.to_nat(capacity),RowMajor.mul_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels),plane,channel_room),output_room)
    : {Nat.is_le(Nat.add(offset_count,source_count),U32.to_nat(capacity)) == True{} : Bool}}
  +offset_room = Order.transitive(offset_count,Nat.add(offset_count,source_count),U32.to_nat(capacity),Order.le_add(offset_count,source_count),storage_room)
  +offset_exact = Equal.trans(Nat,U32.to_nat(BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},channel,rows)),Nat.mul(Nat.mul(U32.to_nat(channel),U32.to_nat(rows)),U32.to_nat(width)),offset_count,
    Encoded.multiply_three(channel,rows,width,capacity,capacity_bound,width_positive,
      %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(channel),U32.to_nat(rows)),U32.to_nat(width)),offset_count,RowMajor.assoc(U32.to_nat(channel),U32.to_nat(rows),U32.to_nat(width))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} offset_room),
    RowMajor.assoc(U32.to_nat(channel),U32.to_nat(rows),U32.to_nat(width)))
  Shape.Both{BandRange.TargetFields{Limits.capacity_fits(limits),capacity_bound,source_positive,width_positive,
      Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(Bands.Geometry{images,output_channels,height,width}),U32.to_nat(Machine.capacity(limits)),rows_fit,BandGather.rows_room(Bands.Geometry{images,output_channels,height,width},limits,global_capacity,domain)),
      %Equal.sym(Nat,U32.to_nat(BandConcat.channel_offset(Bands.Geometry{images,source_channels,height,width},channel,rows)),offset_count,offset_exact) : {Nat.is_le(Nat.add(_,source_count),U32.to_nat(capacity)) == True{} : Bool} storage_room},
    BandShape.Fields{local,rows_fit,{==}}}

def range_targets(~Element: Data,+images: U32,+source_channels: U32,+output_channels: U32,+height: U32,+width: U32,+channel: U32,+limits: Machine.Limits,+global_capacity: U32,output: Model.Model<Element>,
  +domain: BandGather.Domain<Bands.Geometry{images,output_channels,height,width},limits,global_capacity>,
  metadata: BandShape.Contents(~Element,Bands.Geometry{images,output_channels,height,width},Model.content(~Element,output)),ready: Model.Ready(~Element,output),
  +source_positive: {Nat.is_lt(0n,U32.to_nat(source_channels)) == True{} : Bool},
  +channel_room: {Nat.is_le(Nat.add(U32.to_nat(channel),U32.to_nat(source_channels)),U32.to_nat(output_channels)) == True{} : Bool}) ->
  BandConcatExecution.RangeTargets(~Element,Bands.Geometry{images,source_channels,height,width},channel,limits,output):
  match output:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      range_target_leaf(~Element,images,source_channels,output_channels,height,width,channel,first,rows,limits,global_capacity,Buffer.Model{tree,length,certificate},domain,metadata,ready,source_positive,channel_room)
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{range_targets(~Element,images,source_channels,output_channels,height,width,channel,limits,global_capacity,left,domain,left_metadata,left_ready,source_positive,channel_room),
        range_targets(~Element,images,source_channels,output_channels,height,width,channel,limits,global_capacity,right,domain,right_metadata,right_ready,source_positive,channel_room)}

# A zero-channel part performs no stores. This branch preserves the whole
# old output, including every prefix, without a positive-channel Domain.
def zero_count(+images: U32,+height: U32,+width: U32,+rows: U32) -> {Bands.local_count(Bands.Geometry{images,0,height,width},rows) == 0 : U32}:
  %Equal.sym(U32,(0 * rows : U32),0,Arithmetic.multiply_zero_left(rows)) : {(_ * width : U32) == 0 : U32}
  Arithmetic.multiply_zero_left(width)

def empty_range_writes(~Element: Data,+images: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+offset: U32,source: Model.Content<Element>) ->
  {BandRange.content_writes(~Element,BandRange.Request{Bands.Geometry{images,0,height,width},first,rows,offset},source) == [] : List<&2,Writes.Assignment<Element>>}:
  match source:
    case Model.RowBand{Buffer.Content{tree,length},source_first,source_rows}: {==}
    case Model.Joined{+left,+right}:
      %Equal.sym(List<&2,Writes.Assignment<Element>>,BandRange.content_writes(~Element,BandRange.Request{Bands.Geometry{images,0,height,width},first,rows,offset},left),[],empty_range_writes(~Element,images,height,width,first,rows,offset,left)) :
        {List.append(&2,Writes.Assignment<Element>,_,BandRange.content_writes(~Element,BandRange.Request{Bands.Geometry{images,0,height,width},first,rows,offset},right)) == [] : List<&2,Writes.Assignment<Element>>}
      empty_range_writes(~Element,images,height,width,first,rows,offset,right)

def empty_banded_tree(~Element: Data,+images: U32,+height: U32,+width: U32,+channel: U32,+source: Model.Content<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,output: Model.Content<Element>) ->
  {BandConcatExecution.tree_model(~Element,Bands.Geometry{images,0,height,width},channel,BandConcatExecution.BandedContent{source,Bands.Geometry{images,0,height,width},shape,fill,status,workers},output) == output : Model.Content<Element>}:
  match output:
    case Model.RowBand{Buffer.Content{+tree,+length},+first,+rows}:
      %Equal.sym(List<&2,Writes.Assignment<Element>>,BandRange.content_writes(~Element,BandRange.Request{Bands.Geometry{images,0,height,width},first,rows,BandConcat.channel_offset(Bands.Geometry{images,0,height,width},channel,rows)},source),[],
        empty_range_writes(~Element,images,height,width,first,rows,BandConcat.channel_offset(Bands.Geometry{images,0,height,width},channel,rows),source)) : {Model.RowBand{Buffer.Content{Writes.pure(Element,_,tree),length},first,rows} == Model.RowBand{Buffer.Content{tree,length},first,rows} : Model.Content<Element>}
      {==}
    case Model.Joined{+left,+right}:
      %Equal.sym(Model.Content<Element>,BandConcatExecution.tree_model(~Element,Bands.Geometry{images,0,height,width},channel,BandConcatExecution.BandedContent{source,Bands.Geometry{images,0,height,width},shape,fill,status,workers},left),left,empty_banded_tree(~Element,images,height,width,channel,source,shape,fill,status,workers,left)) :
        {Model.Joined{_,BandConcatExecution.tree_model(~Element,Bands.Geometry{images,0,height,width},channel,BandConcatExecution.BandedContent{source,Bands.Geometry{images,0,height,width},shape,fill,status,workers},right)} == Model.Joined{left,right} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,BandConcatExecution.tree_model(~Element,Bands.Geometry{images,0,height,width},channel,BandConcatExecution.BandedContent{source,Bands.Geometry{images,0,height,width},shape,fill,status,workers},right),right,empty_banded_tree(~Element,images,height,width,channel,source,shape,fill,status,workers,right)) : {Model.Joined{left,_} == Model.Joined{left,right} : Model.Content<Element>}
      {==}

def empty_dense_tree(~Element: Data,+images: U32,+height: U32,+width: U32,+channel: U32,+input: Buffer.Content<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,output: Model.Content<Element>) ->
  {BandConcatExecution.tree_model(~Element,Bands.Geometry{images,0,height,width},channel,BandConcatExecution.DenseContent{input,shape,fill,status,workers},output) == output : Model.Content<Element>}:
  Buffer.Content{+input_tree,+input_length} = input
  match output:
    case Model.RowBand{Buffer.Content{tree,length},first,rows}: {==}
    case Model.Joined{+left,+right}:
      %Equal.sym(Model.Content<Element>,BandConcatExecution.tree_model(~Element,Bands.Geometry{images,0,height,width},channel,BandConcatExecution.DenseContent{input,shape,fill,status,workers},left),left,empty_dense_tree(~Element,images,height,width,channel,input,shape,fill,status,workers,left)) :
        {Model.Joined{_,BandConcatExecution.tree_model(~Element,Bands.Geometry{images,0,height,width},channel,BandConcatExecution.DenseContent{input,shape,fill,status,workers},right)} == Model.Joined{left,right} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,BandConcatExecution.tree_model(~Element,Bands.Geometry{images,0,height,width},channel,BandConcatExecution.DenseContent{input,shape,fill,status,workers},right),right,empty_dense_tree(~Element,images,height,width,channel,input,shape,fill,status,workers,right)) : {Model.Joined{left,_} == Model.Joined{left,right} : Model.Content<Element>}
      {==}

def empty_aligned(~Element: Data,+images: U32,+height: U32,+width: U32,+channel: U32,input: Model.Content<Element>,output: Model.Content<Element>) ->
  {BandConcatExecution.aligned_model(~Element,Bands.Geometry{images,0,height,width},channel,input,output) == output : Model.Content<Element>}:
  match input output:
    case Model.RowBand{Buffer.Content{input,input_length},first,rows} Model.RowBand{Buffer.Content{+output,+output_length},+out_first,+out_rows}:
      %Equal.sym(U32,Bands.local_count(Bands.Geometry{images,0,height,width},out_rows),0,zero_count(images,height,width,out_rows)) :
        {Model.RowBand{BandGather.output(~Element,StorageCopyLaws.model(~Element,_,0,BandConcat.channel_offset(Bands.Geometry{images,0,height,width},channel,out_rows),BandGather.owners(~Element,Buffer.Content{input,input_length},Buffer.Content{output,output_length}))),out_first,out_rows} == Model.RowBand{Buffer.Content{output,output_length},out_first,out_rows} : Model.Content<Element>}
      {==}
    case Model.Joined{+left,+right} Model.Joined{+out_left,+out_right}:
      %Equal.sym(Model.Content<Element>,BandConcatExecution.aligned_model(~Element,Bands.Geometry{images,0,height,width},channel,left,out_left),out_left,empty_aligned(~Element,images,height,width,channel,left,out_left)) :
        {Model.Joined{_,BandConcatExecution.aligned_model(~Element,Bands.Geometry{images,0,height,width},channel,right,out_right)} == Model.Joined{out_left,out_right} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,BandConcatExecution.aligned_model(~Element,Bands.Geometry{images,0,height,width},channel,right,out_right),out_right,empty_aligned(~Element,images,height,width,channel,right,out_right)) : {Model.Joined{out_left,_} == Model.Joined{out_left,out_right} : Model.Content<Element>}
      {==}
    case Model.RowBand{input,first,rows} Model.Joined{left,right}: {==}
    case Model.Joined{left,right} Model.RowBand{output,first,rows}: {==}

def empty_selected(~Element: Data,same: Bool,+images: U32,+height: U32,+width: U32,+channel: U32,+input: Model.Content<Element>,+output: Model.Content<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32) ->
  {BandConcatExecution.selected_model(~Element,same,Bands.Geometry{images,0,height,width},channel,input,output,shape,fill,status,workers) == output : Model.Content<Element>}:
  match same:
    case True{}: empty_aligned(~Element,images,height,width,channel,input,output)
    case False{}: empty_banded_tree(~Element,images,height,width,channel,input,shape,fill,status,workers,output)

def empty_storage(~Element: Data,+images: U32,+height: U32,+width: U32,+channel: U32,input: BandConcatExecution.TensorContent<Element>,+output: Model.Content<Element>) ->
  {BandConcatExecution.storage_model(~Element,Bands.Geometry{images,0,height,width},channel,input,output) == output : Model.Content<Element>}:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}: empty_dense_tree(~Element,images,height,width,channel,buffer,shape,fill,status,workers,output)
    case BandConcatExecution.BandedContent{+bands,geometry,+shape,+fill,+status,+workers}:
      empty_selected(~Element,Bands.same_plan(Model.content_layout(~Element,bands),Model.content_layout(~Element,output)),images,height,width,channel,bands,output,shape,fill,status,workers)

def append_zero_channels_packed(~Element: Data,+images: U32,+height: U32,+width: U32,+channel: U32,+workers: U32,+input: BandConcatExecution.TensorOwnerModel<Element>,+output: Model.Model<Element>) ->
  {Model.contents(~Element,BandConcat.append_storage(~Element,Bands.Geometry{images,0,height,width},channel,workers,BandConcatExecution.tensor_owner_pack(~Element,input),Model.pack(~Element,output))) == Model.contents(~Element,Model.pack(~Element,output)) : Model.Content<Element>}:
  Equal.trans(Model.Content<Element>,Model.contents(~Element,BandConcat.append_storage(~Element,Bands.Geometry{images,0,height,width},channel,workers,BandConcatExecution.tensor_owner_pack(~Element,input),Model.pack(~Element,output))),BandConcatExecution.storage_model(~Element,Bands.Geometry{images,0,height,width},channel,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),Model.contents(~Element,Model.pack(~Element,output))),Model.contents(~Element,Model.pack(~Element,output)),
    BandConcatExecution.append_storage(~Element,Bands.Geometry{images,0,height,width},channel,workers,BandConcatExecution.tensor_owner_pack(~Element,input),Model.pack(~Element,output)),empty_storage(~Element,images,height,width,channel,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),Model.contents(~Element,Model.pack(~Element,output))))

def append_zero_channels_observed(~Element: Data,+images: U32,+height: U32,+width: U32,+channel: U32,+workers: U32,-input: Tensors.Tensor<Element>,-output: Bands.Bands(~Element),
  input_observation: BandConcatExecution.TensorObservation<Element,input>,output_observation: Model.Observation<Element,output>) ->
  {Model.contents(~Element,BandConcat.append_storage(~Element,Bands.Geometry{images,0,height,width},channel,workers,input,output)) == Model.contents(~Element,output) : Model.Content<Element>}:
  BandConcatExecution.TensorObservation{+source,source_equal} = input_observation
  Model.Observation{+target,target_equal} = output_observation
  %source_equal : {Model.contents(~Element,BandConcat.append_storage(~Element,Bands.Geometry{images,0,height,width},channel,workers,_,output)) == Model.contents(~Element,output) : Model.Content<Element>}
  %target_equal : {Model.contents(~Element,BandConcat.append_storage(~Element,Bands.Geometry{images,0,height,width},channel,workers,BandConcatExecution.tensor_owner_pack(~Element,source),_)) == Model.contents(~Element,_) : Model.Content<Element>}
  append_zero_channels_packed(~Element,images,height,width,channel,workers,source,target)

def append_zero_channels(~Element: Data,+images: U32,+height: U32,+width: U32,+channel: U32,+workers: U32,input: Tensors.Tensor<Element>,output: Bands.Bands(~Element)) ->
  {Model.contents(~Element,BandConcat.append_storage(~Element,Bands.Geometry{images,0,height,width},channel,workers,input,output)) == Model.contents(~Element,output) : Model.Content<Element>}:
  append_zero_channels_observed(~Element,images,height,width,channel,workers,input,output,BandConcatExecution.observe_tensor(~Element,input),Model.observe(~Element,output))

# Final-owner range stores retain depth, capacity and logical length. This
# structural result needs no bounds on stored values or algebraic assumptions.
def copied_ready(~Element: Data,+count: U32,+offset: U32,+input: Buffer.Content<Element>,+output: Buffer.Content<Element>,ready: Buffer.ContentReady<Element,output>) ->
  Buffer.ContentReady<Element,BandGather.output(~Element,StorageCopyLaws.model(~Element,count,0,offset,BandGather.owners(~Element,input,output)))>:
  Buffer.Content{+input_tree,input_length} = input
  Buffer.Content{+output_tree,+output_length} = output
  Writes.content_ready(~Element,Contiguous.writes(~Element,U32.to_nat(count),input_tree,0,0,offset),output_tree,output_length,ready)

def model_ready(~Element: Data,+geometry: Bands.Geometry,+channel: U32,input: Model.Model<Element>,output: Model.Model<Element>,ready: Model.ContentReady(~Element,Model.content(~Element,output))) ->
  Model.ContentReady(~Element,BandConcatExecution.pipeline_model(~Element,geometry,channel,input,output)):
  match input output:
    case Model.Leaf{+input,first,rows} Model.Leaf{+output,out_first,+out_rows}:
      copied_ready(~Element,Bands.local_count(geometry,out_rows),BandConcat.channel_offset(geometry,channel,out_rows),Buffer.content(~Element,input),Buffer.content(~Element,output),ready)
    case Model.Fork{+left,+right} Model.Fork{+out_left,+out_right}:
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{model_ready(~Element,geometry,channel,left,out_left,left_ready),model_ready(~Element,geometry,channel,right,out_right,right_ready)}
    case Model.Leaf{input,first,rows} Model.Fork{left,right}: ready
    case Model.Fork{left,right} Model.Leaf{output,first,rows}: ready

def append_ready(~Element: Data,+parallel: Bool,+geometry: Bands.Geometry,+channel: U32,+input: Model.Model<Element>,+output: Model.Model<Element>,ready: Model.ContentReady(~Element,Model.content(~Element,output))) ->
  Model.ContentReady(~Element,Model.contents(~Element,BandConcat.append_aligned(~Element,parallel,geometry,channel,Model.pack(~Element,input),Model.pack(~Element,output)))):
  %Equal.sym(Model.Content<Element>,
    Model.contents(~Element,BandConcat.append_aligned(~Element,parallel,geometry,channel,Model.pack(~Element,input),Model.pack(~Element,output))),
    BandConcatExecution.pipeline_model(~Element,geometry,channel,input,output),BandConcatExecution.append_aligned(~Element,parallel,geometry,channel,input,output)) : Model.ContentReady(~Element,_)
  model_ready(~Element,geometry,channel,input,output,ready)

# The final output's layout and metadata survive every aligned append. These
# facts compose with ContentReady; the Parts allocator remains a caller step.
def model_layout(~Element: Data,+geometry: Bands.Geometry,+channel: U32,input: Model.Model<Element>,output: Model.Model<Element>) ->
  {Model.content_layout(~Element,BandConcatExecution.pipeline_model(~Element,geometry,channel,input,output)) == Model.layout(~Element,output) : Bands.Plan}:
  match input output:
    case Model.Leaf{input,first,rows} Model.Leaf{output,out_first,out_rows}: {==}
    case Model.Fork{+left,+right} Model.Fork{+out_left,+out_right}:
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,BandConcatExecution.pipeline_model(~Element,geometry,channel,left,out_left)),Model.layout(~Element,out_left),model_layout(~Element,geometry,channel,left,out_left)) :
        {Bands.Split{_,Model.content_layout(~Element,BandConcatExecution.pipeline_model(~Element,geometry,channel,right,out_right))} == Bands.Split{Model.layout(~Element,out_left),Model.layout(~Element,out_right)} : Bands.Plan}
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,BandConcatExecution.pipeline_model(~Element,geometry,channel,right,out_right)),Model.layout(~Element,out_right),model_layout(~Element,geometry,channel,right,out_right)) :
        {Bands.Split{Model.layout(~Element,out_left),_} == Bands.Split{Model.layout(~Element,out_left),Model.layout(~Element,out_right)} : Bands.Plan}
      {==}
    case Model.Leaf{input,first,rows} Model.Fork{+left,+right}:
      Equal.sym(Bands.Plan,Model.layout(~Element,Model.Fork{left,right}),Model.content_layout(~Element,Model.content(~Element,Model.Fork{left,right})),Model.layout_matches_content(~Element,Model.Fork{left,right}))
    case Model.Fork{left,right} Model.Leaf{+output,+first,+rows}:
      Equal.sym(Bands.Plan,Model.layout(~Element,Model.Leaf{output,first,rows}),Model.content_layout(~Element,Model.content(~Element,Model.Leaf{output,first,rows})),Model.layout_matches_content(~Element,Model.Leaf{output,first,rows}))

def model_metadata(~Element: Data,+source_geometry: Bands.Geometry,+output_geometry: Bands.Geometry,+channel: U32,input: Model.Model<Element>,output: Model.Model<Element>,
  metadata: BandShape.Contents(~Element,output_geometry,Model.content(~Element,output))) ->
  BandShape.Contents(~Element,output_geometry,BandConcatExecution.pipeline_model(~Element,source_geometry,channel,input,output)):
  match input output:
    case Model.Leaf{Buffer.Model{input_tree,input_length,input_certificate},first,rows} Model.Leaf{Buffer.Model{output_tree,output_length,output_certificate},out_first,out_rows}:
      metadata
    case Model.Fork{+left,+right} Model.Fork{+out_left,+out_right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{model_metadata(~Element,source_geometry,output_geometry,channel,left,out_left,left_metadata),model_metadata(~Element,source_geometry,output_geometry,channel,right,out_right,right_metadata)}
    case Model.Leaf{input,first,rows} Model.Fork{left,right}: metadata
    case Model.Fork{left,right} Model.Leaf{output,first,rows}: metadata

def model_well_formed(~Element: Data,+source_geometry: Bands.Geometry,+output_geometry: Bands.Geometry,+channel: U32,+input: Model.Model<Element>,+output: Model.Model<Element>,
  evidence: BandShape.WellFormed<Element,output_geometry,Model.content(~Element,output)>) ->
  BandShape.WellFormed<Element,output_geometry,BandConcatExecution.pipeline_model(~Element,source_geometry,channel,input,output)>:
  BandShape.WellFormed{metadata,ready,coverage} = evidence
  BandShape.WellFormed{model_metadata(~Element,source_geometry,output_geometry,channel,input,output,metadata),model_ready(~Element,source_geometry,channel,input,output,ready),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,BandConcatExecution.pipeline_model(~Element,source_geometry,channel,input,output)),Model.layout(~Element,output),model_layout(~Element,source_geometry,channel,input,output)) : BandPlan.Coverage(_,0n,Inspection.total_rows(output_geometry))
    %Equal.sym(Bands.Plan,Model.layout(~Element,output),Model.content_layout(~Element,Model.content(~Element,output)),Model.layout_matches_content(~Element,output)) : BandPlan.Coverage(_,0n,Inspection.total_rows(output_geometry))
    coverage}

def append_well_formed(~Element: Data,+parallel: Bool,+source_geometry: Bands.Geometry,+output_geometry: Bands.Geometry,+channel: U32,+input: Model.Model<Element>,+output: Model.Model<Element>,
  evidence: BandShape.WellFormed<Element,output_geometry,Model.content(~Element,output)>) ->
  BandShape.WellFormed<Element,output_geometry,Model.contents(~Element,BandConcat.append_aligned(~Element,parallel,source_geometry,channel,Model.pack(~Element,input),Model.pack(~Element,output)))>:
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandConcat.append_aligned(~Element,parallel,source_geometry,channel,Model.pack(~Element,input),Model.pack(~Element,output))),
    BandConcatExecution.pipeline_model(~Element,source_geometry,channel,input,output),BandConcatExecution.append_aligned(~Element,parallel,source_geometry,channel,input,output)) : BandShape.WellFormed<Element,output_geometry,_>
  model_well_formed(~Element,source_geometry,output_geometry,channel,input,output,evidence)

# Different-plan concatenation also stores into the existing final output.
# Its actual range write program preserves allocation and logical length.
def written_ready(~Element: Data,+request: BandRange.Request,+source: Model.Content<Element>,output: Buffer.Content<Element>,ready: Buffer.ContentReady<Element,output>) ->
  Buffer.ContentReady<Element,BandRange.result(~Element,BandRange.written(~Element,request,source,output))>:
  Buffer.Content{+tree,+length} = output
  Writes.content_ready(~Element,BandRange.content_writes(~Element,request,source),tree,length,ready)

def banded_rows_ready(~Element: Data,+external_geometry: Bands.Geometry,+source_geometry: Bands.Geometry,+first: U32,+rows: U32,+channel: U32,
  +source: Model.Model<Element>,+output: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  ready: Buffer.Ready<Element,output>) ->
  Buffer.ContentReady<Element,BandConcatExecution.rows_output_content(~Element,BandConcat.rows(~Element,external_geometry,first,rows,channel,
    Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,workers},Buffer.pack(~Element,output)))>:
  +request = {BandRange.Request{source_geometry,first,rows,BandConcat.channel_offset(source_geometry,channel,rows)} : BandRange.Request}
  %Equal.sym(Buffer.Content<Element>,BandConcatExecution.rows_output_content(~Element,BandConcat.rows(~Element,external_geometry,first,rows,channel,
      Tensors.Banded{Model.pack(~Element,source),source_geometry,shape,fill,status,workers},Buffer.pack(~Element,output))),
    BandRange.result(~Element,BandRange.written(~Element,request,Model.content(~Element,source),Buffer.content(~Element,output))),
    BandConcatExecution.banded_rows(~Element,external_geometry,source_geometry,first,rows,channel,source,output,shape,fill,status,workers)) : Buffer.ContentReady<Element,_>
  written_ready(~Element,request,Model.content(~Element,source),Buffer.content(~Element,output),Buffer.content_ready(~Element,output,ready))
