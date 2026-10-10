# The values of the dense concat. The parts are written one after another
# by set_slice into a tensor that starts at the fill value; appended is that
# fold in the order of the loop, each part written through its slice of the
# output at the running start along the axis.
import Base
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_dense_contract.bend as Dense
import ./tensor_gather_values.bend as GatherValues
import ./tensor_set_slice_values.bend as Slices
import ../tensor_band_concat.bend as BandConcat
import ../tensor_band.bend as Bands
import ./tensor_reduce_axis_values.bend as ReduceAxisValues
import ./tensor_count_refinement.bend as Count
import ./word_encoding.bend as WordEncoding

# The logical parts in order, each with its extents.
type Sources<-Part: Data> is Data:
  Final{part: Part,shape: List<&2,U32>}
  Then{part: Part,shape: List<&2,U32>,rest: Sources<Part>}

def appended(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,sources: Sources<Part>,+axis: U32,+target_shape: List<&2,U32>,+start: U32,accumulator: Element,+position: U32) -> Element:
  match sources:
    case Final{part,shape}: Slices.slice_element(~Element,~Part,~part_value,Slices.slice_of(axis,start,target_shape,shape),part,accumulator,position)
    case Then{part,+shape,rest}:
      appended(~Element,~Part,~part_value,rest,axis,target_shape,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32),
        Slices.slice_element(~Element,~Part,~part_value,Slices.slice_of(axis,start,target_shape,shape),part,accumulator,position),position)

type Appended<-Before: Data,-Part: Data> is Data:
  Appended{before: Before,sources: Sources<Part>,axis: U32,target_shape: List<&2,U32>,start: U32}

def appended_value(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,context: Appended<Before,Part>,position: U32) -> Element:
  Appended{before,sources,axis,target_shape,start} = context
  +position = position
  appended(~Element,~Part,~part_value,sources,axis,target_shape,start,before_value(before,position),position)

# The parts as models: Ready dense tensors whose contents satisfy the facts
# of their logical values.
type Pieces<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element> is Data:
  Last{part: Part,shape: List<&2,U32>,model: Buffer.Model<Element>,fill: Element,workers: U32,
    facts: GatherValues.Facts(~Element,~Part,~part_value,part,shape,Buffer.content(~Element,model))}
  Next{part: Part,shape: List<&2,U32>,model: Buffer.Model<Element>,fill: Element,workers: U32,
    facts: GatherValues.Facts(~Element,~Part,~part_value,part,shape,Buffer.content(~Element,model)),rest: Pieces<Element,Part,part_value>}

def packed(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>) -> Tensors.Parts<Element>:
  match pieces:
    case Last{part,shape,model,fill,workers,facts}: Tensors.Single{Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers}}
    case Next{part,shape,model,fill,workers,facts,rest}:
      Tensors.Cons{Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},packed(~Element,~Part,~part_value,rest)}

def sources_of(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>) -> Sources<Part>:
  match pieces:
    case Last{part,shape,model,fill,workers,facts}: Final{part,shape}
    case Next{part,shape,model,fill,workers,facts,rest}: Then{part,shape,sources_of(~Element,~Part,~part_value,rest)}

# A Failed output stays Failed through every later part.
def sticky(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>,-storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,+problem: Tensors.Problem,+axis: U32,+start: U32) ->
  {Dense.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(Tensors.Tensor{storage,shape,fill,Tensors.Failed{problem},workers},start))) == False{} : Bool}:
  match pieces:
    case Last{part,part_shape,model,part_fill,part_workers,facts}: {==}
    case Next{part,+part_shape,model,part_fill,+part_workers,facts,rest}:
      sticky(~Element,~Part,~part_value,rest,storage,shape,fill,U32.max(workers,part_workers),problem,axis,(start + Layout.extent_at(part_shape,U32.to_nat(axis)) : U32))

# A Ready final result means the set_slice of every part was Ready.
def continued_state(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,state: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool,pieces: Pieces<Element,Part,part_value>,+name: String,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>,+target_fill: Element,+part_fill: Element,+workers: U32,+axis: U32,+start: U32,
  accepted: {Dense.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,state)),start))) == True{} : Bool}) ->
  {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,state))) == True{} : Bool}:
  (owners,valid) = state
  (part,target) = owners
  match valid:
    case True{}: {==}
    case False{}:
      Empty.absurd({Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,((part,target),False{})))) == True{} : Bool},WordEncoding.false_true(
        %sticky(~Element,~Part,~part_value,pieces,target,target_shape,target_fill,workers,Tensors.Problem{name,Tensors.ShapeMismatch{target_shape,part_shape}},axis,start) : {_ == True{} : Bool}
        accepted))

def continued_settled(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,same: Bool,pieces: Pieces<Element,Part,part_value>,+name: String,+target_model: Buffer.Model<Element>,+target_shape: List<&2,U32>,+target_fill: Element,+part_model: Buffer.Model<Element>,+part_shape: List<&2,U32>,+part_fill: Element,+workers: U32,+axis: U32,+start: U32,
  accepted: {Dense.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(Layout.result_of(~Element,Layout.settled(~Element,name,same,Buffer.pack(~Element,part_model),Buffer.pack(~Element,target_model),target_shape,part_shape,target_fill,part_fill,workers)),start))) == True{} : Bool}) ->
  {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.settled(~Element,name,same,Buffer.pack(~Element,part_model),Buffer.pack(~Element,target_model),target_shape,part_shape,target_fill,part_fill,workers))) == True{} : Bool}:
  match same:
    case True{}: {==}
    case False{}:
      Empty.absurd({Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.settled(~Element,name,False{},Buffer.pack(~Element,part_model),Buffer.pack(~Element,target_model),target_shape,part_shape,target_fill,part_fill,workers))) == True{} : Bool},WordEncoding.false_true(
        %sticky(~Element,~Part,~part_value,pieces,Buffer.pack(~Element,target_model),target_shape,target_fill,workers,Tensors.Problem{name,Tensors.ShapeMismatch{target_shape,part_shape}},axis,start) : {_ == True{} : Bool}
        accepted))

def continued_place(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,empty: Bool,slice: Maybe<&2,Views.View>,source: Maybe<&2,Views.View>,pieces: Pieces<Element,Part,part_value>,+name: String,+target_model: Buffer.Model<Element>,+target_shape: List<&2,U32>,+target_fill: Element,+part_model: Buffer.Model<Element>,+part_shape: List<&2,U32>,+part_fill: Element,+workers: U32,+axis: U32,+start: U32,
  accepted: {Dense.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(Layout.result_of(~Element,Layout.place(~Element,name,empty,slice,source,Buffer.pack(~Element,target_model),target_shape,target_fill,Buffer.pack(~Element,part_model),part_shape,part_fill,workers)),start))) == True{} : Bool}) ->
  {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.place(~Element,name,empty,slice,source,Buffer.pack(~Element,target_model),target_shape,target_fill,Buffer.pack(~Element,part_model),part_shape,part_fill,workers))) == True{} : Bool}:
  match empty slice source:
    case False{} Some{slice} Some{source}:
      continued_state(~Element,~Part,~part_value,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),source,Buffer.pack(~Element,target_model),slice)),pieces,name,
        target_shape,part_shape,target_fill,part_fill,workers,axis,start,accepted)
    case True{} Some{slice} Some{source}: continued_settled(~Element,~Part,~part_value,Layout.same_extents(slice,part_shape),pieces,name,target_model,target_shape,target_fill,part_model,part_shape,part_fill,workers,axis,start,accepted)
    case True{} Some{slice} None{}: continued_settled(~Element,~Part,~part_value,Layout.same_extents(slice,part_shape),pieces,name,target_model,target_shape,target_fill,part_model,part_shape,part_fill,workers,axis,start,accepted)
    case False{} Some{slice} None{}: continued_settled(~Element,~Part,~part_value,False{},pieces,name,target_model,target_shape,target_fill,part_model,part_shape,part_fill,workers,axis,start,accepted)
    case False{} None{} Some{source}: continued_settled(~Element,~Part,~part_value,False{},pieces,name,target_model,target_shape,target_fill,part_model,part_shape,part_fill,workers,axis,start,accepted)
    case False{} None{} None{}: continued_settled(~Element,~Part,~part_value,False{},pieces,name,target_model,target_shape,target_fill,part_model,part_shape,part_fill,workers,axis,start,accepted)
    case True{} None{} Some{source}: continued_settled(~Element,~Part,~part_value,False{},pieces,name,target_model,target_shape,target_fill,part_model,part_shape,part_fill,workers,axis,start,accepted)
    case True{} None{} None{}: continued_settled(~Element,~Part,~part_value,False{},pieces,name,target_model,target_shape,target_fill,part_model,part_shape,part_fill,workers,axis,start,accepted)

# The reader of one part is the reader of its set_slice.
def final_reader(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,-tensor: Tensors.Tensor<Element>,+before: Before,+part: Part,+part_shape: List<&2,U32>,+axis: U32,+shape: List<&2,U32>,+start: U32,
  evidence: GatherValues.Shaped<Element,Slices.Placed<Before,Part>,Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),Slices.placement(~Before,~Part,before,part,axis,start,shape,part_shape),shape,tensor>) ->
  GatherValues.Shaped<Element,Appended<Before,Part>,appended_value(~Element,~Before,~Part,~before_value,~part_value),Appended{before,Final{part,part_shape},axis,shape,start},shape,tensor>:
  GatherValues.Shaped{model,fill,workers,equation,facts} = evidence
  GatherValues.Shaped{model,fill,workers,equation,facts}

# The reader of the later parts over the set_slice of the first is the reader of all of them.
def then_reader(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,-tensor: Tensors.Tensor<Element>,+before: Before,+part: Part,+part_shape: List<&2,U32>,+rest: Sources<Part>,+axis: U32,+shape: List<&2,U32>,+start: U32,
  evidence: GatherValues.Shaped<Element,Appended<Slices.Placed<Before,Part>,Part>,appended_value(~Element,~Slices.Placed<Before,Part>,~Part,~Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),~part_value),Appended{Slices.placement(~Before,~Part,before,part,axis,start,shape,part_shape),rest,axis,shape,(start + Layout.extent_at(part_shape,U32.to_nat(axis)) : U32)},shape,tensor>) ->
  GatherValues.Shaped<Element,Appended<Before,Part>,appended_value(~Element,~Before,~Part,~before_value,~part_value),Appended{before,Then{part,part_shape,rest},axis,shape,start},shape,tensor>:
  GatherValues.Shaped{model,fill,workers,equation,facts} = evidence
  GatherValues.Shaped{model,fill,workers,equation,facts}

# The append of the parts onto a shaped output: a Ready result is a shaped
# tensor of the same extents whose reader is appended over the output.
def appended_pieces(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>,-output: Tensors.Tensor<Element>,+axis: U32,+start: U32,+before: Before,+shape: List<&2,U32>,
  output_shaped: GatherValues.Shaped<Element,Before,before_value,before,shape,output>,
  accepted: {Dense.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(output,start))) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Appended<Before,Part>,appended_value(~Element,~Before,~Part,~before_value,~part_value),Appended{before,sources_of(~Element,~Part,~part_value,pieces),axis,shape,start},shape,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(output,start))>:
  match pieces:
    case Last{+part,+part_shape,+model,+fill,+workers,facts}:
      final_reader(~Element,~Before,~Part,~before_value,~part_value,Layout.set_slice(~Element,output,axis,start,Tensors.Tensor{Buffer.pack(~Element,model),part_shape,fill,Tensors.Usable{},workers}),before,part,part_shape,axis,shape,start,
        Slices.shaped_slice(~Element,~Before,~Part,~before_value,~part_value,output,Tensors.Tensor{Buffer.pack(~Element,model),part_shape,fill,Tensors.Usable{},workers},before,part,axis,start,shape,part_shape,output_shaped,GatherValues.Shaped{model,fill,workers,{==},facts},accepted))
    case Next{+part,+part_shape,+model,+fill,+workers,+facts,+rest}:
      GatherValues.Shaped{+output_model,+output_fill,+output_workers,+output_equation,+output_facts} = output_shaped
      +proceeded = {
        %output_equation : {Dense.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,Next{part,part_shape,model,fill,workers,facts,rest}),axis,(_,start))) == True{} : Bool}
        accepted : {Dense.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,rest),axis,(Layout.set_slice(~Element,Tensors.Tensor{Buffer.pack(~Element,output_model),shape,output_fill,Tensors.Usable{},output_workers},axis,start,Tensors.Tensor{Buffer.pack(~Element,model),part_shape,fill,Tensors.Usable{},workers}),(start + Layout.extent_at(part_shape,U32.to_nat(axis)) : U32)))) == True{} : Bool}}
      +sliced_ready = continued_place(~Element,~Part,~part_value,Shapes.has_zero(part_shape),Slices.slice_of(axis,start,shape,part_shape),Tensors.view_of(part_shape),rest,"set_slice",
        output_model,shape,output_fill,model,part_shape,fill,U32.max(output_workers,workers),axis,(start + Layout.extent_at(part_shape,U32.to_nat(axis)) : U32),proceeded)
      %Equal.sym(Tensors.Tensor<Element>,output,Tensors.Tensor{Buffer.pack(~Element,output_model),shape,output_fill,Tensors.Usable{},output_workers},output_equation) :
        GatherValues.Shaped<Element,Appended<Before,Part>,appended_value(~Element,~Before,~Part,~before_value,~part_value),Appended{before,sources_of(~Element,~Part,~part_value,Next{part,part_shape,model,fill,workers,facts,rest}),axis,shape,start},shape,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,Next{part,part_shape,model,fill,workers,facts,rest}),axis,(_,start))>
      then_reader(~Element,~Before,~Part,~before_value,~part_value,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,rest),axis,(Layout.set_slice(~Element,Tensors.Tensor{Buffer.pack(~Element,output_model),shape,output_fill,Tensors.Usable{},output_workers},axis,start,Tensors.Tensor{Buffer.pack(~Element,model),part_shape,fill,Tensors.Usable{},workers}),(start + Layout.extent_at(part_shape,U32.to_nat(axis)) : U32))),before,part,part_shape,sources_of(~Element,~Part,~part_value,rest),axis,shape,start,
        appended_pieces(~Element,~Slices.Placed<Before,Part>,~Part,~Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),~part_value,rest,Layout.set_slice(~Element,Tensors.Tensor{Buffer.pack(~Element,output_model),shape,output_fill,Tensors.Usable{},output_workers},axis,start,Tensors.Tensor{Buffer.pack(~Element,model),part_shape,fill,Tensors.Usable{},workers}),axis,(start + Layout.extent_at(part_shape,U32.to_nat(axis)) : U32),Slices.placement(~Before,~Part,before,part,axis,start,shape,part_shape),shape,
          Slices.shaped_slice(~Element,~Before,~Part,~before_value,~part_value,Tensors.Tensor{Buffer.pack(~Element,output_model),shape,output_fill,Tensors.Usable{},output_workers},Tensors.Tensor{Buffer.pack(~Element,model),part_shape,fill,Tensors.Usable{},workers},before,part,axis,start,shape,part_shape,
            GatherValues.Shaped{output_model,output_fill,output_workers,{==},output_facts},GatherValues.Shaped{model,fill,workers,{==},facts},sliced_ready),
          proceeded))

# The public concat of dense parts. The metadata passes return the parts
# unchanged, dense parts have no band plan, and the output starts as the
# full tensor of the first fill value.
def first_shape(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>) -> List<&2,U32>:
  match pieces:
    case Last{part,shape,model,fill,workers,facts}: shape
    case Next{part,shape,model,fill,workers,facts,rest}: shape

def first_fill(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>) -> Element:
  match pieces:
    case Last{part,shape,model,fill,workers,facts}: fill
    case Next{part,shape,model,fill,workers,facts,rest}: fill

def survey_of(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>,+first: List<&2,U32>,+axis: Nat,summary: Layout.Survey) -> Layout.Survey:
  match pieces:
    case Last{part,shape,model,fill,workers,facts}: Layout.surveyed(summary,first,shape,axis,Tensors.Usable{},workers)
    case Next{part,shape,model,fill,workers,facts,rest}: survey_of(~Element,~Part,~part_value,rest,first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Usable{},workers))

def survey_total(summary: Layout.Survey) -> U32:
  Layout.Survey{valid,total,status,workers} = summary
  total

def led(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>) ->
  {Layout.lead(~Element,packed(~Element,~Part,~part_value,pieces)) == (packed(~Element,~Part,~part_value,pieces),(first_shape(~Element,~Part,~part_value,pieces),first_fill(~Element,~Part,~part_value,pieces))) : Tensors.Parts<Element> & (List<&2,U32> & Element)}:
  match pieces:
    case Last{part,shape,model,fill,workers,facts}: {==}
    case Next{part,shape,model,fill,workers,facts,rest}: {==}

def surveyed(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {Layout.survey(~Element,packed(~Element,~Part,~part_value,pieces),first,axis,summary) == (packed(~Element,~Part,~part_value,pieces),survey_of(~Element,~Part,~part_value,pieces,first,axis,summary)) : Tensors.Parts<Element> & Layout.Survey}:
  match pieces:
    case Last{part,shape,model,fill,workers,facts}: {==}
    case Next{part,+shape,+model,+fill,+workers,facts,+rest}:
      %Equal.sym(Tensors.Parts<Element> & Layout.Survey,Layout.survey(~Element,packed(~Element,~Part,~part_value,rest),first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Usable{},workers)),(packed(~Element,~Part,~part_value,rest),survey_of(~Element,~Part,~part_value,rest,first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Usable{},workers))),
        surveyed(~Element,~Part,~part_value,rest,first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Usable{},workers))) :
        {Layout.prepend_survey(~Element,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},_) == (Tensors.Cons{Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},packed(~Element,~Part,~part_value,rest)},survey_of(~Element,~Part,~part_value,rest,first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Usable{},workers))) : Tensors.Parts<Element> & Layout.Survey}
      {==}

def discovered(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>) ->
  {BandConcat.discover_compatible(~Element,packed(~Element,~Part,~part_value,pieces)) == ((packed(~Element,~Part,~part_value,pieces),None{}),False{}) : (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool}:
  match pieces:
    case Last{part,shape,model,fill,workers,facts}: {==}
    case Next{part,+shape,+model,+fill,+workers,facts,+rest}:
      %Equal.sym((Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool,BandConcat.discover_compatible(~Element,packed(~Element,~Part,~part_value,rest)),((packed(~Element,~Part,~part_value,rest),None{}),False{}),discovered(~Element,~Part,~part_value,rest)) :
        {BandConcat.cons_compatible(~Element,(Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},None{}),_) == ((Tensors.Cons{Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},packed(~Element,~Part,~part_value,rest)},None{}),False{}) : (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool}
      {==}

def counted(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,count: Maybe<&2,U32>,+pieces: Pieces<Element,Part,part_value>,+axis: U32,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +counts: {Tensors.count_of(shape) == count : Maybe<&2,U32>},
  accepted: {Dense.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(Tensors.with_workers(~Element,Tensors.full_count(~Element,shape,count,fill),workers),0))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,shape,0},Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(Tensors.with_workers(~Element,Tensors.full_count(~Element,shape,count,fill),workers),0))):
  match count:
    case Some{+count}:
      GatherValues.shaped_contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,shape,0},shape,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(Tensors.Tensor{Tensors.buffer(~Element,fill,count),shape,fill,Tensors.Usable{},U32.max(1,workers)},0)),
        appended_pieces(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value,pieces,Tensors.Tensor{Tensors.buffer(~Element,fill,count),shape,fill,Tensors.Usable{},U32.max(1,workers)},axis,0,fill,shape,
          GatherValues.shaped_observed(~Element,~Element,~ReduceAxisValues.constant(~Element),Tensors.buffer(~Element,fill,count),fill,shape,fill,U32.max(1,workers),
            Buffer.observe(~Element,Tensors.buffer(~Element,fill,count)),ReduceAxisValues.initial_counted(~Element,shape,fill,count,Count.count_of(shape,count,counts))),
          accepted))
    case None{}:
      Empty.absurd(Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,shape,0},Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(Tensors.with_workers(~Element,Tensors.full_count(~Element,shape,None{},fill),workers),0))),WordEncoding.false_true(
        %sticky(~Element,~Part,~part_value,pieces,Tensors.buffer(~Element,fill,0),shape,fill,U32.max(1,workers),Tensors.Problem{"full",Tensors.InvalidShape{shape}},axis,0) : {_ == True{} : Bool}
        accepted))

def compatible(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+pieces: Pieces<Element,Part,part_value>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,
  accepted: {Dense.tensor_ready(~Element,Layout.join_compatible(~Element,shape,axis,fill,workers,BandConcat.discover_compatible(~Element,packed(~Element,~Part,~part_value,pieces)))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,shape,0},Layout.join_compatible(~Element,shape,axis,fill,workers,BandConcat.discover_compatible(~Element,packed(~Element,~Part,~part_value,pieces)))):
  %Equal.sym((Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool,BandConcat.discover_compatible(~Element,packed(~Element,~Part,~part_value,pieces)),((packed(~Element,~Part,~part_value,pieces),None{}),False{}),discovered(~Element,~Part,~part_value,pieces)) : Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,shape,0},Layout.join_compatible(~Element,shape,axis,fill,workers,_))
  counted(~Element,~Part,~part_value,Tensors.count_of(shape),pieces,axis,shape,fill,workers,{==},
    %discovered(~Element,~Part,~part_value,pieces) : {Dense.tensor_ready(~Element,Layout.join_compatible(~Element,shape,axis,fill,workers,_)) == True{} : Bool}
    accepted)

def joined_status(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,status: Tensors.Status,valid: Bool,+pieces: Pieces<Element,Part,part_value>,+first: List<&2,U32>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,
  accepted: {Dense.tensor_ready(~Element,Layout.join_status(~Element,status,valid,first,shape,axis,fill,workers,packed(~Element,~Part,~part_value,pieces))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,shape,0},Layout.join_status(~Element,status,valid,first,shape,axis,fill,workers,packed(~Element,~Part,~part_value,pieces))):
  match status valid:
    case Tensors.Usable{} True{}: compatible(~Element,~Part,~part_value,pieces,shape,axis,fill,workers,accepted)
    case Tensors.Usable{} False{}: Empty.absurd(Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,shape,0},Layout.join_status(~Element,Tensors.Usable{},False{},first,shape,axis,fill,workers,packed(~Element,~Part,~part_value,pieces))),WordEncoding.false_true(accepted))
    case Tensors.Failed{problem} True{}: Empty.absurd(Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,shape,0},Layout.join_status(~Element,Tensors.Failed{problem},True{},first,shape,axis,fill,workers,packed(~Element,~Part,~part_value,pieces))),WordEncoding.false_true(accepted))
    case Tensors.Failed{problem} False{}: Empty.absurd(Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,shape,0},Layout.join_status(~Element,Tensors.Failed{problem},False{},first,shape,axis,fill,workers,packed(~Element,~Part,~part_value,pieces))),WordEncoding.false_true(accepted))

def joined_summary(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,summary: Layout.Survey,+pieces: Pieces<Element,Part,part_value>,+first: List<&2,U32>,+axis: U32,+fill: Element,
  accepted: {Dense.tensor_ready(~Element,Layout.joined(~Element,first,axis,fill,(packed(~Element,~Part,~part_value,pieces),summary))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,Layout.with_extent(first,U32.to_nat(axis),survey_total(summary)),0},Layout.joined(~Element,first,axis,fill,(packed(~Element,~Part,~part_value,pieces),summary))):
  Layout.Survey{valid,+total,status,+workers} = summary
  joined_status(~Element,~Part,~part_value,status,valid,pieces,first,Layout.with_extent(first,U32.to_nat(axis),total),axis,fill,workers,accepted)

# Ready dense parts with complete values, given as models: a Ready concat
# result is well formed in the extents of the first part with the summed
# extent at the axis, and every index holds appended over the fill value:
# the part element of the last block that covers it.
def concat(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+pieces: Pieces<Element,Part,part_value>,+axis: U32,
  accepted: {Dense.tensor_ready(~Element,Layout.concat(~Element,packed(~Element,~Part,~part_value,pieces),axis)) == True{} : Bool}) ->
  Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{first_fill(~Element,~Part,~part_value,pieces),sources_of(~Element,~Part,~part_value,pieces),axis,Layout.with_extent(first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),survey_total(survey_of(~Element,~Part,~part_value,pieces,first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first_shape(~Element,~Part,~part_value,pieces))),0,Tensors.Usable{},1}))),0},Layout.concat(~Element,packed(~Element,~Part,~part_value,pieces),axis)):
  %Equal.sym(Tensors.Parts<Element> & (List<&2,U32> & Element),Layout.lead(~Element,packed(~Element,~Part,~part_value,pieces)),(packed(~Element,~Part,~part_value,pieces),(first_shape(~Element,~Part,~part_value,pieces),first_fill(~Element,~Part,~part_value,pieces))),led(~Element,~Part,~part_value,pieces)) :
    Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{first_fill(~Element,~Part,~part_value,pieces),sources_of(~Element,~Part,~part_value,pieces),axis,Layout.with_extent(first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),survey_total(survey_of(~Element,~Part,~part_value,pieces,first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first_shape(~Element,~Part,~part_value,pieces))),0,Tensors.Usable{},1}))),0},Layout.concat_led(~Element,axis,_))
  %Equal.sym(Tensors.Parts<Element> & Layout.Survey,Layout.survey(~Element,packed(~Element,~Part,~part_value,pieces),first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first_shape(~Element,~Part,~part_value,pieces))),0,Tensors.Usable{},1}),(packed(~Element,~Part,~part_value,pieces),survey_of(~Element,~Part,~part_value,pieces,first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first_shape(~Element,~Part,~part_value,pieces))),0,Tensors.Usable{},1})),surveyed(~Element,~Part,~part_value,pieces,first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first_shape(~Element,~Part,~part_value,pieces))),0,Tensors.Usable{},1})) :
    Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{first_fill(~Element,~Part,~part_value,pieces),sources_of(~Element,~Part,~part_value,pieces),axis,Layout.with_extent(first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),survey_total(survey_of(~Element,~Part,~part_value,pieces,first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first_shape(~Element,~Part,~part_value,pieces))),0,Tensors.Usable{},1}))),0},Layout.joined(~Element,first_shape(~Element,~Part,~part_value,pieces),axis,first_fill(~Element,~Part,~part_value,pieces),_))
  joined_summary(~Element,~Part,~part_value,survey_of(~Element,~Part,~part_value,pieces,first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first_shape(~Element,~Part,~part_value,pieces))),0,Tensors.Usable{},1}),pieces,first_shape(~Element,~Part,~part_value,pieces),axis,first_fill(~Element,~Part,~part_value,pieces),
    %surveyed(~Element,~Part,~part_value,pieces,first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first_shape(~Element,~Part,~part_value,pieces))),0,Tensors.Usable{},1}) : {Dense.tensor_ready(~Element,Layout.joined(~Element,first_shape(~Element,~Part,~part_value,pieces),axis,first_fill(~Element,~Part,~part_value,pieces),_)) == True{} : Bool}
    %led(~Element,~Part,~part_value,pieces) : {Dense.tensor_ready(~Element,Layout.concat_led(~Element,axis,_)) == True{} : Bool}
    accepted)

# Native parts, each observed once as a shaped tensor: the parts are the
# packed pieces, so the statement over the models is the statement over them.
type Modeled<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element,-parts: Tensors.Parts<Element>> is Data:
  Modeled{pieces: Pieces<Element,Part,part_value>,equation: {parts == packed(~Element,~Part,~part_value,pieces) : Tensors.Parts<Element>}}

def modeled_pieces(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-parts: Tensors.Parts<Element>,modeled: Modeled<Element,Part,part_value,parts>) -> Pieces<Element,Part,part_value>:
  Modeled{pieces,equation} = modeled
  pieces

def modeled_single(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-tensor: Tensors.Tensor<Element>,+part: Part,+shape: List<&2,U32>,
  evidence: GatherValues.Shaped<Element,Part,part_value,part,shape,tensor>) -> Modeled<Element,Part,part_value,Tensors.Single{tensor}>:
  GatherValues.Shaped{+model,+fill,+workers,equation,facts} = evidence
  Modeled{Last{part,shape,model,fill,workers,facts},
    Equal.cong(Tensors.Tensor<Element>,Tensors.Parts<Element>,single => Tensors.Single{single},tensor,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},equation)}

def modeled_cons(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-tensor: Tensors.Tensor<Element>,-rest: Tensors.Parts<Element>,+part: Part,+shape: List<&2,U32>,
  evidence: GatherValues.Shaped<Element,Part,part_value,part,shape,tensor>,others: Modeled<Element,Part,part_value,rest>) ->
  Modeled<Element,Part,part_value,Tensors.Cons{tensor,rest}>:
  GatherValues.Shaped{+model,+fill,+workers,equation,facts} = evidence
  Modeled{+pieces,others_equation} = others
  Modeled{Next{part,shape,model,fill,workers,facts,pieces},
    %Equal.sym(Tensors.Tensor<Element>,tensor,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},equation) : {Tensors.Cons{_,rest} == Tensors.Cons{Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},packed(~Element,~Part,~part_value,pieces)} : Tensors.Parts<Element>}
    %Equal.sym(Tensors.Parts<Element>,rest,packed(~Element,~Part,~part_value,pieces),others_equation) : {Tensors.Cons{Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},_} == Tensors.Cons{Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},packed(~Element,~Part,~part_value,pieces)} : Tensors.Parts<Element>}
    {==}}

# The public concat of native Ready dense parts with complete values.
def concat_native(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-parts: Tensors.Parts<Element>,+axis: U32,+modeled: Modeled<Element,Part,part_value,parts>,
  accepted: {Dense.tensor_ready(~Element,Layout.concat(~Element,parts,axis)) == True{} : Bool}) ->
  Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{first_fill(~Element,~Part,~part_value,modeled_pieces(~Element,~Part,~part_value,parts,modeled)),sources_of(~Element,~Part,~part_value,modeled_pieces(~Element,~Part,~part_value,parts,modeled)),axis,Layout.with_extent(first_shape(~Element,~Part,~part_value,modeled_pieces(~Element,~Part,~part_value,parts,modeled)),U32.to_nat(axis),survey_total(survey_of(~Element,~Part,~part_value,modeled_pieces(~Element,~Part,~part_value,parts,modeled),first_shape(~Element,~Part,~part_value,modeled_pieces(~Element,~Part,~part_value,parts,modeled)),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first_shape(~Element,~Part,~part_value,modeled_pieces(~Element,~Part,~part_value,parts,modeled)))),0,Tensors.Usable{},1}))),0},Layout.concat(~Element,parts,axis)):
  Modeled{+pieces,+equation} = modeled
  %Equal.sym(Tensors.Parts<Element>,parts,packed(~Element,~Part,~part_value,pieces),equation) : Dense.Contract(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Appended{first_fill(~Element,~Part,~part_value,pieces),sources_of(~Element,~Part,~part_value,pieces),axis,Layout.with_extent(first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),survey_total(survey_of(~Element,~Part,~part_value,pieces,first_shape(~Element,~Part,~part_value,pieces),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first_shape(~Element,~Part,~part_value,pieces))),0,Tensors.Usable{},1}))),0},Layout.concat(~Element,_,axis))
  concat(~Element,~Part,~part_value,pieces,axis,
    %equation : {Dense.tensor_ready(~Element,Layout.concat(~Element,_,axis)) == True{} : Bool}
    accepted)
