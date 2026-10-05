# The dense route with operands in either layout. An operation that begins
# with with_dense gathers a band tensor into fresh dense storage and then
# runs its dense code, so its values follow from the dense statement applied
# to the dense form of each operand. Sourced records a Ready tensor in either
# layout as an owner model whose dense form is a shaped tensor.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_layout.bend as Layout
import ../tensor_band.bend as Bands
import ../tensor_f32.bend as F
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_dense_contract.bend as Dense
import ./tensor_gather_values.bend as GatherValues
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_dense_observation.bend as DenseObservation
import ./tensor_ready_densification.bend as ReadyDensification
import ./tensor_set_slice_values.bend as Slices
import ./tensor_reduce_axis_values.bend as ReduceAxisValues
import ./tensor_scalar_values.bend as Scalars
import ../tensor_band_concat.bend as BandConcat
import ../storage_buffer.bend as Storage
import ./tensor_concat_dense_values.bend as Concat
import ./word_encoding.bend as WordEncoding

type Sourced<-Element: Data,-Context: Data,-reader: Context -> U32 -> Element,-context: Context,-shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>> is Data:
  Sourced{owner: BandConcatExecution.TensorOwnerModel<Element>,
    equation: {tensor == BandConcatExecution.tensor_owner_pack(~Element,owner) : Tensors.Tensor<Element>},
    shaped: GatherValues.Shaped<Element,Context,reader,context,shape,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))>}

# A shaped dense tensor is its own dense form.
def dense(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,-tensor: Tensors.Tensor<Element>,+context: Context,+shape: List<&2,U32>,
  evidence: GatherValues.Shaped<Element,Context,reader,context,shape,tensor>) -> Sourced<Element,Context,reader,context,shape,tensor>:
  GatherValues.Shaped{+model,+fill,+workers,equation,facts} = evidence
  Sourced{BandConcatExecution.DenseOwner{model,shape,fill,Tensors.Ready{},workers},equation,GatherValues.Shaped{model,fill,workers,{==},facts}}

# The gathered storage of a Ready normal band tensor holds its logical values.
def gathered(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  prepared: ReadyDensification.Prepared(~Element,~Context,~reader,context,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers})) ->
  GatherValues.Shaped<Element,Context,reader,context,shape,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers})>:
  Shape.Both{normal,Shape.Both{Shape.Both{well_formed,status},values}} = prepared
  GatherValues.shaped_observed(~Element,~Context,~reader,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))),context,shape,fill,workers,Buffer.observe(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))),Shape.Both{well_formed,values})

def banded_model(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers}),
  values: Dense.TensorValues(~Element,~Context,~reader,context,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers})) ->
  Sourced<Element,Context,reader,context,shape,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers}>:
  Sourced{BandConcatExecution.BandedOwner{bands,geometry,shape,fill,Tensors.Ready{},workers},{==},
    gathered(~Element,~Context,~reader,context,bands,geometry,shape,fill,workers,
      ReadyDensification.prepared_owner(~Element,~Context,~reader,context,BandConcatExecution.BandedOwner{bands,geometry,shape,fill,Tensors.Ready{},workers},normal,{==},values))}

# A Ready normal band tensor with complete logical values is a source.
def banded(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,-bands: Bands.Bands(~Element),+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  observation: Model.Observation<Element,bands>,
  normal: Global.Normal(~Element,Tensors.Banded{bands,geometry,shape,fill,Tensors.Ready{},workers}),
  values: Dense.TensorValues(~Element,~Context,~reader,context,Tensors.Banded{bands,geometry,shape,fill,Tensors.Ready{},workers})) ->
  Sourced<Element,Context,reader,context,shape,Tensors.Banded{bands,geometry,shape,fill,Tensors.Ready{},workers}>:
  Model.Observation{+model,+equation} = observation
  %equation : Sourced<Element,Context,reader,context,shape,Tensors.Banded{_,geometry,shape,fill,Tensors.Ready{},workers}>
  banded_model(~Element,~Context,~reader,context,model,geometry,shape,fill,workers,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) : Global.Normal(~Element,Tensors.Banded{_,geometry,shape,fill,Tensors.Ready{},workers})
    normal,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) : Dense.TensorValues(~Element,~Context,~reader,context,Tensors.Banded{_,geometry,shape,fill,Tensors.Ready{},workers})
    values)

# set_slice of two sources: the result is the shaped tensor of the dense
# statement, whatever the layouts of the target and the part were.
def set_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,-target: Tensors.Tensor<Element>,-part_tensor: Tensors.Tensor<Element>,+before: Before,+part: Part,+axis: U32,+start: U32,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>,
  target_sourced: Sourced<Element,Before,before_value,before,target_shape,target>,
  part_sourced: Sourced<Element,Part,part_value,part,part_shape,part_tensor>,
  accepted: {Dense.tensor_ready(~Element,Layout.set_slice(~Element,target,axis,start,part_tensor)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Slices.Placed<Before,Part>,Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),Slices.placement(~Before,~Part,before,part,axis,start,target_shape,part_shape),target_shape,Layout.set_slice(~Element,target,axis,start,part_tensor)>:
  Sourced{+target_owner,+target_equation,target_shaped} = target_sourced
  Sourced{+part_owner,+part_equation,part_shaped} = part_sourced
  %Equal.sym(Tensors.Tensor<Element>,target,BandConcatExecution.tensor_owner_pack(~Element,target_owner),target_equation) : GatherValues.Shaped<Element,Slices.Placed<Before,Part>,Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),Slices.placement(~Before,~Part,before,part,axis,start,target_shape,part_shape),target_shape,Layout.set_slice(~Element,_,axis,start,part_tensor)>
  %Equal.sym(Tensors.Tensor<Element>,part_tensor,BandConcatExecution.tensor_owner_pack(~Element,part_owner),part_equation) : GatherValues.Shaped<Element,Slices.Placed<Before,Part>,Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),Slices.placement(~Before,~Part,before,part,axis,start,target_shape,part_shape),target_shape,Layout.set_slice(~Element,BandConcatExecution.tensor_owner_pack(~Element,target_owner),axis,start,_)>
  %Equal.sym(Tensors.Tensor<Element>,Layout.set_slice(~Element,BandConcatExecution.tensor_owner_pack(~Element,target_owner),axis,start,BandConcatExecution.tensor_owner_pack(~Element,part_owner)),Layout.set_slice(~Element,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,target_owner)),axis,start,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,part_owner))),DenseObservation.set_slice_densified(~Element,BandConcatExecution.tensor_owner_pack(~Element,target_owner),axis,start,BandConcatExecution.tensor_owner_pack(~Element,part_owner))) : GatherValues.Shaped<Element,Slices.Placed<Before,Part>,Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),Slices.placement(~Before,~Part,before,part,axis,start,target_shape,part_shape),target_shape,_>
  Slices.shaped_slice(~Element,~Before,~Part,~before_value,~part_value,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,target_owner)),DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,part_owner)),before,part,axis,start,target_shape,part_shape,target_shaped,part_shaped,
    %DenseObservation.set_slice_densified(~Element,BandConcatExecution.tensor_owner_pack(~Element,target_owner),axis,start,BandConcatExecution.tensor_owner_pack(~Element,part_owner)) : {Dense.tensor_ready(~Element,_) == True{} : Bool}
    %part_equation : {Dense.tensor_ready(~Element,Layout.set_slice(~Element,BandConcatExecution.tensor_owner_pack(~Element,target_owner),axis,start,_)) == True{} : Bool}
    %target_equation : {Dense.tensor_ready(~Element,Layout.set_slice(~Element,_,axis,start,part_tensor)) == True{} : Bool}
    accepted)

# reduce_axis gathers its operand first.
def reduce_densified(~Element: Data,~operation: Element -> Element -> Element,tensor: Tensors.Tensor<Element>,+name: String,+initial: Element,+axis: U32) ->
  {Layout.reduce_axis(~Element,~operation,name,initial,tensor,axis) == Layout.reduce_axis(~Element,~operation,name,initial,DenseObservation.dense_tensor(~Element,tensor),axis) : Tensors.Tensor<Element>}:
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,workers}: {==}
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: {==}

def reduce_axis(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,-tensor: Tensors.Tensor<Element>,+logical: Logical,+name: String,+initial: Element,+axis: U32,+shape: List<&2,U32>,
  sourced: Sourced<Element,Logical,original,logical,shape,tensor>,
  accepted: {Dense.tensor_ready(~Element,Layout.reduce_axis(~Element,~operation,name,initial,tensor,axis)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,ReduceAxisValues.Fold<Element,Logical>,ReduceAxisValues.value(~Element,~Element,~Logical,~ReduceAxisValues.constant(~Element),~original,~operation),ReduceAxisValues.Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Shapes.without_axis(shape,U32.to_nat(axis)),Layout.reduce_axis(~Element,~operation,name,initial,tensor,axis)>:
  Sourced{+owner,+equation,shaped} = sourced
  %Equal.sym(Tensors.Tensor<Element>,tensor,BandConcatExecution.tensor_owner_pack(~Element,owner),equation) : GatherValues.Shaped<Element,ReduceAxisValues.Fold<Element,Logical>,ReduceAxisValues.value(~Element,~Element,~Logical,~ReduceAxisValues.constant(~Element),~original,~operation),ReduceAxisValues.Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Shapes.without_axis(shape,U32.to_nat(axis)),Layout.reduce_axis(~Element,~operation,name,initial,_,axis)>
  %Equal.sym(Tensors.Tensor<Element>,Layout.reduce_axis(~Element,~operation,name,initial,BandConcatExecution.tensor_owner_pack(~Element,owner),axis),Layout.reduce_axis(~Element,~operation,name,initial,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),axis),reduce_densified(~Element,~operation,BandConcatExecution.tensor_owner_pack(~Element,owner),name,initial,axis)) : GatherValues.Shaped<Element,ReduceAxisValues.Fold<Element,Logical>,ReduceAxisValues.value(~Element,~Element,~Logical,~ReduceAxisValues.constant(~Element),~original,~operation),ReduceAxisValues.Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Shapes.without_axis(shape,U32.to_nat(axis)),_>
  ReduceAxisValues.shaped(~Element,~Logical,~original,~operation,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),logical,name,initial,axis,shape,shaped,
    %reduce_densified(~Element,~operation,BandConcatExecution.tensor_owner_pack(~Element,owner),name,initial,axis) : {Dense.tensor_ready(~Element,_) == True{} : Bool}
    %equation : {Dense.tensor_ready(~Element,Layout.reduce_axis(~Element,~operation,name,initial,_,axis)) == True{} : Bool}
    accepted)

# mean_axis reads the extents and then reduces, so it gathers as well.
def mean_densified(tensor: Tensors.Tensor<F32>,+axis: U32) ->
  {F.mean_axis(tensor,axis) == F.mean_axis(DenseObservation.dense_tensor(~F32,tensor),axis) : Tensors.Tensor<F32>}:
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,workers}: {==}
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: {==}

def mean_axis(~Logical: Data,~original: Logical -> U32 -> F32,-tensor: Tensors.Tensor<F32>,+logical: Logical,+axis: U32,+shape: List<&2,U32>,
  sourced: Sourced<F32,Logical,original,logical,shape,tensor>,
  accepted: {Dense.tensor_ready(~F32,F.mean_axis(tensor,axis)) == True{} : Bool}) ->
  GatherValues.Shaped<F32,Scalars.Scaled<ReduceAxisValues.Fold<F32,Logical>,F32>,Scalars.value(~F32,~ReduceAxisValues.Fold<F32,Logical>,~ReduceAxisValues.value(~F32,~F32,~Logical,~ReduceAxisValues.constant(~F32),~original,~F32.add),~F32.div),Scalars.Scaled{ReduceAxisValues.Fold{0.0,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},U32.to_f32(Layout.extent_at(shape,U32.to_nat(axis)))},Shapes.without_axis(shape,U32.to_nat(axis)),F.mean_axis(tensor,axis)>:
  Sourced{+owner,+equation,shaped} = sourced
  %Equal.sym(Tensors.Tensor<F32>,tensor,BandConcatExecution.tensor_owner_pack(~F32,owner),equation) : GatherValues.Shaped<F32,Scalars.Scaled<ReduceAxisValues.Fold<F32,Logical>,F32>,Scalars.value(~F32,~ReduceAxisValues.Fold<F32,Logical>,~ReduceAxisValues.value(~F32,~F32,~Logical,~ReduceAxisValues.constant(~F32),~original,~F32.add),~F32.div),Scalars.Scaled{ReduceAxisValues.Fold{0.0,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},U32.to_f32(Layout.extent_at(shape,U32.to_nat(axis)))},Shapes.without_axis(shape,U32.to_nat(axis)),F.mean_axis(_,axis)>
  %Equal.sym(Tensors.Tensor<F32>,F.mean_axis(BandConcatExecution.tensor_owner_pack(~F32,owner),axis),F.mean_axis(DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner)),axis),mean_densified(BandConcatExecution.tensor_owner_pack(~F32,owner),axis)) : GatherValues.Shaped<F32,Scalars.Scaled<ReduceAxisValues.Fold<F32,Logical>,F32>,Scalars.value(~F32,~ReduceAxisValues.Fold<F32,Logical>,~ReduceAxisValues.value(~F32,~F32,~Logical,~ReduceAxisValues.constant(~F32),~original,~F32.add),~F32.div),Scalars.Scaled{ReduceAxisValues.Fold{0.0,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},U32.to_f32(Layout.extent_at(shape,U32.to_nat(axis)))},Shapes.without_axis(shape,U32.to_nat(axis)),_>
  Scalars.mean_shaped(~Logical,~original,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner)),logical,axis,shape,shaped,
    %mean_densified(BandConcatExecution.tensor_owner_pack(~F32,owner),axis) : {Dense.tensor_ready(~F32,_) == True{} : Bool}
    %equation : {Dense.tensor_ready(~F32,F.mean_axis(_,axis)) == True{} : Bool}
    accepted)

# The public concat on its dense route with parts in either layout. A part
# is a dense model, or band models with the model of their gathered storage;
# the dense route gathers every band part once and appends it as a dense
# part, so its result is the dense statement over the gathered pieces.
type Operand<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element> is Data:
  Stored{part: Part,shape: List<&2,U32>,model: Buffer.Model<Element>,fill: Element,workers: U32,facts: GatherValues.Facts(~Element,~Part,~part_value,part,shape,Buffer.content(~Element,model))}
  Gathered{part: Part,shape: List<&2,U32>,bands: Model.Model<Element>,geometry: Bands.Geometry,model: Buffer.Model<Element>,fill: Element,workers: U32,
    equation: {Buffer.pack(~Element,model) == Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))) : Storage.Buffer<Element>},facts: GatherValues.Facts(~Element,~Part,~part_value,part,shape,Buffer.content(~Element,model))}

type Operands<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element> is Data:
  Only{operand: Operand<Element,Part,part_value>}
  Also{operand: Operand<Element,Part,part_value>,rest: Operands<Element,Part,part_value>}

def native(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,operand: Operand<Element,Part,part_value>) -> Tensors.Tensor<Element>:
  match operand:
    case Stored{part,shape,model,fill,workers,facts}: Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Ready{},workers}
    case Gathered{part,shape,bands,geometry,model,fill,workers,equation,facts}: Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers}

def natives(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,parts: Operands<Element,Part,part_value>) -> Tensors.Parts<Element>:
  match parts:
    case Only{operand}: Tensors.Single{native(~Element,~Part,~part_value,operand)}
    case Also{operand,rest}: Tensors.Cons{native(~Element,~Part,~part_value,operand),natives(~Element,~Part,~part_value,rest)}

# The dense pieces the route appends: a band part contributes its gathered storage.
def pieces(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,parts: Operands<Element,Part,part_value>) -> Concat.Pieces<Element,Part,part_value>:
  match parts:
    case Only{Stored{part,shape,model,fill,workers,facts}}: Concat.Last{part,shape,model,fill,workers,facts}
    case Only{Gathered{part,shape,bands,geometry,model,fill,workers,equation,facts}}: Concat.Last{part,shape,model,fill,workers,facts}
    case Also{Stored{part,shape,model,fill,workers,facts},rest}: Concat.Next{part,shape,model,fill,workers,facts,pieces(~Element,~Part,~part_value,rest)}
    case Also{Gathered{part,shape,bands,geometry,model,fill,workers,equation,facts},rest}: Concat.Next{part,shape,model,fill,workers,facts,pieces(~Element,~Part,~part_value,rest)}

# The band plan the metadata pass finds, and whether every part has the same one.
def layout_of(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,parts: Operands<Element,Part,part_value>) -> Maybe<&2,Bands.Plan>:
  match parts:
    case Only{Stored{part,shape,model,fill,workers,facts}}: None{}
    case Only{Gathered{part,shape,bands,geometry,model,fill,workers,equation,facts}}: Some{Model.layout(~Element,bands)}
    case Also{Stored{part,shape,model,fill,workers,facts},rest}: BandConcat.first_plan(None{},layout_of(~Element,~Part,~part_value,rest))
    case Also{Gathered{part,shape,bands,geometry,model,fill,workers,equation,facts},rest}: BandConcat.first_plan(Some{Model.layout(~Element,bands)},layout_of(~Element,~Part,~part_value,rest))

def compatible(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,parts: Operands<Element,Part,part_value>) -> Bool:
  match parts:
    case Only{Stored{part,shape,model,fill,workers,facts}}: False{}
    case Only{Gathered{part,shape,bands,geometry,model,fill,workers,equation,facts}}: True{}
    case Also{Stored{part,shape,model,fill,workers,facts},+rest}: compatible(~Element,~Part,~part_value,rest) && BandConcat.compatible_plans(None{},layout_of(~Element,~Part,~part_value,rest))
    case Also{Gathered{part,shape,bands,geometry,model,fill,workers,equation,facts},+rest}: compatible(~Element,~Part,~part_value,rest) && BandConcat.compatible_plans(Some{Model.layout(~Element,bands)},layout_of(~Element,~Part,~part_value,rest))

def led(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,parts: Operands<Element,Part,part_value>) ->
  {Layout.lead(~Element,natives(~Element,~Part,~part_value,parts)) == (natives(~Element,~Part,~part_value,parts),(Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),Concat.first_fill(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)))) : Tensors.Parts<Element> & (List<&2,U32> & Element)}:
  match parts:
    case Only{Stored{part,shape,model,fill,workers,facts}}: {==}
    case Only{Gathered{part,shape,bands,geometry,model,fill,workers,equation,facts}}: {==}
    case Also{Stored{part,shape,model,fill,workers,facts},rest}: {==}
    case Also{Gathered{part,shape,bands,geometry,model,fill,workers,equation,facts},rest}: {==}

def surveyed(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,parts: Operands<Element,Part,part_value>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {Layout.survey(~Element,natives(~Element,~Part,~part_value,parts),first,axis,summary) == (natives(~Element,~Part,~part_value,parts),Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts),first,axis,summary)) : Tensors.Parts<Element> & Layout.Survey}:
  match parts:
    case Only{Stored{part,shape,model,fill,workers,facts}}: {==}
    case Only{Gathered{part,shape,bands,geometry,model,fill,workers,equation,facts}}: {==}
    case Also{Stored{part,+shape,+model,+fill,+workers,facts},+rest}:
      %Equal.sym(Tensors.Parts<Element> & Layout.Survey,Layout.survey(~Element,natives(~Element,~Part,~part_value,rest),first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Ready{},workers)),(natives(~Element,~Part,~part_value,rest),Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,rest),first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Ready{},workers))),
        surveyed(~Element,~Part,~part_value,rest,first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Ready{},workers))) :
        {Layout.prepend_survey(~Element,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Ready{},workers},_) == (Tensors.Cons{Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Ready{},workers},natives(~Element,~Part,~part_value,rest)},Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,rest),first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Ready{},workers))) : Tensors.Parts<Element> & Layout.Survey}
      {==}
    case Also{Gathered{part,+shape,+bands,+geometry,+model,+fill,+workers,+equation,facts},+rest}:
      %Equal.sym(Tensors.Parts<Element> & Layout.Survey,Layout.survey(~Element,natives(~Element,~Part,~part_value,rest),first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Ready{},workers)),(natives(~Element,~Part,~part_value,rest),Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,rest),first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Ready{},workers))),
        surveyed(~Element,~Part,~part_value,rest,first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Ready{},workers))) :
        {Layout.prepend_survey(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers},_) == (Tensors.Cons{Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers},natives(~Element,~Part,~part_value,rest)},Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,rest),first,axis,Layout.surveyed(summary,first,shape,axis,Tensors.Ready{},workers))) : Tensors.Parts<Element> & Layout.Survey}
      {==}

def discovered(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,parts: Operands<Element,Part,part_value>) ->
  {BandConcat.discover_compatible(~Element,natives(~Element,~Part,~part_value,parts)) == ((natives(~Element,~Part,~part_value,parts),layout_of(~Element,~Part,~part_value,parts)),compatible(~Element,~Part,~part_value,parts)) : (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool}:
  match parts:
    case Only{Stored{part,shape,model,fill,workers,facts}}: {==}
    case Only{Gathered{part,+shape,+bands,+geometry,+model,+fill,+workers,+equation,facts}}:
      %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,bands)),(Model.pack(~Element,bands),Model.layout(~Element,bands)),Model.plan(~Element,bands)) :
        {BandConcat.single_compatible(~Element,BandConcat.tensor_planned(~Element,geometry,shape,fill,Tensors.Ready{},workers,_)) == ((Tensors.Single{Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers}},Some{Model.layout(~Element,bands)}),True{}) : (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool}
      {==}
    case Also{Stored{part,+shape,+model,+fill,+workers,facts},+rest}:
      %Equal.sym((Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool,BandConcat.discover_compatible(~Element,natives(~Element,~Part,~part_value,rest)),((natives(~Element,~Part,~part_value,rest),layout_of(~Element,~Part,~part_value,rest)),compatible(~Element,~Part,~part_value,rest)),discovered(~Element,~Part,~part_value,rest)) :
        {BandConcat.cons_compatible(~Element,(Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Ready{},workers},None{}),_) == ((Tensors.Cons{Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Ready{},workers},natives(~Element,~Part,~part_value,rest)},BandConcat.first_plan(None{},layout_of(~Element,~Part,~part_value,rest))),compatible(~Element,~Part,~part_value,rest) && BandConcat.compatible_plans(None{},layout_of(~Element,~Part,~part_value,rest))) : (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool}
      {==}
    case Also{Gathered{part,+shape,+bands,+geometry,+model,+fill,+workers,+equation,facts},+rest}:
      %Equal.sym((Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool,BandConcat.discover_compatible(~Element,natives(~Element,~Part,~part_value,rest)),((natives(~Element,~Part,~part_value,rest),layout_of(~Element,~Part,~part_value,rest)),compatible(~Element,~Part,~part_value,rest)),discovered(~Element,~Part,~part_value,rest)) :
        {BandConcat.cons_compatible(~Element,BandConcat.tensor_plan(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers}),_) == ((Tensors.Cons{Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers},natives(~Element,~Part,~part_value,rest)},BandConcat.first_plan(Some{Model.layout(~Element,bands)},layout_of(~Element,~Part,~part_value,rest))),compatible(~Element,~Part,~part_value,rest) && BandConcat.compatible_plans(Some{Model.layout(~Element,bands)},layout_of(~Element,~Part,~part_value,rest))) : (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool}
      %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,bands)),(Model.pack(~Element,bands),Model.layout(~Element,bands)),Model.plan(~Element,bands)) :
        {BandConcat.cons_compatible(~Element,BandConcat.tensor_planned(~Element,geometry,shape,fill,Tensors.Ready{},workers,_),((natives(~Element,~Part,~part_value,rest),layout_of(~Element,~Part,~part_value,rest)),compatible(~Element,~Part,~part_value,rest))) == ((Tensors.Cons{Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers},natives(~Element,~Part,~part_value,rest)},BandConcat.first_plan(Some{Model.layout(~Element,bands)},layout_of(~Element,~Part,~part_value,rest))),compatible(~Element,~Part,~part_value,rest) && BandConcat.compatible_plans(Some{Model.layout(~Element,bands)},layout_of(~Element,~Part,~part_value,rest))) : (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool}
      {==}

# Appending the parts is appending their dense pieces: a band part is
# gathered into the storage its piece models.
def appended(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,parts: Operands<Element,Part,part_value>,-output: Tensors.Tensor<Element>,+axis: U32,+start: U32) ->
  {Layout.append_parts(~Element,natives(~Element,~Part,~part_value,parts),axis,(output,start)) == Layout.append_parts(~Element,Concat.packed(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,(output,start)) : Tensors.Tensor<Element>}:
  match parts:
    case Only{Stored{part,shape,model,fill,workers,facts}}: {==}
    case Only{Gathered{part,+shape,+bands,+geometry,+model,+fill,+workers,+equation,facts}}:
      %equation : {Layout.set_slice(~Element,output,axis,start,Tensors.Tensor{_,shape,fill,Tensors.Ready{},workers}) == Layout.set_slice(~Element,output,axis,start,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Ready{},workers}) : Tensors.Tensor<Element>}
      {==}
    case Also{Stored{part,+shape,+model,+fill,+workers,facts},rest}:
      appended(~Element,~Part,~part_value,rest,Layout.set_slice(~Element,output,axis,start,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Ready{},workers}),axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32))
    case Also{Gathered{part,+shape,+bands,+geometry,+model,+fill,+workers,+equation,facts},+rest}:
      %equation : {Layout.append_parts(~Element,natives(~Element,~Part,~part_value,rest),axis,(Layout.set_slice(~Element,output,axis,start,Tensors.Tensor{_,shape,fill,Tensors.Ready{},workers}),(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32))) == Layout.append_parts(~Element,Concat.packed(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,rest)),axis,(Layout.set_slice(~Element,output,axis,start,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Ready{},workers}),(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32))) : Tensors.Tensor<Element>}
      appended(~Element,~Part,~part_value,rest,Layout.set_slice(~Element,output,axis,start,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Ready{},workers}),axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32))

def appended_dense(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+parts: Operands<Element,Part,part_value>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,
  accepted: {Dense.tensor_ready(~Element,Layout.append_parts(~Element,natives(~Element,~Part,~part_value,parts),axis,(Tensors.with_workers(~Element,Tensors.full(~Element,shape,fill),workers),0))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{fill,Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,shape,0},Layout.append_parts(~Element,natives(~Element,~Part,~part_value,parts),axis,(Tensors.with_workers(~Element,Tensors.full(~Element,shape,fill),workers),0))):
  %Equal.sym(Tensors.Tensor<Element>,Layout.append_parts(~Element,natives(~Element,~Part,~part_value,parts),axis,(Tensors.with_workers(~Element,Tensors.full(~Element,shape,fill),workers),0)),Layout.append_parts(~Element,Concat.packed(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,(Tensors.with_workers(~Element,Tensors.full(~Element,shape,fill),workers),0)),appended(~Element,~Part,~part_value,parts,Tensors.with_workers(~Element,Tensors.full(~Element,shape,fill),workers),axis,0)) : Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{fill,Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,shape,0},_)
  Concat.counted(~Element,~Part,~part_value,Tensors.count_of(shape),pieces(~Element,~Part,~part_value,parts),axis,shape,fill,workers,{==},
    %appended(~Element,~Part,~part_value,parts,Tensors.with_workers(~Element,Tensors.full(~Element,shape,fill),workers),axis,0) : {Dense.tensor_ready(~Element,_) == True{} : Bool}
    accepted)

# The band route is taken only when every part is banded with one plan, the
# output is not empty, and the axis is the channel axis of a counted shape.
def band_route(compatible: Bool,geometry: Maybe<&2,Bands.Geometry>,count: Maybe<&2,U32>,layout: Maybe<&2,Bands.Plan>) -> Bool:
  match compatible geometry count layout:
    case True{} Some{geometry} Some{count} Some{layout}: True{}
    case compatible geometry count layout: False{}

def routed(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,geometry: Maybe<&2,Bands.Geometry>,count: Maybe<&2,U32>,layout: Maybe<&2,Bands.Plan>,+parts: Operands<Element,Part,part_value>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,
  dense: {band_route(True{},geometry,count,layout) == False{} : Bool},
  accepted: {Dense.tensor_ready(~Element,Layout.join_selected(~Element,geometry,count,shape,axis,fill,workers,layout,natives(~Element,~Part,~part_value,parts))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{fill,Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,shape,0},Layout.join_selected(~Element,geometry,count,shape,axis,fill,workers,layout,natives(~Element,~Part,~part_value,parts))):
  match geometry count layout:
    case Some{geometry} Some{count} Some{layout}:
      Empty.absurd(Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{fill,Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,shape,0},Layout.join_selected(~Element,Some{geometry},Some{count},shape,axis,fill,workers,Some{layout},natives(~Element,~Part,~part_value,parts))),WordEncoding.false_true(Equal.sym(Bool,True{},False{},dense)))
    case Some{geometry} Some{count} None{}: appended_dense(~Element,~Part,~part_value,parts,shape,axis,fill,workers,accepted)
    case Some{geometry} None{} Some{layout}: appended_dense(~Element,~Part,~part_value,parts,shape,axis,fill,workers,accepted)
    case Some{geometry} None{} None{}: appended_dense(~Element,~Part,~part_value,parts,shape,axis,fill,workers,accepted)
    case None{} Some{count} Some{layout}: appended_dense(~Element,~Part,~part_value,parts,shape,axis,fill,workers,accepted)
    case None{} Some{count} None{}: appended_dense(~Element,~Part,~part_value,parts,shape,axis,fill,workers,accepted)
    case None{} None{} Some{layout}: appended_dense(~Element,~Part,~part_value,parts,shape,axis,fill,workers,accepted)
    case None{} None{} None{}: appended_dense(~Element,~Part,~part_value,parts,shape,axis,fill,workers,accepted)

def selected(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,flag: Bool,+parts: Operands<Element,Part,part_value>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,
  dense: {band_route(flag,BandConcat.geometry(shape,axis),Tensors.count_of(shape),layout_of(~Element,~Part,~part_value,parts)) == False{} : Bool},
  accepted: {Dense.tensor_ready(~Element,Layout.join_compatible_selected(~Element,flag,shape,axis,fill,workers,(natives(~Element,~Part,~part_value,parts),layout_of(~Element,~Part,~part_value,parts)))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{fill,Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,shape,0},Layout.join_compatible_selected(~Element,flag,shape,axis,fill,workers,(natives(~Element,~Part,~part_value,parts),layout_of(~Element,~Part,~part_value,parts)))):
  match flag:
    case True{}: routed(~Element,~Part,~part_value,BandConcat.geometry(shape,axis),Tensors.count_of(shape),layout_of(~Element,~Part,~part_value,parts),parts,shape,axis,fill,workers,dense,accepted)
    case False{}: appended_dense(~Element,~Part,~part_value,parts,shape,axis,fill,workers,accepted)

def joined_status(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,status: Tensors.Status,valid: Bool,+parts: Operands<Element,Part,part_value>,+first: List<&2,U32>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,
  dense: {band_route(compatible(~Element,~Part,~part_value,parts) && Bool.not(Shapes.has_zero(shape)),BandConcat.geometry(shape,axis),Tensors.count_of(shape),layout_of(~Element,~Part,~part_value,parts)) == False{} : Bool},
  accepted: {Dense.tensor_ready(~Element,Layout.join_status(~Element,status,valid,first,shape,axis,fill,workers,natives(~Element,~Part,~part_value,parts))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{fill,Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,shape,0},Layout.join_status(~Element,status,valid,first,shape,axis,fill,workers,natives(~Element,~Part,~part_value,parts))):
  match status valid:
    case Tensors.Ready{} True{}:
      %Equal.sym((Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool,BandConcat.discover_compatible(~Element,natives(~Element,~Part,~part_value,parts)),((natives(~Element,~Part,~part_value,parts),layout_of(~Element,~Part,~part_value,parts)),compatible(~Element,~Part,~part_value,parts)),discovered(~Element,~Part,~part_value,parts)) : Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{fill,Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,shape,0},Layout.join_compatible(~Element,shape,axis,fill,workers,_))
      selected(~Element,~Part,~part_value,compatible(~Element,~Part,~part_value,parts) && Bool.not(Shapes.has_zero(shape)),parts,shape,axis,fill,workers,dense,
        %discovered(~Element,~Part,~part_value,parts) : {Dense.tensor_ready(~Element,Layout.join_compatible(~Element,shape,axis,fill,workers,_)) == True{} : Bool}
        accepted)
    case Tensors.Ready{} False{}: Empty.absurd(Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{fill,Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,shape,0},Layout.join_status(~Element,Tensors.Ready{},False{},first,shape,axis,fill,workers,natives(~Element,~Part,~part_value,parts))),WordEncoding.false_true(accepted))
    case Tensors.Failed{problem} True{}: Empty.absurd(Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{fill,Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,shape,0},Layout.join_status(~Element,Tensors.Failed{problem},True{},first,shape,axis,fill,workers,natives(~Element,~Part,~part_value,parts))),WordEncoding.false_true(accepted))
    case Tensors.Failed{problem} False{}: Empty.absurd(Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{fill,Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,shape,0},Layout.join_status(~Element,Tensors.Failed{problem},False{},first,shape,axis,fill,workers,natives(~Element,~Part,~part_value,parts))),WordEncoding.false_true(accepted))

def joined_summary(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,summary: Layout.Survey,+parts: Operands<Element,Part,part_value>,+first: List<&2,U32>,+axis: U32,+fill: Element,
  dense: {band_route(compatible(~Element,~Part,~part_value,parts) && Bool.not(Shapes.has_zero(Layout.with_extent(first,U32.to_nat(axis),Concat.survey_total(summary)))),BandConcat.geometry(Layout.with_extent(first,U32.to_nat(axis),Concat.survey_total(summary)),axis),Tensors.count_of(Layout.with_extent(first,U32.to_nat(axis),Concat.survey_total(summary))),layout_of(~Element,~Part,~part_value,parts)) == False{} : Bool},
  accepted: {Dense.tensor_ready(~Element,Layout.joined(~Element,first,axis,fill,(natives(~Element,~Part,~part_value,parts),summary))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{fill,Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,Layout.with_extent(first,U32.to_nat(axis),Concat.survey_total(summary)),0},Layout.joined(~Element,first,axis,fill,(natives(~Element,~Part,~part_value,parts),summary))):
  Layout.Survey{valid,+total,status,+workers} = summary
  joined_status(~Element,~Part,~part_value,status,valid,parts,first,Layout.with_extent(first,U32.to_nat(axis),total),axis,fill,workers,dense,accepted)

# The extents of the result and the route the public concat takes.
def extents(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+parts: Operands<Element,Part,part_value>,+axis: U32) -> List<&2,U32>:
  Layout.with_extent(Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Concat.survey_total(Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts),Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)))),0,Tensors.Ready{},1})))

def banded_join(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+parts: Operands<Element,Part,part_value>,+axis: U32) -> Bool:
  band_route(compatible(~Element,~Part,~part_value,parts) && Bool.not(Shapes.has_zero(extents(~Element,~Part,~part_value,parts,axis))),BandConcat.geometry(extents(~Element,~Part,~part_value,parts,axis),axis),Tensors.count_of(extents(~Element,~Part,~part_value,parts,axis)),layout_of(~Element,~Part,~part_value,parts))

# Ready parts in either layout, on the dense route: a Ready concat result is
# well formed in the extents of the first part with the summed extent at the
# axis, and every index holds appended over the fill value, each band part
# read through its gathered storage.
def concat(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+parts: Operands<Element,Part,part_value>,+axis: U32,
  dense: {banded_join(~Element,~Part,~part_value,parts,axis) == False{} : Bool},
  accepted: {Dense.tensor_ready(~Element,Layout.concat(~Element,natives(~Element,~Part,~part_value,parts),axis)) == True{} : Bool}) ->
  Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{Concat.first_fill(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,Layout.with_extent(Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Concat.survey_total(Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts),Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)))),0,Tensors.Ready{},1}))),0},Layout.concat(~Element,natives(~Element,~Part,~part_value,parts),axis)):
  %Equal.sym(Tensors.Parts<Element> & (List<&2,U32> & Element),Layout.lead(~Element,natives(~Element,~Part,~part_value,parts)),(natives(~Element,~Part,~part_value,parts),(Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),Concat.first_fill(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)))),led(~Element,~Part,~part_value,parts)) :
    Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{Concat.first_fill(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,Layout.with_extent(Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Concat.survey_total(Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts),Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)))),0,Tensors.Ready{},1}))),0},Layout.concat_led(~Element,axis,_))
  %Equal.sym(Tensors.Parts<Element> & Layout.Survey,Layout.survey(~Element,natives(~Element,~Part,~part_value,parts),Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)))),0,Tensors.Ready{},1}),(natives(~Element,~Part,~part_value,parts),Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts),Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)))),0,Tensors.Ready{},1})),surveyed(~Element,~Part,~part_value,parts,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)))),0,Tensors.Ready{},1})) :
    Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{Concat.first_fill(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,Layout.with_extent(Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Concat.survey_total(Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts),Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)))),0,Tensors.Ready{},1}))),0},Layout.joined(~Element,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,Concat.first_fill(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),_))
  joined_summary(~Element,~Part,~part_value,Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts),Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)))),0,Tensors.Ready{},1}),parts,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,Concat.first_fill(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),dense,
    %surveyed(~Element,~Part,~part_value,parts,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)))),0,Tensors.Ready{},1}) : {Dense.tensor_ready(~Element,Layout.joined(~Element,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),axis,Concat.first_fill(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,parts)),_)) == True{} : Bool}
    %led(~Element,~Part,~part_value,parts) : {Dense.tensor_ready(~Element,Layout.concat_led(~Element,axis,_)) == True{} : Bool}
    accepted)

# Native parts observed once each. One relates a native tensor to its
# operand, Observed the native parts to the operands in order.
type One<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element,-tensor: Tensors.Tensor<Element>> is Data:
  One{operand: Operand<Element,Part,part_value>,equation: {tensor == native(~Element,~Part,~part_value,operand) : Tensors.Tensor<Element>}}

def one_stored(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-tensor: Tensors.Tensor<Element>,+part: Part,+shape: List<&2,U32>,
  evidence: GatherValues.Shaped<Element,Part,part_value,part,shape,tensor>) -> One<Element,Part,part_value,tensor>:
  GatherValues.Shaped{+model,+fill,+workers,equation,facts} = evidence
  One{Stored{part,shape,model,fill,workers,facts},equation}

def one_gathered(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+part: Part,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  observation: Buffer.Observation<Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))>,
  prepared: ReadyDensification.Prepared(~Element,~Part,~part_value,part,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers})) -> One<Element,Part,part_value,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Ready{},workers}>:
  Buffer.Observation{+model,+equation} = observation
  Shape.Both{normal,Shape.Both{Shape.Both{well_formed,status},values}} = prepared
  One{Gathered{part,shape,bands,geometry,model,fill,workers,equation,
      %Buffer.packed_contents(~Element,model) : GatherValues.Facts(~Element,~Part,~part_value,part,shape,_)
      %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))),equation) : GatherValues.Facts(~Element,~Part,~part_value,part,shape,Buffer.contents(~Element,_))
      Shape.Both{well_formed,values}},{==}}

# A Ready normal band tensor with complete logical values is a gathered operand.
def one_banded(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-bands: Bands.Bands(~Element),+part: Part,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  observation: Model.Observation<Element,bands>,
  normal: Global.Normal(~Element,Tensors.Banded{bands,geometry,shape,fill,Tensors.Ready{},workers}),
  values: Dense.TensorValues(~Element,~Part,~part_value,part,Tensors.Banded{bands,geometry,shape,fill,Tensors.Ready{},workers})) -> One<Element,Part,part_value,Tensors.Banded{bands,geometry,shape,fill,Tensors.Ready{},workers}>:
  Model.Observation{+model,+equation} = observation
  %equation : One<Element,Part,part_value,Tensors.Banded{_,geometry,shape,fill,Tensors.Ready{},workers}>
  one_gathered(~Element,~Part,~part_value,part,model,geometry,shape,fill,workers,Buffer.observe(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,model),Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
    ReadyDensification.prepared_owner(~Element,~Part,~part_value,part,BandConcatExecution.BandedOwner{model,geometry,shape,fill,Tensors.Ready{},workers},
      %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) : Global.Normal(~Element,Tensors.Banded{_,geometry,shape,fill,Tensors.Ready{},workers})
      normal,{==},
      %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) : Dense.TensorValues(~Element,~Part,~part_value,part,Tensors.Banded{_,geometry,shape,fill,Tensors.Ready{},workers})
      values))

type Observed<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element,-parts: Tensors.Parts<Element>> is Data:
  Observed{operands: Operands<Element,Part,part_value>,equation: {parts == natives(~Element,~Part,~part_value,operands) : Tensors.Parts<Element>}}

def observed_operands(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-parts: Tensors.Parts<Element>,observed: Observed<Element,Part,part_value,parts>) -> Operands<Element,Part,part_value>:
  Observed{operands,equation} = observed
  operands

def observed_single(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-tensor: Tensors.Tensor<Element>,one: One<Element,Part,part_value,tensor>) -> Observed<Element,Part,part_value,Tensors.Single{tensor}>:
  One{+operand,equation} = one
  Observed{Only{operand},Equal.cong(Tensors.Tensor<Element>,Tensors.Parts<Element>,single => Tensors.Single{single},tensor,native(~Element,~Part,~part_value,operand),equation)}

def observed_cons(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-tensor: Tensors.Tensor<Element>,-rest: Tensors.Parts<Element>,one: One<Element,Part,part_value,tensor>,others: Observed<Element,Part,part_value,rest>) -> Observed<Element,Part,part_value,Tensors.Cons{tensor,rest}>:
  One{+operand,equation} = one
  Observed{+operands,others_equation} = others
  Observed{Also{operand,operands},
    %Equal.sym(Tensors.Tensor<Element>,tensor,native(~Element,~Part,~part_value,operand),equation) : {Tensors.Cons{_,rest} == Tensors.Cons{native(~Element,~Part,~part_value,operand),natives(~Element,~Part,~part_value,operands)} : Tensors.Parts<Element>}
    %Equal.sym(Tensors.Parts<Element>,rest,natives(~Element,~Part,~part_value,operands),others_equation) : {Tensors.Cons{native(~Element,~Part,~part_value,operand),_} == Tensors.Cons{native(~Element,~Part,~part_value,operand),natives(~Element,~Part,~part_value,operands)} : Tensors.Parts<Element>}
    {==}}

# The public concat of native Ready parts in either layout, on the dense route.
def concat_native(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-parts: Tensors.Parts<Element>,+axis: U32,+observed: Observed<Element,Part,part_value,parts>,
  dense: {banded_join(~Element,~Part,~part_value,observed_operands(~Element,~Part,~part_value,parts,observed),axis) == False{} : Bool},
  accepted: {Dense.tensor_ready(~Element,Layout.concat(~Element,parts,axis)) == True{} : Bool}) ->
  Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{Concat.first_fill(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,observed_operands(~Element,~Part,~part_value,parts,observed))),Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,observed_operands(~Element,~Part,~part_value,parts,observed))),axis,Layout.with_extent(Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,observed_operands(~Element,~Part,~part_value,parts,observed))),U32.to_nat(axis),Concat.survey_total(Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,observed_operands(~Element,~Part,~part_value,parts,observed)),Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,observed_operands(~Element,~Part,~part_value,parts,observed))),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,observed_operands(~Element,~Part,~part_value,parts,observed))))),0,Tensors.Ready{},1}))),0},Layout.concat(~Element,parts,axis)):
  Observed{+operands,+equation} = observed
  %Equal.sym(Tensors.Parts<Element>,parts,natives(~Element,~Part,~part_value,operands),equation) : Dense.Contract(~Element,~Concat.Appended<Element,Part>,~Concat.appended_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Concat.Appended{Concat.first_fill(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,operands)),Concat.sources_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,operands)),axis,Layout.with_extent(Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,operands)),U32.to_nat(axis),Concat.survey_total(Concat.survey_of(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,operands),Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,operands)),U32.to_nat(axis),Layout.Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Concat.first_shape(~Element,~Part,~part_value,pieces(~Element,~Part,~part_value,operands)))),0,Tensors.Ready{},1}))),0},Layout.concat(~Element,_,axis))
  concat(~Element,~Part,~part_value,operands,axis,dense,
    %equation : {Dense.tensor_ready(~Element,Layout.concat(~Element,_,axis)) == True{} : Bool}
    accepted)
