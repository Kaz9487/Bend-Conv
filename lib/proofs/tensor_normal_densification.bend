# Densification preserves the actual allocation and shape of a ready Normal
# tensor. Gather is a sequence of writes into its real counted allocation;
# zero extents use the same allocation and need no positive-domain premise.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_window.bend as Window
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_dense_observation.bend as DenseObservation
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_shape_count.bend as BandShapeCount
import ./tensor_count_refinement.bend as Count
import ./tensor_allocation_depth.bend as Depth
import ./tensor_dense_contract.bend as Dense
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_band_model.bend as Model
import ./tensor_band_gather.bend as BandGather
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_allocation_refinement.bend as AllocationLaws
import ./write_program.bend as Writes
import ./word_encoding.bend as WordEncoding

def fresh_counted(~Element: Data,+shape: List<&2,U32>,+fill: Element,+count: U32,
  counted: Count.Count<ShapeLaws.elements(shape),count>,allocation: Depth.Ready<count>) ->
  Dense.WellFormed<Element,shape,Buffer.contents(~Element,Tensors.buffer(~Element,fill,count))>:
  Count.Count{exact,bounded} = counted
  Depth.Ready{depth_ok,room} = allocation
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,count)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)},
    AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))) : Dense.WellFormed<Element,shape,_>
  Dense.WellFormed{Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),exact),
    Buffer.content_ready(~Element,Buffer.fresh(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),
      Buffer.fresh_ready(~Element,Tensors.depth_for(count),fill,U32.to_nat(count),depth_ok,room))}

def counted_allocation(+shape: List<&2,U32>,+count: U32,counted: Count.Count<ShapeLaws.elements(shape),count>) -> Depth.Ready<count>:
  Count.Count{exact,bounded} = counted
  Depth.bounded(count,bounded)

def fresh(~Element: Data,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>) ->
  Dense.WellFormed<Element,shape,Buffer.contents(~Element,Tensors.buffer(~Element,fill,Global.count(shape,counted)))>:
  fresh_counted(~Element,shape,fill,Global.count(shape,counted),Global.evidence(shape,counted),
    counted_allocation(shape,Global.count(shape,counted),Global.evidence(shape,counted)))

def gathered_model(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,output: Buffer.Content<Element>,
  well_formed: Dense.WellFormed<Element,shape,output>) ->
  Dense.WellFormed<Element,shape,DenseObservation.gather_content(~Element,geometry,Model.content(~Element,bands),output)>:
  Buffer.Content{+tree,+length} = output
  Dense.WellFormed{length_exact,ready} = well_formed
  %DenseObservation.gather_model_content(~Element,geometry,bands,Buffer.Content{tree,length}) : Dense.WellFormed<Element,shape,_>
  %Equal.sym(Buffer.Content<Element>,BandGather.model(~Element,geometry,bands,Buffer.Content{tree,length}),
    Buffer.Content{Writes.pure(Element,BandGather.refinement_writes(~Element,geometry,bands),tree),length},BandGather.model_writes(~Element,geometry,bands,tree,length)) : Dense.WellFormed<Element,shape,_>
  Dense.WellFormed{length_exact,Writes.content_ready(~Element,BandGather.refinement_writes(~Element,geometry,bands),tree,length,ready)}

def gathered_contents(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+fill: Element) ->
  {Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))) ==
    DenseObservation.gather_content(~Element,geometry,Model.content(~Element,bands),Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))) : Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
    BandGather.model(~Element,geometry,bands,Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
    DenseObservation.gather_content(~Element,geometry,Model.content(~Element,bands),Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
    BandGather.gather(~Element,geometry,bands,Tensors.buffer(~Element,fill,Bands.elements(geometry))),
    DenseObservation.gather_model_content(~Element,geometry,bands,Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))))

def gathered(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,
  coherent: {Window.band_shape(shape,geometry) == True{} : Bool}) ->
  Dense.WellFormed<Element,shape,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))))>:
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
    DenseObservation.gather_content(~Element,geometry,Model.content(~Element,bands),Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))),gathered_contents(~Element,geometry,bands,fill)) : Dense.WellFormed<Element,shape,_>
  gathered_model(~Element,geometry,bands,shape,Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry))),
    %Equal.sym(U32,Bands.elements(geometry),Global.count(shape,counted),BandShapeCount.global_element_count_exact(shape,geometry,counted,coherent)) :
      Dense.WellFormed<Element,shape,Buffer.contents(~Element,Tensors.buffer(~Element,fill,_))>
    fresh(~Element,shape,fill,counted))

def dense_status(~Element: Data,+buffer: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,status: Tensors.Status,+workers: U32,
  normal: Global.Normal(~Element,Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers}),
  ready: {Dense.status_ready(status) == True{} : Bool}) ->
  Dense.TensorWellFormed(~Element,DenseObservation.dense_tensor(~Element,Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers})):
  match status:
    case Tensors.Ready{}:
      Shape.Both{content_ready,Shape.Both{counted,length_exact}} = normal
      Dense.WellFormed{length_exact,content_ready}
    case Tensors.Failed{problem}:
      Empty.absurd(Dense.TensorWellFormed(~Element,DenseObservation.dense_tensor(~Element,Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers})),WordEncoding.false_true(ready))

def band_status(~Element: Data,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,status: Tensors.Status,+workers: U32,
  normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers}),
  ready: {Dense.status_ready(status) == True{} : Bool}) ->
  Dense.TensorWellFormed(~Element,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers})):
  match status:
    case Tensors.Ready{}:
      Shape.Both{metadata,Shape.Both{counted,coherent}} = normal
      gathered(~Element,geometry,bands,shape,fill,counted,coherent)
    case Tensors.Failed{problem}:
      Empty.absurd(Dense.TensorWellFormed(~Element,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers})),WordEncoding.false_true(ready))

def modeled(~Element: Data,+owner: BandConcatExecution.TensorOwnerModel<Element>,
  normal: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),
  ready: {Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == True{} : Bool}) ->
  Dense.TensorWellFormed(~Element,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))):
  match owner:
    case BandConcatExecution.DenseOwner{+buffer,+shape,+fill,+status,+workers}: dense_status(~Element,buffer,shape,fill,status,workers,normal,ready)
    case BandConcatExecution.BandedOwner{+bands,+geometry,+shape,+fill,+status,+workers}: band_status(~Element,bands,geometry,shape,fill,status,workers,normal,ready)

def observation_owner(~Element: Data,+owner: BandConcatExecution.TensorOwnerModel<Element>,+buffer: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  equation: {DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers} : Tensors.Tensor<Element>},
  normal: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),
  ready: {Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == True{} : Bool}) -> Dense.WellFormed<Element,shape,Buffer.content(~Element,buffer)>:
  %Buffer.packed_contents(~Element,buffer) : Dense.WellFormed<Element,shape,_>
  %equation : Dense.TensorWellFormed(~Element,_)
  modeled(~Element,owner,normal,ready)

def observation_modeled(~Element: Data,+owner: BandConcatExecution.TensorOwnerModel<Element>,
  +observation: DenseObservation.DenseObservation<Element,BandConcatExecution.tensor_owner_pack(~Element,owner)>,
  normal: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),
  ready: {Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == True{} : Bool}) ->
  Dense.WellFormed<Element,DenseObservation.observation_shape(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner),observation),
    Buffer.content(~Element,DenseObservation.observation_buffer(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner),observation))>:
  DenseObservation.DenseObservation{+buffer,+shape,+fill,+status,+workers,+equation,status_equal,shape_equal} = observation
  observation_owner(~Element,owner,buffer,shape,fill,status,workers,equation,normal,ready)

def observation_observed(~Element: Data,-input: Tensors.Tensor<Element>,+observation: DenseObservation.DenseObservation<Element,input>,
  source_observation: BandConcatExecution.TensorObservation<Element,input>,+normal: Global.Normal(~Element,input),
  +ready: {Dense.tensor_ready(~Element,input) == True{} : Bool}) ->
  Dense.WellFormed<Element,DenseObservation.observation_shape(~Element,input,observation),Buffer.content(~Element,DenseObservation.observation_buffer(~Element,input,observation))>:
  DenseObservation.DenseObservation{+buffer,+shape,+fill,+status,+workers,+dense_equation,status_equal,shape_equal} = observation
  BandConcatExecution.TensorObservation{+owner,+equation} = source_observation
  observation_owner(~Element,owner,buffer,shape,fill,status,workers,
    Equal.trans(Tensors.Tensor<Element>,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),DenseObservation.dense_tensor(~Element,input),Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers},
      Equal.cong(Tensors.Tensor<Element>,Tensors.Tensor<Element>,DenseObservation.dense_tensor(~Element),BandConcatExecution.tensor_owner_pack(~Element,owner),input,equation),dense_equation),
    %Equal.sym(Tensors.Tensor<Element>,BandConcatExecution.tensor_owner_pack(~Element,owner),input,equation) : Global.Normal(~Element,_) normal,
    %Equal.sym(Tensors.Tensor<Element>,BandConcatExecution.tensor_owner_pack(~Element,owner),input,equation) : {Dense.tensor_ready(~Element,_) == True{} : Bool} ready)

def observation(~Element: Data,input: Tensors.Tensor<Element>,+observed: DenseObservation.DenseObservation<Element,input>,
  normal: Global.Normal(~Element,input),ready: {Dense.tensor_ready(~Element,input) == True{} : Bool}) ->
  Dense.WellFormed<Element,DenseObservation.observation_shape(~Element,input,observed),Buffer.content(~Element,DenseObservation.observation_buffer(~Element,input,observed))>:
  observation_observed(~Element,input,observed,BandConcatExecution.observe_tensor(~Element,input),normal,ready)
