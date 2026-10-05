# Ready Normal tensors keep their actual shape and storage contract when
# with_dense gathers their owners. The output count is completed from the
# existing physical WellFormed producer, never supplied by the caller.
import Base
import ../tensor.bend as Tensors
import ./storage_buffer_model.bend as Buffer
import ./tensor_dense_contract.bend as Dense
import ./tensor_dense_normal.bend as DenseNormal
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_dense_observation.bend as DenseObservation
import ./tensor_normal_densification.bend as Normal
import ./tensor_normal_densification_values.bend as NormalDensificationValues
import ./storage_shape.bend as Shape

def observed_tensor(~Element: Data,-input: Tensors.Tensor<Element>,observation: DenseObservation.DenseObservation<Element,input>) -> Tensors.Tensor<Element>:
  DenseObservation.DenseObservation{buffer,shape,fill,status,workers,equation,status_equal,shape_equal} = observation
  Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers}

def observation_observed(~Element: Data,-input: Tensors.Tensor<Element>,+observation: DenseObservation.DenseObservation<Element,input>,
  source_observation: BandConcatExecution.TensorObservation<Element,input>,
  normal: Global.Normal(~Element,input),ready: {Dense.tensor_ready(~Element,input) == True{} : Bool}) ->
  Global.Normal(~Element,observed_tensor(~Element,input,observation)):
  DenseObservation.DenseObservation{+buffer,+shape,+fill,+status,+workers,+equation,+status_equal,+shape_equal} = observation
  DenseNormal.from_well_formed(~Element,Buffer.pack(~Element,buffer),shape,fill,status,workers,
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,buffer)),Buffer.content(~Element,buffer),Buffer.packed_contents(~Element,buffer)) :
      Dense.WellFormed<Element,shape,_>
    Normal.observation_observed(~Element,input,observation,source_observation,normal,ready))

def observation_modeled(~Element: Data,+owner: BandConcatExecution.TensorOwnerModel<Element>,+observation: DenseObservation.DenseObservation<Element,BandConcatExecution.tensor_owner_pack(~Element,owner)>,
  normal: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),
  ready: {Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == True{} : Bool}) ->
  Global.Normal(~Element,observed_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner),observation)):
  DenseObservation.DenseObservation{+buffer,+shape,+fill,+status,+workers,+equation,+status_equal,+shape_equal} = observation
  DenseNormal.from_well_formed(~Element,Buffer.pack(~Element,buffer),shape,fill,status,workers,
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,buffer)),Buffer.content(~Element,buffer),Buffer.packed_contents(~Element,buffer)) :
      Dense.WellFormed<Element,shape,_>
    Normal.observation_modeled(~Element,owner,observation,normal,ready))

def observation_normal(~Element: Data,input: Tensors.Tensor<Element>,+observation: DenseObservation.DenseObservation<Element,input>,
  normal: Global.Normal(~Element,input),ready: {Dense.tensor_ready(~Element,input) == True{} : Bool}) ->
  Global.Normal(~Element,observed_tensor(~Element,input,observation)):
  observation_observed(~Element,input,observation,BandConcatExecution.observe_tensor(~Element,input),normal,ready)

def dense_observed(~Element: Data,-input: Tensors.Tensor<Element>,+observation: DenseObservation.DenseObservation<Element,input>,
  normal: Global.Normal(~Element,observed_tensor(~Element,input,observation))) ->
  Global.Normal(~Element,DenseObservation.dense_tensor(~Element,input)):
  DenseObservation.DenseObservation{buffer,shape,fill,status,workers,equation,status_equal,shape_equal} = observation
  %Equal.sym(Tensors.Tensor<Element>,DenseObservation.dense_tensor(~Element,input),Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers},equation) : Global.Normal(~Element,_)
  normal

def modeled(~Element: Data,+owner: BandConcatExecution.TensorOwnerModel<Element>,normal: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),
  ready: {Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == True{} : Bool}) ->
  Global.Normal(~Element,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))):
  +observation = DenseObservation.observe_dense(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))
  dense_observed(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner),observation,observation_modeled(~Element,owner,observation,normal,ready))

def observed_normal(~Element: Data,-input: Tensors.Tensor<Element>,source_observation: BandConcatExecution.TensorObservation<Element,input>,
  incoming: Global.Normal(~Element,input),ready: {Dense.tensor_ready(~Element,input) == True{} : Bool}) ->
  Global.Normal(~Element,DenseObservation.dense_tensor(~Element,input)):
  BandConcatExecution.TensorObservation{+owner,+equation} = source_observation
  %equation : Global.Normal(~Element,DenseObservation.dense_tensor(~Element,_))
  modeled(~Element,owner,
    %Equal.sym(Tensors.Tensor<Element>,BandConcatExecution.tensor_owner_pack(~Element,owner),input,equation) : Global.Normal(~Element,_) incoming,
    %Equal.sym(Tensors.Tensor<Element>,BandConcatExecution.tensor_owner_pack(~Element,owner),input,equation) : {Dense.tensor_ready(~Element,_) == True{} : Bool} ready)

def normal(~Element: Data,input: Tensors.Tensor<Element>,incoming: Global.Normal(~Element,input),ready: {Dense.tensor_ready(~Element,input) == True{} : Bool}) ->
  Global.Normal(~Element,DenseObservation.dense_tensor(~Element,input)):
  observed_normal(~Element,input,BandConcatExecution.observe_tensor(~Element,input),incoming,ready)

# A single preparation contract serves every dense fallback. It combines the
# already checked shape, ownership and finite-value gather laws; it does not
# require a positive shape or a successful downstream operation.
def Prepared(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,-input: Tensors.Tensor<Element>) -> Data:
  Shape.Both<Global.Normal(~Element,DenseObservation.dense_tensor(~Element,input)),
    Dense.Contract(~Element,~Logical,~original,logical,DenseObservation.dense_tensor(~Element,input))>

def ready_owner(~Element: Data,owner: BandConcatExecution.TensorOwnerModel<Element>) ->
  {Dense.tensor_ready(~Element,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))) ==
    Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) : Bool}:
  match owner:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}: {==}
    case BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers}: {==}

def prepared_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+owner: BandConcatExecution.TensorOwnerModel<Element>,
  +observation: DenseObservation.DenseObservation<Element,BandConcatExecution.tensor_owner_pack(~Element,owner)>,
  +incoming: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),
  +ready: {Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == True{} : Bool},
  values: Dense.TensorValues(~Element,~Logical,~original,logical,BandConcatExecution.tensor_owner_pack(~Element,owner))) ->
  Prepared(~Element,~Logical,~original,logical,BandConcatExecution.tensor_owner_pack(~Element,owner)):
  DenseObservation.DenseObservation{+buffer,+shape,+fill,+status,+workers,+equation,+status_equal,+shape_equal} = observation
  +target_ready = Equal.trans(Bool,Dense.tensor_ready(~Element,Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers}),
    Dense.tensor_ready(~Element,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))),True{},
    Equal.cong(Tensors.Tensor<Element>,Bool,Dense.tensor_ready(~Element),Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers},
      DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),Equal.sym(Tensors.Tensor<Element>,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),
        Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers},equation)),
    Equal.trans(Bool,Dense.tensor_ready(~Element,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))),
      Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),True{},ready_owner(~Element,owner),ready))
  Shape.Both{dense_observed(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner),observation,observation_modeled(~Element,owner,observation,incoming,ready)),
    %Equal.sym(Tensors.Tensor<Element>,DenseObservation.dense_tensor(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers},equation) :
      Dense.Contract(~Element,~Logical,~original,logical,_)
    Shape.Both{Shape.Both{
        %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,buffer)),Buffer.content(~Element,buffer),Buffer.packed_contents(~Element,buffer)) :
          Dense.WellFormed<Element,shape,_>
        Normal.observation_modeled(~Element,owner,observation,incoming,ready),target_ready},
      %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,buffer)),Buffer.content(~Element,buffer),Buffer.packed_contents(~Element,buffer)) :
        Dense.Values(~Element,~Logical,~original,logical,_)
      NormalDensificationValues.observation_owner(~Element,~Logical,~original,logical,owner,buffer,shape,fill,status,workers,equation,incoming,ready,values)}}

def prepared_owner(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+owner: BandConcatExecution.TensorOwnerModel<Element>,
  incoming: Global.Normal(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),
  ready: {Dense.tensor_ready(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)) == True{} : Bool},
  values: Dense.TensorValues(~Element,~Logical,~original,logical,BandConcatExecution.tensor_owner_pack(~Element,owner))) ->
  Prepared(~Element,~Logical,~original,logical,BandConcatExecution.tensor_owner_pack(~Element,owner)):
  prepared_observed(~Element,~Logical,~original,logical,owner,DenseObservation.observe_dense(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),incoming,ready,values)
