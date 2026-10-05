# The dense form of a tensor: a dense tensor is itself, a banded tensor is
# gathered into fresh storage. An observation names the model of that storage
# with the metadata the operation keeps.
import Base
import ../tensor_layout.bend as Layout
import ../storage_buffer.bend as Storage
import ../tensor_band.bend as Bands
import ../tensor.bend as Tensors
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_model.bend as Model
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_dense_contract.bend as Dense
import ./word_encoding.bend as WordEncoding

def gather_content(~Element: Data,+geometry: Bands.Geometry,bands: Model.Content<Element>,+output: Buffer.Content<Element>) -> Buffer.Content<Element>:
  match bands:
    case Model.RowBand{input,first,rows}: BandGather.leaf_model(~Element,geometry,first,rows,input,output)
    case Model.Joined{left,right}: gather_content(~Element,geometry,right,gather_content(~Element,geometry,left,output))

def gather_model_content(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>,+output: Buffer.Content<Element>) ->
  {BandGather.model(~Element,geometry,bands,output) == gather_content(~Element,geometry,Model.content(~Element,bands),output) : Buffer.Content<Element>}:
  match bands:
    case Model.Leaf{input,first,rows}: {==}
    case Model.Fork{+left,+right}:
      %Equal.sym(Buffer.Content<Element>,BandGather.model(~Element,geometry,left,output),gather_content(~Element,geometry,Model.content(~Element,left),output),
        gather_model_content(~Element,geometry,left,output)) :
        {BandGather.model(~Element,geometry,right,_) == gather_content(~Element,geometry,Model.content(~Element,bands),output) : Buffer.Content<Element>}
      gather_model_content(~Element,geometry,right,gather_content(~Element,geometry,Model.content(~Element,left),output))

def dense_model(~Element: Data,input: BandConcatExecution.TensorContent<Element>) -> BandConcatExecution.TensorContent<Element>:
  match input:
    case BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}: BandConcatExecution.DenseContent{buffer,shape,fill,status,workers}
    case BandConcatExecution.BandedContent{bands,+geometry,shape,+fill,status,workers}:
      BandConcatExecution.DenseContent{gather_content(~Element,geometry,bands,Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))),shape,fill,status,workers}

def dense_contents(~Element: Data,input: Tensors.Tensor<Element>) -> BandConcatExecution.TensorContent<Element>:
  Tensors.with_dense(~Element,~BandConcatExecution.TensorContent<Element>,input,
    buffer => shape => fill => status => workers => BandConcatExecution.DenseContent{Buffer.contents(~Element,buffer),shape,fill,status,workers})

def dense_banded_packed(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32) ->
  {dense_contents(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers}) ==
    dense_model(~Element,BandConcatExecution.tensor_contents(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers})) : BandConcatExecution.TensorContent<Element>}:
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,bands)),Model.content(~Element,bands),Model.packed_contents(~Element,bands)) :
    {dense_contents(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers}) ==
      dense_model(~Element,BandConcatExecution.BandedContent{_,geometry,shape,fill,status,workers}) : BandConcatExecution.TensorContent<Element>}
  Equal.cong(Buffer.Content<Element>,BandConcatExecution.TensorContent<Element>,buffer => BandConcatExecution.DenseContent{buffer,shape,fill,status,workers},
    Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
    gather_content(~Element,geometry,Model.content(~Element,bands),Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
    Equal.trans(Buffer.Content<Element>,
      Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
      BandGather.model(~Element,geometry,bands,Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
      gather_content(~Element,geometry,Model.content(~Element,bands),Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
      BandGather.gather(~Element,geometry,bands,Tensors.buffer(~Element,fill,Bands.elements(geometry))),
      gather_model_content(~Element,geometry,bands,Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry))))))

def dense_banded_observed(~Element: Data,+geometry: Bands.Geometry,-bands: Bands.Bands(~Element),+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  observation: Model.Observation<Element,bands>) ->
  {dense_contents(~Element,Tensors.Banded{bands,geometry,shape,fill,status,workers}) ==
    dense_model(~Element,BandConcatExecution.tensor_contents(~Element,Tensors.Banded{bands,geometry,shape,fill,status,workers})) : BandConcatExecution.TensorContent<Element>}:
  Model.Observation{model,equation} = observation
  %equation : {dense_contents(~Element,Tensors.Banded{_,geometry,shape,fill,status,workers}) ==
    dense_model(~Element,BandConcatExecution.tensor_contents(~Element,Tensors.Banded{_,geometry,shape,fill,status,workers})) : BandConcatExecution.TensorContent<Element>}
  dense_banded_packed(~Element,geometry,model,shape,fill,status,workers)

def dense(~Element: Data,input: Tensors.Tensor<Element>) ->
  {dense_contents(~Element,input) == dense_model(~Element,BandConcatExecution.tensor_contents(~Element,input)) : BandConcatExecution.TensorContent<Element>}:
  match input:
    case Tensors.Tensor{buffer,shape,fill,status,workers}: {==}
    case Tensors.Banded{bands,+geometry,+shape,+fill,+status,+workers}:
      dense_banded_observed(~Element,geometry,bands,shape,fill,status,workers,Model.observe(~Element,bands))

def dense_tensor(~Element: Data,input: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  Tensors.with_dense(~Element,~Tensors.Tensor<Element>,input,
    buffer => shape => fill => status => workers => Tensors.Tensor{buffer,shape,fill,status,workers})

type DenseObservation<-Element: Data,-tensor: Tensors.Tensor<Element>> is Data:
  DenseObservation{buffer: Buffer.Model<Element>,shape: List<&2,U32>,fill: Element,status: Tensors.Status,workers: U32,
    equation: {dense_tensor(~Element,tensor) == Tensors.Tensor{Buffer.pack(~Element,buffer),shape,fill,status,workers} : Tensors.Tensor<Element>},
    status_equal: {Dense.tensor_ready(~Element,tensor) == Dense.status_ready(status) : Bool},
    shape_equal: {BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,tensor)) == shape : List<&2,U32>}}

def observed_dense_buffer(~Element: Data,-input: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  observation: Buffer.Observation<Element,input>) -> DenseObservation<Element,Tensors.Tensor{input,shape,fill,status,workers}>:
  Buffer.Observation{+model,equation} = observation
  DenseObservation{model,shape,fill,status,workers,
    Equal.cong(Storage.Buffer<Element>,Tensors.Tensor<Element>,buffer => Tensors.Tensor{buffer,shape,fill,status,workers},
      input,Buffer.pack(~Element,model),Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),input,equation)),{==},{==}}

def observed_dense_bands(~Element: Data,-bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  observation: Buffer.Observation<Element,Bands.gather(~Element,geometry,bands,Tensors.buffer(~Element,fill,Bands.elements(geometry)))>) ->
  DenseObservation<Element,Tensors.Banded{bands,geometry,shape,fill,status,workers}>:
  Buffer.Observation{+model,equation} = observation
  DenseObservation{model,shape,fill,status,workers,
    Equal.cong(Storage.Buffer<Element>,Tensors.Tensor<Element>,buffer => Tensors.Tensor{buffer,shape,fill,status,workers},
      Bands.gather(~Element,geometry,bands,Tensors.buffer(~Element,fill,Bands.elements(geometry))),Buffer.pack(~Element,model),
      Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),Bands.gather(~Element,geometry,bands,Tensors.buffer(~Element,fill,Bands.elements(geometry))),equation)),{==},{==}}

def observe_dense(~Element: Data,input: Tensors.Tensor<Element>) -> DenseObservation<Element,input>:
  match input:
    case Tensors.Tensor{buffer,+shape,+fill,+status,+workers}:
      observed_dense_buffer(~Element,buffer,shape,fill,status,workers,Buffer.observe(~Element,buffer))
    case Tensors.Banded{bands,+geometry,+shape,+fill,+status,+workers}:
      observed_dense_bands(~Element,bands,geometry,shape,fill,status,workers,
        Buffer.observe(~Element,Bands.gather(~Element,geometry,bands,Tensors.buffer(~Element,fill,Bands.elements(geometry)))))

def observation_buffer(~Element: Data,-input: Tensors.Tensor<Element>,observation: DenseObservation<Element,input>) -> Buffer.Model<Element>:
  DenseObservation{buffer,shape,fill,status,workers,equation,status_equal,shape_equal} = observation
  buffer

def observation_shape(~Element: Data,-input: Tensors.Tensor<Element>,observation: DenseObservation<Element,input>) -> List<&2,U32>:
  DenseObservation{buffer,shape,fill,status,workers,equation,status_equal,shape_equal} = observation
  shape

def ready_status(+status: Tensors.Status,ready: {Dense.status_ready(status) == True{} : Bool}) -> {status == Tensors.Ready{} : Tensors.Status}:
  match status:
    case Tensors.Ready{}: {==}
    case Tensors.Failed{problem}:
      Empty.absurd({Tensors.Failed{problem} == Tensors.Ready{} : Tensors.Status},WordEncoding.false_true(ready))

def set_slice_densified(~Element: Data,target: Tensors.Tensor<Element>,+axis: U32,+start: U32,part: Tensors.Tensor<Element>) ->
  {Layout.set_slice(~Element,target,axis,start,part) == Layout.set_slice(~Element,dense_tensor(~Element,target),axis,start,dense_tensor(~Element,part)) : Tensors.Tensor<Element>}:
  match target part:
    case Tensors.Tensor{target,target_shape,target_fill,target_status,target_workers} Tensors.Tensor{part,part_shape,part_fill,part_status,part_workers}: {==}
    case Tensors.Tensor{target,target_shape,target_fill,target_status,target_workers} Tensors.Banded{part,geometry,part_shape,part_fill,part_status,part_workers}: {==}
    case Tensors.Banded{target,target_geometry,target_shape,target_fill,target_status,target_workers} Tensors.Tensor{part,part_shape,part_fill,part_status,part_workers}: {==}
    case Tensors.Banded{target,target_geometry,target_shape,target_fill,target_status,target_workers} Tensors.Banded{part,part_geometry,part_shape,part_fill,part_status,part_workers}: {==}

def append_at_densified(~Element: Data,+axis: U32,+start: U32,target: Tensors.Tensor<Element>,part: Tensors.Tensor<Element>) ->
  {Layout.append_at(~Element,axis,start,target,part) ==
    Layout.append_at(~Element,axis,start,dense_tensor(~Element,target),dense_tensor(~Element,part)) : Tensors.Tensor<Element> & U32}:
  match target part:
    case Tensors.Tensor{target,target_shape,target_fill,target_status,target_workers} Tensors.Tensor{part,part_shape,part_fill,part_status,part_workers}: {==}
    case Tensors.Tensor{target,target_shape,target_fill,target_status,target_workers} Tensors.Banded{part,geometry,part_shape,part_fill,part_status,part_workers}: {==}
    case Tensors.Banded{target,target_geometry,target_shape,target_fill,target_status,target_workers} Tensors.Tensor{part,part_shape,part_fill,part_status,part_workers}: {==}
    case Tensors.Banded{target,target_geometry,target_shape,target_fill,target_status,target_workers} Tensors.Banded{part,part_geometry,part_shape,part_fill,part_status,part_workers}: {==}
