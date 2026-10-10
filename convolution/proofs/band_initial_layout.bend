# The initial band layout: its contract and worker counts.
import Base
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_tensor_execution.bend as TensorExecution
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/tensor_dense_contract.bend as Dense
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_tile.bend as Tile
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./band_dense_public.bend as DensePublic
import ./band_output_metadata.bend as OutputMetadata
import ./band_output_normal.bend as BandOutputNormal
import ../../lib/default_machine_limits.bend as Default
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/machine_limits.bend as Machine
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_normal_unary.bend as Unary
import ../../lib/proofs/tensor_dense_normal.bend as DenseNormal
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_window.bend as Window

# Dense and band initial layouts share finite output semantics and Normal.
# The band producer keeps its existing leaf proof and constructs this envelope
# from the actual output header; no successful output contract is assumed.
def LayoutContract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,-tensor: Tensors.Tensor<F32>) -> Data:
  Shape.Both<Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,tensor),Global.Normal(~F32,tensor)>

def settings_logical(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>) ->
  {PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) == frame : PipelineExecution.Frame<Logical>}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  {==}

def Related(~Logical: Data,frame: PipelineExecution.Frame<Logical>,context: BandConvolutionValues.Context<Logical>) -> Data:
  PipelineExecution.Related(~Logical,context,frame)

def agreement(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+context: BandConvolutionValues.Context<Logical>,
  +related: Related(~Logical,frame,context),+index: U32) ->
  {PipelineExecution.result_value(~custom,~Logical,~original,frame,index) == BandConvolutionValues.value(~custom,~Logical,~original,context,index) : F32}:
  Equal.sym(F32,BandConvolutionValues.value(~custom,~Logical,~original,context,index),PipelineExecution.result_value(~custom,~Logical,~original,frame,index),PipelineExecution.agreement(~custom,~Logical,~original,context,frame,related,index))

def band_values(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+geometry: Bands.Geometry,+content: Model.Content<F32>,
  values: BandValues.Values(~F32,~PipelineExecution.Frame<Logical>,~PipelineExecution.result_value(~custom,~Logical,~original),PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),geometry,content)) ->
  BandValues.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),PipelineExecution.output_context(~Logical,frame),geometry,content):
  BandValues.values(~F32,~PipelineExecution.Frame<Logical>,~BandConvolutionValues.Context<Logical>,~PipelineExecution.result_value(~custom,~Logical,~original),~BandConvolutionValues.value(~custom,~Logical,~original),
    ~Related(~Logical),~agreement(~custom,~Logical,~original),frame,PipelineExecution.output_context(~Logical,frame),{==},geometry,content,
    %settings_logical(~Logical,~original,frame) :
      BandValues.Values(~F32,~PipelineExecution.Frame<Logical>,~PipelineExecution.result_value(~custom,~Logical,~original),_,geometry,content)
    values)

def from_band_header(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,tensor: Tensors.Tensor<F32>,
  header: BandOutputNormal.Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),tensor),owners: TensorExecution.OwnersContract(~custom,~Logical,~original,frame,tensor),normal: Global.Normal(~F32,tensor)) ->
  LayoutContract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),tensor):
  match tensor:
    case Tensors.Banded{bands,+geometry,shape,fill,+status,workers}:
      Shape.Both{geometry_exact,Shape.Both{shape_exact,status_exact}} = header
      Shape.Both{values,well_formed} = owners
      Shape.Both{Shape.Both{Shape.Both{well_formed,Equal.cong(Tensors.Status,Bool,Dense.status_ready,status,Tensors.Usable{},status_exact)},
        band_values(~custom,~Logical,~original,frame,geometry,Model.contents(~F32,bands),values)},normal}
    case Tensors.Tensor{storage,shape,fill,status,workers}:
      Empty.absurd(LayoutContract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),Tensors.Tensor{storage,shape,fill,status,workers}),header)

def from_initial(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,tensor: Tensors.Tensor<F32>,
  header: BandOutputNormal.Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),tensor),initialized: DensePublic.InitialContract(~custom,~Logical,~original,frame,tensor)) ->
  LayoutContract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),tensor):
  Shape.Both{owners,normal} = initialized
  from_band_header(~custom,~Logical,~original,frame,batched,tensor,header,owners,normal)

def start_header(~custom: F32 -> F32,~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+depth: U32,+workers: U32,+source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>) ->
  BandOutputNormal.Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),TensorConvolution.start_bands(~custom,TensorExecution.execution_activation(~Logical,frame),depth,TensorExecution.execution_parallel(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)):
  %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.start_bands(~custom,TensorExecution.execution_activation(~Logical,frame),depth,TensorExecution.execution_parallel(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),TensorExecution.execution_parallel(~Logical,frame),Bands.split(~F32,U32.to_nat(depth),PipelineExecution.input_geometry(~Logical,frame),0.0,Buffer.pack(~F32,source)),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    DensePublic.start_bands_matches(~custom,~Logical,frame,batched,depth,workers,Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) : BandOutputNormal.Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),_)
  BandOutputNormal.band_output_header(~custom,~Logical,frame,batched,TensorExecution.execution_parallel(~Logical,frame),workers,Bands.split(~F32,U32.to_nat(depth),PipelineExecution.input_geometry(~Logical,frame),0.0,Buffer.pack(~F32,source)),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))

def start_bands(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+depth: U32,+workers: U32,
  +source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),shape,fill,Tensors.Usable{},workers}),
  coherent: {Window.band_shape(shape,PipelineExecution.input_geometry(~Logical,frame)) == True{} : Bool},
  matched: {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  source_values: Dense.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),Buffer.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  LayoutContract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.start_bands(~custom,TensorExecution.execution_activation(~Logical,frame),depth,TensorExecution.execution_parallel(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)):
  from_initial(~custom,~Logical,~original,frame,batched,
    TensorConvolution.start_bands(~custom,TensorExecution.execution_activation(~Logical,frame),depth,TensorExecution.execution_parallel(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    start_header(~custom,~Logical,frame,batched,depth,workers,source,weights,bias),
    DensePublic.start_bands(~custom,~Logical,~original,frame,batched,depth,workers,source,weights,bias,shape,fill,normal,coherent,matched,limits_exact,source_values,weight_exact,bias_exact))

# The actual workers wrapper preserves both layouts' finite contract. Dense
# output is identified by its real constructor before deriving Normal; the
# common Contract alone does not assert that its representation is dense.
def contract_workers(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+workers: U32,tensor: Tensors.Tensor<Element>,
  contract: Dense.Contract(~Element,~Logical,~original,logical,tensor)) -> Dense.Contract(~Element,~Logical,~original,logical,Tensors.with_workers(~Element,tensor,workers)):
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,old_workers}: contract
    case Tensors.Banded{bands,geometry,shape,fill,status,old_workers}: contract

def with_workers(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+workers: U32,tensor: Tensors.Tensor<F32>,
  evidence: LayoutContract(~custom,~Logical,~original,context,tensor)) -> LayoutContract(~custom,~Logical,~original,context,Tensors.with_workers(~F32,tensor,workers)):
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,old_workers}:
      Shape.Both{contract,normal} = evidence
      Shape.Both{contract,Unary.with_workers(~F32,workers,Tensors.Tensor{storage,shape,fill,status,old_workers},normal)}
    case Tensors.Banded{bands,geometry,shape,fill,status,old_workers}:
      Shape.Both{contract,normal} = evidence
      Shape.Both{contract,Unary.with_workers(~F32,workers,Tensors.Banded{bands,geometry,shape,fill,status,old_workers},normal)}

def DenseHeader(~Element: Data,tensor: Tensors.Tensor<Element>) -> Data:
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,workers}: Unit
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: Empty

def dense_normal(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,tensor: Tensors.Tensor<Element>,
  header: DenseHeader(~Element,tensor),contract: Dense.Contract(~Element,~Logical,~original,logical,tensor)) -> Global.Normal(~Element,tensor):
  match tensor:
    case Tensors.Tensor{storage,+shape,+fill,+status,+workers}:
      DenseNormal.from_contract(~Element,~Logical,~original,logical,storage,shape,fill,status,workers,contract)
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}:
      Empty.absurd(Global.Normal(~Element,Tensors.Banded{bands,geometry,shape,fill,status,workers}),header)

def from_dense_header(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,tensor: Tensors.Tensor<F32>,
  header: DenseHeader(~F32,tensor),+contract: Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,tensor)) ->
  LayoutContract(~custom,~Logical,~original,context,tensor):
  Shape.Both{contract,dense_normal(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,tensor,header,contract)}

def arguments_header(~custom: F32 -> F32,+activation: Activations.Activation,+arguments: TensorConvolution.DenseArguments,
  -input: Storage.Buffer<F32>,-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>,+workers: U32) ->
  DenseHeader(~F32,TensorConvolution.execute_dense_arguments(~custom,activation,arguments,input,weights,bias,workers)):
  TensorConvolution.DenseArguments{count,work,input_size,output_size,extents,selected} = arguments
  Unit{}

def execute_dense_header(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: TensorConvolution.Geometry,
  -input: Storage.Buffer<F32>,-weights: Storage.Buffer<F32>,+bias: Buffer.Model<F32>,+workers: U32) ->
  DenseHeader(~F32,TensorConvolution.execute_dense(~custom,activation,geometry,input,TensorConvolution.packed(geometry,weights),Buffer.pack(~F32,bias),workers)):
  Buffer.Model{tree,length,Certificate.Certificate{+depth,equation}} = bias
  arguments_header(~custom,activation,TensorConvolution.dense_arguments(geometry,depth,workers,activation),input,TensorConvolution.packed(geometry,weights),Buffer.pack(~F32,bias),workers)

def dense_workers(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+workers: U32,tensor: Tensors.Tensor<F32>,
  header: DenseHeader(~F32,tensor),+contract: Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,tensor)) ->
  LayoutContract(~custom,~Logical,~original,context,Tensors.with_workers(~F32,tensor,workers)):
  match tensor:
    case Tensors.Tensor{storage,+shape,+fill,+status,old_workers}:
      Shape.Both{contract,DenseNormal.from_contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,storage,shape,fill,status,U32.max(1,workers),contract)}
    case Tensors.Banded{bands,geometry,shape,fill,status,old_workers}:
      Empty.absurd(LayoutContract(~custom,~Logical,~original,context,Tensors.with_workers(~F32,Tensors.Banded{bands,geometry,shape,fill,status,old_workers},workers)),header)
