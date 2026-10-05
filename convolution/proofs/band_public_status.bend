# Public status and replanning of band convolution.
import Base
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_tensor_execution.bend as TensorExecution
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./band_pipeline_well_formed.bend as PipelineWellFormed
import ./band_public_frame.bend as PublicFrame
import ./band_tensor_replanning.bend as Replan
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/tensor.bend as Tensors

# The public existing-layout caller obtains its plan from the actual band
# owners. Observe each owner once, retain its content contracts, and reuse the
# accepted replan schedule with the fixed ordered convolution context.
def planned(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers,Bands.plan(~F32,Model.pack(~F32,source)))):
  %Equal.sym(Bands.Bands(~F32) & Bands.Plan,Bands.plan(~F32,Model.pack(~F32,source)),(Model.pack(~F32,source),Model.layout(~F32,source)),Model.plan(~F32,source)) :
    PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
      TensorConvolution.existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers,_))
  Replan.existing_layout(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,shape,values,weight_exact,bias_exact)

def observed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  -source: Bands.Bands(~F32),-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>,
  source_observation: Model.Observation<F32,source>,weight_observation: Buffer.Observation<F32,weights>,bias_observation: Buffer.Observation<F32,bias>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,source)),
  weight_exact: {Buffer.content_tree(~F32,Buffer.contents(~F32,weights)) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.contents(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),weights,bias,workers,Bands.plan(~F32,source))):
  Model.Observation{+source_model,+source_equal} = source_observation
  Buffer.Observation{+weight_model,+weights_equal} = weight_observation
  Buffer.Observation{+bias_model,+bias_equal} = bias_observation
  %source_equal : PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),weights,bias,workers,Bands.plan(~F32,_)))
  %weights_equal : PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),_,bias,workers,Bands.plan(~F32,Model.pack(~F32,source_model))))
  %bias_equal : PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weight_model),_,workers,Bands.plan(~F32,Model.pack(~F32,source_model))))
  planned(~custom,~Logical,~original,frame,batched,workers,source_model,weight_model,bias_model,
    PipelineWellFormed.source_shape(~Logical,frame,source_model,source,source_equal,shape),TensorExecution.source_values(~Logical,~original,frame,source_model,source,source_equal,values),
    Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,weight_model),Buffer.content_tree(~F32,Buffer.contents(~F32,weights)),PipelineExecution.source_tree(~Logical,frame),
      Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,weight_model),Buffer.content_tree(~F32,Buffer.content(~F32,weight_model)),Buffer.content_tree(~F32,Buffer.contents(~F32,weights)),
        TensorExecution.model_tree(weight_model),Equal.cong(Buffer.Content<F32>,Tree.Tree<F32>,Buffer.content_tree(~F32),Buffer.content(~F32,weight_model),Buffer.contents(~F32,weights),TensorExecution.buffer_contents(~F32,weight_model,weights,weights_equal))),weight_exact),
    Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,bias_model),Buffer.contents(~F32,bias),PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),TensorExecution.buffer_contents(~F32,bias_model,bias,bias_equal),bias_exact))

def owners(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  source: Bands.Bands(~F32),weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,source)),
  weight_exact: {Buffer.content_tree(~F32,Buffer.contents(~F32,weights)) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.contents(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),weights,bias,workers,Bands.plan(~F32,source))):
  observed(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,Model.observe(~F32,source),Buffer.observe(~F32,weights),Buffer.observe(~F32,bias),shape,values,weight_exact,bias_exact)

# Program ownership and sticky status at the actual convolution caller.
# These equalities cover failed branches independently of value acceptance.
def failed_input(~custom: F32 -> F32,+activation: Activations.Activation,+problem: Tensors.Problem,+layout: Bands.Geometry,
  input: Bands.Bands(~F32),+shape: List<&2,U32>,+fill: F32,weights: Storage.Buffer<F32>,weight_shape: List<&2,U32>,bias: Storage.Buffer<F32>,bias_shape: List<&2,U32>,stride: U32,padding: U32,+workers: U32) ->
  {TensorConvolution.band_status(~custom,activation,Tensors.Failed{problem},layout,input,shape,fill,weights,weight_shape,bias,bias_shape,stride,padding,workers) ==
    Tensors.Banded{input,layout,shape,fill,Tensors.Failed{problem},workers} : Tensors.Tensor<F32>}:
  {==}

def failed_parameters(~custom: F32 -> F32,+activation: Activations.Activation,+problem: Tensors.Problem,+layout: Bands.Geometry,
  input: Bands.Bands(~F32),+shape: List<&2,U32>,+fill: F32,+input_workers: U32,weights: Storage.Buffer<F32>,weight_shape: List<&2,U32>,bias: Storage.Buffer<F32>,bias_shape: List<&2,U32>,+parameter_workers: U32,stride: U32,padding: U32) ->
  {TensorConvolution.input_layout(~custom,activation,Tensors.Banded{input,layout,shape,fill,Tensors.Ready{},input_workers},weights,weight_shape,bias,bias_shape,Tensors.Failed{problem},parameter_workers,stride,padding) ==
    Tensors.Banded{input,layout,shape,fill,Tensors.Failed{problem},U32.max(input_workers,parameter_workers)} : Tensors.Tensor<F32>}:
  {==}

def first_input_failure(~custom: F32 -> F32,+activation: Activations.Activation,+problem: Tensors.Problem,+layout: Bands.Geometry,
  input: Bands.Bands(~F32),+shape: List<&2,U32>,+fill: F32,+input_workers: U32,weights: Storage.Buffer<F32>,weight_shape: List<&2,U32>,bias: Storage.Buffer<F32>,bias_shape: List<&2,U32>,parameter_status: Tensors.Status,+parameter_workers: U32,stride: U32,padding: U32) ->
  {TensorConvolution.input_layout(~custom,activation,Tensors.Banded{input,layout,shape,fill,Tensors.Failed{problem},input_workers},weights,weight_shape,bias,bias_shape,parameter_status,parameter_workers,stride,padding) ==
    Tensors.Banded{input,layout,shape,fill,Tensors.Failed{problem},U32.max(input_workers,parameter_workers)} : Tensors.Tensor<F32>}:
  {==}

def dense_parameters_failed_input(~custom: F32 -> F32,+activation: Activations.Activation,+problem: Tensors.Problem,+layout: Bands.Geometry,
  input: Bands.Bands(~F32),+shape: List<&2,U32>,+fill: F32,+input_workers: U32,weights: Storage.Buffer<F32>,weight_shape: List<&2,U32>,weight_fill: F32,weight_status: Tensors.Status,+weight_workers: U32,
  bias: Storage.Buffer<F32>,bias_shape: List<&2,U32>,bias_fill: F32,bias_status: Tensors.Status,+bias_workers: U32,stride: U32,padding: U32) ->
  {TensorConvolution.conv2d(~custom,activation,Tensors.Banded{input,layout,shape,fill,Tensors.Failed{problem},input_workers},
      Tensors.Tensor{weights,weight_shape,weight_fill,weight_status,weight_workers},Tensors.Tensor{bias,bias_shape,bias_fill,bias_status,bias_workers},stride,padding) ==
    Tensors.Banded{input,layout,shape,fill,Tensors.Failed{problem},U32.max(input_workers,U32.max(weight_workers,bias_workers))} : Tensors.Tensor<F32>}:
  {==}

def invalid_geometry(~custom: F32 -> F32,+activation: Activations.Activation,+layout: Bands.Geometry,input: Bands.Bands(~F32),weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {TensorConvolution.band_checked(~custom,activation,None{},layout,input,weights,bias,shape,fill,workers) ==
    Tensors.Banded{input,layout,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidArgument{shape}}},workers} : Tensors.Tensor<F32>}:
  {==}

def rejected_inspection(~custom: F32 -> F32,+activation: Activations.Activation,geometry: TensorConvolution.Geometry,+layout: Bands.Geometry,input: Bands.Bands(~F32),weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {TensorConvolution.band_supported(~custom,activation,False{},geometry,layout,input,weights,bias,shape,fill,workers) ==
    Tensors.Banded{input,layout,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidArgument{shape}}},workers} : Tensors.Tensor<F32>}:
  {==}

def rejected_machine(~custom: F32 -> F32,+activation: Activations.Activation,geometry: TensorConvolution.Geometry,+layout: Bands.Geometry,input: Bands.Bands(~F32),weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {TensorConvolution.band_machine(~custom,activation,False{},geometry,layout,input,weights,bias,shape,fill,workers) ==
    Tensors.Banded{input,layout,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidShape{shape}}},workers} : Tensors.Tensor<F32>}:
  {==}
