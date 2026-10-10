# Public machine acceptance is derived from the frame's checked support and
# batch envelope, using the actual Default limits, before existing replanning.
import Base
import ../specification.bend as ConvSpec
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_tensor_execution.bend as TensorExecution
import ./band_leaf_values.bend as LeafValues
import ./band_caller_guards.bend as Caller
import ./band_frame_configuration.bend as Configuration
import ./convolution_guard_facts.bend as Guards
import ./band_public_frame.bend as PublicFrame
import ./band_public_status.bend as PublicStatus
import ./band_inspected_caller.bend as Inspected

def predicate(geometry: TensorConvolution.Geometry,+limits: Machine.Limits) -> Bool:
  TensorConvolution.machine_support(geometry,limits)

def accepted(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool) ->
  {predicate(TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.convolution_limits(~Logical,frame)) == True{} : Bool}:
  PipelineExecution.Frame{+logical,activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,parallel,+limits,capacity,+weights,+bias,we,be,domain,+valid,supported,room,eligible} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +length_exact = Configuration.weights_length(Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n),
    Guards.valid(Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n),Caller.valid(~Logical,~original,logical,dimensions,weights,bias,0n,valid)))
  %length_exact : {KernelShape.supports_packed(limits,dimensions,Caller.input_length(dimensions),_) &&
    Nat.is_le(LeafValues.output_elements(images,dimensions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  %Equal.sym(Bool,KernelShape.supports_packed(limits,dimensions,Caller.input_length(dimensions),ConvSpec.length(weights)),True{},supported) :
    {_ && Nat.is_le(LeafValues.output_elements(images,dimensions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  room

def eligible(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,batched: Bool) -> {TensorConvolution.uses_band_windows(TensorExecution.execution_geometry(~Logical,frame,batched)) == True{} : Bool}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  eligible

def supported_matches(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: TensorConvolution.Geometry,+layout: Bands.Geometry,-source: Bands.Bands(~F32),
  +weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {TensorConvolution.band_supported(~custom,activation,True{},geometry,layout,source,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) ==
    TensorConvolution.band_machine(~custom,activation,predicate(geometry,Default.limits()),geometry,layout,source,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) : Tensors.Tensor<F32>}:
  {==}

def supported(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,-source: Bands.Bands(~F32),
  +weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  actual_limits: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits}) ->
  {TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),source,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) ==
    TensorConvolution.execute_existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),source,TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers) : Tensors.Tensor<F32>}:
  +geometry = TensorExecution.execution_geometry(~Logical,frame,batched)
  +machine = {
    %actual_limits : {predicate(geometry,_) == True{} : Bool}
    accepted(~Logical,~original,frame,batched) : {predicate(geometry,Default.limits()) == True{} : Bool}}
  %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),True{},geometry,PipelineExecution.input_geometry(~Logical,frame),source,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    TensorConvolution.band_machine(~custom,TensorExecution.execution_activation(~Logical,frame),predicate(geometry,Default.limits()),geometry,PipelineExecution.input_geometry(~Logical,frame),source,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    supported_matches(~custom,TensorExecution.execution_activation(~Logical,frame),geometry,PipelineExecution.input_geometry(~Logical,frame),source,weights,bias,shape,fill,workers)) :
    {_ == TensorConvolution.execute_existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),True{},geometry,PipelineExecution.input_geometry(~Logical,frame),source,TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers) : Tensors.Tensor<F32>}
  %Equal.sym(Bool,predicate(geometry,Default.limits()),True{},machine) :
    {TensorConvolution.band_machine(~custom,TensorExecution.execution_activation(~Logical,frame),_,geometry,PipelineExecution.input_geometry(~Logical,frame),source,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) ==
      TensorConvolution.execute_existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),True{},geometry,PipelineExecution.input_geometry(~Logical,frame),source,TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers) : Tensors.Tensor<F32>}
  Equal.cong(Bool,Tensors.Tensor<F32>,enabled => TensorConvolution.execute_existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),enabled,geometry,PipelineExecution.input_geometry(~Logical,frame),source,TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    TensorConvolution.uses_band_windows(geometry),True{},eligible(~Logical,frame,batched))

def checked(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +shape: List<&2,U32>,+fill: F32,+workers: U32,+actual_limits: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  +source_shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_length: {Buffer.length(~F32,weights) == PublicFrame.weight_length(~Logical,frame) : Nat},bias_length: {Buffer.length(~F32,bias) == PublicFrame.bias_length(~Logical,frame) : Nat},
  weight_ready: Buffer.Sized<F32,weights>,bias_ready: Buffer.Sized<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{TensorExecution.execution_geometry(~Logical,frame,batched)},PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers)):
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{TensorExecution.execution_geometry(~Logical,frame,batched)},PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    Inspected.checked(~custom,~Logical,~original,frame,batched,source,weights,bias,shape,fill,workers,source_shape,weight_length,bias_length,weight_ready,bias_ready)) :
    PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),_)
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    TensorConvolution.execute_existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    supported(~custom,~Logical,~original,frame,batched,Model.pack(~F32,source),weights,bias,shape,fill,workers,actual_limits)) :
    PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),_)
  PublicStatus.planned(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,source_shape,values,weight_exact,bias_exact)
