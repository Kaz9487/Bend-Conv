# Inspection success is produced from the normal source and parameter owners,
# then discharged at the actual public band_inspected call.
import Base
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_inspection.bend as Inspection
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_tensor_execution.bend as TensorExecution
import ./band_public_frame.bend as PublicFrame

def matches(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: TensorConvolution.Geometry,+layout: Bands.Geometry,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +shape: List<&2,U32>,+fill: F32,+workers: U32,-source: Bands.Bands(~F32),+inspection: Bands.Inspection) ->
  {TensorConvolution.band_inspected(~custom,activation,geometry,layout,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers,(source,inspection)) ==
    TensorConvolution.band_supported(~custom,activation,PublicFrame.guard(geometry,layout,weights,bias,inspection),geometry,layout,source,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) : Tensors.Tensor<F32>}:
  TensorConvolution.Geometry{count,batched,KernelShape.Dimensions{ci,co,h,w,k,s,p}} = geometry
  Bands.Geometry{images,channels,height,width} = layout
  Buffer.Model{weight_tree,weight_length,Certificate.Certificate{weight_depth,weight_equation}} = weights
  Buffer.Model{bias_tree,bias_length,Certificate.Certificate{bias_depth,bias_equation}} = bias
  Bands.Inspection{valid,first,end,leaves} = inspection
  {==}

def inspected(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +shape: List<&2,U32>,+fill: F32,+workers: U32,
  source_shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  weight_length: {Buffer.length(~F32,weights) == PublicFrame.weight_length(~Logical,frame) : Nat},bias_length: {Buffer.length(~F32,bias) == PublicFrame.bias_length(~Logical,frame) : Nat},
  weight_ready: Buffer.Sized<F32,weights>,bias_ready: Buffer.Sized<F32,bias>) ->
  {TensorConvolution.band_inspected(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),
      Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers,(Model.pack(~F32,source),Inspection.model(~F32,PipelineExecution.input_geometry(~Logical,frame),source))) ==
    TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) : Tensors.Tensor<F32>}:
  Equal.trans(Tensors.Tensor<F32>,
    TensorConvolution.band_inspected(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers,(Model.pack(~F32,source),Inspection.model(~F32,PipelineExecution.input_geometry(~Logical,frame),source))),
    TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),PublicFrame.guard(TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),weights,bias,Inspection.model(~F32,PipelineExecution.input_geometry(~Logical,frame),source)),TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    matches(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),weights,bias,shape,fill,workers,Model.pack(~F32,source),Inspection.model(~F32,PipelineExecution.input_geometry(~Logical,frame),source)),
    Equal.cong(Bool,Tensors.Tensor<F32>,valid => TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),valid,TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
      PublicFrame.guard(TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),weights,bias,Inspection.model(~F32,PipelineExecution.input_geometry(~Logical,frame),source)),True{},
      PublicFrame.frame(~Logical,~original,frame,batched,source,weights,bias,source_shape,weight_length,bias_length,weight_ready,bias_ready)))

def checked(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +shape: List<&2,U32>,+fill: F32,+workers: U32,
  source_shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  weight_length: {Buffer.length(~F32,weights) == PublicFrame.weight_length(~Logical,frame) : Nat},bias_length: {Buffer.length(~F32,bias) == PublicFrame.bias_length(~Logical,frame) : Nat},
  weight_ready: Buffer.Sized<F32,weights>,bias_ready: Buffer.Sized<F32,bias>) ->
  {TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{TensorExecution.execution_geometry(~Logical,frame,batched)},PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) ==
    TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) : Tensors.Tensor<F32>}:
  %Equal.sym(Bands.Bands(~F32) & Bands.Inspection,Bands.inspect(~F32,PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source)),
    (Model.pack(~F32,source),Inspection.model(~F32,PipelineExecution.input_geometry(~Logical,frame),source)),Inspection.inspect(~F32,PipelineExecution.input_geometry(~Logical,frame),source)) :
    {TensorConvolution.band_inspected(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers,_) ==
      TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) : Tensors.Tensor<F32>}
  inspected(~custom,~Logical,~original,frame,batched,source,weights,bias,shape,fill,workers,source_shape,weight_length,bias_length,weight_ready,bias_ready)
