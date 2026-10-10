# Actual replan produces target Fits/Coverage before aligning the source.
# Numerical refinement and output WF then reuse the convolution leaf schedule.
import Base
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/tensor_band.bend as Bands
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/kernel_band_planning.bend as BandPlanning
import ../../lib/kernel_planning.bend as KernelPlanning
import ../../lib/kernel_cost.bend as Cost
import ../../lib/tensor_band_window.bend as BandWindow
import ../../lib/default_kernel_cost.bend as DefaultCost
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/kernel_band_planning_refinement.bend as KernelBandPlanningRefinement
import ../../lib/proofs/kernel_band_planning_observation.bend as Observation
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/tensor_band_plan.bend as BandPlan
import ../../lib/proofs/tensor_band_inspection.bend as Inspection
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_combine_leaf.bend as BandCombineLeaf
import ../../lib/proofs/tensor_band_dense.bend as BandDense
import ../../lib/tensor.bend as Tensors
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_tensor_execution.bend as TensorExecution

import ./band_public_frame.bend as PublicFrame

import ./band_selected_observation.bend as SelectedObservation

def window_input_geometry(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool) ->
  {BandWindow.input_geometry(TensorConvolution.context_window(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame)))) == PipelineExecution.input_geometry(~Logical,frame) : Bands.Geometry}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  {==}

def existing_layout_matches(~custom: F32 -> F32,~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>) ->
  {TensorConvolution.replanned(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),
      BandPlanning.replan(Model.layout(~F32,source),TensorConvolution.band_work(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame)),U32.shln(1,Buffer.depth(~F32,bias))),
        TensorConvolution.context_window(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame))),workers,DefaultCost.configuration()),Model.pack(~F32,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers) ==
    TensorConvolution.existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers,(Model.pack(~F32,source),Model.layout(~F32,source))) : Tensors.Tensor<F32>}:
  Buffer.Model{tree,length,Certificate.Certificate{depth,certificate}} = bias
  {==}

def existing_layout(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers,(Model.pack(~F32,source),Model.layout(~F32,source)))):
  BandShape.WellFormed{+metadata,+ready,+coverage} = shape
  +limits = PipelineExecution.convolution_limits(~Logical,frame)
  +capacity = PipelineExecution.convolution_capacity(~Logical,frame)
  +domain = PipelineExecution.convolution_domain(~Logical,frame)
  +bias_depth = Buffer.depth(~F32,bias)
  +shape = {BandShape.WellFormed{metadata,ready,coverage} : BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>}
  +context = TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame))
  +plan = Model.layout(~F32,source)
  %existing_layout_matches(~custom,~Logical,frame,batched,workers,source,weights,bias) :
    PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),_)
  SelectedObservation.consume(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,TensorConvolution.band_work(context,U32.shln(1,bias_depth)),TensorConvolution.context_window(context),DefaultCost.configuration(),
    shape,values,weight_exact,bias_exact,window_input_geometry(~Logical,frame,batched),
    Observation.observe(plan,TensorConvolution.band_work(context,U32.shln(1,bias_depth)),TensorConvolution.context_window(context),workers,DefaultCost.configuration(),limits,capacity,
      %Equal.sym(Bands.Geometry,BandWindow.input_geometry(TensorConvolution.context_window(context)),PipelineExecution.input_geometry(~Logical,frame),window_input_geometry(~Logical,frame,batched)) :
        BandGather.Domain<_,limits,capacity>
      domain,
      %Equal.sym(Bands.Geometry,BandWindow.input_geometry(TensorConvolution.context_window(context)),PipelineExecution.input_geometry(~Logical,frame),window_input_geometry(~Logical,frame,batched)) :
        KernelBandPlanningRefinement.LayoutFacts(_,limits,capacity,plan)
      Shape.Both{BandDense.target(~F32,PipelineExecution.input_geometry(~Logical,frame),limits,capacity,source,domain,metadata,Model.ready(~F32,source,ready)),
        %Equal.sym(Bands.Plan,Model.layout(~F32,source),Model.content_layout(~F32,Model.content(~F32,source)),Model.layout_matches_content(~F32,source)) : BandPlan.Coverage(_,0n,Inspection.total_rows(PipelineExecution.input_geometry(~Logical,frame)))
        coverage}))
