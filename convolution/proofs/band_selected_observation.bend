# Consume a checked planner observation before transporting to actual replan.
# The selected convolution reuses the existing leaf schedule.
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
import ../../lib/proofs/tensor_band_combine_leaf.bend as BandCombineLeaf
import ../../lib/proofs/tensor_band_dense.bend as BandDense
import ../../lib/tensor.bend as Tensors
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_tensor_execution.bend as TensorExecution

import ./band_public_frame.bend as PublicFrame

def consume(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+configuration: Cost.Configuration,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>},
  window_exact: {BandWindow.input_geometry(window) == PipelineExecution.input_geometry(~Logical,frame) : Bands.Geometry},
  observation: Observation.ReplanObservation<Model.layout(~F32,source),work,window,workers,configuration,PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame)>) ->
  PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.replanned(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),BandPlanning.replan(Model.layout(~F32,source),work,window,workers,configuration),Model.pack(~F32,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)):
  Observation.ReplanObservation{+selection,exact,facts} = observation
  +selected_facts = {
    %window_exact : KernelBandPlanningRefinement.SelectedFacts(_,PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame),selection)
    facts : KernelBandPlanningRefinement.SelectedFacts(PipelineExecution.input_geometry(~Logical,frame),PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame),selection)}
  %exact : PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.replanned(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),_,Model.pack(~F32,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers))
  PublicFrame.selected(~custom,~Logical,~original,frame,selection,
    TensorConvolution.replanned(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),selection,Model.pack(~F32,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    TensorExecution.selected(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,selection,shape,values,weight_exact,bias_exact,selected_facts))

