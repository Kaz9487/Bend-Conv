# Tensor-level band convolution: owners, alignment and execution.
import Base
import ./band_pipeline_execution.bend as PipelineExecution
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./band_pipeline_well_formed.bend as PipelineWellFormed
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/proofs/tensor_band_map.bend as BandMap
import ../../lib/tensor_band.bend as Bands
import ../../lib/kernel_band_planning.bend as BandPlanning
import ../../lib/machine_limits.bend as Machine
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/kernel_band_planning_refinement.bend as KernelBandPlanningRefinement
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/tensor_band_combine_leaf.bend as BandCombineLeaf
import ../../lib/proofs/tensor_band_inspection.bend as Inspection
import ../../lib/proofs/tensor_band_plan.bend as BandPlan
import ../../lib/storage_buffer.bend as Storage

# Connect the actual Tensor-valued band_output wrapper to the leaf schedule.
# Replanning, public guard construction, and output WellFormed are separate.
def TensorValues(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,frame: PipelineExecution.Frame<Logical>,tensor: Tensors.Tensor<F32>) -> Data:
  +configuration = PipelineExecution.settings(~Logical,~original,frame)
  BandMap.LogicalValues(~F32,~PipelineExecution.Frame<Logical>,~PipelineExecution.result_value(~custom,~Logical,~original),PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,configuration),tensor)

def execution_activation(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Activations.Activation:
  PipelineExecution.Frame{logical,execution_activation,shape,images,execution_parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  execution_activation

def execution_geometry(~Logical: Data,frame: PipelineExecution.Frame<Logical>,batched: Bool) -> TensorConvolution.Geometry:
  PipelineExecution.Frame{logical,execution_activation,shape,images,execution_parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  TensorConvolution.Geometry{images,batched,shape}

def execution_parallel(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Bool:
  PipelineExecution.Frame{logical,execution_activation,shape,images,execution_parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  execution_parallel

def band_output(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  TensorValues(~custom,~Logical,~original,frame,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers)):
  PipelineExecution.Frame{+logical,+execution_activation,KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding},+images,+execution_parallel,+limits,+capacity,+weight_list,+bias_list,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {PipelineExecution.Frame{logical,execution_activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,execution_parallel,limits,capacity,weight_list,bias_list,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  PipelineExecution.execute(~custom,~Logical,~original,frame,source,weights,bias,shape,values,weight_exact,bias_exact)

def band_output_well_formed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))) ->
  BandMap.WellFormed(~F32,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers)):
  PipelineExecution.Frame{+logical,+execution_activation,KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding},+images,+execution_parallel,+limits,+capacity,+weight_list,+bias_list,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {PipelineExecution.Frame{logical,execution_activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,execution_parallel,limits,capacity,weight_list,bias_list,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  PipelineWellFormed.formed_execute(~custom,~Logical,~original,frame,source,weights,bias,shape,values)

def band_output_contract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  +values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  Shape.Both<TensorValues(~custom,~Logical,~original,frame,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers)),
    BandMap.WellFormed(~F32,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers))>:
  Shape.Both{band_output(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,shape,values,weight_exact,bias_exact),
    band_output_well_formed(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,shape,values)}

# The Tensor-valued leaf schedule accepts actual native owners by observation.
# Source and parameter contents are the public representation contracts.
def OwnersContract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,-tensor: Tensors.Tensor<F32>) -> Data:
  Shape.Both<TensorValues(~custom,~Logical,~original,frame,tensor),BandMap.WellFormed(~F32,tensor)>

def source_values(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+model: Model.Model<F32>,-source: Bands.Bands(~F32),
  +equal: {Model.pack(~F32,model) == source : Bands.Bands(~F32)},
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,source))) ->
  BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,model)):
  %Model.packed_contents(~F32,model) : BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),_)
  %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,model),source,equal) : BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,_))
  values

def buffer_contents(~Element: Data,+model: Buffer.Model<Element>,-buffer: Storage.Buffer<Element>,equal: {Buffer.pack(~Element,model) == buffer : Storage.Buffer<Element>}) ->
  {Buffer.content(~Element,model) == Buffer.contents(~Element,buffer) : Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,model),Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.contents(~Element,buffer),
    Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)),
    Equal.cong(Storage.Buffer<Element>,Buffer.Content<Element>,Buffer.contents(~Element),Buffer.pack(~Element,model),buffer,equal))

def model_tree(+model: Buffer.Model<F32>) -> {Buffer.tree(~F32,model) == Buffer.content_tree(~F32,Buffer.content(~F32,model)) : Tree.Tree<F32>}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def packed_tree(+model: Buffer.Model<F32>) -> {Buffer.content_tree(~F32,Buffer.contents(~F32,Buffer.pack(~F32,model))) == Buffer.tree(~F32,model) : Tree.Tree<F32>}:
  Equal.trans(Tree.Tree<F32>,Buffer.content_tree(~F32,Buffer.contents(~F32,Buffer.pack(~F32,model))),Buffer.content_tree(~F32,Buffer.content(~F32,model)),Buffer.tree(~F32,model),
    Equal.cong(Buffer.Content<F32>,Tree.Tree<F32>,Buffer.content_tree(~F32),Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.content(~F32,model),Buffer.packed_contents(~F32,model)),
    Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,model),Buffer.content_tree(~F32,Buffer.content(~F32,model)),model_tree(model)))

def observed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  -source: Bands.Bands(~F32),-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>,
  source_observation: Model.Observation<F32,source>,weight_observation: Buffer.Observation<F32,weights>,bias_observation: Buffer.Observation<F32,bias>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,source)),
  weight_exact: {Buffer.content_tree(~F32,Buffer.contents(~F32,weights)) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.contents(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  OwnersContract(~custom,~Logical,~original,frame,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),source,weights,bias,workers)):
  Model.Observation{+source_model,+source_equal} = source_observation
  Buffer.Observation{+weight_model,+weights_equal} = weight_observation
  Buffer.Observation{+bias_model,+bias_equal} = bias_observation
  %source_equal : OwnersContract(~custom,~Logical,~original,frame,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),_,weights,bias,workers))
  %weights_equal : OwnersContract(~custom,~Logical,~original,frame,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),Model.pack(~F32,source_model),_,bias,workers))
  %bias_equal : OwnersContract(~custom,~Logical,~original,frame,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),Model.pack(~F32,source_model),Buffer.pack(~F32,weight_model),_,workers))
  band_output_contract(~custom,~Logical,~original,frame,batched,workers,source_model,weight_model,bias_model,
    PipelineWellFormed.source_shape(~Logical,frame,source_model,source,source_equal,shape),source_values(~Logical,~original,frame,source_model,source,source_equal,values),
    Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,weight_model),Buffer.content_tree(~F32,Buffer.contents(~F32,weights)),PipelineExecution.source_tree(~Logical,frame),
      Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,weight_model),Buffer.content_tree(~F32,Buffer.content(~F32,weight_model)),Buffer.content_tree(~F32,Buffer.contents(~F32,weights)),
        model_tree(weight_model),Equal.cong(Buffer.Content<F32>,Tree.Tree<F32>,Buffer.content_tree(~F32),Buffer.content(~F32,weight_model),Buffer.contents(~F32,weights),buffer_contents(~F32,weight_model,weights,weights_equal))),weight_exact),
    Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,bias_model),Buffer.contents(~F32,bias),PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),buffer_contents(~F32,bias_model,bias,bias_equal),bias_exact))

def owners(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  source: Bands.Bands(~F32),weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,source)),
  weight_exact: {Buffer.content_tree(~F32,Buffer.contents(~F32,weights)) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.contents(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  OwnersContract(~custom,~Logical,~original,frame,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),source,weights,bias,workers)):
  observed(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,Model.observe(~F32,source),Buffer.observe(~F32,weights),Buffer.observe(~F32,bias),shape,values,weight_exact,bias_exact)

# Aligning a selected layout preserves the source contract.
# The convolution result reuses the existing leaf schedule.
def alignment_parallel(selection: BandPlanning.LayoutPlan) -> Bool:
  BandPlanning.LayoutPlan{layout,alignment_parallel,time,copies} = selection
  alignment_parallel

def with_parallel(~Logical: Data,frame: PipelineExecution.Frame<Logical>,alignment_parallel: Bool) -> PipelineExecution.Frame<Logical>:
  PipelineExecution.Frame{logical,activation,dimensions,images,previous,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  PipelineExecution.Frame{logical,activation,dimensions,images,alignment_parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible}

def selected_frame(~Logical: Data,frame: PipelineExecution.Frame<Logical>,selection: BandPlanning.LayoutPlan) -> PipelineExecution.Frame<Logical>:
  with_parallel(~Logical,frame,alignment_parallel(selection))

def AlignmentContract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,selection: BandPlanning.LayoutPlan,-tensor: Tensors.Tensor<F32>) -> Data:
  OwnersContract(~custom,~Logical,~original,selected_frame(~Logical,frame,selection),tensor)

def selected(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+selection: BandPlanning.LayoutPlan,
  +shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>},
  facts: KernelBandPlanningRefinement.SelectedFacts(PipelineExecution.input_geometry(~Logical,frame),PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame),selection)) ->
  AlignmentContract(~custom,~Logical,~original,frame,selection,TensorConvolution.replanned(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),selection,Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers)):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,previous,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  BandPlanning.LayoutPlan{+layout,+alignment_parallel,+time,+copies} = selection
  Shape.Both{+target,+coverage} = facts
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +frame = {PipelineExecution.Frame{logical,activation,dimensions,images,alignment_parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  +geometry = PipelineExecution.input_geometry(~Logical,frame)
  owners(~custom,~Logical,~original,frame,batched,workers,Bands.align(~F32,geometry,0.0,layout,Model.pack(~F32,source)),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),
    BandCombineLeaf.aligned_well_formed(~F32,geometry,0.0,layout,source,capacity,Machine.half_word(limits),shape,target,coverage),
    BandCombineLeaf.aligned_values(~F32,~Logical,~original,logical,geometry,0.0,layout,source,limits,capacity,domain,shape,values,target),
    Equal.trans(Tree.Tree<F32>,Buffer.content_tree(~F32,Buffer.contents(~F32,Buffer.pack(~F32,weights))),Buffer.tree(~F32,weights),PipelineExecution.source_tree(~Logical,frame),
      packed_tree(weights),weight_exact),
    Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,bias)),Buffer.content(~F32,bias),PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Buffer.packed_contents(~F32,bias),bias_exact))
