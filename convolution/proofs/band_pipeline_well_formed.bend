# The band pipeline output is ready, covered and well formed.
import Base
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_output_metadata.bend as OutputMetadata
import ../../lib/kernel_band_pipeline.bend as BandPipeline
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_halo.bend as BandHalo
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_band_window.bend as BandWindow
import ./band_leaf_values.bend as LeafValues
import ../../lib/kernel_gemm_buffer.bend as GemmBuffer
import ../../lib/proofs/storage_clone_refinement.bend as Cloned
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/storage_clone.bend as Clone
import ../../lib/proofs/tensor_band_plan.bend as BandPlan

# Actual halo windows supply the output plan. The producer's source coverage
# and metadata establish complete output coverage without a nonempty oracle.
def frame_window(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Content<F32>,+above: Buffer.Content<F32>,+below: Buffer.Content<F32>,
  input: PipelineExecution.FrameInput(~Logical,~original,frame,first,rows,core,above,below,window)) ->
  {PipelineExecution.output_window(window) == Bands.Rows{OutputMetadata.native_first(~Logical,frame,first),OutputMetadata.native_rows(~Logical,frame,first,rows)} : Bands.Plan}:
  PipelineExecution.Frame{+logical,+activation,+dimensions,+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  Shape.Both{+window_equal,active} = input
  +frame = {PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  %Equal.sym(BandWindow.RowWindow,window,BandWindow.window(LeafValues.geometry(images,dimensions),first,rows),window_equal) :
    {PipelineExecution.output_window(_) == Bands.Rows{OutputMetadata.native_first(~Logical,frame,first),OutputMetadata.native_rows(~Logical,frame,first,rows)} : Bands.Plan}
  OutputMetadata.window_layout_window(~Logical,frame,first,rows)

def leaf_window(~Logical: Data,~original: Logical -> U32 -> F32,+configuration: PipelineExecution.Settings<PipelineExecution.Frame<Logical>>,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Content<F32>,+above: Buffer.Content<F32>,+below: Buffer.Content<F32>,
  input: PipelineExecution.premise(~Logical,~original,configuration,first,rows,core,above,below,window)) ->
  {PipelineExecution.output_window(window) == Bands.Rows{OutputMetadata.native_first(~Logical,PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,configuration),first),OutputMetadata.native_rows(~Logical,PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,configuration),first,rows)} : Bands.Plan}:
  Shape.Both{consistent,input} = input
  frame_window(~Logical,~original,PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,configuration),first,rows,window,core,above,below,input)

def compatible(~Logical: Data,~original: Logical -> U32 -> F32,+configuration: PipelineExecution.Settings<PipelineExecution.Frame<Logical>>,plan: Bands.Plan,cores: Model.Model<F32>,halos: BandHalo.Owners<F32>,
  inputs: PipelineExecution.Inputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),configuration,plan,cores,halos)) -> PipelineExecution.Compatible(plan,cores,halos):
  match plan cores:
    case Bands.Rows{first,rows} Model.Leaf{core,cf,cr}:
      match halos:
        case BandHalo.OwnedLeaf{above,below,window,hf,hr}: Unit{}
        case BandHalo.OwnedFork{left,right}: match inputs:
    case Bands.Split{+left_plan,+right_plan} Model.Fork{+left,+right}:
      match halos:
        case BandHalo.OwnedFork{left_halo,right_halo}:
          Shape.Both{left_inputs,right_inputs} = inputs
          Shape.Both{compatible(~Logical,~original,configuration,left_plan,left,left_halo,left_inputs),compatible(~Logical,~original,configuration,right_plan,right,right_halo,right_inputs)}
        case BandHalo.OwnedLeaf{above,below,window,hf,hr}: match inputs:
    case Bands.Rows{first,rows} Model.Fork{left,right}: match inputs:
    case Bands.Split{left,right} Model.Leaf{core,first,rows}: match inputs:

def output_layout(~Logical: Data,~original: Logical -> U32 -> F32,+configuration: PipelineExecution.Settings<PipelineExecution.Frame<Logical>>,+plan: Bands.Plan,+cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>,
  inputs: PipelineExecution.Inputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),configuration,plan,cores,halos)) ->
  {PipelineExecution.output_layout(halos) == OutputMetadata.native_plan(~Logical,PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,configuration),plan) : Bands.Plan}:
  match plan cores:
    case Bands.Rows{+first,+rows} Model.Leaf{+core,cf,cr}:
      match halos:
        case BandHalo.OwnedLeaf{+above,+below,+window,hf,hr}: leaf_window(~Logical,~original,configuration,first,rows,window,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),inputs)
        case BandHalo.OwnedFork{left,right}: match inputs:
    case Bands.Split{+left_plan,+right_plan} Model.Fork{+left,+right}:
      match halos:
        case BandHalo.OwnedFork{+left_halo,+right_halo}:
          Shape.Both{left_inputs,right_inputs} = inputs
          +frame = PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,configuration)
          %output_layout(~Logical,~original,configuration,left_plan,left,left_halo,left_inputs) :
            {Bands.Split{PipelineExecution.output_layout(left_halo),PipelineExecution.output_layout(right_halo)} == Bands.Split{_,OutputMetadata.native_plan(~Logical,frame,right_plan)} : Bands.Plan}
          %output_layout(~Logical,~original,configuration,right_plan,right,right_halo,right_inputs) :
            {Bands.Split{PipelineExecution.output_layout(left_halo),PipelineExecution.output_layout(right_halo)} == Bands.Split{PipelineExecution.output_layout(left_halo),_} : Bands.Plan}
          {==}
        case BandHalo.OwnedLeaf{above,below,window,hf,hr}: match inputs:
    case Bands.Rows{first,rows} Model.Fork{left,right}: match inputs:
    case Bands.Split{left,right} Model.Leaf{core,first,rows}: match inputs:

def run_coverage(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,+halos: BandHalo.Owners<F32>,+mode: Bool,
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  +shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  +inputs: PipelineExecution.Inputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),PipelineExecution.settings(~Logical,~original,frame),Model.layout(~F32,source),source,halos)) ->
  BandPlan.Coverage(PipelineExecution.layout(BandPipeline.run(~custom,Model.layout(~F32,source),mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.pack(~F32,source),BandHalo.pack_owners(~F32,halos),weights,bias)),
    0n,Nat.mul(U32.to_nat(OutputMetadata.window_guards_images(~Logical,frame)),U32.to_nat(OutputMetadata.output_height(~Logical,frame)))):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  +configuration = PipelineExecution.settings(~Logical,~original,frame)
  +plan = Model.layout(~F32,source)
  %Equal.sym(Bands.Plan,PipelineExecution.layout(BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),Model.pack(~F32,source),BandHalo.pack_owners(~F32,halos),weights,bias)),PipelineExecution.output_layout(halos),
    PipelineExecution.run_layout(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),source,halos,weights,bias,compatible(~Logical,~original,configuration,plan,source,halos,inputs))) :
    BandPlan.Coverage(_,0n,Nat.mul(U32.to_nat(images),U32.to_nat(U32.from_nat(LeafValues.output_height(KernelShape.Dimensions{ci,co,h,w,k,s,p})))))
  %Equal.sym(Bands.Plan,PipelineExecution.output_layout(halos),OutputMetadata.native_plan(~Logical,frame,plan),output_layout(~Logical,~original,configuration,plan,source,halos,inputs)) :
    BandPlan.Coverage(_,0n,Nat.mul(U32.to_nat(images),U32.to_nat(U32.from_nat(LeafValues.output_height(KernelShape.Dimensions{ci,co,h,w,k,s,p})))))
  OutputMetadata.coverage(~Logical,~original,frame,source,shape)

# Readiness follows the real plan and its existing convolution leaf contract.
# Clone certificates and native owners are observed; no caller leaf oracle.
def WindowReady(~custom: F32 -> F32,~Logical: Data,frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,window: BandWindow.RowWindow,
  core: Buffer.Model<F32>,above: Buffer.Model<F32>,below: Buffer.Model<F32>,weights: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<F32>) -> Data:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.windowed(~custom,activation,LeafValues.geometry(images,KernelShape.Dimensions{ci,co,h,w,k,s,p}),U32.from_nat(co),U32.from_nat(Nat.mul(ci,Nat.mul(k,k))),first,rows,window,
    Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))))

def input_ready(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  input: PipelineExecution.FrameInput(~Logical,~original,frame,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),window)) ->
  WindowReady(~custom,~Logical,frame,first,rows,window,core,above,below,weights,bias):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  Shape.Both{+window_equal,active} = input
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +frame = {PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  %Equal.sym(BandWindow.RowWindow,window,BandWindow.window(LeafValues.geometry(images,dimensions),first,rows),window_equal) :
    WindowReady(~custom,~Logical,frame,first,rows,_,core,above,below,weights,bias)
  OutputMetadata.ready(~custom,~Logical,~original,frame,first,rows,core,above,below,weights,bias)

def frame_ready(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  input: PipelineExecution.FrameInput(~Logical,~original,frame,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),window)) ->
  Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.leaf(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),first,rows,Buffer.pack(~F32,core),
    BandWindow.Halo{Buffer.pack(~F32,above),Buffer.pack(~F32,below),window},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)))):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  input_ready(~custom,~Logical,~original,PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible},first,rows,window,core,above,below,weights,bias,input)

def leaf_ready(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+configuration: PipelineExecution.Settings<PipelineExecution.Frame<Logical>>,+first: U32,+rows: U32,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+window: BandWindow.RowWindow,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  input: PipelineExecution.premise(~Logical,~original,configuration,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),window)) ->
  Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.leaf(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),first,rows,Buffer.pack(~F32,core),
    BandWindow.Halo{Buffer.pack(~F32,above),Buffer.pack(~F32,below),window},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)))):
  Shape.Both{+configuration_equal,input} = input
  +frame = PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,configuration)
  %Equal.sym(PipelineExecution.Settings<PipelineExecution.Frame<Logical>>,configuration,PipelineExecution.settings(~Logical,~original,frame),configuration_equal) :
    Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.leaf(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,_),first,rows,Buffer.pack(~F32,core),
      BandWindow.Halo{Buffer.pack(~F32,above),Buffer.pack(~F32,below),window},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))))
  frame_ready(~custom,~Logical,~original,frame,first,rows,window,core,above,below,weights,bias,input)

def BranchReady(~custom: F32 -> F32,+plan: Bands.Plan,+context: BandPipeline.Context,+cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>) -> Type:
  @+weights: Buffer.Model<Tile.Tile8> -> @+bias: Buffer.Model<F32> ->
  Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.run(~custom,plan,BandPipeline.parallel_plan(context,plan),context,Model.pack(~F32,cores),BandHalo.pack_owners(~F32,halos),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))))

def WeightReady(~custom: F32 -> F32,+left_plan: Bands.Plan,+right_plan: Bands.Plan,+context: BandPipeline.Context,+left: Model.Model<F32>,+right: Model.Model<F32>,
  +left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,+bias: Buffer.Model<F32>,copies: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Type:
  Model.ContentReady(~F32,Model.contents(~F32,PipelineExecution.fork_run(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,copies,Clone.clone(~F32,Buffer.pack(~F32,bias)))))

def BiasReady(~custom: F32 -> F32,+left_plan: Bands.Plan,+right_plan: Bands.Plan,+context: BandPipeline.Context,+left: Model.Model<F32>,+right: Model.Model<F32>,
  +left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,+left_weights: Buffer.Model<Tile.Tile8>,+right_weights: Buffer.Model<Tile.Tile8>,copies: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Type:
  Model.ContentReady(~F32,Model.contents(~F32,PipelineExecution.fork_run(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,(Buffer.pack(~Tile.Tile8,left_weights),Buffer.pack(~Tile.Tile8,right_weights)),copies)))

def split_ready(~custom: F32 -> F32,+left_plan: Bands.Plan,+right_plan: Bands.Plan,+context: BandPipeline.Context,+left: Model.Model<F32>,+right: Model.Model<F32>,
  +left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  left_rule: BranchReady(~custom,left_plan,context,left,left_halo),right_rule: BranchReady(~custom,right_plan,context,right,right_halo)) ->
  WeightReady(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,bias,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights))):
  Cloned.lift_pair(~Tile.Tile8,~WeightReady(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,bias),
    Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),Cloned.observe_pair(~Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights))),
    left_weights => right_weights => copies =>
      +left_weights = left_weights
      +right_weights = right_weights
      Cloned.lift_pair(~F32,~BiasReady(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,left_weights,right_weights),
        Clone.clone(~F32,Buffer.pack(~F32,bias)),Cloned.observe_pair(~F32,Clone.clone(~F32,Buffer.pack(~F32,bias))),
        left_bias => right_bias => copies => Shape.Both{left_rule(left_weights,left_bias),right_rule(right_weights,right_bias)}))

def run_ready(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+configuration: PipelineExecution.Settings<PipelineExecution.Frame<Logical>>,+plan: Bands.Plan,+mode: Bool,
  +cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  inputs: PipelineExecution.Inputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),configuration,plan,cores,halos)) ->
  Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),Model.pack(~F32,cores),BandHalo.pack_owners(~F32,halos),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)))):
  match plan cores:
    case Bands.Rows{+first,+rows} Model.Leaf{+core,cf,cr}:
      match halos:
        case BandHalo.OwnedLeaf{+above,+below,+window,hf,hr}: leaf_ready(~custom,~Logical,~original,configuration,first,rows,core,above,below,window,weights,bias,inputs)
        case BandHalo.OwnedFork{left,right}: match inputs:
    case Bands.Split{+left_plan,+right_plan} Model.Fork{+left,+right}:
      match halos:
        case BandHalo.OwnedFork{+left_halo,+right_halo}:
          Shape.Both{+left_inputs,+right_inputs} = inputs
          +context = PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration)
          %Equal.sym(Bands.Bands(~F32),BandPipeline.run(~custom,Bands.Split{left_plan,right_plan},mode,context,Model.pack(~F32,Model.Fork{left,right}),BandHalo.pack_owners(~F32,BandHalo.OwnedFork{left_halo,right_halo}),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)),
            PipelineExecution.fork_run(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),Clone.clone(~F32,Buffer.pack(~F32,bias))),
            PipelineExecution.values_fork_execution(~custom,left_plan,right_plan,mode,context,left,right,left_halo,right_halo,weights,bias)) : Model.ContentReady(~F32,Model.contents(~F32,_))
          split_ready(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,weights,bias,
            left_weights => left_bias => run_ready(~custom,~Logical,~original,configuration,left_plan,BandPipeline.parallel_plan(context,left_plan),left,left_halo,left_weights,left_bias,left_inputs),
            right_weights => right_bias => run_ready(~custom,~Logical,~original,configuration,right_plan,BandPipeline.parallel_plan(context,right_plan),right,right_halo,right_weights,right_bias,right_inputs))
        case BandHalo.OwnedLeaf{above,below,window,hf,hr}: match inputs:
    case Bands.Rows{first,rows} Model.Fork{left,right}: match inputs:
    case Bands.Split{left,right} Model.Leaf{core,first,rows}: match inputs:

def run_observed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+configuration: PipelineExecution.Settings<PipelineExecution.Frame<Logical>>,+plan: Bands.Plan,+mode: Bool,
  -cores: Bands.Bands(~F32),-halos: BandWindow.Halos(~F32),-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>,
  core_observation: Model.Observation<F32,cores>,halo_observation: BandHalo.OwnerObservation<F32,halos>,
  weight_observation: Buffer.Observation<Tile.Tile8,weights>,bias_observation: Buffer.Observation<F32,bias>,
  inputs: PipelineExecution.NativeInputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),configuration,plan,cores,halos)) ->
  Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),cores,halos,weights,bias))):
  Model.Observation{+core_model,+cores_equal} = core_observation
  BandHalo.OwnerObservation{+halo_model,+halos_equal} = halo_observation
  Buffer.Observation{+weight_model,+weights_equal} = weight_observation
  Buffer.Observation{+bias_model,+bias_equal} = bias_observation
  %cores_equal : Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),_,halos,weights,bias)))
  %halos_equal : Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),Model.pack(~F32,core_model),_,weights,bias)))
  %weights_equal : Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),Model.pack(~F32,core_model),BandHalo.pack_owners(~F32,halo_model),_,bias)))
  %bias_equal : Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),Model.pack(~F32,core_model),BandHalo.pack_owners(~F32,halo_model),Buffer.pack(~Tile.Tile8,weight_model),_)))
  run_ready(~custom,~Logical,~original,configuration,plan,mode,core_model,halo_model,weight_model,bias_model,
    PipelineExecution.observed_inputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),configuration,plan,core_model,halo_model,cores,halos,cores_equal,halos_equal,inputs))

def ready_run_native(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+configuration: PipelineExecution.Settings<PipelineExecution.Frame<Logical>>,+plan: Bands.Plan,+mode: Bool,
  cores: Bands.Bands(~F32),halos: BandWindow.Halos(~F32),weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  inputs: PipelineExecution.NativeInputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),configuration,plan,cores,halos)) ->
  Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),cores,halos,weights,bias))):
  run_observed(~custom,~Logical,~original,configuration,plan,mode,cores,halos,weights,bias,
    Model.observe(~F32,cores),BandHalo.observe_owners(~F32,halos),Buffer.observe(~Tile.Tile8,weights),Buffer.observe(~F32,bias),inputs)

def ready_prepared(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+plan: Bands.Plan,
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,owners: Bands.Bands(~F32) & BandWindow.Halos(~F32),
  inputs: PipelineExecution.PreparedInputs(~Logical,~original,frame,plan,BandHalo.refinement_contents(~F32,owners))) ->
  Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.prepared(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),plan,weights,bias,owners))):
  (cores,halos) = owners
  +configuration = PipelineExecution.settings(~Logical,~original,frame)
  ready_run_native(~custom,~Logical,~original,configuration,plan,BandPipeline.parallel_plan(PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),plan),cores,halos,weights,bias,inputs)

def ready_planned(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))) ->
  Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.planned(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),GemmBuffer.reorder(PipelineExecution.channels(~Logical,frame),PipelineExecution.inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),(Model.pack(~F32,source),Model.layout(~F32,source))))):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  ready_prepared(~custom,~Logical,~original,frame,Model.layout(~F32,source),GemmBuffer.reorder(PipelineExecution.channels(~Logical,frame),PipelineExecution.inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),
    BandWindow.prepare(~F32,PipelineExecution.window_geometry(~Logical,frame),0.0,Model.layout(~F32,source),Model.pack(~F32,source)),PipelineExecution.inputs_prepared(~Logical,~original,frame,source,shape,values))

def ready_execute(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))) ->
  Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.execute(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.pack(~F32,source),GemmBuffer.reorder(PipelineExecution.channels(~Logical,frame),PipelineExecution.inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias)))):
  %Equal.sym(Bands.Bands(~F32) & Bands.Plan,Bands.plan(~F32,Model.pack(~F32,source)),(Model.pack(~F32,source),Model.layout(~F32,source)),Model.plan(~F32,source)) :
    Model.ContentReady(~F32,Model.contents(~F32,BandPipeline.planned(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),GemmBuffer.reorder(PipelineExecution.channels(~Logical,frame),PipelineExecution.inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),_)))
  ready_planned(~custom,~Logical,~original,frame,source,weights,bias,shape,values)

# Metadata follows the real plan, native clone owners and convolution leaf.
# Every row/length premise is produced from source metadata and Frame guards.
def WindowMetadata(~custom: F32 -> F32,~Logical: Data,frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,window: BandWindow.RowWindow,
  core: Buffer.Model<F32>,above: Buffer.Model<F32>,below: Buffer.Model<F32>,weights: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<F32>) -> Data:
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  BandShape.Contents(~F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.windowed(~custom,activation,LeafValues.geometry(images,KernelShape.Dimensions{ci,co,h,w,k,s,p}),U32.from_nat(co),U32.from_nat(Nat.mul(ci,Nat.mul(k,k))),first,rows,window,
    Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))))

def input_metadata(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  source: BandShape.Leaf(PipelineExecution.input_geometry(~Logical,frame),first,rows,Buffer.length(~F32,core)),
  input: PipelineExecution.FrameInput(~Logical,~original,frame,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),window)) ->
  WindowMetadata(~custom,~Logical,frame,first,rows,window,core,above,below,weights,bias):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  Shape.Both{+window_equal,active} = input
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +frame = {PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  %Equal.sym(BandWindow.RowWindow,window,BandWindow.window(LeafValues.geometry(images,dimensions),first,rows),window_equal) :
    WindowMetadata(~custom,~Logical,frame,first,rows,_,core,above,below,weights,bias)
  OutputMetadata.metadata_leaf(~custom,~Logical,~original,frame,first,rows,core,above,below,weights,bias,Buffer.length(~F32,core),source)

def leaf_metadata(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  source: BandShape.Leaf(PipelineExecution.input_geometry(~Logical,frame),first,rows,Buffer.length(~F32,core)),
  input: PipelineExecution.premise(~Logical,~original,PipelineExecution.settings(~Logical,~original,frame),first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),window)) ->
  BandShape.Contents(~F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.leaf(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),first,rows,Buffer.pack(~F32,core),
    BandWindow.Halo{Buffer.pack(~F32,above),Buffer.pack(~F32,below),window},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)))):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  Shape.Both{consistent,input} = input
  input_metadata(~custom,~Logical,~original,PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible},first,rows,window,core,above,below,weights,bias,source,input)

def BranchMetadata(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>) -> Type:
  @+weights: Buffer.Model<Tile.Tile8> -> @+bias: Buffer.Model<F32> ->
  BandShape.Contents(~F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.run(~custom,Model.layout(~F32,cores),BandPipeline.parallel_plan(PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.layout(~F32,cores)),PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.pack(~F32,cores),BandHalo.pack_owners(~F32,halos),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))))

def WeightMetadata(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+left: Model.Model<F32>,+right: Model.Model<F32>,
  +left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,+bias: Buffer.Model<F32>,copies: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Type:
  BandShape.Contents(~F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,PipelineExecution.fork_run(~custom,Model.layout(~F32,left),Model.layout(~F32,right),PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),left,right,left_halo,right_halo,copies,Clone.clone(~F32,Buffer.pack(~F32,bias)))))

def BiasMetadata(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+left: Model.Model<F32>,+right: Model.Model<F32>,
  +left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,+left_weights: Buffer.Model<Tile.Tile8>,+right_weights: Buffer.Model<Tile.Tile8>,copies: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Type:
  BandShape.Contents(~F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,PipelineExecution.fork_run(~custom,Model.layout(~F32,left),Model.layout(~F32,right),PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),left,right,left_halo,right_halo,(Buffer.pack(~Tile.Tile8,left_weights),Buffer.pack(~Tile.Tile8,right_weights)),copies)))

def split_metadata(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+left: Model.Model<F32>,+right: Model.Model<F32>,
  +left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  left_rule: BranchMetadata(~custom,~Logical,~original,frame,left,left_halo),right_rule: BranchMetadata(~custom,~Logical,~original,frame,right,right_halo)) ->
  WeightMetadata(~custom,~Logical,~original,frame,left,right,left_halo,right_halo,bias,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights))):
  Cloned.lift_pair(~Tile.Tile8,~WeightMetadata(~custom,~Logical,~original,frame,left,right,left_halo,right_halo,bias),
    Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),Cloned.observe_pair(~Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights))),
    left_weights => right_weights =>
      copies =>
        +left_weights = left_weights
        +right_weights = right_weights
        Cloned.lift_pair(~F32,~BiasMetadata(~custom,~Logical,~original,frame,left,right,left_halo,right_halo,left_weights,right_weights),
          Clone.clone(~F32,Buffer.pack(~F32,bias)),Cloned.observe_pair(~F32,Clone.clone(~F32,Buffer.pack(~F32,bias))),
          left_bias => right_bias => copies => Shape.Both{left_rule(left_weights,left_bias),right_rule(right_weights,right_bias)}))

def run_metadata(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>,+mode: Bool,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  metadata: BandShape.Contents(~F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,cores)),
  inputs: PipelineExecution.Inputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),PipelineExecution.settings(~Logical,~original,frame),Model.layout(~F32,cores),cores,halos)) ->
  BandShape.Contents(~F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.run(~custom,Model.layout(~F32,cores),mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.pack(~F32,cores),BandHalo.pack_owners(~F32,halos),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)))):
  match cores:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      match halos:
        case BandHalo.OwnedLeaf{+above,+below,+window,hf,hr}:
          leaf_metadata(~custom,~Logical,~original,frame,first,rows,window,Buffer.Model{tree,length,certificate},above,below,weights,bias,metadata,inputs)
        case BandHalo.OwnedFork{left,right}: match inputs:
    case Model.Fork{+left,+right}:
      match halos:
        case BandHalo.OwnedFork{+left_halo,+right_halo}:
          Shape.Both{+left_metadata,+right_metadata} = metadata
          Shape.Both{+left_inputs,+right_inputs} = inputs
          +context = PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame))
          +left_plan = Model.layout(~F32,left)
          +right_plan = Model.layout(~F32,right)
          %Equal.sym(Bands.Bands(~F32),BandPipeline.run(~custom,Bands.Split{left_plan,right_plan},mode,context,Model.pack(~F32,Model.Fork{left,right}),BandHalo.pack_owners(~F32,BandHalo.OwnedFork{left_halo,right_halo}),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)),
            PipelineExecution.fork_run(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),Clone.clone(~F32,Buffer.pack(~F32,bias))),PipelineExecution.values_fork_execution(~custom,left_plan,right_plan,mode,context,left,right,left_halo,right_halo,weights,bias)) : BandShape.Contents(~F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,_))
          split_metadata(~custom,~Logical,~original,frame,left,right,left_halo,right_halo,weights,bias,
            left_weights => left_bias => run_metadata(~custom,~Logical,~original,frame,left,left_halo,BandPipeline.parallel_plan(context,left_plan),left_weights,left_bias,left_metadata,left_inputs),
            right_weights => right_bias => run_metadata(~custom,~Logical,~original,frame,right,right_halo,BandPipeline.parallel_plan(context,right_plan),right_weights,right_bias,right_metadata,right_inputs))
        case BandHalo.OwnedLeaf{above,below,window,hf,hr}: match inputs:

def run_well_formed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,+halos: BandHalo.Owners<F32>,+mode: Bool,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  +shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  +inputs: PipelineExecution.Inputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),PipelineExecution.settings(~Logical,~original,frame),Model.layout(~F32,source),source,halos)) ->
  BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.run(~custom,Model.layout(~F32,source),mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.pack(~F32,source),BandHalo.pack_owners(~F32,halos),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)))>:
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  BandShape.WellFormed{+metadata,+ready,+coverage} = shape
  +frame = {PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  BandShape.WellFormed{
    run_metadata(~custom,~Logical,~original,frame,source,halos,mode,weights,bias,metadata,inputs),
    run_ready(~custom,~Logical,~original,PipelineExecution.settings(~Logical,~original,frame),Model.layout(~F32,source),mode,source,halos,weights,bias,inputs),
    run_coverage(~custom,~Logical,~original,frame,source,halos,mode,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),BandShape.WellFormed{metadata,ready,coverage},inputs)}

# Observe actual prepared owners once, retaining their source metadata.
# No output metadata, leaf rule or allocation bound is supplied externally.
def source_shape(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,-cores: Bands.Bands(~F32),
  +equal: {Model.pack(~F32,source) == cores : Bands.Bands(~F32)},
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,cores)>) ->
  BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>:
  %Model.packed_contents(~F32,source) : BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),_>
  %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,source),cores,equal) : BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,_)>
  shape

def source_plan(+source: Model.Model<F32>,-cores: Bands.Bands(~F32),+plan: Bands.Plan,
  +equal: {Model.pack(~F32,source) == cores : Bands.Bands(~F32)},
  plan_equal: {plan == Model.content_layout(~F32,Model.contents(~F32,cores)) : Bands.Plan}) -> {plan == Model.layout(~F32,source) : Bands.Plan}:
  Equal.trans(Bands.Plan,plan,Model.content_layout(~F32,Model.contents(~F32,cores)),Model.layout(~F32,source),plan_equal,
    Equal.trans(Bands.Plan,Model.content_layout(~F32,Model.contents(~F32,cores)),Model.content_layout(~F32,Model.content(~F32,source)),Model.layout(~F32,source),
      Equal.cong(Model.Content<F32>,Bands.Plan,Model.content_layout(~F32),Model.contents(~F32,cores),Model.content(~F32,source),
        Equal.trans(Model.Content<F32>,Model.contents(~F32,cores),Model.contents(~F32,Model.pack(~F32,source)),Model.content(~F32,source),
          Equal.cong(Bands.Bands(~F32),Model.Content<F32>,Model.contents(~F32),cores,Model.pack(~F32,source),Equal.sym(Bands.Bands(~F32),Model.pack(~F32,source),cores,equal)),Model.packed_contents(~F32,source))),
      Equal.sym(Bands.Plan,Model.layout(~F32,source),Model.content_layout(~F32,Model.content(~F32,source)),Model.layout_matches_content(~F32,source))))

def canonical(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+plan: Bands.Plan,+mode: Bool,
  +cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  +shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,cores)>,
  +plan_equal: {plan == Model.layout(~F32,cores) : Bands.Plan},
  inputs: PipelineExecution.Inputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),PipelineExecution.settings(~Logical,~original,frame),plan,cores,halos)) ->
  BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.pack(~F32,cores),BandHalo.pack_owners(~F32,halos),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)))>:
  %Equal.sym(Bands.Plan,plan,Model.layout(~F32,cores),plan_equal) :
    BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.run(~custom,_,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.pack(~F32,cores),BandHalo.pack_owners(~F32,halos),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)))>
  run_well_formed(~custom,~Logical,~original,frame,cores,halos,mode,weights,bias,shape,
    %plan_equal : PipelineExecution.Inputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),PipelineExecution.settings(~Logical,~original,frame),_,cores,halos)
    inputs)

def observed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+plan: Bands.Plan,+mode: Bool,
  -cores: Bands.Bands(~F32),-halos: BandWindow.Halos(~F32),-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>,
  core_observation: Model.Observation<F32,cores>,halo_observation: BandHalo.OwnerObservation<F32,halos>,
  weight_observation: Buffer.Observation<Tile.Tile8,weights>,bias_observation: Buffer.Observation<F32,bias>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,cores)>,
  plan_equal: {plan == Model.content_layout(~F32,Model.contents(~F32,cores)) : Bands.Plan},
  inputs: PipelineExecution.NativeInputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),PipelineExecution.settings(~Logical,~original,frame),plan,cores,halos)) ->
  BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),cores,halos,weights,bias))>:
  Model.Observation{+core_model,+cores_equal} = core_observation
  BandHalo.OwnerObservation{+halo_model,+halos_equal} = halo_observation
  Buffer.Observation{+weight_model,+weights_equal} = weight_observation
  Buffer.Observation{+bias_model,+bias_equal} = bias_observation
  %cores_equal : BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),_,halos,weights,bias))>
  %halos_equal : BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.pack(~F32,core_model),_,weights,bias))>
  %weights_equal : BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.pack(~F32,core_model),BandHalo.pack_owners(~F32,halo_model),_,bias))>
  %bias_equal : BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.pack(~F32,core_model),BandHalo.pack_owners(~F32,halo_model),Buffer.pack(~Tile.Tile8,weight_model),_))>
  canonical(~custom,~Logical,~original,frame,plan,mode,core_model,halo_model,weight_model,bias_model,source_shape(~Logical,frame,core_model,cores,cores_equal,shape),source_plan(core_model,cores,plan,cores_equal,plan_equal),
    PipelineExecution.observed_inputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),PipelineExecution.settings(~Logical,~original,frame),plan,core_model,halo_model,cores,halos,cores_equal,halos_equal,inputs))

def formed_run_native(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+plan: Bands.Plan,+mode: Bool,
  cores: Bands.Bands(~F32),halos: BandWindow.Halos(~F32),weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.contents(~F32,cores)>,
  plan_equal: {plan == Model.content_layout(~F32,Model.contents(~F32,cores)) : Bands.Plan},
  inputs: PipelineExecution.NativeInputs(~PipelineExecution.Frame<Logical>,~PipelineExecution.premise(~Logical,~original),PipelineExecution.settings(~Logical,~original,frame),plan,cores,halos)) ->
  BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),cores,halos,weights,bias))>:
  observed(~custom,~Logical,~original,frame,plan,mode,cores,halos,weights,bias,Model.observe(~F32,cores),BandHalo.observe_owners(~F32,halos),Buffer.observe(~Tile.Tile8,weights),Buffer.observe(~F32,bias),shape,plan_equal,inputs)

def PreparedShape(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+plan: Bands.Plan,content: BandHalo.Content<F32>) -> Data:
  BandHalo.Content{+cores,halos} = content
  Shape.Both<BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),cores>,{plan == Model.content_layout(~F32,cores) : Bands.Plan}>

def prepare_shape(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>) ->
  PreparedShape(~Logical,frame,Model.layout(~F32,source),BandHalo.refinement_contents(~F32,BandWindow.prepare(~F32,PipelineExecution.window_geometry(~Logical,frame),0.0,Model.layout(~F32,source),Model.pack(~F32,source)))):
  +geometry = PipelineExecution.window_geometry(~Logical,frame)
  +plan = Model.layout(~F32,source)
  %Equal.sym(BandHalo.Content<F32>,BandHalo.refinement_contents(~F32,BandWindow.prepare(~F32,geometry,0.0,plan,Model.pack(~F32,source))),
    BandHalo.Content{Model.contents(~F32,Model.pack(~F32,source)),BandHalo.model(~F32,geometry,0.0,plan,Model.contents(~F32,Model.pack(~F32,source)))},BandHalo.prepare(~F32,geometry,0.0,plan,Model.pack(~F32,source))) : PreparedShape(~Logical,frame,plan,_)
  %Equal.sym(Model.Content<F32>,Model.contents(~F32,Model.pack(~F32,source)),Model.content(~F32,source),Model.packed_contents(~F32,source)) :
    PreparedShape(~Logical,frame,plan,BandHalo.Content{_,BandHalo.model(~F32,geometry,0.0,plan,_)})
  Shape.Both{shape,Model.layout_matches_content(~F32,source)}

def formed_prepared(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+plan: Bands.Plan,
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,owners: Bands.Bands(~F32) & BandWindow.Halos(~F32),
  shape: PreparedShape(~Logical,frame,plan,BandHalo.refinement_contents(~F32,owners)),inputs: PipelineExecution.PreparedInputs(~Logical,~original,frame,plan,BandHalo.refinement_contents(~F32,owners))) ->
  BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.prepared(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),plan,weights,bias,owners))>:
  (cores,halos) = owners
  Shape.Both{shape,plan_equal} = shape
  +configuration = PipelineExecution.settings(~Logical,~original,frame)
  formed_run_native(~custom,~Logical,~original,frame,plan,BandPipeline.parallel_plan(PipelineExecution.context(~PipelineExecution.Frame<Logical>,configuration),plan),cores,halos,weights,bias,shape,plan_equal,inputs)

def formed_planned(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))) ->
  BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.planned(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),GemmBuffer.reorder(PipelineExecution.channels(~Logical,frame),PipelineExecution.inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),(Model.pack(~F32,source),Model.layout(~F32,source))))>:
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  formed_prepared(~custom,~Logical,~original,frame,Model.layout(~F32,source),GemmBuffer.reorder(PipelineExecution.channels(~Logical,frame),PipelineExecution.inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),
    BandWindow.prepare(~F32,PipelineExecution.window_geometry(~Logical,frame),0.0,Model.layout(~F32,source),Model.pack(~F32,source)),prepare_shape(~Logical,frame,source,shape),PipelineExecution.inputs_prepared(~Logical,~original,frame,source,shape,values))

def formed_execute(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))) ->
  BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.execute(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Model.pack(~F32,source),GemmBuffer.reorder(PipelineExecution.channels(~Logical,frame),PipelineExecution.inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias)))>:
  %Equal.sym(Bands.Bands(~F32) & Bands.Plan,Bands.plan(~F32,Model.pack(~F32,source)),(Model.pack(~F32,source),Model.layout(~F32,source)),Model.plan(~F32,source)) :
    BandShape.WellFormed<F32,OutputMetadata.bounds_geometry(~Logical,frame),Model.contents(~F32,BandPipeline.planned(~custom,PipelineExecution.context(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),GemmBuffer.reorder(PipelineExecution.channels(~Logical,frame),PipelineExecution.inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),_))>
  formed_planned(~custom,~Logical,~original,frame,source,weights,bias,shape,values)
