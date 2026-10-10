# Native dense public entry: initial layout, planning, execution and fallback.
import Base
import ./band_dense_native_contract.bend as DenseNativeContract
import ../../lib/math_activation.bend as Activations
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./band_initial_layout.bend as InitialLayout
import ./band_native_caller.bend as Native
import ./band_public_frame.bend as PublicFrame
import ../specification.bend as ConvSpec
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_dense_contract.bend as Dense
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/storage_buffer.bend as Storage
import ./band_parameter_owners.bend as Parameters
import ../../lib/kernel_band_planning.bend as BandPlanning
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_dense_single.bend as DenseSingle
import ./band_dense_sources.bend as DenseSources
import ./band_native_initial_layout.bend as Band
import ./band_normal_caller.bend as NormalCaller
import ../../lib/default_kernel_cost.bend as DefaultCost
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_concat_execution.bend as BandConcatExecution
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_count.bend as BandShapeCount
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/tensor_dense_observation.bend as DenseObservation
import ../../lib/proofs/tensor_count_refinement.bend as Count
import ../../lib/proofs/tensor_dense_normal.bend as DenseNormal
import ../../lib/proofs/tensor_normal_densification.bend as Normal
import ../../lib/proofs/tensor_normal_densification_values.bend as NormalDensificationValues
import ../../lib/proofs/tensor_ready_densification.bend as ReadyDensification
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_window.bend as Window

# The actual initial-layout plan is classified by its observed depth. Both
# branches produce the same finite output and Normal envelope; the dense
# branch retains the original worker-one execution and final worker wrapper.
def time(plan: BandPlanning.Plan) -> F32:
  BandPlanning.Plan{depth,parallel,time} = plan
  time

def zero_plan(+plan: BandPlanning.Plan,empty: {Band.depth(plan) == 0 : U32}) ->
  {plan == BandPlanning.Plan{0,Band.parallel(plan),time(plan)} : BandPlanning.Plan}:
  BandPlanning.Plan{+depth,+parallel,+time} = plan
  Equal.cong(U32,BandPlanning.Plan,selected => BandPlanning.Plan{selected,parallel,time},depth,0,empty)

def bias_tree(+model: Buffer.Model<F32>,+values: List<&2,F32>,+evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>,
  exact: {Buffer.content(~F32,model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),ConvSpec.length(values)} : Buffer.Content<F32>}) ->
  {Buffer.tree(~F32,model) == StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence) : Tree.Tree<F32>}:
  Buffer.Model{+tree,+length,+certificate} = model
  Equal.cong(Buffer.Content<F32>,Tree.Tree<F32>,Buffer.content_tree(~F32),Buffer.content(~F32,model),Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),ConvSpec.length(values)},exact)

def classified(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+plan: BandPlanning.Plan,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>},
  classification: Band.DepthKind<plan>) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.initial_layout(~custom,activation,plan,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  match classification:
    case Band.ZeroDepth{empty}:
      %Equal.sym(BandPlanning.Plan,plan,BandPlanning.Plan{0,Band.parallel(plan),time(plan)},zero_plan(plan,empty)) :
        InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
          TensorConvolution.initial_layout(~custom,activation,_,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers))
      DenseNativeContract.initial_zero(~custom,~Logical,~original,logical,activation,input,weights,bias,Band.parallel(plan),time(plan),source,weight_model,bias_model,fill,input_workers,workers,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_tree(bias_model,bias,bias_evidence,bias_exact),values,validated,supported)
    case Band.PositiveDepth{positive}:
      Band.nonzero(~custom,~Logical,~original,logical,activation,input,plan,weights,bias,weight_evidence,bias_evidence,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,validated,supported,eligible,values,weight_exact,bias_exact,positive)

def initial_layout(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+plan: BandPlanning.Plan,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.initial_layout(~custom,activation,plan,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  classified(~custom,~Logical,~original,logical,activation,input,plan,weights,bias,weight_evidence,bias_evidence,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,validated,supported,eligible,values,weight_exact,bias_exact,Band.classify(plan))

# Observe the real Default band planner at the native dense caller. Input
# geometry remains opaque while its actual bias certificate supplies depth;
# the selected plan is classified by the complete initial-layout producer.
def selected(+activation: Activations.Activation,+input: PublicFrame.NativeInput,bias: Buffer.Model<F32>,+workers: U32) -> BandPlanning.Plan:
  Buffer.Model{tree,length,Certificate.Certificate{+depth,equation}} = bias
  +context = TensorConvolution.band_context(PublicFrame.geometry(input),True{},activation)
  BandPlanning.select(TensorConvolution.band_work(context,U32.shln(1,depth)),TensorConvolution.context_window(context),workers,DefaultCost.configuration())

type PlanObservation<-activation: Activations.Activation,-input: PublicFrame.NativeInput,-bias: Buffer.Model<F32>,-workers: U32> is Data:
  PlanObservation{plan: BandPlanning.Plan,exact: {plan == selected(activation,input,bias,workers) : BandPlanning.Plan}}

def observe(+activation: Activations.Activation,+input: PublicFrame.NativeInput,+bias: Buffer.Model<F32>,+workers: U32) -> PlanObservation<activation,input,bias,workers>:
  PlanObservation{selected(activation,input,bias,workers),{==}}

def header(~custom: F32 -> F32,+activation: Activations.Activation,+input: PublicFrame.NativeInput,-source: Storage.Buffer<F32>,-weights: Storage.Buffer<F32>,+bias: Buffer.Model<F32>,+workers: U32) ->
  {TensorConvolution.execute_many(~custom,activation,PublicFrame.geometry(input),source,TensorConvolution.packed(PublicFrame.geometry(input),weights),Buffer.pack(~F32,bias),workers) ==
    TensorConvolution.initial_layout(~custom,activation,selected(activation,input,bias,workers),PublicFrame.geometry(input),source,TensorConvolution.packed(PublicFrame.geometry(input),weights),Buffer.pack(~F32,bias),workers) : Tensors.Tensor<F32>}:
  Buffer.Model{tree,length,Certificate.Certificate{depth,equation}} = bias
  {==}

def planning_observed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>},
  observation: PlanObservation<activation,input,bias_model,workers>) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute_many(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  PlanObservation{+plan,+exact} = observation
  %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.execute_many(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers),
    TensorConvolution.initial_layout(~custom,activation,selected(activation,input,bias_model,workers),PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers),header(~custom,activation,input,Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),bias_model,workers)) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),_)
  %exact : InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.initial_layout(~custom,activation,_,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers))
  initial_layout(~custom,~Logical,~original,logical,activation,input,plan,weights,bias,weight_evidence,bias_evidence,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,validated,supported,eligible,values,weight_exact,bias_exact)

def execute_many(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute_many(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  planning_observed(~custom,~Logical,~original,logical,activation,input,weights,bias,weight_evidence,bias_evidence,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,validated,supported,eligible,values,weight_exact,bias_exact,observe(activation,input,bias_model,workers))

# Observe the real worker-one predicate before the native execution branch.
# This capability describes the executor head only; it neither chooses a
# planner nor assumes a successful convolution result.
type WorkerObservation<-workers: U32> is Data:
  WorkerObservation{single: Bool,exact: {U32.is_eq(workers,1) == single : Bool}}

def observe_workers(+workers: U32) -> WorkerObservation<workers>:
  WorkerObservation{U32.is_eq(workers,1),{==}}

def other_header(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: TensorConvolution.Geometry,-source: Storage.Buffer<F32>,-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>,+workers: U32,
  not_one: {U32.is_eq(workers,1) == False{} : Bool}) ->
  {TensorConvolution.execute(~custom,activation,geometry,source,TensorConvolution.packed(geometry,weights),bias,workers) ==
    TensorConvolution.execute_layout(~custom,activation,TensorConvolution.uses_band_windows(geometry),geometry,source,TensorConvolution.packed(geometry,weights),bias,workers) : Tensors.Tensor<F32>}:
  %Equal.sym(Bool,U32.is_eq(workers,1),False{},not_one) :
    {TensorConvolution.execute_selected(~custom,activation,_,geometry,source,TensorConvolution.packed(geometry,weights),bias,workers) ==
      TensorConvolution.execute_layout(~custom,activation,TensorConvolution.uses_band_windows(geometry),geometry,source,TensorConvolution.packed(geometry,weights),bias,workers) : Tensors.Tensor<F32>}
  {==}

# Observe the actual worker and band-window branches. Dense execution and
# Default band planning produce one finite layout contract for the real
# executor; branch receipts are constructed from its own native predicates.
type LayoutObservation<-input: PublicFrame.NativeInput> is Data:
  LayoutObservation{enabled: Bool,exact: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == enabled : Bool}}

def observe_layout(+input: PublicFrame.NativeInput) -> LayoutObservation<input>:
  LayoutObservation{TensorConvolution.uses_band_windows(PublicFrame.geometry(input)),{==}}

def observed_layout(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>},
  observation: LayoutObservation<input>) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute_layout(~custom,activation,TensorConvolution.uses_band_windows(PublicFrame.geometry(input)),PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  match observation:
    case LayoutObservation{False{},+exact}:
      %Equal.sym(Bool,TensorConvolution.uses_band_windows(PublicFrame.geometry(input)),False{},exact) :
        InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
          TensorConvolution.execute_layout(~custom,activation,_,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers))
      DenseNativeContract.contract_layout(~custom,~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_tree(bias_model,bias,bias_evidence,bias_exact),values,validated,supported)
    case LayoutObservation{True{},+exact}:
      %Equal.sym(Bool,TensorConvolution.uses_band_windows(PublicFrame.geometry(input)),True{},exact) :
        InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
          TensorConvolution.execute_layout(~custom,activation,_,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers))
      execute_many(~custom,~Logical,~original,logical,activation,input,weights,bias,weight_evidence,bias_evidence,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,validated,supported,exact,values,weight_exact,bias_exact)

def execute_layout(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute_layout(~custom,activation,TensorConvolution.uses_band_windows(PublicFrame.geometry(input)),PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  observed_layout(~custom,~Logical,~original,logical,activation,input,weights,bias,weight_evidence,bias_evidence,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,validated,supported,values,weight_exact,bias_exact,observe_layout(input))

def classified_workers(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>},
  observation: WorkerObservation<workers>) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  match observation:
    case WorkerObservation{False{},not_one}:
      %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers),
        TensorConvolution.execute_layout(~custom,activation,TensorConvolution.uses_band_windows(PublicFrame.geometry(input)),PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers),other_header(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers,not_one)) :
        InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),_)
      execute_layout(~custom,~Logical,~original,logical,activation,input,weights,bias,weight_evidence,bias_evidence,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,validated,supported,values,weight_exact,bias_exact)
    case WorkerObservation{True{},single}:
      %Equal.sym(U32,workers,1,ShapeLaws.native_equal(workers,1,single)) :
        InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
          TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),_))
      DenseNativeContract.contract_layout(~custom,~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,fill,input_workers,1,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_tree(bias_model,bias,bias_evidence,bias_exact),values,validated,supported)

def execute(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  classified_workers(~custom,~Logical,~original,logical,activation,input,weights,bias,weight_evidence,bias_evidence,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,validated,supported,values,weight_exact,bias_exact,observe_workers(workers))

# Actual dense conv2d operands retain their native owners and decoded shape.
# Parameter Normal produces the observed finite sequences, and the checked
# and execution capabilities construct the common result contract.
def checked_contract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>},
  +weight_length: {Buffer.length(~F32,weight_model) == ConvSpec.length(weights) : Nat},+bias_length: {Buffer.length(~F32,bias_model) == ConvSpec.length(bias) : Nat},
  weight_ready: Buffer.Sized<F32,weight_model>,bias_ready: Buffer.Sized<F32,bias_model>) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),Native.input_extents(input),fill,workers)):
  %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),Native.input_extents(input),fill,workers),
    TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers),
    DenseNativeContract.checked(~custom,activation,input,weights,bias,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,validated,supported,weight_length,bias_length,weight_ready,bias_ready)) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),_)
  execute(~custom,~Logical,~original,logical,activation,input,weights,bias,weight_evidence,bias_evidence,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,validated,supported,values,weight_exact,bias_exact)

def checked_parameters(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +source: Buffer.Model<F32>,+weight_owner: Buffer.Model<F32>,+bias_owner: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +weight_parameter: Parameters.Parameter<Buffer.pack(~F32,weight_owner)>,+bias_parameter: Parameters.Parameter<Buffer.pack(~F32,bias_owner)>,
  validated: {PublicFrame.frame_valid(input,Parameters.parameter_values(Buffer.pack(~F32,weight_owner),weight_parameter),Parameters.parameter_values(Buffer.pack(~F32,bias_owner),bias_parameter)) == True{} : Bool}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,Parameters.parameter_values(Buffer.pack(~F32,weight_owner),weight_parameter),Parameters.parameter_values(Buffer.pack(~F32,bias_owner),bias_parameter)),
    TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weight_owner),Buffer.pack(~F32,bias_owner),Native.input_extents(input),fill,workers)):
  Parameters.Parameter{+weights,+weight_equation,+weight_ready,+weight_values,+weight_length,+weight_evidence,+weight_exact} = weight_parameter
  Parameters.Parameter{+bias,+bias_equation,+bias_ready,+bias_values,+bias_length,+bias_evidence,bias_exact} = bias_parameter
  %weight_equation :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weight_values,bias_values),
      TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),_,Buffer.pack(~F32,bias_owner),Native.input_extents(input),fill,workers))
  %bias_equation :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weight_values,bias_values),
      TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weights),_,Native.input_extents(input),fill,workers))
  checked_contract(~custom,~Logical,~original,logical,activation,input,weight_values,bias_values,weight_evidence,bias_evidence,source,weights,bias,fill,input_workers,workers,normal,matched,validated,supported,values,weight_exact,
    Parameters.parameter_content(Buffer.pack(~F32,bias_owner),bias_parameter),Equal.sym(Nat,ConvSpec.length(weight_values),Buffer.length(~F32,weights),weight_length),Equal.sym(Nat,ConvSpec.length(bias_values),Buffer.length(~F32,bias),bias_length),weight_ready,bias_ready)

def conv2d_header(~custom: F32 -> F32,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+weight_workers: U32,+bias_workers: U32,
  matched: {Native.input_matched(input) == True{} : Bool}) ->
  {TensorConvolution.conv2d(~custom,activation,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers},
      Tensors.Tensor{Buffer.pack(~F32,weights),Native.weight_extents(input),0.0,Tensors.Usable{},weight_workers},Tensors.Tensor{Buffer.pack(~F32,bias),Native.bias_extents(input),0.0,Tensors.Usable{},bias_workers},Native.stride(input),Native.padding(input)) ==
    TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers))) : Tensors.Tensor<F32>}:
  %Equal.sym(Maybe<&2,TensorConvolution.Geometry>,TensorConvolution.geometry(TensorConvolution.input_shape(Native.input_extents(input)),Native.weight_extents(input),Native.bias_extents(input),Native.stride(input),Native.padding(input)),Some{PublicFrame.geometry(input)},Native.public_geometry(input,matched)) :
    {TensorConvolution.checked(~custom,activation,_,Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers))) ==
      TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers))) : Tensors.Tensor<F32>}
  {==}

def conv2d(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+weight_workers: U32,+bias_workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +weight_normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,weights),Native.weight_extents(input),0.0,Tensors.Usable{},weight_workers}),
  +bias_normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,bias),Native.bias_extents(input),0.0,Tensors.Usable{},bias_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  validated: {PublicFrame.frame_valid(input,
    Parameters.parameter_values(Buffer.pack(~F32,weights),DenseNativeContract.parameter_normal(weights,Native.weight_extents(input),0.0,weight_workers,weight_normal)),
    Parameters.parameter_values(Buffer.pack(~F32,bias),DenseNativeContract.parameter_normal(bias,Native.bias_extents(input),0.0,bias_workers,bias_normal))) == True{} : Bool}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,
    Parameters.parameter_values(Buffer.pack(~F32,weights),DenseNativeContract.parameter_normal(weights,Native.weight_extents(input),0.0,weight_workers,weight_normal)),
    Parameters.parameter_values(Buffer.pack(~F32,bias),DenseNativeContract.parameter_normal(bias,Native.bias_extents(input),0.0,bias_workers,bias_normal))),
    TensorConvolution.conv2d(~custom,activation,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers},
      Tensors.Tensor{Buffer.pack(~F32,weights),Native.weight_extents(input),0.0,Tensors.Usable{},weight_workers},Tensors.Tensor{Buffer.pack(~F32,bias),Native.bias_extents(input),0.0,Tensors.Usable{},bias_workers},Native.stride(input),Native.padding(input))):
  %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.conv2d(~custom,activation,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers},
      Tensors.Tensor{Buffer.pack(~F32,weights),Native.weight_extents(input),0.0,Tensors.Usable{},weight_workers},Tensors.Tensor{Buffer.pack(~F32,bias),Native.bias_extents(input),0.0,Tensors.Usable{},bias_workers},Native.stride(input),Native.padding(input)),
    TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers))),conv2d_header(~custom,activation,input,source,weights,bias,fill,input_workers,weight_workers,bias_workers,matched)) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,
      Parameters.parameter_values(Buffer.pack(~F32,weights),DenseNativeContract.parameter_normal(weights,Native.weight_extents(input),0.0,weight_workers,weight_normal)),
      Parameters.parameter_values(Buffer.pack(~F32,bias),DenseNativeContract.parameter_normal(bias,Native.bias_extents(input),0.0,bias_workers,bias_normal))),_)
  checked_parameters(~custom,~Logical,~original,logical,activation,input,source,weights,bias,fill,input_workers,U32.max(input_workers,U32.max(weight_workers,bias_workers)),normal,matched,supported,values,
    DenseNativeContract.parameter_normal(weights,Native.weight_extents(input),0.0,weight_workers,weight_normal),DenseNativeContract.parameter_normal(bias,Native.bias_extents(input),0.0,bias_workers,bias_normal),validated)

# Shared band status delegates cover the other representation. These direct
# dense-head equations preserve the same sticky failure and native owner.
def failed_input(~custom: F32 -> F32,+activation: Activations.Activation,+problem: Tensors.Problem,-source: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+input_workers: U32,
  -weights: Storage.Buffer<F32>,+weight_shape: List<&2,U32>,+weight_fill: F32,+weight_status: Tensors.Status,+weight_workers: U32,
  -bias: Storage.Buffer<F32>,+bias_shape: List<&2,U32>,+bias_fill: F32,+bias_status: Tensors.Status,+bias_workers: U32,+stride: U32,+padding: U32) ->
  {TensorConvolution.conv2d(~custom,activation,Tensors.Tensor{source,shape,fill,Tensors.Failed{problem},input_workers},
      Tensors.Tensor{weights,weight_shape,weight_fill,weight_status,weight_workers},Tensors.Tensor{bias,bias_shape,bias_fill,bias_status,bias_workers},stride,padding) ==
    Tensors.Tensor{source,shape,fill,Tensors.Failed{problem},U32.max(input_workers,U32.max(weight_workers,bias_workers))} : Tensors.Tensor<F32>}:
  {==}

def failed_parameters(~custom: F32 -> F32,+activation: Activations.Activation,+problem: Tensors.Problem,-source: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+input_workers: U32,
  -weights: Storage.Buffer<F32>,+weight_shape: List<&2,U32>,-bias: Storage.Buffer<F32>,+bias_shape: List<&2,U32>,+parameter_workers: U32,+stride: U32,+padding: U32) ->
  {TensorConvolution.input_layout(~custom,activation,Tensors.Tensor{source,shape,fill,Tensors.Usable{},input_workers},weights,weight_shape,bias,bias_shape,Tensors.Failed{problem},parameter_workers,stride,padding) ==
    Tensors.Tensor{source,shape,fill,Tensors.Failed{problem},U32.max(input_workers,parameter_workers)} : Tensors.Tensor<F32>}:
  {==}

def invalid_geometry(~custom: F32 -> F32,+activation: Activations.Activation,-source: Storage.Buffer<F32>,-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {TensorConvolution.checked(~custom,activation,None{},source,weights,bias,shape,fill,workers) ==
    Tensors.Tensor{source,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidArgument{shape}}},workers} : Tensors.Tensor<F32>}:
  {==}

def rejected_guard(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: TensorConvolution.Geometry,-source: Storage.Buffer<F32>,-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {TensorConvolution.supported(~custom,activation,False{},geometry,source,weights,bias,shape,fill,workers) ==
    Tensors.Tensor{source,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidArgument{shape}}},workers} : Tensors.Tensor<F32>}:
  {==}

def rejected_machine(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: TensorConvolution.Geometry,-source: Storage.Buffer<F32>,-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {TensorConvolution.machine(~custom,activation,False{},geometry,source,weights,bias,shape,fill,workers) ==
    Tensors.Tensor{source,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidShape{shape}}},workers} : Tensors.Tensor<F32>}:
  {==}

# The existing band fallback observes the real counted gather once. Incoming
# band Normal and finite values produce dense source storage and semantics;
# the native dense caller then supplies the complete convolution result.
# The two public parameter selectors determine this flag from the original
# representations. The dense pair retains the existing hot path.
def bias_selected(+inherited: Bool,bias: BandConcatExecution.TensorOwnerModel<F32>) -> Bool:
  match bias:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}: inherited
    case BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers}: True{}

def parameters_selected(weights: BandConcatExecution.TensorOwnerModel<F32>,+bias: BandConcatExecution.TensorOwnerModel<F32>) -> Bool:
  match weights:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}: bias_selected(False{},bias)
    case BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers}: True{}

def bias_true_selected(+bias: BandConcatExecution.TensorOwnerModel<F32>) -> {bias_selected(True{},bias) == True{} : Bool}:
  match bias:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}: {==}
    case BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers}: {==}

def bias_route_matches(~custom: F32 -> F32,+activation: Activations.Activation,+inherited: Bool,
  +incoming: BandConcatExecution.TensorOwnerModel<F32>,+weights: BandConcatExecution.TensorOwnerModel<F32>,+bias: BandConcatExecution.TensorOwnerModel<F32>,+stride: U32,+padding: U32) ->
  {TensorConvolution.conv2d_bias_layout(~custom,activation,inherited,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) ==
    TensorConvolution.conv2d_selected(~custom,activation,bias_selected(inherited,bias),BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) : Tensors.Tensor<F32>}:
  match bias:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}: {==}
    case BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers}: {==}

def parameter_route_matches(~custom: F32 -> F32,+activation: Activations.Activation,
  +incoming: BandConcatExecution.TensorOwnerModel<F32>,+weights: BandConcatExecution.TensorOwnerModel<F32>,+bias: BandConcatExecution.TensorOwnerModel<F32>,+stride: U32,+padding: U32) ->
  {TensorConvolution.conv2d(~custom,activation,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) ==
    TensorConvolution.conv2d_selected(~custom,activation,parameters_selected(weights,bias),BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) : Tensors.Tensor<F32>}:
  match weights:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}:
      bias_route_matches(~custom,activation,False{},incoming,BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers},bias,stride,padding)
    case BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers}:
      %bias_true_selected(bias) :
        {TensorConvolution.conv2d(~custom,activation,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers}),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) ==
          TensorConvolution.conv2d_selected(~custom,activation,_,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers}),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) : Tensors.Tensor<F32>}
      bias_route_matches(~custom,activation,True{},incoming,BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers},bias,stride,padding)

# The body is a template, so the proof never duplicates an affine callback.
# Applying with_dense twice only projects the first call's dense constructor.
def parameter_composed(~Result: Type,~body: Storage.Buffer<F32> -> List<&2,U32> -> F32 -> Tensors.Status -> U32 -> Result,
  +owner: BandConcatExecution.TensorOwnerModel<F32>) ->
  {Tensors.with_dense(~F32,~Result,BandConcatExecution.tensor_owner_pack(~F32,owner),body) ==
    Tensors.with_dense(~F32,~Result,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner)),body) : Result}:
  match owner:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}: {==}
    case BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers}: {==}

def selected_weight_dense(~custom: F32 -> F32,+activation: Activations.Activation,+dense: Bool,
  +incoming: BandConcatExecution.TensorOwnerModel<F32>,+weights: BandConcatExecution.TensorOwnerModel<F32>,+bias: BandConcatExecution.TensorOwnerModel<F32>,+stride: U32,+padding: U32) ->
  {TensorConvolution.conv2d_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) ==
    TensorConvolution.conv2d_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) : Tensors.Tensor<F32>}:
  parameter_composed(~Tensors.Tensor<F32>,~weight_buffer => weight_shape => weight_fill => weight_status => weight_workers =>
    Tensors.with_dense(~F32,~Tensors.Tensor<F32>,BandConcatExecution.tensor_owner_pack(~F32,bias),bias_buffer => bias_shape => bias_fill => bias_status => bias_workers =>
      TensorConvolution.input_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),weight_buffer,weight_shape,bias_buffer,bias_shape,Tensors.first_failure(weight_status,bias_status),U32.max(weight_workers,bias_workers),stride,padding)),weights)

def selected_bias_dense(~custom: F32 -> F32,+activation: Activations.Activation,+dense: Bool,
  +incoming: BandConcatExecution.TensorOwnerModel<F32>,+weights: BandConcatExecution.TensorOwnerModel<F32>,+bias: BandConcatExecution.TensorOwnerModel<F32>,+stride: U32,+padding: U32) ->
  {TensorConvolution.conv2d_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) ==
    TensorConvolution.conv2d_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias)),stride,padding) : Tensors.Tensor<F32>}:
  match weights:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}:
      parameter_composed(~Tensors.Tensor<F32>,~bias_buffer => bias_shape => bias_fill => bias_status => bias_workers =>
        TensorConvolution.input_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),Buffer.pack(~F32,buffer),shape,bias_buffer,bias_shape,Tensors.first_failure(status,bias_status),U32.max(workers,bias_workers),stride,padding),bias)
    case BandConcatExecution.BandedOwner{bands,+geometry,shape,fill,status,workers}:
      parameter_composed(~Tensors.Tensor<F32>,~bias_buffer => bias_shape => bias_fill => bias_status => bias_workers =>
        TensorConvolution.input_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),Bands.gather(~F32,geometry,Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(geometry))),shape,bias_buffer,bias_shape,Tensors.first_failure(status,bias_status),U32.max(workers,bias_workers),stride,padding),bias)

def selected_parameters_dense(~custom: F32 -> F32,+activation: Activations.Activation,+dense: Bool,
  +incoming: BandConcatExecution.TensorOwnerModel<F32>,+weights: BandConcatExecution.TensorOwnerModel<F32>,+bias: BandConcatExecution.TensorOwnerModel<F32>,+stride: U32,+padding: U32) ->
  {TensorConvolution.conv2d_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) ==
    TensorConvolution.conv2d_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias)),stride,padding) : Tensors.Tensor<F32>}:
  Equal.trans(Tensors.Tensor<F32>,
    TensorConvolution.conv2d_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding),
    TensorConvolution.conv2d_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding),
    TensorConvolution.conv2d_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias)),stride,padding),
    selected_weight_dense(~custom,activation,dense,incoming,weights,bias,stride,padding),selected_bias_dense(~custom,activation,dense,incoming,weights,bias,stride,padding))

# This is an observation of the actual parameter densification, rather than
# an externally supplied successful parameter or an assumed output contract.
type NormalizedParameter<-owner: BandConcatExecution.TensorOwnerModel<F32>,-expected_shape: List<&2,U32>> is Data:
  NormalizedParameter{model: Buffer.Model<F32>,fill: F32,workers: U32,
    equation: {DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner)) == Tensors.Tensor{Buffer.pack(~F32,model),expected_shape,fill,Tensors.Usable{},workers} : Tensors.Tensor<F32>},
    normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,model),expected_shape,0.0,Tensors.Usable{},workers})}

def parameter_observed(+owner: BandConcatExecution.TensorOwnerModel<F32>,+expected_shape: List<&2,U32>,
  +normal: Global.Normal(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner)),+ready: {Dense.tensor_ready(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner)) == True{} : Bool},
  +shape_exact: {BandConcatExecution.tensor_shape(~F32,BandConcatExecution.tensor_contents(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner))) == expected_shape : List<&2,U32>},
  observation: DenseObservation.DenseObservation<F32,BandConcatExecution.tensor_owner_pack(~F32,owner)>) -> NormalizedParameter<owner,expected_shape>:
  DenseObservation.DenseObservation{+buffer,+shape,+fill,+status,+workers,+equation,+status_equal,+shape_equal} = observation
  +observed_shape = Equal.trans(List<&2,U32>,shape,BandConcatExecution.tensor_shape(~F32,BandConcatExecution.tensor_contents(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner))),expected_shape,
    Equal.sym(List<&2,U32>,BandConcatExecution.tensor_shape(~F32,BandConcatExecution.tensor_contents(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner))),shape,shape_equal),shape_exact)
  +observed_status = DenseObservation.ready_status(status,
    %status_equal : {_ == True{} : Bool} ready)
  NormalizedParameter{buffer,fill,workers,
    %observed_shape : {DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner)) == Tensors.Tensor{Buffer.pack(~F32,buffer),_,fill,Tensors.Usable{},workers} : Tensors.Tensor<F32>}
    %observed_status : {DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner)) == Tensors.Tensor{Buffer.pack(~F32,buffer),shape,fill,_,workers} : Tensors.Tensor<F32>}
    equation,
    %observed_shape : Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,buffer),_,fill,Tensors.Usable{},workers})
    %observed_status : Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,buffer),shape,fill,_,workers})
    ReadyDensification.observation_modeled(~F32,owner,DenseObservation.DenseObservation{buffer,shape,fill,status,workers,equation,status_equal,shape_equal},normal,ready)}

def parameter_prepared(+owner: BandConcatExecution.TensorOwnerModel<F32>,+expected_shape: List<&2,U32>,
  +normal: Global.Normal(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner)),+ready: {Dense.tensor_ready(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner)) == True{} : Bool},
  +shape_exact: {BandConcatExecution.tensor_shape(~F32,BandConcatExecution.tensor_contents(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner))) == expected_shape : List<&2,U32>}) -> NormalizedParameter<owner,expected_shape>:
  parameter_observed(owner,expected_shape,normal,ready,shape_exact,DenseObservation.observe_dense(~F32,BandConcatExecution.tensor_owner_pack(~F32,owner)))

def normalized_model(-owner: BandConcatExecution.TensorOwnerModel<F32>,+shape: List<&2,U32>,receipt: NormalizedParameter<owner,shape>) -> Buffer.Model<F32>:
  NormalizedParameter{model,fill,workers,equation,normal} = receipt
  model

def normalized_workers(-owner: BandConcatExecution.TensorOwnerModel<F32>,+shape: List<&2,U32>,receipt: NormalizedParameter<owner,shape>) -> U32:
  NormalizedParameter{model,fill,workers,equation,normal} = receipt
  workers

def normalized_parameter(-owner: BandConcatExecution.TensorOwnerModel<F32>,+shape: List<&2,U32>,+receipt: NormalizedParameter<owner,shape>) -> Parameters.Parameter<Buffer.pack(~F32,normalized_model(owner,shape,receipt))>:
  NormalizedParameter{+model,fill,+workers,equation,+normal} = receipt
  DenseNativeContract.parameter_normal(model,shape,0.0,workers,normal)

def normalized_values(-owner: BandConcatExecution.TensorOwnerModel<F32>,+shape: List<&2,U32>,+receipt: NormalizedParameter<owner,shape>) -> List<&2,F32>:
  Parameters.parameter_values(Buffer.pack(~F32,normalized_model(owner,shape,receipt)),normalized_parameter(owner,shape,receipt))

# The selected path is exactly one with_dense of the original owner. This
# operational equality also applies to failed status and arbitrary metadata.
def selected_dense_matches(~custom: F32 -> F32,+activation: Activations.Activation,+incoming: BandConcatExecution.TensorOwnerModel<F32>,
  -weights: Storage.Buffer<F32>,+weight_shape: List<&2,U32>,-bias: Storage.Buffer<F32>,+bias_shape: List<&2,U32>,+parameter_status: Tensors.Status,+parameter_workers: U32,+stride: U32,+padding: U32) ->
  {TensorConvolution.input_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),weights,weight_shape,bias,bias_shape,parameter_status,parameter_workers,stride,padding) ==
    TensorConvolution.input_layout(~custom,activation,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,incoming)),weights,weight_shape,bias,bias_shape,parameter_status,parameter_workers,stride,padding) : Tensors.Tensor<F32>}:
  match incoming:
    case BandConcatExecution.DenseOwner{buffer,shape,fill,status,workers}: {==}
    case BandConcatExecution.BandedOwner{bands,geometry,shape,fill,status,workers}: {==}

# This operational contract quantifies every source and parameter status and
# metadata. It introduces no Ready, geometry or successful leaf premise.
def MixedDenseEquation(~custom: F32 -> F32,+activation: Activations.Activation,
  +incoming: BandConcatExecution.TensorOwnerModel<F32>,+weights: BandConcatExecution.TensorOwnerModel<F32>,+bias: BandConcatExecution.TensorOwnerModel<F32>,+stride: U32,+padding: U32,dense: Bool) -> Data:
  match dense:
    case False{}: Unit
    case True{}:
      {TensorConvolution.conv2d(~custom,activation,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) ==
        TensorConvolution.conv2d(~custom,activation,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,incoming)),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias)),stride,padding) : Tensors.Tensor<F32>}

def mixed_dense_observed(~custom: F32 -> F32,+activation: Activations.Activation,
  +incoming: BandConcatExecution.TensorOwnerModel<F32>,+weights: BandConcatExecution.TensorOwnerModel<F32>,+bias: BandConcatExecution.TensorOwnerModel<F32>,+stride: U32,+padding: U32,+dense: Bool,
  routed: {TensorConvolution.conv2d(~custom,activation,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) ==
    TensorConvolution.conv2d_selected(~custom,activation,dense,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding) : Tensors.Tensor<F32>},
  weight_observation: DenseObservation.DenseObservation<F32,BandConcatExecution.tensor_owner_pack(~F32,weights)>,
  bias_observation: DenseObservation.DenseObservation<F32,BandConcatExecution.tensor_owner_pack(~F32,bias)>) ->
  MixedDenseEquation(~custom,activation,incoming,weights,bias,stride,padding,dense):
  match dense:
    case False{}: Unit{}
    case True{}:
      DenseObservation.DenseObservation{+weight_buffer,+weight_shape,+weight_fill,+weight_status,+weight_workers,+weight_equation,weight_status_equal,weight_shape_equal} = weight_observation
      DenseObservation.DenseObservation{+bias_buffer,+bias_shape,+bias_fill,+bias_status,+bias_workers,+bias_equation,bias_status_equal,bias_shape_equal} = bias_observation
      Equal.trans(Tensors.Tensor<F32>,
        TensorConvolution.conv2d(~custom,activation,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding),
        TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding),
        TensorConvolution.conv2d(~custom,activation,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,incoming)),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias)),stride,padding),routed,
        Equal.trans(Tensors.Tensor<F32>,
          TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weights),BandConcatExecution.tensor_owner_pack(~F32,bias),stride,padding),
          TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias)),stride,padding),
          TensorConvolution.conv2d(~custom,activation,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,incoming)),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias)),stride,padding),
          selected_parameters_dense(~custom,activation,True{},incoming,weights,bias,stride,padding),
          %Equal.sym(Tensors.Tensor<F32>,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),Tensors.Tensor{Buffer.pack(~F32,weight_buffer),weight_shape,weight_fill,weight_status,weight_workers},weight_equation) :
            {TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),_,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias)),stride,padding) ==
              TensorConvolution.conv2d(~custom,activation,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,incoming)),_,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias)),stride,padding) : Tensors.Tensor<F32>}
          %Equal.sym(Tensors.Tensor<F32>,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias)),Tensors.Tensor{Buffer.pack(~F32,bias_buffer),bias_shape,bias_fill,bias_status,bias_workers},bias_equation) :
            {TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),Tensors.Tensor{Buffer.pack(~F32,weight_buffer),weight_shape,weight_fill,weight_status,weight_workers},_,stride,padding) ==
              TensorConvolution.conv2d(~custom,activation,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,incoming)),Tensors.Tensor{Buffer.pack(~F32,weight_buffer),weight_shape,weight_fill,weight_status,weight_workers},_,stride,padding) : Tensors.Tensor<F32>}
          selected_dense_matches(~custom,activation,incoming,Buffer.pack(~F32,weight_buffer),weight_shape,Buffer.pack(~F32,bias_buffer),bias_shape,Tensors.first_failure(weight_status,bias_status),U32.max(weight_workers,bias_workers),stride,padding)))

# The public composition obtains the selector and observations itself.
# Source densification is handled by selected_dense_matches' two owner cases;
# each parameter observation uses the existing two-case observe_dense helper.
def mixed_dense_equation(~custom: F32 -> F32,+activation: Activations.Activation,
  +incoming: BandConcatExecution.TensorOwnerModel<F32>,+weights: BandConcatExecution.TensorOwnerModel<F32>,+bias: BandConcatExecution.TensorOwnerModel<F32>,+stride: U32,+padding: U32) ->
  MixedDenseEquation(~custom,activation,incoming,weights,bias,stride,padding,parameters_selected(weights,bias)):
  mixed_dense_observed(~custom,activation,incoming,weights,bias,stride,padding,parameters_selected(weights,bias),
    parameter_route_matches(~custom,activation,incoming,weights,bias,stride,padding),
    DenseObservation.observe_dense(~F32,BandConcatExecution.tensor_owner_pack(~F32,weights)),DenseObservation.observe_dense(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias)))

def selected_gather_matches(+geometry: Bands.Geometry,+bands: Model.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {DenseObservation.dense_tensor(~F32,Tensors.Banded{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}) ==
    Tensors.Tensor{Bands.gather(~F32,geometry,Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(geometry))),shape,fill,Tensors.Usable{},workers} : Tensors.Tensor<F32>}:
  {==}

type SelectedGathered<-Logical: Data,-original: Logical -> U32 -> F32,-logical: Logical,-input: PublicFrame.NativeInput,-bands: Model.Model<F32>,-fill: F32,-workers: U32> is Data:
  SelectedGathered{model: Buffer.Model<F32>,
    owner: {Buffer.pack(~F32,model) == Bands.gather(~F32,PublicFrame.source_geometry(input),Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(PublicFrame.source_geometry(input)))) : Storage.Buffer<F32>},
    normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,model),Native.input_extents(input),fill,Tensors.Usable{},workers}),
    values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,model))}

def selected_observed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: PublicFrame.NativeInput,+bands: Model.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},workers}),
  +values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,bands)),
  observation: Buffer.Observation<F32,Bands.gather(~F32,PublicFrame.source_geometry(input),Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(PublicFrame.source_geometry(input))))>) ->
  SelectedGathered<Logical,original,logical,input,bands,fill,workers>:
  Buffer.Observation{+model,+owner} = observation
  +incoming = {BandConcatExecution.BandedOwner{bands,PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},workers} : BandConcatExecution.TensorOwnerModel<F32>}
  +equation = Equal.trans(Tensors.Tensor<F32>,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,incoming)),
    Tensors.Tensor{Bands.gather(~F32,PublicFrame.source_geometry(input),Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(PublicFrame.source_geometry(input)))),Native.input_extents(input),fill,Tensors.Usable{},workers},
    Tensors.Tensor{Buffer.pack(~F32,model),Native.input_extents(input),fill,Tensors.Usable{},workers},selected_gather_matches(PublicFrame.source_geometry(input),bands,Native.input_extents(input),fill,workers),
    Equal.cong(Storage.Buffer<F32>,Tensors.Tensor<F32>,storage => Tensors.Tensor{storage,Native.input_extents(input),fill,Tensors.Usable{},workers},
      Bands.gather(~F32,PublicFrame.source_geometry(input),Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(PublicFrame.source_geometry(input)))),Buffer.pack(~F32,model),
      Equal.sym(Storage.Buffer<F32>,Buffer.pack(~F32,model),Bands.gather(~F32,PublicFrame.source_geometry(input),Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(PublicFrame.source_geometry(input)))),owner)))
  SelectedGathered{model,owner,
    %equation : Global.Normal(~F32,_)
    ReadyDensification.modeled(~F32,incoming,normal,{==}),
    NormalDensificationValues.observation_owner(~F32,~Logical,~original,logical,incoming,model,Native.input_extents(input),fill,Tensors.Usable{},workers,equation,normal,{==},
      %Equal.sym(Model.Content<F32>,Model.contents(~F32,Model.pack(~F32,bands)),Model.content(~F32,bands),Model.packed_contents(~F32,bands)) :
        BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),_)
      values)}

def selected_prepared(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: PublicFrame.NativeInput,+bands: Model.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},workers}),
  values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,bands))) -> SelectedGathered<Logical,original,logical,input,bands,fill,workers>:
  selected_observed(~Logical,~original,logical,input,bands,fill,workers,normal,values,
    Buffer.observe(~F32,Bands.gather(~F32,PublicFrame.source_geometry(input),Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(PublicFrame.source_geometry(input))))))

def selected_input_header(~custom: F32 -> F32,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+parameter_workers: U32,
  matched: {Native.input_matched(input) == True{} : Bool}) ->
  {TensorConvolution.input_layout(~custom,activation,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers},Buffer.pack(~F32,weights),Native.weight_extents(input),Buffer.pack(~F32,bias),Native.bias_extents(input),Tensors.Usable{},parameter_workers,Native.stride(input),Native.padding(input)) ==
    TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,U32.max(input_workers,parameter_workers)) : Tensors.Tensor<F32>}:
  %Equal.sym(Maybe<&2,TensorConvolution.Geometry>,TensorConvolution.geometry(TensorConvolution.input_shape(Native.input_extents(input)),Native.weight_extents(input),Native.bias_extents(input),Native.stride(input),Native.padding(input)),Some{PublicFrame.geometry(input)},Native.public_geometry(input,matched)) :
    {TensorConvolution.checked(~custom,activation,_,Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,U32.max(input_workers,parameter_workers)) ==
      TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,U32.max(input_workers,parameter_workers)) : Tensors.Tensor<F32>}
  {==}

def selected_returned(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +bands: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+weight_workers: U32,+bias_workers: U32,
  +weight_normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,weights),Native.weight_extents(input),0.0,Tensors.Usable{},weight_workers}),
  +bias_normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,bias),Native.bias_extents(input),0.0,Tensors.Usable{},bias_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  validated: {PublicFrame.frame_valid(input,
    Parameters.parameter_values(Buffer.pack(~F32,weights),DenseNativeContract.parameter_normal(weights,Native.weight_extents(input),0.0,weight_workers,weight_normal)),
    Parameters.parameter_values(Buffer.pack(~F32,bias),DenseNativeContract.parameter_normal(bias,Native.bias_extents(input),0.0,bias_workers,bias_normal))) == True{} : Bool},
  receipt: SelectedGathered<Logical,original,logical,input,bands,fill,input_workers>) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,
    Parameters.parameter_values(Buffer.pack(~F32,weights),DenseNativeContract.parameter_normal(weights,Native.weight_extents(input),0.0,weight_workers,weight_normal)),
    Parameters.parameter_values(Buffer.pack(~F32,bias),DenseNativeContract.parameter_normal(bias,Native.bias_extents(input),0.0,bias_workers,bias_normal))),
    TensorConvolution.input_selected(~custom,activation,True{},Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers},Buffer.pack(~F32,weights),Native.weight_extents(input),Buffer.pack(~F32,bias),Native.bias_extents(input),Tensors.Usable{},U32.max(weight_workers,bias_workers),Native.stride(input),Native.padding(input))):
  SelectedGathered{+model,+owner,+normal,+values} = receipt
  +weight_parameter = DenseNativeContract.parameter_normal(weights,Native.weight_extents(input),0.0,weight_workers,weight_normal)
  +bias_parameter = DenseNativeContract.parameter_normal(bias,Native.bias_extents(input),0.0,bias_workers,bias_normal)
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.input_selected(~custom,activation,True{},Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers},Buffer.pack(~F32,weights),Native.weight_extents(input),Buffer.pack(~F32,bias),Native.bias_extents(input),Tensors.Usable{},U32.max(weight_workers,bias_workers),Native.stride(input),Native.padding(input)),
    TensorConvolution.input_layout(~custom,activation,DenseObservation.dense_tensor(~F32,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),Buffer.pack(~F32,weights),Native.weight_extents(input),Buffer.pack(~F32,bias),Native.bias_extents(input),Tensors.Usable{},U32.max(weight_workers,bias_workers),Native.stride(input),Native.padding(input)),
    selected_dense_matches(~custom,activation,BandConcatExecution.BandedOwner{bands,PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers},Buffer.pack(~F32,weights),Native.weight_extents(input),Buffer.pack(~F32,bias),Native.bias_extents(input),Tensors.Usable{},U32.max(weight_workers,bias_workers),Native.stride(input),Native.padding(input))) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,Parameters.parameter_values(Buffer.pack(~F32,weights),weight_parameter),Parameters.parameter_values(Buffer.pack(~F32,bias),bias_parameter)),_)
  %Equal.sym(Tensors.Tensor<F32>,DenseObservation.dense_tensor(~F32,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
    Tensors.Tensor{Bands.gather(~F32,PublicFrame.source_geometry(input),Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(PublicFrame.source_geometry(input)))),Native.input_extents(input),fill,Tensors.Usable{},input_workers},
    selected_gather_matches(PublicFrame.source_geometry(input),bands,Native.input_extents(input),fill,input_workers)) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,Parameters.parameter_values(Buffer.pack(~F32,weights),weight_parameter),Parameters.parameter_values(Buffer.pack(~F32,bias),bias_parameter)),
      TensorConvolution.input_layout(~custom,activation,_,Buffer.pack(~F32,weights),Native.weight_extents(input),Buffer.pack(~F32,bias),Native.bias_extents(input),Tensors.Usable{},U32.max(weight_workers,bias_workers),Native.stride(input),Native.padding(input)))
  %owner :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,Parameters.parameter_values(Buffer.pack(~F32,weights),weight_parameter),Parameters.parameter_values(Buffer.pack(~F32,bias),bias_parameter)),
      TensorConvolution.input_layout(~custom,activation,Tensors.Tensor{_,Native.input_extents(input),fill,Tensors.Usable{},input_workers},Buffer.pack(~F32,weights),Native.weight_extents(input),Buffer.pack(~F32,bias),Native.bias_extents(input),Tensors.Usable{},U32.max(weight_workers,bias_workers),Native.stride(input),Native.padding(input)))
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.input_layout(~custom,activation,Tensors.Tensor{Buffer.pack(~F32,model),Native.input_extents(input),fill,Tensors.Usable{},input_workers},Buffer.pack(~F32,weights),Native.weight_extents(input),Buffer.pack(~F32,bias),Native.bias_extents(input),Tensors.Usable{},U32.max(weight_workers,bias_workers),Native.stride(input),Native.padding(input)),
    TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,model),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers))),
    selected_input_header(~custom,activation,input,model,weights,bias,fill,input_workers,U32.max(weight_workers,bias_workers),matched)) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,Parameters.parameter_values(Buffer.pack(~F32,weights),weight_parameter),Parameters.parameter_values(Buffer.pack(~F32,bias),bias_parameter)),_)
  checked_parameters(~custom,~Logical,~original,logical,activation,input,model,weights,bias,fill,input_workers,U32.max(input_workers,U32.max(weight_workers,bias_workers)),normal,matched,supported,values,weight_parameter,bias_parameter,validated)

def selected_ready(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +bands: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+weight_workers: U32,+bias_workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +weight_normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,weights),Native.weight_extents(input),0.0,Tensors.Usable{},weight_workers}),
  +bias_normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,bias),Native.bias_extents(input),0.0,Tensors.Usable{},bias_workers}),
  +values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,bands)),
  +supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  validated: {PublicFrame.frame_valid(input,
    Parameters.parameter_values(Buffer.pack(~F32,weights),DenseNativeContract.parameter_normal(weights,Native.weight_extents(input),0.0,weight_workers,weight_normal)),
    Parameters.parameter_values(Buffer.pack(~F32,bias),DenseNativeContract.parameter_normal(bias,Native.bias_extents(input),0.0,bias_workers,bias_normal))) == True{} : Bool}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,
    Parameters.parameter_values(Buffer.pack(~F32,weights),DenseNativeContract.parameter_normal(weights,Native.weight_extents(input),0.0,weight_workers,weight_normal)),
    Parameters.parameter_values(Buffer.pack(~F32,bias),DenseNativeContract.parameter_normal(bias,Native.bias_extents(input),0.0,bias_workers,bias_normal))),
    TensorConvolution.input_selected(~custom,activation,True{},Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers},Buffer.pack(~F32,weights),Native.weight_extents(input),Buffer.pack(~F32,bias),Native.bias_extents(input),Tensors.Usable{},U32.max(weight_workers,bias_workers),Native.stride(input),Native.padding(input))):
  selected_returned(~custom,~Logical,~original,logical,activation,input,bands,weights,bias,fill,input_workers,weight_workers,bias_workers,weight_normal,bias_normal,
    NormalCaller.input_matched(input,NormalCaller.source_coherent(input,bands,fill,input_workers,normal)),supported,validated,selected_prepared(~Logical,~original,logical,input,bands,fill,input_workers,normal,values))

def mixed_context(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  -weight_owner: BandConcatExecution.TensorOwnerModel<F32>,-bias_owner: BandConcatExecution.TensorOwnerModel<F32>,
  +weight_receipt: NormalizedParameter<weight_owner,Native.weight_extents(input)>,+bias_receipt: NormalizedParameter<bias_owner,Native.bias_extents(input)>) -> BandConvolutionValues.Context<Logical>:
  DenseNativeContract.sources_context(~Logical,logical,activation,input,normalized_values(weight_owner,Native.weight_extents(input),weight_receipt),normalized_values(bias_owner,Native.bias_extents(input),bias_receipt))

def prepared_parameters_matches(~custom: F32 -> F32,+activation: Activations.Activation,+incoming: BandConcatExecution.TensorOwnerModel<F32>,
  +weight_owner: BandConcatExecution.TensorOwnerModel<F32>,+bias_owner: BandConcatExecution.TensorOwnerModel<F32>,+weight_shape: List<&2,U32>,+bias_shape: List<&2,U32>,
  +weight_receipt: NormalizedParameter<weight_owner,weight_shape>,+bias_receipt: NormalizedParameter<bias_owner,bias_shape>,+stride: U32,+padding: U32) ->
  {TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weight_owner),BandConcatExecution.tensor_owner_pack(~F32,bias_owner),stride,padding) ==
    TensorConvolution.input_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),
      Buffer.pack(~F32,normalized_model(weight_owner,weight_shape,weight_receipt)),weight_shape,
      Buffer.pack(~F32,normalized_model(bias_owner,bias_shape,bias_receipt)),bias_shape,Tensors.Usable{},
      U32.max(normalized_workers(weight_owner,weight_shape,weight_receipt),normalized_workers(bias_owner,bias_shape,bias_receipt)),stride,padding) : Tensors.Tensor<F32>}:
  NormalizedParameter{+weight_model,+weight_fill,+weight_workers,+weight_equation,weight_normal} = weight_receipt
  NormalizedParameter{+bias_model,+bias_fill,+bias_workers,+bias_equation,bias_normal} = bias_receipt
  Equal.trans(Tensors.Tensor<F32>,
    TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weight_owner),BandConcatExecution.tensor_owner_pack(~F32,bias_owner),stride,padding),
    TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weight_owner)),DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias_owner)),stride,padding),
    TensorConvolution.input_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),Buffer.pack(~F32,weight_model),weight_shape,Buffer.pack(~F32,bias_model),bias_shape,Tensors.Usable{},U32.max(weight_workers,bias_workers),stride,padding),
    selected_parameters_dense(~custom,activation,True{},incoming,weight_owner,bias_owner,stride,padding),
    %Equal.sym(Tensors.Tensor<F32>,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,weight_owner)),Tensors.Tensor{Buffer.pack(~F32,weight_model),weight_shape,weight_fill,Tensors.Usable{},weight_workers},weight_equation) :
      {TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),_,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias_owner)),stride,padding) ==
        TensorConvolution.input_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),Buffer.pack(~F32,weight_model),weight_shape,Buffer.pack(~F32,bias_model),bias_shape,Tensors.Usable{},U32.max(weight_workers,bias_workers),stride,padding) : Tensors.Tensor<F32>}
    %Equal.sym(Tensors.Tensor<F32>,DenseObservation.dense_tensor(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias_owner)),Tensors.Tensor{Buffer.pack(~F32,bias_model),bias_shape,bias_fill,Tensors.Usable{},bias_workers},bias_equation) :
      {TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),Tensors.Tensor{Buffer.pack(~F32,weight_model),weight_shape,weight_fill,Tensors.Usable{},weight_workers},_,stride,padding) ==
        TensorConvolution.input_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),Buffer.pack(~F32,weight_model),weight_shape,Buffer.pack(~F32,bias_model),bias_shape,Tensors.Usable{},U32.max(weight_workers,bias_workers),stride,padding) : Tensors.Tensor<F32>}
    {==})

# This capability proves the newly selected non-hot route. False is the
# original dense-parameter hot path and remains with its existing caller.
def MixedContract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,dense: Bool,-result: Tensors.Tensor<F32>) -> Data:
  match dense:
    case False{}: Unit
    case True{}: InitialLayout.LayoutContract(~custom,~Logical,~original,context,result)

def mixed_observed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +bands: Model.Model<F32>,+weight_owner: BandConcatExecution.TensorOwnerModel<F32>,+bias_owner: BandConcatExecution.TensorOwnerModel<F32>,+fill: F32,+input_workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,bands)),
  +supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +weight_receipt: NormalizedParameter<weight_owner,Native.weight_extents(input)>,+bias_receipt: NormalizedParameter<bias_owner,Native.bias_extents(input)>,
  +dense: Bool,
  +routed: {TensorConvolution.conv2d(~custom,activation,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers},BandConcatExecution.tensor_owner_pack(~F32,weight_owner),BandConcatExecution.tensor_owner_pack(~F32,bias_owner),Native.stride(input),Native.padding(input)) ==
    TensorConvolution.conv2d_selected(~custom,activation,dense,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers},BandConcatExecution.tensor_owner_pack(~F32,weight_owner),BandConcatExecution.tensor_owner_pack(~F32,bias_owner),Native.stride(input),Native.padding(input)) : Tensors.Tensor<F32>},
  validated: {PublicFrame.frame_valid(input,normalized_values(weight_owner,Native.weight_extents(input),weight_receipt),normalized_values(bias_owner,Native.bias_extents(input),bias_receipt)) == True{} : Bool}) ->
  MixedContract(~custom,~Logical,~original,mixed_context(~Logical,logical,activation,input,weight_owner,bias_owner,weight_receipt,bias_receipt),dense,
    TensorConvolution.conv2d(~custom,activation,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers},BandConcatExecution.tensor_owner_pack(~F32,weight_owner),BandConcatExecution.tensor_owner_pack(~F32,bias_owner),Native.stride(input),Native.padding(input))):
  NormalizedParameter{+weight_model,+weight_fill,+weight_workers,+weight_equation,+weight_normal} = weight_receipt
  NormalizedParameter{+bias_model,+bias_fill,+bias_workers,+bias_equation,+bias_normal} = bias_receipt
  match dense:
    case False{}: Unit{}
    case True{}:
      +incoming = {BandConcatExecution.BandedOwner{bands,PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers} : BandConcatExecution.TensorOwnerModel<F32>}
      +prepared_weights = {NormalizedParameter{weight_model,weight_fill,weight_workers,weight_equation,weight_normal} : NormalizedParameter<weight_owner,Native.weight_extents(input)>}
      +prepared_bias = {NormalizedParameter{bias_model,bias_fill,bias_workers,bias_equation,bias_normal} : NormalizedParameter<bias_owner,Native.bias_extents(input)>}
      %Equal.sym(Tensors.Tensor<F32>,
        TensorConvolution.conv2d(~custom,activation,BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weight_owner),BandConcatExecution.tensor_owner_pack(~F32,bias_owner),Native.stride(input),Native.padding(input)),
        TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weight_owner),BandConcatExecution.tensor_owner_pack(~F32,bias_owner),Native.stride(input),Native.padding(input)),routed) :
        InitialLayout.LayoutContract(~custom,~Logical,~original,mixed_context(~Logical,logical,activation,input,weight_owner,bias_owner,prepared_weights,prepared_bias),_)
      %Equal.sym(Tensors.Tensor<F32>,
        TensorConvolution.conv2d_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),BandConcatExecution.tensor_owner_pack(~F32,weight_owner),BandConcatExecution.tensor_owner_pack(~F32,bias_owner),Native.stride(input),Native.padding(input)),
        TensorConvolution.input_selected(~custom,activation,True{},BandConcatExecution.tensor_owner_pack(~F32,incoming),Buffer.pack(~F32,weight_model),Native.weight_extents(input),Buffer.pack(~F32,bias_model),Native.bias_extents(input),Tensors.Usable{},U32.max(weight_workers,bias_workers),Native.stride(input),Native.padding(input)),
        prepared_parameters_matches(~custom,activation,incoming,weight_owner,bias_owner,Native.weight_extents(input),Native.bias_extents(input),prepared_weights,prepared_bias,Native.stride(input),Native.padding(input))) :
        InitialLayout.LayoutContract(~custom,~Logical,~original,mixed_context(~Logical,logical,activation,input,weight_owner,bias_owner,prepared_weights,prepared_bias),_)
      selected_ready(~custom,~Logical,~original,logical,activation,input,bands,weight_model,bias_model,fill,input_workers,weight_workers,bias_workers,normal,weight_normal,bias_normal,values,supported,validated)

# All routing and parameter observations are produced here from the original
# owners. The remaining premises describe legal Ready input geometry/storage;
# neither a True selector nor an output/leaf success contract is an argument.
def mixed_ready(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +bands: Model.Model<F32>,+weight_owner: BandConcatExecution.TensorOwnerModel<F32>,+bias_owner: BandConcatExecution.TensorOwnerModel<F32>,+fill: F32,+input_workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +weight_normal: Global.Normal(~F32,BandConcatExecution.tensor_owner_pack(~F32,weight_owner)),+bias_normal: Global.Normal(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias_owner)),
  +weight_ready: {Dense.tensor_ready(~F32,BandConcatExecution.tensor_owner_pack(~F32,weight_owner)) == True{} : Bool},+bias_ready: {Dense.tensor_ready(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias_owner)) == True{} : Bool},
  +weight_shape: {BandConcatExecution.tensor_shape(~F32,BandConcatExecution.tensor_contents(~F32,BandConcatExecution.tensor_owner_pack(~F32,weight_owner))) == Native.weight_extents(input) : List<&2,U32>},
  +bias_shape: {BandConcatExecution.tensor_shape(~F32,BandConcatExecution.tensor_contents(~F32,BandConcatExecution.tensor_owner_pack(~F32,bias_owner))) == Native.bias_extents(input) : List<&2,U32>},
  +values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,bands)),+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  validated: {PublicFrame.frame_valid(input,
    normalized_values(weight_owner,Native.weight_extents(input),parameter_prepared(weight_owner,Native.weight_extents(input),weight_normal,weight_ready,weight_shape)),
    normalized_values(bias_owner,Native.bias_extents(input),parameter_prepared(bias_owner,Native.bias_extents(input),bias_normal,bias_ready,bias_shape))) == True{} : Bool}) ->
  MixedContract(~custom,~Logical,~original,mixed_context(~Logical,logical,activation,input,weight_owner,bias_owner,
      parameter_prepared(weight_owner,Native.weight_extents(input),weight_normal,weight_ready,weight_shape),parameter_prepared(bias_owner,Native.bias_extents(input),bias_normal,bias_ready,bias_shape)),
    parameters_selected(weight_owner,bias_owner),TensorConvolution.conv2d(~custom,activation,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers},BandConcatExecution.tensor_owner_pack(~F32,weight_owner),BandConcatExecution.tensor_owner_pack(~F32,bias_owner),Native.stride(input),Native.padding(input))):
  +prepared_weights = parameter_prepared(weight_owner,Native.weight_extents(input),weight_normal,weight_ready,weight_shape)
  +prepared_bias = parameter_prepared(bias_owner,Native.bias_extents(input),bias_normal,bias_ready,bias_shape)
  mixed_observed(~custom,~Logical,~original,logical,activation,input,bands,weight_owner,bias_owner,fill,input_workers,normal,values,supported,prepared_weights,prepared_bias,
    parameters_selected(weight_owner,bias_owner),parameter_route_matches(~custom,activation,BandConcatExecution.BandedOwner{bands,PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers},weight_owner,bias_owner,Native.stride(input),Native.padding(input)),validated)

def natural_counted(+geometry: Bands.Geometry,+shape: List<&2,U32>,+counted: Global.Counted<shape>,
  +coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  evidence: Count.Count<BandGather.elements(geometry),Global.count(shape,counted)>) -> {DenseSources.count(geometry) == Bands.elements(geometry) : U32}:
  Count.Count{exact,bounded} = evidence
  Equal.trans(U32,DenseSources.count(geometry),Global.count(shape,counted),Bands.elements(geometry),
    Equal.trans(U32,DenseSources.count(geometry),U32.from_nat(U32.to_nat(Global.count(shape,counted))),Global.count(shape,counted),
      Equal.cong(Nat,U32,U32.from_nat,BandGather.elements(geometry),U32.to_nat(Global.count(shape,counted)),exact),Arithmetic.native_roundtrip(Global.count(shape,counted))),
    Equal.sym(U32,Bands.elements(geometry),Global.count(shape,counted),BandShapeCount.global_element_count_exact(shape,geometry,counted,coherent)))

def natural_count(+geometry: Bands.Geometry,+shape: List<&2,U32>,+counted: Global.Counted<shape>,+coherent: {Window.band_shape(shape,geometry) == True{} : Bool}) ->
  {DenseSources.count(geometry) == Bands.elements(geometry) : U32}:
  natural_counted(geometry,shape,counted,coherent,Global.geometry_evidence(shape,geometry,counted,coherent))

def fallback_allocation(+geometry: Bands.Geometry,-bands: Bands.Bands(~F32)) ->
  {DenseSingle.gathered(geometry,bands) == Bands.gather(~F32,geometry,bands,Tensors.buffer(~F32,0.0,DenseSources.count(geometry))) : Storage.Buffer<F32>}:
  Bands.Geometry{images,channels,height,width} = geometry
  {==}

def fallback_counted(+geometry: Bands.Geometry,+shape: List<&2,U32>,+bands: Model.Model<F32>,+counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool}) ->
  {DenseSingle.gathered(geometry,Model.pack(~F32,bands)) == Bands.gather(~F32,geometry,Model.pack(~F32,bands),Tensors.buffer(~F32,0.0,Bands.elements(geometry))) : Storage.Buffer<F32>}:
  Equal.trans(Storage.Buffer<F32>,DenseSingle.gathered(geometry,Model.pack(~F32,bands)),Bands.gather(~F32,geometry,Model.pack(~F32,bands),Tensors.buffer(~F32,0.0,DenseSources.count(geometry))),Bands.gather(~F32,geometry,Model.pack(~F32,bands),Tensors.buffer(~F32,0.0,Bands.elements(geometry))),
    fallback_allocation(geometry,Model.pack(~F32,bands)),
    Equal.cong(U32,Storage.Buffer<F32>,count => Bands.gather(~F32,geometry,Model.pack(~F32,bands),Tensors.buffer(~F32,0.0,count)),DenseSources.count(geometry),Bands.elements(geometry),natural_count(geometry,shape,counted,coherent)))

type Gathered<-Logical: Data,-original: Logical -> U32 -> F32,-logical: Logical,-input: PublicFrame.NativeInput,-bands: Model.Model<F32>,-workers: U32> is Data:
  Gathered{model: Buffer.Model<F32>,
    owner: {Buffer.pack(~F32,model) == DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands)) : Storage.Buffer<F32>},
    normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,model),Native.input_extents(input),0.0,Tensors.Usable{},workers}),
    values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,model))}

def fallback_observed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: PublicFrame.NativeInput,+bands: Model.Model<F32>,+workers: U32,
  +well_formed: Dense.WellFormed<F32,Native.input_extents(input),Buffer.contents(~F32,DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands)))>,
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.contents(~F32,DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands)))),
  observation: Buffer.Observation<F32,DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands))>) -> Gathered<Logical,original,logical,input,bands,workers>:
  Buffer.Observation{+model,+equation} = observation
  +content_exact = Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,model),Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.contents(~F32,DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands))),
    Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.content(~F32,model),Buffer.packed_contents(~F32,model)),
    Equal.cong(Storage.Buffer<F32>,Buffer.Content<F32>,Buffer.contents(~F32),Buffer.pack(~F32,model),DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands)),equation))
  Gathered{model,equation,
    DenseNormal.from_well_formed(~F32,Buffer.pack(~F32,model),Native.input_extents(input),0.0,Tensors.Usable{},workers,
      %Equal.sym(Storage.Buffer<F32>,Buffer.pack(~F32,model),DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands)),equation) : Dense.WellFormed<F32,Native.input_extents(input),Buffer.contents(~F32,_)> well_formed),
    %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,model),Buffer.contents(~F32,DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands))),content_exact) : Dense.Values(~F32,~Logical,~original,logical,_)
    values}

def prepared_shape(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: PublicFrame.NativeInput,+bands: Model.Model<F32>,+workers: U32,
  +metadata: BandShape.WellFormed<F32,PublicFrame.source_geometry(input),Model.content(~F32,bands)>,
  +values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,bands)),
  shape: Shape.Both<Global.Counted<Native.input_extents(input)>,{Window.band_shape(Native.input_extents(input),PublicFrame.source_geometry(input)) == True{} : Bool}>) -> Gathered<Logical,original,logical,input,bands,workers>:
  Shape.Both{+counted,+coherent} = shape
  +owner = fallback_counted(PublicFrame.source_geometry(input),Native.input_extents(input),bands,counted,coherent)
  fallback_observed(~Logical,~original,logical,input,bands,workers,
    %Equal.sym(Storage.Buffer<F32>,DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands)),Bands.gather(~F32,PublicFrame.source_geometry(input),Model.pack(~F32,bands),Tensors.buffer(~F32,0.0,Bands.elements(PublicFrame.source_geometry(input)))),owner) : Dense.WellFormed<F32,Native.input_extents(input),Buffer.contents(~F32,_)>
    Normal.gathered(~F32,PublicFrame.source_geometry(input),bands,Native.input_extents(input),0.0,counted,coherent),
    %Equal.sym(Storage.Buffer<F32>,DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands)),Bands.gather(~F32,PublicFrame.source_geometry(input),Model.pack(~F32,bands),Tensors.buffer(~F32,0.0,Bands.elements(PublicFrame.source_geometry(input)))),owner) : Dense.Values(~F32,~Logical,~original,logical,Buffer.contents(~F32,_))
    NormalDensificationValues.banded(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),bands,Native.input_extents(input),0.0,counted,coherent,metadata,values),
    Buffer.observe(~F32,DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands))))

def prepared(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: PublicFrame.NativeInput,+bands: Model.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},workers}),
  +values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,bands))) -> Gathered<Logical,original,logical,input,bands,workers>:
  prepared_shape(~Logical,~original,logical,input,bands,workers,NormalCaller.source_well_formed(input,bands,fill,workers,normal),values,NormalCaller.source_shape(input,bands,fill,workers,normal))

def returned_false(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +bands: Model.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+input_workers: U32,+workers: U32,
  +matched: {Native.input_matched(input) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  receipt: Gathered<Logical,original,logical,input,bands,input_workers>) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute_existing_layout(~custom,activation,False{},PublicFrame.geometry(input),PublicFrame.source_geometry(input),Model.pack(~F32,bands),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  Gathered{+model,+owner,+normal,+values} = receipt
  %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.execute_existing_layout(~custom,activation,False{},PublicFrame.geometry(input),PublicFrame.source_geometry(input),Model.pack(~F32,bands),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers),
    TensorConvolution.execute_dense(~custom,activation,PublicFrame.geometry(input),DenseSingle.gathered(PublicFrame.source_geometry(input),Model.pack(~F32,bands)),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers),DenseSingle.false_route(~custom,activation,PublicFrame.geometry(input),PublicFrame.source_geometry(input),Model.pack(~F32,bands),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),_)
  %owner : InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute_dense(~custom,activation,PublicFrame.geometry(input),_,TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers))
  DenseNativeContract.contract_layout(~custom,~Logical,~original,logical,activation,input,weights,bias,model,weight_model,bias_model,0.0,input_workers,workers,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_exact,values,validated,supported)

def existing_false(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +bands: Model.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,bands)),
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute_existing_layout(~custom,activation,False{},PublicFrame.geometry(input),PublicFrame.source_geometry(input),Model.pack(~F32,bands),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  returned_false(~custom,~Logical,~original,logical,activation,input,weights,bias,bands,weight_model,bias_model,input_workers,workers,NormalCaller.input_matched(input,NormalCaller.source_coherent(input,bands,fill,input_workers,normal)),weight_evidence,bias_evidence,weight_exact,bias_exact,validated,supported,
    prepared(~Logical,~original,logical,input,bands,fill,input_workers,normal,values))

def selected_false(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +bands: Model.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,bands),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,bands)),
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  disabled: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == False{} : Bool}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute_existing_layout(~custom,activation,TensorConvolution.uses_band_windows(PublicFrame.geometry(input)),PublicFrame.geometry(input),PublicFrame.source_geometry(input),Model.pack(~F32,bands),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  %Equal.sym(Bool,TensorConvolution.uses_band_windows(PublicFrame.geometry(input)),False{},disabled) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
      TensorConvolution.execute_existing_layout(~custom,activation,_,PublicFrame.geometry(input),PublicFrame.source_geometry(input),Model.pack(~F32,bands),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers))
  existing_false(~custom,~Logical,~original,logical,activation,input,weights,bias,bands,weight_model,bias_model,fill,input_workers,workers,normal,values,weight_evidence,bias_evidence,weight_exact,bias_exact,validated,supported)
