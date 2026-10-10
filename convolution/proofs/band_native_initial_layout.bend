# Actual native dense input initializes the selected band schedule. The frame,
# source domain, finite split values and output Normal come from their existing
# producers. Plan classification is kept at the actual initial_layout head.
import Base
import ../specification.bend as ConvSpec
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/storage_buffer.bend as Storage
import ../../lib/kernel_band_planning.bend as BandPlanning
import ../../lib/math_activation.bend as Activations
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_dense_contract.bend as Dense
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/word_count_flags.bend as Flags
import ./band_public_frame.bend as PublicFrame
import ./band_native_caller.bend as Native
import ./band_normal_caller.bend as NormalCaller
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_tensor_execution.bend as TensorExecution
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_output_metadata.bend as OutputMetadata
import ./band_output_normal.bend as BandOutputNormal
import ./band_dense_native_contract.bend as DenseNativeContract
import ./band_dense_public.bend as DensePublic
import ./band_initial_layout.bend as InitialLayout

def context_matches(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>) ->
  {PublicFrame.reference_context(~Logical,logical,activation,input,weights,bias) == DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias) : BandConvolutionValues.Context<Logical>}:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  {==}

def header_fields(~custom: F32 -> F32,~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,+depth: U32,+workers: U32,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +frame: PipelineExecution.Frame<Logical>,+source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,
  +fields: PublicFrame.Fields<Logical,logical,activation,input,weights,bias,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits())>,
  parallel_exact: {TensorExecution.execution_parallel(~Logical,frame) == parallel : Bool}) ->
  BandOutputNormal.Header(Native.batched(input),OutputMetadata.bounds_geometry(~Logical,frame),TensorConvolution.start_bands(~custom,activation,depth,parallel,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  PublicFrame.Fields{logical_exact,+activation_exact,images_exact,dimensions_exact,input_geometry_exact,weights_exact,bias_exact,context_exact,limits_exact,capacity_exact} = fields
  +geometry_exact = Native.geometry_fields(~Logical,logical,activation,input,weights,bias,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits()),fields)
  %activation_exact : BandOutputNormal.Header(Native.batched(input),OutputMetadata.bounds_geometry(~Logical,frame),TensorConvolution.start_bands(~custom,_,depth,parallel,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers))
  %parallel_exact : BandOutputNormal.Header(Native.batched(input),OutputMetadata.bounds_geometry(~Logical,frame),TensorConvolution.start_bands(~custom,TensorExecution.execution_activation(~Logical,frame),depth,_,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers))
  %geometry_exact : BandOutputNormal.Header(Native.batched(input),OutputMetadata.bounds_geometry(~Logical,frame),TensorConvolution.start_bands(~custom,TensorExecution.execution_activation(~Logical,frame),depth,TensorExecution.execution_parallel(~Logical,frame),_,Buffer.pack(~F32,source),TensorConvolution.packed(_,Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers))
  InitialLayout.start_header(~custom,~Logical,frame,Native.batched(input),depth,workers,source,weight_model,bias_model)

def start_fields(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,+depth: U32,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +frame: PipelineExecution.Frame<Logical>,+source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +fields: PublicFrame.Fields<Logical,logical,activation,input,weights,bias,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits())>,
  +parallel_exact: {TensorExecution.execution_parallel(~Logical,frame) == parallel : Bool},+matched: {Native.input_matched(input) == True{} : Bool},
  values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.start_bands(~custom,activation,depth,parallel,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  PublicFrame.Fields{logical_exact,activation_exact,images_exact,dimensions_exact,input_geometry_exact,weights_exact,bias_content_exact,+context_exact,limits_exact,capacity_exact} = fields
  %Equal.trans(BandConvolutionValues.Context<Logical>,PipelineExecution.output_context(~Logical,frame),PublicFrame.reference_context(~Logical,logical,activation,input,weights,bias),DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),context_exact,context_matches(~Logical,logical,activation,input,weights,bias)) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,_,TensorConvolution.start_bands(~custom,activation,depth,parallel,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers))
  InitialLayout.from_initial(~custom,~Logical,~original,frame,Native.batched(input),
    TensorConvolution.start_bands(~custom,activation,depth,parallel,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers),
    header_fields(~custom,~Logical,logical,activation,input,parallel,depth,workers,weights,bias,weight_evidence,bias_evidence,frame,source,weight_model,bias_model,fields,parallel_exact),
    DensePublic.start_bands_native_fields(~custom,~Logical,~original,logical,activation,input,parallel,depth,workers,weights,bias,weight_evidence,bias_evidence,frame,source,weight_model,bias_model,fill,normal,fields,parallel_exact,matched,values,weight_exact,bias_exact))

def start(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,+depth: U32,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.start_bands(~custom,activation,depth,parallel,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  +domain = NormalCaller.source_domain(~Logical,~original,logical,input,weights,bias,Shape.Both{DensePublic.normal_counted(Native.input_extents(input),source,fill,input_workers,normal),DensePublic.source_coherent(input,matched)},validated)
  +frame = PublicFrame.configured_native(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weights,bias,domain,weight_evidence,bias_evidence,validated,supported,eligible)
  start_fields(~custom,~Logical,~original,logical,activation,input,parallel,depth,weights,bias,weight_evidence,bias_evidence,frame,source,weight_model,bias_model,fill,input_workers,workers,normal,
    PublicFrame.configured(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weights,bias,domain,weight_evidence,bias_evidence,validated,supported,eligible),
    DensePublic.configured_parallel(~Logical,~original,logical,activation,input,parallel,weights,bias,domain,weight_evidence,bias_evidence,validated,supported,eligible),matched,values,weight_exact,bias_exact)

def depth(plan: BandPlanning.Plan) -> U32:
  BandPlanning.Plan{depth,parallel,time} = plan
  depth

def parallel(plan: BandPlanning.Plan) -> Bool:
  BandPlanning.Plan{depth,parallel,time} = plan
  parallel

type DepthKind<-plan: BandPlanning.Plan> is Data:
  ZeroDepth{equal: {depth(plan) == 0 : U32}}
  PositiveDepth{positive: {Nat.is_lt(0n,U32.to_nat(depth(plan))) == True{} : Bool}}

type DepthObservation<-plan: BandPlanning.Plan> is Data:
  DepthObservation{value: Nat,exact: {value == U32.to_nat(depth(plan)) : Nat}}

def observe_depth(+plan: BandPlanning.Plan) -> DepthObservation<plan>:
  DepthObservation{U32.to_nat(depth(plan)),{==}}

def classified_depth(+plan: BandPlanning.Plan,value: Nat,+exact: {value == U32.to_nat(depth(plan)) : Nat}) -> DepthKind<plan>:
  match value:
    case 0n:
      ZeroDepth{Equal.trans(U32,depth(plan),U32.from_nat(U32.to_nat(depth(plan))),0,
        Equal.sym(U32,U32.from_nat(U32.to_nat(depth(plan))),depth(plan),Arithmetic.native_roundtrip(depth(plan))),
        Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(depth(plan)),0n,Equal.sym(Nat,0n,U32.to_nat(depth(plan)),exact)))}
    case 1n+previous:
      PositiveDepth{
        %exact : {Nat.is_lt(0n,_) == True{} : Bool}
        {==}}

def observed_depth(+plan: BandPlanning.Plan,observation: DepthObservation<plan>) -> DepthKind<plan>:
  DepthObservation{value,exact} = observation
  classified_depth(plan,value,exact)

def classify(+plan: BandPlanning.Plan) -> DepthKind<plan>:
  observed_depth(plan,observe_depth(plan))

def nonzero_matches(~custom: F32 -> F32,+activation: Activations.Activation,+plan: BandPlanning.Plan,+geometry: TensorConvolution.Geometry,
  -source: Storage.Buffer<F32>,-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>,+workers: U32,
  positive: {Nat.is_lt(0n,U32.to_nat(depth(plan))) == True{} : Bool}) ->
  {TensorConvolution.initial_layout(~custom,activation,plan,geometry,source,TensorConvolution.packed(geometry,weights),bias,workers) == TensorConvolution.start_bands(~custom,activation,depth(plan),parallel(plan),geometry,source,TensorConvolution.packed(geometry,weights),bias,workers) : Tensors.Tensor<F32>}:
  BandPlanning.Plan{+depth,+parallel,time} = plan
  %Equal.sym(Bool,U32.is_eq(depth,0),False{},Flags.zero_false_from_positive(depth,positive)) :
    {TensorConvolution.initial_layout_selected(~custom,activation,_,depth,parallel,geometry,source,TensorConvolution.packed(geometry,weights),bias,workers) == TensorConvolution.start_bands(~custom,activation,depth,parallel,geometry,source,TensorConvolution.packed(geometry,weights),bias,workers) : Tensors.Tensor<F32>}
  {==}

def nonzero(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+plan: BandPlanning.Plan,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias_model) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>},
  positive: {Nat.is_lt(0n,U32.to_nat(depth(plan))) == True{} : Bool}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.initial_layout(~custom,activation,plan,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers)):
  %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.initial_layout(~custom,activation,plan,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers),
    TensorConvolution.start_bands(~custom,activation,depth(plan),parallel(plan),PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weight_model)),Buffer.pack(~F32,bias_model),workers),
    nonzero_matches(~custom,activation,plan,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers,positive)) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,DenseNativeContract.sources_context(~Logical,logical,activation,input,weights,bias),_)
  start(~custom,~Logical,~original,logical,activation,input,parallel(plan),depth(plan),weights,bias,weight_evidence,bias_evidence,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,validated,supported,eligible,values,weight_exact,bias_exact)
