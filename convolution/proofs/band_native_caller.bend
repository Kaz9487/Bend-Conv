# The raw native factory supplies the frame used by the checked caller.
# Source values and owners are incoming contracts; plans, inspection success,
# alignment and leaf scheduling are discharged by the existing caller chain.
import Base
import ../specification.bend as ConvSpec
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/math_activation.bend as Activations
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_count.bend as BandCount
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ./band_leaf_caller.bend as Leaf
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_tensor_execution.bend as TensorExecution
import ./band_public_frame.bend as PublicFrame
import ./band_machine_caller.bend as MachineCaller

def batched(input: PublicFrame.NativeInput) -> Bool:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  batched

def input_extents(input: PublicFrame.NativeInput) -> List<&2,U32>:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  TensorConvolution.output_shape(batched,images,ci,h,w)

def weight_extents(input: PublicFrame.NativeInput) -> List<&2,U32>:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,+k,s,p} = input
  [co,ci,k,k]

def bias_extents(input: PublicFrame.NativeInput) -> List<&2,U32>:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  [co]

def stride(input: PublicFrame.NativeInput) -> U32:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  s

def padding(input: PublicFrame.NativeInput) -> U32:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  p

def input_matched(input: PublicFrame.NativeInput) -> Bool:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  match batched:
    case True{}: True{}
    case False{}: U32.is_eq(images,1)

def input_shape_value(input: PublicFrame.NativeInput) -> TensorConvolution.InputShape:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  TensorConvolution.InputShape{images,batched,ci,h,w}

def input_shape_matches(+input: PublicFrame.NativeInput,matched: {input_matched(input) == True{} : Bool}) ->
  {TensorConvolution.input_shape(input_extents(input)) == Some{input_shape_value(input)} : Maybe<&2,TensorConvolution.InputShape>}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  match batched:
    case True{}: {==}
    case False{}:
      %Equal.sym(U32,images,1,ShapeLaws.native_equal(images,1,matched)) :
        {Some{TensorConvolution.InputShape{1,False{},ci,h,w}} == Some{TensorConvolution.InputShape{_,False{},ci,h,w}} : Maybe<&2,TensorConvolution.InputShape>}
      {==}

def parameter_geometry_matches(+input: PublicFrame.NativeInput) ->
  {TensorConvolution.geometry(Some{input_shape_value(input)},weight_extents(input),bias_extents(input),stride(input),padding(input)) == Some{PublicFrame.geometry(input)} : Maybe<&2,TensorConvolution.Geometry>}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  %Equal.sym(Bool,U32.is_eq(ci,ci),True{},BandShape.equal_words(ci,ci,{==})) :
    {TensorConvolution.matched(_ && U32.is_eq(k,k) && U32.is_eq(co,co),TensorConvolution.Geometry{images,batched,PublicFrame.frame_dimensions(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})}) ==
      Some{PublicFrame.geometry(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})} : Maybe<&2,TensorConvolution.Geometry>}
  %Equal.sym(Bool,U32.is_eq(k,k),True{},BandShape.equal_words(k,k,{==})) :
    {TensorConvolution.matched(True{} && _ && U32.is_eq(co,co),TensorConvolution.Geometry{images,batched,PublicFrame.frame_dimensions(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})}) ==
      Some{PublicFrame.geometry(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})} : Maybe<&2,TensorConvolution.Geometry>}
  %Equal.sym(Bool,U32.is_eq(co,co),True{},BandShape.equal_words(co,co,{==})) :
    {TensorConvolution.matched(True{} && True{} && _,TensorConvolution.Geometry{images,batched,PublicFrame.frame_dimensions(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})}) ==
      Some{PublicFrame.geometry(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})} : Maybe<&2,TensorConvolution.Geometry>}
  {==}

def public_geometry(+input: PublicFrame.NativeInput,matched: {input_matched(input) == True{} : Bool}) ->
  {TensorConvolution.geometry(TensorConvolution.input_shape(input_extents(input)),weight_extents(input),bias_extents(input),stride(input),padding(input)) == Some{PublicFrame.geometry(input)} : Maybe<&2,TensorConvolution.Geometry>}:
  %Equal.sym(Maybe<&2,TensorConvolution.InputShape>,TensorConvolution.input_shape(input_extents(input)),Some{input_shape_value(input)},input_shape_matches(input,matched)) :
    {TensorConvolution.geometry(_,weight_extents(input),bias_extents(input),stride(input),padding(input)) == Some{PublicFrame.geometry(input)} : Maybe<&2,TensorConvolution.Geometry>}
  parameter_geometry_matches(input)

def geometry_parts(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,batched: Bool) ->
  {TensorExecution.execution_geometry(~Logical,frame,batched) == TensorConvolution.Geometry{PublicFrame.frame_images(~Logical,frame),batched,PipelineExecution.dimensions(~Logical,frame)} : TensorConvolution.Geometry}:
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  {==}

def input_geometry_parts(+input: PublicFrame.NativeInput) ->
  {PublicFrame.geometry(input) == TensorConvolution.Geometry{PublicFrame.input_images(input),batched(input),PublicFrame.frame_dimensions(input)} : TensorConvolution.Geometry}:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  {==}

def geometry_fields(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,+frame: PipelineExecution.Frame<Logical>,+limits: Machine.Limits,+capacity: U32,
  fields: PublicFrame.Fields<Logical,logical,activation,input,weights,bias,weight_evidence,bias_evidence,frame,limits,capacity>) ->
  {TensorExecution.execution_geometry(~Logical,frame,batched(input)) == PublicFrame.geometry(input) : TensorConvolution.Geometry}:
  PublicFrame.Fields{logical_exact,activation_exact,+images_exact,+dimensions_exact,input_geometry_exact,weights_exact,bias_exact,context_exact,limits_exact,capacity_exact} = fields
  Equal.trans(TensorConvolution.Geometry,TensorExecution.execution_geometry(~Logical,frame,batched(input)),
    TensorConvolution.Geometry{PublicFrame.frame_images(~Logical,frame),batched(input),PipelineExecution.dimensions(~Logical,frame)},PublicFrame.geometry(input),
    geometry_parts(~Logical,frame,batched(input)),
    %Equal.sym(U32,PublicFrame.frame_images(~Logical,frame),PublicFrame.input_images(input),images_exact) :
      {TensorConvolution.Geometry{_,batched(input),PipelineExecution.dimensions(~Logical,frame)} == PublicFrame.geometry(input) : TensorConvolution.Geometry}
    %Equal.sym(KernelShape.Dimensions,PipelineExecution.dimensions(~Logical,frame),PublicFrame.frame_dimensions(input),dimensions_exact) :
      {TensorConvolution.Geometry{PublicFrame.input_images(input),batched(input),_} == PublicFrame.geometry(input) : TensorConvolution.Geometry}
    Equal.sym(TensorConvolution.Geometry,PublicFrame.geometry(input),TensorConvolution.Geometry{PublicFrame.input_images(input),batched(input),PublicFrame.frame_dimensions(input)},input_geometry_parts(input)))

# Keep limits generic when exposing the factory's parameter sequences.
def configured_lengths(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,+limits: Machine.Limits,+capacity: U32,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+domain: BandGather.Domain<Leaf.input_geometry(PublicFrame.input_images(input),PublicFrame.frame_dimensions(input)),limits,capacity>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,limits) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool}) ->
  Shape.Both<
    {PublicFrame.weight_length(~Logical,PublicFrame.configured_native(~Logical,~original,logical,activation,input,parallel,limits,capacity,weights,bias,domain,weight_evidence,bias_evidence,validated,supported,eligible)) == ConvSpec.length(weights) : Nat},
    {PublicFrame.bias_length(~Logical,PublicFrame.configured_native(~Logical,~original,logical,activation,input,parallel,limits,capacity,weights,bias,domain,weight_evidence,bias_evidence,validated,supported,eligible)) == ConvSpec.length(bias) : Nat}>:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  Shape.Both{{==},{==}}

def accepted_lengths(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,+weights: List<&2,F32>,+bias: List<&2,F32>,+count: U32,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  +counted: {Tensors.count_of(BandCount.extents(PublicFrame.source_geometry(input))) == Some{count} : Maybe<&2,U32>}) ->
  Shape.Both<
    {PublicFrame.weight_length(~Logical,PublicFrame.frame_accepted(~Logical,~original,logical,activation,input,parallel,weights,bias,count,weight_evidence,bias_evidence,validated,supported,eligible,counted)) == ConvSpec.length(weights) : Nat},
    {PublicFrame.bias_length(~Logical,PublicFrame.frame_accepted(~Logical,~original,logical,activation,input,parallel,weights,bias,count,weight_evidence,bias_evidence,validated,supported,eligible,counted)) == ConvSpec.length(bias) : Nat}>:
  configured_lengths(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weights,bias,
    PublicFrame.source_domain(~Logical,~original,logical,input,weights,bias,count,validated,counted),weight_evidence,bias_evidence,validated,supported,eligible)

def checked_fields(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +fields: PublicFrame.Fields<Logical,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits())>,
  lengths: Shape.Both<{PublicFrame.weight_length(~Logical,frame) == ConvSpec.length(weight_list) : Nat},{PublicFrame.bias_length(~Logical,frame) == ConvSpec.length(bias_list) : Nat}>,
  source_shape: BandShape.WellFormed<F32,PublicFrame.source_geometry(input),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,source)),
  weight_length: {Buffer.length(~F32,weights) == ConvSpec.length(weight_list) : Nat},bias_length: {Buffer.length(~F32,bias) == ConvSpec.length(bias_list) : Nat},
  weight_ready: Buffer.Ready<F32,weights>,bias_ready: Buffer.Ready<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  PublicFrame.Contract(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,weight_list,bias_list),
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers)):
  PublicFrame.Fields{+logical_exact,+activation_exact,images_exact,dimensions_exact,+input_geometry_exact,+weights_exact,bias_content_exact,+context_exact,+limits_exact,capacity_exact} = fields
  Shape.Both{weight_length_exact,bias_length_exact} = lengths
  +geometry_exact = geometry_fields(~Logical,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits()),fields)
  +bias_settings = PublicFrame.settings_bias(~Logical,~original,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits()),fields)
  +framed_shape = {
    %Equal.sym(Bands.Geometry,PipelineExecution.input_geometry(~Logical,frame),PublicFrame.source_geometry(input),input_geometry_exact) : BandShape.WellFormed<F32,_,Model.content(~F32,source)>
    source_shape : BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>}
  +framed_values = {
    %Equal.sym(Logical,PipelineExecution.convolution_logical(~Logical,frame),logical,logical_exact) : BandValues.Values(~F32,~Logical,~original,_,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))
    %Equal.sym(Bands.Geometry,PipelineExecution.input_geometry(~Logical,frame),PublicFrame.source_geometry(input),input_geometry_exact) : BandValues.Values(~F32,~Logical,~original,logical,_,Model.content(~F32,source))
    values : BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))}
  %context_exact : PublicFrame.Contract(~custom,~Logical,~original,_,TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  %activation_exact : PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),TensorConvolution.band_checked(~custom,_,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  %geometry_exact : PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{_},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  %input_geometry_exact : PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{TensorExecution.execution_geometry(~Logical,frame,batched(input))},_,Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  MachineCaller.checked(~custom,~Logical,~original,frame,batched(input),source,weights,bias,shape,fill,workers,limits_exact,framed_shape,framed_values,
    Equal.trans(Nat,Buffer.length(~F32,weights),ConvSpec.length(weight_list),PublicFrame.weight_length(~Logical,frame),weight_length,Equal.sym(Nat,PublicFrame.weight_length(~Logical,frame),ConvSpec.length(weight_list),weight_length_exact)),
    Equal.trans(Nat,Buffer.length(~F32,bias),ConvSpec.length(bias_list),PublicFrame.bias_length(~Logical,frame),bias_length,Equal.sym(Nat,PublicFrame.bias_length(~Logical,frame),ConvSpec.length(bias_list),bias_length_exact)),
    weight_ready,bias_ready,
    Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,weights),StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence),PipelineExecution.source_tree(~Logical,frame),weight_exact,
      Equal.sym(Tree.Tree<F32>,PipelineExecution.source_tree(~Logical,frame),StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence),weights_exact)),
    Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,bias),Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)},
      PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),bias_exact,
      Equal.sym(Buffer.Content<F32>,PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)},bias_settings)))

def checked(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +count: U32,+validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},+counted: {Tensors.count_of(BandCount.extents(PublicFrame.source_geometry(input))) == Some{count} : Maybe<&2,U32>},
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  source_shape: BandShape.WellFormed<F32,PublicFrame.source_geometry(input),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,source)),
  weight_length: {Buffer.length(~F32,weights) == ConvSpec.length(weight_list) : Nat},bias_length: {Buffer.length(~F32,bias) == ConvSpec.length(bias_list) : Nat},
  weight_ready: Buffer.Ready<F32,weights>,bias_ready: Buffer.Ready<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  PublicFrame.Contract(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,weight_list,bias_list),
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers)):
  checked_fields(~custom,~Logical,~original,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,
    PublicFrame.frame_accepted(~Logical,~original,logical,activation,input,parallel,weight_list,bias_list,count,weight_evidence,bias_evidence,validated,supported,eligible,counted),
    source,weights,bias,shape,fill,workers,
    PublicFrame.properties_accepted(~Logical,~original,logical,activation,input,parallel,weight_list,bias_list,count,weight_evidence,bias_evidence,validated,supported,eligible,counted),
    accepted_lengths(~Logical,~original,logical,activation,input,parallel,weight_list,bias_list,count,weight_evidence,bias_evidence,validated,supported,eligible,counted),
    source_shape,values,weight_length,bias_length,weight_ready,bias_ready,weight_exact,bias_exact)

# Dense parameter owners are consumed by the real with_dense callbacks. The
# band input and its incoming worker count remain the actual public owners.
def conv2d_matches(~custom: F32 -> F32,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +fill: F32,+input_workers: U32,+weight_workers: U32,+bias_workers: U32,matched: {input_matched(input) == True{} : Bool}) ->
  {TensorConvolution.conv2d(~custom,activation,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),input_extents(input),fill,Tensors.Ready{},input_workers},
      Tensors.Tensor{Buffer.pack(~F32,weights),weight_extents(input),0.0,Tensors.Ready{},weight_workers},Tensors.Tensor{Buffer.pack(~F32,bias),bias_extents(input),0.0,Tensors.Ready{},bias_workers},stride(input),padding(input)) ==
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers))) : Tensors.Tensor<F32>}:
  %Equal.sym(Maybe<&2,TensorConvolution.Geometry>,
    TensorConvolution.geometry(TensorConvolution.input_shape(input_extents(input)),weight_extents(input),bias_extents(input),stride(input),padding(input)),Some{PublicFrame.geometry(input)},public_geometry(input,matched)) :
    {TensorConvolution.band_checked(~custom,activation,_,PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers))) ==
      TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers))) : Tensors.Tensor<F32>}
  {==}

def conv2d(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +count: U32,+validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},+counted: {Tensors.count_of(BandCount.extents(PublicFrame.source_geometry(input))) == Some{count} : Maybe<&2,U32>},
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+weight_workers: U32,+bias_workers: U32,
  matched: {input_matched(input) == True{} : Bool},
  source_shape: BandShape.WellFormed<F32,PublicFrame.source_geometry(input),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,source)),
  weight_length: {Buffer.length(~F32,weights) == ConvSpec.length(weight_list) : Nat},bias_length: {Buffer.length(~F32,bias) == ConvSpec.length(bias_list) : Nat},
  weight_ready: Buffer.Ready<F32,weights>,bias_ready: Buffer.Ready<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  PublicFrame.Contract(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,weight_list,bias_list),
    TensorConvolution.conv2d(~custom,activation,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),input_extents(input),fill,Tensors.Ready{},input_workers},
      Tensors.Tensor{Buffer.pack(~F32,weights),weight_extents(input),0.0,Tensors.Ready{},weight_workers},Tensors.Tensor{Buffer.pack(~F32,bias),bias_extents(input),0.0,Tensors.Ready{},bias_workers},stride(input),padding(input))):
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.conv2d(~custom,activation,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),input_extents(input),fill,Tensors.Ready{},input_workers},
      Tensors.Tensor{Buffer.pack(~F32,weights),weight_extents(input),0.0,Tensors.Ready{},weight_workers},Tensors.Tensor{Buffer.pack(~F32,bias),bias_extents(input),0.0,Tensors.Ready{},bias_workers},stride(input),padding(input)),
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers))),
    conv2d_matches(~custom,activation,input,source,weights,bias,fill,input_workers,weight_workers,bias_workers,matched)) :
    PublicFrame.Contract(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,weight_list,bias_list),_)
  checked(~custom,~Logical,~original,logical,activation,input,parallel,weight_list,bias_list,weight_evidence,bias_evidence,count,validated,supported,eligible,counted,
    source,weights,bias,input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers)),source_shape,values,weight_length,bias_length,weight_ready,bias_ready,weight_exact,bias_exact)
