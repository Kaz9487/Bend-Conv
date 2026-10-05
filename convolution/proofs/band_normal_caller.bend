# Normal ready CHW/NCHW owners supply the complete public shape envelope.
# Reuse the checked convolution caller and its existing leaf refinement.
import Base
import ../specification.bend as ConvSpec
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_window.bend as Window
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
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/boolean_logic.bend as Boolean
import ./convolution_guard_facts.bend as Guards
import ./band_caller_guards.bend as Caller
import ./band_leaf_caller.bend as Leaf
import ./band_native_source_domain.bend as BandNativeSourceDomain
import ./band_public_frame.bend as PublicFrame
import ./band_native_caller.bend as Native
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_tensor_execution.bend as TensorExecution
import ./band_output_metadata.bend as OutputMetadata
import ./band_output_normal.bend as BandOutputNormal

def source_well_formed(+input: PublicFrame.NativeInput,+source: Model.Model<F32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Ready{},workers})) ->
  BandShape.WellFormed<F32,PublicFrame.source_geometry(input),Model.content(~F32,source)>:
  Shape.Both{well_formed,shape} = normal
  %Model.packed_contents(~F32,source) : BandShape.WellFormed<F32,PublicFrame.source_geometry(input),_>
  well_formed

def source_shape(+input: PublicFrame.NativeInput,+source: Model.Model<F32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Ready{},workers})) ->
  Shape.Both<Global.Counted<Native.input_extents(input)>,{Window.band_shape(Native.input_extents(input),PublicFrame.source_geometry(input)) == True{} : Bool}>:
  Shape.Both{well_formed,shape} = normal
  shape

def source_coherent(+input: PublicFrame.NativeInput,+source: Model.Model<F32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Ready{},workers})) ->
  {Window.band_shape(Native.input_extents(input),PublicFrame.source_geometry(input)) == True{} : Bool}:
  Shape.Both{well_formed,Shape.Both{counted,coherent}} = normal
  coherent

def input_matched(+input: PublicFrame.NativeInput,
  coherent: {Window.band_shape(Native.input_extents(input),PublicFrame.source_geometry(input)) == True{} : Bool}) ->
  {Native.input_matched(input) == True{} : Bool}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  match batched:
    case True{}: {==}
    case False{}:
      Boolean.left(U32.is_eq(images,1),U32.is_eq(ci,ci),
        Boolean.left((U32.is_eq(images,1) && U32.is_eq(ci,ci) : Bool),U32.is_eq(h,h),
          Boolean.left((U32.is_eq(images,1) && U32.is_eq(ci,ci) && U32.is_eq(h,h) : Bool),U32.is_eq(w,w),coherent)))

# The existing convolution valid guard supplies positive C/H/W. The shape's
# actual Counted receipt supplies the whole-tensor bound, including batch N.
def counted_domain(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+counted: Global.Counted<shape>,
  coherent: {Window.band_shape(shape,Bands.Geometry{images,channels,height,width}) == True{} : Bool},
  positives: BandNativeSourceDomain.Positive<U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)>) ->
  BandGather.Domain<Bands.Geometry{images,channels,height,width},Default.limits(),Machine.capacity(Default.limits())>:
  BandNativeSourceDomain.Positive{cp,hp,wp} = positives
  Global.domain(shape,Bands.Geometry{images,channels,height,width},counted,coherent,cp,hp,wp)

def source_domain(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: PublicFrame.NativeInput,
  +weights: List<&2,F32>,+bias: List<&2,F32>,
  shape: Shape.Both<Global.Counted<Native.input_extents(input)>,{Window.band_shape(Native.input_extents(input),PublicFrame.source_geometry(input)) == True{} : Bool}>,
  validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool}) ->
  BandGather.Domain<Leaf.input_geometry(PublicFrame.input_images(input),PublicFrame.frame_dimensions(input)),Default.limits(),Machine.capacity(Default.limits())>:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  Shape.Both{+counted,coherent} = shape
  +dimensions = BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p)
  +problem = Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n)
  BandNativeSourceDomain.converted(Bands.Geometry{images,ci,h,w},Leaf.input_geometry(images,dimensions),Default.limits(),Machine.capacity(Default.limits()),
    Equal.sym(Bands.Geometry,Leaf.input_geometry(images,dimensions),Bands.Geometry{images,ci,h,w},BandNativeSourceDomain.geometry_roundtrip(images,ci,co,h,w,k,s,p)),
    counted_domain(Native.input_extents(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p}),images,ci,h,w,counted,coherent,
      BandNativeSourceDomain.positive(problem,Guards.valid(problem,Caller.valid(~Logical,~original,logical,dimensions,weights,bias,0n,validated)))))

def checked(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Ready{},workers}),
  values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,source)),
  weight_length: {Buffer.length(~F32,weights) == ConvSpec.length(weight_list) : Nat},bias_length: {Buffer.length(~F32,bias) == ConvSpec.length(bias_list) : Nat},
  weight_ready: Buffer.Ready<F32,weights>,bias_ready: Buffer.Ready<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  PublicFrame.Contract(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,weight_list,bias_list),
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,workers)):
  +domain = source_domain(~Logical,~original,logical,input,weight_list,bias_list,source_shape(input,source,fill,workers,normal),validated)
  Native.checked_fields(~custom,~Logical,~original,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,
    PublicFrame.configured_native(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported,eligible),
    source,weights,bias,Native.input_extents(input),fill,workers,
    PublicFrame.configured(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported,eligible),
    Native.configured_lengths(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported,eligible),
    source_well_formed(input,source,fill,workers,normal),values,weight_length,bias_length,weight_ready,bias_ready,weight_exact,bias_exact)

def conv2d(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+weight_workers: U32,+bias_workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),
  values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,source)),
  weight_length: {Buffer.length(~F32,weights) == ConvSpec.length(weight_list) : Nat},bias_length: {Buffer.length(~F32,bias) == ConvSpec.length(bias_list) : Nat},
  weight_ready: Buffer.Ready<F32,weights>,bias_ready: Buffer.Ready<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  PublicFrame.Contract(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,weight_list,bias_list),
    TensorConvolution.conv2d(~custom,activation,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Ready{},input_workers},
      Tensors.Tensor{Buffer.pack(~F32,weights),Native.weight_extents(input),0.0,Tensors.Ready{},weight_workers},
      Tensors.Tensor{Buffer.pack(~F32,bias),Native.bias_extents(input),0.0,Tensors.Ready{},bias_workers},Native.stride(input),Native.padding(input))):
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.conv2d(~custom,activation,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Ready{},input_workers},
      Tensors.Tensor{Buffer.pack(~F32,weights),Native.weight_extents(input),0.0,Tensors.Ready{},weight_workers},
      Tensors.Tensor{Buffer.pack(~F32,bias),Native.bias_extents(input),0.0,Tensors.Ready{},bias_workers},Native.stride(input),Native.padding(input)),
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers))),
    Native.conv2d_matches(~custom,activation,input,source,weights,bias,fill,input_workers,weight_workers,bias_workers,input_matched(input,source_coherent(input,source,fill,input_workers,normal)))) :
    PublicFrame.Contract(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,weight_list,bias_list),_)
  checked(~custom,~Logical,~original,logical,activation,input,parallel,weight_list,bias_list,weight_evidence,bias_evidence,validated,supported,eligible,
    source,weights,bias,fill,U32.max(input_workers,U32.max(weight_workers,bias_workers)),normal,values,weight_length,bias_length,weight_ready,bias_ready,weight_exact,bias_exact)

def frame_output_images(~Logical: Data,+frame: PipelineExecution.Frame<Logical>) ->
  {BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)) == PublicFrame.frame_images(~Logical,frame) : U32}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  {==}

def native_output_match(+input: PublicFrame.NativeInput) ->
  {BandOutputNormal.shape_matched(PublicFrame.input_images(input),Native.batched(input)) == Native.input_matched(input) : Bool}:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  match batched:
    case True{}: {==}
    case False{}: {==}

def output_matched(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+input: PublicFrame.NativeInput,
  images_exact: {PublicFrame.frame_images(~Logical,frame) == PublicFrame.input_images(input) : U32},
  matched: {Native.input_matched(input) == True{} : Bool}) ->
  {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),Native.batched(input)) == True{} : Bool}:
  %Equal.sym(U32,BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),PublicFrame.frame_images(~Logical,frame),frame_output_images(~Logical,frame)) :
    {BandOutputNormal.shape_matched(_,Native.batched(input)) == True{} : Bool}
  %Equal.sym(U32,PublicFrame.frame_images(~Logical,frame),PublicFrame.input_images(input),images_exact) :
    {BandOutputNormal.shape_matched(_,Native.batched(input)) == True{} : Bool}
  %Equal.sym(Bool,BandOutputNormal.shape_matched(PublicFrame.input_images(input),Native.batched(input)),Native.input_matched(input),native_output_match(input)) : {_ == True{} : Bool}
  matched

# The same factory fields now connect the stronger output representation
# capability. Values still come from the existing convolution leaf theorem.
def checked_normal_fields(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +fields: PublicFrame.Fields<Logical,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits())>,
  lengths: Shape.Both<{PublicFrame.weight_length(~Logical,frame) == ConvSpec.length(weight_list) : Nat},{PublicFrame.bias_length(~Logical,frame) == ConvSpec.length(bias_list) : Nat}>,
  matched: {Native.input_matched(input) == True{} : Bool},
  source_shape: BandShape.WellFormed<F32,PublicFrame.source_geometry(input),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,source)),
  weight_length: {Buffer.length(~F32,weights) == ConvSpec.length(weight_list) : Nat},bias_length: {Buffer.length(~F32,bias) == ConvSpec.length(bias_list) : Nat},
  weight_ready: Buffer.Ready<F32,weights>,bias_ready: Buffer.Ready<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  Global.Normal(~F32,TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers)):
  PublicFrame.Fields{+logical_exact,+activation_exact,images_exact,dimensions_exact,+input_geometry_exact,+weights_exact,bias_content_exact,context_exact,+limits_exact,capacity_exact} = fields
  Shape.Both{weight_length_exact,bias_length_exact} = lengths
  +geometry_exact = Native.geometry_fields(~Logical,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits()),fields)
  +bias_settings = PublicFrame.settings_bias(~Logical,~original,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits()),fields)
  +output_match = output_matched(~Logical,frame,input,images_exact,matched)
  +framed_shape = {
    %Equal.sym(Bands.Geometry,PipelineExecution.input_geometry(~Logical,frame),PublicFrame.source_geometry(input),input_geometry_exact) : BandShape.WellFormed<F32,_,Model.content(~F32,source)>
    source_shape : BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>}
  +framed_values = {
    %Equal.sym(Logical,PipelineExecution.convolution_logical(~Logical,frame),logical,logical_exact) : BandValues.Values(~F32,~Logical,~original,_,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))
    %Equal.sym(Bands.Geometry,PipelineExecution.input_geometry(~Logical,frame),PublicFrame.source_geometry(input),input_geometry_exact) : BandValues.Values(~F32,~Logical,~original,logical,_,Model.content(~F32,source))
    values : BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))}
  %activation_exact : Global.Normal(~F32,TensorConvolution.band_checked(~custom,_,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  %geometry_exact : Global.Normal(~F32,TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{_},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  %input_geometry_exact : Global.Normal(~F32,TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{TensorExecution.execution_geometry(~Logical,frame,Native.batched(input))},_,Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  BandOutputNormal.checked_normal(~custom,~Logical,~original,frame,Native.batched(input),source,weights,bias,shape,fill,workers,output_match,limits_exact,framed_shape,framed_values,
    Equal.trans(Nat,Buffer.length(~F32,weights),ConvSpec.length(weight_list),PublicFrame.weight_length(~Logical,frame),weight_length,Equal.sym(Nat,PublicFrame.weight_length(~Logical,frame),ConvSpec.length(weight_list),weight_length_exact)),
    Equal.trans(Nat,Buffer.length(~F32,bias),ConvSpec.length(bias_list),PublicFrame.bias_length(~Logical,frame),bias_length,Equal.sym(Nat,PublicFrame.bias_length(~Logical,frame),ConvSpec.length(bias_list),bias_length_exact)),
    weight_ready,bias_ready,
    Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,weights),StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence),PipelineExecution.source_tree(~Logical,frame),weight_exact,
      Equal.sym(Tree.Tree<F32>,PipelineExecution.source_tree(~Logical,frame),StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence),weights_exact)),
    Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,bias),Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)},
      PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),bias_exact,
      Equal.sym(Buffer.Content<F32>,PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)},bias_settings)))

def checked_normal(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Ready{},workers}),
  values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,source)),
  weight_length: {Buffer.length(~F32,weights) == ConvSpec.length(weight_list) : Nat},bias_length: {Buffer.length(~F32,bias) == ConvSpec.length(bias_list) : Nat},
  weight_ready: Buffer.Ready<F32,weights>,bias_ready: Buffer.Ready<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  Global.Normal(~F32,TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,workers)):
  +domain = source_domain(~Logical,~original,logical,input,weight_list,bias_list,source_shape(input,source,fill,workers,normal),validated)
  checked_normal_fields(~custom,~Logical,~original,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,
    PublicFrame.configured_native(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported,eligible),
    source,weights,bias,Native.input_extents(input),fill,workers,
    PublicFrame.configured(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported,eligible),
    Native.configured_lengths(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported,eligible),
    input_matched(input,source_coherent(input,source,fill,workers,normal)),source_well_formed(input,source,fill,workers,normal),values,weight_length,bias_length,weight_ready,bias_ready,weight_exact,bias_exact)

def conv2d_normal(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+weight_workers: U32,+bias_workers: U32,
  +normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),
  values: BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Model.content(~F32,source)),
  weight_length: {Buffer.length(~F32,weights) == ConvSpec.length(weight_list) : Nat},bias_length: {Buffer.length(~F32,bias) == ConvSpec.length(bias_list) : Nat},
  weight_ready: Buffer.Ready<F32,weights>,bias_ready: Buffer.Ready<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  Global.Normal(~F32,TensorConvolution.conv2d(~custom,activation,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Ready{},input_workers},
    Tensors.Tensor{Buffer.pack(~F32,weights),Native.weight_extents(input),0.0,Tensors.Ready{},weight_workers},
    Tensors.Tensor{Buffer.pack(~F32,bias),Native.bias_extents(input),0.0,Tensors.Ready{},bias_workers},Native.stride(input),Native.padding(input))):
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.conv2d(~custom,activation,Tensors.Banded{Model.pack(~F32,source),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Ready{},input_workers},
      Tensors.Tensor{Buffer.pack(~F32,weights),Native.weight_extents(input),0.0,Tensors.Ready{},weight_workers},
      Tensors.Tensor{Buffer.pack(~F32,bias),Native.bias_extents(input),0.0,Tensors.Ready{},bias_workers},Native.stride(input),Native.padding(input)),
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},PublicFrame.source_geometry(input),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,U32.max(input_workers,U32.max(weight_workers,bias_workers))),
    Native.conv2d_matches(~custom,activation,input,source,weights,bias,fill,input_workers,weight_workers,bias_workers,input_matched(input,source_coherent(input,source,fill,input_workers,normal)))) : Global.Normal(~F32,_)
  checked_normal(~custom,~Logical,~original,logical,activation,input,parallel,weight_list,bias_list,weight_evidence,bias_evidence,validated,supported,eligible,
    source,weights,bias,fill,U32.max(input_workers,U32.max(weight_workers,bias_workers)),normal,values,weight_length,bias_length,weight_ready,bias_ready,weight_exact,bias_exact)
