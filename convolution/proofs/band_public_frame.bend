# The public convolution frame: inspection, checked guards and the reference.
import Base
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_tensor_execution.bend as TensorExecution
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./band_leaf_caller.bend as Leaf
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ./band_caller_guards.bend as Caller
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_leaf_values.bend as LeafValues
import ./band_native_source_domain.bend as BandNativeSourceDomain
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_count.bend as BandCount
import ../../lib/proofs/tensor_band_inspection.bend as Inspection
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ./band_frame_configuration.bend as Configuration
import ../../lib/kernel_band_planning.bend as BandPlanning
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/tensor_band_map.bend as BandMap

# Public native dimensions and actual validation/support/count guards construct
# the convolution frame. The input record keeps the complete machine predicate
# opaque until its generic machine-limits consumer inspects the dimensions.
type NativeInput is Data:
  NativeInput{input_images: U32,batched: Bool,input_channels: U32,output_channels: U32,input_height: U32,input_width: U32,kernel: U32,stride: U32,padding: U32}

def input_images(input: NativeInput) -> U32:
  NativeInput{input_images,batched,ci,co,h,w,k,s,p} = input
  input_images

def frame_dimensions(input: NativeInput) -> KernelShape.Dimensions:
  NativeInput{input_images,batched,ci,co,h,w,k,s,p} = input
  BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p)

def geometry(input: NativeInput) -> TensorConvolution.Geometry:
  NativeInput{input_images,batched,ci,co,h,w,k,s,p} = input
  TensorConvolution.Geometry{input_images,batched,BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p)}

def source_geometry(input: NativeInput) -> Bands.Geometry:
  NativeInput{input_images,batched,ci,co,h,w,k,s,p} = input
  Bands.Geometry{input_images,ci,h,w}

def frame_valid(input: NativeInput,weights: List<&2,F32>,bias: List<&2,F32>) -> Bool:
  NativeInput{input_images,batched,ci,co,h,w,k,s,p} = input
  +shape = BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p)
  KernelShape.valid(shape,Caller.input_length(shape),ConvSpec.length(weights),ConvSpec.length(bias))

def machine(input: NativeInput,+limits: Machine.Limits) -> Bool:
  NativeInput{+input_images,batched,ci,co,h,w,k,s,p} = input
  +shape = BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p)
  KernelShape.supports_packed(limits,shape,Caller.input_length(shape),Configuration.weight_size(shape)) &&
    Nat.is_le(LeafValues.output_elements(input_images,shape),U32.to_nat(Machine.capacity(limits)))

def source_domain(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+count: U32,
  frame_accepted: {frame_valid(input,weights,bias) == True{} : Bool},
  counted: {Tensors.count_of(BandCount.extents(source_geometry(input))) == Some{count} : Maybe<&2,U32>}) ->
  BandGather.Domain<Leaf.input_geometry(input_images(input),frame_dimensions(input)),Default.limits(),Machine.capacity(Default.limits())>:
  NativeInput{+input_images,batched,+ci,+co,+h,+w,+k,+s,+p} = input
  BandNativeSourceDomain.source_domain(~Logical,~original,logical,input_images,ci,co,h,w,k,s,p,weights,bias,count,frame_accepted,counted)

def configured_native(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: NativeInput,+parallel: Bool,+limits: Machine.Limits,+capacity: U32,
  +weights: List<&2,F32>,+bias: List<&2,F32>,domain: BandGather.Domain<Leaf.input_geometry(input_images(input),frame_dimensions(input)),limits,capacity>,
  weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  frame_accepted: {frame_valid(input,weights,bias) == True{} : Bool},supported: {machine(input,limits) == True{} : Bool},
  eligible: {TensorConvolution.uses_band_windows(geometry(input)) == True{} : Bool}) -> PipelineExecution.Frame<Logical>:
  NativeInput{+input_images,batched,+ci,+co,+h,+w,+k,+s,+p} = input
  Configuration.configured(~Logical,~original,logical,activation,input_images,BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p),parallel,limits,capacity,weights,bias,
    domain,weight_evidence,bias_evidence,frame_accepted,supported,eligible)

def frame_accepted(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: NativeInput,+parallel: Bool,+weights: List<&2,F32>,+bias: List<&2,F32>,+count: U32,
  weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +validated: {frame_valid(input,weights,bias) == True{} : Bool},supported: {machine(input,Default.limits()) == True{} : Bool},
  eligible: {TensorConvolution.uses_band_windows(geometry(input)) == True{} : Bool},
  counted: {Tensors.count_of(BandCount.extents(source_geometry(input))) == Some{count} : Maybe<&2,U32>}) -> PipelineExecution.Frame<Logical>:
  configured_native(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weights,bias,
    source_domain(~Logical,~original,logical,input,weights,bias,count,validated,counted),weight_evidence,bias_evidence,validated,supported,eligible)

# Public frame construction retains the incoming logical context and parameter
# sequences. Inspect native dimensions only inside the generic configuration
# proof; the actual Default caller keeps the input record opaque.
def frame_images(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> U32:
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  images

def bias_content(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Buffer.Content<F32>:
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,be),ConvSpec.length(bias)}

def reference_context(~Logical: Data,logical: Logical,activation: Activations.Activation,input: NativeInput,weights: List<&2,F32>,bias: List<&2,F32>) -> BandConvolutionValues.Context<Logical>:
  NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  BandConvolutionValues.Context{logical,activation,U32.to_nat(ci),U32.to_nat(co),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p),weights,bias}

type Fields<-Logical: Data,-logical: Logical,-activation: Activations.Activation,-input: NativeInput,-weights: List<&2,F32>,-bias: List<&2,F32>,
  -weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,-bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,-frame: PipelineExecution.Frame<Logical>,-limits: Machine.Limits,-capacity: U32> is Data:
  Fields{logical_exact: {PipelineExecution.convolution_logical(~Logical,frame) == logical : Logical},
    activation_exact: {TensorExecution.execution_activation(~Logical,frame) == activation : Activations.Activation},
    images_exact: {frame_images(~Logical,frame) == input_images(input) : U32},
    dimensions_exact: {PipelineExecution.dimensions(~Logical,frame) == frame_dimensions(input) : KernelShape.Dimensions},
    input_geometry_exact: {PipelineExecution.input_geometry(~Logical,frame) == source_geometry(input) : Bands.Geometry},
    weights_exact: {PipelineExecution.source_tree(~Logical,frame) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
    bias_exact: {bias_content(~Logical,frame) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>},
    context_exact: {PipelineExecution.output_context(~Logical,frame) == reference_context(~Logical,logical,activation,input,weights,bias) : BandConvolutionValues.Context<Logical>},
    limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == limits : Machine.Limits},
    capacity_exact: {PipelineExecution.convolution_capacity(~Logical,frame) == capacity : U32}}

def configured(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: NativeInput,+parallel: Bool,+limits: Machine.Limits,+capacity: U32,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+domain: BandGather.Domain<Leaf.input_geometry(input_images(input),frame_dimensions(input)),limits,capacity>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +validated: {frame_valid(input,weights,bias) == True{} : Bool},+supported: {machine(input,limits) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(geometry(input)) == True{} : Bool}) ->
  Fields<Logical,logical,activation,input,weights,bias,weight_evidence,bias_evidence,
    configured_native(~Logical,~original,logical,activation,input,parallel,limits,capacity,weights,bias,domain,weight_evidence,bias_evidence,validated,supported,eligible),limits,capacity>:
  NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  Fields{{==},{==},{==},{==},BandNativeSourceDomain.geometry_roundtrip(images,ci,co,h,w,k,s,p),{==},{==},{==},{==},{==}}

def properties_accepted(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: NativeInput,+parallel: Bool,+weights: List<&2,F32>,+bias: List<&2,F32>,+count: U32,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +validated: {frame_valid(input,weights,bias) == True{} : Bool},+supported: {machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(geometry(input)) == True{} : Bool},
  +counted: {Tensors.count_of(BandCount.extents(source_geometry(input))) == Some{count} : Maybe<&2,U32>}) ->
  Fields<Logical,logical,activation,input,weights,bias,weight_evidence,bias_evidence,
    frame_accepted(~Logical,~original,logical,activation,input,parallel,weights,bias,count,weight_evidence,bias_evidence,validated,supported,eligible,counted),Default.limits(),Machine.capacity(Default.limits())>:
  configured(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weights,bias,
    source_domain(~Logical,~original,logical,input,weights,bias,count,validated,counted),weight_evidence,bias_evidence,validated,supported,eligible)

def settings_bias_matches(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>) ->
  {PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) == bias_content(~Logical,frame) : Buffer.Content<F32>}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  {==}

def settings_bias(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,+frame: PipelineExecution.Frame<Logical>,+limits: Machine.Limits,+capacity: U32,
  fields: Fields<Logical,logical,activation,input,weights,bias,weight_evidence,bias_evidence,frame,limits,capacity>) ->
  {PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>}:
  Fields{logical_exact,activation_exact,images_exact,dimensions_exact,input_geometry_exact,weights_exact,bias_exact,context_exact,limits_exact,capacity_exact} = fields
  Equal.trans(Buffer.Content<F32>,PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),bias_content(~Logical,frame),
    Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)},settings_bias_matches(~Logical,~original,frame),bias_exact)

# Actual inspection acceptance follows from whole-source metadata, readiness,
# coverage and the public machine domain. No inspection-success oracle is used.
def inspection_accepted(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+inspection_source: Model.Model<Element>,
  domain: BandGather.Domain<geometry,limits,capacity>,shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,inspection_source)>) ->
  {Inspection.valid(Inspection.model(~Element,geometry,inspection_source)) && U32.is_zero(Inspection.first(Inspection.model(~Element,geometry,inspection_source))) &&
    Nat.is_eq(U32.to_nat(Inspection.end(Inspection.model(~Element,geometry,inspection_source))),Inspection.total_rows(geometry)) == True{} : Bool}:
  BandShape.well_formed_checked(~Element,geometry,inspection_source,Machine.capacity(limits),Limits.capacity_fits(limits),BandGather.rows_room(geometry,limits,capacity,domain),shape)

def inspection_source(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+bands: Model.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,bands)>) ->
  {Inspection.valid(Inspection.model(~F32,PipelineExecution.input_geometry(~Logical,frame),bands)) && U32.is_zero(Inspection.first(Inspection.model(~F32,PipelineExecution.input_geometry(~Logical,frame),bands))) &&
    Nat.is_eq(U32.to_nat(Inspection.end(Inspection.model(~F32,PipelineExecution.input_geometry(~Logical,frame),bands))),Inspection.total_rows(PipelineExecution.input_geometry(~Logical,frame))) == True{} : Bool}:
  inspection_accepted(~F32,PipelineExecution.input_geometry(~Logical,frame),PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame),bands,PipelineExecution.convolution_domain(~Logical,frame),shape)

# The inspected public caller checks the source interval, native geometry and
# parameter allocation. These terms produce its actual Boolean conjunction.
def Words(dimensions: KernelShape.Dimensions) -> Data:
  KernelShape.Dimensions{ci,co,h,w,k,s,p} = dimensions
  LeafValues.Words<ci,co,h,w,k,s,p>

def valid_parameters(dimensions: KernelShape.Dimensions,weights: Buffer.Model<F32>,bias: Buffer.Model<F32>) -> Bool:
  KernelShape.Dimensions{+ci,co,+h,+w,k,s,p} = dimensions
  KernelShape.valid(dimensions,Nat.mul(ci,Nat.mul(h,w)),Buffer.length(~F32,weights),Buffer.length(~F32,bias))

def guard(+geometry: TensorConvolution.Geometry,layout: Bands.Geometry,weights: Buffer.Model<F32>,bias: Buffer.Model<F32>,inspection: Bands.Inspection) -> Bool:
  TensorConvolution.Geometry{+count,batched,+dimensions} = geometry
  KernelShape.Dimensions{ci,co,+height,width,kernel,stride,padding} = dimensions
  Bands.Geometry{batch,channels,h,w} = layout
  Buffer.Model{weight_tree,+weight_length,Certificate.Certificate{+weight_depth,weight_equation}} = weights
  Buffer.Model{bias_tree,+bias_length,Certificate.Certificate{+bias_depth,bias_equation}} = bias
  Bands.Inspection{valid,first,end,leaves} = inspection
  valid && U32.is_zero(first) && Nat.is_eq(U32.to_nat(end),Nat.mul(U32.to_nat(count),height)) &&
    U32.is_eq(batch,count) && Nat.is_eq(U32.to_nat(channels),ci) && Nat.is_eq(U32.to_nat(h),height) && Nat.is_eq(U32.to_nat(w),width) &&
    KernelShape.valid(dimensions,Nat.mul(ci,Nat.mul(height,width)),weight_length,bias_length) &&
    Nat.is_le(weight_length,U32.to_nat(U32.shln(1,weight_depth))) && Nat.is_le(bias_length,U32.to_nat(U32.shln(1,bias_depth)))

def allocation_room(~Element: Data,+model: Buffer.Model<Element>,ready: Buffer.Sized<Element,model>) ->
  {Nat.is_le(Buffer.length(~Element,model),U32.to_nat(U32.shln(1,Buffer.depth(~Element,model)))) == True{} : Bool}:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,equation}} = model
  Buffer.Sized{depth_ok,room} = ready
  %Shape.capacity(Element,depth,tree,Buffer.balanced(~Element,model)) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
  room

def checked_guards_accepted(+images: U32,+batched: Bool,+dimensions: KernelShape.Dimensions,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+inspection: Bands.Inspection,
  words: Words(dimensions),
  checked: {Inspection.valid(inspection) && U32.is_zero(Inspection.first(inspection)) &&
    Nat.is_eq(U32.to_nat(Inspection.end(inspection)),Inspection.total_rows(Leaf.input_geometry(images,dimensions))) == True{} : Bool},
  validated: {valid_parameters(dimensions,weights,bias) == True{} : Bool},
  weight_room: {Nat.is_le(Buffer.length(~F32,weights),U32.to_nat(U32.shln(1,Buffer.depth(~F32,weights)))) == True{} : Bool},
  bias_room: {Nat.is_le(Buffer.length(~F32,bias),U32.to_nat(U32.shln(1,Buffer.depth(~F32,bias)))) == True{} : Bool}) ->
  {guard(TensorConvolution.Geometry{images,batched,dimensions},Leaf.input_geometry(images,dimensions),weights,bias,inspection) == True{} : Bool}:
  KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p} = dimensions
  Buffer.Model{weight_tree,+weight_length,Certificate.Certificate{+weight_depth,weight_equation}} = weights
  Buffer.Model{bias_tree,+bias_length,Certificate.Certificate{+bias_depth,bias_equation}} = bias
  Bands.Inspection{+valid,+first,+end,leaves} = inspection
  LeafValues.Words{+ci_word,co_word,+h_word,+w_word,k_word,s_word,p_word,oh_word,ow_word} = words
  +checked = {
    %h_word : {valid && U32.is_zero(first) && Nat.is_eq(U32.to_nat(end),Nat.mul(U32.to_nat(images),_)) == True{} : Bool}
    checked : {valid && U32.is_zero(first) && Nat.is_eq(U32.to_nat(end),Nat.mul(U32.to_nat(images),h)) == True{} : Bool}}
  +ci_equal = {%ci_word : {Nat.is_eq(U32.to_nat(U32.from_nat(ci)),_) == True{} : Bool} Order.equal_reflexive(U32.to_nat(U32.from_nat(ci))) : {Nat.is_eq(U32.to_nat(U32.from_nat(ci)),ci) == True{} : Bool}}
  +height_equal = {%h_word : {Nat.is_eq(U32.to_nat(U32.from_nat(h)),_) == True{} : Bool} Order.equal_reflexive(U32.to_nat(U32.from_nat(h))) : {Nat.is_eq(U32.to_nat(U32.from_nat(h)),h) == True{} : Bool}}
  +width_equal = {%w_word : {Nat.is_eq(U32.to_nat(U32.from_nat(w)),_) == True{} : Bool} Order.equal_reflexive(U32.to_nat(U32.from_nat(w))) : {Nat.is_eq(U32.to_nat(U32.from_nat(w)),w) == True{} : Bool}}
  Boolean.all_hold([Nat.is_le(bias_length,U32.to_nat(U32.shln(1,bias_depth))),Nat.is_le(weight_length,U32.to_nat(U32.shln(1,weight_depth))),
      KernelShape.valid(dimensions,Nat.mul(ci,Nat.mul(h,w)),weight_length,bias_length),Nat.is_eq(U32.to_nat(U32.from_nat(w)),w),Nat.is_eq(U32.to_nat(U32.from_nat(h)),h),
      Nat.is_eq(U32.to_nat(U32.from_nat(ci)),ci),U32.is_eq(images,images),valid && U32.is_zero(first) && Nat.is_eq(U32.to_nat(end),Nat.mul(U32.to_nat(images),h))],
    Shape.Both{bias_room,Shape.Both{weight_room,Shape.Both{validated,Shape.Both{width_equal,Shape.Both{height_equal,
      Shape.Both{ci_equal,Shape.Both{BandShape.equal_words(images,images,{==}),Shape.Both{checked,Unit{}}}}}}}}})

def weight_length(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Nat:
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  ConvSpec.length(weights)

def bias_length(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Nat:
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  ConvSpec.length(bias)

def parameters(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  weight_exact: {Buffer.length(~F32,weights) == weight_length(~Logical,frame) : Nat},
  bias_exact: {Buffer.length(~F32,bias) == bias_length(~Logical,frame) : Nat}) ->
  {valid_parameters(PipelineExecution.dimensions(~Logical,frame),weights,bias) == True{} : Bool}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{+ci,co,+h,+w,k,s,p},images,parallel,limits,capacity,+weight_list,+bias_list,we,be,domain,valid,supported,room,eligible} = frame
  %Equal.sym(Nat,Buffer.length(~F32,weights),ConvSpec.length(weight_list),weight_exact) :
    {KernelShape.valid(KernelShape.Dimensions{ci,co,h,w,k,s,p},Nat.mul(ci,Nat.mul(h,w)),_,Buffer.length(~F32,bias)) == True{} : Bool}
  %Equal.sym(Nat,Buffer.length(~F32,bias),ConvSpec.length(bias_list),bias_exact) :
    {KernelShape.valid(KernelShape.Dimensions{ci,co,h,w,k,s,p},Nat.mul(ci,Nat.mul(h,w)),ConvSpec.length(weight_list),_) == True{} : Bool}
  valid

def frame(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  weight_exact: {Buffer.length(~F32,weights) == weight_length(~Logical,frame) : Nat},bias_exact: {Buffer.length(~F32,bias) == bias_length(~Logical,frame) : Nat},
  weight_ready: Buffer.Sized<F32,weights>,bias_ready: Buffer.Sized<F32,bias>) ->
  {guard(TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),weights,bias,Inspection.model(~F32,PipelineExecution.input_geometry(~Logical,frame),source)) == True{} : Bool}:
  PipelineExecution.Frame{+logical,activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,parallel,+limits,capacity,+weight_list,+bias_list,we,be,domain,+valid,+supported,room,eligible} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  checked_guards_accepted(images,batched,dimensions,weights,bias,Inspection.model(~F32,Leaf.input_geometry(images,dimensions),source),
    Caller.words(limits,Caller.problem(~Logical,~original,logical,dimensions,weight_list,bias_list,0n),
      Caller.valid(~Logical,~original,logical,dimensions,weight_list,bias_list,0n,valid),Caller.supported(~Logical,~original,logical,limits,dimensions,weight_list,bias_list,0n,supported)),
    inspection_source(~Logical,frame,source,shape),parameters(~Logical,frame,weights,bias,weight_exact,bias_exact),allocation_room(~F32,weights,weight_ready),allocation_room(~F32,bias,bias_ready))

# Public caller results use the ordered convolution specification context.
# Planner parallelism changes ownership scheduling while preserving this value.
def Contract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,tensor: Tensors.Tensor<F32>) -> Data:
  Shape.Both<BandMap.LogicalValues(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,tensor),BandMap.WellFormed(~F32,tensor)>

def Related(~Logical: Data,frame: PipelineExecution.Frame<Logical>,context: BandConvolutionValues.Context<Logical>) -> Data:
  {PipelineExecution.output_context(~Logical,frame) == context : BandConvolutionValues.Context<Logical>}

def frame_value(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+index: U32) ->
  {PipelineExecution.result_value(~custom,~Logical,~original,frame,index) == BandConvolutionValues.value(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),index) : F32}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  {==}

def agreement(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+context: BandConvolutionValues.Context<Logical>,+related: Related(~Logical,frame,context),+index: U32) ->
  {PipelineExecution.result_value(~custom,~Logical,~original,frame,index) == BandConvolutionValues.value(~custom,~Logical,~original,context,index) : F32}:
  Equal.trans(F32,PipelineExecution.result_value(~custom,~Logical,~original,frame,index),BandConvolutionValues.value(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),index),BandConvolutionValues.value(~custom,~Logical,~original,context,index),
    frame_value(~custom,~Logical,~original,frame,index),Equal.cong(BandConvolutionValues.Context<Logical>,F32,value => BandConvolutionValues.value(~custom,~Logical,~original,value,index),PipelineExecution.output_context(~Logical,frame),context,related))

def settings_logical(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>) ->
  {PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) == frame : PipelineExecution.Frame<Logical>}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  {==}

def values(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+context: BandConvolutionValues.Context<Logical>,+related: Related(~Logical,frame,context),tensor: Tensors.Tensor<F32>,
  values: TensorExecution.TensorValues(~custom,~Logical,~original,frame,tensor)) ->
  BandMap.LogicalValues(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,tensor):
  match tensor:
    case Tensors.Banded{bands,+geometry,shape,fill,status,workers}:
      BandValues.values(~F32,~PipelineExecution.Frame<Logical>,~BandConvolutionValues.Context<Logical>,~PipelineExecution.result_value(~custom,~Logical,~original),~BandConvolutionValues.value(~custom,~Logical,~original),~Related(~Logical),~agreement(~custom,~Logical,~original),
        frame,context,related,geometry,Model.contents(~F32,bands),
        %settings_logical(~Logical,~original,frame) : BandValues.Values(~F32,~PipelineExecution.Frame<Logical>,~PipelineExecution.result_value(~custom,~Logical,~original),_,geometry,Model.contents(~F32,bands))
        values)
    case Tensors.Tensor{storage,shape,fill,status,workers}: match values:

def from_frame(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+context: BandConvolutionValues.Context<Logical>,+related: Related(~Logical,frame,context),tensor: Tensors.Tensor<F32>,
  evidence: TensorExecution.OwnersContract(~custom,~Logical,~original,frame,tensor)) -> Contract(~custom,~Logical,~original,context,tensor):
  Shape.Both{frame_values,well_formed} = evidence
  Shape.Both{values(~custom,~Logical,~original,frame,context,related,tensor,frame_values),well_formed}

def parallel_context(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+parallel: Bool) ->
  {PipelineExecution.output_context(~Logical,TensorExecution.with_parallel(~Logical,frame,parallel)) == PipelineExecution.output_context(~Logical,frame) : BandConvolutionValues.Context<Logical>}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,previous,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  {==}

def selected(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+selection: BandPlanning.LayoutPlan,tensor: Tensors.Tensor<F32>,
  evidence: TensorExecution.AlignmentContract(~custom,~Logical,~original,frame,selection,tensor)) -> Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),tensor):
  from_frame(~custom,~Logical,~original,TensorExecution.selected_frame(~Logical,frame,selection),PipelineExecution.output_context(~Logical,frame),parallel_context(~Logical,frame,TensorExecution.alignment_parallel(selection)),tensor,evidence)
