# Public shape guards and the single-image leaf problem share one envelope.
# No GEMM detail, image-inside assumption or additional shape guard is needed.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/machine_limits.bend as Machine
import ./band_window_contents.bend as WindowContents
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_leaf_values.bend as LeafValues
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./supported_padded_coordinates.bend as Padded
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./band_window_leaf.bend as WindowLeaf
import ./window_start.bend as Start
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/u32_comparison.bend as Comparison

def problem(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,dimensions: KernelShape.Dimensions,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat) -> ConvSpec.Problem:
  KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} = dimensions
  BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)

def input_length(dimensions: KernelShape.Dimensions) -> Nat:
  KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} = dimensions
  Nat.mul(ci,Nat.mul(height,width))

def valid_matches(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+dimensions: KernelShape.Dimensions,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat) ->
  {ConvSpec.valid(problem(~Logical,~original,logical,dimensions,weights,bias,image)) == KernelShape.valid(dimensions,input_length(dimensions),ConvSpec.length(weights),ConvSpec.length(bias)) : Bool}:
  KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding} = dimensions
  +area = Nat.mul(ci,Nat.mul(height,width))
  Equal.cong(Nat,Bool,length => KernelShape.valid(KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},length,ConvSpec.length(weights),ConvSpec.length(bias)),
    ConvSpec.length(WindowContents.image_values(~Logical,~original,area,Nat.mul(image,area),logical)),area,
    WindowContents.image_length(~Logical,~original,area,Nat.mul(image,area),logical))

def valid(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+dimensions: KernelShape.Dimensions,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,
  checked: {KernelShape.valid(dimensions,input_length(dimensions),ConvSpec.length(weights),ConvSpec.length(bias)) == True{} : Bool}) ->
  {ConvSpec.valid(problem(~Logical,~original,logical,dimensions,weights,bias,image)) == True{} : Bool}:
  %Equal.sym(Bool,ConvSpec.valid(problem(~Logical,~original,logical,dimensions,weights,bias,image)),KernelShape.valid(dimensions,input_length(dimensions),ConvSpec.length(weights),ConvSpec.length(bias)),
    valid_matches(~Logical,~original,logical,dimensions,weights,bias,image)) : {_ == True{} : Bool}
  checked

def supported_matches(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+dimensions: KernelShape.Dimensions,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat) ->
  {NativeConv.supports_packed_with_limits(limits,problem(~Logical,~original,logical,dimensions,weights,bias,image)) == KernelShape.supports_packed(limits,dimensions,input_length(dimensions),ConvSpec.length(weights)) : Bool}:
  KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding} = dimensions
  +area = Nat.mul(ci,Nat.mul(height,width))
  Equal.cong(Nat,Bool,length => KernelShape.supports_packed(limits,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},length,ConvSpec.length(weights)),
    ConvSpec.length(WindowContents.image_values(~Logical,~original,area,Nat.mul(image,area),logical)),area,
    WindowContents.image_length(~Logical,~original,area,Nat.mul(image,area),logical))

def supported(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+dimensions: KernelShape.Dimensions,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,
  checked: {KernelShape.supports_packed(limits,dimensions,input_length(dimensions),ConvSpec.length(weights)) == True{} : Bool}) ->
  {NativeConv.supports_packed_with_limits(limits,problem(~Logical,~original,logical,dimensions,weights,bias,image)) == True{} : Bool}:
  %Equal.sym(Bool,NativeConv.supports_packed_with_limits(limits,problem(~Logical,~original,logical,dimensions,weights,bias,image)),KernelShape.supports_packed(limits,dimensions,input_length(dimensions),ConvSpec.length(weights)),
    supported_matches(~Logical,~original,logical,limits,dimensions,weights,bias,image)) : {_ == True{} : Bool}
  checked

# The nine narrowing facts come from the accepted leaf problem's existing
# guards. Output axes use the coordinate envelope, not an example shape.
def words(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  LeafValues.Words<Guards.input_channel_count(problem),Guards.output_channel_count(problem),Guards.height(problem),Guards.width(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem)>:
  ConvSpec.Problem{ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding},+input,+weights,+bias} = problem
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +conditions = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  LeafValues.Words{
    SupportedCoordinateBounds.ci_exact(limits,problem,support),SupportedCoordinateBounds.co_exact(limits,problem,support),
    SupportedCoordinateBounds.height_exact(limits,problem,support),SupportedCoordinateBounds.width_exact(limits,problem,support),SupportedCoordinateBounds.kernel_exact(limits,problem,support),
    Bounds.roundtrip(stride,Machine.dimension(limits),Limits.dimension_fits(limits),SupportedCoordinateBounds.stride_bound(limits,problem,support)),SupportedCoordinateBounds.pad_exact(limits,problem,support),
    Bounds.roundtrip(ConvSpec.out_height(problem),Machine.coordinate(limits),Limits.coordinate_fits(limits),SupportedCoordinateBounds.row_limit(limits,problem,conditions,support)),
    Bounds.roundtrip(ConvSpec.out_width(problem),Machine.coordinate(limits),Limits.coordinate_fits(limits),SupportedCoordinateBounds.col_limit(limits,problem,conditions,support))}

def padding_small(+images: U32,+batched: Bool,+ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,
  eligible: {TensorConvolution.uses_band_windows(TensorConvolution.Geometry{images,batched,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding}}) == True{} : Bool}) ->
  {Nat.is_lt(padding,kernel) == True{} : Bool}:
  Boolean.right(U32.is_lt(0,images),Nat.is_lt(padding,kernel),eligible)

def dimension_room(+limits: Machine.Limits) -> {Nat.is_le(U32.to_nat(Machine.dimension(limits)),U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool}:
  Order.transitive(U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.coordinate(limits)),U32.to_nat(WindowLeaf.capacity(limits)),
    Bounds.as_nat_le(Machine.dimension(limits),Machine.coordinate(limits),Limits.dimension_within_coordinate(limits)),WindowLeaf.coordinate_room(limits))

# The scalar support envelope supplies window coordinate room; the batch
# output bound is the additional conjunct already checked by the public API.
def room(+limits: Machine.Limits,+images: U32,+problem: ConvSpec.Problem,+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  output_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(Guards.output_channel_count(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  LeafValues.Room<limits,images,Guards.output_channel_count(problem),Guards.height(problem),Guards.width(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem)>:
  ConvSpec.Problem{ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding},+input,+weights,+bias} = problem
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +envelope = WindowLeaf.capacity(limits)
  LeafValues.Room{
    %Start.stride_word(limits,problem,support) : {Nat.is_le(Nat.double(_),U32.to_nat(envelope)) == True{} : Bool}
    Order.transitive(Nat.double(U32.to_nat(U32.from_nat(stride))),U32.to_nat(Machine.coordinate(limits)),U32.to_nat(envelope),
      Start.stride_divisor(limits,problem,support),WindowLeaf.coordinate_room(limits)),
    Order.transitive(Nat.add(Nat.mul(ConvSpec.out_height(problem),stride),kernel),U32.to_nat(Machine.coordinate(limits)),U32.to_nat(envelope),
      SupportedCoordinateBounds.row_envelope(limits,problem,support),WindowLeaf.coordinate_room(limits)),
    Order.transitive(height,U32.to_nat(Machine.dimension(limits)),U32.to_nat(envelope),SupportedCoordinateBounds.height_bound(limits,problem,support),dimension_room(limits)),
    Order.transitive(padding,U32.to_nat(Machine.dimension(limits)),U32.to_nat(envelope),SupportedCoordinateBounds.pad_bound(limits,problem,support),dimension_room(limits)),output_room}

def images_positive(+images: U32,+batched: Bool,+ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,
  eligible: {TensorConvolution.uses_band_windows(TensorConvolution.Geometry{images,batched,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding}}) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(images)) == True{} : Bool}:
  %Comparison.lt(0,images) : {_ == True{} : Bool}
  Boolean.left(U32.is_lt(0,images),Nat.is_lt(padding,kernel),eligible)

def output_positive_facts(+problem: ConvSpec.Problem,facts: Guards.Valid<problem>) -> {Guards.output_channels_positive(problem) == True{} : Bool}:
  Guards.Valid{ci,co,height,width,kernel,stride,kh,kw,input,weights,bias} = facts
  co

def stride_positive_facts(+problem: ConvSpec.Problem,facts: Guards.Valid<problem>) -> {Guards.stride_positive(problem) == True{} : Bool}:
  Guards.Valid{ci,co,height,width,kernel,stride,kh,kw,input,weights,bias} = facts
  stride

def outputs_positive(+problem: ConvSpec.Problem,valid: {ConvSpec.valid(problem) == True{} : Bool}) -> {Guards.output_channels_positive(problem) == True{} : Bool}:
  output_positive_facts(problem,Guards.valid(problem,valid))

def stride_positive(+problem: ConvSpec.Problem,valid: {ConvSpec.valid(problem) == True{} : Bool}) -> {Guards.stride_positive(problem) == True{} : Bool}:
  stride_positive_facts(problem,Guards.valid(problem,valid))
