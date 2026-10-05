# Assemble the convolution proof frame from validated shape and machine facts.
# Machine limits remain parameters; the public caller constructs the source
# domain and chooses the actual limits before using this configuration.
import Base
import ../specification.bend as ConvSpec
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/math_activation.bend as Activations
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/nat_order.bend as Order
import ./convolution_guard_facts.bend as Guards
import ./band_caller_guards.bend as Caller
import ./band_leaf_caller.bend as Leaf
import ./band_leaf_values.bend as LeafValues
import ./band_pipeline_execution.bend as PipelineExecution

def weight_size(dimensions: KernelShape.Dimensions) -> Nat:
  KernelShape.Dimensions{ci,co,h,w,+k,s,p} = dimensions
  Nat.mul(co,Nat.mul(ci,Nat.mul(k,k)))

def weights_length(+problem: ConvSpec.Problem,facts: Guards.Valid<problem>) ->
  {ConvSpec.length(Guards.weights(problem)) == Nat.mul(Guards.output_channel_count(problem),Nat.mul(Guards.input_channel_count(problem),Nat.mul(Guards.kernel(problem),Guards.kernel(problem)))) : Nat}:
  Guards.Valid{ci,co,h,w,k,s,kh,kw,input,weights,bias} = facts
  Order.equal_of_is_eq(ConvSpec.length(Guards.weights(problem)),Nat.mul(Guards.output_channel_count(problem),Nat.mul(Guards.input_channel_count(problem),Nat.mul(Guards.kernel(problem),Guards.kernel(problem)))),weights)

def supported(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+dimensions: KernelShape.Dimensions,+weights: List<&2,F32>,+bias: List<&2,F32>,
  valid: {KernelShape.valid(dimensions,Caller.input_length(dimensions),ConvSpec.length(weights),ConvSpec.length(bias)) == True{} : Bool},
  accepted: {KernelShape.supports_packed(limits,dimensions,Caller.input_length(dimensions),weight_size(dimensions)) == True{} : Bool}) ->
  {KernelShape.supports_packed(limits,dimensions,Caller.input_length(dimensions),ConvSpec.length(weights)) == True{} : Bool}:
  KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p} = dimensions
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +problem = Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n)
  +weight_exact = weights_length(problem,Guards.valid(problem,Caller.valid(~Logical,~original,logical,dimensions,weights,bias,0n,valid)))
  %Equal.sym(Nat,ConvSpec.length(weights),Nat.mul(co,Nat.mul(ci,Nat.mul(k,k))),weight_exact) :
    {KernelShape.supports_packed(limits,dimensions,Caller.input_length(dimensions),_) == True{} : Bool}
  accepted

def configured(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+images: U32,+dimensions: KernelShape.Dimensions,+parallel: Bool,+limits: Machine.Limits,+capacity: U32,
  +weights: List<&2,F32>,+bias: List<&2,F32>,domain: BandGather.Domain<Leaf.input_geometry(images,dimensions),limits,capacity>,
  weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +valid: {KernelShape.valid(dimensions,Caller.input_length(dimensions),ConvSpec.length(weights),ConvSpec.length(bias)) == True{} : Bool},
  +machine: {KernelShape.supports_packed(limits,dimensions,Caller.input_length(dimensions),weight_size(dimensions)) &&
    Nat.is_le(LeafValues.output_elements(images,dimensions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  eligible: {TensorConvolution.uses_band_windows(TensorConvolution.Geometry{images,True{},dimensions}) == True{} : Bool}) -> PipelineExecution.Frame<Logical>:
  +scalar_support = supported(~Logical,~original,logical,limits,dimensions,weights,bias,valid,
    Boolean.left(KernelShape.supports_packed(limits,dimensions,Caller.input_length(dimensions),weight_size(dimensions)),Nat.is_le(LeafValues.output_elements(images,dimensions),U32.to_nat(Machine.capacity(limits))),machine))
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weights,bias,weight_evidence,bias_evidence,domain,valid,scalar_support,
    Boolean.right(KernelShape.supports_packed(limits,dimensions,Caller.input_length(dimensions),weight_size(dimensions)),Nat.is_le(LeafValues.output_elements(images,dimensions),U32.to_nat(Machine.capacity(limits))),machine),eligible}
