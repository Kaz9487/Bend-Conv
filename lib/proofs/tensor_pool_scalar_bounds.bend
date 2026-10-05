# The actual pool guard supplies the formal scalar envelope.
import Base
import ../tensor_operations.bend as Operations
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_pool_scalar_envelope.bend as ScalarEnvelope

# All certificates here compare native words. The derived words follow
# Operations.small's 65535 guard; they are not new acceptance limits.
type Observation is Data:
  Observation{envelope: ScalarEnvelope.Envelope,capacity: U32,half: U32,small: U32,
    limits_default: {ScalarEnvelope.limits(envelope) == Default.limits() : Machine.Limits},
    capacity_model: {Machine.capacity(ScalarEnvelope.limits(envelope)) == capacity : U32},
    capacity_default: {capacity == Machine.capacity(Default.limits()) : U32},
    half_model: {Machine.half_word(ScalarEnvelope.limits(envelope)) == half : U32},
    half_default: {half == Machine.half_word(Default.limits()) : U32},
    small_model: {ScalarEnvelope.small(envelope) == small : U32},
    small_default: {small == 65535 : U32}}

def observe_envelope() -> Observation:
  +envelope = {ScalarEnvelope.Envelope{Default.limits(),65535,131070,196605,196606,262140,327675,
    ScalarEnvelope.Constraints{{==},{==},{==},{==},{==},{==},{==},{==},{==},{==},{==},{==}}} : ScalarEnvelope.Envelope}
  Observation{envelope,Machine.capacity(Default.limits()),Machine.half_word(Default.limits()),65535,
    {==},{==},{==},{==},{==},{==},{==}}

def limits(observation: Observation) -> Machine.Limits:
  Observation{envelope,capacity,half,small,limits_default,capacity_model,capacity_default,half_model,half_default,small_model,small_default} = observation
  ScalarEnvelope.limits(envelope)

def actual_limits(+observation: Observation) -> {limits(observation) == Default.limits() : Machine.Limits}:
  Observation{envelope,capacity,half,small,limits_default,capacity_model,capacity_default,half_model,half_default,small_model,small_default} = observation
  limits_default

def small_guard(+envelope: ScalarEnvelope.Envelope,+bound: U32,+value: U32,
  model: {ScalarEnvelope.small(envelope) == bound : U32},actual: {bound == 65535 : U32},guard: {Operations.small(value) == True{} : Bool}) ->
  {U32.is_le(value,ScalarEnvelope.small(envelope)) == True{} : Bool}:
  %Equal.sym(U32,ScalarEnvelope.small(envelope),65535,Equal.trans(U32,ScalarEnvelope.small(envelope),bound,65535,model,actual)) : {U32.is_le(value,_) == True{} : Bool}
  guard

def small_observed(+observation: Observation,+value: U32,small: {Operations.small(value) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(value),U32.to_nat(Machine.capacity(limits(observation)))) == True{} : Bool}:
  Observation{+envelope,capacity,half,+bound,limits_default,capacity_model,capacity_default,half_model,half_default,small_model,small_default} = observation
  ScalarEnvelope.small_room(envelope,value,small_guard(envelope,bound,value,small_model,small_default,small))

def axis_output_observed(observation: Observation,+extent: U32,+kernel: U32,+stride: U32,+padding: U32,+output_extent: U32,
  extent_small: {Operations.small(extent) == True{} : Bool},padding_small: {Operations.small(padding) == True{} : Bool},
  stride_small: {Operations.small(stride) == True{} : Bool},stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  kernel_inside: {U32.is_le(kernel,(extent + 2 * padding : U32)) == True{} : Bool},
  output_equal: {U32.is_eq(output_extent,((extent + 2 * padding - kernel : U32) / stride + 1 : U32)) == True{} : Bool}) ->
  {U32.to_nat(output_extent) == Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding))),U32.to_nat(kernel)),U32.to_nat(stride)),1n) : Nat}:
  Observation{+envelope,capacity,half,+bound,limits_default,capacity_model,capacity_default,half_model,half_default,+small_model,+small_default} = observation
  ScalarEnvelope.axis_output_exact(envelope,extent,kernel,stride,padding,output_extent,
    small_guard(envelope,bound,extent,small_model,small_default,extent_small),small_guard(envelope,bound,padding,small_model,small_default,padding_small),small_guard(envelope,bound,stride,small_model,small_default,stride_small),stride_positive,kernel_inside,output_equal)

def axis_scalar_observed(+observation: Observation,+extent: U32,+kernel: U32,+stride: U32,+padding: U32,+output_extent: U32,+coordinate: Nat,+tap: Nat,
  extent_small: {Operations.small(extent) == True{} : Bool},padding_small: {Operations.small(padding) == True{} : Bool},
  kernel_small: {Operations.small(kernel) == True{} : Bool},stride_small: {Operations.small(stride) == True{} : Bool},
  stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  output_exact: {U32.to_nat(output_extent) == Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding))),U32.to_nat(kernel)),U32.to_nat(stride)),1n) : Nat},
  coordinate_inside: {Nat.is_le(coordinate,U32.to_nat(output_extent)) == True{} : Bool},tap_inside: {Nat.is_le(tap,U32.to_nat(kernel)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(coordinate,U32.to_nat(stride)),tap),U32.to_nat(Machine.half_word(limits(observation)))) == True{} : Bool}:
  Observation{+envelope,capacity,half,+bound,limits_default,capacity_model,capacity_default,half_model,half_default,+small_model,+small_default} = observation
  ScalarEnvelope.axis_scalar_room(envelope,extent,kernel,stride,padding,output_extent,coordinate,tap,
    small_guard(envelope,bound,extent,small_model,small_default,extent_small),small_guard(envelope,bound,padding,small_model,small_default,padding_small),small_guard(envelope,bound,kernel,small_model,small_default,kernel_small),small_guard(envelope,bound,stride,small_model,small_default,stride_small),stride_positive,output_exact,coordinate_inside,tap_inside)

def small_room(+observation: Observation,+value: U32,small: {Operations.small(value) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(value),U32.to_nat(Machine.capacity(limits(observation)))) == True{} : Bool}:
  small_observed(observation,value,small)

def axis_output_exact(+observation: Observation,+extent: U32,+kernel: U32,+stride: U32,+padding: U32,+output_extent: U32,
  +extent_small: {Operations.small(extent) == True{} : Bool},+padding_small: {Operations.small(padding) == True{} : Bool},
  stride_small: {Operations.small(stride) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  kernel_inside: {U32.is_le(kernel,(extent + 2 * padding : U32)) == True{} : Bool},
  output_equal: {U32.is_eq(output_extent,((extent + 2 * padding - kernel : U32) / stride + 1 : U32)) == True{} : Bool}) ->
  {U32.to_nat(output_extent) == Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding))),U32.to_nat(kernel)),U32.to_nat(stride)),1n) : Nat}:
  axis_output_observed(observation,extent,kernel,stride,padding,output_extent,extent_small,padding_small,stride_small,stride_positive,kernel_inside,output_equal)

def axis_scalar_room(+observation: Observation,+extent: U32,+kernel: U32,+stride: U32,+padding: U32,+output_extent: U32,+coordinate: Nat,+tap: Nat,
  extent_small: {Operations.small(extent) == True{} : Bool},+padding_small: {Operations.small(padding) == True{} : Bool},
  kernel_small: {Operations.small(kernel) == True{} : Bool},stride_small: {Operations.small(stride) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +output_exact: {U32.to_nat(output_extent) == Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding))),U32.to_nat(kernel)),U32.to_nat(stride)),1n) : Nat},
  coordinate_inside: {Nat.is_le(coordinate,U32.to_nat(output_extent)) == True{} : Bool},tap_inside: {Nat.is_le(tap,U32.to_nat(kernel)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(coordinate,U32.to_nat(stride)),tap),U32.to_nat(Machine.half_word(limits(observation)))) == True{} : Bool}:
  axis_scalar_observed(observation,extent,kernel,stride,padding,output_extent,coordinate,tap,extent_small,padding_small,kernel_small,stride_small,stride_positive,output_exact,coordinate_inside,tap_inside)
