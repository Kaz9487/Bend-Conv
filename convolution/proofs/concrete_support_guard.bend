import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
# Runtime guard bodies -> logical facts for any certified machine limits.
# Default execution supplies DefaultMachineLimits.limits(); official checker only.
import Base
import ../../lib/proofs/boolean_logic.bend as Boolean
import ./convolution_guard_facts.bend as Guards
import ./list_array_allocation.bend as ListArrayAllocation
import ../specification.bend as ConvSpec
import ../scalar_kernel.bend as ScalarConv
import ../inference.bend as NativeConv
import ../../lib/proofs/word_encoding.bend as WordEncoding

def scalar(+limits: Machine.Limits,+p: ConvSpec.Problem,h: {ScalarConv.supported_with_limits(limits,
      p) == True{} : Bool}) -> Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>:
  Machine.Limits{dimension,capacity,coordinate,half_word,facts} = limits
  ConvSpec.Problem{shape,+xs,+ws,+bs} = p
  ConvSpec.Shape{+v0,+v1,+v2,+v3,+v4,+v5,+v6} = shape
  Guards.supported_generic(p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits),h)

def padded(+p: ConvSpec.Problem) -> Nat:
  Nat.mul(Nat.div(Nat.add(Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)),7n),8n),8n)

def packed_bound(+limits: Machine.Limits,+p: ConvSpec.Problem) -> Bool:
  Nat.is_le(Nat.mul(padded(p),Nat.mul(ConvSpec.channels(p),Nat.mul(ConvSpec.kernel_size(p),ConvSpec.kernel_size(p)))),
    U32.to_nat(Machine.capacity(limits)))

def output_bound(+limits: Machine.Limits,+p: ConvSpec.Problem) -> Bool:
  Nat.is_le(Nat.mul(padded(p),ConvSpec.outputs(p)),U32.to_nat(Machine.capacity(limits)))

def packed_guard_matches_original_conditions(+limits: Machine.Limits,+p: ConvSpec.Problem) ->
  {NativeConv.supports_packed_with_limits(limits,p) == ScalarConv.supported_with_limits(limits,p) && packed_bound(limits,p) && output_bound(limits,p) : Bool}:
  Machine.Limits{dimension,capacity,coordinate,half_word,facts} = limits
  ConvSpec.Problem{shape,input,weights,bias} = p
  ConvSpec.Shape{input_channel_count,output_channel_count,height,width,kernel,stride,padding} = shape
  {==}

def packed_conditions(+limits: Machine.Limits,+p: ConvSpec.Problem,h: {NativeConv.supports_packed_with_limits(limits,p) == True{} : Bool}) ->
  {ScalarConv.supported_with_limits(limits,p) && packed_bound(limits,p) && output_bound(limits,p) == True{} : Bool}:
  %packed_guard_matches_original_conditions(limits,p) : {_ == True{} : Bool}
  h

type FastSupported<-limits: Machine.Limits,-p: ConvSpec.Problem> is Data:
  FastSupported{scalar: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
    Machine.coordinate(limits)>,packing: {packed_bound(limits,p) == True{} : Bool},output: {output_bound(limits,p) == True{} : Bool}}

def fast(+limits: Machine.Limits,+p: ConvSpec.Problem,+h: {NativeConv.supports_packed_with_limits(limits,
      p) == True{} : Bool}) -> FastSupported<limits,p>:
  +conditions = packed_conditions(limits,p,h)
  +ab = Boolean.left(ScalarConv.supported_with_limits(limits,p) && packed_bound(limits,p),output_bound(limits,p),conditions)
  FastSupported{scalar(limits,p,Boolean.left(ScalarConv.supported_with_limits(limits,p),packed_bound(limits,p),ab)),
    Boolean.right(ScalarConv.supported_with_limits(limits,p),packed_bound(limits,p),ab),
    Boolean.right(ScalarConv.supported_with_limits(limits,p) && packed_bound(limits,p),output_bound(limits,p),conditions)}

def positive_mul(+a: Nat,+b: Nat,ha: {Nat.is_lt(0n,a) == True{} : Bool},hb: {Nat.is_lt(0n,b) == True{} : Bool}) -> {Nat.is_lt(0n,Nat.mul(a,
      b)) == True{} : Bool}:
  match a b:
    case 0n b:
      Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(ha))
    case 1n+a 0n:
      Empty.absurd({Nat.is_lt(0n,Nat.mul(1n+a,0n)) == True{} : Bool},WordEncoding.false_true(hb))
    case 1n+a 1n+b:
      {==}

def input_positive(+p: ConvSpec.Problem,valid: Guards.Valid<p>) -> {Nat.is_lt(0n,
    ConvSpec.length(Guards.input(p))) == True{} : Bool}:
  Guards.Valid{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10} = valid
  %Equal.sym(Nat,ConvSpec.length(Guards.input(p)),Nat.mul(Guards.input_channel_count(p),Nat.mul(Guards.height(p),
        Guards.width(p))),Guards.reflect(ConvSpec.length(Guards.input(p)),
      Nat.mul(Guards.input_channel_count(p),Nat.mul(Guards.height(p),Guards.width(p))),h8)) : {Nat.is_lt(0n,
      _) == True{} : Bool}
  positive_mul(Guards.input_channel_count(p),Nat.mul(Guards.height(p),Guards.width(p)),h0,
    positive_mul(Guards.height(p),Guards.width(p),h2,h3))

def weights_positive(+p: ConvSpec.Problem,valid: Guards.Valid<p>) -> {Nat.is_lt(0n,
    ConvSpec.length(Guards.weights(p))) == True{} : Bool}:
  Guards.Valid{h0,h1,h2,h3,+h4,h5,h6,h7,h8,h9,h10} = valid
  %Equal.sym(Nat,ConvSpec.length(Guards.weights(p)),Nat.mul(Guards.output_channel_count(p),Nat.mul(Guards.input_channel_count(p),
        Nat.mul(Guards.kernel(p),Guards.kernel(p)))),
    Guards.reflect(ConvSpec.length(Guards.weights(p)),Nat.mul(Guards.output_channel_count(p),
        Nat.mul(Guards.input_channel_count(p),Nat.mul(Guards.kernel(p),Guards.kernel(p)))),h9)) : {Nat.is_lt(0n,
      _) == True{} : Bool}
  positive_mul(Guards.output_channel_count(p),Nat.mul(Guards.input_channel_count(p),Nat.mul(Guards.kernel(p),
        Guards.kernel(p))),h1,positive_mul(Guards.input_channel_count(p),Nat.mul(Guards.kernel(p),
        Guards.kernel(p)),h0,positive_mul(Guards.kernel(p),Guards.kernel(p),h4,h4)))

def bias_positive(+p: ConvSpec.Problem,valid: Guards.Valid<p>) -> {Nat.is_lt(0n,
    ConvSpec.length(Guards.bias(p))) == True{} : Bool}:
  Guards.Valid{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10} = valid
  %Equal.sym(Nat,ConvSpec.length(Guards.bias(p)),Guards.output_channel_count(p),
    Guards.reflect(ConvSpec.length(Guards.bias(p)),Guards.output_channel_count(p),h10)) : {Nat.is_lt(0n,
      _) == True{} : Bool}
  h1

def input_ready_facts(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> ListArrayAllocation.Ready(Guards.input(p)):
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  ListArrayAllocation.ready(Guards.input(p),Machine.capacity(limits),Limits.capacity_fits(limits),
    Limits.capacity_allocatable(limits),input_positive(p,valid),h7)

def weights_ready_facts(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> ListArrayAllocation.Ready(Guards.weights(p)):
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  ListArrayAllocation.ready(Guards.weights(p),Machine.capacity(limits),Limits.capacity_fits(limits),
    Limits.capacity_allocatable(limits),weights_positive(p,valid),h8)

def bias_ready_facts(+limits: Machine.Limits,+p: ConvSpec.Problem,+valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> ListArrayAllocation.Ready(Guards.bias(p)):
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  ListArrayAllocation.ready(Guards.bias(p),Machine.dimension(limits),Limits.dimension_fits(limits),
    Limits.dimension_allocatable(limits),bias_positive(p,valid),
    %Equal.sym(Nat,ConvSpec.length(Guards.bias(p)),Guards.output_channel_count(p),
      Guards.reflect(ConvSpec.length(Guards.bias(p)),Guards.output_channel_count(p),v10)) : {Nat.is_le(_,
        U32.to_nat(Machine.dimension(limits))) == True{} : Bool}
    h1)

def scalar_guard(+limits: Machine.Limits,+p: ConvSpec.Problem,+support: {NativeConv.supports_packed_with_limits(limits,
      p) == True{} : Bool}) -> {ScalarConv.supported_with_limits(limits,p) == True{} : Bool}:
  Boolean.left(ScalarConv.supported_with_limits(limits,p),packed_bound(limits,p),Boolean.left(ScalarConv.supported_with_limits(limits,
        p) && packed_bound(limits,p),output_bound(limits,p),packed_conditions(limits,p,support)))
