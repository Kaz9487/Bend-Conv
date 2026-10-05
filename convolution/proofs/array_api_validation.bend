import Base
import ../../lib/proofs/storage_certificate.bend as Certificate
import ./native_inference_law.bend as InferenceLaw
import ./input_storage.bend as Inputs
import ../shape.bend as ConvShape
import ../scalar_kernel.bend as ScalarConv
import ../inference.bend as NativeConv
import ../matrix_specification.bend as MatrixSpec
import ../ordered_reference.bend as OrderedReference
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ./scalar_inference_refinement.bend as ScalarInferenceRefinement
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./matrix_matches_convolution.bend as Matrix
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../specification.bend as ConvSpec
import ../array_api.bend as ArrayApi
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/storage_buffer.bend as Storage

# The API's logical sequences are observations of the caller's actual storage.
# No canonical repacking or equality of physical capacities is required.
def problem(shape: ConvSpec.Shape,input_length: Nat,weight_length: Nat,bias_length: Nat,
  input: Tree.Tree<F32>,weights: Tree.Tree<F32>,bias: Tree.Tree<F32>) -> ConvSpec.Problem:
  ConvSpec.Problem{shape,Tree.read_sequence(~F32,input_length,0,input),
    Tree.read_sequence(~F32,weight_length,0,weights),Tree.read_sequence(~F32,bias_length,0,bias)}

law validation_matches_spec:
  for +shape: ConvSpec.Shape
  for +xs: List<&2,F32>
  for +ws: List<&2,F32>
  for +bs: List<&2,F32>
  {ArrayApi.valid(shape,ConvSpec.length(xs),ConvSpec.length(ws),ConvSpec.length(bs)) == ConvSpec.valid(ConvSpec.Problem{shape,xs,ws,
        bs}) : Bool}
def validation_matches_spec(shape,xs,ws,bs):
  match shape:
    case ConvSpec.Shape{input_channel_count,output_channel_count,h,w,k,stride,pad}:
      {==}

def validation_matches_buffers(+shape: ConvSpec.Shape,+input_length: Nat,+weight_length: Nat,+bias_length: Nat,
  +input: Tree.Tree<F32>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>) ->
  {ArrayApi.valid(shape,input_length,weight_length,bias_length) ==
    ConvSpec.valid(problem(shape,input_length,weight_length,bias_length,input,weights,bias)) : Bool}:
  %Inputs.observed_length(input_length,input) :
    {ArrayApi.valid(shape,_,weight_length,bias_length) == ConvSpec.valid(problem(shape,input_length,weight_length,bias_length,input,weights,bias)) : Bool}
  %Inputs.observed_length(weight_length,weights) :
    {ArrayApi.valid(shape,ConvSpec.length(Tree.read_sequence(~F32,input_length,0,input)),_,bias_length) ==
      ConvSpec.valid(problem(shape,input_length,weight_length,bias_length,input,weights,bias)) : Bool}
  %Inputs.observed_length(bias_length,bias) :
    {ArrayApi.valid(shape,ConvSpec.length(Tree.read_sequence(~F32,input_length,0,input)),ConvSpec.length(Tree.read_sequence(~F32,weight_length,0,weights)),_) ==
      ConvSpec.valid(problem(shape,input_length,weight_length,bias_length,input,weights,bias)) : Bool}
  validation_matches_spec(shape,Tree.read_sequence(~F32,input_length,0,input),Tree.read_sequence(~F32,weight_length,0,weights),Tree.read_sequence(~F32,bias_length,0,bias))

# The caller's physical capacity may exceed the logical sequence. Establish the
# same evidence consumed by scalar's existing reduction and output theorems.
def supplied_inputs(+shape: ConvSpec.Shape,+input_length: Nat,+weight_length: Nat,+bias_length: Nat,
  +input: Tree.Tree<F32>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>,
  input_room: {Nat.is_le(input_length,U32.to_nat(Tree.capacity(F32,input))) == True{} : Bool},
  weight_room: {Nat.is_le(weight_length,U32.to_nat(Tree.capacity(F32,weights))) == True{} : Bool},
  bias_room: {Nat.is_le(bias_length,U32.to_nat(Tree.capacity(F32,bias))) == True{} : Bool}) ->
  Inputs.Contract<problem(shape,input_length,weight_length,bias_length,input,weights,bias)>:
  Inputs.Contract{Inputs.prefix_evidence(input_length,input,input_room),Inputs.prefix_evidence(weight_length,weights,weight_room),Inputs.prefix_evidence(bias_length,bias,bias_room)}

# Both APIs use the same machine envelope. Validation supplies the shape/length
# equalities; no numerical-result premise is accepted here.
def machine_support_matches_problem(+limits: Machine.Limits,+shape: ConvSpec.Shape,
  +input: List<&2,F32>,+weights: List<&2,F32>,+bias: List<&2,F32>,
  facts: Guards.Valid<ConvSpec.Problem{shape,input,weights,bias}>) ->
  {ArrayApi.supported_with_limits(limits,shape) ==
    NativeConv.supports_packed_with_limits(limits,ConvSpec.Problem{shape,input,weights,bias}) : Bool}:
  ConvSpec.Shape{+input_channel_count,+output_channel_count,+height,+width,+kernel,+stride,+padding} = shape
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,input_shape,weight_shape,v10} = facts
  %Guards.reflect(ConvSpec.length(input),Nat.mul(input_channel_count,Nat.mul(height,width)),input_shape) :
    {ConvShape.supports_packed(limits,shape,_,Nat.mul(output_channel_count,Nat.mul(input_channel_count,Nat.mul(kernel,kernel)))) ==
      NativeConv.supports_packed_with_limits(limits,ConvSpec.Problem{shape,input,weights,bias}) : Bool}
  %Guards.reflect(ConvSpec.length(weights),Nat.mul(output_channel_count,Nat.mul(input_channel_count,Nat.mul(kernel,kernel))),weight_shape) :
    {ConvShape.supports_packed(limits,shape,ConvSpec.length(input),_) ==
      NativeConv.supports_packed_with_limits(limits,ConvSpec.Problem{shape,input,weights,bias}) : Bool}
  {==}

# Observe only the logical output. Output allocation capacity is an internal
# choice of strategy; Unsupported is represented by None, not a fake result.
def observe_result(+problem: ConvSpec.Problem,result: ArrayApi.Outcome) -> Maybe<&2,ConvSpec.ConvOutcome>:
  match result:
    case ArrayApi.Rejected{}: Some{ConvSpec.Rejected{}}
    case ArrayApi.Unsupported{}: None{}
    case ArrayApi.Accepted{+height,+width,values}:
      Some{ConvSpec.Accepted{height,width,Storage.read_list(F32,Nat.mul(ConvSpec.outputs(problem),Nat.mul(height,width)),0,values)}}

def execute_matches_matrix(+strategy: NativeConv.Strategy,+limits: Machine.Limits,+shape: ConvSpec.Shape,+input_length: Nat,+weight_length: Nat,+bias_length: Nat,
  +input: Tree.Tree<F32>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>,
  +input_room: {Nat.is_le(input_length,U32.to_nat(Tree.capacity(F32,input))) == True{} : Bool},
  +weight_room: {Nat.is_le(weight_length,U32.to_nat(Tree.capacity(F32,weights))) == True{} : Bool},
  +bias_room: {Nat.is_le(bias_length,U32.to_nat(Tree.capacity(F32,bias))) == True{} : Bool},
  +valid: {ConvSpec.valid(problem(shape,input_length,weight_length,bias_length,input,weights,bias)) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem(shape,input_length,weight_length,bias_length,input,weights,bias)) == True{} : Bool}) ->
  {observe_result(problem(shape,input_length,weight_length,bias_length,input,weights,bias),
      ArrayApi.execute(True{},shape,strategy,Tree.pack(F32,input),Tree.pack(F32,weights),Tree.pack(F32,bias))) ==
    Some{MatrixSpec.infer(problem(shape,input_length,weight_length,bias_length,input,weights,bias))} : Maybe<&2,ConvSpec.ConvOutcome>}:
  ConvSpec.Shape{+input_channel_count,+output_channel_count,+height,+width,+kernel,+stride,+padding} = shape
  +p = problem(shape,input_length,weight_length,bias_length,input,weights,bias)
  +values = MatrixSpec.values(p)
  +count = Nat.mul(output_channel_count,Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)))
  %Equal.sym(List<&2,F32>,Storage.read_list(F32,count,0,
      NativeConv.execute(strategy,False{},ConvShape.from_problem(p),Tree.pack(F32,input),Tree.pack(F32,weights),Tree.pack(F32,bias))),values,
    InferenceLaw.strategy_readback_matches_matrix(strategy,limits,p,
      supplied_inputs(shape,input_length,weight_length,bias_length,input,weights,bias,input_room,weight_room,bias_room),valid,supported)) :
    {Some{ConvSpec.Accepted{ConvSpec.out_height(p),ConvSpec.out_width(p),_}} == Some{MatrixSpec.infer(p)} : Maybe<&2,ConvSpec.ConvOutcome>}
  %Equal.sym(Bool,ConvSpec.valid(p),True{},valid) :
    {Some{ConvSpec.Accepted{ConvSpec.out_height(p),ConvSpec.out_width(p),values}} == Some{MatrixSpec.finish(_,p)} : Maybe<&2,ConvSpec.ConvOutcome>}
  {==}

# A proof-level invocation groups the arguments of the actual Array API. It
# introduces no storage representation, runtime conversion or alternate kernel.
type Call is Data:
  Call{shape: ConvSpec.Shape,input_length: Nat,weight_length: Nat,bias_length: Nat,
    input: Tree.Tree<F32>,weights: Tree.Tree<F32>,bias: Tree.Tree<F32>,
    input_certificate: Certificate.Certificate<F32,Tree.pack(F32,input)>,
    weight_certificate: Certificate.Certificate<F32,Tree.pack(F32,weights)>,
    bias_certificate: Certificate.Certificate<F32,Tree.pack(F32,bias)>}

def call_problem(call: Call) -> ConvSpec.Problem:
  Call{shape,input_length,weight_length,bias_length,input,weights,bias,input_certificate,weight_certificate,bias_certificate} = call
  problem(shape,input_length,weight_length,bias_length,input,weights,bias)

def call_valid(call: Call) -> Bool:
  Call{shape,input_length,weight_length,bias_length,input,weights,bias,input_certificate,weight_certificate,bias_certificate} = call
  ArrayApi.valid(shape,input_length,weight_length,bias_length)

def call_supported(call: Call) -> Bool:
  Call{shape,input_length,weight_length,bias_length,input,weights,bias,input_certificate,weight_certificate,bias_certificate} = call
  ArrayApi.supported(shape)

def call_capacity(call: Call) -> Bool:
  Call{shape,input_length,weight_length,bias_length,input,weights,bias,input_certificate,weight_certificate,bias_certificate} = call
  Nat.is_le(input_length,U32.to_nat(Tree.capacity(F32,input))) &&
  Nat.is_le(weight_length,U32.to_nat(Tree.capacity(F32,weights))) &&
  Nat.is_le(bias_length,U32.to_nat(Tree.capacity(F32,bias)))

def execute_call(strategy: NativeConv.Strategy,ready: Bool,call: Call) -> ArrayApi.Outcome:
  Call{shape,input_length,weight_length,bias_length,input,weights,bias,input_certificate,weight_certificate,bias_certificate} = call
  ArrayApi.execute(ready,shape,strategy,Tree.pack(F32,input),Tree.pack(F32,weights),Tree.pack(F32,bias))

def machine_call(strategy: NativeConv.Strategy,supported: Bool,call: Call) -> ArrayApi.Outcome:
  Call{shape,input_length,weight_length,bias_length,input,weights,bias,input_certificate,weight_certificate,bias_certificate} = call
  ArrayApi.machine(supported,shape,strategy,
    Storage.Buffer{Tree.pack(F32,input),input_length,input_certificate},Storage.Buffer{Tree.pack(F32,weights),weight_length,weight_certificate},Storage.Buffer{Tree.pack(F32,bias),bias_length,bias_certificate})

def checked_call(strategy: NativeConv.Strategy,valid: Bool,call: Call) -> ArrayApi.Outcome:
  Call{shape,input_length,weight_length,bias_length,input,weights,bias,input_certificate,weight_certificate,bias_certificate} = call
  ArrayApi.checked(valid,shape,strategy,
    Storage.Buffer{Tree.pack(F32,input),input_length,input_certificate},Storage.Buffer{Tree.pack(F32,weights),weight_length,weight_certificate},Storage.Buffer{Tree.pack(F32,bias),bias_length,bias_certificate})

def infer_call(strategy: NativeConv.Strategy,call: Call) -> ArrayApi.Outcome:
  Call{shape,input_length,weight_length,bias_length,input,weights,bias,input_certificate,weight_certificate,bias_certificate} = call
  ArrayApi.infer(shape,strategy,
    Storage.Buffer{Tree.pack(F32,input),input_length,input_certificate},Storage.Buffer{Tree.pack(F32,weights),weight_length,weight_certificate},Storage.Buffer{Tree.pack(F32,bias),bias_length,bias_certificate})

# Rejection/unsupported policy is explicit. An accepted result is independently
# specified by the frozen ordered matrix semantics.
def expected_result(valid: Bool,supported: Bool,capacity: Bool,problem: ConvSpec.Problem) -> Maybe<&2,ConvSpec.ConvOutcome>:
  match valid:
    case False{}: Some{ConvSpec.Rejected{}}
    case True{}:
      match supported:
        case False{}: None{}
        case True{}:
          match capacity:
            case False{}: Some{ConvSpec.Rejected{}}
            case True{}: Some{MatrixSpec.infer(problem)}

def AcceptedExecution(+strategy: NativeConv.Strategy,+call: Call) -> Type:
  @valid: {call_valid(call) == True{} : Bool} ->
  @supported: {call_supported(call) == True{} : Bool} ->
  @capacity: {call_capacity(call) == True{} : Bool} ->
  {observe_result(call_problem(call),execute_call(strategy,True{},call)) == Some{MatrixSpec.infer(call_problem(call))} : Maybe<&2,ConvSpec.ConvOutcome>}

# The shared native strategy theorem discharges execution for every supplied owner.
def accepted_execution(+strategy: NativeConv.Strategy,+call: Call) -> AcceptedExecution(strategy,call):
  Call{+shape,+input_length,+weight_length,+bias_length,+input,+weights,+bias,input_certificate,weight_certificate,bias_certificate} = call
  valid => supported => capacity =>
    +capacity = capacity
    +p = call_problem(call)
    +valid_problem = {
      %validation_matches_buffers(shape,input_length,weight_length,bias_length,input,weights,bias) : {_ == True{} : Bool}
      valid : {ConvSpec.valid(p) == True{} : Bool}}
    +packed_support = {
      %machine_support_matches_problem(Default.limits(),shape,Guards.input(p),Guards.weights(p),Guards.bias(p),Guards.valid(p,valid_problem)) :
        {_ == True{} : Bool}
      supported : {NativeConv.supports_packed_with_limits(Default.limits(),p) == True{} : Bool}}
    +input_fits = Nat.is_le(input_length,U32.to_nat(Tree.capacity(F32,input)))
    +weight_fits = Nat.is_le(weight_length,U32.to_nat(Tree.capacity(F32,weights)))
    +bias_fits = Nat.is_le(bias_length,U32.to_nat(Tree.capacity(F32,bias)))
    +first_two = Boolean.left(input_fits && weight_fits,bias_fits,capacity)
    execute_matches_matrix(strategy,Default.limits(),shape,input_length,weight_length,bias_length,input,weights,bias,
      Boolean.left(input_fits,weight_fits,first_two),Boolean.right(input_fits,weight_fits,first_two),
      Boolean.right(input_fits && weight_fits,bias_fits,capacity),valid_problem,
      packed_support)

# The validation, capacity and unsupported branches are proved once for all
# strategies. The only strategy-specific obligation is accepted execution.
def execution_matches_contract(+strategy: NativeConv.Strategy,+ready: Bool,+call: Call,
  valid: {call_valid(call) == True{} : Bool},supported: {call_supported(call) == True{} : Bool},
  capacity: {call_capacity(call) == ready : Bool},accepted: AcceptedExecution(strategy,call)) ->
  {observe_result(call_problem(call),execute_call(strategy,ready,call)) ==
    expected_result(True{},True{},ready,call_problem(call)) : Maybe<&2,ConvSpec.ConvOutcome>}:
  match ready:
    case False{}:
      Call{shape,input_length,weight_length,bias_length,input,weights,bias,input_certificate,weight_certificate,bias_certificate} = call
      {==}
    case True{}: accepted(valid,supported,capacity)

def capacity_reads_match_execution(+strategy: NativeConv.Strategy,+call: Call) ->
  {machine_call(strategy,True{},call) == execute_call(strategy,call_capacity(call),call) : ArrayApi.Outcome}:
  Call{+shape,+input_length,+weight_length,+bias_length,+input,+weights,+bias,input_certificate,weight_certificate,bias_certificate} = call
  %Equal.sym(Array<F32> & U32,Array.size(F32,Tree.pack(F32,input)),(Tree.pack(F32,input),Tree.capacity(F32,input)),StorageLaws.size(F32,input)) :
    {ArrayApi.capacities(shape,strategy,input_length,weight_length,bias_length,_,Array.size(F32,Tree.pack(F32,weights)),Array.size(F32,Tree.pack(F32,bias))) ==
      execute_call(strategy,call_capacity(call),call) : ArrayApi.Outcome}
  %Equal.sym(Array<F32> & U32,Array.size(F32,Tree.pack(F32,weights)),(Tree.pack(F32,weights),Tree.capacity(F32,weights)),StorageLaws.size(F32,weights)) :
    {ArrayApi.capacities(shape,strategy,input_length,weight_length,bias_length,(Tree.pack(F32,input),Tree.capacity(F32,input)),_,Array.size(F32,Tree.pack(F32,bias))) ==
      execute_call(strategy,call_capacity(call),call) : ArrayApi.Outcome}
  %Equal.sym(Array<F32> & U32,Array.size(F32,Tree.pack(F32,bias)),(Tree.pack(F32,bias),Tree.capacity(F32,bias)),StorageLaws.size(F32,bias)) :
    {ArrayApi.capacities(shape,strategy,input_length,weight_length,bias_length,(Tree.pack(F32,input),Tree.capacity(F32,input)),(Tree.pack(F32,weights),Tree.capacity(F32,weights)),_) ==
      execute_call(strategy,call_capacity(call),call) : ArrayApi.Outcome}
  {==}

def machine_matches_contract(+strategy: NativeConv.Strategy,+supported: Bool,+call: Call,
  valid: {call_valid(call) == True{} : Bool},support: {call_supported(call) == supported : Bool},accepted: AcceptedExecution(strategy,call)) ->
  {observe_result(call_problem(call),machine_call(strategy,supported,call)) ==
    expected_result(True{},supported,call_capacity(call),call_problem(call)) : Maybe<&2,ConvSpec.ConvOutcome>}:
  match supported:
    case False{}:
      Call{shape,input_length,weight_length,bias_length,input,weights,bias,input_certificate,weight_certificate,bias_certificate} = call
      {==}
    case True{}:
      %Equal.sym(ArrayApi.Outcome,machine_call(strategy,True{},call),execute_call(strategy,call_capacity(call),call),
        capacity_reads_match_execution(strategy,call)) :
        {observe_result(call_problem(call),_) == expected_result(True{},True{},call_capacity(call),call_problem(call)) : Maybe<&2,ConvSpec.ConvOutcome>}
      execution_matches_contract(strategy,call_capacity(call),call,valid,support,{==},accepted)

def validation_matches_contract(+strategy: NativeConv.Strategy,+valid: Bool,+call: Call,
  validity: {call_valid(call) == valid : Bool},accepted: AcceptedExecution(strategy,call)) ->
  {observe_result(call_problem(call),checked_call(strategy,valid,call)) ==
    expected_result(valid,call_supported(call),call_capacity(call),call_problem(call)) : Maybe<&2,ConvSpec.ConvOutcome>}:
  match valid:
    case False{}:
      Call{shape,input_length,weight_length,bias_length,input,weights,bias,input_certificate,weight_certificate,bias_certificate} = call
      {==}
    case True{}:
      Call{+shape,+input_length,+weight_length,+bias_length,+input,+weights,+bias,input_certificate,weight_certificate,bias_certificate} = call
      machine_matches_contract(strategy,call_supported(call),call,validity,{==},accepted)

def strategy_inference_matches_contract(+strategy: NativeConv.Strategy,+call: Call,accepted: AcceptedExecution(strategy,call)) ->
  {observe_result(call_problem(call),infer_call(strategy,call)) ==
    expected_result(call_valid(call),call_supported(call),call_capacity(call),call_problem(call)) : Maybe<&2,ConvSpec.ConvOutcome>}:
  Call{+shape,+input_length,+weight_length,+bias_length,+input,+weights,+bias,input_certificate,weight_certificate,bias_certificate} = call
  validation_matches_contract(strategy,call_valid(call),call,{==},accepted)

def array_inference_matches_matrix(+strategy: NativeConv.Strategy,+call: Call) ->
  {observe_result(call_problem(call),infer_call(strategy,call)) ==
    expected_result(call_valid(call),call_supported(call),call_capacity(call),call_problem(call)) : Maybe<&2,ConvSpec.ConvOutcome>}:
  strategy_inference_matches_contract(strategy,call,accepted_execution(strategy,call))
