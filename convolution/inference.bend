import ./shape.bend as ConvShape
import ../lib/machine_limits.bend as Machine
import ../lib/default_machine_limits.bend as Default
# Packed convolution; full ordered-matrix refinement is in native_inference_law.
import Base
import ../lib/storage_buffer.bend as Storage
import ./specification.bend as ConvSpec
import ./scalar_kernel.bend as ScalarConv
import ../lib/kernel_array_pipeline.bend as ArrayPipeline
import ../lib/kernel_pipeline.bend as Pipeline
import ../lib/traversal_array.bend as Traversal
import ../lib/math_fp32.bend as Fp32
import ./planning.bend as Planning
import ../lib/kernel_planning.bend as KernelPlanning

# Strategies share dimensions, owners and the exact ordered FP32 specification.
type Strategy is Data:
  Direct{}
  Packed{split_depth: U32}
  Automatic{worker_limit: U32,workspace_budget: Maybe<&2,Nat>}

def automatic(worker_limit: U32,workspace_budget: Maybe<&2,Nat>) -> Strategy:
  Automatic{worker_limit,workspace_budget}

def selected(plan: KernelPlanning.Plan) -> Strategy:
  KernelPlanning.Plan{packed,depth,workspace,time} = plan
  Bool.pick(Strategy,packed,Packed{depth},Direct{})

def activate(context: Unit,value: F32) -> F32:
  Fp32.activate(value,True{})

def activate_output(enabled: Bool,count: Nat,output: Array<F32>) -> Array<F32>:
  match enabled:
    case False{}: output
    case True{}: Traversal.map_in_place(~F32,~Unit,~activate,count,0,Unit{},output)

# The six-row kernel stores accumulator plus bias. Activation, when requested,
# is one contiguous pass over the finished output, as for Direct.
def work(+shape: ConvShape.Shape) -> Pipeline.Work:
  ConvShape.Shape{+input_channels,+output_channels,height,width,+kernel_size,stride,padding,+output_height,+output_width} = shape
  Pipeline.Work{output_channels,height,width,kernel_size,stride,padding,output_width,(output_height * output_width : U32),(input_channels * kernel_size * kernel_size : U32)}

def output_count(shape: ConvShape.Shape) -> Nat:
  ConvShape.Shape{input_channel_count,+output_channel_count,h,w,k,s,p,+output_height,+output_width} = shape
  U32.to_nat((output_channel_count * output_height * output_width : U32))

def run(+shape: ConvShape.Shape,depth: U32,input_values: Array<F32>,weights: Array<F32>,bias: Array<F32>) -> Array<F32>:
  ArrayPipeline.run(depth,work(shape),input_values,weights,bias)

def execute_selected(packed: Bool,depth: U32,apply_activation: Bool,+shape: ConvShape.Shape,
  input: Array<F32>,weights: Array<F32>,bias: Array<F32>) -> Array<F32>:
  match packed:
    case False{}:
      ConvShape.Shape{input_channel_count,+output_channel_count,h,w,k,s,p,+output_height,+output_width} = shape
      activate_output(apply_activation,U32.to_nat((output_channel_count * output_height * output_width : U32)),ScalarConv.run(shape,input,weights,bias))
    case True{}:
      activate_output(apply_activation,output_count(shape),run(shape,depth,input,weights,bias))

def execute_plan(plan: KernelPlanning.Plan,apply_activation: Bool,shape: ConvShape.Shape,input: Array<F32>,weights: Array<F32>,bias: Array<F32>) -> Array<F32>:
  KernelPlanning.Plan{packed,depth,workspace,time} = plan
  execute_selected(packed,depth,apply_activation,shape,input,weights,bias)

def execute_sized(worker_limit: U32,workspace_budget: Maybe<&2,Nat>,apply_activation: Bool,+shape: ConvShape.Shape,
  input_sized: Array<F32> & U32,weights: Array<F32>,bias_sized: Array<F32> & U32) -> Array<F32>:
  (input,input_capacity) = input_sized
  (bias,bias_capacity) = bias_sized
  execute_plan(Planning.plan(shape,worker_limit,workspace_budget,KernelPlanning.Owners{input_capacity,bias_capacity}),apply_activation,shape,input,weights,bias)

# array_api performs semantic, machine and capacity guards before this internal
# executor. Planning reads the real owner capacities only after those guards.
def execute(strategy: Strategy,apply_activation: Bool,shape: ConvShape.Shape,
  input: Array<F32>,weights: Array<F32>,bias: Array<F32>) -> Array<F32>:
  match strategy:
    case Direct{}: execute_selected(False{},0,apply_activation,shape,input,weights,bias)
    case Packed{depth}: execute_selected(True{},depth,apply_activation,shape,input,weights,bias)
    case Automatic{workers,budget}:
      execute_sized(workers,budget,apply_activation,shape,Array.size(F32,input),weights,Array.size(F32,bias))

def supports_packed_with_limits(+limits: Machine.Limits,problem: ConvSpec.Problem) -> Bool:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvShape.supports_packed(limits,shape,ConvSpec.length(input),ConvSpec.length(weights))

def supports_packed_convolution(+p: ConvSpec.Problem) -> Bool:
  supports_packed_with_limits(Default.limits(),p)

def infer_packed(+problem: ConvSpec.Problem, depth: U32) -> ConvSpec.ConvOutcome:
  ConvSpec.Problem{shape,input_values,weights,bias} = problem
  +output_height = ConvSpec.out_height(problem)
  +output_width = ConvSpec.out_width(problem)
  native_shape = ConvShape.from_spec(shape)
  input_array = ScalarConv.pack(input_values)
  weight_array = ScalarConv.pack(weights)
  bias_array = ScalarConv.pack(bias)
  output_array = run(native_shape,depth,input_array,weight_array,bias_array)
  output_count = Nat.mul(ConvSpec.outputs(problem),Nat.mul(output_height,output_width))
  ConvSpec.Accepted{output_height,output_width,Storage.read_list(F32,output_count,0,output_array)}

def select_backend(use_packed: Bool, +problem: ConvSpec.Problem, depth: U32) -> ConvSpec.ConvOutcome:
  match use_packed:
    case False{}:
      ScalarConv.infer(problem)
    case True{}:
      infer_packed(problem,depth)

def infer_if_valid(is_valid: Bool, +problem: ConvSpec.Problem, depth: U32) -> ConvSpec.ConvOutcome:
  match is_valid:
    case False{}:
      ConvSpec.Rejected{}
    case True{}:
      select_backend(supports_packed_convolution(problem),problem,depth)

def infer_with_split_depth(+problem: ConvSpec.Problem, depth: U32) -> ConvSpec.ConvOutcome:
  infer_if_valid(ConvSpec.valid(problem),problem,depth)

def plan_problem(+problem: ConvSpec.Problem,workers: U32,budget: Maybe<&2,Nat>) -> KernelPlanning.Plan:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  Planning.plan(ConvShape.from_spec(shape),workers,budget,KernelPlanning.Owners{
    U32.from_nat(KernelPlanning.capacity(U32.from_nat(ConvSpec.length(input)))),
    U32.from_nat(KernelPlanning.capacity(U32.from_nat(ConvSpec.length(bias))))})

def infer_plan(plan: KernelPlanning.Plan,problem: ConvSpec.Problem) -> ConvSpec.ConvOutcome:
  KernelPlanning.Plan{packed,depth,workspace,time} = plan
  select_backend(packed,problem,depth)

def plan_supported(supported: Bool,+problem: ConvSpec.Problem,workers: U32,budget: Maybe<&2,Nat>) -> ConvSpec.ConvOutcome:
  match supported:
    case False{}: ScalarConv.infer(problem)
    case True{}: infer_plan(plan_problem(problem,workers,budget),problem)

def plan_valid(valid: Bool,+problem: ConvSpec.Problem,workers: U32,budget: Maybe<&2,Nat>) -> ConvSpec.ConvOutcome:
  match valid:
    case False{}: ConvSpec.Rejected{}
    case True{}: plan_supported(supports_packed_convolution(problem),problem,workers,budget)

# A requested strategy never bypasses semantic validation or machine guards.
def infer_with_strategy(strategy: Strategy,+problem: ConvSpec.Problem) -> ConvSpec.ConvOutcome:
  match strategy:
    case Direct{}: ScalarConv.infer(problem)
    case Packed{depth}: infer_with_split_depth(problem,depth)
    case Automatic{workers,budget}: plan_valid(ConvSpec.valid(problem),problem,workers,budget)

def infer(problem: ConvSpec.Problem) -> ConvSpec.ConvOutcome:
  infer_with_strategy(Automatic{1,None{}},problem)

