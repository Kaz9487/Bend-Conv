# Complete native inference: invalid input, scalar fallback, and packed execution.
import Base
import ../../lib/storage_buffer.bend as Storage
import ../../lib/machine_limits.bend as Machine
import ../shape.bend as ConvShape
import ../ordered_reference.bend as OrderedReference
import ./input_storage.bend as Inputs
import ./concrete_support_guard.bend as Support
import ./matrix_matches_convolution.bend as Matrix
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../inference.bend as NativeConv
import ../../lib/default_machine_limits.bend as Default
import ./packed_inference_refinement.bend as Packed
import ./scalar_inference_refinement.bend as ScalarInferenceRefinement
import ../planning.bend as Planning
import ../../lib/kernel_planning.bend as KernelPlanning
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws

law native_inference_matches_matrix:
  for +problem: ConvSpec.Problem
  {NativeConv.infer(problem) == MatrixSpec.infer(problem) : ConvSpec.ConvOutcome}

def default_packed_inference_matches_matrix(+problem: ConvSpec.Problem,+depth: U32,
  valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_convolution(problem) == True{} : Bool}) ->
  {NativeConv.infer_packed(problem,depth) == MatrixSpec.finish(True{},problem) : ConvSpec.ConvOutcome}:
  Packed.packed_inference_matches_matrix(Default.limits(),problem,depth,valid,supported)

def selected_backend_matches_matrix(+use_packed: Bool,+problem: ConvSpec.Problem,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_convolution(problem) == use_packed : Bool}) ->
  {NativeConv.select_backend(use_packed,problem,depth) == MatrixSpec.finish(True{},problem) : ConvSpec.ConvOutcome}:
  match use_packed:
    case False{}: ScalarInferenceRefinement.scalar_inference_when_valid(True{},problem,valid)
    case True{}: default_packed_inference_matches_matrix(problem,depth,valid,supported)

def validated_inference_matches_matrix(+is_valid: Bool,+problem: ConvSpec.Problem,+depth: U32,
  valid: {ConvSpec.valid(problem) == is_valid : Bool}) ->
  {NativeConv.infer_if_valid(is_valid,problem,depth) == MatrixSpec.finish(is_valid,problem) : ConvSpec.ConvOutcome}:
  match is_valid:
    case False{}: {==}
    case True{}: selected_backend_matches_matrix(NativeConv.supports_packed_convolution(problem),problem,depth,valid,{==})

def inference_with_any_split_depth_matches_matrix(+problem: ConvSpec.Problem,+depth: U32) ->
  {NativeConv.infer_with_split_depth(problem,depth) == MatrixSpec.infer(problem) : ConvSpec.ConvOutcome}:
  validated_inference_matches_matrix(ConvSpec.valid(problem),problem,depth,{==})

def selected_plan_matches(+packed: Bool,+depth: U32,+problem: ConvSpec.Problem,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_convolution(problem) == True{} : Bool}) ->
  {NativeConv.select_backend(packed,problem,depth) == MatrixSpec.finish(True{},problem) : ConvSpec.ConvOutcome}:
  match packed:
    case False{}: ScalarInferenceRefinement.scalar_inference_when_valid(True{},problem,valid)
    case True{}: default_packed_inference_matches_matrix(problem,depth,valid,supported)

def planned_inference_matches(plan: KernelPlanning.Plan,+problem: ConvSpec.Problem,
  valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_convolution(problem) == True{} : Bool}) ->
  {NativeConv.infer_plan(plan,problem) == MatrixSpec.finish(True{},problem) : ConvSpec.ConvOutcome}:
  KernelPlanning.Plan{packed,depth,workspace,time} = plan
  selected_plan_matches(packed,depth,problem,valid,supported)

def supported_plan_matches(+supported: Bool,+problem: ConvSpec.Problem,workers: U32,budget: Maybe<&2,Nat>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},support: {NativeConv.supports_packed_convolution(problem) == supported : Bool}) ->
  {NativeConv.plan_supported(supported,problem,workers,budget) == MatrixSpec.finish(True{},problem) : ConvSpec.ConvOutcome}:
  match supported:
    case False{}: ScalarInferenceRefinement.scalar_inference_when_valid(True{},problem,valid)
    case True{}: planned_inference_matches(NativeConv.plan_problem(problem,workers,budget),problem,valid,support)

def validated_plan_matches(+valid: Bool,+problem: ConvSpec.Problem,workers: U32,budget: Maybe<&2,Nat>,
  validity: {ConvSpec.valid(problem) == valid : Bool}) ->
  {NativeConv.plan_valid(valid,problem,workers,budget) == MatrixSpec.finish(valid,problem) : ConvSpec.ConvOutcome}:
  match valid:
    case False{}: {==}
    case True{}: supported_plan_matches(NativeConv.supports_packed_convolution(problem),problem,workers,budget,validity,{==})

def native_inference_matches_matrix(problem):
  validated_plan_matches(ConvSpec.valid(problem),problem,1,None{},{==})

# Every policy may choose either implementation without reopening storage,
# packing, coverage or FP32 reduction proofs.
def every_strategy_matches_matrix(strategy: NativeConv.Strategy,+problem: ConvSpec.Problem) ->
  {NativeConv.infer_with_strategy(strategy,problem) == MatrixSpec.infer(problem) : ConvSpec.ConvOutcome}:
  match strategy:
    case NativeConv.Direct{}: ScalarInferenceRefinement.scalar_inference_matches_matrix(problem)
    case NativeConv.Packed{depth}: inference_with_any_split_depth_matches_matrix(problem,depth)
    case NativeConv.Automatic{workers,budget}: validated_plan_matches(ConvSpec.valid(problem),problem,workers,budget,{==})

# Both actual Array strategies refine one ordered matrix result. The caller
# supplies contents/capacity evidence; it never supplies the desired output.
def selected_readback_matches_matrix(+packed: Bool,+depth: U32,+limits: Machine.Limits,+problem: ConvSpec.Problem,+inputs: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Storage.read_list(F32,Nat.mul(ConvSpec.outputs(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))),0,
    NativeConv.execute_selected(packed,depth,False{},ConvShape.from_problem(problem),Inputs.input_owner(problem,inputs),Inputs.weight_owner(problem,inputs),Inputs.bias_owner(problem,inputs))) == MatrixSpec.values(problem) : List<&2,F32>}:
  match packed:
    case False{}:
      ConvSpec.Problem{shape,+input,+weights,+bias} = problem
      ConvSpec.Shape{+input_channels,+output_channels,+height,+width,+kernel,+stride,+padding} = shape
      %Equal.sym(List<&2,F32>,MatrixSpec.values(problem),OrderedReference.values(problem),Matrix.channels(ConvSpec.outputs(problem),0n,problem)) :
        {Storage.read_list(F32,Nat.mul(ConvSpec.outputs(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))),0,
          NativeConv.execute(NativeConv.Direct{},False{},ConvShape.from_problem(problem),Inputs.input_owner(problem,inputs),Inputs.weight_owner(problem,inputs),Inputs.bias_owner(problem,inputs))) == _ : List<&2,F32>}
      ScalarInferenceRefinement.run_readback_matches_values(limits,problem,inputs,valid,Support.scalar_guard(limits,problem,supported))
    case True{}: Packed.run_readback_matches_matrix(limits,problem,inputs,depth,valid,supported)

def values_of(+problem: ConvSpec.Problem,output: Array<F32>) -> List<&2,F32>:
  Storage.read_list(F32,Nat.mul(ConvSpec.outputs(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))),0,output)

def plan_readback(plan: KernelPlanning.Plan,+limits: Machine.Limits,+problem: ConvSpec.Problem,+inputs: Inputs.Contract<problem>,
  valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {values_of(problem,NativeConv.execute_plan(plan,False{},ConvShape.from_problem(problem),Inputs.input_owner(problem,inputs),Inputs.weight_owner(problem,inputs),Inputs.bias_owner(problem,inputs))) == MatrixSpec.values(problem) : List<&2,F32>}:
  KernelPlanning.Plan{packed,depth,workspace,time} = plan
  selected_readback_matches_matrix(packed,depth,limits,problem,inputs,valid,supported)

def strategy_readback_matches_matrix(strategy: NativeConv.Strategy,+limits: Machine.Limits,+problem: ConvSpec.Problem,+inputs: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {values_of(problem,NativeConv.execute(strategy,False{},ConvShape.from_problem(problem),Inputs.input_owner(problem,inputs),Inputs.weight_owner(problem,inputs),Inputs.bias_owner(problem,inputs))) == MatrixSpec.values(problem) : List<&2,F32>}:
  match strategy:
    case NativeConv.Direct{}: selected_readback_matches_matrix(False{},0,limits,problem,inputs,valid,supported)
    case NativeConv.Packed{depth}: selected_readback_matches_matrix(True{},depth,limits,problem,inputs,valid,supported)
    case NativeConv.Automatic{+workers,+budget}:
      %Equal.sym(Array<F32> & U32,Array.size(F32,Inputs.input_owner(problem,inputs)),(Inputs.input_owner(problem,inputs),Tree.capacity(F32,Inputs.input_tree(problem,inputs))),StorageLaws.size(F32,Inputs.input_tree(problem,inputs))) :
        {values_of(problem,NativeConv.execute_sized(workers,budget,False{},ConvShape.from_problem(problem),_,Inputs.weight_owner(problem,inputs),Array.size(F32,Inputs.bias_owner(problem,inputs)))) == MatrixSpec.values(problem) : List<&2,F32>}
      %Equal.sym(Array<F32> & U32,Array.size(F32,Inputs.bias_owner(problem,inputs)),(Inputs.bias_owner(problem,inputs),Tree.capacity(F32,Inputs.bias_tree(problem,inputs))),StorageLaws.size(F32,Inputs.bias_tree(problem,inputs))) :
        {values_of(problem,NativeConv.execute_sized(workers,budget,False{},ConvShape.from_problem(problem),(Inputs.input_owner(problem,inputs),Tree.capacity(F32,Inputs.input_tree(problem,inputs))),Inputs.weight_owner(problem,inputs),_)) == MatrixSpec.values(problem) : List<&2,F32>}
      plan_readback(Planning.plan(ConvShape.from_problem(problem),workers,budget,KernelPlanning.Owners{Tree.capacity(F32,Inputs.input_tree(problem,inputs)),Tree.capacity(F32,Inputs.bias_tree(problem,inputs))}),limits,problem,inputs,valid,supported)
