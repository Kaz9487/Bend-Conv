import ./direct_kernel_refinement.bend as KernelLaws
import ./execution_storage.bend as ExecutionStorage
import ./output_readback.bend as ProofsOutputReadback
import ./scalar_output_coordinates.bend as OutputCoordinates
import ../../lib/proofs/array_allocation_bounds.bend as ArrayAllocation
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/traversal_output_refinement.bend as OutputLaws
import ../shape.bend as ConvShape
# Complete scalar backend: actual allocation, loops, stores and readback.
import Base
import ./input_storage.bend as Inputs
import ../../lib/storage_buffer.bend as Storage
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../scalar_kernel.bend as ScalarConv
import ../reference.bend as Reference
import ../ordered_reference.bend as OrderedReference
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./scalar_tensor_lengths.bend as Lengths
import ./output_sequence_layout.bend as SequenceLayout
import ./list_array_storage.bend as ListStorage
import ./supported_padded_coordinates.bend as Padded
import ./matrix_matches_convolution.bend as Matrix
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order

def native_count(+problem: ConvSpec.Problem) -> U32:
  (U32.from_nat(Guards.output_channel_count(problem)) * U32.from_nat(ConvSpec.out_height(problem)) * U32.from_nat(ConvSpec.out_width(problem)) : U32)

type CheckedExecution is Data:
  CheckedExecution{limits: Machine.Limits,problem: ConvSpec.Problem,inputs: Inputs.Contract<problem>,
    valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool},
    allocation: ArrayAllocation.Allocation(native_count(problem))}

def execution_problem(execution: CheckedExecution) -> ConvSpec.Problem:
  CheckedExecution{limits,problem,inputs,valid,supported,allocation} = execution
  problem

def execution_output(execution: CheckedExecution) -> Tree.Tree<F32>:
  CheckedExecution{limits,+problem,+inputs,valid,supported,allocation} = execution
  ExecutionStorage.direct_result(ConvShape.from_problem(problem),Inputs.input_tree(problem,inputs),Inputs.weight_tree(problem,inputs),Inputs.bias_tree(problem,inputs))

def output_allocation(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,
  +supported: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) -> ArrayAllocation.Allocation(native_count(problem)):
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+channels,+outputs,+height,+width,+kernel,+stride,+padding} = shape
  Guards.Valid{channels_positive,outputs_positive,h,w,k,s,ph,pw,input_length,weight_length,bias_length} = valid
  +count = Nat.mul(outputs,Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)))
  +exact = Lengths.output_length(limits,problem,valid,supported)
  ArrayAllocation.allocated(native_count(problem),Machine.capacity(limits),Limits.capacity_fits(limits),Limits.capacity_allocatable(limits),
    %Equal.sym(Nat,U32.to_nat(native_count(problem)),count,exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Support.positive_mul(outputs,Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)),outputs_positive,
      Support.positive_mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem),{==},{==})),
    %Equal.sym(Nat,U32.to_nat(native_count(problem)),count,exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    Padded.scalar_output_bound(limits,problem,supported))

def output_contents(+execution: CheckedExecution,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,ConvSpec.outputs(execution_problem(execution))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,ConvSpec.out_height(execution_problem(execution))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,ConvSpec.out_width(execution_problem(execution))) == True{} : Bool}) ->
  {Tree.read(F32,execution_output(execution),U32.from_nat(RowMajor.index(channel,ConvSpec.out_height(execution_problem(execution)),ConvSpec.out_width(execution_problem(execution)),row,column))) ==
    (OrderedReference.input_channels(ConvSpec.channels(execution_problem(execution)),0n,channel,row,column,execution_problem(execution),0.0) + ConvSpec.bias_at(execution_problem(execution),channel) : F32) : F32}:
  CheckedExecution{+limits,+problem,+inputs,+valid,+supported,allocation} = execution
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+channels,+outputs,+height,+width,+kernel,+stride,+padding} = shape
  Shape.Both{+depth_bound,+room} = allocation
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  +pixel = {OutputCoordinates.OutputPixel{channel,row,column,channel_inside,row_inside,column_inside} : OutputCoordinates.OutputPixel<problem>}
  +evaluation = {KernelLaws.Evaluation{limits,problem,inputs,pixel,valid,supported} : KernelLaws.Evaluation}
  +count = Nat.mul(outputs,Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)))
  +depth = Storage.allocation_depth(native_count(problem))
  +index = RowMajor.index(channel,ConvSpec.out_height(problem),ConvSpec.out_width(problem),row,column)
  +exact = Lengths.output_length(limits,problem,Guards.valid(problem,valid),Support.scalar(limits,problem,supported))
  %Matrix.kchannels(channels,0n,channel,row,column,problem,0.0) :
    {Tree.read(F32,execution_output(execution),U32.from_nat(index)) == (_ + ConvSpec.bias_at(problem,channel) : F32) : F32}
  %KernelLaws.value_matches_matrix(evaluation) : {Tree.read(F32,execution_output(execution),U32.from_nat(index)) == _ : F32}
  ProofsOutputReadback.direct_complete_readback(ProofsOutputReadback.DirectQuery{ConvShape.from_problem(problem),KernelLaws.sources(evaluation),U32.to_nat(native_count(problem)),index,depth,depth_bound,room,
      %Equal.sym(Nat,U32.to_nat(native_count(problem)),count,exact) : {Nat.is_lt(index,_) == True{} : Bool}
      RowMajor.in_bounds(channel,outputs,ConvSpec.out_height(problem),ConvSpec.out_width(problem),row,column,channel_inside,row_inside,column_inside)},
    Tree.fresh(F32,depth,0.0),Shape.fresh(F32,depth,0.0))

# Public strategy contract observes contents. No canonical allocator identity.
def run_readback_matches_values(+limits: Machine.Limits,+problem: ConvSpec.Problem,+inputs: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool}) ->
  {Storage.read_list(F32,Nat.mul(ConvSpec.outputs(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))),0,
      ScalarConv.run(ConvShape.from_problem(problem),Inputs.input_owner(problem,inputs),Inputs.weight_owner(problem,inputs),Inputs.bias_owner(problem,inputs))) ==
    OrderedReference.values(problem) : List<&2,F32>}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+channels,+outputs,+height,+width,+kernel,+stride,+padding} = shape
  +execution = {CheckedExecution{limits,problem,inputs,valid,supported,output_allocation(limits,problem,Guards.valid(problem,valid),Support.scalar(limits,problem,supported))} : CheckedExecution}
  +count = Nat.mul(outputs,Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)))
  %Equal.sym(Array<F32>,ScalarConv.run(ConvShape.from_problem(problem),Inputs.input_owner(problem,inputs),Inputs.weight_owner(problem,inputs),Inputs.bias_owner(problem,inputs)),
    Tree.pack(F32,execution_output(execution)),ExecutionStorage.direct_entry_matches_storage(ConvShape.from_problem(problem),Inputs.input_tree(problem,inputs),Inputs.weight_tree(problem,inputs),Inputs.bias_tree(problem,inputs))) :
    {Storage.read_list(F32,count,0,_) == OrderedReference.values(problem) : List<&2,F32>}
  %Equal.sym(List<&2,F32>,Storage.read_list(F32,count,0,Tree.pack(F32,execution_output(execution))),Tree.read_sequence(~F32,count,0,execution_output(execution)),
    StorageLaws.read_sequence(~F32,count,0,execution_output(execution))) : {_ == OrderedReference.values(problem) : List<&2,F32>}
  SequenceLayout.readback_channels(~CheckedExecution,~execution_problem,~execution_output,~output_contents,outputs,0n,execution,Order.reflexive(outputs))

def native_matches_reference(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool}) ->
  {ScalarConv.native(problem) == Reference.finish(True{},problem) : ConvSpec.ConvOutcome}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+input_channel_count,+output_channel_count,+height,+width,+kernel,+stride,+padding} = shape
  +inputs = Inputs.from_lists(limits,problem,valid,supported)
  +values = OrderedReference.values(problem)
  +count = Nat.mul(output_channel_count,Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)))
  %Equal.sym(Array<F32>,ScalarConv.pack(input),Inputs.input_owner(problem,inputs),ListStorage.packed(input)) :
    {ConvSpec.Accepted{ConvSpec.out_height(problem),ConvSpec.out_width(problem),Storage.read_list(F32,count,0,
      ScalarConv.run(ConvShape.from_problem(problem),_,ScalarConv.pack(weights),ScalarConv.pack(bias)))} == Reference.finish(True{},problem) : ConvSpec.ConvOutcome}
  %Equal.sym(Array<F32>,ScalarConv.pack(weights),Inputs.weight_owner(problem,inputs),ListStorage.packed(weights)) :
    {ConvSpec.Accepted{ConvSpec.out_height(problem),ConvSpec.out_width(problem),Storage.read_list(F32,count,0,
      ScalarConv.run(ConvShape.from_problem(problem),Inputs.input_owner(problem,inputs),_,ScalarConv.pack(bias)))} == Reference.finish(True{},problem) : ConvSpec.ConvOutcome}
  %Equal.sym(Array<F32>,ScalarConv.pack(bias),Inputs.bias_owner(problem,inputs),ListStorage.packed(bias)) :
    {ConvSpec.Accepted{ConvSpec.out_height(problem),ConvSpec.out_width(problem),Storage.read_list(F32,count,0,
      ScalarConv.run(ConvShape.from_problem(problem),Inputs.input_owner(problem,inputs),Inputs.weight_owner(problem,inputs),_))} == Reference.finish(True{},problem) : ConvSpec.ConvOutcome}
  %Equal.sym(List<&2,F32>,Storage.read_list(F32,count,0,
      ScalarConv.run(ConvShape.from_problem(problem),Inputs.input_owner(problem,inputs),Inputs.weight_owner(problem,inputs),Inputs.bias_owner(problem,inputs))),values,
    run_readback_matches_values(limits,problem,inputs,valid,supported)) :
    {ConvSpec.Accepted{ConvSpec.out_height(problem),ConvSpec.out_width(problem),_} == Reference.finish(True{},problem) : ConvSpec.ConvOutcome}
  {==}

def chosen_backend_matches_reference(+use_native: Bool,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  supported: {ScalarConv.supported(problem) == use_native : Bool}) ->
  {ScalarConv.choose(use_native,problem) == Reference.finish(True{},problem) : ConvSpec.ConvOutcome}:
  match use_native:
    case False{}:
      %Equal.sym(Bool,ConvSpec.valid(problem),True{},valid) : {Reference.finish(_,problem) == Reference.finish(True{},problem) : ConvSpec.ConvOutcome}
      {==}
    case True{}:
      native_matches_reference(Default.limits(),problem,valid,supported)

def validated_matches_reference(+is_valid: Bool,+problem: ConvSpec.Problem,valid: {ConvSpec.valid(problem) == is_valid : Bool}) ->
  {ScalarConv.validated(is_valid,problem) == Reference.finish(is_valid,problem) : ConvSpec.ConvOutcome}:
  match is_valid:
    case False{}: {==}
    case True{}: chosen_backend_matches_reference(ScalarConv.supported(problem),problem,valid,{==})

def scalar_inference_matches_matrix(+problem: ConvSpec.Problem) -> {ScalarConv.infer(problem) == MatrixSpec.infer(problem) : ConvSpec.ConvOutcome}:
  Equal.trans(ConvSpec.ConvOutcome,ScalarConv.infer(problem),Reference.infer(problem),MatrixSpec.infer(problem),
    validated_matches_reference(ConvSpec.valid(problem),problem,{==}),
    Equal.sym(ConvSpec.ConvOutcome,MatrixSpec.infer(problem),Reference.infer(problem),Matrix.validated(ConvSpec.valid(problem),problem)))

def scalar_inference_when_valid(+is_valid: Bool,+problem: ConvSpec.Problem,
  valid: {ConvSpec.valid(problem) == is_valid : Bool}) ->
  {ScalarConv.infer(problem) == MatrixSpec.finish(is_valid,problem) : ConvSpec.ConvOutcome}:
  %valid :
    {ScalarConv.infer(problem) == MatrixSpec.finish(_,problem) : ConvSpec.ConvOutcome}
  scalar_inference_matches_matrix(problem)
