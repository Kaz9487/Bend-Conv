# Full packed entry, from List inputs through actual Array execution and unpack.
import Base
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../shape.bend as ConvShape
import ./microkernel_inference_refinement.bend as Inference
import ./prepared_output_readback.bend as PreparedOutputReadback
import ./input_storage.bend as Inputs
import ./concrete_support_guard.bend as Support
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../../lib/machine_limits.bend as Machine
import ../inference.bend as NativeConv
import ./list_array_storage.bend as ListStorage
import ../scalar_kernel.bend as ScalarConv
import ./packed_output_sequence.bend as PackedOutputSequence

# The complete native Array computation consumes the same source contract as
# Direct. Physical capacities and allocation history are not part of the result.
def run_readback_matches_matrix(+limits: Machine.Limits,+problem: ConvSpec.Problem,+inputs: Inputs.Contract<problem>,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Storage.read_list(F32,Nat.mul(ConvSpec.outputs(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))),0,
      NativeConv.run(ConvShape.from_problem(problem),depth,Inputs.input_owner(problem,inputs),Inputs.weight_owner(problem,inputs),Inputs.bias_owner(problem,inputs))) == MatrixSpec.values(problem) : List<&2,F32>}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+input_channels,+output_channels,+height,+width,+kernel,+stride,+padding} = shape
  +count = Nat.mul(output_channels,Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)))
  +output = PreparedOutputReadback.output(problem,inputs,depth)
  %Equal.sym(Array<F32>,NativeConv.run(ConvShape.from_problem(problem),depth,Inputs.input_owner(problem,inputs),Inputs.weight_owner(problem,inputs),Inputs.bias_owner(problem,inputs)),
    Tree.pack(F32,output),Inference.native_run_matches(limits,problem,inputs,depth,valid,supported)) :
    {Storage.read_list(F32,count,0,_) == MatrixSpec.values(problem) : List<&2,F32>}
  %Equal.sym(List<&2,F32>,Storage.read_list(F32,count,0,Tree.pack(F32,output)),Tree.read_sequence(~F32,count,0,output),StorageLaws.read_sequence(~F32,count,0,output)) :
    {_ == MatrixSpec.values(problem) : List<&2,F32>}
  PackedOutputSequence.complete_output_matches_matrix(limits,problem,inputs,depth,valid,supported)

# List is only an input adapter to the same actual Array result theorem.
def packed_inference_matches_matrix(+limits: Machine.Limits,+problem: ConvSpec.Problem,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {NativeConv.infer_packed(problem,depth) == MatrixSpec.finish(True{},problem) : ConvSpec.ConvOutcome}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+input_channels,+output_channels,+height,+width,+kernel,+stride,+padding} = shape
  +inputs = Inputs.from_lists(limits,problem,valid,Support.scalar_guard(limits,problem,supported))
  +count = Nat.mul(output_channels,Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)))
  %run_readback_matches_matrix(limits,problem,inputs,depth,valid,supported) :
    {NativeConv.infer_packed(problem,depth) == ConvSpec.Accepted{ConvSpec.out_height(problem),ConvSpec.out_width(problem),_} : ConvSpec.ConvOutcome}
  %ListStorage.packed(input) :
    {NativeConv.infer_packed(problem,depth) == ConvSpec.Accepted{ConvSpec.out_height(problem),ConvSpec.out_width(problem),
      Storage.read_list(F32,count,0,NativeConv.run(ConvShape.from_problem(problem),depth,_,Inputs.weight_owner(problem,inputs),Inputs.bias_owner(problem,inputs)))} : ConvSpec.ConvOutcome}
  %ListStorage.packed(weights) :
    {NativeConv.infer_packed(problem,depth) == ConvSpec.Accepted{ConvSpec.out_height(problem),ConvSpec.out_width(problem),
      Storage.read_list(F32,count,0,NativeConv.run(ConvShape.from_problem(problem),depth,ScalarConv.pack(input),_,Inputs.bias_owner(problem,inputs)))} : ConvSpec.ConvOutcome}
  %ListStorage.packed(bias) :
    {NativeConv.infer_packed(problem,depth) == ConvSpec.Accepted{ConvSpec.out_height(problem),ConvSpec.out_width(problem),
      Storage.read_list(F32,count,0,NativeConv.run(ConvShape.from_problem(problem),depth,ScalarConv.pack(input),ScalarConv.pack(weights),_))} : ConvSpec.ConvOutcome}
  {==}
