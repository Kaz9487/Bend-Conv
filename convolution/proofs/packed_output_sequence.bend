# Strategy adapter only. NCHW serialization is shared by every backend.
import Base
import ./input_storage.bend as Inputs
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../../lib/machine_limits.bend as Machine
import ../inference.bend as NativeConv
import ../ordered_reference.bend as OrderedReference
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/row_major_index.bend as RowMajor
import ./prepared_output_readback.bend as PreparedOutputReadback
import ./output_values.bend as Values
import ./matrix_matches_convolution.bend as Matrix
import ./output_sequence_layout.bend as SequenceLayout

type CheckedExecution is Data:
  CheckedExecution{limits: Machine.Limits,problem: ConvSpec.Problem,storage_contract: Inputs.Contract<problem>,depth: U32,
    valid: {ConvSpec.valid(problem) == True{} : Bool},
    supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}}

def execution_problem(execution: CheckedExecution) -> ConvSpec.Problem:
  CheckedExecution{limits,problem,storage_contract,depth,valid,supported} = execution
  problem

def execution_output(execution: CheckedExecution) -> Tree.Tree<F32>:
  CheckedExecution{limits,problem,storage_contract,depth,valid,supported} = execution
  PreparedOutputReadback.output(problem,storage_contract,depth)

def matrix_value_matches_stream(+problem: ConvSpec.Problem,+channel: Nat,+row: Nat,+column: Nat) ->
  {Values.matrix_value(problem,channel,row,column) == (OrderedReference.input_channels(ConvSpec.channels(problem),0n,channel,row,column,problem,0.0) + ConvSpec.bias_at(problem,channel) : F32) : F32}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+input_channel_count,+output_channel_count,+height,+width,+kernel,+stride,+padding} = shape
  %Matrix.kchannels(input_channel_count,0n,channel,row,column,problem,0.0) :
    {Values.matrix_value(problem,channel,row,column) == (_ + ConvSpec.bias_at(problem,channel) : F32) : F32}
  {==}

def output_contents(+execution: CheckedExecution,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,ConvSpec.outputs(execution_problem(execution))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,ConvSpec.out_height(execution_problem(execution))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,ConvSpec.out_width(execution_problem(execution))) == True{} : Bool}) ->
  {Tree.read(F32,execution_output(execution),U32.from_nat(RowMajor.index(channel,ConvSpec.out_height(execution_problem(execution)),ConvSpec.out_width(execution_problem(execution)),row,column))) ==
    (OrderedReference.input_channels(ConvSpec.channels(execution_problem(execution)),0n,channel,row,column,execution_problem(execution),0.0) + ConvSpec.bias_at(execution_problem(execution),channel) : F32) : F32}:
  CheckedExecution{+limits,+problem,+storage_contract,+depth,valid,supported} = execution
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+input_channel_count,+output_channel_count,+height,+width,+kernel,+stride,+padding} = shape
  Equal.trans(F32,
    Tree.read(F32,PreparedOutputReadback.output(problem,storage_contract,depth),U32.from_nat(RowMajor.index(channel,ConvSpec.out_height(problem),ConvSpec.out_width(problem),row,column))),
    Values.matrix_value(problem,channel,row,column),
    (OrderedReference.input_channels(ConvSpec.channels(problem),0n,channel,row,column,problem,0.0) + ConvSpec.bias_at(problem,channel) : F32),
    PreparedOutputReadback.every_output_matches_matrix(limits,problem,storage_contract,depth,channel,row,column,valid,supported,channel_inside,row_inside,column_inside),
    matrix_value_matches_stream(problem,channel,row,column))

def complete_output_matches_matrix(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+depth: U32,
  valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,Nat.mul(ConvSpec.outputs(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))),0,PreparedOutputReadback.output(problem,storage_contract,depth)) == MatrixSpec.values(problem) : List<&2,F32>}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+input_channel_count,+output_channel_count,+height,+width,+kernel,+stride,+padding} = shape
  %Equal.sym(List<&2,F32>,MatrixSpec.values(problem),OrderedReference.values(problem),Matrix.channels(output_channel_count,0n,problem)) :
    {Tree.read_sequence(~F32,Nat.mul(output_channel_count,Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))),0,PreparedOutputReadback.output(problem,storage_contract,depth)) == _ : List<&2,F32>}
  SequenceLayout.readback_channels(~CheckedExecution,~execution_problem,~execution_output,~output_contents,output_channel_count,0n,
    CheckedExecution{limits,problem,storage_contract,depth,valid,supported},Order.reflexive(output_channel_count))
