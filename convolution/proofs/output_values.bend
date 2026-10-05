# Complete FP32 output value: ordered reduction followed by one bias addition.
import Base
import ./input_storage.bend as Inputs
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../../lib/machine_limits.bend as Machine
import ../inference.bend as NativeConv
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./list_array_storage.bend as ListStorage

def matrix_dot(indices: List<&2,MatrixSpec.Reduction>,p: ConvSpec.Problem,output_channel_index: Nat,output_row_index: Nat,output_column_index: Nat,accumulator: F32) -> F32:
  OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,indices,output_channel_index,
    MatrixSpec.Position{output_row_index,output_column_index},p,accumulator)

def matrix_value(+problem: ConvSpec.Problem,+channel: Nat,row: Nat,column: Nat) -> F32:
  (matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,row,column,0.0) + ConvSpec.bias_at(problem,channel) : F32)

def bias_value(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,channel: Nat) -> F32:
  Tree.read(F32,Inputs.bias_tree(problem,storage_contract),U32.from_nat(channel))

def bias_matches_spec(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+channel: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},
  inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) -> {bias_value(problem,storage_contract,channel) == ConvSpec.bias_at(problem,channel) : F32}:
  Inputs.bias_matches_spec(problem,storage_contract,
    channel,Guards.valid(problem,valid),inside)

