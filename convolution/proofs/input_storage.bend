# Input contents and capacity, shared by every convolution execution strategy.
import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/machine_limits.bend as Machine
import ../specification.bend as ConvSpec
import ../scalar_kernel.bend as ScalarConv
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./list_array_roundtrip.bend as Roundtrip
import ./native_weight_refinement.bend as NativeWeightRefinement

def length_matches_list(+values: List<&2,F32>) ->
  {ConvSpec.length(values) == List.length(&2,F32,values) : Nat}:
  match values:
    case Nil{}: {==}
    case head <> +tail:
      %length_matches_list(tail) : {1n+ConvSpec.length(tail) == 1n+_ : Nat}
      {==}

def observed_length(+length: Nat,+tree: Tree.Tree<F32>) ->
  {ConvSpec.length(Tree.read_sequence(~F32,length,0,tree)) == length : Nat}:
  Equal.trans(Nat,ConvSpec.length(Tree.read_sequence(~F32,length,0,tree)),
    List.length(&2,F32,Tree.read_sequence(~F32,length,0,tree)),length,
    length_matches_list(Tree.read_sequence(~F32,length,0,tree)),StorageLaws.sequence_length(~F32,length,0,tree))

def lookup_matches_list(+index: Nat,+values: List<&2,F32>) ->
  {ConvSpec.at(index,values) == Maybe.default(&2,F32,List.get(&2,F32,values,index),0.0) : F32}:
  match index values:
    case 0n Nil{}: {==}
    case 1n+previous Nil{}: {==}
    case 0n head <> tail: {==}
    case 1n+previous head <> tail: lookup_matches_list(previous,tail)

def prefix_evidence(+length: Nat,+storage: Tree.Tree<F32>,
  room: {Nat.is_le(length,U32.to_nat(Tree.capacity(F32,storage))) == True{} : Bool}) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(Tree.read_sequence(~F32,length,0,storage)),Tree.read_sequence(~F32,length,0,storage)>:
  StorageLaws.SequenceEvidence{storage,
    %Equal.sym(Nat,ConvSpec.length(Tree.read_sequence(~F32,length,0,storage)),length,observed_length(length,storage)) :
      {Tree.read_sequence(~F32,_,0,storage) == Tree.read_sequence(~F32,length,0,storage) : List<&2,F32>}
    {==},
    %Equal.sym(Nat,ConvSpec.length(Tree.read_sequence(~F32,length,0,storage)),length,observed_length(length,storage)) :
      {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,storage))) == True{} : Bool}
    room}

# Evidence is attached to the supplied storage, not to an allocator identity.
type Contract<-problem: ConvSpec.Problem> is Data:
  Contract{
    input: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.input(problem)),Guards.input(problem)>,
    weights: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.weights(problem)),Guards.weights(problem)>,
    bias: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.bias(problem)),Guards.bias(problem)>}

def input_sequence(-problem: ConvSpec.Problem,inputs: Contract<problem>) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.input(problem)),Guards.input(problem)>:
  Contract{input,weights,bias} = inputs
  input

def input_tree(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Tree.Tree<F32>:
  StorageLaws.sequence_storage(F32,ConvSpec.length(Guards.input(problem)),Guards.input(problem),input_sequence(problem,inputs))

def weight_tree(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Tree.Tree<F32>:
  Contract{input,weights,bias} = inputs
  StorageLaws.sequence_storage(F32,ConvSpec.length(Guards.weights(problem)),Guards.weights(problem),weights)

def bias_tree(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Tree.Tree<F32>:
  Contract{input,weights,bias} = inputs
  StorageLaws.sequence_storage(F32,ConvSpec.length(Guards.bias(problem)),Guards.bias(problem),bias)

def input_owner(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Array<F32>:
  Tree.pack(F32,input_tree(problem,inputs))

def weight_owner(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Array<F32>:
  Tree.pack(F32,weight_tree(problem,inputs))

def bias_owner(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Array<F32>:
  Tree.pack(F32,bias_tree(problem,inputs))

def stored_value(+values: List<&2,F32>,+evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>,+index: Nat,
  inside: {Nat.is_lt(index,ConvSpec.length(values)) == True{} : Bool}) ->
  {Tree.read(F32,StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),U32.from_nat(index)) == ConvSpec.at(index,values) : F32}:
  %Equal.sym(F32,ConvSpec.at(index,values),Maybe.default(&2,F32,List.get(&2,F32,values,index),0.0),lookup_matches_list(index,values)) :
    {Tree.read(F32,StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),U32.from_nat(index)) == _ : F32}
  StorageLaws.stored_sequence_element(~F32,0.0,ConvSpec.length(values),values,evidence,index,inside)

# Coordinates establish logical bounds; the common prefix theorem establishes
# the read. Neither theorem below depends on how these owners were allocated.
def weight_matches_spec(+problem: ConvSpec.Problem,+inputs: Contract<problem>,
  +output_channel: Nat,+input_channel: Nat,+kernel_row: Nat,+kernel_column: Nat,
  valid: Guards.Valid<problem>,output_inside: {Nat.is_lt(output_channel,Guards.output_channel_count(problem)) == True{} : Bool},
  input_inside: {Nat.is_lt(input_channel,Guards.input_channel_count(problem)) == True{} : Bool},
  row_inside: {Nat.is_lt(kernel_row,Guards.kernel(problem)) == True{} : Bool},
  column_inside: {Nat.is_lt(kernel_column,Guards.kernel(problem)) == True{} : Bool}) ->
  {Tree.read(F32,weight_tree(problem,inputs),U32.from_nat(NativeWeightRefinement.weight_index(output_channel,Guards.input_channel_count(problem),input_channel,
      Guards.kernel(problem),kernel_row,kernel_column))) == ConvSpec.weight(problem,output_channel,input_channel,kernel_row,kernel_column) : F32}:
  ConvSpec.Problem{shape,input_values,weight_values,bias_values} = problem
  ConvSpec.Shape{input_channel_count,output_channel_count,height,width,kernel,stride,padding} = shape
  Contract{input,+weights,bias} = inputs
  stored_value(weight_values,weights,
    NativeWeightRefinement.weight_index(output_channel,input_channel_count,input_channel,kernel,kernel_row,kernel_column),
    NativeWeightRefinement.index_bound(problem,output_channel,input_channel,kernel_row,kernel_column,valid,output_inside,input_inside,row_inside,column_inside))

def bias_matches_spec(+problem: ConvSpec.Problem,+inputs: Contract<problem>,+channel: Nat,
  valid: Guards.Valid<problem>,inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {Tree.read(F32,bias_tree(problem,inputs),U32.from_nat(channel)) == ConvSpec.bias_at(problem,channel) : F32}:
  ConvSpec.Problem{shape,input_values,weight_values,bias_values} = problem
  Contract{input,weights,+bias} = inputs
  stored_value(bias_values,bias,channel,
    NativeWeightRefinement.bias_index(problem,channel,valid,inside))

def from_lists(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool}) -> Contract<problem>:
  +facts = Guards.valid(problem,valid)
  +bounds = Support.scalar(limits,problem,supported)
  Contract{
    Roundtrip.sequence_evidence(Guards.input(problem),Support.input_ready_facts(limits,problem,facts,bounds)),
    Roundtrip.sequence_evidence(Guards.weights(problem),Support.weights_ready_facts(limits,problem,facts,bounds)),
    Roundtrip.sequence_evidence(Guards.bias(problem),Support.bias_ready_facts(limits,problem,facts,bounds))}
