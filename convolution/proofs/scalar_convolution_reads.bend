# The three actual scalar-loop reads, discharged from runtime validation.
import Base
import ./input_storage.bend as Inputs
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ./list_array_roundtrip.bend as Roundtrip
import ./list_array_allocation.bend as ListArrayAllocation
import ./list_array_storage.bend as ListStorage
import ../../lib/storage_buffer.bend as Storage
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../scalar_kernel.bend as ScalarConv
import ../../lib/kernel_spatial_packing.bend as Packing
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./native_sample_refinement.bend as NativeSampleRefinement
import ./native_weight_refinement.bend as NativeWeightRefinement
import ./scalar_tensor_lengths.bend as Lengths
import ./scalar_input_read.bend as InputRead
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/machine_limit_bounds.bend as Limits

def input_read(+problem: ConvSpec.Problem,owner: Array<F32>,+channel: Nat,output_row: Nat,output_column: Nat,kernel_row: Nat,kernel_column: Nat) -> Array<F32> & F32:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{channels,outputs,+height,+width,kernel,+stride,+padding} = shape
  +row = (U32.from_nat(output_row) * U32.from_nat(stride) + U32.from_nat(kernel_row) : U32)
  +column = (U32.from_nat(output_column) * U32.from_nat(stride) + U32.from_nat(kernel_column) : U32)
  ScalarConv.sample(~Array<F32>,~Storage.read_or(F32,0.0),U32.is_le(U32.from_nat(padding),row) && U32.is_lt(row,(U32.from_nat(padding) + U32.from_nat(height) : U32)) &&
    U32.is_le(U32.from_nat(padding),column) && U32.is_lt(column,(U32.from_nat(padding) + U32.from_nat(width) : U32)),owner,
    (U32.from_nat(channel) * U32.from_nat(height) * U32.from_nat(width) + (row - U32.from_nat(padding)) * U32.from_nat(width) + column - U32.from_nat(padding) : U32),
    (U32.from_nat(channels) * U32.from_nat(height) * U32.from_nat(width) : U32))

def canonical_input_matches_spec(+limits: Machine.Limits,+problem: ConvSpec.Problem,+channel: Nat,+output_row: Nat,+output_column: Nat,+kernel_row: Nat,+kernel_column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool},
  +channel_bound: {Nat.is_lt(channel,Guards.input_channel_count(problem)) == True{} : Bool},output_row_bound: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},
  output_column_bound: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},kernel_row_bound: {Nat.is_lt(kernel_row,Guards.kernel(problem)) == True{} : Bool},
  kernel_column_bound: {Nat.is_lt(kernel_column,Guards.kernel(problem)) == True{} : Bool}) ->
  {input_read(problem,ScalarConv.pack(Guards.input(problem)),channel,output_row,output_column,kernel_row,kernel_column) ==
    (ScalarConv.pack(Guards.input(problem)),ConvSpec.sample(problem,channel,output_row,output_column,kernel_row,kernel_column)) : Array<F32> & F32}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+input_channel_count,+output_channel_count,+height,+width,+kernel,+stride,+padding} = shape
  +facts = Guards.valid(problem,valid)
  +bounds = Support.scalar(limits,problem,supported)
  +row = (U32.from_nat(output_row) * U32.from_nat(stride) + U32.from_nat(kernel_row) : U32)
  +column = (U32.from_nat(output_column) * U32.from_nat(stride) + U32.from_nat(kernel_column) : U32)
  Equal.trans(Array<F32> & F32,input_read(problem,ScalarConv.pack(Guards.input(problem)),channel,output_row,output_column,kernel_row,kernel_column),
    Packing.pixel(ScalarConv.pack(input),(row - U32.from_nat(padding) : U32),(column - U32.from_nat(padding) : U32),
      (U32.from_nat(channel) * U32.from_nat(height) * U32.from_nat(width) : U32),U32.from_nat(height),U32.from_nat(width)),
    (ScalarConv.pack(input),ConvSpec.sample(problem,channel,output_row,output_column,kernel_row,kernel_column)),
    InputRead.padded_pixel_matches_packed(input,U32.from_nat(channel),U32.from_nat(input_channel_count),U32.from_nat(height),U32.from_nat(width),row,column,U32.from_nat(padding),
      (U32.from_nat(input_channel_count) * U32.from_nat(height) * U32.from_nat(width) : U32),Support.input_ready_facts(limits,problem,facts,bounds),NativeSampleRefinement.input_shape(limits,problem,facts,bounds),
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(channel)),channel,NativeSampleRefinement.channel_exact(limits,problem,channel,bounds,channel_bound)) :
        {Nat.is_lt(_,U32.to_nat(U32.from_nat(input_channel_count))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(input_channel_count)),input_channel_count,SupportedCoordinateBounds.ci_exact(limits,problem,bounds)) : {Nat.is_lt(channel,_) == True{} : Bool}
      channel_bound,
      NativeSampleRefinement.width_positive(limits,problem,facts,bounds),Lengths.input_length(limits,problem,facts,bounds),Machine.coordinate(limits),Limits.coordinate_fits(limits),
      NativeSampleRefinement.height_padding(limits,problem,bounds),NativeSampleRefinement.width_padding(limits,problem,bounds)),
    %Equal.sym(Array<F32>,ScalarConv.pack(input),Inputs.input_owner(problem,Inputs.from_lists(limits,problem,valid,supported)),ListStorage.packed(input)) :
      {Packing.pixel(_,(row - U32.from_nat(padding) : U32),(column - U32.from_nat(padding) : U32),
        (U32.from_nat(channel) * U32.from_nat(height) * U32.from_nat(width) : U32),U32.from_nat(height),U32.from_nat(width)) ==
        (_,ConvSpec.sample(problem,channel,output_row,output_column,kernel_row,kernel_column)) : Array<F32> & F32}
    NativeSampleRefinement.native_sample(limits,problem,Inputs.from_lists(limits,problem,valid,supported),channel,output_row,output_column,kernel_row,kernel_column,valid,supported,channel_bound,output_row_bound,output_column_bound,kernel_row_bound,kernel_column_bound))

# Transport a bounded read using the shared prefix contract. No coordinate or
# reduction proof is repeated, and no runtime Array is repacked.
def read_from_sequence(+values: List<&2,F32>,+evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>,
  +ready: ListArrayAllocation.Ready(values),+address: U32,+length: U32,+exact: {U32.to_nat(length) == ConvSpec.length(values) : Nat}) ->
  {Storage.read_or(F32,0.0,StorageLaws.sequence_owner(F32,ConvSpec.length(values),values,evidence),address,length) ==
    StorageLaws.replace_read_owner(~Array<F32>,~F32,StorageLaws.sequence_owner(F32,ConvSpec.length(values),values,evidence),
      Storage.read_or(F32,0.0,ScalarConv.pack(values),address,length)) : Array<F32> & F32}:
  %Equal.sym(Array<F32>,ScalarConv.pack(values),StorageLaws.sequence_owner(F32,ConvSpec.length(values),values,Roundtrip.sequence_evidence(values,ready)),
    ListStorage.packed(values)) :
    {Storage.read_or(F32,0.0,StorageLaws.sequence_owner(F32,ConvSpec.length(values),values,evidence),address,length) ==
      StorageLaws.replace_read_owner(~Array<F32>,~F32,StorageLaws.sequence_owner(F32,ConvSpec.length(values),values,evidence),
        Storage.read_or(F32,0.0,_,address,length)) : Array<F32> & F32}
  StorageLaws.equivalent_sequence_read(~F32,0.0,ConvSpec.length(values),values,evidence,Roundtrip.sequence_evidence(values,ready),address,length,exact)

def sample_from_sequence(+inside: Bool,+values: List<&2,F32>,+evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>,
  +ready: ListArrayAllocation.Ready(values),+address: U32,+length: U32,+exact: {U32.to_nat(length) == ConvSpec.length(values) : Nat}) ->
  {ScalarConv.sample(~Array<F32>,~Storage.read_or(F32,0.0),inside,StorageLaws.sequence_owner(F32,ConvSpec.length(values),values,evidence),address,length) ==
    StorageLaws.replace_read_owner(~Array<F32>,~F32,StorageLaws.sequence_owner(F32,ConvSpec.length(values),values,evidence),
      ScalarConv.sample(~Array<F32>,~Storage.read_or(F32,0.0),inside,ScalarConv.pack(values),address,length)) : Array<F32> & F32}:
  match inside:
    case False{}: {==}
    case True{}: read_from_sequence(values,evidence,ready,address,length,exact)

def input_matches_spec(+limits: Machine.Limits,+problem: ConvSpec.Problem,+inputs: Inputs.Contract<problem>,
  +channel: Nat,+output_row: Nat,+output_column: Nat,+kernel_row: Nat,+kernel_column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool},
  +channel_bound: {Nat.is_lt(channel,Guards.input_channel_count(problem)) == True{} : Bool},output_row_bound: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},
  output_column_bound: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},kernel_row_bound: {Nat.is_lt(kernel_row,Guards.kernel(problem)) == True{} : Bool},
  kernel_column_bound: {Nat.is_lt(kernel_column,Guards.kernel(problem)) == True{} : Bool}) ->
  {input_read(problem,Inputs.input_owner(problem,inputs),channel,output_row,output_column,kernel_row,kernel_column) ==
    (Inputs.input_owner(problem,inputs),ConvSpec.sample(problem,channel,output_row,output_column,kernel_row,kernel_column)) : Array<F32> & F32}:
  ConvSpec.Problem{shape,+input,weights,bias} = problem
  ConvSpec.Shape{+input_channel_count,output_channel_count,+height,+width,kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,weight_evidence,bias_evidence} = inputs
  +facts = Guards.valid(problem,valid)
  +bounds = Support.scalar(limits,problem,supported)
  +row = (U32.from_nat(output_row) * U32.from_nat(stride) + U32.from_nat(kernel_row) : U32)
  +column = (U32.from_nat(output_column) * U32.from_nat(stride) + U32.from_nat(kernel_column) : U32)
  %canonical_input_matches_spec(limits,problem,channel,output_row,output_column,kernel_row,kernel_column,valid,supported,
    channel_bound,output_row_bound,output_column_bound,kernel_row_bound,kernel_column_bound) :
    {input_read(problem,Inputs.input_owner(problem,inputs),channel,output_row,output_column,kernel_row,kernel_column) ==
      StorageLaws.replace_read_owner(~Array<F32>,~F32,Inputs.input_owner(problem,inputs),_) : Array<F32> & F32}
  sample_from_sequence(U32.is_le(U32.from_nat(padding),row) && U32.is_lt(row,(U32.from_nat(padding) + U32.from_nat(height) : U32)) &&
    U32.is_le(U32.from_nat(padding),column) && U32.is_lt(column,(U32.from_nat(padding) + U32.from_nat(width) : U32)),input,input_evidence,
    Support.input_ready_facts(limits,problem,facts,bounds),
    (U32.from_nat(channel) * U32.from_nat(height) * U32.from_nat(width) + (row - U32.from_nat(padding)) * U32.from_nat(width) + column - U32.from_nat(padding) : U32),
    (U32.from_nat(input_channel_count) * U32.from_nat(height) * U32.from_nat(width) : U32),Lengths.input_length(limits,problem,facts,bounds))

def bias_matches_spec(+limits: Machine.Limits,+problem: ConvSpec.Problem,+inputs: Inputs.Contract<problem>,+output_channel: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool},
  +output_channel_bound: {Nat.is_lt(output_channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {Storage.read_or(F32,0.0,Inputs.bias_owner(problem,inputs),U32.from_nat(output_channel),U32.from_nat(Guards.output_channel_count(problem))) ==
    (Inputs.bias_owner(problem,inputs),ConvSpec.bias_at(problem,output_channel)) : Array<F32> & F32}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  Inputs.Contract{input_evidence,weight_evidence,+bias_evidence} = inputs
  +facts = Guards.valid(problem,valid)
  +bounds = Support.scalar(limits,problem,supported)
  +address_exact = Bounds.roundtrip(output_channel,Machine.capacity(limits),Limits.capacity_fits(limits),
    NativeWeightRefinement.below_dimension(limits,output_channel,Guards.output_channel_count(problem),output_channel_bound,SupportedCoordinateBounds.co_bound(limits,problem,bounds)))
  %Equal.sym(F32,ConvSpec.at(output_channel,bias),Maybe.default(&2,F32,List.get(&2,F32,bias,output_channel),0.0),
    Inputs.lookup_matches_list(output_channel,bias)) :
    {Storage.read_or(F32,0.0,Inputs.bias_owner(problem,inputs),U32.from_nat(output_channel),U32.from_nat(Guards.output_channel_count(problem))) ==
      (Inputs.bias_owner(problem,inputs),_) : Array<F32> & F32}
  StorageLaws.read_encoded_sequence(~F32,0.0,ConvSpec.length(bias),bias,bias_evidence,U32.from_nat(output_channel),output_channel,
    U32.from_nat(Guards.output_channel_count(problem)),address_exact,Lengths.bias_length(limits,problem,facts,bounds))
