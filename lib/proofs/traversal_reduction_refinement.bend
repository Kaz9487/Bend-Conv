# One ownership/read/update refinement for every accumulator and address policy.
# Read premises must be discharged by the actual readers, not by their callers.
import Base
import ../traversal_reduction.bend as Reduction
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./loop_projection.bend as Projection
import ../ordered_matrix.bend as OrderedMatrix
import ../math_fp32.bend as Fp32

# The read/update contract does not require a particular storage model. The
# erased owners occur only in propositions; executing this theorem clones none.
def step_from_reads(~ownership: Quant,~Input: Kind(ownership),~Weights: Kind(ownership),~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~read_input: Input -> Context -> U32 -> Input & InputSample,
  ~read_weights: Weights -> Context -> U32 -> Weights & WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  -input: Input,-weights: Weights,+input_index: U32,+weight_index: U32,+context: Context,+accumulator: Accumulator,
  +input_sample: InputSample,+weight_sample: WeightSample,
  input_read: {read_input(input,context,input_index) == (input,input_sample) : Input & InputSample},
  weight_read: {read_weights(weights,context,weight_index) == (weights,weight_sample) : Weights & WeightSample}) ->
  {Reduction.step(~ownership,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,
      Reduction.State{input,weights,input_index,weight_index,context,accumulator}) ==
    Reduction.advance(~ownership,~Input,~Weights,~Context,~Accumulator,input,weights,input_index,weight_index,context,
      update(context,accumulator,input_sample,weight_sample)) : Reduction.State<ownership,Input,Weights,Context,Accumulator>}:
  %Equal.sym(Weights & WeightSample,read_weights(weights,context,weight_index),(weights,weight_sample),weight_read) :
    {Reduction.after_weights(~ownership,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~update,
      input,input_index,weight_index,context,accumulator,_) ==
      Reduction.advance(~ownership,~Input,~Weights,~Context,~Accumulator,input,weights,input_index,weight_index,context,
        update(context,accumulator,input_sample,weight_sample)) : Reduction.State<ownership,Input,Weights,Context,Accumulator>}
  %Equal.sym(Input & InputSample,read_input(input,context,input_index),(input,input_sample),input_read) :
    {Reduction.after_input(~ownership,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~update,
      weights,input_index,weight_index,context,accumulator,weight_sample,_) ==
      Reduction.advance(~ownership,~Input,~Weights,~Context,~Accumulator,input,weights,input_index,weight_index,context,
        update(context,accumulator,input_sample,weight_sample)) : Reduction.State<ownership,Input,Weights,Context,Accumulator>}
  {==}

def pack(~Input: Data,~Weight: Data,~Context: Data,~Accumulator: Data,
  state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) ->
  Reduction.State<&1,Array<Input>,Array<Weight>,Context,Accumulator>:
  Reduction.State{input,weights,input_index,weight_index,context,accumulator} = state
  Reduction.State{Tree.pack(Input,input),Tree.pack(Weight,weights),input_index,weight_index,context,accumulator}

# A read-only model supplies samples through the same owner-returning interface.
def observe_read(~Owner: Data,~Sample: Data,~Context: Data,~sample: Owner -> Context -> U32 -> Sample,
  owner: Owner,context: Context,index: U32) -> Owner & Sample:
  +owner = owner
  (owner,sample(owner,context,index))

def model_step(~Input: Data,~Weight: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Weight> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) ->
  Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>:
  Reduction.step(~&2,~Tree.Tree<Input>,~Tree.Tree<Weight>,~InputSample,~WeightSample,~Context,~Accumulator,
    ~observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
    ~observe_read(~Tree.Tree<Weight>,~WeightSample,~Context,~sample_weights),~update,state)

def step_refines_model(~Input: Data,~Weight: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~read_input: Array<Input> -> Context -> U32 -> Array<Input> & InputSample,
  ~read_weights: Array<Weight> -> Context -> U32 -> Array<Weight> & WeightSample,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Weight> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~input_read: @+tree: Tree.Tree<Input> -> @+context: Context -> @+index: U32 ->
    {read_input(Tree.pack(Input,tree),context,index) == (Tree.pack(Input,tree),sample_input(tree,context,index)) : Array<Input> & InputSample},
  ~weight_read: @+tree: Tree.Tree<Weight> -> @+context: Context -> @+index: U32 ->
    {read_weights(Tree.pack(Weight,tree),context,index) == (Tree.pack(Weight,tree),sample_weights(tree,context,index)) : Array<Weight> & WeightSample},
  +state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) ->
  {Reduction.step(~&1,~Array<Input>,~Array<Weight>,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,
      pack(~Input,~Weight,~Context,~Accumulator,state)) ==
    pack(~Input,~Weight,~Context,~Accumulator,model_step(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,
      ~sample_input,~sample_weights,~update,state)) : Reduction.State<&1,Array<Input>,Array<Weight>,Context,Accumulator>}:
  Reduction.State{+input,+weights,+input_index,+weight_index,+context,+accumulator} = state
  step_from_reads(~&1,~Array<Input>,~Array<Weight>,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,
    Tree.pack(Input,input),Tree.pack(Weight,weights),input_index,weight_index,context,accumulator,
    sample_input(input,context,input_index),sample_weights(weights,context,weight_index),
    input_read(input,context,input_index),weight_read(weights,context,weight_index))

# Lift the same read/update refinement through the complete inner reduction.
# The model runs Reduction.run itself with observational readers.
def run_refines_model(~Input: Data,~Weight: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~read_input: Array<Input> -> Context -> U32 -> Array<Input> & InputSample,
  ~read_weights: Array<Weight> -> Context -> U32 -> Array<Weight> & WeightSample,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Weight> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~input_read: @+tree: Tree.Tree<Input> -> @+context: Context -> @+index: U32 ->
    {read_input(Tree.pack(Input,tree),context,index) == (Tree.pack(Input,tree),sample_input(tree,context,index)) : Array<Input> & InputSample},
  ~weight_read: @+tree: Tree.Tree<Weight> -> @+context: Context -> @+index: U32 ->
    {read_weights(Tree.pack(Weight,tree),context,index) == (Tree.pack(Weight,tree),sample_weights(tree,context,index)) : Array<Weight> & WeightSample},
  +count: Nat,+state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) ->
  {Reduction.run(~&1,~Array<Input>,~Array<Weight>,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,
      count,pack(~Input,~Weight,~Context,~Accumulator,state)) ==
    pack(~Input,~Weight,~Context,~Accumulator,
      Reduction.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Weight>,~InputSample,~WeightSample,~Context,~Accumulator,
        ~observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
        ~observe_read(~Tree.Tree<Weight>,~WeightSample,~Context,~sample_weights),~update,count,state)) :
    Reduction.State<&1,Array<Input>,Array<Weight>,Context,Accumulator>}:
  Projection.iteration_refines_model(~Reduction.State<&1,Array<Input>,Array<Weight>,Context,Accumulator>,
    ~Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>,
    ~Reduction.step(~&1,~Array<Input>,~Array<Weight>,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update),
    ~Reduction.step(~&2,~Tree.Tree<Input>,~Tree.Tree<Weight>,~InputSample,~WeightSample,~Context,~Accumulator,
      ~observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
      ~observe_read(~Tree.Tree<Weight>,~WeightSample,~Context,~sample_weights),~update),
    ~pack(~Input,~Weight,~Context,~Accumulator),
    ~step_refines_model(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,
      ~read_input,~read_weights,~sample_input,~sample_weights,~update,~input_read,~weight_read),count,state)

# A projected accumulator follows an ordered dot product for every output key.
# The key may describe a scalar, a lane, a row, or any other local block index.
# Only the local update law is strategy-specific; there is one loop induction.
type AddressPair is Data:
  AddressPair{input: U32,weight: U32}

type SampleSource<-Input: Data,-Weight: Data,-Context: Data> is Data:
  SampleSource{input: Tree.Tree<Input>,weights: Tree.Tree<Weight>,context: Context}

def source(~Input: Data,~Weight: Data,~Context: Data,~Accumulator: Data,
  state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) -> SampleSource<Input,Weight,Context>:
  Reduction.State{input,weights,input_index,weight_index,context,accumulator} = state
  SampleSource{input,weights,context}

def step_preserves_source(~Input: Data,~Weight: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Weight> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  +state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) ->
  {source(~Input,~Weight,~Context,~Accumulator,model_step(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,~sample_input,~sample_weights,~update,state)) ==
    source(~Input,~Weight,~Context,~Accumulator,state) : SampleSource<Input,Weight,Context>}:
  Reduction.State{input,weights,input_index,weight_index,context,accumulator} = state
  {==}

def loop_preserves_source(~Input: Data,~Weight: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Weight> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  +count: Nat,+state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) ->
  {source(~Input,~Weight,~Context,~Accumulator,
      Reduction.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Weight>,~InputSample,~WeightSample,~Context,~Accumulator,
        ~observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),~observe_read(~Tree.Tree<Weight>,~WeightSample,~Context,~sample_weights),~update,count,state)) ==
    source(~Input,~Weight,~Context,~Accumulator,state) : SampleSource<Input,Weight,Context>}:
  Projection.preserves_observation(~Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>,~SampleSource<Input,Weight,Context>,
    ~Reduction.step(~&2,~Tree.Tree<Input>,~Tree.Tree<Weight>,~InputSample,~WeightSample,~Context,~Accumulator,
      ~observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),~observe_read(~Tree.Tree<Weight>,~WeightSample,~Context,~sample_weights),~update),
    ~source(~Input,~Weight,~Context,~Accumulator),
    ~step_preserves_source(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,~sample_input,~sample_weights,~update),count,state)

def addresses(count: Nat,+input: U32,+weight: U32) -> List<&2,AddressPair>:
  match count:
    case 0n: []
    case 1n+rest: AddressPair{input,weight} <> addresses(rest,(input + 1 : U32),(weight + 1 : U32))

def input_element(~Input: Data,~Weight: Data,~Context: Data,~Sample: Data,~Key: Data,
  ~sample: Tree.Tree<Input> -> Context -> U32 -> Sample,~element: Key -> Sample -> F32,
  source: SampleSource<Input,Weight,Context>,key: Key,address: AddressPair) -> F32:
  SampleSource{input,weights,context} = source
  AddressPair{input_index,weight_index} = address
  element(key,sample(input,context,input_index))

def weight_element(~Input: Data,~Weight: Data,~Context: Data,~Sample: Data,~Key: Data,
  ~sample: Tree.Tree<Weight> -> Context -> U32 -> Sample,~element: Key -> Sample -> F32,
  source: SampleSource<Input,Weight,Context>,key: Key,address: AddressPair) -> F32:
  SampleSource{input,weights,context} = source
  AddressPair{input_index,weight_index} = address
  element(key,sample(weights,context,weight_index))

def project(~Input: Data,~Weight: Data,~Context: Data,~Accumulator: Data,~Key: Data,~element: Key -> Accumulator -> F32,
  key: Key,state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) -> F32:
  element(key,Reduction.accumulator(~&2,~Tree.Tree<Input>,~Tree.Tree<Weight>,~Context,~Accumulator,state))

def ordered_dot(~Input: Data,~Weight: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,~Key: Data,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Weight> -> Context -> U32 -> WeightSample,
  ~input_at: Key -> InputSample -> F32,~weight_at: Key -> WeightSample -> F32,~accumulator_at: Key -> Accumulator -> F32,
  key: Key,count: Nat,state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) -> F32:
  Reduction.State{input,weights,input_index,weight_index,context,accumulator} = state
  +key = key
  OrderedMatrix.dot(~Key,~AddressPair,~Key,~SampleSource<Input,Weight,Context>,
    ~input_element(~Input,~Weight,~Context,~InputSample,~Key,~sample_input,~input_at),
    ~weight_element(~Input,~Weight,~Context,~WeightSample,~Key,~sample_weights,~weight_at),
    addresses(count,input_index,weight_index),key,key,SampleSource{input,weights,context},accumulator_at(key,accumulator))

def ordered_dot_zero(~Input: Data,~Weight: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,~Key: Data,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Weight> -> Context -> U32 -> WeightSample,
  ~input_at: Key -> InputSample -> F32,~weight_at: Key -> WeightSample -> F32,~accumulator_at: Key -> Accumulator -> F32,
  +key: Key,+state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) ->
  {project(~Input,~Weight,~Context,~Accumulator,~Key,~accumulator_at,key,state) ==
    ordered_dot(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,~Key,~sample_input,~sample_weights,~input_at,~weight_at,~accumulator_at,key,0n,state) : F32}:
  Reduction.State{input,weights,input_index,weight_index,context,accumulator} = state
  {==}

def ordered_dot_step(~Input: Data,~Weight: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,~Key: Data,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Weight> -> Context -> U32 -> WeightSample,
  ~input_at: Key -> InputSample -> F32,~weight_at: Key -> WeightSample -> F32,~accumulator_at: Key -> Accumulator -> F32,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~update_matches: @+context: Context -> @+key: Key -> @+accumulator: Accumulator -> @+input: InputSample -> @+weight: WeightSample ->
    {accumulator_at(key,update(context,accumulator,input,weight)) == Fp32.multiply_add(accumulator_at(key,accumulator),input_at(key,input),weight_at(key,weight)) : F32},
  +key: Key,+count: Nat,+state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) ->
  {ordered_dot(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,~Key,~sample_input,~sample_weights,~input_at,~weight_at,~accumulator_at,key,count,
      model_step(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,~sample_input,~sample_weights,~update,state)) ==
    ordered_dot(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,~Key,~sample_input,~sample_weights,~input_at,~weight_at,~accumulator_at,key,1n+count,state) : F32}:
  Reduction.State{+input,+weights,+input_index,+weight_index,+context,+accumulator} = state
  %Equal.sym(F32,accumulator_at(key,update(context,accumulator,sample_input(input,context,input_index),sample_weights(weights,context,weight_index))),
    Fp32.multiply_add(accumulator_at(key,accumulator),input_at(key,sample_input(input,context,input_index)),weight_at(key,sample_weights(weights,context,weight_index))),
    update_matches(context,key,accumulator,sample_input(input,context,input_index),sample_weights(weights,context,weight_index))) :
    {OrderedMatrix.dot(~Key,~AddressPair,~Key,~SampleSource<Input,Weight,Context>,
      ~input_element(~Input,~Weight,~Context,~InputSample,~Key,~sample_input,~input_at),
      ~weight_element(~Input,~Weight,~Context,~WeightSample,~Key,~sample_weights,~weight_at),
      addresses(count,(input_index + 1 : U32),(weight_index + 1 : U32)),key,key,SampleSource{input,weights,context},_) ==
      ordered_dot(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,~Key,~sample_input,~sample_weights,~input_at,~weight_at,~accumulator_at,key,1n+count,state) : F32}
  {==}

def model_loop_matches_ordered_dot(~Input: Data,~Weight: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,~Key: Data,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Weight> -> Context -> U32 -> WeightSample,
  ~input_at: Key -> InputSample -> F32,~weight_at: Key -> WeightSample -> F32,~accumulator_at: Key -> Accumulator -> F32,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~update_matches: @+context: Context -> @+key: Key -> @+accumulator: Accumulator -> @+input: InputSample -> @+weight: WeightSample ->
    {accumulator_at(key,update(context,accumulator,input,weight)) == Fp32.multiply_add(accumulator_at(key,accumulator),input_at(key,input),weight_at(key,weight)) : F32},
  +key: Key,+count: Nat,+state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>) ->
  {project(~Input,~Weight,~Context,~Accumulator,~Key,~accumulator_at,key,
      Reduction.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Weight>,~InputSample,~WeightSample,~Context,~Accumulator,
        ~observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),~observe_read(~Tree.Tree<Weight>,~WeightSample,~Context,~sample_weights),~update,count,state)) ==
    ordered_dot(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,~Key,~sample_input,~sample_weights,~input_at,~weight_at,~accumulator_at,key,count,state) : F32}:
  Projection.iteration_matches_denotation(~F32,~Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Weight>,Context,Accumulator>,~Key,
    ~Reduction.step(~&2,~Tree.Tree<Input>,~Tree.Tree<Weight>,~InputSample,~WeightSample,~Context,~Accumulator,
      ~observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
      ~observe_read(~Tree.Tree<Weight>,~WeightSample,~Context,~sample_weights),~update),
    ~project(~Input,~Weight,~Context,~Accumulator,~Key,~accumulator_at),
    ~ordered_dot(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,~Key,~sample_input,~sample_weights,~input_at,~weight_at,~accumulator_at),
    ~ordered_dot_zero(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,~Key,~sample_input,~sample_weights,~input_at,~weight_at,~accumulator_at),
    ~ordered_dot_step(~Input,~Weight,~InputSample,~WeightSample,~Context,~Accumulator,~Key,~sample_input,~sample_weights,~input_at,~weight_at,~accumulator_at,~update,~update_matches),
    count,key,state)
