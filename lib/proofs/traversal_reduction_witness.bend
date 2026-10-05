# Read-only reduction returns both owners. Samples and accumulators are erased
# fields of this evidence, so their values never become live proof inputs.
import Base
import ../traversal_reduction.bend as Reduction
import ./traversal_witness.bend as TraversalWitness

type Evidence<-Input: Type,-Weights: Type,-Context: Data,-Accumulator: Data,
  -input: Input,-weights: Weights,-state: Reduction.State<&1,Input,Weights,Context,Accumulator>> is Data:
  Evidence{input_index: U32,weight_index: U32,context: Context,-accumulator: Accumulator,
    equation: {state == Reduction.State{input,weights,input_index,weight_index,context,accumulator} : Reduction.State<&1,Input,Weights,Context,Accumulator>}}

def step_reads(~Input: Type,~Weights: Type,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~read_input: Input -> Context -> U32 -> Input & InputSample,
  ~read_weights: Weights -> Context -> U32 -> Weights & WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  -input: Input,-weights: Weights,+input_index: U32,+weight_index: U32,+context: Context,-accumulator: Accumulator,
  input_read: TraversalWitness.Sample<Input,InputSample,input,read_input(input,context,input_index)>,
  weight_read: TraversalWitness.Sample<Weights,WeightSample,weights,read_weights(weights,context,weight_index)>) ->
  Evidence<Input,Weights,Context,Accumulator,input,weights,
    Reduction.step(~&1,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,
      Reduction.State{input,weights,input_index,weight_index,context,accumulator})>:
  TraversalWitness.Sample{input_sample,input_equation} = input_read
  TraversalWitness.Sample{weight_sample,weight_equation} = weight_read
  Evidence{(input_index + 1 : U32),(weight_index + 1 : U32),context,update(context,accumulator,input_sample,weight_sample),
    %Equal.sym(Weights & WeightSample,read_weights(weights,context,weight_index),(weights,weight_sample),weight_equation) :
      {Reduction.after_weights(~&1,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~update,
        input,input_index,weight_index,context,accumulator,_) ==
        Reduction.advance(~&1,~Input,~Weights,~Context,~Accumulator,input,weights,input_index,weight_index,context,
          update(context,accumulator,input_sample,weight_sample)) : Reduction.State<&1,Input,Weights,Context,Accumulator>}
    %Equal.sym(Input & InputSample,read_input(input,context,input_index),(input,input_sample),input_equation) :
      {Reduction.after_input(~&1,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~update,
        weights,input_index,weight_index,context,accumulator,weight_sample,_) ==
        Reduction.advance(~&1,~Input,~Weights,~Context,~Accumulator,input,weights,input_index,weight_index,context,
          update(context,accumulator,input_sample,weight_sample)) : Reduction.State<&1,Input,Weights,Context,Accumulator>}
    {==}}

def step(~Input: Type,~Weights: Type,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~InputEvidence: @-input: Input -> Data,~WeightEvidence: @-weights: Weights -> Data,
  ~read_input: Input -> Context -> U32 -> Input & InputSample,
  ~read_weights: Weights -> Context -> U32 -> Weights & WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~input_preserves: @-input: Input -> @context: Context -> @index: U32 -> InputEvidence(input) -> TraversalWitness.Sample<Input,InputSample,input,read_input(input,context,index)>,
  ~weights_preserve: @-weights: Weights -> @context: Context -> @index: U32 -> WeightEvidence(weights) -> TraversalWitness.Sample<Weights,WeightSample,weights,read_weights(weights,context,index)>,
  -input: Input,-weights: Weights,-state: Reduction.State<&1,Input,Weights,Context,Accumulator>,
  input_evidence: InputEvidence(input),weight_evidence: WeightEvidence(weights),evidence: Evidence<Input,Weights,Context,Accumulator,input,weights,state>) ->
  Evidence<Input,Weights,Context,Accumulator,input,weights,
    Reduction.step(~&1,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,state)>:
  Evidence{+input_index,+weight_index,+context,accumulator,equation} = evidence
  %Equal.sym(Reduction.State<&1,Input,Weights,Context,Accumulator>,state,
    Reduction.State{input,weights,input_index,weight_index,context,accumulator},equation) :
    Evidence<Input,Weights,Context,Accumulator,input,weights,
      Reduction.step(~&1,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,_)>
  step_reads(~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,
    input,weights,input_index,weight_index,context,accumulator,input_preserves(input,context,input_index,input_evidence),
    weights_preserve(weights,context,weight_index,weight_evidence))

def run(~Input: Type,~Weights: Type,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~InputEvidence: @-input: Input -> Data,~WeightEvidence: @-weights: Weights -> Data,
  ~read_input: Input -> Context -> U32 -> Input & InputSample,
  ~read_weights: Weights -> Context -> U32 -> Weights & WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~input_preserves: @-input: Input -> @context: Context -> @index: U32 -> InputEvidence(input) -> TraversalWitness.Sample<Input,InputSample,input,read_input(input,context,index)>,
  ~weights_preserve: @-weights: Weights -> @context: Context -> @index: U32 -> WeightEvidence(weights) -> TraversalWitness.Sample<Weights,WeightSample,weights,read_weights(weights,context,index)>,
  count: Nat,-input: Input,-weights: Weights,-state: Reduction.State<&1,Input,Weights,Context,Accumulator>,
  +input_evidence: InputEvidence(input),+weight_evidence: WeightEvidence(weights),evidence: Evidence<Input,Weights,Context,Accumulator,input,weights,state>) ->
  Evidence<Input,Weights,Context,Accumulator,input,weights,
    Reduction.run(~&1,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      run(~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~InputEvidence,~WeightEvidence,
        ~read_input,~read_weights,~update,~input_preserves,~weights_preserve,rest,input,weights,
        Reduction.step(~&1,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,state),
        input_evidence,weight_evidence,
        step(~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~InputEvidence,~WeightEvidence,
          ~read_input,~read_weights,~update,~input_preserves,~weights_preserve,input,weights,state,input_evidence,weight_evidence,evidence))
