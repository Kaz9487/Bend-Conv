# Two owned readers and an ordered accumulator. Readers describe addresses;
# this core owns sequencing, cursor advancement and the reduction loop.
# Templates are specialized by Bend; the loop performs no clone or allocation
# of input/weight buffers. Accumulator representation still needs measurement.
import Base
import ./traversal_loop.bend as Loop

type State<ownership,-Input: Kind(ownership),-Weights: Kind(ownership),-Context: Data,-Accumulator: Data> is Kind(ownership):
  State{input: Input,weights: Weights,input_index: U32,weight_index: U32,context: Context,accumulator: Accumulator}

def accumulator(~ownership: Quant,~Input: Kind(ownership),~Weights: Kind(ownership),~Context: Data,~Accumulator: Data,
  state: State<ownership,Input,Weights,Context,Accumulator>) -> Accumulator:
  State{input,weights,input_index,weight_index,context,accumulator} = state
  accumulator

def advance(~ownership: Quant,~Input: Kind(ownership),~Weights: Kind(ownership),~Context: Data,~Accumulator: Data,
  input: Input,weights: Weights,input_index: U32,weight_index: U32,context: Context,accumulator: Accumulator) ->
  State<ownership,Input,Weights,Context,Accumulator>:
  State{input,weights,(input_index + 1 : U32),(weight_index + 1 : U32),context,accumulator}

def after_input(~ownership: Quant,~Input: Kind(ownership),~Weights: Kind(ownership),~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  weights: Weights,input_index: U32,weight_index: U32,+context: Context,accumulator: Accumulator,weight_sample: WeightSample,
  result: Input & InputSample) -> State<ownership,Input,Weights,Context,Accumulator>:
  (input,input_sample) = result
  advance(~ownership,~Input,~Weights,~Context,~Accumulator,input,weights,input_index,weight_index,context,
    update(context,accumulator,input_sample,weight_sample))

def after_weights(~ownership: Quant,~Input: Kind(ownership),~Weights: Kind(ownership),~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~read_input: Input -> Context -> U32 -> Input & InputSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  input: Input,+input_index: U32,weight_index: U32,+context: Context,accumulator: Accumulator,
  result: Weights & WeightSample) -> State<ownership,Input,Weights,Context,Accumulator>:
  (weights,weight_sample) = result
  after_input(~ownership,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~update,
    weights,input_index,weight_index,context,accumulator,weight_sample,read_input(input,context,input_index))

def step(~ownership: Quant,~Input: Kind(ownership),~Weights: Kind(ownership),~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~read_input: Input -> Context -> U32 -> Input & InputSample,
  ~read_weights: Weights -> Context -> U32 -> Weights & WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  state: State<ownership,Input,Weights,Context,Accumulator>) -> State<ownership,Input,Weights,Context,Accumulator>:
  State{input,weights,input_index,+weight_index,+context,accumulator} = state
  after_weights(~ownership,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~update,
    input,input_index,weight_index,context,accumulator,read_weights(weights,context,weight_index))

def run(~ownership: Quant,~Input: Kind(ownership),~Weights: Kind(ownership),~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~read_input: Input -> Context -> U32 -> Input & InputSample,
  ~read_weights: Weights -> Context -> U32 -> Weights & WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  count: Nat,state: State<ownership,Input,Weights,Context,Accumulator>) -> State<ownership,Input,Weights,Context,Accumulator>:
  Loop.run(~State<ownership,Input,Weights,Context,Accumulator>,
    ~step(~ownership,~Input,~Weights,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update),count,state)
