# A complete ordered reduce-and-write driver. Strategies supply readers, a
# local accumulator and a writer; they do not own the outer output traversal.
import Base
import ./traversal_array.bend as Traversal
import ./traversal_reduction.bend as Reduction

type Buffers<ownership,-Input: Kind(ownership),-Storage: Kind(ownership)> is Kind(ownership):
  Buffers{input: Input,weights: Storage,bias: Storage,output: Storage}

type Job<-Context: Data,-Accumulator: Data> is Data:
  Job{context: Context,count: Nat,input_index: U32,weight_index: U32,initial: Accumulator}

def after_write(~ownership: Quant,~Input: Kind(ownership),~Storage: Kind(ownership),
  input: Input,weights: Storage,written: Storage & Storage) -> Buffers<ownership,Input,Storage>:
  (bias,output) = written
  Buffers{input,weights,bias,output}

def after_reduction(~ownership: Quant,~Input: Kind(ownership),~Storage: Kind(ownership),~Context: Data,~Accumulator: Data,
  ~write: Context -> Accumulator -> Storage -> Storage -> Storage & Storage,
  bias: Storage,output: Storage,result: Reduction.State<ownership,Input,Storage,Context,Accumulator>) -> Buffers<ownership,Input,Storage>:
  Reduction.State{input,weights,input_index,weight_index,context,accumulator} = result
  after_write(~ownership,~Input,~Storage,input,weights,write(context,accumulator,bias,output))

def execute_job(~ownership: Quant,~Input: Kind(ownership),~Storage: Kind(ownership),~InputSample: Data,~WeightSample: Data,
  ~Context: Data,~Accumulator: Data,
  ~read_input: Input -> Context -> U32 -> Input & InputSample,
  ~read_weights: Storage -> Context -> U32 -> Storage & WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Storage -> Storage -> Storage & Storage,
  job: Job<Context,Accumulator>,buffers: Buffers<ownership,Input,Storage>) -> Buffers<ownership,Input,Storage>:
  Job{context,count,input_index,weight_index,initial} = job
  Buffers{input,weights,bias,output} = buffers
  after_reduction(~ownership,~Input,~Storage,~Context,~Accumulator,~write,bias,output,
    Reduction.run(~ownership,~Input,~Storage,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,
      count,Reduction.State{input,weights,input_index,weight_index,context,initial}))

def output_step(~ownership: Quant,~Input: Kind(ownership),~Storage: Kind(ownership),~InputSample: Data,~WeightSample: Data,
  ~Layout: Data,~Context: Data,~Accumulator: Data,
  ~job: Layout -> U32 -> Job<Context,Accumulator>,
  ~read_input: Input -> Context -> U32 -> Input & InputSample,
  ~read_weights: Storage -> Context -> U32 -> Storage & WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Storage -> Storage -> Storage & Storage,
  index: U32,layout: Layout,buffers: Buffers<ownership,Input,Storage>) -> Buffers<ownership,Input,Storage>:
  execute_job(~ownership,~Input,~Storage,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,~write,
    job(layout,index),buffers)

def run(~ownership: Quant,~Input: Kind(ownership),~Storage: Kind(ownership),~InputSample: Data,~WeightSample: Data,
  ~Layout: Data,~Context: Data,~Accumulator: Data,
  ~job: Layout -> U32 -> Job<Context,Accumulator>,
  ~read_input: Input -> Context -> U32 -> Input & InputSample,
  ~read_weights: Storage -> Context -> U32 -> Storage & WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Storage -> Storage -> Storage & Storage,
  count: Nat,first: U32,layout: Layout,buffers: Buffers<ownership,Input,Storage>) -> Buffers<ownership,Input,Storage>:
  Traversal.indexed_fold(~Buffers<ownership,Input,Storage>,~Layout,
    ~output_step(~ownership,~Input,~Storage,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,
      ~job,~read_input,~read_weights,~update,~write),count,first,layout,buffers)

def output(~ownership: Quant,~Input: Kind(ownership),~Storage: Kind(ownership),buffers: Buffers<ownership,Input,Storage>) -> Storage:
  Buffers{input,weights,bias,output} = buffers
  output
