# The actual output driver and its storage observation use the same algorithm.
# Readers and the local writer are the only representation-specific premises.
import Base
import ../traversal_output.bend as OutputTraversal
import ../traversal_reduction.bend as Reduction
import ./storage_tree_model.bend as Tree
import ./traversal_reduction_refinement.bend as ReductionLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ../traversal_array.bend as Traversal

# Sources are fixed throughout a traversal. This is a proof observation of the
# existing buffers, not another runtime container or a copy of their contents.
type Sources<-Input: Data,-Value: Data> is Data:
  Sources{input: Tree.Tree<Input>,weights: Tree.Tree<Value>,bias: Tree.Tree<Value>}

def with_output(~Input: Data,~Value: Data,source: Sources<Input,Value>,output: Tree.Tree<Value>) -> OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>:
  Sources{input,weights,bias} = source
  OutputTraversal.Buffers{input,weights,bias,output}

def represent(~Input: Data,~Value: Data,buffers: OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>) ->
  OutputTraversal.Buffers<&1,Array<Input>,Array<Value>>:
  OutputTraversal.Buffers{input,weights,bias,output} = buffers
  OutputTraversal.Buffers{Tree.pack(Input,input),Tree.pack(Value,weights),Tree.pack(Value,bias),Tree.pack(Value,output)}

def output_refines_model(~Input: Data,~Value: Data,+buffers: OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>) ->
  {OutputTraversal.output(~&1,~Array<Input>,~Array<Value>,represent(~Input,~Value,buffers)) ==
    Tree.pack(Value,OutputTraversal.output(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,buffers)) : Array<Value>}:
  OutputTraversal.Buffers{input,weights,bias,output} = buffers
  {==}

def observe_write(~Value: Data,~Context: Data,~Accumulator: Data,
  ~write: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  context: Context,accumulator: Accumulator,bias: Tree.Tree<Value>,output: Tree.Tree<Value>) -> Tree.Tree<Value> & Tree.Tree<Value>:
  +bias = bias
  (bias,write(context,accumulator,bias,output))

def after_reduction_refines_model(~Input: Data,~Value: Data,~Context: Data,~Accumulator: Data,
  ~write: Context -> Accumulator -> Array<Value> -> Array<Value> -> Array<Value> & Array<Value>,
  ~write_values: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  ~write_matches: @+context: Context -> @+accumulator: Accumulator -> @+bias: Tree.Tree<Value> -> @+output: Tree.Tree<Value> ->
    {write(context,accumulator,Tree.pack(Value,bias),Tree.pack(Value,output)) ==
      (Tree.pack(Value,bias),Tree.pack(Value,write_values(context,accumulator,bias,output))) : Array<Value> & Array<Value>},
  +bias: Tree.Tree<Value>,+output: Tree.Tree<Value>,+state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Value>,Context,Accumulator>) ->
  {OutputTraversal.after_reduction(~&1,~Array<Input>,~Array<Value>,~Context,~Accumulator,~write,
      Tree.pack(Value,bias),Tree.pack(Value,output),ReductionLaws.pack(~Input,~Value,~Context,~Accumulator,state)) ==
    represent(~Input,~Value,OutputTraversal.after_reduction(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~Context,~Accumulator,
      ~observe_write(~Value,~Context,~Accumulator,~write_values),bias,output,state)) : OutputTraversal.Buffers<&1,Array<Input>,Array<Value>>}:
  Reduction.State{+input,+weights,input_index,weight_index,+context,+accumulator} = state
  %Equal.sym(Array<Value> & Array<Value>,write(context,accumulator,Tree.pack(Value,bias),Tree.pack(Value,output)),
    (Tree.pack(Value,bias),Tree.pack(Value,write_values(context,accumulator,bias,output))),write_matches(context,accumulator,bias,output)) :
    {OutputTraversal.after_write(~&1,~Array<Input>,~Array<Value>,Tree.pack(Input,input),Tree.pack(Value,weights),_) ==
      OutputTraversal.Buffers{Tree.pack(Input,input),Tree.pack(Value,weights),Tree.pack(Value,bias),Tree.pack(Value,write_values(context,accumulator,bias,output))} :
      OutputTraversal.Buffers<&1,Array<Input>,Array<Value>>}
  {==}

def execute_job_refines_model(~Input: Data,~Value: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~read_input: Array<Input> -> Context -> U32 -> Array<Input> & InputSample,
  ~read_weights: Array<Value> -> Context -> U32 -> Array<Value> & WeightSample,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Value> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Array<Value> -> Array<Value> -> Array<Value> & Array<Value>,
  ~write_values: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  ~input_read: @+tree: Tree.Tree<Input> -> @+context: Context -> @+index: U32 ->
    {read_input(Tree.pack(Input,tree),context,index) == (Tree.pack(Input,tree),sample_input(tree,context,index)) : Array<Input> & InputSample},
  ~weight_read: @+tree: Tree.Tree<Value> -> @+context: Context -> @+index: U32 ->
    {read_weights(Tree.pack(Value,tree),context,index) == (Tree.pack(Value,tree),sample_weights(tree,context,index)) : Array<Value> & WeightSample},
  ~write_matches: @+context: Context -> @+accumulator: Accumulator -> @+bias: Tree.Tree<Value> -> @+output: Tree.Tree<Value> ->
    {write(context,accumulator,Tree.pack(Value,bias),Tree.pack(Value,output)) ==
      (Tree.pack(Value,bias),Tree.pack(Value,write_values(context,accumulator,bias,output))) : Array<Value> & Array<Value>},
  +job: OutputTraversal.Job<Context,Accumulator>,+buffers: OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>) ->
  {OutputTraversal.execute_job(~&1,~Array<Input>,~Array<Value>,~InputSample,~WeightSample,~Context,~Accumulator,
      ~read_input,~read_weights,~update,~write,job,represent(~Input,~Value,buffers)) ==
    represent(~Input,~Value,OutputTraversal.execute_job(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Context,~Accumulator,
      ~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
      ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,
      ~observe_write(~Value,~Context,~Accumulator,~write_values),job,buffers)) : OutputTraversal.Buffers<&1,Array<Input>,Array<Value>>}:
  OutputTraversal.Job{+context,+count,+input_index,+weight_index,+initial} = job
  OutputTraversal.Buffers{+input,+weights,+bias,+output} = buffers
  +state = {Reduction.State{input,weights,input_index,weight_index,context,initial} : Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Value>,Context,Accumulator>}
  +model_result = Reduction.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Context,~Accumulator,
    ~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
    ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,count,state)
  %Equal.sym(Reduction.State<&1,Array<Input>,Array<Value>,Context,Accumulator>,
    Reduction.run(~&1,~Array<Input>,~Array<Value>,~InputSample,~WeightSample,~Context,~Accumulator,~read_input,~read_weights,~update,
      count,ReductionLaws.pack(~Input,~Value,~Context,~Accumulator,state)),ReductionLaws.pack(~Input,~Value,~Context,~Accumulator,model_result),
    ReductionLaws.run_refines_model(~Input,~Value,~InputSample,~WeightSample,~Context,~Accumulator,
      ~read_input,~read_weights,~sample_input,~sample_weights,~update,~input_read,~weight_read,count,state)) :
    {OutputTraversal.after_reduction(~&1,~Array<Input>,~Array<Value>,~Context,~Accumulator,~write,Tree.pack(Value,bias),Tree.pack(Value,output),_) ==
      represent(~Input,~Value,OutputTraversal.after_reduction(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~Context,~Accumulator,
        ~observe_write(~Value,~Context,~Accumulator,~write_values),bias,output,model_result)) : OutputTraversal.Buffers<&1,Array<Input>,Array<Value>>}
  after_reduction_refines_model(~Input,~Value,~Context,~Accumulator,~write,~write_values,~write_matches,bias,output,model_result)

# The job proof is independent of how many jobs the caller requests. Both
# strategies instantiate this theorem; neither implements another induction.
def run_refines_model(~Input: Data,~Value: Data,~InputSample: Data,~WeightSample: Data,~Layout: Data,~Context: Data,~Accumulator: Data,
  ~job: Layout -> U32 -> OutputTraversal.Job<Context,Accumulator>,
  ~read_input: Array<Input> -> Context -> U32 -> Array<Input> & InputSample,
  ~read_weights: Array<Value> -> Context -> U32 -> Array<Value> & WeightSample,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Value> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Array<Value> -> Array<Value> -> Array<Value> & Array<Value>,
  ~write_values: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  ~input_read: @+tree: Tree.Tree<Input> -> @+context: Context -> @+index: U32 ->
    {read_input(Tree.pack(Input,tree),context,index) == (Tree.pack(Input,tree),sample_input(tree,context,index)) : Array<Input> & InputSample},
  ~weight_read: @+tree: Tree.Tree<Value> -> @+context: Context -> @+index: U32 ->
    {read_weights(Tree.pack(Value,tree),context,index) == (Tree.pack(Value,tree),sample_weights(tree,context,index)) : Array<Value> & WeightSample},
  ~write_matches: @+context: Context -> @+accumulator: Accumulator -> @+bias: Tree.Tree<Value> -> @+output: Tree.Tree<Value> ->
    {write(context,accumulator,Tree.pack(Value,bias),Tree.pack(Value,output)) ==
      (Tree.pack(Value,bias),Tree.pack(Value,write_values(context,accumulator,bias,output))) : Array<Value> & Array<Value>},
  +count: Nat,+first: U32,+layout: Layout,+buffers: OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>) ->
  {OutputTraversal.run(~&1,~Array<Input>,~Array<Value>,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,
      ~job,~read_input,~read_weights,~update,~write,count,first,layout,represent(~Input,~Value,buffers)) ==
    represent(~Input,~Value,OutputTraversal.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,
      ~job,~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
      ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,
      ~observe_write(~Value,~Context,~Accumulator,~write_values),count,first,layout,buffers)) : OutputTraversal.Buffers<&1,Array<Input>,Array<Value>>}:
  ArrayLaws.indexed_fold_refines_model(~OutputTraversal.Buffers<&1,Array<Input>,Array<Value>>,~OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>,~Layout,
    ~OutputTraversal.output_step(~&1,~Array<Input>,~Array<Value>,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,
      ~job,~read_input,~read_weights,~update,~write),
    ~OutputTraversal.output_step(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,
      ~job,~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
      ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,~observe_write(~Value,~Context,~Accumulator,~write_values)),
    ~represent(~Input,~Value),
    ~(index => layout => state => execute_job_refines_model(~Input,~Value,~InputSample,~WeightSample,~Context,~Accumulator,
      ~read_input,~read_weights,~sample_input,~sample_weights,~update,~write,~write_values,~input_read,~weight_read,~write_matches,job(layout,index),state)),
    count,first,layout,buffers)


def finish_observation(~Input: Data,~Value: Data,~Context: Data,~Accumulator: Data,
  ~write: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  source: ReductionLaws.SampleSource<Input,Value,Context>,value: Accumulator,bias: Tree.Tree<Value>,output: Tree.Tree<Value>) ->
  OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>:
  ReductionLaws.SampleSource{input,weights,context} = source
  +bias = bias
  OutputTraversal.Buffers{input,weights,bias,write(context,value,bias,output)}

def finish_matches_observation(~Input: Data,~Value: Data,~Context: Data,~Accumulator: Data,
  ~write: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  +bias: Tree.Tree<Value>,+output: Tree.Tree<Value>,+state: Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Value>,Context,Accumulator>) ->
  {OutputTraversal.after_reduction(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~Context,~Accumulator,~observe_write(~Value,~Context,~Accumulator,~write),bias,output,state) ==
    finish_observation(~Input,~Value,~Context,~Accumulator,~write,ReductionLaws.source(~Input,~Value,~Context,~Accumulator,state),
      Reduction.accumulator(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~Context,~Accumulator,state),bias,output) : OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>}:
  Reduction.State{input,weights,input_index,weight_index,context,value} = state
  {==}

# A job changes only the output. Its numeric value is still the actual ordered
# Reduction.run, so this normalization never assumes a dot-product result.
def job_accumulator(~Input: Data,~Value: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Value> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  job: OutputTraversal.Job<Context,Accumulator>,source: Sources<Input,Value>) -> Accumulator:
  OutputTraversal.Job{context,count,input_index,weight_index,initial} = job
  Sources{input,weights,bias} = source
  Reduction.accumulator(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~Context,~Accumulator,
    Reduction.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Context,~Accumulator,
      ~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
      ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,
      count,Reduction.State{input,weights,input_index,weight_index,context,initial}))

def job_output(~Input: Data,~Value: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Value> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  job: OutputTraversal.Job<Context,Accumulator>,source: Sources<Input,Value>,output: Tree.Tree<Value>) -> Tree.Tree<Value>:
  OutputTraversal.Job{+context,count,input_index,weight_index,initial} = job
  Sources{input,weights,+bias} = source
  write(context,job_accumulator(~Input,~Value,~InputSample,~WeightSample,~Context,~Accumulator,~sample_input,~sample_weights,~update,
    OutputTraversal.Job{context,count,input_index,weight_index,initial},Sources{input,weights,bias}),bias,output)

def job_changes_only_output(~Input: Data,~Value: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Value> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  +job: OutputTraversal.Job<Context,Accumulator>,+source: Sources<Input,Value>,+output: Tree.Tree<Value>) ->
  {OutputTraversal.execute_job(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Context,~Accumulator,
      ~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
      ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,~observe_write(~Value,~Context,~Accumulator,~write),
      job,with_output(~Input,~Value,source,output)) ==
    with_output(~Input,~Value,source,job_output(~Input,~Value,~InputSample,~WeightSample,~Context,~Accumulator,
      ~sample_input,~sample_weights,~update,~write,job,source,output)) : OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>}:
  OutputTraversal.Job{+context,+count,+input_index,+weight_index,+initial} = job
  Sources{+input,+weights,+bias} = source
  +state = {Reduction.State{input,weights,input_index,weight_index,context,initial} : Reduction.State<&2,Tree.Tree<Input>,Tree.Tree<Value>,Context,Accumulator>}
  +result = Reduction.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Context,~Accumulator,
    ~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
    ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,count,state)
  %ReductionLaws.loop_preserves_source(~Input,~Value,~InputSample,~WeightSample,~Context,~Accumulator,~sample_input,~sample_weights,~update,count,state) :
    {OutputTraversal.after_reduction(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~Context,~Accumulator,~observe_write(~Value,~Context,~Accumulator,~write),bias,output,result) ==
      finish_observation(~Input,~Value,~Context,~Accumulator,~write,_,Reduction.accumulator(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~Context,~Accumulator,result),bias,output) :
      OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>}
  finish_matches_observation(~Input,~Value,~Context,~Accumulator,~write,bias,output,result)

type OutputContext<-Input: Data,-Value: Data,-Layout: Data> is Data:
  OutputContext{layout: Layout,source: Sources<Input,Value>}

def output_update(~Input: Data,~Value: Data,~InputSample: Data,~WeightSample: Data,~Layout: Data,~Context: Data,~Accumulator: Data,
  ~job: Layout -> U32 -> OutputTraversal.Job<Context,Accumulator>,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Value> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  index: U32,context: OutputContext<Input,Value,Layout>,output: Tree.Tree<Value>) -> Tree.Tree<Value>:
  OutputContext{layout,source} = context
  job_output(~Input,~Value,~InputSample,~WeightSample,~Context,~Accumulator,~sample_input,~sample_weights,~update,~write,job(layout,index),source,output)

# The full driver is one output fold over fixed sources. Correctness arguments
# for coverage and writes can therefore ignore owner plumbing and reduction
# bookkeeping. Direct, paired and partial jobs all use this same induction.
def run_is_output_fold(~Input: Data,~Value: Data,~InputSample: Data,~WeightSample: Data,~Layout: Data,~Context: Data,~Accumulator: Data,
  ~job: Layout -> U32 -> OutputTraversal.Job<Context,Accumulator>,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Value> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  +count: Nat,+first: U32,+layout: Layout,+source: Sources<Input,Value>,+output: Tree.Tree<Value>) ->
  {OutputTraversal.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,
      ~job,~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
      ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,~observe_write(~Value,~Context,~Accumulator,~write),
      count,first,layout,with_output(~Input,~Value,source,output)) ==
    with_output(~Input,~Value,source,Traversal.indexed_fold(~Tree.Tree<Value>,~OutputContext<Input,Value,Layout>,
      ~output_update(~Input,~Value,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,~job,~sample_input,~sample_weights,~update,~write),
      count,first,OutputContext{layout,source},output)) : OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +next = job_output(~Input,~Value,~InputSample,~WeightSample,~Context,~Accumulator,~sample_input,~sample_weights,~update,~write,job(layout,first),source,output)
      %Equal.sym(OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>,
        OutputTraversal.execute_job(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Context,~Accumulator,
          ~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
          ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,~observe_write(~Value,~Context,~Accumulator,~write),
          job(layout,first),with_output(~Input,~Value,source,output)),with_output(~Input,~Value,source,next),
        job_changes_only_output(~Input,~Value,~InputSample,~WeightSample,~Context,~Accumulator,~sample_input,~sample_weights,~update,~write,job(layout,first),source,output)) :
        {OutputTraversal.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,
          ~job,~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
          ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,~observe_write(~Value,~Context,~Accumulator,~write),
          rest,(first + 1 : U32),layout,_) ==
          with_output(~Input,~Value,source,Traversal.indexed_fold(~Tree.Tree<Value>,~OutputContext<Input,Value,Layout>,
            ~output_update(~Input,~Value,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,~job,~sample_input,~sample_weights,~update,~write),
            rest,(first + 1 : U32),OutputContext{layout,source},next)) : OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>}
      run_is_output_fold(~Input,~Value,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,~job,~sample_input,~sample_weights,~update,~write,
        rest,(first + 1 : U32),layout,source,next)

def output_fold(~Input: Data,~Value: Data,~InputSample: Data,~WeightSample: Data,~Layout: Data,~Context: Data,~Accumulator: Data,
  ~job: Layout -> U32 -> OutputTraversal.Job<Context,Accumulator>,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Value> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  count: Nat,first: U32,layout: Layout,source: Sources<Input,Value>,output: Tree.Tree<Value>) -> Tree.Tree<Value>:
  Traversal.indexed_fold(~Tree.Tree<Value>,~OutputContext<Input,Value,Layout>,
    ~output_update(~Input,~Value,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,~job,~sample_input,~sample_weights,~update,~write),
    count,first,OutputContext{layout,source},output)

def run_output_is_fold(~Input: Data,~Value: Data,~InputSample: Data,~WeightSample: Data,~Layout: Data,~Context: Data,~Accumulator: Data,
  ~job: Layout -> U32 -> OutputTraversal.Job<Context,Accumulator>,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Value> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  +count: Nat,+first: U32,+layout: Layout,+source: Sources<Input,Value>,+output: Tree.Tree<Value>) ->
  {OutputTraversal.output(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,
      OutputTraversal.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,
        ~job,~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
        ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,~observe_write(~Value,~Context,~Accumulator,~write),
        count,first,layout,with_output(~Input,~Value,source,output))) ==
    output_fold(~Input,~Value,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,~job,~sample_input,~sample_weights,~update,~write,
      count,first,layout,source,output) : Tree.Tree<Value>}:
  Sources{+input,+weights,+bias} = source
  +source = {Sources{input,weights,bias} : Sources<Input,Value>}
  %Equal.sym(OutputTraversal.Buffers<&2,Tree.Tree<Input>,Tree.Tree<Value>>,
    OutputTraversal.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,
      ~job,~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
      ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,~observe_write(~Value,~Context,~Accumulator,~write),
      count,first,layout,with_output(~Input,~Value,source,output)),
    with_output(~Input,~Value,source,output_fold(~Input,~Value,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,~job,~sample_input,~sample_weights,~update,~write,
      count,first,layout,source,output)),
    run_is_output_fold(~Input,~Value,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,~job,~sample_input,~sample_weights,~update,~write,
      count,first,layout,source,output)) :
    {OutputTraversal.output(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,_) ==
      output_fold(~Input,~Value,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,~job,~sample_input,~sample_weights,~update,~write,
        count,first,layout,source,output) : Tree.Tree<Value>}
  {==}

# Structural output properties (for example balanced storage and capacity) are
# independent of the reduction, local block width and traversal length.
def job_preserves_property(~Input: Data,~Value: Data,~InputSample: Data,~WeightSample: Data,~Context: Data,~Accumulator: Data,~Parameter: Data,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Value> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  ~Property: Parameter -> Tree.Tree<Value> -> Data,
  ~preserve: @+parameter: Parameter -> @+context: Context -> @+accumulator: Accumulator -> @+bias: Tree.Tree<Value> -> @+output: Tree.Tree<Value> ->
    Property(parameter,output) -> Property(parameter,write(context,accumulator,bias,output)),
  +parameter: Parameter,+job: OutputTraversal.Job<Context,Accumulator>,+source: Sources<Input,Value>,+output: Tree.Tree<Value>,evidence: Property(parameter,output)) ->
  Property(parameter,job_output(~Input,~Value,~InputSample,~WeightSample,~Context,~Accumulator,~sample_input,~sample_weights,~update,~write,job,source,output)):
  OutputTraversal.Job{+context,count,input_index,weight_index,initial} = job
  Sources{input,weights,+bias} = source
  preserve(parameter,context,Reduction.accumulator(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~Context,~Accumulator,
    Reduction.run(~&2,~Tree.Tree<Input>,~Tree.Tree<Value>,~InputSample,~WeightSample,~Context,~Accumulator,
      ~ReductionLaws.observe_read(~Tree.Tree<Input>,~InputSample,~Context,~sample_input),
      ~ReductionLaws.observe_read(~Tree.Tree<Value>,~WeightSample,~Context,~sample_weights),~update,
      count,Reduction.State{input,weights,input_index,weight_index,context,initial})),bias,output,evidence)

def output_update_preserves_property(~Input: Data,~Value: Data,~InputSample: Data,~WeightSample: Data,~Layout: Data,~Context: Data,~Accumulator: Data,~Parameter: Data,
  ~job: Layout -> U32 -> OutputTraversal.Job<Context,Accumulator>,
  ~sample_input: Tree.Tree<Input> -> Context -> U32 -> InputSample,
  ~sample_weights: Tree.Tree<Value> -> Context -> U32 -> WeightSample,
  ~update: Context -> Accumulator -> InputSample -> WeightSample -> Accumulator,
  ~write: Context -> Accumulator -> Tree.Tree<Value> -> Tree.Tree<Value> -> Tree.Tree<Value>,
  ~Property: Parameter -> Tree.Tree<Value> -> Data,
  ~preserve: @+parameter: Parameter -> @+context: Context -> @+accumulator: Accumulator -> @+bias: Tree.Tree<Value> -> @+output: Tree.Tree<Value> ->
    Property(parameter,output) -> Property(parameter,write(context,accumulator,bias,output)),
  +context: OutputContext<Input,Value,Layout>,+parameter: Parameter,+index: U32,+output: Tree.Tree<Value>,evidence: Property(parameter,output)) ->
  Property(parameter,output_update(~Input,~Value,~InputSample,~WeightSample,~Layout,~Context,~Accumulator,~job,~sample_input,~sample_weights,~update,~write,index,context,output)):
  OutputContext{layout,+source} = context
  job_preserves_property(~Input,~Value,~InputSample,~WeightSample,~Context,~Accumulator,~Parameter,
    ~sample_input,~sample_weights,~update,~write,~Property,~preserve,parameter,job(layout,index),source,output,evidence)

