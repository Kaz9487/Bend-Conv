# Representation obligations for the two strategies of OutputKernel.
# These theorems cover actual readers and writers on arbitrary supplied storage.
# Convolution geometry and the final matrix result are separate obligations.
import Base
import ../scalar_kernel.bend as ScalarConv
import ../shape.bend as ConvShape
import ../../lib/storage_buffer.bend as Storage
import ../../lib/traversal_reduction.bend as Reduction
import ../../lib/math_fp32.bend as Fp32
import ../../lib/traversal_output.bend as OutputTraversal
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/traversal_reduction_refinement.bend as ReductionLaws
import ../../lib/proofs/traversal_output_refinement.bend as OutputLaws
import ../../lib/proofs/storage_shape.bend as Shape

def bounded_value(+tree: Tree.Tree<F32>,+index: U32,length: U32) -> F32:
  +length = length
  Bool.pick(F32,U32.is_lt(index,length) && U32.is_le(length,Tree.capacity(F32,tree)),Tree.read(F32,tree,index),0.0)

def address_value(tree: Tree.Tree<F32>,address: ScalarConv.SampleAddress) -> F32:
  ScalarConv.SampleAddress{inside,index,length} = address
  Bool.pick(F32,inside,bounded_value(tree,index,length),0.0)

def address_read(+tree: Tree.Tree<F32>,+address: ScalarConv.SampleAddress) ->
  {ScalarConv.read_address(~Array<F32>,~Storage.read_or(F32,0.0),Tree.pack(F32,tree),address) ==
    (Tree.pack(F32,tree),address_value(tree,address)) : Array<F32> & F32}:
  ScalarConv.SampleAddress{inside,+index,+length} = address
  match inside:
    case False{}: {==}
    case True{}: StorageLaws.bounded_read(~F32,0.0,tree,index,length)

def direct_input_value(tree: Tree.Tree<F32>,context: ScalarConv.DirectContext,index: U32) -> F32:
  address_value(tree,ScalarConv.direct_address(context,index))

def direct_weight_value(tree: Tree.Tree<F32>,context: ScalarConv.DirectContext,index: U32) -> F32:
  address_value(tree,ScalarConv.weight_address(context,index))

def direct_input_read(+tree: Tree.Tree<F32>,+context: ScalarConv.DirectContext,+index: U32) ->
  {ScalarConv.direct_input(~Array<F32>,~Storage.read_or(F32,0.0),Tree.pack(F32,tree),context,index) ==
    (Tree.pack(F32,tree),direct_input_value(tree,context,index)) : Array<F32> & F32}:
  address_read(tree,ScalarConv.direct_address(context,index))

def direct_weight_read(+tree: Tree.Tree<F32>,+context: ScalarConv.DirectContext,+index: U32) ->
  {ScalarConv.direct_weight(~Array<F32>,~Storage.read_or(F32,0.0),Tree.pack(F32,tree),context,index) ==
    (Tree.pack(F32,tree),direct_weight_value(tree,context,index)) : Array<F32> & F32}:
  address_read(tree,ScalarConv.weight_address(context,index))

def direct_write_values(context: ScalarConv.DirectContext,accumulator: F32,bias: Tree.Tree<F32>,output: Tree.Tree<F32>) -> Tree.Tree<F32>:
  ScalarConv.DirectContext{ConvShape.Shape{input_channel_count,output_channel_count,h,w,kernel,stride,padding,output_height,output_width},index,channel,row,column} = context
  Tree.write(F32,output,index,(accumulator + bounded_value(bias,channel,output_channel_count) : F32))

def direct_write_matches(+context: ScalarConv.DirectContext,+accumulator: F32,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>) ->
  {ScalarConv.direct_write(~Array<F32>,~Storage.read_or(F32,0.0),~Array.set(F32),context,accumulator,Tree.pack(F32,bias),Tree.pack(F32,output)) ==
    (Tree.pack(F32,bias),Tree.pack(F32,direct_write_values(context,accumulator,bias,output))) : Array<F32> & Array<F32>}:
  ScalarConv.DirectContext{ConvShape.Shape{input_channel_count,+output_channel_count,h,w,kernel,stride,padding,output_height,output_width},+index,+channel,row,column} = context
  %Equal.sym(Array<F32> & F32,Storage.read_or(F32,0.0,Tree.pack(F32,bias),channel,output_channel_count),
    (Tree.pack(F32,bias),bounded_value(bias,channel,output_channel_count)),StorageLaws.bounded_read(~F32,0.0,bias,channel,output_channel_count)) :
    {ScalarConv.direct_written(~Array<F32>,~Array.set(F32),Tree.pack(F32,output),index,accumulator,_) ==
      (Tree.pack(F32,bias),Tree.pack(F32,Tree.write(F32,output,index,(accumulator + bounded_value(bias,channel,output_channel_count) : F32)))) : Array<F32> & Array<F32>}
  %StorageLaws.write(F32,output,index,(accumulator + bounded_value(bias,channel,output_channel_count) : F32)) :
    {(Tree.pack(F32,bias),Array.set(F32,Tree.pack(F32,output),index,(accumulator + bounded_value(bias,channel,output_channel_count) : F32))) ==
      (Tree.pack(F32,bias),_) : Array<F32> & Array<F32>}
  {==}

# These are full driver instances, not one-step or fixed-length examples.
# No premises ask the caller to assume that reads or writes are correct.
def direct_run_refines_storage(+count: Nat,+first: U32,+shape: ConvShape.Shape,+buffers: OutputTraversal.Buffers<&2,Tree.Tree<F32>,Tree.Tree<F32>>) ->
  {OutputTraversal.run(~&1,~Array<F32>,~Array<F32>,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,
      ~ScalarConv.direct_job,~ScalarConv.direct_input(~Array<F32>,~Storage.read_or(F32,0.0)),~ScalarConv.direct_weight(~Array<F32>,~Storage.read_or(F32,0.0)),~ScalarConv.direct_update,
      ~ScalarConv.direct_write(~Array<F32>,~Storage.read_or(F32,0.0),~Array.set(F32)),count,first,shape,OutputLaws.represent(~F32,~F32,buffers)) ==
    OutputLaws.represent(~F32,~F32,OutputTraversal.run(~&2,~Tree.Tree<F32>,~Tree.Tree<F32>,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,
      ~ScalarConv.direct_job,~ReductionLaws.observe_read(~Tree.Tree<F32>,~F32,~ScalarConv.DirectContext,~direct_input_value),
      ~ReductionLaws.observe_read(~Tree.Tree<F32>,~F32,~ScalarConv.DirectContext,~direct_weight_value),~ScalarConv.direct_update,
      ~OutputLaws.observe_write(~F32,~ScalarConv.DirectContext,~F32,~direct_write_values),count,first,shape,buffers)) : OutputTraversal.Buffers<&1,Array<F32>,Array<F32>>}:
  OutputLaws.run_refines_model(~F32,~F32,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,~ScalarConv.direct_job,
    ~ScalarConv.direct_input(~Array<F32>,~Storage.read_or(F32,0.0)),~ScalarConv.direct_weight(~Array<F32>,~Storage.read_or(F32,0.0)),
    ~direct_input_value,~direct_weight_value,~ScalarConv.direct_update,~ScalarConv.direct_write(~Array<F32>,~Storage.read_or(F32,0.0),~Array.set(F32)),
    ~direct_write_values,~direct_input_read,~direct_weight_read,~direct_write_matches,count,first,shape,buffers)

# Complete native entry points: these observers execute the same driver, with
# storage observations replacing Array primitives. They are not new algorithms.
def direct_execution(count: Nat,shape: ConvShape.Shape,buffers: OutputTraversal.Buffers<&2,Tree.Tree<F32>,Tree.Tree<F32>>) -> OutputTraversal.Buffers<&2,Tree.Tree<F32>,Tree.Tree<F32>>:
  OutputTraversal.run(~&2,~Tree.Tree<F32>,~Tree.Tree<F32>,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,
    ~ScalarConv.direct_job,~ReductionLaws.observe_read(~Tree.Tree<F32>,~F32,~ScalarConv.DirectContext,~direct_input_value),
    ~ReductionLaws.observe_read(~Tree.Tree<F32>,~F32,~ScalarConv.DirectContext,~direct_weight_value),~ScalarConv.direct_update,
    ~OutputLaws.observe_write(~F32,~ScalarConv.DirectContext,~F32,~direct_write_values),count,0,shape,buffers)

def direct_outputs(count: Nat,first: U32,shape: ConvShape.Shape,source: OutputLaws.Sources<F32,F32>,output: Tree.Tree<F32>) -> Tree.Tree<F32>:
  OutputLaws.output_fold(~F32,~F32,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,
    ~ScalarConv.direct_job,~direct_input_value,~direct_weight_value,~ScalarConv.direct_update,~direct_write_values,count,first,shape,source,output)

def direct_value(+shape: ConvShape.Shape,+index: U32,source: OutputLaws.Sources<F32,F32>) -> F32:
  ConvShape.Shape{input_channel_count,+output_channel_count,h,w,kernel,stride,padding,+output_height,+output_width} = shape
  OutputLaws.Sources{+input,+weights,+bias} = source
  (OutputLaws.job_accumulator(~F32,~F32,~F32,~F32,~ScalarConv.DirectContext,~F32,~direct_input_value,~direct_weight_value,~ScalarConv.direct_update,
    ScalarConv.direct_job(shape,index),OutputLaws.Sources{input,weights,bias}) + bounded_value(bias,(index / (output_height * output_width) : U32),output_channel_count) : F32)

def direct_job_writes_index(+shape: ConvShape.Shape,+index: U32,+source: OutputLaws.Sources<F32,F32>,+output: Tree.Tree<F32>) ->
  {OutputLaws.job_output(~F32,~F32,~F32,~F32,~ScalarConv.DirectContext,~F32,~direct_input_value,~direct_weight_value,~ScalarConv.direct_update,~direct_write_values,
      ScalarConv.direct_job(shape,index),source,output) == Tree.write(F32,output,index,direct_value(shape,index,source)) : Tree.Tree<F32>}:
  ConvShape.Shape{input_channel_count,output_channel_count,h,w,kernel,stride,padding,output_height,output_width} = shape
  OutputLaws.Sources{input,weights,bias} = source
  {==}

def direct_write_preserves_balance(+depth: Nat,+context: ScalarConv.DirectContext,+accumulator: F32,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,
  balanced: Shape.Balanced(F32,depth,output)) -> Shape.Balanced(F32,depth,direct_write_values(context,accumulator,bias,output)):
  ScalarConv.DirectContext{ConvShape.Shape{input_channel_count,+output_channel_count,h,w,kernel,stride,padding,output_height,output_width},+index,+channel,row,column} = context
  Shape.write(F32,depth,output,index,(accumulator + bounded_value(bias,channel,output_channel_count) : F32),balanced)

# Both whole traversals inherit the structural invariant from their local
# writes. Neither path contains another induction over output jobs.
def direct_execution_output_matches_fold(+count: Nat,+shape: ConvShape.Shape,+input: Tree.Tree<F32>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>) ->
  {OutputTraversal.output(~&2,~Tree.Tree<F32>,~Tree.Tree<F32>,direct_execution(count,shape,OutputTraversal.Buffers{input,weights,bias,output})) ==
    direct_outputs(count,0,shape,OutputLaws.Sources{input,weights,bias},output) : Tree.Tree<F32>}:
  OutputLaws.run_output_is_fold(~F32,~F32,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,
    ~ScalarConv.direct_job,~direct_input_value,~direct_weight_value,~ScalarConv.direct_update,~direct_write_values,count,0,shape,OutputLaws.Sources{input,weights,bias},output)

def direct_result(+shape: ConvShape.Shape,input: Tree.Tree<F32>,weights: Tree.Tree<F32>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  ConvShape.Shape{input_channel_count,+output_channel_count,h,w,kernel,stride,padding,+output_height,+output_width} = shape
  +count = (output_channel_count * output_height * output_width : U32)
  direct_outputs(U32.to_nat(count),0,shape,OutputLaws.Sources{input,weights,bias},Tree.fresh(F32,Storage.allocation_depth(count),0.0))

def direct_entry_matches_storage(+shape: ConvShape.Shape,+input: Tree.Tree<F32>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>) ->
  {ScalarConv.run(shape,Tree.pack(F32,input),Tree.pack(F32,weights),Tree.pack(F32,bias)) ==
    Tree.pack(F32,direct_result(shape,input,weights,bias)) : Array<F32>}:
  ConvShape.Shape{+input_channel_count,+output_channel_count,+h,+w,+kernel,+stride,+padding,+output_height,+output_width} = shape
  +count = (output_channel_count * output_height * output_width : U32)
  +output = Tree.fresh(F32,Storage.allocation_depth(count),0.0)
  +buffers = {OutputTraversal.Buffers{input,weights,bias,output} : OutputTraversal.Buffers<&2,Tree.Tree<F32>,Tree.Tree<F32>>}
  %Equal.sym(Array<F32>,Array.new(F32,Storage.allocation_depth(count),0.0),Tree.pack(F32,output),StorageLaws.fresh(F32,Storage.allocation_depth(count),0.0)) :
    {OutputTraversal.output(~&1,~Array<F32>,~Array<F32>,OutputTraversal.run(~&1,~Array<F32>,~Array<F32>,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,
      ~ScalarConv.direct_job,~ScalarConv.direct_input(~Array<F32>,~Storage.read_or(F32,0.0)),~ScalarConv.direct_weight(~Array<F32>,~Storage.read_or(F32,0.0)),~ScalarConv.direct_update,
      ~ScalarConv.direct_write(~Array<F32>,~Storage.read_or(F32,0.0),~Array.set(F32)),U32.to_nat(count),0,shape,
      OutputTraversal.Buffers{Tree.pack(F32,input),Tree.pack(F32,weights),Tree.pack(F32,bias),_})) == Tree.pack(F32,direct_result(shape,input,weights,bias)) : Array<F32>}
  %Equal.sym(OutputTraversal.Buffers<&1,Array<F32>,Array<F32>>,
    OutputTraversal.run(~&1,~Array<F32>,~Array<F32>,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,
      ~ScalarConv.direct_job,~ScalarConv.direct_input(~Array<F32>,~Storage.read_or(F32,0.0)),~ScalarConv.direct_weight(~Array<F32>,~Storage.read_or(F32,0.0)),~ScalarConv.direct_update,
      ~ScalarConv.direct_write(~Array<F32>,~Storage.read_or(F32,0.0),~Array.set(F32)),U32.to_nat(count),0,shape,OutputLaws.represent(~F32,~F32,buffers)),
    OutputLaws.represent(~F32,~F32,direct_execution(U32.to_nat(count),shape,buffers)),direct_run_refines_storage(U32.to_nat(count),0,shape,buffers)) :
    {OutputTraversal.output(~&1,~Array<F32>,~Array<F32>,_) == Tree.pack(F32,direct_result(shape,input,weights,bias)) : Array<F32>}
  %direct_execution_output_matches_fold(U32.to_nat(count),shape,input,weights,bias,output) :
    {OutputTraversal.output(~&1,~Array<F32>,~Array<F32>,OutputLaws.represent(~F32,~F32,direct_execution(U32.to_nat(count),shape,buffers))) == Tree.pack(F32,_) : Array<F32>}
  OutputLaws.output_refines_model(~F32,~F32,direct_execution(U32.to_nat(count),shape,buffers))

def scalar_at(key: Unit,value: F32) -> F32:
  value

def direct_update_matches(+context: ScalarConv.DirectContext,+key: Unit,+accumulator: F32,+input: F32,+weight: F32) ->
  {scalar_at(key,ScalarConv.direct_update(context,accumulator,input,weight)) ==
    Fp32.multiply_add(scalar_at(key,accumulator),scalar_at(key,input),scalar_at(key,weight)) : F32}:
  {==}

# Both strategies use the same ordered-dot theorem. Arbitrary count and key;
# no floating-point associativity, finite test enumeration or lane induction.
def direct_reduction_matches_dot(+count: Nat,+state: Reduction.State<&2,Tree.Tree<F32>,Tree.Tree<F32>,ScalarConv.DirectContext,F32>) ->
  {ReductionLaws.project(~F32,~F32,~ScalarConv.DirectContext,~F32,~Unit,~scalar_at,Unit{},
      Reduction.run(~&2,~Tree.Tree<F32>,~Tree.Tree<F32>,~F32,~F32,~ScalarConv.DirectContext,~F32,
        ~ReductionLaws.observe_read(~Tree.Tree<F32>,~F32,~ScalarConv.DirectContext,~direct_input_value),
        ~ReductionLaws.observe_read(~Tree.Tree<F32>,~F32,~ScalarConv.DirectContext,~direct_weight_value),~ScalarConv.direct_update,count,state)) ==
    ReductionLaws.ordered_dot(~F32,~F32,~F32,~F32,~ScalarConv.DirectContext,~F32,~Unit,
      ~direct_input_value,~direct_weight_value,~scalar_at,~scalar_at,~scalar_at,Unit{},count,state) : F32}:
  ReductionLaws.model_loop_matches_ordered_dot(~F32,~F32,~F32,~F32,~ScalarConv.DirectContext,~F32,~Unit,
    ~direct_input_value,~direct_weight_value,~scalar_at,~scalar_at,~scalar_at,~ScalarConv.direct_update,~direct_update_matches,Unit{},count,state)