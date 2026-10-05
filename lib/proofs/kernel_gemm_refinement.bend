# Execution refinement for the whole GEMM schedule. The native flat loop
# refines the shared ordered reduction; stores use the actual output model.
# Address coverage is separate.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../kernel_rows.bend as Rows
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../traversal_reduction.bend as Reduction
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./loop_projection.bend as Projection
import ./kernel_reduction_model.bend as ReductionModel
import ./kernel_output_refinement.bend as KernelOutputLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./write_program.bend as Writes

type Buffers is Data:
  Buffers{input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Writes.Program<F32>}

def represent(buffers: Buffers) -> Gemm.Buffers:
  Buffers{input,weights,bias,output} = buffers
  Gemm.Buffers{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output))}

# A model reduction state as the native loop state. The loop index is the
# model weight index; the model input index is the input base plus that index.
def represent_loop(bias: Tree.Tree<F32>,output: Writes.Program<F32>,context: Gemm.Context,state: ReductionModel.State()) -> Gemm.Loop:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
    weight_index,context,accumulators}

def step_matches(~upper: Bool,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,
  +index: U32,+row: U32,+position: U32,+rows: U32,+columns: U32,+input_base: U32,+weight_base: U32,+accumulators: Rows.Rows<Tile.Tile8>) ->
  {Gemm.step(~upper,Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
      index,Gemm.Context{row,position,rows,columns,input_base,weight_base},accumulators}) ==
    Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
      (index + 1 : U32),Gemm.Context{row,position,rows,columns,input_base,weight_base},
      ReductionModel.update(weight_base,accumulators,ReductionModel.input_tile(input,weight_base,(input_base + index : U32)),ReductionModel.weights_half(~upper,weights,weight_base,index))} :
    Gemm.Loop}:
  -input_array = Tree.pack(Tile.Tile8,input)
  -weight_array = Tree.pack(Tile.Tile8,weights)
  -bias_array = Tree.pack(F32,bias)
  -output_array = Tree.pack(F32,Writes.evaluate(F32,output))
  +context = {Gemm.Context{row,position,rows,columns,input_base,weight_base} : Gemm.Context}
  -result = {Gemm.Loop{input_array,weight_array,bias_array,output_array,(index + 1 : U32),context,
    ReductionModel.update(weight_base,accumulators,ReductionModel.input_tile(input,weight_base,(input_base + index : U32)),ReductionModel.weights_half(~upper,weights,weight_base,index))} : Gemm.Loop}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,weight_array,(weight_base + index : U32)),
    (weight_array,Tree.read(Tile.Tile8,weights,(weight_base + index : U32))),StorageLaws.read(Tile.Tile8,weights,(weight_base + index : U32))) :
    {Gemm.with_weights(~upper,input_array,bias_array,output_array,index,context,input_base,accumulators,_) == result : Gemm.Loop}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,input_array,(input_base + index : U32)),
    (input_array,Tree.read(Tile.Tile8,input,(input_base + index : U32))),StorageLaws.read(Tile.Tile8,input,(input_base + index : U32))) :
    {Gemm.with_input(weight_array,bias_array,output_array,index,context,accumulators,Rows.half(upper,Tree.read(Tile.Tile8,weights,(weight_base + index : U32))),_) ==
      result : Gemm.Loop}
  {==}

# The native loop reads input at input_base + index and weights at
# weight_base + index; the model advances an absolute input index instead.
def loop_matches(~upper: Bool,count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,
  +index: U32,+row: U32,+position: U32,+rows: U32,+columns: U32,+input_base: U32,+weight_base: U32,+accumulators: Rows.Rows<Tile.Tile8>) ->
  {Loop.run(~Gemm.Loop,~Gemm.step(~upper),count,
      Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
        index,Gemm.Context{row,position,rows,columns,input_base,weight_base},accumulators}) ==
    represent_loop(bias,output,Gemm.Context{row,position,rows,columns,input_base,weight_base},
      ReductionModel.run(~upper,count,Reduction.State{input,weights,(input_base + index : U32),index,weight_base,accumulators})) : Gemm.Loop}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +context = {Gemm.Context{row,position,rows,columns,input_base,weight_base} : Gemm.Context}
      -start = {Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
        index,context,accumulators} : Gemm.Loop}
      +next = ReductionModel.update(weight_base,accumulators,ReductionModel.input_tile(input,weight_base,(input_base + index : U32)),ReductionModel.weights_half(~upper,weights,weight_base,index))
      -stepped = {Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
        (index + 1 : U32),context,next} : Gemm.Loop}
      %Equal.sym(Gemm.Loop,Gemm.step(~upper,start),stepped,
        step_matches(~upper,input,weights,bias,output,index,row,position,rows,columns,input_base,weight_base,accumulators)) :
        {Loop.run(~Gemm.Loop,~Gemm.step(~upper),rest,_) ==
          represent_loop(bias,output,context,ReductionModel.run(~upper,1n+rest,Reduction.State{input,weights,(input_base + index : U32),index,weight_base,accumulators})) : Gemm.Loop}
      %Equal.sym(U32,((input_base + index : U32) + 1 : U32),(input_base + (index + 1 : U32) : U32),Arithmetic.add_associative(input_base,index,1)) :
        {Loop.run(~Gemm.Loop,~Gemm.step(~upper),rest,stepped) ==
          represent_loop(bias,output,context,ReductionModel.run(~upper,rest,Reduction.State{input,weights,_,(index + 1 : U32),weight_base,next})) : Gemm.Loop}
      loop_matches(~upper,rest,input,weights,bias,output,(index + 1 : U32),row,position,rows,columns,input_base,weight_base,next)

def after_write(input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,owners: KernelOutputLaws.Owners) -> Buffers:
  KernelOutputLaws.Owners{bias,output} = owners
  Buffers{input,weights,bias,output}

def written(+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+owners: KernelOutputLaws.Owners) ->
  {Gemm.after_write(Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),KernelOutputLaws.represent(owners)) == represent(after_write(input,weights,owners)) : Gemm.Buffers}:
  KernelOutputLaws.Owners{bias,output} = owners
  {==}

def finish(bias: Tree.Tree<F32>,output: Writes.Program<F32>,context: Gemm.Context,state: ReductionModel.State()) -> Buffers:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  after_write(input,weights,KernelOutputLaws.write(context,accumulators,bias,output))

def finished(+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+context: Gemm.Context,+state: ReductionModel.State()) ->
  {Gemm.finish(represent_loop(bias,output,context,state)) == represent(finish(bias,output,context,state)) : Gemm.Buffers}:
  Reduction.State{+input,+weights,input_index,weight_index,base,+accumulators} = state
  %Equal.sym(Array<F32> & Array<F32>,Gemm.write(context,accumulators,Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output))),
    KernelOutputLaws.represent(KernelOutputLaws.write(context,accumulators,bias,output)),KernelOutputLaws.write_matches(context,accumulators,bias,output)) :
    {Gemm.after_write(Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),_) ==
      represent(after_write(input,weights,KernelOutputLaws.write(context,accumulators,bias,output))) : Gemm.Buffers}
  written(input,weights,KernelOutputLaws.write(context,accumulators,bias,output))

def reduce(upper: Bool,count: Nat,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,context: Gemm.Context) -> ReductionModel.State():
  Gemm.Context{row,position,rows,columns,+input_base,+weight_base} = context
  ReductionModel.selected(upper,count,ReductionModel.initial(input,weights,input_base,0,weight_base))

def reduced(~upper: Bool,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,
  +row: U32,+position: U32,+rows: U32,+columns: U32,+input_base: U32,+weight_base: U32) ->
  {Loop.run_then(~Gemm.Loop,~Gemm.Buffers,~Gemm.step(~upper),~Gemm.finish,count,
      Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
        0,Gemm.Context{row,position,rows,columns,input_base,weight_base},Rows.filled(~Tile.Tile8,Tile.filled(0.0))}) ==
    represent(finish(bias,output,Gemm.Context{row,position,rows,columns,input_base,weight_base},
      ReductionModel.run(~upper,count,ReductionModel.initial(input,weights,input_base,0,weight_base)))) : Gemm.Buffers}:
  +context = {Gemm.Context{row,position,rows,columns,input_base,weight_base} : Gemm.Context}
  +zeros = Rows.filled(~Tile.Tile8,Tile.filled(0.0))
  -start = {Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
    0,context,zeros} : Gemm.Loop}
  +model = ReductionModel.run(~upper,count,ReductionModel.initial(input,weights,input_base,0,weight_base))
  %Equal.sym(Gemm.Buffers,Loop.run_then(~Gemm.Loop,~Gemm.Buffers,~Gemm.step(~upper),~Gemm.finish,count,start),
    Gemm.finish(Loop.run(~Gemm.Loop,~Gemm.step(~upper),count,start)),
    Projection.run_then_finishes(~Gemm.Loop,~Gemm.Buffers,~Gemm.step(~upper),~Gemm.finish,count,start)) :
    {_ == represent(finish(bias,output,context,model)) : Gemm.Buffers}
  %Equal.sym(Gemm.Loop,Loop.run(~Gemm.Loop,~Gemm.step(~upper),count,start),
    represent_loop(bias,output,context,ReductionModel.run(~upper,count,Reduction.State{input,weights,(input_base + 0 : U32),0,weight_base,zeros})),
    loop_matches(~upper,count,input,weights,bias,output,0,row,position,rows,columns,input_base,weight_base,zeros)) :
    {Gemm.finish(_) == represent(finish(bias,output,context,model)) : Gemm.Buffers}
  %Equal.sym(U32,(input_base + 0 : U32),input_base,Arithmetic.add_zero(input_base)) :
    {Gemm.finish(represent_loop(bias,output,context,ReductionModel.run(~upper,count,Reduction.State{input,weights,_,0,weight_base,zeros}))) ==
      represent(finish(bias,output,context,model)) : Gemm.Buffers}
  finished(bias,output,context,model)

def execute(job: Gemm.Job,buffers: Buffers) -> Buffers:
  Gemm.Job{+context,upper,count} = job
  Buffers{input,weights,bias,output} = buffers
  finish(bias,output,context,reduce(upper,count,input,weights,context))

def executed(+upper: Bool,+count: Nat,+context: Gemm.Context,+buffers: Buffers) ->
  {Gemm.execute(Gemm.Job{context,upper,count},represent(buffers)) == represent(execute(Gemm.Job{context,upper,count},buffers)) : Gemm.Buffers}:
  match upper:
    case False{}:
      Gemm.Context{+row,+position,+rows,+columns,+input_base,+weight_base} = context
      Buffers{+input,+weights,+bias,+output} = buffers
      reduced(~False{},count,input,weights,bias,output,row,position,rows,columns,input_base,weight_base)
    case True{}:
      Gemm.Context{+row,+position,+rows,+columns,+input_base,+weight_base} = context
      Buffers{+input,+weights,+bias,+output} = buffers
      reduced(~True{},count,input,weights,bias,output,row,position,rows,columns,input_base,weight_base)

def execute_matches(+job: Gemm.Job,+buffers: Buffers) ->
  {Gemm.execute(job,represent(buffers)) == represent(execute(job,buffers)) : Gemm.Buffers}:
  Gemm.Job{+context,+upper,+count} = job
  executed(upper,count,context,buffers)

def step(index: U32,layout: Gemm.Layout,buffers: Buffers) -> Buffers:
  execute(Gemm.job(layout,index),buffers)

def step_job_matches(+index: U32,+layout: Gemm.Layout,+buffers: Buffers) ->
  {Gemm.step_job(index,layout,represent(buffers)) == represent(step(index,layout,buffers)) : Gemm.Buffers}:
  execute_matches(Gemm.job(layout,index),buffers)

def result(buffers: Buffers) -> Tree.Tree<F32>:
  Buffers{input,weights,bias,output} = buffers
  Writes.evaluate(F32,output)

def result_matches(+buffers: Buffers) ->
  {Gemm.result(represent(buffers)) == Tree.pack(F32,result(buffers)) : Array<F32>}:
  Buffers{input,weights,bias,output} = buffers
  {==}

def run(+rows: U32,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Tree.Tree<F32>,+columns: U32,inner: U32) -> Tree.Tree<F32>:
  result(Traversal.indexed_fold(~Buffers,~Gemm.Layout,~step,U32.to_nat(Gemm.job_count(rows,columns)),0,
    Gemm.Layout{rows,columns,inner},Buffers{input,weights,bias,Writes.Initial{output}}))

def run_matches(+rows: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+columns: U32,+inner: U32) ->
  {Gemm.run(rows,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),columns,inner) ==
    Tree.pack(F32,run(rows,input,weights,bias,output,columns,inner)) : Array<F32>}:
  +count = U32.to_nat(Gemm.job_count(rows,columns))
  +layout = {Gemm.Layout{rows,columns,inner} : Gemm.Layout}
  +buffers = {Buffers{input,weights,bias,Writes.Initial{output}} : Buffers}
  +final = Traversal.indexed_fold(~Buffers,~Gemm.Layout,~step,count,0,layout,buffers)
  %Equal.sym(Gemm.Buffers,Traversal.indexed_fold(~Gemm.Buffers,~Gemm.Layout,~Gemm.step_job,count,0,layout,represent(buffers)),
    represent(final),ArrayLaws.indexed_fold_refines_model(~Gemm.Buffers,~Buffers,~Gemm.Layout,~Gemm.step_job,~step,~represent,~step_job_matches,count,0,layout,buffers)) :
    {Gemm.result(_) == Tree.pack(F32,result(final)) : Array<F32>}
  result_matches(final)
