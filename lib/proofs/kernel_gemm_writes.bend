# The whole GEMM storage model is one ordered write list. Readback proofs use
# that list; they do not re-derive the job traversal or the store sequence.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ../traversal_reduction.bend as Reduction
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes
import ./kernel_output_refinement.bend as KernelOutputLaws
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_reduction_model.bend as ReductionModel
import ./traversal_reduction_refinement.bend as ReductionLaws

def row_writes(valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+bias: Tree.Tree<F32>) ->
  List<&2,Writes.Assignment<F32>>:
  match valid:
    case False{}: []
    case True{}: KernelOutputLaws.assignments((index * columns + position : U32),Tree.read(F32,bias,index),accumulator)

# Rows are appended in store order: row r writes after rows 0..r-1.
def context_writes(context: Gemm.Context,+accumulators: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  List<&2,Writes.Assignment<F32>>:
  Gemm.Context{+index,+position,+rows,+columns,input_base,weight_base} = context
  List.append(&2,Writes.Assignment<F32>,
    List.append(&2,Writes.Assignment<F32>,
      List.append(&2,Writes.Assignment<F32>,
        row_writes(U32.is_lt(index,rows),index,position,columns,Rows.first(~Tile.Tile8,accumulators),bias),
        row_writes(U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,Rows.second(~Tile.Tile8,accumulators),bias)),
      row_writes(U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,Rows.third(~Tile.Tile8,accumulators),bias)),
    row_writes(U32.is_lt((index + 3 : U32),rows),(index + 3 : U32),position,columns,Rows.fourth(~Tile.Tile8,accumulators),bias))

def owner_bias(owners: KernelOutputLaws.Owners) -> Tree.Tree<F32>:
  KernelOutputLaws.Owners{bias,output} = owners
  bias

def owner_output(owners: KernelOutputLaws.Owners) -> Tree.Tree<F32>:
  KernelOutputLaws.Owners{bias,output} = owners
  Writes.evaluate(F32,output)

def row_bias(+valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+owners: KernelOutputLaws.Owners) ->
  {owner_bias(KernelOutputLaws.row(valid,index,position,columns,accumulator,owners)) == owner_bias(owners) : Tree.Tree<F32>}:
  match valid owners:
    case False{} owners: {==}
    case True{} KernelOutputLaws.Owners{bias,output}: {==}

def row_output(+valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+owners: KernelOutputLaws.Owners) ->
  {owner_output(KernelOutputLaws.row(valid,index,position,columns,accumulator,owners)) ==
    Writes.pure(F32,row_writes(valid,index,position,columns,accumulator,owner_bias(owners)),owner_output(owners)) : Tree.Tree<F32>}:
  match valid owners:
    case False{} owners: {==}
    case True{} KernelOutputLaws.Owners{bias,output}: {==}

def Observed(+owners: KernelOutputLaws.Owners,+bias: Tree.Tree<F32>,+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>) -> Data:
  Shape.Both<{owner_bias(owners) == bias : Tree.Tree<F32>},{owner_output(owners) == Writes.pure(F32,writes,tree) : Tree.Tree<F32>}>

def appended_output(+valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+owners: KernelOutputLaws.Owners,
  +bias: Tree.Tree<F32>,+previous: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,
  same_bias: {owner_bias(owners) == bias : Tree.Tree<F32>},same_output: {owner_output(owners) == Writes.pure(F32,previous,tree) : Tree.Tree<F32>}) ->
  {Writes.pure(F32,row_writes(valid,index,position,columns,accumulator,owner_bias(owners)),owner_output(owners)) ==
    Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,previous,row_writes(valid,index,position,columns,accumulator,bias)),tree) : Tree.Tree<F32>}:
  %Equal.sym(Tree.Tree<F32>,owner_bias(owners),bias,same_bias) :
    {Writes.pure(F32,row_writes(valid,index,position,columns,accumulator,_),owner_output(owners)) ==
      Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,previous,row_writes(valid,index,position,columns,accumulator,bias)),tree) : Tree.Tree<F32>}
  %Equal.sym(Tree.Tree<F32>,owner_output(owners),Writes.pure(F32,previous,tree),same_output) :
    {Writes.pure(F32,row_writes(valid,index,position,columns,accumulator,bias),_) ==
      Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,previous,row_writes(valid,index,position,columns,accumulator,bias)),tree) : Tree.Tree<F32>}
  Equal.sym(Tree.Tree<F32>,
    Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,previous,row_writes(valid,index,position,columns,accumulator,bias)),tree),
    Writes.pure(F32,row_writes(valid,index,position,columns,accumulator,bias),Writes.pure(F32,previous,tree)),
    Writes.concatenated_writes(F32,previous,row_writes(valid,index,position,columns,accumulator,bias),tree))

def next_row(+valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+owners: KernelOutputLaws.Owners,
  +bias: Tree.Tree<F32>,+previous: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,observed: Observed(owners,bias,previous,tree)) ->
  Observed(KernelOutputLaws.row(valid,index,position,columns,accumulator,owners),bias,
    List.append(&2,Writes.Assignment<F32>,previous,row_writes(valid,index,position,columns,accumulator,bias)),tree):
  Shape.Both{+same_bias,+same_output} = observed
  Shape.Both{
    Equal.trans(Tree.Tree<F32>,owner_bias(KernelOutputLaws.row(valid,index,position,columns,accumulator,owners)),owner_bias(owners),bias,
      row_bias(valid,index,position,columns,accumulator,owners),same_bias),
    Equal.trans(Tree.Tree<F32>,owner_output(KernelOutputLaws.row(valid,index,position,columns,accumulator,owners)),
      Writes.pure(F32,row_writes(valid,index,position,columns,accumulator,owner_bias(owners)),owner_output(owners)),
      Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,previous,row_writes(valid,index,position,columns,accumulator,bias)),tree),
      row_output(valid,index,position,columns,accumulator,owners),
      appended_output(valid,index,position,columns,accumulator,owners,bias,previous,tree,same_bias,same_output))}

def write_observed(+context: Gemm.Context,+accumulators: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>) ->
  Observed(KernelOutputLaws.write(context,accumulators,bias,output),bias,context_writes(context,accumulators,bias),Writes.evaluate(F32,output)):
  Gemm.Context{+index,+position,+rows,+columns,input_base,weight_base} = context
  +tree = Writes.evaluate(F32,output)
  +a = Rows.first(~Tile.Tile8,accumulators)
  +b = Rows.second(~Tile.Tile8,accumulators)
  +c = Rows.third(~Tile.Tile8,accumulators)
  +d = Rows.fourth(~Tile.Tile8,accumulators)
  +w0 = row_writes(U32.is_lt(index,rows),index,position,columns,a,bias)
  +w1 = row_writes(U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,b,bias)
  +w2 = row_writes(U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,c,bias)
  +initial = {KernelOutputLaws.Owners{bias,output} : KernelOutputLaws.Owners}
  +first = KernelOutputLaws.row(U32.is_lt(index,rows),index,position,columns,a,initial)
  +second = KernelOutputLaws.row(U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,b,first)
  +third = KernelOutputLaws.row(U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,c,second)
  +two = List.append(&2,Writes.Assignment<F32>,w0,w1)
  +three = List.append(&2,Writes.Assignment<F32>,two,w2)
  +start = {Shape.Both{{==},{==}} : Observed(initial,bias,[],tree)}
  +after_first = next_row(U32.is_lt(index,rows),index,position,columns,a,initial,bias,[],tree,start)
  next_row(U32.is_lt((index + 3 : U32),rows),(index + 3 : U32),position,columns,d,third,bias,three,tree,
    next_row(U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,c,second,bias,two,tree,
      next_row(U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,b,first,bias,w0,tree,after_first)))

def observed_bias(+owners: KernelOutputLaws.Owners,+bias: Tree.Tree<F32>,+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,
  observed: Observed(owners,bias,writes,tree)) -> {owner_bias(owners) == bias : Tree.Tree<F32>}:
  Shape.Both{same_bias,same_output} = observed
  same_bias

def observed_output(+owners: KernelOutputLaws.Owners,+bias: Tree.Tree<F32>,+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,
  observed: Observed(owners,bias,writes,tree)) -> {owner_output(owners) == Writes.pure(F32,writes,tree) : Tree.Tree<F32>}:
  Shape.Both{same_bias,same_output} = observed
  same_output

def state_input(state: ReductionModel.State()) -> Tree.Tree<Tile.Tile8>:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  input

def state_weights(state: ReductionModel.State()) -> Tree.Tree<Tile.Tile8>:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  weights

def source_input(source: ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>) -> Tree.Tree<Tile.Tile8>:
  ReductionLaws.SampleSource{input,weights,context} = source
  input

def source_weights(source: ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>) -> Tree.Tree<Tile.Tile8>:
  ReductionLaws.SampleSource{input,weights,context} = source
  weights

def state_source(state: ReductionModel.State()) -> ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>:
  ReductionLaws.source(~Tile.Tile8,~Tile.Tile8,~U32,~Rows.Rows<Tile.Tile8>,state)

def state_source_input(+state: ReductionModel.State()) -> {state_input(state) == source_input(state_source(state)) : Tree.Tree<Tile.Tile8>}:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  {==}

def state_source_weights(+state: ReductionModel.State()) -> {state_weights(state) == source_weights(state_source(state)) : Tree.Tree<Tile.Tile8>}:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  {==}

def selected_source(+upper: Bool,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+input_index: U32,+weight_index: U32,+base: U32) ->
  {state_source(ReductionModel.selected(upper,count,ReductionModel.initial(input,weights,input_index,weight_index,base))) == ReductionLaws.SampleSource{input,weights,base} :
    ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>}:
  match upper:
    case False{}:
      ReductionLaws.loop_preserves_source(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,
        ~ReductionModel.input_tile,~ReductionModel.weights_half(~False{}),~ReductionModel.update,count,ReductionModel.initial(input,weights,input_index,weight_index,base))
    case True{}:
      ReductionLaws.loop_preserves_source(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,
        ~ReductionModel.input_tile,~ReductionModel.weights_half(~True{}),~ReductionModel.update,count,ReductionModel.initial(input,weights,input_index,weight_index,base))

def reduce_input(+upper: Bool,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+context: Gemm.Context) ->
  {state_input(GemmLaws.reduce(upper,count,input,weights,context)) == input : Tree.Tree<Tile.Tile8>}:
  Gemm.Context{row,position,rows,columns,+input_base,+weight_base} = context
  +state = ReductionModel.selected(upper,count,ReductionModel.initial(input,weights,input_base,0,weight_base))
  Equal.trans(Tree.Tree<Tile.Tile8>,state_input(state),source_input(state_source(state)),input,state_source_input(state),
    Equal.cong(ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>,Tree.Tree<Tile.Tile8>,source => source_input(source),state_source(state),
      ReductionLaws.SampleSource{input,weights,weight_base},selected_source(upper,count,input,weights,input_base,0,weight_base)))

def reduce_weights(+upper: Bool,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+context: Gemm.Context) ->
  {state_weights(GemmLaws.reduce(upper,count,input,weights,context)) == weights : Tree.Tree<Tile.Tile8>}:
  Gemm.Context{row,position,rows,columns,+input_base,+weight_base} = context
  +state = ReductionModel.selected(upper,count,ReductionModel.initial(input,weights,input_base,0,weight_base))
  Equal.trans(Tree.Tree<Tile.Tile8>,state_weights(state),source_weights(state_source(state)),weights,state_source_weights(state),
    Equal.cong(ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>,Tree.Tree<Tile.Tile8>,source => source_weights(source),state_source(state),
      ReductionLaws.SampleSource{input,weights,weight_base},selected_source(upper,count,input,weights,input_base,0,weight_base)))

def buffers_input(buffers: GemmLaws.Buffers) -> Tree.Tree<Tile.Tile8>:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  input

def buffers_weights(buffers: GemmLaws.Buffers) -> Tree.Tree<Tile.Tile8>:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  weights

def buffers_bias(buffers: GemmLaws.Buffers) -> Tree.Tree<F32>:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  bias

def buffers_program(buffers: GemmLaws.Buffers) -> Writes.Program<F32>:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  output

def buffers_eta(+buffers: GemmLaws.Buffers) ->
  {buffers == GemmLaws.Buffers{buffers_input(buffers),buffers_weights(buffers),buffers_bias(buffers),buffers_program(buffers)} : GemmLaws.Buffers}:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  {==}

def job_writes(job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  List<&2,Writes.Assignment<F32>>:
  Gemm.Job{+context,upper,count} = job
  context_writes(context,
    Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,
      GemmLaws.reduce(upper,count,input,weights,context)),bias)

def written_input(+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+owners: KernelOutputLaws.Owners) ->
  {buffers_input(GemmLaws.after_write(input,weights,owners)) == input : Tree.Tree<Tile.Tile8>}:
  KernelOutputLaws.Owners{bias,output} = owners
  {==}

def written_weights(+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+owners: KernelOutputLaws.Owners) ->
  {buffers_weights(GemmLaws.after_write(input,weights,owners)) == weights : Tree.Tree<Tile.Tile8>}:
  KernelOutputLaws.Owners{bias,output} = owners
  {==}

def written_bias(+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+owners: KernelOutputLaws.Owners) ->
  {buffers_bias(GemmLaws.after_write(input,weights,owners)) == owner_bias(owners) : Tree.Tree<F32>}:
  KernelOutputLaws.Owners{bias,output} = owners
  {==}

def written_output(+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+owners: KernelOutputLaws.Owners) ->
  {Writes.evaluate(F32,buffers_program(GemmLaws.after_write(input,weights,owners))) == owner_output(owners) : Tree.Tree<F32>}:
  KernelOutputLaws.Owners{bias,output} = owners
  {==}

def reduced_input(+context: Gemm.Context,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+state: ReductionModel.State()) ->
  {buffers_input(GemmLaws.finish(bias,output,context,state)) == state_input(state) : Tree.Tree<Tile.Tile8>}:
  Reduction.State{+input,+weights,input_index,weight_index,base,+accumulators} = state
  written_input(input,weights,KernelOutputLaws.write(context,accumulators,bias,output))

def reduced_weights(+context: Gemm.Context,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+state: ReductionModel.State()) ->
  {buffers_weights(GemmLaws.finish(bias,output,context,state)) == state_weights(state) : Tree.Tree<Tile.Tile8>}:
  Reduction.State{+input,+weights,input_index,weight_index,base,+accumulators} = state
  written_weights(input,weights,KernelOutputLaws.write(context,accumulators,bias,output))

def reduced_bias(+context: Gemm.Context,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+state: ReductionModel.State()) ->
  {buffers_bias(GemmLaws.finish(bias,output,context,state)) == bias : Tree.Tree<F32>}:
  Reduction.State{+input,+weights,input_index,weight_index,base,+accumulators} = state
  Equal.trans(Tree.Tree<F32>,buffers_bias(GemmLaws.after_write(input,weights,KernelOutputLaws.write(context,accumulators,bias,output))),
    owner_bias(KernelOutputLaws.write(context,accumulators,bias,output)),bias,
    written_bias(input,weights,KernelOutputLaws.write(context,accumulators,bias,output)),
    observed_bias(KernelOutputLaws.write(context,accumulators,bias,output),bias,context_writes(context,accumulators,bias),Writes.evaluate(F32,output),
      write_observed(context,accumulators,bias,output)))

def reduced_output(+context: Gemm.Context,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+state: ReductionModel.State()) ->
  {Writes.evaluate(F32,buffers_program(GemmLaws.finish(bias,output,context,state))) ==
    Writes.pure(F32,context_writes(context,
      Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,state),bias),Writes.evaluate(F32,output)) : Tree.Tree<F32>}:
  Reduction.State{+input,+weights,input_index,weight_index,base,+accumulators} = state
  Equal.trans(Tree.Tree<F32>,Writes.evaluate(F32,buffers_program(GemmLaws.after_write(input,weights,KernelOutputLaws.write(context,accumulators,bias,output)))),
    owner_output(KernelOutputLaws.write(context,accumulators,bias,output)),
    Writes.pure(F32,context_writes(context,accumulators,bias),Writes.evaluate(F32,output)),
    written_output(input,weights,KernelOutputLaws.write(context,accumulators,bias,output)),
    observed_output(KernelOutputLaws.write(context,accumulators,bias,output),bias,context_writes(context,accumulators,bias),Writes.evaluate(F32,output),
      write_observed(context,accumulators,bias,output)))

# One job leaves every source owner unchanged and appends exactly its writes.
def executed(+job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+program: Writes.Program<F32>) ->
  {GemmLaws.execute(job,GemmLaws.Buffers{input,weights,bias,program}) ==
    GemmLaws.Buffers{input,weights,bias,buffers_program(GemmLaws.execute(job,GemmLaws.Buffers{input,weights,bias,program}))} : GemmLaws.Buffers}:
  Gemm.Job{+context,+upper,+count} = job
  +state = GemmLaws.reduce(upper,count,input,weights,context)
  +result = GemmLaws.finish(bias,program,context,state)
  %Equal.trans(Tree.Tree<Tile.Tile8>,buffers_input(result),state_input(state),input,
    reduced_input(context,bias,program,state),reduce_input(upper,count,input,weights,context)) :
    {result == GemmLaws.Buffers{_,weights,bias,buffers_program(result)} : GemmLaws.Buffers}
  %Equal.trans(Tree.Tree<Tile.Tile8>,buffers_weights(result),state_weights(state),weights,
    reduced_weights(context,bias,program,state),reduce_weights(upper,count,input,weights,context)) :
    {result == GemmLaws.Buffers{buffers_input(result),_,bias,buffers_program(result)} : GemmLaws.Buffers}
  %reduced_bias(context,bias,program,state) :
    {result == GemmLaws.Buffers{buffers_input(result),buffers_weights(result),_,buffers_program(result)} : GemmLaws.Buffers}
  buffers_eta(result)

def executed_output(+job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+program: Writes.Program<F32>) ->
  {Writes.evaluate(F32,buffers_program(GemmLaws.execute(job,GemmLaws.Buffers{input,weights,bias,program}))) ==
    Writes.pure(F32,job_writes(job,input,weights,bias),Writes.evaluate(F32,program)) : Tree.Tree<F32>}:
  Gemm.Job{+context,+upper,+count} = job
  reduced_output(context,bias,program,GemmLaws.reduce(upper,count,input,weights,context))

def all_writes(count: Nat,+first: U32,+layout: Gemm.Layout,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  List<&2,Writes.Assignment<F32>>:
  match count:
    case 0n: []
    case 1n+rest:
      List.append(&2,Writes.Assignment<F32>,job_writes(Gemm.job(layout,first),input,weights,bias),
        all_writes(rest,(first + 1 : U32),layout,input,weights,bias))

def fold_output(+count: Nat,+first: U32,+layout: Gemm.Layout,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,
  +bias: Tree.Tree<F32>,+program: Writes.Program<F32>) ->
  {GemmLaws.result(Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step,count,first,layout,GemmLaws.Buffers{input,weights,bias,program})) ==
    Writes.pure(F32,all_writes(count,first,layout,input,weights,bias),Writes.evaluate(F32,program)) : Tree.Tree<F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +job = Gemm.job(layout,first)
      +next = buffers_program(GemmLaws.execute(job,GemmLaws.Buffers{input,weights,bias,program}))
      +here = job_writes(job,input,weights,bias)
      +later = all_writes(rest,(first + 1 : U32),layout,input,weights,bias)
      %Equal.sym(GemmLaws.Buffers,GemmLaws.execute(job,GemmLaws.Buffers{input,weights,bias,program}),GemmLaws.Buffers{input,weights,bias,next},
        executed(job,input,weights,bias,program)) :
        {GemmLaws.result(Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step,rest,(first + 1 : U32),layout,_)) ==
          Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,here,later),Writes.evaluate(F32,program)) : Tree.Tree<F32>}
      %Equal.sym(Tree.Tree<F32>,Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,here,later),Writes.evaluate(F32,program)),
        Writes.pure(F32,later,Writes.pure(F32,here,Writes.evaluate(F32,program))),Writes.concatenated_writes(F32,here,later,Writes.evaluate(F32,program))) :
        {GemmLaws.result(Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step,rest,(first + 1 : U32),layout,GemmLaws.Buffers{input,weights,bias,next})) == _ : Tree.Tree<F32>}
      %executed_output(job,input,weights,bias,program) :
        {GemmLaws.result(Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step,rest,(first + 1 : U32),layout,GemmLaws.Buffers{input,weights,bias,next})) ==
          Writes.pure(F32,later,_) : Tree.Tree<F32>}
      fold_output(rest,(first + 1 : U32),layout,input,weights,bias,next)

def job_count(+rows: U32,+columns: U32) -> Nat:
  U32.to_nat(Gemm.job_count(rows,columns))

def run_writes(+rows: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,
  +output: Tree.Tree<F32>,+columns: U32,+inner: U32) ->
  {GemmLaws.run(rows,input,weights,bias,output,columns,inner) ==
    Writes.pure(F32,all_writes(job_count(rows,columns),0,Gemm.Layout{rows,columns,inner},input,weights,bias),output) : Tree.Tree<F32>}:
  fold_output(job_count(rows,columns),0,Gemm.Layout{rows,columns,inner},input,weights,bias,Writes.Initial{output})
