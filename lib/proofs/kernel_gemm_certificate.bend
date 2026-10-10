# Shape evidence follows the real readers and writers. Only the final flat
# certificate is live; witness traversals occur beneath erased rewrites.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_witness.bend as Witness
import ./storage_certificate.bend as Certificate
import ./storage_tree_model.bend as Tree
import ./loop_projection.bend as Projection
import ./kernel_read_witness.bend as ReadWitness

def store_lane(+depth: Nat,inside: Bool,-output: Array<F32>,+offset: U32,-value: F32,evidence: Witness.Witness(F32,depth,output)) ->
  Witness.Witness(F32,depth,Gemm.store_lane(inside,output,offset,value)):
  match inside:
    case True{}: Witness.write(F32,depth,output,offset,value,evidence)
    case False{}: evidence

def store_tail(+count: U32,+depth: Nat,-output: Array<F32>,+offset: U32,-bias: F32,-accumulator: Tile.Tile8,
  evidence: Witness.Witness(F32,depth,output)) ->
  Witness.Witness(F32,depth,Gemm.store_tail(count,output,offset,bias,accumulator)):
  -a = (Tile.lane0(accumulator) + bias : F32)
  -b = (Tile.lane1(accumulator) + bias : F32)
  -c = (Tile.lane2(accumulator) + bias : F32)
  -d = (Tile.lane3(accumulator) + bias : F32)
  -e = (Tile.lane4(accumulator) + bias : F32)
  -f = (Tile.lane5(accumulator) + bias : F32)
  -g = (Tile.lane6(accumulator) + bias : F32)
  -h = (Tile.lane7(accumulator) + bias : F32)
  -o1 = Gemm.store_lane(U32.is_lt(0,count),output,offset,a)
  -o2 = Gemm.store_lane(U32.is_lt(1,count),o1,(offset + 1 : U32),b)
  -o3 = Gemm.store_lane(U32.is_lt(2,count),o2,(offset + 2 : U32),c)
  -o4 = Gemm.store_lane(U32.is_lt(3,count),o3,(offset + 3 : U32),d)
  -o5 = Gemm.store_lane(U32.is_lt(4,count),o4,(offset + 4 : U32),e)
  -o6 = Gemm.store_lane(U32.is_lt(5,count),o5,(offset + 5 : U32),f)
  -o7 = Gemm.store_lane(U32.is_lt(6,count),o6,(offset + 6 : U32),g)
  evidence = store_lane(depth,U32.is_lt(0,count),output,offset,a,evidence)
  evidence = store_lane(depth,U32.is_lt(1,count),o1,(offset + 1 : U32),b,evidence)
  evidence = store_lane(depth,U32.is_lt(2,count),o2,(offset + 2 : U32),c,evidence)
  evidence = store_lane(depth,U32.is_lt(3,count),o3,(offset + 3 : U32),d,evidence)
  evidence = store_lane(depth,U32.is_lt(4,count),o4,(offset + 4 : U32),e,evidence)
  evidence = store_lane(depth,U32.is_lt(5,count),o5,(offset + 5 : U32),f,evidence)
  evidence = store_lane(depth,U32.is_lt(6,count),o6,(offset + 6 : U32),g,evidence)
  store_lane(depth,U32.is_lt(7,count),o7,(offset + 7 : U32),h,evidence)

type Written<-bias: Array<F32>,-depth: Nat,-result: Array<F32> & Array<F32>> is Data:
  Written{-output: Array<F32>,equation: {result == (bias,output) : Array<F32> & Array<F32>},balanced: Witness.Witness(F32,depth,output)}

def write_bias(+depth: Nat,-bias: Array<F32>,-output: Array<F32>,+offset: U32,+row: U32,-accumulator: Tile.Tile8,+count: U32,
  read: Witness.Read<F32,bias,Array.get(F32,bias,row)>,evidence: Witness.Witness(F32,depth,output)) ->
  Written<bias,depth,Gemm.write_bias(output,offset,accumulator,count,Array.get(F32,bias,row))>:
  Witness.Read{value,equation} = read
  Written{Gemm.store_tail(count,output,offset,value,accumulator),
    %Equal.sym(Array<F32> & F32,Array.get(F32,bias,row),(bias,value),equation) :
      {Gemm.write_bias(output,offset,accumulator,count,_) == (bias,Gemm.store_tail(count,output,offset,value,accumulator)) : Array<F32> & Array<F32>}
    {==},store_tail(count,depth,output,offset,value,accumulator,evidence)}

def write_row(+depth: Nat,valid: Bool,+row: U32,+position: U32,+columns: U32,-accumulator: Tile.Tile8,
  -bias: Array<F32>,-output: Array<F32>,bias_evidence: Witness.ArrayWitness<F32,bias>,evidence: Witness.Witness(F32,depth,output)) ->
  Written<bias,depth,Gemm.write_row(valid,row,position,columns,accumulator,(bias,output))>:
  match valid:
    case False{}: Written{output,{==},evidence}
    case True{}:
      write_bias(depth,bias,output,(row * columns + position : U32),row,accumulator,U32.min(8,(columns - position : U32)),Witness.observed(F32,bias,row,bias_evidence),evidence)

def next(+depth: Nat,valid: Bool,row: U32,position: U32,columns: U32,-accumulator: Tile.Tile8,
  -bias: Array<F32>,-owners: Array<F32> & Array<F32>,bias_evidence: Witness.ArrayWitness<F32,bias>,evidence: Written<bias,depth,owners>) ->
  Written<bias,depth,Gemm.write_row(valid,row,position,columns,accumulator,owners)>:
  Written{output,equation,balanced} = evidence
  %Equal.sym(Array<F32> & Array<F32>,owners,(bias,output),equation) :
    Written<bias,depth,Gemm.write_row(valid,row,position,columns,accumulator,_)>
  write_row(depth,valid,row,position,columns,accumulator,bias,output,bias_evidence,balanced)

def write(+depth: Nat,context: Gemm.Context,-accumulators: Rows.Rows<Tile.Tile8>,
  -bias: Array<F32>,-output: Array<F32>,+bias_evidence: Witness.ArrayWitness<F32,bias>,balanced: Witness.Witness(F32,depth,output)) ->
  Written<bias,depth,Gemm.write(context,accumulators,bias,output)>:
  Gemm.Context{+row,+position,+rows,+columns,input_base,weight_base} = context
  -a = Rows.first(~Tile.Tile8,accumulators)
  -b = Rows.second(~Tile.Tile8,accumulators)
  -c = Rows.third(~Tile.Tile8,accumulators)
  -d = Rows.fourth(~Tile.Tile8,accumulators)
  evidence = write_row(depth,U32.is_lt(row,rows),row,position,columns,a,bias,output,bias_evidence,balanced)
  -owners = Gemm.write_row(U32.is_lt(row,rows),row,position,columns,a,(bias,output))
  evidence = next(depth,U32.is_lt((row + 1 : U32),rows),(row + 1 : U32),position,columns,b,bias,owners,bias_evidence,evidence)
  -owners = Gemm.write_row(U32.is_lt((row + 1 : U32),rows),(row + 1 : U32),position,columns,b,owners)
  evidence = next(depth,U32.is_lt((row + 2 : U32),rows),(row + 2 : U32),position,columns,c,bias,owners,bias_evidence,evidence)
  -owners = Gemm.write_row(U32.is_lt((row + 2 : U32),rows),(row + 2 : U32),position,columns,c,owners)
  next(depth,U32.is_lt((row + 3 : U32),rows),(row + 3 : U32),position,columns,d,bias,owners,bias_evidence,evidence)

type Evidence<-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-depth: Nat,-state: Gemm.Buffers> is Data:
  Evidence{-output: Array<F32>,equation: {state == Gemm.Buffers{input,weights,bias,output} : Gemm.Buffers},balanced: Witness.Witness(F32,depth,output)}

def after_write(-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-depth: Nat,-written: Array<F32> & Array<F32>,
  evidence: Written<bias,depth,written>) -> Evidence<input,weights,bias,depth,Gemm.after_write(input,weights,written)>:
  Written{output,equation,balanced} = evidence
  Evidence{output,
    %Equal.sym(Array<F32> & Array<F32>,written,(bias,output),equation) :
      {Gemm.after_write(input,weights,_) == Gemm.Buffers{input,weights,bias,output} : Gemm.Buffers}
    {==},balanced}

# The final step stores into the owners the loop returned unchanged.
def finished(+depth: Nat,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,-state: Gemm.Loop,
  +bias_evidence: Witness.ArrayWitness<F32,bias>,balanced: Witness.Witness(F32,depth,output),
  evidence: ReadWitness.Looped<input,weights,bias,output,state>) -> Evidence<input,weights,bias,depth,Gemm.finish(state)>:
  ReadWitness.Looped{index,context,accumulators,equation} = evidence
  %Equal.sym(Gemm.Loop,state,Gemm.Loop{input,weights,bias,output,index,context,accumulators},equation) :
    Evidence<input,weights,bias,depth,Gemm.finish(_)>
  after_write(input,weights,bias,depth,Gemm.write(context,accumulators,bias,output),
    write(depth,context,accumulators,bias,output,bias_evidence,balanced))

def reduced(~upper: Bool,+depth: Nat,+count: Nat,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,-state: Gemm.Loop,
  +input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,+weight_evidence: Witness.ArrayWitness<Tile.Tile8,weights>,
  +bias_evidence: Witness.ArrayWitness<F32,bias>,balanced: Witness.Witness(F32,depth,output),evidence: ReadWitness.Looped<input,weights,bias,output,state>) ->
  Evidence<input,weights,bias,depth,Loop.run_then(~Gemm.Loop,~Gemm.Buffers,~Gemm.step(~upper),~Gemm.finish,count,state)>:
  %Equal.sym(Gemm.Buffers,Loop.run_then(~Gemm.Loop,~Gemm.Buffers,~Gemm.step(~upper),~Gemm.finish,count,state),
    Gemm.finish(Loop.run(~Gemm.Loop,~Gemm.step(~upper),count,state)),
    Projection.run_then_finishes(~Gemm.Loop,~Gemm.Buffers,~Gemm.step(~upper),~Gemm.finish,count,state)) :
    Evidence<input,weights,bias,depth,_>
  finished(depth,input,weights,bias,output,Loop.run(~Gemm.Loop,~Gemm.step(~upper),count,state),bias_evidence,balanced,
    ReadWitness.run(~upper,count,input,weights,bias,output,state,input_evidence,weight_evidence,evidence))

def reduce(upper: Bool,+depth: Nat,+count: Nat,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,+context: Gemm.Context,
  +input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,+weight_evidence: Witness.ArrayWitness<Tile.Tile8,weights>,
  +bias_evidence: Witness.ArrayWitness<F32,bias>,balanced: Witness.Witness(F32,depth,output)) ->
  Evidence<input,weights,bias,depth,Gemm.reduce(upper,count,Gemm.Loop{input,weights,bias,output,0,context,Rows.filled(~Tile.Tile8,Tile.filled(0.0))})>:
  match upper:
    case False{}:
      reduced(~False{},depth,count,input,weights,bias,output,Gemm.Loop{input,weights,bias,output,0,context,Rows.filled(~Tile.Tile8,Tile.filled(0.0))},
        input_evidence,weight_evidence,bias_evidence,balanced,ReadWitness.Looped{0,context,Rows.filled(~Tile.Tile8,Tile.filled(0.0)),{==}})
    case True{}:
      reduced(~True{},depth,count,input,weights,bias,output,Gemm.Loop{input,weights,bias,output,0,context,Rows.filled(~Tile.Tile8,Tile.filled(0.0))},
        input_evidence,weight_evidence,bias_evidence,balanced,ReadWitness.Looped{0,context,Rows.filled(~Tile.Tile8,Tile.filled(0.0)),{==}})

def execute(+depth: Nat,job: Gemm.Job,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-state: Gemm.Buffers,
  +input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,+weight_evidence: Witness.ArrayWitness<Tile.Tile8,weights>,+bias_evidence: Witness.ArrayWitness<F32,bias>,
  evidence: Evidence<input,weights,bias,depth,state>) -> Evidence<input,weights,bias,depth,Gemm.execute(job,state)>:
  Gemm.Job{+context,+upper,+count} = job
  Evidence{output,equation,balanced} = evidence
  %Equal.sym(Gemm.Buffers,state,Gemm.Buffers{input,weights,bias,output},equation) :
    Evidence<input,weights,bias,depth,Gemm.execute(Gemm.Job{context,upper,count},_)>
  reduce(upper,depth,count,input,weights,bias,output,context,input_evidence,weight_evidence,bias_evidence,balanced)

def traverse(+depth: Nat,count: Nat,+index: U32,+layout: Gemm.Layout,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-state: Gemm.Buffers,
  +input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,+weight_evidence: Witness.ArrayWitness<Tile.Tile8,weights>,+bias_evidence: Witness.ArrayWitness<F32,bias>,
  evidence: Evidence<input,weights,bias,depth,state>) ->
  Evidence<input,weights,bias,depth,Traversal.indexed_fold(~Gemm.Buffers,~Gemm.Layout,~Gemm.step_job,count,index,layout,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      traverse(depth,rest,(index + 1 : U32),layout,input,weights,bias,Gemm.step_job(index,layout,state),input_evidence,weight_evidence,bias_evidence,
        execute(depth,Gemm.job(layout,index),input,weights,bias,state,input_evidence,weight_evidence,bias_evidence,evidence))

def finish(-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-depth: Nat,-state: Gemm.Buffers,
  evidence: Evidence<input,weights,bias,depth,state>) -> Witness.Witness(F32,depth,Gemm.result(state)):
  Evidence{output,equation,balanced} = evidence
  %Equal.sym(Gemm.Buffers,state,Gemm.Buffers{input,weights,bias,output},equation) : Witness.Witness(F32,depth,Gemm.result(_))
  balanced

def run(+depth: Nat,+rows: U32,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,+columns: U32,+inner: U32,+start: U32,+output_columns: U32,
  input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,weight_evidence: Witness.ArrayWitness<Tile.Tile8,weights>,bias_evidence: Witness.ArrayWitness<F32,bias>,
  balanced: Witness.Witness(F32,depth,output)) -> Witness.Witness(F32,depth,Gemm.run_at(rows,input,weights,bias,output,columns,inner,start,output_columns)):
  +count = U32.to_nat(Gemm.job_count(rows,columns))
  +layout = {Gemm.Layout{rows,columns,inner,start,output_columns} : Gemm.Layout}
  -state = {Gemm.Buffers{input,weights,bias,output} : Gemm.Buffers}
  finish(input,weights,bias,depth,Traversal.indexed_fold(~Gemm.Buffers,~Gemm.Layout,~Gemm.step_job,count,0,layout,state),
    traverse(depth,count,0,layout,input,weights,bias,state,input_evidence,weight_evidence,bias_evidence,Evidence{output,{==},balanced}))

def equation(+depth: Nat,+rows: U32,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,+columns: U32,+inner: U32,+start: U32,+output_columns: U32,
  input_certificate: Certificate.Certificate<Tile.Tile8,input>,weight_certificate: Certificate.Certificate<Tile.Tile8,weights>,bias_certificate: Certificate.Certificate<F32,bias>,
  balanced: {output == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,output))) : Array<F32>}) ->
  {Gemm.run_at(rows,input,weights,bias,output,columns,inner,start,output_columns) ==
    Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,Gemm.run_at(rows,input,weights,bias,output,columns,inner,start,output_columns)))) : Array<F32>}:
  %Certificate.restored(F32,depth,Gemm.run_at(rows,input,weights,bias,output,columns,inner,start,output_columns),
    run(depth,rows,input,weights,bias,output,columns,inner,start,output_columns,Certificate.witness(Tile.Tile8,input,input_certificate),
      Certificate.witness(Tile.Tile8,weights,weight_certificate),Certificate.witness(F32,bias,bias_certificate),Certificate.witness_at(F32,depth,output,balanced))) :
    {_ == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,Gemm.run_at(rows,input,weights,bias,output,columns,inner,start,output_columns)))) : Array<F32>}
  {==}

def certificate(rows: U32,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,columns: U32,inner: U32,start: U32,output_columns: U32,
  input_certificate: Certificate.Certificate<Tile.Tile8,input>,weight_certificate: Certificate.Certificate<Tile.Tile8,weights>,bias_certificate: Certificate.Certificate<F32,bias>,
  output_certificate: Certificate.Certificate<F32,output>) -> Certificate.Certificate<F32,Gemm.run_at(rows,input,weights,bias,output,columns,inner,start,output_columns)>:
  Certificate.Certificate{+depth,balanced} = output_certificate
  Certificate.Certificate{depth,equation(depth,rows,input,weights,bias,output,columns,inner,start,output_columns,input_certificate,weight_certificate,bias_certificate,balanced)}

# The fold gives back the three buffers it only reads, next to the output of run.
def kept_state(-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-depth: Nat,-state: Gemm.Buffers,
  evidence: Evidence<input,weights,bias,depth,state>) -> {state == Gemm.Buffers{input,weights,bias,Gemm.result(state)} : Gemm.Buffers}:
  Evidence{output,equation,balanced} = evidence
  %Equal.sym(Gemm.Buffers,state,Gemm.Buffers{input,weights,bias,output},equation) :
    {_ == Gemm.Buffers{input,weights,bias,Gemm.result(_)} : Gemm.Buffers}
  {==}

def kept_equation(+depth: Nat,+rows: U32,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,+columns: U32,+inner: U32,+start: U32,+output_columns: U32,
  input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,weight_evidence: Witness.ArrayWitness<Tile.Tile8,weights>,bias_evidence: Witness.ArrayWitness<F32,bias>,
  balanced: Witness.Witness(F32,depth,output)) ->
  {Gemm.run_keeping(rows,input,weights,bias,output,columns,inner,start,output_columns) ==
    Gemm.Buffers{input,weights,bias,Gemm.run_at(rows,input,weights,bias,output,columns,inner,start,output_columns)} : Gemm.Buffers}:
  +count = U32.to_nat(Gemm.job_count(rows,columns))
  +layout = {Gemm.Layout{rows,columns,inner,start,output_columns} : Gemm.Layout}
  -state = {Gemm.Buffers{input,weights,bias,output} : Gemm.Buffers}
  kept_state(input,weights,bias,depth,Traversal.indexed_fold(~Gemm.Buffers,~Gemm.Layout,~Gemm.step_job,count,0,layout,state),
    traverse(depth,count,0,layout,input,weights,bias,state,input_evidence,weight_evidence,bias_evidence,Evidence{output,{==},balanced}))

type Kept<-state: Gemm.Buffers> is Data:
  Kept{input: Certificate.Certificate<Tile.Tile8,Gemm.kept_input(state)>,weights: Certificate.Certificate<Tile.Tile8,Gemm.kept_weights(state)>,
    bias: Certificate.Certificate<F32,Gemm.kept_bias(state)>,output: Certificate.Certificate<F32,Gemm.result(state)>}

# The certificates of the read-only buffers are the ones that came in.
def kept(+rows: U32,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,+columns: U32,+inner: U32,+start: U32,+output_columns: U32,
  input_certificate: Certificate.Certificate<Tile.Tile8,input>,weight_certificate: Certificate.Certificate<Tile.Tile8,weights>,bias_certificate: Certificate.Certificate<F32,bias>,
  output_certificate: Certificate.Certificate<F32,output>) -> Kept<Gemm.run_keeping(rows,input,weights,bias,output,columns,inner,start,output_columns)>:
  Certificate.Certificate{+input_depth,+input_equation} = input_certificate
  Certificate.Certificate{+weight_depth,+weight_equation} = weight_certificate
  Certificate.Certificate{+bias_depth,+bias_equation} = bias_certificate
  Certificate.Certificate{+depth,+balanced} = output_certificate
  -state = Gemm.run_keeping(rows,input,weights,bias,output,columns,inner,start,output_columns)
  -result = Gemm.run_at(rows,input,weights,bias,output,columns,inner,start,output_columns)
  +owners = kept_equation(depth,rows,input,weights,bias,output,columns,inner,start,output_columns,
    Certificate.witness(Tile.Tile8,input,Certificate.Certificate{input_depth,input_equation}),Certificate.witness(Tile.Tile8,weights,Certificate.Certificate{weight_depth,weight_equation}),
    Certificate.witness(F32,bias,Certificate.Certificate{bias_depth,bias_equation}),Certificate.witness_at(F32,depth,output,balanced))
  Kept{
    Certificate.Certificate{input_depth,
      %Equal.sym(Gemm.Buffers,state,Gemm.Buffers{input,weights,bias,result},owners) :
        {Gemm.kept_input(_) == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,input_depth,Tree.reflect(Tile.Tile8,Gemm.kept_input(_)))) : Array<Tile.Tile8>}
      input_equation},
    Certificate.Certificate{weight_depth,
      %Equal.sym(Gemm.Buffers,state,Gemm.Buffers{input,weights,bias,result},owners) :
        {Gemm.kept_weights(_) == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,weight_depth,Tree.reflect(Tile.Tile8,Gemm.kept_weights(_)))) : Array<Tile.Tile8>}
      weight_equation},
    Certificate.Certificate{bias_depth,
      %Equal.sym(Gemm.Buffers,state,Gemm.Buffers{input,weights,bias,result},owners) :
        {Gemm.kept_bias(_) == Tree.pack(F32,Certificate.normalized(F32,bias_depth,Tree.reflect(F32,Gemm.kept_bias(_)))) : Array<F32>}
      bias_equation},
    Certificate.Certificate{depth,equation(depth,rows,input,weights,bias,output,columns,inner,start,output_columns,Certificate.Certificate{input_depth,input_equation},
      Certificate.Certificate{weight_depth,weight_equation},Certificate.Certificate{bias_depth,bias_equation},balanced)}}
