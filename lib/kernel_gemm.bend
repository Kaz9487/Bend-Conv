import Base
import ./kernel_rows.bend as Rows
import ./kernel_tile.bend as Tile
import ./traversal_loop.bend as Iteration
import ./traversal_array.bend as Traversal

type Buffers is Type:
  Buffers{input: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,bias: Array<F32>,output: Array<F32>}

# Row-major output, eight contiguous columns; packed weights are [M/8,K,8].
# A job is one half of a weight block, four rows, at eight positions. Only
# valid rows are stored; the last weight block is zero-padded to eight rows.
type Layout is Data:
  Layout{rows: U32,columns: U32,inner: U32}

# input_base and weight_base address the job's first input group and weight
# tile; the loop index advances both.
type Context is Data:
  Context{row: U32,position: U32,rows: U32,columns: U32,input_base: U32,weight_base: U32}

type Job is Data:
  Job{context: Context,upper: Bool,count: Nat}

# One flat loop state. Every owner, the index and the accumulators are fields
# of this record, and each helper that passes the accumulators on carries few
# other values: a nested owner record or a longer helper chain made every k
# step a real call.
type Loop is Type:
  Loop{input: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,bias: Array<F32>,output: Array<F32>,index: U32,context: Context,
    accumulators: Rows.Rows<Tile.Tile8>}

# A job owns four output rows: one half of an eight-row weight block.
def row_units(rows: U32) -> U32:
  ((rows + 3) / 4 : U32)

def column_groups(columns: U32) -> U32:
  (columns / 8 : U32)

def job_count(rows: U32,columns: U32) -> U32:
  (column_groups(columns) * row_units(rows) : U32)

# Input-major: one input group stays in cache while every row unit passes
# over it (the BLIS loop order). Units 2b and 2b+1 share weight block b.
def job(layout: Layout,index: U32) -> Job:
  Layout{+rows,+columns,+inner} = layout
  +index = index
  +units = row_units(rows)
  +unit = (index % units : U32)
  +group = (index / units : U32)
  Job{Context{(unit * 4 : U32),(group * 8 : U32),rows,columns,(group * inner : U32),(unit / 2 * inner : U32)},U32.is_eq((unit % 2 : U32),1),U32.to_nat(inner)}

def update(accumulators: Rows.Rows<Tile.Tile8>,+input: Tile.Tile8,weights: Rows.Rows<F32>) -> Rows.Rows<Tile.Tile8>:
  Rows.map(~Tile.Tile8,~Tile.Tile8,~F32,~Tile.update,accumulators,input,weights)

def with_input(weights: Array<Tile.Tile8>,bias: Array<F32>,output: Array<F32>,+index: U32,context: Context,
  accumulators: Rows.Rows<Tile.Tile8>,samples: Rows.Rows<F32>,read: Array<Tile.Tile8> & Tile.Tile8) -> Loop:
  (input,tile) = read
  Loop{input,weights,bias,output,(index + 1 : U32),context,update(accumulators,tile,samples)}

def with_weights(~upper: Bool,input: Array<Tile.Tile8>,bias: Array<F32>,output: Array<F32>,+index: U32,context: Context,+input_base: U32,
  accumulators: Rows.Rows<Tile.Tile8>,read: Array<Tile.Tile8> & Tile.Tile8) -> Loop:
  (weights,tile) = read
  with_input(weights,bias,output,index,context,accumulators,Rows.half(upper,tile),Array.get(Tile.Tile8,input,(input_base + index : U32)))

# One ordered k step: read the weight tile, then the input tile, then MAC.
def step(~upper: Bool,state: Loop) -> Loop:
  Loop{input,weights,bias,output,+index,context,accumulators} = state
  Context{+row,+position,+rows,+columns,+input_base,+weight_base} = context
  with_weights(~upper,input,bias,output,index,Context{row,position,rows,columns,input_base,weight_base},input_base,accumulators,
    Array.get(Tile.Tile8,weights,(weight_base + index : U32)))

def store(output: Array<F32>,+offset: U32,+bias: F32,+accumulator: Tile.Tile8) -> Array<F32>:
  output = Array.set(F32,output,offset,(Tile.lane0(accumulator) + bias : F32))
  output = Array.set(F32,output,(offset + 1 : U32),(Tile.lane1(accumulator) + bias : F32))
  output = Array.set(F32,output,(offset + 2 : U32),(Tile.lane2(accumulator) + bias : F32))
  output = Array.set(F32,output,(offset + 3 : U32),(Tile.lane3(accumulator) + bias : F32))
  output = Array.set(F32,output,(offset + 4 : U32),(Tile.lane4(accumulator) + bias : F32))
  output = Array.set(F32,output,(offset + 5 : U32),(Tile.lane5(accumulator) + bias : F32))
  output = Array.set(F32,output,(offset + 6 : U32),(Tile.lane6(accumulator) + bias : F32))
  Array.set(F32,output,(offset + 7 : U32),(Tile.lane7(accumulator) + bias : F32))

def write_bias(output: Array<F32>,offset: U32,accumulator: Tile.Tile8,read: Array<F32> & F32) -> Array<F32> & Array<F32>:
  (bias,value) = read
  (bias,store(output,offset,value,accumulator))

def write_row(valid: Bool,+row: U32,position: U32,columns: U32,accumulator: Tile.Tile8,
  owners: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match valid:
    case False{}: owners
    case True{}:
      (bias,output) = owners
      write_bias(output,(row * columns + position : U32),accumulator,Array.get(F32,bias,row))

def write(context: Context,+accumulators: Rows.Rows<Tile.Tile8>,bias: Array<F32>,output: Array<F32>) -> Array<F32> & Array<F32>:
  Context{+row,+position,+rows,+columns,input_base,weight_base} = context
  owners = write_row(U32.is_lt(row,rows),row,position,columns,Rows.first(~Tile.Tile8,accumulators),(bias,output))
  owners = write_row(U32.is_lt((row + 1 : U32),rows),(row + 1 : U32),position,columns,Rows.second(~Tile.Tile8,accumulators),owners)
  owners = write_row(U32.is_lt((row + 2 : U32),rows),(row + 2 : U32),position,columns,Rows.third(~Tile.Tile8,accumulators),owners)
  write_row(U32.is_lt((row + 3 : U32),rows),(row + 3 : U32),position,columns,Rows.fourth(~Tile.Tile8,accumulators),owners)

def after_write(input: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,written: Array<F32> & Array<F32>) -> Buffers:
  (bias,output) = written
  Buffers{input,weights,bias,output}

# The loop's final step stores its accumulators; bias is added once.
def finish(state: Loop) -> Buffers:
  Loop{input,weights,bias,output,index,+context,accumulators} = state
  after_write(input,weights,write(context,accumulators,bias,output))

def reduce(upper: Bool,count: Nat,state: Loop) -> Buffers:
  match upper:
    case False{}: Iteration.run_then(~Loop,~Buffers,~step(~False{}),~finish,count,state)
    case True{}: Iteration.run_then(~Loop,~Buffers,~step(~True{}),~finish,count,state)

# Every output folds k in order from +0, then adds its bias once.
def execute(job: Job,buffers: Buffers) -> Buffers:
  Job{context,upper,count} = job
  Buffers{input,weights,bias,output} = buffers
  reduce(upper,count,Loop{input,weights,bias,output,0,context,Rows.filled(~Tile.Tile8,Tile.filled(0.0))})

def step_job(index: U32,layout: Layout,buffers: Buffers) -> Buffers:
  execute(job(layout,index),buffers)

def result(buffers: Buffers) -> Array<F32>:
  Buffers{input,weights,bias,output} = buffers
  output

def run(+rows: U32,input: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,bias: Array<F32>,output: Array<F32>,+columns: U32,inner: U32) -> Array<F32>:
  result(
    Traversal.indexed_fold(~Buffers,~Layout,~step_job,U32.to_nat(job_count(rows,columns)),0,Layout{rows,columns,inner},Buffers{input,weights,bias,output}))
