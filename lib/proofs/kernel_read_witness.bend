import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ../traversal_loop.bend as Loop
import ./storage_witness.bend as Witness

# The k loop returns every owner it received. Index, context and the erased
# accumulators are whatever the steps produced.
type Looped<-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,-state: Gemm.Loop> is Data:
  Looped{index: U32,context: Gemm.Context,-accumulators: Rows.Rows<Tile.Tile8>,
    equation: {state == Gemm.Loop{input,weights,bias,output,index,context,accumulators} : Gemm.Loop}}

def after_input(-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,+index: U32,+context: Gemm.Context,
  -accumulators: Rows.Rows<Tile.Tile8>,-samples: Rows.Rows<F32>,+address: U32,read: Witness.Read<Tile.Tile8,input,Array.get(Tile.Tile8,input,address)>) ->
  Looped<input,weights,bias,output,Gemm.with_input(weights,bias,output,index,context,accumulators,samples,Array.get(Tile.Tile8,input,address))>:
  Witness.Read{value,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,input,address),(input,value),equation) :
    Looped<input,weights,bias,output,Gemm.with_input(weights,bias,output,index,context,accumulators,samples,_)>
  Looped{(index + 1 : U32),context,Gemm.update(accumulators,value,samples),{==}}

def after_weights(~upper: Bool,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,+index: U32,+context: Gemm.Context,
  +input_base: U32,-accumulators: Rows.Rows<Tile.Tile8>,+address: U32,input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,
  read: Witness.Read<Tile.Tile8,weights,Array.get(Tile.Tile8,weights,address)>) ->
  Looped<input,weights,bias,output,Gemm.with_weights(~upper,input,bias,output,index,context,input_base,accumulators,Array.get(Tile.Tile8,weights,address))>:
  Witness.Read{value,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,weights,address),(weights,value),equation) :
    Looped<input,weights,bias,output,Gemm.with_weights(~upper,input,bias,output,index,context,input_base,accumulators,_)>
  after_input(input,weights,bias,output,index,context,accumulators,Rows.half(upper,value),(input_base + index : U32),
    Witness.observed(Tile.Tile8,input,(input_base + index : U32),input_evidence))

def step_at(~upper: Bool,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,+index: U32,context: Gemm.Context,
  -accumulators: Rows.Rows<Tile.Tile8>,+input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,+weight_evidence: Witness.ArrayWitness<Tile.Tile8,weights>) ->
  Looped<input,weights,bias,output,Gemm.step(~upper,Gemm.Loop{input,weights,bias,output,index,context,accumulators})>:
  Gemm.Context{+row,+position,+rows,+columns,+input_base,+weight_base} = context
  after_weights(~upper,input,weights,bias,output,index,Gemm.Context{row,position,rows,columns,input_base,weight_base},input_base,accumulators,
    (weight_base + index : U32),input_evidence,Witness.observed(Tile.Tile8,weights,(weight_base + index : U32),weight_evidence))

def step(~upper: Bool,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,-state: Gemm.Loop,
  +input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,+weight_evidence: Witness.ArrayWitness<Tile.Tile8,weights>,
  evidence: Looped<input,weights,bias,output,state>) -> Looped<input,weights,bias,output,Gemm.step(~upper,state)>:
  Looped{+index,+context,accumulators,equation} = evidence
  %Equal.sym(Gemm.Loop,state,Gemm.Loop{input,weights,bias,output,index,context,accumulators},equation) :
    Looped<input,weights,bias,output,Gemm.step(~upper,_)>
  step_at(~upper,input,weights,bias,output,index,context,accumulators,input_evidence,weight_evidence)

def run(~upper: Bool,count: Nat,-input: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<F32>,-state: Gemm.Loop,
  +input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,+weight_evidence: Witness.ArrayWitness<Tile.Tile8,weights>,
  evidence: Looped<input,weights,bias,output,state>) ->
  Looped<input,weights,bias,output,Loop.run(~Gemm.Loop,~Gemm.step(~upper),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      run(~upper,rest,input,weights,bias,output,Gemm.step(~upper,state),input_evidence,weight_evidence,
        step(~upper,input,weights,bias,output,state,input_evidence,weight_evidence,evidence))
