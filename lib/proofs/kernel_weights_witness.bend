import Base
import ../kernel_weights.bend as Weights
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness

# Missing rows are zero-filled without changing the source owner.
def conditional(-input: Array<F32>,valid: Bool,+index: U32,evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,F32,input,Storage.read_when(~F32,0.0,valid,input,index)>:
  match valid:
    case False{}: TraversalWitness.Sample{0.0,{==}}
    case True{}: TraversalWitness.array_sample(~F32,input,Array.get(F32,input,index),Witness.observed(F32,input,index,evidence))

def eighth(-input: Array<F32>,-a: F32,-b: F32,-c: F32,-d: F32,-e: F32,-f: F32,-g: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.eighth(a,b,c,d,e,f,g,result)>:
  TraversalWitness.Sample{h,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,h),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.eighth(a,b,c,d,e,f,g,_)>
  TraversalWitness.Sample{Tile.Tile{a,b,c,d,e,f,g,h},{==}}

def seventh(+rows: U32,+inner: U32,+row: U32,+index: U32,-input: Array<F32>,-a: F32,-b: F32,-c: F32,-d: F32,-e: F32,-f: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.seventh(rows,inner,row,index,a,b,c,d,e,f,result)>:
  TraversalWitness.Sample{g,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,g),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.seventh(rows,inner,row,index,a,b,c,d,e,f,_)>
  eighth(input,a,b,c,d,e,f,g,Storage.read_when(~F32,0.0,U32.is_lt((row + 7 : U32),rows),input,((row + 7) * inner + index : U32)),
    conditional(input,U32.is_lt((row + 7 : U32),rows),((row + 7) * inner + index : U32),evidence))

def sixth(+rows: U32,+inner: U32,+row: U32,+index: U32,-input: Array<F32>,-a: F32,-b: F32,-c: F32,-d: F32,-e: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.sixth(rows,inner,row,index,a,b,c,d,e,result)>:
  TraversalWitness.Sample{f,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,f),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.sixth(rows,inner,row,index,a,b,c,d,e,_)>
  seventh(rows,inner,row,index,input,a,b,c,d,e,f,Storage.read_when(~F32,0.0,U32.is_lt((row + 6 : U32),rows),input,((row + 6) * inner + index : U32)),
    conditional(input,U32.is_lt((row + 6 : U32),rows),((row + 6) * inner + index : U32),evidence),evidence)

def fifth(+rows: U32,+inner: U32,+row: U32,+index: U32,-input: Array<F32>,-a: F32,-b: F32,-c: F32,-d: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.fifth(rows,inner,row,index,a,b,c,d,result)>:
  TraversalWitness.Sample{e,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,e),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.fifth(rows,inner,row,index,a,b,c,d,_)>
  sixth(rows,inner,row,index,input,a,b,c,d,e,Storage.read_when(~F32,0.0,U32.is_lt((row + 5 : U32),rows),input,((row + 5) * inner + index : U32)),
    conditional(input,U32.is_lt((row + 5 : U32),rows),((row + 5) * inner + index : U32),evidence),evidence)

def fourth(+rows: U32,+inner: U32,+row: U32,+index: U32,-input: Array<F32>,-a: F32,-b: F32,-c: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.fourth(rows,inner,row,index,a,b,c,result)>:
  TraversalWitness.Sample{d,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,d),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.fourth(rows,inner,row,index,a,b,c,_)>
  fifth(rows,inner,row,index,input,a,b,c,d,Storage.read_when(~F32,0.0,U32.is_lt((row + 4 : U32),rows),input,((row + 4) * inner + index : U32)),
    conditional(input,U32.is_lt((row + 4 : U32),rows),((row + 4) * inner + index : U32),evidence),evidence)

def third(+rows: U32,+inner: U32,+row: U32,+index: U32,-input: Array<F32>,-a: F32,-b: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.third(rows,inner,row,index,a,b,result)>:
  TraversalWitness.Sample{c,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,c),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.third(rows,inner,row,index,a,b,_)>
  fourth(rows,inner,row,index,input,a,b,c,Storage.read_when(~F32,0.0,U32.is_lt((row + 3 : U32),rows),input,((row + 3) * inner + index : U32)),
    conditional(input,U32.is_lt((row + 3 : U32),rows),((row + 3) * inner + index : U32),evidence),evidence)

def second(+rows: U32,+inner: U32,+row: U32,+index: U32,-input: Array<F32>,-a: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.second(rows,inner,row,index,a,result)>:
  TraversalWitness.Sample{b,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,b),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.second(rows,inner,row,index,a,_)>
  third(rows,inner,row,index,input,a,b,Storage.read_when(~F32,0.0,U32.is_lt((row + 2 : U32),rows),input,((row + 2) * inner + index : U32)),
    conditional(input,U32.is_lt((row + 2 : U32),rows),((row + 2) * inner + index : U32),evidence),evidence)

def first(+rows: U32,+inner: U32,+row: U32,+index: U32,-input: Array<F32>,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.first(rows,inner,row,index,result)>:
  TraversalWitness.Sample{a,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,a),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.first(rows,inner,row,index,_)>
  second(rows,inner,row,index,input,a,Storage.read_when(~F32,0.0,U32.is_lt((row + 1 : U32),rows),input,((row + 1) * inner + index : U32)),
    conditional(input,U32.is_lt((row + 1 : U32),rows),((row + 1) * inner + index : U32),evidence),evidence)

def read(-input: Array<F32>,context: Weights.Context,index: U32,evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.read(input,context,index)>:
  Weights.Context{+rows,+inner,+row,destination} = context
  +index = index
  +evidence = evidence
  first(rows,inner,row,index,input,Storage.read_when(~F32,0.0,U32.is_lt(row,rows),input,(row * inner + index : U32)),
    conditional(input,U32.is_lt(row,rows),(row * inner + index : U32),evidence),evidence)
