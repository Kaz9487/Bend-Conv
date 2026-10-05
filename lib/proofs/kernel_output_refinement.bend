import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../kernel_rows.bend as Rows
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./write_program.bend as Writes

type Owners is Data:
  Owners{bias: Tree.Tree<F32>,output: Writes.Program<F32>}

def represent(owners: Owners) -> Array<F32> & Array<F32>:
  Owners{bias,output} = owners
  (Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)))

def assignments(+offset: U32,+bias: F32,+accumulator: Tile.Tile8) -> List<&2,Writes.Assignment<F32>>:
  [Writes.Put{offset,(Tile.lane0(accumulator) + bias : F32)},
   Writes.Put{(offset + 1 : U32),(Tile.lane1(accumulator) + bias : F32)},
   Writes.Put{(offset + 2 : U32),(Tile.lane2(accumulator) + bias : F32)},
   Writes.Put{(offset + 3 : U32),(Tile.lane3(accumulator) + bias : F32)},
   Writes.Put{(offset + 4 : U32),(Tile.lane4(accumulator) + bias : F32)},
   Writes.Put{(offset + 5 : U32),(Tile.lane5(accumulator) + bias : F32)},
   Writes.Put{(offset + 6 : U32),(Tile.lane6(accumulator) + bias : F32)},
   Writes.Put{(offset + 7 : U32),(Tile.lane7(accumulator) + bias : F32)}]

# Keep stores as syntax: readback proofs can reason about one row without
# repeatedly normalizing the capacity of all thirty-two nested writes.
def store(output: Writes.Program<F32>,offset: U32,bias: F32,accumulator: Tile.Tile8) -> Writes.Program<F32>:
  Writes.Append{assignments(offset,bias,accumulator),output}

def stored(+output: Writes.Program<F32>,+offset: U32,+bias: F32,+accumulator: Tile.Tile8) ->
  {Gemm.store(Tree.pack(F32,Writes.evaluate(F32,output)),offset,bias,accumulator) == Tree.pack(F32,Writes.evaluate(F32,store(output,offset,bias,accumulator))) : Array<F32>}:
  Writes.refinement(F32,assignments(offset,bias,accumulator),Writes.evaluate(F32,output))

def row(valid: Bool,+index: U32,position: U32,columns: U32,accumulator: Tile.Tile8,owners: Owners) -> Owners:
  match valid owners:
    case False{} owners: owners
    case True{} Owners{+bias,output}:
      Owners{bias,store(output,(index * columns + position : U32),Tree.read(F32,bias,index),accumulator)}

def row_matches(+valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+owners: Owners) ->
  {Gemm.write_row(valid,index,position,columns,accumulator,represent(owners)) ==
    represent(row(valid,index,position,columns,accumulator,owners)) : Array<F32> & Array<F32>}:
  match valid owners:
    case False{} owners: {==}
    case True{} Owners{+bias,+output}:
      %Equal.sym(Array<F32> & F32,Array.get(F32,Tree.pack(F32,bias),index),(Tree.pack(F32,bias),Tree.read(F32,bias,index)),StorageLaws.read(F32,bias,index)) :
        {Gemm.write_bias(Tree.pack(F32,Writes.evaluate(F32,output)),(index * columns + position : U32),accumulator,_) ==
          (Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,store(output,(index * columns + position : U32),Tree.read(F32,bias,index),accumulator)))) : Array<F32> & Array<F32>}
      Equal.cong(Array<F32>,Array<F32> & Array<F32>,result => (Tree.pack(F32,bias),result),
        Gemm.store(Tree.pack(F32,Writes.evaluate(F32,output)),(index * columns + position : U32),Tree.read(F32,bias,index),accumulator),
        Tree.pack(F32,Writes.evaluate(F32,store(output,(index * columns + position : U32),Tree.read(F32,bias,index),accumulator))),
        stored(output,(index * columns + position : U32),Tree.read(F32,bias,index),accumulator))

def next(+valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,-native: Array<F32> & Array<F32>,+owners: Owners,
  equation: {native == represent(owners) : Array<F32> & Array<F32>}) ->
  {Gemm.write_row(valid,index,position,columns,accumulator,native) == represent(row(valid,index,position,columns,accumulator,owners)) : Array<F32> & Array<F32>}:
  %Equal.sym(Array<F32> & Array<F32>,native,represent(owners),equation) :
    {Gemm.write_row(valid,index,position,columns,accumulator,_) == represent(row(valid,index,position,columns,accumulator,owners)) : Array<F32> & Array<F32>}
  row_matches(valid,index,position,columns,accumulator,owners)

def write(context: Gemm.Context,+accumulators: Rows.Rows<Tile.Tile8>,bias: Tree.Tree<F32>,output: Writes.Program<F32>) -> Owners:
  Gemm.Context{+index,+position,+rows,+columns,input_base,weight_base} = context
  owners = row(U32.is_lt(index,rows),index,position,columns,Rows.first(~Tile.Tile8,accumulators),Owners{bias,output})
  owners = row(U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,Rows.second(~Tile.Tile8,accumulators),owners)
  owners = row(U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,Rows.third(~Tile.Tile8,accumulators),owners)
  row(U32.is_lt((index + 3 : U32),rows),(index + 3 : U32),position,columns,Rows.fourth(~Tile.Tile8,accumulators),owners)

def write_matches(+context: Gemm.Context,+accumulators: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>) ->
  {Gemm.write(context,accumulators,Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output))) == represent(write(context,accumulators,bias,output)) : Array<F32> & Array<F32>}:
  Gemm.Context{+index,+position,+rows,+columns,input_base,weight_base} = context
  +a = Rows.first(~Tile.Tile8,accumulators)
  +b = Rows.second(~Tile.Tile8,accumulators)
  +c = Rows.third(~Tile.Tile8,accumulators)
  +d = Rows.fourth(~Tile.Tile8,accumulators)
  +initial = {Owners{bias,output} : Owners}
  +first = row(U32.is_lt(index,rows),index,position,columns,a,initial)
  +second = row(U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,b,first)
  +third = row(U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,c,second)
  -one = Gemm.write_row(U32.is_lt(index,rows),index,position,columns,a,represent(initial))
  -two = Gemm.write_row(U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,b,one)
  -three = Gemm.write_row(U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,c,two)
  next(U32.is_lt((index + 3 : U32),rows),(index + 3 : U32),position,columns,d,three,third,
    next(U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,c,two,second,
      next(U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,b,one,first,
        row_matches(U32.is_lt(index,rows),index,position,columns,a,initial))))
