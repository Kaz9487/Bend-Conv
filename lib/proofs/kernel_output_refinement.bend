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

def lane_writes(inside: Bool,offset: U32,value: F32) -> List<&2,Writes.Assignment<F32>>:
  match inside:
    case False{}: []
    case True{}: [Writes.Put{offset,value}]

def append_lane(+inside: Bool,previous: List<&2,Writes.Assignment<F32>>,offset: U32,value: F32) -> List<&2,Writes.Assignment<F32>>:
  List.append(&2,Writes.Assignment<F32>,previous,lane_writes(inside,offset,value))

def native_lane(+inside: Bool,-output: Array<F32>,+offset: U32,+value: F32) ->
  {Gemm.store_lane(inside,output,offset,value) == Writes.native(F32,lane_writes(inside,offset,value),output) : Array<F32>}:
  match inside:
    case False{}: {==}
    case True{}: {==}

def next_lane(+inside: Bool,-native: Array<F32>,+previous: List<&2,Writes.Assignment<F32>>,-output: Array<F32>,+offset: U32,+value: F32,
  equation: {native == Writes.native(F32,previous,output) : Array<F32>}) ->
  {Gemm.store_lane(inside,native,offset,value) == Writes.native(F32,append_lane(inside,previous,offset,value),output) : Array<F32>}:
  %Equal.sym(Array<F32>,native,Writes.native(F32,previous,output),equation) :
    {Gemm.store_lane(inside,_,offset,value) == Writes.native(F32,append_lane(inside,previous,offset,value),output) : Array<F32>}
  Equal.trans(Array<F32>,Gemm.store_lane(inside,Writes.native(F32,previous,output),offset,value),
    Writes.native(F32,lane_writes(inside,offset,value),Writes.native(F32,previous,output)),
    Writes.native(F32,append_lane(inside,previous,offset,value),output),
    native_lane(inside,Writes.native(F32,previous,output),offset,value),
    Equal.sym(Array<F32>,Writes.native(F32,append_lane(inside,previous,offset,value),output),
      Writes.native(F32,lane_writes(inside,offset,value),Writes.native(F32,previous,output)),
      Writes.concatenated_native(F32,previous,lane_writes(inside,offset,value),output)))

def tail_assignments(+count: U32,+offset: U32,+bias: F32,+accumulator: Tile.Tile8) -> List<&2,Writes.Assignment<F32>>:
  a = lane_writes(U32.is_lt(0,count),offset,(Tile.lane0(accumulator) + bias : F32))
  a = append_lane(U32.is_lt(1,count),a,(offset + 1 : U32),(Tile.lane1(accumulator) + bias : F32))
  a = append_lane(U32.is_lt(2,count),a,(offset + 2 : U32),(Tile.lane2(accumulator) + bias : F32))
  a = append_lane(U32.is_lt(3,count),a,(offset + 3 : U32),(Tile.lane3(accumulator) + bias : F32))
  a = append_lane(U32.is_lt(4,count),a,(offset + 4 : U32),(Tile.lane4(accumulator) + bias : F32))
  a = append_lane(U32.is_lt(5,count),a,(offset + 5 : U32),(Tile.lane5(accumulator) + bias : F32))
  a = append_lane(U32.is_lt(6,count),a,(offset + 6 : U32),(Tile.lane6(accumulator) + bias : F32))
  append_lane(U32.is_lt(7,count),a,(offset + 7 : U32),(Tile.lane7(accumulator) + bias : F32))

def tail_native(+count: U32,-output: Array<F32>,+offset: U32,+bias: F32,+accumulator: Tile.Tile8) ->
  {Gemm.store_tail(count,output,offset,bias,accumulator) == Writes.native(F32,tail_assignments(count,offset,bias,accumulator),output) : Array<F32>}:
  +v0 = (Tile.lane0(accumulator) + bias : F32)
  +v1 = (Tile.lane1(accumulator) + bias : F32)
  +v2 = (Tile.lane2(accumulator) + bias : F32)
  +v3 = (Tile.lane3(accumulator) + bias : F32)
  +v4 = (Tile.lane4(accumulator) + bias : F32)
  +v5 = (Tile.lane5(accumulator) + bias : F32)
  +v6 = (Tile.lane6(accumulator) + bias : F32)
  +v7 = (Tile.lane7(accumulator) + bias : F32)
  +a0 = lane_writes(U32.is_lt(0,count),offset,v0)
  +a1 = append_lane(U32.is_lt(1,count),a0,(offset + 1 : U32),v1)
  +a2 = append_lane(U32.is_lt(2,count),a1,(offset + 2 : U32),v2)
  +a3 = append_lane(U32.is_lt(3,count),a2,(offset + 3 : U32),v3)
  +a4 = append_lane(U32.is_lt(4,count),a3,(offset + 4 : U32),v4)
  +a5 = append_lane(U32.is_lt(5,count),a4,(offset + 5 : U32),v5)
  +a6 = append_lane(U32.is_lt(6,count),a5,(offset + 6 : U32),v6)
  -o0 = Gemm.store_lane(U32.is_lt(0,count),output,offset,v0)
  -o1 = Gemm.store_lane(U32.is_lt(1,count),o0,(offset + 1 : U32),v1)
  -o2 = Gemm.store_lane(U32.is_lt(2,count),o1,(offset + 2 : U32),v2)
  -o3 = Gemm.store_lane(U32.is_lt(3,count),o2,(offset + 3 : U32),v3)
  -o4 = Gemm.store_lane(U32.is_lt(4,count),o3,(offset + 4 : U32),v4)
  -o5 = Gemm.store_lane(U32.is_lt(5,count),o4,(offset + 5 : U32),v5)
  -o6 = Gemm.store_lane(U32.is_lt(6,count),o5,(offset + 6 : U32),v6)
  next_lane(U32.is_lt(7,count),o6,a6,output,(offset + 7 : U32),v7,
    next_lane(U32.is_lt(6,count),o5,a5,output,(offset + 6 : U32),v6,
      next_lane(U32.is_lt(5,count),o4,a4,output,(offset + 5 : U32),v5,
        next_lane(U32.is_lt(4,count),o3,a3,output,(offset + 4 : U32),v4,
          next_lane(U32.is_lt(3,count),o2,a2,output,(offset + 3 : U32),v3,
            next_lane(U32.is_lt(2,count),o1,a1,output,(offset + 2 : U32),v2,
              next_lane(U32.is_lt(1,count),o0,a0,output,(offset + 1 : U32),v1,
                native_lane(U32.is_lt(0,count),output,offset,v0))))))))

# Keep stores as syntax: readback proofs can reason about one row without
# repeatedly normalizing the capacity of all thirty-two nested writes.
def store(count: U32,output: Writes.Program<F32>,offset: U32,bias: F32,accumulator: Tile.Tile8) -> Writes.Program<F32>:
  Writes.Append{tail_assignments(count,offset,bias,accumulator),output}

def stored(+count: U32,+output: Writes.Program<F32>,+offset: U32,+bias: F32,+accumulator: Tile.Tile8) ->
  {Gemm.store_tail(count,Tree.pack(F32,Writes.evaluate(F32,output)),offset,bias,accumulator) == Tree.pack(F32,Writes.evaluate(F32,store(count,output,offset,bias,accumulator))) : Array<F32>}:
  Equal.trans(Array<F32>,Gemm.store_tail(count,Tree.pack(F32,Writes.evaluate(F32,output)),offset,bias,accumulator),
    Writes.native(F32,tail_assignments(count,offset,bias,accumulator),Tree.pack(F32,Writes.evaluate(F32,output))),
    Tree.pack(F32,Writes.evaluate(F32,store(count,output,offset,bias,accumulator))),
    tail_native(count,Tree.pack(F32,Writes.evaluate(F32,output)),offset,bias,accumulator),
    Writes.refinement(F32,tail_assignments(count,offset,bias,accumulator),Writes.evaluate(F32,output)))

def row(valid: Bool,+index: U32,+position: U32,+columns: U32,accumulator: Tile.Tile8,owners: Owners) -> Owners:
  match valid owners:
    case False{} owners: owners
    case True{} Owners{+bias,output}:
      Owners{bias,store(U32.min(8,(columns - position : U32)),output,(index * columns + position : U32),Tree.read(F32,bias,index),accumulator)}

def row_matches(+valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+owners: Owners) ->
  {Gemm.write_row(valid,index,position,columns,accumulator,represent(owners)) ==
    represent(row(valid,index,position,columns,accumulator,owners)) : Array<F32> & Array<F32>}:
  match valid owners:
    case False{} owners: {==}
    case True{} Owners{+bias,+output}:
      %Equal.sym(Array<F32> & F32,Array.get(F32,Tree.pack(F32,bias),index),(Tree.pack(F32,bias),Tree.read(F32,bias,index)),StorageLaws.read(F32,bias,index)) :
        {Gemm.write_bias(Tree.pack(F32,Writes.evaluate(F32,output)),(index * columns + position : U32),accumulator,U32.min(8,(columns - position : U32)),_) ==
          (Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,store(U32.min(8,(columns - position : U32)),output,(index * columns + position : U32),Tree.read(F32,bias,index),accumulator)))) : Array<F32> & Array<F32>}
      Equal.cong(Array<F32>,Array<F32> & Array<F32>,result => (Tree.pack(F32,bias),result),
        Gemm.store_tail(U32.min(8,(columns - position : U32)),Tree.pack(F32,Writes.evaluate(F32,output)),(index * columns + position : U32),Tree.read(F32,bias,index),accumulator),
        Tree.pack(F32,Writes.evaluate(F32,store(U32.min(8,(columns - position : U32)),output,(index * columns + position : U32),Tree.read(F32,bias,index),accumulator))),
        stored(U32.min(8,(columns - position : U32)),output,(index * columns + position : U32),Tree.read(F32,bias,index),accumulator))

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
