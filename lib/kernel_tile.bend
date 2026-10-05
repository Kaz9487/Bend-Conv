# Measured flat FP32 tile representation; logical views stay out of hot loops.
import Base
import ./kernel_block.bend as Block
import ./math_sequence.bend as Sequence
import ./math_fp32.bend as Fp32

type Tile8 is Data:
  Tile{a0: F32, a1: F32, a2: F32, a3: F32, a4: F32, a5: F32, a6: F32, a7: F32}

def filled(+value: F32) -> Tile8:
  Tile{value,value,value,value,value,value,value,value}

def lane0(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  a

def lane1(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  b

def lane2(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  c

def lane3(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  d

def lane4(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  e

def lane5(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  f

def lane6(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  g

def lane7(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  h

# One ordered multiply-add per lane: eight output positions, one weight.
def update(accumulator: Tile8,input: Tile8,weight: F32) -> Tile8:
  Tile{a,b,c,d,e,f,g,h} = accumulator
  Tile{x0,x1,x2,x3,x4,x5,x6,x7} = input
  +weight = weight
  Tile{Fp32.multiply_add(a,x0,weight),Fp32.multiply_add(b,x1,weight),Fp32.multiply_add(c,x2,weight),Fp32.multiply_add(d,x3,weight),
    Fp32.multiply_add(e,x4,weight),Fp32.multiply_add(f,x5,weight),Fp32.multiply_add(g,x6,weight),Fp32.multiply_add(h,x7,weight)}

def from_sequence(vector: Sequence.Values(F32,8n)) -> Tile8:
  Sequence.Elements{a0,Sequence.Elements{a1,Sequence.Elements{a2,Sequence.Elements{a3,Sequence.Elements{a4,Sequence.Elements{a5,Sequence.Elements{a6,
          Sequence.Elements{a7,Unit{}}}}}}}}} = vector
  Tile{a0,a1,a2,a3,a4,a5,a6,a7}

type HeadContext<-Context: Data> is Data:
  HeadContext{head: F32,context: Context}

def read_with_head(
  ~State: Type,~Context: Data,~read: State -> Context -> Nat -> State & F32,
  state: State,combined: HeadContext<Context>,index: Nat
) -> State & F32:
  HeadContext{head,context} = combined
  match index:
    case 0n: (state,head)
    case 1n+rest: read(state,context,1n+rest)

def pairs_value(vector: Block.Twin<Block.Twin<Block.Twin<F32>>>) -> Tile8:
  Block.Twin{Block.Twin{Block.Twin{a0,a1},Block.Twin{a2,a3}},Block.Twin{Block.Twin{a4,a5},
      Block.Twin{a6,a7}}} = vector
  Tile{a0,a1,a2,a3,a4,a5,a6,a7}

def finish_pairs(~State: Type,result: State & Block.Twin<Block.Twin<Block.Twin<F32>>>) -> State & Tile8:
  (state,vector) = result
  (state,pairs_value(vector))

def read_tail(~State: Type,~Context: Data,~read: State -> Context -> Nat -> State & F32,first: State & F32,context: Context) -> State & Tile8:
  (state,head) = first
  finish_pairs(~State,
    Block.read_pair(~State,~HeadContext<Context>,~Block.Twin<Block.Twin<F32>>,
      ~Block.read_pair(~State,~HeadContext<Context>,~Block.Twin<F32>,
        ~Block.read_pair(~State,~HeadContext<Context>,~F32,
          ~read_with_head(~State,~Context,~read),~1n),~2n),~4n,
      state,HeadContext{head,context},0n))

# Logical view only; runtime kernels never traverse it.

def view(tile: Tile8) -> Sequence.Values(F32,8n):
  Tile{a0,a1,a2,a3,a4,a5,a6,a7} = tile
  Sequence.Elements{a0,Sequence.Elements{a1,Sequence.Elements{a2,Sequence.Elements{a3,Sequence.Elements{a4,Sequence.Elements{a5,Sequence.Elements{a6,
          Sequence.Elements{a7,Unit{}}}}}}}}}

def element_at(index: Sequence.Index(8n),tile: Tile8) -> F32:
  Sequence.get(F32,8n,index,view(tile))