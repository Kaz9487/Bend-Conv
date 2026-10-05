# Indexed mapping with a runtime context and a closed index template. The
# source and destination traversal are shared with checked tensor operations.
import Base
import ./traversal_array.bend as Traversal
import ./traversal_loop.bend as Loop
import ./default_machine_limits.bend as Default

def value(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  context: Context,position: U32,result: Array<Element> & Element) -> Array<Element> & Element:
  (source,element) = result
  (source,operation(index(context,position),element))

def read(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  source: Array<Element>,context: Context,position: U32) -> Array<Element> & Element:
  +position = position
  value(~Element,~Context,~index,~operation,context,position,Array.get(Element,source,position))

def destination(~Context: Data,context: Context,position: U32) -> U32:
  position

type Call<-Context: Data> is Data:
  Call{count: Nat,context: Context}

def run(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  call: Call<Context>,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  Call{count,context} = call
  Traversal.finish(~Array<Element>,~Element,~Context,
    Traversal.run(~Array<Element>,~Element,~Context,~read(~Element,~Context,~index,~operation),~destination(~Context),count,
      Traversal.State{input,output,0,context}))

# The same mapping over blocks of consecutive positions. A block starts where
# the previous one ended and gives position p the index base + (p - start),
# so an index that is a quotient and a remainder of the position costs one
# base per block instead of a division per element.
type Block is Data:
  Block{base: U32,start: U32}

def block_index(block: Block,position: U32) -> U32:
  Block{base,start} = block
  (base + (position - start) : U32)

type Blocks<-Context: Data> is Data:
  Blocks{block: U32,size: U32,context: Context}

def next_block(~Element: Data,~Context: Data,block: U32,size: U32,context: Context,
  state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Block>) -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Blocks<Context>>:
  Traversal.State{input,output,position,inner} = state
  Traversal.State{input,output,position,Blocks{(block + 1 : U32),size,context}}

def block_step(~Element: Data,~Context: Data,~base: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Blocks<Context>>) -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Blocks<Context>>:
  Traversal.State{input,output,+position,Blocks{+block,+size,+context}} = state
  next_block(~Element,~Context,block,size,context,
    Traversal.run(~Array<Element>,~Element,~Block,~read(~Element,~Block,~block_index,~operation),~destination(~Block),U32.to_nat(size),
      Traversal.State{input,output,position,Block{base(context,block),position}}))

def run_blocks(~Element: Data,~Context: Data,~base: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  blocks: Nat,size: U32,context: Context,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  Traversal.finish(~Array<Element>,~Element,~Blocks<Context>,
    Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Blocks<Context>>,~block_step(~Element,~Context,~base,~operation),blocks,
      Traversal.State{input,output,0,Blocks{0,size,context}}))

def run_whole(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  whole: Bool,blocks: U32,size: U32,total: U32,context: Context,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  match whole:
    case True{}: run_blocks(~Element,~Context,~base,~operation,U32.to_nat(blocks),size,context,input,output)
    case False{}: run(~Element,~Context,~index,~operation,Call{U32.to_nat(total),context},input,output)

# Blocks are used only up to the machine capacity, where the block size
# divides a position in U32 without overflow. No storage is larger.
def block_limit() -> U32:
  Default.capacity()

# total positions in blocks(context) blocks of size(context) positions. The
# blocks are used only when they are exactly the total and the total is
# within the limit; any other count keeps the index function.
def run_tiled(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~blocks: Context -> U32,~size: Context -> U32,
  ~operation: U32 -> Element -> Element,+total: U32,+context: Context,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  run_whole(~Element,~Context,~index,~base,~operation,U32.is_le(total,block_limit()) && Nat.is_eq(Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))),U32.to_nat(total)),
    blocks(context),size(context),total,context,input,output)
