import Base
import ./storage_buffer.bend as Storage
import ./storage_allocation.bend as Allocation
import ./traversal_array.bend as Traversal
import ./kernel_tile.bend as Tile
import ./tensor.bend as Tensors

# Each ordered K coordinate owns eight adjacent weights, one per output row of
# a block; rows past the last output read zero. One native tile read replaces
# eight separately masked scalar reads.
type Layout is Data:
  Layout{rows: U32,inner: U32}

type Context is Data:
  Context{rows: U32,inner: U32,row: U32,destination: U32}

def eighth(a: F32,b: F32,c: F32,d: F32,e: F32,f: F32,g: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,h) = read
  (input,Tile.Tile{a,b,c,d,e,f,g,h})

def seventh(+rows: U32,+inner: U32,+row: U32,+index: U32,a: F32,b: F32,c: F32,d: F32,e: F32,f: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,g) = read
  eighth(a,b,c,d,e,f,g,Storage.read_when(~F32,0.0,U32.is_lt((row + 7 : U32),rows),input,((row + 7) * inner + index : U32)))

def sixth(+rows: U32,+inner: U32,+row: U32,+index: U32,a: F32,b: F32,c: F32,d: F32,e: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,f) = read
  seventh(rows,inner,row,index,a,b,c,d,e,f,Storage.read_when(~F32,0.0,U32.is_lt((row + 6 : U32),rows),input,((row + 6) * inner + index : U32)))

def fifth(+rows: U32,+inner: U32,+row: U32,+index: U32,a: F32,b: F32,c: F32,d: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,e) = read
  sixth(rows,inner,row,index,a,b,c,d,e,Storage.read_when(~F32,0.0,U32.is_lt((row + 5 : U32),rows),input,((row + 5) * inner + index : U32)))

def fourth(+rows: U32,+inner: U32,+row: U32,+index: U32,a: F32,b: F32,c: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,d) = read
  fifth(rows,inner,row,index,a,b,c,d,Storage.read_when(~F32,0.0,U32.is_lt((row + 4 : U32),rows),input,((row + 4) * inner + index : U32)))

def third(+rows: U32,+inner: U32,+row: U32,+index: U32,a: F32,b: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,c) = read
  fourth(rows,inner,row,index,a,b,c,Storage.read_when(~F32,0.0,U32.is_lt((row + 3 : U32),rows),input,((row + 3) * inner + index : U32)))

def second(+rows: U32,+inner: U32,+row: U32,+index: U32,a: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,b) = read
  third(rows,inner,row,index,a,b,Storage.read_when(~F32,0.0,U32.is_lt((row + 2 : U32),rows),input,((row + 2) * inner + index : U32)))

def first(+rows: U32,+inner: U32,+row: U32,+index: U32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,a) = read
  second(rows,inner,row,index,a,Storage.read_when(~F32,0.0,U32.is_lt((row + 1 : U32),rows),input,((row + 1) * inner + index : U32)))

def read(input: Array<F32>,context: Context,index: U32) -> Array<F32> & Tile.Tile8:
  Context{+rows,+inner,+row,destination} = context
  +index = index
  first(rows,inner,row,index,Storage.read_when(~F32,0.0,U32.is_lt(row,rows),input,(row * inner + index : U32)))

def destination(context: Context,index: U32) -> U32:
  Context{rows,inner,row,destination} = context
  (destination + index : U32)

def copy_block(block: U32,layout: Layout,owners: Array<F32> & Array<Tile.Tile8>) -> Array<F32> & Array<Tile.Tile8>:
  Layout{rows,+inner} = layout
  (input,output) = owners
  +block = block
  Traversal.finish(~Array<F32>,~Tile.Tile8,~Context,
    Traversal.run(~Array<F32>,~Tile.Tile8,~Context,~read,~destination,U32.to_nat(inner),
      Traversal.State{input,output,0,Context{rows,inner,(block * 8 : U32),(block * inner : U32)}}))

def reordered(pair: Array<F32> & Array<Tile.Tile8>) -> Array<Tile.Tile8>:
  (input,output) = pair
  output

def run(+rows: U32,+inner: U32,input: Array<F32>) -> Array<Tile.Tile8>:
  +blocks = ((rows + 7) / 8 : U32)
  reordered(Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~Layout,~copy_block,U32.to_nat(blocks),0,
    Layout{rows,inner},(input,Allocation.array(~Tile.Tile8,Tensors.depth_for((blocks * inner : U32)),Tile.filled(0.0)))))
