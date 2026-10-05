# Reads of a tile inside one output row or two: the native pixel and block
# readers on a packed tree return the tree unchanged and the masked value of
# the tree model; the eight lanes of a block are the tabulation of its reader.
import Base
import ../../lib/math_sequence.bend as Sequence
import ../../lib/kernel_packing_samples.bend as PackingSamples
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/indexed_sequence_laws.bend as SequenceLaws
import ../../lib/proofs/natural_addition.bend as Addition

def value(+x: Tree.Tree<F32>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32) -> F32:
  PackingSamples.masked(Tree.read(F32,x,(cb + iy * w + ix : U32)),U32.is_lt(iy,h) && U32.is_lt(ix,w))

def pixel(+x: Tree.Tree<F32>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32) -> {Packing.pixel(Tree.pack(F32,x),iy,ix,cb,h,
    w) == (Tree.pack(F32,x),value(x,iy,ix,cb,h,w)) : Array<F32> & F32}:
  %Equal.sym(Array<F32> & F32,Array.get(F32,Tree.pack(F32,x),(cb + iy * w + ix : U32)),(Tree.pack(F32,x),Tree.read(F32,x,
        (cb + iy * w + ix : U32))),StorageLaws.read(F32,x,(cb + iy * w + ix : U32))) : {Packing.masked(U32.is_lt(iy,h) && U32.is_lt(ix,
        w),_) == (Tree.pack(F32,x),value(x,iy,ix,cb,h,w)) : Array<F32> & F32}
  {==}

def pack_input(input: Tree.Tree<F32>) -> Array<F32>:
  Tree.pack(F32,input)

def block_sample(input: Tree.Tree<F32>,context: Packing.PixelBlock,index: Nat) -> F32:
  Packing.PixelBlock{row,column,channel_base,height,width,+stride} = context
  value(input,row,(column + U32.from_nat(index) * stride : U32),channel_base,height,width)

def native_block_read(+input: Tree.Tree<F32>,+context: Packing.PixelBlock,+index: Nat) -> {Packing.read_block_pixel(pack_input(input),
    context,index) == (pack_input(input),block_sample(input,context,index)) : Array<F32> & F32}:
  Packing.PixelBlock{row,column,channel_base,height,width,+stride} = context
  pixel(input,row,(column + U32.from_nat(index) * stride : U32),channel_base,height,width)

def native_vector(input: Array<F32>,context: Packing.PixelBlock) -> Array<F32> & Tile.Tile8:
  Packing.PixelBlock{+row,+column,+channel_base,+height,+width,+stride} = context
  Tile.read_tail(~Array<F32>,~Packing.PixelBlock,~Packing.read_block_pixel,
    Packing.pixel(input,row,column,channel_base,height,width),context)

def pure_vector(+input: Tree.Tree<F32>,+context: Packing.PixelBlock) -> Tile.Tile8:
  Tile.from_sequence(Sequence.tabulate(~F32,~TileLaws.SampleContext<Tree.Tree<F32>,Packing.PixelBlock>,
    ~TileLaws.sample_context(~Tree.Tree<F32>,~Packing.PixelBlock,~block_sample),8n,0n,TileLaws.SampleContext{input,context}))

def zero_lane_column(+column: U32,+stride: U32) -> {(column + 0 * stride : U32) == column : U32}:
  %Equal.sym(U32,(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)) : {(column + _ : U32) == column : U32}
  Arithmetic.add_zero(column)

def native_vector_matches_tabulation(+input: Tree.Tree<F32>,+context: Packing.PixelBlock) ->
  {native_vector(Tree.pack(F32,input),context) == (Tree.pack(F32,input),pure_vector(input,context)) : Array<F32> & Tile.Tile8}:
  Packing.PixelBlock{+row,+column,+channel_base,+height,+width,+stride} = context
  %Equal.sym(Array<F32> & F32,Packing.pixel(Tree.pack(F32,input),row,column,channel_base,height,width),
    (Tree.pack(F32,input),value(input,row,column,channel_base,height,width)),pixel(input,row,column,channel_base,height,width)) :
    {Tile.read_tail(~Array<F32>,~Packing.PixelBlock,~Packing.read_block_pixel,_,context) == (Tree.pack(F32,input),pure_vector(input,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(U32,(column + 0 * stride : U32),column,zero_lane_column(column,stride)) :
    {Tile.read_tail(~Array<F32>,~Packing.PixelBlock,~Packing.read_block_pixel,(Tree.pack(F32,input),value(input,row,column,channel_base,height,width)),context) ==
      (Tree.pack(F32,input),Tile.from_sequence(Sequence.Elements{value(input,row,_,channel_base,height,width),
        Sequence.tabulate(~F32,~TileLaws.SampleContext<Tree.Tree<F32>,Packing.PixelBlock>,~TileLaws.sample_context(~Tree.Tree<F32>,~Packing.PixelBlock,~block_sample),7n,1n,TileLaws.SampleContext{input,context})})) : Array<F32> & Tile.Tile8}
  TileLaws.tail_block_refines(~Array<F32>,~Tree.Tree<F32>,~Packing.PixelBlock,~pack_input,~Packing.read_block_pixel,
    ~block_sample,~native_block_read,input,context,value(input,row,column,channel_base,height,width))

# A lane of a tile that may end in the next output row.
def wrap_value(same: Bool,+x: Tree.Tree<F32>,+row: U32,+column: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32,+lane: U32) -> F32:
  match same:
    case True{}: value(x,row,(column + lane * s : U32),cb,h,w)
    case False{}: value(x,next_row,((offset + lane - ow) * s + kc - p : U32),cb,h,w)

def wrap_pixel(+same: Bool,+x: Tree.Tree<F32>,+row: U32,+column: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32,+lane: U32) ->
  {Packing.wrap_pixel(same,Tree.pack(F32,x),row,column,cb,h,w,s,next_row,offset,ow,kc,p,lane) == (Tree.pack(F32,x),wrap_value(same,x,row,column,cb,h,w,s,next_row,offset,ow,kc,p,lane)) : Array<F32> & F32}:
  match same:
    case True{}: pixel(x,row,(column + lane * s : U32),cb,h,w)
    case False{}: pixel(x,next_row,((offset + lane - ow) * s + kc - p : U32),cb,h,w)

def wrap_lane(+x: Tree.Tree<F32>,+row: U32,+column: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32,+lane: U32) -> F32:
  wrap_value(U32.is_lt((offset + lane : U32),ow),x,row,column,cb,h,w,s,next_row,offset,ow,kc,p,lane)

def wrap_sample(input: Tree.Tree<F32>,context: Packing.WrapBlock,index: Nat) -> F32:
  Packing.WrapBlock{row,column,cb,h,w,s,next_row,offset,ow,kc,p} = context
  wrap_lane(input,row,column,cb,h,w,s,next_row,offset,ow,kc,p,U32.from_nat(index))

def native_wrap_read(+input: Tree.Tree<F32>,+context: Packing.WrapBlock,+index: Nat) -> {Packing.read_wrap_pixel(pack_input(input),
    context,index) == (pack_input(input),wrap_sample(input,context,index)) : Array<F32> & F32}:
  Packing.WrapBlock{+row,+column,+cb,+h,+w,+s,+next_row,+offset,+ow,+kc,+p} = context
  wrap_pixel(U32.is_lt((offset + U32.from_nat(index) : U32),ow),input,row,column,cb,h,w,s,next_row,offset,ow,kc,p,U32.from_nat(index))

# Lane 0 lies in the output row of the tile.
def pure_wrap_vector(+x: Tree.Tree<F32>,+row: U32,+column: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32) -> Tile.Tile8:
  Tile.from_sequence(Sequence.Elements{value(x,row,column,cb,h,w),
    Sequence.tabulate(~F32,~TileLaws.SampleContext<Tree.Tree<F32>,Packing.WrapBlock>,~TileLaws.sample_context(~Tree.Tree<F32>,~Packing.WrapBlock,~wrap_sample),7n,1n,
      TileLaws.SampleContext{x,Packing.WrapBlock{row,column,cb,h,w,s,next_row,offset,ow,kc,p}})})

def native_wrap_vector_matches_tabulation(+x: Tree.Tree<F32>,+row: U32,+column: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32) ->
  {Packing.wrap_block(Tree.pack(F32,x),row,column,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),pure_wrap_vector(x,row,column,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Packing.pixel(Tree.pack(F32,x),row,column,cb,h,w),(Tree.pack(F32,x),value(x,row,column,cb,h,w)),pixel(x,row,column,cb,h,w)) :
    {Tile.read_tail(~Array<F32>,~Packing.WrapBlock,~Packing.read_wrap_pixel,_,Packing.WrapBlock{row,column,cb,h,w,s,next_row,offset,ow,kc,p}) ==
      (Tree.pack(F32,x),pure_wrap_vector(x,row,column,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8}
  TileLaws.tail_block_refines(~Array<F32>,~Tree.Tree<F32>,~Packing.WrapBlock,~pack_input,~Packing.read_wrap_pixel,
    ~wrap_sample,~native_wrap_read,x,Packing.WrapBlock{row,column,cb,h,w,s,next_row,offset,ow,kc,p},value(x,row,column,cb,h,w))
