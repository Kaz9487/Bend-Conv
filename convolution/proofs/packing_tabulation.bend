# The flat tabulation of packed entries: entry i holds tile i / inner at reduction
# index i % inner, every lane read independently. packing_contents proves the
# packer stores it.
# The tabulation still uses native pixel addresses. Connecting its quotient /
# remainder coordinates to CONV.patch is a separate theorem, not assumed here.
import Base
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/math_sequence.bend as Sequence
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./packing_lane_reads.bend as LaneReads
import ./tile_array_projection.bend as ArrayProjection
import ./packed_reduction_samples.bend as ReductionSamples
import ../specification.bend as ConvSpec
import ../../lib/kernel_tile.bend as Tile

def indexed_position(+position: U32,index: Nat) -> U32:
  match index:
    case 0n: position
    case 1n+rest: (position + U32.from_nat(1n+rest) : U32)

type PixelContext is Data:
  PixelContext{input: Tree.Tree<F32>,position: U32,reduction: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,
    output_width: U32}

def sample_pixel(context: PixelContext,index: Nat) -> F32:
  PixelContext{input,position,reduction,height,width,kernel,stride,padding,output_width} = context
  LaneReads.value(input,indexed_position(position,index),reduction,height,width,kernel,stride,padding,output_width)

def vector(+input: Tree.Tree<F32>,+position: U32,+reduction: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_width: U32) -> Tile.Tile8:
  Tile.from_sequence(Sequence.tabulate(~F32,~PixelContext,~sample_pixel,8n,0n,
    PixelContext{input,position,reduction,height,width,kernel,stride,padding,output_width}))

def entry(+i: U32,+start: U32,+x: Tree.Tree<F32>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,+inner: U32) -> Tile.Tile8:
  vector(x,(start + (i / inner) * 8 : U32),(i % inner : U32),h,w,k,stride,pad,output_width)

def end(n: Nat,i: U32) -> U32:
  match n:
    case 0n:
      i
    case 1n+rest:
      end(rest,(i + 1 : U32))

def fill(n: Nat,+i: U32,+start: U32,+x: Tree.Tree<F32>,y: Tree.Tree<Tile.Tile8>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,
  +output_width: U32,+inner: U32) -> Tree.Tree<Tile.Tile8>:
  match n:
    case 0n:
      y
    case 1n+rest:
      fill(rest,(i + 1 : U32),start,x,Tree.write(Tile.Tile8,y,i,entry(i,start,x,h,w,k,stride,pad,output_width,inner)),h,w,k,stride,pad,output_width,inner)

def lane_values(n: Nat,+i: U32,+start: U32,+x: Tree.Tree<F32>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,+inner: U32,
  +lane: Sequence.Index(8n)) -> List<&2,F32>:
  match n:
    case 0n:
      []
    case 1n+rest:
      ReductionSamples.select(lane,entry(i,start,x,h,w,k,stride,pad,output_width,inner)) <> lane_values(rest,(i + 1 : U32),start,x,h,w,k,stride,pad,
        output_width,inner,lane)

def length(+n: Nat,+i: U32,+start: U32,+x: Tree.Tree<F32>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,+inner: U32,
  +lane: Sequence.Index(8n)) -> {ConvSpec.length(lane_values(n,i,start,x,h,w,k,stride,pad,output_width,inner,lane)) == n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %length(rest,(i + 1 : U32),start,x,h,w,k,stride,pad,output_width,inner,lane) : {1n+ConvSpec.length(lane_values(rest,(i + 1 : U32),start,x,h,w,k,
            stride,pad,output_width,inner,lane)) == 1n+_ : Nat}
      {==}

law projected:
  for +n: Nat
  for +i: U32
  for +start: U32
  for +x: Tree.Tree<F32>
  for +y: Tree.Tree<Tile.Tile8>
  for +h: U32
  for +w: U32
  for +k: U32
  for +stride: U32
  for +pad: U32
  for +output_width: U32
  for +inner: U32
  for +lane: Sequence.Index(8n)
  {ArrayProjection.project(fill(n,i,start,x,y,h,w,k,stride,pad,output_width,inner),lane) == Tree.write_sequence(~F32,lane_values(n,i,start,x,h,w,k,
        stride,pad,output_width,inner,lane),i,ArrayProjection.project(y,lane)) : Tree.Tree<F32>}
def projected(n,i,start,x,y,h,w,k,stride,pad,output_width,inner,lane):
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %ArrayProjection.write_projected(y,lane,i,entry(i,start,x,h,w,k,stride,pad,output_width,inner)) : {ArrayProjection.project(fill(1n+rest,i,start,x,y,h,w,k,
            stride,pad,output_width,inner),lane) == Tree.write_sequence(~F32,lane_values(rest,(i + 1 : U32),start,x,h,w,k,stride,pad,output_width,inner,lane),
          (i + 1 : U32),_) : Tree.Tree<F32>}
      projected(rest,(i + 1 : U32),start,x,Tree.write(Tile.Tile8,y,i,entry(i,start,x,h,w,k,stride,pad,output_width,inner)),h,w,k,stride,pad,output_width,inner,lane)

# Overwriting any prefix leaves the reusable tile workspace's depth unchanged.
def fill_balanced(+n: Nat,+i: U32,+start: U32,+x: Tree.Tree<F32>,+y: Tree.Tree<Tile.Tile8>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,
  +output_width: U32,+inner: U32,+depth: Nat,balanced: Shape.Balanced(Tile.Tile8,depth,y)) ->
  Shape.Balanced(Tile.Tile8,depth,fill(n,i,start,x,y,h,w,k,stride,pad,output_width,inner)):
  match n:
    case 0n: balanced
    case 1n+ +rest:
      fill_balanced(rest,(i + 1 : U32),start,x,Tree.write(Tile.Tile8,y,i,entry(i,start,x,h,w,k,stride,pad,output_width,inner)),h,w,k,stride,pad,output_width,inner,depth,
        Shape.write(Tile.Tile8,depth,y,i,entry(i,start,x,h,w,k,stride,pad,output_width,inner),balanced))
