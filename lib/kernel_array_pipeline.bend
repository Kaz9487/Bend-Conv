# Raw-owner adapter for the internal Array executor and its unconditional root.
# Public tensors use the certified adapter of the very same partition schedule.
import Base
import ./kernel_pipeline.bend as Pipeline
import ./kernel_gemm.bend as Gemm
import ./kernel_weights.bend as Weights
import ./kernel_spatial_packing.bend as Packing
import ./kernel_gather.bend as Gather
import ./kernel_tile.bend as Tile
import ./storage_allocation.bend as Allocation
import ./tensor.bend as Tensors
import ./traversal_partition.bend as Partition
import ./kernel_window.bend as KernelWindow

def allocated(+rows: U32,+span: U32) -> Array<F32>:
  Allocation.array(~F32,Tensors.depth_for((rows * span : U32)),0.0)

def computed(+rows: U32,+span: U32,inner: U32,weights: Array<Tile.Tile8>,bias: Array<F32>,packed: Array<F32> & Array<Tile.Tile8>) -> Array<F32>:
  (input,tiles) = packed
  Gemm.run(rows,tiles,weights,bias,allocated(rows,span),span,inner)

# Rows [first, first + rows) of every channel of the image, stored contiguously.
type InputWindow is Type:
  InputWindow{owner: Array<F32>,first: U32,rows: U32}

def windows(+first: U32,+rows: U32,+left_first: U32,+left_rows: U32,owners: Array<F32> & Array<F32>) -> InputWindow & InputWindow:
  (source,target) = owners
  (InputWindow{target,left_first,left_rows},InputWindow{source,first,rows})

def cloned(+first: U32,+rows: U32,owners: Array<F32> & Array<F32>) -> InputWindow & InputWindow:
  (left,right) = owners
  (InputWindow{left,first,rows},InputWindow{right,first,rows})

def copied(+channels: U32,+width: U32,+first: U32,+rows: U32,band: KernelWindow.Band,owner: Array<F32>) -> InputWindow & InputWindow:
  KernelWindow.Band{+left_first,+left_rows} = band
  +count = (channels * left_rows * width : U32)
  windows(first,rows,left_first,left_rows,KernelWindow.copy_planes(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),
    ((left_first - first) * width : U32),(left_rows * width : U32),(owner,Allocation.array(~F32,Tensors.depth_for(count),0.0))))

def split_when(copy: Bool,+channels: U32,+width: U32,+first: U32,+rows: U32,+band: KernelWindow.Band,owner: Array<F32>) -> InputWindow & InputWindow:
  match copy:
    case False{}: cloned(first,rows,Array.clone(F32,owner))
    case True{}: copied(channels,width,first,rows,band,owner)

# The right child keeps the parent's rows; the left child copies only its band.
# Unaligned rows would break whole eight-position blocks, so they are cloned;
# so is a band that fails its coverage check.
def split(work: Pipeline.Work,offset: U32,half: U32,window: InputWindow) -> InputWindow & InputWindow:
  Pipeline.Work{channels_out,+height,+width,+kernel,+stride,+padding,+output_width,positions,+inner} = work
  InputWindow{owner,+first,+rows} = window
  +offset = offset
  +half = half
  +band = KernelWindow.band(first,rows,height,kernel,stride,padding,output_width,offset,half)
  split_when(KernelWindow.copies_band(first,rows,height,kernel,stride,padding,output_width,offset,half,band),
    (inner / (kernel * kernel) : U32),width,first,rows,band,owner)

def leaf(work: Pipeline.Work,offset: U32,span: U32,window: InputWindow,weights: Array<Tile.Tile8>,bias: Array<F32>) -> Partition.Tree<&1,Array<F32>,U32>:
  Pipeline.Work{+rows,h,w,k,+s,p,+ow,positions,+inner} = work
  InputWindow{input,+first,height} = window
  +offset = offset
  +span = span
  +count = (span / 8 * inner : U32)
  Partition.Leaf{computed(rows,span,inner,weights,bias,
    Packing.run(Packing.Call{U32.to_nat(count),KernelWindow.start(offset,s,ow,first),height,w,k,s,p,ow,inner},input,
      Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))),offset,span,rows}

def partition(depth: Nat,work: Pipeline.Work,offset: U32,span: U32,input: InputWindow,weights: Array<Tile.Tile8>,bias: Array<F32>) -> Partition.Tree<&1,Array<F32>,U32>:
  Partition.run(~InputWindow,~Array<Tile.Tile8>,~Array<F32>,~Array<F32>,~Pipeline.Work,
    ~split,~Array.clone(Tile.Tile8),~Array.clone(F32),~leaf,depth,work,offset,span,input,weights,bias)

def finish_leaf(identity: Bool,input: Array<F32>,+rows: U32,offset: U32,span: U32,+positions: U32) -> Array<F32>:
  match identity:
    case True{}: input
    case False{}: Gather.gather_block(rows,offset,positions,span,input,allocated(rows,positions))

def finish(partitions: Partition.Tree<&1,Array<F32>,U32>,+rows: U32,+positions: U32) -> Array<F32>:
  match partitions:
    case Partition.Leaf{input,+offset,+span,channels}:
      finish_leaf(U32.is_zero(offset) && U32.is_eq(span,positions),input,channels,offset,span,positions)
    case Partition.Fork{left,right}:
      Partition.gather(~Array<F32>,~(rows => offset => positions => span => input => output => Gather.gather_block(rows,offset,positions,span,input,output)),Partition.Fork{left,right},allocated(rows,positions),positions)

def run(depth: U32,work: Pipeline.Work,input: Array<F32>,weights: Array<F32>,bias: Array<F32>) -> Array<F32>:
  Pipeline.Work{+rows,+h,w,k,s,p,ow,+positions,+inner} = work
  +padded = KernelWindow.padded(positions)
  +limit = Pipeline.depth_limit(padded)
  finish(partition(U32.to_nat(U32.min(depth,limit)),Pipeline.Work{rows,h,w,k,s,p,ow,positions,inner},0,padded,InputWindow{input,0,h},Weights.run(rows,inner,weights),bias),rows,positions)
