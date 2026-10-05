# Each leaf owns its input, packs it, and computes before returning. The caller
# validates dimensions and chooses depth; the runtime schedules parallel leaves.
import Base
import ./kernel_gemm_buffer.bend as GemmBuffer
import ./kernel_tile.bend as Tile
import ./kernel_spatial_packing.bend as Packing
import ./kernel_packing_buffer.bend as PackingBuffer
import ./kernel_gather_buffer.bend as GatherBuffer
import ./storage_buffer.bend as Storage
import ./storage_clone.bend as Clone
import ./tensor.bend as Tensors
import ./traversal_partition.bend as Partition
import ./storage_copy.bend as Copy
import ./kernel_window.bend as KernelWindow
import ./traversal_array.bend as Traversal
import ./math_activation.bend as Activations
import ./proofs/traversal_certificate.bend as TraversalCertificate

type Work is Data:
  Work{rows: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,output_width: U32,positions: U32,inner: U32}

# A partition node's work and the activation its leaves apply.
type Task is Data:
  Task{work: Work,activation: Activations.Activation}

def Outputs() -> Type:
  Partition.Tree<&1,Storage.Buffer<F32>,U32>

def activated_value(~activation: F32 -> F32,context: Unit,value: F32) -> F32:
  activation(value)

def mapped(~Context: Data,~operation: Context -> F32 -> F32,+count: U32,+context: Context,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Storage.Buffer{storage,length,certificate} = buffer
  Storage.Buffer{Traversal.map_in_place(~F32,~Context,~operation,U32.to_nat(count),0,context,storage),length,
    TraversalCertificate.map_in_place(~F32,~Context,~operation,U32.to_nat(count),0,context,storage,certificate)}

def leaky_relu_value(slope: F32,value: F32) -> F32:
  Activations.leaky_relu_value(value,slope)

def unit_mapped(~operation: F32 -> F32,+count: U32,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  mapped(~Unit,~activated_value(~operation),count,Unit{},buffer)

# One tight pass over each finished leaf while it is still in cache; applying
# the activation inside the store measured about twice as slow. The built-in
# is chosen once per leaf, so each pass is specialized; the identity skips it.
def activated(~custom: F32 -> F32,activation: Activations.Activation,+count: U32,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  match activation:
    case Activations.Identity{}: buffer
    case Activations.ReLU{}: unit_mapped(~Activations.relu_value,count,buffer)
    case Activations.ReLU6{}: unit_mapped(~Activations.relu6_value,count,buffer)
    case Activations.LeakyReLU{slope}: mapped(~F32,~leaky_relu_value,count,slope,buffer)
    case Activations.Sigmoid{}: unit_mapped(~Activations.sigmoid_value,count,buffer)
    case Activations.SiLU{}: unit_mapped(~Activations.silu_value,count,buffer)
    case Activations.Tanh{}: unit_mapped(~Activations.tanh_value,count,buffer)
    case Activations.HardSigmoid{}: unit_mapped(~Activations.hard_sigmoid_value,count,buffer)
    case Activations.HardSwish{}: unit_mapped(~Activations.hard_swish_value,count,buffer)
    case Activations.GELUTanh{}: unit_mapped(~Activations.gelu_tanh_value,count,buffer)
    case Activations.Custom{operations}: unit_mapped(~custom,count,buffer)

def compute_tiles(~custom: F32 -> F32,activation: Activations.Activation,+rows: U32,+span: U32,inner: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  tiles: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<F32>:
  activated(~custom,activation,(rows * span : U32),GemmBuffer.run(rows,tiles,weights,bias,Tensors.buffer(~F32,0.0,(rows * span : U32)),span,inner))

def computed(~custom: F32 -> F32,activation: Activations.Activation,rows: U32,span: U32,inner: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  packed: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<F32>:
  (input,tiles) = packed
  compute_tiles(~custom,activation,rows,span,inner,weights,bias,tiles)

# Rows [first, first + rows) of every channel of the image, stored contiguously.
type InputWindow is Type:
  InputWindow{buffer: Storage.Buffer<F32>,first: U32,rows: U32}

def copy_rows(remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+target_stride: U32,
  owners: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Storage.Buffer<F32> & Storage.Buffer<F32>:
  match remaining:
    case 0n: owners
    case 1n+rest:
      (source,target) = owners
      copy_rows(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,
        Copy.range(~F32,count,(channel * source_stride + source_start : U32),(channel * target_stride : U32),source,target))

def windows(+first: U32,+rows: U32,+left_first: U32,+left_rows: U32,owners: Storage.Buffer<F32> & Storage.Buffer<F32>) -> InputWindow & InputWindow:
  (source,target) = owners
  (InputWindow{target,left_first,left_rows},InputWindow{source,first,rows})

def cloned(+first: U32,+rows: U32,owners: Storage.Buffer<F32> & Storage.Buffer<F32>) -> InputWindow & InputWindow:
  (left,right) = owners
  (InputWindow{left,first,rows},InputWindow{right,first,rows})

def copied(+channels: U32,+width: U32,+first: U32,+rows: U32,band: KernelWindow.Band,buffer: Storage.Buffer<F32>) -> InputWindow & InputWindow:
  KernelWindow.Band{+left_first,+left_rows} = band
  windows(first,rows,left_first,left_rows,copy_rows(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),
    ((left_first - first) * width : U32),(left_rows * width : U32),(buffer,Tensors.buffer(~F32,0.0,(channels * left_rows * width : U32)))))

def split_when(copy: Bool,+channels: U32,+width: U32,+first: U32,+rows: U32,+band: KernelWindow.Band,buffer: Storage.Buffer<F32>) -> InputWindow & InputWindow:
  match copy:
    case False{}: cloned(first,rows,Clone.clone(~F32,buffer))
    case True{}: copied(channels,width,first,rows,band,buffer)

# The right child keeps the parent's rows; the left child copies only its band.
# Unaligned rows would break whole eight-position blocks, so they are cloned;
# so is a band that fails its coverage check.
def split(work: Work,offset: U32,half: U32,window: InputWindow) -> InputWindow & InputWindow:
  Work{channels_out,+height,+width,+kernel,+stride,+padding,+output_width,positions,+inner} = work
  InputWindow{buffer,+first,+rows} = window
  +offset = offset
  +half = half
  +band = KernelWindow.band(first,rows,height,kernel,stride,padding,output_width,offset,half)
  split_when(KernelWindow.copies_band(first,rows,height,kernel,stride,padding,output_width,offset,half,band),
    (inner / (kernel * kernel) : U32),width,first,rows,band,buffer)

def split_task(task: Task,offset: U32,half: U32,window: InputWindow) -> InputWindow & InputWindow:
  Task{work,activation} = task
  split(work,offset,half,window)

def leaf(~custom: F32 -> F32,task: Task,offset: U32,span: U32,window: InputWindow,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) -> Outputs():
  Task{Work{+rows,h,w,k,+s,p,+ow,positions,+inner},activation} = task
  InputWindow{input,+first,height} = window
  +offset = offset
  +span = span
  +count = (span / 8 * inner : U32)
  Partition.Leaf{computed(~custom,activation,rows,span,inner,weights,bias,
    PackingBuffer.run(Packing.Call{U32.to_nat(count),KernelWindow.start(offset,s,ow,first),height,w,k,s,p,ow,inner},input,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),offset,span,rows}

def partition(~custom: F32 -> F32,depth: Nat,task: Task,offset: U32,span: U32,input: InputWindow,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) -> Outputs():
  Partition.run(~InputWindow,~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Storage.Buffer<F32>,~Task,
    ~split_task,~Clone.clone(~Tile.Tile8),~Clone.clone(~F32),~leaf(~custom),depth,task,offset,span,input,weights,bias)

def gather(partitions: Outputs(),output: Storage.Buffer<F32>,+positions: U32) -> Storage.Buffer<F32>:
  Partition.gather(~Storage.Buffer<F32>,~(rows => offset => positions => span => input => output => GatherBuffer.run(rows,offset,positions,span,input,output)),partitions,output,positions)

def finish_leaf(identity: Bool,input: Storage.Buffer<F32>,+rows: U32,offset: U32,span: U32,+positions: U32) -> Storage.Buffer<F32>:
  match identity:
    case True{}: input
    case False{}: GatherBuffer.run(rows,offset,positions,span,input,Tensors.buffer(~F32,0.0,(rows * positions : U32)))

def finish(partitions: Outputs(),+rows: U32,+positions: U32) -> Storage.Buffer<F32>:
  match partitions:
    case Partition.Leaf{input,+offset,+span,channels}:
      finish_leaf(U32.is_zero(offset) && U32.is_eq(span,positions),input,channels,offset,span,positions)
    case Partition.Fork{left,right}:
      gather(Partition.Fork{left,right},Tensors.buffer(~F32,0.0,(rows * positions : U32)),positions)

# The deepest split that leaves every leaf at least one eight-position block.
def depth_limit(+padded: U32) -> U32:
  (U32.from_nat(Tensors.significant_bits(32n,(padded / 8 : U32),0n,U32.is_zero((padded / 8 : U32)))) - 1 : U32)

def run_packed(~custom: F32 -> F32,activation: Activations.Activation,depth: U32,work: Work,input: Storage.Buffer<F32>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Work{+rows,+h,w,k,s,p,ow,+positions,+inner} = work
  +padded = KernelWindow.padded(positions)
  +limit = depth_limit(padded)
  finish(partition(~custom,U32.to_nat(U32.min(depth,limit)),Task{Work{rows,h,w,k,s,p,ow,positions,inner},activation},0,padded,InputWindow{input,0,h},weights,bias),rows,positions)

def run(~custom: F32 -> F32,activation: Activations.Activation,depth: U32,+work: Work,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Work{rows,h,w,k,s,p,ow,positions,inner} = work
  run_packed(~custom,activation,depth,work,input,GemmBuffer.reorder(rows,inner,weights),bias)
