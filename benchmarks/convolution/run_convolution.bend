import Base
import ../../convolution/planning.bend as Planning
import ../../convolution/shape.bend as ConvShape
import ../../lib/kernel_planning.bend as KernelPlanning
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_array_pipeline.bend as ArrayPipeline
import ../../lib/kernel_weights.bend as Weights
import ../../lib/kernel_tile.bend as Tile
import ../../lib/tensor.bend as Tensors
import ../../lib/traversal_partition.bend as Partition

def Lab.param(i: U32) -> IO(U32):
  import "./io.c"
def Lab.reps() -> IO(U32):
  import "./io.c"
def Lab.load(i: U32, count: U32) -> IO(Array<F32>):
  import "./io.c"
def Lab.start() -> IO(Unit):
  import "./io.c"
def Lab.reordered(a: Array<Tile.Tile8>) -> IO(Array<Tile.Tile8>):
  import "./io.c"
def Lab.computed(a: Partition.Tree<&1,Array<F32>,U32>) -> IO(Partition.Tree<&1,Array<F32>,U32>):
  import "./io.c"
def Lab.finish(a: Array<F32>) -> IO(Array<F32>):
  import "./io.c"
def Lab.save(a: Array<F32>, count: U32) -> IO(Unit):
  import "./io.c"

# The same phases as ArrayPipeline.run, separated only by clock hooks: weight
# reorder, per-leaf pack plus six-row GEMM, then gather.
def run_depth(+input_channel_count: U32, +output_channel_count: U32, +input_height: U32, +input_width: U32, +kernel_size: U32, stride: U32, padding: U32, +output_height: U32, +output_width: U32, depth: U32) -> IO(Unit):
  +reduction_length = (input_channel_count * kernel_size * kernel_size : U32)
  +output_positions = (output_height * output_width : U32)
  +padded_positions = ((output_positions + 7) / 8 * 8 : U32)
  +limit = (U32.from_nat(Tensors.significant_bits(32n,(padded_positions / 8 : U32),0n,U32.is_zero((padded_positions / 8 : U32)))) - 1 : U32)
  do IO<Unit>:
    input : Array<F32> <- Lab.load(0,(input_channel_count * input_height * input_width : U32))
    weights : Array<F32> <- Lab.load(1,(output_channel_count * reduction_length : U32))
    bias : Array<F32> <- Lab.load(2,output_channel_count)
    Lab.start()
    reordered : Array<Tile.Tile8> <- Lab.reordered(Weights.run(output_channel_count,reduction_length,weights))
    result : Partition.Tree<&1,Array<F32>,U32> <- Lab.computed(ArrayPipeline.partition(U32.to_nat(U32.min(depth,limit)),
      Pipeline.Work{output_channel_count,input_height,input_width,kernel_size,stride,padding,output_width,output_positions,reduction_length},
      0,padded_positions,ArrayPipeline.InputWindow{input,0,input_height},reordered,bias))
    output : Array<F32> <- Lab.finish(ArrayPipeline.finish(result,output_channel_count,output_positions))
    Lab.save(output,(output_channel_count * output_positions : U32))

def run_plan(plan: KernelPlanning.Plan,shape: ConvShape.Shape) -> IO(Unit):
  KernelPlanning.Plan{packed,depth,workspace,time} = plan
  ConvShape.Shape{ci,co,h,w,k,s,p,oh,ow} = shape
  run_depth(ci,co,h,w,k,s,p,oh,ow,depth)

# Benchmark files provide exactly sized owners; the default unlimited workspace
# always admits packed depth zero. Worker count is a ceiling, never a depth.
def run(+ci: U32,+co: U32,+h: U32,+w: U32,+k: U32,s: U32,p: U32,oh: U32,ow: U32,workers: U32) -> IO(Unit):
  +shape = {ConvShape.Shape{ci,co,h,w,k,s,p,oh,ow} : ConvShape.Shape}
  run_plan(Planning.plan(shape,workers,None{},KernelPlanning.Owners{
    U32.from_nat(KernelPlanning.capacity((ci * h * w : U32))),
    U32.from_nat(KernelPlanning.capacity(co))}),shape)

def repeat(n: Nat, +input_channel_count: U32, +output_channel_count: U32, +input_height: U32, +input_width: U32, +kernel_size: U32, +stride: U32, +padding: U32, +output_height: U32, +output_width: U32, +workers: U32) -> IO(Unit):
  match n:
    case 0n:
      IO.pure(Unit,Unit{})
    case 1n+rest:
      do IO<Unit>:
        run(input_channel_count,output_channel_count,input_height,input_width,kernel_size,stride,padding,output_height,output_width,workers)
        repeat(rest,input_channel_count,output_channel_count,input_height,input_width,kernel_size,stride,padding,output_height,output_width,workers)

def main() -> IO(Unit):
  do IO<Unit>:
    +input_channel_count : U32 <- Lab.param(0)
    +output_channel_count : U32 <- Lab.param(1)
    +input_height : U32 <- Lab.param(2)
    +input_width : U32 <- Lab.param(3)
    +kernel_size : U32 <- Lab.param(4)
    stride : U32 <- Lab.param(5)
    padding : U32 <- Lab.param(6)
    +output_height : U32 <- Lab.param(7)
    +output_width : U32 <- Lab.param(8)
    workers : U32 <- Lab.param(9)
    reps : U32 <- Lab.reps()
    repeat(U32.to_nat(reps),input_channel_count,output_channel_count,input_height,input_width,kernel_size,stride,padding,output_height,output_width,workers)


