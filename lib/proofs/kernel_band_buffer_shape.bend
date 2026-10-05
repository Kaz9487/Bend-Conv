# Output owner shape is established at the convolution leaf boundary. No
# multiplication, accumulation or GEMM traversal theorem is used here.
import Base
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ../kernel_band_pipeline.bend as BandPipeline
import ../kernel_pipeline.bend as Pipeline
import ../kernel_spatial_packing.bend as Packing
import ../kernel_packing_buffer.bend as PackingBuffer
import ../kernel_window.bend as KernelWindow
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition
import ../tensor.bend as Tensors
import ../math_activation.bend as Activations
import ./kernel_pipeline_buffer_shape.bend as KernelPipelineBufferShape
import ./storage_buffer_model.bend as Buffer
import ./kernel_pipeline_allocation_ready.bend as KernelPipelineAllocationReady
import ./tensor_band_model.bend as Model
import ./tensor_band_window.bend as TensorBandWindow
import ./storage_certificate.bend as Certificate

def AllocatedLeaf(+count: U32,bands: Bands.Bands(~F32)) -> Type:
  match bands:
    case Partition.Leaf{buffer,first,rows,details}:
      {KernelPipelineBufferShape.footprint(~F32,buffer) == KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,count)) : KernelPipelineBufferShape.Footprint}
    case Partition.Fork{left,right}: Empty

type FinishFrame is Data:
  FinishFrame{activation: Activations.Activation,channels: U32,span: U32,inner: U32,positions: U32,weights: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<F32>}

def FinishFootprint(~custom: F32 -> F32,frame: FinishFrame,-tiles: Storage.Buffer<Tile.Tile8>) -> Type:
  FinishFrame{activation,+channels,+span,inner,+positions,weights,bias} = frame
  {KernelPipelineBufferShape.footprint(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),Pipeline.compute_tiles(~custom,activation,channels,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),tiles),channels,0,span,positions)) ==
    KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,(channels * positions : U32))) : KernelPipelineBufferShape.Footprint}

def finish_canonical(~custom: F32 -> F32,+frame: FinishFrame,+tiles: Buffer.Model<Tile.Tile8>) -> FinishFootprint(~custom,frame,Buffer.pack(~Tile.Tile8,tiles)):
  FinishFrame{activation,channels,span,inner,positions,weights,bias} = frame
  KernelPipelineBufferShape.finish_owned(U32.is_eq(positions,span),channels,0,span,positions,
    Pipeline.compute_tiles(~custom,activation,channels,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Buffer.pack(~Tile.Tile8,tiles)),
    KernelPipelineBufferShape.compute_tiles(~custom,activation,channels,span,inner,weights,bias,tiles),{==})

def finish_computed(~custom: F32 -> F32,+activation: Activations.Activation,+channels: U32,+span: U32,+inner: U32,+positions: U32,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  owners: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) ->
  {KernelPipelineBufferShape.footprint(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),Pipeline.computed(~custom,activation,channels,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),owners),channels,0,span,positions)) ==
    KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,(channels * positions : U32))) : KernelPipelineBufferShape.Footprint}:
  (input,tiles) = owners
  +frame = {FinishFrame{activation,channels,span,inner,positions,weights,bias} : FinishFrame}
  Buffer.lift(~Tile.Tile8,~FinishFrame,~FinishFootprint(~custom),frame,tiles,Buffer.observe(~Tile.Tile8,tiles),model => finish_canonical(~custom,frame,model))

def output_rows(window: BandWindow.RowWindow) -> U32:
  TensorBandWindow.output_rows(window)

def output_width(geometry: BandWindow.Geometry) -> U32:
  BandWindow.Geometry{input,kernel,stride,padding,height,width} = geometry
  width

def windowed(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: BandWindow.Geometry,+channels: U32,+inner: U32,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  core: Storage.Buffer<F32>,above: Storage.Buffer<F32>,below: Storage.Buffer<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  AllocatedLeaf((channels * (output_rows(window) * output_width(geometry) : U32) : U32),
    BandPipeline.windowed(~custom,activation,geometry,channels,inner,first,rows,window,core,above,below,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))):
  BandWindow.Geometry{Bands.Geometry{batch,+ci,+height,+width},+kernel,+stride,+padding,+oh,+ow} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+of,+out_rows},+above_range,+below_range,BandWindow.Rows{+wf,+wr}} = window
  +positions = (out_rows * ow : U32)
  +padded = ((positions + 7) / 8 * 8 : U32)
  +count = (padded / 8 * inner : U32)
  finish_computed(~custom,activation,channels,padded,inner,positions,weights,bias,
    PackingBuffer.run(Packing.Call{U32.to_nat(count),KernelWindow.start(((of - first / height * oh) * ow : U32),stride,ow,(wf - first / height * height : U32)),wr,width,kernel,stride,padding,ow,inner},
      BandWindow.assemble(~F32,0.0,Bands.Geometry{batch,ci,height,width},BandWindow.Rows{wf,wr},BandWindow.Rows{first,rows},above_range,below_range,core,above,below),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)))

def ReadyFromAllocation(count: U32,-buffer: Storage.Buffer<F32>) -> Type:
  @exact: {KernelPipelineBufferShape.footprint(~F32,buffer) == KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,count)) : KernelPipelineBufferShape.Footprint} ->
  @within: {U32.is_le(count,1073741824) == True{} : Bool} -> Buffer.ContentReady<F32,Buffer.contents(~F32,buffer)>

def ready_canonical(+count: U32,+model: Buffer.Model<F32>,
  exact: {KernelPipelineBufferShape.footprint(~F32,Buffer.pack(~F32,model)) == KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,count)) : KernelPipelineBufferShape.Footprint},
  within: {U32.is_le(count,1073741824) == True{} : Bool}) -> Buffer.ContentReady<F32,Buffer.contents(~F32,Buffer.pack(~F32,model))>:
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.content(~F32,model),Buffer.packed_contents(~F32,model)) : Buffer.ContentReady<F32,_>
  KernelPipelineAllocationReady.bounded(~F32,0.0,count,model,exact,within)

def ready_owned(+count: U32,buffer: Storage.Buffer<F32>,
  exact: {KernelPipelineBufferShape.footprint(~F32,buffer) == KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,count)) : KernelPipelineBufferShape.Footprint},
  within: {U32.is_le(count,1073741824) == True{} : Bool}) -> Buffer.ContentReady<F32,Buffer.contents(~F32,buffer)>:
  Buffer.lift(~F32,~U32,~ReadyFromAllocation,count,buffer,Buffer.observe(~F32,buffer),model => exact => within => ready_canonical(count,model,exact,within))(exact,within)

def ready_leaf(+count: U32,bands: Bands.Bands(~F32),exact: AllocatedLeaf(count,bands),within: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Model.ContentReady(~F32,Model.contents(~F32,bands)):
  match bands:
    case Partition.Leaf{buffer,first,rows,details}: ready_owned(count,buffer,exact,within)
    case Partition.Fork{left,right}: match exact:

def length_canonical(+count: U32,+model: Buffer.Model<F32>,
  exact: {KernelPipelineBufferShape.footprint(~F32,Buffer.pack(~F32,model)) == KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,count)) : KernelPipelineBufferShape.Footprint}) ->
  {Buffer.content_length(~F32,Buffer.contents(~F32,Buffer.pack(~F32,model))) == U32.to_nat(count) : Nat}:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  +model = {Buffer.Model{tree,length,Certificate.Certificate{depth,equation}} : Buffer.Model<F32>}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.content(~F32,model),Buffer.packed_contents(~F32,model)) :
    {Buffer.content_length(~F32,_) == U32.to_nat(count) : Nat}
  Equal.trans(Nat,length,KernelPipelineBufferShape.length(KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,count))),U32.to_nat(count),
    Equal.cong(KernelPipelineBufferShape.Footprint,Nat,footprint => KernelPipelineBufferShape.length(footprint),KernelPipelineBufferShape.footprint(~F32,Buffer.pack(~F32,model)),KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,count)),exact),
    Equal.cong(KernelPipelineBufferShape.Footprint,Nat,footprint => KernelPipelineBufferShape.length(footprint),KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,count)),KernelPipelineBufferShape.Footprint{Tensors.depth_for(count),U32.to_nat(count)},KernelPipelineBufferShape.allocated(~F32,0.0,count)))
