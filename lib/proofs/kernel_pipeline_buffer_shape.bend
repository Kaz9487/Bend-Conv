# Buffer footprint facts only: each output keeps its allocator's depth and
# logical length. The proof uses wrapper certificates, never MAC internals.
import Base
import ../storage_buffer.bend as Storage
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../kernel_gather_buffer.bend as GatherBuffer
import ../kernel_pipeline.bend as Pipeline
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../math_activation.bend as Activations
import ./storage_certificate.bend as Certificate
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./storage_allocation_refinement.bend as AllocationLaws
import ../storage_allocation.bend as Allocation
import ./tensor_band_range.bend as BandRange

type Footprint is Data:
  Footprint{depth: Nat,length: Nat}

def footprint(~Element: Data,buffer: Storage.Buffer<Element>) -> Footprint:
  Storage.Buffer{array,length,Certificate.Certificate{depth,equation}} = buffer
  Footprint{depth,length}

def depth(footprint: Footprint) -> Nat:
  Footprint{depth,length} = footprint
  depth

def length(footprint: Footprint) -> Nat:
  Footprint{depth,length} = footprint
  length

def model_ready(~Element: Data,+model: Buffer.Model<Element>,
  depth_ok: {Nat.is_le(depth(footprint(~Element,Buffer.pack(~Element,model))),30n) == True{} : Bool},
  room: {Nat.is_le(length(footprint(~Element,Buffer.pack(~Element,model))),U32.to_nat(U32.shln(1,depth(footprint(~Element,Buffer.pack(~Element,model)))))) == True{} : Bool}) ->
  Buffer.Ready<Element,model>:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  +model = {Buffer.Model{tree,length,Certificate.Certificate{depth,equation}} : Buffer.Model<Element>}
  Buffer.Ready{depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,tree),U32.shln(1,depth),Shape.capacity(Element,depth,tree,Buffer.balanced(~Element,model))) :
      {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
    room}

def content_ready(~Element: Data,+model: Buffer.Model<Element>,
  depth_ok: {Nat.is_le(depth(footprint(~Element,Buffer.pack(~Element,model))),30n) == True{} : Bool},
  room: {Nat.is_le(length(footprint(~Element,Buffer.pack(~Element,model))),U32.to_nat(U32.shln(1,depth(footprint(~Element,Buffer.pack(~Element,model)))))) == True{} : Bool}) ->
  Buffer.ContentReady<Element,Buffer.content(~Element,model)>:
  Buffer.content_ready(~Element,model,model_ready(~Element,model,depth_ok,room))

def allocated(~Element: Data,+fill: Element,+count: U32) ->
  {footprint(~Element,Tensors.buffer(~Element,fill,count)) == Footprint{Tensors.depth_for(count),U32.to_nat(count)} : Footprint}:
  %Equal.sym(Storage.Buffer<Element>,Allocation.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),
    AllocationLaws.dispatch_same(Element,Allocation.allocated(~Element),Tensors.depth_for(count),fill,U32.to_nat(count))) :
    {footprint(~Element,_) == Footprint{Tensors.depth_for(count),U32.to_nat(count)} : Footprint}
  {==}

def mapped(~Context: Data,~operation: Context -> F32 -> F32,+count: U32,+context: Context,buffer: Storage.Buffer<F32>) ->
  {footprint(~F32,Pipeline.mapped(~Context,~operation,count,context,buffer)) == footprint(~F32,buffer) : Footprint}:
  Storage.Buffer{array,length,Certificate.Certificate{depth,equation}} = buffer
  {==}

def activated(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,buffer: Storage.Buffer<F32>) ->
  {footprint(~F32,Pipeline.activated(~custom,activation,count,buffer)) == footprint(~F32,buffer) : Footprint}:
  match activation:
    case Activations.Identity{}: {==}
    case Activations.ReLU{}: mapped(~Unit,~Pipeline.activated_value(~Activations.relu_value),count,Unit{},buffer)
    case Activations.ReLU6{}: mapped(~Unit,~Pipeline.activated_value(~Activations.relu6_value),count,Unit{},buffer)
    case Activations.LeakyReLU{slope}: mapped(~F32,~Pipeline.leaky_relu_value,count,slope,buffer)
    case Activations.Sigmoid{}: mapped(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value),count,Unit{},buffer)
    case Activations.SiLU{}: mapped(~Unit,~Pipeline.activated_value(~Activations.silu_value),count,Unit{},buffer)
    case Activations.Tanh{}: mapped(~Unit,~Pipeline.activated_value(~Activations.tanh_value),count,Unit{},buffer)
    case Activations.HardSigmoid{}: mapped(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value),count,Unit{},buffer)
    case Activations.HardSwish{}: mapped(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value),count,Unit{},buffer)
    case Activations.GELUTanh{}: mapped(~Unit,~Pipeline.activated_value(~Activations.gelu_tanh_value),count,Unit{},buffer)
    case Activations.Custom{operations}: mapped(~Unit,~Pipeline.activated_value(~custom),count,Unit{},buffer)

def computed(+rows: U32,input: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<F32>,+span: U32,+inner: U32) ->
  {footprint(~F32,GemmBuffer.run(rows,input,weights,bias,output,span,inner)) == footprint(~F32,output) : Footprint}:
  Storage.Buffer{input,il,ic} = input
  Storage.Buffer{weights,wl,wc} = weights
  Storage.Buffer{bias,bl,bc} = bias
  Storage.Buffer{output,length,Certificate.Certificate{depth,equation}} = output
  {==}

def gathered(+channels: U32,+offset: U32,+positions: U32,+span: U32,input: Storage.Buffer<F32>,output: Storage.Buffer<F32>) ->
  {footprint(~F32,GatherBuffer.run(channels,offset,positions,span,input,output)) == footprint(~F32,output) : Footprint}:
  Storage.Buffer{input,il,ic} = input
  Storage.Buffer{output,length,Certificate.Certificate{depth,equation}} = output
  {==}

def compute_tiles(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>) ->
  {footprint(~F32,Pipeline.compute_tiles(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Buffer.pack(~Tile.Tile8,tiles))) == footprint(~F32,Tensors.buffer(~F32,0.0,(rows * span : U32))) : Footprint}:
  %Equal.sym(Footprint,footprint(~F32,Pipeline.activated(~custom,activation,(rows * span : U32),GemmBuffer.run(rows,Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * span : U32)),span,inner))),
    footprint(~F32,GemmBuffer.run(rows,Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * span : U32)),span,inner)),
    activated(~custom,activation,(rows * span : U32),GemmBuffer.run(rows,Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * span : U32)),span,inner))) :
    {_ == footprint(~F32,Tensors.buffer(~F32,0.0,(rows * span : U32))) : Footprint}
  computed(rows,Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * span : U32)),span,inner)

def finished(+identity: Bool,+rows: U32,+offset: U32,+span: U32,+positions: U32,+input: Buffer.Model<F32>,
  source_exact: {footprint(~F32,Buffer.pack(~F32,input)) == footprint(~F32,Tensors.buffer(~F32,0.0,(rows * span : U32))) : Footprint},
  aligned: {U32.is_eq(positions,span) == identity : Bool}) ->
  {footprint(~F32,Pipeline.finish_leaf(identity,Buffer.pack(~F32,input),rows,offset,span,positions)) == footprint(~F32,Tensors.buffer(~F32,0.0,(rows * positions : U32))) : Footprint}:
  match identity:
    case False{}: gathered(rows,offset,positions,span,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,(rows * positions : U32)))
    case True{}:
      %Equal.sym(U32,positions,span,BandRange.same_word(positions,span,aligned)) :
        {footprint(~F32,Buffer.pack(~F32,input)) == footprint(~F32,Tensors.buffer(~F32,0.0,(rows * _ : U32))) : Footprint}
      source_exact

# The packed pair is observed only after execution; its tile contents do not
# affect the depth or length of the newly allocated output owner.
type ComputeFrame is Data:
  ComputeFrame{activation: Activations.Activation,rows: U32,span: U32,inner: U32,weights: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<F32>}

def ComputeFootprint(~custom: F32 -> F32,frame: ComputeFrame,-tiles: Storage.Buffer<Tile.Tile8>) -> Type:
  ComputeFrame{activation,rows,span,inner,weights,bias} = frame
  {footprint(~F32,Pipeline.compute_tiles(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),tiles)) == footprint(~F32,Tensors.buffer(~F32,0.0,(rows * span : U32))) : Footprint}

def compute_frame(~custom: F32 -> F32,+frame: ComputeFrame,+tiles: Buffer.Model<Tile.Tile8>) -> ComputeFootprint(~custom,frame,Buffer.pack(~Tile.Tile8,tiles)):
  ComputeFrame{activation,rows,span,inner,weights,bias} = frame
  compute_tiles(~custom,activation,rows,span,inner,weights,bias,tiles)

def compute_owned(~custom: F32 -> F32,+frame: ComputeFrame,tiles: Storage.Buffer<Tile.Tile8>) -> ComputeFootprint(~custom,frame,tiles):
  Buffer.lift(~Tile.Tile8,~ComputeFrame,~ComputeFootprint(~custom),frame,tiles,Buffer.observe(~Tile.Tile8,tiles),model => compute_frame(~custom,frame,model))

def computed_owners(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  owners: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) ->
  {footprint(~F32,Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),owners)) == footprint(~F32,Tensors.buffer(~F32,0.0,(rows * span : U32))) : Footprint}:
  (input,tiles) = owners
  compute_owned(~custom,ComputeFrame{activation,rows,span,inner,weights,bias},tiles)

def finish_owned(+identity: Bool,+rows: U32,+offset: U32,+span: U32,+positions: U32,input: Storage.Buffer<F32>,
  source_exact: {footprint(~F32,input) == footprint(~F32,Tensors.buffer(~F32,0.0,(rows * span : U32))) : Footprint},
  aligned: {U32.is_eq(positions,span) == identity : Bool}) ->
  {footprint(~F32,Pipeline.finish_leaf(identity,input,rows,offset,span,positions)) == footprint(~F32,Tensors.buffer(~F32,0.0,(rows * positions : U32))) : Footprint}:
  match identity:
    case False{}: gathered(rows,offset,positions,span,input,Tensors.buffer(~F32,0.0,(rows * positions : U32)))
    case True{}:
      %Equal.sym(U32,positions,span,BandRange.same_word(positions,span,aligned)) :
        {footprint(~F32,input) == footprint(~F32,Tensors.buffer(~F32,0.0,(rows * _ : U32))) : Footprint}
      source_exact
