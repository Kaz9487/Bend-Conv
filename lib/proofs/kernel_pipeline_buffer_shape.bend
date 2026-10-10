# Buffer footprint facts only: each output keeps its allocator's depth and
# logical length. The proof uses wrapper certificates, never MAC internals.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_spatial_packing.bend as Packing
import ../kernel_packing_buffer.bend as PackingBuffer
import ../kernel_window.bend as Window
import ./kernel_gemm_certificate.bend as GemmCertificate
import ../storage_buffer.bend as Storage
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../kernel_gather_buffer.bend as GatherBuffer
import ../kernel_pipeline.bend as Pipeline
import ../tensor_operations.bend as Operations
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../math_activation.bend as Activations
import ./storage_certificate.bend as Certificate
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./storage_allocation_refinement.bend as AllocationLaws
import ../storage_allocation.bend as Allocation

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
  Buffer.Sized<Element,model>:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  +model = {Buffer.Model{tree,length,Certificate.Certificate{depth,equation}} : Buffer.Model<Element>}
  Buffer.Sized{depth_ok,
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

def lanes(~Context: Data,~operation: Context -> F32 -> F32,+count: Nat,+position: U32,+context: Context,buffer: Storage.Buffer<F32>) ->
  {footprint(~F32,Operations.lanes(~Context,~operation,count,position,context,buffer)) == footprint(~F32,buffer) : Footprint}:
  Storage.Buffer{array,length,Certificate.Certificate{depth,equation}} = buffer
  {==}

def activated(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,buffer: Storage.Buffer<F32>) ->
  {footprint(~F32,Pipeline.activated(~custom,activation,count,buffer)) == footprint(~F32,buffer) : Footprint}:
  match activation:
    case Activations.Identity{}: {==}
    case Activations.ReLU{}: mapped(~Unit,~Pipeline.activated_value(~Activations.relu_value),count,Unit{},buffer)
    case Activations.ReLU6{}: mapped(~Unit,~Pipeline.activated_value(~Activations.relu6_value),count,Unit{},buffer)
    case Activations.LeakyReLU{slope}: mapped(~F32,~Pipeline.leaky_relu_value,count,slope,buffer)
    case Activations.Sigmoid{}: lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value),U32.to_nat(count),0,Unit{},buffer)
    case Activations.SiLU{}: lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value),U32.to_nat(count),0,Unit{},buffer)
    case Activations.Tanh{}: lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value),U32.to_nat(count),0,Unit{},buffer)
    case Activations.HardSigmoid{}: mapped(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value),count,Unit{},buffer)
    case Activations.HardSwish{}: mapped(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value),count,Unit{},buffer)
    case Activations.GELUTanh{}: lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_tanh_value),U32.to_nat(count),0,Unit{},buffer)
    case Activations.Custom{operations}: mapped(~Unit,~Pipeline.activated_value(~custom),count,Unit{},buffer)

def gathered(+channels: U32,+offset: U32,+positions: U32,+span: U32,input: Storage.Buffer<F32>,output: Storage.Buffer<F32>) ->
  {footprint(~F32,GatherBuffer.run(channels,offset,positions,span,input,output)) == footprint(~F32,output) : Footprint}:
  Storage.Buffer{input,il,ic} = input
  Storage.Buffer{output,length,Certificate.Certificate{depth,equation}} = output
  {==}

type BlockModel is Data:
  BlockModel{input: Buffer.Model<F32>,tiles: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<F32>,output: Buffer.Model<F32>}

def pack_blocks(model: BlockModel) -> Pipeline.BlockState:
  BlockModel{input,tiles,weights,bias,output} = model
  Pipeline.BlockState{Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Buffer.pack(~F32,output)}

type BlockObservation<-state: Pipeline.BlockState> is Data:
  BlockObservation{model: BlockModel,equation: {pack_blocks(model) == state : Pipeline.BlockState}}

def observed_blocks(-input: Storage.Buffer<F32>,-tiles: Storage.Buffer<Tile.Tile8>,-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>,-output: Storage.Buffer<F32>,
  input_observation: Buffer.Observation<F32,input>,tile_observation: Buffer.Observation<Tile.Tile8,tiles>,weight_observation: Buffer.Observation<Tile.Tile8,weights>,
  bias_observation: Buffer.Observation<F32,bias>,output_observation: Buffer.Observation<F32,output>) -> BlockObservation<Pipeline.BlockState{input,tiles,weights,bias,output}>:
  Buffer.Observation{+im,ie} = input_observation
  Buffer.Observation{+tm,te} = tile_observation
  Buffer.Observation{+wm,we} = weight_observation
  Buffer.Observation{+bm,be} = bias_observation
  Buffer.Observation{+om,oe} = output_observation
  BlockObservation{BlockModel{im,tm,wm,bm,om},
    %ie : {pack_blocks(BlockModel{im,tm,wm,bm,om}) == Pipeline.BlockState{_,tiles,weights,bias,output} : Pipeline.BlockState}
    %te : {pack_blocks(BlockModel{im,tm,wm,bm,om}) == Pipeline.BlockState{Buffer.pack(~F32,im),_,weights,bias,output} : Pipeline.BlockState}
    %we : {pack_blocks(BlockModel{im,tm,wm,bm,om}) == Pipeline.BlockState{Buffer.pack(~F32,im),Buffer.pack(~Tile.Tile8,tm),_,bias,output} : Pipeline.BlockState}
    %be : {pack_blocks(BlockModel{im,tm,wm,bm,om}) == Pipeline.BlockState{Buffer.pack(~F32,im),Buffer.pack(~Tile.Tile8,tm),Buffer.pack(~Tile.Tile8,wm),_,output} : Pipeline.BlockState}
    %oe : {pack_blocks(BlockModel{im,tm,wm,bm,om}) == Pipeline.BlockState{Buffer.pack(~F32,im),Buffer.pack(~Tile.Tile8,tm),Buffer.pack(~Tile.Tile8,wm),Buffer.pack(~F32,bm),_} : Pipeline.BlockState}
    {==}}

def observe_blocks(state: Pipeline.BlockState) -> BlockObservation<state>:
  Pipeline.BlockState{input,tiles,weights,bias,output} = state
  observed_blocks(input,tiles,weights,bias,output,Buffer.observe(~F32,input),Buffer.observe(~Tile.Tile8,tiles),Buffer.observe(~Tile.Tile8,weights),Buffer.observe(~F32,bias),Buffer.observe(~F32,output))

def output(state: Pipeline.BlockState) -> Storage.Buffer<F32>:
  Pipeline.BlockState{input,tiles,weights,bias,output} = state
  output

def output_depth(-state: Gemm.Buffers,certificates: GemmCertificate.Kept<state>) -> Nat:
  GemmCertificate.Kept{ic,wc,bc,Certificate.Certificate{depth,equation}} = certificates
  depth

def kept_footprint_projection(input: Storage.Buffer<F32>,state: Gemm.Buffers,+tl: Nat,+wl: Nat,+bl: Nat,+ol: Nat,+certificates: GemmCertificate.Kept<state>) ->
  {footprint(~F32,output(Pipeline.block_kept(input,GemmBuffer.kept(state,tl,wl,bl,ol,certificates)))) == Footprint{output_depth(state,certificates),ol} : Footprint}:
  Gemm.Buffers{tiles,weights,bias,output} = state
  GemmCertificate.Kept{ic,wc,bc,Certificate.Certificate{depth,equation}} = certificates
  {==}

def kept_footprint(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,target: Storage.Buffer<F32>,input: Storage.Buffer<F32>,tiles: Storage.Buffer<Tile.Tile8>) ->
  {footprint(~F32,output(Pipeline.block_packed(rows,positions,inner,start,span,weights,bias,target,(input,tiles)))) == footprint(~F32,target) : Footprint}:
  Storage.Buffer{w,wl,Certificate.Certificate{+wd,+we}} = weights
  Storage.Buffer{b,bl,Certificate.Certificate{+bd,+be}} = bias
  Storage.Buffer{o,ol,Certificate.Certificate{+od,+oe}} = target
  Storage.Buffer{t,tl,Certificate.Certificate{+td,+te}} = tiles
  kept_footprint_projection(input,Gemm.run_keeping(rows,t,w,b,o,span,inner,start,positions),tl,wl,bl,ol,
    GemmCertificate.kept(rows,t,w,b,o,span,inner,start,positions,Certificate.Certificate{td,te},Certificate.Certificate{wd,we},Certificate.Certificate{bd,be},Certificate.Certificate{od,oe}))

def packed_footprint(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,target: Storage.Buffer<F32>,pair: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) ->
  {footprint(~F32,output(Pipeline.block_packed(rows,positions,inner,start,span,weights,bias,target,pair))) == footprint(~F32,target) : Footprint}:
  (input,tiles) = pair
  kept_footprint(rows,positions,inner,start,span,weights,bias,target,input,tiles)

def step_footprint(+work: Pipeline.Work,+pack_start: U32,+start: U32,+span: U32,state: Pipeline.BlockState) ->
  {footprint(~F32,output(Pipeline.block_step(work,pack_start,start,span,state))) == footprint(~F32,output(state)) : Footprint}:
  Pipeline.Work{+rows,+h,+w,+k,+s,+p,+ow,+positions,+inner} = work
  Pipeline.BlockState{input,tiles,weights,bias,target} = state
  packed_footprint(rows,positions,inner,start,span,weights,bias,target,
    PackingBuffer.run(Packing.Call{U32.to_nat((span / 8 * inner : U32)),pack_start,h,w,k,s,p,ow,inner},input,tiles))

def blocks_footprint(+remaining: Nat,+work: Pipeline.Work,+pack_start: U32,+start: U32,+padded: U32,+span: U32,-state: Pipeline.BlockState,observation: BlockObservation<state>) ->
  {footprint(~F32,output(Pipeline.blocks(remaining,work,pack_start,start,padded,span,state))) == footprint(~F32,output(state)) : Footprint}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      BlockObservation{+model,equation} = observation
      %equation : {footprint(~F32,output(Pipeline.blocks(1n+rest,work,pack_start,start,padded,span,_))) == footprint(~F32,output(_)) : Footprint}
      -next = Pipeline.block_step(work,pack_start,start,U32.min(span,(padded - start : U32)),pack_blocks(model))
      Equal.trans(Footprint,footprint(~F32,output(Pipeline.blocks(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,next))),footprint(~F32,output(next)),footprint(~F32,output(pack_blocks(model))),
        blocks_footprint(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,next,observe_blocks(Pipeline.block_step(work,pack_start,start,U32.min(span,(padded - start : U32)),pack_blocks(model)))),
        step_footprint(work,pack_start,start,U32.min(span,(padded - start : U32)),pack_blocks(model)))

def finish_footprint(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,state: Pipeline.BlockState) ->
  {footprint(~F32,Pipeline.blocks_finished(~custom,activation,count,state)) == footprint(~F32,output(state)) : Footprint}:
  Pipeline.BlockState{input,tiles,weights,bias,target} = state
  activated(~custom,activation,count,target)

def output_size(work: Pipeline.Work) -> U32:
  Pipeline.Work{rows,h,w,k,s,p,ow,positions,inner} = work
  (rows * positions : U32)

def run_footprint(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {footprint(~F32,Pipeline.blocked(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    Footprint{Tensors.depth_for(output_size(work)),U32.to_nat(output_size(work))} : Footprint}:
  Pipeline.Work{+rows,h,w,k,s,p,ow,+positions,+inner} = work
  +padded = Window.padded(positions)
  +span = Pipeline.block_positions(inner)
  +remaining = U32.to_nat(((padded + span - 1) / span : U32))
  -state = {Pipeline.BlockState{Buffer.pack(~F32,input),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),U32.max(Pipeline.block_tiles(),inner)),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * positions : U32))} : Pipeline.BlockState}
  %allocated(~F32,0.0,(rows * positions : U32)) :
    {footprint(~F32,Pipeline.blocks_finished(~custom,activation,(rows * positions : U32),Pipeline.blocks(remaining,work,pack_start,0,padded,span,state))) == _ : Footprint}
  %blocks_footprint(remaining,work,pack_start,0,padded,span,state,observe_blocks(Pipeline.BlockState{Buffer.pack(~F32,input),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),U32.max(Pipeline.block_tiles(),inner)),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * positions : U32))})) :
    {footprint(~F32,Pipeline.blocks_finished(~custom,activation,(rows * positions : U32),Pipeline.blocks(remaining,work,pack_start,0,padded,span,state))) == _ : Footprint}
  finish_footprint(~custom,activation,(rows * positions : U32),Pipeline.blocks(remaining,work,pack_start,0,padded,span,
    Pipeline.BlockState{Buffer.pack(~F32,input),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),U32.max(Pipeline.block_tiles(),inner)),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * positions : U32))}))

# Lift the same footprint fact to an input owner before its contents are observed.
type BlockFrame is Data:
  BlockFrame{activation: Activations.Activation,work: Pipeline.Work,pack_start: U32,weights: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<F32>}

def BlockFootprint(~custom: F32 -> F32,frame: BlockFrame,-input: Storage.Buffer<F32>) -> Type:
  BlockFrame{activation,+work,pack_start,weights,bias} = frame
  {footprint(~F32,Pipeline.blocked(~custom,activation,work,pack_start,input,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    Footprint{Tensors.depth_for(output_size(work)),U32.to_nat(output_size(work))} : Footprint}

def blocked_canonical(~custom: F32 -> F32,+frame: BlockFrame,+input: Buffer.Model<F32>) -> BlockFootprint(~custom,frame,Buffer.pack(~F32,input)):
  BlockFrame{activation,work,pack_start,weights,bias} = frame
  run_footprint(~custom,activation,work,pack_start,input,weights,bias)

def blocked_owned(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,input: Storage.Buffer<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {footprint(~F32,Pipeline.blocked(~custom,activation,work,pack_start,input,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    Footprint{Tensors.depth_for(output_size(work)),U32.to_nat(output_size(work))} : Footprint}:
  +frame = {BlockFrame{activation,work,pack_start,weights,bias} : BlockFrame}
  Buffer.lift(~F32,~BlockFrame,~BlockFootprint(~custom),frame,input,Buffer.observe(~F32,input),model => blocked_canonical(~custom,frame,model))
