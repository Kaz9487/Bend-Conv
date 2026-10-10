# Erasing buffer certificates exposes the same packing and GEMM block loop.
import Base
import ../../lib/proofs/kernel_pipeline_activation.bend as ActivationProofs
import ../../lib/proofs/kernel_pipeline_buffer_shape.bend as BufferShape
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/kernel_packing_certificate.bend as PackingCertificate
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_gemm_buffer.bend as GemmBuffer
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_packing_buffer.bend as PackingBuffer
import ../../lib/kernel_tile.bend as Tile
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_window.bend as Window
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/kernel_gemm_certificate.bend as GemmCertificate
import ../../lib/proofs/traversal_certificate.bend as TraversalCertificate
import ../../lib/proofs/storage_certificate.bend as Certificate

type State is Type:
  State{input: Array<F32>,tiles: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,bias: Array<F32>,output: Array<F32>}

def project(state: Pipeline.BlockState) -> State:
  Pipeline.BlockState{input,tiles,weights,bias,output} = state
  State{ActivationProofs.storage(~F32,input),ActivationProofs.storage(~Tile.Tile8,tiles),ActivationProofs.storage(~Tile.Tile8,weights),ActivationProofs.storage(~F32,bias),ActivationProofs.storage(~F32,output)}

def kept(input: Array<F32>,state: Gemm.Buffers) -> State:
  Gemm.Buffers{tiles,weights,bias,output} = state
  State{input,tiles,weights,bias,output}

def kept_projection(input: Storage.Buffer<F32>,state: Gemm.Buffers,+input_length: Nat,+weight_length: Nat,+bias_length: Nat,+length: Nat,
  certificates: GemmCertificate.Kept<state>) ->
  {project(Pipeline.block_kept(input,GemmBuffer.kept(state,input_length,weight_length,bias_length,length,certificates))) == kept(ActivationProofs.storage(~F32,input),state) : State}:
  Gemm.Buffers{tiles,weights,bias,output} = state
  GemmCertificate.Kept{input_certificate,weight_certificate,bias_certificate,certificate} = certificates
  {==}

def packed(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,weights: Array<Tile.Tile8>,bias: Array<F32>,output: Array<F32>,pair: Array<F32> & Array<Tile.Tile8>) -> State:
  (input,tiles) = pair
  kept(input,Gemm.run_keeping(rows,tiles,weights,bias,output,span,inner,start,positions))

def packed_projection(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<F32>,input: Storage.Buffer<F32>,tiles: Storage.Buffer<Tile.Tile8>) ->
  {project(Pipeline.block_packed(rows,positions,inner,start,span,weights,bias,output,(input,tiles))) ==
    packed(rows,positions,inner,start,span,ActivationProofs.storage(~Tile.Tile8,weights),ActivationProofs.storage(~F32,bias),ActivationProofs.storage(~F32,output),(ActivationProofs.storage(~F32,input),ActivationProofs.storage(~Tile.Tile8,tiles))) : State}:
  Storage.Buffer{weights,wl,wc} = weights
  Storage.Buffer{bias,bl,bc} = bias
  Storage.Buffer{output,ol,oc} = output
  Storage.Buffer{tiles,tl,tc} = tiles
  kept_projection(input,Gemm.run_keeping(rows,tiles,weights,bias,output,span,inner,start,positions),tl,wl,bl,ol,
    GemmCertificate.kept(rows,tiles,weights,bias,output,span,inner,start,positions,tc,wc,bc,oc))

def packing_projection(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<F32>,
  +input_length: Nat,+tile_length: Nat,result: Array<F32> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),result>) ->
  {project(Pipeline.block_packed(rows,positions,inner,start,span,weights,bias,output,PackingBuffer.buffers(input_length,tile_length,result,certificates))) ==
    packed(rows,positions,inner,start,span,ActivationProofs.storage(~Tile.Tile8,weights),ActivationProofs.storage(~F32,bias),ActivationProofs.storage(~F32,output),result) : State}:
  (input,tiles) = result
  TraversalCertificate.Certificates{ic,tc} = certificates
  packed_projection(rows,positions,inner,start,span,weights,bias,output,Storage.Buffer{input,input_length,ic},Storage.Buffer{tiles,tile_length,tc})

def step(work: Pipeline.Work,+pack_start: U32,+start: U32,+span: U32,state: State) -> State:
  Pipeline.Work{rows,h,w,k,s,p,ow,positions,+inner} = work
  State{input,tiles,weights,bias,output} = state
  packed(rows,positions,inner,start,span,weights,bias,output,Packing.run(Packing.Call{U32.to_nat((span / 8 * inner : U32)),pack_start,h,w,k,s,p,ow,inner},input,tiles))

def step_projection(+work: Pipeline.Work,+pack_start: U32,+start: U32,+span: U32,state: Pipeline.BlockState) ->
  {project(Pipeline.block_step(work,pack_start,start,span,state)) == step(work,pack_start,start,span,project(state)) : State}:
  Pipeline.Work{+rows,+h,+w,+k,+s,+p,+ow,+positions,+inner} = work
  Pipeline.BlockState{Storage.Buffer{input,il,ic},Storage.Buffer{tiles,tl,tc},weights,bias,output} = state
  +call = {Packing.Call{U32.to_nat((span / 8 * inner : U32)),pack_start,h,w,k,s,p,ow,inner} : Packing.Call}
  packing_projection(rows,positions,inner,start,span,weights,bias,output,il,tl,Packing.run(call,input,tiles),PackingCertificate.certificates(call,input,tiles,ic,tc))

def blocks(remaining: Nat,+work: Pipeline.Work,+pack_start: U32,+start: U32,+padded: U32,+span: U32,state: State) -> State:
  match remaining:
    case 0n: state
    case 1n+rest:
      blocks(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,step(work,pack_start,start,U32.min(span,(padded - start : U32)),state))

def blocks_projection(+remaining: Nat,+work: Pipeline.Work,+pack_start: U32,+start: U32,+padded: U32,+span: U32,-state: Pipeline.BlockState,observation: BufferShape.BlockObservation<state>) ->
  {project(Pipeline.blocks(remaining,work,pack_start,start,padded,span,state)) == blocks(remaining,work,pack_start,start,padded,span,project(state)) : State}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      BufferShape.BlockObservation{+model,equation} = observation
      %equation : {project(Pipeline.blocks(1n+rest,work,pack_start,start,padded,span,_)) == blocks(1n+rest,work,pack_start,start,padded,span,project(_)) : State}
      %step_projection(work,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model)) :
        {project(Pipeline.blocks(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,Pipeline.block_step(work,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model)))) ==
          blocks(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,_) : State}
      blocks_projection(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,Pipeline.block_step(work,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model)),
        BufferShape.observe_blocks(Pipeline.block_step(work,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model))))

def finish(~custom: F32 -> F32,activation: Activations.Activation,count: U32,state: State) -> Array<F32>:
  State{input,tiles,weights,bias,output} = state
  ActivationProofs.activation_array(~custom,activation,count,output)

def finish_projection(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,state: Pipeline.BlockState) ->
  {ActivationProofs.storage(~F32,Pipeline.blocks_finished(~custom,activation,count,state)) == finish(~custom,activation,count,project(state)) : Array<F32>}:
  Pipeline.BlockState{input,tiles,weights,bias,output} = state
  ActivationProofs.activated(~custom,activation,count,output)

def allocated(~Element: Data,fill: Element,count: U32) -> Array<Element>:
  ActivationProofs.storage(~Element,Tensors.buffer(~Element,fill,count))

def run(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,input: Array<F32>,weights: Array<Tile.Tile8>,bias: Array<F32>) -> Array<F32>:
  Pipeline.Work{+rows,h,w,k,s,p,ow,+positions,+inner} = work
  +padded = Window.padded(positions)
  +span = Pipeline.block_positions(inner)
  finish(~custom,activation,(rows * positions : U32),blocks(U32.to_nat(((padded + span - 1) / span : U32)),work,pack_start,0,padded,span,
    State{input,allocated(~Tile.Tile8,Tile.filled(0.0),U32.max(Pipeline.block_tiles(),inner)),weights,bias,allocated(~F32,0.0,(rows * positions : U32))}))

def run_projection(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {ActivationProofs.storage(~F32,Pipeline.blocked(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) == run(~custom,activation,work,pack_start,ActivationProofs.storage(~F32,Buffer.pack(~F32,input)),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),ActivationProofs.storage(~F32,Buffer.pack(~F32,bias))) : Array<F32>}:
  Pipeline.Work{+rows,h,w,k,s,p,ow,+positions,+inner} = work
  +padded = Window.padded(positions)
  +span = Pipeline.block_positions(inner)
  +remaining = U32.to_nat(((padded + span - 1) / span : U32))
  -state = {Pipeline.BlockState{Buffer.pack(~F32,input),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),U32.max(Pipeline.block_tiles(),inner)),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * positions : U32))} : Pipeline.BlockState}
  %blocks_projection(remaining,work,pack_start,0,padded,span,state,BufferShape.observe_blocks(Pipeline.BlockState{Buffer.pack(~F32,input),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),U32.max(Pipeline.block_tiles(),inner)),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * positions : U32))})) :
    {ActivationProofs.storage(~F32,Pipeline.blocks_finished(~custom,activation,(rows * positions : U32),Pipeline.blocks(remaining,work,pack_start,0,padded,span,state))) == finish(~custom,activation,(rows * positions : U32),_) : Array<F32>}
  finish_projection(~custom,activation,(rows * positions : U32),Pipeline.blocks(remaining,work,pack_start,0,padded,span,Pipeline.BlockState{Buffer.pack(~F32,input),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),U32.max(Pipeline.block_tiles(),inner)),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * positions : U32))}))
