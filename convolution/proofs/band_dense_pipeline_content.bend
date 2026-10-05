# The actual dense pipeline's contents and logical length follow its storage
# model and allocator footprint. Readiness is constructed from the actual
# observed certificate and the existing allocation bound.
import Base
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/storage_buffer.bend as Storage
import ../../lib/math_activation.bend as Activations
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/kernel_pipeline_buffer_shape.bend as KernelPipelineBufferShape
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ./microkernel_inference_refinement.bend as Inference
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./band_leaf_execution.bend as LeafExecution
import ./band_dense_convolution.bend as DenseConvolution
import ./band_dense_pipeline.bend as DensePipeline

def length_projection(buffer: Storage.Buffer<F32>) ->
  {LeafExecution.buffer_length(~F32,buffer) == KernelPipelineBufferShape.length(KernelPipelineBufferShape.footprint(~F32,buffer)) : Nat}:
  Storage.Buffer{storage,length,Certificate.Certificate{depth,equation}} = buffer
  {==}

def run_length(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>) ->
  {LeafExecution.buffer_length(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) == U32.to_nat(DensePipeline.output_size(work)) : Nat}:
  Equal.trans(Nat,LeafExecution.buffer_length(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),
    KernelPipelineBufferShape.length(KernelPipelineBufferShape.footprint(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)))),U32.to_nat(DensePipeline.output_size(work)),
    length_projection(Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),
    Equal.cong(KernelPipelineBufferShape.Footprint,Nat,value => KernelPipelineBufferShape.length(value),KernelPipelineBufferShape.footprint(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),
      DensePipeline.allocated_footprint(DensePipeline.output_size(work)),DensePipeline.run_footprint(~custom,activation,depth,work,input,weights,bias)))

def run_contents(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) ==
    Buffer.Content{DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)),U32.to_nat(DensePipeline.output_size(work))} : Buffer.Content<F32>}:
  Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),
    Buffer.Content{Tree.reflect(F32,LeafExecution.storage(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)))),
      LeafExecution.buffer_length(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)))},
    Buffer.Content{DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)),U32.to_nat(DensePipeline.output_size(work))},
    LeafExecution.contents_parts(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),
    %Equal.sym(Nat,LeafExecution.buffer_length(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),U32.to_nat(DensePipeline.output_size(work)),run_length(~custom,activation,depth,work,input,weights,bias)) :
      {Buffer.Content{Tree.reflect(F32,LeafExecution.storage(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)))),_} ==
        Buffer.Content{DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)),U32.to_nat(DensePipeline.output_size(work))} : Buffer.Content<F32>}
    Equal.cong(Tree.Tree<F32>,Buffer.Content<F32>,tree => Buffer.Content{tree,U32.to_nat(DensePipeline.output_size(work))},
      Tree.reflect(F32,LeafExecution.storage(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)))),
      DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)),
      Equal.trans(Tree.Tree<F32>,Tree.reflect(F32,LeafExecution.storage(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)))),
        Tree.reflect(F32,Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)))),
        DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)),
        Equal.cong(Array<F32>,Tree.Tree<F32>,array => Tree.reflect(F32,array),
          LeafExecution.storage(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),
          Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))),DenseConvolution.run_refines_model(~custom,activation,depth,work,input,weights,bias)),
        StorageLaws.reflect_packed(F32,DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))))))

def ready_model(+count: U32,+model: Buffer.Model<F32>,+footprint: {KernelPipelineBufferShape.footprint(~F32,Buffer.pack(~F32,model)) == DensePipeline.allocated_footprint(count) : KernelPipelineBufferShape.Footprint},
  allocation: Depth.Ready<count>) -> Buffer.ContentReady<F32,Buffer.content(~F32,model)>:
  Depth.Ready{depth_ok,room} = allocation
  KernelPipelineBufferShape.content_ready(~F32,model,
    %Equal.sym(KernelPipelineBufferShape.Footprint,KernelPipelineBufferShape.footprint(~F32,Buffer.pack(~F32,model)),DensePipeline.allocated_footprint(count),footprint) : {Nat.is_le(KernelPipelineBufferShape.depth(_),30n) == True{} : Bool}
    depth_ok,
    %Equal.sym(KernelPipelineBufferShape.Footprint,KernelPipelineBufferShape.footprint(~F32,Buffer.pack(~F32,model)),DensePipeline.allocated_footprint(count),footprint) :
      {Nat.is_le(KernelPipelineBufferShape.length(_),U32.to_nat(U32.shln(1,KernelPipelineBufferShape.depth(_)))) == True{} : Bool}
    room)

def ready_observed(+count: U32,-buffer: Storage.Buffer<F32>,observation: Buffer.Observation<F32,buffer>,
  footprint: {KernelPipelineBufferShape.footprint(~F32,buffer) == DensePipeline.allocated_footprint(count) : KernelPipelineBufferShape.Footprint},allocation: Depth.Ready<count>) ->
  Buffer.ContentReady<F32,Buffer.contents(~F32,buffer)>:
  Buffer.Observation{+model,+equation} = observation
  %equation : Buffer.ContentReady<F32,Buffer.contents(~F32,_)>
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.content(~F32,model),Buffer.packed_contents(~F32,model)) : Buffer.ContentReady<F32,_>
  ready_model(count,model,
    %Equal.sym(Storage.Buffer<F32>,Buffer.pack(~F32,model),buffer,equation) : {KernelPipelineBufferShape.footprint(~F32,_) == DensePipeline.allocated_footprint(count) : KernelPipelineBufferShape.Footprint}
    footprint,allocation)

def run_ready(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,allocation: Depth.Ready<DensePipeline.output_size(work)>) ->
  Buffer.ContentReady<F32,Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)))>:
  ready_observed(DensePipeline.output_size(work),Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),
    Buffer.observe(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),DensePipeline.run_footprint(~custom,activation,depth,work,input,weights,bias),allocation)

def run_supported(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+depth: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  Buffer.ContentReady<F32,Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)))>:
  ConvSpec.Problem{ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding},+input_values,+weight_values,+bias_values} = problem
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  +fast = Support.fast(limits,problem,supported)
  +span_bound = {
    %Equal.sym(Nat,U32.to_nat(Padded.spatial(problem)),GatherCopyCoordinates.spatial(problem),Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),fast)) :
      {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    Padded.spatial_within_padded(limits,problem,Guards.valid(problem,valid),fast) : {Nat.is_le(U32.to_nat(Padded.spatial(problem)),Support.padded(problem)) == True{} : Bool}}
  run_ready(~custom,activation,depth,Inference.work(problem),input,weights,bias,
    LeafExecution.product_ready(limits,MicroLeaf.rows(problem),Padded.spatial(problem),DenseConvolution.leaf_room(limits,problem,Padded.spatial(problem),fast,span_bound)))
