# The dense pipeline partition: split, footprint, finish and refinement.
import Base
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/storage_buffer.bend as Storage
import ./band_leaf_execution.bend as LeafExecution
import ../../lib/kernel_array_pipeline.bend as ArrayPipeline
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_activation.bend as Activations
import ../../lib/tensor.bend as Tensors
import ../../lib/traversal_partition.bend as Partition
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/proofs/storage_allocation_refinement.bend as AllocationLaws
import ../../lib/storage_clone.bend as Clone
import ../../lib/kernel_gemm_buffer.bend as GemmBuffer
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_packing_buffer.bend as PackingBuffer
import ../../lib/proofs/storage_clone_refinement.bend as Cloned
import ../../lib/storage_allocation.bend as Allocation
import ../../lib/kernel_gather.bend as Gather
import ../../lib/kernel_gather_buffer.bend as GatherBuffer
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_weights.bend as Weights
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/kernel_pipeline_buffer_shape.bend as KernelPipelineBufferShape
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_clone_certificate.bend as CertificateClone
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_access_certificate.bend as AccessCertificate
import ../../lib/proofs/tensor_band_range.bend as BandRange
import ../../lib/proofs/traversal_certificate.bend as TraversalCertificate
import ../../lib/storage_copy.bend as Copy
import ../../lib/tensor_access.bend as Access
import ../../lib/tensor_operations.bend as Operations
import ../../lib/tensor_view.bend as Views
import ../../lib/traversal_array.bend as Traversal

# Certificate erasure connects the certified dense leaf to the existing Array
# convolution leaf. Activation uses the same established pass over the leaf's
# values; no ordered MAC implementation is unfolded by these proof terms.
def storage_matches(~Element: Data,+model: Buffer.Model<Element>) ->
  {LeafExecution.storage(~Element,Buffer.pack(~Element,model)) == Tree.pack(Element,Buffer.tree(~Element,model)) : Array<Element>}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def project_window(window: Pipeline.InputWindow) -> ArrayPipeline.InputWindow:
  Pipeline.InputWindow{buffer,first,rows} = window
  ArrayPipeline.InputWindow{LeafExecution.storage(~F32,buffer),first,rows}

def project_outputs(outputs: Pipeline.Outputs()) -> Partition.Tree<&1,Array<F32>,U32>:
  match outputs:
    case Partition.Leaf{buffer,offset,span,rows}: Partition.Leaf{LeafExecution.storage(~F32,buffer),offset,span,rows}
    case Partition.Fork{left,right}: Partition.Fork{project_outputs(left),project_outputs(right)}

def activate_outputs(~custom: F32 -> F32,+activation: Activations.Activation,outputs: Partition.Tree<&1,Array<F32>,U32>) -> Partition.Tree<&1,Array<F32>,U32>:
  match outputs:
    case Partition.Leaf{array,offset,+span,+rows}: Partition.Leaf{LeafExecution.activation_array(~custom,activation,(rows * span : U32),array),offset,span,rows}
    case Partition.Fork{left,right}: Partition.Fork{activate_outputs(~custom,activation,left),activate_outputs(~custom,activation,right)}

def array_leaf(~custom: F32 -> F32,task: Pipeline.Task,offset: U32,span: U32,input: ArrayPipeline.InputWindow,weights: Array<Tile.Tile8>,bias: Array<F32>) -> Partition.Tree<&1,Array<F32>,U32>:
  Pipeline.Task{work,activation} = task
  activate_outputs(~custom,activation,ArrayPipeline.leaf(work,offset,span,input,weights,bias))

def packed_allocation(+call: Packing.Call,-input: Array<F32>,+count: U32) ->
  {LeafExecution.packed_array(call,input,count) == LeafExecution.second(Packing.run(call,input,Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))) : Array<Tile.Tile8>}:
  %AllocationLaws.array_same(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)) :
    {LeafExecution.second(Packing.run(call,input,_)) == LeafExecution.second(Packing.run(call,input,Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))) : Array<Tile.Tile8>}
  {==}

def computed_allocation(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,
  -tiles: Array<Tile.Tile8>,-weights: Array<Tile.Tile8>,-bias: Array<F32>) ->
  {LeafExecution.computed_array(~custom,activation,rows,span,inner,tiles,weights,bias) ==
    LeafExecution.activation_array(~custom,activation,(rows * span : U32),Gemm.run(rows,tiles,weights,bias,ArrayPipeline.allocated(rows,span),span,inner)) : Array<F32>}:
  %AllocationLaws.array_same(F32,Tensors.depth_for((rows * span : U32)),0.0) :
    {LeafExecution.activation_array(~custom,activation,(rows * span : U32),Gemm.run(rows,tiles,weights,bias,_,span,inner)) ==
      LeafExecution.activation_array(~custom,activation,(rows * span : U32),Gemm.run(rows,tiles,weights,bias,ArrayPipeline.allocated(rows,span),span,inner)) : Array<F32>}
  {==}

# ArrayPipeline.computed destructs an affine packing pair. Its projections alone do
# not expose that match, so introduce the real pair as a formal parameter.
def computed_pair(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,
  -weights: Array<Tile.Tile8>,-bias: Array<F32>,packed: Array<F32> & Array<Tile.Tile8>) ->
  {LeafExecution.computed_array(~custom,activation,rows,span,inner,LeafExecution.second(packed),weights,bias) ==
    LeafExecution.activation_array(~custom,activation,(rows * span : U32),ArrayPipeline.computed(rows,span,inner,weights,bias,packed)) : Array<F32>}:
  (input,tiles) = packed
  computed_allocation(~custom,activation,rows,span,inner,tiles,weights,bias)

def leaf_projection(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+offset: U32,+span: U32,+window_first: U32,+window_rows: U32,
  +input: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {project_outputs(Pipeline.leaf(~custom,Pipeline.Task{work,activation},offset,span,Pipeline.InputWindow{Buffer.pack(~F32,input),window_first,window_rows},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    array_leaf(~custom,Pipeline.Task{work,activation},offset,span,ArrayPipeline.InputWindow{LeafExecution.storage(~F32,Buffer.pack(~F32,input)),window_first,window_rows},LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) : Partition.Tree<&1,Array<F32>,U32>}:
  Pipeline.Work{+rows,h,+width,+kernel,+stride,+padding,+output_width,positions,+inner} = work
  +count = (span / 8 * inner : U32)
  +call = {Packing.Call{U32.to_nat(count),KernelWindow.start(offset,stride,output_width,window_first),window_rows,width,kernel,stride,padding,output_width,inner} : Packing.Call}
  %Equal.sym(Array<F32>,LeafExecution.storage(~F32,Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),
      PackingBuffer.run(call,Buffer.pack(~F32,input),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)))),
    LeafExecution.computed_array(~custom,activation,rows,span,inner,LeafExecution.packed_array(call,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),count),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))),
    LeafExecution.computed(~custom,activation,rows,span,inner,call,count,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) :
    {Partition.Leaf{_,offset,span,rows} == array_leaf(~custom,Pipeline.Task{Pipeline.Work{rows,h,width,kernel,stride,padding,output_width,positions,inner},activation},offset,span,
      ArrayPipeline.InputWindow{LeafExecution.storage(~F32,Buffer.pack(~F32,input)),window_first,window_rows},LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) : Partition.Tree<&1,Array<F32>,U32>}
  %Equal.sym(Array<Tile.Tile8>,LeafExecution.packed_array(call,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),count),
    LeafExecution.second(Packing.run(call,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))),packed_allocation(call,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),count)) :
    {Partition.Leaf{LeafExecution.computed_array(~custom,activation,rows,span,inner,_,LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))),offset,span,rows} ==
      array_leaf(~custom,Pipeline.Task{Pipeline.Work{rows,h,width,kernel,stride,padding,output_width,positions,inner},activation},offset,span,
        ArrayPipeline.InputWindow{LeafExecution.storage(~F32,Buffer.pack(~F32,input)),window_first,window_rows},LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) : Partition.Tree<&1,Array<F32>,U32>}
  Equal.cong(Array<F32>,Partition.Tree<&1,Array<F32>,U32>,array => Partition.Leaf{array,offset,span,rows},
    LeafExecution.computed_array(~custom,activation,rows,span,inner,LeafExecution.second(Packing.run(call,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))),
    LeafExecution.activation_array(~custom,activation,(rows * span : U32),ArrayPipeline.computed(rows,span,inner,LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias)),
      Packing.run(call,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))))),
    computed_pair(~custom,activation,rows,span,inner,LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias)),
      Packing.run(call,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))))

# The certified dense split has the same affine owners and copies as the
# established Array split. Certificates are erased through the actual returned
# owner pairs; no replacement certificate or caller bounds are assumed.
def project_buffers(~Element: Data,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Array<Element> & Array<Element>:
  (left,right) = owners
  (LeafExecution.storage(~Element,left),LeafExecution.storage(~Element,right))

def project_windows(windows: Pipeline.InputWindow & Pipeline.InputWindow) -> ArrayPipeline.InputWindow & ArrayPipeline.InputWindow:
  (left,right) = windows
  (project_window(left),project_window(right))

def clone_buffers(~Element: Data,+length: Nat,owners: Array<Element> & Array<Element>,certificates: CertificateClone.Copies<Element,owners>) ->
  {project_buffers(~Element,Clone.buffers(~Element,length,owners,certificates)) == owners : Array<Element> & Array<Element>}:
  (left,right) = owners
  CertificateClone.Copies{left_certificate,right_certificate} = certificates
  {==}

def clone_projection(~Element: Data,input: Storage.Buffer<Element>) ->
  {project_buffers(~Element,Clone.clone(~Element,input)) == Array.clone(Element,LeafExecution.storage(~Element,input)) : Array<Element> & Array<Element>}:
  Storage.Buffer{input,+length,Certificate.Certificate{+depth,+equation}} = input
  clone_buffers(~Element,length,Array.clone(Element,input),CertificateClone.clone(~Element,input,
    Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation}))

def operation_buffers(~Element: Data,+input_length: Nat,+output_length: Nat,owners: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),owners>) ->
  {project_buffers(~Element,Operations.pair_buffers(~Element,input_length,output_length,owners,certificates)) == owners : Array<Element> & Array<Element>}:
  (input,output) = owners
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  {==}

def copy_projection(+count: U32,+source: U32,+destination: U32,input: Storage.Buffer<F32>,output: Storage.Buffer<F32>) ->
  {project_buffers(~F32,Copy.range(~F32,count,source,destination,input,output)) ==
    Traversal.copy(~F32,U32.to_nat(count),source,destination,LeafExecution.storage(~F32,input),LeafExecution.storage(~F32,output)) : Array<F32> & Array<F32>}:
  match input output:
    case Storage.Buffer{input,+input_length,+input_certificate} Storage.Buffer{output,+output_length,+output_certificate}:
      +view = {Views.View{[Views.Axis{count,1}],source} : Views.View}
      operation_buffers(~F32,input_length,output_length,
        Access.copy_run(~F32,True{},U32.to_nat(count),view,destination,input,output),
        AccessCertificate.copy(~F32,True{},U32.to_nat(count),view,destination,input,output,input_certificate,output_certificate))

def copy_rows_projected(+remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+target_stride: U32,
  -owners: Storage.Buffer<F32> & Storage.Buffer<F32>,observation: Cloned.Observation<F32,owners>) ->
  {project_buffers(~F32,Pipeline.copy_rows(remaining,channel,count,source_stride,source_start,target_stride,owners)) ==
    KernelWindow.copy_planes(remaining,channel,count,source_stride,source_start,target_stride,project_buffers(~F32,owners)) : Array<F32> & Array<F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Cloned.Observation{+source,+target,equation} = observation
      %equation :
        {project_buffers(~F32,Pipeline.copy_rows(1n+rest,channel,count,source_stride,source_start,target_stride,_)) ==
          KernelWindow.copy_planes(1n+rest,channel,count,source_stride,source_start,target_stride,project_buffers(~F32,_)) : Array<F32> & Array<F32>}
      %copy_projection(count,(channel * source_stride + source_start : U32),(channel * target_stride : U32),Buffer.pack(~F32,source),Buffer.pack(~F32,target)) :
        {project_buffers(~F32,Pipeline.copy_rows(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,
          Copy.range(~F32,count,(channel * source_stride + source_start : U32),(channel * target_stride : U32),Buffer.pack(~F32,source),Buffer.pack(~F32,target)))) ==
          KernelWindow.copy_planes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,_) : Array<F32> & Array<F32>}
      copy_rows_projected(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,
        Copy.range(~F32,count,(channel * source_stride + source_start : U32),(channel * target_stride : U32),Buffer.pack(~F32,source),Buffer.pack(~F32,target)),
        Cloned.observe_pair(~F32,Copy.range(~F32,count,(channel * source_stride + source_start : U32),(channel * target_stride : U32),Buffer.pack(~F32,source),Buffer.pack(~F32,target))))

def copy_rows_projection(+remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+target_stride: U32,
  owners: Storage.Buffer<F32> & Storage.Buffer<F32>) ->
  {project_buffers(~F32,Pipeline.copy_rows(remaining,channel,count,source_stride,source_start,target_stride,owners)) ==
    KernelWindow.copy_planes(remaining,channel,count,source_stride,source_start,target_stride,project_buffers(~F32,owners)) : Array<F32> & Array<F32>}:
  copy_rows_projected(remaining,channel,count,source_stride,source_start,target_stride,owners,Cloned.observe_pair(~F32,owners))

def windows_projection(+first: U32,+rows: U32,+left_first: U32,+left_rows: U32,owners: Storage.Buffer<F32> & Storage.Buffer<F32>) ->
  {project_windows(Pipeline.windows(first,rows,left_first,left_rows,owners)) ==
    ArrayPipeline.windows(first,rows,left_first,left_rows,project_buffers(~F32,owners)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  (source,target) = owners
  {==}

def cloned_projection(+first: U32,+rows: U32,owners: Storage.Buffer<F32> & Storage.Buffer<F32>) ->
  {project_windows(Pipeline.cloned(first,rows,owners)) == ArrayPipeline.cloned(first,rows,project_buffers(~F32,owners)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  (left,right) = owners
  {==}

def allocated_projection(+count: U32) ->
  {LeafExecution.storage(~F32,Tensors.buffer(~F32,0.0,count)) == Allocation.array(~F32,Tensors.depth_for(count),0.0) : Array<F32>}:
  %Equal.sym(Storage.Buffer<F32>,Tensors.buffer(~F32,0.0,count),Storage.allocate(~F32,Tensors.depth_for(count),0.0,U32.to_nat(count)),LeafExecution.allocated(~F32,0.0,count)) :
    {LeafExecution.storage(~F32,_) == Allocation.array(~F32,Tensors.depth_for(count),0.0) : Array<F32>}
  Equal.sym(Array<F32>,Allocation.array(~F32,Tensors.depth_for(count),0.0),Array.new(F32,Tensors.depth_for(count),0.0),AllocationLaws.array_same(F32,Tensors.depth_for(count),0.0))

def copied_projection(+channels: U32,+width: U32,+first: U32,+rows: U32,+band: KernelWindow.Band,+input: Buffer.Model<F32>) ->
  {project_windows(Pipeline.copied(channels,width,first,rows,band,Buffer.pack(~F32,input))) ==
    ArrayPipeline.copied(channels,width,first,rows,band,LeafExecution.storage(~F32,Buffer.pack(~F32,input))) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  KernelWindow.Band{+left_first,+left_rows} = band
  +count = (channels * left_rows * width : U32)
  %allocated_projection(count) :
    {project_windows(Pipeline.copied(channels,width,first,rows,KernelWindow.Band{left_first,left_rows},Buffer.pack(~F32,input))) ==
      ArrayPipeline.windows(first,rows,left_first,left_rows,KernelWindow.copy_planes(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),
        ((left_first - first) * width : U32),(left_rows * width : U32),(LeafExecution.storage(~F32,Buffer.pack(~F32,input)),_))) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}
  Equal.trans(ArrayPipeline.InputWindow & ArrayPipeline.InputWindow,
    project_windows(Pipeline.copied(channels,width,first,rows,KernelWindow.Band{left_first,left_rows},Buffer.pack(~F32,input))),
    ArrayPipeline.windows(first,rows,left_first,left_rows,project_buffers(~F32,Pipeline.copy_rows(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),
      ((left_first - first) * width : U32),(left_rows * width : U32),(Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count))))),
    ArrayPipeline.windows(first,rows,left_first,left_rows,KernelWindow.copy_planes(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),
      ((left_first - first) * width : U32),(left_rows * width : U32),(LeafExecution.storage(~F32,Buffer.pack(~F32,input)),LeafExecution.storage(~F32,Tensors.buffer(~F32,0.0,count))))),
    windows_projection(first,rows,left_first,left_rows,Pipeline.copy_rows(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),
      ((left_first - first) * width : U32),(left_rows * width : U32),(Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count)))),
    Equal.cong(Array<F32> & Array<F32>,ArrayPipeline.InputWindow & ArrayPipeline.InputWindow,owners => ArrayPipeline.windows(first,rows,left_first,left_rows,owners),
      project_buffers(~F32,Pipeline.copy_rows(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),
        ((left_first - first) * width : U32),(left_rows * width : U32),(Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count)))),
      KernelWindow.copy_planes(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),
        ((left_first - first) * width : U32),(left_rows * width : U32),(LeafExecution.storage(~F32,Buffer.pack(~F32,input)),LeafExecution.storage(~F32,Tensors.buffer(~F32,0.0,count)))),
      copy_rows_projection(U32.to_nat(channels),0,(left_rows * width : U32),(rows * width : U32),((left_first - first) * width : U32),(left_rows * width : U32),
        (Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count)))))

def split_when_projection(+copy: Bool,+channels: U32,+width: U32,+first: U32,+rows: U32,+band: KernelWindow.Band,+input: Buffer.Model<F32>) ->
  {project_windows(Pipeline.split_when(copy,channels,width,first,rows,band,Buffer.pack(~F32,input))) ==
    ArrayPipeline.split_when(copy,channels,width,first,rows,band,LeafExecution.storage(~F32,Buffer.pack(~F32,input))) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  match copy:
    case False{}:
      %clone_projection(~F32,Buffer.pack(~F32,input)) :
        {project_windows(Pipeline.cloned(first,rows,Clone.clone(~F32,Buffer.pack(~F32,input)))) == ArrayPipeline.cloned(first,rows,_) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}
      cloned_projection(first,rows,Clone.clone(~F32,Buffer.pack(~F32,input)))
    case True{}: copied_projection(channels,width,first,rows,band,input)

def split_projection(+work: Pipeline.Work,+offset: U32,+half: U32,+first: U32,+rows: U32,+input: Buffer.Model<F32>) ->
  {project_windows(Pipeline.split(work,offset,half,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows})) ==
    ArrayPipeline.split(work,offset,half,ArrayPipeline.InputWindow{LeafExecution.storage(~F32,Buffer.pack(~F32,input)),first,rows}) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  Pipeline.Work{channels_out,+height,+width,+kernel,+stride,+padding,+output_width,positions,+inner} = work
  +band = {KernelWindow.band(first,rows,height,kernel,stride,padding,output_width,offset,half) : KernelWindow.Band}
  split_when_projection(U32.is_zero((output_width % 8 : U32)) && KernelWindow.covers(first,rows,height,kernel,stride,padding,output_width,offset,half,band),
    (inner / (kernel * kernel) : U32),width,first,rows,band,input)

# Whole partition projection follows the actual split and cloned owners once.
# The Array adapter keeps activation at its existing leaf boundary. Recursive
# calls observe returned certificates rather than assuming cloned identities.
type WindowModel is Data:
  WindowModel{buffer: Buffer.Model<F32>,first: U32,rows: U32}

def pack_window(+model: WindowModel) -> Pipeline.InputWindow:
  WindowModel{buffer,first,rows} = model
  Pipeline.InputWindow{Buffer.pack(~F32,buffer),first,rows}

type WindowObservation<-window: Pipeline.InputWindow> is Data:
  WindowObservation{model: WindowModel,equation: {pack_window(model) == window : Pipeline.InputWindow}}

def observed_window(-buffer: Storage.Buffer<F32>,observation: Buffer.Observation<F32,buffer>,+first: U32,+rows: U32) ->
  WindowObservation<Pipeline.InputWindow{buffer,first,rows}>:
  Buffer.Observation{+model,equation} = observation
  WindowObservation{WindowModel{model,first,rows},
    Equal.cong(Storage.Buffer<F32>,Pipeline.InputWindow,buffer => Pipeline.InputWindow{buffer,first,rows},Buffer.pack(~F32,model),buffer,equation)}

def observe_window(window: Pipeline.InputWindow) -> WindowObservation<window>:
  Pipeline.InputWindow{buffer,+first,+rows} = window
  observed_window(buffer,Buffer.observe(~F32,buffer),first,rows)

type WindowPairObservation<-windows: Pipeline.InputWindow & Pipeline.InputWindow> is Data:
  WindowPairObservation{left: WindowModel,right: WindowModel,
    equation: {(pack_window(left),pack_window(right)) == windows : Pipeline.InputWindow & Pipeline.InputWindow}}

def observed_window_pair(-left: Pipeline.InputWindow,-right: Pipeline.InputWindow,left_observation: WindowObservation<left>,right_observation: WindowObservation<right>) ->
  WindowPairObservation<(left,right)>:
  match left_observation right_observation:
    case WindowObservation{+left_model,left_equal} WindowObservation{+right_model,right_equal}:
      WindowPairObservation{left_model,right_model,
        %left_equal : {(pack_window(left_model),pack_window(right_model)) == (_,right) : Pipeline.InputWindow & Pipeline.InputWindow}
        %right_equal : {(pack_window(left_model),pack_window(right_model)) == (pack_window(left_model),_) : Pipeline.InputWindow & Pipeline.InputWindow}
        {==}}

def observe_window_pair(windows: Pipeline.InputWindow & Pipeline.InputWindow) -> WindowPairObservation<windows>:
  (left,right) = windows
  observed_window_pair(left,right,observe_window(left),observe_window(right))

def array_split(task: Pipeline.Task,offset: U32,half: U32,input: ArrayPipeline.InputWindow) -> ArrayPipeline.InputWindow & ArrayPipeline.InputWindow:
  Pipeline.Task{work,activation} = task
  ArrayPipeline.split(work,offset,half,input)

def array_partition(~custom: F32 -> F32,+depth: Nat,+task: Pipeline.Task,+offset: U32,+span: U32,input: ArrayPipeline.InputWindow,weights: Array<Tile.Tile8>,bias: Array<F32>) -> Partition.Tree<&1,Array<F32>,U32>:
  Partition.run(~ArrayPipeline.InputWindow,~Array<Tile.Tile8>,~Array<F32>,~Array<F32>,~Pipeline.Task,
    ~array_split,~Array.clone(Tile.Tile8),~Array.clone(F32),~array_leaf(~custom),depth,task,offset,span,input,weights,bias)

def native_children(~custom: F32 -> F32,+depth: Nat,+task: Pipeline.Task,+offset: U32,+span: U32,
  inputs: Pipeline.InputWindow & Pipeline.InputWindow,weights: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Pipeline.Outputs():
  +half = (span / 16 * 8 : U32)
  Storage.bind_pair(~Pipeline.InputWindow,~Pipeline.InputWindow,~Pipeline.Outputs(),inputs,left_input => right_input =>
    Storage.bind_pair(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~Pipeline.Outputs(),weights,left_weights => right_weights =>
      Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~Pipeline.Outputs(),bias,left_bias => right_bias =>
        left right = Pipeline.partition(~custom,depth,task,offset,half,left_input,left_weights,left_bias)
          Pipeline.partition(~custom,depth,task,(offset + half : U32),(span - half : U32),right_input,right_weights,right_bias)
        Partition.Fork{left,right})))

def array_children(~custom: F32 -> F32,+depth: Nat,+task: Pipeline.Task,+offset: U32,+span: U32,
  inputs: ArrayPipeline.InputWindow & ArrayPipeline.InputWindow,weights: Array<Tile.Tile8> & Array<Tile.Tile8>,bias: Array<F32> & Array<F32>) -> Partition.Tree<&1,Array<F32>,U32>:
  +half = (span / 16 * 8 : U32)
  Storage.bind_pair(~ArrayPipeline.InputWindow,~ArrayPipeline.InputWindow,~Partition.Tree<&1,Array<F32>,U32>,inputs,left_input => right_input =>
    Storage.bind_pair(~Array<Tile.Tile8>,~Array<Tile.Tile8>,~Partition.Tree<&1,Array<F32>,U32>,weights,left_weights => right_weights =>
      Storage.bind_pair(~Array<F32>,~Array<F32>,~Partition.Tree<&1,Array<F32>,U32>,bias,left_bias => right_bias =>
        left right = array_partition(~custom,depth,task,offset,half,left_input,left_weights,left_bias)
          array_partition(~custom,depth,task,(offset + half : U32),(span - half : U32),right_input,right_weights,right_bias)
        Partition.Fork{left,right})))

# One split, whatever lies below it: the last split calls its leaves directly
# and an earlier one recurses, and both equal the two child partitions.
def native_step(~custom: F32 -> F32,+rest: Nat,+task: Pipeline.Task,+offset: U32,+span: U32,
  -input: Pipeline.InputWindow,-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>) ->
  {Pipeline.partition(~custom,1n+rest,task,offset,span,input,weights,bias) ==
    native_children(~custom,rest,task,offset,span,Pipeline.split_task(task,offset,(span / 16 * 8 : U32),input),Clone.clone(~Tile.Tile8,weights),Clone.clone(~F32,bias)) : Pipeline.Outputs()}:
  match rest:
    case 0n: {==}
    case 1n+more: {==}

def array_step(~custom: F32 -> F32,+rest: Nat,+task: Pipeline.Task,+offset: U32,+span: U32,
  -input: ArrayPipeline.InputWindow,-weights: Array<Tile.Tile8>,-bias: Array<F32>) ->
  {array_partition(~custom,1n+rest,task,offset,span,input,weights,bias) ==
    array_children(~custom,rest,task,offset,span,array_split(task,offset,(span / 16 * 8 : U32),input),Array.clone(Tile.Tile8,weights),Array.clone(F32,bias)) : Partition.Tree<&1,Array<F32>,U32>}:
  match rest:
    case 0n: {==}
    case 1n+more: {==}

def children_projected(~custom: F32 -> F32,
  ~child_projection: @+child_depth: Nat -> @+child_activation: Activations.Activation -> @+child_work: Pipeline.Work -> @+child_offset: U32 -> @+child_span: U32 -> @+window: WindowModel -> @+weights: Buffer.Model<Tile.Tile8> -> @+bias: Buffer.Model<F32> ->
    {project_outputs(Pipeline.partition(~custom,child_depth,Pipeline.Task{child_work,child_activation},child_offset,child_span,pack_window(window),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
      array_partition(~custom,child_depth,Pipeline.Task{child_work,child_activation},child_offset,child_span,project_window(pack_window(window)),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) : Partition.Tree<&1,Array<F32>,U32>},
  +depth: Nat,+activation: Activations.Activation,+work: Pipeline.Work,+offset: U32,+span: U32,
  -inputs: Pipeline.InputWindow & Pipeline.InputWindow,-weights: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32> & Storage.Buffer<F32>,
  input_observation: WindowPairObservation<inputs>,weight_observation: Cloned.Observation<Tile.Tile8,weights>,bias_observation: Cloned.Observation<F32,bias>) ->
  {project_outputs(native_children(~custom,depth,Pipeline.Task{work,activation},offset,span,inputs,weights,bias)) ==
    array_children(~custom,depth,Pipeline.Task{work,activation},offset,span,project_windows(inputs),project_buffers(~Tile.Tile8,weights),project_buffers(~F32,bias)) : Partition.Tree<&1,Array<F32>,U32>}:
  match input_observation weight_observation bias_observation:
    case WindowPairObservation{+left_input,+right_input,input_equal} Cloned.Observation{+left_weights,+right_weights,weight_equal} Cloned.Observation{+left_bias,+right_bias,bias_equal}:
      +task = {Pipeline.Task{work,activation} : Pipeline.Task}
      %input_equal :
        {project_outputs(native_children(~custom,depth,task,offset,span,_,weights,bias)) ==
          array_children(~custom,depth,task,offset,span,project_windows(_),project_buffers(~Tile.Tile8,weights),project_buffers(~F32,bias)) : Partition.Tree<&1,Array<F32>,U32>}
      %weight_equal :
        {project_outputs(native_children(~custom,depth,task,offset,span,(pack_window(left_input),pack_window(right_input)),_,bias)) ==
          array_children(~custom,depth,task,offset,span,project_windows((pack_window(left_input),pack_window(right_input))),project_buffers(~Tile.Tile8,_),project_buffers(~F32,bias)) : Partition.Tree<&1,Array<F32>,U32>}
      %bias_equal :
        {project_outputs(native_children(~custom,depth,task,offset,span,(pack_window(left_input),pack_window(right_input)),(Buffer.pack(~Tile.Tile8,left_weights),Buffer.pack(~Tile.Tile8,right_weights)),_)) ==
          array_children(~custom,depth,task,offset,span,project_windows((pack_window(left_input),pack_window(right_input))),project_buffers(~Tile.Tile8,(Buffer.pack(~Tile.Tile8,left_weights),Buffer.pack(~Tile.Tile8,right_weights))),project_buffers(~F32,_)) : Partition.Tree<&1,Array<F32>,U32>}
      +half = (span / 16 * 8 : U32)
      %Equal.sym(Partition.Tree<&1,Array<F32>,U32>,
        project_outputs(Pipeline.partition(~custom,depth,task,offset,half,pack_window(left_input),Buffer.pack(~Tile.Tile8,left_weights),Buffer.pack(~F32,left_bias))),
        array_partition(~custom,depth,task,offset,half,project_window(pack_window(left_input)),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left_weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,left_bias))),
        child_projection(depth,activation,work,offset,half,left_input,left_weights,left_bias)) :
        {Partition.Fork{_,project_outputs(Pipeline.partition(~custom,depth,task,(offset + half : U32),(span - half : U32),pack_window(right_input),Buffer.pack(~Tile.Tile8,right_weights),Buffer.pack(~F32,right_bias)))} ==
          Partition.Fork{array_partition(~custom,depth,task,offset,half,project_window(pack_window(left_input)),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left_weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,left_bias))),
            array_partition(~custom,depth,task,(offset + half : U32),(span - half : U32),project_window(pack_window(right_input)),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,right_weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,right_bias)))} : Partition.Tree<&1,Array<F32>,U32>}
      Equal.cong(Partition.Tree<&1,Array<F32>,U32>,Partition.Tree<&1,Array<F32>,U32>,right =>
        Partition.Fork{array_partition(~custom,depth,task,offset,half,project_window(pack_window(left_input)),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left_weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,left_bias))),right},
        project_outputs(Pipeline.partition(~custom,depth,task,(offset + half : U32),(span - half : U32),pack_window(right_input),Buffer.pack(~Tile.Tile8,right_weights),Buffer.pack(~F32,right_bias))),
        array_partition(~custom,depth,task,(offset + half : U32),(span - half : U32),project_window(pack_window(right_input)),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,right_weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,right_bias))),
        child_projection(depth,activation,work,(offset + half : U32),(span - half : U32),right_input,right_weights,right_bias))

def partition_projection(~custom: F32 -> F32,+depth: Nat,+activation: Activations.Activation,+work: Pipeline.Work,+offset: U32,+span: U32,
  +window: WindowModel,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {project_outputs(Pipeline.partition(~custom,depth,Pipeline.Task{work,activation},offset,span,pack_window(window),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    array_partition(~custom,depth,Pipeline.Task{work,activation},offset,span,project_window(pack_window(window)),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) : Partition.Tree<&1,Array<F32>,U32>}:
  match depth window:
    case 0n WindowModel{+input,+first,+rows}: leaf_projection(~custom,activation,work,offset,span,first,rows,input,weights,bias)
    case 1n+ +rest WindowModel{+input,+first,+rows}:
      +task = {Pipeline.Task{work,activation} : Pipeline.Task}
      +half = (span / 16 * 8 : U32)
      %Equal.sym(Pipeline.Outputs(),
        Pipeline.partition(~custom,1n+rest,task,offset,span,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)),
        native_children(~custom,rest,task,offset,span,Pipeline.split(work,offset,half,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows}),Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),Clone.clone(~F32,Buffer.pack(~F32,bias))),
        native_step(~custom,rest,task,offset,span,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) :
        {project_outputs(_) ==
          array_partition(~custom,1n+rest,task,offset,span,project_window(Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows}),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) : Partition.Tree<&1,Array<F32>,U32>}
      %Equal.sym(Partition.Tree<&1,Array<F32>,U32>,
        array_partition(~custom,1n+rest,task,offset,span,project_window(Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows}),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))),
        array_children(~custom,rest,task,offset,span,array_split(task,offset,half,project_window(Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows})),
          Array.clone(Tile.Tile8,LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights))),Array.clone(F32,LeafExecution.storage(~F32,Buffer.pack(~F32,bias)))),
        array_step(~custom,rest,task,offset,span,project_window(Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows}),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias)))) :
        {project_outputs(native_children(~custom,rest,task,offset,span,Pipeline.split(work,offset,half,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows}),Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),Clone.clone(~F32,Buffer.pack(~F32,bias)))) == _ : Partition.Tree<&1,Array<F32>,U32>}
      %split_projection(work,offset,half,first,rows,input) :
        {project_outputs(native_children(~custom,rest,task,offset,span,Pipeline.split(work,offset,half,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows}),Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),Clone.clone(~F32,Buffer.pack(~F32,bias)))) ==
          array_children(~custom,rest,task,offset,span,_,Array.clone(Tile.Tile8,LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights))),Array.clone(F32,LeafExecution.storage(~F32,Buffer.pack(~F32,bias)))) : Partition.Tree<&1,Array<F32>,U32>}
      %clone_projection(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)) :
        {project_outputs(native_children(~custom,rest,task,offset,span,Pipeline.split(work,offset,half,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows}),Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),Clone.clone(~F32,Buffer.pack(~F32,bias)))) ==
          array_children(~custom,rest,task,offset,span,project_windows(Pipeline.split(work,offset,half,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows})),_,Array.clone(F32,LeafExecution.storage(~F32,Buffer.pack(~F32,bias)))) : Partition.Tree<&1,Array<F32>,U32>}
      %clone_projection(~F32,Buffer.pack(~F32,bias)) :
        {project_outputs(native_children(~custom,rest,task,offset,span,Pipeline.split(work,offset,half,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows}),Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),Clone.clone(~F32,Buffer.pack(~F32,bias)))) ==
          array_children(~custom,rest,task,offset,span,project_windows(Pipeline.split(work,offset,half,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows})),project_buffers(~Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights))),_) : Partition.Tree<&1,Array<F32>,U32>}
      children_projected(~custom,~partition_projection(~custom),rest,activation,work,offset,span,
        Pipeline.split(work,offset,half,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows}),Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),Clone.clone(~F32,Buffer.pack(~F32,bias)),
        observe_window_pair(Pipeline.split(work,offset,half,Pipeline.InputWindow{Buffer.pack(~F32,input),first,rows})),
        Cloned.observe_pair(~Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights))),Cloned.observe_pair(~F32,Clone.clone(~F32,Buffer.pack(~F32,bias))))

# Certificate erasure for the complete dense pipeline. The model retains every
# output's actual certificate so recursive gathers observe the real returned
# owner. Numerical convolution remains the established leaf's responsibility.
def ModelOutputs() -> Data:
  Partition.Tree<&2,Buffer.Model<F32>,U32>

def pack_outputs(outputs: ModelOutputs()) -> Pipeline.Outputs():
  match outputs:
    case Partition.Leaf{buffer,offset,span,rows}: Partition.Leaf{Buffer.pack(~F32,buffer),offset,span,rows}
    case Partition.Fork{left,right}: Partition.Fork{pack_outputs(left),pack_outputs(right)}

type OutputObservation<-outputs: Pipeline.Outputs()> is Data:
  OutputObservation{model: ModelOutputs(),equation: {pack_outputs(model) == outputs : Pipeline.Outputs()}}

def observed_leaf(-buffer: Storage.Buffer<F32>,observation: Buffer.Observation<F32,buffer>,+offset: U32,+span: U32,+rows: U32) ->
  OutputObservation<Partition.Leaf{buffer,offset,span,rows}>:
  Buffer.Observation{+model,equation} = observation
  OutputObservation{Partition.Leaf{model,offset,span,rows},
    Equal.cong(Storage.Buffer<F32>,Pipeline.Outputs(),buffer => Partition.Leaf{buffer,offset,span,rows},Buffer.pack(~F32,model),buffer,equation)}

def observed_fork(-left: Pipeline.Outputs(),-right: Pipeline.Outputs(),left_observation: OutputObservation<left>,right_observation: OutputObservation<right>) ->
  OutputObservation<Partition.Fork{left,right}>:
  match left_observation right_observation:
    case OutputObservation{+left_model,left_equal} OutputObservation{+right_model,right_equal}:
      OutputObservation{Partition.Fork{left_model,right_model},
        %left_equal : {Partition.Fork{pack_outputs(left_model),pack_outputs(right_model)} == Partition.Fork{_,right} : Pipeline.Outputs()}
        %right_equal : {Partition.Fork{pack_outputs(left_model),pack_outputs(right_model)} == Partition.Fork{pack_outputs(left_model),_} : Pipeline.Outputs()}
        {==}}

def observe_outputs(outputs: Pipeline.Outputs()) -> OutputObservation<outputs>:
  match outputs:
    case Partition.Leaf{buffer,+offset,+span,+rows}: observed_leaf(buffer,Buffer.observe(~F32,buffer),offset,span,rows)
    case Partition.Fork{left,right}: observed_fork(left,right,observe_outputs(left),observe_outputs(right))

def array_gather(partitions: Partition.Tree<&1,Array<F32>,U32>,output: Array<F32>,+positions: U32) -> Array<F32>:
  Partition.gather(~Array<F32>,~(rows => offset => positions => span => input => output => Gather.gather_block(rows,offset,positions,span,input,output)),partitions,output,positions)

def gather_leaf_projection(+rows: U32,+offset: U32,+positions: U32,+span: U32,input: Storage.Buffer<F32>,output: Storage.Buffer<F32>) ->
  {LeafExecution.storage(~F32,GatherBuffer.run(rows,offset,positions,span,input,output)) ==
    Gather.gather_block(rows,offset,positions,span,LeafExecution.storage(~F32,input),LeafExecution.storage(~F32,output)) : Array<F32>}:
  match input output:
    case Storage.Buffer{input,input_length,input_certificate} Storage.Buffer{output,output_length,output_certificate}: {==}

def gather_projected(+partitions: ModelOutputs(),+positions: U32,-output: Storage.Buffer<F32>,observation: Buffer.Observation<F32,output>) ->
  {LeafExecution.storage(~F32,Pipeline.gather(pack_outputs(partitions),output,positions)) ==
    array_gather(project_outputs(pack_outputs(partitions)),LeafExecution.storage(~F32,output),positions) : Array<F32>}:
  match partitions observation:
    case Partition.Leaf{+input,+offset,+span,+rows} Buffer.Observation{+output_model,equation}:
      %equation :
        {LeafExecution.storage(~F32,GatherBuffer.run(rows,offset,positions,span,Buffer.pack(~F32,input),_)) ==
          Gather.gather_block(rows,offset,positions,span,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),LeafExecution.storage(~F32,_)) : Array<F32>}
      gather_leaf_projection(rows,offset,positions,span,Buffer.pack(~F32,input),Buffer.pack(~F32,output_model))
    case Partition.Fork{+left,+right} Buffer.Observation{+output_model,equation}:
      %equation :
        {LeafExecution.storage(~F32,Pipeline.gather(pack_outputs(right),Pipeline.gather(pack_outputs(left),_,positions),positions)) ==
          array_gather(project_outputs(pack_outputs(right)),array_gather(project_outputs(pack_outputs(left)),LeafExecution.storage(~F32,_),positions),positions) : Array<F32>}
      %gather_projected(left,positions,Buffer.pack(~F32,output_model),Buffer.observe(~F32,Buffer.pack(~F32,output_model))) :
        {LeafExecution.storage(~F32,Pipeline.gather(pack_outputs(right),Pipeline.gather(pack_outputs(left),Buffer.pack(~F32,output_model),positions),positions)) ==
          array_gather(project_outputs(pack_outputs(right)),_,positions) : Array<F32>}
      gather_projected(right,positions,Pipeline.gather(pack_outputs(left),Buffer.pack(~F32,output_model),positions),
        Buffer.observe(~F32,Pipeline.gather(pack_outputs(left),Buffer.pack(~F32,output_model),positions)))

def finish_leaf_allocation(+identity: Bool,-input: Array<F32>,+rows: U32,+offset: U32,+span: U32,+positions: U32) ->
  {LeafExecution.finish_array(identity,input,rows,offset,span,positions) == ArrayPipeline.finish_leaf(identity,input,rows,offset,span,positions) : Array<F32>}:
  match identity:
    case True{}: {==}
    case False{}:
      %AllocationLaws.array_same(F32,Tensors.depth_for((rows * positions : U32)),0.0) :
        {Gather.gather_block(rows,offset,positions,span,input,_) == ArrayPipeline.finish_leaf(False{},input,rows,offset,span,positions) : Array<F32>}
      {==}

def finish_model(+partitions: ModelOutputs(),+rows: U32,+positions: U32) ->
  {LeafExecution.storage(~F32,Pipeline.finish(pack_outputs(partitions),rows,positions)) == ArrayPipeline.finish(project_outputs(pack_outputs(partitions)),rows,positions) : Array<F32>}:
  match partitions:
    case Partition.Leaf{+input,+offset,+span,+channels}:
      +identity = U32.is_zero(offset) && U32.is_eq(span,positions)
      Equal.trans(Array<F32>,LeafExecution.storage(~F32,Pipeline.finish_leaf(identity,Buffer.pack(~F32,input),channels,offset,span,positions)),
        LeafExecution.finish_array(identity,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),channels,offset,span,positions),
        ArrayPipeline.finish_leaf(identity,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),channels,offset,span,positions),
        LeafExecution.execution_finished(identity,Buffer.pack(~F32,input),channels,offset,span,positions),
        finish_leaf_allocation(identity,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),channels,offset,span,positions))
    case Partition.Fork{+left,+right}:
      +count = (rows * positions : U32)
      %allocated_projection(count) :
        {LeafExecution.storage(~F32,Pipeline.gather(pack_outputs(Partition.Fork{left,right}),Tensors.buffer(~F32,0.0,count),positions)) ==
          array_gather(project_outputs(pack_outputs(Partition.Fork{left,right})),_,positions) : Array<F32>}
      gather_projected(Partition.Fork{left,right},positions,Tensors.buffer(~F32,0.0,count),Buffer.observe(~F32,Tensors.buffer(~F32,0.0,count)))

def pipeline_finish_observed(-partitions: Pipeline.Outputs(),observation: OutputObservation<partitions>,+rows: U32,+positions: U32) ->
  {LeafExecution.storage(~F32,Pipeline.finish(partitions,rows,positions)) == ArrayPipeline.finish(project_outputs(partitions),rows,positions) : Array<F32>}:
  OutputObservation{+model,equation} = observation
  %equation : {LeafExecution.storage(~F32,Pipeline.finish(_,rows,positions)) == ArrayPipeline.finish(project_outputs(_),rows,positions) : Array<F32>}
  finish_model(model,rows,positions)

def finish_projection(partitions: Pipeline.Outputs(),+rows: U32,+positions: U32) ->
  {LeafExecution.storage(~F32,Pipeline.finish(partitions,rows,positions)) == ArrayPipeline.finish(project_outputs(partitions),rows,positions) : Array<F32>}:
  pipeline_finish_observed(partitions,observe_outputs(partitions),rows,positions)

def array_run_packed(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,input: Array<F32>,weights: Array<Tile.Tile8>,bias: Array<F32>) -> Array<F32>:
  Pipeline.Work{+rows,+height,width,kernel,stride,padding,output_width,+positions,inner} = work
  +padded = ((positions + 7) / 8 * 8 : U32)
  +limit = (U32.from_nat(Tensors.significant_bits(32n,(padded / 8 : U32),0n,U32.is_zero((padded / 8 : U32)))) - 1 : U32)
  ArrayPipeline.finish(array_partition(~custom,U32.to_nat(U32.min(depth,limit)),Pipeline.Task{Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner},activation},0,padded,ArrayPipeline.InputWindow{input,0,height},weights,bias),rows,positions)

def run_packed_projection(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {LeafExecution.storage(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    array_run_packed(~custom,activation,depth,work,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) : Array<F32>}:
  Pipeline.Work{+rows,+height,+width,+kernel,+stride,+padding,+output_width,+positions,+inner} = work
  +work = {Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner} : Pipeline.Work}
  +padded = ((positions + 7) / 8 * 8 : U32)
  +limit = (U32.from_nat(Tensors.significant_bits(32n,(padded / 8 : U32),0n,U32.is_zero((padded / 8 : U32)))) - 1 : U32)
  +actual_depth = U32.to_nat(U32.min(depth,limit))
  %partition_projection(~custom,actual_depth,activation,work,0,padded,WindowModel{input,0,height},weights,bias) :
    {LeafExecution.storage(~F32,Pipeline.finish(Pipeline.partition(~custom,actual_depth,Pipeline.Task{work,activation},0,padded,Pipeline.InputWindow{Buffer.pack(~F32,input),0,height},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)),rows,positions)) ==
      ArrayPipeline.finish(_,rows,positions) : Array<F32>}
  finish_projection(Pipeline.partition(~custom,actual_depth,Pipeline.Task{work,activation},0,padded,Pipeline.InputWindow{Buffer.pack(~F32,input),0,height},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)),rows,positions)

def reordered_projection(+rows: U32,+inner: U32,weights: Storage.Buffer<F32>) ->
  {LeafExecution.storage(~Tile.Tile8,GemmBuffer.reorder(rows,inner,weights)) == Weights.run(rows,inner,LeafExecution.storage(~F32,weights)) : Array<Tile.Tile8>}:
  Storage.Buffer{weights,length,certificate} = weights
  {==}

def finish_reordered_run(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  -weights: Storage.Buffer<Tile.Tile8>,observation: Buffer.Observation<Tile.Tile8,weights>) ->
  {LeafExecution.storage(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),weights,Buffer.pack(~F32,bias))) ==
    array_run_packed(~custom,activation,depth,work,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),LeafExecution.storage(~Tile.Tile8,weights),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) : Array<F32>}:
  Buffer.Observation{+model,equation} = observation
  %equation :
    {LeafExecution.storage(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),_,Buffer.pack(~F32,bias))) ==
      array_run_packed(~custom,activation,depth,work,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),LeafExecution.storage(~Tile.Tile8,_),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) : Array<F32>}
  run_packed_projection(~custom,activation,depth,work,input,model,bias)

def array_run(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,input: Array<F32>,weights: Array<F32>,bias: Array<F32>) -> Array<F32>:
  Pipeline.Work{+rows,height,width,kernel,stride,padding,output_width,positions,+inner} = work
  array_run_packed(~custom,activation,depth,Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner},input,Weights.run(rows,inner,weights),bias)

def run_projection(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>) ->
  {LeafExecution.storage(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) ==
    array_run(~custom,activation,depth,work,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),LeafExecution.storage(~F32,Buffer.pack(~F32,weights)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) : Array<F32>}:
  Pipeline.Work{+rows,+height,+width,+kernel,+stride,+padding,+output_width,+positions,+inner} = work
  +work = {Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner} : Pipeline.Work}
  %reordered_projection(rows,inner,Buffer.pack(~F32,weights)) :
    {LeafExecution.storage(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias))) ==
      array_run_packed(~custom,activation,depth,work,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),_,LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) : Array<F32>}
  finish_reordered_run(~custom,activation,depth,work,input,bias,GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights)),Buffer.observe(~Tile.Tile8,GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights))))

# Actual dense owners keep their allocation footprint through the partition and
# gather schedule. Leaf certificates and logical lengths are produced by the
# existing Buffer wrapper laws, independently of the convolution values.
def work_rows(work: Pipeline.Work) -> U32:
  Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner} = work
  rows

def output_size(work: Pipeline.Work) -> U32:
  Pipeline.Work{+rows,height,width,kernel,stride,padding,output_width,+positions,inner} = work
  (rows * positions : U32)

def allocated_footprint(+count: U32) -> KernelPipelineBufferShape.Footprint:
  KernelPipelineBufferShape.Footprint{Tensors.depth_for(count),U32.to_nat(count)}

def LeafFootprints(+rows: U32,outputs: Pipeline.Outputs()) -> Data:
  match outputs:
    case Partition.Leaf{buffer,offset,+span,+channels}:
      Shape.Both<{channels == rows : U32},{KernelPipelineBufferShape.footprint(~F32,buffer) == allocated_footprint((channels * span : U32)) : KernelPipelineBufferShape.Footprint}>
    case Partition.Fork{left,right}: Shape.Both<LeafFootprints(rows,left),LeafFootprints(rows,right)>

def leaf_models(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+offset: U32,+span: U32,
  input: Pipeline.InputWindow,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  LeafFootprints(work_rows(work),Pipeline.leaf(~custom,Pipeline.Task{work,activation},offset,span,input,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))):
  match work input:
    case Pipeline.Work{+rows,height,+width,+kernel,+stride,+padding,+output_width,positions,+inner} Pipeline.InputWindow{input,+first,+window_rows}:
      +count = (span / 8 * inner : U32)
      +call = {Packing.Call{U32.to_nat(count),KernelWindow.start(offset,stride,output_width,first),window_rows,width,kernel,stride,padding,output_width,inner} : Packing.Call}
      Shape.Both{{==},Equal.trans(KernelPipelineBufferShape.Footprint,
        KernelPipelineBufferShape.footprint(~F32,Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),PackingBuffer.run(call,input,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)))),
        KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,(rows * span : U32))),allocated_footprint((rows * span : U32)),
        KernelPipelineBufferShape.computed_owners(~custom,activation,rows,span,inner,weights,bias,PackingBuffer.run(call,input,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),KernelPipelineBufferShape.allocated(~F32,0.0,(rows * span : U32)))}

def leaf_observed(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+offset: U32,+span: U32,
  input: Pipeline.InputWindow,-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>,
  weight_observation: Buffer.Observation<Tile.Tile8,weights>,bias_observation: Buffer.Observation<F32,bias>) ->
  LeafFootprints(work_rows(work),Pipeline.leaf(~custom,Pipeline.Task{work,activation},offset,span,input,weights,bias)):
  match weight_observation bias_observation:
    case Buffer.Observation{+weight_model,weight_equal} Buffer.Observation{+bias_model,bias_equal}:
      %weight_equal : LeafFootprints(work_rows(work),Pipeline.leaf(~custom,Pipeline.Task{work,activation},offset,span,input,_,bias))
      %bias_equal : LeafFootprints(work_rows(work),Pipeline.leaf(~custom,Pipeline.Task{work,activation},offset,span,input,Buffer.pack(~Tile.Tile8,weight_model),_))
      leaf_models(~custom,activation,work,offset,span,input,weight_model,bias_model)

def leaf_footprint(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+offset: U32,+span: U32,
  input: Pipeline.InputWindow,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) ->
  LeafFootprints(work_rows(work),Pipeline.leaf(~custom,Pipeline.Task{work,activation},offset,span,input,weights,bias)):
  leaf_observed(~custom,activation,work,offset,span,input,weights,bias,Buffer.observe(~Tile.Tile8,weights),Buffer.observe(~F32,bias))

def children_footprints(~custom: F32 -> F32,
  ~child_proof: @+depth: Nat -> @+activation: Activations.Activation -> @+work: Pipeline.Work -> @+offset: U32 -> @+span: U32 ->
    @input: Pipeline.InputWindow -> @weights: Storage.Buffer<Tile.Tile8> -> @bias: Storage.Buffer<F32> ->
    LeafFootprints(work_rows(work),Pipeline.partition(~custom,depth,Pipeline.Task{work,activation},offset,span,input,weights,bias)),
  +depth: Nat,+activation: Activations.Activation,+work: Pipeline.Work,+offset: U32,+span: U32,
  inputs: Pipeline.InputWindow & Pipeline.InputWindow,weights: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32> & Storage.Buffer<F32>) ->
  LeafFootprints(work_rows(work),native_children(~custom,depth,Pipeline.Task{work,activation},offset,span,inputs,weights,bias)):
  (left_input,right_input) = inputs
  (left_weights,right_weights) = weights
  (left_bias,right_bias) = bias
  +half = (span / 16 * 8 : U32)
  Shape.Both{child_proof(depth,activation,work,offset,half,left_input,left_weights,left_bias),
    child_proof(depth,activation,work,(offset + half : U32),(span - half : U32),right_input,right_weights,right_bias)}

def partition_footprints(~custom: F32 -> F32,+depth: Nat,+activation: Activations.Activation,+work: Pipeline.Work,+offset: U32,+span: U32,
  input: Pipeline.InputWindow,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) ->
  LeafFootprints(work_rows(work),Pipeline.partition(~custom,depth,Pipeline.Task{work,activation},offset,span,input,weights,bias)):
  match depth:
    case 0n: leaf_footprint(~custom,activation,work,offset,span,input,weights,bias)
    case 1n+ +rest:
      +half = (span / 16 * 8 : U32)
      %Equal.sym(Pipeline.Outputs(),Pipeline.partition(~custom,1n+rest,Pipeline.Task{work,activation},offset,span,input,weights,bias),
        native_children(~custom,rest,Pipeline.Task{work,activation},offset,span,Pipeline.split(work,offset,half,input),Clone.clone(~Tile.Tile8,weights),Clone.clone(~F32,bias)),
        native_step(~custom,rest,Pipeline.Task{work,activation},offset,span,input,weights,bias)) : LeafFootprints(work_rows(work),_)
      children_footprints(~custom,~partition_footprints(~custom),rest,activation,work,offset,span,
        Pipeline.split(work,offset,half,input),Clone.clone(~Tile.Tile8,weights),Clone.clone(~F32,bias))

def gather_footprint(+partitions: ModelOutputs(),+positions: U32,-output: Storage.Buffer<F32>,observation: Buffer.Observation<F32,output>) ->
  {KernelPipelineBufferShape.footprint(~F32,Pipeline.gather(pack_outputs(partitions),output,positions)) == KernelPipelineBufferShape.footprint(~F32,output) : KernelPipelineBufferShape.Footprint}:
  match partitions observation:
    case Partition.Leaf{+input,+offset,+span,+rows} Buffer.Observation{+output_model,equation}:
      %equation : {KernelPipelineBufferShape.footprint(~F32,Pipeline.gather(pack_outputs(Partition.Leaf{input,offset,span,rows}),_,positions)) == KernelPipelineBufferShape.footprint(~F32,_) : KernelPipelineBufferShape.Footprint}
      KernelPipelineBufferShape.gathered(rows,offset,positions,span,Buffer.pack(~F32,input),Buffer.pack(~F32,output_model))
    case Partition.Fork{+left,+right} Buffer.Observation{+output_model,equation}:
      %equation : {KernelPipelineBufferShape.footprint(~F32,Pipeline.gather(pack_outputs(Partition.Fork{left,right}),_,positions)) == KernelPipelineBufferShape.footprint(~F32,_) : KernelPipelineBufferShape.Footprint}
      Equal.trans(KernelPipelineBufferShape.Footprint,
        KernelPipelineBufferShape.footprint(~F32,Pipeline.gather(pack_outputs(right),Pipeline.gather(pack_outputs(left),Buffer.pack(~F32,output_model),positions),positions)),
        KernelPipelineBufferShape.footprint(~F32,Pipeline.gather(pack_outputs(left),Buffer.pack(~F32,output_model),positions)),KernelPipelineBufferShape.footprint(~F32,Buffer.pack(~F32,output_model)),
        gather_footprint(right,positions,Pipeline.gather(pack_outputs(left),Buffer.pack(~F32,output_model),positions),Buffer.observe(~F32,Pipeline.gather(pack_outputs(left),Buffer.pack(~F32,output_model),positions))),
        gather_footprint(left,positions,Buffer.pack(~F32,output_model),Buffer.observe(~F32,Buffer.pack(~F32,output_model))))

def finish_leaf(+identity: Bool,+rows: U32,+positions: U32,+offset: U32,+span: U32,+channels: U32,+input: Buffer.Model<F32>,
  +channels_equal: {channels == rows : U32},+source_exact: {KernelPipelineBufferShape.footprint(~F32,Buffer.pack(~F32,input)) == allocated_footprint((channels * span : U32)) : KernelPipelineBufferShape.Footprint},
  +guard: {U32.is_zero(offset) && U32.is_eq(span,positions) == identity : Bool}) ->
  {KernelPipelineBufferShape.footprint(~F32,Pipeline.finish_leaf(identity,Buffer.pack(~F32,input),channels,offset,span,positions)) == allocated_footprint((rows * positions : U32)) : KernelPipelineBufferShape.Footprint}:
  match identity:
    case False{}:
      %channels_equal : {KernelPipelineBufferShape.footprint(~F32,Pipeline.finish_leaf(False{},Buffer.pack(~F32,input),channels,offset,span,positions)) == allocated_footprint((_ * positions : U32)) : KernelPipelineBufferShape.Footprint}
      Equal.trans(KernelPipelineBufferShape.Footprint,KernelPipelineBufferShape.footprint(~F32,Pipeline.finish_leaf(False{},Buffer.pack(~F32,input),channels,offset,span,positions)),
        KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,(channels * positions : U32))),allocated_footprint((channels * positions : U32)),
        KernelPipelineBufferShape.gathered(channels,offset,positions,span,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,(channels * positions : U32))),KernelPipelineBufferShape.allocated(~F32,0.0,(channels * positions : U32)))
    case True{}:
      %channels_equal : {KernelPipelineBufferShape.footprint(~F32,Buffer.pack(~F32,input)) == allocated_footprint((_ * positions : U32)) : KernelPipelineBufferShape.Footprint}
      %BandRange.same_word(span,positions,Boolean.right(U32.is_zero(offset),U32.is_eq(span,positions),guard)) :
        {KernelPipelineBufferShape.footprint(~F32,Buffer.pack(~F32,input)) == allocated_footprint((channels * _ : U32)) : KernelPipelineBufferShape.Footprint}
      source_exact

def finish_models(+rows: U32,+positions: U32,+partitions: ModelOutputs(),facts: LeafFootprints(rows,pack_outputs(partitions))) ->
  {KernelPipelineBufferShape.footprint(~F32,Pipeline.finish(pack_outputs(partitions),rows,positions)) == allocated_footprint((rows * positions : U32)) : KernelPipelineBufferShape.Footprint}:
  match partitions:
    case Partition.Leaf{+input,+offset,+span,+channels}:
      Shape.Both{channels_equal,source_exact} = facts
      finish_leaf(U32.is_zero(offset) && U32.is_eq(span,positions),rows,positions,offset,span,channels,input,channels_equal,source_exact,{==})
    case Partition.Fork{+left,+right}:
      +count = (rows * positions : U32)
      Equal.trans(KernelPipelineBufferShape.Footprint,KernelPipelineBufferShape.footprint(~F32,Pipeline.gather(pack_outputs(Partition.Fork{left,right}),Tensors.buffer(~F32,0.0,count),positions)),
        KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,count)),allocated_footprint(count),
        gather_footprint(Partition.Fork{left,right},positions,Tensors.buffer(~F32,0.0,count),Buffer.observe(~F32,Tensors.buffer(~F32,0.0,count))),KernelPipelineBufferShape.allocated(~F32,0.0,count))

def footprint_finish_observed(+rows: U32,+positions: U32,-partitions: Pipeline.Outputs(),observation: OutputObservation<partitions>,facts: LeafFootprints(rows,partitions)) ->
  {KernelPipelineBufferShape.footprint(~F32,Pipeline.finish(partitions,rows,positions)) == allocated_footprint((rows * positions : U32)) : KernelPipelineBufferShape.Footprint}:
  OutputObservation{+model,+equation} = observation
  %equation : {KernelPipelineBufferShape.footprint(~F32,Pipeline.finish(_,rows,positions)) == allocated_footprint((rows * positions : U32)) : KernelPipelineBufferShape.Footprint}
  finish_models(rows,positions,model,
    %Equal.sym(Pipeline.Outputs(),pack_outputs(model),partitions,equation) : LeafFootprints(rows,_)
    facts)

def run_packed_footprint(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {KernelPipelineBufferShape.footprint(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) == allocated_footprint(output_size(work)) : KernelPipelineBufferShape.Footprint}:
  Pipeline.Work{+rows,+height,+width,+kernel,+stride,+padding,+output_width,+positions,+inner} = work
  +work = {Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner} : Pipeline.Work}
  +padded = ((positions + 7) / 8 * 8 : U32)
  +limit = (U32.from_nat(Tensors.significant_bits(32n,(padded / 8 : U32),0n,U32.is_zero((padded / 8 : U32)))) - 1 : U32)
  +actual_depth = U32.to_nat(U32.min(depth,limit))
  footprint_finish_observed(rows,positions,Pipeline.partition(~custom,actual_depth,Pipeline.Task{work,activation},0,padded,Pipeline.InputWindow{Buffer.pack(~F32,input),0,height},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)),
    observe_outputs(Pipeline.partition(~custom,actual_depth,Pipeline.Task{work,activation},0,padded,Pipeline.InputWindow{Buffer.pack(~F32,input),0,height},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))),
    partition_footprints(~custom,actual_depth,activation,work,0,padded,Pipeline.InputWindow{Buffer.pack(~F32,input),0,height},Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)))

def footprint_reordered_run(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  -weights: Storage.Buffer<Tile.Tile8>,observation: Buffer.Observation<Tile.Tile8,weights>) ->
  {KernelPipelineBufferShape.footprint(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),weights,Buffer.pack(~F32,bias))) == allocated_footprint(output_size(work)) : KernelPipelineBufferShape.Footprint}:
  Buffer.Observation{+model,equation} = observation
  %equation : {KernelPipelineBufferShape.footprint(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),_,Buffer.pack(~F32,bias))) == allocated_footprint(output_size(work)) : KernelPipelineBufferShape.Footprint}
  run_packed_footprint(~custom,activation,depth,work,input,model,bias)

def run_footprint(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>) ->
  {KernelPipelineBufferShape.footprint(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) == allocated_footprint(output_size(work)) : KernelPipelineBufferShape.Footprint}:
  Pipeline.Work{+rows,+height,+width,+kernel,+stride,+padding,+output_width,+positions,+inner} = work
  +work = {Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner} : Pipeline.Work}
  footprint_reordered_run(~custom,activation,depth,work,input,bias,GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights)),Buffer.observe(~Tile.Tile8,GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights))))
