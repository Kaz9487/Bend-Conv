# Sources of the dense callers: image copies, packed images, weight owners and
# gather allocation.
import Base
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/storage_buffer.bend as Storage
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ./band_dense_convolution.bend as DenseConvolution
import ./band_dense_pipeline.bend as DensePipeline
import ./band_window_contents.bend as WindowContents
import ../../lib/proofs/tensor_dense_contract.bend as Dense
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./band_dense_batch.bend as Batch
import ./band_dense_single.bend as DenseSingle
import ./band_leaf_execution.bend as LeafExecution
import ./concrete_support_guard.bend as Support
import ./convolution_guard_facts.bend as Guards
import ../specification.bend as ConvSpec
import ../../lib/kernel_gemm_buffer.bend as GemmBuffer
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/storage_allocation_refinement.bend as AllocationLaws
import ../../lib/proofs/storage_clone_refinement.bend as Cloned
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/storage_copy.bend as Copy
import ../../lib/tensor_band.bend as Bands
import ../inference.bend as NativeConv
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_dense_pipeline_content.bend as Content
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ./microkernel_inference_refinement.bend as Inference
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./reduction_coordinate_bounds.bend as Coordinates
import ./supported_padded_coordinates.bend as Padded
import ../../lib/kernel_weights.bend as Weights
import ../../lib/proofs/kernel_pipeline_buffer_shape.bend as KernelPipelineBufferShape
import ../../lib/proofs/kernel_weights_buffer_refinement.bend as KernelWeightsBufferRefinement
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/storage_copy_refinement.bend as StorageCopyLaws
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/tensor_band_dense.bend as BandDense
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/word_count_flags.bend as Flags
import ../../lib/storage_clone.bend as Clone

# A batch image copy retains the original source owner and constructs the
# destination prefix from the source tensor's finite logical values.
def source_owner(owners: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  (source,output) = owners
  source

def source_content(owners: StorageCopyLaws.Owners<F32>) -> Buffer.Content<F32>:
  StorageCopyLaws.Owners{source,source_length,output,output_length} = owners
  Buffer.Content{source,source_length}

def source_projection(owners: Storage.Buffer<F32> & Storage.Buffer<F32>) ->
  {Buffer.contents(~F32,source_owner(owners)) == source_content(StorageCopyLaws.observe(~F32,owners)) : Buffer.Content<F32>}:
  (Storage.Buffer{source,source_length,source_certificate},Storage.Buffer{output,output_length,output_certificate}) = owners
  {==}

def source_contents(+count: U32,+source: U32,+destination: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,source_owner(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))) == Buffer.content(~F32,input) : Buffer.Content<F32>}:
  Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,source_owner(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),
    source_content(StorageCopyLaws.observe(~F32,Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),Buffer.content(~F32,input),
    source_projection(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output))),
    Equal.trans(Buffer.Content<F32>,source_content(StorageCopyLaws.observe(~F32,Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),
      source_content(StorageCopyLaws.model(~F32,count,source,destination,StorageCopyLaws.observe(~F32,(Buffer.pack(~F32,input),Buffer.pack(~F32,output))))),Buffer.content(~F32,input),
      Equal.cong(StorageCopyLaws.Owners<F32>,Buffer.Content<F32>,source_content,
        StorageCopyLaws.observe(~F32,Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output))),
        StorageCopyLaws.model(~F32,count,source,destination,StorageCopyLaws.observe(~F32,(Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),
        StorageCopyLaws.range(~F32,count,source,destination,(Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),
      %Equal.sym(StorageCopyLaws.Owners<F32>,StorageCopyLaws.observe(~F32,(Buffer.pack(~F32,input),Buffer.pack(~F32,output))),
        StorageCopyLaws.Owners{Buffer.tree(~F32,input),Buffer.length(~F32,input),Buffer.tree(~F32,output),Buffer.length(~F32,output)},Batch.packed_owners(input,output)) :
        {source_content(StorageCopyLaws.model(~F32,count,source,destination,_)) == Buffer.content(~F32,input) : Buffer.Content<F32>}
      Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,input),Buffer.Content{Buffer.tree(~F32,input),Buffer.length(~F32,input)},Batch.content_parts(input))))

def observed_source_contents(+count: U32,+source: U32,+destination: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,
  equation: {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)) : Storage.Buffer<F32> & Storage.Buffer<F32>}) ->
  {Buffer.content(~F32,left) == Buffer.content(~F32,input) : Buffer.Content<F32>}:
  Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,left),Buffer.contents(~F32,source_owner((Buffer.pack(~F32,left),Buffer.pack(~F32,right)))),Buffer.content(~F32,input),
    Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,left)),Buffer.content(~F32,left),Buffer.packed_contents(~F32,left)),
    Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,source_owner((Buffer.pack(~F32,left),Buffer.pack(~F32,right)))),
      Buffer.contents(~F32,source_owner(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),Buffer.content(~F32,input),
      Equal.cong(Storage.Buffer<F32> & Storage.Buffer<F32>,Buffer.Content<F32>,owners => Buffer.contents(~F32,source_owner(owners)),
        (Buffer.pack(~F32,left),Buffer.pack(~F32,right)),Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)),equation),
      source_contents(count,source,destination,input,output)))

def observed_output_contents(+count: U32,+source: U32,+destination: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,
  equation: {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)) : Storage.Buffer<F32> & Storage.Buffer<F32>}) ->
  {Buffer.content(~F32,right) == Batch.copied_content(count,source,destination,input,output) : Buffer.Content<F32>}:
  Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,right),Buffer.contents(~F32,TensorConvolution.copied_output((Buffer.pack(~F32,left),Buffer.pack(~F32,right)))),Batch.copied_content(count,source,destination,input,output),
    Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,right)),Buffer.content(~F32,right),Buffer.packed_contents(~F32,right)),
    Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,TensorConvolution.copied_output((Buffer.pack(~F32,left),Buffer.pack(~F32,right)))),
      Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),Batch.copied_content(count,source,destination,input,output),
      Equal.cong(Storage.Buffer<F32> & Storage.Buffer<F32>,Buffer.Content<F32>,owners => Buffer.contents(~F32,TensorConvolution.copied_output(owners)),
        (Buffer.pack(~F32,left),Buffer.pack(~F32,right)),Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)),equation),
      Batch.copy_output_contents(count,source,destination,input,output)))

type Copied<-count: U32,-source: U32,-destination: U32,-input: Buffer.Model<F32>,-output: Buffer.Model<F32>> is Data:
  Copied{source_model: Buffer.Model<F32>,output_model: Buffer.Model<F32>,
    owners: {(Buffer.pack(~F32,source_model),Buffer.pack(~F32,output_model)) == Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
    retained: {Buffer.content(~F32,source_model) == Buffer.content(~F32,input) : Buffer.Content<F32>},
    written: {Buffer.content(~F32,output_model) == Batch.copied_content(count,source,destination,input,output) : Buffer.Content<F32>},
    source_ready: Buffer.Ready<F32,source_model>,output_ready: Buffer.Ready<F32,output_model>}

def copied_observed(+count: U32,+source: U32,+destination: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+input_ready: Buffer.Ready<F32,input>,+output_ready: Buffer.Ready<F32,output>,
  observation: Cloned.Observation<F32,Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output))>) -> Copied<count,source,destination,input,output>:
  Cloned.Observation{+left,+right,+equation} = observation
  +retained = observed_source_contents(count,source,destination,input,output,left,right,equation)
  +written = observed_output_contents(count,source,destination,input,output,left,right,equation)
  +actual_output = Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,right),Batch.copied_content(count,source,destination,input,output),
    Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),written,
    Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),Batch.copied_content(count,source,destination,input,output),Batch.copy_output_contents(count,source,destination,input,output)))
  Copied{left,right,equation,retained,written,
    Buffer.ready_content(~F32,left,
      %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,left),Buffer.content(~F32,input),retained) : Buffer.ContentReady<F32,_>
      Buffer.content_ready(~F32,input,input_ready)),
    Buffer.ready_content(~F32,right,
      %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,right),Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),actual_output) : Buffer.ContentReady<F32,_>
      Batch.copy_output_ready(count,source,destination,input,output,output_ready))}

def copied(+count: U32,+source: U32,+destination: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+input_ready: Buffer.Ready<F32,input>,+output_ready: Buffer.Ready<F32,output>) -> Copied<count,source,destination,input,output>:
  copied_observed(count,source,destination,input,output,input_ready,output_ready,
    Cloned.observe_pair(~F32,Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output))))

def model_read(+index: U32,model: Buffer.Model<F32>) -> {Tree.read(F32,Buffer.tree(~F32,model),index) == Buffer.value(~F32,index,Buffer.content(~F32,model)) : F32}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def image_copy_model_length(model: Buffer.Model<F32>) -> {Buffer.content_length(~F32,Buffer.content(~F32,model)) == Buffer.length(~F32,model) : Nat}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def copied_point(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+count: U32,+source: Nat,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+index: Nat,
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,input)),+ready: Buffer.Ready<F32,output>,
  +destination_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool},
  +source_room: {Nat.is_le(Nat.add(source,U32.to_nat(count)),Buffer.length(~F32,input)) == True{} : Bool},
  +inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(index),Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,U32.from_nat(source),0,Buffer.pack(~F32,input),Buffer.pack(~F32,output))))) == original(logical,U32.from_nat(Nat.add(source,index))) : F32}:
  %Arithmetic.add_zero_left(U32.from_nat(index)) :
    {Buffer.value(~F32,_,Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,U32.from_nat(source),0,Buffer.pack(~F32,input),Buffer.pack(~F32,output))))) == original(logical,U32.from_nat(Nat.add(source,index))) : F32}
  Equal.trans(F32,Buffer.value(~F32,(0 + U32.from_nat(index) : U32),Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,U32.from_nat(source),0,Buffer.pack(~F32,input),Buffer.pack(~F32,output))))),
    Tree.read(F32,Buffer.tree(~F32,input),(U32.from_nat(source) + U32.from_nat(index) : U32)),original(logical,U32.from_nat(Nat.add(source,index))),
    Batch.copy_inside(count,U32.from_nat(source),0,input,output,index,ready,destination_room,inside),
    Equal.trans(F32,Tree.read(F32,Buffer.tree(~F32,input),(U32.from_nat(source) + U32.from_nat(index) : U32)),
      Buffer.value(~F32,U32.from_nat(Nat.add(source,index)),Buffer.content(~F32,input)),original(logical,U32.from_nat(Nat.add(source,index))),
      Equal.trans(F32,Tree.read(F32,Buffer.tree(~F32,input),(U32.from_nat(source) + U32.from_nat(index) : U32)),
        Tree.read(F32,Buffer.tree(~F32,input),U32.from_nat(Nat.add(source,index))),Buffer.value(~F32,U32.from_nat(Nat.add(source,index)),Buffer.content(~F32,input)),
        Equal.cong(U32,F32,query => Tree.read(F32,Buffer.tree(~F32,input),query),(U32.from_nat(source) + U32.from_nat(index) : U32),U32.from_nat(Nat.add(source,index)),WordArithmetic.native_add(source,index)),
        model_read(U32.from_nat(Nat.add(source,index)),input)),
      Dense.read(~F32,~Logical,~original,logical,Buffer.content(~F32,input),Nat.add(source,index),values,
        %Equal.sym(Nat,Buffer.content_length(~F32,Buffer.content(~F32,input)),Buffer.length(~F32,input),image_copy_model_length(input)) : {Nat.is_lt(Nat.add(source,index),_) == True{} : Bool}
        Bounds.nat_trans(Nat.add(source,index),Nat.add(source,U32.to_nat(count)),Buffer.length(~F32,input),Order.offset_lt(source,index,U32.to_nat(count),inside),source_room))))

def image_copy_content_read(+index: U32,content: Buffer.Content<F32>) -> {Tree.read(F32,Buffer.content_tree(~F32,content),index) == Buffer.value(~F32,index,content) : F32}:
  Buffer.Content{tree,length} = content
  {==}

def image_copy_prefix(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+count: U32,+source: Nat,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+remaining: Nat,+start: Nat,
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,input)),+ready: Buffer.Ready<F32,output>,
  +destination_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool},
  +source_room: {Nat.is_le(Nat.add(source,U32.to_nat(count)),Buffer.length(~F32,input)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,remaining),U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,remaining,U32.from_nat(start),Buffer.content_tree(~F32,Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,U32.from_nat(source),0,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))))) == WindowContents.image_values(~Logical,~original,remaining,Nat.add(source,start),logical) : List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +content = Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,U32.from_nat(source),0,Buffer.pack(~F32,input),Buffer.pack(~F32,output))))
      +head = Equal.trans(F32,Tree.read(F32,Buffer.content_tree(~F32,content),U32.from_nat(start)),Buffer.value(~F32,U32.from_nat(start),content),original(logical,U32.from_nat(Nat.add(source,start))),
        image_copy_content_read(U32.from_nat(start),content),copied_point(~Logical,~original,logical,count,source,input,output,start,values,ready,destination_room,source_room,Order.room_index(start,rest,U32.to_nat(count),room)))
      %Equal.sym(F32,Tree.read(F32,Buffer.content_tree(~F32,content),U32.from_nat(start)),original(logical,U32.from_nat(Nat.add(source,start))),head) :
        {_ <> Tree.read_sequence(~F32,rest,(U32.from_nat(start) + 1 : U32),Buffer.content_tree(~F32,content)) == WindowContents.image_values(~Logical,~original,1n+rest,Nat.add(source,start),logical) : List<&2,F32>}
      %Equal.sym(U32,(U32.from_nat(start) + 1 : U32),U32.from_nat(1n+start),Bounds.add_one(U32.from_nat(start))) :
        {original(logical,U32.from_nat(Nat.add(source,start))) <> Tree.read_sequence(~F32,rest,_,Buffer.content_tree(~F32,content)) == WindowContents.image_values(~Logical,~original,1n+rest,Nat.add(source,start),logical) : List<&2,F32>}
      %Addition.successor_right(source,start) :
        {original(logical,U32.from_nat(Nat.add(source,start))) <> Tree.read_sequence(~F32,rest,U32.from_nat(1n+start),Buffer.content_tree(~F32,content)) ==
          original(logical,U32.from_nat(Nat.add(source,start))) <> WindowContents.image_values(~Logical,~original,rest,_,logical) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => original(logical,U32.from_nat(Nat.add(source,start))) <> tail,
        Tree.read_sequence(~F32,rest,U32.from_nat(1n+start),Buffer.content_tree(~F32,content)),WindowContents.image_values(~Logical,~original,rest,Nat.add(source,1n+start),logical),
        image_copy_prefix(~Logical,~original,logical,count,source,input,output,rest,1n+start,values,ready,destination_room,source_room,Order.advance_room(start,rest,U32.to_nat(count),room)))

# Allocate the image buffer actually used by the dense batch. Its copy prefix
# supplies the image's input storage contract; weight and bias owners are kept.
def image_size(~Logical: Data,context: BandConvolutionValues.Context<Logical>) -> Nat:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  Nat.mul(channels,Nat.mul(height,width))

def image_problem(~Logical: Data,~original: Logical -> U32 -> F32,context: BandConvolutionValues.Context<Logical>,image: Nat) -> ConvSpec.Problem:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  BandConvolutionValues.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)

def logical_source(~Logical: Data,context: BandConvolutionValues.Context<Logical>) -> Logical:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  logical

def image_length(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+image: Nat) ->
  {ConvSpec.length(Guards.input(image_problem(~Logical,~original,context,image))) == image_size(~Logical,context) : Nat}:
  BandConvolutionValues.Context{+logical,activation,+channels,outputs,+height,+width,kernel,stride,padding,weights,bias} = context
  WindowContents.image_length(~Logical,~original,Nat.mul(channels,Nat.mul(height,width)),Nat.mul(image,Nat.mul(channels,Nat.mul(height,width))),logical)

def image_source_allocation(+count: U32,+limits: Machine.Limits,+bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) -> Depth.Ready<count>:
  Depth.bounded_with(count,Machine.capacity(limits),Limits.capacity_fits(limits),Limits.capacity_allocatable(limits),
    %Equal.sym(Bool,U32.is_le(1,Machine.capacity(limits)),Nat.is_le(1n,U32.to_nat(Machine.capacity(limits))),Comparison.le(1,Machine.capacity(limits))) : {_ == True{} : Bool}
    Order.transitive(1n,7n,U32.to_nat(Machine.capacity(limits)),{==},Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits))),bounded)

def image_source_ready_depth(+count: U32,evidence: Depth.Ready<count>) -> Buffer.ContentReady<F32,Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count))>:
  Depth.Ready{+depth_ok,+room} = evidence
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count)),Buffer.Content{Tree.fresh(F32,Tensors.depth_for(count),0.0),U32.to_nat(count)},AllocationLaws.contents(~F32,Tensors.depth_for(count),0.0,U32.to_nat(count))) : Buffer.ContentReady<F32,_>
  Buffer.content_ready(~F32,Buffer.fresh(~F32,Tensors.depth_for(count),0.0,U32.to_nat(count)),Buffer.fresh_ready(~F32,Tensors.depth_for(count),0.0,U32.to_nat(count),depth_ok,room))

type ImageSourceAllocated<-count: U32> is Data:
  ImageSourceAllocated{model: Buffer.Model<F32>,owner: {Buffer.pack(~F32,model) == Tensors.buffer(~F32,0.0,count) : Storage.Buffer<F32>},
    ready: Buffer.Ready<F32,model>,length_exact: {Buffer.length(~F32,model) == U32.to_nat(count) : Nat}}

def allocated_observed(+count: U32,+limits: Machine.Limits,+bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},observation: Buffer.Observation<F32,Tensors.buffer(~F32,0.0,count)>) -> ImageSourceAllocated<count>:
  Buffer.Observation{+model,+equation} = observation
  +contents_equal = Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,model),Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count)),
    Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.content(~F32,model),Buffer.packed_contents(~F32,model)),
    Equal.cong(Storage.Buffer<F32>,Buffer.Content<F32>,Buffer.contents(~F32),Buffer.pack(~F32,model),Tensors.buffer(~F32,0.0,count),equation))
  ImageSourceAllocated{model,equation,
    Buffer.ready_content(~F32,model,
      %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,model),Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count)),contents_equal) : Buffer.ContentReady<F32,_>
      image_source_ready_depth(count,image_source_allocation(count,limits,bounded))),
    Equal.trans(Nat,Buffer.length(~F32,model),Buffer.content_length(~F32,Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count))),U32.to_nat(count),
      Equal.trans(Nat,Buffer.length(~F32,model),Buffer.content_length(~F32,Buffer.content(~F32,model)),Buffer.content_length(~F32,Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count))),
        Equal.sym(Nat,Buffer.content_length(~F32,Buffer.content(~F32,model)),Buffer.length(~F32,model),image_copy_model_length(model)),
        Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.content(~F32,model),Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count)),contents_equal)),
      Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count)),Buffer.Content{Tree.fresh(F32,Tensors.depth_for(count),0.0),U32.to_nat(count)},AllocationLaws.contents(~F32,Tensors.depth_for(count),0.0,U32.to_nat(count))))}

def image_source_allocated(+count: U32,+limits: Machine.Limits,+bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) -> ImageSourceAllocated<count>:
  allocated_observed(count,limits,bounded,Buffer.observe(~F32,Tensors.buffer(~F32,0.0,count)))

def image_source_model_tree(model: Buffer.Model<F32>) -> {Buffer.tree(~F32,model) == Buffer.content_tree(~F32,Buffer.content(~F32,model)) : Tree.Tree<F32>}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def returned_contents(+count: U32,+source: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,
  +owners: {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,count,source,0,Buffer.pack(~F32,input),Buffer.pack(~F32,output)) : Storage.Buffer<F32> & Storage.Buffer<F32>}) ->
  {Buffer.content(~F32,right) == Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,0,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))) : Buffer.Content<F32>}:
  Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,right),Batch.copied_content(count,source,0,input,output),Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,0,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),
    observed_output_contents(count,source,0,input,output,left,right,owners),
    Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,0,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),Batch.copied_content(count,source,0,input,output),Batch.copy_output_contents(count,source,0,input,output)))

def returned_prefix(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+count: U32,+source: Nat,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,
  +owners: {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,count,U32.from_nat(source),0,Buffer.pack(~F32,input),Buffer.pack(~F32,output)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,input)),+ready: Buffer.Ready<F32,output>,+length_exact: {Buffer.length(~F32,output) == U32.to_nat(count) : Nat},
  +source_room: {Nat.is_le(Nat.add(source,U32.to_nat(count)),Buffer.length(~F32,input)) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,U32.to_nat(count),0,Buffer.tree(~F32,right)) == WindowContents.image_values(~Logical,~original,U32.to_nat(count),source,logical) : List<&2,F32>}:
  Buffer.Ready{+depth_ok,+room} = ready
  %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,right),Buffer.content_tree(~F32,Buffer.content(~F32,right)),image_source_model_tree(right)) :
    {Tree.read_sequence(~F32,U32.to_nat(count),0,_) == WindowContents.image_values(~Logical,~original,U32.to_nat(count),source,logical) : List<&2,F32>}
  %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,right),Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,U32.from_nat(source),0,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),returned_contents(count,U32.from_nat(source),input,output,left,right,owners)) :
    {Tree.read_sequence(~F32,U32.to_nat(count),0,Buffer.content_tree(~F32,_)) == WindowContents.image_values(~Logical,~original,U32.to_nat(count),source,logical) : List<&2,F32>}
  %Addition.zero_right(source) :
    {Tree.read_sequence(~F32,U32.to_nat(count),0,Buffer.content_tree(~F32,Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,U32.from_nat(source),0,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))))) == WindowContents.image_values(~Logical,~original,U32.to_nat(count),_,logical) : List<&2,F32>}
  image_copy_prefix(~Logical,~original,logical,count,source,input,output,U32.to_nat(count),0n,values,Buffer.Ready{depth_ok,room},
    %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool} room,
    source_room,Order.reflexive(U32.to_nat(count)))

def input_prefix(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+image: Nat,+count: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,
  +count_exact: {U32.to_nat(count) == image_size(~Logical,context) : Nat},
  +owners: {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,count,U32.from_nat(Nat.mul(image,image_size(~Logical,context))),0,Buffer.pack(~F32,input),Buffer.pack(~F32,output)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
  +values: Dense.Values(~F32,~Logical,~original,logical_source(~Logical,context),Buffer.content(~F32,input)),+ready: Buffer.Ready<F32,output>,+length_exact: {Buffer.length(~F32,output) == U32.to_nat(count) : Nat},
  +source_room: {Nat.is_le(Nat.add(Nat.mul(image,image_size(~Logical,context)),U32.to_nat(count)),Buffer.length(~F32,input)) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,ConvSpec.length(Guards.input(image_problem(~Logical,~original,context,image))),0,Buffer.tree(~F32,right)) == Guards.input(image_problem(~Logical,~original,context,image)) : List<&2,F32>}:
  BandConvolutionValues.Context{+logical,activation,+channels,outputs,+height,+width,kernel,stride,padding,weights,bias} = context
  +context = {BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} : BandConvolutionValues.Context<Logical>}
  %Equal.sym(Nat,ConvSpec.length(Guards.input(image_problem(~Logical,~original,context,image))),image_size(~Logical,context),image_length(~Logical,~original,context,image)) :
    {Tree.read_sequence(~F32,_,0,Buffer.tree(~F32,right)) == Guards.input(image_problem(~Logical,~original,context,image)) : List<&2,F32>}
  %count_exact :
    {Tree.read_sequence(~F32,_,0,Buffer.tree(~F32,right)) == WindowContents.image_values(~Logical,~original,_,Nat.mul(image,image_size(~Logical,context)),logical) : List<&2,F32>}
  returned_prefix(~Logical,~original,logical,count,Nat.mul(image,image_size(~Logical,context)),input,output,left,right,owners,values,ready,length_exact,source_room)

def returned_length(+count: U32,+source: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,
  +owners: {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,count,source,0,Buffer.pack(~F32,input),Buffer.pack(~F32,output)) : Storage.Buffer<F32> & Storage.Buffer<F32>}) ->
  {Buffer.length(~F32,right) == Buffer.length(~F32,output) : Nat}:
  Equal.trans(Nat,Buffer.length(~F32,right),Buffer.content_length(~F32,Buffer.content(~F32,right)),Buffer.length(~F32,output),
    Equal.sym(Nat,Buffer.content_length(~F32,Buffer.content(~F32,right)),Buffer.length(~F32,right),image_copy_model_length(right)),
    Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.content(~F32,right),Batch.copied_content(count,source,0,input,output),
      observed_output_contents(count,source,0,input,output,left,right,owners)))

def input_room(+model: Buffer.Model<F32>,+length: Nat,+exact: {Buffer.length(~F32,model) == length : Nat},ready: Buffer.Ready<F32,model>) ->
  {Nat.is_le(length,U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,model)))) == True{} : Bool}:
  Buffer.Ready{depth_ok,room} = ready
  %exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,model)))) == True{} : Bool} room

def storage_parts(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+image: Nat,+count: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +count_exact: {U32.to_nat(count) == image_size(~Logical,context) : Nat},
  +owners: {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,count,U32.from_nat(Nat.mul(image,image_size(~Logical,context))),0,Buffer.pack(~F32,input),Buffer.pack(~F32,output)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
  +values: Dense.Values(~F32,~Logical,~original,logical_source(~Logical,context),Buffer.content(~F32,input)),+ready: Buffer.Ready<F32,output>,+copied_ready: Buffer.Ready<F32,right>,
  +length_exact: {Buffer.length(~F32,output) == U32.to_nat(count) : Nat},
  +source_room: {Nat.is_le(Nat.add(Nat.mul(image,image_size(~Logical,context)),U32.to_nat(count)),Buffer.length(~F32,input)) == True{} : Bool},
  +weight_values: {Tree.read_sequence(~F32,ConvSpec.length(Guards.weights(image_problem(~Logical,~original,context,image))),0,Buffer.tree(~F32,weights)) == Guards.weights(image_problem(~Logical,~original,context,image)) : List<&2,F32>},
  +bias_values: {Tree.read_sequence(~F32,ConvSpec.length(Guards.bias(image_problem(~Logical,~original,context,image))),0,Buffer.tree(~F32,bias)) == Guards.bias(image_problem(~Logical,~original,context,image)) : List<&2,F32>},
  +weight_room: {Nat.is_le(ConvSpec.length(Guards.weights(image_problem(~Logical,~original,context,image))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,weights)))) == True{} : Bool},
  +bias_room: {Nat.is_le(ConvSpec.length(Guards.bias(image_problem(~Logical,~original,context,image))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,bias)))) == True{} : Bool}) ->
  DenseSingle.SourceStorage<image_problem(~Logical,~original,context,image),right,weights,bias>:
  +problem = image_problem(~Logical,~original,context,image)
  +input_length_exact = Equal.trans(Nat,Buffer.length(~F32,right),Buffer.length(~F32,output),ConvSpec.length(Guards.input(problem)),
    returned_length(count,U32.from_nat(Nat.mul(image,image_size(~Logical,context))),input,output,left,right,owners),
    Equal.trans(Nat,Buffer.length(~F32,output),U32.to_nat(count),ConvSpec.length(Guards.input(problem)),length_exact,
      Equal.trans(Nat,U32.to_nat(count),image_size(~Logical,context),ConvSpec.length(Guards.input(problem)),count_exact,
        Equal.sym(Nat,ConvSpec.length(Guards.input(problem)),image_size(~Logical,context),image_length(~Logical,~original,context,image)))))
  DenseSingle.source_storage(problem,right,weights,bias,
    input_prefix(~Logical,~original,context,image,count,input,output,left,right,count_exact,owners,values,ready,length_exact,source_room),
    weight_values,bias_values,input_room(right,ConvSpec.length(Guards.input(problem)),input_length_exact,copied_ready),weight_room,bias_room)

# Batch leaves consume the once-reordered Tile owner. Its real content
# relation to the original weights connects the existing packed leaf model;
# allocation and finite values are produced independently of owner cloning.
def work_inner(work: Pipeline.Work) -> U32:
  Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner} = work
  inner

def array_matches(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Tree.Tree<F32>,+tiles: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>,
  +reordered: {tiles == WeightsLaws.run(DensePipeline.work_rows(work),work_inner(work),weights) : Tree.Tree<Tile.Tile8>}) ->
  {DensePipeline.array_run_packed(~custom,activation,depth,work,Tree.pack(F32,input),Tree.pack(Tile.Tile8,tiles),Tree.pack(F32,bias)) == Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,input,weights,bias)) : Array<F32>}:
  Pipeline.Work{+rows,+height,+width,+kernel,+stride,+padding,+output_width,+positions,+inner} = work
  +work = {Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner} : Pipeline.Work}
  %Equal.sym(Tree.Tree<Tile.Tile8>,tiles,WeightsLaws.run(rows,inner,weights),reordered) :
    {DensePipeline.array_run_packed(~custom,activation,depth,work,Tree.pack(F32,input),Tree.pack(Tile.Tile8,_),Tree.pack(F32,bias)) == Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,input,weights,bias)) : Array<F32>}
  %WeightsLaws.run_matches(rows,inner,weights) :
    {DensePipeline.array_run_packed(~custom,activation,depth,work,Tree.pack(F32,input),_,Tree.pack(F32,bias)) == Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,input,weights,bias)) : Array<F32>}
  DenseConvolution.array_run_matches(~custom,activation,depth,work,input,weights,bias)

def storage_matches(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +reordered: {Buffer.tree(~Tile.Tile8,tiles) == WeightsLaws.run(DensePipeline.work_rows(work),work_inner(work),Buffer.tree(~F32,weights)) : Tree.Tree<Tile.Tile8>}) ->
  {LeafExecution.storage(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))) ==
    Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}:
  Equal.trans(Array<F32>,LeafExecution.storage(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),
    DensePipeline.array_run_packed(~custom,activation,depth,work,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))),
    Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))),DensePipeline.run_packed_projection(~custom,activation,depth,work,input,tiles,bias),
    %Equal.sym(Array<F32>,LeafExecution.storage(~F32,Buffer.pack(~F32,input)),Tree.pack(F32,Buffer.tree(~F32,input)),DensePipeline.storage_matches(~F32,input)) :
      {DensePipeline.array_run_packed(~custom,activation,depth,work,_,LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) == Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}
    %Equal.sym(Array<Tile.Tile8>,LeafExecution.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),Tree.pack(Tile.Tile8,Buffer.tree(~Tile.Tile8,tiles)),DensePipeline.storage_matches(~Tile.Tile8,tiles)) :
      {DensePipeline.array_run_packed(~custom,activation,depth,work,Tree.pack(F32,Buffer.tree(~F32,input)),_,LeafExecution.storage(~F32,Buffer.pack(~F32,bias))) == Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}
    %Equal.sym(Array<F32>,LeafExecution.storage(~F32,Buffer.pack(~F32,bias)),Tree.pack(F32,Buffer.tree(~F32,bias)),DensePipeline.storage_matches(~F32,bias)) :
      {DensePipeline.array_run_packed(~custom,activation,depth,work,Tree.pack(F32,Buffer.tree(~F32,input)),Tree.pack(Tile.Tile8,Buffer.tree(~Tile.Tile8,tiles)),_) == Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))) : Array<F32>}
    array_matches(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias),reordered))

def length_matches(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {LeafExecution.buffer_length(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))) == U32.to_nat(DensePipeline.output_size(work)) : Nat}:
  Equal.trans(Nat,LeafExecution.buffer_length(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),
    KernelPipelineBufferShape.length(KernelPipelineBufferShape.footprint(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)))),U32.to_nat(DensePipeline.output_size(work)),
    Content.length_projection(Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),
    Equal.cong(KernelPipelineBufferShape.Footprint,Nat,value => KernelPipelineBufferShape.length(value),KernelPipelineBufferShape.footprint(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),
      DensePipeline.allocated_footprint(DensePipeline.output_size(work)),DensePipeline.run_packed_footprint(~custom,activation,depth,work,input,tiles,bias)))

def contents(~custom: F32 -> F32,+activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +reordered: {Buffer.tree(~Tile.Tile8,tiles) == WeightsLaws.run(DensePipeline.work_rows(work),work_inner(work),Buffer.tree(~F32,weights)) : Tree.Tree<Tile.Tile8>}) ->
  {Buffer.contents(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))) ==
    Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) : Buffer.Content<F32>}:
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),
    Buffer.Content{DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)),U32.to_nat(DensePipeline.output_size(work))},Content.run_contents(~custom,activation,depth,work,input,weights,bias)) :
    {Buffer.contents(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))) == _ : Buffer.Content<F32>}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),
    Buffer.Content{Tree.reflect(F32,LeafExecution.storage(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)))),LeafExecution.buffer_length(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)))},
    LeafExecution.contents_parts(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)))) :
    {_ == Buffer.Content{DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)),U32.to_nat(DensePipeline.output_size(work))} : Buffer.Content<F32>}
  %Equal.sym(Nat,LeafExecution.buffer_length(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),U32.to_nat(DensePipeline.output_size(work)),length_matches(~custom,activation,depth,work,input,tiles,bias)) :
    {Buffer.Content{Tree.reflect(F32,LeafExecution.storage(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)))),_} == Buffer.Content{DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)),U32.to_nat(DensePipeline.output_size(work))} : Buffer.Content<F32>}
  %Equal.sym(Array<F32>,LeafExecution.storage(~F32,Pipeline.run_packed(~custom,activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))),storage_matches(~custom,activation,depth,work,input,tiles,weights,bias,reordered)) :
    {Buffer.Content{Tree.reflect(F32,_),U32.to_nat(DensePipeline.output_size(work))} == Buffer.Content{DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)),U32.to_nat(DensePipeline.output_size(work))} : Buffer.Content<F32>}
  Equal.cong(Tree.Tree<F32>,Buffer.Content<F32>,tree => Buffer.Content{tree,U32.to_nat(DensePipeline.output_size(work))},Tree.reflect(F32,Tree.pack(F32,DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)))),
    DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)),StorageLaws.reflect_packed(F32,DenseConvolution.run_model(~custom,activation,depth,work,Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias))))

def values(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+source: DenseSingle.SourceStorage<problem,input,weights,bias>,
  +reordered: {Buffer.tree(~Tile.Tile8,tiles) == WeightsLaws.run(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Buffer.tree(~F32,weights)) : Tree.Tree<Tile.Tile8>},
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  Dense.Values(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{problem,activation},Buffer.contents(~F32,Pipeline.run_packed(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)))):
  ConvSpec.Problem{ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding},+xs,+ws,+bs} = problem
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},xs,ws,bs} : ConvSpec.Problem}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Pipeline.run_packed(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),
    Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),contents(~custom,activation,depth,Inference.work(problem),input,tiles,weights,bias,reordered)) :
    Dense.Values(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{problem,activation},_)
  DenseSingle.run_values(~custom,activation,limits,problem,depth,input,weights,bias,source,valid,supported)

def packed_image_ready(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  Buffer.ContentReady<F32,Buffer.contents(~F32,Pipeline.run_packed(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)))>:
  ConvSpec.Problem{ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding},+xs,+ws,+bs} = problem
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},xs,ws,bs} : ConvSpec.Problem}
  +fast = Support.fast(limits,problem,supported)
  +span_bound = {
    %Equal.sym(Nat,U32.to_nat(Padded.spatial(problem)),GatherCopyCoordinates.spatial(problem),Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),fast)) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    Padded.spatial_within_padded(limits,problem,Guards.valid(problem,valid),fast) : {Nat.is_le(U32.to_nat(Padded.spatial(problem)),Support.padded(problem)) == True{} : Bool}}
  Content.ready_observed(DensePipeline.output_size(Inference.work(problem)),Pipeline.run_packed(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)),
    Buffer.observe(~F32,Pipeline.run_packed(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),DensePipeline.run_packed_footprint(~custom,activation,depth,Inference.work(problem),input,tiles,bias),
    LeafExecution.product_ready(limits,MicroLeaf.rows(problem),Padded.spatial(problem),DenseConvolution.leaf_room(limits,problem,Padded.spatial(problem),fast,span_bound)))

# Reorder once, observe the actual Tile owner, and carry its content relation
# through real batch clones. The shared reorder and clone laws supply every
# equality; no numerical kernel or alternate allocation is introduced.
def weight_owners_model_tree(+model: Buffer.Model<Tile.Tile8>) ->
  {Buffer.tree(~Tile.Tile8,model) == Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,model)) : Tree.Tree<Tile.Tile8>}:
  Buffer.Model{tree,length,certificate} = model
  {==}

type Reordered<-rows: U32,-inner: U32,-weights: Buffer.Model<F32>> is Data:
  Reordered{model: Buffer.Model<Tile.Tile8>,
    owner: {Buffer.pack(~Tile.Tile8,model) == GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights)) : Storage.Buffer<Tile.Tile8>},
    exact: {Buffer.tree(~Tile.Tile8,model) == WeightsLaws.run(rows,inner,Buffer.tree(~F32,weights)) : Tree.Tree<Tile.Tile8>}}

def reordered_observed(+rows: U32,+inner: U32,+weights: Buffer.Model<F32>,observation: Buffer.Observation<Tile.Tile8,GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights))>) -> Reordered<rows,inner,weights>:
  Buffer.Observation{+model,+equation} = observation
  +content_equal = Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,model),
    Buffer.contents(~Tile.Tile8,GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights))),
    Buffer.Content{WeightsLaws.run(rows,inner,Buffer.tree(~F32,weights)),U32.to_nat((((rows + 7) / 8) * inner : U32))},
    Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,model),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)),Buffer.contents(~Tile.Tile8,GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights))),
      Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)),Buffer.content(~Tile.Tile8,model),Buffer.packed_contents(~Tile.Tile8,model)),
      Equal.cong(Storage.Buffer<Tile.Tile8>,Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8),Buffer.pack(~Tile.Tile8,model),GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights)),equation)),
    KernelWeightsBufferRefinement.reordered(rows,inner,weights))
  Reordered{model,equation,
    Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,model),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,model)),WeightsLaws.run(rows,inner,Buffer.tree(~F32,weights)),weight_owners_model_tree(model),
      Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8),Buffer.content(~Tile.Tile8,model),
        Buffer.Content{WeightsLaws.run(rows,inner,Buffer.tree(~F32,weights)),U32.to_nat((((rows + 7) / 8) * inner : U32))},content_equal))}

def reordered(+rows: U32,+inner: U32,+weights: Buffer.Model<F32>) -> Reordered<rows,inner,weights>:
  reordered_observed(rows,inner,weights,Buffer.observe(~Tile.Tile8,GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights))))

def cloned_tree(+source: Buffer.Model<Tile.Tile8>,+child: Buffer.Model<Tile.Tile8>,+contents: {Buffer.content(~Tile.Tile8,child) == Buffer.content(~Tile.Tile8,source) : Buffer.Content<Tile.Tile8>}) ->
  {Buffer.tree(~Tile.Tile8,child) == Buffer.tree(~Tile.Tile8,source) : Tree.Tree<Tile.Tile8>}:
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,child),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,child)),Buffer.tree(~Tile.Tile8,source),weight_owners_model_tree(child),
    Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,child)),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,source)),Buffer.tree(~Tile.Tile8,source),
      Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8),Buffer.content(~Tile.Tile8,child),Buffer.content(~Tile.Tile8,source),contents),
      Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,source),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,source)),weight_owners_model_tree(source))))

# The fallback's actual whole-source buffer is allocated from the inspected
# logical element count. Observation retains that returned certificate, and
# the machine envelope constructs its Ready and gather-domain obligations.
def count(geometry: Bands.Geometry) -> U32:
  U32.from_nat(BandGather.elements(geometry))

# The actual word count selects the native empty, single or batch procedure.
# Classification is constructed by matching that word, never supplied as a
# successful execution or an equality to an alternative implementation.
def rows(work: Pipeline.Work) -> U32:
  Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner} = work
  rows

def inner(work: Pipeline.Work) -> U32:
  Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner} = work
  inner

type CountKind<-images: U32> is Data:
  EmptyCount{equal: {images == 0 : U32}}
  SingleCount{equal: {images == 1 : U32}}
  BatchCount{bound: {Nat.is_lt(1n,U32.to_nat(images)) == True{} : Bool}}

type NatObservation<-images: U32> is Data:
  NatObservation{count: Nat,exact: {count == U32.to_nat(images) : Nat}}

def observe_count(+images: U32) -> NatObservation<images>:
  NatObservation{U32.to_nat(images),{==}}

def native_count(+images: U32,+count: Nat,+exact: {count == U32.to_nat(images) : Nat}) -> {images == U32.from_nat(count) : U32}:
  Equal.trans(U32,images,U32.from_nat(U32.to_nat(images)),U32.from_nat(count),
    Equal.sym(U32,U32.from_nat(U32.to_nat(images)),images,Arithmetic.native_roundtrip(images)),
    Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(images),count,Equal.sym(Nat,count,U32.to_nat(images),exact)))

def classified(+images: U32,count: Nat,+exact: {count == U32.to_nat(images) : Nat}) -> CountKind<images>:
  match count:
    case 0n: EmptyCount{native_count(images,0n,exact)}
    case 1n: SingleCount{native_count(images,1n,exact)}
    case 2n+ +rest:
      BatchCount{
        %exact : {Nat.is_lt(1n,_) == True{} : Bool}
        {==}}

def images_route_observed(+images: U32,observation: NatObservation<images>) -> CountKind<images>:
  NatObservation{count,exact} = observation
  classified(images,count,exact)

def classify(+images: U32) -> CountKind<images>:
  images_route_observed(images,observe_count(images))

def many(~custom: F32 -> F32,+activation: Activations.Activation,+images: U32,+depth: U32,+work: Pipeline.Work,+input_size: U32,+output_size: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +bound: {Nat.is_lt(1n,U32.to_nat(images)) == True{} : Bool}) ->
  {TensorConvolution.images(~custom,activation,images,depth,work,input_size,output_size,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)) ==
    TensorConvolution.batch(~custom,activation,U32.to_nat(images),0,depth,work,input_size,output_size,Buffer.pack(~F32,input),GemmBuffer.reorder(rows(work),inner(work),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(images * output_size : U32))) : Storage.Buffer<F32>}:
  Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner} = work
  %Equal.sym(Bool,U32.is_eq(images,0),False{},Flags.zero_false(images,bound)) :
    {TensorConvolution.images_selected(~custom,activation,_,U32.is_eq(images,1),images,depth,work,input_size,output_size,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)) ==
      TensorConvolution.batch(~custom,activation,U32.to_nat(images),0,depth,work,input_size,output_size,Buffer.pack(~F32,input),GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(images * output_size : U32))) : Storage.Buffer<F32>}
  %Equal.sym(Bool,U32.is_eq(images,1),False{},Flags.one_false(images,bound)) :
    {TensorConvolution.images_selected(~custom,activation,False{},_,images,depth,work,input_size,output_size,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)) ==
      TensorConvolution.batch(~custom,activation,U32.to_nat(images),0,depth,work,input_size,output_size,Buffer.pack(~F32,input),GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(images * output_size : U32))) : Storage.Buffer<F32>}
  {==}
