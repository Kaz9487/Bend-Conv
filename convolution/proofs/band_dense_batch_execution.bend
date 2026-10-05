# The dense batch caller: inputs, per-image steps, prefix, values and its
# contract.
import Base
import ./band_dense_sources.bend as DenseSources
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ./band_convolution_values.bend as BandConvolutionValues
import ../../lib/machine_limits.bend as Machine
import ./prepared_output_readback.bend as PreparedOutputReadback
import ../specification.bend as ConvSpec
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/tensor_dense_contract.bend as Dense
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./convolution_guard_facts.bend as Guards
import ./microkernel_inference_refinement.bend as Inference
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/storage_buffer.bend as Storage
import ../../lib/storage_copy.bend as Copy
import ../inference.bend as NativeConv
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ./band_dense_batch.bend as Batch
import ./band_dense_convolution.bend as DenseConvolution
import ./band_dense_single.bend as DenseSingle
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./reduction_coordinate_bounds.bend as Coordinates
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/storage_clone.bend as Clone
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/word_multiplication.bend as Multiplication
import ../../lib/tensor.bend as Tensors
import ./band_dense_pipeline.bend as DensePipeline
import ./band_leaf_execution.bend as LeafExecution
import ./concrete_support_guard.bend as Support
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ./supported_padded_coordinates.bend as Padded
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ../../lib/proofs/contiguous_copy_readback.bend as CopyReadback
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_interval.bend as Interval
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/storage_clone_refinement.bend as Cloned
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic

# One actual batch image owns its copied input and the packed leaf result.
# Copy, allocation and finite leaf evidence are produced along that schedule.
def image_activation(~Logical: Data,context: BandConvolutionValues.Context<Logical>) -> Activations.Activation:
  BandConvolutionValues.Context{logical,image_activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  image_activation

def after_copy(~custom: F32 -> F32,+image_activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,owners: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Storage.Buffer<F32> & Storage.Buffer<F32>:
  Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~(Storage.Buffer<F32> & Storage.Buffer<F32>),owners,source => batch_image =>
    (source,Pipeline.run_packed(~custom,image_activation,depth,work,batch_image,Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))))

def image_procedure(~custom: F32 -> F32,+image_activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+index: U32,+count: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {TensorConvolution.image(~custom,image_activation,depth,work,index,count,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)) ==
    after_copy(~custom,image_activation,depth,work,tiles,bias,Copy.range(~F32,count,(index * count : U32),0,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count))) : Storage.Buffer<F32> & Storage.Buffer<F32>}:
  {==}

def copied_procedure(~custom: F32 -> F32,+image_activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {after_copy(~custom,image_activation,depth,work,tiles,bias,(Buffer.pack(~F32,left),Buffer.pack(~F32,right))) ==
    (Buffer.pack(~F32,left),Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))) : Storage.Buffer<F32> & Storage.Buffer<F32>}:
  {==}

def offset_matches(+index: U32,+count: U32,+size: Nat,+exact: {U32.to_nat(count) == size : Nat}) ->
  {(index * count : U32) == U32.from_nat(Nat.mul(U32.to_nat(index),size)) : U32}:
  Equal.trans(U32,(index * count : U32),U32.from_nat(Nat.mul(U32.to_nat(index),U32.to_nat(count))),U32.from_nat(Nat.mul(U32.to_nat(index),size)),Multiplication.native_multiply(index,count),
    Equal.cong(Nat,U32,last => U32.from_nat(Nat.mul(U32.to_nat(index),last)),U32.to_nat(count),size,exact))

def native_copy(+index: U32,+count: U32,+size: Nat,+input: Buffer.Model<F32>,+allocated: Buffer.Model<F32>,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,
  +exact: {U32.to_nat(count) == size : Nat},+allocation: {Buffer.pack(~F32,allocated) == Tensors.buffer(~F32,0.0,count) : Storage.Buffer<F32>},
  +owners: {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,count,U32.from_nat(Nat.mul(U32.to_nat(index),size)),0,Buffer.pack(~F32,input),Buffer.pack(~F32,allocated)) : Storage.Buffer<F32> & Storage.Buffer<F32>}) ->
  {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,count,(index * count : U32),0,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count)) : Storage.Buffer<F32> & Storage.Buffer<F32>}:
  Equal.trans(Storage.Buffer<F32> & Storage.Buffer<F32>,(Buffer.pack(~F32,left),Buffer.pack(~F32,right)),
    Copy.range(~F32,count,U32.from_nat(Nat.mul(U32.to_nat(index),size)),0,Buffer.pack(~F32,input),Buffer.pack(~F32,allocated)),Copy.range(~F32,count,(index * count : U32),0,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count)),owners,
    Equal.trans(Storage.Buffer<F32> & Storage.Buffer<F32>,Copy.range(~F32,count,U32.from_nat(Nat.mul(U32.to_nat(index),size)),0,Buffer.pack(~F32,input),Buffer.pack(~F32,allocated)),
      Copy.range(~F32,count,U32.from_nat(Nat.mul(U32.to_nat(index),size)),0,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count)),Copy.range(~F32,count,(index * count : U32),0,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count)),
      Equal.cong(Storage.Buffer<F32>,Storage.Buffer<F32> & Storage.Buffer<F32>,target => Copy.range(~F32,count,U32.from_nat(Nat.mul(U32.to_nat(index),size)),0,Buffer.pack(~F32,input),target),Buffer.pack(~F32,allocated),Tensors.buffer(~F32,0.0,count),allocation),
      Equal.cong(U32,Storage.Buffer<F32> & Storage.Buffer<F32>,offset => Copy.range(~F32,count,offset,0,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count)),U32.from_nat(Nat.mul(U32.to_nat(index),size)),(index * count : U32),
        Equal.sym(U32,(index * count : U32),U32.from_nat(Nat.mul(U32.to_nat(index),size)),offset_matches(index,count,size,exact)))))

def scheduled(~custom: F32 -> F32,+image_activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+index: U32,+count: U32,+size: Nat,+input: Buffer.Model<F32>,+allocated: Buffer.Model<F32>,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,
  +exact: {U32.to_nat(count) == size : Nat},+allocation: {Buffer.pack(~F32,allocated) == Tensors.buffer(~F32,0.0,count) : Storage.Buffer<F32>},
  +owners: {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,count,U32.from_nat(Nat.mul(U32.to_nat(index),size)),0,Buffer.pack(~F32,input),Buffer.pack(~F32,allocated)) : Storage.Buffer<F32> & Storage.Buffer<F32>}) ->
  {TensorConvolution.image(~custom,image_activation,depth,work,index,count,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)) ==
    (Buffer.pack(~F32,left),Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))) : Storage.Buffer<F32> & Storage.Buffer<F32>}:
  Equal.trans(Storage.Buffer<F32> & Storage.Buffer<F32>,TensorConvolution.image(~custom,image_activation,depth,work,index,count,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)),
    after_copy(~custom,image_activation,depth,work,tiles,bias,Copy.range(~F32,count,(index * count : U32),0,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count))),
    (Buffer.pack(~F32,left),Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),image_procedure(~custom,image_activation,depth,work,index,count,input,tiles,bias),
    Equal.trans(Storage.Buffer<F32> & Storage.Buffer<F32>,after_copy(~custom,image_activation,depth,work,tiles,bias,Copy.range(~F32,count,(index * count : U32),0,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count))),
      after_copy(~custom,image_activation,depth,work,tiles,bias,(Buffer.pack(~F32,left),Buffer.pack(~F32,right))),
      (Buffer.pack(~F32,left),Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),
      Equal.cong(Storage.Buffer<F32> & Storage.Buffer<F32>,Storage.Buffer<F32> & Storage.Buffer<F32>,after_copy(~custom,image_activation,depth,work,tiles,bias),
        Copy.range(~F32,count,(index * count : U32),0,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count)),(Buffer.pack(~F32,left),Buffer.pack(~F32,right)),
        Equal.sym(Storage.Buffer<F32> & Storage.Buffer<F32>,(Buffer.pack(~F32,left),Buffer.pack(~F32,right)),Copy.range(~F32,count,(index * count : U32),0,Buffer.pack(~F32,input),Tensors.buffer(~F32,0.0,count)),native_copy(index,count,size,input,allocated,left,right,exact,allocation,owners))),
      copied_procedure(~custom,image_activation,depth,work,left,right,tiles,bias)))

def output_contents(~custom: F32 -> F32,+image_activation: Activations.Activation,+depth: U32,+work: Pipeline.Work,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +equation: {Buffer.pack(~F32,output) == Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)) : Storage.Buffer<F32>}) ->
  {Buffer.content(~F32,output) == Buffer.contents(~F32,Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))) : Buffer.Content<F32>}:
  Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,output),Buffer.contents(~F32,Buffer.pack(~F32,output)),Buffer.contents(~F32,Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),
    Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,output)),Buffer.content(~F32,output),Buffer.packed_contents(~F32,output)),
    Equal.cong(Storage.Buffer<F32>,Buffer.Content<F32>,Buffer.contents(~F32),Buffer.pack(~F32,output),Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)),equation))

def native_length(output: Storage.Buffer<F32>) -> {Buffer.content_length(~F32,Buffer.contents(~F32,output)) == LeafExecution.buffer_length(~F32,output) : Nat}:
  Storage.Buffer{storage,length,certificate} = output
  {==}

type ImageResult<-Logical: Data,-original: Logical -> U32 -> F32,-custom: F32 -> F32,-context: BandConvolutionValues.Context<Logical>,-index: U32,-count: U32,-depth: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-bias: Buffer.Model<F32>> is Data:
  ImageResult{source_model: Buffer.Model<F32>,output_model: Buffer.Model<F32>,
    owners: {(Buffer.pack(~F32,source_model),Buffer.pack(~F32,output_model)) == TensorConvolution.image(~custom,image_activation(~Logical,context),depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))),index,count,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
    retained: {Buffer.content(~F32,source_model) == Buffer.content(~F32,input) : Buffer.Content<F32>},
    source_values: Dense.Values(~F32,~Logical,~original,DenseSources.logical_source(~Logical,context),Buffer.content(~F32,source_model)),
    source_ready: Buffer.Ready<F32,source_model>,output_ready: Buffer.Ready<F32,output_model>,
    length_exact: {Buffer.length(~F32,output_model) == U32.to_nat(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index)))) : Nat},
    values: Dense.Values(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index)),image_activation(~Logical,context)},Buffer.content(~F32,output_model))}

def leaf_observed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+index: U32,+count: U32,+depth: U32,+input: Buffer.Model<F32>,+allocated: Buffer.Model<F32>,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +count_exact: {U32.to_nat(count) == DenseSources.image_size(~Logical,context) : Nat},+allocation: {Buffer.pack(~F32,allocated) == Tensors.buffer(~F32,0.0,count) : Storage.Buffer<F32>},
  +owners: {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,count,U32.from_nat(Nat.mul(U32.to_nat(index),DenseSources.image_size(~Logical,context))),0,Buffer.pack(~F32,input),Buffer.pack(~F32,allocated)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
  +retained: {Buffer.content(~F32,left) == Buffer.content(~F32,input) : Buffer.Content<F32>},+source_ready: Buffer.Ready<F32,left>,
  +source_values: Dense.Values(~F32,~Logical,~original,DenseSources.logical_source(~Logical,context),Buffer.content(~F32,input)),
  +storage: DenseSingle.SourceStorage<DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index)),right,weights,bias>,
  +reordered: {Buffer.tree(~Tile.Tile8,tiles) == WeightsLaws.run(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))),Buffer.tree(~F32,weights)) : Tree.Tree<Tile.Tile8>},
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))) == True{} : Bool},
  observation: Buffer.Observation<F32,Pipeline.run_packed(~custom,image_activation(~Logical,context),depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))),Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))>) -> ImageResult<Logical,original,custom,context,index,count,depth,input,tiles,bias>:
  Buffer.Observation{+output,+equation} = observation
  +problem = DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))
  +image_activation = image_activation(~Logical,context)
  +work = Inference.work(problem)
  +content_exact = output_contents(~custom,image_activation,depth,work,right,tiles,bias,output,equation)
  ImageResult{left,output,
    Equal.trans(Storage.Buffer<F32> & Storage.Buffer<F32>,(Buffer.pack(~F32,left),Buffer.pack(~F32,output)),
      (Buffer.pack(~F32,left),Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),TensorConvolution.image(~custom,image_activation,depth,work,index,count,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)),
      Equal.cong(Storage.Buffer<F32>,Storage.Buffer<F32> & Storage.Buffer<F32>,result => (Buffer.pack(~F32,left),result),Buffer.pack(~F32,output),Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)),equation),
      Equal.sym(Storage.Buffer<F32> & Storage.Buffer<F32>,TensorConvolution.image(~custom,image_activation,depth,work,index,count,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)),
        (Buffer.pack(~F32,left),Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),scheduled(~custom,image_activation,depth,work,index,count,DenseSources.image_size(~Logical,context),input,allocated,left,right,tiles,bias,count_exact,allocation,owners))),
    retained,
    %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,left),Buffer.content(~F32,input),retained) : Dense.Values(~F32,~Logical,~original,DenseSources.logical_source(~Logical,context),_) source_values,
    source_ready,
    Buffer.ready_content(~F32,output,
      %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,output),Buffer.contents(~F32,Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),content_exact) : Buffer.ContentReady<F32,_>
      DenseSources.packed_image_ready(~custom,image_activation,limits,problem,depth,right,tiles,bias,valid,supported)),
    Equal.trans(Nat,Buffer.length(~F32,output),Buffer.content_length(~F32,Buffer.contents(~F32,Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)))),U32.to_nat(PreparedOutputReadback.count(problem)),
      Equal.trans(Nat,Buffer.length(~F32,output),Buffer.content_length(~F32,Buffer.content(~F32,output)),Buffer.content_length(~F32,Buffer.contents(~F32,Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)))),
        Equal.sym(Nat,Buffer.content_length(~F32,Buffer.content(~F32,output)),Buffer.length(~F32,output),DenseSources.image_copy_model_length(output)),
        Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.content(~F32,output),Buffer.contents(~F32,Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),content_exact)),
      Equal.trans(Nat,Buffer.content_length(~F32,Buffer.contents(~F32,Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)))),LeafExecution.buffer_length(~F32,Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),U32.to_nat(PreparedOutputReadback.count(problem)),
        native_length(Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),
        Equal.trans(Nat,LeafExecution.buffer_length(~F32,Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),U32.to_nat(DensePipeline.output_size(work)),U32.to_nat(PreparedOutputReadback.count(problem)),DenseSources.length_matches(~custom,image_activation,depth,work,right,tiles,bias),Equal.cong(U32,Nat,U32.to_nat,DensePipeline.output_size(work),PreparedOutputReadback.count(problem),DenseSingle.count_matches(problem))))),
    %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,output),Buffer.contents(~F32,Pipeline.run_packed(~custom,image_activation,depth,work,Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))),content_exact) :
      Dense.Values(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{problem,image_activation},_)
    DenseSources.values(~custom,image_activation,limits,problem,depth,right,tiles,weights,bias,storage,reordered,valid,supported)}

# These are source tensor and geometry contracts. The batch_image's storage, copy
# success, output values and output readiness are constructed by the factory.
type ImageInputs<-Logical: Data,-original: Logical -> U32 -> F32,-context: BandConvolutionValues.Context<Logical>,-limits: Machine.Limits,-index: U32,-count: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-weights: Buffer.Model<F32>,-bias: Buffer.Model<F32>> is Data:
  ImageInputs{count_exact: {U32.to_nat(count) == DenseSources.image_size(~Logical,context) : Nat},
    count_within: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},
    source_values: Dense.Values(~F32,~Logical,~original,DenseSources.logical_source(~Logical,context),Buffer.content(~F32,input)),source_ready: Buffer.Ready<F32,input>,
    source_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(index),DenseSources.image_size(~Logical,context)),U32.to_nat(count)),Buffer.length(~F32,input)) == True{} : Bool},
    weight_values: {Tree.read_sequence(~F32,ConvSpec.length(Guards.weights(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index)))),0,Buffer.tree(~F32,weights)) == Guards.weights(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))) : List<&2,F32>},
    bias_values: {Tree.read_sequence(~F32,ConvSpec.length(Guards.bias(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index)))),0,Buffer.tree(~F32,bias)) == Guards.bias(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))) : List<&2,F32>},
    weight_room: {Nat.is_le(ConvSpec.length(Guards.weights(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index)))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,weights)))) == True{} : Bool},
    bias_room: {Nat.is_le(ConvSpec.length(Guards.bias(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index)))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,bias)))) == True{} : Bool},
    reordered: {Buffer.tree(~Tile.Tile8,tiles) == WeightsLaws.run(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))),Buffer.tree(~F32,weights)) : Tree.Tree<Tile.Tile8>},
    valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))) == True{} : Bool},
    supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))) == True{} : Bool}}

def input_ready(~Logical: Data,~original: Logical -> U32 -> F32,-context: BandConvolutionValues.Context<Logical>,-limits: Machine.Limits,-index: U32,-count: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-weights: Buffer.Model<F32>,-bias: Buffer.Model<F32>,evidence: ImageInputs<Logical,original,context,limits,index,count,input,tiles,weights,bias>) -> Buffer.Ready<F32,input>:
  ImageInputs{count_exact,count_within,source_values,source_ready,source_room,weight_values,bias_values,weight_room,bias_room,reordered,valid,supported} = evidence
  source_ready

def count_within(~Logical: Data,~original: Logical -> U32 -> F32,-context: BandConvolutionValues.Context<Logical>,-limits: Machine.Limits,-index: U32,-count: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-weights: Buffer.Model<F32>,-bias: Buffer.Model<F32>,evidence: ImageInputs<Logical,original,context,limits,index,count,input,tiles,weights,bias>) -> {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}:
  ImageInputs{count_exact,count_within,source_values,source_ready,source_room,weight_values,bias_values,weight_room,bias_room,reordered,valid,supported} = evidence
  count_within

def image_with_copy(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+index: U32,+count: U32,+depth: U32,+input: Buffer.Model<F32>,+allocated: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +allocation: {Buffer.pack(~F32,allocated) == Tensors.buffer(~F32,0.0,count) : Storage.Buffer<F32>},+allocation_ready: Buffer.Ready<F32,allocated>,+allocation_length: {Buffer.length(~F32,allocated) == U32.to_nat(count) : Nat},
  evidence: ImageInputs<Logical,original,context,limits,index,count,input,tiles,weights,bias>,
  copied: DenseSources.Copied<count,U32.from_nat(Nat.mul(U32.to_nat(index),DenseSources.image_size(~Logical,context))),0,input,allocated>) -> ImageResult<Logical,original,custom,context,index,count,depth,input,tiles,bias>:
  ImageInputs{+count_exact,count_within,+source_values,source_ready,+source_room,+weight_values,+bias_values,+weight_room,+bias_room,+reordered,+valid,+supported} = evidence
  DenseSources.Copied{+left,+right,+owners,+retained,written,+left_ready,+right_ready} = copied
  +storage = DenseSources.storage_parts(~Logical,~original,context,U32.to_nat(index),count,input,allocated,left,right,weights,bias,count_exact,owners,source_values,allocation_ready,right_ready,allocation_length,source_room,weight_values,bias_values,weight_room,bias_room)
  leaf_observed(~custom,~Logical,~original,context,limits,index,count,depth,input,allocated,left,right,tiles,weights,bias,count_exact,allocation,owners,retained,left_ready,source_values,storage,reordered,valid,supported,
    Buffer.observe(~F32,Pipeline.run_packed(~custom,image_activation(~Logical,context),depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(index))),Buffer.pack(~F32,right),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias))))

def with_allocation(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+index: U32,+count: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +evidence: ImageInputs<Logical,original,context,limits,index,count,input,tiles,weights,bias>,allocation: DenseSources.ImageSourceAllocated<count>) -> ImageResult<Logical,original,custom,context,index,count,depth,input,tiles,bias>:
  DenseSources.ImageSourceAllocated{+model,+owner,+ready,+length_exact} = allocation
  image_with_copy(~custom,~Logical,~original,context,limits,index,count,depth,input,model,tiles,weights,bias,owner,ready,length_exact,evidence,
    DenseSources.copied(count,U32.from_nat(Nat.mul(U32.to_nat(index),DenseSources.image_size(~Logical,context))),0,input,model,input_ready(~Logical,~original,context,limits,index,count,input,tiles,weights,bias,evidence),ready))

def batch_image(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+index: U32,+count: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +evidence: ImageInputs<Logical,original,context,limits,index,count,input,tiles,weights,bias>) -> ImageResult<Logical,original,custom,context,index,count,depth,input,tiles,bias>:
  with_allocation(~custom,~Logical,~original,context,limits,index,count,depth,input,tiles,weights,bias,evidence,
    DenseSources.image_source_allocated(count,limits,count_within(~Logical,~original,context,limits,index,count,input,tiles,weights,bias,evidence)))

# One source contract supplies every actual batch image's input obligations.
# Image membership produces the copy bounds; no image result is assumed.
def dimensions(~Logical: Data,context: BandConvolutionValues.Context<Logical>) -> KernelShape.Dimensions:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  KernelShape.Dimensions{channels,outputs,height,width,kernel,stride,padding}

def valid_size(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+image: Nat) ->
  {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,image)) ==
    KernelShape.valid(dimensions(~Logical,context),DenseSources.image_size(~Logical,context),ConvSpec.length(Guards.weights(DenseSources.image_problem(~Logical,~original,context,0n))),ConvSpec.length(Guards.bias(DenseSources.image_problem(~Logical,~original,context,0n)))) : Bool}:
  BandConvolutionValues.Context{+logical,+activation,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias} = context
  +context = {BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} : BandConvolutionValues.Context<Logical>}
  Equal.cong(Nat,Bool,length => KernelShape.valid(dimensions(~Logical,context),length,ConvSpec.length(weights),ConvSpec.length(bias)),
    ConvSpec.length(Guards.input(DenseSources.image_problem(~Logical,~original,context,image))),DenseSources.image_size(~Logical,context),DenseSources.image_length(~Logical,~original,context,image))

def supported_size(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+image: Nat) ->
  {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,image)) ==
    KernelShape.supports_packed(limits,dimensions(~Logical,context),DenseSources.image_size(~Logical,context),ConvSpec.length(Guards.weights(DenseSources.image_problem(~Logical,~original,context,0n)))) : Bool}:
  BandConvolutionValues.Context{+logical,+activation,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias} = context
  +context = {BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} : BandConvolutionValues.Context<Logical>}
  Equal.cong(Nat,Bool,length => KernelShape.supports_packed(limits,dimensions(~Logical,context),length,ConvSpec.length(weights)),
    ConvSpec.length(Guards.input(DenseSources.image_problem(~Logical,~original,context,image))),DenseSources.image_size(~Logical,context),DenseSources.image_length(~Logical,~original,context,image))

def valid_image(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+image: Nat,+valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool}) ->
  {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,image)) == True{} : Bool}:
  Equal.trans(Bool,ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,image)),ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)),True{},
    Equal.trans(Bool,ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,image)),
      KernelShape.valid(dimensions(~Logical,context),DenseSources.image_size(~Logical,context),ConvSpec.length(Guards.weights(DenseSources.image_problem(~Logical,~original,context,0n))),ConvSpec.length(Guards.bias(DenseSources.image_problem(~Logical,~original,context,0n)))),ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)),valid_size(~Logical,~original,context,image),
      Equal.sym(Bool,ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)),KernelShape.valid(dimensions(~Logical,context),DenseSources.image_size(~Logical,context),ConvSpec.length(Guards.weights(DenseSources.image_problem(~Logical,~original,context,0n))),ConvSpec.length(Guards.bias(DenseSources.image_problem(~Logical,~original,context,0n)))),valid_size(~Logical,~original,context,0n))),valid)

def supported_image(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+image: Nat,+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool}) ->
  {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,image)) == True{} : Bool}:
  Equal.trans(Bool,NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,image)),NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,0n)),True{},
    Equal.trans(Bool,NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,image)),
      KernelShape.supports_packed(limits,dimensions(~Logical,context),DenseSources.image_size(~Logical,context),ConvSpec.length(Guards.weights(DenseSources.image_problem(~Logical,~original,context,0n)))),NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,0n)),supported_size(~Logical,~original,context,limits,image),
      Equal.sym(Bool,NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,0n)),KernelShape.supports_packed(limits,dimensions(~Logical,context),DenseSources.image_size(~Logical,context),ConvSpec.length(Guards.weights(DenseSources.image_problem(~Logical,~original,context,0n)))),supported_size(~Logical,~original,context,limits,0n))),supported)

def weight_image(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+image: Nat) ->
  {Guards.weights(DenseSources.image_problem(~Logical,~original,context,0n)) == Guards.weights(DenseSources.image_problem(~Logical,~original,context,image)) : List<&2,F32>}:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  {==}

def bias_image(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+image: Nat) ->
  {Guards.bias(DenseSources.image_problem(~Logical,~original,context,0n)) == Guards.bias(DenseSources.image_problem(~Logical,~original,context,image)) : List<&2,F32>}:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  {==}

def reordered_image(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+image: Nat,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,
  +before: {Buffer.tree(~Tile.Tile8,tiles) == WeightsLaws.run(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,context,0n)),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,context,0n)),Buffer.tree(~F32,weights)) : Tree.Tree<Tile.Tile8>}) ->
  {Buffer.tree(~Tile.Tile8,tiles) == WeightsLaws.run(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,context,image)),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,context,image)),Buffer.tree(~F32,weights)) : Tree.Tree<Tile.Tile8>}:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,ws,bs} = context
  before

def size_room(+images: U32,+index: U32,+size: Nat,+limit: U32,+inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(images)) == True{} : Bool},+whole: {Nat.is_le(Nat.mul(U32.to_nat(images),size),U32.to_nat(limit)) == True{} : Bool}) ->
  {Nat.is_le(size,U32.to_nat(limit)) == True{} : Bool}:
  Order.transitive(size,Nat.mul(U32.to_nat(images),size),U32.to_nat(limit),
    Padded.product_has_right(U32.to_nat(images),size,Order.le_lt_trans(0n,U32.to_nat(index),U32.to_nat(images),Order.zero_le(U32.to_nat(index)),inside)),whole)

def source_room(+images: U32,+index: U32,+size: Nat,+length: Nat,+inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(images)) == True{} : Bool},+exact: {length == Nat.mul(U32.to_nat(images),size) : Nat}) ->
  {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(index),size),size),length) == True{} : Bool}:
  %Equal.sym(Nat,length,Nat.mul(U32.to_nat(images),size),exact) : {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(index),size),size),_) == True{} : Bool}
  %Order.commutative(size,Nat.mul(U32.to_nat(index),size)) : {Nat.is_le(_,Nat.mul(U32.to_nat(images),size)) == True{} : Bool}
  RowMajor.mul_le(1n+U32.to_nat(index),U32.to_nat(images),size,Order.strict_follow(U32.to_nat(index),U32.to_nat(images),inside))

type Sources<-Logical: Data,-original: Logical -> U32 -> F32,-context: BandConvolutionValues.Context<Logical>,-limits: Machine.Limits,-images: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-weights: Buffer.Model<F32>,-bias: Buffer.Model<F32>> is Data:
  Sources{length_exact: {Buffer.length(~F32,input) == Nat.mul(U32.to_nat(images),DenseSources.image_size(~Logical,context)) : Nat},
    whole_room: {Nat.is_le(Nat.mul(U32.to_nat(images),DenseSources.image_size(~Logical,context)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    source_values: Dense.Values(~F32,~Logical,~original,DenseSources.logical_source(~Logical,context),Buffer.content(~F32,input)),source_ready: Buffer.Ready<F32,input>,
    weight_values: {Tree.read_sequence(~F32,ConvSpec.length(Guards.weights(DenseSources.image_problem(~Logical,~original,context,0n))),0,Buffer.tree(~F32,weights)) == Guards.weights(DenseSources.image_problem(~Logical,~original,context,0n)) : List<&2,F32>},
    bias_values: {Tree.read_sequence(~F32,ConvSpec.length(Guards.bias(DenseSources.image_problem(~Logical,~original,context,0n))),0,Buffer.tree(~F32,bias)) == Guards.bias(DenseSources.image_problem(~Logical,~original,context,0n)) : List<&2,F32>},
    weight_room: {Nat.is_le(ConvSpec.length(Guards.weights(DenseSources.image_problem(~Logical,~original,context,0n))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,weights)))) == True{} : Bool},
    bias_room: {Nat.is_le(ConvSpec.length(Guards.bias(DenseSources.image_problem(~Logical,~original,context,0n))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,bias)))) == True{} : Bool},
    reordered: {Buffer.tree(~Tile.Tile8,tiles) == WeightsLaws.run(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,context,0n)),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,context,0n)),Buffer.tree(~F32,weights)) : Tree.Tree<Tile.Tile8>},
    valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool}}

def inputs(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+index: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  evidence: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>,+inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(images)) == True{} : Bool}) ->
  ImageInputs<Logical,original,context,limits,index,U32.from_nat(DenseSources.image_size(~Logical,context)),input,tiles,weights,bias>:
  Sources{+length_exact,+whole_room,+source_values,+source_ready,+weight_values,+bias_values,+weight_room,+bias_room,+reordered,+valid,+supported} = evidence
  +size = DenseSources.image_size(~Logical,context)
  +count = U32.from_nat(size)
  +image = U32.to_nat(index)
  +bounded = size_room(images,index,size,Machine.capacity(limits),inside,whole_room)
  +exact = Bounds.roundtrip(size,Machine.capacity(limits),Limits.capacity_fits(limits),bounded)
  ImageInputs{exact,
    %Equal.sym(Bool,U32.is_le(count,Machine.capacity(limits)),Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))),Comparison.le(count,Machine.capacity(limits))) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(count),size,exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} bounded,
    source_values,source_ready,
    %Equal.sym(Nat,U32.to_nat(count),size,exact) : {Nat.is_le(Nat.add(Nat.mul(image,size),_),Buffer.length(~F32,input)) == True{} : Bool}
    source_room(images,index,size,Buffer.length(~F32,input),inside,length_exact),
    %weight_image(~Logical,~original,context,image) : {Tree.read_sequence(~F32,ConvSpec.length(_),0,Buffer.tree(~F32,weights)) == _ : List<&2,F32>} weight_values,
    %bias_image(~Logical,~original,context,image) : {Tree.read_sequence(~F32,ConvSpec.length(_),0,Buffer.tree(~F32,bias)) == _ : List<&2,F32>} bias_values,
    %weight_image(~Logical,~original,context,image) : {Nat.is_le(ConvSpec.length(_),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,weights)))) == True{} : Bool} weight_room,
    %bias_image(~Logical,~original,context,image) : {Nat.is_le(ConvSpec.length(_),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,bias)))) == True{} : Bool} bias_room,
    reordered_image(~Logical,~original,context,image,tiles,weights,reordered),valid_image(~Logical,~original,context,image,valid),supported_image(~Logical,~original,context,limits,image,supported)}

# Actual image cursors and stores are bounded by the whole output owner.
# The selected kernel depth does not occur in these allocation obligations.
def natural_room(+images: U32,+image: U32,+volume: Nat,+output: Buffer.Model<F32>,ready: Buffer.Ready<F32,output>,
  +length_exact: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),volume) : Nat},
  +inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),volume),volume),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool}:
  Buffer.Ready{depth_ok,+room} = ready
  Order.transitive(Nat.add(Nat.mul(U32.to_nat(image),volume),volume),Buffer.length(~F32,output),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output))),
    source_room(images,image,volume,Buffer.length(~F32,output),inside,length_exact),room)

def cursor_exact(+image: U32,+size: U32,+volume: Nat,+output: Buffer.Model<F32>,ready: Buffer.Ready<F32,output>,
  +size_exact: {U32.to_nat(size) == volume : Nat},
  +room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),volume),volume),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool}) ->
  {U32.to_nat((image * size : U32)) == Nat.mul(U32.to_nat(image),volume) : Nat}:
  Buffer.Ready{+depth_ok,allocation_room} = ready
  %Equal.sym(U32,(image * size : U32),U32.from_nat(Nat.mul(U32.to_nat(image),volume)),offset_matches(image,size,volume,size_exact)) :
    {U32.to_nat(_) == Nat.mul(U32.to_nat(image),volume) : Nat}
  Bounds.roundtrip(Nat.mul(U32.to_nat(image),volume),Tree.capacity(F32,Buffer.tree(~F32,output)),Buffer.capacity_bound(~F32,output,depth_ok),
    Order.transitive(Nat.mul(U32.to_nat(image),volume),Nat.add(Nat.mul(U32.to_nat(image),volume),volume),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output))),
      Order.le_add(Nat.mul(U32.to_nat(image),volume),volume),room))

type StoreBounds<-image: U32,-size: U32,-volume: Nat,-output: Buffer.Model<F32>> is Data:
  StoreBounds{cursor: {U32.to_nat((image * size : U32)) == Nat.mul(U32.to_nat(image),volume) : Nat},
    natural: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),volume),volume),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool},
    store: {Nat.is_le(Nat.add(U32.to_nat((image * size : U32)),U32.to_nat(size)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool}}

def prepared(+images: U32,+image: U32,+size: U32,+volume: Nat,+output: Buffer.Model<F32>,+ready: Buffer.Ready<F32,output>,
  +size_exact: {U32.to_nat(size) == volume : Nat},
  +length_exact: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),volume) : Nat},
  +inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool}) -> StoreBounds<image,size,volume,output>:
  +room = natural_room(images,image,volume,output,ready,length_exact,inside)
  +cursor = cursor_exact(image,size,volume,output,ready,size_exact,room)
  StoreBounds{cursor,room,
    %Equal.sym(Nat,U32.to_nat((image * size : U32)),Nat.mul(U32.to_nat(image),volume),cursor) :
      {Nat.is_le(Nat.add(_,U32.to_nat(size)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(size),volume,size_exact) :
      {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),volume),_),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool}
    room}

def query_room(+image: U32,+volume: Nat,+output: Buffer.Model<F32>,+index: Nat,
  +room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),volume),volume),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool},
  +inside: {Nat.is_lt(index,volume) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),volume),index),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool}:
  Bounds.strict_le(Nat.add(Nat.mul(U32.to_nat(image),volume),index),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output))),
    Bounds.nat_trans(Nat.add(Nat.mul(U32.to_nat(image),volume),index),Nat.add(Nat.mul(U32.to_nat(image),volume),volume),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output))),
      Order.offset_lt(Nat.mul(U32.to_nat(image),volume),index,volume,inside),room))

# A packed image's finite activated reference is the corresponding image
# interval of the public NCHW reference. Copy readback uses the actual cursor.
def reference_volume(~Logical: Data,context: BandConvolutionValues.Context<Logical>) -> Nat:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  Nat.mul(outputs,BandConvolutionValues.plane(height,width,kernel,stride,padding))

def count_exact(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+image: Nat,
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,image)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,image)) == True{} : Bool}) ->
  {U32.to_nat(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,image))) == reference_volume(~Logical,context) : Nat}:
  BandConvolutionValues.Context{+logical,+activation,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias} = context
  PreparedOutputReadback.count_exact(limits,BandConvolutionValues.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),valid,supported)

def original_matches(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+image: Nat,+index: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),reference_volume(~Logical,context)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(image,reference_volume(~Logical,context)),U32.to_nat(index)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {DenseConvolution.original(~custom,DenseConvolution.Logical{DenseSources.image_problem(~Logical,~original,context,image),image_activation(~Logical,context)},index) ==
    BandConvolutionValues.value(~custom,~Logical,~original,context,U32.from_nat(Nat.add(Nat.mul(image,reference_volume(~Logical,context)),U32.to_nat(index)))) : F32}:
  BandConvolutionValues.Context{+logical,+activation,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias} = context
  +problem = BandConvolutionValues.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)
  +flat = U32.to_nat(index)
  +area = BandConvolutionValues.plane(height,width,kernel,stride,padding)
  +channel = Nat.div(flat,area)
  +position = Nat.mod(flat,area)
  +positive = Inference.spatial_positive(problem)
  +channel_inside = RowMajor.cancel_multiply_lt(channel,outputs,area,
    Order.le_lt_trans(Nat.mul(channel,area),flat,Nat.mul(outputs,area),DivisionOrder.product_below(flat,area,positive),inside))
  +position_inside = DivisionOrder.remainder_small(flat,area,positive)
  +global_index = Nat.add(Nat.mul(image,Nat.mul(outputs,area)),flat)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(global_index)),global_index,Bounds.roundtrip(global_index,capacity,capacity_ok,room)) :
    {BandConvolutionValues.at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,position) ==
      BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,_) : F32}
  %Equal.sym(Nat,flat,Nat.add(Nat.mul(channel,area),position),DivisionOrder.quotient_exact(flat,area,positive)) :
    {BandConvolutionValues.at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,position) ==
      BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,Nat.add(Nat.mul(image,Nat.mul(outputs,area)),_)) : F32}
  Equal.sym(F32,BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,Nat.add(Nat.mul(image,Nat.mul(outputs,area)),Nat.add(Nat.mul(channel,area),position))),
    BandConvolutionValues.at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,position),
    BandConvolutionValues.decoded(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,position,channel_inside,position_inside))

def local_exact(+model: Buffer.Model<F32>,+index: Nat,ready: Buffer.Ready<F32,model>,+inside: {Nat.is_lt(index,Buffer.length(~F32,model)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(index)) == index : Nat}:
  Buffer.Ready{+depth_ok,+room} = ready
  Bounds.roundtrip(index,Tree.capacity(F32,Buffer.tree(~F32,model)),Buffer.capacity_bound(~F32,model,depth_ok),
    Bounds.strict_le(index,U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,model))),Bounds.nat_trans(index,Buffer.length(~F32,model),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,model))),inside,room)))

def destination_query(+image: U32,+size: U32,+count: Nat,+index: Nat,+exact: {U32.to_nat(size) == count : Nat}) ->
  {((image * size : U32) + U32.from_nat(index) : U32) == U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),count),index)) : U32}:
  Equal.trans(U32,((image * size : U32) + U32.from_nat(index) : U32),
    (U32.from_nat(Nat.mul(U32.to_nat(image),count)) + U32.from_nat(index) : U32),U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),count),index)),
    Equal.cong(U32,U32,offset => (offset + U32.from_nat(index) : U32),(image * size : U32),U32.from_nat(Nat.mul(U32.to_nat(image),count)),offset_matches(image,size,count,exact)),
    WordArithmetic.native_add(Nat.mul(U32.to_nat(image),count),index))

def copied_point(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+image: U32,+result: Buffer.Model<F32>,+output: Buffer.Model<F32>,+index: Nat,
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) == True{} : Bool},
  +values: Dense.Values(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image)),image_activation(~Logical,context)},Buffer.content(~F32,result)),
  +result_ready: Buffer.Ready<F32,result>,output_ready: Buffer.Ready<F32,output>,
  +length_exact: {Buffer.length(~F32,result) == U32.to_nat(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image)))) : Nat},
  +copy_room: {Nat.is_le(Nat.add(U32.to_nat((image * PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) : U32)),U32.to_nat(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool},
  +global_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),index),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool},
  +inside: {Nat.is_lt(index,Buffer.length(~F32,result)) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),index)),
      Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))),0,(image * PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) : U32),Buffer.pack(~F32,result),Buffer.pack(~F32,output))))) ==
    BandConvolutionValues.value(~custom,~Logical,~original,context,U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),index))) : F32}:
  Buffer.Ready{+depth_ok,+output_room} = output_ready
  +problem = DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))
  +size = PreparedOutputReadback.count(problem)
  +destination = (image * size : U32)
  +volume_exact = count_exact(~Logical,~original,context,limits,U32.to_nat(image),valid,supported)
  +index_exact = local_exact(result,index,result_ready,inside)
  %destination_query(image,size,reference_volume(~Logical,context),index,volume_exact) :
    {Buffer.value(~F32,_,Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,size,0,destination,Buffer.pack(~F32,result),Buffer.pack(~F32,output))))) ==
      BandConvolutionValues.value(~custom,~Logical,~original,context,_) : F32}
  Equal.trans(F32,Buffer.value(~F32,(destination + U32.from_nat(index) : U32),Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,size,0,destination,Buffer.pack(~F32,result),Buffer.pack(~F32,output))))),
    DenseConvolution.original(~custom,DenseConvolution.Logical{problem,image_activation(~Logical,context)},U32.from_nat(index)),BandConvolutionValues.value(~custom,~Logical,~original,context,(destination + U32.from_nat(index) : U32)),
    Equal.trans(F32,Buffer.value(~F32,(destination + U32.from_nat(index) : U32),Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,size,0,destination,Buffer.pack(~F32,result),Buffer.pack(~F32,output))))),
      Buffer.value(~F32,U32.from_nat(index),Buffer.content(~F32,result)),DenseConvolution.original(~custom,DenseConvolution.Logical{problem,image_activation(~Logical,context)},U32.from_nat(index)),
      Equal.trans(F32,Buffer.value(~F32,(destination + U32.from_nat(index) : U32),Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,size,0,destination,Buffer.pack(~F32,result),Buffer.pack(~F32,output))))),Tree.read(F32,Buffer.tree(~F32,result),(0 + U32.from_nat(index) : U32)),Buffer.value(~F32,U32.from_nat(index),Buffer.content(~F32,result)),
        Batch.copy_inside(size,0,destination,result,output,index,Buffer.Ready{depth_ok,output_room},copy_room,
          %length_exact : {Nat.is_lt(index,_) == True{} : Bool} inside),
        %Equal.sym(U32,(0 + U32.from_nat(index) : U32),U32.from_nat(index),Arithmetic.add_zero_left(U32.from_nat(index))) :
          {Tree.read(F32,Buffer.tree(~F32,result),_) == Buffer.value(~F32,U32.from_nat(index),Buffer.content(~F32,result)) : F32}
        DenseSources.model_read(U32.from_nat(index),result)),
      Dense.read(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{problem,image_activation(~Logical,context)},Buffer.content(~F32,result),index,values,
        %Equal.sym(Nat,Buffer.content_length(~F32,Buffer.content(~F32,result)),Buffer.length(~F32,result),DenseSources.image_copy_model_length(result)) : {Nat.is_lt(index,_) == True{} : Bool} inside)),
    %Equal.sym(U32,(destination + U32.from_nat(index) : U32),U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),index)),destination_query(image,size,reference_volume(~Logical,context),index,volume_exact)) :
      {DenseConvolution.original(~custom,DenseConvolution.Logical{problem,image_activation(~Logical,context)},U32.from_nat(index)) == BandConvolutionValues.value(~custom,~Logical,~original,context,_) : F32}
    %index_exact :
      {DenseConvolution.original(~custom,DenseConvolution.Logical{problem,image_activation(~Logical,context)},U32.from_nat(index)) == BandConvolutionValues.value(~custom,~Logical,~original,context,U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),_))) : F32}
    original_matches(~custom,~Logical,~original,context,U32.to_nat(image),U32.from_nat(index),Tree.capacity(F32,Buffer.tree(~F32,output)),Buffer.capacity_bound(~F32,output,depth_ok),
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,index_exact) : {Nat.is_lt(_,reference_volume(~Logical,context)) == True{} : Bool}
      %volume_exact : {Nat.is_lt(index,_) == True{} : Bool}
      %length_exact : {Nat.is_lt(index,_) == True{} : Bool} inside,
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,index_exact) : {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),_),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool}
      global_room))

# An actual image copy appends one public NCHW image interval while keeping
# every already completed output element. All witnesses are finite ranges.
def Values(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+content: Buffer.Content<F32>,+completed: Nat) -> Data:
  Members.All(Nat,{index => Dense.At(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,content,index) : Nat -> Data},Ranges.range(completed,0n))

def read(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+content: Buffer.Content<F32>,+completed: Nat,+index: Nat,
  values: Values(~custom,~Logical,~original,context,content,completed),inside: {Nat.is_lt(index,completed) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(index),content) == BandConvolutionValues.value(~custom,~Logical,~original,context,U32.from_nat(index)) : F32}:
  Members.all_member(Nat,{index => Dense.At(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,content,index) : Nat -> Data},index,Ranges.range(completed,0n),values,Members.range_member(completed,0n,index,inside))

def copied_content(+size: U32,+image: U32,+result: Buffer.Model<F32>,+output: Buffer.Model<F32>) -> Buffer.Content<F32>:
  Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,size,0,(image * size : U32),Buffer.pack(~F32,result),Buffer.pack(~F32,output))))

def earlier_point(+size: U32,+destination: U32,+result: Buffer.Model<F32>,+output: Buffer.Model<F32>,+index: Nat,+ready: Buffer.Ready<F32,output>,
  +room: {Nat.is_le(Nat.add(U32.to_nat(destination),U32.to_nat(size)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool},
  +inside: {Nat.is_lt(index,Buffer.length(~F32,output)) == True{} : Bool},
  +before: {Nat.is_lt(index,U32.to_nat(destination)) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(index),Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,size,0,destination,Buffer.pack(~F32,result),Buffer.pack(~F32,output))))) == Buffer.value(~F32,U32.from_nat(index),Buffer.content(~F32,output)) : F32}:
  Buffer.Ready{+depth_ok,+allocation_room} = ready
  +exact = local_exact(output,index,Buffer.Ready{depth_ok,allocation_room},inside)
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,size,0,destination,Buffer.pack(~F32,result),Buffer.pack(~F32,output)))),Batch.copied_content(size,0,destination,result,output),Batch.copy_output_contents(size,0,destination,result,output)) :
    {Buffer.value(~F32,U32.from_nat(index),_) == Buffer.value(~F32,U32.from_nat(index),Buffer.content(~F32,output)) : F32}
  Equal.trans(F32,Buffer.value(~F32,U32.from_nat(index),Batch.copied_content(size,0,destination,result,output)),Tree.read(F32,Buffer.tree(~F32,output),U32.from_nat(index)),Buffer.value(~F32,U32.from_nat(index),Buffer.content(~F32,output)),
    CopyReadback.read_outside(~F32,U32.to_nat(size),0n,U32.from_nat(index),Buffer.tree(~F32,result),Buffer.tree(~F32,output),0,destination,Buffer.depth(~F32,output),Buffer.balanced(~F32,output),depth_ok,Buffer.capacity_bound(~F32,output,depth_ok),
      %Equal.sym(Nat,Nat.add(U32.to_nat(destination),0n),U32.to_nat(destination),Addition.zero_right(U32.to_nat(destination))) :
        {Nat.is_le(Nat.add(_,U32.to_nat(size)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool} room,
      %Equal.sym(Bool,U32.is_lt(U32.from_nat(index),Tree.capacity(F32,Buffer.tree(~F32,output))),Nat.is_lt(U32.to_nat(U32.from_nat(index)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))),Comparison.lt(U32.from_nat(index),Tree.capacity(F32,Buffer.tree(~F32,output)))) : {_ == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,exact) : {Nat.is_lt(_,U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool}
      Bounds.nat_trans(index,Buffer.length(~F32,output),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output))),inside,allocation_room),
      Interval.Before{
        %Equal.sym(Nat,Nat.add(U32.to_nat(destination),0n),U32.to_nat(destination),Addition.zero_right(U32.to_nat(destination))) : {Nat.is_lt(U32.to_nat(U32.from_nat(index)),_) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,exact) : {Nat.is_lt(_,U32.to_nat(destination)) == True{} : Bool} before}),
    DenseSources.model_read(U32.from_nat(index),output))

def retained_range(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+size: U32,+image: U32,+volume: Nat,+result: Buffer.Model<F32>,+output: Buffer.Model<F32>,remaining: Nat,+start: Nat,
  +before: Values(~custom,~Logical,~original,context,Buffer.content(~F32,output),Nat.mul(U32.to_nat(image),volume)),+ready: Buffer.Ready<F32,output>,
  +cursor: {U32.to_nat((image * size : U32)) == Nat.mul(U32.to_nat(image),volume) : Nat},
  +store_room: {Nat.is_le(Nat.add(U32.to_nat((image * size : U32)),U32.to_nat(size)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool},
  +prefix_room: {Nat.is_le(Nat.mul(U32.to_nat(image),volume),Buffer.length(~F32,output)) == True{} : Bool},
  +range_room: {Nat.is_le(Nat.add(start,remaining),Nat.mul(U32.to_nat(image),volume)) == True{} : Bool}) ->
  Members.All(Nat,{index => Dense.At(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,copied_content(size,image,result,output),index) : Nat -> Data},Ranges.range(remaining,start)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +inside = Order.room_index(start,rest,Nat.mul(U32.to_nat(image),volume),range_room)
      Shape.Both{
        Equal.trans(F32,Buffer.value(~F32,U32.from_nat(start),copied_content(size,image,result,output)),Buffer.value(~F32,U32.from_nat(start),Buffer.content(~F32,output)),BandConvolutionValues.value(~custom,~Logical,~original,context,U32.from_nat(start)),
          earlier_point(size,(image * size : U32),result,output,start,ready,store_room,Bounds.nat_trans(start,Nat.mul(U32.to_nat(image),volume),Buffer.length(~F32,output),inside,prefix_room),
            %Equal.sym(Nat,U32.to_nat((image * size : U32)),Nat.mul(U32.to_nat(image),volume),cursor) : {Nat.is_lt(start,_) == True{} : Bool} inside),
          read(~custom,~Logical,~original,context,Buffer.content(~F32,output),Nat.mul(U32.to_nat(image),volume),start,before,inside)),
        retained_range(~custom,~Logical,~original,context,size,image,volume,result,output,rest,1n+start,before,ready,cursor,store_room,prefix_room,Order.advance_room(start,rest,Nat.mul(U32.to_nat(image),volume),range_room))}

def image_range(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+image: U32,+result: Buffer.Model<F32>,+output: Buffer.Model<F32>,remaining: Nat,+start: Nat,
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) == True{} : Bool},
  +values: Dense.Values(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image)),image_activation(~Logical,context)},Buffer.content(~F32,result)),
  +result_ready: Buffer.Ready<F32,result>,+output_ready: Buffer.Ready<F32,output>,
  +length_exact: {Buffer.length(~F32,result) == U32.to_nat(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image)))) : Nat},
  +bounds: StoreBounds<image,PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))),reference_volume(~Logical,context),output>,
  +range_room: {Nat.is_le(Nat.add(start,remaining),Buffer.length(~F32,result)) == True{} : Bool}) ->
  Members.All(Nat,{index => Dense.At(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,copied_content(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))),image,result,output),index) : Nat -> Data},Ranges.range(remaining,Nat.add(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),start))):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      StoreBounds{cursor,+natural,+store} = bounds
      +inside = Order.room_index(start,rest,Buffer.length(~F32,result),range_room)
      +volume_inside = {
        %count_exact(~Logical,~original,context,limits,U32.to_nat(image),valid,supported) : {Nat.is_lt(start,_) == True{} : Bool}
        %length_exact : {Nat.is_lt(start,_) == True{} : Bool} inside : {Nat.is_lt(start,reference_volume(~Logical,context)) == True{} : Bool}}
      Shape.Both{
        copied_point(~custom,~Logical,~original,context,limits,image,result,output,start,valid,supported,values,result_ready,output_ready,length_exact,store,query_room(image,reference_volume(~Logical,context),output,start,natural,volume_inside),inside),
        %Addition.successor_right(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),start) :
          Members.All(Nat,{index => Dense.At(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,copied_content(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))),image,result,output),index) : Nat -> Data},Ranges.range(rest,_))
        image_range(~custom,~Logical,~original,context,limits,image,result,output,rest,1n+start,valid,supported,values,result_ready,output_ready,length_exact,bounds,Order.advance_room(start,rest,Buffer.length(~F32,result),range_room))}

def copied_prefix_parts(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+image: U32,+result: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +before: Values(~custom,~Logical,~original,context,Buffer.content(~F32,output),Nat.mul(U32.to_nat(image),reference_volume(~Logical,context))),
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) == True{} : Bool},
  +values: Dense.Values(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image)),image_activation(~Logical,context)},Buffer.content(~F32,result)),
  +result_ready: Buffer.Ready<F32,result>,+output_ready: Buffer.Ready<F32,output>,
  +length_exact: {Buffer.length(~F32,result) == U32.to_nat(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image)))) : Nat},
  +output_length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat},
  +inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},
  +bounds: StoreBounds<image,PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))),reference_volume(~Logical,context),output>,
  +result_length: {Buffer.length(~F32,result) == reference_volume(~Logical,context) : Nat}) ->
  Values(~custom,~Logical,~original,context,copied_content(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))),image,result,output),Nat.add(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),reference_volume(~Logical,context))):
  StoreBounds{+cursor,+natural,+store} = bounds
  +volume = reference_volume(~Logical,context)
  +completed = Nat.mul(U32.to_nat(image),volume)
  +size = PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image)))
  +content = copied_content(size,image,result,output)
  +prefix_room = Order.transitive(completed,Nat.add(completed,volume),Buffer.length(~F32,output),Order.le_add(completed,volume),source_room(images,image,volume,Buffer.length(~F32,output),inside,output_length))
  %Equal.sym(List<&2,Nat>,Ranges.range(Nat.add(completed,volume),0n),
    OrderedMatrix.cat(Nat,Ranges.range(completed,0n),Ranges.range(volume,completed)),Ranges.split(completed,volume,0n)) :
    Members.All(Nat,{index => Dense.At(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,content,index) : Nat -> Data},_)
  Members.all_concatenated(Nat,{index => Dense.At(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,content,index) : Nat -> Data},Ranges.range(completed,0n),Ranges.range(volume,completed),
    retained_range(~custom,~Logical,~original,context,size,image,volume,result,output,completed,0n,before,output_ready,cursor,store,prefix_room,Order.reflexive(completed)),
    %Addition.zero_right(completed) :
      Members.All(Nat,{index => Dense.At(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,content,index) : Nat -> Data},Ranges.range(volume,_))
    image_range(~custom,~Logical,~original,context,limits,image,result,output,volume,0n,valid,supported,values,result_ready,output_ready,length_exact,bounds,
      %Equal.sym(Nat,Buffer.length(~F32,result),volume,result_length) : {Nat.is_le(volume,_) == True{} : Bool} Order.reflexive(volume)))

def copied_prefix(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+image: U32,+result: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +before: Values(~custom,~Logical,~original,context,Buffer.content(~F32,output),Nat.mul(U32.to_nat(image),reference_volume(~Logical,context))),
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) == True{} : Bool},
  +values: Dense.Values(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image)),image_activation(~Logical,context)},Buffer.content(~F32,result)),
  +result_ready: Buffer.Ready<F32,result>,+output_ready: Buffer.Ready<F32,output>,
  +length_exact: {Buffer.length(~F32,result) == U32.to_nat(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image)))) : Nat},
  +output_length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat},
  +inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool}) ->
  Values(~custom,~Logical,~original,context,copied_content(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))),image,result,output),Nat.add(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),reference_volume(~Logical,context))):
  +volume = reference_volume(~Logical,context)
  +size = PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image)))
  +size_exact = count_exact(~Logical,~original,context,limits,U32.to_nat(image),valid,supported)
  +result_length = Equal.trans(Nat,Buffer.length(~F32,result),U32.to_nat(size),volume,length_exact,size_exact)
  +bounds = prepared(images,image,size,volume,output,output_ready,size_exact,output_length,inside)
  copied_prefix_parts(~custom,~Logical,~original,context,limits,images,image,result,output,before,valid,supported,values,result_ready,output_ready,length_exact,output_length,inside,bounds,result_length)

# Every image uses the same genuine work and output count. Logical batch
# membership, together with the actual input bound, prevents cursor wrapping.
def work_image(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+image: Nat) ->
  {Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)) == Inference.work(DenseSources.image_problem(~Logical,~original,context,image)) : Pipeline.Work}:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  {==}

def output_size(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+image: Nat) ->
  {PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,0n)) == PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,image)) : U32}:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  {==}

def input_positive(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool}) ->
  {Nat.is_lt(0n,DenseSources.image_size(~Logical,context)) == True{} : Bool}:
  %DenseSources.image_length(~Logical,~original,context,0n) : {Nat.is_lt(0n,_) == True{} : Bool}
  Support.input_positive(DenseSources.image_problem(~Logical,~original,context,0n),Guards.valid(DenseSources.image_problem(~Logical,~original,context,0n),valid))

def geometry_count_room(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool},
  +whole_room: {Nat.is_le(Nat.mul(U32.to_nat(images),DenseSources.image_size(~Logical,context)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(images),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(U32.to_nat(images),Nat.mul(U32.to_nat(images),DenseSources.image_size(~Logical,context)),U32.to_nat(Machine.capacity(limits)),
    RowMajor.product_covers(U32.to_nat(images),DenseSources.image_size(~Logical,context),input_positive(~Logical,~original,context,valid)),whole_room)

def next_exact(+limits: Machine.Limits,+images: U32,+index: U32,
  +geometry_count_room: {Nat.is_le(U32.to_nat(images),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(images)) == True{} : Bool}) ->
  {U32.to_nat((index + 1 : U32)) == 1n+U32.to_nat(index) : Nat}:
  Equal.trans(Nat,U32.to_nat((index + 1 : U32)),Nat.add(U32.to_nat(index),1n),1n+U32.to_nat(index),
    Arithmetic.add(index,1,Machine.capacity(limits),Limits.capacity_fits(limits),
      %Equal.sym(Nat,Nat.add(U32.to_nat(index),1n),1n+U32.to_nat(index),Addition.successor_width(U32.to_nat(index))) :
        {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      Order.transitive(1n+U32.to_nat(index),U32.to_nat(images),U32.to_nat(Machine.capacity(limits)),Order.strict_follow(U32.to_nat(index),U32.to_nat(images),inside),geometry_count_room)),
    Addition.successor_width(U32.to_nat(index)))

def geometry_next_remaining(+index: U32,+rest: Nat,+images: U32,
  +exact: {U32.to_nat((index + 1 : U32)) == 1n+U32.to_nat(index) : Nat},
  +remaining: {Nat.add(U32.to_nat(index),2n+rest) == U32.to_nat(images) : Nat}) ->
  {Nat.add(U32.to_nat((index + 1 : U32)),1n+rest) == U32.to_nat(images) : Nat}:
  Equal.trans(Nat,Nat.add(U32.to_nat((index + 1 : U32)),1n+rest),Nat.add(U32.to_nat(index),2n+rest),U32.to_nat(images),
    Equal.trans(Nat,Nat.add(U32.to_nat((index + 1 : U32)),1n+rest),1n+Nat.add(U32.to_nat(index),1n+rest),Nat.add(U32.to_nat(index),2n+rest),
      Equal.cong(Nat,Nat,cursor => Nat.add(cursor,1n+rest),U32.to_nat((index + 1 : U32)),1n+U32.to_nat(index),exact),
      Equal.sym(Nat,Nat.add(U32.to_nat(index),2n+rest),1n+Nat.add(U32.to_nat(index),1n+rest),Addition.successor_right(U32.to_nat(index),1n+rest))),remaining)

def completed_next(+index: U32,+volume: Nat,+exact: {U32.to_nat((index + 1 : U32)) == 1n+U32.to_nat(index) : Nat}) ->
  {Nat.add(Nat.mul(U32.to_nat(index),volume),volume) == Nat.mul(U32.to_nat((index + 1 : U32)),volume) : Nat}:
  %Equal.sym(Nat,U32.to_nat((index + 1 : U32)),1n+U32.to_nat(index),exact) : {Nat.add(Nat.mul(U32.to_nat(index),volume),volume) == Nat.mul(_,volume) : Nat}
  Order.commutative(Nat.mul(U32.to_nat(index),volume),volume)

# The batch's affine callbacks consume the actual observed image and clone
# owners. These equalities preserve the original zero/one/many procedures.
def one(~custom: F32 -> F32,+activation: Activations.Activation,+index: U32,+depth: U32,+work: Pipeline.Work,+input_size: U32,+output_size: U32,
  +input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,+source: Buffer.Model<F32>,+result: Buffer.Model<F32>,+target: Buffer.Model<F32>,
  +image_owners: {(Buffer.pack(~F32,source),Buffer.pack(~F32,result)) == TensorConvolution.image(~custom,activation,depth,work,index,input_size,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
  +copy_owners: {Buffer.pack(~F32,target) == TensorConvolution.copied_output(Copy.range(~F32,output_size,0,(index * output_size : U32),Buffer.pack(~F32,result),Buffer.pack(~F32,output))) : Storage.Buffer<F32>}) ->
  {Buffer.pack(~F32,target) == TensorConvolution.batch(~custom,activation,1n,index,depth,work,input_size,output_size,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias),Buffer.pack(~F32,output)) : Storage.Buffer<F32>}:
  %image_owners :
    {Buffer.pack(~F32,target) == Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~Storage.Buffer<F32>,_,source => result =>
      TensorConvolution.copied_output(Copy.range(~F32,output_size,0,(index * output_size : U32),result,Buffer.pack(~F32,output)))) : Storage.Buffer<F32>}
  copy_owners

def many(~custom: F32 -> F32,+activation: Activations.Activation,+rest: Nat,+index: U32,+depth: U32,+work: Pipeline.Work,+input_size: U32,+output_size: U32,
  +input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +left_tiles: Buffer.Model<Tile.Tile8>,+right_tiles: Buffer.Model<Tile.Tile8>,+left_bias: Buffer.Model<F32>,+right_bias: Buffer.Model<F32>,
  +source: Buffer.Model<F32>,+result: Buffer.Model<F32>,+target: Buffer.Model<F32>,+final: Buffer.Model<F32>,
  +tile_owners: {(Buffer.pack(~Tile.Tile8,left_tiles),Buffer.pack(~Tile.Tile8,right_tiles)) == Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>},
  +bias_owners: {(Buffer.pack(~F32,left_bias),Buffer.pack(~F32,right_bias)) == Clone.clone(~F32,Buffer.pack(~F32,bias)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
  +image_owners: {(Buffer.pack(~F32,source),Buffer.pack(~F32,result)) == TensorConvolution.image(~custom,activation,depth,work,index,input_size,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,left_tiles),Buffer.pack(~F32,left_bias)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
  +copy_owners: {Buffer.pack(~F32,target) == TensorConvolution.copied_output(Copy.range(~F32,output_size,0,(index * output_size : U32),Buffer.pack(~F32,result),Buffer.pack(~F32,output))) : Storage.Buffer<F32>},
  +remaining: {Buffer.pack(~F32,final) == TensorConvolution.batch(~custom,activation,1n+rest,(index + 1 : U32),depth,work,input_size,output_size,Buffer.pack(~F32,source),Buffer.pack(~Tile.Tile8,right_tiles),Buffer.pack(~F32,right_bias),Buffer.pack(~F32,target)) : Storage.Buffer<F32>}) ->
  {Buffer.pack(~F32,final) == TensorConvolution.batch(~custom,activation,2n+rest,index,depth,work,input_size,output_size,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias),Buffer.pack(~F32,output)) : Storage.Buffer<F32>}:
  %tile_owners :
    {Buffer.pack(~F32,final) == Storage.bind_pair(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,_,current_weights => remaining_weights =>
      Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~Storage.Buffer<F32>,Clone.clone(~F32,Buffer.pack(~F32,bias)),current_bias => remaining_bias =>
        Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~Storage.Buffer<F32>,TensorConvolution.image(~custom,activation,depth,work,index,input_size,Buffer.pack(~F32,input),current_weights,current_bias),source => result =>
          TensorConvolution.batch(~custom,activation,1n+rest,(index + 1 : U32),depth,work,input_size,output_size,source,remaining_weights,remaining_bias,
            TensorConvolution.copied_output(Copy.range(~F32,output_size,0,(index * output_size : U32),result,Buffer.pack(~F32,output))))))) : Storage.Buffer<F32>}
  %bias_owners :
    {Buffer.pack(~F32,final) == Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~Storage.Buffer<F32>,_,current_bias => remaining_bias =>
      Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~Storage.Buffer<F32>,TensorConvolution.image(~custom,activation,depth,work,index,input_size,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,left_tiles),current_bias),source => result =>
        TensorConvolution.batch(~custom,activation,1n+rest,(index + 1 : U32),depth,work,input_size,output_size,source,Buffer.pack(~Tile.Tile8,right_tiles),remaining_bias,
          TensorConvolution.copied_output(Copy.range(~F32,output_size,0,(index * output_size : U32),result,Buffer.pack(~F32,output)))))) : Storage.Buffer<F32>}
  %image_owners :
    {Buffer.pack(~F32,final) == Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~Storage.Buffer<F32>,_,source => result =>
      TensorConvolution.batch(~custom,activation,1n+rest,(index + 1 : U32),depth,work,input_size,output_size,source,Buffer.pack(~Tile.Tile8,right_tiles),Buffer.pack(~F32,right_bias),
        TensorConvolution.copied_output(Copy.range(~F32,output_size,0,(index * output_size : U32),result,Buffer.pack(~F32,output))))) : Storage.Buffer<F32>}
  %copy_owners :
    {Buffer.pack(~F32,final) == TensorConvolution.batch(~custom,activation,1n+rest,(index + 1 : U32),depth,work,input_size,output_size,Buffer.pack(~F32,source),Buffer.pack(~Tile.Tile8,right_tiles),Buffer.pack(~F32,right_bias),_) : Storage.Buffer<F32>}
  remaining

# Batch source contracts follow real content-preserving image and clone
# owners. Certificates may change; finite logical values and room do not.
def model_tree(model: Buffer.Model<F32>) -> {Buffer.tree(~F32,model) == Buffer.content_tree(~F32,Buffer.content(~F32,model)) : Tree.Tree<F32>}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def tree_equal(+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+contents: {Buffer.content(~F32,left) == Buffer.content(~F32,right) : Buffer.Content<F32>}) ->
  {Buffer.tree(~F32,left) == Buffer.tree(~F32,right) : Tree.Tree<F32>}:
  Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,left),Buffer.content_tree(~F32,Buffer.content(~F32,left)),Buffer.tree(~F32,right),model_tree(left),
    Equal.trans(Tree.Tree<F32>,Buffer.content_tree(~F32,Buffer.content(~F32,left)),Buffer.content_tree(~F32,Buffer.content(~F32,right)),Buffer.tree(~F32,right),
      Equal.cong(Buffer.Content<F32>,Tree.Tree<F32>,Buffer.content_tree(~F32),Buffer.content(~F32,left),Buffer.content(~F32,right),contents),
      Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,right),Buffer.content_tree(~F32,Buffer.content(~F32,right)),model_tree(right))))

def length_equal(+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+contents: {Buffer.content(~F32,left) == Buffer.content(~F32,right) : Buffer.Content<F32>}) ->
  {Buffer.length(~F32,left) == Buffer.length(~F32,right) : Nat}:
  Equal.trans(Nat,Buffer.length(~F32,left),Buffer.content_length(~F32,Buffer.content(~F32,left)),Buffer.length(~F32,right),
    Equal.sym(Nat,Buffer.content_length(~F32,Buffer.content(~F32,left)),Buffer.length(~F32,left),DenseSources.image_copy_model_length(left)),
    Equal.trans(Nat,Buffer.content_length(~F32,Buffer.content(~F32,left)),Buffer.content_length(~F32,Buffer.content(~F32,right)),Buffer.length(~F32,right),
      Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.content(~F32,left),Buffer.content(~F32,right),contents),DenseSources.image_copy_model_length(right)))

def input_changed(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+input: Buffer.Model<F32>,+next: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +retained: {Buffer.content(~F32,next) == Buffer.content(~F32,input) : Buffer.Content<F32>},evidence: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>) ->
  Sources<Logical,original,context,limits,images,next,tiles,weights,bias>:
  Sources{+length_exact,whole_room,source_values,source_ready,weight_values,bias_values,weight_room,bias_room,reordered,valid,supported} = evidence
  Sources{Equal.trans(Nat,Buffer.length(~F32,next),Buffer.length(~F32,input),Nat.mul(U32.to_nat(images),DenseSources.image_size(~Logical,context)),length_equal(next,input,retained),length_exact),whole_room,
    %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,next),Buffer.content(~F32,input),retained) : Dense.Values(~F32,~Logical,~original,DenseSources.logical_source(~Logical,context),_) source_values,
    Buffer.ready_content(~F32,next,
      %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,next),Buffer.content(~F32,input),retained) : Buffer.ContentReady<F32,_>
      Buffer.content_ready(~F32,input,source_ready)),weight_values,bias_values,weight_room,bias_room,reordered,valid,supported}

def bias_changed(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+next: Buffer.Model<F32>,
  +contents: {Buffer.content(~F32,next) == Buffer.content(~F32,bias) : Buffer.Content<F32>},evidence: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>) ->
  Sources<Logical,original,context,limits,images,input,tiles,weights,next>:
  Sources{length_exact,whole_room,source_values,source_ready,weight_values,bias_values,weight_room,bias_room,reordered,valid,supported} = evidence
  +tree_exact = tree_equal(next,bias,contents)
  Sources{length_exact,whole_room,source_values,source_ready,weight_values,
    %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,next),Buffer.tree(~F32,bias),tree_exact) :
      {Tree.read_sequence(~F32,ConvSpec.length(Guards.bias(DenseSources.image_problem(~Logical,~original,context,0n))),0,_) == Guards.bias(DenseSources.image_problem(~Logical,~original,context,0n)) : List<&2,F32>} bias_values,
    weight_room,
    %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,next),Buffer.tree(~F32,bias),tree_exact) :
      {Nat.is_le(ConvSpec.length(Guards.bias(DenseSources.image_problem(~Logical,~original,context,0n))),U32.to_nat(Tree.capacity(F32,_))) == True{} : Bool} bias_room,
    reordered,valid,supported}

def tiles_changed(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+next: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +contents: {Buffer.content(~Tile.Tile8,next) == Buffer.content(~Tile.Tile8,tiles) : Buffer.Content<Tile.Tile8>},evidence: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>) ->
  Sources<Logical,original,context,limits,images,input,next,weights,bias>:
  Sources{length_exact,whole_room,source_values,source_ready,weight_values,bias_values,weight_room,bias_room,reordered,valid,supported} = evidence
  Sources{length_exact,whole_room,source_values,source_ready,weight_values,bias_values,weight_room,bias_room,
    Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,next),Buffer.tree(~Tile.Tile8,tiles),
      WeightsLaws.run(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,context,0n)),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,context,0n)),Buffer.tree(~F32,weights)),DenseSources.cloned_tree(tiles,next,contents),reordered),valid,supported}

type Copies<-Logical: Data,-original: Logical -> U32 -> F32,-context: BandConvolutionValues.Context<Logical>,-limits: Machine.Limits,-images: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-weights: Buffer.Model<F32>,-bias: Buffer.Model<F32>> is Data:
  Copies{left_tiles: Buffer.Model<Tile.Tile8>,right_tiles: Buffer.Model<Tile.Tile8>,left_bias: Buffer.Model<F32>,right_bias: Buffer.Model<F32>,
    tile_owners: {(Buffer.pack(~Tile.Tile8,left_tiles),Buffer.pack(~Tile.Tile8,right_tiles)) == Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>},
    bias_owners: {(Buffer.pack(~F32,left_bias),Buffer.pack(~F32,right_bias)) == Clone.clone(~F32,Buffer.pack(~F32,bias)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
    left: Sources<Logical,original,context,limits,images,input,left_tiles,weights,left_bias>,
    right: Sources<Logical,original,context,limits,images,input,right_tiles,weights,right_bias>}

def copies_observed(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +evidence: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>,
  tile_observation: Cloned.Observation<Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles))>,bias_observation: Cloned.Observation<F32,Clone.clone(~F32,Buffer.pack(~F32,bias))>) ->
  Copies<Logical,original,context,limits,images,input,tiles,weights,bias>:
  Cloned.Observation{+left_tiles,+right_tiles,+tile_owners} = tile_observation
  Cloned.Observation{+left_bias,+right_bias,+bias_owners} = bias_observation
  Copies{left_tiles,right_tiles,left_bias,right_bias,tile_owners,bias_owners,
    bias_changed(~Logical,~original,context,limits,images,input,left_tiles,weights,bias,left_bias,Cloned.left_contents(~F32,bias,left_bias,right_bias,bias_owners),
      tiles_changed(~Logical,~original,context,limits,images,input,tiles,left_tiles,weights,bias,Cloned.left_contents(~Tile.Tile8,tiles,left_tiles,right_tiles,tile_owners),evidence)),
    bias_changed(~Logical,~original,context,limits,images,input,right_tiles,weights,bias,right_bias,Cloned.right_contents(~F32,bias,left_bias,right_bias,bias_owners),
      tiles_changed(~Logical,~original,context,limits,images,input,tiles,right_tiles,weights,bias,Cloned.right_contents(~Tile.Tile8,tiles,left_tiles,right_tiles,tile_owners),evidence))}

def sources_copies(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +evidence: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>) -> Copies<Logical,original,context,limits,images,input,tiles,weights,bias>:
  copies_observed(~Logical,~original,context,limits,images,input,tiles,weights,bias,evidence,
    Cloned.observe_pair(~Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles))),Cloned.observe_pair(~F32,Clone.clone(~F32,Buffer.pack(~F32,bias))))

# One true batch image and output copy produce owners, readiness and the
# extended finite public prefix. No completed image result is a public input.
def copy_length(+size: U32,+image: U32,+result: Buffer.Model<F32>,+output: Buffer.Model<F32>) ->
  {Buffer.content_length(~F32,Batch.copied_content(size,0,(image * size : U32),result,output)) == Buffer.length(~F32,output) : Nat}:
  {==}

def copy_owner(+size: U32,+image: U32,+result: Buffer.Model<F32>,+output: Buffer.Model<F32>,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,
  +owners: {(Buffer.pack(~F32,left),Buffer.pack(~F32,right)) == Copy.range(~F32,size,0,(image * size : U32),Buffer.pack(~F32,result),Buffer.pack(~F32,output)) : Storage.Buffer<F32> & Storage.Buffer<F32>}) ->
  {Buffer.pack(~F32,right) == TensorConvolution.copied_output(Copy.range(~F32,size,0,(image * size : U32),Buffer.pack(~F32,result),Buffer.pack(~F32,output))) : Storage.Buffer<F32>}:
  Equal.cong(Storage.Buffer<F32> & Storage.Buffer<F32>,Storage.Buffer<F32>,TensorConvolution.copied_output,(Buffer.pack(~F32,left),Buffer.pack(~F32,right)),Copy.range(~F32,size,0,(image * size : U32),Buffer.pack(~F32,result),Buffer.pack(~F32,output)),owners)

type Step<-Logical: Data,-original: Logical -> U32 -> F32,-custom: F32 -> F32,-context: BandConvolutionValues.Context<Logical>,-image: U32,-depth: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-bias: Buffer.Model<F32>,-output: Buffer.Model<F32>> is Data:
  Step{source_model: Buffer.Model<F32>,result_model: Buffer.Model<F32>,output_model: Buffer.Model<F32>,
    image_owners: {(Buffer.pack(~F32,source_model),Buffer.pack(~F32,result_model)) == TensorConvolution.image(~custom,image_activation(~Logical,context),depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),image,U32.from_nat(DenseSources.image_size(~Logical,context)),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
    copy_owners: {Buffer.pack(~F32,output_model) == TensorConvolution.copied_output(Copy.range(~F32,PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,0n)),0,(image * PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,0n)) : U32),Buffer.pack(~F32,result_model),Buffer.pack(~F32,output))) : Storage.Buffer<F32>},
    retained: {Buffer.content(~F32,source_model) == Buffer.content(~F32,input) : Buffer.Content<F32>},
    length_exact: {Buffer.length(~F32,output_model) == Buffer.length(~F32,output) : Nat},
    ready: Buffer.Ready<F32,output_model>,
    values: Values(~custom,~Logical,~original,context,Buffer.content(~F32,output_model),Nat.add(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),reference_volume(~Logical,context)))}

def image_result(~Logical: Data,~original: Logical -> U32 -> F32,~custom: F32 -> F32,-context: BandConvolutionValues.Context<Logical>,-image: U32,-depth: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-bias: Buffer.Model<F32>,leaf: ImageResult<Logical,original,custom,context,image,U32.from_nat(DenseSources.image_size(~Logical,context)),depth,input,tiles,bias>) -> Buffer.Model<F32>:
  ImageResult{source,result,owners,retained,source_values,source_ready,result_ready,length_exact,values} = leaf
  result

def steps_with_copy(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+image: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +before: Values(~custom,~Logical,~original,context,Buffer.content(~F32,output),Nat.mul(U32.to_nat(image),reference_volume(~Logical,context))),+output_ready: Buffer.Ready<F32,output>,
  +output_length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat},
  +inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},
  sources: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>,
  leaf: ImageResult<Logical,original,custom,context,image,U32.from_nat(DenseSources.image_size(~Logical,context)),depth,input,tiles,bias>,
  copied: DenseSources.Copied<PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))),0,(image * PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) : U32),
    image_result(~Logical,~original,~custom,context,image,depth,input,tiles,bias,leaf),output>) -> Step<Logical,original,custom,context,image,depth,input,tiles,bias,output>:
  Sources{length_input,whole_room,original_values,original_ready,weight_values,bias_values,weight_room,bias_room,reordered,+valid,+supported} = sources
  ImageResult{+source,+result,+image_owners,+retained,source_values,source_ready,+result_ready,+result_length,+result_values} = leaf
  DenseSources.Copied{+left,+right,+copy_owners,copy_retained,+written,copy_ready,+right_ready} = copied
  +problem = DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))
  +size = PreparedOutputReadback.count(problem)
  +image_valid = valid_image(~Logical,~original,context,U32.to_nat(image),valid)
  +image_supported = supported_image(~Logical,~original,context,limits,U32.to_nat(image),supported)
  +actual_content = Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,right),Batch.copied_content(size,0,(image * size : U32),result,output),copied_content(size,image,result,output),written,
    Equal.sym(Buffer.Content<F32>,copied_content(size,image,result,output),Batch.copied_content(size,0,(image * size : U32),result,output),Batch.copy_output_contents(size,0,(image * size : U32),result,output)))
  Step{source,result,right,
    %Equal.sym(Pipeline.Work,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),Inference.work(problem),work_image(~Logical,~original,context,U32.to_nat(image))) :
      {(Buffer.pack(~F32,source),Buffer.pack(~F32,result)) == TensorConvolution.image(~custom,image_activation(~Logical,context),depth,_,image,U32.from_nat(DenseSources.image_size(~Logical,context)),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias)) : Storage.Buffer<F32> & Storage.Buffer<F32>} image_owners,
    %Equal.sym(U32,PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,0n)),size,output_size(~Logical,~original,context,U32.to_nat(image))) :
      {Buffer.pack(~F32,right) == TensorConvolution.copied_output(Copy.range(~F32,_,0,(image * _ : U32),Buffer.pack(~F32,result),Buffer.pack(~F32,output))) : Storage.Buffer<F32>}
    copy_owner(size,image,result,output,left,right,copy_owners),
    retained,
    Equal.trans(Nat,Buffer.length(~F32,right),Buffer.content_length(~F32,Buffer.content(~F32,right)),Buffer.length(~F32,output),Equal.sym(Nat,Buffer.content_length(~F32,Buffer.content(~F32,right)),Buffer.length(~F32,right),DenseSources.image_copy_model_length(right)),
      Equal.trans(Nat,Buffer.content_length(~F32,Buffer.content(~F32,right)),Buffer.content_length(~F32,Batch.copied_content(size,0,(image * size : U32),result,output)),Buffer.length(~F32,output),
        Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.content(~F32,right),Batch.copied_content(size,0,(image * size : U32),result,output),written),copy_length(size,image,result,output))),
    right_ready,
    %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,right),copied_content(size,image,result,output),actual_content) :
      Values(~custom,~Logical,~original,context,_,Nat.add(Nat.mul(U32.to_nat(image),reference_volume(~Logical,context)),reference_volume(~Logical,context)))
    copied_prefix(~custom,~Logical,~original,context,limits,images,image,result,output,before,image_valid,image_supported,result_values,result_ready,output_ready,result_length,output_length,inside)}

def with_image(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+image: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +before: Values(~custom,~Logical,~original,context,Buffer.content(~F32,output),Nat.mul(U32.to_nat(image),reference_volume(~Logical,context))),+output_ready: Buffer.Ready<F32,output>,
  +output_length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat},
  +inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},
  +sources: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>,
  +leaf: ImageResult<Logical,original,custom,context,image,U32.from_nat(DenseSources.image_size(~Logical,context)),depth,input,tiles,bias>) -> Step<Logical,original,custom,context,image,depth,input,tiles,bias,output>:
  ImageResult{source,+result,owners,retained,source_values,source_ready,+result_ready,length_exact,values} = leaf
  steps_with_copy(~custom,~Logical,~original,context,limits,images,image,depth,input,tiles,weights,bias,output,before,output_ready,output_length,inside,sources,leaf,
    DenseSources.copied(PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))),0,(image * PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,U32.to_nat(image))) : U32),result,output,result_ready,output_ready))

def steps_step(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+image: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +before: Values(~custom,~Logical,~original,context,Buffer.content(~F32,output),Nat.mul(U32.to_nat(image),reference_volume(~Logical,context))),+output_ready: Buffer.Ready<F32,output>,
  +output_length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat},
  +inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},
  +sources: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>) -> Step<Logical,original,custom,context,image,depth,input,tiles,bias,output>:
  with_image(~custom,~Logical,~original,context,limits,images,image,depth,input,tiles,weights,bias,output,before,output_ready,output_length,inside,sources,
    batch_image(~custom,~Logical,~original,context,limits,image,U32.from_nat(DenseSources.image_size(~Logical,context)),depth,input,tiles,weights,bias,inputs(~Logical,~original,context,limits,images,image,input,tiles,weights,bias,sources,inside)))

# Finite induction over the true zero/one/many batch callbacks. Source
# receipts are transported along the returned image owner and actual clones.
def values_procedure(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,count: Nat,index: U32,depth: U32,input: Buffer.Model<F32>,tiles: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<F32>,output: Buffer.Model<F32>) -> Storage.Buffer<F32>:
  TensorConvolution.batch(~custom,image_activation(~Logical,context),count,index,depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),U32.from_nat(DenseSources.image_size(~Logical,context)),PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,0n)),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias),Buffer.pack(~F32,output))

type Completed<-Logical: Data,-original: Logical -> U32 -> F32,-custom: F32 -> F32,-context: BandConvolutionValues.Context<Logical>,-count: Nat,-index: U32,-depth: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-bias: Buffer.Model<F32>,-output: Buffer.Model<F32>> is Data:
  Completed{model: Buffer.Model<F32>,owner: {Buffer.pack(~F32,model) == values_procedure(~custom,~Logical,~original,context,count,index,depth,input,tiles,bias,output) : Storage.Buffer<F32>},
    length_exact: {Buffer.length(~F32,model) == Buffer.length(~F32,output) : Nat},ready: Buffer.Ready<F32,model>,
    values: Dense.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,Buffer.content(~F32,model))}

def image_inside(+index: U32,+rest: Nat,+images: U32,+remaining: {Nat.add(U32.to_nat(index),1n+rest) == U32.to_nat(images) : Nat}) ->
  {Nat.is_lt(U32.to_nat(index),U32.to_nat(images)) == True{} : Bool}:
  Order.room_index(U32.to_nat(index),rest,U32.to_nat(images),
    %remaining : {Nat.is_le(Nat.add(U32.to_nat(index),1n+rest),_) == True{} : Bool}
    Order.reflexive(Nat.add(U32.to_nat(index),1n+rest)))

def values_count_room(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-weights: Buffer.Model<F32>,-bias: Buffer.Model<F32>,sources: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>) ->
  {Nat.is_le(U32.to_nat(images),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Sources{length,whole,source_values,source_ready,weight_values,bias_values,weight_room,bias_room,reordered,valid,supported} = sources
  geometry_count_room(~Logical,~original,context,limits,images,valid,whole)

def finite_values(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+model: Buffer.Model<F32>,+completed: Nat,
  +prefix: Values(~custom,~Logical,~original,context,Buffer.content(~F32,model),completed),+exact: {completed == Buffer.length(~F32,model) : Nat}) ->
  Dense.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,Buffer.content(~F32,model)):
  %Equal.trans(Nat,completed,Buffer.length(~F32,model),Buffer.content_length(~F32,Buffer.content(~F32,model)),exact,Equal.sym(Nat,Buffer.content_length(~F32,Buffer.content(~F32,model)),Buffer.length(~F32,model),DenseSources.image_copy_model_length(model))) :
    Values(~custom,~Logical,~original,context,Buffer.content(~F32,model),_)
  prefix

def last_extent(+index: U32,+images: U32,+volume: Nat,+remaining: {Nat.add(U32.to_nat(index),1n) == U32.to_nat(images) : Nat}) ->
  {Nat.add(Nat.mul(U32.to_nat(index),volume),volume) == Nat.mul(U32.to_nat(images),volume) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(U32.to_nat(index),volume),volume),Nat.mul(1n+U32.to_nat(index),volume),Nat.mul(U32.to_nat(images),volume),Order.commutative(Nat.mul(U32.to_nat(index),volume),volume),
    Equal.cong(Nat,Nat,count => Nat.mul(count,volume),1n+U32.to_nat(index),U32.to_nat(images),
      Equal.trans(Nat,1n+U32.to_nat(index),Nat.add(U32.to_nat(index),1n),U32.to_nat(images),Equal.sym(Nat,Nat.add(U32.to_nat(index),1n),1n+U32.to_nat(index),Addition.successor_width(U32.to_nat(index))),remaining)))

def empty(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+images: U32,+index: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +before: Values(~custom,~Logical,~original,context,Buffer.content(~F32,output),Nat.mul(U32.to_nat(index),reference_volume(~Logical,context))),+ready: Buffer.Ready<F32,output>,
  +length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat},
  +remaining: {Nat.add(U32.to_nat(index),0n) == U32.to_nat(images) : Nat}) -> Completed<Logical,original,custom,context,0n,index,depth,input,tiles,bias,output>:
  +index_exact = Equal.trans(Nat,U32.to_nat(index),Nat.add(U32.to_nat(index),0n),U32.to_nat(images),Equal.sym(Nat,Nat.add(U32.to_nat(index),0n),U32.to_nat(index),Addition.zero_right(U32.to_nat(index))),remaining)
  Completed{output,{==},{==},ready,
    finite_values(~custom,~Logical,~original,context,output,Nat.mul(U32.to_nat(index),reference_volume(~Logical,context)),before,
      Equal.trans(Nat,Nat.mul(U32.to_nat(index),reference_volume(~Logical,context)),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),Buffer.length(~F32,output),
        Equal.cong(Nat,Nat,image => Nat.mul(image,reference_volume(~Logical,context)),U32.to_nat(index),U32.to_nat(images),index_exact),Equal.sym(Nat,Buffer.length(~F32,output),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),length)))}

def last(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+images: U32,+index: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat},
  +remaining: {Nat.add(U32.to_nat(index),1n) == U32.to_nat(images) : Nat},step: Step<Logical,original,custom,context,index,depth,input,tiles,bias,output>) -> Completed<Logical,original,custom,context,1n,index,depth,input,tiles,bias,output>:
  Step{+source,+result,+target,+image_owners,+copy_owners,retained,+target_length,+ready,+values} = step
  Completed{target,
    one(~custom,image_activation(~Logical,context),index,depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),U32.from_nat(DenseSources.image_size(~Logical,context)),PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,0n)),input,tiles,bias,output,source,result,target,image_owners,copy_owners),target_length,ready,
    finite_values(~custom,~Logical,~original,context,target,Nat.add(Nat.mul(U32.to_nat(index),reference_volume(~Logical,context)),reference_volume(~Logical,context)),values,
      Equal.trans(Nat,Nat.add(Nat.mul(U32.to_nat(index),reference_volume(~Logical,context)),reference_volume(~Logical,context)),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),Buffer.length(~F32,target),last_extent(index,images,reference_volume(~Logical,context),remaining),
        Equal.sym(Nat,Buffer.length(~F32,target),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),Equal.trans(Nat,Buffer.length(~F32,target),Buffer.length(~F32,output),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),target_length,length))))}

def Continue(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+rest: Nat,+index: U32,+depth: U32,+weights: Buffer.Model<F32>) -> Type:
  @+input: Buffer.Model<F32> -> @+tiles: Buffer.Model<Tile.Tile8> -> @+bias: Buffer.Model<F32> -> @+output: Buffer.Model<F32> ->
  @+sources: Sources<Logical,original,context,limits,images,input,tiles,weights,bias> ->
  @+before: Values(~custom,~Logical,~original,context,Buffer.content(~F32,output),Nat.mul(U32.to_nat(index),reference_volume(~Logical,context))) ->
  @+ready: Buffer.Ready<F32,output> ->
  @+length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat} ->
  @+remaining: {Nat.add(U32.to_nat(index),1n+rest) == U32.to_nat(images) : Nat} ->
  Completed<Logical,original,custom,context,1n+rest,index,depth,input,tiles,bias,output>

def many_finished(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+rest: Nat,+index: U32,+depth: U32,
  +input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +left_tiles: Buffer.Model<Tile.Tile8>,+right_tiles: Buffer.Model<Tile.Tile8>,+left_bias: Buffer.Model<F32>,+right_bias: Buffer.Model<F32>,
  +source: Buffer.Model<F32>,+result: Buffer.Model<F32>,+target: Buffer.Model<F32>,
  +tile_owners: {(Buffer.pack(~Tile.Tile8,left_tiles),Buffer.pack(~Tile.Tile8,right_tiles)) == Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>},
  +bias_owners: {(Buffer.pack(~F32,left_bias),Buffer.pack(~F32,right_bias)) == Clone.clone(~F32,Buffer.pack(~F32,bias)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
  +image_owners: {(Buffer.pack(~F32,source),Buffer.pack(~F32,result)) == TensorConvolution.image(~custom,image_activation(~Logical,context),depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),index,U32.from_nat(DenseSources.image_size(~Logical,context)),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,left_tiles),Buffer.pack(~F32,left_bias)) : Storage.Buffer<F32> & Storage.Buffer<F32>},
  +copy_owners: {Buffer.pack(~F32,target) == TensorConvolution.copied_output(Copy.range(~F32,PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,0n)),0,(index * PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,0n)) : U32),Buffer.pack(~F32,result),Buffer.pack(~F32,output))) : Storage.Buffer<F32>},
  +target_length: {Buffer.length(~F32,target) == Buffer.length(~F32,output) : Nat},
  future: Completed<Logical,original,custom,context,1n+rest,(index + 1 : U32),depth,source,right_tiles,right_bias,target>) -> Completed<Logical,original,custom,context,2n+rest,index,depth,input,tiles,bias,output>:
  Completed{+model,+owner,+length,+ready,+values} = future
  Completed{model,many(~custom,image_activation(~Logical,context),rest,index,depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),U32.from_nat(DenseSources.image_size(~Logical,context)),PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,0n)),input,tiles,bias,output,left_tiles,right_tiles,left_bias,right_bias,source,result,target,model,tile_owners,bias_owners,image_owners,copy_owners,owner),
    Equal.trans(Nat,Buffer.length(~F32,model),Buffer.length(~F32,target),Buffer.length(~F32,output),length,target_length),ready,values}

def copy_left_tiles(~Logical: Data,~original: Logical -> U32 -> F32,-context: BandConvolutionValues.Context<Logical>,-limits: Machine.Limits,-images: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-weights: Buffer.Model<F32>,-bias: Buffer.Model<F32>,copies: Copies<Logical,original,context,limits,images,input,tiles,weights,bias>) -> Buffer.Model<Tile.Tile8>:
  Copies{left_tiles,right_tiles,left_bias,right_bias,tile_owners,bias_owners,left,right} = copies
  left_tiles

def copy_left_bias(~Logical: Data,~original: Logical -> U32 -> F32,-context: BandConvolutionValues.Context<Logical>,-limits: Machine.Limits,-images: U32,-input: Buffer.Model<F32>,-tiles: Buffer.Model<Tile.Tile8>,-weights: Buffer.Model<F32>,-bias: Buffer.Model<F32>,copies: Copies<Logical,original,context,limits,images,input,tiles,weights,bias>) -> Buffer.Model<F32>:
  Copies{left_tiles,right_tiles,left_bias,right_bias,tile_owners,bias_owners,left,right} = copies
  left_bias

def many_stepped(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+rest: Nat,+index: U32,+depth: U32,
  +input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat},
  +remaining: {Nat.add(U32.to_nat(index),2n+rest) == U32.to_nat(images) : Nat},
  +copies: Copies<Logical,original,context,limits,images,input,tiles,weights,bias>,
  step: Step<Logical,original,custom,context,index,depth,input,copy_left_tiles(~Logical,~original,context,limits,images,input,tiles,weights,bias,copies),copy_left_bias(~Logical,~original,context,limits,images,input,tiles,weights,bias,copies),output>,
  continue: Continue(~custom,~Logical,~original,context,limits,images,rest,(index + 1 : U32),depth,weights)) -> Completed<Logical,original,custom,context,2n+rest,index,depth,input,tiles,bias,output>:
  Copies{+left_tiles,+right_tiles,+left_bias,+right_bias,+tile_owners,+bias_owners,left_sources,+right_sources} = copies
  Step{+source,+result,+target,+image_owners,+copy_owners,+retained,+target_length,+ready,+values} = step
  +inside = image_inside(index,1n+rest,images,remaining)
  +room = values_count_room(~Logical,~original,context,limits,images,input,right_tiles,weights,right_bias,right_sources)
  +next = next_exact(limits,images,index,room,inside)
  +next_remaining = geometry_next_remaining(index,rest,images,next,remaining)
  +next_sources = input_changed(~Logical,~original,context,limits,images,input,source,right_tiles,weights,right_bias,retained,right_sources)
  +next_values = {
    %completed_next(index,reference_volume(~Logical,context),next) : Values(~custom,~Logical,~original,context,Buffer.content(~F32,target),_)
    values : Values(~custom,~Logical,~original,context,Buffer.content(~F32,target),Nat.mul(U32.to_nat((index + 1 : U32)),reference_volume(~Logical,context)))}
  +next_length = Equal.trans(Nat,Buffer.length(~F32,target),Buffer.length(~F32,output),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),target_length,length)
  many_finished(~custom,~Logical,~original,context,rest,index,depth,input,tiles,bias,output,left_tiles,right_tiles,left_bias,right_bias,source,result,target,tile_owners,bias_owners,image_owners,copy_owners,target_length,
    continue(source,right_tiles,right_bias,target,next_sources,next_values,ready,next_length,next_remaining))

def many_prepared(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+rest: Nat,+index: U32,+depth: U32,
  +input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +before: Values(~custom,~Logical,~original,context,Buffer.content(~F32,output),Nat.mul(U32.to_nat(index),reference_volume(~Logical,context))),+ready: Buffer.Ready<F32,output>,
  +length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat},
  +remaining: {Nat.add(U32.to_nat(index),2n+rest) == U32.to_nat(images) : Nat},
  +copies: Copies<Logical,original,context,limits,images,input,tiles,weights,bias>,
  continue: Continue(~custom,~Logical,~original,context,limits,images,rest,(index + 1 : U32),depth,weights)) -> Completed<Logical,original,custom,context,2n+rest,index,depth,input,tiles,bias,output>:
  Copies{+left_tiles,right_tiles,+left_bias,right_bias,tile_owners,bias_owners,+left_sources,right_sources} = copies
  many_stepped(~custom,~Logical,~original,context,limits,images,rest,index,depth,input,tiles,weights,bias,output,length,remaining,copies,
    steps_step(~custom,~Logical,~original,context,limits,images,index,depth,input,left_tiles,weights,left_bias,output,before,ready,length,image_inside(index,1n+rest,images,remaining),left_sources),continue)

def loop(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,count: Nat,+index: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,
  +sources: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>,
  +before: Values(~custom,~Logical,~original,context,Buffer.content(~F32,output),Nat.mul(U32.to_nat(index),reference_volume(~Logical,context))),+ready: Buffer.Ready<F32,output>,
  +length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat},
  +remaining: {Nat.add(U32.to_nat(index),count) == U32.to_nat(images) : Nat}) -> Completed<Logical,original,custom,context,count,index,depth,input,tiles,bias,output>:
  match count:
    case 0n: empty(~custom,~Logical,~original,context,images,index,depth,input,tiles,bias,output,before,ready,length,remaining)
    case 1n:
      last(~custom,~Logical,~original,context,images,index,depth,input,tiles,bias,output,length,remaining,
        steps_step(~custom,~Logical,~original,context,limits,images,index,depth,input,tiles,weights,bias,output,before,ready,length,image_inside(index,0n,images,remaining),sources))
    case 2n+ +rest:
      many_prepared(~custom,~Logical,~original,context,limits,images,rest,index,depth,input,tiles,weights,bias,output,before,ready,length,remaining,sources_copies(~Logical,~original,context,limits,images,input,tiles,weights,bias,sources),
        source => child_tiles => child_bias => target => child_sources => child_values => child_ready => child_length => child_remaining =>
          loop(~custom,~Logical,~original,context,limits,images,1n+rest,(index + 1 : U32),depth,source,child_tiles,weights,child_bias,target,child_sources,child_values,child_ready,child_length,child_remaining))

# Allocate the actual whole batch output and discharge the loop's finite
# contract with its public NCHW extents, Ready status and allocation witness.
def size(~Logical: Data,~original: Logical -> U32 -> F32,context: BandConvolutionValues.Context<Logical>) -> U32:
  PreparedOutputReadback.count(DenseSources.image_problem(~Logical,~original,context,0n))

def extents(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,images: U32) -> List<&2,U32>:
  TensorConvolution.output_shape(True{},images,U32.from_nat(Guards.output_channel_count(DenseSources.image_problem(~Logical,~original,context,0n))),U32.from_nat(ConvSpec.out_height(DenseSources.image_problem(~Logical,~original,context,0n))),U32.from_nat(ConvSpec.out_width(DenseSources.image_problem(~Logical,~original,context,0n))))

def shape_header(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+images: U32) ->
  {ShapeLaws.elements(extents(~Logical,~original,context,images)) == Nat.mul(U32.to_nat(images),ShapeLaws.elements(DenseSingle.output_extents(DenseSources.image_problem(~Logical,~original,context,0n),False{}))) : Nat}:
  {==}

def shape_count(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool}) ->
  {ShapeLaws.elements(extents(~Logical,~original,context,images)) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat}:
  Equal.trans(Nat,ShapeLaws.elements(extents(~Logical,~original,context,images)),Nat.mul(U32.to_nat(images),ShapeLaws.elements(DenseSingle.output_extents(DenseSources.image_problem(~Logical,~original,context,0n),False{}))),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),shape_header(~Logical,~original,context,images),
    Equal.cong(Nat,Nat,volume => Nat.mul(U32.to_nat(images),volume),ShapeLaws.elements(DenseSingle.output_extents(DenseSources.image_problem(~Logical,~original,context,0n),False{})),reference_volume(~Logical,context),
      Equal.trans(Nat,ShapeLaws.elements(DenseSingle.output_extents(DenseSources.image_problem(~Logical,~original,context,0n),False{})),U32.to_nat(size(~Logical,~original,context)),reference_volume(~Logical,context),DenseSingle.shape_count(limits,DenseSources.image_problem(~Logical,~original,context,0n),False{},valid,supported),count_exact(~Logical,~original,context,limits,0n,valid,supported))))

def allocation_count(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((images * size(~Logical,~original,context) : U32)) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat}:
  %count_exact(~Logical,~original,context,limits,0n,valid,supported) : {U32.to_nat((images * size(~Logical,~original,context) : U32)) == Nat.mul(U32.to_nat(images),_) : Nat}
  Arithmetic.multiply(images,size(~Logical,~original,context),Machine.capacity(limits),Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat(size(~Logical,~original,context)),reference_volume(~Logical,context),count_exact(~Logical,~original,context,limits,0n,valid,supported)) : {Nat.is_le(Nat.mul(U32.to_nat(images),_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room)

def target(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> DenseSources.ImageSourceAllocated<(images * size(~Logical,~original,context) : U32)>:
  DenseSources.image_source_allocated((images * size(~Logical,~original,context) : U32),limits,
    %Equal.sym(Bool,U32.is_le((images * size(~Logical,~original,context) : U32),Machine.capacity(limits)),Nat.is_le(U32.to_nat((images * size(~Logical,~original,context) : U32)),U32.to_nat(Machine.capacity(limits))),Comparison.le((images * size(~Logical,~original,context) : U32),Machine.capacity(limits))) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat((images * size(~Logical,~original,context) : U32)),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),allocation_count(~Logical,~original,context,limits,images,valid,supported,room)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room)

def completed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,+workers: U32,
  +output_length: {Buffer.length(~F32,output) == Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)) : Nat},
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool},
  result: Completed<Logical,original,custom,context,U32.to_nat(images),0,depth,input,tiles,bias,output>) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,Tensors.Tensor{values_procedure(~custom,~Logical,~original,context,U32.to_nat(images),0,depth,input,tiles,bias,output),extents(~Logical,~original,context,images),0.0,Tensors.Ready{},workers}):
  Completed{+model,+owner,+length_exact,+ready,+values} = result
  %owner : Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,Tensors.Tensor{_,extents(~Logical,~original,context,images),0.0,Tensors.Ready{},workers})
  Shape.Both{Shape.Both{
    %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.content(~F32,model),Buffer.packed_contents(~F32,model)) : Dense.WellFormed<F32,extents(~Logical,~original,context,images),_>
    Dense.WellFormed{Equal.trans(Nat,Buffer.content_length(~F32,Buffer.content(~F32,model)),Buffer.length(~F32,model),ShapeLaws.elements(extents(~Logical,~original,context,images)),DenseSources.image_copy_model_length(model),
      Equal.trans(Nat,Buffer.length(~F32,model),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),ShapeLaws.elements(extents(~Logical,~original,context,images)),
        Equal.trans(Nat,Buffer.length(~F32,model),Buffer.length(~F32,output),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),length_exact,output_length),
        Equal.sym(Nat,ShapeLaws.elements(extents(~Logical,~original,context,images)),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),shape_count(~Logical,~original,context,limits,images,valid,supported)))),Buffer.content_ready(~F32,model,ready)},{==}},
    %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.content(~F32,model),Buffer.packed_contents(~F32,model)) : Dense.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,_) values}

def allocated(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32,
  +sources: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>,
  +valid: {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,0n)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  allocation: DenseSources.ImageSourceAllocated<(images * size(~Logical,~original,context) : U32)>) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,Tensors.Tensor{
    TensorConvolution.batch(~custom,image_activation(~Logical,context),U32.to_nat(images),0,depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),U32.from_nat(DenseSources.image_size(~Logical,context)),size(~Logical,~original,context),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(images * size(~Logical,~original,context) : U32))),extents(~Logical,~original,context,images),0.0,Tensors.Ready{},workers}):
  DenseSources.ImageSourceAllocated{+model,+owner,+ready,+length_exact} = allocation
  +output_length = Equal.trans(Nat,Buffer.length(~F32,model),U32.to_nat((images * size(~Logical,~original,context) : U32)),Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),length_exact,allocation_count(~Logical,~original,context,limits,images,valid,supported,room))
  %owner : Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,Tensors.Tensor{
    TensorConvolution.batch(~custom,image_activation(~Logical,context),U32.to_nat(images),0,depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),U32.from_nat(DenseSources.image_size(~Logical,context)),size(~Logical,~original,context),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias),_),extents(~Logical,~original,context,images),0.0,Tensors.Ready{},workers})
  completed(~custom,~Logical,~original,context,limits,images,depth,input,tiles,bias,model,workers,output_length,valid,supported,
    loop(~custom,~Logical,~original,context,limits,images,U32.to_nat(images),0,depth,input,tiles,weights,bias,model,sources,Unit{},ready,output_length,{==}))

def batch(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32,
  +sources: Sources<Logical,original,context,limits,images,input,tiles,weights,bias>,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(images),reference_volume(~Logical,context)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,Tensors.Tensor{
    TensorConvolution.batch(~custom,image_activation(~Logical,context),U32.to_nat(images),0,depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),U32.from_nat(DenseSources.image_size(~Logical,context)),size(~Logical,~original,context),Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(images * size(~Logical,~original,context) : U32))),extents(~Logical,~original,context,images),0.0,Tensors.Ready{},workers}):
  Sources{length,whole_room,source_values,source_ready,weight_values,bias_values,weight_room,bias_room,reordered,+valid,+supported} = sources
  allocated(~custom,~Logical,~original,context,limits,images,depth,input,tiles,weights,bias,workers,sources,valid,supported,room,target(~Logical,~original,context,limits,images,valid,supported,room))
