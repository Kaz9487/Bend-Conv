# Native dense callers: sources, checked calls, batch and single contracts.
import Base
import ./band_dense_sources.bend as DenseSources
import ./band_dense_public.bend as DensePublic
import ./band_public_frame.bend as PublicFrame
import ../../lib/math_activation.bend as Activations
import ./band_dense_batch_execution.bend as DenseBatchExecution
import ./band_native_caller.bend as Native
import ../specification.bend as ConvSpec
import ../../lib/default_machine_limits.bend as Default
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/tensor.bend as Tensors
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_dense_single.bend as DenseSingle
import ./band_initial_layout.bend as InitialLayout
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_dense_contract.bend as Dense
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./band_caller_guards.bend as Caller
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../inference.bend as NativeConv
import ./band_frame_configuration.bend as Configuration
import ./band_leaf_values.bend as LeafValues
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./reduction_coordinate_bounds.bend as Coordinates
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ./band_dense_batch.bend as Batch
import ./band_machine_caller.bend as MachineCaller
import ./band_parameter_owners.bend as Parameters
import ./band_window_contents.bend as WindowContents
import ./convolution_guard_facts.bend as Guards
import ../../lib/kernel_band_planning.bend as BandPlanning
import ../../lib/kernel_gemm_buffer.bend as GemmBuffer
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_count.bend as BandCount
import ../../lib/proofs/tensor_count_refinement.bend as Count
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/storage_buffer.bend as Storage

# Actual decoded dense input and finite source values supply the batch inputs.
# Validation and machine guards are reused at the existing convolution leaf;
# the actual weight reorder is observed once and retained with its owner law.
def sources_context(~Logical: Data,+logical: Logical,activation: Activations.Activation,input: PublicFrame.NativeInput,weights: List<&2,F32>,bias: List<&2,F32>) -> BandConvolutionValues.Context<Logical>:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  BandConvolutionValues.Context{logical,activation,U32.to_nat(ci),U32.to_nat(co),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p),weights,bias}

def whole_length(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},workers}),matched: {Native.input_matched(input) == True{} : Bool}) ->
  {Buffer.length(~F32,source) == Nat.mul(U32.to_nat(PublicFrame.input_images(input)),DenseSources.image_size(~Logical,sources_context(~Logical,logical,activation,input,weights,bias))) : Nat}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  DensePublic.length(input,source,fill,workers,normal,matched)

def whole_room(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,+limits: Machine.Limits,
  +capacity_actual: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},workers}),matched: {Native.input_matched(input) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(PublicFrame.input_images(input)),DenseSources.image_size(~Logical,sources_context(~Logical,logical,activation,input,weights,bias))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  DensePublic.room(input,source,fill,workers,Machine.capacity(limits),capacity_actual,normal,matched)

def sequence_values(+values: List<&2,F32>,+model: Buffer.Model<F32>,+evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>,
  exact: {Buffer.tree(~F32,model) == StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence) : Tree.Tree<F32>}) ->
  {Tree.read_sequence(~F32,ConvSpec.length(values),0,Buffer.tree(~F32,model)) == values : List<&2,F32>}:
  StorageLaws.SequenceEvidence{tree,contents,room} = evidence
  %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,model),tree,exact) : {Tree.read_sequence(~F32,ConvSpec.length(values),0,_) == values : List<&2,F32>}
  contents

def sequence_room(+values: List<&2,F32>,+model: Buffer.Model<F32>,+evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>,
  exact: {Buffer.tree(~F32,model) == StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence) : Tree.Tree<F32>}) ->
  {Nat.is_le(ConvSpec.length(values),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,model)))) == True{} : Bool}:
  StorageLaws.SequenceEvidence{tree,contents,room} = evidence
  %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,model),tree,exact) : {Nat.is_le(ConvSpec.length(values),U32.to_nat(Tree.capacity(F32,_))) == True{} : Bool}
  room

def valid_zero(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool}) ->
  {ConvSpec.valid(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)) == True{} : Bool}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  Caller.valid(~Logical,~original,logical,PublicFrame.frame_dimensions(input),weights,bias,0n,validated)

def supported_zero(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+limits: Machine.Limits,
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},supported: {PublicFrame.machine(input,limits) == True{} : Bool}) ->
  {NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)) == True{} : Bool}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  Caller.supported(~Logical,~original,logical,limits,PublicFrame.frame_dimensions(input),weights,bias,0n,
    Configuration.supported(~Logical,~original,logical,limits,PublicFrame.frame_dimensions(input),weights,bias,validated,
      Boolean.left(KernelShape.supports_packed(limits,PublicFrame.frame_dimensions(input),Caller.input_length(PublicFrame.frame_dimensions(input)),Configuration.weight_size(PublicFrame.frame_dimensions(input))),
        Nat.is_le(LeafValues.output_elements(PublicFrame.input_images(input),PublicFrame.frame_dimensions(input)),U32.to_nat(Machine.capacity(limits))),supported)))

def source_logical(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>) ->
  {DenseSources.logical_source(~Logical,sources_context(~Logical,logical,activation,input,weights,bias)) == logical : Logical}:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  {==}

def sources(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+limits: Machine.Limits,
  +capacity_actual: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +source: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},workers}),+matched: {Native.input_matched(input) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  reordered: {Buffer.tree(~Tile.Tile8,tiles) == WeightsLaws.run(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)),Buffer.tree(~F32,weight_model)) : Tree.Tree<Tile.Tile8>},
  source_values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},supported: {PublicFrame.machine(input,limits) == True{} : Bool}) ->
  DenseBatchExecution.Sources<Logical,original,sources_context(~Logical,logical,activation,input,weights,bias),limits,PublicFrame.input_images(input),source,tiles,weight_model,bias_model>:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  DenseBatchExecution.Sources{whole_length(~Logical,logical,activation,input,weights,bias,source,fill,workers,normal,matched),whole_room(~Logical,logical,activation,input,weights,bias,source,fill,workers,limits,capacity_actual,normal,matched),
    %Equal.sym(Logical,DenseSources.logical_source(~Logical,sources_context(~Logical,logical,activation,input,weights,bias)),logical,source_logical(~Logical,logical,activation,input,weights,bias)) : Dense.Values(~F32,~Logical,~original,_,Buffer.content(~F32,source)) source_values,
    DensePublic.normal_ready(Native.input_extents(input),source,fill,workers,normal),sequence_values(weights,weight_model,weight_evidence,weight_exact),sequence_values(bias,bias_model,bias_evidence,bias_exact),
    sequence_room(weights,weight_model,weight_evidence,weight_exact),sequence_room(bias,bias_model,bias_evidence,bias_exact),reordered,
    valid_zero(~Logical,~original,logical,activation,input,weights,bias,validated),supported_zero(~Logical,~original,logical,activation,input,weights,bias,limits,validated,supported)}

type PreparedSources<-Logical: Data,-original: Logical -> U32 -> F32,-sources_context: BandConvolutionValues.Context<Logical>,-limits: Machine.Limits,-images: U32,-source: Buffer.Model<F32>,-weights: Buffer.Model<F32>,-bias: Buffer.Model<F32>> is Data:
  PreparedSources{tiles: Buffer.Model<Tile.Tile8>,
    owner: {Buffer.pack(~Tile.Tile8,tiles) == GemmBuffer.reorder(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,sources_context,0n)),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,sources_context,0n)),Buffer.pack(~F32,weights)) : Storage.Buffer<Tile.Tile8>},
    sources: DenseBatchExecution.Sources<Logical,original,sources_context,limits,images,source,tiles,weights,bias>}

def prepared_reorder(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+limits: Machine.Limits,
  +capacity_actual: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},workers}),+matched: {Native.input_matched(input) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  source_values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},supported: {PublicFrame.machine(input,limits) == True{} : Bool},
  reordered: DenseSources.Reordered<MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)),weight_model>) ->
  PreparedSources<Logical,original,sources_context(~Logical,logical,activation,input,weights,bias),limits,PublicFrame.input_images(input),source,weight_model,bias_model>:
  DenseSources.Reordered{+tiles,owner,exact} = reordered
  PreparedSources{tiles,owner,sources(~Logical,~original,logical,activation,input,weights,bias,limits,capacity_actual,source,tiles,weight_model,bias_model,fill,workers,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_exact,exact,source_values,validated,supported)}

type LimitsReceipt is Data:
  LimitsReceipt{limits: Machine.Limits,actual: {limits == Default.limits() : Machine.Limits}}

def default_limits() -> LimitsReceipt:
  LimitsReceipt{Default.limits(),{==}}

def prepared_limits(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},workers}),+matched: {Native.input_matched(input) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  source_values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},certificate: LimitsReceipt) ->
  PreparedSources<Logical,original,sources_context(~Logical,logical,activation,input,weights,bias),Default.limits(),PublicFrame.input_images(input),source,weight_model,bias_model>:
  LimitsReceipt{+limits,+actual} = certificate
  %actual : PreparedSources<Logical,original,sources_context(~Logical,logical,activation,input,weights,bias),_,PublicFrame.input_images(input),source,weight_model,bias_model>
  prepared_reorder(~Logical,~original,logical,activation,input,weights,bias,limits,Equal.cong(Machine.Limits,U32,Machine.capacity,limits,Default.limits(),actual),source,weight_model,bias_model,fill,workers,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_exact,source_values,validated,
    %Equal.sym(Machine.Limits,limits,Default.limits(),actual) : {PublicFrame.machine(input,_) == True{} : Bool} supported,
    DenseSources.reordered(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)),weight_model))

def sources_prepared(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},workers}),+matched: {Native.input_matched(input) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  source_values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool}) ->
  PreparedSources<Logical,original,sources_context(~Logical,logical,activation,input,weights,bias),Default.limits(),PublicFrame.input_images(input),source,weight_model,bias_model>:
  prepared_limits(~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,fill,workers,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_exact,source_values,validated,supported,default_limits())

# This prefix induction uses only the tensor's finite logical relation. The
# one-image branch itself supplies images==1; no unbounded agreement is used.
def prefix(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: Buffer.Model<F32>,+remaining: Nat,+start: Nat,
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),+room: {Nat.is_le(Nat.add(start,remaining),Buffer.length(~F32,source)) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,remaining,U32.from_nat(start),Buffer.tree(~F32,source)) == WindowContents.image_values(~Logical,~original,remaining,start,logical) : List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +head = Equal.trans(F32,Tree.read(F32,Buffer.tree(~F32,source),U32.from_nat(start)),Buffer.value(~F32,U32.from_nat(start),Buffer.content(~F32,source)),original(logical,U32.from_nat(start)),
        DenseSources.model_read(U32.from_nat(start),source),
        Dense.read(~F32,~Logical,~original,logical,Buffer.content(~F32,source),start,values,
          %Equal.sym(Nat,Buffer.content_length(~F32,Buffer.content(~F32,source)),Buffer.length(~F32,source),DenseSources.image_copy_model_length(source)) : {Nat.is_lt(start,_) == True{} : Bool}
          Order.room_index(start,rest,Buffer.length(~F32,source),room)))
      %Equal.sym(F32,Tree.read(F32,Buffer.tree(~F32,source),U32.from_nat(start)),original(logical,U32.from_nat(start)),head) :
        {_ <> Tree.read_sequence(~F32,rest,(U32.from_nat(start) + 1 : U32),Buffer.tree(~F32,source)) == WindowContents.image_values(~Logical,~original,1n+rest,start,logical) : List<&2,F32>}
      %Equal.sym(U32,(U32.from_nat(start) + 1 : U32),U32.from_nat(1n+start),Bounds.add_one(U32.from_nat(start))) :
        {original(logical,U32.from_nat(start)) <> Tree.read_sequence(~F32,rest,_,Buffer.tree(~F32,source)) == WindowContents.image_values(~Logical,~original,1n+rest,start,logical) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => original(logical,U32.from_nat(start)) <> tail,
        Tree.read_sequence(~F32,rest,U32.from_nat(1n+start),Buffer.tree(~F32,source)),WindowContents.image_values(~Logical,~original,rest,1n+start,logical),
        prefix(~Logical,~original,logical,source,rest,1n+start,values,Order.advance_room(start,rest,Buffer.length(~F32,source),room)))

def single_length(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},workers}),matched: {Native.input_matched(input) == True{} : Bool},one: {PublicFrame.input_images(input) == 1 : U32}) ->
  {Buffer.length(~F32,source) == DenseSources.image_size(~Logical,sources_context(~Logical,logical,activation,input,weights,bias)) : Nat}:
  +size = DenseSources.image_size(~Logical,sources_context(~Logical,logical,activation,input,weights,bias))
  Equal.trans(Nat,Buffer.length(~F32,source),Nat.mul(U32.to_nat(PublicFrame.input_images(input)),size),size,
    whole_length(~Logical,logical,activation,input,weights,bias,source,fill,workers,normal,matched),
    %Equal.sym(U32,PublicFrame.input_images(input),1,one) : {Nat.mul(U32.to_nat(_),size) == size : Nat}
    Addition.zero_right(size))

def input_room(~Logical: Data,~original: Logical -> U32 -> F32,+sources_context: BandConvolutionValues.Context<Logical>,+source: Buffer.Model<F32>,
  +length_exact: {Buffer.length(~F32,source) == DenseSources.image_size(~Logical,sources_context) : Nat},ready: Buffer.Ready<F32,source>) ->
  {Nat.is_le(ConvSpec.length(Guards.input(DenseSources.image_problem(~Logical,~original,sources_context,0n))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,source)))) == True{} : Bool}:
  Buffer.Ready{depth_ok,room} = ready
  %Equal.sym(Nat,ConvSpec.length(Guards.input(DenseSources.image_problem(~Logical,~original,sources_context,0n))),DenseSources.image_size(~Logical,sources_context),DenseSources.image_length(~Logical,~original,sources_context,0n)) :
    {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,source)))) == True{} : Bool}
  %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,source)))) == True{} : Bool}
  room

def single(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},workers}),+matched: {Native.input_matched(input) == True{} : Bool},+one: {PublicFrame.input_images(input) == 1 : U32},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source))) ->
  DenseSingle.SourceStorage<DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n),source,weight_model,bias_model>:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  +sources_context = {sources_context(~Logical,logical,activation,input,weights,bias) : BandConvolutionValues.Context<Logical>}
  +size = DenseSources.image_size(~Logical,sources_context)
  +length_exact = single_length(~Logical,logical,activation,input,weights,bias,source,fill,workers,normal,matched,one)
  DenseSingle.source_storage(DenseSources.image_problem(~Logical,~original,sources_context,0n),source,weight_model,bias_model,
    %Equal.sym(Nat,ConvSpec.length(Guards.input(DenseSources.image_problem(~Logical,~original,sources_context,0n))),size,DenseSources.image_length(~Logical,~original,sources_context,0n)) :
      {Tree.read_sequence(~F32,_,0,Buffer.tree(~F32,source)) == Guards.input(DenseSources.image_problem(~Logical,~original,sources_context,0n)) : List<&2,F32>}
    prefix(~Logical,~original,logical,source,size,0n,values,
      %length_exact : {Nat.is_le(_,Buffer.length(~F32,source)) == True{} : Bool} Order.reflexive(Buffer.length(~F32,source))),
    sequence_values(weights,weight_model,weight_evidence,weight_exact),sequence_values(bias,bias_model,bias_evidence,bias_exact),
    input_room(~Logical,~original,sources_context,source,length_exact,DensePublic.normal_ready(Native.input_extents(input),source,fill,workers,normal)),
    sequence_room(weights,weight_model,weight_evidence,weight_exact),sequence_room(bias,bias_model,bias_evidence,bias_exact))

# Scalar fields of the actual raw native input agree with its public dense
# reference context. These equalities do not inspect a planner or owner.
def activation_field(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>) ->
  {DenseBatchExecution.image_activation(~Logical,sources_context(~Logical,logical,activation,input,weights,bias)) == activation : Activations.Activation}:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  {==}

def dimensions_field(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>) ->
  {PublicFrame.frame_dimensions(input) == DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)) : KernelShape.Dimensions}:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  {==}

# Actual native shape/owner guards produce the public batch caller contract.
# Scalar geometry equalities are separated from the opaque dense execution.
def batched_empty(+input: PublicFrame.NativeInput,+matched: {Native.input_matched(input) == True{} : Bool},
  +batch_empty: {PublicFrame.input_images(input) == 0 : U32}) -> {Native.batched(input) == True{} : Bool}:
  PublicFrame.NativeInput{+images,batched,ci,co,h,w,k,s,p} = input
  match batched:
    case True{}: {==}
    case False{}:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(
        %batch_empty : {U32.is_eq(_,1) == True{} : Bool} matched))

def empty_geometry(+input: PublicFrame.NativeInput,+matched: {Native.input_matched(input) == True{} : Bool},
  +batch_empty: {PublicFrame.input_images(input) == 0 : U32}) ->
  {PublicFrame.geometry(input) == TensorConvolution.Geometry{0,True{},PublicFrame.frame_dimensions(input)} : TensorConvolution.Geometry}:
  Equal.trans(TensorConvolution.Geometry,PublicFrame.geometry(input),TensorConvolution.Geometry{PublicFrame.input_images(input),Native.batched(input),PublicFrame.frame_dimensions(input)},TensorConvolution.Geometry{0,True{},PublicFrame.frame_dimensions(input)},Native.input_geometry_parts(input),
    %Equal.sym(U32,PublicFrame.input_images(input),0,batch_empty) :
      {TensorConvolution.Geometry{_,Native.batched(input),PublicFrame.frame_dimensions(input)} == TensorConvolution.Geometry{0,True{},PublicFrame.frame_dimensions(input)} : TensorConvolution.Geometry}
    Equal.cong(Bool,TensorConvolution.Geometry,batched => TensorConvolution.Geometry{0,batched,PublicFrame.frame_dimensions(input)},Native.batched(input),True{},batched_empty(input,matched,batch_empty)))

def batched_many(+input: PublicFrame.NativeInput,+matched: {Native.input_matched(input) == True{} : Bool},
  +batch_many: {Nat.is_lt(1n,U32.to_nat(PublicFrame.input_images(input))) == True{} : Bool}) -> {Native.batched(input) == True{} : Bool}:
  PublicFrame.NativeInput{+images,batched,ci,co,h,w,k,s,p} = input
  match batched:
    case True{}: {==}
    case False{}:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(
        %ShapeLaws.native_equal(images,1,matched) : {Nat.is_lt(1n,U32.to_nat(_)) == True{} : Bool} batch_many))

def geometry_field(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +matched: {Native.input_matched(input) == True{} : Bool},+batch_many: {Nat.is_lt(1n,U32.to_nat(PublicFrame.input_images(input))) == True{} : Bool}) ->
  {PublicFrame.geometry(input) == TensorConvolution.Geometry{PublicFrame.input_images(input),True{},DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n))} : TensorConvolution.Geometry}:
  Equal.trans(TensorConvolution.Geometry,PublicFrame.geometry(input),TensorConvolution.Geometry{PublicFrame.input_images(input),Native.batched(input),PublicFrame.frame_dimensions(input)},TensorConvolution.Geometry{PublicFrame.input_images(input),True{},DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n))},Native.input_geometry_parts(input),
    %Equal.sym(Bool,Native.batched(input),True{},batched_many(input,matched,batch_many)) :
      {TensorConvolution.Geometry{PublicFrame.input_images(input),_,PublicFrame.frame_dimensions(input)} == TensorConvolution.Geometry{PublicFrame.input_images(input),True{},DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n))} : TensorConvolution.Geometry}
    Equal.cong(KernelShape.Dimensions,TensorConvolution.Geometry,dimensions => TensorConvolution.Geometry{PublicFrame.input_images(input),True{},dimensions},PublicFrame.frame_dimensions(input),DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)),dimensions_field(~Logical,~original,logical,activation,input,weights,bias)))

def room(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+limits: Machine.Limits,
  +supported: {PublicFrame.machine(input,limits) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(PublicFrame.input_images(input)),DenseBatchExecution.reference_volume(~Logical,sources_context(~Logical,logical,activation,input,weights,bias))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  Boolean.right(KernelShape.supports_packed(limits,PublicFrame.frame_dimensions(input),Caller.input_length(PublicFrame.frame_dimensions(input)),Configuration.weight_size(PublicFrame.frame_dimensions(input))),
    Nat.is_le(LeafValues.output_elements(images,PublicFrame.frame_dimensions(input)),U32.to_nat(Machine.capacity(limits))),supported)

def batch_public_input(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,source: Buffer.Model<F32>,weight_model: Buffer.Model<F32>,bias_model: Buffer.Model<F32>,workers: U32) -> DensePublic.Input<Logical>:
  DensePublic.Input{sources_context(~Logical,logical,activation,input,weights,bias),PublicFrame.input_images(input),source,weight_model,bias_model,workers}

def result(~custom: F32 -> F32,+activation: Activations.Activation,+input: PublicFrame.NativeInput,source: Buffer.Model<F32>,weight_model: Buffer.Model<F32>,bias_model: Buffer.Model<F32>,workers: U32) -> Tensors.Tensor<F32>:
  TensorConvolution.execute_dense(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers)

def batch_empty(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+workers: U32,
  +matched: {Native.input_matched(input) == True{} : Bool},+batch_empty: {PublicFrame.input_images(input) == 0 : U32}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),sources_context(~Logical,logical,activation,input,weights,bias),result(~custom,activation,input,source,weight_model,bias_model,workers)):
  %Equal.sym(Tensors.Tensor<F32>,result(~custom,activation,input,source,weight_model,bias_model,workers),TensorConvolution.execute_dense(~custom,activation,TensorConvolution.Geometry{0,True{},PublicFrame.frame_dimensions(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers),
    Equal.cong(TensorConvolution.Geometry,Tensors.Tensor<F32>,geometry => TensorConvolution.execute_dense(~custom,activation,geometry,Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers),PublicFrame.geometry(input),TensorConvolution.Geometry{0,True{},PublicFrame.frame_dimensions(input)},empty_geometry(input,matched,batch_empty))) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),sources_context(~Logical,logical,activation,input,weights,bias),_)
  Batch.empty_dense(~custom,~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),activation,PublicFrame.frame_dimensions(input),Buffer.pack(~F32,source),weight_model,bias_model,workers)

def header(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+workers: U32,
  +matched: {Native.input_matched(input) == True{} : Bool},+batch_many: {Nat.is_lt(1n,U32.to_nat(PublicFrame.input_images(input))) == True{} : Bool}) ->
  {result(~custom,activation,input,source,weight_model,bias_model,workers) == DensePublic.batch_result(~custom,~Logical,~original,batch_public_input(~Logical,logical,activation,input,weights,bias,source,weight_model,bias_model,workers)) : Tensors.Tensor<F32>}:
  +context = sources_context(~Logical,logical,activation,input,weights,bias)
  +geometry = {TensorConvolution.Geometry{PublicFrame.input_images(input),True{},DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,context,0n))} : TensorConvolution.Geometry}
  Equal.trans(Tensors.Tensor<F32>,result(~custom,activation,input,source,weight_model,bias_model,workers),TensorConvolution.execute_dense(~custom,activation,geometry,Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers),DensePublic.batch_result(~custom,~Logical,~original,batch_public_input(~Logical,logical,activation,input,weights,bias,source,weight_model,bias_model,workers)),
    Equal.cong(TensorConvolution.Geometry,Tensors.Tensor<F32>,shape => TensorConvolution.execute_dense(~custom,activation,shape,Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers),PublicFrame.geometry(input),geometry,geometry_field(~Logical,~original,logical,activation,input,weights,bias,matched,batch_many)),
    Equal.cong(Activations.Activation,Tensors.Tensor<F32>,operation => TensorConvolution.execute_dense(~custom,operation,geometry,Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers),activation,DenseBatchExecution.image_activation(~Logical,context),Equal.sym(Activations.Activation,DenseBatchExecution.image_activation(~Logical,context),activation,activation_field(~Logical,logical,activation,input,weights,bias))))

def batch_prepared(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+limits: Machine.Limits,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+workers: U32,
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,limits) == True{} : Bool},
  +batch_many: {Nat.is_lt(1n,U32.to_nat(PublicFrame.input_images(input))) == True{} : Bool},
  receipt: PreparedSources<Logical,original,sources_context(~Logical,logical,activation,input,weights,bias),limits,PublicFrame.input_images(input),source,weight_model,bias_model>) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),sources_context(~Logical,logical,activation,input,weights,bias),result(~custom,activation,input,source,weight_model,bias_model,workers)):
  PreparedSources{+tiles,+owner,+sources} = receipt
  %Equal.sym(Tensors.Tensor<F32>,result(~custom,activation,input,source,weight_model,bias_model,workers),DensePublic.batch_result(~custom,~Logical,~original,batch_public_input(~Logical,logical,activation,input,weights,bias,source,weight_model,bias_model,workers)),header(~custom,~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,workers,matched,batch_many)) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),sources_context(~Logical,logical,activation,input,weights,bias),_)
  DensePublic.batch_many(~custom,~Logical,~original,batch_public_input(~Logical,logical,activation,input,weights,bias,source,weight_model,bias_model,workers),limits,tiles,sources,owner,room(~Logical,logical,activation,input,weights,bias,limits,supported),valid_zero(~Logical,~original,logical,activation,input,weights,bias,validated),supported_zero(~Logical,~original,logical,activation,input,weights,bias,limits,validated,supported),batch_many)

def batch_many(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+limits: Machine.Limits,
  +capacity_actual: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +source_values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,limits) == True{} : Bool},
  +batch_many: {Nat.is_lt(1n,U32.to_nat(PublicFrame.input_images(input))) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),sources_context(~Logical,logical,activation,input,weights,bias),result(~custom,activation,input,source,weight_model,bias_model,workers)):
  batch_prepared(~custom,~Logical,~original,logical,activation,input,weights,bias,limits,source,weight_model,bias_model,workers,matched,validated,supported,batch_many,
    prepared_reorder(~Logical,~original,logical,activation,input,weights,bias,limits,capacity_actual,source,weight_model,bias_model,fill,input_workers,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_exact,source_values,validated,supported,
      DenseSources.reordered(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),0n)),weight_model)))

# Dense checked acceptance reuses the shared parameter and allocation guards.
# The incoming Normal receipt supplies the actual complete source length;
# no output guard, successful execution or caller equality is assumed.
def machine_matches(+input: PublicFrame.NativeInput,+limits: Machine.Limits) ->
  {MachineCaller.predicate(PublicFrame.geometry(input),limits) == PublicFrame.machine(input,limits) : Bool}:
  PublicFrame.NativeInput{images,batched,channels,outputs,height,width,kernel,stride,padding} = input
  {==}

def parameters_valid(+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,
  +weight_length: {Buffer.length(~F32,weight_model) == ConvSpec.length(weights) : Nat},+bias_length: {Buffer.length(~F32,bias_model) == ConvSpec.length(bias) : Nat},
  validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool}) ->
  {PublicFrame.valid_parameters(PublicFrame.frame_dimensions(input),weight_model,bias_model) == True{} : Bool}:
  PublicFrame.NativeInput{images,batched,channels,outputs,height,width,kernel,stride,padding} = input
  %Equal.sym(Nat,Buffer.length(~F32,weight_model),ConvSpec.length(weights),weight_length) :
    {KernelShape.valid(PublicFrame.frame_dimensions(input),Caller.input_length(PublicFrame.frame_dimensions(input)),_,Buffer.length(~F32,bias_model)) == True{} : Bool}
  %Equal.sym(Nat,Buffer.length(~F32,bias_model),ConvSpec.length(bias),bias_length) :
    {KernelShape.valid(PublicFrame.frame_dimensions(input),Caller.input_length(PublicFrame.frame_dimensions(input)),ConvSpec.length(weights),_) == True{} : Bool}
  validated

def source_equal(+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},workers}),
  matched: {Native.input_matched(input) == True{} : Bool}) ->
  {Nat.is_eq(Buffer.length(~F32,source),BandGather.elements(PublicFrame.source_geometry(input))) == True{} : Bool}:
  %Equal.sym(Nat,Buffer.length(~F32,source),BandGather.elements(PublicFrame.source_geometry(input)),DensePublic.length(input,source,fill,workers,normal,matched)) :
    {Nat.is_eq(_,BandGather.elements(PublicFrame.source_geometry(input))) == True{} : Bool}
  Order.equal_reflexive(BandGather.elements(PublicFrame.source_geometry(input)))

def guard(+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>) -> Bool:
  PublicFrame.valid_parameters(PublicFrame.frame_dimensions(input),weights,bias) &&
    Nat.is_eq(Buffer.length(~F32,source),BandGather.elements(PublicFrame.source_geometry(input))) &&
    Nat.is_le(Buffer.length(~F32,source),U32.to_nat(U32.shln(1,Buffer.depth(~F32,source)))) &&
    Nat.is_le(Buffer.length(~F32,weights),U32.to_nat(U32.shln(1,Buffer.depth(~F32,weights)))) &&
    Nat.is_le(Buffer.length(~F32,bias),U32.to_nat(U32.shln(1,Buffer.depth(~F32,bias))))

def accepted(+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},
  +weight_length: {Buffer.length(~F32,weight_model) == ConvSpec.length(weights) : Nat},+bias_length: {Buffer.length(~F32,bias_model) == ConvSpec.length(bias) : Nat},
  weight_ready: Buffer.Ready<F32,weight_model>,bias_ready: Buffer.Ready<F32,bias_model>) ->
  {guard(input,source,weight_model,bias_model) == True{} : Bool}:
  Boolean.all_hold([
      Nat.is_le(Buffer.length(~F32,bias_model),U32.to_nat(U32.shln(1,Buffer.depth(~F32,bias_model)))),
      Nat.is_le(Buffer.length(~F32,weight_model),U32.to_nat(U32.shln(1,Buffer.depth(~F32,weight_model)))),
      Nat.is_le(Buffer.length(~F32,source),U32.to_nat(U32.shln(1,Buffer.depth(~F32,source)))),
      Nat.is_eq(Buffer.length(~F32,source),BandGather.elements(PublicFrame.source_geometry(input))),
      PublicFrame.valid_parameters(PublicFrame.frame_dimensions(input),weight_model,bias_model)],
    Shape.Both{PublicFrame.allocation_room(~F32,bias_model,bias_ready),Shape.Both{PublicFrame.allocation_room(~F32,weight_model,weight_ready),
      Shape.Both{PublicFrame.allocation_room(~F32,source,DensePublic.normal_ready(Native.input_extents(input),source,fill,workers,normal)),
        Shape.Both{source_equal(input,source,fill,workers,normal,matched),Shape.Both{parameters_valid(input,weights,bias,weight_model,bias_model,weight_length,bias_length,validated),Unit{}}}}}})

def checked_header(~custom: F32 -> F32,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) ==
    TensorConvolution.supported(~custom,activation,guard(input,source,weights,bias),PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) : Tensors.Tensor<F32>}:
  PublicFrame.NativeInput{images,batched,channels,outputs,height,width,kernel,stride,padding} = input
  Buffer.Model{source_tree,source_length,Certificate.Certificate{source_depth,source_equation}} = source
  Buffer.Model{weight_tree,weight_length,Certificate.Certificate{weight_depth,weight_equation}} = weights
  Buffer.Model{bias_tree,bias_length,Certificate.Certificate{bias_depth,bias_equation}} = bias
  {==}

def supported(~custom: F32 -> F32,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  support: {PublicFrame.machine(input,Default.limits()) == True{} : Bool}) ->
  {TensorConvolution.supported(~custom,activation,True{},PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) ==
    TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers) : Tensors.Tensor<F32>}:
  %Equal.sym(Bool,MachineCaller.predicate(PublicFrame.geometry(input),Default.limits()),True{},
    %Equal.sym(Bool,MachineCaller.predicate(PublicFrame.geometry(input),Default.limits()),PublicFrame.machine(input,Default.limits()),machine_matches(input,Default.limits())) : {_ == True{} : Bool}
    support) :
    {TensorConvolution.machine(~custom,activation,_,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) ==
      TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers) : Tensors.Tensor<F32>}
  {==}

def checked(~custom: F32 -> F32,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+support: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +weight_length: {Buffer.length(~F32,weight_model) == ConvSpec.length(weights) : Nat},+bias_length: {Buffer.length(~F32,bias_model) == ConvSpec.length(bias) : Nat},
  weight_ready: Buffer.Ready<F32,weight_model>,bias_ready: Buffer.Ready<F32,bias_model>) ->
  {TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),Native.input_extents(input),fill,workers) ==
    TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers) : Tensors.Tensor<F32>}:
  Equal.trans(Tensors.Tensor<F32>,TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),Native.input_extents(input),fill,workers),
    TensorConvolution.supported(~custom,activation,guard(input,source,weight_model,bias_model),PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),Native.input_extents(input),fill,workers),
    TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers),checked_header(~custom,activation,input,source,weight_model,bias_model,Native.input_extents(input),fill,workers),
    %Equal.sym(Bool,guard(input,source,weight_model,bias_model),True{},accepted(input,weights,bias,source,weight_model,bias_model,fill,input_workers,normal,matched,validated,weight_length,bias_length,weight_ready,bias_ready)) :
      {TensorConvolution.supported(~custom,activation,_,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),Native.input_extents(input),fill,workers) ==
        TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers) : Tensors.Tensor<F32>}
    supported(~custom,activation,input,source,weight_model,bias_model,Native.input_extents(input),fill,workers,support))

def parameter_normal(+model: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,model),shape,fill,Tensors.Ready{},workers})) -> Parameters.Parameter<Buffer.pack(~F32,model)>:
  Parameters.parameter(Buffer.pack(~F32,model),
    %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.content(~F32,model),Buffer.packed_contents(~F32,model)) : Buffer.ContentReady<F32,_>
    Buffer.content_ready(~F32,model,DensePublic.normal_ready(shape,model,fill,workers,normal)))

def checked_parameters(~custom: F32 -> F32,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+weight_owner: Buffer.Model<F32>,+bias_owner: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+support: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +weight_parameter: Parameters.Parameter<Buffer.pack(~F32,weight_owner)>,+bias_parameter: Parameters.Parameter<Buffer.pack(~F32,bias_owner)>,
  validated: {PublicFrame.frame_valid(input,Parameters.parameter_values(Buffer.pack(~F32,weight_owner),weight_parameter),Parameters.parameter_values(Buffer.pack(~F32,bias_owner),bias_parameter)) == True{} : Bool}) ->
  {TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weight_owner),Buffer.pack(~F32,bias_owner),Native.input_extents(input),fill,workers) ==
    TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_owner),Buffer.pack(~F32,bias_owner),workers) : Tensors.Tensor<F32>}:
  Parameters.Parameter{+weights,+weight_equation,+weight_ready,+weight_values,+weight_length,weight_evidence,weight_exact} = weight_parameter
  Parameters.Parameter{+bias,+bias_equation,+bias_ready,+bias_values,+bias_length,bias_evidence,bias_exact} = bias_parameter
  %weight_equation :
    {TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),_,Buffer.pack(~F32,bias_owner),Native.input_extents(input),fill,workers) ==
      TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),_,Buffer.pack(~F32,bias_owner),workers) : Tensors.Tensor<F32>}
  %bias_equation :
    {TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weights),_,Native.input_extents(input),fill,workers) ==
      TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weights),_,workers) : Tensors.Tensor<F32>}
  checked(~custom,activation,input,weight_values,bias_values,source,weights,bias,fill,input_workers,workers,normal,matched,validated,support,
    Equal.sym(Nat,ConvSpec.length(weight_values),Buffer.length(~F32,weights),weight_length),Equal.sym(Nat,ConvSpec.length(bias_values),Buffer.length(~F32,bias),bias_length),weight_ready,bias_ready)

def checked_normal(~custom: F32 -> F32,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+weight_workers: U32,+bias_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),
  +weight_normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,weights),Native.weight_extents(input),0.0,Tensors.Ready{},weight_workers}),
  +bias_normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,bias),Native.bias_extents(input),0.0,Tensors.Ready{},bias_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+support: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  validated: {PublicFrame.frame_valid(input,
    Parameters.parameter_values(Buffer.pack(~F32,weights),parameter_normal(weights,Native.weight_extents(input),0.0,weight_workers,weight_normal)),
    Parameters.parameter_values(Buffer.pack(~F32,bias),parameter_normal(bias,Native.bias_extents(input),0.0,bias_workers,bias_normal))) == True{} : Bool}) ->
  {TensorConvolution.checked(~custom,activation,Some{PublicFrame.geometry(input)},Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),Native.input_extents(input),fill,workers) ==
    TensorConvolution.execute(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers) : Tensors.Tensor<F32>}:
  checked_parameters(~custom,activation,input,source,weights,bias,fill,input_workers,workers,normal,matched,support,
    parameter_normal(weights,Native.weight_extents(input),0.0,weight_workers,weight_normal),parameter_normal(bias,Native.bias_extents(input),0.0,bias_workers,bias_normal),validated)

# The actual decoded CHW/NCHW one-image call reuses the public dense leaf.
# Its input storage and guards are constructed from finite source contracts;
# actual default planning remains behind the accepted argument observation.
def problem_exact(~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>) ->
  {DensePublic.reference_problem(~Logical,~original,context) == DenseSources.image_problem(~Logical,~original,context,0n) : ConvSpec.Problem}:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  {==}

def activation_exact(~Logical: Data,+context: BandConvolutionValues.Context<Logical>) ->
  {DensePublic.reference_activation(~Logical,context) == DenseBatchExecution.image_activation(~Logical,context) : Activations.Activation}:
  BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  {==}

def single_public_input(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  source: Buffer.Model<F32>,weight_model: Buffer.Model<F32>,bias_model: Buffer.Model<F32>,workers: U32) -> DensePublic.PublicInput<Logical>:
  DensePublic.PublicInput{sources_context(~Logical,logical,activation,input,weights,bias),Native.batched(input),source,weight_model,bias_model,workers}

def geometry_exact(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  one: {PublicFrame.input_images(input) == 1 : U32}) ->
  {TensorConvolution.Geometry{1,Native.batched(input),DenseSingle.dimensions(DensePublic.reference_problem(~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias)))} == PublicFrame.geometry(input) : TensorConvolution.Geometry}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+height,+width,+kernel,+stride,+padding} = input
  +context = {sources_context(~Logical,logical,activation,input,weights,bias) : BandConvolutionValues.Context<Logical>}
  Equal.trans(TensorConvolution.Geometry,TensorConvolution.Geometry{1,batched,DenseSingle.dimensions(DensePublic.reference_problem(~Logical,~original,context))},TensorConvolution.Geometry{1,batched,PublicFrame.frame_dimensions(input)},PublicFrame.geometry(input),
    Equal.cong(KernelShape.Dimensions,TensorConvolution.Geometry,dimensions => TensorConvolution.Geometry{1,batched,dimensions},DenseSingle.dimensions(DensePublic.reference_problem(~Logical,~original,context)),PublicFrame.frame_dimensions(input),
      Equal.trans(KernelShape.Dimensions,DenseSingle.dimensions(DensePublic.reference_problem(~Logical,~original,context)),DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,context,0n)),PublicFrame.frame_dimensions(input),
        Equal.cong(ConvSpec.Problem,KernelShape.Dimensions,DenseSingle.dimensions,DensePublic.reference_problem(~Logical,~original,context),DenseSources.image_problem(~Logical,~original,context,0n),problem_exact(~Logical,~original,context)),
        Equal.sym(KernelShape.Dimensions,PublicFrame.frame_dimensions(input),DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,context,0n)),dimensions_field(~Logical,~original,logical,activation,input,weights,bias)))),
    Equal.cong(U32,TensorConvolution.Geometry,count => TensorConvolution.Geometry{count,batched,PublicFrame.frame_dimensions(input)},1,images,Equal.sym(U32,images,1,one)))

def result_matches(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+workers: U32,+one: {PublicFrame.input_images(input) == 1 : U32}) ->
  {DensePublic.image_result(~custom,~Logical,~original,single_public_input(~Logical,logical,activation,input,weights,bias,source,weight_model,bias_model,workers)) ==
    TensorConvolution.execute_dense(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers) : Tensors.Tensor<F32>}:
  +context = {sources_context(~Logical,logical,activation,input,weights,bias) : BandConvolutionValues.Context<Logical>}
  Equal.trans(Tensors.Tensor<F32>,DensePublic.image_result(~custom,~Logical,~original,single_public_input(~Logical,logical,activation,input,weights,bias,source,weight_model,bias_model,workers)),
    TensorConvolution.execute_dense(~custom,DensePublic.reference_activation(~Logical,context),PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers),
    TensorConvolution.execute_dense(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers),
    Equal.cong(TensorConvolution.Geometry,Tensors.Tensor<F32>,geometry => TensorConvolution.execute_dense(~custom,DensePublic.reference_activation(~Logical,context),geometry,Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers),
      TensorConvolution.Geometry{1,Native.batched(input),DenseSingle.dimensions(DensePublic.reference_problem(~Logical,~original,context))},PublicFrame.geometry(input),geometry_exact(~Logical,~original,logical,activation,input,weights,bias,one)),
    Equal.cong(Activations.Activation,Tensors.Tensor<F32>,activation => TensorConvolution.execute_dense(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers),
      DensePublic.reference_activation(~Logical,context),activation,Equal.trans(Activations.Activation,DensePublic.reference_activation(~Logical,context),DenseBatchExecution.image_activation(~Logical,context),activation,
        activation_exact(~Logical,context),activation_field(~Logical,logical,activation,input,weights,bias))))

def single_contract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),+matched: {Native.input_matched(input) == True{} : Bool},+one: {PublicFrame.input_images(input) == 1 : U32},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute_dense(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers)):
  +context = {sources_context(~Logical,logical,activation,input,weights,bias) : BandConvolutionValues.Context<Logical>}
  %result_matches(~custom,~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,workers,one) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,_)
  DensePublic.image_single_image(~custom,~Logical,~original,single_public_input(~Logical,logical,activation,input,weights,bias,source,weight_model,bias_model,workers),Default.limits(),
    %Equal.sym(ConvSpec.Problem,DensePublic.reference_problem(~Logical,~original,context),DenseSources.image_problem(~Logical,~original,context,0n),problem_exact(~Logical,~original,context)) : DenseSingle.SourceStorage<_,source,weight_model,bias_model>
    single(~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,fill,input_workers,normal,matched,one,weight_evidence,bias_evidence,weight_exact,bias_exact,values),
    %Equal.sym(ConvSpec.Problem,DensePublic.reference_problem(~Logical,~original,context),DenseSources.image_problem(~Logical,~original,context,0n),problem_exact(~Logical,~original,context)) : {ConvSpec.valid(_) == True{} : Bool}
    valid_zero(~Logical,~original,logical,activation,input,weights,bias,validated),
    %Equal.sym(ConvSpec.Problem,DensePublic.reference_problem(~Logical,~original,context),DenseSources.image_problem(~Logical,~original,context,0n),problem_exact(~Logical,~original,context)) : {NativeConv.supports_packed_with_limits(Default.limits(),_) == True{} : Bool}
    supported_zero(~Logical,~original,logical,activation,input,weights,bias,Default.limits(),validated,supported))

def single_layout(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),+matched: {Native.input_matched(input) == True{} : Bool},+one: {PublicFrame.input_images(input) == 1 : U32},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),+validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute_dense(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers)):
  InitialLayout.from_dense_header(~custom,~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.execute_dense(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers),
    InitialLayout.execute_dense_header(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),bias_model,workers),
    single_contract(~custom,~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,one,weight_evidence,bias_evidence,weight_exact,bias_exact,values,validated,supported))

# The actual decoded image count selects the proven native dense procedure.
# Default limits are observed through a formal receipt; finite input and
# parameter contracts supply the leaf premises without an execution oracle.
def classified(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+limits: Machine.Limits,
  +limits_exact: {limits == Default.limits() : Machine.Limits},
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,limits) == True{} : Bool},
  classification: DenseSources.CountKind<PublicFrame.input_images(input)>) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),sources_context(~Logical,logical,activation,input,weights,bias),result(~custom,activation,input,source,weight_model,bias_model,workers)):
  match classification:
    case DenseSources.EmptyCount{empty}:
      batch_empty(~custom,~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,workers,matched,empty)
    case DenseSources.SingleCount{one}:
      single_contract(~custom,~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,one,weight_evidence,bias_evidence,weight_exact,bias_exact,values,validated,
        %limits_exact : {PublicFrame.machine(input,_) == True{} : Bool} supported)
    case DenseSources.BatchCount{many}:
      batch_many(~custom,~Logical,~original,logical,activation,input,weights,bias,limits,Equal.cong(Machine.Limits,U32,Machine.capacity,limits,Default.limits(),limits_exact),source,weight_model,bias_model,fill,input_workers,workers,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_exact,values,validated,supported,many)

def observed_limits(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  receipt: LimitsReceipt) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),sources_context(~Logical,logical,activation,input,weights,bias),result(~custom,activation,input,source,weight_model,bias_model,workers)):
  LimitsReceipt{+limits,+actual} = receipt
  classified(~custom,~Logical,~original,logical,activation,input,weights,bias,limits,actual,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_exact,values,validated,
    %Equal.sym(Machine.Limits,limits,Default.limits(),actual) : {PublicFrame.machine(input,_) == True{} : Bool} supported,
    DenseSources.classify(PublicFrame.input_images(input)))

def native_contract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),sources_context(~Logical,logical,activation,input,weights,bias),result(~custom,activation,input,source,weight_model,bias_model,workers)):
  observed_limits(~custom,~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_exact,values,validated,supported,default_limits())

def contract_layout(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),result(~custom,activation,input,source,weight_model,bias_model,workers)):
  InitialLayout.from_dense_header(~custom,~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),result(~custom,activation,input,source,weight_model,bias_model,workers),
    InitialLayout.execute_dense_header(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),bias_model,workers),
    native_contract(~custom,~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,fill,input_workers,workers,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_exact,values,validated,supported))

def initial_zero(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,+parallel: Bool,+time: F32,
  +source: Buffer.Model<F32>,+weight_model: Buffer.Model<F32>,+bias_model: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Ready{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_exact: {Buffer.tree(~F32,weight_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool}) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),
    TensorConvolution.initial_layout(~custom,activation,BandPlanning.Plan{0,parallel,time},PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),workers)):
  InitialLayout.dense_workers(~custom,~Logical,~original,sources_context(~Logical,logical,activation,input,weights,bias),workers,result(~custom,activation,input,source,weight_model,bias_model,1),
    InitialLayout.execute_dense_header(~custom,activation,PublicFrame.geometry(input),Buffer.pack(~F32,source),Buffer.pack(~F32,weight_model),bias_model,1),
    native_contract(~custom,~Logical,~original,logical,activation,input,weights,bias,source,weight_model,bias_model,fill,input_workers,1,normal,matched,weight_evidence,bias_evidence,weight_exact,bias_exact,values,validated,supported))
