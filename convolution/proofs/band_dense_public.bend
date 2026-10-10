# Public dense callers: initialization, source shapes, reference agreement and
# the planner schedule.
import Base
import ./band_dense_batch_execution.bend as DenseBatchExecution
import ./band_dense_single.bend as DenseSingle
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_dense_contract.bend as Dense
import ../../lib/tensor.bend as Tensors
import ../specification.bend as ConvSpec
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_dense_convolution.bend as DenseConvolution
import ./microkernel_inference_refinement.bend as Inference
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../inference.bend as NativeConv
import ./band_dense_sources.bend as DenseSources
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_public_frame.bend as PublicFrame
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/storage_buffer.bend as Storage
import ./band_native_caller.bend as Native
import ./band_tensor_execution.bend as TensorExecution
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./prepared_output_readback.bend as PreparedOutputReadback
import ./reduction_coordinate_bounds.bend as Coordinates
import ../../lib/default_machine_limits.bend as Default
import ../../lib/kernel_gemm_buffer.bend as GemmBuffer
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/tensor_band_count.bend as BandCount
import ../../lib/proofs/tensor_band_dense.bend as BandDense
import ../../lib/proofs/tensor_count_refinement.bend as Count
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ./band_leaf_caller.bend as Leaf
import ./band_native_source_domain.bend as BandNativeSourceDomain
import ./band_normal_caller.bend as NormalCaller
import ./band_output_metadata.bend as OutputMetadata
import ./band_output_normal.bend as BandOutputNormal
import ./concrete_support_guard.bend as Support
import ./convolution_guard_facts.bend as Guards
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ./supported_padded_coordinates.bend as Padded
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_copy.bend as BandCopy
import ../../lib/proofs/tensor_band_map.bend as BandMap
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_window.bend as Window

# The genuine many-image procedure reorders its weights once, allocates the
# whole output, and consumes the batch contract with the actual selected depth.
def reorder_header(+problem: ConvSpec.Problem,+weights: Buffer.Model<F32>) ->
  {GemmBuffer.reorder(DenseSources.rows(Inference.work(problem)),DenseSources.inner(Inference.work(problem)),Buffer.pack(~F32,weights)) == GemmBuffer.reorder(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Buffer.pack(~F32,weights)) : Storage.Buffer<Tile.Tile8>}:
  ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,ws,bs} = problem
  {==}

def images_contract_many(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+limits: Machine.Limits,+images: U32,+depth: U32,+input: Buffer.Model<F32>,+tiles: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32,
  +sources: DenseBatchExecution.Sources<Logical,original,context,limits,images,input,tiles,weights,bias>,
  +tiles_owner: {Buffer.pack(~Tile.Tile8,tiles) == GemmBuffer.reorder(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,context,0n)),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,context,0n)),Buffer.pack(~F32,weights)) : Storage.Buffer<Tile.Tile8>},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(images),DenseBatchExecution.reference_volume(~Logical,context)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count_bound: {Nat.is_lt(1n,U32.to_nat(images)) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
    TensorConvolution.execute_dense_at_depth(~custom,DenseBatchExecution.image_activation(~Logical,context),images,depth,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),U32.from_nat(DenseSources.image_size(~Logical,context)),DenseBatchExecution.size(~Logical,~original,context),DenseBatchExecution.extents(~Logical,~original,context,images),Buffer.pack(~F32,input),GemmBuffer.reorder(DenseSources.rows(Inference.work(DenseSources.image_problem(~Logical,~original,context,0n))),DenseSources.inner(Inference.work(DenseSources.image_problem(~Logical,~original,context,0n))),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)):
  +problem = DenseSources.image_problem(~Logical,~original,context,0n)
  %Equal.sym(Storage.Buffer<F32>,TensorConvolution.images(~custom,DenseBatchExecution.image_activation(~Logical,context),images,depth,Inference.work(problem),U32.from_nat(DenseSources.image_size(~Logical,context)),DenseBatchExecution.size(~Logical,~original,context),Buffer.pack(~F32,input),GemmBuffer.reorder(DenseSources.rows(Inference.work(problem)),DenseSources.inner(Inference.work(problem)),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias)),
    TensorConvolution.batch(~custom,DenseBatchExecution.image_activation(~Logical,context),U32.to_nat(images),0,depth,Inference.work(problem),U32.from_nat(DenseSources.image_size(~Logical,context)),DenseBatchExecution.size(~Logical,~original,context),Buffer.pack(~F32,input),GemmBuffer.reorder(DenseSources.rows(Inference.work(problem)),DenseSources.inner(Inference.work(problem)),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(images * DenseBatchExecution.size(~Logical,~original,context) : U32))),
    DenseSources.many(~custom,DenseBatchExecution.image_activation(~Logical,context),images,depth,Inference.work(problem),U32.from_nat(DenseSources.image_size(~Logical,context)),DenseBatchExecution.size(~Logical,~original,context),input,weights,bias,count_bound)) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,Tensors.Tensor{_,DenseBatchExecution.extents(~Logical,~original,context,images),0.0,Tensors.Usable{},workers})
  %Equal.sym(Storage.Buffer<Tile.Tile8>,GemmBuffer.reorder(DenseSources.rows(Inference.work(problem)),DenseSources.inner(Inference.work(problem)),Buffer.pack(~F32,weights)),GemmBuffer.reorder(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Buffer.pack(~F32,weights)),reorder_header(problem,weights)) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,Tensors.Tensor{
      TensorConvolution.batch(~custom,DenseBatchExecution.image_activation(~Logical,context),U32.to_nat(images),0,depth,Inference.work(problem),U32.from_nat(DenseSources.image_size(~Logical,context)),DenseBatchExecution.size(~Logical,~original,context),Buffer.pack(~F32,input),_,Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(images * DenseBatchExecution.size(~Logical,~original,context) : U32))),DenseBatchExecution.extents(~Logical,~original,context,images),0.0,Tensors.Usable{},workers})
  %tiles_owner :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,Tensors.Tensor{
      TensorConvolution.batch(~custom,DenseBatchExecution.image_activation(~Logical,context),U32.to_nat(images),0,depth,Inference.work(problem),U32.from_nat(DenseSources.image_size(~Logical,context)),DenseBatchExecution.size(~Logical,~original,context),Buffer.pack(~F32,input),_,Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(images * DenseBatchExecution.size(~Logical,~original,context) : U32))),DenseBatchExecution.extents(~Logical,~original,context,images),0.0,Tensors.Usable{},workers})
  DenseBatchExecution.batch(~custom,~Logical,~original,context,limits,images,depth,input,tiles,weights,bias,workers,sources,room)

def content_value(~Element: Data,+source: Buffer.Model<Element>,+index: U32) ->
  {Tree.read(Element,Buffer.tree(~Element,source),index) == Buffer.value(~Element,index,Buffer.content(~Element,source)) : Element}:
  Buffer.Model{tree,length,certificate} = source
  {==}

def content_length(~Element: Data,+source: Buffer.Model<Element>) ->
  {Buffer.content_length(~Element,Buffer.content(~Element,source)) == Buffer.length(~Element,source) : Nat}:
  Buffer.Model{tree,length,certificate} = source
  {==}

def source_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Buffer.Model<Element>,+index: U32,
  source_values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,source)),
  inside: {Nat.is_lt(U32.to_nat(index),Buffer.length(~Element,source)) == True{} : Bool}) ->
  {Tree.read(Element,Buffer.tree(~Element,source),index) == original(logical,index) : Element}:
  Equal.trans(Element,Tree.read(Element,Buffer.tree(~Element,source),index),Buffer.value(~Element,index,Buffer.content(~Element,source)),original(logical,index),
    content_value(~Element,source,index),
    %Arithmetic.native_roundtrip(index) : {Buffer.value(~Element,_,Buffer.content(~Element,source)) == original(logical,_) : Element}
    Dense.read(~Element,~Logical,~original,logical,Buffer.content(~Element,source),U32.to_nat(index),source_values,
      %Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,source)),Buffer.length(~Element,source),content_length(~Element,source)) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool}
      inside))

# Dense values reach each band cell through the pixel of the cell. The
# allocation domain is the logical dense count, rather than spare storage.
def dense_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+count: U32,
  +source: Buffer.Model<Element>,+target: Buffer.Model<Element>,+first: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat,
  +domain: BandGather.Domain<geometry,limits,count>,metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,target)),
  length_exact: {U32.to_nat(count) == Buffer.length(~Element,source) : Nat},
  source_values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,source)),
  cell: BandValues.At(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),Buffer.tree(~Element,source),geometry,first,rows,Buffer.tree(~Element,target),Buffer.length(~Element,target),channel,row,column),
  +channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool}) ->
  BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.tree(~Element,target),Buffer.length(~Element,target),channel,row,column):
  Shape.Both{inside,equal} = cell
  +global = U32.from_nat(BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))
  Shape.Both{inside,
    Equal.trans(Element,Tree.read(Element,Buffer.tree(~Element,target),U32.from_nat(BandValues.position(geometry,rows,channel,row,column))),Tree.read(Element,Buffer.tree(~Element,source),global),original(logical,global),
      equal,
      source_read(~Element,~Logical,~original,logical,source,global,source_values,
        %length_exact : {Nat.is_lt(U32.to_nat(global),_) == True{} : Bool}
        BandGather.pixel_word_inside(geometry,limits,count,channel,Nat.add(U32.to_nat(first),row),column,domain,channel_inside,
          BandGather.cell_row_inside(geometry,first,rows,Buffer.length(~Element,target),row,metadata,row_inside),column_inside)))}

# What a band leaf of a dense source supplies to every cell.
type DenseCells<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  DenseCells{logical: Logical,geometry: Bands.Geometry,limits: Machine.Limits,count: U32,source: Buffer.Model<Element>,target: Buffer.Model<Element>,first: U32,rows: U32,
    domain: BandGather.Domain<geometry,limits,count>,metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,target)),
    length_exact: {U32.to_nat(count) == Buffer.length(~Element,source) : Nat},
    source_values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,source)),
    target_values: BandValues.Leaf(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),Buffer.tree(~Element,source),geometry,first,rows,Buffer.tree(~Element,target),Buffer.length(~Element,target))}

def dense_geometry(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: DenseCells<Element,Logical,original>) -> Bands.Geometry:
  DenseCells{logical,geometry,limits,count,source,target,first,rows,domain,metadata,length_exact,source_values,target_values} = evidence
  geometry

def dense_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: DenseCells<Element,Logical,original>) -> U32:
  DenseCells{logical,geometry,limits,count,source,target,first,rows,domain,metadata,length_exact,source_values,target_values} = evidence
  rows

def DenseCell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: DenseCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  DenseCells{+logical,+geometry,+limits,+count,+source,+target,+first,+rows,+domain,+metadata,+length_exact,+source_values,+target_values} = evidence
  BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.tree(~Element,target),Buffer.length(~Element,target),channel,row,column)

def dense_leaf_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: DenseCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(dense_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(dense_rows(~Element,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(dense_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool}) ->
  DenseCell(~Element,~Logical,~original,evidence,channel,row,column):
  DenseCells{+logical,+geometry,+limits,+count,+source,+target,+first,+rows,+domain,+metadata,+length_exact,+source_values,+target_values} = evidence
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  dense_cell(~Element,~Logical,~original,logical,geometry,limits,count,source,target,first,rows,channel,row,column,domain,metadata,length_exact,source_values,
    BandValues.leaf_cell(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),Buffer.tree(~Element,source),geometry,first,rows,Buffer.tree(~Element,target),Buffer.length(~Element,target),channel,row,column,target_values,channel_inside,row_inside,column_inside),
    channel_inside,row_inside,column_inside)

def modeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+count: U32,+source: Buffer.Model<Element>,
  target: Model.Model<Element>,+domain: BandGather.Domain<geometry,limits,count>,+length_exact: {U32.to_nat(count) == Buffer.length(~Element,source) : Nat},
  +source_values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,source)),
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,target)),
  target_values: BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),Buffer.tree(~Element,source),geometry,Model.content(~Element,target))) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,target)):
  match target:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      +leaf = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
      BandValues.cells(~DenseCells<Element,Logical,original>,~DenseCell(~Element,~Logical,~original),~dense_geometry(~Element,~Logical,~original),~dense_rows(~Element,~Logical,~original),~dense_leaf_cell(~Element,~Logical,~original),
        DenseCells{logical,geometry,limits,count,source,leaf,first,rows,domain,metadata,length_exact,source_values,target_values})
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_values,right_values} = target_values
      Shape.Both{modeled(~Element,~Logical,~original,logical,geometry,limits,count,source,left,domain,length_exact,source_values,left_metadata,left_values),
        modeled(~Element,~Logical,~original,logical,geometry,limits,count,source,right,domain,length_exact,source_values,right_metadata,right_values)}

def normal_counted(+shape: List<&2,U32>,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),shape,fill,Tensors.Usable{},workers})) -> Global.Counted<shape>:
  Shape.Both{ready,Shape.Both{counted,length}} = normal
  counted

def packed_length(+source: Buffer.Model<F32>) -> {Buffer.content_length(~F32,Buffer.contents(~F32,Buffer.pack(~F32,source))) == Buffer.length(~F32,source) : Nat}:
  Buffer.Model{+tree,+length,+certificate} = source
  Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.contents(~F32,Buffer.pack(~F32,source)),Buffer.content(~F32,source),Buffer.packed_contents(~F32,source))

def normal_length(+shape: List<&2,U32>,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),shape,fill,Tensors.Usable{},workers})) ->
  {Buffer.length(~F32,source) == ShapeLaws.elements(shape) : Nat}:
  Shape.Both{ready,Shape.Both{counted,length_exact}} = normal
  Equal.trans(Nat,Buffer.length(~F32,source),Buffer.content_length(~F32,Buffer.contents(~F32,Buffer.pack(~F32,source))),ShapeLaws.elements(shape),
    Equal.sym(Nat,Buffer.content_length(~F32,Buffer.contents(~F32,Buffer.pack(~F32,source))),Buffer.length(~F32,source),packed_length(source)),length_exact)

def normal_ready(+shape: List<&2,U32>,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),shape,fill,Tensors.Usable{},workers})) -> Buffer.Sized<F32,source>:
  Shape.Both{ready,shape_facts} = normal
  Buffer.ready_content(~F32,source,
    %Buffer.packed_contents(~F32,source) : Buffer.ContentReady<F32,_>
    ready)

def source_coherent(+input: PublicFrame.NativeInput,+matched: {Native.input_matched(input) == True{} : Bool}) ->
  {Window.band_shape(Native.input_extents(input),PublicFrame.source_geometry(input)) == True{} : Bool}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  BandOutputNormal.coherent(batched,Bands.Geometry{images,ci,h,w},
    %Equal.sym(Bool,BandOutputNormal.shape_matched(images,batched),Native.input_matched(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p}),
      NormalCaller.native_output_match(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})) : {_ == True{} : Bool}
    matched)

def geometry_exact(+input: PublicFrame.NativeInput) ->
  {Leaf.input_geometry(PublicFrame.input_images(input),PublicFrame.frame_dimensions(input)) == PublicFrame.source_geometry(input) : Bands.Geometry}:
  PublicFrame.NativeInput{+images,batched,+ci,+co,+h,+w,+k,+s,+p} = input
  BandNativeSourceDomain.geometry_roundtrip(images,ci,co,h,w,k,s,p)

def whole_domain(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: PublicFrame.NativeInput,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  +source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},workers}),
  matched: {Native.input_matched(input) == True{} : Bool},validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool}) ->
  BandGather.Domain<PublicFrame.source_geometry(input),Default.limits(),Machine.capacity(Default.limits())>:
  BandNativeSourceDomain.converted(Leaf.input_geometry(PublicFrame.input_images(input),PublicFrame.frame_dimensions(input)),PublicFrame.source_geometry(input),Default.limits(),Machine.capacity(Default.limits()),geometry_exact(input),
    NormalCaller.source_domain(~Logical,~original,logical,input,weight_list,bias_list,Shape.Both{normal_counted(Native.input_extents(input),source,fill,workers,normal),source_coherent(input,matched)},validated))

def logical_domain(+geometry: Bands.Geometry,+limits: Machine.Limits,+count: U32,+previous_capacity: U32,
  whole: BandGather.Domain<geometry,limits,previous_capacity>,+exact: {BandGather.elements(geometry) == U32.to_nat(count) : Nat}) -> BandGather.Domain<geometry,limits,count>:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,+machine_room,allocation_room} = whole
  BandGather.Domain{Arithmetic.count_within(count,BandGather.elements(geometry),Machine.capacity(limits),2147483648,
      Equal.sym(Nat,BandGather.elements(geometry),U32.to_nat(count),exact),machine_room,Limits.capacity_fits(limits)),
    channels_positive,height_positive,width_positive,machine_room,
    %Equal.sym(Nat,BandGather.elements(geometry),U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(count)) == True{} : Bool}
    Order.reflexive(U32.to_nat(count))}

def expected_length(+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},workers}),
  matched: {Native.input_matched(input) == True{} : Bool}) -> {BandGather.elements(PublicFrame.source_geometry(input)) == Buffer.length(~F32,source) : Nat}:
  Equal.trans(Nat,BandGather.elements(PublicFrame.source_geometry(input)),ShapeLaws.elements(Native.input_extents(input)),Buffer.length(~F32,source),
    Equal.sym(Nat,ShapeLaws.elements(Native.input_extents(input)),BandGather.elements(PublicFrame.source_geometry(input)),BandCount.shape_coherence_elements(Native.input_extents(input),PublicFrame.source_geometry(input),source_coherent(input,matched))),
    Equal.sym(Nat,Buffer.length(~F32,source),ShapeLaws.elements(Native.input_extents(input)),normal_length(Native.input_extents(input),source,fill,workers,normal)))

def owner_domain(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: PublicFrame.NativeInput,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  +source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool}) ->
  BandGather.Domain<PublicFrame.source_geometry(input),Default.limits(),Tree.capacity(F32,Buffer.tree(~F32,source))>:
  BandDense.owned(~F32,PublicFrame.source_geometry(input),Default.limits(),Machine.capacity(Default.limits()),source,
    expected_length(input,source,fill,workers,normal,matched),normal_ready(Native.input_extents(input),source,fill,workers,normal),whole_domain(~Logical,~original,logical,input,weight_list,bias_list,source,fill,workers,normal,matched,validated))

def split_normal(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: PublicFrame.NativeInput,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+depth: Nat,
  +source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool}) ->
  Global.Normal(~F32,Tensors.Banded{Bands.split(~F32,depth,PublicFrame.source_geometry(input),0.0,Buffer.pack(~F32,source)),PublicFrame.source_geometry(input),Native.input_extents(input),fill,Tensors.Usable{},workers}):
  Shape.Both{BandDense.split_well_formed(~F32,PublicFrame.source_geometry(input),Default.limits(),depth,0.0,source,owner_domain(~Logical,~original,logical,input,weight_list,bias_list,source,fill,workers,normal,matched,validated)),
    Shape.Both{normal_counted(Native.input_extents(input),source,fill,workers,normal),source_coherent(input,matched)}}

def count_exact(+expected: Nat,+count: U32,evidence: Count.Count<expected,count>) -> {expected == U32.to_nat(count) : Nat}:
  Count.Count{exact,bounded} = evidence
  exact

def count_domain(+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},whole: BandGather.Domain<PublicFrame.source_geometry(input),Default.limits(),Machine.capacity(Default.limits())>) ->
  BandGather.Domain<PublicFrame.source_geometry(input),Default.limits(),Global.count(Native.input_extents(input),normal_counted(Native.input_extents(input),source,fill,workers,normal))>:
  +counted = normal_counted(Native.input_extents(input),source,fill,workers,normal)
  +count = Global.count(Native.input_extents(input),counted)
  logical_domain(PublicFrame.source_geometry(input),Default.limits(),count,Machine.capacity(Default.limits()),whole,
    count_exact(BandGather.elements(PublicFrame.source_geometry(input)),count,Global.geometry_evidence(Native.input_extents(input),PublicFrame.source_geometry(input),counted,source_coherent(input,matched))))

def source_count_length(+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},workers})) ->
  {U32.to_nat(Global.count(Native.input_extents(input),normal_counted(Native.input_extents(input),source,fill,workers,normal))) == Buffer.length(~F32,source) : Nat}:
  +counted = normal_counted(Native.input_extents(input),source,fill,workers,normal)
  +count = Global.count(Native.input_extents(input),counted)
  Equal.trans(Nat,U32.to_nat(count),ShapeLaws.elements(Native.input_extents(input)),Buffer.length(~F32,source),
    Equal.sym(Nat,ShapeLaws.elements(Native.input_extents(input)),U32.to_nat(count),count_exact(ShapeLaws.elements(Native.input_extents(input)),count,Global.evidence(Native.input_extents(input),counted))),
    Equal.sym(Nat,Buffer.length(~F32,source),ShapeLaws.elements(Native.input_extents(input)),normal_length(Native.input_extents(input),source,fill,workers,normal)))

def contents_equal(~Element: Data,+model: Model.Model<Element>,-bands: Bands.Bands(~Element),equal: {Model.pack(~Element,model) == bands : Bands.Bands(~Element)}) ->
  {Model.content(~Element,model) == Model.contents(~Element,bands) : Model.Content<Element>}:
  Equal.trans(Model.Content<Element>,Model.content(~Element,model),Model.contents(~Element,Model.pack(~Element,model)),Model.contents(~Element,bands),
    Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,model)),Model.content(~Element,model),Model.packed_contents(~Element,model)),
    Equal.cong(Bands.Bands(~Element),Model.Content<Element>,Model.contents(~Element),Model.pack(~Element,model),bands,equal))

def modeled_well_formed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+count: U32,+source: Buffer.Model<Element>,
  +target: Model.Model<Element>,domain: BandGather.Domain<geometry,limits,count>,length_exact: {U32.to_nat(count) == Buffer.length(~Element,source) : Nat},
  source_values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,source)),
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,target)>,
  target_values: BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),Buffer.tree(~Element,source),geometry,Model.content(~Element,target))) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,target)):
  BandShape.WellFormed{metadata,ready,coverage} = well_formed
  modeled(~Element,~Logical,~original,logical,geometry,limits,count,source,target,domain,length_exact,source_values,metadata,target_values)

def observed_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+count: U32,+source: Buffer.Model<Element>,
  -bands: Bands.Bands(~Element),observation: Model.Observation<Element,bands>,domain: BandGather.Domain<geometry,limits,count>,length_exact: {U32.to_nat(count) == Buffer.length(~Element,source) : Nat},
  source_values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,source)),
  well_formed: BandShape.WellFormed<Element,geometry,Model.contents(~Element,bands)>,
  target_values: BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),Buffer.tree(~Element,source),geometry,Model.contents(~Element,bands))) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.contents(~Element,bands)):
  Model.Observation{+model,equal} = observation
  +content_equal = contents_equal(~Element,model,bands,equal)
  %content_equal : BandValues.Values(~Element,~Logical,~original,logical,geometry,_)
  modeled_well_formed(~Element,~Logical,~original,logical,geometry,limits,count,source,model,domain,length_exact,source_values,
    %Equal.sym(Model.Content<Element>,Model.content(~Element,model),Model.contents(~Element,bands),content_equal) : BandShape.WellFormed<Element,geometry,_>
    well_formed,
    %Equal.sym(Model.Content<Element>,Model.content(~Element,model),Model.contents(~Element,bands),content_equal) : BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.dense_value(~Element),Buffer.tree(~Element,source),geometry,_)
    target_values)

def split_values(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: PublicFrame.NativeInput,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+depth: Nat,
  +source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool},
  source_values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source))) ->
  BandValues.Values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),
    Model.contents(~F32,Bands.split(~F32,depth,PublicFrame.source_geometry(input),0.0,Buffer.pack(~F32,source)))):
  +whole = whole_domain(~Logical,~original,logical,input,weight_list,bias_list,source,fill,workers,normal,matched,validated)
  +owned = owner_domain(~Logical,~original,logical,input,weight_list,bias_list,source,fill,workers,normal,matched,validated)
  +count = Global.count(Native.input_extents(input),normal_counted(Native.input_extents(input),source,fill,workers,normal))
  observed_values(~F32,~Logical,~original,logical,PublicFrame.source_geometry(input),Default.limits(),count,source,
    Bands.split(~F32,depth,PublicFrame.source_geometry(input),0.0,Buffer.pack(~F32,source)),
    Model.observe(~F32,Bands.split(~F32,depth,PublicFrame.source_geometry(input),0.0,Buffer.pack(~F32,source))),
    count_domain(input,source,fill,workers,normal,matched,whole),source_count_length(input,source,fill,workers,normal),source_values,
    BandDense.split_well_formed(~F32,PublicFrame.source_geometry(input),Default.limits(),depth,0.0,source,owned),BandDense.bounds_split(~F32,PublicFrame.source_geometry(input),Default.limits(),depth,0.0,source,owned))

def frame_owner_domain(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+source: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),shape,fill,Tensors.Usable{},workers}),
  coherent: {Window.band_shape(shape,PipelineExecution.input_geometry(~Logical,frame)) == True{} : Bool}) ->
  BandGather.Domain<PipelineExecution.input_geometry(~Logical,frame),PipelineExecution.convolution_limits(~Logical,frame),Tree.capacity(F32,Buffer.tree(~F32,source))>:
  BandDense.owned(~F32,PipelineExecution.input_geometry(~Logical,frame),PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame),source,
    Equal.trans(Nat,BandGather.elements(PipelineExecution.input_geometry(~Logical,frame)),ShapeLaws.elements(shape),Buffer.length(~F32,source),
      Equal.sym(Nat,ShapeLaws.elements(shape),BandGather.elements(PipelineExecution.input_geometry(~Logical,frame)),BandCount.shape_coherence_elements(shape,PipelineExecution.input_geometry(~Logical,frame),coherent)),
      Equal.sym(Nat,Buffer.length(~F32,source),ShapeLaws.elements(shape),normal_length(shape,source,fill,workers,normal))),
    normal_ready(shape,source,fill,workers,normal),PipelineExecution.convolution_domain(~Logical,frame))

def frame_split_values(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+depth: Nat,+source: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),shape,fill,Tensors.Usable{},workers}),
  +coherent: {Window.band_shape(shape,PipelineExecution.input_geometry(~Logical,frame)) == True{} : Bool},
  source_values: Dense.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),Buffer.content(~F32,source))) ->
  BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),
    Model.contents(~F32,Bands.split(~F32,depth,PipelineExecution.input_geometry(~Logical,frame),0.0,Buffer.pack(~F32,source)))):
  +geometry = PipelineExecution.input_geometry(~Logical,frame)
  +limits = PipelineExecution.convolution_limits(~Logical,frame)
  +counted = normal_counted(shape,source,fill,workers,normal)
  +count = Global.count(shape,counted)
  +owned = frame_owner_domain(~Logical,frame,source,shape,fill,workers,normal,coherent)
  observed_values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),geometry,limits,count,source,
    Bands.split(~F32,depth,geometry,0.0,Buffer.pack(~F32,source)),Model.observe(~F32,Bands.split(~F32,depth,geometry,0.0,Buffer.pack(~F32,source))),
    logical_domain(geometry,limits,count,PipelineExecution.convolution_capacity(~Logical,frame),PipelineExecution.convolution_domain(~Logical,frame),
      count_exact(BandGather.elements(geometry),count,Global.geometry_evidence(shape,geometry,counted,coherent))),
    Equal.trans(Nat,U32.to_nat(count),ShapeLaws.elements(shape),Buffer.length(~F32,source),
      Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),count_exact(ShapeLaws.elements(shape),count,Global.evidence(shape,counted))),
      Equal.sym(Nat,Buffer.length(~F32,source),ShapeLaws.elements(shape),normal_length(shape,source,fill,workers,normal))),
    source_values,BandDense.split_well_formed(~F32,geometry,limits,depth,0.0,source,owned),BandDense.bounds_split(~F32,geometry,limits,depth,0.0,source,owned))

def start_bands_matches(~custom: F32 -> F32,~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+depth: U32,+workers: U32,
  -source: Storage.Buffer<F32>,-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>) ->
  {TensorConvolution.start_bands(~custom,TensorExecution.execution_activation(~Logical,frame),depth,TensorExecution.execution_parallel(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),source,TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),weights),bias,workers) ==
    TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),TensorExecution.execution_parallel(~Logical,frame),
      Bands.split(~F32,U32.to_nat(depth),PipelineExecution.input_geometry(~Logical,frame),0.0,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),weights),bias,workers) : Tensors.Tensor<F32>}:
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weight_list,bias_list,we,be,domain,valid,supported,room,eligible} = frame
  KernelShape.Dimensions{ci,co,h,w,k,s,p} = dimensions
  {==}

def InitialContract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,-tensor: Tensors.Tensor<F32>) -> Data:
  Shape.Both<TensorExecution.OwnersContract(~custom,~Logical,~original,frame,tensor),Global.Normal(~F32,tensor)>

def owner_shape(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,-tensor: Tensors.Tensor<F32>,
  contract: TensorExecution.OwnersContract(~custom,~Logical,~original,frame,tensor)) -> BandMap.WellFormed(~F32,tensor):
  Shape.Both{values,well_formed} = contract
  well_formed

def start_bands(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+depth: U32,+workers: U32,
  +source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),shape,fill,Tensors.Usable{},workers}),
  +coherent: {Window.band_shape(shape,PipelineExecution.input_geometry(~Logical,frame)) == True{} : Bool},
  +matched: {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  source_values: Dense.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),Buffer.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  InitialContract(~custom,~Logical,~original,frame,
    TensorConvolution.start_bands(~custom,TensorExecution.execution_activation(~Logical,frame),depth,TensorExecution.execution_parallel(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)):
  +geometry = PipelineExecution.input_geometry(~Logical,frame)
  +owned = frame_owner_domain(~Logical,frame,source,shape,fill,workers,normal,coherent)
  +bands_shape = BandDense.split_well_formed(~F32,geometry,PipelineExecution.convolution_limits(~Logical,frame),U32.to_nat(depth),0.0,source,owned)
  +bands_values = frame_split_values(~Logical,~original,frame,U32.to_nat(depth),source,shape,fill,workers,normal,coherent,source_values)
  +contract = TensorExecution.owners(~custom,~Logical,~original,frame,batched,workers,Bands.split(~F32,U32.to_nat(depth),geometry,0.0,Buffer.pack(~F32,source)),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),
    bands_shape,bands_values,
    Equal.trans(Tree.Tree<F32>,Buffer.content_tree(~F32,Buffer.contents(~F32,Buffer.pack(~F32,weights))),Buffer.tree(~F32,weights),PipelineExecution.source_tree(~Logical,frame),TensorExecution.packed_tree(weights),weight_exact),
    Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,bias)),Buffer.content(~F32,bias),PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Buffer.packed_contents(~F32,bias),bias_exact))
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.start_bands(~custom,TensorExecution.execution_activation(~Logical,frame),depth,TensorExecution.execution_parallel(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,source),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),TensorExecution.execution_parallel(~Logical,frame),Bands.split(~F32,U32.to_nat(depth),geometry,0.0,Buffer.pack(~F32,source)),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    start_bands_matches(~custom,~Logical,frame,batched,depth,workers,Buffer.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) : InitialContract(~custom,~Logical,~original,frame,_)
  Shape.Both{contract,
    BandOutputNormal.normal_from_header(batched,OutputMetadata.bounds_geometry(~Logical,frame),
      TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),TensorExecution.execution_parallel(~Logical,frame),Bands.split(~F32,U32.to_nat(depth),geometry,0.0,Buffer.pack(~F32,source)),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
      BandOutputNormal.band_output_header(~custom,~Logical,frame,batched,TensorExecution.execution_parallel(~Logical,frame),workers,Bands.split(~F32,U32.to_nat(depth),geometry,0.0,Buffer.pack(~F32,source)),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),
      BandOutputNormal.completed_counted(BandOutputNormal.extents(batched,OutputMetadata.bounds_geometry(~Logical,frame)),Bands.elements(OutputMetadata.bounds_geometry(~Logical,frame)),BandOutputNormal.counted(~Logical,~original,frame,batched,matched,limits_exact)),
      matched,owner_shape(~custom,~Logical,~original,frame,
        TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),TensorExecution.execution_parallel(~Logical,frame),Bands.split(~F32,U32.to_nat(depth),geometry,0.0,Buffer.pack(~F32,source)),TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),contract))}

def configured_parallel(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+domain: BandGather.Domain<Leaf.input_geometry(PublicFrame.input_images(input),PublicFrame.frame_dimensions(input)),Default.limits(),Machine.capacity(Default.limits())>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool}) ->
  {TensorExecution.execution_parallel(~Logical,PublicFrame.configured_native(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported,eligible)) == parallel : Bool}:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  {==}

def start_bands_native_fields(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,+depth: U32,+workers: U32,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +frame: PipelineExecution.Frame<Logical>,+source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},workers}),
  +fields: PublicFrame.Fields<Logical,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits())>,
  parallel_exact: {TensorExecution.execution_parallel(~Logical,frame) == parallel : Bool},+matched: {Native.input_matched(input) == True{} : Bool},
  source_values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  InitialContract(~custom,~Logical,~original,frame,
    TensorConvolution.start_bands(~custom,activation,depth,parallel,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)):
  PublicFrame.Fields{+logical_exact,activation_exact,images_exact,dimensions_exact,+input_geometry_exact,weights_exact,bias_content_exact,context_exact,limits_exact,capacity_exact} = fields
  +geometry_matches = Native.geometry_fields(~Logical,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits()),fields)
  +bias_settings = PublicFrame.settings_bias(~Logical,~original,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits()),fields)
  +framed_coherent = {
    %Equal.sym(Bands.Geometry,PipelineExecution.input_geometry(~Logical,frame),PublicFrame.source_geometry(input),input_geometry_exact) : {Window.band_shape(Native.input_extents(input),_) == True{} : Bool}
    source_coherent(input,matched) : {Window.band_shape(Native.input_extents(input),PipelineExecution.input_geometry(~Logical,frame)) == True{} : Bool}}
  +framed_values = {
    %Equal.sym(Logical,PipelineExecution.convolution_logical(~Logical,frame),logical,logical_exact) : Dense.Values(~F32,~Logical,~original,_,Buffer.content(~F32,source))
    source_values : Dense.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),Buffer.content(~F32,source))}
  +output_matched = NormalCaller.output_matched(~Logical,frame,input,images_exact,matched)
  %activation_exact : InitialContract(~custom,~Logical,~original,frame,TensorConvolution.start_bands(~custom,_,depth,parallel,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers))
  %parallel_exact : InitialContract(~custom,~Logical,~original,frame,TensorConvolution.start_bands(~custom,TensorExecution.execution_activation(~Logical,frame),depth,_,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers))
  %geometry_matches : InitialContract(~custom,~Logical,~original,frame,TensorConvolution.start_bands(~custom,TensorExecution.execution_activation(~Logical,frame),depth,TensorExecution.execution_parallel(~Logical,frame),_,Buffer.pack(~F32,source),TensorConvolution.packed(_,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers))
  start_bands(~custom,~Logical,~original,frame,Native.batched(input),depth,workers,source,weights,bias,Native.input_extents(input),fill,normal,framed_coherent,output_matched,limits_exact,framed_values,
    Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,weights),StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence),PipelineExecution.source_tree(~Logical,frame),weight_exact,
      Equal.sym(Tree.Tree<F32>,PipelineExecution.source_tree(~Logical,frame),StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence),weights_exact)),
    Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,bias),Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)},PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),bias_exact,
      Equal.sym(Buffer.Content<F32>,PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)},bias_settings)))

def start_bands_native(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,+depth: U32,+workers: U32,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +source: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},+validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  source_values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  InitialContract(~custom,~Logical,~original,
    PublicFrame.configured_native(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weight_list,bias_list,
      NormalCaller.source_domain(~Logical,~original,logical,input,weight_list,bias_list,Shape.Both{normal_counted(Native.input_extents(input),source,fill,workers,normal),source_coherent(input,matched)},validated),weight_evidence,bias_evidence,validated,supported,eligible),
    TensorConvolution.start_bands(~custom,activation,depth,parallel,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)):
  +domain = NormalCaller.source_domain(~Logical,~original,logical,input,weight_list,bias_list,Shape.Both{normal_counted(Native.input_extents(input),source,fill,workers,normal),source_coherent(input,matched)},validated)
  start_bands_native_fields(~custom,~Logical,~original,logical,activation,input,parallel,depth,workers,weight_list,bias_list,weight_evidence,bias_evidence,
    PublicFrame.configured_native(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported,eligible),source,weights,bias,fill,normal,
    PublicFrame.configured(~Logical,~original,logical,activation,input,parallel,Default.limits(),Machine.capacity(Default.limits()),weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported,eligible),
    configured_parallel(~Logical,~original,logical,activation,input,parallel,weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported,eligible),matched,source_values,weight_exact,bias_exact)

# The real CHW/NCHW decoder and Dense Normal produce whole source bounds.
# Capacity stays a formal native word until the final caller transports its
# actual Default equality, so no billion-element Nat is compared here.
def elements(+input: PublicFrame.NativeInput,matched: {Native.input_matched(input) == True{} : Bool}) ->
  {ShapeLaws.elements(Native.input_extents(input)) == BandGather.elements(PublicFrame.source_geometry(input)) : Nat}:
  BandCount.shape_coherence_elements(Native.input_extents(input),PublicFrame.source_geometry(input),source_coherent(input,matched))

def count_room(+shape: List<&2,U32>,+count: U32,+capacity: U32,+actual: {capacity == Machine.capacity(Default.limits()) : U32},facts: Count.Count<ShapeLaws.elements(shape),count>) ->
  {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(capacity)) == True{} : Bool}:
  Count.Count{exact,bounded} = facts
  %Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
  Bounds.as_nat_le(count,capacity,
    %Equal.sym(U32,capacity,Machine.capacity(Default.limits()),actual) : {U32.is_le(count,_) == True{} : Bool}
    bounded)

def counted_room(+shape: List<&2,U32>,+capacity: U32,+actual: {capacity == Machine.capacity(Default.limits()) : U32},counted: Global.Counted<shape>) ->
  {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(capacity)) == True{} : Bool}:
  Global.Counted{+count,+accepted} = counted
  count_room(shape,count,capacity,actual,Count.count_of(shape,count,accepted))

def length(+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,
  normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},workers}),matched: {Native.input_matched(input) == True{} : Bool}) ->
  {Buffer.length(~F32,source) == BandGather.elements(PublicFrame.source_geometry(input)) : Nat}:
  Equal.trans(Nat,Buffer.length(~F32,source),ShapeLaws.elements(Native.input_extents(input)),BandGather.elements(PublicFrame.source_geometry(input)),
    normal_length(Native.input_extents(input),source,fill,workers,normal),elements(input,matched))

def room(+input: PublicFrame.NativeInput,+source: Buffer.Model<F32>,+fill: F32,+workers: U32,+capacity: U32,+actual: {capacity == Machine.capacity(Default.limits()) : U32},
  normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},workers}),matched: {Native.input_matched(input) == True{} : Bool}) ->
  {Nat.is_le(BandGather.elements(PublicFrame.source_geometry(input)),U32.to_nat(capacity)) == True{} : Bool}:
  %elements(input,matched) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
  counted_room(Native.input_extents(input),capacity,actual,normal_counted(Native.input_extents(input),source,fill,workers,normal))

# Observe the actual dense_arguments producer for a public NCHW batch.
# Generic metadata consumption keeps the selected default depth opaque.
type Input<-Logical: Data> is Data:
  Input{context: BandConvolutionValues.Context<Logical>,images: U32,source: Buffer.Model<F32>,weights: Buffer.Model<F32>,bias: Buffer.Model<F32>,workers: U32}

def batch_output_context(~Logical: Data,input: Input<Logical>) -> BandConvolutionValues.Context<Logical>:
  Input{context,images,source,weights,bias,workers} = input
  context

def image_count(~Logical: Data,input: Input<Logical>) -> U32:
  Input{context,images,source,weights,bias,workers} = input
  images

def arguments(~Logical: Data,~original: Logical -> U32 -> F32,input: Input<Logical>) -> TensorConvolution.DenseArguments:
  Input{+context,images,source,weights,bias,workers} = input
  TensorConvolution.dense_arguments(TensorConvolution.Geometry{images,True{},DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,context,0n))},Buffer.depth(~F32,bias),workers,DenseBatchExecution.image_activation(~Logical,context))

# The weights in the order the kernel reads them, for the work of the arguments.
def packed_for(arguments: TensorConvolution.DenseArguments,weights: Buffer.Model<F32>) -> Storage.Buffer<Tile.Tile8>:
  TensorConvolution.DenseArguments{count,+work,input_size,output_size,extents,selected} = arguments
  GemmBuffer.reorder(DenseSources.rows(work),DenseSources.inner(work),Buffer.pack(~F32,weights))

def execute(~custom: F32 -> F32,~Logical: Data,input: Input<Logical>,+arguments: TensorConvolution.DenseArguments) -> Tensors.Tensor<F32>:
  Input{context,images,source,weights,bias,workers} = input
  TensorConvolution.execute_dense_arguments(~custom,DenseBatchExecution.image_activation(~Logical,context),arguments,Buffer.pack(~F32,source),packed_for(arguments,weights),Buffer.pack(~F32,bias),workers)

def batch_result(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,input: Input<Logical>) -> Tensors.Tensor<F32>:
  Input{+context,+images,source,weights,bias,workers} = input
  TensorConvolution.execute_dense(~custom,DenseBatchExecution.image_activation(~Logical,context),TensorConvolution.Geometry{images,True{},DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,context,0n))},Buffer.pack(~F32,source),TensorConvolution.packed(TensorConvolution.Geometry{images,True{},DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,context,0n))},Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)

def packed_arguments(+images: U32,+dimensions: KernelShape.Dimensions,+depth: Nat,+workers: U32,+activation: Activations.Activation,+weights: Buffer.Model<F32>) ->
  {TensorConvolution.packed(TensorConvolution.Geometry{images,True{},dimensions},Buffer.pack(~F32,weights)) ==
    packed_for(TensorConvolution.dense_arguments(TensorConvolution.Geometry{images,True{},dimensions},depth,workers,activation),weights) : Storage.Buffer<Tile.Tile8>}:
  KernelShape.Dimensions{channels,outputs,height,width,kernel,stride,padding} = dimensions
  {==}

def header(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+input: Input<Logical>) ->
  {batch_result(~custom,~Logical,~original,input) == execute(~custom,~Logical,input,arguments(~Logical,~original,input)) : Tensors.Tensor<F32>}:
  Input{+context,+images,+source,+weights,Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}},+workers} = input
  %packed_arguments(images,DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,context,0n)),depth,workers,DenseBatchExecution.image_activation(~Logical,context),weights) :
    {TensorConvolution.execute_dense(~custom,DenseBatchExecution.image_activation(~Logical,context),TensorConvolution.Geometry{images,True{},DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,context,0n))},Buffer.pack(~F32,source),TensorConvolution.packed(TensorConvolution.Geometry{images,True{},DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,context,0n))},Buffer.pack(~F32,weights)),Buffer.pack(~F32,Buffer.Model{tree,length,Certificate.Certificate{depth,equation}}),workers) ==
      TensorConvolution.execute_dense_arguments(~custom,DenseBatchExecution.image_activation(~Logical,context),TensorConvolution.dense_arguments(TensorConvolution.Geometry{images,True{},DenseSingle.dimensions(DenseSources.image_problem(~Logical,~original,context,0n))},depth,workers,DenseBatchExecution.image_activation(~Logical,context)),Buffer.pack(~F32,source),_,Buffer.pack(~F32,Buffer.Model{tree,length,Certificate.Certificate{depth,equation}}),workers) : Tensors.Tensor<F32>}
  {==}

def is_valid(~Logical: Data,~original: Logical -> U32 -> F32,input: Input<Logical>) -> Bool:
  Input{context,images,source,weights,bias,workers} = input
  ConvSpec.valid(DenseSources.image_problem(~Logical,~original,context,0n))

def is_supported(~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,input: Input<Logical>) -> Bool:
  Input{context,images,source,weights,bias,workers} = input
  NativeConv.supports_packed_with_limits(limits,DenseSources.image_problem(~Logical,~original,context,0n))

def BatchSource(~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,input: Input<Logical>,tiles: Buffer.Model<Tile.Tile8>) -> Data:
  Input{context,images,source,weights,bias,workers} = input
  DenseBatchExecution.Sources<Logical,original,context,limits,images,source,tiles,weights,bias>

def TilesOwner(~Logical: Data,~original: Logical -> U32 -> F32,input: Input<Logical>,tiles: Buffer.Model<Tile.Tile8>) -> Data:
  Input{context,images,source,weights,bias,workers} = input
  {Buffer.pack(~Tile.Tile8,tiles) == GemmBuffer.reorder(MicroLeaf.rows(DenseSources.image_problem(~Logical,~original,context,0n)),Coordinates.native_inner(DenseSources.image_problem(~Logical,~original,context,0n)),Buffer.pack(~F32,weights)) : Storage.Buffer<Tile.Tile8>}

def Room(~Logical: Data,+limits: Machine.Limits,input: Input<Logical>) -> Data:
  Input{context,images,source,weights,bias,workers} = input
  {Nat.is_le(Nat.mul(U32.to_nat(images),DenseBatchExecution.reference_volume(~Logical,context)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}

type Metadata<-Logical: Data,-original: Logical -> U32 -> F32,-input: Input<Logical>,-actual: TensorConvolution.DenseArguments> is Data:
  Metadata{count_exact: {DenseSingle.count(actual) == image_count(~Logical,input) : U32},
    work_exact: {DenseSingle.arguments_work(actual) == Inference.work(DenseSources.image_problem(~Logical,~original,batch_output_context(~Logical,input),0n)) : Pipeline.Work},
    input_exact: {DenseSingle.arguments_input_size(actual) == U32.from_nat(DenseSources.image_size(~Logical,batch_output_context(~Logical,input))) : U32},
    output_exact: {DenseSingle.output_size(actual) == DenseBatchExecution.size(~Logical,~original,batch_output_context(~Logical,input)) : U32},
    extents_exact: {DenseSingle.extents(actual) == DenseBatchExecution.extents(~Logical,~original,batch_output_context(~Logical,input),image_count(~Logical,input)) : List<&2,U32>}}

def metadata(~Logical: Data,~original: Logical -> U32 -> F32,+input: Input<Logical>,+limits: Machine.Limits,
  +valid: {is_valid(~Logical,~original,input) == True{} : Bool},+supported: {is_supported(~Logical,~original,limits,input) == True{} : Bool}) -> Metadata<Logical,original,input,arguments(~Logical,~original,input)>:
  Input{BandConvolutionValues.Context{+logical,+activation,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+ws,+bs},+images,+source,+weights,+bias,+workers} = input
  +context = {BandConvolutionValues.Context{logical,activation,channels,outputs,height,width,kernel,stride,padding,ws,bs} : BandConvolutionValues.Context<Logical>}
  +problem = DenseSources.image_problem(~Logical,~original,context,0n)
  Metadata{{==},DenseSingle.work_matches(limits,problem,valid,supported),{==},DenseConvolution.matches(limits,problem,valid,supported),{==}}

type Observation<-Logical: Data,-original: Logical -> U32 -> F32,-input: Input<Logical>> is Data:
  Observation{actual: TensorConvolution.DenseArguments,exact: {actual == arguments(~Logical,~original,input) : TensorConvolution.DenseArguments}}

def observe(~Logical: Data,~original: Logical -> U32 -> F32,+input: Input<Logical>) -> Observation<Logical,original,input>:
  Observation{arguments(~Logical,~original,input),{==}}

def batch_configured(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+input: Input<Logical>,+limits: Machine.Limits,+actual: TensorConvolution.DenseArguments,+tiles: Buffer.Model<Tile.Tile8>,
  +facts: Metadata<Logical,original,input,actual>,+source: BatchSource(~Logical,~original,limits,input,tiles),+tiles_owner: TilesOwner(~Logical,~original,input,tiles),+room: Room(~Logical,limits,input),
  +batch_many: {Nat.is_lt(1n,U32.to_nat(image_count(~Logical,input))) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),batch_output_context(~Logical,input),execute(~custom,~Logical,input,actual)):
  Input{+context,+images,+source_model,+weights,+bias,+workers} = input
  TensorConvolution.DenseArguments{+count,+work,+input_size,+output_size,+extents,+selected} = actual
  Metadata{+count_exact,+work_exact,+input_exact,+output_exact,+extents_exact} = facts
  %Equal.sym(U32,count,images,count_exact) : Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
    TensorConvolution.execute_dense_at_depth(~custom,DenseBatchExecution.image_activation(~Logical,context),_,selected,work,input_size,output_size,extents,Buffer.pack(~F32,source_model),GemmBuffer.reorder(DenseSources.rows(work),DenseSources.inner(work),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers))
  %Equal.sym(Pipeline.Work,work,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),work_exact) : Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
    TensorConvolution.execute_dense_at_depth(~custom,DenseBatchExecution.image_activation(~Logical,context),images,selected,_,input_size,output_size,extents,Buffer.pack(~F32,source_model),GemmBuffer.reorder(DenseSources.rows(_),DenseSources.inner(_),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers))
  %Equal.sym(U32,input_size,U32.from_nat(DenseSources.image_size(~Logical,context)),input_exact) : Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
    TensorConvolution.execute_dense_at_depth(~custom,DenseBatchExecution.image_activation(~Logical,context),images,selected,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),_,output_size,extents,Buffer.pack(~F32,source_model),GemmBuffer.reorder(DenseSources.rows(Inference.work(DenseSources.image_problem(~Logical,~original,context,0n))),DenseSources.inner(Inference.work(DenseSources.image_problem(~Logical,~original,context,0n))),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers))
  %Equal.sym(U32,output_size,DenseBatchExecution.size(~Logical,~original,context),output_exact) : Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
    TensorConvolution.execute_dense_at_depth(~custom,DenseBatchExecution.image_activation(~Logical,context),images,selected,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),U32.from_nat(DenseSources.image_size(~Logical,context)),_,extents,Buffer.pack(~F32,source_model),GemmBuffer.reorder(DenseSources.rows(Inference.work(DenseSources.image_problem(~Logical,~original,context,0n))),DenseSources.inner(Inference.work(DenseSources.image_problem(~Logical,~original,context,0n))),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers))
  %Equal.sym(List<&2,U32>,extents,DenseBatchExecution.extents(~Logical,~original,context,images),extents_exact) : Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
    TensorConvolution.execute_dense_at_depth(~custom,DenseBatchExecution.image_activation(~Logical,context),images,selected,Inference.work(DenseSources.image_problem(~Logical,~original,context,0n)),U32.from_nat(DenseSources.image_size(~Logical,context)),DenseBatchExecution.size(~Logical,~original,context),_,Buffer.pack(~F32,source_model),GemmBuffer.reorder(DenseSources.rows(Inference.work(DenseSources.image_problem(~Logical,~original,context,0n))),DenseSources.inner(Inference.work(DenseSources.image_problem(~Logical,~original,context,0n))),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers))
  images_contract_many(~custom,~Logical,~original,context,limits,images,selected,source_model,tiles,weights,bias,workers,source,tiles_owner,room,batch_many)

def observed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+input: Input<Logical>,+limits: Machine.Limits,+tiles: Buffer.Model<Tile.Tile8>,
  observation: Observation<Logical,original,input>,+source: BatchSource(~Logical,~original,limits,input,tiles),+tiles_owner: TilesOwner(~Logical,~original,input,tiles),+room: Room(~Logical,limits,input),
  +valid: {is_valid(~Logical,~original,input) == True{} : Bool},+supported: {is_supported(~Logical,~original,limits,input) == True{} : Bool},
  +batch_many: {Nat.is_lt(1n,U32.to_nat(image_count(~Logical,input))) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),batch_output_context(~Logical,input),batch_result(~custom,~Logical,~original,input)):
  Observation{+actual,+exact} = observation
  +facts = {
    %Equal.sym(TensorConvolution.DenseArguments,actual,arguments(~Logical,~original,input),exact) : Metadata<Logical,original,input,_>
    metadata(~Logical,~original,input,limits,valid,supported) : Metadata<Logical,original,input,actual>}
  %Equal.sym(Tensors.Tensor<F32>,batch_result(~custom,~Logical,~original,input),execute(~custom,~Logical,input,arguments(~Logical,~original,input)),header(~custom,~Logical,~original,input)) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),batch_output_context(~Logical,input),_)
  %exact : Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),batch_output_context(~Logical,input),execute(~custom,~Logical,input,_))
  batch_configured(~custom,~Logical,~original,input,limits,actual,tiles,facts,source,tiles_owner,room,batch_many)

def batch_many(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+input: Input<Logical>,+limits: Machine.Limits,+tiles: Buffer.Model<Tile.Tile8>,
  +source: BatchSource(~Logical,~original,limits,input,tiles),+tiles_owner: TilesOwner(~Logical,~original,input,tiles),+room: Room(~Logical,limits,input),
  +valid: {is_valid(~Logical,~original,input) == True{} : Bool},+supported: {is_supported(~Logical,~original,limits,input) == True{} : Bool},
  +batch_many: {Nat.is_lt(1n,U32.to_nat(image_count(~Logical,input))) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),batch_output_context(~Logical,input),batch_result(~custom,~Logical,~original,input)):
  observed(~custom,~Logical,~original,input,limits,tiles,observe(~Logical,~original,input),source,tiles_owner,room,valid,supported,batch_many)

# Dense image zero uses the same public ordered reference as band convolution.
# Relabelling values changes only their logical context; it preserves the
# actual content and derives every native index bound from its logical count.
def reference_problem(~Logical: Data,~original: Logical -> U32 -> F32,context: BandConvolutionValues.Context<Logical>) -> ConvSpec.Problem:
  BandConvolutionValues.Context{+logical,reference_activation,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias} = context
  BandConvolutionValues.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,0n)

def reference_activation(~Logical: Data,context: BandConvolutionValues.Context<Logical>) -> Activations.Activation:
  BandConvolutionValues.Context{logical,reference_activation,channels,outputs,height,width,kernel,stride,padding,weights,bias} = context
  reference_activation

def original_matches(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),Nat.mul(Guards.output_channel_count(reference_problem(~Logical,~original,context)),GatherCopyCoordinates.spatial(reference_problem(~Logical,~original,context)))) == True{} : Bool}) ->
  {DenseConvolution.original(~custom,DenseConvolution.Logical{reference_problem(~Logical,~original,context),reference_activation(~Logical,context)},index) == BandConvolutionValues.value(~custom,~Logical,~original,context,index) : F32}:
  BandConvolutionValues.Context{+logical,+reference_activation,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias} = context
  +reference_problem = BandConvolutionValues.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,0n)
  +flat = U32.to_nat(index)
  +area = BandConvolutionValues.plane(height,width,kernel,stride,padding)
  +channel = Nat.div(flat,area)
  +position = Nat.mod(flat,area)
  +positive = Inference.spatial_positive(reference_problem)
  +channel_inside = RowMajor.cancel_multiply_lt(channel,outputs,area,
    Order.le_lt_trans(Nat.mul(channel,area),flat,Nat.mul(outputs,area),DivisionOrder.product_below(flat,area,positive),inside))
  +position_inside = DivisionOrder.remainder_small(flat,area,positive)
  %Equal.sym(Nat,flat,Nat.add(Nat.mul(channel,area),position),DivisionOrder.quotient_exact(flat,area,positive)) :
    {BandConvolutionValues.at_position(~custom,~Logical,~original,logical,reference_activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,0n,channel,position) ==
      BandConvolutionValues.value_at(~custom,~Logical,~original,logical,reference_activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,_) : F32}
  Equal.sym(F32,BandConvolutionValues.value_at(~custom,~Logical,~original,logical,reference_activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,Nat.add(Nat.mul(channel,area),position)),
    BandConvolutionValues.at_position(~custom,~Logical,~original,logical,reference_activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,0n,channel,position),
    BandConvolutionValues.decoded(~custom,~Logical,~original,logical,reference_activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,0n,channel,position,channel_inside,position_inside))

def native_inside(+limits: Machine.Limits,+reference_problem: ConvSpec.Problem,+index: Nat,
  +valid: {ConvSpec.valid(reference_problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,reference_problem) == True{} : Bool},
  +inside: {Nat.is_lt(index,U32.to_nat(PreparedOutputReadback.count(reference_problem))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(index)),Nat.mul(Guards.output_channel_count(reference_problem),GatherCopyCoordinates.spatial(reference_problem))) == True{} : Bool}:
  +count_exact = PreparedOutputReadback.count_exact(limits,reference_problem,valid,supported)
  +natural_inside = {
    %count_exact : {Nat.is_lt(index,_) == True{} : Bool}
    inside : {Nat.is_lt(index,Nat.mul(Guards.output_channel_count(reference_problem),GatherCopyCoordinates.spatial(reference_problem))) == True{} : Bool}}
  +word_exact = Bounds.roundtrip(index,Machine.capacity(limits),Limits.capacity_fits(limits),Bounds.strict_le(index,U32.to_nat(Machine.capacity(limits)),
    Bounds.nat_trans(index,Nat.mul(Guards.output_channel_count(reference_problem),GatherCopyCoordinates.spatial(reference_problem)),U32.to_nat(Machine.capacity(limits)),natural_inside,
      Padded.scalar_output_bound(limits,reference_problem,Padded.scalar(limits,reference_problem,Support.fast(limits,reference_problem,supported))))))
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,word_exact) :
    {Nat.is_lt(_,Nat.mul(Guards.output_channel_count(reference_problem),GatherCopyCoordinates.spatial(reference_problem))) == True{} : Bool}
  natural_inside

def range_values(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,+context: BandConvolutionValues.Context<Logical>,+content: Buffer.Content<F32>,+remaining: Nat,+start: Nat,
  +valid: {ConvSpec.valid(reference_problem(~Logical,~original,context)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,reference_problem(~Logical,~original,context)) == True{} : Bool},
  before: Members.All(Nat,{index => Dense.At(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{reference_problem(~Logical,~original,context),reference_activation(~Logical,context)},content,index) : Nat -> Data},Ranges.range(remaining,start)),
  +room: {Nat.is_le(Nat.add(start,remaining),U32.to_nat(PreparedOutputReadback.count(reference_problem(~Logical,~original,context)))) == True{} : Bool}) ->
  Members.All(Nat,{index => Dense.At(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,content,index) : Nat -> Data},Ranges.range(remaining,start)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{head,tail} = before
      +reference_problem = reference_problem(~Logical,~original,context)
      Shape.Both{Equal.trans(F32,Buffer.value(~F32,U32.from_nat(start),content),DenseConvolution.original(~custom,DenseConvolution.Logical{reference_problem,reference_activation(~Logical,context)},U32.from_nat(start)),
        BandConvolutionValues.value(~custom,~Logical,~original,context,U32.from_nat(start)),head,
        original_matches(~custom,~Logical,~original,context,U32.from_nat(start),native_inside(limits,reference_problem,start,valid,supported,Order.room_index(start,rest,U32.to_nat(PreparedOutputReadback.count(reference_problem)),room)))),
        range_values(~custom,~Logical,~original,limits,context,content,rest,1n+start,valid,supported,tail,Order.advance_room(start,rest,U32.to_nat(PreparedOutputReadback.count(reference_problem)),room))}

def reference_values(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,+context: BandConvolutionValues.Context<Logical>,+content: Buffer.Content<F32>,
  +valid: {ConvSpec.valid(reference_problem(~Logical,~original,context)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,reference_problem(~Logical,~original,context)) == True{} : Bool},
  before: Dense.Values(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{reference_problem(~Logical,~original,context),reference_activation(~Logical,context)},content),
  +length_exact: {Buffer.content_length(~F32,content) == U32.to_nat(PreparedOutputReadback.count(reference_problem(~Logical,~original,context))) : Nat}) ->
  Dense.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,content):
  range_values(~custom,~Logical,~original,limits,context,content,Buffer.content_length(~F32,content),0n,valid,supported,before,
    %length_exact : {Nat.is_le(Buffer.content_length(~F32,content),_) == True{} : Bool}
    Order.reflexive(Buffer.content_length(~F32,content)))

# The single-image dense caller returns the public activated convolution
# reference, with its actual shape, Ready status and allocation certificate.
def produced(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,+context: BandConvolutionValues.Context<Logical>,+batched: Bool,+depth: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32,
  before: Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{reference_problem(~Logical,~original,context),reference_activation(~Logical,context)},
    Tensors.Tensor{Pipeline.run(~custom,reference_activation(~Logical,context),depth,Inference.work(reference_problem(~Logical,~original,context)),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),DenseSingle.output_extents(reference_problem(~Logical,~original,context),batched),0.0,Tensors.Usable{},workers}),
  +valid: {ConvSpec.valid(reference_problem(~Logical,~original,context)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,reference_problem(~Logical,~original,context)) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
    Tensors.Tensor{Pipeline.run(~custom,reference_activation(~Logical,context),depth,Inference.work(reference_problem(~Logical,~original,context)),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),DenseSingle.output_extents(reference_problem(~Logical,~original,context),batched),0.0,Tensors.Usable{},workers}):
  Shape.Both{Shape.Both{Dense.WellFormed{+length_exact,+ready},+status},values} = before
  +problem = reference_problem(~Logical,~original,context)
  +activation = reference_activation(~Logical,context)
  +content = Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)))
  Shape.Both{Shape.Both{Dense.WellFormed{length_exact,ready},status},
    reference_values(~custom,~Logical,~original,limits,context,content,valid,supported,values,
      Equal.trans(Nat,Buffer.content_length(~F32,content),ShapeLaws.elements(DenseSingle.output_extents(problem,batched)),U32.to_nat(PreparedOutputReadback.count(problem)),length_exact,DenseSingle.shape_count(limits,problem,batched,valid,supported)))}

def run(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,+context: BandConvolutionValues.Context<Logical>,+batched: Bool,+depth: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32,
  +source: DenseSingle.SourceStorage<reference_problem(~Logical,~original,context),input,weights,bias>,
  +valid: {ConvSpec.valid(reference_problem(~Logical,~original,context)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,reference_problem(~Logical,~original,context)) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
    Tensors.Tensor{Pipeline.run(~custom,reference_activation(~Logical,context),depth,Inference.work(reference_problem(~Logical,~original,context)),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),DenseSingle.output_extents(reference_problem(~Logical,~original,context),batched),0.0,Tensors.Usable{},workers}):
  produced(~custom,~Logical,~original,limits,context,batched,depth,input,weights,bias,workers,
    DenseSingle.run_contract(~custom,reference_activation(~Logical,context),limits,reference_problem(~Logical,~original,context),batched,depth,input,weights,bias,workers,source,valid,supported),valid,supported)

type PublicInput<-Logical: Data> is Data:
  PublicInput{context: BandConvolutionValues.Context<Logical>,batched: Bool,input: Buffer.Model<F32>,weights: Buffer.Model<F32>,bias: Buffer.Model<F32>,workers: U32}

def native_input(~Logical: Data,~original: Logical -> U32 -> F32,input: PublicInput<Logical>) -> DenseSingle.ActualInput:
  PublicInput{+context,batched,source,weights,bias,workers} = input
  DenseSingle.ActualInput{reference_problem(~Logical,~original,context),reference_activation(~Logical,context),batched,source,weights,bias,workers}

def image_output_context(~Logical: Data,input: PublicInput<Logical>) -> BandConvolutionValues.Context<Logical>:
  PublicInput{context,batched,source,weights,bias,workers} = input
  context

def ImageSource(~Logical: Data,~original: Logical -> U32 -> F32,input: PublicInput<Logical>) -> Data:
  DenseSingle.Source(native_input(~Logical,~original,input))

def image_result(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,input: PublicInput<Logical>) -> Tensors.Tensor<F32>:
  DenseSingle.result(~custom,native_input(~Logical,~original,input))

def configured_parts(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,+context: BandConvolutionValues.Context<Logical>,+batched: Bool,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32,+selected: U32,
  +source: DenseSingle.SourceStorage<reference_problem(~Logical,~original,context),input,weights,bias>,
  +valid: {ConvSpec.valid(reference_problem(~Logical,~original,context)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,reference_problem(~Logical,~original,context)) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
    TensorConvolution.execute_dense_at_depth(~custom,reference_activation(~Logical,context),1,selected,DenseSingle.image_work(reference_problem(~Logical,~original,context)),DenseSingle.caller_input_size(reference_problem(~Logical,~original,context)),PreparedOutputReadback.count(reference_problem(~Logical,~original,context)),DenseSingle.output_extents(reference_problem(~Logical,~original,context),batched),Buffer.pack(~F32,input),DenseSingle.packed(reference_problem(~Logical,~original,context),weights),Buffer.pack(~F32,bias),workers)):
  +problem = reference_problem(~Logical,~original,context)
  +activation = reference_activation(~Logical,context)
  %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.execute_dense_at_depth(~custom,activation,1,selected,DenseSingle.image_work(problem),DenseSingle.caller_input_size(problem),PreparedOutputReadback.count(problem),DenseSingle.output_extents(problem,batched),Buffer.pack(~F32,input),DenseSingle.packed(problem,weights),Buffer.pack(~F32,bias),workers),
    Tensors.Tensor{Pipeline.run(~custom,activation,selected,DenseSingle.image_work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),DenseSingle.output_extents(problem,batched),0.0,Tensors.Usable{},workers},DenseSingle.executed(~custom,activation,problem,batched,selected,input,weights,bias,workers)) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,_)
  %Equal.sym(Pipeline.Work,DenseSingle.image_work(problem),Inference.work(problem),DenseSingle.work_matches(limits,problem,valid,supported)) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
      Tensors.Tensor{Pipeline.run(~custom,activation,selected,_,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),DenseSingle.output_extents(problem,batched),0.0,Tensors.Usable{},workers})
  run(~custom,~Logical,~original,limits,context,batched,selected,input,weights,bias,workers,source,valid,supported)

def selected_contract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+input: PublicInput<Logical>,+limits: Machine.Limits,+selected: U32,+source: ImageSource(~Logical,~original,input),
  +valid: {DenseSingle.is_valid(native_input(~Logical,~original,input)) == True{} : Bool},+supported: {DenseSingle.is_supported(limits,native_input(~Logical,~original,input)) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),image_output_context(~Logical,input),DenseSingle.at_depth(~custom,native_input(~Logical,~original,input),selected)):
  PublicInput{+context,+batched,+source_model,+weights,+bias,+workers} = input
  configured_parts(~custom,~Logical,~original,limits,context,batched,source_model,weights,bias,workers,selected,source,valid,supported)

def image_configured(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+input: PublicInput<Logical>,+limits: Machine.Limits,
  observation: DenseSingle.Observation<native_input(~Logical,~original,input)>,+source: ImageSource(~Logical,~original,input),
  +valid: {DenseSingle.is_valid(native_input(~Logical,~original,input)) == True{} : Bool},+supported: {DenseSingle.is_supported(limits,native_input(~Logical,~original,input)) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),image_output_context(~Logical,input),image_result(~custom,~Logical,~original,input)):
  DenseSingle.Observation{+actual,+exact} = observation
  +native = native_input(~Logical,~original,input)
  +facts = {
    %Equal.sym(TensorConvolution.DenseArguments,actual,DenseSingle.arguments(native),exact) : DenseSingle.Metadata<native,_>
    DenseSingle.metadata(native,limits,valid,supported) : DenseSingle.Metadata<native,actual>}
  %Equal.sym(Tensors.Tensor<F32>,image_result(~custom,~Logical,~original,input),
    DenseSingle.execute(~custom,native,DenseSingle.arguments(native)),DenseSingle.header(~custom,native)) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),image_output_context(~Logical,input),_)
  %exact : Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),image_output_context(~Logical,input),DenseSingle.execute(~custom,native,_))
  %Equal.sym(Tensors.Tensor<F32>,DenseSingle.execute(~custom,native,actual),DenseSingle.at_depth(~custom,native,DenseSingle.selected(actual)),DenseSingle.scheduled(~custom,native,actual,facts)) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),image_output_context(~Logical,input),_)
  selected_contract(~custom,~Logical,~original,input,limits,DenseSingle.selected(actual),source,valid,supported)

def image_single_image(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+input: PublicInput<Logical>,+limits: Machine.Limits,+source: ImageSource(~Logical,~original,input),
  +valid: {DenseSingle.is_valid(native_input(~Logical,~original,input)) == True{} : Bool},+supported: {DenseSingle.is_supported(limits,native_input(~Logical,~original,input)) == True{} : Bool}) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),image_output_context(~Logical,input),image_result(~custom,~Logical,~original,input)):
  image_configured(~custom,~Logical,~original,input,limits,DenseSingle.observe(native_input(~Logical,~original,input)),source,valid,supported)

# Actual planning is retained in the runtime-produced argument record.
# Generic leaf contracts consume its selected depth without unfolding cost.
# Every premise here comes from the true argument producer or actual guards.
def scheduled(~custom: F32 -> F32,+input: DenseSingle.ActualInput,+limits: Machine.Limits,+valid: {DenseSingle.is_valid(input) == True{} : Bool},+supported: {DenseSingle.is_supported(limits,input) == True{} : Bool}) ->
  {DenseSingle.result(~custom,input) == DenseSingle.at_depth(~custom,input,DenseSingle.selected(DenseSingle.arguments(input))) : Tensors.Tensor<F32>}:
  Equal.trans(Tensors.Tensor<F32>,DenseSingle.result(~custom,input),DenseSingle.execute(~custom,input,DenseSingle.arguments(input)),DenseSingle.at_depth(~custom,input,DenseSingle.selected(DenseSingle.arguments(input))),
    DenseSingle.header(~custom,input),DenseSingle.scheduled(~custom,input,DenseSingle.arguments(input),DenseSingle.metadata(input,limits,valid,supported)))

def planner_schedule_configured(~custom: F32 -> F32,+input: DenseSingle.ActualInput,+limits: Machine.Limits,observation: DenseSingle.Observation<input>,+source: DenseSingle.Source(input),
  +valid: {DenseSingle.is_valid(input) == True{} : Bool},+supported: {DenseSingle.is_supported(limits,input) == True{} : Bool}) ->
  Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseSingle.logical(input),DenseSingle.result(~custom,input)):
  DenseSingle.Observation{+actual,+exact} = observation
  +facts = {
    %Equal.sym(TensorConvolution.DenseArguments,actual,DenseSingle.arguments(input),exact) : DenseSingle.Metadata<input,_>
    DenseSingle.metadata(input,limits,valid,supported) : DenseSingle.Metadata<input,actual>}
  %Equal.sym(Tensors.Tensor<F32>,DenseSingle.result(~custom,input),DenseSingle.execute(~custom,input,DenseSingle.arguments(input)),DenseSingle.header(~custom,input)) :
    Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseSingle.logical(input),_)
  %exact : Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseSingle.logical(input),DenseSingle.execute(~custom,input,_))
  DenseSingle.configured(~custom,input,limits,actual,facts,source,valid,supported)

def planner_schedule_single_image(~custom: F32 -> F32,+input: DenseSingle.ActualInput,+limits: Machine.Limits,+source: DenseSingle.Source(input),
  +valid: {DenseSingle.is_valid(input) == True{} : Bool},+supported: {DenseSingle.is_supported(limits,input) == True{} : Bool}) ->
  Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseSingle.logical(input),DenseSingle.result(~custom,input)):
  planner_schedule_configured(~custom,input,limits,DenseSingle.observe(input),source,valid,supported)
