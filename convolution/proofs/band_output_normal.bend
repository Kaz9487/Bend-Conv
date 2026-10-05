# The actual band output retains every public axis and the checked batch room.
# Count completion is shared with the other tensor producers.
import Base
import ../specification.bend as ConvSpec
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_band_window.bend as BandWindow
import ../../lib/tensor_window.bend as Window
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_count.bend as BandCount
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_count_refinement.bend as Count
import ../../lib/proofs/tensor_count_completion.bend as CountCompletion
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/tensor_band_map.bend as BandMap
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/storage_buffer.bend as Storage
import ../../lib/math_activation.bend as Activations
import ../../lib/kernel_band_planning.bend as BandPlanning
import ../../lib/default_kernel_cost.bend as DefaultCost
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_output_metadata.bend as OutputMetadata
import ./band_leaf_values.bend as LeafValues
import ./band_caller_guards.bend as Caller
import ./band_tensor_execution.bend as TensorExecution
import ./band_tensor_replanning.bend as Replan
import ./band_machine_caller.bend as MachineCaller
import ./band_inspected_caller.bend as Inspected
import ./band_public_frame.bend as PublicFrame

def images(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  images

def extents(batched: Bool,geometry: Bands.Geometry) -> List<&2,U32>:
  Bands.Geometry{images,channels,height,width} = geometry
  TensorConvolution.output_shape(batched,images,channels,height,width)

def shape_matched(+images: U32,batched: Bool) -> Bool:
  match batched:
    case True{}: True{}
    case False{}: U32.is_eq(images,1)

def coherent(+batched: Bool,+geometry: Bands.Geometry,
  matched: {shape_matched(images(geometry),batched) == True{} : Bool}) ->
  {Window.band_shape(extents(batched,geometry),geometry) == True{} : Bool}:
  match batched geometry:
    case True{} Bands.Geometry{+images,+channels,+height,+width}:
      %Equal.sym(Bool,U32.is_eq(images,images),True{},BandShape.equal_words(images,images,{==})) : {_ && U32.is_eq(channels,channels) && U32.is_eq(height,height) && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(channels,channels),True{},BandShape.equal_words(channels,channels,{==})) : {True{} && _ && U32.is_eq(height,height) && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(height,height),True{},BandShape.equal_words(height,height,{==})) : {True{} && True{} && _ && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(width,width),True{},BandShape.equal_words(width,width,{==})) : {True{} && True{} && True{} && _ == True{} : Bool}
      {==}
    case False{} Bands.Geometry{+images,+channels,+height,+width}:
      %Equal.sym(Bool,U32.is_eq(images,1),True{},matched) : {_ && U32.is_eq(channels,channels) && U32.is_eq(height,height) && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(channels,channels),True{},BandShape.equal_words(channels,channels,{==})) : {True{} && _ && U32.is_eq(height,height) && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(height,height),True{},BandShape.equal_words(height,height,{==})) : {True{} && True{} && _ && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(width,width),True{},BandShape.equal_words(width,width,{==})) : {True{} && True{} && True{} && _ == True{} : Bool}
      {==}

def domain_from_words(+images: U32,+ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+limits: Machine.Limits,
  words: LeafValues.Words<ci,co,height,width,kernel,stride,padding>,
  co_positive: {Nat.is_lt(0n,co) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  BandGather.Domain<Bands.Geometry{images,U32.from_nat(co),U32.from_nat(ConvSpec.axis_size(height,kernel,stride,padding)),U32.from_nat(ConvSpec.axis_size(width,kernel,stride,padding))},limits,Machine.capacity(limits)>:
  LeafValues.Words{ci_word,+co_word,height_word,width_word,kernel_word,stride_word,padding_word,+oh_word,+ow_word} = words
  +oh = ConvSpec.axis_size(height,kernel,stride,padding)
  +ow = ConvSpec.axis_size(width,kernel,stride,padding)
  +native_room = {
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(co)),co,co_word) : {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(_,Nat.mul(U32.to_nat(U32.from_nat(oh)),U32.to_nat(U32.from_nat(ow))))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(oh)),oh,oh_word) : {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(_,U32.to_nat(U32.from_nat(ow))))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ow)),ow,ow_word) : {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(oh,_))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    room : {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(U32.from_nat(co)),Nat.mul(U32.to_nat(U32.from_nat(oh)),U32.to_nat(U32.from_nat(ow))))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  BandGather.Domain{Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(co)),co,co_word) : {Nat.is_lt(0n,_) == True{} : Bool} co_positive,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(oh)),oh,oh_word) : {Nat.is_lt(0n,_) == True{} : Bool} {==},
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ow)),ow,ow_word) : {Nat.is_lt(0n,_) == True{} : Bool} {==},
    native_room,native_room}

def domain(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>) ->
  BandGather.Domain<OutputMetadata.bounds_geometry(~Logical,frame),PipelineExecution.convolution_limits(~Logical,frame),Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame))>:
  PipelineExecution.Frame{+logical,activation,KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding},+images,parallel,+limits,capacity,+weights,+bias,we,be,input_domain,+valid,+supported,room,eligible} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} : KernelShape.Dimensions}
  +problem = Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n)
  +problem_valid = Caller.valid(~Logical,~original,logical,dimensions,weights,bias,0n,valid)
  domain_from_words(images,ci,co,height,width,kernel,stride,padding,limits,
    Caller.words(limits,problem,problem_valid,Caller.supported(~Logical,~original,logical,limits,dimensions,weights,bias,0n,supported)),
    Caller.outputs_positive(problem,problem_valid),room)

def elements(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,
  matched: {shape_matched(images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool}) ->
  {ShapeLaws.elements(extents(batched,OutputMetadata.bounds_geometry(~Logical,frame))) == BandGather.elements(OutputMetadata.bounds_geometry(~Logical,frame)) : Nat}:
  BandCount.shape_coherence_elements(extents(batched,OutputMetadata.bounds_geometry(~Logical,frame)),OutputMetadata.bounds_geometry(~Logical,frame),coherent(batched,OutputMetadata.bounds_geometry(~Logical,frame),matched))

def counted(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,
  matched: {shape_matched(images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits}) ->
  Count.Count<ShapeLaws.elements(extents(batched,OutputMetadata.bounds_geometry(~Logical,frame))),Bands.elements(OutputMetadata.bounds_geometry(~Logical,frame))>:
  +output_domain = domain(~Logical,~original,frame)
  Count.Count{
    Equal.trans(Nat,ShapeLaws.elements(extents(batched,OutputMetadata.bounds_geometry(~Logical,frame))),BandGather.elements(OutputMetadata.bounds_geometry(~Logical,frame)),U32.to_nat(Bands.elements(OutputMetadata.bounds_geometry(~Logical,frame))),
      elements(~Logical,frame,batched,matched),
      Equal.sym(Nat,U32.to_nat(Bands.elements(OutputMetadata.bounds_geometry(~Logical,frame))),BandGather.elements(OutputMetadata.bounds_geometry(~Logical,frame)),
        BandCount.exact(OutputMetadata.bounds_geometry(~Logical,frame),PipelineExecution.convolution_limits(~Logical,frame),Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame)),output_domain))),
    %limits_exact : {U32.is_le(Bands.elements(OutputMetadata.bounds_geometry(~Logical,frame)),Machine.capacity(_)) == True{} : Bool}
    BandCount.within(OutputMetadata.bounds_geometry(~Logical,frame),PipelineExecution.convolution_limits(~Logical,frame),Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame)),output_domain)}

# Header evidence is produced from the real constructors before attaching
# the existing owner WellFormed proof. It does not inspect convolution values.
def Header(+batched: Bool,+expected_geometry: Bands.Geometry,tensor: Tensors.Tensor<F32>) -> Data:
  match tensor:
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}:
      Shape.Both<{geometry == expected_geometry : Bands.Geometry},
        Shape.Both<{shape == extents(batched,expected_geometry) : List<&2,U32>},{status == Tensors.Ready{} : Tensors.Status}>>
    case Tensors.Tensor{storage,shape,fill,status,workers}: Empty

def band_output_header(~custom: F32 -> F32,~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+parallel: Bool,+workers: U32,
  -source: Bands.Bands(~F32),-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>) ->
  Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),
    TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),parallel,source,weights,bias,workers)):
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,old_parallel,limits,capacity,weight_list,bias_list,we,be,input_domain,valid,supported,room,eligible} = frame
  Shape.Both{{==},Shape.Both{{==},{==}}}

def replanned_header(~custom: F32 -> F32,~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  selection: BandPlanning.LayoutPlan,-source: Bands.Bands(~F32),-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>) ->
  Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),
    TensorConvolution.replanned(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),selection,source,weights,bias,workers)):
  BandPlanning.LayoutPlan{layout,+parallel,time,repartition} = selection
  band_output_header(~custom,~Logical,frame,batched,parallel,workers,
    Bands.align(~F32,BandWindow.input_geometry(TensorConvolution.context_window(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),parallel,Activations.Identity{}))),0.0,layout,source),weights,bias)

def existing_layout_header(~custom: F32 -> F32,~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>) ->
  Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),
    TensorConvolution.existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers,
      (Model.pack(~F32,source),Model.layout(~F32,source)))):
  %Replan.existing_layout_matches(~custom,~Logical,frame,batched,workers,source,weights,bias) : Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),_)
  replanned_header(~custom,~Logical,frame,batched,workers,
    BandPlanning.replan(Model.layout(~F32,source),TensorConvolution.band_work(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame)),U32.shln(1,Buffer.depth(~F32,bias))),
      TensorConvolution.context_window(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame))),workers,DefaultCost.configuration()),
    Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))

def planned_header(~custom: F32 -> F32,~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>) ->
  Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),
    TensorConvolution.execute_existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),
      Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers)):
  %Equal.sym(Bands.Bands(~F32) & Bands.Plan,Bands.plan(~F32,Model.pack(~F32,source)),(Model.pack(~F32,source),Model.layout(~F32,source)),Model.plan(~F32,source)) :
    Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),TensorConvolution.existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),
      Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers,_))
  existing_layout_header(~custom,~Logical,frame,batched,workers,source,weights,bias)

def checked_header(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  source_shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  weight_length: {Buffer.length(~F32,weights) == PublicFrame.weight_length(~Logical,frame) : Nat},bias_length: {Buffer.length(~F32,bias) == PublicFrame.bias_length(~Logical,frame) : Nat},
  weight_ready: Buffer.Ready<F32,weights>,bias_ready: Buffer.Ready<F32,bias>) ->
  Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),
    TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{TensorExecution.execution_geometry(~Logical,frame,batched)},PipelineExecution.input_geometry(~Logical,frame),
      Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers)):
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{TensorExecution.execution_geometry(~Logical,frame,batched)},PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    Inspected.checked(~custom,~Logical,~original,frame,batched,source,weights,bias,shape,fill,workers,source_shape,weight_length,bias_length,weight_ready,bias_ready)) : Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),_)
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.band_supported(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    TensorConvolution.execute_existing_layout(~custom,TensorExecution.execution_activation(~Logical,frame),True{},TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers),
    MachineCaller.supported(~custom,~Logical,~original,frame,batched,Model.pack(~F32,source),weights,bias,shape,fill,workers,limits_exact)) : Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),_)
  planned_header(~custom,~Logical,frame,batched,workers,source,weights,bias)

def shape_receipt(+batched: Bool,+expected_geometry: Bands.Geometry,+geometry: Bands.Geometry,+shape: List<&2,U32>,+status: Tensors.Status,
  geometry_exact: {geometry == expected_geometry : Bands.Geometry},shape_exact: {shape == extents(batched,expected_geometry) : List<&2,U32>},status_exact: {status == Tensors.Ready{} : Tensors.Status},
  counted: Global.Counted<extents(batched,expected_geometry)>,matched: {shape_matched(images(expected_geometry),batched) == True{} : Bool}) ->
  Global.BandShape(status,shape,geometry):
  %Equal.sym(Tensors.Status,status,Tensors.Ready{},status_exact) : Global.BandShape(_,shape,geometry)
  %Equal.sym(List<&2,U32>,shape,extents(batched,expected_geometry),shape_exact) : Global.BandShape(Tensors.Ready{},_,geometry)
  %Equal.sym(Bands.Geometry,geometry,expected_geometry,geometry_exact) : Global.BandShape(Tensors.Ready{},extents(batched,expected_geometry),_)
  Shape.Both{counted,coherent(batched,expected_geometry,matched)}

def normal_from_header(+batched: Bool,+expected_geometry: Bands.Geometry,tensor: Tensors.Tensor<F32>,
  header: Header(batched,expected_geometry,tensor),counted: Global.Counted<extents(batched,expected_geometry)>,
  matched: {shape_matched(images(expected_geometry),batched) == True{} : Bool},well_formed: BandMap.WellFormed(~F32,tensor)) -> Global.Normal(~F32,tensor):
  match tensor:
    case Tensors.Banded{bands,+geometry,+shape,fill,+status,workers}:
      Shape.Both{geometry_exact,Shape.Both{shape_exact,status_exact}} = header
      Shape.Both{well_formed,shape_receipt(batched,expected_geometry,geometry,shape,status,geometry_exact,shape_exact,status_exact,counted,matched)}
    case Tensors.Tensor{storage,shape,fill,status,workers}: Empty.absurd(Global.Normal(~F32,tensor),header)

def completed_counted(+extents: List<&2,U32>,+count: U32,evidence: Count.Count<ShapeLaws.elements(extents),count>) -> Global.Counted<extents>:
  Count.Count{exact,bounded} = evidence
  Global.Counted{count,CountCompletion.count_of(extents,count,exact,bounded)}

def contract_shape(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,-tensor: Tensors.Tensor<F32>,
  contract: PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),tensor)) -> BandMap.WellFormed(~F32,tensor):
  Shape.Both{values,well_formed} = contract
  well_formed

# The output receipt is made from the actual checked batch envelope; callers
# do not supply an output count or a precomputed shape-success equation.
def band_output_normal(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +matched: {shape_matched(images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  source_shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))) ->
  Global.Normal(~F32,TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),TensorExecution.execution_parallel(~Logical,frame),
    Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers)):
  normal_from_header(batched,OutputMetadata.bounds_geometry(~Logical,frame),
    TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),TensorExecution.execution_parallel(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers),
    band_output_header(~custom,~Logical,frame,batched,TensorExecution.execution_parallel(~Logical,frame),workers,Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),
    completed_counted(extents(batched,OutputMetadata.bounds_geometry(~Logical,frame)),Bands.elements(OutputMetadata.bounds_geometry(~Logical,frame)),counted(~Logical,~original,frame,batched,matched,limits_exact)),
    matched,TensorExecution.band_output_well_formed(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,source_shape,values))

def checked_normal(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +matched: {shape_matched(images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  +limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  +source_shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  +weight_length: {Buffer.length(~F32,weights) == PublicFrame.weight_length(~Logical,frame) : Nat},+bias_length: {Buffer.length(~F32,bias) == PublicFrame.bias_length(~Logical,frame) : Nat},
  +weight_ready: Buffer.Ready<F32,weights>,+bias_ready: Buffer.Ready<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  Global.Normal(~F32,TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{TensorExecution.execution_geometry(~Logical,frame,batched)},PipelineExecution.input_geometry(~Logical,frame),
    Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers)):
  normal_from_header(batched,OutputMetadata.bounds_geometry(~Logical,frame),
    TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{TensorExecution.execution_geometry(~Logical,frame,batched)},PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    checked_header(~custom,~Logical,~original,frame,batched,source,weights,bias,shape,fill,workers,limits_exact,source_shape,weight_length,bias_length,weight_ready,bias_ready),
    completed_counted(extents(batched,OutputMetadata.bounds_geometry(~Logical,frame)),Bands.elements(OutputMetadata.bounds_geometry(~Logical,frame)),counted(~Logical,~original,frame,batched,matched,limits_exact)),
    matched,contract_shape(~custom,~Logical,~original,frame,
      TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{TensorExecution.execution_geometry(~Logical,frame,batched)},PipelineExecution.input_geometry(~Logical,frame),Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
      MachineCaller.checked(~custom,~Logical,~original,frame,batched,source,weights,bias,shape,fill,workers,limits_exact,source_shape,values,weight_length,bias_length,weight_ready,bias_ready,weight_exact,bias_exact)))
