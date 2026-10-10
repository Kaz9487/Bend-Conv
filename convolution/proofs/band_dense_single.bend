# The single-image dense caller: arguments, image, fallback.
import Base
import ./band_dense_convolution.bend as DenseConvolution
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../inference.bend as NativeConv
import ./prepared_output_readback.bend as PreparedOutputReadback
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/tensor_dense_contract.bend as Dense
import ./input_storage.bend as Inputs
import ./microkernel_inference_refinement.bend as Inference
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../matrix_specification.bend as MatrixSpec
import ./band_dense_pipeline.bend as DensePipeline
import ./band_dense_pipeline_content.bend as Content
import ./concrete_support_guard.bend as Support
import ./convolution_guard_facts.bend as Guards
import ./native_inference_law.bend as InferenceLaw
import ./reduction_coordinate_bounds.bend as Coordinates
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ../shape.bend as ConvShape
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/storage_allocation_refinement.bend as AllocationLaws
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor_band.bend as Bands
import ../../lib/kernel_tile.bend as Tile

# A dense image keeps the supplied input trees and their certificates. The
# storage contract is constructed from their real leading contents and room;
# finite output values and Ready are then produced by the dense leaf schedule.
type SourceStorage<-problem: ConvSpec.Problem,-input: Buffer.Model<F32>,-weights: Buffer.Model<F32>,-bias: Buffer.Model<F32>> is Data:
  SourceStorage{storage: Inputs.Contract<problem>,
    input_tree: {Inputs.input_tree(problem,storage) == Buffer.tree(~F32,input) : Tree.Tree<F32>},
    weight_tree: {Inputs.weight_tree(problem,storage) == Buffer.tree(~F32,weights) : Tree.Tree<F32>},
    bias_tree: {Inputs.bias_tree(problem,storage) == Buffer.tree(~F32,bias) : Tree.Tree<F32>}}

def source_storage(+problem: ConvSpec.Problem,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  input_values: {Tree.read_sequence(~F32,ConvSpec.length(Guards.input(problem)),0,Buffer.tree(~F32,input)) == Guards.input(problem) : List<&2,F32>},
  weight_values: {Tree.read_sequence(~F32,ConvSpec.length(Guards.weights(problem)),0,Buffer.tree(~F32,weights)) == Guards.weights(problem) : List<&2,F32>},
  bias_values: {Tree.read_sequence(~F32,ConvSpec.length(Guards.bias(problem)),0,Buffer.tree(~F32,bias)) == Guards.bias(problem) : List<&2,F32>},
  input_room: {Nat.is_le(ConvSpec.length(Guards.input(problem)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,input)))) == True{} : Bool},
  weight_room: {Nat.is_le(ConvSpec.length(Guards.weights(problem)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,weights)))) == True{} : Bool},
  bias_room: {Nat.is_le(ConvSpec.length(Guards.bias(problem)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,bias)))) == True{} : Bool}) -> SourceStorage<problem,input,weights,bias>:
  ConvSpec.Problem{+shape,+xs,+ws,+bs} = problem
  SourceStorage{Inputs.Contract{
    StorageLaws.SequenceEvidence{Buffer.tree(~F32,input),input_values,input_room},
    StorageLaws.SequenceEvidence{Buffer.tree(~F32,weights),weight_values,weight_room},
    StorageLaws.SequenceEvidence{Buffer.tree(~F32,bias),bias_values,bias_room}},
    {==},{==},{==}}

def count_matches(+problem: ConvSpec.Problem) -> {DensePipeline.output_size(Inference.work(problem)) == PreparedOutputReadback.count(problem) : U32}:
  ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},xs,ws,bs} = problem
  {==}

def source_contract(-problem: ConvSpec.Problem,-input: Buffer.Model<F32>,-weights: Buffer.Model<F32>,-bias: Buffer.Model<F32>,source: SourceStorage<problem,input,weights,bias>) -> Inputs.Contract<problem>:
  SourceStorage{storage,input_tree,weight_tree,bias_tree} = source
  storage

def run_content(~custom: F32 -> F32,+activation: Activations.Activation,+problem: ConvSpec.Problem,+depth: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +source: SourceStorage<problem,input,weights,bias>) ->
  {Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) ==
    DenseConvolution.selected_content(~custom,activation,problem,source_contract(problem,input,weights,bias,source),depth) : Buffer.Content<F32>}:
  SourceStorage{+storage,+input_tree,+weight_tree,+bias_tree} = source
  %Equal.sym(Tree.Tree<F32>,Inputs.input_tree(problem,storage),Buffer.tree(~F32,input),input_tree) :
    {Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) ==
      Buffer.Content{DenseConvolution.run_model(~custom,activation,depth,Inference.work(problem),_,Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)),U32.to_nat(PreparedOutputReadback.count(problem))} : Buffer.Content<F32>}
  %Equal.sym(Tree.Tree<F32>,Inputs.weight_tree(problem,storage),Buffer.tree(~F32,weights),weight_tree) :
    {Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) ==
      Buffer.Content{DenseConvolution.run_model(~custom,activation,depth,Inference.work(problem),Buffer.tree(~F32,input),_,Inputs.bias_tree(problem,storage)),U32.to_nat(PreparedOutputReadback.count(problem))} : Buffer.Content<F32>}
  %Equal.sym(Tree.Tree<F32>,Inputs.bias_tree(problem,storage),Buffer.tree(~F32,bias),bias_tree) :
    {Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) ==
      Buffer.Content{DenseConvolution.run_model(~custom,activation,depth,Inference.work(problem),Buffer.tree(~F32,input),Buffer.tree(~F32,weights),_),U32.to_nat(PreparedOutputReadback.count(problem))} : Buffer.Content<F32>}
  %count_matches(problem) :
    {Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))) ==
      Buffer.Content{DenseConvolution.run_model(~custom,activation,depth,Inference.work(problem),Buffer.tree(~F32,input),Buffer.tree(~F32,weights),Buffer.tree(~F32,bias)),U32.to_nat(_)} : Buffer.Content<F32>}
  Content.run_contents(~custom,activation,depth,Inference.work(problem),input,weights,bias)

def run_values(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+depth: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+source: SourceStorage<problem,input,weights,bias>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  Dense.Values(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{problem,activation},
    Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)))):
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),
    DenseConvolution.selected_content(~custom,activation,problem,source_contract(problem,input,weights,bias,source),depth),run_content(~custom,activation,problem,depth,input,weights,bias,source)) :
    Dense.Values(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{problem,activation},_)
  DenseConvolution.finite_values(~custom,activation,limits,problem,source_contract(problem,input,weights,bias,source),depth,valid,supported)

def dimensions(problem: ConvSpec.Problem) -> KernelShape.Dimensions:
  ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},xs,ws,bs} = problem
  KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding}

def image_work(problem: ConvSpec.Problem) -> Pipeline.Work:
  ConvSpec.Problem{ConvSpec.Shape{+ci,co,+height,+width,+kernel,+stride,+padding},xs,ws,bs} = problem
  +oh = U32.from_nat(ConvSpec.axis_size(height,kernel,stride,padding))
  +ow = U32.from_nat(ConvSpec.axis_size(width,kernel,stride,padding))
  Pipeline.Work{U32.from_nat(co),U32.from_nat(height),U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),ow,(oh * ow : U32),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))}

def inner_matches(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {U32.from_nat(Coordinates.inner(problem)) == Coordinates.native_inner(problem) : U32}:
  Equal.trans(U32,U32.from_nat(Coordinates.inner(problem)),U32.from_nat(U32.to_nat(Coordinates.native_inner(problem))),Coordinates.native_inner(problem),
    Equal.cong(Nat,U32,value => U32.from_nat(value),Coordinates.inner(problem),U32.to_nat(Coordinates.native_inner(problem)),
      Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(problem)),Coordinates.inner(problem),Coordinates.inner_exact(limits,problem,Guards.valid(problem,valid),Support.scalar(limits,problem,Support.scalar_guard(limits,problem,supported))))),
    Arithmetic.native_roundtrip(Coordinates.native_inner(problem)))

def work_matches(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {image_work(problem) == Inference.work(problem) : Pipeline.Work}:
  ConvSpec.Problem{ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding},+xs,+ws,+bs} = problem
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},xs,ws,bs} : ConvSpec.Problem}
  Equal.cong(U32,Pipeline.Work,inner => Pipeline.Work{U32.from_nat(co),U32.from_nat(height),U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),U32.from_nat(ConvSpec.out_width(problem)),
      (U32.from_nat(ConvSpec.out_height(problem)) * U32.from_nat(ConvSpec.out_width(problem)) : U32),inner},
    U32.from_nat(Coordinates.inner(problem)),Coordinates.native_inner(problem),inner_matches(limits,problem,valid,supported))

def output_extents(+problem: ConvSpec.Problem,batched: Bool) -> List<&2,U32>:
  TensorConvolution.output_shape(batched,1,U32.from_nat(Guards.output_channel_count(problem)),U32.from_nat(ConvSpec.out_height(problem)),U32.from_nat(ConvSpec.out_width(problem)))

def shape_words(+batched: Bool,+rows: U32,+height: U32,+width: U32) ->
  {ShapeLaws.elements(TensorConvolution.output_shape(batched,1,rows,height,width)) == Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat}:
  match batched:
    case False{}:
      Equal.cong(Nat,Nat,last => Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(height),last)),Nat.mul(U32.to_nat(width),1n),U32.to_nat(width),ShapeLaws.times_one(U32.to_nat(width)))
    case True{}:
      Equal.trans(Nat,ShapeLaws.elements(TensorConvolution.output_shape(True{},1,rows,height,width)),
        Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(height),Nat.mul(U32.to_nat(width),1n))),
        Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(height),U32.to_nat(width))),
        Addition.zero_right(Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(height),Nat.mul(U32.to_nat(width),1n)))),
        Equal.cong(Nat,Nat,last => Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(height),last)),Nat.mul(U32.to_nat(width),1n),U32.to_nat(width),ShapeLaws.times_one(U32.to_nat(width))))

def shape_count(+limits: Machine.Limits,+problem: ConvSpec.Problem,+batched: Bool,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {ShapeLaws.elements(output_extents(problem,batched)) == U32.to_nat(PreparedOutputReadback.count(problem)) : Nat}:
  +facts = Guards.valid(problem,valid)
  +scalar = Support.scalar(limits,problem,Support.scalar_guard(limits,problem,supported))
  +rows_exact = SupportedCoordinateBounds.co_exact(limits,problem,scalar)
  +height_exact = Bounds.roundtrip(ConvSpec.out_height(problem),Machine.coordinate(limits),Limits.coordinate_fits(limits),SupportedCoordinateBounds.row_limit(limits,problem,facts,scalar))
  +width_exact = Bounds.roundtrip(ConvSpec.out_width(problem),Machine.coordinate(limits),Limits.coordinate_fits(limits),SupportedCoordinateBounds.col_limit(limits,problem,facts,scalar))
  Equal.trans(Nat,ShapeLaws.elements(output_extents(problem,batched)),
    Nat.mul(U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))),Nat.mul(U32.to_nat(U32.from_nat(ConvSpec.out_height(problem))),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))))),
    U32.to_nat(PreparedOutputReadback.count(problem)),shape_words(batched,U32.from_nat(Guards.output_channel_count(problem)),U32.from_nat(ConvSpec.out_height(problem)),U32.from_nat(ConvSpec.out_width(problem))),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))),Guards.output_channel_count(problem),rows_exact) :
      {Nat.mul(_,Nat.mul(U32.to_nat(U32.from_nat(ConvSpec.out_height(problem))),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))))) == U32.to_nat(PreparedOutputReadback.count(problem)) : Nat}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_height(problem))),ConvSpec.out_height(problem),height_exact) :
      {Nat.mul(Guards.output_channel_count(problem),Nat.mul(_,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))))) == U32.to_nat(PreparedOutputReadback.count(problem)) : Nat}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ConvSpec.out_width(problem),width_exact) :
      {Nat.mul(Guards.output_channel_count(problem),Nat.mul(ConvSpec.out_height(problem),_)) == U32.to_nat(PreparedOutputReadback.count(problem)) : Nat}
    Equal.sym(Nat,U32.to_nat(PreparedOutputReadback.count(problem)),Nat.mul(Guards.output_channel_count(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))),PreparedOutputReadback.count_exact(limits,problem,valid,supported)))

def run_contract(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+batched: Bool,+depth: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32,+source: SourceStorage<problem,input,weights,bias>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{problem,activation},
    Tensors.Tensor{Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),output_extents(problem,batched),0.0,Tensors.Usable{},workers}):
  Shape.Both{Shape.Both{Dense.WellFormed{
    Equal.trans(Nat,Buffer.content_length(~F32,Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)))),
      U32.to_nat(PreparedOutputReadback.count(problem)),ShapeLaws.elements(output_extents(problem,batched)),
      Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),
        Buffer.contents(~F32,Pipeline.run(~custom,activation,depth,Inference.work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))),
        DenseConvolution.selected_content(~custom,activation,problem,source_contract(problem,input,weights,bias,source),depth),run_content(~custom,activation,problem,depth,input,weights,bias,source)),
      Equal.sym(Nat,ShapeLaws.elements(output_extents(problem,batched)),U32.to_nat(PreparedOutputReadback.count(problem)),shape_count(limits,problem,batched,valid,supported))),
    Content.run_supported(~custom,activation,limits,problem,depth,input,weights,bias,valid,supported)},{==}},
    run_values(~custom,activation,limits,problem,depth,input,weights,bias,source,valid,supported)}

# A single-image execution at any selected depth has the finite leaf contract.
# Actual default planning is isolated in band_dense_planner_schedule.
def caller_input_size(problem: ConvSpec.Problem) -> U32:
  ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},xs,ws,bs} = problem
  U32.from_nat(Nat.mul(ci,Nat.mul(height,width)))

# The caller's weights in the order the kernel reads them.
def packed(problem: ConvSpec.Problem,weights: Buffer.Model<F32>) -> Storage.Buffer<Tile.Tile8>:
  TensorConvolution.packed(TensorConvolution.Geometry{1,False{},dimensions(problem)},Buffer.pack(~F32,weights))

def executed(~custom: F32 -> F32,+activation: Activations.Activation,+problem: ConvSpec.Problem,+batched: Bool,+selected: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32) ->
  {TensorConvolution.execute_dense_at_depth(~custom,activation,1,selected,image_work(problem),caller_input_size(problem),PreparedOutputReadback.count(problem),output_extents(problem,batched),Buffer.pack(~F32,input),packed(problem,weights),Buffer.pack(~F32,bias),workers) ==
    Tensors.Tensor{Pipeline.run(~custom,activation,selected,image_work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),output_extents(problem,batched),0.0,Tensors.Usable{},workers} : Tensors.Tensor<F32>}:
  ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},xs,ws,bs} = problem
  {==}

type ActualInput is Data:
  ActualInput{problem: ConvSpec.Problem,activation: Activations.Activation,batched: Bool,
    input: Buffer.Model<F32>,weights: Buffer.Model<F32>,bias: Buffer.Model<F32>,workers: U32}

def at_depth(~custom: F32 -> F32,input: ActualInput,selected: U32) -> Tensors.Tensor<F32>:
  ActualInput{+problem,activation,+batched,source,weights,bias,workers} = input
  TensorConvolution.execute_dense_at_depth(~custom,activation,1,selected,image_work(problem),caller_input_size(problem),PreparedOutputReadback.count(problem),output_extents(problem,batched),Buffer.pack(~F32,source),packed(problem,weights),Buffer.pack(~F32,bias),workers)

def logical(input: ActualInput) -> DenseConvolution.Logical:
  ActualInput{problem,activation,batched,source,weights,bias,workers} = input
  DenseConvolution.Logical{problem,activation}

def Source(input: ActualInput) -> Data:
  ActualInput{problem,activation,batched,source,weights,bias,workers} = input
  SourceStorage<problem,source,weights,bias>

def is_valid(input: ActualInput) -> Bool:
  ActualInput{problem,activation,batched,source,weights,bias,workers} = input
  ConvSpec.valid(problem)

def is_supported(+limits: Machine.Limits,input: ActualInput) -> Bool:
  ActualInput{problem,activation,batched,source,weights,bias,workers} = input
  NativeConv.supports_packed_with_limits(limits,problem)

def configured_parts(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+batched: Bool,+input: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32,+selected: U32,
  +source: SourceStorage<problem,input,weights,bias>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{problem,activation},
    TensorConvolution.execute_dense_at_depth(~custom,activation,1,selected,image_work(problem),caller_input_size(problem),PreparedOutputReadback.count(problem),output_extents(problem,batched),Buffer.pack(~F32,input),packed(problem,weights),Buffer.pack(~F32,bias),workers)):
  %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.execute_dense_at_depth(~custom,activation,1,selected,image_work(problem),caller_input_size(problem),PreparedOutputReadback.count(problem),output_extents(problem,batched),Buffer.pack(~F32,input),packed(problem,weights),Buffer.pack(~F32,bias),workers),
    Tensors.Tensor{Pipeline.run(~custom,activation,selected,image_work(problem),Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),output_extents(problem,batched),0.0,Tensors.Usable{},workers},
    executed(~custom,activation,problem,batched,selected,input,weights,bias,workers)) : Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{problem,activation},_)
  %Equal.sym(Pipeline.Work,image_work(problem),Inference.work(problem),work_matches(limits,problem,valid,supported)) :
    Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),DenseConvolution.Logical{problem,activation},Tensors.Tensor{Pipeline.run(~custom,activation,selected,_,Buffer.pack(~F32,input),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias)),output_extents(problem,batched),0.0,Tensors.Usable{},workers})
  run_contract(~custom,activation,limits,problem,batched,selected,input,weights,bias,workers,source,valid,supported)

def selected_contract(~custom: F32 -> F32,+input: ActualInput,+limits: Machine.Limits,+selected: U32,+source: Source(input),
  +valid: {is_valid(input) == True{} : Bool},+supported: {is_supported(limits,input) == True{} : Bool}) ->
  Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),logical(input),at_depth(~custom,input,selected)):
  ActualInput{+problem,+activation,+batched,+source_model,+weights,+bias,+workers} = input
  configured_parts(~custom,activation,limits,problem,batched,source_model,weights,bias,workers,selected,source,valid,supported)

# Observe the runtime's actual dense argument record before consuming it.
# Metadata projections never inspect the selected planner depth.
def problem(input: ActualInput) -> ConvSpec.Problem:
  ActualInput{problem,activation,batched,source,weights,bias,workers} = input
  problem

def batched(input: ActualInput) -> Bool:
  ActualInput{problem,activation,batched,source,weights,bias,workers} = input
  batched

def arguments(input: ActualInput) -> TensorConvolution.DenseArguments:
  ActualInput{problem,activation,batched,source,weights,bias,workers} = input
  TensorConvolution.dense_arguments(TensorConvolution.Geometry{1,batched,dimensions(problem)},Buffer.depth(~F32,bias),workers,activation)

def count(arguments: TensorConvolution.DenseArguments) -> U32:
  TensorConvolution.DenseArguments{count,arguments_work,arguments_input_size,output_size,extents,selected} = arguments
  count

def arguments_work(arguments: TensorConvolution.DenseArguments) -> Pipeline.Work:
  TensorConvolution.DenseArguments{count,arguments_work,arguments_input_size,output_size,extents,selected} = arguments
  arguments_work

def arguments_input_size(arguments: TensorConvolution.DenseArguments) -> U32:
  TensorConvolution.DenseArguments{count,arguments_work,arguments_input_size,output_size,extents,selected} = arguments
  arguments_input_size

def output_size(arguments: TensorConvolution.DenseArguments) -> U32:
  TensorConvolution.DenseArguments{count,arguments_work,arguments_input_size,output_size,extents,selected} = arguments
  output_size

def extents(arguments: TensorConvolution.DenseArguments) -> List<&2,U32>:
  TensorConvolution.DenseArguments{count,arguments_work,arguments_input_size,output_size,extents,selected} = arguments
  extents

def selected(arguments: TensorConvolution.DenseArguments) -> U32:
  TensorConvolution.DenseArguments{count,arguments_work,arguments_input_size,output_size,extents,selected} = arguments
  selected

def execute(~custom: F32 -> F32,input: ActualInput,arguments: TensorConvolution.DenseArguments) -> Tensors.Tensor<F32>:
  ActualInput{problem,activation,batched,source,weights,bias,workers} = input
  TensorConvolution.execute_dense_arguments(~custom,activation,arguments,Buffer.pack(~F32,source),packed(problem,weights),Buffer.pack(~F32,bias),workers)

def result(~custom: F32 -> F32,input: ActualInput) -> Tensors.Tensor<F32>:
  ActualInput{+problem,activation,batched,source,weights,bias,workers} = input
  TensorConvolution.execute_dense(~custom,activation,TensorConvolution.Geometry{1,batched,dimensions(problem)},Buffer.pack(~F32,source),packed(problem,weights),Buffer.pack(~F32,bias),workers)

# Leave the problem opaque here. Both executor heads contain the exact same
# dense_arguments producer, including its original default cost calculation.
def header(~custom: F32 -> F32,+input: ActualInput) ->
  {result(~custom,input) == execute(~custom,input,arguments(input)) : Tensors.Tensor<F32>}:
  ActualInput{problem,activation,batched,source,weights,Buffer.Model{tree,length,Certificate.Certificate{depth,equation}},workers} = input
  {==}

type Observation<-input: ActualInput> is Data:
  Observation{actual: TensorConvolution.DenseArguments,exact: {actual == arguments(input) : TensorConvolution.DenseArguments}}

def observe(+input: ActualInput) -> Observation<input>:
  Observation{arguments(input),{==}}

type Metadata<-input: ActualInput,-actual: TensorConvolution.DenseArguments> is Data:
  Metadata{count_exact: {count(actual) == 1 : U32},
    work_exact: {arguments_work(actual) == image_work(problem(input)) : Pipeline.Work},
    input_exact: {arguments_input_size(actual) == caller_input_size(problem(input)) : U32},
    output_exact: {output_size(actual) == PreparedOutputReadback.count(problem(input)) : U32},
    extents_exact: {extents(actual) == output_extents(problem(input),batched(input)) : List<&2,U32>}}

def metadata(+input: ActualInput,+limits: Machine.Limits,+valid: {is_valid(input) == True{} : Bool},+supported: {is_supported(limits,input) == True{} : Bool}) -> Metadata<input,arguments(input)>:
  ActualInput{ConvSpec.Problem{ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding},+xs,+ws,+bs},+activation,+batched,+source,+weights,+bias,+workers} = input
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},xs,ws,bs} : ConvSpec.Problem}
  Metadata{{==},{==},{==},DenseConvolution.matches(limits,problem,valid,supported),{==}}

def scheduled(~custom: F32 -> F32,+input: ActualInput,+actual: TensorConvolution.DenseArguments,evidence: Metadata<input,actual>) ->
  {execute(~custom,input,actual) == at_depth(~custom,input,selected(actual)) : Tensors.Tensor<F32>}:
  ActualInput{+problem,+activation,+batched,+source,+weights,+bias,+workers} = input
  TensorConvolution.DenseArguments{+count,+arguments_work,+arguments_input_size,+output_size,+extents,+selected} = actual
  Metadata{+count_exact,+work_exact,+input_exact,+output_exact,+extents_exact} = evidence
  %Equal.sym(U32,count,1,count_exact) :
    {TensorConvolution.execute_dense_at_depth(~custom,activation,_,selected,arguments_work,arguments_input_size,output_size,extents,Buffer.pack(~F32,source),packed(problem,weights),Buffer.pack(~F32,bias),workers) == at_depth(~custom,input,selected) : Tensors.Tensor<F32>}
  %Equal.sym(Pipeline.Work,arguments_work,image_work(problem),work_exact) :
    {TensorConvolution.execute_dense_at_depth(~custom,activation,1,selected,_,arguments_input_size,output_size,extents,Buffer.pack(~F32,source),packed(problem,weights),Buffer.pack(~F32,bias),workers) == at_depth(~custom,input,selected) : Tensors.Tensor<F32>}
  %Equal.sym(U32,arguments_input_size,caller_input_size(problem),input_exact) :
    {TensorConvolution.execute_dense_at_depth(~custom,activation,1,selected,image_work(problem),_,output_size,extents,Buffer.pack(~F32,source),packed(problem,weights),Buffer.pack(~F32,bias),workers) == at_depth(~custom,input,selected) : Tensors.Tensor<F32>}
  %Equal.sym(U32,output_size,PreparedOutputReadback.count(problem),output_exact) :
    {TensorConvolution.execute_dense_at_depth(~custom,activation,1,selected,image_work(problem),caller_input_size(problem),_,extents,Buffer.pack(~F32,source),packed(problem,weights),Buffer.pack(~F32,bias),workers) == at_depth(~custom,input,selected) : Tensors.Tensor<F32>}
  %Equal.sym(List<&2,U32>,extents,output_extents(problem,batched),extents_exact) :
    {TensorConvolution.execute_dense_at_depth(~custom,activation,1,selected,image_work(problem),caller_input_size(problem),PreparedOutputReadback.count(problem),_,Buffer.pack(~F32,source),packed(problem,weights),Buffer.pack(~F32,bias),workers) == at_depth(~custom,input,selected) : Tensors.Tensor<F32>}
  {==}

def configured(~custom: F32 -> F32,+input: ActualInput,+limits: Machine.Limits,+actual: TensorConvolution.DenseArguments,+facts: Metadata<input,actual>,+source: Source(input),
  +valid: {is_valid(input) == True{} : Bool},+supported: {is_supported(limits,input) == True{} : Bool}) ->
  Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),logical(input),execute(~custom,input,actual)):
  %Equal.sym(Tensors.Tensor<F32>,execute(~custom,input,actual),at_depth(~custom,input,selected(actual)),scheduled(~custom,input,actual,facts)) :
    Dense.Contract(~F32,~DenseConvolution.Logical,~DenseConvolution.original(~custom),logical(input),_)
  selected_contract(~custom,input,limits,selected(actual),source,valid,supported)

# Dense fallback keeps the existing gather and dense convolution caller.
# These bridges establish the actual route and the empty-batch result. The
# nonempty Buffer pipeline, activation and batched copies still need their
# caller refinement before the Array readback theorem proves that Tensor path.
def gathered(+layout: Bands.Geometry,source: Bands.Bands(~F32)) -> Storage.Buffer<F32>:
  Bands.Geometry{+images,+channels,+height,+width} = layout
  Bands.gather(~F32,layout,source,Tensors.buffer(~F32,0.0,U32.from_nat(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))))))

def false_route(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: TensorConvolution.Geometry,+layout: Bands.Geometry,
  -source: Bands.Bands(~F32),-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>,+workers: U32) ->
  {TensorConvolution.execute_existing_layout(~custom,activation,False{},geometry,layout,source,weights,bias,workers) ==
    TensorConvolution.execute_dense(~custom,activation,geometry,gathered(layout,source),weights,bias,workers) : Tensors.Tensor<F32>}:
  Bands.Geometry{images,channels,height,width} = layout
  {==}

def selected_false(~custom: F32 -> F32,+activation: Activations.Activation,+selected: Bool,+geometry: TensorConvolution.Geometry,+layout: Bands.Geometry,
  -source: Bands.Bands(~F32),-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>,+workers: U32,disabled: {selected == False{} : Bool}) ->
  {TensorConvolution.execute_existing_layout(~custom,activation,selected,geometry,layout,source,weights,bias,workers) ==
    TensorConvolution.execute_dense(~custom,activation,geometry,gathered(layout,source),weights,bias,workers) : Tensors.Tensor<F32>}:
  %Equal.sym(Bool,selected,False{},disabled) :
    {TensorConvolution.execute_existing_layout(~custom,activation,_,geometry,layout,source,weights,bias,workers) ==
      TensorConvolution.execute_dense(~custom,activation,geometry,gathered(layout,source),weights,bias,workers) : Tensors.Tensor<F32>}
  false_route(~custom,activation,geometry,layout,source,weights,bias,workers)

def and_false_right(first: Bool) -> {first && False{} == False{} : Bool}:
  match first:
    case False{}: {==}
    case True{}: {==}

def padding_not_less(+kernel: Nat,+padding: Nat,large: {Nat.is_le(kernel,padding) == True{} : Bool}) -> {Nat.is_lt(padding,kernel) == False{} : Bool}:
  match kernel padding:
    case 0n +padding: Division.lt_zero(padding)
    case 1n+rest 0n: Empty.absurd({Nat.is_lt(0n,1n+rest) == False{} : Bool},WordEncoding.false_true(large))
    case 1n+kernel_rest 1n+padding_rest: padding_not_less(kernel_rest,padding_rest,large)

def padding_selector(+images: U32,+batched: Bool,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,
  large: {Nat.is_le(kernel,padding) == True{} : Bool}) ->
  {TensorConvolution.uses_band_windows(TensorConvolution.Geometry{images,batched,KernelShape.Dimensions{channels,outputs,height,width,kernel,stride,padding}}) == False{} : Bool}:
  %Equal.sym(Bool,Nat.is_lt(padding,kernel),False{},padding_not_less(kernel,padding,large)) : {U32.is_lt(0,images) && _ == False{} : Bool}
  and_false_right(U32.is_lt(0,images))

def empty_selector(+batched: Bool,dimensions: KernelShape.Dimensions) ->
  {TensorConvolution.uses_band_windows(TensorConvolution.Geometry{0,batched,dimensions}) == False{} : Bool}:
  KernelShape.Dimensions{channels,outputs,height,width,kernel,stride,padding} = dimensions
  {==}

def padding_route(~custom: F32 -> F32,+activation: Activations.Activation,+images: U32,+batched: Bool,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+layout: Bands.Geometry,
  -source: Bands.Bands(~F32),-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>,+workers: U32,large: {Nat.is_le(kernel,padding) == True{} : Bool}) ->
  {TensorConvolution.execute_existing_layout(~custom,activation,TensorConvolution.uses_band_windows(TensorConvolution.Geometry{images,batched,KernelShape.Dimensions{channels,outputs,height,width,kernel,stride,padding}}),
      TensorConvolution.Geometry{images,batched,KernelShape.Dimensions{channels,outputs,height,width,kernel,stride,padding}},layout,source,weights,bias,workers) ==
    TensorConvolution.execute_dense(~custom,activation,TensorConvolution.Geometry{images,batched,KernelShape.Dimensions{channels,outputs,height,width,kernel,stride,padding}},gathered(layout,source),weights,bias,workers) : Tensors.Tensor<F32>}:
  +geometry = {TensorConvolution.Geometry{images,batched,KernelShape.Dimensions{channels,outputs,height,width,kernel,stride,padding}} : TensorConvolution.Geometry}
  selected_false(~custom,activation,TensorConvolution.uses_band_windows(geometry),geometry,layout,source,weights,bias,workers,padding_selector(images,batched,channels,outputs,height,width,kernel,stride,padding,large))

def empty_output_shape(+batched: Bool,dimensions: KernelShape.Dimensions) -> List<&2,U32>:
  KernelShape.Dimensions{channels,outputs,height,width,+kernel,+stride,+padding} = dimensions
  TensorConvolution.output_shape(batched,0,U32.from_nat(outputs),U32.from_nat(KernelShape.axis_size(height,kernel,stride,padding)),U32.from_nat(KernelShape.axis_size(width,kernel,stride,padding)))

def empty_dense(~custom: F32 -> F32,+activation: Activations.Activation,+batched: Bool,+dimensions: KernelShape.Dimensions,-input: Storage.Buffer<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32) ->
  {TensorConvolution.execute_dense(~custom,activation,TensorConvolution.Geometry{0,batched,dimensions},input,TensorConvolution.packed(TensorConvolution.Geometry{0,batched,dimensions},Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers) ==
    Tensors.Tensor{Tensors.buffer(~F32,0.0,0),empty_output_shape(batched,dimensions),0.0,Tensors.Usable{},workers} : Tensors.Tensor<F32>}:
  KernelShape.Dimensions{channels,outputs,height,width,kernel,stride,padding} = dimensions
  Buffer.Model{tree,length,Certificate.Certificate{depth,equation}} = bias
  {==}

def empty_existing_layout(~custom: F32 -> F32,+activation: Activations.Activation,+batched: Bool,+dimensions: KernelShape.Dimensions,+layout: Bands.Geometry,
  -source: Bands.Bands(~F32),+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32) ->
  {TensorConvolution.execute_existing_layout(~custom,activation,TensorConvolution.uses_band_windows(TensorConvolution.Geometry{0,batched,dimensions}),TensorConvolution.Geometry{0,batched,dimensions},layout,source,TensorConvolution.packed(TensorConvolution.Geometry{0,batched,dimensions},Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers) ==
    Tensors.Tensor{Tensors.buffer(~F32,0.0,0),empty_output_shape(batched,dimensions),0.0,Tensors.Usable{},workers} : Tensors.Tensor<F32>}:
  +geometry = {TensorConvolution.Geometry{0,batched,dimensions} : TensorConvolution.Geometry}
  Equal.trans(Tensors.Tensor<F32>,TensorConvolution.execute_existing_layout(~custom,activation,TensorConvolution.uses_band_windows(geometry),geometry,layout,source,TensorConvolution.packed(TensorConvolution.Geometry{0,batched,dimensions},Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    TensorConvolution.execute_dense(~custom,activation,geometry,gathered(layout,source),TensorConvolution.packed(TensorConvolution.Geometry{0,batched,dimensions},Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    Tensors.Tensor{Tensors.buffer(~F32,0.0,0),empty_output_shape(batched,dimensions),0.0,Tensors.Usable{},workers},
    selected_false(~custom,activation,TensorConvolution.uses_band_windows(geometry),geometry,layout,source,TensorConvolution.packed(TensorConvolution.Geometry{0,batched,dimensions},Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers,empty_selector(batched,dimensions)),
    empty_dense(~custom,activation,batched,dimensions,gathered(layout,source),weights,bias,workers))

# Existing Array strategies already cover arbitrary supported padding. This
# is the leaf contract to reuse when the nonempty Tensor caller is connected;
# it does not assert that its Buffer pipeline equals the Array strategy.
def dense_leaf_readback(strategy: NativeConv.Strategy,+limits: Machine.Limits,+problem: ConvSpec.Problem,+inputs: Inputs.Contract<problem>,
  valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {InferenceLaw.values_of(problem,NativeConv.execute(strategy,False{},ConvShape.from_problem(problem),Inputs.input_owner(problem,inputs),Inputs.weight_owner(problem,inputs),Inputs.bias_owner(problem,inputs))) == MatrixSpec.values(problem) : List<&2,F32>}:
  InferenceLaw.strategy_readback_matches_matrix(strategy,limits,problem,inputs,valid,supported)
