# Direct sampling supplies the same ordered matrix inputs to OutputKernel.
# Storage and loop correctness are provided by the common library contracts.
import Base
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../scalar_kernel.bend as ScalarConv
import ../shape.bend as ConvShape
import ../../lib/storage_buffer.bend as Storage
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/word_multiplication.bend as Multiplication
import ../../lib/proofs/traversal_reduction_refinement.bend as ReductionLaws
import ../../lib/proofs/ordered_reduction_split.bend as Ordered
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/traversal_output_refinement.bend as OutputLaws
import ../../lib/traversal_output.bend as OutputTraversal
import ../../lib/traversal_reduction.bend as Reduction
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./scalar_output_coordinates.bend as OutputCoordinates
import ./reduction_coordinate_bounds.bend as Coordinates
import ./matrix_reduction_order.bend as MatrixOrder
import ./matrix_reduction_bounds.bend as ReductionBounds
import ./scalar_tensor_lengths.bend as Lengths
import ./native_weight_refinement.bend as NativeWeightRefinement
import ./input_storage.bend as Inputs
import ./scalar_convolution_reads.bend as ConvolutionReads
import ./execution_storage.bend as ExecutionStorage

type Evaluation is Data:
  Evaluation{limits: Machine.Limits,problem: ConvSpec.Problem,inputs: Inputs.Contract<problem>,pixel: OutputCoordinates.OutputPixel<problem>,
    valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool}}

def problem(evaluation: Evaluation) -> ConvSpec.Problem:
  Evaluation{limits,problem,inputs,pixel,valid,supported} = evaluation
  problem

def input_contract(+evaluation: Evaluation) -> Inputs.Contract<problem(evaluation)>:
  Evaluation{limits,problem,inputs,pixel,valid,supported} = evaluation
  inputs

def output_pixel(+evaluation: Evaluation) -> OutputCoordinates.OutputPixel<problem(evaluation)>:
  Evaluation{limits,problem,inputs,pixel,valid,supported} = evaluation
  pixel

def matrix_pixel(evaluation: Evaluation) -> MatrixSpec.Pixel:
  Evaluation{limits,problem,inputs,pixel,valid,supported} = evaluation
  OutputCoordinates.OutputPixel{channel,row,column,channel_inside,row_inside,column_inside} = pixel
  MatrixSpec.Position{row,column}

def context(+problem: ConvSpec.Problem,+pixel: OutputCoordinates.OutputPixel<problem>) -> ScalarConv.DirectContext:
  OutputCoordinates.OutputPixel{channel,row,column,channel_inside,row_inside,column_inside} = pixel
  ScalarConv.DirectContext{ConvShape.from_problem(problem),U32.from_nat(OutputCoordinates.index(problem,pixel)),U32.from_nat(channel),U32.from_nat(row),U32.from_nat(column)}

def read_value(result: Array<F32> & F32) -> F32:
  (owner,value) = result
  value

def reduction_index(+problem: ConvSpec.Problem,index: MatrixSpec.Reduction) -> Nat:
  MatrixSpec.KIndex{channel,row,column} = index
  Coordinates.reduction(channel,row,column,Guards.kernel(problem))

def reduction_exact(+evaluation: Evaluation,+index: MatrixSpec.Reduction,bounds: ReductionBounds.IndexBounds(problem(evaluation),index)) ->
  {U32.to_nat(U32.from_nat(reduction_index(problem(evaluation),index))) == reduction_index(problem(evaluation),index) : Nat}:
  Evaluation{+limits,+problem,inputs,pixel,+valid,+supported} = evaluation
  MatrixSpec.KIndex{+channel,+row,+column} = index
  Shape.Both{channel_inside,spatial} = bounds
  Shape.Both{row_inside,column_inside} = spatial
  Bounds.roundtrip(reduction_index(problem,index),Machine.capacity(limits),Limits.capacity_fits(limits),
    Bounds.strict_le(reduction_index(problem,index),U32.to_nat(Machine.capacity(limits)),
      Bounds.nat_trans(reduction_index(problem,index),Coordinates.inner(problem),U32.to_nat(Machine.capacity(limits)),
        Coordinates.reduction_inside(problem,channel,row,column,channel_inside,row_inside,column_inside),
        Coordinates.inner_bound(limits,problem,Guards.valid(problem,valid),Support.scalar(limits,problem,supported)))))

def reduction_channel(index: MatrixSpec.Reduction) -> Nat:
  MatrixSpec.KIndex{channel,row,column} = index
  channel

def reduction_row(index: MatrixSpec.Reduction) -> Nat:
  MatrixSpec.KIndex{channel,row,column} = index
  row

def reduction_column(index: MatrixSpec.Reduction) -> Nat:
  MatrixSpec.KIndex{channel,row,column} = index
  column

def input_read_decoded(+evaluation: Evaluation,+index: MatrixSpec.Reduction,+bounds: ReductionBounds.IndexBounds(problem(evaluation),index),
  decoded: Coordinates.DecodedReduction<U32.from_nat(reduction_index(problem(evaluation),index)),U32.from_nat(Guards.kernel(problem(evaluation))),
    reduction_channel(index),reduction_row(index),reduction_column(index)>) ->
  {ScalarConv.direct_input(~Array<F32>,~Storage.read_or(F32,0.0),
      Inputs.input_owner(problem(evaluation),input_contract(evaluation)),context(problem(evaluation),output_pixel(evaluation)),
      U32.from_nat(reduction_index(problem(evaluation),index))) ==
    (Inputs.input_owner(problem(evaluation),input_contract(evaluation)),MatrixSpec.patch(problem(evaluation),matrix_pixel(evaluation),index)) : Array<F32> & F32}:
  Evaluation{+limits,+problem,+inputs,+pixel,+valid,+supported} = evaluation
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+channels,+outputs,+height,+width,+kernel,+stride,+padding} = shape
  OutputCoordinates.OutputPixel{+output_channel,+output_row,+output_column,output_inside,row_inside,column_inside} = pixel
  MatrixSpec.KIndex{+input_channel,+kernel_row,+kernel_column} = index
  Shape.Both{+channel_inside,spatial} = bounds
  Shape.Both{+kernel_row_inside,+kernel_column_inside} = spatial
  Coordinates.DecodedReduction{channel_exact,row_exact,column_exact} = decoded
  +address = U32.from_nat(reduction_index(problem,index))
  %Equal.sym(U32,(address / (U32.from_nat(kernel) * U32.from_nat(kernel)) : U32),U32.from_nat(input_channel),channel_exact) :
    {ScalarConv.read_input(~Array<F32>,~Storage.read_or(F32,0.0),Inputs.input_owner(problem,inputs),
      ScalarConv.SampleContext{ConvShape.from_problem(problem),(address / U32.from_nat(kernel) % U32.from_nat(kernel) : U32),_,U32.from_nat(output_channel),U32.from_nat(output_row),U32.from_nat(output_column)},
      (address % U32.from_nat(kernel) : U32)) == (Inputs.input_owner(problem,inputs),MatrixSpec.patch(problem,MatrixSpec.Position{output_row,output_column},index)) : Array<F32> & F32}
  %Equal.sym(U32,(address / U32.from_nat(kernel) % U32.from_nat(kernel) : U32),U32.from_nat(kernel_row),row_exact) :
    {ScalarConv.read_input(~Array<F32>,~Storage.read_or(F32,0.0),Inputs.input_owner(problem,inputs),
      ScalarConv.SampleContext{ConvShape.from_problem(problem),_,U32.from_nat(input_channel),U32.from_nat(output_channel),U32.from_nat(output_row),U32.from_nat(output_column)},
      (address % U32.from_nat(kernel) : U32)) == (Inputs.input_owner(problem,inputs),MatrixSpec.patch(problem,MatrixSpec.Position{output_row,output_column},index)) : Array<F32> & F32}
  %Equal.sym(U32,(address % U32.from_nat(kernel) : U32),U32.from_nat(kernel_column),column_exact) :
    {ScalarConv.read_input(~Array<F32>,~Storage.read_or(F32,0.0),Inputs.input_owner(problem,inputs),
      ScalarConv.SampleContext{ConvShape.from_problem(problem),U32.from_nat(kernel_row),U32.from_nat(input_channel),U32.from_nat(output_channel),U32.from_nat(output_row),U32.from_nat(output_column)},
      _) == (Inputs.input_owner(problem,inputs),MatrixSpec.patch(problem,MatrixSpec.Position{output_row,output_column},index)) : Array<F32> & F32}
  ConvolutionReads.input_matches_spec(limits,problem,inputs,input_channel,output_row,output_column,kernel_row,kernel_column,valid,supported,
    channel_inside,row_inside,column_inside,kernel_row_inside,kernel_column_inside)

def input_read_matches(+evaluation: Evaluation,+index: MatrixSpec.Reduction,+bounds: ReductionBounds.IndexBounds(problem(evaluation),index)) ->
  {ScalarConv.direct_input(~Array<F32>,~Storage.read_or(F32,0.0),
      Inputs.input_owner(problem(evaluation),input_contract(evaluation)),context(problem(evaluation),output_pixel(evaluation)),
      U32.from_nat(reduction_index(problem(evaluation),index))) ==
    (Inputs.input_owner(problem(evaluation),input_contract(evaluation)),MatrixSpec.patch(problem(evaluation),matrix_pixel(evaluation),index)) : Array<F32> & F32}:
  Evaluation{+limits,+problem,inputs,pixel,+valid,+supported} = evaluation
  MatrixSpec.KIndex{+channel,+row,+column} = index
  Shape.Both{channel_inside,spatial} = bounds
  Shape.Both{+row_inside,+column_inside} = spatial
  input_read_decoded(evaluation,index,bounds,Coordinates.decode_reduction(limits,problem,U32.from_nat(reduction_index(problem,index)),channel,row,column,
    Guards.valid(problem,valid),Support.scalar(limits,problem,supported),reduction_exact(evaluation,index,bounds),row_inside,column_inside))

def weight_length_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,
  +supported: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {U32.to_nat((U32.from_nat(Guards.output_channel_count(problem)) * Coordinates.native_inner(problem) : U32)) == ConvSpec.length(Guards.weights(problem)) : Nat}:
  Guards.Valid{channels,outputs,height,width,kernel,stride,padding,extent,input,weights,bias} = valid
  +count = ConvSpec.length(Guards.weights(problem))
  +shape_exact = Guards.reflect(count,Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem)),weights)
  %Equal.sym(Nat,count,Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem)),shape_exact) :
    {U32.to_nat((U32.from_nat(Guards.output_channel_count(problem)) * Coordinates.native_inner(problem) : U32)) == _ : Nat}
  %Equal.sym(U32,(U32.from_nat(Guards.output_channel_count(problem)) * Coordinates.native_inner(problem) : U32),
    U32.from_nat(Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem))),
    Encoded.multiply_known_values(U32.from_nat(Guards.output_channel_count(problem)),Coordinates.native_inner(problem),Guards.output_channel_count(problem),Coordinates.inner(problem),
      SupportedCoordinateBounds.co_exact(limits,problem,supported),Coordinates.inner_exact(limits,problem,valid,supported))) :
    {U32.to_nat(_) == Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem)) : Nat}
  %shape_exact : {U32.to_nat(U32.from_nat(_)) == _ : Nat}
  Bounds.roundtrip(count,Machine.capacity(limits),Limits.capacity_fits(limits),NativeWeightRefinement.weights_bound(limits,problem,supported))

def weight_read_matches(+evaluation: Evaluation,+index: MatrixSpec.Reduction,+bounds: ReductionBounds.IndexBounds(problem(evaluation),index)) ->
  {ScalarConv.direct_weight(~Array<F32>,~Storage.read_or(F32,0.0),
      Inputs.weight_owner(problem(evaluation),input_contract(evaluation)),context(problem(evaluation),output_pixel(evaluation)),
      U32.from_nat(reduction_index(problem(evaluation),index))) ==
    (Inputs.weight_owner(problem(evaluation),input_contract(evaluation)),MatrixSpec.weight(problem(evaluation),OutputCoordinates.output_channel(problem(evaluation),output_pixel(evaluation)),index)) : Array<F32> & F32}:
  Evaluation{+limits,+problem,+inputs,+pixel,+valid,+supported} = evaluation
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+channels,+outputs,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{input_evidence,+weight_evidence,bias_evidence} = inputs
  OutputCoordinates.OutputPixel{+output_channel,output_row,output_column,+output_inside,row_inside,column_inside} = pixel
  MatrixSpec.KIndex{+input_channel,+kernel_row,+kernel_column} = index
  Shape.Both{+channel_inside,spatial} = bounds
  Shape.Both{+kernel_row_inside,+kernel_column_inside} = spatial
  +facts = Guards.valid(problem,valid)
  +support = Support.scalar(limits,problem,supported)
  +inner = Coordinates.inner(problem)
  +reduction = reduction_index(problem,index)
  +offset = Nat.mul(output_channel,inner)
  +weight_index = NativeWeightRefinement.weight_index(output_channel,channels,input_channel,kernel,kernel_row,kernel_column)
  +weight_index_exact = Bounds.roundtrip(weight_index,Machine.capacity(limits),Limits.capacity_fits(limits),
    Bounds.strict_le(weight_index,U32.to_nat(Machine.capacity(limits)),Bounds.nat_trans(weight_index,ConvSpec.length(weights),U32.to_nat(Machine.capacity(limits)),
      NativeWeightRefinement.index_bound(problem,output_channel,input_channel,kernel_row,kernel_column,facts,output_inside,channel_inside,kernel_row_inside,kernel_column_inside),NativeWeightRefinement.weights_bound(limits,problem,support))))
  %Equal.sym(U32,(U32.from_nat(output_channel) * Coordinates.native_inner(problem) : U32),U32.from_nat(offset),
    Encoded.multiply_known_values(U32.from_nat(output_channel),Coordinates.native_inner(problem),output_channel,inner,
      Bounds.roundtrip(output_channel,Machine.capacity(limits),Limits.capacity_fits(limits),NativeWeightRefinement.below_dimension(limits,output_channel,outputs,output_inside,SupportedCoordinateBounds.co_bound(limits,problem,support))),
      Coordinates.inner_exact(limits,problem,facts,support))) :
    {Storage.read_or(F32,0.0,Inputs.weight_owner(problem,inputs),(_ + U32.from_nat(reduction) : U32),(U32.from_nat(outputs) * Coordinates.native_inner(problem) : U32)) ==
      (Inputs.weight_owner(problem,inputs),ConvSpec.weight(problem,output_channel,input_channel,kernel_row,kernel_column)) : Array<F32> & F32}
  %Equal.sym(U32,(U32.from_nat(offset) + U32.from_nat(reduction) : U32),U32.from_nat(Nat.add(offset,reduction)),WordArithmetic.native_add(offset,reduction)) :
    {Storage.read_or(F32,0.0,Inputs.weight_owner(problem,inputs),_,(U32.from_nat(outputs) * Coordinates.native_inner(problem) : U32)) ==
      (Inputs.weight_owner(problem,inputs),ConvSpec.weight(problem,output_channel,input_channel,kernel_row,kernel_column)) : Array<F32> & F32}
  %Coordinates.weight_index(output_channel,channels,input_channel,kernel,kernel_row,kernel_column) :
    {Storage.read_or(F32,0.0,Inputs.weight_owner(problem,inputs),U32.from_nat(_),(U32.from_nat(outputs) * Coordinates.native_inner(problem) : U32)) ==
      (Inputs.weight_owner(problem,inputs),ConvSpec.weight(problem,output_channel,input_channel,kernel_row,kernel_column)) : Array<F32> & F32}
  %Equal.sym(F32,ConvSpec.at(weight_index,weights),Maybe.default(&2,F32,List.get(&2,F32,weights,weight_index),0.0),Inputs.lookup_matches_list(weight_index,weights)) :
    {Storage.read_or(F32,0.0,Inputs.weight_owner(problem,inputs),U32.from_nat(weight_index),(U32.from_nat(outputs) * Coordinates.native_inner(problem) : U32)) ==
      (Inputs.weight_owner(problem,inputs),_) : Array<F32> & F32}
  StorageLaws.read_encoded_sequence(~F32,0.0,ConvSpec.length(weights),weights,weight_evidence,U32.from_nat(weight_index),weight_index,
    (U32.from_nat(outputs) * Coordinates.native_inner(problem) : U32),weight_index_exact,weight_length_exact(limits,problem,facts,support))

def NativeSource() -> Data:
  ReductionLaws.SampleSource<F32,F32,ScalarConv.DirectContext>

def ReindexedSource() -> Data:
  Shape.Both<MatrixOrder.TapLayout,NativeSource()>

def input_at() -> NativeSource() -> Unit -> ReductionLaws.AddressPair -> F32:
  ReductionLaws.input_element(~F32,~F32,~ScalarConv.DirectContext,~F32,~Unit,~ExecutionStorage.direct_input_value,~ExecutionStorage.scalar_at)

def weight_at() -> NativeSource() -> Unit -> ReductionLaws.AddressPair -> F32:
  ReductionLaws.weight_element(~F32,~F32,~ScalarConv.DirectContext,~F32,~Unit,~ExecutionStorage.direct_weight_value,~ExecutionStorage.scalar_at)

def matrix_input() -> ReindexedSource() -> Unit -> MatrixSpec.Reduction -> F32:
  Ordered.mapped_sample(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~Unit,~NativeSource(),~MatrixOrder.TapLayout,~MatrixOrder.mapped_tap,~input_at())

def matrix_weight() -> ReindexedSource() -> Unit -> MatrixSpec.Reduction -> F32:
  Ordered.mapped_sample(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~Unit,~NativeSource(),~MatrixOrder.TapLayout,~MatrixOrder.mapped_tap,~weight_at())

def native_source(evaluation: Evaluation) -> NativeSource():
  Evaluation{limits,+problem,+inputs,pixel,valid,supported} = evaluation
  ReductionLaws.SampleSource{Inputs.input_tree(problem,inputs),Inputs.weight_tree(problem,inputs),context(problem,pixel)}

def matrix_source(+evaluation: Evaluation) -> ReindexedSource():
  Shape.Both{MatrixOrder.TapLayout{Guards.kernel(problem(evaluation)),0n,0n},native_source(evaluation)}

def output_channel(+evaluation: Evaluation) -> Nat:
  OutputCoordinates.output_channel(problem(evaluation),output_pixel(evaluation))

def SampleBounds(+evaluation: Evaluation,+index: MatrixSpec.Reduction) -> Data:
  ReductionBounds.IndexBounds(problem(evaluation),index)

def SampleAgreement(+evaluation: Evaluation,+index: MatrixSpec.Reduction) -> Data:
  Shape.Both<
    {matrix_input()(matrix_source(evaluation),Unit{},index) == MatrixSpec.patch(problem(evaluation),matrix_pixel(evaluation),index) : F32},
    {matrix_weight()(matrix_source(evaluation),Unit{},index) == MatrixSpec.weight(problem(evaluation),output_channel(evaluation),index) : F32}>

def input_matches_patch(+evaluation: Evaluation,+index: MatrixSpec.Reduction,bounds: SampleBounds(evaluation,index)) ->
  {matrix_input()(matrix_source(evaluation),Unit{},index) == MatrixSpec.patch(problem(evaluation),matrix_pixel(evaluation),index) : F32}:
  Evaluation{limits,+problem,+inputs,+pixel,valid,supported} = evaluation
  MatrixSpec.KIndex{+channel,+row,+column} = index
  +address = U32.from_nat(Coordinates.reduction(channel,row,column,Guards.kernel(problem)))
  %ExecutionStorage.direct_input_read(Inputs.input_tree(problem,inputs),context(problem,pixel),address) :
    {read_value(_) == MatrixSpec.patch(problem,matrix_pixel(evaluation),index) : F32}
  Equal.cong(Array<F32> & F32,F32,read_value,
    ScalarConv.direct_input(~Array<F32>,~Storage.read_or(F32,0.0),Inputs.input_owner(problem,inputs),context(problem,pixel),address),
    (Inputs.input_owner(problem,inputs),MatrixSpec.patch(problem,matrix_pixel(evaluation),index)),input_read_matches(evaluation,index,bounds))

def weight_matches_matrix(+evaluation: Evaluation,+index: MatrixSpec.Reduction,bounds: SampleBounds(evaluation,index)) ->
  {matrix_weight()(matrix_source(evaluation),Unit{},index) == MatrixSpec.weight(problem(evaluation),output_channel(evaluation),index) : F32}:
  Evaluation{limits,+problem,+inputs,+pixel,valid,supported} = evaluation
  MatrixSpec.KIndex{+channel,+row,+column} = index
  +address = U32.from_nat(Coordinates.reduction(channel,row,column,Guards.kernel(problem)))
  %ExecutionStorage.direct_weight_read(Inputs.weight_tree(problem,inputs),context(problem,pixel),address) :
    {read_value(_) == MatrixSpec.weight(problem,output_channel(evaluation),index) : F32}
  Equal.cong(Array<F32> & F32,F32,read_value,
    ScalarConv.direct_weight(~Array<F32>,~Storage.read_or(F32,0.0),Inputs.weight_owner(problem,inputs),context(problem,pixel),address),
    (Inputs.weight_owner(problem,inputs),MatrixSpec.weight(problem,output_channel(evaluation),index)),weight_read_matches(evaluation,index,bounds))

def samples_match(+evaluation: Evaluation,+index: MatrixSpec.Reduction,+bounds: SampleBounds(evaluation,index)) -> SampleAgreement(evaluation,index):
  Shape.Both{input_matches_patch(evaluation,index,bounds),weight_matches_matrix(evaluation,index,bounds)}

def ordered_dot_matches_matrix(+evaluation: Evaluation,+indices: List<&2,MatrixSpec.Reduction>,+initial: F32,
  bounds: ReductionBounds.AllBounds(problem(evaluation),indices)) ->
  {OrderedMatrix.dot(~Unit,~MatrixSpec.Reduction,~Unit,~ReindexedSource(),~matrix_input(),~matrix_weight(),indices,Unit{},Unit{},matrix_source(evaluation),initial) ==
    OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,indices,output_channel(evaluation),matrix_pixel(evaluation),problem(evaluation),initial) : F32}:
  Ordered.dot_congruence(~MatrixSpec.Reduction,~Unit,~Unit,~ReindexedSource(),~Nat,~MatrixSpec.Pixel,~ConvSpec.Problem,
    ~matrix_input(),~matrix_weight(),~MatrixSpec.patch,~MatrixSpec.weight,indices,Unit{},Unit{},matrix_source(evaluation),output_channel(evaluation),matrix_pixel(evaluation),problem(evaluation),initial,
    Members.transform_all(~MatrixSpec.Reduction,~Evaluation,~SampleBounds,~SampleAgreement,~samples_match,indices,evaluation,bounds))

def accumulator(evaluation: Evaluation,initial: F32) -> F32:
  Evaluation{limits,+problem,+inputs,pixel,valid,supported} = evaluation
  OutputLaws.job_accumulator(~F32,~F32,~F32,~F32,~ScalarConv.DirectContext,~F32,~ExecutionStorage.direct_input_value,~ExecutionStorage.direct_weight_value,~ScalarConv.direct_update,
    OutputTraversal.Job{context(problem,pixel),Coordinates.inner(problem),0,0,initial},
    OutputLaws.Sources{Inputs.input_tree(problem,inputs),Inputs.weight_tree(problem,inputs),Inputs.bias_tree(problem,inputs)})

def accumulator_matches_matrix(+evaluation: Evaluation,+initial: F32) ->
  {accumulator(evaluation,initial) == OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,
      MatrixSpec.kchannels(Guards.input_channel_count(problem(evaluation)),0n,Guards.kernel(problem(evaluation))),output_channel(evaluation),matrix_pixel(evaluation),problem(evaluation),initial) : F32}:
  Evaluation{limits,+problem,+inputs,+pixel,valid,supported} = evaluation
  +indices = MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem))
  +layout = {MatrixOrder.TapLayout{Guards.kernel(problem),0n,0n} : MatrixOrder.TapLayout}
  %Equal.sym(F32,accumulator(evaluation,initial),
    OrderedMatrix.dot(~Unit,~ReductionLaws.AddressPair,~Unit,~NativeSource(),~input_at(),~weight_at(),ReductionLaws.addresses(Coordinates.inner(problem),0,0),Unit{},Unit{},native_source(evaluation),initial),
    ExecutionStorage.direct_reduction_matches_dot(Coordinates.inner(problem),Reduction.State{Inputs.input_tree(problem,inputs),Inputs.weight_tree(problem,inputs),0,0,context(problem,pixel),initial})) :
    {_ == OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,indices,output_channel(evaluation),matrix_pixel(evaluation),problem,initial) : F32}
  %Equal.sym(List<&2,ReductionLaws.AddressPair>,ReductionLaws.addresses(Coordinates.inner(problem),0,0),MatrixOrder.matrix_taps(indices,Guards.kernel(problem),0n,0n),
    MatrixOrder.kernel_taps_follow_matrix_reduction(Guards.input_channel_count(problem),Guards.kernel(problem),0n,0n)) :
    {OrderedMatrix.dot(~Unit,~ReductionLaws.AddressPair,~Unit,~NativeSource(),~input_at(),~weight_at(),_,Unit{},Unit{},native_source(evaluation),initial) ==
      OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,indices,output_channel(evaluation),matrix_pixel(evaluation),problem,initial) : F32}
  %Equal.sym(F32,OrderedMatrix.dot(~Unit,~ReductionLaws.AddressPair,~Unit,~NativeSource(),~input_at(),~weight_at(),MatrixOrder.matrix_taps(indices,Guards.kernel(problem),0n,0n),Unit{},Unit{},native_source(evaluation),initial),
    OrderedMatrix.dot(~Unit,~MatrixSpec.Reduction,~Unit,~ReindexedSource(),~matrix_input(),~matrix_weight(),indices,Unit{},Unit{},matrix_source(evaluation),initial),
    Ordered.dot_reindexed(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~Unit,~Unit,~NativeSource(),~MatrixOrder.TapLayout,~MatrixOrder.mapped_tap,~input_at(),~weight_at(),
      indices,layout,Unit{},Unit{},native_source(evaluation),initial)) :
    {_ == OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,indices,output_channel(evaluation),matrix_pixel(evaluation),problem,initial) : F32}
  ordered_dot_matches_matrix(evaluation,indices,initial,ReductionBounds.matrix_reduction_in_bounds(problem))

def sources(evaluation: Evaluation) -> OutputLaws.Sources<F32,F32>:
  Evaluation{limits,+problem,+inputs,pixel,valid,supported} = evaluation
  OutputLaws.Sources{Inputs.input_tree(problem,inputs),Inputs.weight_tree(problem,inputs),Inputs.bias_tree(problem,inputs)}

def value_from_coordinates(+evaluation: Evaluation,decoded: OutputCoordinates.Decoded<problem(evaluation),output_pixel(evaluation)>) ->
  {ExecutionStorage.direct_value(ConvShape.from_problem(problem(evaluation)),U32.from_nat(OutputCoordinates.index(problem(evaluation),output_pixel(evaluation))),sources(evaluation)) ==
    (accumulator(evaluation,0.0) + ExecutionStorage.bounded_value(Inputs.bias_tree(problem(evaluation),input_contract(evaluation)),U32.from_nat(output_channel(evaluation)),U32.from_nat(Guards.output_channel_count(problem(evaluation)))) : F32) : F32}:
  Evaluation{+limits,+problem,+inputs,+pixel,+valid,+supported} = evaluation
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+channels,+outputs,+height,+width,+kernel,+stride,+padding} = shape
  OutputCoordinates.OutputPixel{+channel,+row,+column,channel_inside,row_inside,column_inside} = pixel
  OutputCoordinates.Decoded{channel_exact,row_exact,column_exact} = decoded
  +address = U32.from_nat(OutputCoordinates.index(problem,pixel))
  +spatial = (U32.from_nat(ConvSpec.out_height(problem)) * U32.from_nat(ConvSpec.out_width(problem)) : U32)
  +wanted = (accumulator(evaluation,0.0) + ExecutionStorage.bounded_value(Inputs.bias_tree(problem,inputs),U32.from_nat(channel),U32.from_nat(outputs)) : F32)
  %Equal.sym(U32,(address / spatial : U32),U32.from_nat(channel),channel_exact) :
    {(OutputLaws.job_accumulator(~F32,~F32,~F32,~F32,~ScalarConv.DirectContext,~F32,~ExecutionStorage.direct_input_value,~ExecutionStorage.direct_weight_value,~ScalarConv.direct_update,
      OutputTraversal.Job{ScalarConv.DirectContext{ConvShape.from_problem(problem),address,_,(address % spatial / U32.from_nat(ConvSpec.out_width(problem)) : U32),(address % U32.from_nat(ConvSpec.out_width(problem)) : U32)},
        U32.to_nat(Coordinates.native_inner(problem)),0,0,0.0},sources(evaluation)) + ExecutionStorage.bounded_value(Inputs.bias_tree(problem,inputs),_,U32.from_nat(outputs)) : F32) == wanted : F32}
  %Equal.sym(U32,(address % spatial / U32.from_nat(ConvSpec.out_width(problem)) : U32),U32.from_nat(row),row_exact) :
    {(OutputLaws.job_accumulator(~F32,~F32,~F32,~F32,~ScalarConv.DirectContext,~F32,~ExecutionStorage.direct_input_value,~ExecutionStorage.direct_weight_value,~ScalarConv.direct_update,
      OutputTraversal.Job{ScalarConv.DirectContext{ConvShape.from_problem(problem),address,U32.from_nat(channel),_,(address % U32.from_nat(ConvSpec.out_width(problem)) : U32)},
        U32.to_nat(Coordinates.native_inner(problem)),0,0,0.0},sources(evaluation)) + ExecutionStorage.bounded_value(Inputs.bias_tree(problem,inputs),U32.from_nat(channel),U32.from_nat(outputs)) : F32) == wanted : F32}
  %Equal.sym(U32,(address % U32.from_nat(ConvSpec.out_width(problem)) : U32),U32.from_nat(column),column_exact) :
    {(OutputLaws.job_accumulator(~F32,~F32,~F32,~F32,~ScalarConv.DirectContext,~F32,~ExecutionStorage.direct_input_value,~ExecutionStorage.direct_weight_value,~ScalarConv.direct_update,
      OutputTraversal.Job{ScalarConv.DirectContext{ConvShape.from_problem(problem),address,U32.from_nat(channel),U32.from_nat(row),_},
        U32.to_nat(Coordinates.native_inner(problem)),0,0,0.0},sources(evaluation)) + ExecutionStorage.bounded_value(Inputs.bias_tree(problem,inputs),U32.from_nat(channel),U32.from_nat(outputs)) : F32) == wanted : F32}
  %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(problem)),Coordinates.inner(problem),Coordinates.inner_exact(limits,problem,Guards.valid(problem,valid),Support.scalar(limits,problem,supported))) :
    {(OutputLaws.job_accumulator(~F32,~F32,~F32,~F32,~ScalarConv.DirectContext,~F32,~ExecutionStorage.direct_input_value,~ExecutionStorage.direct_weight_value,~ScalarConv.direct_update,
      OutputTraversal.Job{context(problem,pixel),_,0,0,0.0},sources(evaluation)) + ExecutionStorage.bounded_value(Inputs.bias_tree(problem,inputs),U32.from_nat(channel),U32.from_nat(outputs)) : F32) == wanted : F32}
  {==}

def bias_matches_spec(+evaluation: Evaluation) ->
  {ExecutionStorage.bounded_value(Inputs.bias_tree(problem(evaluation),input_contract(evaluation)),U32.from_nat(output_channel(evaluation)),U32.from_nat(Guards.output_channel_count(problem(evaluation)))) ==
    ConvSpec.bias_at(problem(evaluation),output_channel(evaluation)) : F32}:
  Evaluation{+limits,+problem,+inputs,pixel,+valid,+supported} = evaluation
  OutputCoordinates.OutputPixel{+channel,row,column,channel_inside,row_inside,column_inside} = pixel
  %StorageLaws.bounded_read(~F32,0.0,Inputs.bias_tree(problem,inputs),U32.from_nat(channel),U32.from_nat(Guards.output_channel_count(problem))) :
    {read_value(_) == ConvSpec.bias_at(problem,channel) : F32}
  Equal.cong(Array<F32> & F32,F32,read_value,
    Storage.read_or(F32,0.0,Inputs.bias_owner(problem,inputs),U32.from_nat(channel),U32.from_nat(Guards.output_channel_count(problem))),
    (Inputs.bias_owner(problem,inputs),ConvSpec.bias_at(problem,channel)),ConvolutionReads.bias_matches_spec(limits,problem,inputs,channel,valid,supported,channel_inside))

def value_matches_matrix(+evaluation: Evaluation) ->
  {ExecutionStorage.direct_value(ConvShape.from_problem(problem(evaluation)),U32.from_nat(OutputCoordinates.index(problem(evaluation),output_pixel(evaluation))),sources(evaluation)) ==
    (OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,
      MatrixSpec.kchannels(Guards.input_channel_count(problem(evaluation)),0n,Guards.kernel(problem(evaluation))),output_channel(evaluation),matrix_pixel(evaluation),problem(evaluation),0.0) +
      ConvSpec.bias_at(problem(evaluation),output_channel(evaluation)) : F32) : F32}:
  Evaluation{+limits,+problem,inputs,+pixel,+valid,+supported} = evaluation
  +value = ExecutionStorage.direct_value(ConvShape.from_problem(problem),U32.from_nat(OutputCoordinates.index(problem,pixel)),sources(evaluation))
  %accumulator_matches_matrix(evaluation,0.0) : {value == (_ + ConvSpec.bias_at(problem,output_channel(evaluation)) : F32) : F32}
  %bias_matches_spec(evaluation) : {value == (accumulator(evaluation,0.0) + _ : F32) : F32}
  value_from_coordinates(evaluation,OutputCoordinates.decode(limits,problem,pixel,valid,supported))
