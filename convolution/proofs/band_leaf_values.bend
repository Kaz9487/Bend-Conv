# Values of the convolution band leaf on active and inactive windows.
import Base
import ../../lib/proofs/tensor_band_window.bend as TensorBandWindow
import ./band_leaf_execution.bend as LeafExecution
import ./band_window_contents.bend as WindowContents
import ./band_convolution_values.bend as BandConvolutionValues
import ../specification.bend as ConvSpec
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/tensor_band.bend as Bands
import ../inference.bend as NativeConv
import ./band_window_leaf.bend as WindowLeaf
import ./convolution_guard_facts.bend as Guards
import ./reduction_coordinate_bounds.bend as Coordinates
import ./supported_padded_coordinates.bend as Padded
import ./window_samples.bend as Samples
import ../../lib/kernel_band_pipeline.bend as BandPipeline
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_halo.bend as BandHalo
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/tensor_band_window.bend as BandWindow
import ./concrete_support_guard.bend as Support
import ./output_values.bend as Values
import ./window_band_values.bend as WindowBandValues
import ./window_positions.bend as Positions
import ./window_problem.bend as WindowProblem
import ../../lib/kernel_packing_buffer.bend as PackingBuffer
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_extrema.bend as Extrema
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/tensor_band_coordinates.bend as BandCoordinates
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor.bend as Tensors

# The band leaf as the runtime builds it: its reduction length, packing call
# and weight/bias trees are the window problem's, so each real output
# position holds the activated original matrix value.
# The runtime reduction length is the problem's native one.
def inner_word(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {U32.from_nat(Coordinates.inner(problem)) == Coordinates.native_inner(problem) : U32}:
  +native = Coordinates.native_inner(problem)
  Equal.trans(U32,U32.from_nat(Coordinates.inner(problem)),U32.from_nat(U32.to_nat(native)),native,
    Equal.cong(Nat,U32,value => U32.from_nat(value),Coordinates.inner(problem),U32.to_nat(native),
      Equal.sym(Nat,U32.to_nat(native),Coordinates.inner(problem),
        Coordinates.inner_exact(limits,problem,Guards.valid(problem,valid),Padded.scalar(limits,problem,Support.fast(limits,problem,supported))))),
    Arithmetic.native_roundtrip(native))

# The leaf tree with its reduction length, window height and parameter trees open.
def tree_at(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+positions: U32,+start: U32,+width: U32,+kernel: U32,+stride: U32,
  +padding: U32,+output_width: U32,+window: Tree.Tree<F32>,+inner: U32,+height: U32,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  LeafExecution.leaf_tree(~custom,activation,rows,span,inner,positions,Packing.Call{U32.to_nat((span / 8 * inner : U32)),start,height,width,kernel,stride,padding,output_width,inner},
    (span / 8 * inner : U32),window,weights,bias)

def runtime_value(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.weights(problem)),Guards.weights(problem)>,
  +bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.bias(problem)),Guards.bias(problem)>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset: U32,+positions: U32,
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(positions)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},
  +tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(positions)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +weight_tree: Tree.Tree<Tile.Tile8>,
  +weight_exact: {weight_tree == WeightsLaws.run(U32.from_nat(Guards.output_channel_count(problem)),Coordinates.native_inner(problem),
    StorageLaws.sequence_storage(F32,ConvSpec.length(Guards.weights(problem)),Guards.weights(problem),weight_evidence)) : Tree.Tree<Tile.Tile8>},
  +bias_tree: Tree.Tree<F32>,
  +bias_exact: {bias_tree == StorageLaws.sequence_storage(F32,ConvSpec.length(Guards.bias(problem)),Guards.bias(problem),bias_evidence) : Tree.Tree<F32>},
  +channel: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(positions)) == True{} : Bool}) ->
  {Tree.read(F32,tree_at(~custom,activation,U32.from_nat(Guards.output_channel_count(problem)),WindowLeaf.padded(positions),positions,
      KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first),
      U32.from_nat(Guards.width(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),U32.from_nat(Guards.pad(problem)),
      U32.from_nat(ConvSpec.out_width(problem)),tree,U32.from_nat(Coordinates.inner(problem)),rows,weight_tree,bias_tree),
      U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),column))) ==
    LeafExecution.activation_value(~custom,activation,
      Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)))) : F32}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +span = WindowLeaf.padded(positions)
  +start = KernelWindow.start(offset,U32.from_nat(stride),U32.from_nat(ConvSpec.out_width(problem)),first)
  +position = U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),column))
  +native = Coordinates.native_inner(problem)
  +reordered = WeightsLaws.run(U32.from_nat(co),native,StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence))
  +stored_bias = StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence)
  +runtime = tree_at(~custom,activation,U32.from_nat(co),span,positions,start,U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
    U32.from_nat(ConvSpec.out_width(problem)),tree,U32.from_nat(Coordinates.inner(problem)),rows,weight_tree,bias_tree)
  +modeled = tree_at(~custom,activation,U32.from_nat(co),span,positions,start,U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
    U32.from_nat(ConvSpec.out_width(problem)),tree,native,U32.from_nat(U32.to_nat(rows)),reordered,stored_bias)
  +step_inner = Equal.cong(U32,Tree.Tree<F32>,
    value => tree_at(~custom,activation,U32.from_nat(co),span,positions,start,U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
      U32.from_nat(ConvSpec.out_width(problem)),tree,value,rows,weight_tree,bias_tree),
    U32.from_nat(Coordinates.inner(problem)),native,inner_word(limits,problem,valid,supported))
  +step_height = Equal.cong(U32,Tree.Tree<F32>,
    value => tree_at(~custom,activation,U32.from_nat(co),span,positions,start,U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
      U32.from_nat(ConvSpec.out_width(problem)),tree,native,value,weight_tree,bias_tree),
    rows,U32.from_nat(U32.to_nat(rows)),Equal.sym(U32,U32.from_nat(U32.to_nat(rows)),rows,Arithmetic.native_roundtrip(rows)))
  +step_weights = Equal.cong(Tree.Tree<Tile.Tile8>,Tree.Tree<F32>,
    value => tree_at(~custom,activation,U32.from_nat(co),span,positions,start,U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
      U32.from_nat(ConvSpec.out_width(problem)),tree,native,U32.from_nat(U32.to_nat(rows)),value,bias_tree),
    weight_tree,reordered,weight_exact)
  +step_bias = Equal.cong(Tree.Tree<F32>,Tree.Tree<F32>,
    value => tree_at(~custom,activation,U32.from_nat(co),span,positions,start,U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
      U32.from_nat(ConvSpec.out_width(problem)),tree,native,U32.from_nat(U32.to_nat(rows)),reordered,value),
    bias_tree,stored_bias,bias_exact)
  +trees = Equal.trans(Tree.Tree<F32>,runtime,
    tree_at(~custom,activation,U32.from_nat(co),span,positions,start,U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
      U32.from_nat(ConvSpec.out_width(problem)),tree,native,rows,weight_tree,bias_tree),modeled,step_inner,
    Equal.trans(Tree.Tree<F32>,
      tree_at(~custom,activation,U32.from_nat(co),span,positions,start,U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
        U32.from_nat(ConvSpec.out_width(problem)),tree,native,rows,weight_tree,bias_tree),
      tree_at(~custom,activation,U32.from_nat(co),span,positions,start,U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
        U32.from_nat(ConvSpec.out_width(problem)),tree,native,U32.from_nat(U32.to_nat(rows)),weight_tree,bias_tree),modeled,step_height,
      Equal.trans(Tree.Tree<F32>,
        tree_at(~custom,activation,U32.from_nat(co),span,positions,start,U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
          U32.from_nat(ConvSpec.out_width(problem)),tree,native,U32.from_nat(U32.to_nat(rows)),weight_tree,bias_tree),
        tree_at(~custom,activation,U32.from_nat(co),span,positions,start,U32.from_nat(image_width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
          U32.from_nat(ConvSpec.out_width(problem)),tree,native,U32.from_nat(U32.to_nat(rows)),reordered,bias_tree),modeled,step_weights,step_bias)))
  Equal.trans(F32,Tree.read(F32,runtime,position),Tree.read(F32,modeled,position),
    LeafExecution.activation_value(~custom,activation,
      Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(offset),column),ConvSpec.out_width(problem)))),
    Equal.cong(Tree.Tree<F32>,F32,values => Tree.read(F32,values,position),runtime,modeled,trees),
    WindowLeaf.leaf_value(~custom,activation,limits,problem,weight_evidence,bias_evidence,valid,supported,offset,positions,grid,tree,first,rows,
      cover,contents,room,channel,column,channel_inside,column_inside))

# The output row a leaf position reads, as the window offset plus its column.
def output_position(+first: Nat,+row: Nat,+width: Nat,+column: Nat,+offset: U32,+index: Nat,
  +offset_exact: {U32.to_nat(offset) == Nat.mul(first,width) : Nat},+index_exact: {index == Nat.add(Nat.mul(row,width),column) : Nat}) ->
  {Nat.add(Nat.mul(Nat.add(first,row),width),column) == Nat.add(U32.to_nat(offset),index) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(first,row),width),column),Nat.add(Nat.add(Nat.mul(first,width),Nat.mul(row,width)),column),Nat.add(U32.to_nat(offset),index),
    Equal.cong(Nat,Nat,value => Nat.add(value,column),Nat.mul(Nat.add(first,row),width),Nat.add(Nat.mul(first,width),Nat.mul(row,width)),RowMajor.distrib(first,row,width)),
    Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(first,width),Nat.mul(row,width)),column),Nat.add(Nat.mul(first,width),index),Nat.add(U32.to_nat(offset),index),
      Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(first,width),Nat.mul(row,width)),column),Nat.add(Nat.mul(first,width),Nat.add(Nat.mul(row,width),column)),
        Nat.add(Nat.mul(first,width),index),Addition.associative(Nat.mul(first,width),Nat.mul(row,width),column),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(first,width),value),Nat.add(Nat.mul(row,width),column),index,Equal.sym(Nat,index,Nat.add(Nat.mul(row,width),column),index_exact))),
      Equal.cong(Nat,Nat,value => Nat.add(value,index),Nat.mul(first,width),U32.to_nat(offset),Equal.sym(Nat,U32.to_nat(offset),Nat.mul(first,width),offset_exact))))

# The output tensor facts a leaf needs about its rows.
type OutputFacts<-images: U32,-outputs: U32,-output_height: U32,-output_width: U32,-first: U32,-rows: U32,-capacity: U32,
  -image: Nat,-offset_rows: Nat,-co: Nat,-oh: Nat,-ow: Nat> is Data:
  OutputFacts{outputs_exact: {U32.to_nat(outputs) == co : Nat},height_exact: {U32.to_nat(output_height) == oh : Nat},width_exact: {U32.to_nat(output_width) == ow : Nat},
    first_exact: {U32.to_nat(first) == Nat.add(Nat.mul(image,U32.to_nat(output_height)),offset_rows) : Nat},
    image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},
    row_fits: {Nat.is_le(Nat.add(offset_rows,U32.to_nat(rows)),U32.to_nat(output_height)) == True{} : Bool},
    capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    total_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(outputs),Nat.mul(U32.to_nat(output_height),U32.to_nat(output_width)))),U32.to_nat(capacity)) == True{} : Bool}}

# The window facts runtime_value needs for one leaf.
type LeafWindow<-limits: Machine.Limits,-problem: ConvSpec.Problem,-offset: U32,-offset_rows: Nat,-positions: U32,-tree: Tree.Tree<F32>,-first: U32,-rows: U32> is Data:
  LeafWindow{offset_exact: {U32.to_nat(offset) == Nat.mul(offset_rows,ConvSpec.out_width(problem)) : Nat},
    grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(positions)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},
    cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(positions)>,
    contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
    room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}}

def leaf_tree(~custom: F32 -> F32,+activation: Activations.Activation,+ci: Nat,+co: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+output_width: Nat,
  +offset: U32,+positions: U32,+tree: Tree.Tree<F32>,+first: U32,+rows: U32,+weight_tree: Tree.Tree<Tile.Tile8>,+bias_tree: Tree.Tree<F32>) -> Tree.Tree<F32>:
  tree_at(~custom,activation,U32.from_nat(co),WindowLeaf.padded(positions),positions,KernelWindow.start(offset,U32.from_nat(stride),U32.from_nat(output_width),first),
    U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),U32.from_nat(output_width),tree,U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),rows,
    weight_tree,bias_tree)

# One cell of an active leaf is the dense output element of its pixel.
def active_cell(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+limits: Machine.Limits,+ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +valid: {ConvSpec.valid(BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},
  +weight_tree: Tree.Tree<Tile.Tile8>,
  +weight_exact: {weight_tree == WeightsLaws.run(U32.from_nat(co),Coordinates.native_inner(BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)),
    StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence)) : Tree.Tree<Tile.Tile8>},
  +bias_tree: Tree.Tree<F32>,+bias_exact: {bias_tree == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +offset: U32,+offset_rows: Nat,+positions: U32,+tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +window: LeafWindow<limits,BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image),offset,offset_rows,positions,tree,first,rows>,
  +images: U32,+outputs: U32,+output_height: U32,+output_width: U32,+output_first: U32,+output_rows: U32,+capacity: U32,
  +output: OutputFacts<images,outputs,output_height,output_width,output_first,output_rows,capacity,image,offset_rows,co,
    ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)>,
  +positions_exact: {U32.to_nat(positions) == Nat.mul(U32.to_nat(output_rows),ConvSpec.axis_size(width,kernel,stride,padding)) : Nat},
  +channel: Nat,+row: Nat,+line: Nat,+channel_inside: {Nat.is_lt(channel,U32.to_nat(outputs)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(output_rows)) == True{} : Bool},+line_inside: {Nat.is_lt(line,U32.to_nat(output_width)) == True{} : Bool}) ->
  BandValues.At(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),BandConvolutionValues.Context{logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},
    Bands.Geometry{images,outputs,output_height,output_width},output_first,output_rows,
    leaf_tree(~custom,activation,ci,co,width,kernel,stride,padding,ConvSpec.axis_size(width,kernel,stride,padding),offset,positions,tree,first,rows,weight_tree,bias_tree),Nat.mul(co,U32.to_nat(positions)),channel,row,line):
  LeafWindow{+offset_exact,+grid,+cover,+contents,+room} = window
  OutputFacts{+outputs_exact,+height_exact,+width_exact,+first_exact,+image_inside,+row_fits,+capacity_ok,+total_room} = output
  +problem = BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)
  +oh = ConvSpec.axis_size(height,kernel,stride,padding)
  +ow = ConvSpec.axis_size(width,kernel,stride,padding)
  +plane = Nat.mul(oh,ow)
  +geometry = {Bands.Geometry{images,outputs,output_height,output_width} : Bands.Geometry}
  +runtime = leaf_tree(~custom,activation,ci,co,width,kernel,stride,padding,ow,offset,positions,tree,first,rows,weight_tree,bias_tree)
  +count = U32.to_nat(positions)
  +output_count = U32.to_nat(output_rows)
  +column = Nat.add(Nat.mul(row,ow),line)
  +channel_small = {%outputs_exact : {Nat.is_lt(channel,_) == True{} : Bool} channel_inside : {Nat.is_lt(channel,co) == True{} : Bool}}
  +line_small = {%width_exact : {Nat.is_lt(line,_) == True{} : Bool} line_inside : {Nat.is_lt(line,ow) == True{} : Bool}}
  +column_inside = {
    %Equal.sym(Nat,count,Nat.mul(output_count,ow),positions_exact) : {Nat.is_lt(column,_) == True{} : Bool}
    RowMajor.radix(row,output_count,line,ow,row_inside,line_small) : {Nat.is_lt(column,count) == True{} : Bool}}
  +image_row = Bounds.nat_trans(Nat.add(offset_rows,row),Nat.add(offset_rows,output_count),U32.to_nat(output_height),Order.offset_lt(offset_rows,row,output_count,row_inside),row_fits)
  +image_row_small = {%height_exact : {Nat.is_lt(Nat.add(offset_rows,row),_) == True{} : Bool} image_row : {Nat.is_lt(Nat.add(offset_rows,row),oh) == True{} : Bool}}
  +plane_inside = RowMajor.radix(Nat.add(offset_rows,row),oh,line,ow,image_row_small,line_small)
  +pixel = BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(output_first),row),line)
  +native = BandValues.point(geometry,image,channel,Nat.add(offset_rows,row),line)
  +located = BandValues.located_pixel(geometry,image,channel,Nat.add(offset_rows,row),line,Nat.add(U32.to_nat(output_first),row),image_row,
    %Equal.sym(Nat,U32.to_nat(output_first),Nat.add(Nat.mul(image,U32.to_nat(output_height)),offset_rows),first_exact) : {Nat.add(Nat.mul(image,U32.to_nat(output_height)),Nat.add(offset_rows,row)) == Nat.add(_,row) : Nat}
    Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(image,U32.to_nat(output_height)),offset_rows),row),Nat.add(Nat.mul(image,U32.to_nat(output_height)),Nat.add(offset_rows,row)),Addition.associative(Nat.mul(image,U32.to_nat(output_height)),offset_rows,row)))
  +total = Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(outputs)),Nat.mul(U32.to_nat(output_height),U32.to_nat(output_width)))
  +native_inside = Bounds.nat_trans(native,total,U32.to_nat(capacity),
    RowMajor.in_bounds(Nat.add(Nat.mul(image,U32.to_nat(outputs)),channel),Nat.mul(U32.to_nat(images),U32.to_nat(outputs)),U32.to_nat(output_height),U32.to_nat(output_width),Nat.add(offset_rows,row),line,
      RowMajor.radix(image,U32.to_nat(images),channel,U32.to_nat(outputs),image_inside,channel_inside),image_row,line_inside),
    %Equal.sym(Nat,total,Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(outputs),Nat.mul(U32.to_nat(output_height),U32.to_nat(output_width)))),RowMajor.assoc(U32.to_nat(images),U32.to_nat(outputs),Nat.mul(U32.to_nat(output_height),U32.to_nat(output_width)))) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} total_room)
  +position_in_plane = Nat.add(Nat.mul(Nat.add(offset_rows,row),ow),line)
  +flat = Nat.add(Nat.mul(image,Nat.mul(co,plane)),Nat.add(Nat.mul(channel,plane),position_in_plane))
  +native_exact = {
    %outputs_exact : {native == RowMajor.index(Nat.add(Nat.mul(image,_),channel),oh,ow,Nat.add(offset_rows,row),line) : Nat}
    %height_exact : {native == RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(outputs)),channel),_,ow,Nat.add(offset_rows,row),line) : Nat}
    %width_exact : {native == RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(outputs)),channel),U32.to_nat(output_height),_,Nat.add(offset_rows,row),line) : Nat}
    {==} : {native == RowMajor.index(Nat.add(Nat.mul(image,co),channel),oh,ow,Nat.add(offset_rows,row),line) : Nat}}
  +address = U32.from_nat(pixel)
  +address_exact = {
    %Equal.sym(Nat,pixel,native,located) : {U32.to_nat(U32.from_nat(_)) == flat : Nat}
    Equal.trans(Nat,U32.to_nat(U32.from_nat(native)),native,flat,Bounds.roundtrip(native,capacity,capacity_ok,Bounds.strict_le(native,U32.to_nat(capacity),native_inside)),
      Equal.trans(Nat,native,RowMajor.index(Nat.add(Nat.mul(image,co),channel),oh,ow,Nat.add(offset_rows,row),line),flat,native_exact,
        WindowContents.image_index(image,co,plane,channel,Nat.mul(Nat.add(offset_rows,row),ow),line))) :
    {U32.to_nat(address) == flat : Nat}}
  +shifted = output_position(offset_rows,row,ow,line,offset,column,offset_exact,{==})
  +computed = runtime_value(~custom,activation,limits,problem,weight_evidence,bias_evidence,valid,supported,offset,positions,grid,tree,first,rows,
    cover,contents,room,weight_tree,weight_exact,bias_tree,bias_exact,channel,column,channel_small,column_inside)
  +placed = Equal.cong(Nat,F32,value => BandConvolutionValues.at_position(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,image,channel,value),
    Nat.add(U32.to_nat(offset),column),position_in_plane,Equal.sym(Nat,position_in_plane,Nat.add(U32.to_nat(offset),column),shifted))
  +decoded = Equal.sym(F32,BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,flat),
    BandConvolutionValues.at_position(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,image,channel,position_in_plane),
    BandConvolutionValues.decoded(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,image,channel,position_in_plane,channel_small,plane_inside))
  +addressed = Equal.cong(Nat,F32,value => BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,value),
    flat,U32.to_nat(address),Equal.sym(Nat,U32.to_nat(address),flat,address_exact))
  +valued = Equal.trans(F32,Tree.read(F32,runtime,U32.from_nat(Nat.add(Nat.mul(channel,count),column))),
    BandConvolutionValues.at_position(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,image,channel,Nat.add(U32.to_nat(offset),column)),
    BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat(address)),computed,
    Equal.trans(F32,BandConvolutionValues.at_position(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,image,channel,Nat.add(U32.to_nat(offset),column)),
      BandConvolutionValues.at_position(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,image,channel,position_in_plane),
      BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat(address)),placed,
      Equal.trans(F32,BandConvolutionValues.at_position(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,image,channel,position_in_plane),
        BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,flat),
        BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat(address)),decoded,addressed)))
  +inside = {
    %Equal.sym(Nat,count,Nat.mul(output_count,ow),positions_exact) : {Nat.is_lt(BandValues.position(geometry,output_rows,channel,row,line),Nat.mul(co,_)) == True{} : Bool}
    %outputs_exact : {Nat.is_lt(BandValues.position(geometry,output_rows,channel,row,line),Nat.mul(_,Nat.mul(output_count,ow))) == True{} : Bool}
    %width_exact : {Nat.is_lt(BandValues.position(geometry,output_rows,channel,row,line),Nat.mul(U32.to_nat(outputs),Nat.mul(output_count,_))) == True{} : Bool}
    BandValues.position_inside(geometry,output_rows,channel,row,line,channel_inside,row_inside,line_inside) :
    {Nat.is_lt(BandValues.position(geometry,output_rows,channel,row,line),Nat.mul(co,count)) == True{} : Bool}}
  +read = {
    %Equal.sym(Nat,U32.to_nat(output_width),ow,width_exact) : {Tree.read(F32,runtime,U32.from_nat(RowMajor.index(channel,output_count,_,row,line))) == BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat(address)) : F32}
    %positions_exact : {Tree.read(F32,runtime,U32.from_nat(Nat.add(Nat.mul(channel,_),column))) == BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat(address)) : F32}
    valued : {Tree.read(F32,runtime,U32.from_nat(BandValues.position(geometry,output_rows,channel,row,line))) == BandConvolutionValues.value_at(~custom,~Logical,~original,logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat(address)) : F32}}
  Shape.Both{inside,read}

# The decoded window of an active leaf satisfies runtime_value's window facts.
def leaf_window(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,
  +ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,
  +valid: {ConvSpec.valid(BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},
  +padding_small: {Nat.is_lt(padding,kernel) == True{} : Bool},
  +output_first: Nat,+output_end: Nat,+ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},
  +bounded: {Nat.is_le(output_end,ConvSpec.axis_size(height,kernel,stride,padding)) == True{} : Bool},
  +offset: U32,+offset_exact: {U32.to_nat(offset) == Nat.mul(output_first,ConvSpec.axis_size(width,kernel,stride,padding)) : Nat},
  +positions: U32,+positions_exact: {U32.to_nat(positions) == Nat.mul(Nat.sub(output_end,output_first),ConvSpec.axis_size(width,kernel,stride,padding)) : Nat},
  +local: U32,+local_exact: {U32.to_nat(local) == WindowContents.window_first(height,output_first,stride,padding) : Nat},
  +rows: U32,+rows_exact: {U32.to_nat(rows) == Nat.sub(WindowContents.window_end(height,output_end,stride,kernel,padding),WindowContents.window_first(height,output_first,stride,padding)) : Nat},
  +images: U32,+channels_word: U32,+height_word: U32,+width_word: U32,
  +channels_exact: {U32.to_nat(channels_word) == ci : Nat},+height_exact: {U32.to_nat(height_word) == height : Nat},+width_exact: {U32.to_nat(width_word) == width : Nat},
  +first: U32,+window_first_exact: {U32.to_nat(first) == Nat.add(Nat.mul(image,U32.to_nat(height_word)),U32.to_nat(local)) : Nat},
  +window_fits: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(rows)),U32.to_nat(height_word)) == True{} : Bool},
  +tree: Tree.Tree<F32>,+length: Nat,+length_exact: {length == Nat.mul(U32.to_nat(channels_word),Nat.mul(U32.to_nat(rows),U32.to_nat(width_word))) : Nat},
  +values: BandValues.Leaf(~F32,~Logical,~original,logical,Bands.Geometry{images,channels_word,height_word,width_word},first,rows,tree,length),
  +room: {Nat.is_le(length,U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}) ->
  LeafWindow<limits,BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image),offset,output_first,positions,tree,local,rows>:
  +problem = BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)
  +oh = ConvSpec.axis_size(height,kernel,stride,padding)
  +ow = ConvSpec.axis_size(width,kernel,stride,padding)
  +start = WindowContents.window_first(height,output_first,stride,padding)
  +count = Nat.sub(WindowContents.window_end(height,output_end,stride,kernel,padding),start)
  +covered = {
    %Equal.sym(Nat,U32.to_nat(local),start,local_exact) : Positions.Cover<problem,_,U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(positions)>
    %Equal.sym(Nat,U32.to_nat(rows),count,rows_exact) : Positions.Cover<problem,start,_,U32.to_nat(offset),U32.to_nat(positions)>
    %Equal.sym(Nat,U32.to_nat(offset),Nat.mul(output_first,ow),offset_exact) : Positions.Cover<problem,start,count,_,U32.to_nat(positions)>
    %Equal.sym(Nat,U32.to_nat(positions),Nat.mul(Nat.sub(output_end,output_first),ow),positions_exact) : Positions.Cover<problem,start,count,Nat.mul(output_first,ow),_>
    WindowContents.cover(problem,valid,padding_small,output_first,output_end,ordered,bounded) :
    Positions.Cover<problem,U32.to_nat(local),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(positions)>}
  +grid = {
    %Equal.sym(Nat,U32.to_nat(offset),Nat.mul(output_first,ow),offset_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(positions)),Nat.mul(oh,ow)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(positions),Nat.mul(Nat.sub(output_end,output_first),ow),positions_exact) : {Nat.is_le(Nat.add(Nat.mul(output_first,ow),_),Nat.mul(oh,ow)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.mul(output_first,ow),Nat.mul(Nat.sub(output_end,output_first),ow)),Nat.mul(output_end,ow),
      WindowContents.positions_end(output_first,output_end,ow,Bounds.strict_le(output_first,output_end,ordered))) : {Nat.is_le(_,Nat.mul(oh,ow)) == True{} : Bool}
    RowMajor.mul_le(output_end,oh,ow,bounded) :
    {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(positions)),Nat.mul(oh,ow)) == True{} : Bool}}
  +band = {
    %channels_exact : WindowBandValues.Contents(Nat.mul(_,Nat.mul(U32.to_nat(rows),width)),WindowContents.image_band(~Logical,~original,logical,_,height,width,image,U32.to_nat(local),U32.to_nat(rows)),tree)
    %height_exact : WindowBandValues.Contents(Nat.mul(U32.to_nat(channels_word),Nat.mul(U32.to_nat(rows),width)),
      WindowContents.image_band(~Logical,~original,logical,U32.to_nat(channels_word),_,width,image,U32.to_nat(local),U32.to_nat(rows)),tree)
    %width_exact : WindowBandValues.Contents(Nat.mul(U32.to_nat(channels_word),Nat.mul(U32.to_nat(rows),_)),
      WindowContents.image_band(~Logical,~original,logical,U32.to_nat(channels_word),U32.to_nat(height_word),_,image,U32.to_nat(local),U32.to_nat(rows)),tree)
    WindowContents.contents(~Logical,~original,logical,images,channels_word,height_word,width_word,image,U32.to_nat(local),first,rows,window_first_exact,window_fits,tree,length,length_exact,values) :
    WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(local),U32.to_nat(rows)),tree)}
  +tree_room = {
    %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}
    room : {Nat.is_le(Nat.mul(U32.to_nat(channels_word),Nat.mul(U32.to_nat(rows),U32.to_nat(width_word))),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}}
  +window_room = {
    %channels_exact : {Nat.is_le(Nat.mul(_,Nat.mul(U32.to_nat(rows),width)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}
    %width_exact : {Nat.is_le(Nat.mul(U32.to_nat(channels_word),Nat.mul(U32.to_nat(rows),_)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}
    tree_room : {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}}
  LeafWindow{offset_exact,grid,covered,band,window_room}

# What an active leaf supplies to every cell of its output.
type ActiveCells<-Logical: Data,-original: Logical -> U32 -> F32> is Data:
  ActiveCells{logical: Logical,activation: Activations.Activation,limits: Machine.Limits,ci: Nat,co: Nat,height: Nat,width: Nat,kernel: Nat,stride: Nat,padding: Nat,weights: List<&2,F32>,bias: List<&2,F32>,image: Nat,
    weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
    valid: {ConvSpec.valid(BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},
    supported: {NativeConv.supports_packed_with_limits(limits,BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},
    weight_tree: Tree.Tree<Tile.Tile8>,
    weight_exact: {weight_tree == WeightsLaws.run(U32.from_nat(co),Coordinates.native_inner(BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)),
      StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence)) : Tree.Tree<Tile.Tile8>},
    bias_tree: Tree.Tree<F32>,bias_exact: {bias_tree == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
    offset: U32,offset_rows: Nat,positions: U32,tree: Tree.Tree<F32>,first: U32,rows: U32,
    window: LeafWindow<limits,BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image),offset,offset_rows,positions,tree,first,rows>,
    images: U32,outputs: U32,output_height: U32,output_width: U32,output_first: U32,output_rows: U32,capacity: U32,
    output: OutputFacts<images,outputs,output_height,output_width,output_first,output_rows,capacity,image,offset_rows,co,
      ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)>,
    positions_exact: {U32.to_nat(positions) == Nat.mul(U32.to_nat(output_rows),ConvSpec.axis_size(width,kernel,stride,padding)) : Nat}}

def active_geometry(~Logical: Data,~original: Logical -> U32 -> F32,evidence: ActiveCells<Logical,original>) -> Bands.Geometry:
  ActiveCells{logical,activation,limits,ci,co,height,width,kernel,stride,padding,weights,bias,image,weight_evidence,bias_evidence,valid,supported,weight_tree,weight_exact,bias_tree,bias_exact,offset,offset_rows,positions,tree,first,rows,window,images,outputs,output_height,output_width,output_first,output_rows,capacity,output,positions_exact} = evidence
  Bands.Geometry{images,outputs,output_height,output_width}

def active_rows(~Logical: Data,~original: Logical -> U32 -> F32,evidence: ActiveCells<Logical,original>) -> U32:
  ActiveCells{logical,activation,limits,ci,co,height,width,kernel,stride,padding,weights,bias,image,weight_evidence,bias_evidence,valid,supported,weight_tree,weight_exact,bias_tree,bias_exact,offset,offset_rows,positions,tree,first,rows,window,images,outputs,output_height,output_width,output_first,output_rows,capacity,output,positions_exact} = evidence
  output_rows

def ActiveCell(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+evidence: ActiveCells<Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  ActiveCells{+logical,+activation,+limits,+ci,+co,+height,+width,+kernel,+stride,+padding,+weights,+bias,+image,+weight_evidence,+bias_evidence,+valid,+supported,+weight_tree,+weight_exact,+bias_tree,+bias_exact,+offset,+offset_rows,+positions,+tree,+first,+rows,+window,+images,+outputs,+output_height,+output_width,+output_first,+output_rows,+capacity,+output,+positions_exact} = evidence
  BandValues.At(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),BandConvolutionValues.Context{logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},
    Bands.Geometry{images,outputs,output_height,output_width},output_first,output_rows,
    leaf_tree(~custom,activation,ci,co,width,kernel,stride,padding,ConvSpec.axis_size(width,kernel,stride,padding),offset,positions,tree,first,rows,weight_tree,bias_tree),Nat.mul(co,U32.to_nat(positions)),channel,row,column)

def active_leaf_cell(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+evidence: ActiveCells<Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(active_geometry(~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(active_rows(~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(active_geometry(~Logical,~original,evidence))) == True{} : Bool}) ->
  ActiveCell(~custom,~Logical,~original,evidence,channel,row,column):
  ActiveCells{+logical,+activation,+limits,+ci,+co,+height,+width,+kernel,+stride,+padding,+weights,+bias,+image,+weight_evidence,+bias_evidence,+valid,+supported,+weight_tree,+weight_exact,+bias_tree,+bias_exact,+offset,+offset_rows,+positions,+tree,+first,+rows,+window,+images,+outputs,+output_height,+output_width,+output_first,+output_rows,+capacity,+output,+positions_exact} = evidence
  active_cell(~custom,~Logical,~original,logical,activation,limits,ci,co,height,width,kernel,stride,padding,weights,bias,image,weight_evidence,bias_evidence,valid,supported,weight_tree,weight_exact,bias_tree,bias_exact,offset,offset_rows,positions,tree,first,rows,window,images,outputs,output_height,output_width,output_first,output_rows,capacity,output,positions_exact,channel,row,column,channel_inside,row_inside,column_inside)

# Every cell of an active leaf is the dense output element of its pixel.
def leaf_all(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+limits: Machine.Limits,+ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +valid: {ConvSpec.valid(BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},
  +weight_tree: Tree.Tree<Tile.Tile8>,
  +weight_exact: {weight_tree == WeightsLaws.run(U32.from_nat(co),Coordinates.native_inner(BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image)),
    StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence)) : Tree.Tree<Tile.Tile8>},
  +bias_tree: Tree.Tree<F32>,+bias_exact: {bias_tree == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +offset: U32,+offset_rows: Nat,+positions: U32,+tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +window: LeafWindow<limits,BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,image),offset,offset_rows,positions,tree,first,rows>,
  +images: U32,+outputs: U32,+output_height: U32,+output_width: U32,+output_first: U32,+output_rows: U32,+capacity: U32,
  +output: OutputFacts<images,outputs,output_height,output_width,output_first,output_rows,capacity,image,offset_rows,co,
    ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)>,
  +positions_exact: {U32.to_nat(positions) == Nat.mul(U32.to_nat(output_rows),ConvSpec.axis_size(width,kernel,stride,padding)) : Nat}) ->
  BandValues.Leaf(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),BandConvolutionValues.Context{logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},
    Bands.Geometry{images,outputs,output_height,output_width},output_first,output_rows,
    leaf_tree(~custom,activation,ci,co,width,kernel,stride,padding,ConvSpec.axis_size(width,kernel,stride,padding),offset,positions,tree,first,rows,weight_tree,bias_tree),
    Nat.mul(co,U32.to_nat(positions))):
  BandValues.cells(~ActiveCells<Logical,original>,~ActiveCell(~custom,~Logical,~original),~active_geometry(~Logical,~original),~active_rows(~Logical,~original),~active_leaf_cell(~custom,~Logical,~original),
    ActiveCells{logical,activation,limits,ci,co,height,width,kernel,stride,padding,weights,bias,image,weight_evidence,bias_evidence,valid,supported,weight_tree,weight_exact,bias_tree,bias_exact,offset,offset_rows,positions,tree,first,rows,window,images,outputs,output_height,output_width,output_first,output_rows,capacity,output,positions_exact})

# One active band leaf, as the runtime computes it, holds the dense output
# rows of its image: the window arithmetic decodes to the covered rows, the
# assembled window holds the problem's band, and every element of the leaf
# is the dense convolution value at its output index.
# Natural dimensions survive the runtime's U32 conversion.
# The row arithmetic of BandWindow.window for one band, named once. Every name
# unfolds to the term the runtime computes.
def output_rows_word(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat) -> U32:
  U32.from_nat(ConvSpec.axis_size(height,kernel,stride,padding))

def output_columns_word(+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat) -> U32:
  U32.from_nat(ConvSpec.axis_size(width,kernel,stride,padding))

def core_start(+height: Nat,+first: U32) -> U32:
  (first % U32.from_nat(height) : U32)

def core_stop(+height: Nat,+first: U32,+rows: U32) -> U32:
  (core_start(height,first) + rows : U32)

def image_start(+height: Nat,+first: U32) -> U32:
  (first / U32.from_nat(height) * U32.from_nat(height) : U32)

def band_output_first(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32) -> U32:
  U32.min(output_rows_word(height,kernel,stride,padding),BandWindow.ceiling(core_start(height,first),U32.from_nat(stride)))

def band_output_end(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32) -> U32:
  Bool.pick(U32,U32.is_eq(core_stop(height,first,rows),U32.from_nat(height)),output_rows_word(height,kernel,stride,padding),U32.min(output_rows_word(height,kernel,stride,padding),BandWindow.ceiling(core_stop(height,first,rows),U32.from_nat(stride))))

def band_needed_first(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32) -> U32:
  BandWindow.start(U32.from_nat(height),band_output_first(height,kernel,stride,padding,first),U32.from_nat(stride),U32.from_nat(padding))

def band_needed_end(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32) -> U32:
  U32.min(U32.from_nat(height),BandWindow.subtract_clipped((BandWindow.subtract_clipped(band_output_end(height,kernel,stride,padding,first,rows),1) * U32.from_nat(stride) + U32.from_nat(kernel) : U32),U32.from_nat(padding)))

def window_first_row(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32) -> U32:
  (image_start(height,first) + band_needed_first(height,kernel,stride,padding,first) : U32)

def window_end_row(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32) -> U32:
  (image_start(height,first) + band_needed_end(height,kernel,stride,padding,first,rows) : U32)

def below_first_row(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32) -> U32:
  (image_start(height,first) + U32.max(core_stop(height,first,rows),band_needed_first(height,kernel,stride,padding,first)) : U32)

def above_end_row(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32) -> U32:
  (image_start(height,first) + U32.min(core_start(height,first),band_needed_end(height,kernel,stride,padding,first,rows)) : U32)

def needed_rows(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32) -> BandWindow.Rows:
  BandWindow.Rows{window_first_row(height,kernel,stride,padding,first),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first))}

def halo_above_rows(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32) -> BandWindow.Rows:
  BandWindow.Rows{window_first_row(height,kernel,stride,padding,first),BandWindow.subtract_clipped(above_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first))}

def halo_below_rows(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32) -> BandWindow.Rows:
  BandWindow.Rows{below_first_row(height,kernel,stride,padding,first,rows),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),below_first_row(height,kernel,stride,padding,first,rows))}

def band_output_count(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32) -> U32:
  BandWindow.subtract_clipped(band_output_end(height,kernel,stride,padding,first,rows),band_output_first(height,kernel,stride,padding,first))

def output_first_row(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32) -> U32:
  ((first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) + band_output_first(height,kernel,stride,padding,first) : U32)

def output_band_rows(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32) -> BandWindow.Rows:
  BandWindow.Rows{output_first_row(height,kernel,stride,padding,first),band_output_count(height,kernel,stride,padding,first,rows)}

def band_positions(+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32) -> U32:
  (band_output_count(height,kernel,stride,padding,first,rows) * output_columns_word(width,kernel,stride,padding) : U32)

def input_geometry(+images: U32,+ci: Nat,+height: Nat,+width: Nat) -> Bands.Geometry:
  Bands.Geometry{images,U32.from_nat(ci),U32.from_nat(height),U32.from_nat(width)}

def assembled_window(+images: U32,+ci: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: U32,+rows: U32,+core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>) -> Storage.Buffer<F32>:
  BandWindow.assemble(~F32,0.0,input_geometry(images,ci,height,width),needed_rows(height,kernel,stride,padding,first,rows),BandWindow.Rows{first,rows},halo_above_rows(height,kernel,stride,padding,first,rows),halo_below_rows(height,kernel,stride,padding,first,rows),Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below))

type Words<-ci: Nat,-co: Nat,-height: Nat,-width: Nat,-kernel: Nat,-stride: Nat,-padding: Nat> is Data:
  Words{channels_word: {U32.to_nat(U32.from_nat(ci)) == ci : Nat},outputs_word: {U32.to_nat(U32.from_nat(co)) == co : Nat},
    height_word: {U32.to_nat(U32.from_nat(height)) == height : Nat},width_word: {U32.to_nat(U32.from_nat(width)) == width : Nat},
    kernel_word: {U32.to_nat(U32.from_nat(kernel)) == kernel : Nat},stride_word: {U32.to_nat(U32.from_nat(stride)) == stride : Nat},
    padding_word: {U32.to_nat(U32.from_nat(padding)) == padding : Nat},
    output_height_word: {U32.to_nat(output_rows_word(height,kernel,stride,padding)) == ConvSpec.axis_size(height,kernel,stride,padding) : Nat},
    output_width_word: {U32.to_nat(output_columns_word(width,kernel,stride,padding)) == ConvSpec.axis_size(width,kernel,stride,padding) : Nat}}

# The machine room the window arithmetic and the output tensor need.
type Room<-limits: Machine.Limits,-images: U32,-co: Nat,-height: Nat,-width: Nat,-kernel: Nat,-stride: Nat,-padding: Nat> is Data:
  Room{stride_room: {Nat.is_le(Nat.double(stride),U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},
    coordinate_room: {Nat.is_le(Nat.add(Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),stride),kernel),U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},
    height_room: {Nat.is_le(height,U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},
    padding_room: {Nat.is_le(padding,U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)))),
      U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def ready_room(~Element: Data,+content: Buffer.Content<Element>,ready: Buffer.ContentReady<Element,content>) ->
  {Nat.is_le(Buffer.content_length(~Element,content),U32.to_nat(Tree.capacity(Element,Buffer.content_tree(~Element,content)))) == True{} : Bool}:
  Buffer.ContentReady{depth,balanced,depth_ok,room} = ready
  room

def fields_length(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+length: Nat,
  fields: BandShape.Fields<images,channels,height,width,first,rows,length>) ->
  {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat}:
  BandShape.Fields{local,rows_fit,length_exact} = fields
  length_exact

def range_metadata(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+count: U32,+tree: Tree.Tree<F32>,+length: Nat,
  values: BandHalo.RangeValues(~F32,~Logical,~original,logical,geometry,BandWindow.Rows{first,count},Buffer.Content{tree,length})) -> BandShape.Leaf(geometry,first,count,length):
  Shape.Both{ready,Shape.Both{metadata,leaf}} = values
  metadata

def range_ready(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+count: U32,+tree: Tree.Tree<F32>,+length: Nat,
  values: BandHalo.RangeValues(~F32,~Logical,~original,logical,geometry,BandWindow.Rows{first,count},Buffer.Content{tree,length})) -> Buffer.ContentReady<F32,Buffer.Content{tree,length}>:
  Shape.Both{ready,Shape.Both{metadata,leaf}} = values
  ready

def range_leaf(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+count: U32,+tree: Tree.Tree<F32>,+length: Nat,
  values: BandHalo.RangeValues(~F32,~Logical,~original,logical,geometry,BandWindow.Rows{first,count},Buffer.Content{tree,length})) ->
  BandValues.Leaf(~F32,~Logical,~original,logical,geometry,first,count,tree,length):
  Shape.Both{ready,Shape.Both{metadata,leaf}} = values
  leaf

def model_metadata(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+count: U32,+model: Buffer.Model<F32>,
  +values: BandHalo.RangeValues(~F32,~Logical,~original,logical,geometry,BandWindow.Rows{first,count},Buffer.content(~F32,model))) ->
  BandShape.Leaf(geometry,first,count,Buffer.length(~F32,model)):
  match model:
    case Buffer.Model{+tree,+length,certificate}: range_metadata(~Logical,~original,logical,geometry,first,count,tree,length,values)

# The window arithmetic of an active leaf, decoded to natural numbers.
type Decoded<-ci: Nat,-co: Nat,-height: Nat,-width: Nat,-kernel: Nat,-stride: Nat,-padding: Nat,-first: U32,-rows: U32> is Data:
  Decoded{ordered: {Nat.is_lt(U32.to_nat(band_output_first(height,kernel,stride,padding,first)),U32.to_nat(band_output_end(height,kernel,stride,padding,first,rows))) == True{} : Bool},
    bounded: {Nat.is_le(U32.to_nat(band_output_end(height,kernel,stride,padding,first,rows)),ConvSpec.axis_size(height,kernel,stride,padding)) == True{} : Bool},
    offset_exact: {U32.to_nat(((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32)) == Nat.mul(U32.to_nat(band_output_first(height,kernel,stride,padding,first)),ConvSpec.axis_size(width,kernel,stride,padding)) : Nat},
    positions_exact: {U32.to_nat(band_positions(height,width,kernel,stride,padding,first,rows)) == Nat.mul(Nat.sub(U32.to_nat(band_output_end(height,kernel,stride,padding,first,rows)),U32.to_nat(band_output_first(height,kernel,stride,padding,first))),ConvSpec.axis_size(width,kernel,stride,padding)) : Nat},
    output_rows_exact: {U32.to_nat(band_output_count(height,kernel,stride,padding,first,rows)) == Nat.sub(U32.to_nat(band_output_end(height,kernel,stride,padding,first,rows)),U32.to_nat(band_output_first(height,kernel,stride,padding,first))) : Nat},
    local_exact: {U32.to_nat((window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)) == WindowContents.window_first(height,U32.to_nat(band_output_first(height,kernel,stride,padding,first)),stride,padding) : Nat},
    rows_exact: {U32.to_nat(BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first))) == Nat.sub(WindowContents.window_end(height,U32.to_nat(band_output_end(height,kernel,stride,padding,first,rows)),stride,kernel,padding),WindowContents.window_first(height,U32.to_nat(band_output_first(height,kernel,stride,padding,first)),stride,padding)) : Nat},
    output_first_exact: {U32.to_nat(output_first_row(height,kernel,stride,padding,first)) == Nat.add(Nat.mul(U32.to_nat((first / U32.from_nat(height) : U32)),U32.to_nat(output_rows_word(height,kernel,stride,padding))),U32.to_nat(band_output_first(height,kernel,stride,padding,first))) : Nat},
    window_first_exact: {U32.to_nat(window_first_row(height,kernel,stride,padding,first)) == Nat.add(Nat.mul(U32.to_nat((first / U32.from_nat(height) : U32)),U32.to_nat(U32.from_nat(height))),U32.to_nat((window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32))) : Nat},
    window_fits: {Nat.is_le(Nat.add(U32.to_nat((window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),U32.to_nat(BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)))),U32.to_nat(U32.from_nat(height))) == True{} : Bool},
    count_exact: {U32.to_nat((U32.from_nat(co) * band_positions(height,width,kernel,stride,padding,first,rows) : U32)) == Nat.mul(co,U32.to_nat(band_positions(height,width,kernel,stride,padding,first,rows))) : Nat}}

def active_values(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+limits: Machine.Limits,
  +ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +images: U32,+capacity: U32,+domain: BandGather.Domain<input_geometry(images,ci,height,width),limits,capacity>,+first: U32,+rows: U32,
  core_geometry: TensorBandWindow.CoreGeometry<images,U32.from_nat(height),first,rows,Machine.capacity(limits)>,
  +channels_word: {U32.to_nat(U32.from_nat(ci)) == ci : Nat},+outputs_word: {U32.to_nat(U32.from_nat(co)) == co : Nat},+height_word: {U32.to_nat(U32.from_nat(height)) == height : Nat},+width_word: {U32.to_nat(U32.from_nat(width)) == width : Nat},+output_height_word: {U32.to_nat(output_rows_word(height,kernel,stride,padding)) == ConvSpec.axis_size(height,kernel,stride,padding) : Nat},+output_width_word: {U32.to_nat(output_columns_word(width,kernel,stride,padding)) == ConvSpec.axis_size(width,kernel,stride,padding) : Nat},
  +valid: {ConvSpec.valid(BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat((first / U32.from_nat(height) : U32)))) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat((first / U32.from_nat(height) : U32)))) == True{} : Bool},
  +padding_small: {Nat.is_lt(padding,kernel) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_model: Buffer.Model<Tile.Tile8>,
  +weight_exact: {Buffer.tree(~Tile.Tile8,weight_model) == WeightsLaws.run(U32.from_nat(co),Coordinates.native_inner(BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat((first / U32.from_nat(height) : U32)))),
    StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence)) : Tree.Tree<Tile.Tile8>},
  +bias_model: Buffer.Model<F32>,+bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,
  +above_values: BandHalo.RangeValues(~F32,~Logical,~original,logical,input_geometry(images,ci,height,width),halo_above_rows(height,kernel,stride,padding,first,rows),Buffer.content(~F32,above)),
  +core_values: BandHalo.RangeValues(~F32,~Logical,~original,logical,input_geometry(images,ci,height,width),BandWindow.Rows{first,rows},Buffer.content(~F32,core)),
  +below_values: BandHalo.RangeValues(~F32,~Logical,~original,logical,input_geometry(images,ci,height,width),halo_below_rows(height,kernel,stride,padding,first,rows),Buffer.content(~F32,below)),
  +guard: {U32.is_lt(band_output_first(height,kernel,stride,padding,first),band_output_end(height,kernel,stride,padding,first,rows)) == True{} : Bool},decoded_window: Decoded<ci,co,height,width,kernel,stride,padding,first,rows>,
  +output_total_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(U32.from_nat(co)),Nat.mul(U32.to_nat(output_rows_word(height,kernel,stride,padding)),U32.to_nat(output_columns_word(width,kernel,stride,padding))))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  BandValues.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),BandConvolutionValues.Context{logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},Bands.Geometry{images,U32.from_nat(co),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding)},
    Model.contents(~F32,BandPipeline.windowed(~custom,activation,BandWindow.Geometry{input_geometry(images,ci,height,width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding)},U32.from_nat(co),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),first,rows,TensorBandWindow.raw_window(U32.from_nat(height),(first / U32.from_nat(height) : U32),core_start(height,first),core_stop(height,first,rows),band_needed_first(height,kernel,stride,padding,first),band_needed_end(height,kernel,stride,padding,first,rows),output_band_rows(height,kernel,stride,padding,first,rows)),
      Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model)))):
  match core_geometry decoded_window:
    case TensorBandWindow.CoreGeometry{+image_inside,+core_inside,+first_exact,+core_exact,+endpoint_room} Decoded{+ordered,+output_bounded,+offset_exact,+positions_exact,+output_rows_exact,+local_exact,+rows_exact,+output_first_exact,+window_first_exact,+window_fits,+count_exact}:
    +image = U32.to_nat((first / U32.from_nat(height) : U32))
    +output_first = U32.to_nat(band_output_first(height,kernel,stride,padding,first))
    +output_end = U32.to_nat(band_output_end(height,kernel,stride,padding,first,rows))
    +half = Machine.half_word(limits)
    +half_ok = Limits.half_word_fits(limits)
    +locations = {%guard : TensorBandWindow.Sourced(~F32,input_geometry(images,ci,height,width),BandWindow.nonempty(_,TensorBandWindow.raw_window(U32.from_nat(height),(first / U32.from_nat(height) : U32),core_start(height,first),core_stop(height,first,rows),band_needed_first(height,kernel,stride,padding,first),band_needed_end(height,kernel,stride,padding,first,rows),output_band_rows(height,kernel,stride,padding,first,rows))),above,core,below,first,rows)
      TensorBandWindow.window_sources(~F32,images,U32.from_nat(ci),U32.from_nat(height),U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding),first,rows,above,core,below,limits,TensorBandWindow.CoreGeometry{image_inside,core_inside,first_exact,core_exact,endpoint_room}) :
      TensorBandWindow.Sourced(~F32,input_geometry(images,ci,height,width),BandWindow.nonempty(True{},TensorBandWindow.raw_window(U32.from_nat(height),(first / U32.from_nat(height) : U32),core_start(height,first),core_stop(height,first,rows),band_needed_first(height,kernel,stride,padding,first),band_needed_end(height,kernel,stride,padding,first,rows),output_band_rows(height,kernel,stride,padding,first,rows))),above,core,below,first,rows)}
    +target = {%guard : TensorBandWindow.WindowFacts(input_geometry(images,ci,height,width),BandWindow.nonempty(_,TensorBandWindow.raw_window(U32.from_nat(height),(first / U32.from_nat(height) : U32),core_start(height,first),core_stop(height,first,rows),band_needed_first(height,kernel,stride,padding,first),band_needed_end(height,kernel,stride,padding,first,rows),output_band_rows(height,kernel,stride,padding,first,rows))),capacity,half)
      TensorBandWindow.window_fits(images,U32.from_nat(ci),U32.from_nat(height),U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding),first,rows,limits,capacity,domain,TensorBandWindow.CoreGeometry{image_inside,core_inside,first_exact,core_exact,endpoint_room}) :
      TensorBandWindow.WindowFacts(input_geometry(images,ci,height,width),BandWindow.nonempty(True{},TensorBandWindow.raw_window(U32.from_nat(height),(first / U32.from_nat(height) : U32),core_start(height,first),core_stop(height,first,rows),band_needed_first(height,kernel,stride,padding,first),band_needed_end(height,kernel,stride,padding,first,rows),output_band_rows(height,kernel,stride,padding,first,rows))),capacity,half)}
    +within = TensorBandWindow.sources_within(~F32,images,U32.from_nat(ci),U32.from_nat(height),U32.from_nat(width),limits,capacity,domain,above,window_first_row(height,kernel,stride,padding,first),BandWindow.subtract_clipped(above_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),core,first,rows,below,below_first_row(height,kernel,stride,padding,first,rows),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),below_first_row(height,kernel,stride,padding,first,rows)),
      model_metadata(~Logical,~original,logical,input_geometry(images,ci,height,width),window_first_row(height,kernel,stride,padding,first),BandWindow.subtract_clipped(above_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),above,above_values),model_metadata(~Logical,~original,logical,input_geometry(images,ci,height,width),first,rows,core,core_values),model_metadata(~Logical,~original,logical,input_geometry(images,ci,height,width),below_first_row(height,kernel,stride,padding,first,rows),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),below_first_row(height,kernel,stride,padding,first,rows)),below,below_values))
    +window_values = TensorBandWindow.assembly_window(~F32,~Logical,~original,logical,input_geometry(images,ci,height,width),0.0,window_first_row(height,kernel,stride,padding,first),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),above,window_first_row(height,kernel,stride,padding,first),BandWindow.subtract_clipped(above_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),core,first,rows,below,below_first_row(height,kernel,stride,padding,first,rows),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),below_first_row(height,kernel,stride,padding,first,rows)),
      limits,capacity,domain,above_values,core_values,below_values,within,locations,target)
    +window_split = {%LeafExecution.contents_parts(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below)) : BandHalo.RangeValues(~F32,~Logical,~original,logical,input_geometry(images,ci,height,width),needed_rows(height,kernel,stride,padding,first,rows),_)
      window_values : BandHalo.RangeValues(~F32,~Logical,~original,logical,input_geometry(images,ci,height,width),needed_rows(height,kernel,stride,padding,first,rows),Buffer.Content{Tree.reflect(F32,LeafExecution.storage(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below))),LeafExecution.buffer_length(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below))})}
    +window_ready = range_ready(~Logical,~original,logical,input_geometry(images,ci,height,width),window_first_row(height,kernel,stride,padding,first),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),Tree.reflect(F32,LeafExecution.storage(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below))),LeafExecution.buffer_length(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below)),window_split)
    +window_metadata = range_metadata(~Logical,~original,logical,input_geometry(images,ci,height,width),window_first_row(height,kernel,stride,padding,first),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),Tree.reflect(F32,LeafExecution.storage(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below))),LeafExecution.buffer_length(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below)),window_split)
    +window_leaf = range_leaf(~Logical,~original,logical,input_geometry(images,ci,height,width),window_first_row(height,kernel,stride,padding,first),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),Tree.reflect(F32,LeafExecution.storage(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below))),LeafExecution.buffer_length(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below)),window_split)
    +length_exact = fields_length(images,U32.from_nat(ci),U32.from_nat(height),U32.from_nat(width),window_first_row(height,kernel,stride,padding,first),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),LeafExecution.buffer_length(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below)),window_metadata)
    +tree_room = ready_room(~F32,Buffer.Content{Tree.reflect(F32,LeafExecution.storage(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below))),LeafExecution.buffer_length(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below))},window_ready)
    +window = leaf_window(~Logical,~original,logical,limits,ci,co,height,width,kernel,stride,padding,weights,bias,image,valid,padding_small,output_first,output_end,ordered,output_bounded,
      ((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32),offset_exact,band_positions(height,width,kernel,stride,padding,first,rows),positions_exact,(window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32),local_exact,BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),rows_exact,images,U32.from_nat(ci),U32.from_nat(height),U32.from_nat(width),
      channels_word,height_word,width_word,window_first_row(height,kernel,stride,padding,first),window_first_exact,window_fits,Tree.reflect(F32,LeafExecution.storage(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below))),LeafExecution.buffer_length(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below)),length_exact,window_leaf,tree_room)
    +output_row_fits = {
      %Equal.sym(Nat,U32.to_nat(band_output_count(height,kernel,stride,padding,first,rows)),Nat.sub(output_end,output_first),output_rows_exact) : {Nat.is_le(Nat.add(output_first,_),U32.to_nat(output_rows_word(height,kernel,stride,padding))) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(output_first,Nat.sub(output_end,output_first)),output_end,Order.restore_shift(output_first,output_end,Bounds.strict_le(output_first,output_end,ordered))) : {Nat.is_le(_,U32.to_nat(output_rows_word(height,kernel,stride,padding))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(output_rows_word(height,kernel,stride,padding)),ConvSpec.axis_size(height,kernel,stride,padding),output_height_word) : {Nat.is_le(output_end,_) == True{} : Bool}
      output_bounded : {Nat.is_le(Nat.add(output_first,U32.to_nat(band_output_count(height,kernel,stride,padding,first,rows))),U32.to_nat(output_rows_word(height,kernel,stride,padding))) == True{} : Bool}}
    +output = {OutputFacts{outputs_word,output_height_word,output_width_word,output_first_exact,image_inside,output_row_fits,half_ok,output_total_room} :
      OutputFacts<images,U32.from_nat(co),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding),output_first_row(height,kernel,stride,padding,first),band_output_count(height,kernel,stride,padding,first,rows),half,image,output_first,co,ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)>}
    +positions_rows = Equal.trans(Nat,U32.to_nat(band_positions(height,width,kernel,stride,padding,first,rows)),Nat.mul(Nat.sub(output_end,output_first),ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mul(U32.to_nat(band_output_count(height,kernel,stride,padding,first,rows)),ConvSpec.axis_size(width,kernel,stride,padding)),positions_exact,
      Equal.cong(Nat,Nat,value => Nat.mul(value,ConvSpec.axis_size(width,kernel,stride,padding)),Nat.sub(output_end,output_first),U32.to_nat(band_output_count(height,kernel,stride,padding,first,rows)),
        Equal.sym(Nat,U32.to_nat(band_output_count(height,kernel,stride,padding,first,rows)),Nat.sub(output_end,output_first),output_rows_exact)))
    +all = leaf_all(~custom,~Logical,~original,logical,activation,limits,ci,co,height,width,kernel,stride,padding,weights,bias,image,weight_evidence,bias_evidence,valid,supported,
      Buffer.tree(~Tile.Tile8,weight_model),weight_exact,Buffer.tree(~F32,bias_model),bias_exact,((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32),output_first,band_positions(height,width,kernel,stride,padding,first,rows),Tree.reflect(F32,LeafExecution.storage(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below))),(window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),window,
      images,U32.from_nat(co),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding),output_first_row(height,kernel,stride,padding,first),band_output_count(height,kernel,stride,padding,first,rows),half,output,positions_rows)
    %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Pipeline.finish_leaf(U32.is_eq(band_positions(height,width,kernel,stride,padding,first,rows),((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32)),Pipeline.computed(~custom,activation,U32.from_nat(co),((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model),PackingBuffer.run(Packing.Call{U32.to_nat((((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32)),KernelWindow.start(((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32),U32.from_nat(stride),output_columns_word(width,kernel,stride,padding),(window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_columns_word(width,kernel,stride,padding),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))},assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),(((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32)))),U32.from_nat(co),0,((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32),band_positions(height,width,kernel,stride,padding,first,rows))),Buffer.Content{LeafExecution.leaf_tree(~custom,activation,U32.from_nat(co),((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),band_positions(height,width,kernel,stride,padding,first,rows),Packing.Call{U32.to_nat((((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32)),KernelWindow.start(((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32),U32.from_nat(stride),output_columns_word(width,kernel,stride,padding),(window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_columns_word(width,kernel,stride,padding),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))},(((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32),Tree.reflect(F32,LeafExecution.storage(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below))),Buffer.tree(~Tile.Tile8,weight_model),Buffer.tree(~F32,bias_model)),U32.to_nat((U32.from_nat(co) * band_positions(height,width,kernel,stride,padding,first,rows) : U32))},
      LeafExecution.leaf_contents_buffer(~custom,activation,U32.from_nat(co),((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),band_positions(height,width,kernel,stride,padding,first,rows),Packing.Call{U32.to_nat((((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32)),KernelWindow.start(((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32),U32.from_nat(stride),output_columns_word(width,kernel,stride,padding),(window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_columns_word(width,kernel,stride,padding),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))},(((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32),assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below),weight_model,bias_model)) :
      BandValues.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),BandConvolutionValues.Context{logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},Bands.Geometry{images,U32.from_nat(co),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding)},Model.RowBand{_,output_first_row(height,kernel,stride,padding,first),band_output_count(height,kernel,stride,padding,first,rows)})
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(co) * band_positions(height,width,kernel,stride,padding,first,rows) : U32)),Nat.mul(co,U32.to_nat(band_positions(height,width,kernel,stride,padding,first,rows))),count_exact) :
      BandValues.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),BandConvolutionValues.Context{logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},Bands.Geometry{images,U32.from_nat(co),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding)},Model.RowBand{Buffer.Content{LeafExecution.leaf_tree(~custom,activation,U32.from_nat(co),((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),band_positions(height,width,kernel,stride,padding,first,rows),Packing.Call{U32.to_nat((((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32)),KernelWindow.start(((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32),U32.from_nat(stride),output_columns_word(width,kernel,stride,padding),(window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)),U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_columns_word(width,kernel,stride,padding),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))},(((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32),Tree.reflect(F32,LeafExecution.storage(~F32,assembled_window(images,ci,height,width,kernel,stride,padding,first,rows,core,above,below))),Buffer.tree(~Tile.Tile8,weight_model),Buffer.tree(~F32,bias_model)),_},output_first_row(height,kernel,stride,padding,first),band_output_count(height,kernel,stride,padding,first,rows)})
    all

# The window arithmetic of an active leaf, decoded from the input geometry.
def decode(+limits: Machine.Limits,+ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+images: U32,+capacity: U32,+domain: BandGather.Domain<input_geometry(images,ci,height,width),limits,capacity>,+first: U32,+rows: U32,
  core_geometry: TensorBandWindow.CoreGeometry<images,U32.from_nat(height),first,rows,Machine.capacity(limits)>,+outputs_word: {U32.to_nat(U32.from_nat(co)) == co : Nat},+height_word: {U32.to_nat(U32.from_nat(height)) == height : Nat},+kernel_word: {U32.to_nat(U32.from_nat(kernel)) == kernel : Nat},+stride_word: {U32.to_nat(U32.from_nat(stride)) == stride : Nat},+padding_word: {U32.to_nat(U32.from_nat(padding)) == padding : Nat},+output_height_word: {U32.to_nat(output_rows_word(height,kernel,stride,padding)) == ConvSpec.axis_size(height,kernel,stride,padding) : Nat},+output_width_word: {U32.to_nat(output_columns_word(width,kernel,stride,padding)) == ConvSpec.axis_size(width,kernel,stride,padding) : Nat},
  +stride_room: {Nat.is_le(Nat.double(stride),U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},+coordinate_room: {Nat.is_le(Nat.add(Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),stride),kernel),U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},+height_room: {Nat.is_le(height,U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},+padding_room: {Nat.is_le(padding,U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},+output_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)))),
      U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outputs_positive: {Nat.is_lt(0n,co) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +guard: {U32.is_lt(band_output_first(height,kernel,stride,padding,first),band_output_end(height,kernel,stride,padding,first,rows)) == True{} : Bool}) -> Decoded<ci,co,height,width,kernel,stride,padding,first,rows>:
  match core_geometry:
    case TensorBandWindow.CoreGeometry{+image_inside,+core_inside,+first_exact,+core_exact,+endpoint_room}:
      +cap = Machine.capacity(limits)
      +capn = U32.to_nat(cap)
      +ok = Limits.capacity_fits(limits)
      +wcap = WindowLeaf.capacity(limits)
      +wcapn = U32.to_nat(wcap)
      +wok = WindowLeaf.capacity_fits(limits)
      +image = U32.to_nat((first / U32.from_nat(height) : U32))
      +of = U32.to_nat(band_output_first(height,kernel,stride,padding,first))
      +oe = U32.to_nat(band_output_end(height,kernel,stride,padding,first,rows))
      +oh = ConvSpec.axis_size(height,kernel,stride,padding)
      +ow = ConvSpec.axis_size(width,kernel,stride,padding)
      +images_n = U32.to_nat(images)
      +ordered = Equal.trans(Bool,Nat.is_lt(of,oe),U32.is_lt(band_output_first(height,kernel,stride,padding,first),band_output_end(height,kernel,stride,padding,first,rows)),True{},
        Equal.sym(Bool,U32.is_lt(band_output_first(height,kernel,stride,padding,first),band_output_end(height,kernel,stride,padding,first,rows)),Nat.is_lt(of,oe),Comparison.lt(band_output_first(height,kernel,stride,padding,first),band_output_end(height,kernel,stride,padding,first,rows))),guard)
      +bounded = {%output_height_word : {Nat.is_le(oe,_) == True{} : Bool}
        TensorBandWindow.end_below(output_rows_word(height,kernel,stride,padding),core_stop(height,first,rows),U32.from_nat(stride),U32.is_eq(core_stop(height,first,rows),U32.from_nat(height))) : {Nat.is_le(oe,oh) == True{} : Bool}}
      +first_before = Order.transitive(of,oe,oh,Bounds.strict_le(of,oe,ordered),bounded)
      +images_positive = Order.le_lt_trans(0n,image,images_n,Order.zero_le(image),image_inside)
      +width_positive = {{==} : {Nat.is_lt(0n,ow) == True{} : Bool}}
      +height_positive = {{==} : {Nat.is_lt(0n,oh) == True{} : Bool}}
      +plane = Nat.mul(oh,ow)
      +volume = Nat.mul(co,plane)
      +volume_room = Order.transitive(volume,Nat.mul(images_n,volume),capn,Padded.product_has_right(images_n,volume,images_positive),output_room)
      +plane_volume = Padded.product_has_right(co,plane,outputs_positive)
      +plane_room = Order.transitive(plane,volume,capn,plane_volume,volume_room)
      +height_plane = RowMajor.product_covers(oh,ow,width_positive)
      +end_room = Order.transitive(oe,oh,capn,bounded,Order.transitive(oh,plane,capn,height_plane,plane_room))
      +first_room = Order.transitive(of,oe,capn,Bounds.strict_le(of,oe,ordered),end_room)
      +rows_room = Order.transitive(Nat.mul(1n+image,oh),Nat.mul(images_n,oh),capn,RowMajor.mul_le(1n+image,images_n,oh,Order.strict_follow(image,images_n,image_inside)),
        Order.transitive(Nat.mul(images_n,oh),Nat.mul(images_n,volume),capn,
          WindowContents.multiply_left_le(images_n,oh,volume,Order.transitive(oh,plane,volume,height_plane,plane_volume)),output_room))
      +base_room = Order.transitive(Nat.mul(image,oh),Nat.mul(1n+image,oh),capn,Samples.le_after(oh,Nat.mul(image,oh)),rows_room)
      +base_exact = {%output_height_word : {U32.to_nat((first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32)) == Nat.mul(image,_) : Nat}
        Arithmetic.multiply((first / U32.from_nat(height) : U32),output_rows_word(height,kernel,stride,padding),cap,ok,
          %Equal.sym(Nat,U32.to_nat(output_rows_word(height,kernel,stride,padding)),oh,output_height_word) : {Nat.is_le(Nat.mul(image,_),capn) == True{} : Bool} base_room) :
        {U32.to_nat((first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32)) == Nat.mul(image,oh) : Nat}}
      +start_room = {%Equal.sym(Nat,U32.to_nat((first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32)),Nat.mul(image,oh),base_exact) : {Nat.is_le(Nat.add(_,of),capn) == True{} : Bool}
        Order.transitive(Nat.add(Nat.mul(image,oh),of),Nat.add(Nat.mul(image,oh),oh),capn,Order.offset_le(Nat.mul(image,oh),of,oh,first_before),
          %Order.commutative(oh,Nat.mul(image,oh)) : {Nat.is_le(_,capn) == True{} : Bool} rows_room) :
        {Nat.is_le(Nat.add(U32.to_nat((first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32)),of),capn) == True{} : Bool}}
      +output_first_exact = Equal.trans(Nat,U32.to_nat(output_first_row(height,kernel,stride,padding,first)),Nat.add(U32.to_nat((first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32)),of),Nat.add(Nat.mul(image,U32.to_nat(output_rows_word(height,kernel,stride,padding))),of),
        LeafExecution.sum_exact((first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32),band_output_first(height,kernel,stride,padding,first),cap,ok,start_room),
        Equal.cong(Nat,Nat,value => Nat.add(value,of),U32.to_nat((first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32)),Nat.mul(image,U32.to_nat(output_rows_word(height,kernel,stride,padding))),
          Equal.trans(Nat,U32.to_nat((first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32)),Nat.mul(image,oh),Nat.mul(image,U32.to_nat(output_rows_word(height,kernel,stride,padding))),base_exact,
            Equal.cong(Nat,Nat,value => Nat.mul(image,value),oh,U32.to_nat(output_rows_word(height,kernel,stride,padding)),Equal.sym(Nat,U32.to_nat(output_rows_word(height,kernel,stride,padding)),oh,output_height_word)))))
      +offset_room = {%Equal.sym(Nat,U32.to_nat(output_columns_word(width,kernel,stride,padding)),ow,output_width_word) : {Nat.is_le(Nat.mul(of,_),capn) == True{} : Bool}
        Order.transitive(Nat.mul(of,ow),plane,capn,RowMajor.mul_le(of,oh,ow,first_before),plane_room) : {Nat.is_le(Nat.mul(of,U32.to_nat(output_columns_word(width,kernel,stride,padding))),capn) == True{} : Bool}}
      +offset_exact = Equal.trans(Nat,U32.to_nat(((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32)),Nat.mul(of,U32.to_nat(output_columns_word(width,kernel,stride,padding))),Nat.mul(of,ow),
        LeafExecution.scaled_prefix((first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32),band_output_first(height,kernel,stride,padding,first),output_columns_word(width,kernel,stride,padding),cap,ok,start_room,offset_room),
        Equal.cong(Nat,Nat,value => Nat.mul(of,value),U32.to_nat(output_columns_word(width,kernel,stride,padding)),ow,output_width_word))
      +count_room = Order.transitive(Nat.mul(Nat.sub(oe,of),ow),plane,capn,RowMajor.mul_le(Nat.sub(oe,of),oh,ow,Order.transitive(Nat.sub(oe,of),oe,oh,Order.sub_le(oe,of),bounded)),plane_room)
      +positions_exact = Equal.trans(Nat,U32.to_nat(band_positions(height,width,kernel,stride,padding,first,rows)),Nat.mul(Nat.sub(oe,of),U32.to_nat(output_columns_word(width,kernel,stride,padding))),Nat.mul(Nat.sub(oe,of),ow),
        LeafExecution.clipped_scaled(band_output_end(height,kernel,stride,padding,first,rows),band_output_first(height,kernel,stride,padding,first),output_columns_word(width,kernel,stride,padding),cap,ok,end_room,first_room,
          %Equal.sym(Nat,U32.to_nat(output_columns_word(width,kernel,stride,padding)),ow,output_width_word) : {Nat.is_le(Nat.mul(Nat.sub(oe,of),_),capn) == True{} : Bool} count_room),
        Equal.cong(Nat,Nat,value => Nat.mul(Nat.sub(oe,of),value),U32.to_nat(output_columns_word(width,kernel,stride,padding)),ow,output_width_word))
      +output_rows_exact = TensorBandWindow.subtract_clipped(band_output_end(height,kernel,stride,padding,first,rows),band_output_first(height,kernel,stride,padding,first),cap,ok,end_room,first_room)
      +needed_start = Order.transitive(U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),U32.to_nat(TensorBandWindow.samples_first(U32.from_nat(height),band_output_first(height,kernel,stride,padding,first),U32.from_nat(stride),U32.from_nat(padding))),U32.to_nat(U32.from_nat(height)),
        TensorBandWindow.clipped_below(TensorBandWindow.samples_first(U32.from_nat(height),band_output_first(height,kernel,stride,padding,first),U32.from_nat(stride),U32.from_nat(padding)),(TensorBandWindow.samples_first(U32.from_nat(height),band_output_first(height,kernel,stride,padding,first),U32.from_nat(stride),U32.from_nat(padding)) % U32.from_nat(stride) : U32)),
        TensorBandWindow.minimum_below(U32.from_nat(height),BandWindow.subtract_clipped((band_output_first(height,kernel,stride,padding,first) * U32.from_nat(stride) : U32),U32.from_nat(padding))))
      +needed_end = TensorBandWindow.minimum_below(U32.from_nat(height),BandWindow.subtract_clipped((BandWindow.subtract_clipped(band_output_end(height,kernel,stride,padding,first,rows),1) * U32.from_nat(stride) + U32.from_nat(kernel) : U32),U32.from_nat(padding)))
      +local_start = LeafExecution.prefix_removed(image_start(height,first),band_needed_first(height,kernel,stride,padding,first),cap,ok,
        Order.transitive(Nat.add(U32.to_nat(image_start(height,first)),U32.to_nat(band_needed_first(height,kernel,stride,padding,first))),Nat.add(U32.to_nat(image_start(height,first)),U32.to_nat(U32.from_nat(height))),capn,
          Order.offset_le(U32.to_nat(image_start(height,first)),U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),U32.to_nat(U32.from_nat(height)),needed_start),endpoint_room))
      +coordinate = Nat.add(Nat.mul(oh,stride),kernel)
      +stride_word_room = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(stride)),stride,stride_word) : {Nat.is_le(Nat.double(_),wcapn) == True{} : Bool}
        stride_room : {Nat.is_le(Nat.double(U32.to_nat(U32.from_nat(stride))),wcapn) == True{} : Bool}}
      +stride_word_positive = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(stride)),stride,stride_word) : {Nat.is_lt(0n,_) == True{} : Bool}
        stride_positive : {Nat.is_lt(0n,U32.to_nat(U32.from_nat(stride))) == True{} : Bool}}
      +padding_word_room = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(padding)),padding,padding_word) : {Nat.is_le(_,wcapn) == True{} : Bool}
        padding_room : {Nat.is_le(U32.to_nat(U32.from_nat(padding)),wcapn) == True{} : Bool}}
      +height_word_room = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(height)),height,height_word) : {Nat.is_le(_,wcapn) == True{} : Bool}
        height_room : {Nat.is_le(U32.to_nat(U32.from_nat(height)),wcapn) == True{} : Bool}}
      +product_first = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(stride)),stride,stride_word) : {Nat.is_le(Nat.mul(of,_),wcapn) == True{} : Bool}
        Order.transitive(Nat.mul(of,stride),coordinate,wcapn,
          Order.transitive(Nat.mul(of,stride),Nat.mul(oh,stride),coordinate,RowMajor.mul_le(of,oh,stride,first_before),Order.le_add(Nat.mul(oh,stride),kernel)),coordinate_room) :
        {Nat.is_le(Nat.mul(of,U32.to_nat(U32.from_nat(stride))),wcapn) == True{} : Bool}}
      +last_before = Order.transitive(Nat.sub(oe,1n),oe,oh,Order.sub_le(oe,1n),bounded)
      +last_room = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(stride)),stride,stride_word) : {Nat.is_le(Nat.add(Nat.mul(Nat.sub(oe,1n),_),U32.to_nat(U32.from_nat(kernel))),wcapn) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(kernel)),kernel,kernel_word) : {Nat.is_le(Nat.add(Nat.mul(Nat.sub(oe,1n),stride),_),wcapn) == True{} : Bool}
        Order.transitive(Nat.add(Nat.mul(Nat.sub(oe,1n),stride),kernel),coordinate,wcapn,
          Order.add_le(Nat.mul(Nat.sub(oe,1n),stride),Nat.mul(oh,stride),kernel,RowMajor.mul_le(Nat.sub(oe,1n),oh,stride,last_before)),coordinate_room) :
        {Nat.is_le(Nat.add(Nat.mul(Nat.sub(oe,1n),U32.to_nat(U32.from_nat(stride))),U32.to_nat(U32.from_nat(kernel))),wcapn) == True{} : Bool}}
      +end_window_room = Order.transitive(oe,oh,wcapn,bounded,
        Order.transitive(oh,coordinate,wcapn,Order.transitive(oh,Nat.mul(oh,stride),coordinate,RowMajor.product_covers(oh,stride,stride_positive),Order.le_add(Nat.mul(oh,stride),kernel)),
          coordinate_room))
      +one_room = Order.transitive(1n,oe,wcapn,Order.strict_follow(0n,oe,Order.le_lt_trans(0n,of,oe,Order.zero_le(of),ordered)),end_window_room)
      +start_exact = {
        %height_word : {U32.to_nat(band_needed_first(height,kernel,stride,padding,first)) == WindowContents.window_first(_,of,stride,padding) : Nat}
        %stride_word : {U32.to_nat(band_needed_first(height,kernel,stride,padding,first)) == WindowContents.window_first(U32.to_nat(U32.from_nat(height)),of,_,padding) : Nat}
        %padding_word : {U32.to_nat(band_needed_first(height,kernel,stride,padding,first)) == WindowContents.window_first(U32.to_nat(U32.from_nat(height)),of,U32.to_nat(U32.from_nat(stride)),_) : Nat}
        LeafExecution.start_exact(U32.from_nat(height),band_output_first(height,kernel,stride,padding,first),U32.from_nat(stride),U32.from_nat(padding),wcap,wok,product_first,padding_word_room,height_word_room,stride_word_room,stride_word_positive) :
        {U32.to_nat(band_needed_first(height,kernel,stride,padding,first)) == WindowContents.window_first(height,of,stride,padding) : Nat}}
      +end_exact = {
        %height_word : {U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)) == WindowContents.window_end(_,oe,stride,kernel,padding) : Nat}
        %stride_word : {U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)) == WindowContents.window_end(U32.to_nat(U32.from_nat(height)),oe,_,kernel,padding) : Nat}
        %kernel_word : {U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)) == WindowContents.window_end(U32.to_nat(U32.from_nat(height)),oe,U32.to_nat(U32.from_nat(stride)),_,padding) : Nat}
        %padding_word : {U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)) == WindowContents.window_end(U32.to_nat(U32.from_nat(height)),oe,U32.to_nat(U32.from_nat(stride)),U32.to_nat(U32.from_nat(kernel)),_) : Nat}
        TensorBandWindow.end_decoded(U32.from_nat(height),band_output_end(height,kernel,stride,padding,first,rows),U32.from_nat(stride),U32.from_nat(kernel),U32.from_nat(padding),wcap,wok,end_window_room,one_room,last_room,padding_word_room) :
        {U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)) == WindowContents.window_end(height,oe,stride,kernel,padding) : Nat}}
      +start = WindowContents.window_first(height,of,stride,padding)
      +finish = WindowContents.window_end(height,oe,stride,kernel,padding)
      +window_rows = LeafExecution.rows_exact(image_start(height,first),band_needed_first(height,kernel,stride,padding,first),band_needed_end(height,kernel,stride,padding,first,rows),U32.from_nat(height),cap,ok,endpoint_room,needed_start,needed_end)
      +rows_exact = Equal.trans(Nat,U32.to_nat(BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first))),Nat.sub(U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)),U32.to_nat(band_needed_first(height,kernel,stride,padding,first))),Nat.sub(finish,start),window_rows,
        Equal.trans(Nat,Nat.sub(U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)),U32.to_nat(band_needed_first(height,kernel,stride,padding,first))),Nat.sub(finish,U32.to_nat(band_needed_first(height,kernel,stride,padding,first))),Nat.sub(finish,start),
          Equal.cong(Nat,Nat,value => Nat.sub(value,U32.to_nat(band_needed_first(height,kernel,stride,padding,first))),U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)),finish,end_exact),
          Equal.cong(Nat,Nat,value => Nat.sub(finish,value),U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),start,start_exact)))
      +input_rows = BandGather.rows_room(input_geometry(images,ci,height,width),limits,capacity,domain)
      +input_base = Arithmetic.multiply((first / U32.from_nat(height) : U32),U32.from_nat(height),cap,ok,
        Order.transitive(Nat.mul(image,U32.to_nat(U32.from_nat(height))),Nat.mul(images_n,U32.to_nat(U32.from_nat(height))),capn,
          RowMajor.mul_le(image,images_n,U32.to_nat(U32.from_nat(height)),Bounds.strict_le(image,images_n,image_inside)),input_rows))
      +window_first_exact = Equal.trans(Nat,U32.to_nat(window_first_row(height,kernel,stride,padding,first)),Nat.add(U32.to_nat(image_start(height,first)),U32.to_nat(band_needed_first(height,kernel,stride,padding,first))),
        Nat.add(Nat.mul(image,U32.to_nat(U32.from_nat(height))),U32.to_nat((window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32))),
        TensorBandWindow.shifted(image_start(height,first),band_needed_first(height,kernel,stride,padding,first),U32.from_nat(height),cap,ok,endpoint_room,needed_start),
        Equal.trans(Nat,Nat.add(U32.to_nat(image_start(height,first)),U32.to_nat(band_needed_first(height,kernel,stride,padding,first))),Nat.add(Nat.mul(image,U32.to_nat(U32.from_nat(height))),U32.to_nat(band_needed_first(height,kernel,stride,padding,first))),
          Nat.add(Nat.mul(image,U32.to_nat(U32.from_nat(height))),U32.to_nat((window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32))),
          Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(band_needed_first(height,kernel,stride,padding,first))),U32.to_nat(image_start(height,first)),Nat.mul(image,U32.to_nat(U32.from_nat(height))),input_base),
          Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(image,U32.to_nat(U32.from_nat(height))),value),U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),U32.to_nat((window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),
            Equal.sym(Nat,U32.to_nat((window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),local_start))))
      +window_fits = {
        %Equal.sym(Nat,U32.to_nat((window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),local_start) :
          {Nat.is_le(Nat.add(_,U32.to_nat(BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)))),U32.to_nat(U32.from_nat(height))) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first))),Nat.sub(U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)),U32.to_nat(band_needed_first(height,kernel,stride,padding,first))),window_rows) :
          {Nat.is_le(Nat.add(U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),_),U32.to_nat(U32.from_nat(height))) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),Nat.sub(U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)),U32.to_nat(band_needed_first(height,kernel,stride,padding,first)))),Nat.max(U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows))),
          Extrema.clipped_end(U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)))) : {Nat.is_le(_,U32.to_nat(U32.from_nat(height))) == True{} : Bool}
        Extrema.maximum_upper(U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),U32.to_nat(band_needed_end(height,kernel,stride,padding,first,rows)),U32.to_nat(U32.from_nat(height)),needed_start,needed_end) :
        {Nat.is_le(Nat.add(U32.to_nat((window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),U32.to_nat(BandWindow.subtract_clipped(window_end_row(height,kernel,stride,padding,first,rows),window_first_row(height,kernel,stride,padding,first)))),U32.to_nat(U32.from_nat(height))) == True{} : Bool}}
      +count_exact = Equal.trans(Nat,U32.to_nat((U32.from_nat(co) * band_positions(height,width,kernel,stride,padding,first,rows) : U32)),Nat.mul(U32.to_nat(U32.from_nat(co)),U32.to_nat(band_positions(height,width,kernel,stride,padding,first,rows))),Nat.mul(co,U32.to_nat(band_positions(height,width,kernel,stride,padding,first,rows))),
        Arithmetic.multiply(U32.from_nat(co),band_positions(height,width,kernel,stride,padding,first,rows),cap,ok,
          %Equal.sym(Nat,U32.to_nat(U32.from_nat(co)),co,outputs_word) : {Nat.is_le(Nat.mul(_,U32.to_nat(band_positions(height,width,kernel,stride,padding,first,rows))),capn) == True{} : Bool}
          %Equal.sym(Nat,U32.to_nat(band_positions(height,width,kernel,stride,padding,first,rows)),Nat.mul(Nat.sub(oe,of),ow),positions_exact) : {Nat.is_le(Nat.mul(co,_),capn) == True{} : Bool}
          Order.transitive(Nat.mul(co,Nat.mul(Nat.sub(oe,of),ow)),volume,capn,
            WindowContents.multiply_left_le(co,Nat.mul(Nat.sub(oe,of),ow),plane,RowMajor.mul_le(Nat.sub(oe,of),oh,ow,Order.transitive(Nat.sub(oe,of),oe,oh,Order.sub_le(oe,of),bounded))),volume_room)),
        Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(band_positions(height,width,kernel,stride,padding,first,rows))),U32.to_nat(U32.from_nat(co)),co,outputs_word))
      Decoded{ordered,bounded,offset_exact,positions_exact,output_rows_exact,Equal.trans(Nat,U32.to_nat((window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),U32.to_nat(band_needed_first(height,kernel,stride,padding,first)),start,local_start,start_exact),
        rows_exact,output_first_exact,window_first_exact,window_fits,count_exact}

def half_room(+limits: Machine.Limits,+value: Nat,+room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(value,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(value,U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),room,Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits)))

# The output plane fits: every output bound follows from the total output room.
def output_plane_room(+limits: Machine.Limits,+images: U32,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+image: Nat,+image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},
  +outputs_positive: {Nat.is_lt(0n,co) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +plane = Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding))
  +volume = Nat.mul(co,plane)
  Order.transitive(plane,volume,U32.to_nat(Machine.capacity(limits)),Padded.product_has_right(co,plane,outputs_positive),
    Order.transitive(volume,Nat.mul(U32.to_nat(images),volume),U32.to_nat(Machine.capacity(limits)),
      Padded.product_has_right(U32.to_nat(images),volume,Order.le_lt_trans(0n,image,U32.to_nat(images),Order.zero_le(image),image_inside)),output_room))

def output_total_room(+limits: Machine.Limits,+images: U32,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outputs_word: {U32.to_nat(U32.from_nat(co)) == co : Nat},+output_height_word: {U32.to_nat(output_rows_word(height,kernel,stride,padding)) == ConvSpec.axis_size(height,kernel,stride,padding) : Nat},
  +output_width_word: {U32.to_nat(output_columns_word(width,kernel,stride,padding)) == ConvSpec.axis_size(width,kernel,stride,padding) : Nat}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(U32.from_nat(co)),Nat.mul(U32.to_nat(output_rows_word(height,kernel,stride,padding)),U32.to_nat(output_columns_word(width,kernel,stride,padding))))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(co)),co,outputs_word) : {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(_,Nat.mul(U32.to_nat(output_rows_word(height,kernel,stride,padding)),U32.to_nat(output_columns_word(width,kernel,stride,padding))))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(output_rows_word(height,kernel,stride,padding)),ConvSpec.axis_size(height,kernel,stride,padding),output_height_word) : {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(_,U32.to_nat(output_columns_word(width,kernel,stride,padding))))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(output_columns_word(width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding),output_width_word) : {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),_))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
  half_room(limits,Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)))),output_room)

def not_less(+left: Nat,+right: Nat,+outside: {Nat.is_lt(left,right) == False{} : Bool}) -> {Nat.is_le(right,left) == True{} : Bool}:
  match left right:
    case 0n 0n: {==}
    case 0n 1n+right: Empty.absurd({Nat.is_le(1n+right,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},outside)))
    case 1n+left 0n: {==}
    case 1n+ +left 1n+ +right: not_less(left,right,outside)

# A leaf whose core produces no output rows has no cells.
def inactive_values(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+limits: Machine.Limits,
  +ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+images: U32,+first: U32,+rows: U32,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  +first_room: {Nat.is_le(U32.to_nat(band_output_first(height,kernel,stride,padding,first)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+end_room: {Nat.is_le(U32.to_nat(band_output_end(height,kernel,stride,padding,first,rows)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +guard: {U32.is_lt(band_output_first(height,kernel,stride,padding,first),band_output_end(height,kernel,stride,padding,first,rows)) == False{} : Bool}) ->
  BandValues.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),BandConvolutionValues.Context{logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},Bands.Geometry{images,U32.from_nat(co),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding)},Model.contents(~F32,BandPipeline.windowed(~custom,activation,BandWindow.Geometry{input_geometry(images,ci,height,width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding)},U32.from_nat(co),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),first,rows,BandWindow.nonempty(False{},TensorBandWindow.raw_window(U32.from_nat(height),(first / U32.from_nat(height) : U32),core_start(height,first),core_stop(height,first,rows),band_needed_first(height,kernel,stride,padding,first),band_needed_end(height,kernel,stride,padding,first,rows),output_band_rows(height,kernel,stride,padding,first,rows))),Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model)))):
  +cap = Machine.capacity(limits)
  +ok = Limits.capacity_fits(limits)
  +of = U32.to_nat(band_output_first(height,kernel,stride,padding,first))
  +oe = U32.to_nat(band_output_end(height,kernel,stride,padding,first,rows))
  +outside = Equal.trans(Bool,Nat.is_lt(of,oe),U32.is_lt(band_output_first(height,kernel,stride,padding,first),band_output_end(height,kernel,stride,padding,first,rows)),False{},
    Equal.sym(Bool,U32.is_lt(band_output_first(height,kernel,stride,padding,first),band_output_end(height,kernel,stride,padding,first,rows)),Nat.is_lt(of,oe),Comparison.lt(band_output_first(height,kernel,stride,padding,first),band_output_end(height,kernel,stride,padding,first,rows))),guard)
  +rows_zero = Equal.trans(Nat,U32.to_nat(band_output_count(height,kernel,stride,padding,first,rows)),Nat.sub(oe,of),0n,TensorBandWindow.subtract_clipped(band_output_end(height,kernel,stride,padding,first,rows),band_output_first(height,kernel,stride,padding,first),cap,ok,end_room,first_room),
    WindowContents.sub_below(oe,of,not_less(of,oe,outside)))
  +empty = {
    %Equal.sym(Nat,U32.to_nat(band_output_count(height,kernel,stride,padding,first,rows)),0n,rows_zero) : {Nat.mul(U32.to_nat(U32.from_nat(co)),Nat.mul(_,U32.to_nat(output_columns_word(width,kernel,stride,padding)))) == 0n : Nat}
    RowMajor.multiply_zero_right(U32.to_nat(U32.from_nat(co))) : {Nat.mul(U32.to_nat(U32.from_nat(co)),Nat.mul(U32.to_nat(band_output_count(height,kernel,stride,padding,first,rows)),U32.to_nat(output_columns_word(width,kernel,stride,padding)))) == 0n : Nat}}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Pipeline.finish_leaf(U32.is_eq(band_positions(height,width,kernel,stride,padding,first,rows),((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32)),Pipeline.computed(~custom,activation,U32.from_nat(co),((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model),PackingBuffer.run(Packing.Call{U32.to_nat((((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32)),KernelWindow.start(((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32),U32.from_nat(stride),output_columns_word(width,kernel,stride,padding),(window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),0,U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_columns_word(width,kernel,stride,padding),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))},BandWindow.assemble(~F32,0.0,input_geometry(images,ci,height,width),BandWindow.Rows{window_first_row(height,kernel,stride,padding,first),0},BandWindow.Rows{first,rows},BandWindow.Rows{window_first_row(height,kernel,stride,padding,first),0},BandWindow.Rows{below_first_row(height,kernel,stride,padding,first,rows),0},Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below)),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),(((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32)))),U32.from_nat(co),0,((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32),band_positions(height,width,kernel,stride,padding,first,rows))),Buffer.Content{LeafExecution.leaf_tree(~custom,activation,U32.from_nat(co),((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),band_positions(height,width,kernel,stride,padding,first,rows),Packing.Call{U32.to_nat((((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32)),KernelWindow.start(((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32),U32.from_nat(stride),output_columns_word(width,kernel,stride,padding),(window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),0,U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_columns_word(width,kernel,stride,padding),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))},(((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32),Tree.reflect(F32,LeafExecution.storage(~F32,BandWindow.assemble(~F32,0.0,input_geometry(images,ci,height,width),BandWindow.Rows{window_first_row(height,kernel,stride,padding,first),0},BandWindow.Rows{first,rows},BandWindow.Rows{window_first_row(height,kernel,stride,padding,first),0},BandWindow.Rows{below_first_row(height,kernel,stride,padding,first,rows),0},Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below)))),Buffer.tree(~Tile.Tile8,weight_model),Buffer.tree(~F32,bias_model)),U32.to_nat((U32.from_nat(co) * band_positions(height,width,kernel,stride,padding,first,rows) : U32))},
    LeafExecution.leaf_contents_buffer(~custom,activation,U32.from_nat(co),((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),band_positions(height,width,kernel,stride,padding,first,rows),Packing.Call{U32.to_nat((((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32)),KernelWindow.start(((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32),U32.from_nat(stride),output_columns_word(width,kernel,stride,padding),(window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),0,U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_columns_word(width,kernel,stride,padding),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))},(((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32),BandWindow.assemble(~F32,0.0,input_geometry(images,ci,height,width),BandWindow.Rows{window_first_row(height,kernel,stride,padding,first),0},BandWindow.Rows{first,rows},BandWindow.Rows{window_first_row(height,kernel,stride,padding,first),0},BandWindow.Rows{below_first_row(height,kernel,stride,padding,first,rows),0},Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below)),weight_model,bias_model)) :
    BandValues.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),BandConvolutionValues.Context{logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},Bands.Geometry{images,U32.from_nat(co),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding)},Model.RowBand{_,output_first_row(height,kernel,stride,padding,first),band_output_count(height,kernel,stride,padding,first,rows)})
  BandValues.empty_leaf(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),BandConvolutionValues.Context{logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},Bands.Geometry{images,U32.from_nat(co),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding)},
    output_first_row(height,kernel,stride,padding,first),band_output_count(height,kernel,stride,padding,first,rows),
    LeafExecution.leaf_tree(~custom,activation,U32.from_nat(co),((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),band_positions(height,width,kernel,stride,padding,first,rows),Packing.Call{U32.to_nat((((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32)),KernelWindow.start(((output_first_row(height,kernel,stride,padding,first) - (first / U32.from_nat(height) * output_rows_word(height,kernel,stride,padding) : U32) : U32) * output_columns_word(width,kernel,stride,padding) : U32),U32.from_nat(stride),output_columns_word(width,kernel,stride,padding),(window_first_row(height,kernel,stride,padding,first) - image_start(height,first) : U32)),0,U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_columns_word(width,kernel,stride,padding),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))},(((band_positions(height,width,kernel,stride,padding,first,rows) + 7) / 8 * 8 : U32) / 8 * U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))) : U32),Tree.reflect(F32,LeafExecution.storage(~F32,BandWindow.assemble(~F32,0.0,input_geometry(images,ci,height,width),BandWindow.Rows{window_first_row(height,kernel,stride,padding,first),0},BandWindow.Rows{first,rows},BandWindow.Rows{window_first_row(height,kernel,stride,padding,first),0},BandWindow.Rows{below_first_row(height,kernel,stride,padding,first,rows),0},Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below)))),Buffer.tree(~Tile.Tile8,weight_model),Buffer.tree(~F32,bias_model)),
    U32.to_nat((U32.from_nat(co) * band_positions(height,width,kernel,stride,padding,first,rows) : U32)),empty)

def core_image(+images: U32,+height: U32,+first: U32,+rows: U32,+capacity: U32,+core_geometry: TensorBandWindow.CoreGeometry<images,height,first,rows,capacity>) ->
  {Nat.is_lt(U32.to_nat((first / height : U32)),U32.to_nat(images)) == True{} : Bool}:
  match core_geometry:
    case TensorBandWindow.CoreGeometry{image_inside,core_inside,first_exact,core_exact,endpoint_room}: image_inside

def halo_above(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+output: BandWindow.Rows,+above_rows: BandWindow.Rows,+below_rows: BandWindow.Rows,
  +read: BandWindow.Rows,+first: U32,+rows: U32,+above: Buffer.Content<F32>,+below: Buffer.Content<F32>,
  +values: BandHalo.TreeValues(~F32,~Logical,~original,logical,geometry,BandHalo.Leaf{above,below,BandWindow.RowWindow{output,above_rows,below_rows,read},first,rows})) ->
  BandHalo.RangeValues(~F32,~Logical,~original,logical,geometry,above_rows,above):
  Shape.Both{upper,lower} = values
  upper

def halo_below(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+output: BandWindow.Rows,+above_rows: BandWindow.Rows,+below_rows: BandWindow.Rows,
  +read: BandWindow.Rows,+first: U32,+rows: U32,+above: Buffer.Content<F32>,+below: Buffer.Content<F32>,
  +values: BandHalo.TreeValues(~F32,~Logical,~original,logical,geometry,BandHalo.Leaf{above,below,BandWindow.RowWindow{output,above_rows,below_rows,read},first,rows})) ->
  BandHalo.RangeValues(~F32,~Logical,~original,logical,geometry,below_rows,below):
  Shape.Both{upper,lower} = values
  lower

# Every leaf the runtime builds holds its dense output rows, whether its core
# produces output rows or not.
def leaf_when(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,active: Bool,+logical: Logical,+activation: Activations.Activation,+limits: Machine.Limits,
  +ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +images: U32,+capacity: U32,+domain: BandGather.Domain<input_geometry(images,ci,height,width),limits,capacity>,+first: U32,+rows: U32,
  +core_geometry: TensorBandWindow.CoreGeometry<images,U32.from_nat(height),first,rows,Machine.capacity(limits)>,
  +channels_word: {U32.to_nat(U32.from_nat(ci)) == ci : Nat},+outputs_word: {U32.to_nat(U32.from_nat(co)) == co : Nat},+height_word: {U32.to_nat(U32.from_nat(height)) == height : Nat},+width_word: {U32.to_nat(U32.from_nat(width)) == width : Nat},+kernel_word: {U32.to_nat(U32.from_nat(kernel)) == kernel : Nat},+stride_word: {U32.to_nat(U32.from_nat(stride)) == stride : Nat},+padding_word: {U32.to_nat(U32.from_nat(padding)) == padding : Nat},+output_height_word: {U32.to_nat(output_rows_word(height,kernel,stride,padding)) == ConvSpec.axis_size(height,kernel,stride,padding) : Nat},+output_width_word: {U32.to_nat(output_columns_word(width,kernel,stride,padding)) == ConvSpec.axis_size(width,kernel,stride,padding) : Nat},
  +stride_room: {Nat.is_le(Nat.double(stride),U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},+coordinate_room: {Nat.is_le(Nat.add(Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),stride),kernel),U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},+height_room: {Nat.is_le(height,U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},+padding_room: {Nat.is_le(padding,U32.to_nat(WindowLeaf.capacity(limits))) == True{} : Bool},+output_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)))),
      U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outputs_positive: {Nat.is_lt(0n,co) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +valid: {ConvSpec.valid(BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat((first / U32.from_nat(height) : U32)))) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat((first / U32.from_nat(height) : U32)))) == True{} : Bool},
  +padding_small: {Nat.is_lt(padding,kernel) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +weight_model: Buffer.Model<Tile.Tile8>,
  +weight_exact: {Buffer.tree(~Tile.Tile8,weight_model) == WeightsLaws.run(U32.from_nat(co),Coordinates.native_inner(BandConvolutionValues.problem(~Logical,~original,logical,ci,co,height,width,kernel,stride,padding,weights,bias,U32.to_nat((first / U32.from_nat(height) : U32)))),
    StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence)) : Tree.Tree<Tile.Tile8>},
  +bias_model: Buffer.Model<F32>,+bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,
  +core_values: BandHalo.RangeValues(~F32,~Logical,~original,logical,input_geometry(images,ci,height,width),BandWindow.Rows{first,rows},Buffer.content(~F32,core)),
  +halo_values: BandHalo.TreeValues(~F32,~Logical,~original,logical,input_geometry(images,ci,height,width),
    BandHalo.Leaf{Buffer.content(~F32,above),Buffer.content(~F32,below),BandWindow.nonempty(active,TensorBandWindow.raw_window(U32.from_nat(height),(first / U32.from_nat(height) : U32),core_start(height,first),core_stop(height,first,rows),band_needed_first(height,kernel,stride,padding,first),band_needed_end(height,kernel,stride,padding,first,rows),output_band_rows(height,kernel,stride,padding,first,rows))),first,rows}),
  +guard: {U32.is_lt(band_output_first(height,kernel,stride,padding,first),band_output_end(height,kernel,stride,padding,first,rows)) == active : Bool}) ->
  BandValues.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),BandConvolutionValues.Context{logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},Bands.Geometry{images,U32.from_nat(co),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding)},Model.contents(~F32,BandPipeline.windowed(~custom,activation,BandWindow.Geometry{input_geometry(images,ci,height,width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),output_rows_word(height,kernel,stride,padding),output_columns_word(width,kernel,stride,padding)},U32.from_nat(co),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),first,rows,BandWindow.nonempty(active,TensorBandWindow.raw_window(U32.from_nat(height),(first / U32.from_nat(height) : U32),core_start(height,first),core_stop(height,first,rows),band_needed_first(height,kernel,stride,padding,first),band_needed_end(height,kernel,stride,padding,first,rows),output_band_rows(height,kernel,stride,padding,first,rows))),Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model)))):
  match active:
    case True{}:
      +image = U32.to_nat((first / U32.from_nat(height) : U32))
      +image_inside = core_image(images,U32.from_nat(height),first,rows,Machine.capacity(limits),core_geometry)
      active_values(~custom,~Logical,~original,logical,activation,limits,ci,co,height,width,kernel,stride,padding,weights,bias,images,capacity,domain,first,rows,core_geometry,
        channels_word,outputs_word,height_word,width_word,output_height_word,output_width_word,valid,supported,padding_small,weight_evidence,bias_evidence,weight_model,weight_exact,bias_model,bias_exact,core,above,below,
        halo_above(~Logical,~original,logical,input_geometry(images,ci,height,width),output_band_rows(height,kernel,stride,padding,first,rows),halo_above_rows(height,kernel,stride,padding,first,rows),halo_below_rows(height,kernel,stride,padding,first,rows),
          needed_rows(height,kernel,stride,padding,first,rows),first,rows,Buffer.content(~F32,above),Buffer.content(~F32,below),halo_values),
        core_values,
        halo_below(~Logical,~original,logical,input_geometry(images,ci,height,width),output_band_rows(height,kernel,stride,padding,first,rows),halo_above_rows(height,kernel,stride,padding,first,rows),halo_below_rows(height,kernel,stride,padding,first,rows),
          needed_rows(height,kernel,stride,padding,first,rows),first,rows,Buffer.content(~F32,above),Buffer.content(~F32,below),halo_values),
        guard,decode(limits,ci,co,height,width,kernel,stride,padding,images,capacity,domain,first,rows,core_geometry,outputs_word,height_word,kernel_word,stride_word,padding_word,output_height_word,output_width_word,stride_room,coordinate_room,height_room,padding_room,output_room,outputs_positive,stride_positive,guard),
        output_total_room(limits,images,co,height,width,kernel,stride,padding,output_room,outputs_word,output_height_word,output_width_word))
    case False{}:
      +image = U32.to_nat((first / U32.from_nat(height) : U32))
      +height_room = {%Equal.sym(Nat,U32.to_nat(output_rows_word(height,kernel,stride,padding)),ConvSpec.axis_size(height,kernel,stride,padding),output_height_word) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
        Order.transitive(ConvSpec.axis_size(height,kernel,stride,padding),Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding),{==}),
          output_plane_room(limits,images,co,height,width,kernel,stride,padding,image,core_image(images,U32.from_nat(height),first,rows,Machine.capacity(limits),core_geometry),outputs_positive,output_room)) :
        {Nat.is_le(U32.to_nat(output_rows_word(height,kernel,stride,padding)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
      inactive_values(~custom,~Logical,~original,logical,activation,limits,ci,co,height,width,kernel,stride,padding,weights,bias,images,first,rows,core,above,below,weight_model,bias_model,
        Order.transitive(U32.to_nat(band_output_first(height,kernel,stride,padding,first)),U32.to_nat(output_rows_word(height,kernel,stride,padding)),U32.to_nat(Machine.capacity(limits)),TensorBandWindow.below(output_rows_word(height,kernel,stride,padding),core_start(height,first),U32.from_nat(stride)),height_room),
        Order.transitive(U32.to_nat(band_output_end(height,kernel,stride,padding,first,rows)),U32.to_nat(output_rows_word(height,kernel,stride,padding)),U32.to_nat(Machine.capacity(limits)),TensorBandWindow.end_below(output_rows_word(height,kernel,stride,padding),core_stop(height,first,rows),U32.from_nat(stride),U32.is_eq(core_stop(height,first,rows),U32.from_nat(height))),height_room),
        guard)

# An inactive actual window needs no image-inside or core-value premise.
# In particular, a terminal zero-row owner may start at images * height.
def right_factor_room(+left: Nat,+right: Nat,+capacity: U32,positive: {Nat.is_lt(0n,left) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(left,right),U32.to_nat(capacity)) == True{} : Bool}) -> {Nat.is_le(right,U32.to_nat(capacity)) == True{} : Bool}:
  Order.transitive(right,Nat.mul(left,right),U32.to_nat(capacity),
    %RowMajor.multiply_commutative(right,left) : {Nat.is_le(right,_) == True{} : Bool}
    RowMajor.product_covers(right,left,positive),room)

def output_height_room(+limits: Machine.Limits,+images: U32,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,
  images_positive: {Nat.is_lt(0n,U32.to_nat(images)) == True{} : Bool},outputs_positive: {Nat.is_lt(0n,co) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(ConvSpec.axis_size(height,kernel,stride,padding),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +oh = ConvSpec.axis_size(height,kernel,stride,padding)
  +ow = ConvSpec.axis_size(width,kernel,stride,padding)
  Order.transitive(oh,Nat.mul(oh,ow),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(oh,ow,{==}),
    right_factor_room(co,Nat.mul(oh,ow),Machine.capacity(limits),outputs_positive,
      right_factor_room(U32.to_nat(images),Nat.mul(co,Nat.mul(oh,ow)),Machine.capacity(limits),images_positive,room)))

def geometry(+images: U32,dimensions: KernelShape.Dimensions) -> BandWindow.Geometry:
  KernelShape.Dimensions{ci,co,+height,+width,+kernel,+stride,+padding} = dimensions
  BandWindow.Geometry{Bands.Geometry{images,U32.from_nat(ci),U32.from_nat(height),U32.from_nat(width)},U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
    U32.from_nat(ConvSpec.axis_size(height,kernel,stride,padding)),U32.from_nat(ConvSpec.axis_size(width,kernel,stride,padding))}

def output_height(dimensions: KernelShape.Dimensions) -> Nat:
  KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} = dimensions
  KernelShape.axis_size(height,kernel,stride,padding)

def output_channels(dimensions: KernelShape.Dimensions) -> Nat:
  KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} = dimensions
  co

def output_elements(+images: U32,dimensions: KernelShape.Dimensions) -> Nat:
  KernelShape.Dimensions{ci,co,height,width,+kernel,+stride,+padding} = dimensions
  Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(KernelShape.axis_size(height,kernel,stride,padding),KernelShape.axis_size(width,kernel,stride,padding))))

def active(dimensions: KernelShape.Dimensions,+first: U32,+rows: U32) -> Bool:
  KernelShape.Dimensions{ci,co,+height,width,+kernel,+stride,+padding} = dimensions
  +h = U32.from_nat(height)
  +oh = U32.from_nat(ConvSpec.axis_size(height,kernel,stride,padding))
  U32.is_lt(TensorBandWindow.output_first(h,oh,U32.from_nat(stride),first),TensorBandWindow.output_end(h,oh,U32.from_nat(stride),first,rows))

def raw(dimensions: KernelShape.Dimensions,+first: U32,+rows: U32) -> BandWindow.RowWindow:
  KernelShape.Dimensions{ci,co,+height,width,+kernel,+stride,+padding} = dimensions
  +h = U32.from_nat(height)
  +oh = U32.from_nat(ConvSpec.axis_size(height,kernel,stride,padding))
  +s = U32.from_nat(stride)
  +of = TensorBandWindow.output_first(h,oh,s,first)
  +oe = TensorBandWindow.output_end(h,oh,s,first,rows)
  TensorBandWindow.raw_window(h,(first / h : U32),(first % h : U32),(first % h + rows : U32),BandWindow.start(h,of,s,U32.from_nat(padding)),
    TensorBandWindow.samples_end(h,oe,s,U32.from_nat(kernel),U32.from_nat(padding)),BandWindow.Rows{(first / h * oh + of : U32),BandWindow.subtract_clipped(oe,of)})

def window_when(enabled: Bool,dimensions: KernelShape.Dimensions,first: U32,rows: U32) -> BandWindow.RowWindow:
  BandWindow.nonempty(enabled,raw(dimensions,first,rows))

def window_matches(+images: U32,+dimensions: KernelShape.Dimensions,+first: U32,+rows: U32) ->
  {BandWindow.window(geometry(images,dimensions),first,rows) == window_when(active(dimensions,first,rows),dimensions,first,rows) : BandWindow.RowWindow}:
  KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} = dimensions
  {==}

def WindowValues(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+images: U32,dimensions: KernelShape.Dimensions,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+first: U32,+rows: U32,window: BandWindow.RowWindow,core: Buffer.Model<F32>,above: Buffer.Model<F32>,below: Buffer.Model<F32>,weight_model: Buffer.Model<Tile.Tile8>,bias_model: Buffer.Model<F32>) -> Data:
  KernelShape.Dimensions{ci,+co,+height,+width,+kernel,+stride,+padding} = dimensions
  +ci = ci
  +dimensions = {KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} : KernelShape.Dimensions}
  BandValues.Values(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),BandConvolutionValues.Context{logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},
    Bands.Geometry{images,U32.from_nat(co),U32.from_nat(ConvSpec.axis_size(height,kernel,stride,padding)),U32.from_nat(ConvSpec.axis_size(width,kernel,stride,padding))},
    Model.contents(~F32,BandPipeline.windowed(~custom,activation,geometry(images,dimensions),U32.from_nat(co),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),first,rows,window,
      Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model))))

def inactive(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+limits: Machine.Limits,+images: U32,+dimensions: KernelShape.Dimensions,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+first: U32,+rows: U32,+core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  +height_room: {Nat.is_le(U32.to_nat(U32.from_nat(output_height(dimensions))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +guard: {active(dimensions,first,rows) == False{} : Bool}) ->
  WindowValues(~custom,~Logical,~original,logical,activation,images,dimensions,weights,bias,first,rows,BandWindow.window(geometry(images,dimensions),first,rows),core,above,below,weight_model,bias_model):
  KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding} = dimensions
  +dimensions = {KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} : KernelShape.Dimensions}
  %Equal.sym(BandWindow.RowWindow,BandWindow.window(geometry(images,dimensions),first,rows),window_when(active(dimensions,first,rows),dimensions,first,rows),window_matches(images,dimensions,first,rows)) :
    WindowValues(~custom,~Logical,~original,logical,activation,images,dimensions,weights,bias,first,rows,_,core,above,below,weight_model,bias_model)
  %Equal.sym(Bool,active(dimensions,first,rows),False{},guard) :
    WindowValues(~custom,~Logical,~original,logical,activation,images,dimensions,weights,bias,first,rows,window_when(_,dimensions,first,rows),core,above,below,weight_model,bias_model)
  +h = U32.from_nat(height)
  +oh = U32.from_nat(ConvSpec.axis_size(height,kernel,stride,padding))
  +s = U32.from_nat(stride)
  inactive_values(~custom,~Logical,~original,logical,activation,limits,ci,co,height,width,kernel,stride,padding,weights,bias,images,first,rows,core,above,below,weight_model,bias_model,
    Order.transitive(U32.to_nat(TensorBandWindow.output_first(h,oh,s,first)),U32.to_nat(oh),U32.to_nat(Machine.capacity(limits)),TensorBandWindow.below(oh,(first % h : U32),s),height_room),
    Order.transitive(U32.to_nat(TensorBandWindow.output_end(h,oh,s,first,rows)),U32.to_nat(oh),U32.to_nat(Machine.capacity(limits)),TensorBandWindow.end_below(oh,(first % h + rows : U32),s,U32.is_eq((first % h + rows : U32),h)),height_room),guard)

def inactive_checked(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+limits: Machine.Limits,+images: U32,+dimensions: KernelShape.Dimensions,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+first: U32,+rows: U32,+core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  images_positive: {Nat.is_lt(0n,U32.to_nat(images)) == True{} : Bool},outputs_positive: {Nat.is_lt(0n,output_channels(dimensions)) == True{} : Bool},
  +output_room: {Nat.is_le(output_elements(images,dimensions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},guard: {active(dimensions,first,rows) == False{} : Bool}) ->
  WindowValues(~custom,~Logical,~original,logical,activation,images,dimensions,weights,bias,first,rows,BandWindow.window(geometry(images,dimensions),first,rows),core,above,below,weight_model,bias_model):
  KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding} = dimensions
  +dimensions = {KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} : KernelShape.Dimensions}
  +height_room = output_height_room(limits,images,co,height,width,kernel,stride,padding,images_positive,outputs_positive,output_room)
  +height_word = Bounds.roundtrip(output_height(dimensions),Machine.capacity(limits),Limits.capacity_fits(limits),height_room)
  inactive(~custom,~Logical,~original,logical,activation,limits,images,dimensions,weights,bias,first,rows,core,above,below,weight_model,bias_model,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(output_height(dimensions))),output_height(dimensions),height_word) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    height_room,guard)
