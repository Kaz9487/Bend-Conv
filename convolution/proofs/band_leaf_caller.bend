# Reuse the existing convolution leaf refinement, with shape/machine facts
# supplied by the public guards. This layer never unfolds the GEMM kernel.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_band_window.bend as BandWindow
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_activation.bend as Activations
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_halo.bend as BandHalo
import ../../lib/proofs/tensor_band_window.bend as TensorBandWindow
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ./reduction_coordinate_bounds.bend as Coordinates
import ./band_leaf_values.bend as LeafValues
import ./band_caller_guards.bend as Caller

def input_geometry(+images: U32,dimensions: KernelShape.Dimensions) -> Bands.Geometry:
  KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} = dimensions
  Bands.Geometry{images,U32.from_nat(ci),U32.from_nat(height),U32.from_nat(width)}

def height(dimensions: KernelShape.Dimensions) -> Nat:
  KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} = dimensions
  height

def expected_weights(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+dimensions: KernelShape.Dimensions,+weights: List<&2,F32>,+bias: List<&2,F32>,+first: U32,
  evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>) -> Tree.Tree<Tile.Tile8>:
  KernelShape.Dimensions{+ci,+co,+h,+width,+kernel,+stride,+padding} = dimensions
  WeightsLaws.run(U32.from_nat(co),Coordinates.native_inner(Caller.problem(~Logical,~original,logical,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},weights,bias,U32.to_nat((first / U32.from_nat(h) : U32)))),
    StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,evidence))

def active_with_bounds(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+limits: Machine.Limits,
  +images: U32,+ci: Nat,+co: Nat,+h: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+first: U32,+rows: U32,+capacity: U32,
  +domain: BandGather.Domain<input_geometry(images,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding}),limits,capacity>,
  +core_geometry: TensorBandWindow.CoreGeometry<images,U32.from_nat(h),first,rows,Machine.capacity(limits)>,
  +words: LeafValues.Words<ci,co,h,width,kernel,stride,padding>,+room: LeafValues.Room<limits,images,co,h,width,kernel,stride,padding>,
  +outputs_positive: {Nat.is_lt(0n,co) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +valid: {ConvSpec.valid(Caller.problem(~Logical,~original,logical,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},weights,bias,U32.to_nat((first / U32.from_nat(h) : U32)))) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,Caller.problem(~Logical,~original,logical,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},weights,bias,U32.to_nat((first / U32.from_nat(h) : U32)))) == True{} : Bool},
  +padding_small: {Nat.is_lt(padding,kernel) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  +weight_exact: {Buffer.tree(~Tile.Tile8,weight_model) == expected_weights(~Logical,~original,logical,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},weights,bias,first,weight_evidence) : Tree.Tree<Tile.Tile8>},
  +bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  +core_values: BandHalo.RangeValues(~F32,~Logical,~original,logical,input_geometry(images,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding}),BandWindow.Rows{first,rows},Buffer.content(~F32,core)),
  +halo_values: BandHalo.TreeValues(~F32,~Logical,~original,logical,input_geometry(images,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding}),
    BandHalo.Leaf{Buffer.content(~F32,above),Buffer.content(~F32,below),BandWindow.window(LeafValues.geometry(images,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding}),first,rows),first,rows}),
  +guard: {LeafValues.active(KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},first,rows) == True{} : Bool}) ->
  LeafValues.WindowValues(~custom,~Logical,~original,logical,activation,images,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},weights,bias,first,rows,
    BandWindow.window(LeafValues.geometry(images,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding}),first,rows),core,above,below,weight_model,bias_model):
  LeafValues.Words{+channels_word,+outputs_word,+height_word,+width_word,+kernel_word,+stride_word,+padding_word,+output_height_word,+output_width_word} = words
  LeafValues.Room{+stride_room,+coordinate_room,+height_room,+padding_room,+output_room} = room
  +dimensions = {KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding} : KernelShape.Dimensions}
  +window_equal = Equal.trans(BandWindow.RowWindow,BandWindow.window(LeafValues.geometry(images,dimensions),first,rows),LeafValues.window_when(LeafValues.active(dimensions,first,rows),dimensions,first,rows),
    LeafValues.window_when(True{},dimensions,first,rows),LeafValues.window_matches(images,dimensions,first,rows),
    Equal.cong(Bool,BandWindow.RowWindow,enabled => LeafValues.window_when(enabled,dimensions,first,rows),LeafValues.active(dimensions,first,rows),True{},guard))
  %Equal.sym(BandWindow.RowWindow,BandWindow.window(LeafValues.geometry(images,dimensions),first,rows),LeafValues.window_when(True{},dimensions,first,rows),window_equal) :
    LeafValues.WindowValues(~custom,~Logical,~original,logical,activation,images,dimensions,weights,bias,first,rows,_,core,above,below,weight_model,bias_model)
  LeafValues.leaf_when(~custom,~Logical,~original,True{},logical,activation,limits,ci,co,h,width,kernel,stride,padding,weights,bias,images,capacity,domain,first,rows,core_geometry,
    channels_word,outputs_word,height_word,width_word,kernel_word,stride_word,padding_word,output_height_word,output_width_word,
    stride_room,coordinate_room,height_room,padding_room,output_room,outputs_positive,stride_positive,valid,supported,padding_small,weight_evidence,bias_evidence,
    weight_model,weight_exact,bias_model,bias_exact,core,above,below,core_values,
    %window_equal : BandHalo.TreeValues(~F32,~Logical,~original,logical,input_geometry(images,dimensions),BandHalo.Leaf{Buffer.content(~F32,above),Buffer.content(~F32,below),_,first,rows})
    halo_values,guard)

def active_checked(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+limits: Machine.Limits,+images: U32,+dimensions: KernelShape.Dimensions,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+first: U32,+rows: U32,+capacity: U32,+domain: BandGather.Domain<input_geometry(images,dimensions),limits,capacity>,
  +core_geometry: TensorBandWindow.CoreGeometry<images,U32.from_nat(height(dimensions)),first,rows,Machine.capacity(limits)>,
  +valid: {KernelShape.valid(dimensions,Caller.input_length(dimensions),ConvSpec.length(weights),ConvSpec.length(bias)) == True{} : Bool},
  +supported: {KernelShape.supports_packed(limits,dimensions,Caller.input_length(dimensions),ConvSpec.length(weights)) == True{} : Bool},
  +output_room: {Nat.is_le(LeafValues.output_elements(images,dimensions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  eligible: {TensorConvolution.uses_band_windows(TensorConvolution.Geometry{images,True{},dimensions}) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  weight_exact: {Buffer.tree(~Tile.Tile8,weight_model) == expected_weights(~Logical,~original,logical,dimensions,weights,bias,first,weight_evidence) : Tree.Tree<Tile.Tile8>},
  bias_exact: {Buffer.tree(~F32,bias_model) == StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence) : Tree.Tree<F32>},
  core_values: BandHalo.RangeValues(~F32,~Logical,~original,logical,input_geometry(images,dimensions),BandWindow.Rows{first,rows},Buffer.content(~F32,core)),
  halo_values: BandHalo.TreeValues(~F32,~Logical,~original,logical,input_geometry(images,dimensions),BandHalo.Leaf{Buffer.content(~F32,above),Buffer.content(~F32,below),BandWindow.window(LeafValues.geometry(images,dimensions),first,rows),first,rows}),
  guard: {LeafValues.active(dimensions,first,rows) == True{} : Bool}) ->
  LeafValues.WindowValues(~custom,~Logical,~original,logical,activation,images,dimensions,weights,bias,first,rows,BandWindow.window(LeafValues.geometry(images,dimensions),first,rows),core,above,below,weight_model,bias_model):
  KernelShape.Dimensions{+ci,+co,+h,+width,+kernel,+stride,+padding} = dimensions
  +dimensions = {KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding} : KernelShape.Dimensions}
  +image = U32.to_nat((first / U32.from_nat(h) : U32))
  +problem = Caller.problem(~Logical,~original,logical,dimensions,weights,bias,image)
  +checked = Caller.valid(~Logical,~original,logical,dimensions,weights,bias,image,valid)
  +packed = Caller.supported(~Logical,~original,logical,limits,dimensions,weights,bias,image,supported)
  active_with_bounds(~custom,~Logical,~original,logical,activation,limits,images,ci,co,h,width,kernel,stride,padding,weights,bias,first,rows,capacity,domain,core_geometry,
    Caller.words(limits,problem,checked,packed),Caller.room(limits,images,problem,packed,output_room),Caller.outputs_positive(problem,checked),Caller.stride_positive(problem,checked),
    checked,packed,Caller.padding_small(images,True{},ci,co,h,width,kernel,stride,padding,eligible),weight_evidence,bias_evidence,core,above,below,weight_model,bias_model,weight_exact,bias_exact,core_values,halo_values,guard)
