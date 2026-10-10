# The band convolution pipeline: inputs, weights, layout, execution and values.
import Base
import ../../lib/proofs/tensor_band_halo.bend as BandHalo
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_band_window.bend as BandWindow
import ../../lib/kernel_band_pipeline.bend as BandPipeline
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/tensor_band_window.bend as TensorBandWindow
import ./band_caller_guards.bend as Caller
import ./band_leaf_values.bend as BandLeafValues
import ../specification.bend as ConvSpec
import ../../lib/kernel_gemm_buffer.bend as GemmBuffer
import ../../lib/machine_limits.bend as Machine
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/tensor_band_each.bend as Each
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/storage_clone.bend as Clone
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_leaf_caller.bend as Leaf
import ./reduction_coordinate_bounds.bend as Coordinates
import ../../lib/proofs/kernel_weights_buffer_refinement.bend as KernelWeightsBufferRefinement
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/storage_clone_refinement.bend as Cloned
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/traversal_partition.bend as Partition

# Plan induction for the actual band pipeline. The leaf premise and its
# discharge remain explicit; this theorem never unfolds GEMM internals.
type Settings<-Logical: Data> is Data:
  Settings{values_logical: Logical,geometry: Bands.Geometry,context: BandPipeline.Context,values_weights: Buffer.Content<Tile.Tile8>,values_bias: Buffer.Content<F32>}

def values_logical(~Logical: Data,settings: Settings<Logical>) -> Logical:
  Settings{values_logical,geometry,context,values_weights,values_bias} = settings
  values_logical

def geometry(~Logical: Data,settings: Settings<Logical>) -> Bands.Geometry:
  Settings{values_logical,geometry,context,values_weights,values_bias} = settings
  geometry

def context(~Logical: Data,settings: Settings<Logical>) -> BandPipeline.Context:
  Settings{values_logical,geometry,context,values_weights,values_bias} = settings
  context

def values_weights(~Logical: Data,settings: Settings<Logical>) -> Buffer.Content<Tile.Tile8>:
  Settings{values_logical,geometry,context,values_weights,values_bias} = settings
  values_weights

def values_bias(~Logical: Data,settings: Settings<Logical>) -> Buffer.Content<F32>:
  Settings{values_logical,geometry,context,values_weights,values_bias} = settings
  values_bias

def Premise(~Logical: Data) -> Type:
  Settings<Logical> -> U32 -> U32 -> Buffer.Content<F32> -> Buffer.Content<F32> -> Buffer.Content<F32> -> BandWindow.RowWindow -> Data

# Shape mismatches take the runtime fallback; Inputs excludes those branches.
# The public producer must construct this evidence, including zero-row leaves.
def ContentInputs(~Logical: Data,~premise: Premise(~Logical),+settings: Settings<Logical>,plan: Bands.Plan,cores: Model.Content<F32>,halos: BandHalo.Tree<F32>) -> Data:
  match plan cores:
    case Bands.Rows{+first,+rows} Model.RowBand{+core,cf,cr}:
      match halos:
        case BandHalo.Leaf{above,below,window,hf,hr}: premise(settings,first,rows,core,above,below,window)
        case halos: Empty
    case Bands.Split{+left_plan,+right_plan} Model.Joined{+left,+right}:
      match halos:
        case BandHalo.Fork{left_halo,right_halo}: Shape.Both<ContentInputs(~Logical,~premise,settings,left_plan,left,left_halo),ContentInputs(~Logical,~premise,settings,right_plan,right,right_halo)>
        case halos: Empty
    case plan cores: Empty

def Inputs(~Logical: Data,~premise: Premise(~Logical),settings: Settings<Logical>,plan: Bands.Plan,cores: Model.Model<F32>,halos: BandHalo.Owners<F32>) -> Data:
  ContentInputs(~Logical,~premise,settings,plan,Model.content(~F32,cores),BandHalo.owner_contents(~F32,halos))

def NativeInputs(~Logical: Data,~premise: Premise(~Logical),settings: Settings<Logical>,plan: Bands.Plan,cores: Bands.Bands(~F32),halos: BandWindow.Halos(~F32)) -> Data:
  ContentInputs(~Logical,~premise,settings,plan,Model.contents(~F32,cores),BandHalo.reflect(~F32,halos))

def packed_inputs(~Logical: Data,~premise: Premise(~Logical),+settings: Settings<Logical>,+plan: Bands.Plan,+cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>,
  inputs: NativeInputs(~Logical,~premise,settings,plan,Model.pack(~F32,cores),BandHalo.pack_owners(~F32,halos))) -> Inputs(~Logical,~premise,settings,plan,cores,halos):
  %Model.packed_contents(~F32,cores) : ContentInputs(~Logical,~premise,settings,plan,_,BandHalo.owner_contents(~F32,halos))
  %BandHalo.packed_owners(~F32,halos) : ContentInputs(~Logical,~premise,settings,plan,Model.contents(~F32,Model.pack(~F32,cores)),_)
  inputs

def observed_inputs(~Logical: Data,~premise: Premise(~Logical),+settings: Settings<Logical>,+plan: Bands.Plan,+cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>,
  -source: Bands.Bands(~F32),-halo_owners: BandWindow.Halos(~F32),+cores_equal: {Model.pack(~F32,cores) == source : Bands.Bands(~F32)},
  +halos_equal: {BandHalo.pack_owners(~F32,halos) == halo_owners : BandWindow.Halos(~F32)},inputs: NativeInputs(~Logical,~premise,settings,plan,source,halo_owners)) ->
  Inputs(~Logical,~premise,settings,plan,cores,halos):
  packed_inputs(~Logical,~premise,settings,plan,cores,halos,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,cores),source,cores_equal) : NativeInputs(~Logical,~premise,settings,plan,_,BandHalo.pack_owners(~F32,halos))
    %Equal.sym(BandWindow.Halos(~F32),BandHalo.pack_owners(~F32,halos),halo_owners,halos_equal) : NativeInputs(~Logical,~premise,settings,plan,source,_)
    inputs)

def LeafRule(~custom: F32 -> F32,~Logical: Data,~value: Logical -> U32 -> F32,~premise: Premise(~Logical)) -> Type:
  @+settings: Settings<Logical> ->
  @+first: U32 -> @+rows: U32 -> @+core: Buffer.Model<F32> -> @+above: Buffer.Model<F32> -> @+below: Buffer.Model<F32> -> @+window: BandWindow.RowWindow ->
  @+weight_model: Buffer.Model<Tile.Tile8> -> @+bias_model: Buffer.Model<F32> ->
  @input: premise(settings,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),window) ->
  @weight_exact: {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Logical,settings) : Buffer.Content<Tile.Tile8>} ->
  @bias_exact: {Buffer.content(~F32,bias_model) == values_bias(~Logical,settings) : Buffer.Content<F32>} ->
  BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),
    Model.contents(~F32,BandPipeline.leaf(~custom,context(~Logical,settings),first,rows,Buffer.pack(~F32,core),
      BandWindow.Halo{Buffer.pack(~F32,above),Buffer.pack(~F32,below),window},Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model))))

def fork_run(~custom: F32 -> F32,+left_plan: Bands.Plan,+right_plan: Bands.Plan,+context: BandPipeline.Context,+left: Model.Model<F32>,+right: Model.Model<F32>,
  +left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,weight_copies: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,
  bias_copies: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Bands.Bands(~F32):
  Each.forked(~Bands.Paired<Storage.Buffer<F32>,BandWindow.Halo<F32>>,~Storage.Buffer<F32>,~BandPipeline.Parameters(),~BandPipeline.Context,~BandPipeline.share,
    ~Bands.paired(~Storage.Buffer<F32>,~BandWindow.Halo<F32>,~BandPipeline.Parameters(),~BandPipeline.Context,~BandPipeline.band(~custom)),
    ~Bands.left_of(~Storage.Buffer<F32>,~BandWindow.Halo<F32>),
    left_plan,right_plan,BandPipeline.parallel(context),BandPipeline.input_geometry(context),context,
    Bands.zip(~Storage.Buffer<F32>,~BandWindow.Halo<F32>,Model.pack(~F32,left),BandHalo.pack_owners(~F32,left_halo)),
    Bands.zip(~Storage.Buffer<F32>,~BandWindow.Halo<F32>,Model.pack(~F32,right),BandHalo.pack_owners(~F32,right_halo)),
    BandPipeline.shared(weight_copies,bias_copies))

# A split runs both halves, each on its own copy of the values_weights and values_bias; the
# parallel split and its leaf tasks equal the sequential recursion.
def values_fork_execution(~custom: F32 -> F32,+left_plan: Bands.Plan,+right_plan: Bands.Plan,mode: Bool,+context: BandPipeline.Context,+left: Model.Model<F32>,+right: Model.Model<F32>,
  +left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>) ->
  {BandPipeline.run(~custom,Bands.Split{left_plan,right_plan},mode,context,Model.pack(~F32,Model.Fork{left,right}),BandHalo.pack_owners(~F32,BandHalo.OwnedFork{left_halo,right_halo}),
      Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model)) ==
    fork_run(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weight_model)),Clone.clone(~F32,Buffer.pack(~F32,bias_model))) : Bands.Bands(~F32)}:
  Each.fork(~Bands.Paired<Storage.Buffer<F32>,BandWindow.Halo<F32>>,~Storage.Buffer<F32>,~BandPipeline.Parameters(),~BandPipeline.Context,~BandPipeline.share,
    ~Bands.paired(~Storage.Buffer<F32>,~BandWindow.Halo<F32>,~BandPipeline.Parameters(),~BandPipeline.Context,~BandPipeline.band(~custom)),
    ~Bands.left_of(~Storage.Buffer<F32>,~BandWindow.Halo<F32>),
    left_plan,right_plan,mode,BandPipeline.parallel(context),BandPipeline.input_geometry(context),context,
    Bands.zip(~Storage.Buffer<F32>,~BandWindow.Halo<F32>,Model.pack(~F32,left),BandHalo.pack_owners(~F32,left_halo)),
    Bands.zip(~Storage.Buffer<F32>,~BandWindow.Halo<F32>,Model.pack(~F32,right),BandHalo.pack_owners(~F32,right_halo)),
    (Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model)))

def weight_property(~custom: F32 -> F32,~Logical: Data,~value: Logical -> U32 -> F32,+settings: Settings<Logical>,+left_plan: Bands.Plan,+right_plan: Bands.Plan,
  +left: Model.Model<F32>,+right: Model.Model<F32>,+left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,+bias_model: Buffer.Model<F32>,
  weight_copies: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Type:
  BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),
    Model.contents(~F32,fork_run(~custom,left_plan,right_plan,context(~Logical,settings),left,right,left_halo,right_halo,weight_copies,Clone.clone(~F32,Buffer.pack(~F32,bias_model)))))

def bias_property(~custom: F32 -> F32,~Logical: Data,~value: Logical -> U32 -> F32,+settings: Settings<Logical>,+left_plan: Bands.Plan,+right_plan: Bands.Plan,
  +left: Model.Model<F32>,+right: Model.Model<F32>,+left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,+left_weights: Buffer.Model<Tile.Tile8>,+right_weights: Buffer.Model<Tile.Tile8>,
  bias_copies: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Type:
  BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),
    Model.contents(~F32,fork_run(~custom,left_plan,right_plan,context(~Logical,settings),left,right,left_halo,right_halo,
      (Buffer.pack(~Tile.Tile8,left_weights),Buffer.pack(~Tile.Tile8,right_weights)),bias_copies)))

def BranchRule(~custom: F32 -> F32,~Logical: Data,~value: Logical -> U32 -> F32,+settings: Settings<Logical>,+plan: Bands.Plan,+cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>) -> Type:
  @+weight_model: Buffer.Model<Tile.Tile8> -> @+bias_model: Buffer.Model<F32> ->
  @weight_exact: {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Logical,settings) : Buffer.Content<Tile.Tile8>} ->
  @bias_exact: {Buffer.content(~F32,bias_model) == values_bias(~Logical,settings) : Buffer.Content<F32>} ->
  BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),
    Model.contents(~F32,BandPipeline.run(~custom,plan,BandPipeline.parallel_plan(context(~Logical,settings),plan),context(~Logical,settings),
      Model.pack(~F32,cores),BandHalo.pack_owners(~F32,halos),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model))))

def split_values(~custom: F32 -> F32,~Logical: Data,~value: Logical -> U32 -> F32,
  +settings: Settings<Logical>,+left_plan: Bands.Plan,+right_plan: Bands.Plan,+left: Model.Model<F32>,+right: Model.Model<F32>,+left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,
  +weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  left_rule: BranchRule(~custom,~Logical,~value,settings,left_plan,left,left_halo),right_rule: BranchRule(~custom,~Logical,~value,settings,right_plan,right,right_halo),
  +weight_exact: {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Logical,settings) : Buffer.Content<Tile.Tile8>},+bias_exact: {Buffer.content(~F32,bias_model) == values_bias(~Logical,settings) : Buffer.Content<F32>}) ->
  weight_property(~custom,~Logical,~value,settings,left_plan,right_plan,left,right,left_halo,right_halo,bias_model,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weight_model))):
  Cloned.lift_pair(~Tile.Tile8,~weight_property(~custom,~Logical,~value,settings,left_plan,right_plan,left,right,left_halo,right_halo,bias_model),
    Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weight_model)),Cloned.observe_pair(~Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weight_model))),
    left_weights => right_weights => weight_copies =>
      +left_weights = left_weights
      +right_weights = right_weights
      +weight_copies = weight_copies
      Cloned.lift_pair(~F32,~bias_property(~custom,~Logical,~value,settings,left_plan,right_plan,left,right,left_halo,right_halo,left_weights,right_weights),
        Clone.clone(~F32,Buffer.pack(~F32,bias_model)),Cloned.observe_pair(~F32,Clone.clone(~F32,Buffer.pack(~F32,bias_model))),
        left_bias => right_bias => bias_copies =>
          +left_bias = left_bias
          +right_bias = right_bias
          +bias_copies = bias_copies
          Shape.Both{
            left_rule(left_weights,left_bias,
              Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,left_weights),Buffer.content(~Tile.Tile8,weight_model),values_weights(~Logical,settings),Cloned.left_contents(~Tile.Tile8,weight_model,left_weights,right_weights,weight_copies),weight_exact),
              Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,left_bias),Buffer.content(~F32,bias_model),values_bias(~Logical,settings),Cloned.left_contents(~F32,bias_model,left_bias,right_bias,bias_copies),bias_exact)),
            right_rule(right_weights,right_bias,
              Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,right_weights),Buffer.content(~Tile.Tile8,weight_model),values_weights(~Logical,settings),Cloned.right_contents(~Tile.Tile8,weight_model,left_weights,right_weights,weight_copies),weight_exact),
              Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,right_bias),Buffer.content(~F32,bias_model),values_bias(~Logical,settings),Cloned.right_contents(~F32,bias_model,left_bias,right_bias,bias_copies),bias_exact))}))

def values_run_values(~custom: F32 -> F32,~Logical: Data,~value: Logical -> U32 -> F32,~premise: Premise(~Logical),~leaf_rule: LeafRule(~custom,~Logical,~value,~premise),
  +settings: Settings<Logical>,+plan: Bands.Plan,+mode: Bool,+cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  inputs: Inputs(~Logical,~premise,settings,plan,cores,halos),weight_exact: {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Logical,settings) : Buffer.Content<Tile.Tile8>},
  bias_exact: {Buffer.content(~F32,bias_model) == values_bias(~Logical,settings) : Buffer.Content<F32>}) ->
  BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),
    Model.contents(~F32,BandPipeline.run(~custom,plan,mode,context(~Logical,settings),Model.pack(~F32,cores),BandHalo.pack_owners(~F32,halos),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model)))):
  match plan cores:
    case Bands.Rows{+first,+rows} Model.Leaf{+core,cf,cr}:
      match halos:
        case BandHalo.OwnedLeaf{+above,+below,+window,hf,hr}: leaf_rule(settings,first,rows,core,above,below,window,weight_model,bias_model,inputs,weight_exact,bias_exact)
        case BandHalo.OwnedFork{left_halo,right_halo}: match inputs:
    case Bands.Split{+left_plan,+right_plan} Model.Fork{+left,+right}:
      match halos:
        case BandHalo.OwnedFork{+left_halo,+right_halo}:
          Shape.Both{+left_inputs,+right_inputs} = inputs
          %Equal.sym(Bands.Bands(~F32),BandPipeline.run(~custom,Bands.Split{left_plan,right_plan},mode,context(~Logical,settings),Model.pack(~F32,Model.Fork{left,right}),BandHalo.pack_owners(~F32,BandHalo.OwnedFork{left_halo,right_halo}),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model)),
            fork_run(~custom,left_plan,right_plan,context(~Logical,settings),left,right,left_halo,right_halo,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weight_model)),Clone.clone(~F32,Buffer.pack(~F32,bias_model))),
            values_fork_execution(~custom,left_plan,right_plan,mode,context(~Logical,settings),left,right,left_halo,right_halo,weight_model,bias_model)) :
            BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),Model.contents(~F32,_))
          split_values(~custom,~Logical,~value,settings,left_plan,right_plan,left,right,left_halo,right_halo,weight_model,bias_model,
            (left_weights => left_bias => weight_exact => bias_exact =>
              values_run_values(~custom,~Logical,~value,~premise,~leaf_rule,settings,left_plan,BandPipeline.parallel_plan(context(~Logical,settings),left_plan),left,left_halo,left_weights,left_bias,left_inputs,weight_exact,bias_exact)),
            (right_weights => right_bias => weight_exact => bias_exact =>
              values_run_values(~custom,~Logical,~value,~premise,~leaf_rule,settings,right_plan,BandPipeline.parallel_plan(context(~Logical,settings),right_plan),right,right_halo,right_weights,right_bias,right_inputs,weight_exact,bias_exact)),
            weight_exact,bias_exact)
        case BandHalo.OwnedLeaf{above,below,window,hf,hr}: match inputs:
    case Bands.Rows{first,rows} Model.Fork{left,right}: match inputs:
    case Bands.Split{left_plan,right_plan} Model.Leaf{core,cf,cr}: match inputs:

def observed_content(~Element: Data,+model: Buffer.Model<Element>,-buffer: Storage.Buffer<Element>,
  equation: {Buffer.pack(~Element,model) == buffer : Storage.Buffer<Element>}) ->
  {Buffer.content(~Element,model) == Buffer.contents(~Element,buffer) : Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,model),Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.contents(~Element,buffer),
    Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)),
    Equal.cong(Storage.Buffer<Element>,Buffer.Content<Element>,owner => Buffer.contents(~Element,owner),Buffer.pack(~Element,model),buffer,equation))

def run_observed_values(~custom: F32 -> F32,~Logical: Data,~value: Logical -> U32 -> F32,~premise: Premise(~Logical),~leaf_rule: LeafRule(~custom,~Logical,~value,~premise),
  +settings: Settings<Logical>,+plan: Bands.Plan,+mode: Bool,-cores: Bands.Bands(~F32),-halos: BandWindow.Halos(~F32),-weights_owner: Storage.Buffer<Tile.Tile8>,-bias_owner: Storage.Buffer<F32>,
  cores_observation: Model.Observation<F32,cores>,halos_observation: BandHalo.OwnerObservation<F32,halos>,
  weights_observation: Buffer.Observation<Tile.Tile8,weights_owner>,bias_observation: Buffer.Observation<F32,bias_owner>,
  inputs: NativeInputs(~Logical,~premise,settings,plan,cores,halos),
  weight_exact: {Buffer.contents(~Tile.Tile8,weights_owner) == values_weights(~Logical,settings) : Buffer.Content<Tile.Tile8>},
  bias_exact: {Buffer.contents(~F32,bias_owner) == values_bias(~Logical,settings) : Buffer.Content<F32>}) ->
  BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,context(~Logical,settings),cores,halos,weights_owner,bias_owner))):
  Model.Observation{+core_model,+cores_equal} = cores_observation
  BandHalo.OwnerObservation{+halo_model,+halos_equal} = halos_observation
  Buffer.Observation{+weight_model,+weights_equal} = weights_observation
  Buffer.Observation{+bias_model,+bias_equal} = bias_observation
  %cores_equal : BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,context(~Logical,settings),_,halos,weights_owner,bias_owner)))
  %halos_equal : BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,context(~Logical,settings),Model.pack(~F32,core_model),_,weights_owner,bias_owner)))
  %weights_equal : BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,context(~Logical,settings),Model.pack(~F32,core_model),BandHalo.pack_owners(~F32,halo_model),_,bias_owner)))
  %bias_equal : BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,context(~Logical,settings),Model.pack(~F32,core_model),BandHalo.pack_owners(~F32,halo_model),Buffer.pack(~Tile.Tile8,weight_model),_)))
  values_run_values(~custom,~Logical,~value,~premise,~leaf_rule,settings,plan,mode,core_model,halo_model,weight_model,bias_model,
    observed_inputs(~Logical,~premise,settings,plan,core_model,halo_model,cores,halos,cores_equal,halos_equal,inputs),
    Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,weight_model),Buffer.contents(~Tile.Tile8,weights_owner),values_weights(~Logical,settings),observed_content(~Tile.Tile8,weight_model,weights_owner,weights_equal),weight_exact),
    Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,bias_model),Buffer.contents(~F32,bias_owner),values_bias(~Logical,settings),observed_content(~F32,bias_model,bias_owner,bias_equal),bias_exact))

# Native owners are observed once. Their original certificates are retained;
# the conclusion refers to the actual caller, rather than a chosen model.
def run_native_values(~custom: F32 -> F32,~Logical: Data,~value: Logical -> U32 -> F32,~premise: Premise(~Logical),~leaf_rule: LeafRule(~custom,~Logical,~value,~premise),
  +settings: Settings<Logical>,+plan: Bands.Plan,+mode: Bool,cores: Bands.Bands(~F32),halos: BandWindow.Halos(~F32),weights_owner: Storage.Buffer<Tile.Tile8>,bias_owner: Storage.Buffer<F32>,
  inputs: NativeInputs(~Logical,~premise,settings,plan,cores,halos),weight_exact: {Buffer.contents(~Tile.Tile8,weights_owner) == values_weights(~Logical,settings) : Buffer.Content<Tile.Tile8>},
  bias_exact: {Buffer.contents(~F32,bias_owner) == values_bias(~Logical,settings) : Buffer.Content<F32>}) ->
  BandValues.Values(~F32,~Logical,~value,values_logical(~Logical,settings),geometry(~Logical,settings),Model.contents(~F32,BandPipeline.run(~custom,plan,mode,context(~Logical,settings),cores,halos,weights_owner,bias_owner))):
  run_observed_values(~custom,~Logical,~value,~premise,~leaf_rule,settings,plan,mode,cores,halos,weights_owner,bias_owner,
    Model.observe(~F32,cores),BandHalo.observe_owners(~F32,halos),Buffer.observe(~Tile.Tile8,weights_owner),Buffer.observe(~F32,bias_owner),inputs,weight_exact,bias_exact)

# Convolution supplies the generic schedule's leaf premise. The proof frame
# contains only data and evidence; executable functions stay direct templates.
type Frame<-Logical: Data> is Data:
  Frame{convolution_logical: Logical,activation: Activations.Activation,dimensions: KernelShape.Dimensions,images: U32,parallel: Bool,convolution_limits: Machine.Limits,convolution_capacity: U32,
    weights: List<&2,F32>,bias: List<&2,F32>,
    weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
    convolution_domain: BandGather.Domain<Leaf.input_geometry(images,dimensions),convolution_limits,convolution_capacity>,
    valid: {KernelShape.valid(dimensions,Caller.input_length(dimensions),ConvSpec.length(weights),ConvSpec.length(bias)) == True{} : Bool},
    supported: {KernelShape.supports_packed(convolution_limits,dimensions,Caller.input_length(dimensions),ConvSpec.length(weights)) == True{} : Bool},
    output_room: {Nat.is_le(BandLeafValues.output_elements(images,dimensions),U32.to_nat(Machine.capacity(convolution_limits))) == True{} : Bool},
    eligible: {TensorConvolution.uses_band_windows(TensorConvolution.Geometry{images,True{},dimensions}) == True{} : Bool}}

def dimensions(~Logical: Data,frame: Frame<Logical>) -> KernelShape.Dimensions:
  Frame{convolution_logical,activation,dimensions,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  dimensions

def convolution_logical(~Logical: Data,frame: Frame<Logical>) -> Logical:
  Frame{convolution_logical,activation,dimensions,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  convolution_logical

def input_geometry(~Logical: Data,frame: Frame<Logical>) -> Bands.Geometry:
  Frame{convolution_logical,activation,dimensions,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  Leaf.input_geometry(images,dimensions)

def window_geometry(~Logical: Data,frame: Frame<Logical>) -> BandWindow.Geometry:
  Frame{convolution_logical,activation,dimensions,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  BandLeafValues.geometry(images,dimensions)

def convolution_limits(~Logical: Data,frame: Frame<Logical>) -> Machine.Limits:
  Frame{convolution_logical,activation,dimensions,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  convolution_limits

def convolution_capacity(~Logical: Data,frame: Frame<Logical>) -> U32:
  Frame{convolution_logical,activation,dimensions,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  convolution_capacity

def convolution_domain(~Logical: Data,+frame: Frame<Logical>) -> BandGather.Domain<input_geometry(~Logical,frame),convolution_limits(~Logical,frame),convolution_capacity(~Logical,frame)>:
  Frame{convolution_logical,activation,dimensions,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  convolution_domain

def input_height(~Logical: Data,frame: Frame<Logical>) -> U32:
  Frame{convolution_logical,activation,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  U32.from_nat(h)

def input_height_room(~Logical: Data,+frame: Frame<Logical>) -> {Nat.is_le(U32.to_nat(input_height(~Logical,frame)),U32.to_nat(Machine.capacity(convolution_limits(~Logical,frame)))) == True{} : Bool}:
  Frame{convolution_logical,activation,KernelShape.Dimensions{+ci,+co,+h,+width,+kernel,+stride,+padding},+images,parallel,+convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  BandLeafValues.right_factor_room(U32.to_nat(images),U32.to_nat(U32.from_nat(h)),Machine.capacity(convolution_limits),Caller.images_positive(images,True{},ci,co,h,width,kernel,stride,padding,eligible),
    BandGather.rows_room(Leaf.input_geometry(images,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding}),convolution_limits,convolution_capacity,convolution_domain))

def result_value(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,frame: Frame<Logical>,index: U32) -> F32:
  Frame{convolution_logical,activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  BandConvolutionValues.value(~custom,~Logical,~original,BandConvolutionValues.Context{convolution_logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias},index)

def output_context(~Logical: Data,frame: Frame<Logical>) -> BandConvolutionValues.Context<Logical>:
  Frame{convolution_logical,activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  BandConvolutionValues.Context{convolution_logical,activation,ci,co,height,width,kernel,stride,padding,weights,bias}

def Related(~Logical: Data,before: BandConvolutionValues.Context<Logical>,after: Frame<Logical>) -> Data:
  {before == output_context(~Logical,after) : BandConvolutionValues.Context<Logical>}

def agreement(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+before: BandConvolutionValues.Context<Logical>,+after: Frame<Logical>,+related: Related(~Logical,before,after),+index: U32) ->
  {BandConvolutionValues.value(~custom,~Logical,~original,before,index) == result_value(~custom,~Logical,~original,after,index) : F32}:
  Frame{+convolution_logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+width,+kernel,+stride,+padding},+images,+parallel,+convolution_limits,+convolution_capacity,+weights,+bias,+we,+be,+convolution_domain,+valid,+supported,+room,+eligible} = after
  +after = {Frame{convolution_logical,activation,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} : Frame<Logical>}
  %Equal.sym(BandConvolutionValues.Context<Logical>,before,output_context(~Logical,after),related) : {BandConvolutionValues.value(~custom,~Logical,~original,_,index) == result_value(~custom,~Logical,~original,after,index) : F32}
  {==}

def settings(~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>) -> Settings<Frame<Logical>>:
  Frame{+convolution_logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+width,+kernel,+stride,+padding},+images,+parallel,convolution_limits,convolution_capacity,+weights,+bias,+we,be,convolution_domain,valid,supported,room,eligible} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding} : KernelShape.Dimensions}
  +inner = U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))
  Settings{Frame{convolution_logical,activation,dimensions,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible},
    Bands.Geometry{images,U32.from_nat(co),U32.from_nat(KernelShape.axis_size(h,kernel,stride,padding)),U32.from_nat(KernelShape.axis_size(width,kernel,stride,padding))},
    TensorConvolution.band_context(TensorConvolution.Geometry{images,True{},dimensions},parallel,activation),
    Buffer.Content{Leaf.expected_weights(~Logical,~original,convolution_logical,dimensions,weights,bias,0,we),U32.to_nat((((U32.from_nat(co) + 7) / 8) * inner : U32))},
    Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,be),ConvSpec.length(bias)}}

def ActiveInput(~Logical: Data,~original: Logical -> U32 -> F32,frame: Frame<Logical>,+first: U32,+rows: U32,core: Buffer.Content<F32>,above: Buffer.Content<F32>,below: Buffer.Content<F32>,window: BandWindow.RowWindow) -> Data:
  Frame{+convolution_logical,activation,+dimensions,+images,parallel,+convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  Shape.Both<TensorBandWindow.CoreGeometry<images,U32.from_nat(Leaf.height(dimensions)),first,rows,Machine.capacity(convolution_limits)>,
    Shape.Both<BandHalo.RangeValues(~F32,~Logical,~original,convolution_logical,Leaf.input_geometry(images,dimensions),BandWindow.Rows{first,rows},core),
      BandHalo.TreeValues(~F32,~Logical,~original,convolution_logical,Leaf.input_geometry(images,dimensions),BandHalo.Leaf{above,below,window,first,rows})>>

def FrameInput(~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+first: U32,+rows: U32,core: Buffer.Content<F32>,above: Buffer.Content<F32>,below: Buffer.Content<F32>,+window: BandWindow.RowWindow) -> Data:
  Frame{convolution_logical,activation,+shape,+images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  +frame = {Frame{convolution_logical,activation,shape,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} : Frame<Logical>}
  Shape.Both<{window == BandWindow.window(BandLeafValues.geometry(images,shape),first,rows) : BandWindow.RowWindow},
    Boolean.When(BandLeafValues.active(shape,first,rows),ActiveInput(~Logical,~original,frame,first,rows,core,above,below,window))>

def premise(~Logical: Data,~original: Logical -> U32 -> F32,configuration: Settings<Frame<Logical>>,first: U32,rows: U32,core: Buffer.Content<F32>,above: Buffer.Content<F32>,below: Buffer.Content<F32>,window: BandWindow.RowWindow) -> Data:
  +configuration = configuration
  Shape.Both<{configuration == settings(~Logical,~original,values_logical(~Frame<Logical>,configuration)) : Settings<Frame<Logical>>},
    FrameInput(~Logical,~original,values_logical(~Frame<Logical>,configuration),first,rows,core,above,below,window)>

def LeafValues(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  core: Buffer.Model<F32>,above: Buffer.Model<F32>,below: Buffer.Model<F32>,weight_model: Buffer.Model<Tile.Tile8>,bias_model: Buffer.Model<F32>) -> Data:
  +configuration = settings(~Logical,~original,frame)
  BandValues.Values(~F32,~Frame<Logical>,~result_value(~custom,~Logical,~original),frame,geometry(~Frame<Logical>,configuration),
    Model.contents(~F32,BandPipeline.leaf(~custom,context(~Frame<Logical>,configuration),first,rows,Buffer.pack(~F32,core),
      BandWindow.Halo{Buffer.pack(~F32,above),Buffer.pack(~F32,below),window},Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model))))

# Only the convolution_logical context changes here; the executable convolution leaf is
# exactly the one refined by band_leaf_caller and band_leaf_inactive.
def FrameWindowValues(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,frame: Frame<Logical>,first: U32,rows: U32,window: BandWindow.RowWindow,
  core: Buffer.Model<F32>,above: Buffer.Model<F32>,below: Buffer.Model<F32>,weight_model: Buffer.Model<Tile.Tile8>,bias_model: Buffer.Model<F32>) -> Data:
  Frame{convolution_logical,activation,shape,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  BandLeafValues.WindowValues(~custom,~Logical,~original,convolution_logical,activation,images,shape,weights,bias,first,rows,window,core,above,below,weight_model,bias_model)

def window_values(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  values: FrameWindowValues(~custom,~Logical,~original,frame,first,rows,window,core,above,below,weight_model,bias_model)) ->
  LeafValues(~custom,~Logical,~original,frame,first,rows,window,core,above,below,weight_model,bias_model):
  Frame{+convolution_logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+width,+kernel,+stride,+padding},+images,+parallel,+convolution_limits,+convolution_capacity,+weights,+bias,+we,+be,+convolution_domain,+valid,+supported,+room,+eligible} = frame
  +frame = {Frame{convolution_logical,activation,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} : Frame<Logical>}
  BandValues.values(~F32,~BandConvolutionValues.Context<Logical>,~Frame<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),~result_value(~custom,~Logical,~original),~Related(~Logical),~agreement(~custom,~Logical,~original),
    BandConvolutionValues.Context{convolution_logical,activation,ci,co,h,width,kernel,stride,padding,weights,bias},frame,{==},geometry(~Frame<Logical>,settings(~Logical,~original,frame)),
    Model.contents(~F32,BandPipeline.leaf(~custom,context(~Frame<Logical>,settings(~Logical,~original,frame)),first,rows,Buffer.pack(~F32,core),
      BandWindow.Halo{Buffer.pack(~F32,above),Buffer.pack(~F32,below),window},Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model))),values)

def actual_window(~Logical: Data,frame: Frame<Logical>,first: U32,rows: U32) -> BandWindow.RowWindow:
  Frame{convolution_logical,activation,shape,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  BandWindow.window(BandLeafValues.geometry(images,shape),first,rows)

def enabled(~Logical: Data,frame: Frame<Logical>,first: U32,rows: U32) -> Bool:
  BandLeafValues.active(dimensions(~Logical,frame),first,rows)

def expected_weights(~Logical: Data,~original: Logical -> U32 -> F32,frame: Frame<Logical>,first: U32) -> Tree.Tree<Tile.Tile8>:
  Frame{convolution_logical,activation,shape,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  Leaf.expected_weights(~Logical,~original,convolution_logical,shape,weights,bias,first,we)

def model_tree(~Element: Data,+model: Buffer.Model<Element>) -> {Buffer.content_tree(~Element,Buffer.content(~Element,model)) == Buffer.tree(~Element,model) : Tree.Tree<Element>}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def expected_bias(~Logical: Data,frame: Frame<Logical>) -> Tree.Tree<F32>:
  Frame{convolution_logical,activation,shape,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,be)

def bias_tree_exact(~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+bias_model: Buffer.Model<F32>,
  exact: {Buffer.content(~F32,bias_model) == values_bias(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  {Buffer.tree(~F32,bias_model) == expected_bias(~Logical,frame) : Tree.Tree<F32>}:
  Frame{+convolution_logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+width,+kernel,+stride,+padding},+images,+parallel,+convolution_limits,+convolution_capacity,+weights,+bias,+we,+be,+convolution_domain,+valid,+supported,+room,+eligible} = frame
  +frame = {Frame{convolution_logical,activation,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} : Frame<Logical>}
  %model_tree(~F32,bias_model) : {_ == expected_bias(~Logical,frame) : Tree.Tree<F32>}
  Equal.cong(Buffer.Content<F32>,Tree.Tree<F32>,content => Buffer.content_tree(~F32,content),Buffer.content(~F32,bias_model),values_bias(~Frame<Logical>,settings(~Logical,~original,frame)),exact)

def weight_tree_exact(~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+first: U32,+weight_model: Buffer.Model<Tile.Tile8>,
  exact: {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<Tile.Tile8>}) ->
  {Buffer.tree(~Tile.Tile8,weight_model) == expected_weights(~Logical,~original,frame,first) : Tree.Tree<Tile.Tile8>}:
  Frame{+convolution_logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+width,+kernel,+stride,+padding},+images,+parallel,+convolution_limits,+convolution_capacity,+weights,+bias,+we,+be,+convolution_domain,+valid,+supported,+room,+eligible} = frame
  +frame = {Frame{convolution_logical,activation,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} : Frame<Logical>}
  %model_tree(~Tile.Tile8,weight_model) : {_ == expected_weights(~Logical,~original,frame,first) : Tree.Tree<Tile.Tile8>}
  Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,content => Buffer.content_tree(~Tile.Tile8,content),
    Buffer.content(~Tile.Tile8,weight_model),values_weights(~Frame<Logical>,settings(~Logical,~original,frame)),exact)

def CoreGeometry(~Logical: Data,frame: Frame<Logical>,first: U32,rows: U32) -> Data:
  Frame{convolution_logical,activation,shape,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  TensorBandWindow.CoreGeometry<images,U32.from_nat(Leaf.height(shape)),first,rows,Machine.capacity(convolution_limits)>

def SourceValues(~Logical: Data,~original: Logical -> U32 -> F32,frame: Frame<Logical>,+first: U32,+rows: U32,core: Buffer.Content<F32>,above: Buffer.Content<F32>,below: Buffer.Content<F32>) -> Data:
  Frame{+convolution_logical,activation,+shape,+images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  Shape.Both<BandHalo.RangeValues(~F32,~Logical,~original,convolution_logical,Leaf.input_geometry(images,shape),BandWindow.Rows{first,rows},core),
    BandHalo.TreeValues(~F32,~Logical,~original,convolution_logical,Leaf.input_geometry(images,shape),BandHalo.Leaf{above,below,BandWindow.window(BandLeafValues.geometry(images,shape),first,rows),first,rows})>

def active_sources(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+first: U32,+rows: U32,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  core_geometry: CoreGeometry(~Logical,frame,first,rows),
  sources: SourceValues(~Logical,~original,frame,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below)),
  +weight_exact: {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<Tile.Tile8>},
  bias_exact: {Buffer.content(~F32,bias_model) == values_bias(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<F32>},
  guard: {enabled(~Logical,frame,first,rows) == True{} : Bool}) ->
  FrameWindowValues(~custom,~Logical,~original,frame,first,rows,actual_window(~Logical,frame,first,rows),core,above,below,weight_model,bias_model):
  Frame{+convolution_logical,+activation,+shape,+images,+parallel,+convolution_limits,+convolution_capacity,+weights,+bias,+we,+be,+convolution_domain,+valid,+supported,+room,+eligible} = frame
  Shape.Both{core_values,halo_values} = sources
  +frame = {Frame{convolution_logical,activation,shape,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} : Frame<Logical>}
  +weight_tree = weight_tree_exact(~Logical,~original,frame,first,weight_model,weight_exact)
  +bias_tree = bias_tree_exact(~Logical,~original,frame,bias_model,bias_exact)
  Leaf.active_checked(~custom,~Logical,~original,convolution_logical,activation,convolution_limits,images,shape,weights,bias,first,rows,convolution_capacity,convolution_domain,core_geometry,valid,supported,room,eligible,
    we,be,core,above,below,weight_model,bias_model,weight_tree,bias_tree,core_values,halo_values,guard)

def active_frame(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+first: U32,+rows: U32,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  input: ActiveInput(~Logical,~original,frame,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),actual_window(~Logical,frame,first,rows)),
  weight_exact: {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<Tile.Tile8>},
  bias_exact: {Buffer.content(~F32,bias_model) == values_bias(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<F32>},
  guard: {enabled(~Logical,frame,first,rows) == True{} : Bool}) ->
  FrameWindowValues(~custom,~Logical,~original,frame,first,rows,actual_window(~Logical,frame,first,rows),core,above,below,weight_model,bias_model):
  Frame{convolution_logical,activation,shape,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  Shape.Both{core_geometry,sources} = input
  +frame = {Frame{convolution_logical,activation,shape,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} : Frame<Logical>}
  active_sources(~custom,~Logical,~original,frame,first,rows,core,above,below,weight_model,bias_model,core_geometry,sources,weight_exact,bias_exact,guard)

def inactive_frame(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+first: U32,+rows: U32,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  guard: {enabled(~Logical,frame,first,rows) == False{} : Bool}) ->
  FrameWindowValues(~custom,~Logical,~original,frame,first,rows,actual_window(~Logical,frame,first,rows),core,above,below,weight_model,bias_model):
  Frame{+convolution_logical,activation,KernelShape.Dimensions{+ci,+co,+h,+width,+kernel,+stride,+padding},+images,parallel,convolution_limits,convolution_capacity,+weights,+bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  +shape = {KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding} : KernelShape.Dimensions}
  +problem = Caller.problem(~Logical,~original,convolution_logical,shape,weights,bias,0n)
  BandLeafValues.inactive_checked(~custom,~Logical,~original,convolution_logical,activation,convolution_limits,images,shape,weights,bias,first,rows,core,above,below,weight_model,bias_model,
    Caller.images_positive(images,True{},ci,co,h,width,kernel,stride,padding,eligible),Caller.outputs_positive(problem,Caller.valid(~Logical,~original,convolution_logical,shape,weights,bias,0n,valid)),room,guard)

def branch_frame(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+active: Bool,+frame: Frame<Logical>,+first: U32,+rows: U32,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  input: Boolean.When(active,ActiveInput(~Logical,~original,frame,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),actual_window(~Logical,frame,first,rows))),
  weight_exact: {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<Tile.Tile8>},
  bias_exact: {Buffer.content(~F32,bias_model) == values_bias(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<F32>},
  guard: {enabled(~Logical,frame,first,rows) == active : Bool}) ->
  FrameWindowValues(~custom,~Logical,~original,frame,first,rows,actual_window(~Logical,frame,first,rows),core,above,below,weight_model,bias_model):
  match active:
    case False{}: inactive_frame(~custom,~Logical,~original,frame,first,rows,core,above,below,weight_model,bias_model,guard)
    case True{}: active_frame(~custom,~Logical,~original,frame,first,rows,core,above,below,weight_model,bias_model,input,weight_exact,bias_exact,guard)

def leaf_frame(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+first: U32,+rows: U32,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+window: BandWindow.RowWindow,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  input: FrameInput(~Logical,~original,frame,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),window),
  weight_exact: {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<Tile.Tile8>},
  bias_exact: {Buffer.content(~F32,bias_model) == values_bias(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  LeafValues(~custom,~Logical,~original,frame,first,rows,window,core,above,below,weight_model,bias_model):
  Frame{convolution_logical,activation,shape,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  Shape.Both{+window_exact,active_input} = input
  +frame = {Frame{convolution_logical,activation,shape,images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} : Frame<Logical>}
  %Equal.sym(BandWindow.RowWindow,window,actual_window(~Logical,frame,first,rows),window_exact) : LeafValues(~custom,~Logical,~original,frame,first,rows,_,core,above,below,weight_model,bias_model)
  window_values(~custom,~Logical,~original,frame,first,rows,actual_window(~Logical,frame,first,rows),core,above,below,weight_model,bias_model,
    branch_frame(~custom,~Logical,~original,enabled(~Logical,frame,first,rows),frame,first,rows,core,above,below,weight_model,bias_model,
      %window_exact : Boolean.When(enabled(~Logical,frame,first,rows),ActiveInput(~Logical,~original,frame,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),_))
      active_input,weight_exact,bias_exact,{==}))

def LeafResult(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+configuration: Settings<Frame<Logical>>,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  core: Buffer.Model<F32>,above: Buffer.Model<F32>,below: Buffer.Model<F32>,weight_model: Buffer.Model<Tile.Tile8>,bias_model: Buffer.Model<F32>) -> Data:
  BandValues.Values(~F32,~Frame<Logical>,~result_value(~custom,~Logical,~original),values_logical(~Frame<Logical>,configuration),geometry(~Frame<Logical>,configuration),
    Model.contents(~F32,BandPipeline.leaf(~custom,context(~Frame<Logical>,configuration),first,rows,Buffer.pack(~F32,core),
      BandWindow.Halo{Buffer.pack(~F32,above),Buffer.pack(~F32,below),window},Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~F32,bias_model))))

def leaf_prototype(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+first: U32,+rows: U32,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+window: BandWindow.RowWindow,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  input: FrameInput(~Logical,~original,frame,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),window),
  weight_exact: {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<Tile.Tile8>},
  bias_exact: {Buffer.content(~F32,bias_model) == values_bias(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  LeafResult(~custom,~Logical,~original,settings(~Logical,~original,frame),first,rows,window,core,above,below,weight_model,bias_model):
  Frame{convolution_logical,activation,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible} = frame
  leaf_frame(~custom,~Logical,~original,Frame{convolution_logical,activation,KernelShape.Dimensions{ci,co,h,width,kernel,stride,padding},images,parallel,convolution_limits,convolution_capacity,weights,bias,we,be,convolution_domain,valid,supported,room,eligible},
    first,rows,core,above,below,window,weight_model,bias_model,input,weight_exact,bias_exact)

def leaf_rule(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+configuration: Settings<Frame<Logical>>,+first: U32,+rows: U32,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+window: BandWindow.RowWindow,+weight_model: Buffer.Model<Tile.Tile8>,+bias_model: Buffer.Model<F32>,
  input: premise(~Logical,~original,configuration,first,rows,Buffer.content(~F32,core),Buffer.content(~F32,above),Buffer.content(~F32,below),window),
  weight_exact: {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Frame<Logical>,configuration) : Buffer.Content<Tile.Tile8>},
  bias_exact: {Buffer.content(~F32,bias_model) == values_bias(~Frame<Logical>,configuration) : Buffer.Content<F32>}) ->
  LeafResult(~custom,~Logical,~original,configuration,first,rows,window,core,above,below,weight_model,bias_model):
  Shape.Both{+consistent,frame_input} = input
  +frame = values_logical(~Frame<Logical>,configuration)
  %Equal.sym(Settings<Frame<Logical>>,configuration,settings(~Logical,~original,frame),consistent) : LeafResult(~custom,~Logical,~original,_,first,rows,window,core,above,below,weight_model,bias_model)
  leaf_prototype(~custom,~Logical,~original,frame,first,rows,core,above,below,window,weight_model,bias_model,frame_input,
    %consistent : {Buffer.content(~Tile.Tile8,weight_model) == values_weights(~Frame<Logical>,_) : Buffer.Content<Tile.Tile8>}
    weight_exact,
    %consistent : {Buffer.content(~F32,bias_model) == values_bias(~Frame<Logical>,_) : Buffer.Content<F32>}
    bias_exact)

# The leaf rule is supplied by convolution's existing precision theorem.
# NativeInputs is still a producer obligation, not an arbitrary leaf oracle.
def convolution_run_values(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+configuration: Settings<Frame<Logical>>,+plan: Bands.Plan,+mode: Bool,
  cores: Bands.Bands(~F32),halos: BandWindow.Halos(~F32),weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  inputs: NativeInputs(~Frame<Logical>,~premise(~Logical,~original),configuration,plan,cores,halos),
  weight_exact: {Buffer.contents(~Tile.Tile8,weights) == values_weights(~Frame<Logical>,configuration) : Buffer.Content<Tile.Tile8>},
  bias_exact: {Buffer.contents(~F32,bias) == values_bias(~Frame<Logical>,configuration) : Buffer.Content<F32>}) ->
  BandValues.Values(~F32,~Frame<Logical>,~result_value(~custom,~Logical,~original),values_logical(~Frame<Logical>,configuration),geometry(~Frame<Logical>,configuration),
    Model.contents(~F32,BandPipeline.run(~custom,plan,mode,context(~Frame<Logical>,configuration),cores,halos,weights,bias))):
  run_native_values(~custom,~Frame<Logical>,~result_value(~custom,~Logical,~original),~premise(~Logical,~original),~leaf_rule(~custom,~Logical,~original),
    configuration,plan,mode,cores,halos,weights,bias,inputs,weight_exact,bias_exact)

# Reorder is connected through its existing content theorem. The caller's
# guard proves that the natural and native reduction lengths coincide.
def channels(~Logical: Data,frame: Frame<Logical>) -> U32:
  Frame{logical,activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  U32.from_nat(co)

def inner(~Logical: Data,frame: Frame<Logical>) -> U32:
  Frame{logical,activation,KernelShape.Dimensions{ci,co,height,width,+kernel,stride,padding},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))

def source_tree(~Logical: Data,frame: Frame<Logical>) -> Tree.Tree<F32>:
  Frame{logical,activation,shape,images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,we)

def reordered(~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+weights_model: Buffer.Model<F32>,
  +source_exact: {Buffer.tree(~F32,weights_model) == source_tree(~Logical,frame) : Tree.Tree<F32>}) ->
  {Buffer.contents(~Tile.Tile8,GemmBuffer.reorder(channels(~Logical,frame),inner(~Logical,frame),Buffer.pack(~F32,weights_model))) ==
    values_weights(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<Tile.Tile8>}:
  Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding},+images,+parallel,+limits,+capacity,+weights,+bias,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {Frame{logical,activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} : Frame<Logical>}
  +shape = {KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} : KernelShape.Dimensions}
  +problem = Caller.problem(~Logical,~original,logical,shape,weights,bias,0n)
  +channels = U32.from_nat(co)
  +inner = U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))
  +length = U32.to_nat((((channels + 7) / 8) * inner : U32))
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,GemmBuffer.reorder(channels,inner,Buffer.pack(~F32,weights_model))),
    Buffer.Content{WeightsLaws.run(channels,inner,Buffer.tree(~F32,weights_model)),length},KernelWeightsBufferRefinement.reordered(channels,inner,weights_model)) :
    {_ == values_weights(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<Tile.Tile8>}
  %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,weights_model),source_tree(~Logical,frame),source_exact) :
    {Buffer.Content{WeightsLaws.run(channels,inner,_),length} == values_weights(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<Tile.Tile8>}
  %Equal.sym(U32,inner,Coordinates.native_inner(problem),BandLeafValues.inner_word(limits,problem,Caller.valid(~Logical,~original,logical,shape,weights,bias,0n,valid),Caller.supported(~Logical,~original,logical,limits,shape,weights,bias,0n,supported))) :
    {Buffer.Content{WeightsLaws.run(channels,_,source_tree(~Logical,frame)),length} == values_weights(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<Tile.Tile8>}
  {==}

# Produce the leaf premises from the inspected cores and actual halo model.
# Active cores derive their geometry and halo bounds. Inactive cores require
# neither positive rows nor image-inside evidence.
def above(window: BandWindow.RowWindow) -> BandWindow.Rows:
  BandWindow.RowWindow{output,above,below,read} = window
  above

def leaf_model_exact(+geometry: Bands.Geometry,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+source: Model.Content<F32>) ->
  {BandHalo.leaf_model(~F32,geometry,0.0,first,rows,window,source) == BandHalo.Leaf{
    BandHalo.extract_refinement_result(~F32,geometry,0.0,above(window),source),BandHalo.extract_refinement_result(~F32,geometry,0.0,BandHalo.below(window),source),window,first,rows} : BandHalo.Tree<F32>}:
  BandWindow.RowWindow{output,above,below,read} = window
  {==}

def active_leaf(~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+first: U32,+rows: U32,+core: Buffer.Content<F32>,+source: Model.Model<F32>,
  +shape: BandShape.WellFormed<F32,input_geometry(~Logical,frame),Model.content(~F32,source)>,
  +values: BandValues.Values(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),Model.content(~F32,source)),
  +core_values: BandHalo.RangeValues(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),BandWindow.Rows{first,rows},core),
  guard: {enabled(~Logical,frame,first,rows) == True{} : Bool}) ->
  ActiveInput(~Logical,~original,frame,first,rows,core,
    BandHalo.extract_refinement_result(~F32,input_geometry(~Logical,frame),0.0,above(actual_window(~Logical,frame,first,rows)),Model.content(~F32,source)),
    BandHalo.extract_refinement_result(~F32,input_geometry(~Logical,frame),0.0,BandHalo.below(actual_window(~Logical,frame,first,rows)),Model.content(~F32,source)),actual_window(~Logical,frame,first,rows)):
  Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding},+images,+parallel,+limits,+capacity,+weights,+bias,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  Buffer.Content{+tree,+length} = core
  Shape.Both{+ready,Shape.Both{+metadata,+points}} = core_values
  +frame = {Frame{logical,activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} : Frame<Logical>}
  +core = {Buffer.Content{tree,length} : Buffer.Content<F32>}
  +coordinates = TensorBandWindow.active_bounds_coordinates(window_geometry(~Logical,frame),first,rows,Buffer.content_length(~F32,core),convolution_limits(~Logical,frame),convolution_capacity(~Logical,frame),convolution_domain(~Logical,frame),
    input_height_room(~Logical,frame),metadata,guard)
  Shape.Both{coordinates,Shape.Both{Shape.Both{ready,Shape.Both{metadata,points}},
    %leaf_model_exact(input_geometry(~Logical,frame),first,rows,actual_window(~Logical,frame,first,rows),Model.content(~F32,source)) :
      BandHalo.TreeValues(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),_)
    BandHalo.leaf(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),0.0,first,rows,actual_window(~Logical,frame,first,rows),source,
      convolution_limits(~Logical,frame),convolution_capacity(~Logical,frame),convolution_domain(~Logical,frame),shape,values,
      TensorBandWindow.active_bounds_window(window_geometry(~Logical,frame),first,rows,convolution_limits(~Logical,frame),convolution_capacity(~Logical,frame),convolution_domain(~Logical,frame),coordinates))}}

def leaf_branch(~Logical: Data,~original: Logical -> U32 -> F32,+active: Bool,+frame: Frame<Logical>,+first: U32,+rows: U32,+core: Buffer.Content<F32>,+source: Model.Model<F32>,
  shape: BandShape.WellFormed<F32,input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),Model.content(~F32,source)),
  core_values: BandHalo.RangeValues(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),BandWindow.Rows{first,rows},core),
  guard: {enabled(~Logical,frame,first,rows) == active : Bool}) ->
  Boolean.When(active,ActiveInput(~Logical,~original,frame,first,rows,core,
    BandHalo.extract_refinement_result(~F32,input_geometry(~Logical,frame),0.0,above(actual_window(~Logical,frame,first,rows)),Model.content(~F32,source)),
    BandHalo.extract_refinement_result(~F32,input_geometry(~Logical,frame),0.0,BandHalo.below(actual_window(~Logical,frame,first,rows)),Model.content(~F32,source)),actual_window(~Logical,frame,first,rows))):
  match active:
    case False{}: Unit{}
    case True{}: active_leaf(~Logical,~original,frame,first,rows,core,source,shape,values,core_values,guard)

def leaf(~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+first: U32,+rows: U32,+tree: Tree.Tree<F32>,+length: Nat,+source: Model.Model<F32>,
  shape: BandShape.WellFormed<F32,input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),Model.content(~F32,source)),
  ready: Buffer.ContentReady<F32,Buffer.Content{tree,length}>,
  metadata: BandShape.Leaf(input_geometry(~Logical,frame),first,rows,length),
  points: BandValues.Values(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),Model.RowBand{Buffer.Content{tree,length},first,rows})) ->
  ContentInputs(~Frame<Logical>,~premise(~Logical,~original),settings(~Logical,~original,frame),Bands.Rows{first,rows},Model.RowBand{Buffer.Content{tree,length},first,rows},
    BandHalo.model(~F32,window_geometry(~Logical,frame),0.0,Bands.Rows{first,rows},Model.content(~F32,source))):
  Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding},+images,+parallel,+limits,+capacity,+weights,+bias,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {Frame{logical,activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} : Frame<Logical>}
  +geometry = input_geometry(~Logical,frame)
  +window = actual_window(~Logical,frame,first,rows)
  %Equal.sym(BandHalo.Tree<F32>,BandHalo.leaf_model(~F32,geometry,0.0,first,rows,window,Model.content(~F32,source)),
    BandHalo.Leaf{BandHalo.extract_refinement_result(~F32,geometry,0.0,above(window),Model.content(~F32,source)),BandHalo.extract_refinement_result(~F32,geometry,0.0,BandHalo.below(window),Model.content(~F32,source)),window,first,rows},
    leaf_model_exact(geometry,first,rows,window,Model.content(~F32,source))) :
    ContentInputs(~Frame<Logical>,~premise(~Logical,~original),settings(~Logical,~original,frame),Bands.Rows{first,rows},Model.RowBand{Buffer.Content{tree,length},first,rows},_)
  Shape.Both{{==},Shape.Both{{==},
    leaf_branch(~Logical,~original,enabled(~Logical,frame,first,rows),frame,first,rows,Buffer.Content{tree,length},source,shape,values,
      Shape.Both{ready,Shape.Both{metadata,points}},{==})}}

def modeled(~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+source: Model.Model<F32>,cores: Model.Content<F32>,
  +shape: BandShape.WellFormed<F32,input_geometry(~Logical,frame),Model.content(~F32,source)>,
  +values: BandValues.Values(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),Model.content(~F32,source)),
  metadata: BandShape.Contents(~F32,input_geometry(~Logical,frame),cores),ready: Model.ContentReady(~F32,cores),
  points: BandValues.Values(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),cores)) ->
  ContentInputs(~Frame<Logical>,~premise(~Logical,~original),settings(~Logical,~original,frame),Model.content_layout(~F32,cores),cores,
    BandHalo.model(~F32,window_geometry(~Logical,frame),0.0,Model.content_layout(~F32,cores),Model.content(~F32,source))):
  match cores:
    case Model.RowBand{Buffer.Content{tree,length},first,rows}: leaf(~Logical,~original,frame,first,rows,tree,length,source,shape,values,ready,metadata,points)
    case Model.Joined{left,right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_points,right_points} = points
      Shape.Both{modeled(~Logical,~original,frame,source,left,shape,values,left_metadata,left_ready,left_points),
        modeled(~Logical,~original,frame,source,right,shape,values,right_metadata,right_ready,right_points)}

# No externally supplied per-leaf correctness rule or halo readback premise.
# The source WellFormed and Values relations are the public producer inputs.
def inputs_planned(~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+source: Model.Model<F32>,
  +shape: BandShape.WellFormed<F32,input_geometry(~Logical,frame),Model.content(~F32,source)>,
  +values: BandValues.Values(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),Model.content(~F32,source))) ->
  ContentInputs(~Frame<Logical>,~premise(~Logical,~original),settings(~Logical,~original,frame),Model.layout(~F32,source),Model.content(~F32,source),
    BandHalo.model(~F32,window_geometry(~Logical,frame),0.0,Model.layout(~F32,source),Model.content(~F32,source))):
  BandShape.WellFormed{+metadata,+ready,+coverage} = shape
  %Equal.sym(Bands.Plan,Model.layout(~F32,source),Model.content_layout(~F32,Model.content(~F32,source)),Model.layout_matches_content(~F32,source)) :
    ContentInputs(~Frame<Logical>,~premise(~Logical,~original),settings(~Logical,~original,frame),_,Model.content(~F32,source),
      BandHalo.model(~F32,window_geometry(~Logical,frame),0.0,_,Model.content(~F32,source)))
  modeled(~Logical,~original,frame,source,Model.content(~F32,source),BandShape.WellFormed{metadata,ready,coverage},values,metadata,ready,values)

def PreparedInputs(~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+plan: Bands.Plan,content: BandHalo.Content<F32>) -> Data:
  BandHalo.Content{cores,halos} = content
  ContentInputs(~Frame<Logical>,~premise(~Logical,~original),settings(~Logical,~original,frame),plan,cores,halos)

def inputs_prepared(~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+source: Model.Model<F32>,
  shape: BandShape.WellFormed<F32,input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),Model.content(~F32,source))) ->
  PreparedInputs(~Logical,~original,frame,Model.layout(~F32,source),
    BandHalo.refinement_contents(~F32,BandWindow.prepare(~F32,window_geometry(~Logical,frame),0.0,Model.layout(~F32,source),Model.pack(~F32,source)))):
  +geometry = window_geometry(~Logical,frame)
  +plan = Model.layout(~F32,source)
  %Equal.sym(BandHalo.Content<F32>,BandHalo.refinement_contents(~F32,BandWindow.prepare(~F32,geometry,0.0,plan,Model.pack(~F32,source))),
    BandHalo.Content{Model.contents(~F32,Model.pack(~F32,source)),BandHalo.model(~F32,geometry,0.0,plan,Model.contents(~F32,Model.pack(~F32,source)))},
    BandHalo.prepare(~F32,geometry,0.0,plan,Model.pack(~F32,source))) : PreparedInputs(~Logical,~original,frame,plan,_)
  %Equal.sym(Model.Content<F32>,Model.contents(~F32,Model.pack(~F32,source)),Model.content(~F32,source),Model.packed_contents(~F32,source)) :
    PreparedInputs(~Logical,~original,frame,plan,BandHalo.Content{_,BandHalo.model(~F32,geometry,0.0,plan,_)})
  inputs_planned(~Logical,~original,frame,source,shape,values)

# The real prepare/plan/execute callers discharge the leaf schedule inputs.
# Input semantics and parameter contents come from the caller contracts.
def PipelineValues(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,bands: Bands.Bands(~F32)) -> Data:
  +configuration = settings(~Logical,~original,frame)
  BandValues.Values(~F32,~Frame<Logical>,~result_value(~custom,~Logical,~original),values_logical(~Frame<Logical>,configuration),geometry(~Frame<Logical>,configuration),Model.contents(~F32,bands))

def execution_prepared(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+plan: Bands.Plan,
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,owners: Bands.Bands(~F32) & BandWindow.Halos(~F32),
  inputs: PreparedInputs(~Logical,~original,frame,plan,BandHalo.refinement_contents(~F32,owners)),
  weight_exact: {Buffer.contents(~Tile.Tile8,weights) == values_weights(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<Tile.Tile8>},
  bias_exact: {Buffer.contents(~F32,bias) == values_bias(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  PipelineValues(~custom,~Logical,~original,frame,BandPipeline.prepared(~custom,context(~Frame<Logical>,settings(~Logical,~original,frame)),plan,weights,bias,owners)):
  (cores,halos) = owners
  +configuration = settings(~Logical,~original,frame)
  convolution_run_values(~custom,~Logical,~original,configuration,plan,BandPipeline.parallel_plan(context(~Frame<Logical>,configuration),plan),cores,halos,weights,bias,inputs,weight_exact,bias_exact)

def execution_planned(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == values_bias(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  PipelineValues(~custom,~Logical,~original,frame,BandPipeline.planned(~custom,context(~Frame<Logical>,settings(~Logical,~original,frame)),
    GemmBuffer.reorder(channels(~Logical,frame),inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),(Model.pack(~F32,source),Model.layout(~F32,source)))):
  Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding},+images,+parallel,+limits,+capacity,+weight_list,+bias_list,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {Frame{logical,activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,parallel,limits,capacity,weight_list,bias_list,we,be,domain,valid,supported,room,eligible} : Frame<Logical>}
  execution_prepared(~custom,~Logical,~original,frame,Model.layout(~F32,source),GemmBuffer.reorder(channels(~Logical,frame),inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),
    BandWindow.prepare(~F32,window_geometry(~Logical,frame),0.0,Model.layout(~F32,source),Model.pack(~F32,source)),inputs_prepared(~Logical,~original,frame,source,shape,values),
    reordered(~Logical,~original,frame,weights,weight_exact),
    Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,bias)),Buffer.content(~F32,bias),values_bias(~Frame<Logical>,settings(~Logical,~original,frame)),Buffer.packed_contents(~F32,bias),bias_exact))

def execute(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: Frame<Logical>,+source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,convolution_logical(~Logical,frame),input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == values_bias(~Frame<Logical>,settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  PipelineValues(~custom,~Logical,~original,frame,BandPipeline.execute(~custom,context(~Frame<Logical>,settings(~Logical,~original,frame)),Model.pack(~F32,source),GemmBuffer.reorder(channels(~Logical,frame),inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias))):
  %Equal.sym(Bands.Bands(~F32) & Bands.Plan,Bands.plan(~F32,Model.pack(~F32,source)),(Model.pack(~F32,source),Model.layout(~F32,source)),Model.plan(~F32,source)) :
    PipelineValues(~custom,~Logical,~original,frame,BandPipeline.planned(~custom,context(~Frame<Logical>,settings(~Logical,~original,frame)),GemmBuffer.reorder(channels(~Logical,frame),inner(~Logical,frame),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),_))
  execution_planned(~custom,~Logical,~original,frame,source,weights,bias,shape,values,weight_exact,bias_exact)

# The actual schedule's output partition is exactly the halo window's output
# partition. Buffer contents, activation and GEMM internals are irrelevant here.
def layout(bands: Bands.Bands(~F32)) -> Bands.Plan:
  Model.content_layout(~F32,Model.contents(~F32,bands))

def output_window(window: BandWindow.RowWindow) -> Bands.Plan:
  BandWindow.RowWindow{BandWindow.Rows{first,rows},above,below,read} = window
  Bands.Rows{first,rows}

def output_layout(halos: BandHalo.Owners<F32>) -> Bands.Plan:
  match halos:
    case BandHalo.OwnedLeaf{above,below,window,first,rows}: output_window(window)
    case BandHalo.OwnedFork{left,right}: Bands.Split{output_layout(left),output_layout(right)}

def Compatible(plan: Bands.Plan,cores: Model.Model<F32>,halos: BandHalo.Owners<F32>) -> Data:
  match plan cores:
    case Bands.Rows{first,rows} Model.Leaf{core,cf,cr}:
      match halos:
        case BandHalo.OwnedLeaf{above,below,window,hf,hr}: Unit
        case halos: Empty
    case Bands.Split{left_plan,right_plan} Model.Fork{left,right}:
      match halos:
        case BandHalo.OwnedFork{left_halo,right_halo}: Shape.Both<Compatible(left_plan,left,left_halo),Compatible(right_plan,right,right_halo)>
        case halos: Empty
    case plan cores: Empty

def windowed_layout(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: BandWindow.Geometry,+channels: U32,+inner: U32,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  -core: Storage.Buffer<F32>,-above: Storage.Buffer<F32>,-below: Storage.Buffer<F32>,-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>) ->
  {layout(BandPipeline.windowed(~custom,activation,geometry,channels,inner,first,rows,window,core,above,below,weights,bias)) == output_window(window) : Bands.Plan}:
  BandWindow.Geometry{Bands.Geometry{batch,ci,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{output_first,output_rows},above_range,below_range,BandWindow.Rows{window_first,window_rows}} = window
  {==}

def leaf_layout(~custom: F32 -> F32,+context: BandPipeline.Context,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  -core: Storage.Buffer<F32>,-above: Storage.Buffer<F32>,-below: Storage.Buffer<F32>,-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>) ->
  {layout(BandPipeline.leaf(~custom,context,first,rows,core,BandWindow.Halo{above,below,window},weights,bias)) == output_window(window) : Bands.Plan}:
  BandPipeline.Context{geometry,channels,inner,parallel,activation} = context
  windowed_layout(~custom,activation,geometry,channels,inner,first,rows,window,core,above,below,weights,bias)

def layout_fork_execution(~custom: F32 -> F32,+left_plan: Bands.Plan,+right_plan: Bands.Plan,mode: Bool,+context: BandPipeline.Context,
  +left: Model.Model<F32>,+right: Model.Model<F32>,+left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,
  -weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<F32>) ->
  {BandPipeline.run(~custom,Bands.Split{left_plan,right_plan},mode,context,Model.pack(~F32,Model.Fork{left,right}),BandHalo.pack_owners(~F32,BandHalo.OwnedFork{left_halo,right_halo}),weights,bias) ==
    fork_run(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,Clone.clone(~Tile.Tile8,weights),Clone.clone(~F32,bias)) : Bands.Bands(~F32)}:
  Each.fork(~Bands.Paired<Storage.Buffer<F32>,BandWindow.Halo<F32>>,~Storage.Buffer<F32>,~BandPipeline.Parameters(),~BandPipeline.Context,~BandPipeline.share,
    ~Bands.paired(~Storage.Buffer<F32>,~BandWindow.Halo<F32>,~BandPipeline.Parameters(),~BandPipeline.Context,~BandPipeline.band(~custom)),
    ~Bands.left_of(~Storage.Buffer<F32>,~BandWindow.Halo<F32>),
    left_plan,right_plan,mode,BandPipeline.parallel(context),BandPipeline.input_geometry(context),context,
    Bands.zip(~Storage.Buffer<F32>,~BandWindow.Halo<F32>,Model.pack(~F32,left),BandHalo.pack_owners(~F32,left_halo)),
    Bands.zip(~Storage.Buffer<F32>,~BandWindow.Halo<F32>,Model.pack(~F32,right),BandHalo.pack_owners(~F32,right_halo)),
    (weights,bias))

def LayoutRule(~custom: F32 -> F32,+plan: Bands.Plan,+context: BandPipeline.Context,+cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>) -> Type:
  @weights: Storage.Buffer<Tile.Tile8> -> @bias: Storage.Buffer<F32> ->
  {layout(BandPipeline.run(~custom,plan,BandPipeline.parallel_plan(context,plan),context,Model.pack(~F32,cores),BandHalo.pack_owners(~F32,halos),weights,bias)) == output_layout(halos) : Bands.Plan}

def fork_layout(~custom: F32 -> F32,+left_plan: Bands.Plan,+right_plan: Bands.Plan,+context: BandPipeline.Context,
  +left: Model.Model<F32>,+right: Model.Model<F32>,+left_halo: BandHalo.Owners<F32>,+right_halo: BandHalo.Owners<F32>,
  weight_copies: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,bias_copies: Storage.Buffer<F32> & Storage.Buffer<F32>,
  left_rule: LayoutRule(~custom,left_plan,context,left,left_halo),right_rule: LayoutRule(~custom,right_plan,context,right,right_halo)) ->
  {layout(fork_run(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,weight_copies,bias_copies)) ==
    Bands.Split{output_layout(left_halo),output_layout(right_halo)} : Bands.Plan}:
  (left_weights,right_weights) = weight_copies
  (left_bias,right_bias) = bias_copies
  %left_rule(left_weights,left_bias) :
    {Bands.Split{layout(BandPipeline.run(~custom,left_plan,BandPipeline.parallel_plan(context,left_plan),context,Model.pack(~F32,left),BandHalo.pack_owners(~F32,left_halo),left_weights,left_bias)),
      layout(BandPipeline.run(~custom,right_plan,BandPipeline.parallel_plan(context,right_plan),context,Model.pack(~F32,right),BandHalo.pack_owners(~F32,right_halo),right_weights,right_bias))} ==
      Bands.Split{_,output_layout(right_halo)} : Bands.Plan}
  %right_rule(right_weights,right_bias) :
    {Bands.Split{layout(BandPipeline.run(~custom,left_plan,BandPipeline.parallel_plan(context,left_plan),context,Model.pack(~F32,left),BandHalo.pack_owners(~F32,left_halo),left_weights,left_bias)),
      layout(BandPipeline.run(~custom,right_plan,BandPipeline.parallel_plan(context,right_plan),context,Model.pack(~F32,right),BandHalo.pack_owners(~F32,right_halo),right_weights,right_bias))} ==
      Bands.Split{layout(BandPipeline.run(~custom,left_plan,BandPipeline.parallel_plan(context,left_plan),context,Model.pack(~F32,left),BandHalo.pack_owners(~F32,left_halo),left_weights,left_bias)),_} : Bands.Plan}
  {==}

def run_layout(~custom: F32 -> F32,+plan: Bands.Plan,+mode: Bool,+context: BandPipeline.Context,+cores: Model.Model<F32>,+halos: BandHalo.Owners<F32>,
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,compatible: Compatible(plan,cores,halos)) ->
  {layout(BandPipeline.run(~custom,plan,mode,context,Model.pack(~F32,cores),BandHalo.pack_owners(~F32,halos),weights,bias)) == output_layout(halos) : Bands.Plan}:
  match plan cores:
    case Bands.Rows{+first,+rows} Model.Leaf{+core,cf,cr}:
      match halos:
        case BandHalo.OwnedLeaf{+above,+below,+window,hf,hr}:
          leaf_layout(~custom,context,first,rows,window,Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below),weights,bias)
        case BandHalo.OwnedFork{left_halo,right_halo}: match compatible:
    case Bands.Split{+left_plan,+right_plan} Model.Fork{+left,+right}:
      match halos:
        case BandHalo.OwnedFork{+left_halo,+right_halo}:
          Shape.Both{+left_compatible,+right_compatible} = compatible
          %Equal.sym(Bands.Bands(~F32),BandPipeline.run(~custom,Bands.Split{left_plan,right_plan},mode,context,Model.pack(~F32,Model.Fork{left,right}),BandHalo.pack_owners(~F32,BandHalo.OwnedFork{left_halo,right_halo}),weights,bias),
            fork_run(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,Clone.clone(~Tile.Tile8,weights),Clone.clone(~F32,bias)),
            layout_fork_execution(~custom,left_plan,right_plan,mode,context,left,right,left_halo,right_halo,weights,bias)) :
            {layout(_) == Bands.Split{output_layout(left_halo),output_layout(right_halo)} : Bands.Plan}
          fork_layout(~custom,left_plan,right_plan,context,left,right,left_halo,right_halo,Clone.clone(~Tile.Tile8,weights),Clone.clone(~F32,bias),
            left_weights => left_bias => run_layout(~custom,left_plan,BandPipeline.parallel_plan(context,left_plan),context,left,left_halo,left_weights,left_bias,left_compatible),
            right_weights => right_bias => run_layout(~custom,right_plan,BandPipeline.parallel_plan(context,right_plan),context,right,right_halo,right_weights,right_bias,right_compatible))
        case BandHalo.OwnedLeaf{above,below,window,hf,hr}: match compatible:
    case Bands.Rows{first,rows} Model.Fork{left,right}: match compatible:
    case Bands.Split{left_plan,right_plan} Model.Leaf{core,cf,cr}: match compatible:
