# The parameter receipts observe the actual physical owners. Their exact
# logical lengths supply the full bias content contract used by band leaves.
import Base
import ../specification.bend as ConvSpec
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/storage_buffer.bend as Storage
import ../../lib/kernel_band_planning.bend as BandPlanning
import ../../lib/math_activation.bend as Activations
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/tensor_dense_contract.bend as Dense
import ./band_public_frame.bend as PublicFrame
import ./band_native_caller.bend as Native
import ./band_parameter_owners.bend as Parameters
import ./band_dense_native_contract.bend as DenseNativeContract
import ./band_dense_native_public.bend as DenseNativePublic
import ./band_initial_layout.bend as InitialLayout

def parameter_tree(-buffer: Storage.Buffer<F32>,+parameter: Parameters.Parameter<buffer>) ->
  {Buffer.tree(~F32,Parameters.parameter_model(buffer,parameter)) ==
    StorageLaws.sequence_storage(F32,ConvSpec.length(Parameters.parameter_values(buffer,parameter)),Parameters.parameter_values(buffer,parameter),Parameters.parameter_evidence(buffer,parameter)) : Tree.Tree<F32>}:
  Parameters.Parameter{model,equation,ready,values,length_exact,evidence,tree_exact} = parameter
  tree_exact

def initial_layout(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+plan: BandPlanning.Plan,
  +source: Buffer.Model<F32>,-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>,+weight_parameter: Parameters.Parameter<weights>,+bias_parameter: Parameters.Parameter<bias>,
  +fill: F32,+input_workers: U32,+workers: U32,
  +normal: Global.Normal(~F32,Tensors.Tensor{Buffer.pack(~F32,source),Native.input_extents(input),fill,Tensors.Usable{},input_workers}),
  +matched: {Native.input_matched(input) == True{} : Bool},
  +validated: {PublicFrame.frame_valid(input,Parameters.parameter_values(weights,weight_parameter),Parameters.parameter_values(bias,bias_parameter)) == True{} : Bool},
  +supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +eligible: {TensorConvolution.uses_band_windows(PublicFrame.geometry(input)) == True{} : Bool},
  values: Dense.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,source))) ->
  InitialLayout.LayoutContract(~custom,~Logical,~original,
    DenseNativeContract.sources_context(~Logical,logical,activation,input,Parameters.parameter_values(weights,weight_parameter),Parameters.parameter_values(bias,bias_parameter)),
    TensorConvolution.initial_layout(~custom,activation,plan,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),weights),bias,workers)):
  %Parameters.parameter_equation(weights,weight_parameter) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,
      DenseNativeContract.sources_context(~Logical,logical,activation,input,Parameters.parameter_values(weights,weight_parameter),Parameters.parameter_values(bias,bias_parameter)),
      TensorConvolution.initial_layout(~custom,activation,plan,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),_),bias,workers))
  %Parameters.parameter_equation(bias,bias_parameter) :
    InitialLayout.LayoutContract(~custom,~Logical,~original,
      DenseNativeContract.sources_context(~Logical,logical,activation,input,Parameters.parameter_values(weights,weight_parameter),Parameters.parameter_values(bias,bias_parameter)),
      TensorConvolution.initial_layout(~custom,activation,plan,PublicFrame.geometry(input),Buffer.pack(~F32,source),TensorConvolution.packed(PublicFrame.geometry(input),Buffer.pack(~F32,Parameters.parameter_model(weights,weight_parameter))),_,workers))
  DenseNativePublic.initial_layout(~custom,~Logical,~original,logical,activation,input,plan,
    Parameters.parameter_values(weights,weight_parameter),Parameters.parameter_values(bias,bias_parameter),
    Parameters.parameter_evidence(weights,weight_parameter),Parameters.parameter_evidence(bias,bias_parameter),
    source,Parameters.parameter_model(weights,weight_parameter),Parameters.parameter_model(bias,bias_parameter),fill,input_workers,workers,
    normal,matched,validated,supported,eligible,values,
    parameter_tree(weights,weight_parameter),Parameters.parameter_content(bias,bias_parameter))
