# The public eight-lane functions of one tensor are the ordered map of their
# scalar function, on dense and on band storage. The mapper law is discharged
# here, not left as a premise of the public result.
import Base
import ../tensor_f32.bend as F32Tensor
import ../tensor.bend as Tensors
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../tensor_view.bend as Views
import ../tensor_band.bend as Bands
import ../storage_buffer.bend as Storage
import ../math_activation.bend as Activations
import ../math_fp32.bend as Arithmetic
import ./tensor_operations_refinement.bend as OperationLaws
import ./tensor_public_refinement.bend as PublicLaws
import ./tensor_band_map.bend as BandMap
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./storage_shape.bend as Shape

# A usable dense tensor goes to the checked mapper on its whole view.
def DenseCall(~operation: F32 -> F32,~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32>,~name: String) -> Type:
  @storage: Storage.Buffer<F32> -> @+shape: List<&2,U32> -> @+fill: F32 -> @+view: Views.View -> @+workers: U32 ->
  @views: {Tensors.view_of(shape) == Some{view} : Maybe<&2,Views.View>} ->
  {call(Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers}) ==
    Tensors.applied(~F32,name,shape,fill,workers,Operations.apply_context(~F32,~Unit,~Operations.without_context(~F32,~Operations.mapped_lanes(~operation)),Unit{},storage,view)) : Tensors.Tensor<F32>}

def dense_call(~operation: F32 -> F32,~name: String,~cost: F32) -> DenseCall(~operation,~Tensors.apply(~F32,~Operations.mapped_lanes(~operation),name,cost),~name):
  storage => shape => fill => view => workers => views =>
    PublicLaws.apply_ready(~F32,~Unit,~Operations.without_context(~F32,~Operations.mapped_lanes(~operation)),Unit{},name,cost,storage,shape,fill,view,workers,views)

# That mapper leaves the ordered map of the operation, or rejects the view.
def DenseValues(~operation: F32 -> F32) -> Type:
  @storage: Storage.Buffer<F32> -> @+view: Views.View ->
  {OperationLaws.observe_outcome(~F32,Operations.apply_context(~F32,~Unit,~Operations.without_context(~F32,~Operations.mapped_lanes(~operation)),Unit{},storage,view)) ==
    OperationLaws.apply_denotation(~F32,~operation,storage,view) : Operations.Outcome<OperationLaws.Owner<&1,Array<F32>>>}

def dense_values(~operation: F32 -> F32) -> DenseValues(~operation):
  storage => view => OperationLaws.apply(~F32,~operation,~Operations.mapped_lanes(~operation),~OperationLaws.mapped_lanes(~operation),storage,view)

def BandLaw(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32,~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32>) -> Type:
  @+logical: Logical -> @+geometry: Bands.Geometry -> @+shape: List<&2,U32> -> @+fill: F32 -> @+workers: U32 -> @+bands: Model.Model<F32> ->
  @+well_formed: BandShape.WellFormed<F32,geometry,Model.content(~F32,bands)> ->
  @values: BandValues.Values(~F32,~Logical,~original,logical,geometry,Model.content(~F32,bands)) ->
  Shape.Both<BandMap.WellFormed(~F32,call(Tensors.Banded{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers})),
    BandMap.LogicalValues(~F32,~BandMap.Context<Unit,Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~operation),~original),BandMap.Context{Unit{},logical},
      call(Tensors.Banded{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}))>

def banded(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32,~name: String,~cost: F32) ->
  BandLaw(~Logical,~original,~operation,~Tensors.apply(~F32,~Operations.mapped_lanes(~operation),name,cost)):
  logical => geometry => shape => fill => workers => bands => well_formed => values =>
    BandMap.apply_ready(~F32,~Logical,~operation,~Operations.mapped_lanes(~operation),~OperationLaws.mapped_lanes(~operation),~original,name,cost,
      logical,geometry,shape,fill,workers,bands,well_formed,values)

type Rules<-Call: Type,-Values: Type,-Banded: Type> is Type:
  Rules{dense_call: Call,dense_values: Values,banded: Banded}

def PublicLaw(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32,~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32>,~name: String) -> Type:
  Rules<DenseCall(~operation,~call,~name),DenseValues(~operation),BandLaw(~Logical,~original,~operation,~call)>

def public(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32,~name: String,~cost: F32) ->
  PublicLaw(~Logical,~original,~operation,~Tensors.apply(~F32,~Operations.mapped_lanes(~operation),name,cost),~name):
  Rules{dense_call(~operation,~name,~cost),dense_values(~operation),banded(~Logical,~original,~operation,~name,~cost)}

def sigmoid(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.sigmoid_value,~F32Tensor.sigmoid,~"sigmoid"):
  public(~Logical,~original,~Activations.sigmoid_value,~"sigmoid",~Activations.operations(Activations.Sigmoid{}))

def silu(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.silu_value,~F32Tensor.silu,~"silu"):
  public(~Logical,~original,~Activations.silu_value,~"silu",~Activations.operations(Activations.SiLU{}))

def tanh(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.tanh_value,~F32Tensor.tanh,~"tanh"):
  public(~Logical,~original,~Activations.tanh_value,~"tanh",~Activations.operations(Activations.Tanh{}))

def gelu_tanh(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.gelu_tanh_value,~F32Tensor.gelu_tanh,~"gelu_tanh"):
  public(~Logical,~original,~Activations.gelu_tanh_value,~"gelu_tanh",~Activations.operations(Activations.GELUTanh{}))

def exp(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Arithmetic.exp,~F32Tensor.exp,~"exp"):
  public(~Logical,~original,~Arithmetic.exp,~"exp",~4.5)

def log(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Arithmetic.log,~F32Tensor.log,~"log"):
  public(~Logical,~original,~Arithmetic.log,~"log",~8.25)
