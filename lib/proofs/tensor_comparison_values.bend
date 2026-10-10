# The six real public comparisons inherit the binary value theorems. Their
# kernel law is discharged here, not left as a premise of the public result.
import Base
import ../tensor_f32.bend as F32Tensor
import ../tensor.bend as Tensors
import ../tensor_lanes.bend as Lanes
import ../math_compare.bend as Compare
import ../storage_buffer.bend as Storage
import ../tensor_shape.bend as Shapes
import ../tensor_band.bend as Bands
import ./tensor_lanes_refinement.bend as LanesLaws
import ./tensor_combine_dense_values.bend as DenseCombine
import ./tensor_dense_contract.bend as Dense
import ./tensor_band_combine.bend as BandCombine
import ./tensor_band_values.bend as BandValues
import ./tensor_band_model.bend as Model
import ./tensor_global_shape_count.bend as Global
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree

def DenseLaw(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  ~operation: F32 -> F32 -> F32,~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32> -> Tensors.Tensor<F32>) -> Type:
  @+left: Left -> @+right: Right -> @left_storage: Storage.Buffer<F32> -> @+left_shape: List<&2,U32> -> @+left_fill: F32 -> @+left_workers: U32 ->
  @right_storage: Storage.Buffer<F32> -> @+right_shape: List<&2,U32> -> @+right_fill: F32 -> @+right_workers: U32 ->
  @left_contract: Dense.Contract(~F32,~Left,~left_value,left,Tensors.Tensor{left_storage,left_shape,left_fill,Tensors.Usable{},left_workers}) ->
  @right_contract: Dense.Contract(~F32,~Right,~right_value,right,Tensors.Tensor{right_storage,right_shape,right_fill,Tensors.Usable{},right_workers}) ->
  @accepted: {Dense.tensor_ready(~F32,call(Tensors.Tensor{left_storage,left_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Tensor{right_storage,right_shape,right_fill,Tensors.Usable{},right_workers})) == True{} : Bool} ->
  DenseCombine.Stated(~F32,~Left,~Right,~left_value,~right_value,~operation,Shapes.same_extents(left_shape,right_shape),left,right,left_shape,right_shape,
    call(Tensors.Tensor{left_storage,left_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Tensor{right_storage,right_shape,right_fill,Tensors.Usable{},right_workers}))

def dense(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,~name: String,~cost: F32) ->
  DenseLaw(~Left,~Right,~left_value,~right_value,~operation,~Tensors.combine_with(~F32,~Lanes.combine(~operation),name,cost)):
  left => right => left_storage => left_shape => left_fill => left_workers => right_storage => right_shape => right_fill => right_workers => left_contract => right_contract => accepted =>
    DenseCombine.combine(~F32,~Left,~Right,~left_value,~right_value,~operation,~Lanes.combine(~operation),~LanesLaws.combine(~operation),left,right,name,cost,
      left_storage,left_shape,left_fill,left_workers,right_storage,right_shape,right_fill,right_workers,left_contract,right_contract,accepted)

def BandLaw(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  ~operation: F32 -> F32 -> F32,~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32> -> Tensors.Tensor<F32>) -> Type:
  @+left: Left -> @+right: Right -> @+left_geometry: Bands.Geometry -> @+right_geometry: Bands.Geometry -> @+shape: List<&2,U32> ->
  @+left_fill: F32 -> @+right_fill: F32 -> @+left_workers: U32 -> @+right_workers: U32 -> @left_bands: Bands.Bands(~F32) -> @right_bands: Bands.Bands(~F32) ->
  @left_normal: Global.Normal(~F32,Tensors.Banded{left_bands,left_geometry,shape,left_fill,Tensors.Usable{},left_workers}) ->
  @right_normal: Global.Normal(~F32,Tensors.Banded{right_bands,right_geometry,shape,right_fill,Tensors.Usable{},right_workers}) ->
  @left_values: BandValues.Values(~F32,~Left,~left_value,left,left_geometry,Model.contents(~F32,left_bands)) ->
  @right_values: BandValues.Values(~F32,~Right,~right_value,right,right_geometry,Model.contents(~F32,right_bands)) ->
  BandCombine.NormalContract(~F32,~Left,~Right,~operation,~left_value,~right_value,left,right,
    call(Tensors.Banded{left_bands,left_geometry,shape,left_fill,Tensors.Usable{},left_workers},Tensors.Banded{right_bands,right_geometry,shape,right_fill,Tensors.Usable{},right_workers}))

def banded(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,~name: String,~cost: F32) ->
  BandLaw(~Left,~Right,~left_value,~right_value,~operation,~Tensors.combine_with(~F32,~Lanes.combine(~operation),name,cost)):
  left => right => left_geometry => right_geometry => shape => left_fill => right_fill => left_workers => right_workers => left_bands => right_bands => left_normal => right_normal => left_values => right_values =>
    BandCombine.native_pair_ready(~F32,~Left,~Right,~operation,~Lanes.combine(~operation),~LanesLaws.combine(~operation),~left_value,~right_value,
      left,right,name,cost,left_geometry,right_geometry,shape,left_fill,right_fill,left_workers,right_workers,left_bands,right_bands,left_normal,right_normal,left_values,right_values)

def LeftBandLaw(~Left: Data,~left_value: Left -> U32 -> F32,~operation: F32 -> F32 -> F32,
  ~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32> -> Tensors.Tensor<F32>) -> Type:
  @+left: Left -> @+geometry: Bands.Geometry -> @+shape: List<&2,U32> -> @+left_fill: F32 -> @+right_fill: F32 -> @+left_workers: U32 -> @+right_workers: U32 ->
  @left_bands: Bands.Bands(~F32) -> @right_buffer: Storage.Buffer<F32> ->
  @left_normal: Global.Normal(~F32,Tensors.Banded{left_bands,geometry,shape,left_fill,Tensors.Usable{},left_workers}) ->
  @right_normal: Global.Normal(~F32,Tensors.Tensor{right_buffer,shape,right_fill,Tensors.Usable{},right_workers}) ->
  @left_values: BandValues.Values(~F32,~Left,~left_value,left,geometry,Model.contents(~F32,left_bands)) ->
  BandCombine.NormalContract(~F32,~Left,~Tree.Tree<F32>,~operation,~left_value,~BandValues.dense_value(~F32),left,Buffer.content_tree(~F32,Buffer.contents(~F32,right_buffer)),
    call(Tensors.Banded{left_bands,geometry,shape,left_fill,Tensors.Usable{},left_workers},Tensors.Tensor{right_buffer,shape,right_fill,Tensors.Usable{},right_workers}))

def left_band(~Left: Data,~left_value: Left -> U32 -> F32,~operation: F32 -> F32 -> F32,~name: String,~cost: F32) ->
  LeftBandLaw(~Left,~left_value,~operation,~Tensors.combine_with(~F32,~Lanes.combine(~operation),name,cost)):
  left => geometry => shape => left_fill => right_fill => left_workers => right_workers => left_bands => right_buffer => left_normal => right_normal => left_values =>
    BandCombine.native_band_left_ready(~F32,~Left,~operation,~Lanes.combine(~operation),~LanesLaws.combine(~operation),~left_value,
      left,name,cost,geometry,shape,left_fill,right_fill,left_workers,right_workers,left_bands,right_buffer,left_normal,right_normal,left_values)

def RightBandLaw(~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,
  ~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32> -> Tensors.Tensor<F32>) -> Type:
  @+right: Right -> @+geometry: Bands.Geometry -> @+shape: List<&2,U32> -> @+left_fill: F32 -> @+right_fill: F32 -> @+left_workers: U32 -> @+right_workers: U32 ->
  @left_buffer: Storage.Buffer<F32> -> @right_bands: Bands.Bands(~F32) ->
  @left_normal: Global.Normal(~F32,Tensors.Tensor{left_buffer,shape,left_fill,Tensors.Usable{},left_workers}) ->
  @right_normal: Global.Normal(~F32,Tensors.Banded{right_bands,geometry,shape,right_fill,Tensors.Usable{},right_workers}) ->
  @right_values: BandValues.Values(~F32,~Right,~right_value,right,geometry,Model.contents(~F32,right_bands)) ->
  BandCombine.NormalContract(~F32,~Tree.Tree<F32>,~Right,~operation,~BandValues.dense_value(~F32),~right_value,Buffer.content_tree(~F32,Buffer.contents(~F32,left_buffer)),right,
    call(Tensors.Tensor{left_buffer,shape,left_fill,Tensors.Usable{},left_workers},Tensors.Banded{right_bands,geometry,shape,right_fill,Tensors.Usable{},right_workers}))

def right_band(~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,~name: String,~cost: F32) ->
  RightBandLaw(~Right,~right_value,~operation,~Tensors.combine_with(~F32,~Lanes.combine(~operation),name,cost)):
  right => geometry => shape => left_fill => right_fill => left_workers => right_workers => left_buffer => right_bands => left_normal => right_normal => right_values =>
    BandCombine.native_dense_left_ready(~F32,~Right,~operation,~Lanes.combine(~operation),~LanesLaws.combine(~operation),~right_value,
      right,name,cost,geometry,shape,left_fill,right_fill,left_workers,right_workers,left_buffer,right_bands,left_normal,right_normal,right_values)

type Rules<-Dense: Type,-Banded: Type,-LeftBand: Type,-RightBand: Type> is Type:
  Rules{dense: Dense,banded: Banded,left_band: LeftBand,right_band: RightBand}

def PublicLaw(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  ~operation: F32 -> F32 -> F32,~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32> -> Tensors.Tensor<F32>) -> Type:
  Rules<DenseLaw(~Left,~Right,~left_value,~right_value,~operation,~call),BandLaw(~Left,~Right,~left_value,~right_value,~operation,~call),
    LeftBandLaw(~Left,~left_value,~operation,~call),RightBandLaw(~Right,~right_value,~operation,~call)>

def public(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,~name: String,~cost: F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~operation,~Tensors.combine_with(~F32,~Lanes.combine(~operation),name,cost)):
  Rules{dense(~Left,~Right,~left_value,~right_value,~operation,~name,~cost),banded(~Left,~Right,~left_value,~right_value,~operation,~name,~cost),
    left_band(~Left,~left_value,~operation,~name,~cost),right_band(~Right,~right_value,~operation,~name,~cost)}

def less(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.less_elements,~F32Tensor.less):
  public(~Left,~Right,~left_value,~right_value,~Compare.less_elements,~"less",~2.625)

def less_equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.less_equal_elements,~F32Tensor.less_equal):
  public(~Left,~Right,~left_value,~right_value,~Compare.less_equal_elements,~"less_equal",~2.625)

def greater(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.greater_elements,~F32Tensor.greater):
  public(~Left,~Right,~left_value,~right_value,~Compare.greater_elements,~"greater",~2.625)

def greater_equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.greater_equal_elements,~F32Tensor.greater_equal):
  public(~Left,~Right,~left_value,~right_value,~Compare.greater_equal_elements,~"greater_equal",~2.625)

def equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.equal_elements,~F32Tensor.equal):
  public(~Left,~Right,~left_value,~right_value,~Compare.equal_elements,~"equal",~1.375)

def not_equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.not_equal_elements,~F32Tensor.not_equal):
  public(~Left,~Right,~left_value,~right_value,~Compare.not_equal_elements,~"not_equal",~1.375)
