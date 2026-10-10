# Unary band operations preserve the whole public shape/count invariant as
# well as the existing per-owner representation invariant.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../storage_buffer.bend as Storage
import ./tensor_operations_refinement.bend as OperationLaws
import ../tensor_window.bend as Window
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_map.bend as BandMap
import ./tensor_band_indexed.bend as BandIndexed
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_band_gather.bend as BandGather
import ./storage_shape.bend as Shape
import ./tensor_dense_observation.bend as DenseObservation

def apply_ready(~Element: Data,~operation: Element -> Element,~mapper: @+count: Nat -> @+position: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element>,~lawful: OperationLaws.MapLaw(~Element,~operation,~mapper),+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+source: Model.Model<Element>,
  source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool}) ->
  Global.Normal(~Element,Tensors.apply(~Element,~mapper,name,operations,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,Tensors.Usable{},workers})):
  Shape.Both{BandMap.scheduled_map_well_formed(~Element,~operation,~mapper,~lawful,operations,geometry,workers,source,source_shape),Shape.Both{counted,coherent}}

def scalar_ready(~Element: Data,~operation: Element -> Element -> Element,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+source: Model.Model<Element>,+scalar: Element,
  source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool}) ->
  Global.Normal(~Element,Tensors.combine_scalar(~Element,~operation,name,operations,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,Tensors.Usable{},workers},scalar)):
  Shape.Both{BandMap.scheduled_scalar_well_formed(~Element,~operation,operations,scalar,geometry,workers,source,source_shape),Shape.Both{counted,coherent}}

def indexed_ready(~Element: Data,~operation: U32 -> Element -> Element,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+source: Model.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool}) ->
  Global.Normal(~Element,Tensors.apply_indexed(~Element,~operation,operations,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,Tensors.Usable{},workers})):
  %Equal.sym(Tensors.Tensor<Element>,Tensors.apply_indexed(~Element,~operation,operations,Tensors.Banded{Model.pack(~Element,source),geometry,shape,fill,Tensors.Usable{},workers}),
    Tensors.apply_indexed_positive(~Element,~operation,True{},operations,Model.pack(~Element,source),geometry,shape,fill,workers),
    BandMap.indexed_selected(~Element,~operation,operations,geometry,shape,fill,workers,Model.pack(~Element,source),width_positive)) : Global.Normal(~Element,_)
  Shape.Both{BandIndexed.scheduled_well_formed(~Element,~operation,operations,geometry,workers,fill,source,width_positive,source_shape),Shape.Both{counted,coherent}}

# One equation covers the dense route independently of the particular empty
# shape. The shared densification contracts apply to this exact dense input.
def indexed_dense(~Element: Data,~operation: U32 -> Element -> Element,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,bands: Bands.Bands(~Element),
  rejected: {Tensors.indexed_width_positive(geometry) == False{} : Bool}) ->
  {Tensors.apply_indexed(~Element,~operation,operations,Tensors.Banded{bands,geometry,shape,fill,Tensors.Usable{},workers}) ==
    Tensors.apply_indexed(~Element,~operation,operations,DenseObservation.dense_tensor(~Element,Tensors.Banded{bands,geometry,shape,fill,Tensors.Usable{},workers})) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Tensors.indexed_width_positive(geometry),False{},rejected) :
    {Tensors.apply_indexed_positive(~Element,~operation,_,operations,bands,geometry,shape,fill,workers) ==
      Tensors.apply_indexed(~Element,~operation,operations,DenseObservation.dense_tensor(~Element,Tensors.Banded{bands,geometry,shape,fill,Tensors.Usable{},workers})) : Tensors.Tensor<Element>}
  {==}

def with_workers(~Element: Data,+workers: U32,tensor: Tensors.Tensor<Element>,normal: Global.Normal(~Element,tensor)) -> Global.Normal(~Element,Tensors.with_workers(~Element,tensor,workers)):
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,old_workers}: normal
    case Tensors.Banded{bands,geometry,shape,fill,status,old_workers}: normal
