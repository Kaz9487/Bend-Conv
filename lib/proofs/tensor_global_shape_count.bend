# Normal ready tensors retain their actual public shape count. Per-band
# metadata alone does not imply this whole-tensor machine envelope.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_window.bend as Window
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as Metadata
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_count.bend as BandCount

type Counted<-shape: List<&2,U32>> is Data:
  Counted{count: U32,accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}}

def BandShape(status: Tensors.Status,shape: List<&2,U32>,geometry: Bands.Geometry) -> Data:
  match status:
    case Tensors.Usable{}: Shape.Both<Counted<shape>,{Window.band_shape(shape,geometry) == True{} : Bool}>
    case Tensors.Failed{problem}: Unit

def DenseShape(~Element: Data,status: Tensors.Status,shape: List<&2,U32>,content: Buffer.Content<Element>) -> Data:
  match status:
    case Tensors.Usable{}: Shape.Both<Counted<shape>,{Buffer.content_length(~Element,content) == ShapeLaws.elements(shape) : Nat}>
    case Tensors.Failed{problem}: Unit

def Normal(~Element: Data,tensor: Tensors.Tensor<Element>) -> Data:
  match tensor:
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}:
      Shape.Both<Metadata.WellFormed<Element,geometry,Model.contents(~Element,bands)>,BandShape(status,shape,geometry)>
    case Tensors.Tensor{storage,shape,fill,status,workers}:
      Shape.Both<Buffer.ContentReady<Element,Buffer.contents(~Element,storage)>,DenseShape(~Element,status,shape,Buffer.contents(~Element,storage))>

def count(+shape: List<&2,U32>,evidence: Counted<shape>) -> U32:
  Counted{count,accepted} = evidence
  count

def evidence(+shape: List<&2,U32>,+counted: Counted<shape>) -> Count.Count<ShapeLaws.elements(shape),count(shape,counted)>:
  Counted{+count,accepted} = counted
  Count.count_of(shape,count,accepted)

def geometry_evidence(+shape: List<&2,U32>,+geometry: Bands.Geometry,+counted: Counted<shape>,
  coherent: {Window.band_shape(shape,geometry) == True{} : Bool}) -> Count.Count<BandGather.elements(geometry),count(shape,counted)>:
  %BandCount.shape_coherence_elements(shape,geometry,coherent) : Count.Count<_,count(shape,counted)>
  evidence(shape,counted)

def domain(+shape: List<&2,U32>,+geometry: Bands.Geometry,+counted: Counted<shape>,
  coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool}) ->
  BandGather.Domain<geometry,Default.limits(),Machine.capacity(Default.limits())>:
  BandCount.evidence(geometry,count(shape,counted),channels_positive,height_positive,width_positive,
    %Equal.sym(Nat,ShapeLaws.elements(BandCount.extents(geometry)),BandGather.elements(geometry),BandCount.domain_elements(geometry)) : Count.Count<_,count(shape,counted)>
    geometry_evidence(shape,geometry,counted,coherent))
