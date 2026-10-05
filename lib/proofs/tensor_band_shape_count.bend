# The public shape count agrees with the band allocation count, including
# empty extents.
import Base
import ./tensor_band_count.bend as BandCount
import ./tensor_shape_refinement.bend as ShapeLaws
import ../tensor_band.bend as Bands
import ./row_major_index.bend as RowMajor
import ../default_machine_limits.bend as Default
import ../machine_limits.bend as Machine
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_contiguous_bounds.bend as ContiguousBounds
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_count_refinement.bend as Count
import ./tensor_global_shape_count.bend as Global
import ./word_encoding.bend as WordEncoding
import ../tensor_shape.bend as Shapes
import ../tensor_window.bend as Window

# The actual global shape count equals the native band allocation count.
# Zero axes use modular zero identities; they need no positive dimensions.
type GeometryCase<-geometry: Bands.Geometry> is Data:
  ZeroGeometry{native_zero: {Bands.elements(geometry) == 0 : U32},natural_zero: {BandGather.elements(geometry) == 0n : Nat}}
  PositiveGeometry{channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},
    height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool}}

def zero_after(+prefix: U32,+factor: U32,zero: {prefix == 0 : U32}) -> {(prefix * factor : U32) == 0 : U32}:
  Equal.trans(U32,(prefix * factor : U32),(0 * factor : U32),0,
    Equal.cong(U32,U32,value => (value * factor : U32),prefix,0,zero),Arithmetic.multiply_zero_left(factor))

def images_zero(+channels: U32,+height: U32,+width: U32) -> GeometryCase<Bands.Geometry{0,channels,height,width}>:
  ZeroGeometry{zero_after((0 * channels * height : U32),width,
    zero_after((0 * channels : U32),height,Arithmetic.multiply_zero_left(channels))),{==}}

def channels_zero(+images: U32,+height: U32,+width: U32) -> GeometryCase<Bands.Geometry{images,0,height,width}>:
  ZeroGeometry{zero_after((images * 0 * height : U32),width,
    zero_after((images * 0 : U32),height,BandCount.multiply_zero_right(images))),RowMajor.multiply_zero_right(U32.to_nat(images))}

def height_zero(+images: U32,+channels: U32,+width: U32) -> GeometryCase<Bands.Geometry{images,channels,0,width}>:
  ZeroGeometry{zero_after((images * channels * 0 : U32),width,BandCount.multiply_zero_right((images * channels : U32))),
    Equal.trans(Nat,BandGather.elements(Bands.Geometry{images,channels,0,width}),Nat.mul(U32.to_nat(images),0n),0n,
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(images),value),Nat.mul(U32.to_nat(channels),0n),0n,RowMajor.multiply_zero_right(U32.to_nat(channels))),
      RowMajor.multiply_zero_right(U32.to_nat(images)))}

def width_zero(+images: U32,+channels: U32,+height: U32) -> GeometryCase<Bands.Geometry{images,channels,height,0}>:
  ZeroGeometry{BandCount.multiply_zero_right((images * channels * height : U32)),
    Equal.trans(Nat,BandGather.elements(Bands.Geometry{images,channels,height,0}),Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),0n)),0n,
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),value)),Nat.mul(U32.to_nat(height),0n),0n,RowMajor.multiply_zero_right(U32.to_nat(height))),
      Equal.trans(Nat,Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),0n)),Nat.mul(U32.to_nat(images),0n),0n,
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(images),value),Nat.mul(U32.to_nat(channels),0n),0n,RowMajor.multiply_zero_right(U32.to_nat(channels))),
        RowMajor.multiply_zero_right(U32.to_nat(images))))}

def zero_geometry(+before: Bands.Geometry,+after: Bands.Geometry,equal: {before == after : Bands.Geometry},evidence: GeometryCase<after>) -> GeometryCase<before>:
  %Equal.sym(Bands.Geometry,before,after,equal) : GeometryCase<_>
  evidence

def images(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  images

def channels(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  channels

def height(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  height

def width(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  width

def global_element_count_classified(+geometry: Bands.Geometry,images_empty: Bool,channels_empty: Bool,height_empty: Bool,width_empty: Bool,
  images_compared: {U32.is_zero(images(geometry)) == images_empty : Bool},
  channels_compared: {U32.is_zero(channels(geometry)) == channels_empty : Bool},
  height_compared: {U32.is_zero(height(geometry)) == height_empty : Bool},
  width_compared: {U32.is_zero(width(geometry)) == width_empty : Bool}) -> GeometryCase<geometry>:
  Bands.Geometry{+image_count,+channel_count,+row_count,+column_count} = geometry
  match images_empty channels_empty height_empty width_empty:
    case True{} channels_empty height_empty width_empty:
      zero_geometry(geometry,Bands.Geometry{0,channel_count,row_count,column_count},
        Equal.cong(U32,Bands.Geometry,value => Bands.Geometry{value,channel_count,row_count,column_count},image_count,0,ShapeLaws.native_equal(image_count,0,images_compared)),
        images_zero(channel_count,row_count,column_count))
    case False{} True{} height_empty width_empty:
      zero_geometry(geometry,Bands.Geometry{image_count,0,row_count,column_count},
        Equal.cong(U32,Bands.Geometry,value => Bands.Geometry{image_count,value,row_count,column_count},channel_count,0,ShapeLaws.native_equal(channel_count,0,channels_compared)),
        channels_zero(image_count,row_count,column_count))
    case False{} False{} True{} width_empty:
      zero_geometry(geometry,Bands.Geometry{image_count,channel_count,0,column_count},
        Equal.cong(U32,Bands.Geometry,value => Bands.Geometry{image_count,channel_count,value,column_count},row_count,0,ShapeLaws.native_equal(row_count,0,height_compared)),
        height_zero(image_count,channel_count,column_count))
    case False{} False{} False{} True{}:
      zero_geometry(geometry,Bands.Geometry{image_count,channel_count,row_count,0},
        Equal.cong(U32,Bands.Geometry,value => Bands.Geometry{image_count,channel_count,row_count,value},column_count,0,ShapeLaws.native_equal(column_count,0,width_compared)),
        width_zero(image_count,channel_count,row_count))
    case False{} False{} False{} False{}:
      PositiveGeometry{CountCompletion.positive_word(channel_count,channels_compared),CountCompletion.positive_word(row_count,height_compared),CountCompletion.positive_word(column_count,width_compared)}

def classify(+geometry: Bands.Geometry) -> GeometryCase<geometry>:
  global_element_count_classified(geometry,U32.is_zero(images(geometry)),U32.is_zero(channels(geometry)),U32.is_zero(height(geometry)),U32.is_zero(width(geometry)),{==},{==},{==},{==})

def zero_counted(+geometry: Bands.Geometry,+count: U32,native_zero: {Bands.elements(geometry) == 0 : U32},natural_zero: {BandGather.elements(geometry) == 0n : Nat},
  count_exact: {BandGather.elements(geometry) == U32.to_nat(count) : Nat}) -> {Bands.elements(geometry) == count : U32}:
  Equal.trans(U32,Bands.elements(geometry),0,count,native_zero,
    Equal.trans(U32,0,U32.from_nat(U32.to_nat(count)),count,
      Equal.cong(Nat,U32,U32.from_nat,0n,U32.to_nat(count),
        Equal.trans(Nat,0n,BandGather.elements(geometry),U32.to_nat(count),Equal.sym(Nat,BandGather.elements(geometry),0n,natural_zero),count_exact)),
      Arithmetic.native_roundtrip(count)))

def global_element_count_selected(+shape: List<&2,U32>,+geometry: Bands.Geometry,+counted: Global.Counted<shape>,+coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  count_evidence: Count.Count<BandGather.elements(geometry),Global.count(shape,counted)>,case_evidence: GeometryCase<geometry>) ->
  {Bands.elements(geometry) == Global.count(shape,counted) : U32}:
  Count.Count{count_exact,bounded} = count_evidence
  match case_evidence:
    case ZeroGeometry{native_zero,natural_zero}: zero_counted(geometry,Global.count(shape,counted),native_zero,natural_zero,count_exact)
    case PositiveGeometry{channels_positive,height_positive,width_positive}:
      BandCount.element_count_counted(geometry,Default.limits(),Machine.capacity(Default.limits()),Global.count(shape,counted),
        Global.domain(shape,geometry,counted,coherent,channels_positive,height_positive,width_positive),count_exact)

def global_element_count_exact(+shape: List<&2,U32>,+geometry: Bands.Geometry,+counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool}) ->
  {Bands.elements(geometry) == Global.count(shape,counted) : U32}:
  +coherent = coherent
  global_element_count_selected(shape,geometry,counted,coherent,Global.geometry_evidence(shape,geometry,counted,coherent),classify(geometry))

# Every checked band owner supplies an exact local count. Zero width uses
# its native zero product; nonzero width uses the owner's allocation room.
def count_refinement_width(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,count_refinement_width} = geometry
  count_refinement_width

def count_refinement_exact(+geometry: Bands.Geometry,+rows: U32,+length: Nat,counted: BandShape.Count<geometry,rows,length>) ->
  {U32.to_nat(Bands.local_count(geometry,rows)) == length : Nat}:
  BandShape.Count{count_refinement_exact,bounded} = counted
  count_refinement_exact

def count_refinement_selected(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,empty_width: Bool,
  classified: {U32.is_zero(count_refinement_width(geometry)) == empty_width : Bool},+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Ready<Element,input>) ->
  BandShape.Count<geometry,rows,Buffer.length(~Element,input)>:
  Bands.Geometry{+images,+channels,+height,+count_refinement_width} = geometry
  Buffer.Model{+tree,+length,+certificate} = input
  match empty_width:
    case True{}:
      +width_zero = ShapeLaws.native_equal(count_refinement_width,0,classified)
      +metadata_zero = {
        %width_zero : BandShape.Leaf(Bands.Geometry{images,channels,height,_},first,rows,length)
        metadata : BandShape.Leaf(Bands.Geometry{images,channels,height,0},first,rows,length)}
      %Equal.sym(U32,count_refinement_width,0,width_zero) : BandShape.Count<Bands.Geometry{images,channels,height,_},rows,length>
      BandCount.width(~Element,images,channels,height,first,rows,Buffer.Model{tree,length,certificate},metadata_zero)
    case False{}:
      BandShape.count(~Element,Bands.Geometry{images,channels,height,count_refinement_width},first,rows,Buffer.Model{tree,length,certificate},ContiguousBounds.extent_positive(count_refinement_width,classified),metadata,ready)

def counted(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Ready<Element,input>) ->
  BandShape.Count<geometry,rows,Buffer.length(~Element,input)>:
  count_refinement_selected(~Element,geometry,first,rows,input,U32.is_zero(count_refinement_width(geometry)),{==},metadata,ready)

# Public empty extents identify either a zero local element axis or zero
# total rows. This classification uses the actual shape's zero checks.
def empty_geometry_width(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,empty_geometry_channels,height,empty_geometry_width} = geometry
  empty_geometry_width

def empty_geometry_channels(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,empty_geometry_channels,height,empty_geometry_width} = geometry
  empty_geometry_channels

type EmptyGeometry<-geometry: Bands.Geometry> is Data:
  Width{zero: {empty_geometry_width(geometry) == 0 : U32}}
  Channels{zero: {empty_geometry_channels(geometry) == 0 : U32}}
  Rows{zero: {Inspection.total_rows(geometry) == 0n : Nat}}

def width_selected(empty: Bool,+images: U32,+empty_geometry_channels: U32,+height: U32,+empty_geometry_width: U32,
  empty_geometry_classified: {U32.is_zero(empty_geometry_width) == empty : Bool},truth: {empty || False{} == True{} : Bool}) ->
  EmptyGeometry<Bands.Geometry{images,empty_geometry_channels,height,empty_geometry_width}>:
  match empty:
    case True{}: Width{ShapeLaws.native_equal(empty_geometry_width,0,empty_geometry_classified)}
    case False{}: Empty.absurd(EmptyGeometry<Bands.Geometry{images,empty_geometry_channels,height,empty_geometry_width}>,WordEncoding.false_true(truth))

def height_selected(empty: Bool,+images: U32,+empty_geometry_channels: U32,+height: U32,+empty_geometry_width: U32,
  empty_geometry_classified: {U32.is_zero(height) == empty : Bool},truth: {empty || (U32.is_zero(empty_geometry_width) || False{}) == True{} : Bool}) ->
  EmptyGeometry<Bands.Geometry{images,empty_geometry_channels,height,empty_geometry_width}>:
  match empty:
    case True{}:
      Rows{
        %Equal.sym(U32,height,0,ShapeLaws.native_equal(height,0,empty_geometry_classified)) : {Nat.mul(U32.to_nat(images),U32.to_nat(_)) == 0n : Nat}
        RowMajor.multiply_zero_right(U32.to_nat(images))}
    case False{}: width_selected(U32.is_zero(empty_geometry_width),images,empty_geometry_channels,height,empty_geometry_width,{==},truth)

def channels_selected(empty: Bool,+images: U32,+empty_geometry_channels: U32,+height: U32,+empty_geometry_width: U32,
  empty_geometry_classified: {U32.is_zero(empty_geometry_channels) == empty : Bool},truth: {empty || (U32.is_zero(height) || (U32.is_zero(empty_geometry_width) || False{})) == True{} : Bool}) ->
  EmptyGeometry<Bands.Geometry{images,empty_geometry_channels,height,empty_geometry_width}>:
  match empty:
    case True{}: Channels{ShapeLaws.native_equal(empty_geometry_channels,0,empty_geometry_classified)}
    case False{}: height_selected(U32.is_zero(height),images,empty_geometry_channels,height,empty_geometry_width,{==},truth)

def images_selected(empty: Bool,+images: U32,+empty_geometry_channels: U32,+height: U32,+empty_geometry_width: U32,
  empty_geometry_classified: {U32.is_zero(images) == empty : Bool},truth: {empty || (U32.is_zero(empty_geometry_channels) || (U32.is_zero(height) || (U32.is_zero(empty_geometry_width) || False{}))) == True{} : Bool}) ->
  EmptyGeometry<Bands.Geometry{images,empty_geometry_channels,height,empty_geometry_width}>:
  match empty:
    case True{}:
      Rows{
        %Equal.sym(U32,images,0,ShapeLaws.native_equal(images,0,empty_geometry_classified)) : {Nat.mul(U32.to_nat(_),U32.to_nat(height)) == 0n : Nat}
        {==}}
    case False{}: channels_selected(U32.is_zero(empty_geometry_channels),images,empty_geometry_channels,height,empty_geometry_width,{==},truth)

def empty_geometry_classified(+geometry: Bands.Geometry,truth: {Shapes.has_zero(BandCount.extents(geometry)) == True{} : Bool}) -> EmptyGeometry<geometry>:
  Bands.Geometry{+images,+empty_geometry_channels,+height,+empty_geometry_width} = geometry
  images_selected(U32.is_zero(images),images,empty_geometry_channels,height,empty_geometry_width,{==},truth)
