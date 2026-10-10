# Actual nearest driver branches. Repeated extents keep both scale axes;
# wrapped scaled labels alone are never used as an arithmetic certificate.
import Base
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_band.bend as Bands
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_band_upsample_nearest.bend as Nearest
import ./tensor_band_upsample_execution.bend as BandUpsampleExecution
import ./tensor_band_upsample_samples.bend as BandUpsampleSamples
import ./tensor_band_count.bend as BandCount
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ../tensor_window.bend as Window
import ./tensor_normal_densification.bend as Normal
import ./tensor_dense_observation.bend as DenseObservation
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_ready_densification.bend as ReadyDensification
import ./tensor_global_shape_count.bend as Global
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_count_refinement.bend as Count
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_dense_contract.bend as Dense
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_public_refinement.bend as Public
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./boolean_logic.bend as Boolean
import ./u32_full_range.bend as FullRange
import ./word_encoding.bend as WordEncoding
import ./word_multiplication.bend as Multiplication
import ./tensor_copy_values.bend as CopyValues
import ./tensor_view_metadata_refinement.bend as Metadata
import ./tensor_gather_values.bend as GatherValues

type FailedOutput<-Element: Data,-output: Tensors.Tensor<Element>,-shape: List<&2,U32>,-problem: Tensors.Problem> is Data:
  FailedOutput{normal: Global.Normal(~Element,output),
    status_equal: {Nearest.tensor_status(~Element,output) == Tensors.Failed{problem} : Tensors.Status},
    shape_equal: {Nearest.tensor_shape(~Element,output) == shape : List<&2,U32>}}

def failed_empty(~Element: Data,+shape: List<&2,U32>,+fill: Element,+workers: U32,+problem: Tensors.Problem) ->
  FailedOutput<Element,Tensors.Tensor{Tensors.buffer(~Element,fill,0),shape,fill,Tensors.Failed{problem},workers},shape,problem>:
  FailedOutput{Shape.Both{Dense.empty_ready(~Element,fill),Unit{}},{==},{==}}

def failed_dense(~Element: Data,+input: Buffer.Model<Element>,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+problem: Tensors.Problem) ->
  FailedOutput<Element,Layout.upsample_nearest(~Element,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,Tensors.Failed{problem},workers},scale),Layout.scaled_extents(shape,scale),problem>:
  failed_empty(~Element,Layout.scaled_extents(shape,scale),fill,workers,problem)

def failed_dense_observed(~Element: Data,-input: Storage.Buffer<Element>,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+problem: Tensors.Problem,
  observation: Buffer.Observation<Element,input>) ->
  FailedOutput<Element,Layout.upsample_nearest(~Element,Tensors.Tensor{input,shape,fill,Tensors.Failed{problem},workers},scale),Layout.scaled_extents(shape,scale),problem>:
  Buffer.Observation{+model,equation} = observation
  %equation : FailedOutput<Element,Layout.upsample_nearest(~Element,Tensors.Tensor{_,shape,fill,Tensors.Failed{problem},workers},scale),Layout.scaled_extents(shape,scale),problem>
  failed_dense(~Element,model,shape,scale,fill,workers,problem)

def failed_dense_native(~Element: Data,input: Storage.Buffer<Element>,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+problem: Tensors.Problem) ->
  FailedOutput<Element,Layout.upsample_nearest(~Element,Tensors.Tensor{input,shape,fill,Tensors.Failed{problem},workers},scale),Layout.scaled_extents(shape,scale),problem>:
  failed_dense_observed(~Element,input,shape,scale,fill,workers,problem,Buffer.observe(~Element,input))

def failed_band(~Element: Data,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+problem: Tensors.Problem) ->
  FailedOutput<Element,Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Failed{problem},workers},scale),Layout.scaled_extents(shape,scale),problem>:
  failed_empty(~Element,Layout.scaled_extents(shape,scale),fill,workers,problem)

type RepeatedSource<-shape: List<&2,U32>,-scale: U32> is Data:
  RepeatedSource{view: Views.View,
    actual: {Layout.repeated(Tensors.view_of(shape),scale) == Some{view} : Maybe<&2,Views.View>},
    extents: {Views.axis_extents_of(view) == Nearest.repeated_extents(shape,scale) : List<&2,U32>},
    offset: {CopyValues.view_offset(view) == 0 : U32}}

def repeated_three(+channels: U32,+height: U32,+width: U32,+scale: U32,+count: U32,
  exact: {ShapeLaws.elements([channels,height,width]) == U32.to_nat(count) : Nat}) -> RepeatedSource<[channels,height,width],scale>:
  RepeatedSource{BandUpsampleExecution.repeated_view(Bands.Geometry{1,channels,height,width},height,scale,count),
    BandUpsampleExecution.repeated(Bands.Geometry{1,channels,height,width},height,scale,count,exact),{==},{==}}

def repeated_four(+images: U32,+channels: U32,+height: U32,+width: U32,+scale: U32,+count: U32,
  exact: {ShapeLaws.elements([images,channels,height,width]) == U32.to_nat(count) : Nat}) -> RepeatedSource<[images,channels,height,width],scale>:
  +stride = Bool.pick(U32,U32.is_zero(count),0,1)
  +view = {Views.View{[Views.Axis{images,(((stride * width : U32) * height : U32) * channels : U32)},Views.Axis{channels,((stride * width : U32) * height : U32)},
    Views.Axis{height,(stride * width : U32)},Views.Axis{scale,0},Views.Axis{width,stride},Views.Axis{scale,0}],0} : Views.View}
  RepeatedSource{view,
    %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of([images,channels,height,width]),Some{BandUpsampleExecution.canonical_view([images,channels,height,width],count)},
      BandUpsampleExecution.view([images,channels,height,width],count,exact)) : {Layout.repeated(_,scale) == Some{view} : Maybe<&2,Views.View>} {==},
    {==},{==}}

def repeated_coherent(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+count: U32,
  exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat},coherence: Nearest.ShapeCoherence<geometry,shape>) -> RepeatedSource<shape,scale>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  match coherence:
    case Nearest.Batched{+shape_equal}:
      %Equal.sym(List<&2,U32>,shape,BandCount.extents(geometry),shape_equal) : RepeatedSource<_,scale>
      repeated_four(images,channels,height,width,scale,count,%shape_equal : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat} exact)
    case Nearest.Unbatched{image_equal,+shape_equal}:
      %Equal.sym(List<&2,U32>,shape,[channels,height,width],shape_equal) : RepeatedSource<_,scale>
      repeated_three(channels,height,width,scale,count,%shape_equal : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat} exact)

def repeated_counted(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,counted: Global.Counted<shape>,coherence: Nearest.ShapeCoherence<geometry,shape>) -> RepeatedSource<shape,scale>:
  Global.Counted{+count,+accepted} = counted
  repeated_coherent(geometry,shape,scale,count,BandUpsampleExecution.count_exact(ShapeLaws.elements(shape),count,Count.count_of(shape,count,accepted)),coherence)

def rejected_count_views(~Element: Data,+source: Views.View,target: Maybe<&2,Views.View>,+input: Buffer.Model<Element>,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32) ->
  FailedOutput<Element,Layout.result_of(~Element,Layout.gather_views(~Element,"upsample_nearest",Some{source},target,None{},Buffer.pack(~Element,input),shape,Layout.scaled_extents(shape,scale),fill,workers)),
    Layout.scaled_extents(shape,scale),Tensors.Problem{"upsample_nearest",Tensors.InvalidArgument{shape}}>:
  match target:
    case None{}: failed_empty(~Element,Layout.scaled_extents(shape,scale),fill,workers,Tensors.Problem{"upsample_nearest",Tensors.InvalidArgument{shape}})
    case Some{view}: failed_empty(~Element,Layout.scaled_extents(shape,scale),fill,workers,Tensors.Problem{"upsample_nearest",Tensors.InvalidArgument{shape}})

def rejected_repeated_count(~Element: Data,+input: Buffer.Model<Element>,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,
  source: RepeatedSource<shape,scale>,rejected: {Tensors.count_of(Nearest.repeated_extents(shape,scale)) == None{} : Maybe<&2,U32>}) ->
  FailedOutput<Element,Layout.upsample_nearest(~Element,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,Tensors.Usable{},workers},scale),
    Layout.scaled_extents(shape,scale),Tensors.Problem{"upsample_nearest",Tensors.InvalidArgument{shape}}>:
  RepeatedSource{+view,+actual,+extents,offset} = source
  %Equal.sym(Maybe<&2,Views.View>,Layout.repeated(Tensors.view_of(shape),scale),Some{view},actual) :
    FailedOutput<Element,Layout.result_of(~Element,Layout.upsample_view(~Element,_,Tensors.Usable{},Buffer.pack(~Element,input),shape,scale,fill,workers)),Layout.scaled_extents(shape,scale),Tensors.Problem{"upsample_nearest",Tensors.InvalidArgument{shape}}>
  %Equal.sym(List<&2,U32>,Views.axis_extents_of(view),Nearest.repeated_extents(shape,scale),extents) :
    FailedOutput<Element,Layout.result_of(~Element,Layout.gather_views(~Element,"upsample_nearest",Some{view},Tensors.view_of(_),Tensors.count_of(_),Buffer.pack(~Element,input),shape,Layout.scaled_extents(shape,scale),fill,workers)),Layout.scaled_extents(shape,scale),Tensors.Problem{"upsample_nearest",Tensors.InvalidArgument{shape}}>
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(Nearest.repeated_extents(shape,scale)),None{},rejected) :
    FailedOutput<Element,Layout.result_of(~Element,Layout.gather_views(~Element,"upsample_nearest",Some{view},Tensors.view_of(Nearest.repeated_extents(shape,scale)),_,Buffer.pack(~Element,input),shape,Layout.scaled_extents(shape,scale),fill,workers)),Layout.scaled_extents(shape,scale),Tensors.Problem{"upsample_nearest",Tensors.InvalidArgument{shape}}>
  rejected_count_views(~Element,view,Tensors.view_of(Nearest.repeated_extents(shape,scale)),input,shape,scale,fill,workers)

def pair_room(+extents: List<&2,U32>,+offset: Nat,+limit: Nat,
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},room: {Nat.is_le(ShapeLaws.elements(extents),limit) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(List.take(&2,U32,List.drop(&2,U32,extents,offset),2n)),limit) == True{} : Bool}:
  CountCompletion.prefix_room(List.drop(&2,U32,extents,offset),2n,limit,
    CountCompletion.suffix_positive(extents,offset,positive),CountCompletion.suffix_room(extents,offset,limit,positive,room))

def merged_pair_elements(+height: U32,+width: U32,+scale: U32,
  +height_exact: {Nat.mul(U32.to_nat(height),U32.to_nat(scale)) == U32.to_nat((height * scale : U32)) : Nat},
  +width_exact: {Nat.mul(U32.to_nat(width),U32.to_nat(scale)) == U32.to_nat((width * scale : U32)) : Nat}) ->
  {ShapeLaws.elements([height,scale,width,scale]) == ShapeLaws.elements([(height * scale : U32),(width * scale : U32)]) : Nat}:
  %Equal.sym(Nat,Nat.mul(U32.to_nat(scale),1n),U32.to_nat(scale),ShapeLaws.times_one(U32.to_nat(scale))) :
    {Nat.mul(U32.to_nat(height),Nat.mul(U32.to_nat(scale),Nat.mul(U32.to_nat(width),_))) == ShapeLaws.elements([(height * scale : U32),(width * scale : U32)]) : Nat}
  %Equal.sym(Nat,Nat.mul(U32.to_nat((width * scale : U32)),1n),U32.to_nat((width * scale : U32)),ShapeLaws.times_one(U32.to_nat((width * scale : U32)))) :
    {Nat.mul(U32.to_nat(height),Nat.mul(U32.to_nat(scale),Nat.mul(U32.to_nat(width),U32.to_nat(scale)))) == Nat.mul(U32.to_nat((height * scale : U32)),_) : Nat}
  %RowMajor.assoc(U32.to_nat(height),U32.to_nat(scale),Nat.mul(U32.to_nat(width),U32.to_nat(scale))) :
    {_ == Nat.mul(U32.to_nat((height * scale : U32)),U32.to_nat((width * scale : U32))) : Nat}
  Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat(height),U32.to_nat(scale)),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),
    Nat.mul(U32.to_nat((height * scale : U32)),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),Nat.mul(U32.to_nat((height * scale : U32)),U32.to_nat((width * scale : U32))),
    Equal.cong(Nat,Nat,value => Nat.mul(value,Nat.mul(U32.to_nat(width),U32.to_nat(scale))),Nat.mul(U32.to_nat(height),U32.to_nat(scale)),U32.to_nat((height * scale : U32)),height_exact),
    Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat((height * scale : U32)),value),Nat.mul(U32.to_nat(width),U32.to_nat(scale)),U32.to_nat((width * scale : U32)),width_exact))

type PairWord<-left: U32,-right: U32> is Data:
  PairWord{checked: {Shapes.checked_multiply(left,right) == Some{(left * right : U32)} : Maybe<&2,U32>},
    exact: {Nat.mul(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat((left * right : U32)) : Nat}}

def pair_word(+extents: List<&2,U32>,+offset: Nat,+left: U32,+right: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(limit)) == True{} : Bool},
  slice: {List.take(&2,U32,List.drop(&2,U32,extents,offset),2n) == [left,right] : List<&2,U32>}) -> PairWord<left,right>:
  +bounded = {
    %ShapeLaws.times_one(U32.to_nat(right)) : {Nat.is_le(Nat.mul(U32.to_nat(left),_),U32.to_nat(limit)) == True{} : Bool}
    %slice : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(limit)) == True{} : Bool}
    pair_room(extents,offset,U32.to_nat(limit),positive,room) : {Nat.is_le(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit)) == True{} : Bool}}
  PairWord{CountCompletion.checked_product(left,right,limit,maximum,bounded),
    Equal.sym(Nat,U32.to_nat((left * right : U32)),Nat.mul(U32.to_nat(left),U32.to_nat(right)),FullRange.multiply(left,right,limit,bounded))}

def pair_exact(+left: U32,+right: U32,word: PairWord<left,right>) -> {Nat.mul(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat((left * right : U32)) : Nat}:
  PairWord{checked,exact} = word
  exact

def pair_present(+left: U32,+right: U32,word: PairWord<left,right>) -> {Views.has_count(Shapes.checked_multiply(left,right)) == True{} : Bool}:
  PairWord{checked,exact} = word
  Equal.cong(Maybe<&2,U32>,Bool,Views.has_count,Shapes.checked_multiply(left,right),Some{(left * right : U32)},checked)

type PairWords<-geometry: Bands.Geometry,-scale: U32> is Data:
  PairWords{height: PairWord<BandUpsampleExecution.producer_height(geometry),scale>,width: PairWord<BandUpsampleExecution.producer_width(geometry),scale>}

def pair_words_coherent(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  coherence: Nearest.ShapeCoherence<geometry,shape>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(Nearest.repeated_extents(shape,scale))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Nearest.repeated_extents(shape,scale)),U32.to_nat(limit)) == True{} : Bool}) -> PairWords<geometry,scale>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  match coherence:
    case Nearest.Batched{+shape_equal}:
      PairWords{
        pair_word([images,channels,height,scale,width,scale],2n,height,scale,limit,maximum,
          %Equal.cong(List<&2,U32>,List<&2,U32>,extents => Nearest.repeated_extents(extents,scale),shape,BandCount.extents(geometry),shape_equal) : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} positive,
          %Equal.cong(List<&2,U32>,List<&2,U32>,extents => Nearest.repeated_extents(extents,scale),shape,BandCount.extents(geometry),shape_equal) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(limit)) == True{} : Bool} room,{==}),
        pair_word([images,channels,height,scale,width,scale],4n,width,scale,limit,maximum,
          %Equal.cong(List<&2,U32>,List<&2,U32>,extents => Nearest.repeated_extents(extents,scale),shape,BandCount.extents(geometry),shape_equal) : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} positive,
          %Equal.cong(List<&2,U32>,List<&2,U32>,extents => Nearest.repeated_extents(extents,scale),shape,BandCount.extents(geometry),shape_equal) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(limit)) == True{} : Bool} room,{==})}
    case Nearest.Unbatched{image_equal,+shape_equal}:
      PairWords{
        pair_word([channels,height,scale,width,scale],1n,height,scale,limit,maximum,
          %Equal.cong(List<&2,U32>,List<&2,U32>,extents => Nearest.repeated_extents(extents,scale),shape,[channels,height,width],shape_equal) : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} positive,
          %Equal.cong(List<&2,U32>,List<&2,U32>,extents => Nearest.repeated_extents(extents,scale),shape,[channels,height,width],shape_equal) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(limit)) == True{} : Bool} room,{==}),
        pair_word([channels,height,scale,width,scale],3n,width,scale,limit,maximum,
          %Equal.cong(List<&2,U32>,List<&2,U32>,extents => Nearest.repeated_extents(extents,scale),shape,[channels,height,width],shape_equal) : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} positive,
          %Equal.cong(List<&2,U32>,List<&2,U32>,extents => Nearest.repeated_extents(extents,scale),shape,[channels,height,width],shape_equal) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(limit)) == True{} : Bool} room,{==})}

def merged_elements(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,coherence: Nearest.ShapeCoherence<geometry,shape>,words: PairWords<geometry,scale>) ->
  {ShapeLaws.elements(Nearest.repeated_extents(shape,scale)) == ShapeLaws.elements(Layout.scaled_extents(shape,scale)) : Nat}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  match coherence:
    case Nearest.Batched{+shape_equal}:
      PairWords{height_word,width_word} = words
      %Equal.sym(List<&2,U32>,shape,BandCount.extents(geometry),shape_equal) : {ShapeLaws.elements(Nearest.repeated_extents(_,scale)) == ShapeLaws.elements(Layout.scaled_extents(_,scale)) : Nat}
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),value)),ShapeLaws.elements([height,scale,width,scale]),ShapeLaws.elements([(height * scale : U32),(width * scale : U32)]),
        merged_pair_elements(height,width,scale,pair_exact(height,scale,height_word),pair_exact(width,scale,width_word)))
    case Nearest.Unbatched{image_equal,+shape_equal}:
      PairWords{height_word,width_word} = words
      %Equal.sym(List<&2,U32>,shape,[channels,height,width],shape_equal) : {ShapeLaws.elements(Nearest.repeated_extents(_,scale)) == ShapeLaws.elements(Layout.scaled_extents(_,scale)) : Nat}
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(channels),value),ShapeLaws.elements([height,scale,width,scale]),ShapeLaws.elements([(height * scale : U32),(width * scale : U32)]),
        merged_pair_elements(height,width,scale,pair_exact(height,scale,height_word),pair_exact(width,scale,width_word)))

def repeated_scale_positive(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,coherence: Nearest.ShapeCoherence<geometry,shape>,
  positive: {Nat.is_lt(0n,ShapeLaws.elements(Nearest.repeated_extents(shape,scale))) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(scale)) == True{} : Bool}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  match coherence:
    case Nearest.Batched{shape_equal}:
      RowMajor.left_factor_positive(U32.to_nat(scale),ShapeLaws.elements([width,scale]),
        CountCompletion.suffix_positive([images,channels,height,scale,width,scale],3n,
          %Equal.cong(List<&2,U32>,List<&2,U32>,extents => Nearest.repeated_extents(extents,scale),shape,BandCount.extents(geometry),shape_equal) : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} positive))
    case Nearest.Unbatched{image_equal,shape_equal}:
      RowMajor.left_factor_positive(U32.to_nat(scale),ShapeLaws.elements([width,scale]),
        CountCompletion.suffix_positive([channels,height,scale,width,scale],2n,
          %Equal.cong(List<&2,U32>,List<&2,U32>,extents => Nearest.repeated_extents(extents,scale),shape,[channels,height,width],shape_equal) : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} positive))

type Active<-geometry: Bands.Geometry,-shape: List<&2,U32>,-scale: U32> is Data:
  Active{guard: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool},enabled: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}}

def active_words(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+count: U32,
  counted: Count.Count<ShapeLaws.elements(Nearest.repeated_extents(shape,scale)),count>,+coherence: Nearest.ShapeCoherence<geometry,shape>,
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(Nearest.repeated_extents(shape,scale))) == True{} : Bool},words: PairWords<geometry,scale>) -> Active<geometry,shape,scale>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Count.Count{+exact,bounded} = counted
  PairWords{+height_word,+width_word} = words
  +words = {PairWords{height_word,width_word} : PairWords<geometry,scale>}
  +merged = merged_elements(geometry,shape,scale,coherence,words)
  +scaled_exact = Equal.trans(Nat,ShapeLaws.elements(Layout.scaled_extents(shape,scale)),ShapeLaws.elements(Nearest.repeated_extents(shape,scale)),U32.to_nat(count),
    Equal.sym(Nat,ShapeLaws.elements(Nearest.repeated_extents(shape,scale)),ShapeLaws.elements(Layout.scaled_extents(shape,scale)),merged),exact)
  +scaled_count = CountCompletion.count_of(Layout.scaled_extents(shape,scale),count,scaled_exact,bounded)
  +scale_false = Nearest.positive_word_not_zero(scale,repeated_scale_positive(geometry,shape,scale,coherence,positive))
  +scale_true = {%Equal.sym(Bool,U32.is_zero(scale),False{},scale_false) : {Bool.not(_) == True{} : Bool} {==} : {Bool.not(U32.is_zero(scale)) == True{} : Bool}}
  +height_true = pair_present(height,scale,height_word)
  +width_true = pair_present(width,scale,width_word)
  +count_true = Equal.cong(Maybe<&2,U32>,Bool,Views.has_count,Tensors.count_of(Layout.scaled_extents(shape,scale)),Some{count},scaled_count)
  +guard = Nearest.and_true(Bool.not(U32.is_zero(scale)) && Views.has_count(Shapes.checked_multiply(height,scale)) && Views.has_count(Shapes.checked_multiply(width,scale)),Views.has_count(Tensors.count_of(Layout.scaled_extents(shape,scale))),
    Nearest.and_true(Bool.not(U32.is_zero(scale)) && Views.has_count(Shapes.checked_multiply(height,scale)),Views.has_count(Shapes.checked_multiply(width,scale)),
      Nearest.and_true(Bool.not(U32.is_zero(scale)),Views.has_count(Shapes.checked_multiply(height,scale)),scale_true,height_true),width_true),count_true)
  +nonempty = CountCompletion.positive_not_zero(Layout.scaled_extents(shape,scale),%merged : {Nat.is_lt(0n,_) == True{} : Bool} positive)
  Active{guard,Nearest.and_true(BandUpsampleExecution.guard(geometry,shape,scale),Bool.not(Shapes.has_zero(Layout.scaled_extents(shape,scale))),guard,
    %Equal.sym(Bool,Shapes.has_zero(Layout.scaled_extents(shape,scale)),False{},nonempty) : {Bool.not(_) == True{} : Bool} {==})}

def active_counted_at(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+count: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  counted: Count.Count<ShapeLaws.elements(Nearest.repeated_extents(shape,scale)),count>,+coherence: Nearest.ShapeCoherence<geometry,shape>,nonzero: {U32.is_zero(count) == False{} : Bool}) -> Active<geometry,shape,scale>:
  Count.Count{+exact,+bounded} = counted
  +positive = {%Equal.sym(Nat,ShapeLaws.elements(Nearest.repeated_extents(shape,scale)),U32.to_nat(count),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    CountCompletion.positive_word(count,nonzero) : {Nat.is_lt(0n,ShapeLaws.elements(Nearest.repeated_extents(shape,scale))) == True{} : Bool}}
  +room = {%Equal.sym(Nat,ShapeLaws.elements(Nearest.repeated_extents(shape,scale)),U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
    Bounds.as_nat_le(count,limit,%Equal.sym(U32,limit,Machine.word_max(),maximum) : {U32.is_le(count,_) == True{} : Bool} FullRange.maximum(count)) : {Nat.is_le(ShapeLaws.elements(Nearest.repeated_extents(shape,scale)),U32.to_nat(limit)) == True{} : Bool}}
  active_words(geometry,shape,scale,count,Count.Count{exact,bounded},coherence,positive,pair_words_coherent(geometry,shape,scale,limit,maximum,coherence,positive,room))

def active_counted(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+count: U32,
  counted: {Tensors.count_of(Nearest.repeated_extents(shape,scale)) == Some{count} : Maybe<&2,U32>},coherence: Nearest.ShapeCoherence<geometry,shape>,nonzero: {U32.is_zero(count) == False{} : Bool}) -> Active<geometry,shape,scale>:
  active_counted_at(geometry,shape,scale,count,Machine.word_max(),{==},Count.count_of(Nearest.repeated_extents(shape,scale),count,counted),coherence,nonzero)

def multiply_right_zero(+value: U32) -> {(value * 0 : U32) == 0 : U32}:
  Equal.trans(U32,(value * 0 : U32),U32.from_nat(Nat.mul(U32.to_nat(value),0n)),0,Multiplication.native_multiply(value,0),
    Equal.cong(Nat,U32,U32.from_nat,Nat.mul(U32.to_nat(value),0n),0n,RowMajor.multiply_zero_right(U32.to_nat(value))))

def scaled_pair_empty(+height: U32,+width: U32,+scale: U32,height_zero: Bool,scale_zero: Bool,width_zero: Bool,
  +height_classified: {U32.is_zero(height) == height_zero : Bool},+scale_classified: {U32.is_zero(scale) == scale_zero : Bool},+width_classified: {U32.is_zero(width) == width_zero : Bool},
  empty: {Shapes.has_zero([height,scale,width,scale]) == True{} : Bool}) -> {Shapes.has_zero([(height * scale : U32),(width * scale : U32)]) == True{} : Bool}:
  match height_zero scale_zero width_zero:
    case True{} scale_zero width_zero:
      %Equal.sym(U32,height,0,ShapeLaws.native_equal(height,0,height_classified)) : {Shapes.has_zero([(_ * scale : U32),(width * scale : U32)]) == True{} : Bool}
      %Equal.sym(U32,(0 * scale : U32),0,Arithmetic.multiply_zero_left(scale)) : {Shapes.has_zero([_,(width * scale : U32)]) == True{} : Bool}
      {==}
    case False{} True{} width_zero:
      %Equal.sym(U32,scale,0,ShapeLaws.native_equal(scale,0,scale_classified)) : {Shapes.has_zero([(height * _ : U32),(width * _ : U32)]) == True{} : Bool}
      %Equal.sym(U32,(height * 0 : U32),0,multiply_right_zero(height)) : {Shapes.has_zero([_,(width * 0 : U32)]) == True{} : Bool}
      {==}
    case False{} False{} True{}:
      %Equal.sym(U32,width,0,ShapeLaws.native_equal(width,0,width_classified)) : {Shapes.has_zero([(height * scale : U32),(_ * scale : U32)]) == True{} : Bool}
      %Equal.sym(U32,(0 * scale : U32),0,Arithmetic.multiply_zero_left(scale)) : {Shapes.has_zero([(height * scale : U32),_]) == True{} : Bool}
      Boolean.or_true_right(U32.is_zero((height * scale : U32)))
    case False{} False{} False{}:
      Empty.absurd({Shapes.has_zero([(height * scale : U32),(width * scale : U32)]) == True{} : Bool},WordEncoding.false_true(
        %scale_classified : {False{} || (False{} || (False{} || (_ || False{}))) == True{} : Bool}
        %width_classified : {False{} || (False{} || (_ || (U32.is_zero(scale) || False{}))) == True{} : Bool}
        %scale_classified : {False{} || (_ || (U32.is_zero(width) || (U32.is_zero(scale) || False{}))) == True{} : Bool}
        %height_classified : {_ || (U32.is_zero(scale) || (U32.is_zero(width) || (U32.is_zero(scale) || False{}))) == True{} : Bool} empty))

def scaled_empty_head(+head: U32,+tail: List<&2,U32>,+scale: U32,zero: Bool,+classified: {U32.is_zero(head) == zero : Bool},
  +empty: {Shapes.has_zero(head <> Nearest.repeated_extents(tail,scale)) == True{} : Bool},
  next: {Shapes.has_zero(Nearest.repeated_extents(tail,scale)) == True{} : Bool} -> {Shapes.has_zero(Layout.scaled_extents(tail,scale)) == True{} : Bool}) ->
  {Shapes.has_zero(head <> Layout.scaled_extents(tail,scale)) == True{} : Bool}:
  match zero:
    case True{}:
      %Equal.sym(Bool,U32.is_zero(head),True{},classified) : {_ || Shapes.has_zero(Layout.scaled_extents(tail,scale)) == True{} : Bool}
      {==}
    case False{}:
      %Equal.sym(Bool,U32.is_zero(head),False{},classified) : {_ || Shapes.has_zero(Layout.scaled_extents(tail,scale)) == True{} : Bool}
      next(%classified : {_ || Shapes.has_zero(Nearest.repeated_extents(tail,scale)) == True{} : Bool} empty)

def scaled_empty(+extents: List<&2,U32>,+scale: U32,empty: {Shapes.has_zero(Nearest.repeated_extents(extents,scale)) == True{} : Bool}) ->
  {Shapes.has_zero(Layout.scaled_extents(extents,scale)) == True{} : Bool}:
  match extents:
    case [+height,+width]: scaled_pair_empty(height,width,scale,U32.is_zero(height),U32.is_zero(scale),U32.is_zero(width),{==},{==},{==},empty)
    case Con{+head,Nil{}}: scaled_empty_head(head,[],scale,U32.is_zero(head),{==},empty,tail_empty => scaled_empty([],scale,tail_empty))
    case Con{+head,Con{+second,Con{+third,+rest}}}:
      +tail = {second <> third <> rest : List<&2,U32>}
      scaled_empty_head(head,tail,scale,U32.is_zero(head),{==},empty,tail_empty => scaled_empty(tail,scale,tail_empty))
    case []: Empty.absurd({Shapes.has_zero([]) == True{} : Bool},WordEncoding.false_true(empty))

def empty_guard(~Element: Data,+input: Buffer.Model<Element>,+source: Views.View,+extents: List<&2,U32>,+count: U32,+fill: Element,
  +source_shape: {Views.axis_extents_of(source) == extents : List<&2,U32>},source_offset: {CopyValues.view_offset(source) == 0 : U32},
  +empty: {Shapes.has_zero(extents) == True{} : Bool},input_exact: {U32.to_nat(count) == Buffer.length(~Element,input) : Nat},input_ready: Buffer.Sized<Element,input>) ->
  {CopyValues.copy_guard(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),BandUpsampleExecution.canonical_view(extents,0)) == True{} : Bool}:
  +exact = ShapeLaws.empty_product(extents,empty)
  +target_shape = BandUpsampleExecution.canonical_shape(extents,0,exact)
  +same = {%Equal.sym(Bool,Shapes.same(Views.shape(source),Views.shape(BandUpsampleExecution.canonical_view(extents,0))),
      Shapes.same_extents(Views.axis_extents_of(source),Views.axis_extents_of(BandUpsampleExecution.canonical_view(extents,0))),Nearest.view_shapes_projection(source,BandUpsampleExecution.canonical_view(extents,0))) : {_ == True{} : Bool}
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(source),extents,source_shape) : {Shapes.same_extents(_,Views.axis_extents_of(BandUpsampleExecution.canonical_view(extents,0))) == True{} : Bool}
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(BandUpsampleExecution.canonical_view(extents,0)),extents,target_shape) : {Shapes.same_extents(extents,_) == True{} : Bool}
    Nearest.same_extents(extents) : {Shapes.same(Views.shape(source),Views.shape(BandUpsampleExecution.canonical_view(extents,0))) == True{} : Bool}}
  CopyValues.joined_guard(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,0),BandUpsampleExecution.canonical_view(extents,0),same,BandUpsampleSamples.canonical_contiguous(extents,0,exact),
    CopyValues.valid_model(~Element,input,source,count,input_exact,input_ready,
      Nearest.empty_view_fits(source,count,%Equal.sym(List<&2,U32>,Views.axis_extents_of(source),extents,source_shape) : {Shapes.has_zero(_) == True{} : Bool} empty,source_offset)),
    CopyValues.valid_allocated(~Element,0,fill,BandUpsampleExecution.canonical_view(extents,0),BandUpsampleExecution.allocated_from_count(ShapeLaws.elements(extents),0,Count.Count{exact,{==}}),BandUpsampleSamples.canonical_fits(extents,0,exact)))

# An empty repeated view accepts a nonempty source buffer as well. In
# particular scale zero must not require the original tensor to be empty.
def empty_dense_model(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+input: Buffer.Model<Element>,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,
  counted: Global.Counted<shape>,well_formed: Dense.WellFormed<Element,shape,Buffer.content(~Element,input)>,source: RepeatedSource<shape,scale>,+empty: {Shapes.has_zero(Nearest.repeated_extents(shape,scale)) == True{} : Bool}) ->
  Nearest.UpsampleResult(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.upsample_nearest(~Element,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,Tensors.Usable{},workers},scale)):
  Buffer.Model{+input_tree,+input_size,+input_certificate} = input
  Global.Counted{+count,+source_count} = counted
  Dense.WellFormed{+input_length,input_content_ready} = well_formed
  RepeatedSource{+source_view,+actual,+extents,+offset} = source
  +view_shape = Nearest.repeated_extents(shape,scale)
  +exact = ShapeLaws.empty_product(view_shape,empty)
  +target = BandUpsampleExecution.canonical_view(view_shape,0)
  +bounded = {{==} : {U32.is_le(0,Machine.capacity(Default.limits())) == True{} : Bool}}
  +target_counted = BandUpsampleExecution.target_counted_from_count(view_shape,0,Count.Count{exact,bounded})
  +source_exact = BandUpsampleExecution.count_exact(ShapeLaws.elements(shape),count,Count.count_of(shape,count,source_count))
  +guarded = empty_guard(~Element,input,source_view,view_shape,count,fill,extents,offset,empty,
    Equal.trans(Nat,U32.to_nat(count),ShapeLaws.elements(shape),Buffer.length(~Element,input),Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),source_exact),Equal.sym(Nat,Buffer.length(~Element,input),ShapeLaws.elements(shape),input_length)),
    Buffer.ready_content(~Element,input,input_content_ready))
  +observed = Equal.cong(Operations.Outcome<OperationsLaws.PairOwners(Element)>,Buffer.Content<Element>,CopyValues.right_content(~Element),
    OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target)),
    OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target),OperationsLaws.copy(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target))
  %Equal.sym(Maybe<&2,Views.View>,Layout.repeated(Tensors.view_of(shape),scale),Some{source_view},actual) :
    Nearest.UpsampleResult(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.result_of(~Element,Layout.upsample_view(~Element,_,Tensors.Usable{},Buffer.pack(~Element,input),shape,scale,fill,workers)))
  %Equal.sym(List<&2,U32>,Views.axis_extents_of(source_view),view_shape,extents) :
    Nearest.UpsampleResult(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.result_of(~Element,Layout.gather_status(~Element,"upsample_nearest",Tensors.Usable{},Some{source_view},_,Layout.scaled_extents(shape,scale),Buffer.pack(~Element,input),shape,fill,workers)))
  %Equal.sym(Tensors.Tensor<Element> & Tensors.Tensor<Element>,Layout.gather_status(~Element,"upsample_nearest",Tensors.Usable{},Some{source_view},view_shape,Layout.scaled_extents(shape,scale),Buffer.pack(~Element,input),shape,fill,workers),
    Layout.copied(~Element,"upsample_nearest",shape,Layout.scaled_extents(shape,scale),fill,workers,Operations.copy(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target)),
    Public.gathered(~Element,"upsample_nearest",source_view,view_shape,Layout.scaled_extents(shape,scale),Buffer.pack(~Element,input),shape,fill,target,0,workers,BandUpsampleExecution.view(view_shape,0,exact),CountCompletion.count_of(view_shape,0,exact,bounded))) :
    Nearest.UpsampleResult(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.result_of(~Element,_))
  Nearest.copied_empty_result(~Element,~Context,~original,context,shape,Layout.scaled_extents(shape,scale),fill,workers,
    Operations.copy(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target),Nearest.copy_accepted(~Element,input,source_view,fill,target,guarded,target_counted),
    %Equal.sym(Buffer.Content<Element>,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target))),
      CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target)),observed) : Buffer.ContentReady<Element,_>
      CopyValues.copy_observed_ready(~Element,Buffer.pack(~Element,input),source_view,Tensors.buffer(~Element,fill,0),target,Nearest.allocation_zero_ready(~Element,fill)),
    Nearest.copy_zero_length(~Element,input,source_view,fill,target,guarded,target_counted,Nearest.allocation_zero_length(~Element,fill)),
    ShapeLaws.empty_product(Layout.scaled_extents(shape,scale),scaled_empty(shape,scale,empty)))

def Readback(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,status: Tensors.Status,output: Tensors.Tensor<Element>) -> Data:
  match status:
    case Tensors.Usable{}: Nearest.logical_values(~Element,~Context,~original,context,output)
    case Tensors.Failed{problem}: Unit

def UpsampleContract(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,output: Tensors.Tensor<Element>) -> Data:
  Shape.Both<Global.Normal(~Element,output),Shape.Both<{Nearest.tensor_shape(~Element,output) == shape : List<&2,U32>},Readback(~Element,~Context,~original,context,Nearest.tensor_status(~Element,output),output)>>

def ready_result(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,-output: Tensors.Tensor<Element>,
  receipt: Nearest.UpsampleResult(~Element,~Context,~original,context,shape,output)) -> UpsampleContract(~Element,~Context,~original,context,shape,output):
  Shape.Both{normal,Shape.Both{status_equal,Shape.Both{shape_equal,values}}} = receipt
  Shape.Both{normal,Shape.Both{shape_equal,
    %Equal.sym(Tensors.Status,Nearest.tensor_status(~Element,output),Tensors.Usable{},status_equal) : Readback(~Element,~Context,~original,context,_,output) values}}

def failed_result(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,+problem: Tensors.Problem,-output: Tensors.Tensor<Element>,receipt: FailedOutput<Element,output,shape,problem>) ->
  UpsampleContract(~Element,~Context,~original,context,shape,output):
  FailedOutput{normal,status_equal,shape_equal} = receipt
  Shape.Both{normal,Shape.Both{shape_equal,
    %Equal.sym(Tensors.Status,Nearest.tensor_status(~Element,output),Tensors.Failed{problem},status_equal) : Readback(~Element,~Context,~original,context,_,output) Unit{}}}

def empty_of_zero_when(+extents: List<&2,U32>,empty: Bool,classified: {Shapes.has_zero(extents) == empty : Bool},zero: {ShapeLaws.elements(extents) == 0n : Nat}) ->
  {Shapes.has_zero(extents) == True{} : Bool}:
  match empty:
    case True{}: classified
    case False{}:
      Empty.absurd({Shapes.has_zero(extents) == True{} : Bool},WordEncoding.false_true(%zero : {Nat.is_lt(0n,_) == True{} : Bool} CountCompletion.positive_elements(extents,classified)))

def empty_of_zero(+extents: List<&2,U32>,zero: {ShapeLaws.elements(extents) == 0n : Nat}) -> {Shapes.has_zero(extents) == True{} : Bool}:
  empty_of_zero_when(extents,Shapes.has_zero(extents),{==},zero)

def fallback_positive(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+input: Buffer.Model<Element>,+fill: Element,+workers: U32,
  disabled: {BandUpsampleExecution.enabled(geometry,shape,scale) == False{} : Bool},active: Active<geometry,shape,scale>) ->
  UpsampleContract(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.upsample_nearest(~Element,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,Tensors.Usable{},workers},scale)):
  Active{guard,enabled} = active
  Empty.absurd(UpsampleContract(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.upsample_nearest(~Element,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,Tensors.Usable{},workers},scale)),WordEncoding.false_true(
    Equal.trans(Bool,False{},BandUpsampleExecution.enabled(geometry,shape,scale),True{},Equal.sym(Bool,BandUpsampleExecution.enabled(geometry,shape,scale),False{},disabled),enabled)))

def fallback_count_zero(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+input: Buffer.Model<Element>,+fill: Element,+workers: U32,+count: U32,zero: Bool,
  +classified: {U32.is_zero(count) == zero : Bool},+counted: {Tensors.count_of(Nearest.repeated_extents(shape,scale)) == Some{count} : Maybe<&2,U32>},+coherence: Nearest.ShapeCoherence<geometry,shape>,
  source_count: Global.Counted<shape>,well_formed: Dense.WellFormed<Element,shape,Buffer.content(~Element,input)>,source: RepeatedSource<shape,scale>,disabled: {BandUpsampleExecution.enabled(geometry,shape,scale) == False{} : Bool}) ->
  UpsampleContract(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.upsample_nearest(~Element,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,Tensors.Usable{},workers},scale)):
  match zero:
    case True{}:
      ready_result(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.upsample_nearest(~Element,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,Tensors.Usable{},workers},scale),
        empty_dense_model(~Element,~Context,~original,context,input,shape,scale,fill,workers,source_count,well_formed,source,
          empty_of_zero(Nearest.repeated_extents(shape,scale),Equal.trans(Nat,ShapeLaws.elements(Nearest.repeated_extents(shape,scale)),U32.to_nat(count),0n,
            BandUpsampleExecution.count_exact(ShapeLaws.elements(Nearest.repeated_extents(shape,scale)),count,Count.count_of(Nearest.repeated_extents(shape,scale),count,counted)),
            Equal.cong(U32,Nat,U32.to_nat,count,0,ShapeLaws.native_equal(count,0,classified))))))
    case False{}:
      fallback_positive(~Element,~Context,~original,context,geometry,shape,scale,input,fill,workers,disabled,active_counted(geometry,shape,scale,count,counted,coherence,classified))

def fallback_count(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+input: Buffer.Model<Element>,+fill: Element,+workers: U32,result: Maybe<&2,U32>,
  computed: {Tensors.count_of(Nearest.repeated_extents(shape,scale)) == result : Maybe<&2,U32>},+coherence: Nearest.ShapeCoherence<geometry,shape>,+source_count: Global.Counted<shape>,
  +well_formed: Dense.WellFormed<Element,shape,Buffer.content(~Element,input)>,+source: RepeatedSource<shape,scale>,+disabled: {BandUpsampleExecution.enabled(geometry,shape,scale) == False{} : Bool}) ->
  UpsampleContract(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.upsample_nearest(~Element,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,Tensors.Usable{},workers},scale)):
  match result:
    case None{}:
      failed_result(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Tensors.Problem{"upsample_nearest",Tensors.InvalidArgument{shape}},
        Layout.upsample_nearest(~Element,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,Tensors.Usable{},workers},scale),rejected_repeated_count(~Element,input,shape,scale,fill,workers,source,computed))
    case Some{+count}:
      fallback_count_zero(~Element,~Context,~original,context,geometry,shape,scale,input,fill,workers,count,U32.is_zero(count),{==},computed,coherence,source_count,well_formed,source,disabled)

def fallback_band_equation(~Element: Data,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,
  disabled: {BandUpsampleExecution.enabled(geometry,shape,scale) == False{} : Bool}) ->
  {Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale) ==
    Layout.upsample_nearest(~Element,Tensors.Tensor{Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))),shape,fill,Tensors.Usable{},workers},scale) : Tensors.Tensor<Element>}:
  Bands.Geometry{images,channels,height,width} = geometry
  %Equal.sym(Bool,BandUpsampleExecution.enabled(geometry,shape,scale),False{},disabled) :
    {Layout.upsample_supported(~Element,_,Model.pack(~Element,bands),geometry,shape,scale,fill,workers) ==
      Layout.upsample_nearest(~Element,Tensors.Tensor{Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))),shape,fill,Tensors.Usable{},workers},scale) : Tensors.Tensor<Element>}
  {==}

# One equation identifies the actual disabled selector with the existing
# dense operation. Source preparation is shared across all dense fallbacks.
def fallback_selected(~Element: Data,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32) ->
  {Layout.upsample_supported(~Element,False{},bands,geometry,shape,scale,fill,workers) ==
    Layout.upsample_nearest(~Element,DenseObservation.dense_tensor(~Element,Tensors.Banded{bands,geometry,shape,fill,Tensors.Usable{},workers}),scale) : Tensors.Tensor<Element>}:
  {==}

def fallback_selected_prepared(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,
  normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  values: Dense.TensorValues(~Element,~Context,~original,context,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Shape.Both<{Layout.upsample_supported(~Element,False{},Model.pack(~Element,bands),geometry,shape,scale,fill,workers) ==
    Layout.upsample_nearest(~Element,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),scale) : Tensors.Tensor<Element>},
    ReadyDensification.Prepared(~Element,~Context,~original,context,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})>:
  Shape.Both{fallback_selected(~Element,Model.pack(~Element,bands),geometry,shape,scale,fill,workers),
    ReadyDensification.prepared_owner(~Element,~Context,~original,context,BandConcatExecution.BandedOwner{bands,geometry,shape,fill,Tensors.Usable{},workers},normal,{==},values)}

def fallback_band_observed(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,
  +counted: Global.Counted<shape>,+coherent: {Window.band_shape(shape,geometry) == True{} : Bool},+coherence: Nearest.ShapeCoherence<geometry,shape>,+disabled: {BandUpsampleExecution.enabled(geometry,shape,scale) == False{} : Bool},
  observation: Buffer.Observation<Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))>) ->
  UpsampleContract(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale)):
  Buffer.Observation{+input,+equation} = observation
  +well_formed = {%Buffer.packed_contents(~Element,input) : Dense.WellFormed<Element,shape,_>
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input),Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))),equation) : Dense.WellFormed<Element,shape,Buffer.contents(~Element,_)>
    Normal.gathered(~Element,geometry,bands,shape,fill,counted,coherent) : Dense.WellFormed<Element,shape,Buffer.content(~Element,input)>}
  %Equal.sym(Tensors.Tensor<Element>,Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale),
    Layout.upsample_nearest(~Element,Tensors.Tensor{Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))),shape,fill,Tensors.Usable{},workers},scale),fallback_band_equation(~Element,bands,geometry,shape,scale,fill,workers,disabled)) :
    UpsampleContract(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),_)
  %equation : UpsampleContract(~Element,~Context,~original,context,Layout.scaled_extents(shape,scale),Layout.upsample_nearest(~Element,Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers},scale))
  fallback_count(~Element,~Context,~original,context,geometry,shape,scale,input,fill,workers,Tensors.count_of(Nearest.repeated_extents(shape,scale)),{==},coherence,counted,well_formed,repeated_counted(geometry,shape,scale,counted,coherence),disabled)

def ready_band_when(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,enabled: Bool,
  +classified: {BandUpsampleExecution.enabled(geometry,shape,scale) == enabled : Bool},normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  values: BandValues.Values(~Element,~Context,~original,context,geometry,Model.content(~Element,bands))) ->
  UpsampleContract(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),
    Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale)):
  match enabled:
    case True{}:
      Shape.Both{metadata,Shape.Both{counted,+coherent}} = normal
      ready_result(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),
        Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale),
        Nearest.enabled_result(~Element,~Context,~original,context,geometry,shape,scale,fill,workers,bands,%Model.packed_contents(~Element,bands) : BandShape.WellFormed<Element,geometry,_> metadata,values,
          Nearest.coherence_from_decoded(shape,geometry,BandCount.decoded(shape,geometry,coherent)),classified))
    case False{}:
      Shape.Both{metadata,Shape.Both{+counted,+coherent}} = normal
      fallback_band_observed(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},bands,geometry,shape,scale,fill,workers,counted,coherent,
        Nearest.coherence_from_decoded(shape,geometry,BandCount.decoded(shape,geometry,coherent)),classified,
        Buffer.observe(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))))

# The selector and every fallback count are computed here from the actual
# native call. Source invariants and finite source values are the premises.
def ready_band(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,
  normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),values: BandValues.Values(~Element,~Context,~original,context,geometry,Model.content(~Element,bands))) ->
  UpsampleContract(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),
    Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scale)):
  ready_band_when(~Element,~Context,~original,context,bands,geometry,shape,scale,fill,workers,BandUpsampleExecution.enabled(geometry,shape,scale),{==},normal,values)

def BandResult(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,status: Tensors.Status,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,output: Tensors.Tensor<Element>) -> Data:
  match status:
    case Tensors.Usable{}: UpsampleContract(~Element,~BandUpsampleExecution.Context<Context>,~BandUpsampleExecution.value(~Element,~Context,~original),BandUpsampleExecution.Context{context,geometry,scale},Layout.scaled_extents(shape,scale),output)
    case Tensors.Failed{problem}: FailedOutput<Element,output,Layout.scaled_extents(shape,scale),problem>

def SourceValues(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,status: Tensors.Status,+geometry: Bands.Geometry,+content: Model.Content<Element>) -> Data:
  match status:
    case Tensors.Usable{}: BandValues.Values(~Element,~Context,~original,context,geometry,content)
    case Tensors.Failed{problem}: Unit

def band_status(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,status: Tensors.Status,
  normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers}),values: SourceValues(~Element,~Context,~original,context,status,geometry,Model.content(~Element,bands))) ->
  BandResult(~Element,~Context,~original,context,status,geometry,shape,scale,
    Layout.upsample_nearest(~Element,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers},scale)):
  match status:
    case Tensors.Usable{}: ready_band(~Element,~Context,~original,context,bands,geometry,shape,scale,fill,workers,normal,values)
    case Tensors.Failed{+problem}:
      failed_band(~Element,bands,geometry,shape,scale,fill,workers,problem)

def band_observed(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,-bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+status: Tensors.Status,
  observation: Model.Observation<Element,bands>,+normal: Global.Normal(~Element,Tensors.Banded{bands,geometry,shape,fill,status,workers}),
  +values: SourceValues(~Element,~Context,~original,context,status,geometry,Model.contents(~Element,bands))) ->
  BandResult(~Element,~Context,~original,context,status,geometry,shape,scale,
    Layout.upsample_nearest(~Element,Tensors.Banded{bands,geometry,shape,fill,status,workers},scale)):
  Model.Observation{+model,+equation} = observation
  %equation : BandResult(~Element,~Context,~original,context,status,geometry,shape,scale,
    Layout.upsample_nearest(~Element,Tensors.Banded{_,geometry,shape,fill,status,workers},scale))
  band_status(~Element,~Context,~original,context,model,geometry,shape,scale,fill,workers,status,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) : Global.Normal(~Element,Tensors.Banded{_,geometry,shape,fill,status,workers}) normal,
    %Model.packed_contents(~Element,model) : SourceValues(~Element,~Context,~original,context,status,geometry,_)
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) : SourceValues(~Element,~Context,~original,context,status,geometry,Model.contents(~Element,_)) values)

# Original native bands are observed once. Ready and sticky Failed inputs
# share this invocation; neither a guard nor a completed nearest equation is
# an argument of the public factory.
def bands(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,native_bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+status: Tensors.Status,
  normal: Global.Normal(~Element,Tensors.Banded{native_bands,geometry,shape,fill,status,workers}),
  values: SourceValues(~Element,~Context,~original,context,status,geometry,Model.contents(~Element,native_bands))) ->
  BandResult(~Element,~Context,~original,context,status,geometry,shape,scale,
    Layout.upsample_nearest(~Element,Tensors.Banded{native_bands,geometry,shape,fill,status,workers},scale)):
  band_observed(~Element,~Context,~original,context,native_bands,geometry,shape,scale,fill,workers,status,Model.observe(~Element,native_bands),normal,values)

# A dense input of any rank is one gather through the repeated view. The
# statements below read the extents of that view from the actual view
# computation, and the elements of the scaled label from the count the call
# itself accepted; a wrapped scaled extent occurs only with an empty result.
def repeated_extents_or(result: Maybe<&2,List<&2,Views.Axis>>,fallback: List<&2,U32>) -> List<&2,U32>:
  match result:
    case Some{axes}: Views.axis_extents(axes)
    case None{}: fallback

def prepended_extents(result: Maybe<&2,List<&2,Views.Axis>>,+extent: U32,+stride: U32,+fallback: List<&2,U32>,
  previous: {repeated_extents_or(result,fallback) == fallback : List<&2,U32>}) ->
  {repeated_extents_or(Layout.prepend_repeated(Views.Axis{extent,stride},result),extent <> fallback) == extent <> fallback : List<&2,U32>}:
  match result:
    case Some{+axes}: Equal.cong(List<&2,U32>,List<&2,U32>,tail => extent <> tail,Views.axis_extents(axes),fallback,previous)
    case None{}: {==}

def repeated_axes_extents(+axes: List<&2,Views.Axis>,+scale: U32) ->
  {repeated_extents_or(Layout.repeated_axes(axes,scale),Nearest.repeated_extents(Views.axis_extents(axes),scale)) ==
    Nearest.repeated_extents(Views.axis_extents(axes),scale) : List<&2,U32>}:
  match axes:
    case [Views.Axis{height,height_stride},Views.Axis{width,width_stride}]: {==}
    case Con{Views.Axis{extent,stride},Nil{}}: {==}
    case Con{Views.Axis{+extent,+stride},Con{Views.Axis{+second,+second_stride},Con{Views.Axis{+third,+third_stride},+rest}}}:
      +tail = {Views.Axis{second,second_stride} <> Views.Axis{third,third_stride} <> rest : List<&2,Views.Axis>}
      prepended_extents(Layout.repeated_axes(tail,scale),extent,stride,Nearest.repeated_extents(Views.axis_extents(tail),scale),repeated_axes_extents(tail,scale))
    case []: {==}

def has_view(view: Maybe<&2,Views.View>) -> Bool:
  match view:
    case Some{view}: True{}
    case None{}: False{}

def offset_extents(result: Maybe<&2,List<&2,Views.Axis>>,+view: Views.View,+offset: U32,+fallback: List<&2,U32>,
  previous: {repeated_extents_or(result,fallback) == fallback : List<&2,U32>},
  actual: {Layout.with_offset(result,offset) == Some{view} : Maybe<&2,Views.View>}) -> {Views.axis_extents_of(view) == fallback : List<&2,U32>}:
  match result:
    case Some{+axes}:
      Views.View{+view_axes,+view_offset} = view
      Equal.trans(List<&2,U32>,Views.axis_extents(view_axes),Views.axis_extents(axes),fallback,
        Equal.sym(List<&2,U32>,Views.axis_extents(axes),Views.axis_extents(view_axes),
          Equal.cong(Maybe<&2,Views.View>,List<&2,U32>,found => Metadata.observed_extents(found,fallback),Some{Views.View{axes,offset}},Some{Views.View{view_axes,view_offset}},actual)),previous)
    case None{}:
      Empty.absurd({Views.axis_extents_of(view) == fallback : List<&2,U32>},WordEncoding.false_true(
        Equal.cong(Maybe<&2,Views.View>,Bool,has_view,None{},Some{view},actual)))

def repeated_view_extents(base: Maybe<&2,Views.View>,+shape: List<&2,U32>,+scale: U32,+view: Views.View,
  +based: {Tensors.view_of(shape) == base : Maybe<&2,Views.View>},
  actual: {Layout.repeated(base,scale) == Some{view} : Maybe<&2,Views.View>}) -> {Views.axis_extents_of(view) == Nearest.repeated_extents(shape,scale) : List<&2,U32>}:
  match base:
    case Some{Views.View{+axes,+offset}}:
      +axes_shape = {
        %based : {Metadata.observed_extents(_,shape) == shape : List<&2,U32>}
        Metadata.contiguous_shape(shape,0) : {Views.axis_extents(axes) == shape : List<&2,U32>}}
      %axes_shape : {Views.axis_extents_of(view) == Nearest.repeated_extents(_,scale) : List<&2,U32>}
      offset_extents(Layout.repeated_axes(axes,scale),view,offset,Nearest.repeated_extents(Views.axis_extents(axes),scale),repeated_axes_extents(axes,scale),actual)
    case None{}:
      Empty.absurd({Views.axis_extents_of(view) == Nearest.repeated_extents(shape,scale) : List<&2,U32>},WordEncoding.false_true(
        Equal.cong(Maybe<&2,Views.View>,Bool,has_view,None{},Some{view},actual)))

def tail_positive(+head: Nat,+tail: Nat,positive: {Nat.is_lt(0n,Nat.mul(head,tail)) == True{} : Bool}) -> {Nat.is_lt(0n,tail) == True{} : Bool}:
  RowMajor.left_factor_positive(tail,head,%RowMajor.multiply_commutative(head,tail) : {Nat.is_lt(0n,_) == True{} : Bool} positive)

def tail_room(+head: Nat,+tail: Nat,+limit: Nat,positive: {Nat.is_lt(0n,Nat.mul(head,tail)) == True{} : Bool},room: {Nat.is_le(Nat.mul(head,tail),limit) == True{} : Bool}) ->
  {Nat.is_le(tail,limit) == True{} : Bool}:
  Order.transitive(tail,Nat.mul(head,tail),limit,
    %RowMajor.multiply_commutative(tail,head) : {Nat.is_le(tail,_) == True{} : Bool}
    RowMajor.product_covers(tail,head,RowMajor.left_factor_positive(head,tail,positive)),room)

def merged_positive(+extents: List<&2,U32>,+scale: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(Nearest.repeated_extents(extents,scale))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Nearest.repeated_extents(extents,scale)),U32.to_nat(limit)) == True{} : Bool}) ->
  {ShapeLaws.elements(Nearest.repeated_extents(extents,scale)) == ShapeLaws.elements(Layout.scaled_extents(extents,scale)) : Nat}:
  match extents:
    case [+height,+width]:
      merged_pair_elements(height,width,scale,
        pair_exact(height,scale,pair_word([height,scale,width,scale],0n,height,scale,limit,maximum,positive,room,{==})),
        pair_exact(width,scale,pair_word([height,scale,width,scale],2n,width,scale,limit,maximum,positive,room,{==})))
    case Con{+head,Nil{}}: {==}
    case Con{+head,Con{+second,Con{+third,+rest}}}:
      +tail = {second <> third <> rest : List<&2,U32>}
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(head),value),ShapeLaws.elements(Nearest.repeated_extents(tail,scale)),ShapeLaws.elements(Layout.scaled_extents(tail,scale)),
        merged_positive(tail,scale,limit,maximum,
          tail_positive(U32.to_nat(head),ShapeLaws.elements(Nearest.repeated_extents(tail,scale)),positive),
          tail_room(U32.to_nat(head),ShapeLaws.elements(Nearest.repeated_extents(tail,scale)),U32.to_nat(limit),positive,room)))
    case []: {==}

def merged_counted(+extents: List<&2,U32>,+scale: U32,+count: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},empty: Bool,
  +classified: {Shapes.has_zero(Nearest.repeated_extents(extents,scale)) == empty : Bool},evidence: Count.Count<ShapeLaws.elements(Nearest.repeated_extents(extents,scale)),count>) ->
  {ShapeLaws.elements(Nearest.repeated_extents(extents,scale)) == ShapeLaws.elements(Layout.scaled_extents(extents,scale)) : Nat}:
  match empty:
    case True{}:
      Equal.trans(Nat,ShapeLaws.elements(Nearest.repeated_extents(extents,scale)),0n,ShapeLaws.elements(Layout.scaled_extents(extents,scale)),ShapeLaws.empty_product(Nearest.repeated_extents(extents,scale),classified),
        Equal.sym(Nat,ShapeLaws.elements(Layout.scaled_extents(extents,scale)),0n,ShapeLaws.empty_product(Layout.scaled_extents(extents,scale),scaled_empty(extents,scale,classified))))
    case False{}:
      Count.Count{+exact,bounded} = evidence
      merged_positive(extents,scale,limit,maximum,CountCompletion.positive_elements(Nearest.repeated_extents(extents,scale),classified),
        %Equal.sym(Nat,ShapeLaws.elements(Nearest.repeated_extents(extents,scale)),U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
        Bounds.as_nat_le(count,limit,%Equal.sym(U32,limit,Machine.word_max(),maximum) : {U32.is_le(count,_) == True{} : Bool} FullRange.maximum(count)))

def dense_elements(+shape: List<&2,U32>,+scale: U32,+view: Views.View,+extents: {Views.axis_extents_of(view) == Nearest.repeated_extents(shape,scale) : List<&2,U32>},+count: U32,
  counted: {Tensors.count_of(Views.axis_extents_of(view)) == Some{count} : Maybe<&2,U32>}) ->
  {ShapeLaws.elements(Views.axis_extents_of(view)) == ShapeLaws.elements(Layout.scaled_extents(shape,scale)) : Nat}:
  %Equal.sym(List<&2,U32>,Views.axis_extents_of(view),Nearest.repeated_extents(shape,scale),extents) : {ShapeLaws.elements(_) == ShapeLaws.elements(Layout.scaled_extents(shape,scale)) : Nat}
  merged_counted(shape,scale,count,Machine.word_max(),{==},Shapes.has_zero(Nearest.repeated_extents(shape,scale)),{==},
    Count.count_of(Nearest.repeated_extents(shape,scale),count,%extents : {Tensors.count_of(_) == Some{count} : Maybe<&2,U32>} counted))

def dense_selected(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,source: Maybe<&2,Views.View>,+logical: Logical,+scale: U32,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  actual: {Layout.repeated(Tensors.view_of(shape),scale) == source : Maybe<&2,Views.View>},
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers}),
  accepted: {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.upsample_view(~Element,source,Tensors.Usable{},storage,shape,scale,fill,workers))) == True{} : Bool}) ->
  GatherValues.Viewed(~Element,~Logical,~original,logical,source,Layout.result_of(~Element,Layout.upsample_view(~Element,source,Tensors.Usable{},storage,shape,scale,fill,workers))):
  match source:
    case Some{+view}:
      +extents = repeated_view_extents(Tensors.view_of(shape),shape,scale,view,{==},actual)
      GatherValues.gather(~Element,~Logical,~original,logical,"upsample_nearest",view,Views.axis_extents_of(view),storage,shape,Layout.scaled_extents(shape,scale),fill,workers,contract,
        count => counted => dense_elements(shape,scale,view,extents,count,counted),accepted)
    case None{}: WordEncoding.false_true(accepted)

# A Ready dense tensor with complete values, upsampled by the public call:
# when the call returns Ready, the result is well formed and holds, at every
# index, the logical input at the address of that index in the repeated view
# (the two scale axes have stride zero).
def dense(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+scale: U32,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers}),
  accepted: {Dense.tensor_ready(~Element,Layout.upsample_nearest(~Element,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},scale)) == True{} : Bool}) ->
  GatherValues.Viewed(~Element,~Logical,~original,logical,Layout.repeated(Tensors.view_of(shape),scale),Layout.upsample_nearest(~Element,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},scale)):
  dense_selected(~Element,~Logical,~original,Layout.repeated(Tensors.view_of(shape),scale),logical,scale,storage,shape,fill,workers,{==},contract,accepted)
