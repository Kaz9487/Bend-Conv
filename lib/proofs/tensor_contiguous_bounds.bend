# The successful contiguous guard describes one interval of exactly numel cells.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_view_bounds.bend as ViewBounds
import ./checked_word_refinement.bend as Checked
import ./boolean_logic.bend as Boolean
import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./storage_shape.bend as Shape

def nonzero_positive(+value: Nat,nonzero: {Nat.is_eq(value,0n) == False{} : Bool}) -> {Nat.is_lt(0n,value) == True{} : Bool}:
  match value:
    case 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},nonzero)))
    case 1n+rest: {==}

def extent_positive(+extent: U32,nonzero: {U32.is_zero(extent) == False{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}:
  nonzero_positive(U32.to_nat(extent),{
    %Comparison.eq(extent,0) : {_ == False{} : Bool}
    nonzero : {Nat.is_eq(U32.to_nat(extent),0n) == False{} : Bool}})

def stride_guard(+extent: U32,+stride: U32,+count: U32,
  valid: {U32.is_le(extent,1) || U32.is_eq(stride,count) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(extent),1n) || Nat.is_eq(U32.to_nat(stride),U32.to_nat(count)) == True{} : Bool}:
  %Comparison.le(extent,1) : {_ || Nat.is_eq(U32.to_nat(stride),U32.to_nat(count)) == True{} : Bool}
  %Comparison.eq(stride,count) : {U32.is_le(extent,1) || _ == True{} : Bool}
  valid

def head_span(+extent: Nat,+stride: Nat,+count: Nat,+span: Nat,
  positive: {Nat.is_lt(0n,extent) == True{} : Bool},
  valid: {Nat.is_le(extent,1n) || Nat.is_eq(stride,count) == True{} : Bool},
  previous: {1n+span == count : Nat}) ->
  {1n+Nat.add(Nat.mul(Nat.sub(extent,1n),stride),span) == Nat.mul(extent,count) : Nat}:
  match extent:
    case 0n: Empty.absurd({1n+span == 0n : Nat},WordEncoding.false_true(positive))
    case 1n:
      %Equal.sym(Nat,Nat.add(count,0n),count,Addition.zero_right(count)) : {1n+span == _ : Nat}
      previous
    case 2n+ +rest:
      %Equal.sym(Nat,stride,count,Order.equal_of_is_eq(stride,count,valid)) :
        {1n+Nat.add(Nat.mul(1n+rest,_),span) == Nat.mul(2n+rest,count) : Nat}
      %previous : {1n+Nat.add(Nat.mul(1n+rest,count),span) == Nat.add(_,Nat.mul(1n+rest,count)) : Nat}
      %Addition.successor_right(Nat.mul(1n+rest,count),span) :
        {_ == Nat.add(1n+span,Nat.mul(1n+rest,count)) : Nat}
      Order.commutative(Nat.mul(1n+rest,count),1n+span)

law SpanCount:
  for result: Maybe<&2,U32>
  for product: Nat
  for span: Nat
  Data

def SpanCount(result,product,span):
  match result:
    case None{}: Unit
    case Some{count}: Shape.Both<{product == U32.to_nat(count) : Nat},{1n+span == U32.to_nat(count) : Nat}>

def multiplied(+result: Maybe<&2,U32>,+extent: U32,+count: U32,+stride: U32,+tail_product: Nat,+tail_span: Nat,
  computed: {Shapes.checked_multiply(extent,count) == result : Maybe<&2,U32>},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  valid: {U32.is_le(extent,1) || U32.is_eq(stride,count) == True{} : Bool},
  product_equal: {tail_product == U32.to_nat(count) : Nat},span_equal: {1n+tail_span == U32.to_nat(count) : Nat}) ->
  SpanCount(result,Nat.mul(U32.to_nat(extent),tail_product),Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),tail_span)):
  match result:
    case None{}: Unit{}
    case Some{+output}:
      +product = Checked.multiply_sound(extent,count,output,computed)
      Shape.Both{Equal.trans(Nat,Nat.mul(U32.to_nat(extent),tail_product),Nat.mul(U32.to_nat(extent),U32.to_nat(count)),U32.to_nat(output),
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(extent),value),tail_product,U32.to_nat(count),product_equal),product),{
        %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(U32.to_nat(extent),1n),
          FullRange.subtract(extent,1,extent,Order.strict_follow(0n,U32.to_nat(extent),positive),Order.sub_le(U32.to_nat(extent),1n))) :
          {1n+Nat.add(Nat.mul(_,U32.to_nat(stride)),tail_span) == U32.to_nat(output) : Nat}
        Equal.trans(Nat,1n+Nat.add(Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride)),tail_span),Nat.mul(U32.to_nat(extent),U32.to_nat(count)),U32.to_nat(output),
          head_span(U32.to_nat(extent),U32.to_nat(stride),U32.to_nat(count),tail_span,positive,stride_guard(extent,stride,count,valid),span_equal),product) :
          {1n+Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),tail_span) == U32.to_nat(output) : Nat}}}

def selected(+valid: Bool,+extent: U32,+count: U32,+stride: U32,+tail_product: Nat,+tail_span: Nat,
  positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  guard: {U32.is_le(extent,1) || U32.is_eq(stride,count) == valid : Bool},
  product_equal: {tail_product == U32.to_nat(count) : Nat},span_equal: {1n+tail_span == U32.to_nat(count) : Nat}) ->
  SpanCount(Views.contiguous_product(valid,extent,count),Nat.mul(U32.to_nat(extent),tail_product),Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),tail_span)):
  match valid:
    case False{}: Unit{}
    case True{}: multiplied(Shapes.checked_multiply(extent,count),extent,count,stride,tail_product,tail_span,{==},positive,guard,product_equal,span_equal)

def axis(+inner: Maybe<&2,U32>,+extent: U32,+stride: U32,+tail_product: Nat,+tail_span: Nat,
  positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},previous: SpanCount(inner,tail_product,tail_span)) ->
  SpanCount(Views.contiguous_axis(extent,stride,inner),Nat.mul(U32.to_nat(extent),tail_product),Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),tail_span)):
  match inner:
    case None{}: Unit{}
    case Some{+count}:
      Shape.Both{product_equal,span_equal} = previous
      selected(U32.is_le(extent,1) || U32.is_eq(stride,count),extent,count,stride,tail_product,tail_span,positive,{==},product_equal,span_equal)

def contiguous(+axes: List<&2,Views.Axis>,nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}) ->
  SpanCount(Views.contiguous_size(axes),ShapeLaws.elements(Views.axis_extents(axes)),ViewBounds.last_span(axes)):
  match axes:
    case []: Shape.Both{{==},{==}}
    case Views.Axis{+extent,+stride} <> +tail:
      +nonempty = nonempty
      axis(Views.contiguous_size(tail),extent,stride,ShapeLaws.elements(Views.axis_extents(tail)),ViewBounds.last_span(tail),
        extent_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty)),
        contiguous(tail,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty)))

def accepted_span(+result: Maybe<&2,U32>,+product: Nat,+span: Nat,
  accepted: {Views.has_count(result) == True{} : Bool},evidence: SpanCount(result,product,span)) -> {product == 1n+span : Nat}:
  match result:
    case None{}: Empty.absurd({product == 1n+span : Nat},WordEncoding.false_true(accepted))
    case Some{+count}:
      Shape.Both{product_equal,span_equal} = evidence
      Equal.trans(Nat,product,U32.to_nat(count),1n+span,product_equal,Equal.sym(Nat,1n+span,U32.to_nat(count),span_equal))

def contiguous_span(+axes: List<&2,Views.Axis>,+offset: U32,
  +nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  accepted: {Views.is_contiguous(Views.View{axes,offset}) == True{} : Bool}) ->
  {ShapeLaws.elements(Views.axis_extents(axes)) == 1n+ViewBounds.last_span(axes) : Nat}:
  accepted_span(Views.contiguous_size(axes),ShapeLaws.elements(Views.axis_extents(axes)),ViewBounds.last_span(axes),{
    %nonempty : {Views.contiguous_nonempty(_,axes) == True{} : Bool}
    accepted : {Views.has_count(Views.contiguous_size(axes)) == True{} : Bool}},contiguous(axes,nonempty))

def interval(+axes: List<&2,Views.Axis>,+offset: U32,+count: U32,+length: U32,
  +nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  contiguous: {Views.is_contiguous(Views.View{axes,offset}) == True{} : Bool},
  counted: {Views.numel(Views.View{axes,offset}) == Some{count} : Maybe<&2,U32>},
  fits: {Views.fits(Views.View{axes,offset},length) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(count)),U32.to_nat(length)) == True{} : Bool}:
  +count_equal = {
    %counted : ShapeLaws.Count(_,ShapeLaws.elements(Views.axis_extents(axes)))
    ShapeLaws.nonempty_numel(Views.axis_extents(axes),nonempty) : {ShapeLaws.elements(Views.axis_extents(axes)) == U32.to_nat(count) : Nat}}
  %count_equal : {Nat.is_le(Nat.add(U32.to_nat(offset),_),U32.to_nat(length)) == True{} : Bool}
  %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(axes)),1n+ViewBounds.last_span(axes),contiguous_span(axes,offset,nonempty,contiguous)) :
    {Nat.is_le(Nat.add(U32.to_nat(offset),_),U32.to_nat(length)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(U32.to_nat(offset),1n+ViewBounds.last_span(axes)),1n+Nat.add(U32.to_nat(offset),ViewBounds.last_span(axes)),
    Addition.successor_right(U32.to_nat(offset),ViewBounds.last_span(axes))) : {Nat.is_le(_,U32.to_nat(length)) == True{} : Bool}
  Order.strict_follow(Nat.add(U32.to_nat(offset),ViewBounds.last_span(axes)),U32.to_nat(length),{
    %ViewBounds.last_location(axes,U32.to_nat(offset)) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool}
    ViewBounds.last_address(axes,offset,length,{
      %nonempty : {Views.fits_nonempty(_,Views.View{axes,offset},length) == True{} : Bool}
      fits : {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool}}) :
      {Nat.is_lt(Nat.add(U32.to_nat(offset),ViewBounds.last_span(axes)),U32.to_nat(length)) == True{} : Bool}})

def selected_interval(+empty: Bool,+axes: List<&2,Views.Axis>,+offset: U32,+count: U32,+length: U32,
  +classified: {Shapes.has_zero(Views.axis_extents(axes)) == empty : Bool},
  contiguous: {Views.is_contiguous(Views.View{axes,offset}) == True{} : Bool},
  counted: {Views.numel(Views.View{axes,offset}) == Some{count} : Maybe<&2,U32>},
  fits: {Views.fits(Views.View{axes,offset},length) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(count)),U32.to_nat(length)) == True{} : Bool}:
  match empty:
    case False{}: interval(axes,offset,count,length,classified,contiguous,counted,fits)
    case True{}:
      %Equal.sym(Nat,U32.to_nat(count),0n,ShapeLaws.count_zero(count,{
        %classified : {Shapes.count_nonempty(_,Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>}
        counted : {Some{0} == Some{count} : Maybe<&2,U32>}})) : {Nat.is_le(Nat.add(U32.to_nat(offset),_),U32.to_nat(length)) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(U32.to_nat(offset),0n),U32.to_nat(offset),Addition.zero_right(U32.to_nat(offset))) : {Nat.is_le(_,U32.to_nat(length)) == True{} : Bool}
      Bounds.as_nat_le(offset,length,{
        %classified : {Views.fits_nonempty(_,Views.View{axes,offset},length) == True{} : Bool}
        fits : {U32.is_le(offset,length) == True{} : Bool}})

def offset(view: Views.View) -> U32:
  Views.View{axes,offset} = view
  offset

def checked_interval(+view: Views.View,+count: U32,+length: U32,
  contiguous: {Views.is_contiguous(view) == True{} : Bool},counted: {Views.numel(view) == Some{count} : Maybe<&2,U32>},
  fits: {Views.fits(view,length) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(offset(view)),U32.to_nat(count)),U32.to_nat(length)) == True{} : Bool}:
  Views.View{+axes,offset} = view
  selected_interval(Shapes.has_zero(Views.axis_extents(axes)),axes,offset,count,length,{==},contiguous,counted,fits)
