# Upsample samples: copy guards, scaled geometry, placements and the finite
# samples with their destination stores.
import Base
import ./tensor_band_upsample_execution.bend as BandUpsampleExecution
import ../machine_limits.bend as Machine
import ./nat_order.bend as Order
import ./tensor_count_completion.bend as CountCompletion
import ../tensor_band.bend as Bands
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./machine_limit_bounds.bend as Limits
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_band_count.bend as BandCount
import ./tensor_band_shape_count.bend as BandShapeCount
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_shape_refinement.bend as ShapeLaws
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../default_machine_limits.bend as Default
import ./boolean_logic.bend as Boolean
import ./checked_word_refinement.bend as Checked
import ./nat_division.bend as Division
import ./natural_addition.bend as Addition
import ./storage_tree_model.bend as Tree
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_model.bend as Model
import ./tensor_count_refinement.bend as Count
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding
import ../tensor_view.bend as Views
import ./checked_word_completion.bend as Completion
import ./contiguous_copy.bend as Contiguous
import ./nat_division_order.bend as DivisionOrder
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_certificate.bend as Certificate
import ./tensor_band_combine_leaf.bend as BandCombineLeaf
import ./tensor_band_values.bend as BandValues
import ./tensor_contiguous_bounds.bend as ContiguousBounds
import ./tensor_cursor_bounds.bend as CursorBounds
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_order.bend as CursorOrder
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_visit_refinement.bend as TensorVisit
import ./tensor_view_bounds.bend as ViewBounds
import ./traversal_indexed_writes.bend as TraversalIndexedWrites
import ./traversal_map_readback.bend as MapReadback
import ./traversal_visit_writes.bend as TraversalVisitWrites
import ./u32_division.bend as U32Division
import ./u32_full_range.bend as FullRange
import ./word_multiplication.bend as Multiplication
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ../storage_allocation.bend as Allocation
import ../storage_buffer.bend as Storage
import ../tensor_band_schedule.bend as BandSchedule
import ../tensor_layout.bend as Layout
import ../tensor_operations.bend as Operations
import ../traversal_loop.bend as Loop
import ./tensor_copy_values.bend as CopyValues

# Successful checked-copy guards for the actual upsample descriptors.
# Empty axes take Views.fits_nonempty/contiguous_nonempty's empty branch.
# Positive axes use the checked final-coordinate address, never a supplied
# successful copy Bool. Sample matching and store coverage are separate work.
def axes(view: Views.View) -> List<&2,Views.Axis>:
  Views.View{axes,offset} = view
  axes

def span(view: Views.View) -> Nat:
  Views.View{axes,offset} = view
  ViewBounds.last_span(axes)

def native_le(+left: U32,+right: U32,truth: {Nat.is_le(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool}) ->
  {U32.is_le(left,right) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(left,right),Nat.is_le(U32.to_nat(left),U32.to_nat(right)),Comparison.le(left,right)) : {_ == True{} : Bool} truth

def native_lt(+left: U32,+right: U32,truth: {Nat.is_lt(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool}) ->
  {U32.is_lt(left,right) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt(left,right),Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),Comparison.lt(left,right)) : {_ == True{} : Bool} truth

def positive_nonzero(+value: Nat,positive: {Nat.is_lt(0n,value) == True{} : Bool}) -> {Nat.is_eq(value,0n) == False{} : Bool}:
  match value:
    case 0n: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(positive))
    case 1n+rest: {==}

def count_nonzero(+extents: List<&2,U32>,+count: U32,nonempty: {Shapes.has_zero(extents) == False{} : Bool},
  exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) -> {U32.is_zero(count) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_zero(count),Nat.is_eq(U32.to_nat(count),0n),Comparison.eq(count,0)) : {_ == False{} : Bool}
  positive_nonzero(U32.to_nat(count),%exact :
    {Nat.is_lt(0n,_) == True{} : Bool} CountCompletion.positive_elements(extents,nonempty))

# Inverse of the remaining-room comparison, on unbounded natural sums.
def subtract_room(+offset: Nat,+increment: Nat,+last: Nat,room: {Nat.is_le(Nat.add(offset,increment),last) == True{} : Bool}) ->
  {Nat.is_le(increment,Nat.sub(last,offset)) == True{} : Bool}:
  match offset last:
    case 0n +last:
      %Equal.sym(Nat,Nat.sub(last,0n),last,Order.subtract_zero(last)) : {Nat.is_le(increment,_) == True{} : Bool} room
    case 1n+offset 0n: Empty.absurd({Nat.is_le(increment,0n) == True{} : Bool},WordEncoding.false_true(room))
    case 1n+offset 1n+last: subtract_room(offset,increment,last,room)

def advance_complete_bounded(+offset: U32,+coordinate: U32,+stride: U32,+length: U32,+word_limit: U32,+maximum: {word_limit == Machine.word_max() : U32},
  +room: {Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))),U32.to_nat(length)) == True{} : Bool}) ->
  {Views.advance(offset,coordinate,stride,length) == Some{(offset + coordinate * stride : U32)} : Maybe<&2,U32>}:
  +increment = (coordinate * stride : U32)
  +natural_increment = Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))
  +sum = Nat.add(U32.to_nat(offset),natural_increment)
  +inside = Order.le_lt_trans(U32.to_nat(offset),sum,U32.to_nat(length),Order.le_add(U32.to_nat(offset),natural_increment),room)
  +length_room = {
    %Equal.sym(U32,word_limit,Machine.word_max(),maximum) : {U32.is_le(length,_) == True{} : Bool}
    FullRange.maximum(length) : {U32.is_le(length,word_limit) == True{} : Bool}}
  +product_room = Order.transitive(natural_increment,sum,U32.to_nat(word_limit),
    %Order.commutative(natural_increment,U32.to_nat(offset)) : {Nat.is_le(natural_increment,_) == True{} : Bool}
    Order.le_add(natural_increment,U32.to_nat(offset)),
    Order.transitive(sum,U32.to_nat(length),U32.to_nat(word_limit),Bounds.strict_le(sum,U32.to_nat(length),room),Bounds.as_nat_le(length,word_limit,length_room)))
  +product_exact = FullRange.multiply(coordinate,stride,word_limit,product_room)
  +remaining = {
    %Equal.sym(Nat,U32.to_nat(increment),natural_increment,product_exact) : {Nat.is_le(_,U32.to_nat((length - 1 - offset : U32))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat((length - 1 - offset : U32)),Nat.sub(Nat.sub(U32.to_nat(length),1n),U32.to_nat(offset)),ViewBounds.room_exact(offset,length,inside)) :
      {Nat.is_le(natural_increment,_) == True{} : Bool}
    subtract_room(U32.to_nat(offset),natural_increment,Nat.sub(U32.to_nat(length),1n),ViewBounds.before_last(sum,U32.to_nat(length),room)) :
      {Nat.is_le(U32.to_nat(increment),U32.to_nat((length - 1 - offset : U32))) == True{} : Bool}}
  %Equal.sym(Bool,U32.is_lt(offset,length),True{},native_lt(offset,length,inside)) :
    {Views.advance_inside(_,offset,coordinate,stride,length) == Some{(offset + increment : U32)} : Maybe<&2,U32>}
  %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(coordinate,stride),Some{increment},CountCompletion.checked_product(coordinate,stride,word_limit,maximum,product_room)) :
    {Views.advance_product(_,offset,(length - 1 - offset : U32)) == Some{(offset + increment : U32)} : Maybe<&2,U32>}
  %Equal.sym(Bool,U32.is_le(increment,(length - 1 - offset : U32)),True{},native_le(increment,(length - 1 - offset : U32),remaining)) :
    {Views.address_when(_,offset,increment) == Some{(offset + increment : U32)} : Maybe<&2,U32>}
  {==}

def advance_complete(+offset: U32,+coordinate: U32,+stride: U32,+length: U32,
  room: {Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))),U32.to_nat(length)) == True{} : Bool}) ->
  {Views.advance(offset,coordinate,stride,length) == Some{(offset + coordinate * stride : U32)} : Maybe<&2,U32>}:
  advance_complete_bounded(offset,coordinate,stride,length,Machine.word_max(),{==},room)

def locate_complete(+axes: List<&2,Views.Axis>,+offset: U32,+length: U32,
  +nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +room: {Nat.is_lt(Nat.add(U32.to_nat(offset),ViewBounds.last_span(axes)),U32.to_nat(length)) == True{} : Bool}) ->
  {Views.found(Views.locate(axes,Views.last_coordinates(axes),Some{offset},length)) == True{} : Bool}:
  match axes:
    case []:
      %Equal.sym(Bool,U32.is_lt(offset,length),True{},native_lt(offset,length,
        %Addition.zero_right(U32.to_nat(offset)) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool} room)) :
        {Views.found(Views.finish_address(_,offset)) == True{} : Bool}
      {==}
    case Views.Axis{+extent,+stride} <> +tail:
      +head_nonzero = Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty)
      +tail_nonempty = Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty)
      +coordinate = (extent - 1 : U32)
      +increment = (coordinate * stride : U32)
      +next = (offset + increment : U32)
      +sum = Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride)))
      +step_room = Order.le_lt_trans(sum,Nat.add(sum,ViewBounds.last_span(tail)),U32.to_nat(length),Order.le_add(sum,ViewBounds.last_span(tail)),
        %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))),ViewBounds.last_span(tail)),
          Nat.add(U32.to_nat(offset),Nat.add(Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride)),ViewBounds.last_span(tail))),
          Addition.associative(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride)),ViewBounds.last_span(tail))) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool} room)
      +product_room = Order.transitive(Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride)),sum,U32.to_nat(length),
        %Order.commutative(Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride)),U32.to_nat(offset)) : {Nat.is_le(Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride)),_) == True{} : Bool}
        Order.le_add(Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride)),U32.to_nat(offset)),Bounds.strict_le(sum,U32.to_nat(length),step_room))
      +product_exact = FullRange.multiply(coordinate,stride,length,product_room)
      +sum_exact = {
        %product_exact : {U32.to_nat(next) == Nat.add(U32.to_nat(offset),_) : Nat}
        FullRange.add(offset,increment,length,
          %Equal.sym(Nat,U32.to_nat(increment),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride)),product_exact) : {Nat.is_le(Nat.add(U32.to_nat(offset),_),U32.to_nat(length)) == True{} : Bool}
          Bounds.strict_le(sum,U32.to_nat(length),step_room)) : {U32.to_nat(next) == sum : Nat}}
      %Equal.sym(Bool,U32.is_lt(coordinate,extent),True{},BandCombineLeaf.predecessor_inside(extent,head_nonzero)) :
        {Views.found(Views.locate(tail,Views.last_coordinates(tail),Views.advance_inside(_ && U32.is_lt(offset,length),offset,coordinate,stride,length),length)) == True{} : Bool}
      %Equal.sym(Maybe<&2,U32>,Views.advance(offset,coordinate,stride,length),Some{next},advance_complete(offset,coordinate,stride,length,step_room)) :
        {Views.found(Views.locate(tail,Views.last_coordinates(tail),_,length)) == True{} : Bool}
      locate_complete(tail,next,length,tail_nonempty,
        %Equal.sym(Nat,U32.to_nat(next),sum,sum_exact) : {Nat.is_lt(Nat.add(_,ViewBounds.last_span(tail)),U32.to_nat(length)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride))),ViewBounds.last_span(tail)),
          Nat.add(U32.to_nat(offset),Nat.add(Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride)),ViewBounds.last_span(tail))),
          Addition.associative(U32.to_nat(offset),Nat.mul(U32.to_nat(coordinate),U32.to_nat(stride)),ViewBounds.last_span(tail))) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool} room)

def fits_complete(+view: Views.View,+length: U32,+nonempty: {Shapes.has_zero(Views.axis_extents_of(view)) == False{} : Bool},
  room: {Nat.is_lt(Nat.add(U32.to_nat(ContiguousBounds.offset(view)),span(view)),U32.to_nat(length)) == True{} : Bool}) -> {Views.fits(view,length) == True{} : Bool}:
  Views.View{+axes,+offset} = view
  %Equal.sym(Bool,Shapes.has_zero(Views.axis_extents(axes)),False{},nonempty) : {Views.fits_nonempty(_,view,length) == True{} : Bool}
  locate_complete(axes,offset,length,nonempty,room)

# A successful build from an already contiguous suffix remains contiguous.
def ContiguousResult(result: Maybe<&2,Views.View>) -> Data:
  match result:
    case None{}: Unit
    case Some{view}: {Views.is_contiguous(view) == True{} : Bool}

def failed_build(reversed: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32) -> ContiguousResult(Views.build_axes(reversed,None{},axes,offset)):
  match reversed:
    case Nil{}: Unit{}
    case Con{extent,tail}: Unit{}

def contiguous_nonempty(+empty: Bool,+axes: List<&2,Views.Axis>,accepted: {Views.has_count(Views.contiguous_size(axes)) == True{} : Bool}) ->
  {Views.contiguous_nonempty(empty,axes) == True{} : Bool}:
  match empty:
    case True{}: {==}
    case False{}: accepted

def same_stride(+extent: U32,+stride: U32) -> {U32.is_le(extent,1) || U32.is_eq(stride,stride) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(stride,stride),True{},
    %Equal.sym(Bool,U32.is_eq(stride,stride),Nat.is_eq(U32.to_nat(stride),U32.to_nat(stride)),Comparison.eq(stride,stride)) : {_ == True{} : Bool} Order.equal_reflexive(U32.to_nat(stride))) :
    {U32.is_le(extent,1) || _ == True{} : Bool}
  Boolean.or_true_right(U32.is_le(extent,1))

def prepended_size(+extent: U32,+stride: U32,+axes: List<&2,Views.Axis>,inner: {Views.contiguous_size(axes) == Some{stride} : Maybe<&2,U32>}) ->
  {Views.contiguous_size(Views.Axis{extent,stride} <> axes) == Shapes.checked_multiply(extent,stride) : Maybe<&2,U32>}:
  %Equal.sym(Maybe<&2,U32>,Views.contiguous_size(axes),Some{stride},inner) : {Views.contiguous_axis(extent,stride,_) == Shapes.checked_multiply(extent,stride) : Maybe<&2,U32>}
  %Equal.sym(Bool,U32.is_le(extent,1) || U32.is_eq(stride,stride),True{},same_stride(extent,stride)) :
    {Views.contiguous_product(_,extent,stride) == Shapes.checked_multiply(extent,stride) : Maybe<&2,U32>}
  {==}

def built_step_bounded(+reversed: List<&2,U32>,+result: Maybe<&2,U32>,+extent: U32,+stride: U32,+axes: List<&2,Views.Axis>,+offset: U32,+word_limit: U32,+maximum: {word_limit == Machine.word_max() : U32},
  inner: {Views.contiguous_size(axes) == Some{stride} : Maybe<&2,U32>},computed: {Shapes.checked_multiply(stride,extent) == result : Maybe<&2,U32>},
  next: @value: U32 -> {Views.contiguous_size(Views.Axis{extent,stride} <> axes) == Some{value} : Maybe<&2,U32>} ->
    ContiguousResult(Views.build_axes(reversed,Some{value},Views.Axis{extent,stride} <> axes,offset))) ->
  ContiguousResult(Views.build_axes(reversed,result,Views.Axis{extent,stride} <> axes,offset)):
  match result:
    case None{}: failed_build(reversed,Views.Axis{extent,stride} <> axes,offset)
    case Some{+value}:
      next(value)(
        Equal.trans(Maybe<&2,U32>,Views.contiguous_size(Views.Axis{extent,stride} <> axes),Shapes.checked_multiply(extent,stride),Some{value},
          prepended_size(extent,stride,axes,inner),
          Completion.actual(extent,stride,word_limit,value,
            Equal.trans(Nat,Nat.mul(U32.to_nat(extent),U32.to_nat(stride)),Nat.mul(U32.to_nat(stride),U32.to_nat(extent)),U32.to_nat(value),
              RowMajor.multiply_commutative(U32.to_nat(extent),U32.to_nat(stride)),Checked.multiply_sound(stride,extent,value,computed)),
            %Equal.sym(U32,word_limit,Machine.word_max(),maximum) : {U32.is_le(value,_) == True{} : Bool} FullRange.maximum(value),maximum)))

def built_step(+reversed: List<&2,U32>,+result: Maybe<&2,U32>,+extent: U32,+stride: U32,+axes: List<&2,Views.Axis>,+offset: U32,
  inner: {Views.contiguous_size(axes) == Some{stride} : Maybe<&2,U32>},computed: {Shapes.checked_multiply(stride,extent) == result : Maybe<&2,U32>},
  next: @value: U32 -> {Views.contiguous_size(Views.Axis{extent,stride} <> axes) == Some{value} : Maybe<&2,U32>} ->
    ContiguousResult(Views.build_axes(reversed,Some{value},Views.Axis{extent,stride} <> axes,offset))) ->
  ContiguousResult(Views.build_axes(reversed,result,Views.Axis{extent,stride} <> axes,offset)):
  built_step_bounded(reversed,result,extent,stride,axes,offset,Machine.word_max(),{==},inner,computed,next)

def built_contiguous(+reversed: List<&2,U32>,+stride: U32,+axes: List<&2,Views.Axis>,+offset: U32,
  inner: {Views.contiguous_size(axes) == Some{stride} : Maybe<&2,U32>}) -> ContiguousResult(Views.build_axes(reversed,Some{stride},axes,offset)):
  match reversed:
    case []:
      contiguous_nonempty(Shapes.has_zero(Views.axis_extents(axes)),axes,
        Equal.cong(Maybe<&2,U32>,Bool,Views.has_count,Views.contiguous_size(axes),Some{stride},inner))
    case +extent <> +tail:
      built_step(tail,Shapes.checked_multiply(stride,extent),extent,stride,axes,offset,inner,{==},
        value => inner => built_contiguous(tail,value,Views.Axis{extent,stride} <> axes,offset,inner))

def canonical_contiguous_when(+extents: List<&2,U32>,+count: U32,empty: Bool,+classified: {Shapes.has_zero(extents) == empty : Bool},
  +exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) -> {Views.is_contiguous(BandUpsampleExecution.canonical_view(extents,count)) == True{} : Bool}:
  match empty:
    case True{}:
      %Equal.sym(List<&2,U32>,Views.axis_extents_of(BandUpsampleExecution.canonical_view(extents,count)),extents,BandUpsampleExecution.canonical_shape(extents,count,exact)) :
        {Views.contiguous_nonempty(Shapes.has_zero(_),axes(BandUpsampleExecution.canonical_view(extents,count))) == True{} : Bool}
      %Equal.sym(Bool,Shapes.has_zero(extents),True{},classified) : {Views.contiguous_nonempty(_,axes(BandUpsampleExecution.canonical_view(extents,count))) == True{} : Bool}
      {==}
    case False{}:
      +nonzero = count_nonzero(extents,count,classified,exact)
      %BandUpsampleExecution.view_when(extents,count,False{},nonzero,exact) : ContiguousResult(_)
      %Equal.sym(Bool,U32.is_zero(count),False{},nonzero) : ContiguousResult(Views.build_axes(List.reverse(&2,U32,extents),Some{Bool.pick(U32,_,0,1)},[],0))
      built_contiguous(List.reverse(&2,U32,extents),1,[],0,{==})

def canonical_contiguous(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Views.is_contiguous(BandUpsampleExecution.canonical_view(extents,count)) == True{} : Bool}:
  canonical_contiguous_when(extents,count,Shapes.has_zero(extents),{==},exact)

def canonical_fits_when(+extents: List<&2,U32>,+count: U32,empty: Bool,+classified: {Shapes.has_zero(extents) == empty : Bool},
  +exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) -> {Views.fits(BandUpsampleExecution.canonical_view(extents,count),count) == True{} : Bool}:
  match empty:
    case True{}:
      %Equal.sym(List<&2,U32>,Views.axis_extents_of(BandUpsampleExecution.canonical_view(extents,count)),extents,BandUpsampleExecution.canonical_shape(extents,count,exact)) :
        {Views.fits_nonempty(Shapes.has_zero(_),BandUpsampleExecution.canonical_view(extents,count),count) == True{} : Bool}
      %Equal.sym(Bool,Shapes.has_zero(extents),True{},classified) : {Views.fits_nonempty(_,BandUpsampleExecution.canonical_view(extents,count),count) == True{} : Bool}
      native_le(0,count,Order.zero_le(U32.to_nat(count)))
    case False{}:
      +view = {BandUpsampleExecution.canonical_view(extents,count) : Views.View}
      +nonempty = {
        %Equal.sym(List<&2,U32>,Views.axis_extents_of(view),extents,BandUpsampleExecution.canonical_shape(extents,count,exact)) : {Shapes.has_zero(_) == False{} : Bool} classified :
        {Shapes.has_zero(Views.axis_extents_of(view)) == False{} : Bool}}
      +span_exact = ContiguousBounds.contiguous_span(axes(view),0,nonempty,canonical_contiguous(extents,count,exact))
      fits_complete(view,count,nonempty,
        %Equal.sym(Nat,Nat.add(0n,span(view)),span(view),{==}) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool}
        %exact : {Nat.is_lt(span(view),_) == True{} : Bool}
        %BandUpsampleExecution.canonical_shape(extents,count,exact) : {Nat.is_lt(span(view),ShapeLaws.elements(_)) == True{} : Bool}
        %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents_of(view)),1n+span(view),span_exact) : {Nat.is_lt(span(view),_) == True{} : Bool}
        Order.lt_succ(span(view),span(view),Order.reflexive(span(view))))

def canonical_fits(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Views.fits(BandUpsampleExecution.canonical_view(extents,count),count) == True{} : Bool}:
  canonical_fits_when(extents,count,Shapes.has_zero(extents),{==},exact)

def zero_axis_span(+extent: U32,+tail: List<&2,Views.Axis>) -> {ViewBounds.last_span(Views.Axis{extent,0} <> tail) == ViewBounds.last_span(tail) : Nat}:
  %Equal.sym(Nat,Nat.mul(U32.to_nat((extent - 1 : U32)),0n),0n,RowMajor.multiply_zero_right(U32.to_nat((extent - 1 : U32)))) :
    {Nat.add(_,ViewBounds.last_span(tail)) == ViewBounds.last_span(tail) : Nat}
  {==}

def broadcast_span(+scale: U32,+extent: U32,+stride: U32) ->
  {ViewBounds.last_span([Views.Axis{scale,0},Views.Axis{extent,stride},Views.Axis{scale,0}]) == ViewBounds.last_span([Views.Axis{extent,stride}]) : Nat}:
  Equal.trans(Nat,ViewBounds.last_span([Views.Axis{scale,0},Views.Axis{extent,stride},Views.Axis{scale,0}]),ViewBounds.last_span([Views.Axis{extent,stride},Views.Axis{scale,0}]),ViewBounds.last_span([Views.Axis{extent,stride}]),
    zero_axis_span(scale,[Views.Axis{extent,stride},Views.Axis{scale,0}]),
    Equal.cong(Nat,Nat,tail => Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),tail),ViewBounds.last_span([Views.Axis{scale,0}]),0n,zero_axis_span(scale,[])))

def repeated_span(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+count: U32) ->
  {span(BandUpsampleExecution.repeated_view(geometry,rows,scale,count)) == span(BandUpsampleExecution.canonical_view(BandUpsampleExecution.local_shape(geometry,rows),count)) : Nat}:
  Bands.Geometry{batch,+channels,height,+width} = geometry
  +base = Bool.pick(U32,U32.is_zero(count),0,1)
  Equal.cong(Nat,Nat,tail => Nat.add(Nat.mul(U32.to_nat((channels - 1 : U32)),U32.to_nat((base * width * rows : U32))),
    Nat.add(Nat.mul(U32.to_nat((rows - 1 : U32)),U32.to_nat((base * width : U32))),tail)),
    ViewBounds.last_span([Views.Axis{scale,0},Views.Axis{width,base},Views.Axis{scale,0}]),ViewBounds.last_span([Views.Axis{width,base}]),broadcast_span(scale,width,base))

def source_nonempty(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+nonempty: {Shapes.has_zero(BandUpsampleExecution.repeated_shape(geometry,rows,scale)) == False{} : Bool}) ->
  {Shapes.has_zero(BandUpsampleExecution.local_shape(geometry,rows)) == False{} : Bool}:
  Bands.Geometry{batch,+channels,height,+width} = geometry
  +tail = Shapes.has_zero([rows,scale,width,scale])
  +first_false = Boolean.or_false_left(U32.is_zero(channels),tail,nonempty)
  +tail_false = Boolean.or_false_right(U32.is_zero(channels),tail,nonempty)
  +rows_false = Boolean.or_false_left(U32.is_zero(rows),Shapes.has_zero([scale,width,scale]),tail_false)
  +rest_false = Boolean.or_false_right(U32.is_zero(rows),Shapes.has_zero([scale,width,scale]),tail_false)
  +wide_false = Boolean.or_false_left(U32.is_zero(width),Shapes.has_zero([scale]),Boolean.or_false_right(U32.is_zero(scale),Shapes.has_zero([width,scale]),rest_false))
  %Equal.sym(Bool,U32.is_zero(channels),False{},first_false) : {_ || Shapes.has_zero([rows,width]) == False{} : Bool}
  %Equal.sym(Bool,U32.is_zero(rows),False{},rows_false) : {False{} || (_ || Shapes.has_zero([width])) == False{} : Bool}
  %Equal.sym(Bool,U32.is_zero(width),False{},wide_false) : {False{} || (False{} || (_ || False{})) == False{} : Bool}
  {==}

def repeated_fits_when(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+count: U32,empty: Bool,+classified: {Shapes.has_zero(BandUpsampleExecution.repeated_shape(geometry,rows,scale)) == empty : Bool},
  +exact: {ShapeLaws.elements(BandUpsampleExecution.local_shape(geometry,rows)) == U32.to_nat(count) : Nat}) -> {Views.fits(BandUpsampleExecution.repeated_view(geometry,rows,scale,count),count) == True{} : Bool}:
  Bands.Geometry{batch,channels,height,width} = geometry
  match empty:
    case True{}:
      %Equal.sym(Bool,Shapes.has_zero(BandUpsampleExecution.repeated_shape(geometry,rows,scale)),True{},classified) :
        {Views.fits_nonempty(_,BandUpsampleExecution.repeated_view(geometry,rows,scale,count),count) == True{} : Bool}
      native_le(0,count,Order.zero_le(U32.to_nat(count)))
    case False{}:
      +source = {BandUpsampleExecution.canonical_view(BandUpsampleExecution.local_shape(geometry,rows),count) : Views.View}
      +local_nonempty = source_nonempty(geometry,rows,scale,classified)
      +source_axes_nonempty = {
        %Equal.sym(List<&2,U32>,Views.axis_extents_of(source),BandUpsampleExecution.local_shape(geometry,rows),BandUpsampleExecution.canonical_shape(BandUpsampleExecution.local_shape(geometry,rows),count,exact)) :
          {Shapes.has_zero(_) == False{} : Bool} local_nonempty : {Shapes.has_zero(Views.axis_extents_of(source)) == False{} : Bool}}
      +source_span = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents_of(source)),ShapeLaws.elements(BandUpsampleExecution.local_shape(geometry,rows)),U32.to_nat(count),
        Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,Views.axis_extents_of(source),BandUpsampleExecution.local_shape(geometry,rows),BandUpsampleExecution.canonical_shape(BandUpsampleExecution.local_shape(geometry,rows),count,exact)),exact)
      +count_span = Equal.trans(Nat,U32.to_nat(count),ShapeLaws.elements(Views.axis_extents_of(source)),1n+span(source),
        Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents_of(source)),U32.to_nat(count),source_span),
        ContiguousBounds.contiguous_span(axes(source),0,source_axes_nonempty,canonical_contiguous(BandUpsampleExecution.local_shape(geometry,rows),count,exact)))
      fits_complete(BandUpsampleExecution.repeated_view(geometry,rows,scale,count),count,classified,
        %Equal.sym(Nat,span(BandUpsampleExecution.repeated_view(geometry,rows,scale,count)),span(source),repeated_span(geometry,rows,scale,count)) :
          {Nat.is_lt(Nat.add(0n,_),U32.to_nat(count)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(count),1n+span(source),count_span) : {Nat.is_lt(span(source),_) == True{} : Bool}
        Order.lt_succ(span(source),span(source),Order.reflexive(span(source))))

def repeated_fits(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+count: U32,+exact: {ShapeLaws.elements(BandUpsampleExecution.local_shape(geometry,rows)) == U32.to_nat(count) : Nat}) ->
  {Views.fits(BandUpsampleExecution.repeated_view(geometry,rows,scale,count),count) == True{} : Bool}:
  repeated_fits_when(geometry,rows,scale,count,Shapes.has_zero(BandUpsampleExecution.repeated_shape(geometry,rows,scale)),{==},exact)

def copy_guard(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output_count: U32,
  +descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),output_count>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>) ->
  {CopyValues.copy_guard(~Element,Buffer.pack(~Element,input),BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)),Tensors.buffer(~Element,fill,output_count),
    BandUpsampleExecution.canonical_view(BandUpsampleExecution.repeated_shape(geometry,rows,scale),output_count)) == True{} : Bool}:
  +input_counted = {BandShapeCount.counted(~Element,geometry,first,rows,input,metadata,ready) : BandShape.Count<geometry,rows,Buffer.length(~Element,input)>}
  +input_exact = BandShapeCount.count_refinement_exact(geometry,rows,Buffer.length(~Element,input),input_counted)
  +input_shape_exact = BandUpsampleExecution.source_exact(geometry,first,rows,Buffer.length(~Element,input),metadata,input_counted)
  +output_exact = BandUpsampleExecution.count_exact(ShapeLaws.elements(BandUpsampleExecution.repeated_shape(geometry,rows,scale)),output_count,
    BandUpsampleExecution.counted(geometry,rows,scale,Bands.local_count(geometry,rows),output_count,descriptor))
  +source = {BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)) : Views.View}
  +target = {BandUpsampleExecution.canonical_view(BandUpsampleExecution.repeated_shape(geometry,rows,scale),output_count) : Views.View}
  CopyValues.joined_guard(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,output_count),target,
    BandUpsampleExecution.same_shape(geometry,scale,rows,Bands.local_count(geometry,rows),output_count,descriptor),
    canonical_contiguous(BandUpsampleExecution.repeated_shape(geometry,rows,scale),output_count,output_exact),
    CopyValues.valid_model(~Element,input,source,Bands.local_count(geometry,rows),input_exact,ready,repeated_fits(geometry,rows,scale,Bands.local_count(geometry,rows),input_shape_exact)),
    CopyValues.valid_allocated(~Element,output_count,fill,target,BandUpsampleExecution.allocated(geometry,rows,scale,Bands.local_count(geometry,rows),output_count,descriptor),
      canonical_fits(BandUpsampleExecution.repeated_shape(geometry,rows,scale),output_count,output_exact)))

# Scaled row geometry uses formal machine limits. Actual default limits and
# upsample guards are supplied by the placement caller's checked receipts.
def scaled_divmod(+value: Nat,+divisor: Nat,+scale: Nat,+divisor_positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+scale_positive: {Nat.is_lt(0n,scale) == True{} : Bool}) ->
  {Nat.divmod(Nat.mul(value,scale),Nat.mul(divisor,scale)) == (Nat.div(value,divisor),Nat.mul(Nat.mod(value,divisor),scale)) : Nat & Nat}:
  +quotient = Nat.div(value,divisor)
  +remainder = Nat.mod(value,divisor)
  Division.unique(Nat.mul(value,scale),Nat.mul(divisor,scale),quotient,Nat.mul(remainder,scale),
    Equal.trans(Nat,Nat.mul(value,scale),Nat.mul(Nat.add(Nat.mul(quotient,divisor),remainder),scale),Nat.add(Nat.mul(quotient,Nat.mul(divisor,scale)),Nat.mul(remainder,scale)),
      Equal.cong(Nat,Nat,value => Nat.mul(value,scale),value,Nat.add(Nat.mul(quotient,divisor),remainder),DivisionOrder.quotient_exact(value,divisor,divisor_positive)),
      Equal.trans(Nat,Nat.mul(Nat.add(Nat.mul(quotient,divisor),remainder),scale),Nat.add(Nat.mul(Nat.mul(quotient,divisor),scale),Nat.mul(remainder,scale)),
        Nat.add(Nat.mul(quotient,Nat.mul(divisor,scale)),Nat.mul(remainder,scale)),RowMajor.distrib(Nat.mul(quotient,divisor),remainder,scale),
        Equal.cong(Nat,Nat,part => Nat.add(part,Nat.mul(remainder,scale)),Nat.mul(Nat.mul(quotient,divisor),scale),Nat.mul(quotient,Nat.mul(divisor,scale)),RowMajor.assoc(quotient,divisor,scale)))),
    CountCompletion.positive_product(divisor,scale,divisor_positive,scale_positive),
    RowMajor.multiply_lt(remainder,divisor,scale,DivisionOrder.remainder_small(value,divisor,divisor_positive),scale_positive))

def scaled_remainder(+value: Nat,+divisor: Nat,+scale: Nat,divisor_positive: {Nat.is_lt(0n,divisor) == True{} : Bool},scale_positive: {Nat.is_lt(0n,scale) == True{} : Bool}) ->
  {Nat.mod(Nat.mul(value,scale),Nat.mul(divisor,scale)) == Nat.mul(Nat.mod(value,divisor),scale) : Nat}:
  Equal.cong(Nat & Nat,Nat,(pair => Pair.snd(Nat,Nat,pair)),Nat.divmod(Nat.mul(value,scale),Nat.mul(divisor,scale)),(Nat.div(value,divisor),Nat.mul(Nat.mod(value,divisor),scale)),scaled_divmod(value,divisor,scale,divisor_positive,scale_positive))

type GeometryRowEnvelope<-geometry: Bands.Geometry,-scale: U32,-capacity: U32> is Data:
  GeometryRowEnvelope{room: {Nat.is_le(Inspection.total_rows(BandUpsampleExecution.scaled_geometry(geometry,scale)),U32.to_nat(capacity)) == True{} : Bool}}

def height_room(+geometry: Bands.Geometry,+scale: U32,+capacity: U32,images_positive: {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_batch(geometry))) == True{} : Bool},envelope: GeometryRowEnvelope<geometry,scale,capacity>) ->
  {Nat.is_le(U32.to_nat((BandUpsampleExecution.producer_height(geometry) * scale : U32)),U32.to_nat(capacity)) == True{} : Bool}:
  Bands.Geometry{+batch,channels,+height,width} = geometry
  GeometryRowEnvelope{room} = envelope
  Order.transitive(U32.to_nat((height * scale : U32)),Nat.mul(U32.to_nat(batch),U32.to_nat((height * scale : U32))),U32.to_nat(capacity),
    %RowMajor.multiply_commutative(U32.to_nat((height * scale : U32)),U32.to_nat(batch)) : {Nat.is_le(U32.to_nat((height * scale : U32)),_) == True{} : Bool}
    RowMajor.product_covers(U32.to_nat((height * scale : U32)),U32.to_nat(batch),images_positive),room)

def doubled_room(+limits: Machine.Limits,+capacity: U32,capacity_actual: {capacity == Machine.capacity(limits) : U32},+half: U32,half_actual: {half == Machine.half_word(limits) : U32},+value: Nat,room: {Nat.is_le(value,U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(value),U32.to_nat(half)) == True{} : Bool}:
  %Equal.sym(U32,half,Machine.half_word(limits),half_actual) : {Nat.is_le(Nat.double(value),U32.to_nat(_)) == True{} : Bool}
  Limits.double_bound(limits,value,%capacity_actual : {Nat.is_le(value,U32.to_nat(_)) == True{} : Bool} room)

def scaled_span(+first: Nat,+height: Nat,+rows: Nat,+scale: Nat,+height_positive: {Nat.is_lt(0n,height) == True{} : Bool},scale_positive: {Nat.is_lt(0n,scale) == True{} : Bool},
  source: {Nat.is_le(Nat.add(Nat.mod(first,height),rows),height) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mod(Nat.mul(first,scale),Nat.mul(height,scale)),Nat.mul(rows,scale)),Nat.mul(height,scale)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mod(Nat.mul(first,scale),Nat.mul(height,scale)),Nat.mul(Nat.mod(first,height),scale),scaled_remainder(first,height,scale,height_positive,scale_positive)) :
    {Nat.is_le(Nat.add(_,Nat.mul(rows,scale)),Nat.mul(height,scale)) == True{} : Bool}
  %RowMajor.distrib(Nat.mod(first,height),rows,scale) : {Nat.is_le(_,Nat.mul(height,scale)) == True{} : Bool}
  RowMajor.mul_le(Nat.add(Nat.mod(first,height),rows),height,scale,source)

def geometry_local_guard(+limits: Machine.Limits,+capacity: U32,+capacity_actual: {capacity == Machine.capacity(limits) : U32},+half: U32,+half_actual: {half == Machine.half_word(limits) : U32},+half_fits: {U32.is_le(half,2147483648) == True{} : Bool},
  +height: U32,+first: U32,+rows: U32,+scale: U32,
  first_exact: {U32.to_nat((first * scale : U32)) == Nat.mul(U32.to_nat(first),U32.to_nat(scale)) : Nat},rows_exact: {U32.to_nat((rows * scale : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(scale)) : Nat},
  +height_exact: {Nat.mul(U32.to_nat(height),U32.to_nat(scale)) == U32.to_nat((height * scale : U32)) : Nat},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+scale_positive: {Nat.is_lt(0n,U32.to_nat(scale)) == True{} : Bool},
  +scaled_room: {Nat.is_le(U32.to_nat((height * scale : U32)),U32.to_nat(capacity)) == True{} : Bool},
  source_local: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool}) ->
  {U32.is_le((rows * scale : U32),(height * scale - (first * scale : U32) % (height * scale : U32) : U32)) == True{} : Bool}:
  +source_room = Order.transitive(U32.to_nat(height),U32.to_nat((height * scale : U32)),U32.to_nat(capacity),
    %height_exact : {Nat.is_le(U32.to_nat(height),_) == True{} : Bool} RowMajor.product_covers(U32.to_nat(height),U32.to_nat(scale),scale_positive),scaled_room)
  +scaled_positive = {%height_exact : {Nat.is_lt(0n,_) == True{} : Bool}
    CountCompletion.positive_product(U32.to_nat(height),U32.to_nat(scale),height_positive,scale_positive) : {Nat.is_lt(0n,U32.to_nat((height * scale : U32))) == True{} : Bool}}
  +source_span = Inspection.local_span(height,first,rows,half,half_fits,height_positive,doubled_room(limits,capacity,capacity_actual,half,half_actual,U32.to_nat(height),source_room),source_local)
  Inspection.local_guard((height * scale : U32),(first * scale : U32),(rows * scale : U32),half,half_fits,scaled_positive,
    doubled_room(limits,capacity,capacity_actual,half,half_actual,U32.to_nat((height * scale : U32)),scaled_room),
    %Equal.sym(Nat,U32.to_nat((first * scale : U32)),Nat.mul(U32.to_nat(first),U32.to_nat(scale)),first_exact) :
      {Nat.is_le(Nat.add(Nat.mod(_,U32.to_nat((height * scale : U32))),U32.to_nat((rows * scale : U32))),U32.to_nat((height * scale : U32))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat((rows * scale : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),rows_exact) :
      {Nat.is_le(Nat.add(Nat.mod(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),U32.to_nat((height * scale : U32))),_),U32.to_nat((height * scale : U32))) == True{} : Bool}
    %height_exact : {Nat.is_le(Nat.add(Nat.mod(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),_),Nat.mul(U32.to_nat(rows),U32.to_nat(scale))),_) == True{} : Bool}
    scaled_span(U32.to_nat(first),U32.to_nat(height),U32.to_nat(rows),U32.to_nat(scale),height_positive,scale_positive,source_span))

# Scaled image-local placement from actual source metadata.
# The row envelope is automatic for positive C/W and for zero N. Zero C/W
# alone cannot bound scaled row metadata: a caller must use an empty result
# path or independently establish the row envelope before transforming bands.
type PlacementsRowEnvelope<-geometry: Bands.Geometry,-scale: U32> is Data:
  PlacementsRowEnvelope{limits: Machine.Limits,actual_limits: {limits == Default.limits() : Machine.Limits},
    capacity: U32,capacity_actual: {capacity == Machine.capacity(limits) : U32},
    half: U32,half_actual: {half == Machine.half_word(limits) : U32},half_fits: {U32.is_le(half,2147483648) == True{} : Bool},
    rows: GeometryRowEnvelope<geometry,scale,capacity>}

def active_envelope_at(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+limits: Machine.Limits,+actual_limits: {limits == Default.limits() : Machine.Limits},
  +capacity: U32,+capacity_default: {capacity == Machine.capacity(Default.limits()) : U32},+half: U32,half_actual: {half == Machine.half_word(limits) : U32},half_fits: {U32.is_le(half,2147483648) == True{} : Bool},
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  original: {shape == BandCount.extents(geometry) : List<&2,U32>},truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) -> PlacementsRowEnvelope<geometry,scale>:
  PlacementsRowEnvelope{limits,actual_limits,capacity,
    Equal.trans(U32,capacity,Machine.capacity(Default.limits()),Machine.capacity(limits),capacity_default,
      Equal.sym(U32,Machine.capacity(limits),Machine.capacity(Default.limits()),Equal.cong(Machine.Limits,U32,Machine.capacity,limits,Default.limits(),actual_limits))),
    half,half_actual,half_fits,GeometryRowEnvelope{BandUpsampleExecution.total_rows_bound(geometry,shape,scale,capacity,capacity_default,channels_positive,width_positive,original,truth)}}

def active_envelope(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  original: {shape == BandCount.extents(geometry) : List<&2,U32>},truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) -> PlacementsRowEnvelope<geometry,scale>:
  active_envelope_at(geometry,shape,scale,Default.limits(),{==},Machine.capacity(Default.limits()),{==},Machine.half_word(Default.limits()),{==},Limits.half_word_fits(Default.limits()),channels_positive,width_positive,original,truth)

def scaled_words(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+first: U32,+rows: U32,+length: Nat,
  metadata: BandShape.Leaf(geometry,first,rows,length),envelope: PlacementsRowEnvelope<geometry,scale>,+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) -> BandUpsampleExecution.Words<scale,first,rows>:
  Bands.Geometry{batch,channels,+height,width} = geometry
  PlacementsRowEnvelope{limits,actual_limits,+capacity,capacity_actual,half,half_actual,half_fits,GeometryRowEnvelope{+room}} = envelope
  BandUpsampleExecution.scaled_words(geometry,scale,first,rows,length,capacity,metadata,
    BandUpsampleExecution.native_exact(height,scale,BandUpsampleExecution.height_scaled(geometry,shape,scale,truth)),room)

def row_fit(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,metadata: BandShape.Leaf(geometry,first,rows,length)) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry)) == True{} : Bool}:
  Bands.Geometry{batch,channels,height,width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  rows_fit

def owner_descriptor(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Sized<Element,input>,+original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) ->
  BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale)>:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  +allocation = {BandUpsampleExecution.guards_allocation(geometry,shape,scale,truth) : Count.Count<ShapeLaws.elements(Layout.scaled_extents(shape,scale)),Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale)))>}
  +global_count = Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale)))
  +global_exact = Equal.trans(Nat,Nat.mul(U32.to_nat(BandUpsampleExecution.producer_batch(geometry)),Nat.mul(U32.to_nat(BandUpsampleExecution.producer_channels(geometry)),
      Nat.mul(Nat.mul(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(scale)),Nat.mul(U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale))))),ShapeLaws.elements(Layout.scaled_extents(shape,scale)),U32.to_nat(global_count),
    Equal.sym(Nat,ShapeLaws.elements(Layout.scaled_extents(shape,scale)),Nat.mul(U32.to_nat(BandUpsampleExecution.producer_batch(geometry)),Nat.mul(U32.to_nat(BandUpsampleExecution.producer_channels(geometry)),
      Nat.mul(Nat.mul(U32.to_nat(BandUpsampleExecution.producer_height(geometry)),U32.to_nat(scale)),Nat.mul(U32.to_nat(BandUpsampleExecution.producer_width(geometry)),U32.to_nat(scale))))),BandUpsampleExecution.scaled_elements(geometry,shape,scale,original,truth)),
    BandUpsampleExecution.count_exact(ShapeLaws.elements(Layout.scaled_extents(shape,scale)),global_count,allocation))
  BandUpsampleExecution.from_global(geometry,rows,scale,Bands.local_count(geometry,rows),global_count,
    BandUpsampleExecution.source_exact(geometry,first,rows,Buffer.length(~Element,input),metadata,BandShapeCount.counted(~Element,geometry,first,rows,input,metadata,ready)),
    BandUpsampleExecution.local_bound(geometry,first,rows,scale,global_count,row_fit(geometry,first,rows,Buffer.length(~Element,input),metadata),global_exact),
    BandUpsampleExecution.count_bounded(ShapeLaws.elements(Layout.scaled_extents(shape,scale)),global_count,allocation))

type PositiveRows<-geometry: Bands.Geometry> is Data:
  PositiveRows{images: {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_batch(geometry))) == True{} : Bool},height: {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_height(geometry))) == True{} : Bool}}

def positive_rows(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,metadata: BandShape.Leaf(geometry,first,rows,length),nonzero: {U32.is_zero(rows) == False{} : Bool}) -> PositiveRows<geometry>:
  Bands.Geometry{+batch,channels,+height,width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  +total_positive = Bounds.nat_trans(0n,U32.to_nat(rows),Nat.mul(U32.to_nat(batch),U32.to_nat(height)),CountCompletion.positive_word(rows,nonzero),
    BandUpsampleExecution.bounds_rows_within(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(batch),U32.to_nat(height),rows_fit))
  PositiveRows{RowMajor.left_factor_positive(U32.to_nat(batch),U32.to_nat(height),total_positive),
    RowMajor.left_factor_positive(U32.to_nat(height),U32.to_nat(batch),
      %Equal.sym(Nat,Nat.mul(U32.to_nat(height),U32.to_nat(batch)),Nat.mul(U32.to_nat(batch),U32.to_nat(height)),RowMajor.multiply_commutative(U32.to_nat(height),U32.to_nat(batch))) : {Nat.is_lt(0n,_) == True{} : Bool} total_positive)}

def images_positive(+geometry: Bands.Geometry,evidence: PositiveRows<geometry>) -> {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_batch(geometry))) == True{} : Bool}:
  PositiveRows{images,height} = evidence
  images

def height_positive(+geometry: Bands.Geometry,evidence: PositiveRows<geometry>) -> {Nat.is_lt(0n,U32.to_nat(BandUpsampleExecution.producer_height(geometry))) == True{} : Bool}:
  PositiveRows{images,height} = evidence
  height

def local_field(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,metadata: BandShape.Leaf(geometry,first,rows,length)) ->
  {U32.is_le(rows,(BandUpsampleExecution.producer_height(geometry) - first % BandUpsampleExecution.producer_height(geometry) : U32)) == True{} : Bool}:
  Bands.Geometry{batch,channels,height,width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  local

def local_guard_when(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+first: U32,+rows: U32,+length: Nat,empty: Bool,
  classified: {U32.is_zero(rows) == empty : Bool},+metadata: BandShape.Leaf(geometry,first,rows,length),+words: BandUpsampleExecution.Words<scale,first,rows>,+envelope: PlacementsRowEnvelope<geometry,scale>,
  +truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) ->
  {U32.is_le((rows * scale : U32),(BandUpsampleExecution.producer_height(geometry) * scale - (first * scale : U32) % (BandUpsampleExecution.producer_height(geometry) * scale : U32) : U32)) == True{} : Bool}:
  Bands.Geometry{batch,channels,+height,width} = geometry
  match empty:
    case True{}:
      +zero = Equal.trans(U32,(rows * scale : U32),(0 * scale : U32),0,
        Equal.cong(U32,U32,row => (row * scale : U32),rows,0,ShapeLaws.native_equal(rows,0,classified)),Arithmetic.multiply_zero_left(scale))
      %Equal.sym(U32,(rows * scale : U32),0,zero) : {U32.is_le(_,(height * scale - (first * scale : U32) % (height * scale : U32) : U32)) == True{} : Bool}
      %Equal.sym(Bool,U32.is_le(0,(height * scale - (first * scale : U32) % (height * scale : U32) : U32)),Nat.is_le(0n,U32.to_nat((height * scale - (first * scale : U32) % (height * scale : U32) : U32))),
        Comparison.le(0,(height * scale - (first * scale : U32) % (height * scale : U32) : U32))) : {_ == True{} : Bool}
      Order.zero_le(U32.to_nat((height * scale - (first * scale : U32) % (height * scale : U32) : U32)))
    case False{}:
      PlacementsRowEnvelope{+limits,actual_limits,+capacity,+capacity_actual,+half,+half_actual,+half_fits,+row_envelope} = envelope
      +positive = {positive_rows(geometry,first,rows,length,metadata,classified) : PositiveRows<geometry>}
      geometry_local_guard(limits,capacity,capacity_actual,half,half_actual,half_fits,height,first,rows,scale,
        BandUpsampleExecution.words_first_exact(scale,first,rows,words),BandUpsampleExecution.words_rows_exact(scale,first,rows,words),BandUpsampleExecution.native_exact(height,scale,BandUpsampleExecution.height_scaled(geometry,shape,scale,truth)),
        height_positive(geometry,positive),BandUpsampleExecution.scale_positive(geometry,shape,scale,truth),
        height_room(geometry,scale,capacity,images_positive(geometry,positive),row_envelope),local_field(geometry,first,rows,length,metadata))

def placements_local_guard(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+first: U32,+rows: U32,+length: Nat,+metadata: BandShape.Leaf(geometry,first,rows,length),
  +words: BandUpsampleExecution.Words<scale,first,rows>,envelope: PlacementsRowEnvelope<geometry,scale>,truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) ->
  {U32.is_le((rows * scale : U32),(BandUpsampleExecution.producer_height(geometry) * scale - (first * scale : U32) % (BandUpsampleExecution.producer_height(geometry) * scale : U32) : U32)) == True{} : Bool}:
  local_guard_when(geometry,shape,scale,first,rows,length,U32.is_zero(rows),{==},metadata,words,envelope,truth)

def placement(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Sized<Element,input>,+envelope: PlacementsRowEnvelope<geometry,scale>,
  +original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) -> BandUpsampleExecution.Placement<geometry,scale,first,rows>:
  Bands.Geometry{batch,channels,height,width} = geometry
  +words = {scaled_words(geometry,shape,scale,first,rows,Buffer.length(~Element,input),metadata,envelope,truth) : BandUpsampleExecution.Words<scale,first,rows>}
  +descriptor = {owner_descriptor(~Element,geometry,shape,scale,first,rows,input,metadata,ready,original,truth) : BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale)>}
  BandUpsampleExecution.Placement{BandUpsampleExecution.words_first_exact(scale,first,rows,words),BandUpsampleExecution.words_rows_exact(scale,first,rows,words),
    BandShape.Fields{placements_local_guard(geometry,shape,scale,first,rows,Buffer.length(~Element,input),metadata,words,envelope,truth),
      BandUpsampleExecution.scaled_rows_fit(geometry,scale,first,rows,Buffer.length(~Element,input),metadata,words,BandUpsampleExecution.native_exact(BandUpsampleExecution.producer_height(geometry),scale,BandUpsampleExecution.height_scaled(geometry,shape,scale,truth))),
      BandUpsampleExecution.output_length_exact(geometry,shape,scale,first,rows,descriptor,words,truth)}}

def placements(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,bands: Model.Model<Element>,
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands),+envelope: PlacementsRowEnvelope<geometry,scale>,
  +original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) -> BandUpsampleExecution.Placements(~Element,geometry,scale,bands):
  match bands:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      placement(~Element,geometry,shape,scale,first,rows,{Buffer.Model{tree,length,certificate} : Buffer.Model<Element>},metadata,ready,envelope,original,truth)
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{placements(~Element,geometry,shape,scale,left,left_metadata,left_ready,envelope,original,truth),
        placements(~Element,geometry,shape,scale,right,right_metadata,right_ready,envelope,original,truth)}

def active_placements(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+bands: Model.Model<Element>,
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands),
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  +original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool}) -> BandUpsampleExecution.Placements(~Element,geometry,scale,bands):
  placements(~Element,geometry,shape,scale,bands,metadata,ready,active_envelope(geometry,shape,scale,channels_positive,width_positive,original,truth),original,truth)

type Dimensions<-batch: U32,-channels: U32,-height: U32,-width: U32> is Data:
  Dimensions{images: {Nat.is_lt(0n,U32.to_nat(batch)) == True{} : Bool},channels: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
    height: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},width: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool}}

def rank_four_dimensions(+batch: U32,+channels: U32,+height: U32,+width: U32,+nonempty: {Shapes.has_zero([batch,channels,height,width]) == False{} : Bool}) -> Dimensions<batch,channels,height,width>:
  +images_false = Boolean.or_false_left(U32.is_zero(batch),Shapes.has_zero([channels,height,width]),nonempty)
  +channels_tail = Boolean.or_false_right(U32.is_zero(batch),Shapes.has_zero([channels,height,width]),nonempty)
  +channels_false = Boolean.or_false_left(U32.is_zero(channels),Shapes.has_zero([height,width]),channels_tail)
  +height_tail = Boolean.or_false_right(U32.is_zero(channels),Shapes.has_zero([height,width]),channels_tail)
  +height_false = Boolean.or_false_left(U32.is_zero(height),Shapes.has_zero([width]),height_tail)
  +width_false = Boolean.or_false_left(U32.is_zero(width),False{},Boolean.or_false_right(U32.is_zero(height),Shapes.has_zero([width]),height_tail))
  Dimensions{CountCompletion.positive_word(batch,images_false),CountCompletion.positive_word(channels,channels_false),CountCompletion.positive_word(height,height_false),CountCompletion.positive_word(width,width_false)}

def source_dimensions(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+truth: {BandUpsampleExecution.guard(geometry,shape,scale) == True{} : Bool},
  dimensions: Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),(BandUpsampleExecution.producer_height(geometry) * scale : U32),(BandUpsampleExecution.producer_width(geometry) * scale : U32)>) ->
  Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)>:
  Bands.Geometry{batch,channel_count,+height,+width} = geometry
  Dimensions{images,channels_positive,+height_positive,+width_positive} = dimensions
  Dimensions{images,channels_positive,
    RowMajor.left_factor_positive(U32.to_nat(height),U32.to_nat(scale),
      %Equal.sym(Nat,Nat.mul(U32.to_nat(height),U32.to_nat(scale)),U32.to_nat((height * scale : U32)),BandUpsampleExecution.native_exact(height,scale,BandUpsampleExecution.height_scaled(geometry,shape,scale,truth))) : {Nat.is_lt(0n,_) == True{} : Bool} height_positive),
    RowMajor.left_factor_positive(U32.to_nat(width),U32.to_nat(scale),
      %Equal.sym(Nat,Nat.mul(U32.to_nat(width),U32.to_nat(scale)),U32.to_nat((width * scale : U32)),BandUpsampleExecution.native_exact(width,scale,BandUpsampleExecution.width_scaled(geometry,shape,scale,truth))) : {Nat.is_lt(0n,_) == True{} : Bool} width_positive)}

def enabled_dimensions(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)>:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  source_dimensions(geometry,shape,scale,BandUpsampleExecution.legal(geometry,shape,scale,truth),rank_four_dimensions(batch,channels,(height * scale : U32),(width * scale : U32),
    %BandUpsampleExecution.scaled_shape(geometry,shape,scale,original) : {Shapes.has_zero(_) == False{} : Bool}
    BandUpsampleExecution.nonempty(geometry,shape,scale,truth)))

def channels_positive_dimensions(+geometry: Bands.Geometry,dimensions: Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)>) ->
  {BandGather.channels_positive(geometry) == True{} : Bool}:
  Bands.Geometry{native_batch,native_channels,native_height,native_width} = geometry
  Dimensions{images,channels,height,width} = dimensions
  channels

def width_positive_dimensions(+geometry: Bands.Geometry,dimensions: Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)>) ->
  {BandGather.width_positive(geometry) == True{} : Bool}:
  Bands.Geometry{native_batch,native_channels,native_height,native_width} = geometry
  Dimensions{images,channels,height,width} = dimensions
  width

# The actual selector's nonempty condition discharges positive dimensions.
# Zero dimensions remain legal public shapes and use the dense fallback.
def enabled_placements(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+bands: Model.Model<Element>,
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands),
  +original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) -> BandUpsampleExecution.Placements(~Element,geometry,scale,bands):
  +dimensions = {enabled_dimensions(geometry,shape,scale,original,truth) : Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)>}
  active_placements(~Element,geometry,shape,scale,bands,metadata,ready,channels_positive_dimensions(geometry,dimensions),width_positive_dimensions(geometry,dimensions),original,BandUpsampleExecution.legal(geometry,shape,scale,truth))

def enabled_well_formed(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  source: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,+original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  BandShape.WellFormed<Element,BandUpsampleExecution.scaled_geometry(geometry,scale),Model.contents(~Element,BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),
    Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands)))>:
  Bands.Geometry{batch,channels,height,width} = geometry
  BandShape.WellFormed{+metadata,+ready,coverage} = source
  +dimensions = {enabled_dimensions(geometry,shape,scale,original,truth) : Dimensions<BandUpsampleExecution.producer_batch(geometry),BandUpsampleExecution.producer_channels(geometry),BandUpsampleExecution.producer_height(geometry),BandUpsampleExecution.producer_width(geometry)>}
  BandUpsampleExecution.scheduled_well_formed(~Element,geometry,shape,scale,fill,workers,bands,width_positive_dimensions(geometry,dimensions),BandShape.WellFormed{metadata,ready,coverage},original,
    BandUpsampleExecution.legal(geometry,shape,scale,truth),enabled_placements(~Element,geometry,shape,scale,bands,metadata,Model.ready(~Element,bands,ready),original,truth))

def leaf_finite_from_ready(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+rows: U32,+input_count: U32,+output_count: U32,+fill: Element,+input: Buffer.Model<Element>,ready: Depth.Sized<output_count>) ->
  CopyValues.Finite<Element,Views.is_contiguous(BandUpsampleExecution.repeated_view(geometry,rows,scale,input_count)),U32.to_nat(output_count),BandUpsampleExecution.repeated_view(geometry,rows,scale,input_count),Buffer.tree(~Element,input),Tree.capacity(Element,BandUpsampleExecution.fresh_tree(~Element,fill,output_count))>:
  Depth.Sized{depth_ok,room} = ready
  CopyValues.owned_finite(~Element,Views.is_contiguous(BandUpsampleExecution.repeated_view(geometry,rows,scale,input_count)),U32.to_nat(output_count),BandUpsampleExecution.repeated_view(geometry,rows,scale,input_count),Buffer.tree(~Element,input),
    Buffer.fresh(~Element,Tensors.depth_for(output_count),fill,U32.to_nat(output_count)),{==},Buffer.fresh_ready(~Element,Tensors.depth_for(output_count),fill,U32.to_nat(output_count),depth_ok,room))

def leaf_finite(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+rows: U32,+input_count: U32,+output_count: U32,+fill: Element,+input: Buffer.Model<Element>,
  +descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,input_count,output_count>) ->
  CopyValues.Finite<Element,Views.is_contiguous(BandUpsampleExecution.repeated_view(geometry,rows,scale,input_count)),U32.to_nat(output_count),BandUpsampleExecution.repeated_view(geometry,rows,scale,input_count),Buffer.tree(~Element,input),Tree.capacity(Element,BandUpsampleExecution.fresh_tree(~Element,fill,output_count))>:
  leaf_finite_from_ready(~Element,geometry,scale,rows,input_count,output_count,fill,input,BandUpsampleExecution.allocated(geometry,rows,scale,input_count,output_count,descriptor))

def source_sample(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+coordinates: CursorLaws.Coordinates,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,+metadata: BandShape.Leaf(geometry,first,rows,length),ready: Buffer.ContentReady<Element,Buffer.Content{input,length}>,
  values: BandValues.Leaf(~Element,~Context,~original,context,geometry,first,rows,input,length),+inside: {Nat.is_lt(U32.to_nat(CopyValues.cursor_address(coordinates)),length) == True{} : Bool}) ->
  {TensorVisit.sample(~Element,input,coordinates) == original(context,Bands.global_index(geometry,first,rows,CopyValues.cursor_address(coordinates))) : Element}:
  CursorLaws.Coordinates{+axes,+offset,valid} = coordinates
  +address = U32.from_nat(CursorModel.address(axes,offset))
  %Arithmetic.native_roundtrip(address) :
    {Tree.read(Element,input,_) == original(context,Bands.global_index(geometry,first,rows,_)) : Element}
  BandGather.flat_read_content(~Element,~Context,~original,context,geometry,limits,first,rows,Buffer.Content{input,length},capacity,U32.to_nat(address),domain,metadata,ready,values,inside)

def SourceSamples(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,count: Nat,+coordinates: CursorLaws.Coordinates) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      Shape.Both<{TensorVisit.sample(~Element,input,coordinates) == original(context,Bands.global_index(geometry,first,rows,CopyValues.cursor_address(coordinates))) : Element},
        SourceSamples(~Element,~Context,~original,context,geometry,first,rows,input,rest,CursorLaws.next(coordinates))>

def source_samples(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,count: Nat,coordinates: CursorLaws.Coordinates,+length: U32,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,+metadata: BandShape.Leaf(geometry,first,rows,U32.to_nat(length)),+ready: Buffer.ContentReady<Element,Buffer.Content{input,U32.to_nat(length)}>,
  +values: BandValues.Leaf(~Element,~Context,~original,context,geometry,first,rows,input,U32.to_nat(length)),reads: CursorBounds.ReadSchedule(count,CopyValues.cursor_axes(coordinates),CopyValues.cursor_offset(coordinates),length)) ->
  SourceSamples(~Element,~Context,~original,context,geometry,first,rows,input,count,coordinates):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      CursorLaws.Coordinates{+axes,+offset,+valid} = coordinates
      Shape.Both{inside,tail_reads} = reads
      Shape.Both{
        source_sample(~Element,~Context,~original,context,geometry,first,rows,input,U32.to_nat(length),CursorLaws.Coordinates{axes,offset,valid},limits,capacity,domain,metadata,ready,values,inside),
        source_samples(~Element,~Context,~original,context,geometry,first,rows,input,rest,CursorLaws.Coordinates{CursorModel.advance(axes),offset,CursorLaws.advance_valid(axes,valid)},length,limits,capacity,domain,metadata,ready,values,tail_reads)}

def SourceAt(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+coordinates: CursorLaws.Coordinates,index: Nat) -> Data:
  {TensorVisit.sample(~Element,input,Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,coordinates)) ==
    original(context,Bands.global_index(geometry,first,rows,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,coordinates)))) : Element}

def source_sample_at(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,count: Nat,+coordinates: CursorLaws.Coordinates,index: Nat,
  samples: SourceSamples(~Element,~Context,~original,context,geometry,first,rows,input,count,coordinates),inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  SourceAt(~Element,~Context,~original,context,geometry,first,rows,input,coordinates,index):
  match count index:
    case 0n index:
      Empty.absurd(SourceAt(~Element,~Context,~original,context,geometry,first,rows,input,coordinates,index),
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),inside)))
    case 1n+rest 0n:
      Shape.Both{head,tail} = samples
      head
    case 1n+ +rest 1n+ +previous:
      Shape.Both{head,tail} = samples
      source_sample_at(~Element,~Context,~original,context,geometry,first,rows,input,rest,CursorLaws.next(coordinates),previous,tail,inside)

def view_source_samples(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+count: U32,view: Views.View,+length: U32,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,+metadata: BandShape.Leaf(geometry,first,rows,U32.to_nat(length)),+ready: Buffer.ContentReady<Element,Buffer.Content{input,U32.to_nat(length)}>,
  +values: BandValues.Leaf(~Element,~Context,~original,context,geometry,first,rows,input,U32.to_nat(length)),
  counted: {Views.numel(view) == Some{count} : Maybe<&2,U32>},fits: {Views.fits(view,length) == True{} : Bool}) ->
  SourceSamples(~Element,~Context,~original,context,geometry,first,rows,input,U32.to_nat(count),CursorLaws.initial(view)):
  Views.View{+axes,+offset} = view
  source_samples(~Element,~Context,~original,context,geometry,first,rows,input,U32.to_nat(count),CursorLaws.initial(Views.View{axes,offset}),length,limits,capacity,domain,metadata,ready,values,
    CursorBounds.checked_reads(axes,offset,count,length,counted,fits))

def repeated_counted(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+input_count: U32,+output_count: U32,+descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,input_count,output_count>) ->
  {Views.numel(BandUpsampleExecution.repeated_view(geometry,rows,scale,input_count)) == Some{output_count} : Maybe<&2,U32>}:
  Bands.Geometry{batch,channels,height,width} = geometry
  CountCompletion.numel(BandUpsampleExecution.repeated_shape(geometry,rows,scale),output_count,
    BandUpsampleExecution.count_exact(ShapeLaws.elements(BandUpsampleExecution.repeated_shape(geometry,rows,scale)),output_count,BandUpsampleExecution.counted(geometry,rows,scale,input_count,output_count,descriptor)))

def repeated_elements_exact(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+input_count: U32,+output_count: U32,+descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,input_count,output_count>) ->
  {ShapeLaws.elements(Views.axis_extents_of(BandUpsampleExecution.repeated_view(geometry,rows,scale,input_count))) == U32.to_nat(output_count) : Nat}:
  Bands.Geometry{batch,channels,height,width} = geometry
  BandUpsampleExecution.count_exact(ShapeLaws.elements(BandUpsampleExecution.repeated_shape(geometry,rows,scale)),output_count,BandUpsampleExecution.counted(geometry,rows,scale,input_count,output_count,descriptor))

def count_capacity_bounded(+count: U32,+limits: Machine.Limits,within: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) ->
  {U32.is_le(count,2147483648) == True{} : Bool}:
  Arithmetic.count_within(count,U32.to_nat(count),Machine.capacity(limits),2147483648,{==},Bounds.as_nat_le(count,Machine.capacity(limits),within),Limits.capacity_fits(limits))

def count_capacity(+elements: Nat,+count: U32,evidence: Count.Count<elements,count>) -> {U32.is_le(count,2147483648) == True{} : Bool}:
  Count.Count{exact,bounded} = evidence
  count_capacity_bounded(count,Default.limits(),bounded)

def repeated_sample_value(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+scale: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output_count: U32,+index: Nat,
  +descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),output_count>,
  +samples: SourceSamples(~Element,~Context,~original,context,geometry,first,rows,Buffer.tree(~Element,input),U32.to_nat(output_count),CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)))),
  +inside: {Nat.is_lt(index,U32.to_nat(output_count)) == True{} : Bool}) ->
  {CopyValues.sample_value(~Element,CopyValues.CopyContext{Views.is_contiguous(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))),BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)),Buffer.tree(~Element,input)},U32.from_nat(index)) ==
    original(context,Bands.global_index(geometry,first,rows,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))))))) : Element}:
  +view = BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))
  Equal.trans(Element,CopyValues.sample_value(~Element,CopyValues.CopyContext{Views.is_contiguous(view),view,Buffer.tree(~Element,input)},U32.from_nat(index)),
    TensorVisit.sample(~Element,Buffer.tree(~Element,input),Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view))),
    original(context,Bands.global_index(geometry,first,rows,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view))))),
    CopyValues.cursor_sample(~Element,Views.is_contiguous(view),view,Buffer.tree(~Element,input),U32.to_nat(output_count),index,output_count,{==},repeated_elements_exact(geometry,rows,scale,Bands.local_count(geometry,rows),output_count,descriptor),
      count_capacity(ShapeLaws.elements(BandUpsampleExecution.repeated_shape(geometry,rows,scale)),output_count,BandUpsampleExecution.counted(geometry,rows,scale,Bands.local_count(geometry,rows),output_count,descriptor)),Order.reflexive(U32.to_nat(output_count)),inside),
    source_sample_at(~Element,~Context,~original,context,geometry,first,rows,Buffer.tree(~Element,input),U32.to_nat(output_count),CursorLaws.initial(view),index,samples,inside))

# The readiness of a model, stated for its tree and its length.
def tree_ready(~Element: Data,+input: Buffer.Model<Element>,ready: Buffer.Sized<Element,input>) ->
  Buffer.ContentReady<Element,Buffer.Content{Buffer.tree(~Element,input),Buffer.length(~Element,input)}>:
  Buffer.Model{+tree,+length,+certificate} = input
  Buffer.content_ready(~Element,Buffer.Model{tree,length,certificate},ready)

def repeated_source_samples(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+scale: U32,+input: Buffer.Model<Element>,+output_count: U32,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  +values: BandValues.Leaf(~Element,~Context,~original,context,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input)),
  +descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),output_count>) ->
  SourceSamples(~Element,~Context,~original,context,geometry,first,rows,Buffer.tree(~Element,input),U32.to_nat(output_count),CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)))):
  +source_counted = {BandShapeCount.counted(~Element,geometry,first,rows,input,metadata,ready) : BandShape.Count<geometry,rows,Buffer.length(~Element,input)>}
  +source_count = BandShapeCount.count_refinement_exact(geometry,rows,Buffer.length(~Element,input),source_counted)
  view_source_samples(~Element,~Context,~original,context,geometry,first,rows,Buffer.tree(~Element,input),output_count,BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)),Bands.local_count(geometry,rows),limits,capacity,domain,
    %Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),Buffer.length(~Element,input),source_count) : BandShape.Leaf(geometry,first,rows,_) metadata,
    %Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),Buffer.length(~Element,input),source_count) : Buffer.ContentReady<Element,Buffer.Content{Buffer.tree(~Element,input),_}> tree_ready(~Element,input,ready),
    %Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),Buffer.length(~Element,input),source_count) :
      BandValues.Leaf(~Element,~Context,~original,context,geometry,first,rows,Buffer.tree(~Element,input),_) values,
    repeated_counted(geometry,rows,scale,Bands.local_count(geometry,rows),output_count,descriptor),
    repeated_fits(geometry,rows,scale,Bands.local_count(geometry,rows),BandUpsampleExecution.source_exact(geometry,first,rows,Buffer.length(~Element,input),metadata,source_counted)))

type ActualLeaf<-Element: Data,-Context: Data,-original: Context -> U32 -> Element,-context: Context,-geometry: Bands.Geometry,-scale: U32,-fill: Element,-first: U32,-rows: U32,-input: Buffer.Model<Element>> is Data:
  ActualLeaf{
    descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale)>,
    guarded: {CopyValues.copy_guard(~Element,Buffer.pack(~Element,input),BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)),Tensors.buffer(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale)),
      BandUpsampleExecution.canonical_view(BandUpsampleExecution.repeated_shape(geometry,rows,scale),BandUpsampleExecution.producer_output_count(geometry,rows,scale))) == True{} : Bool},
    stores: CopyValues.Finite<Element,Views.is_contiguous(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))),U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale)),
      BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)),Buffer.tree(~Element,input),Tree.capacity(Element,BandUpsampleExecution.fresh_tree(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale)))>,
    source: SourceSamples(~Element,~Context,~original,context,geometry,first,rows,Buffer.tree(~Element,input),U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale)),
      CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))))}

def actual_leaf(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale)>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  +values: BandValues.Leaf(~Element,~Context,~original,context,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input))) ->
  ActualLeaf<Element,Context,original,context,geometry,scale,fill,first,rows,input>:
  ActualLeaf{descriptor,copy_guard(~Element,geometry,scale,fill,first,rows,input,BandUpsampleExecution.producer_output_count(geometry,rows,scale),descriptor,metadata,ready),
    leaf_finite(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),fill,input,descriptor),
    repeated_source_samples(~Element,~Context,~original,context,geometry,first,rows,scale,input,BandUpsampleExecution.producer_output_count(geometry,rows,scale),limits,capacity,domain,metadata,ready,values,descriptor)}

def source_point_from_store(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+scale: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output_count: U32,+capacity: U32,+index: Nat,
  +descriptor: BandUpsampleExecution.Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),output_count>,
  +samples: SourceSamples(~Element,~Context,~original,context,geometry,first,rows,Buffer.tree(~Element,input),U32.to_nat(output_count),CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)))),
  +inside: {Nat.is_lt(index,U32.to_nat(output_count)) == True{} : Bool},
  store: CopyValues.StorePoint(~Element,Views.is_contiguous(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))),U32.to_nat(output_count),BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)),Buffer.tree(~Element,input),capacity,index)) ->
  Shape.Both<{U32.is_lt(U32.from_nat(index),capacity) == True{} : Bool},
    CopyValues.Point<Element,original(context,Bands.global_index(geometry,first,rows,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))))))),
      U32.from_nat(index),BandUpsampleExecution.leaf_writes(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),output_count,input)>>:
  Shape.Both{query_inside,point} = store
  Shape.Both{query_inside,
    %repeated_sample_value(~Element,~Context,~original,context,geometry,scale,first,rows,input,output_count,index,descriptor,samples,inside) :
      CopyValues.Point<Element,_,U32.from_nat(index),BandUpsampleExecution.leaf_writes(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),output_count,input)>
    point}

def actual_source_point(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+index: Nat,
  evidence: ActualLeaf<Element,Context,original,context,geometry,scale,fill,first,rows,input>,+inside: {Nat.is_lt(index,U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale))) == True{} : Bool}) ->
  Shape.Both<{U32.is_lt(U32.from_nat(index),Tree.capacity(Element,BandUpsampleExecution.fresh_tree(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale)))) == True{} : Bool},
    CopyValues.Point<Element,original(context,Bands.global_index(geometry,first,rows,CopyValues.cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))))))),
      U32.from_nat(index),BandUpsampleExecution.leaf_writes(~Element,geometry,scale,rows,Bands.local_count(geometry,rows),BandUpsampleExecution.producer_output_count(geometry,rows,scale),input)>>:
  ActualLeaf{descriptor,guarded,stores,source} = evidence
  source_point_from_store(~Element,~Context,~original,context,geometry,scale,first,rows,input,BandUpsampleExecution.producer_output_count(geometry,rows,scale),Tree.capacity(Element,BandUpsampleExecution.fresh_tree(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale))),index,descriptor,source,inside,
    CopyValues.finite_point(~Element,Views.is_contiguous(BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows))),U32.to_nat(BandUpsampleExecution.producer_output_count(geometry,rows,scale)),BandUpsampleExecution.repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)),Buffer.tree(~Element,input),
      Tree.capacity(Element,BandUpsampleExecution.fresh_tree(~Element,fill,BandUpsampleExecution.producer_output_count(geometry,rows,scale))),index,stores,inside))

def metadata_width_guard(geometry: Bands.Geometry) ->
  {BandShape.width_positive(geometry) == BandGather.width_positive(geometry) : Bool}:
  Bands.Geometry{batch,channels,height,width} = geometry
  {==}

def enabled_leaf(~Element: Data,~Context: Data,~original: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +domain: BandGather.Domain<geometry,Default.limits(),Machine.capacity(Default.limits())>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  +values: BandValues.Leaf(~Element,~Context,~original,context,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input)),
  +original_shape: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {BandUpsampleExecution.enabled(geometry,shape,scale) == True{} : Bool}) ->
  ActualLeaf<Element,Context,original,context,geometry,scale,fill,first,rows,input>:
  actual_leaf(~Element,~Context,~original,context,geometry,scale,fill,first,rows,input,
    BandUpsampleExecution.execution_owner_descriptor(~Element,geometry,shape,scale,first,rows,input,
      Equal.trans(Bool,BandShape.width_positive(geometry),BandGather.width_positive(geometry),True{},metadata_width_guard(geometry),
        width_positive_dimensions(geometry,enabled_dimensions(geometry,shape,scale,original_shape,truth))),metadata,ready,original_shape,BandUpsampleExecution.legal(geometry,shape,scale,truth)),
    Default.limits(),Machine.capacity(Default.limits()),domain,metadata,ready,values)
