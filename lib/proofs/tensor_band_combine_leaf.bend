# The binary band leaf: alignment, storage guards, the checked traversal it
# refines, ordered readback and pointwise semantics.
import Base
import ../tensor_band.bend as Bands
import ./storage_tree_model.bend as Tree
import ./tensor_band_model.bend as Model
import ./nat_order.bend as Order
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./word_encoding.bend as WordEncoding
import ./u32_comparison.bend as Comparison
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../tensor_view.bend as Views
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_band_repartition.bend as BandRepartition
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_vector_view_refinement.bend as Vector
import ../tensor_shape.bend as Shapes
import ../machine_limits.bend as Machine
import ./boolean_logic.bend as Boolean
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_division.bend as Division
import ./nat_interval.bend as Interval
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./storage_certificate.bend as Certificate
import ./storage_readback.bend as StorageReadback
import ./storage_refinement.bend as StorageLaws
import ./tensor_band_dense.bend as BandDense
import ./tensor_band_each.bend as Each
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_plan.bend as BandPlan
import ./tensor_shape_refinement.bend as ShapeLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_map_readback.bend as MapReadback
import ./u32_full_range.bend as FullRange
import ../tensor_access.bend as Access
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../traversal_partition.bend as Partition

# Alignment preserves values and valid band storage in both actual branches.
# The target planner supplies Fits and Coverage; no same-plan premise is needed.
def same_plan(+left: Bands.Plan,+right: Bands.Plan,+same: {Bands.same_plan(left,right) == True{} : Bool}) -> {left == right : Bands.Plan}:
  match left right:
    case Bands.Rows{+first,+rows} Bands.Rows{+other_first,+other_rows}:
      %ShapeLaws.native_equal(first,other_first,Boolean.left(U32.is_eq(first,other_first),U32.is_eq(rows,other_rows),same)) : {Bands.Rows{first,rows} == Bands.Rows{_,other_rows} : Bands.Plan}
      %ShapeLaws.native_equal(rows,other_rows,Boolean.right(U32.is_eq(first,other_first),U32.is_eq(rows,other_rows),same)) : {Bands.Rows{first,rows} == Bands.Rows{first,_} : Bands.Plan}
      {==}
    case Bands.Split{+left,+right} Bands.Split{+other_left,+other_right}:
      %same_plan(left,other_left,Boolean.left(Bands.same_plan(left,other_left),Bands.same_plan(right,other_right),same)) : {Bands.Split{left,right} == Bands.Split{_,other_right} : Bands.Plan}
      %same_plan(right,other_right,Boolean.right(Bands.same_plan(left,other_left),Bands.same_plan(right,other_right),same)) : {Bands.Split{left,right} == Bands.Split{left,_} : Bands.Plan}
      {==}
    case Bands.Rows{first,rows} Bands.Split{left,right}:
      Empty.absurd({Bands.Rows{first,rows} == Bands.Split{left,right} : Bands.Plan},WordEncoding.false_true(same))
    case Bands.Split{left,right} Bands.Rows{first,rows}:
      Empty.absurd({Bands.Split{left,right} == Bands.Rows{first,rows} : Bands.Plan},WordEncoding.false_true(same))

def selected_layout(~Element: Data,same: Bool,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>,
  compared: {Bands.same_plan(layout,Model.layout(~Element,source)) == same : Bool}) ->
  {Model.content_layout(~Element,Model.contents(~Element,Bands.align_when(~Element,same,geometry,fill,layout,Model.pack(~Element,source)))) == layout : Bands.Plan}:
  match same:
    case True{}:
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,source)),Model.content(~Element,source),Model.packed_contents(~Element,source)) : {Model.content_layout(~Element,_) == layout : Bands.Plan}
      Equal.trans(Bands.Plan,Model.content_layout(~Element,Model.content(~Element,source)),Model.layout(~Element,source),layout,
        Equal.sym(Bands.Plan,Model.layout(~Element,source),Model.content_layout(~Element,Model.content(~Element,source)),Model.layout_matches_content(~Element,source)),
        Equal.sym(Bands.Plan,layout,Model.layout(~Element,source),same_plan(layout,Model.layout(~Element,source),compared)))
    case False{}:
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),
        BandRepartition.model(~Element,geometry,fill,layout,Model.content(~Element,source)),BandRepartition.repartitioned(~Element,geometry,fill,layout,source)) : {Model.content_layout(~Element,_) == layout : Bands.Plan}
      BandRepartition.model_layout(~Element,geometry,fill,layout,Model.content(~Element,source))

def aligned_layout(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>) ->
  {Model.content_layout(~Element,Model.contents(~Element,Bands.align(~Element,geometry,fill,layout,Model.pack(~Element,source)))) == layout : Bands.Plan}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,source)),(Model.pack(~Element,source),Model.layout(~Element,source)),Model.plan(~Element,source)) :
    {Model.content_layout(~Element,Model.contents(~Element,Bands.align_planned(~Element,geometry,fill,layout,_))) == layout : Bands.Plan}
  selected_layout(~Element,Bands.same_plan(layout,Model.layout(~Element,source)),geometry,fill,layout,source,{==})

def selected_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,same: Bool,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>,
  shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  target: BandDense.Fits(geometry,layout,capacity,Machine.half_word(limits))) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.contents(~Element,Bands.align_when(~Element,same,geometry,fill,layout,Model.pack(~Element,source)))):
  match same:
    case True{}:
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,source)),Model.content(~Element,source),Model.packed_contents(~Element,source)) : BandValues.Values(~Element,~Logical,~original,logical,geometry,_)
      values
    case False{}: BandRepartition.repartitioned_values(~Element,~Logical,~original,logical,geometry,fill,layout,source,limits,capacity,domain,shape,values,target)

def selected_well_formed(~Element: Data,same: Bool,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>,+capacity: U32,+division_capacity: U32,
  shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,target: BandDense.Fits(geometry,layout,capacity,division_capacity),coverage: BandPlan.Coverage(layout,0n,Inspection.total_rows(geometry))) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.align_when(~Element,same,geometry,fill,layout,Model.pack(~Element,source)))>:
  match same:
    case True{}:
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,source)),Model.content(~Element,source),Model.packed_contents(~Element,source)) : BandShape.WellFormed<Element,geometry,_>
      shape
    case False{}: BandRepartition.repartitioned_well_formed(~Element,geometry,fill,layout,source,capacity,division_capacity,target,coverage)

def aligned_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>,
  shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  target: BandDense.Fits(geometry,layout,capacity,Machine.half_word(limits))) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.contents(~Element,Bands.align(~Element,geometry,fill,layout,Model.pack(~Element,source)))):
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,source)),(Model.pack(~Element,source),Model.layout(~Element,source)),Model.plan(~Element,source)) :
    BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.contents(~Element,Bands.align_planned(~Element,geometry,fill,layout,_)))
  selected_values(~Element,~Logical,~original,Bands.same_plan(layout,Model.layout(~Element,source)),logical,geometry,fill,layout,source,limits,capacity,domain,shape,values,target)

def aligned_well_formed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>,+capacity: U32,+division_capacity: U32,
  shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,target: BandDense.Fits(geometry,layout,capacity,division_capacity),coverage: BandPlan.Coverage(layout,0n,Inspection.total_rows(geometry))) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.align(~Element,geometry,fill,layout,Model.pack(~Element,source)))>:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,source)),(Model.pack(~Element,source),Model.layout(~Element,source)),Model.plan(~Element,source)) :
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.align_planned(~Element,geometry,fill,layout,_))>
  selected_well_formed(~Element,Bands.same_plan(layout,Model.layout(~Element,source)),geometry,fill,layout,source,capacity,division_capacity,shape,target,coverage)

# Actual one-axis storage fits and checked Buffer validity. Counting and
# contiguous descriptor facts belong to tensor_vector_view_refinement.
def positive_nat(value: Nat,nonzero: {Nat.is_eq(value,0n) == False{} : Bool}) -> {Nat.is_lt(0n,value) == True{} : Bool}:
  match value:
    case 0n:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},nonzero)))
    case 1n+rest: {==}

def positive(+count: U32,nonzero: {U32.is_zero(count) == False{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}:
  positive_nat(U32.to_nat(count),
    %Comparison.eq(count,0) : {_ == False{} : Bool}
    nonzero)

def native_positive(+count: U32,nonzero: {U32.is_zero(count) == False{} : Bool}) -> {U32.is_lt(0,count) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt(0,count),Nat.is_lt(0n,U32.to_nat(count)),Comparison.lt(0,count)) : {_ == True{} : Bool}
  positive(count,nonzero)

def predecessor_inside(+count: U32,nonzero: {U32.is_zero(count) == False{} : Bool}) -> {U32.is_lt((count - 1 : U32),count) == True{} : Bool}:
  +lower = positive(count,nonzero)
  +ordered = Order.strict_follow(0n,U32.to_nat(count),lower)
  %Equal.sym(Bool,U32.is_lt((count - 1 : U32),count),Nat.is_lt(U32.to_nat((count - 1 : U32)),U32.to_nat(count)),Comparison.lt((count - 1 : U32),count)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((count - 1 : U32)),Nat.sub(U32.to_nat(count),1n),
    FullRange.subtract(count,1,count,ordered,Order.sub_le(U32.to_nat(count),1n))) :
    {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool}
  Order.sub_lt(U32.to_nat(count),1n,U32.to_nat(count),ordered,Order.lt_succ(U32.to_nat(count),U32.to_nat(count),Order.reflexive(U32.to_nat(count))))

def native_reflexive(+word: U32) -> {U32.is_le(word,word) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(word,word),Nat.is_le(U32.to_nat(word),U32.to_nat(word)),Comparison.le(word,word)) : {_ == True{} : Bool}
  Order.reflexive(U32.to_nat(word))

def advance_last(+count: U32,nonzero: {U32.is_zero(count) == False{} : Bool},
  one_last: {Shapes.checked_multiply((count - 1 : U32),1) == Some{(count - 1 : U32)} : Maybe<&2,U32>}) ->
  {Views.advance(0,(count - 1 : U32),1,count) == Some{(count - 1 : U32)} : Maybe<&2,U32>}:
  +last = (count - 1 : U32)
  %Equal.sym(Bool,U32.is_lt(0,count),True{},native_positive(count,nonzero)) :
    {Views.advance_inside(_,0,last,1,count) == Some{last} : Maybe<&2,U32>}
  %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(last,1),Some{last},one_last) :
    {Views.advance_product(_,0,(last - 0 : U32)) == Some{last} : Maybe<&2,U32>}
  %Equal.sym(U32,(last - 0 : U32),last,Arithmetic.subtract_zero(last)) :
    {Views.advance_product(Some{last},0,_) == Some{last} : Maybe<&2,U32>}
  %Equal.sym(Bool,U32.is_le(last,last),True{},native_reflexive(last)) :
    {Views.address_when(_,0,last) == Some{last} : Maybe<&2,U32>}
  Equal.cong(U32,Maybe<&2,U32>,word => Some{word},(0 + last : U32),last,Arithmetic.add_zero_left(last))

def last_address(+count: U32,+nonzero: {U32.is_zero(count) == False{} : Bool},
  one_last: {Shapes.checked_multiply((count - 1 : U32),1) == Some{(count - 1 : U32)} : Maybe<&2,U32>}) ->
  {Views.address(Vector.vector(count),[(count - 1 : U32)],count) == Some{(count - 1 : U32)} : Maybe<&2,U32>}:
  +last = (count - 1 : U32)
  %Equal.sym(Bool,U32.is_lt(last,count),True{},predecessor_inside(count,nonzero)) :
    {Views.locate([],[],Views.advance_inside(_ && U32.is_lt(0,count),0,last,1,count),count) == Some{last} : Maybe<&2,U32>}
  %Equal.sym(Maybe<&2,U32>,Views.advance(0,last,1,count),Some{last},advance_last(count,nonzero,one_last)) :
    {Views.locate([],[],_,count) == Some{last} : Maybe<&2,U32>}
  %Equal.sym(Bool,U32.is_lt(last,count),True{},predecessor_inside(count,nonzero)) :
    {Views.finish_address(_,last) == Some{last} : Maybe<&2,U32>}
  {==}

def fits_when(+count: U32,empty: Bool,classified: {U32.is_zero(count) == empty : Bool},
  one_last: {Shapes.checked_multiply((count - 1 : U32),1) == Some{(count - 1 : U32)} : Maybe<&2,U32>}) ->
  {Views.fits_nonempty(empty,Vector.vector(count),count) == True{} : Bool}:
  match empty:
    case True{}:
      %Equal.sym(Bool,U32.is_le(0,count),Nat.is_le(0n,U32.to_nat(count)),Comparison.le(0,count)) : {_ == True{} : Bool}
      Order.zero_le(U32.to_nat(count))
    case False{}:
      Equal.cong(Maybe<&2,U32>,Bool,result => Views.found(result),Views.address(Vector.vector(count),[(count - 1 : U32)],count),Some{(count - 1 : U32)},last_address(count,classified,one_last))

def fits(+count: U32,one_last: {Shapes.checked_multiply((count - 1 : U32),1) == Some{(count - 1 : U32)} : Maybe<&2,U32>}) -> {Views.fits(Vector.vector(count),count) == True{} : Bool}:
  %Equal.sym(Bool,Shapes.has_zero([count]),U32.is_zero(count),Vector.zero_flag(count)) :
    {Views.fits_nonempty(_,Vector.vector(count),count) == True{} : Bool}
  fits_when(count,U32.is_zero(count),{==},one_last)

def native_length(+count: U32,+length: Nat,exact: {U32.to_nat(count) == length : Nat}) -> {U32.from_nat(length) == count : U32}:
  %exact : {U32.from_nat(_) == count : U32}
  Arithmetic.native_roundtrip(count)

def certificate(~Element: Data,input: Buffer.Model<Element>) -> Certificate.Certificate<Element,Tree.pack(Element,Buffer.tree(~Element,input))>:
  Buffer.Model{tree,length,certificate} = input
  certificate

def valid(~Element: Data,+count: U32,+input: Buffer.Model<Element>,exact: {U32.to_nat(count) == Buffer.length(~Element,input) : Nat},ready: Buffer.Ready<Element,input>,
  fits: {Views.fits(Vector.vector(count),count) == True{} : Bool}) ->
  {Operations.valid_buffer(Buffer.length(~Element,input),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,input)),
      certificate(~Element,input)),Vector.vector(count)) == True{} : Bool}:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = input
  Buffer.Ready{depth_ok,+room} = ready
  +model = {Buffer.Model{tree,length,Certificate.Certificate{depth,equation}} : Buffer.Model<Element>}
  %Shape.capacity(Element,depth,tree,Buffer.balanced(~Element,model)) :
    {Operations.valid_buffer(length,_,Vector.vector(count)) == True{} : Bool}
  %Equal.sym(U32,U32.from_nat(length),count,native_length(count,length,exact)) :
    {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,tree))) && Views.fits(Vector.vector(count),_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_le(length,U32.to_nat(Tree.capacity(Element,tree))),True{},room) :
    {_ && Views.fits(Vector.vector(count),count) == True{} : Bool}
  %Equal.sym(Bool,Views.fits(Vector.vector(count),count),True{},fits) : {True{} && _ == True{} : Bool}
  {==}

# Band binary operations reuse the checked elementwise traversal refinement.
# Both scheduling modes have the same tree denotation, including rejection
# and malformed-shape ownership fallbacks. Logical Values are proved separately.
def left_content(~Element: Data,result: Operations.Outcome<OperationsLaws.PairOwners(Element)>) -> Buffer.Content<Element>:
  match result:
    case Operations.Accepted{(OperationsLaws.Owner{storage,length},right)}:
      Buffer.Content{Tree.reflect(Element,storage),length}
    case Operations.Rejected{(OperationsLaws.Owner{storage,length},right)}:
      Buffer.Content{Tree.reflect(Element,storage),length}

def selected_content(~Element: Data,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {Buffer.contents(~Element,Bands.combined(~Element,result)) == left_content(~Element,OperationsLaws.observe_pair_outcome(~Element,result)) : Buffer.Content<Element>}:
  match result:
    case Operations.Accepted{(Storage.Buffer{storage,length,certificate},right)}: {==}
    case Operations.Rejected{(Storage.Buffer{storage,length,certificate},right)}: {==}

def refinement_leaf_model(~Element: Data,~operation: Element -> Element -> Element,+count: U32,left: Buffer.Model<Element>,right: Buffer.Model<Element>) -> Buffer.Content<Element>:
  +view = {Views.View{[Views.Axis{count,1}],0} : Views.View}
  left_content(~Element,OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view))

def refinement_leaf(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Bands.combine_leaf(~Element,~operation,count,Buffer.pack(~Element,left),Buffer.pack(~Element,right))) ==
    refinement_leaf_model(~Element,~operation,count,left,right) : Buffer.Content<Element>}:
  +view = {Views.View{[Views.Axis{count,1}],0} : Views.View}
  Equal.trans(Buffer.Content<Element>,
    Buffer.contents(~Element,Bands.combine_leaf(~Element,~operation,count,Buffer.pack(~Element,left),Buffer.pack(~Element,right))),
    left_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.combine(~Element,~operation,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view))),
    refinement_leaf_model(~Element,~operation,count,left,right),
    selected_content(~Element,Operations.combine(~Element,~operation,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view)),
    Equal.cong(Operations.Outcome<OperationsLaws.PairOwners(Element)>,Buffer.Content<Element>,result => left_content(~Element,result),
      OperationsLaws.observe_pair_outcome(~Element,Operations.combine(~Element,~operation,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view)),
      OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view),
      OperationsLaws.combine(~Element,~operation,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view)))

def band_model(~Element: Data,~operation: Element -> Element -> Element,geometry: Bands.Geometry,first: U32,rows: U32,
  left: Buffer.Model<Element>,right: Buffer.Model<Element>) -> Model.Content<Element>:
  +rows = rows
  Model.RowBand{refinement_leaf_model(~Element,~operation,Bands.local_count(geometry,rows),left,right),first,rows}

def model(~Element: Data,~operation: Element -> Element -> Element,geometry: Bands.Geometry,left: Model.Model<Element>,right: Model.Model<Element>) -> Model.Content<Element>:
  Each.content_zip(~Element,~Bands.Geometry,~band_model(~Element,~operation),geometry,left,right)

def band(~Element: Data,~operation: Element -> Element -> Element,+geometry: Bands.Geometry,+first: U32,+rows: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>) ->
  {Model.contents(~Element,Bands.combine_band(~Element,~operation,geometry,first,rows,Buffer.pack(~Element,left),Buffer.pack(~Element,right),Unit{})) ==
    band_model(~Element,~operation,geometry,first,rows,left,right) : Model.Content<Element>}:
  Equal.cong(Buffer.Content<Element>,Model.Content<Element>,buffer => Model.RowBand{buffer,first,rows},
    Buffer.contents(~Element,Bands.combine_leaf(~Element,~operation,Bands.local_count(geometry,rows),Buffer.pack(~Element,left),Buffer.pack(~Element,right))),
    refinement_leaf_model(~Element,~operation,Bands.local_count(geometry,rows),left,right),refinement_leaf(~Element,~operation,Bands.local_count(geometry,rows),left,right))

# The band combine in either mode. Where the two trees differ in shape, the
# left bands are kept.
def combine(~Element: Data,~operation: Element -> Element -> Element,+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>) ->
  {Model.contents(~Element,Bands.combine(~Element,~operation,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))) ==
    model(~Element,~operation,geometry,left,right) : Model.Content<Element>}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,left)),(Model.pack(~Element,left),Model.layout(~Element,left)),Model.plan(~Element,left)) :
    {Model.contents(~Element,Bands.combine_planned(~Element,~operation,geometry,parallel,_,Model.pack(~Element,right))) == model(~Element,~operation,geometry,left,right) : Model.Content<Element>}
  Each.zipped(~Element,~Bands.Geometry,~Bands.combine_band(~Element,~operation),~band_model(~Element,~operation),~band(~Element,~operation),
    left,right,parallel && Bands.same_image(geometry,Model.layout(~Element,left)),parallel,geometry,geometry)

# The actual binary operation retains the left owner's complete row layout,
# independently of scheduling, checked rejection, or malformed right trees.
def model_layout(~Element: Data,~operation: Element -> Element -> Element,+geometry: Bands.Geometry,left: Model.Model<Element>,right: Model.Model<Element>) ->
  {Model.content_layout(~Element,model(~Element,~operation,geometry,left,right)) == Model.layout(~Element,left) : Bands.Plan}:
  match left right:
    case Model.Leaf{buffer,first,rows} Model.Leaf{other,other_first,other_rows}: {==}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,model(~Element,~operation,geometry,left,other_left)),Model.layout(~Element,left),
        model_layout(~Element,~operation,geometry,left,other_left)) :
        {Bands.Split{_,Model.content_layout(~Element,model(~Element,~operation,geometry,right,other_right))} == Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,model(~Element,~operation,geometry,right,other_right)),Model.layout(~Element,right),
        model_layout(~Element,~operation,geometry,right,other_right)) :
        {Bands.Split{Model.layout(~Element,left),_} == Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      {==}
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{other_left,other_right}:
      Equal.sym(Bands.Plan,Model.layout(~Element,Model.Leaf{buffer,first,rows}),Model.content_layout(~Element,Model.content(~Element,Model.Leaf{buffer,first,rows})),
        Model.layout_matches_content(~Element,Model.Leaf{buffer,first,rows}))
    case Model.Fork{+left,+right} Model.Leaf{buffer,first,rows}:
      Equal.sym(Bands.Plan,Model.layout(~Element,Model.Fork{left,right}),Model.content_layout(~Element,Model.content(~Element,Model.Fork{left,right})),
        Model.layout_matches_content(~Element,Model.Fork{left,right}))

def combine_layout(~Element: Data,~operation: Element -> Element -> Element,+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>) ->
  {Model.content_layout(~Element,Model.contents(~Element,Bands.combine(~Element,~operation,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))) ==
    Model.layout(~Element,left) : Bands.Plan}:
  Equal.trans(Bands.Plan,
    Model.content_layout(~Element,Model.contents(~Element,Bands.combine(~Element,~operation,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))),
    Model.content_layout(~Element,model(~Element,~operation,geometry,left,right)),Model.layout(~Element,left),
    Equal.cong(Model.Content<Element>,Bands.Plan,content => Model.content_layout(~Element,content),
      Model.contents(~Element,Bands.combine(~Element,~operation,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))),
      model(~Element,~operation,geometry,left,right),combine(~Element,~operation,geometry,parallel,left,right)),
    model_layout(~Element,~operation,geometry,left,right))

# Actual alignment yields equal layout trees. Their constructor fields give
# the common first/rows needed by every checked binary leaf and recursive fork.
def row_first(layout: Bands.Plan) -> U32:
  match layout:
    case Bands.Rows{first,rows}: first
    case Bands.Split{left,right}: 0

def row_count(layout: Bands.Plan) -> U32:
  match layout:
    case Bands.Rows{first,rows}: rows
    case Bands.Split{left,right}: 0

def layout_match_leaf(layout: Bands.Plan) -> Bool:
  match layout:
    case Bands.Rows{first,rows}: True{}
    case Bands.Split{left,right}: False{}

def left_plan(layout: Bands.Plan) -> Bands.Plan:
  match layout:
    case Bands.Rows{first,rows}: Bands.Rows{0,0}
    case Bands.Split{left,right}: left

def right_plan(layout: Bands.Plan) -> Bands.Plan:
  match layout:
    case Bands.Rows{first,rows}: Bands.Rows{0,0}
    case Bands.Split{left,right}: right

def Common(~Element: Data,left: Model.Model<Element>,right: Model.Model<Element>) -> Data:
  match left right:
    case Model.Leaf{left,+first,+rows} Model.Leaf{right,+other_first,+other_rows}:
      Shape.Both<{first == other_first : U32},{rows == other_rows : U32}>
    case Model.Fork{left,right} Model.Fork{other_left,other_right}:
      Shape.Both<Common(~Element,left,other_left),Common(~Element,right,other_right)>
    case left right: Empty

def matched(~Element: Data,+left: Model.Model<Element>,+right: Model.Model<Element>,+same: {Model.layout(~Element,left) == Model.layout(~Element,right) : Bands.Plan}) -> Common(~Element,left,right):
  match left right:
    case Model.Leaf{left,+first,+rows} Model.Leaf{right,+other_first,+other_rows}:
      Shape.Both{Equal.cong(Bands.Plan,U32,layout => row_first(layout),Bands.Rows{first,rows},Bands.Rows{other_first,other_rows},same),
        Equal.cong(Bands.Plan,U32,layout => row_count(layout),Bands.Rows{first,rows},Bands.Rows{other_first,other_rows},same)}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      +first_plan = {Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      +second_plan = {Bands.Split{Model.layout(~Element,other_left),Model.layout(~Element,other_right)} : Bands.Plan}
      Shape.Both{matched(~Element,left,other_left,Equal.cong(Bands.Plan,Bands.Plan,layout => left_plan(layout),first_plan,second_plan,same)),
        matched(~Element,right,other_right,Equal.cong(Bands.Plan,Bands.Plan,layout => right_plan(layout),first_plan,second_plan,same))}
    case Model.Leaf{buffer,+first,+rows} Model.Fork{+left,+right}:
      WordEncoding.false_true(Equal.sym(Bool,True{},False{},Equal.cong(Bands.Plan,Bool,layout => layout_match_leaf(layout),Bands.Rows{first,rows},Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},same)))
    case Model.Fork{+left,+right} Model.Leaf{buffer,+first,+rows}:
      WordEncoding.false_true(Equal.cong(Bands.Plan,Bool,layout => layout_match_leaf(layout),Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},Bands.Rows{first,rows},same))

# Ordered in-place binary readback. A store changes only its own address;
# later addresses still contain their original left values. The right reader
# is immutable in this tree model. No algebraic law on operation is assumed.
def readback_output(~Element: Data,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) -> Tree.Tree<Element>:
  AccessLaws.ModelResult{reader,readback_output} = result
  readback_output

def run(~Element: Data,~operation: Element -> Element -> Element,count: Nat,position: U32,right: Tree.Tree<Element>,left: Tree.Tree<Element>) -> Tree.Tree<Element>:
  readback_output(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Unit,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Unit>,
      ~ArrayLaws.model_update_step(~Tree.Tree<Element>,~Element,~Element,~Unit,~AccessLaws.sample_index(~Element),~operation,~Traversal.index),
      count,Traversal.State{right,left,position,Unit{}})))

def capacity(~Element: Data,~operation: Element -> Element -> Element,count: Nat,+position: U32,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>) ->
  {Tree.capacity(Element,run(~Element,~operation,count,position,right,left)) == Tree.capacity(Element,left) : U32}:
  match count:
    case 0n: {==}
    case 1n+rest:
      +value = operation(Tree.read(Element,left,position),Tree.read(Element,right,position))
      +written = Tree.write(Element,left,position,value)
      Equal.trans(U32,Tree.capacity(Element,run(~Element,~operation,rest,(position + 1 : U32),right,written)),Tree.capacity(Element,written),Tree.capacity(Element,left),
        capacity(~Element,~operation,rest,(position + 1 : U32),right,written),StorageReadback.write_capacity(~Element,left,position,value))

def balanced(~Element: Data,~operation: Element -> Element -> Element,count: Nat,+position: U32,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>,+depth: Nat,
  evidence: Shape.Balanced(Element,depth,left)) -> Shape.Balanced(Element,depth,run(~Element,~operation,count,position,right,left)):
  match count:
    case 0n: evidence
    case 1n+rest:
      +value = operation(Tree.read(Element,left,position),Tree.read(Element,right,position))
      balanced(~Element,~operation,rest,(position + 1 : U32),right,Tree.write(Element,left,position,value),depth,Shape.write(Element,depth,left,position,value,evidence))

def outside(~Element: Data,~operation: Element -> Element -> Element,count: Nat,+position: Nat,+query: U32,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Element,left) == capacity : U32},+balanced: Shape.Balanced(Element,depth,left),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool},
  +query_inside: {U32.is_lt(query,capacity) == True{} : Bool},+separate: Interval.Outside<position,count,U32.to_nat(query)>) ->
  {Tree.read(Element,run(~Element,~operation,count,U32.from_nat(position),right,left),query) == Tree.read(Element,left,query) : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +value = operation(Tree.read(Element,left,U32.from_nat(position)),Tree.read(Element,right,U32.from_nat(position)))
      +written = Tree.write(Element,left,U32.from_nat(position),value)
      +position_inside = Order.room_index(position,rest,U32.to_nat(capacity),room)
      +different = {
        %Equal.sym(Bool,U32.is_eq(U32.from_nat(position),query),Nat.is_eq(U32.to_nat(U32.from_nat(position)),U32.to_nat(query)),Comparison.eq(U32.from_nat(position),query)) : {_ == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) :
          {Nat.is_eq(_,U32.to_nat(query)) == False{} : Bool}
        %Order.equal_symmetric(U32.to_nat(query),position) : {_ == False{} : Bool}
        Interval.outside_head(position,rest,U32.to_nat(query),separate) : {U32.is_eq(U32.from_nat(position),query) == False{} : Bool}}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,run(~Element,~operation,rest,_,right,written),query) == Tree.read(Element,left,query) : Element}
      Equal.trans(Element,Tree.read(Element,run(~Element,~operation,rest,U32.from_nat(1n+position),right,written),query),Tree.read(Element,written,query),Tree.read(Element,left,query),
        outside(~Element,~operation,rest,1n+position,query,right,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,left),capacity,StorageReadback.write_capacity(~Element,left,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,left,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),query_inside,
          Interval.outside_tail(position,rest,U32.to_nat(query),separate)),
        MapReadback.unchanged(~Element,depth,left,U32.from_nat(position),query,value,capacity,capacity_equal,balanced,limit,MapReadback.address_inside(position,capacity,capacity_ok,position_inside),query_inside,different))

def inside(~Element: Data,~operation: Element -> Element -> Element,count: Nat,+position: Nat,index: Nat,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Element,left) == capacity : U32},+balanced: Shape.Balanced(Element,depth,left),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool},
  +selected: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tree.read(Element,run(~Element,~operation,count,U32.from_nat(position),right,left),U32.from_nat(Nat.add(position,index))) ==
    operation(Tree.read(Element,left,U32.from_nat(Nat.add(position,index))),Tree.read(Element,right,U32.from_nat(Nat.add(position,index)))) : Element}:
  match count index:
    case 0n +index:
      Empty.absurd({Tree.read(Element,left,U32.from_nat(Nat.add(position,index))) ==
        operation(Tree.read(Element,left,U32.from_nat(Nat.add(position,index))),Tree.read(Element,right,U32.from_nat(Nat.add(position,index)))) : Element},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),selected)))
    case 1n+ +rest 0n:
      +value = operation(Tree.read(Element,left,U32.from_nat(position)),Tree.read(Element,right,U32.from_nat(position)))
      +written = Tree.write(Element,left,U32.from_nat(position),value)
      +position_inside = Order.room_index(position,rest,U32.to_nat(capacity),room)
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {Tree.read(Element,run(~Element,~operation,1n+rest,U32.from_nat(position),right,left),U32.from_nat(_)) ==
          operation(Tree.read(Element,left,U32.from_nat(_)),Tree.read(Element,right,U32.from_nat(_))) : Element}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,run(~Element,~operation,rest,_,right,written),U32.from_nat(position)) == value : Element}
      Equal.trans(Element,Tree.read(Element,run(~Element,~operation,rest,U32.from_nat(1n+position),right,written),U32.from_nat(position)),Tree.read(Element,written,U32.from_nat(position)),value,
        outside(~Element,~operation,rest,1n+position,U32.from_nat(position),right,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,left),capacity,StorageReadback.write_capacity(~Element,left,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,left,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),MapReadback.address_inside(position,capacity,capacity_ok,position_inside),
          Interval.Before{
            %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) : {Nat.is_lt(_,1n+position) == True{} : Bool}
            Order.lt_succ(position,position,Order.reflexive(position))}),
        StorageReadback.read_written(~Element,left,U32.from_nat(position),value))
    case 1n+ +rest 1n+ +index:
      +value = operation(Tree.read(Element,left,U32.from_nat(position)),Tree.read(Element,right,U32.from_nat(position)))
      +written = Tree.write(Element,left,U32.from_nat(position),value)
      +query_position = Nat.add(1n+position,index)
      +query = U32.from_nat(query_position)
      +position_inside = Order.room_index(position,rest,U32.to_nat(capacity),room)
      +query_inside = Bounds.nat_trans(query_position,Nat.add(1n+position,rest),U32.to_nat(capacity),Order.offset_lt(1n+position,index,rest,selected),Order.advance_room(position,rest,U32.to_nat(capacity),room))
      +different = {
        %Equal.sym(Bool,U32.is_eq(U32.from_nat(position),query),Nat.is_eq(U32.to_nat(U32.from_nat(position)),U32.to_nat(query)),Comparison.eq(U32.from_nat(position),query)) : {_ == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) : {Nat.is_eq(_,U32.to_nat(query)) == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(query),query_position,Bounds.roundtrip(query_position,capacity,capacity_ok,Bounds.strict_le(query_position,U32.to_nat(capacity),query_inside))) : {Nat.is_eq(position,_) == False{} : Bool}
        Order.less_not_equal(position,query_position,Bounds.nat_trans(position,1n+position,query_position,Order.lt_succ(position,position,Order.reflexive(position)),Order.le_add(1n+position,index))) : {U32.is_eq(U32.from_nat(position),query) == False{} : Bool}}
      %Equal.sym(Nat,Nat.add(position,1n+index),query_position,Addition.successor_right(position,index)) :
        {Tree.read(Element,run(~Element,~operation,1n+rest,U32.from_nat(position),right,left),U32.from_nat(_)) ==
          operation(Tree.read(Element,left,U32.from_nat(_)),Tree.read(Element,right,U32.from_nat(_))) : Element}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,run(~Element,~operation,rest,_,right,written),query) == operation(Tree.read(Element,left,query),Tree.read(Element,right,query)) : Element}
      Equal.trans(Element,Tree.read(Element,run(~Element,~operation,rest,U32.from_nat(1n+position),right,written),query),operation(Tree.read(Element,written,query),Tree.read(Element,right,query)),
        operation(Tree.read(Element,left,query),Tree.read(Element,right,query)),
        inside(~Element,~operation,rest,1n+position,index,right,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,left),capacity,StorageReadback.write_capacity(~Element,left,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,left,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),selected),
        Equal.cong(Element,Element,element => operation(element,Tree.read(Element,right,query)),Tree.read(Element,written,query),Tree.read(Element,left,query),
          MapReadback.unchanged(~Element,depth,left,U32.from_nat(position),query,value,capacity,capacity_equal,balanced,limit,MapReadback.address_inside(position,capacity,capacity_ok,position_inside),MapReadback.address_inside(query_position,capacity,capacity_ok,query_inside),different)))

# Pointwise binary semantics over a common row layout. The operation keeps
# its exact left/right argument order; arbitrary Element operations are valid.
# Connecting the checked native leaf guard to this model is a separate step.
type Context<-Left: Data,-Right: Data> is Data:
  Context{left: Left,right: Right}

def value(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  context: Context<Left,Right>,index: U32) -> Element:
  Context{left,right} = context
  +index = index
  operation(left_value(left,index),right_value(right,index))

def leaf_content(~Element: Data,~operation: Element -> Element -> Element,left: Buffer.Content<Element>,right: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Buffer.Content{left,+length} = left
  Buffer.Content{right,right_length} = right
  Buffer.Content{run(~Element,~operation,length,0,right,left),length}

def leaf_ready(~Element: Data,~operation: Element -> Element -> Element,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,ready: Buffer.Ready<Element,left>) ->
  Buffer.ContentReady<Element,leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right))>:
  Buffer.Model{+left_tree,+length,+certificate} = left
  Buffer.Model{+right_tree,right_length,right_certificate} = right
  Buffer.Ready{+depth_ok,room} = ready
  +left_model = {Buffer.Model{left_tree,length,certificate} : Buffer.Model<Element>}
  Buffer.ContentReady{Buffer.depth(~Element,left_model),
    balanced(~Element,~operation,length,0,right_tree,left_tree,Buffer.depth(~Element,left_model),Buffer.balanced(~Element,left_model)),depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,run(~Element,~operation,length,0,right_tree,left_tree)),Tree.capacity(Element,left_tree),
      capacity(~Element,~operation,length,0,right_tree,left_tree)) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
    room}

# A combined cell stays at its position and applies the operation to the two
# values of the same pixel.
def combined_cell(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+first: U32,+rows: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,+channel: Nat,+row: Nat,+column: Nat,
  ready: Buffer.Ready<Element,left>,left_cell: BandValues.At(~Element,~Left,~left_value,left_context,geometry,first,rows,Buffer.tree(~Element,left),Buffer.length(~Element,left),channel,row,column),
  right_cell: BandValues.At(~Element,~Right,~right_value,right_context,geometry,first,rows,Buffer.tree(~Element,right),Buffer.length(~Element,right),channel,row,column)) ->
  BandValues.At(~Element,~Context<Left,Right>,~value(~Element,~Left,~Right,~operation,~left_value,~right_value),Context{left_context,right_context},geometry,first,rows,
    Buffer.content_tree(~Element,leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right))),Buffer.content_length(~Element,leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right))),channel,row,column):
  Buffer.Model{+left_tree,+length,+left_certificate} = left
  Buffer.Model{+right_tree,+right_length,right_certificate} = right
  Buffer.Ready{+depth_ok,+room} = ready
  Shape.Both{+selected,left_equal} = left_cell
  Shape.Both{right_selected,right_equal} = right_cell
  +left_model = {Buffer.Model{left_tree,length,left_certificate} : Buffer.Model<Element>}
  +position = BandValues.position(geometry,rows,channel,row,column)
  +index = U32.from_nat(position)
  +pixel = U32.from_nat(BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))
  Shape.Both{selected,
    Equal.trans(Element,Tree.read(Element,run(~Element,~operation,length,0,right_tree,left_tree),index),
      operation(Tree.read(Element,left_tree,index),Tree.read(Element,right_tree,index)),
      operation(left_value(left_context,pixel),right_value(right_context,pixel)),
      inside(~Element,~operation,length,0n,position,right_tree,left_tree,Buffer.depth(~Element,left_model),Tree.capacity(Element,left_tree),
        {==},Buffer.balanced(~Element,left_model),depth_ok,Buffer.capacity_bound(~Element,left_model,depth_ok),room,selected),
      Equal.trans(Element,operation(Tree.read(Element,left_tree,index),Tree.read(Element,right_tree,index)),
        operation(left_value(left_context,pixel),Tree.read(Element,right_tree,index)),operation(left_value(left_context,pixel),right_value(right_context,pixel)),
        Equal.cong(Element,Element,element => operation(element,Tree.read(Element,right_tree,index)),Tree.read(Element,left_tree,index),left_value(left_context,pixel),left_equal),
        Equal.cong(Element,Element,element => operation(left_value(left_context,pixel),element),Tree.read(Element,right_tree,index),right_value(right_context,pixel),right_equal)))}

# What one pair of aligned leaves supplies to every cell.
type LeafCells<-Element: Data,-Left: Data,-Right: Data,-left_value: Left -> U32 -> Element,-right_value: Right -> U32 -> Element> is Data:
  LeafCells{left_context: Left,right_context: Right,geometry: Bands.Geometry,first: U32,rows: U32,left: Buffer.Model<Element>,right: Buffer.Model<Element>,
    ready: Buffer.Ready<Element,left>,left_values: BandValues.Leaf(~Element,~Left,~left_value,left_context,geometry,first,rows,Buffer.tree(~Element,left),Buffer.length(~Element,left)),
    right_values: BandValues.Leaf(~Element,~Right,~right_value,right_context,geometry,first,rows,Buffer.tree(~Element,right),Buffer.length(~Element,right))}

def cells_geometry(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,evidence: LeafCells<Element,Left,Right,left_value,right_value>) -> Bands.Geometry:
  LeafCells{left_context,right_context,geometry,first,rows,left,right,ready,left_values,right_values} = evidence
  geometry

def cells_rows(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,evidence: LeafCells<Element,Left,Right,left_value,right_value>) -> U32:
  LeafCells{left_context,right_context,geometry,first,rows,left,right,ready,left_values,right_values} = evidence
  rows

def Cell(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,+evidence: LeafCells<Element,Left,Right,left_value,right_value>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  LeafCells{left_context,right_context,geometry,first,rows,+left,+right,ready,left_values,right_values} = evidence
  BandValues.At(~Element,~Context<Left,Right>,~value(~Element,~Left,~Right,~operation,~left_value,~right_value),Context{left_context,right_context},geometry,first,rows,
    Buffer.content_tree(~Element,leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right))),Buffer.content_length(~Element,leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right))),channel,row,column)

def leaf_cell(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,+evidence: LeafCells<Element,Left,Right,left_value,right_value>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(cells_geometry(~Element,~Left,~Right,~left_value,~right_value,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(cells_rows(~Element,~Left,~Right,~left_value,~right_value,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(cells_geometry(~Element,~Left,~Right,~left_value,~right_value,evidence))) == True{} : Bool}) ->
  Cell(~Element,~Left,~Right,~operation,~left_value,~right_value,evidence,channel,row,column):
  LeafCells{+left_context,+right_context,+geometry,+first,+rows,+left,+right,ready,left_values,right_values} = evidence
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  combined_cell(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,geometry,first,rows,left,right,channel,row,column,ready,
    BandValues.leaf_cell(~Element,~Left,~left_value,left_context,geometry,first,rows,Buffer.tree(~Element,left),Buffer.length(~Element,left),channel,row,column,left_values,channel_inside,row_inside,column_inside),
    BandValues.leaf_cell(~Element,~Right,~right_value,right_context,geometry,first,rows,Buffer.tree(~Element,right),Buffer.length(~Element,right),channel,row,column,right_values,channel_inside,row_inside,column_inside))

def semantics_leaf(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+first: U32,+rows: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  ready: Buffer.Ready<Element,left>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.RowBand{Buffer.content(~Element,left),first,rows}),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.RowBand{Buffer.content(~Element,right),first,rows})) ->
  BandValues.Values(~Element,~Context<Left,Right>,~value(~Element,~Left,~Right,~operation,~left_value,~right_value),Context{left_context,right_context},geometry,
    Model.RowBand{leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right)),first,rows}):
  Buffer.Model{+tree,+length,+certificate} = left
  Buffer.Model{+right_tree,+right_length,+right_certificate} = right
  BandValues.cells(~LeafCells<Element,Left,Right,left_value,right_value>,~Cell(~Element,~Left,~Right,~operation,~left_value,~right_value),~cells_geometry(~Element,~Left,~Right,~left_value,~right_value),~cells_rows(~Element,~Left,~Right,~left_value,~right_value),~leaf_cell(~Element,~Left,~Right,~operation,~left_value,~right_value),
    LeafCells{left_context,right_context,geometry,first,rows,Buffer.Model{tree,length,certificate},Buffer.Model{right_tree,right_length,right_certificate},ready,left_values,right_values})

# Connect the checked native binary leaf to its pointwise tree model. Its
# premises are the actual owners' Ready, exact local counts, and successful
# descriptor computations. Descriptor producer evidence remains to discharge.
def same_shape(+count: U32) ->
  {Shapes.same(Views.shape(Vector.vector(count)),Views.shape(Vector.vector(count))) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(count,count),True{},
    {%Equal.sym(Bool,U32.is_eq(count,count),Nat.is_eq(U32.to_nat(count),U32.to_nat(count)),Comparison.eq(count,count)) : {_ == True{} : Bool}
      Order.equal_reflexive(U32.to_nat(count)) : {U32.is_eq(count,count) == True{} : Bool}}) : {_ && True{} == True{} : Bool}
  {==}

def guard(~Element: Data,+count: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>) -> Bool:
  +view = Vector.vector(count)
  Shapes.same(Views.shape(view),Views.shape(view)) && Views.is_contiguous(view) &&
    Operations.valid_buffer(Buffer.length(~Element,left),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,left)),certificate(~Element,left)),view) &&
    Operations.valid_buffer(Buffer.length(~Element,right),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,right)),certificate(~Element,right)),view)

def descriptor_counted(+count: U32,descriptor: Vector.Descriptor<count>) -> {Views.numel(Vector.vector(count)) == Some{count} : Maybe<&2,U32>}:
  Vector.Descriptor{counted,contiguous,fits} = descriptor
  counted

def descriptor_contiguous(+count: U32,descriptor: Vector.Descriptor<count>) -> {Views.is_contiguous(Vector.vector(count)) == True{} : Bool}:
  Vector.Descriptor{counted,contiguous,fits} = descriptor
  contiguous

def call_guard(~Element: Data,+count: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  left_exact: {U32.to_nat(count) == Buffer.length(~Element,left) : Nat},right_exact: {U32.to_nat(count) == Buffer.length(~Element,right) : Nat},
  left_ready: Buffer.Ready<Element,left>,right_ready: Buffer.Ready<Element,right>,+descriptor: Vector.Descriptor<count>) -> {guard(~Element,count,left,right) == True{} : Bool}:
  Vector.Descriptor{counted,+contiguous,+fits} = descriptor
  %Equal.sym(Bool,Shapes.same(Views.shape(Vector.vector(count)),Views.shape(Vector.vector(count))),True{},same_shape(count)) :
    {_ && Views.is_contiguous(Vector.vector(count)) &&
      Operations.valid_buffer(Buffer.length(~Element,left),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,left)),certificate(~Element,left)),Vector.vector(count)) &&
      Operations.valid_buffer(Buffer.length(~Element,right),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,right)),certificate(~Element,right)),Vector.vector(count)) == True{} : Bool}
  %Equal.sym(Bool,Views.is_contiguous(Vector.vector(count)),True{},contiguous) :
    {True{} && _ && Operations.valid_buffer(Buffer.length(~Element,left),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,left)),certificate(~Element,left)),Vector.vector(count)) &&
      Operations.valid_buffer(Buffer.length(~Element,right),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,right)),certificate(~Element,right)),Vector.vector(count)) == True{} : Bool}
  %Equal.sym(Bool,Operations.valid_buffer(Buffer.length(~Element,left),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,left)),certificate(~Element,left)),Vector.vector(count)),
    True{},valid(~Element,count,left,left_exact,left_ready,fits)) :
    {True{} && True{} && _ && Operations.valid_buffer(Buffer.length(~Element,right),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,right)),certificate(~Element,right)),Vector.vector(count)) == True{} : Bool}
  %Equal.sym(Bool,Operations.valid_buffer(Buffer.length(~Element,right),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,right)),certificate(~Element,right)),Vector.vector(count)),
    True{},valid(~Element,count,right,right_exact,right_ready,fits)) : {True{} && True{} && True{} && _ == True{} : Bool}
  {==}

def fast(+count: U32,contiguous: {Views.is_contiguous(Vector.vector(count)) == True{} : Bool}) -> {Access.zero_contiguous(Vector.vector(count)) && U32.is_zero(0) == True{} : Bool}:
  %Equal.sym(Bool,Views.is_contiguous(Vector.vector(count)),True{},contiguous) : {_ && True{} == True{} : Bool}
  {==}

def model_contents(~Element: Data,+left_length: Nat,+right_length: Nat,+result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {left_content(~Element,Operations.Accepted{OperationsLaws.combined_pair(~Element,left_length,right_length,result)}) ==
    Buffer.Content{readback_output(~Element,result),left_length} : Buffer.Content<Element>}:
  AccessLaws.ModelResult{reader,+output} = result
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,left_length},
    Tree.reflect(Element,Tree.pack(Element,output)),output,StorageLaws.reflect_packed(Element,output))

def accepted_content(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+left_length: Nat,+right_length: Nat,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,
  contiguous: {Views.is_contiguous(Vector.vector(count)) == True{} : Bool}) ->
  {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
      OperationsLaws.PairCall{True{},Some{count},Vector.vector(count),Vector.vector(count),right_length,left_length},right,left)) ==
    Buffer.Content{run(~Element,~operation,U32.to_nat(count),0,right,left),left_length} : Buffer.Content<Element>}:
  %Equal.sym(Bool,Access.zero_contiguous(Vector.vector(count)) && U32.is_zero(0),True{},fast(count,contiguous)) :
    {left_content(~Element,Operations.Accepted{OperationsLaws.combined_pair(~Element,left_length,right_length,
      AccessLaws.combine_model(~Element,~operation,_,U32.to_nat(count),Vector.vector(count),0,right,left))}) ==
      Buffer.Content{run(~Element,~operation,U32.to_nat(count),0,right,left),left_length} : Buffer.Content<Element>}
  model_contents(~Element,left_length,right_length,AccessLaws.combine_model(~Element,~operation,True{},U32.to_nat(count),Vector.vector(count),0,right,left))

def checked_leaf_model(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  +left_exact: {U32.to_nat(count) == Buffer.length(~Element,left) : Nat},+right_exact: {U32.to_nat(count) == Buffer.length(~Element,right) : Nat},
  left_ready: Buffer.Ready<Element,left>,right_ready: Buffer.Ready<Element,right>,+descriptor: Vector.Descriptor<count>) ->
  {refinement_leaf_model(~Element,~operation,count,left,right) == leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right)) : Buffer.Content<Element>}:
  match left right:
    case Buffer.Model{+left_tree,+left_length,+left_certificate} Buffer.Model{+right_tree,+right_length,+right_certificate}:
      +left_model = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
      +right_model = {Buffer.Model{right_tree,right_length,right_certificate} : Buffer.Model<Element>}
      %Equal.sym(Bool,guard(~Element,count,left_model,right_model),True{},call_guard(~Element,count,left_model,right_model,left_exact,right_exact,left_ready,right_ready,descriptor)) :
        {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
          OperationsLaws.PairCall{_,Views.numel(Vector.vector(count)),Vector.vector(count),Vector.vector(count),right_length,left_length},
          Tree.reflect(Element,Tree.pack(Element,right_tree)),Tree.reflect(Element,Tree.pack(Element,left_tree)))) ==
          Buffer.Content{run(~Element,~operation,left_length,0,right_tree,left_tree),left_length} : Buffer.Content<Element>}
      %Equal.sym(Maybe<&2,U32>,Views.numel(Vector.vector(count)),Some{count},descriptor_counted(count,descriptor)) :
        {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
          OperationsLaws.PairCall{True{},_,Vector.vector(count),Vector.vector(count),right_length,left_length},Tree.reflect(Element,Tree.pack(Element,right_tree)),Tree.reflect(Element,Tree.pack(Element,left_tree)))) ==
          Buffer.Content{run(~Element,~operation,left_length,0,right_tree,left_tree),left_length} : Buffer.Content<Element>}
      %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,right_tree)),right_tree,StorageLaws.reflect_packed(Element,right_tree)) :
        {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
          OperationsLaws.PairCall{True{},Some{count},Vector.vector(count),Vector.vector(count),right_length,left_length},_,Tree.reflect(Element,Tree.pack(Element,left_tree)))) ==
          Buffer.Content{run(~Element,~operation,left_length,0,right_tree,left_tree),left_length} : Buffer.Content<Element>}
      %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,left_tree)),left_tree,StorageLaws.reflect_packed(Element,left_tree)) :
        {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
          OperationsLaws.PairCall{True{},Some{count},Vector.vector(count),Vector.vector(count),right_length,left_length},right_tree,_)) ==
          Buffer.Content{run(~Element,~operation,left_length,0,right_tree,left_tree),left_length} : Buffer.Content<Element>}
      %left_exact :
        {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
          OperationsLaws.PairCall{True{},Some{count},Vector.vector(count),Vector.vector(count),right_length,left_length},right_tree,left_tree)) ==
          Buffer.Content{run(~Element,~operation,_,0,right_tree,left_tree),left_length} : Buffer.Content<Element>}
      accepted_content(~Element,~operation,count,left_length,right_length,left_tree,right_tree,descriptor_contiguous(count,descriptor))
