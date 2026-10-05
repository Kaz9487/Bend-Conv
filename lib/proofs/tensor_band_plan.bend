# The row planner: coverage of its interval, bounds of every leaf, and monotone
# endpoint maps between input and output rows.
import Base
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_band_coordinates.bend as BandCoordinates
import ./row_major_index.bend as RowMajor
import ./word_encoding.bend as WordEncoding
import ../tensor_band.bend as Bands
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_division_order.bend as DivisionOrder
import ./storage_shape.bend as Shape
import ./binary_radix.bend as Radix
import ./nat_ceiling.bend as Ceiling
import ./nat_extrema.bend as Extrema
import ./strided_range_bounds.bend as Strided
import ./u32_division.bend as U32Division

# The real U32 row splitter covers its original interval exactly. Forks join
# consecutive intervals, so their leaves neither omit nor duplicate a row.
law Coverage:
  for plan: Bands.Plan
  for first: Nat
  for rows: Nat
  Data

type LeafCoverage<-native_first: U32,-native_rows: U32,-first: Nat,-rows: Nat> is Data:
  LeafCoverage{first_equal: {U32.to_nat(native_first) == first : Nat},rows_equal: {U32.to_nat(native_rows) == rows : Nat}}

type ForkCoverage<-left: Bands.Plan,-right: Bands.Plan,-first: Nat,-rows: Nat> is Data:
  ForkCoverage{left_rows: Nat,right_rows: Nat,rows_equal: {rows == Nat.add(left_rows,right_rows) : Nat},
    left_coverage: Coverage(left,first,left_rows),right_coverage: Coverage(right,Nat.add(first,left_rows),right_rows)}

def Coverage(plan,first,rows):
  match plan:
    case Bands.Rows{native_first,native_rows}: LeafCoverage<native_first,native_rows,first,rows>
    case Bands.Split{left,right}: ForkCoverage<left,right,first,rows>

law Located:
  for plan: Bands.Plan
  for row: Nat
  Data

type LeafLocation<-first: U32,-rows: U32,-row: Nat> is Data:
  LeafLocation{local: Nat,inside: {Nat.is_lt(local,U32.to_nat(rows)) == True{} : Bool},address: {row == Nat.add(U32.to_nat(first),local) : Nat}}

def Located(plan,row):
  match plan:
    case Bands.Rows{first,rows}: LeafLocation<first,rows,row>
    case Bands.Split{left,right}: Either<&2,&2,Located(left,row),Located(right,row)>

def locate_branch(+left: Bands.Plan,+right: Bands.Plan,+first: Nat,+left_rows: Nat,+right_rows: Nat,+position: Nat,choose_left: Bool,
  +choice: {Nat.is_lt(position,left_rows) == choose_left : Bool},inside: {Nat.is_lt(position,Nat.add(left_rows,right_rows)) == True{} : Bool},
  left_case: {Nat.is_lt(position,left_rows) == True{} : Bool} -> Located(left,Nat.add(first,position)),
  right_case: {Nat.is_lt(Nat.sub(position,left_rows),right_rows) == True{} : Bool} -> Located(right,Nat.add(Nat.add(first,left_rows),Nat.sub(position,left_rows)))) ->
  Located(Bands.Split{left,right},Nat.add(first,position)):
  match choose_left:
    case True{}: Inl{left_case(choice)}
    case False{}:
      +above = Order.ge_true(position,left_rows,Order.comparison_not_less(Nat.cmp(position,left_rows),choice))
      +restored = Equal.trans(Nat,Nat.add(Nat.add(first,left_rows),Nat.sub(position,left_rows)),Nat.add(first,Nat.add(left_rows,Nat.sub(position,left_rows))),Nat.add(first,position),
        Addition.associative(first,left_rows,Nat.sub(position,left_rows)),Equal.cong(Nat,Nat,value => Nat.add(first,value),Nat.add(left_rows,Nat.sub(position,left_rows)),position,Order.restore_shift(left_rows,position,above)))
      Inr{
        %restored : Located(right,_)
        right_case(Order.sub_lt(position,left_rows,right_rows,above,inside))}

def locate(plan: Bands.Plan,+first: Nat,+rows: Nat,+position: Nat,coverage: Coverage(plan,first,rows),inside: {Nat.is_lt(position,rows) == True{} : Bool}) ->
  Located(plan,Nat.add(first,position)):
  match plan:
    case Bands.Rows{+native_first,+native_rows}:
      LeafCoverage{first_equal,rows_equal} = coverage
      LeafLocation{position,
        %Equal.sym(Nat,U32.to_nat(native_rows),rows,rows_equal) : {Nat.is_lt(position,_) == True{} : Bool} inside,
        Equal.cong(Nat,Nat,start => Nat.add(start,position),first,U32.to_nat(native_first),Equal.sym(Nat,U32.to_nat(native_first),first,first_equal))}
    case Bands.Split{+left,+right}:
      ForkCoverage{+left_rows,+right_rows,rows_equal,left_coverage,right_coverage} = coverage
      locate_branch(left,right,first,left_rows,right_rows,position,Nat.is_lt(position,left_rows),{==},
        %rows_equal : {Nat.is_lt(position,_) == True{} : Bool} inside,
        in_left => locate(left,first,left_rows,position,left_coverage,in_left),
        in_right => locate(right,Nat.add(first,left_rows),right_rows,Nat.sub(position,left_rows),right_coverage,in_right))

def half_bound(+rows: U32) -> {Nat.is_le(U32.to_nat((rows / 2 : U32)),U32.to_nat(rows)) == True{} : Bool}:
  +quotient = Nat.div(U32.to_nat(rows),2n)
  +remainder = Nat.mod(U32.to_nat(rows),2n)
  %Equal.sym(Nat,U32.to_nat((rows / 2 : U32)),quotient,U32Division.divide(rows,2,2147483648,{==},{==},{==})) : {Nat.is_le(_,U32.to_nat(rows)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(rows),Nat.add(Nat.mul(quotient,2n),remainder),Radix.division_identity(rows,2n,{==})) : {Nat.is_le(quotient,_) == True{} : Bool}
  Order.transitive(quotient,Nat.mul(quotient,2n),Nat.add(Nat.mul(quotient,2n),remainder),RowMajor.product_covers(quotient,2n,{==}),Order.le_add(Nat.mul(quotient,2n),remainder))

def rows_bound(+first: U32,+rows: U32,+capacity: U32,room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(rows),U32.to_nat(capacity)) == True{} : Bool}:
  Order.transitive(U32.to_nat(rows),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(capacity),
    %Order.commutative(U32.to_nat(rows),U32.to_nat(first)) : {Nat.is_le(U32.to_nat(rows),_) == True{} : Bool}
    Order.le_add(U32.to_nat(rows),U32.to_nat(first)),room)

def refinement_row_plan(depth: Nat,+split: Bool,+first: U32,+rows: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(capacity)) == True{} : Bool}) ->
  Coverage(Bands.row_plan(depth,split,first,rows),U32.to_nat(first),U32.to_nat(rows)):
  match depth split:
    case 0n split: LeafCoverage{{==},{==}}
    case 1n+rest False{}: LeafCoverage{{==},{==}}
    case 1n+ +rest True{}:
      +half = (rows / 2 : U32)
      +remainder = (rows - half : U32)
      +next = (first + half : U32)
      +half_inside = half_bound(rows)
      +left_room = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(half)),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(capacity),
        Order.offset_le(U32.to_nat(first),U32.to_nat(half),U32.to_nat(rows),half_inside),room)
      +next_exact = Arithmetic.add(first,half,capacity,capacity_ok,left_room)
      +remainder_exact = Arithmetic.subtract(rows,half,capacity,capacity_ok,half_inside,
        Order.transitive(Nat.sub(U32.to_nat(rows),U32.to_nat(half)),U32.to_nat(rows),U32.to_nat(capacity),Order.sub_le(U32.to_nat(rows),U32.to_nat(half)),rows_bound(first,rows,capacity,room)))
      +joined = {
        %Equal.sym(Nat,U32.to_nat(remainder),Nat.sub(U32.to_nat(rows),U32.to_nat(half)),remainder_exact) : {Nat.add(U32.to_nat(half),_) == U32.to_nat(rows) : Nat}
        Order.restore_shift(U32.to_nat(half),U32.to_nat(rows),half_inside) : {Nat.add(U32.to_nat(half),U32.to_nat(remainder)) == U32.to_nat(rows) : Nat}}
      +right_room = {
        %Equal.sym(Nat,U32.to_nat(next),Nat.add(U32.to_nat(first),U32.to_nat(half)),next_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(remainder)),U32.to_nat(capacity)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(half)),U32.to_nat(remainder)),Nat.add(U32.to_nat(first),Nat.add(U32.to_nat(half),U32.to_nat(remainder))),
          Addition.associative(U32.to_nat(first),U32.to_nat(half),U32.to_nat(remainder))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(U32.to_nat(half),U32.to_nat(remainder)),U32.to_nat(rows),joined) : {Nat.is_le(Nat.add(U32.to_nat(first),_),U32.to_nat(capacity)) == True{} : Bool}
        room : {Nat.is_le(Nat.add(U32.to_nat(next),U32.to_nat(remainder)),U32.to_nat(capacity)) == True{} : Bool}}
      ForkCoverage{U32.to_nat(half),U32.to_nat(remainder),
        Equal.sym(Nat,Nat.add(U32.to_nat(half),U32.to_nat(remainder)),U32.to_nat(rows),joined),
        refinement_row_plan(rest,U32.is_lt(1,half),first,half,capacity,capacity_ok,left_room),
        %next_exact : Coverage(Bands.row_plan(rest,U32.is_lt(1,remainder),next,remainder),_,U32.to_nat(remainder))
        refinement_row_plan(rest,U32.is_lt(1,remainder),next,remainder,capacity,capacity_ok,right_room)}

def image_first(count: Nat,image: U32,height: U32) -> Nat:
  match count:
    case 0n: 0n
    case 1n+rest: Nat.mul(U32.to_nat(image),U32.to_nat(height))

def image_room(+image: U32,+height: U32,+rest: Nat,+capacity: U32,
  room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(image),1n+rest),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}:
  Strided.end_inside(0n,0n,U32.to_nat(height),U32.to_nat(height),Nat.add(U32.to_nat(image),1n+rest),U32.to_nat(image),U32.to_nat(capacity),
    Order.room_index(U32.to_nat(image),rest,Nat.add(U32.to_nat(image),1n+rest),Order.reflexive(Nat.add(U32.to_nat(image),1n+rest))),Order.reflexive(U32.to_nat(height)),room)

def image_start_exact(+image: U32,+height: U32,+rest: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(image),1n+rest),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((image * height : U32)) == Nat.mul(U32.to_nat(image),U32.to_nat(height)) : Nat}:
  Arithmetic.multiply(image,height,capacity,capacity_ok,Order.transitive(Nat.mul(U32.to_nat(image),U32.to_nat(height)),
    Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(height)),U32.to_nat(capacity),Order.le_add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(height)),image_room(image,height,rest,capacity,room)))

def refinement_image_plan(count: Nat,+image: U32,+depth: Nat,+height: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +images_fit: {Nat.is_le(Nat.add(U32.to_nat(image),count),U32.to_nat(capacity)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(image),count),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}) ->
  Coverage(Bands.image_plan(count,image,depth,height),image_first(count,image,height),Nat.mul(count,U32.to_nat(height))):
  match count:
    case 0n: LeafCoverage{{==},{==}}
    case 1n:
      +start_exact = image_start_exact(image,height,0n,capacity,capacity_ok,room)
      %Equal.sym(Nat,Nat.mul(1n,U32.to_nat(height)),U32.to_nat(height),Addition.zero_right(U32.to_nat(height))) :
        Coverage(Bands.row_plan(depth,U32.is_lt(1,height),(image * height : U32),height),Nat.mul(U32.to_nat(image),U32.to_nat(height)),_)
      %start_exact : Coverage(Bands.row_plan(depth,U32.is_lt(1,height),(image * height : U32),height),_,U32.to_nat(height))
      refinement_row_plan(depth,U32.is_lt(1,height),(image * height : U32),height,capacity,capacity_ok,
        %Equal.sym(Nat,U32.to_nat((image * height : U32)),Nat.mul(U32.to_nat(image),U32.to_nat(height)),start_exact) :
          {Nat.is_le(Nat.add(_,U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}
        image_room(image,height,0n,capacity,room))
    case 2n+ +rest:
      +start_exact = image_start_exact(image,height,1n+rest,capacity,capacity_ok,room)
      +next = (image + 1 : U32)
      +next_exact = Equal.trans(Nat,U32.to_nat(next),Nat.add(U32.to_nat(image),1n),1n+U32.to_nat(image),
        Arithmetic.add(image,1,capacity,capacity_ok,
          %Equal.sym(Nat,Nat.add(U32.to_nat(image),1n),1n+U32.to_nat(image),Addition.successor_width(U32.to_nat(image))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
          Order.strict_follow(U32.to_nat(image),U32.to_nat(capacity),Order.room_index(U32.to_nat(image),1n+rest,U32.to_nat(capacity),images_fit))),Addition.successor_width(U32.to_nat(image)))
      +right_first = {
        %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(image),next_exact) : {Nat.mul(_,U32.to_nat(height)) == Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(height)) : Nat}
        Order.commutative(U32.to_nat(height),Nat.mul(U32.to_nat(image),U32.to_nat(height))) :
        {Nat.mul(U32.to_nat(next),U32.to_nat(height)) == Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(height)) : Nat}}
      ForkCoverage{U32.to_nat(height),Nat.mul(1n+rest,U32.to_nat(height)),{==},
        %start_exact : Coverage(Bands.row_plan(depth,U32.is_lt(1,height),(image * height : U32),height),_,U32.to_nat(height))
        refinement_row_plan(depth,U32.is_lt(1,height),(image * height : U32),height,capacity,capacity_ok,
          %Equal.sym(Nat,U32.to_nat((image * height : U32)),Nat.mul(U32.to_nat(image),U32.to_nat(height)),start_exact) :
            {Nat.is_le(Nat.add(_,U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}
          image_room(image,height,1n+rest,capacity,room)),
        %right_first : Coverage(Bands.image_plan(1n+rest,next,depth,height),_,Nat.mul(1n+rest,U32.to_nat(height)))
        refinement_image_plan(1n+rest,next,depth,height,capacity,capacity_ok,
          %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(image),next_exact) : {Nat.is_le(Nat.add(_,1n+rest),U32.to_nat(capacity)) == True{} : Bool}
          Order.advance_room(U32.to_nat(image),1n+rest,U32.to_nat(capacity),images_fit),
          %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(image),next_exact) : {Nat.is_le(Nat.mul(Nat.add(_,1n+rest),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}
          %Addition.successor_right(U32.to_nat(image),1n+rest) : {Nat.is_le(Nat.mul(_,U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}
          room)}

# Coverage of a subpartition also bounds every leaf. Applying this once per
# actual image plan proves that the U32 row splitter never crosses an image.
type LeafWithin<-first: U32,-rows: U32,-lower: Nat,-upper: Nat> is Data:
  LeafWithin{lower: {Nat.is_le(lower,U32.to_nat(first)) == True{} : Bool},
    upper: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),upper) == True{} : Bool}}

def Within(plan: Bands.Plan,+lower: Nat,+upper: Nat) -> Data:
  match plan:
    case Bands.Rows{first,rows}: LeafWithin<first,rows,lower,upper>
    case Bands.Split{left,right}: Shape.Both<Within(left,lower,upper),Within(right,lower,upper)>

def Nonempty(plan: Bands.Plan) -> Data:
  match plan:
    case Bands.Rows{first,rows}: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}
    case Bands.Split{left,right}: Shape.Both<Nonempty(left),Nonempty(right)>

def covered(plan: Bands.Plan,+first: Nat,+rows: Nat,+lower: Nat,+upper: Nat,coverage: Coverage(plan,first,rows),
  +lower_bound: {Nat.is_le(lower,first) == True{} : Bool},+upper_bound: {Nat.is_le(Nat.add(first,rows),upper) == True{} : Bool}) -> Within(plan,lower,upper):
  match plan:
    case Bands.Rows{+native_first,+native_rows}:
      LeafCoverage{+first_equal,rows_equal} = coverage
      LeafWithin{
        %Equal.sym(Nat,U32.to_nat(native_first),first,first_equal) : {Nat.is_le(lower,_) == True{} : Bool} lower_bound,
        %Equal.sym(Nat,U32.to_nat(native_first),first,first_equal) : {Nat.is_le(Nat.add(_,U32.to_nat(native_rows)),upper) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(native_rows),rows,rows_equal) : {Nat.is_le(Nat.add(first,_),upper) == True{} : Bool} upper_bound}
    case Bands.Split{+left,+right}:
      ForkCoverage{+left_rows,+right_rows,+rows_equal,left_coverage,right_coverage} = coverage
      Shape.Both{
        covered(left,first,left_rows,lower,upper,left_coverage,lower_bound,
          Order.transitive(Nat.add(first,left_rows),Nat.add(first,Nat.add(left_rows,right_rows)),upper,
            Order.offset_le(first,left_rows,Nat.add(left_rows,right_rows),Order.le_add(left_rows,right_rows)),
            %rows_equal : {Nat.is_le(Nat.add(first,_),upper) == True{} : Bool} upper_bound)),
        covered(right,Nat.add(first,left_rows),right_rows,lower,upper,right_coverage,
          Order.transitive(lower,first,Nat.add(first,left_rows),lower_bound,Order.le_add(first,left_rows)),
          %Equal.sym(Nat,Nat.add(Nat.add(first,left_rows),right_rows),Nat.add(first,Nat.add(left_rows,right_rows)),Addition.associative(first,left_rows,right_rows)) :
            {Nat.is_le(_,upper) == True{} : Bool}
          %rows_equal : {Nat.is_le(Nat.add(first,_),upper) == True{} : Bool} upper_bound)}

type ImageSpan<-first: U32,-rows: U32,-images: Nat,-height: Nat> is Data:
  EmptySpan{empty: {U32.to_nat(rows) == 0n : Nat}}
  ImageSpan{image: Nat,inside: {Nat.is_lt(image,images) == True{} : Bool},
    bounds: LeafWithin<first,rows,Nat.mul(image,height),Nat.add(Nat.mul(image,height),height)>}

def ImageLocal(plan: Bands.Plan,+images: Nat,+height: Nat) -> Data:
  match plan:
    case Bands.Rows{first,rows}: ImageSpan<first,rows,images,height>
    case Bands.Split{left,right}: Shape.Both<ImageLocal(left,images,height),ImageLocal(right,images,height)>

def within_image(plan: Bands.Plan,+images: Nat,+height: Nat,+image: Nat,+inside: {Nat.is_lt(image,images) == True{} : Bool},
  bounds: Within(plan,Nat.mul(image,height),Nat.add(Nat.mul(image,height),height))) -> ImageLocal(plan,images,height):
  match plan:
    case Bands.Rows{first,rows}: ImageSpan{image,inside,bounds}
    case Bands.Split{+left,+right}:
      Shape.Both{left_bounds,right_bounds} = bounds
      Shape.Both{within_image(left,images,height,image,inside,left_bounds),within_image(right,images,height,image,inside,right_bounds)}

def offset_span(+first: U32,+rows: U32,+image: Nat,+height: Nat,bounds: LeafWithin<first,rows,Nat.mul(image,height),Nat.add(Nat.mul(image,height),height)>) ->
  {Nat.is_le(Nat.add(Nat.sub(U32.to_nat(first),Nat.mul(image,height)),U32.to_nat(rows)),height) == True{} : Bool}:
  LeafWithin{+lower,upper} = bounds
  +base = Nat.mul(image,height)
  +offset = Nat.sub(U32.to_nat(first),base)
  Order.cancel_offset_le(base,Nat.add(offset,U32.to_nat(rows)),height,
    %Addition.associative(base,offset,U32.to_nat(rows)) : {Nat.is_le(_,Nat.add(base,height)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(base,offset),U32.to_nat(first),Order.restore_shift(base,U32.to_nat(first),lower)) :
      {Nat.is_le(Nat.add(_,U32.to_nat(rows)),Nat.add(base,height)) == True{} : Bool} upper)

def local_span(+first: U32,+rows: U32,+image: Nat,+height: Nat,+height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+bounds: LeafWithin<first,rows,Nat.mul(image,height),Nat.add(Nat.mul(image,height),height)>) ->
  {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(first),height),U32.to_nat(rows)),height) == True{} : Bool}:
  LeafWithin{+lower,upper} = bounds
  +base = Nat.mul(image,height)
  +offset = Nat.sub(U32.to_nat(first),base)
  +span = offset_span(first,rows,image,height,bounds)
  +inside = Bounds.nat_trans(offset,Nat.add(offset,U32.to_nat(rows)),height,
    %Addition.zero_right(offset) : {Nat.is_lt(_,Nat.add(offset,U32.to_nat(rows))) == True{} : Bool} Order.offset_lt(offset,0n,U32.to_nat(rows),rows_positive),span)
  +remainder = {
    %Order.restore_shift(base,U32.to_nat(first),lower) :
      {Nat.mod(_,height) == offset : Nat}
    BandCoordinates.remainder(image,offset,height,height_positive,inside) : {Nat.mod(U32.to_nat(first),height) == offset : Nat}}
  %Equal.sym(Nat,Nat.mod(U32.to_nat(first),height),offset,remainder) : {Nat.is_le(Nat.add(_,U32.to_nat(rows)),height) == True{} : Bool} span

def total_rows(+first: U32,+rows: U32,+image: Nat,+images: Nat,+height: Nat,inside: {Nat.is_lt(image,images) == True{} : Bool},
  bounds: LeafWithin<first,rows,Nat.mul(image,height),Nat.add(Nat.mul(image,height),height)>) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(images,height)) == True{} : Bool}:
  LeafWithin{lower,upper} = bounds
  Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.add(Nat.mul(image,height),height),Nat.mul(images,height),upper,
    %Order.commutative(height,Nat.mul(image,height)) : {Nat.is_le(_,Nat.mul(images,height)) == True{} : Bool}
    RowMajor.mul_le(1n+image,images,height,Order.strict_follow(image,images,inside)))

def image_prefix_room(+image: U32,+height: U32,+images: Nat,+capacity: U32,
  inside: {Nat.is_lt(U32.to_nat(image),images) == True{} : Bool},room: {Nat.is_le(Nat.mul(images,U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(image),1n),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}:
  Order.transitive(Nat.mul(Nat.add(U32.to_nat(image),1n),U32.to_nat(height)),Nat.mul(images,U32.to_nat(height)),U32.to_nat(capacity),
    RowMajor.mul_le(Nat.add(U32.to_nat(image),1n),images,U32.to_nat(height),
      %Equal.sym(Nat,Nat.add(U32.to_nat(image),1n),1n+U32.to_nat(image),Addition.successor_width(U32.to_nat(image))) : {Nat.is_le(_,images) == True{} : Bool}
      Order.strict_follow(U32.to_nat(image),images,inside)),room)

def bounds_row_plan(+depth: Nat,+split: Bool,+image: U32,+height: U32,+images: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(image),images) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(images,U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}) ->
  ImageLocal(Bands.row_plan(depth,split,(image * height : U32),height),images,U32.to_nat(height)):
  +prefix_room = image_prefix_room(image,height,images,capacity,inside,room)
  +start = (image * height : U32)
  +start_exact = image_start_exact(image,height,0n,capacity,capacity_ok,prefix_room)
  within_image(Bands.row_plan(depth,split,start,height),images,U32.to_nat(height),U32.to_nat(image),inside,
    %start_exact : Within(Bands.row_plan(depth,split,start,height),_,Nat.add(_,U32.to_nat(height)))
    covered(Bands.row_plan(depth,split,start,height),U32.to_nat(start),U32.to_nat(height),U32.to_nat(start),Nat.add(U32.to_nat(start),U32.to_nat(height)),
      refinement_row_plan(depth,split,start,height,capacity,capacity_ok,
        %Equal.sym(Nat,U32.to_nat(start),Nat.mul(U32.to_nat(image),U32.to_nat(height)),start_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}
        image_room(image,height,0n,capacity,prefix_room)),Order.reflexive(U32.to_nat(start)),Order.reflexive(Nat.add(U32.to_nat(start),U32.to_nat(height)))))

def bounds_image_plan(count: Nat,+image: U32,+depth: Nat,+height: U32,+images: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+remaining_fit: {Nat.is_le(Nat.add(U32.to_nat(image),count),images) == True{} : Bool},
  +images_fit: {Nat.is_le(images,U32.to_nat(capacity)) == True{} : Bool},+rows_fit: {Nat.is_le(Nat.mul(images,U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}) ->
  ImageLocal(Bands.image_plan(count,image,depth,height),images,U32.to_nat(height)):
  match count:
    case 0n: EmptySpan{{==}}
    case 1n: bounds_row_plan(depth,U32.is_lt(1,height),image,height,images,capacity,capacity_ok,Order.room_index(U32.to_nat(image),0n,images,remaining_fit),rows_fit)
    case 2n+ +rest:
      +inside = Order.room_index(U32.to_nat(image),1n+rest,images,remaining_fit)
      +next = (image + 1 : U32)
      +next_exact = Equal.trans(Nat,U32.to_nat(next),Nat.add(U32.to_nat(image),1n),1n+U32.to_nat(image),
        Arithmetic.add(image,1,capacity,capacity_ok,
          %Equal.sym(Nat,Nat.add(U32.to_nat(image),1n),1n+U32.to_nat(image),Addition.successor_width(U32.to_nat(image))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
          Order.transitive(1n+U32.to_nat(image),images,U32.to_nat(capacity),Order.strict_follow(U32.to_nat(image),images,inside),images_fit)),Addition.successor_width(U32.to_nat(image)))
      Shape.Both{bounds_row_plan(depth,U32.is_lt(1,height),image,height,images,capacity,capacity_ok,inside,rows_fit),
        bounds_image_plan(1n+rest,next,depth,height,images,capacity,capacity_ok,
          %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(image),next_exact) : {Nat.is_le(Nat.add(_,1n+rest),images) == True{} : Bool}
          Order.advance_room(U32.to_nat(image),1n+rest,images,remaining_fit),images_fit,rows_fit)}

def first_zero(count: Nat,+height: U32) -> {image_first(count,0,height) == 0n : Nat}:
  match count:
    case 0n: {==}
    case 1n+rest: {==}

def image_coverage(+images: U32,+height: U32,+depth: Nat,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  images_fit: {Nat.is_le(U32.to_nat(images),U32.to_nat(capacity)) == True{} : Bool},rows_fit: {Nat.is_le(Nat.mul(U32.to_nat(images),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}) ->
  Coverage(Bands.image_plan(U32.to_nat(images),0,depth,height),0n,Nat.mul(U32.to_nat(images),U32.to_nat(height))):
  %first_zero(U32.to_nat(images),height) : Coverage(Bands.image_plan(U32.to_nat(images),0,depth,height),_,Nat.mul(U32.to_nat(images),U32.to_nat(height)))
  refinement_image_plan(U32.to_nat(images),0,depth,height,capacity,capacity_ok,images_fit,rows_fit)

def image_within(+images: U32,+height: U32,+depth: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  images_fit: {Nat.is_le(U32.to_nat(images),U32.to_nat(capacity)) == True{} : Bool},rows_fit: {Nat.is_le(Nat.mul(U32.to_nat(images),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool}) ->
  Within(Bands.image_plan(U32.to_nat(images),0,depth,height),0n,Nat.mul(U32.to_nat(images),U32.to_nat(height))):
  covered(Bands.image_plan(U32.to_nat(images),0,depth,height),0n,Nat.mul(U32.to_nat(images),U32.to_nat(height)),0n,Nat.mul(U32.to_nat(images),U32.to_nat(height)),
    image_coverage(images,height,depth,capacity,capacity_ok,images_fit,rows_fit),{==},Order.reflexive(Nat.mul(U32.to_nat(images),U32.to_nat(height))))

# Image-relative ceil boundaries extend to a monotone global row map. At an
# image endpoint the next image contributes exactly its output-row base.
def local(output_height: Nat,row: Nat,stride: Nat) -> Nat:
  Nat.min(output_height,Ceiling.ceiling(row,stride))

def output_boundaries_boundary(+height: Nat,+output_height: Nat,+stride: Nat,+row: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.div(row,height),output_height),local(output_height,Nat.mod(row,height),stride))

type Mapping is Data:
  Mapping{height: Nat,output_height: Nat,stride: Nat}

def mapping(context: Mapping,row: Nat) -> Nat:
  Mapping{height,output_height,stride} = context
  output_boundaries_boundary(height,output_height,stride,row)

def same_image(+height: Nat,+output_height: Nat,+stride: Nat,+first: Nat,+end: Nat,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  ordered: {Nat.is_le(first,end) == True{} : Bool},+same: {Nat.div(first,height) == Nat.div(end,height) : Nat}) ->
  {Nat.is_le(output_boundaries_boundary(height,output_height,stride,first),output_boundaries_boundary(height,output_height,stride,end)) == True{} : Bool}:
  +image = Nat.div(first,height)
  +first_row = Nat.mod(first,height)
  +end_row = Nat.mod(end,height)
  +right_exact = Equal.trans(Nat,end,Nat.add(Nat.mul(Nat.div(end,height),height),end_row),Nat.add(Nat.mul(image,height),end_row),
    DivisionOrder.quotient_exact(end,height,height_positive),
    Equal.cong(Nat,Nat,n => Nat.add(Nat.mul(n,height),end_row),Nat.div(end,height),image,Equal.sym(Nat,image,Nat.div(end,height),same)))
  +row_order = Order.cancel_offset_le(Nat.mul(image,height),first_row,end_row,
    %DivisionOrder.quotient_exact(first,height,height_positive) : {Nat.is_le(_,Nat.add(Nat.mul(image,height),end_row)) == True{} : Bool}
    %right_exact : {Nat.is_le(first,_) == True{} : Bool}
    ordered)
  %same :
    {Nat.is_le(Nat.add(Nat.mul(image,output_height),local(output_height,first_row,stride)),Nat.add(Nat.mul(_,output_height),local(output_height,end_row,stride))) == True{} : Bool}
  Order.offset_le(Nat.mul(image,output_height),local(output_height,first_row,stride),local(output_height,end_row,stride),
    Extrema.minimum_monotone(Ceiling.ceiling(first_row,stride),Ceiling.ceiling(end_row,stride),output_height,Ceiling.monotone(first_row,end_row,stride,stride_positive,row_order)))

def later_image(+height: Nat,+output_height: Nat,+stride: Nat,+first: Nat,+end: Nat,
  different: {Nat.is_lt(Nat.div(first,height),Nat.div(end,height)) == True{} : Bool}) ->
  {Nat.is_le(output_boundaries_boundary(height,output_height,stride,first),output_boundaries_boundary(height,output_height,stride,end)) == True{} : Bool}:
  +image = Nat.div(first,height)
  +next_base = Nat.mul(1n+image,output_height)
  +end_base = Nat.mul(Nat.div(end,height),output_height)
  Order.transitive(output_boundaries_boundary(height,output_height,stride,first),next_base,output_boundaries_boundary(height,output_height,stride,end),
    %Order.commutative(Nat.mul(image,output_height),output_height) :
      {Nat.is_le(output_boundaries_boundary(height,output_height,stride,first),_) == True{} : Bool}
    Order.offset_le(Nat.mul(image,output_height),local(output_height,Nat.mod(first,height),stride),output_height,
      Extrema.minimum_left(output_height,Ceiling.ceiling(Nat.mod(first,height),stride))),
    Order.transitive(next_base,end_base,output_boundaries_boundary(height,output_height,stride,end),
      RowMajor.mul_le(1n+image,Nat.div(end,height),output_height,Order.strict_follow(image,Nat.div(end,height),different)),
      Order.le_add(end_base,local(output_height,Nat.mod(end,height),stride))))

def compared(+height: Nat,+output_height: Nat,+stride: Nat,+first: Nat,+end: Nat,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  ordered: {Nat.is_le(first,end) == True{} : Bool},comparison: Order.UpperBound<Nat.div(first,height),Nat.div(end,height)>) ->
  {Nat.is_le(output_boundaries_boundary(height,output_height,stride,first),output_boundaries_boundary(height,output_height,stride,end)) == True{} : Bool}:
  match comparison:
    case Order.Before{different}: later_image(height,output_height,stride,first,end,different)
    case Order.At{same}: same_image(height,output_height,stride,first,end,height_positive,stride_positive,ordered,same)

def monotone(+height: Nat,+output_height: Nat,+stride: Nat,+first: Nat,+end: Nat,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},+ordered: {Nat.is_le(first,end) == True{} : Bool}) ->
  {Nat.is_le(output_boundaries_boundary(height,output_height,stride,first),output_boundaries_boundary(height,output_height,stride,end)) == True{} : Bool}:
  compared(height,output_height,stride,first,end,height_positive,stride_positive,ordered,
    Order.upper_bound_cases(Nat.div(first,height),Nat.div(end,height),DivisionOrder.quotient_le(first,end,height,height_positive,ordered)))

def zero_bound(+value: Nat,bounded: {Nat.is_le(value,0n) == True{} : Bool}) -> {value == 0n : Nat}:
  match value:
    case 0n: {==}
    case 1n+rest: Empty.absurd({1n+rest == 0n : Nat},WordEncoding.false_true(bounded))

def minimum_zero(+value: Nat) -> {Nat.min(value,0n) == 0n : Nat}:
  match value:
    case 0n: {==}
    case 1n+rest: {==}

def local_zero(+output_height: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) -> {local(output_height,0n,stride) == 0n : Nat}:
  %Equal.sym(Nat,Ceiling.ceiling(0n,stride),0n,zero_bound(Ceiling.ceiling(0n,stride),Ceiling.minimal(0n,stride,0n,positive,{==}))) : {Nat.min(output_height,_) == 0n : Nat}
  minimum_zero(output_height)

def image_end(+images: Nat,+height: Nat,+output_height: Nat,+stride: Nat,+height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {output_boundaries_boundary(height,output_height,stride,Nat.mul(images,height)) == Nat.mul(images,output_height) : Nat}:
  +quotient = {
    %Addition.zero_right(Nat.mul(images,height)) : {Nat.div(_,height) == images : Nat}
    BandCoordinates.quotient(images,0n,height,height_positive,height_positive) : {Nat.div(Nat.mul(images,height),height) == images : Nat}}
  +remainder = {
    %Addition.zero_right(Nat.mul(images,height)) : {Nat.mod(_,height) == 0n : Nat}
    BandCoordinates.remainder(images,0n,height,height_positive,height_positive) : {Nat.mod(Nat.mul(images,height),height) == 0n : Nat}}
  %Equal.sym(Nat,Nat.div(Nat.mul(images,height),height),images,quotient) :
    {Nat.add(Nat.mul(_,output_height),local(output_height,Nat.mod(Nat.mul(images,height),height),stride)) == Nat.mul(images,output_height) : Nat}
  %Equal.sym(Nat,Nat.mod(Nat.mul(images,height),height),0n,remainder) :
    {Nat.add(Nat.mul(images,output_height),local(output_height,_,stride)) == Nat.mul(images,output_height) : Nat}
  %Equal.sym(Nat,local(output_height,0n,stride),0n,local_zero(output_height,stride,stride_positive)) : {Nat.add(Nat.mul(images,output_height),_) == Nat.mul(images,output_height) : Nat}
  Addition.zero_right(Nat.mul(images,output_height))

# A local endpoint before the image output_boundaries_boundary uses the same ceil/min formula.
def in_image(+height: Nat,+output_height: Nat,+stride: Nat,+image: Nat,+row: Nat,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+inside: {Nat.is_lt(row,height) == True{} : Bool}) ->
  {output_boundaries_boundary(height,output_height,stride,Nat.add(Nat.mul(image,height),row)) == Nat.add(Nat.mul(image,output_height),local(output_height,row,stride)) : Nat}:
  +position = Nat.add(Nat.mul(image,height),row)
  %Equal.sym(Nat,Nat.div(position,height),image,BandCoordinates.quotient(image,row,height,height_positive,inside)) :
    {Nat.add(Nat.mul(_,output_height),local(output_height,Nat.mod(position,height),stride)) == Nat.add(Nat.mul(image,output_height),local(output_height,row,stride)) : Nat}
  %Equal.sym(Nat,Nat.mod(position,height),row,BandCoordinates.remainder(image,row,height,height_positive,inside)) :
    {Nat.add(Nat.mul(image,output_height),local(output_height,_,stride)) == Nat.add(Nat.mul(image,output_height),local(output_height,row,stride)) : Nat}
  {==}

def successor_image(+image: Nat,+height: Nat) -> {Nat.mul(1n+image,height) == Nat.add(Nat.mul(image,height),height) : Nat}:
  Order.commutative(height,Nat.mul(image,height))

def last_in_image(+height: Nat,+output_height: Nat,+stride: Nat,+image: Nat,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {output_boundaries_boundary(height,output_height,stride,Nat.add(Nat.mul(image,height),height)) == Nat.add(Nat.mul(image,output_height),output_height) : Nat}:
  %successor_image(image,height) : {output_boundaries_boundary(height,output_height,stride,_) == Nat.add(Nat.mul(image,output_height),output_height) : Nat}
  Equal.trans(Nat,output_boundaries_boundary(height,output_height,stride,Nat.mul(1n+image,height)),Nat.mul(1n+image,output_height),Nat.add(Nat.mul(image,output_height),output_height),
    image_end(1n+image,height,output_height,stride,height_positive,stride_positive),successor_image(image,output_height))

def local_end(+height: Nat,+output_height: Nat,+stride: Nat,+row: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_eq(row,height),output_height,local(output_height,row,stride))

def endpoint_case(+height: Nat,+output_height: Nat,+stride: Nat,+image: Nat,+row: Nat,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},bounds: Order.UpperBound<row,height>) ->
  {output_boundaries_boundary(height,output_height,stride,Nat.add(Nat.mul(image,height),row)) == Nat.add(Nat.mul(image,output_height),local_end(height,output_height,stride,row)) : Nat}:
  match bounds:
    case Order.Before{+inside}:
      %Equal.sym(Bool,Nat.is_eq(row,height),False{},Order.less_not_equal(row,height,inside)) :
        {output_boundaries_boundary(height,output_height,stride,Nat.add(Nat.mul(image,height),row)) == Nat.add(Nat.mul(image,output_height),Bool.pick(Nat,_,output_height,local(output_height,row,stride))) : Nat}
      in_image(height,output_height,stride,image,row,height_positive,inside)
    case Order.At{+same}:
      %Equal.sym(Nat,row,height,same) :
        {output_boundaries_boundary(height,output_height,stride,Nat.add(Nat.mul(image,height),_)) == Nat.add(Nat.mul(image,output_height),local_end(height,output_height,stride,_)) : Nat}
      %Equal.sym(Bool,Nat.is_eq(height,height),True{},Order.equal_reflexive(height)) :
        {output_boundaries_boundary(height,output_height,stride,Nat.add(Nat.mul(image,height),height)) == Nat.add(Nat.mul(image,output_height),Bool.pick(Nat,_,output_height,local(output_height,height,stride))) : Nat}
      last_in_image(height,output_height,stride,image,height_positive,stride_positive)

def endpoint(+height: Nat,+output_height: Nat,+stride: Nat,+image: Nat,+row: Nat,
  height_positive: {Nat.is_lt(0n,height) == True{} : Bool},stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},inside: {Nat.is_le(row,height) == True{} : Bool}) ->
  {output_boundaries_boundary(height,output_height,stride,Nat.add(Nat.mul(image,height),row)) == Nat.add(Nat.mul(image,output_height),local_end(height,output_height,stride,row)) : Nat}:
  endpoint_case(height,output_height,stride,image,row,height_positive,stride_positive,Order.upper_bound_cases(row,height,inside))

# Monotone endpoint maps preserve a partition's coverage, including empty
# output intervals. All finite word bounds are explicit producer obligations.
def mapped(~Context: Data,~boundary: Context -> Nat -> Nat,+context: Context,plan: Bands.Plan) -> Bands.Plan:
  match plan:
    case Bands.Rows{+first,+rows}:
      +begin = boundary(context,U32.to_nat(first))
      +end = boundary(context,Nat.add(U32.to_nat(first),U32.to_nat(rows)))
      Bands.Rows{U32.from_nat(begin),U32.from_nat(Nat.sub(end,begin))}
    case Bands.Split{left,right}: Bands.Split{mapped(~Context,~boundary,context,left),mapped(~Context,~boundary,context,right)}

type LeafBounds<-first: Nat,-end: Nat,-capacity: U32> is Data:
  LeafBounds{first_room: {Nat.is_le(first,U32.to_nat(capacity)) == True{} : Bool},end_room: {Nat.is_le(end,U32.to_nat(capacity)) == True{} : Bool},
    ordered: {Nat.is_le(first,end) == True{} : Bool}}

def Evidence(~Context: Data,~boundary: Context -> Nat -> Nat,+context: Context,plan: Bands.Plan,+capacity: U32) -> Data:
  match plan:
    case Bands.Rows{+first,+rows}: LeafBounds<boundary(context,U32.to_nat(first)),boundary(context,Nat.add(U32.to_nat(first),U32.to_nat(rows))),capacity>
    case Bands.Split{left,right}: Shape.Both<Evidence(~Context,~boundary,context,left,capacity),Evidence(~Context,~boundary,context,right,capacity)>

type Interval<-layout: Bands.Plan,-first: Nat,-end: Nat> is Data:
  Interval{ordered: {Nat.is_le(first,end) == True{} : Bool},coverage: Coverage(layout,first,Nat.sub(end,first))}

def differences(+first: Nat,+middle: Nat,+end: Nat,+left: {Nat.is_le(first,middle) == True{} : Bool},+right: {Nat.is_le(middle,end) == True{} : Bool}) ->
  {Nat.sub(end,first) == Nat.add(Nat.sub(middle,first),Nat.sub(end,middle)) : Nat}:
  +left_rows = Nat.sub(middle,first)
  +right_rows = Nat.sub(end,middle)
  Order.cancel_sum(first,Nat.sub(end,first),Nat.add(left_rows,right_rows),
    Equal.trans(Nat,Nat.add(first,Nat.sub(end,first)),end,Nat.add(first,Nat.add(left_rows,right_rows)),
      Order.restore_shift(first,end,Order.transitive(first,middle,end,left,right)),
      Equal.sym(Nat,Nat.add(first,Nat.add(left_rows,right_rows)),end,
        Equal.trans(Nat,Nat.add(first,Nat.add(left_rows,right_rows)),Nat.add(Nat.add(first,left_rows),right_rows),end,
          Equal.sym(Nat,Nat.add(Nat.add(first,left_rows),right_rows),Nat.add(first,Nat.add(left_rows,right_rows)),Addition.associative(first,left_rows,right_rows)),
          Equal.trans(Nat,Nat.add(Nat.add(first,left_rows),right_rows),Nat.add(middle,right_rows),end,
            Equal.cong(Nat,Nat,position => Nat.add(position,right_rows),Nat.add(first,left_rows),middle,Order.restore_shift(first,middle,left)),
            Order.restore_shift(middle,end,right))))))

def joined(+left_plan: Bands.Plan,+right_plan: Bands.Plan,+first: Nat,+middle: Nat,+end: Nat,
  left: Interval<left_plan,first,middle>,right: Interval<right_plan,middle,end>) -> Interval<Bands.Split{left_plan,right_plan},first,end>:
  Interval{+left_order,left_coverage} = left
  Interval{+right_order,right_coverage} = right
  Interval{Order.transitive(first,middle,end,left_order,right_order),
    ForkCoverage{Nat.sub(middle,first),Nat.sub(end,middle),differences(first,middle,end,left_order,right_order),left_coverage,
      %Equal.sym(Nat,Nat.add(first,Nat.sub(middle,first)),middle,Order.restore_shift(first,middle,left_order)) : Coverage(right_plan,_,Nat.sub(end,middle))
      right_coverage}}

def leaf(~Context: Data,~boundary: Context -> Nat -> Nat,+context: Context,+first: U32,+rows: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  evidence: LeafBounds<boundary(context,U32.to_nat(first)),boundary(context,Nat.add(U32.to_nat(first),U32.to_nat(rows))),capacity>) ->
  Interval<mapped(~Context,~boundary,context,Bands.Rows{first,rows}),boundary(context,U32.to_nat(first)),boundary(context,Nat.add(U32.to_nat(first),U32.to_nat(rows)))>:
  LeafBounds{+first_room,+end_room,+ordered} = evidence
  +begin = boundary(context,U32.to_nat(first))
  +end = boundary(context,Nat.add(U32.to_nat(first),U32.to_nat(rows)))
  Interval{ordered,LeafCoverage{Bounds.roundtrip(begin,capacity,capacity_ok,first_room),
    Bounds.roundtrip(Nat.sub(end,begin),capacity,capacity_ok,Order.transitive(Nat.sub(end,begin),end,U32.to_nat(capacity),Order.sub_le(end,begin),end_room))}}

def coverage(~Context: Data,~boundary: Context -> Nat -> Nat,+context: Context,+plan: Bands.Plan,+first: Nat,+rows: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  source: Coverage(plan,first,rows),evidence: Evidence(~Context,~boundary,context,plan,capacity)) -> Interval<mapped(~Context,~boundary,context,plan),boundary(context,first),boundary(context,Nat.add(first,rows))>:
  match plan:
    case Bands.Rows{+native_first,+native_rows}:
      LeafCoverage{first_equal,rows_equal} = source
      %first_equal : Interval<mapped(~Context,~boundary,context,Bands.Rows{native_first,native_rows}),boundary(context,_),boundary(context,Nat.add(_,rows))>
      %rows_equal : Interval<mapped(~Context,~boundary,context,Bands.Rows{native_first,native_rows}),boundary(context,U32.to_nat(native_first)),boundary(context,Nat.add(U32.to_nat(native_first),_))>
      leaf(~Context,~boundary,context,native_first,native_rows,capacity,capacity_ok,evidence)
    case Bands.Split{+left,+right}:
      ForkCoverage{+left_rows,+right_rows,+rows_equal,left_coverage,right_coverage} = source
      Shape.Both{left_evidence,right_evidence} = evidence
      +end_equal = Equal.trans(Nat,Nat.add(Nat.add(first,left_rows),right_rows),Nat.add(first,Nat.add(left_rows,right_rows)),Nat.add(first,rows),
        Addition.associative(first,left_rows,right_rows),
        Equal.cong(Nat,Nat,count => Nat.add(first,count),Nat.add(left_rows,right_rows),rows,Equal.sym(Nat,rows,Nat.add(left_rows,right_rows),rows_equal)))
      %end_equal : Interval<mapped(~Context,~boundary,context,Bands.Split{left,right}),boundary(context,first),boundary(context,_)>
      joined(mapped(~Context,~boundary,context,left),mapped(~Context,~boundary,context,right),boundary(context,first),boundary(context,Nat.add(first,left_rows)),boundary(context,Nat.add(Nat.add(first,left_rows),right_rows)),
        coverage(~Context,~boundary,context,left,first,left_rows,capacity,capacity_ok,left_coverage,left_evidence),
        coverage(~Context,~boundary,context,right,Nat.add(first,left_rows),right_rows,capacity,capacity_ok,right_coverage,right_evidence))

def output_endpoint_room(+images: Nat,+height: Nat,+output_height: Nat,+stride: Nat,+row: Nat,+capacity: U32,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  output_room: {Nat.is_le(Nat.mul(images,output_height),U32.to_nat(capacity)) == True{} : Bool},row_inside: {Nat.is_le(row,Nat.mul(images,height)) == True{} : Bool}) ->
  {Nat.is_le(output_boundaries_boundary(height,output_height,stride,row),U32.to_nat(capacity)) == True{} : Bool}:
  Order.transitive(output_boundaries_boundary(height,output_height,stride,row),output_boundaries_boundary(height,output_height,stride,Nat.mul(images,height)),U32.to_nat(capacity),
    monotone(height,output_height,stride,row,Nat.mul(images,height),height_positive,stride_positive,row_inside),
    %Equal.sym(Nat,output_boundaries_boundary(height,output_height,stride,Nat.mul(images,height)),Nat.mul(images,output_height),image_end(images,height,output_height,stride,height_positive,stride_positive)) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    output_room)

# Coverage supplies every leaf's source bounds. The global map discharges
# every output endpoint bound and ordering; no positive leaf size is assumed.
def output_evidence(+images: Nat,+height: Nat,+output_height: Nat,+stride: Nat,+plan: Bands.Plan,+capacity: U32,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(images,output_height),U32.to_nat(capacity)) == True{} : Bool},within: Within(plan,0n,Nat.mul(images,height))) ->
  Evidence(~Mapping,~mapping,Mapping{height,output_height,stride},plan,capacity):
  match plan:
    case Bands.Rows{+first,+rows}:
      LeafWithin{lower,+upper} = within
      +begin = U32.to_nat(first)
      +end = Nat.add(begin,U32.to_nat(rows))
      LeafBounds{
        output_endpoint_room(images,height,output_height,stride,begin,capacity,height_positive,stride_positive,output_room,
          Order.transitive(begin,end,Nat.mul(images,height),Order.le_add(begin,U32.to_nat(rows)),upper)),
        output_endpoint_room(images,height,output_height,stride,end,capacity,height_positive,stride_positive,output_room,upper),
        monotone(height,output_height,stride,begin,end,height_positive,stride_positive,Order.le_add(begin,U32.to_nat(rows)))}
    case Bands.Split{+left,+right}:
      Shape.Both{left_bounds,right_bounds} = within
      Shape.Both{output_evidence(images,height,output_height,stride,left,capacity,height_positive,stride_positive,output_room,left_bounds),
        output_evidence(images,height,output_height,stride,right,capacity,height_positive,stride_positive,output_room,right_bounds)}

def interval_coverage(-layout: Bands.Plan,+begin: Nat,+end: Nat,+total: Nat,interval: Interval<layout,begin,end>,
  +begin_equal: {begin == 0n : Nat},+end_equal: {end == total : Nat}) -> Coverage(layout,0n,total):
  Interval{ordered,coverage} = interval
  +count_equal = Equal.trans(Nat,Nat.sub(end,begin),Nat.sub(total,0n),total,
    Equal.trans(Nat,Nat.sub(end,begin),Nat.sub(total,begin),Nat.sub(total,0n),
      Equal.cong(Nat,Nat,n => Nat.sub(n,begin),end,total,end_equal),
      Equal.cong(Nat,Nat,n => Nat.sub(total,n),begin,0n,begin_equal)),Order.subtract_zero(total))
  %begin_equal : Coverage(layout,_,total)
  %count_equal : Coverage(layout,begin,_)
  coverage

def output_coverage(+images: Nat,+height: Nat,+output_height: Nat,+stride: Nat,+plan: Bands.Plan,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(images,output_height),U32.to_nat(capacity)) == True{} : Bool},+source: Coverage(plan,0n,Nat.mul(images,height))) ->
  Coverage(mapped(~Mapping,~mapping,Mapping{height,output_height,stride},plan),0n,Nat.mul(images,output_height)):
  interval_coverage(mapped(~Mapping,~mapping,Mapping{height,output_height,stride},plan),output_boundaries_boundary(height,output_height,stride,0n),
    output_boundaries_boundary(height,output_height,stride,Nat.mul(images,height)),Nat.mul(images,output_height),
    coverage(~Mapping,~mapping,Mapping{height,output_height,stride},plan,0n,Nat.mul(images,height),capacity,capacity_ok,source,
      output_evidence(images,height,output_height,stride,plan,capacity,height_positive,stride_positive,output_room,
        covered(plan,0n,Nat.mul(images,height),0n,Nat.mul(images,height),source,{==},Order.reflexive(Nat.mul(images,height))))),
    image_end(0n,height,output_height,stride,height_positive,stride_positive),
    image_end(images,height,output_height,stride,height_positive,stride_positive))

# A mapped interval remains image-local, including a zero interval at the
# next image boundary. This lemma separates row geometry from stored values.
def subtract_below(+value: Nat,+amount: Nat,inside: {Nat.is_le(value,amount) == True{} : Bool}) -> {Nat.sub(value,amount) == 0n : Nat}:
  match value amount:
    case 0n amount: Order.zero_subtract(amount)
    case 1n+value 0n: Empty.absurd({Nat.sub(1n+value,0n) == 0n : Nat},WordEncoding.false_true(inside))
    case 1n+ +value 1n+ +amount: subtract_below(value,amount,inside)

def span_case(+height: Nat,+image: Nat,+begin: Nat,+end: Nat,+first: Nat,+rows: Nat,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
  +ordered: {Nat.is_le(begin,end) == True{} : Bool},+end_inside: {Nat.is_le(end,height) == True{} : Bool},
  +first_exact: {first == Nat.add(Nat.mul(image,height),begin) : Nat},+rows_exact: {rows == Nat.sub(end,begin) : Nat},
  bounds: Order.UpperBound<begin,height>) -> {Nat.is_le(Nat.add(Nat.mod(first,height),rows),height) == True{} : Bool}:
  match bounds:
    case Order.Before{+inside}:
      %Equal.sym(Nat,first,Nat.add(Nat.mul(image,height),begin),first_exact) : {Nat.is_le(Nat.add(Nat.mod(_,height),rows),height) == True{} : Bool}
      %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(image,height),begin),height),begin,BandCoordinates.remainder(image,begin,height,height_positive,inside)) :
        {Nat.is_le(Nat.add(_,rows),height) == True{} : Bool}
      %Equal.sym(Nat,rows,Nat.sub(end,begin),rows_exact) : {Nat.is_le(Nat.add(begin,_),height) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(begin,Nat.sub(end,begin)),end,Order.restore_shift(begin,end,ordered)) : {Nat.is_le(_,height) == True{} : Bool}
      end_inside
    case Order.At{+same}:
      +difference_zero = {
        %Equal.sym(Nat,begin,height,same) : {Nat.sub(end,_) == 0n : Nat}
        subtract_below(end,height,end_inside) : {Nat.sub(end,begin) == 0n : Nat}}
      %Equal.sym(Nat,rows,0n,Equal.trans(Nat,rows,Nat.sub(end,begin),0n,rows_exact,difference_zero)) : {Nat.is_le(Nat.add(Nat.mod(first,height),_),height) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(Nat.mod(first,height),0n),Nat.mod(first,height),Addition.zero_right(Nat.mod(first,height))) : {Nat.is_le(_,height) == True{} : Bool}
      Bounds.strict_le(Nat.mod(first,height),height,DivisionOrder.remainder_small(first,height,height_positive))

def span(+height: Nat,+image: Nat,+begin: Nat,+end: Nat,+first: Nat,+rows: Nat,
  height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+begin_inside: {Nat.is_le(begin,height) == True{} : Bool},
  ordered: {Nat.is_le(begin,end) == True{} : Bool},end_inside: {Nat.is_le(end,height) == True{} : Bool},
  first_exact: {first == Nat.add(Nat.mul(image,height),begin) : Nat},rows_exact: {rows == Nat.sub(end,begin) : Nat}) ->
  {Nat.is_le(Nat.add(Nat.mod(first,height),rows),height) == True{} : Bool}:
  span_case(height,image,begin,end,first,rows,height_positive,ordered,end_inside,first_exact,rows_exact,Order.upper_bound_cases(begin,height,begin_inside))
