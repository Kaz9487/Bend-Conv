# Recover complete row coverage from the actual public convolution inspection.
# Metadata is projected without copying any native owner or trusting a caller's
# assertion that the leaves happen to form a contiguous partition.
import Base
import ../tensor_band.bend as Bands
import ./tensor_band_model.bend as Model
import ./storage_buffer_model.bend as Buffer
import ./storage_certificate.bend as Certificate
import ./tensor_band_plan.bend as BandPlan
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./boolean_logic.bend as Boolean
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_division.bend as U32Division
import ./nat_division_order.bend as DivisionOrder
import ./tensor_view_bounds.bend as ViewBounds
import ./u32_comparison.bend as Comparison
import ./tensor_band_coordinates.bend as BandCoordinates

def valid(inspection: Bands.Inspection) -> Bool:
  Bands.Inspection{valid,first,end,leaves} = inspection
  valid

def first(inspection: Bands.Inspection) -> U32:
  Bands.Inspection{valid,first,end,leaves} = inspection
  first

def end(inspection: Bands.Inspection) -> U32:
  Bands.Inspection{valid,first,end,leaves} = inspection
  end

def leaf(~Element: Data,geometry: Bands.Geometry,buffer: Buffer.Model<Element>,first: U32,rows: U32) -> Bands.Inspection:
  Buffer.Model{tree,length,Certificate.Certificate{depth,equation}} = buffer
  Bands.leaf_inspection(geometry,length,depth,first,rows)

def model(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>) -> Bands.Inspection:
  match bands:
    case Model.Leaf{buffer,first,rows}: leaf(~Element,geometry,buffer,first,rows)
    case Model.Fork{left,right}: Bands.join_inspection(model(~Element,geometry,left),model(~Element,geometry,right))

def inspect(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>) ->
  {Bands.inspect(~Element,geometry,Model.pack(~Element,bands)) == (Model.pack(~Element,bands),model(~Element,geometry,bands)) : Bands.Bands(~Element) & Bands.Inspection}:
  match bands:
    case Model.Leaf{Buffer.Model{tree,length,Certificate.Certificate{depth,equation}},first,rows}: {==}
    case Model.Fork{+left,+right}:
      %Equal.sym(Bands.Bands(~Element) & Bands.Inspection,Bands.inspect(~Element,geometry,Model.pack(~Element,left)),(Model.pack(~Element,left),model(~Element,geometry,left)),inspect(~Element,geometry,left)) :
        {Bands.inspected_pair(~Element,_,Bands.inspect(~Element,geometry,Model.pack(~Element,right))) == (Model.pack(~Element,bands),model(~Element,geometry,bands)) : Bands.Bands(~Element) & Bands.Inspection}
      %Equal.sym(Bands.Bands(~Element) & Bands.Inspection,Bands.inspect(~Element,geometry,Model.pack(~Element,right)),(Model.pack(~Element,right),model(~Element,geometry,right)),inspect(~Element,geometry,right)) :
        {Bands.inspected_pair(~Element,(Model.pack(~Element,left),model(~Element,geometry,left)),_) == (Model.pack(~Element,bands),model(~Element,geometry,bands)) : Bands.Bands(~Element) & Bands.Inspection}
      {==}

type Coverage<-plan: Bands.Plan,-inspection: Bands.Inspection> is Data:
  Coverage{rows: Nat,coverage: BandPlan.Coverage(plan,U32.to_nat(first(inspection)),rows),
    end_exact: {U32.to_nat(end(inspection)) == Nat.add(U32.to_nat(first(inspection)),rows) : Nat}}

def total_rows(geometry: Bands.Geometry) -> Nat:
  Bands.Geometry{batch,channels,height,width} = geometry
  Nat.mul(U32.to_nat(batch),U32.to_nat(height))

# Decode the exact subtraction used by inspect_leaf. The remainder bound is
# proved by division, so the unsigned subtraction cannot wrap.
def local_span(+height: U32,+first: U32,+rows: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool},
  accepted: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool}:
  +offset = (first % height : U32)
  +offset_exact = U32Division.modulo(first,height,capacity,capacity_ok,height_bound,positive)
  +offset_inside = {
    %Equal.sym(Nat,U32.to_nat(offset),Nat.mod(U32.to_nat(first),U32.to_nat(height)),offset_exact) : {Nat.is_lt(_,U32.to_nat(height)) == True{} : Bool}
    DivisionOrder.remainder_small(U32.to_nat(first),U32.to_nat(height),positive) : {Nat.is_lt(U32.to_nat(offset),U32.to_nat(height)) == True{} : Bool}}
  +offset_le = Bounds.strict_le(U32.to_nat(offset),U32.to_nat(height),offset_inside)
  +height_room = Order.transitive(U32.to_nat(height),Nat.double(U32.to_nat(height)),U32.to_nat(capacity),Order.le_double(U32.to_nat(height)),height_bound)
  +room_exact = Arithmetic.subtract(height,offset,capacity,capacity_ok,offset_le,
    Order.transitive(Nat.sub(U32.to_nat(height),U32.to_nat(offset)),U32.to_nat(height),U32.to_nat(capacity),Order.sub_le(U32.to_nat(height),U32.to_nat(offset)),height_room))
  %offset_exact : {Nat.is_le(Nat.add(_,U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool}
  ViewBounds.sum_within(U32.to_nat(offset),U32.to_nat(rows),U32.to_nat(height),offset_le,
    %room_exact : {Nat.is_le(U32.to_nat(rows),_) == True{} : Bool}
    Bounds.as_nat_le(rows,(height - offset : U32),accepted))

def local_guard(+height: U32,+first: U32,+rows: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},+height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool},
  span: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool}) ->
  {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool}:
  +offset = (first % height : U32)
  +offset_exact = U32Division.modulo(first,height,capacity,capacity_ok,height_bound,positive)
  +span = {
    %Equal.sym(Nat,U32.to_nat(offset),Nat.mod(U32.to_nat(first),U32.to_nat(height)),offset_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool}
    span : {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool}}
  +offset_inside = Order.transitive(U32.to_nat(offset),Nat.add(U32.to_nat(offset),U32.to_nat(rows)),U32.to_nat(height),Order.le_add(U32.to_nat(offset),U32.to_nat(rows)),span)
  +room_exact = Arithmetic.subtract(height,offset,capacity,capacity_ok,offset_inside,
    Order.transitive(Nat.sub(U32.to_nat(height),U32.to_nat(offset)),U32.to_nat(height),U32.to_nat(capacity),Order.sub_le(U32.to_nat(height),U32.to_nat(offset)),
      Order.transitive(U32.to_nat(height),Nat.double(U32.to_nat(height)),U32.to_nat(capacity),Order.le_double(U32.to_nat(height)),height_bound)))
  %Equal.sym(Bool,U32.is_le(rows,(height - offset : U32)),Nat.is_le(U32.to_nat(rows),U32.to_nat((height - offset : U32))),Comparison.le(rows,(height - offset : U32))) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((height - offset : U32)),Nat.sub(U32.to_nat(height),U32.to_nat(offset)),room_exact) : {Nat.is_le(U32.to_nat(rows),_) == True{} : Bool}
  Order.cancel_offset_le(U32.to_nat(offset),U32.to_nat(rows),Nat.sub(U32.to_nat(height),U32.to_nat(offset)),
    %Equal.sym(Nat,Nat.add(U32.to_nat(offset),Nat.sub(U32.to_nat(height),U32.to_nat(offset))),U32.to_nat(height),Order.restore_shift(U32.to_nat(offset),U32.to_nat(height),offset_inside)) :
      {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(rows)),_) == True{} : Bool} span)

type LeafFacts<-batch: U32,-channels: U32,-height: U32,-width: U32,-first: U32,-rows: U32,-length: Nat,-depth: Nat> is Data:
  LeafFacts{local: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool},
    rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) == True{} : Bool},
    length_exact: {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
    allocation_fit: {Nat.is_le(length,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}}

def leaf_facts(+batch: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+length: Nat,+depth: Nat,
  +accepted: {valid(Bands.leaf_inspection(Bands.Geometry{batch,channels,height,width},length,depth,first,rows)) == True{} : Bool}) ->
  LeafFacts<batch,channels,height,width,first,rows,length,depth>:
  +local = U32.is_le(rows,(height - first % height : U32))
  +room = Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height)))
  +counted = Nat.is_eq(length,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))))
  +allocated = Nat.is_le(length,U32.to_nat(U32.shln(1,depth)))
  +prefix = Boolean.left(((local && room : Bool) && counted : Bool),allocated,accepted)
  +rows_valid = Boolean.left((local && room : Bool),counted,prefix)
  LeafFacts{Boolean.left(local,room,rows_valid),Boolean.right(local,room,rows_valid),
    Order.equal_of_is_eq(length,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),Boolean.right((local && room : Bool),counted,prefix)),
    Boolean.right(((local && room : Bool) && counted : Bool),allocated,accepted)}

def leaf_coverage(+geometry: Bands.Geometry,+length: Nat,+depth: Nat,+start: U32,+rows: U32,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},total_room: {Nat.is_le(total_rows(geometry),U32.to_nat(capacity)) == True{} : Bool},
  accepted: {valid(Bands.leaf_inspection(geometry,length,depth,start,rows)) == True{} : Bool}) ->
  Coverage<Bands.Rows{start,rows},Bands.leaf_inspection(geometry,length,depth,start,rows)>:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  +local = U32.is_le(rows,(height - start % height : U32))
  +room = Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height)))
  +counted = Nat.is_eq(length,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))))
  +allocated = Nat.is_le(length,U32.to_nat(U32.shln(1,depth)))
  Coverage{U32.to_nat(rows),BandPlan.LeafCoverage{{==},{==}},Arithmetic.add(start,rows,capacity,capacity_ok,
    Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height)),U32.to_nat(capacity),
      Boolean.right(local,room,Boolean.left((local && room : Bool),counted,Boolean.left(((local && room : Bool) && counted : Bool),allocated,accepted))),total_room))}

def joined_coverage(+left_plan: Bands.Plan,+right_plan: Bands.Plan,+left: Bands.Inspection,+right: Bands.Inspection,
  left_coverage: Coverage<left_plan,left>,right_coverage: Coverage<right_plan,right>,
  adjacent: {U32.is_eq(end(left),first(right)) == True{} : Bool}) -> Coverage<Bands.Split{left_plan,right_plan},Bands.join_inspection(left,right)>:
  Bands.Inspection{lv,+lf,+le,ln} = left
  Bands.Inspection{rv,+rf,+re,rn} = right
  Coverage{+left_rows,left_coverage,+left_end} = left_coverage
  Coverage{+right_rows,right_coverage,+right_end} = right_coverage
  +joined = Equal.trans(Nat,U32.to_nat(rf),U32.to_nat(le),Nat.add(U32.to_nat(lf),left_rows),
    Equal.cong(U32,Nat,U32.to_nat,rf,le,Equal.sym(U32,le,rf,ShapeLaws.native_equal(le,rf,adjacent))),left_end)
  Coverage{Nat.add(left_rows,right_rows),BandPlan.ForkCoverage{left_rows,right_rows,{==},left_coverage,
    %joined : BandPlan.Coverage(right_plan,_,right_rows) right_coverage},
    Equal.trans(Nat,U32.to_nat(re),Nat.add(U32.to_nat(rf),right_rows),Nat.add(U32.to_nat(lf),Nat.add(left_rows,right_rows)),right_end,
      Equal.trans(Nat,Nat.add(U32.to_nat(rf),right_rows),Nat.add(Nat.add(U32.to_nat(lf),left_rows),right_rows),Nat.add(U32.to_nat(lf),Nat.add(left_rows,right_rows)),
        Equal.cong(Nat,Nat,value => Nat.add(value,right_rows),U32.to_nat(rf),Nat.add(U32.to_nat(lf),left_rows),joined),Addition.associative(U32.to_nat(lf),left_rows,right_rows)))}

def joined_left(+left: Bands.Inspection,+right: Bands.Inspection,accepted: {valid(Bands.join_inspection(left,right)) == True{} : Bool}) -> {valid(left) == True{} : Bool}:
  Bands.Inspection{+lv,lf,+le,ln} = left
  Bands.Inspection{+rv,+rf,re,rn} = right
  Boolean.left(lv,rv,Boolean.left((lv && rv : Bool),U32.is_eq(le,rf),accepted))

def joined_right(+left: Bands.Inspection,+right: Bands.Inspection,accepted: {valid(Bands.join_inspection(left,right)) == True{} : Bool}) -> {valid(right) == True{} : Bool}:
  Bands.Inspection{+lv,lf,+le,ln} = left
  Bands.Inspection{+rv,+rf,re,rn} = right
  Boolean.right(lv,rv,Boolean.left((lv && rv : Bool),U32.is_eq(le,rf),accepted))

def joined_adjacent(+left: Bands.Inspection,+right: Bands.Inspection,accepted: {valid(Bands.join_inspection(left,right)) == True{} : Bool}) -> {U32.is_eq(end(left),first(right)) == True{} : Bool}:
  Bands.Inspection{+lv,lf,+le,ln} = left
  Bands.Inspection{+rv,+rf,re,rn} = right
  Boolean.right((lv && rv : Bool),U32.is_eq(le,rf),accepted)

def coverage(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+total_room: {Nat.is_le(total_rows(geometry),U32.to_nat(capacity)) == True{} : Bool},
  accepted: {valid(model(~Element,geometry,bands)) == True{} : Bool}) -> Coverage<Model.layout(~Element,bands),model(~Element,geometry,bands)>:
  match bands:
    case Model.Leaf{Buffer.Model{tree,+length,Certificate.Certificate{+depth,equation}},+first,+rows}:
      leaf_coverage(geometry,length,depth,first,rows,capacity,capacity_ok,total_room,accepted)
    case Model.Fork{+left,+right}:
      +accepted = accepted
      joined_coverage(Model.layout(~Element,left),Model.layout(~Element,right),model(~Element,geometry,left),model(~Element,geometry,right),
        coverage(~Element,geometry,left,capacity,capacity_ok,total_room,joined_left(model(~Element,geometry,left),model(~Element,geometry,right),accepted)),
        coverage(~Element,geometry,right,capacity,capacity_ok,total_room,joined_right(model(~Element,geometry,left),model(~Element,geometry,right),accepted)),
        joined_adjacent(model(~Element,geometry,left),model(~Element,geometry,right),accepted))

def complete(+plan: Bands.Plan,+inspection: Bands.Inspection,+total: Nat,covered: Coverage<plan,inspection>,
  starts_zero: {U32.is_zero(first(inspection)) == True{} : Bool},ends_total: {Nat.is_eq(U32.to_nat(end(inspection)),total) == True{} : Bool}) -> BandPlan.Coverage(plan,0n,total):
  Coverage{+rows,covered,end_exact} = covered
  +zero = Equal.cong(U32,Nat,U32.to_nat,first(inspection),0,ShapeLaws.native_equal(first(inspection),0,starts_zero))
  +rows_equal = Equal.trans(Nat,rows,Nat.add(U32.to_nat(first(inspection)),rows),total,
    Equal.cong(Nat,Nat,value => Nat.add(value,rows),0n,U32.to_nat(first(inspection)),Equal.sym(Nat,U32.to_nat(first(inspection)),0n,zero)),
    Equal.trans(Nat,Nat.add(U32.to_nat(first(inspection)),rows),U32.to_nat(end(inspection)),total,
      Equal.sym(Nat,U32.to_nat(end(inspection)),Nat.add(U32.to_nat(first(inspection)),rows),end_exact),Order.equal_of_is_eq(U32.to_nat(end(inspection)),total,ends_total)))
  %rows_equal : BandPlan.Coverage(plan,0n,_)
  %zero : BandPlan.Coverage(plan,_,rows)
  covered

# These are the same three metadata conditions used by band_inspected. The
# machine row bound is separate from the inspected storage-length conditions.
def checked(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},total_room: {Nat.is_le(total_rows(geometry),U32.to_nat(capacity)) == True{} : Bool},
  +accepted: {valid(model(~Element,geometry,bands)) && U32.is_zero(first(model(~Element,geometry,bands))) &&
    Nat.is_eq(U32.to_nat(end(model(~Element,geometry,bands))),total_rows(geometry)) == True{} : Bool}) ->
  BandPlan.Coverage(Model.layout(~Element,bands),0n,total_rows(geometry)):
  +inspection = model(~Element,geometry,bands)
  +starts = U32.is_zero(first(inspection))
  +ends = Nat.is_eq(U32.to_nat(end(inspection)),total_rows(geometry))
  +prefix = Boolean.left((valid(inspection) && starts : Bool),ends,accepted)
  complete(Model.layout(~Element,bands),inspection,total_rows(geometry),coverage(~Element,geometry,bands,capacity,capacity_ok,total_room,Boolean.left(valid(inspection),starts,prefix)),
    Boolean.right(valid(inspection),starts,prefix),Boolean.right((valid(inspection) && starts : Bool),ends,accepted))
