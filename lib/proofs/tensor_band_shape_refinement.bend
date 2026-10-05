# Shape metadata is independent of element values and ownership certificates.
# These fields plus ContentReady imply the actual leaf inspection accepts.
import Base
import ../tensor_band.bend as Bands
import ./tensor_band_model.bend as Model
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_certificate.bend as Certificate
import ./tensor_band_inspection.bend as Inspection
import ./nat_order.bend as Order
import ./tensor_band_plan.bend as BandPlan
import ./natural_addition.bend as Addition
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./row_major_index.bend as RowMajor

def width_positive(geometry: Bands.Geometry) -> Bool:
  Bands.Geometry{images,channels,height,width} = geometry
  Nat.is_lt(0n,U32.to_nat(width))

type Fields<-images: U32,-channels: U32,-height: U32,-width: U32,-first: U32,-rows: U32,-length: Nat> is Data:
  Fields{local: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool},
    rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool},
    length_exact: {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat}}

def Leaf(geometry: Bands.Geometry,first: U32,rows: U32,length: Nat) -> Data:
  Bands.Geometry{images,channels,height,width} = geometry
  Fields<images,channels,height,width,first,rows,length>

type Count<-geometry: Bands.Geometry,-rows: U32,-length: Nat> is Data:
  Count{exact: {U32.to_nat(Bands.local_count(geometry,rows)) == length : Nat},
    allocatable: {U32.is_le(Bands.local_count(geometry,rows),1073741824) == True{} : Bool}}

def count(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  width_positive: {width_positive(geometry) == True{} : Bool},metadata: Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Ready<Element,input>) ->
  Count<geometry,rows,Buffer.length(~Element,input)>:
  Bands.Geometry{images,+channels,height,+width} = geometry
  Fields{local,rows_fit,+length_exact} = metadata
  Buffer.Ready{+depth_ok,+room} = ready
  +capacity = Tree.capacity(Element,Buffer.tree(~Element,input))
  +expected = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))
  +exact = Equal.trans(Nat,U32.to_nat(Bands.local_count(geometry,rows)),expected,Buffer.length(~Element,input),
    Equal.trans(Nat,U32.to_nat(Bands.local_count(geometry,rows)),Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),U32.to_nat(width)),expected,
      Encoded.multiply_three(channels,rows,width,capacity,Buffer.capacity_bound(~Element,input,depth_ok),width_positive,
        %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),U32.to_nat(width)),expected,RowMajor.assoc(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        %length_exact : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} room),RowMajor.assoc(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width))),
    Equal.sym(Nat,Buffer.length(~Element,input),expected,length_exact))
  Count{exact,Arithmetic.count_within(Bands.local_count(geometry,rows),Buffer.length(~Element,input),capacity,1073741824,exact,room,Buffer.capacity_allocatable(~Element,input,depth_ok))}

def Contents(~Element: Data,+geometry: Bands.Geometry,content: Model.Content<Element>) -> Data:
  match content:
    case Model.RowBand{Buffer.Content{tree,length},first,rows}: Leaf(geometry,first,rows,length)
    case Model.Joined{left,right}: Shape.Both<Contents(~Element,geometry,left),Contents(~Element,geometry,right)>

def leaf_valid(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  metadata: Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Ready<Element,input>) ->
  {Inspection.valid(Inspection.leaf(~Element,geometry,input,first,rows)) == True{} : Bool}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = input
  Fields{local,rows_fit,length_exact} = metadata
  Buffer.Ready{depth_ok,room} = ready
  +shape_length = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))
  +counted = {
    %length_exact : {Nat.is_eq(length,_) == True{} : Bool} Order.equal_reflexive(length) : {Nat.is_eq(length,shape_length) == True{} : Bool}}
  +allocated = {
    %Shape.capacity(Element,depth,tree,Buffer.balanced(~Element,input)) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
    room : {Nat.is_le(length,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}}
  %Equal.sym(Bool,U32.is_le(rows,(height - first % height : U32)),True{},local) :
    {_ && Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height))) && Nat.is_eq(length,shape_length) && Nat.is_le(length,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height))),True{},rows_fit) :
    {_ && Nat.is_eq(length,shape_length) && Nat.is_le(length,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_eq(length,shape_length),True{},counted) : {_ && Nat.is_le(length,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
  allocated

type Accepted<-inspection: Bands.Inspection,-first: Nat,-rows: Nat> is Data:
  Accepted{valid: {Inspection.valid(inspection) == True{} : Bool},
    first_exact: {U32.to_nat(Inspection.first(inspection)) == first : Nat},
    end_exact: {U32.to_nat(Inspection.end(inspection)) == Nat.add(first,rows) : Nat}}

def equal_words(+left: U32,+right: U32,equation: {U32.to_nat(left) == U32.to_nat(right) : Nat}) -> {U32.is_eq(left,right) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(left,right),Nat.is_eq(U32.to_nat(left),U32.to_nat(right)),Comparison.eq(left,right)) : {_ == True{} : Bool}
  %equation : {Nat.is_eq(U32.to_nat(left),_) == True{} : Bool}
  Order.equal_reflexive(U32.to_nat(left))

def joined(+left: Bands.Inspection,+right: Bands.Inspection,+first: Nat,+left_rows: Nat,+right_rows: Nat,
  left_ok: Accepted<left,first,left_rows>,right_ok: Accepted<right,Nat.add(first,left_rows),right_rows>) ->
  Accepted<Bands.join_inspection(left,right),first,Nat.add(left_rows,right_rows)>:
  Bands.Inspection{+lv,+lf,+le,ln} = left
  Bands.Inspection{+rv,+rf,+re,rn} = right
  Accepted{left_valid,left_first,left_end} = left_ok
  Accepted{right_valid,right_first,right_end} = right_ok
  +adjacent = equal_words(le,rf,Equal.trans(Nat,U32.to_nat(le),Nat.add(first,left_rows),U32.to_nat(rf),left_end,
    Equal.sym(Nat,U32.to_nat(rf),Nat.add(first,left_rows),right_first)))
  Accepted{
    %Equal.sym(Bool,lv,True{},left_valid) : {_ && rv && U32.is_eq(le,rf) == True{} : Bool}
    %Equal.sym(Bool,rv,True{},right_valid) : {_ && U32.is_eq(le,rf) == True{} : Bool} adjacent,
    left_first,Equal.trans(Nat,U32.to_nat(re),Nat.add(Nat.add(first,left_rows),right_rows),Nat.add(first,Nat.add(left_rows,right_rows)),
      right_end,Addition.associative(first,left_rows,right_rows))}

def leaf_accepted(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+start: Nat,+count: Nat,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},total_room: {Nat.is_le(Inspection.total_rows(geometry),U32.to_nat(capacity)) == True{} : Bool},
  +metadata: Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Ready<Element,input>,
  covered: BandPlan.LeafCoverage<first,rows,start,count>) -> Accepted<Inspection.leaf(~Element,geometry,input,first,rows),start,count>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = input
  Fields{local,rows_fit,length_exact} = metadata
  BandPlan.LeafCoverage{+first_exact,rows_exact} = covered
  +end_exact = Arithmetic.add(first,rows,capacity,capacity_ok,
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry),U32.to_nat(capacity),rows_fit,total_room))
  Accepted{leaf_valid(~Element,geometry,first,rows,input,metadata,ready),first_exact,
    %first_exact : {U32.to_nat((first + rows : U32)) == Nat.add(_,count) : Nat}
    %rows_exact : {U32.to_nat((first + rows : U32)) == Nat.add(U32.to_nat(first),_) : Nat}
    end_exact}

def accepted(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>,+first: Nat,+rows: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+total_room: {Nat.is_le(Inspection.total_rows(geometry),U32.to_nat(capacity)) == True{} : Bool},
  metadata: Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands),
  covered: BandPlan.Coverage(Model.layout(~Element,bands),first,rows)) -> Accepted<Inspection.model(~Element,geometry,bands),first,rows>:
  match bands:
    case Model.Leaf{input,native_first,native_rows}:
      Buffer.Model{+tree,+length,+certificate} = input
      leaf_accepted(~Element,geometry,native_first,native_rows,input,first,rows,capacity,capacity_ok,total_room,metadata,ready,covered)
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      BandPlan.ForkCoverage{+left_rows,+right_rows,rows_equal,left_covered,right_covered} = covered
      %Equal.sym(Nat,rows,Nat.add(left_rows,right_rows),rows_equal) : Accepted<Inspection.model(~Element,geometry,bands),first,_>
      joined(Inspection.model(~Element,geometry,left),Inspection.model(~Element,geometry,right),first,left_rows,right_rows,
        accepted(~Element,geometry,left,first,left_rows,capacity,capacity_ok,total_room,left_metadata,left_ready,left_covered),
        accepted(~Element,geometry,right,Nat.add(first,left_rows),right_rows,capacity,capacity_ok,total_room,right_metadata,right_ready,right_covered))

def checked(+inspection: Bands.Inspection,+total: Nat,accepted: Accepted<inspection,0n,total>) ->
  {Inspection.valid(inspection) && U32.is_zero(Inspection.first(inspection)) && Nat.is_eq(U32.to_nat(Inspection.end(inspection)),total) == True{} : Bool}:
  Accepted{valid,first_exact,end_exact} = accepted
  +starts = equal_words(Inspection.first(inspection),0,first_exact)
  +ends = {
    %end_exact : {Nat.is_eq(U32.to_nat(Inspection.end(inspection)),_) == True{} : Bool}
    Order.equal_reflexive(U32.to_nat(Inspection.end(inspection))) : {Nat.is_eq(U32.to_nat(Inspection.end(inspection)),total) == True{} : Bool}}
  %Equal.sym(Bool,Inspection.valid(inspection),True{},valid) : {_ && U32.is_zero(Inspection.first(inspection)) && Nat.is_eq(U32.to_nat(Inspection.end(inspection)),total) == True{} : Bool}
  %Equal.sym(Bool,U32.is_zero(Inspection.first(inspection)),True{},starts) : {_ && Nat.is_eq(U32.to_nat(Inspection.end(inspection)),total) == True{} : Bool}
  ends

type WellFormed<-Element: Data,-geometry: Bands.Geometry,-content: Model.Content<Element>> is Data:
  WellFormed{metadata: Contents(~Element,geometry,content),ready: Model.ContentReady(~Element,content),
    coverage: BandPlan.Coverage(Model.content_layout(~Element,content),0n,Inspection.total_rows(geometry))}

def well_formed_checked(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},total_room: {Nat.is_le(Inspection.total_rows(geometry),U32.to_nat(capacity)) == True{} : Bool},
  evidence: WellFormed<Element,geometry,Model.content(~Element,bands)>) ->
  {Inspection.valid(Inspection.model(~Element,geometry,bands)) && U32.is_zero(Inspection.first(Inspection.model(~Element,geometry,bands))) &&
    Nat.is_eq(U32.to_nat(Inspection.end(Inspection.model(~Element,geometry,bands))),Inspection.total_rows(geometry)) == True{} : Bool}:
  WellFormed{metadata,ready,coverage} = evidence
  checked(Inspection.model(~Element,geometry,bands),Inspection.total_rows(geometry),
    accepted(~Element,geometry,bands,0n,Inspection.total_rows(geometry),capacity,capacity_ok,total_room,metadata,Model.ready(~Element,bands,ready),
      %Equal.sym(Bands.Plan,Model.layout(~Element,bands),Model.content_layout(~Element,Model.content(~Element,bands)),Model.layout_matches_content(~Element,bands)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry)) coverage))
