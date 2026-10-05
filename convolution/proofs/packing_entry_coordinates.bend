# Coordinates of a packed entry of a supported problem, as machine words:
# the lanes of a tile inside one output row share its row and move along its
# columns, a lane past the last column lies in the next row, a reduction index
# decodes into channel, kernel row and kernel column, and an entry index
# decodes into its tile and reduction index. All bounds come from the guards
# of the problem.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/traversal_write_bounds.bend as WriteBounds
import ./supported_padded_coordinates.bend as Padded
import ./convolution_guard_facts.bend as Guards
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packing_tabulation.bend as PackingTabulation
import ./packing_tile_equality.bend as Tiles

def reduction_word(+p: ConvSpec.Problem,c: Nat,kr: Nat,kc: Nat) -> U32:
  U32.from_nat(Coordinates.reduction(c,kr,kc,Guards.kernel(p)))

def reduction_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,+c: Nat,+kr: Nat,+kc: Nat,valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool},kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool},
  kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool}) ->
  {U32.to_nat(reduction_word(p,c,kr,kc)) == Coordinates.reduction(c,kr,kc,Guards.kernel(p)) : Nat}:
  Bounds.roundtrip(Coordinates.reduction(c,kr,kc,Guards.kernel(p)),Machine.capacity(limits),Limits.capacity_fits(limits),
    Bounds.strict_le(Coordinates.reduction(c,kr,kc,Guards.kernel(p)),U32.to_nat(Machine.capacity(limits)),
      Bounds.nat_trans(Coordinates.reduction(c,kr,kc,Guards.kernel(p)),Coordinates.inner(p),U32.to_nat(Machine.capacity(limits)),
        Coordinates.reduction_inside(p,c,kr,kc,c_inside,kr_inside,kc_inside),Coordinates.inner_bound(limits,p,valid,support))))

def channel_of(+q: U32,+k: U32,+c: Nat,+kr: Nat,+kc: Nat,+h: U32,+w: U32,decoded: Coordinates.DecodedReduction<q,k,c,kr,kc>) ->
  {((q / (k * k)) * h * w : U32) == (U32.from_nat(c) * h * w : U32) : U32}:
  Coordinates.DecodedReduction{channel,row,column} = decoded
  %Equal.sym(U32,(q / (k * k) : U32),U32.from_nat(c),channel) : {(_ * h * w : U32) == (U32.from_nat(c) * h * w : U32) : U32}
  {==}

def row_of(+position: U32,+q: U32,+j: Nat,+row_n: Nat,+c: Nat,+kr: Nat,+kc: Nat,+k: U32,+s: U32,+p: U32,+ow: U32,
  decoded: Coordinates.DecodedReduction<q,k,c,kr,kc>,
  lane_row: {(PackingTabulation.indexed_position(position,j) / ow : U32) == U32.from_nat(row_n) : U32},
  tile_row: {(position / ow : U32) == U32.from_nat(row_n) : U32}) ->
  {((PackingTabulation.indexed_position(position,j) / ow) * s + (q / k) % k - p : U32) == ((position / ow : U32) * s + U32.from_nat(kr) - p : U32) : U32}:
  Coordinates.DecodedReduction{channel,row,column} = decoded
  %Equal.sym(U32,(PackingTabulation.indexed_position(position,j) / ow : U32),U32.from_nat(row_n),lane_row) :
    {(_ * s + (q / k) % k - p : U32) == ((position / ow : U32) * s + U32.from_nat(kr) - p : U32) : U32}
  %Equal.sym(U32,(position / ow : U32),U32.from_nat(row_n),tile_row) : {(U32.from_nat(row_n) * s + (q / k) % k - p : U32) == (_ * s + U32.from_nat(kr) - p : U32) : U32}
  %Equal.sym(U32,(q / k % k : U32),U32.from_nat(kr),row) : {(U32.from_nat(row_n) * s + _ - p : U32) == (U32.from_nat(row_n) * s + U32.from_nat(kr) - p : U32) : U32}
  {==}

def column_of(+position: U32,+q: U32,+j: Nat,+col_n: Nat,+c: Nat,+kr: Nat,+kc: Nat,+k: U32,+s: U32,+p: U32,+ow: U32,
  decoded: Coordinates.DecodedReduction<q,k,c,kr,kc>,
  lane_column: {(PackingTabulation.indexed_position(position,j) % ow : U32) == U32.from_nat(Nat.add(col_n,j)) : U32},
  tile_column: {(position % ow : U32) == U32.from_nat(col_n) : U32},
  shifted: {((U32.from_nat(col_n) * s + U32.from_nat(kc) - p : U32) + U32.from_nat(j) * s : U32) == (U32.from_nat(Nat.add(col_n,j)) * s + U32.from_nat(kc) - p : U32) : U32}) ->
  {((PackingTabulation.indexed_position(position,j) % ow) * s + q % k - p : U32) == (((position % ow : U32) * s + U32.from_nat(kc) - p : U32) + U32.from_nat(j) * s : U32) : U32}:
  Coordinates.DecodedReduction{channel,row,column} = decoded
  %Equal.sym(U32,(PackingTabulation.indexed_position(position,j) % ow : U32),U32.from_nat(Nat.add(col_n,j)),lane_column) :
    {(_ * s + q % k - p : U32) == (((position % ow : U32) * s + U32.from_nat(kc) - p : U32) + U32.from_nat(j) * s : U32) : U32}
  %Equal.sym(U32,(position % ow : U32),U32.from_nat(col_n),tile_column) :
    {(U32.from_nat(Nat.add(col_n,j)) * s + q % k - p : U32) == ((_ * s + U32.from_nat(kc) - p : U32) + U32.from_nat(j) * s : U32) : U32}
  %Equal.sym(U32,(q % k : U32),U32.from_nat(kc),column) :
    {(U32.from_nat(Nat.add(col_n,j)) * s + _ - p : U32) == ((U32.from_nat(col_n) * s + U32.from_nat(kc) - p : U32) + U32.from_nat(j) * s : U32) : U32}
  Equal.sym(U32,((U32.from_nat(col_n) * s + U32.from_nat(kc) - p : U32) + U32.from_nat(j) * s : U32),(U32.from_nat(Nat.add(col_n,j)) * s + U32.from_nat(kc) - p : U32),shifted)

def indexed_encoded(+position: Nat,index: Nat) -> {PackingTabulation.indexed_position(U32.from_nat(position),index) == U32.from_nat(Nat.add(position,index)) : U32}:
  match index:
    case 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) : {U32.from_nat(position) == U32.from_nat(_) : U32}
      {==}
    case 1n+rest:
      WordArithmetic.native_add(position,1n+rest)

# Lane j of the tile at position is output position row_n * width + col_n + j.
def lane_located(+limits: Machine.Limits,+position: U32,+j: Nat,+row_n: Nat,+col_n: Nat,+width: Nat,
  located: {U32.to_nat(position) == Nat.add(Nat.mul(row_n,width),col_n) : Nat},
  room: {Nat.is_le(Nat.add(U32.to_nat(position),j),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.to_nat(PackingTabulation.indexed_position(position,j)) == Nat.add(Nat.mul(row_n,width),Nat.add(col_n,j)) : Nat}:
  %Equal.sym(U32,position,U32.from_nat(U32.to_nat(position)),Encoded.encoding(position,U32.to_nat(position),{==})) :
    {U32.to_nat(PackingTabulation.indexed_position(_,j)) == Nat.add(Nat.mul(row_n,width),Nat.add(col_n,j)) : Nat}
  %Equal.sym(U32,PackingTabulation.indexed_position(U32.from_nat(U32.to_nat(position)),j),U32.from_nat(Nat.add(U32.to_nat(position),j)),indexed_encoded(U32.to_nat(position),j)) :
    {U32.to_nat(_) == Nat.add(Nat.mul(row_n,width),Nat.add(col_n,j)) : Nat}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Nat.add(U32.to_nat(position),j))),Nat.add(U32.to_nat(position),j),
    Bounds.roundtrip(Nat.add(U32.to_nat(position),j),Machine.half_word(limits),Limits.half_word_fits(limits),room)) :
    {_ == Nat.add(Nat.mul(row_n,width),Nat.add(col_n,j)) : Nat}
  %Equal.sym(Nat,U32.to_nat(position),Nat.add(Nat.mul(row_n,width),col_n),located) : {Nat.add(_,j) == Nat.add(Nat.mul(row_n,width),Nat.add(col_n,j)) : Nat}
  Addition.associative(Nat.mul(row_n,width),col_n,j)

# The row and column equations of lane j of a tile inside one output row.
def lane_facts(+limits: Machine.Limits,+p: ConvSpec.Problem,+position: U32,+j: Nat,+row_n: Nat,+col_n: Nat,+c: Nat,+kr: Nat,+kc: Nat,
  +valid: Guards.Valid<p>,+support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +located: {U32.to_nat(position) == Nat.add(Nat.mul(row_n,ConvSpec.out_width(p)),col_n) : Nat},
  +column_inside: {Nat.is_lt(col_n,ConvSpec.out_width(p)) == True{} : Bool},
  +lane_inside: {Nat.is_lt(Nat.add(col_n,j),ConvSpec.out_width(p)) == True{} : Bool},
  room: {Nat.is_le(Nat.add(U32.to_nat(position),j),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool},+kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool},
  +kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool}) ->
  Tiles.LaneFacts<position,reduction_word(p,c,kr,kc),j,U32.from_nat(kr),U32.from_nat(kc),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
    U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p))>:
  +wn = ConvSpec.out_width(p)
  +uw = U32.from_nat(wn)
  +uk = U32.from_nat(Guards.kernel(p))
  +us = U32.from_nat(Guards.stride(p))
  +up = U32.from_nat(Guards.pad(p))
  +q = reduction_word(p,c,kr,kc)
  +pj = PackingTabulation.indexed_position(position,j)
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +we = Coordinates.width_exact(limits,p,valid,support)
  +wp = Coordinates.width_positive(p)
  +wd = Limits.double_bound(limits,wn,Coordinates.width_bound(limits,p,valid,support))
  +lane_index = lane_located(limits,position,j,row_n,col_n,wn,located,room)
  +decoded = Coordinates.decode_reduction(limits,p,q,c,kr,kc,valid,support,reduction_exact(limits,p,c,kr,kc,valid,support,c_inside,kr_inside,kc_inside),kr_inside,kc_inside)
  +coordinate = Machine.coordinate(limits)
  Tiles.LaneFacts{
    row_of(position,q,j,row_n,c,kr,kc,uk,us,up,uw,decoded,
      Encoded.encoding((pj / uw : U32),row_n,Radix.quotient_word(pj,uw,row_n,Nat.add(col_n,j),wn,half,half_ok,lane_index,we,wp,lane_inside,wd)),
      Encoded.encoding((position / uw : U32),row_n,Radix.quotient_word(position,uw,row_n,col_n,wn,half,half_ok,located,we,wp,column_inside,wd))),
    column_of(position,q,j,col_n,c,kr,kc,uk,us,up,uw,decoded,
      Encoded.encoding((pj % uw : U32),Nat.add(col_n,j),Radix.remainder_word(pj,uw,row_n,Nat.add(col_n,j),wn,half,half_ok,lane_index,we,wp,lane_inside,wd)),
      Encoded.encoding((position % uw : U32),col_n,Radix.remainder_word(position,uw,row_n,col_n,wn,half,half_ok,located,we,wp,column_inside,wd)),
      Encoded.padded_scaled_offset(col_n,j,Guards.stride(p),kc,up,coordinate,Limits.coordinate_fits(limits),
        Bounds.strict_le(Nat.add(col_n,j),U32.to_nat(coordinate),Bounds.nat_trans(Nat.add(col_n,j),wn,U32.to_nat(coordinate),lane_inside,SupportedCoordinateBounds.col_limit(limits,p,valid,support))),
        SupportedCoordinateBounds.stride_large(limits,p,support),
        Bounds.strict_le(kc,U32.to_nat(coordinate),Bounds.nat_trans(kc,Guards.kernel(p),U32.to_nat(coordinate),kc_inside,SupportedCoordinateBounds.kernel_large(limits,p,support))),
        Encoded.rectangle(Nat.add(col_n,j),wn,Guards.stride(p),kc,Guards.kernel(p),U32.to_nat(coordinate),lane_inside,kc_inside,SupportedCoordinateBounds.col_envelope(limits,p,support))))}

# The output column of lane j, before the test against the width, fits a word.
def lane_room(+limits: Machine.Limits,+p: ConvSpec.Problem,+j: Nat,+col_n: Nat,+valid: Guards.Valid<p>,+support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +column_inside: {Nat.is_lt(col_n,ConvSpec.out_width(p)) == True{} : Bool},+j_le: {Nat.is_le(j,7n) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(col_n,j),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +capacity = U32.to_nat(Machine.capacity(limits))
  Order.transitive(Nat.add(col_n,j),Nat.add(col_n,7n),U32.to_nat(Machine.half_word(limits)),Order.offset_le(col_n,j,7n,j_le),
    Padded.add_seven_bound(limits,col_n,Bounds.strict_le(col_n,capacity,
      Bounds.nat_trans(col_n,ConvSpec.out_width(p),capacity,column_inside,Coordinates.width_bound(limits,p,valid,support)))))

def lane_column(+limits: Machine.Limits,+p: ConvSpec.Problem,+position: U32,+j: Nat,+row_n: Nat,+col_n: Nat,+valid: Guards.Valid<p>,+support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +located: {U32.to_nat(position) == Nat.add(Nat.mul(row_n,ConvSpec.out_width(p)),col_n) : Nat},+column_inside: {Nat.is_lt(col_n,ConvSpec.out_width(p)) == True{} : Bool},+j_le: {Nat.is_le(j,7n) == True{} : Bool}) ->
  {U32.to_nat(((position % U32.from_nat(ConvSpec.out_width(p)) : U32) + U32.from_nat(j) : U32)) == Nat.add(col_n,j) : Nat}:
  +wn = ConvSpec.out_width(p)
  +uw = U32.from_nat(wn)
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  %Equal.sym(U32,(position % uw : U32),U32.from_nat(col_n),Encoded.encoding((position % uw : U32),col_n,
      Radix.remainder_word(position,uw,row_n,col_n,wn,half,half_ok,located,Coordinates.width_exact(limits,p,valid,support),Coordinates.width_positive(p),column_inside,
        Limits.double_bound(limits,wn,Coordinates.width_bound(limits,p,valid,support))))) :
    {U32.to_nat((_ + U32.from_nat(j) : U32)) == Nat.add(col_n,j) : Nat}
  %Equal.sym(U32,(U32.from_nat(col_n) + U32.from_nat(j) : U32),U32.from_nat(Nat.add(col_n,j)),WordArithmetic.native_add(col_n,j)) : {U32.to_nat(_) == Nat.add(col_n,j) : Nat}
  Bounds.roundtrip(Nat.add(col_n,j),half,half_ok,lane_room(limits,p,j,col_n,valid,support,column_inside,j_le))

# The native test of lane j is the comparison of its output column with the width.
def lane_test(+limits: Machine.Limits,+p: ConvSpec.Problem,+position: U32,+j: Nat,+row_n: Nat,+col_n: Nat,+inside: Bool,+valid: Guards.Valid<p>,+support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +located: {U32.to_nat(position) == Nat.add(Nat.mul(row_n,ConvSpec.out_width(p)),col_n) : Nat},+column_inside: {Nat.is_lt(col_n,ConvSpec.out_width(p)) == True{} : Bool},+j_le: {Nat.is_le(j,7n) == True{} : Bool},
  decided: {Nat.is_lt(Nat.add(col_n,j),ConvSpec.out_width(p)) == inside : Bool}) ->
  {U32.is_lt(((position % U32.from_nat(ConvSpec.out_width(p)) : U32) + U32.from_nat(j) : U32),U32.from_nat(ConvSpec.out_width(p))) == inside : Bool}:
  %Equal.sym(Bool,U32.is_lt(((position % U32.from_nat(ConvSpec.out_width(p)) : U32) + U32.from_nat(j) : U32),U32.from_nat(ConvSpec.out_width(p))),Nat.is_lt(U32.to_nat(((position % U32.from_nat(ConvSpec.out_width(p)) : U32) + U32.from_nat(j) : U32)),U32.to_nat(U32.from_nat(ConvSpec.out_width(p)))),Comparison.lt(((position % U32.from_nat(ConvSpec.out_width(p)) : U32) + U32.from_nat(j) : U32),U32.from_nat(ConvSpec.out_width(p)))) : {_ == inside : Bool}
  %Equal.sym(Nat,U32.to_nat(((position % U32.from_nat(ConvSpec.out_width(p)) : U32) + U32.from_nat(j) : U32)),Nat.add(col_n,j),lane_column(limits,p,position,j,row_n,col_n,valid,support,located,column_inside,j_le)) :
    {Nat.is_lt(_,U32.to_nat(U32.from_nat(ConvSpec.out_width(p)))) == inside : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(p))),ConvSpec.out_width(p),Coordinates.width_exact(limits,p,valid,support)) : {Nat.is_lt(Nat.add(col_n,j),_) == inside : Bool}
  decided

def next_row_index(+row_n: Nat,+width: Nat,+sum: Nat,+m: Nat,split: {Nat.add(width,m) == sum : Nat}) ->
  {Nat.add(Nat.mul(row_n,width),sum) == Nat.add(Nat.mul(1n+row_n,width),m) : Nat}:
  %split : {Nat.add(Nat.mul(row_n,width),_) == Nat.add(Nat.mul(1n+row_n,width),m) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(width,Nat.mul(row_n,width)),m),Nat.add(width,Nat.add(Nat.mul(row_n,width),m)),Addition.associative(width,Nat.mul(row_n,width),m)) :
    {Nat.add(Nat.mul(row_n,width),Nat.add(width,m)) == _ : Nat}
  RowMajor.add_swap(Nat.mul(row_n,width),width,m)

# Lane j past the last column is output position (row_n + 1) * width + m.
def next_located(+limits: Machine.Limits,+position: U32,+j: Nat,+row_n: Nat,+col_n: Nat,+width: Nat,+m: Nat,
  located: {U32.to_nat(position) == Nat.add(Nat.mul(row_n,width),col_n) : Nat},
  room: {Nat.is_le(Nat.add(U32.to_nat(position),j),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  split: {Nat.add(width,m) == Nat.add(col_n,j) : Nat}) ->
  {U32.to_nat(PackingTabulation.indexed_position(position,j)) == Nat.add(Nat.mul(1n+row_n,width),m) : Nat}:
  Equal.trans(Nat,U32.to_nat(PackingTabulation.indexed_position(position,j)),Nat.add(Nat.mul(row_n,width),Nat.add(col_n,j)),Nat.add(Nat.mul(1n+row_n,width),m),
    lane_located(limits,position,j,row_n,col_n,width,located,room),next_row_index(row_n,width,Nat.add(col_n,j),m,split))

def next_row_of(+position: U32,+q: U32,+j: Nat,+row_n: Nat,+c: Nat,+kr: Nat,+kc: Nat,+k: U32,+s: U32,+p: U32,+ow: U32,
  decoded: Coordinates.DecodedReduction<q,k,c,kr,kc>,
  lane_row: {(PackingTabulation.indexed_position(position,j) / ow : U32) == U32.from_nat(1n+row_n) : U32},
  tile_row: {(position / ow : U32) == U32.from_nat(row_n) : U32}) ->
  {((PackingTabulation.indexed_position(position,j) / ow) * s + (q / k) % k - p : U32) == (((position / ow : U32) + 1) * s + U32.from_nat(kr) - p : U32) : U32}:
  Coordinates.DecodedReduction{channel,row,column} = decoded
  %Equal.sym(U32,(PackingTabulation.indexed_position(position,j) / ow : U32),U32.from_nat(1n+row_n),lane_row) :
    {(_ * s + (q / k) % k - p : U32) == (((position / ow : U32) + 1) * s + U32.from_nat(kr) - p : U32) : U32}
  %Equal.sym(U32,(position / ow : U32),U32.from_nat(row_n),tile_row) : {(U32.from_nat(1n+row_n) * s + (q / k) % k - p : U32) == ((_ + 1) * s + U32.from_nat(kr) - p : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(row_n) + 1 : U32),U32.from_nat(1n+row_n),WriteBounds.increment_encoded(row_n)) :
    {(U32.from_nat(1n+row_n) * s + (q / k) % k - p : U32) == (_ * s + U32.from_nat(kr) - p : U32) : U32}
  %Equal.sym(U32,(q / k % k : U32),U32.from_nat(kr),row) : {(U32.from_nat(1n+row_n) * s + _ - p : U32) == (U32.from_nat(1n+row_n) * s + U32.from_nat(kr) - p : U32) : U32}
  {==}

def next_column_of(+position: U32,+q: U32,+j: Nat,+m: Nat,+c: Nat,+kr: Nat,+kc: Nat,+k: U32,+s: U32,+p: U32,+ow: U32,
  decoded: Coordinates.DecodedReduction<q,k,c,kr,kc>,
  lane_column: {(PackingTabulation.indexed_position(position,j) % ow : U32) == U32.from_nat(m) : U32},
  wrapped: {((position % ow : U32) + U32.from_nat(j) - ow : U32) == U32.from_nat(m) : U32}) ->
  {((PackingTabulation.indexed_position(position,j) % ow) * s + q % k - p : U32) == (((position % ow : U32) + U32.from_nat(j) - ow) * s + U32.from_nat(kc) - p : U32) : U32}:
  Coordinates.DecodedReduction{channel,row,column} = decoded
  %Equal.sym(U32,(PackingTabulation.indexed_position(position,j) % ow : U32),U32.from_nat(m),lane_column) :
    {(_ * s + q % k - p : U32) == (((position % ow : U32) + U32.from_nat(j) - ow) * s + U32.from_nat(kc) - p : U32) : U32}
  %Equal.sym(U32,((position % ow : U32) + U32.from_nat(j) - ow : U32),U32.from_nat(m),wrapped) : {(U32.from_nat(m) * s + q % k - p : U32) == (_ * s + U32.from_nat(kc) - p : U32) : U32}
  %Equal.sym(U32,(q % k : U32),U32.from_nat(kc),column) : {(U32.from_nat(m) * s + _ - p : U32) == (U32.from_nat(m) * s + U32.from_nat(kc) - p : U32) : U32}
  {==}

def prefix_exact(+value: U32,+base: U32,+width: Nat,+sum: Nat,+m: Nat,exact: {U32.to_nat(value) == sum : Nat},size: {U32.to_nat(base) == width : Nat},
  split: {Nat.add(width,m) == sum : Nat}) -> {U32.to_nat(value) == Nat.add(U32.to_nat(base),m) : Nat}:
  %Equal.sym(Nat,U32.to_nat(base),width,size) : {U32.to_nat(value) == Nat.add(_,m) : Nat}
  %Equal.sym(Nat,Nat.add(width,m),sum,split) : {U32.to_nat(value) == _ : Nat}
  exact

def prefix_room(+value: U32,+sum: Nat,+cap: Nat,exact: {U32.to_nat(value) == sum : Nat},bound: {Nat.is_le(sum,cap) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(value),cap) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(value),sum,exact) : {Nat.is_le(_,cap) == True{} : Bool}
  bound

# The equations of a lane past the last output column: it lies in the next output row.
def next_lane(+limits: Machine.Limits,+p: ConvSpec.Problem,+position: U32,+j: Nat,+row_n: Nat,+col_n: Nat,+c: Nat,+kr: Nat,+kc: Nat,
  +valid: Guards.Valid<p>,+support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +located: {U32.to_nat(position) == Nat.add(Nat.mul(row_n,ConvSpec.out_width(p)),col_n) : Nat},
  +column_inside: {Nat.is_lt(col_n,ConvSpec.out_width(p)) == True{} : Bool},
  +outside: {Nat.is_lt(Nat.add(col_n,j),ConvSpec.out_width(p)) == False{} : Bool},
  +wraps: {Nat.is_le(Nat.add(col_n,8n),Nat.add(ConvSpec.out_width(p),ConvSpec.out_width(p))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),j),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +j_lt: {Nat.is_lt(j,8n) == True{} : Bool},+j_le: {Nat.is_le(j,7n) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool},+kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool},
  +kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool}) ->
  Tiles.WrapLane<position,reduction_word(p,c,kr,kc),j,U32.from_nat(kr),U32.from_nat(kc),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
    U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p))>:
  +wn = ConvSpec.out_width(p)
  +uw = U32.from_nat(wn)
  +uk = U32.from_nat(Guards.kernel(p))
  +us = U32.from_nat(Guards.stride(p))
  +up = U32.from_nat(Guards.pad(p))
  +q = reduction_word(p,c,kr,kc)
  +pj = PackingTabulation.indexed_position(position,j)
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +we = Coordinates.width_exact(limits,p,valid,support)
  +wp = Coordinates.width_positive(p)
  +wd = Limits.double_bound(limits,wn,Coordinates.width_bound(limits,p,valid,support))
  +sum = Nat.add(col_n,j)
  +m = Nat.sub(sum,wn)
  +above = Order.ge_true(sum,wn,Order.comparison_not_less(Nat.cmp(sum,wn),outside))
  +split = Order.restore_shift(wn,sum,above)
  +m_inside = Order.sub_lt(sum,wn,wn,above,Bounds.nat_trans(sum,Nat.add(col_n,8n),Nat.add(wn,wn),Order.offset_lt(col_n,j,8n,j_lt),wraps))
  +lane_index = next_located(limits,position,j,row_n,col_n,wn,m,located,room,split)
  +decoded = Coordinates.decode_reduction(limits,p,q,c,kr,kc,valid,support,reduction_exact(limits,p,c,kr,kc,valid,support,c_inside,kr_inside,kc_inside),kr_inside,kc_inside)
  +word = lane_column(limits,p,position,j,row_n,col_n,valid,support,located,column_inside,j_le)
  Tiles.Next{
    lane_test(limits,p,position,j,row_n,col_n,False{},valid,support,located,column_inside,j_le,outside),
    next_row_of(position,q,j,row_n,c,kr,kc,uk,us,up,uw,decoded,
      Encoded.encoding((pj / uw : U32),1n+row_n,Radix.quotient_word(pj,uw,1n+row_n,m,wn,half,half_ok,lane_index,we,wp,m_inside,wd)),
      Encoded.encoding((position / uw : U32),row_n,Radix.quotient_word(position,uw,row_n,col_n,wn,half,half_ok,located,we,wp,column_inside,wd))),
    next_column_of(position,q,j,m,c,kr,kc,uk,us,up,uw,decoded,
      Encoded.encoding((pj % uw : U32),m,Radix.remainder_word(pj,uw,1n+row_n,m,wn,half,half_ok,lane_index,we,wp,m_inside,wd)),
      Encoded.encoding(((position % uw : U32) + U32.from_nat(j) - uw : U32),m,
        Arithmetic.remove_prefix(((position % uw : U32) + U32.from_nat(j) : U32),uw,m,half,half_ok,
          prefix_exact(((position % uw : U32) + U32.from_nat(j) : U32),uw,wn,sum,m,word,we,split),
          prefix_room(((position % uw : U32) + U32.from_nat(j) : U32),sum,U32.to_nat(half),word,lane_room(limits,p,j,col_n,valid,support,column_inside,j_le)))))}

# The equations of lane j of a tile that ends in its output row or the next.
def wrap_lane_facts(+limits: Machine.Limits,+p: ConvSpec.Problem,+position: U32,+j: Nat,+row_n: Nat,+col_n: Nat,+c: Nat,+kr: Nat,+kc: Nat,+inside: Bool,
  +valid: Guards.Valid<p>,+support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +located: {U32.to_nat(position) == Nat.add(Nat.mul(row_n,ConvSpec.out_width(p)),col_n) : Nat},
  +column_inside: {Nat.is_lt(col_n,ConvSpec.out_width(p)) == True{} : Bool},
  +decided: {Nat.is_lt(Nat.add(col_n,j),ConvSpec.out_width(p)) == inside : Bool},
  +wraps: {Nat.is_le(Nat.add(col_n,8n),Nat.add(ConvSpec.out_width(p),ConvSpec.out_width(p))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),j),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +j_lt: {Nat.is_lt(j,8n) == True{} : Bool},+j_le: {Nat.is_le(j,7n) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool},+kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool},
  +kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool}) ->
  Tiles.WrapLane<position,reduction_word(p,c,kr,kc),j,U32.from_nat(kr),U32.from_nat(kc),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
    U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p))>:
  match inside:
    case True{}:
      Tiles.Same{lane_test(limits,p,position,j,row_n,col_n,True{},valid,support,located,column_inside,j_le,decided),
        lane_facts(limits,p,position,j,row_n,col_n,c,kr,kc,valid,support,located,column_inside,decided,room,c_inside,kr_inside,kc_inside)}
    case False{}:
      next_lane(limits,p,position,j,row_n,col_n,c,kr,kc,valid,support,located,column_inside,decided,wraps,room,j_lt,j_le,c_inside,kr_inside,kc_inside)

# The channel equation of a reduction index.
def channel_fact(+limits: Machine.Limits,+p: ConvSpec.Problem,+c: Nat,+kr: Nat,+kc: Nat,
  +valid: Guards.Valid<p>,+support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool},+kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool},
  +kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool}) ->
  {((reduction_word(p,c,kr,kc) / (U32.from_nat(Guards.kernel(p)) * U32.from_nat(Guards.kernel(p)))) * U32.from_nat(Guards.height(p)) * U32.from_nat(Guards.width(p)) : U32) ==
    (U32.from_nat(c) * U32.from_nat(Guards.height(p)) * U32.from_nat(Guards.width(p)) : U32) : U32}:
  channel_of(reduction_word(p,c,kr,kc),U32.from_nat(Guards.kernel(p)),c,kr,kc,U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),
    Coordinates.decode_reduction(limits,p,reduction_word(p,c,kr,kc),c,kr,kc,valid,support,reduction_exact(limits,p,c,kr,kc,valid,support,c_inside,kr_inside,kc_inside),kr_inside,kc_inside))
