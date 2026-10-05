# Every entry the packer stores is the entry of the flat tabulation at its
# index. An entry index decodes into its tile and reduction index; a tile
# read lane by lane stores the tabulated lanes by definition; a tile inside
# one output row stores the same eight lanes because its positions share one
# output row; a tile that ends in the next row stores them because each lane
# lies in one of the two rows. The bounds are those of the guards of the
# problem.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./convolution_guard_facts.bend as Guards
import ./reduction_coordinate_bounds.bend as Coordinates
import ./supported_padded_coordinates.bend as Padded
import ./packing_tabulation.bend as PackingTabulation
import ./packing_loop_refinement.bend as LoopLaws
import ./packing_tile_equality.bend as Tiles
import ./packing_entry_coordinates.bend as Entry

def Supported(limits: Machine.Limits,p: ConvSpec.Problem) -> Data:
  Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>

def native_height(p: ConvSpec.Problem) -> U32:
  U32.from_nat(Guards.height(p))

def native_width(p: ConvSpec.Problem) -> U32:
  U32.from_nat(Guards.width(p))

def native_kernel(p: ConvSpec.Problem) -> U32:
  U32.from_nat(Guards.kernel(p))

def native_stride(p: ConvSpec.Problem) -> U32:
  U32.from_nat(Guards.stride(p))

def native_padding(p: ConvSpec.Problem) -> U32:
  U32.from_nat(Guards.pad(p))

def native_output_width(p: ConvSpec.Problem) -> U32:
  U32.from_nat(ConvSpec.out_width(p))

# Entry qn of tile g.
def entry_index(+p: ConvSpec.Problem,g: Nat,qn: Nat) -> Nat:
  Nat.add(Nat.mul(g,Coordinates.inner(p)),qn)

def index_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,+tiles: Nat,+g: Nat,+qn: Nat,
  total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  g_inside: {Nat.is_lt(g,tiles) == True{} : Bool},q_inside: {Nat.is_lt(qn,Coordinates.inner(p)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(entry_index(p,g,qn))) == entry_index(p,g,qn) : Nat}:
  Bounds.roundtrip(entry_index(p,g,qn),Machine.capacity(limits),Limits.capacity_fits(limits),
    Bounds.strict_le(entry_index(p,g,qn),U32.to_nat(Machine.capacity(limits)),
      Bounds.nat_trans(entry_index(p,g,qn),Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits)),
        RowMajor.radix(g,tiles,qn,Coordinates.inner(p),g_inside,q_inside),total)))

def index_group(+limits: Machine.Limits,+p: ConvSpec.Problem,+tiles: Nat,+g: Nat,+qn: Nat,+valid: Guards.Valid<p>,+support: Supported(limits,p),
  total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  g_inside: {Nat.is_lt(g,tiles) == True{} : Bool},+q_inside: {Nat.is_lt(qn,Coordinates.inner(p)) == True{} : Bool}) ->
  {(U32.from_nat(entry_index(p,g,qn)) / Coordinates.native_inner(p) : U32) == U32.from_nat(g) : U32}:
  Encoded.encoding((U32.from_nat(entry_index(p,g,qn)) / Coordinates.native_inner(p) : U32),g,
    Radix.quotient_word(U32.from_nat(entry_index(p,g,qn)),Coordinates.native_inner(p),g,qn,Coordinates.inner(p),Machine.half_word(limits),
      Limits.half_word_fits(limits),index_exact(limits,p,tiles,g,qn,total,g_inside,q_inside),Coordinates.inner_exact(limits,p,valid,support),
      Coordinates.inner_positive(p,valid),q_inside,Limits.double_bound(limits,Coordinates.inner(p),Coordinates.inner_bound(limits,p,valid,support))))

def index_reduction(+limits: Machine.Limits,+p: ConvSpec.Problem,+tiles: Nat,+g: Nat,+qn: Nat,+valid: Guards.Valid<p>,+support: Supported(limits,p),
  total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  g_inside: {Nat.is_lt(g,tiles) == True{} : Bool},+q_inside: {Nat.is_lt(qn,Coordinates.inner(p)) == True{} : Bool}) ->
  {(U32.from_nat(entry_index(p,g,qn)) % Coordinates.native_inner(p) : U32) == U32.from_nat(qn) : U32}:
  Encoded.encoding((U32.from_nat(entry_index(p,g,qn)) % Coordinates.native_inner(p) : U32),qn,
    Radix.remainder_word(U32.from_nat(entry_index(p,g,qn)),Coordinates.native_inner(p),g,qn,Coordinates.inner(p),Machine.half_word(limits),
      Limits.half_word_fits(limits),index_exact(limits,p,tiles,g,qn,total,g_inside,q_inside),Coordinates.inner_exact(limits,p,valid,support),
      Coordinates.inner_positive(p,valid),q_inside,Limits.double_bound(limits,Coordinates.inner(p),Coordinates.inner_bound(limits,p,valid,support))))

# The lanes tabulated at the position and reduction index of entry qn of tile g are its flat entry.
def crossing_entry(+limits: Machine.Limits,+p: ConvSpec.Problem,+tiles: Nat,+g: Nat,+qn: Nat,+start: U32,+position: U32,+x: Tree.Tree<F32>,
  +valid: Guards.Valid<p>,+support: Supported(limits,p),
  +total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +g_inside: {Nat.is_lt(g,tiles) == True{} : Bool},+q_inside: {Nat.is_lt(qn,Coordinates.inner(p)) == True{} : Bool},
  position_exact: {(start + U32.from_nat(g) * 8 : U32) == position : U32}) ->
  {PackingTabulation.vector(x,position,U32.from_nat(qn),native_height(p),native_width(p),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p)) ==
    PackingTabulation.entry(U32.from_nat(entry_index(p,g,qn)),start,x,native_height(p),native_width(p),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p),
      Coordinates.native_inner(p)) : Tile.Tile8}:
  +i = U32.from_nat(entry_index(p,g,qn))
  +inner = Coordinates.native_inner(p)
  %Equal.sym(U32,(i / inner : U32),U32.from_nat(g),index_group(limits,p,tiles,g,qn,valid,support,total,g_inside,q_inside)) :
    {PackingTabulation.vector(x,position,U32.from_nat(qn),native_height(p),native_width(p),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p)) ==
      PackingTabulation.vector(x,(start + _ * 8 : U32),(i % inner : U32),native_height(p),native_width(p),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p)) : Tile.Tile8}
  %Equal.sym(U32,(i % inner : U32),U32.from_nat(qn),index_reduction(limits,p,tiles,g,qn,valid,support,total,g_inside,q_inside)) :
    {PackingTabulation.vector(x,position,U32.from_nat(qn),native_height(p),native_width(p),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p)) ==
      PackingTabulation.vector(x,(start + U32.from_nat(g) * 8 : U32),_,native_height(p),native_width(p),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p)) : Tile.Tile8}
  %Equal.sym(U32,(start + U32.from_nat(g) * 8 : U32),position,position_exact) :
    {PackingTabulation.vector(x,position,U32.from_nat(qn),native_height(p),native_width(p),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p)) ==
      PackingTabulation.vector(x,_,U32.from_nat(qn),native_height(p),native_width(p),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p)) : Tile.Tile8}
  {==}

def lane(+limits: Machine.Limits,+p: ConvSpec.Problem,+position: U32,+j: Nat,+row_n: Nat,+col_n: Nat,+c: Nat,+kr: Nat,+kc: Nat,
  +valid: Guards.Valid<p>,+support: Supported(limits,p),
  +located: {U32.to_nat(position) == Nat.add(Nat.mul(row_n,ConvSpec.out_width(p)),col_n) : Nat},
  +column_inside: {Nat.is_lt(col_n,ConvSpec.out_width(p)) == True{} : Bool},
  within: {Nat.is_le(Nat.add(col_n,8n),ConvSpec.out_width(p)) == True{} : Bool},
  room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  j_lt: {Nat.is_lt(j,8n) == True{} : Bool},j_le: {Nat.is_le(j,8n) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool},+kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool},
  +kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool}) ->
  Tiles.LaneFacts<position,Entry.reduction_word(p,c,kr,kc),j,U32.from_nat(kr),U32.from_nat(kc),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p)>:
  Entry.lane_facts(limits,p,position,j,row_n,col_n,c,kr,kc,valid,support,located,column_inside,
    Bounds.nat_trans(Nat.add(col_n,j),Nat.add(col_n,8n),ConvSpec.out_width(p),Order.offset_lt(col_n,j,8n,j_lt),within),
    Order.transitive(Nat.add(U32.to_nat(position),j),Nat.add(U32.to_nat(position),8n),U32.to_nat(Machine.half_word(limits)),
      Order.offset_le(U32.to_nat(position),j,8n,j_le),room),
    c_inside,kr_inside,kc_inside)

# The block of a tile inside one output row, at reduction coordinates (c, kr, kc) of tile g, is its flat entry.
def row_entry(+limits: Machine.Limits,+p: ConvSpec.Problem,+tiles: Nat,+g: Nat,+c: Nat,+kr: Nat,+kc: Nat,+start: U32,+position: U32,+row_n: Nat,+col_n: Nat,
  +x: Tree.Tree<F32>,+valid: Guards.Valid<p>,+support: Supported(limits,p),
  +total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +g_inside: {Nat.is_lt(g,tiles) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool},+kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool},
  +kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool},
  position_exact: {(start + U32.from_nat(g) * 8 : U32) == position : U32},
  +located: {U32.to_nat(position) == Nat.add(Nat.mul(row_n,ConvSpec.out_width(p)),col_n) : Nat},
  +column_inside: {Nat.is_lt(col_n,ConvSpec.out_width(p)) == True{} : Bool},
  +within: {Nat.is_le(Nat.add(col_n,8n),ConvSpec.out_width(p)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {LoopLaws.block(x,(position / native_output_width(p) : U32),(position % native_output_width(p) : U32),U32.from_nat(c),U32.from_nat(kr),U32.from_nat(kc),
      native_height(p),native_width(p),native_stride(p),native_padding(p)) ==
    PackingTabulation.entry(U32.from_nat(entry_index(p,g,Coordinates.reduction(c,kr,kc,Guards.kernel(p)))),start,x,native_height(p),native_width(p),native_kernel(p),native_stride(p),
      native_padding(p),native_output_width(p),Coordinates.native_inner(p)) : Tile.Tile8}:
  Equal.trans(Tile.Tile8,
    LoopLaws.block(x,(position / native_output_width(p) : U32),(position % native_output_width(p) : U32),U32.from_nat(c),U32.from_nat(kr),U32.from_nat(kc),
      native_height(p),native_width(p),native_stride(p),native_padding(p)),
    PackingTabulation.vector(x,position,Entry.reduction_word(p,c,kr,kc),native_height(p),native_width(p),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p)),
    PackingTabulation.entry(U32.from_nat(entry_index(p,g,Coordinates.reduction(c,kr,kc,Guards.kernel(p)))),start,x,native_height(p),native_width(p),native_kernel(p),native_stride(p),
      native_padding(p),native_output_width(p),Coordinates.native_inner(p)),
    Tiles.tile_equal(x,position,Entry.reduction_word(p,c,kr,kc),U32.from_nat(c),U32.from_nat(kr),U32.from_nat(kc),native_height(p),native_width(p),native_kernel(p),native_stride(p),
      native_padding(p),native_output_width(p),Entry.channel_fact(limits,p,c,kr,kc,valid,support,c_inside,kr_inside,kc_inside),
      lane(limits,p,position,0n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,within,room,{==},{==},c_inside,kr_inside,kc_inside),
      lane(limits,p,position,1n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,within,room,{==},{==},c_inside,kr_inside,kc_inside),
      lane(limits,p,position,2n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,within,room,{==},{==},c_inside,kr_inside,kc_inside),
      lane(limits,p,position,3n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,within,room,{==},{==},c_inside,kr_inside,kc_inside),
      lane(limits,p,position,4n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,within,room,{==},{==},c_inside,kr_inside,kc_inside),
      lane(limits,p,position,5n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,within,room,{==},{==},c_inside,kr_inside,kc_inside),
      lane(limits,p,position,6n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,within,room,{==},{==},c_inside,kr_inside,kc_inside),
      lane(limits,p,position,7n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,within,room,{==},{==},c_inside,kr_inside,kc_inside)),
    crossing_entry(limits,p,tiles,g,Coordinates.reduction(c,kr,kc,Guards.kernel(p)),start,position,x,valid,support,total,g_inside,
      Coordinates.reduction_inside(p,c,kr,kc,c_inside,kr_inside,kc_inside),position_exact))

# The tile at position starts at output column to_nat(position) mod width. That column plus eight, as a word, is the natural sum.
def column_end(+limits: Machine.Limits,+p: ConvSpec.Problem,+position: U32,+valid: Guards.Valid<p>,+support: Supported(limits,p)) ->
  {U32.to_nat((position % native_output_width(p) + 8 : U32)) == Nat.add(Nat.mod(U32.to_nat(position),ConvSpec.out_width(p)),8n) : Nat}:
  +wn = ConvSpec.out_width(p)
  +uw = U32.from_nat(wn)
  +pn = U32.to_nat(position)
  +col_n = Nat.mod(pn,wn)
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +wp = Coordinates.width_positive(p)
  +small = DivisionOrder.remainder_small(pn,wn,wp)
  +column_exact = Radix.remainder_word(position,uw,Nat.div(pn,wn),col_n,wn,half,half_ok,DivisionOrder.quotient_exact(pn,wn,wp),Coordinates.width_exact(limits,p,valid,support),wp,small,
    Limits.double_bound(limits,wn,Coordinates.width_bound(limits,p,valid,support)))
  %column_exact : {U32.to_nat((position % uw + 8 : U32)) == Nat.add(_,8n) : Nat}
  Arithmetic.add((position % uw : U32),8,half,half_ok,
    %Equal.sym(Nat,U32.to_nat((position % uw : U32)),col_n,column_exact) : {Nat.is_le(Nat.add(_,8n),U32.to_nat(half)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(col_n,8n),1n+Nat.add(col_n,7n),Addition.successor_right(col_n,7n)) : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool}
    Padded.add_seven_bound(limits,1n+col_n,Order.transitive(1n+col_n,wn,U32.to_nat(Machine.capacity(limits)),Order.strict_follow(col_n,wn,small),
        Coordinates.width_bound(limits,p,valid,support))))

# The first native test says the eight positions of the tile stay in its output row.
def within_columns(+limits: Machine.Limits,+p: ConvSpec.Problem,+position: U32,+valid: Guards.Valid<p>,+support: Supported(limits,p),
  within: {U32.is_le((position % native_output_width(p) + 8 : U32),native_output_width(p)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),ConvSpec.out_width(p)),8n),ConvSpec.out_width(p)) == True{} : Bool}:
  %column_end(limits,p,position,valid,support) : {Nat.is_le(_,ConvSpec.out_width(p)) == True{} : Bool}
  %Coordinates.width_exact(limits,p,valid,support) : {Nat.is_le(U32.to_nat((position % native_output_width(p) + 8 : U32)),_) == True{} : Bool}
  %Comparison.le((position % native_output_width(p) + 8 : U32),native_output_width(p)) : {_ == True{} : Bool}
  within

def width_doubled(+limits: Machine.Limits,+p: ConvSpec.Problem,+valid: Guards.Valid<p>,+support: Supported(limits,p)) ->
  {U32.to_nat((native_output_width(p) + native_output_width(p) : U32)) == Nat.add(ConvSpec.out_width(p),ConvSpec.out_width(p)) : Nat}:
  +wn = ConvSpec.out_width(p)
  %Equal.sym(U32,(U32.from_nat(wn) + U32.from_nat(wn) : U32),U32.from_nat(Nat.add(wn,wn)),WordArithmetic.native_add(wn,wn)) : {U32.to_nat(_) == Nat.add(wn,wn) : Nat}
  Bounds.roundtrip(Nat.add(wn,wn),Machine.half_word(limits),Limits.half_word_fits(limits),
    %Equal.sym(Nat,Nat.add(wn,wn),Nat.double(wn),Addition.double(wn)) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    Limits.double_bound(limits,wn,Coordinates.width_bound(limits,p,valid,support)))

# The second native test says they stay in that row and the next.
def wraps_columns(+limits: Machine.Limits,+p: ConvSpec.Problem,+position: U32,+valid: Guards.Valid<p>,+support: Supported(limits,p),
  wraps: {U32.is_le((position % native_output_width(p) + 8 : U32),(native_output_width(p) + native_output_width(p) : U32)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),ConvSpec.out_width(p)),8n),Nat.add(ConvSpec.out_width(p),ConvSpec.out_width(p))) == True{} : Bool}:
  %column_end(limits,p,position,valid,support) : {Nat.is_le(_,Nat.add(ConvSpec.out_width(p),ConvSpec.out_width(p))) == True{} : Bool}
  %width_doubled(limits,p,valid,support) : {Nat.is_le(U32.to_nat((position % native_output_width(p) + 8 : U32)),_) == True{} : Bool}
  %Comparison.le((position % native_output_width(p) + 8 : U32),(native_output_width(p) + native_output_width(p) : U32)) : {_ == True{} : Bool}
  wraps

def first_inside(+col_n: Nat,+width: Nat,column_inside: {Nat.is_lt(col_n,width) == True{} : Bool}) -> {Nat.is_lt(Nat.add(col_n,0n),width) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(col_n,0n),col_n,Addition.zero_right(col_n)) : {Nat.is_lt(_,width) == True{} : Bool}
  column_inside

def lane_room(+limits: Machine.Limits,+position: U32,+j: Nat,j_le: {Nat.is_le(j,8n) == True{} : Bool},
  room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(position),j),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(Nat.add(U32.to_nat(position),j),Nat.add(U32.to_nat(position),8n),U32.to_nat(Machine.half_word(limits)),
    Order.offset_le(U32.to_nat(position),j,8n,j_le),room)

def wrap_lane(+limits: Machine.Limits,+p: ConvSpec.Problem,+position: U32,+j: Nat,+row_n: Nat,+col_n: Nat,+c: Nat,+kr: Nat,+kc: Nat,
  +valid: Guards.Valid<p>,+support: Supported(limits,p),
  +located: {U32.to_nat(position) == Nat.add(Nat.mul(row_n,ConvSpec.out_width(p)),col_n) : Nat},
  +column_inside: {Nat.is_lt(col_n,ConvSpec.out_width(p)) == True{} : Bool},
  +wraps: {Nat.is_le(Nat.add(col_n,8n),Nat.add(ConvSpec.out_width(p),ConvSpec.out_width(p))) == True{} : Bool},
  room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +j_lt: {Nat.is_lt(j,8n) == True{} : Bool},+j_last: {Nat.is_le(j,7n) == True{} : Bool},j_le: {Nat.is_le(j,8n) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool},+kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool},
  +kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool}) ->
  Tiles.WrapLane<position,Entry.reduction_word(p,c,kr,kc),j,U32.from_nat(kr),U32.from_nat(kc),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p)>:
  Entry.wrap_lane_facts(limits,p,position,j,row_n,col_n,c,kr,kc,Nat.is_lt(Nat.add(col_n,j),ConvSpec.out_width(p)),valid,support,located,column_inside,{==},wraps,
    lane_room(limits,position,j,j_le,room),j_lt,j_last,c_inside,kr_inside,kc_inside)

# The lanes of a tile that ends in its output row or the next, at reduction coordinates (c, kr, kc) of tile g, are its flat entry.
def wrap_entry(+limits: Machine.Limits,+p: ConvSpec.Problem,+tiles: Nat,+g: Nat,+c: Nat,+kr: Nat,+kc: Nat,+start: U32,+position: U32,+row_n: Nat,+col_n: Nat,
  +x: Tree.Tree<F32>,+valid: Guards.Valid<p>,+support: Supported(limits,p),
  +total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +g_inside: {Nat.is_lt(g,tiles) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool},+kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool},
  +kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool},
  position_exact: {(start + U32.from_nat(g) * 8 : U32) == position : U32},
  +located: {U32.to_nat(position) == Nat.add(Nat.mul(row_n,ConvSpec.out_width(p)),col_n) : Nat},
  +column_inside: {Nat.is_lt(col_n,ConvSpec.out_width(p)) == True{} : Bool},
  +wraps: {Nat.is_le(Nat.add(col_n,8n),Nat.add(ConvSpec.out_width(p),ConvSpec.out_width(p))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {LoopLaws.wrap_lanes(x,((position / native_output_width(p) : U32) * native_stride(p) + U32.from_nat(kr) - native_padding(p) : U32),
      ((position % native_output_width(p) : U32) * native_stride(p) + U32.from_nat(kc) - native_padding(p) : U32),(U32.from_nat(c) * native_height(p) * native_width(p) : U32),
      native_height(p),native_width(p),native_stride(p),(((position / native_output_width(p) : U32) + 1) * native_stride(p) + U32.from_nat(kr) - native_padding(p) : U32),
      (position % native_output_width(p) : U32),native_output_width(p),U32.from_nat(kc),native_padding(p)) ==
    PackingTabulation.entry(U32.from_nat(entry_index(p,g,Coordinates.reduction(c,kr,kc,Guards.kernel(p)))),start,x,native_height(p),native_width(p),native_kernel(p),native_stride(p),
      native_padding(p),native_output_width(p),Coordinates.native_inner(p)) : Tile.Tile8}:
  Equal.trans(Tile.Tile8,
    LoopLaws.wrap_lanes(x,((position / native_output_width(p) : U32) * native_stride(p) + U32.from_nat(kr) - native_padding(p) : U32),
      ((position % native_output_width(p) : U32) * native_stride(p) + U32.from_nat(kc) - native_padding(p) : U32),(U32.from_nat(c) * native_height(p) * native_width(p) : U32),
      native_height(p),native_width(p),native_stride(p),(((position / native_output_width(p) : U32) + 1) * native_stride(p) + U32.from_nat(kr) - native_padding(p) : U32),
      (position % native_output_width(p) : U32),native_output_width(p),U32.from_nat(kc),native_padding(p)),
    PackingTabulation.vector(x,position,Entry.reduction_word(p,c,kr,kc),native_height(p),native_width(p),native_kernel(p),native_stride(p),native_padding(p),native_output_width(p)),
    PackingTabulation.entry(U32.from_nat(entry_index(p,g,Coordinates.reduction(c,kr,kc,Guards.kernel(p)))),start,x,native_height(p),native_width(p),native_kernel(p),native_stride(p),
      native_padding(p),native_output_width(p),Coordinates.native_inner(p)),
    Tiles.wrap_tile_equal(x,position,Entry.reduction_word(p,c,kr,kc),U32.from_nat(c),U32.from_nat(kr),U32.from_nat(kc),native_height(p),native_width(p),native_kernel(p),native_stride(p),
      native_padding(p),native_output_width(p),Entry.channel_fact(limits,p,c,kr,kc,valid,support,c_inside,kr_inside,kc_inside),
      Entry.lane_facts(limits,p,position,0n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,first_inside(col_n,ConvSpec.out_width(p),column_inside),
        lane_room(limits,position,0n,{==},room),c_inside,kr_inside,kc_inside),
      wrap_lane(limits,p,position,1n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,wraps,room,{==},{==},{==},c_inside,kr_inside,kc_inside),
      wrap_lane(limits,p,position,2n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,wraps,room,{==},{==},{==},c_inside,kr_inside,kc_inside),
      wrap_lane(limits,p,position,3n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,wraps,room,{==},{==},{==},c_inside,kr_inside,kc_inside),
      wrap_lane(limits,p,position,4n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,wraps,room,{==},{==},{==},c_inside,kr_inside,kc_inside),
      wrap_lane(limits,p,position,5n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,wraps,room,{==},{==},{==},c_inside,kr_inside,kc_inside),
      wrap_lane(limits,p,position,6n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,wraps,room,{==},{==},{==},c_inside,kr_inside,kc_inside),
      wrap_lane(limits,p,position,7n,row_n,col_n,c,kr,kc,valid,support,located,column_inside,wraps,room,{==},{==},{==},c_inside,kr_inside,kc_inside)),
    crossing_entry(limits,p,tiles,g,Coordinates.reduction(c,kr,kc,Guards.kernel(p)),start,position,x,valid,support,total,g_inside,
      Coordinates.reduction_inside(p,c,kr,kc,c_inside,kr_inside,kc_inside),position_exact))
