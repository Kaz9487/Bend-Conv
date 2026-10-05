# The packer model stores the flat tabulation: its loops write the entries
# of consecutive indices, and every entry is the tabulation entry of its
# index (packing_entries). Counters and indices are followed as naturals
# that advance by one per entry. The nested loops are proved once for any
# lanes of an entry whose rule says they are the tabulation entry.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/traversal_loop.bend as Loop
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/traversal_write_bounds.bend as WriteBounds
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./convolution_guard_facts.bend as Guards
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packing_tabulation.bend as PackingTabulation
import ./packing_loop_refinement.bend as LoopLaws
import ./packing_tile_equality.bend as Tiles
import ./packing_entries.bend as Entries
import ./packing_storage_refinement.bend as PackingStorage
import ./concrete_support_guard.bend as Support
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/binary_radix.bend as Radix
import ./supported_padded_coordinates.bend as Padded

# k after n steps of one.
def advance(n: Nat,k: Nat) -> Nat:
  match n:
    case 0n: k
    case 1n+rest: advance(rest,1n+k)

def end_encoded(+n: Nat,+index: Nat) -> {PackingTabulation.end(n,U32.from_nat(index)) == U32.from_nat(advance(n,index)) : U32}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),WriteBounds.increment_encoded(index)) :
        {PackingTabulation.end(rest,_) == U32.from_nat(advance(rest,1n+index)) : U32}
      end_encoded(rest,1n+index)

def shift(+m: Nat,+a: Nat,+b: Nat,e: {a == Nat.add(m,b) : Nat}) -> {1n+a == Nat.add(m,1n+b) : Nat}:
  %Equal.sym(Nat,Nat.add(m,1n+b),1n+Nat.add(m,b),Addition.successor_right(m,b)) : {1n+a == _ : Nat}
  Equal.cong(Nat,Nat,n => 1n+n,a,Nat.add(m,b),e)

def filled(+p: ConvSpec.Problem,n: Nat,+i: U32,+start: U32,+x: Tree.Tree<F32>,y: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  PackingTabulation.fill(n,i,start,x,y,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
    Entries.native_output_width(p),Coordinates.native_inner(p))

def state(+p: ConvSpec.Problem,x: Tree.Tree<F32>,y: Tree.Tree<Tile.Tile8>,i: U32,c: U32,kr: U32,kc: U32,+position: U32) -> LoopLaws.RowTiles():
  Packing.RowTile{x,y,i,c,kr,kc,(position / Entries.native_output_width(p) : U32),(position % Entries.native_output_width(p) : U32),Entries.native_output_width(p),
    Entries.native_height(p),Entries.native_width(p),Entries.native_stride(p),Entries.native_padding(p),Guards.kernel(p)}

# The output columns the eight positions of a tile may reach: those of its output row, or of two rows.
def one_row(width: Nat) -> Nat:
  width

def two_rows(width: Nat) -> Nat:
  +width = width
  Nat.add(width,width)

# What a tile read by its rows needs: tile g of tiles starts at position, and its eight positions stay within span output columns of its row.
type TileFacts<-limits: Machine.Limits,-p: ConvSpec.Problem,-tiles: Nat,-g: Nat,-start: U32,-position: U32,-span: Nat> is Data:
  TileFacts{valid: Guards.Valid<p>,support: Entries.Supported(limits,p),
    total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    g_inside: {Nat.is_lt(g,tiles) == True{} : Bool},
    position_exact: {(start + U32.from_nat(g) * 8 : U32) == position : U32},
    within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),ConvSpec.out_width(p)),8n),span) == True{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}

# The lanes of the entry at reduction coordinates (c, kr, kc) of such a tile are the tabulation entry of its index.
def EntryRule(~lanes: LoopLaws.Lanes(),~span: Nat -> Nat) -> Type:
  @+limits: Machine.Limits -> @+p: ConvSpec.Problem -> @+tiles: Nat -> @+g: Nat -> @+c: Nat -> @+kr: Nat -> @+kc: Nat -> @+qn: Nat -> @+index: Nat -> @+start: U32 -> @+position: U32 ->
  @+x: Tree.Tree<F32> -> @facts: TileFacts<limits,p,tiles,g,start,position,span(ConvSpec.out_width(p))> ->
  @reduction: {qn == Coordinates.reduction(c,kr,kc,Guards.kernel(p)) : Nat} -> @indexed: {index == Entries.entry_index(p,g,qn) : Nat} ->
  @+c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool} -> @+kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool} ->
  @+kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool} ->
  {lanes(x,((position / Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),((position % Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kc) - Entries.native_padding(p) : U32),
      (U32.from_nat(c) * Entries.native_height(p) * Entries.native_width(p) : U32),Entries.native_height(p),Entries.native_width(p),Entries.native_stride(p),
      (((position / Entries.native_output_width(p) : U32) + 1) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),(position % Entries.native_output_width(p) : U32),Entries.native_output_width(p),U32.from_nat(kc),Entries.native_padding(p)) ==
    PackingTabulation.entry(U32.from_nat(index),start,x,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
      Entries.native_output_width(p),Coordinates.native_inner(p)) : Tile.Tile8}

# A tile inside one output row.
def column_entry(+limits: Machine.Limits,+p: ConvSpec.Problem,+tiles: Nat,+g: Nat,+c: Nat,+kr: Nat,+kc: Nat,+qn: Nat,+index: Nat,+start: U32,+position: U32,
  +x: Tree.Tree<F32>,facts: TileFacts<limits,p,tiles,g,start,position,one_row(ConvSpec.out_width(p))>,
  reduction: {qn == Coordinates.reduction(c,kr,kc,Guards.kernel(p)) : Nat},indexed: {index == Entries.entry_index(p,g,qn) : Nat},
  +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool},+kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool},
  +kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool}) ->
  {LoopLaws.row_lanes(x,((position / Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),((position % Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kc) - Entries.native_padding(p) : U32),
      (U32.from_nat(c) * Entries.native_height(p) * Entries.native_width(p) : U32),Entries.native_height(p),Entries.native_width(p),Entries.native_stride(p),
      (((position / Entries.native_output_width(p) : U32) + 1) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),(position % Entries.native_output_width(p) : U32),Entries.native_output_width(p),U32.from_nat(kc),Entries.native_padding(p)) ==
    PackingTabulation.entry(U32.from_nat(index),start,x,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
      Entries.native_output_width(p),Coordinates.native_inner(p)) : Tile.Tile8}:
  TileFacts{+valid,+support,+total,+g_inside,position_exact,+within,+room} = facts
  %Equal.sym(Nat,index,Entries.entry_index(p,g,qn),indexed) :
    {LoopLaws.row_lanes(x,((position / Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),((position % Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kc) - Entries.native_padding(p) : U32),
      (U32.from_nat(c) * Entries.native_height(p) * Entries.native_width(p) : U32),Entries.native_height(p),Entries.native_width(p),Entries.native_stride(p),
      (((position / Entries.native_output_width(p) : U32) + 1) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),(position % Entries.native_output_width(p) : U32),Entries.native_output_width(p),U32.from_nat(kc),Entries.native_padding(p)) ==
      PackingTabulation.entry(U32.from_nat(_),start,x,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
        Entries.native_output_width(p),Coordinates.native_inner(p)) : Tile.Tile8}
  %Equal.sym(Nat,qn,Coordinates.reduction(c,kr,kc,Guards.kernel(p)),reduction) :
    {LoopLaws.row_lanes(x,((position / Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),((position % Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kc) - Entries.native_padding(p) : U32),
      (U32.from_nat(c) * Entries.native_height(p) * Entries.native_width(p) : U32),Entries.native_height(p),Entries.native_width(p),Entries.native_stride(p),
      (((position / Entries.native_output_width(p) : U32) + 1) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),(position % Entries.native_output_width(p) : U32),Entries.native_output_width(p),U32.from_nat(kc),Entries.native_padding(p)) ==
      PackingTabulation.entry(U32.from_nat(Entries.entry_index(p,g,_)),start,x,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),
        Entries.native_padding(p),Entries.native_output_width(p),Coordinates.native_inner(p)) : Tile.Tile8}
  Entries.row_entry(limits,p,tiles,g,c,kr,kc,start,position,Nat.div(U32.to_nat(position),ConvSpec.out_width(p)),Nat.mod(U32.to_nat(position),ConvSpec.out_width(p)),x,
    valid,support,total,g_inside,c_inside,kr_inside,kc_inside,position_exact,
    DivisionOrder.quotient_exact(U32.to_nat(position),ConvSpec.out_width(p),Coordinates.width_positive(p)),
    DivisionOrder.remainder_small(U32.to_nat(position),ConvSpec.out_width(p),Coordinates.width_positive(p)),within,room)

# A tile that ends in its output row or the next.
def wrap_column_entry(+limits: Machine.Limits,+p: ConvSpec.Problem,+tiles: Nat,+g: Nat,+c: Nat,+kr: Nat,+kc: Nat,+qn: Nat,+index: Nat,+start: U32,+position: U32,
  +x: Tree.Tree<F32>,facts: TileFacts<limits,p,tiles,g,start,position,two_rows(ConvSpec.out_width(p))>,
  reduction: {qn == Coordinates.reduction(c,kr,kc,Guards.kernel(p)) : Nat},indexed: {index == Entries.entry_index(p,g,qn) : Nat},
  +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool},+kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool},
  +kc_inside: {Nat.is_lt(kc,Guards.kernel(p)) == True{} : Bool}) ->
  {LoopLaws.wrap_lanes(x,((position / Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),((position % Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kc) - Entries.native_padding(p) : U32),
      (U32.from_nat(c) * Entries.native_height(p) * Entries.native_width(p) : U32),Entries.native_height(p),Entries.native_width(p),Entries.native_stride(p),
      (((position / Entries.native_output_width(p) : U32) + 1) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),(position % Entries.native_output_width(p) : U32),Entries.native_output_width(p),U32.from_nat(kc),Entries.native_padding(p)) ==
    PackingTabulation.entry(U32.from_nat(index),start,x,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
      Entries.native_output_width(p),Coordinates.native_inner(p)) : Tile.Tile8}:
  TileFacts{+valid,+support,+total,+g_inside,position_exact,+within,+room} = facts
  %Equal.sym(Nat,index,Entries.entry_index(p,g,qn),indexed) :
    {LoopLaws.wrap_lanes(x,((position / Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),((position % Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kc) - Entries.native_padding(p) : U32),
      (U32.from_nat(c) * Entries.native_height(p) * Entries.native_width(p) : U32),Entries.native_height(p),Entries.native_width(p),Entries.native_stride(p),
      (((position / Entries.native_output_width(p) : U32) + 1) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),(position % Entries.native_output_width(p) : U32),Entries.native_output_width(p),U32.from_nat(kc),Entries.native_padding(p)) ==
      PackingTabulation.entry(U32.from_nat(_),start,x,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
        Entries.native_output_width(p),Coordinates.native_inner(p)) : Tile.Tile8}
  %Equal.sym(Nat,qn,Coordinates.reduction(c,kr,kc,Guards.kernel(p)),reduction) :
    {LoopLaws.wrap_lanes(x,((position / Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),((position % Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kc) - Entries.native_padding(p) : U32),
      (U32.from_nat(c) * Entries.native_height(p) * Entries.native_width(p) : U32),Entries.native_height(p),Entries.native_width(p),Entries.native_stride(p),
      (((position / Entries.native_output_width(p) : U32) + 1) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),(position % Entries.native_output_width(p) : U32),Entries.native_output_width(p),U32.from_nat(kc),Entries.native_padding(p)) ==
      PackingTabulation.entry(U32.from_nat(Entries.entry_index(p,g,_)),start,x,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),
        Entries.native_padding(p),Entries.native_output_width(p),Coordinates.native_inner(p)) : Tile.Tile8}
  Entries.wrap_entry(limits,p,tiles,g,c,kr,kc,start,position,Nat.div(U32.to_nat(position),ConvSpec.out_width(p)),Nat.mod(U32.to_nat(position),ConvSpec.out_width(p)),x,
    valid,support,total,g_inside,c_inside,kr_inside,kc_inside,position_exact,
    DivisionOrder.quotient_exact(U32.to_nat(position),ConvSpec.out_width(p),Coordinates.width_positive(p)),
    DivisionOrder.remainder_small(U32.to_nat(position),ConvSpec.out_width(p),Coordinates.width_positive(p)),within,room)

# n kernel columns from kc.
law columns:
  for ~lanes: LoopLaws.Lanes()
  for ~span: Nat -> Nat
  for ~rule: EntryRule(~lanes,~span)
  for +limits: Machine.Limits
  for +p: ConvSpec.Problem
  for +tiles: Nat
  for +g: Nat
  for +start: U32
  for +position: U32
  for +x: Tree.Tree<F32>
  for +facts: TileFacts<limits,p,tiles,g,start,position,span(ConvSpec.out_width(p))>
  for +n: Nat
  for +c: Nat
  for +kr: Nat
  for +kc: Nat
  for +qn: Nat
  for +index: Nat
  for +y: Tree.Tree<Tile.Tile8>
  for +reduction: {qn == Coordinates.reduction(c,kr,kc,Guards.kernel(p)) : Nat}
  for +indexed: {index == Entries.entry_index(p,g,qn) : Nat}
  for +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool}
  for +kr_inside: {Nat.is_lt(kr,Guards.kernel(p)) == True{} : Bool}
  for +room: {Nat.is_le(Nat.add(kc,n),Guards.kernel(p)) == True{} : Bool}
  {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.column_step(~lanes),n,state(p,x,y,U32.from_nat(index),U32.from_nat(c),U32.from_nat(kr),U32.from_nat(kc),position)) ==
    state(p,x,filled(p,n,U32.from_nat(index),start,x,y),U32.from_nat(advance(n,index)),U32.from_nat(c),U32.from_nat(kr),U32.from_nat(advance(n,kc)),position) : LoopLaws.RowTiles()}
def columns(lanes,span,rule,limits,p,tiles,g,start,position,x,facts,n,c,kr,kc,qn,index,y,reduction,indexed,c_inside,kr_inside,room):
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +entry = PackingTabulation.entry(U32.from_nat(index),start,x,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),
        Entries.native_padding(p),Entries.native_output_width(p),Coordinates.native_inner(p))
      +next = Tree.write(Tile.Tile8,y,U32.from_nat(index),entry)
      %Equal.sym(Tile.Tile8,lanes(x,((position / Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),((position % Entries.native_output_width(p) : U32) * Entries.native_stride(p) + U32.from_nat(kc) - Entries.native_padding(p) : U32),
      (U32.from_nat(c) * Entries.native_height(p) * Entries.native_width(p) : U32),Entries.native_height(p),Entries.native_width(p),Entries.native_stride(p),
      (((position / Entries.native_output_width(p) : U32) + 1) * Entries.native_stride(p) + U32.from_nat(kr) - Entries.native_padding(p) : U32),(position % Entries.native_output_width(p) : U32),Entries.native_output_width(p),U32.from_nat(kc),Entries.native_padding(p)),entry,
        rule(limits,p,tiles,g,c,kr,kc,qn,index,start,position,x,facts,reduction,indexed,c_inside,kr_inside,Order.room_index(kc,rest,Guards.kernel(p),room))) :
        {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.column_step(~lanes),rest,state(p,x,Tree.write(Tile.Tile8,y,U32.from_nat(index),_),(U32.from_nat(index) + 1 : U32),
            U32.from_nat(c),U32.from_nat(kr),(U32.from_nat(kc) + 1 : U32),position)) ==
          state(p,x,filled(p,rest,(U32.from_nat(index) + 1 : U32),start,x,next),U32.from_nat(advance(rest,1n+index)),U32.from_nat(c),U32.from_nat(kr),
            U32.from_nat(advance(rest,1n+kc)),position) : LoopLaws.RowTiles()}
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),WriteBounds.increment_encoded(index)) :
        {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.column_step(~lanes),rest,state(p,x,next,_,U32.from_nat(c),U32.from_nat(kr),(U32.from_nat(kc) + 1 : U32),position)) ==
          state(p,x,filled(p,rest,_,start,x,next),U32.from_nat(advance(rest,1n+index)),U32.from_nat(c),U32.from_nat(kr),U32.from_nat(advance(rest,1n+kc)),position) : LoopLaws.RowTiles()}
      %Equal.sym(U32,(U32.from_nat(kc) + 1 : U32),U32.from_nat(1n+kc),WriteBounds.increment_encoded(kc)) :
        {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.column_step(~lanes),rest,state(p,x,next,U32.from_nat(1n+index),U32.from_nat(c),U32.from_nat(kr),_,position)) ==
          state(p,x,filled(p,rest,U32.from_nat(1n+index),start,x,next),U32.from_nat(advance(rest,1n+index)),U32.from_nat(c),U32.from_nat(kr),
            U32.from_nat(advance(rest,1n+kc)),position) : LoopLaws.RowTiles()}
      columns(~lanes,~span,~rule,limits,p,tiles,g,start,position,x,facts,rest,c,kr,1n+kc,1n+qn,1n+index,next,
        shift(Nat.mul(Nat.add(Nat.mul(c,Guards.kernel(p)),kr),Guards.kernel(p)),qn,kc,reduction),
        shift(Nat.mul(g,Coordinates.inner(p)),index,qn,indexed),c_inside,kr_inside,Order.advance_room(kc,rest,Guards.kernel(p),room))

def advance_add(+n: Nat,+k: Nat) -> {advance(n,k) == Nat.add(n,k) : Nat}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      %Addition.successor_right(rest,k) : {advance(rest,1n+k) == _ : Nat}
      advance_add(rest,1n+k)

def advance_append(+a: Nat,+b: Nat,+k: Nat) -> {advance(Nat.add(a,b),k) == advance(b,advance(a,k)) : Nat}:
  match a:
    case 0n: {==}
    case 1n+ +rest: advance_append(rest,b,1n+k)

def advance_shift(+n: Nat,+m: Nat,+a: Nat,+b: Nat,e: {a == Nat.add(m,b) : Nat}) -> {advance(n,a) == Nat.add(m,advance(n,b)) : Nat}:
  match n:
    case 0n: e
    case 1n+ +rest: advance_shift(rest,m,1n+a,1n+b,shift(m,a,b,e))

# One more group of k after b groups.
def group_carry(+k: Nat,+b: Nat,+q: Nat,e: {q == Nat.add(Nat.mul(b,k),0n) : Nat}) -> {advance(k,q) == Nat.add(Nat.mul(1n+b,k),0n) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.add(k,Nat.mul(b,k)),0n),Nat.add(k,Nat.add(Nat.mul(b,k),0n)),Addition.associative(k,Nat.mul(b,k),0n)) : {advance(k,q) == _ : Nat}
  %e : {advance(k,q) == Nat.add(k,_) : Nat}
  advance_add(k,q)

def next_row_reduction(+c: Nat,+kr: Nat,+k: Nat,+qn: Nat,e: {qn == Coordinates.reduction(c,kr,0n,k) : Nat}) -> {advance(k,qn) == Coordinates.reduction(c,1n+kr,0n,k) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.mul(c,k),1n+kr),1n+Nat.add(Nat.mul(c,k),kr),Addition.successor_right(Nat.mul(c,k),kr)) : {advance(k,qn) == Nat.add(Nat.mul(_,k),0n) : Nat}
  group_carry(k,Nat.add(Nat.mul(c,k),kr),qn,e)

def rows_reduction(+n: Nat,+c: Nat,+kr: Nat,+k: Nat,+qn: Nat,e: {qn == Coordinates.reduction(c,kr,0n,k) : Nat}) ->
  {advance(Nat.mul(n,k),qn) == Coordinates.reduction(c,advance(n,kr),0n,k) : Nat}:
  match n:
    case 0n: e
    case 1n+ +rest:
      %Equal.sym(Nat,advance(Nat.add(k,Nat.mul(rest,k)),qn),advance(Nat.mul(rest,k),advance(k,qn)),advance_append(k,Nat.mul(rest,k),qn)) :
        {_ == Coordinates.reduction(c,advance(rest,1n+kr),0n,k) : Nat}
      rows_reduction(rest,c,1n+kr,k,advance(k,qn),next_row_reduction(c,kr,k,qn,e))

def channel_carry(+c: Nat,+k: Nat) -> {Nat.add(Nat.mul(c,k),advance(k,0n)) == Nat.add(Nat.add(k,Nat.mul(c,k)),0n) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.add(k,Nat.mul(c,k)),0n),Nat.add(k,Nat.mul(c,k)),Addition.zero_right(Nat.add(k,Nat.mul(c,k)))) : {Nat.add(Nat.mul(c,k),advance(k,0n)) == _ : Nat}
  %Equal.sym(Nat,advance(k,0n),Nat.add(k,0n),advance_add(k,0n)) : {Nat.add(Nat.mul(c,k),_) == Nat.add(k,Nat.mul(c,k)) : Nat}
  %Equal.sym(Nat,Nat.add(k,0n),k,Addition.zero_right(k)) : {Nat.add(Nat.mul(c,k),_) == Nat.add(k,Nat.mul(c,k)) : Nat}
  Order.commutative(Nat.mul(c,k),k)

def next_channel_reduction(+c: Nat,+k: Nat,+qn: Nat,e: {qn == Coordinates.reduction(c,0n,0n,k) : Nat}) ->
  {advance(Nat.mul(k,k),qn) == Coordinates.reduction(1n+c,0n,0n,k) : Nat}:
  %channel_carry(c,k) : {advance(Nat.mul(k,k),qn) == Nat.add(Nat.mul(_,k),0n) : Nat}
  rows_reduction(k,c,0n,k,qn,e)

# The tabulation of a + m entries is the tabulation of m entries after the tabulation of a.
def grouped(+p: ConvSpec.Problem,+a: Nat,+m: Nat,+index: Nat,+start: U32,+x: Tree.Tree<F32>,+y: Tree.Tree<Tile.Tile8>) ->
  {filled(p,Nat.add(a,m),U32.from_nat(index),start,x,y) == filled(p,m,U32.from_nat(advance(a,index)),start,x,filled(p,a,U32.from_nat(index),start,x,y)) : Tree.Tree<Tile.Tile8>}:
  %end_encoded(a,index) : {filled(p,Nat.add(a,m),U32.from_nat(index),start,x,y) == filled(p,m,_,start,x,filled(p,a,U32.from_nat(index),start,x,y)) : Tree.Tree<Tile.Tile8>}
  Tiles.fill_append(a,m,U32.from_nat(index),start,x,y,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
    Entries.native_output_width(p),Coordinates.native_inner(p))

# n kernel rows from kr.
law kernel_rows:
  for ~lanes: LoopLaws.Lanes()
  for ~span: Nat -> Nat
  for ~rule: EntryRule(~lanes,~span)
  for +limits: Machine.Limits
  for +p: ConvSpec.Problem
  for +tiles: Nat
  for +g: Nat
  for +start: U32
  for +position: U32
  for +x: Tree.Tree<F32>
  for +facts: TileFacts<limits,p,tiles,g,start,position,span(ConvSpec.out_width(p))>
  for +n: Nat
  for +c: Nat
  for +kr: Nat
  for +qn: Nat
  for +index: Nat
  for +y: Tree.Tree<Tile.Tile8>
  for +reduction: {qn == Coordinates.reduction(c,kr,0n,Guards.kernel(p)) : Nat}
  for +indexed: {index == Entries.entry_index(p,g,qn) : Nat}
  for +c_inside: {Nat.is_lt(c,Guards.input_channel_count(p)) == True{} : Bool}
  for +room: {Nat.is_le(Nat.add(kr,n),Guards.kernel(p)) == True{} : Bool}
  {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.kernel_row_step(~lanes),n,state(p,x,y,U32.from_nat(index),U32.from_nat(c),U32.from_nat(kr),0,position)) ==
    state(p,x,filled(p,Nat.mul(n,Guards.kernel(p)),U32.from_nat(index),start,x,y),U32.from_nat(advance(Nat.mul(n,Guards.kernel(p)),index)),U32.from_nat(c),
      U32.from_nat(advance(n,kr)),0,position) : LoopLaws.RowTiles()}
def kernel_rows(lanes,span,rule,limits,p,tiles,g,start,position,x,facts,n,c,kr,qn,index,y,reduction,indexed,c_inside,room):
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +k = Guards.kernel(p)
      +m = Nat.mul(rest,k)
      +next = filled(p,k,U32.from_nat(index),start,x,y)
      +after = state(p,x,filled(p,Nat.add(k,m),U32.from_nat(index),start,x,y),U32.from_nat(advance(Nat.add(k,m),index)),U32.from_nat(c),U32.from_nat(advance(rest,1n+kr)),0,position)
      %Equal.sym(LoopLaws.RowTiles(),Loop.run(~LoopLaws.RowTiles(),~LoopLaws.column_step(~lanes),k,state(p,x,y,U32.from_nat(index),U32.from_nat(c),U32.from_nat(kr),0,position)),
          state(p,x,next,U32.from_nat(advance(k,index)),U32.from_nat(c),U32.from_nat(kr),U32.from_nat(advance(k,0n)),position),
          columns(~lanes,~span,~rule,limits,p,tiles,g,start,position,x,facts,k,c,kr,0n,qn,index,y,reduction,indexed,c_inside,Order.room_index(kr,rest,k,room),Order.reflexive(k))) :
        {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.kernel_row_step(~lanes),rest,LoopLaws.next_kernel_row(_)) == after : LoopLaws.RowTiles()}
      %Equal.sym(U32,(U32.from_nat(kr) + 1 : U32),U32.from_nat(1n+kr),WriteBounds.increment_encoded(kr)) :
        {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.kernel_row_step(~lanes),rest,state(p,x,next,U32.from_nat(advance(k,index)),U32.from_nat(c),_,0,position)) == after : LoopLaws.RowTiles()}
      %Equal.sym(Tree.Tree<Tile.Tile8>,filled(p,Nat.add(k,m),U32.from_nat(index),start,x,y),filled(p,m,U32.from_nat(advance(k,index)),start,x,next),
          grouped(p,k,m,index,start,x,y)) :
        {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.kernel_row_step(~lanes),rest,state(p,x,next,U32.from_nat(advance(k,index)),U32.from_nat(c),U32.from_nat(1n+kr),0,position)) ==
          state(p,x,_,U32.from_nat(advance(Nat.add(k,m),index)),U32.from_nat(c),U32.from_nat(advance(rest,1n+kr)),0,position) : LoopLaws.RowTiles()}
      %Equal.sym(Nat,advance(Nat.add(k,m),index),advance(m,advance(k,index)),advance_append(k,m,index)) :
        {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.kernel_row_step(~lanes),rest,state(p,x,next,U32.from_nat(advance(k,index)),U32.from_nat(c),U32.from_nat(1n+kr),0,position)) ==
          state(p,x,filled(p,m,U32.from_nat(advance(k,index)),start,x,next),U32.from_nat(_),U32.from_nat(c),U32.from_nat(advance(rest,1n+kr)),0,position) : LoopLaws.RowTiles()}
      kernel_rows(~lanes,~span,~rule,limits,p,tiles,g,start,position,x,facts,rest,c,1n+kr,advance(k,qn),advance(k,index),next,next_row_reduction(c,kr,k,qn,reduction),
        advance_shift(k,Nat.mul(g,Coordinates.inner(p)),index,qn,indexed),c_inside,Order.advance_room(kr,rest,k,room))

# n input channels from c.
law channels:
  for ~lanes: LoopLaws.Lanes()
  for ~span: Nat -> Nat
  for ~rule: EntryRule(~lanes,~span)
  for +limits: Machine.Limits
  for +p: ConvSpec.Problem
  for +tiles: Nat
  for +g: Nat
  for +start: U32
  for +position: U32
  for +x: Tree.Tree<F32>
  for +facts: TileFacts<limits,p,tiles,g,start,position,span(ConvSpec.out_width(p))>
  for +n: Nat
  for +c: Nat
  for +qn: Nat
  for +index: Nat
  for +y: Tree.Tree<Tile.Tile8>
  for +reduction: {qn == Coordinates.reduction(c,0n,0n,Guards.kernel(p)) : Nat}
  for +indexed: {index == Entries.entry_index(p,g,qn) : Nat}
  for +room: {Nat.is_le(Nat.add(c,n),Guards.input_channel_count(p)) == True{} : Bool}
  {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.channel_step(~lanes),n,state(p,x,y,U32.from_nat(index),U32.from_nat(c),0,0,position)) ==
    state(p,x,filled(p,Nat.mul(n,Nat.mul(Guards.kernel(p),Guards.kernel(p))),U32.from_nat(index),start,x,y),
      U32.from_nat(advance(Nat.mul(n,Nat.mul(Guards.kernel(p),Guards.kernel(p))),index)),U32.from_nat(advance(n,c)),0,0,position) : LoopLaws.RowTiles()}
def channels(lanes,span,rule,limits,p,tiles,g,start,position,x,facts,n,c,qn,index,y,reduction,indexed,room):
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +k = Guards.kernel(p)
      +square = Nat.mul(k,k)
      +m = Nat.mul(rest,square)
      +next = filled(p,square,U32.from_nat(index),start,x,y)
      +after = state(p,x,filled(p,Nat.add(square,m),U32.from_nat(index),start,x,y),U32.from_nat(advance(Nat.add(square,m),index)),U32.from_nat(advance(rest,1n+c)),0,0,position)
      %Equal.sym(LoopLaws.RowTiles(),Loop.run(~LoopLaws.RowTiles(),~LoopLaws.kernel_row_step(~lanes),k,state(p,x,y,U32.from_nat(index),U32.from_nat(c),0,0,position)),
          state(p,x,next,U32.from_nat(advance(square,index)),U32.from_nat(c),U32.from_nat(advance(k,0n)),0,position),
          kernel_rows(~lanes,~span,~rule,limits,p,tiles,g,start,position,x,facts,k,c,0n,qn,index,y,reduction,indexed,Order.room_index(c,rest,Guards.input_channel_count(p),room),
            Order.reflexive(k))) :
        {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.channel_step(~lanes),rest,LoopLaws.next_channel(_)) == after : LoopLaws.RowTiles()}
      %Equal.sym(U32,(U32.from_nat(c) + 1 : U32),U32.from_nat(1n+c),WriteBounds.increment_encoded(c)) :
        {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.channel_step(~lanes),rest,state(p,x,next,U32.from_nat(advance(square,index)),_,0,0,position)) == after : LoopLaws.RowTiles()}
      %Equal.sym(Tree.Tree<Tile.Tile8>,filled(p,Nat.add(square,m),U32.from_nat(index),start,x,y),filled(p,m,U32.from_nat(advance(square,index)),start,x,next),
          grouped(p,square,m,index,start,x,y)) :
        {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.channel_step(~lanes),rest,state(p,x,next,U32.from_nat(advance(square,index)),U32.from_nat(1n+c),0,0,position)) ==
          state(p,x,_,U32.from_nat(advance(Nat.add(square,m),index)),U32.from_nat(advance(rest,1n+c)),0,0,position) : LoopLaws.RowTiles()}
      %Equal.sym(Nat,advance(Nat.add(square,m),index),advance(m,advance(square,index)),advance_append(square,m,index)) :
        {Loop.run(~LoopLaws.RowTiles(),~LoopLaws.channel_step(~lanes),rest,state(p,x,next,U32.from_nat(advance(square,index)),U32.from_nat(1n+c),0,0,position)) ==
          state(p,x,filled(p,m,U32.from_nat(advance(square,index)),start,x,next),U32.from_nat(_),U32.from_nat(advance(rest,1n+c)),0,0,position) : LoopLaws.RowTiles()}
      channels(~lanes,~span,~rule,limits,p,tiles,g,start,position,x,facts,rest,1n+c,advance(square,qn),advance(square,index),next,next_channel_reduction(c,k,qn,reduction),
        advance_shift(square,Nat.mul(g,Coordinates.inner(p)),index,qn,indexed),Order.advance_room(c,rest,Guards.input_channel_count(p),room))

def crossing_state(+p: ConvSpec.Problem,x: Tree.Tree<F32>,y: Tree.Tree<Tile.Tile8>,i: U32,q: U32,position: U32) -> LoopLaws.Crossings():
  Packing.Crossing{x,y,i,q,position,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
    Entries.native_output_width(p)}

def crossing_at(+limits: Machine.Limits,+p: ConvSpec.Problem,+tiles: Nat,+g: Nat,+qn: Nat,+index: Nat,+start: U32,+position: U32,+x: Tree.Tree<F32>,
  +valid: Guards.Valid<p>,+support: Entries.Supported(limits,p),
  +total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +g_inside: {Nat.is_lt(g,tiles) == True{} : Bool},+q_inside: {Nat.is_lt(qn,Coordinates.inner(p)) == True{} : Bool},
  +position_exact: {(start + U32.from_nat(g) * 8 : U32) == position : U32},indexed: {index == Entries.entry_index(p,g,qn) : Nat}) ->
  {PackingTabulation.vector(x,position,U32.from_nat(qn),Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
      Entries.native_output_width(p)) ==
    PackingTabulation.entry(U32.from_nat(index),start,x,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
      Entries.native_output_width(p),Coordinates.native_inner(p)) : Tile.Tile8}:
  %Equal.sym(Nat,index,Entries.entry_index(p,g,qn),indexed) :
    {PackingTabulation.vector(x,position,U32.from_nat(qn),Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
        Entries.native_output_width(p)) ==
      PackingTabulation.entry(U32.from_nat(_),start,x,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
        Entries.native_output_width(p),Coordinates.native_inner(p)) : Tile.Tile8}
  Entries.crossing_entry(limits,p,tiles,g,qn,start,position,x,valid,support,total,g_inside,q_inside,position_exact)

# n entries of a tile that crosses an output row, from reduction index qn.
law crossings:
  for +limits: Machine.Limits
  for +p: ConvSpec.Problem
  for +tiles: Nat
  for +g: Nat
  for +start: U32
  for +position: U32
  for +x: Tree.Tree<F32>
  for +valid: Guards.Valid<p>
  for +support: Entries.Supported(limits,p)
  for +total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  for +g_inside: {Nat.is_lt(g,tiles) == True{} : Bool}
  for +position_exact: {(start + U32.from_nat(g) * 8 : U32) == position : U32}
  for +n: Nat
  for +qn: Nat
  for +index: Nat
  for +y: Tree.Tree<Tile.Tile8>
  for +indexed: {index == Entries.entry_index(p,g,qn) : Nat}
  for +room: {Nat.is_le(Nat.add(qn,n),Coordinates.inner(p)) == True{} : Bool}
  {Loop.run(~LoopLaws.Crossings(),~LoopLaws.crossing_step,n,crossing_state(p,x,y,U32.from_nat(index),U32.from_nat(qn),position)) ==
    crossing_state(p,x,filled(p,n,U32.from_nat(index),start,x,y),U32.from_nat(advance(n,index)),U32.from_nat(advance(n,qn)),position) : LoopLaws.Crossings()}
def crossings(limits,p,tiles,g,start,position,x,valid,support,total,g_inside,position_exact,n,qn,index,y,indexed,room):
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +entry = PackingTabulation.entry(U32.from_nat(index),start,x,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),
        Entries.native_padding(p),Entries.native_output_width(p),Coordinates.native_inner(p))
      +next = Tree.write(Tile.Tile8,y,U32.from_nat(index),entry)
      +after = crossing_state(p,x,filled(p,rest,U32.from_nat(1n+index),start,x,next),U32.from_nat(advance(rest,1n+index)),U32.from_nat(advance(rest,1n+qn)),position)
      %Equal.sym(Tile.Tile8,PackingTabulation.vector(x,position,U32.from_nat(qn),Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),
          Entries.native_padding(p),Entries.native_output_width(p)),entry,
        crossing_at(limits,p,tiles,g,qn,index,start,position,x,valid,support,total,g_inside,Order.room_index(qn,rest,Coordinates.inner(p),room),position_exact,indexed)) :
        {Loop.run(~LoopLaws.Crossings(),~LoopLaws.crossing_step,rest,crossing_state(p,x,Tree.write(Tile.Tile8,y,U32.from_nat(index),_),(U32.from_nat(index) + 1 : U32),
            (U32.from_nat(qn) + 1 : U32),position)) ==
          crossing_state(p,x,filled(p,rest,(U32.from_nat(index) + 1 : U32),start,x,next),U32.from_nat(advance(rest,1n+index)),U32.from_nat(advance(rest,1n+qn)),position) : LoopLaws.Crossings()}
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),WriteBounds.increment_encoded(index)) :
        {Loop.run(~LoopLaws.Crossings(),~LoopLaws.crossing_step,rest,crossing_state(p,x,next,_,(U32.from_nat(qn) + 1 : U32),position)) ==
          crossing_state(p,x,filled(p,rest,_,start,x,next),U32.from_nat(advance(rest,1n+index)),U32.from_nat(advance(rest,1n+qn)),position) : LoopLaws.Crossings()}
      %Equal.sym(U32,(U32.from_nat(qn) + 1 : U32),U32.from_nat(1n+qn),WriteBounds.increment_encoded(qn)) :
        {Loop.run(~LoopLaws.Crossings(),~LoopLaws.crossing_step,rest,crossing_state(p,x,next,U32.from_nat(1n+index),_,position)) == after : LoopLaws.Crossings()}
      crossings(limits,p,tiles,g,start,position,x,valid,support,total,g_inside,position_exact,rest,1n+qn,1n+index,next,
        shift(Nat.mul(g,Coordinates.inner(p)),index,qn,indexed),Order.advance_room(qn,rest,Coordinates.inner(p),room))

def tiled(+p: ConvSpec.Problem,x: Tree.Tree<F32>,y: Tree.Tree<Tile.Tile8>,i: U32,position: U32) -> LoopLaws.Tiled():
  Packing.Tiles{x,y,i,position,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),
    Entries.native_output_width(p),Guards.input_channel_count(p),Guards.kernel(p),Coordinates.inner(p)}

def room_ahead(+pn: Nat,+m: Nat,+cap: Nat,room: {Nat.is_le(Nat.add(pn,Nat.add(8n,m)),cap) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.add(pn,8n),m),cap) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(Nat.add(pn,8n),m),Nat.add(pn,Nat.add(8n,m)),Addition.associative(pn,8n,m)) : {Nat.is_le(_,cap) == True{} : Bool}
  room

def grid(+p: ConvSpec.Problem) -> Nat:
  Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p))

# The eight positions of a tile that starts inside the grid fit a word.
def tile_room(+limits: Machine.Limits,+p: ConvSpec.Problem,+position: U32,+valid: Guards.Valid<p>,+support: Entries.Supported(limits,p),
  below: {Nat.is_lt(U32.to_nat(position),grid(p)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +pn = U32.to_nat(position)
  +capacity = U32.to_nat(Machine.capacity(limits))
  %Equal.sym(Nat,Nat.add(pn,8n),1n+Nat.add(pn,7n),Addition.successor_right(pn,7n)) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
  Padded.add_seven_bound(limits,1n+pn,Order.strict_follow(pn,capacity,Bounds.nat_trans(pn,grid(p),capacity,below,Padded.spatial_bound_facts(limits,p,valid,support))))

def shifted_room(+pn: Nat,+limit: Nat,first: {Nat.is_le(Nat.add(pn,8n),Nat.add(limit,7n)) == True{} : Bool}) -> {Nat.is_le(Nat.add(7n,1n+pn),Nat.add(7n,limit)) == True{} : Bool}:
  %Order.commutative(limit,7n) : {Nat.is_le(Nat.add(8n,pn),_) == True{} : Bool}
  %Order.commutative(pn,8n) : {Nat.is_le(_,Nat.add(limit,7n)) == True{} : Bool}
  first

# A remaining run of tiles that ends within seven of the grid starts inside it.
def below_grid(+pn: Nat,+m: Nat,+limit: Nat,room: {Nat.is_le(Nat.add(pn,Nat.add(8n,m)),Nat.add(limit,7n)) == True{} : Bool}) -> {Nat.is_lt(pn,limit) == True{} : Bool}:
  +first = Order.transitive(Nat.add(pn,8n),Nat.add(Nat.add(pn,8n),m),Nat.add(limit,7n),Order.le_add(Nat.add(pn,8n),m),room_ahead(pn,m,Nat.add(limit,7n),room))
  Bounds.nat_trans(pn,1n+pn,limit,Order.lt_succ(pn,pn,Order.reflexive(pn)),Order.cancel_offset_le(7n,1n+pn,limit,shifted_room(pn,limit,first)))

# One tile: inside one output row, ending in the next, or read lane by lane.
def tile_filled(+limits: Machine.Limits,+p: ConvSpec.Problem,+tiles: Nat,+g: Nat,+index: Nat,+start: U32,+position: U32,+x: Tree.Tree<F32>,
  +y: Tree.Tree<Tile.Tile8>,+within: Bool,+wraps: Bool,+valid: Guards.Valid<p>,+support: Entries.Supported(limits,p),
  +total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +g_inside: {Nat.is_lt(g,tiles) == True{} : Bool},+position_exact: {(start + U32.from_nat(g) * 8 : U32) == position : U32},
  +below: {Nat.is_lt(U32.to_nat(position),grid(p)) == True{} : Bool},
  +indexed: {index == Entries.entry_index(p,g,0n) : Nat},
  +decided: {U32.is_le((position % Entries.native_output_width(p) + 8 : U32),Entries.native_output_width(p)) == within : Bool},
  +spans: {U32.is_le((position % Entries.native_output_width(p) + 8 : U32),(Entries.native_output_width(p) + Entries.native_output_width(p) : U32)) == wraps : Bool}) ->
  {LoopLaws.tile(within,wraps,x,y,U32.from_nat(index),position,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),
      Entries.native_padding(p),Entries.native_output_width(p),Guards.input_channel_count(p),Guards.kernel(p),Coordinates.inner(p)) ==
    Packing.Tiles{x,filled(p,Coordinates.inner(p),U32.from_nat(index),start,x,y),U32.from_nat(advance(Coordinates.inner(p),index)),(position + 8 : U32),
      Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),Entries.native_output_width(p),
      Guards.input_channel_count(p),Guards.kernel(p),Coordinates.inner(p)} : LoopLaws.Tiled()}:
  match within wraps:
    case True{} wraps:
      %Equal.sym(LoopLaws.RowTiles(),Loop.run(~LoopLaws.RowTiles(),~LoopLaws.channel_step(~LoopLaws.row_lanes),Guards.input_channel_count(p),state(p,x,y,U32.from_nat(index),U32.from_nat(0n),0,0,position)),
          state(p,x,filled(p,Coordinates.inner(p),U32.from_nat(index),start,x,y),U32.from_nat(advance(Coordinates.inner(p),index)),
            U32.from_nat(advance(Guards.input_channel_count(p),0n)),0,0,position),
          channels(~LoopLaws.row_lanes,~one_row,~column_entry,limits,p,tiles,g,start,position,x,
            TileFacts{valid,support,total,g_inside,position_exact,Entries.within_columns(limits,p,position,valid,support,decided),tile_room(limits,p,position,valid,support,below)},
            Guards.input_channel_count(p),0n,0n,index,y,{==},indexed,Order.reflexive(Guards.input_channel_count(p)))) :
        {LoopLaws.after_row_tile(position,Entries.native_kernel(p),Guards.input_channel_count(p),Coordinates.inner(p),_) ==
          Packing.Tiles{x,filled(p,Coordinates.inner(p),U32.from_nat(index),start,x,y),U32.from_nat(advance(Coordinates.inner(p),index)),(position + 8 : U32),
            Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),Entries.native_output_width(p),
            Guards.input_channel_count(p),Guards.kernel(p),Coordinates.inner(p)} : LoopLaws.Tiled()}
      {==}
    case False{} True{}:
      %Equal.sym(LoopLaws.RowTiles(),Loop.run(~LoopLaws.RowTiles(),~LoopLaws.channel_step(~LoopLaws.wrap_lanes),Guards.input_channel_count(p),state(p,x,y,U32.from_nat(index),U32.from_nat(0n),0,0,position)),
          state(p,x,filled(p,Coordinates.inner(p),U32.from_nat(index),start,x,y),U32.from_nat(advance(Coordinates.inner(p),index)),
            U32.from_nat(advance(Guards.input_channel_count(p),0n)),0,0,position),
          channels(~LoopLaws.wrap_lanes,~two_rows,~wrap_column_entry,limits,p,tiles,g,start,position,x,
            TileFacts{valid,support,total,g_inside,position_exact,Entries.wraps_columns(limits,p,position,valid,support,spans),tile_room(limits,p,position,valid,support,below)},
            Guards.input_channel_count(p),0n,0n,index,y,{==},indexed,Order.reflexive(Guards.input_channel_count(p)))) :
        {LoopLaws.after_row_tile(position,Entries.native_kernel(p),Guards.input_channel_count(p),Coordinates.inner(p),_) ==
          Packing.Tiles{x,filled(p,Coordinates.inner(p),U32.from_nat(index),start,x,y),U32.from_nat(advance(Coordinates.inner(p),index)),(position + 8 : U32),
            Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),Entries.native_output_width(p),
            Guards.input_channel_count(p),Guards.kernel(p),Coordinates.inner(p)} : LoopLaws.Tiled()}
      {==}
    case False{} False{}:
      %Equal.sym(LoopLaws.Crossings(),Loop.run(~LoopLaws.Crossings(),~LoopLaws.crossing_step,Coordinates.inner(p),crossing_state(p,x,y,U32.from_nat(index),U32.from_nat(0n),position)),
          crossing_state(p,x,filled(p,Coordinates.inner(p),U32.from_nat(index),start,x,y),U32.from_nat(advance(Coordinates.inner(p),index)),
            U32.from_nat(advance(Coordinates.inner(p),0n)),position),
          crossings(limits,p,tiles,g,start,position,x,valid,support,total,g_inside,position_exact,Coordinates.inner(p),0n,index,y,indexed,Order.reflexive(Coordinates.inner(p)))) :
        {LoopLaws.after_crossing(Guards.input_channel_count(p),Guards.kernel(p),Coordinates.inner(p),_) ==
          Packing.Tiles{x,filled(p,Coordinates.inner(p),U32.from_nat(index),start,x,y),U32.from_nat(advance(Coordinates.inner(p),index)),(position + 8 : U32),
            Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),Entries.native_output_width(p),
            Guards.input_channel_count(p),Guards.kernel(p),Coordinates.inner(p)} : LoopLaws.Tiled()}
      {==}

def rebound(+a: Nat,+b: Nat,+cap: Nat,e: {a == b : Nat},h: {Nat.is_le(a,cap) == True{} : Bool}) -> {Nat.is_le(b,cap) == True{} : Bool}:
  %e : {Nat.is_le(_,cap) == True{} : Bool}
  h

# The first position of tile g, without wrapping.
def group_position(+limits: Machine.Limits,+start: U32,+g: Nat,+pn: Nat,+located: {pn == Nat.add(U32.to_nat(start),Nat.mul(g,8n)) : Nat},
  bound: {Nat.is_le(pn,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {(start + U32.from_nat(g) * 8 : U32) == U32.from_nat(pn) : U32}:
  +upper = Machine.capacity(limits)
  +upper_ok = Limits.capacity_fits(limits)
  +product = Nat.mul(g,8n)
  +base = Nat.add(U32.to_nat(start),product)
  +base_bound = rebound(pn,base,U32.to_nat(upper),located,bound)
  +product_bound = Order.transitive(product,base,U32.to_nat(upper),
    %Order.commutative(product,U32.to_nat(start)) : {Nat.is_le(product,_) == True{} : Bool}
    Order.le_add(product,U32.to_nat(start)),base_bound)
  +group_exact = Bounds.roundtrip(g,upper,upper_ok,Order.transitive(g,product,U32.to_nat(upper),RowMajor.product_covers(g,8n,{==}),product_bound))
  +encoded_product = Arithmetic.multiply(U32.from_nat(g),8,upper,upper_ok,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(g)),g,group_exact) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(upper)) == True{} : Bool}
    product_bound)
  +product_exact = Equal.trans(Nat,U32.to_nat((U32.from_nat(g) * 8 : U32)),Nat.mul(U32.to_nat(U32.from_nat(g)),8n),product,
    encoded_product,Equal.cong(Nat,Nat,n => Nat.mul(n,8n),U32.to_nat(U32.from_nat(g)),g,group_exact))
  %Equal.sym(Nat,pn,base,located) : {(start + U32.from_nat(g) * 8 : U32) == U32.from_nat(_) : U32}
  Equal.trans(U32,(start + U32.from_nat(g) * 8 : U32),U32.from_nat(Nat.add(U32.to_nat(start),U32.to_nat((U32.from_nat(g) * 8 : U32)))),U32.from_nat(base),
    Arithmetic.native_add(start,(U32.from_nat(g) * 8 : U32)),
    Equal.cong(Nat,U32,n => U32.from_nat(Nat.add(U32.to_nat(start),n)),U32.to_nat((U32.from_nat(g) * 8 : U32)),product,product_exact))

def next_located(+sn: Nat,+m: Nat,+pn: Nat,e: {pn == Nat.add(sn,m) : Nat}) -> {Nat.add(pn,8n) == Nat.add(sn,Nat.add(8n,m)) : Nat}:
  %Order.commutative(m,8n) : {Nat.add(pn,8n) == Nat.add(sn,_) : Nat}
  %Addition.associative(sn,m,8n) : {Nat.add(pn,8n) == _ : Nat}
  Equal.cong(Nat,Nat,v => Nat.add(v,8n),pn,Nat.add(sn,m),e)

# n tiles from tile g at position pn.
law tiles_fill:
  for +limits: Machine.Limits
  for +p: ConvSpec.Problem
  for +tiles: Nat
  for +start: U32
  for +x: Tree.Tree<F32>
  for +valid: Guards.Valid<p>
  for +support: Entries.Supported(limits,p)
  for +total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  for +n: Nat
  for +g: Nat
  for +pn: Nat
  for +index: Nat
  for +y: Tree.Tree<Tile.Tile8>
  for +indexed: {index == Entries.entry_index(p,g,0n) : Nat}
  for +located: {pn == Nat.add(U32.to_nat(start),Nat.mul(g,8n)) : Nat}
  for +inside: {Nat.is_le(Nat.add(g,n),tiles) == True{} : Bool}
  for +room: {Nat.is_le(Nat.add(pn,Nat.mul(n,8n)),Nat.add(grid(p),7n)) == True{} : Bool}
  {PackingStorage.finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,n,tiled(p,x,y,U32.from_nat(index),U32.from_nat(pn)))) ==
    PackingStorage.Packed{x,filled(p,Nat.mul(n,Coordinates.inner(p)),U32.from_nat(index),start,x,y)} : PackingStorage.Packed}
def tiles_fill(limits,p,tiles,start,x,valid,support,total,n,g,pn,index,y,indexed,located,inside,room):
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +cap = U32.to_nat(Machine.capacity(limits))
      +inner = Coordinates.inner(p)
      +m = Nat.mul(rest,8n)
      +position = U32.from_nat(pn)
      +next = filled(p,inner,U32.from_nat(index),start,x,y)
      +below = below_grid(pn,m,grid(p),room)
      +bound = Bounds.strict_le(pn,cap,Bounds.nat_trans(pn,grid(p),cap,below,Padded.spatial_bound_facts(limits,p,valid,support)))
      +below_word = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(pn)),pn,Bounds.roundtrip(pn,Machine.capacity(limits),Limits.capacity_fits(limits),bound)) : {Nat.is_lt(_,grid(p)) == True{} : Bool}
        below : {Nat.is_lt(U32.to_nat(position),grid(p)) == True{} : Bool}}
      +result = {PackingStorage.Packed{x,filled(p,Nat.add(inner,Nat.mul(rest,inner)),U32.from_nat(index),start,x,y)} : PackingStorage.Packed}
      %Equal.sym(LoopLaws.Tiled(),LoopLaws.tile(U32.is_le((position % Entries.native_output_width(p) + 8 : U32),Entries.native_output_width(p)),U32.is_le((position % Entries.native_output_width(p) + 8 : U32),(Entries.native_output_width(p) + Entries.native_output_width(p) : U32)),
          x,y,U32.from_nat(index),position,
          Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),Entries.native_output_width(p),
          Guards.input_channel_count(p),Guards.kernel(p),inner),
          tiled(p,x,next,U32.from_nat(advance(inner,index)),(position + 8 : U32)),
          tile_filled(limits,p,tiles,g,index,start,position,x,y,U32.is_le((position % Entries.native_output_width(p) + 8 : U32),Entries.native_output_width(p)),U32.is_le((position % Entries.native_output_width(p) + 8 : U32),(Entries.native_output_width(p) + Entries.native_output_width(p) : U32)),
            valid,support,total,Order.room_index(g,rest,tiles,inside),group_position(limits,start,g,pn,located,bound),below_word,indexed,{==},{==})) :
        {PackingStorage.finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,rest,_)) == result : PackingStorage.Packed}
      %Equal.sym(U32,(position + U32.from_nat(8n) : U32),U32.from_nat(Nat.add(pn,8n)),WordArithmetic.native_add(pn,8n)) :
        {PackingStorage.finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,rest,tiled(p,x,next,U32.from_nat(advance(inner,index)),_))) == result : PackingStorage.Packed}
      %Equal.sym(Tree.Tree<Tile.Tile8>,filled(p,Nat.add(inner,Nat.mul(rest,inner)),U32.from_nat(index),start,x,y),
          filled(p,Nat.mul(rest,inner),U32.from_nat(advance(inner,index)),start,x,next),grouped(p,inner,Nat.mul(rest,inner),index,start,x,y)) :
        {PackingStorage.finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,rest,tiled(p,x,next,U32.from_nat(advance(inner,index)),U32.from_nat(Nat.add(pn,8n))))) ==
          PackingStorage.Packed{x,_} : PackingStorage.Packed}
      tiles_fill(limits,p,tiles,start,x,valid,support,total,rest,1n+g,Nat.add(pn,8n),advance(inner,index),next,group_carry(inner,g,index,indexed),
        next_located(U32.to_nat(start),Nat.mul(g,8n),pn,located),Order.advance_room(g,rest,tiles,inside),room_ahead(pn,m,Nat.add(grid(p),7n),room))

def channels_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,+valid: Guards.Valid<p>,+support: Entries.Supported(limits,p)) ->
  {U32.to_nat((Coordinates.native_inner(p) / (Entries.native_kernel(p) * Entries.native_kernel(p)) : U32)) == Guards.input_channel_count(p) : Nat}:
  +kernel = Guards.kernel(p)
  +positive = Coordinates.kernel_positive(p,valid)
  +square_positive = Support.positive_mul(kernel,kernel,positive,positive)
  Radix.quotient_word(Coordinates.native_inner(p),(Entries.native_kernel(p) * Entries.native_kernel(p) : U32),Guards.input_channel_count(p),0n,Nat.mul(kernel,kernel),
    Machine.half_word(limits),Limits.half_word_fits(limits),
    Equal.trans(Nat,U32.to_nat(Coordinates.native_inner(p)),Coordinates.inner(p),Nat.add(Coordinates.inner(p),0n),Coordinates.inner_exact(limits,p,valid,support),
      Equal.sym(Nat,Nat.add(Coordinates.inner(p),0n),Coordinates.inner(p),Addition.zero_right(Coordinates.inner(p)))),
    Coordinates.square_exact(limits,p,valid,support),square_positive,square_positive,
    Limits.double_bound(limits,Nat.mul(kernel,kernel),Coordinates.square_bound(limits,p,valid,support)))

def tile_count(+tiles: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool}) -> {Nat.div(Nat.mul(tiles,inner),inner) == tiles : Nat}:
  %Addition.zero_right(Nat.mul(tiles,inner)) : {Nat.div(_,inner) == tiles : Nat}
  Radix.quotient(tiles,0n,inner,positive,positive)

# The packer of tiles * inner entries stores the flat tabulation of its entries.
law packing_contents:
  for +limits: Machine.Limits
  for +p: ConvSpec.Problem
  for +tiles: Nat
  for +start: U32
  for +x: Tree.Tree<F32>
  for +y: Tree.Tree<Tile.Tile8>
  for +valid: Guards.Valid<p>
  for +support: Entries.Supported(limits,p)
  for +total: {Nat.is_le(Nat.mul(tiles,Coordinates.inner(p)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  for +range: {Nat.is_le(Nat.add(U32.to_nat(start),Nat.mul(tiles,8n)),Nat.add(grid(p),7n)) == True{} : Bool}
  {PackingStorage.packing(Nat.mul(tiles,Coordinates.inner(p)),start,x,y,Entries.native_height(p),Entries.native_width(p),Entries.native_kernel(p),Entries.native_stride(p),
      Entries.native_padding(p),Entries.native_output_width(p),Coordinates.native_inner(p)) ==
    PackingStorage.Packed{x,filled(p,Nat.mul(tiles,Coordinates.inner(p)),0,start,x,y)} : PackingStorage.Packed}
def packing_contents(limits,p,tiles,start,x,y,valid,support,total,range):
  +count = Nat.mul(tiles,Coordinates.inner(p))
  +result = {PackingStorage.Packed{x,filled(p,count,0,start,x,y)} : PackingStorage.Packed}
  %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(p)),Coordinates.inner(p),Coordinates.inner_exact(limits,p,valid,support)) :
    {PackingStorage.finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,Nat.div(count,_),Packing.Tiles{x,y,0,start,Entries.native_height(p),Entries.native_width(p),
        Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),Entries.native_output_width(p),
        U32.to_nat((Coordinates.native_inner(p) / (Entries.native_kernel(p) * Entries.native_kernel(p)) : U32)),U32.to_nat(Entries.native_kernel(p)),_})) == result : PackingStorage.Packed}
  %Equal.sym(Nat,U32.to_nat((Coordinates.native_inner(p) / (Entries.native_kernel(p) * Entries.native_kernel(p)) : U32)),Guards.input_channel_count(p),
      channels_exact(limits,p,valid,support)) :
    {PackingStorage.finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,Nat.div(count,Coordinates.inner(p)),Packing.Tiles{x,y,0,start,Entries.native_height(p),Entries.native_width(p),
        Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),Entries.native_output_width(p),_,U32.to_nat(Entries.native_kernel(p)),Coordinates.inner(p)})) ==
      result : PackingStorage.Packed}
  %Equal.sym(Nat,U32.to_nat(Entries.native_kernel(p)),Guards.kernel(p),SupportedCoordinateBounds.kernel_exact(limits,p,support)) :
    {PackingStorage.finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,Nat.div(count,Coordinates.inner(p)),Packing.Tiles{x,y,0,start,Entries.native_height(p),Entries.native_width(p),
        Entries.native_kernel(p),Entries.native_stride(p),Entries.native_padding(p),Entries.native_output_width(p),Guards.input_channel_count(p),_,Coordinates.inner(p)})) ==
      result : PackingStorage.Packed}
  %Equal.sym(Nat,Nat.div(count,Coordinates.inner(p)),tiles,tile_count(tiles,Coordinates.inner(p),Coordinates.inner_positive(p,valid))) :
    {PackingStorage.finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,_,tiled(p,x,y,0,start))) == result : PackingStorage.Packed}
  %Arithmetic.native_roundtrip(start) : {PackingStorage.finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,tiles,tiled(p,x,y,0,_))) == result : PackingStorage.Packed}
  tiles_fill(limits,p,tiles,start,x,valid,support,total,tiles,0n,U32.to_nat(start),0n,y,{==},
    Equal.sym(Nat,Nat.add(U32.to_nat(start),0n),U32.to_nat(start),Addition.zero_right(U32.to_nat(start))),Order.reflexive(tiles),range)
