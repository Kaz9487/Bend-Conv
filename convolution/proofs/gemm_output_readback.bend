# Every in-range output of the actual four-row-unit GEMM is written by exactly
# one job, row and lane; its value is that job's ordered accumulator plus bias.
# Jobs are input-major: index = group * units + unit.
import Base
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_rows.bend as Rows
import ../../lib/kernel_tile.bend as Tile
import ../../lib/traversal_reduction.bend as Reduction
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/kernel_gemm_writes.bend as GemmWrites
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/proofs/nat_division.bend as Division
import ./bounded_write_readback.bend as WriteReadback
import ./conditional_write_coverage.bend as Coverage
import ./gather_write_program.bend as GatherWrites
import ./storage_capacity.bend as StorageCapacity

type Query is Data:
  Query{rows: U32,columns: U32,inner: U32,groups: Nat,limit: U32,
    input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,channel: Nat,column: Nat,
    limit_fits: {U32.is_le(limit,2147483648) == True{} : Bool},
    columns_exact: {U32.to_nat(columns) == Nat.mul(groups,8n) : Nat},
    rows_room: {Nat.is_le(Nat.add(U32.to_nat(rows),3n),U32.to_nat(limit)) == True{} : Bool},
    area_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(columns)),U32.to_nat(limit)) == True{} : Bool},
    channel_inside: {Nat.is_lt(channel,U32.to_nat(rows)) == True{} : Bool},
    column_inside: {Nat.is_lt(column,U32.to_nat(columns)) == True{} : Bool}}

def rows(query: Query) -> U32:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  rows

def columns(query: Query) -> U32:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  columns

def groups(query: Query) -> Nat:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  groups

def limit(query: Query) -> U32:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  limit

def layout(query: Query) -> Gemm.Layout:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  Gemm.Layout{rows,columns,inner}

def input(query: Query) -> Tree.Tree<Tile.Tile8>:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  input

def weights(query: Query) -> Tree.Tree<Tile.Tile8>:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  weights

def bias(query: Query) -> Tree.Tree<F32>:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  bias

def channel(query: Query) -> Nat:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  channel

def column(query: Query) -> Nat:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  column

def inner(query: Query) -> U32:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  inner

def limit_fits(+query: Query) -> {U32.is_le(limit(query),2147483648) == True{} : Bool}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  limit_fits

def columns_exact(+query: Query) -> {U32.to_nat(columns(query)) == Nat.mul(groups(query),8n) : Nat}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  columns_exact

def rows_room(+query: Query) -> {Nat.is_le(Nat.add(U32.to_nat(rows(query)),3n),U32.to_nat(limit(query))) == True{} : Bool}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  rows_room

def area_room(+query: Query) -> {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(limit(query))) == True{} : Bool}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  area_room

def channel_inside(+query: Query) -> {Nat.is_lt(channel(query),U32.to_nat(rows(query))) == True{} : Bool}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  channel_inside

def column_inside(+query: Query) -> {Nat.is_lt(column(query),U32.to_nat(columns(query))) == True{} : Bool}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  column_inside

def select_row(row: Nat,values: Rows.Rows<Tile.Tile8>) -> Tile.Tile8:
  match row:
    case 0n: Rows.first(~Tile.Tile8,values)
    case 1n+a:
      match a:
        case 0n: Rows.second(~Tile.Tile8,values)
        case 1n+b:
          match b:
            case 0n: Rows.third(~Tile.Tile8,values)
            case 1n+c: Rows.fourth(~Tile.Tile8,values)

def select_lane(lane: Nat,values: Tile.Tile8) -> F32:
  TileLaws.select_lane(lane,values)

# Units of four output rows, as Gemm.row_units counts them.
def units(query: Query) -> Nat:
  Nat.div(Nat.add(U32.to_nat(rows(query)),3n),4n)

def job_index(unit: Nat,group: Nat,units: Nat) -> Nat:
  Nat.add(Nat.mul(group,units),unit)

def accumulators(job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>) -> Rows.Rows<Tile.Tile8>:
  Gemm.Job{+context,upper,count} = job
  Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,
    GemmLaws.reduce(upper,count,input,weights,context))

def unit_accumulators(+query: Query,unit: Nat,group: Nat) -> Rows.Rows<Tile.Tile8>:
  accumulators(Gemm.job(layout(query),U32.from_nat(job_index(unit,group,units(query)))),input(query),weights(query))

def output_value(+query: Query,+unit: Nat,row: Nat,+group: Nat,lane: Nat,row_word: U32) -> F32:
  (select_lane(lane,select_row(row,unit_accumulators(query,unit,group))) + Tree.read(F32,bias(query),row_word) : F32)

def address(+query: Query) -> Nat:
  Nat.add(Nat.mul(channel(query),U32.to_nat(columns(query))),column(query))

def wanted(+query: Query) -> F32:
  output_value(query,Nat.div(channel(query),4n),Nat.mod(channel(query),4n),Nat.div(column(query),8n),Nat.mod(column(query),8n),U32.from_nat(channel(query)))

def row_number(unit: Nat,row: Nat) -> Nat:
  Nat.add(Nat.mul(unit,4n),row)

def lane_number(group: Nat,lane: Nat) -> Nat:
  Nat.add(Nat.mul(group,8n),lane)

def columns_positive(+query: Query) -> {Nat.is_lt(0n,U32.to_nat(columns(query))) == True{} : Bool}:
  Order.le_lt_trans(0n,column(query),U32.to_nat(columns(query)),Order.zero_le(column(query)),column_inside(query))

def rows_positive(+query: Query) -> {Nat.is_lt(0n,U32.to_nat(rows(query))) == True{} : Bool}:
  Order.le_lt_trans(0n,channel(query),U32.to_nat(rows(query)),Order.zero_le(channel(query)),channel_inside(query))

def lane_inside(+query: Query,+group: Nat,+lane: Nat,group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},lane_small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Nat.is_lt(lane_number(group,lane),U32.to_nat(columns(query))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(columns(query)),Nat.mul(groups(query),8n),columns_exact(query)) : {Nat.is_lt(lane_number(group,lane),_) == True{} : Bool}
  RowMajor.radix(group,groups(query),lane,8n,group_inside,lane_small)

def row_word_matches(+row_word: U32,+value: Nat,exact: {U32.to_nat(row_word) == value : Nat}) -> {U32.from_nat(value) == row_word : U32}:
  %exact : {U32.from_nat(_) == row_word : U32}
  Arithmetic.native_roundtrip(row_word)

# If the query address equals an actual store address, the store is the one
# selected by the query: same unit, row, group and lane, hence same value.
def selected_value(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,+row_word: U32,
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat},
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},
  +row_small: {Nat.is_lt(row,4n) == True{} : Bool},+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  {output_value(query,unit,row,group,lane,row_word) == wanted(query) : F32}:
  +width = U32.to_nat(columns(query))
  +positive = columns_positive(query)
  +lane_bound = lane_inside(query,group,lane,group_inside,lane_small)
  +channel_same = Radix.row_unique(channel(query),row_number(unit,row),width,column(query),lane_number(group,lane),positive,column_inside(query),lane_bound,same)
  +column_same = Radix.col_unique(channel(query),row_number(unit,row),width,column(query),lane_number(group,lane),positive,column_inside(query),lane_bound,same)
  %Equal.sym(Nat,channel(query),row_number(unit,row),channel_same) :
    {output_value(query,unit,row,group,lane,row_word) ==
      output_value(query,Nat.div(_,4n),Nat.mod(_,4n),Nat.div(column(query),8n),Nat.mod(column(query),8n),U32.from_nat(_)) : F32}
  %Equal.sym(Nat,column(query),lane_number(group,lane),column_same) :
    {output_value(query,unit,row,group,lane,row_word) ==
      output_value(query,Nat.div(row_number(unit,row),4n),Nat.mod(row_number(unit,row),4n),Nat.div(_,8n),Nat.mod(_,8n),U32.from_nat(row_number(unit,row))) : F32}
  %Equal.sym(Nat,Nat.div(row_number(unit,row),4n),unit,Radix.quotient(unit,row,4n,{==},row_small)) :
    {output_value(query,unit,row,group,lane,row_word) ==
      output_value(query,_,Nat.mod(row_number(unit,row),4n),Nat.div(lane_number(group,lane),8n),Nat.mod(lane_number(group,lane),8n),U32.from_nat(row_number(unit,row))) : F32}
  %Equal.sym(Nat,Nat.mod(row_number(unit,row),4n),row,Radix.remainder(unit,row,4n,{==},row_small)) :
    {output_value(query,unit,row,group,lane,row_word) ==
      output_value(query,unit,_,Nat.div(lane_number(group,lane),8n),Nat.mod(lane_number(group,lane),8n),U32.from_nat(row_number(unit,row))) : F32}
  %Equal.sym(Nat,Nat.div(lane_number(group,lane),8n),group,Radix.quotient(group,lane,8n,{==},lane_small)) :
    {output_value(query,unit,row,group,lane,row_word) ==
      output_value(query,unit,row,_,Nat.mod(lane_number(group,lane),8n),U32.from_nat(row_number(unit,row))) : F32}
  %Equal.sym(Nat,Nat.mod(lane_number(group,lane),8n),lane,Radix.remainder(group,lane,8n,{==},lane_small)) :
    {output_value(query,unit,row,group,lane,row_word) ==
      output_value(query,unit,row,group,_,U32.from_nat(row_number(unit,row))) : F32}
  %Equal.sym(U32,U32.from_nat(row_number(unit,row)),row_word,row_word_matches(row_word,row_number(unit,row),row_exact)) :
    {output_value(query,unit,row,group,lane,row_word) == output_value(query,unit,row,group,lane,_) : F32}
  {==}

def put_matches(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,+row_word: U32,+address_word: U32,+equal: Bool,
  exact: {Nat.is_eq(address(query),U32.to_nat(address_word)) == equal : Bool},
  +address_exact: {U32.to_nat(address_word) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat},
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},
  +row_small: {Nat.is_lt(row,4n) == True{} : Bool},+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  Coverage.Matches(equal,output_value(query,unit,row,group,lane,row_word),wanted(query)):
  match equal:
    case False{}: Unit{}
    case True{}:
      selected_value(query,unit,row,group,lane,row_word,
        Equal.trans(Nat,address(query),U32.to_nat(address_word),Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)),
          Order.equal_of_is_eq(address(query),U32.to_nat(address_word),exact),address_exact),
        row_exact,row_small,lane_small,group_inside)

def address_bound(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,
  row_inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},
  group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},lane_small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)),
    Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query)))) == True{} : Bool}:
  RowMajor.radix(row_number(unit,row),U32.to_nat(rows(query)),lane_number(group,lane),U32.to_nat(columns(query)),row_inside,
    lane_inside(query,group,lane,group_inside,lane_small))

def area_within(+query: Query,+capacity: U32,+value: Nat,
  inside: {Nat.is_lt(value,Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query)))) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(value,U32.to_nat(capacity)) == True{} : Bool}:
  Order.transitive(value,Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(capacity),
    Bounds.strict_le(value,Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),inside),room)

def put_evidence(+query: Query,+capacity: U32,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,+row_word: U32,+address_word: U32,+value: F32,
  +rest: List<&2,Writes.Assignment<F32>>,
  +capacity_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(capacity)) == True{} : Bool},
  +address_exact: {U32.to_nat(address_word) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat},
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},
  +row_small: {Nat.is_lt(row,4n) == True{} : Bool},+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},
  value_exact: {value == output_value(query,unit,row,group,lane,row_word) : F32},
  tail: WriteReadback.WriteEvidence<address(query),wanted(query),capacity,rest>) ->
  WriteReadback.WriteEvidence<address(query),wanted(query),capacity,Writes.Put{address_word,value} <> rest>:
  +position = Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane))
  WriteReadback.prepend_evidence(address(query),wanted(query),capacity,address_word,value,rest,
    U32Division.nat_lt_native(address_word,capacity,
      %Equal.sym(Nat,U32.to_nat(address_word),position,address_exact) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
      Bounds.nat_trans(position,Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(capacity),
        address_bound(query,unit,row,group,lane,row_inside,group_inside,lane_small),capacity_room)),
    %Equal.sym(F32,value,output_value(query,unit,row,group,lane,row_word),value_exact) :
      Coverage.Matches(Nat.is_eq(address(query),U32.to_nat(address_word)),_,wanted(query))
    put_matches(query,unit,row,group,lane,row_word,address_word,Nat.is_eq(address(query),U32.to_nat(address_word)),{==},
      address_exact,row_exact,row_small,lane_small,group_inside),
    tail)

def columns_within(+query: Query) -> {Nat.is_le(U32.to_nat(columns(query)),U32.to_nat(limit(query))) == True{} : Bool}:
  Order.transitive(U32.to_nat(columns(query)),Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(limit(query)),
    %RowMajor.multiply_commutative(U32.to_nat(columns(query)),U32.to_nat(rows(query))) : {Nat.is_le(U32.to_nat(columns(query)),_) == True{} : Bool}
    RowMajor.product_covers(U32.to_nat(columns(query)),U32.to_nat(rows(query)),rows_positive(query)),area_room(query))

def lane_offset(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,+offset: U32,
  +offset_exact: {U32.to_nat(offset) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),Nat.mul(group,8n)) : Nat},
  +row_inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {U32.to_nat((offset + U32.from_nat(lane) : U32)) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat}:
  +base = Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),Nat.mul(group,8n))
  +lane_exact = Bounds.roundtrip(lane,limit(query),limit_fits(query),
    Order.transitive(lane,U32.to_nat(columns(query)),U32.to_nat(limit(query)),
      Order.transitive(lane,lane_number(group,lane),U32.to_nat(columns(query)),
        %Order.commutative(lane,Nat.mul(group,8n)) : {Nat.is_le(lane,_) == True{} : Bool}
        Order.le_add(lane,Nat.mul(group,8n)),
        Bounds.strict_le(lane_number(group,lane),U32.to_nat(columns(query)),lane_inside(query,group,lane,group_inside,lane_small))),
      columns_within(query)))
  %Addition.associative(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),Nat.mul(group,8n),lane) :
    {U32.to_nat((offset + U32.from_nat(lane) : U32)) == _ : Nat}
  %offset_exact : {U32.to_nat((offset + U32.from_nat(lane) : U32)) == Nat.add(_,lane) : Nat}
  %lane_exact : {U32.to_nat((offset + U32.from_nat(lane) : U32)) == Nat.add(U32.to_nat(offset),_) : Nat}
  Arithmetic.add(offset,U32.from_nat(lane),limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(offset),base,offset_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(lane))),U32.to_nat(limit(query))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,lane_exact) : {Nat.is_le(Nat.add(base,_),U32.to_nat(limit(query))) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),Nat.mul(group,8n)),lane),
      Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)),
      Addition.associative(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),Nat.mul(group,8n),lane)) : {Nat.is_le(_,U32.to_nat(limit(query))) == True{} : Bool}
    area_within(query,limit(query),Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)),
      address_bound(query,unit,row,group,lane,row_inside,group_inside,lane_small),area_room(query)))

def row_inside(+query: Query,+unit: Nat,+row: Nat,+row_word: U32,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},valid: {U32.is_lt(row_word,rows(query)) == True{} : Bool}) ->
  {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool}:
  %row_exact : {Nat.is_lt(_,U32.to_nat(rows(query))) == True{} : Bool}
  %Comparison.lt(row_word,rows(query)) : {_ == True{} : Bool}
  valid

def offset_exact(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +row_inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  {U32.to_nat((row_word * columns(query) + position_word : U32)) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),Nat.mul(group,8n)) : Nat}:
  +width = U32.to_nat(columns(query))
  +product = (row_word * columns(query) : U32)
  +first = Nat.add(Nat.mul(row_number(unit,row),width),lane_number(group,0n))
  +first_inside = address_bound(query,unit,row,group,0n,row_inside,group_inside,{==})
  +product_exact = Arithmetic.multiply(row_word,columns(query),limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(row_word),row_number(unit,row),row_exact) : {Nat.is_le(Nat.mul(_,width),U32.to_nat(limit(query))) == True{} : Bool}
    Order.transitive(Nat.mul(row_number(unit,row),width),Nat.mul(U32.to_nat(rows(query)),width),U32.to_nat(limit(query)),
      RowMajor.mul_le(row_number(unit,row),U32.to_nat(rows(query)),width,Bounds.strict_le(row_number(unit,row),U32.to_nat(rows(query)),row_inside)),area_room(query)))
  +sum_exact = Arithmetic.add(product,position_word,limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(product),Nat.mul(U32.to_nat(row_word),width),product_exact) :
      {Nat.is_le(Nat.add(_,U32.to_nat(position_word)),U32.to_nat(limit(query))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(row_word),row_number(unit,row),row_exact) :
      {Nat.is_le(Nat.add(Nat.mul(_,width),U32.to_nat(position_word)),U32.to_nat(limit(query))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(position_word),Nat.mul(group,8n),position_exact) :
      {Nat.is_le(Nat.add(Nat.mul(row_number(unit,row),width),_),U32.to_nat(limit(query))) == True{} : Bool}
    %Addition.zero_right(Nat.mul(group,8n)) :
      {Nat.is_le(Nat.add(Nat.mul(row_number(unit,row),width),_),U32.to_nat(limit(query))) == True{} : Bool}
    area_within(query,limit(query),first,first_inside,area_room(query)))
  %Equal.sym(Nat,U32.to_nat((product + position_word : U32)),Nat.add(U32.to_nat(product),U32.to_nat(position_word)),sum_exact) :
    {_ == Nat.add(Nat.mul(row_number(unit,row),width),Nat.mul(group,8n)) : Nat}
  %Equal.sym(Nat,U32.to_nat(product),Nat.mul(U32.to_nat(row_word),width),product_exact) :
    {Nat.add(_,U32.to_nat(position_word)) == Nat.add(Nat.mul(row_number(unit,row),width),Nat.mul(group,8n)) : Nat}
  %row_exact : {Nat.add(Nat.mul(U32.to_nat(row_word),width),U32.to_nat(position_word)) == Nat.add(Nat.mul(_,width),Nat.mul(group,8n)) : Nat}
  %position_exact : {Nat.add(Nat.mul(U32.to_nat(row_word),width),U32.to_nat(position_word)) == Nat.add(Nat.mul(U32.to_nat(row_word),width),_) : Nat}
  {==}

def first_lane(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+offset: U32,
  +exact: {U32.to_nat(offset) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),Nat.mul(group,8n)) : Nat}) ->
  {U32.to_nat(offset) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,0n)) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.mul(group,8n),0n),Nat.mul(group,8n),Addition.zero_right(Nat.mul(group,8n))) :
    {U32.to_nat(offset) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),_) : Nat}
  exact

def lane_value(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,+row_word: U32,+values: Tile.Tile8,
  +values_exact: {values == select_row(row,unit_accumulators(query,unit,group)) : Tile.Tile8}) ->
  {(select_lane(lane,values) + Tree.read(F32,bias(query),row_word) : F32) == output_value(query,unit,row,group,lane,row_word) : F32}:
  Equal.cong(Tile.Tile8,F32,selected => (select_lane(lane,selected) + Tree.read(F32,bias(query),row_word) : F32),
    values,select_row(row,unit_accumulators(query,unit,group)),values_exact)

def row_evidence(+query: Query,+capacity: U32,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+values: Tile.Tile8,+valid: Bool,
  +valid_exact: {U32.is_lt(row_word,rows(query)) == valid : Bool},
  +capacity_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(capacity)) == True{} : Bool},
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +row_small: {Nat.is_lt(row,4n) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +values_exact: {values == select_row(row,unit_accumulators(query,unit,group)) : Tile.Tile8}) ->
  WriteReadback.WriteEvidence<address(query),wanted(query),capacity,GemmWrites.row_writes(valid,row_word,position_word,columns(query),values,bias(query))>:
  match valid:
    case False{}: WriteReadback.WriteEvidence{Unit{},Unit{}}
    case True{}:
      +inside = row_inside(query,unit,row,row_word,row_exact,valid_exact)
      +offset = (row_word * columns(query) + position_word : U32)
      +exact = offset_exact(query,unit,row,group,row_word,position_word,row_exact,position_exact,inside,group_inside)
      +bias_value = Tree.read(F32,bias(query),row_word)
      +o1 = (offset + 1 : U32)
      +o2 = (offset + 2 : U32)
      +o3 = (offset + 3 : U32)
      +o4 = (offset + 4 : U32)
      +o5 = (offset + 5 : U32)
      +o6 = (offset + 6 : U32)
      +o7 = (offset + 7 : U32)
      +v0 = (Tile.lane0(values) + bias_value : F32)
      +v1 = (Tile.lane1(values) + bias_value : F32)
      +v2 = (Tile.lane2(values) + bias_value : F32)
      +v3 = (Tile.lane3(values) + bias_value : F32)
      +v4 = (Tile.lane4(values) + bias_value : F32)
      +v5 = (Tile.lane5(values) + bias_value : F32)
      +v6 = (Tile.lane6(values) + bias_value : F32)
      +v7 = (Tile.lane7(values) + bias_value : F32)
      put_evidence(query,capacity,unit,row,group,0n,row_word,offset,v0,[Writes.Put{o1,v1},Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
        capacity_room,first_lane(query,unit,row,group,offset,exact),row_exact,row_small,{==},group_inside,inside,
        lane_value(query,unit,row,group,0n,row_word,values,values_exact),
        put_evidence(query,capacity,unit,row,group,1n,row_word,o1,v1,[Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
          capacity_room,lane_offset(query,unit,row,group,1n,offset,exact,inside,group_inside,{==}),row_exact,row_small,{==},group_inside,inside,
          lane_value(query,unit,row,group,1n,row_word,values,values_exact),
          put_evidence(query,capacity,unit,row,group,2n,row_word,o2,v2,[Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
            capacity_room,lane_offset(query,unit,row,group,2n,offset,exact,inside,group_inside,{==}),row_exact,row_small,{==},group_inside,inside,
            lane_value(query,unit,row,group,2n,row_word,values,values_exact),
            put_evidence(query,capacity,unit,row,group,3n,row_word,o3,v3,[Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
              capacity_room,lane_offset(query,unit,row,group,3n,offset,exact,inside,group_inside,{==}),row_exact,row_small,{==},group_inside,inside,
              lane_value(query,unit,row,group,3n,row_word,values,values_exact),
              put_evidence(query,capacity,unit,row,group,4n,row_word,o4,v4,[Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
                capacity_room,lane_offset(query,unit,row,group,4n,offset,exact,inside,group_inside,{==}),row_exact,row_small,{==},group_inside,inside,
                lane_value(query,unit,row,group,4n,row_word,values,values_exact),
                put_evidence(query,capacity,unit,row,group,5n,row_word,o5,v5,[Writes.Put{o6,v6},Writes.Put{o7,v7}],
                  capacity_room,lane_offset(query,unit,row,group,5n,offset,exact,inside,group_inside,{==}),row_exact,row_small,{==},group_inside,inside,
                  lane_value(query,unit,row,group,5n,row_word,values,values_exact),
                  put_evidence(query,capacity,unit,row,group,6n,row_word,o6,v6,[Writes.Put{o7,v7}],
                    capacity_room,lane_offset(query,unit,row,group,6n,offset,exact,inside,group_inside,{==}),row_exact,row_small,{==},group_inside,inside,
                    lane_value(query,unit,row,group,6n,row_word,values,values_exact),
                    put_evidence(query,capacity,unit,row,group,7n,row_word,o7,v7,[],
                      capacity_room,lane_offset(query,unit,row,group,7n,offset,exact,inside,group_inside,{==}),row_exact,row_small,{==},group_inside,inside,
                      lane_value(query,unit,row,group,7n,row_word,values,values_exact),
                      WriteReadback.WriteEvidence{Unit{},Unit{}}))))))))

def add_small(+bound: Nat,+small: Nat,+large: Nat,h: {Nat.is_le(small,large) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(bound,small),Nat.add(bound,large)) == True{} : Bool}:
  %Order.commutative(small,bound) : {Nat.is_le(_,Nat.add(bound,large)) == True{} : Bool}
  %Order.commutative(large,bound) : {Nat.is_le(Nat.add(small,bound),_) == True{} : Bool}
  Order.add_le(small,large,bound,h)

def double_within_four(+value: Nat) -> {Nat.is_le(Nat.double(value),Nat.mul(value,4n)) == True{} : Bool}:
  match value:
    case 0n: {==}
    case 1n+ +rest:
      Order.successor(Nat.double(rest),1n+Nat.mul(rest,4n),Order.successor(Nat.double(rest),Nat.mul(rest,4n),double_within_four(rest)))

def groups_within_columns(+query: Query) -> {Nat.is_le(groups(query),U32.to_nat(columns(query))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(columns(query)),Nat.mul(groups(query),8n),columns_exact(query)) : {Nat.is_le(groups(query),_) == True{} : Bool}
  RowMajor.product_covers(groups(query),8n,{==})

def groups_word_exact(+query: Query) -> {U32.to_nat(Gemm.column_groups(columns(query))) == groups(query) : Nat}:
  Equal.trans(Nat,U32.to_nat(Gemm.column_groups(columns(query))),Nat.div(U32.to_nat(columns(query)),8n),groups(query),
    U32Division.divide(columns(query),8,2147483648,{==},{==},{==}),
    %Equal.sym(Nat,U32.to_nat(columns(query)),Nat.mul(groups(query),8n),columns_exact(query)) : {Nat.div(_,8n) == groups(query) : Nat}
    %Addition.zero_right(Nat.mul(groups(query),8n)) : {Nat.div(_,8n) == groups(query) : Nat}
    Radix.quotient(groups(query),0n,8n,{==},{==}))

def units_word_exact(+query: Query) -> {U32.to_nat(Gemm.row_units(rows(query))) == units(query) : Nat}:
  Equal.trans(Nat,U32.to_nat(Gemm.row_units(rows(query))),Nat.div(U32.to_nat((rows(query) + 3 : U32)),4n),units(query),
    U32Division.divide((rows(query) + 3 : U32),4,2147483648,{==},{==},{==}),
    Equal.cong(Nat,Nat,value => Nat.div(value,4n),U32.to_nat((rows(query) + 3 : U32)),Nat.add(U32.to_nat(rows(query)),3n),
      Arithmetic.add(rows(query),3,limit(query),limit_fits(query),rows_room(query))))

def decomposed(+value: Nat,+divisor: Nat,+limit: U32,+fits: {U32.is_le(limit,2147483648) == True{} : Bool},
  +small: {Nat.is_le(value,U32.to_nat(limit)) == True{} : Bool},+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {value == Nat.add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor)) : Nat}:
  %Bounds.roundtrip(value,limit,fits,small) :
    {_ == Nat.add(Nat.mul(Nat.div(_,divisor),divisor),Nat.mod(_,divisor)) : Nat}
  Radix.division_identity(U32.from_nat(value),divisor,positive)

def remainder_small(+value: Nat,+divisor: Nat,+limit: U32,+fits: {U32.is_le(limit,2147483648) == True{} : Bool},
  +small: {Nat.is_le(value,U32.to_nat(limit)) == True{} : Bool},+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mod(value,divisor),divisor) == True{} : Bool}:
  %Bounds.roundtrip(value,limit,fits,small) : {Nat.is_lt(Nat.mod(_,divisor),divisor) == True{} : Bool}
  Radix.remainder_inside(U32.from_nat(value),divisor,positive)

def units_room(+query: Query) -> {Nat.is_le(Nat.mul(units(query),4n),Nat.add(U32.to_nat(rows(query)),3n)) == True{} : Bool}:
  +padded = Nat.add(U32.to_nat(rows(query)),3n)
  %Equal.sym(Nat,padded,Nat.add(Nat.mul(units(query),4n),Nat.mod(padded,4n)),decomposed(padded,4n,limit(query),limit_fits(query),rows_room(query),{==})) :
    {Nat.is_le(Nat.mul(units(query),4n),_) == True{} : Bool}
  Order.le_add(Nat.mul(units(query),4n),Nat.mod(padded,4n))

# A unit of four fits below the channel count whenever units*4 <= rows+3.
def unit_row_room(+rows: Nat,+unit: Nat,room: {Nat.is_le(Nat.mul(1n+unit,4n),Nat.add(rows,3n)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(unit,4n),rows) == True{} : Bool}:
  Bounds.predecessor_strict(Nat.mul(unit,4n),rows,
    %Order.commutative(rows,3n) : {Nat.is_le(Nat.mul(1n+unit,4n),_) == True{} : Bool}
    room)

def units_within_rows(+rows: Nat,+count: Nat,room: {Nat.is_le(Nat.mul(count,4n),Nat.add(rows,3n)) == True{} : Bool}) ->
  {Nat.is_le(count,rows) == True{} : Bool}:
  match count:
    case 0n: Order.zero_le(rows)
    case 1n+ +unit:
      Order.strict_follow(unit,rows,Order.le_lt_trans(unit,Nat.mul(unit,4n),rows,RowMajor.product_covers(unit,4n,{==}),unit_row_room(rows,unit,room)))

def unit_bound(+query: Query,+unit: Nat,inside: {Nat.is_lt(unit,units(query)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool}:
  unit_row_room(U32.to_nat(rows(query)),unit,
    Order.transitive(Nat.mul(1n+unit,4n),Nat.mul(units(query),4n),Nat.add(U32.to_nat(rows(query)),3n),
      RowMajor.mul_le(1n+unit,units(query),4n,Order.strict_follow(unit,units(query),inside)),units_room(query)))

def unit_before(+rows: Nat,+count: Nat,+unit: Nat,+remainder: Nat,
  +unit_inside: {Nat.is_lt(Nat.mul(unit,4n),rows) == True{} : Bool},
  +identity: {Nat.add(rows,3n) == Nat.add(Nat.mul(count,4n),remainder) : Nat},+remainder_small: {Nat.is_lt(remainder,4n) == True{} : Bool}) ->
  {Nat.is_lt(unit,count) == True{} : Bool}:
  RowMajor.cancel_multiply_lt(1n+unit,1n+count,4n,
    Order.le_lt_trans(Nat.mul(1n+unit,4n),Nat.add(rows,3n),Nat.mul(1n+count,4n),
      %Order.commutative(3n,rows) : {Nat.is_le(Nat.mul(1n+unit,4n),_) == True{} : Bool}
      Order.strict_follow(Nat.mul(unit,4n),rows,unit_inside),
      %Equal.sym(Nat,Nat.add(rows,3n),Nat.add(Nat.mul(count,4n),remainder),identity) : {Nat.is_lt(_,Nat.mul(1n+count,4n)) == True{} : Bool}
      %Order.commutative(remainder,Nat.mul(count,4n)) : {Nat.is_lt(_,Nat.mul(1n+count,4n)) == True{} : Bool}
      Order.add_lt(remainder,4n,Nat.mul(count,4n),remainder_small)))

def unit_counted(+query: Query,+unit: Nat,+unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool}) ->
  {Nat.is_lt(unit,units(query)) == True{} : Bool}:
  +padded = Nat.add(U32.to_nat(rows(query)),3n)
  unit_before(U32.to_nat(rows(query)),units(query),unit,Nat.mod(padded,4n),unit_rows,
    decomposed(padded,4n,limit(query),limit_fits(query),rows_room(query),{==}),
    remainder_small(padded,4n,limit(query),limit_fits(query),rows_room(query),{==}))

def units_positive(+query: Query) -> {Nat.is_lt(0n,units(query)) == True{} : Bool}:
  +padded = Nat.add(U32.to_nat(rows(query)),3n)
  unit_before(U32.to_nat(rows(query)),units(query),0n,Nat.mod(padded,4n),rows_positive(query),
    decomposed(padded,4n,limit(query),limit_fits(query),rows_room(query),{==}),
    remainder_small(padded,4n,limit(query),limit_fits(query),rows_room(query),{==}))

def double_units(+query: Query) -> {Nat.is_le(Nat.double(units(query)),U32.to_nat(limit(query))) == True{} : Bool}:
  Order.transitive(Nat.double(units(query)),Nat.mul(units(query),4n),U32.to_nat(limit(query)),double_within_four(units(query)),
    Order.transitive(Nat.mul(units(query),4n),Nat.add(U32.to_nat(rows(query)),3n),U32.to_nat(limit(query)),units_room(query),rows_room(query)))

def jobs_within(+query: Query) -> {Nat.is_le(Nat.mul(groups(query),units(query)),U32.to_nat(limit(query))) == True{} : Bool}:
  +rows_nat = U32.to_nat(rows(query))
  +width = U32.to_nat(columns(query))
  +count = units(query)
  Order.transitive(Nat.mul(groups(query),count),Nat.mul(rows_nat,width),U32.to_nat(limit(query)),
    Order.transitive(Nat.mul(groups(query),count),Nat.mul(width,count),Nat.mul(rows_nat,width),
      RowMajor.mul_le(groups(query),width,count,groups_within_columns(query)),
      %RowMajor.multiply_commutative(count,width) : {Nat.is_le(_,Nat.mul(rows_nat,width)) == True{} : Bool}
      RowMajor.mul_le(count,rows_nat,width,units_within_rows(rows_nat,count,units_room(query)))),
    area_room(query))

def index_exact(+query: Query,+unit: Nat,+group: Nat,
  unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(job_index(unit,group,units(query)))) == job_index(unit,group,units(query)) : Nat}:
  Bounds.roundtrip(job_index(unit,group,units(query)),limit(query),limit_fits(query),
    Order.transitive(job_index(unit,group,units(query)),Nat.mul(groups(query),units(query)),U32.to_nat(limit(query)),
      Bounds.strict_le(job_index(unit,group,units(query)),Nat.mul(groups(query),units(query)),
        RowMajor.radix(group,groups(query),unit,units(query),group_inside,unit_inside)),
      jobs_within(query)))

def unit_word(+query: Query,unit: Nat,group: Nat) -> U32:
  (U32.from_nat(job_index(unit,group,units(query))) % Gemm.row_units(rows(query)) : U32)

def group_word(+query: Query,unit: Nat,group: Nat) -> U32:
  (U32.from_nat(job_index(unit,group,units(query))) / Gemm.row_units(rows(query)) : U32)

def unit_exact(+query: Query,+unit: Nat,+group: Nat,
  +unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  {U32.to_nat(unit_word(query,unit,group)) == unit : Nat}:
  Radix.remainder_word(U32.from_nat(job_index(unit,group,units(query))),Gemm.row_units(rows(query)),group,unit,units(query),limit(query),limit_fits(query),
    index_exact(query,unit,group,unit_inside,group_inside),units_word_exact(query),units_positive(query),unit_inside,double_units(query))

def group_exact(+query: Query,+unit: Nat,+group: Nat,
  +unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  {U32.to_nat(group_word(query,unit,group)) == group : Nat}:
  Radix.quotient_word(U32.from_nat(job_index(unit,group,units(query))),Gemm.row_units(rows(query)),group,unit,units(query),limit(query),limit_fits(query),
    index_exact(query,unit,group,unit_inside,group_inside),units_word_exact(query),units_positive(query),unit_inside,double_units(query))

def first_row_exact(+query: Query,+unit: Nat,+word: U32,+exact: {U32.to_nat(word) == unit : Nat},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool}) ->
  {U32.to_nat((word * 4 : U32)) == Nat.mul(unit,4n) : Nat}:
  %exact : {U32.to_nat((word * 4 : U32)) == Nat.mul(_,4n) : Nat}
  Arithmetic.multiply(word,4,limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(word),unit,exact) : {Nat.is_le(Nat.mul(_,4n),U32.to_nat(limit(query))) == True{} : Bool}
    Order.transitive(Nat.mul(unit,4n),U32.to_nat(rows(query)),U32.to_nat(limit(query)),Bounds.strict_le(Nat.mul(unit,4n),U32.to_nat(rows(query)),unit_rows),
      Order.transitive(U32.to_nat(rows(query)),Nat.add(U32.to_nat(rows(query)),3n),U32.to_nat(limit(query)),Order.le_add(U32.to_nat(rows(query)),3n),rows_room(query))))

def row_zero_exact(+query: Query,+unit: Nat,+word: U32,+exact: {U32.to_nat(word) == unit : Nat},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool}) ->
  {U32.to_nat((word * 4 : U32)) == row_number(unit,0n) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.mul(unit,4n),0n),Nat.mul(unit,4n),Addition.zero_right(Nat.mul(unit,4n))) : {U32.to_nat((word * 4 : U32)) == _ : Nat}
  first_row_exact(query,unit,word,exact,unit_rows)

def row_offset_exact(+query: Query,+unit: Nat,+word: U32,+row: Nat,+exact: {U32.to_nat(word) == unit : Nat},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool},+row_bound: {Nat.is_le(row,3n) == True{} : Bool}) ->
  {U32.to_nat(((word * 4 : U32) + U32.from_nat(row) : U32)) == row_number(unit,row) : Nat}:
  +rows_nat = U32.to_nat(rows(query))
  +product = (word * 4 : U32)
  +row_exact = Bounds.roundtrip(row,limit(query),limit_fits(query),
    Order.transitive(row,3n,U32.to_nat(limit(query)),row_bound,
      Order.transitive(3n,Nat.add(rows_nat,3n),U32.to_nat(limit(query)),
        %Order.commutative(3n,rows_nat) : {Nat.is_le(3n,_) == True{} : Bool}
        Order.le_add(3n,rows_nat),rows_room(query))))
  +product_exact = first_row_exact(query,unit,word,exact,unit_rows)
  %product_exact : {U32.to_nat((product + U32.from_nat(row) : U32)) == Nat.add(_,row) : Nat}
  %row_exact : {U32.to_nat((product + U32.from_nat(row) : U32)) == Nat.add(U32.to_nat(product),_) : Nat}
  Arithmetic.add(product,U32.from_nat(row),limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(product),Nat.mul(unit,4n),product_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(row))),U32.to_nat(limit(query))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(row)),row,row_exact) : {Nat.is_le(Nat.add(Nat.mul(unit,4n),_),U32.to_nat(limit(query))) == True{} : Bool}
    Order.transitive(Nat.add(Nat.mul(unit,4n),row),Nat.add(rows_nat,row),U32.to_nat(limit(query)),
      Order.add_le(Nat.mul(unit,4n),rows_nat,row,Bounds.strict_le(Nat.mul(unit,4n),rows_nat,unit_rows)),
      Order.transitive(Nat.add(rows_nat,row),Nat.add(rows_nat,3n),U32.to_nat(limit(query)),add_small(rows_nat,row,3n,row_bound),rows_room(query))))

def position_exact(+query: Query,+group: Nat,+word: U32,+exact: {U32.to_nat(word) == group : Nat},
  +group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  {U32.to_nat((word * 8 : U32)) == Nat.mul(group,8n) : Nat}:
  %exact : {U32.to_nat((word * 8 : U32)) == Nat.mul(_,8n) : Nat}
  Arithmetic.multiply(word,8,limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(word),group,exact) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(limit(query))) == True{} : Bool}
    Order.transitive(Nat.mul(group,8n),U32.to_nat(columns(query)),U32.to_nat(limit(query)),
      %Equal.sym(Nat,U32.to_nat(columns(query)),Nat.mul(groups(query),8n),columns_exact(query)) : {Nat.is_le(Nat.mul(group,8n),_) == True{} : Bool}
      RowMajor.mul_le(group,groups(query),8n,Bounds.strict_le(group,groups(query),group_inside)),columns_within(query)))

def job_writes(+query: Query,+unit: Nat,+group: Nat) -> List<&2,Writes.Assignment<F32>>:
  GemmWrites.job_writes(Gemm.job(layout(query),U32.from_nat(job_index(unit,group,units(query)))),input(query),weights(query),bias(query))

def JobEvidence(+query: Query,+capacity: U32,+writes: List<&2,Writes.Assignment<F32>>) -> Data:
  WriteReadback.WriteEvidence<address(query),wanted(query),capacity,writes>

def job_evidence(+query: Query,+capacity: U32,+unit: Nat,+group: Nat,
  +capacity_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(capacity)) == True{} : Bool},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool},
  +unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  JobEvidence(query,capacity,job_writes(query,unit,group)):
  Query{+rows,+columns,+inner,+groups,+limit,+input,+weights,+bias,+channel,+column,+limit_fits,+columns_exact,+rows_room,+area_room,+channel_inside,+column_inside} = query
  +query = {Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} : Query}
  +word = unit_word(query,unit,group)
  +group_value = group_word(query,unit,group)
  +word_exact = unit_exact(query,unit,group,unit_inside,group_inside)
  +position = (group_value * 8 : U32)
  +position_exact = position_exact(query,group,group_value,group_exact(query,unit,group,unit_inside,group_inside),group_inside)
  +accumulators = unit_accumulators(query,unit,group)
  +index = (word * 4 : U32)
  +i1 = (index + 1 : U32)
  +i2 = (index + 2 : U32)
  +i3 = (index + 3 : U32)
  +a0 = Rows.first(~Tile.Tile8,accumulators)
  +a1 = Rows.second(~Tile.Tile8,accumulators)
  +a2 = Rows.third(~Tile.Tile8,accumulators)
  +a3 = Rows.fourth(~Tile.Tile8,accumulators)
  +w0 = GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,a0,bias)
  +w1 = GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,a1,bias)
  +w2 = GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,a2,bias)
  +w3 = GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,a3,bias)
  +l01 = List.append(&2,Writes.Assignment<F32>,w0,w1)
  +l02 = List.append(&2,Writes.Assignment<F32>,l01,w2)
  +e0 = row_evidence(query,capacity,unit,0n,group,index,position,a0,U32.is_lt(index,rows),{==},capacity_room,
    row_zero_exact(query,unit,word,word_exact,unit_rows),position_exact,{==},group_inside,{==})
  +e1 = row_evidence(query,capacity,unit,1n,group,i1,position,a1,U32.is_lt(i1,rows),{==},capacity_room,
    row_offset_exact(query,unit,word,1n,word_exact,unit_rows,{==}),position_exact,{==},group_inside,{==})
  +e2 = row_evidence(query,capacity,unit,2n,group,i2,position,a2,U32.is_lt(i2,rows),{==},capacity_room,
    row_offset_exact(query,unit,word,2n,word_exact,unit_rows,{==}),position_exact,{==},group_inside,{==})
  +e3 = row_evidence(query,capacity,unit,3n,group,i3,position,a3,U32.is_lt(i3,rows),{==},capacity_room,
    row_offset_exact(query,unit,word,3n,word_exact,unit_rows,{==}),position_exact,{==},group_inside,{==})
  +q = address(query)
  +v = wanted(query)
  WriteReadback.concatenated_evidence(q,v,capacity,l02,w3,
    WriteReadback.concatenated_evidence(q,v,capacity,l01,w2,
      WriteReadback.concatenated_evidence(q,v,capacity,w0,w1,e0,e1),e2),e3)

def job_evidence_at(+query: Query,+capacity: U32,+index: Nat,
  +capacity_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(capacity)) == True{} : Bool},
  +inside: {Nat.is_lt(index,Nat.mul(groups(query),units(query))) == True{} : Bool}) ->
  JobEvidence(query,capacity,GemmWrites.job_writes(Gemm.job(layout(query),U32.from_nat(index)),input(query),weights(query),bias(query))):
  +count = units(query)
  +small = Order.transitive(index,Nat.mul(groups(query),count),U32.to_nat(limit(query)),Bounds.strict_le(index,Nat.mul(groups(query),count),inside),jobs_within(query))
  +unit = Nat.mod(index,count)
  +group = Nat.div(index,count)
  +identity = decomposed(index,count,limit(query),limit_fits(query),small,units_positive(query))
  +unit_inside = remainder_small(index,count,limit(query),limit_fits(query),small,units_positive(query))
  +group_inside = RowMajor.row_inside(group,groups(query),count,unit,
    %identity : {Nat.is_lt(_,Nat.mul(groups(query),count)) == True{} : Bool}
    inside)
  %Equal.sym(Nat,index,job_index(unit,group,count),identity) :
    JobEvidence(query,capacity,GemmWrites.job_writes(Gemm.job(layout(query),U32.from_nat(_)),input(query),weights(query),bias(query)))
  job_evidence(query,capacity,unit,group,capacity_room,unit_bound(query,unit,unit_inside),unit_inside,group_inside)

def all_evidence(+query: Query,+capacity: U32,+count: Nat,+first: Nat,
  +capacity_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(capacity)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(first,count),Nat.mul(groups(query),units(query))) == True{} : Bool}) ->
  JobEvidence(query,capacity,GemmWrites.all_writes(count,U32.from_nat(first),layout(query),input(query),weights(query),bias(query))):
  match count:
    case 0n: WriteReadback.WriteEvidence{Unit{},Unit{}}
    case 1n+ +rest:
      +here = GemmWrites.job_writes(Gemm.job(layout(query),U32.from_nat(first)),input(query),weights(query),bias(query))
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        JobEvidence(query,capacity,List.append(&2,Writes.Assignment<F32>,here,
          GemmWrites.all_writes(rest,_,layout(query),input(query),weights(query),bias(query))))
      WriteReadback.concatenated_evidence(address(query),wanted(query),capacity,here,
        GemmWrites.all_writes(rest,U32.from_nat(1n+first),layout(query),input(query),weights(query),bias(query)),
        job_evidence_at(query,capacity,first,capacity_room,Order.room_index(first,rest,Nat.mul(groups(query),units(query)),room)),
        all_evidence(query,capacity,rest,1n+first,capacity_room,Order.advance_room(first,rest,Nat.mul(groups(query),units(query)),room)))

def contains(+query: Query,writes: List<&2,Writes.Assignment<F32>>) -> Bool:
  Coverage.contains(address(query),WriteReadback.natural_write_addresses(writes))

def Covered(+query: Query,+writes: List<&2,Writes.Assignment<F32>>) -> Data:
  {contains(query,writes) == True{} : Bool}

def appended_left(+query: Query,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>,covered: Covered(query,left)) ->
  Covered(query,List.append(&2,Writes.Assignment<F32>,left,right)):
  GatherWrites.concatenation_covered(address(query),left,right,Inl{covered})

def appended_right(+query: Query,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>,covered: Covered(query,right)) ->
  Covered(query,List.append(&2,Writes.Assignment<F32>,left,right)):
  GatherWrites.concatenation_covered(address(query),left,right,Inr{covered})

def hit(+query: Query,+destination: U32,+value: F32,+rest: List<&2,Writes.Assignment<F32>>,
  same: {Nat.is_eq(address(query),U32.to_nat(destination)) == True{} : Bool}) -> Covered(query,Writes.Put{destination,value} <> rest):
  %Equal.sym(Bool,Nat.is_eq(address(query),U32.to_nat(destination)),True{},same) : {_ || contains(query,rest) == True{} : Bool}
  {==}

def later(+query: Query,+destination: U32,+value: F32,+rest: List<&2,Writes.Assignment<F32>>,covered: Covered(query,rest)) ->
  Covered(query,Writes.Put{destination,value} <> rest):
  %Equal.sym(Bool,contains(query,rest),True{},covered) : {Nat.is_eq(address(query),U32.to_nat(destination)) || _ == True{} : Bool}
  Boolean.or_true_right(Nat.is_eq(address(query),U32.to_nat(destination)))

def lane_hit(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,+destination: U32,
  +exact: {U32.to_nat(destination) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat}) ->
  {Nat.is_eq(address(query),U32.to_nat(destination)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(destination),Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)),exact) :
    {Nat.is_eq(address(query),_) == True{} : Bool}
  %same : {Nat.is_eq(address(query),_) == True{} : Bool}
  Order.equal_reflexive(address(query))

def row_accepted(+query: Query,+unit: Nat,+row: Nat,+row_word: U32,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool}) ->
  {U32.is_lt(row_word,rows(query)) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt(row_word,rows(query)),Nat.is_lt(U32.to_nat(row_word),U32.to_nat(rows(query))),Comparison.lt(row_word,rows(query))) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(row_word),row_number(unit,row),row_exact) : {Nat.is_lt(_,U32.to_nat(rows(query))) == True{} : Bool}
  inside

def lane0_hit(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+values: Tile.Tile8,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,0n)) : Nat}) ->
  Covered(query,GemmWrites.row_writes(True{},row_word,position_word,columns(query),values,bias(query))):
  +offset = (row_word * columns(query) + position_word : U32)
  +exact = offset_exact(query,unit,row,group,row_word,position_word,row_exact,position_exact,inside,group_inside)
  +bias_value = Tree.read(F32,bias(query),row_word)
  +o1 = (offset + 1 : U32)
  +o2 = (offset + 2 : U32)
  +o3 = (offset + 3 : U32)
  +o4 = (offset + 4 : U32)
  +o5 = (offset + 5 : U32)
  +o6 = (offset + 6 : U32)
  +o7 = (offset + 7 : U32)
  +v0 = (Tile.lane0(values) + bias_value : F32)
  +v1 = (Tile.lane1(values) + bias_value : F32)
  +v2 = (Tile.lane2(values) + bias_value : F32)
  +v3 = (Tile.lane3(values) + bias_value : F32)
  +v4 = (Tile.lane4(values) + bias_value : F32)
  +v5 = (Tile.lane5(values) + bias_value : F32)
  +v6 = (Tile.lane6(values) + bias_value : F32)
  +v7 = (Tile.lane7(values) + bias_value : F32)
  hit(query,offset,v0,[Writes.Put{o1,v1},Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
    lane_hit(query,unit,row,group,0n,offset,first_lane(query,unit,row,group,offset,exact),same))

def lane1_hit(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+values: Tile.Tile8,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,1n)) : Nat}) ->
  Covered(query,GemmWrites.row_writes(True{},row_word,position_word,columns(query),values,bias(query))):
  +offset = (row_word * columns(query) + position_word : U32)
  +exact = offset_exact(query,unit,row,group,row_word,position_word,row_exact,position_exact,inside,group_inside)
  +bias_value = Tree.read(F32,bias(query),row_word)
  +o1 = (offset + 1 : U32)
  +o2 = (offset + 2 : U32)
  +o3 = (offset + 3 : U32)
  +o4 = (offset + 4 : U32)
  +o5 = (offset + 5 : U32)
  +o6 = (offset + 6 : U32)
  +o7 = (offset + 7 : U32)
  +v0 = (Tile.lane0(values) + bias_value : F32)
  +v1 = (Tile.lane1(values) + bias_value : F32)
  +v2 = (Tile.lane2(values) + bias_value : F32)
  +v3 = (Tile.lane3(values) + bias_value : F32)
  +v4 = (Tile.lane4(values) + bias_value : F32)
  +v5 = (Tile.lane5(values) + bias_value : F32)
  +v6 = (Tile.lane6(values) + bias_value : F32)
  +v7 = (Tile.lane7(values) + bias_value : F32)
  later(query,offset,v0,[Writes.Put{o1,v1},Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
    hit(query,o1,v1,[Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
      lane_hit(query,unit,row,group,1n,o1,lane_offset(query,unit,row,group,1n,offset,exact,inside,group_inside,{==}),same)))

def lane2_hit(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+values: Tile.Tile8,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,2n)) : Nat}) ->
  Covered(query,GemmWrites.row_writes(True{},row_word,position_word,columns(query),values,bias(query))):
  +offset = (row_word * columns(query) + position_word : U32)
  +exact = offset_exact(query,unit,row,group,row_word,position_word,row_exact,position_exact,inside,group_inside)
  +bias_value = Tree.read(F32,bias(query),row_word)
  +o1 = (offset + 1 : U32)
  +o2 = (offset + 2 : U32)
  +o3 = (offset + 3 : U32)
  +o4 = (offset + 4 : U32)
  +o5 = (offset + 5 : U32)
  +o6 = (offset + 6 : U32)
  +o7 = (offset + 7 : U32)
  +v0 = (Tile.lane0(values) + bias_value : F32)
  +v1 = (Tile.lane1(values) + bias_value : F32)
  +v2 = (Tile.lane2(values) + bias_value : F32)
  +v3 = (Tile.lane3(values) + bias_value : F32)
  +v4 = (Tile.lane4(values) + bias_value : F32)
  +v5 = (Tile.lane5(values) + bias_value : F32)
  +v6 = (Tile.lane6(values) + bias_value : F32)
  +v7 = (Tile.lane7(values) + bias_value : F32)
  later(query,offset,v0,[Writes.Put{o1,v1},Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
    later(query,o1,v1,[Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
      hit(query,o2,v2,[Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
        lane_hit(query,unit,row,group,2n,o2,lane_offset(query,unit,row,group,2n,offset,exact,inside,group_inside,{==}),same))))

def lane3_hit(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+values: Tile.Tile8,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,3n)) : Nat}) ->
  Covered(query,GemmWrites.row_writes(True{},row_word,position_word,columns(query),values,bias(query))):
  +offset = (row_word * columns(query) + position_word : U32)
  +exact = offset_exact(query,unit,row,group,row_word,position_word,row_exact,position_exact,inside,group_inside)
  +bias_value = Tree.read(F32,bias(query),row_word)
  +o1 = (offset + 1 : U32)
  +o2 = (offset + 2 : U32)
  +o3 = (offset + 3 : U32)
  +o4 = (offset + 4 : U32)
  +o5 = (offset + 5 : U32)
  +o6 = (offset + 6 : U32)
  +o7 = (offset + 7 : U32)
  +v0 = (Tile.lane0(values) + bias_value : F32)
  +v1 = (Tile.lane1(values) + bias_value : F32)
  +v2 = (Tile.lane2(values) + bias_value : F32)
  +v3 = (Tile.lane3(values) + bias_value : F32)
  +v4 = (Tile.lane4(values) + bias_value : F32)
  +v5 = (Tile.lane5(values) + bias_value : F32)
  +v6 = (Tile.lane6(values) + bias_value : F32)
  +v7 = (Tile.lane7(values) + bias_value : F32)
  later(query,offset,v0,[Writes.Put{o1,v1},Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
    later(query,o1,v1,[Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
      later(query,o2,v2,[Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
        hit(query,o3,v3,[Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
          lane_hit(query,unit,row,group,3n,o3,lane_offset(query,unit,row,group,3n,offset,exact,inside,group_inside,{==}),same)))))

def lane4_hit(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+values: Tile.Tile8,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,4n)) : Nat}) ->
  Covered(query,GemmWrites.row_writes(True{},row_word,position_word,columns(query),values,bias(query))):
  +offset = (row_word * columns(query) + position_word : U32)
  +exact = offset_exact(query,unit,row,group,row_word,position_word,row_exact,position_exact,inside,group_inside)
  +bias_value = Tree.read(F32,bias(query),row_word)
  +o1 = (offset + 1 : U32)
  +o2 = (offset + 2 : U32)
  +o3 = (offset + 3 : U32)
  +o4 = (offset + 4 : U32)
  +o5 = (offset + 5 : U32)
  +o6 = (offset + 6 : U32)
  +o7 = (offset + 7 : U32)
  +v0 = (Tile.lane0(values) + bias_value : F32)
  +v1 = (Tile.lane1(values) + bias_value : F32)
  +v2 = (Tile.lane2(values) + bias_value : F32)
  +v3 = (Tile.lane3(values) + bias_value : F32)
  +v4 = (Tile.lane4(values) + bias_value : F32)
  +v5 = (Tile.lane5(values) + bias_value : F32)
  +v6 = (Tile.lane6(values) + bias_value : F32)
  +v7 = (Tile.lane7(values) + bias_value : F32)
  later(query,offset,v0,[Writes.Put{o1,v1},Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
    later(query,o1,v1,[Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
      later(query,o2,v2,[Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
        later(query,o3,v3,[Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
          hit(query,o4,v4,[Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
            lane_hit(query,unit,row,group,4n,o4,lane_offset(query,unit,row,group,4n,offset,exact,inside,group_inside,{==}),same))))))

def lane5_hit(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+values: Tile.Tile8,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,5n)) : Nat}) ->
  Covered(query,GemmWrites.row_writes(True{},row_word,position_word,columns(query),values,bias(query))):
  +offset = (row_word * columns(query) + position_word : U32)
  +exact = offset_exact(query,unit,row,group,row_word,position_word,row_exact,position_exact,inside,group_inside)
  +bias_value = Tree.read(F32,bias(query),row_word)
  +o1 = (offset + 1 : U32)
  +o2 = (offset + 2 : U32)
  +o3 = (offset + 3 : U32)
  +o4 = (offset + 4 : U32)
  +o5 = (offset + 5 : U32)
  +o6 = (offset + 6 : U32)
  +o7 = (offset + 7 : U32)
  +v0 = (Tile.lane0(values) + bias_value : F32)
  +v1 = (Tile.lane1(values) + bias_value : F32)
  +v2 = (Tile.lane2(values) + bias_value : F32)
  +v3 = (Tile.lane3(values) + bias_value : F32)
  +v4 = (Tile.lane4(values) + bias_value : F32)
  +v5 = (Tile.lane5(values) + bias_value : F32)
  +v6 = (Tile.lane6(values) + bias_value : F32)
  +v7 = (Tile.lane7(values) + bias_value : F32)
  later(query,offset,v0,[Writes.Put{o1,v1},Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
    later(query,o1,v1,[Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
      later(query,o2,v2,[Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
        later(query,o3,v3,[Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
          later(query,o4,v4,[Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
            hit(query,o5,v5,[Writes.Put{o6,v6},Writes.Put{o7,v7}],
              lane_hit(query,unit,row,group,5n,o5,lane_offset(query,unit,row,group,5n,offset,exact,inside,group_inside,{==}),same)))))))

def lane6_hit(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+values: Tile.Tile8,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,6n)) : Nat}) ->
  Covered(query,GemmWrites.row_writes(True{},row_word,position_word,columns(query),values,bias(query))):
  +offset = (row_word * columns(query) + position_word : U32)
  +exact = offset_exact(query,unit,row,group,row_word,position_word,row_exact,position_exact,inside,group_inside)
  +bias_value = Tree.read(F32,bias(query),row_word)
  +o1 = (offset + 1 : U32)
  +o2 = (offset + 2 : U32)
  +o3 = (offset + 3 : U32)
  +o4 = (offset + 4 : U32)
  +o5 = (offset + 5 : U32)
  +o6 = (offset + 6 : U32)
  +o7 = (offset + 7 : U32)
  +v0 = (Tile.lane0(values) + bias_value : F32)
  +v1 = (Tile.lane1(values) + bias_value : F32)
  +v2 = (Tile.lane2(values) + bias_value : F32)
  +v3 = (Tile.lane3(values) + bias_value : F32)
  +v4 = (Tile.lane4(values) + bias_value : F32)
  +v5 = (Tile.lane5(values) + bias_value : F32)
  +v6 = (Tile.lane6(values) + bias_value : F32)
  +v7 = (Tile.lane7(values) + bias_value : F32)
  later(query,offset,v0,[Writes.Put{o1,v1},Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
    later(query,o1,v1,[Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
      later(query,o2,v2,[Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
        later(query,o3,v3,[Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
          later(query,o4,v4,[Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
            later(query,o5,v5,[Writes.Put{o6,v6},Writes.Put{o7,v7}],
              hit(query,o6,v6,[Writes.Put{o7,v7}],
                lane_hit(query,unit,row,group,6n,o6,lane_offset(query,unit,row,group,6n,offset,exact,inside,group_inside,{==}),same))))))))

def lane7_hit(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+values: Tile.Tile8,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,7n)) : Nat}) ->
  Covered(query,GemmWrites.row_writes(True{},row_word,position_word,columns(query),values,bias(query))):
  +offset = (row_word * columns(query) + position_word : U32)
  +exact = offset_exact(query,unit,row,group,row_word,position_word,row_exact,position_exact,inside,group_inside)
  +bias_value = Tree.read(F32,bias(query),row_word)
  +o1 = (offset + 1 : U32)
  +o2 = (offset + 2 : U32)
  +o3 = (offset + 3 : U32)
  +o4 = (offset + 4 : U32)
  +o5 = (offset + 5 : U32)
  +o6 = (offset + 6 : U32)
  +o7 = (offset + 7 : U32)
  +v0 = (Tile.lane0(values) + bias_value : F32)
  +v1 = (Tile.lane1(values) + bias_value : F32)
  +v2 = (Tile.lane2(values) + bias_value : F32)
  +v3 = (Tile.lane3(values) + bias_value : F32)
  +v4 = (Tile.lane4(values) + bias_value : F32)
  +v5 = (Tile.lane5(values) + bias_value : F32)
  +v6 = (Tile.lane6(values) + bias_value : F32)
  +v7 = (Tile.lane7(values) + bias_value : F32)
  later(query,offset,v0,[Writes.Put{o1,v1},Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
    later(query,o1,v1,[Writes.Put{o2,v2},Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
      later(query,o2,v2,[Writes.Put{o3,v3},Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
        later(query,o3,v3,[Writes.Put{o4,v4},Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
          later(query,o4,v4,[Writes.Put{o5,v5},Writes.Put{o6,v6},Writes.Put{o7,v7}],
            later(query,o5,v5,[Writes.Put{o6,v6},Writes.Put{o7,v7}],
              later(query,o6,v6,[Writes.Put{o7,v7}],
                hit(query,o7,v7,[],
                  lane_hit(query,unit,row,group,7n,o7,lane_offset(query,unit,row,group,7n,offset,exact,inside,group_inside,{==}),same)))))))))

def lane_hit_at(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+values: Tile.Tile8,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +lane: Nat,+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat}) ->
  Covered(query,GemmWrites.row_writes(True{},row_word,position_word,columns(query),values,bias(query))):
  match lane:
    case 0n: lane0_hit(query,unit,row,group,row_word,position_word,values,row_exact,position_exact,inside,group_inside,same)
    case 1n+a:
      match a:
        case 0n: lane1_hit(query,unit,row,group,row_word,position_word,values,row_exact,position_exact,inside,group_inside,same)
        case 1n+b:
          match b:
            case 0n: lane2_hit(query,unit,row,group,row_word,position_word,values,row_exact,position_exact,inside,group_inside,same)
            case 1n+c:
              match c:
                case 0n: lane3_hit(query,unit,row,group,row_word,position_word,values,row_exact,position_exact,inside,group_inside,same)
                case 1n+d:
                  match d:
                    case 0n: lane4_hit(query,unit,row,group,row_word,position_word,values,row_exact,position_exact,inside,group_inside,same)
                    case 1n+e:
                      match e:
                        case 0n: lane5_hit(query,unit,row,group,row_word,position_word,values,row_exact,position_exact,inside,group_inside,same)
                        case 1n+f:
                          match f:
                            case 0n: lane6_hit(query,unit,row,group,row_word,position_word,values,row_exact,position_exact,inside,group_inside,same)
                            case 1n+g:
                              match g:
                                case 0n: lane7_hit(query,unit,row,group,row_word,position_word,values,row_exact,position_exact,inside,group_inside,same)
                                case 1n+h: Empty.absurd(Covered(query,GemmWrites.row_writes(True{},row_word,position_word,columns(query),values,bias(query))),WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(h,0n),True{},Equal.sym(Bool,Nat.is_lt(h,0n),False{},Division.lt_zero(h)),lane_small)))

def row_hit(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+values: Tile.Tile8,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},+position_exact: {U32.to_nat(position_word) == Nat.mul(group,8n) : Nat},
  +inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +valid: Bool,+valid_exact: {U32.is_lt(row_word,rows(query)) == valid : Bool},
  +lane: Nat,+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat}) ->
  Covered(query,GemmWrites.row_writes(valid,row_word,position_word,columns(query),values,bias(query))):
  match valid:
    case False{}:
      Empty.absurd(Covered(query,[]),WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_lt(row_word,rows(query)),True{},
        Equal.sym(Bool,U32.is_lt(row_word,rows(query)),False{},valid_exact),row_accepted(query,unit,row,row_word,row_exact,inside))))
    case True{}:
      lane_hit_at(query,unit,row,group,row_word,position_word,values,row_exact,position_exact,inside,group_inside,lane,lane_small,same)

def row0_cover(+query: Query,+unit: Nat,+group: Nat,+lane: Nat,+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,0n),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool},
  +unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +inside: {Nat.is_lt(row_number(unit,0n),U32.to_nat(rows(query))) == True{} : Bool}) ->
  Covered(query,job_writes(query,unit,group)):
  Query{+rows,+columns,+inner,+groups,+limit,+input,+weights,+bias,+channel,+column,+limit_fits,+columns_exact,+rows_room,+area_room,+channel_inside,+column_inside} = query
  +query = {Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} : Query}
  +word = unit_word(query,unit,group)
  +group_value = group_word(query,unit,group)
  +word_exact = unit_exact(query,unit,group,unit_inside,group_inside)
  +position = (group_value * 8 : U32)
  +position_exact = position_exact(query,group,group_value,group_exact(query,unit,group,unit_inside,group_inside),group_inside)
  +accumulators = unit_accumulators(query,unit,group)
  +index = (word * 4 : U32)
  +i1 = (index + 1 : U32)
  +i2 = (index + 2 : U32)
  +i3 = (index + 3 : U32)
  +a0 = Rows.first(~Tile.Tile8,accumulators)
  +a1 = Rows.second(~Tile.Tile8,accumulators)
  +a2 = Rows.third(~Tile.Tile8,accumulators)
  +a3 = Rows.fourth(~Tile.Tile8,accumulators)
  +w0 = GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,a0,bias)
  +w1 = GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,a1,bias)
  +w2 = GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,a2,bias)
  +w3 = GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,a3,bias)
  +l01 = List.append(&2,Writes.Assignment<F32>,w0,w1)
  +l02 = List.append(&2,Writes.Assignment<F32>,l01,w2)
  +hit_row = row_hit(query,unit,0n,group,index,position,a0,row_zero_exact(query,unit,word,word_exact,unit_rows),position_exact,inside,group_inside,
    U32.is_lt(index,rows),{==},lane,lane_small,same)
  appended_left(query,l02,w3,appended_left(query,l01,w2,appended_left(query,w0,w1,hit_row)))

def row1_cover(+query: Query,+unit: Nat,+group: Nat,+lane: Nat,+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,1n),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool},
  +unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +inside: {Nat.is_lt(row_number(unit,1n),U32.to_nat(rows(query))) == True{} : Bool}) ->
  Covered(query,job_writes(query,unit,group)):
  Query{+rows,+columns,+inner,+groups,+limit,+input,+weights,+bias,+channel,+column,+limit_fits,+columns_exact,+rows_room,+area_room,+channel_inside,+column_inside} = query
  +query = {Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} : Query}
  +word = unit_word(query,unit,group)
  +group_value = group_word(query,unit,group)
  +word_exact = unit_exact(query,unit,group,unit_inside,group_inside)
  +position = (group_value * 8 : U32)
  +position_exact = position_exact(query,group,group_value,group_exact(query,unit,group,unit_inside,group_inside),group_inside)
  +accumulators = unit_accumulators(query,unit,group)
  +index = (word * 4 : U32)
  +i1 = (index + 1 : U32)
  +i2 = (index + 2 : U32)
  +i3 = (index + 3 : U32)
  +a0 = Rows.first(~Tile.Tile8,accumulators)
  +a1 = Rows.second(~Tile.Tile8,accumulators)
  +a2 = Rows.third(~Tile.Tile8,accumulators)
  +a3 = Rows.fourth(~Tile.Tile8,accumulators)
  +w0 = GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,a0,bias)
  +w1 = GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,a1,bias)
  +w2 = GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,a2,bias)
  +w3 = GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,a3,bias)
  +l01 = List.append(&2,Writes.Assignment<F32>,w0,w1)
  +l02 = List.append(&2,Writes.Assignment<F32>,l01,w2)
  +hit_row = row_hit(query,unit,1n,group,i1,position,a1,row_offset_exact(query,unit,word,1n,word_exact,unit_rows,{==}),position_exact,inside,group_inside,
    U32.is_lt(i1,rows),{==},lane,lane_small,same)
  appended_left(query,l02,w3,appended_left(query,l01,w2,appended_right(query,w0,w1,hit_row)))

def row2_cover(+query: Query,+unit: Nat,+group: Nat,+lane: Nat,+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,2n),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool},
  +unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +inside: {Nat.is_lt(row_number(unit,2n),U32.to_nat(rows(query))) == True{} : Bool}) ->
  Covered(query,job_writes(query,unit,group)):
  Query{+rows,+columns,+inner,+groups,+limit,+input,+weights,+bias,+channel,+column,+limit_fits,+columns_exact,+rows_room,+area_room,+channel_inside,+column_inside} = query
  +query = {Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} : Query}
  +word = unit_word(query,unit,group)
  +group_value = group_word(query,unit,group)
  +word_exact = unit_exact(query,unit,group,unit_inside,group_inside)
  +position = (group_value * 8 : U32)
  +position_exact = position_exact(query,group,group_value,group_exact(query,unit,group,unit_inside,group_inside),group_inside)
  +accumulators = unit_accumulators(query,unit,group)
  +index = (word * 4 : U32)
  +i1 = (index + 1 : U32)
  +i2 = (index + 2 : U32)
  +i3 = (index + 3 : U32)
  +a0 = Rows.first(~Tile.Tile8,accumulators)
  +a1 = Rows.second(~Tile.Tile8,accumulators)
  +a2 = Rows.third(~Tile.Tile8,accumulators)
  +a3 = Rows.fourth(~Tile.Tile8,accumulators)
  +w0 = GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,a0,bias)
  +w1 = GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,a1,bias)
  +w2 = GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,a2,bias)
  +w3 = GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,a3,bias)
  +l01 = List.append(&2,Writes.Assignment<F32>,w0,w1)
  +l02 = List.append(&2,Writes.Assignment<F32>,l01,w2)
  +hit_row = row_hit(query,unit,2n,group,i2,position,a2,row_offset_exact(query,unit,word,2n,word_exact,unit_rows,{==}),position_exact,inside,group_inside,
    U32.is_lt(i2,rows),{==},lane,lane_small,same)
  appended_left(query,l02,w3,appended_right(query,l01,w2,hit_row))

def row3_cover(+query: Query,+unit: Nat,+group: Nat,+lane: Nat,+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,3n),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool},
  +unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +inside: {Nat.is_lt(row_number(unit,3n),U32.to_nat(rows(query))) == True{} : Bool}) ->
  Covered(query,job_writes(query,unit,group)):
  Query{+rows,+columns,+inner,+groups,+limit,+input,+weights,+bias,+channel,+column,+limit_fits,+columns_exact,+rows_room,+area_room,+channel_inside,+column_inside} = query
  +query = {Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} : Query}
  +word = unit_word(query,unit,group)
  +group_value = group_word(query,unit,group)
  +word_exact = unit_exact(query,unit,group,unit_inside,group_inside)
  +position = (group_value * 8 : U32)
  +position_exact = position_exact(query,group,group_value,group_exact(query,unit,group,unit_inside,group_inside),group_inside)
  +accumulators = unit_accumulators(query,unit,group)
  +index = (word * 4 : U32)
  +i1 = (index + 1 : U32)
  +i2 = (index + 2 : U32)
  +i3 = (index + 3 : U32)
  +a0 = Rows.first(~Tile.Tile8,accumulators)
  +a1 = Rows.second(~Tile.Tile8,accumulators)
  +a2 = Rows.third(~Tile.Tile8,accumulators)
  +a3 = Rows.fourth(~Tile.Tile8,accumulators)
  +w0 = GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,a0,bias)
  +w1 = GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,a1,bias)
  +w2 = GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,a2,bias)
  +w3 = GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,a3,bias)
  +l01 = List.append(&2,Writes.Assignment<F32>,w0,w1)
  +l02 = List.append(&2,Writes.Assignment<F32>,l01,w2)
  +hit_row = row_hit(query,unit,3n,group,i3,position,a3,row_offset_exact(query,unit,word,3n,word_exact,unit_rows,{==}),position_exact,inside,group_inside,
    U32.is_lt(i3,rows),{==},lane,lane_small,same)
  appended_right(query,l02,w3,hit_row)

def job_cover(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,
  +row_small: {Nat.is_lt(row,4n) == True{} : Bool},+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool},
  +unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
  +inside: {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool}) ->
  Covered(query,job_writes(query,unit,group)):
  match row:
    case 0n: row0_cover(query,unit,group,lane,lane_small,same,unit_rows,unit_inside,group_inside,inside)
    case 1n+a:
      match a:
        case 0n: row1_cover(query,unit,group,lane,lane_small,same,unit_rows,unit_inside,group_inside,inside)
        case 1n+b:
          match b:
            case 0n: row2_cover(query,unit,group,lane,lane_small,same,unit_rows,unit_inside,group_inside,inside)
            case 1n+c:
              match c:
                case 0n: row3_cover(query,unit,group,lane,lane_small,same,unit_rows,unit_inside,group_inside,inside)
                case 1n+d:
                  Empty.absurd(Covered(query,job_writes(query,unit,group)),WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(d,0n),True{},Equal.sym(Bool,Nat.is_lt(d,0n),False{},Division.lt_zero(d)),row_small)))

def step_after(+first: Nat,+target: Nat,after: {Nat.is_le(first,target) == True{} : Bool},differ: {Nat.is_eq(first,target) == False{} : Bool}) ->
  {Nat.is_le(1n+first,target) == True{} : Bool}:
  match first target:
    case 0n 0n: Empty.absurd({Nat.is_le(1n,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},differ)))
    case 0n 1n+ +rest: Order.zero_le(rest)
    case 1n+ +rest 0n: Empty.absurd({Nat.is_le(2n+rest,0n) == True{} : Bool},WordEncoding.false_true(after))
    case 1n+ +rest 1n+ +other: step_after(rest,other,after,differ)

def all_cover_step(+query: Query,+rest: Nat,+first: Nat,+target: Nat,+equal: Bool,
  exact: {Nat.is_eq(first,target) == equal : Bool},
  +job: Covered(query,GemmWrites.job_writes(Gemm.job(layout(query),U32.from_nat(target)),input(query),weights(query),bias(query))),
  +after: {Nat.is_le(first,target) == True{} : Bool},
  later_cover: {Nat.is_le(1n+first,target) == True{} : Bool} -> Covered(query,GemmWrites.all_writes(rest,U32.from_nat(1n+first),layout(query),input(query),weights(query),bias(query)))) ->
  Covered(query,List.append(&2,Writes.Assignment<F32>,GemmWrites.job_writes(Gemm.job(layout(query),U32.from_nat(first)),input(query),weights(query),bias(query)),
    GemmWrites.all_writes(rest,U32.from_nat(1n+first),layout(query),input(query),weights(query),bias(query)))):
  match equal:
    case True{}:
      +here = GemmWrites.job_writes(Gemm.job(layout(query),U32.from_nat(first)),input(query),weights(query),bias(query))
      +later = GemmWrites.all_writes(rest,U32.from_nat(1n+first),layout(query),input(query),weights(query),bias(query))
      appended_left(query,here,later,
        %Equal.sym(Nat,first,target,Order.equal_of_is_eq(first,target,exact)) :
          Covered(query,GemmWrites.job_writes(Gemm.job(layout(query),U32.from_nat(_)),input(query),weights(query),bias(query)))
        job)
    case False{}:
      +here = GemmWrites.job_writes(Gemm.job(layout(query),U32.from_nat(first)),input(query),weights(query),bias(query))
      +later = GemmWrites.all_writes(rest,U32.from_nat(1n+first),layout(query),input(query),weights(query),bias(query))
      appended_right(query,here,later,later_cover(step_after(first,target,after,exact)))

def all_cover(+query: Query,+count: Nat,+first: Nat,+target: Nat,
  +job: Covered(query,GemmWrites.job_writes(Gemm.job(layout(query),U32.from_nat(target)),input(query),weights(query),bias(query))),
  +after: {Nat.is_le(first,target) == True{} : Bool},+before: {Nat.is_lt(target,Nat.add(first,count)) == True{} : Bool}) ->
  Covered(query,GemmWrites.all_writes(count,U32.from_nat(first),layout(query),input(query),weights(query),bias(query))):
  match count:
    case 0n:
      +below = {%Addition.zero_right(first) : {Nat.is_lt(target,_) == True{} : Bool}
        before : {Nat.is_lt(target,first) == True{} : Bool}}
      Empty.absurd(Covered(query,[]),WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(first,first),True{},
        Equal.sym(Bool,Nat.is_eq(first,first),False{},Order.less_not_equal(first,first,Order.le_lt_trans(first,target,first,after,below))),
        Order.equal_reflexive(first))))
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        Covered(query,List.append(&2,Writes.Assignment<F32>,GemmWrites.job_writes(Gemm.job(layout(query),U32.from_nat(first)),input(query),weights(query),bias(query)),
          GemmWrites.all_writes(rest,_,layout(query),input(query),weights(query),bias(query))))
      all_cover_step(query,rest,first,target,Nat.is_eq(first,target),{==},job,after,
        next => all_cover(query,rest,1n+first,target,job,next,
          %Addition.successor_right(first,rest) : {Nat.is_lt(target,_) == True{} : Bool}
          before))

def layout_exact(+query: Query) -> {layout(query) == Gemm.Layout{rows(query),columns(query),inner(query)} : Gemm.Layout}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  {==}

def evidence_readback(+writes: List<&2,Writes.Assignment<F32>>,+output: Tree.Tree<F32>,+depth: Nat,+balanced: Shape.Balanced(F32,depth,output),
  +depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+target: Nat,+value: F32,
  +small: {Nat.is_lt(target,U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  evidence: WriteReadback.WriteEvidence<target,value,Tree.capacity(F32,output),writes>,
  covered: {Coverage.contains(target,WriteReadback.natural_write_addresses(writes)) == True{} : Bool}) ->
  {Tree.read(F32,Writes.pure(F32,writes,output),U32.from_nat(target)) == value : F32}:
  WriteReadback.WriteEvidence{bounds,values} = evidence
  +exact = Bounds.roundtrip(target,Tree.capacity(F32,output),StorageCapacity.small(depth,output,balanced,depth_ok),
    Bounds.strict_le(target,U32.to_nat(Tree.capacity(F32,output)),small))
  WriteReadback.readback_from_matching_covered_writes(writes,output,depth,U32.from_nat(target),value,balanced,depth_ok,bounds,
    U32Division.nat_lt_native(U32.from_nat(target),Tree.capacity(F32,output),
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(target)),target,exact) : {Nat.is_lt(_,U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool}
      small),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(target)),target,exact) : Coverage.MatchingWritesHaveValue(_,value,WriteReadback.natural_write_addresses(writes))
    values,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(target)),target,exact) : {Coverage.contains(_,WriteReadback.natural_write_addresses(writes)) == True{} : Bool}
    covered)

def split_address(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,
  +channel_split: {channel(query) == row_number(unit,row) : Nat},+column_split: {column(query) == lane_number(group,lane) : Nat}) ->
  {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat}:
  %Equal.sym(Nat,channel(query),row_number(unit,row),channel_split) :
    {Nat.add(Nat.mul(_,U32.to_nat(columns(query))),column(query)) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat}
  %Equal.sym(Nat,column(query),lane_number(group,lane),column_split) :
    {Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),_) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat}
  {==}

def split_row_inside(+query: Query,+unit: Nat,+row: Nat,+channel_split: {channel(query) == row_number(unit,row) : Nat}) ->
  {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool}:
  %channel_split : {Nat.is_lt(_,U32.to_nat(rows(query))) == True{} : Bool}
  channel_inside(query)

def split_unit_inside(+query: Query,+unit: Nat,+row: Nat,+channel_split: {channel(query) == row_number(unit,row) : Nat}) ->
  {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool}:
  Order.le_lt_trans(Nat.mul(unit,4n),channel(query),U32.to_nat(rows(query)),
    %Equal.sym(Nat,channel(query),row_number(unit,row),channel_split) : {Nat.is_le(Nat.mul(unit,4n),_) == True{} : Bool}
    Order.le_add(Nat.mul(unit,4n),row),channel_inside(query))

def split_group_inside(+query: Query,+group: Nat,+lane: Nat,+column_split: {column(query) == lane_number(group,lane) : Nat}) ->
  {Nat.is_lt(group,groups(query)) == True{} : Bool}:
  RowMajor.row_inside(group,groups(query),8n,lane,
    %column_split : {Nat.is_lt(_,Nat.mul(groups(query),8n)) == True{} : Bool}
    %columns_exact(query) : {Nat.is_lt(column(query),_) == True{} : Bool}
    column_inside(query))

def count_exact(+query: Query) -> {GemmWrites.job_count(rows(query),columns(query)) == Nat.mul(groups(query),units(query)) : Nat}:
  +group_count = Gemm.column_groups(columns(query))
  +unit_count = Gemm.row_units(rows(query))
  %groups_word_exact(query) : {GemmWrites.job_count(rows(query),columns(query)) == Nat.mul(_,units(query)) : Nat}
  %units_word_exact(query) : {GemmWrites.job_count(rows(query),columns(query)) == Nat.mul(U32.to_nat(group_count),_) : Nat}
  Arithmetic.multiply(group_count,unit_count,limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(group_count),groups(query),groups_word_exact(query)) : {Nat.is_le(Nat.mul(_,U32.to_nat(unit_count)),U32.to_nat(limit(query))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(unit_count),units(query),units_word_exact(query)) : {Nat.is_le(Nat.mul(groups(query),_),U32.to_nat(limit(query))) == True{} : Bool}
    jobs_within(query))

# The complete schedule: any in-range output reads back its job value.
def run_readback(+query: Query,+output: Tree.Tree<F32>,+depth: Nat,+balanced: Shape.Balanced(F32,depth,output),
  +depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool}) ->
  {Tree.read(F32,GemmLaws.run(rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query)),U32.from_nat(address(query))) ==
    wanted(query) : F32}:
  +rows_nat = U32.to_nat(rows(query))
  +padded = Nat.add(rows_nat,3n)
  +count = units(query)
  +jobs = Nat.mul(groups(query),count)
  +channel_small = Order.transitive(channel(query),rows_nat,U32.to_nat(limit(query)),Bounds.strict_le(channel(query),rows_nat,channel_inside(query)),
    Order.transitive(rows_nat,padded,U32.to_nat(limit(query)),Order.le_add(rows_nat,3n),rows_room(query)))
  +column_small = Order.transitive(column(query),U32.to_nat(columns(query)),U32.to_nat(limit(query)),
    Bounds.strict_le(column(query),U32.to_nat(columns(query)),column_inside(query)),columns_within(query))
  +unit = Nat.div(channel(query),4n)
  +row = Nat.mod(channel(query),4n)
  +group = Nat.div(column(query),8n)
  +lane = Nat.mod(column(query),8n)
  +channel_split = decomposed(channel(query),4n,limit(query),limit_fits(query),channel_small,{==})
  +column_split = decomposed(column(query),8n,limit(query),limit_fits(query),column_small,{==})
  +unit_rows = split_unit_inside(query,unit,row,channel_split)
  +unit_inside = unit_counted(query,unit,unit_rows)
  +group_inside = split_group_inside(query,group,lane,column_split)
  +target = job_index(unit,group,count)
  +target_inside = RowMajor.radix(group,groups(query),unit,count,group_inside,unit_inside)
  +covered = all_cover(query,jobs,0n,target,
    job_cover(query,unit,row,group,lane,
      remainder_small(channel(query),4n,limit(query),limit_fits(query),channel_small,{==}),
      remainder_small(column(query),8n,limit(query),limit_fits(query),column_small,{==}),
      split_address(query,unit,row,group,lane,channel_split,column_split),unit_rows,unit_inside,group_inside,split_row_inside(query,unit,row,channel_split)),
    Order.zero_le(target),target_inside)
  +evidence = all_evidence(query,Tree.capacity(F32,output),jobs,0n,output_room,Order.reflexive(jobs))
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run(rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query)),
    Writes.pure(F32,GemmWrites.all_writes(GemmWrites.job_count(rows(query),columns(query)),0,Gemm.Layout{rows(query),columns(query),inner(query)},input(query),weights(query),bias(query)),output),
    GemmWrites.run_writes(rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query))) :
    {Tree.read(F32,_,U32.from_nat(address(query))) == wanted(query) : F32}
  %Equal.sym(Nat,GemmWrites.job_count(rows(query),columns(query)),jobs,count_exact(query)) :
    {Tree.read(F32,Writes.pure(F32,GemmWrites.all_writes(_,0,Gemm.Layout{rows(query),columns(query),inner(query)},input(query),weights(query),bias(query)),output),
      U32.from_nat(address(query))) == wanted(query) : F32}
  %layout_exact(query) :
    {Tree.read(F32,Writes.pure(F32,GemmWrites.all_writes(jobs,0,_,input(query),weights(query),bias(query)),output),
      U32.from_nat(address(query))) == wanted(query) : F32}
  evidence_readback(GemmWrites.all_writes(jobs,U32.from_nat(0n),layout(query),input(query),weights(query),bias(query)),output,depth,balanced,depth_ok,
    address(query),wanted(query),
    Bounds.nat_trans(address(query),Nat.mul(rows_nat,U32.to_nat(columns(query))),U32.to_nat(Tree.capacity(F32,output)),
      RowMajor.radix(channel(query),rows_nat,column(query),U32.to_nat(columns(query)),channel_inside(query),column_inside(query)),output_room),
    evidence,covered)
