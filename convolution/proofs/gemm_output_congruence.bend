# Two GEMM runs that differ only in their input tiles store identical write
# lists when every valid row and lane of every job has equal accumulators.
# Link names the caller's relation between the query and the other tiles.
import Base
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_rows.bend as Rows
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/kernel_gemm_writes.bend as GemmWrites
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ./gemm_output_readback.bend as GemmReadback

def other(+query: GemmReadback.Query,+tiles: Tree.Tree<Tile.Tile8>,unit: Nat,group: Nat) -> Rows.Rows<Tile.Tile8>:
  GemmReadback.accumulators(Gemm.job(GemmReadback.layout(query),U32.from_nat(GemmReadback.job_index(unit,group,GemmReadback.units(query)))),tiles,GemmReadback.weights(query))

def other_job_writes(+query: GemmReadback.Query,+tiles: Tree.Tree<Tile.Tile8>,+unit: Nat,+group: Nat) -> List<&2,Writes.Assignment<F32>>:
  GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(GemmReadback.job_index(unit,group,GemmReadback.units(query)))),tiles,GemmReadback.weights(query),GemmReadback.bias(query))

def Agreement(~Link: GemmReadback.Query -> Tree.Tree<Tile.Tile8> -> Data) -> Type:
  @+query: GemmReadback.Query -> @+tiles: Tree.Tree<Tile.Tile8> -> @+unit: Nat -> @+group: Nat -> @+row: Nat -> @+lane: Nat ->
  @+link: Link(query,tiles) ->
  @+unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(GemmReadback.rows(query))) == True{} : Bool} ->
  @+group_inside: {Nat.is_lt(group,GemmReadback.groups(query)) == True{} : Bool} ->
  @+row_small: {Nat.is_lt(row,4n) == True{} : Bool} -> @+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool} ->
  @+row_inside: {Nat.is_lt(GemmReadback.row_number(unit,row),U32.to_nat(GemmReadback.rows(query))) == True{} : Bool} ->
  {GemmReadback.select_lane(lane,GemmReadback.select_row(row,GemmReadback.unit_accumulators(query,unit,group))) == GemmReadback.select_lane(lane,GemmReadback.select_row(row,other(query,tiles,unit,group))) : F32}

def row_congruence(~Link: GemmReadback.Query -> Tree.Tree<Tile.Tile8> -> Data,~agree: Agreement(~Link),+query: GemmReadback.Query,+tiles: Tree.Tree<Tile.Tile8>,+link: Link(query,tiles),+unit: Nat,+row: Nat,+group: Nat,+row_word: U32,+position_word: U32,+valid: Bool,
  +valid_exact: {U32.is_lt(row_word,GemmReadback.rows(query)) == valid : Bool},+row_exact: {U32.to_nat(row_word) == GemmReadback.row_number(unit,row) : Nat},
  +row_small: {Nat.is_lt(row,4n) == True{} : Bool},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(GemmReadback.rows(query))) == True{} : Bool},+group_inside: {Nat.is_lt(group,GemmReadback.groups(query)) == True{} : Bool}) ->
  {GemmWrites.row_writes(valid,row_word,position_word,GemmReadback.columns(query),GemmReadback.select_row(row,GemmReadback.unit_accumulators(query,unit,group)),GemmReadback.bias(query)) ==
    GemmWrites.row_writes(valid,row_word,position_word,GemmReadback.columns(query),GemmReadback.select_row(row,other(query,tiles,unit,group)),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}:
  match valid:
    case False{}: {==}
    case True{}:
      +inside = GemmReadback.row_inside(query,unit,row,row_word,row_exact,valid_exact)
      +left = GemmReadback.select_row(row,GemmReadback.unit_accumulators(query,unit,group))
      +right = GemmReadback.select_row(row,other(query,tiles,unit,group))
      %TileLaws.tiles_equal(left,right,
        agree(query,tiles,unit,group,row,0n,link,unit_rows,group_inside,row_small,{==},inside),
        agree(query,tiles,unit,group,row,1n,link,unit_rows,group_inside,row_small,{==},inside),
        agree(query,tiles,unit,group,row,2n,link,unit_rows,group_inside,row_small,{==},inside),
        agree(query,tiles,unit,group,row,3n,link,unit_rows,group_inside,row_small,{==},inside),
        agree(query,tiles,unit,group,row,4n,link,unit_rows,group_inside,row_small,{==},inside),
        agree(query,tiles,unit,group,row,5n,link,unit_rows,group_inside,row_small,{==},inside),
        agree(query,tiles,unit,group,row,6n,link,unit_rows,group_inside,row_small,{==},inside),
        agree(query,tiles,unit,group,row,7n,link,unit_rows,group_inside,row_small,{==},inside)) :
        {GemmWrites.row_writes(True{},row_word,position_word,GemmReadback.columns(query),left,GemmReadback.bias(query)) ==
          GemmWrites.row_writes(True{},row_word,position_word,GemmReadback.columns(query),_,GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}
      {==}

def job_congruence(~Link: GemmReadback.Query -> Tree.Tree<Tile.Tile8> -> Data,~agree: Agreement(~Link),+query: GemmReadback.Query,+tiles: Tree.Tree<Tile.Tile8>,+link: Link(query,tiles),+unit: Nat,+group: Nat,
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(GemmReadback.rows(query))) == True{} : Bool},
  +unit_inside: {Nat.is_lt(unit,GemmReadback.units(query)) == True{} : Bool},+group_inside: {Nat.is_lt(group,GemmReadback.groups(query)) == True{} : Bool}) ->
  {GemmReadback.job_writes(query,unit,group) == other_job_writes(query,tiles,unit,group) : List<&2,Writes.Assignment<F32>>}:
  GemmReadback.Query{+rows,+columns,+inner,+groups,+limit,+input,+weights,+bias,+channel,+column,+limit_fits,+columns_exact,+rows_room,+area_room,+channel_inside,+column_inside} = query
  +query = {GemmReadback.Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} : GemmReadback.Query}
  +word = GemmReadback.unit_word(query,unit,group)
  +group_value = GemmReadback.group_word(query,unit,group)
  +word_exact = GemmReadback.unit_exact(query,unit,group,unit_inside,group_inside)
  +position = (0 + group_value * 8 : U32)
  +accumulators = GemmReadback.unit_accumulators(query,unit,group)
  +others = other(query,tiles,unit,group)
  +index = (word * 4 : U32)
  +i1 = (index + 1 : U32)
  +i2 = (index + 2 : U32)
  +i3 = (index + 3 : U32)
  +e0 = row_congruence(~Link,~agree,query,tiles,link,unit,0n,group,index,position,U32.is_lt(index,rows),{==},GemmReadback.row_zero_exact(query,unit,word,word_exact,unit_rows),{==},unit_rows,group_inside)
  +e1 = row_congruence(~Link,~agree,query,tiles,link,unit,1n,group,i1,position,U32.is_lt(i1,rows),{==},GemmReadback.row_offset_exact(query,unit,word,1n,word_exact,unit_rows,{==}),{==},unit_rows,group_inside)
  +e2 = row_congruence(~Link,~agree,query,tiles,link,unit,2n,group,i2,position,U32.is_lt(i2,rows),{==},GemmReadback.row_offset_exact(query,unit,word,2n,word_exact,unit_rows,{==}),{==},unit_rows,group_inside)
  +e3 = row_congruence(~Link,~agree,query,tiles,link,unit,3n,group,i3,position,U32.is_lt(i3,rows),{==},GemmReadback.row_offset_exact(query,unit,word,3n,word_exact,unit_rows,{==}),{==},unit_rows,group_inside)
  %e0 : {List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,Rows.first(~Tile.Tile8,accumulators),bias),GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,Rows.second(~Tile.Tile8,accumulators),bias)),GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,Rows.third(~Tile.Tile8,accumulators),bias)),GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,Rows.fourth(~Tile.Tile8,accumulators),bias)) ==
    List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,_,GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,Rows.second(~Tile.Tile8,others),bias)),GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,Rows.third(~Tile.Tile8,others),bias)),GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,Rows.fourth(~Tile.Tile8,others),bias)) : List<&2,Writes.Assignment<F32>>}
  %e1 : {List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,Rows.first(~Tile.Tile8,accumulators),bias),GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,Rows.second(~Tile.Tile8,accumulators),bias)),GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,Rows.third(~Tile.Tile8,accumulators),bias)),GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,Rows.fourth(~Tile.Tile8,accumulators),bias)) ==
    List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,Rows.first(~Tile.Tile8,accumulators),bias),_),GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,Rows.third(~Tile.Tile8,others),bias)),GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,Rows.fourth(~Tile.Tile8,others),bias)) : List<&2,Writes.Assignment<F32>>}
  %e2 : {List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,Rows.first(~Tile.Tile8,accumulators),bias),GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,Rows.second(~Tile.Tile8,accumulators),bias)),GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,Rows.third(~Tile.Tile8,accumulators),bias)),GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,Rows.fourth(~Tile.Tile8,accumulators),bias)) ==
    List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,Rows.first(~Tile.Tile8,accumulators),bias),GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,Rows.second(~Tile.Tile8,accumulators),bias)),_),GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,Rows.fourth(~Tile.Tile8,others),bias)) : List<&2,Writes.Assignment<F32>>}
  %e3 : {List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,Rows.first(~Tile.Tile8,accumulators),bias),GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,Rows.second(~Tile.Tile8,accumulators),bias)),GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,Rows.third(~Tile.Tile8,accumulators),bias)),GemmWrites.row_writes(U32.is_lt(i3,rows),i3,position,columns,Rows.fourth(~Tile.Tile8,accumulators),bias)) ==
    List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,GemmWrites.row_writes(U32.is_lt(index,rows),index,position,columns,Rows.first(~Tile.Tile8,accumulators),bias),GemmWrites.row_writes(U32.is_lt(i1,rows),i1,position,columns,Rows.second(~Tile.Tile8,accumulators),bias)),GemmWrites.row_writes(U32.is_lt(i2,rows),i2,position,columns,Rows.third(~Tile.Tile8,accumulators),bias)),_) : List<&2,Writes.Assignment<F32>>}
  {==}

def job_congruence_at(~Link: GemmReadback.Query -> Tree.Tree<Tile.Tile8> -> Data,~agree: Agreement(~Link),+query: GemmReadback.Query,+tiles: Tree.Tree<Tile.Tile8>,+link: Link(query,tiles),+index: Nat,
  +inside: {Nat.is_lt(index,Nat.mul(GemmReadback.groups(query),GemmReadback.units(query))) == True{} : Bool}) ->
  {GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(index)),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) ==
    GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(index)),tiles,GemmReadback.weights(query),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}:
  +count = GemmReadback.units(query)
  +small = Order.transitive(index,Nat.mul(GemmReadback.groups(query),count),U32.to_nat(GemmReadback.limit(query)),Bounds.strict_le(index,Nat.mul(GemmReadback.groups(query),count),inside),GemmReadback.jobs_within(query))
  +unit = Nat.mod(index,count)
  +group = Nat.div(index,count)
  +identity = GemmReadback.decomposed(index,count,GemmReadback.limit(query),GemmReadback.limit_fits(query),small,GemmReadback.units_positive(query))
  +unit_inside = GemmReadback.remainder_small(index,count,GemmReadback.limit(query),GemmReadback.limit_fits(query),small,GemmReadback.units_positive(query))
  +group_inside = RowMajor.row_inside(group,GemmReadback.groups(query),count,unit,
    %identity : {Nat.is_lt(_,Nat.mul(GemmReadback.groups(query),count)) == True{} : Bool}
    inside)
  %Equal.sym(Nat,index,GemmReadback.job_index(unit,group,count),identity) :
    {GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(_)),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) ==
      GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(index)),tiles,GemmReadback.weights(query),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}
  %Equal.sym(Nat,index,GemmReadback.job_index(unit,group,count),identity) :
    {GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(GemmReadback.job_index(unit,group,count))),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) ==
      GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(_)),tiles,GemmReadback.weights(query),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}
  job_congruence(~Link,~agree,query,tiles,link,unit,group,GemmReadback.unit_bound(query,unit,unit_inside),unit_inside,group_inside)

def all_congruence(~Link: GemmReadback.Query -> Tree.Tree<Tile.Tile8> -> Data,~agree: Agreement(~Link),+query: GemmReadback.Query,+tiles: Tree.Tree<Tile.Tile8>,+link: Link(query,tiles),+count: Nat,+first: Nat,
  +room: {Nat.is_le(Nat.add(first,count),Nat.mul(GemmReadback.groups(query),GemmReadback.units(query))) == True{} : Bool}) ->
  {GemmWrites.all_writes(count,U32.from_nat(first),GemmReadback.layout(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) ==
    GemmWrites.all_writes(count,U32.from_nat(first),GemmReadback.layout(query),tiles,GemmReadback.weights(query),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +here = GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(first)),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query))
      +there = GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(first)),tiles,GemmReadback.weights(query),GemmReadback.bias(query))
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        {List.append(&2,Writes.Assignment<F32>,here,GemmWrites.all_writes(rest,_,GemmReadback.layout(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query))) ==
          List.append(&2,Writes.Assignment<F32>,there,GemmWrites.all_writes(rest,(U32.from_nat(first) + 1 : U32),GemmReadback.layout(query),tiles,GemmReadback.weights(query),GemmReadback.bias(query))) : List<&2,Writes.Assignment<F32>>}
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        {List.append(&2,Writes.Assignment<F32>,here,GemmWrites.all_writes(rest,U32.from_nat(1n+first),GemmReadback.layout(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query))) ==
          List.append(&2,Writes.Assignment<F32>,there,GemmWrites.all_writes(rest,_,GemmReadback.layout(query),tiles,GemmReadback.weights(query),GemmReadback.bias(query))) : List<&2,Writes.Assignment<F32>>}
      %job_congruence_at(~Link,~agree,query,tiles,link,first,Order.room_index(first,rest,Nat.mul(GemmReadback.groups(query),GemmReadback.units(query)),room)) :
        {List.append(&2,Writes.Assignment<F32>,here,GemmWrites.all_writes(rest,U32.from_nat(1n+first),GemmReadback.layout(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query))) ==
          List.append(&2,Writes.Assignment<F32>,_,GemmWrites.all_writes(rest,U32.from_nat(1n+first),GemmReadback.layout(query),tiles,GemmReadback.weights(query),GemmReadback.bias(query))) : List<&2,Writes.Assignment<F32>>}
      %all_congruence(~Link,~agree,query,tiles,link,rest,1n+first,Order.advance_room(first,rest,Nat.mul(GemmReadback.groups(query),GemmReadback.units(query)),room)) :
        {List.append(&2,Writes.Assignment<F32>,here,GemmWrites.all_writes(rest,U32.from_nat(1n+first),GemmReadback.layout(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query))) ==
          List.append(&2,Writes.Assignment<F32>,here,_) : List<&2,Writes.Assignment<F32>>}
      {==}

# Whole GEMM runs over the two tile trees store the same output tree.
def run_congruence(~Link: GemmReadback.Query -> Tree.Tree<Tile.Tile8> -> Data,~agree: Agreement(~Link),+query: GemmReadback.Query,+tiles: Tree.Tree<Tile.Tile8>,
  +link: Link(query,tiles),+output: Tree.Tree<F32>) ->
  {GemmLaws.run(GemmReadback.rows(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query)) ==
    GemmLaws.run(GemmReadback.rows(query),tiles,GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query)) : Tree.Tree<F32>}:
  +count = Nat.mul(GemmReadback.groups(query),GemmReadback.units(query))
  +layout = {Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),0,GemmReadback.columns(query)} : Gemm.Layout}
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run(GemmReadback.rows(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query)),
    Writes.pure(F32,GemmWrites.all_writes(GemmWrites.job_count(GemmReadback.rows(query),GemmReadback.columns(query)),0,layout,GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)),output),
    GemmWrites.run_writes(GemmReadback.rows(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query))) :
    {_ == GemmLaws.run(GemmReadback.rows(query),tiles,GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query)) : Tree.Tree<F32>}
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run(GemmReadback.rows(query),tiles,GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query)),
    Writes.pure(F32,GemmWrites.all_writes(GemmWrites.job_count(GemmReadback.rows(query),GemmReadback.columns(query)),0,layout,tiles,GemmReadback.weights(query),GemmReadback.bias(query)),output),
    GemmWrites.run_writes(GemmReadback.rows(query),tiles,GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query))) :
    {Writes.pure(F32,GemmWrites.all_writes(GemmWrites.job_count(GemmReadback.rows(query),GemmReadback.columns(query)),0,layout,GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)),output) == _ : Tree.Tree<F32>}
  %Equal.sym(Nat,GemmWrites.job_count(GemmReadback.rows(query),GemmReadback.columns(query)),count,GemmReadback.count_exact(query)) :
    {Writes.pure(F32,GemmWrites.all_writes(_,0,layout,GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)),output) ==
      Writes.pure(F32,GemmWrites.all_writes(_,0,layout,tiles,GemmReadback.weights(query),GemmReadback.bias(query)),output) : Tree.Tree<F32>}
  %GemmReadback.layout_exact(query) :
    {Writes.pure(F32,GemmWrites.all_writes(count,0,_,GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)),output) ==
      Writes.pure(F32,GemmWrites.all_writes(count,0,_,tiles,GemmReadback.weights(query),GemmReadback.bias(query)),output) : Tree.Tree<F32>}
  Equal.cong(List<&2,Writes.Assignment<F32>>,Tree.Tree<F32>,writes => Writes.pure(F32,writes,output),
    GemmWrites.all_writes(count,U32.from_nat(0n),GemmReadback.layout(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)),
    GemmWrites.all_writes(count,U32.from_nat(0n),GemmReadback.layout(query),tiles,GemmReadback.weights(query),GemmReadback.bias(query)),
    all_congruence(~Link,~agree,query,tiles,link,count,0n,Order.reflexive(count)))
