# A query selects its producing block in the actual indexed traversal. The
# allocation/guard adapter supplies the extent evidence; later blocks preserve
# the value through the common contiguous-write proof.
import Base
import ../../lib/kernel_weights.bend as Weights
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_sequence.bend as Sequence
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/kernel_weights_readback.bend as WeightsReadback
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/indexed_output_readback.bend as OutputReadback
import ../../lib/traversal_array.bend as Traversal
import ./weight_reorder_contents.bend as ReorderContents
import ./tile_array_projection.bend as ArrayProjection
import ./storage_capacity.bend as StorageCapacity
import ./supported_padded_coordinates.bend as Padded
import ./dense_array_initialization.bend as DenseArrayInitialization
import ../../lib/proofs/nat_interval.bend as Interval

def block_count(rows: U32) -> Nat:
  U32.to_nat(((rows + 7) / 8 : U32))

type Query is Data:
  Query{rows: U32,inner: U32,depth: Nat,input: Tree.Tree<F32>,lane: Sequence.Index(8n),target: Nat,tap: Nat,
    depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(block_count(rows),U32.to_nat(inner)),U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
    target_inside: {Nat.is_lt(target,block_count(rows)) == True{} : Bool},
    tap_inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool}}

def layout(query: Query) -> Weights.Layout:
  Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  Weights.Layout{rows,inner}

def inner(query: Query) -> U32:
  Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  inner

def depth(query: Query) -> Nat:
  Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  depth

def input(query: Query) -> Tree.Tree<F32>:
  Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  input

def lane(query: Query) -> Sequence.Index(8n):
  Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  lane

def target(query: Query) -> Nat:
  Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  target

def tap(query: Query) -> Nat:
  Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  tap

def limit(query: Query) -> Nat:
  Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  block_count(rows)

def target_inside(+query: Query) -> {Nat.is_lt(target(query),limit(query)) == True{} : Bool}:
  Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  target_inside

def capacity(query: Query) -> U32:
  U32.shln(1,depth(query))

def address(+query: Query) -> U32:
  U32.from_nat(Nat.add(Nat.mul(target(query),U32.to_nat(inner(query))),tap(query)))

def context(query: Query,+index: U32) -> Weights.Context:
  Query{rows,+inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  Weights.Context{rows,inner,(index * 8 : U32),(index * inner : U32)}

def source(owners: WeightsLaws.Owners) -> Tree.Tree<F32>:
  WeightsLaws.Owners{input,output} = owners
  input

def wanted(+query: Query) -> F32:
  Tile.element_at(lane(query),WeightsReadback.sample(input(query),context(query,U32.from_nat(target(query))),U32.from_nat(tap(query))))

def read(+query: Query,owners: WeightsLaws.Owners) -> F32:
  Tile.element_at(lane(query),Tree.read(Tile.Tile8,WeightsLaws.output(owners),address(query)))

def step(index: U32,query: Query,owners: WeightsLaws.Owners) -> WeightsLaws.Owners:
  WeightsLaws.block(index,layout(query),owners)

def Frame(+query: Query,+owners: WeightsLaws.Owners) -> Data:
  Shape.Both<{source(owners) == input(query) : Tree.Tree<F32>},Shape.Balanced(F32,depth(query),ArrayProjection.project(WeightsLaws.output(owners),lane(query)))>

def frame_step(+query: Query,+index: U32,+owners: WeightsLaws.Owners,frame: Frame(query,owners)) -> Frame(query,step(index,query,owners)):
  Query{+rows,+inner,+depth,+original,+lane,+target,+tap,+depth_ok,+room,+target_inside,+tap_inside} = query
  WeightsLaws.Owners{+input,+output} = owners
  Shape.Both{same,balanced} = frame
  +query = {Query{rows,inner,depth,original,lane,target,tap,depth_ok,room,target_inside,tap_inside} : Query}
  %Equal.sym(WeightsLaws.Owners,WeightsLaws.block(index,Weights.Layout{rows,inner},WeightsLaws.Owners{input,output}),
    WeightsLaws.Owners{input,ReorderContents.fill(U32.to_nat(inner),0,context(query,index),input,output)},ReorderContents.block_matches_fill(index,rows,inner,input,output)) : Frame(query,_)
  Shape.Both{same,ReorderContents.balanced(U32.to_nat(inner),0,context(query,index),input,output,lane,depth,balanced)}

def inner_positive(+query: Query) -> {Nat.is_lt(0n,U32.to_nat(inner(query))) == True{} : Bool}:
  Query{rows,inner,depth,input,lane,target,+tap,depth_ok,room,target_inside,tap_inside} = query
  Order.le_lt_trans(0n,tap,U32.to_nat(inner),Order.zero_le(tap),tap_inside)

def limit_positive(+query: Query) -> {Nat.is_lt(0n,limit(query)) == True{} : Bool}:
  Query{rows,inner,depth,input,lane,+target,tap,depth_ok,room,target_inside,tap_inside} = query
  Order.le_lt_trans(0n,target,block_count(rows),Order.zero_le(target),target_inside)

def capacity_fits(+query: Query) -> {U32.is_le(capacity(query),2147483648) == True{} : Bool}:
  Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  StorageCapacity.depth_bound(depth,depth_ok)

def elements_room(+query: Query) -> {Nat.is_le(Nat.mul(limit(query),U32.to_nat(inner(query))),U32.to_nat(capacity(query))) == True{} : Bool}:
  Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} = query
  room

def index_bound(+query: Query,+index: Nat,inside: {Nat.is_lt(index,limit(query)) == True{} : Bool}) ->
  {Nat.is_le(index,U32.to_nat(capacity(query))) == True{} : Bool}:
  Order.transitive(index,limit(query),U32.to_nat(capacity(query)),Bounds.strict_le(index,limit(query),inside),
    Order.transitive(limit(query),Nat.mul(limit(query),U32.to_nat(inner(query))),U32.to_nat(capacity(query)),
      RowMajor.product_covers(limit(query),U32.to_nat(inner(query)),inner_positive(query)),elements_room(query)))

def inner_bound(+query: Query) -> {Nat.is_le(U32.to_nat(inner(query)),U32.to_nat(capacity(query))) == True{} : Bool}:
  Order.transitive(U32.to_nat(inner(query)),Nat.mul(limit(query),U32.to_nat(inner(query))),U32.to_nat(capacity(query)),
    Padded.product_has_right(limit(query),U32.to_nat(inner(query)),limit_positive(query)),elements_room(query))

def block_end_room(+query: Query,+index: Nat,inside: {Nat.is_lt(index,limit(query)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(index,U32.to_nat(inner(query))),U32.to_nat(inner(query))),U32.to_nat(capacity(query))) == True{} : Bool}:
  %Order.commutative(U32.to_nat(inner(query)),Nat.mul(index,U32.to_nat(inner(query)))) :
    {Nat.is_le(_,U32.to_nat(capacity(query))) == True{} : Bool}
  Order.transitive(Nat.mul(1n+index,U32.to_nat(inner(query))),Nat.mul(limit(query),U32.to_nat(inner(query))),U32.to_nat(capacity(query)),
    RowMajor.mul_le(1n+index,limit(query),U32.to_nat(inner(query)),Order.strict_follow(index,limit(query),inside)),elements_room(query))

def base_exact(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,limit(query)) == True{} : Bool}) ->
  {U32.to_nat(Weights.destination(context(query,U32.from_nat(index)),0)) == Nat.mul(index,U32.to_nat(inner(query))) : Nat}:
  Query{+rows,+inner,+depth,+input,+lane,+target,+tap,+depth_ok,+room,+target_inside,+tap_inside} = query
  +query = {Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} : Query}
  %Equal.sym(U32,((U32.from_nat(index) * inner : U32) + 0 : U32),(U32.from_nat(index) * inner : U32),Arithmetic.add_zero((U32.from_nat(index) * inner : U32))) :
    {U32.to_nat(_) == Nat.mul(index,U32.to_nat(inner)) : Nat}
  %Arithmetic.native_roundtrip(inner) : {U32.to_nat((U32.from_nat(index) * _ : U32)) == Nat.mul(index,U32.to_nat(inner)) : Nat}
  Encoded.product(index,U32.to_nat(inner),capacity(query),capacity_fits(query),index_bound(query,index,inside),inner_bound(query),
    Order.transitive(Nat.mul(index,U32.to_nat(inner)),Nat.mul(limit(query),U32.to_nat(inner)),U32.to_nat(capacity(query)),
      RowMajor.mul_le(index,limit(query),U32.to_nat(inner),Bounds.strict_le(index,limit(query),inside)),elements_room(query)))

def address_exact(+query: Query) ->
  {U32.to_nat(address(query)) == Nat.add(Nat.mul(target(query),U32.to_nat(inner(query))),tap(query)) : Nat}:
  Query{+rows,+inner,+depth,+input,+lane,+target,+tap,+depth_ok,+room,+target_inside,+tap_inside} = query
  +query = {Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} : Query}
  Bounds.roundtrip(Nat.add(Nat.mul(target,U32.to_nat(inner)),tap),capacity(query),capacity_fits(query),
    Bounds.strict_le(Nat.add(Nat.mul(target,U32.to_nat(inner)),tap),U32.to_nat(capacity(query)),
      Bounds.nat_trans(Nat.add(Nat.mul(target,U32.to_nat(inner)),tap),Nat.mul(block_count(rows),U32.to_nat(inner)),U32.to_nat(capacity(query)),
        RowMajor.radix(target,block_count(rows),tap,U32.to_nat(inner),target_inside,tap_inside),room)))

def projected_capacity(+query: Query,+output: Tree.Tree<Tile.Tile8>,balanced: Shape.Balanced(F32,depth(query),ArrayProjection.project(output,lane(query)))) ->
  {Tree.capacity(F32,ArrayProjection.project(output,lane(query))) == capacity(query) : U32}:
  Shape.capacity(F32,depth(query),ArrayProjection.project(output,lane(query)),balanced)

def block_room(+query: Query,+index: Nat,+output: Tree.Tree<Tile.Tile8>,balanced: Shape.Balanced(F32,depth(query),ArrayProjection.project(output,lane(query))),
  +inside: {Nat.is_lt(index,limit(query)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Weights.destination(context(query,U32.from_nat(index)),0)),U32.to_nat(inner(query))),
    U32.to_nat(Tree.capacity(F32,ArrayProjection.project(output,lane(query))))) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(F32,ArrayProjection.project(output,lane(query))),capacity(query),projected_capacity(query,output,balanced)) :
    {Nat.is_le(Nat.add(U32.to_nat(Weights.destination(context(query,U32.from_nat(index)),0)),U32.to_nat(inner(query))),U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Weights.destination(context(query,U32.from_nat(index)),0)),Nat.mul(index,U32.to_nat(inner(query))),base_exact(query,index,inside)) :
    {Nat.is_le(Nat.add(_,U32.to_nat(inner(query))),U32.to_nat(capacity(query))) == True{} : Bool}
  block_end_room(query,index,inside)

def address_inside(+query: Query,+output: Tree.Tree<Tile.Tile8>,balanced: Shape.Balanced(F32,depth(query),ArrayProjection.project(output,lane(query)))) ->
  {U32.is_lt(address(query),Tree.capacity(F32,ArrayProjection.project(output,lane(query)))) == True{} : Bool}:
  Query{+rows,+inner,+depth,+input,+lane,+target,+tap,+depth_ok,+room,+target_inside,+tap_inside} = query
  +query = {Query{rows,inner,depth,input,lane,target,tap,depth_ok,room,target_inside,tap_inside} : Query}
  %Equal.sym(U32,Tree.capacity(F32,ArrayProjection.project(output,lane)),capacity(query),projected_capacity(query,output,balanced)) :
    {U32.is_lt(address(query),_) == True{} : Bool}
  U32Division.nat_lt_native(address(query),capacity(query),
    %Equal.sym(Nat,U32.to_nat(address(query)),Nat.add(Nat.mul(target,U32.to_nat(inner)),tap),address_exact(query)) :
      {Nat.is_lt(_,U32.to_nat(capacity(query))) == True{} : Bool}
    Bounds.nat_trans(Nat.add(Nat.mul(target,U32.to_nat(inner)),tap),Nat.mul(block_count(rows),U32.to_nat(inner)),U32.to_nat(capacity(query)),
      RowMajor.radix(target,block_count(rows),tap,U32.to_nat(inner),target_inside,tap_inside),room))

def at_target(+query: Query,+owners: WeightsLaws.Owners,frame: Frame(query,owners)) ->
  {read(query,step(U32.from_nat(target(query)),query,owners)) == wanted(query) : F32}:
  Query{+rows,+inner,+depth,+original,+lane,+target,+tap,+depth_ok,+room,+target_inside,+tap_inside} = query
  WeightsLaws.Owners{+input,+output} = owners
  Shape.Both{same,+balanced} = frame
  +query = {Query{rows,inner,depth,original,lane,target,tap,depth_ok,room,target_inside,tap_inside} : Query}
  +context = context(query,U32.from_nat(target))
  %Equal.sym(WeightsLaws.Owners,WeightsLaws.block(U32.from_nat(target),Weights.Layout{rows,inner},WeightsLaws.Owners{input,output}),
    WeightsLaws.Owners{input,ReorderContents.fill(U32.to_nat(inner),0,context,input,output)},ReorderContents.block_matches_fill(U32.from_nat(target),rows,inner,input,output)) :
    {read(query,_) == wanted(query) : F32}
  %Equal.sym(Tree.Tree<F32>,input,original,same) :
    {Tile.element_at(lane,Tree.read(Tile.Tile8,ReorderContents.fill(U32.to_nat(inner),0,context,_,output),address(query))) == wanted(query) : F32}
  %Arithmetic.add_zero_left(U32.from_nat(tap)) :
    {Tile.element_at(lane,Tree.read(Tile.Tile8,ReorderContents.fill(U32.to_nat(inner),0,context,original,output),address(query))) ==
      Tile.element_at(lane,WeightsReadback.sample(original,context,_)) : F32}
  ReorderContents.readback(U32.to_nat(inner),0,context,original,output,lane,depth,address(query),tap,balanced,depth_ok,
    %Equal.sym(Nat,U32.to_nat(Weights.destination(context,0)),Nat.mul(target,U32.to_nat(inner)),base_exact(query,target,target_inside)) :
      {U32.to_nat(address(query)) == Nat.add(_,tap) : Nat}
    address_exact(query),block_room(query,target,output,balanced,target_inside),tap_inside)

def later(+query: Query,+index: Nat,+owners: WeightsLaws.Owners,frame: Frame(query,owners),
  after: {Nat.is_lt(target(query),index) == True{} : Bool},inside: {Nat.is_lt(index,limit(query)) == True{} : Bool}) ->
  {read(query,step(U32.from_nat(index),query,owners)) == read(query,owners) : F32}:
  Query{+rows,+inner,+depth,+original,+lane,+target,+tap,+depth_ok,+room,+target_inside,+tap_inside} = query
  WeightsLaws.Owners{+input,+output} = owners
  Shape.Both{same,+balanced} = frame
  +inside = inside
  +query = {Query{rows,inner,depth,original,lane,target,tap,depth_ok,room,target_inside,tap_inside} : Query}
  +context = context(query,U32.from_nat(index))
  %Equal.sym(WeightsLaws.Owners,WeightsLaws.block(U32.from_nat(index),Weights.Layout{rows,inner},WeightsLaws.Owners{input,output}),
    WeightsLaws.Owners{input,ReorderContents.fill(U32.to_nat(inner),0,context,input,output)},ReorderContents.block_matches_fill(U32.from_nat(index),rows,inner,input,output)) :
    {read(query,_) == read(query,WeightsLaws.Owners{input,output}) : F32}
  ReorderContents.preserves_outside(U32.to_nat(inner),0,context,input,output,lane,depth,address(query),balanced,depth_ok,
    block_room(query,index,output,balanced,inside),address_inside(query,output,balanced),
    Interval.Before{
      %Equal.sym(Nat,U32.to_nat(address(query)),Nat.add(Nat.mul(target,U32.to_nat(inner)),tap),address_exact(query)) :
        {Nat.is_lt(_,U32.to_nat(Weights.destination(context,0))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(Weights.destination(context,0)),Nat.mul(index,U32.to_nat(inner)),base_exact(query,index,inside)) :
        {Nat.is_lt(Nat.add(Nat.mul(target,U32.to_nat(inner)),tap),_) == True{} : Bool}
      RowMajor.radix(target,index,tap,U32.to_nat(inner),after,tap_inside)})

def same_traversal(+count: Nat,+first: U32,+query: Query,+owners: WeightsLaws.Owners) ->
  {Traversal.indexed_fold(~WeightsLaws.Owners,~Weights.Layout,~WeightsLaws.block,count,first,layout(query),owners) ==
    Traversal.indexed_fold(~WeightsLaws.Owners,~Query,~step,count,first,query,owners) : WeightsLaws.Owners}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      same_traversal(rest,(first + 1 : U32),query,step(first,query,owners))

def complete(+query: Query,+output: Tree.Tree<Tile.Tile8>,balanced: Shape.Balanced(F32,depth(query),ArrayProjection.project(output,lane(query)))) ->
  {read(query,Traversal.indexed_fold(~WeightsLaws.Owners,~Weights.Layout,~WeightsLaws.block,limit(query),0,layout(query),WeightsLaws.Owners{input(query),output})) == wanted(query) : F32}:
  +owners = {WeightsLaws.Owners{input(query),output} : WeightsLaws.Owners}
  %Equal.sym(WeightsLaws.Owners,Traversal.indexed_fold(~WeightsLaws.Owners,~Weights.Layout,~WeightsLaws.block,limit(query),0,layout(query),owners),
    Traversal.indexed_fold(~WeightsLaws.Owners,~Query,~step,limit(query),0,query,owners),same_traversal(limit(query),0,query,owners)) :
    {read(query,_) == wanted(query) : F32}
  OutputReadback.complete_readback(~WeightsLaws.Owners,~Query,~F32,~limit,~target,~(query => wanted(query)),~(query => owners => Frame(query,owners)),
    ~(query => owners => read(query,owners)),~step,~frame_step,~at_target,~later,
    query,owners,Shape.Both{{==},balanced},target_inside(query))
