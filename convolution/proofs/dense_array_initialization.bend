import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/nat_interval.bend as Interval
import Base
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./dense_read_refinement.bend as DenseRead
import ./scalar_write_refinement.bend as ScalarWriteRefinement
import ./storage_capacity.bend as StorageCapacity
import ../../lib/proofs/u32_successor.bend as U32Successor
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_division.bend as U32Division
import ./tree_indexing.bend as TreeIndex
import ../specification.bend as ConvSpec
import ./conditional_write_coverage.bend as Coverage
import ./native_array_model.bend as NativeArrayModel

def fill(xs: List<&2,F32>,+i: Nat,ys: List<&2,F32>) -> List<&2,F32>:
  match xs:
    case []:
      ys
    case v <> tail:
      fill(tail,1n+i,ScalarWriteRefinement.set(i,v,ys))

def fill_preserves_outside(+values: List<&2,F32>,+start: Nat,+query: Nat,+original: List<&2,F32>,
  +query_inside: {Nat.is_lt(query,ConvSpec.length(original)) == True{} : Bool},+outside: Interval.Outside<start,ConvSpec.length(values),query>) ->
  {ConvSpec.at(query,fill(values,start,original)) == ConvSpec.at(query,original) : F32}:
  match values:
    case Nil{}: {==}
    case +head <> +tail:
      +updated = ScalarWriteRefinement.set(start,head,original)
      %Equal.sym(F32,ConvSpec.at(query,fill(tail,1n+start,updated)),ConvSpec.at(query,updated),
        fill_preserves_outside(tail,1n+start,query,updated,Coverage.set_room(query,start,head,original,query_inside),Interval.outside_tail(start,ConvSpec.length(tail),query,outside))) :
        {_ == ConvSpec.at(query,original) : F32}
      %Equal.sym(F32,ConvSpec.at(query,updated),Bool.pick(F32,Nat.is_eq(query,start),head,ConvSpec.at(query,original)),Coverage.at_set(query,start,head,original,query_inside)) :
        {_ == ConvSpec.at(query,original) : F32}
      %Equal.sym(Bool,Nat.is_eq(query,start),False{},Interval.outside_head(start,ConvSpec.length(tail),query,outside)) :
        {Bool.pick(F32,_,head,ConvSpec.at(query,original)) == ConvSpec.at(query,original) : F32}
      {==}

def tail_room(+depth: Nat,+tree: Tree.Tree<F32>,+i: U32,+v: F32,+tail: Nat,+balanced: Shape.Balanced(F32,depth,
    tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(i),1n+tail),
      U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}) -> {Nat.is_le(Nat.add(U32.to_nat((i + 1 : U32)),tail),
    U32.to_nat(Tree.capacity(F32,Tree.write(F32,tree,i,v)))) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(F32,Tree.write(F32,tree,i,v)),Tree.capacity(F32,tree),StorageCapacity.after_write(depth,tree,
      i,v,balanced)) : {Nat.is_le(Nat.add(U32.to_nat((i + 1 : U32)),tail),U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((i + 1 : U32)),1n+U32.to_nat(i),U32Successor.successor(i)(Bounds.uint_trans(i,Tree.capacity(F32,tree),
        2147483648,U32Division.nat_lt_native(i,Tree.capacity(F32,tree),Order.room_index(U32.to_nat(i),tail,
          U32.to_nat(Tree.capacity(F32,tree)),room)),StorageCapacity.small(depth,tree,balanced,limit)))) : {Nat.is_le(Nat.add(_,tail),
      U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}
  Order.advance_room(U32.to_nat(i),tail,U32.to_nat(Tree.capacity(F32,tree)),room)

law filled:
  for +xs: List<&2,F32>
  for +i: U32
  for +tree: Tree.Tree<F32>
  for +depth: Nat
  for +balanced: Shape.Balanced(F32,depth,tree)
  for +limit: {Nat.is_le(depth,30n) == True{} : Bool}
  for room: {Nat.is_le(Nat.add(U32.to_nat(i),ConvSpec.length(xs)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}
  {TreeIndex.flatten(Tree.write_sequence(~F32,xs,i,tree)) == fill(xs,U32.to_nat(i),
      TreeIndex.flatten(tree)) : List<&2,F32>}
def filled(xs,i,tree,depth,balanced,limit,room):
  match xs:
    case []:
      {==}
    case +v <> +tail:
      +room = room
      +inside = U32Division.nat_lt_native(i,Tree.capacity(F32,tree),Order.room_index(U32.to_nat(i),ConvSpec.length(tail),
          U32.to_nat(Tree.capacity(F32,tree)),room))
      +small = Bounds.uint_trans(i,Tree.capacity(F32,tree),2147483648,inside,StorageCapacity.small(depth,tree,balanced,limit))
      %U32Successor.successor(i)(small) : {TreeIndex.flatten(Tree.write_sequence(~F32,tail,(i + 1 : U32),
              Tree.write(F32,tree,i,v))) == fill(tail,_,ScalarWriteRefinement.set(U32.to_nat(i),v,
            TreeIndex.flatten(tree))) : List<&2,F32>}
      %ScalarWriteRefinement.dense(depth,tree,i,v,balanced,limit,inside) : {TreeIndex.flatten(Tree.write_sequence(~F32,tail,(i + 1 : U32),Tree.write(F32,tree,i,
              v))) == fill(tail,U32.to_nat((i + 1 : U32)),_) : List<&2,F32>}
      filled(tail,(i + 1 : U32),Tree.write(F32,tree,i,v),depth,Shape.write(F32,depth,tree,i,v,balanced),limit,tail_room(depth,tree,i,
          v,ConvSpec.length(tail),balanced,limit,room))

def take(n: Nat,xs: List<&2,F32>) -> List<&2,F32>:
  match n xs:
    case 0n xs:
      []
    case 1n+p Nil{}:
      []
    case 1n+p h <> tail:
      h <> take(p,tail)

def offset(+xs: List<&2,F32>,+i: Nat,+h: F32,+ys: List<&2,F32>) -> {fill(xs,1n+i,h <> ys) == h <> fill(xs,i,ys) : List<&2,F32>}:
  match xs:
    case []:
      {==}
    case +v <> +tail:
      offset(tail,1n+i,h,ScalarWriteRefinement.set(i,v,ys))

law prefix:
  for +xs: List<&2,F32>
  for +ys: List<&2,F32>
  for room: {Nat.is_le(ConvSpec.length(xs),ConvSpec.length(ys)) == True{} : Bool}
  {take(ConvSpec.length(xs),fill(xs,0n,ys)) == xs : List<&2,F32>}
def prefix(xs,ys,room):
  match xs:
    case []:
      {==}
    case +v <> +tail:
      match ys:
        case []:
          Empty.absurd({take(1n+ConvSpec.length(tail),fill(v <> tail,0n,[])) == v <> tail : List<&2,F32>},WordEncoding.false_true(room))
        case old <> +rest:
          %Equal.sym(List<&2,F32>,fill(tail,1n,v <> rest),v <> fill(tail,0n,rest),offset(tail,0n,v,rest)) : {take(1n+ConvSpec.length(tail),
              _) == v <> tail : List<&2,F32>}
          %Equal.sym(List<&2,F32>,take(ConvSpec.length(tail),fill(tail,0n,rest)),tail,prefix(tail,rest,room)) : {v <> _ == v <> tail : List<&2,F32>}
          {==}

def balanced(+xs: List<&2,F32>,+i: U32,+tree: Tree.Tree<F32>,+depth: Nat,proof: Shape.Balanced(F32,depth,
    tree)) -> Shape.Balanced(F32,depth,Tree.write_sequence(~F32,xs,i,tree)):
  match xs:
    case []:
      proof
    case +v <> +tail:
      balanced(tail,(i + 1 : U32),Tree.write(F32,tree,i,v),depth,Shape.write(F32,depth,tree,i,v,proof))

def capacity(+xs: List<&2,F32>,+i: U32,+tree: Tree.Tree<F32>,+depth: Nat,+proof: Shape.Balanced(F32,depth,
    tree)) -> {Tree.capacity(F32,Tree.write_sequence(~F32,xs,i,tree)) == Tree.capacity(F32,tree) : U32}:
  Equal.trans(U32,Tree.capacity(F32,Tree.write_sequence(~F32,xs,i,tree)),U32.shln(1,depth),Tree.capacity(F32,tree),
    StorageCapacity.exact(depth,Tree.write_sequence(~F32,xs,i,tree),balanced(xs,i,tree,depth,proof)),Equal.sym(U32,Tree.capacity(F32,
        tree),U32.shln(1,depth),StorageCapacity.exact(depth,tree,proof)))

def take_at(+n: Nat,+i: Nat,+xs: List<&2,F32>,h: {Nat.is_lt(i,n) == True{} : Bool}) -> {ConvSpec.at(i,take(n,xs)) == ConvSpec.at(i,xs) : F32}:
  match n:
    case 0n:
      Empty.absurd({ConvSpec.at(i,[]) == ConvSpec.at(i,xs) : F32},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(i,0n),True{},
            Equal.sym(Bool,Nat.is_lt(i,0n),False{},Division.lt_zero(i)),h)))
    case 1n+ +p:
      match i xs:
        case 0n Nil{}:
          {==}
        case 0n v <> tail:
          {==}
        case 1n+j Nil{}:
          {==}
        case 1n+j v <> tail:
          take_at(p,j,tail,h)

# General slice readback: the filled interval may start anywhere in the array.
def filled_list_element(+start: Nat,+values: List<&2,F32>,+index: Nat,+original: List<&2,F32>,
  room: {Nat.is_le(Nat.add(start,ConvSpec.length(values)),ConvSpec.length(original)) == True{} : Bool},
  +inside: {Nat.is_lt(index,ConvSpec.length(values)) == True{} : Bool}) ->
  {ConvSpec.at(Nat.add(start,index),fill(values,start,original)) == ConvSpec.at(index,values) : F32}:
  match start:
    case 0n:
      Equal.trans(F32,ConvSpec.at(index,fill(values,0n,original)),ConvSpec.at(index,take(ConvSpec.length(values),fill(values,0n,original))),ConvSpec.at(index,values),
        Equal.sym(F32,ConvSpec.at(index,take(ConvSpec.length(values),fill(values,0n,original))),ConvSpec.at(index,fill(values,0n,original)),take_at(ConvSpec.length(values),index,fill(values,0n,original),inside)),
        Equal.cong(List<&2,F32>,F32,xs => ConvSpec.at(index,xs),take(ConvSpec.length(values),fill(values,0n,original)),values,prefix(values,original,room)))
    case 1n+ +rest:
      match original:
        case Nil{}:
          Empty.absurd({ConvSpec.at(1n+Nat.add(rest,index),fill(values,1n+rest,[])) == ConvSpec.at(index,values) : F32},WordEncoding.false_true(room))
        case +head <> +tail:
          %Equal.sym(List<&2,F32>,fill(values,1n+rest,head <> tail),head <> fill(values,rest,tail),offset(values,rest,head,tail)) :
            {ConvSpec.at(1n+Nat.add(rest,index),_) == ConvSpec.at(index,values) : F32}
          filled_list_element(rest,values,index,tail,room,inside)

def query_in_filled_range(+values: List<&2,F32>,+start: U32,+query: U32,+index: Nat,+capacity: U32,
  query_exact: {U32.to_nat(query) == Nat.add(U32.to_nat(start),index) : Nat},
  room: {Nat.is_le(Nat.add(U32.to_nat(start),ConvSpec.length(values)),U32.to_nat(capacity)) == True{} : Bool},
  inside: {Nat.is_lt(index,ConvSpec.length(values)) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(query),U32.to_nat(capacity)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(query),Nat.add(U32.to_nat(start),index),query_exact) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
  Bounds.nat_trans(Nat.add(U32.to_nat(start),index),Nat.add(U32.to_nat(start),ConvSpec.length(values)),U32.to_nat(capacity),Order.offset_lt(U32.to_nat(start),index,ConvSpec.length(values),inside),room)

def read_filled_range(+values: List<&2,F32>,+start: U32,+tree: Tree.Tree<F32>,+depth: Nat,+query: U32,+index: Nat,
  +balance_proof: Shape.Balanced(F32,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +query_exact: {U32.to_nat(query) == Nat.add(U32.to_nat(start),index) : Nat},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),ConvSpec.length(values)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +inside: {Nat.is_lt(index,ConvSpec.length(values)) == True{} : Bool}) ->
  {Tree.read(F32,Tree.write_sequence(~F32,values,start,tree),query) == ConvSpec.at(index,values) : F32}:
  +filled_tree = Tree.write_sequence(~F32,values,start,tree)
  +dense = TreeIndex.flatten(tree)
  +query_bound = U32Division.nat_lt_native(query,Tree.capacity(F32,filled_tree),
    %Equal.sym(U32,Tree.capacity(F32,filled_tree),Tree.capacity(F32,tree),capacity(values,start,tree,depth,balance_proof)) :
      {Nat.is_lt(U32.to_nat(query),U32.to_nat(_)) == True{} : Bool}
    query_in_filled_range(values,start,query,index,Tree.capacity(F32,tree),query_exact,room,inside))
  %filled_list_element(U32.to_nat(start),values,index,dense,
    %Equal.sym(Nat,ConvSpec.length(dense),U32.to_nat(Tree.capacity(F32,tree)),StorageCapacity.length(depth,tree,balance_proof,depth_ok)) :
      {Nat.is_le(Nat.add(U32.to_nat(start),ConvSpec.length(values)),_) == True{} : Bool}
    room,inside) : {Tree.read(F32,filled_tree,query) == _ : F32}
  %query_exact : {Tree.read(F32,filled_tree,query) == ConvSpec.at(_,fill(values,U32.to_nat(start),dense)) : F32}
  %filled(values,start,tree,depth,balance_proof,depth_ok,room) : {Tree.read(F32,filled_tree,query) == ConvSpec.at(U32.to_nat(query),_) : F32}
  DenseRead.dense(depth,filled_tree,query,balanced(values,start,tree,depth,balance_proof),depth_ok,query_bound)

# Prefix readback is the zero-offset instance of the same range theorem.
def read_filled_prefix(+values: List<&2,F32>,+tree: Tree.Tree<F32>,+depth: Nat,+query: U32,+balance_proof: Shape.Balanced(F32,depth,
    tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+room: {Nat.is_le(ConvSpec.length(values),
      U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(query),
      ConvSpec.length(values)) == True{} : Bool}) -> {Tree.read(F32,Tree.write_sequence(~F32,values,0,tree),
    query) == ConvSpec.at(U32.to_nat(query),values) : F32}:
  read_filled_range(values,0,tree,depth,query,U32.to_nat(query),balance_proof,depth_ok,{==},room,inside)

def read_filled_outside(+values: List<&2,F32>,+start: U32,+tree: Tree.Tree<F32>,+depth: Nat,+query: U32,
  +balance_proof: Shape.Balanced(F32,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),ConvSpec.length(values)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +query_inside: {U32.is_lt(query,Tree.capacity(F32,tree)) == True{} : Bool},outside: Interval.Outside<U32.to_nat(start),ConvSpec.length(values),U32.to_nat(query)>) ->
  {Tree.read(F32,Tree.write_sequence(~F32,values,start,tree),query) == Tree.read(F32,tree,query) : F32}:
  +updated = Tree.write_sequence(~F32,values,start,tree)
  +original = TreeIndex.flatten(tree)
  +new_inside = {
    %Equal.sym(U32,Tree.capacity(F32,updated),Tree.capacity(F32,tree),capacity(values,start,tree,depth,balance_proof)) :
      {U32.is_lt(query,_) == True{} : Bool}
    query_inside : {U32.is_lt(query,Tree.capacity(F32,updated)) == True{} : Bool}}
  %Equal.sym(F32,Tree.read(F32,updated,query),ConvSpec.at(U32.to_nat(query),TreeIndex.flatten(updated)),
    DenseRead.dense(depth,updated,query,balanced(values,start,tree,depth,balance_proof),depth_ok,new_inside)) :
    {_ == Tree.read(F32,tree,query) : F32}
  %Equal.sym(List<&2,F32>,TreeIndex.flatten(updated),fill(values,U32.to_nat(start),original),filled(values,start,tree,depth,balance_proof,depth_ok,room)) :
    {ConvSpec.at(U32.to_nat(query),_) == Tree.read(F32,tree,query) : F32}
  %Equal.sym(F32,ConvSpec.at(U32.to_nat(query),fill(values,U32.to_nat(start),original)),ConvSpec.at(U32.to_nat(query),original),
    fill_preserves_outside(values,U32.to_nat(start),U32.to_nat(query),original,
      %Equal.sym(Nat,ConvSpec.length(original),U32.to_nat(Tree.capacity(F32,tree)),StorageCapacity.length(depth,tree,balance_proof,depth_ok)) :
        {Nat.is_lt(U32.to_nat(query),_) == True{} : Bool}
      Bounds.as_nat_lt(query,Tree.capacity(F32,tree),query_inside),outside)) :
    {_ == Tree.read(F32,tree,query) : F32}
  Equal.sym(F32,Tree.read(F32,tree,query),ConvSpec.at(U32.to_nat(query),original),DenseRead.dense(depth,tree,query,balance_proof,depth_ok,query_inside))
