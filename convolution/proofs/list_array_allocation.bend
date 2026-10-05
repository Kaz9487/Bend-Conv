import ../../lib/proofs/storage_shape.bend as Shape
# Actual ScalarConv.pack -> original logical List, including allocator adequacy.
import Base
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./list_array_storage.bend as ListStorage
import ./dense_read_refinement.bend as DenseRead
import ./dense_array_initialization.bend as DenseArrayInitialization
import ./storage_capacity.bend as StorageCapacity
import ../../lib/proofs/array_allocation_bounds.bend as ArrayAllocation
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/u32_division.bend as U32Division
import ./tree_indexing.bend as TreeIndex
import ../specification.bend as ConvSpec
import ../scalar_kernel.bend as ScalarConv

def depth(+xs: List<&2,F32>) -> Nat:
  Storage.allocation_depth(U32.from_nat(ConvSpec.length(xs)))

def initial(+xs: List<&2,F32>) -> Tree.Tree<F32>:
  Tree.fresh(F32,depth(xs),0.0)

def Ready(+xs: List<&2,F32>) -> Data:
  Shape.Both<{Nat.is_le(depth(xs),30n) == True{} : Bool},{Nat.is_le(ConvSpec.length(xs),U32.to_nat(Tree.capacity(F32,
          initial(xs)))) == True{} : Bool}>

def ready_finish(+xs: List<&2,F32>,eq: {U32.to_nat(U32.from_nat(ConvSpec.length(xs))) == ConvSpec.length(xs) : Nat},
  proof: ArrayAllocation.Allocation(U32.from_nat(ConvSpec.length(xs)))) -> Ready(xs):
  Shape.Both{limit,room} = proof
  Shape.Both{limit,
    %Equal.sym(U32,Tree.capacity(F32,initial(xs)),U32.shln(1,depth(xs)),StorageCapacity.exact(depth(xs),initial(xs),
        Shape.fresh(F32,depth(xs),0.0))) : {Nat.is_le(ConvSpec.length(xs),U32.to_nat(_)) == True{} : Bool}
    %eq : {Nat.is_le(_,U32.to_nat(U32.shln(1,depth(xs)))) == True{} : Bool}
    room}

def ready(+xs: List<&2,F32>,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},+upper_bound: {U32.is_le(upper,
      1073741824) == True{} : Bool},positive: {Nat.is_lt(0n,ConvSpec.length(xs)) == True{} : Bool},+bound: {Nat.is_le(ConvSpec.length(xs),
      U32.to_nat(upper)) == True{} : Bool}) -> Ready(xs):
  +eq = Bounds.roundtrip(ConvSpec.length(xs),upper,upper_ok,bound)
  ready_finish(xs,eq,ArrayAllocation.allocated(U32.from_nat(ConvSpec.length(xs)),upper,upper_ok,upper_bound,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.length(xs))),ConvSpec.length(xs),eq) : {Nat.is_lt(0n,_) == True{} : Bool}
    positive,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.length(xs))),ConvSpec.length(xs),eq) : {Nat.is_le(_,U32.to_nat(upper)) == True{} : Bool}
    bound))

def prefix(+xs: List<&2,F32>,proof: Ready(xs)) -> {DenseArrayInitialization.take(ConvSpec.length(xs),
    TreeIndex.flatten(ListStorage.representation(xs))) == xs : List<&2,F32>}:
  Shape.Both{+limit,+room} = proof
  %Equal.sym(List<&2,F32>,TreeIndex.flatten(ListStorage.representation(xs)),DenseArrayInitialization.fill(xs,
      0n,TreeIndex.flatten(initial(xs))),DenseArrayInitialization.filled(xs,0,initial(xs),depth(xs),
      Shape.fresh(F32,depth(xs),0.0),limit,room)) : {DenseArrayInitialization.take(ConvSpec.length(xs),_) == xs : List<&2,F32>}
  DenseArrayInitialization.prefix(xs,TreeIndex.flatten(initial(xs)),
    %Equal.sym(Nat,ConvSpec.length(TreeIndex.flatten(initial(xs))),U32.to_nat(Tree.capacity(F32,initial(xs))),
      StorageCapacity.length(depth(xs),initial(xs),Shape.fresh(F32,depth(xs),0.0),limit)) : {Nat.is_le(ConvSpec.length(xs),_) == True{} : Bool}
    room)

def inside(+xs: List<&2,F32>,+i: U32,proof: Ready(xs),h: {Nat.is_lt(U32.to_nat(i),ConvSpec.length(xs)) == True{} : Bool}) -> {U32.is_lt(i,
    Tree.capacity(F32,ListStorage.representation(xs))) == True{} : Bool}:
  Shape.Both{limit,room} = proof
  %Equal.sym(U32,Tree.capacity(F32,ListStorage.representation(xs)),Tree.capacity(F32,initial(xs)),
    DenseArrayInitialization.capacity(xs,0,initial(xs),depth(xs),Shape.fresh(F32,depth(xs),0.0))) : {U32.is_lt(i,_) == True{} : Bool}
  U32Division.nat_lt_native(i,Tree.capacity(F32,initial(xs)),Bounds.nat_trans(U32.to_nat(i),ConvSpec.length(xs),
      U32.to_nat(Tree.capacity(F32,initial(xs))),h,room))

law read:
  for +xs: List<&2,F32>
  for +i: U32
  for +proof: Ready(xs)
  for +index_bound: {Nat.is_lt(U32.to_nat(i),ConvSpec.length(xs)) == True{} : Bool}
  {Tree.read(F32,ListStorage.representation(xs),i) == ConvSpec.at(U32.to_nat(i),xs) : F32}

def read(xs,i,proof,index_bound):
  Shape.Both{+limit,+room} = proof
  %prefix(xs,Shape.Both{limit,room}) : {Tree.read(F32,ListStorage.representation(xs),i) == ConvSpec.at(U32.to_nat(i),
      _) : F32}
  %Equal.sym(F32,ConvSpec.at(U32.to_nat(i),DenseArrayInitialization.take(ConvSpec.length(xs),
        TreeIndex.flatten(ListStorage.representation(xs)))),ConvSpec.at(U32.to_nat(i),
      TreeIndex.flatten(ListStorage.representation(xs))),DenseArrayInitialization.take_at(ConvSpec.length(xs),
      U32.to_nat(i),TreeIndex.flatten(ListStorage.representation(xs)),index_bound)) : {Tree.read(F32,
      ListStorage.representation(xs),i) == _ : F32}
  DenseRead.dense(depth(xs),ListStorage.representation(xs),i,DenseArrayInitialization.balanced(xs,0,initial(xs),depth(xs),
      Shape.fresh(F32,depth(xs),0.0)),limit,inside(xs,i,
        Shape.Both{limit,room},index_bound))

def expression_exact(+xs: List<&2,F32>,+address: U32,+index: Nat,proof: Ready(xs),encoded: {address == U32.from_nat(index) : U32},
  index_bound: {Nat.is_lt(index,ConvSpec.length(xs)) == True{} : Bool}) -> {U32.to_nat(address) == index : Nat}:
  Shape.Both{limit,room} = proof
  %Equal.sym(U32,address,U32.from_nat(index),encoded) : {U32.to_nat(_) == index : Nat}
  Bounds.roundtrip(index,Tree.capacity(F32,initial(xs)),StorageCapacity.small(depth(xs),initial(xs),
      Shape.fresh(F32,depth(xs),0.0),limit),Bounds.strict_le(index,U32.to_nat(Tree.capacity(F32,initial(xs))),
      Bounds.nat_trans(index,ConvSpec.length(xs),U32.to_nat(Tree.capacity(F32,initial(xs))),index_bound,room)))

def read_encoded(+xs: List<&2,F32>,+address: U32,+index: Nat,+proof: Ready(xs),encoded: {address == U32.from_nat(index) : U32},
  +index_bound: {Nat.is_lt(index,ConvSpec.length(xs)) == True{} : Bool}) -> {Tree.read(F32,ListStorage.representation(xs),
    address) == ConvSpec.at(index,xs) : F32}:
  +eq = expression_exact(xs,address,index,proof,encoded,index_bound)
  %eq : {Tree.read(F32,ListStorage.representation(xs),address) == ConvSpec.at(_,xs) : F32}
  read(xs,address,proof,
    %Equal.sym(Nat,U32.to_nat(address),index,eq) : {Nat.is_lt(_,ConvSpec.length(xs)) == True{} : Bool}
    index_bound)
