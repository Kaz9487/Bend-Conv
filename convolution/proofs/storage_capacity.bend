import ../../lib/proofs/storage_shape.bend as Shape
import Base
import ../../lib/proofs/u32_shift.bend as U32Shift
import ../../lib/proofs/nat_order.bend as Order
import ./tree_indexing.bend as TreeIndex
import ../specification.bend as ConvSpec
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./array_tree_structure.bend as ArrayTreeStructure
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ./scalar_write_refinement.bend as ScalarWriteRefinement
import ./native_array_model.bend as NativeArrayModel

def exact(+depth: Nat,+tree: Tree.Tree<F32>,balanced: Shape.Balanced(F32,depth,
    tree)) -> {Tree.capacity(F32,tree) == U32.shln(1,depth) : U32}:
  Shape.capacity(F32,depth,tree,balanced)

def after_write(+depth: Nat,+tree: Tree.Tree<F32>,+i: U32,+v: F32,+balanced: Shape.Balanced(F32,depth,
    tree)) -> {Tree.capacity(F32,Tree.write(F32,tree,i,v)) == Tree.capacity(F32,tree) : U32}:
  Equal.trans(U32,Tree.capacity(F32,Tree.write(F32,tree,i,v)),U32.shln(1,depth),Tree.capacity(F32,tree),exact(depth,
      Tree.write(F32,tree,i,v),Shape.write(F32,depth,tree,i,v,balanced)),Equal.sym(U32,Tree.capacity(F32,tree),U32.shln(1,
        depth),exact(depth,tree,balanced)))

def depth_bound(+depth: Nat,h: {Nat.is_le(depth,30n) == True{} : Bool}) -> {U32.is_le(U32.shln(1,depth),2147483648) == True{} : Bool}:
  match depth:
    case 0n:
      {==}
    case 1n:
      {==}
    case 2n:
      {==}
    case 3n:
      {==}
    case 4n:
      {==}
    case 5n:
      {==}
    case 6n:
      {==}
    case 7n:
      {==}
    case 8n:
      {==}
    case 9n:
      {==}
    case 10n:
      {==}
    case 11n:
      {==}
    case 12n:
      {==}
    case 13n:
      {==}
    case 14n:
      {==}
    case 15n:
      {==}
    case 16n:
      {==}
    case 17n:
      {==}
    case 18n:
      {==}
    case 19n:
      {==}
    case 20n:
      {==}
    case 21n:
      {==}
    case 22n:
      {==}
    case 23n:
      {==}
    case 24n:
      {==}
    case 25n:
      {==}
    case 26n:
      {==}
    case 27n:
      {==}
    case 28n:
      {==}
    case 29n:
      {==}
    case 30n:
      {==}
    case 31n+rest:
      Empty.absurd({U32.is_le(U32.shln(1,31n+rest),2147483648) == True{} : Bool},WordEncoding.false_true(h))

def small(+depth: Nat,+tree: Tree.Tree<F32>,balanced: Shape.Balanced(F32,depth,tree),h: {Nat.is_le(depth,
      30n) == True{} : Bool}) -> {U32.is_le(Tree.capacity(F32,tree),2147483648) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(F32,tree),U32.shln(1,depth),exact(depth,tree,balanced)) : {U32.is_le(_,2147483648) == True{} : Bool}
  depth_bound(depth,h)

def power_small(+depth: Nat,h: {Nat.is_le(depth,30n) == True{} : Bool}) -> {U32.is_lt(U32.shln(1,depth),2147483648) == True{} : Bool}:
  match depth:
    case 0n:
      {==}
    case 1n:
      {==}
    case 2n:
      {==}
    case 3n:
      {==}
    case 4n:
      {==}
    case 5n:
      {==}
    case 6n:
      {==}
    case 7n:
      {==}
    case 8n:
      {==}
    case 9n:
      {==}
    case 10n:
      {==}
    case 11n:
      {==}
    case 12n:
      {==}
    case 13n:
      {==}
    case 14n:
      {==}
    case 15n:
      {==}
    case 16n:
      {==}
    case 17n:
      {==}
    case 18n:
      {==}
    case 19n:
      {==}
    case 20n:
      {==}
    case 21n:
      {==}
    case 22n:
      {==}
    case 23n:
      {==}
    case 24n:
      {==}
    case 25n:
      {==}
    case 26n:
      {==}
    case 27n:
      {==}
    case 28n:
      {==}
    case 29n:
      {==}
    case 30n:
      {==}
    case 31n+rest:
      Empty.absurd({U32.is_lt(U32.shln(1,31n+rest),2147483648) == True{} : Bool},WordEncoding.false_true(h))

def extent(+depth: Nat,h: {Nat.is_le(depth,30n) == True{} : Bool}) -> {U32.to_nat(U32.shln(1,depth)) == Shape.extent(depth) : Nat}:
  match depth:
    case 0n:
      {==}
    case 1n+ +d:
      +limit = Order.transitive(d,1n+d,30n,Order.successor(d,d,Order.reflexive(d)),h)
      %extent(d,limit) : {U32.to_nat(U32.shl(U32.shln(1,d))) == Nat.double(_) : Nat}
      U32Shift.double(U32.shln(1,d),power_small(d,limit))

def length(+depth: Nat,+tree: Tree.Tree<F32>,+balanced: Shape.Balanced(F32,depth,tree),h: {Nat.is_le(depth,
      30n) == True{} : Bool}) -> {ConvSpec.length(TreeIndex.flatten(tree)) == U32.to_nat(Tree.capacity(F32,
      tree)) : Nat}:
  %Equal.sym(U32,Tree.capacity(F32,tree),U32.shln(1,depth),exact(depth,tree,
      balanced)) : {ConvSpec.length(TreeIndex.flatten(tree)) == U32.to_nat(_) : Nat}
  %Equal.sym(Nat,U32.to_nat(U32.shln(1,depth)),Shape.extent(depth),extent(depth,
      h)) : {ConvSpec.length(TreeIndex.flatten(tree)) == _ : Nat}
  ArrayTreeStructure.length(depth,tree,balanced)
