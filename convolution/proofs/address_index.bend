import ../../lib/proofs/storage_address_bounds.bend as Address
import ../../lib/proofs/storage_shape.bend as Shape
# Balanced-tree indexing, by tree depth; machine arithmetic is discharged below.
import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/natural_addition.bend as Addition
import ./tree_indexing.bend as TreeIndex
import ./native_array_model.bend as NativeArrayModel
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ./address_step.bend as AddressStep
import ../../lib/proofs/word_power_of_two.bend as Power
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/word_subtraction.bend as Subtraction

def Goal(+depth: Nat,+tree: Tree.Tree<F32>,+index: U32) -> Type:
  {TreeIndex.index(tree,NativeArrayModel.route(tree,Address.capacity(depth),index)) == U32.to_nat(index) : Nat}
law by_depth:
  for +depth: Nat
  for +tree: Tree.Tree<F32>
  for +index: U32
  for balanced: Shape.Balanced(F32,depth,tree)
  for limit: {Nat.is_le(depth,30n) == True{} : Bool}
  for bound: {U32.is_lt(index,Address.capacity(depth)) == True{} : Bool}
  Goal(depth,tree,index)

def branch(+depth: Nat,+left: Tree.Tree<F32>,+right: Tree.Tree<F32>,+index: U32,choose_left: Bool,
  +balanced_left: Shape.Balanced(F32,depth,left),+limit: {Nat.is_le(1n+depth,
      30n) == True{} : Bool},+bound: {U32.is_lt(index,Address.capacity(1n+depth)) == True{} : Bool},+choice: {U32.is_lt(index,
      Address.capacity(depth)) == choose_left : Bool},left_case: {U32.is_lt(index,Address.capacity(depth)) == True{} : Bool} -> Goal(depth,left,index),
  right_case: {U32.is_lt(U32.sub(index,Address.capacity(depth)),Address.capacity(depth)) == True{} : Bool} -> Goal(depth,right,U32.sub(index,
      Address.capacity(depth)))) -> {TreeIndex.index(Tree.Node{left,right},Bool.pick(Either<&2,&2,TreeIndex.Routes(left),
      TreeIndex.Routes(right)>,choose_left,Inl{NativeArrayModel.route(left,Address.capacity(depth),index)},Inr{NativeArrayModel.route(right,
        Address.capacity(depth),U32.sub(index,Address.capacity(depth)))})) == U32.to_nat(index) : Nat}:
  match choose_left:
    case True{}:
      left_case(choice)
    case False{}:
      AddressStep.right(depth,left,right,Address.capacity(depth),index,U32.sub(index,Address.capacity(depth)),balanced_left,Address.right_restores_index(depth,index,
          limit,bound,choice),right_case(Address.right_stays_in_child(depth,index,limit,bound,choice)))

def by_depth(depth,tree,index,balanced,limit,bound):
  match depth tree:
    case 0n Tree.Leaf{value}:
      Address.below_one_value(index,bound)
    case 0n Tree.Node{left,right}:
      match balanced:
    case 1n+rest Tree.Leaf{value}:
      match balanced:
    case 1n+ +rest Tree.Node{+left,+right}:
      Shape.Both{+balanced_left,+balanced_right} = balanced
      +limit = limit
      +bound = bound
      %Equal.sym(U32,U32.shr(Address.capacity(1n+rest)),Address.capacity(rest),Power.native_half_power(rest,Order.lt_succ(rest,30n,Address.child_limit(rest,
              limit)))) : {TreeIndex.index(Tree.Node{left,right},Bool.pick(Either<&2,&2,TreeIndex.Routes(left),
            TreeIndex.Routes(right)>,U32.is_lt(index,_),Inl{NativeArrayModel.route(left,_,index)},Inr{NativeArrayModel.route(right,_,
              U32.sub(index,_))})) == U32.to_nat(index) : Nat}
      branch(rest,left,right,index,U32.is_lt(index,Address.capacity(rest)),balanced_left,limit,bound,{==},
        in_left => by_depth(rest,left,index,balanced_left,Address.child_limit(rest,limit),in_left),
        in_right => by_depth(rest,right,U32.sub(index,Address.capacity(rest)),balanced_right,Address.child_limit(rest,limit),in_right))
