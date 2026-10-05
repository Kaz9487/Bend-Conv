# Exact writes of the native channel/block gather; padding contributes no assignment.
import Base
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/write_program.bend as Writes
import ./gather_blocks.bend as Blocks
import ./gather_storage_model.bend as GatherModel
import ./bounded_write_readback.bend as WriteReadback
import ./conditional_write_coverage.bend as Coverage
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/boolean_logic.bend as Boolean

def prepend(keep: Bool,index: U32,value: F32,rest: List<&2,Writes.Assignment<F32>>) -> List<&2,Writes.Assignment<F32>>:
  match keep:
    case False{}: rest
    case True{}: Writes.Put{index,value} <> rest

def gather_writes(result: GatherModel.ConvResult(),+spatial: U32) -> List<&2,Writes.Assignment<F32>>:
  match result:
    case Partition.Leaf{input,offset,+span,channels}:
      Blocks.block_writes(channels,offset,spatial,span,input)
    case Partition.Fork{left,right}:
      List.append(&2,Writes.Assignment<F32>,gather_writes(left,spatial),gather_writes(right,spatial))

def gather_matches_writes(+result: GatherModel.ConvResult(),+output: Tree.Tree<F32>,+spatial: U32) ->
  {GatherModel.gather(result,output,spatial) == Writes.pure(F32,gather_writes(result,spatial),output) : Tree.Tree<F32>}:
  match result:
    case Partition.Leaf{input,offset,+span,channels}:
      {==}
    case Partition.Fork{+left,+right}:
      %Equal.sym(Tree.Tree<F32>,GatherModel.gather(left,output,spatial),Writes.pure(F32,gather_writes(left,spatial),output),gather_matches_writes(left,output,spatial)) :
        {GatherModel.gather(right,_,spatial) == Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,gather_writes(left,spatial),gather_writes(right,spatial)),output) : Tree.Tree<F32>}
      %Equal.sym(Tree.Tree<F32>,Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,gather_writes(left,spatial),gather_writes(right,spatial)),output),
        Writes.pure(F32,gather_writes(right,spatial),Writes.pure(F32,gather_writes(left,spatial),output)),
        Writes.concatenated_writes(F32,gather_writes(left,spatial),gather_writes(right,spatial),output)) :
        {GatherModel.gather(right,Writes.pure(F32,gather_writes(left,spatial),output),spatial) == _ : Tree.Tree<F32>}
      gather_matches_writes(right,Writes.pure(F32,gather_writes(left,spatial),output),spatial)

def contains(query: Nat,writes: List<&2,Writes.Assignment<F32>>) -> Bool:
  Coverage.contains(query,WriteReadback.natural_write_addresses(writes))

def concatenated_contains(+query: Nat,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>) ->
  {contains(query,List.append(&2,Writes.Assignment<F32>,left,right)) == contains(query,left) || contains(query,right) : Bool}:
  match left:
    case Nil{}: {==}
    case Writes.Put{+destination,value} <> +rest:
      %Equal.sym(Bool,contains(query,List.append(&2,Writes.Assignment<F32>,rest,right)),contains(query,rest) || contains(query,right),concatenated_contains(query,rest,right)) :
        {Nat.is_eq(query,U32.to_nat(destination)) || _ == (Nat.is_eq(query,U32.to_nat(destination)) || contains(query,rest)) || contains(query,right) : Bool}
      Boolean.or_associative(Nat.is_eq(query,U32.to_nat(destination)),contains(query,rest),contains(query,right))

def concatenation_covered(+query: Nat,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>,
  choice: Either<&2,&2,{contains(query,left) == True{} : Bool},{contains(query,right) == True{} : Bool}>) ->
  {contains(query,List.append(&2,Writes.Assignment<F32>,left,right)) == True{} : Bool}:
  match choice:
    case Inl{covered}:
      %Equal.sym(Bool,contains(query,List.append(&2,Writes.Assignment<F32>,left,right)),contains(query,left) || contains(query,right),concatenated_contains(query,left,right)) : {_ == True{} : Bool}
      %Equal.sym(Bool,contains(query,left),True{},covered) : {_ || contains(query,right) == True{} : Bool}
      {==}
    case Inr{covered}:
      %Equal.sym(Bool,contains(query,List.append(&2,Writes.Assignment<F32>,left,right)),contains(query,left) || contains(query,right),concatenated_contains(query,left,right)) : {_ == True{} : Bool}
      %Equal.sym(Bool,contains(query,right),True{},covered) : {contains(query,left) || _ == True{} : Bool}
      Boolean.or_true_right(contains(query,left))

def prepend_keeps_coverage(+keep: Bool,+query: Nat,+index: U32,+value: F32,+rest: List<&2,Writes.Assignment<F32>>,
  covered: {contains(query,rest) == True{} : Bool}) -> {contains(query,prepend(keep,index,value,rest)) == True{} : Bool}:
  match keep:
    case False{}: covered
    case True{}:
      %Equal.sym(Bool,contains(query,rest),True{},covered) : {Nat.is_eq(query,U32.to_nat(index)) || _ == True{} : Bool}
      Boolean.or_true_right(Nat.is_eq(query,U32.to_nat(index)))
