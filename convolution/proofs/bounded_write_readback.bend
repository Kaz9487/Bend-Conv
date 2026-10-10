import ../../lib/proofs/storage_shape.bend as Shape
# Whole native write-list refinement, including every capacity/balance step.
import Base
import ../../lib/proofs/storage_readback.bend as StorageReadback
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/write_program.bend as Writes
import ./conditional_write_coverage.bend as Coverage
import ./dense_read_refinement.bend as DenseRead
import ./scalar_write_refinement.bend as ScalarWriteRefinement
import ./storage_capacity.bend as StorageCapacity
import ./tree_indexing.bend as TreeIndex
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../specification.bend as ConvSpec
import ./dense_array_initialization.bend as DenseArrayInitialization

# Exact read-over-write semantics. Both addresses must belong to the actual
# allocation, so this result cannot hide the native array's wrapping behavior.
def natural_write_addresses(writes: List<&2,Writes.Assignment<F32>>) -> List<&2,Coverage.Write>:
  match writes:
    case Nil{}:
      []
    case Writes.Put{i,v} <> rest:
      Coverage.Write{U32.to_nat(i),v} <> natural_write_addresses(rest)

def WritesWithinCapacity(writes: List<&2,Writes.Assignment<F32>>,+cap: U32) -> Data:
  match writes:
    case Nil{}:
      Unit
    case Writes.Put{i,v} <> rest:
      Shape.Both<{U32.is_lt(i,cap) == True{} : Bool},WritesWithinCapacity(rest,cap)>

def tail_bounds(+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,+depth: Nat,+i: U32,+v: F32,
  balanced: Shape.Balanced(F32,depth,tree),bounds: WritesWithinCapacity(writes,Tree.capacity(F32,
      tree))) -> WritesWithinCapacity(writes,Tree.capacity(F32,Tree.write(F32,tree,i,v))):
  %Equal.sym(U32,Tree.capacity(F32,Tree.write(F32,tree,i,v)),Tree.capacity(F32,tree),StorageCapacity.after_write(depth,tree,
      i,v,balanced)) : WritesWithinCapacity(writes,_)
  bounds

law writes_match_dense_updates:
  for +writes: List<&2,Writes.Assignment<F32>>
  for +tree: Tree.Tree<F32>
  for +depth: Nat
  for +balanced: Shape.Balanced(F32,depth,tree)
  for +limit: {Nat.is_le(depth,30n) == True{} : Bool}
  for bounds: WritesWithinCapacity(writes,Tree.capacity(F32,tree))
  {TreeIndex.flatten(Writes.pure(F32,writes,
          tree)) == Coverage.apply(natural_write_addresses(writes),TreeIndex.flatten(tree)) : List<&2,F32>}
def writes_match_dense_updates(writes,tree,depth,balanced,limit,bounds):
  match writes:
    case Nil{}:
      {==}
    case Writes.Put{+i,+v} <> +rest:
      Shape.Both{inside,tail} = bounds
      %ScalarWriteRefinement.dense(depth,tree,i,v,balanced,limit,inside) : {TreeIndex.flatten(Writes.pure(F32,rest,Tree.write(F32,tree,i,
              v))) == Coverage.apply(natural_write_addresses(rest),_) : List<&2,F32>}
      writes_match_dense_updates(rest,Tree.write(F32,tree,i,v),depth,Shape.write(F32,depth,tree,i,v,balanced),limit,tail_bounds(rest,tree,depth,i,v,
          balanced,tail))

def readback_from_matching_covered_writes(+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,+depth: Nat,+i: U32,+wanted: F32,
  +shape: Shape.Balanced(F32,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  bounds: WritesWithinCapacity(writes,Tree.capacity(F32,tree)),+inside: {U32.is_lt(i,Tree.capacity(F32,tree)) == True{} : Bool},
  correct: Coverage.MatchingWritesHaveValue(U32.to_nat(i),wanted,natural_write_addresses(writes)),
  coverage: {Coverage.contains(U32.to_nat(i),natural_write_addresses(writes)) == True{} : Bool}) -> {Tree.read(F32,
    Writes.pure(F32,writes,tree),i) == wanted : F32}:
  %Coverage.covered(natural_write_addresses(writes),U32.to_nat(i),wanted,TreeIndex.flatten(tree),correct,
    %Equal.sym(Nat,ConvSpec.length(TreeIndex.flatten(tree)),U32.to_nat(Tree.capacity(F32,tree)),
      StorageCapacity.length(depth,tree,shape,limit)) : {Nat.is_lt(U32.to_nat(i),_) == True{} : Bool}
    Bounds.as_nat_lt(i,Tree.capacity(F32,tree),inside),coverage) : {Tree.read(F32,Writes.pure(F32,writes,tree),
      i) == _ : F32}
  %writes_match_dense_updates(writes,tree,depth,shape,limit,bounds) : {Tree.read(F32,Writes.pure(F32,writes,tree),i) == ConvSpec.at(U32.to_nat(i),_) : F32}
  DenseRead.dense(depth,Writes.pure(F32,writes,tree),i,Writes.writes_preserve_balance(F32,writes,tree,depth,shape),limit,
    %Equal.sym(U32,Tree.capacity(F32,Writes.pure(F32,writes,tree)),Tree.capacity(F32,tree),Writes.writes_preserve_capacity(F32,writes,tree,depth,
        shape)) : {U32.is_lt(i,_) == True{} : Bool}
    inside)

# Bounds and matching values of one ordered write list, split so producers can
# supply evidence per block and concatenate it in execution order.
type WriteEvidence<-query: Nat,-value: F32,-capacity: U32,-writes: List<&2,Writes.Assignment<F32>>> is Data:
  WriteEvidence{bounds: WritesWithinCapacity(writes,capacity),values: Coverage.MatchingWritesHaveValue(query,value,natural_write_addresses(writes))}

def prepend_evidence(+query: Nat,+value: F32,+capacity: U32,+address: U32,+actual: F32,+rest: List<&2,Writes.Assignment<F32>>,
  inside: {U32.is_lt(address,capacity) == True{} : Bool},same: Coverage.Matches(Nat.is_eq(query,U32.to_nat(address)),actual,value),
  tail: WriteEvidence<query,value,capacity,rest>) -> WriteEvidence<query,value,capacity,Writes.Put{address,actual} <> rest>:
  WriteEvidence{bounds,values} = tail
  WriteEvidence{Shape.Both{inside,bounds},Shape.Both{same,values}}

def concatenated_evidence(+query: Nat,+value: F32,+capacity: U32,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>,
  first: WriteEvidence<query,value,capacity,left>,second: WriteEvidence<query,value,capacity,right>) ->
  WriteEvidence<query,value,capacity,List.append(&2,Writes.Assignment<F32>,left,right)>:
  match left:
    case Nil{}: second
    case Writes.Put{+address,+actual} <> +rest:
      WriteEvidence{bounds,values} = first
      Shape.Both{inside,tail_bounds} = bounds
      Shape.Both{same,tail_values} = values
      prepend_evidence(query,value,capacity,address,actual,List.append(&2,Writes.Assignment<F32>,rest,right),inside,same,
        concatenated_evidence(query,value,capacity,rest,right,WriteEvidence{tail_bounds,tail_values},second))

# A write list whose addresses exclude the query preserves its previous value.
def Excludes(+query: U32,writes: List<&2,Writes.Assignment<F32>>) -> Data:
  match writes:
    case Nil{}: Unit
    case Writes.Put{index,value} <> rest:
      Shape.Both<{U32.is_eq(index,query) == False{} : Bool},Excludes(query,rest)>

def excluded_readback(+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,+depth: Nat,+query: U32,
  +balanced: Shape.Balanced(F32,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  bounds: WritesWithinCapacity(writes,Tree.capacity(F32,tree)),+inside: {U32.is_lt(query,Tree.capacity(F32,tree)) == True{} : Bool},
  excludes: Excludes(query,writes)) -> {Tree.read(F32,Writes.pure(F32,writes,tree),query) == Tree.read(F32,tree,query) : F32}:
  match writes:
    case Nil{}: {==}
    case Writes.Put{+index,+value} <> +rest:
      Shape.Both{+index_inside,rest_inside} = bounds
      Shape.Both{different,rest_excludes} = excludes
      +next = Tree.write(F32,tree,index,value)
      Equal.trans(F32,Tree.read(F32,Writes.pure(F32,rest,next),query),Tree.read(F32,next,query),Tree.read(F32,tree,query),
        excluded_readback(rest,next,depth,query,Shape.write(F32,depth,tree,index,value,balanced),depth_ok,
          tail_bounds(rest,tree,depth,index,value,balanced,rest_inside),
          %Equal.sym(U32,Tree.capacity(F32,next),Tree.capacity(F32,tree),StorageCapacity.after_write(depth,tree,index,value,balanced)) : {U32.is_lt(query,_) == True{} : Bool} inside,
          rest_excludes),
        StorageReadback.read_preserved(~F32,depth,tree,index,query,value,balanced,depth_ok,index_inside,inside,different))

def evidence_bounds(+query: Nat,+value: F32,+capacity: U32,+writes: List<&2,Writes.Assignment<F32>>,
  evidence: WriteEvidence<query,value,capacity,writes>) -> WritesWithinCapacity(writes,capacity):
  WriteEvidence{bounds,values} = evidence
  bounds

def evidence_values(+query: Nat,+value: F32,+capacity: U32,+writes: List<&2,Writes.Assignment<F32>>,
  evidence: WriteEvidence<query,value,capacity,writes>) -> Coverage.MatchingWritesHaveValue(query,value,natural_write_addresses(writes)):
  WriteEvidence{bounds,values} = evidence
  values
