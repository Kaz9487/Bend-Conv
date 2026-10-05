import Base
import ../../lib/traversal_partition.bend as Partition
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/write_program.bend as Writes
import ./gather_blocks.bend as Blocks

def ConvResult() -> Data:
  Partition.Tree<&2,Tree.Tree<F32>,U32>

def gather(r: ConvResult(), y: Tree.Tree<F32>, +spatial: U32) -> Tree.Tree<F32>:
  match r:
    case Partition.Leaf{x,offset,+span,output_channel_count}:
      Writes.pure(F32,Blocks.block_writes(output_channel_count,offset,spatial,span,x),y)
    case Partition.Fork{left,right}:
      gather(right,gather(left,y,spatial),spatial)

def finish_leaf(identity: Bool,x: Tree.Tree<F32>,offset: U32,+span: U32,+channels: U32,+spatial: U32) -> Tree.Tree<F32>:
  match identity:
    case True{}:
      x
    case False{}:
      Writes.pure(F32,Blocks.block_writes(channels,offset,spatial,span,x),Tree.fresh(F32,Storage.allocation_depth((channels * spatial : U32)),0.0))

def finish_result(result: ConvResult(),+output_channel_count: U32,+spatial: U32) -> Tree.Tree<F32>:
  match result:
    case Partition.Leaf{x,+offset,+span,+channels}:
      finish_leaf(U32.is_zero(offset) && U32.is_eq(span,spatial),x,offset,span,channels,spatial)
    case Partition.Fork{left,right}:
      gather(Partition.Fork{left,right},Tree.fresh(F32,Storage.allocation_depth((output_channel_count * spatial : U32)),0.0),spatial)

