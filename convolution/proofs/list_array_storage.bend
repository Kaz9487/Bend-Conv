import Base
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../scalar_kernel.bend as ScalarConv
import ../specification.bend as ConvSpec

def representation(+xs: List<&2,F32>) -> Tree.Tree<F32>:
  Tree.write_sequence(~F32,xs,0,Tree.fresh(F32,Storage.allocation_depth(U32.from_nat(ConvSpec.length(xs))),0.0))

law packed:
  for +xs: List<&2,F32>
  {ScalarConv.pack(xs) == Tree.pack(F32,representation(xs)) : Array<F32>}
def packed(xs):
  %Equal.sym(Array<F32>,Array.new(F32,Storage.allocation_depth(U32.from_nat(ConvSpec.length(xs))),0.0),Tree.pack(F32,Tree.fresh(F32,
        Storage.allocation_depth(U32.from_nat(ConvSpec.length(xs))),0.0)),StorageLaws.fresh(F32,
      Storage.allocation_depth(U32.from_nat(ConvSpec.length(xs))),0.0)) : {Storage.write_list(F32,xs,0,_) == Tree.pack(F32,
      representation(xs)) : Array<F32>}
  StorageLaws.write_sequence(~F32,xs,0,Tree.fresh(F32,Storage.allocation_depth(U32.from_nat(ConvSpec.length(xs))),0.0))
