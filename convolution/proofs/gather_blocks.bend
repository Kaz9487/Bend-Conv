# The actual channel/block schedule. The inner traversal reuses the library's
# exact contiguous-copy theorem; concatenation preserves the order of stores.
import Base
import ../../lib/kernel_gather.bend as Gather
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/write_program.bend as Writes

def writes(count: Nat,+channel: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+input: Tree.Tree<F32>) -> List<&2,Writes.Assignment<F32>>:
  match count:
    case 0n: []
    case 1n+rest:
      List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),(channel * spatial + offset : U32)),
        writes(rest,(channel + 1 : U32),width,offset,spatial,span,input))

def channel(+channel: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+input: Tree.Tree<F32>,+output: Tree.Tree<F32>) ->
  {Gather.gather_channel(channel,width,offset,spatial,span,(Tree.pack(F32,input),Tree.pack(F32,output))) ==
    (Tree.pack(F32,input),Tree.pack(F32,Writes.pure(F32,Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),(channel * spatial + offset : U32)),output))) : Array<F32> & Array<F32>}:
  Contiguous.copy(~F32,U32.to_nat(width),input,output,(channel * span : U32),(channel * spatial + offset : U32))

def channels(+count: Nat,+row: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+input: Tree.Tree<F32>,+output: Tree.Tree<F32>) ->
  {Gather.gather_channels(count,row,width,offset,spatial,span,(Tree.pack(F32,input),Tree.pack(F32,output))) ==
    (Tree.pack(F32,input),Tree.pack(F32,Writes.pure(F32,writes(count,row,width,offset,spatial,span,input),output))) : Array<F32> & Array<F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Array<F32> & Array<F32>,Gather.gather_channel(row,width,offset,spatial,span,(Tree.pack(F32,input),Tree.pack(F32,output))),
        (Tree.pack(F32,input),Tree.pack(F32,Writes.pure(F32,Contiguous.writes(~F32,U32.to_nat(width),input,0,(row * span : U32),(row * spatial + offset : U32)),output))),
        channel(row,width,offset,spatial,span,input,output)) :
        {Gather.gather_channels(rest,(row + 1 : U32),width,offset,spatial,span,_) ==
          (Tree.pack(F32,input),Tree.pack(F32,Writes.pure(F32,writes(1n+rest,row,width,offset,spatial,span,input),output))) : Array<F32> & Array<F32>}
      %Equal.sym(Tree.Tree<F32>,Writes.pure(F32,writes(1n+rest,row,width,offset,spatial,span,input),output),
        Writes.pure(F32,writes(rest,(row + 1 : U32),width,offset,spatial,span,input),
          Writes.pure(F32,Contiguous.writes(~F32,U32.to_nat(width),input,0,(row * span : U32),(row * spatial + offset : U32)),output)),
        Writes.concatenated_writes(F32,Contiguous.writes(~F32,U32.to_nat(width),input,0,(row * span : U32),(row * spatial + offset : U32)),
          writes(rest,(row + 1 : U32),width,offset,spatial,span,input),output)) :
        {Gather.gather_channels(rest,(row + 1 : U32),width,offset,spatial,span,
          (Tree.pack(F32,input),Tree.pack(F32,Writes.pure(F32,Contiguous.writes(~F32,U32.to_nat(width),input,0,(row * span : U32),(row * spatial + offset : U32)),output)))) ==
          (Tree.pack(F32,input),Tree.pack(F32,_)) : Array<F32> & Array<F32>}
      channels(rest,(row + 1 : U32),width,offset,spatial,span,input,
        Writes.pure(F32,Contiguous.writes(~F32,U32.to_nat(width),input,0,(row * span : U32),(row * spatial + offset : U32)),output))

def block_writes(channels: U32,+offset: U32,+spatial: U32,+span: U32,input: Tree.Tree<F32>) -> List<&2,Writes.Assignment<F32>>:
  writes(U32.to_nat(channels),0,Gather.gather_width(offset,spatial,span),offset,spatial,span,input)

def block(+count: U32,+offset: U32,+spatial: U32,+span: U32,+input: Tree.Tree<F32>,+output: Tree.Tree<F32>) ->
  {Gather.gather_block(count,offset,spatial,span,Tree.pack(F32,input),Tree.pack(F32,output)) ==
    Tree.pack(F32,Writes.pure(F32,block_writes(count,offset,spatial,span,input),output)) : Array<F32>}:
  %Equal.sym(Array<F32> & Array<F32>,Gather.gather_channels(U32.to_nat(count),0,Gather.gather_width(offset,spatial,span),offset,spatial,span,(Tree.pack(F32,input),Tree.pack(F32,output))),
    (Tree.pack(F32,input),Tree.pack(F32,Writes.pure(F32,block_writes(count,offset,spatial,span,input),output))),
    channels(U32.to_nat(count),0,Gather.gather_width(offset,spatial,span),offset,spatial,span,input,output)) :
    {Gather.gathered_output(_) == Tree.pack(F32,Writes.pure(F32,block_writes(count,offset,spatial,span,input),output)) : Array<F32>}
  {==}
