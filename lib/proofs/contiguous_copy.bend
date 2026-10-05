# Exact stores of the shared contiguous copy, including preservation of its
# input owner. No bounds premise is needed for this execution refinement.
import Base
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./write_program.bend as Writes
import ./bounded_u32_arithmetic.bend as Arithmetic

def writes(~Element: Data,count: Nat,+input: Tree.Tree<Element>,+position: U32,+source: U32,+destination: U32) -> List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+rest:
      Writes.Put{(destination + position : U32),Tree.read(Element,input,(source + position : U32))} <>
        writes(~Element,rest,input,(position + 1 : U32),source,destination)

def traversal(~Element: Data,+count: Nat,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32,+source: U32,+destination: U32) ->
  {Traversal.finish(~Array<Element>,~Element,~Traversal.Offsets,
      Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.source_offset(~Element),~Traversal.destination_offset,count,
        Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),position,Traversal.Offsets{source,destination}})) ==
    (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,count,input,position,source,destination),output))) : Array<Element> & Array<Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),(source + position : U32)),
        (Tree.pack(Element,input),Tree.read(Element,input,(source + position : U32))),StorageLaws.read(Element,input,(source + position : U32))) :
        {Traversal.finish(~Array<Element>,~Element,~Traversal.Offsets,
          Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.source_offset(~Element),~Traversal.destination_offset,rest,
            Traversal.store(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.destination_offset,position,Traversal.Offsets{source,destination},Tree.pack(Element,output),_))) ==
          (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,1n+rest,input,position,source,destination),output))) : Array<Element> & Array<Element>}
      %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),(destination + position : U32),Tree.read(Element,input,(source + position : U32))),
        Tree.pack(Element,Tree.write(Element,output,(destination + position : U32),Tree.read(Element,input,(source + position : U32)))),
        StorageLaws.write(Element,output,(destination + position : U32),Tree.read(Element,input,(source + position : U32)))) :
        {Traversal.finish(~Array<Element>,~Element,~Traversal.Offsets,
          Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.source_offset(~Element),~Traversal.destination_offset,rest,
            Traversal.State{Tree.pack(Element,input),_,(position + 1 : U32),Traversal.Offsets{source,destination}})) ==
          (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,1n+rest,input,position,source,destination),output))) : Array<Element> & Array<Element>}
      traversal(~Element,rest,input,Tree.write(Element,output,(destination + position : U32),Tree.read(Element,input,(source + position : U32))),
        (position + 1 : U32),source,destination)

def copy(~Element: Data,+count: Nat,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+source: U32,+destination: U32) ->
  {Traversal.copy(~Element,count,source,destination,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,count,input,0,source,destination),output))) : Array<Element> & Array<Element>}:
  traversal(~Element,count,input,output,0,source,destination)

# The same copy program is a contiguous read sequence followed by a write
# sequence. This is shared by the raw window and certified band callers.
def sequence(~Element: Data,count: Nat,+input: Tree.Tree<Element>,+position: U32,+source: U32,+destination: U32,+tree: Tree.Tree<Element>) ->
  {Writes.pure(Element,writes(~Element,count,input,position,source,destination),tree) ==
    Tree.write_sequence(~Element,Tree.read_sequence(~Element,count,(source + position : U32),input),(destination + position : U32),tree) : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +value = Tree.read(Element,input,(source + position : U32))
      +written = Tree.write(Element,tree,(destination + position : U32),value)
      %Equal.sym(U32,((source + position : U32) + 1 : U32),(source + (position + 1 : U32) : U32),Arithmetic.add_associative(source,position,1)) :
        {Writes.pure(Element,writes(~Element,rest,input,(position + 1 : U32),source,destination),written) ==
          Tree.write_sequence(~Element,Tree.read_sequence(~Element,rest,_,input),((destination + position : U32) + 1 : U32),written) : Tree.Tree<Element>}
      %Equal.sym(U32,((destination + position : U32) + 1 : U32),(destination + (position + 1 : U32) : U32),Arithmetic.add_associative(destination,position,1)) :
        {Writes.pure(Element,writes(~Element,rest,input,(position + 1 : U32),source,destination),written) ==
          Tree.write_sequence(~Element,Tree.read_sequence(~Element,rest,(source + (position + 1 : U32) : U32),input),_,written) : Tree.Tree<Element>}
      sequence(~Element,rest,input,(position + 1 : U32),source,destination,written)
