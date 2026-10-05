# Erase data buffers from the actual prepare recursion, retaining every tile.
import Base
import ../../lib/traversal_partition.bend as Partition
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./packing_storage_refinement.bend as PackingStorage
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_gather.bend as Gather

def Layout() -> Data:
  Partition.Tree<&2,Unit,Unit>

def tiles(t: PackingStorage.Tiles()) -> Layout():
  match t:
    case Partition.Leaf{data,offset,span,Unit{}}:
      Partition.Leaf{Unit{},offset,span,Unit{}}
    case Partition.Fork{left,right}:
      Partition.Fork{tiles(left),tiles(right)}

def prepared(p: PackingStorage.Prepared) -> Layout():
  PackingStorage.Prepared{x,t} = p
  tiles(t)

def divide(depth: Nat,+offset: U32,+span: U32) -> Layout():
  match depth:
    case 0n:
      Partition.Leaf{Unit{},offset,span,Unit{}}
    case 1n+ +rest:
      +half = (span / 16 * 8 : U32)
      Partition.Fork{divide(rest,offset,half),divide(rest,(offset + half : U32),(span - half : U32))}

def leaf(+offset: U32,+span: U32,+pair: PackingStorage.Packed) -> {prepared(PackingStorage.leaf(offset,span,
      pair)) == Partition.Leaf{Unit{},offset,span,Unit{}} : Layout()}:
  PackingStorage.Packed{x,y} = pair
  {==}

def attach(+left: PackingStorage.Tiles(),+p: PackingStorage.Prepared) -> {prepared(PackingStorage.attach(left,
      p)) == Partition.Fork{tiles(left),prepared(p)} : Layout()}:
  PackingStorage.Prepared{x,right} = p
  {==}

def after_left(+depth: Nat,+offset: U32,+span: U32,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,+inner: U32,
  +p: PackingStorage.Prepared,ih: @x: Tree.Tree<F32> -> {prepared(PackingStorage.prepare(depth,
        (offset + span / 16 * 8 : U32),(span - span / 16 * 8 : U32),x,h,w,k,stride,pad,output_width,inner)) == divide(depth,(offset + span / 16 * 8 : U32),
      (span - span / 16 * 8 : U32)) : Layout()}) -> {prepared(PackingStorage.after_left(depth,offset,span,h,w,k,stride,pad,output_width,inner,
      p)) == Partition.Fork{prepared(p),divide(depth,(offset + span / 16 * 8 : U32),(span - span / 16 * 8 : U32))} : Layout()}:
  PackingStorage.Prepared{+x,+left} = p
  %ih(x) : {prepared(PackingStorage.after_left(depth,offset,span,h,w,k,stride,pad,output_width,inner,PackingStorage.Prepared{x,
          left})) == Partition.Fork{tiles(left),_} : Layout()}
  attach(left,PackingStorage.prepare(depth,(offset + span / 16 * 8 : U32),(span - span / 16 * 8 : U32),x,h,w,k,stride,pad,output_width,inner))

law prepare:
  for +depth: Nat
  for +offset: U32
  for +span: U32
  for +x: Tree.Tree<F32>
  for +h: U32
  for +w: U32
  for +k: U32
  for +stride: U32
  for +pad: U32
  for +output_width: U32
  for +inner: U32
  {prepared(PackingStorage.prepare(depth,offset,span,x,h,w,k,stride,pad,output_width,inner)) == divide(depth,offset,span) : Layout()}
def prepare(depth,offset,span,x,h,w,k,stride,pad,output_width,inner):
  match depth:
    case 0n:
      leaf(offset,span,PackingStorage.packing(U32.to_nat((span / 8 * inner : U32)),offset,x,
          Tree.fresh(Tile.Tile8,Storage.allocation_depth((span / 8 * inner : U32)),Tile.filled(0.0)),h,w,k,stride,pad,
          output_width,inner))
    case 1n+ +rest:
      +half = (span / 16 * 8 : U32)
      %prepare(rest,offset,half,x,h,w,k,stride,pad,output_width,inner) : {prepared(PackingStorage.prepare(1n+rest,offset,span,x,h,w,k,stride,pad,
            output_width,inner)) == Partition.Fork{_,divide(rest,(offset + half : U32),(span - half : U32))} : Layout()}
      after_left(rest,offset,span,h,w,k,stride,pad,output_width,inner,PackingStorage.prepare(rest,offset,half,x,h,w,k,stride,pad,output_width,inner),
        x => prepare(rest,(offset + half : U32),(span - half : U32),x,h,w,k,stride,pad,output_width,inner))
