import Base
import ../../lib/traversal_partition.bend as Partition
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/kernel_tile.bend as Tile
import ../../lib/traversal_loop.bend as Loop
import ./packing_loop_refinement.bend as LoopLaws
import ../../lib/kernel_spatial_packing.bend as Packing

type Packed is Data:
  Packed{input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>}

def Tiles() -> Data:
  Partition.Tree<&2,Tree.Tree<Tile.Tile8>,Unit>

type Prepared is Data:
  Prepared{input: Tree.Tree<F32>,tiles: Tiles()}

def pack_pair(p: Packed) -> Array<F32> & Array<Tile.Tile8>:
  Packed{input_storage,packed_storage} = p
  (Tree.pack(F32,input_storage),Tree.pack(Tile.Tile8,packed_storage))

def finish_tiles(state: LoopLaws.Tiled()) -> Packed:
  Packing.Tiles{input_storage,packed_storage,index,position,input_height,input_width,kernel_size,stride,padding,output_width,channels,taps,inner} = state
  Packed{input_storage,packed_storage}

def finish_refines_storage(+state: LoopLaws.Tiled()) -> {Packing.finish(LoopLaws.pack_tiles(state)) == pack_pair(finish_tiles(state)) : Array<F32> & Array<Tile.Tile8>}:
  Packing.Tiles{input_storage,packed_storage,index,position,input_height,input_width,kernel_size,stride,padding,output_width,channels,taps,inner} = state
  {==}

def initial(start: U32,input_storage: Tree.Tree<F32>,packed_storage: Tree.Tree<Tile.Tile8>,input_height: U32,input_width: U32,
  +kernel_size: U32,stride: U32,padding: U32,output_width: U32,+reduction_length: U32) -> LoopLaws.Tiled():
  Packing.Tiles{input_storage,packed_storage,0,start,input_height,input_width,kernel_size,stride,padding,output_width,
    U32.to_nat((reduction_length / (kernel_size * kernel_size) : U32)),U32.to_nat(kernel_size),U32.to_nat(reduction_length)}

def packing(count: Nat,start: U32,input_storage: Tree.Tree<F32>,packed_storage: Tree.Tree<Tile.Tile8>,input_height: U32,input_width: U32,
  kernel_size: U32,stride: U32,padding: U32,output_width: U32,+reduction_length: U32) -> Packed:
  finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,Nat.div(count,U32.to_nat(reduction_length)),
    initial(start,input_storage,packed_storage,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length)))

law packing_refines_storage:
  for +count: Nat
  for +start: U32
  for +input_storage: Tree.Tree<F32>
  for +packed_storage: Tree.Tree<Tile.Tile8>
  for +input_height: U32
  for +input_width: U32
  for +kernel_size: U32
  for +stride: U32
  for +padding: U32
  for +output_width: U32
  for +reduction_length: U32
  {Packing.pack(count,start,Tree.pack(F32,input_storage),Tree.pack(Tile.Tile8,packed_storage),input_height,input_width,kernel_size,stride,padding,
    output_width,reduction_length) == pack_pair(packing(count,start,input_storage,packed_storage,input_height,input_width,kernel_size,stride,padding,output_width,
        reduction_length)) : Array<F32> & Array<Tile.Tile8>}
def packing_refines_storage(count,start,input_storage,packed_storage,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length):
  +tiles = Nat.div(count,U32.to_nat(reduction_length))
  +state = initial(start,input_storage,packed_storage,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length)
  %Equal.sym(Packing.Tiled(),Loop.run(~Packing.Tiled(),~Packing.tile_step,tiles,LoopLaws.pack_tiles(state)),
    LoopLaws.pack_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,tiles,state)),LoopLaws.tiles_refine(tiles,state)) :
    {Packing.finish(_) == pack_pair(finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,tiles,state))) : Array<F32> & Array<Tile.Tile8>}
  finish_refines_storage(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,tiles,state))

def leaf(offset: U32,span: U32,p: Packed) -> Prepared:
  Packed{input_storage,packed_storage} = p
  Prepared{input_storage,Partition.Leaf{packed_storage,offset,span,Unit{}}}

def attach(left: Tiles(),p: Prepared) -> Prepared:
  Prepared{input_storage,right} = p
  Prepared{input_storage,Partition.Fork{left,right}}

def bind_prepared(p: Prepared,f: Tree.Tree<F32> -> Tiles() -> Prepared) -> Prepared:
  Prepared{input_storage,t} = p
  f(input_storage,t)

def prepare(depth: Nat,+offset: U32,+spatial: U32,input_storage: Tree.Tree<F32>,+input_height: U32,+input_width: U32,+kernel_size: U32,+stride: U32,+padding: U32,
  +output_width: U32,
  +reduction_length: U32) -> Prepared:
  match depth:
    case 0n:
      leaf(offset,spatial,packing(U32.to_nat((spatial / 8 * reduction_length : U32)),offset,input_storage,Tree.fresh(Tile.Tile8,
            Storage.allocation_depth((spatial / 8 * reduction_length : U32)),Tile.filled(0.0)),input_height,input_width,kernel_size,stride,padding,output_width,
              reduction_length))
    case 1n+ +rest:
      +half = (spatial / 16 * 8 : U32)
      bind_prepared(prepare(rest,offset,half,input_storage,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length),input_storage => left =>
        attach(left,prepare(rest,(offset + half : U32),(spatial - half : U32),input_storage,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length)))

def after_left(depth: Nat,+offset: U32,+spatial: U32,input_height: U32,input_width: U32,kernel_size: U32,stride: U32,padding: U32,output_width: U32,reduction_length: U32,
  p: Prepared) -> Prepared:
  +half = (spatial / 16 * 8 : U32)
  bind_prepared(p,input_storage => left =>
    attach(left,prepare(depth,(offset + half : U32),(spatial - half : U32),input_storage,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length)))

# Read-only input invariants justify independent preparation of both partitions.
def packed_input(p: Packed) -> Tree.Tree<F32>:
  Packed{input_storage,packed_storage} = p
  input_storage

def finish_preserves_input(+state: LoopLaws.Tiled()) -> {packed_input(finish_tiles(state)) == LoopLaws.tiles_input(state) : Tree.Tree<F32>}:
  Packing.Tiles{input_storage,packed_storage,index,position,input_height,input_width,kernel_size,stride,padding,output_width,channels,taps,inner} = state
  {==}

law packing_preserves_input:
  for +count: Nat
  for +start: U32
  for +input_storage: Tree.Tree<F32>
  for +packed_storage: Tree.Tree<Tile.Tile8>
  for +input_height: U32
  for +input_width: U32
  for +kernel_size: U32
  for +stride: U32
  for +padding: U32
  for +output_width: U32
  for +reduction_length: U32
  {packed_input(packing(count,start,input_storage,packed_storage,input_height,input_width,kernel_size,stride,padding,output_width,
    reduction_length)) == input_storage : Tree.Tree<F32>}
def packing_preserves_input(count,start,input_storage,packed_storage,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length):
  +tiles = Nat.div(count,U32.to_nat(reduction_length))
  +state = initial(start,input_storage,packed_storage,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length)
  %LoopLaws.tiles_keep_input(tiles,state) : {packed_input(finish_tiles(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,tiles,state))) == _ : Tree.Tree<F32>}
  finish_preserves_input(Loop.run(~LoopLaws.Tiled(),~LoopLaws.tile_step,tiles,state))

def prepared_input(p: Prepared) -> Tree.Tree<F32>:
  Prepared{input_storage,t} = p
  input_storage

def leaf_preserves_input(+offset: U32,+span: U32,+p: Packed) -> {prepared_input(leaf(offset,span,
      p)) == packed_input(p) : Tree.Tree<F32>}:
  Packed{input_storage,packed_storage} = p
  {==}

def attach_preserves_input(+left: Tiles(),+p: Prepared) -> {prepared_input(attach(left,
      p)) == prepared_input(p) : Tree.Tree<F32>}:
  Prepared{input_storage,right} = p
  {==}

def after_left_preserves_input(+depth: Nat,+offset: U32,+span: U32,+input_height: U32,+input_width: U32,+kernel_size: U32,+stride: U32,+padding: U32,+output_width: U32,
  +reduction_length: U32,
  +p: Prepared,ih: @input_storage: Tree.Tree<F32> -> {prepared_input(prepare(depth,
        (offset + span / 16 * 8 : U32),(span - span / 16 * 8 : U32),input_storage,input_height,input_width,kernel_size,stride,padding,output_width,
        reduction_length)) == input_storage : Tree.Tree<F32>}) -> {prepared_input(after_left(depth,offset,span,input_height,input_width,kernel_size,stride,padding,
          output_width,
      reduction_length,p)) == prepared_input(p) : Tree.Tree<F32>}:
  Prepared{+input_storage,+left} = p
  %ih(input_storage) : {prepared_input(after_left(depth,offset,span,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length,Prepared{input_storage,
          left})) == _ : Tree.Tree<F32>}
  attach_preserves_input(left,prepare(depth,(offset + span / 16 * 8 : U32),(span - span / 16 * 8 : U32),input_storage,input_height,input_width,kernel_size,stride,padding,
    output_width,reduction_length))

law prepare_preserves_input:
  for +depth: Nat
  for +offset: U32
  for +span: U32
  for +input_storage: Tree.Tree<F32>
  for +input_height: U32
  for +input_width: U32
  for +kernel_size: U32
  for +stride: U32
  for +padding: U32
  for +output_width: U32
  for +reduction_length: U32
  {prepared_input(prepare(depth,offset,span,input_storage,input_height,input_width,kernel_size,stride,padding,output_width,
    reduction_length)) == input_storage : Tree.Tree<F32>}
def prepare_preserves_input(depth,offset,span,input_storage,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length):
  match depth:
    case 0n:
      +count = (span / 8 * reduction_length : U32)
      +zero = {Tile.filled(0.0) : Tile.Tile8}
      +packed_storage = Tree.fresh(Tile.Tile8,Storage.allocation_depth(count),zero)
      %packing_preserves_input(U32.to_nat(count),offset,input_storage,packed_storage,input_height,input_width,kernel_size,stride,padding,
        output_width,
        reduction_length) : {prepared_input(prepare(0n,offset,span,input_storage,input_height,input_width,kernel_size,stride,padding,output_width,
          reduction_length)) == _ : Tree.Tree<F32>}
      leaf_preserves_input(offset,span,packing(U32.to_nat(count),offset,input_storage,packed_storage,input_height,input_width,kernel_size,
        stride,padding,output_width,reduction_length))
    case 1n+ +rest:
      +half = (span / 16 * 8 : U32)
      %prepare_preserves_input(rest,offset,half,input_storage,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length) : {prepared_input(prepare(1n+rest,
        offset,span,input_storage,input_height,input_width,kernel_size,
            stride,padding,output_width,reduction_length)) == _ : Tree.Tree<F32>}
      after_left_preserves_input(rest,offset,span,input_height,input_width,kernel_size,stride,padding,output_width,reduction_length,prepare(rest,offset,half,input_storage,
        input_height,input_width,kernel_size,stride,padding,output_width,reduction_length),
        input_storage => prepare_preserves_input(rest,(offset + half : U32),(span - half : U32),input_storage,input_height,input_width,kernel_size,stride,padding,output_width,
          reduction_length))

def prepared_tiles(prepared: Prepared) -> Tiles():
  Prepared{input,tiles} = prepared
  tiles