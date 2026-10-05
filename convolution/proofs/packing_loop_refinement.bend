# The packer on storage trees: every loop of kernel_spatial_packing with the
# reads and writes of the tree model. A tile inside one output row stores the
# tabulated block of its row and column; a tile that ends in the next output
# row stores the lanes of both rows; any other tile stores the tabulated
# lanes of its position and reduction index. The native
# loops on packed trees are the packed model loops, with no premise.
import Base
import ../../lib/traversal_loop.bend as Loop
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_packing_samples.bend as PackingSamples
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/loop_projection.bend as Projection
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ./packing_block_reads.bend as Reads
import ./packing_lane_reads.bend as LaneReads
import ./packing_tabulation.bend as PackingTabulation

def RowTiles() -> Data:
  Packing.RowState<&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>>

def Crossings() -> Data:
  Packing.CrossingState<&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>>

def Tiled() -> Data:
  Packing.TileState<&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>>

def pack_row(state: RowTiles()) -> Packing.RowTiles():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  Packing.RowTile{Tree.pack(F32,x),Tree.pack(Tile.Tile8,y),i,c,kr,kc,row,offset,ow,h,w,s,p,taps}

def pack_crossing(state: Crossings()) -> Packing.Crossings():
  Packing.Crossing{x,y,i,q,position,h,w,k,s,p,ow} = state
  Packing.Crossing{Tree.pack(F32,x),Tree.pack(Tile.Tile8,y),i,q,position,h,w,k,s,p,ow}

def pack_tiles(state: Tiled()) -> Packing.Tiled():
  Packing.Tiles{x,y,i,position,h,w,k,s,p,ow,channels,taps,inner} = state
  Packing.Tiles{Tree.pack(F32,x),Tree.pack(Tile.Tile8,y),i,position,h,w,k,s,p,ow,channels,taps,inner}

# The block of a tile inside an output row, at one reduction entry.
def block(+x: Tree.Tree<F32>,+row: U32,+offset: U32,+c: U32,+kr: U32,+kc: U32,+h: U32,+w: U32,+s: U32,+p: U32) -> Tile.Tile8:
  Reads.pure_vector(x,Packing.PixelBlock{(row * s + kr - p : U32),(offset * s + kc - p : U32),(c * h * w : U32),h,w,s})

# The eight lanes of one reduction entry: of one output row, or of an output row and the next.
def Lanes() -> Type:
  Tree.Tree<F32> -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> Tile.Tile8

def row_lanes(x: Tree.Tree<F32>,iy: U32,ix: U32,cb: U32,h: U32,w: U32,s: U32,next_row: U32,offset: U32,ow: U32,kc: U32,p: U32) -> Tile.Tile8:
  Reads.pure_vector(x,Packing.PixelBlock{iy,ix,cb,h,w,s})

def wrap_lanes(x: Tree.Tree<F32>,iy: U32,ix: U32,cb: U32,h: U32,w: U32,s: U32,next_row: U32,offset: U32,ow: U32,kc: U32,p: U32) -> Tile.Tile8:
  Reads.pure_wrap_vector(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)

def column_step(~lanes: Lanes(),state: RowTiles()) -> RowTiles():
  Packing.RowTile{+x,y,+i,+c,+kr,+kc,+row,+offset,+ow,+h,+w,+s,+p,taps} = state
  Packing.RowTile{x,Tree.write(Tile.Tile8,y,i,lanes(x,(row * s + kr - p : U32),(offset * s + kc - p : U32),(c * h * w : U32),h,w,s,((row + 1) * s + kr - p : U32),offset,ow,kc,p)),(i + 1 : U32),c,kr,(kc + 1 : U32),row,offset,ow,h,w,s,p,taps}

def next_kernel_row(state: RowTiles()) -> RowTiles():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  Packing.RowTile{x,y,i,c,(kr + 1 : U32),0,row,offset,ow,h,w,s,p,taps}

def kernel_row_step(~lanes: Lanes(),state: RowTiles()) -> RowTiles():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,+taps} = state
  next_kernel_row(Loop.run(~RowTiles(),~column_step(~lanes),taps,Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps}))

def next_channel(state: RowTiles()) -> RowTiles():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  Packing.RowTile{x,y,i,(c + 1 : U32),0,0,row,offset,ow,h,w,s,p,taps}

def channel_step(~lanes: Lanes(),state: RowTiles()) -> RowTiles():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,+taps} = state
  next_channel(Loop.run(~RowTiles(),~kernel_row_step(~lanes),taps,Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps}))

def crossing_step(state: Crossings()) -> Crossings():
  Packing.Crossing{+x,y,+i,+q,+position,+h,+w,+k,+s,+p,+ow} = state
  Packing.Crossing{x,Tree.write(Tile.Tile8,y,i,PackingTabulation.vector(x,position,q,h,w,k,s,p,ow)),(i + 1 : U32),(q + 1 : U32),position,h,w,k,s,p,ow}

def after_row_tile(position: U32,kernel: U32,channels: Nat,inner: Nat,state: RowTiles()) -> Tiled():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  Packing.Tiles{x,y,i,(position + 8 : U32),h,w,kernel,s,p,ow,channels,taps,inner}

def after_crossing(channels: Nat,taps: Nat,inner: Nat,state: Crossings()) -> Tiled():
  Packing.Crossing{x,y,i,q,position,h,w,k,s,p,ow} = state
  Packing.Tiles{x,y,i,(position + 8 : U32),h,w,k,s,p,ow,channels,taps,inner}

def tile(within: Bool,wraps: Bool,x: Tree.Tree<F32>,y: Tree.Tree<Tile.Tile8>,i: U32,+position: U32,h: U32,w: U32,k: U32,s: U32,p: U32,+ow: U32,
  +channels: Nat,taps: Nat,+inner: Nat) -> Tiled():
  match within wraps:
    case True{} wraps:
      after_row_tile(position,k,channels,inner,Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),ow,h,w,s,p,taps}))
    case False{} True{}:
      after_row_tile(position,k,channels,inner,Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),ow,h,w,s,p,taps}))
    case False{} False{}:
      after_crossing(channels,taps,inner,Loop.run(~Crossings(),~crossing_step,inner,Packing.Crossing{x,y,i,0,position,h,w,k,s,p,ow}))

def tile_step(state: Tiled()) -> Tiled():
  Packing.Tiles{x,y,i,+position,h,w,k,s,p,+ow,channels,taps,inner} = state
  tile(U32.is_le((position % ow + 8 : U32),ow),U32.is_le((position % ow + 8 : U32),(ow + ow : U32)),x,y,i,position,h,w,k,s,p,ow,channels,taps,inner)

# Each native reader on a packed tree returns the lanes of the model.
def row_agree(+x: Tree.Tree<F32>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32) ->
  {Packing.block(Tree.pack(F32,x),iy,ix,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),row_lanes(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8}:
  Reads.native_vector_matches_tabulation(x,Packing.PixelBlock{iy,ix,cb,h,w,s})

def wrap_agree(+x: Tree.Tree<F32>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32) ->
  {Packing.wrap_block(Tree.pack(F32,x),iy,ix,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),wrap_lanes(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8}:
  Reads.native_wrap_vector_matches_tabulation(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)

def column_refines(~block: Packing.Block(),~lanes: Lanes(),
  ~agree: @+x: Tree.Tree<F32> -> @+iy: U32 -> @+ix: U32 -> @+cb: U32 -> @+h: U32 -> @+w: U32 -> @+s: U32 -> @+next_row: U32 -> @+offset: U32 -> @+ow: U32 ->
    @+kc: U32 -> @+p: U32 -> {block(Tree.pack(F32,x),iy,ix,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),lanes(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8},
  +state: RowTiles()) -> {Packing.column_step(~block,pack_row(state)) == pack_row(column_step(~lanes,state)) : Packing.RowTiles()}:
  Packing.RowTile{+x,+y,+i,+c,+kr,+kc,+row,+offset,+ow,+h,+w,+s,+p,+taps} = state
  +value = lanes(x,(row * s + kr - p : U32),(offset * s + kc - p : U32),(c * h * w : U32),h,w,s,((row + 1) * s + kr - p : U32),offset,ow,kc,p)
  %Equal.sym(Array<F32> & Tile.Tile8,block(Tree.pack(F32,x),(row * s + kr - p : U32),(offset * s + kc - p : U32),(c * h * w : U32),h,w,s,((row + 1) * s + kr - p : U32),offset,ow,kc,p),(Tree.pack(F32,x),value),
    agree(x,(row * s + kr - p : U32),(offset * s + kc - p : U32),(c * h * w : U32),h,w,s,((row + 1) * s + kr - p : U32),offset,ow,kc,p)) :
    {Packing.store_block(Tree.pack(Tile.Tile8,y),i,c,kr,kc,row,offset,ow,h,w,s,p,taps,_) ==
      pack_row(Packing.RowTile{x,Tree.write(Tile.Tile8,y,i,value),(i + 1 : U32),c,kr,(kc + 1 : U32),row,offset,ow,h,w,s,p,taps}) : Packing.RowTiles()}
  %Equal.sym(Array<Tile.Tile8>,Array.set(Tile.Tile8,Tree.pack(Tile.Tile8,y),i,value),Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,y,i,value)),
    StorageLaws.write(Tile.Tile8,y,i,value)) :
    {Packing.RowTile{Tree.pack(F32,x),_,(i + 1 : U32),c,kr,(kc + 1 : U32),row,offset,ow,h,w,s,p,taps} ==
      pack_row(Packing.RowTile{x,Tree.write(Tile.Tile8,y,i,value),(i + 1 : U32),c,kr,(kc + 1 : U32),row,offset,ow,h,w,s,p,taps}) : Packing.RowTiles()}
  {==}

def next_kernel_row_refines(+state: RowTiles()) -> {Packing.next_kernel_row(pack_row(state)) == pack_row(next_kernel_row(state)) : Packing.RowTiles()}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  {==}

def kernel_row_refines(~block: Packing.Block(),~lanes: Lanes(),
  ~agree: @+x: Tree.Tree<F32> -> @+iy: U32 -> @+ix: U32 -> @+cb: U32 -> @+h: U32 -> @+w: U32 -> @+s: U32 -> @+next_row: U32 -> @+offset: U32 -> @+ow: U32 ->
    @+kc: U32 -> @+p: U32 -> {block(Tree.pack(F32,x),iy,ix,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),lanes(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8},
  +state: RowTiles()) -> {Packing.kernel_row_step(~block,pack_row(state)) == pack_row(kernel_row_step(~lanes,state)) : Packing.RowTiles()}:
  Packing.RowTile{+x,+y,+i,+c,+kr,+kc,+row,+offset,+ow,+h,+w,+s,+p,+taps} = state
  +start = {Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} : RowTiles()}
  %Equal.sym(Packing.RowTiles(),Loop.run(~Packing.RowTiles(),~Packing.column_step(~block),taps,pack_row(start)),pack_row(Loop.run(~RowTiles(),~column_step(~lanes),taps,start)),
    Projection.iteration_refines_model(~Packing.RowTiles(),~RowTiles(),~Packing.column_step(~block),~column_step(~lanes),~pack_row,~column_refines(~block,~lanes,~agree),taps,start)) :
    {Packing.next_kernel_row(_) == pack_row(next_kernel_row(Loop.run(~RowTiles(),~column_step(~lanes),taps,start))) : Packing.RowTiles()}
  next_kernel_row_refines(Loop.run(~RowTiles(),~column_step(~lanes),taps,start))

def next_channel_refines(+state: RowTiles()) -> {Packing.next_channel(pack_row(state)) == pack_row(next_channel(state)) : Packing.RowTiles()}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  {==}

def channel_refines(~block: Packing.Block(),~lanes: Lanes(),
  ~agree: @+x: Tree.Tree<F32> -> @+iy: U32 -> @+ix: U32 -> @+cb: U32 -> @+h: U32 -> @+w: U32 -> @+s: U32 -> @+next_row: U32 -> @+offset: U32 -> @+ow: U32 ->
    @+kc: U32 -> @+p: U32 -> {block(Tree.pack(F32,x),iy,ix,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),lanes(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8},
  +state: RowTiles()) -> {Packing.channel_step(~block,pack_row(state)) == pack_row(channel_step(~lanes,state)) : Packing.RowTiles()}:
  Packing.RowTile{+x,+y,+i,+c,+kr,+kc,+row,+offset,+ow,+h,+w,+s,+p,+taps} = state
  +start = {Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} : RowTiles()}
  %Equal.sym(Packing.RowTiles(),Loop.run(~Packing.RowTiles(),~Packing.kernel_row_step(~block),taps,pack_row(start)),pack_row(Loop.run(~RowTiles(),~kernel_row_step(~lanes),taps,start)),
    Projection.iteration_refines_model(~Packing.RowTiles(),~RowTiles(),~Packing.kernel_row_step(~block),~kernel_row_step(~lanes),~pack_row,~kernel_row_refines(~block,~lanes,~agree),taps,start)) :
    {Packing.next_channel(_) == pack_row(next_channel(Loop.run(~RowTiles(),~kernel_row_step(~lanes),taps,start))) : Packing.RowTiles()}
  next_channel_refines(Loop.run(~RowTiles(),~kernel_row_step(~lanes),taps,start))

# The eight lanes of a crossing tile, read natively, are its tabulated vector.
def crossing_vector(+x: Tree.Tree<F32>,+position: U32,+q: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+ow: U32) ->
  {Tile.read_tail(~Array<F32>,~PackingSamples.PixelBlock,~PackingSamples.read_block_pixel,PackingSamples.pixel(Tree.pack(F32,x),position,q,h,w,k,s,p,ow),
      PackingSamples.pixel_block(position,q,h,w,k,s,p,ow)) == (Tree.pack(F32,x),PackingTabulation.vector(x,position,q,h,w,k,s,p,ow)) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,PackingSamples.pixel(Tree.pack(F32,x),position,q,h,w,k,s,p,ow),(Tree.pack(F32,x),LaneReads.value(x,position,q,h,w,k,s,p,ow)),
    LaneReads.pixel(x,position,q,h,w,k,s,p,ow)) :
    {Tile.read_tail(~Array<F32>,~PackingSamples.PixelBlock,~PackingSamples.read_block_pixel,_,PackingSamples.pixel_block(position,q,h,w,k,s,p,ow)) ==
      (Tree.pack(F32,x),PackingTabulation.vector(x,position,q,h,w,k,s,p,ow)) : Array<F32> & Tile.Tile8}
  TileLaws.tail_block_refines(~Array<F32>,~Tree.Tree<F32>,~PackingSamples.PixelBlock,~LaneReads.pack_input,~PackingSamples.read_block_pixel,
    ~LaneReads.block_sample,~LaneReads.native_block_read,x,PackingSamples.pixel_block(position,q,h,w,k,s,p,ow),LaneReads.value(x,position,q,h,w,k,s,p,ow))

def crossing_refines(+state: Crossings()) -> {Packing.crossing_step(pack_crossing(state)) == pack_crossing(crossing_step(state)) : Packing.Crossings()}:
  Packing.Crossing{+x,+y,+i,+q,+position,+h,+w,+k,+s,+p,+ow} = state
  +value = PackingTabulation.vector(x,position,q,h,w,k,s,p,ow)
  %Equal.sym(Array<F32> & Tile.Tile8,Tile.read_tail(~Array<F32>,~PackingSamples.PixelBlock,~PackingSamples.read_block_pixel,
      PackingSamples.pixel(Tree.pack(F32,x),position,q,h,w,k,s,p,ow),PackingSamples.pixel_block(position,q,h,w,k,s,p,ow)),(Tree.pack(F32,x),value),
    crossing_vector(x,position,q,h,w,k,s,p,ow)) :
    {Packing.store_crossing(Tree.pack(Tile.Tile8,y),i,q,position,h,w,k,s,p,ow,_) ==
      pack_crossing(Packing.Crossing{x,Tree.write(Tile.Tile8,y,i,value),(i + 1 : U32),(q + 1 : U32),position,h,w,k,s,p,ow}) : Packing.Crossings()}
  %Equal.sym(Array<Tile.Tile8>,Array.set(Tile.Tile8,Tree.pack(Tile.Tile8,y),i,value),Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,y,i,value)),
    StorageLaws.write(Tile.Tile8,y,i,value)) :
    {Packing.Crossing{Tree.pack(F32,x),_,(i + 1 : U32),(q + 1 : U32),position,h,w,k,s,p,ow} ==
      pack_crossing(Packing.Crossing{x,Tree.write(Tile.Tile8,y,i,value),(i + 1 : U32),(q + 1 : U32),position,h,w,k,s,p,ow}) : Packing.Crossings()}
  {==}

def after_row_tile_refines(+position: U32,+kernel: U32,+channels: Nat,+inner: Nat,+state: RowTiles()) ->
  {Packing.after_row_tile(position,kernel,channels,inner,pack_row(state)) == pack_tiles(after_row_tile(position,kernel,channels,inner,state)) : Packing.Tiled()}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  {==}

def after_crossing_refines(+channels: Nat,+taps: Nat,+inner: Nat,+state: Crossings()) ->
  {Packing.after_crossing(channels,taps,inner,pack_crossing(state)) == pack_tiles(after_crossing(channels,taps,inner,state)) : Packing.Tiled()}:
  Packing.Crossing{x,y,i,q,position,h,w,k,s,p,ow} = state
  {==}

def tile_refines(+within: Bool,+wraps: Bool,+x: Tree.Tree<F32>,+y: Tree.Tree<Tile.Tile8>,+i: U32,+position: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+ow: U32,
  +channels: Nat,+taps: Nat,+inner: Nat) ->
  {Packing.tile(within,wraps,Tree.pack(F32,x),Tree.pack(Tile.Tile8,y),i,position,h,w,k,s,p,ow,channels,taps,inner) ==
    pack_tiles(tile(within,wraps,x,y,i,position,h,w,k,s,p,ow,channels,taps,inner)) : Packing.Tiled()}:
  match within wraps:
    case True{} wraps:
      +start = {Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),ow,h,w,s,p,taps} : RowTiles()}
      %Equal.sym(Packing.RowTiles(),Loop.run(~Packing.RowTiles(),~Packing.channel_step(~Packing.block),channels,pack_row(start)),pack_row(Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,start)),
        Projection.iteration_refines_model(~Packing.RowTiles(),~RowTiles(),~Packing.channel_step(~Packing.block),~channel_step(~row_lanes),~pack_row,~channel_refines(~Packing.block,~row_lanes,~row_agree),channels,start)) :
        {Packing.after_row_tile(position,k,channels,inner,_) == pack_tiles(after_row_tile(position,k,channels,inner,Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,start))) : Packing.Tiled()}
      after_row_tile_refines(position,k,channels,inner,Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,start))
    case False{} True{}:
      +start = {Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),ow,h,w,s,p,taps} : RowTiles()}
      %Equal.sym(Packing.RowTiles(),Loop.run(~Packing.RowTiles(),~Packing.channel_step(~Packing.wrap_block),channels,pack_row(start)),pack_row(Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,start)),
        Projection.iteration_refines_model(~Packing.RowTiles(),~RowTiles(),~Packing.channel_step(~Packing.wrap_block),~channel_step(~wrap_lanes),~pack_row,~channel_refines(~Packing.wrap_block,~wrap_lanes,~wrap_agree),channels,start)) :
        {Packing.after_row_tile(position,k,channels,inner,_) == pack_tiles(after_row_tile(position,k,channels,inner,Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,start))) : Packing.Tiled()}
      after_row_tile_refines(position,k,channels,inner,Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,start))
    case False{} False{}:
      +start = {Packing.Crossing{x,y,i,0,position,h,w,k,s,p,ow} : Crossings()}
      %Equal.sym(Packing.Crossings(),Loop.run(~Packing.Crossings(),~Packing.crossing_step,inner,pack_crossing(start)),pack_crossing(Loop.run(~Crossings(),~crossing_step,inner,start)),
        Projection.iteration_refines_model(~Packing.Crossings(),~Crossings(),~Packing.crossing_step,~crossing_step,~pack_crossing,~crossing_refines,inner,start)) :
        {Packing.after_crossing(channels,taps,inner,_) == pack_tiles(after_crossing(channels,taps,inner,Loop.run(~Crossings(),~crossing_step,inner,start))) : Packing.Tiled()}
      after_crossing_refines(channels,taps,inner,Loop.run(~Crossings(),~crossing_step,inner,start))

def tile_step_refines(+state: Tiled()) -> {Packing.tile_step(pack_tiles(state)) == pack_tiles(tile_step(state)) : Packing.Tiled()}:
  Packing.Tiles{+x,+y,+i,+position,+h,+w,+k,+s,+p,+ow,+channels,+taps,+inner} = state
  tile_refines(U32.is_le((position % ow + 8 : U32),ow),U32.is_le((position % ow + 8 : U32),(ow + ow : U32)),x,y,i,position,h,w,k,s,p,ow,channels,taps,inner)

def tiles_refine(+count: Nat,+state: Tiled()) ->
  {Loop.run(~Packing.Tiled(),~Packing.tile_step,count,pack_tiles(state)) == pack_tiles(Loop.run(~Tiled(),~tile_step,count,state)) : Packing.Tiled()}:
  Projection.iteration_refines_model(~Packing.Tiled(),~Tiled(),~Packing.tile_step,~tile_step,~pack_tiles,~tile_step_refines,count,state)

# No loop changes its input tree.
def row_input(state: RowTiles()) -> Tree.Tree<F32>:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  x

def crossing_input(state: Crossings()) -> Tree.Tree<F32>:
  Packing.Crossing{x,y,i,q,position,h,w,k,s,p,ow} = state
  x

def tiles_input(state: Tiled()) -> Tree.Tree<F32>:
  Packing.Tiles{x,y,i,position,h,w,k,s,p,ow,channels,taps,inner} = state
  x

def column_keeps_input(~lanes: Lanes(),+state: RowTiles()) -> {row_input(column_step(~lanes,state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  {==}

def next_kernel_row_keeps_input(+state: RowTiles()) -> {row_input(next_kernel_row(state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  {==}

def kernel_row_keeps_input(~lanes: Lanes(),+state: RowTiles()) -> {row_input(kernel_row_step(~lanes,state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{+x,+y,+i,+c,+kr,+kc,+row,+offset,+ow,+h,+w,+s,+p,+taps} = state
  +start = {Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} : RowTiles()}
  %Projection.preserves_observation(~RowTiles(),~Tree.Tree<F32>,~column_step(~lanes),~row_input,~column_keeps_input(~lanes),taps,start) :
    {row_input(next_kernel_row(Loop.run(~RowTiles(),~column_step(~lanes),taps,start))) == _ : Tree.Tree<F32>}
  next_kernel_row_keeps_input(Loop.run(~RowTiles(),~column_step(~lanes),taps,start))

def next_channel_keeps_input(+state: RowTiles()) -> {row_input(next_channel(state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  {==}

def channel_keeps_input(~lanes: Lanes(),+state: RowTiles()) -> {row_input(channel_step(~lanes,state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{+x,+y,+i,+c,+kr,+kc,+row,+offset,+ow,+h,+w,+s,+p,+taps} = state
  +start = {Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} : RowTiles()}
  %Projection.preserves_observation(~RowTiles(),~Tree.Tree<F32>,~kernel_row_step(~lanes),~row_input,~kernel_row_keeps_input(~lanes),taps,start) :
    {row_input(next_channel(Loop.run(~RowTiles(),~kernel_row_step(~lanes),taps,start))) == _ : Tree.Tree<F32>}
  next_channel_keeps_input(Loop.run(~RowTiles(),~kernel_row_step(~lanes),taps,start))

def crossing_keeps_input(+state: Crossings()) -> {crossing_input(crossing_step(state)) == crossing_input(state) : Tree.Tree<F32>}:
  Packing.Crossing{x,y,i,q,position,h,w,k,s,p,ow} = state
  {==}

def after_row_tile_input(+position: U32,+kernel: U32,+channels: Nat,+inner: Nat,+state: RowTiles()) ->
  {tiles_input(after_row_tile(position,kernel,channels,inner,state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = state
  {==}

def after_crossing_input(+channels: Nat,+taps: Nat,+inner: Nat,+state: Crossings()) ->
  {tiles_input(after_crossing(channels,taps,inner,state)) == crossing_input(state) : Tree.Tree<F32>}:
  Packing.Crossing{x,y,i,q,position,h,w,k,s,p,ow} = state
  {==}

def tile_keeps_input(+within: Bool,+wraps: Bool,+x: Tree.Tree<F32>,+y: Tree.Tree<Tile.Tile8>,+i: U32,+position: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+ow: U32,
  +channels: Nat,+taps: Nat,+inner: Nat) -> {tiles_input(tile(within,wraps,x,y,i,position,h,w,k,s,p,ow,channels,taps,inner)) == x : Tree.Tree<F32>}:
  match within wraps:
    case True{} wraps:
      +start = {Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),ow,h,w,s,p,taps} : RowTiles()}
      %Projection.preserves_observation(~RowTiles(),~Tree.Tree<F32>,~channel_step(~row_lanes),~row_input,~channel_keeps_input(~row_lanes),channels,start) :
        {tiles_input(after_row_tile(position,k,channels,inner,Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,start))) == _ : Tree.Tree<F32>}
      after_row_tile_input(position,k,channels,inner,Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,start))
    case False{} True{}:
      +start = {Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),ow,h,w,s,p,taps} : RowTiles()}
      %Projection.preserves_observation(~RowTiles(),~Tree.Tree<F32>,~channel_step(~wrap_lanes),~row_input,~channel_keeps_input(~wrap_lanes),channels,start) :
        {tiles_input(after_row_tile(position,k,channels,inner,Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,start))) == _ : Tree.Tree<F32>}
      after_row_tile_input(position,k,channels,inner,Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,start))
    case False{} False{}:
      +start = {Packing.Crossing{x,y,i,0,position,h,w,k,s,p,ow} : Crossings()}
      %Projection.preserves_observation(~Crossings(),~Tree.Tree<F32>,~crossing_step,~crossing_input,~crossing_keeps_input,inner,start) :
        {tiles_input(after_crossing(channels,taps,inner,Loop.run(~Crossings(),~crossing_step,inner,start))) == _ : Tree.Tree<F32>}
      after_crossing_input(channels,taps,inner,Loop.run(~Crossings(),~crossing_step,inner,start))

def tile_step_keeps_input(+state: Tiled()) -> {tiles_input(tile_step(state)) == tiles_input(state) : Tree.Tree<F32>}:
  Packing.Tiles{+x,+y,+i,+position,+h,+w,+k,+s,+p,+ow,+channels,+taps,+inner} = state
  tile_keeps_input(U32.is_le((position % ow + 8 : U32),ow),U32.is_le((position % ow + 8 : U32),(ow + ow : U32)),x,y,i,position,h,w,k,s,p,ow,channels,taps,inner)

def tiles_keep_input(+count: Nat,+state: Tiled()) -> {tiles_input(Loop.run(~Tiled(),~tile_step,count,state)) == tiles_input(state) : Tree.Tree<F32>}:
  Projection.preserves_observation(~Tiled(),~Tree.Tree<F32>,~tile_step,~tiles_input,~tile_step_keeps_input,count,state)
