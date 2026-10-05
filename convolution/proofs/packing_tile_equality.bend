# A tile inside one output row, or one that ends in the next row, and the flat
# tabulation entry of the same position and reduction index are the same tile,
# lane by lane, once the address equations of each lane hold. The tabulation
# of consecutive entries splits at any count. No arithmetic fact is proved
# here.
import Base
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./packing_block_reads.bend as Reads
import ./packing_lane_reads.bend as LaneReads
import ./packing_tabulation.bend as PackingTabulation
import ./packing_loop_refinement.bend as LoopLaws

law fill_append:
  for +n: Nat
  for +m: Nat
  for +i: U32
  for +start: U32
  for +x: Tree.Tree<F32>
  for +y: Tree.Tree<Tile.Tile8>
  for +h: U32
  for +w: U32
  for +k: U32
  for +stride: U32
  for +pad: U32
  for +output_width: U32
  for +inner: U32
  {PackingTabulation.fill(Nat.add(n,m),i,start,x,y,h,w,k,stride,pad,output_width,inner) ==
    PackingTabulation.fill(m,PackingTabulation.end(n,i),start,x,PackingTabulation.fill(n,i,start,x,y,h,w,k,stride,pad,output_width,inner),h,w,k,stride,pad,output_width,inner) : Tree.Tree<Tile.Tile8>}
def fill_append(n,m,i,start,x,y,h,w,k,stride,pad,output_width,inner):
  match n:
    case 0n: {==}
    case 1n+ +rest:
      fill_append(rest,m,(i + 1 : U32),start,x,Tree.write(Tile.Tile8,y,i,PackingTabulation.entry(i,start,x,h,w,k,stride,pad,output_width,inner)),h,w,k,stride,pad,output_width,inner)

# The row and column of lane j, as the tabulation computes them and as the block reads them.
type LaneFacts<-position: U32,-q: U32,-j: Nat,-kr: U32,-kc: U32,-k: U32,-s: U32,-p: U32,-ow: U32> is Data:
  LaneFacts{
    row: {((PackingTabulation.indexed_position(position,j) / ow) * s + (q / k) % k - p : U32) == ((position / ow : U32) * s + kr - p : U32) : U32},
    column: {((PackingTabulation.indexed_position(position,j) % ow) * s + q % k - p : U32) == (((position % ow : U32) * s + kc - p : U32) + U32.from_nat(j) * s : U32) : U32}}

def lane_equal(+x: Tree.Tree<F32>,+position: U32,+q: U32,+j: Nat,+c: U32,+kr: U32,+kc: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+ow: U32,
  channel: {((q / (k * k)) * h * w : U32) == (c * h * w : U32) : U32},facts: LaneFacts<position,q,j,kr,kc,k,s,p,ow>) ->
  {Reads.block_sample(x,Packing.PixelBlock{((position / ow : U32) * s + kr - p : U32),((position % ow : U32) * s + kc - p : U32),(c * h * w : U32),h,w,s},j) ==
    LaneReads.value(x,PackingTabulation.indexed_position(position,j),q,h,w,k,s,p,ow) : F32}:
  LaneFacts{row,column} = facts
  %column : {Reads.value(x,((position / ow : U32) * s + kr - p : U32),_,(c * h * w : U32),h,w) == LaneReads.value(x,PackingTabulation.indexed_position(position,j),q,h,w,k,s,p,ow) : F32}
  %row : {Reads.value(x,_,((PackingTabulation.indexed_position(position,j) % ow) * s + q % k - p : U32),(c * h * w : U32),h,w) ==
    LaneReads.value(x,PackingTabulation.indexed_position(position,j),q,h,w,k,s,p,ow) : F32}
  %channel : {Reads.value(x,((PackingTabulation.indexed_position(position,j) / ow) * s + (q / k) % k - p : U32),((PackingTabulation.indexed_position(position,j) % ow) * s + q % k - p : U32),_,h,w) ==
    LaneReads.value(x,PackingTabulation.indexed_position(position,j),q,h,w,k,s,p,ow) : F32}
  {==}

def tile_equal(+x: Tree.Tree<F32>,+position: U32,+q: U32,+c: U32,+kr: U32,+kc: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+ow: U32,
  +channel: {((q / (k * k)) * h * w : U32) == (c * h * w : U32) : U32},
  lane0: LaneFacts<position,q,0n,kr,kc,k,s,p,ow>,lane1: LaneFacts<position,q,1n,kr,kc,k,s,p,ow>,lane2: LaneFacts<position,q,2n,kr,kc,k,s,p,ow>,
  lane3: LaneFacts<position,q,3n,kr,kc,k,s,p,ow>,lane4: LaneFacts<position,q,4n,kr,kc,k,s,p,ow>,lane5: LaneFacts<position,q,5n,kr,kc,k,s,p,ow>,
  lane6: LaneFacts<position,q,6n,kr,kc,k,s,p,ow>,lane7: LaneFacts<position,q,7n,kr,kc,k,s,p,ow>) ->
  {LoopLaws.block(x,(position / ow : U32),(position % ow : U32),c,kr,kc,h,w,s,p) == PackingTabulation.vector(x,position,q,h,w,k,s,p,ow) : Tile.Tile8}:
  TileLaws.tiles_equal(LoopLaws.block(x,(position / ow : U32),(position % ow : U32),c,kr,kc,h,w,s,p),PackingTabulation.vector(x,position,q,h,w,k,s,p,ow),
    lane_equal(x,position,q,0n,c,kr,kc,h,w,k,s,p,ow,channel,lane0),lane_equal(x,position,q,1n,c,kr,kc,h,w,k,s,p,ow,channel,lane1),
    lane_equal(x,position,q,2n,c,kr,kc,h,w,k,s,p,ow,channel,lane2),lane_equal(x,position,q,3n,c,kr,kc,h,w,k,s,p,ow,channel,lane3),
    lane_equal(x,position,q,4n,c,kr,kc,h,w,k,s,p,ow,channel,lane4),lane_equal(x,position,q,5n,c,kr,kc,h,w,k,s,p,ow,channel,lane5),
    lane_equal(x,position,q,6n,c,kr,kc,h,w,k,s,p,ow,channel,lane6),lane_equal(x,position,q,7n,c,kr,kc,h,w,k,s,p,ow,channel,lane7))

# Lane j of a tile that may end in the next output row: in the output row of
# lane 0, with the equations of that row, or at the column past the width in
# the next row.
type WrapLane<-position: U32,-q: U32,-j: Nat,-kr: U32,-kc: U32,-k: U32,-s: U32,-p: U32,-ow: U32> is Data:
  Same{same: {U32.is_lt(((position % ow : U32) + U32.from_nat(j) : U32),ow) == True{} : Bool},lane: LaneFacts<position,q,j,kr,kc,k,s,p,ow>}
  Next{next: {U32.is_lt(((position % ow : U32) + U32.from_nat(j) : U32),ow) == False{} : Bool},
    row: {((PackingTabulation.indexed_position(position,j) / ow) * s + (q / k) % k - p : U32) == (((position / ow : U32) + 1) * s + kr - p : U32) : U32},
    column: {((PackingTabulation.indexed_position(position,j) % ow) * s + q % k - p : U32) == (((position % ow : U32) + U32.from_nat(j) - ow) * s + kc - p : U32) : U32}}

def first_lane_equal(+x: Tree.Tree<F32>,+position: U32,+q: U32,+c: U32,+kr: U32,+kc: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+ow: U32,
  channel: {((q / (k * k)) * h * w : U32) == (c * h * w : U32) : U32},facts: LaneFacts<position,q,0n,kr,kc,k,s,p,ow>) ->
  {Reads.value(x,((position / ow : U32) * s + kr - p : U32),((position % ow : U32) * s + kc - p : U32),(c * h * w : U32),h,w) == LaneReads.value(x,position,q,h,w,k,s,p,ow) : F32}:
  %Reads.zero_lane_column(((position % ow : U32) * s + kc - p : U32),s) : {Reads.value(x,((position / ow : U32) * s + kr - p : U32),_,(c * h * w : U32),h,w) == LaneReads.value(x,position,q,h,w,k,s,p,ow) : F32}
  lane_equal(x,position,q,0n,c,kr,kc,h,w,k,s,p,ow,channel,facts)

def wrap_lane_equal(+x: Tree.Tree<F32>,+position: U32,+q: U32,+j: Nat,+c: U32,+kr: U32,+kc: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+ow: U32,
  channel: {((q / (k * k)) * h * w : U32) == (c * h * w : U32) : U32},facts: WrapLane<position,q,j,kr,kc,k,s,p,ow>) ->
  {Reads.wrap_sample(x,Packing.WrapBlock{((position / ow : U32) * s + kr - p : U32),((position % ow : U32) * s + kc - p : U32),(c * h * w : U32),h,w,s,(((position / ow : U32) + 1) * s + kr - p : U32),(position % ow : U32),ow,kc,p},j) == LaneReads.value(x,PackingTabulation.indexed_position(position,j),q,h,w,k,s,p,ow) : F32}:
  match facts:
    case Same{same,lane}:
      %Equal.sym(Bool,U32.is_lt(((position % ow : U32) + U32.from_nat(j) : U32),ow),True{},same) : {Reads.wrap_value(_,x,((position / ow : U32) * s + kr - p : U32),((position % ow : U32) * s + kc - p : U32),(c * h * w : U32),h,w,s,(((position / ow : U32) + 1) * s + kr - p : U32),(position % ow : U32),ow,kc,p,U32.from_nat(j)) == LaneReads.value(x,PackingTabulation.indexed_position(position,j),q,h,w,k,s,p,ow) : F32}
      lane_equal(x,position,q,j,c,kr,kc,h,w,k,s,p,ow,channel,lane)
    case Next{next,row,column}:
      %Equal.sym(Bool,U32.is_lt(((position % ow : U32) + U32.from_nat(j) : U32),ow),False{},next) : {Reads.wrap_value(_,x,((position / ow : U32) * s + kr - p : U32),((position % ow : U32) * s + kc - p : U32),(c * h * w : U32),h,w,s,(((position / ow : U32) + 1) * s + kr - p : U32),(position % ow : U32),ow,kc,p,U32.from_nat(j)) == LaneReads.value(x,PackingTabulation.indexed_position(position,j),q,h,w,k,s,p,ow) : F32}
      %column : {Reads.value(x,(((position / ow : U32) + 1) * s + kr - p : U32),_,(c * h * w : U32),h,w) == LaneReads.value(x,PackingTabulation.indexed_position(position,j),q,h,w,k,s,p,ow) : F32}
      %row : {Reads.value(x,_,((PackingTabulation.indexed_position(position,j) % ow) * s + q % k - p : U32),(c * h * w : U32),h,w) == LaneReads.value(x,PackingTabulation.indexed_position(position,j),q,h,w,k,s,p,ow) : F32}
      %channel : {Reads.value(x,((PackingTabulation.indexed_position(position,j) / ow) * s + (q / k) % k - p : U32),((PackingTabulation.indexed_position(position,j) % ow) * s + q % k - p : U32),_,h,w) == LaneReads.value(x,PackingTabulation.indexed_position(position,j),q,h,w,k,s,p,ow) : F32}
      {==}

def wrap_tile_equal(+x: Tree.Tree<F32>,+position: U32,+q: U32,+c: U32,+kr: U32,+kc: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+ow: U32,
  +channel: {((q / (k * k)) * h * w : U32) == (c * h * w : U32) : U32},
  lane0: LaneFacts<position,q,0n,kr,kc,k,s,p,ow>,lane1: WrapLane<position,q,1n,kr,kc,k,s,p,ow>,lane2: WrapLane<position,q,2n,kr,kc,k,s,p,ow>,lane3: WrapLane<position,q,3n,kr,kc,k,s,p,ow>,lane4: WrapLane<position,q,4n,kr,kc,k,s,p,ow>,lane5: WrapLane<position,q,5n,kr,kc,k,s,p,ow>,lane6: WrapLane<position,q,6n,kr,kc,k,s,p,ow>,lane7: WrapLane<position,q,7n,kr,kc,k,s,p,ow>) ->
  {LoopLaws.wrap_lanes(x,((position / ow : U32) * s + kr - p : U32),((position % ow : U32) * s + kc - p : U32),(c * h * w : U32),h,w,s,(((position / ow : U32) + 1) * s + kr - p : U32),(position % ow : U32),ow,kc,p) == PackingTabulation.vector(x,position,q,h,w,k,s,p,ow) : Tile.Tile8}:
  TileLaws.tiles_equal(LoopLaws.wrap_lanes(x,((position / ow : U32) * s + kr - p : U32),((position % ow : U32) * s + kc - p : U32),(c * h * w : U32),h,w,s,(((position / ow : U32) + 1) * s + kr - p : U32),(position % ow : U32),ow,kc,p),PackingTabulation.vector(x,position,q,h,w,k,s,p,ow),
    first_lane_equal(x,position,q,c,kr,kc,h,w,k,s,p,ow,channel,lane0),wrap_lane_equal(x,position,q,1n,c,kr,kc,h,w,k,s,p,ow,channel,lane1),wrap_lane_equal(x,position,q,2n,c,kr,kc,h,w,k,s,p,ow,channel,lane2),wrap_lane_equal(x,position,q,3n,c,kr,kc,h,w,k,s,p,ow,channel,lane3),wrap_lane_equal(x,position,q,4n,c,kr,kc,h,w,k,s,p,ow,channel,lane4),wrap_lane_equal(x,position,q,5n,c,kr,kc,h,w,k,s,p,ow,channel,lane5),wrap_lane_equal(x,position,q,6n,c,kr,kc,h,w,k,s,p,ow,channel,lane6),wrap_lane_equal(x,position,q,7n,c,kr,kc,h,w,k,s,p,ow,channel,lane7))
