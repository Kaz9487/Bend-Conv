# The packer hands back its input unchanged and keeps the witness of its
# output: every loop of kernel_spatial_packing preserves both.
import Base
import ../traversal_loop.bend as Loop
import ../kernel_spatial_packing.bend as Packing
import ../kernel_packing_samples.bend as PackingSamples
import ../kernel_tile.bend as Tile
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness
import ./kernel_packing_read_witness.bend as PackingReadWitness
import ./kernel_tile_witness.bend as TileWitness

type RowCursor is Data:
  RowCursor{index: U32,channel: U32,kernel_row: U32,kernel_column: U32,row: U32,offset: U32,output_width: U32,height: U32,width: U32,stride: U32,padding: U32,taps: Nat}

def restore_row(input: Array<F32>,output: Array<Tile.Tile8>,cursor: RowCursor) -> Packing.RowTiles():
  RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = cursor
  Packing.RowTile{input,output,i,c,kr,kc,row,offset,ow,h,w,s,p,taps}

type RowEvidence<-input: Array<F32>,-depth: Nat,-state: Packing.RowTiles()> is Data:
  RowEvidence{-output: Array<Tile.Tile8>,cursor: RowCursor,equation: {state == restore_row(input,output,cursor) : Packing.RowTiles()},balanced: Witness.Witness(Tile.Tile8,depth,output)}

def stored_row(output: Array<Tile.Tile8>,cursor: RowCursor,result: Array<F32> & Tile.Tile8) -> Packing.RowTiles():
  RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = cursor
  Packing.store_block(output,i,c,kr,kc,row,offset,ow,h,w,s,p,taps,result)

def store_row(+depth: Nat,-input: Array<F32>,-output: Array<Tile.Tile8>,cursor: RowCursor,-result: Array<F32> & Tile.Tile8,
  balanced: Witness.Witness(Tile.Tile8,depth,output),sample: TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,result>) ->
  RowEvidence<input,depth,stored_row(output,cursor,result)>:
  RowCursor{+i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = cursor
  TraversalWitness.Sample{value,equation} = sample
  %Equal.sym(Array<F32> & Tile.Tile8,result,(input,value),equation) : RowEvidence<input,depth,stored_row(output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps},_)>
  RowEvidence{Array.set(Tile.Tile8,output,i,value),RowCursor{(i + 1 : U32),c,kr,(kc + 1 : U32),row,offset,ow,h,w,s,p,taps},{==},Witness.write(Tile.Tile8,depth,output,i,value,balanced)}

def column(~block: Packing.Block(),
  ~sampled: @-input: Array<F32> -> @+row: U32 -> @+column: U32 -> @+channel: U32 -> @+height: U32 -> @+width: U32 -> @+stride: U32 -> @+next_row: U32 -> @+offset: U32 -> @+output_width: U32 ->
    @+kernel_column: U32 -> @+padding: U32 ->
    @+evidence: PackingReadWitness.family(input) -> TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,block(input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding)>,
  +depth: Nat,-input: Array<F32>,-state: Packing.RowTiles(),+input_evidence: PackingReadWitness.family(input),evidence: RowEvidence<input,depth,state>) ->
  RowEvidence<input,depth,Packing.column_step(~block,state)>:
  RowEvidence{output,cursor,equation,balanced} = evidence
  RowCursor{+i,+c,+kr,+kc,+row,+offset,+ow,+h,+w,+s,+p,+taps} = cursor
  %Equal.sym(Packing.RowTiles(),state,restore_row(input,output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps}),equation) : RowEvidence<input,depth,Packing.column_step(~block,_)>
  +iy = (row * s + kr - p : U32)
  +ix = (offset * s + kc - p : U32)
  +cb = (c * h * w : U32)
  store_row(depth,input,output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps},block(input,iy,ix,cb,h,w,s,((row + 1) * s + kr - p : U32),offset,ow,kc,p),balanced,
    sampled(input,iy,ix,cb,h,w,s,((row + 1) * s + kr - p : U32),offset,ow,kc,p,input_evidence))

def column_loop(~block: Packing.Block(),
  ~sampled: @-input: Array<F32> -> @+row: U32 -> @+column: U32 -> @+channel: U32 -> @+height: U32 -> @+width: U32 -> @+stride: U32 -> @+next_row: U32 -> @+offset: U32 -> @+output_width: U32 ->
    @+kernel_column: U32 -> @+padding: U32 ->
    @+evidence: PackingReadWitness.family(input) -> TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,block(input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding)>,
  +depth: Nat,count: Nat,-input: Array<F32>,-state: Packing.RowTiles(),+input_evidence: PackingReadWitness.family(input),evidence: RowEvidence<input,depth,state>) ->
  RowEvidence<input,depth,Loop.run(~Packing.RowTiles(),~Packing.column_step(~block),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: column_loop(~block,~sampled,depth,rest,input,Packing.column_step(~block,state),input_evidence,column(~block,~sampled,depth,input,state,input_evidence,evidence))

def after_columns(-depth: Nat,-input: Array<F32>,-state: Packing.RowTiles(),evidence: RowEvidence<input,depth,state>) ->
  RowEvidence<input,depth,Packing.next_kernel_row(state)>:
  RowEvidence{output,cursor,equation,balanced} = evidence
  RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = cursor
  %Equal.sym(Packing.RowTiles(),state,restore_row(input,output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps}),equation) : RowEvidence<input,depth,Packing.next_kernel_row(_)>
  RowEvidence{output,RowCursor{i,c,(kr + 1 : U32),0,row,offset,ow,h,w,s,p,taps},{==},balanced}

def kernel_row(~block: Packing.Block(),
  ~sampled: @-input: Array<F32> -> @+row: U32 -> @+column: U32 -> @+channel: U32 -> @+height: U32 -> @+width: U32 -> @+stride: U32 -> @+next_row: U32 -> @+offset: U32 -> @+output_width: U32 ->
    @+kernel_column: U32 -> @+padding: U32 ->
    @+evidence: PackingReadWitness.family(input) -> TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,block(input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding)>,
  +depth: Nat,-input: Array<F32>,-state: Packing.RowTiles(),+input_evidence: PackingReadWitness.family(input),evidence: RowEvidence<input,depth,state>) ->
  RowEvidence<input,depth,Packing.kernel_row_step(~block,state)>:
  RowEvidence{output,cursor,equation,balanced} = evidence
  RowCursor{+i,+c,+kr,+kc,+row,+offset,+ow,+h,+w,+s,+p,+taps} = cursor
  %Equal.sym(Packing.RowTiles(),state,restore_row(input,output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps}),equation) : RowEvidence<input,depth,Packing.kernel_row_step(~block,_)>
  after_columns(depth,input,Loop.run(~Packing.RowTiles(),~Packing.column_step(~block),taps,restore_row(input,output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps})),
    column_loop(~block,~sampled,depth,taps,input,restore_row(input,output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps}),input_evidence,
      RowEvidence{output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps},{==},balanced}))

def kernel_row_loop(~block: Packing.Block(),
  ~sampled: @-input: Array<F32> -> @+row: U32 -> @+column: U32 -> @+channel: U32 -> @+height: U32 -> @+width: U32 -> @+stride: U32 -> @+next_row: U32 -> @+offset: U32 -> @+output_width: U32 ->
    @+kernel_column: U32 -> @+padding: U32 ->
    @+evidence: PackingReadWitness.family(input) -> TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,block(input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding)>,
  +depth: Nat,count: Nat,-input: Array<F32>,-state: Packing.RowTiles(),+input_evidence: PackingReadWitness.family(input),evidence: RowEvidence<input,depth,state>) ->
  RowEvidence<input,depth,Loop.run(~Packing.RowTiles(),~Packing.kernel_row_step(~block),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: kernel_row_loop(~block,~sampled,depth,rest,input,Packing.kernel_row_step(~block,state),input_evidence,kernel_row(~block,~sampled,depth,input,state,input_evidence,evidence))

def after_kernel_rows(-depth: Nat,-input: Array<F32>,-state: Packing.RowTiles(),evidence: RowEvidence<input,depth,state>) ->
  RowEvidence<input,depth,Packing.next_channel(state)>:
  RowEvidence{output,cursor,equation,balanced} = evidence
  RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = cursor
  %Equal.sym(Packing.RowTiles(),state,restore_row(input,output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps}),equation) : RowEvidence<input,depth,Packing.next_channel(_)>
  RowEvidence{output,RowCursor{i,(c + 1 : U32),0,0,row,offset,ow,h,w,s,p,taps},{==},balanced}

def channel(~block: Packing.Block(),
  ~sampled: @-input: Array<F32> -> @+row: U32 -> @+column: U32 -> @+channel: U32 -> @+height: U32 -> @+width: U32 -> @+stride: U32 -> @+next_row: U32 -> @+offset: U32 -> @+output_width: U32 ->
    @+kernel_column: U32 -> @+padding: U32 ->
    @+evidence: PackingReadWitness.family(input) -> TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,block(input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding)>,
  +depth: Nat,-input: Array<F32>,-state: Packing.RowTiles(),+input_evidence: PackingReadWitness.family(input),evidence: RowEvidence<input,depth,state>) ->
  RowEvidence<input,depth,Packing.channel_step(~block,state)>:
  RowEvidence{output,cursor,equation,balanced} = evidence
  RowCursor{+i,+c,+kr,+kc,+row,+offset,+ow,+h,+w,+s,+p,+taps} = cursor
  %Equal.sym(Packing.RowTiles(),state,restore_row(input,output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps}),equation) : RowEvidence<input,depth,Packing.channel_step(~block,_)>
  after_kernel_rows(depth,input,Loop.run(~Packing.RowTiles(),~Packing.kernel_row_step(~block),taps,restore_row(input,output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps})),
    kernel_row_loop(~block,~sampled,depth,taps,input,restore_row(input,output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps}),input_evidence,
      RowEvidence{output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps},{==},balanced}))

def channel_loop(~block: Packing.Block(),
  ~sampled: @-input: Array<F32> -> @+row: U32 -> @+column: U32 -> @+channel: U32 -> @+height: U32 -> @+width: U32 -> @+stride: U32 -> @+next_row: U32 -> @+offset: U32 -> @+output_width: U32 ->
    @+kernel_column: U32 -> @+padding: U32 ->
    @+evidence: PackingReadWitness.family(input) -> TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,block(input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding)>,
  +depth: Nat,count: Nat,-input: Array<F32>,-state: Packing.RowTiles(),+input_evidence: PackingReadWitness.family(input),evidence: RowEvidence<input,depth,state>) ->
  RowEvidence<input,depth,Loop.run(~Packing.RowTiles(),~Packing.channel_step(~block),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: channel_loop(~block,~sampled,depth,rest,input,Packing.channel_step(~block,state),input_evidence,channel(~block,~sampled,depth,input,state,input_evidence,evidence))

type CrossingCursor is Data:
  CrossingCursor{index: U32,reduction: U32,position: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,output_width: U32}

def restore_crossing(input: Array<F32>,output: Array<Tile.Tile8>,cursor: CrossingCursor) -> Packing.Crossings():
  CrossingCursor{i,q,position,h,w,k,s,p,ow} = cursor
  Packing.Crossing{input,output,i,q,position,h,w,k,s,p,ow}

type CrossingEvidence<-input: Array<F32>,-depth: Nat,-state: Packing.Crossings()> is Data:
  CrossingEvidence{-output: Array<Tile.Tile8>,cursor: CrossingCursor,equation: {state == restore_crossing(input,output,cursor) : Packing.Crossings()},
    balanced: Witness.Witness(Tile.Tile8,depth,output)}

def stored_crossing(output: Array<Tile.Tile8>,cursor: CrossingCursor,result: Array<F32> & Tile.Tile8) -> Packing.Crossings():
  CrossingCursor{i,q,position,h,w,k,s,p,ow} = cursor
  Packing.store_crossing(output,i,q,position,h,w,k,s,p,ow,result)

def store_crossing(+depth: Nat,-input: Array<F32>,-output: Array<Tile.Tile8>,cursor: CrossingCursor,-result: Array<F32> & Tile.Tile8,
  balanced: Witness.Witness(Tile.Tile8,depth,output),sample: TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,result>) ->
  CrossingEvidence<input,depth,stored_crossing(output,cursor,result)>:
  CrossingCursor{+i,q,position,h,w,k,s,p,ow} = cursor
  TraversalWitness.Sample{value,equation} = sample
  %Equal.sym(Array<F32> & Tile.Tile8,result,(input,value),equation) : CrossingEvidence<input,depth,stored_crossing(output,CrossingCursor{i,q,position,h,w,k,s,p,ow},_)>
  CrossingEvidence{Array.set(Tile.Tile8,output,i,value),CrossingCursor{(i + 1 : U32),(q + 1 : U32),position,h,w,k,s,p,ow},{==},Witness.write(Tile.Tile8,depth,output,i,value,balanced)}

def crossing(+depth: Nat,-input: Array<F32>,-state: Packing.Crossings(),+input_evidence: PackingReadWitness.family(input),evidence: CrossingEvidence<input,depth,state>) ->
  CrossingEvidence<input,depth,Packing.crossing_step(state)>:
  CrossingEvidence{output,cursor,equation,balanced} = evidence
  CrossingCursor{+i,+q,+position,+h,+w,+k,+s,+p,+ow} = cursor
  %Equal.sym(Packing.Crossings(),state,restore_crossing(input,output,CrossingCursor{i,q,position,h,w,k,s,p,ow}),equation) : CrossingEvidence<input,depth,Packing.crossing_step(_)>
  +context = PackingSamples.pixel_block(position,q,h,w,k,s,p,ow)
  -first = PackingSamples.pixel(input,position,q,h,w,k,s,p,ow)
  store_crossing(depth,input,output,CrossingCursor{i,q,position,h,w,k,s,p,ow},Tile.read_tail(~Array<F32>,~PackingSamples.PixelBlock,~PackingSamples.read_block_pixel,first,context),balanced,
    TileWitness.tail(~Array<F32>,~PackingSamples.PixelBlock,~PackingReadWitness.family,~PackingSamples.read_block_pixel,~PackingReadWitness.crossing_read,input,context,first,input_evidence,
      PackingReadWitness.first(input,position,q,h,w,k,s,p,ow,input_evidence)))

def crossing_loop(+depth: Nat,count: Nat,-input: Array<F32>,-state: Packing.Crossings(),+input_evidence: PackingReadWitness.family(input),evidence: CrossingEvidence<input,depth,state>) ->
  CrossingEvidence<input,depth,Loop.run(~Packing.Crossings(),~Packing.crossing_step,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: crossing_loop(depth,rest,input,Packing.crossing_step(state),input_evidence,crossing(depth,input,state,input_evidence,evidence))

type TileCursor is Data:
  TileCursor{index: U32,position: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,output_width: U32,channels: Nat,taps: Nat,inner: Nat}

def restore_tiles(input: Array<F32>,output: Array<Tile.Tile8>,cursor: TileCursor) -> Packing.Tiled():
  TileCursor{i,position,h,w,k,s,p,ow,channels,taps,inner} = cursor
  Packing.Tiles{input,output,i,position,h,w,k,s,p,ow,channels,taps,inner}

type TileEvidence<-input: Array<F32>,-depth: Nat,-state: Packing.Tiled()> is Data:
  TileEvidence{-output: Array<Tile.Tile8>,cursor: TileCursor,equation: {state == restore_tiles(input,output,cursor) : Packing.Tiled()},balanced: Witness.Witness(Tile.Tile8,depth,output)}

def after_row_tile(-depth: Nat,-input: Array<F32>,position: U32,kernel: U32,channels: Nat,inner: Nat,-state: Packing.RowTiles(),
  evidence: RowEvidence<input,depth,state>) -> TileEvidence<input,depth,Packing.after_row_tile(position,kernel,channels,inner,state)>:
  RowEvidence{output,cursor,equation,balanced} = evidence
  RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps} = cursor
  %Equal.sym(Packing.RowTiles(),state,restore_row(input,output,RowCursor{i,c,kr,kc,row,offset,ow,h,w,s,p,taps}),equation) :
    TileEvidence<input,depth,Packing.after_row_tile(position,kernel,channels,inner,_)>
  TileEvidence{output,TileCursor{i,(position + 8 : U32),h,w,kernel,s,p,ow,channels,taps,inner},{==},balanced}

def after_crossing(-depth: Nat,-input: Array<F32>,channels: Nat,taps: Nat,inner: Nat,-state: Packing.Crossings(),evidence: CrossingEvidence<input,depth,state>) ->
  TileEvidence<input,depth,Packing.after_crossing(channels,taps,inner,state)>:
  CrossingEvidence{output,cursor,equation,balanced} = evidence
  CrossingCursor{i,q,position,h,w,k,s,p,ow} = cursor
  %Equal.sym(Packing.Crossings(),state,restore_crossing(input,output,CrossingCursor{i,q,position,h,w,k,s,p,ow}),equation) :
    TileEvidence<input,depth,Packing.after_crossing(channels,taps,inner,_)>
  TileEvidence{output,TileCursor{i,(position + 8 : U32),h,w,k,s,p,ow,channels,taps,inner},{==},balanced}

def tile(+depth: Nat,within: Bool,wraps: Bool,-input: Array<F32>,-output: Array<Tile.Tile8>,+i: U32,+position: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+ow: U32,
  +channels: Nat,+taps: Nat,+inner: Nat,+input_evidence: PackingReadWitness.family(input),balanced: Witness.Witness(Tile.Tile8,depth,output)) ->
  TileEvidence<input,depth,Packing.tile(within,wraps,input,output,i,position,h,w,k,s,p,ow,channels,taps,inner)>:
  match within wraps:
    case True{} wraps:
      +cursor = {RowCursor{i,0,0,0,(position / ow : U32),(position % ow : U32),ow,h,w,s,p,taps} : RowCursor}
      after_row_tile(depth,input,position,k,channels,inner,Loop.run(~Packing.RowTiles(),~Packing.channel_step(~Packing.block),channels,restore_row(input,output,cursor)),
        channel_loop(~Packing.block,~PackingReadWitness.row_block,depth,channels,input,restore_row(input,output,cursor),input_evidence,RowEvidence{output,cursor,{==},balanced}))
    case False{} True{}:
      +cursor = {RowCursor{i,0,0,0,(position / ow : U32),(position % ow : U32),ow,h,w,s,p,taps} : RowCursor}
      after_row_tile(depth,input,position,k,channels,inner,Loop.run(~Packing.RowTiles(),~Packing.channel_step(~Packing.wrap_block),channels,restore_row(input,output,cursor)),
        channel_loop(~Packing.wrap_block,~PackingReadWitness.wrap_block,depth,channels,input,restore_row(input,output,cursor),input_evidence,RowEvidence{output,cursor,{==},balanced}))
    case False{} False{}:
      +cursor = {CrossingCursor{i,0,position,h,w,k,s,p,ow} : CrossingCursor}
      after_crossing(depth,input,channels,taps,inner,Loop.run(~Packing.Crossings(),~Packing.crossing_step,inner,restore_crossing(input,output,cursor)),
        crossing_loop(depth,inner,input,restore_crossing(input,output,cursor),input_evidence,CrossingEvidence{output,cursor,{==},balanced}))

def tile_step(+depth: Nat,-input: Array<F32>,-state: Packing.Tiled(),+input_evidence: PackingReadWitness.family(input),evidence: TileEvidence<input,depth,state>) ->
  TileEvidence<input,depth,Packing.tile_step(state)>:
  TileEvidence{output,cursor,equation,balanced} = evidence
  TileCursor{+i,+position,+h,+w,+k,+s,+p,+ow,+channels,+taps,+inner} = cursor
  %Equal.sym(Packing.Tiled(),state,restore_tiles(input,output,TileCursor{i,position,h,w,k,s,p,ow,channels,taps,inner}),equation) : TileEvidence<input,depth,Packing.tile_step(_)>
  tile(depth,U32.is_le((position % ow + 8 : U32),ow),U32.is_le((position % ow + 8 : U32),(ow + ow : U32)),input,output,i,position,h,w,k,s,p,ow,channels,taps,inner,input_evidence,balanced)

def tile_loop(+depth: Nat,count: Nat,-input: Array<F32>,-state: Packing.Tiled(),+input_evidence: PackingReadWitness.family(input),evidence: TileEvidence<input,depth,state>) ->
  TileEvidence<input,depth,Loop.run(~Packing.Tiled(),~Packing.tile_step,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: tile_loop(depth,rest,input,Packing.tile_step(state),input_evidence,tile_step(depth,input,state,input_evidence,evidence))

def finish(-depth: Nat,-input: Array<F32>,-state: Packing.Tiled(),evidence: TileEvidence<input,depth,state>) ->
  TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,Packing.finish(state)>:
  TileEvidence{output,cursor,equation,balanced} = evidence
  TileCursor{i,position,h,w,k,s,p,ow,channels,taps,inner} = cursor
  %Equal.sym(Packing.Tiled(),state,restore_tiles(input,output,TileCursor{i,position,h,w,k,s,p,ow,channels,taps,inner}),equation) :
    TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,Packing.finish(_)>
  TraversalWitness.ResultEvidence{output,{==},balanced}

def pack(+depth: Nat,+count: Nat,+start: U32,-input: Array<F32>,-output: Array<Tile.Tile8>,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+ow: U32,+inner: U32,
  +input_evidence: PackingReadWitness.family(input),balanced: Witness.Witness(Tile.Tile8,depth,output)) ->
  TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,Packing.pack(count,start,input,output,h,w,k,s,p,ow,inner)>:
  +cursor = {TileCursor{0,start,h,w,k,s,p,ow,U32.to_nat((inner / (k * k) : U32)),U32.to_nat(k),U32.to_nat(inner)} : TileCursor}
  finish(depth,input,Loop.run(~Packing.Tiled(),~Packing.tile_step,Nat.div(count,U32.to_nat(inner)),restore_tiles(input,output,cursor)),
    tile_loop(depth,Nat.div(count,U32.to_nat(inner)),input,restore_tiles(input,output,cursor),input_evidence,TileEvidence{output,cursor,{==},balanced}))
