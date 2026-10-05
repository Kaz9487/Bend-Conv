import Base
import ../kernel_spatial_packing.bend as Packing
import ../kernel_packing_samples.bend as PackingSamples
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness
import ../kernel_tile.bend as Tile
import ./kernel_tile_witness.bend as TileWitness

def family(-input: Array<F32>) -> Data:
  Witness.ArrayWitness<F32,input>

def masked(-input: Array<F32>,+inside: Bool,-result: Array<F32> & F32,observed: Witness.Read<F32,input,result>) ->
  TraversalWitness.Sample<Array<F32>,F32,input,PackingSamples.mask_pixel(inside,result)>:
  Witness.Read{value,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,value),equation) : TraversalWitness.Sample<Array<F32>,F32,input,PackingSamples.mask_pixel(inside,_)>
  TraversalWitness.Sample{PackingSamples.masked(value,inside),{==}}

def masked_read(-input: Array<F32>,+inside: Bool,-result: Array<F32> & F32,observed: Witness.Read<F32,input,result>) ->
  TraversalWitness.Sample<Array<F32>,F32,input,Packing.masked(inside,result)>:
  Witness.Read{value,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,value),equation) : TraversalWitness.Sample<Array<F32>,F32,input,Packing.masked(inside,_)>
  TraversalWitness.Sample{PackingSamples.masked(value,inside),{==}}

def pixel(-input: Array<F32>,+row: U32,+column: U32,+channel: U32,+height: U32,+width: U32,evidence: family(input)) ->
  TraversalWitness.Sample<Array<F32>,F32,input,Packing.pixel(input,row,column,channel,height,width)>:
  +index = (channel + row * width + column : U32)
  masked_read(input,U32.is_lt(row,height) && U32.is_lt(column,width),Array.get(F32,input,index),Witness.observed(F32,input,index,evidence))

def row_read(-input: Array<F32>,context: Packing.PixelBlock,index: Nat,evidence: family(input)) ->
  TraversalWitness.Sample<Array<F32>,F32,input,Packing.read_block_pixel(input,context,index)>:
  Packing.PixelBlock{row,column,channel,height,width,stride} = context
  pixel(input,row,(column + U32.from_nat(index) * stride : U32),channel,height,width,evidence)

def row_block(-input: Array<F32>,+row: U32,+column: U32,+channel: U32,+height: U32,+width: U32,+stride: U32,+next_row: U32,+offset: U32,+output_width: U32,+kernel_column: U32,+padding: U32,+evidence: family(input)) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Packing.block(input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding)>:
  TileWitness.tail(~Array<F32>,~Packing.PixelBlock,~family,~Packing.read_block_pixel,~row_read,input,Packing.PixelBlock{row,column,channel,height,width,stride},
    Packing.pixel(input,row,column,channel,height,width),evidence,pixel(input,row,column,channel,height,width,evidence))

def wrap_pixel(same: Bool,-input: Array<F32>,+row: U32,+column: U32,+channel: U32,+height: U32,+width: U32,+stride: U32,+next_row: U32,+offset: U32,+output_width: U32,+kernel_column: U32,+padding: U32,+lane: U32,evidence: family(input)) ->
  TraversalWitness.Sample<Array<F32>,F32,input,Packing.wrap_pixel(same,input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding,lane)>:
  match same:
    case True{}: pixel(input,row,(column + lane * stride : U32),channel,height,width,evidence)
    case False{}: pixel(input,next_row,((offset + lane - output_width) * stride + kernel_column - padding : U32),channel,height,width,evidence)

def wrap_lane(-input: Array<F32>,+row: U32,+column: U32,+channel: U32,+height: U32,+width: U32,+stride: U32,+next_row: U32,+offset: U32,+output_width: U32,+kernel_column: U32,+padding: U32,+lane: U32,evidence: family(input)) ->
  TraversalWitness.Sample<Array<F32>,F32,input,Packing.wrap_lane(input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding,lane)>:
  wrap_pixel(U32.is_lt((offset + lane : U32),output_width),input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding,lane,evidence)

def wrap_read(-input: Array<F32>,context: Packing.WrapBlock,index: Nat,evidence: family(input)) ->
  TraversalWitness.Sample<Array<F32>,F32,input,Packing.read_wrap_pixel(input,context,index)>:
  Packing.WrapBlock{+row,+column,+channel,+height,+width,+stride,+next_row,+offset,+output_width,+kernel_column,+padding} = context
  wrap_lane(input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding,U32.from_nat(index),evidence)

def wrap_block(-input: Array<F32>,+row: U32,+column: U32,+channel: U32,+height: U32,+width: U32,+stride: U32,+next_row: U32,+offset: U32,+output_width: U32,+kernel_column: U32,+padding: U32,+evidence: family(input)) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Packing.wrap_block(input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding)>:
  TileWitness.tail(~Array<F32>,~Packing.WrapBlock,~family,~Packing.read_wrap_pixel,~wrap_read,input,Packing.WrapBlock{row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding},
    Packing.pixel(input,row,column,channel,height,width),evidence,pixel(input,row,column,channel,height,width,evidence))

def address(-input: Array<F32>,address: PackingSamples.PixelAddress,evidence: family(input)) ->
  TraversalWitness.Sample<Array<F32>,F32,input,PackingSamples.read_pixel_address(input,address)>:
  PackingSamples.PixelAddress{+index,inside} = address
  masked(input,inside,Array.get(F32,input,index),Witness.observed(F32,input,index,evidence))

def crossing_read(-input: Array<F32>,context: PackingSamples.PixelBlock,index: Nat,evidence: family(input)) ->
  TraversalWitness.Sample<Array<F32>,F32,input,PackingSamples.read_block_pixel(input,context,index)>:
  address(input,PackingSamples.pixel_address(context,index),evidence)

def first(-input: Array<F32>,+position: U32,+reduction: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_width: U32,evidence: family(input)) ->
  TraversalWitness.Sample<Array<F32>,F32,input,PackingSamples.pixel(input,position,reduction,height,width,kernel,stride,padding,output_width)>:
  +row = (position / output_width * stride + reduction / kernel % kernel - padding : U32)
  +column = (position % output_width * stride + reduction % kernel - padding : U32)
  +index = (reduction / (kernel * kernel) * height * width + row * width + column : U32)
  masked(input,U32.is_lt(row,height) && U32.is_lt(column,width),Array.get(F32,input,index),Witness.observed(F32,input,index,evidence))
