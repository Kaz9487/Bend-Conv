# Pixel reads of an entry whose tile crosses an output row.
import Base
import ./kernel_tile.bend as Tile

def masked(value: F32, inside: Bool) -> F32:
  match inside:
    case True{}:
      value
    case False{}:
      0.0

def mask_pixel(inside: Bool, pair: Array<F32> & F32) -> Array<F32> & F32:
  (input_storage, value) = pair
  (input_storage, masked(value, inside))

def pixel(input_storage: Array<F32>, +output_position: U32, +reduction_index: U32, +input_height: U32, +input_width: U32, +kernel_size: U32, +stride: U32, +padding: U32, +output_width: U32) -> Array<F32> & F32:
  +input_row = ((output_position / output_width) * stride + (reduction_index / kernel_size) % kernel_size - padding : U32)
  +input_column = ((output_position % output_width) * stride + reduction_index % kernel_size - padding : U32)
  at = ((reduction_index / (kernel_size * kernel_size)) * input_height * input_width + input_row * input_width + input_column : U32)
  mask_pixel(U32.is_lt(input_row, input_height) && U32.is_lt(input_column, input_width), Array.get(F32, input_storage, at))

type PixelBlock is Data:
  PixelBlock{position: U32,kernel_row: U32,kernel_column: U32,channel_base: U32,height: U32,width: U32,stride: U32,padding: U32,output_width: U32}

# Reduction coordinates are invariant across the whole block, at any width.

def pixel_block(position: U32,+reduction: U32,+height: U32,+width: U32,+kernel: U32,stride: U32,padding: U32,output_width: U32) -> PixelBlock:
  PixelBlock{position,(reduction / kernel % kernel : U32),(reduction % kernel : U32),
    (reduction / (kernel * kernel) * height * width : U32),height,width,stride,padding,output_width}

type PixelAddress is Data:
  PixelAddress{index: U32,inside: Bool}

def pixel_address(context: PixelBlock,index: Nat) -> PixelAddress:
  PixelBlock{position,kernel_row,kernel_column,channel_base,+height,+width,+stride,+padding,+output_width} = context
  +position = (position + U32.from_nat(index) : U32)
  +row = (position / output_width * stride + kernel_row - padding : U32)
  +column = (position % output_width * stride + kernel_column - padding : U32)
  PixelAddress{(channel_base + row * width + column : U32),U32.is_lt(row,height) && U32.is_lt(column,width)}

def read_pixel_address(input: Array<F32>,address: PixelAddress) -> Array<F32> & F32:
  PixelAddress{index,inside} = address
  mask_pixel(inside,Array.get(F32,input,index))

def read_block_pixel(input: Array<F32>,context: PixelBlock,index: Nat) -> Array<F32> & F32:
  read_pixel_address(input,pixel_address(context,index))
