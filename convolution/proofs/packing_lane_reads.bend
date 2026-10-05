import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/kernel_packing_samples.bend as PackingSamples

def value(+x: Tree.Tree<F32>,+pos: U32,+q: U32,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32) -> F32:
  PackingSamples.masked(Tree.read(F32,x,
      ((q / (k * k)) * h * w + ((pos / output_width) * stride + (q / k) % k - pad : U32) * w + ((pos % output_width) * stride + q % k - pad : U32) : U32)),
    U32.is_lt(((pos / output_width) * stride + (q / k) % k - pad : U32),h) && U32.is_lt(((pos % output_width) * stride + q % k - pad : U32),w))

def pixel(+x: Tree.Tree<F32>,+pos: U32,+q: U32,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,
  +output_width: U32) -> {PackingSamples.pixel(Tree.pack(F32,x),pos,q,h,w,k,stride,pad,output_width) == (Tree.pack(F32,x),value(x,pos,q,h,w,k,stride,pad,
      output_width)) : Array<F32> & F32}:
  %Equal.sym(Array<F32> & F32,Array.get(F32,Tree.pack(F32,x),
      ((q / (k * k)) * h * w + ((pos / output_width) * stride + (q / k) % k - pad : U32) * w + ((pos % output_width) * stride + q % k - pad : U32) : U32)),
    (Tree.pack(F32,x),Tree.read(F32,x,
        ((q / (k * k)) * h * w + ((pos / output_width) * stride + (q / k) % k - pad : U32) * w + ((pos % output_width) * stride + q % k - pad : U32) : U32))),
    StorageLaws.read(F32,x,
      ((q / (k * k)) * h * w + ((pos / output_width) * stride + (q / k) % k - pad : U32) * w + ((pos % output_width) * stride + q % k - pad : U32) :
        U32))) : {PackingSamples.mask_pixel(U32.is_lt(((pos / output_width) * stride + (q / k) % k - pad : U32),h) && U32.is_lt(((pos % output_width) * stride + q % k - pad :
        U32),w),_) == (Tree.pack(F32,x),value(x,pos,q,h,w,k,stride,pad,output_width)) : Array<F32> & F32}
  {==}

def pack_input(input: Tree.Tree<F32>) -> Array<F32>:
  Tree.pack(F32,input)

def address_sample(input: Tree.Tree<F32>,address: PackingSamples.PixelAddress) -> F32:
  PackingSamples.PixelAddress{index,inside} = address
  PackingSamples.masked(Tree.read(F32,input,index),inside)

def block_sample(input: Tree.Tree<F32>,context: PackingSamples.PixelBlock,index: Nat) -> F32:
  address_sample(input,PackingSamples.pixel_address(context,index))

def native_address_read(+input: Tree.Tree<F32>,+address: PackingSamples.PixelAddress) ->
  {PackingSamples.read_pixel_address(pack_input(input),address) ==
    (pack_input(input),address_sample(input,address)) : Array<F32> & F32}:
  PackingSamples.PixelAddress{+index,+inside} = address
  %Equal.sym(Array<F32> & F32,Array.get(F32,pack_input(input),index),
    (pack_input(input),Tree.read(F32,input,index)),StorageLaws.read(F32,input,index)) :
    {PackingSamples.mask_pixel(inside,_) == (pack_input(input),PackingSamples.masked(Tree.read(F32,input,index),inside)) : Array<F32> & F32}
  {==}

# The cache itself computes the original address for every index, not only 8 lanes.
law prepared_pixel_matches_direct_read:
  for -input: Array<F32>
  for +position: U32
  for +reduction: U32
  for +height: U32
  for +width: U32
  for +kernel: U32
  for +stride: U32
  for +padding: U32
  for +output_width: U32
  for +index: Nat
  {PackingSamples.read_block_pixel(input,PackingSamples.pixel_block(position,reduction,height,width,kernel,stride,padding,output_width),index) ==
    PackingSamples.pixel(input,(position + U32.from_nat(index) : U32),reduction,height,width,kernel,stride,padding,output_width) : Array<F32> & F32}

def prepared_pixel_matches_direct_read(input,position,reduction,height,width,kernel,stride,padding,output_width,index):
  {==}

def native_block_read(+input: Tree.Tree<F32>,+context: PackingSamples.PixelBlock,+index: Nat) -> {PackingSamples.read_block_pixel(pack_input(input),context,
    index) == (pack_input(input),block_sample(input,context,index)) : Array<F32> & F32}:
  native_address_read(input,PackingSamples.pixel_address(context,index))
