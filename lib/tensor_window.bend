# Two-dimensional pooling over the last two axes, as two ordered sweeps. Along
# the columns, each row of the window folds its k samples in increasing kx
# from pad: line = op(...op(op(pad, x0), x1)..., x(k-1)). Along the rows, the
# output folds the k lines in increasing ky from pad. A sample outside the
# input reads pad, so pad should be the identity of the operation.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_shape.bend as Shapes
import ./tensor_access.bend as Access
import ./tensor_operations.bend as Operations
import ./tensor.bend as Tensors
import ./tensor_layout.bend as Layout
import ./tensor_band.bend as Bands
import ./tensor_band_window.bend as BandWindow
import ./tensor_band_pool.bend as BandPool

def out_extent(+extent: U32,+kernel: U32,+stride: U32,+padding: U32) -> U32:
  ((extent + 2 * padding - kernel : U32) / U32.max(stride,1) + 1 : U32)

def geometry(+shape: List<&2,U32>,+kernel: U32,+stride: U32,+padding: U32) -> Access.Pool:
  +rank = List.length(&2,U32,shape)
  +height = Layout.extent_at(shape,Nat.sub(rank,2n))
  +width = Layout.extent_at(shape,Nat.sub(rank,1n))
  Access.Pool{height,width,kernel,stride,padding,out_extent(height,kernel,stride,padding),out_extent(width,kernel,stride,padding)}

def leading(ranked: Bool,+shape: List<&2,U32>) -> Maybe<&2,U32>:
  match ranked:
    case True{}: Shapes.product(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),Some{1})
    case False{}: None{}

# The number of planes before the last two axes; none below rank two.
def planes(+shape: List<&2,U32>) -> Maybe<&2,U32>:
  leading(Nat.is_le(2n,List.length(&2,U32,shape)),shape)

def pooled(~Element: Data,name: String,+shape: List<&2,U32>,+out_shape: List<&2,U32>,fill: Element,+workers: U32,
  result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Tensors.Tensor<Element>:
  match result:
    case Operations.Accepted{(input,output)}: Tensors.Tensor{output,out_shape,fill,Tensors.Usable{},workers}
    case Operations.Rejected{(input,output)}: Tensors.Tensor{output,out_shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.InvalidArgument{shape}}},workers}

def pool_middle(~Element: Data,~operation: Element -> Element -> Element,name: String,+pad: Element,middle: Maybe<&2,U32>,count: U32,+channels: U32,+pool: Access.Pool,
  storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+out_shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match middle:
    case Some{middle}:
      pooled(~Element,name,shape,out_shape,fill,workers,
        Operations.pool(~Element,~operation,pad,storage,pool,channels,Tensors.buffer(~Element,fill,middle),Tensors.buffer(~Element,fill,count)))
    case None{}: Tensors.failed(~Element,name,Tensors.InvalidArgument{shape},shape,fill,workers)

# The column-swept planes keep the rows of the input and have the columns of
# the output.
def middle_shape(+shape: List<&2,U32>,pool: Access.Pool) -> List<&2,U32>:
  Access.Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[height,out_width])

def pool_counts(~Element: Data,~operation: Element -> Element -> Element,name: String,+pad: Element,channels: Maybe<&2,U32>,count: Maybe<&2,U32>,+pool: Access.Pool,
  storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+out_shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match channels count:
    case Some{+channels} Some{count}:
      pool_middle(~Element,~operation,name,pad,Tensors.count_of(middle_shape(shape,pool)),count,channels,pool,storage,shape,out_shape,fill,workers)
    case channels count: Tensors.failed(~Element,name,Tensors.InvalidArgument{shape},shape,fill,workers)

def pool_planes(~Element: Data,~operation: Element -> Element -> Element,name: String,+pad: Element,channels: Maybe<&2,U32>,+pool: Access.Pool,
  storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  Access.Pool{height,width,kernel,stride,padding,+out_height,+out_width} = pool
  +out_shape = List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[out_height,out_width])
  pool_counts(~Element,~operation,name,pad,channels,Tensors.count_of(out_shape),pool,storage,shape,out_shape,fill,workers)

def pool_status(~Element: Data,~operation: Element -> Element -> Element,name: String,status: Tensors.Status,+pad: Element,+kernel: U32,+stride: U32,+padding: U32,
  storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match status:
    case Tensors.Usable{}:
      pool_planes(~Element,~operation,name,pad,planes(shape),geometry(shape,kernel,stride,padding),storage,shape,fill,workers)
    case Tensors.Failed{problem}: Tensors.Tensor{storage,shape,fill,Tensors.Failed{problem},workers}

# pad is both the value outside the input and the initial accumulator.
def band_shape(shape: List<&2,U32>,geometry: Bands.Geometry) -> Bool:
  match shape geometry:
    case [c,h,w] Bands.Geometry{n,ci,hi,wi}: U32.is_eq(n,1) && U32.is_eq(c,ci) && U32.is_eq(h,hi) && U32.is_eq(w,wi)
    case [n,c,h,w] Bands.Geometry{ni,ci,hi,wi}: U32.is_eq(n,ni) && U32.is_eq(c,ci) && U32.is_eq(h,hi) && U32.is_eq(w,wi)
    case shape geometry: False{}

def band_execute(~Element: Data,~operation: Element -> Element -> Element,+pad: Element,pool: Access.Pool,+geometry: Bands.Geometry,bands: Bands.Bands(~Element),
  out_shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match pool geometry:
    case Access.Pool{height,width,kernel,stride,padding,+oh,+ow} Bands.Geometry{batch,channels,ih,iw}:
      Tensors.Banded{BandPool.execute(~Element,~operation,BandPool.Context{BandWindow.Geometry{geometry,kernel,stride,padding,oh,ow},pad,fill,workers},bands),
        Bands.Geometry{batch,channels,oh,ow},out_shape,fill,Tensors.Usable{},workers}

# The band route needs the same counts as the dense route: the planes swept
# along the columns and the output.
def band_count(~Element: Data,~operation: Element -> Element -> Element,name: String,valid: Bool,planes: Maybe<&2,U32>,count: Maybe<&2,U32>,+pad: Element,+pool: Access.Pool,+geometry: Bands.Geometry,bands: Bands.Bands(~Element),
  +shape: List<&2,U32>,out_shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match valid planes count pool:
    case True{} Some{planes} Some{count} pool: band_execute(~Element,~operation,pad,pool,geometry,bands,out_shape,fill,workers)
    case valid planes count Access.Pool{height,width,kernel,stride,padding,oh,ow}:
      Tensors.with_dense(~Element,~Tensors.Tensor<Element>,Tensors.Banded{bands,geometry,shape,fill,Tensors.Usable{},workers},storage => shape => fill => status => workers =>
        pool_status(~Element,~operation,name,status,pad,kernel,stride,padding,storage,shape,fill,workers))

def band_geometry(~Element: Data,~operation: Element -> Element -> Element,name: String,pad: Element,+pool: Access.Pool,+geometry: Bands.Geometry,bands: Bands.Bands(~Element),
  +shape: List<&2,U32>,fill: Element,workers: U32) -> Tensors.Tensor<Element>:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  +out_shape = List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[out_height,out_width])
  band_count(~Element,~operation,name,band_shape(shape,geometry) && Operations.pool_geometry(pool) && U32.is_lt(0,height) && U32.is_lt(0,width) &&
    Bool.not(Shapes.has_zero(out_shape)),Tensors.count_of(middle_shape(shape,pool)),Tensors.count_of(out_shape),pad,pool,geometry,bands,shape,out_shape,fill,workers)

def band_status(~Element: Data,~operation: Element -> Element -> Element,name: String,status: Tensors.Status,pad: Element,+kernel: U32,+stride: U32,+padding: U32,
  layout: Bands.Geometry,bands: Bands.Bands(~Element),+shape: List<&2,U32>,fill: Element,workers: U32) -> Tensors.Tensor<Element>:
  match status:
    case Tensors.Failed{+problem}:
      Tensors.with_dense(~Element,~Tensors.Tensor<Element>,Tensors.Banded{bands,layout,shape,fill,Tensors.Failed{problem},workers},
        storage => shape => fill => status => workers =>
          pool_status(~Element,~operation,name,status,pad,kernel,stride,padding,storage,shape,fill,workers))
    case Tensors.Usable{}: band_geometry(~Element,~operation,name,pad,geometry(shape,kernel,stride,padding),layout,bands,shape,fill,workers)

def pool2d(~Element: Data,~operation: Element -> Element -> Element,name: String,+pad: Element,tensor: Tensors.Tensor<Element>,
  +kernel: U32,+stride: U32,+padding: U32) -> Tensors.Tensor<Element>:
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,workers}: pool_status(~Element,~operation,name,status,pad,kernel,stride,padding,storage,shape,fill,workers)
    case Tensors.Banded{bands,layout,shape,fill,status,workers}: band_status(~Element,~operation,name,status,pad,kernel,stride,padding,layout,bands,shape,fill,workers)
