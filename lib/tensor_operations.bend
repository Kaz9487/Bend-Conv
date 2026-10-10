# Checked operations preserve all owners and their balanced-storage certificates.
# Element functions enter hot loops only as direct templates.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./traversal_array.bend as Traversal
import ./tensor_access.bend as Access
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/traversal_certificate.bend as TraversalCertificate
import ./proofs/traversal_witness.bend as TraversalWitness
import ./proofs/tensor_access_certificate.bend as AccessCertificate
import ./traversal_lanes.bend as Lanes
import ./proofs/traversal_lanes_certificate.bend as LanesCertificate

type Outcome<-Value: Type> is Type:
  Accepted{value: Value}
  Rejected{value: Value}

def valid_buffer(+length: Nat,capacity: U32,view: Views.View) -> Bool:
  Nat.is_le(length,U32.to_nat(capacity)) && Views.fits(view,U32.from_nat(length))

def pair_buffers(~Element: Data,input_length: Nat,output_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})

def copy_when(~Element: Data,valid: Bool,count: Maybe<&2,U32>,+source: Views.View,target: Views.View,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match valid count:
    case True{} Some{+count}:
      Views.View{axes,+offset} = target
      Storage.Buffer{input,input_length,input_certificate} = input
      Storage.Buffer{output,output_length,output_certificate} = output
      +contiguous = Views.is_contiguous(source)
      Accepted{pair_buffers(~Element,input_length,output_length,
        Access.copy_run(~Element,contiguous,U32.to_nat(count),source,offset,input,output),
        AccessCertificate.copy(~Element,contiguous,U32.to_nat(count),source,offset,input,output,input_certificate,output_certificate))}
    case valid count: Rejected{(input,output)}

def copy(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,
  output: Storage.Buffer<Element>,+target: Views.View) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_depth,input_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_depth,output_equation} = output_certificate
  copy_when(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(target) &&
    valid_buffer(input_length,U32.shln(1,input_depth),source) &&
    valid_buffer(output_length,U32.shln(1,output_depth),target),Views.numel(target),source,target,
    Storage.Buffer{input,input_length,Certificate.Certificate{input_depth,input_equation}},Storage.Buffer{output,output_length,Certificate.Certificate{output_depth,output_equation}})

# output[address of element i of target] = input[offset of source + i]: a
# contiguous source written through a strided target view, in row-major order.
def scatter_when(~Element: Data,valid: Bool,count: Maybe<&2,U32>,source: Views.View,+target: Views.View,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match valid count:
    case True{} Some{+count}:
      Views.View{axes,+offset} = source
      Storage.Buffer{input,input_length,input_certificate} = input
      Storage.Buffer{output,output_length,output_certificate} = output
      Accepted{pair_buffers(~Element,input_length,output_length,
        Access.scatter_run(~Element,U32.to_nat(count),target,offset,input,output),
        AccessCertificate.scatter(~Element,U32.to_nat(count),target,offset,input,output,input_certificate,output_certificate))}
    case valid count: Rejected{(input,output)}

def scatter(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,
  output: Storage.Buffer<Element>,+target: Views.View) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_depth,input_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_depth,output_equation} = output_certificate
  scatter_when(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(source) &&
    valid_buffer(input_length,U32.shln(1,input_depth),source) &&
    valid_buffer(output_length,U32.shln(1,output_depth),target),Views.numel(target),source,target,
    Storage.Buffer{input,input_length,Certificate.Certificate{input_depth,input_equation}},Storage.Buffer{output,output_length,Certificate.Certificate{output_depth,output_equation}})

def combine_buffers(~Element: Data,left_length: Nat,right_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  (right,left) = result
  TraversalCertificate.Certificates{right_certificate,left_certificate} = certificates
  (Storage.Buffer{left,left_length,left_certificate},Storage.Buffer{right,right_length,right_certificate})

def combine_kernel(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+right_view: Views.View,+offset: U32,
  left: Storage.Buffer<Element>,right: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Storage.Buffer{left,left_length,left_certificate} = left
  Storage.Buffer{right,right_length,right_certificate} = right
  +fast = Access.zero_contiguous(right_view) && U32.is_zero(offset)
  combine_buffers(~Element,left_length,right_length,
    Access.combine_run(~Element,~operation,fast,count,right_view,offset,right,left),
    AccessCertificate.combine(~Element,~operation,fast,count,right_view,offset,right,left,right_certificate,left_certificate))

def combine_when_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,
  valid: Bool,count: Maybe<&2,U32>,right_view: Views.View,target: Views.View,left: Storage.Buffer<Element>,right: Storage.Buffer<Element>) ->
  Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match valid count:
    case True{} Some{count}:
      Views.View{axes,offset} = target
      Accepted{kernel(U32.to_nat(count),right_view,offset,left,right)}
    case valid count: Rejected{(left,right)}

def combine_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,
  left: Storage.Buffer<Element>,+left_view: Views.View,right: Storage.Buffer<Element>,+right_view: Views.View) ->
  Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Storage.Buffer{left,+left_length,left_certificate} = left
  Certificate.Certificate{+left_depth,left_equation} = left_certificate
  Storage.Buffer{right,+right_length,right_certificate} = right
  Certificate.Certificate{+right_depth,right_equation} = right_certificate
  combine_when_with(~Element,~kernel,Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) &&
    valid_buffer(left_length,U32.shln(1,left_depth),left_view) &&
    valid_buffer(right_length,U32.shln(1,right_depth),right_view),Views.numel(left_view),right_view,left_view,
    Storage.Buffer{left,left_length,Certificate.Certificate{left_depth,left_equation}},Storage.Buffer{right,right_length,Certificate.Certificate{right_depth,right_equation}})

def combine(~Element: Data,~operation: Element -> Element -> Element,
  left: Storage.Buffer<Element>,+left_view: Views.View,right: Storage.Buffer<Element>,+right_view: Views.View) ->
  Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  combine_with(~Element,~combine_kernel(~Element,~operation),left,left_view,right,right_view)

# A mapper is the ordered map of count values from position on a certified
# Buffer. mapped applies the operation to one value at a time.
def mapped(~Element: Data,~operation: Element -> Element,+count: Nat,+position: U32,buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Storage.Buffer{storage,length,certificate} = buffer
  Storage.Buffer{Traversal.map_in_place(~Element,~Unit,~Access.apply_element(~Element,~operation),count,position,Unit{},storage),length,
    TraversalCertificate.map_in_place(~Element,~Unit,~Access.apply_element(~Element,~operation),count,position,Unit{},storage,certificate)}

# The same map with the values of each round of eight computed together: an
# operation written with FP32 arithmetic alone then compiles to vector code.
# The certificate's depth decides whether the array holds the positions.
def lanes(~Context: Data,~operation: Context -> F32 -> F32,+count: Nat,+position: U32,+context: Context,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Storage.Buffer{storage,length,Certificate.Certificate{+depth,equation}} = buffer
  Storage.Buffer{Lanes.map_when(~Context,~operation,Lanes.covered(depth,position,count),count,position,context,storage),length,
    LanesCertificate.map_when(~Context,~operation,Lanes.covered(depth,position,count),count,position,context,storage,Certificate.Certificate{depth,equation})}

def mapped_lanes(~operation: F32 -> F32,+count: Nat,+position: U32,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  lanes(~Unit,~Access.apply_element(~F32,~operation),count,position,Unit{},buffer)

def apply_context_when(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,valid: Bool,count: Maybe<&2,U32>,view: Views.View,
  buffer: Storage.Buffer<Element>) -> Outcome<Storage.Buffer<Element>>:
  match valid count:
    case True{} Some{+count}:
      Views.View{axes,+offset} = view
      Accepted{mapper(U32.to_nat(count),offset,context,buffer)}
    case valid count: Rejected{buffer}

def apply_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,buffer: Storage.Buffer<Element>,+view: Views.View) -> Outcome<Storage.Buffer<Element>>:
  Storage.Buffer{storage,+length,certificate} = buffer
  Certificate.Certificate{+depth,equation} = certificate
  apply_context_when(~Element,~Context,~mapper,context,Views.is_contiguous(view) && valid_buffer(length,U32.shln(1,depth),view),Views.numel(view),view,
    Storage.Buffer{storage,length,Certificate.Certificate{depth,equation}})

# A mapper without a context, as the Unit-context case.
def without_context(~Element: Data,~mapper: @+count: Nat -> @+position: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element>,+count: Nat,+position: U32,+context: Unit,buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  mapper(count,position,buffer)

def apply_indexed_when(~Element: Data,~operation: U32 -> Element -> Element,valid: Bool,count: Maybe<&2,U32>,+view: Views.View,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match valid count:
    case True{} Some{+count}:
      Storage.Buffer{input,input_length,input_certificate} = input
      Storage.Buffer{output,output_length,output_certificate} = output
      Accepted{pair_buffers(~Element,input_length,output_length,
        Access.apply_indexed_run(~Element,~operation,U32.to_nat(count),view,input,output),
        AccessCertificate.apply_indexed(~Element,~operation,U32.to_nat(count),view,input,output,input_certificate,output_certificate))}
    case valid count: Rejected{(input,output)}

def holds(count: Maybe<&2,U32>,+length: Nat,capacity: U32) -> Bool:
  match count:
    case None{}: False{}
    case Some{count}: Nat.is_le(U32.to_nat(count),length) && Nat.is_le(length,U32.to_nat(capacity))

# Writes f(index, element) for a contiguous input view into output[0..count).
def apply_indexed(~Element: Data,~operation: U32 -> Element -> Element,input: Storage.Buffer<Element>,+view: Views.View,
  output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_depth,input_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_depth,output_equation} = output_certificate
  apply_indexed_when(~Element,~operation,Views.is_contiguous(view) && valid_buffer(input_length,U32.shln(1,input_depth),view) &&
    holds(Views.numel(view),output_length,U32.shln(1,output_depth)),Views.numel(view),view,
    Storage.Buffer{input,input_length,Certificate.Certificate{input_depth,input_equation}},Storage.Buffer{output,output_length,Certificate.Certificate{output_depth,output_equation}})

def small(value: U32) -> Bool:
  U32.is_le(value,65535)

# The window geometry is exact and every sample address stays in U32.
def pool_geometry(+pool: Access.Pool) -> Bool:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  small(height) && small(width) && small(kernel) && small(stride) && small(padding) && U32.is_lt(0,kernel) && U32.is_lt(0,stride) &&
    U32.is_le(kernel,(height + 2 * padding : U32)) && U32.is_le(kernel,(width + 2 * padding : U32)) &&
    U32.is_eq(out_height,((height + 2 * padding - kernel : U32) / stride + 1 : U32)) &&
    U32.is_eq(out_width,((width + 2 * padding - kernel : U32) / stride + 1 : U32))

# The unchecked sweep of two owners; the callers establish its bounds.
def swept(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+context: Access.SweepContext<Element>,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{output,output_length,output_certificate} = output
  pair_buffers(~Element,input_length,output_length,
    Access.sweep_run(~Element,~operation,count,context,input,output),
    AccessCertificate.sweep(~Element,~operation,count,context,input,output,input_certificate,output_certificate))

def sweep_when(~Element: Data,~operation: Element -> Element -> Element,valid: Bool,count: Maybe<&2,U32>,+context: Access.SweepContext<Element>,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match valid count:
    case True{} Some{count}: Accepted{swept(~Element,~operation,U32.to_nat(count),context,input,output)}
    case valid count: Rejected{(input,output)}

# The window geometry of a whole axis is exact and every sample address stays in U32.
def sweep_geometry(+sweep: Access.Sweep) -> Bool:
  Access.Sweep{+extent,+step,+kernel,+stride,+padding,+out_extent,+origin} = sweep
  small(extent) && small(kernel) && small(stride) && small(padding) && U32.is_lt(0,kernel) && U32.is_lt(0,stride) && U32.is_lt(0,step) &&
    U32.is_eq(origin,0) && U32.is_le(kernel,(extent + 2 * padding : U32)) &&
    U32.is_eq(out_extent,((extent + 2 * padding - kernel : U32) / stride + 1 : U32))

def sweep_input(+sweep: Access.Sweep,+outer: U32) -> Maybe<&2,U32>:
  Access.Sweep{+extent,+step,kernel,stride,padding,out_extent,origin} = sweep
  Shapes.product([outer,extent,step],Some{1})

def sweep_output(+sweep: Access.Sweep,+outer: U32) -> Maybe<&2,U32>:
  Access.Sweep{extent,+step,kernel,stride,padding,+out_extent,origin} = sweep
  Shapes.product([outer,out_extent,step],Some{1})

# Windows along one axis of outer contiguous blocks; see Access.Sweep.
def sweep(~Element: Data,~operation: Element -> Element -> Element,+pad: Element,input: Storage.Buffer<Element>,+sweep: Access.Sweep,+outer: U32,
  output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_depth,input_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_depth,output_equation} = output_certificate
  sweep_when(~Element,~operation,sweep_geometry(sweep) && holds(sweep_input(sweep,outer),input_length,U32.shln(1,input_depth)) &&
    holds(sweep_output(sweep,outer),output_length,U32.shln(1,output_depth)),sweep_output(sweep,outer),Access.SweepContext{sweep,pad},
    Storage.Buffer{input,input_length,Certificate.Certificate{input_depth,input_equation}},Storage.Buffer{output,output_length,Certificate.Certificate{output_depth,output_equation}})

# The input and the output of a pool around its column-swept planes.
def pool_finished(~Element: Data,input: Storage.Buffer<Element>,result: Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match result:
    case Accepted{(middle,output)}: Accepted{(input,output)}
    case Rejected{(middle,output)}: Rejected{(input,output)}

def pool_rows(~Element: Data,~operation: Element -> Element -> Element,+pad: Element,result: Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,
  +rows: Access.Sweep,+channels: U32,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match result:
    case Accepted{(input,middle)}: pool_finished(~Element,input,sweep(~Element,~operation,pad,middle,rows,channels,output))
    case Rejected{(input,middle)}: Rejected{(input,output)}

def pool_lines(~Element: Data,~operation: Element -> Element -> Element,+pad: Element,lines: Maybe<&2,U32>,input: Storage.Buffer<Element>,+pool: Access.Pool,
  +channels: U32,middle: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match lines:
    case None{}: Rejected{(input,output)}
    case Some{+lines}:
      pool_rows(~Element,~operation,pad,sweep(~Element,~operation,pad,input,Access.pool_columns(pool),lines,middle),Access.pool_rows(pool,0),channels,output)

# Windows over the last two axes of channels contiguous planes: a sweep along
# the columns of every row into middle, [channels,height,out_width], then a
# sweep along the rows into output. Each sweep checks its own geometry and
# counts.
def pool(~Element: Data,~operation: Element -> Element -> Element,+pad: Element,input: Storage.Buffer<Element>,+pool: Access.Pool,+channels: U32,
  middle: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  pool_lines(~Element,~operation,pad,Shapes.product([channels,height],Some{1}),input,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},channels,middle,output)

def reduce_buffer(~Element: Data,~Accumulator: Data,length: Nat,result: Array<Element> & Accumulator,
  certificate: Certificate.Certificate<Element,TraversalWitness.first(~Array<Element>,~Accumulator,result)>) -> Storage.Buffer<Element> & Accumulator:
  (storage,value) = result
  (Storage.Buffer{storage,length,certificate},value)

def reduce_when(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,
  valid: Bool,count: Maybe<&2,U32>,+view: Views.View,buffer: Storage.Buffer<Element>,initial: Accumulator) ->
  Outcome<(Storage.Buffer<Element> & Accumulator)>:
  match valid count:
    case True{} Some{+count}:
      Storage.Buffer{storage,length,certificate} = buffer
      +fast = Views.is_contiguous(view)
      Accepted{reduce_buffer(~Element,~Accumulator,length,
        Access.reduce_run(~Element,~Accumulator,~operation,fast,U32.to_nat(count),view,storage,initial),
        AccessCertificate.reduce(~Element,~Accumulator,~operation,fast,U32.to_nat(count),view,storage,initial,certificate))}
    case valid count: Rejected{(buffer,initial)}

def reduce(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,
  buffer: Storage.Buffer<Element>,+view: Views.View,initial: Accumulator) -> Outcome<(Storage.Buffer<Element> & Accumulator)>:
  Storage.Buffer{storage,+length,certificate} = buffer
  Certificate.Certificate{+depth,equation} = certificate
  reduce_when(~Element,~Accumulator,~operation,valid_buffer(length,U32.shln(1,depth),view),Views.numel(view),view,
    Storage.Buffer{storage,length,Certificate.Certificate{depth,equation}},initial)
