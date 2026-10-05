# Each band leaf assembles the rows its outputs read into one contiguous
# window and runs the two sweeps of the dense pool on it: along the columns of
# every window row, then along the rows, with output row zero's window
# starting at the window row that the image row would. Sample order is the
# dense order.
import Base
import ./tensor_access.bend as Access
import ./tensor_operations.bend as Operations
import ./tensor_band.bend as Bands
import ./tensor_band_window.bend as BandWindow
import ./traversal_partition.bend as Partition
import ./storage_buffer.bend as Storage
import ./tensor.bend as Tensors
import ./tensor_band_schedule.bend as BandSchedule
import ./kernel_cost.bend as Cost
import ./default_kernel_cost.bend as DefaultCost

type Context<-Element: Data> is Data:
  Context{geometry: BandWindow.Geometry,pad: Element,fill: Element,workers: U32}

def window_size(window: BandWindow.RowWindow) -> BandSchedule.Size:
  BandWindow.RowWindow{BandWindow.Rows{first,+count},above,below,rows} = window
  BandSchedule.Size{1,count,count}

def plan_size(layout: Bands.Plan,+window: BandWindow.Geometry) -> BandSchedule.Size:
  match layout:
    case Bands.Rows{first,rows}: window_size(BandWindow.window(window,first,rows))
    case Bands.Split{left,right}: BandSchedule.merge(plan_size(left,window),plan_size(right,window))

# Bytes price loading, storing and the loop that moves them (copy_byte_ns is
# measured on copy loops); operations count the rest. Per sample
# (Access.read_line and the operation): an offset and a subtraction, one bound
# check, a two-operation index, the padding select and the compare-and-select
# maximum: 8. Each line adds ten for its origin (Access.line_at). An output is
# one line along the rows; the lines along the columns are one for every
# window row, stride for every output row.
def parallel(window: BandWindow.Geometry,workers: U32,size: BandSchedule.Size) -> Bool:
  BandWindow.Geometry{Bands.Geometry{batch,channels,ih,iw},kernel,stride,padding,oh,ow} = window
  +config = DefaultCost.configuration()
  +samples = U32.to_f32(kernel)
  +lines = (U32.to_f32(stride) + 1.0 : F32)
  +line_bytes = (samples * 4.0 + 8.0 : F32)
  +line_operations = (samples * 8.0 + 10.0 : F32)
  BandSchedule.sized(Bands.Geometry{batch,channels,oh,ow},workers,Cost.element_ns((lines * line_bytes : F32),(lines * line_operations : F32),config),config,size)

# The destination of the two sweeps of channels stored planes: along the
# columns into middle, [channels,height,out_width], then along the rows from
# origin into output.
def pooled(~Element: Data,~operation: Element -> Element -> Element,+pool: Access.Pool,+pad: Element,origin: U32,+channels: U32,
  input: Storage.Buffer<Element>,middle: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  Bands.destination(~Element,Operations.swept(~Element,~operation,U32.to_nat((channels * out_height * out_width : U32)),
    Access.SweepContext{Access.pool_rows(pool,origin),pad},
    Bands.destination(~Element,Operations.swept(~Element,~operation,U32.to_nat((channels * height * out_width : U32)),
      Access.SweepContext{Access.pool_columns(pool),pad},input,middle)),output))

def windowed(~Element: Data,~operation: Element -> Element -> Element,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+first: U32,+rows: U32,
  +rows_read: BandWindow.RowWindow,core: Storage.Buffer<Element>,above: Storage.Buffer<Element>,below: Storage.Buffer<Element>) -> Bands.Bands(~Element):
  BandWindow.Geometry{Bands.Geometry{batch,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},+above_range,+below_range,+window} = rows_read
  BandWindow.Rows{+window_first,+window_rows} = window
  Partition.Leaf{pooled(~Element,~operation,Access.Pool{window_rows,width,kernel,stride,padding,output_rows,output_width},pad,
    ((output_first - first / height * output_height) * stride + first / height * height - window_first : U32),channels,
    BandWindow.assemble(~Element,fill,BandWindow.input_geometry(geometry),window,BandWindow.Rows{first,rows},above_range,below_range,core,above,below),
    Tensors.buffer(~Element,fill,(channels * window_rows * output_width : U32)),
    Tensors.buffer(~Element,fill,(channels * output_rows * output_width : U32))),output_first,output_rows,Unit{}}

def leaf(~Element: Data,~operation: Element -> Element -> Element,context: Context<Element>,first: U32,rows: U32,core: Storage.Buffer<Element>,halo: BandWindow.Halo<Element>) -> Bands.Bands(~Element):
  Context{+geometry,+pad,+fill,workers} = context
  BandWindow.Halo{above,below,+rows_read} = halo
  +first = first
  +rows = rows
  windowed(~Element,~operation,geometry,pad,fill,first,rows,rows_read,core,above,below)

def prepared(~Element: Data,~operation: Element -> Element -> Element,+context: Context<Element>,+layout: Bands.Plan,owners: Bands.Bands(~Element) & BandWindow.Halos(~Element)) -> Bands.Bands(~Element):
  Context{+window,pad,fill,workers} = context
  (bands,halos) = owners
  +geometry = BandWindow.input_geometry(window)
  +enabled = parallel(window,workers,plan_size(layout,window))
  BandWindow.apply(~Element,~Context<Element>,~leaf(~Element,~operation),layout,enabled && Bands.same_image(geometry,layout),enabled,geometry,context,bands,halos)

def planned(~Element: Data,~operation: Element -> Element -> Element,+context: Context<Element>,owners: Bands.Bands(~Element) & Bands.Plan) -> Bands.Bands(~Element):
  Context{window,pad,fill,workers} = context
  (bands,+layout) = owners
  prepared(~Element,~operation,context,layout,BandWindow.prepare(~Element,window,pad,layout,bands))

def execute(~Element: Data,~operation: Element -> Element -> Element,context: Context<Element>,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  planned(~Element,~operation,context,Bands.plan(~Element,bands))
