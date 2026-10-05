# Window geometry is independent of convolution arithmetic. Each output row
# belongs to one existing core band; neighbours contribute only halo rows.
import Base
import ./tensor_band.bend as Bands
import ./storage_buffer.bend as Storage
import ./traversal_partition.bend as Partition

type Geometry is Data:
  Geometry{input: Bands.Geometry,kernel: U32,stride: U32,padding: U32,output_height: U32,output_width: U32}

type Rows is Data:
  Rows{first: U32,count: U32}

# A leaf's output rows, its halo rows above and below the core, and the
# image rows its window holds: from the stride-aligned first row its outputs
# read to the last, so the window packs as a shorter image.
type RowWindow is Data:
  RowWindow{output: Rows,above: Rows,below: Rows,window: Rows}

def ceiling(+value: U32,+stride: U32) -> U32:
  (value / stride + Bool.pick(U32,U32.is_zero((value % stride : U32)),0,1) : U32)

def subtract_clipped(+value: U32,+amount: U32) -> U32:
  (U32.max(value,amount) - amount : U32)

def clipped(+first: U32,+end: U32) -> Rows:
  Rows{first,subtract_clipped(end,first)}

def nonempty(active: Bool,window: RowWindow) -> RowWindow:
  match active window:
    case True{} window: window
    case False{} RowWindow{output,Rows{above_first,above_count},Rows{below_first,below_count},Rows{window_first,window_count}}:
      RowWindow{output,Rows{above_first,0},Rows{below_first,0},Rows{window_first,0}}

# The first row a window holds: the first row its outputs read, moved down
# to a stride multiple.
def start(+height: U32,+output_first: U32,+stride: U32,+padding: U32) -> U32:
  +needed = U32.min(height,subtract_clipped((output_first * stride : U32),padding))
  subtract_clipped(needed,(needed % stride : U32))

# Output partition boundaries follow the input cores, with the first and last
# cores also owning any output rows introduced by padding. The halos hold
# every window row outside the core, so a window is its image rows exactly.
# Caller geometry checks establish positive stride/height and bounded products.
def window(geometry: Geometry,+first: U32,+count: U32) -> RowWindow:
  Geometry{Bands.Geometry{batch,channels,+height,width},+kernel,+stride,+padding,+output_height,output_width} = geometry
  +image = (first / height : U32)
  +core_first = (first % height : U32)
  +core_end = (core_first + count : U32)
  +out_first = U32.min(output_height,ceiling(core_first,stride))
  +out_end = Bool.pick(U32,U32.is_eq(core_end,height),output_height,U32.min(output_height,ceiling(core_end,stride)))
  +needed_first = start(height,out_first,stride,padding)
  +needed_end = U32.min(height,subtract_clipped((subtract_clipped(out_end,1) * stride + kernel : U32),padding))
  +above_first = (image * height + needed_first : U32)
  +above_end = (image * height + U32.min(core_first,needed_end) : U32)
  +below_first = (image * height + U32.max(core_end,needed_first) : U32)
  +below_end = (image * height + needed_end : U32)
  nonempty(U32.is_lt(out_first,out_end),RowWindow{Rows{(image * output_height + out_first : U32),subtract_clipped(out_end,out_first)},
    clipped(above_first,above_end),clipped(below_first,below_end),clipped(above_first,below_end)})

type Halo<-Element: Data> is Type:
  Halo{above: Storage.Buffer<Element>,below: Storage.Buffer<Element>,window: RowWindow}

def Halos(~Element: Data) -> Type:
  Partition.Tree<&1,Halo<Element>,Unit>

def extract(~Element: Data,+geometry: Bands.Geometry,fill: Element,range: Rows,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element) & Storage.Buffer<Element>:
  Rows{+first,+count} = range
  Bands.range(~Element,geometry,first,count,bands,Bands.allocate(~Element,fill,Bands.local_count(geometry,count)))

def halo(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+count: U32,window: RowWindow,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element) & Halos(~Element):
  RowWindow{+output,+above,+below,+rows} = window
  Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Bands.Bands(~Element) & Halos(~Element)),
    extract(~Element,geometry,fill,above,bands),bands => upper =>
      Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Bands.Bands(~Element) & Halos(~Element)),
        extract(~Element,geometry,fill,below,bands),bands => lower =>
          (bands,Partition.Leaf{Halo{upper,lower,RowWindow{output,above,below,rows}},first,count,Unit{}})))

def copied_window(~Element: Data,owners: Bands.Bands(~Element) & Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  (sources,window) = owners
  window

# The window rows of every channel, stored contiguously. A window equal to
# the core (for example a 1x1 stride-1 operation) is the core itself; any
# other window is the shared range copy of its rows from the upper halo, the
# core and the lower halo, one contiguous range per channel per source. The
# three copies are direct calls: a leaf task must not reach a closure.
def assembled(~Element: Data,exact: Bool,+fill: Element,+geometry: Bands.Geometry,+window: Rows,+core: Rows,+above: Rows,+below: Rows,
  core_buffer: Storage.Buffer<Element>,above_buffer: Storage.Buffer<Element>,below_buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  match exact:
    case True{}: core_buffer
    case False{}:
      Rows{+window_first,+window_rows} = window
      Rows{core_first,core_rows} = core
      Rows{above_first,above_rows} = above
      Rows{below_first,below_rows} = below
      copied_window(~Element,Bands.range_leaf(~Element,geometry,window_first,window_rows,0,below_first,below_rows,below_buffer,
        copied_window(~Element,Bands.range_leaf(~Element,geometry,window_first,window_rows,0,core_first,core_rows,core_buffer,
          copied_window(~Element,Bands.range_leaf(~Element,geometry,window_first,window_rows,0,above_first,above_rows,above_buffer,
            Bands.allocate(~Element,fill,Bands.local_count(geometry,window_rows))))))))

def assemble(~Element: Data,+fill: Element,+geometry: Bands.Geometry,+window: Rows,+core: Rows,+above: Rows,+below: Rows,
  core_buffer: Storage.Buffer<Element>,above_buffer: Storage.Buffer<Element>,below_buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Rows{+window_first,+window_rows} = window
  Rows{+core_first,+core_rows} = core
  assembled(~Element,U32.is_eq(window_first,core_first) && U32.is_eq(window_rows,core_rows),fill,geometry,window,core,above,below,
    core_buffer,above_buffer,below_buffer)

def input_geometry(geometry: Geometry) -> Bands.Geometry:
  Geometry{input,kernel,stride,padding,output_height,output_width} = geometry
  input

# The full core tree is threaded through the preparation pass without cloning.
# Each allocation is a requested neighbour range, never a core-sized window.
# The resulting halo tree follows the same layout, ready to pair with cores.
def prepare(~Element: Data,+geometry: Geometry,+fill: Element,layout: Bands.Plan,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element) & Halos(~Element):
  match layout:
    case Bands.Rows{+first,+count}: halo(~Element,input_geometry(geometry),fill,first,count,window(geometry,first,count),bands)
    case Bands.Split{left,right}:
      Storage.bind_pair(~Bands.Bands(~Element),~Halos(~Element),~(Bands.Bands(~Element) & Halos(~Element)),
        prepare(~Element,geometry,fill,left,bands),bands => left =>
          Storage.bind_pair(~Bands.Bands(~Element),~Halos(~Element),~(Bands.Bands(~Element) & Halos(~Element)),
            prepare(~Element,geometry,fill,right,bands),bands => right => (bands,Partition.Fork{left,right})))

def window_band(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Halo<Element> -> Bands.Bands(~Element),
  context: Context,first: U32,rows: U32,input: Storage.Buffer<Element>,halo: Halo<Element>,resource: Unit) -> Bands.Bands(~Element):
  leaf(context,first,rows,input,halo)

# Every core leaf with its halo, through the shared band recursion. Trees of
# another shape keep the cores.
def apply(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Halo<Element> -> Bands.Bands(~Element),
  layout: Bands.Plan,mode: Bool,enabled: Bool,geometry: Bands.Geometry,context: Context,bands: Bands.Bands(~Element),halos: Halos(~Element)) -> Bands.Bands(~Element):
  Bands.each(~Bands.Paired<Storage.Buffer<Element>,Halo<Element>>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,
    ~Bands.paired(~Storage.Buffer<Element>,~Halo<Element>,~Unit,~Context,~window_band(~Element,~Context,~leaf)),
    ~Bands.left_of(~Storage.Buffer<Element>,~Halo<Element>),
    layout,mode,enabled,geometry,context,Bands.zip(~Storage.Buffer<Element>,~Halo<Element>,bands,halos),Unit{})
