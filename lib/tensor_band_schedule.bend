# Conservative memory-work estimates for elementwise and layout operations.
# Small operations retain their owners and run sequentially. Multi-image trees
# remain sequential here; window operations have explicit per-image dispatch.
import Base
import ./tensor_band.bend as Bands
import ./storage_buffer.bend as Storage
import ./tensor_operations.bend as Operations
import ./kernel_cost.bend as Cost
import ./default_kernel_cost.bend as DefaultCost

type Size is Data:
  Size{leaves: U32,minimum_rows: U32,maximum_rows: U32}

def merge(left: Size,right: Size) -> Size:
  Size{ln,lmin,lmax} = left
  Size{rn,rmin,rmax} = right
  Size{(ln + rn : U32),U32.min(lmin,rmin),U32.max(lmax,rmax)}

def size(layout: Bands.Plan) -> Size:
  match layout:
    case Bands.Rows{first,+rows}: Size{1,rows,rows}
    case Bands.Split{left,right}: merge(size(left),size(right))

# The whole operation, not each leaf, must reach the grain: small leaves only
# lengthen the runtime's rounds, which the region estimate already counts.
def sized(geometry: Bands.Geometry,+workers: U32,element_ns: F32,config: Cost.Configuration,size: Size) -> Bool:
  Bands.Geometry{batch,channels,height,width} = geometry
  Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} = config
  Size{+leaves,rows,maximum} = size
  +row_ns = (U32.to_f32(channels) * U32.to_f32(width) * element_ns : F32)
  +total = (U32.to_f32(height) * row_ns : F32)
  U32.is_eq(batch,1) && U32.is_lt(1,workers) && U32.is_lt(1,leaves) && U32.is_le(leaves,(U32.min(workers,858993459) * 5 : U32)) &&
    F32.is_le(grain,total) &&
    F32.is_lt((Cost.region_ns(workers,leaves,(U32.to_f32(maximum) * row_ns : F32)) + 2.0 * wake : F32),total)

def parallel(geometry: Bands.Geometry,workers: U32,bytes_per_element: F32,operations: F32,layout: Bands.Plan) -> Bool:
  +config = DefaultCost.configuration()
  sized(geometry,workers,Cost.element_ns(bytes_per_element,operations,config),config,size(layout))

# For equal-shaped operands the logical read/write volume of repartition is
# identical. The target owners' rounded capacities decide the remaining cost.
def allocation_elements(layout: Bands.Plan,+geometry: Bands.Geometry) -> Nat:
  match layout:
    case Bands.Rows{first,rows}: U32.to_nat(U32.shln(1,Storage.depth_for(Bands.local_count(geometry,rows))))
    case Bands.Split{left,right}: Nat.add(allocation_elements(left,geometry),allocation_elements(right,geometry))

def alignment(+geometry: Bands.Geometry,+left: Bands.Plan,+right: Bands.Plan) -> Bands.Plan:
  Bool.pick(Bands.Plan,Bands.same_plan(left,right) || Nat.is_le(allocation_elements(left,geometry),allocation_elements(right,geometry)),left,right)

def mapped_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,+geometry: Bands.Geometry,workers: U32,operations: F32,owners: Bands.Bands(~Element) & Bands.Plan) -> Bands.Bands(~Element):
  (bands,+layout) = owners
  Bands.map_with(~Element,~Context,~Bands.map_whole_context(~Element,~Context,~mapper),layout,parallel(geometry,workers,8.0,operations,layout),geometry,context,bands)

def map_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,geometry: Bands.Geometry,workers: U32,operations: F32,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  mapped_context(~Element,~Context,~mapper,context,geometry,workers,operations,Bands.plan(~Element,bands))

def scalar_planned(~Element: Data,~operation: Element -> Element -> Element,+geometry: Bands.Geometry,workers: U32,operations: F32,value: Element,owners: Bands.Bands(~Element) & Bands.Plan) -> Bands.Bands(~Element):
  (bands,+layout) = owners
  Bands.scalar(~Element,~operation,layout,parallel(geometry,workers,8.0,operations,layout),geometry,value,bands)

def scalar(~Element: Data,~operation: Element -> Element -> Element,geometry: Bands.Geometry,workers: U32,operations: F32,value: Element,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  scalar_planned(~Element,~operation,geometry,workers,operations,value,Bands.plan(~Element,bands))

def indexed_planned(~Element: Data,~operation: U32 -> Element -> Element,+geometry: Bands.Geometry,workers: U32,operations: F32,fill: Element,owners: Bands.Bands(~Element) & Bands.Plan) -> Bands.Bands(~Element):
  (bands,+layout) = owners
  Bands.apply_indexed(~Element,~operation,layout,parallel(geometry,workers,12.0,operations,layout),geometry,fill,bands)

def indexed(~Element: Data,~operation: U32 -> Element -> Element,geometry: Bands.Geometry,workers: U32,operations: F32,fill: Element,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  indexed_planned(~Element,~operation,geometry,workers,operations,fill,Bands.plan(~Element,bands))

def transformed(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Bands.Bands(~Element),
  context: Context,+geometry: Bands.Geometry,workers: U32,bytes_per_element: F32,owners: Bands.Bands(~Element) & Bands.Plan) -> Bands.Bands(~Element):
  (bands,+layout) = owners
  Bands.transform(~Element,~Context,~leaf,layout,parallel(geometry,workers,bytes_per_element,0.0,layout),geometry,context,bands)

def transform(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Bands.Bands(~Element),
  context: Context,geometry: Bands.Geometry,workers: U32,bytes_per_element: F32,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  transformed(~Element,~Context,~leaf,context,geometry,workers,bytes_per_element,Bands.plan(~Element,bands))
