# Least-significant axis first. Singleton axes are omitted. The two innermost
# axes are flat levels, so an element step changes registers only; outer axes
# stay in a list that is rebuilt only when every level wraps. Every step is a
# direct template or a tail call, so native code keeps the advance in one loop.
import Base
import ./tensor_view.bend as Views
import ./traversal_array.bend as Traversal

type Axis is Data:
  Axis{extent: Nat,remaining: Nat,stride: U32,rewind: U32}

def include_axis(active: Bool,+extent: U32,+stride: U32,axes: List<&2,Axis>) -> List<&2,Axis>:
  match active:
    case False{}: axes
    case True{}:
      +count = U32.to_nat(extent)
      Axis{count,count,stride,((extent - 1) * stride : U32)} <> axes

def axes(source: List<&2,Views.Axis>,reversed: List<&2,Axis>) -> List<&2,Axis>:
  match source:
    case []: reversed
    case Views.Axis{+extent,stride} <> tail:
      axes(tail,include_axis(U32.is_lt(1,extent),extent,stride,reversed))

# A level with no remaining count never moves; it stands for an absent axis.
type Level<-Rest: Data> is Data:
  Level{axis: Axis,rest: Rest}

def absent() -> Axis:
  Axis{0n,0n,0,0}

# Outer axes: reset axes are kept reversed and restored after the carry stops.
def restore(reset: List<&2,Axis>,axes: List<&2,Axis>) -> List<&2,Axis>:
  match reset:
    case []: axes
    case axis <> tail: restore(tail,axis <> axes)

def carry(axes: List<&2,Axis>,address: U32,reset: List<&2,Axis>) -> List<&2,Axis> & U32:
  match axes:
    case []: (restore(reset,[]),address)
    case Axis{extent,0n,stride,rewind} <> tail: (restore(reset,Axis{extent,0n,stride,rewind} <> tail),address)
    case Axis{+extent,1n,stride,+rewind} <> tail: carry(tail,(address - rewind : U32),Axis{extent,extent,stride,rewind} <> reset)
    case Axis{extent,1n+1n+remaining,+stride,rewind} <> tail:
      (restore(reset,Axis{extent,1n+remaining,stride,rewind} <> tail),(address + stride : U32))

def advance_outer(axes: List<&2,Axis>,address: U32) -> List<&2,Axis> & U32:
  carry(axes,address,[])

def reset(~Rest: Data,+extent: Nat,stride: U32,rewind: U32,result: Rest & U32) -> Level<Rest> & U32:
  (rest,address) = result
  (Level{Axis{extent,extent,stride,rewind},rest},address)

def step(~Rest: Data,~next: Rest -> U32 -> Rest & U32,level: Level<Rest>,address: U32) -> Level<Rest> & U32:
  Level{axis,rest} = level
  match axis:
    case Axis{extent,0n,stride,rewind}: (Level{Axis{extent,0n,stride,rewind},rest},address)
    case Axis{+extent,1n,stride,+rewind}: reset(~Rest,extent,stride,rewind,next(rest,(address - rewind : U32)))
    case Axis{extent,1n+1n+remaining,+stride,rewind}: (Level{Axis{extent,1n+remaining,stride,rewind},rest},(address + stride : U32))

def Depth0() -> Data:
  List<&2,Axis>

def Depth1() -> Data:
  Level<Depth0()>

def Levels() -> Data:
  Level<Depth1()>

def advance_levels(axes: Levels(),address: U32) -> Levels() & U32:
  step(~Depth1(),~step(~Depth0(),~advance_outer),axes,address)

type Cursor is Data:
  Cursor{axes: Levels(),address: U32}

def moved(result: Levels() & U32) -> Cursor:
  (axes,address) = result
  Cursor{axes,address}

def advance(axes: Levels(),address: U32) -> Cursor:
  moved(advance_levels(axes,address))

def split_level(~Rest: Data,~split_rest: List<&2,Axis> -> Rest,axes: List<&2,Axis>) -> Level<Rest>:
  match axes:
    case []: Level{absent(),split_rest([])}
    case axis <> tail: Level{axis,split_rest(tail)}

def outer(axes: List<&2,Axis>) -> List<&2,Axis>:
  axes

def split(axes: List<&2,Axis>) -> Levels():
  split_level(~Depth1(),~split_level(~Depth0(),~outer),axes)

def prepare(view: Views.View) -> Cursor:
  Views.View{source,offset} = view
  Cursor{split(axes(source,[])),offset}

def read_value(~Element: Data,axes: Levels(),address: U32,result: Array<Element> & Element) -> Array<Element> & Traversal.Visited<Cursor,Element>:
  (storage,value) = result
  (storage,Traversal.Visited{advance(axes,address),value})

def read(~Element: Data,storage: Array<Element>,cursor: Cursor) -> Array<Element> & Traversal.Visited<Cursor,Element>:
  Cursor{axes,+address} = cursor
  read_value(~Element,axes,address,Array.get(Element,storage,address))

# One run of the innermost axis reads address, address + stride, ... with no
# countdown. Its two words stay in registers in whatever program the cursor is
# compiled into; the full cursor is advanced once, at the end of the run.
type Stride is Data:
  Stride{address: U32,stride: U32}

def stride_value(~Element: Data,address: U32,+stride: U32,result: Array<Element> & Element) -> Array<Element> & Traversal.Visited<Stride,Element>:
  (storage,value) = result
  (storage,Traversal.Visited{Stride{(address + stride : U32),stride},value})

def read_stride(~Element: Data,storage: Array<Element>,cursor: Stride) -> Array<Element> & Traversal.Visited<Stride,Element>:
  Stride{+address,+stride} = cursor
  stride_value(~Element,address,stride,Array.get(Element,storage,address))

# The address a scatter stores at, with the cursor of the next element.
def target(cursor: Cursor) -> Traversal.Target<Cursor>:
  Cursor{axes,+address} = cursor
  Traversal.Target{advance(axes,address),address}

def target_stride(cursor: Stride) -> Traversal.Target<Stride>:
  Stride{+address,+stride} = cursor
  Traversal.Target{Stride{(address + stride : U32),stride},address}

type CursorPair is Data:
  CursorPair{left: Cursor,right: Cursor}
