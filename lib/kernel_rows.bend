import Base
import ./kernel_tile.bend as Tile

# Four independent output rows share one input tile: half of an eight-row
# weight block. This is a fixed product, not an indexed type or a list.
type Rows<-Element: Data> is Data:
  Rows{r0: Element,r1: Element,r2: Element,r3: Element}

def filled(~Element: Data,+value: Element) -> Rows<Element>:
  Rows{value,value,value,value}

def first(~Element: Data,values: Rows<Element>) -> Element:
  Rows{a,b,c,d} = values
  a

def second(~Element: Data,values: Rows<Element>) -> Element:
  Rows{a,b,c,d} = values
  b

def third(~Element: Data,values: Rows<Element>) -> Element:
  Rows{a,b,c,d} = values
  c

def fourth(~Element: Data,values: Rows<Element>) -> Element:
  Rows{a,b,c,d} = values
  d

# The lower or upper four rows of an eight-row weight tile.
def half(upper: Bool,tile: Tile.Tile8) -> Rows<F32>:
  match upper:
    case False{}:
      Tile.Tile{a,b,c,d,e,f,g,h} = tile
      Rows{a,b,c,d}
    case True{}:
      Tile.Tile{a,b,c,d,e,f,g,h} = tile
      Rows{e,f,g,h}

def map(~Element: Data,~Input: Data,~Weight: Data,
  ~update: Element -> Input -> Weight -> Element,
  accumulators: Rows<Element>,+input: Input,weights: Rows<Weight>) -> Rows<Element>:
  Rows{a0,a1,a2,a3} = accumulators
  Rows{w0,w1,w2,w3} = weights
  Rows{update(a0,input,w0),update(a1,input,w1),update(a2,input,w2),update(a3,input,w3)}
