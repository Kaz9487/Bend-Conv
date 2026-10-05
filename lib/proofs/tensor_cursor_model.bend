# Coordinate semantics use increasing unbounded coordinates and a dot product.
# Runtime countdowns and incremental addresses do not occur in this model.
import Base
import ../tensor_view.bend as Views

type Coordinate is Data:
  Coordinate{extent: Nat,coordinate: Nat,stride: Nat}

def address(axes: List<&2,Coordinate>,offset: Nat) -> Nat:
  match axes:
    case []: offset
    case Coordinate{extent,coordinate,stride} <> tail:
      Nat.add(Nat.mul(coordinate,stride),address(tail,offset))

# The least significant coordinate is first. Exhausting an axis sets its
# coordinate to zero and increments the next coordinate (row-major order).
def advance(axes: List<&2,Coordinate>) -> List<&2,Coordinate>:
  match axes:
    case []: []
    case Coordinate{+extent,+coordinate,+stride} <> +tail:
      Bool.pick(List<&2,Coordinate>,Nat.is_lt(1n+coordinate,extent),
        Coordinate{extent,1n+coordinate,stride} <> tail,
        Coordinate{extent,0n,stride} <> advance(tail))

def extent_product(axes: List<&2,Coordinate>) -> Nat:
  match axes:
    case []: 1n
    case Coordinate{extent,coordinate,stride} <> tail: Nat.mul(extent,extent_product(tail))

def position(axes: List<&2,Coordinate>) -> Nat:
  match axes:
    case []: 0n
    case Coordinate{extent,coordinate,stride} <> tail: Nat.add(coordinate,Nat.mul(extent,position(tail)))

def span(axes: List<&2,Coordinate>) -> Nat:
  match axes:
    case []: 0n
    case Coordinate{extent,coordinate,stride} <> tail: Nat.add(Nat.mul(Nat.sub(extent,1n),stride),span(tail))

def prepare_axes(source: List<&2,Views.Axis>,reversed: List<&2,Coordinate>) -> List<&2,Coordinate>:
  match source:
    case []: reversed
    case Views.Axis{extent,stride} <> tail:
      +extent = U32.to_nat(extent)
      +reversed = reversed
      prepare_axes(tail,Bool.pick(List<&2,Coordinate>,Nat.is_lt(1n,extent),Coordinate{extent,0n,U32.to_nat(stride)} <> reversed,reversed))
