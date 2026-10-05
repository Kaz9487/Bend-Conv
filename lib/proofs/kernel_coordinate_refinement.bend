# A job's four accumulator rows are one half of the eight-lane weight tile.
# These erased observations connect the native projections to shared samples.
import Base
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ../math_sequence.bend as Sequence
import ./kernel_accumulator.bend as Accumulator

def row_number(row: Accumulator.Row) -> Nat:
  match row:
    case Accumulator.R0{}: 0n
    case Accumulator.R1{}: 1n
    case Accumulator.R2{}: 2n
    case Accumulator.R3{}: 3n

def lower_lane(row: Accumulator.Row) -> Sequence.Index(4n):
  match row:
    case Accumulator.R0{}: Sequence.First{}
    case Accumulator.R1{}: Sequence.Next{Sequence.First{}}
    case Accumulator.R2{}: Sequence.Next{Sequence.Next{Sequence.First{}}}
    case Accumulator.R3{}: Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}

def weight_lane(upper: Bool,row: Accumulator.Row) -> Sequence.Index(8n):
  match upper row:
    case False{} Accumulator.R0{}: Sequence.First{}
    case False{} Accumulator.R1{}: Sequence.Next{Sequence.First{}}
    case False{} Accumulator.R2{}: Sequence.Next{Sequence.Next{Sequence.First{}}}
    case False{} Accumulator.R3{}: Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}
    case True{} row: Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{lower_lane(row)}}}}

def weight_coordinate(+upper: Bool,+row: Accumulator.Row) ->
  {Sequence.index_value(8n,weight_lane(upper,row)) == Nat.add(Bool.pick(Nat,upper,4n,0n),row_number(row)) : Nat}:
  match upper row:
    case False{} Accumulator.R0{}: {==}
    case False{} Accumulator.R1{}: {==}
    case False{} Accumulator.R2{}: {==}
    case False{} Accumulator.R3{}: {==}
    case True{} Accumulator.R0{}: {==}
    case True{} Accumulator.R1{}: {==}
    case True{} Accumulator.R2{}: {==}
    case True{} Accumulator.R3{}: {==}

def weight_sample(+upper: Bool,+row: Accumulator.Row,+tile: Tile.Tile8) ->
  {Accumulator.row(~F32,row,Rows.half(upper,tile)) == Tile.element_at(weight_lane(upper,row),tile) : F32}:
  match upper row tile:
    case False{} Accumulator.R0{} Tile.Tile{a,b,c,d,e,f,g,h}: {==}
    case False{} Accumulator.R1{} Tile.Tile{a,b,c,d,e,f,g,h}: {==}
    case False{} Accumulator.R2{} Tile.Tile{a,b,c,d,e,f,g,h}: {==}
    case False{} Accumulator.R3{} Tile.Tile{a,b,c,d,e,f,g,h}: {==}
    case True{} Accumulator.R0{} Tile.Tile{a,b,c,d,e,f,g,h}: {==}
    case True{} Accumulator.R1{} Tile.Tile{a,b,c,d,e,f,g,h}: {==}
    case True{} Accumulator.R2{} Tile.Tile{a,b,c,d,e,f,g,h}: {==}
    case True{} Accumulator.R3{} Tile.Tile{a,b,c,d,e,f,g,h}: {==}
