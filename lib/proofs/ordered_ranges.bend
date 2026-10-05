# A finite half-open interval, represented as its consecutive positions.
import Base
import ./natural_addition.bend as Addition
import ../ordered_matrix.bend as OrderedMatrix

def range(n: Nat,+start: Nat) -> List<&2,Nat>:
  match n:
    case 0n:
      []
    case 1n+rest:
      start <> range(rest,1n+start)

def split(+a: Nat,+b: Nat,+start: Nat) -> {range(Nat.add(a,b),start) == OrderedMatrix.cat(Nat,range(a,start),range(b,Nat.add(start,
        a))) : List<&2,Nat>}:
  match a:
    case 0n:
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) : {range(b,start) == range(b,_) : List<&2,Nat>}
      {==}
    case 1n+ +p:
      %Equal.sym(Nat,Nat.add(start,1n+p),1n+Nat.add(start,p),Addition.successor_right(start,p)) : {start <> range(Nat.add(p,b),
          1n+start) == start <> OrderedMatrix.cat(Nat,range(p,1n+start),range(b,_)) : List<&2,Nat>}
      %split(p,b,1n+start) : {start <> range(Nat.add(p,b),1n+start) == start <> _ : List<&2,Nat>}
      {==}

# [start, end) followed by [end, start + whole) is the whole interval.
def join(+start: Nat,+left: Nat,+right: Nat,+end: Nat,+whole: Nat,end_exact: {end == Nat.add(start,left) : Nat},size_exact: {Nat.add(left,
      right) == whole : Nat}) -> {OrderedMatrix.cat(Nat,range(left,start),range(right,end)) == range(whole,start) : List<&2,Nat>}:
  %size_exact : {OrderedMatrix.cat(Nat,range(left,start),range(right,end)) == range(_,start) : List<&2,Nat>}
  %Equal.sym(Nat,end,Nat.add(start,left),end_exact) : {OrderedMatrix.cat(Nat,range(left,start),range(right,_)) == range(Nat.add(left,right),
      start) : List<&2,Nat>}
  Equal.sym(List<&2,Nat>,range(Nat.add(left,right),start),OrderedMatrix.cat(Nat,range(left,start),range(right,Nat.add(start,left))),split(left,
      right,start))
