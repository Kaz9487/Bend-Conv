# W rows use oc; patch columns use (oy,ox); shared K axis is (ic,ky,kx).
# Axes have explicit order. They do not bake any runtime YOLO dimensions in.
import Base
import ../lib/ordered_matrix.bend as O
import ./specification.bend as S

type Reduction is Data:
  KIndex{ic: Nat,ky: Nat,kx: Nat}
type Pixel is Data:
  Position{oy: Nat,ox: Nat}

def kcols(n: Nat,+kx: Nat,+ky: Nat,+ic: Nat) -> List<&2,Reduction>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      KIndex{ic,ky,kx} <> kcols(rest,1n+kx,ky,ic)
def krows(n: Nat,+ky: Nat,+ic: Nat,+k: Nat) -> List<&2,Reduction>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      O.cat(Reduction,kcols(k,0n,ky,ic),krows(rest,1n+ky,ic,k))
def kchannels(n: Nat,+ic: Nat,+k: Nat) -> List<&2,Reduction>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      O.cat(Reduction,krows(k,0n,ic,k),kchannels(rest,1n+ic,k))
def columns(n: Nat,+ox: Nat,+oy: Nat) -> List<&2,Pixel>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      Position{oy,ox} <> columns(rest,1n+ox,oy)
def rows(n: Nat,+oy: Nat,+ow: Nat) -> List<&2,Pixel>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      O.cat(Pixel,columns(ow,0n,oy),rows(rest,1n+oy,ow))
def channels(n: Nat,+oc: Nat) -> List<&2,Nat>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      oc <> channels(rest,1n+oc)

def patch(p: S.Problem,c: Pixel,k: Reduction) -> F32:
  Position{oy,ox} = c
  KIndex{ic,ky,kx} = k
  S.sample(p,ic,oy,ox,ky,kx)
def weight(p: S.Problem,oc: Nat,k: Reduction) -> F32:
  KIndex{ic,ky,kx} = k
  S.weight(p,oc,ic,ky,kx)
def values(+p: S.Problem) -> List<&2,F32>:
  O.matmul(~Nat,~Reduction,~Pixel,~S.Problem,~patch,~weight,~S.bias_at,channels(S.outputs(p),0n),kchannels(S.channels(p),0n,S.kernel_size(p)),rows(S.out_height(p),0n,S.out_width(p)),p)

def finish(ok: Bool,+p: S.Problem) -> S.ConvOutcome:
  match ok:
    case False{}:
      S.Rejected{}
    case True{}:
      S.Accepted{S.out_height(p),S.out_width(p),values(p)}
def infer(+p: S.Problem) -> S.ConvOutcome:
  finish(S.valid(p),p)



