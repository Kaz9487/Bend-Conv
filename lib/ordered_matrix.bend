# Ordered FP32 matrices with closed lookup templates and runtime data/dimensions.
import Base

def cat(-A: Data,xs: List<&2,A>,ys: List<&2,A>) -> List<&2,A>:
  match xs:
    case Nil{}:
      ys
    case h <> tail:
      h <> cat(A,tail,ys)

def dot(~R: Data,~K: Data,~C: Data,~P: Data,~x: P -> C -> K -> F32,~w: P -> R -> K -> F32,ks: List<&2,K>,+r: R,+c: C,+p: P,acc: F32) -> F32:
  match ks:
    case Nil{}:
      acc
    case +k <> tail:
      dot(~R,~K,~C,~P,~x,~w,tail,r,c,p,(acc + x(p,c,k) * w(p,r,k) : F32))

def row(~R: Data,~K: Data,~C: Data,~P: Data,~x: P -> C -> K -> F32,~w: P -> R -> K -> F32,~bias: P -> R -> F32,cs: List<&2,C>,+ks: List<&2,K>,+r: R,+p: P) -> List<&2,F32>:
  match cs:
    case Nil{}:
      Nil{}
    case c <> tail:
      (dot(~R,~K,~C,~P,~x,~w,ks,r,c,p,0.0) + bias(p,r) : F32) <> row(~R,~K,~C,~P,~x,~w,~bias,tail,ks,r,p)

def matmul(~R: Data,~K: Data,~C: Data,~P: Data,~x: P -> C -> K -> F32,~w: P -> R -> K -> F32,~bias: P -> R -> F32,rs: List<&2,R>,+ks: List<&2,K>,+cs: List<&2,C>,+p: P) -> List<&2,F32>:
  match rs:
    case Nil{}:
      Nil{}
    case +r <> tail:
      cat(F32,row(~R,~K,~C,~P,~x,~w,~bias,cs,ks,r,p),matmul(~R,~K,~C,~P,~x,~w,~bias,tail,ks,cs,p))
