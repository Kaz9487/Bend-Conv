# Exact destination used by direct_matrix.copied, and injectivity of its
# logical map. Iteration coverage and actual tile-range premises are separate.
import Base
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition

def madd_words(+a: U32,+b: U32,+c: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +bound: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(a),U32.to_nat(b)),U32.to_nat(c)),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((a * b + c : U32)) == Nat.add(Nat.mul(U32.to_nat(a),U32.to_nat(b)),U32.to_nat(c)) : Nat}:
  +ep = Arithmetic.multiply(a,b,cap,cap_ok,Order.transitive(Nat.mul(U32.to_nat(a),U32.to_nat(b)),Nat.add(Nat.mul(U32.to_nat(a),
          U32.to_nat(b)),U32.to_nat(c)),U32.to_nat(cap),Order.le_add(Nat.mul(U32.to_nat(a),U32.to_nat(b)),U32.to_nat(c)),bound))
  %ep : {U32.to_nat((a * b + c : U32)) == Nat.add(_,U32.to_nat(c)) : Nat}
  Arithmetic.add((a * b : U32),c,cap,cap_ok,
    %Equal.sym(Nat,U32.to_nat((a * b : U32)),Nat.mul(U32.to_nat(a),U32.to_nat(b)),ep) : {Nat.is_le(Nat.add(_,U32.to_nat(c)),
        U32.to_nat(cap)) == True{} : Bool}
    bound)

def destination(+row: Nat,+col: Nat,+spatial: Nat,+offset: Nat) -> Nat:
  Nat.add(Nat.mul(row,spatial),Nat.add(offset,col))

def expanded(+row: Nat,+col: Nat,+spatial: Nat,+offset: Nat) -> {Nat.add(Nat.add(Nat.mul(row,spatial),offset),col) == destination(row,col,
    spatial,offset) : Nat}:
  Addition.associative(Nat.mul(row,spatial),offset,col)

def bound_machine(+q: U32,+r: U32,+spatial: U32,+offset: U32,+row: Nat,+col: Nat,+space: Nat,+start: Nat,+cap: U32,+eq: {U32.to_nat(q) == row : Nat},+er: {U32.to_nat(r) == col : Nat},
  +es: {U32.to_nat(spatial) == space : Nat},+eo: {U32.to_nat(offset) == start : Nat},bound: {Nat.is_le(destination(row,col,space,start),
      U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.add(Nat.mul(U32.to_nat(q),U32.to_nat(spatial)),U32.to_nat(offset)),
      U32.to_nat(r)),U32.to_nat(cap)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(q),row,eq) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(_,U32.to_nat(spatial)),U32.to_nat(offset)),U32.to_nat(r)),
      U32.to_nat(cap)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(r),col,er) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(row,U32.to_nat(spatial)),U32.to_nat(offset)),_),
      U32.to_nat(cap)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(spatial),space,es) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(row,_),U32.to_nat(offset)),col),U32.to_nat(cap)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(offset),start,eo) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(row,space),_),col),U32.to_nat(cap)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(row,space),start),col),destination(row,col,space,start),expanded(row,col,space,start)) : {Nat.is_le(_,
      U32.to_nat(cap)) == True{} : Bool}
  bound

def combine(+q: U32,+r: U32,+spatial: U32,+offset: U32,+row: Nat,+col: Nat,+space: Nat,+start: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,
      2147483648) == True{} : Bool},+eq: {U32.to_nat(q) == row : Nat},+er: {U32.to_nat(r) == col : Nat},
  +es: {U32.to_nat(spatial) == space : Nat},+eo: {U32.to_nat(offset) == start : Nat},bound: {Nat.is_le(destination(row,col,space,start),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((q * spatial + offset + r : U32)) == destination(row,col,space,start) : Nat}:
  +hb = bound_machine(q,r,spatial,offset,row,col,space,start,cap,eq,er,es,eo,bound)
  +em = madd_words(q,spatial,offset,cap,cap_ok,Order.transitive(Nat.add(Nat.mul(U32.to_nat(q),U32.to_nat(spatial)),U32.to_nat(offset)),
      Nat.add(Nat.add(Nat.mul(U32.to_nat(q),U32.to_nat(spatial)),U32.to_nat(offset)),U32.to_nat(r)),U32.to_nat(cap),
      Order.le_add(Nat.add(Nat.mul(U32.to_nat(q),U32.to_nat(spatial)),U32.to_nat(offset)),U32.to_nat(r)),hb))
  %expanded(row,col,space,start) : {U32.to_nat((q * spatial + offset + r : U32)) == _ : Nat}
  %eq : {U32.to_nat((q * spatial + offset + r : U32)) == Nat.add(Nat.add(Nat.mul(_,space),start),col) : Nat}
  %er : {U32.to_nat((q * spatial + offset + r : U32)) == Nat.add(Nat.add(Nat.mul(U32.to_nat(q),space),start),_) : Nat}
  %es : {U32.to_nat((q * spatial + offset + r : U32)) == Nat.add(Nat.add(Nat.mul(U32.to_nat(q),_),start),U32.to_nat(r)) : Nat}
  %eo : {U32.to_nat((q * spatial + offset + r : U32)) == Nat.add(Nat.add(Nat.mul(U32.to_nat(q),U32.to_nat(spatial)),_),U32.to_nat(r)) : Nat}
  %em : {U32.to_nat((q * spatial + offset + r : U32)) == Nat.add(_,U32.to_nat(r)) : Nat}
  Arithmetic.add((q * spatial + offset : U32),r,cap,cap_ok,
    %Equal.sym(Nat,U32.to_nat((q * spatial + offset : U32)),Nat.add(Nat.mul(U32.to_nat(q),U32.to_nat(spatial)),U32.to_nat(offset)),
      em) : {Nat.is_le(Nat.add(_,U32.to_nat(r)),U32.to_nat(cap)) == True{} : Bool}
    hb)
