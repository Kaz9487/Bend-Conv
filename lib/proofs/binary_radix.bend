# Quotient/remainder recover logical row and column from a bounded native index.
import Base
import ./nat_division.bend as Division
import ./u32_division.bend as U32Division
import ./binary_division.bend as BinaryDivision
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition

def result_total(+result: BinaryDivision.Quotient,+divisor: Nat) ->
  {BinaryDivision.total(result,divisor) == Nat.add(Nat.mul(U32Division.quotient(result),divisor),BinaryDivision.remainder(result)) : Nat}:
  BinaryDivision.Quotient{quotient,remainder} = result
  {==}

def division_identity(+value: U32,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {U32.to_nat(value) == Nat.add(Nat.mul(Nat.div(U32.to_nat(value),divisor),divisor),Nat.mod(U32.to_nat(value),divisor)) : Nat}:
  U32{+bits} = value
  %Equal.sym(Nat,Nat.div(Word.to_nat(32n,bits),divisor),U32Division.quotient(BinaryDivision.go(32n,bits,divisor)),U32Division.natural_quotient(bits,divisor,positive)) :
    {Word.to_nat(32n,bits) == Nat.add(Nat.mul(_,divisor),Nat.mod(Word.to_nat(32n,bits),divisor)) : Nat}
  %Equal.sym(Nat,Nat.mod(Word.to_nat(32n,bits),divisor),BinaryDivision.remainder(BinaryDivision.go(32n,bits,divisor)),U32Division.natural_remainder(bits,divisor,positive)) :
    {Word.to_nat(32n,bits) == Nat.add(Nat.mul(U32Division.quotient(BinaryDivision.go(32n,bits,divisor)),divisor),_) : Nat}
  %result_total(BinaryDivision.go(32n,bits,divisor),divisor) : {Word.to_nat(32n,bits) == _ : Nat}
  Equal.sym(Nat,BinaryDivision.total(BinaryDivision.go(32n,bits,divisor),divisor),Word.to_nat(32n,bits),BinaryDivision.go_total(32n,bits,divisor))

def remainder_inside(+value: U32,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mod(U32.to_nat(value),divisor),divisor) == True{} : Bool}:
  U32{+bits} = value
  %Equal.sym(Nat,Nat.mod(Word.to_nat(32n,bits),divisor),BinaryDivision.remainder(BinaryDivision.go(32n,bits,divisor)),U32Division.natural_remainder(bits,divisor,positive)) :
    {Nat.is_lt(_,divisor) == True{} : Bool}
  BinaryDivision.go_bound(32n,bits,divisor,positive)

def divide_eight_exact_multiple(+value: U32,aligned: {U32.is_zero((value % 8 : U32)) == True{} : Bool}) ->
  {Nat.mul(U32.to_nat((value / 8 : U32)),8n) == U32.to_nat(value) : Nat}:
  +quotient_exact = U32Division.divide(value,8,2147483648,{==},{==},{==})
  +remainder_exact = U32Division.modulo(value,8,2147483648,{==},{==},{==})
  +zero_remainder = Order.equal_of_is_eq(U32.to_nat((value % 8 : U32)),0n,
    %Comparison.eq((value % 8 : U32),0) : {_ == True{} : Bool}
    aligned)
  %Equal.sym(Nat,U32.to_nat((value / 8 : U32)),Nat.div(U32.to_nat(value),8n),quotient_exact) : {Nat.mul(_,8n) == U32.to_nat(value) : Nat}
  %Addition.zero_right(Nat.mul(Nat.div(U32.to_nat(value),8n),8n)) : {_ == U32.to_nat(value) : Nat}
  %Equal.trans(Nat,Nat.mod(U32.to_nat(value),8n),U32.to_nat((value % 8 : U32)),0n,Equal.sym(Nat,U32.to_nat((value % 8 : U32)),Nat.mod(U32.to_nat(value),8n),remainder_exact),zero_remainder) :
    {Nat.add(Nat.mul(Nat.div(U32.to_nat(value),8n),8n),_) == U32.to_nat(value) : Nat}
  Equal.sym(Nat,U32.to_nat(value),Nat.add(Nat.mul(Nat.div(U32.to_nat(value),8n),8n),Nat.mod(U32.to_nat(value),8n)),division_identity(value,8n,{==}))

def quotient(+row: Nat,+col: Nat,+width: Nat,positive: {Nat.is_lt(0n,width) == True{} : Bool},bound: {Nat.is_lt(col,
      width) == True{} : Bool}) -> {Nat.div(Nat.add(Nat.mul(row,width),col),width) == row : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(row,width),col),width),(row,col),Division.quotient_remainder(row,col,width,positive,
      bound)) : {Pair.fst(Nat,Nat,_) == row : Nat}
  {==}

def remainder(+row: Nat,+col: Nat,+width: Nat,positive: {Nat.is_lt(0n,width) == True{} : Bool},bound: {Nat.is_lt(col,
      width) == True{} : Bool}) -> {Nat.mod(Nat.add(Nat.mul(row,width),col),width) == col : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(row,width),col),width),(row,col),Division.quotient_remainder(row,col,width,positive,
      bound)) : {Pair.snd(Nat,Nat,_) == col : Nat}
  {==}

def quotient_word(+i: U32,+w: U32,+row: Nat,+col: Nat,+width: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +index: {U32.to_nat(i) == Nat.add(Nat.mul(row,width),col) : Nat},+size: {U32.to_nat(w) == width : Nat},+positive: {Nat.is_lt(0n,
      width) == True{} : Bool},bounded: {Nat.is_lt(col,width) == True{} : Bool},double_bound: {Nat.is_le(Nat.double(width),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((i / w : U32)) == row : Nat}:
  %quotient(row,col,width,positive,bounded) : {U32.to_nat((i / w : U32)) == _ : Nat}
  %index : {U32.to_nat((i / w : U32)) == Nat.div(_,width) : Nat}
  %size : {U32.to_nat((i / w : U32)) == Nat.div(U32.to_nat(i),_) : Nat}
  U32Division.divide(i,w,cap,cap_ok,
    %Equal.sym(Nat,U32.to_nat(w),width,size) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool}
    double_bound,
    %Equal.sym(Nat,U32.to_nat(w),width,size) : {Nat.is_lt(0n,_) == True{} : Bool}
    positive)

def remainder_word(+i: U32,+w: U32,+row: Nat,+col: Nat,+width: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +index: {U32.to_nat(i) == Nat.add(Nat.mul(row,width),col) : Nat},+size: {U32.to_nat(w) == width : Nat},+positive: {Nat.is_lt(0n,
      width) == True{} : Bool},bounded: {Nat.is_lt(col,width) == True{} : Bool},double_bound: {Nat.is_le(Nat.double(width),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((i % w : U32)) == col : Nat}:
  %remainder(row,col,width,positive,bounded) : {U32.to_nat((i % w : U32)) == _ : Nat}
  %index : {U32.to_nat((i % w : U32)) == Nat.mod(_,width) : Nat}
  %size : {U32.to_nat((i % w : U32)) == Nat.mod(U32.to_nat(i),_) : Nat}
  U32Division.modulo(i,w,cap,cap_ok,
    %Equal.sym(Nat,U32.to_nat(w),width,size) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool}
    double_bound,
    %Equal.sym(Nat,U32.to_nat(w),width,size) : {Nat.is_lt(0n,_) == True{} : Bool}
    positive)

def row_unique(+a: Nat,+b: Nat,+width: Nat,+x: Nat,+y: Nat,+positive: {Nat.is_lt(0n,width) == True{} : Bool},hx: {Nat.is_lt(x,
      width) == True{} : Bool},hy: {Nat.is_lt(y,width) == True{} : Bool},h: {Nat.add(Nat.mul(a,width),x) == Nat.add(Nat.mul(b,width),
      y) : Nat}) -> {a == b : Nat}:
  %quotient(a,x,width,positive,hx) : {_ == b : Nat}
  %quotient(b,y,width,positive,hy) : {Nat.div(Nat.add(Nat.mul(a,width),x),width) == _ : Nat}
  Equal.cong(Nat,Nat,n => Nat.div(n,width),Nat.add(Nat.mul(a,width),x),Nat.add(Nat.mul(b,width),y),h)

def col_unique(+a: Nat,+b: Nat,+width: Nat,+x: Nat,+y: Nat,+positive: {Nat.is_lt(0n,width) == True{} : Bool},hx: {Nat.is_lt(x,
      width) == True{} : Bool},hy: {Nat.is_lt(y,width) == True{} : Bool},h: {Nat.add(Nat.mul(a,width),x) == Nat.add(Nat.mul(b,width),
      y) : Nat}) -> {x == y : Nat}:
  %remainder(a,x,width,positive,hx) : {_ == y : Nat}
  %remainder(b,y,width,positive,hy) : {Nat.mod(Nat.add(Nat.mul(a,width),x),width) == _ : Nat}
  Equal.cong(Nat,Nat,n => Nat.mod(n,width),Nat.add(Nat.mul(a,width),x),Nat.add(Nat.mul(b,width),y),h)
