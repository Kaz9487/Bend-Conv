import Base
import ./natural_addition.bend as Addition

import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./word_encoding.bend as WordEncoding

def not_below(+lower: Nat,+value: Nat,room: {Nat.is_le(lower,value) == True{} : Bool}) -> {Nat.is_lt(value,lower) == False{} : Bool}:
  match lower value:
    case 0n 0n: {==}
    case 0n 1n+value: {==}
    case 1n+lower 0n: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(room))
    case 1n+lower 1n+value: not_below(lower,value,room)

def cancel_offset_le(+offset: Nat,+left: Nat,+right: Nat,ordered: {Nat.is_le(Nat.add(offset,left),Nat.add(offset,right)) == True{} : Bool}) ->
  {Nat.is_le(left,right) == True{} : Bool}:
  match offset:
    case 0n: ordered
    case 1n+previous: cancel_offset_le(previous,left,right,ordered)

def equal_of_is_eq(+left: Nat,+right: Nat,equal: {Nat.is_eq(left,right) == True{} : Bool}) -> {left == right : Nat}:
  match left right:
    case 0n 0n: {==}
    case 0n 1n+rest: Empty.absurd({0n == 1n+rest : Nat},WordEncoding.false_true(equal))
    case 1n+rest 0n: Empty.absurd({1n+rest == 0n : Nat},WordEncoding.false_true(equal))
    case 1n+ +previous 1n+ +last:
      %equal_of_is_eq(previous,last,equal) : {1n+previous == 1n+_ : Nat}
      {==}

def subtract_zero(+value: Nat) -> {Nat.sub(value,0n) == value : Nat}:
  match value:
    case 0n: {==}
    case 1n+rest: {==}

def zero_subtract(value: Nat) -> {Nat.sub(0n,value) == 0n : Nat}:
  match value:
    case 0n: {==}
    case 1n+rest: {==}

def subtract_shift(+first: Nat,+row: Nat) -> {Nat.sub(Nat.add(first,row),first) == row : Nat}:
  match first:
    case 0n: subtract_zero(row)
    case 1n+rest: subtract_shift(rest,row)

def restore_shift(+first: Nat,+row: Nat,lower: {Nat.is_le(first,row) == True{} : Bool}) -> {Nat.add(first,Nat.sub(row,first)) == row : Nat}:
  match first row:
    case 0n row: subtract_zero(row)
    case 1n+first 0n: Empty.absurd({Nat.add(1n+first,Nat.sub(0n,1n+first)) == 0n : Nat},WordEncoding.false_true(lower))
    case 1n+ +first 1n+ +row:
      Equal.cong(Nat,Nat,n => 1n+n,Nat.add(first,Nat.sub(row,first)),row,restore_shift(first,row,lower))

def cancel_sum(+offset: Nat,+left: Nat,+right: Nat,equation: {Nat.add(offset,left) == Nat.add(offset,right) : Nat}) -> {left == right : Nat}:
  Equal.trans(Nat,left,Nat.sub(Nat.add(offset,left),offset),right,Equal.sym(Nat,Nat.sub(Nat.add(offset,left),offset),left,subtract_shift(offset,left)),
    Equal.trans(Nat,Nat.sub(Nat.add(offset,left),offset),Nat.sub(Nat.add(offset,right),offset),right,
      Equal.cong(Nat,Nat,sum => Nat.sub(sum,offset),Nat.add(offset,left),Nat.add(offset,right),equation),subtract_shift(offset,right)))

def positive_difference(+end: Nat,+begin: Nat,positive: {Nat.is_lt(begin,end) == True{} : Bool}) -> {Nat.is_lt(0n,Nat.sub(end,begin)) == True{} : Bool}:
  match end begin:
    case 0n 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(positive))
    case 0n 1n+begin: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+end 0n: {==}
    case 1n+end 1n+begin: positive_difference(end,begin,positive)

def positive_has_last(+size: Nat,positive: {Nat.is_lt(0n,size) == True{} : Bool}) -> {1n+Nat.sub(size,1n) == size : Nat}:
  match size:
    case 0n: Empty.absurd({1n == 0n : Nat},WordEncoding.false_true(positive))
    case 1n+ +rest:
      %subtract_zero(rest) : {1n+Nat.sub(rest,0n) == 1n+_ : Nat}
      {==}

def split_at_element(+total: Nat,+index: Nat,inside: {Nat.is_lt(index,total) == True{} : Bool}) ->
  {Nat.add(index,1n+Nat.sub(total,1n+index)) == total : Nat}:
  match total index:
    case 0n 0n: Empty.absurd({1n == 0n : Nat},WordEncoding.false_true(inside))
    case 1n+ +rest 0n:
      Equal.cong(Nat,Nat,n => 1n+n,Nat.sub(rest,0n),rest,subtract_zero(rest))
    case 0n 1n+ +index:
      Empty.absurd({Nat.add(1n+index,1n) == 0n : Nat},WordEncoding.false_true(inside))
    case 1n+ +rest 1n+ +index:
      Equal.cong(Nat,Nat,n => 1n+n,Nat.add(index,1n+Nat.sub(rest,1n+index)),rest,split_at_element(rest,index,inside))

type UpperBound<-value: Nat,-limit: Nat> is Data:
  Before{inside: {Nat.is_lt(value,limit) == True{} : Bool}}
  At{equal: {value == limit : Nat}}

def lift_upper_bound(+value: Nat,+limit: Nat,bound: UpperBound<value,limit>) -> UpperBound<1n+value,1n+limit>:
  match bound:
    case Before{inside}: Before{inside}
    case At{equal}: At{Equal.cong(Nat,Nat,n => 1n+n,value,limit,equal)}

def upper_bound_cases(+value: Nat,+limit: Nat,bounded: {Nat.is_le(value,limit) == True{} : Bool}) -> UpperBound<value,limit>:
  match value limit:
    case 0n 0n: At{{==}}
    case 0n 1n+rest: Before{{==}}
    case 1n+rest 0n: Empty.absurd(UpperBound<1n+rest,0n>,WordEncoding.false_true(bounded))
    case 1n+ +previous 1n+ +last: lift_upper_bound(previous,last,upper_bound_cases(previous,last,bounded))

def comparison_not_less(+comparison: Cmp,h: {Cmp.is_lt(comparison) == False{} : Bool}) -> {Cmp.is_ge(comparison) == True{} : Bool}:
  match comparison:
    case LT{}:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},h)))
    case EQ{}: {==}
    case GT{}: {==}

def comparison_less_not_equal(+comparison: Cmp,less: {Cmp.is_lt(comparison) == True{} : Bool}) -> {Cmp.is_eq(comparison) == False{} : Bool}:
  match comparison:
    case LT{}: {==}
    case EQ{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(less))
    case GT{}: Empty.absurd({False{} == False{} : Bool},WordEncoding.false_true(less))

def less_not_equal(+left: Nat,+right: Nat,less: {Nat.is_lt(left,right) == True{} : Bool}) -> {Nat.is_eq(left,right) == False{} : Bool}:
  comparison_less_not_equal(Nat.cmp(left,right),less)

def equal_reflexive(+value: Nat) -> {Nat.is_eq(value,value) == True{} : Bool}:
  match value:
    case 0n: {==}
    case 1n+rest: equal_reflexive(rest)

def equal_symmetric(+left: Nat,+right: Nat) -> {Nat.is_eq(left,right) == Nat.is_eq(right,left) : Bool}:
  match left right:
    case 0n 0n: {==}
    case 0n 1n+right: {==}
    case 1n+left 0n: {==}
    case 1n+left 1n+right: equal_symmetric(left,right)

def zero_le(+a: Nat) -> {Nat.is_le(0n,a) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      {==}

def reflexive(+a: Nat) -> {Nat.is_le(a,a) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      reflexive(p)

def offset_le(offset: Nat,+left: Nat,+right: Nat,ordered: {Nat.is_le(left,right) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(offset,left),Nat.add(offset,right)) == True{} : Bool}:
  match offset:
    case 0n: ordered
    case 1n+rest: offset_le(rest,left,right,ordered)

def offset_lt(+offset: Nat,+left: Nat,+right: Nat,ordered: {Nat.is_lt(left,right) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(offset,left),Nat.add(offset,right)) == True{} : Bool}:
  match offset:
    case 0n: ordered
    case 1n+rest: offset_lt(rest,left,right,ordered)

def cancel_offset_lt(+offset: Nat,+left: Nat,+right: Nat,ordered: {Nat.is_lt(Nat.add(offset,left),Nat.add(offset,right)) == True{} : Bool}) ->
  {Nat.is_lt(left,right) == True{} : Bool}:
  match offset:
    case 0n: ordered
    case 1n+rest: cancel_offset_lt(rest,left,right,ordered)

def successor(+a: Nat,+b: Nat,h: {Nat.is_le(a,b) == True{} : Bool}) -> {Nat.is_le(a,1n+b) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+ +p:
      match b:
        case 0n:
          Empty.absurd({Nat.is_le(1n+p,1n) == True{} : Bool},WordEncoding.false_true(h))
        case 1n+q:
          successor(p,q,h)

def transitive(+a: Nat,+b: Nat,+c: Nat,ab: {Nat.is_le(a,b) == True{} : Bool},bc: {Nat.is_le(b,c) == True{} : Bool}) -> {Nat.is_le(a,
    c) == True{} : Bool}:
  match a:
    case 0n:
      zero_le(c)
    case 1n+ +p:
      match b:
        case 0n:
          Empty.absurd({Nat.is_le(1n+p,c) == True{} : Bool},WordEncoding.false_true(ab))
        case 1n+ +q:
          match c:
            case 0n:
              Empty.absurd({Nat.is_le(1n+p,0n) == True{} : Bool},WordEncoding.false_true(bc))
            case 1n+r:
              transitive(p,q,r,ab,bc)

def le_add(+a: Nat,+b: Nat) -> {Nat.is_le(a,Nat.add(a,b)) == True{} : Bool}:
  match a:
    case 0n:
      zero_le(b)
    case 1n+p:
      le_add(p,b)

def le_double(+a: Nat) -> {Nat.is_le(a,Nat.double(a)) == True{} : Bool}:
  %Addition.double(a) : {Nat.is_le(a,_) == True{} : Bool}
  le_add(a,a)

def double_le(+a: Nat,+b: Nat,ordered: {Nat.is_le(a,b) == True{} : Bool}) -> {Nat.is_le(Nat.double(a),Nat.double(b)) == True{} : Bool}:
  match a b:
    case 0n b: zero_le(Nat.double(b))
    case 1n+a 0n: Empty.absurd({Nat.is_le(Nat.double(1n+a),0n) == True{} : Bool},WordEncoding.false_true(ordered))
    case 1n+a 1n+b: double_le(a,b,ordered)

def sub_le(+a: Nat,+b: Nat) -> {Nat.is_le(Nat.sub(a,b),a) == True{} : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+b:
      {==}
    case 1n+ +a 0n:
      reflexive(1n+a)
    case 1n+ +a 1n+ +b:
      successor(Nat.sub(a,b),a,sub_le(a,b))

def sub_lt(+a: Nat,+b: Nat,+c: Nat,order: {Nat.is_le(b,a) == True{} : Bool},bound: {Nat.is_lt(a,Nat.add(b,
        c)) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(a,b),c) == True{} : Bool}:
  match a b:
    case 0n 0n:
      bound
    case 0n 1n+b:
      Empty.absurd({Nat.is_lt(0n,c) == True{} : Bool},WordEncoding.false_true(order))
    case 1n+a 0n:
      bound
    case 1n+a 1n+b:
      sub_lt(a,b,c,order,bound)

def ge_le(+a: Nat,+b: Nat) -> {Nat.is_ge(a,b) == Nat.is_le(b,a) : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+b:
      {==}
    case 1n+a 0n:
      {==}
    case 1n+a 1n+b:
      ge_le(a,b)

def ge_true(+a: Nat,+b: Nat,h: {Nat.is_ge(a,b) == True{} : Bool}) -> {Nat.is_le(b,a) == True{} : Bool}:
  %ge_le(a,b) : {_ == True{} : Bool}
  h

def ge_false(+a: Nat,+b: Nat,h: {Nat.is_ge(a,b) == False{} : Bool}) -> {Nat.is_lt(a,b) == True{} : Bool}:
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},h)))
    case 0n 1n+b:
      {==}
    case 1n+a 0n:
      Empty.absurd({Nat.is_lt(1n+a,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},h)))
    case 1n+a 1n+b:
      ge_false(a,b,h)

def digit_lt(+a: Nat,+b: Nat,+bit: Bool,h: {Nat.is_lt(a,b) == True{} : Bool}) -> {Nat.is_lt(Comparison.digit(bit,a),
    Nat.double(b)) == True{} : Bool}:
  match a b bit:
    case 0n 0n bit:
      Empty.absurd({Nat.is_lt(Comparison.digit(bit,0n),0n) == True{} : Bool},WordEncoding.false_true(h))
    case 0n 1n+b False{}:
      {==}
    case 0n 1n+b True{}:
      {==}
    case 1n+a 0n bit:
      Empty.absurd({Nat.is_lt(Comparison.digit(bit,1n+a),0n) == True{} : Bool},WordEncoding.false_true(h))
    case 1n+a 1n+b False{}:
      digit_lt(a,b,False{},h)
    case 1n+a 1n+b True{}:
      digit_lt(a,b,True{},h)

def digit_bounded(+a: Nat,+b: Nat,+bit: Bool,+cap: Nat,small: {Nat.is_lt(a,b) == True{} : Bool},large: {Nat.is_le(Nat.double(b),
      cap) == True{} : Bool}) -> {Nat.is_le(Comparison.digit(bit,a),cap) == True{} : Bool}:
  Bounds.strict_le(Comparison.digit(bit,a),cap,Bounds.nat_trans(Comparison.digit(bit,a),Nat.double(b),cap,digit_lt(a,b,bit,
        small),large))

def commutative(+a: Nat,+b: Nat) -> {Nat.add(a,b) == Nat.add(b,a) : Nat}:
  match a:
    case 0n:
      Equal.sym(Nat,Nat.add(b,0n),b,Addition.zero_right(b))
    case 1n+ +p:
      %Equal.sym(Nat,Nat.add(b,1n+p),1n+Nat.add(b,p),Addition.successor_right(b,p)) : {1n+Nat.add(p,b) == _ : Nat}
      %commutative(p,b) : {1n+Nat.add(p,b) == 1n+_ : Nat}
      {==}

def le_lt_trans(+a: Nat,+b: Nat,+c: Nat,ab: {Nat.is_le(a,b) == True{} : Bool},bc: {Nat.is_lt(b,c) == True{} : Bool}) -> {Nat.is_lt(a,
    c) == True{} : Bool}:
  match a b c:
    case 0n 0n 0n:
      Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(bc))
    case 0n 1n+b 0n:
      Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(bc))
    case 0n b 1n+c:
      {==}
    case 1n+a 0n c:
      Empty.absurd({Nat.is_lt(1n+a,c) == True{} : Bool},WordEncoding.false_true(ab))
    case 1n+a 1n+b 0n:
      Empty.absurd({Nat.is_lt(1n+a,0n) == True{} : Bool},WordEncoding.false_true(bc))
    case 1n+a 1n+b 1n+c:
      le_lt_trans(a,b,c,ab,bc)

def advance_room(+i: Nat,+tail: Nat,+cap: Nat,h: {Nat.is_le(Nat.add(i,1n+tail),cap) == True{} : Bool}) -> {Nat.is_le(Nat.add(1n+i,tail),
    cap) == True{} : Bool}:
  %Addition.successor_right(i,tail) : {Nat.is_le(_,cap) == True{} : Bool}
  h

def room_index(+i: Nat,+tail: Nat,+cap: Nat,h: {Nat.is_le(Nat.add(i,1n+tail),cap) == True{} : Bool}) -> {Nat.is_lt(i,cap) == True{} : Bool}:
  le_lt_trans(i,Nat.add(i,tail),cap,le_add(i,tail),Bounds.predecessor_strict(Nat.add(i,tail),cap,advance_room(i,tail,cap,h)))

# Order structure shared by padding, allocation and mixed-radix indexing.
def lt_succ(+a: Nat,+b: Nat,h: {Nat.is_le(a,b) == True{} : Bool}) -> {Nat.is_lt(a,1n+b) == True{} : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+b:
      {==}
    case 1n+a 0n:
      Empty.absurd({Nat.is_lt(1n+a,1n) == True{} : Bool},WordEncoding.false_true(h))
    case 1n+a 1n+b:
      lt_succ(a,b,h)

# Order structure shared by padding, allocation and mixed-radix indexing.
def add_le(+a: Nat,+b: Nat,+c: Nat,h: {Nat.is_le(a,b) == True{} : Bool}) -> {Nat.is_le(Nat.add(a,c),Nat.add(b,c)) == True{} : Bool}:
  match a b:
    case 0n 0n:
      reflexive(c)
    case 0n 1n+ +b:
      successor(c,Nat.add(b,c),
        %commutative(c,b) : {Nat.is_le(c,_) == True{} : Bool}
        le_add(c,b))
    case 1n+a 0n:
      Empty.absurd({Nat.is_le(Nat.add(1n+a,c),c) == True{} : Bool},WordEncoding.false_true(h))
    case 1n+a 1n+b:
      add_le(a,b,c,h)

# Order structure shared by padding, allocation and mixed-radix indexing.
def add_lt(+a: Nat,+b: Nat,+c: Nat,h: {Nat.is_lt(a,b) == True{} : Bool}) -> {Nat.is_lt(Nat.add(a,c),Nat.add(b,c)) == True{} : Bool}:
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(c,c) == True{} : Bool},WordEncoding.false_true(h))
    case 0n 1n+ +b:
      lt_succ(c,Nat.add(b,c),
        %commutative(c,b) : {Nat.is_le(c,_) == True{} : Bool}
        le_add(c,b))
    case 1n+a 0n:
      Empty.absurd({Nat.is_lt(Nat.add(1n+a,c),c) == True{} : Bool},WordEncoding.false_true(h))
    case 1n+a 1n+b:
      add_lt(a,b,c,h)

# Order structure shared by padding, allocation and mixed-radix indexing.
def strict_follow(+a: Nat,+b: Nat,h: {Nat.is_lt(a,b) == True{} : Bool}) -> {Nat.is_le(1n+a,b) == True{} : Bool}:
  match a b:
    case 0n 0n:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(h))
    case 0n 1n+b:
      zero_le(b)
    case 1n+a 0n:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(h))
    case 1n+a 1n+b:
      strict_follow(a,b,h)

# Translate a comparison against a remaining extent to its absolute position.
def below_difference(+offset: Nat,+value: Nat,+end: Nat,ordered: {Nat.is_le(offset,end) == True{} : Bool}) ->
  {Nat.is_lt(value,Nat.sub(end,offset)) == Nat.is_lt(Nat.add(offset,value),end) : Bool}:
  match offset end:
    case 0n end:
      %Equal.sym(Nat,Nat.sub(end,0n),end,subtract_zero(end)) : {Nat.is_lt(value,_) == Nat.is_lt(value,end) : Bool}
      {==}
    case 1n+offset 0n:
      Empty.absurd({Nat.is_lt(value,Nat.sub(0n,1n+offset)) == Nat.is_lt(Nat.add(1n+offset,value),0n) : Bool},WordEncoding.false_true(ordered))
    case 1n+offset 1n+end: below_difference(offset,value,end,ordered)

# Two opposite bounds identify a natural number.
def antisymmetric(+left: Nat,+right: Nat,forward: {Nat.is_le(left,right) == True{} : Bool},backward: {Nat.is_le(right,left) == True{} : Bool}) -> {left == right : Nat}:
  match left right:
    case 0n 0n: {==}
    case 0n 1n+right: Empty.absurd({0n == 1n+right : Nat},WordEncoding.false_true(backward))
    case 1n+left 0n: Empty.absurd({1n+left == 0n : Nat},WordEncoding.false_true(forward))
    case 1n+ +left 1n+ +right:
      Equal.cong(Nat,Nat,value => 1n+value,left,right,antisymmetric(left,right,forward,backward))
