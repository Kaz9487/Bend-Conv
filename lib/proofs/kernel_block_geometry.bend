# Logical eight-row block coverage, independent of convolution or native bounds.
import Base
import ./nat_division.bend as Division
import ./nat_order.bend as Order
import ./word_encoding.bend as WordEncoding

def blocks(rows: Nat) -> Nat:
  Nat.div(7n+rows,8n)

def next_block(+rows: Nat) -> {blocks(8n+rows) == 1n+blocks(rows) : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(15n+rows,8n),Division.bump(Nat.divmod(7n+rows,8n)),Division.add_divisor(7n+rows,8n,{==})) :
    {Pair.fst(Nat,Nat,_) == 1n+blocks(rows) : Nat}
  Division.bumped_quotient(Nat.divmod(7n+rows,8n))

def count_bound(+rows: Nat) -> {Nat.is_le(blocks(rows),rows) == True{} : Bool}:
  match rows:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+ +rest:
      %Equal.sym(Nat,blocks(8n+rest),1n+blocks(rest),next_block(rest)) : {Nat.is_le(_,8n+rest) == True{} : Bool}
      Order.successor(blocks(rest),6n+rest,Order.successor(blocks(rest),5n+rest,Order.successor(blocks(rest),4n+rest,Order.successor(blocks(rest),3n+rest,Order.successor(blocks(rest),2n+rest,Order.successor(blocks(rest),1n+rest,Order.successor(blocks(rest),rest,count_bound(rest))))))))

def contains_row(+rows: Nat,+block: Nat,inside: {Nat.is_lt(Nat.mul(block,8n),rows) == True{} : Bool}) ->
  {Nat.is_lt(block,blocks(rows)) == True{} : Bool}:
  match rows block:
    case 0n block:
      Empty.absurd({Nat.is_lt(block,blocks(0n)) == True{} : Bool},WordEncoding.false_true(
        %Division.lt_zero(Nat.mul(block,8n)) : {_ == True{} : Bool}
        inside))
    case 1n 0n: {==}
    case 2n 0n: {==}
    case 3n 0n: {==}
    case 4n 0n: {==}
    case 5n 0n: {==}
    case 6n 0n: {==}
    case 7n 0n: {==}
    case 1n 1n+block:
      Empty.absurd({Nat.is_lt(1n+block,blocks(1n)) == True{} : Bool},WordEncoding.false_true(
        %Division.lt_zero(7n+Nat.mul(block,8n)) : {_ == True{} : Bool}
        inside))
    case 2n 1n+block:
      Empty.absurd({Nat.is_lt(1n+block,blocks(2n)) == True{} : Bool},WordEncoding.false_true(
        %Division.lt_zero(6n+Nat.mul(block,8n)) : {_ == True{} : Bool}
        inside))
    case 3n 1n+block:
      Empty.absurd({Nat.is_lt(1n+block,blocks(3n)) == True{} : Bool},WordEncoding.false_true(
        %Division.lt_zero(5n+Nat.mul(block,8n)) : {_ == True{} : Bool}
        inside))
    case 4n 1n+block:
      Empty.absurd({Nat.is_lt(1n+block,blocks(4n)) == True{} : Bool},WordEncoding.false_true(
        %Division.lt_zero(4n+Nat.mul(block,8n)) : {_ == True{} : Bool}
        inside))
    case 5n 1n+block:
      Empty.absurd({Nat.is_lt(1n+block,blocks(5n)) == True{} : Bool},WordEncoding.false_true(
        %Division.lt_zero(3n+Nat.mul(block,8n)) : {_ == True{} : Bool}
        inside))
    case 6n 1n+block:
      Empty.absurd({Nat.is_lt(1n+block,blocks(6n)) == True{} : Bool},WordEncoding.false_true(
        %Division.lt_zero(2n+Nat.mul(block,8n)) : {_ == True{} : Bool}
        inside))
    case 7n 1n+block:
      Empty.absurd({Nat.is_lt(1n+block,blocks(7n)) == True{} : Bool},WordEncoding.false_true(
        %Division.lt_zero(1n+Nat.mul(block,8n)) : {_ == True{} : Bool}
        inside))
    case 8n+ +rest 0n:
      %Equal.sym(Nat,blocks(8n+rest),1n+blocks(rest),next_block(rest)) : {Nat.is_lt(0n,_) == True{} : Bool}
      {==}
    case 8n+ +rest 1n+ +previous:
      %Equal.sym(Nat,blocks(8n+rest),1n+blocks(rest),next_block(rest)) : {Nat.is_lt(1n+previous,_) == True{} : Bool}
      contains_row(rest,previous,inside)
