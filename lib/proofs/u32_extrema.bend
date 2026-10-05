# Native interval endpoints agree with their unbounded minimum/maximum.
import Base
import ./u32_comparison.bend as Comparison
import ./nat_extrema.bend as Extrema

def selected(choice: Bool,+left: U32,+right: U32) ->
  {U32.to_nat(Bool.pick(U32,choice,left,right)) == Bool.pick(Nat,choice,U32.to_nat(left),U32.to_nat(right)) : Nat}:
  match choice:
    case False{}: {==}
    case True{}: {==}

def minimum(+left: U32,+right: U32) -> {U32.to_nat(U32.min(left,right)) == Nat.min(U32.to_nat(left),U32.to_nat(right)) : Nat}:
  U32{+left_word} = left
  U32{+right_word} = right
  +left = {U32{left_word} : U32}
  +right = {U32{right_word} : U32}
  Equal.trans(Nat,U32.to_nat(U32.min(left,right)),Bool.pick(Nat,U32.is_lt(left,right),U32.to_nat(left),U32.to_nat(right)),Nat.min(U32.to_nat(left),U32.to_nat(right)),
    selected(U32.is_lt(left,right),left,right),
    %Equal.sym(Bool,U32.is_lt(left,right),Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),Comparison.lt(left,right)) : {Bool.pick(Nat,_,U32.to_nat(left),U32.to_nat(right)) == Nat.min(U32.to_nat(left),U32.to_nat(right)) : Nat}
    Equal.sym(Nat,Nat.min(U32.to_nat(left),U32.to_nat(right)),Bool.pick(Nat,Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(left),U32.to_nat(right)),Extrema.minimum_choice(U32.to_nat(left),U32.to_nat(right))))

def maximum(+left: U32,+right: U32) -> {U32.to_nat(U32.max(left,right)) == Nat.max(U32.to_nat(left),U32.to_nat(right)) : Nat}:
  U32{+left_word} = left
  U32{+right_word} = right
  +left = {U32{left_word} : U32}
  +right = {U32{right_word} : U32}
  Equal.trans(Nat,U32.to_nat(U32.max(left,right)),Bool.pick(Nat,U32.is_lt(left,right),U32.to_nat(right),U32.to_nat(left)),Nat.max(U32.to_nat(left),U32.to_nat(right)),
    selected(U32.is_lt(left,right),right,left),
    %Equal.sym(Bool,U32.is_lt(left,right),Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),Comparison.lt(left,right)) : {Bool.pick(Nat,_,U32.to_nat(right),U32.to_nat(left)) == Nat.max(U32.to_nat(left),U32.to_nat(right)) : Nat}
    Equal.sym(Nat,Nat.max(U32.to_nat(left),U32.to_nat(right)),Bool.pick(Nat,Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(right),U32.to_nat(left)),Extrema.maximum_choice(U32.to_nat(left),U32.to_nat(right))))

# Base opens both arguments before it selects. These equations restore the
# selection form for arguments that are not constructor terms.
def minimum_pick(left: U32,right: U32) -> {U32.min(left,right) == Bool.pick(U32,U32.is_lt(left,right),left,right) : U32}:
  U32{left_word} = left
  U32{right_word} = right
  {==}

def pick_minimum(left: U32,right: U32) -> {Bool.pick(U32,U32.is_lt(left,right),left,right) == U32.min(left,right) : U32}:
  U32{left_word} = left
  U32{right_word} = right
  {==}

def pick_maximum(left: U32,right: U32) -> {Bool.pick(U32,U32.is_lt(left,right),right,left) == U32.max(left,right) : U32}:
  U32{left_word} = left
  U32{right_word} = right
  {==}
