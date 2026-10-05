# The same binary-digit interpretation serves long division and division by two.
import Base
import ./u32_comparison.bend as Comparison

def word_digit(+n: Nat,+bit: Bool,+w: Word(n)) -> {Word.to_nat(1n+n,WCon{bit,w}) == Comparison.digit(bit,Word.to_nat(n,w)) : Nat}:
  match bit:
    case False{}:
      {==}
    case True{}:
      {==}
