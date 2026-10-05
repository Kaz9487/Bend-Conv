# Scalar unit products use the actual checked metadata fast branches.
# This module is independent of View descriptors and storage ownership.
import Base
import ../math_checked_word.bend as CheckedWord
import ./checked_word_refinement.bend as Checked

def one_left(+count: U32) -> {CheckedWord.multiply(1,count) == Some{count} : Maybe<&2,U32>}:
  {==}

def right_when(+count: U32,unit: Bool,classified: {U32.is_eq(count,1) == unit : Bool}) ->
  {CheckedWord.multiply_left(unit,count,1) == Some{count} : Maybe<&2,U32>}:
  match unit:
    case True{}:
      %Equal.sym(U32,count,1,Checked.classified_equal(count,1,classified)) : {Some{1} == Some{_} : Maybe<&2,U32>}
      {==}
    case False{}: {==}

def one_right(+count: U32) -> {CheckedWord.multiply(count,1) == Some{count} : Maybe<&2,U32>}:
  right_when(count,U32.is_eq(count,1),{==})
