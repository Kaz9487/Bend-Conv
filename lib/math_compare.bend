# IEEE comparisons expressed as 32-bit masks, including unordered inputs.
import Base
import ./math_fp32.bend as FP32

# One for a nonzero word, zero otherwise, without a Bool match.
def nonzero(+word: U32) -> U32:
  U32.shrn(U32.or(word,(0 - word : U32)),31n)

def number(word: U32) -> U32:
  U32.xor(1,U32.shrn((2139095040 - U32.and(word,2147483647) : U32),31n))

def ordered(+left: U32,+right: U32) -> U32:
  U32.and(number(left),number(right))

# Magnitudes fit in 31 bits, so the top bit of their difference is the
# unsigned comparison. Reverse it for negatives and handle opposite signs.
def less_bit(+left: U32,+right: U32) -> U32:
  +a = U32.and(left,2147483647)
  +b = U32.and(right,2147483647)
  +negative = U32.shrn(left,31n)
  +different = U32.shrn(U32.xor(left,right),31n)
  +below = U32.shrn((a - b : U32),31n)
  +above = U32.shrn((b - a : U32),31n)
  +same = U32.xor(below,U32.and(negative,U32.xor(below,above)))
  U32.and(U32.xor(same,U32.and(different,U32.xor(same,negative))),nonzero(U32.or(a,b)))

def equal_bit(+left: U32,+right: U32) -> U32:
  U32.or(U32.xor(1,nonzero(U32.xor(left,right))),U32.xor(1,nonzero(U32.and(U32.or(left,right),2147483647))))

def mask_value(bit: U32) -> F32:
  FP32.from_bits(U32.and((0 - bit : U32),1065353216))

def less_words(+left: U32,+right: U32) -> F32:
  mask_value(U32.and(ordered(left,right),less_bit(left,right)))

def less_equal_words(+left: U32,+right: U32) -> F32:
  mask_value(U32.and(ordered(left,right),U32.xor(1,less_bit(right,left))))

def equal_words(+left: U32,+right: U32) -> F32:
  mask_value(U32.and(ordered(left,right),equal_bit(left,right)))

def not_equal_words(+left: U32,+right: U32) -> F32:
  mask_value(U32.xor(1,U32.and(ordered(left,right),equal_bit(left,right))))

# Context comes first in Operations.lanes; the tensor element comes second.
def less(value: F32,element: F32) -> F32:
  less_words(F32.bits(element),F32.bits(value))

def less_equal(value: F32,element: F32) -> F32:
  less_equal_words(F32.bits(element),F32.bits(value))

def greater(value: F32,element: F32) -> F32:
  less_words(F32.bits(value),F32.bits(element))

def greater_equal(value: F32,element: F32) -> F32:
  less_equal_words(F32.bits(value),F32.bits(element))

def equal(value: F32,element: F32) -> F32:
  equal_words(F32.bits(element),F32.bits(value))

def not_equal(value: F32,element: F32) -> F32:
  not_equal_words(F32.bits(element),F32.bits(value))

type Bounds is Data:
  Bounds{low: F32,high: F32}

def clamp(bounds: Bounds,element: F32) -> F32:
  Bounds{low,high} = bounds
  F32.min(F32.max(element,low),high)

# Binary element order is left then right.
def less_elements(left: F32,right: F32) -> F32:
  less(right,left)

def less_equal_elements(left: F32,right: F32) -> F32:
  less_equal(right,left)

def greater_elements(left: F32,right: F32) -> F32:
  greater(right,left)

def greater_equal_elements(left: F32,right: F32) -> F32:
  greater_equal(right,left)

def equal_elements(left: F32,right: F32) -> F32:
  equal(right,left)

def not_equal_elements(left: F32,right: F32) -> F32:
  not_equal(right,left)

# Selection only changes words; the chosen payload is never FP arithmetic.
# selected is 0 or all ones, so selected + 1 is 1 or 0 modulo 2^32.
# This masked integer product keeps the native selection in value lanes.
def select(mask: F32,yes: F32,no: F32) -> F32:
  +magnitude = U32.and(F32.bits(mask),2147483647)
  +selected = (0 - U32.shrn((magnitude + 2147483647 : U32),31n) : U32)
  +no = F32.bits(no)
  FP32.from_bits((U32.and(F32.bits(yes),selected) + no * (selected + 1 : U32) : U32))

def selected(value: F32,mask: F32) -> F32:
  select(mask,value,0.0)

def rejected(value: F32,mask: F32) -> F32:
  select(mask,0.0,value)

def union(left: F32,right: F32) -> F32:
  FP32.from_bits(U32.or(F32.bits(left),F32.bits(right)))
