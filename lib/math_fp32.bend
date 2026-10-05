# Ordered FP32 arithmetic on the shared sequence model. No separate container.
import Base
import ./math_sequence.bend as Sequence

def multiply_add(accumulator: F32,input: F32,weight: F32) -> F32:
  (accumulator + input * weight : F32)

def weighted_add(weight: F32,accumulator: F32,input: F32) -> F32:
  multiply_add(accumulator,input,weight)

def zip_mac(length: Nat,accumulator: Sequence.Values(F32,length),input: Sequence.Values(F32,length),weight: F32) -> Sequence.Values(F32,length):
  Sequence.zip_map(~F32,~F32,~weighted_add,length,weight,accumulator,input)

def sigmoid(x: F32) -> F32:
  (1.0 / (1.0 + F32.exp(F32.neg(x))) : F32)
def activate(+x: F32, enabled: Bool) -> F32:
  match enabled:
    case True{}:
      (x * sigmoid(x) : F32)
    case False{}:
      x
