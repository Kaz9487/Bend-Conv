# Numerical integration output only; contains no inference implementation.
import Base
import ../../convolution/specification.bend as ConvSpec

def Test.header(id: U32, h: U32, w: U32) -> IO(Unit):
  import "../../benchmarks/checks/io.c"
def Test.value(x: F32) -> IO(Unit):
  import "../../benchmarks/checks/io.c"
def Test.end() -> IO(Unit):
  import "../../benchmarks/checks/io.c"

def values(xs: List<&2,F32>) -> IO(Unit):
  match xs:
    case Nil{}:
      Test.end()
    case h <> tail:
      do IO<Unit>:
        Test.value(h)
        values(tail)

def dump(id: U32, result: ConvSpec.ConvOutcome) -> IO(Unit):
  match result:
    case ConvSpec.Rejected{}:
      do IO<Unit>:
        Test.header(id,0,0)
        Test.end()
    case ConvSpec.Accepted{h,w,xs}:
      do IO<Unit>:
        Test.header(id,U32.from_nat(h),U32.from_nat(w))
        values(xs)

