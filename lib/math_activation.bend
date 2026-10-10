# FP32 activations, imported by callers as Act: Act.silu() is a value for
# F.conv2d or F.activate, Act.silu_value the scalar function. Each built-in
# states its operations per element for the planners, counted from the emitted
# C and assembly: arithmetic, comparisons, selects and word operations;
# addressing, moves and loop control are excluded. The sigmoid, SiLU, tanh and
# GELU run eight lanes at a time, so their counts are the instructions per
# eight lanes divided by eight; the others are scalar counts. The sigmoid and
# tanh use ordered FP32 arithmetic and word operations (lib/math_fp32.bend).
# Sigmoid and SiLU scale their negative tails before the final underflow;
# accuracy is measured against MPFR rather than a global rounding guarantee.
import Base
import ./math_fp32.bend as Fp32

# Custom is a caller-supplied function; only its operation count lives here.
type Activation is Data:
  Identity{}
  ReLU{}
  ReLU6{}
  LeakyReLU{slope: F32}
  Sigmoid{}
  SiLU{}
  Tanh{}
  HardSigmoid{}
  HardSwish{}
  GELUTanh{}
  Custom{operations: F32}

def identity_value(value: F32) -> F32:
  value

def relu_value(value: F32) -> F32:
  F32.max(value,0.0)

def relu6_value(value: F32) -> F32:
  F32.min(F32.max(value,0.0),6.0)

def leaky(slope: F32,+value: F32) -> F32:
  Bool.pick(F32,F32.is_lt(value,0.0),(value * slope : F32),value)

# x for x >= 0, slope * x otherwise.
def leaky_relu_value(value: F32,slope: F32) -> F32:
  leaky(slope,value)

def sigmoid_value(value: F32) -> F32:
  Fp32.sigmoid(value)

def silu_value(value: F32) -> F32:
  Fp32.activate(value,True{})

def tanh_value(value: F32) -> F32:
  Fp32.tanh(value)

# PyTorch's order: relu6(x + 3) / 6 and x * relu6(x + 3) / 6.
def hard_sigmoid_value(value: F32) -> F32:
  (relu6_value((value + 3.0 : F32)) / 6.0 : F32)

def swished(+value: F32) -> F32:
  ((value * relu6_value((value + 3.0 : F32)) : F32) / 6.0 : F32)

def hard_swish_value(value: F32) -> F32:
  swished(value)

# PyTorch's gelu(approximate='tanh'): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
def gelu_tanh_of(+value: F32) -> F32:
  +cube = ((value * value : F32) * value : F32)
  ((0.5 * value : F32) * (1.0 + tanh_value((0.7978845608 * (value + (0.044715 * cube : F32) : F32) : F32)) : F32) : F32)

def gelu_tanh_value(value: F32) -> F32:
  gelu_tanh_of(value)

def identity() -> Activation:
  Identity{}

def relu() -> Activation:
  ReLU{}

def relu6() -> Activation:
  ReLU6{}

def leaky_relu(slope: F32) -> Activation:
  LeakyReLU{slope}

def sigmoid() -> Activation:
  Sigmoid{}

def silu() -> Activation:
  SiLU{}

def tanh() -> Activation:
  Tanh{}

def hard_sigmoid() -> Activation:
  HardSigmoid{}

def hard_swish() -> Activation:
  HardSwish{}

def gelu_tanh() -> Activation:
  GELUTanh{}

# The identity needs no pass at all.
def is_identity(activation: Activation) -> Bool:
  match activation:
    case Identity{}: True{}
    case activation: False{}

# Operations per element, excluding the pass that applies them.
def operations(activation: Activation) -> F32:
  match activation:
    case Identity{}: 0.0
    # compare and select
    case ReLU{}: 2.0
    case ReLU6{}: 4.0
    # compare, multiply and select
    case LeakyReLU{slope}: 3.0
    # 31 arithmetic, comparison, select and word instructions per eight lanes
    case Sigmoid{}: 3.875
    # 34 per eight lanes: scaled numerator and positive-tail selection
    case SiLU{}: 4.25
    # 54 per eight lanes
    case Tanh{}: 6.75
    # add, relu6 and divide; hard_swish also multiplies
    case HardSigmoid{}: 6.0
    case HardSwish{}: 7.0
    # 62 per eight lanes: the PyTorch expression around the tanh
    case GELUTanh{}: 7.75
    case Custom{operations}: operations

# A custom function (F.conv2d_custom, F.apply) states its scalar operations per
# element: one per arithmetic, comparison or select; F32.exp about 37 and
# F32.tanh about 82 with their conversions. The count only steers parallel
# planning; results never depend on it. When unsure, use this conservative
# count: a call into the C math library and a few operations around it.
def unknown_operations() -> F32:
  90.0
