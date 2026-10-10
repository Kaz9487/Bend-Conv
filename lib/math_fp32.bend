# Ordered FP32 arithmetic on the shared sequence model. No separate container.
import Base
import ./math_sequence.bend as Sequence

def multiply_add(accumulator: F32,input: F32,weight: F32) -> F32:
  (accumulator + input * weight : F32)

def weighted_add(weight: F32,accumulator: F32,input: F32) -> F32:
  multiply_add(accumulator,input,weight)

def zip_mac(length: Nat,accumulator: Sequence.Values(F32,length),input: Sequence.Values(F32,length),weight: F32) -> Sequence.Values(F32,length):
  Sequence.zip_map(~F32,~F32,~weighted_add,length,weight,accumulator,input)

# The Cephes expf polynomial after its constant and linear terms.
def exponential_polynomial(+r: F32) -> F32:
  (((((0.00019875691500 * r + 0.0013981999507 : F32) * r + 0.0083334519073 : F32) * r + 0.041665795894 : F32) * r + 0.16666665459 : F32) * r + 0.50000001201 : F32)

# The 32-bit word of a U32, read as an F32.
def from_bits(value: U32) -> F32:
  U32{word} = value
  F32{word}

# exp(r)-1-r near |r| <= ln(2)/2. Chebyshev coefficients, rounded to F32;
# paired evaluation shortens the dependency chain without contracting products.
def exponential_remainder(+r: F32) -> F32:
  +z = (r * r : F32)
  (z * ((0.5 + 0.16666577756404877 * r : F32) + z * ((0.041666556149721146 + 0.008363173343241215 * r : F32) + z * 0.0013926175888627768 : F32) : F32) : F32)

# b=RN(1+q). Recover the small term lost in that sum before adding the
# reduced exponential. Writing (1-b)+q keeps the eight lanes together under
# the default vectorizer; q-(b-1) makes it mix unrelated addition trees.
def logistic_denominator(q: F32,+base: F32,r: F32,correction: F32) -> F32:
  (base + (r + (correction + ((1.0 - base : F32) + q : F32) : F32) : F32) : F32)

def selected(+mask: U32,yes: F32,+no: F32) -> F32:
  from_bits(U32.xor(F32.bits(no),U32.and(U32.xor(F32.bits(yes),F32.bits(no)),mask)))

# Above 18 SiLU rounds to x. Pass x through, including positive infinity;
# the negative tail uses the clamped input before the final underflow.
def silu_finish(+x: F32,value: F32) -> F32:
  +word = F32.bits(x)
  +positive = U32.not((0 - U32.shrn(word,31n) : U32))
  +large = U32.and(positive,(0 - U32.shrn((1099956224 - word : U32),31n) : U32))
  selected(large,x,value)

def logistic_scaled(silu: Bool,x: F32,value: F32,power: F32,denominator: F32) -> F32:
  match silu:
    case True{}:
      silu_finish(x,(((value * power : F32) / denominator : F32) * 0.00000000000000000005421010862427522 : F32))
    case False{}:
      ((power / denominator : F32) * 0.00000000000000000005421010862427522 : F32)

# u=-value, n approximates u/ln(2) to the nearest integer, r=u-n*ln(2).
# The exponent word makes
# H=2^(64-n), so sigmoid(x)=H/(exp(r)+2^-n)*2^-64. H remains normal;
# only the last multiplication underflows. The tiny denominator addend q
# may be floored to the least normal F32 without changing its rounded sum.
def sigmoid_reduced(silu: Bool,x: F32,+value: F32,+biased: F32) -> F32:
  +n = (biased - 12582912.0 : F32)
  +r = ((n * F32.neg(0.693359375) - value : F32) + n * 0.00021219444005469058 : F32)
  +power = from_bits((1602224128 - U32.shln(F32.bits(biased),23n) : U32))
  +q = from_bits((U32.max(F32.bits(power),545259520) - 536870912 : U32))
  logistic_scaled(silu,x,value,power,logistic_denominator(q,(1.0 + q : F32),r,exponential_remainder(r)))

def sigmoid_input(silu: Bool,x: F32,+value: F32) -> F32:
  sigmoid_reduced(silu,x,value,(value * F32.neg(1.44269504088896341) + 12582912.0 : F32))

# At x>=18 sigmoid rounds to 1. Below -110 both sigmoid and SiLU round
# to zero; multiplying the clamped x before the final scaling preserves -0
# for SiLU, including at -infinity. NaNs pass the clamp and propagate.
def sigmoid(+x: F32) -> F32:
  sigmoid_input(False{},x,F32.max(F32.min(18.0,x),F32.neg(110.0)))

def activate(+x: F32,enabled: Bool) -> F32:
  match enabled:
    case True{}:
      sigmoid_input(True{},x,F32.max(F32.min(18.0,x),F32.neg(110.0)))
    case False{}:
      x

# Sign restoration and the tiny interval are word selections. For |x|<2^-12,
# |x-tanh(x)|<|x|^3/3 is below half an F32 ulp, so return x, including -0.
def tanh_finish(+x: F32,+value: F32) -> F32:
  +word = F32.bits(x)
  +small = (0 - U32.shrn((U32.and(word,2147483647) - 964689920 : U32),31n) : U32)
  +keep = U32.or(small,2147483648)
  from_bits(U32.or(U32.and(word,keep),U32.and(F32.bits(value),U32.not(keep))))

# At |x|>=10, 1-tanh(|x|)<2*exp(-20)<2^-25. Clamping before arithmetic
# also keeps infinities away from intermediate products; NaNs propagate.
def tanh_magnitude(+x: F32) -> F32:
  F32.max(F32.min(10.0,from_bits(U32.and(F32.bits(x),2147483647))),0.000244140625)

# Chebyshev interpolation of (tanh(sqrt(z))/sqrt(z)-1)/z, |t|<=0.75.
# The coefficients are F32 values.
def tanh_small(+t: F32,+z: F32) -> F32:
  (t + (t * z : F32) * (((((0.001909503829665482 * z + F32.neg(0.007933800108730793) : F32) * z + 0.021615320816636086 : F32) * z + F32.neg(0.05393586680293083) : F32) * z + 0.1333317905664444 : F32) * z + F32.neg(0.3333333134651184) : F32) : F32)

# One additional term for the expm1 correction in the tanh quotient.
def tanh_remainder(+r: F32) -> F32:
  ((((((0.00019890980911441147 * r + 0.0013933641603216529 : F32) * r + 0.00833331048488617 : F32) * r + 0.04166646674275398 : F32) * r + 0.1666666716337204 : F32) * r + 0.5 : F32) * (r * r : F32) : F32)

def tanh_exponential(+a: F32,+biased: F32,x: F32) -> F32:
  +n = (biased - 12582912.0 : F32)
  +r = (((a * 2.0 : F32) - n * 0.693359375 : F32) - n * F32.neg(0.00021219444005469058) : F32)
  +q = from_bits((1065353216 - U32.shln(F32.bits(biased),23n) : U32))
  +t = F32.min(0.75,a)
  +small = tanh_small(t,(t * t : F32))
  +large = (1.0 - 2.0 * (q / ((1.0 + q : F32) + (r + tanh_remainder(r) : F32) : F32) : F32) : F32)
  +mask = (0 - U32.shrn((F32.bits(a) - 1061158912 : U32),31n) : U32)
  tanh_finish(x,selected(mask,small,large))

def tanh_input(+a: F32,x: F32) -> F32:
  tanh_exponential(a,((a * 2.0 : F32) * 1.44269504088896341 + 12582912.0 : F32),x)

def tanh(+x: F32) -> F32:
  tanh_input(tanh_magnitude(x),x)

# Split 2^n at +64 for nonnegative inputs and -64 for negative inputs.
# The first product is always normal and exact; only the last product can
# underflow. The exponent words come directly from the biased integer n.
def exponential_scale(biased: F32,x: F32,p: F32) -> F32:
  +sign = U32.shrn(U32.and(F32.bits(x),2147483648),1n)
  ((p * from_bits((1602224128 - sign : U32)) : F32) *
    from_bits((528482304 + U32.shln(F32.bits(biased),23n) + sign : U32)) : F32)

# FastTwoSum retains the low bits of 1+r before adding the quadratic tail.
def exponential_precise(+r: F32) -> F32:
  +sum = (1.0 + r : F32)
  +error = (r - (sum - 1.0 : F32) : F32)
  (sum + ((exponential_polynomial(r) * (r * r : F32) : F32) + error : F32) : F32)

def exponential_reduced(+x: F32,+biased: F32) -> F32:
  +n = (biased - 12582912.0 : F32)
  +high = (x - n * 0.693359375 : F32)
  +low = (n * F32.neg(0.00021219444005469058) : F32)
  +r = (high - low : F32)
  exponential_scale(biased,x,exponential_precise(r))

def exponential_input(+x: F32) -> F32:
  exponential_reduced(x,(x * 1.44269504088896341 + 12582912.0 : F32))

def exp(x: F32) -> F32:
  exponential_input(F32.max(F32.min(89.0,x),F32.neg(104.0)))

# For s near r/(2+r), d = r-(2+r)*s gives the exact real identity
# log(1+r) = 2*atanh(s) + log(1+d/(1+s)). The odd series ends at s^11.
# Split r and s into 12-bit heads so their head product is exact; retain
# the low products in d. (1-s)*(1+s^2+s^4) approximates 1/(1+s).
# FastTwoSum retains the low bits when adding the exponent's ln(2) part.
def logarithm_series(+r: F32,+n: F32) -> F32:
  +s = (r / (2.0 + r : F32) : F32)
  +square = (s * s : F32)
  +r_high = from_bits(U32.and(F32.bits(r),4294963200))
  +r_low = (r - r_high : F32)
  +s_high = from_bits(U32.and(F32.bits(s),4294963200))
  +s_low = (s - s_high : F32)
  +product_low = ((r_high * s_low + r_low * s_high : F32) + r_low * s_low : F32)
  +correction = (((r - 2.0 * s : F32) - r_high * s_high : F32) - product_low : F32)
  +p = ((((0.09090909090909091 * square + 0.1111111111111111 : F32) * square + 0.14285714285714285 : F32) * square + 0.2 : F32) * square + 0.3333333333333333 : F32)
  +main = (2.0 * s : F32)
  +rest = ((main * square : F32) * p + correction * ((1.0 - s : F32) * (1.0 + square * (1.0 + square : F32) : F32) : F32) : F32)
  +high = (n * 0.693359375 : F32)
  +sum = (high + main : F32)
  +tail = (main - (sum - high : F32) : F32)
  (sum + (rest + (n * F32.neg(0.000212194440) + tail : F32) : F32) : F32)

# The exponent carry at mantissa word 0x3504f4 selects the same reduced
# interval as halving m above sqrt(2), without a floating-point selection.
def logarithm_normalized(x: F32,adjustment: F32) -> F32:
  +word = F32.bits(x)
  +adjusted = (word + 4913932 : U32)
  logarithm_series((from_bits((word - U32.and(adjusted,4286578688) + 1065353216 : U32)) - 1.0 : F32),
    ((from_bits(U32.or(U32.and(U32.shrn(adjusted,23n),255),1258291200)) - 8388735.0 : F32) - adjustment : F32))

# Scaling a subnormal by 2^23 is exact. Zero, negatives, infinities and
# NaNs are selected explicitly, after normalizing the finite positive input.
def logarithm_finish(+x: F32,value: F32) -> F32:
  +word = F32.bits(x)
  +magnitude = U32.and(word,2147483647)
  +zero = (0 - U32.shrn((magnitude - 1 : U32),31n) : U32)
  +negative = (0 - U32.shrn(word,31n) : U32)
  +special = U32.not((0 - U32.shrn((magnitude - 2139095040 : U32),31n) : U32))
  selected(zero,from_bits(4286578688),selected(negative,from_bits(2143289344),selected(special,(x + x : F32),value)))

def logarithm_input(+x: F32) -> F32:
  +small = (0 - U32.shrn((U32.and(F32.bits(x),2147483647) - 8388608 : U32),31n) : U32)
  logarithm_finish(x,logarithm_normalized(selected(small,(x * 8388608.0 : F32),x),from_bits(U32.and(small,1102577664))))

def log(x: F32) -> F32:
  logarithm_input(x)
