# One clock assumption, 2.5 GHz (0.4 ns per cycle), prices both vector MACs
# and scalar operations. A 4x8 step issues four 256-bit multiplies and four
# adds for 32 MACs; with multiplies and adds on separate ports that is
# conservatively four cycles. Scalar operations counted from source are
# charged one cycle each: pool maxima and elementwise loops carry dependency
# chains, so the core's issue width is not assumed.
# Copy cost rounds the retained 0.090833 ns/byte upper quartile upward.
# See docs/design.md for instruction sources, assumptions and residuals.
import Base
import ./kernel_cost.bend as Cost

def cycle_ns() -> F32:
  0.4

def configuration() -> Cost.Configuration:
  # Prior task traces measured 40--180 us per wake round. Two rounds are
  # charged explicitly: the 250 us leaf grain alone cannot cover both maxima.
  Cost.Configuration{(4.0 / 32.0 * cycle_ns() : F32),0.100,cycle_ns(),250000.0,180000.0}
