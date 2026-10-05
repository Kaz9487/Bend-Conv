# Public tensor basics: build tensors, apply operations, read the values back.
# Operations consume their arguments; a failure travels with the result and
# is reported where the value is read.
import Base
import ../lib/tensor_f32.bend as F
import ../lib/math_activation.bend as Act

# A 4x4 image of 1 .. 16 through a 3x3 convolution of ones with ReLU, then
# 2x2 max pooling: [54, 63, 90, 99].
def main() -> List<&2,F32>:
  # 1.0 .. 16.0 as [batch, channels, height, width]
  image   = F.reshape(F.arange(1.0,16),[1,1,4,4])
  # [output channels, input channels, kernel height, kernel width]
  weights = F.ones([1,1,3,3])
  # one value per output channel
  bias    = F.zeros([1])
  # stride 1, padding 1, then ReLU: the size stays 4x4
  feature = F.conv2d(image,weights,bias,1,1,Act.relu())
  # 2x2 windows: 4x4 -> 2x2
  pooled  = F.max_pool2d(feature,2)
  # the values, or [] if any step above failed
  F.to_list_or(pooled,[])
