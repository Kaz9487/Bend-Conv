# A network is a function: two convolution blocks, then the mean of each
# channel. Weights are tensors passed as arguments; this one fills them with
# constants so that the program needs no files.
import Base
import ../lib/tensor_f32.bend as F
import ../lib/math_activation.bend as Act

# 3x3 convolution with ReLU that keeps the size, then 2x2 max pooling.
def block(input: F.Tensor(),weights: F.Tensor(),bias: F.Tensor()) -> F.Tensor():
  F.max_pool2d(F.conv2d(input,weights,bias,1,1,Act.relu()),2)

# [1,1,8,8] -> [1,4,4,4] -> [1,8,2,2] -> [1,8]
def network(image: F.Tensor()) -> F.Tensor():
  first  = block(image,F.full([4,1,3,3],0.1),F.zeros([4]))
  second = block(first,F.full([8,4,3,3],0.01),F.ones([8]))
  F.mean_axis(F.mean_axis(second,3),2)

def main() -> List<&2,F32>:
  F.to_list_or(network(F.reshape(F.arange(0.0,64),[1,1,8,8])),[])
