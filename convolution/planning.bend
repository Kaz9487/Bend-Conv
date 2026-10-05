# Resource policy, not a correctness guard. The structural cost model is
# lib/kernel_planning.bend; its machine profile is lib/default_kernel_cost.bend.
import Base
import ./shape.bend as ConvShape
import ../lib/kernel_planning.bend as KernelPlanning
import ../lib/default_kernel_cost.bend as DefaultCost
import ../lib/math_activation.bend as Activations

# The Array engine's leaves only store; inference applies its activation after
# the gather, outside the planned partition.
def work(shape: ConvShape.Shape,owners: KernelPlanning.Owners) -> KernelPlanning.Work:
  ConvShape.Shape{+input_channels,channels,height,width,+kernel,stride,padding,+output_height,+output_width} = shape
  KernelPlanning.Work{(output_height * output_width : U32),(input_channels * kernel * kernel : U32),channels,owners,
    KernelPlanning.Geometry{input_channels,height,width,kernel,stride,padding,output_width},Activations.Identity{}}

def plan(shape: ConvShape.Shape,worker_limit: U32,workspace_budget: Maybe<&2,Nat>,owners: KernelPlanning.Owners) -> KernelPlanning.Plan:
  KernelPlanning.select(work(shape,owners),worker_limit,workspace_budget,DefaultCost.configuration())
