# Semantic contract v1: independent of native kernels.
# Nat coordinates, dense NCHW ordering, and an explicitly ordered F32 reduction.
import Base

type Shape is Data:
  Shape{ci: Nat, co: Nat, height: Nat, width: Nat, kernel: Nat, stride: Nat, pad: Nat}

type Problem is Data:
  Problem{shape: Shape, input: List<&2,F32>, weights: List<&2,F32>, bias: List<&2,F32>}

type Product is Data:
  Product{sample: F32, weight: F32}

type DotJob is Data:
  DotJob{products: List<&2,Product>, bias: F32}

type ConvOutcome is Data:
  Rejected{}
  Accepted{height: Nat, width: Nat, values: List<&2,F32>}

def length(xs: List<&2,F32>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case h <> tail:
      1n+length(tail)

# Total helper. Exact-length validation and coordinate checks prevent the empty
# fallback from defining the behavior of malformed input tensors.
def at(n: Nat, xs: List<&2,F32>) -> F32:
  match n xs:
    case 0n Nil{}:
      0.0
    case 0n h <> tail:
      h
    case 1n+rest Nil{}:
      0.0
    case 1n+rest h <> tail:
      at(rest,tail)

def axis_size(+size: Nat, +kernel: Nat, stride: Nat, +pad: Nat) -> Nat:
  1n+Nat.div(Nat.sub(Nat.add(size,Nat.mul(2n,pad)),kernel),stride)

def valid(p: Problem) -> Bool:
  Problem{shape,input,weights,bias} = p
  Shape{+ci,+co,+height,+width,+kernel,+stride,+pad} = shape
  Nat.is_lt(0n,ci) && Nat.is_lt(0n,co) && Nat.is_lt(0n,height) && Nat.is_lt(0n,width) && Nat.is_lt(0n,kernel) && Nat.is_lt(0n,stride) &&
  Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,pad))) && Nat.is_le(kernel,Nat.add(width,Nat.mul(2n,pad))) &&
  Nat.is_eq(length(input),Nat.mul(ci,Nat.mul(height,width))) &&
  Nat.is_eq(length(weights),Nat.mul(co,Nat.mul(ci,Nat.mul(kernel,kernel)))) &&
  Nat.is_eq(length(bias),co)

def channels(p: Problem) -> Nat:
  Problem{shape,input,weights,bias} = p
  Shape{ci,co,height,width,kernel,stride,pad} = shape
  ci

def outputs(p: Problem) -> Nat:
  Problem{shape,input,weights,bias} = p
  Shape{ci,co,height,width,kernel,stride,pad} = shape
  co

def kernel_size(p: Problem) -> Nat:
  Problem{shape,input,weights,bias} = p
  Shape{ci,co,height,width,kernel,stride,pad} = shape
  kernel

def out_height(p: Problem) -> Nat:
  Problem{shape,input,weights,bias} = p
  Shape{ci,co,height,width,kernel,stride,pad} = shape
  axis_size(height,kernel,stride,pad)

def out_width(p: Problem) -> Nat:
  Problem{shape,input,weights,bias} = p
  Shape{ci,co,height,width,kernel,stride,pad} = shape
  axis_size(width,kernel,stride,pad)

def sample_if(inside: Bool, index: Nat, input: List<&2,F32>) -> F32:
  match inside:
    case True{}:
      at(index,input)
    case False{}:
      0.0

def sample(p: Problem, ic: Nat, oy: Nat, ox: Nat, ky: Nat, kx: Nat) -> F32:
  Problem{shape,input,weights,bias} = p
  Shape{ci,co,+height,+width,kernel,+stride,+pad} = shape
  +py = Nat.add(Nat.mul(oy,stride),ky)
  +px = Nat.add(Nat.mul(ox,stride),kx)
  sample_if(Nat.is_le(pad,py) && Nat.is_lt(py,Nat.add(pad,height)) && Nat.is_le(pad,px) && Nat.is_lt(px,Nat.add(pad,width)),
    Nat.add(Nat.mul(ic,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(py,pad),width),Nat.sub(px,pad))),input)

def weight(p: Problem, oc: Nat, ic: Nat, ky: Nat, kx: Nat) -> F32:
  Problem{shape,input,weights,bias} = p
  Shape{ci,co,height,width,+kernel,stride,pad} = shape
  at(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(oc,ci),ic),kernel),ky),kernel),kx),weights)

def bias_at(p: Problem, oc: Nat) -> F32:
  Problem{shape,input,weights,bias} = p
  at(oc,bias)

def cat_products(xs: List<&2,Product>, ys: List<&2,Product>) -> List<&2,Product>:
  match xs:
    case Nil{}:
      ys
    case h <> tail:
      h <> cat_products(tail,ys)

# Explicit lexicographic enumeration: ic outermost, ky next, kx innermost.
def kernel_columns(n: Nat, +kx: Nat, +ky: Nat, +ic: Nat, +oc: Nat, +oy: Nat, +ox: Nat, +p: Problem) -> List<&2,Product>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      Product{sample(p,ic,oy,ox,ky,kx),weight(p,oc,ic,ky,kx)} <> kernel_columns(rest,1n+kx,ky,ic,oc,oy,ox,p)

def kernel_rows(n: Nat, +ky: Nat, +ic: Nat, +oc: Nat, +oy: Nat, +ox: Nat, +p: Problem) -> List<&2,Product>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      cat_products(kernel_columns(kernel_size(p),0n,ky,ic,oc,oy,ox,p),kernel_rows(rest,1n+ky,ic,oc,oy,ox,p))

def input_channels(n: Nat, +ic: Nat, +oc: Nat, +oy: Nat, +ox: Nat, +p: Problem) -> List<&2,Product>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      cat_products(kernel_rows(kernel_size(p),0n,ic,oc,oy,ox,p),input_channels(rest,1n+ic,oc,oy,ox,p))

def cat_jobs(xs: List<&2,DotJob>, ys: List<&2,DotJob>) -> List<&2,DotJob>:
  match xs:
    case Nil{}:
      ys
    case h <> tail:
      h <> cat_jobs(tail,ys)

# Output order: oc, then oy, then ox. No physical tile/lane is specified.
def output_columns(n: Nat, +ox: Nat, +oy: Nat, +oc: Nat, +p: Problem) -> List<&2,DotJob>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      DotJob{input_channels(channels(p),0n,oc,oy,ox,p),bias_at(p,oc)} <> output_columns(rest,1n+ox,oy,oc,p)

def output_rows(n: Nat, +oy: Nat, +oc: Nat, +p: Problem) -> List<&2,DotJob>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      cat_jobs(output_columns(out_width(p),0n,oy,oc,p),output_rows(rest,1n+oy,oc,p))

def output_channels(n: Nat, +oc: Nat, +p: Problem) -> List<&2,DotJob>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      cat_jobs(output_rows(out_height(p),0n,oc,p),output_channels(rest,1n+oc,p))

def jobs(+p: Problem) -> List<&2,DotJob>:
  output_channels(outputs(p),0n,p)

def multiply(xs: List<&2,Product>) -> List<&2,F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Product{x,w} <> tail:
      (x * w : F32) <> multiply(tail)

def sum(xs: List<&2,F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case value <> tail:
      sum(tail,(acc + value : F32))

def evaluate(jobs: List<&2,DotJob>) -> List<&2,F32>:
  match jobs:
    case Nil{}:
      Nil{}
    case DotJob{products,bias} <> tail:
      (sum(multiply(products),0.0) + bias : F32) <> evaluate(tail)

def finish(ok: Bool, +p: Problem) -> ConvOutcome:
  match ok:
    case False{}:
      Rejected{}
    case True{}:
      Accepted{out_height(p),out_width(p),evaluate(jobs(p))}

def reference(+p: Problem) -> ConvOutcome:
  finish(valid(p),p)

