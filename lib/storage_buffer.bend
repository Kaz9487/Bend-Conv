# Shared affine storage operations. Shape descriptors do not own another buffer.
import Base
import ./traversal_array.bend as Traversal
import ./proofs/storage_certificate.bend as Certificate

type Buffer<-Element: Data> is Type:
  Buffer{storage: Array<Element>,length: Nat,certificate: Certificate.Certificate<Element,storage>}

# Construction records evidence once, without visiting the owner or carrying
# certificates through the elementwise loop.
def allocate(~Element: Data,+depth: Nat,+value: Element,length: Nat) -> Buffer<Element>:
  Buffer{Array.new(Element,depth,value),length,Certificate.allocated(Element,depth,value)}

def from_list(~Element: Data,+values: List<&2,Element>,+depth: Nat,+fallback: Element,length: Nat) -> Buffer<Element>:
  Buffer{Traversal.write_list(~Element,values,0,Array.new(Element,depth,fallback)),length,
    Certificate.write_list(~Element,values,0,Array.new(Element,depth,fallback),Certificate.allocated(Element,depth,fallback))}

# The values written from the offset on; the certificate follows the storage.
def write_at(~Element: Data,+values: List<&2,Element>,+offset: U32,buffer: Buffer<Element>) -> Buffer<Element>:
  Buffer{storage,length,certificate} = buffer
  Buffer{Traversal.write_list(~Element,values,offset,storage),length,Certificate.write_list(~Element,values,offset,storage,certificate)}

def capacity(~Element: Data,-storage: Array<Element>,certificate: Certificate.Certificate<Element,storage>) -> U32:
  Certificate.Certificate{depth,equation} = certificate
  U32.shln(1,depth)

def bind_pair(~First: Type,~Second: Type,~Result: Type,pair: First & Second,body: First -> Second -> Result) -> Result:
  (first,second) = pair
  body(first,second)

# At most 32 word steps, independent of the allocation length.
def bit_length(remaining_bits: Nat,+value: U32) -> Nat:
  match remaining_bits:
    case 0n: 0n
    case 1n+rest:
      Bool.pick(Nat,U32.is_zero(value),0n,1n+bit_length(rest,U32.shr(value)))

# Positive requested lengths round to the next power-of-two capacity.
# The caller establishes the machine allocation limit before calling Array.new.
def allocation_depth(length: U32) -> Nat:
  bit_length(32n,(length - 1 : U32))

# Shared runtime depth calculation. The tail call stops at the significant
# word bits; allocation_depth remains the arithmetic model used by proofs.
def significant_bits(remaining: Nat,+value: U32,+bits: Nat,zero: Bool) -> Nat:
  match remaining zero:
    case 0n zero: bits
    case 1n+rest True{}: bits
    case 1n+rest False{}:
      +next = U32.shr(value)
      significant_bits(rest,next,1n+bits,U32.is_zero(next))

def depth_for(+count: U32) -> Nat:
  +value = (U32.max(count,1) - 1 : U32)
  significant_bits(32n,value,0n,U32.is_zero(value))

def write_list(~Element: Data,values: List<&2,Element>,+offset: U32,storage: Array<Element>) -> Array<Element>:
  Traversal.write_list(~Element,values,offset,storage)

def read_list(~Element: Data,count: Nat,+offset: U32,storage: Array<Element>) -> List<&2,Element>:
  match count:
    case 0n: []
    case 1n+rest:
      bind_pair(Array<Element>,Element,List<&2,Element>,Array.get(Element,storage,offset),storage => value =>
        value <> read_list(Element,rest,(offset + 1 : U32),storage))

def read_when(~Element: Data,fallback: Element,valid: Bool,storage: Array<Element>,index: U32) -> Array<Element> & Element:
  match valid:
    case False{}: (storage,fallback)
    case True{}: Array.get(Element,storage,index)

def read_sized(~Element: Data,fallback: Element,+index: U32,+logical_length: U32,sized: Array<Element> & U32) -> Array<Element> & Element:
  (storage,capacity) = sized
  read_when(Element,fallback,U32.is_lt(index,logical_length) && U32.is_le(logical_length,capacity),storage,index)

def read_or(~Element: Data,fallback: Element,storage: Array<Element>,index: U32,logical_length: U32) -> Array<Element> & Element:
  read_sized(Element,fallback,index,logical_length,Array.size(Element,storage))
