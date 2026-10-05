# Keep the native allocation extent constant at each call site. With a dynamic
# depth, Clang 19 retains a scalar initialization loop with a shift per element;
# constant depths allow memset or vectorized fills. Dispatch once per buffer,
# covering every power of two below the native depth limit, independently of
# model shapes. The fallback preserves Storage.allocate for larger depths.
import Base
import ./storage_buffer.bend as Storage

def dispatch(~Element: Data,~allocator: Nat -> Element -> Nat -> Storage.Buffer<Element>,depth: Nat,+value: Element,length: Nat) -> Storage.Buffer<Element>:
  match depth:
    case 0n: allocator(0n,value,length)
    case 1n: allocator(1n,value,length)
    case 2n: allocator(2n,value,length)
    case 3n: allocator(3n,value,length)
    case 4n: allocator(4n,value,length)
    case 5n: allocator(5n,value,length)
    case 6n: allocator(6n,value,length)
    case 7n: allocator(7n,value,length)
    case 8n: allocator(8n,value,length)
    case 9n: allocator(9n,value,length)
    case 10n: allocator(10n,value,length)
    case 11n: allocator(11n,value,length)
    case 12n: allocator(12n,value,length)
    case 13n: allocator(13n,value,length)
    case 14n: allocator(14n,value,length)
    case 15n: allocator(15n,value,length)
    case 16n: allocator(16n,value,length)
    case 17n: allocator(17n,value,length)
    case 18n: allocator(18n,value,length)
    case 19n: allocator(19n,value,length)
    case 20n: allocator(20n,value,length)
    case 21n: allocator(21n,value,length)
    case 22n: allocator(22n,value,length)
    case 23n: allocator(23n,value,length)
    case 24n: allocator(24n,value,length)
    case 25n: allocator(25n,value,length)
    case 26n: allocator(26n,value,length)
    case 27n: allocator(27n,value,length)
    case 28n: allocator(28n,value,length)
    case 29n: allocator(29n,value,length)
    case 30n: allocator(30n,value,length)
    case 31n+rest: allocator(31n+rest,value,length)

def allocated(~Element: Data,depth: Nat,value: Element,length: Nat) -> Storage.Buffer<Element>:
  Storage.allocate(~Element,depth,value,length)

def allocate(~Element: Data,depth: Nat,value: Element,length: Nat) -> Storage.Buffer<Element>:
  dispatch(~Element,~allocated(~Element),depth,value,length)

# Internal Array callers can discard the construction certificate after the
# same constant-depth dispatch; no traversal or clone is introduced.
def storage(~Element: Data,buffer: Storage.Buffer<Element>) -> Array<Element>:
  Storage.Buffer{storage,length,certificate} = buffer
  storage

def array(~Element: Data,depth: Nat,value: Element) -> Array<Element>:
  storage(~Element,allocate(~Element,depth,value,0n))
