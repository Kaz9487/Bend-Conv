"""Small Bend programs that pin down the rules in docs/toolchain_rules.md.

CHECKER maps a probe name to (expected, source). expected is "pass" or a
fragment of the message the official checker must print.
CODEGEN maps a probe name to (present, absent, shared, source): strings that
must and must not occur in the generated C, and whether "none", "some" or
"all" constructors are shared. verify.py writes each source under out/ after
an `import Base` line and runs the pinned compiler on it.
"""

# Entry shared by the code generation probes: it calls run(n) with a runtime value.
MAIN = '''
def pick(zero: Bool) -> String:
  match zero:
    case True{}: "zero"
    case False{}: "other"

def done(value: U32) -> IO(Unit):
  IO.print(pick(U32.is_zero(value)))

def main() -> IO(Unit):
  IO.bind(Nat, Unit, IO.now(), n => done(run(n)))
'''

CHECKER = {
    'match_order_bad': ("can't be matched in this position", '''
def f(a: Bool, b: Bool) -> Bool:
  match b:
    case True{}:
      match a:
        case True{}: True{}
        case False{}: False{}
    case False{}: False{}
'''),
    'match_order_ok': ('pass', '''
def f(a: Bool, b: Bool) -> Bool:
  match a:
    case True{}:
      match b:
        case True{}: True{}
        case False{}: False{}
    case False{}: False{}
'''),
    'match_after_let': ("can't be matched in this position", '''
def k(a: Bool, b: Bool) -> Bool:
  x = a
  match b:
    case True{}: x
    case False{}: x
'''),
    'match_computed': ('a match cannot scrutinize a computed value', '''
def flip(a: Bool) -> Bool:
  match a:
    case True{}: False{}
    case False{}: True{}

def g(a: Bool) -> Bool:
  match flip(a):
    case True{}: True{}
    case False{}: False{}
'''),
    'match_local': ('a match cannot scrutinize a local binder', '''
def flip(a: Bool) -> Bool:
  match a:
    case True{}: False{}
    case False{}: True{}

def h(a: Bool) -> Bool:
  x = flip(a)
  match x:
    case True{}: True{}
    case False{}: False{}
'''),
    'erased_scrutinee': ('a live scrutinee (a - scrutinee matches only in a dead region)', '''
def era(-a: Bool) -> Bool:
  match a:
    case True{}: True{}
    case False{}: False{}
'''),
    'use_twice': ('x (consumed more than once)', '''
def dup(x: Nat) -> Nat:
  Nat.add(x, x)
'''),
    'use_twice_plus': ('pass', '''
def dup(+x: Nat) -> Nat:
  Nat.add(x, x)
'''),
    'plus_needs_data': ('can be used many times, so its type must be Data', '''
def keep(+f: Nat -> Nat) -> Nat:
  f(0n)
'''),
    'tilde_not_leading': ('a plain binder (only leading binders take ~)', '''
def t(a: Nat, ~b: Nat) -> Nat:
  a
'''),
    'tilde_on_plain_call': ('expected : a term', '''
def same(A: Type, x: A) -> A:
  x

def use() -> Nat:
  same(~Nat, 1n)
'''),
    'law_template_ok': ('pass', '''
law same:
  for ~f: Nat -> Nat
  for +x: Nat
  {f(x) == f(x) : Nat}
def same(f, x):
  {==}

def keep(n: Nat) -> Nat:
  n

def use(+x: Nat) -> {keep(x) == keep(x) : Nat}:
  same(~keep, x)
'''),
    'non_decreasing': ('a decreasing self-call', '''
def spin(n: Nat) -> Nat:
  spin(n)
'''),
    'non_tail_recursion_ok': ('pass', '''
def count(n: Nat) -> Nat:
  match n:
    case 0n: 0n
    case 1n+p: 1n+count(p)
'''),
    'rewrite_direction_ok': ('pass', '''
def rw(+a: Nat, +b: Nat, e: {a == b : Nat}, base: {Nat.add(a, 0n) == a : Nat}) -> {Nat.add(b, 0n) == b : Nat}:
  %e : {Nat.add(_, 0n) == _ : Nat}
  base
'''),
    'rewrite_direction_bad': ('expected : {Nat.add(b, 0n) == b : Nat}', '''
def rw(+a: Nat, +b: Nat, e: {b == a : Nat}, base: {Nat.add(a, 0n) == a : Nat}) -> {Nat.add(b, 0n) == b : Nat}:
  %e : {Nat.add(_, 0n) == _ : Nat}
  base
'''),
    'nat_compute_10000000': ('pass', '''
def huge() -> {Nat.is_le(10000000n, 20000000n) == True{} : Bool}:
  {==}
'''),
    'u32_compute_large': ('pass', '''
def words() -> {U32.is_le(1073741824, 2147483648) == True{} : Bool}:
  {==}
'''),
    'to_nat_large_compare': ('pass', '''
def bound() -> {Nat.is_le(1n, U32.to_nat(1073741824)) == True{} : Bool}:
  {==}
'''),
    'to_nat_equal_100000': ('pass', '''
def same() -> {Nat.is_le(U32.to_nat(100000), U32.to_nat(100000)) == True{} : Bool}:
  {==}
'''),
    'to_nat_symbolic_bound': ('pass', '''
def bound(+x: Nat, room: {Nat.is_le(x, U32.to_nat(1073741824)) == True{} : Bool}) -> {Nat.is_le(x, U32.to_nat(1073741824)) == True{} : Bool}:
  room
'''),
    'to_nat_mixed_literal': ('the machine stack overflowed', '''
def mixed(+x: Nat, room: {Nat.is_le(x, U32.to_nat(1073741824)) == True{} : Bool}) -> {Nat.is_le(x, 1073741824n) == True{} : Bool}:
  room
'''),
}


def chain(steps):
    """A def with one let per step; enough steps make the native function FAR."""
    lets = ''.join(f'  x{step} = (x{step - 1} * 3 + {step} : U32)\n' for step in range(1, steps))
    return (f'\ndef body(a: U32) -> U32:\n  x0 = (a + 1 : U32)\n{lets}  (x{steps - 1} + 5 : U32)\n\n'
            'def run(n: Nat) -> U32:\n  body(U32.from_nat(n))\n')


CODEGEN = {
    'shapes': (['FID_NON_TAIL_K', 'define FID_MIXED ', 'FID_FORKS_J', 'define FID_KID ', 'WL_AGAIN(FID_KID)'], ['FID_TAIL_ONLY', 'FID_WRAPPER'], 'none', '''
def tail_only(n: Nat, acc: U32) -> U32:
  match n:
    case 0n: acc
    case 1n+p: tail_only(p, (acc + 1 : U32))

def wrapper(n: Nat) -> U32:
  (tail_only(n, 0) + 3 : U32)

def non_tail(n: Nat) -> U32:
  match n:
    case 0n: 0
    case 1n+p: (1 + non_tail(p) : U32)

def mixed(+n: Nat) -> U32:
  (wrapper(n) + non_tail(n) : U32)

def kid(n: Nat, acc: U32) -> U32:
  match n:
    case 0n: acc
    case 1n+p: kid(p, (acc + 2 : U32))

def forks(+n: Nat) -> U32:
  left right = kid(n, 0) kid(n, 1)
  (left + right : U32)

def run(+n: Nat) -> U32:
  (mixed(n) + forks(n) : U32)
''' + MAIN),
    'inline_small': (['INLINE Term spin_'], ['FAR Term spin_'], 'none', '''
def body(a: U32) -> U32:
  x0 = (a + 1 : U32)
  x1 = (x0 * 3 : U32)
  (x1 + 5 : U32)

def run(n: Nat) -> U32:
  body(U32.from_nat(n))
''' + MAIN),
    'far_large': (['FAR Term spin_'], [], 'none', chain(400) + MAIN),
    'share_none': ([], [], 'none', '''
def total(xs: List<&2,U32>, acc: U32) -> U32:
  match xs:
    case Nil{}: acc
    case Con{head,tail}: total(tail, (acc + head : U32))

def build(n: Nat, acc: List<&2,U32>) -> List<&2,U32>:
  match n:
    case 0n: acc
    case 1n+p: build(p, 7 <> acc)

def run(n: Nat) -> U32:
  total(build(n, []), 0)
''' + MAIN),
    'share_list': ([], [], 'some', '''
def total(xs: List<&2,U32>, acc: U32) -> U32:
  match xs:
    case Nil{}: acc
    case Con{head,tail}: total(tail, (acc + head : U32))

def build(n: Nat, acc: List<&2,U32>) -> List<&2,U32>:
  match n:
    case 0n: acc
    case 1n+p: build(p, 7 <> acc)

def both(+xs: List<&2,U32>) -> U32:
  (total(xs, 0) + total(xs, 1) : U32)

def run(n: Nat) -> U32:
  both(build(n, []))
''' + MAIN),
    'share_live_parameter': ([], [], 'some', '''
def total(xs: List<&2,U32>, acc: U32) -> U32:
  match xs:
    case Nil{}: acc
    case Con{head,tail}: total(tail, (acc + head : U32))

def build(n: Nat, acc: List<&2,U32>) -> List<&2,U32>:
  match n:
    case 0n: acc
    case 1n+p: build(p, 7 <> acc)

def twice(A: Data, +x: A) -> A & A:
  (x, x)

def first(pair: List<&2,U32> & List<&2,U32>) -> List<&2,U32>:
  (left, right) = pair
  left

def run(n: Nat) -> U32:
  total(first(twice(List<&2,U32>, build(n, []))), 0)
''' + MAIN),
    'share_template_parameter': ([], [], 'some', '''
def total(xs: List<&2,U32>, acc: U32) -> U32:
  match xs:
    case Nil{}: acc
    case Con{head,tail}: total(tail, (acc + head : U32))

def build(n: Nat, acc: List<&2,U32>) -> List<&2,U32>:
  match n:
    case 0n: acc
    case 1n+p: build(p, 7 <> acc)

def twice(~A: Data, +x: A) -> A & A:
  (x, x)

def first(pair: List<&2,U32> & List<&2,U32>) -> List<&2,U32>:
  (left, right) = pair
  left

def run(n: Nat) -> U32:
  total(first(twice(~List<&2,U32>, build(n, []))), 0)
''' + MAIN),
    'share_family_match_live_index': ([], [], 'some', '''
def total(xs: List<&2,U32>, acc: U32) -> U32:
  match xs:
    case Nil{}: acc
    case Con{head,tail}: total(tail, (acc + head : U32))

def build(n: Nat, acc: List<&2,U32>) -> List<&2,U32>:
  match n:
    case 0n: acc
    case 1n+p: build(p, 7 <> acc)

def Family(n: Nat) -> Data:
  match n:
    case 0n: List<&2,U32>
    case 1n+p: List<&2,U32>

def twice(+n: Nat, +x: Family(n)) -> Family(n) & Family(n):
  (x, x)

def first(-n: Nat, pair: Family(n) & Family(n)) -> Family(n):
  (left, right) = pair
  left

def cast(+n: Nat, xs: List<&2,U32>) -> Family(n):
  match n:
    case 0n: xs
    case 1n+p: xs

def back(+n: Nat, xs: Family(n)) -> List<&2,U32>:
  match n:
    case 0n: xs
    case 1n+p: xs

def run(+n: Nat) -> U32:
  total(back(n, first(n, twice(n, cast(n, build(n, []))))), 0)
''' + MAIN),
    'call_template': (['INLINE Term spin_'], ['define FID_RUN ', 'define FID_INCREMENT ', 'term_clo(FID_RUN_C'], 'none', '''
def increment(value: U32) -> U32:
  (value + 1 : U32)

def apply(~operation: U32 -> U32, value: U32) -> U32:
  operation(operation(value))

def run(n: Nat) -> U32:
  apply(~increment, U32.from_nat(n))
''' + MAIN),
    'call_closure': (['define FID_RUN ', 'term_clo(FID_RUN_C'], [], 'none', '''
def increment(value: U32) -> U32:
  (value + 1 : U32)

def apply(operation: U32 -> U32, value: U32) -> U32:
  operation(value)

def run(n: Nat) -> U32:
  apply(increment, U32.from_nat(n))
''' + MAIN),
    'leaf_task': (['define FID_LEAF ', 'define FID_TREE '], ['define FID_HEAVY '], 'none', '''
# The leaf work is a fork-free def called directly by the parallel let, so
# the runtime keeps each leaf as its own task for the work turn.
def heavy(n: Nat, acc: U32) -> U32:
  match n:
    case 0n: acc
    case 1n+p: heavy(p, (acc * 1664525 + 1013904223 : U32))

def leaf(n: Nat, seed: U32) -> U32:
  heavy(n, seed)

def tree(depth: Nat, +n: Nat, +seed: U32) -> U32:
  match depth:
    case 0n: heavy(n, seed)
    case 1n:
      left right = leaf(n, (seed * 2 : U32)) leaf(n, (seed * 2 + 1 : U32))
      (left + right : U32)
    case 2n+ +rest:
      left right = tree(1n+rest, n, (seed * 2 : U32)) tree(1n+rest, n, (seed * 2 + 1 : U32))
      (left + right : U32)

def depth_of(leaves: U32) -> Nat:
  match leaves:
    case 16: 4n
    case 8: 3n
    case _: 2n

def run(+now: Nat) -> U32:
  tree(depth_of(U32.from_nat(Nat.mod(now, 1n))), U32.to_nat(400000000), 1)

def main() -> IO(Unit):
  IO.bind(Nat, Unit, IO.now(), now => IO.print(U32.show(run(now))))
'''),
    'leaf_inline': (['define FID_TREE '], ['define FID_HEAVY ', 'define FID_LEAF '], 'none', '''
# The leaf work is a flat loop called inside the forking recursion.
def heavy(n: Nat, acc: U32) -> U32:
  match n:
    case 0n: acc
    case 1n+p: heavy(p, (acc * 1664525 + 1013904223 : U32))

def tree(depth: Nat, +n: Nat, +seed: U32) -> U32:
  match depth:
    case 0n: heavy(n, seed)
    case 1n+ +rest:
      left right = tree(rest, n, (seed * 2 : U32)) tree(rest, n, (seed * 2 + 1 : U32))
      (left + right : U32)

def depth_of(leaves: U32) -> Nat:
  match leaves:
    case 16: 4n
    case 8: 3n
    case _: 2n

def run(+now: Nat) -> U32:
  tree(depth_of(U32.from_nat(Nat.mod(now, 1n))), U32.to_nat(400000000), 1)

def main() -> IO(Unit):
  IO.bind(Nat, Unit, IO.now(), now => IO.print(U32.show(run(now))))
'''),
}
