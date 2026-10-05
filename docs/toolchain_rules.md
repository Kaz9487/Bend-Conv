# Bend toolchain rules

[English](toolchain_rules.md) | [繁體中文](toolchain_rules.zh-TW.md)

These are the rules of the official checker, compiler and runtime that this project's code and proofs depend on. They were read from the unmodified official Bend sources (`bend.ts`, `comp.ts`, `main.ts` and `base.bend` under `.tools/bend/bend2/`) and are confirmed by small programs. They were last reviewed for **Bend v2.0.35**.

Each rule is supported in up to three ways:

- **Source.** The passage that decides the rule. [anchors.json](../scripts/toolchain_rules/anchors.json) holds one exact string per rule, so a change in the sources is noticed.
- **Probe.** A small program in [probes.py](../scripts/toolchain_rules/probes.py) that the pinned checker or compiler must treat as recorded. A rule marked *(probe: name)* has one.
- **Inference.** A rule marked *(inferred)* follows from the source and has no probe.

## Tools

| Command, from the repository root | Use |
| --- | --- |
| `python -B scripts/toolchain_rules/lint.py FILE_OR_DIR ...` | Before a checker run: scan Bend sources for spellings the checker rejects. It only reads files and takes seconds |
| `python -B scripts/toolchain_rules/verify.py` | After changing the Bend pin: check the anchors, the checker's list of error messages and every probe. It prints `RULES HOLD` or each difference |
| `python -B scripts/toolchain_rules/verify.py --snapshot` | After the rules were reread and corrected for a new version: record that version and its messages |
| `python -B scripts/toolchain_rules/sweep.py OUT.json` | After an upgrade, when the full proof run fails: check files bottom-up and report those that fail on their own. It starts the checker once per file and is slow |
| `python -B scripts/toolchain_rules/fork_flags.py GENERATED.c NAME ...` | List which tasks in generated C are fork-free (rule H4) |
| `python -B benchmarks/model/timeline_claims.py TIMELINE_DIR` | Show how many workers are busy in each operation of a [model timeline](../benchmarks/model/README.md) |

A single file is checked with `node scripts/bend_launcher.mjs FILE --check-only`; it prints `ALL PROOFS CHECK` on success. [diagnostics.txt](../scripts/toolchain_rules/diagnostics.txt) lists every checker message.

## A. Checker: match

The body of a definition is compiled into a case tree (`match_flatten` in `bend.ts`). The checker keeps a list of parameters that are not yet lambdas. A match on the first of them destructs it and puts its fields at the front. A match on a later one first turns the parameters before it into lambdas, and those can no longer be matched.

- **A1** Match parameters, and the fields they produce, from left to right. After a later one is destructed, an earlier one cannot be: `can't be matched in this position`. The limit is the order, not the number of scrutinees. *(probes: `match_order_bad`, `match_order_ok`)*
- **A2** After a let, including `+x = ...`, no outer parameter can be matched. Write all parameter matches before the first let. *(probe: `match_after_let`)*
- **A3** A computed value cannot be matched: `a match cannot scrutinize a computed value`. Write a helper def that takes the value as a parameter. *(probe: `match_computed`)*
- **A4** A let-bound local cannot be matched: `cannot scrutinize a local binder`. Use a helper def as in A3. *(probe: `match_local`)*
- **A5** A value that is already a constructor cannot be matched again: `an undestructed scrutinee`. Bind its fields directly, or fold the pattern into the outer case.
- **A6** Live code cannot match an erased (`-`) parameter: `a live scrutinee`. Make the parameter live, or match a live value that determines it. *(probe: `erased_scrutinee`)*
- **A7** Write list patterns as `Con{head,tail}` and `Nil{}`. `tail []` parses as an array read.

## B. Checker: usage

A variable has one of three quantities: `-` (none), unmarked (once) and `+` (many). A live variable is used at most once. Type positions, `-` arguments, both sides of an equation and the motive of a rewrite are checked as dead code and do not count as uses.

- **B1** An unmarked value used twice fails with `x (consumed more than once)`. The usual causes are using a whole parameter after it was destructed, and a local alias that already consumed the value. *(probes: `use_twice`, `use_twice_plus`)*
- **B2** The type of `+x` must be Data: `its type must be Data`. Function types, Array and Buffer are not Data. *(probe: `plus_needs_data`)*
- **B3** Passing an argument to a `+` parameter counts as one use, so a once-usable Data value may be passed to a `+` parameter.
- **B4** A pattern binding has the field's quantity times the scrutinee's. A `+` field of a node matched once is still usable many times.
- **B5** An equation `{a == b : T}` is Data, because its evidence is erased. A proof term can be bound with `+` and reused.

## C. Checker: templates, laws and recursion

- **C1** `~k: T` may appear only at the front of a parameter list. *(probe: `tilde_not_leading`)*
- **C2** A call writes `~` only on parameters the callee declares with `~`. `f(~T, ...)` on a plain parameter fails in the parser with `expected : a term / observed : '~'`, often reported on a nearby line. A law declares its `~` parameters with leading `for ~name: T` lines, and a call writes `~` on them in the same way. *(probes: `tilde_on_plain_call`, `law_template_ok`)*
- **C3** A template's `~` arguments must be closed: `a template applied to closed ~ arguments`. The instance is created and checked at its first live call. A template may instantiate itself at most 64 levels deep.
- **C4** A self-call must decrease. The arguments are compared with the definition's parameters from left to right: each is passed unchanged until one is smaller; erased parameters are skipped. Otherwise `a decreasing self-call`. Non-tail recursion is accepted. *(probes: `non_decreasing`, `non_tail_recursion_ok`)*
- **C5** Live code cannot refer to a def that has no body yet. So there is no mutual recursion and no forward reference: `a filled definition (an unfilled law is a dead claim ...)`.
- **C6** A law is visible but stuck until it is filled; afterwards it unfolds. Declaring a law first does not keep it opaque later.
- **C7** A stuck call is canonical: two of them are compared by the name of the def and then argument by argument.

## D. Checker: rewrite and type comparison

- **D1** In `%e : P; f` with `e : {a == b : T}`, **the current goal must be `P[_ := b]` and the body `f` proves `P[_ := a]`**. `_` may occur several times. For the other direction use `Equal.sym`. *(probes: `rewrite_direction_ok`, `rewrite_direction_bad`)*
- **D2** Types are compared by `term_compare`: weak-head normalize, then compare; functions are compared up to eta.
- **D3** Two spellings that mean the same but normalize differently need an equation lemma. For large terms, `Equal.cong` with a lambda is more stable than a long motive.
- **D4** Write `+x = {Ctor{...} : Type}`. A constructor needs a target type to be checked.
- **D5** Write operators as `(a + b : T)`, put spaces around comparisons, and parenthesize a compound first type argument of `X<A, B>`.

## E. Checker: Base definitions and large numbers

- **E0** Base definitions change between versions. Since v2.0.33, `U32.min`, `U32.max`, `U32.div`, `U32.mod`, `F32.min` and `F32.max` match both arguments before they unfold, so a call on abstract variables stays stuck in a proof. Destruct both arguments into `U32{..}` or `F32{..}` first, or use the lemmas in `lib/proofs/u32_extrema.bend` and `lib/proofs/u32_division.bend`. `Nat.div.fin` and `Nat.mod.fin` were removed; use `Pair.fst(Nat,Nat,..)` and `Pair.snd(Nat,Nat,..)` on `Nat.divmod`.

A literal is one node that unfolds one layer at a time when it is read. A U32 is a structure of 32 bits and is small. A Nat is unary, so any computation that walks a whole Nat costs time proportional to its value.

| Case | Result | Probe |
| --- | --- | --- |
| `Nat.is_le(10000000n, 20000000n) == True` by `{==}` | Passes in about a second | `nat_compute_10000000` |
| `U32.is_le(1073741824, 2147483648) == True` by `{==}` | Passes | `u32_compute_large` |
| `Nat.is_le(1n, U32.to_nat(1073741824)) == True` by `{==}` | Passes; one layer is unfolded | `to_nat_large_compare` |
| Both sides are `U32.to_nat(100000)` | Passes in about two seconds | `to_nat_equal_100000` |
| A type contains a closed `U32.to_nat(1073741824)` and is compared with an **identically spelled** type | Passes since v2.0.34: the two are compared before any def is unfolded. It overflowed the stack in v2.0.32 | `to_nat_symbolic_bound` |
| The same, with the other side spelled as the literal `1073741824n` | **Stack overflow** | `to_nat_mixed_literal` |
| `Nat.is_le` on two Nat literals of 2^30 | Does not finish in 60 seconds | none; tried by hand |

- **E1** Keep closed, computable large Nats out of types: for example `U32.to_nat(1073741824)`, or `U32.to_nat(Machine.capacity(default))` after unfolding. When the two sides are spelled differently, the comparison normalizes both and its recursion depth is proportional to the value. Above about 10^5 it may overflow; at 2^30 it does, and the message gives no location.
- **E2** Keep such a quantity **abstract**. Make it a parameter (`limits: Machine.Limits`, `capacity: U32`), use it only through equations and inequalities about it, and substitute the actual value in the last step. `count_within` in `lib/proofs/bounded_u32_arithmetic.bend` and `lib/proofs/checked_word_completion.bend` are examples.
- **E3** Compare concrete numbers on U32 (`U32.is_le`), not after converting them to Nat.
- **E4** Small Nat literal computations, up to about 10^7, can use `{==}`; their time is proportional to the value.

## F. Compiler: the shape of a compiled def

How to read the generated C:

- `spin_N`: a native function. A def that exists only in this form has no `FID_NAME` of its own.
- `FID_NAME` with `WL_CASE(FID_NAME)`: a worklist segment.
- `FID_NAME_K<n>`: the continuation after a non-tail call. `FID_NAME_J<n>`: the join of a fork. `FID_NAME_C<n>`: a closure.
- `WL_AGAIN(FID_NAME)`: a tail self-jump inside a segment.

Rules:

- **F1** A def is **flat** when all of these hold: its body has no parallel let (one let binding two or more names); it has no bang call `f!(..)`; it calls itself only in tail position; and every def it calls directly is flat. *(probe: `shapes`)*
- **F2** A flat def compiles to a native C function `spin_N` whose parameters are flattened machine words. Its tail self-call is a loop, and its result is written to the output array `o[]`. Any other def runs on the worklist. *(probe: `shapes`)*
- **F3** Being non-flat spreads to callers, not to callees. *(probe: `shapes`: `mixed` calls a non-tail-recursive def and gets `FID_MIXED`; the flat `wrapper` it also calls stays a `spin`)*
- **F4** `SPIN_FAR = 256`: a `spin_N` with fewer than 256 lines of generated C is `INLINE`; a longer one is `FAR`, a real function call whose result returns through memory. *(probes: `inline_small`, `far_large`)*
- **F5** `FOLD_FUEL = 8192` bounds the nodes that unfolding may add within one segment.
- **F6** **A def called directly by a parallel let becomes a task**, even when it is flat. It gets its own worklist segment, and its tail recursion becomes a self-jump in that segment, not a loop in a `spin`. So a hot loop should not be the direct child of a fork; put it in a def one level below. *(probe: `shapes`: `kid` has `FID_KID` and `WL_AGAIN(FID_KID)`)*
- **F7** `INLINE` is a request to the C compiler, not a guarantee. A chain of small `spin_N` functions that pass a wide state through `o[]` is merged into one loop only while the C compiler chooses to inline every link, and it stops doing so when the chain gains more call sites. So a per-element helper shared by several operations can cost several real calls per element in a full program and nothing in a program that links one operation. Keep the per-element path a small loop of its own and step wide state once per run; measure inside a program that links the other users. *(measured, no probe: the element cursor of a strided copy took 1.3 ns per element alone and 6.7 ns once combine and reduce were linked; Clang 19 recovered it only with both inline thresholds at 20000, and was slower at 5000)*

## G. Compiler: layout and sharing

- **G1** Layout (`lay_of`). A non-recursive data type is flattened into machine words and allocates nothing; U32 and F32 take 32 bits, Nat 64. `Array`, `IO.OP` and recursive types (lists, trees) are heap boxes. A type wider than `WIDE = 247` words becomes a box; when the parameters together exceed it, multi-word parameters become boxes.
- **G2** Sharing (`facts_hot`). A value used more than once marks its type hot. That type's constructors become shared, a match on it goes through `ctr_take`, which checks the reference count, and the heat spreads to the field types. *(probes: `share_none` shares nothing; `share_list`, which uses a list twice, shares some constructors)* The project met this with `Views.View`: a view that one definition handed to two others, and a view carried through a loop as a parameter, each made every `View` constructor shared. With the first, the four-thread model measured 136.7 against 143.2 ms and 132.1 against 145.3 ms in two alternating comparisons (interquartile range about 25 ms); after building the view once per use it measured 130.5 against 130.9 ms. `scripts/check_generated_c.py` rejects a shared `View` in the model, the benchmarks and the integration entries; the check the launcher runs on every compilation leaves it out, because a program's own definitions may share a view.
- **G3** Global sharing. The source has a branch that marks every constructor shared (`fl.hot.add("*")`). The project met it once, when convolution became 16–24% slower; `scripts/check_generated_c.py` guards against it. **The probes do not reproduce it**: sharing a value whose type is a type parameter (`share_live_parameter`, `share_template_parameter`), or a type function defined by match and applied to a runtime index (`share_family_match_live_index`), shares only some constructors. The smallest trigger is not known. Until it is, rely on `check_generated_c.py` and the `N of M constructors shared` line that every build prints.
- **G4** A binding counts its uses: the last use takes the value and earlier uses share it. Sharing a flattened value copies words, not nodes.
- **G5** Templates and closures. A `~` template argument is instantiated by the checker, so the compiler sees a direct call. *(probe: `call_template`: the whole chain is a `spin`)* A function passed as an ordinary parameter is a closure, and **the def that calls a closure is not flat**. *(probe: `call_closure`)* A function stored in a record was measured 5.85–6.5 times slower than a template.
- **G6** A generic `~Context: Data` with `+context` heats only the instantiated type. *(probe: `share_template_parameter`)*

## H. Runtime: scheduling

- **H1** A fork (parallel let) creates a join task and one kid per call. When the kids run in sequence, one frame serves all steps.
- **H2** Workers rotate over rings. A lane in a grow turn skips a task that is fork-free (`fid_nofk` in `monk_step`). The host grows one ring after another and then drains them; since v2.0.35 the host thread also takes units in every turn, next to `threads - 1` pool threads, and a thread yields a bounded number of times before it sleeps (`POOL_WAIT`). There is no work stealing. One parallel step is therefore one grow turn plus one work turn, and the grow turn is not meant to finish leaves; a planner must count whole turns.
- **H3** An Array passed through a fork is a redirect handle with an atomic count. `Array.clone` copies the whole block (`blk_copy`).
- **H4** **Leaf work must be a fork-free task. Otherwise four leaves use two workers.** The mechanism, in `cube_run`, `row_grow`, `monk_step`, `task_deal` and `emit_jump`:
  1. At the top fork, the host puts the two kids on **two different rows** (`ring_flip`).
  2. In the grow turn, one worker owns a row. It runs the tasks that may fork; the kids they fork stay in **the same row** (`ring_pick`).
  3. If a kid is the forking recursion itself and the leaf is a flat def inlined into it, that worker runs **one whole leaf inside the grow turn**. After it finishes that task, the row's grow turn ends.
  4. In the work turn, the 128 rings of a row are divided into units (`pool_step`: more units when few rows are in use, eight when many), and each thread claims the next unit with an atomic counter. The remaining leaves run here.

  A tree of depth two with four leaves thus runs two leaves in the grow turn on two workers and two in the work turn on two workers. The total is the time of two leaves, not one. With 16 leaves it is two in the grow turn and 14 in the work turn.

  The remedy: where a child is a leaf, the parallel let calls a def that **neither forks nor calls a closure**. By F6 it becomes a task, and its row in the `FID_T` table has bit 2 set (`fork_flags.py` prints this). The grow turn skips it, so all leaves run in the work turn in different units. The hot loop goes in a flat def one level below (F6). A segment that calls a closure counts as one that may fork, so the leaf task must not contain a lambda.

  *(probes: `leaf_inline` and `leaf_task`, four leaves each. Measured on one 16-thread machine: 268 and 266 ms at one thread, 170 and 172 ms at two, and **171 ms against 121 ms** at four. The inlined form is no faster at four threads than at two.)*

## I. What this means for design

- **Skeletons** that turn a tree recursion into a map or zip are never flat (F1). That is acceptable, because a skeleton runs once per node. Pass the leaf as a `~` template parameter (G5) and keep it flat (F1); by F3 the skeleton does not slow it. Where a child is a leaf, call a fork-free task (H4), with the hot loop one level below it (F6). Do not let a Buffer, a certificate or a function value be shared (G2, G3).
- **When merging code**, compare the generated C before and after: whether the leaf's `spin_N` is `INLINE` or `FAR`, how often `ctr_take` appears, and how many constructors are shared. Timing is the final confirmation.
- **In proofs**, keep large machine bounds abstract (E1, E2).

## J. What lint.py finds

| Rule | Finding |
| --- | --- |
| A2 | A match or destructuring after a let |
| A3 | A match or destructuring of a computed value |
| A7 | A pattern followed by `[]` |
| C1 | A `~` parameter that is not leading |
| C2 | A call whose number of `~` arguments differs from the callee's declaration, resolved through import aliases |
| D4 | A let of a constructor without a type annotation |
| E1 | `U32.to_nat(large number)`, or a Nat literal of six or more digits |
| E1? | `U32.to_nat(f(X.limits()))`: a default machine limit converted to Nat. The question mark means a risk, not a certain failure |

On the accepted sources of `lib` and `convolution` it reports three findings: `Nat.double(2147483648n)` in `lib/kernel_planning.bend`, and two `E1?` in `lib/proofs/tensor_band_upsample_nearest.bend`. It looks at spelling only, so it finds neither usage errors (B1) nor type mismatches, and it skips C2 for a file whose imports do not resolve.
