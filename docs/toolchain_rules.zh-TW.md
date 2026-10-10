# Bend 工具鏈規則

[English](toolchain_rules.md) | [繁體中文](toolchain_rules.zh-TW.md)

這裡整理本專案的程式與證明所依賴的官方 checker、編譯器與 runtime 規則。規則是從未修改的官方 Bend 原始碼讀出來的（`.tools/bend/bend2/` 下的 `bend.ts`、`comp.ts`、`main.ts`、`base.bend`），並以小程式確認。最近一次核對的版本是 **Bend v2.0.36**。

每條規則最多有三種依據：

- **原始碼**：決定該規則的程式段。[anchors.json](../scripts/toolchain_rules/anchors.json) 為每條規則存一段完全相同的字串，原始碼一改就會被發現。
- **Probe**：[probes.py](../scripts/toolchain_rules/probes.py) 裡的小程式，固定版本的 checker 或編譯器對它的處理必須和記錄一致。標示 *（probe：名稱）* 的規則有 probe。
- **推論**：標示 *（推論）* 的規則由原始碼推出，沒有 probe。

## 工具

| 指令（在 repository 根目錄執行） | 用途 |
| --- | --- |
| `python -B scripts/toolchain_rules/lint.py 檔案或資料夾 ...` | 送 checker 之前：掃描 Bend 原始碼中 checker 會拒絕的寫法。只讀檔案，幾秒就跑完 |
| `python -B scripts/toolchain_rules/verify.py` | 更換 Bend 版本之後：檢查錨點、checker 的錯誤訊息清單與每個 probe。印出 `RULES HOLD` 或逐項差異 |
| `python -B scripts/toolchain_rules/verify.py --snapshot` | 依新版本重讀並修正規則之後：記錄該版本與它的訊息清單 |
| `python -B scripts/toolchain_rules/sweep.py OUT.json` | 升級後完整證明失敗時：由下往上逐檔檢查，回報「自己壞掉」的檔。每個檔啟動一次 checker，所以很慢 |
| `python -B scripts/toolchain_rules/fork_flags.py GENERATED.c 名稱 ...` | 列出生成的 C 中哪些 task 不含 fork（規則 H4） |
| `python -B benchmarks/model/timeline_claims.py TIMELINE_DIR` | 顯示[模型時間軸](../benchmarks/model/README.zh-TW.md)中每個運算有幾個 worker 在工作 |

單一檔案用 `node scripts/bend_launcher.mjs FILE --check-only` 檢查；通過時印出 `ALL PROOFS CHECK`。[diagnostics.txt](../scripts/toolchain_rules/diagnostics.txt) 列出 checker 的全部訊息。

## A. checker：match

定義的本體會被編譯成一棵 case tree（`bend.ts` 的 `match_flatten`）。checker 保留一串「還沒變成 lambda 的參數」。match 這一串的第一個，會把它拆開，欄位放到最前面。match 後面的某一個，會先把它前面的參數變成 lambda，那些參數之後就不能再 match。

- **A1** 參數（以及拆出來的欄位）要由左到右依序 match。先拆了後面的，前面的就不能再拆：`can't be matched in this position`。限制是順序，不是 scrutinee 的個數。*（probe：`match_order_bad`、`match_order_ok`）*
- **A2** let（包括 `+x = ...`）之後，外層的參數都不能再 match。所有對參數的 match 要寫在第一個 let 之前。*（probe：`match_after_let`）*
- **A3** 不能 match 計算出來的值：`a match cannot scrutinize a computed value`。寫一個以該值為參數的輔助 def。*（probe：`match_computed`）*
- **A4** 不能 match let 綁定的區域變數：`cannot scrutinize a local binder`。做法同 A3。*（probe：`match_local`）*
- **A5** 已經是建構子的值不能再 match：`can't be matched (this value is already a constructor: bind its fields directly)`。直接綁它的欄位，或把 pattern 併進外層的 case。
- **A6** live 的程式不能 match erased（`-`）參數：`a live scrutinee`。把參數改成 live，或改 match 一個能決定它的 live 值。*（probe：`erased_scrutinee`）*
- **A7** list 的 pattern 寫成 `Con{head,tail}` 與 `Nil{}`。`tail []` 會被解析成讀取陣列元素。

## B. checker：用量

變數的用量有三種：`-`（不用）、不加標記（一次）、`+`（多次）。live 的變數最多用一次。型別位置、`-` 引數、等式的兩端、rewrite 的 motive 都當成 dead 程式檢查，不算用量。

- **B1** 沒有標記的值用兩次，會報 `x (consumed more than once)`。常見原因有兩種：參數被拆開之後又整個使用；區域別名已經把值用掉。*（probe：`use_twice`、`use_twice_plus`）*
- **B2** `+x` 的型別必須是 Data：`its type must be Data`。函式型別、Array、Buffer 都不是 Data。*（probe：`plus_needs_data`）*
- **B3** 把引數傳給 `+` 參數只算一次使用，所以只能用一次的 Data 值可以傳給 `+` 參數。
- **B4** pattern 綁定的用量等於欄位的用量乘上 scrutinee 的用量。只 match 一次的節點，它的 `+` 欄位仍然可以用多次。
- **B5** 等式 `{a == b : T}` 是 Data，因為它的證據會被抹除。證明項可以用 `+` 綁定並重複使用。

## C. checker：樣板、law 與遞迴

- **C1** `~k: T` 只能出現在參數列的最前面。*（probe：`tilde_not_leading`）*
- **C2** 呼叫時只有被呼叫者宣告為 `~` 的參數才能寫 `~`。對一般參數寫 `f(~T, ...)` 會在解析階段失敗，報 `expected : a term / observed : '~'`，位置常常指在附近的行。law 用最前面的 `for ~name: T` 宣告 `~` 參數，呼叫時同樣寫 `~`。*（probe：`tilde_on_plain_call`、`law_template_ok`）*
- **C3** 樣板的 `~` 引數必須是封閉的：`a template applied to closed ~ arguments`。實例在第一次 live 呼叫時產生並檢查。樣板自我實例化最多 64 層。
- **C4** 自我呼叫必須遞減。引數由左到右與定義的參數比較：每個照原樣傳，直到某一個變小；erased 的參數會跳過。否則報 `a decreasing self-call`。非尾遞迴可以通過。*（probe：`non_decreasing`、`non_tail_recursion_ok`）*
- **C5** live 的程式不能引用還沒有本體的 def。所以沒有互相遞迴，也沒有往前引用：`a filled definition (an unfilled law is a dead claim ...)`。
- **C6** law 在填入之前可見但卡住（stuck），填入之後就會展開。先宣告 law，並不會讓它之後保持不透明。
- **C7** 卡住的呼叫是 canonical 的：比較時先比 def 的名字，再逐一比引數。

## D. checker：rewrite 與型別比較

- **D1** `%e : P; f` 中，設 `e : {a == b : T}`，**目前的目標必須是 `P[_ := b]`，body `f` 要證 `P[_ := a]`**。`_` 可以出現多次。另一個方向用 `Equal.sym`。*（probe：`rewrite_direction_ok`、`rewrite_direction_bad`）*
- **D2** 型別以 `term_compare` 比較：先化成 weak-head normal form 再比；函式之間考慮 eta。
- **D3** 兩個寫法意思相同但 normal form 不同時，需要一條等式引理。對大型的項，`Equal.cong` 加 lambda 比寫長的 motive 穩定。
- **D4** 寫成 `+x = {Ctor{...} : Type}`。建構子需要目標型別才能檢查。
- **D5** 運算子寫成 `(a + b : T)`；比較運算的兩側加空白；`X<A, B>` 的第一個型別引數如果是複合型別要加括號。

## E. checker：Base 的定義與大數

- **E0** Base 的定義會隨版本改變。從 v2.0.33 起，`U32.min`、`U32.max`、`U32.div`、`U32.mod`、`F32.min`、`F32.max` 先 match 兩個引數才展開，所以在證明裡對抽象變數的呼叫會卡住。先把兩個引數拆成 `U32{..}` 或 `F32{..}`，或使用 `lib/proofs/u32_extrema.bend` 與 `lib/proofs/u32_division.bend` 中的引理。`Nat.div.fin` 與 `Nat.mod.fin` 已被移除；改用 `Nat.divmod` 上的 `Pair.fst(Nat,Nat,..)` 與 `Pair.snd(Nat,Nat,..)`。

字面值是一個節點，被讀取時才一層一層展開。U32 是 32 個 bit 的結構，本身很小。Nat 是一元表示，所以任何走完整個 Nat 的計算，時間都和數值成正比。

| 情況 | 結果 | Probe |
| --- | --- | --- |
| `Nat.is_le(10000000n, 20000000n) == True` 用 `{==}` | 通過，約一秒 | `nat_compute_10000000` |
| `U32.is_le(1073741824, 2147483648) == True` 用 `{==}` | 通過 | `u32_compute_large` |
| `Nat.is_le(1n, U32.to_nat(1073741824)) == True` 用 `{==}` | 通過；只展開一層 | `to_nat_large_compare` |
| 兩邊都是 `U32.to_nat(100000)` | 通過，約兩秒 | `to_nat_equal_100000` |
| 型別中有封閉的 `U32.to_nat(1073741824)`，並與另一個**寫法完全相同**的型別比較 | v2.0.34 起通過：展開任何 def 之前先比較兩者。v2.0.32 會堆疊溢位 | `to_nat_symbolic_bound` |
| 同上，但另一邊寫成字面值 `1073741824n` | **堆疊溢位** | `to_nat_mixed_literal` |
| 對兩個 2^30 的 Nat 字面值做 `Nat.is_le` | 60 秒內跑不完 | 無；手動試過 |

- **E1** 不要讓封閉、可以被算出來的大 Nat 出現在型別裡：例如 `U32.to_nat(1073741824)`，或展開後的 `U32.to_nat(Machine.capacity(default))`。兩邊寫法不同時，比較會把兩邊都正規化，遞迴深度和數值成正比。大約 10^5 以上就可能溢位；2^30 一定溢位，而且訊息不給位置。
- **E2** 讓這種量保持**抽象**。把它寫成參數（`limits: Machine.Limits`、`capacity: U32`），只透過關於它的等式與不等式使用，最後一步才代入實際的值。`lib/proofs/bounded_u32_arithmetic.bend` 的 `count_within` 與 `lib/proofs/checked_word_completion.bend` 是例子。
- **E3** 具體數字的大小比較在 U32 上做（`U32.is_le`），不要轉成 Nat 再比。
- **E4** 小的 Nat 字面值計算（到大約 10^7）可以用 `{==}`；時間和數值成正比。

## F. 編譯器：一個 def 編譯出來的形狀

生成的 C 怎麼讀：

- `spin_N`：原生函式。只以這種形式存在的 def，沒有自己的 `FID_名稱`。
- `FID_名稱` 加上 `WL_CASE(FID_名稱)`：worklist 的 segment。
- `FID_名稱_K<n>`：非尾端呼叫之後的接續。`FID_名稱_J<n>`：fork 的匯合。`FID_名稱_C<n>`：閉包。
- `WL_AGAIN(FID_名稱)`：segment 內的尾端自我跳轉。

規則：

- **F1** 一個 def 是 **flat**，要同時滿足：本體沒有平行 let（一個 let 綁定兩個以上的名字）；沒有 bang 呼叫 `f!(..)`；只在尾端呼叫自己；它直接呼叫的每個 def 也都是 flat。*（probe：`shapes`）*
- **F2** flat 的 def 編譯成原生 C 函式 `spin_N`，參數是攤平的機器字。尾端自我呼叫是迴圈，結果寫到輸出陣列 `o[]`。其他的 def 都在 worklist 上執行。*（probe：`shapes`）*
- **F3** 不 flat 會往呼叫者傳，不會往被呼叫者傳。*（probe：`shapes`：`mixed` 呼叫了一個非尾遞迴的 def，所以有 `FID_MIXED`；它同時呼叫的 flat `wrapper` 仍然是 `spin`）*
- **F4** `SPIN_FAR = 256`：生成的 C 少於 256 行的 `spin_N` 是 `INLINE`；更長的是 `FAR`，也就是真正的函式呼叫，結果經記憶體傳回。*（probe：`inline_small`、`far_large`）*
- **F5** `FOLD_FUEL = 8192` 限制一個 segment 內展開最多能新增的節點數。
- **F6** **被平行 let 直接呼叫的 def 會變成 task**，即使它是 flat。它得到自己的 worklist segment，尾遞迴變成 segment 內的自我跳轉，不是 `spin` 裡的迴圈。所以熱迴圈不要直接當 fork 的子呼叫；把它放在下一層的 def 裡。*（probe：`shapes`：`kid` 有 `FID_KID` 與 `WL_AGAIN(FID_KID)`）*
- **F7** `INLINE` 是對 C 編譯器的請求，不是保證。一串以 `o[]` 傳遞寬狀態的小 `spin_N`，只有在 C 編譯器願意把每一環都 inline 時才會合成一個迴圈；呼叫點變多，它就不再這樣做。所以多個運算共用的逐元素輔助函式，在完整程式裡可能每個元素花好幾次真正的呼叫，在只連結一個運算的程式裡卻沒有成本。逐元素的路徑要是一個自己的小迴圈，寬狀態每段只前進一次；量測要在連結了其他使用者的程式裡做。*（量測，無探針：非連續複製的逐元素游標單獨是每元素 1.3 ns，連結 combine 與 reduce 後是 6.7 ns；Clang 19 只有在兩個 inline 門檻都設為 20000 時才恢復，設為 5000 反而更慢）*

## G. 編譯器：版面與共享

- **G1** 版面（`lay_of`）。非遞迴的資料型別攤平成機器字，不配置記憶體；U32 與 F32 佔 32 位元，Nat 佔 64 位元。`Array`、`IO.OP` 與遞迴型別（list、樹）是 heap 上的 box。寬度超過 `WIDE = 247` 個字的型別變成 box；參數合計超過時，多字的參數變成 box。
- **G2** 共享（`facts_hot`）。一個值被使用超過一次，它的型別就被標成 hot。該型別的建構子變成共享的，對它的 match 改走會檢查引用計數的 `ctr_take`，而且 hot 會傳到欄位的型別。*（probe：`share_none` 沒有共享；`share_list` 把一個 list 用兩次，部分建構子變成共享）*本專案在 `Views.View` 上遇過：一個定義把同一個 view 交給另外兩個定義，以及迴圈把 view 當參數帶過每次迭代，都讓所有 `View` 建構子變成共享。前者使四執行緒模型在兩次交替量測中為 136.7 對 143.2 ms、132.1 對 145.3 ms（四分位距約 25 ms）；改成每次使用各建一次 view 後為 130.5 對 130.9 ms。`scripts/check_generated_c.py` 對模型、benchmark 與整合入口會拒絕共享的 `View`；launcher 對每次編譯做的檢查不含這一項，因為程式自己的定義可能共享 view。
- **G3** 全域共享。原始碼裡有一個把所有建構子都標成共享的分支（`FL.hot.add("*")`）。專案遇過一次，當時卷積慢了 16–24%；`scripts/check_generated_c.py` 就是用來擋它的。**probe 沒有重現它**：共享一個型別是型別參數的值（`share_live_parameter`、`share_template_parameter`），或型別是「以 match 定義的型別函式套用在執行期索引上」的值（`share_family_match_live_index`），都只讓部分建構子變成共享。最小的觸發條件還不知道。在找到之前，依靠 `check_generated_c.py` 與每次建置印出的 `N of M constructors shared`。
- **G4** 綁定會計算自己的使用次數：最後一次使用取走值，之前的使用是共享。攤平的值被共享時，複製的是字，不是節點。
- **G5** 樣板與閉包。`~` 樣板引數由 checker 實例化，所以編譯器看到的是直接呼叫。*（probe：`call_template`：整條鏈都是 `spin`）* 當成一般參數傳入的函式是閉包，而且**呼叫閉包的 def 不是 flat**。*（probe：`call_closure`）* 存在 record 裡的函式，量測結果比樣板慢 5.85–6.5 倍。
- **G6** 泛型的 `~Context: Data` 配 `+context`，只會把實例化後的那個型別標成 hot。*（probe：`share_template_parameter`）*

## H. runtime：排程

- **H1** fork（平行 let）產生一個 join task，每個呼叫是一個 kid。kid 依序執行時，一個 frame 供所有步驟使用。
- **H2** worker 在 ring 上輪轉。成長輪（grow）中的 lane 會跳過不含 fork 的 task（`monk_step` 裡的 `fid_nofk`）。host 一個 ring 接一個 ring 地成長，然後清空；v2.0.35 起 host 執行緒也在每一輪領取單位，另有 `threads - 1` 個 pool 執行緒，執行緒睡著之前會先讓出有限次數（`POOL_WAIT`）。沒有 work stealing。所以一次平行就是一輪成長加一輪工作，成長輪本來不該完成 leaf；planner 要以完整的輪次計算。
- **H3** 經過 fork 傳遞的 Array 是帶原子計數的 redirect handle。`Array.clone` 複製整個區塊（`blk_copy`）。
- **H4** **leaf 的工作必須是不含 fork 的 task。否則四個 leaf 只會用到兩個 worker。** 機制在 `cube_run`、`row_grow`、`monk_step`、`task_deal`、`emit_jump`：
  1. 在最上層的 fork，host 把兩個 kid 放到**兩個不同的 row**（`ring_flip`）。
  2. 成長輪中，一個 row 由一個 worker 負責。它執行可能 fork 的 task；這些 task fork 出來的 kid 留在**同一個 row**（`ring_pick`）。
  3. 如果 kid 就是會 fork 的遞迴本身，而 leaf 是被 inline 進去的 flat def，這個 worker 會**在成長輪裡做完一整個 leaf**。做完這個 task 之後，這個 row 的成長輪就結束。
  4. 工作輪中，一個 row 的 128 個 ring 分成若干單位（`pool_step`：使用的 row 少時單位多，row 多時是八個），各執行緒以原子計數器領取下一個單位。剩下的 leaf 在這裡執行。

  所以深度二、四個 leaf 的樹，在成長輪用兩個 worker 做兩個 leaf，在工作輪用兩個 worker 做另外兩個。總時間是兩個 leaf 的時間，不是一個。16 個 leaf 時是兩個在成長輪、14 個在工作輪。

  做法：子節點是 leaf 時，平行 let 呼叫一個**既不 fork、也不呼叫閉包**的 def。依 F6 它會變成 task，它在 `FID_T` 表中那一列的 bit 2 會被設定（`fork_flags.py` 會印出來）。成長輪會跳過它，所有 leaf 都在工作輪的不同單位中執行。熱迴圈放在再下一層的 flat def（F6）。呼叫閉包的 segment 會被當成可能 fork，所以 leaf task 裡不能有 lambda。

  *（probe：`leaf_inline` 與 `leaf_task`，各四個 leaf。在一台 16 執行緒的機器上量到：一執行緒 268 與 266 ms，二執行緒 170 與 172 ms，四執行緒 **171 ms 對 121 ms**。inline 的寫法在四執行緒並不比二執行緒快。）*

## I. 對設計的含意

- 把樹的遞迴收成 map 或 zip 的**骨架**一定不是 flat（F1）。這可以接受，因為骨架在每個節點只執行一次。leaf 以 `~` 樣板參數傳入（G5）並保持 flat（F1）；依 F3，骨架不會拖慢它。子節點是 leaf 時，呼叫一個不含 fork 的 task（H4），熱迴圈放在它的下一層（F6）。不要讓 Buffer、憑證或函式值被共享（G2、G3）。
- **合併程式時**，比較前後生成的 C：leaf 的 `spin_N` 是 `INLINE` 還是 `FAR`、`ctr_take` 出現幾次、有幾個建構子被共享。計時是最後的確認。
- **證明中**，大的機器界限保持抽象（E1、E2）。

## J. lint.py 會找到什麼

| 規則 | 找到的寫法 |
| --- | --- |
| A2 | let 之後的 match 或拆解 |
| A3 | match 或拆解一個計算出來的值 |
| A7 | pattern 後面接 `[]` |
| C1 | `~` 參數不在最前面 |
| C2 | 呼叫時 `~` 引數的個數與被呼叫者的宣告不符（依 import 的別名解析） |
| D4 | let 一個沒有型別標註的建構子 |
| E1 | `U32.to_nat(大數字)`，或六位數以上的 Nat 字面值 |
| E1? | `U32.to_nat(f(X.limits()))`：把預設的機器界限轉成 Nat。問號表示有風險，不一定失敗 |

在 `lib` 與 `convolution` 已驗收的原始碼上，它回報三筆：`lib/kernel_planning.bend` 的 `Nat.double(2147483648n)`，以及 `lib/proofs/tensor_band_upsample_nearest.bend` 的兩處 `E1?`。它只看寫法，所以找不到用量錯誤（B1）與型別不符；檔案的 import 解析不到時，會略過 C2。
