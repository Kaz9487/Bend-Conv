# 開發

[English](development.md) | [繁體中文](development.zh-TW.md)

## 指令

完成[開始使用](getting_started.zh-TW.md)後，在儲存庫根目錄執行。在 Windows 上，原生步驟在 WSL 裡執行。

| 指令 | 作用 |
|---|---|
| `python run.py proofs` | 以官方檢查器檢查所有證明，並確認根定理完整 |
| `python run.py strict` | 同上，另外要求編譯器是未修改的指定版本 |
| `python run.py layout` | 原始碼配置、檔名、import、連結、凍結檔雜湊 |
| `python run.py integration` | 原生卷積案例，1、2、4、8 執行緒，逐位元比對 |
| `python run.py api` | 只用 `F` 寫成的程式，1 與 4 執行緒，與 NumPy 逐位元比對 |
| `python run.py model --case bus_640 --threads 4` | YOLOv5n：產生、編譯、執行，與 PyTorch 比對 |
| `python run.py benchmark` | 卷積運算的計時 |
| `python run.py backends` | YOLOv5n 在 NumPy、Bend-Conv 與 PyTorch 上的 forward 時間 |
| `python run.py summary` | 列出已記錄的結果哪些通過，不重新執行 |

檢查單一 Bend 檔；先掃描檢查器會拒絕的寫法：

```sh
python -B scripts/toolchain_rules/lint.py FILE_OR_DIR ...
node scripts/bend_launcher.mjs FILE --check-only      # 印出 ALL PROOFS CHECK
```

更換指定的 Bend 版本後：

```sh
python -B scripts/toolchain_rules/verify.py
```

各項掃描結果的說明見[工具鏈規則](toolchain_rules.zh-TW.md)。[CI](../.github/workflows/ci.yml) 執行 `layout`、`proofs` 與 `api`。

## 配置

| 資料夾 | 內容 |
|---|---|
| `lib/` | 函式庫；`lib/proofs/` 是它的證明。不 import 其他資料夾 |
| `convolution/` | 凍結的規格、參考實作與已證明的 Array 引擎；`convolution/proofs/` |
| `examples/` | 使用公開 API 的程式 |
| `benchmarks/` | 數值檢查與計時流程 |
| `scripts/` | 檢查器入口、配置與稽核工具、工具鏈規則 |
| `data/` | 凍結檔雜湊，以及標明出處的範例素材 |
| `out/` | 所有產生的檔案。不提交，也不被 import |

編譯器版本只有一個來源：[toolchain.json](../scripts/toolchain.json)。

## 規則

- 儲存、走訪與 reduction 各只有一份實作。特化版本要附量測。
- 不得弱化 `native_inference_matches_matrix`，不得更動凍結檔。
- 只用官方檢查器。函式庫與卷積程式碼不用 `@unsafe`。
- 不用 C 算術、FMA 或 fast-math。累加順序是契約的一部分。
- 證明是手寫的，不用產生器。依賴卷積假設的證明放在 `convolution/proofs/`。
- 帶前提的引理，只有在前提被解除的地方才算結果。
- 不測試已證明的定律。數值檢查是使用公開 API 的程式。
- 規劃器的常數不針對單一機器調整。
- 檔案與函式用 snake_case，型別與 import 別名用 PascalCase；每個被 import 的檔案只用一個別名。
- 文件以英文撰寫，並附 `.zh-TW.md`。文字檔使用 LF。

Python 用 `ruff format`，C 用 `clang-format`；版本在 [requirements-dev.txt](../requirements-dev.txt)。

## 貢獻

從 topic branch 開 pull request。執行上面的檢查，原始碼變更時一併更新文件，效能主張要附工作負載與量測方式。貢獻的內容與專案其餘部分相同，採 MIT 或 Apache-2.0 授權。
