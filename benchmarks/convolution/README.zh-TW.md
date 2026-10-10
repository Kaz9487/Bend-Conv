# 卷積 benchmark

[English](README.md) | [繁體中文](README.zh-TW.md)

量測單一卷積形狀在 Stelliferous、packed C 參考實作與 NumPy 上的時間。

```sh
python run.py benchmark
```

樣本、環境與報告寫在 `out/results/cpu_convolution/`。

## 方法

- 1 與 4 執行緒；每個後端在同一個 process 內做兩次暖機、七個樣本。
- Total 延遲包含 packing、配置、kernel 與 gather。Core 延遲只含乘加迴圈。檔案 I/O 不計時。
- C 參考實作採用相同的 packed 演算法與累加順序。它建置兩次，分別開啟與關閉自動向量化，兩者都必須與 Bend 逐位元相同。
- C 參考實作依通道分割，Bend 依輸出位置分割，所以這個比較只隔離出向量化，不代表產生程式碼的全部成本。
- `BENCH_PYTHON` 指定有 NumPy 與 threadpoolctl 的 Linux Python，預設 `python3`。

## 檔案

| 檔案 | 角色 |
|---|---|
| [benchmark_cpu.py](benchmark_cpu.py) | 執行順序、執行緒數、取樣與數值比對 |
| [numpy_convolution.py](numpy_convolution.py) | 形狀、固定種子的輸入與預訓練權重 |
| [packed_convolution.c](packed_convolution.c) | C 參考實作 |
| [run_cpu.sh](run_cpu.sh) | 建置並執行三個後端 |
| [report_plans.py](report_plans.py) | 規劃器的決定，從編譯後的 Bend 讀出，不計時任何 kernel |
| [profile_allocations.py](profile_allocations.py) | 計算配置與複製請求，與規劃器的 workspace 核對，並確認輸出不變。它的計時不採用 |
