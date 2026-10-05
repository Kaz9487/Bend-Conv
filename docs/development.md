# Development

[English](development.md) | [繁體中文](development.zh-TW.md)

## Commands

Run from the repository root, after [getting started](getting_started.md). On Windows the native steps run inside WSL.

| Command | What it does |
|---|---|
| `python run.py proofs` | Runs the official checker on every proof and confirms the root theorem is complete |
| `python run.py strict` | The same, requiring an unmodified pinned compiler checkout |
| `python run.py layout` | Source layout, file names, imports, links, frozen file hashes |
| `python run.py integration` | Native convolution cases on 1, 2, 4 and 8 threads, compared bit for bit |
| `python run.py api` | Programs written against `F` only, on 1 and 4 threads, compared bit for bit with NumPy |
| `python run.py model --case bus_640 --threads 4` | YOLOv5n: generate, compile, run, compare with PyTorch |
| `python run.py benchmark` | Convolution operator timings |
| `python run.py backends` | YOLOv5n forward time on NumPy, Bend-Conv and PyTorch |
| `python run.py summary` | Lists which recorded results passed, without rerunning them |

Check one Bend file, after scanning it for spellings the checker rejects:

```sh
python -B scripts/toolchain_rules/lint.py FILE_OR_DIR ...
node scripts/bend_launcher.mjs FILE --check-only      # prints ALL PROOFS CHECK
```

After changing the pinned Bend version:

```sh
python -B scripts/toolchain_rules/verify.py
```

The [toolchain rules](toolchain_rules.md) explain each finding. [CI](../.github/workflows/ci.yml) runs `layout`, `proofs` and `api`.

## Layout

| Folder | Content |
|---|---|
| `lib/` | The library; `lib/proofs/` holds its proofs. It imports nothing from the other folders |
| `convolution/` | The frozen specification, the reference and the proved Array engine; `convolution/proofs/` |
| `examples/` | Programs that use the public API |
| `benchmarks/` | Numerical checks and timing workflows |
| `scripts/` | The checker entry, layout and audit tools, the toolchain rules |
| `data/` | Frozen file hashes and the attributed example assets |
| `out/` | Everything generated. Never committed, never imported |

The compiler version has one source, [toolchain.json](../scripts/toolchain.json).

## Rules

- Storage, traversal and reduction have one implementation. A specialization comes with a measurement.
- Do not weaken `native_inference_matches_matrix` or change the frozen files.
- Only the official checker. No `@unsafe` in library or convolution code.
- No C arithmetic, fused multiply-add or fast-math. The accumulation order is part of the contract.
- Proofs are written by hand, not generated. A proof that depends on convolution assumptions lives in `convolution/proofs/`.
- A lemma with premises is a result only where its premises are discharged.
- Do not test a proved law. Numerical checks are programs against the public API.
- Planner constants are not tuned to one machine.
- snake_case files and functions, PascalCase types and import aliases, one alias per imported file.
- Documents are in English with a `.zh-TW.md` companion. Text files use LF.

Format Python with `ruff format` and C with `clang-format`; versions are in [requirements-dev.txt](../requirements-dev.txt).

## Contributing

Open a pull request from a topic branch. Run the checks above, update the documents with the source, and give the workload and protocol behind any performance claim. Contributions are licensed under MIT or Apache-2.0, like the rest of the project.
