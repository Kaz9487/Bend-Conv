"""Profile allocation/copy requests in generated C; never modify Bend/upstream sources.

Instrumentation is a separate executable. Its timings are deliberately discarded.
The normal benchmark remains the performance measurement.
"""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'benchmarks'))
from allocation_instrumentation import instrument, replace_once, write_counts

BUILD = ROOT / 'out/build/cpu_convolution'
REPORT = ROOT / 'out/results/cpu_convolution'


def main():
    original = (BUILD / 'bend_convolution.c').read_text()
    source = instrument(original, 3)
    source = replace_once(source, 'INLINE Term blk_new(Env e, bool arr, u64 d, u32 lgs, u32 n, THR Term* v) {', '''
static unsigned long long workspace_new_bytes;
INLINE Term blk_new(Env e, bool arr, u64 d, u32 lgs, u32 n, THR Term* v) {
if(profile_phase<3){unsigned long long bytes=(1ull<<(d+lgs))*(arr?8:4);
__atomic_fetch_add(&workspace_new_bytes,bytes<8?8:bytes,__ATOMIC_RELAXED);}
''')
    source = replace_once(
        source,
        'lab_ticks[0]=io_tick();',
        """
workspace_new_bytes=0;
for(unsigned j=0;j<3;j++)profile_requests[j]=profile_bytes[j]=profile_clones[j]=profile_copy_bytes[j]=0;
profile_phase=0;lab_ticks[0]=io_tick();""",
    )
    source = replace_once(
        source, 'lab_ticks[1]=io_tick();', 'profile_phase=1;lab_ticks[1]=io_tick();'
    )
    source = replace_once(
        source, 'lab_ticks[2]=io_tick();', 'profile_phase=2;lab_ticks[2]=io_tick();'
    )
    source = replace_once(
        source,
        'lab_ticks[3]=io_tick();',
        'profile_phase=3;lab_ticks[3]=io_tick();' + write_counts('"allocation_profile.json"', 3) + '''
FILE*payload=fopen("workspace_payload.json","w");if(!payload)exit(2);
fprintf(payload,"%llu",workspace_new_bytes+profile_copy_bytes[0]+profile_copy_bytes[1]+profile_copy_bytes[2]);fclose(payload);
''',
    )
    path = BUILD / 'allocation_profile.c'
    path.write_text(source)
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/check_generated_c.py'), str(path)], check=True)
    executable = BUILD / 'allocation_profile'
    compiler = os.environ.get('CC') or shutil.which('clang')
    if not compiler:
        sys.exit('Set CC to a Clang-compatible compiler')
    subprocess.run(
        [
            compiler,
            '-O3',
            '-march=native',
            '-ffp-contract=off',
            '-std=c11',
            str(path),
            '-lpthread',
            '-lm',
            '-o',
            str(executable),
        ],
        check=True,
    )
    rows = []
    # Profile the same case/partition combinations; only one numerical run is needed.
    decisions = {(row['case'], row['threads']): row['plan'] for row in json.loads((REPORT / 'results.json').read_text())}
    cases = dict.fromkeys(r['case'] for r in json.loads((REPORT / 'results.json').read_text()))
    for case in cases:
        folder = REPORT / case
        config_path = folder / 'config.bin'
        original_config = config_path.read_bytes()
        output_path = folder / 'bend.bin'
        reference = output_path.read_bytes()
        timing_files = {
            folder / name: (folder / name).read_bytes()
            for name in ['bend.json', 'bend_samples.jsonl']
        }
        try:
            for threads in [1, 4]:
                config = np.frombuffer(original_config, np.uint32).copy()
                config[9] = threads
                config.tofile(config_path)
                subprocess.run(
                    [str(executable), '--threads', str(threads), '--gpu', 'off'],
                    cwd=folder,
                    env=dict(os.environ, CONV_REPS='1'),
                    stdout=subprocess.DEVNULL,
                    check=True,
                    timeout=180,
                )
                assert output_path.read_bytes() == reference, 'Profiling changed numerical output'
                phases = json.loads((folder / 'allocation_profile.json').read_text())
                output_bytes = max(8, 4 * (1 << (int(config[1]) * int(config[7]) * int(config[8]) - 1).bit_length()))
                measured_workspace = json.loads((folder / 'workspace_payload.json').read_text()) - output_bytes
                assert measured_workspace == decisions[case, threads]['workspace_bytes'], (case, threads, measured_workspace, decisions[case, threads])
                rows.append(
                    dict(
                        case=case,
                        threads=threads,
                        workspace_payload_bytes=decisions[case, threads]['workspace_bytes'],
                        measured_workspace_payload_bytes=measured_workspace,
                        returned_output_bytes=output_bytes,
                        budget_semantics='Cumulative Array payload excluding the returned output; allocator metadata and RSS are reported separately.',

                        phases=dict(zip(['reorder', 'pack_gemm', 'gather'], phases)),
                    )
                )
        finally:
            config_path.write_bytes(original_config)
            for timing_path, content in timing_files.items():
                timing_path.write_bytes(content)
            (folder / 'allocation_profile.json').unlink(missing_ok=True)
            (folder / 'workspace_payload.json').unlink(missing_ok=True)
    result = dict(
        interpretation='Allocation requests/size classes, not live memory or RSS. Clone bytes include rounded native capacity. Instrumented times are not performance results.',
        rows=rows,
    )
    (REPORT / 'allocation_profile.json').write_text(json.dumps(result, indent=2))
    print('Allocation and clone counts recorded for', len(rows), 'case/thread combinations.')


if __name__ == '__main__':
    main()
