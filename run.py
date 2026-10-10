"""One entry for every check and measurement: python run.py ACTION [options].

The same command works on Linux, macOS and Windows. Bend builds native code
on Linux and macOS; on Windows the native steps run inside WSL in this
checkout (BEND_WSL_DISTRO selects a distribution, otherwise the default).
Host steps use the Python that runs this file. Generated files go under out/.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WINDOWS = sys.platform == 'win32'
CASES = ['bus_320', 'bus_640', 'zidane_320', 'zidane_640']
INTEGRATION_ENTRIES = ['convolution', 'scalar_convolution', 'array_api', 'array_api_zero_workspace', 'band_layout']


def run(command, failure):
    """Run one step from the repository root and stop at the first failure."""
    if subprocess.run([str(part) for part in command], cwd=ROOT).returncode:
        sys.exit(f'{failure}. See out/results/.')


def python(script, *arguments, failure):
    run([sys.executable, '-B', script, *arguments], failure)


def bend(source, output):
    """Emit C with the pinned official compiler."""
    run(['node', 'scripts/bend_launcher.mjs', source, '-o', output], f'Bend could not compile {source}')
    python('scripts/check_generated_c.py', output, failure=f'{output} shares convolution constructors')


def linux(*command, failure):
    """Run a native step: directly, or inside WSL on Windows."""
    if WINDOWS:
        distribution = ['-d', os.environ['BEND_WSL_DISTRO']] if os.environ.get('BEND_WSL_DISTRO') else []
        command = ['wsl.exe', *distribution, '--cd', ROOT, '--exec', *command]
    run(command, failure)


def linux_python(script, *arguments, failure):
    """Run a Python tool where the native programs run."""
    if WINDOWS:
        linux('sh', '-c', '. scripts/linux_environment.sh && exec "$py" -B "$@"', 'python', script, *arguments, failure=failure)
    else:
        python(script, *arguments, failure=failure)


def proofs(options):
    python('scripts/audit_proofs.py', '--check', failure='The official checker did not accept every proof')


def strict(options):
    python('scripts/check_official_proofs.py', '--require-complete', failure='The strict proof check did not pass')


def layout(options):
    python('scripts/check_layout.py', failure='The layout check did not pass')


def summary(options):
    python('scripts/summarize_validation.py', failure='The summary could not be written')


def integration(options):
    (ROOT / 'out/build/integration').mkdir(parents=True, exist_ok=True)
    python('benchmarks/checks/generate_cases.py', failure='Integration cases could not be generated')
    python('benchmarks/checks/band_cases.py', failure='Band cases could not be generated')
    for entry in INTEGRATION_ENTRIES:
        bend(f'out/checks/{entry}.bend', f'out/build/integration/{entry}.c')
    linux('sh', 'benchmarks/checks/run.sh', failure='Native integration did not pass')


def api(options):
    linux_python('benchmarks/checks/public_api_cases.py', failure='The public API cases did not pass')
    linux_python('benchmarks/checks/tensor_file_cases.py', failure='The tensor file cases did not pass')


def reference(case):
    """Export the official reference and fused weights once per case."""
    if not (ROOT / f'out/results/{case}/reference.json').is_file():
        image, size = case.split('_')
        python('examples/yolov5/prepare_reference.py', '--image', image, '--size', size, failure='The official reference could not be prepared')


def model(options):
    case, threads = options.case, options.threads
    reference(case)
    (ROOT / 'out/models').mkdir(parents=True, exist_ok=True)
    bend('examples/yolov5/yolov5n.bend', 'out/models/yolov5n.c')
    linux('sh', 'examples/yolov5/build_run.sh', case, threads, failure='The native model did not build or run')
    python('examples/yolov5/compare_results.py', case, '--threads', threads, failure='The numerical comparison did not pass')
    report = 'comparison_native.json' if threads == 1 else f'comparison_native_{threads}t.json'
    print(f'PASS: out/results/{case}/{report}')


def benchmark(options):
    (ROOT / 'out/build/cpu_convolution').mkdir(parents=True, exist_ok=True)
    bend('benchmarks/convolution/run_convolution.bend', 'out/build/cpu_convolution/bend_convolution.c')
    linux('sh', 'benchmarks/convolution/run_cpu.sh', failure='The convolution comparison did not pass')


def backends(options):
    # The validation also compares every backend with the example's own native output.
    if not (ROOT / 'out/results/bus_640/native/pred.npy').is_file():
        model(argparse.Namespace(case='bus_640', threads=1))
    python('benchmarks/backends/generate_graph.py', failure='The graph could not be exported')
    bend('examples/yolov5/yolov5n.bend', 'out/models/yolov5n.c')
    if options.cuda:
        python('benchmarks/model/generate_yolo_graph.py', '--case', 'bus_640', '--target', 'cuda', failure='The CUDA graph could not be generated')
        run(['node', 'scripts/bend_launcher.mjs', 'out/models/bus_640_cuda.bend', '-o', 'out/backends/bend_cuda.c'], 'Bend could not compile the CUDA model')
        linux('sh', 'benchmarks/backends/run_cpu.sh', failure='The CPU comparison did not pass')
        linux('sh', 'benchmarks/backends/run_cuda.sh', failure='The CUDA comparison did not pass')
        python('benchmarks/backends/validate.py', failure='The numerical verification did not pass')
        python('benchmarks/backends/write_report.py', failure='The report could not be written')
    else:
        linux('sh', 'benchmarks/backends/run_cpu.sh', 'libraries', failure='The CPU comparison did not pass')
        selected = ['numpy_1t', 'numpy_4t', 'bend_cpu', 'bend_cpu_4t', 'torch_cpu_1t', 'torch_cpu_4t']
        python('benchmarks/backends/validate.py', '--backends', *selected, failure='The numerical verification did not pass')
        python('benchmarks/backends/write_report.py', '--cpu-only', failure='The report could not be written')
    print('Report: out/results/backend_comparison.md')


ACTIONS = {
    'proofs': (proofs, 'check every proof with the official checker and confirm the root theorem'),
    'strict': (strict, 'the same, requiring an unmodified pinned compiler checkout'),
    'layout': (layout, 'source layout, names, imports, links and frozen hashes'),
    'integration': (integration, 'native convolution cases on 1, 2, 4 and 8 threads'),
    'api': (api, 'programs against the public API, compared bit for bit with NumPy'),
    'model': (model, 'YOLOv5n: generate, compile, run and compare with the reference'),
    'benchmark': (benchmark, 'time single convolution shapes'),
    'backends': (backends, 'time the YOLOv5n forward on NumPy, Stelliferous and PyTorch'),
    'summary': (summary, 'list which recorded results passed, without rerunning them'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='action', required=True)
    for name, (function, text) in ACTIONS.items():
        command = commands.add_parser(name, help=text)
        command.set_defaults(function=function)
        if name == 'model':
            command.add_argument('--case', choices=CASES, default='bus_640')
            command.add_argument('--threads', type=int, default=1)
        if name == 'backends':
            command.add_argument('--cuda', action='store_true', help='also run the handwritten C and CUDA comparisons')
    options = parser.parse_args()
    if getattr(options, 'threads', 1) < 1:
        parser.error('--threads must be positive')
    os.environ['PYTHONPYCACHEPREFIX'] = str(ROOT / 'out/python_cache')
    if not WINDOWS:
        os.environ.setdefault('BENCH_PYTHON', sys.executable)
    options.function(options)


if __name__ == '__main__':
    main()
