"""Check maintained component proofs and the exact root with upstream Bend."""

from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

sys.dont_write_bytecode = True
from proof_layout import file_digest, frozen_files

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'out/results/matrix'
COMPONENT_ENTRIES = {
    'convolution/proofs/reference_refinement.bend': 'reference-contract.log',
    'convolution/proofs/matrix_matches_convolution.bend': 'convolution.log',
    'convolution/proofs/matrix_axis_partitions.bend': 'tiles.log',
    'convolution/proofs/matrix_axis_lengths.bend': 'dimensions.log',
    'convolution/proofs/array_api_validation.bend': 'api.log',
    'scripts/check_official.bend': 'native-components.log',
}
ROOT_LAW = 'convolution/proofs/native_inference_law.bend'
ROOT_STATEMENT = """law native_inference_matches_matrix:
  for +problem: ConvSpec.Problem
  {NativeConv.infer(problem) == MatrixSpec.infer(problem) : ConvSpec.ConvOutcome}"""


def check_local_trust_boundary():
    """Enforce project trust policy separately from the official verdict.

    The official checker still checks every imported body. Separately enforce
    our project policy: no unsafe definitions and foreign code only for IO.
    This is a source policy check, not a second type checker or a proof.
    """
    foreign_io = []
    for folder in ['lib', 'convolution']:
        for path in (ROOT / folder).rglob('*.bend'):
            source = '\n'.join(line.split('#', 1)[0] for line in path.read_text().splitlines())
            if '@unsafe' in source:
                raise SystemExit('Unsafe maintained definition is not accepted: ' + str(path))
            for definition in re.split(r'(?=^def )', source, flags=re.M):
                if not re.search(r'^\s+import\s+"', definition, re.M):
                    continue
                header = definition.split('\n  import', 1)[0].strip()
                if folder != 'convolution' or not re.search(r'->\s*IO\([^\n]+\):$', header):
                    raise SystemExit('Foreign code outside the declared IO boundary: ' + str(path))
                foreign_io.append(
                    path.relative_to(ROOT).as_posix() + ':' + header.split('(', 1)[0][4:]
                )
    return foreign_io


def check(entry, log):
    result = subprocess.run(
        ['node', str(ROOT / 'scripts/bend_launcher.mjs'), entry, '--check-only'],
        cwd=ROOT,
        capture_output=True,
        timeout=120,
    )
    output = (result.stdout + result.stderr).decode('utf-8', errors='replace')
    (REPORT / log).write_text(output, encoding='utf-8')
    # The CLI can print a checker error without a failing process exit code.
    accepted = result.returncode == 0 and 'ALL PROOFS CHECK' in output
    return accepted and 'TODO' not in output, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    status_path = REPORT / 'proof-status.json'
    # A failed run must not leave a previous successful status as current evidence.
    status_path.unlink(missing_ok=True)

    toolchain = json.loads((ROOT / 'scripts/toolchain.json').read_text())
    upstream = ROOT / '.tools/bend'
    git = ['git', '-c', f'safe.directory={upstream}', '-C', str(upstream)]
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    if commit != toolchain['commit']:
        raise SystemExit(
            'Bend checkout differs from scripts/toolchain.json; validate the toolchain upgrade first.'
        )
    if subprocess.run(git + ['diff', '--quiet', 'HEAD']).returncode:
        raise SystemExit(
            'The official Bend checkout has tracked modifications; a checker fork is not accepted.'
        )
    foreign_io = check_local_trust_boundary()
    root_source = (ROOT / ROOT_LAW).read_text(encoding='utf-8-sig')
    root_declaration = re.search(
        r'^law native_inference_matches_matrix:.*?(?=^(?:def|law|type) |\Z)',
        root_source,
        re.M | re.S,
    )
    if not root_declaration or ''.join(root_declaration[0].split()) != ''.join(
        ROOT_STATEMENT.split()
    ):
        raise SystemExit('The exact unconditional native-inference root statement changed.')

    for filename, expected in frozen_files().items():
        if file_digest(filename) != expected:
            raise SystemExit('A frozen specification file changed: ' + filename)

    for entry, log in COMPONENT_ENTRIES.items():
        accepted, output = check(entry, log)
        if not accepted:
            raise SystemExit(f'Component proof failed: {entry}\n{output}')

    complete, output = check(ROOT_LAW, 'native-obligation.log')
    if not complete:
        raise SystemExit('The complete native-inference root did not pass:\n' + output)

    status = dict(
        checker='unmodified upstream Bend via bend_launcher.mjs',
        checker_commit=commit,
        checker_version=subprocess.check_output(
            ['node', str(ROOT / 'scripts/bend_launcher.mjs'), 'version'], cwd=ROOT, text=True
        ).strip(),
        local_unsafe_definitions=False,
        foreign_io_boundary=foreign_io,
        component_entries_checked=list(COMPONENT_ENTRIES),
        all_component_entries_proved=True,
        frozen_contracts_verified=True,
        native_all_split_depths_proved=complete,
        native_to_matrix_proved=complete,
        scope='Exact whole native-inference root, including invalid inputs and scalar fallback; ordered source-level F32 semantics. Component premises are discharged in the root chain. Compiler/runtime, activation-enabled Array convolution API and end-to-end YOLO remain outside this theorem. See docs/proofs.md.',
    )
    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    print('All component proofs passed the unmodified official checker.')
    print('Exact native-to-matrix root complete:', complete)
    if args.require_complete and not complete:
        raise SystemExit('Exact whole-convolution law remains open; promotion refused.')


if __name__ == '__main__':
    main()
