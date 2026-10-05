"""Proof/source audit, not a numerical test suite or a replacement checker.

--check runs the single official proof gate. The inventory
records full declarations and assumptions without treating imports as proof
dependencies or treating conditional theorems as discharged applications.
"""

from pathlib import Path
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from reporting import write_report

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / 'bend.cmd').is_file()
)
REPORT = ROOT / 'out/results/proof-audit'
ENTRY_OFFICIAL = [
    'scripts/check_official.bend',
    'convolution/proofs/matrix_matches_convolution.bend',
    'convolution/proofs/matrix_axis_partitions.bend',
    'convolution/proofs/matrix_axis_lengths.bend',
    'convolution/proofs/array_api_validation.bend',
    'convolution/proofs/reference_refinement.bend',
]
parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
REPORT.mkdir(parents=True, exist_ok=True)




def relative(p):
    return p.resolve().relative_to(ROOT.resolve()).as_posix()


if args.check:
    for gate in ['check_official_proofs.py']:
        p = subprocess.run(
            [sys.executable, str(ROOT / 'scripts' / gate)],
            cwd=ROOT,
            capture_output=True,
            # The gate runs seven official entries sequentially. A large
            # component closure can use most of its own 120-second budget.
            timeout=600,
        )
        output = (p.stdout + p.stderr).decode('utf-8', errors='replace')
        (REPORT / (gate + '.log')).write_text(output, encoding='utf-8')
        if p.returncode:
            raise SystemExit(f'Proof gate failed: {gate}; see out/results/proof-audit')

# The upstream parser supplies navigation references only; official acceptance
# above remains the sole proof verdict. Keep the map current after renaming.
# Upstream resolves imports from POSIX paths, so Windows runs it in WSL.
if sys.platform == 'win32':
    distribution = ['-d', os.environ['BEND_WSL_DISTRO']] if os.environ.get('BEND_WSL_DISTRO') else []
    dependencies = ['wsl.exe', *distribution, '--cd', str(ROOT), '--exec', 'sh', '-c',
                    '. scripts/linux_environment.sh && exec bun scripts/proof_dependencies.ts']
else:
    local_bun = ROOT / '.tools/bun/node_modules/@oven/bun-linux-x64/bin/bun'
    if not (platform.machine() == 'x86_64' and local_bun.is_file()):
        local_bun = 'bun'
    dependencies = [str(local_bun), str(ROOT / 'scripts/proof_dependencies.ts')]
subprocess.run(dependencies, cwd=ROOT, check=True, timeout=180)


files = {}
for folder in [
    'lib',
    'convolution',
    'examples',
    'benchmarks',
    'scripts',
    'out/models',
    'out/backends',
    'out/checks',
]:
    for p in (ROOT / folder).rglob('*.bend'):
        files[relative(p)] = p.read_text(encoding='utf-8-sig')

edges = {}
aliases = {}
unresolved = []
for path, source in files.items():
    edges[path] = []
    aliases[path] = {}
    for rel, alias in re.findall(r'^import\s+(\S+)(?:\s+as\s+(\w+))?', source, re.M):
        if not rel.startswith('.'):
            continue
        target = relative((ROOT / path).parent / rel)
        if target not in files:
            unresolved.append(dict(source=path, target=target))
            continue
        edges[path].append(target)
        if alias:
            aliases[path][alias] = target


def closure(entries):
    reached = set()
    todo = list(entries)
    while todo:
        p = todo.pop()
        if p in reached:
            continue
        reached.add(p)
        todo.extend(edges.get(p, []))
    return reached


official = closure(ENTRY_OFFICIAL)
root_imports = closure(['convolution/proofs/native_inference_law.bend'])
assert not [e for e in unresolved if e['source'] in official | root_imports], (
    'Active entry has an unresolved import'
)
roles = json.loads((ROOT / 'scripts/proof_roles.json').read_text(encoding='utf8'))
for entry in ['scripts/check_official.bend']:
    roles[entry] = 'checker-entry'


def classify(path):
    return roles.get(
        path, 'unclassified'
    ), 'Exact premises and results are recorded in declarations.'


declarations = []
definitions = set()
for path, source in files.items():
    for m in re.finditer(r'^(def|law|type)\s+([\w.]+)', source, re.M):
        kind, name = m.groups()
        end = re.search(r'^(?:def|law|type|import)\s', source[m.end() :], re.M)
        block = source[m.start() : m.end() + end.start()] if end else source[m.start() :]
        if kind == 'def':
            alias, sep, local = name.partition('.')
            owner = aliases[path].get(alias, path) if sep else path
            definitions.add((owner, local if sep else name))
        if '/proofs/' not in path and path != 'scripts/check_official.bend':
            continue
        # Full law block contains every binder. Def header contains every
        # argument type, including premise proofs, and its complete result.
        header_end = re.search(r'(?m)^.*:\s*$', block)
        statement = (
            block.strip()
            if kind == 'law'
            else block[: header_end.end()].strip()
            if header_end
            else block.strip()
        )
        declarations.append(
            dict(
                file=path,
                line=source.count('\n', 0, m.start()) + 1,
                kind=kind,
                name=name,
                statement=statement,
            )
        )

for d in declarations:
    if d['kind'] == 'law':
        d['definition_present'] = (d['file'], d['name']) in definitions
        d['warning'] = (
            'Presence of a definition is not checker acceptance or discharge of its premises.'
        )

# Human-reviewed semantic distinctions. Never infer these from import counts.
judgments = {
    (
        'convolution/proofs/scalar_inference_refinement.bend',
        'scalar_inference_matches_matrix',
    ): 'Unconditional complete scalar inference to frozen matrix: all guards, real allocation/read/loop/store/output order/unpack and reference fallback are proved.',
    (
        'convolution/proofs/native_inference_law.bend',
        'native_inference_matches_matrix',
    ): 'Complete exact root with no additional premises: invalid input, scalar fallback and packed execution. Automatic planning selects the proved scalar or arbitrary-U32-depth packed path after the original guards.',
    (
        'convolution/proofs/packed_inference_refinement.bend',
        'run_readback_matches_matrix',
    ): 'Actual packed Array execution to the frozen ordered matrix values for any caller storage satisfying the common contents/capacity contract; includes weight reorder, input row windows (copied bands over aligned rows, the caller input otherwise), packing, arbitrary split depth, four-row GEMM jobs at eight positions and gather.',
    (
        'convolution/proofs/conditional_write_coverage.bend',
        'MatchingWritesHaveValue',
    ): 'A premise family, not a proof that actual writes are correct. Multiple matching writes of the same value are allowed.',
    (
        'convolution/proofs/packing_contents.bend',
        'packing_contents',
    ): 'The actual packer of tiles * inner entries, at every output width and start position, stores the flat tabulation of its entries. Validity, support, total capacity and a start range ending within seven of the output grid are premises; prepare and band leaves discharge them.',
    (
        'convolution/proofs/packing_readback.bend',
        'allocated_readback',
    ): 'The flat tabulation over a freshly allocated buffer reads back the native pixel of each lane; positive count, capacity bound and query range remain premises and are discharged by prepared_packing_samples.leaf_samples.',
    (
        'convolution/proofs/matrix_reduction_order.bend',
        'matrix_reduction_is_consecutive',
    ): 'Unconditional equality of the complete flattened frozen matrix K sequence with range(ci*k*k,0), including order. Native tap accessors and FP32 dot results are connected by gemm_dot_matches_matrix.',
    (
        'convolution/proofs/gemm_dot_matches_matrix.bend',
        'job_dot',
    ): 'Ordered FP32 accumulator of one four-row GEMM job at eight positions, row and lane equals the frozen matrix dot over the reordered weights under a LeafCertificate and valid group/output membership. Certificate is produced by actual prepare; caller scheduling and output writes remain separate.',
    (
        'convolution/proofs/microkernel_leaf_readback.bend',
        'leaf_every_position',
    ): 'Every in-range output of an actual pack plus four-row GEMM leaf at eight positions reads back the frozen matrix value plus bias. Takes a LeafCertificate premise (discharged by the gather proofs from actual prepare), rebuilds the certificate for its own allocated pack buffer, and uses complete coverage for full and partial channel blocks.',
    (
        'convolution/proofs/packing_lane_reads.bend',
        'prepared_pixel_matches_direct_read',
    ): 'Actual computed pixel-block geometry, for every Nat index and U32 dimension, gives the same native Array read and padding mask as the direct pixel function. No premise assumes a cache field is correct; no restriction to eight lanes. This is not the whole convolution root.',
    (
        'convolution/proofs/array_api_validation.bend',
        'validation_matches_spec',
    ): 'Validation Bool only; not a full Array convolution API to List Problem contents refinement.',
}
available = {(d['file'], d['name']) for d in declarations}
assert judgments.keys() <= available, 'Reviewed claim moved/renamed; update the audit explicitly'
for d in declarations:
    if (d['file'], d['name']) in judgments:
        d['reviewed_scope'] = judgments[d['file'], d['name']]

modules = []
for path in sorted(files):
    if '/proofs/' not in path and path != 'scripts/check_official.bend':
        continue
    role, note = classify(path)
    evidence = 'official-entry' if path in official else 'not-in-current-entries'
    modules.append(
        dict(
            path=path,
            role=role,
            scope=note,
            checker_entry_membership=evidence,
            imported_by_root_source=path in root_imports,
            imports=edges[path],
        )
    )

assert all(m['role'] != 'unclassified' for m in modules), 'A new module needs classification'
root = 'convolution/proofs/native_inference_law.bend'
root_defined = (root, 'native_inference_matches_matrix') in definitions
gate_status = {}
for name in ['proof-status.json']:
    p = ROOT / 'out/results/matrix' / name
    gate_status[name] = json.loads(p.read_text()) if p.exists() else None
# The verdict counts only when this run started the official checker itself.
whole = args.check and all(
    status and status.get('native_to_matrix_proved') is True for status in gate_status.values()
)
data = dict(
    root=dict(
        file=root,
        statement='F.infer(p) == M.infer(p)',
        definition_present=root_defined,
        checked_complete=whole,
        note='Completion comes from checking the exact root body; import membership and conditional premises alone are not completion evidence.',
    ),
    verification_run=args.check,
    gate_status=gate_status,
    modules=modules,
    declarations=declarations,
    unresolved_outside_active_entries=unresolved,
    terminology='Import closure is load membership, not a theorem dependency graph. The semantic obligation graph is docs/proofs.md.',
)
(REPORT / 'inventory.json').write_text(
    json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8'
)
write_report(
    REPORT / 'inventory.md',
    [
        ('# Proof inventory', '# 證明清冊'),
        (
            '[Declarations and premises](inventory.json). The `verification_run` field records whether this invocation ran the checker. Entry membership identifies loaded modules. [Proof map](../../../docs/proofs.md).',
            '[完整宣告與前提](inventory.json)。`verification_run` 記錄此次是否執行 checker；入口歸屬標示載入的模組。[證明導覽](../../../docs/proofs.zh-TW.md)。',
        ),
        (
            '| Module | Role | Check entry |\n|---|---|---|',
            '| 模組 | 角色 | 檢查入口 |\n|---|---|---|',
        ),
        '\n'.join(
            f'| [{m["path"]}](../../../{m["path"]}) | {m["role"]} | {m["checker_entry_membership"]} |'
            for m in modules
        ),
    ],
)
dependencies = json.loads((REPORT / 'dependencies.json').read_text(encoding='utf-8'))
write_report(
    REPORT / 'dependencies.md',
    [
        ('# Proof references', '# 證明引用導覽'),
        (
            '[Reference data](dependencies.json) contains signature and body references from the official parser, including types and all branches. [Semantic obligations](../../../docs/proofs.md) explain the execution paths. The following interfaces have independent uses outside the root.',
            '[引用資料](dependencies.json) 記錄官方 parser 的簽章與內容引用，包含型別及所有分支。[語意責任](../../../docs/proofs.zh-TW.md)說明執行路徑。以下介面在根定理之外也有獨立用途。',
        ),
        (
            '| Module | Interface |\n|---|---|',
            '| 模組 | 介面 |\n|---|---|',
        ),
        '\n'.join(
            f'| [{item["file"]}](../../../{item["file"]}) | `{item["symbol"].rsplit(".", 1)[-1]}` |'
            for item in dependencies['retained_interfaces']
        ),
    ],
)
# Inventory every maintained file, including non-proof tools and comparisons.
# Regex findings are review aids, never evidence of semantic equivalence.
folder_roles = {
    '.github': 'repository automation',
    'lib': 'reusable library',
    'convolution': 'convolution implementation and specification',
    'benchmarks': 'performance comparison or integration tool',
    'scripts': 'validation and inventory tool',
    'examples': 'library usage and complete YOLO application',
    'docs': 'maintained explanation',
    'data': 'input, pretrained weights or frozen hashes',
}
workspace = []
for folder, purpose in folder_roles.items():
    for path in sorted((ROOT / folder).rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        item = dict(path=relative(path), purpose=purpose, bytes=path.stat().st_size)
        if '/proofs/' in item['path'] and path.suffix == '.bend':
            item['purpose'], item['scope'] = classify(item['path'])
        elif '/proofs/' in item['path']:
            item['purpose'] = 'proof navigation and documentation'
        if path.suffix == '.bend':
            source = path.read_text(encoding='utf-8-sig')
            names = re.findall(r'^(?:def|law)\s+([\w.]+)', source, re.M)
            groups = {}
            for name in names:
                match = re.fullmatch(r'(.+?)(\d+)', name)
                if match:
                    groups.setdefault(match[1], set()).add(name)
            item['numbered_families'] = [sorted(v) for v in groups.values() if len(v) > 1]
            item['max_line_length'] = max(map(len, source.splitlines()), default=0)
        workspace.append(item)
for path in sorted(ROOT.iterdir()):
    if path.is_file():
        workspace.append(
            dict(
                path=path.name,
                purpose='workspace entry or configuration',
                bytes=path.stat().st_size,
            )
        )
assert all(item['purpose'] != 'unclassified' for item in workspace), 'Unclassified maintained file'
candidates = [
    p.name
    for p in ROOT.iterdir()
    if p.is_dir() and p.name not in {*folder_roles, '.git', '.tools', '.venv', 'out'}
]
# Git-ignored local folders are never published, like the exceptions above.
# The names go to git as bytes: text mode on Windows would end each with a
# carriage return, which git reads as part of the name.
ignored = subprocess.run(
    ['git', 'check-ignore', '--stdin'], cwd=ROOT, input='\n'.join(n + '/' for n in candidates).encode(),
    capture_output=True,
).stdout.decode().split()
unknown = sorted(name for name in candidates if name + '/' not in ignored)
assert not unknown, 'Unclassified workspace directories: ' + str(unknown)
workspace_report = dict(
    files=workspace,
    directory_exceptions={
        '.git': 'Local version-control metadata; never published as source.',
        '.tools': 'Upstream toolchains and dependencies; preserved, not custom proof sources.',
        '.venv': 'Python environment; not maintained project source.',
        'out': 'Generated builds, wiring, inventories, runtime checks and benchmark evidence; never imported by maintained library, convolution or package proofs.',
    },
    review='docs/design.md',
    warning='A numbered family or matching body is not, by itself, a duplicate theorem.',
)
(REPORT / 'workspace.json').write_text(
    json.dumps(workspace_report, ensure_ascii=False, indent=2), encoding='utf-8'
)
purpose_labels = {
    'repository automation': '儲存庫自動化',
    'reusable library': '可重用函式庫',
    'convolution implementation and specification': '卷積實作與規格',
    'performance comparison or integration tool': '效能比較或整合工具',
    'validation and inventory tool': '驗收與清冊工具',
    'library usage and complete YOLO application': '函式庫用法與完整 YOLO 應用',
    'maintained explanation': '正式文件',
    'input, pretrained weights or frozen hashes': '輸入、預訓練權重或凍結雜湊',
    'proof navigation and documentation': '證明導覽與文件',
    'workspace entry or configuration': '專案入口或配置',
}
workspace_rows = []
for edition in range(2):
    rows = []
    for item in workspace:
        purpose = item['purpose']
        if edition and purpose in purpose_labels:
            purpose = purpose_labels[purpose]
        rows.append(f'| [{item["path"]}](../../../{item["path"]}) | {purpose} |')
    workspace_rows.append('\n'.join(rows))
write_report(
    REPORT / 'workspace.md',
    [
        ('# Workspace inventory', '# 專案檔案清冊'),
        (
            '[Source and artifact rules](../../../docs/development.md). [Machine-readable inventory](workspace.json).',
            '[來源與產物規則](../../../docs/development.zh-TW.md)。[機器可讀清冊](workspace.json)。',
        ),
        (
            '| File | Purpose |\n|---|---|',
            '| 檔案 | 用途 |\n|---|---|',
        ),
        tuple(workspace_rows),
    ],
)
print(
    f'Audited {len(modules)} modules and {len(declarations)} declarations. Root complete: {whole}.'
)
