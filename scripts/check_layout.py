"""Check maintained source layout and references; this is not a numerical suite."""

from pathlib import Path
import ast
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ['lib', 'convolution', 'scripts', 'benchmarks', 'examples']
GENERATED_DIRS = ['out/models', 'out/checks', 'out/backends']
# Explicit representation adapters may inspect the backend; their callers may not.
REPRESENTATION_ADAPTERS = {
    'convolution/proofs/address_index.bend',
    'convolution/proofs/array_tree_refinement.bend',
    'convolution/proofs/array_tree_structure.bend',
    'convolution/proofs/guarded_read_refinement.bend',
    'convolution/proofs/native_array_model.bend',
    'convolution/proofs/scalar_write_refinement.bend',
    'convolution/proofs/tree_indexing.bend',
}
# Examples use the library as its users do: only the public tensor modules.
PUBLIC_MODULES = {
    'lib/tensor.bend',
    'lib/tensor_f32.bend',
    'lib/tensor_layout.bend',
    'lib/tensor_window.bend',
    'lib/math_activation.bend',
}
errors = []
source_files = []

for folder in SOURCE_DIRS:
    for path in (ROOT / folder).rglob('*'):
        if path.is_dir() and path.name == '__pycache__':
            errors.append(f'Generated Python cache outside out/: {path.relative_to(ROOT)}')
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(ROOT)
        source_files.append(path)
        if len(relative.parts) > 3:
            errors.append(f'Source directory is too deep: {relative}')
        if path.suffix in {'.bend', '.py', '.mjs', '.ts', '.sh', '.c', '.cu'}:
            if path.name != '__init__.py' and not re.fullmatch(r'[a-z][a-z0-9_]*', path.stem):
                errors.append(f'Use descriptive snake_case: {relative}')
        if path.suffix in {'.log', '.csv', '.bin', '.exe', '.pt', '.png', '.jpg', '.pyc'}:
            errors.append(f'Artifact outside out/: {relative}')
        if path.suffix == '.json' and folder != 'scripts':
            errors.append(f'Generated JSON outside out/: {relative}')
        if path.suffix == '.py':
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(relative))
            except SyntaxError as error:
                errors.append(str(error))
        if path.suffix == '.sh' and b'\r' in path.read_bytes():
            errors.append(f'Shell script must use LF: {relative}')

all_files = source_files + [
    path for folder in GENERATED_DIRS for path in (ROOT / folder).rglob('*.bend')
]
for path in all_files:
    if path.suffix not in {'.bend', '.mjs', '.ts', '.c', '.cu'}:
        continue
    source = path.read_text(encoding='utf-8-sig')
    imports = []
    if path.suffix == '.bend':
        imports += re.findall(r'^\s*import\s+"?(\.[^\s"\r\n]+)', source, re.M)
        aliases = re.findall(r'^import\s+\S+\s+as\s+(\w+)', source, re.M)
        if len(aliases) != len(set(aliases)):
            errors.append(f'Duplicate import alias: {path.relative_to(ROOT)}')
        location = path.relative_to(ROOT).as_posix()
        for imported, alias in re.findall(r'^import\s+(\S+)\s+as\s+(\w+)', source, re.M):
            target = (path.parent / imported).resolve()
            if location.startswith('lib/proofs/') and target.is_relative_to(
                ROOT / 'convolution/proofs'
            ):
                errors.append(
                    f'Generic foundation imports convolution proof: {location} -> {imported}'
                )
            if (
                location.startswith('convolution/proofs/')
                and location not in REPRESENTATION_ADAPTERS
            ):
                constructors = {
                    'storage_tree_model.bend': '(?:Leaf|Node)',
                    'kernel_tile.bend': 'Tile',
                }
                constructor = constructors.get(target.name)
                code = re.sub(r'#.*', '', source)
                if constructor and re.search(
                    r'\b' + re.escape(alias) + r'\.' + constructor + r'\s*\{', code
                ):
                    errors.append(
                        f'Concrete storage/tile representation leaked into convolution: {location}'
                    )
                if re.search(r'\b(?:ALeaf|ANode)\s*\{', code):
                    errors.append(
                        f'Native Array representation leaked into convolution: {location}'
                    )
    elif path.suffix in {'.mjs', '.ts'}:
        imports += re.findall(r"(?:from\s+|new URL\()['\"](\.[^'\"]+)['\"]", source)
    else:
        imports += re.findall(r'^#include\s+"([^"]+)"', source, re.M)
    for imported in imports:
        target = (path.parent / imported).resolve()
        if path.parent in {ROOT / 'lib', ROOT / 'convolution'} and (
            ('proofs' in target.relative_to(ROOT).parts and not target.is_relative_to(path.parent / 'proofs'))
            or any(target.is_relative_to(ROOT / folder) for folder in ['benchmarks', 'examples'])
        ):
            errors.append(
                f'Runtime source imports a foreign proof or benchmark: {path.relative_to(ROOT)} -> {imported}'
            )
        if path.relative_to(ROOT).parts[0] in {
            'lib',
            'convolution',
            'proofs',
        } and target.is_relative_to(ROOT / 'out'):
            errors.append(
                f'Maintained source imports generated output: {path.relative_to(ROOT)} -> {imported}'
            )
        if path.relative_to(ROOT).parts[0] == 'lib' and any(
            target.is_relative_to(ROOT / folder)
            for folder in ['convolution', 'examples', 'benchmarks']
        ):
            errors.append(
                f'Library depends on an application or verification layer: {path.relative_to(ROOT)} -> {imported}'
            )
        if not target.is_file():
            errors.append(f'Missing import: {path.relative_to(ROOT)} -> {imported}')
        elif path.suffix == '.bend' and (path.is_relative_to(ROOT / 'examples') or path.is_relative_to(ROOT / 'out/models')):
            module = target.relative_to(ROOT).as_posix()
            if module.split('/')[0] in {'lib', 'convolution'} and module not in PUBLIC_MODULES:
                errors.append(f'Example uses an internal module instead of the public API: {path.relative_to(ROOT)} -> {imported}')

# Public prose has one primary edition and two linked translations.
public_documents = [*ROOT.glob('README*.md'), *ROOT.glob('CHANGELOG*.md'), *ROOT.glob('CONTRIBUTING*.md')]
for folder in ['docs', 'lib', 'convolution', 'benchmarks', 'examples', 'data']:
    public_documents.extend((ROOT / folder).rglob('*.md'))
for path in public_documents:
    source = path.read_text(encoding='utf-8-sig')
    base = re.sub(r'\.zh-TW$', '', path.stem)
    for suffix in ['', '.zh-TW']:
        companion = path.with_name(base + suffix + '.md')
        if not companion.is_file():
            errors.append(f'Missing language edition: {companion.relative_to(ROOT)}')
        elif f']({companion.name})' not in source:
            errors.append(
                f'Missing language navigation: {path.relative_to(ROOT)} -> {companion.name}'
            )
    prose = re.sub(r'^\[English\].*$', '', source, flags=re.M)
    if path.stem == base and re.search('[\u4e00-\u9fff]', prose):
        errors.append(f'Primary document contains untranslated prose: {path.relative_to(ROOT)}')
    for destination in re.findall(r'\]\(([^)]+)\)', source):
        if re.match(r'^(?:https?://|mailto:|#)', destination):
            continue
        target = destination.split('#', 1)[0].strip('<>')
        if target and not (path.parent / target).exists():
            errors.append(f'Missing document target: {path.relative_to(ROOT)} -> {target}')

report = dict(
    source_files=len(source_files),
    errors=errors,
    note='Names of laws and mathematical meaning still require human review.',
)
output = ROOT / 'out/results/proof-audit/layout.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'Layout, filenames, imports and Python syntax passed ({len(source_files)} source files).')
