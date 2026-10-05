import importlib.metadata as m
from pathlib import Path

root = Path(__file__).resolve().parents[2]
packages = {d.metadata['Name']: d.version for d in m.distributions() if d.metadata['Name']}
(root / 'out/results/backends/requirements-linux.txt').write_text(
    '\n'.join(f'{k}=={v}' for k, v in sorted(packages.items())) + '\n'
)
