import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--threads', type=int, default=1)
args = parser.parse_args()
if args.threads < 1:
    parser.error('--threads must be positive')
label = args.backend if args.threads == 1 else f'{args.backend}_{args.threads}t'
folder = Path(__file__).resolve().parents[2] / f'out/results/backends/bend_{label}'
if args.backend == 'cpu':
    # The example prints one line per forward; its clock counts milliseconds.
    samples = [int(value) / 1000 for value in re.findall(r'^forward (\d+) ms$', (folder / 'samples.log').read_text(), re.M)]
    protocol = 'same process; the image and the weights are copied outside the timed forward'
else:
    rows = [json.loads(line) for line in (folder / 'native_samples.jsonl').read_text().splitlines()]
    assert [row['sample'] for row in rows] == list(range(len(rows))), 'Incomplete same-process run'
    assert all(row['dump_seconds'] == 0 for row in rows)
    samples = [row['seconds'] for row in rows]
    protocol = 'same process; reload consumed inputs outside timed forward'
assert len(samples) == 9 and all(value > 0 for value in samples), 'Incomplete same-process run'
(folder / 'timing.json').write_text(
    json.dumps(
        dict(
            backend=f'Bend {args.backend.upper()}',
            variant='convolution/inference',
            threads=args.threads,
            warmups=2,
            process_protocol=protocol,
            seconds=samples[2:],
        ),
        indent=2,
    )
)
