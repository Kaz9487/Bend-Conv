import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--threads', type=int, default=1)
args = parser.parse_args()
if args.threads < 1:
    parser.error('--threads must be positive')
label = args.backend if args.threads == 1 else f'{args.backend}_{args.threads}t'
folder = Path(__file__).resolve().parents[2] / f'out/results/backends/bend_{label}'
rows = [json.loads(line) for line in (folder / 'native_samples.jsonl').read_text().splitlines()]
assert len(rows) == 9 and [row['sample'] for row in rows] == list(range(9)), 'Incomplete same-process run'
assert all(row['dump_seconds'] == 0 and row['seconds'] > 0 for row in rows)
times = [row['seconds'] for row in rows[2:]]
(folder / 'timing.json').write_text(
    json.dumps(
        dict(
            backend=f'Bend {args.backend.upper()}',
            variant='convolution/inference',
            threads=args.threads,
            warmups=2,
            process_protocol='same process; reload consumed inputs outside timed forward',
            seconds=times,
        ),
        indent=2,
    )
)
