"""Count how many workers hold a ring claim during each graph operation.

Run: python -B benchmarks/model/timeline_claims.py TIMELINE_DIR [--kind KIND]
TIMELINE_DIR is an output folder of profile_timeline.py. Without --kind it
prints one row per operation kind; with --kind, one row per operation of that
kind. Each value is a median over the measured samples, in milliseconds: wall
time, time in grow turns, time in work turns, and the time with 0, 1, 2, 3 and
4 or more distinct workers holding a claim. A claim is ring occupancy, not CPU
time.
"""
import argparse
import json
from pathlib import Path
import statistics

from profile_timeline import KINDS, read_events


def occupancy(start, stop, claims):
    """Nanoseconds of [start, stop) with exactly k distinct claiming workers; the last bin is 4 or more."""
    points = []
    for begin, end, worker in claims:
        begin, end = max(begin, start), min(end, stop)
        if begin < end:
            points.extend([(begin, 1, worker), (end, -1, worker)])
    bins = [0] * 5
    active = {}
    cursor = start
    for time, change, worker in sorted(points):
        bins[min(sum(1 for count in active.values() if count), 4)] += time - cursor
        cursor = time
        active[worker] = active.get(worker, 0) + change
    bins[min(sum(1 for count in active.values() if count), 4)] += stop - cursor
    return bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('timeline', type=Path)
    parser.add_argument('--kind', help='List every operation of this kind, for example conv')
    args = parser.parse_args()
    events = read_events(args.timeline / 'trace.bin')
    operations = json.loads((args.timeline / 'graph.json').read_text())['ops']
    claims = events[events['kind'] == KINDS['claim']]
    turns = events[events['kind'] == KINDS['turn']]
    begins = events[events['kind'] == KINDS['operation_begin']]
    ends = events[events['kind'] == KINDS['operation_end']]
    samples = sorted(set(begins['sample'].tolist()))[2:]
    rows = {}
    for sample in samples:
        totals = {}
        for begin in begins[begins['sample'] == sample]:
            index = int(begin['operation'])
            kind = operations[index]['kind']
            if args.kind and kind != args.kind:
                continue
            end = ends[(ends['sample'] == sample) & (ends['operation'] == index)]
            start, stop = int(begin['t0']), int(end[0]['t0'])
            inside = claims[(claims['t1'] > start) & (claims['t0'] < stop)]
            bins = occupancy(start, stop, [(int(c['t0']), int(c['t1']), int(c['worker'])) for c in inside])
            phase = [0, 0]
            for turn in turns[(turns['t1'] > start) & (turns['t0'] < stop)]:
                phase[int(turn['value']) & 1] += min(int(turn['t1']), stop) - max(int(turn['t0']), start)
            total = totals.setdefault(index if args.kind else kind, [0] * 8)
            for position, value in enumerate([stop - start, phase[1], phase[0]] + bins):
                total[position] += value
        for key, total in totals.items():
            rows.setdefault(key, []).append(total)
    print(f'{args.timeline.as_posix()}: {len(samples)} samples, {len(events)} events; ms, median of samples')
    print(f'{"operation":18}' + ''.join(f'{name:>9}' for name in ('wall', 'grow', 'work', '0', '1', '2', '3', '4+')))
    for key in sorted(rows):
        medians = [statistics.median(row[position] for row in rows[key]) / 1e6 for position in range(8)]
        label = key
        if args.kind:
            operation = operations[key]
            shape = 'x'.join(str(operation[name]) for name in ('output_channels', 'output_height', 'output_width') if name in operation)
            label = f'{key} {shape}'
        print(f'{label:18}' + ''.join(f'{value:9.2f}' for value in medians))


if __name__ == '__main__':
    main()
