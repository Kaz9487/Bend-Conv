"""Locate maintained proofs and the frozen specification files."""

from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def proof_modules():
    return sorted(
        [ROOT / 'scripts/check_official.bend']
        + [
            path
            for folder in ['lib', 'convolution']
            for path in (ROOT / folder / 'proofs').glob('*.bend')
        ]
    )


def frozen_files():
    """The files whose bytes define the convolution semantics, with their recorded SHA-256."""
    return json.loads((ROOT / 'data/contracts/frozen.json').read_text(encoding='utf-8'))


def file_digest(relative_path):
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
