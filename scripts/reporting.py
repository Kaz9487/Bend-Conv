"""Write two human-readable editions from shared measurements and sections."""

from pathlib import Path

SUFFIXES = ('', '.zh-TW')


def write_report(path: Path, sections: list[str | tuple[str, str]]) -> None:
    """Strings contain shared data; tuples contain English and Traditional Chinese text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    links = [f'{path.stem}{suffix}{path.suffix}' for suffix in SUFFIXES]
    navigation = f'[English]({links[0]}) | [繁體中文]({links[1]})'
    for edition, filename in enumerate(links):
        parts = [section if isinstance(section, str) else section[edition] for section in sections]
        path.with_name(filename).write_text(
            parts[0].rstrip()
            + '\n\n'
            + navigation
            + '\n\n'
            + '\n\n'.join(parts[1:]).replace('|\n\n|', '|\n|')
            + '\n',
            encoding='utf-8',
        )
