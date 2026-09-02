from __future__ import annotations
import re
from pathlib import Path

MAIN_XES = Path('/mnt/d/cybersec_iot_datastream_xes/MainProcess.xes')
BACKUP   = MAIN_XES.with_suffix('.xes.preLogWindowFix.bak')


def main():
    if not MAIN_XES.exists():
        raise SystemExit(f"Not found: {MAIN_XES}")

    text = MAIN_XES.read_text(encoding='utf-8')
    print(f"Read {len(text):,} bytes from {MAIN_XES}")

    if not BACKUP.exists():
        BACKUP.write_text(text, encoding='utf-8')
        print(f"Backup written to {BACKUP}")
    else:
        print(f"Backup already exists at {BACKUP} (left untouched)")

    event_pat = re.compile(
        r'\s*<event>\s*'
        r'<date[^/]*/>\s*'
        r'<string\s+key="concept:name"\s+value="LogWindow_(?:train|test)"\s*/>\s*'
        r'<string\s+key="lifecycle:transition"\s+value="complete"\s*/>\s*'
        r'</event>',
        re.DOTALL,
    )

    n_before = len(event_pat.findall(text))
    text_new = event_pat.sub('', text)
    n_after  = len(event_pat.findall(text_new))

    if n_after != 0:
        raise SystemExit(f"Pattern still matches {n_after} blocks after substitution -- aborting.")

    print(f"Removed {n_before:,} synthetic LogWindow_* events.")
    MAIN_XES.write_text(text_new, encoding='utf-8')
    print(f"Wrote {len(text_new):,} bytes back to {MAIN_XES}")
    print(f"Reduction: {len(text) - len(text_new):,} bytes ({100*(len(text)-len(text_new))/len(text):.1f}% smaller)")


if __name__ == '__main__':
    main()
