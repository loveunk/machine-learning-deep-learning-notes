#!/usr/bin/env python3
"""Check local Markdown links.

This intentionally ignores external URLs and noisy math-like patterns. It is a
small repo maintenance helper, not a full Markdown parser.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
EXTERNAL_RE = re.compile(r"^(https?:|mailto:|tel:|#|javascript:)")


def iter_markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if ".git" not in path.parts and "node_modules" not in path.parts
    )


def normalize_target(raw: str) -> str | None:
    target = raw.strip()
    if not target or EXTERNAL_RE.match(target):
        return None

    target = target.split("#", 1)[0].strip()
    if not target:
        return None

    # Avoid false positives from old inline math/images that accidentally match.
    if len(target) > 240 or "\n" in target:
        return None
    if len(target) <= 3 and "/" not in target and "." not in target:
        return None

    return unquote(target.replace("\\", "/"))


def main() -> int:
    missing: list[tuple[Path, int, str]] = []

    for path in iter_markdown_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in LINK_RE.finditer(text):
            target = normalize_target(match.group(1))
            if target is None:
                continue

            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                line = text.count("\n", 0, match.start()) + 1
                missing.append((path.relative_to(ROOT), line, match.group(1).strip()))

    if missing:
        print(f"Missing local links: {len(missing)}")
        for path, line, target in missing:
            print(f"{path}:{line} -> {target}")
        return 1

    print("All local Markdown links are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
