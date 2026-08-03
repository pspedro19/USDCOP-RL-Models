"""Small dependency-free parser for governed relative Markdown links.

Both knowledge gates consume this module so they agree on what constitutes an edge.
The parser deliberately supports balanced brackets in labels (for example ranges such as
``[0, 1]``), skips fenced/inline code examples, and normalizes angle-bracket destinations.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import unquote

if TYPE_CHECKING:
    from collections.abc import Iterator

CODE_SPAN_RE = re.compile(r"`[^`]*`")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SKIP_PREFIXES = ("http://", "https://", "//", "#", "mailto:")


def markdown_destinations(line: str) -> tuple[str, ...]:
    """Return inline-link destinations, including labels with balanced brackets."""
    destinations: list[str] = []
    cursor = 0
    while cursor < len(line):
        start = line.find("[", cursor)
        if start < 0:
            break

        depth = 1
        end_label = start + 1
        while end_label < len(line) and depth:
            char = line[end_label]
            if char == "\\":
                end_label += 2
                continue
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            end_label += 1

        if depth or end_label >= len(line) or line[end_label] != "(":
            cursor = start + 1
            continue

        target_start = end_label + 1
        end_target = target_start
        paren_depth = 1
        quote: str | None = None
        in_angle = False
        while end_target < len(line) and paren_depth:
            char = line[end_target]
            if char == "\\":
                end_target += 2
                continue
            if char == "<" and quote is None:
                in_angle = True
            elif char == ">" and in_angle:
                in_angle = False
            elif quote is not None and char == quote and not in_angle:
                quote = None
            elif (
                quote is None
                and char in {'"', "'"}
                and not in_angle
                and end_target > target_start
                and line[end_target - 1].isspace()
            ):
                quote = char
            elif quote is None and not in_angle:
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
            end_target += 1

        if paren_depth:
            cursor = start + 1
            continue
        destinations.append(line[target_start : end_target - 1])
        cursor = end_target
    return tuple(destinations)


def normalize_destination(raw_target: str) -> str | None:
    """Return a decoded local path, or ``None`` for external/fragment-only links."""
    target = raw_target.strip()
    if not target:
        return None
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = re.split(r'\s+["\']', target, maxsplit=1)[0]
    target = target.strip()
    if not target or target.startswith(SKIP_PREFIXES):
        return None
    path_part = unquote(target.split("#", 1)[0].strip())
    return path_part or None


def iter_relative_links(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(line number, local path)`` for real links outside code examples."""
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = CODE_SPAN_RE.sub("", line)
        for raw_target in markdown_destinations(line):
            target = normalize_destination(raw_target)
            if target is not None:
                yield lineno, target
