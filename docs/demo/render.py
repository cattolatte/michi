"""Build an animated SVG terminal demo for michi.

Self-contained: pure SVG plus one CSS keyframe timeline. No JavaScript, no
external font, no build step — the same constraints michi puts on its own
reports, so the demo cannot rot when a toolchain moves on.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path

# michi's report palette, so the demo and the product look like one thing.
BG = "#17161a"
PANEL = "#211f26"
INK = "#ece9e4"
MUTED = "#948d87"
DIM = "#6f6a66"
ACCENT = "#b23a30"
SEAL = "#c0392b"
GOOD = "#6aab6e"
WARN = "#d4a548"
HIGH = "#e2645a"
LINE = "#332f39"
RULE = "#4a4553"

CHAR_W = 7.82
LINE_H = 18.5
PAD_X = 24.0
BAR_H = 36.0
FIRST_LINE = BAR_H + 30.0


@dataclass
class Scene:
    """One command and the output it produced."""

    command: str
    output: list[str] = field(default_factory=list)
    hold: float = 3.6
    note: str = ""


_SEVERITY = {"high": HIGH, "warn": WARN, "info": MUTED}


_BLOCK_CHARS = set("█╗║╔╝╚═ ")


def _style(text: str) -> str:
    """Colour a line of michi output by what it is."""
    escaped = html.escape(text)

    # The banner's block capitals, and the seal riding the third row. Matched
    # by character class rather than by position so the art can be re-cut
    # without the renderer needing to know its shape.
    stripped_raw = text.strip()
    if stripped_raw and set(text) <= _BLOCK_CHARS | {"道"}:
        return f'<tspan fill="{ACCENT}" font-weight="700">{escaped}</tspan>'

    # The inventory block: chrome dim, contents readable.
    brackets = re.match(r"^([+\- ]*=\[)(.*)(\])\s*$", escaped)
    if brackets:
        opening, body, closing = brackets.groups()
        return (
            f'<tspan fill="{DIM}">{opening}</tspan>'
            f'<tspan fill="{MUTED}">{body}</tspan>'
            f'<tspan fill="{DIM}">{closing}</tspan>'
        )

    match = re.match(r"^(\s*)(high|warn|info)(\s.*)$", escaped)
    if match:
        lead, word, rest = match.groups()
        colour = _SEVERITY[word]
        return f'{lead}<tspan fill="{colour}" font-weight="600">{word}</tspan>{rest}'

    # A console prompt: the context is the point, so it reads brighter than
    # the command typed after it.
    prompt = re.match(r"^(michi[^›]*)(›)(\s*)(.*)$", escaped)
    if prompt:
        head, arrow, space, rest = prompt.groups()
        return (
            f'<tspan fill="{MUTED}">{head}</tspan>'
            f'<tspan fill="{ACCENT}" font-weight="700">{arrow}</tspan>'
            f'{space}<tspan fill="{INK}">{rest}</tspan>'
        )

    stripped = escaped.strip()
    if stripped and set(stripped) <= {"─", "━", "-"}:
        return f'<tspan fill="{RULE}">{escaped}</tspan>'
    if stripped.startswith(("Findings", "Results", "Metrics", "Columns")):
        return f'<tspan fill="{INK}" font-weight="700">{escaped}</tspan>'
    if stripped.startswith("Verdict"):
        return escaped.replace(
            "Verdict", f'<tspan fill="{INK}" font-weight="700">Verdict</tspan>', 1
        )
    if stripped.startswith(("道", "心得")):
        return f'<tspan fill="{ACCENT}" font-weight="700">{escaped}</tspan>'
    if stripped.startswith(("tip:", "✓ marks", "run one and stop")):
        return f'<tspan fill="{DIM}">{escaped}</tspan>'
    if stripped[:1] in {"✓", "·"} and "  " in stripped:
        # A `path` row: the mark carries the state, the rest is a label. The
        # leading indent is preserved separately — partitioning the padded
        # line would hand back an empty mark and colour nothing.
        lead = escaped[: len(escaped) - len(escaped.lstrip())]
        mark, rest = stripped[0], stripped[1:]
        colour = GOOD if mark == "✓" else DIM
        return f'{lead}<tspan fill="{colour}" font-weight="700">{mark}</tspan>{rest}'
    if "leader" in escaped and "tied" not in escaped:
        return escaped.replace(
            "leader", f'<tspan fill="{GOOD}" font-weight="600">leader</tspan>', 1
        )
    if "tied with leader" in escaped:
        return escaped.replace(
            "tied with leader", f'<tspan fill="{WARN}">tied with leader</tspan>', 1
        )
    if stripped.startswith(("loaded ", "target = ", "unsaved")):
        return f'<tspan fill="{DIM}">{escaped}</tspan>'
    if stripped.startswith(("column ", "model ", "setting", "wrote", "Run again")):
        return f'<tspan fill="{DIM}">{escaped}</tspan>'
    return escaped


def build(scenes: list[Scene], destination: Path, *, cols: int = 84) -> None:
    """Render the scenes into one looping animated SVG."""
    type_time = 1.0
    gap = 0.35

    starts: list[float] = []
    clock = 0.5
    for scene in scenes:
        starts.append(clock)
        clock += type_time + gap + scene.hold
    total = clock + 0.6

    tallest = max(len(scene.output) + (3 if scene.note else 0) for scene in scenes)
    width = PAD_X * 2 + CHAR_W * cols
    height = FIRST_LINE + LINE_H * tallest + 26

    def pct(seconds: float) -> float:
        return max(0.0, min(100.0, 100.0 * seconds / total))

    rules: list[str] = []
    body: list[str] = []

    for index, (scene, start) in enumerate(zip(scenes, starts, strict=True)):
        typed_at = start
        output_at = start + type_time + gap
        ends = start + type_time + gap + scene.hold

        # The scene as a whole fades in, holds, and fades out.
        rules.append(
            f"@keyframes s{index}{{"
            f"0%,{pct(start - 0.25):.3f}%{{opacity:0}}"
            f"{pct(start):.3f}%,{pct(ends - 0.25):.3f}%{{opacity:1}}"
            f"{pct(ends):.3f}%,100%{{opacity:0}}}}"
        )
        rules.append(f".s{index}{{animation:s{index} {total:.2f}s infinite}}")

        # The command types itself: a clip window widening in character steps.
        length = max(len(scene.command), 1)
        rules.append(
            f"@keyframes t{index}{{"
            f"0%,{pct(typed_at):.3f}%{{width:0}}"
            f"{pct(typed_at + type_time):.3f}%,100%{{width:{length * CHAR_W:.1f}px}}}}"
        )
        rules.append(
            f".t{index}{{animation:t{index} {total:.2f}s steps({length},end) infinite}}"
        )

        # Output arrives as a block, a beat after the command lands.
        rules.append(
            f"@keyframes o{index}{{"
            f"0%,{pct(output_at):.3f}%{{opacity:0}}"
            f"{pct(output_at + 0.22):.3f}%,100%{{opacity:1}}}}"
        )
        rules.append(f".o{index}{{animation:o{index} {total:.2f}s infinite}}")

        if scene.note:
            rules.append(
                f"@keyframes n{index}{{"
                f"0%,{pct(output_at + 0.9):.3f}%{{opacity:0}}"
                f"{pct(output_at + 1.3):.3f}%,100%{{opacity:1}}}}"
            )
            rules.append(f".n{index}{{animation:n{index} {total:.2f}s infinite}}")

        lines: list[str] = [f'<g class="s{index}">']
        command_x = PAD_X + CHAR_W * 2
        lines.append(
            f'<clipPath id="c{index}">'
            f'<rect class="t{index}" x="{command_x:.1f}" y="{FIRST_LINE - 14:.1f}" '
            f'width="0" height="20"/></clipPath>'
            f'<text x="{PAD_X:.1f}" y="{FIRST_LINE:.1f}" class="mono" '
            f'fill="{ACCENT}" font-weight="700">❯</text>'
            f'<g clip-path="url(#c{index})">'
            f'<text x="{command_x:.1f}" y="{FIRST_LINE:.1f}" class="mono" '
            f'fill="{INK}">{html.escape(scene.command)}</text></g>'
        )

        y = FIRST_LINE
        for line in scene.output:
            y += LINE_H
            if line.strip():
                lines.append(
                    f'<text x="{PAD_X:.1f}" y="{y:.1f}" class="mono o{index}" '
                    f'fill="{INK}">{_style(line)}</text>'
                )

        if scene.note:
            y += LINE_H * 1.9
            lines.append(
                f'<text x="{PAD_X:.1f}" y="{y:.1f}" class="mono n{index}" '
                f'fill="{ACCENT}" font-weight="600">{html.escape(scene.note)}</text>'
            )

        lines.append("</g>")
        body.append("".join(lines))

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" \
height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" role="img" \
aria-label="michi: inspect a dataset, clean it, compare models, export code">
<style>
.mono {{
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 12.6px;
  white-space: pre;
}}
.ui {{
  font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 12px;
}}
{chr(10).join(rules)}
</style>
<rect width="{width:.0f}" height="{height:.0f}" rx="11" fill="{BG}"/>
<path d="M0 11a11 11 0 0 1 11-11h{width - 22:.0f}a11 11 0 0 1 11 11v25H0z"
      fill="{PANEL}"/>
<line x1="0" y1="{BAR_H:.0f}" x2="{width:.0f}" y2="{BAR_H:.0f}" stroke="{LINE}"/>
<rect x="0.5" y="0.5" width="{width - 1:.0f}" height="{height - 1:.0f}" rx="11"
      fill="none" stroke="{LINE}"/>

<rect x="17" y="9" width="19" height="19" rx="3.5" fill="{SEAL}"/>
<text x="26.5" y="23.5" text-anchor="middle" font-size="12.5" fill="#fff"
      font-family="Hiragino Sans, Yu Gothic, Noto Sans JP, sans-serif">道</text>
<text x="46" y="23.5" class="ui" fill="{MUTED}">michi<tspan fill="{DIM}">\
 — a local-first ML workbench</tspan></text>

{chr(10).join(body)}
</svg>
"""
    destination.write_text(svg, encoding="utf-8")
    print(
        f"wrote {destination.name}  "
        f"{len(svg) // 1024} KB · {width:.0f}×{height:.0f} · {total:.1f}s loop"
    )
