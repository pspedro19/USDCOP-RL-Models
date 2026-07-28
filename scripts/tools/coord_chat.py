#!/usr/bin/env python3
"""Visor de la conversacion CLAUDE <-> CODEX como un chat de terminal.

Fusiona los dos inboxes append-only de `.claude/coordination/` en un solo hilo
cronologico y lo pinta como una conversacion, con el heartbeat de ambos y el
estado del monitor.

    python scripts/tools/coord_chat.py                 # ultimos 12 mensajes
    python scripts/tools/coord_chat.py --all           # hilo completo
    python scripts/tools/coord_chat.py -n 30           # ultimos 30
    python scripts/tools/coord_chat.py --grep billing  # solo los que mencionan algo
    python scripts/tools/coord_chat.py --full          # cuerpo entero, sin recortar
    python scripts/tools/coord_chat.py --follow        # modo chat en vivo

`--follow` refresca cada 5s y solo pinta lo nuevo: es la vista util mientras los
dos agentes trabajan.

Solo lectura. No escribe en los canales.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

COORD = Path(__file__).resolve().parents[2] / ".claude" / "coordination"
INBOX_FROM_CODEX = COORD / "INBOX-CLAUDE.md"   # lo que CODEX me deja a mi
INBOX_FROM_CLAUDE = COORD / "INBOX-CODEX.md"   # lo que yo le dejo a CODEX

# - [CLD-193][P0][TAG][ACK<=10m] cuerpo...
MSG = re.compile(r"^-\s*\[(?P<id>(?:CLD|CXD|MSG)[A-Z0-9-]*)\]\[(?P<pri>P\d)\]\[(?P<tag>[^\]]*)\]")


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"


class C:
    """Paleta. Se apaga sola si la salida no es un terminal."""

    on = _supports_color()

    def __getattr__(self, name: str) -> str:
        codes = {
            "reset": "0", "dim": "2", "bold": "1",
            "claude": "38;5;39",    # azul
            "codex": "38;5;208",    # naranja
            "p0": "38;5;203",       # rojo suave
            "p1": "38;5;179",       # ambar
            "ok": "38;5;78",        # verde
            "meta": "38;5;245",     # gris
        }
        if name not in codes:
            raise AttributeError(name)
        return f"\033[{codes[name]}m" if C.on else ""


c = C()


TS = re.compile(r"\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?)")


@dataclass
class Msg:
    mid: str
    num: int
    pri: str
    tag: str
    body: str
    who: str   # CLAUDE | CODEX
    seq: int   # orden de aparicion dentro de su fichero
    ts: str    # sello extraido del cuerpo, "" si no lo declara

    @property
    def color(self) -> str:
        return c.claude if self.who == "CLAUDE" else c.codex

    @property
    def when(self) -> str:
        return self.ts.replace("T", " ")[5:16] if self.ts else "--:--"


def _num(mid: str) -> int:
    m = re.search(r"(\d+)", mid)
    return int(m.group(1)) if m else 0


def parse(path: Path, who: str) -> list[Msg]:
    if not path.exists():
        return []
    out: list[Msg] = []
    cur: Msg | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = MSG.match(line)
        if m:
            if cur:
                out.append(cur)
            cur = Msg(
                mid=m.group("id"), num=_num(m.group("id")), pri=m.group("pri"),
                tag=m.group("tag"), body=line[m.end():].strip(), who=who,
                seq=len(out), ts="",
            )
        elif cur and line.strip():
            cur.body += " " + line.strip()
    if cur:
        out.append(cur)
    for msg in out:                       # el sello suele ir al principio del cuerpo
        t = TS.search(msg.body[:220])
        msg.ts = t.group(1) if t else ""
    return out


def thread() -> list[Msg]:
    """Hilo fusionado e INTERCALADO de verdad.

    CLD-NNN y CXD-NNN son dos secuencias independientes (hoy van por 194 y 76),
    asi que ordenar por numero separaria los dos lados en bloques. Se ordena por
    el sello real que cada mensaje declara en su cuerpo; los que no lo traen
    heredan el del ultimo mensaje fechado de su propio fichero, para que no
    caigan todos al principio.
    """
    msgs: list[Msg] = []
    for path, who in ((INBOX_FROM_CODEX, "CODEX"), (INBOX_FROM_CLAUDE, "CLAUDE")):
        parsed = parse(path, who)
        heredado = ""
        for m in parsed:
            if m.ts:
                heredado = m.ts
            else:
                m.ts = heredado
        msgs += parsed
    return sorted(msgs, key=lambda x: (x.ts or "0000", x.seq))


def heartbeats() -> list[str]:
    rows = []
    for name, who in (("CLAUDE-STATUS.md", "CLAUDE"), ("CODEX-STATUS.md", "CODEX")):
        p = COORD / name
        if not p.exists():
            continue
        ts = estado = "?"
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[:30]:
            if line.startswith("timestamp:"):
                ts = line.split(":", 1)[1].strip()[:25]
            elif line.startswith("estado:"):
                estado = line.split(":", 1)[1].split("#")[0].strip()
        age = (time.time() - p.stat().st_mtime) / 60
        col = c.claude if who == "CLAUDE" else c.codex
        stale = f"{c.p0}{age:.0f}m stale{c.reset}" if age > 15 else f"{c.ok}{age:.0f}m{c.reset}"
        rows.append(f"  {col}{who:<7}{c.reset} {estado:<12} {c.meta}{ts}{c.reset}  {stale}")
    return rows


def render(m: Msg, width: int, full: bool) -> str:
    pri = c.p0 if m.pri == "P0" else (c.p1 if m.pri == "P1" else c.meta)
    arrow = "->" if m.who == "CLAUDE" else "<-"
    head = (f"{c.meta}{m.when}{c.reset} {m.color}{c.bold}{m.who} {arrow}{c.reset} "
            f"{m.color}{m.mid}{c.reset} {pri}{m.pri}{c.reset} {c.meta}{m.tag}{c.reset}")
    body = re.sub(r"\s+", " ", m.body)
    if not full and len(body) > width * 3:
        body = body[: width * 3 - 3] + "..."
    lines, cur = [], ""
    for word in body.split():
        if len(cur) + len(word) + 1 > width - 4:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    ind = "    " if m.who == "CLAUDE" else "  "
    return head + "\n" + "\n".join(f"{ind}{c.dim}|{c.reset} {ln}" for ln in lines)


def main() -> int:
    # La consola de Windows entrega cp1252 por defecto y los mensajes traen
    # flechas y guiones largos: sin esto revienta con UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="Chat CLAUDE <-> CODEX en terminal")
    ap.add_argument("-n", "--last", type=int, default=12)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--full", action="store_true", help="cuerpo completo")
    ap.add_argument("--grep", metavar="TXT", help="filtra por texto")
    ap.add_argument("--follow", action="store_true", help="modo vivo, refresca 5s")
    a = ap.parse_args()

    width = min(shutil.get_terminal_size((100, 24)).columns, 110)
    seen: set[str] = set()

    while True:
        msgs = thread()
        if a.grep:
            g = a.grep.lower()
            msgs = [m for m in msgs if g in m.body.lower() or g in m.tag.lower()]

        if a.follow and seen:
            nuevos = [m for m in msgs if m.mid not in seen]
            for m in nuevos:
                print(render(m, width, a.full) + "\n")
                seen.add(m.mid)
            if not nuevos:
                print(f"{c.meta}.{c.reset}", end="", flush=True)
            time.sleep(5)
            continue

        os.system("cls" if os.name == "nt" else "clear") if a.follow else None
        print(f"\n{c.bold}CANAL DE COORDINACION{c.reset}  {c.meta}"
              f"{len(msgs)} mensajes | {c.claude}CLAUDE ->{c.reset}{c.meta} envia | "
              f"{c.codex}<- CODEX{c.reset}{c.meta} recibe{c.reset}")
        print(f"{c.meta}{'-' * width}{c.reset}")
        for row in heartbeats():
            print(row)
        print(f"{c.meta}{'-' * width}{c.reset}\n")

        shown = msgs if a.all else msgs[-a.last:]
        for m in shown:
            print(render(m, width, a.full) + "\n")
            seen.add(m.mid)

        if not a.all and len(msgs) > len(shown):
            print(f"{c.meta}  ... {len(msgs) - len(shown)} anteriores "
                  f"(usa --all o -n N){c.reset}\n")

        if not a.follow:
            return 0
        time.sleep(5)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{c.meta}fin{c.reset}")
