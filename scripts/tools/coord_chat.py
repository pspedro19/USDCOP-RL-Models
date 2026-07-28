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

También puede emitir mensajes de operador append-only con ``--send``. El envío
requiere ``--message`` (o lo solicita de forma interactiva) y nunca sobrescribe
un inbox.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import threading
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
            "pedro": "38;5;213",    # magenta — el operador
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
    who: str   # CLAUDE | CODEX | PEDRO
    seq: int   # orden de aparicion dentro de su fichero
    ts: str    # sello extraido del cuerpo, "" si no lo declara

    @property
    def color(self) -> str:
        return {"CLAUDE": c.claude, "CODEX": c.codex}.get(self.who, c.pedro)

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
            mid = m.group("id")
            # Los mensajes del operador (`--send`) se appendean al inbox del
            # destinatario, asi que el fichero NO dice quien los escribio: hay
            # que reconocerlos por su prefijo o saldrian atribuidos al agente.
            autor = "PEDRO" if mid.startswith("MSG-OPERATOR") else who
            cur = Msg(
                mid=mid, num=_num(mid), pri=m.group("pri"),
                tag=m.group("tag"), body=line[m.end():].strip(), who=autor,
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
        if not parsed:
            continue
        # Los ficheros son APPEND-ONLY, asi que su orden ya es cronologico. Un
        # mensaje sin sello se escribio ANTES que el siguiente que si lo trae:
        # rellenar hacia ATRAS (con el siguiente fechado) lo coloca bien, mientras
        # que rellenar hacia adelante lo mandaba al pasado — por eso un mensaje
        # de las 18:36 salia fechado a las 14:41.
        mtime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))
        siguiente = mtime
        for m in reversed(parsed):
            if m.ts:
                siguiente = m.ts
            else:
                m.ts = siguiente
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


def send_message(target: str, body: str, priority: str, tag: str) -> str:
    """Append one operator message to the selected inbox and return its ID."""
    if not body.strip():
        raise ValueError("message cannot be empty")
    if priority not in {"P0", "P1", "P2", "P3"}:
        raise ValueError("priority must be P0, P1, P2 or P3")
    destinations = {
        "claude": INBOX_FROM_CODEX,
        "codex": INBOX_FROM_CLAUDE,
    }
    if target not in destinations:
        raise ValueError("target must be claude or codex")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    iso = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    mid = f"MSG-OPERATOR-{stamp}"
    line = f"- [{mid}][{priority}][{tag}][ACK<=10m] [{iso}] {body.strip()}\n"
    path = destinations[target]
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return mid


def render(m: Msg, width: int, full: bool) -> str:
    pri = c.p0 if m.pri == "P0" else (c.p1 if m.pri == "P1" else c.meta)
    arrow = {"CLAUDE": "->", "CODEX": "<-"}.get(m.who, "**")
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
    ind = {"CLAUDE": "    ", "CODEX": "  "}.get(m.who, "      ")
    return head + "\n" + "\n".join(f"{ind}{c.dim}|{c.reset} {ln}" for ln in lines)


def interactive(width: int, full: bool, last: int, show_all: bool) -> int:
    """Chat a tres bandas: escribes, se envia, y ves lo que responden.

    Al entrar pinta el historial (por defecto los ultimos `last`; todo con
    `--all`), y luego un bucle: lo nuevo desde la ultima vuelta + tu linea.
    Enter vacio = solo refrescar (util para ver si ya contestaron).
    """
    destino, prioridad = "both", "P1"

    print(f"\n{c.bold}CHAT DE COORDINACION{c.reset}  "
          f"{c.claude}CLAUDE{c.reset} · {c.codex}CODEX{c.reset} · {c.pedro}PEDRO{c.reset}")
    for row in heartbeats():
        print(row)

    historial = thread()
    seen = {m.mid for m in historial}
    visibles = historial if show_all else historial[-last:]
    omitidos = len(historial) - len(visibles)
    print(f"{c.meta}{'-' * width}{c.reset}")
    if omitidos:
        print(f"{c.meta}  ... {omitidos} mensajes anteriores "
              f"(/todo para verlos, o arranca con --all){c.reset}\n")
    for m in visibles:
        print(render(m, width, full) + "\n")

    print(f"{c.meta}{'-' * width}\n  Escribe y pulsa Enter para enviar.  Enter vacio = refrescar.\n"
          f"  /claude /codex /both   cambia destinatario (ahora: {destino})\n"
          f"  /p0 /p1 /p2            cambia prioridad     (ahora: {prioridad})\n"
          f"  /ver [n]               repinta los ultimos n (por defecto 10)\n"
          f"  /todo                  historial COMPLETO\n"
          f"  /buscar <texto>        filtra el hilo entero\n"
          f"  /salir{c.reset}\n")

    # Vigilante en segundo plano: pinta lo que llegue MIENTRAS escribes, sin que
    # tengas que pulsar Enter. Sin esto, un mensaje nuestro se quedaba invisible
    # hasta tu siguiente turno, que es justo lo contrario de un chat.
    lock = threading.Lock()
    parar = threading.Event()

    def vigilar() -> None:
        while not parar.wait(3):
            try:
                nuevos = [m for m in thread() if m.mid not in seen]
            except OSError:
                continue
            if not nuevos:
                continue
            with lock:
                print()
                for m in nuevos:
                    print(render(m, width, full) + "\n")
                    seen.add(m.mid)
                print(f"{c.pedro}{destino}/{prioridad} >{c.reset} ", end="", flush=True)

    hilo = threading.Thread(target=vigilar, daemon=True)
    hilo.start()

    while True:
        try:
            linea = input(f"{c.pedro}{destino}/{prioridad} >{c.reset} ").strip()
        except EOFError:
            parar.set()
            return 0

        if not linea:
            continue
        low = linea.lower()
        if low in ("/salir", "/quit", "/exit"):
            return 0
        if low in ("/claude", "/codex", "/both"):
            destino = low[1:]
            print(f"{c.meta}  destinatario -> {destino}{c.reset}")
            continue
        if low in ("/p0", "/p1", "/p2", "/p3"):
            prioridad = low[1:].upper()
            print(f"{c.meta}  prioridad -> {prioridad}{c.reset}")
            continue
        if low.startswith("/ver"):
            partes = low.split()
            n = int(partes[1]) if len(partes) > 1 and partes[1].isdigit() else 10
            for m in thread()[-n:]:
                print(render(m, width, full) + "\n")
            continue
        if low in ("/todo", "/all", "/historial"):
            todos = thread()
            print(f"{c.meta}  historial completo: {len(todos)} mensajes{c.reset}\n")
            for m in todos:
                print(render(m, width, full) + "\n")
            continue
        if low.startswith(("/buscar", "/grep")):
            partes = linea.split(maxsplit=1)
            if len(partes) < 2:
                print(f"{c.p0}  uso: /buscar <texto>{c.reset}")
                continue
            q = partes[1].lower()
            hits = [m for m in thread() if q in m.body.lower() or q in m.tag.lower()]
            print(f"{c.meta}  {len(hits)} mensajes mencionan '{partes[1]}'{c.reset}\n")
            for m in hits:
                print(render(m, width, full) + "\n")
            continue
        if linea.startswith("/"):
            print(f"{c.p0}  comando desconocido{c.reset}")
            continue

        objetivos = ("claude", "codex") if destino == "both" else (destino,)
        for t in objetivos:
            try:
                mid = send_message(t, linea, prioridad, "OPERADOR")
                seen.add(mid)
                print(f"{c.ok}  -> enviado a {t.upper()} como {mid}{c.reset}")
            except (OSError, ValueError) as exc:
                print(f"{c.p0}  fallo el envio a {t}: {exc}{c.reset}")


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
    ap.add_argument("--chat", action="store_true",
                    help="modo interactivo: escribes y se envia sin salir")
    ap.add_argument("--send", choices=("claude", "codex", "both"),
                    help="envia un mensaje de operador al inbox indicado")
    ap.add_argument("--message", help="texto del mensaje; si falta, se solicita")
    ap.add_argument("--priority", choices=("P0", "P1", "P2", "P3"), default="P1")
    ap.add_argument("--tag", default="OPERATOR", help="etiqueta de coordinación")
    a = ap.parse_args()

    if a.send:
        body = a.message or input("Mensaje para {}: ".format(a.send)).strip()
        targets = ("claude", "codex") if a.send == "both" else (a.send,)
        for target in targets:
            mid = send_message(target, body, a.priority, a.tag)
            print(f"sent {mid} -> {target}")
        return 0

    width = min(shutil.get_terminal_size((100, 24)).columns, 110)

    if a.chat:
        return interactive(width, a.full, a.last, a.all)

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
              f"{len(msgs)} mensajes | {c.claude}CLAUDE{c.reset}{c.meta} · "
              f"{c.codex}CODEX{c.reset}{c.meta} · {c.pedro}PEDRO{c.reset}{c.meta} (operador){c.reset}")
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
