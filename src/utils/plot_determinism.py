"""PNG deterministas: mismo grafico -> mismos bytes -> git no crea un blob nuevo.

POR QUE EXISTE (auditoria de limpieza 2026-08-24)
-------------------------------------------------
El dashboard sirve 1.492 PNG versionados bajo
`usdcop-trading-dashboard/public/forecasting/`, que los DAGs regeneran cada semana.
La directiva del operador en `CLAUDE.md` exige que **un clon limpio renderice todas
las paginas**, asi que esos PNG SI se versionan a proposito.

Medido sobre la historia real del repo: 2.788 blobs PNG unicos frente a 1.492 en
HEAD; 305,5 MB del pack de 811 MB (38%) son estos graficos, y ~142 MB son versiones
ya superadas. Eso es tolerable porque matplotlib **no** incrusta timestamp en el PNG:
regenerar un grafico con los mismos datos produce bytes identicos y git deduplica.

El problema es el unico dato de entorno que si se cuela: el chunk `tEXt`

    Software  Matplotlib version3.11.1, https://matplotlib.org/

Al subir matplotlib, ese chunk cambia en **todos** los ficheros a la vez, aunque
ningun grafico haya cambiado: un `pip install -U matplotlib` reescribe los 1.492 PNG
y mete ~163 MB de golpe en el pack, con un diff que no significa nada.

Este modulo elimina ese chunk. Los PNG quedan reproducibles a traves de versiones de
matplotlib, y el peso solo crece cuando el CONTENIDO cambia de verdad — que es lo que
se queria versionar.

USO
---
Llamar una vez, despues de `matplotlib.use("Agg")` y antes del primer `savefig`:

    from src.utils.plot_determinism import enable_deterministic_png
    enable_deterministic_png()

Es idempotente y no cambia ningun `savefig` existente: solo fija el valor por defecto
de `metadata`. Un llamador que pase su propio `metadata=` sigue mandando.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_usdcop_deterministic_png"

# `None` le dice a matplotlib que OMITA el chunk (distinto de la cadena vacia, que
# escribiria un tEXt vacio y seguiria siendo un byte de diferencia).
_OMIT_SOFTWARE = {"Software": None}


def enable_deterministic_png() -> bool:
    """Omite el chunk `Software` en todo PNG que se guarde desde este proceso.

    Returns:
        True si se aplico el parche, False si ya estaba puesto o matplotlib no esta
        disponible. Nunca lanza: un script de graficos no debe morir porque no se
        pudo optimizar el tamano de un blob.
    """
    try:
        from matplotlib.figure import Figure
    except Exception as exc:  # pragma: no cover - matplotlib ausente
        logger.debug("plot_determinism: matplotlib no disponible (%s)", exc)
        return False

    if getattr(Figure.savefig, _PATCH_FLAG, False):
        return False

    original = Figure.savefig

    # `functools.wraps` NO es cosmetico aqui. Al importarse, `matplotlib.pyplot` construye
    # `plt.savefig` copiando la docstring de `Figure.savefig` y valida su `__qualname__`
    # con `_add_pyplot_note`; un qualname que no empiece por `Figure.` levanta
    # "Wrapped method from unexpected class" y **rompe el import de pyplot**. Si este
    # parche se aplica antes de que pyplot se importe —que es lo normal en un script que
    # llama a `enable_deterministic_png()` al arrancar— sin `wraps` el proceso muere.
    @functools.wraps(original)
    def savefig(self, fname, *args, **kwargs):
        # Solo PNG: PDF/SVG tienen sus propias claves de metadata y `Software` no es
        # valida ahi — pasarla levantaria un ValueError en el backend.
        if _targets_png(fname, kwargs.get("format")):
            merged = dict(_OMIT_SOFTWARE)
            merged.update(kwargs.get("metadata") or {})
            kwargs["metadata"] = merged
        return original(self, fname, *args, **kwargs)

    setattr(savefig, _PATCH_FLAG, True)   # despues de `wraps`: copia `__dict__`
    Figure.savefig = savefig  # type: ignore[method-assign]
    logger.debug("plot_determinism: chunk `Software` omitido en PNG")
    return True


def _targets_png(fname, explicit_format: str | None) -> bool:
    if explicit_format:
        return explicit_format.lower() == "png"
    name = getattr(fname, "name", fname)
    return isinstance(name, str) and name.lower().endswith(".png")


def png_has_software_chunk(path) -> bool:
    """True si el PNG lleva el chunk `Software` (usado por el guard de regresion)."""
    import struct

    data = open(path, "rb").read()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    i = 8
    while i + 8 <= len(data):
        (length,) = struct.unpack(">I", data[i : i + 4])
        chunk_type = data[i + 4 : i + 8]
        if chunk_type == b"IDAT":
            return False
        if chunk_type == b"tEXt" and data[i + 8 : i + 8 + length].startswith(b"Software\x00"):
            return True
        i += 12 + length
    return False
