"""Build editable/PDF documents from the signed confirmatory v4 chapters.

This is a document-only renderer: it does not train, call providers, load secrets,
or recompute a hold-out. It keeps the v4 addendum separate from the historical
retrospective manuscript so readers cannot confuse the evidence scopes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)

    from scripts.presentation import build_thesis_manuscript as renderer

    c34 = renderer.strip_frontmatter(
        (ROOT / "docs/thesis/chapters_3_4_v4_addendum.md").read_text(encoding="utf-8")
    )
    c5 = renderer.strip_frontmatter(
        (ROOT / "docs/thesis/chapter_5_conclusions_v4.md").read_text(encoding="utf-8")
    )
    text = "# Evidencia confirmatoria v4: capítulos 3 y 4\n\n" + c34
    text += "\n\n# Capítulo 5. Conclusiones y límites v4\n\n" + c5
    figure_source = ROOT / "outputs/thesis-repair/confirmatory_v4_stable/figure_pack_v4"
    figure_names = [
        "01_capital_holdout_v4.png",
        "02_drawdown_holdout_v4.png",
        "03_sharpe_movil_holdout_v4.png",
        "04_cost_stress_holdout_v4.png",
        "05_acciones_regimen_forward_v4.png",
        "06_semillas_holdout_v4.png",
    ]
    figure_dir = out / "figures"
    figure_dir.mkdir()
    for name in figure_names:
        source = figure_source / name
        if not source.is_file():
            raise SystemExit(f"missing v4 figure: {source}")
        shutil.copy2(source, figure_dir / name)
    text += "\n\n# Figuras v4 generadas desde artefactos congelados\n\n"
    text += "\n\n".join(
        f"![Figura v4 {index + 1}: {name}](figures/{name})"
        for index, name in enumerate(figure_names)
    )
    renderer.render_document(text, "capitulos_3_4_5_v4", out, short=True)
    (out / "source_chapters_3_4_v4.md").write_text(c34, encoding="utf-8")
    (out / "source_chapter_5_v4.md").write_text(c5, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
