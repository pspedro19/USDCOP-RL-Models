#!/usr/bin/env bash
# convert_to_pdf.sh — Convierte GlobalMinds_Pitch_Deck.pptx a PDF
#
# Uso:
#   ./convert_to_pdf.sh
#
# Requiere LibreOffice instalado:
#   sudo apt install libreoffice-impress libreoffice-core
#
# Alternativas si no quieres instalar LibreOffice:
#   1) Abrir el .pptx en PowerPoint / Keynote / Google Slides → Exportar PDF
#   2) Subir a Google Slides (Drive) → Archivo → Descargar → PDF
#   3) Usar https://cloudconvert.com (suba .pptx, descarga .pdf)

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPTX="$DIR/GlobalMinds_Pitch_Deck.pptx"

if [[ ! -f "$PPTX" ]]; then
  echo "ERROR: no encontré $PPTX"
  echo "Ejecuta primero:  python3 scripts/build_pitch_deck.py"
  exit 1
fi

if command -v libreoffice >/dev/null 2>&1; then
  CONVERT="libreoffice"
elif command -v soffice >/dev/null 2>&1; then
  CONVERT="soffice"
else
  echo "✗ No encontré LibreOffice."
  echo ""
  echo "Instalalo con:"
  echo "    sudo apt install libreoffice-impress libreoffice-core"
  echo ""
  echo "O abre $PPTX en PowerPoint/Keynote y exporta a PDF manualmente."
  exit 2
fi

echo "→ Convirtiendo .pptx a PDF con $CONVERT (puede tardar ~30s)..."
"$CONVERT" --headless --convert-to pdf --outdir "$DIR" "$PPTX"

PDF="$DIR/GlobalMinds_Pitch_Deck.pdf"
if [[ -f "$PDF" ]]; then
  echo "✓ $PDF ($(du -h "$PDF" | cut -f1))"
else
  echo "✗ Conversión falló. Revisa la salida arriba."
  exit 3
fi
