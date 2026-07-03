# GlobalMinds Pitch Deck — Microsoft / Inversores · Mayo 2026

> Pitch deck profesional de 23 slides para presentación a Microsoft for Startups
> y contactos comerciales. Construido reproduciblemente con datos reales del proyecto.

---

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `GlobalMinds_Pitch_Deck.pptx` | Deck principal (23 slides + speaker notes) |
| `convert_to_pdf.sh` | Helper para generar PDF (requiere LibreOffice) |
| `SOURCES.md` | Apéndice de fuentes públicas verificables |
| `assets/screenshots/` | 3 capturas reales del dashboard (Playwright headless) |
| `assets/charts/` | 2 charts con datos reales (`07_equity_curve_2025.png`, `14_mrr_projection.png`) |
| `assets/diagrams/` | 10 diagramas custom (hero, arquitectura, oportunidades, etc.) |

---

## Estructura del deck (23 slides)

| Bloque | Slides | Contenido |
|--------|--------|-----------|
| **Apertura** | 1–3 | Portada · El problema · La oportunidad |
| **Solución** | 4–7 | ¿Qué construimos? · Arquitectura · Demo dashboard · Performance |
| **Comercial** | 8–10 | Las 4 oportunidades · Datos validan · Comparables internacionales |
| **Monetización** | 11–14 | 4 motores ingreso · Pricing · Bot P2P punta de lanza · Proyección MRR |
| **Roadmap** | 15–17 | Estado actual · Roadmap 12 meses · Hitos 90 días |
| **Pedido** | 18–20 | ¿Qué necesitamos de Microsoft? · ¿Qué ofrecemos? · Equipo |
| **Cierre** | 21–22 | Tracción · Call to action + disclaimer |
| **Apéndice** | 23 | Fuentes y referencias |

---

## Cómo generar el PDF

### Opción A — LibreOffice (recomendado, automático)

```bash
sudo apt install libreoffice-impress libreoffice-core
./convert_to_pdf.sh
```

### Opción B — PowerPoint / Keynote (manual)

1. Abrir `GlobalMinds_Pitch_Deck.pptx`
2. Archivo → Exportar → PDF
3. Guardar como `GlobalMinds_Pitch_Deck.pdf` en esta misma carpeta

### Opción C — Google Slides

1. Subir el `.pptx` a Google Drive
2. Abrir con Google Slides
3. Archivo → Descargar → PDF (.pdf)

---

## Cómo regenerar todo (si cambia algo)

```bash
# Desde el root del repo:

# 1. Screenshots del dashboard (requiere localhost:5000 corriendo)
python3 scripts/build_pitch_screenshots.py

# 2. PNG assets (charts y diagramas)
python3 scripts/build_pitch_assets.py

# 3. Deck completo
python3 scripts/build_pitch_deck.py

# 4. PDF (opcional)
cd presentation/globalminds_microsoft_pitch_may2026/
./convert_to_pdf.sh
```

Los 3 scripts son idempotentes y deterministas — siempre producen el mismo deck.

---

## Datos reales usados

- **Backtest 2025**: leído de `usdcop-trading-dashboard/public/data/production/summary_2025.json` y `trades/smart_simple_v11_2025.json`
- **KPIs**: +25.63%, Sharpe 3.35, p-value 0.006, Win rate 82.4%, MaxDD 6.12%, 34 trades (5L/29S)
- **Comparable Buy & Hold**: −14.48% mismo período
- **Screenshots**: capturados en vivo del dashboard Next.js corriendo en `localhost:5000`

---

## Diseño

- **Paleta**: `#0B1426` azul oscuro · `#FFFFFF` blanco · `#C5FF4A` lima accent
- **Tipografía**: Inter (con fallback DejaVu Sans / Liberation Sans)
- **Aspect ratio**: 16:9 widescreen (13.333" × 7.5")
- **Footer**: "Confidencial · GlobalMinds · Mayo 2026"
- **Disclaimer**: prominente en slide 22 (rojo, texto completo)

---

## Speaker notes

Cada slide incluye notas del orador en español con:

- Tiempo estimado por slide (30–60s)
- Hooks y cifras clave a memorizar
- Defensas de datos blindados (ej: contexto del 17% de cobertura FX)
- Tono y pausas recomendadas

Total estimado: **20–25 minutos** de presentación.

---

## Edición rápida pre-presentación

Antes de la reunión, abrí el `.pptx` y editá manualmente:

1. **Slide 20**: agregar apellido de Freddy
2. **Slide 22**: actualizar email/web si cambió
3. **Slide 1**: ajustar fecha si la presentación se mueve

---

*GlobalMinds · IA aplicada al mercado cambiario de Latinoamérica · Mayo 2026*
