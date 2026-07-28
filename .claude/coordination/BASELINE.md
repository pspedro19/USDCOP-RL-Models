# BASELINE de monitores (comparar DELTAS, no totales) — 2026-07-27T21:40:09-05:00
- test_knowledge_frontmatter: **47 failed pre-existentes** (ADR-0021 sin front-matter,
  BOOK-LEVERAGE, EXP-DIR, .claude/codex/* legacy, etc. — anteriores al protocolo).
  Aprobación = "sin fallos NUEVOS vs esta lista". Saneo = BL futuro, no bloqueo.
- tsc --noEmit dashboard: ~639 líneas de error pre-existentes (ForecastingView
  DIRECTION_TONE ya eliminado; el resto en tests/rutas API no tocados).
- pytest colección: tests/scripts/test_feature_builder.py INTERNALERROR pre-existente
  (get_config del RL).
- Todo lo demás (manifests, layout, caveat/honesty, rbac:check): VERDE = 0 fallos.
