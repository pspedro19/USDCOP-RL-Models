> **Status: ✅ RETAINED** from NewsEngine v1.0.0 — No changes needed for unified platform.

# SDD-04: Enrichment Pipeline

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-04 |
| **Título** | Enrichment Pipeline Specification |
| **Versión** | 1.0.0 |
| **Fecha** | 2025-02-25 |
| **Depende de** | SDD-02, SDD-03 |
| **Requerido por** | SDD-05, SDD-06 |

---

## 1. Responsabilidad

La Enrichment Pipeline toma artículos `RawArticle` ya almacenados y les agrega:

1. **Categoría** (`category`)
2. **Relevancia** (`relevance_score`)
3. **Sentimiento** (`sentiment_score` + `sentiment_source`)
4. **Tags** (`tags`)
5. **Flag de análisis semanal** (`is_weekly_analysis`)

Opera como un paso **post-ingesta**, independiente. Se puede re-ejecutar sobre artículos históricos sin re-ingestar.

---

## 2. Pipeline Flow

```
articles (sin enrichment)
    │
    ▼
┌────────────────┐
│ 1. Categorizer │  → category: "forex" | "oil" | "macro" | "policy" | ...
└───────┬────────┘
        ▼
┌────────────────┐
│ 2. Tagger      │  → tags: ["dólar", "WTI", "Banrep"]
└───────┬────────┘
        ▼
┌────────────────┐
│ 3. Relevance   │  → relevance_score: 0.0 - 1.0
│    Scorer      │
└───────┬────────┘
        ▼
┌────────────────┐
│ 4. Sentiment   │  → sentiment_score: -1.0 a +1.0
│    Analyzer    │     sentiment_source: "gdelt" | "vader"
└───────┬────────┘
        ▼
┌────────────────┐
│ 5. Weekly      │  → is_weekly_analysis: bool
│    Detector    │
└───────┬────────┘
        ▼
articles (enriched)
    → UPDATE en DB
```

---

## 3. Componente 1 — Categorizer

### 3.1 Categorías

| Categoría | ID | Descripción |
|-----------|----|-------------|
| Forex | `forex` | Dólar, USD/COP, tasa de cambio, divisas |
| Oil | `oil` | Petróleo, WTI, Brent, Ecopetrol, OPEP |
| Macro | `macro` | PIB, inflación, desempleo, crecimiento, economía |
| Policy | `policy` | Banco de la República, MinHacienda, reformas, tasas de interés |
| Markets | `markets` | Bolsa, acciones, BVC, COLCAP, bonos, TES |
| Trade | `trade` | Exportaciones, importaciones, balanza comercial, remesas |
| Global | `global` | Fed, Powell, DXY, Treasury, Wall Street |
| Social | `social` | Protestas, paros, seguridad, conflicto |
| General | `general` | No clasificable en las anteriores |

### 3.2 Algoritmo

```
INPUT: title (str), summary (str)
text = normalize(title + " " + summary)

Para cada categoría:
    score = sum(1 por cada keyword de la categoría encontrado en text)

Si max(scores) > 0:
    category = categoría con mayor score
    En caso de empate: priorizar por orden de relevancia para trading
        (forex > oil > policy > macro > markets > global > trade > social > general)
Sino:
    category = "general"

RETURN category
```

### 3.3 Keywords por Categoría

| Categoría | Keywords |
|-----------|----------|
| `forex` | dólar, usd/cop, tasa de cambio, divisa, moneda, cambio, devaluación, revaluación |
| `oil` | petróleo, wti, brent, crudo, ecopetrol, opep, opec, refinería, barril |
| `macro` | pib, inflación, desempleo, crecimiento, economía colombiana, recesión, ipc |
| `policy` | banco de la república, banrep, tasa de interés, ministerio de hacienda, reforma, presupuesto, minhacienda, superfinanciera |
| `markets` | bolsa, acciones, bvc, colcap, rendimiento, bonos, tes, renta fija, renta variable |
| `trade` | exportaciones, importaciones, balanza comercial, remesas, aranceles, tlc |
| `global` | fed, powell, dxy, treasury, wall street, nasdaq, s&p, reserva federal, bce |
| `social` | protesta, paro, huelga, conflicto, seguridad, manifestación |

---

## 4. Componente 2 — Tagger

### 4.1 Algoritmo

```
INPUT: title (str), summary (str)
text = normalize(title + " " + summary)

active_keywords = DB.get_active_keywords()  # tabla keywords
matched = [kw for kw in active_keywords if kw.keyword.lower() in text]

RETURN [kw.keyword for kw in matched]
```

Los tags se almacenan como `JSONB` array: `["dólar", "WTI", "Banrep"]`

---

## 5. Componente 3 — Relevance Scorer

### 5.1 Algoritmo

Calcula relevancia para el trading USD/COP basado en keywords ponderados.

```
INPUT: title (str), summary (str)
text = normalize(title + " " + summary)

keyword_priorities = DB.get_active_keywords()  # {keyword: priority(1-10)}
matched_priorities = [p for kw, p in keyword_priorities if kw.lower() in text]

Si no hay matches:
    RETURN 0.0

# Tomar top 3 keywords más relevantes encontrados
top3 = sorted(matched_priorities, reverse=True)[:3]
max_possible = max(keyword_priorities.values()) * 3

score = sum(top3) / max_possible
RETURN min(score, 1.0)
```

### 5.2 Ejemplos

| Artículo | Keywords encontrados | Prioridades | Score |
|----------|---------------------|-------------|-------|
| "Dólar supera $4,500 tras caída del WTI" | dólar(10), WTI(10) | [10,10] | 20/30 = 0.67 |
| "Fed mantiene tasas, DXY sube" | Fed(9), DXY(9) | [9,9] | 18/30 = 0.60 |
| "Ecopetrol reporta utilidades" | Ecopetrol(8) | [8] | 8/30 = 0.27 |
| "Turismo crece en Cartagena" | — | [] | 0.0 |

---

## 6. Componente 4 — Sentiment Analyzer

### 6.1 Strategy Pattern

```
Si artículo tiene external_tone (viene de GDELT):
    sentiment_score = normalize_gdelt_tone(external_tone)
    sentiment_source = "gdelt"

Sino:
    sentiment_score = vader_analyze(title + " " + summary)
    sentiment_source = "vader"
```

### 6.2 Normalización de GDELT Tone

GDELT tone: -100 a +100 (práctico: -20 a +20)
Target: -1.0 a +1.0

```
def normalize_gdelt_tone(tone: float) -> float:
    """Normaliza GDELT tone a rango [-1, 1]."""
    # Clipear al rango práctico
    clipped = max(min(tone, 20), -20)
    # Normalizar
    return clipped / 20.0
```

### 6.3 VADER para Artículos Scrapeados

Para artículos que NO tienen GDELT tone (scrapers sin match):

```
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def vader_analyze(text: str) -> float:
    scores = analyzer.polarity_scores(text)
    return scores['compound']  # Ya está en rango [-1, 1]
```

**Nota sobre idioma:** VADER funciona mejor en inglés. Para artículos en español, considerar:
- Opción A: Traducir título a inglés via API antes de analizar (costoso)
- Opción B: Usar VADER directamente (funcionalidad reducida pero aceptable)
- Opción C: Usar librería `pysentimiento` (diseñada para español)
- **Decisión:** Opción B para MVP, migrar a C en V2

### 6.4 Prioridad de Sentimiento

| Fuente | Prioridad | Cuando se usa |
|--------|-----------|--------------|
| GDELT tone | 1 (primaria) | Artículos que vienen de GDELT o tienen match |
| VADER | 2 (fallback) | Artículos solo de scrapers sin tone externo |
| Manual | 3 (override) | Correcciones manuales (futuro) |

---

## 7. Componente 5 — Weekly Analysis Detector

### 7.1 Algoritmo

```
INPUT: title (str)
title_lower = title.lower()

indicators = [
    "semana", "semanal", "resumen", "análisis de la semana",
    "perspectiva", "panorama", "cierre semanal", "balance",
    "weekly", "week ahead", "outlook", "review",
    "en la mira", "qué esperar", "lo que viene"
]

RETURN any(indicator in title_lower for indicator in indicators)
```

---

## 8. Ejecución del Pipeline

### 8.1 Batch Mode (normal)

```
def enrich_batch(date: str):
    """Enriquecer todos los artículos de un día."""
    articles = DB.get_unenriched_articles(date)

    for article in articles:
        text = f"{article.title} {article.summary or ''}"

        article.category = categorizer.categorize(text)
        article.tags = tagger.extract(text)
        article.relevance_score = relevance.score(text)
        article.sentiment_score = sentiment.analyze(article)
        article.is_weekly_analysis = weekly_detector.check(article.title)

        DB.update_enrichment(article)
```

### 8.2 Re-enrichment Mode (histórico)

```
def re_enrich_all(start_date, end_date):
    """Re-enriquecer artículos históricos (ej: si cambian keywords)."""
    articles = DB.get_articles_range(start_date, end_date)
    for article in articles:
        # Mismo proceso que batch
        ...
```

### 8.3 Idempotencia

El enrichment es idempotente: ejecutar dos veces produce el mismo resultado. Los campos se sobrescriben, no se acumulan.

---

## 9. Extensiones Futuras (V2)

| Feature | Descripción | Prioridad |
|---------|-------------|-----------|
| NER español | Extracción de entidades nombradas con spaCy/Stanza | Media |
| Topic modeling | LDA/BERTopic para descubrir temas emergentes | Media |
| Sentiment español | `pysentimiento` para sentiment en español nativo | Alta |
| GPT summarization | Resumen automático de artículos largos | Baja |
| Urgency scoring | Detección de noticias "breaking" vs rutinarias | Alta |
