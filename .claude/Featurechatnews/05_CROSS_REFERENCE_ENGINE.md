> **Status: ✅ RETAINED** from NewsEngine v1.0.0 — No changes needed for unified platform.

# SDD-05: Cross-Reference Engine

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-05 |
| **Título** | Cross-Reference Engine Specification |
| **Versión** | 1.0.0 |
| **Fecha** | 2025-02-25 |
| **Depende de** | SDD-03, SDD-04 |
| **Requerido por** | SDD-06 |

---

## 1. Objetivo

Detectar cuando la **misma noticia o tema** aparece en **múltiples fuentes** (ej: La República + Portafolio + GDELT reportando sobre la decisión del Banrep). Esto genera:

- `cross_references`: Grupos de artículos relacionados
- `match_score`: Confianza del match (0.0 - 1.0)
- Feature para el modelo: `cross_reference_count` (indicador de importancia)

**Hipótesis de trading:** Una noticia cubierta por 3+ fuentes tiene mayor impacto potencial en USD/COP que una cubierta por 1 sola fuente.

---

## 2. Algoritmo General

```
INPUT: articles[] del día (ya enriched)

1. Agrupar artículos por source_id
2. Si solo hay 1 fuente → RETURN [] (no hay cross-ref posible)
3. Para cada par de artículos (a, b) donde a.source_id ≠ b.source_id:
   a. Calcular similarity(a, b)
   b. Si similarity >= THRESHOLD → marcar como candidato
4. Clusterizar candidatos en grupos (topic clusters)
5. Para cada cluster con 2+ fuentes distintas:
   a. Extraer topic representativo
   b. Calcular match_score del cluster
   c. Almacenar cross_reference
```

---

## 3. Cálculo de Similaridad

### 3.1 Señales y Pesos

| Señal | Peso | Descripción |
|-------|------|-------------|
| **Title keyword overlap** | 0.40 | Jaccard + SequenceMatcher sobre keywords de títulos |
| **Named entity overlap** | 0.30 | Entidades financieras comunes detectadas |
| **Summary keyword overlap** | 0.20 | Jaccard sobre keywords de resúmenes |
| **Category match** | 0.10 | Bonus si misma categoría asignada |

### 3.2 Title Keyword Overlap (40%)

```
def keyword_overlap(title_a: str, title_b: str) -> float:
    words_a = extract_keywords(title_a)  # Remover stopwords, min 3 chars
    words_b = extract_keywords(title_b)

    jaccard = len(words_a & words_b) / len(words_a | words_b)
    seq_ratio = SequenceMatcher(None, sorted(words_a), sorted(words_b)).ratio()

    RETURN jaccard * 0.6 + seq_ratio * 0.4
```

**Stopwords español (a filtrar):**
```
el, la, los, las, un, una, de, del, en, y, que, es, por, con, 
para, se, al, lo, como, más, su, le, ya, o, fue, ha, era, son, 
no, a, e, the, of, and, in, to, for, is, on
```

### 3.3 Named Entity Overlap (30%)

Entidades financieras predefinidas (regex patterns):

| Pattern | Entidad |
|---------|---------|
| `banco\s+de\s+la\s+rep[úu]blica` | Banco de la República |
| `ecopetrol` | Ecopetrol |
| `usd[/\s]?cop` | Par cambiario |
| `petro\b` | Presidente Petro |
| `minhacienda` | Ministerio de Hacienda |
| `superfinanciera` | Superfinanciera |
| `fed\b` | Federal Reserve |
| `opep\|opec` | OPEP/OPEC |
| `wti\|brent` | Benchmarks de crudo |
| `dxy` | Dollar Index |

Además se detectan menciones contextuales:
- `\d+[.,]?\d*\s*%` → `percentage_mention`
- `\$[\d,]+` → `usd_amount`
- `cop\s*[\d,]+` → `cop_amount`

```
def entity_overlap(text_a: str, text_b: str) -> float:
    entities_a = extract_entities(text_a)
    entities_b = extract_entities(text_b)
    
    RETURN jaccard(entities_a, entities_b)
```

### 3.4 Similarity Score Final

```
similarity = (
    title_keyword_overlap * 0.40 +
    entity_overlap * 0.30 +
    summary_keyword_overlap * 0.20 +
    category_match * 0.10
)
```

---

## 4. Threshold

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `SIMILARITY_THRESHOLD` | **0.35** | Calibrado para balancear precision/recall. Demasiado alto (>0.5) pierde matches por diferencia de idioma (GDELT en inglés vs scrapers en español). Demasiado bajo (<0.25) genera falsos positivos. |
| `TIME_WINDOW` | **48 horas** | Artículos de fuentes diferentes pueden publicarse con horas de diferencia |

---

## 5. Clustering

Después de encontrar pares similares, agrupar en clusters:

```
def cluster_matches(pairs: List[Tuple[Article, Article, float]]) -> List[Cluster]:
    """
    Greedy clustering:
    1. Ordenar pares por similarity DESC
    2. Para cada par:
       a. Si ninguno de los dos está en un cluster → crear cluster nuevo
       b. Si uno está en un cluster → agregar el otro al cluster
       c. Si ambos están en clusters diferentes → merge clusters
    3. Filtrar clusters: mantener solo los que tienen 2+ sources distintos
    """
```

---

## 6. Topic Extraction

Para cada cluster, extraer un topic representativo:

```
def extract_topic(cluster: List[Article]) -> str:
    all_keywords = Counter()
    for article in cluster:
        keywords = extract_keywords(article.title)
        all_keywords.update(keywords)
    
    # Keywords que aparecen en 2+ artículos del cluster
    common = {kw: count for kw, count in all_keywords.items() if count > 1}
    
    Si common:
        RETURN " ".join(top 5 keywords por frecuencia)
    Sino:
        RETURN cluster[0].title[:100]  # Fallback: título del primer artículo
```

---

## 7. Output

Cada cross-reference genera:

```
CrossReference:
    id: UUID
    topic: str                  # "banrep tasa interés decisión"
    match_score: float          # Mejor score del cluster
    sources_count: int          # Cantidad de fuentes distintas
    reference_date: date
    article_ids: List[UUID]     # Artículos del cluster
```

---

## 8. Feature para Trading

El cross-reference engine produce features diarios:

| Feature | Cálculo | Interpretación |
|---------|---------|----------------|
| `crossref_count` | Número de cross-references del día | Mayor = más temas de alto impacto |
| `crossref_avg_score` | Promedio de match_score | Mayor = temas más consolidados |
| `crossref_max_sources` | Máximo de sources_count | 3+ = evento de alta importancia |
| `crossref_forex_count` | Cross-refs donde topic contiene keywords forex | Atención mediática en USD/COP |
| `crossref_oil_count` | Cross-refs con keywords de petróleo | Atención en oil |

---

## 9. Ejecución

```
def run_daily_crossref(date: str):
    articles = DB.get_enriched_articles(date)
    
    if len(articles) < 2:
        return []
    
    # Solo comparar entre fuentes diferentes
    pairs = []
    for i, a in enumerate(articles):
        for b in articles[i+1:]:
            if a.source_id != b.source_id:
                sim = calculate_similarity(a, b)
                if sim >= THRESHOLD:
                    pairs.append((a, b, sim))
    
    clusters = cluster_matches(pairs)
    
    for cluster in clusters:
        topic = extract_topic(cluster.articles)
        ref = CrossReference(
            topic=topic,
            match_score=cluster.best_score,
            sources_count=cluster.unique_sources,
            reference_date=date,
        )
        DB.save_cross_reference(ref, cluster.article_ids)
    
    return clusters
```

---

## 10. Complejidad y Optimización

**Complejidad bruta:** O(n²) donde n = artículos del día.
- Con ~500 artículos/día → ~125,000 comparaciones.
- Cada comparación es rápida (string ops), estimado ~0.1ms → ~12.5 seg total.

**Optimizaciones si es necesario:**
1. **Pre-filtro por categoría:** Solo comparar artículos de la misma categoría → reduce n² por factor ~7
2. **Pre-filtro por entidades:** Solo comparar si comparten al menos 1 entidad
3. **MinHash/LSH:** Para approximate nearest neighbors si n > 5,000
