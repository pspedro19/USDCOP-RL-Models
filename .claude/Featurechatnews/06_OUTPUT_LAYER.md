> **Status: ✅ RETAINED** from NewsEngine v1.0.0 — No changes needed for unified platform.

# SDD-06: Output Layer

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-06 |
| **Título** | Output Layer Specification |
| **Versión** | 1.0.0 |
| **Fecha** | 2025-02-25 |
| **Depende de** | SDD-03, SDD-04, SDD-05 |
| **Requerido por** | SDD-07, Pipeline RL (externo) |

---

## 1. Outputs del Sistema

| Output | Consumidor | Formato | Frecuencia |
|--------|-----------|---------|------------|
| **Feature Vector** | Modelo RL (PPO/SAC) | CSV/Parquet + DB | Diario |
| **Daily Digest** | Humano (Pedro) | Texto/JSON | Diario |
| **Weekly Digest** | Humano (Pedro) | Texto/JSON | Semanal (lunes) |
| **Breaking Alert** | Alerta automática | JSON → Slack/Telegram | Continuo |

---

## 2. Feature Vector — Specification

### 2.1 Estructura

Un registro por día. Cada registro es un vector de ~80 features flat, listo para concatenar con datos de mercado en el pipeline de RL.

### 2.2 Feature Groups

#### Group A: News Volume (12 features)

| Feature | Tipo | Cálculo |
|---------|------|---------|
| `news_total_count` | int | COUNT(articles) del día |
| `news_forex_count` | int | COUNT WHERE category = 'forex' |
| `news_oil_count` | int | COUNT WHERE category = 'oil' |
| `news_macro_count` | int | COUNT WHERE category = 'macro' |
| `news_policy_count` | int | COUNT WHERE category = 'policy' |
| `news_markets_count` | int | COUNT WHERE category = 'markets' |
| `news_trade_count` | int | COUNT WHERE category = 'trade' |
| `news_global_count` | int | COUNT WHERE category = 'global' |
| `news_social_count` | int | COUNT WHERE category = 'social' |
| `news_investing_count` | int | COUNT WHERE source = 'investing' |
| `news_larepublica_count` | int | COUNT WHERE source = 'larepublica' |
| `news_portafolio_count` | int | COUNT WHERE source = 'portafolio' |

#### Group B: Keyword Mentions (14 features)

| Feature | Tipo | Keywords buscados en title+summary |
|---------|------|------------------------------------|
| `mention_wti` | int | wti |
| `mention_brent` | int | brent |
| `mention_dxy` | int | dxy, dólar index |
| `mention_usdcop` | int | usd/cop, usd cop, dólar, tasa de cambio |
| `mention_fed` | int | fed, powell, reserva federal |
| `mention_banrep` | int | banco de la república, banrep |
| `mention_inflation` | int | inflación, ipc |
| `mention_oil` | int | petróleo, crudo |
| `mention_interest_rate` | int | tasa de interés |
| `mention_ecopetrol` | int | ecopetrol |
| `mention_reform` | int | reforma, presupuesto |
| `mention_exports` | int | exportaciones, importaciones |
| `mention_remesas` | int | remesas |
| `mention_bonds` | int | bonos, tes, colcap |

#### Group C: Sentiment Features (22 features)

Generados a partir de GDELT timelines (SDD-02 §4.1 métodos extra):

| Feature | Tipo | Cálculo |
|---------|------|---------|
| `gdelt_tone_avg_{cat}` | float | Promedio tone GDELT por categoría (11 cats) |
| `gdelt_tone_momentum_{cat}` | float | Tone reciente - tone anterior (11 cats) |

Donde `{cat}` ∈ {macro, inflacion, banrep, peso, petroleo, fiscal, comercio, inversion, riesgo, politica, social}

#### Group D: Article-Level Sentiment (6 features)

| Feature | Tipo | Cálculo |
|---------|------|---------|
| `sentiment_avg` | float | AVG(sentiment_score) de todos los artículos |
| `sentiment_std` | float | STD(sentiment_score) |
| `sentiment_min` | float | MIN(sentiment_score) |
| `sentiment_max` | float | MAX(sentiment_score) |
| `sentiment_negative_ratio` | float | % artículos con sentiment < -0.2 |
| `sentiment_positive_ratio` | float | % artículos con sentiment > 0.2 |

#### Group E: GDELT Volume Features (11 features)

| Feature | Tipo | Cálculo |
|---------|------|---------|
| `gdelt_volume_avg_{cat}` | float | Volumen promedio GDELT por categoría |

11 categorías de queries GDELT.

#### Group F: Cross-Reference Features (5 features)

| Feature | Tipo | Cálculo |
|---------|------|---------|
| `crossref_count` | int | Número de cross-references del día |
| `crossref_avg_score` | float | Promedio match_score |
| `crossref_max_sources` | int | Máximo sources_count |
| `crossref_forex_topics` | int | Cross-refs con keywords forex |
| `crossref_oil_topics` | int | Cross-refs con keywords oil |

#### Group G: Relevance Features (4 features)

| Feature | Tipo | Cálculo |
|---------|------|---------|
| `relevance_avg` | float | AVG(relevance_score) |
| `relevance_max` | float | MAX(relevance_score) |
| `relevance_high_count` | int | COUNT WHERE relevance > 0.5 |
| `has_weekly_analysis` | bool→int | ANY(is_weekly_analysis) |

#### Group H: Macro Data (7 features)

| Feature | Tipo | Source |
|---------|------|--------|
| `macro_trm` | float | BanRep — TRM del día |
| `macro_wti` | float | FRED — DCOILWTICO |
| `macro_dxy` | float | FRED — DTWEXBGS |
| `macro_fed_rate` | float | FRED — DFF |
| `macro_us10y` | float | FRED — DGS10 |
| `macro_vix` | float | FRED — VIXCLS |
| `macro_hy_spread` | float | FRED — BAMLHE00EHYIEY |

### 2.3 Total Features

| Group | Count |
|-------|-------|
| A: Volume | 12 |
| B: Mentions | 14 |
| C: GDELT Sentiment | 22 |
| D: Article Sentiment | 6 |
| E: GDELT Volume | 11 |
| F: Cross-Reference | 5 |
| G: Relevance | 4 |
| H: Macro | 7 |
| **TOTAL** | **~81** |

### 2.4 Formato de Export

```
Archivo: features_{start_date}_{end_date}.csv
Encoding: UTF-8
Separador: comma
Header: primera fila
Tipos: float, int (no strings excepto date)

Columna index: date (YYYY-MM-DD)
Missing values: NaN (forward-fill en pipeline RL)
```

Adicionalmente, cada snapshot se guarda en `feature_snapshots` (SDD-03 §2.8) para reproducibilidad.

### 2.5 Integración con Pipeline RL

```python
# En el feature engineering del modelo PPO/SAC:
news_features = pd.read_csv('features_2020-01-01_2025-02-25.csv', 
                            parse_dates=['date'], index_col='date')

# Merge con datos de mercado
df = market_data.join(news_features, how='left')

# Forward-fill para días sin noticias (weekends, holidays)
news_cols = [c for c in df.columns if c.startswith(('news_', 'mention_', 'gdelt_', 
                                                      'sentiment_', 'crossref_', 
                                                      'relevance_', 'macro_'))]
df[news_cols] = df[news_cols].fillna(method='ffill').fillna(0)
```

---

## 3. Daily Digest — Specification

### 3.1 Contenido

| Sección | Contenido |
|---------|-----------|
| Header | Fecha, total artículos |
| Sources | Desglose por fuente |
| Top Stories | Top 5-10 por relevance_score |
| Cross References | Temas cubiertos por múltiples fuentes |
| Market Indicators | Conteo de menciones de indicadores clave |
| Category Breakdown | Distribución por categoría |
| Key Topics | Palabras más frecuentes del día |

### 3.2 Formato Texto

```
============================================================
📊 DAILY NEWS DIGEST — 2025-02-25
============================================================
Total articles scraped: 342

📡 Sources:
   • La República: 125 articles
   • Portafolio: 98 articles
   • GDELT: 85 articles
   • Investing.com: 22 articles
   • NewsAPI: 12 articles

🔥 Top Stories (by relevance):
   1. [forex] Dólar supera los $4,500 tras decisión del Banrep
      Relevance: 0.87 | larepublica.co
   2. [oil] WTI cae por debajo de $70 por temores de recesión
      Relevance: 0.73 | investing.com
   ...

🔗 Cross-Referenced Topics (5):
   📰 "banrep tasa interés" (3 sources, score: 0.72)
   📰 "wti petróleo caída" (2 sources, score: 0.45)
   ...

📈 Market Indicator Mentions:
   dólar                     ████████████████ (42)
   petróleo                  ██████████ (28)
   inflación                 ████████ (21)
   ...

🏷️ Key Topics: dólar, petróleo, banrep, inflación, ...
============================================================
```

### 3.3 Formato JSON

Misma información pero como dict serializable para APIs o storage.

---

## 4. Weekly Digest — Specification

### 4.1 Contenido Adicional vs Daily

| Sección | Contenido |
|---------|-----------|
| Daily Breakdown | Artículos por día (lun-dom) |
| Weekly Analyses | Artículos tipo "análisis semanal" |
| Indicator Trends | Cómo evolucionaron las menciones día a día |
| Top Cross-Refs | Temas más cross-referenciados de la semana |

### 4.2 Ejecución

- Se ejecuta los **lunes** cubriendo lunes anterior → domingo
- Agrega daily digests + análisis de tendencia

---

## 5. Breaking Alert System — Specification

### 5.1 Trigger Conditions

```
CADA 1-2 HORAS:
    1. Ejecutar GDELT search con crisis keywords
    2. Filtrar artículos con tone < -5.0
    3. Contar artículos encontrados

    SI count > 20 → alert_level = "high"
    SI count > 5  → alert_level = "medium"
    SI count > 0  → alert_level = "low"
    SINO          → no alert
```

### 5.2 Crisis Keywords

```
"Colombia crisis", "Colombian peso crash",
"Banco Republica emergency", "Colombia default",
"Ecopetrol crisis", "Colombia downgrade",
"Colombia sanctions", "peso plunge"
```

### 5.3 Alert Payload

```json
{
    "timestamp": "2025-02-25T14:30:00Z",
    "alert_level": "high",
    "count": 23,
    "avg_tone": -8.4,
    "top_articles": [
        {"title": "...", "url": "...", "tone": -12.3},
        ...
    ],
    "recommended_action": "Review positions, check TRM movement"
}
```

### 5.4 Delivery Channels (futuro)

| Canal | Implementación | Prioridad |
|-------|---------------|-----------|
| Console log | Inmediato | MVP |
| Slack webhook | HTTP POST a webhook URL | V1 |
| Telegram bot | Bot API | V1 |
| Email | SMTP / SendGrid | V2 |
