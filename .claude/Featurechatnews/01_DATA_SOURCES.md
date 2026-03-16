> **Status: ✅ RETAINED** from NewsEngine v1.0.0 — No changes needed for unified platform.

# SDD-01: Data Sources Specification

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-01 |
| **Título** | Data Sources Specification |
| **Versión** | 1.0.0 |
| **Autor** | Pedro — Finaipro / Lean Tech |
| **Fecha** | 2025-02-25 |
| **Estado** | Draft |
| **Depende de** | SDD-00 |
| **Requerido por** | SDD-02, SDD-03 |

---

## 1. Inventario de Fuentes

El NewsEngine consume datos de **9 fuentes** organizadas en 3 categorías:

```
FUENTES
├── News APIs (artículos vía API REST)
│   ├── GDELT DOC 2.0
│   ├── GDELT Context 2.0
│   └── NewsAPI.org
│
├── Web Scrapers (artículos vía HTML/RSS parsing)
│   ├── Investing.com
│   ├── La República
│   └── Portafolio
│
└── Macro Data APIs (indicadores económicos)
    ├── FRED (Federal Reserve)
    ├── Banco de la República (vía datos.gov.co)
    └── datos.gov.co (DANE / Socrata)
```

---

## 2. Fuentes de Noticias — APIs

### 2.1 GDELT DOC 2.0

| Atributo | Valor |
|----------|-------|
| **ID interno** | `gdelt_doc` |
| **Base URL** | `https://api.gdeltproject.org/api/v2/doc/doc` |
| **Autenticación** | Ninguna |
| **Rate limit** | Sin límite documentado. Recomendado: 1 req/seg |
| **Formato respuesta** | JSON |
| **Cobertura temporal** | Últimos 3 meses (API). Histórico completo vía BigQuery |
| **Idiomas** | 65 idiomas traducidos a inglés |
| **Actualización** | Cada 15 minutos |
| **Sentimiento incluido** | Sí — `tone` score (-100 a +100, práctico -20 a +20) |

**Campos de respuesta (mode=artlist):**

| Campo | Tipo | Descripción | Mapeo a `RawArticle` |
|-------|------|-------------|----------------------|
| `url` | string | URL del artículo | `url` |
| `url_mobile` | string | URL mobile | — (descartado) |
| `title` | string | Título | `title` |
| `seendate` | string | Fecha en formato `YYYYMMDDTHHMMSSz` | `published_at` (parsear) |
| `socialimage` | string | URL imagen social | `image_url` |
| `domain` | string | Dominio fuente | `source_domain` |
| `language` | string | Idioma detectado | `language` |
| `sourcecountry` | string | País de la fuente (FIPS) | `source_country` |

**Campos de respuesta (mode=timelinetone):**

| Campo | Tipo | Descripción | Uso |
|-------|------|-------------|-----|
| `date` | string | Timestamp del bin (15 min) | Eje X de feature |
| `value` | float | Tone score promedio del bin | Feature de sentimiento |

**Queries predefinidas para el sistema:**

| Query ID | Query String | Categoría |
|----------|-------------|-----------|
| `gdelt_macro` | `"Colombia" (economy OR GDP OR growth OR recession)` | macro |
| `gdelt_inflacion` | `"Colombia" (inflation OR CPI OR "consumer prices")` | macro |
| `gdelt_banrep` | `"Banco de la Republica" (rate OR interest OR monetary)` | policy |
| `gdelt_peso` | `(Colombian peso OR "USD COP" OR "peso devaluation")` | forex |
| `gdelt_petroleo` | `"Colombia" (oil OR petroleum OR Ecopetrol OR WTI)` | oil |
| `gdelt_fiscal` | `"Colombia" (fiscal OR budget OR deficit OR "tax reform")` | policy |
| `gdelt_comercio` | `"Colombia" (trade OR exports OR imports OR tariff)` | trade |
| `gdelt_inversion` | `"Colombia" (investment OR FDI OR "capital flows")` | macro |
| `gdelt_riesgo` | `"Colombia" (sovereign OR "credit rating" OR bonds OR CDS)` | markets |
| `gdelt_politica` | `"Colombia" (political OR president OR congress OR reform)` | policy |
| `gdelt_social` | `"Colombia" (protest OR strike OR unrest OR security)` | social |

**Operadores GDELT relevantes:**

| Operador | Ejemplo | Uso en sistema |
|----------|---------|----------------|
| `sourcecountry:` | `sourcecountry:CO` | Filtrar fuentes colombianas |
| `sourcelang:` | `sourcelang:Spanish` | Solo español |
| `domain:` | `domain:portafolio.co` | Fuente específica |
| `tone>` / `tone<` | `tone<-5` | Filtrar negativos para alertas |
| `near:` | `near5:"peso crisis"` | Proximidad de palabras |
| `theme:` | `theme:ECON_BANKRUPTCY` | Temas del GKG |

**Limitaciones clave:**

- Máximo 250 artículos por request en `artlist`
- Máximo 3 meses de historial en API (para más → BigQuery)
- Delay de 15-30 minutos en datos más recientes
- Tone score es general, no calibrado específicamente para finanzas
- No todos los artículos en español de Colombia aparecen (sesgo anglófono)

---

### 2.2 GDELT Context 2.0

| Atributo | Valor |
|----------|-------|
| **ID interno** | `gdelt_context` |
| **Base URL** | `https://api.gdeltproject.org/api/v2/context/context` |
| **Autenticación** | Ninguna |
| **Diferencia vs DOC** | Busca a nivel de **oración**, no de artículo completo |
| **Respuesta extra** | Incluye snippet de texto con contexto (~600 chars) |

**Uso específico en el sistema:**
- Detección precisa de menciones: ej. buscar `"Banco de la Republica" "interest rate"` retorna solo artículos donde ambas frases aparecen en la misma oración.
- Útil para feature de "precisión de mención" vs el ruido de GDELT DOC.

**Campos adicionales vs DOC 2.0:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `context` | string | Snippet de ~600 chars con la oración que matcheó |

---

### 2.3 NewsAPI.org

| Atributo | Valor |
|----------|-------|
| **ID interno** | `newsapi` |
| **Base URL** | `https://newsapi.org/v2` |
| **Autenticación** | API Key en header `X-Api-Key` |
| **API Key** | `11caa418c21541eead0637ee63dd66bb` |
| **Plan** | Developer (gratis) |
| **Rate limit** | 1,000 requests/día |
| **Cobertura temporal** | Solo últimas 24h en `/everything` (plan Developer) |
| **Sentimiento incluido** | No |

**Endpoints utilizados:**

| Endpoint | Uso | Params clave |
|----------|-----|-------------|
| `/v2/top-headlines` | Titulares Colombia por categoría | `country=co`, `category=business` |
| `/v2/everything` | Búsqueda en medios colombianos | `domains=`, `q=`, `language=es` |
| `/v2/sources` | Inventario de fuentes | `country=co` |

**Campos de respuesta:**

| Campo | Tipo | Mapeo a `RawArticle` |
|-------|------|----------------------|
| `source.name` | string | `source_name` |
| `author` | string | `author` |
| `title` | string | `title` |
| `description` | string | `summary` |
| `url` | string | `url` |
| `urlToImage` | string | `image_url` |
| `publishedAt` | ISO 8601 | `published_at` |
| `content` | string (truncado) | `content` (parcial, truncado a 200 chars) |

**Dominios colombianos configurados:**

| Dominio | Medio | Categoría predominante |
|---------|-------|----------------------|
| `portafolio.co` | Portafolio | Economía, negocios |
| `larepublica.co` | La República | Finanzas, economía |
| `eltiempo.com` | El Tiempo | General |
| `elespectador.com` | El Espectador | General, política |
| `semana.com` | Semana | Análisis político |
| `bloomberglinea.com` | Bloomberg Línea | Finanzas LatAm |

**Limitaciones clave:**

- 1,000 req/día — priorizar headlines sobre everything
- `/everything` solo 24h en plan Developer
- Content truncado a 200 chars (requiere scraping para full text)
- No incluye sentimiento
- En plan Developer no usar en producción

---

## 3. Fuentes de Noticias — Scrapers

### 3.1 Investing.com

| Atributo | Valor |
|----------|-------|
| **ID interno** | `investing` |
| **Base URL** | `https://www.investing.com` |
| **Método primario** | RSS feeds |
| **Método secundario** | HTML scraping de búsqueda |
| **Anti-scraping** | Agresivo (Cloudflare, rate limiting, JS rendering) |
| **Delay requerido** | 3-5 segundos entre requests |
| **Cobertura** | Global con filtro por keywords Colombia/LatAm |

**RSS Feeds configurados:**

| Feed ID | URL | Contenido |
|---------|-----|-----------|
| `inv_commodities` | `https://www.investing.com/rss/news_14.rss` | Commodities (WTI, Brent) |
| `inv_forex` | `https://www.investing.com/rss/news_1.rss` | Forex (USD, DXY) |
| `inv_economy` | `https://www.investing.com/rss/news_2.rss` | Economía global |
| `inv_stocks` | `https://www.investing.com/rss/news_6.rss` | Mercados accionarios |

**Filtros de relevancia para Colombia (post-fetch):**

```
colombia, colombian, cop, peso, ecopetrol,
latam, latin america, emerging markets,
oil, crude, wti, brent, opec,
dollar, dxy, fed, interest rate
```

**Queries de búsqueda HTML:**

```
Colombia, Colombian peso, USD COP, Ecopetrol,
Banco de la Republica Colombia, Petro Colombia economy
```

**Selectores HTML (search results):**

| Elemento | Selector primario | Selector fallback |
|----------|-------------------|-------------------|
| Container | `.searchSectionMain .js-article-item` | `[class*="article"]` |
| Título/Link | `a.title` | `a[href*="/news/"]` |
| Resumen | `.articleDetails` | `p` |
| Fecha | `.dateText` | `time` |

**Limitaciones clave:**

- Anti-scraping agresivo — RSS es mucho más confiable
- HTML selectors cambian frecuentemente
- No hay paginación de búsqueda confiable para históricos
- Content en inglés predominantemente
- Requiere `feedparser` para RSS

---

### 3.2 La República

| Atributo | Valor |
|----------|-------|
| **ID interno** | `larepublica` |
| **Base URL** | `https://www.larepublica.co` |
| **Método primario** | RSS feeds por sección |
| **Método secundario** | HTML scraping de secciones |
| **Anti-scraping** | Bajo (sitio amigable) |
| **Delay requerido** | 2-3 segundos |
| **Cobertura** | Colombia — economía, finanzas, empresas |

**Secciones relevantes:**

| Sección | Path | Contenido |
|---------|------|-----------|
| Economía | `/economia` | Macro, PIB, inflación |
| Finanzas | `/finanzas` | USD/COP, tasas, mercados |
| Globoeconomía | `/globoeconomia` | Fed, WTI, economía mundial |
| Empresas | `/empresas` | Ecopetrol, BVC, corporativo |

**RSS Feeds:**

| Feed | URL |
|------|-----|
| Principal | `https://www.larepublica.co/rss` |
| Economía | `https://www.larepublica.co/economia/rss` |
| Finanzas | `https://www.larepublica.co/finanzas/rss` |
| Globoeconomía | `https://www.larepublica.co/globoeconomia/rss` |

**Selectores HTML:**

| Elemento | Selectores (prioridad) |
|----------|----------------------|
| Artículo container | `article`, `.V_Title a`, `[class*="article"]` |
| Título | `h1, h2, h3, .title, [class*="title"]` |
| Resumen | `p, .summary, .description` |
| Autor | `.author, [class*="author"], .byline` |
| Fecha | `time, [class*="date"], .fecha` (attr `datetime` primero) |
| Imagen | `img` → `data-src` o `src` |

**Extracción de fecha desde URL:**
- Patrón: `/YYYY/MM/DD/` → `re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)`

**Formatos de fecha en texto:**
- `"15 de enero de 2024"` → regex con mapa de meses español
- `"hace X horas"` → `datetime.now()`
- `"dd/mm/yyyy"`, `"yyyy-mm-dd"`

---

### 3.3 Portafolio

| Atributo | Valor |
|----------|-------|
| **ID interno** | `portafolio` |
| **Base URL** | `https://www.portafolio.co` |
| **Método primario** | RSS feeds |
| **Método secundario** | HTML scraping de secciones + homepage |
| **Método histórico** | Sitemap crawling |
| **Grupo editorial** | El Tiempo |
| **Delay requerido** | 2-3 segundos |
| **Cobertura** | Colombia — negocios, economía, empresas |

**Secciones relevantes:**

| Sección | Path | Contenido |
|---------|------|-----------|
| Economía | `/economia` | PIB, inflación, macro |
| Finanzas | `/finanzas` | Dólar, TES, bolsa |
| Negocios | `/negocios` | Empresas, Ecopetrol |
| Internacional | `/internacional` | Fed, petróleo, global |
| Tendencias | `/tendencias` | Análisis, perspectivas |

**Sitemap (para históricos):**
- URL: `https://www.portafolio.co/sitemap.xml`
- Contiene sub-sitemaps por año/mes
- Formato: sitemap index → sitemap por período → URLs de artículos

---

## 4. Fuentes de Datos Macro

### 4.1 FRED (Federal Reserve Economic Data)

| Atributo | Valor |
|----------|-------|
| **ID interno** | `fred` |
| **Base URL** | `https://api.stlouisfed.org/fred` |
| **Autenticación** | API Key (`api_key` param) |
| **Rate limit** | 120 requests/minuto |
| **Formato** | JSON |

**Series configuradas — Colombia:**

| Serie ID | Nombre | Frecuencia | Uso |
|----------|--------|------------|-----|
| `NGDPRSAXDCCOA` | GDP real Colombia | Trimestral | Feature macro |
| `FPCPITOTLZGCOL` | Inflación Colombia (%) | Anual | Feature macro |
| `COLCPIALLMINMEI` | CPI Colombia | Mensual | Feature macro |
| `LRHUTTTTCOM156S` | Desempleo Colombia | Mensual | Feature macro |
| `IRSTCB01COQ156N` | Tasa interés Colombia | Trimestral | Feature policy |

**Series configuradas — Global:**

| Serie ID | Nombre | Frecuencia | Uso |
|----------|--------|------------|-----|
| `DFF` | Fed Funds Rate | Diario | Feature global (clave) |
| `DTWEXBGS` | USD Trade Weighted Index | Diario | Proxy DXY |
| `DGS10` | US 10Y Treasury Yield | Diario | Feature global |
| `DCOILWTICO` | WTI Crude Oil Price | Diario | Feature oil (clave) |
| `VIXCLS` | VIX Volatility Index | Diario | Feature risk |
| `BAMLHE00EHYIEY` | High Yield Spread | Diario | Feature risk |
| `T10Y2Y` | Yield Curve (10Y-2Y) | Diario | Feature macro |

**Endpoint:**
```
GET /fred/series/observations
  ?series_id={SERIES_ID}
  &api_key={KEY}
  &file_type=json
  &observation_start={YYYY-MM-DD}
  &observation_end={YYYY-MM-DD}
  &frequency={d|w|m|q|a}
```

**Campos de respuesta:**

| Campo | Tipo | Mapeo |
|-------|------|-------|
| `date` | string (YYYY-MM-DD) | `date` |
| `value` | string (number o ".") | `value` (parsear a float, "." = NaN) |

---

### 4.2 Banco de la República / datos.gov.co

| Atributo | Valor |
|----------|-------|
| **ID interno** | `banrep` |
| **Endpoint TRM** | `https://www.datos.gov.co/resource/mcec-87by.json` |
| **Autenticación** | App Token opcional (mejora rate limit) |
| **Rate limit** | 1,000 req/hr sin token |
| **Formato** | JSON (API Socrata/SODA) |

**Datasets configurados:**

| Dataset ID | Contenido | Campos clave |
|------------|-----------|-------------|
| `mcec-87by` | TRM (USD/COP oficial) | `vigenciahasta`, `valor` |
| `y2hx-5g5b` | IPC por ciudades | Variable |
| `mzha-bk3v` | Exportaciones | Variable |
| `nz7g-9fqe` | Importaciones | Variable |

**Query SoQL para TRM:**
```
$where=vigenciahasta >= '2024-01-01T00:00:00.000'
$order=vigenciahasta DESC
$limit=5000
```

---

## 5. Matriz de Campos — Normalización a `RawArticle`

Todas las fuentes se normalizan al schema `RawArticle` antes de almacenarse:

| Campo `RawArticle` | GDELT | NewsAPI | Investing | LaRepública | Portafolio | Requerido |
|---------------------|-------|---------|-----------|-------------|------------|-----------|
| `url` | `url` | `url` | `link` (RSS) / `href` | `link` (RSS) / `href` | `link` (RSS) / `href` | **Sí** |
| `title` | `title` | `title` | `title` | `title` | `title` | **Sí** |
| `summary` | — | `description` | `summary` (RSS) | `summary` (RSS) / `p` | `summary` / `p` | No |
| `content` | — | `content` (truncado) | — | `article-body` (scrape) | `article-body` | No |
| `author` | — | `author` | `author` (RSS) | `.author` | `.author` | No |
| `published_at` | `seendate` | `publishedAt` | `published` (RSS) | `time[datetime]` / URL | `time[datetime]` / URL | **Sí** |
| `image_url` | `socialimage` | `urlToImage` | `media:content` | `img[data-src]` | `img[data-src]` | No |
| `source_id` | FK | FK | FK | FK | FK | **Sí** |
| `source_domain` | `domain` | `source.name` | — | — | — | No |
| `source_country` | `sourcecountry` | — | — | `"CO"` (hardcode) | `"CO"` (hardcode) | No |
| `language` | `language` | — | — | `"es"` (hardcode) | `"es"` (hardcode) | No |
| `external_tone` | timeline value | — | — | — | — | No |

---

## 6. Prioridad de Fuentes

Para el modelo de trading, las fuentes tienen diferente valor:

| Prioridad | Fuente | Justificación |
|-----------|--------|---------------|
| 1 (crítica) | GDELT | Sentimiento incluido, cobertura global, gratis ilimitado |
| 2 (alta) | FRED | Datos macro oficiales, series diarias |
| 3 (alta) | BanRep/datos.gov.co | TRM oficial, datos DANE |
| 4 (media) | La República | Mejor análisis macro colombiano |
| 5 (media) | Portafolio | Buena cobertura corporativa |
| 6 (media) | NewsAPI | Buen complemento, pero limitado en free |
| 7 (baja) | Investing.com | Anti-scraping agresivo, contenido en inglés |

**Regla de degradación:** Si una fuente falla, el pipeline continúa con las demás. El feature vector resultante tendrá NaN en los campos de la fuente fallida, que se rellenan con forward-fill o 0 según el campo.
