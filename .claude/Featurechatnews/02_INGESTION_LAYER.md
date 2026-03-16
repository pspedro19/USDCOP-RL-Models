> **Status: ✅ RETAINED** from NewsEngine v1.0.0 — No changes needed for unified platform.

# SDD-02: Ingestion Layer

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-02 |
| **Título** | Ingestion Layer Specification |
| **Versión** | 1.0.0 |
| **Fecha** | 2025-02-25 |
| **Depende de** | SDD-00, SDD-01 |
| **Requerido por** | SDD-03, SDD-04 |

---

## 1. Responsabilidad

La Ingestion Layer es responsable de:
- Conectar con cada fuente (API o scraper)
- Normalizar la respuesta al schema `RawArticle`
- Manejar errores, retries y rate limiting
- Registrar cada ejecución en `ingestion_log`
- **NO** enriquecer, categorizar ni analizar sentimiento (eso es SDD-04)

---

## 2. Interfaz Base — `SourceAdapter`

Todas las fuentes implementan esta interfaz abstracta:

```
class SourceAdapter(ABC):
    """Interfaz que toda fuente debe implementar."""

    source_id: str          # Identificador único (ej: "gdelt_doc")
    source_type: str        # "api" | "scraper" | "macro"
    is_enabled: bool        # Flag de activación

    @abstractmethod
    def fetch_latest(self, **kwargs) -> List[RawArticle]:
        """Obtiene artículos/datos más recientes."""

    @abstractmethod
    def fetch_historical(self, start_date: date, end_date: date, **kwargs) -> List[RawArticle]:
        """Obtiene artículos/datos en rango de fechas."""

    @abstractmethod
    def health_check(self) -> HealthStatus:
        """Verifica que la fuente esté accesible."""
```

---

## 3. Modelo de Datos — `RawArticle`

Schema normalizado que toda fuente produce:

```
RawArticle:
    # Identificación
    url: str                    # UNIQUE, PK lógica de deduplicación
    source_id: str              # FK a sources table
    ingestion_id: str           # UUID de esta ejecución de ingesta

    # Contenido
    title: str                  # Requerido, min 10 chars
    summary: Optional[str]      # Resumen/descripción
    content: Optional[str]      # Texto completo (si disponible)
    author: Optional[str]
    image_url: Optional[str]

    # Temporal
    published_at: Optional[datetime]  # Fecha de publicación original
    ingested_at: datetime             # Timestamp de ingesta (auto)

    # Metadata de fuente
    source_domain: Optional[str]      # ej: "portafolio.co"
    source_country: Optional[str]     # FIPS code: "CO"
    language: Optional[str]           # ISO 639: "es", "en"

    # GDELT-specific (opcional)
    external_tone: Optional[float]    # Tone score de GDELT (-100 a +100)

    # Raw (para debugging)
    raw_data: Optional[dict]          # JSON original de la fuente
```

**Validaciones en ingesta:**

| Campo | Validación | Acción si falla |
|-------|-----------|-----------------|
| `url` | No vacío, formato URL válido | **Descartar artículo** |
| `title` | No vacío, ≥ 10 chars | **Descartar artículo** |
| `published_at` | Parseable a datetime | Set `None`, usar `ingested_at` |
| `summary` | — | Aceptar vacío |
| `content` | — | Aceptar vacío |

---

## 4. Adapter Specifications

### 4.1 GDELT DOC Adapter

```
GDELTDocAdapter(SourceAdapter):
    source_id = "gdelt_doc"
    source_type = "api"

    config:
        base_url: "https://api.gdeltproject.org/api/v2/doc/doc"
        rate_limit_delay: 1.0 sec
        max_records_per_query: 250
        default_timespan: "24h"
        queries: Dict[str, str]  # De SDD-01 §2.1

    fetch_latest(**kwargs):
        Para cada query en self.queries:
            1. Construir URL con mode=artlist, format=json, timespan=default
            2. GET request con delay
            3. Parsear response["articles"] → List[RawArticle]
            4. Asignar external_tone si disponible
        Deduplicar por URL entre queries
        Return lista unificada

    fetch_tone_timeline(query, timespan) -> DataFrame:
        """Método extra específico de GDELT para features de sentimiento."""
        1. GET con mode=timelinetone
        2. Parsear timeline → DataFrame[datetime, tone]
        Return DataFrame

    fetch_volume_timeline(query, timespan) -> DataFrame:
        """Volumen de cobertura para features."""
        1. GET con mode=timelinevolraw
        2. Parsear timeline → DataFrame[datetime, volume]
        Return DataFrame
```

**Particularidades:**
- GDELT adapter tiene métodos extra (`fetch_tone_timeline`, `fetch_volume_timeline`) que NO producen `RawArticle` sino DataFrames para features directamente.
- Estos métodos son consumidos por la Output Layer (SDD-06), no por la Enrichment Layer.

### 4.2 GDELT Context Adapter

```
GDELTContextAdapter(SourceAdapter):
    source_id = "gdelt_context"
    source_type = "api"

    config:
        base_url: "https://api.gdeltproject.org/api/v2/context/context"
        max_records: 75

    fetch_latest(**kwargs):
        Similar a DOC pero:
        - Retorna snippets a nivel de oración
        - Campo extra: context (string ~600 chars)
        - Usado principalmente para validar menciones precisas
```

### 4.3 NewsAPI Adapter

```
NewsAPIAdapter(SourceAdapter):
    source_id = "newsapi"
    source_type = "api"

    config:
        base_url: "https://newsapi.org/v2"
        api_key: str (de settings)
        daily_request_budget: 1000
        requests_used_today: int (tracked)

    fetch_latest(**kwargs):
        budget_check: Si requests_used_today >= 900, skip (reservar 100)

        1. GET /top-headlines?country=co&category=business (1 request)
        2. GET /everything?domains={COLOMBIA_DOMAINS}&language=es (1 request)
        3. GET /everything?q=Colombia economy&language=en (1 request)
        Total: 3 requests por ejecución

        Parsear articles → List[RawArticle]
        Deduplicar por URL

    fetch_historical():
        NOTA: No disponible en plan Developer (solo 24h)
        Retornar lista vacía con warning log
```

**Budget management:**
- Cada ejecución consume ~3 requests
- Con 6 ejecuciones/día × 3 req = 18 req/día (muy conservador)
- Reservar 100 req/día para queries ad-hoc o debugging

### 4.4 Scraper Adapters (Investing, La República, Portafolio)

```
BaseScraperAdapter(SourceAdapter):
    source_type = "scraper"

    config:
        min_delay: 2 sec
        max_delay: 5 sec
        max_pages: 3 (daily) | 50 (historical)
        user_agents: List[str] (rotación)
        timeout: 30 sec

    _polite_delay():
        sleep(random.uniform(min_delay, max_delay))

    _get_page(url) -> Optional[BeautifulSoup]:
        1. _polite_delay()
        2. GET con headers rotados
        3. Parsear HTML
        4. Return BeautifulSoup o None

    _parse_rss(feed_url) -> List[RawArticle]:
        1. feedparser.parse(feed_url)
        2. Filtrar por keywords de relevancia
        3. Normalizar a RawArticle
```

**Investing.com specifics:**
- Priorizar RSS sobre HTML scraping
- Filtro post-fetch por keywords Colombia/LatAm/commodities
- HTML selectors inestables → RSS como fallback primario

**La República specifics:**
- RSS feeds por sección (más confiable)
- Scraping de 4 secciones con paginación
- Soporte de full article content scraping

**Portafolio specifics:**
- RSS + secciones + homepage
- Sitemap crawling para históricos (`sitemap.xml`)
- Grupo El Tiempo → HTML relativamente estable

### 4.5 FRED Adapter

```
FREDAdapter(SourceAdapter):
    source_id = "fred"
    source_type = "macro"

    config:
        base_url: "https://api.stlouisfed.org/fred"
        api_key: str
        series: Dict[str, str]  # name → series_id

    fetch_latest(**kwargs):
        Para cada serie en self.series:
            GET /series/observations?series_id={id}&observation_start={30d ago}
            Parsear → List[MacroDataPoint]
        Return unificado

    Nota: FRED no produce RawArticle sino MacroDataPoint
    (ver SDD-03 §2.3 para schema)
```

### 4.6 BanRep Adapter

```
BanRepAdapter(SourceAdapter):
    source_id = "banrep"
    source_type = "macro"

    config:
        trm_url: "https://www.datos.gov.co/resource/mcec-87by.json"
        app_token: Optional[str]

    fetch_latest():
        GET TRM con $order=vigenciahasta DESC&$limit=30
        Parsear → List[MacroDataPoint]
```

---

## 5. Source Registry

```
SourceRegistry:
    """Registro central de todas las fuentes."""

    _adapters: Dict[str, SourceAdapter]

    register(adapter: SourceAdapter)
    get(source_id: str) -> SourceAdapter
    get_all_enabled() -> List[SourceAdapter]
    get_by_type(source_type: str) -> List[SourceAdapter]

    health_check_all() -> Dict[str, HealthStatus]
```

Inicialización:
```
registry = SourceRegistry()
registry.register(GDELTDocAdapter(config))
registry.register(GDELTContextAdapter(config))
registry.register(NewsAPIAdapter(config))
registry.register(InvestingScraperAdapter(config))
registry.register(LaRepublicaScraperAdapter(config))
registry.register(PortafolioScraperAdapter(config))
registry.register(FREDAdapter(config))
registry.register(BanRepAdapter(config))
```

---

## 6. Error Handling & Retry

| Error | Acción | Max retries |
|-------|--------|-------------|
| `ConnectionError` | Retry con exponential backoff | 3 |
| `TimeoutError` | Retry con delay × 2 | 2 |
| `HTTPError 429` (rate limited) | Esperar `Retry-After` header o 60s | 3 |
| `HTTPError 403` (blocked) | Log warning, skip fuente | 0 |
| `HTTPError 5xx` | Retry con backoff | 3 |
| `ParseError` (HTML/JSON) | Log error, skip artículo | 0 |
| `ValidationError` (RawArticle) | Log warning, skip artículo | 0 |

**Backoff strategy:**
```
delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
```

---

## 7. Ingestion Log

Cada ejecución de `fetch_latest()` o `fetch_historical()` produce un registro:

```
IngestionLog:
    id: UUID
    source_id: str
    started_at: datetime
    finished_at: datetime
    status: "success" | "partial" | "failed"
    articles_fetched: int
    articles_new: int          # No duplicados
    articles_skipped: int      # Duplicados o inválidos
    error_message: Optional[str]
    execution_time_seconds: float
```

---

## 8. Deduplicación en Ingesta

**Nivel 1 — URL exact match:**
- `INSERT ... ON CONFLICT (url) DO NOTHING`
- Rápido, determinista

**Nivel 2 — Cross-source dedup (en Enrichment, no aquí):**
- Fuzzy title matching para detectar mismo artículo en diferentes fuentes
- Esto se maneja en SDD-05 (Cross-Reference Engine)

---

## 9. Configuración (Pydantic Settings)

```
class IngestionConfig(BaseSettings):
    # GDELT
    gdelt_enabled: bool = True
    gdelt_delay: float = 1.0
    gdelt_default_timespan: str = "24h"
    gdelt_queries: Dict[str, str] = GDELT_QUERIES

    # NewsAPI
    newsapi_enabled: bool = True
    newsapi_key: str
    newsapi_daily_budget: int = 1000

    # Scrapers
    scrapers_enabled: bool = True
    scraper_min_delay: float = 2.0
    scraper_max_delay: float = 5.0
    scraper_max_pages: int = 3

    # FRED
    fred_enabled: bool = True
    fred_api_key: str = ""

    # BanRep
    banrep_enabled: bool = True

    class Config:
        env_prefix = "NEWS_"
        env_file = ".env"
```
