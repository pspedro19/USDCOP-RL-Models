# USDCOP Trading RL Pipeline

Production-ready reinforcement learning pipeline for USDCOP trading with complete data processing from acquisition to serving, featuring an advanced Next.js dashboard for real-time visualization and analytics.

## Architecture

### Data Pipeline Layers

| Layer | DAG | Purpose | Output |
|-------|-----|---------|--------|
| **L0 - Acquire** | `usdcop_m5__01_l0_acquire` | Data acquisition from MT5/TwelveData | Raw 5-minute bars |
| **L1 - Standardize** | `usdcop_m5__02_l1_standardize` | Standardization and quality checks | Clean OHLCV data |
| **L2 - Prepare** | `usdcop_m5__03_l2_prepare` | Technical indicators and preprocessing | 60+ indicators |
| **L3 - Feature** | `usdcop_m5__04_l3_feature` | Feature engineering and selection | 30 curated features |
| **L4 - RLReady** | `usdcop_m5__05_l4_rlready` | RL environment preparation | Episodes with 17 observations |
| **L5 - Serving** | `usdcop_m5__06_l5_serving_final` | Model training and deployment | ONNX model + serving bundle |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.9+
- MinIO (S3-compatible storage)
- Apache Airflow

### Installation

```bash
# Clone repository
git clone <repository-url>
cd USDCOP_Trading_RL

# Install dependencies
pip install -r requirements.txt

# Start services
docker-compose up -d
```

### Running the Pipeline

```bash
# Trigger pipeline layers in sequence
airflow dags trigger usdcop_m5__01_l0_acquire_sync_incremental
airflow dags trigger usdcop_m5__02_l1_standardize
airflow dags trigger usdcop_m5__03_l2_prepare
airflow dags trigger usdcop_m5__04_l3_feature
airflow dags trigger usdcop_m5__05_l4_rlready
airflow dags trigger usdcop_m5__06_l5_serving_final
```

### Monitoring & Dashboards
- **Trading Dashboard**: http://localhost:3001 - Advanced Next.js trading terminal with ML analytics
- **Airflow UI**: http://localhost:8081 - Pipeline orchestration and monitoring
- **MinIO Console**: http://localhost:9001 - S3-compatible storage management
- **Prometheus**: http://localhost:9090 - Metrics and monitoring
- **PgAdmin**: http://localhost:5050 - PostgreSQL database administration

## 🚀 Professional Trading Dashboard

### Overview
The USDCOP Trading Dashboard is a cutting-edge Next.js application built for professional traders and analysts, featuring advanced visualizations, real-time data integration, and machine learning analytics. The dashboard runs on **port 3001** and provides a comprehensive view of the entire trading pipeline.

![Dashboard Status](https://img.shields.io/badge/Status-Live-brightgreen) ![Port](https://img.shields.io/badge/Port-3001-blue) ![Tech](https://img.shields.io/badge/Tech-Next.js%2015-black)

### 🎯 Key Features

#### **Multi-View Trading Terminal**

**📈 Vista Trading Terminal Principal:**
El corazón del dashboard es una **terminal de trading profesional** que reproduce la experiencia de Bloomberg Terminal:
- **Gráfico principal de candlesticks** ocupando el 70% de la pantalla central
- **Panel de precios en tiempo real** con USDCOP actual, cambio porcentual y volumen
- **Barra de herramientas superior** con timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- **Overlays de indicadores técnicos**: RSI, MACD, Bollinger Bands superpuestos elegantemente
- **Controles de reproducción temporal** estilo video player para replay de datos históricos
- **Status bar inferior** mostrando conectividad de datos (L0/L1/Mock) con indicadores LED

**🧠 Dashboard de ML Analytics:**
Una **suite completa de análisis de machine learning** con visualizaciones avanzadas:
- **Gráfico de Performance de Modelos**: Líneas temporales mostrando accuracy de PPO, SAC, DDQN
- **Matriz de Feature Importance**: Heatmap interactivo mostrando qué variables impactan más
- **Panel Predictions vs Actuals**: Scatter plots y líneas de tendencia comparando predicciones vs realidad
- **Monitor de Drift**: Alertas visuales cuando los modelos se desvían de la normalidad
- **ONNX Model Status**: Panel técnico mostrando latencia de inferencia y health checks

**🛡️ Suite de Risk Management:**
**Dashboards especializados en gestión de riesgo** con alertas visuales prominentes:
- **Risk Exposure Heatmap**: Mapa de calor mostrando exposición por sectores/timeframes
- **VaR (Value at Risk) Gauges**: Medidores circulares estilo speedometer para riesgo actual
- **Alerts Center**: Panel tipo "mission control" con alertas críticas destacadas en rojo
- **Portfolio Breakdown**: Pie charts y donut charts mostrando distribución de posiciones
- **Drawdown Analysis**: Gráficos de área mostrando períodos de pérdidas históricas

**🔄 Pipeline Health Monitor:**
**Visualización del estado completo del pipeline** L0-L5 con códigos de color únicos:
- **Flow diagram horizontal** mostrando L0→L1→L2→L3→L4→L5 con flechas animadas
- **Status cards por capa**: Cada capa (L0-L5) tiene su card con métricas específicas
- **Data quality indicators**: Barras de progreso mostrando % de datos válidos por capa
- **Processing time metrics**: Cronómetros mostrando tiempo de procesamiento de cada DAG
- **Error logs integrados**: Panel expandible con logs de errores por capa

#### **Advanced Charting & Visualization**

**📊 Sistema de Gráficos de Nivel Profesional:**
El dashboard integra **múltiples librerías de charting** para crear visualizaciones que rivalizan con Bloomberg y TradingView:

**🕯️ Candlestick Charts Principales:**
- **Lightweight Charts de TradingView**: Para gráficos de precios principales con performance optimizada
- **Candlesticks con volumen**: Barras de volumen sincronizadas bajo el gráfico principal
- **Overlays técnicos transparentes**: RSI, MACD, Bollinger Bands como overlays semi-transparentes
- **Pattern recognition visual**: Patrones como Hammer, Doji, Engulfing destacados con colores únicos
- **Zoom y pan fluidos**: Navegación suave con mouse wheel y drag para explorar datos históricos

**🎨 Visualizaciones Avanzadas D3.js:**
- **Heatmaps interactivos**: Para correlaciones de features y risk exposure con hover details
- **Network diagrams**: Mostrando relaciones entre variables de ML con nodos animados
- **Sankey diagrams**: Para visualizar flujo de datos through el pipeline L0-L5
- **Force-directed graphs**: Para mostrar clustering de comportamientos de mercado
- **Custom animated transitions**: Morphing entre diferentes vistas de datos con suavidad

**📈 Charts Especializados Plotly.js:**
- **3D Surface plots**: Para visualizar performance de modelos ML en múltiples dimensiones
- **Scatter plots interactivos**: Predictions vs actuals con zoom y brush selection
- **Box plots animados**: Para mostrar distribuciones de returns y volatilidad
- **Violin plots**: Distribución de features con density curves elegantes
- **Subplots sincronizados**: Múltiples timeframes alineados temporalmente

#### **Real-Time Data Integration & Streaming**

**🔴 Datos en Vivo con Latencia Ultra-Baja:**
El sistema maneja **84,455+ puntos de datos históricos** desde enero 2020 con streaming en tiempo real:

**⚡ WebSocket Streaming Architecture:**
- **Conexiones WebSocket dedicadas** para cada tipo de dato (precios, alertas, ML predictions)
- **Heartbeat system**: Ping/pong cada 30 segundos para mantener conexiones vivas
- **Auto-reconnection inteligente**: Reconexión automática con exponential backoff
- **Message queuing**: Buffer local para manejar bursts de datos sin pérdida
- **Real-time price updates**: USDCOP tick data actualizado cada segundo

**📊 Multi-Source Data Integration:**
- **L0 Raw Data**: Datos crudos de TwelveData API con 5-minute OHLCV bars
- **L1 Processed Data**: 2.06M+ puntos procesados con quality metrics integrados
- **L3 Feature Data**: Features engineered en tiempo real con correlation updates
- **L5 ML Predictions**: Inferencias de modelos ONNX servidas con <100ms latency
- **Mock Data Mode**: Simulador para testing y demos sin consumir API real

**🚨 Sistema de Alertas Inteligentes:**
- **Price alerts visuales**: Notificaciones tipo toast cuando USDCOP cruza niveles clave
- **ML confidence alerts**: Avisos cuando models confidence < 70%
- **Risk threshold alerts**: Alertas rojas cuando VaR excede límites predefinidos
- **Pipeline health alerts**: Notificaciones cuando DAGs L0-L5 fallan o se retrasan
- **API rate limit warnings**: Advertencias amarillas cuando se acerca al límite de TwelveData

**⚙️ Performance & Caching Inteligente:**
- **Redis caching layer**: Cache de 15 minutos para datos históricos frecuentemente accedidos
- **Smart prefetching**: Precarga datos likely to be needed based en user behavior
- **Compression algorithms**: Datos comprimidos para reducir bandwidth en 60%
- **Virtual scrolling**: Solo renderiza datos visibles para manejar datasets masivos
- **Progressive loading**: Carga inicial rápida, detalles adicionales on-demand

#### **Machine Learning Analytics**
- **Model Performance Dashboard** with accuracy metrics and drift detection
- **Feature Importance Analysis** with interactive visualization
- **Predictions vs Actuals** comparison charts
- **RL Agent Monitoring** for PPO, SAC, and DDQN models
- **Real-time Inference** with ONNX model serving

### 🎨 Design & Visual Experience

#### **Estilo Terminal Profesional Bloomberg-Inspired**
El dashboard presenta una **interfaz visual impresionante** que combina la sofisticación de terminales financieros profesionales como Bloomberg con tecnología web moderna:

**🌟 Paleta de Colores Profesional:**
- **Fondo principal**: Negro slate profundo (slate-950/slate-900) que reduce fatiga visual
- **Acentos primarios**: Cyan brillante (#06b6d4) para datos críticos y alertas positivas
- **Acentos secundarios**: Púrpura vibrante (#8b5cf6) para elementos de ML y análisis avanzado
- **Colores semánticos**: 
  - Verde esmeralda (#10b981) para ganancias, estados "UP" y confirmaciones
  - Rojo coral (#ef4444) para pérdidas, alertas críticas y riesgos
  - Amarillo ámbar (#f59e0b) para advertencias y estados pendientes
  - Azul índigo (#6366f1) para datos L0-L4 de pipeline

**✨ Efectos Visuales Glassmorphism:**
- **Transparencias sofisticadas** con backdrop-blur que crean profundidad visual
- **Sombras suaves tipo vidrio** que dan sensación de flotación a los componentes
- **Bordes sutiles translúcidos** que definen áreas sin crear pesadez visual
- **Gradientes dinámicos** que se mueven sutilmente en el fondo
- **Efectos de brillo (glow)** en elementos importantes como precios y alertas

#### **Arquitectura de Interfaz Multi-Panel**

**🔧 Sidebar Izquierdo - Control Center:**
Ubicado en la **esquina izquierda**, este panel se comporta como una **consola de control central**:
- **Panel superior**: Muestra el precio actual de USDCOP en tiempo real con colores dinámicos
- **Controles de reproducción**: Botones estilo media player (▶️ ⏸️ ⏹️) para controlar datos históricos
- **Selector de fuente**: Badges elegantes para cambiar entre L0 (crudo), L1 (procesado), Mock
- **Status indicators**: Luces LED virtuales que indican estado del mercado y conexiones
- **Se colapsa inteligentemente** en pantallas pequeñas manteniendo funcionalidad esencial

**🗂️ Sidebar Derecho - Navigation Hub:**
El **panel de navegación principal** presenta una experiencia similar a un explorador profesional:
- **16 vistas organizadas** en 4 categorías cromáticamente diferenciadas:
  - 📈 **Trading** (cyan): Terminal principal, gráficos tiempo real, señales ML
  - 🛡️ **Risk** (rojo/coral): Monitoreo de riesgo, alertas, análisis de exposición  
  - 🔄 **Pipeline** (gradiente multicolor): Estados L0-L5 con códigos de color únicos
  - ⚙️ **System** (amarillo/ámbar): Salud del sistema, uso de API, herramientas legacy
- **Iconos intuitivos** de Lucide React que comunican función instantáneamente
- **Animaciones suaves** al hacer hover que elevan los elementos
- **Indicadores de estado** con pulsos luminosos para vistas activas

**🖥️ Área Central - Vista Principal:**
El **espacio de trabajo principal** domina la pantalla con:
- **Gráficos full-screen** de candlesticks profesionales estilo TradingView
- **Grids responsivos** que se adaptan automáticamente al contenido
- **Overlays transparentes** para información contextual sin obstruir datos
- **Transiciones cinematográficas** entre vistas con efectos de desvanecimiento

#### **Experiencia Visual Inmersiva**

**🌊 Fondo Dinámico Inteligente:**
- **Gradiente base** que va del negro profundo a tonos slate con sutiles variaciones
- **Orbes luminosos animados** que flotan suavemente creando profundidad
- **Grid pattern sutil** que proporciona estructura visual sin distraer
- **Efectos parallax ligeros** que responden al movimiento del cursor

**⚡ Animaciones y Micro-interacciones:**
- **Hover effects sofisticados**: Los elementos se elevan con sombras más profundas
- **Loading spinners elegantes**: Múltiples anillos concéntricos con colores gradient
- **Transiciones de estado suaves**: Los números cambian con animación de conteo
- **Pulse effects en tiempo real**: Los indicadores "live" pulsan con el latido del mercado

**📱 Adaptabilidad Visual Inteligente:**
- **Breakpoints inteligentes** que reorganizan completamente la interfaz según el dispositivo
- **Sidebars que se transforman** en navigation drawers en móviles
- **Escalado automático de fuentes** manteniendo legibilidad en todas las pantallas
- **Touch targets optimizados** para interacción móvil sin sacrificar precisión desktop

#### **Detalles de Experiencia Premium**

**🎯 Indicadores Visuales Profesionales:**
- **Status dots animados**: Verde pulsante para "online", rojo intermitente para errores
- **Progress bars con gradientes**: Indican progreso de carga de datos con colores semánticos
- **Badge system sofisticado**: Categorías de vistas con colores distintivos y esquinas redondeadas
- **Tooltips contextualles**: Información adicional que aparece con timing perfecto

**💫 Efectos de Profundidad y Capas:**
- **Z-index inteligente**: Modals y overlays aparecen con backdrop blur correcto
- **Shadow system coherente**: Desde sombras sutiles hasta dramatic drop-shadows
- **Border system unificado**: Bordes que van desde casi invisibles hasta accent brillantes
- **Opacity layers**: Múltiples niveles de transparencia que crean jerarquía visual

**🔄 Feedback Visual en Tiempo Real:**
- **Color coding dinámico**: Los precios cambian de color según dirección (verde up, rojo down)
- **Loading states visuales**: Cada componente tiene su estado de carga único y elegante
- **Error boundaries visuales**: Los errores se muestran con diseño consistente y recovery options
- **Success confirmations**: Acciones completadas se confirman con green checkmarks animados

Esta experiencia visual está **específicamente diseñada para traders profesionales** que necesitan procesar información rápidamente, con cada color, animación y elemento posicionado estratégicamente para **maximizar eficiencia y reducir fatiga visual** durante sesiones de trading extensas.

### 📊 Technical Architecture

#### **Frontend Stack**
```typescript
{
  "framework": "Next.js 15.5.2",
  "ui": "React 19 + TypeScript",
  "styling": "Tailwind CSS 4.0 + Custom Glassmorphism",
  "charts": "D3.js + Lightweight Charts + Plotly.js + Recharts",
  "animations": "Framer Motion 12.23",
  "state": "Zustand + React Context",
  "data": "SWR + Axios + WebSocket",
  "auth": "NextAuth.js + Session Management"
}
```

#### **Component Architecture**
```
usdcop-trading-dashboard/
├── app/                              # Next.js App Router
│   ├── api/                         # API endpoints
│   │   ├── data/                    # Data layer APIs (L0-L5)
│   │   ├── trading/                 # Trading signals & analytics
│   │   ├── market/                  # Market data & health
│   │   └── ml-analytics/            # ML model endpoints
│   ├── page.tsx                     # Main dashboard page
│   └── layout.tsx                   # Root layout with metadata
├── components/                       # Reusable components
│   ├── charts/                      # Advanced charting components
│   │   ├── LightweightChart.tsx     # TradingView-style charts
│   │   ├── InteractiveChart.tsx     # D3.js interactive plots
│   │   └── AdvancedTechnicalIndicators.tsx
│   ├── views/                       # Dashboard views
│   │   ├── EnhancedTradingDashboard.tsx  # Main trading terminal
│   │   ├── RealTimeChart.tsx        # Live market visualization
│   │   ├── L5ModelDashboard.tsx     # ML model dashboard
│   │   └── RiskManagement.tsx       # Risk analytics suite
│   ├── ui/                          # UI primitives
│   │   ├── AnimatedSidebar.tsx      # Collapsible sidebar system
│   │   ├── MobileControlsBar.tsx    # Mobile trading controls
│   │   └── SidebarToggleButtons.tsx # Navigation controls
│   ├── ml-analytics/                # ML-specific components
│   │   ├── ModelPerformanceDashboard.tsx
│   │   ├── FeatureImportanceChart.tsx
│   │   └── PredictionsVsActualsChart.tsx
│   └── common/                      # Shared utilities
│       ├── ErrorBoundary.tsx        # Error handling
│       └── GracefulDegradation.tsx  # Progressive enhancement
├── lib/                             # Utility libraries
├── hooks/                           # Custom React hooks
├── styles/                          # Global styles & themes
└── server/                          # WebSocket server
    └── websocket-server.js          # Real-time data streaming
```

#### **Data Integration Layer**

**API Endpoints**
- `/api/data/l0` - Raw market data from acquisition layer
- `/api/data/l1` - Standardized OHLCV data with quality metrics
- `/api/data/l3` - Feature-engineered dataset with correlations
- `/api/data/l5` - Model predictions and serving data
- `/api/trading/signals` - ML-powered trading signals
- `/api/market/realtime` - Live market data streaming
- `/api/ml-analytics/models` - Model performance metrics

**Data Sources & Backends**
- **PostgreSQL** - Primary data storage for trading data
- **Redis** - Real-time caching and session management  
- **MinIO (S3)** - Pipeline data storage for L0-L5 buckets
- **TwelveData API** - External market data provider
- **WebSocket Server** - Real-time data streaming to frontend

### 🔧 Configuration & Deployment

#### **Environment Setup**
```bash
# Navigate to dashboard directory
cd usdcop-trading-dashboard

# Install dependencies
npm install

# Start development server
npm run dev              # Starts on http://localhost:3001
npm run dev:all          # Includes WebSocket server

# Production build
npm run build
npm start
```

#### **Environment Variables**
```bash
# .env.local
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8080
NEXTAUTH_SECRET=your-secret-key
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_db
REDIS_URL=redis://localhost:6379
```

### 📱 Mobile & Responsive Features

#### **Mobile-First Design**
- **Adaptive Sidebar System** that transforms into drawer navigation on mobile
- **Touch-Optimized Controls** with larger hit targets and gesture support
- **Mobile Controls Bar** with essential trading functions at bottom
- **Responsive Charts** that maintain readability across all screen sizes
- **Progressive Web App** capabilities with offline support

#### **Performance Optimizations**
- **Server-Side Rendering** with Next.js for fast initial loads
- **Component Lazy Loading** for improved bundle splitting
- **Virtual Scrolling** for large datasets and chart data
- **Optimized Images** with Next.js Image component
- **Intelligent Caching** with SWR for data fetching

### 🔐 Security & Authentication

#### **Authentication System**
- **NextAuth.js Integration** with session-based authentication
- **Role-Based Access Control** for different user types
- **Session Persistence** with both sessionStorage and localStorage fallback
- **Automatic Session Renewal** to prevent data loss
- **Secure API Routes** with authentication middleware

#### **Security Features**
- **CSRF Protection** built into Next.js API routes
- **XSS Prevention** through React's built-in sanitization
- **Secure Headers** configured in next.config.js
- **Environment Variable Protection** for sensitive data
- **API Rate Limiting** to prevent abuse

### 👥 Experiencias de Usuario por Perfil

#### **🔧 Para Desarrolladores & Expertos Técnicos**

**Entorno de Desarrollo Profesional:**
- **Codebase completamente tipado** en TypeScript con strict mode enabled
- **Hot reload ultrarrápido** con Next.js 15 y optimizaciones Turbopack
- **DevTools integradas**: ESLint, Prettier, TypeScript con configuraciones estrictas
- **API REST completa** con endpoints documentados y ejemplos de uso
- **Component isolation**: Cada componente es independiente y reutilizable
- **Performance profiling**: Built-in metrics con Web Vitals y custom timing
- **Error boundaries inteligentes**: Catch de errores con recovery automático
- **Git hooks**: Pre-commit con linting y type checking automático

**Arquitectura Extensible:**
- **Custom hooks**: useWebSocket, useMarketData, useMLPredictions reutilizables
- **Context providers**: Estado global con Zustand para performance óptima
- **Styled system**: Tailwind CSS con custom utilities para glassmorphism
- **API middleware**: Interceptors para autenticación, error handling, y retry logic

#### **📈 Para Traders & Usuarios de Negocio**

**Experiencia Trading Intuitiva:**
Al abrir el dashboard, el trader encuentra una **experiencia familiar pero potenciada**:

**🚀 Primera Sesión (Onboarding):**
1. **Login simple**: admin/admin123 → acceso inmediato sin configuraciones complejas
2. **Vista Trading Terminal por defecto**: Gráfico USDCOP ocupando toda la pantalla
3. **Precio actual prominente**: 4,150.25 COP en fuente grande con +15.75 (+0.38%) en verde
4. **Controls intuitivos**: Play ▶️ para datos en vivo, Pause ⏸️ para análisis histórico

**💼 Flujo de Trabajo Diario:**
- **Morning briefing**: Pipeline Health muestra estado de datos overnight (L0-L5 all green)
- **Market analysis**: Click en "Real-Time Chart" → análisis técnico completo con indicadores
- **Signal review**: "Trading Signals" panel muestra recomendaciones ML con confidence scores
- **Risk check**: "Risk Management" dashboard con exposure actual y VaR calculations
- **Position monitoring**: Alerts automáticas cuando límites de riesgo se acercan

**📊 Análisis Visual Sin Complejidad Técnica:**
- **Colores intuitivos**: Verde = good/up, Rojo = bad/down, Amarillo = warning
- **Iconos universales**: ▶️▶️ play, 📈 trending up, 🛡️ shield for risk, ⚠️ alerts
- **Tooltips contextuales**: Hover sobre cualquier elemento explica qué significa
- **Export one-click**: Botón "Export PDF" genera reportes profesionales instantáneamente

#### **🏛️ Para Analistas & Risk Managers**

**Centro de Comando Analítico:**
El dashboard se convierte en un **centro de control de riesgo** completo:

**📊 Dashboard Ejecutivo:**
- **Vista panorámica**: 16 paneles organizados por categoría con colores distintivos
- **KPIs prominentes**: VaR, Sharpe Ratio, Max Drawdown en cards de gran tamaño
- **Health overview**: Semáforo de colores para cada subsistema (trading, risk, ML, pipeline)
- **Executive summary**: Panel superior con métricas más críticas siempre visibles

**🔍 Análisis Profundo:**
- **Drill-down capability**: Click en cualquier métrica revela breakdown detallado
- **Historical comparisons**: Ventanas side-by-side comparando períodos
- **Correlation analysis**: Heatmaps mostrando qué factors mueven el mercado
- **Scenario analysis**: "What-if" scenarios con modelos ML en tiempo real

**📋 Reportes & Compliance:**
- **Audit trails automáticos**: Cada decisión y señal queda registrada con timestamp
- **Regulatory reports**: Formatos precargados para reportes regulatorios
- **Risk attribution**: Breakdown de dónde viene el riesgo (temporal, sectorial, model)
- **Model validation**: Backtesting results con statistical significance tests

**⚡ Alertas Inteligentes Personalizadas:**
- **Thresholds personalizables**: Cada risk manager puede configurar sus límites
- **Escalation paths**: Alertas críticas se envían por múltiples canales
- **Contextual information**: Alertas incluyen "why this happened" y "recommended actions"
- **Historical context**: "Last time this happened..." información para decision making

#### **🌍 Accesibilidad & Inclusividad**

**Diseño Universal:**
- **Multiple languages**: UI preparada para i18n (currently English/Spanish)
- **Keyboard navigation**: Todos los controles accesibles via keyboard shortcuts
- **Screen reader compatible**: Semantic HTML y ARIA labels apropiados
- **High contrast mode**: Colores ajustables para visibilidad óptima
- **Font scaling**: Soporte para zoom hasta 200% sin pérdida de funcionalidad

**Responsive & Mobile-First:**
- **Breakpoints inteligentes**: 3 layouts distintos (mobile, tablet, desktop)
- **Touch-optimized**: Controles táctiles de 44px+ para usabilidad móvil
- **Offline capability**: PWA features con caching para funcionalidad básica offline
- **Network awareness**: Adjusts functionality based en connection quality

### 🚀 Getting Started

1. **Start the Backend Services** (if not already running)
   ```bash
   docker-compose up -d
   ```

2. **Access the Trading Dashboard**
   ```bash
   # Navigate to dashboard
   cd usdcop-trading-dashboard
   
   # Install dependencies (first time only)
   npm install
   
   # Start development server
   npm run dev
   ```

3. **Open in Browser**
   ```
   🌐 Main Dashboard: http://localhost:3001
   🔑 Default Login: admin / admin123
   ```

4. **Explore the Views**
   - Start with **Trading Terminal** for market overview
   - Check **Pipeline Health** to monitor data processing
   - Review **ML Analytics** for model performance
   - Use **Risk Management** for exposure analysis

### 🎬 Impresiones Visuales del Dashboard

#### **Primera Impresión al Abrir http://localhost:3001:**
```
🌑 FONDO: Negro profundo con gradientes cyan/purple sutiles flotando
├─ 🔵 Orbes de luz animados creando profundidad atmosférica  
├─ ⚡ Grid pattern de líneas cyan muy tenues como estructura base
└─ 🌌 Efecto parallax ligero que responde al cursor

┌─ SIDEBAR IZQUIERDO (Control Center) ─┐
│ 💲 USDCOP: 4,150.25 COP ↗️ +0.38%   │
│ 🟢 Live ● Market Open                │  
│ ▶️ ⏸️ ⏹️ [Replay Controls]           │
│ 📊 L1 📡 L0 🧪 Mock [Sources]       │
└─ 🎛️ [Advanced Controls Panel]       ┘

┌─ ÁREA CENTRAL (Main Trading View) ─────────────────────────────────┐
│                                                                    │
│     📈 USDCOP/USD - 5M Candlestick Chart (TradingView Style)      │
│   4,200 ┼─────────────────────🕯️🕯️🕯️─────────────────────        │
│         │                                                          │  
│   4,150 ┼─────🕯️────🕯️🕯️──────────🕯️🕯️─────────────            │
│         │                                                          │
│   4,100 ┼─────────────────────────────────────────────────        │
│         └────────────────────────────────────────────────        │
│         Jan    Feb    Mar    Apr    May    Jun    Jul             │
│                                                                    │
│   📊 RSI (68.3) 📊 MACD (divergence) 📊 BB (squeeze) [Overlays]  │
└─ 🔍 Zoom Controls • 📅 Timeframes • 🎨 Drawing Tools             ┘

┌─ SIDEBAR DERECHO (Navigation Hub) ─┐
│ ✨ Navigation Hub - 16 Views       │
│                                     │
│ 📈 TRADING                          │
│ ● Enhanced Terminal    [ACTIVE] 🔵  │
│ ○ Real-Time Chart                   │
│ ○ Trading Signals                   │  
│ ○ ML Analytics                      │
│                                     │
│ 🛡️ RISK                            │
│ ○ Risk Management                   │
│ ○ Real-Time Monitor                 │
│ ○ Exposure Analysis                 │
│ ○ Alerts Center                     │
│                                     │
│ 🔄 PIPELINE                         │
│ ○ L0 Raw Data        [🟢 HEALTHY]  │
│ ○ L1 Features        [🟢 HEALTHY]  │ 
│ ○ L3 Correlations    [🟢 HEALTHY]  │
│ ○ L5 Model Serving   [🟡 LOADING]  │
│                                     │
│ ⚙️ SYSTEM                          │
│ ○ Pipeline Health     [🟢 ONLINE]  │
│ ○ API Usage          [🟡 75%]     │
└─ v2.1.0 • All Systems Online 🟢   ┘
```

#### **Transición Visual Entre Vistas:**
Cuando el usuario hace click en "ML Analytics":
```
🎭 ANIMACIÓN DE TRANSICIÓN (300ms):
├─ Vista actual: Fade out con scale(0.95)
├─ Loading spinner: Anillos concéntricos cyan/purple  
├─ Nueva vista: Fade in desde scale(0.95) → scale(1.0)
└─ Sidebar indicator: Pulso luminoso se mueve al nuevo item
```

#### **Vista ML Analytics Resultante:**
```
┌─ ML PERFORMANCE DASHBOARD ─────────────────────────────────────────┐
│                                                                    │
│  📊 Model Accuracy Over Time        🧠 Feature Importance          │
│  ┌─PPO: 89.2% ↗️ ┐  ┌─SAC: 87.5%─┐   ┌─RSI: ████████ 89%─┐      │
│  │     Green line  │  │ Orange line │   │ MACD: ██████ 76%  │      │  
│  │     trending ↗️  │  │ trending ↗️  │   │ Volume: ████ 64% │      │
│  └─────────────────┘  └─────────────┘   └───────────────────┘      │
│                                                                    │
│  🎯 Predictions vs Actuals           ⚠️ Model Health Alerts       │  
│  ┌─Scatter plot with━┐  ┌─Correlation─┐   🟢 All Models Healthy    │
│  │ diagonal trend   │  │ line r=0.94  │   🟡 Slight drift detected │
│  │ showing accuracy │  │ (very good)  │   📈 Retraining scheduled  │
│  └─────────────────┘  └─────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

#### **Experiencia Mobile (Responsive Transform):**
```
📱 MOBILE VIEW (< 768px):
┌─ Header Bar ─────────────────────────────────┐
│ ☰ Menu   🪙 USDCOP 4,150.25 ↗️ +0.38%  👤 │
└─────────────────────────────────────────────┘
┌─ Main Chart (Full Width) ───────────────────┐
│        📈 Candlestick Chart                  │
│    (Optimized for touch interaction)        │
└─────────────────────────────────────────────┘
┌─ Mobile Controls (Bottom Bar) ──────────────┐
│ ▶️ ⏸️ │ 📊 L1 │ 🔴 Live │ 📈 5M │ ⚙️      │
└─────────────────────────────────────────────┘

☰ Menu Drawer (Slides from Left):
┌─ Navigation Drawer ─┐
│ ✨ 16 Professional │ 
│ Views Available     │
│                     │
│ 📈 Trading (4)      │
│ 🛡️ Risk (4)        │  
│ 🔄 Pipeline (6)     │
│ ⚙️ System (2)       │
└─────────────────────┘
```

#### **Paleta de Colores Definitiva en Acción:**
```
🎨 COLOR SYSTEM IN USE:
├─ 🖤 Base: slate-950 (Background principal)
├─ 🔵 Primary: #06b6d4 (Cyan - Price up, confirmations) 
├─ 🟣 Secondary: #8b5cf6 (Purple - ML/AI elements)
├─ 🟢 Success: #10b981 (Green - Profit, healthy states)
├─ 🔴 Danger: #ef4444 (Red - Loss, critical alerts)
├─ 🟡 Warning: #f59e0b (Yellow - Caution, pending)
├─ 🔵 Info: #6366f1 (Indigo - Pipeline L0-L4)
├─ 🤍 Text: slate-100/200/300 (Hierarchy levels)
└─ ✨ Glass: backdrop-blur + rgba transparency
```

Este dashboard representa la **evolución natural de las herramientas de trading tradicionales**, combinando la potencia de Bloomberg Terminal con la accesibilidad de interfaces web modernas y la inteligencia de machine learning de última generación.

## Key Features

### L5 Serving Pipeline
- **RL Training**: PPO-LSTM, SAC, DDQN with Stable-Baselines3
- **7 Acceptance Gates**: Comprehensive quality checks
- **ONNX Export**: Optimized model for inference
- **Full Compliance**: Auditor requirements implemented

### Data Quality
- Observation clip rate ≤ 0.5%
- Zero rate < 50% per feature
- Robust median/MAD normalization
- 7-bar global lag for anti-leakage

## Infrastructure

### Docker Services
```yaml
services:
  - airflow-webserver
  - airflow-scheduler
  - airflow-worker
  - postgres
  - redis
  - minio
```

### MinIO Buckets
- `00-l0-ds-usdcop-acquire`
- `01-l1-ds-usdcop-standardize`
- `02-l2-ds-usdcop-prepare`
- `03-l3-ds-usdcop-feature`
- `04-l4-ds-usdcop-rlready`
- `05-l5-ds-usdcop-serving`

## Project Structure

```
USDCOP_Trading_RL/
├── airflow/
│   ├── dags/           # Pipeline DAGs (L0-L5)
│   └── configs/        # YAML configurations
├── src/
│   ├── core/           # Core components
│   ├── models/         # RL models
│   └── trading/        # Trading logic
├── dashboard/          # Web interface
├── docker-compose.yml  # Service orchestration
└── requirements.txt    # Python dependencies
```

## Model Training

L5 trains multiple RL models with 5 seeds each:
- **PPO-LSTM**: Recurrent policy network
- **SAC**: Soft Actor-Critic
- **DDQN**: Double Deep Q-Network

## Deployment

Production serving bundle includes:
- `policy.onnx` - Optimized inference model
- `model_manifest.json` - Metadata and lineage
- `serving_config.json` - Deployment configuration
- L4 contracts for observation processing

## Development

### Testing
```bash
python verify_l5_final.py
```

### Requirements
- Python 3.9+
- Docker 20.10+
- 16GB+ RAM
- 100GB+ storage

## Support

For issues or questions, please contact the development team.

---
**Version**: 2.0.0  
**Status**: Production Ready