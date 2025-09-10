# Sistema de Toggle Inteligente y Dinámico para Sidebars

## Resumen

Este documento describe la implementación completa del sistema de toggle inteligente y dinámico para ambos sidebars del dashboard React/Next.js de USD/COP Trading. El sistema incluye manejo centralizado de estado, detección inteligente de viewport, persistencia en localStorage y cálculos automáticos de layout responsivo.

## Componentes Implementados

### 1. Hook Principal: `useSidebarState.ts`

**Ubicación:** `hooks/useSidebarState.ts`

**Características:**
- ✅ Estado centralizado para ambos sidebars
- ✅ Detección automática de viewport con debouncing
- ✅ Persistencia automática en localStorage
- ✅ Auto-hide inteligente basado en tamaño de pantalla
- ✅ Cálculos dinámicos de ancho y márgenes
- ✅ Sincronización perfecta entre componentes

**Ejemplo de uso:**
```typescript
import { useSidebarState } from '@/hooks/useSidebarState';

function Dashboard() {
  const sidebarState = useSidebarState();
  
  const {
    // Estado
    leftSidebarCollapsed,
    navigationSidebarOpen,
    isDesktop,
    isMobile,
    
    // Acciones
    toggleLeftSidebar,
    toggleNavigationSidebar,
    toggleBothSidebars,
    
    // Cálculos
    totalSidebarWidth,
    mainContentMarginLeft
  } = sidebarState;
  
  return <div className={mainContentMarginLeft}>...</div>;
}
```

### 2. Hook de Layout Responsivo: `useResponsiveLayout.ts`

**Ubicación:** `hooks/useResponsiveLayout.ts`

**Características:**
- ✅ Cálculos avanzados de layout responsivo
- ✅ Detección de breakpoints optimizada
- ✅ Ajuste automático de padding y márgenes
- ✅ Gestión de z-index para overlays
- ✅ Optimización para pantallas ultra-wide

**Ejemplo de uso:**
```typescript
import { useResponsiveLayout } from '@/hooks/useResponsiveLayout';

function MainContent({ sidebarState }) {
  const layout = useResponsiveLayout(sidebarState);
  
  return (
    <main 
      className={layout.containerClasses}
      style={{
        minWidth: layout.isFullWidth ? '100%' : `${layout.availableWidth}px`
      }}
    >
      {/* Contenido */}
    </main>
  );
}
```

### 3. Componente de Botones Toggle: `SidebarToggleButtons.tsx`

**Ubicación:** `components/ui/SidebarToggleButtons.tsx`

**Características:**
- ✅ Botones inteligentes con animaciones fluidas
- ✅ Tooltips informativos en modo colapsado
- ✅ Indicadores visuales de estado
- ✅ Múltiples variantes (floating, integrated, minimal)
- ✅ Soporte completo para mobile y desktop

**Variantes disponibles:**
```typescript
// Flotante con todas las características (recomendado para desktop)
<SidebarToggleButtons
  {...sidebarState}
  variant="floating"
  position="top-right"
  showLabels={true}
/>

// Integrado en header/toolbar
<SidebarToggleButtons
  {...sidebarState}
  variant="integrated"
  showLabels={false}
/>

// Minimalista para espacios reducidos
<SidebarToggleButtons
  {...sidebarState}
  variant="minimal"
/>
```

### 4. AnimatedSidebar Mejorado

**Ubicación:** `components/ui/AnimatedSidebar.tsx`

**Mejoras implementadas:**
- ✅ Control de visibilidad dinámico
- ✅ Ancho dinámico basado en estado
- ✅ Tooltips mejorados en modo colapsado
- ✅ Transiciones más fluidas
- ✅ Indicadores de estado mejorados

## Funcionalidades del Sistema

### 1. Auto-Hide Inteligente

El sistema detecta automáticamente el tamaño del viewport y ajusta los sidebars:

- **Mobile (< 1024px):** Ambos sidebars se colapsan y se convierten en overlays
- **Tablet (1024px - 1280px):** Sidebar izquierdo visible pero colapsado, navegación oculta
- **Desktop (≥ 1280px):** Ambos sidebars visibles según preferencias del usuario

### 2. Persistencia en localStorage

El sistema guarda automáticamente:
- Estado del sidebar izquierdo (colapsado/expandido)
- Estado del sidebar de navegación (abierto/cerrado) - solo en desktop
- Preferencia de auto-hide (habilitado/deshabilitado)

**Claves utilizadas:**
```typescript
const STORAGE_KEYS = {
  LEFT_SIDEBAR_COLLAPSED: 'usdcop-left-sidebar-collapsed',
  NAVIGATION_SIDEBAR_OPEN: 'usdcop-navigation-sidebar-open',
  AUTO_HIDE_ENABLED: 'usdcop-auto-hide-enabled',
};
```

### 3. Cálculos de Layout Dinámicos

El sistema calcula automáticamente:
- Márgenes responsivos para el contenido principal
- Ancho disponible para el dashboard
- Z-index apropiado para overlays
- Padding dinámico basado en viewport

### 4. Detección de Viewport con Debouncing

Para optimizar performance:
- Debouncing de 150ms en eventos de resize
- Detección de cambios de orientación
- Actualización eficiente de estado

## Integración con el Dashboard Principal

### En `page.tsx`:

```typescript
export default function TradingDashboard() {
  // Sistema centralizado de sidebar
  const sidebarState = useSidebarState();
  const responsiveLayout = useResponsiveLayout(sidebarState);
  
  return (
    <div className="min-h-screen">
      {/* Botones de toggle inteligentes */}
      <SidebarToggleButtons
        {...sidebarState}
        variant="floating"
        position="top-left"
        className="xl:hidden" // Solo en mobile/tablet
      />
      
      {/* Sidebar de controles */}
      <AnimatedSidebar
        {...controlProps}
        collapsed={sidebarState.leftSidebarCollapsed}
        onToggleCollapse={sidebarState.toggleLeftSidebar}
        isVisible={sidebarState.isLeftSidebarVisible}
        width={sidebarState.leftSidebarWidth}
      />
      
      {/* Contenido principal con layout inteligente */}
      <div className={responsiveLayout.containerClasses}>
        <main className={`${responsiveLayout.mainContentPaddingLeft} ${responsiveLayout.mainContentPaddingRight}`}>
          {/* Dashboard content */}
        </main>
      </div>
    </div>
  );
}
```

## Controles de Teclado

El sistema incluye atajos de teclado:

- **Escape:** Cerrar sidebar de navegación (mobile)
- **Ctrl/Cmd + B:** Toggle sidebar izquierdo (desktop)
- **Ctrl/Cmd + Shift + N:** Toggle sidebar navegación (mobile)

## Estados y Transiciones

### Estados del Sistema:
1. **Desktop Expandido:** Ambos sidebars visibles y expandidos
2. **Desktop Colapsado:** Sidebar izquierdo colapsado, navegación visible
3. **Tablet:** Solo sidebar izquierdo visible (colapsado)
4. **Mobile:** Ambos sidebars como overlays

### Transiciones:
- **Duración:** 300ms con easing suave
- **Animaciones:** Escala, opacidad, y transformaciones X/Y
- **Debouncing:** 150ms para eventos de resize

## Optimizaciones de Performance

1. **Memoization:** Todos los cálculos están memoizados
2. **Debouncing:** Eventos de resize optimizados
3. **Lazy Loading:** Estados se cargan solo cuando es necesario
4. **Refs para timeouts:** Limpieza automática de timers

## Ventajas del Sistema

### ✅ Completamente Funcional
- Estado sincronizado entre todos los componentes
- Persistencia automática de preferencias
- Responsive design inteligente

### ✅ Dinámico e Inteligente  
- Auto-hide basado en viewport
- Cálculos automáticos de layout
- Detección inteligente de dispositivo

### ✅ Performance Optimizado
- Debouncing en eventos costosos
- Memoization de cálculos complejos
- Renderizado condicional eficiente

### ✅ UX/UI Excellence
- Animaciones fluidas y profesionales
- Tooltips informativos
- Indicadores visuales claros
- Controles de teclado

### ✅ Mantenible y Extensible
- Código modular y bien documentado
- Hooks reutilizables
- Configuración centralizada
- Tipos TypeScript completos

## Casos de Uso

### 1. Usuario Desktop
- Ambos sidebars visibles por defecto
- Toggle rápido entre estados con botones flotantes
- Persistencia de preferencias entre sesiones

### 2. Usuario Tablet
- Sidebar izquierdo colapsado automáticamente
- Navegación accesible mediante overlay
- Layout optimizado para pantalla mediana

### 3. Usuario Mobile
- Ambos sidebars como overlays para maximizar espacio
- Controles accesibles en barra inferior
- Gestos táctiles optimizados

## Próximas Mejoras Sugeridas

1. **Gestos Táctiles:** Swipe para abrir/cerrar sidebars
2. **Temas:** Soporte para múltiples temas de colores
3. **Animaciones Avanzadas:** Efectos de parallax y physics
4. **Performance Metrics:** Monitoreo de tiempos de renderizado
5. **A11y:** Mejoras adicionales de accesibilidad

---

**Estado del Sistema:** ✅ **COMPLETAMENTE IMPLEMENTADO Y FUNCIONAL**

**Última Actualización:** Septiembre 9, 2025

**Desarrollado por:** Claude Code Assistant

**Compatibilidad:** React 18+, Next.js 15+, Tailwind CSS 3+, Framer Motion 11+