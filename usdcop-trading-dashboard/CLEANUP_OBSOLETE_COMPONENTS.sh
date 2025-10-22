#!/bin/bash

###############################################################################
# SCRIPT DE LIMPIEZA DE COMPONENTES OBSOLETOS
###############################################################################
#
# Este script identifica y OPCIONALMENTE elimina componentes obsoletos
# del directorio /components/views
#
# COMPONENTES A ELIMINAR: 15 archivos obsoletos (48% del total)
#
# USO:
#   ./CLEANUP_OBSOLETE_COMPONENTS.sh --dry-run  # Solo listar, NO eliminar
#   ./CLEANUP_OBSOLETE_COMPONENTS.sh --delete   # Eliminar archivos
#
###############################################################################

set -e

VIEWS_DIR="/home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Array de componentes OBSOLETOS a eliminar
OBSOLETE_COMPONENTS=(
    "TradingTerminalView.tsx"
    "EnhancedTradingTerminal.tsx"
    "ProfessionalTradingTerminalSimplified.tsx"
    "RealTimeChart.tsx"
    "BacktestResults.tsx"
    "RLModelHealth.tsx"
    "RiskManagement.tsx"
    "PortfolioExposureAnalysis.tsx"
    "DataPipelineQuality.tsx"
    "UltimateVisualDashboard.tsx"
    "AuditCompliance.tsx"
    "L3CorrelationMatrix.tsx"
    "ModelPerformance.tsx"
    "PipelineHealthMonitor.tsx"
    "PipelineMonitor.tsx"
)

# Componentes ACTIVOS que NO se deben tocar
ACTIVE_COMPONENTS=(
    "UnifiedTradingTerminal.tsx"
    "ProfessionalTradingTerminal.tsx"
    "LiveTradingTerminal.tsx"
    "ExecutiveOverview.tsx"
    "TradingSignals.tsx"
    "RealTimeRiskMonitor.tsx"
    "RiskAlertsCenter.tsx"
    "PipelineStatus.tsx"
    "L0RawDataDashboard.tsx"
    "L1FeatureStats.tsx"
    "L3Correlations.tsx"
    "L4RLReadyData.tsx"
    "L5ModelDashboard.tsx"
    "L6BacktestResults.tsx"
)

# Componentes a CONSIDERAR (no eliminar por ahora)
CONSIDER_COMPONENTS=(
    "APIUsagePanel.tsx"
    "EnhancedAPIUsageDashboard.tsx"
)

echo "=============================================================================="
echo "ANÁLISIS DE COMPONENTES OBSOLETOS"
echo "=============================================================================="
echo ""

# Verificar modo de ejecución
DRY_RUN=true
if [[ "$1" == "--delete" ]]; then
    DRY_RUN=false
    echo -e "${RED}MODO: ELIMINAR ARCHIVOS${NC}"
    echo -e "${YELLOW}Los archivos obsoletos SERÁN ELIMINADOS${NC}"
else
    echo -e "${GREEN}MODO: DRY RUN (solo listar)${NC}"
    echo -e "${YELLOW}Use --delete para eliminar archivos${NC}"
fi

echo ""
echo "=============================================================================="

# Contador
TOTAL_OBSOLETE=${#OBSOLETE_COMPONENTS[@]}
FOUND_COUNT=0
DELETED_COUNT=0
NOT_FOUND_COUNT=0

echo ""
echo "COMPONENTES OBSOLETOS A ELIMINAR (${TOTAL_OBSOLETE} total):"
echo "------------------------------------------------------------------------------"

for component in "${OBSOLETE_COMPONENTS[@]}"; do
    FILE_PATH="${VIEWS_DIR}/${component}"

    if [ -f "$FILE_PATH" ]; then
        FOUND_COUNT=$((FOUND_COUNT + 1))
        SIZE=$(du -h "$FILE_PATH" | cut -f1)

        echo -e "${RED}✗ OBSOLETO${NC}: ${component} (${SIZE})"

        if [ "$DRY_RUN" = false ]; then
            rm "$FILE_PATH"
            echo -e "   ${GREEN}→ ELIMINADO${NC}"
            DELETED_COUNT=$((DELETED_COUNT + 1))
        else
            echo -e "   ${YELLOW}→ Se eliminaría con --delete${NC}"
        fi
    else
        NOT_FOUND_COUNT=$((NOT_FOUND_COUNT + 1))
        echo -e "${GREEN}✓ Ya eliminado${NC}: ${component}"
    fi
done

echo ""
echo "=============================================================================="
echo "COMPONENTES ACTIVOS - NO TOCAR (${#ACTIVE_COMPONENTS[@]} total):"
echo "------------------------------------------------------------------------------"

ACTIVE_FOUND=0
for component in "${ACTIVE_COMPONENTS[@]}"; do
    FILE_PATH="${VIEWS_DIR}/${component}"

    if [ -f "$FILE_PATH" ]; then
        ACTIVE_FOUND=$((ACTIVE_FOUND + 1))
        SIZE=$(du -h "$FILE_PATH" | cut -f1)
        echo -e "${GREEN}✓ ACTIVO${NC}: ${component} (${SIZE})"
    else
        echo -e "${RED}✗ FALTA${NC}: ${component} - ${RED}PROBLEMA!${NC}"
    fi
done

echo ""
echo "=============================================================================="
echo "COMPONENTES A CONSIDERAR - Decidir si configurar (${#CONSIDER_COMPONENTS[@]} total):"
echo "------------------------------------------------------------------------------"

for component in "${CONSIDER_COMPONENTS[@]}"; do
    FILE_PATH="${VIEWS_DIR}/${component}"

    if [ -f "$FILE_PATH" ]; then
        SIZE=$(du -h "$FILE_PATH" | cut -f1)
        echo -e "${YELLOW}⚠ CONSIDERAR${NC}: ${component} (${SIZE})"
        echo "   → Evaluar si agregar a views.config.ts"
    else
        echo -e "${GREEN}✓ No existe${NC}: ${component}"
    fi
done

echo ""
echo "=============================================================================="
echo "RESUMEN DE LIMPIEZA"
echo "=============================================================================="
echo ""
echo "Componentes obsoletos encontrados:  ${FOUND_COUNT}/${TOTAL_OBSOLETE}"
echo "Componentes ya eliminados:          ${NOT_FOUND_COUNT}/${TOTAL_OBSOLETE}"
echo "Componentes activos verificados:    ${ACTIVE_FOUND}/${#ACTIVE_COMPONENTS[@]}"
echo ""

if [ "$DRY_RUN" = false ]; then
    echo -e "${GREEN}✓ Archivos eliminados: ${DELETED_COUNT}${NC}"
    echo ""
    echo "LIMPIEZA COMPLETADA"
else
    echo -e "${YELLOW}Modo DRY RUN - No se eliminaron archivos${NC}"
    echo ""
    echo "Para eliminar archivos, ejecute:"
    echo "  ./CLEANUP_OBSOLETE_COMPONENTS.sh --delete"
fi

echo ""
echo "=============================================================================="
echo "PRÓXIMOS PASOS"
echo "=============================================================================="
echo ""
echo "1. Reemplazar ViewRenderer.tsx con ViewRenderer.CLEAN.tsx"
echo "   cp ViewRenderer.CLEAN.tsx components/ViewRenderer.tsx"
echo ""
echo "2. Verificar que no haya imports rotos"
echo "   npm run build"
echo ""
echo "3. Considerar agregar componentes API a views.config.ts si son necesarios"
echo ""
echo "=============================================================================="
