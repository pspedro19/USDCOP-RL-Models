/**
 * /catalog — Catálogo de activos (CTR-FE-BE-001 §4.3). Página fina: monta la
 * vista CatalogView dentro del chrome TerminalShell (sección hub: el catálogo
 * es gestión de "Mis activos", no un módulo de análisis).
 */
import { TerminalShell } from '@/components/gm';
import CatalogView from '@/components/gm/views/CatalogView';

export default function CatalogPage() {
  return (
    <TerminalShell active="catalog">
      <CatalogView />
    </TerminalShell>
  );
}
