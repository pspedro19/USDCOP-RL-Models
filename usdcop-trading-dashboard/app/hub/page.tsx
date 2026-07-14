/**
 * /hub — GlobalMarkets Terminal (CTR-GM-UI-001). Página fina: monta la vista
 * HubView dentro del chrome TerminalShell. La versión previa vive en
 * /legacy/hub (admin:all) durante el strangler.
 */
import { TerminalShell } from '@/components/gm';
import HubView from '@/components/gm/views/HubView';

export default function HubPage() {
  return (
    <TerminalShell active="hub">
      <HubView />
    </TerminalShell>
  );
}
