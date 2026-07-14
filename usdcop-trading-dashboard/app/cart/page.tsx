/**
 * /cart — página completa del carrito (CTR-FE-BE-001 §4.3; prototipo Var B
 * l.306-362). Página fina: monta CartView dentro del chrome TerminalShell
 * (sección hub — el carrito es gestión de la cuenta, no un módulo de análisis).
 * RBAC: 'authenticated' en PAGE_ROUTES (rbac.contract.ts).
 */
import { TerminalShell } from '@/components/gm';
import CartView from '@/components/gm/views/CartView';

export default function CartPage() {
  return (
    <TerminalShell active="catalog">
      <CartView />
    </TerminalShell>
  );
}
