import { redirect } from 'next/navigation';
import { EXECUTION_ROUTES } from '@/lib/config/execution/constants';

/**
 * Execution Module Root Page
 * Redirects to dashboard
 */
export default function ExecutionPage() {
  redirect(EXECUTION_ROUTES.DASHBOARD);
}
