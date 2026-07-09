/**
 * THE approval-queue predicate (CTR-ADMIN-CONSOLE-001, spec C1).
 *
 * The queue table, the queue counter and the Overview "Pendientes" KPI must all
 * derive from this ONE definition — the original bug was a counter and a table
 * reading different sources and disagreeing ("cola dice 0 con 4 pending visibles").
 */
export const PENDING_QUEUE_WHERE = `status = 'pending'`;

export const PENDING_QUEUE_SELECT = `
  SELECT id, email, name, status, role, created_at,
         COALESCE(is_test, FALSE) AS is_test,
         EXTRACT(EPOCH FROM (NOW() - created_at)) / 3600.0 AS waiting_hours
  FROM sb_users
  WHERE ${PENDING_QUEUE_WHERE}
  ORDER BY created_at ASC`;
