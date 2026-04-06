/**
 * Shared Database Pool — Single source of truth for PostgreSQL connection.
 *
 * ALWAYS uses DATABASE_URL if available (Docker Compose sets this).
 * Falls back to individual POSTGRES_* env vars.
 *
 * NEVER use hardcoded passwords in API routes. Import this pool instead:
 *   import { pool } from '@/lib/db';
 */

import { Pool } from 'pg';

export const pool = process.env.DATABASE_URL
  ? new Pool({
      connectionString: process.env.DATABASE_URL,
      max: 10,
      idleTimeoutMillis: 30000,
    })
  : new Pool({
      host: process.env.POSTGRES_HOST || 'usdcop-postgres-timescale',
      port: parseInt(process.env.POSTGRES_PORT || '5432'),
      database: process.env.POSTGRES_DB || 'usdcop_trading',
      user: process.env.POSTGRES_USER || 'admin',
      password: process.env.POSTGRES_PASSWORD || 'admin123',
      max: 10,
      idleTimeoutMillis: 30000,
    });
