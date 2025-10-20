/**
 * PostgreSQL Client Utility for TimescaleDB
 * Provides connection pooling and query execution for market data
 */

import { Pool, PoolClient, QueryResult } from 'pg';

// Create a connection pool
let pool: Pool | null = null;

interface PostgresConfig {
  host: string;
  port: number;
  database: string;
  user: string;
  password: string;
}

function getConfig(): PostgresConfig {
  // Check for DATABASE_URL first (Docker environment)
  const databaseUrl = process.env.DATABASE_URL;
  if (databaseUrl) {
    try {
      const url = new URL(databaseUrl);
      return {
        host: url.hostname,
        port: parseInt(url.port || '5432'),
        database: url.pathname.slice(1), // Remove leading slash
        user: url.username,
        password: url.password,
      };
    } catch (error) {
      console.warn('[PostgreSQL] Failed to parse DATABASE_URL, using fallback config');
    }
  }

  // Fallback to individual environment variables
  return {
    host: process.env.POSTGRES_HOST || 'postgres', // Changed from localhost to postgres for Docker
    port: parseInt(process.env.POSTGRES_PORT || '5432'),
    database: process.env.POSTGRES_DB || 'usdcop_trading',
    user: process.env.POSTGRES_USER || 'admin',
    password: process.env.POSTGRES_PASSWORD || 'admin123',
  };
}

/**
 * Get or create connection pool
 */
export function getPool(): Pool {
  if (!pool) {
    const config = getConfig();
    pool = new Pool({
      host: config.host,
      port: config.port,
      database: config.database,
      user: config.user,
      password: config.password,
      max: 20, // Maximum pool size
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    pool.on('error', (err) => {
      console.error('[PostgreSQL] Unexpected error on idle client', err);
    });
  }

  return pool;
}

/**
 * Execute a query
 */
export async function query<T = any>(
  text: string,
  params?: any[]
): Promise<QueryResult<T>> {
  const pool = getPool();
  const start = Date.now();

  try {
    const result = await pool.query<T>(text, params);
    const duration = Date.now() - start;

    if (duration > 1000) {
      console.warn(`[PostgreSQL] Slow query (${duration}ms):`, text.substring(0, 100));
    }

    return result;
  } catch (error) {
    console.error('[PostgreSQL] Query error:', error);
    throw error;
  }
}

/**
 * Get a client from the pool (for transactions)
 */
export async function getClient(): Promise<PoolClient> {
  const pool = getPool();
  return pool.connect();
}

/**
 * Close the pool (for graceful shutdown)
 */
export async function closePool(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
  }
}

/**
 * Test database connection
 */
export async function testConnection(): Promise<boolean> {
  try {
    const result = await query('SELECT NOW()');
    console.log('[PostgreSQL] Connection test successful:', result.rows[0]);
    return true;
  } catch (error) {
    console.error('[PostgreSQL] Connection test failed:', error);
    return false;
  }
}

export default {
  query,
  getPool,
  getClient,
  closePool,
  testConnection,
};
