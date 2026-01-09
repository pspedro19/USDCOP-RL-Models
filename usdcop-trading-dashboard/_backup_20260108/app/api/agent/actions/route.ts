/**
 * API Route: /api/agent/actions
 *
 * Endpoints para obtener acciones del agente RL y métricas de performance
 *
 * GET /api/agent/actions?date=YYYY-MM-DD&limit=100
 *   - Obtiene acciones del agente para una fecha
 *   - Incluye métricas de performance de la sesión
 *
 * GET /api/agent/actions?action=latest
 *   - Obtiene la última acción del agente
 *
 * GET /api/agent/actions?action=today
 *   - Obtiene todas las acciones de hoy con métricas en tiempo real
 */

import { NextRequest, NextResponse } from 'next/server'
import { query as pgQuery, getPool } from '@/lib/db/postgres-client'
import { withAuth } from '@/lib/auth/api-auth'
import { createApiResponse } from '@/lib/types/api'

// Force dynamic rendering to avoid build-time DB connection
export const dynamic = 'force-dynamic'

// Use shared connection pool from postgres-client
const pool = getPool()

// Tipos
interface AgentAction {
  action_id: number
  timestamp_cot: string
  session_date: string
  bar_number: number
  action_type: string
  side: string | null
  price_at_action: number
  position_before: number
  position_after: number
  position_change: number
  pnl_action: number | null
  pnl_daily: number | null
  model_confidence: number
  marker_type: string
  marker_color: string
  reason_code: string | null
}

interface SessionPerformance {
  session_date: string
  total_bars: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number | null
  profit_factor: number | null
  daily_pnl: number | null
  daily_return_pct: number | null
  starting_equity: number | null
  ending_equity: number | null
  max_drawdown_intraday_pct: number | null
  intraday_sharpe: number | null
  total_long_bars: number
  total_short_bars: number
  total_flat_bars: number
  status: string
}

interface EquityCurvePoint {
  bar_number: number
  timestamp_cot: string
  equity_value: number
  return_daily_pct: number
  current_drawdown_pct: number
  current_position: number
  position_side: string | null
  market_price: number
}

// Empty response templates for when no data exists
const EMPTY_ACTIONS_RESPONSE = {
  actions: [],
  performance: null,
  alerts: [],
  date: new Date().toISOString().split('T')[0],
  totalActions: 0
}

const EMPTY_TODAY_RESPONSE = {
  actions: [],
  performance: null,
  equityCurve: [],
  latestInference: null,
  realtimeMetrics: {
    currentPosition: 0,
    currentSide: 'FLAT',
    totalTrades: 0,
    winRate: 0,
    dailyPnL: 0,
    dailyReturnPct: 0,
    currentDrawdown: 0,
    avgConfidence: 0
  },
  date: new Date().toISOString().split('T')[0],
  totalActions: 0,
  isLive: false
}

// GET handler - internal function
async function getHandler(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action')
  const date = searchParams.get('date') || new Date().toISOString().split('T')[0]
  const limit = parseInt(searchParams.get('limit') || '100')

  try {
    // Acción especial: última acción
    if (action === 'latest') {
      return await getLatestAction()
    }

    // Acción especial: datos de hoy con equity curve
    if (action === 'today') {
      return await getTodayData()
    }

    // Por defecto: acciones de una fecha específica
    return await getActionsForDate(date, limit)

  } catch (error) {
    console.error('Error in agent actions API:', error)
    // Return empty data instead of 500 error
    const emptyResponse = action === 'today'
      ? { ...EMPTY_TODAY_RESPONSE, date }
      : { ...EMPTY_ACTIONS_RESPONSE, date }
    return NextResponse.json(
      createApiResponse(emptyResponse, 'fallback'),
      { status: 200 }
    )
  }
}

// Obtener acciones para una fecha
async function getActionsForDate(date: string, limit: number) {
  let client;
  try {
    client = await pool.connect()
  } catch (error) {
    console.error('Database connection error:', error)
    return NextResponse.json(
      createApiResponse({ ...EMPTY_ACTIONS_RESPONSE, date }, 'fallback')
    )
  }

  try {
    // Obtener acciones del agente using view
    const actionsResult = await client.query<AgentAction>(`
      SELECT
        action_id,
        timestamp_cot,
        session_date,
        bar_number,
        action_type,
        side,
        price_at_action,
        position_before,
        position_after,
        position_change,
        pnl_action,
        pnl_daily,
        model_confidence,
        marker_type,
        marker_color,
        reason_code
      FROM dw.v_agent_actions
      WHERE session_date = $1
      ORDER BY bar_number DESC
      LIMIT $2
    `, [date, limit])

    // Obtener performance de la sesión
    const perfResult = await client.query<SessionPerformance>(`
      SELECT
        session_date,
        total_bars,
        total_trades,
        winning_trades,
        losing_trades,
        win_rate,
        profit_factor,
        daily_pnl,
        daily_return_pct,
        starting_equity,
        ending_equity,
        max_drawdown_intraday_pct,
        intraday_sharpe,
        total_long_bars,
        total_short_bars,
        total_flat_bars,
        status
      FROM dw.fact_session_performance
      WHERE session_date = $1
    `, [date])

    // Obtener alertas activas
    const alertsResult = await client.query(`
      SELECT
        alert_id,
        alert_type,
        severity,
        message,
        timestamp_utc
      FROM dw.fact_inference_alerts
      WHERE session_date = $1
        AND resolved = FALSE
      ORDER BY
        CASE severity
          WHEN 'CRITICAL' THEN 1
          WHEN 'ERROR' THEN 2
          WHEN 'WARNING' THEN 3
          ELSE 4
        END,
        timestamp_utc DESC
      LIMIT 10
    `, [date])

    return NextResponse.json(
      createApiResponse(
        {
          actions: actionsResult.rows,
          performance: perfResult.rows[0] || null,
          alerts: alertsResult.rows,
          date: date,
          totalActions: actionsResult.rowCount
        },
        'postgres'
      )
    )

  } finally {
    client.release()
  }
}

// Obtener última acción
async function getLatestAction() {
  let client;
  try {
    client = await pool.connect()
  } catch (error) {
    console.error('Database connection error:', error)
    return NextResponse.json(createApiResponse(null, 'fallback'))
  }

  try {
    const result = await client.query<AgentAction>(`
      SELECT
        action_id,
        timestamp_cot,
        session_date,
        bar_number,
        action_type,
        side,
        price_at_action,
        position_before,
        position_after,
        position_change,
        pnl_action,
        pnl_daily,
        model_confidence,
        marker_type,
        marker_color,
        reason_code
      FROM dw.v_agent_actions
      ORDER BY timestamp_utc DESC
      LIMIT 1
    `)

    return NextResponse.json(
      createApiResponse(result.rows[0] || null, 'postgres')
    )

  } catch (error) {
    console.error('Query error:', error)
    return NextResponse.json(createApiResponse(null, 'fallback'))
  } finally {
    if (client) client.release()
  }
}

// Obtener datos de hoy con equity curve
async function getTodayData() {
  const today = new Date().toISOString().split('T')[0]
  let client;

  try {
    client = await pool.connect()
  } catch (error) {
    console.error('Database connection error:', error)
    return NextResponse.json(
      createApiResponse({ ...EMPTY_TODAY_RESPONSE, date: today }, 'fallback')
    )
  }

  try {
    // Acciones de hoy
    const actionsResult = await client.query<AgentAction>(`
      SELECT
        action_id,
        timestamp_cot,
        session_date,
        bar_number,
        action_type,
        side,
        price_at_action,
        position_before,
        position_after,
        position_change,
        pnl_action,
        pnl_daily,
        model_confidence,
        marker_type,
        marker_color,
        reason_code
      FROM dw.v_agent_actions
      WHERE session_date = $1
      ORDER BY bar_number ASC
    `, [today])

    // Performance
    const perfResult = await client.query<SessionPerformance>(`
      SELECT *
      FROM dw.fact_session_performance
      WHERE session_date = $1
    `, [today])

    // Equity curve
    const equityResult = await client.query<EquityCurvePoint>(`
      SELECT
        bar_number,
        timestamp_cot,
        equity_value,
        return_daily_pct,
        current_drawdown_pct,
        current_position,
        position_side,
        market_price
      FROM dw.fact_equity_curve_realtime
      WHERE session_date = $1
      ORDER BY bar_number ASC
    `, [today])

    // Última inferencia
    const latestInferenceResult = await client.query(`
      SELECT
        inference_id,
        timestamp_cot,
        action_raw,
        action_discretized,
        confidence,
        close_price,
        position_after,
        portfolio_value_after,
        latency_ms
      FROM dw.fact_rl_inference
      WHERE DATE(timestamp_cot) = $1
      ORDER BY timestamp_utc DESC
      LIMIT 1
    `, [today])

    // Calcular métricas en tiempo real
    const realtimeMetrics = calculateRealtimeMetrics(
      actionsResult.rows,
      equityResult.rows
    )

    return NextResponse.json(
      createApiResponse(
        {
          actions: actionsResult.rows,
          performance: perfResult.rows[0] || null,
          equityCurve: equityResult.rows,
          latestInference: latestInferenceResult.rows[0] || null,
          realtimeMetrics,
          date: today,
          totalActions: actionsResult.rowCount,
          isLive: isMarketOpen()
        },
        'postgres'
      )
    )

  } finally {
    client.release()
  }
}

// Calcular métricas en tiempo real
function calculateRealtimeMetrics(
  actions: AgentAction[],
  equityCurve: EquityCurvePoint[]
) {
  if (actions.length === 0) {
    return {
      currentPosition: 0,
      currentSide: 'FLAT',
      totalTrades: 0,
      winRate: 0,
      dailyPnL: 0,
      dailyReturnPct: 0,
      currentDrawdown: 0,
      avgConfidence: 0
    }
  }

  const lastAction = actions[actions.length - 1]
  const lastEquity = equityCurve.length > 0 ? equityCurve[equityCurve.length - 1] : null

  // Contar trades (excluyendo HOLD)
  const trades = actions.filter(a => a.action_type !== 'HOLD')
  const winningTrades = trades.filter(a => (a.pnl_action || 0) > 0)

  // Calcular promedio de confianza
  const avgConfidence = actions.reduce((sum, a) => sum + (a.model_confidence || 0), 0) / actions.length

  return {
    currentPosition: lastAction.position_after,
    currentSide: lastAction.position_after > 0.1 ? 'LONG' :
                 lastAction.position_after < -0.1 ? 'SHORT' : 'FLAT',
    totalTrades: trades.length,
    winRate: trades.length > 0 ? winningTrades.length / trades.length : 0,
    dailyPnL: lastAction.pnl_daily || 0,
    dailyReturnPct: lastEquity?.return_daily_pct || 0,
    currentDrawdown: lastEquity?.current_drawdown_pct || 0,
    avgConfidence,
    lastBar: lastAction.bar_number,
    lastPrice: lastAction.price_at_action
  }
}

// Verificar si el mercado está abierto
function isMarketOpen(): boolean {
  const now = new Date()

  // Convertir a hora Colombia (UTC-5)
  const cotOffset = -5 * 60 // minutos
  const utcMinutes = now.getUTCHours() * 60 + now.getUTCMinutes()
  const cotMinutes = utcMinutes + cotOffset
  const cotHour = Math.floor(((cotMinutes % 1440) + 1440) % 1440 / 60)

  // Verificar día de semana
  const dayOfWeek = now.getUTCDay()
  if (dayOfWeek === 0 || dayOfWeek === 6) {
    return false // Fin de semana
  }

  // Horario: 8:00 - 12:55 COT
  return cotHour >= 8 && cotHour < 13
}

// POST handler para insertar acciones manuales (testing) - internal function
async function postHandler(request: NextRequest) {
  try {
    const body = await request.json()

    // Validar campos requeridos
    if (!body.action_type || body.position_after === undefined) {
      return NextResponse.json(
        createApiResponse(null, 'none', 'Missing required fields: action_type, position_after'),
        { status: 400 }
      )
    }

    // Validar que price esté presente (no usar valores por defecto)
    if (body.price === undefined || body.price === null) {
      return NextResponse.json(
        createApiResponse(null, 'none', 'Missing required field: price'),
        { status: 400 }
      )
    }

    const client = await pool.connect()

    try {
      const now = new Date()
      const cotTime = new Date(now.getTime() - 5 * 60 * 60 * 1000) // UTC-5
      const sessionDate = cotTime.toISOString().split('T')[0]
      const hour = cotTime.getHours()
      const minute = cotTime.getMinutes()
      const barNumber = Math.max(1, Math.min(60, (hour - 8) * 12 + Math.floor(minute / 5) + 1))

      const result = await client.query(`
        INSERT INTO dw.fact_agent_actions (
          timestamp_utc, timestamp_cot,
          session_date, bar_number,
          action_type, side,
          price_at_action,
          position_before, position_after,
          model_confidence, model_id
        ) VALUES (
          $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
        ) RETURNING action_id
      `, [
        now.toISOString(),
        cotTime.toISOString(),
        sessionDate,
        barNumber,
        body.action_type,
        body.side || null,
        body.price,
        body.position_before || 0,
        body.position_after,
        body.confidence || 0.5,
        body.model_id || 'manual_test'
      ])

      return NextResponse.json(
        createApiResponse(
          {
            action_id: result.rows[0].action_id,
            session_date: sessionDate,
            bar_number: barNumber
          },
          'postgres'
        )
      )

    } finally {
      client.release()
    }

  } catch (error) {
    console.error('Error inserting action:', error)
    return NextResponse.json(
      createApiResponse(null, 'none', 'Insert error', { message: String(error) }),
      { status: 500 }
    )
  }
}

// SECURITY: Protect endpoints with role-based authentication
// GET: Allow admin, trader, and viewer roles to read agent actions
export const GET = withAuth(getHandler, {
  requiredRole: ['admin', 'trader', 'viewer'],
});

// POST: Only admin and trader roles can insert actions (testing/manual actions)
export const POST = withAuth(postHandler, {
  requiredRole: ['admin', 'trader'],
});
