"""
DAG: usdcop_m5__99_alert_monitor
================================================================================
Monitorea el sistema de inferencia y genera alertas cuando detecta problemas.

Ejecuta cada 5 minutos y verifica:
- Latencia de inferencia
- Drawdown excesivo
- Datos faltantes
- Errores en scrapers
- Conexión a APIs

Envía notificaciones via:
- Logging a fact_inference_alerts
- Webhook (Slack/Discord) opcional
- Email opcional
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

DAG_ID = 'usdcop_m5__99_alert_monitor'

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE ALERTAS
# ═══════════════════════════════════════════════════════════════════════════════════

ALERT_CONFIG = {
    # Umbrales de latencia (ms)
    'latency_warning': 500,
    'latency_error': 1000,
    'latency_critical': 5000,

    # Umbrales de drawdown (%)
    'drawdown_warning': 2.0,
    'drawdown_error': 5.0,
    'drawdown_critical': 10.0,

    # Umbrales de pérdida diaria (%)
    'daily_loss_warning': 1.0,
    'daily_loss_error': 3.0,
    'daily_loss_critical': 5.0,

    # Tiempo sin inferencia (minutos)
    'no_inference_warning': 10,
    'no_inference_error': 20,

    # Webhook para notificaciones (opcional)
    'webhook_url': os.environ.get('ALERT_WEBHOOK_URL'),

    # Email para notificaciones (opcional)
    'email_recipients': os.environ.get('ALERT_EMAIL_RECIPIENTS', '').split(','),
}

default_args = {
    'owner': 'monitoring',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}


def get_db_connection():
    """Get database connection"""
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'usdcop-postgres-timescale'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        database=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        user=os.environ.get('POSTGRES_USER', 'admin'),
        password=os.environ.get('POSTGRES_PASSWORD', 'admin123')
    )


def check_inference_latency(**context):
    """Check inference latency"""
    conn = get_db_connection()
    cur = conn.cursor()

    alerts = []

    try:
        # Get recent inference latencies
        cur.execute("""
            SELECT
                AVG(latency_ms) as avg_latency,
                MAX(latency_ms) as max_latency,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
                COUNT(*) as inference_count
            FROM dw.fact_rl_inference
            WHERE timestamp_utc > NOW() - INTERVAL '10 minutes'
        """)

        row = cur.fetchone()

        if row and row[3] > 0:  # Has inferences
            avg_latency = row[0]
            max_latency = row[1]
            p95_latency = row[2]

            if p95_latency > ALERT_CONFIG['latency_critical']:
                alerts.append({
                    'type': 'HIGH_LATENCY',
                    'severity': 'CRITICAL',
                    'message': f'Latencia P95 crítica: {p95_latency:.0f}ms',
                    'details': {'avg': avg_latency, 'max': max_latency, 'p95': p95_latency}
                })
            elif p95_latency > ALERT_CONFIG['latency_error']:
                alerts.append({
                    'type': 'HIGH_LATENCY',
                    'severity': 'ERROR',
                    'message': f'Latencia P95 alta: {p95_latency:.0f}ms',
                    'details': {'avg': avg_latency, 'max': max_latency, 'p95': p95_latency}
                })
            elif p95_latency > ALERT_CONFIG['latency_warning']:
                alerts.append({
                    'type': 'HIGH_LATENCY',
                    'severity': 'WARNING',
                    'message': f'Latencia P95 elevada: {p95_latency:.0f}ms',
                    'details': {'avg': avg_latency, 'max': max_latency, 'p95': p95_latency}
                })

    finally:
        cur.close()
        conn.close()

    context['ti'].xcom_push(key='latency_alerts', value=alerts)
    return {'alerts': len(alerts)}


def check_drawdown(**context):
    """Check drawdown levels"""
    conn = get_db_connection()
    cur = conn.cursor()

    alerts = []

    try:
        today = datetime.now().date()

        # Get current drawdown
        cur.execute("""
            SELECT
                current_drawdown_pct,
                equity_value,
                return_daily_pct
            FROM dw.fact_equity_curve_realtime
            WHERE session_date = %s
            ORDER BY bar_number DESC
            LIMIT 1
        """, (today,))

        row = cur.fetchone()

        if row:
            drawdown_pct = row[0] * 100 if row[0] else 0
            equity = row[1]
            daily_return = row[2] if row[2] else 0

            # Check drawdown
            if drawdown_pct > ALERT_CONFIG['drawdown_critical']:
                alerts.append({
                    'type': 'DRAWDOWN_WARNING',
                    'severity': 'CRITICAL',
                    'message': f'Drawdown crítico: {drawdown_pct:.2f}%',
                    'details': {'drawdown': drawdown_pct, 'equity': equity}
                })
            elif drawdown_pct > ALERT_CONFIG['drawdown_error']:
                alerts.append({
                    'type': 'DRAWDOWN_WARNING',
                    'severity': 'ERROR',
                    'message': f'Drawdown alto: {drawdown_pct:.2f}%',
                    'details': {'drawdown': drawdown_pct, 'equity': equity}
                })
            elif drawdown_pct > ALERT_CONFIG['drawdown_warning']:
                alerts.append({
                    'type': 'DRAWDOWN_WARNING',
                    'severity': 'WARNING',
                    'message': f'Drawdown elevado: {drawdown_pct:.2f}%',
                    'details': {'drawdown': drawdown_pct, 'equity': equity}
                })

            # Check daily loss
            if daily_return < -ALERT_CONFIG['daily_loss_critical']:
                alerts.append({
                    'type': 'DRAWDOWN_WARNING',
                    'severity': 'CRITICAL',
                    'message': f'Pérdida diaria crítica: {daily_return:.2f}%',
                    'details': {'daily_return': daily_return, 'equity': equity}
                })
            elif daily_return < -ALERT_CONFIG['daily_loss_error']:
                alerts.append({
                    'type': 'DRAWDOWN_WARNING',
                    'severity': 'ERROR',
                    'message': f'Pérdida diaria alta: {daily_return:.2f}%',
                    'details': {'daily_return': daily_return, 'equity': equity}
                })

    finally:
        cur.close()
        conn.close()

    context['ti'].xcom_push(key='drawdown_alerts', value=alerts)
    return {'alerts': len(alerts)}


def check_data_freshness(**context):
    """Check if we're receiving fresh data"""
    conn = get_db_connection()
    cur = conn.cursor()

    alerts = []

    try:
        # Check last inference time
        cur.execute("""
            SELECT
                timestamp_utc,
                EXTRACT(EPOCH FROM (NOW() - timestamp_utc)) / 60 as minutes_ago
            FROM dw.fact_rl_inference
            ORDER BY timestamp_utc DESC
            LIMIT 1
        """)

        row = cur.fetchone()

        if row:
            minutes_ago = row[1]

            # Only alert during market hours
            now = datetime.utcnow()
            cot_hour = (now.hour - 5) % 24

            if 8 <= cot_hour < 13:  # Market hours
                if minutes_ago > ALERT_CONFIG['no_inference_error']:
                    alerts.append({
                        'type': 'DATA_MISSING',
                        'severity': 'ERROR',
                        'message': f'Sin inferencia por {minutes_ago:.0f} minutos',
                        'details': {'minutes_ago': minutes_ago}
                    })
                elif minutes_ago > ALERT_CONFIG['no_inference_warning']:
                    alerts.append({
                        'type': 'DATA_MISSING',
                        'severity': 'WARNING',
                        'message': f'Inferencia retrasada: {minutes_ago:.0f} minutos',
                        'details': {'minutes_ago': minutes_ago}
                    })
        else:
            # No inferences at all today
            now = datetime.utcnow()
            cot_hour = (now.hour - 5) % 24

            if 8 <= cot_hour < 13:
                alerts.append({
                    'type': 'DATA_MISSING',
                    'severity': 'ERROR',
                    'message': 'No hay inferencias registradas hoy',
                    'details': {}
                })

        # Check last macro data
        cur.execute("""
            SELECT
                timestamp_utc,
                EXTRACT(EPOCH FROM (NOW() - timestamp_utc)) / 3600 as hours_ago
            FROM dw.fact_macro_realtime
            ORDER BY timestamp_utc DESC
            LIMIT 1
        """)

        row = cur.fetchone()

        if row:
            hours_ago = row[1]

            if hours_ago > 12:
                alerts.append({
                    'type': 'DATA_MISSING',
                    'severity': 'WARNING',
                    'message': f'Datos macro desactualizados: {hours_ago:.1f} horas',
                    'details': {'hours_ago': hours_ago}
                })

    finally:
        cur.close()
        conn.close()

    context['ti'].xcom_push(key='freshness_alerts', value=alerts)
    return {'alerts': len(alerts)}


def check_model_anomalies(**context):
    """Check for anomalies in model behavior"""
    conn = get_db_connection()
    cur = conn.cursor()

    alerts = []

    try:
        today = datetime.now().date()

        # Check for too many position changes (overtrade)
        cur.execute("""
            SELECT COUNT(*) as trades
            FROM dw.fact_agent_actions
            WHERE session_date = %s
              AND action_type != 'HOLD'
        """, (today,))

        row = cur.fetchone()
        trades_today = row[0] if row else 0

        if trades_today > 30:  # More than 30 trades in a day is unusual
            alerts.append({
                'type': 'ANOMALY_DETECTED',
                'severity': 'WARNING',
                'message': f'Posible overtrading: {trades_today} trades hoy',
                'details': {'trades': trades_today}
            })

        # Check for confidence anomalies
        cur.execute("""
            SELECT
                AVG(confidence) as avg_confidence,
                STDDEV(confidence) as std_confidence
            FROM dw.fact_rl_inference
            WHERE timestamp_utc > NOW() - INTERVAL '1 hour'
        """)

        row = cur.fetchone()
        if row and row[0]:
            avg_conf = row[0]
            std_conf = row[1] if row[1] else 0

            if avg_conf < 0.2:
                alerts.append({
                    'type': 'ANOMALY_DETECTED',
                    'severity': 'WARNING',
                    'message': f'Confianza del modelo muy baja: {avg_conf:.2f}',
                    'details': {'avg_confidence': avg_conf, 'std': std_conf}
                })

    finally:
        cur.close()
        conn.close()

    context['ti'].xcom_push(key='anomaly_alerts', value=alerts)
    return {'alerts': len(alerts)}


def store_alerts(**context):
    """Store all alerts in database"""
    # Collect all alerts
    latency_alerts = context['ti'].xcom_pull(key='latency_alerts') or []
    drawdown_alerts = context['ti'].xcom_pull(key='drawdown_alerts') or []
    freshness_alerts = context['ti'].xcom_pull(key='freshness_alerts') or []
    anomaly_alerts = context['ti'].xcom_pull(key='anomaly_alerts') or []

    all_alerts = latency_alerts + drawdown_alerts + freshness_alerts + anomaly_alerts

    if not all_alerts:
        print("No alerts to store")
        return {'stored': 0}

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        today = datetime.now().date()
        stored = 0

        for alert in all_alerts:
            # Check if similar alert exists recently (deduplicate)
            cur.execute("""
                SELECT alert_id FROM dw.fact_inference_alerts
                WHERE alert_type = %s
                  AND severity = %s
                  AND timestamp_utc > NOW() - INTERVAL '30 minutes'
                  AND resolved = FALSE
                LIMIT 1
            """, (alert['type'], alert['severity']))

            if cur.fetchone():
                print(f"Skipping duplicate alert: {alert['type']}")
                continue

            # Insert new alert
            cur.execute("""
                INSERT INTO dw.fact_inference_alerts (
                    alert_type, severity, message,
                    session_date, details
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING alert_id
            """, (
                alert['type'],
                alert['severity'],
                alert['message'],
                today,
                json.dumps(alert.get('details', {}))
            ))

            alert_id = cur.fetchone()[0]
            stored += 1
            print(f"Alert stored: [{alert['severity']}] {alert['message']} (ID={alert_id})")

        conn.commit()
        return {'stored': stored}

    finally:
        cur.close()
        conn.close()


def send_notifications(**context):
    """Send notifications for critical/error alerts"""
    import requests

    latency_alerts = context['ti'].xcom_pull(key='latency_alerts') or []
    drawdown_alerts = context['ti'].xcom_pull(key='drawdown_alerts') or []
    freshness_alerts = context['ti'].xcom_pull(key='freshness_alerts') or []
    anomaly_alerts = context['ti'].xcom_pull(key='anomaly_alerts') or []

    all_alerts = latency_alerts + drawdown_alerts + freshness_alerts + anomaly_alerts

    # Filter critical and error alerts
    critical_alerts = [a for a in all_alerts if a['severity'] in ['CRITICAL', 'ERROR']]

    if not critical_alerts:
        print("No critical alerts to notify")
        return {'notified': 0}

    webhook_url = ALERT_CONFIG.get('webhook_url')

    if webhook_url:
        try:
            # Format message for webhook
            message = "**USD/COP Trading Alerts**\n\n"
            for alert in critical_alerts:
                emoji = "🚨" if alert['severity'] == 'CRITICAL' else "❌"
                message += f"{emoji} **[{alert['severity']}]** {alert['message']}\n"

            # Send to webhook (Slack/Discord format)
            payload = {
                'content': message,
                'username': 'Trading Alert Bot'
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"Webhook notification sent: {len(critical_alerts)} alerts")
            else:
                print(f"Webhook failed: {response.status_code}")

        except Exception as e:
            print(f"Webhook error: {e}")

    return {'notified': len(critical_alerts)}


def generate_report(**context):
    """Generate summary report"""
    latency_alerts = context['ti'].xcom_pull(key='latency_alerts') or []
    drawdown_alerts = context['ti'].xcom_pull(key='drawdown_alerts') or []
    freshness_alerts = context['ti'].xcom_pull(key='freshness_alerts') or []
    anomaly_alerts = context['ti'].xcom_pull(key='anomaly_alerts') or []

    all_alerts = latency_alerts + drawdown_alerts + freshness_alerts + anomaly_alerts

    by_severity = {}
    for alert in all_alerts:
        sev = alert['severity']
        by_severity[sev] = by_severity.get(sev, 0) + 1

    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'total_alerts': len(all_alerts),
        'by_severity': by_severity,
        'alerts': all_alerts
    }

    print("\n" + "=" * 60)
    print("ALERT MONITOR REPORT")
    print("=" * 60)
    print(f"Total alerts: {len(all_alerts)}")
    print(f"By severity: {by_severity}")

    if all_alerts:
        print("\nAlerts:")
        for alert in all_alerts:
            print(f"  [{alert['severity']:8s}] {alert['type']:20s} - {alert['message']}")

    print("=" * 60)

    return report


# ═══════════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Monitoreo de alertas del sistema de inferencia',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['monitoring', 'alerts', 'production']
) as dag:

    start = EmptyOperator(task_id='start')

    # Check tasks in parallel
    check_latency = PythonOperator(
        task_id='check_inference_latency',
        python_callable=check_inference_latency
    )

    check_dd = PythonOperator(
        task_id='check_drawdown',
        python_callable=check_drawdown
    )

    check_fresh = PythonOperator(
        task_id='check_data_freshness',
        python_callable=check_data_freshness
    )

    check_anomaly = PythonOperator(
        task_id='check_model_anomalies',
        python_callable=check_model_anomalies
    )

    # Store and notify
    store = PythonOperator(
        task_id='store_alerts',
        python_callable=store_alerts
    )

    notify = PythonOperator(
        task_id='send_notifications',
        python_callable=send_notifications
    )

    report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report
    )

    end = EmptyOperator(task_id='end')

    # Flow
    start >> [check_latency, check_dd, check_fresh, check_anomaly]
    [check_latency, check_dd, check_fresh, check_anomaly] >> store >> notify >> report >> end
