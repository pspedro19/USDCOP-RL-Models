"""
Airflow Callbacks Module
========================
Callbacks centralizados para manejo de errores, alertas y logging en DAGs.

Funciones:
    - task_failure_callback: Envio de email + log en fallo de tarea
    - task_success_callback: Log de exito de tarea
    - dag_failure_callback: Resumen de errores del DAG
    - sla_miss_callback: Alerta cuando se incumple SLA
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACION
# =============================================================================

def get_alert_email() -> str:
    """Obtiene el email de alertas desde variable de entorno."""
    return os.getenv('ALERT_EMAIL', 'alerts@example.com')


def get_smtp_config() -> Dict[str, Any]:
    """Obtiene configuracion SMTP desde variables de entorno."""
    return {
        'host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
        'port': int(os.getenv('SMTP_PORT', '587')),
        'user': os.getenv('SMTP_USER', ''),
        'password': os.getenv('SMTP_PASSWORD', ''),
        'use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true',
        'from_email': os.getenv('SMTP_FROM_EMAIL', 'airflow@usdcop-pipeline.com'),
    }


# =============================================================================
# UTILIDADES DE EMAIL
# =============================================================================

def send_email(
    to_email: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None
) -> bool:
    """
    Envia un email de alerta.

    Args:
        to_email: Direccion de destino
        subject: Asunto del email
        body: Cuerpo del mensaje (texto plano)
        html_body: Cuerpo HTML opcional

    Returns:
        True si se envio correctamente, False en caso contrario
    """
    smtp_config = get_smtp_config()

    # Si no hay credenciales SMTP configuradas, solo loggear
    if not smtp_config['user'] or not smtp_config['password']:
        logger.warning(f"SMTP no configurado. Email no enviado: {subject}")
        logger.info(f"Destinatario: {to_email}")
        logger.info(f"Contenido: {body[:500]}...")
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = smtp_config['from_email']
        msg['To'] = to_email

        # Agregar parte de texto
        msg.attach(MIMEText(body, 'plain'))

        # Agregar parte HTML si existe
        if html_body:
            msg.attach(MIMEText(html_body, 'html'))

        # Conectar y enviar
        with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
            if smtp_config['use_tls']:
                server.starttls()
            server.login(smtp_config['user'], smtp_config['password'])
            server.sendmail(smtp_config['from_email'], to_email, msg.as_string())

        logger.info(f"Email enviado a {to_email}: {subject}")
        return True

    except Exception as e:
        logger.error(f"Error enviando email: {e}")
        return False


# =============================================================================
# CALLBACKS PRINCIPALES
# =============================================================================

def task_failure_callback(context: Dict[str, Any]) -> None:
    """
    Callback ejecutado cuando una tarea falla.
    Envia email de alerta y registra detalles en log.

    Args:
        context: Contexto de Airflow con informacion de la tarea
    """
    # Extraer informacion del contexto
    task_instance = context.get('task_instance')
    dag_id = context.get('dag').dag_id if context.get('dag') else 'unknown'
    task_id = task_instance.task_id if task_instance else 'unknown'
    execution_date = context.get('execution_date', datetime.now())
    exception = context.get('exception', 'No exception info')
    try_number = task_instance.try_number if task_instance else 0
    max_tries = task_instance.max_tries if task_instance else 0

    # Formatear mensaje de log
    log_message = f"""
{'='*60}
TASK FAILURE ALERT
{'='*60}
DAG ID:          {dag_id}
Task ID:         {task_id}
Execution Date:  {execution_date}
Try Number:      {try_number} / {max_tries + 1}
Exception:       {exception}
Timestamp:       {datetime.now().isoformat()}
{'='*60}
"""
    logger.error(log_message)

    # Preparar email
    subject = f"[AIRFLOW ALERT] Task Failed: {dag_id}.{task_id}"

    body = f"""
Airflow Task Failure Alert
===========================

DAG:             {dag_id}
Task:            {task_id}
Execution Date:  {execution_date}
Attempt:         {try_number} of {max_tries + 1}

Error Details:
{'-'*40}
{exception}
{'-'*40}

Timestamp: {datetime.now().isoformat()}

Please check the Airflow UI for more details.
"""

    html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <h2 style="color: #dc3545;">Airflow Task Failure Alert</h2>
    <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>DAG</strong></td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{dag_id}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Task</strong></td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{task_id}</td>
        </tr>
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Execution Date</strong></td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{execution_date}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Attempt</strong></td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{try_number} of {max_tries + 1}</td>
        </tr>
    </table>
    <h3 style="color: #dc3545;">Error Details:</h3>
    <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto;">
{exception}
    </pre>
    <p style="color: #6c757d; font-size: 12px;">Timestamp: {datetime.now().isoformat()}</p>
</body>
</html>
"""

    # Enviar email
    send_email(get_alert_email(), subject, body, html_body)


def task_success_callback(context: Dict[str, Any]) -> None:
    """
    Callback ejecutado cuando una tarea se completa exitosamente.
    Solo registra en log (sin email para evitar spam).

    Args:
        context: Contexto de Airflow con informacion de la tarea
    """
    task_instance = context.get('task_instance')
    dag_id = context.get('dag').dag_id if context.get('dag') else 'unknown'
    task_id = task_instance.task_id if task_instance else 'unknown'
    execution_date = context.get('execution_date', datetime.now())
    duration = task_instance.duration if task_instance else 0

    log_message = f"""
[SUCCESS] Task completed successfully
  DAG:       {dag_id}
  Task:      {task_id}
  Exec Date: {execution_date}
  Duration:  {duration:.2f}s
  Timestamp: {datetime.now().isoformat()}
"""
    logger.info(log_message)


def dag_failure_callback(context: Dict[str, Any]) -> None:
    """
    Callback ejecutado cuando el DAG completo falla.
    Genera resumen de todas las tareas fallidas y envia email.

    Args:
        context: Contexto de Airflow con informacion del DAG
    """
    dag = context.get('dag')
    dag_run = context.get('dag_run')
    execution_date = context.get('execution_date', datetime.now())

    dag_id = dag.dag_id if dag else 'unknown'
    run_id = dag_run.run_id if dag_run else 'unknown'

    # Recopilar tareas fallidas
    failed_tasks = []
    if dag_run:
        for ti in dag_run.get_task_instances():
            if ti.state == 'failed':
                failed_tasks.append({
                    'task_id': ti.task_id,
                    'start_date': str(ti.start_date) if ti.start_date else 'N/A',
                    'end_date': str(ti.end_date) if ti.end_date else 'N/A',
                    'try_number': ti.try_number,
                })

    # Log detallado
    log_message = f"""
{'='*60}
DAG FAILURE SUMMARY
{'='*60}
DAG ID:          {dag_id}
Run ID:          {run_id}
Execution Date:  {execution_date}
Failed Tasks:    {len(failed_tasks)}
{'='*60}
"""

    for task in failed_tasks:
        log_message += f"""
  Task: {task['task_id']}
    Start:    {task['start_date']}
    End:      {task['end_date']}
    Attempts: {task['try_number']}
"""

    log_message += f"{'='*60}\n"
    logger.error(log_message)

    # Preparar email con resumen
    subject = f"[AIRFLOW CRITICAL] DAG Failed: {dag_id}"

    failed_tasks_text = "\n".join([
        f"  - {t['task_id']} (attempts: {t['try_number']})"
        for t in failed_tasks
    ]) if failed_tasks else "  No failed task details available"

    body = f"""
Airflow DAG Failure Alert
==========================

DAG:             {dag_id}
Run ID:          {run_id}
Execution Date:  {execution_date}
Failed Tasks:    {len(failed_tasks)}

Failed Task Details:
{'-'*40}
{failed_tasks_text}
{'-'*40}

Timestamp: {datetime.now().isoformat()}

IMMEDIATE ACTION REQUIRED: Please check the Airflow UI for details.
"""

    failed_tasks_html = "".join([
        f"""<tr>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{t['task_id']}</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{t['start_date']}</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{t['try_number']}</td>
        </tr>"""
        for t in failed_tasks
    ]) if failed_tasks else "<tr><td colspan='3'>No details available</td></tr>"

    html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <h2 style="color: #dc3545;">Airflow DAG Failure Alert</h2>
    <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
        <strong>CRITICAL:</strong> The DAG <code>{dag_id}</code> has failed!
    </div>
    <table style="border-collapse: collapse; width: 100%; max-width: 600px; margin-bottom: 20px;">
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>DAG</strong></td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{dag_id}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Run ID</strong></td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{run_id}</td>
        </tr>
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Execution Date</strong></td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{execution_date}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Failed Tasks</strong></td>
            <td style="padding: 8px; border: 1px solid #dee2e6; color: #dc3545;"><strong>{len(failed_tasks)}</strong></td>
        </tr>
    </table>

    <h3>Failed Task Details:</h3>
    <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
        <tr style="background-color: #343a40; color: white;">
            <th style="padding: 8px; border: 1px solid #dee2e6;">Task ID</th>
            <th style="padding: 8px; border: 1px solid #dee2e6;">Start Time</th>
            <th style="padding: 8px; border: 1px solid #dee2e6;">Attempts</th>
        </tr>
        {failed_tasks_html}
    </table>

    <p style="color: #6c757d; font-size: 12px; margin-top: 20px;">Timestamp: {datetime.now().isoformat()}</p>
</body>
</html>
"""

    # Enviar email
    send_email(get_alert_email(), subject, body, html_body)


def sla_miss_callback(
    dag: Any,
    task_list: str,
    blocking_task_list: str,
    slas: List[Any],
    blocking_tis: List[Any]
) -> None:
    """
    Callback ejecutado cuando se incumple un SLA.
    Envia alerta inmediata.

    Args:
        dag: Objeto DAG
        task_list: Lista de tareas que incumplieron SLA
        blocking_task_list: Lista de tareas bloqueantes
        slas: Lista de objetos SLA
        blocking_tis: Lista de TaskInstances bloqueantes
    """
    dag_id = dag.dag_id if dag else 'unknown'

    log_message = f"""
{'='*60}
SLA MISS ALERT
{'='*60}
DAG ID:              {dag_id}
Tasks Missing SLA:   {task_list}
Blocking Tasks:      {blocking_task_list}
Timestamp:           {datetime.now().isoformat()}
{'='*60}
"""
    logger.warning(log_message)

    subject = f"[AIRFLOW SLA MISS] DAG: {dag_id}"

    body = f"""
Airflow SLA Miss Alert
=======================

DAG:              {dag_id}
Tasks Missing:    {task_list}
Blocking Tasks:   {blocking_task_list}

This indicates that tasks are taking longer than expected.
Please investigate potential performance issues.

Timestamp: {datetime.now().isoformat()}
"""

    html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <h2 style="color: #ffc107;">Airflow SLA Miss Alert</h2>
    <div style="background-color: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
        <strong>WARNING:</strong> SLA has been missed for DAG <code>{dag_id}</code>
    </div>
    <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>DAG</strong></td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{dag_id}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Tasks Missing SLA</strong></td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{task_list}</td>
        </tr>
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Blocking Tasks</strong></td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{blocking_task_list}</td>
        </tr>
    </table>
    <p style="margin-top: 15px;">This indicates that tasks are taking longer than expected. Please investigate potential performance issues.</p>
    <p style="color: #6c757d; font-size: 12px;">Timestamp: {datetime.now().isoformat()}</p>
</body>
</html>
"""

    # Enviar email
    send_email(get_alert_email(), subject, body, html_body)


# =============================================================================
# CALLBACKS ADICIONALES UTILES
# =============================================================================

def task_retry_callback(context: Dict[str, Any]) -> None:
    """
    Callback ejecutado cuando una tarea entra en reintento.

    Args:
        context: Contexto de Airflow
    """
    task_instance = context.get('task_instance')
    dag_id = context.get('dag').dag_id if context.get('dag') else 'unknown'
    task_id = task_instance.task_id if task_instance else 'unknown'
    try_number = task_instance.try_number if task_instance else 0
    max_tries = task_instance.max_tries if task_instance else 0
    exception = context.get('exception', 'No exception info')

    log_message = f"""
[RETRY] Task entering retry
  DAG:       {dag_id}
  Task:      {task_id}
  Attempt:   {try_number} of {max_tries + 1}
  Exception: {exception}
  Timestamp: {datetime.now().isoformat()}
"""
    logger.warning(log_message)


def data_quality_alert_callback(
    context: Dict[str, Any],
    issue_type: str,
    details: str
) -> None:
    """
    Callback para alertas de calidad de datos.

    Args:
        context: Contexto de Airflow
        issue_type: Tipo de problema (e.g., 'missing_data', 'stale_data', 'validation_failed')
        details: Detalles del problema
    """
    dag_id = context.get('dag').dag_id if context.get('dag') else 'unknown'
    execution_date = context.get('execution_date', datetime.now())

    log_message = f"""
{'='*60}
DATA QUALITY ALERT
{'='*60}
DAG ID:       {dag_id}
Issue Type:   {issue_type}
Exec Date:    {execution_date}
Details:      {details}
Timestamp:    {datetime.now().isoformat()}
{'='*60}
"""
    logger.error(log_message)

    subject = f"[AIRFLOW DATA QUALITY] {issue_type}: {dag_id}"

    body = f"""
Airflow Data Quality Alert
===========================

DAG:          {dag_id}
Issue Type:   {issue_type}
Exec Date:    {execution_date}

Details:
{'-'*40}
{details}
{'-'*40}

Timestamp: {datetime.now().isoformat()}
"""

    send_email(get_alert_email(), subject, body)


def stale_data_alert(context: Dict[str, Any], variable_name: str, last_update: str, threshold_days: int) -> None:
    """
    Alerta especifica para datos obsoletos (stale).

    Args:
        context: Contexto de Airflow
        variable_name: Nombre de la variable con datos obsoletos
        last_update: Fecha de ultima actualizacion
        threshold_days: Umbral de dias para considerar datos obsoletos
    """
    details = f"""
Variable:       {variable_name}
Last Update:    {last_update}
Threshold:      {threshold_days} days
Status:         DATA IS STALE - Requires attention

This variable has not been updated in more than {threshold_days} days.
Please verify the data source is functioning correctly.
"""
    data_quality_alert_callback(context, 'stale_data', details)
