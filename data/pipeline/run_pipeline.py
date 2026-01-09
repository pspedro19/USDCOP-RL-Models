#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
USD/COP DATA PIPELINE v4.0 - Script Principal de Ejecucion
================================================================================

Pipeline reorganizado con estructura clara y prefijos numericos:

    00_config/          -> Configuracion centralizada
    01_sources/         -> Datos crudos (fuentes externas)
    02_scrapers/        -> Actualizacion automatica de datos
    03_fusion/          -> Fusion de datos historicos
    04_cleaning/        -> Limpieza y normalizacion
    05_resampling/      -> Resampleo a 5min y diario
    06_rl_dataset_builder/ -> Generacion de datasets RL
    07_output/          -> Datasets finales para modelos

Uso:
    python run_pipeline.py              # Ejecuta todo el pipeline
    python run_pipeline.py --check      # Solo verifica estado
    python run_pipeline.py --step 3     # Ejecuta solo paso 3 (fusion)
    python run_pipeline.py --from 4     # Ejecuta desde paso 4 en adelante

Autor: Sistema Automatizado
Fecha: 2025-12-15
Version: 4.0
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# RUTAS DEL PIPELINE
# =============================================================================
PIPELINE_DIR = Path(__file__).parent  # pipeline/
PROJECT_ROOT = PIPELINE_DIR.parent.parent  # USDCOP-RL-Models/
DATA_DIR = PIPELINE_DIR.parent  # data/

# Estructura del pipeline
STEPS = {
    1: {
        'name': '01_sources',
        'description': 'Datos crudos de fuentes externas',
        'script': None,  # No tiene script, es manual
        'type': 'data'
    },
    2: {
        'name': '02_scrapers',
        'description': 'Actualizacion automatica de datos macro',
        'script': PIPELINE_DIR / '02_scrapers' / '01_orchestrator' / 'actualizador_hpc_v3.py',
        'type': 'scraper'
    },
    3: {
        'name': '03_fusion',
        'description': 'Fusion de datos historicos',
        'script': PIPELINE_DIR / '03_fusion' / 'run_fusion.py',
        'type': 'transform'
    },
    4: {
        'name': '04_cleaning',
        'description': 'Limpieza y normalizacion de datos',
        'script': PIPELINE_DIR / '04_cleaning' / 'run_clean.py',
        'type': 'transform'
    },
    5: {
        'name': '05_resampling',
        'description': 'Resampleo a 5min y consolidacion diaria',
        'script': PIPELINE_DIR / '05_resampling' / 'run_resample.py',
        'type': 'transform'
    },
    6: {
        'name': '06_rl_dataset_builder',
        'description': 'Generacion de 10 datasets RL',
        'scripts': [
            PIPELINE_DIR / '06_rl_dataset_builder' / '01_build_5min_datasets.py',
            PIPELINE_DIR / '06_rl_dataset_builder' / '02_build_daily_datasets.py',
            PIPELINE_DIR / '06_rl_dataset_builder' / '03_analyze_datasets.py',
        ],
        'type': 'generate'
    },
    7: {
        'name': '07_output',
        'description': 'Datasets finales listos para RL',
        'script': None,  # No tiene script, es output
        'type': 'output'
    }
}


# =============================================================================
# COLORES PARA CONSOLA
# =============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'


def print_header(text):
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}")


def print_step(step_num, text):
    print(f"\n{Colors.BLUE}[PASO {step_num}]{Colors.END} {Colors.BOLD}{text}{Colors.END}")


def print_success(text):
    print(f"  {Colors.GREEN}[OK]{Colors.END} {text}")


def print_warning(text):
    print(f"  {Colors.YELLOW}[!]{Colors.END} {text}")


def print_error(text):
    print(f"  {Colors.RED}[X]{Colors.END} {text}")


def print_info(text):
    print(f"  {Colors.CYAN}[i]{Colors.END} {text}")


# =============================================================================
# FUNCIONES DE EJECUCION
# =============================================================================
def run_script(script_path: Path, timeout: int = 600) -> Tuple[bool, str]:
    """Ejecuta un script Python y retorna (success, message)"""
    if not script_path.exists():
        return False, f"Script no encontrado: {script_path}"

    logger.info(f"Ejecutando: {script_path.name}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(script_path.parent)
        )

        if result.returncode == 0:
            return True, "Completado"
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            return False, error_msg

    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def execute_step(step_num: int) -> bool:
    """Ejecuta un paso del pipeline"""
    if step_num not in STEPS:
        print_error(f"Paso {step_num} no existe")
        return False

    step = STEPS[step_num]
    print_step(step_num, f"{step['name']} - {step['description']}")

    # Pasos sin script (data/output)
    if step['type'] in ['data', 'output']:
        folder = PIPELINE_DIR / step['name']
        if folder.exists():
            files = list(folder.rglob('*'))
            data_files = [f for f in files if f.is_file() and not f.name.startswith('.')]
            print_info(f"Carpeta: {step['name']}/ ({len(data_files)} archivos)")
            return True
        else:
            print_warning(f"Carpeta no existe: {step['name']}/")
            return True

    # Pasos con multiples scripts
    if 'scripts' in step:
        all_success = True
        for script in step['scripts']:
            success, msg = run_script(script)
            if success:
                print_success(f"{script.name}")
            else:
                print_error(f"{script.name}: {msg}")
                all_success = False
        return all_success

    # Pasos con un solo script
    if step.get('script'):
        success, msg = run_script(step['script'])
        if success:
            print_success(msg)
        else:
            print_error(msg)
        return success

    return True


def check_status():
    """Verifica el estado actual del pipeline"""
    print_header("ESTADO DEL PIPELINE USD/COP v4.0")

    for step_num, step in STEPS.items():
        folder = PIPELINE_DIR / step['name']

        if folder.exists():
            # Contar archivos
            all_files = list(folder.rglob('*'))
            data_files = [f for f in all_files if f.is_file() and not f.name.startswith('.')]
            csv_files = [f for f in data_files if f.suffix.lower() == '.csv']
            py_files = [f for f in data_files if f.suffix.lower() == '.py']

            # Calcular tamano
            total_size = sum(f.stat().st_size for f in data_files) / (1024*1024)

            status = f"{Colors.GREEN}OK{Colors.END}"
            print(f"  [{status}] {step_num:02d}. {step['name']:<25} | {len(csv_files):3} CSV | {len(py_files):2} PY | {total_size:7.1f} MB")
        else:
            status = f"{Colors.RED}--{Colors.END}"
            print(f"  [{status}] {step_num:02d}. {step['name']:<25} | No existe")

    # Mostrar datasets finales
    print(f"\n{Colors.CYAN}DATASETS FINALES:{Colors.END}")

    datasets_5min = PIPELINE_DIR / '07_output' / 'datasets_5min'
    datasets_daily = PIPELINE_DIR / '07_output' / 'datasets_daily'

    for ds_path, ds_name in [(datasets_5min, '5 minutos'), (datasets_daily, 'Diarios')]:
        if ds_path.exists():
            files = list(ds_path.glob('RL_DS*.csv'))
            total_size = sum(f.stat().st_size for f in files) / (1024*1024)
            print(f"  - {ds_name}: {len(files)} datasets, {total_size:.1f} MB")
        else:
            print(f"  - {ds_name}: No generados")

    print()


def print_pipeline_diagram():
    """Imprime diagrama visual del pipeline"""
    diagram = """
    FLUJO DEL PIPELINE USD/COP v4.0
    ================================

    [01_sources]          Datos crudos de fuentes externas
         |                (commodities, exchange_rates, etc.)
         v
    [02_scrapers]         Actualizacion automatica via web
         |                (investing.com, FRED, DANE, etc.)
         v
    [03_fusion]           Fusion de todos los datos historicos
         |                -> DATASET_MACRO_*.csv
         v
    [04_cleaning]         Limpieza y normalizacion
         |                -> MACRO_*_CLEAN.csv
         v
    [05_resampling]       Resampleo a 5min + filtro festivos
         |                -> MACRO_5MIN_CONSOLIDATED.csv
         v
    [06_rl_builder]       Generacion de 10 datasets RL
         |                (DS1-DS10 para diferentes estrategias)
         v
    [07_output]           DATASETS FINALES LISTOS PARA RL
                          datasets_5min/ (250MB)
                          datasets_daily/ (5MB)
    """
    print(diagram)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Pipeline de datos USD/COP para Reinforcement Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run_pipeline.py              # Ejecuta todo el pipeline
  python run_pipeline.py --check      # Verifica estado actual
  python run_pipeline.py --step 3     # Solo paso 3 (fusion)
  python run_pipeline.py --from 4     # Desde paso 4 en adelante
  python run_pipeline.py --diagram    # Muestra diagrama del pipeline
        """
    )

    parser.add_argument('--check', action='store_true',
                        help='Solo verificar estado actual')
    parser.add_argument('--step', type=int, choices=range(1, 8),
                        help='Ejecutar solo un paso especifico (1-7)')
    parser.add_argument('--from', type=int, dest='from_step', choices=range(1, 8),
                        help='Ejecutar desde un paso especifico')
    parser.add_argument('--diagram', action='store_true',
                        help='Mostrar diagrama del pipeline')

    args = parser.parse_args()

    # Mostrar diagrama
    if args.diagram:
        print_pipeline_diagram()
        return

    # Solo check
    if args.check:
        check_status()
        return

    start_time = datetime.now()

    # Header
    print_header("PIPELINE USD/COP v4.0 - Generacion de Datasets RL")
    print(f"Fecha: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directorio: {PIPELINE_DIR}")

    # Determinar pasos a ejecutar
    if args.step:
        steps_to_run = [args.step]
        print(f"Modo: Solo paso {args.step}")
    elif args.from_step:
        steps_to_run = list(range(args.from_step, 8))
        print(f"Modo: Desde paso {args.from_step}")
    else:
        # Ejecutar pasos 3-6 por defecto (omite sources y scrapers)
        steps_to_run = [3, 4, 5, 6]
        print("Modo: Regenerar datasets (pasos 3-6)")

    # Ejecutar pasos
    results = {}
    for step_num in steps_to_run:
        success = execute_step(step_num)
        results[step_num] = success

        if not success and step_num < max(steps_to_run):
            print_warning("Continuando con siguiente paso...")

    # Resumen
    elapsed = (datetime.now() - start_time).total_seconds() / 60

    print_header("RESUMEN DE EJECUCION")
    print(f"Tiempo total: {elapsed:.1f} minutos")

    success_count = sum(1 for s in results.values() if s)
    total_count = len(results)

    if success_count == total_count:
        print_success(f"Todos los pasos completados ({success_count}/{total_count})")
    else:
        print_warning(f"Completados: {success_count}/{total_count}")
        for step_num, success in results.items():
            status = "OK" if success else "FAIL"
            print(f"  - Paso {step_num}: {status}")

    # Mostrar estado final
    print()
    check_status()


if __name__ == "__main__":
    main()
