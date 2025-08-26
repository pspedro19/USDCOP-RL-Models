#!/usr/bin/env python3
"""
📦 Install Dependencies for USDCOP Data Extraction
==================================================

Script para instalar todas las dependencias necesarias para el extractor
de datos históricos de USD/COP usando Twelve Data API.

Autor: Sistema USDCOP Trading
Fecha: Agosto 2025
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """
    Verifica que la versión de Python sea compatible
    
    Returns:
        bool: True si la versión es compatible
    """
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} no es compatible")
        print("✅ Se requiere Python 3.8 o superior")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} es compatible")
    return True

def install_package(package_name: str, upgrade: bool = False) -> bool:
    """
    Instala un paquete usando pip
    
    Args:
        package_name (str): Nombre del paquete a instalar
        upgrade (bool): Si actualizar el paquete si ya existe
        
    Returns:
        bool: True si la instalación fue exitosa
    """
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        
        if upgrade:
            cmd.append("--upgrade")
        
        cmd.append(package_name)
        
        print(f"📦 Instalando {package_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.returncode == 0:
            print(f"✅ {package_name} instalado exitosamente")
            return True
        else:
            print(f"❌ Error instalando {package_name}")
            print(f"📝 Error: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en la instalación de {package_name}")
        print(f"📝 Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado instalando {package_name}: {str(e)}")
        return False

def install_requirements_file():
    """
    Instala dependencias desde requirements.txt si existe
    
    Returns:
        bool: True si la instalación fue exitosa
    """
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("⚠️  Archivo requirements.txt no encontrado")
        return False
    
    try:
        print("📦 Instalando dependencias desde requirements.txt...")
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.returncode == 0:
            print("✅ Dependencias de requirements.txt instaladas exitosamente")
            return True
        else:
            print("❌ Error instalando dependencias de requirements.txt")
            print(f"📝 Error: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print("❌ Error en la instalación de requirements.txt")
        print(f"📝 Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        return False

def create_directories():
    """
    Crea los directorios necesarios para el proyecto
    
    Returns:
        bool: True si todos los directorios se crearon exitosamente
    """
    directories = [
        "data/raw/twelve_data",
        "data/processed",
        "logs",
        "config",
        "scripts"
    ]
    
    print("📁 Creando directorios del proyecto...")
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Directorio creado: {directory}")
        except Exception as e:
            print(f"❌ Error creando directorio {directory}: {str(e)}")
            return False
    
    return True

def test_imports():
    """
    Prueba que todas las dependencias se puedan importar correctamente
    
    Returns:
        bool: True si todos los imports son exitosos
    """
    print("🧪 Probando imports de dependencias...")
    
    required_packages = [
        "requests",
        "pandas",
        "numpy",
        "pyyaml",
        "logging"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} importado correctamente")
        except ImportError as e:
            print(f"❌ Error importando {package}: {str(e)}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"❌ Fallaron {len(failed_imports)} imports: {', '.join(failed_imports)}")
        return False
    
    print("✅ Todos los imports fueron exitosos")
    return True

def main():
    """
    Función principal para instalar todas las dependencias
    """
    print("🚀 Instalador de Dependencias - USDCOP Data Extraction")
    print("=" * 60)
    
    # Verificar versión de Python
    if not check_python_version():
        print("❌ Versión de Python incompatible. Abortando instalación.")
        return
    
    print("-" * 40)
    
    # Crear directorios
    if not create_directories():
        print("❌ Error creando directorios. Abortando instalación.")
        return
    
    print("-" * 40)
    
    # Instalar dependencias básicas
    basic_packages = [
        "requests>=2.28.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0"
    ]
    
    print("📦 Instalando dependencias básicas...")
    
    failed_installations = []
    
    for package in basic_packages:
        if not install_package(package):
            failed_installations.append(package)
    
    if failed_installations:
        print(f"❌ Fallaron {len(failed_installations)} instalaciones: {', '.join(failed_installations)}")
        print("🔧 Intenta instalar manualmente los paquetes fallidos")
        return
    
    print("-" * 40)
    
    # Intentar instalar desde requirements.txt
    install_requirements_file()
    
    print("-" * 40)
    
    # Probar imports
    if not test_imports():
        print("❌ Algunos imports fallaron. Verifica la instalación.")
        return
    
    print("-" * 40)
    
    # Resumen final
    print("🎉 INSTALACIÓN COMPLETADA EXITOSAMENTE!")
    print("=" * 40)
    print("✅ Python compatible")
    print("✅ Directorios creados")
    print("✅ Dependencias instaladas")
    print("✅ Imports funcionando")
    print("\n🚀 Próximos pasos:")
    print("1. Ejecuta: python scripts/test_twelve_data_api.py")
    print("2. Si las pruebas pasan, ejecuta: python scripts/extract_usdcop_historical.py")
    print("\n📚 Documentación disponible en README.md")

if __name__ == "__main__":
    main()
