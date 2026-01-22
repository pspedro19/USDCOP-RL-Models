#!/usr/bin/env python3
"""
Script de scraping optimizado para Fedesarrollo - Adaptado para USDCOP Trading System
Soporta dos encuestas:
1. CCI (Índice de Confianza del Consumidor)
2. ICI (Índice de Confianza Industrial/Comercial - ICCO)

Características:
- Procesamiento del PDF solo en memoria (sin guardar archivo)
- Retorna formato estándar: DataFrame con columnas ['fecha', 'valor']
- Compatible con actualizador_completo_v2.py
- Extracción automática desde abstract o PDF histórico

Uso:
    from scraper_fedesarrollo import obtener_cci, obtener_ici

    df_cci = obtener_cci(n=3)  # Últimos 3 valores de CCI
    df_ici = obtener_ici(n=3)  # Últimos 3 valores de ICI
"""

import requests
from bs4 import BeautifulSoup
import re
import PyPDF2
import io
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional


class FedesarrolloScraper:
    """
    Scraper optimizado para índices de confianza de Fedesarrollo
    """

    def __init__(self, tipo: str = 'consumidor'):
        """
        Inicializa el scraper

        Args:
            tipo: 'consumidor' para CCI o 'empresarial' para ICI
        """
        self.tipo = tipo.lower()

        if self.tipo not in ['consumidor', 'empresarial']:
            raise ValueError("tipo debe ser 'consumidor' o 'empresarial'")

        self.base_url = "https://www.repository.fedesarrollo.org.co"

        # Configuración según el tipo de encuesta
        if self.tipo == 'consumidor':
            self.collection_url = f"{self.base_url}/handle/11445/36"
            self.indice_codigo = "CCI"
            self.indice_nombre = "ICC"  # Fedesarrollo usa ICC
            self.indice_completo = "Índice de Confianza del Consumidor"
            self.encuesta_nombre = "Encuesta de Opinión del Consumidor"
            self.patron_indice = r'ICC\).*?alcanzó.*?balance de\s+(-?\d+[.,]\d+)%'
        else:  # empresarial
            self.collection_url = f"{self.base_url}/handle/11445/37"
            self.indice_codigo = "ICI"
            self.indice_nombre = "ICCO"  # Fedesarrollo usa ICCO
            self.indice_completo = "Índice de Confianza Comercial"
            self.encuesta_nombre = "Encuesta de Opinión Empresarial"
            self.patron_indice = r'ICCO\).*?se ubicó en\s+(-?\d+[.,]\d+)%'

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Mapeo de meses español -> número
        self.meses_map = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
            'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
            'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }

        # Mapeo inverso para parsing
        self.meses_abbr = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
            'Ene': 1, 'Feb': 2, 'Mar': 3, 'Abr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Ago': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dic': 12
        }

    def get_latest_survey(self) -> Optional[str]:
        """
        Obtiene la URL de la encuesta más reciente
        """
        print(f"[INFO] Scraper Fedesarrollo - {self.indice_codigo}")
        print(f"[INFO] Buscando {self.encuesta_nombre}...")

        try:
            response = self.session.get(self.collection_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Buscar todos los enlaces a encuestas
            links = soup.find_all('a', href=re.compile(r'/handle/11445/\d+'))

            # Filtrar solo los que son de la encuesta correspondiente
            survey_links = []
            for link in links:
                text = link.get_text().strip()
                if self.encuesta_nombre in text and 'Resultados' in text:
                    href = link.get('href')
                    if not href.startswith('http'):
                        href = self.base_url + href

                    # Extraer fecha si es posible
                    date_match = re.search(r'(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+(\d{4})', text, re.IGNORECASE)
                    if date_match:
                        survey_links.append({
                            'url': href,
                            'text': text,
                            'month': date_match.group(1).lower(),
                            'year': int(date_match.group(2))
                        })

            # Ordenar por año y mes (más reciente primero)
            if survey_links:
                survey_links.sort(
                    key=lambda x: (x.get('year', 0), self.meses_map.get(x.get('month', ''), 0)),
                    reverse=True
                )

                latest = survey_links[0]
                print(f"[OK] Encuesta encontrada: {latest['month'].capitalize()} {latest['year']}")
                return latest['url']
            else:
                print("[WARNING] No se encontraron encuestas en la página principal")
                # URL de respaldo conocida
                fallback_urls = {
                    'consumidor': f"{self.base_url}/handle/11445/4844",  # Oct 2024
                    'empresarial': f"{self.base_url}/handle/11445/4839"   # Sep 2024
                }
                latest_url = fallback_urls[self.tipo]
                print(f"[INFO] Usando URL de respaldo")
                return latest_url

        except Exception as e:
            print(f"[ERROR] Error al buscar encuesta: {e}")
            # Fallback a URL conocida
            fallback_urls = {
                'consumidor': f"{self.base_url}/handle/11445/4844",
                'empresarial': f"{self.base_url}/handle/11445/4839"
            }
            return fallback_urls[self.tipo]

    def extract_indice_from_abstract(self, survey_url: str) -> Optional[Dict]:
        """
        Extrae el valor del índice directamente del abstract en la página
        """
        print(f"[INFO] Extrayendo {self.indice_codigo} del abstract...")

        try:
            response = self.session.get(survey_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Buscar el resumen/abstract
            abstract_sections = soup.find_all(['p', 'div'], text=re.compile(self.indice_completo))

            for section in abstract_sections:
                text = section.get_text()

                # Buscar patrón según el tipo de encuesta
                match = re.search(self.patron_indice, text, re.IGNORECASE)
                if match:
                    value_str = match.group(1).replace(',', '.')
                    value = float(value_str)

                    # Buscar el mes/periodo
                    month_match = re.search(r'En\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+(\d{4})', text, re.IGNORECASE)

                    if month_match:
                        month_name = month_match.group(1).lower()
                        year = int(month_match.group(2))
                        month_num = self.meses_map.get(month_name, 1)

                        # Crear fecha (último día del mes)
                        if month_num == 12:
                            fecha = datetime(year, month_num, 31)
                        elif month_num in [1, 3, 5, 7, 8, 10]:
                            fecha = datetime(year, month_num, 31)
                        elif month_num in [4, 6, 9, 11]:
                            fecha = datetime(year, month_num, 30)
                        else:  # febrero
                            fecha = datetime(year, month_num, 28)
                    else:
                        # Si no se encuentra fecha, usar fecha actual
                        fecha = datetime.now()

                    print(f"[OK] {self.indice_codigo} extraído: {value} ({fecha.strftime('%Y-%m-%d')})")
                    return {
                        'fecha': fecha,
                        'valor': value,
                        'fuente': 'abstract'
                    }

            print(f"[WARNING] No se pudo extraer {self.indice_codigo} del abstract")
            return None

        except Exception as e:
            print(f"[ERROR] Error al extraer del abstract: {e}")
            return None

    def get_pdf_historic_url(self, survey_url: str) -> Optional[str]:
        """
        Obtiene la URL del PDF histórico desde la página de la encuesta.

        IMPORTANTE para ICI (Empresarial):
        - Hay DOS PDFs históricos: Comercio (ICCO) e Industria (ICI)
        - Debemos seleccionar el de INDUSTRIA para obtener el ICI correcto
        """
        print(f"[INFO] Buscando PDF histórico para {self.indice_codigo}...")

        try:
            response = self.session.get(survey_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Buscar enlaces al PDF histórico
            all_links = soup.find_all('a', href=re.compile(r'\.pdf', re.IGNORECASE))

            # Para ICI (empresarial), debemos buscar específicamente el PDF de INDUSTRIA
            # Para CCI (consumidor), cualquier PDF histórico sirve
            pdf_candidates = []

            for link in all_links:
                href = link.get('href', '')
                text = link.get_text().strip()
                href_lower = href.lower()
                text_lower = text.lower()

                # Verificar si es un PDF histórico
                is_historic = ('histórico' in href_lower or 'historico' in href_lower or
                               'histórico' in text_lower or 'historico' in text_lower)

                if is_historic:
                    pdf_href = href
                    if not pdf_href.startswith('http'):
                        pdf_url = self.base_url + pdf_href
                    else:
                        pdf_url = pdf_href

                    # Clasificar el PDF
                    is_industria = 'industria' in href_lower or 'industria' in text_lower
                    is_comercio = 'comercio' in href_lower or 'comercio' in text_lower

                    pdf_candidates.append({
                        'url': pdf_url,
                        'text': text,
                        'is_industria': is_industria,
                        'is_comercio': is_comercio
                    })

            # Seleccionar el PDF correcto según el tipo de índice
            if pdf_candidates:
                if self.tipo == 'empresarial':
                    # Para ICI: buscar específicamente el de INDUSTRIA
                    for pdf in pdf_candidates:
                        if pdf['is_industria']:
                            print(f"[OK] PDF histórico INDUSTRIA encontrado: {pdf['text']}")
                            return pdf['url']

                    # Si no hay de industria, usar el primero (fallback)
                    print(f"[WARNING] No se encontró PDF de Industria, usando: {pdf_candidates[0]['text']}")
                    return pdf_candidates[0]['url']
                else:
                    # Para CCI: usar el primer histórico disponible
                    print(f"[OK] PDF histórico encontrado: {pdf_candidates[0]['text']}")
                    return pdf_candidates[0]['url']

            print("[WARNING] No se encontró PDF histórico")
            return None

        except Exception as e:
            print(f"[ERROR] Error al buscar PDF: {e}")
            return None

    def download_and_parse_pdf(self, pdf_url: str) -> Optional[List[Dict]]:
        """
        Descarga el PDF histórico y extrae los últimos valores del índice
        """
        print(f"[INFO] Descargando y procesando PDF...")

        try:
            response = self.session.get(pdf_url, timeout=60)
            response.raise_for_status()

            # Procesar el PDF directamente en memoria
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            print(f"[INFO] PDF tiene {len(pdf_reader.pages)} páginas")

            # Extraer texto de todas las páginas
            full_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                full_text += text + "\n"

            # Buscar los valores del índice en el texto
            indice_values = self.extract_indice_values_from_text(full_text)

            return indice_values

        except Exception as e:
            print(f"[ERROR] Error al procesar PDF: {e}")
            return None

    def extract_indice_values_from_text(self, text: str) -> Optional[List[Dict]]:
        """
        Extrae los valores del índice del texto del PDF histórico.

        ESTRUCTURA DE LOS PDFs:
        - CCI (ICC): mes-año ICC IEC ICE  (3 columnas, tomamos la 1ra)
          Ejemplo: "oct-25 13,6 18,1 6,9"

        - ICI (ICCO): mes-año ICCO SitEcon NivelExist Expect (4 columnas, tomamos la 1ra)
          Ejemplo: "sep-25 20,4 40,8 10,4 30,8"

        NOTA: Algunos meses tienen espacio extra: "may -25" en vez de "may-25"
        """
        print(f"[INFO] Extrayendo valores históricos de {self.indice_codigo}...")

        # Mapeo de abreviaciones de mes (español)
        month_map = {
            'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'ago': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12
        }

        all_values = []

        # Procesar línea por línea para mayor precisión
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Patrón mejorado: captura mes-año y TODOS los valores numéricos de la línea
            # Maneja espacios extra como "may -25"
            # Formato: mes-año valor1 valor2 valor3 [valor4]
            pattern = r'^([a-z]{3})\s*-\s*(\d{2})\s+(-?\d+[,\.]\d+)\s+(-?\d+[,\.]\d+)\s+(-?\d+[,\.]\d+)(?:\s+(-?\d+[,\.]\d+))?'

            match = re.match(pattern, line, re.IGNORECASE)

            if match:
                try:
                    month_abbr = match.group(1).lower()
                    year_2digit = int(match.group(2))

                    # El PRIMER valor numérico es siempre el índice principal (ICC o ICCO)
                    value_str = match.group(3).replace(',', '.')
                    value = float(value_str)

                    # Validar mes
                    if month_abbr not in month_map:
                        continue

                    month = month_map[month_abbr]

                    # Convertir año de 2 dígitos a 4 dígitos
                    # Lógica: 00-50 → 2000-2050, 51-99 → 1951-1999
                    if year_2digit <= 50:
                        year = 2000 + year_2digit
                    else:
                        year = 1900 + year_2digit

                    # Crear fecha (último día del mes)
                    if month == 12:
                        fecha = datetime(year, month, 31)
                    elif month in [1, 3, 5, 7, 8, 10]:
                        fecha = datetime(year, month, 31)
                    elif month in [4, 6, 9, 11]:
                        fecha = datetime(year, month, 30)
                    else:  # febrero
                        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                            fecha = datetime(year, month, 29)
                        else:
                            fecha = datetime(year, month, 28)

                    # Filtrar valores razonables para los índices (típicamente -60 a 60)
                    if -70 <= value <= 70:
                        all_values.append({
                            'fecha': fecha,
                            'valor': value
                        })

                except Exception as e:
                    continue

        # Eliminar duplicados basados en fecha (mantener el primero encontrado)
        seen_dates = {}
        for item in all_values:
            date_key = item['fecha'].strftime('%Y-%m')
            if date_key not in seen_dates:
                seen_dates[date_key] = item

        unique_values = list(seen_dates.values())

        # Ordenar por fecha (más reciente primero)
        unique_values.sort(key=lambda x: x['fecha'], reverse=True)

        if unique_values:
            print(f"[OK] Se encontraron {len(unique_values)} valores históricos")
            # Mostrar los 5 más recientes para verificación
            print(f"[DEBUG] Últimos 5 valores:")
            for i, v in enumerate(unique_values[:5]):
                print(f"        {v['fecha'].strftime('%Y-%m-%d')}: {v['valor']}")
            return unique_values
        else:
            print(f"[WARNING] No se pudieron extraer valores del PDF")
            return None

    def scrape(self, n: int = 3) -> Optional[pd.DataFrame]:
        """
        Ejecuta el proceso completo de scraping y retorna DataFrame estándar

        Args:
            n: Número de registros a retornar (por defecto 3)

        Returns:
            DataFrame con columnas ['fecha', 'valor'] o None si falla
        """
        print(f"\n[INFO] Iniciando scraper {self.indice_codigo} - Fedesarrollo")

        results = []

        try:
            # Obtener URL de la encuesta más reciente
            survey_url = self.get_latest_survey()

            if survey_url:
                # Método 1: Intentar extraer del abstract (más rápido)
                abstract_value = self.extract_indice_from_abstract(survey_url)
                if abstract_value:
                    results.append(abstract_value)

                # Método 2: Intentar PDF histórico para obtener serie completa
                pdf_url = self.get_pdf_historic_url(survey_url)

                if pdf_url:
                    indice_values = self.download_and_parse_pdf(pdf_url)
                    if indice_values:
                        # Si tenemos valores del PDF, usamos esos (son más completos)
                        results = indice_values

            # Convertir a DataFrame
            if results:
                df = pd.DataFrame(results)
                df = df[['fecha', 'valor']].head(n)

                print(f"\n[OK] {self.indice_codigo} extraído exitosamente:")
                print(f"     Registros: {len(df)}")
                print(f"     Más reciente: {df.iloc[0]['valor']:.2f} ({df.iloc[0]['fecha'].strftime('%Y-%m-%d')})")

                return df
            else:
                print(f"[ERROR] No se pudieron extraer valores de {self.indice_codigo}")
                return None

        except Exception as e:
            print(f"[ERROR] Error en scraper {self.indice_codigo}: {e}")
            return None


def obtener_cci(n: int = 3) -> Optional[pd.DataFrame]:
    """
    Obtiene los últimos n valores del Índice de Confianza del Consumidor (CCI)

    Args:
        n: Número de registros a retornar (por defecto 3)

    Returns:
        DataFrame con columnas ['fecha', 'valor'] o None si falla
    """
    scraper = FedesarrolloScraper(tipo='consumidor')
    return scraper.scrape(n=n)


def obtener_ici(n: int = 3) -> Optional[pd.DataFrame]:
    """
    Obtiene los últimos n valores del Índice de Confianza Industrial/Comercial (ICI/ICCO)

    Args:
        n: Número de registros a retornar (por defecto 3)

    Returns:
        DataFrame con columnas ['fecha', 'valor'] o None si falla
    """
    scraper = FedesarrolloScraper(tipo='empresarial')
    return scraper.scrape(n=n)


if __name__ == "__main__":
    """
    Test del scraper
    """
    print("=" * 70)
    print("TEST SCRAPER FEDESARROLLO")
    print("=" * 70)

    # Test CCI
    print("\n### TEST 1: CCI (Confianza del Consumidor) ###")
    df_cci = obtener_cci(n=3)
    if df_cci is not None:
        print("\nResultado CCI:")
        print(df_cci.to_string(index=False))

    print("\n" + "=" * 70)

    # Test ICI
    print("\n### TEST 2: ICI (Confianza Industrial) ###")
    df_ici = obtener_ici(n=3)
    if df_ici is not None:
        print("\nResultado ICI:")
        print(df_ici.to_string(index=False))

    print("\n" + "=" * 70)
    print("TEST COMPLETADO")
    print("=" * 70)
