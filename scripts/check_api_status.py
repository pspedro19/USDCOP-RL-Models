#!/usr/bin/env python3
"""
Script para verificar el estado de todas las APIs de TwelveData
"""

import requests
import time
import logging
from datetime import datetime, timedelta
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Keys
API_KEYS = [
    # Group 1
    '7bb5ad48501b4a6ea9fbedba2b0247f9',
    '21de7ed341dc4ad5a20c292cee4652cb',
    '06f4622c4cfd4fd78795777367bb07d6',
    '0cabdba116f2415c82f7922c704427d9',
    '93104476650b4d25aaf033ce06585ba2',
    '9c61d83999c949d9866d92661f0f7c25',
    '3d465022bf0c489f84270d5f1cd771b9',
    '46d5e594b4af4282a935e1b1da8edc19',
    # Group 2
    'bf67096c644f4d20821540de2ad344dc',
    'a752322b9f3d44328a61cb53c24666b6',
    '69f2c0d75209451981be8d208204ea13',
    'b3faeff15a634855aab906858d2c8486',
    'e56b2aec322e4fe1b2b43bb419d5fde8',
    'fabe96c251694ab38a0c4a794244ae58',
    'c6d60eefb2b347f8a94f3cbc919fc33a',
    'e0cd278e177a469ba91992f1487c9c0e',
    # Group 3
    'f8ed98d381d545b5932f51be1252e3d6',
    '5f77fd41b46e40d492b6c7a8e75e9795',
    '531f28c28e814bdd9613e4440ad39467',
    'd1807504792643158df7e3b834c5535f'
]

def check_api_quota(api_key, key_index):
    """Check API quota and recent data availability"""
    try:
        # Test 1: Check quota
        quota_url = "https://api.twelvedata.com/api_usage"
        quota_params = {'apikey': api_key}

        logger.info(f"🔑 Testing API Key {key_index + 1}/{len(API_KEYS)}: {api_key[:8]}...")

        quota_response = requests.get(quota_url, params=quota_params, timeout=10)
        quota_info = {}

        if quota_response.status_code == 200:
            quota_data = quota_response.json()
            quota_info = {
                'quota_used': quota_data.get('current_usage', 'unknown'),
                'quota_limit': quota_data.get('plan_limit', 'unknown'),
                'plan': quota_data.get('plan_name', 'unknown')
            }
            logger.info(f"  📊 Quota: {quota_info['quota_used']}/{quota_info['quota_limit']} ({quota_info['plan']})")
        else:
            logger.warning(f"  ⚠️ Quota check failed: {quota_response.status_code}")

        # Test 2: Try recent data (last week)
        recent_date = datetime.now(pytz.timezone('America/Bogota')) - timedelta(days=7)
        end_date = recent_date + timedelta(days=1)

        data_url = "https://api.twelvedata.com/time_series"
        data_params = {
            'symbol': 'USD/COP',
            'interval': '5min',
            'apikey': api_key,
            'start_date': recent_date.strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': end_date.strftime('%Y-%m-%d %H:%M:%S'),
            'timezone': 'America/Bogota',
            'outputsize': 50,
            'format': 'JSON'
        }

        data_response = requests.get(data_url, params=data_params, timeout=15)
        data_info = {}

        if data_response.status_code == 200:
            data_json = data_response.json()

            if 'values' in data_json and data_json['values']:
                data_info['recent_data'] = True
                data_info['records_count'] = len(data_json['values'])
                data_info['latest_timestamp'] = data_json['values'][0].get('datetime', 'unknown')
                logger.info(f"  ✅ Recent data available: {data_info['records_count']} records, latest: {data_info['latest_timestamp']}")
            elif 'code' in data_json:
                data_info['recent_data'] = False
                data_info['error_code'] = data_json.get('code')
                data_info['error_message'] = data_json.get('message', 'unknown error')
                logger.warning(f"  ❌ API Error: {data_info['error_code']} - {data_info['error_message']}")
            else:
                data_info['recent_data'] = False
                data_info['error'] = 'No data returned'
                logger.warning(f"  ⚠️ No recent data available")
        else:
            data_info['recent_data'] = False
            data_info['http_error'] = data_response.status_code
            logger.warning(f"  ❌ HTTP Error: {data_response.status_code}")

        # Test 3: Try older data (2024)
        old_date = datetime(2024, 1, 15, 8, 0, tzinfo=pytz.timezone('America/Bogota'))
        old_end = old_date + timedelta(days=1)

        old_params = data_params.copy()
        old_params.update({
            'start_date': old_date.strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': old_end.strftime('%Y-%m-%d %H:%M:%S'),
            'outputsize': 20
        })

        old_response = requests.get(data_url, params=old_params, timeout=15)
        old_info = {}

        if old_response.status_code == 200:
            old_json = old_response.json()
            if 'values' in old_json and old_json['values']:
                old_info['old_data'] = True
                old_info['records_count'] = len(old_json['values'])
                logger.info(f"  📅 2024 data available: {old_info['records_count']} records")
            else:
                old_info['old_data'] = False
                logger.warning(f"  📅 No 2024 data available")

        result = {
            'api_key': api_key[:8] + '...',
            'index': key_index + 1,
            'quota_info': quota_info,
            'recent_data_info': data_info,
            'old_data_info': old_info,
            'working': data_info.get('recent_data', False) or old_info.get('old_data', False)
        }

        time.sleep(2)  # Rate limiting
        return result

    except Exception as e:
        logger.error(f"  ❌ Error testing key {key_index + 1}: {e}")
        return {
            'api_key': api_key[:8] + '...',
            'index': key_index + 1,
            'error': str(e),
            'working': False
        }

def main():
    logger.info("🔍 VERIFICANDO ESTADO DE TODAS LAS APIs DE TWELVEDATA")
    logger.info("=" * 60)

    results = []
    working_keys = []

    for i, api_key in enumerate(API_KEYS):
        result = check_api_quota(api_key, i)
        results.append(result)

        if result.get('working', False):
            working_keys.append(api_key)

        logger.info("-" * 40)

    # Summary
    logger.info("=" * 60)
    logger.info("📊 RESUMEN FINAL")
    logger.info(f"🔑 Total APIs probadas: {len(API_KEYS)}")
    logger.info(f"✅ APIs funcionando: {len(working_keys)}")
    logger.info(f"❌ APIs con problemas: {len(API_KEYS) - len(working_keys)}")

    if working_keys:
        logger.info("🚀 APIs disponibles para usar:")
        for i, key in enumerate(working_keys[:5]):  # Show first 5
            logger.info(f"  - {key[:8]}... (Key {API_KEYS.index(key) + 1})")
    else:
        logger.error("❌ NINGUNA API ESTÁ FUNCIONANDO!")

    logger.info("=" * 60)

    return working_keys

if __name__ == "__main__":
    working_apis = main()