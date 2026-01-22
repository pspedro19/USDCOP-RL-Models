"""
Script de pruebas para verificar que el API funciona correctamente.
Ejecutar con: python test_api.py
"""

import httpx
import asyncio
from pprint import pprint

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api"

# Variables globales para las pruebas
access_token = None
user_id = None
credential_id = None
signal_id = None


async def test_health():
    """Prueba 1: Health check"""
    print("\n" + "="*50)
    print("TEST 1: Health Check")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        pprint(response.json())
        return response.status_code == 200


async def test_register():
    """Prueba 2: Registro de usuario"""
    global access_token, user_id

    print("\n" + "="*50)
    print("TEST 2: Registro de Usuario")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/auth/register",
            json={
                "email": "test@signalbridge.com",
                "password": "TestPassword123",
                "name": "Test User"
            }
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        pprint(data)

        if response.status_code == 201:
            access_token = data.get("access_token")
            return True
        elif response.status_code == 409:
            print("Usuario ya existe, intentando login...")
            return await test_login()
        return False


async def test_login():
    """Prueba 3: Login de usuario"""
    global access_token

    print("\n" + "="*50)
    print("TEST 3: Login de Usuario")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/auth/login",
            json={
                "email": "test@signalbridge.com",
                "password": "TestPassword123"
            }
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        pprint(data)

        if response.status_code == 200:
            access_token = data.get("access_token")
            return True
        return False


async def test_get_profile():
    """Prueba 4: Obtener perfil del usuario"""
    global user_id

    print("\n" + "="*50)
    print("TEST 4: Obtener Perfil")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/users/me",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        pprint(data)

        if response.status_code == 200:
            user_id = data.get("id")
            return True
        return False


async def test_get_supported_exchanges():
    """Prueba 5: Obtener exchanges soportados"""
    print("\n" + "="*50)
    print("TEST 5: Exchanges Soportados")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/exchanges/supported",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        print(f"Status: {response.status_code}")
        pprint(response.json())
        return response.status_code == 200


async def test_create_exchange_credential(api_key: str, api_secret: str, exchange: str = "mexc"):
    """Prueba 6: Crear credenciales de exchange"""
    global credential_id

    print("\n" + "="*50)
    print(f"TEST 6: Crear Credenciales de {exchange.upper()}")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/exchanges/credentials",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "exchange": exchange,
                "label": f"Mi cuenta {exchange.upper()} Test",
                "api_key": api_key,
                "api_secret": api_secret,
                "is_testnet": False  # Cambiar a True para testnet
            }
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        pprint(data)

        if response.status_code == 201:
            credential_id = data.get("id")
            return True
        return False


async def test_validate_credential():
    """Prueba 7: Validar credenciales"""
    print("\n" + "="*50)
    print("TEST 7: Validar Credenciales")
    print("="*50)

    if not credential_id:
        print("No hay credential_id, saltando...")
        return False

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/exchanges/credentials/{credential_id}/validate",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        print(f"Status: {response.status_code}")
        pprint(response.json())
        return response.status_code == 200


async def test_get_balances():
    """Prueba 8: Obtener balances"""
    print("\n" + "="*50)
    print("TEST 8: Obtener Balances del Exchange")
    print("="*50)

    if not credential_id:
        print("No hay credential_id, saltando...")
        return False

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/exchanges/credentials/{credential_id}/balances",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        print(f"Status: {response.status_code}")
        pprint(response.json())
        return response.status_code == 200


async def test_get_trading_config():
    """Prueba 9: Obtener configuracion de trading"""
    print("\n" + "="*50)
    print("TEST 9: Configuracion de Trading")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/trading/config",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        print(f"Status: {response.status_code}")
        pprint(response.json())
        return response.status_code == 200


async def test_update_trading_config():
    """Prueba 10: Actualizar configuracion de trading"""
    print("\n" + "="*50)
    print("TEST 10: Actualizar Config de Trading")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.patch(
            f"{API_URL}/trading/config",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "default_exchange": "mexc",
                "max_position_size": 0.05,
                "stop_loss_percent": 3.0,
                "take_profit_percent": 5.0,
                "max_daily_trades": 20
            }
        )
        print(f"Status: {response.status_code}")
        pprint(response.json())
        return response.status_code == 200


async def test_create_signal():
    """Prueba 11: Crear una señal de trading"""
    global signal_id

    print("\n" + "="*50)
    print("TEST 11: Crear Señal de Trading")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/signals",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "symbol": "BTCUSDT",
                "action": 1,  # 1=BUY, 2=SELL, 3=CLOSE
                "price": 50000.0,
                "quantity": 0.001,
                "stop_loss": 49000.0,
                "take_profit": 52000.0,
                "source": "api_test",
                "metadata": {"test": True}
            }
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        pprint(data)

        if response.status_code == 201:
            signal_id = data.get("id")
            return True
        return False


async def test_get_signals():
    """Prueba 12: Obtener señales"""
    print("\n" + "="*50)
    print("TEST 12: Listar Señales")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/signals",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        print(f"Status: {response.status_code}")
        pprint(response.json())
        return response.status_code == 200


async def test_get_signal_stats():
    """Prueba 13: Obtener estadisticas de señales"""
    print("\n" + "="*50)
    print("TEST 13: Estadisticas de Señales")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/signals/stats?days=7",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        print(f"Status: {response.status_code}")
        pprint(response.json())
        return response.status_code == 200


async def test_trading_status():
    """Prueba 14: Estado del trading"""
    print("\n" + "="*50)
    print("TEST 14: Estado del Trading")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/trading/status",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        print(f"Status: {response.status_code}")
        pprint(response.json())
        return response.status_code == 200


async def test_today_stats():
    """Prueba 15: Estadisticas de hoy"""
    print("\n" + "="*50)
    print("TEST 15: Estadisticas de Hoy")
    print("="*50)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/executions/today",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        print(f"Status: {response.status_code}")
        pprint(response.json())
        return response.status_code == 200


async def run_basic_tests():
    """Ejecutar pruebas basicas (sin API keys)"""
    print("\n" + "#"*60)
    print("# PRUEBAS BASICAS (Sin API Keys de Exchange)")
    print("#"*60)

    results = []

    # Test basicos
    results.append(("Health Check", await test_health()))
    results.append(("Registro/Login", await test_register()))

    if access_token:
        results.append(("Perfil Usuario", await test_get_profile()))
        results.append(("Exchanges Soportados", await test_get_supported_exchanges()))
        results.append(("Trading Config", await test_get_trading_config()))
        results.append(("Update Trading Config", await test_update_trading_config()))
        results.append(("Crear Señal", await test_create_signal()))
        results.append(("Listar Señales", await test_get_signals()))
        results.append(("Stats Señales", await test_get_signal_stats()))
        results.append(("Estado Trading", await test_trading_status()))
        results.append(("Stats Hoy", await test_today_stats()))

    return results


async def run_exchange_tests(api_key: str, api_secret: str, exchange: str = "mexc"):
    """Ejecutar pruebas de exchange (requiere API keys)"""
    print("\n" + "#"*60)
    print(f"# PRUEBAS DE EXCHANGE: {exchange.upper()}")
    print("#"*60)

    results = []

    # Primero asegurar que tenemos token
    if not access_token:
        await test_register()

    if access_token and api_key and api_secret:
        results.append(("Crear Credencial", await test_create_exchange_credential(api_key, api_secret, exchange)))

        if credential_id:
            results.append(("Validar Credencial", await test_validate_credential()))
            results.append(("Obtener Balances", await test_get_balances()))

    return results


def print_results(results):
    """Imprimir resumen de resultados"""
    print("\n" + "="*60)
    print("RESUMEN DE PRUEBAS")
    print("="*60)

    passed = 0
    failed = 0

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
        if result:
            passed += 1
        else:
            failed += 1

    print("="*60)
    print(f"Total: {passed} pasaron, {failed} fallaron")
    print("="*60)


async def main():
    """Funcion principal"""
    print("\n" + "🚀"*30)
    print("  SIGNALBRIDGE API TEST SUITE")
    print("🚀"*30)

    # Pruebas basicas
    basic_results = await run_basic_tests()

    # Para pruebas con exchange, descomentar y agregar tus keys:
    # exchange_results = await run_exchange_tests(
    #     api_key="TU_API_KEY",
    #     api_secret="TU_API_SECRET",
    #     exchange="mexc"  # o "binance"
    # )
    # basic_results.extend(exchange_results)

    print_results(basic_results)


if __name__ == "__main__":
    asyncio.run(main())
