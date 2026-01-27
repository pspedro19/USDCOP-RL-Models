import requests

data = {
    "email": "admin@trading.usdcop.com",
    "password": "Admin123xA",
    "name": "admin"
}

try:
    r = requests.post("http://localhost:8085/api/auth/register", json=data)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
except Exception as e:
    print(f"Error: {e}")
