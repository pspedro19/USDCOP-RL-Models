#!/bin/bash

# Setup Airflow Variables with REAL credentials
# THESE ARE REAL CREDENTIALS - USE WITH CARE

echo "Setting up REAL credentials in Airflow..."

# MT5 Credentials
docker exec airflow-webserver airflow variables set mt5_login "7114378"
docker exec airflow-webserver airflow variables set mt5_password "Satori1996**"
docker exec airflow-webserver airflow variables set mt5_server "FPMarketsSC-Demo"
docker exec airflow-webserver airflow variables set mt5_symbol "USDCOP"

# TwelveData API Key (Updated)
docker exec airflow-webserver airflow variables set twelvedata_api_key "9d7c480871f54f66bb933d96d5837d28"

echo "✅ Credentials configured successfully!"
echo ""
echo "Verifying configuration..."
docker exec airflow-webserver airflow variables list

echo ""
echo "⚠️  IMPORTANT: These are REAL credentials"
echo "   - MT5 Login: 7114378"
echo "   - MT5 Server: FPMarketsSC-Demo"
echo "   - TwelveData API Key: 085ba06...5af8ece"
echo ""
echo "The system will now use REAL data sources."
echo "Fallback data will only be used if connection fails."