#!/usr/bin/env node

/**
 * Chart Connectivity Test Script
 * =============================
 *
 * Tests the complete data flow from backend to frontend charts
 */

const http = require('http');
const fs = require('fs');

console.log('🚀 Testing Chart Component Connectivity...\n');

// Test configuration
const TRADING_API_URL = 'http://localhost:8000';
const FRONTEND_URL = 'http://localhost:3000';

async function makeRequest(url, description) {
  return new Promise((resolve) => {
    const startTime = Date.now();

    http.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        const duration = Date.now() - startTime;
        console.log(`✅ ${description}: ${res.statusCode} (${duration}ms)`);

        if (res.statusCode === 200) {
          try {
            const jsonData = JSON.parse(data);
            if (jsonData.data && Array.isArray(jsonData.data)) {
              console.log(`   📊 Data points: ${jsonData.data.length}`);
            }
            if (jsonData.total_records) {
              console.log(`   📈 Total records: ${jsonData.total_records}`);
            }
            if (jsonData.price) {
              console.log(`   💰 Current price: $${jsonData.price}`);
            }
          } catch (e) {
            console.log(`   📄 Response: ${data.substring(0, 100)}...`);
          }
        } else {
          console.log(`   ❌ Error: ${data.substring(0, 200)}`);
        }

        resolve({ status: res.statusCode, duration, data });
      });
    }).on('error', (err) => {
      console.log(`❌ ${description}: Connection failed - ${err.message}`);
      resolve({ status: 'error', error: err.message });
    });
  });
}

async function runTests() {
  console.log('1. Testing Trading API Backend...');
  await makeRequest(`${TRADING_API_URL}/api/health`, 'API Health Check');
  await makeRequest(`${TRADING_API_URL}/api/candlesticks/USDCOP?timeframe=5m&limit=10`, 'Candlestick Data');
  await makeRequest(`${TRADING_API_URL}/api/latest/USDCOP`, 'Latest Price');

  console.log('\n2. Testing Frontend Proxy Endpoints...');
  await makeRequest(`${FRONTEND_URL}/api/proxy/trading/health`, 'Frontend → Trading API Health');
  await makeRequest(`${FRONTEND_URL}/api/proxy/trading/candlesticks/USDCOP?timeframe=5m&limit=10`, 'Frontend → Candlestick Data');
  await makeRequest(`${FRONTEND_URL}/api/proxy/ws`, 'Frontend → WebSocket Proxy');

  console.log('\n3. Testing Chart Components Status...');

  // Check if main chart components exist
  const chartComponents = [
    'components/charts/RealDataTradingChart.tsx',
    'components/views/UnifiedTradingTerminal.tsx',
    'components/views/UltimateVisualDashboard.tsx',
    'lib/services/market-data-service.ts'
  ];

  chartComponents.forEach(component => {
    if (fs.existsSync(component)) {
      console.log(`✅ Chart Component: ${component}`);
    } else {
      console.log(`❌ Missing: ${component}`);
    }
  });

  console.log('\n📊 Chart Connectivity Test Complete!');
  console.log('\nRecommendations:');
  console.log('1. Verify Next.js server is accessible on port 3000');
  console.log('2. Check that both backend (8000) and frontend (3000) are running');
  console.log('3. Ensure proxy routes are correctly configured');
  console.log('4. Test real-time data flow in browser DevTools');
}

runTests().catch(console.error);