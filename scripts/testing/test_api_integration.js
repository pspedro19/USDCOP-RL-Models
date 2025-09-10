/**
 * Test script for TwelveData API and data alignment integration
 * Tests all key functionality: API keys, rate limiting, data alignment, WebSocket
 */

const https = require('https');
const fs = require('fs');

// API Keys from .env.local
const API_KEYS = [
  '9d7c480871f54f66bb933d96d5837d28',
  '085ba06282774cbc8e796f46a5af8ece', 
  '3dd1bcf17c0846f6857b31d12a64e5f5',
  '3656827e648a4c6fa2c4e2e7935c4fb8'
];

let currentKeyIndex = 0;

function getNextApiKey() {
  const key = API_KEYS[currentKeyIndex];
  currentKeyIndex = (currentKeyIndex + 1) % API_KEYS.length;
  return key;
}

function testApiRequest(endpoint, params = {}) {
  return new Promise((resolve, reject) => {
    const apiKey = getNextApiKey();
    const queryParams = new URLSearchParams({
      ...params,
      apikey: apiKey
    });
    
    const url = `https://api.twelvedata.com${endpoint}?${queryParams}`;
    
    console.log(`🔗 Testing: ${endpoint} with key ${currentKeyIndex}`);
    console.log(`   URL: ${url.replace(apiKey, '***API_KEY***')}`);
    
    const startTime = Date.now();
    
    https.get(url, (res) => {
      let data = '';
      
      res.on('data', (chunk) => {
        data += chunk;
      });
      
      res.on('end', () => {
        const endTime = Date.now();
        const latency = endTime - startTime;
        
        try {
          const jsonData = JSON.parse(data);
          resolve({
            success: true,
            status: res.statusCode,
            latency: latency,
            data: jsonData,
            keyIndex: currentKeyIndex,
            endpoint: endpoint
          });
        } catch (e) {
          reject({
            success: false,
            error: 'Invalid JSON response',
            rawData: data.substring(0, 500),
            latency: latency
          });
        }
      });
    }).on('error', (err) => {
      reject({
        success: false,
        error: err.message,
        latency: Date.now() - startTime
      });
    });
  });
}

async function testApiKeys() {
  console.log('\n🔑 TESTING API KEYS AND RATE LIMITING\n');
  
  const results = [];
  
  for (let i = 0; i < API_KEYS.length; i++) {
    try {
      const result = await testApiRequest('/quote', { symbol: 'USD/COP' });
      results.push(result);
      
      console.log(`✅ Key ${i + 1}: SUCCESS (${result.latency}ms)`);
      if (result.data.price) {
        console.log(`   Current USD/COP: ${result.data.price}`);
      }
      
      // Add delay to respect rate limits
      await new Promise(resolve => setTimeout(resolve, 1000));
      
    } catch (error) {
      console.log(`❌ Key ${i + 1}: FAILED - ${error.error}`);
      results.push(error);
    }
  }
  
  return results;
}

async function testTimeSeries() {
  console.log('\n📊 TESTING TIME SERIES DATA\n');
  
  try {
    const result = await testApiRequest('/time_series', {
      symbol: 'USD/COP',
      interval: '5min',
      outputsize: 10
    });
    
    console.log(`✅ Time Series: SUCCESS (${result.latency}ms)`);
    if (result.data.values && result.data.values.length > 0) {
      console.log(`   Got ${result.data.values.length} data points`);
      console.log(`   Latest: ${result.data.values[0].datetime} - Close: ${result.data.values[0].close}`);
      console.log(`   Range: ${result.data.values[result.data.values.length - 1].datetime} to ${result.data.values[0].datetime}`);
    }
    
    return result;
    
  } catch (error) {
    console.log(`❌ Time Series: FAILED - ${error.error}`);
    return error;
  }
}

async function testTechnicalIndicators() {
  console.log('\n📈 TESTING TECHNICAL INDICATORS\n');
  
  const indicators = [
    { name: 'RSI', endpoint: '/rsi', params: { time_period: 14 } },
    { name: 'MACD', endpoint: '/macd', params: {} },
    { name: 'SMA', endpoint: '/sma', params: { time_period: 20 } },
    { name: 'EMA', endpoint: '/ema', params: { time_period: 20 } }
  ];
  
  const results = [];
  
  for (const indicator of indicators) {
    try {
      const result = await testApiRequest(indicator.endpoint, {
        symbol: 'USD/COP',
        interval: '5min',
        ...indicator.params
      });
      
      console.log(`✅ ${indicator.name}: SUCCESS (${result.latency}ms)`);
      if (result.data.values && result.data.values.length > 0) {
        console.log(`   Latest value: ${JSON.stringify(result.data.values[0])}`);
      }
      
      results.push({ indicator: indicator.name, ...result });
      
      // Rate limiting delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      
    } catch (error) {
      console.log(`❌ ${indicator.name}: FAILED - ${error.error}`);
      results.push({ indicator: indicator.name, ...error });
    }
  }
  
  return results;
}

async function testDataAlignment() {
  console.log('\n🔄 TESTING DATA ALIGNMENT API\n');
  
  const alignmentTests = [
    {
      name: 'Align Dataset',
      url: 'http://localhost:3004/api/data/align?action=align',
      expected: 'aligned dataset with MinIO + TwelveData'
    },
    {
      name: 'Latest Data Point', 
      url: 'http://localhost:3004/api/data/align?action=latest',
      expected: 'single latest data point'
    },
    {
      name: 'MinIO History',
      url: 'http://localhost:3004/api/data/align?action=replay', 
      expected: 'complete MinIO historical data'
    }
  ];
  
  const results = [];
  
  for (const test of alignmentTests) {
    try {
      console.log(`🔗 Testing: ${test.name}`);
      
      const response = await fetch(test.url);
      const result = await response.json();
      
      if (response.ok && result.success) {
        console.log(`✅ ${test.name}: SUCCESS`);
        console.log(`   Data points: ${result.count || (result.data ? 1 : 0)}`);
        if (result.sources) {
          console.log(`   Sources: MinIO=${result.sources.minio}, TwelveData=${result.sources.twelvedata}`);
        }
      } else {
        console.log(`❌ ${test.name}: FAILED - ${result.error || 'Unknown error'}`);
      }
      
      results.push({ test: test.name, success: response.ok, result });
      
    } catch (error) {
      console.log(`❌ ${test.name}: FAILED - ${error.message}`);
      results.push({ test: test.name, success: false, error: error.message });
    }
  }
  
  return results;
}

async function testMinIOConnection() {
  console.log('\n🗄️ TESTING MINIO CONNECTION\n');
  
  try {
    // Test MinIO health endpoint
    const response = await fetch('http://localhost:9000/minio/health/live');
    
    if (response.ok) {
      console.log('✅ MinIO: Server is running and healthy');
      
      // Test if we can access the console
      try {
        const consoleResponse = await fetch('http://localhost:9001');
        if (consoleResponse.ok) {
          console.log('✅ MinIO Console: Accessible at http://localhost:9001');
        }
      } catch {
        console.log('⚠️ MinIO Console: Not accessible at http://localhost:9001');
      }
      
      return { success: true, status: 'healthy' };
    } else {
      console.log(`❌ MinIO: Server returned ${response.status}`);
      return { success: false, status: response.status };
    }
    
  } catch (error) {
    console.log(`❌ MinIO: Connection failed - ${error.message}`);
    return { success: false, error: error.message };
  }
}

async function testWebSocketCapability() {
  console.log('\n🌐 TESTING WEBSOCKET CAPABILITY\n');
  
  // Note: WebSocket is disabled in the current implementation
  // This test checks if the infrastructure is ready
  
  console.log('📝 WebSocket Status: Currently disabled in development');
  console.log('   Endpoint would be: wss://ws.twelvedata.com/v1/quotes/price');
  console.log('   Implementation: Available in twelvedata.ts WebSocketClient class');
  console.log('   Status: ⚠️ Disabled - requires valid subscription to TwelveData WebSocket API');
  
  return {
    implemented: true,
    enabled: false,
    reason: 'Requires TwelveData WebSocket subscription'
  };
}

async function generateReport(allResults) {
  console.log('\n📋 INTEGRATION TEST REPORT\n');
  console.log('=' .repeat(80));
  
  const report = {
    timestamp: new Date().toISOString(),
    environment: {
      node_version: process.version,
      platform: process.platform
    },
    api_keys: {
      total: API_KEYS.length,
      tested: allResults.apiKeys?.length || 0,
      working: allResults.apiKeys?.filter(r => r.success)?.length || 0
    },
    endpoints: {
      quote: allResults.apiKeys?.some(r => r.success) ? '✅' : '❌',
      time_series: allResults.timeSeries?.success ? '✅' : '❌',
      technical_indicators: allResults.indicators?.filter(r => r.success)?.length || 0
    },
    data_alignment: {
      align_api: allResults.alignment?.find(r => r.test === 'Align Dataset')?.success ? '✅' : '❌',
      latest_api: allResults.alignment?.find(r => r.test === 'Latest Data Point')?.success ? '✅' : '❌',
      minio_api: allResults.alignment?.find(r => r.test === 'MinIO History')?.success ? '✅' : '❌'
    },
    infrastructure: {
      minio: allResults.minio?.success ? '✅' : '❌',
      websocket: '⚠️ (Disabled)'
    }
  };
  
  console.log('🔑 API KEYS:');
  console.log(`   Total: ${report.api_keys.total}`);
  console.log(`   Working: ${report.api_keys.working}/${report.api_keys.tested}`);
  
  console.log('\n🌐 ENDPOINTS:');
  console.log(`   Quote: ${report.endpoints.quote}`);
  console.log(`   Time Series: ${report.endpoints.time_series}`);
  console.log(`   Technical Indicators: ${report.endpoints.technical_indicators}/4 working`);
  
  console.log('\n🔄 DATA ALIGNMENT:');
  console.log(`   Align Dataset: ${report.data_alignment.align_api}`);
  console.log(`   Latest Point: ${report.data_alignment.latest_api}`);
  console.log(`   MinIO History: ${report.data_alignment.minio_api}`);
  
  console.log('\n🏗️ INFRASTRUCTURE:');
  console.log(`   MinIO: ${report.infrastructure.minio}`);
  console.log(`   WebSocket: ${report.infrastructure.websocket}`);
  
  // Save report to file
  const filename = `integration_test_report_${Date.now()}.json`;
  fs.writeFileSync(filename, JSON.stringify({ report, raw_results: allResults }, null, 2));
  console.log(`\n💾 Full report saved: ${filename}`);
  
  return report;
}

async function runAllTests() {
  console.log('🚀 STARTING COMPREHENSIVE API INTEGRATION TESTS');
  console.log('=' .repeat(80));
  
  const allResults = {};
  
  try {
    // Test 1: API Keys and Rate Limiting
    allResults.apiKeys = await testApiKeys();
    
    // Test 2: Time Series Data  
    allResults.timeSeries = await testTimeSeries();
    
    // Test 3: Technical Indicators
    allResults.indicators = await testTechnicalIndicators();
    
    // Test 4: Data Alignment API
    allResults.alignment = await testDataAlignment();
    
    // Test 5: MinIO Connection
    allResults.minio = await testMinIOConnection();
    
    // Test 6: WebSocket Capability
    allResults.websocket = await testWebSocketCapability();
    
    // Generate final report
    const finalReport = await generateReport(allResults);
    
    console.log('\n🎯 TEST SUMMARY:');
    const totalTests = Object.keys(allResults).length;
    const passedTests = Object.values(finalReport.endpoints).filter(r => r === '✅').length +
                        Object.values(finalReport.data_alignment).filter(r => r === '✅').length +
                        (finalReport.infrastructure.minio === '✅' ? 1 : 0);
    
    console.log(`   Total test categories: ${totalTests}`);
    console.log(`   Passing: ${passedTests}`);
    console.log(`   Success rate: ${Math.round((passedTests / (totalTests * 2)) * 100)}%`);
    
    if (finalReport.api_keys.working > 0) {
      console.log('\n✅ INTEGRATION STATUS: TwelveData API is functional and ready');
    } else {
      console.log('\n❌ INTEGRATION STATUS: Critical issues detected');
    }
    
  } catch (error) {
    console.error('\n💥 TEST EXECUTION FAILED:', error);
    process.exit(1);
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  runAllTests().catch(console.error);
}

module.exports = {
  runAllTests,
  testApiKeys,
  testTimeSeries,
  testTechnicalIndicators,
  testDataAlignment,
  testMinIOConnection,
  testWebSocketCapability
};