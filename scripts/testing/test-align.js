// Test script to check align API response
// Node 18+ has native fetch

async function testAlign() {
  try {
    console.log('Fetching aligned dataset...');
    const response = await fetch('http://localhost:3010/api/market/realtime?action=align');
    const result = await response.json();
    
    console.log('\n=== API Response Meta ===');
    console.log('Total points:', result.meta.total);
    console.log('Historical points:', result.meta.historical);
    console.log('Realtime points:', result.meta.realtime);
    console.log('Date range:', result.meta.startDate, 'to', result.meta.endDate);
    
    console.log('\n=== Last 10 Data Points ===');
    const last10 = result.data.slice(-10);
    last10.forEach(point => {
      console.log(`${point.datetime} - Close: ${point.close}, Source: ${point.source}`);
    });
    
    console.log('\n=== Check for Today\'s Data ===');
    const today = new Date().toISOString().split('T')[0];
    const todayData = result.data.filter(p => p.datetime.includes(today));
    console.log(`Found ${todayData.length} points for today (${today})`);
    if (todayData.length > 0) {
      console.log('Sample today data:');
      todayData.slice(-5).forEach(p => {
        console.log(`  ${p.datetime}: ${p.close}`);
      });
    }
    
  } catch (error) {
    console.error('Error:', error);
  }
}

testAlign();