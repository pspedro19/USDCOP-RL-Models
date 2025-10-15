import { chromium, FullConfig } from '@playwright/test'

async function globalSetup(config: FullConfig) {
  // Launch browser for global setup
  const browser = await chromium.launch()
  const page = await browser.newPage()

  // Wait for the application to be ready
  try {
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' })
    console.log('✅ Application is ready for testing')
  } catch (error) {
    console.error('❌ Application failed to start:', error)
    throw error
  } finally {
    await browser.close()
  }
}

export default globalSetup