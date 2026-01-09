import { FullConfig } from '@playwright/test'

async function globalSetup(config: FullConfig) {
  // Wait for the application to be ready using fetch
  const baseURL = 'http://localhost:5000'
  const maxRetries = 30
  const retryDelay = 1000

  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(`${baseURL}/login`)
      if (response.ok) {
        console.log('✅ Application is ready for testing')
        return
      }
    } catch (error) {
      // Server not ready yet
    }
    await new Promise(resolve => setTimeout(resolve, retryDelay))
  }

  throw new Error('❌ Application failed to start after 30 seconds')
}

export default globalSetup