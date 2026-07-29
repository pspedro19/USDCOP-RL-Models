import { FullConfig } from '@playwright/test'
import { assertArtifactCoversCode } from './support/artifact-freshness'

async function waitForApp(baseURL: string): Promise<void> {
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

async function globalSetup(config: FullConfig) {
  const baseURL = process.env.BASE_URL || 'http://localhost:5000'

  await waitForApp(baseURL)

  // K-044: identificar el artefacto ANTES de medir nada contra el. Va FUERA del bucle de
  // espera a proposito: dentro, el `catch` de "servidor aun no listo" se tragaba el abort
  // y lo convertia en 30 reintentos silenciosos — el mismo defecto que un `except Exception`
  // que se come una muralla. Ver support/artifact-freshness.ts.
  await assertArtifactCoversCode(baseURL)
}

export default globalSetup
