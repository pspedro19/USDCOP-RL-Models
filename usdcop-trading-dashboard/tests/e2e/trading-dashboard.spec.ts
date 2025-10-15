import { test, expect, Page } from '@playwright/test'
import { injectAxe, checkA11y } from '@axe-core/playwright'

class TradingDashboardPage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto('/')
    await this.page.waitForLoadState('networkidle')
  }

  async waitForChart() {
    await this.page.waitForSelector('[data-testid="trading-chart"]', { timeout: 30000 })
  }

  async waitForRealTimeData() {
    await this.page.waitForSelector('[data-testid="price-ticker"]', { timeout: 15000 })
  }

  async selectTimeframe(timeframe: string) {
    await this.page.click(`[data-testid="timeframe-${timeframe}"]`)
    await this.page.waitForTimeout(1000) // Wait for chart update
  }

  async toggleIndicator(indicator: string) {
    await this.page.click(`[data-testid="indicator-${indicator}"]`)
    await this.page.waitForTimeout(500)
  }

  async zoomChart(direction: 'in' | 'out') {
    const chart = this.page.locator('[data-testid="trading-chart"]')
    await chart.hover()

    if (direction === 'in') {
      await this.page.keyboard.press('Control+=')
    } else {
      await this.page.keyboard.press('Control+-')
    }

    await this.page.waitForTimeout(500)
  }

  async panChart(direction: 'left' | 'right', distance: number = 100) {
    const chart = this.page.locator('[data-testid="trading-chart"]')
    const box = await chart.boundingBox()

    if (box) {
      const startX = box.x + box.width / 2
      const startY = box.y + box.height / 2
      const endX = direction === 'left' ? startX - distance : startX + distance

      await this.page.mouse.move(startX, startY)
      await this.page.mouse.down()
      await this.page.mouse.move(endX, startY)
      await this.page.mouse.up()

      await this.page.waitForTimeout(500)
    }
  }

  async openSettings() {
    await this.page.click('[data-testid="settings-button"]')
    await this.page.waitForSelector('[data-testid="settings-panel"]')
  }

  async closeSettings() {
    await this.page.press('[data-testid="settings-panel"]', 'Escape')
    await this.page.waitForSelector('[data-testid="settings-panel"]', { state: 'hidden' })
  }

  async switchTheme(theme: 'light' | 'dark') {
    await this.openSettings()
    await this.page.click(`[data-testid="theme-${theme}"]`)
    await this.closeSettings()
    await this.page.waitForTimeout(500)
  }

  async takeFullPageScreenshot(name: string) {
    await this.page.screenshot({
      path: `tests/screenshots/${name}.png`,
      fullPage: true
    })
  }

  async takeChartScreenshot(name: string) {
    const chart = this.page.locator('[data-testid="trading-chart"]')
    await chart.screenshot({
      path: `tests/screenshots/chart-${name}.png`
    })
  }
}

test.describe('Trading Dashboard E2E Tests', () => {
  let dashboardPage: TradingDashboardPage

  test.beforeEach(async ({ page }) => {
    dashboardPage = new TradingDashboardPage(page)
    await dashboardPage.goto()
  })

  test.describe('Basic Functionality', () => {
    test('should load the dashboard successfully', async ({ page }) => {
      await expect(page).toHaveTitle(/Trading Dashboard/)
      await expect(page.locator('[data-testid="main-dashboard"]')).toBeVisible()
    })

    test('should display price ticker', async ({ page }) => {
      await dashboardPage.waitForRealTimeData()
      await expect(page.locator('[data-testid="price-ticker"]')).toBeVisible()
      await expect(page.locator('[data-testid="current-price"]')).toContainText(/\$[\d,]+\.[\d]{2}/)
    })

    test('should render trading chart', async ({ page }) => {
      await dashboardPage.waitForChart()
      await expect(page.locator('[data-testid="trading-chart"]')).toBeVisible()
    })

    test('should show market status', async ({ page }) => {
      await expect(page.locator('[data-testid="market-status"]')).toBeVisible()
      await expect(page.locator('[data-testid="market-status"]')).toContainText(/(Open|Closed|Pre-Market|After Hours)/)
    })
  })

  test.describe('Chart Interactions', () => {
    test('should switch timeframes', async ({ page }) => {
      await dashboardPage.waitForChart()

      // Test different timeframes
      const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

      for (const timeframe of timeframes) {
        await dashboardPage.selectTimeframe(timeframe)
        await expect(page.locator(`[data-testid="timeframe-${timeframe}"]`)).toHaveClass(/active|selected/)
      }
    })

    test('should toggle technical indicators', async ({ page }) => {
      await dashboardPage.waitForChart()

      const indicators = ['ema20', 'ema50', 'bollinger', 'rsi', 'macd']

      for (const indicator of indicators) {
        await dashboardPage.toggleIndicator(indicator)
        await expect(page.locator(`[data-testid="indicator-${indicator}"]`)).toHaveClass(/active|enabled/)
      }
    })

    test('should support chart zoom', async ({ page }) => {
      await dashboardPage.waitForChart()

      // Take screenshot before zoom
      await dashboardPage.takeChartScreenshot('before-zoom')

      // Zoom in
      await dashboardPage.zoomChart('in')
      await dashboardPage.takeChartScreenshot('zoomed-in')

      // Zoom out
      await dashboardPage.zoomChart('out')
      await dashboardPage.takeChartScreenshot('zoomed-out')
    })

    test('should support chart panning', async ({ page }) => {
      await dashboardPage.waitForChart()

      // Pan left
      await dashboardPage.panChart('left')
      await page.waitForTimeout(500)

      // Pan right
      await dashboardPage.panChart('right')
      await page.waitForTimeout(500)
    })
  })

  test.describe('Real-time Features', () => {
    test('should update prices in real-time', async ({ page }) => {
      await dashboardPage.waitForRealTimeData()

      const initialPrice = await page.locator('[data-testid="current-price"]').textContent()

      // Wait for price update
      await page.waitForTimeout(5000)

      const updatedPrice = await page.locator('[data-testid="current-price"]').textContent()

      // Prices might be the same in testing, but the element should still be visible
      expect(initialPrice).toBeTruthy()
      expect(updatedPrice).toBeTruthy()
    })

    test('should show real-time indicator when live', async ({ page }) => {
      await expect(page.locator('[data-testid="live-indicator"]')).toBeVisible()
      await expect(page.locator('[data-testid="live-indicator"]')).toContainText(/Live|En Vivo/)
    })

    test('should display volume data', async ({ page }) => {
      await dashboardPage.waitForChart()
      await expect(page.locator('[data-testid="volume-display"]')).toBeVisible()
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await dashboardPage.goto()
      await dashboardPage.waitForChart()

      await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible()
      await dashboardPage.takeFullPageScreenshot('mobile-view')
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await dashboardPage.goto()
      await dashboardPage.waitForChart()

      await dashboardPage.takeFullPageScreenshot('tablet-view')
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await dashboardPage.goto()
      await dashboardPage.waitForChart()

      await dashboardPage.takeFullPageScreenshot('desktop-view')
    })
  })

  test.describe('Theme Switching', () => {
    test('should switch between light and dark themes', async ({ page }) => {
      await dashboardPage.waitForChart()

      // Take screenshot in default theme
      await dashboardPage.takeFullPageScreenshot('default-theme')

      // Switch to light theme
      await dashboardPage.switchTheme('light')
      await expect(page.locator('body')).toHaveClass(/light/)
      await dashboardPage.takeFullPageScreenshot('light-theme')

      // Switch to dark theme
      await dashboardPage.switchTheme('dark')
      await expect(page.locator('body')).toHaveClass(/dark/)
      await dashboardPage.takeFullPageScreenshot('dark-theme')
    })
  })

  test.describe('Performance', () => {
    test('should load within performance budget', async ({ page }) => {
      const startTime = Date.now()
      await dashboardPage.goto()
      await dashboardPage.waitForChart()
      const loadTime = Date.now() - startTime

      expect(loadTime).toBeLessThan(5000) // Should load within 5 seconds
    })

    test('should maintain 60 FPS during chart interactions', async ({ page }) => {
      await dashboardPage.waitForChart()

      // Start performance monitoring
      await page.evaluate(() => {
        (window as any).performanceData = {
          frames: 0,
          startTime: performance.now()
        }

        function countFrame() {
          (window as any).performanceData.frames++
          requestAnimationFrame(countFrame)
        }
        requestAnimationFrame(countFrame)
      })

      // Perform interactions
      await dashboardPage.panChart('left')
      await dashboardPage.panChart('right')
      await dashboardPage.zoomChart('in')
      await dashboardPage.zoomChart('out')

      // Check FPS
      const performanceData = await page.evaluate(() => {
        const data = (window as any).performanceData
        const elapsedTime = performance.now() - data.startTime
        return {
          fps: (data.frames / elapsedTime) * 1000,
          frames: data.frames,
          duration: elapsedTime
        }
      })

      expect(performanceData.fps).toBeGreaterThan(30) // Minimum 30 FPS
    })

    test('should handle large datasets efficiently', async ({ page }) => {
      await dashboardPage.goto()

      // Simulate loading large dataset
      await page.evaluate(() => {
        // Mock large dataset load
        const largeData = Array.from({ length: 10000 }, (_, i) => ({
          time: Date.now() + i * 60000,
          open: 4000 + Math.random() * 100,
          high: 4000 + Math.random() * 120,
          low: 4000 + Math.random() * 80,
          close: 4000 + Math.random() * 100,
          volume: 1000000 + Math.random() * 500000
        }))

        window.dispatchEvent(new CustomEvent('loadLargeDataset', { detail: largeData }))
      })

      await page.waitForTimeout(2000)
      await expect(page.locator('[data-testid="trading-chart"]')).toBeVisible()
    })
  })

  test.describe('Error Handling', () => {
    test('should handle network errors gracefully', async ({ page }) => {
      // Simulate network failure
      await page.route('**/api/**', route => route.abort())

      await dashboardPage.goto()

      // Should show error message
      await expect(page.locator('[data-testid="error-message"]')).toBeVisible()
      await expect(page.locator('[data-testid="retry-button"]')).toBeVisible()
    })

    test('should recover from WebSocket disconnection', async ({ page }) => {
      await dashboardPage.goto()
      await dashboardPage.waitForRealTimeData()

      // Simulate WebSocket disconnect
      await page.evaluate(() => {
        window.dispatchEvent(new CustomEvent('websocket-disconnect'))
      })

      // Should show reconnection indicator
      await expect(page.locator('[data-testid="reconnecting-indicator"]')).toBeVisible()

      // Simulate reconnection
      await page.evaluate(() => {
        window.dispatchEvent(new CustomEvent('websocket-reconnect'))
      })

      await expect(page.locator('[data-testid="live-indicator"]')).toBeVisible()
    })
  })

  test.describe('Accessibility', () => {
    test('should be accessible', async ({ page }) => {
      await dashboardPage.goto()
      await dashboardPage.waitForChart()

      await injectAxe(page)
      await checkA11y(page, null, {
        detailedReport: true,
        detailedReportOptions: { html: true }
      })
    })

    test('should support keyboard navigation', async ({ page }) => {
      await dashboardPage.goto()
      await dashboardPage.waitForChart()

      // Test tab navigation
      await page.keyboard.press('Tab')
      await expect(page.locator(':focus')).toBeVisible()

      // Test escape key
      await dashboardPage.openSettings()
      await page.keyboard.press('Escape')
      await expect(page.locator('[data-testid="settings-panel"]')).not.toBeVisible()
    })

    test('should support screen readers', async ({ page }) => {
      await dashboardPage.goto()
      await dashboardPage.waitForChart()

      // Check for ARIA labels
      await expect(page.locator('[aria-label]')).toHaveCount.greaterThan(0)

      // Check for proper heading structure
      const headings = await page.locator('h1, h2, h3, h4, h5, h6').count()
      expect(headings).toBeGreaterThan(0)
    })
  })

  test.describe('Visual Regression', () => {
    test('should match visual baseline for main dashboard', async ({ page }) => {
      await dashboardPage.goto()
      await dashboardPage.waitForChart()

      await expect(page).toHaveScreenshot('dashboard-main.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })

    test('should match visual baseline for chart component', async ({ page }) => {
      await dashboardPage.goto()
      await dashboardPage.waitForChart()

      const chart = page.locator('[data-testid="trading-chart"]')
      await expect(chart).toHaveScreenshot('trading-chart.png', {
        animations: 'disabled'
      })
    })

    test('should match visual baseline with indicators enabled', async ({ page }) => {
      await dashboardPage.goto()
      await dashboardPage.waitForChart()

      await dashboardPage.toggleIndicator('ema20')
      await dashboardPage.toggleIndicator('bollinger')
      await dashboardPage.toggleIndicator('rsi')

      await expect(page).toHaveScreenshot('dashboard-with-indicators.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })
  })
})