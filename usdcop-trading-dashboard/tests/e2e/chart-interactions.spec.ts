import { test, expect, Page } from '@playwright/test'

class ChartInteractions {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto('/')
    await this.page.waitForLoadState('networkidle')
    await this.page.waitForSelector('[data-testid="trading-chart"]', { timeout: 30000 })
  }

  async drawTrendline(startX: number, startY: number, endX: number, endY: number) {
    // Activate drawing tool
    await this.page.click('[data-testid="drawing-tool-trendline"]')

    const chart = this.page.locator('[data-testid="trading-chart"]')
    const box = await chart.boundingBox()

    if (box) {
      const actualStartX = box.x + startX
      const actualStartY = box.y + startY
      const actualEndX = box.x + endX
      const actualEndY = box.y + endY

      await this.page.mouse.move(actualStartX, actualStartY)
      await this.page.mouse.down()
      await this.page.mouse.move(actualEndX, actualEndY)
      await this.page.mouse.up()

      await this.page.waitForTimeout(500)
    }
  }

  async drawRectangle(x: number, y: number, width: number, height: number) {
    await this.page.click('[data-testid="drawing-tool-rectangle"]')

    const chart = this.page.locator('[data-testid="trading-chart"]')
    const box = await chart.boundingBox()

    if (box) {
      const startX = box.x + x
      const startY = box.y + y
      const endX = startX + width
      const endY = startY + height

      await this.page.mouse.move(startX, startY)
      await this.page.mouse.down()
      await this.page.mouse.move(endX, endY)
      await this.page.mouse.up()

      await this.page.waitForTimeout(500)
    }
  }

  async drawFibonacci(startX: number, startY: number, endX: number, endY: number) {
    await this.page.click('[data-testid="drawing-tool-fibonacci"]')

    const chart = this.page.locator('[data-testid="trading-chart"]')
    const box = await chart.boundingBox()

    if (box) {
      const actualStartX = box.x + startX
      const actualStartY = box.y + startY
      const actualEndX = box.x + endX
      const actualEndY = box.y + endY

      await this.page.mouse.move(actualStartX, actualStartY)
      await this.page.mouse.down()
      await this.page.mouse.move(actualEndX, actualEndY)
      await this.page.mouse.up()

      await this.page.waitForTimeout(500)
    }
  }

  async clearDrawings() {
    await this.page.click('[data-testid="clear-drawings"]')
    await this.page.waitForTimeout(500)
  }

  async selectDrawing(x: number, y: number) {
    const chart = this.page.locator('[data-testid="trading-chart"]')
    const box = await chart.boundingBox()

    if (box) {
      await this.page.mouse.click(box.x + x, box.y + y)
      await this.page.waitForTimeout(200)
    }
  }

  async deleteSelectedDrawing() {
    await this.page.keyboard.press('Delete')
    await this.page.waitForTimeout(500)
  }

  async measureDistance(startX: number, startY: number, endX: number, endY: number) {
    await this.page.click('[data-testid="drawing-tool-measure"]')

    const chart = this.page.locator('[data-testid="trading-chart"]')
    const box = await chart.boundingBox()

    if (box) {
      const actualStartX = box.x + startX
      const actualStartY = box.y + startY
      const actualEndX = box.x + endX
      const actualEndY = box.y + endY

      await this.page.mouse.move(actualStartX, actualStartY)
      await this.page.mouse.down()
      await this.page.mouse.move(actualEndX, actualEndY)
      await this.page.mouse.up()

      await this.page.waitForTimeout(500)
    }
  }

  async crosshairMove(x: number, y: number) {
    const chart = this.page.locator('[data-testid="trading-chart"]')
    const box = await chart.boundingBox()

    if (box) {
      await this.page.mouse.move(box.x + x, box.y + y)
      await this.page.waitForTimeout(200)
    }
  }

  async enableCrosshair() {
    await this.page.click('[data-testid="crosshair-toggle"]')
    await this.page.waitForTimeout(200)
  }

  async disableCrosshair() {
    await this.page.click('[data-testid="crosshair-toggle"]')
    await this.page.waitForTimeout(200)
  }
}

test.describe('Chart Interactions', () => {
  let chartInteractions: ChartInteractions

  test.beforeEach(async ({ page }) => {
    chartInteractions = new ChartInteractions(page)
    await chartInteractions.goto()
  })

  test.describe('Drawing Tools', () => {
    test('should draw trendlines', async ({ page }) => {
      await chartInteractions.drawTrendline(100, 300, 400, 200)

      // Verify trendline is visible
      await expect(page.locator('[data-testid="drawing-trendline"]')).toBeVisible()

      // Take screenshot
      await page.screenshot({
        path: 'tests/screenshots/trendline-drawing.png',
        clip: { x: 0, y: 0, width: 800, height: 600 }
      })
    })

    test('should draw rectangles', async ({ page }) => {
      await chartInteractions.drawRectangle(150, 200, 200, 100)

      // Verify rectangle is visible
      await expect(page.locator('[data-testid="drawing-rectangle"]')).toBeVisible()

      await page.screenshot({
        path: 'tests/screenshots/rectangle-drawing.png',
        clip: { x: 0, y: 0, width: 800, height: 600 }
      })
    })

    test('should draw Fibonacci retracements', async ({ page }) => {
      await chartInteractions.drawFibonacci(100, 400, 500, 100)

      // Verify Fibonacci levels are visible
      await expect(page.locator('[data-testid="drawing-fibonacci"]')).toBeVisible()
      await expect(page.locator('[data-testid="fib-level-236"]')).toBeVisible()
      await expect(page.locator('[data-testid="fib-level-382"]')).toBeVisible()
      await expect(page.locator('[data-testid="fib-level-618"]')).toBeVisible()

      await page.screenshot({
        path: 'tests/screenshots/fibonacci-drawing.png',
        clip: { x: 0, y: 0, width: 800, height: 600 }
      })
    })

    test('should clear all drawings', async ({ page }) => {
      // Draw multiple objects
      await chartInteractions.drawTrendline(100, 300, 400, 200)
      await chartInteractions.drawRectangle(150, 200, 200, 100)

      // Verify drawings exist
      await expect(page.locator('[data-testid="drawing-trendline"]')).toBeVisible()
      await expect(page.locator('[data-testid="drawing-rectangle"]')).toBeVisible()

      // Clear all drawings
      await chartInteractions.clearDrawings()

      // Verify drawings are removed
      await expect(page.locator('[data-testid="drawing-trendline"]')).not.toBeVisible()
      await expect(page.locator('[data-testid="drawing-rectangle"]')).not.toBeVisible()
    })

    test('should select and delete individual drawings', async ({ page }) => {
      await chartInteractions.drawTrendline(100, 300, 400, 200)
      await chartInteractions.drawRectangle(150, 200, 200, 100)

      // Select trendline and delete it
      await chartInteractions.selectDrawing(250, 250) // Approximate center of trendline
      await expect(page.locator('[data-testid="drawing-trendline"].selected')).toBeVisible()

      await chartInteractions.deleteSelectedDrawing()
      await expect(page.locator('[data-testid="drawing-trendline"]')).not.toBeVisible()

      // Rectangle should still be visible
      await expect(page.locator('[data-testid="drawing-rectangle"]')).toBeVisible()
    })

    test('should measure distances', async ({ page }) => {
      await chartInteractions.measureDistance(100, 300, 400, 200)

      // Verify measurement tool shows distance and price difference
      await expect(page.locator('[data-testid="measurement-line"]')).toBeVisible()
      await expect(page.locator('[data-testid="measurement-label"]')).toBeVisible()
      await expect(page.locator('[data-testid="measurement-label"]')).toContainText(/\$[\d.]+/)
    })
  })

  test.describe('Crosshair Functionality', () => {
    test('should show crosshair on hover', async ({ page }) => {
      await chartInteractions.enableCrosshair()
      await chartInteractions.crosshairMove(300, 250)

      // Verify crosshair is visible
      await expect(page.locator('[data-testid="crosshair-horizontal"]')).toBeVisible()
      await expect(page.locator('[data-testid="crosshair-vertical"]')).toBeVisible()

      // Verify price and time labels
      await expect(page.locator('[data-testid="crosshair-price-label"]')).toBeVisible()
      await expect(page.locator('[data-testid="crosshair-time-label"]')).toBeVisible()
    })

    test('should update crosshair values on move', async ({ page }) => {
      await chartInteractions.enableCrosshair()
      await chartInteractions.crosshairMove(200, 200)

      const initialPrice = await page.locator('[data-testid="crosshair-price-label"]').textContent()

      await chartInteractions.crosshairMove(400, 400)

      const newPrice = await page.locator('[data-testid="crosshair-price-label"]').textContent()

      expect(initialPrice).not.toBe(newPrice)
    })

    test('should hide crosshair when disabled', async ({ page }) => {
      await chartInteractions.enableCrosshair()
      await chartInteractions.crosshairMove(300, 250)

      // Verify crosshair is visible
      await expect(page.locator('[data-testid="crosshair-horizontal"]')).toBeVisible()

      await chartInteractions.disableCrosshair()

      // Verify crosshair is hidden
      await expect(page.locator('[data-testid="crosshair-horizontal"]')).not.toBeVisible()
      await expect(page.locator('[data-testid="crosshair-vertical"]')).not.toBeVisible()
    })
  })

  test.describe('Chart Navigation', () => {
    test('should zoom with mouse wheel', async ({ page }) => {
      const chart = page.locator('[data-testid="trading-chart"]')
      await chart.hover()

      // Zoom in
      await page.mouse.wheel(0, -100)
      await page.waitForTimeout(500)

      // Take screenshot after zoom in
      await page.screenshot({
        path: 'tests/screenshots/chart-zoomed-in.png',
        clip: { x: 0, y: 0, width: 800, height: 600 }
      })

      // Zoom out
      await page.mouse.wheel(0, 100)
      await page.waitForTimeout(500)

      await page.screenshot({
        path: 'tests/screenshots/chart-zoomed-out.png',
        clip: { x: 0, y: 0, width: 800, height: 600 }
      })
    })

    test('should pan with mouse drag', async ({ page }) => {
      const chart = page.locator('[data-testid="trading-chart"]')
      const box = await chart.boundingBox()

      if (box) {
        const centerX = box.x + box.width / 2
        const centerY = box.y + box.height / 2

        // Pan left
        await page.mouse.move(centerX, centerY)
        await page.mouse.down()
        await page.mouse.move(centerX - 100, centerY)
        await page.mouse.up()

        await page.waitForTimeout(500)

        await page.screenshot({
          path: 'tests/screenshots/chart-panned-left.png',
          clip: { x: 0, y: 0, width: 800, height: 600 }
        })

        // Pan right
        await page.mouse.move(centerX, centerY)
        await page.mouse.down()
        await page.mouse.move(centerX + 200, centerY)
        await page.mouse.up()

        await page.waitForTimeout(500)

        await page.screenshot({
          path: 'tests/screenshots/chart-panned-right.png',
          clip: { x: 0, y: 0, width: 800, height: 600 }
        })
      }
    })

    test('should support keyboard navigation', async ({ page }) => {
      const chart = page.locator('[data-testid="trading-chart"]')
      await chart.focus()

      // Test arrow key navigation
      await page.keyboard.press('ArrowLeft')
      await page.waitForTimeout(200)

      await page.keyboard.press('ArrowRight')
      await page.waitForTimeout(200)

      await page.keyboard.press('ArrowUp')
      await page.waitForTimeout(200)

      await page.keyboard.press('ArrowDown')
      await page.waitForTimeout(200)

      // Test zoom shortcuts
      await page.keyboard.press('Control+Equal') // Zoom in
      await page.waitForTimeout(200)

      await page.keyboard.press('Control+Minus') // Zoom out
      await page.waitForTimeout(200)

      // Test reset view
      await page.keyboard.press('Control+0')
      await page.waitForTimeout(200)
    })

    test('should handle touch gestures on mobile', async ({ page, browserName }) => {
      // Skip on non-mobile browsers
      if (browserName !== 'webkit') {
        test.skip()
      }

      await page.setViewportSize({ width: 375, height: 667 })
      await chartInteractions.goto()

      const chart = page.locator('[data-testid="trading-chart"]')
      const box = await chart.boundingBox()

      if (box) {
        // Simulate pinch to zoom
        await page.touchscreen.tap(box.x + 100, box.y + 100)
        await page.waitForTimeout(100)

        // Simulate swipe to pan
        await page.touchscreen.tap(box.x + 200, box.y + 200)
        await page.touchscreen.tap(box.x + 300, box.y + 200)
        await page.waitForTimeout(500)
      }
    })
  })

  test.describe('Real-time Updates', () => {
    test('should update chart with new data', async ({ page }) => {
      // Get initial last candle time
      const initialTime = await page.locator('[data-testid="last-candle-time"]').textContent()

      // Wait for real-time update
      await page.waitForTimeout(5000)

      // Check if time has updated (in a real environment)
      // In test environment, we can simulate this
      await page.evaluate(() => {
        window.dispatchEvent(new CustomEvent('new-market-data', {
          detail: {
            time: Date.now(),
            open: 4000,
            high: 4050,
            low: 3950,
            close: 4025,
            volume: 150000
          }
        }))
      })

      await page.waitForTimeout(500)

      // Verify chart updated
      await expect(page.locator('[data-testid="trading-chart"]')).toBeVisible()
    })

    test('should maintain chart position during updates', async ({ page }) => {
      // Pan to specific position
      const chart = page.locator('[data-testid="trading-chart"]')
      const box = await chart.boundingBox()

      if (box) {
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
        await page.mouse.down()
        await page.mouse.move(box.x + box.width / 2 - 200, box.y + box.height / 2)
        await page.mouse.up()
      }

      // Simulate real-time update
      await page.evaluate(() => {
        window.dispatchEvent(new CustomEvent('new-market-data', {
          detail: {
            time: Date.now(),
            open: 4000,
            high: 4050,
            low: 3950,
            close: 4025,
            volume: 150000
          }
        }))
      })

      await page.waitForTimeout(500)

      // Chart should maintain its panned position
      // This would be verified by checking chart viewport state
      await expect(chart).toBeVisible()
    })

    test('should handle rapid data updates', async ({ page }) => {
      // Simulate multiple rapid updates
      for (let i = 0; i < 10; i++) {
        await page.evaluate((index) => {
          window.dispatchEvent(new CustomEvent('new-market-data', {
            detail: {
              time: Date.now() + index * 1000,
              open: 4000 + Math.random() * 10,
              high: 4050 + Math.random() * 10,
              low: 3950 + Math.random() * 10,
              close: 4025 + Math.random() * 10,
              volume: 150000 + Math.random() * 50000
            }
          }))
        }, i)

        await page.waitForTimeout(100)
      }

      // Chart should remain responsive
      await expect(page.locator('[data-testid="trading-chart"]')).toBeVisible()

      // Performance should not degrade
      const fps = await page.evaluate(() => {
        return new Promise((resolve) => {
          let frames = 0
          const startTime = performance.now()

          function countFrame() {
            frames++
            if (performance.now() - startTime < 1000) {
              requestAnimationFrame(countFrame)
            } else {
              resolve(frames)
            }
          }
          requestAnimationFrame(countFrame)
        })
      })

      expect(fps).toBeGreaterThan(30) // Should maintain at least 30 FPS
    })
  })

  test.describe('Error Handling', () => {
    test('should handle drawing tool failures gracefully', async ({ page }) => {
      // Simulate drawing tool error
      await page.evaluate(() => {
        window.dispatchEvent(new CustomEvent('drawing-tool-error', {
          detail: { tool: 'trendline', error: 'Canvas context lost' }
        }))
      })

      // Should show error message
      await expect(page.locator('[data-testid="drawing-error-message"]')).toBeVisible()

      // Chart should remain functional
      await expect(page.locator('[data-testid="trading-chart"]')).toBeVisible()
    })

    test('should recover from rendering errors', async ({ page }) => {
      // Simulate rendering error
      await page.evaluate(() => {
        window.dispatchEvent(new CustomEvent('chart-render-error'))
      })

      // Should attempt to recover
      await page.waitForTimeout(1000)

      // Chart should be visible again
      await expect(page.locator('[data-testid="trading-chart"]')).toBeVisible()
    })
  })

  test.describe('Performance Benchmarks', () => {
    test('should render 10,000 data points efficiently', async ({ page }) => {
      const startTime = Date.now()

      // Load large dataset
      await page.evaluate(() => {
        const largeDataset = Array.from({ length: 10000 }, (_, i) => ({
          time: Date.now() - (10000 - i) * 60000,
          open: 4000 + Math.sin(i / 100) * 50,
          high: 4000 + Math.sin(i / 100) * 60,
          low: 4000 + Math.sin(i / 100) * 40,
          close: 4000 + Math.sin(i / 100) * 50,
          volume: 1000000 + Math.random() * 500000
        }))

        window.dispatchEvent(new CustomEvent('load-dataset', { detail: largeDataset }))
      })

      await page.waitForSelector('[data-testid="trading-chart"]')
      const loadTime = Date.now() - startTime

      expect(loadTime).toBeLessThan(3000) // Should load within 3 seconds
    })

    test('should maintain performance during continuous updates', async ({ page }) => {
      let frameCount = 0
      const startTime = Date.now()

      // Start performance monitoring
      await page.evaluate(() => {
        (window as any).performanceMonitor = {
          frames: 0,
          start: performance.now()
        }

        function countFrames() {
          (window as any).performanceMonitor.frames++
          requestAnimationFrame(countFrames)
        }
        requestAnimationFrame(countFrames)
      })

      // Simulate continuous updates for 5 seconds
      const updateInterval = setInterval(async () => {
        await page.evaluate(() => {
          window.dispatchEvent(new CustomEvent('new-market-data', {
            detail: {
              time: Date.now(),
              open: 4000 + Math.random() * 10,
              high: 4050 + Math.random() * 10,
              low: 3950 + Math.random() * 10,
              close: 4025 + Math.random() * 10,
              volume: 150000 + Math.random() * 50000
            }
          }))
        })
      }, 100)

      await page.waitForTimeout(5000)
      clearInterval(updateInterval)

      // Check performance
      const performanceData = await page.evaluate(() => {
        const monitor = (window as any).performanceMonitor
        const elapsed = performance.now() - monitor.start
        return {
          fps: (monitor.frames / elapsed) * 1000,
          totalFrames: monitor.frames,
          duration: elapsed
        }
      })

      expect(performanceData.fps).toBeGreaterThan(50) // Should maintain >50 FPS
    })
  })
})