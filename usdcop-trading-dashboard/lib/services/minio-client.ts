export interface DataQuery {
  bucket: string
  key?: string
  startDate?: string
  endDate?: string
  filters?: Record<string, any>
}

export interface MinioObjectInfo {
  name: string
  lastModified: Date
  size: number
  etag: string
}

// Mock MinIO client for development
export const minioClient = {
  async listObjects(bucket: string, prefix?: string): Promise<MinioObjectInfo[]> {
    // Mock implementation - would connect to actual MinIO in production
    console.log(`[MinioClient] Listing objects in bucket: ${bucket}, prefix: ${prefix}`)

    // Return mock object list
    return [
      {
        name: 'backtest-results/latest.json',
        lastModified: new Date(),
        size: 1024 * 50, // 50KB
        etag: 'mock-etag-123'
      },
      {
        name: 'market-data/usdcop-daily.parquet',
        lastModified: new Date(Date.now() - 86400000), // Yesterday
        size: 1024 * 1024 * 5, // 5MB
        etag: 'mock-etag-456'
      }
    ]
  },

  async getObject(bucket: string, objectName: string): Promise<any> {
    console.log(`[MinioClient] Getting object: ${bucket}/${objectName}`)

    // Mock data based on object type
    if (objectName.includes('backtest')) {
      return {
        runId: `run_${Date.now()}`,
        timestamp: new Date().toISOString(),
        data: 'mock backtest data'
      }
    }

    if (objectName.includes('market-data')) {
      return {
        symbol: 'USDCOP',
        data: Array.from({ length: 100 }, (_, i) => ({
          timestamp: new Date(Date.now() - i * 86400000).toISOString(),
          open: 4000 + Math.random() * 100,
          high: 4050 + Math.random() * 100,
          low: 3950 + Math.random() * 100,
          close: 4000 + Math.random() * 100,
          volume: Math.floor(Math.random() * 1000000)
        }))
      }
    }

    return null
  },

  async putObject(bucket: string, objectName: string, data: any): Promise<void> {
    console.log(`[MinioClient] Putting object: ${bucket}/${objectName}`)
    // Mock implementation - would upload to actual MinIO in production
  },

  async query(query: DataQuery): Promise<any> {
    console.log('[MinioClient] Executing query:', query)

    // Mock query response based on bucket
    switch (query.bucket) {
      case 'l6-backtest':
        return {
          results: 'mock backtest query results',
          count: 1
        }

      case 'market-data':
        return {
          results: 'mock market data query results',
          count: 100
        }

      default:
        return {
          results: 'mock query results',
          count: 0
        }
    }
  },

  async bucketExists(bucket: string): Promise<boolean> {
    // Mock - assume all buckets exist
    return true
  },

  async makeBucket(bucket: string): Promise<void> {
    console.log(`[MinioClient] Creating bucket: ${bucket}`)
    // Mock implementation
  }
}