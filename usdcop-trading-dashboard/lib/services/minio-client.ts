/**
 * Enhanced MinIO Client for Pipeline Data Access
 * Supports L0-L6 pipeline bucket access with real MinIO integration
 */

import * as Minio from 'minio';

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

// MinIO client instance (lazy initialization)
let minioClientInstance: Minio.Client | null = null;

/**
 * Get or create MinIO client
 */
function getMinioClient(): Minio.Client {
  if (!minioClientInstance) {
    minioClientInstance = new Minio.Client({
      endPoint: process.env.MINIO_ENDPOINT || 'localhost',
      port: parseInt(process.env.MINIO_PORT || '9000'),
      useSSL: process.env.MINIO_USE_SSL === 'true',
      accessKey: process.env.MINIO_ACCESS_KEY || 'minioadmin',
      secretKey: process.env.MINIO_SECRET_KEY || 'minioadmin123',
    });
  }
  return minioClientInstance;
}

// Enhanced MinIO client with real implementation
export const minioClient = {
  async listObjects(bucket: string, prefix?: string): Promise<MinioObjectInfo[]> {
    const client = getMinioClient();
    const objects: MinioObjectInfo[] = [];

    try {
      const stream = client.listObjectsV2(bucket, prefix || '', true);

      for await (const obj of stream) {
        objects.push({
          name: obj.name,
          lastModified: obj.lastModified,
          size: obj.size,
          etag: obj.etag || '',
        });
      }

      console.log(`[MinioClient] Listed ${objects.length} objects in ${bucket}/${prefix || ''}`);
      return objects;
    } catch (error) {
      console.error(`[MinioClient] Error listing objects in ${bucket}:`, error);
      // Return empty array on error (bucket might not exist yet)
      return [];
    }
  },

  async getObject(bucket: string, objectName: string): Promise<any> {
    const client = getMinioClient();

    try {
      console.log(`[MinioClient] Getting object: ${bucket}/${objectName}`);
      const stream = await client.getObject(bucket, objectName);
      const chunks: Buffer[] = [];

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      const content = Buffer.concat(chunks).toString();

      // Parse JSON if possible
      try {
        return JSON.parse(content);
      } catch {
        // Return raw content if not JSON
        return content;
      }
    } catch (error) {
      console.error(`[MinioClient] Error getting object ${bucket}/${objectName}:`, error);
      throw error;
    }
  },

  async putObject(bucket: string, objectName: string, data: any): Promise<void> {
    const client = getMinioClient();

    try {
      console.log(`[MinioClient] Putting object: ${bucket}/${objectName}`);
      const content = typeof data === 'string' ? data : JSON.stringify(data);
      await client.putObject(bucket, objectName, Buffer.from(content));
    } catch (error) {
      console.error(`[MinioClient] Error putting object ${bucket}/${objectName}:`, error);
      throw error;
    }
  },

  async query(query: DataQuery): Promise<any> {
    console.log('[MinioClient] Executing query:', query);

    // List objects based on query parameters
    const objects = await minioClient.listObjects(
      query.bucket,
      query.key || ''
    );

    // Filter by date if provided
    let filteredObjects = objects;
    if (query.startDate || query.endDate) {
      const start = query.startDate ? new Date(query.startDate) : new Date(0);
      const end = query.endDate ? new Date(query.endDate) : new Date();

      filteredObjects = objects.filter(obj => {
        return obj.lastModified >= start && obj.lastModified <= end;
      });
    }

    return {
      results: filteredObjects,
      count: filteredObjects.length
    };
  },

  async bucketExists(bucket: string): Promise<boolean> {
    const client = getMinioClient();
    try {
      return await client.bucketExists(bucket);
    } catch (error) {
      console.error(`[MinioClient] Error checking bucket ${bucket}:`, error);
      return false;
    }
  },

  async makeBucket(bucket: string): Promise<void> {
    const client = getMinioClient();
    try {
      console.log(`[MinioClient] Creating bucket: ${bucket}`);
      await client.makeBucket(bucket, 'us-east-1');
    } catch (error) {
      console.error(`[MinioClient] Error creating bucket ${bucket}:`, error);
      throw error;
    }
  },

  /**
   * Get JSON object from MinIO with error handling
   */
  async getJSON(bucket: string, objectName: string): Promise<any | null> {
    try {
      const data = await minioClient.getObject(bucket, objectName);
      if (typeof data === 'object') {
        return data;
      }
      return JSON.parse(data);
    } catch (error) {
      console.error(`[MinioClient] Error getting JSON from ${bucket}/${objectName}:`, error);
      return null;
    }
  },

  /**
   * List buckets
   */
  async listBuckets(): Promise<string[]> {
    const client = getMinioClient();
    try {
      const buckets = await client.listBuckets();
      return buckets.map(b => b.name);
    } catch (error) {
      console.error('[MinioClient] Error listing buckets:', error);
      return [];
    }
  }
}