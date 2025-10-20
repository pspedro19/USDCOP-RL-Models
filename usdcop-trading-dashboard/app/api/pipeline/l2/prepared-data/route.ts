/**
 * L2 Prepared Data Endpoint
 * GET /api/pipeline/l2/prepared-data
 *
 * Provides L2 prepared/deseasonalized data
 * Bucket: 02-l2-ds-usdcop-prep
 * Features: HoD baselines, deseasonalized OHLC, return series
 */

import { NextRequest, NextResponse } from 'next/server';
import { minioClient } from '@/lib/services/minio-client';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const episodeId = searchParams.get('episode_id');
  const limit = Math.min(parseInt(searchParams.get('limit') || '100'), 1000);

  try {
    const bucket = '02-l2-ds-usdcop-prep';
    const bucketExists = await minioClient.bucketExists(bucket);

    if (!bucketExists) {
      return NextResponse.json({
        success: false,
        error: `Bucket ${bucket} does not exist. L2 pipeline may not have run yet.`,
      }, { status: 404 });
    }

    // List prepared data files
    const objects = await minioClient.listObjects(bucket, 'data/');
    const dataFiles = objects.filter(obj => obj.name.endsWith('.parquet'));

    if (episodeId) {
      const episode = dataFiles.find(e => e.name.includes(episodeId));

      if (!episode) {
        return NextResponse.json({
          success: false,
          message: `Episode ${episodeId} not found in L2`,
        }, { status: 404 });
      }

      return NextResponse.json({
        success: true,
        episode: {
          id: episodeId,
          file: episode.name,
          lastModified: episode.lastModified,
          size: episode.size,
          layer: 'L2',
          description: 'Deseasonalized and prepared data with HoD baselines',
        },
      });
    }

    // Get HoD baselines if available
    const hodObjects = await minioClient.listObjects(bucket, 'hod_baselines/');

    return NextResponse.json({
      success: true,
      timestamp: new Date().toISOString(),
      count: dataFiles.length,
      preparedData: dataFiles.slice(0, limit).map(file => ({
        file: file.name,
        lastModified: file.lastModified,
        size: file.size,
        sizeKB: (file.size / 1024).toFixed(2),
      })),
      hodBaselines: {
        count: hodObjects.length,
        files: hodObjects.slice(0, 10).map(obj => obj.name),
      },
      pagination: {
        limit,
        total: dataFiles.length,
        hasMore: dataFiles.length > limit,
      },
    });

  } catch (error) {
    console.error('[L2 Prepared Data API] Error:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to retrieve L2 prepared data',
      details: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
