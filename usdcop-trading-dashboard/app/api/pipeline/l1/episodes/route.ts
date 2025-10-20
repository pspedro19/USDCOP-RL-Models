/**
 * L1 Episodes Endpoint
 * GET /api/pipeline/l1/episodes
 *
 * Lists and retrieves L1 standardized episodes
 * Each episode contains 60 bars (5-minute OHLC) that passed quality gates
 *
 * Bucket: 01-l1-ds-usdcop-standardize
 * Format: Parquet files partitioned by episode_date
 */

import { NextRequest, NextResponse } from 'next/server';
import { minioClient } from '@/lib/services/minio-client';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const episodeId = searchParams.get('episode_id');
  const limit = Math.min(parseInt(searchParams.get('limit') || '100'), 1000);
  const startDate = searchParams.get('start_date');

  try {
    const bucket = '01-l1-ds-usdcop-standardize';
    const bucketExists = await minioClient.bucketExists(bucket);

    if (!bucketExists) {
      return NextResponse.json({
        success: false,
        error: `Bucket ${bucket} does not exist`,
      }, { status: 404 });
    }

    // List episodes
    const prefix = startDate ? `data/episode_date=${startDate}/` : 'data/';
    const objects = await minioClient.listObjects(bucket, prefix);

    // Filter for parquet files
    const episodes = objects.filter(obj => obj.name.endsWith('.parquet'));

    if (episodeId) {
      // Get specific episode
      const episode = episodes.find(e => e.name.includes(episodeId));

      if (!episode) {
        return NextResponse.json({
          success: false,
          message: `Episode ${episodeId} not found`,
        }, { status: 404 });
      }

      // Note: Actual parquet parsing would require a parquet library
      // For now, return metadata
      return NextResponse.json({
        success: true,
        episode: {
          id: episodeId,
          file: episode.name,
          lastModified: episode.lastModified,
          size: episode.size,
          format: 'parquet',
          note: 'Parquet file download available via direct MinIO access',
        },
      });
    }

    // Return episode list
    return NextResponse.json({
      success: true,
      timestamp: new Date().toISOString(),
      count: episodes.length,
      episodes: episodes.slice(0, limit).map(ep => ({
        file: ep.name,
        lastModified: ep.lastModified,
        size: ep.size,
        sizeKB: (ep.size / 1024).toFixed(2),
      })),
      pagination: {
        limit,
        total: episodes.length,
        hasMore: episodes.length > limit,
      },
    });

  } catch (error) {
    console.error('[L1 Episodes API] Error:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to retrieve L1 episodes',
      details: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
