/**
 * L3 Features Endpoint
 * GET /api/pipeline/l3/features
 *
 * Provides L3 engineered features (17 features per episode)
 * Bucket: 03-l3-ds-usdcop-features
 *
 * Features include:
 * - Price momentum, volatility, volume features
 * - Technical indicators
 * - Market microstructure features
 * - IC (Information Coefficient) compliance checks
 */

import { NextRequest, NextResponse } from 'next/server';
import { minioClient } from '@/lib/services/minio-client';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const episodeId = searchParams.get('episode_id');
  const limit = Math.min(parseInt(searchParams.get('limit') || '100'), 1000);

  try {
    const bucket = '03-l3-ds-usdcop-features';
    const bucketExists = await minioClient.bucketExists(bucket);

    if (!bucketExists) {
      return NextResponse.json({
        success: false,
        error: `Bucket ${bucket} does not exist. L3 pipeline may not have run yet.`,
      }, { status: 404 });
    }

    // List feature files
    const objects = await minioClient.listObjects(bucket, 'data/');
    const featureFiles = objects.filter(obj => obj.name.endsWith('.parquet'));

    if (episodeId) {
      const episode = featureFiles.find(e => e.name.includes(episodeId));

      if (!episode) {
        return NextResponse.json({
          success: false,
          message: `Episode ${episodeId} not found in L3`,
        }, { status: 404 });
      }

      return NextResponse.json({
        success: true,
        episode: {
          id: episodeId,
          file: episode.name,
          lastModified: episode.lastModified,
          size: episode.size,
          layer: 'L3',
          featureCount: 17,
          description: 'Engineered features with IC compliance',
        },
      });
    }

    // Get IC analysis reports if available
    const icReports = await minioClient.listObjects(bucket, '_reports/ic_analysis_');

    // Get feature importance if available
    const featureImportance = await minioClient.listObjects(bucket, '_reports/feature_importance_');

    return NextResponse.json({
      success: true,
      timestamp: new Date().toISOString(),
      count: featureFiles.length,
      features: featureFiles.slice(0, limit).map(file => ({
        file: file.name,
        lastModified: file.lastModified,
        size: file.size,
        sizeKB: (file.size / 1024).toFixed(2),
      })),
      analysis: {
        icReports: icReports.length,
        featureImportance: featureImportance.length,
        latestICReport: icReports.length > 0 ? icReports[icReports.length - 1].name : null,
      },
      pagination: {
        limit,
        total: featureFiles.length,
        hasMore: featureFiles.length > limit,
      },
    });

  } catch (error) {
    console.error('[L3 Features API] Error:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to retrieve L3 features',
      details: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
