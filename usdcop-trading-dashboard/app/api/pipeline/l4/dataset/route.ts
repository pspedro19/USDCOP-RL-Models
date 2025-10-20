/**
 * L4 RL-Ready Dataset Endpoint
 * GET /api/pipeline/l4/dataset
 *
 * Provides L4 RL-ready datasets with train/val/test splits
 * Bucket: 04-l4-ds-usdcop-rlready
 *
 * Dataset: 929 episodes → 557 train / 186 val / 186 test
 * Observations: Normalized features for RL model input
 */

import { NextRequest, NextResponse } from 'next/server';
import { minioClient } from '@/lib/services/minio-client';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const split = searchParams.get('split'); // 'train', 'val', or 'test'
  const episodeId = searchParams.get('episode_id');

  try {
    const bucket = '04-l4-ds-usdcop-rlready';
    const bucketExists = await minioClient.bucketExists(bucket);

    if (!bucketExists) {
      return NextResponse.json({
        success: false,
        error: `Bucket ${bucket} does not exist. L4 pipeline may not have run yet.`,
      }, { status: 404 });
    }

    // Get dataset manifest
    const manifest = await minioClient.getJSON(bucket, 'dataset_manifest.json');

    // List data by split
    const splits = ['train', 'val', 'test'];
    const datasetInfo: any = {
      manifest,
      splits: {},
    };

    for (const splitName of splits) {
      if (!split || split === splitName) {
        const splitObjects = await minioClient.listObjects(bucket, `data/split=${splitName}/`);
        const parquetFiles = splitObjects.filter(obj => obj.name.endsWith('.parquet'));

        datasetInfo.splits[splitName] = {
          count: parquetFiles.length,
          totalSize: splitObjects.reduce((sum, obj) => sum + obj.size, 0),
          files: parquetFiles.slice(0, 10).map(file => ({
            name: file.name,
            lastModified: file.lastModified,
            size: file.size,
          })),
        };
      }
    }

    // Get specific episode if requested
    if (episodeId && split) {
      const episodeObjects = await minioClient.listObjects(
        bucket,
        `data/split=${split}/episode_id=${episodeId}/`
      );

      if (episodeObjects.length === 0) {
        return NextResponse.json({
          success: false,
          message: `Episode ${episodeId} not found in split ${split}`,
        }, { status: 404 });
      }

      return NextResponse.json({
        success: true,
        episode: {
          id: episodeId,
          split,
          files: episodeObjects.map(obj => ({
            name: obj.name,
            lastModified: obj.lastModified,
            size: obj.size,
          })),
        },
      });
    }

    return NextResponse.json({
      success: true,
      timestamp: new Date().toISOString(),
      dataset: datasetInfo,
      summary: {
        totalEpisodes: manifest?.total_episodes || 929,
        trainEpisodes: manifest?.train_episodes || 557,
        valEpisodes: manifest?.val_episodes || 186,
        testEpisodes: manifest?.test_episodes || 186,
      },
    });

  } catch (error) {
    console.error('[L4 Dataset API] Error:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to retrieve L4 RL-ready dataset',
      details: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
