/**
 * L1 Quality Report Endpoint
 * GET /api/pipeline/l1/quality-report
 *
 * Provides L1 standardization quality reports from MinIO:
 * Bucket: 01-l1-ds-usdcop-standardize
 * Path: _reports/quality_report_*.json
 */

import { NextRequest, NextResponse } from 'next/server';
import { minioClient } from '@/lib/services/minio-client';
import { createApiResponse } from '@/lib/types/api';
import { withAuth } from '@/lib/auth/api-auth';

export const GET = withAuth(async (request, { user }) => {
  const searchParams = request.nextUrl.searchParams;
  const runId = searchParams.get('run_id');
  const startDate = searchParams.get('start_date');
  const endDate = searchParams.get('end_date');

  try {
    const bucket = '01-l1-ds-usdcop-standardize';
    const bucketExists = await minioClient.bucketExists(bucket);

    if (!bucketExists) {
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: `Bucket ${bucket} does not exist. L1 pipeline may not have run yet.`,
        }),
        { status: 404 }
      );
    }

    // List quality reports
    const reports = await minioClient.listObjects(bucket, '_reports/');

    if (reports.length === 0) {
      return NextResponse.json(
        createApiResponse(false, 'minio', {
          message: 'No quality reports found',
          bucket,
        }),
        { status: 404 }
      );
    }

    // Filter reports by date if provided
    let filteredReports = reports;
    if (startDate || endDate) {
      const start = startDate ? new Date(startDate) : new Date(0);
      const end = endDate ? new Date(endDate) : new Date();

      filteredReports = reports.filter(r =>
        r.lastModified >= start && r.lastModified <= end
      );
    }

    // Get latest report or specific run_id
    const targetReport = runId
      ? filteredReports.find(r => r.name.includes(runId))
      : filteredReports.sort((a, b) => b.lastModified.getTime() - a.lastModified.getTime())[0];

    if (!targetReport) {
      return NextResponse.json(
        createApiResponse(false, 'minio', {
          message: runId ? `Report for run_id ${runId} not found` : 'No reports available',
        }),
        { status: 404 }
      );
    }

    // Load the quality report
    const qualityReport = await minioClient.getObject(bucket, targetReport.name);

    // Also list available episodes
    const episodes = await minioClient.listObjects(bucket, 'data/');

    return NextResponse.json(
      createApiResponse(true, 'minio', {
        data: {
          report: {
            file: targetReport.name,
            lastModified: targetReport.lastModified,
            size: targetReport.size,
            data: qualityReport,
          },
          summary: {
            totalReports: reports.length,
            filteredReports: filteredReports.length,
            totalEpisodes: episodes.length,
          },
          availableReports: filteredReports.map(r => ({
            name: r.name,
            lastModified: r.lastModified,
            size: r.size,
          })).slice(0, 10),
        }
      })
    );

  } catch (error) {
    console.error('[L1 Quality Report API] Error:', error);
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Failed to retrieve L1 quality report',
        message: error instanceof Error ? error.message : 'Unknown error',
      }),
      { status: 500 }
    );
  }
});
