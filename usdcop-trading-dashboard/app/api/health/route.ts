/**
 * Health check endpoint for Docker container healthcheck
 * Returns 200 OK if the service is running
 */
import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'usdcop-dashboard'
  });
}
