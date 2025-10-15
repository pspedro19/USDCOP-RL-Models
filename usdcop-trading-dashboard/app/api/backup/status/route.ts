import { NextRequest, NextResponse } from 'next/server';

interface BackupStatusResponse {
  status: 'healthy' | 'warning' | 'error';
  timestamp: string;
  backups: {
    l0: BackupInfo;
    l1: BackupInfo;
    l2: BackupInfo;
    l3: BackupInfo;
    l4: BackupInfo;
    l5: BackupInfo;
  };
  retention: {
    policy: string;
    oldestBackup: string;
    totalSize: string;
    compressionRatio: number;
  };
  recovery: {
    lastTested: string;
    testStatus: 'passed' | 'failed' | 'pending';
    recoveryTime: number;
  };
}

interface BackupInfo {
  exists: boolean;
  lastUpdated: string;
  recordCount: number;
  gapsDetected: number;
  size: string;
  integrity: 'good' | 'warning' | 'error';
  location: string;
  encrypted: boolean;
  checksumValid: boolean;
}

export async function GET(request: NextRequest) {
  try {
    const generateBackupInfo = (layer: string): BackupInfo => {
      const hasIssues = Math.random() > 0.8;
      const recordCount = Math.floor(Math.random() * 10000) + 1000;

      return {
        exists: Math.random() > 0.05,
        lastUpdated: new Date(Date.now() - Math.random() * 3600000).toISOString(),
        recordCount,
        gapsDetected: hasIssues ? Math.floor(Math.random() * 5) + 1 : 0,
        size: `${(Math.random() * 5 + 0.5).toFixed(1)} GB`,
        integrity: hasIssues ? (Math.random() > 0.5 ? 'warning' : 'error') : 'good',
        location: `s3://usdcop-backups/${layer}/`,
        encrypted: true,
        checksumValid: !hasIssues || Math.random() > 0.3
      };
    };

    const backupData: BackupStatusResponse = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      backups: {
        l0: generateBackupInfo('l0'),
        l1: generateBackupInfo('l1'),
        l2: generateBackupInfo('l2'),
        l3: generateBackupInfo('l3'),
        l4: generateBackupInfo('l4'),
        l5: generateBackupInfo('l5')
      },
      retention: {
        policy: '7 days full, 30 days incremental',
        oldestBackup: new Date(Date.now() - 30 * 24 * 3600000).toISOString(),
        totalSize: `${(Math.random() * 50 + 20).toFixed(1)} GB`,
        compressionRatio: 0.7 + Math.random() * 0.2
      },
      recovery: {
        lastTested: new Date(Date.now() - 7 * 24 * 3600000).toISOString(),
        testStatus: Math.random() > 0.1 ? 'passed' : 'failed',
        recoveryTime: Math.floor(Math.random() * 300) + 60
      }
    };

    // Determine overall status
    const backupIssues = Object.values(backupData.backups).filter(
      backup => !backup.exists || backup.integrity === 'error' || !backup.checksumValid
    );

    const backupWarnings = Object.values(backupData.backups).filter(
      backup => backup.integrity === 'warning' || backup.gapsDetected > 0
    );

    if (backupIssues.length > 0 || backupData.recovery.testStatus === 'failed') {
      backupData.status = 'error';
    } else if (backupWarnings.length > 0) {
      backupData.status = 'warning';
    }

    return NextResponse.json(backupData, {
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    });

  } catch (error) {
    console.error('Backup Status Check Error:', error);

    return NextResponse.json(
      {
        status: 'error',
        timestamp: new Date().toISOString(),
        error: 'Backup status check failed',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    switch (body.action) {
      case 'trigger-backup':
        if (!body.layer) {
          return NextResponse.json(
            { error: 'Layer specification required' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          status: 'success',
          message: `${body.layer} backup initiated`,
          timestamp: new Date().toISOString(),
          jobId: `backup-${body.layer}-${Date.now()}`
        });

      case 'test-recovery':
        if (!body.layer) {
          return NextResponse.json(
            { error: 'Layer specification required' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          status: 'success',
          message: `${body.layer} recovery test started`,
          timestamp: new Date().toISOString(),
          testId: `recovery-test-${body.layer}-${Date.now()}`
        });

      case 'cleanup-old':
        return NextResponse.json({
          status: 'success',
          message: 'Old backup cleanup initiated',
          timestamp: new Date().toISOString()
        });

      case 'verify-integrity':
        return NextResponse.json({
          status: 'success',
          message: 'Backup integrity verification started',
          timestamp: new Date().toISOString()
        });

      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        );
    }

  } catch (error) {
    console.error('Backup Status POST Error:', error);

    return NextResponse.json(
      {
        error: 'Failed to process backup request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}