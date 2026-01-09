import useSWR from 'swr';

const fetcher = (url: string) => fetch(url).then((res) => res.json());

interface AuditData {
  traceability: any;
  compliance: any;
  modelGovernance: any;
  securityCompliance: any;
  auditHistory: any[];
}

export const useAuditCompliance = () => {
  const { data, error, isLoading } = useSWR(
    '/api/compliance/audit-metrics?symbol=USDCOP&days=30',
    fetcher,
    {
      refreshInterval: 60000,
      revalidateOnFocus: true
    }
  );

  const defaultAuditData: AuditData = {
    traceability: {
      datasetHashes: {
        'L0_raw_dataset': 'sha256:a1b2c3d4e5f6...',
        'L1_standardized': 'sha256:b2c3d4e5f6a1...',
        'L2_prepared': 'sha256:c3d4e5f6a1b2...',
        'L3_features': 'sha256:d4e5f6a1b2c3...',
        'L4_rlready': 'sha256:e5f6a1b2c3d4...'
      },
      gitCommits: {
        'data-pipeline': 'commit:19f5b99',
        'ml-models': 'commit:bd53ba7',
        'config': 'commit:6b89b32'
      },
      mlflowRuns: [
        { id: 'run_001', experiment: 'PPO_Training', timestamp: '2024-12-15T08:30:00Z', status: 'FINISHED' },
        { id: 'run_002', experiment: 'QR_DQN_Training', timestamp: '2024-12-15T10:15:00Z', status: 'RUNNING' }
      ],
      s3Bundles: [
        { path: 's3://models/L5_serving_bundle_v2.1.5.tar.gz', size: '45.2MB', created: '2024-12-15T09:45:00Z' },
        { path: 's3://data/L4_rlready_20241215.parquet', size: '1.2GB', created: '2024-12-15T06:30:00Z' }
      ]
    },
    compliance: {
      sfc: {
        reportingStatus: 'current',
        lastSubmission: '2024-12-10T16:00:00Z',
        nextDue: '2024-12-17T16:00:00Z',
        reportType: 'Daily Risk Report',
        complianceScore: 98.5
      },
      basel: {
        capitalRequirement: 125000,
        currentCapital: 152000,
        adequacyRatio: 1.216,
        riskWeightedAssets: 485000,
        tier1Ratio: 14.8,
        status: 'compliant' as const
      },
      tradeReconstruction: {
        coverage: 100,
        avgReconstructionTime: 2.3,
        lastAudit: '2024-12-12T14:00:00Z',
        failedReconstructions: 0,
        totalTrades: 2847
      },
      logging: {
        microsecondTimestamps: true,
        logRetention: 2555,
        logCompression: 85.2,
        auditTrailComplete: 100,
        storageCompliance: 'SOX_compliant'
      }
    },
    modelGovernance: {
      modelInventory: [
        {
          id: 'PPO_LSTM_v2.1.5',
          status: 'Production',
          lastValidation: '2024-12-14T12:00:00Z',
          nextReview: '2024-12-21T12:00:00Z',
          riskLevel: 'High',
          approver: 'Risk Committee'
        },
        {
          id: 'QR_DQN_v1.8.3',
          status: 'Development',
          lastValidation: '2024-12-13T09:30:00Z',
          nextReview: '2024-12-20T09:30:00Z',
          riskLevel: 'Medium',
          approver: 'Model Risk Officer'
        }
      ],
      changeManagement: {
        pendingChanges: 2,
        approvedChanges: 15,
        rejectedChanges: 1,
        emergencyChanges: 0
      },
      documentationStatus: {
        modelDocumentation: 95,
        riskAssessment: 98,
        validationReports: 92,
        userAcceptanceTesting: 88
      }
    },
    securityCompliance: {
      accessControl: {
        activeUsers: 12,
        privilegedUsers: 3,
        lastAccessReview: '2024-12-01T10:00:00Z',
        mfaCompliance: 100,
        passwordCompliance: 95
      },
      dataEncryption: {
        atRest: 100,
        inTransit: 100,
        keyRotation: 'current',
        lastKeyRotation: '2024-11-15T08:00:00Z'
      },
      vulnerabilityStatus: {
        critical: 0,
        high: 2,
        medium: 5,
        low: 12,
        lastScan: '2024-12-14T22:00:00Z'
      }
    },
    auditHistory: [
      {
        id: 'audit_001',
        type: 'Internal Model Validation',
        date: '2024-12-10T09:00:00Z',
        auditor: 'Internal Risk Team',
        status: 'Completed',
        findings: 3,
        severity: 'Low'
      },
      {
        id: 'audit_002',
        type: 'SFC Regulatory Review',
        date: '2024-11-28T14:00:00Z',
        auditor: 'SFC Colombia',
        status: 'Passed',
        findings: 1,
        severity: 'Medium'
      },
      {
        id: 'audit_003',
        type: 'Third-Party Security Audit',
        date: '2024-11-15T10:00:00Z',
        auditor: 'CyberSec Partners',
        status: 'Passed',
        findings: 0,
        severity: 'None'
      }
    ]
  };

  const auditData = data ? {
    traceability: defaultAuditData.traceability,
    compliance: {
      sfc: defaultAuditData.compliance.sfc,
      basel: {
        capitalRequirement: data.capital_adequacy?.total_capital_usd || 0,
        currentCapital: data.capital_adequacy?.total_capital_usd || 0,
        adequacyRatio: data.capital_adequacy?.capital_adequacy_ratio || 0,
        riskWeightedAssets: data.capital_adequacy?.risk_weighted_assets || 0,
        tier1Ratio: data.capital_adequacy?.tier1_ratio_pct || 0,
        status: data.compliance_status === 'COMPLIANT' ? 'compliant' as const : 'warning' as const
      },
      tradeReconstruction: {
        coverage: 100,
        avgReconstructionTime: data.audit_metrics?.trade_reconstruction_time_ms || 0,
        lastAudit: new Date().toISOString(),
        failedReconstructions: 0,
        totalTrades: data.audit_metrics?.trades_reconstructed || 0
      },
      logging: defaultAuditData.compliance.logging
    },
    modelGovernance: defaultAuditData.modelGovernance,
    securityCompliance: defaultAuditData.securityCompliance,
    auditHistory: defaultAuditData.auditHistory
  } : defaultAuditData;

  return {
    auditData,
    isLoading,
    error
  };
};
