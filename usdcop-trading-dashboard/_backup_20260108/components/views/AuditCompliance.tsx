'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield, CheckCircle, AlertTriangle, Flag, Lock, Archive,
  Key, Clock, BarChart3, Gavel, Loader2
} from 'lucide-react';
import { useAuditCompliance } from '@/hooks/compliance/useAuditCompliance';
import {
  ComplianceMetricCard,
  TraceabilityPanel,
  RegulatoryStatusPanel,
  AuditHistoryPanel
} from '@/components/compliance';

export default function AuditCompliance() {
  const { auditData, isLoading, error } = useAuditCompliance();
  const [selectedTab, setSelectedTab] = useState<'overview' | 'traceability' | 'regulatory' | 'security'>('overview');

  if (isLoading && !auditData) {
    return (
      <div className="w-full bg-fintech-dark-950 p-6 flex items-center justify-center min-h-screen">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-12 h-12 text-fintech-cyan-400 animate-spin" />
          <p className="text-fintech-dark-300">Loading audit & compliance metrics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full bg-fintech-dark-950 p-6 flex items-center justify-center min-h-screen">
        <div className="glass-surface p-8 rounded-xl border border-market-down max-w-md">
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="w-8 h-8 text-market-down" />
            <h2 className="text-xl font-bold text-white">Failed to Load Compliance Data</h2>
          </div>
          <p className="text-fintech-dark-300 mb-4">
            Unable to connect to the Compliance API at <code className="text-fintech-cyan-400">/api/compliance/audit-metrics</code>
          </p>
          <div className="text-sm text-fintech-dark-400">
            Please ensure the Compliance API service is running on port 8003.
          </div>
        </div>
      </div>
    );
  }

  const complianceMetrics = [
    {
      title: 'SFC Compliance',
      value: auditData.compliance.sfc.complianceScore,
      subtitle: 'Daily reporting current',
      status: auditData.compliance.sfc.complianceScore >= 95 ? 'compliant' as const : 'warning' as const,
      icon: Flag,
      format: 'percentage' as const
    },
    {
      title: 'Basel III Status',
      value: auditData.compliance.basel.status,
      subtitle: `Adequacy ratio ${auditData.compliance.basel.adequacyRatio.toFixed(3)}`,
      status: auditData.compliance.basel.status === 'compliant' ? 'compliant' as const : 'non-compliant' as const,
      icon: Shield,
      format: 'text' as const
    },
    {
      title: 'Trade Reconstruction',
      value: auditData.compliance.tradeReconstruction.coverage,
      subtitle: `${auditData.compliance.tradeReconstruction.totalTrades} total trades`,
      status: auditData.compliance.tradeReconstruction.coverage === 100 ? 'compliant' as const : 'warning' as const,
      icon: Archive,
      format: 'percentage' as const
    },
    {
      title: 'Data Encryption',
      value: 'AES-256',
      subtitle: 'At rest & in transit',
      status: 'compliant' as const,
      icon: Lock,
      format: 'text' as const
    }
  ];

  const securityMetrics = [
    {
      title: 'MFA Compliance',
      value: auditData.securityCompliance.accessControl.mfaCompliance,
      subtitle: 'All privileged users',
      status: auditData.securityCompliance.accessControl.mfaCompliance === 100 ? 'compliant' as const : 'warning' as const,
      icon: Key,
      format: 'percentage' as const
    },
    {
      title: 'Critical Vulnerabilities',
      value: auditData.securityCompliance.vulnerabilityStatus.critical,
      subtitle: 'Last scan 24h ago',
      status: auditData.securityCompliance.vulnerabilityStatus.critical === 0 ? 'compliant' as const : 'non-compliant' as const,
      icon: Shield,
      format: 'count' as const
    },
    {
      title: 'Log Retention',
      value: auditData.compliance.logging.logRetention,
      subtitle: 'SOX compliant storage',
      status: auditData.compliance.logging.logRetention >= 2555 ? 'compliant' as const : 'warning' as const,
      icon: Archive,
      format: 'count' as const
    },
    {
      title: 'Audit Trail',
      value: auditData.compliance.logging.auditTrailComplete,
      subtitle: 'Microsecond timestamps',
      status: auditData.compliance.logging.auditTrailComplete === 100 ? 'compliant' as const : 'warning' as const,
      icon: Clock,
      format: 'percentage' as const
    }
  ];

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'traceability', label: 'Traceability', icon: Shield },
    { id: 'regulatory', label: 'Regulatory', icon: Gavel },
    { id: 'security', label: 'Security', icon: Shield }
  ];

  return (
    <div className="min-h-screen bg-fintech-dark-950 p-6">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Audit & Compliance</h1>
            <p className="text-fintech-dark-300">
              Complete Traceability • Regulatory Compliance • Model Governance • Security Audits
            </p>
          </div>

          <div className="flex items-center gap-4">
            <div className="glass-surface px-4 py-2 rounded-xl border border-market-up shadow-market-up">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-market-up" />
                <span className="text-market-up font-medium">Audit Ready</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-8"
      >
        <div className="flex items-center gap-2 bg-fintech-dark-800 rounded-xl p-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                selectedTab === tab.id
                  ? 'bg-fintech-cyan-500 text-white shadow-glow-cyan'
                  : 'text-fintech-dark-400 hover:text-white'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>
      </motion.div>

      <AnimatePresence mode="wait">
        {selectedTab === 'overview' && (
          <motion.div
            key="overview"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
          >
            <div className="mb-8">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                <CheckCircle className="w-6 h-6 text-market-up" />
                Compliance Overview
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {complianceMetrics.map((metric, index) => (
                  <motion.div
                    key={metric.title}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <ComplianceMetricCard {...metric} />
                  </motion.div>
                ))}
              </div>
            </div>

            <div className="mb-8">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                <Shield className="w-6 h-6 text-fintech-cyan-500" />
                Security Compliance
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {securityMetrics.map((metric, index) => (
                  <motion.div
                    key={metric.title}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 + index * 0.05 }}
                  >
                    <ComplianceMetricCard {...metric} />
                  </motion.div>
                ))}
              </div>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <AuditHistoryPanel auditHistory={auditData.auditHistory} />
            </motion.div>
          </motion.div>
        )}

        {selectedTab === 'traceability' && (
          <motion.div
            key="traceability"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
          >
            <TraceabilityPanel traceability={auditData.traceability} />
          </motion.div>
        )}

        {selectedTab === 'regulatory' && (
          <motion.div
            key="regulatory"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
          >
            <RegulatoryStatusPanel compliance={auditData.compliance} />
          </motion.div>
        )}

        {selectedTab === 'security' && (
          <motion.div
            key="security"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="space-y-6"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {securityMetrics.map((metric, index) => (
                <motion.div
                  key={metric.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <ComplianceMetricCard {...metric} />
                </motion.div>
              ))}
            </div>

            <div className="glass-surface p-6 rounded-xl">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5 text-fintech-cyan-500" />
                Security Assessment Details
              </h3>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-sm font-semibold text-white mb-3">Access Control</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-fintech-dark-300">Active Users:</span>
                      <span className="text-white font-medium">{auditData.securityCompliance.accessControl.activeUsers}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-fintech-dark-300">Privileged Users:</span>
                      <span className="text-white font-medium">{auditData.securityCompliance.accessControl.privilegedUsers}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-fintech-dark-300">Password Compliance:</span>
                      <span className="text-market-up font-medium">{auditData.securityCompliance.accessControl.passwordCompliance}%</span>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-white mb-3">Vulnerability Status</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-fintech-dark-300">Critical:</span>
                      <span className={`font-medium ${auditData.securityCompliance.vulnerabilityStatus.critical === 0 ? 'text-market-up' : 'text-market-down'}`}>
                        {auditData.securityCompliance.vulnerabilityStatus.critical}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-fintech-dark-300">High:</span>
                      <span className="text-fintech-purple-400 font-medium">{auditData.securityCompliance.vulnerabilityStatus.high}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-fintech-dark-300">Medium:</span>
                      <span className="text-fintech-cyan-400 font-medium">{auditData.securityCompliance.vulnerabilityStatus.medium}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-fintech-dark-300">Low:</span>
                      <span className="text-market-up font-medium">{auditData.securityCompliance.vulnerabilityStatus.low}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
