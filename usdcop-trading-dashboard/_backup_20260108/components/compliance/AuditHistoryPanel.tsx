import React from 'react';
import { motion } from 'framer-motion';
import { FileSearch } from 'lucide-react';

interface AuditEntry {
  id: string;
  type: string;
  date: string;
  auditor: string;
  status: string;
  findings: number;
  severity: string;
}

interface AuditHistoryPanelProps {
  auditHistory: AuditEntry[];
}

export const AuditHistoryPanel: React.FC<AuditHistoryPanelProps> = ({ auditHistory }) => {
  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'none': return 'text-market-up';
      case 'low': return 'text-fintech-cyan-400';
      case 'medium': return 'text-fintech-purple-400';
      case 'high': return 'text-market-down';
      case 'critical': return 'text-red-600';
      default: return 'text-fintech-dark-400';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'passed': return 'text-market-up';
      case 'completed': return 'text-market-up';
      case 'in progress': return 'text-fintech-purple-400';
      case 'failed': return 'text-market-down';
      default: return 'text-fintech-dark-400';
    }
  };

  return (
    <div className="glass-surface p-6 rounded-xl">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <FileSearch className="w-5 h-5 text-fintech-cyan-500" />
        Audit History
      </h3>

      <div className="space-y-4">
        {auditHistory.map((audit) => (
          <motion.div
            key={audit.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 bg-fintech-dark-800/30 rounded-lg border border-fintech-dark-700"
          >
            <div className="flex items-center justify-between mb-2">
              <div>
                <h4 className="text-sm font-semibold text-white">{audit.type}</h4>
                <p className="text-xs text-fintech-dark-300">by {audit.auditor}</p>
              </div>
              <div className="text-right">
                <div className={`text-sm font-medium ${getStatusColor(audit.status)}`}>
                  {audit.status}
                </div>
                <div className="text-xs text-fintech-dark-400">
                  {new Date(audit.date).toLocaleDateString()}
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-4">
                <span className="text-fintech-dark-300">
                  Findings: <span className="text-white font-medium">{audit.findings}</span>
                </span>
                <span className="text-fintech-dark-300">
                  Severity: <span className={`font-medium ${getSeverityColor(audit.severity)}`}>{audit.severity}</span>
                </span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};
