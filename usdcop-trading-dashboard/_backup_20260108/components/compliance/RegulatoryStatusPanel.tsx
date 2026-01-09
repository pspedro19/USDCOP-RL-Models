import React from 'react';
import { Gavel, Flag, Shield, Archive } from 'lucide-react';

interface RegulatoryStatusPanelProps {
  compliance: {
    sfc: {
      reportingStatus: string;
      nextDue: string;
      complianceScore: number;
    };
    basel: {
      adequacyRatio: number;
      tier1Ratio: number;
      status: string;
    };
    tradeReconstruction: {
      coverage: number;
      avgReconstructionTime: number;
      failedReconstructions: number;
    };
  };
}

export const RegulatoryStatusPanel: React.FC<RegulatoryStatusPanelProps> = ({ compliance }) => {
  const calculateDaysUntilDue = (dueDate: string) => {
    const due = new Date(dueDate);
    const now = new Date();
    const diffTime = due.getTime() - now.getTime();
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  };

  return (
    <div className="glass-surface p-6 rounded-xl">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Gavel className="w-5 h-5 text-fintech-cyan-500" />
        Regulatory Compliance
      </h3>

      <div className="space-y-6">
        <div>
          <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Flag className="w-4 h-4 text-yellow-400" />
            SFC Colombia Reporting
          </h4>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-fintech-dark-300">Status:</span>
              <span className={`font-medium ${
                compliance.sfc.reportingStatus === 'current' ? 'text-market-up' : 'text-market-down'
              }`}>
                {compliance.sfc.reportingStatus.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-fintech-dark-300">Next Due:</span>
              <span className="text-white font-medium">
                {calculateDaysUntilDue(compliance.sfc.nextDue)} days
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-fintech-dark-300">Compliance Score:</span>
              <span className="text-market-up font-bold">{compliance.sfc.complianceScore.toFixed(1)}%</span>
            </div>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Shield className="w-4 h-4 text-blue-400" />
            Basel III Capital Requirements
          </h4>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-fintech-dark-300">Capital Adequacy:</span>
              <span className={`font-bold ${
                compliance.basel.adequacyRatio >= 1.0 ? 'text-market-up' : 'text-market-down'
              }`}>
                {compliance.basel.adequacyRatio.toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-fintech-dark-300">Tier 1 Ratio:</span>
              <span className="text-market-up font-medium">{compliance.basel.tier1Ratio.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-fintech-dark-300">Status:</span>
              <span className="text-market-up font-medium">{compliance.basel.status.toUpperCase()}</span>
            </div>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Archive className="w-4 h-4 text-green-400" />
            Trade Reconstruction
          </h4>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-fintech-dark-300">Coverage:</span>
              <span className="text-market-up font-bold">{compliance.tradeReconstruction.coverage}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-fintech-dark-300">Avg Reconstruction:</span>
              <span className="text-white font-medium">{compliance.tradeReconstruction.avgReconstructionTime.toFixed(1)}s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-fintech-dark-300">Failed Reconstructions:</span>
              <span className={`font-medium ${
                compliance.tradeReconstruction.failedReconstructions === 0 ? 'text-market-up' : 'text-market-down'
              }`}>
                {compliance.tradeReconstruction.failedReconstructions}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
