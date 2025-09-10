'use client';

import React from 'react';
import ModelPerformanceDashboard from '@/components/ml-analytics/ModelPerformanceDashboard';

export default function MLAnalyticsPage() {
  return (
    <div className="container mx-auto p-6 space-y-6">
      <ModelPerformanceDashboard />
    </div>
  );
}