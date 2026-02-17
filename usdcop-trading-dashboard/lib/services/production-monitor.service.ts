/**
 * Production Monitor Service
 * ==========================
 * Client-side service for real-time production model monitoring.
 */

import {
  ProductionModelInfo,
  LiveInferenceState,
  SessionEquityCurve,
  PendingExperimentsSummary,
  ProductionMonitorResponse,
} from '@/lib/contracts/production-monitor.contract';

const API_BASE = '/api/production';

/**
 * Fetch current production model info
 */
export async function fetchProductionModel(): Promise<ProductionModelInfo | null> {
  const response = await fetch(`${API_BASE}/model`);
  if (!response.ok) {
    if (response.status === 404) return null;
    throw new Error(`Failed to fetch production model: ${response.statusText}`);
  }
  const data = await response.json();
  return data.model;
}

/**
 * Fetch live inference state
 */
export async function fetchLiveState(): Promise<LiveInferenceState | null> {
  const response = await fetch(`${API_BASE}/live-state`);
  if (!response.ok) {
    if (response.status === 404) return null;
    throw new Error(`Failed to fetch live state: ${response.statusText}`);
  }
  const data = await response.json();
  return data.state;
}

/**
 * Fetch session equity curve
 */
export async function fetchEquityCurve(): Promise<SessionEquityCurve | null> {
  const response = await fetch(`${API_BASE}/equity-curve`);
  if (!response.ok) {
    if (response.status === 404) return null;
    throw new Error(`Failed to fetch equity curve: ${response.statusText}`);
  }
  const data = await response.json();
  return data.equityCurve;
}

/**
 * Fetch pending experiments summary
 */
export async function fetchPendingSummary(): Promise<PendingExperimentsSummary> {
  const response = await fetch(`${API_BASE}/pending-summary`);
  if (!response.ok) {
    throw new Error(`Failed to fetch pending summary: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetch all production monitor data in one call
 */
export async function fetchProductionMonitorData(): Promise<ProductionMonitorResponse> {
  const response = await fetch(`${API_BASE}/monitor`);
  if (!response.ok) {
    throw new Error(`Failed to fetch monitor data: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Production monitor service singleton
 */
export const productionMonitorService = {
  fetchProductionModel,
  fetchLiveState,
  fetchEquityCurve,
  fetchPendingSummary,
  fetchProductionMonitorData,
};
