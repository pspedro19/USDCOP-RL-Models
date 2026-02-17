/**
 * Experiments Service
 * ===================
 * Client-side service for experiment approval workflow.
 */

import {
  Experiment,
  ExperimentsListResponse,
  PendingExperimentsResponse,
  ExperimentDetailResponse,
  ApproveRequest,
  RejectRequest,
  ApproveResponse,
  RejectResponse,
} from '@/lib/contracts/experiments.contract';

const API_BASE = '/api/experiments';

/**
 * Fetch all experiments with optional status filter
 */
export async function fetchExperiments(
  status?: string,
  limit: number = 50
): Promise<ExperimentsListResponse> {
  const params = new URLSearchParams();
  if (status) params.set('status', status);
  params.set('limit', limit.toString());

  const response = await fetch(`${API_BASE}?${params.toString()}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch experiments: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetch pending experiments requiring approval
 */
export async function fetchPendingExperiments(): Promise<PendingExperimentsResponse> {
  const response = await fetch(`${API_BASE}/pending`);
  if (!response.ok) {
    throw new Error(`Failed to fetch pending experiments: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetch single experiment by proposal ID
 */
export async function fetchExperiment(proposalId: string): Promise<Experiment> {
  const response = await fetch(`${API_BASE}/${proposalId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch experiment: ${response.statusText}`);
  }
  const data: ExperimentDetailResponse = await response.json();
  return data.experiment;
}

/**
 * Approve an experiment (second vote - promotes to production)
 */
export async function approveExperiment(
  proposalId: string,
  request: ApproveRequest = {}
): Promise<ApproveResponse> {
  const response = await fetch(`${API_BASE}/${proposalId}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to approve experiment');
  }

  return response.json();
}

/**
 * Reject an experiment
 */
export async function rejectExperiment(
  proposalId: string,
  request: RejectRequest = {}
): Promise<RejectResponse> {
  const response = await fetch(`${API_BASE}/${proposalId}/reject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to reject experiment');
  }

  return response.json();
}

/**
 * Fetch experiment by model ID (pending proposal for specific model)
 */
export async function fetchExperimentByModelId(modelId: string): Promise<Experiment | null> {
  try {
    const response = await fetch(`${API_BASE}/by-model/${modelId}`);
    if (!response.ok) {
      // 401/403 = auth not configured, 404 = no experiment — both expected
      if (response.status === 401 || response.status === 403 || response.status === 404) {
        return null;
      }
      throw new Error(`Failed to fetch experiment for model: ${response.statusText}`);
    }
    const data = await response.json();
    return data.experiment || null;
  } catch (error) {
    // Network errors or auth issues — return null silently
    if (error instanceof TypeError) return null; // fetch failed
    throw error;
  }
}

/**
 * Experiments service singleton
 */
export const experimentsService = {
  fetchExperiments,
  fetchPendingExperiments,
  fetchExperiment,
  fetchExperimentByModelId,
  approveExperiment,
  rejectExperiment,
};
