/**
 * User Profile Service - SignalBridge Integration
 * ================================================
 *
 * Thin client for the current user's profile, backed by SignalBridge
 * `GET/PATCH /api/users/me` (relayed via `/api/execution/users/me`). The
 * backend only supports `{ name, email }`.
 */

import { api } from './api';

const API_BASE = '/users';

export interface UserProfile {
  id: string;
  email: string;
  name: string;
  role?: string;
  is_active?: boolean;
  is_verified?: boolean;
  created_at?: string;
  last_login?: string | null;
}

export interface UserProfileUpdate {
  name?: string;
  email?: string;
}

export const userProfileService = {
  /** Get the current user's profile. */
  async getProfile(): Promise<UserProfile> {
    const { data } = await api.get<UserProfile>(`${API_BASE}/me`);
    return data;
  },

  /** Update the current user's profile (name/email). */
  async updateProfile(update: UserProfileUpdate): Promise<UserProfile> {
    const { data } = await api.patch<UserProfile>(`${API_BASE}/me`, update);
    return data;
  },
};

export default userProfileService;
