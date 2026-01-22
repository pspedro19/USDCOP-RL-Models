/**
 * Auth Contract - SignalBridge Integration
 * =========================================
 *
 * SSOT for authentication in the execution module.
 * Mirrors backend: services/signalbridge_api/app/contracts/auth.py
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { z } from 'zod';

// ============================================================================
// SUBSCRIPTION TIERS (SSOT)
// ============================================================================

export const SUBSCRIPTION_TIERS = ['free', 'pro', 'enterprise'] as const;
export type SubscriptionTier = typeof SUBSCRIPTION_TIERS[number];

export const SUBSCRIPTION_TIER_LIMITS: Record<SubscriptionTier, {
  maxDailyTrades: number;
  maxExchanges: number;
  maxPositionSizePct: number;
}> = {
  free: { maxDailyTrades: 10, maxExchanges: 1, maxPositionSizePct: 0.01 },
  pro: { maxDailyTrades: 100, maxExchanges: 3, maxPositionSizePct: 0.05 },
  enterprise: { maxDailyTrades: 500, maxExchanges: 10, maxPositionSizePct: 0.10 },
};

// ============================================================================
// RISK PROFILES (SSOT)
// ============================================================================

export const RISK_PROFILES = ['conservative', 'moderate', 'aggressive'] as const;
export type RiskProfile = typeof RISK_PROFILES[number];

export const RISK_PROFILE_MULTIPLIERS: Record<RiskProfile, number> = {
  conservative: 0.5,
  moderate: 1.0,
  aggressive: 1.5,
};

// ============================================================================
// AUTH SCHEMAS
// ============================================================================

/**
 * Login request
 */
export const LoginRequestSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
});
export type LoginRequest = z.infer<typeof LoginRequestSchema>;

/**
 * Register request with password validation
 */
export const RegisterRequestSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[0-9]/, 'Password must contain at least one number'),
  confirmPassword: z.string(),
  acceptTerms: z.boolean().refine(v => v === true, 'You must accept the terms and conditions'),
}).refine(data => data.password === data.confirmPassword, {
  message: 'Passwords do not match',
  path: ['confirmPassword'],
});
export type RegisterRequest = z.infer<typeof RegisterRequestSchema>;

/**
 * Forgot password request
 */
export const ForgotPasswordRequestSchema = z.object({
  email: z.string().email('Invalid email address'),
});
export type ForgotPasswordRequest = z.infer<typeof ForgotPasswordRequestSchema>;

/**
 * Reset password request
 */
export const ResetPasswordRequestSchema = z.object({
  token: z.string().min(1, 'Token is required'),
  password: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[0-9]/, 'Password must contain at least one number'),
  confirmPassword: z.string(),
}).refine(data => data.password === data.confirmPassword, {
  message: 'Passwords do not match',
  path: ['confirmPassword'],
});
export type ResetPasswordRequest = z.infer<typeof ResetPasswordRequestSchema>;

/**
 * Auth token response
 */
export const AuthTokenSchema = z.object({
  access_token: z.string(),
  token_type: z.literal('bearer'),
  expires_at: z.string().datetime(),
  refresh_token: z.string().optional(),
});
export type AuthToken = z.infer<typeof AuthTokenSchema>;

/**
 * User schema
 */
export const UserSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  subscription_tier: z.enum(SUBSCRIPTION_TIERS),
  risk_profile: z.enum(RISK_PROFILES),
  max_daily_trades: z.number().int().min(1).max(500),
  max_position_size_pct: z.number().min(0.001).max(0.1),
  connected_exchanges: z.array(z.string()),
  is_active: z.boolean(),
  created_at: z.string().datetime(),
});
export type User = z.infer<typeof UserSchema>;

/**
 * Auth response (login/register success)
 */
export const AuthResponseSchema = z.object({
  user: UserSchema,
  tokens: AuthTokenSchema,
});
export type AuthResponse = z.infer<typeof AuthResponseSchema>;

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

export const validateLoginRequest = (data: unknown) =>
  LoginRequestSchema.safeParse(data);

export const validateRegisterRequest = (data: unknown) =>
  RegisterRequestSchema.safeParse(data);

export const validateUser = (data: unknown) =>
  UserSchema.safeParse(data);

export const validateAuthToken = (data: unknown) =>
  AuthTokenSchema.safeParse(data);

export const validateAuthResponse = (data: unknown) =>
  AuthResponseSchema.safeParse(data);

/**
 * Check if user can connect more exchanges
 */
export function canConnectMoreExchanges(user: User): boolean {
  const limits = SUBSCRIPTION_TIER_LIMITS[user.subscription_tier];
  return user.connected_exchanges.length < limits.maxExchanges;
}

/**
 * Get user's daily trade limit
 */
export function getDailyTradeLimit(user: User): number {
  return SUBSCRIPTION_TIER_LIMITS[user.subscription_tier].maxDailyTrades;
}

/**
 * Get user's max position size percentage
 */
export function getMaxPositionSizePct(user: User): number {
  const baseLimit = SUBSCRIPTION_TIER_LIMITS[user.subscription_tier].maxPositionSizePct;
  const riskMultiplier = RISK_PROFILE_MULTIPLIERS[user.risk_profile];
  return Math.min(baseLimit * riskMultiplier, 0.1); // Cap at 10%
}
