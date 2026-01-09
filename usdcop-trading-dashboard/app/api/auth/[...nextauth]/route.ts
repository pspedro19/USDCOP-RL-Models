/**
 * NextAuth.js API Route Handler
 * =============================
 *
 * Handles all authentication requests:
 * - POST /api/auth/signin
 * - POST /api/auth/signout
 * - GET /api/auth/session
 * - GET /api/auth/csrf
 * - GET /api/auth/providers
 */

import NextAuth from 'next-auth';
import { authOptions } from '@/lib/auth/next-auth-options';

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };
