/**
 * NextAuth.js Configuration
 * =========================
 *
 * Centralized NextAuth configuration following best practices.
 * Uses Credentials Provider with our custom authentication service.
 */

import type { NextAuthOptions, Session, User as NextAuthUser } from 'next-auth';
import type { JWT } from 'next-auth/jwt';
import CredentialsProvider from 'next-auth/providers/credentials';
import { authService } from './auth-service';
import type { SessionUser, UserRole } from './types';

// ============================================================================
// Type Extensions
// ============================================================================

declare module 'next-auth' {
  interface Session {
    user: SessionUser;
    accessToken?: string;
  }

  interface User extends SessionUser {}
}

declare module 'next-auth/jwt' {
  interface JWT extends SessionUser {
    accessToken?: string;
  }
}

// ============================================================================
// NextAuth Options
// ============================================================================

export const authOptions: NextAuthOptions = {
  // Session configuration
  session: {
    strategy: 'jwt',
    maxAge: 24 * 60 * 60, // 24 hours
    updateAge: 60 * 60, // Update session every hour
  },

  // JWT configuration
  jwt: {
    maxAge: 24 * 60 * 60, // 24 hours
  },

  // Cookie configuration - allow HTTP for local development
  useSecureCookies: false,
  cookies: {
    sessionToken: {
      name: `next-auth.session-token`,
      options: {
        httpOnly: true,
        sameSite: 'lax',
        path: '/',
        secure: false,
      },
    },
    callbackUrl: {
      name: `next-auth.callback-url`,
      options: {
        sameSite: 'lax',
        path: '/',
        secure: false,
      },
    },
    csrfToken: {
      name: `next-auth.csrf-token`,
      options: {
        httpOnly: true,
        sameSite: 'lax',
        path: '/',
        secure: false,
      },
    },
  },

  // Pages
  pages: {
    signIn: '/login',
    signOut: '/login',
    error: '/login',
  },

  // Providers
  providers: [
    CredentialsProvider({
      id: 'credentials',
      name: 'Credentials',
      credentials: {
        identifier: {
          label: 'Email or Username',
          type: 'text',
          placeholder: 'admin or admin@example.com',
        },
        password: {
          label: 'Password',
          type: 'password',
        },
      },

      async authorize(credentials, req) {
        if (!credentials?.identifier || !credentials?.password) {
          throw new Error('Email/username and password are required');
        }

        // Get client IP (for rate limiting)
        const forwarded = req?.headers?.['x-forwarded-for'];
        const ip = typeof forwarded === 'string'
          ? forwarded.split(',')[0]
          : req?.headers?.['x-real-ip'] || '127.0.0.1';

        // Get user agent for audit logging
        const userAgent = req?.headers?.['user-agent'];

        // Authenticate using our service
        const result = await authService.login(
          {
            identifier: credentials.identifier,
            password: credentials.password,
          },
          ip as string,
          userAgent as string
        );

        if (!result.success) {
          throw new Error(result.error || 'Authentication failed');
        }

        if (result.requiresTwoFactor) {
          // TODO: Handle 2FA flow
          throw new Error('Two-factor authentication required');
        }

        if (!result.user) {
          throw new Error('Authentication failed');
        }

        // Return user for session
        return {
          id: result.user.id,
          email: result.user.email,
          username: result.user.username,
          role: result.user.role,
          fullName: result.user.fullName,
          avatarUrl: result.user.avatarUrl,
        } as NextAuthUser;
      },
    }),
  ],

  // Callbacks
  callbacks: {
    /**
     * JWT callback - called when JWT is created or updated
     */
    async jwt({ token, user, trigger, session }) {
      // Initial sign in
      if (user) {
        token.id = user.id;
        token.email = user.email;
        token.username = (user as SessionUser).username;
        token.role = (user as SessionUser).role;
        token.fullName = (user as SessionUser).fullName;
        token.avatarUrl = (user as SessionUser).avatarUrl;
      }

      // Handle session update
      if (trigger === 'update' && session) {
        token.fullName = session.user?.fullName;
        token.avatarUrl = session.user?.avatarUrl;
      }

      return token;
    },

    /**
     * Session callback - called whenever session is checked
     */
    async session({ session, token }) {
      if (token) {
        session.user = {
          id: token.id as string,
          email: token.email as string,
          username: token.username as string,
          role: token.role as UserRole,
          fullName: token.fullName as string | undefined,
          avatarUrl: token.avatarUrl as string | undefined,
        };
      }

      return session;
    },

    /**
     * Sign in callback - called after successful authentication
     */
    async signIn({ user, account, profile }) {
      // Additional validation if needed
      return true;
    },

    /**
     * Redirect callback - customize redirect behavior
     */
    async redirect({ url, baseUrl }) {
      // Redirect to dashboard after login
      if (url.startsWith(baseUrl)) {
        return url;
      }
      return baseUrl;
    },
  },

  // Events
  events: {
    async signIn({ user, account, isNewUser }) {
      console.log('[NextAuth] User signed in:', user.email);
    },

    async signOut({ token }) {
      console.log('[NextAuth] User signed out:', token?.email);
    },

    async session({ session, token }) {
      // Session accessed
    },
  },

  // Debug mode in development
  debug: process.env.NODE_ENV === 'development',

  // Secret (required in production)
  secret: process.env.NEXTAUTH_SECRET,
};

export default authOptions;
