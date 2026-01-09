/**
 * Next.js Middleware
 * ==================
 *
 * Handles authentication and route protection at the edge.
 * Runs before every request to protected routes.
 */

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getToken } from 'next-auth/jwt';

// ============================================================================
// Route Configuration
// ============================================================================

// Routes that don't require authentication
const PUBLIC_ROUTES = [
  '/',           // Landing page is public
  '/login',
  '/api/auth',
  '/api/health',
  '/api/proxy/trading/health',  // Health checks don't need auth
  '/_next',
  '/favicon.ico',
  '/images',
  '/fonts',
];

// Routes that require admin role
const ADMIN_ROUTES = [
  '/admin',
  '/api/admin',
  '/api/users',
];

// API routes that require authentication
const PROTECTED_API_ROUTES = [
  '/api/trading',
  '/api/signals',
  '/api/backtest',
  '/api/pipeline',
  '/api/agent',
];

// ============================================================================
// Middleware
// ============================================================================

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Skip middleware for public routes
  if (isPublicRoute(pathname)) {
    return NextResponse.next();
  }

  // DEV MODE: Skip auth entirely when NEXT_PUBLIC_DEV_MODE is true
  // This allows the app to work without backend authentication
  if (process.env.NEXT_PUBLIC_DEV_MODE === 'true') {
    const response = NextResponse.next();
    addSecurityHeaders(response);
    return response;
  }

  // Get JWT token - explicitly specify cookie name for HTTP mode
  const token = await getToken({
    req: request,
    secret: process.env.NEXTAUTH_SECRET,
    cookieName: 'next-auth.session-token',
  });

  // Check if route requires authentication
  const isProtectedPage = !isPublicRoute(pathname) && !pathname.startsWith('/api');
  const isProtectedApi = PROTECTED_API_ROUTES.some(route => pathname.startsWith(route));
  const isAdminRoute = ADMIN_ROUTES.some(route => pathname.startsWith(route));

  // Redirect to login if not authenticated
  if ((isProtectedPage || isProtectedApi) && !token) {
    // For API routes, return 401
    if (pathname.startsWith('/api')) {
      return NextResponse.json(
        {
          error: 'Authentication required',
          message: 'Please log in to access this resource',
          timestamp: new Date().toISOString(),
        },
        { status: 401 }
      );
    }

    // For pages, redirect to login
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('callbackUrl', pathname);
    return NextResponse.redirect(loginUrl);
  }

  // Check admin authorization
  if (isAdminRoute && token) {
    const role = token.role as string;
    if (role !== 'admin') {
      if (pathname.startsWith('/api')) {
        return NextResponse.json(
          {
            error: 'Forbidden',
            message: 'Admin access required',
            timestamp: new Date().toISOString(),
          },
          { status: 403 }
        );
      }

      // Redirect non-admin to hub
      return NextResponse.redirect(new URL('/hub', request.url));
    }
  }

  // Add security headers
  const response = NextResponse.next();
  addSecurityHeaders(response);

  // Add user info to request headers for API routes
  if (token && pathname.startsWith('/api')) {
    response.headers.set('x-user-id', token.id as string);
    response.headers.set('x-user-role', token.role as string);
  }

  return response;
}

// ============================================================================
// Helper Functions
// ============================================================================

function isPublicRoute(pathname: string): boolean {
  // Check for exact match first (for '/')
  if (PUBLIC_ROUTES.includes(pathname)) {
    return true;
  }
  // Then check for prefix match (excluding '/')
  return PUBLIC_ROUTES.filter(r => r !== '/').some(route => pathname.startsWith(route));
}

function addSecurityHeaders(response: NextResponse): void {
  // Prevent clickjacking
  response.headers.set('X-Frame-Options', 'DENY');

  // Prevent MIME type sniffing
  response.headers.set('X-Content-Type-Options', 'nosniff');

  // XSS protection
  response.headers.set('X-XSS-Protection', '1; mode=block');

  // Referrer policy
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');

  // Permissions policy
  response.headers.set(
    'Permissions-Policy',
    'camera=(), microphone=(), geolocation=(), interest-cohort=()'
  );
}

// ============================================================================
// Middleware Config
// ============================================================================

export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
};
