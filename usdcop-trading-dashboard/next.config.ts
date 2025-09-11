import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    ignoreBuildErrors: true,
  },
  
  // Performance optimizations for large datasets
  experimental: {
    // Enable modern bundle optimization
    optimizeCss: true,
    // Improve cold start times
    webVitalsAttribution: ['CLS', 'LCP', 'FCP'],
    // Optimize for trading dashboard
    optimizePackageImports: ['recharts', 'framer-motion', 'lucide-react'],
  },

  // External packages for server components
  serverExternalPackages: ['lightweight-charts', 'd3'],

  // Bundle optimization
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Optimize bundle splitting for large datasets
    if (!dev && !isServer) {
      config.optimization = {
        ...config.optimization,
        splitChunks: {
          ...config.optimization.splitChunks,
          cacheGroups: {
            ...config.optimization.splitChunks.cacheGroups,
            // Separate chunk for chart libraries
            charts: {
              name: 'charts',
              test: /[\\/]node_modules[\\/](lightweight-charts|recharts|d3)[\\/]/,
              chunks: 'all',
              priority: 30,
            },
            // Separate chunk for animation libraries
            animations: {
              name: 'animations',
              test: /[\\/]node_modules[\\/](framer-motion|lottie-react)[\\/]/,
              chunks: 'all',
              priority: 25,
            },
            // Separate chunk for UI libraries
            ui: {
              name: 'ui',
              test: /[\\/]node_modules[\\/](@radix-ui|lucide-react)[\\/]/,
              chunks: 'all',
              priority: 20,
            },
            // Common vendor chunk
            vendor: {
              name: 'vendor',
              test: /[\\/]node_modules[\\/]/,
              chunks: 'all',
              priority: 10,
            },
          },
        },
      };

      // Tree shaking optimization
      config.optimization.usedExports = true;
      config.optimization.sideEffects = false;
    }

    // Optimize for large data processing
    config.resolve.alias = {
      ...config.resolve.alias,
      // Use optimized builds when available
      'date-fns': 'date-fns/esm',
    };

    // Performance plugins
    config.plugins.push(
      new webpack.DefinePlugin({
        'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
      })
    );

    return config;
  },

  // Optimize images and static assets
  images: {
    formats: ['image/webp', 'image/avif'],
    minimumCacheTTL: 31536000, // 1 year
  },

  // Compress responses
  compress: true,

  // Enable static optimization
  trailingSlash: false,
  
  // PoweredBy header removal for security
  poweredByHeader: false,

  // Output configuration for deployment
  output: 'standalone',
  
  // Enable gzip compression
  compress: true,
  
  // SWC minification is enabled by default in Next.js 15
  
  // Environment-specific configurations
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },
  
  // Security headers
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
    ];
  },
  
  // Redirects for SEO and performance
  async redirects() {
    return [
      {
        source: '/dashboard',
        destination: '/',
        permanent: true,
      },
    ];
  },
};

export default nextConfig;
