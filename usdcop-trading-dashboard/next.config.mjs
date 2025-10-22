/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    // Disable ESLint during production builds
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Ignore TypeScript errors during builds
    ignoreBuildErrors: true,
  },
  output: 'standalone',
  // Enable React strict mode for better debugging
  reactStrictMode: true,
};

export default nextConfig;
