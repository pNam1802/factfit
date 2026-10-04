import type { NextConfig } from "next";

// The Python API (FastAPI). `uv run factfit dev` starts it on this port.
const API_URL = process.env.FACTFIT_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // The browser calls /api/... on this server, which forwards to FastAPI:
  // same origin, so no CORS setup is needed.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },
  experimental: {
    // Parsing and matching a long job description can take about a minute;
    // the default proxy timeout (30 s) would cut those requests off.
    proxyTimeout: 120_000,
  },
};

export default nextConfig;
