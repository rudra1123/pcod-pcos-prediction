import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server on port 5173. /api is proxied to the Spring Boot backend on
// port 8080, so the UI works with relative URLs in dev. Set VITE_API_URL
// (e.g. http://localhost:8080) to call the backend directly instead.
// The proxy target is env-driven so the same config works inside Docker
// Compose, where the backend service is reachable as http://backend:8080:
// set VITE_BACKEND_URL=http://backend:8080 there. Locally it defaults to
// http://localhost:8080, exactly as before.
const backendTarget = process.env.VITE_BACKEND_URL || 'http://localhost:8080'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 5173,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
})
