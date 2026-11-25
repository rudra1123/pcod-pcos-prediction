# Frontend image: installs dependencies, builds the Vite production bundle,
# and serves it with `vite preview` on port 5173. /api is proxied to the
# backend — the proxy target comes from VITE_BACKEND_URL (set to
# http://backend:8080 by docker-compose.yml; defaults to
# http://localhost:8080 when running Vite locally).
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 5173
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "5173"]
