# Frontend image: multi-stage build. Stage 1 installs dependencies and
# builds the Vite production bundle; stage 2 serves the static files with
# nginx (a real production server, not the Vite dev/preview server) and
# proxies /api to the backend service inside the Compose network
# (see nginx.conf). The frontend calls the backend with relative /api
# URLs, so no build-time API URL is needed.
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 5173
CMD ["nginx", "-g", "daemon off;"]
