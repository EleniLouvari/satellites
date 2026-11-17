# syntax=docker/dockerfile:1
FROM node:22-alpine AS build
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/index.html web/tsconfig.json web/vite.config.ts ./
COPY web/src/ ./src/
COPY web/scripts/ ./scripts/
# Generate the guide from its authoritative sources, including source-package reference guides.
COPY docs/ /app/docs/
COPY src/ml_classification/*.md /app/src/ml_classification/
RUN npm run build

FROM nginxinc/nginx-unprivileged:stable-alpine
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/web/dist /usr/share/nginx/html
EXPOSE 8080
