# Frontend Dockerfile: build with Node, serve preview (for production use, consider building static files and serving with nginx)
FROM node:18-alpine
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --silent || npm install --silent
COPY frontend /app
RUN npm run build
EXPOSE 5173
CMD ["npm", "run", "preview", "--", "--host"]
