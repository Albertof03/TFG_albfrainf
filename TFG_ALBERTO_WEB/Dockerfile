# Usa una imagen base con Node.js
FROM node:18

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar todo el código de backend y frontend
COPY . .

# Instalar dependencias del backend
RUN cd backend && npm install

# Instalar dependencias del frontend
RUN cd frontend/TFG && npm install

# Construir el frontend
RUN cd frontend/TFG && npm run build

# Exponer el puerto de la aplicación
EXPOSE 3000

# Iniciar la aplicación
CMD ["npm", "start", "--prefix", "backend"]
