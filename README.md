
# Laberinto: Cazador vs Evasor

Simulador con Flask (backend) y HTML (frontend) en un solo contenedor.

## ¿Cómo correrlo con Docker Compose?

1. Asegúrate de tener Docker y Docker Compose instalados.
2. En la carpeta del proyecto, ejecuta:

```sh
docker compose up -d
```

3. Abre en tu navegador:
   - http://localhost:5000
   - Salud backend: http://localhost:5000/api/health

Para detener:

```sh
docker compose down
```

## Estructura mínima

- `maze_simulator_backend.py` — Backend Flask
- `index.html` — Frontend estático
- `Dockerfile` — Imagen única
- `docker-compose.yml` — Orquestador simple
- `requirements.txt`

## Notas

- Todo se sirve en el mismo puerto (5000).
- No necesitas configurar variables de entorno extra.
- Compatible con despliegue en Render (Web Service tipo Docker).
