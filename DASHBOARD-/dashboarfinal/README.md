# Dashboard de Salud del Sueño

Dashboard interactivo para el análisis y visualización de trastornos del sueño basado en técnicas de Machine Learning (PCA, t-SNE, UMAP) y un motor de reglas clínicas.

## Requisitos

- Python 3.9+
- pip

## Mejora del vector de caracteristicas (PCA)

El vector de caracteristicas (`vector_caracteristicas_sleep.csv`) se redujo de 20 a
11 variables de salud/sueno + dummies de ocupacion:

- **Se excluyeron del calculo de PCA/embeddings** las 8 columnas `ocupacion_*`
  (one-hot, ruido disperso sin correlacion con el sueno). Esto ya mejora bastante
  la varianza explicada por si solo (`app.py` y `preprocesamiento_mineria_datos.py`).
- **Se eliminaron variables redundantes** (correlacion > 0.9 con otra variable ya
  incluida): `duracion_sueno_cuadrado_norm`, `deficit_sueno_norm`,
  `calidad_sueno_ordinal_norm`, `estres_ordinal_norm`, `interaccion_sueno_estres_norm`,
  `interaccion_sueno_actividad_norm`, `calidad_por_estres_norm`, `puntaje_salud_sueno_norm`.
- **Se combinaron** `presion_sistolica_norm` + `presion_diastolica_norm` (r=0.97) en una
  sola variable clinica estandar: `presion_arterial_media_norm` (MAP).

Nota importante: quitar variables redundantes reduce la multicolinealidad del vector
(bueno para que K-Means/DBSCAN no dupliquen la misma senal en varios ejes), pero por
como funciona PCA esto **no necesariamente sube el % de varianza explicada por PC1+PC2**
(las variables correlacionadas son justamente las que antes concentraban varianza en
pocos componentes). Si vuelves a correr `preprocesamiento_mineria_datos.py`,
`clustering_kmeans.py` y `clustering_dbscan.py`, se regeneran todos los resultados con
el nuevo vector de 11 features.

## Instalación

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

### Levantar el dashboard web

```bash
source .venv/Scripts/activate   # Windows
# o source .venv/bin/activate   # Linux/Mac
python app.py
```

Abrir en el navegador: **http://127.0.0.1:5000**

### Ejecutar el pipeline de ML (supervisado)

```bash
python pipeline_ml.py
```

Esto genera:
- `resultados_ml.json` — métricas y resultados
- `static/matriz_confusion.png` — matriz de confusión
- `static/feature_importance.png` — top 5 variables
- `static/distribucion_clases.png` — distribución de clases

### Ejecutar el pipeline de CLUSTERING (no supervisado)

```bash
python clustering_kmeans.py
```

Segmenta a las personas con **K-Means** según su perfil de salud/sueño. Elige el
número óptimo de clusters `k` combinando el **Método del Codo** (inercia/WCSS) y
el **Coeficiente de Silueta** (se queda con el `k` de mayor silueta media). Genera:
- `resultados_clustering.json` — k óptimo, curvas de codo/silueta, perfil y asignación de cada cluster
- `clustering_asignaciones.csv` — `id_persona` → cluster asignado
- `static/clustering_codo.png` — método del codo
- `static/clustering_silueta.png` — coeficiente de silueta por k
- `static/clustering_scatter.png` — clusters sobre proyección PCA 2D

Luego abre el dashboard y pulsa el botón **🧩 Clustering** del encabezado para ver
las curvas, el mapa de clusters y la caracterización de cada segmento.

Además de la silueta, el resultado incluye **Davies-Bouldin** y **Calinski-Harabasz**
y la **distancia de cada punto a su propio centroide** (`distancia_centroide`), y el
mapa de dispersión ahora combina **color + forma** por cluster para verificar
visualmente la calidad de la segmentación.

### Ejecutar el pipeline de DBSCAN (clustering por densidad + anomalías)

```bash
python clustering_dbscan.py
```

A diferencia de K-Means, **DBSCAN** no necesita que definas el número de clusters:
los descubre agrupando zonas densas del espacio de características y marca como
**anomalías** (`ruido`) a los puntos que no encajan en ninguna zona densa. Esto
responde directamente a "algoritmo de datos anómalos" y "calcular la distancia a
qué cluster pertenece cada punto":

- `eps` (radio de vecindad) se elige automáticamente con el **gráfico de
  Distancia-k**, buscando el "codo" de la curva (método kneedle).
- `min_samples` se fija en `n_features + 1`, la heurística estándar recomendada
  por Sander et al. para datos con varias dimensiones.
- Para cada punto calcula la **distancia a su propio cluster** y la
  **distancia al cluster más cercano** (si es anomalía, a cuál se parece más).
- Valida los clusters con **silueta, Davies-Bouldin y Calinski-Harabasz**
  (excluyendo anomalías del cálculo).

Genera:
- `resultados_dbscan.json` — parámetros, curva de distancia-k, métricas, clusters, anomalías y coordenadas 2D
- `dbscan_asignaciones.csv` — `id_persona` → cluster (o `-1` si es anomalía) + distancias
- `static/dbscan_k_distancia.png` — gráfico de distancia-k con el punto de corte (eps)
- `static/dbscan_scatter.png` — clusters (color + forma) y anomalías (cruces) sobre PCA 2D

Luego pulsa el botón **🚨 DBSCAN** del encabezado del dashboard.

## Endpoints API

| Ruta | Descripción |
|------|-------------|
| `GET /` | Página principal del dashboard |
| `GET /api/data` | Datos completos + embeddings PCA/t-SNE/UMAP |
| `GET /api/ml/results` | Resultados del pipeline de ML supervisado |
| `GET /api/clustering` | Resultados del clustering K-Means (segmentos) |
| `GET /api/dbscan` | Resultados del clustering DBSCAN (densidad + anomalías) |
| `GET /api/riesgo-global` | Resumen de riesgo organizacional |
| `GET /api/alertas/<id>` | Alertas clínicas por empleado |
| `POST /api/predict` | Evaluación de nuevo paciente (envía JSON) |

## Estructura del proyecto

```
dashboarfinal/
├── app.py                              # Servidor Flask + motor de reglas
├── pipeline_ml.py                      # Pipeline de Machine Learning
├── clustering_kmeans.py                # Pipeline de clustering K-Means
├── clustering_dbscan.py                # Pipeline de clustering DBSCAN (densidad + anomalías)
├── requirements.txt                    # Dependencias
├── vector_caracteristicas_sleep.csv    # Features normalizadas
├── Sleep_health_dataset_es_transformado.csv  # Datos transformados
├── resultados_ml.json                  # Resultados del ML
├── templates/
│   └── index.html                      # Frontend del dashboard
└── static/
    ├── matriz_confusion.png
    ├── feature_importance.png
    └── distribucion_clases.png
```
