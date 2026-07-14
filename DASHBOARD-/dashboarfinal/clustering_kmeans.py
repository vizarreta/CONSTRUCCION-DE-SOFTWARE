"""
Pipeline de CLUSTERING (aprendizaje NO supervisado) para segmentar personas
segun su perfil de salud del sueno.

Metodologia de mineria de datos:
  1. Carga del vector de caracteristicas normalizado.
  2. Seleccion de features de salud/sueno (se excluyen las dummies de ocupacion
     para obtener segmentos clinicamente interpretables).
  3. Estandarizacion (StandardScaler).
  4. Busqueda del numero optimo de clusters k:
        - Metodo del Codo (inercia / WCSS)
        - Coeficiente de Silueta (se elige el k con mayor silueta media)
  5. K-Means final con el k optimo.
  6. Proyeccion PCA 2D para visualizar los clusters.
  7. Caracterizacion (perfil medio) de cada cluster.

Genera:
  - resultados_clustering.json      -> lo consume el endpoint /api/clustering
  - clustering_asignaciones.csv     -> id_persona + cluster asignado
  - static/clustering_codo.png
  - static/clustering_silueta.png
  - static/clustering_scatter.png

Ejecucion:
    python clustering_kmeans.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
VECTOR_PATH = BASE_DIR / "vector_caracteristicas_sleep.csv"
DETALLES_PATH = BASE_DIR / "Sleep_health_dataset_es_transformado.csv"
RESULTADOS_PATH = BASE_DIR / "resultados_clustering.json"
ASIGNACIONES_PATH = BASE_DIR / "clustering_asignaciones.csv"
STATIC = BASE_DIR / "static"
STATIC.mkdir(exist_ok=True)
CODO_PNG = STATIC / "clustering_codo.png"
SILUETA_PNG = STATIC / "clustering_silueta.png"
SCATTER_PNG = STATIC / "clustering_scatter.png"

EXCLUIR = {"id_persona", "trastorno_sueno"}
RANDOM_STATE = 42
K_RANGE = list(range(2, 9))          # k = 2..8
PERFIL_VARS = [
    "edad", "duracion_sueno_horas", "calidad_sueno", "nivel_estres",
    "nivel_actividad_fisica", "frecuencia_cardiaca", "pasos_diarios",
    "deficit_sueno", "puntaje_salud_sueno",
]
PALETA = ["#5E9FE8", "#EAC26B", "#72BC8F", "#BF8EDA", "#DE9255", "#DF84A8", "#4FB9C9", "#E97366"]


def cargar_features():
    df_vec = pd.read_csv(VECTOR_PATH, encoding="utf-8-sig")
    # Excluir id, target y las dummies de ocupacion -> segmentos por salud/sueno
    feat_cols = [
        c for c in df_vec.columns
        if c not in EXCLUIR and not c.startswith("ocupacion_")
    ]
    X = df_vec[feat_cols].astype(float).to_numpy()
    X = StandardScaler().fit_transform(X)
    return df_vec, X, feat_cols


def buscar_k_optimo(X):
    inercias, siluetas = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inercias.append(round(float(km.inertia_), 2))
        siluetas.append(round(float(silhouette_score(X, labels)), 4))
        print(f"  k={k}  inercia={inercias[-1]:.1f}  silueta={siluetas[-1]:.4f}")
    k_opt = K_RANGE[int(np.argmax(siluetas))]
    return inercias, siluetas, k_opt


def graficar_curva(x, y, titulo, ylabel, color, k_opt, path):
    plt.figure(figsize=(7, 4.2))
    plt.plot(x, y, "o-", color=color, linewidth=2, markersize=7)
    plt.axvline(k_opt, color="#E56458", linestyle="--", label=f"k optimo = {k_opt}")
    plt.title(titulo, fontweight="bold")
    plt.xlabel("Numero de clusters (k)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def ejecutar():
    print("Cargando features...")
    df_vec, X, feat_cols = cargar_features()
    print(f"  {X.shape[0]} muestras x {X.shape[1]} features de salud/sueno")

    print("Buscando k optimo (Codo + Silueta)...")
    inercias, siluetas, k_opt = buscar_k_optimo(X)
    print(f"  -> k optimo = {k_opt}")

    print("Ajustando K-Means final...")
    km = KMeans(n_clusters=k_opt, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    sil_prom = float(silhouette_score(X, labels))
    sil_ind = silhouette_samples(X, labels)
    davies_bouldin = round(float(davies_bouldin_score(X, labels)), 4)
    calinski_harabasz = round(float(calinski_harabasz_score(X, labels)), 2)
    # Distancia (euclidiana, en espacio estandarizado) de cada punto a SU
    # propio centroide -> responde "calcular la distancia a que cluster
    # pertenece cada punto" / "distancia menor dentro del cluster".
    dist_centroide = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)

    print("Proyeccion PCA 2D...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    # ── Perfiles de cada cluster con variables reales ──────────────────────
    real = df_vec[["id_persona"]].copy()
    det = pd.read_csv(DETALLES_PATH, encoding="utf-8-sig").set_index("id_persona")
    ids = df_vec["id_persona"].to_numpy()

    clusters = []
    for c in range(k_opt):
        mask = labels == c
        sub = det.loc[ids[mask]]
        trast = sub["trastorno_sueno"].value_counts()
        perfil = {v: round(float(sub[v].mean()), 2) for v in PERFIL_VARS if v in sub.columns}
        ocup = sub["ocupacion"].value_counts().head(3).to_dict() if "ocupacion" in sub.columns else {}
        clusters.append({
            "cluster": int(c),
            "tamano": int(mask.sum()),
            "porcentaje": round(float(mask.mean() * 100), 1),
            "silueta_media": round(float(sil_ind[mask].mean()), 4),
            "trastorno_dominante": str(trast.idxmax()),
            "distribucion_trastorno": {str(k): int(v) for k, v in trast.items()},
            "top_ocupaciones": {str(k): int(v) for k, v in ocup.items()},
            "perfil_medio": perfil,
        })

    puntos = [
        {
            "id": int(ids[i]),
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "cluster": int(labels[i]),
            "trastorno": str(det.loc[ids[i], "trastorno_sueno"]),
            "distancia_centroide": round(float(dist_centroide[i]), 4),
        }
        for i in range(len(ids))
    ]

    resultado = {
        "algoritmo": "K-Means (k-means++)",
        "n_muestras": int(len(ids)),
        "n_features": len(feat_cols),
        "features_usadas": feat_cols,
        "rango_k_evaluado": K_RANGE,
        "curva_codo": {"k": K_RANGE, "inercia": inercias},
        "curva_silueta": {"k": K_RANGE, "silueta": siluetas},
        "k_optimo": int(k_opt),
        "criterio_k": "Coeficiente de silueta maximo",
        "silueta_promedio": round(sil_prom, 4),
        "metricas_validacion": {
            "silueta": round(sil_prom, 4),
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz,
        },
        "varianza_pca": [round(float(var[0]), 4), round(float(var[1]), 4)],
        "clusters": clusters,
        "puntos": puntos,
    }
    RESULTADOS_PATH.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Guardado: {RESULTADOS_PATH.name}")

    pd.DataFrame({
        "id_persona": ids,
        "cluster": labels,
        "trastorno_sueno": [det.loc[i, "trastorno_sueno"] for i in ids],
        "silueta": np.round(sil_ind, 4),
        "distancia_centroide": np.round(dist_centroide, 4),
    }).to_csv(ASIGNACIONES_PATH, index=False, encoding="utf-8-sig")
    print(f"  Guardado: {ASIGNACIONES_PATH.name}")

    # ── Graficos ────────────────────────────────────────────────────────────
    graficar_curva(K_RANGE, inercias, "Metodo del Codo (Elbow)", "Inercia (WCSS)",
                   "#2783DE", k_opt, CODO_PNG)
    graficar_curva(K_RANGE, siluetas, "Coeficiente de Silueta por k", "Silueta promedio",
                   "#46A171", k_opt, SILUETA_PNG)

    plt.figure(figsize=(7, 5.2))
    for c in range(k_opt):
        m = labels == c
        plt.scatter(coords[m, 0], coords[m, 1], s=28, alpha=.75,
                    color=PALETA[c % len(PALETA)], edgecolors="white", linewidths=.4,
                    label=f"Cluster {c} (n={int(m.sum())})")
    plt.scatter(pca.transform(km.cluster_centers_)[:, 0],
                pca.transform(km.cluster_centers_)[:, 1],
                marker="X", s=180, c="#2C2C2B", edgecolors="white", linewidths=1.2,
                label="Centroides", zorder=5)
    plt.title(f"Clusters K-Means (k={k_opt}) sobre proyeccion PCA 2D", fontweight="bold")
    plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
    plt.legend(fontsize=8)
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=160)
    plt.close()
    print(f"  Graficos guardados en static/")

    print("\nResumen de clusters:")
    for c in clusters:
        print(f"  Cluster {c['cluster']}: n={c['tamano']} ({c['porcentaje']}%) "
              f"| dominante: {c['trastorno_dominante']} | silueta={c['silueta_media']}")
    print(f"\nSilueta promedio global: {sil_prom:.4f}")
    print(f"Davies-Bouldin: {davies_bouldin}  |  Calinski-Harabasz: {calinski_harabasz}")
    return resultado


if __name__ == "__main__":
    ejecutar()
