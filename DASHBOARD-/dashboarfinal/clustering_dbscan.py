"""
Pipeline de CLUSTERING BASADO EN DENSIDAD (DBSCAN) para deteccion de
GRUPOS DE FORMA ARBITRARIA y DATOS ANOMALOS (outliers) en el perfil de
salud del sueno.

A diferencia de K-Means (clustering_kmeans.py), DBSCAN no obliga a cada
punto a pertenecer a un cluster: los puntos que no tienen suficientes
vecinos cercanos quedan marcados como RUIDO / ANOMALIA (label = -1).
Esto permite responder la pregunta "¿que registros no encajan en ningun
patron de sueno conocido?".

Metodologia (Ester et al. 1996, "A density-based algorithm for discovering
clusters in large spatial databases with noise"):
  1. Se usa el mismo vector de features de salud/sueno que K-Means
     (estandarizado con StandardScaler), para que ambos algoritmos sean
     comparables.
  2. Seleccion de min_samples: heuristica practica minPts >= dimension + 1
     (Ester et al. 1996; Schubert et al. 2017 recomiendan 2*dim para datos
     ruidosos, pero con n=374 y dim=20 eso deja casi todo como ruido, por lo
     que se usa minPts = dim + 1 como cota inferior razonable).
  3. Seleccion de eps con el GRAFICO DE DISTANCIA-K (k-distance plot):
     se calcula la distancia de cada punto a su k-esimo vecino mas cercano
     (k = min_samples), se ordenan esas distancias de forma ascendente y se
     busca el "codo" (punto de maxima curvatura) -> ese valor de distancia es
     el eps recomendado (Ester et al. 1996; Rahmah & Sitanggang 2016 aplican
     el mismo criterio para elegir eps de forma reproducible/justificada).
  4. DBSCAN final con (eps, min_samples) elegidos.
  5. Para cada punto se calcula la DISTANCIA a los centroides de cada
     cluster encontrado y se guarda la distancia MINIMA (distancia al
     cluster mas cercano) y el cluster al que pertenece (o el cluster mas
     cercano si es anomalia). Esto responde directamente "calcular la
     distancia a que cluster pertenece cada punto".
  6. Metricas de validacion de clustering (aplicadas sobre los puntos que
     NO son ruido, siguiendo la practica estandar cuando hay outliers):
        - Coeficiente de Silueta         (cohesion/separacion, [-1,1])
        - Indice de Davies-Bouldin       (menor es mejor)
        - Indice de Calinski-Harabasz    (mayor es mejor)
  7. Proyeccion PCA 2D para visualizar clusters + anomalias.
  8. Caracterizacion (perfil medio) de cada cluster y listado de las
     anomalias mas alejadas de cualquier cluster.

Genera:
  - resultados_dbscan.json          -> lo consume el endpoint /api/dbscan
  - dbscan_asignaciones.csv         -> id_persona, cluster/anomalia, distancia
  - static/dbscan_k_distancia.png
  - static/dbscan_scatter.png

Ejecucion:
    python clustering_dbscan.py
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
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
VECTOR_PATH = BASE_DIR / "vector_caracteristicas_sleep.csv"
DETALLES_PATH = BASE_DIR / "Sleep_health_dataset_es_transformado.csv"
RESULTADOS_PATH = BASE_DIR / "resultados_dbscan.json"
ASIGNACIONES_PATH = BASE_DIR / "dbscan_asignaciones.csv"
STATIC = BASE_DIR / "static"
STATIC.mkdir(exist_ok=True)
KDIST_PNG = STATIC / "dbscan_k_distancia.png"
SCATTER_PNG = STATIC / "dbscan_scatter.png"

# Mismo criterio de exclusion que clustering_kmeans.py para que ambos
# algoritmos trabajen sobre exactamente el mismo espacio de features.
EXCLUIR = {"id_persona", "trastorno_sueno"}
RANDOM_STATE = 42
PERFIL_VARS = [
    "edad", "duracion_sueno_horas", "calidad_sueno", "nivel_estres",
    "nivel_actividad_fisica", "frecuencia_cardiaca", "pasos_diarios",
    "deficit_sueno", "puntaje_salud_sueno",
]
PALETA = ["#5E9FE8", "#EAC26B", "#72BC8F", "#BF8EDA", "#DE9255", "#DF84A8", "#4FB9C9", "#E97366"]
ANOMALIA_COLOR = "#2C2C2B"
TOP_ANOMALIAS = 25


def cargar_features():
    df_vec = pd.read_csv(VECTOR_PATH, encoding="utf-8-sig")
    feat_cols = [
        c for c in df_vec.columns
        if c not in EXCLUIR and not c.startswith("ocupacion_")
    ]
    X = df_vec[feat_cols].astype(float).to_numpy()
    X = StandardScaler().fit_transform(X)
    return df_vec, X, feat_cols


def encontrar_codo(y):
    """Punto de maxima distancia perpendicular a la cuerda entre el primer y
    ultimo punto de la curva (heuristica tipo 'kneedle', Satopaa et al. 2011).
    Evita depender de una libreria externa para justificar el eps elegido.
    """
    n = len(y)
    x = np.arange(n, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x0, y0 = x_n[0], y_n[0]
    x1, y1 = x_n[-1], y_n[-1]
    # distancia perpendicular de cada punto a la recta (x0,y0)-(x1,y1)
    num = np.abs((y1 - y0) * x_n - (x1 - x0) * y_n + x1 * y0 - y1 * x0)
    den = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2) + 1e-12
    dist = num / den
    return int(np.argmax(dist))


def calcular_min_samples(n_features):
    return int(n_features) + 1


def graficar_k_distancia(distancias_ordenadas, idx_codo, eps, min_samples, path):
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(distancias_ordenadas, color="#2783DE", linewidth=2)
    plt.axhline(eps, color="#E56458", linestyle="--", label=f"eps elegido = {eps:.3f}")
    plt.scatter([idx_codo], [distancias_ordenadas[idx_codo]], color="#E56458", zorder=5, s=45)
    plt.title(f"Grafico de Distancia-k (k = min_samples = {min_samples})", fontweight="bold")
    plt.xlabel("Puntos ordenados por distancia")
    plt.ylabel(f"Distancia al {min_samples}-esimo vecino mas cercano")
    plt.legend()
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def ejecutar():
    print("Cargando features (mismo espacio que K-Means)...")
    df_vec, X, feat_cols = cargar_features()
    n_features = len(feat_cols)
    print(f"  {X.shape[0]} muestras x {n_features} features de salud/sueno")

    min_samples = calcular_min_samples(n_features)
    print(f"Calculando grafico de distancia-k (min_samples={min_samples})...")
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dist_k, _ = nn.kneighbors(X)
    dist_k_max = np.sort(dist_k[:, -1])  # distancia al vecino mas lejano (k-esimo)
    idx_codo = encontrar_codo(dist_k_max)
    eps = float(dist_k_max[idx_codo])
    print(f"  -> eps elegido (codo del grafico distancia-k) = {eps:.4f}")

    print("Ajustando DBSCAN...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    ids_clusters = sorted(c for c in set(labels) if c != -1)
    n_clusters = len(ids_clusters)
    es_ruido = labels == -1
    n_anomalias = int(es_ruido.sum())
    pct_anomalias = round(n_anomalias / len(labels) * 100, 1)
    print(f"  -> {n_clusters} clusters encontrados, {n_anomalias} anomalias ({pct_anomalias}%)")

    # ── Metricas de validacion (excluyendo ruido, practica estandar) ───────
    metricas = {"silueta": None, "davies_bouldin": None, "calinski_harabasz": None}
    if n_clusters >= 2:
        mask_validos = ~es_ruido
        Xv, Lv = X[mask_validos], labels[mask_validos]
        metricas["silueta"] = round(float(silhouette_score(Xv, Lv)), 4)
        metricas["davies_bouldin"] = round(float(davies_bouldin_score(Xv, Lv)), 4)
        metricas["calinski_harabasz"] = round(float(calinski_harabasz_score(Xv, Lv)), 2)
        sil_ind_validos = silhouette_samples(Xv, Lv)
        sil_por_punto = np.full(len(labels), np.nan)
        sil_por_punto[mask_validos] = sil_ind_validos
    else:
        sil_por_punto = np.full(len(labels), np.nan)
        print("  Aviso: menos de 2 clusters validos, no se calculan metricas de validacion.")

    # ── Centroides por cluster + distancia de CADA punto a CADA centroide ──
    centroides = np.array([X[labels == c].mean(axis=0) for c in ids_clusters]) if n_clusters else np.empty((0, X.shape[1]))
    if n_clusters:
        # distancia euclidiana de cada punto a cada centroide
        dists_a_centroides = np.linalg.norm(X[:, None, :] - centroides[None, :, :], axis=2)  # (n, n_clusters)
        cluster_mas_cercano = np.array(ids_clusters)[np.argmin(dists_a_centroides, axis=1)]
        distancia_cluster_mas_cercano = dists_a_centroides.min(axis=1)
        # para puntos ya asignados, tambien queremos su distancia a SU PROPIO centroide
        idx_propio = np.array([ids_clusters.index(l) if l != -1 else -1 for l in labels])
        distancia_propio_cluster = np.array([
            dists_a_centroides[i, idx_propio[i]] if idx_propio[i] != -1 else distancia_cluster_mas_cercano[i]
            for i in range(len(labels))
        ])
    else:
        cluster_mas_cercano = np.full(len(labels), -1)
        distancia_cluster_mas_cercano = np.zeros(len(labels))
        distancia_propio_cluster = np.zeros(len(labels))

    print("Proyeccion PCA 2D...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    det = pd.read_csv(DETALLES_PATH, encoding="utf-8-sig").set_index("id_persona")
    ids = df_vec["id_persona"].to_numpy()

    clusters_info = []
    for c in ids_clusters:
        mask = labels == c
        sub = det.loc[ids[mask]]
        trast = sub["trastorno_sueno"].value_counts()
        perfil = {v: round(float(sub[v].mean()), 2) for v in PERFIL_VARS if v in sub.columns}
        ocup = sub["ocupacion"].value_counts().head(3).to_dict() if "ocupacion" in sub.columns else {}
        sil_c = sil_por_punto[mask]
        clusters_info.append({
            "cluster": int(c),
            "tamano": int(mask.sum()),
            "porcentaje": round(float(mask.mean() * 100), 1),
            "silueta_media": round(float(np.nanmean(sil_c)), 4) if np.any(~np.isnan(sil_c)) else None,
            "distancia_media_centroide": round(float(distancia_propio_cluster[mask].mean()), 4),
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
            "es_anomalia": bool(labels[i] == -1),
            "trastorno": str(det.loc[ids[i], "trastorno_sueno"]),
            "distancia_propio_cluster": round(float(distancia_propio_cluster[i]), 4),
            "cluster_mas_cercano": int(cluster_mas_cercano[i]),
            "distancia_cluster_mas_cercano": round(float(distancia_cluster_mas_cercano[i]), 4),
        }
        for i in range(len(ids))
    ]

    anomalias = sorted(
        [p for p in puntos if p["es_anomalia"]],
        key=lambda p: p["distancia_cluster_mas_cercano"],
        reverse=True,
    )[:TOP_ANOMALIAS]

    resultado = {
        "algoritmo": "DBSCAN (density-based, Ester et al. 1996)",
        "n_muestras": int(len(ids)),
        "n_features": n_features,
        "features_usadas": feat_cols,
        "parametros": {"eps": round(eps, 4), "min_samples": min_samples},
        "criterio_eps": "Codo del grafico de distancia-k (maxima distancia perpendicular a la cuerda)",
        "criterio_min_samples": "minPts = n_features + 1 (cota inferior de Ester et al. 1996)",
        "curva_k_distancia": {
            "orden": list(range(len(dist_k_max))),
            "distancia": [round(float(d), 4) for d in dist_k_max],
            "idx_codo": idx_codo,
        },
        "n_clusters": n_clusters,
        "n_anomalias": n_anomalias,
        "porcentaje_anomalias": pct_anomalias,
        "metricas_validacion": metricas,
        "varianza_pca": [round(float(var[0]), 4), round(float(var[1]), 4)],
        "clusters": clusters_info,
        "anomalias": anomalias,
        "puntos": puntos,
    }
    RESULTADOS_PATH.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Guardado: {RESULTADOS_PATH.name}")

    pd.DataFrame({
        "id_persona": ids,
        "cluster": labels,
        "es_anomalia": es_ruido,
        "trastorno_sueno": [det.loc[i, "trastorno_sueno"] for i in ids],
        "distancia_propio_cluster": np.round(distancia_propio_cluster, 4),
        "cluster_mas_cercano": cluster_mas_cercano,
        "distancia_cluster_mas_cercano": np.round(distancia_cluster_mas_cercano, 4),
    }).to_csv(ASIGNACIONES_PATH, index=False, encoding="utf-8-sig")
    print(f"  Guardado: {ASIGNACIONES_PATH.name}")

    # ── Graficos ────────────────────────────────────────────────────────────
    graficar_k_distancia(dist_k_max, idx_codo, eps, min_samples, KDIST_PNG)

    plt.figure(figsize=(7.4, 5.4))
    for c in ids_clusters:
        m = labels == c
        plt.scatter(coords[m, 0], coords[m, 1], s=30, alpha=.75,
                    color=PALETA[c % len(PALETA)], edgecolors="white", linewidths=.4,
                    label=f"Cluster {c} (n={int(m.sum())})")
    plt.scatter(coords[es_ruido, 0], coords[es_ruido, 1], marker="x", s=55,
                color=ANOMALIA_COLOR, linewidths=1.4,
                label=f"Anomalias (n={n_anomalias})", zorder=5)
    plt.title(f"DBSCAN: {n_clusters} clusters + anomalias sobre proyeccion PCA 2D", fontweight="bold")
    plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
    plt.legend(fontsize=8)
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=160)
    plt.close()
    print("  Graficos guardados en static/")

    print("\nResumen de clusters DBSCAN:")
    for c in clusters_info:
        print(f"  Cluster {c['cluster']}: n={c['tamano']} ({c['porcentaje']}%) "
              f"| dominante: {c['trastorno_dominante']} | dist. media al centroide={c['distancia_media_centroide']}")
    print(f"\nAnomalias detectadas: {n_anomalias} ({pct_anomalias}%)")
    if metricas["silueta"] is not None:
        print(f"Silueta={metricas['silueta']}  Davies-Bouldin={metricas['davies_bouldin']}  Calinski-Harabasz={metricas['calinski_harabasz']}")
    return resultado


if __name__ == "__main__":
    ejecutar()
