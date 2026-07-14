import hashlib
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
import umap

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
VECTOR_PATH = BASE_DIR / "vector_caracteristicas_sleep.csv"
DETALLES_PATH = BASE_DIR / "Sleep_health_dataset_es_transformado.csv"
CACHE_DIR = BASE_DIR / "cache"

COLUMNAS_EXCLUIR = {"id_persona", "trastorno_sueno"}
COLUMNAS_DETALLE = [
    "id_persona",
    "genero",
    "edad",
    "ocupacion",
    "duracion_sueno_horas",
    "calidad_sueno",
    "nivel_actividad_fisica",
    "nivel_estres",
    "categoria_imc",
    "presion_arterial",
    "frecuencia_cardiaca",
    "pasos_diarios",
    "trastorno_sueno",
    "calidad_sueno_nivel",
    "estres_nivel",
    "tiene_anomalia_iqr",
    "deficit_sueno",
    "eficiencia_sueno",
    "interaccion_sueno_estres",
    "interaccion_sueno_actividad",
    "calidad_por_estres",
    "duracion_sueno_cuadrado",
    "puntaje_salud_sueno",
]
ATRIBUTOS_BARRA = [
    "duracion_sueno_horas_norm",
    "calidad_sueno_norm",
    "nivel_actividad_fisica_norm",
    "nivel_estres_norm",
    "frecuencia_cardiaca_norm",
    "pasos_diarios_norm",
    "presion_sistolica_norm",
    "presion_diastolica_norm",
    "edad_norm",
    "genero_binario_mujer",
    "calidad_sueno_ordinal_norm",
    "estres_ordinal_norm",
    "categoria_imc_ordinal_norm",
    "deficit_sueno_norm",
    "eficiencia_sueno_norm",
    "interaccion_sueno_estres_norm",
    "interaccion_sueno_actividad_norm",
    "calidad_por_estres_norm",
    "duracion_sueno_cuadrado_norm",
    "puntaje_salud_sueno_norm",
]
ATRIBUTOS_LABELS = {
    "duracion_sueno_horas_norm": "Horas sueño",
    "calidad_sueno_norm": "Calidad sueño",
    "nivel_actividad_fisica_norm": "Actividad física",
    "nivel_estres_norm": "Estrés",
    "frecuencia_cardiaca_norm": "Freq. cardíaca",
    "pasos_diarios_norm": "Pasos diarios",
    "presion_sistolica_norm": "Presión sistólica",
    "presion_diastolica_norm": "Presión diastólica",
    "presion_arterial_media_norm": "Presión arterial media",
    "edad_norm": "Edad",
    "genero_binario_mujer": "Género (Mujer)",
    "calidad_sueno_ordinal_norm": "Calidad ordinal",
    "estres_ordinal_norm": "Estrés ordinal",
    "categoria_imc_ordinal_norm": "IMC ordinal",
    "deficit_sueno_norm": "Déficit sueño",
    "eficiencia_sueno_norm": "Eficiencia sueño",
    "interaccion_sueno_estres_norm": "Sueño×Estrés",
    "interaccion_sueno_actividad_norm": "Sueño×Actividad",
    "calidad_por_estres_norm": "Cal/estrés",
    "duracion_sueno_cuadrado_norm": "Sueño²",
    "puntaje_salud_sueno_norm": "Salud sueño",
}

COLORES_TRASTORNO = {
    "Ninguno": "#2f6f9f",
    "Insomnio": "#c94c4c",
    "Apnea del sueño": "#5c8f3f",
}

IMC_RIESGO = {"Overweight", "Obese", "Sobrepeso", "Obesidad", "Obeso", "Con sobrepeso"}


def _parsear_presion(presion_str):
    """Extrae sistólica y diastólica de un string tipo '130/85'."""
    try:
        partes = str(presion_str).split("/")
        return int(partes[0]), int(partes[1])
    except Exception:
        return None, None


def generar_alertas_clinicas(datos):
    """
    Aplica el motor de reglas clínicas basado en el paper IEEE.
    Recibe un dict con los campos del paciente y devuelve lista de alertas.
    """
    alertas = []

    imc = str(datos.get("categoria_imc") or "").strip()
    presion_raw = datos.get("presion_arterial")
    duracion = datos.get("duracion_sueno_horas")
    estres = datos.get("nivel_estres")
    calidad = datos.get("calidad_sueno")
    fc = datos.get("frecuencia_cardiaca")
    actividad = datos.get("nivel_actividad_fisica")

    sistolica, diastolica = _parsear_presion(presion_raw)

    # ── Regla 1: Riesgo Apnea (IMC + presión) ──────────────────────────────
    if imc in IMC_RIESGO and sistolica is not None and sistolica >= 130:
        alertas.append({
            "tipo": "peligro",
            "codigo": "APNEA_IMC_PRESION",
            "titulo": "Riesgo de Apnea del Sueño",
            "mensaje": (
                f"IMC elevado ({imc}) + Presión sistólica {sistolica} mmHg ≥ 130. "
                "Patrón fuertemente asociado a Apnea del Sueño según el paper IEEE."
            ),
            "recomendacion": "Derivar a evaluación de especialista en medicina del sueño.",
        })

    # ── Regla 2: Riesgo Insomnio (sueño + estrés) ──────────────────────────
    try:
        dur_f = float(duracion)
        est_f = float(estres)
        if dur_f < 6.0 and est_f >= 7:
            alertas.append({
                "tipo": "advertencia",
                "codigo": "INSOMNIO_SUENO_ESTRES",
                "titulo": "Riesgo de Insomnio",
                "mensaje": (
                    f"Duración del sueño {dur_f:.1f} h < 6 h y estrés {est_f:.0f}/10 ≥ 7. "
                    "Déficit de sueño combinado con estrés elevado: patrón de insomnio conductual."
                ),
                "recomendacion": "Programa de higiene del sueño y gestión del estrés.",
            })
    except (TypeError, ValueError):
        pass

    # ── Regla 3: Taquicardia en reposo ──────────────────────────────────────
    try:
        fc_f = float(fc)
        if fc_f > 90:
            alertas.append({
                "tipo": "advertencia",
                "codigo": "TAQUICARDIA",
                "titulo": "Frecuencia Cardíaca Elevada",
                "mensaje": (
                    f"Frecuencia cardíaca en reposo {fc_f:.0f} bpm > 90 bpm. "
                    "Puede indicar estrés crónico o alteración del sueño subyacente."
                ),
                "recomendacion": "Control médico preventivo.",
            })
    except (TypeError, ValueError):
        pass

    # ── Regla 4: Sedentarismo + baja calidad de sueño ───────────────────────
    try:
        act_f = float(actividad)
        cal_f = float(calidad)
        if act_f < 30 and cal_f < 6:
            alertas.append({
                "tipo": "info",
                "codigo": "SEDENTARISMO_CALIDAD",
                "titulo": "Sedentarismo y Baja Calidad de Sueño",
                "mensaje": (
                    f"Actividad física {act_f:.0f} min/día < 30 min y calidad del sueño "
                    f"{cal_f:.0f}/10 < 6. El sedentarismo está correlacionado con peor calidad de sueño."
                ),
                "recomendacion": "Aumentar actividad física aeróbica a ≥ 30 min/día.",
            })
    except (TypeError, ValueError):
        pass

    # ── Regla 5: Presión diastólica alta (hipertensión) ─────────────────────
    if diastolica is not None and diastolica >= 90:
        alertas.append({
            "tipo": "advertencia",
            "codigo": "HIPERTENSION_DIASTOLICA",
            "titulo": "Presión Diastólica Elevada",
            "mensaje": (
                f"Presión diastólica {diastolica} mmHg ≥ 90. "
                "Hipertensión diastólica asociada a fragmentación del sueño."
            ),
            "recomendacion": "Control de presión arterial con profesional de salud.",
        })

    return alertas


class DatosNoDisponibles(Exception):
    """Se lanza cuando faltan archivos CSV necesarios para servir los datos."""
    def __init__(self, faltantes):
        self.faltantes = faltantes
        super().__init__("Faltan archivos de datos: " + ", ".join(faltantes))


@app.errorhandler(DatosNoDisponibles)
def _manejar_datos_no_disponibles(err):
    return jsonify({
        "error": "Faltan archivos de datos necesarios en el servidor.",
        "archivos_faltantes": err.faltantes,
        "solucion": "Genera o copia los CSV requeridos (por ejemplo ejecutando "
                    "preprocesamiento_mineria_datos.py) y reinicia la aplicacion.",
    }), 503


def cargar_datos():
    faltantes = [p.name for p in (VECTOR_PATH, DETALLES_PATH) if not p.exists()]
    if faltantes:
        raise DatosNoDisponibles(faltantes)
    df_vector = pd.read_csv(VECTOR_PATH, encoding="utf-8-sig")
    df_detalles = pd.read_csv(DETALLES_PATH, encoding="utf-8-sig")
    detalle_cols = [col for col in COLUMNAS_DETALLE if col in df_detalles.columns]

    df = df_vector.merge(
        df_detalles[detalle_cols],
        on="id_persona",
        how="left",
        suffixes=("", "_detalle"),
    )
    if "trastorno_sueno_detalle" in df.columns:
        df["trastorno_sueno"] = df["trastorno_sueno_detalle"].fillna(
            df["trastorno_sueno"]
        )
        df = df.drop(columns=["trastorno_sueno_detalle"])

    # Las columnas "ocupacion_*" son dummies one-hot (una persona activa
    # solo 1 de 8) que casi no correlacionan con las variables de sueno/
    # salud; incluirlas en PCA/t-SNE/UMAP/MDS reparte la varianza entre
    # muchas dimensiones poco informativas y baja la varianza explicada
    # de los embeddings 2D. Se excluyen aqui para mejorar los embeddings.
    feature_cols = [
        c for c in df_vector.columns
        if c not in COLUMNAS_EXCLUIR and not c.startswith("ocupacion_")
    ]
    X = df[feature_cols].astype(float).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X, X_scaled, feature_cols


def valor_json(valor):
    if pd.isna(valor):
        return None
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return float(valor)
    return valor


def calcular_embedding_pca(X):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_.tolist()
    loadings = pca.components_.tolist()
    return coords.tolist(), var_exp, loadings


def calcular_embedding_tsne(X):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(X)
    return coords.tolist()


def calcular_embedding_umap(X):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(X)
    return coords.tolist()


def calcular_embedding_mds(X):
    mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean", max_iter=300)
    coords = mds.fit_transform(X)
    return coords.tolist()


def _cache_embeddings(X_scaled):
    """Calcula PCA + t-SNE + UMAP + MDS una sola vez y cachea el resultado en disco.

    Reproducibilidad / determinismo:
      - Las 4 tecnicas usan random_state=42 (semillas fijas), por lo que sus
        resultados son deterministas para los mismos datos de entrada.
      - La clave del cache es un hash del array escalado; si los datos cambian,
        el cache se invalida y se recalcula automaticamente.
      - t-SNE y UMAP son costosos: con el cache solo se ejecutan la primera vez
        y en las recargas siguientes la vista es identica (estable) e inmediata.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    clave = hashlib.md5(np.ascontiguousarray(X_scaled).tobytes()).hexdigest()[:16]
    cache_file = CACHE_DIR / ("embeddings_" + clave + ".json")
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass  # cache corrupto -> recalcular
    pca_coords, var_exp, pca_loadings = calcular_embedding_pca(X_scaled)
    resultado = {
        "pca_coords": pca_coords,
        "var_exp": var_exp,
        "pca_loadings": pca_loadings,
        "tsne": calcular_embedding_tsne(X_scaled),
        "umap": calcular_embedding_umap(X_scaled),
        "mds": calcular_embedding_mds(X_scaled),
    }
    try:
        cache_file.write_text(json.dumps(resultado), encoding="utf-8")
    except Exception:
        pass  # si no se puede escribir, se sirve sin cache
    return resultado


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    df, X, X_scaled, feature_cols = cargar_datos()

    emb = _cache_embeddings(X_scaled)
    pca_coords, var_exp, pca_loadings = emb["pca_coords"], emb["var_exp"], emb["pca_loadings"]
    loadings_list = [
        {
            "feature": col,
            "label": ATRIBUTOS_LABELS.get(col, col),
            "pc1": round(float(pca_loadings[0][i]), 4),
            "pc2": round(float(pca_loadings[1][i]), 4),
        }
        for i, col in enumerate(feature_cols)
    ]
    tsne_coords = emb["tsne"]
    umap_coords = emb["umap"]
    mds_coords = emb["mds"]

    histogram_features = [
        {"key": "edad_norm", "label": "Edad"},
        {"key": "duracion_sueno_horas_norm", "label": "Duración sueño"},
        {"key": "calidad_sueno_norm", "label": "Calidad sueño"},
        {"key": "nivel_estres_norm", "label": "Nivel estrés"},
        {"key": "frecuencia_cardiaca_norm", "label": "Freq. cardíaca"},
        {"key": "deficit_sueno_norm", "label": "Déficit sueño"},
        {"key": "eficiencia_sueno_norm", "label": "Eficiencia sueño"},
        {"key": "puntaje_salud_sueno_norm", "label": "Salud sueño"},
    ]

    registros = []
    for i, (_, row) in enumerate(df.iterrows()):
        r = {
            "id": int(row["id_persona"]),
            "id_persona": int(row["id_persona"]),
            "trastorno": valor_json(row["trastorno_sueno"]),
        }
        for col in ATRIBUTOS_BARRA:
            r[col] = float(row[col]) if col in row and not pd.isna(row[col]) else 0.0
        for col in COLUMNAS_DETALLE:
            if col in row and col not in {"id_persona", "trastorno_sueno"}:
                r[col] = valor_json(row[col])
        r["mds_x"] = float(mds_coords[i][0])
        r["mds_y"] = float(mds_coords[i][1])
        r["alertas"] = generar_alertas_clinicas(r)
        registros.append(r)

    bar_atributos = [
        {"key": k, "label": ATRIBUTOS_LABELS.get(k, k)}
        for k in ATRIBUTOS_BARRA
    ]

    return jsonify({
        "registros": registros,
        "feature_cols": feature_cols,
        "pca": {"coords": pca_coords, "varianza_explicada": var_exp, "loadings": loadings_list},
        "tsne": {"coords": tsne_coords},
        "umap": {"coords": umap_coords},
        "mds": {"coords": mds_coords},
        "colores_trastorno": COLORES_TRASTORNO,
        "bar_atributos": bar_atributos,
        "histogram_features": histogram_features,
    })


@app.route("/api/metodologia")
def api_metodologia():
    """Ficha de metodología: parámetros y métricas de cada técnica, pensada
    para que un especialista pueda documentar/citar el análisis."""
    df, X, X_scaled, feature_cols = cargar_datos()
    _, var_exp_live, _ = calcular_embedding_pca(X_scaled)

    metodologia = {
        "pca_vivo": {
            "n_muestras": int(len(df)),
            "n_features": len(feature_cols),
            "features": feature_cols,
            "varianza_pc1": round(float(var_exp_live[0]), 4),
            "varianza_pc2": round(float(var_exp_live[1]), 4),
            "varianza_total": round(float(var_exp_live[0] + var_exp_live[1]), 4),
        },
        "tsne": {"perplexity": 30, "max_iter": 1000},
        "umap": {"n_neighbors": 15, "min_dist": 0.1},
        "mds": {"dissimilarity": "euclidean", "max_iter": 300},
        "kmeans": None,
        "dbscan": None,
    }

    kmeans_path = BASE_DIR / "resultados_clustering.json"
    if kmeans_path.exists():
        k = json.loads(kmeans_path.read_text(encoding="utf-8"))
        metodologia["kmeans"] = {
            "algoritmo": k.get("algoritmo"),
            "n_muestras": k.get("n_muestras"),
            "n_features": k.get("n_features"),
            "k_optimo": k.get("k_optimo"),
            "criterio_k": k.get("criterio_k"),
            "silueta_promedio": k.get("silueta_promedio"),
            "varianza_pca": k.get("varianza_pca"),
            "metricas_validacion": k.get("metricas_validacion"),
        }

    dbscan_path = BASE_DIR / "resultados_dbscan.json"
    if dbscan_path.exists():
        d = json.loads(dbscan_path.read_text(encoding="utf-8"))
        metodologia["dbscan"] = {
            "algoritmo": d.get("algoritmo"),
            "n_muestras": d.get("n_muestras"),
            "n_features": d.get("n_features"),
            "parametros": d.get("parametros"),
            "criterio_eps": d.get("criterio_eps"),
            "criterio_min_samples": d.get("criterio_min_samples"),
            "n_clusters": d.get("n_clusters"),
            "n_anomalias": d.get("n_anomalias"),
            "porcentaje_anomalias": d.get("porcentaje_anomalias"),
            "metricas_validacion": d.get("metricas_validacion"),
        }

    return jsonify(metodologia)


@app.route("/api/alertas/<int:id_persona>")
def api_alertas(id_persona):
    """Nivel 1 – Alertas clínicas para un empleado existente por ID."""
    df, _, _, _ = cargar_datos()
    row = df[df["id_persona"] == id_persona]
    if row.empty:
        return jsonify({"error": "Empleado no encontrado"}), 404
    datos = row.iloc[0].to_dict()
    alertas = generar_alertas_clinicas(datos)
    return jsonify({"id_persona": id_persona, "alertas": alertas})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Nivel 2 – Evaluación de nuevo paciente.
    Recibe JSON con los campos clínicos y devuelve alertas + clasificación
    basada en reglas (sin necesidad de re-entrenar el modelo en runtime).
    """
    datos = request.get_json(silent=True)
    if datos is None:
        return jsonify({"error": "El cuerpo debe ser JSON valido (usa Content-Type: application/json)."}), 400
    if not isinstance(datos, dict):
        return jsonify({"error": "El JSON debe ser un objeto con los campos clinicos del paciente."}), 400

    campos_numericos = ["duracion_sueno_horas", "calidad_sueno", "nivel_estres", "frecuencia_cardiaca"]
    rangos = {"duracion_sueno_horas": (0, 24), "calidad_sueno": (0, 10), "nivel_estres": (0, 10), "frecuencia_cardiaca": (20, 250)}
    presentes = [x for x in campos_numericos if datos.get(x) not in (None, "")]
    if not presentes and not datos.get("categoria_imc") and not datos.get("presion_arterial"):
        return jsonify({"error": "Faltan datos clinicos minimos. Envia al menos alguno de: " + ", ".join(campos_numericos + ["categoria_imc", "presion_arterial"]) + "."}), 400
    errores = {}
    for x in campos_numericos:
        v = datos.get(x)
        if v in (None, ""):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            errores[x] = "debe ser numerico"
            continue
        lo, hi = rangos[x]
        if fv < lo or fv > hi:
            errores[x] = "fuera de rango (" + str(lo) + "-" + str(hi) + ")"
    if errores:
        return jsonify({"error": "Valores invalidos en la evaluacion.", "detalles": errores}), 400

    alertas = generar_alertas_clinicas(datos)

    # Clasificación heurística basada en reglas (interpretación explicable)
    codigos = {a["codigo"] for a in alertas}
    if "APNEA_IMC_PRESION" in codigos:
        prediccion = "Apnea del sueño"
        confianza = "Alta"
        explicacion = "IMC elevado y presión sistólica ≥ 130 son los predictores #1 y #5 del modelo GBM entrenado."
    elif "INSOMNIO_SUENO_ESTRES" in codigos:
        prediccion = "Insomnio"
        confianza = "Alta"
        explicacion = "Déficit de sueño + estrés ≥ 7 activa el patrón conductual del modelo."
    elif alertas:
        prediccion = "Riesgo moderado"
        confianza = "Media"
        explicacion = "Se detectaron factores de riesgo que requieren seguimiento."
    else:
        prediccion = "Ninguno"
        confianza = "Alta"
        explicacion = "No se detectaron patrones clínicos de riesgo en las variables evaluadas."

    return jsonify({
        "prediccion": prediccion,
        "confianza": confianza,
        "explicacion": explicacion,
        "alertas": alertas,
        "total_alertas": len(alertas),
    })


@app.route("/api/riesgo-global")
def api_riesgo_global():
    """
    Nivel 3 – Resumen de riesgo para toda la organización.
    Cuenta cuántos empleados activan cada regla clínica.
    """
    df, _, _, _ = cargar_datos()

    resumen = {
        "total_empleados": int(len(df)),
        "con_alertas": 0,
        "sin_alertas": 0,
        "por_tipo": {"peligro": 0, "advertencia": 0, "info": 0},
        "por_regla": {},
        "empleados_alto_riesgo": [],  # los que tienen ≥ 2 alertas de tipo peligro/advertencia
    }

    for _, row in df.iterrows():
        datos = row.to_dict()
        alertas = generar_alertas_clinicas(datos)

        if alertas:
            resumen["con_alertas"] += 1
        else:
            resumen["sin_alertas"] += 1

        for a in alertas:
            tipo = a.get("tipo", "info")
            resumen["por_tipo"][tipo] = resumen["por_tipo"].get(tipo, 0) + 1
            codigo = a.get("codigo", "DESCONOCIDO")
            resumen["por_regla"][codigo] = resumen["por_regla"].get(codigo, 0) + 1

        alertas_criticas = [a for a in alertas if a["tipo"] in ("peligro", "advertencia")]
        if len(alertas_criticas) >= 2:
            resumen["empleados_alto_riesgo"].append({
                "id_persona": int(row.get("id_persona", 0)),
                "ocupacion": str(row.get("ocupacion", "")),
                "trastorno_sueno": str(row.get("trastorno_sueno", "")),
                "num_alertas": len(alertas_criticas),
            })

    resumen["pct_con_alertas"] = round(resumen["con_alertas"] / resumen["total_empleados"] * 100, 1)
    return jsonify(resumen)


@app.route("/api/ml/results")
def api_ml_results():
    path = BASE_DIR / "resultados_ml.json"
    if not path.exists():
        return jsonify({"error": "Ejecuta pipeline_ml.py primero"}), 404
    data = json.loads(path.read_text(encoding="utf-8"))
    return jsonify(data)


@app.route("/api/clustering")
def api_clustering():
    """Segmentacion no supervisada (K-Means). Lee resultados_clustering.json.
    Genera ese archivo ejecutando:  python clustering_kmeans.py
    """
    path = BASE_DIR / "resultados_clustering.json"
    if not path.exists():
        return jsonify({"error": "Ejecuta clustering_kmeans.py primero"}), 404
    data = json.loads(path.read_text(encoding="utf-8"))
    return jsonify(data)


@app.route("/api/dbscan")
def api_dbscan():
    """Clustering por densidad (DBSCAN) + deteccion de anomalias.
    Lee resultados_dbscan.json. Genera ese archivo ejecutando:
        python clustering_dbscan.py
    """
    path = BASE_DIR / "resultados_dbscan.json"
    if not path.exists():
        return jsonify({"error": "Ejecuta clustering_dbscan.py primero"}), 404
    data = json.loads(path.read_text(encoding="utf-8"))
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
