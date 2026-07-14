# Guion de exposición — Estudio de casos (4 patrones con la herramienta)

Criterio de evaluación: el estudio de casos es el componente con más peso.
Se necesitan **mínimo 4 patrones consistentes**, cada uno demostrado en vivo con el dashboard.

---

## Patrón 1 — Segmentación de perfiles de salud del sueño (K-Means)
**Qué es:** el algoritmo agrupó a los 374 pacientes en **7 clusters** (silueta = 0.58, Davies-Bouldin = 0.67, Calinski-Harabasz = 249.6), cada uno con un perfil de sueño/estrés/actividad distinto.

**Patrón consistente a resaltar:** los clusters más grandes están dominados por un trastorno específico:
- Cluster 2 → 102 pacientes (27.3%) → dominante **Insomnio**
- Cluster 5 → 97 pacientes (25.9%) → dominante **Ninguno**
- Cluster 3/4 → dominante **Apnea del sueño**

**Cómo mostrarlo en vivo:**
1. Menú **📊 Análisis → Clustering**.
2. Mostrar método del codo + silueta (justifica por qué k=7).
3. En el scatter PCA, usar **◀ Anterior / Siguiente ▶** para recorrer cluster por cluster; la región activa se resalta y la tarjeta de perfil correspondiente se ilumina automáticamente.
4. Comentar 2-3 clusters contrastantes (ej. Cluster 2 vs Cluster 5).

---

## Patrón 2 — Detección de anomalías / casos atípicos (DBSCAN)
**Qué es:** DBSCAN (eps=0.60, min_samples=21) encontró **10 clusters densos** y marcó **77 pacientes (20.6%)** como anomalías: casos que no encajan en ningún perfil típico.

**Patrón consistente a resaltar:** las anomalías no son errores de datos, son pacientes con combinaciones inusuales de variables (ej. buena calidad de sueño pero estrés muy alto) — útiles como alerta clínica temprana.

**Cómo mostrarlo en vivo:**
1. Menú **📊 Análisis → DBSCAN**.
2. Mostrar el gráfico de distancia-k (cómo se eligió eps).
3. En el scatter, señalar las **cruces negras** = anomalías, y navegar clusters con los mismos botones ◀/▶.
4. Abrir la tabla **"Anomalías más alejadas"** y explicar la distancia al cluster más cercano de 2-3 casos.

---

## Patrón 3 — Relación entre variables de riesgo y trastorno (correlación/contraste)
**Qué es:** patrón repetido entre variables clínicas y el trastorno del sueño (ej. estrés alto + baja calidad de sueño → mayor prevalencia de Insomnio; IMC/presión arterial elevados → asociados a Apnea del sueño).

**Cómo mostrarlo en vivo:**
1. En el dashboard principal, usar el gráfico de **contraste/comparación** y los filtros por trastorno.
2. Cambiar el filtro entre "Insomnio", "Apnea del sueño" y "Ninguno" y mostrar cómo cambian las distribuciones de estrés/calidad de sueño.
3. Usar el buscador por ID de paciente para mostrar un caso puntual que confirme el patrón.

---

## Patrón 4 — Predicción individual consistente (modelo evaluador)
**Qué es:** el modelo de clasificación (pipeline ML) detecta el mismo patrón de forma consistente: dado un conjunto de indicadores (presión, sueño, estrés, actividad, frecuencia cardiaca), predice el trastorno más probable con una confianza y explicación.

**Cómo mostrarlo en vivo:**
1. Menú **📊 Análisis → Evaluar**.
2. Ingresar 2-3 casos distintos (uno saludable, uno de riesgo de insomnio, uno de riesgo de apnea) y mostrar que la predicción cambia de forma coherente con las variables.
3. Abrir el **Historial de evaluaciones** (🕓) para mostrar todos los casos evaluados juntos y remarcar la consistencia del patrón entre ellos.

---

## Cierre sugerido
Resumir que los 4 patrones fueron encontrados con técnicas de minería de datos distintas (clustering K-Means, detección de anomalías DBSCAN, análisis de correlación, y un modelo predictivo) y que **todos son reproducibles en vivo dentro del mismo dashboard**, lo que demuestra que la herramienta soporta el ciclo completo de descubrimiento de conocimiento (KDD).


---

# Walkthrough real — demostración de uso end-to-end

> **Idea central:** no se enseña el dashboard "botón por botón", sino que **una usuaria se sienta a responder preguntas reales** y, al usar las vistas, va **descubriendo patrones** y saliendo con **insights transferibles** (que sirven fuera de este dataset).

**Persona:** *María*, analista de bienestar laboral. Tiene los datos de **374 empleados** y una sola pregunta de negocio: **¿a quién interviene primero y con qué acción?**

Recorre 5 tareas. En cada una: **qué pregunta se hace → qué vista usa → qué patrón descubre → qué insight se lleva.**

---

## Tarea 1 — ¿Existen grupos naturales de empleados? (Cuadrante 1 + marquesina)

**Vista:** Cuadrante 1 (espacio latente PCA). Las 11 variables de salud/sueño se comprimen a 2D conservando **~63% de la varianza** (PC1 ≈ 38%, PC2 ≈ 25%).

**Qué hace:** en lugar de clicar punto por punto, **arrastra la marquesina** sobre la nube superior-izquierda y selecciona de golpe ~30 puntos. El Cuadrante 3 (coordenadas paralelas) y el Cuadrante 4 (perfil) se actualizan con esa cohorte.

**Patrón que descubre:** los puntos **no están mezclados al azar**: forman nubes separadas. La cohorte que rodeó comparte perfil (sueño ≈ 8.4 h, calidad 9/10, estrés 3/10) y casi todos son **Ingenieros/as sanos**.

**Insight transferible:** *la estructura 2D ya revela segmentos antes de correr ningún algoritmo*. Una reducción de dimensionalidad bien hecha convierte "20 columnas" en un mapa navegable — técnica reutilizable en cualquier dataset multivariable.

---

## Tarea 2 — ¿Qué separa a los sanos de los de riesgo? (Cuadrante 1 vs 2 comparativa + marquesina)

**Vista:** modo **comparación manual** (Cuadrante 4 → pestaña Comparar). Marca **Grupo A** arrastrando la marquesina sobre la zona sana del Cuadrante 1, y **Grupo B** sobre la zona densa central del Cuadrante 2.

**Patrón que descubre:** el contraste automático muestra que los dos grupos se diferencian sobre todo en **estrés (≈ 3 vs ≈ 7/10)** y **puntaje de salud del sueño (≈ 8.9 vs ≈ 5.6)**, no tanto en edad. La zona central es una **"zona gris"** donde conviven Insomnio, Ninguno y algo de Apnea (el Cluster 2 real: 45 / 44 / 13).

**Insight transferible:** *el eje discriminante es el estrés combinado con el déficit de sueño*, y las zonas de solapamiento en el mapa son exactamente los **casos límite** que ningún modelo clasifica con seguridad. Comparar cohortes seleccionadas a mano es una forma rápida de validar hipótesis sin escribir código.

---

## Tarea 3 — ¿Cuántos perfiles hay y quién domina cada uno? (Análisis → Clustering K-Means)

**Vista:** panel de Clustering. El codo + silueta justifican **k = 7** (silueta 0.58). Recorre los clusters con ◀ / ▶.

**Patrón que descubre — los perfiles se alinean con la ocupación:**
- **C0** (Ingenieros/as, n=32): sueño 8.4 h, estrés 3 → sanos consolidados (silueta 0.95).
- **C1** (Docentes, n=28): sueño 6.6 h, déficit 1.4 h → **Insomnio** por privación de sueño.
- **C2** (Doctores/Vendedores, n=102): estrés 7.2, calidad 6 → la **zona gris** de mayor riesgo mezclado.
- **C3 y C4** (Enfermeros/as, n=35 y 32): dominante **Apnea del sueño** pese a buena actividad física.
- **C5 y C6** (Abogados/Contadores, n=97 y 48): mayormente **Ninguno**.

**Insight transferible:** *el trastorno no es aleatorio: se agrupa por ocupación y estilo de vida* (enfermeros → apnea; docentes → insomnio; ingenieros → sanos). Esto permite diseñar intervenciones **por rol**, no individuales.

---

## Tarea 4 — ¿Hay casos que no encajan en ningún perfil? (Análisis → DBSCAN)

**Vista:** panel DBSCAN (eps = 0.60, min_samples = 21). Marca **77 empleados (20.6%)** como anomalías (cruces negras) sobre 10 clusters densos.

**Patrón que descubre:** las anomalías más alejadas (ej. IDs 4, 5 a distancia 6.37 del cluster más cercano) **no son errores de datos**, sino combinaciones inusuales (buena calidad de sueño + estrés muy alto, o apnea con hábitos saludables).

**Insight transferible:** *la detección por densidad sirve como alerta clínica temprana*: en vez de revisar 374 fichas, el especialista revisa manualmente solo el ~20% atípico. Es una técnica de **triaje** aplicable a fraude, fallas o control de calidad.

---

## Tarea 5 — ¿Puedo evaluar a un empleado nuevo? (Análisis → Evaluar + modelo ML)

**Vista:** formulario de evaluación. Ingresa 3 casos (sano / riesgo insomnio / riesgo apnea) y observa que la predicción y las alertas cambian de forma coherente. El modelo (Gradient Boosting) logra **accuracy 0.88, F1 0.88, AUC 0.93**.

**Patrón que descubre:** la importancia de variables confirma lo visto en las tareas anteriores — el **IMC es el predictor #1 (0.73)**, seguido de la **interacción sueño×actividad (0.22)**.

**Insight transferible:** *las palancas accionables son el IMC y el balance sueño/actividad*, no la edad (importancia < 0.01). El mismo patrón aparece en el mapa (Tarea 1-2), en los clusters (Tarea 3) y en el modelo (Tarea 5): **triangulación = confianza en el hallazgo**.

---

## Cierre del walkthrough

María entró con una pregunta de negocio y salió con un **plan priorizado**: intervenir el Cluster 2 (zona gris, estrés alto), derivar el 20.6% atípico a evaluación manual, y montar programas por ocupación. Lo importante: **no memorizó la herramienta, usó las vistas para pensar**. Ese es el criterio de un buen dashboard — soporta el ciclo completo de descubrimiento de conocimiento (KDD) y deja insights que se transfieren a otros datasets y dominios.
