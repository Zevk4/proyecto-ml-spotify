# Inteligencia Musical y Predicción de Popularidad de Canciones (Spotify Tracks)

Proyecto integral de Machine Learning aplicado a la industria del streaming musical. A través de la metodología CRISP-DM, se abordan la segmentación no supervisada del catálogo, la clasificación de tracks por niveles de demanda y la regresión continua del puntaje de popularidad, mitigando el riesgo de sobreajuste y previniendo la fuga de datos (*data leakage*).

---

## 1. Descripción del Problema de Negocio

En un catálogo masivo como el provisto por **Spotify Tracks**, evaluar manualmente cada pista para tomar decisiones de curaduría, inclusión en listas editoriales y distribución presupuestaria resulta inviable operativa y económicamente. Asimismo, las características acústicas individuales (tempo, bailabilidad, volumen) no garantizan de forma univariada el éxito de una composición.

El problema consiste en determinar hasta qué punto las señales acústicas y los metadatos disponibles permiten modelar la popularidad y extraer arquetipos sonoros homogéneos que sirvan de soporte analítico.

### Delimitación y Alcance Ético

* **Apoyo analítico, no determinismo:** Los modelos no aseguran el éxito futuro ni certifican que una campaña incrementará reproducciones de forma causal.
* **Factores externos ausentes:** La popularidad en streaming depende fuertemente de variables no contenidas en el dataset (presupuesto publicitario, exposición en redes sociales, trayectoria del artista o viralidad coyuntural).
* **Separación cualitativa:** La métrica de popularidad refleja tracción de consumo dentro de la plataforma, mas no la calidad artística o el valor cultural de una obra.

---

## 2. Objetivos del Proyecto

* **Objetivo General:** Desarrollar, optimizar y comparar modelos de aprendizaje automático supervisado (clasificación y regresión) y no supervisado (clustering) sobre el catálogo de Spotify para orientar la selección estratégica y el análisis estructural del contenido.
* **Objetivos Específicos:**
1. Ejecutar una auditoría de integridad criptográfica (SHA-256), depuración de duplicados por pista y tratamiento justificado de *outliers* sin distorsionar la variabilidad musical real.
2. Implementar un modelo de agrupamiento con **K-Means** sobre variables acústicas estandarizadas, determinando el número óptimo de clusters mediante el análisis combinado del método del codo y el coeficiente de silueta.
3. Construir y calibrar modelos de **clasificación multiclase** (Decision Tree vs. Random Forest) para identificar pistas de popularidad alta, priorizando la tasa de recuperación de hits.
4. Implementar y evaluar modelos de **regresión continua** (Decision Tree vs. HistGradientBoosting) para predecir el índice exacto de popularidad (0–100) minimizando el error absoluto.
5. Formular recomendaciones de negocio e hipótesis operativas a partir de los patrones descubiertos, previniendo sesgos de exclusión sistemática de géneros subrepresentados.

---

## 3. Definición de KPIs de Negocio

Para medir el impacto técnico y la utilidad analítica de cada componente según los requerimientos de evaluación, se establecen tres indicadores clave:

| Área Analítica | KPI Primario | Definición Técnica | Propósito en el Negocio |
| --- | --- | --- | --- |
| **Clasificación** | **Recall (Popularidad Alta)** | $\frac{TP_{\text{alta}}}{TP_{\text{alta}} + FN_{\text{alta}}}$ | Medir la proporción de canciones exitosas reales que el sistema logra rescatar para evaluación prioritaria (minimizar falsos descartes). |
| **Regresión** | **MAE (Mean Absolute Error)** | $\frac{1}{n}\sum_{i=1}^{n} \vert{}y_i - \hat{y}_i\vert{}$ | Cuantificar la desviación promedio de la estimación en puntos directos de la escala de popularidad (0 a 100), complementado con $R^2$. |
| **Segmentación** | **Silhouette Score** | $\frac{b(i) - a(i)}{\max(a(i), b(i))}$ | Evaluar la cohesión intra-cluster frente a la separación inter-cluster de los perfiles acústicos descubiertos. |

---

## 4. Fuentes de Datos y Análisis Exploratorio (EDA)

### Origen y Calidad de los Datos

* **Dataset:** *Spotify Tracks Dataset* (`Spotify_Tracks_Dataset.csv`), compuesto originalmente por 114.000 observaciones y 21 atributos descriptivos y acústicos.
* **Auditoría e Integridad:** Verificación criptográfica mediante hash SHA-256 (`b202fa4990c...`) en la ingesta para garantizar reproducibilidad e inmutabilidad.
* **Deduplicación:** Se eliminaron 24.259 registros duplicados basados en `track_id` (conservando la primera aparición), consolidando un corpus maestro de **89.741 pistas únicas**. Esto impide que temas repetidos por variación de género aparezcan simultáneamente en entrenamiento y prueba, eliminando el riesgo de fuga de datos.
* **Limpieza de Columnas:** Exclusión de índices espurios (`Unnamed: 0`) y atributos nominales de alta cardinalidad con valores nulos marginales (`artists`, `album_name`, `track_name`), los cuales no ingresan al espacio de características para evitar sesgos de memorización.

### Hallazgos del EDA

* **Distribución de Popularidad:** Media de 33,2 puntos y mediana de 33 puntos, mostrando una asimetría positiva donde los temas altamente exitosos constituyen una minoría dentro de la plataforma.
* **Tratamiento de Outliers:** Se detectaron distribuciones asimétricas en `duration_ms` (pistas instrumentales de hasta 80 minutos), `speechiness` y `liveness`. Se decidió **no eliminar ni truncar outliers**; en música, pistas extensas o grabaciones en vivo constituyen variaciones legítimas del dominio y no errores de captura.
* **No-linealidad y Multicolinealidad:** Ninguna característica acústica individual presentó correlación de Spearman superior a 0,13 frente a `popularity`. Adicionalmente, se retiró la variable `energy` del conjunto predictor por redundancia y alta colinealidad con `loudness` y `acousticness`.

---

## 5. Metodología Utilizada (CRISP-DM)

El proyecto adoptó las seis fases del estándar de minería de datos CRISP-DM:

```mermaid
flowchart LR
    A[Comprensión del Negocio] --> B[Comprensión de los Datos]
    B --> C[Preparación de Datos]
    C --> D[Modelado]
    D --> E[Evaluación]
    E --> F[Despliegue y Reproducibilidad]
    E -. Retroalimentación .-> C

```

1. **Comprensión del Negocio:** Formulación del problema de estimación y segmentación de catálogo, delimitación de KPIs operativos y consideración del impacto ético/comercial.
2. **Comprensión de los Datos:** Exploración univariada y multivariada, análisis de asimetrías, detección de anomalías y cálculo de correlaciones monótonas.
3. **Preparación de Datos:** Deduplicación estricta por identificador, división estratificada 80/20 antes del modelado supervisado, codificación de variables nominales (`OneHotEncoder`) y escalado con `StandardScaler` reservado para K-Means.
4. **Modelado:** Construcción de pipelines de segmentación (K-Means + PCA), clasificación (Decision Tree y Random Forest) y regresión continua (Decision Tree e HistGradientBoosting) con optimización mediante `RandomizedSearchCV`.
5. **Evaluación:** Contraste técnico contra KPIs predefinidos, cálculo de matrices de confusión, estudio de residuos e inspección de sobreajuste entre splits de *train* y *test*.
6. **Despliegue y Reproducibilidad:** Exportación de artefactos entrenados (`.joblib` / `.zip`), trazabilidad de código y estructuración de reportes ejecutivos.

---

## 6. Modelos Implementados, Métricas e Interpretación

### 6.1. Aprendizaje No Supervisado: Segmentación Musical (K-Means)

* **Espacio de Características:** 9 atributos acústicos continuos (`duration_ms`, `danceability`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`) imputados por mediana y escalados con `StandardScaler`. Se excluyeron variables categóricas y `popularity` para que los grupos respondieran puramente a la naturaleza musical y no al éxito comercial.
* **Selección de $K$:** La inercia geométrica determinó un punto de inflexión en $K = 4$. Aunque $K = 2$ maximiza el coeficiente de silueta (0,2474), una división binaria resulta poco informativa para la curaduría. Se adoptó $K = 4$ (Silhouette de 0,1596) como el balance adecuado entre cohesión y granularidad de negocio.

| Cluster | % Catálogo | N° Pistas | Perfil Acústico Dominante | Géneros Representativos | Plan de Acción / Curaduría |
| --- | --- | --- | --- | --- | --- |
| **0** | 15,46% | 13.870 | Acústico, instrumental, bajo tempo y sonoridad suave. | *sleep*, *ambient*, *classical*, *new-age* | Playlists de descanso, concentración y estudio; monitoreo de tiempo de permanencia. |
| **1** | 8,11% | 7.277 | Grabaciones en vivo, alta presencia de voz hablada. | *comedy*, *pagode*, *sertanejo*, *samba* | Recomendación de sesiones en vivo y shows acústicos; control de skip rate. |
| **2** | 32,88% | 29.505 | Rápido, enérgico, baja acústica, alta distorsión/volumen. | *black-metal*, *grindcore*, *death-metal*, *drum-and-bass* | Playlists de entrenamiento de alta intensidad, gaming y nichos de tempo elevado. |
| **3** | 43,56% | 39.089 | Alta bailabilidad, valencia positiva, tempo moderado. | *dancehall*, *salsa*, *kids*, *forro* | Inserción en playlists para eventos sociales, virales y campañas de alto engagement. |

> *Nota:* El Cluster 1 absorbió el 47,01% de los valores extremos multivariados ($\vert{}z\vert{} > 3$), aislando de forma natural los contenidos atípicos (podcasts y comedia).

---

### 6.2. Aprendizaje Supervisado: Clasificación de Popularidad

El target continuo se discretizó en 3 niveles de negocio para preservar interpretabilidad: **Baja** (0–24 puntos, 37,2%), **Media** (25–49 puntos, 39,2%) y **Alta** (50–100 puntos, 23,6%). Los modelos evaluados fueron:

| Modelo | Accuracy Test | F1-Score Macro | Recall (Clase Alta) | Precision (Clase Alta) | Diagnóstico Técnico |
| --- | --- | --- | --- | --- | --- |
| **Decision Tree Base** | 62,44% | 0,6104 | 50,52% | 52,46% | Sobreajuste crítico ($F1_{\text{train}} = 0,993$ vs. $F1_{\text{test}} = 0,610$). |
| **Decision Tree Opt.** | 64,28% | 0,6261 | 49,13% | 55,47% | Reduce varianza, pero castiga la detección de hits. |
| **Random Forest Base** | 71,11% | 0,6955 | 54,22% | **68,29%** | Modelo conservador; alta precisión en tracks seleccionados. |
| **Random Forest Opt.** | **71,16%** | **0,7027** | **64,47%** | 62,99% | **Modelo Seleccionado:** Maximiza la recuperación de oportunidades. |

* **Traducción Operativa:** De las 4.242 canciones de alta popularidad del conjunto de test, el Random Forest optimizado rescata **2.735 canciones**. Frente a la versión base, recupera **+435 tracks de alta demanda**, asumiendo un costo de 539 falsas alarmas adicionales. Para una etapa inicial de screening de catálogo, el canje es favorable al negocio.

---

### 6.3. Aprendizaje Supervisado: Regresión Continua

Modelado sobre el puntaje escalar directo de `popularity` (0 a 100) utilizando un pipeline con imputación preventiva y codificación one-hot sin escalado numérico.

| Modelo | MAE Test (Puntos) | $R^2$ Test | Diagnóstico de Ajuste |
| --- | --- | --- | --- |
| **Decision Tree Base** (`max_depth=6`) | 15,81 | 0,10 | **Subajuste (*underfitting*):** Estructura rígida incapaz de capturar varianza. |
| **Decision Tree Optimizado** | 11,82 | 0,31 | Mejora sustancial en MAE, pero retiene sobreajuste ($R^2_{\text{train}} = 0,52$ vs. $R^2_{\text{test}} = 0,31$). |
| **HistGradientBoosting Base** | 11,78 | 0,37 | Gran capacidad de generalización nativa sobre tabular grande. |
| **HistGradientBoosting Optimizado** | **11,15** | **0,40** | **Modelo Seleccionado:** Menor error medio absoluto y máxima varianza explicada. |

* **Factores Relevantes (Permutation Importance):** Al permutar aleatoriamente `track_genre`, el MAE se degrada incrementándose en **~6,924 puntos**, posicionando al género como el predictor más determinante por encima de las métricas acústicas puras.
* **Residuos y Límites:** El modelo presenta sobreestimación en canciones con popularidad exacta de cero y ligera subestimación en canciones por sobre los 85 puntos, confirmando la existencia de factores externos no reflejados en el dataset.

---

## 7. Stack Tecnológico

| Capa | Herramientas | Justificación Técnica |
| --- | --- | --- |
| **Lenguaje / Entorno** | Python 3.10+, Google Colab, Jupyter Notebooks | Ecosistema estándar para computación científica reproducible. |
| **Manipulación y Análisis** | `pandas`, `numpy` | Limpieza, estructuración matricial, agregaciones y cálculo vectorial. |
| **Visualización** | `matplotlib`, `seaborn` | Generación de distribuciones, mapas de calor, matrices de confusión y diagnósticos residuales. |
| **Modelado y Preprocesamiento** | `scikit-learn` (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `KMeans`, `PCA`, `DecisionTreeClassifier`, `RandomForestClassifier`, `DecisionTreeRegressor`, `HistGradientBoostingRegressor`, `RandomizedSearchCV`, `permutation_importance`) | Construcción de flujos de transformación libres de fuga de datos y algoritmos de optimización de hiperparámetros. |
| **Evaluación de Clustering** | `scikit-learn.metrics` (`silhouette_score`, `silhouette_samples`) | Medición cuantitativa de cohesión y separación inter-cluster. |
| **Serialización de Artefactos** | `joblib`, `json`, `zipfile` | Empaquetado binario de pipelines entrenados y registro de hiperparámetros óptimos. |
| **Integridad y Red** | `hashlib` (SHA-256), `requests`, `io` | Validación criptográfica de datos crudos y descarga reproducible. |
| **Control de Versiones** | Git, **Git LFS** (*Large File Storage*) | Trazabilidad de código y almacenamiento versionado de modelos serializados (`.joblib`, `.zip`). |

---

## 8. Recomendaciones para el Negocio

1. **Estrategia Combinada de Cribado (Screening):** Implementar una arquitectura en cascada: utilizar el modelo *Random Forest optimizado* como filtro rápido para no descartar temas con alto potencial comercial, seguido por el modelo *HistGradientBoosting optimizado* para refinar la estimación cuantitativa de popularidad esperada.
2. **Uso de Segmentos Acústicos para Curaduría Contextual:** Aprovechar los 4 grupos identificados por K-Means para poblar playlists temáticas y funcionales basadas en atributos objetivos del audio, sin supeditar la inclusión exclusivamente al nivel de reproducciones del artista.
3. **Mantenimiento del Criterio Humano (*Human-in-the-loop*):** Dado que el MAE promedio se sitúa en ~11 puntos y los algoritmos basados en audio tienden a subestimar temas de nicho con baja exposición previa, las puntuaciones deben emplearse como herramientas de priorización y apoyo a la toma de decisiones, nunca como filtros de descarte automático.

---

## 9. Estructura del Proyecto

```text
proyecto-ml-spotify/
│
├── README.md                                          # Informe técnico del proyecto (CRISP-DM)
├── .gitattributes                                     # Configuración de Git LFS para artefactos binarios
│
├── data/
│   ├── raw/
│   │   └── Spotify_Tracks_Dataset.csv                 # Dataset original (~114k registros)
│   │
│   ├── processed/                                     # Datos depurados y particiones libres de fuga
│   │   ├── Spotify_Tracks_Procesado.csv               # Dataset limpio (89.741 registros únicos)
│   │   ├── X_train_ready.csv                          # Características de entrenamiento (71.793)
│   │   ├── X_test_ready.csv                           # Características de prueba (17.950)
│   │   ├── y_train_ready.csv                          # Target continuo de entrenamiento
│   │   └── y_test_ready.csv                           # Target continuo de prueba
│   │
│   ├── segmentation/                                  # Datos y evaluación de K-Means
│   │   ├── X_segmentacion_original.csv                # 9 variables acústicas sin estandarizar
│   │   ├── X_segmentacion_escalado.csv                # Variables escaladas con StandardScaler
│   │   ├── contexto_segmentacion.csv                  # Metadatos cualitativos desacoplados
│   │   ├── centroides_originales.csv                  # Centroides en unidades físicas de audio
│   │   ├── centroides_estandarizados.csv              # Centroides en escala Z-score
│   │   ├── evaluacion_kmeans.csv                      # Histórico de Inercia y Silueta (K=1 a 10)
│   │   └── resultado_segmentacion.csv                 # Asignación final de cluster por canción
│   │
│   └── regression/                                    # Evaluación del modelado supervisado
│       ├── comparativa_modelos.csv                    # Tabla comparativa de métricas (MAE, R²)
│       ├── importancia_caracteristicas.csv            # Permutation importance de los predictores
│       └── predicciones_test.csv                      # Valores reales vs. predichos en prueba
│
├── images/                                            # Evidencias gráficas e infografías
│   ├── extras/
│   │   ├── clasificacion_de_segmentos.png             # Infografía: caracterización y plan de acción K-Means
│   │   ├── infografia_clasificacion.png               # Infografía: comparativa Decision Tree vs. Random Forest
│   │   ├── infografia_regresion.png                   # Infografía: diagnóstico y métricas de regresión
│   │   └── matriz_de_responsabilidades.png            # Matriz formal de asignación de roles
│   ├── modelo_segmentacion/
│   │   ├── PCA2.png                                   # Proyección bidimensional de clusters
│   │   ├── elbow_method.png                           # Curva de inercia geométrica (Elbow)
│   │   ├── heatmap_centroides.png                     # Mapa de calor de perfiles acústicos
│   │   ├── silhouette_analysis.png                    # Coeficiente de silueta por número de clusters
│   │   ├── top5_generos_predominantes.png             # Distribución de géneros por cluster
│   │   └── varianza_explicada_acumulada.png           # Varianza acumulada por componentes principales
│   ├── modelos_clasificacion/
│   │   ├── desempeño_base_vs_optmizado_dctree.png     # Curvas de ajuste árbol base vs. podado
│   │   ├── distribucion_popularidad_3_grupos.png      # Distribución de clases (Baja, Media, Alta)
│   │   ├── importancia_variables_dctree_optm.png      # Importancia relativa de variables
│   │   ├── matriz_confusion_dctree.png                # Matriz de confusión árbol base
│   │   ├── matriz_confusion_rdforest.png              # Matriz de confusión Random Forest base
│   │   ├── rd_forest_optmizado_matriz_confusion.png   # Matriz de confusión Random Forest optimizado
│   │   └── rforest_base_Vs_optimizado.png             # Comparativa visual de desempeño ensamble
│   └── modelos_regresion/
│       ├── comparativa_metricas_regresion.png         # Comparativa de barras MAE y R²
│       ├── estructura_arbol_base.png                  # Diagrama del árbol de regresión inicial
│       ├── estructura_arbol_optimizado.png            # Diagrama del árbol de regresión regularizado
│       ├── importancia_permutacion_hgb.png            # Importancia de permutación de predictores
│       └── residuos_real_vs_predicho_hgb.png          # Gráfico de dispersión y análisis de residuos
│
├── models/                                            # Artefactos y modelos serializados (Git LFS)
│   ├── Segmentacion_modelo/
│   │   ├── preprocessor_clustering.joblib             # Pipeline de imputación y escalado
│   │   ├── kmeans_final.joblib                        # Modelo K-Means calibrado (K=4)
│   │   ├── pca_completo.joblib                        # Transformador PCA (9 dimensiones)
│   │   └── pca_2d.joblib                              # Transformador PCA (2 dimensiones)
│   ├── Clasificacion_modelos/
│   │   ├── Clasificacion_decision_tree/
│   │   │   ├── decision_tree_multiclase_base.zip
│   │   │   └── decision_tree_multiclase_optimizado.zip
│   │   └── Clasificacion_RandomForest/
│   │       ├── random_forest_multiclase_base.zip
│   │       └── random_forest_multiclase_optimizado.zip
│   └── Regresion_modelos/
│       ├── decision_tree/
│       │   ├── pipeline_decision_tree_optimizado.zip
│       │   └── mejores_hiperparametros_tree.zip
│       └── histgradient_boosting/
│           ├── pipeline_hgb_optimizado.zip            # Pipeline final de regresión continua
│           └── mejores_hiperparametros_hgb.zip        # Metadatos JSON de hiperparámetros óptimos
│
├── notebooks/                                         # Cuadernos Jupyter reproducibles
│   ├── 01_eda_limpieza_datos.ipynb                    # Verificación SHA-256, deduplicación y análisis univariado
│   ├── 02_clustering_kmeans.ipynb                     # Pipeline de K-Means, silueta, codo y proyección PCA
│   ├── 03_modelos_clasificacion.ipynb                 # Clasificación multiclase, optimización y matrices
│   └── 04_modelos_regresion.ipynb                     # Modelado continuo de popularidad y análisis de residuos
│
└── reports/                                           # Entregables técnicos y defensas
    ├── Informe_Machine_Learning_2.pdf                 # Informe técnico consolidado
    ├── Informe_Machine_Learning_2.md                 # Informe técnico consolidado en formato .md
    └── presentacion_modelos_spotify.pptx              # Soporte visual para defensa técnica

```

---

## 10. Instrucciones de Instalación y Reproducibilidad

1. **Clonar el repositorio con soporte para archivos grandes:**
```bash
git clone https://github.com/<tu-usuario>/proyecto-ml-spotify.git
cd proyecto-ml-spotify
git lfs pull

```

2. **Crear y activar un entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate        # En Linux/macOS
# venv\Scripts\activate         # En Windows

```

3. **Instalar dependencias requeridas:**
```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn joblib requests

```

4. **Ejecución secuencial de los notebooks:**
Los cuadernos están ordenados numéricamente para preservar la trazabilidad analítica de los datos:
* `notebooks/01_eda_limpieza_datos.ipynb`: Valida el hash SHA-256 de los datos crudos, remueve duplicados y genera los conjuntos limpios en `data/processed/`.
* `notebooks/02_clustering_kmeans.ipynb`: Ejecuta el escalado, calcula métricas de inercia y silueta, y entrena el modelo de segmentación.
* `notebooks/03_modelos_clasificacion.ipynb`: Construye los pipelines de clasificación binaria y multiclase, ejecutando la validación cruzada y los reportes de clasificación.
* `notebooks/04_modelos_regresion.ipynb`: Ajusta los modelos de regresión continua y genera los gráficos de dispersión de residuos y permutación de características.

---

## 11. Matriz de Responsabilidades

| Integrante | Rol Técnico Específico | Entregable Directo |
| --- | --- | --- |
| **Diego Améstica** | Análisis exploratorio, ingeniería de atributos y modelo de clustering no supervisado (K-Means + PCA). | `02_clustering_kmeans.ipynb`, artefactos de segmentación y documentación de perfiles. |
| **Gino Andrades** | Análisis exploratorio, formulación de pipelines y modelos de regresión continua (Decision Tree vs. HistGradientBoosting). | `04_modelos_regresion.ipynb`, calibración de hiperparámetros y análisis de residuos. |
| **Agustín Cáceres** | Análisis exploratorio, esquemas de estratificación y modelos de clasificación supervisada (Decision Tree vs. Random Forest). | `03_modelos_clasificacion.ipynb`, optimización de métricas de clase alta y matrices de confusión. |

> *Nota:* Para verificar el desglose detallado de las actividades realizadas por cada integrante, consulte la infografía desarrollada, la cual se encuentra en la ruta images/extras/matriz_de_responsabilidades.png.

