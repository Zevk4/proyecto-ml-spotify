## Estructura del Proyecto

```text
proyecto-ml-spotify/
│
├── README.md                           # Informe técnico ejecutivo en Markdown (documento central)
├── requirements.txt                    # Dependencias congeladas (pandas, scikit-learn, joblib, etc.)
├── .gitignore                          # Exclusión de checkpoints de Jupyter, entornos virtuales, etc.
│
├── data/                               # Gestión de conjuntos de datos
│   ├── raw/
│   │   └── Spotify_Tracks_Dataset.csv  # Dataset crudo original
│   ├── processed/
│   │   └── spotify_clean.csv           # Dataset sin duplicados (89.741 registros)
│   └── segmentation/                   # Archivos generados en la fase de clustering
│       ├── X_segmentacion_original.csv
│       ├── X_segmentacion_escalado.csv
│       └── contexto_segmentacion.csv
│
├── notebooks/                          # Cuadernos ordenados según el flujo de trabajo
│   ├── 01_eda_limpieza_datos.ipynb     # Auditoría SHA-256, deduplicación, análisis estadístico y boxplots
│   ├── 02_clustering_kmeans.ipynb      # Estandarización, Elbow, Silhouette, K-Means (K=4) y PCA
│   ├── 03_modelos_clasificacion.ipynb  # Decision Tree (binario y 3 clases), evaluación y overfitting
│   └── 04_modelos_regresion.ipynb      # DecisionTreeRegressor y HistGradientBoostingRegressor (Base vs Tuned)
│
├── models/                             # Modelos y pipelines serializados reproducibles
│   ├── preprocessor_clustering.joblib  # Pipeline de preprocesamiento no supervisado
│   ├── kmeans_k4_model.joblib          # Modelo K-Means definitivo
│   ├── pipeline_decision_tree_reg.joblib # Mejor árbol de regresión ajustado
│   └── pipeline_hist_gradient_boost.joblib # Pipeline optimizado de boosting
│
├── reports/                            # Entregables formales y material de apoyo
│   ├── informe_tecnico_ejecutivo.md    # Copia extendida o modular del informe (CRISP-DM)
│   └── presentacion_defensa.pdf        # Diapositivas para la defensa de 10 minutos
│
└── images/                             # Gráficos y diagramas exportados para el README
    ├── pie_duplicados.png              # Gráfico de torta de duplicados
    ├── boxplots_acusticos.png          # Boxplots de outliers
    ├── elbow_method.png                # Curva del método del codo
    ├── silhouette_analysis.png         # Curva de Silhouette score
    ├── heatmap_centroides.png          # Mapa de calor de centroides
    ├── pca_clusters_2d.png             # Proyección 2D de clusters
    └── comparativa_metricas_regresion.png # Gráfico de barras MAE vs R2
```
