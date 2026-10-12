## Estructura del Proyecto

```text
proyecto-ml-spotify/
│
├── README.md                                  # Informe técnico ejecutivo en Markdown (metodología CRISP-DM)
│
├── data/                                      # Gestión de datos, particiones y resultados tabulares
│   ├── raw/
│   │   └── Spotify_Tracks_Dataset.csv         # Dataset original validado con hash SHA-256
│   ├── processed/                             # Datos depurados y particiones preparadas
│   │   ├── spotify_clean.csv                  # Dataset sin duplicados (89.741 registros únicos)
│   │   ├── X_train_ready.csv                  # Características de entrenamiento preparadas
│   │   ├── X_test_ready.csv                   # Características de prueba preparadas
│   │   ├── y_train_ready.csv                  # Variable objetivo de entrenamiento
│   │   └── y_test_ready.csv                   # Variable objetivo de prueba
│   ├── segmentation/                          # Tablas generadas en la etapa de clustering (K-Means)
│   │   ├── X_segmentacion_original.csv        # Variables acústicas sin estandarizar
│   │   ├── X_segmentacion_escalado.csv        # Variables acústicas escaladas (StandardScaler)
│   │   ├── contexto_segmentacion.csv          # Metadatos no utilizados en el entrenamiento
│   │   ├── centroides_originales.csv          # Centroides en escala musical real
│   │   ├── centroides_estandarizados.csv      # Centroides en escala Z-score
│   │   ├── evaluacion_kmeans.csv              # Métricas de Inercia y Silhouette (K=1 a 10)
│   │   └── resultado_segmentacion.csv         # Dataset con asignación final de cluster
│   └── regression/                            # Resultados y evaluaciones del modelado supervisado
│       ├── comparativa_modelos.csv            # Tabla comparativa de métricas (MAE, R2)
│       ├── importancia_caracteristicas.csv    # Relevancia predictiva de cada variable
│       └── predicciones_test.csv              # Comparación de valores reales vs. predichos
│
├── models/                                    # Artefactos serializados y configuraciones óptimas
│   ├── preprocessor_clustering.joblib         # Pipeline de preprocesamiento no supervisado
│   ├── kmeans_final.joblib                    # Modelo K-Means definitivo (K=4)
│   ├── pca_completo.joblib                    # Modelo PCA completo (9 componentes)
│   ├── pca_2d.joblib                          # Modelo PCA proyectado a 2 dimensiones
│   ├── pipeline_decision_tree_optimizado.joblib # Pipeline de Árbol de Regresión ajustado
│   ├── pipeline_hgb_optimizado.joblib         # Pipeline de HistGradientBoostingRegressor optimizado
│   ├── mejores_hiperparametros_tree.json      # Hiperparámetros óptimos del Decision Tree
│   └── mejores_hiperparametros_hgb.json       # Hiperparámetros óptimos del modelo de ensamble
│
├── notebooks/                                 # Cuadernos Jupyter reproducibles
│   ├── 01_eda_limpieza_datos.ipynb            # Auditoría SHA-256, deduplicación y análisis de outliers
│   ├── 02_clustering_kmeans.ipynb             # Segmentación acústica K-Means, Codo, Silueta y PCA
│   ├── 03_modelos_clasificacion.ipynb         # Clasificación en 2 y 3 niveles de popularidad
│   └── 04_modelos_regresion.ipynb             # Regresión para estimar popularidad continua
│
├── reports/                                   # Informes ejecutivos y presentaciones
│   ├── informe_tecnico_ejecutivo.md           # Informe formal estructurado según CRISP-DM
│   └── presentacion_defensa.pdf               # Diapositivas para la defensa técnica (10 minutos)
│
└── images/                                    # Figuras y visualizaciones de soporte para el README
    ├── pie_duplicados.png                     # Proporción de duplicados eliminados
    ├── boxplots_acusticos.png                 # Distribución y asimetría de variables acústicas
    ├── elbow_method.png                       # Curva de inercia del método del codo
    ├── silhouette_analysis.png                # Coeficiente de silueta según número de clusters
    ├── heatmap_centroides.png                 # Perfil relativo de los 4 clusters
    └── comparativa_metricas_regresion.png     # Gráfico comparativo de desempeño de modelos
```
```
