# Autonomous Data Analyst Agent - AI Agent

Agente de análisis de datos que recibe un dataset CSV y ejecuta, de forma autónoma, un pipeline completo de EDA, detección de patrones, inferencia de tarea ML, entrenamiento de modelos baseline, generación de visualizaciones y producción de un reporte orientado a negocio.

---

## ¿Qué hace?

1. **EDA automatizado** — shape, tipos de columna, valores faltantes, duplicados, estadísticas descriptivas y correlaciones.
2. **Detección de patrones** — correlaciones fuertes, columnas con alto porcentaje de nulos, desbalance de clases y outliers.
3. **Inferencia de tarea ML** — decide automáticamente entre regresión, clasificación, clustering o EDA-only según las características del dataset.
4. **Entrenamiento baseline** — modelos simples con métricas apropiadas, sin configuración manual.
5. **Visualizaciones** — heatmap de correlación, distribución del target, valores faltantes, feature importance.
6. **Reporte estructurado** — resumen del dataset, hallazgos de calidad, patrones, resultados de modelos e insights accionables.

---

## Modelos y métricas

| Tarea | Modelos | Métricas |
|---|---|---|
| Regresión | Linear Regression, Random Forest | MAE, RMSE, R² |
| Clasificación | Logistic Regression, Random Forest | Accuracy, Precision, Recall, F1 |
| Clustering | KMeans (k óptimo por silhouette) | Silhouette Score |

---

## Arquitectura

```
src/
├── agent/          # Orquestador: coordina el pipeline completo
├── analysis/       # EDA y detección de patrones
├── modeling/       # Inferencia de tarea ML y entrenamiento
├── reporting/      # Generación del reporte de texto
├── visualization/  # Generación de gráficas
├── utils/          # Configuración y logging
└── main.py         # CLI entry point
```

---


## Tecnologías

- **Python**
- **pandas** — manipulación y análisis de datos
- **scikit-learn** — Machine Learning y métricas
- **matplotlib / seaborn** — visualizaciones
- **PyYAML** — configuración
- **pytest** — testing

Sin APIs de pago. Funciona completamente offline.

---

