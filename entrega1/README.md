# Entrega 1: integración de datos, etiquetado y línea base

## Contenido

| Actividad | Notebook | Qué hace |
|---|---|---|
| Pipeline | [`notebooks/00_pipeline.ipynb`](notebooks/00_pipeline.ipynb) | CSV → `data/interim/{train,val,test}.parquet` con variables y etiquetas |
| 1.1 Análisis exploratorio | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Sensores por modelo, eventos por componente, telemetría antes de una falla |
| 1.2 Registros de mantenimiento | [`notebooks/02_maintenance.ipynb`](notebooks/02_maintenance.ipynb) | Relación mantenimiento–fallas y riesgo de fuga |
| 1.3 Etiquetas | [`notebooks/03_labels.ipynb`](notebooks/03_labels.ipynb) | Etiqueta binaria y multiclase; proporción de positivos |
| 1.4 Variables | [`notebooks/04_features.ipynb`](notebooks/04_features.ipynb) | Resumen de las 31 variables |
| 1.5 Línea base | [`notebooks/05_baseline.ipynb`](notebooks/05_baseline.ipynb) | Regresión logística y LightGBM en validación |

Salidas:

- **`figures/`:** las 8 figuras de los notebooks. El reporte usa 01, 02, 03, 04 y 08.
- **`results/`:** métricas en validación (`baseline_val_binary.csv`, `baseline_val_confusion.csv`).

## Reproducir

Desde la raíz del repositorio, con el entorno instalado (ver el README principal). `00_pipeline` va primero, porque 04 y 05 leen las tablas que escribe.

```bash
for n in 00_pipeline 01_eda 02_maintenance 03_labels 04_features 05_baseline; do
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 entrega1/notebooks/$n.ipynb
done
```

Tarda varios minutos, sobre todo `05_baseline` por el LightGBM multiclase. Todas las semillas están fijas (42).

## Decisiones

- **Partición temporal fija**:
  - entrenamiento: ene–ago 2015
  - validación: sep–oct 2015
  - prueba: nov–dic 2015

  Se excluyen las últimas 24 h de cada conjunto para que ninguna etiqueta dependa del conjunto siguiente. `test.parquet` no se usa hasta la semana 3.
- **Etiqueta.** `y_bin(t) = 1` si hay una falla en (t, t + 24 h]; `y_cls(t)` indica su componente. Se construye solo con `PdM_failures`.
- **Sin fuga.** Todas las variables usan únicamente información con marca de tiempo ≤ t. `PdM_maint` contiene las fallas (743 de 761 tienen un reemplazo a la misma hora), así que nunca se usa hacia adelante ni agregado por día calendario.

## Resultados (validación)

| Modelo | PR-AUC | ROC-AUC | Recall con precisión ≥ 0.80 |
|---|---|---|---|
| Regresión logística | 0.858 | 0.998 | 0.836 |
| LightGBM | 0.999 | 1.000 | 1.000 |

- ≈ 2 % de las horas son positivas.
- El desempeño casi perfecto de LightGBM no es una fuga, sino un artefacto de la simulación: el error asociado a cada componente aparece exactamente 24 h antes de la falla en el 94 % de los casos, y los sensores cambian en escalón 48 h antes.
