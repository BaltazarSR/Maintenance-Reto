# Predicción de fallas en una flota de máquinas

Proyecto del curso *Machine Learning y Deep Learning*: un sistema de alerta temprana que estima, para cada máquina y cada hora, si fallará en las próximas 24 h y qué componente fallará. Se usa el conjunto *Microsoft Azure Predictive Maintenance*.

## Entregas

| Entrega | Contenido | Carpeta | Reporte |
|---|---|---|---|
| 1 | Integración de datos, etiquetado y línea base (actividades 1.1–1.5) | [`entrega1/`](entrega1/README.md) | [`entrega1.pdf`](entrega1/reporte/entrega1.pdf) |

## Estructura

```
README.md
PROJECT.md               Enunciado del proyecto (instrucciones del curso)
requirements.txt
docs/                    Material de referencia (guía de EDA)
data/
  raw/                   CSV originales (PdM_*.csv)
  interim/               Tablas generadas por el pipeline (no versionadas)
src/                     Código compartido entre entregas
  data.py                Carga de datos, constantes y partición temporal
  labels.py              Etiquetas binaria y multiclase (H = 24 h)
  features.py            Variables de la línea base
  metrics.py             PR-AUC, ROC-AUC, recall con precisión ≥ 0.80, matriz de confusión
  plots.py               Estilo y guardado de figuras
entrega1/                Notebooks, figuras, resultados y reporte de la Entrega 1
```

## Instalación

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

En macOS, LightGBM necesita OpenMP: `brew install libomp`.

## Datos

Microsoft. *Azure Predictive Maintenance* (Azure AI Gallery). Los cinco CSV van en `data/raw/`:

- `PdM_telemetry.csv`
- `PdM_errors.csv`
- `PdM_maint.csv`
- `PdM_failures.csv`
- `PdM_machines.csv`

Cómo reproducir cada entrega está en el README de su carpeta.
