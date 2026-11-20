# Entrega 2: modelos secuenciales de deep learning

## Contenido

| Actividad | Archivo | Qué hace |
|---|---|---|
| Experimentos | [`run_experiments.py`](run_experiments.py) | Entrena cada configuración y escribe `results/runs/<nombre>/` (historial, métricas, predicciones en validación, tiempo de anticipación) |
| 2.1 Tensor de entrada | [`notebooks/01_tensor.ipynb`](notebooks/01_tensor.ipynb) | Canales, normalización (solo con train), verificación de causalidad, diagrama |
| 2.2 Arquitecturas | [`notebooks/02_architectures.ipynb`](notebooks/02_architectures.ipynb) | Diagramas de GRU, TCN y CNN-LSTM; comparación contra la línea base; curvas de aprendizaje |
| 2.3 Ventana y desbalance | [`notebooks/03_window_imbalance.ipynb`](notebooks/03_window_imbalance.ipynb) | Efecto de L ∈ {24, 72, 168}; pérdida ponderada vs submuestreo; calibración |
| 2.4 Ajuste y modelo final | [`notebooks/04_tuning_final.ipynb`](notebooks/04_tuning_final.ipynb) | Optuna (20 ensayos), curvas de aprendizaje, F1 por componente, tiempo de anticipación |

Código compartido en `src/`: `windows.py` (tensor), `models.py` (arquitecturas), `train.py` (pérdida, desbalance, entrenamiento) y las métricas nuevas en `metrics.py` (`f1_per_class`, `lead_time`).

Salidas:

- **`figures/`:** las 9 figuras de los notebooks. El reporte usa 01, 02, 03, 05, 07 y 08.
- **`results/`:** tablas en CSV que lee el reporte (`comparison_architectures.csv`, `window_length.csv`, `imbalance.csv`, `f1_per_class.csv`, `lead_time_summary.csv`, `final_summary.csv`, `optuna_trials.csv`), la selección por etapa (`stages.json`, `optuna_best.json`) y una carpeta por corrida en `results/runs/` (las predicciones y los pesos no se versionan).
- **`reporte/entrega2.pdf`:** el reporte de la entrega.

## Reproducir

Desde la raíz del repositorio, con el entorno instalado (ver el README principal) y las tablas `data/interim/*.parquet` generadas por `entrega1/notebooks/00_pipeline.ipynb`.

```bash
.venv/bin/python entrega2/run_experiments.py all --trials 20
for n in 01_tensor 02_architectures 03_window_imbalance 04_tuning_final; do
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 entrega2/notebooks/$n.ipynb
done
```

`run_experiments.py all` ejecuta las cuatro etapas en orden y elige en cada una la configuración con mayor PR-AUC binario en validación: (1) tres arquitecturas con L = 72 y pérdida ponderada; (2) L ∈ {24, 72, 168}; (3) pérdida ponderada vs submuestreo de horas sanas; (4) Optuna, 20 ensayos de hasta 8 épocas con poda, y corrida final. Semilla 14 en todo (PyTorch, NumPy, Optuna, submuestreo). Las corridas sueltas se pueden lanzar con `run` y `tune` (ver la ayuda del script).

## Decisiones

- **Tensor.** Ventana de L horas con 13 canales (4 sensores, 5 indicadores de error, log de horas desde el último reemplazo de cada componente) más modelo (embedding) y antigüedad. Normalización z-score global con estadísticas de train; las ventanas solo contienen valores con marca de tiempo ≤ t.
- **Dos salidas** (binaria y multiclase) con pérdida BCE + λ·CE, λ = 1 por defecto y dentro de la búsqueda de hiperparámetros.
- **Early stopping** sobre el PR-AUC binario de validación; prueba no se abre.
- **Tiempo de anticipación**: horas entre la primera alerta sostenida (p ≥ umbral durante ≥ 3 h seguidas, dentro de las 168 h previas y después de la falla anterior de la misma máquina) y la falla; umbral = el que maximiza el recall con precisión ≥ 0.80.

## Resultados (validación)

Selección por etapa (mayor PR-AUC binario): CNN-LSTM → L = 24 → pérdida ponderada. Cifras exactas en `results/*.csv`.

| Modelo | PR-AUC | ROC-AUC | Recall con precisión ≥ 0.80 | F1 macro | Anticipación mediana |
|---|---|---|---|---|---|
| Regresión logística (E1) | 0.858 | 0.998 | 0.836 | 0.943 | — |
| LightGBM (E1) | 0.999 | 1.000 | 1.000 | 0.996 | — |
| GRU, L = 72 | 0.994 | 0.999 | 0.992 | 0.964 | 24 h |
| TCN, L = 72 | 0.992 | 1.000 | 0.991 | 0.817 | 24 h |
| CNN-LSTM, L = 72 | 0.995 | 1.000 | 0.991 | 0.963 | 24 h |
| CNN-LSTM, L = 24 | 0.996 | 1.000 | 1.000 | 0.941 | 24 h |
| CNN-LSTM, L = 168 | 0.988 | 1.000 | 0.990 | 0.937 | 24 h |
| CNN-LSTM, L = 24, submuestreo | 0.989 | 1.000 | 0.993 | 0.945 | 24 h |
| CNN-LSTM final (Optuna) | 0.991 | 1.000 | 0.992 | 0.963 | 24 h |

- El deep learning supera a la regresión logística pero **no** a LightGBM: la simulación hace el problema casi determinista (error asociado 24 h antes de la falla, escalón de sensores 48 h antes) y esa regla la aprende mejor un gradient boosting sobre variables agregadas.
- Ventanas más largas no ayudan: toda la información útil cabe en 24 h.
- Las dos estrategias de desbalance ordenan igual; la pérdida ponderada sobreestima más la probabilidad en la zona ambigua (prior efectiva 50 % vs ≈ 9 % del submuestreo).
- Tiempo de anticipación: 117 de 118 fallas anticipadas, mediana 24 h (la alerta se enciende con el error precursor).
- La búsqueda de hiperparámetros no mejoró la configuración inicial; la variabilidad entre corridas (≈ 0.005 de PR-AUC) es del orden de las diferencias entre configuraciones.
