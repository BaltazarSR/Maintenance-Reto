"""Variables para la línea base clásica (actividad 1.4).

Todas son causales: la fila (m, t) solo usa información con marca de tiempo <= t.
Las ventanas son retrospectivas e incluyen la hora t: (t - w, t].

- Sensores: media y desviación estándar móviles de 3 h y 24 h.
- Errores: conteo de cada tipo (y total) en las últimas 24 h.
- Mantenimiento: horas desde el último reemplazo de cada componente (registros <= t,
  incluyendo 2014 para inicializar). Ver 02_maintenance: codifican en parte el
  calendario de visitas, por eso forman un grupo aparte.
- Estáticas: modelo (one-hot) y antigüedad.
"""
import pandas as pd

from .data import COMPS, ERRORS, MODELS, SENSORS

WINDOWS = (3, 24)
ERROR_WINDOW = 24

FEATURE_GROUPS = {
    "sensores": [f"{s}_{stat}_{w}h" for w in WINDOWS for s in SENSORS for stat in ("mean", "std")],
    "errores": [f"{e}_count_{ERROR_WINDOW}h" for e in ERRORS] + [f"error_total_count_{ERROR_WINDOW}h"],
    "mantenimiento": [f"h_since_{c}" for c in COMPS],
    "estaticas": [f"model_{m}" for m in MODELS] + ["age"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def sensor_features(tel, windows=WINDOWS):
    """Medias y desviaciones móviles por máquina. La primera hora de cada máquina
    tiene std NaN (una sola lectura)."""
    tel = tel.sort_values(["machineID", "datetime"], ignore_index=True)
    out = tel[["machineID", "datetime"]].copy()
    grouped = tel.groupby("machineID")[SENSORS]
    for w in windows:
        roll = grouped.rolling(w, min_periods=1)
        mean = roll.mean().reset_index(level=0, drop=True)
        std = grouped.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
        for s in SENSORS:
            out[f"{s}_mean_{w}h"] = mean[s].to_numpy()
            out[f"{s}_std_{w}h"] = std[s].to_numpy()
    return out


def error_features(grid, errors, window=ERROR_WINDOW):
    """Conteo de cada tipo de error en (t - window, t] sobre la rejilla horaria `grid`."""
    grid = grid[["machineID", "datetime"]].sort_values(["machineID", "datetime"], ignore_index=True)
    hourly = (errors.assign(n=1)
              .pivot_table(index=["machineID", "datetime"], columns="errorID", values="n",
                           aggfunc="sum", fill_value=0, observed=False)
              .reindex(columns=ERRORS, fill_value=0))
    ind = grid.join(hourly, on=["machineID", "datetime"]).fillna({e: 0 for e in ERRORS})
    counts = (ind.groupby("machineID")[ERRORS].rolling(window, min_periods=1).sum()
              .reset_index(level=0, drop=True))
    out = grid.copy()
    for e in ERRORS:
        out[f"{e}_count_{window}h"] = counts[e].to_numpy().astype("int16")
    out[f"error_total_count_{window}h"] = out[[f"{e}_count_{window}h" for e in ERRORS]].sum(axis=1).astype("int16")
    return out


def maint_features(grid, maint):
    """Horas desde el último reemplazo de cada componente con datetime <= t."""
    grid = grid[["machineID", "datetime"]].sort_values(["machineID", "datetime"], ignore_index=True)
    left = grid.reset_index().sort_values("datetime")
    out = grid.copy()
    for c in COMPS:
        m = maint.loc[maint["comp"] == c, ["machineID", "datetime"]].rename(columns={"datetime": "_t"})
        r = pd.merge_asof(left, m.sort_values("_t"), left_on="datetime", right_on="_t",
                          by="machineID", direction="backward", allow_exact_matches=True)
        r = r.set_index("index").sort_index()
        out[f"h_since_{c}"] = ((r["datetime"] - r["_t"]) / pd.Timedelta(hours=1)).to_numpy()
    return out


def static_features(grid, machines):
    m = machines.copy()
    for k in MODELS:
        m[f"model_{k}"] = (m["model"] == k).astype("int8")
    cols = ["machineID", "age"] + [f"model_{k}" for k in MODELS]
    return grid[["machineID", "datetime"]].merge(m[cols], on="machineID", how="left")
