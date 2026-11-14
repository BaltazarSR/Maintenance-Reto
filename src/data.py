"""Carga de los CSV de Azure Predictive Maintenance y constantes del proyecto."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"

SENSORS = ["volt", "rotate", "pressure", "vibration"]
COMPS = ["comp1", "comp2", "comp3", "comp4"]
ERRORS = ["error1", "error2", "error3", "error4", "error5"]
MODELS = ["model1", "model2", "model3", "model4"]

H = 24  # horizonte de predicción en horas

# Partición temporal obligatoria (README, Tabla 2); límites inclusivos por día.
SPLITS = {
    "train": (pd.Timestamp("2015-01-01"), pd.Timestamp("2015-08-31 23:59:59")),
    "val": (pd.Timestamp("2015-09-01"), pd.Timestamp("2015-10-31 23:59:59")),
    "test": (pd.Timestamp("2015-11-01"), pd.Timestamp("2015-12-31 23:59:59")),
}


def _read(name, **kw):
    return pd.read_csv(DATA_DIR / name, parse_dates=["datetime"], **kw)


def load_telemetry(use_cache=True):
    cache = INTERIM_DIR / "telemetry.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    df = _read("PdM_telemetry.csv").sort_values(["machineID", "datetime"], ignore_index=True)
    if use_cache:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def load_errors():
    df = _read("PdM_errors.csv")
    df["errorID"] = pd.Categorical(df["errorID"], categories=ERRORS)
    return df.sort_values(["machineID", "datetime"], ignore_index=True)


def load_maint():
    df = _read("PdM_maint.csv")
    df["comp"] = pd.Categorical(df["comp"], categories=COMPS)
    return df.sort_values(["machineID", "datetime"], ignore_index=True)


def load_failures():
    df = _read("PdM_failures.csv")
    df["failure"] = pd.Categorical(df["failure"], categories=COMPS)
    return df.sort_values(["machineID", "datetime"], ignore_index=True)


def load_machines():
    df = pd.read_csv(DATA_DIR / "PdM_machines.csv")
    df["model"] = pd.Categorical(df["model"], categories=MODELS)
    return df


def load_all():
    return {
        "telemetry": load_telemetry(),
        "errors": load_errors(),
        "maint": load_maint(),
        "failures": load_failures(),
        "machines": load_machines(),
    }



def load_split(name):
    """Tabla de modelado de un conjunto ('train', 'val' o 'test') generada por entrega1/notebooks/00_pipeline.ipynb."""
    return pd.read_parquet(INTERIM_DIR / f"{name}.parquet")

def assign_split(ts):
    """Etiqueta cada timestamp con 'train' / 'val' / 'test' (o NA fuera de 2015)."""
    ts = pd.Series(pd.to_datetime(ts))
    out = pd.Series(pd.NA, index=ts.index, dtype="string")
    for name, (lo, hi) in SPLITS.items():
        out[(ts >= lo) & (ts <= hi)] = name
    return out


def classify_maint(maint, failures):
    """Añade la columna 'tipo' a cada reemplazo:

    - 'falla': reemplaza el componente que falló en esa misma máquina y hora.
    - 'oportunista': otro componente reemplazado en la misma visita que una falla.
    - 'programado': reemplazo sin falla en esa máquina y hora.
    """
    f = failures.rename(columns={"failure": "comp"})[["machineID", "datetime", "comp"]].assign(_f=True)
    visits = failures[["machineID", "datetime"]].drop_duplicates().assign(_v=True)
    out = maint.merge(f, on=["machineID", "datetime", "comp"], how="left").merge(
        visits, on=["machineID", "datetime"], how="left")
    is_fail = out["_f"].notna().to_numpy()
    at_visit = out["_v"].notna().to_numpy()
    tipo = pd.Series("programado", index=out.index)
    tipo[at_visit] = "oportunista"
    tipo[is_fail] = "falla"
    out["tipo"] = pd.Categorical(tipo, categories=["falla", "oportunista", "programado"])
    return out.drop(columns=["_f", "_v"])
