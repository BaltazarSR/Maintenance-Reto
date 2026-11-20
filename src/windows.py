"""Tensor de entrada para los modelos secuenciales (actividad 2.1).

Para cada (m, t) la entrada es una ventana de L horas (t - L + 1, ..., t) con 13 canales
dinámicos, más dos entradas estáticas:

- Dinámicos (CHANNELS): los 4 sensores, 5 indicadores horarios de error (1 si hubo un error
  de ese tipo en esa hora) y log(1 + horas desde el último reemplazo) de cada componente.
- Estáticos: índice del modelo de máquina (para un embedding) y antigüedad.

Normalización: z-score GLOBAL por canal con estadísticas calculadas solo en las horas de
entrenamiento (ver 01_tensor: los cuatro modelos tienen la misma distribución de sensores,
así que normalizar por modelo no aporta, y normalizar por máquina absorbería el escalón
previo a la falla). Los indicadores de error se dejan en 0/1.

La ventana se extrae por indexación sobre un tensor (100, T, 13); no se materializan las
ventanas. Las horas anteriores al inicio de la serie se rellenan con ceros (= la media).
"""
import json

import numpy as np
import pandas as pd
import torch

from .data import COMPS, ERRORS, MODELS, SENSORS, SPLITS, load_all, load_split
from .features import maint_features

CHANNELS = SENSORS + [f"{e}_ind" for e in ERRORS] + [f"log_h_since_{c}" for c in COMPS]
STANDARDIZE = [c for c in CHANNELS if not c.endswith("_ind")]
L_VALUES = (24, 72, 168)


def build_channels(d=None):
    """Devuelve (X, machines, t0):
    X: float32 (n_máquinas, T, 13) sin normalizar, en la rejilla horaria de la telemetría;
    machines: DataFrame con machineID, model_idx, age (orden de la primera dimensión);
    t0: primer timestamp de la rejilla (t_idx = horas desde t0)."""
    d = d or load_all()
    tel = d["telemetry"].sort_values(["machineID", "datetime"], ignore_index=True)
    ids = tel["machineID"].unique()
    T = len(tel) // len(ids)
    assert len(tel) == len(ids) * T, "la rejilla horaria no es rectangular"
    t0 = tel["datetime"].min()
    assert ((tel["datetime"] - t0) / pd.Timedelta(hours=1)).to_numpy().reshape(len(ids), T).std(axis=0).max() == 0

    grid = tel[["machineID", "datetime"]]
    ind = (d["errors"].assign(n=1)
           .pivot_table(index=["machineID", "datetime"], columns="errorID", values="n", aggfunc="max",
                        fill_value=0, observed=False)
           .reindex(columns=ERRORS, fill_value=0))
    err = grid.join(ind, on=["machineID", "datetime"]).fillna({e: 0 for e in ERRORS})[ERRORS].to_numpy()
    hs = maint_features(grid, d["maint"])[[f"h_since_{c}" for c in COMPS]].to_numpy()
    assert not np.isnan(hs).any(), "horas desde el último reemplazo sin inicializar"

    X = np.concatenate([tel[SENSORS].to_numpy(), err, np.log1p(hs)], axis=1).astype("float32")
    X = X.reshape(len(ids), T, len(CHANNELS))

    m = d["machines"].set_index("machineID").loc[ids].reset_index()
    machines = pd.DataFrame({"machineID": ids, "model_idx": m["model"].map({k: i for i, k in enumerate(MODELS)}).astype(int),
                             "age": m["age"].astype(float)})
    return X, machines, t0


class Normalizer:
    """z-score global por canal (y de la antigüedad), ajustado solo con horas de entrenamiento."""

    def __init__(self):
        self.mean = self.std = None
        self.age_mean = self.age_std = None

    def fit(self, X, t0, machines):
        lo, hi = SPLITS["train"]
        times = t0 + pd.to_timedelta(np.arange(X.shape[1]), unit="h")
        mask = (times >= lo) & (times <= hi)
        flat = X[:, mask, :].reshape(-1, X.shape[2])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0)
        keep = np.array([c not in STANDARDIZE for c in CHANNELS])
        self.mean[keep], self.std[keep] = 0.0, 1.0
        self.age_mean, self.age_std = machines["age"].mean(), machines["age"].std()
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype("float32")

    def transform_age(self, age):
        return ((np.asarray(age, dtype="float64") - self.age_mean) / self.age_std).astype("float32")

    def to_dict(self):
        return {"channels": CHANNELS, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "age_mean": float(self.age_mean), "age_std": float(self.age_std)}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)


class SplitIndex:
    """Índices (m_idx, t_idx) y etiquetas de las horas a predecir de un conjunto."""

    def __init__(self, split, machines, t0):
        df = load_split(split)
        pos = {mid: i for i, mid in enumerate(machines["machineID"])}
        self.datetime = df["datetime"].to_numpy()
        self.machineID = df["machineID"].to_numpy()
        self.m = df["machineID"].map(pos).to_numpy().astype("int64")
        self.t = ((df["datetime"] - t0) / pd.Timedelta(hours=1)).to_numpy().astype("int64")
        self.y_bin = df["y_bin"].to_numpy().astype("float32")
        self.y_cls = df["y_cls"].to_numpy().astype("int64")
        self.h_to_failure = df["h_to_failure"].to_numpy()

    def __len__(self):
        return len(self.t)


class WindowTensor:
    """Tensor normalizado en el dispositivo; extrae lotes de ventanas por indexación."""

    def __init__(self, X_norm, machines, norm, device, pad=max(L_VALUES) - 1):
        self.pad = pad
        Xp = np.concatenate([np.zeros((X_norm.shape[0], pad, X_norm.shape[2]), dtype="float32"), X_norm], axis=1)
        self.X = torch.from_numpy(Xp).to(device)
        self.model_idx = torch.from_numpy(machines["model_idx"].to_numpy()).to(device)
        self.age = torch.from_numpy(norm.transform_age(machines["age"])).to(device)
        self.device = device

    def batch(self, m, t, L):
        """m, t: tensores long (B,) en el dispositivo. Devuelve x (B, L, C), model_idx (B,), age (B,)."""
        offsets = torch.arange(-L + 1, 1, device=self.device)
        idx = t[:, None] + offsets[None, :] + self.pad
        x = self.X[m[:, None], idx]
        return x, self.model_idx[m], self.age[m]


def prepare(device):
    """Pipeline completo: canales, normalizador (train), tensor en dispositivo e índices de train/val."""
    X, machines, t0 = build_channels()
    norm = Normalizer().fit(X, t0, machines)
    wt = WindowTensor(norm.transform(X), machines, norm, device)
    return wt, norm, {s: SplitIndex(s, machines, t0) for s in ("train", "val")}, machines, t0
