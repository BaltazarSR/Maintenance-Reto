"""Etiquetas de predicción con horizonte H (actividad 1.3).

Para cada máquina m y hora t:
    y_bin(t) = 1  si hay una falla de cualquier componente en (t, t + H]
    y_cls(t) ∈ {0: ninguno, 1: comp1, ..., 4: comp4}           (multiclase)

El extremo izquierdo es abierto: la hora de la falla no es positiva para sí misma
(en esa hora el registro de reparación ya existe; ver 02_maintenance).

Multiclase cuando hay más de un componente en la ventana: se toma el de la falla
más próxima; si fallan a la misma hora, el componente menos frecuente en
entrenamiento (TIE_PRIORITY), para no diluir las clases minoritarias.
"""
import numpy as np
import pandas as pd

from .data import COMPS, H, SPLITS

CLASSES = ["ninguno"] + COMPS
# Orden de desempate: de la clase menos a la más frecuente en train (ver 03_labels).
TIE_PRIORITY = ["comp3", "comp4", "comp1", "comp2"]


def make_labels(index, failures, h=H):
    """index: DataFrame con machineID, datetime (las horas a etiquetar).

    Devuelve index + y_bin, y_cls, h_to_failure (horas hasta la próxima
    falla de cualquier componente, NaN si no hay).
    """
    out = index[["machineID", "datetime"]].copy()
    left = out.reset_index().sort_values("datetime")

    # Horas hasta la próxima falla de cada componente (estrictamente posterior a t)
    nxt = {}
    for c in COMPS:
        f = failures.loc[failures["failure"] == c, ["machineID", "datetime"]].rename(columns={"datetime": "_tf"})
        m = pd.merge_asof(left, f.sort_values("_tf"), left_on="datetime", right_on="_tf",
                          by="machineID", direction="forward", allow_exact_matches=False)
        nxt[c] = (m.set_index("index").sort_index()["_tf"] - out["datetime"]) / pd.Timedelta(hours=1)
    nxt = pd.DataFrame(nxt)

    out["h_to_failure"] = nxt.min(axis=1)
    out["y_bin"] = (out["h_to_failure"] <= h).astype("int8")

    # Multiclase: falla más próxima dentro del horizonte, desempate por TIE_PRIORITY
    within = nxt.where(nxt <= h)
    earliest = within.min(axis=1)
    y_cls = np.zeros(len(out), dtype="int8")
    for c in reversed(TIE_PRIORITY):  # el último asignado gana -> mayor prioridad
        hit = (within[c] == earliest) & earliest.notna()
        y_cls[hit.to_numpy()] = CLASSES.index(c)
    out["y_cls"] = y_cls
    return out


def assign_label_split(ts, h=H):
    """Partición de las horas t a predecir dejando un margen de h horas en cada frontera:
    t pertenece a un conjunto solo si t + h también cae dentro de él, de modo que ninguna
    etiqueta depende de eventos del conjunto siguiente. Las horas del margen quedan NA.
    """
    ts = pd.Series(pd.to_datetime(ts))
    end_of_label = ts + pd.Timedelta(hours=h)
    out = pd.Series(pd.NA, index=ts.index, dtype="string")
    for name, (lo, hi) in SPLITS.items():
        out[(ts >= lo) & (end_of_label <= hi)] = name
    return out
