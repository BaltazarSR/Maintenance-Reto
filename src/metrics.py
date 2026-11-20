"""Métricas de evaluación (Entrega 1 y siguientes)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)


def recall_at_precision(y, p, min_precision=0.80):
    """Máximo recall alcanzable con precisión >= min_precision, y el umbral que lo logra.
    Devuelve (recall, precision, umbral); recall = 0 y umbral = NaN si nunca se alcanza."""
    prec, rec, thr = precision_recall_curve(y, p)
    # prec/rec tienen un elemento más que thr (el punto final recall=0, precision=1)
    ok = prec[:-1] >= min_precision
    if not ok.any():
        return 0.0, np.nan, np.nan
    r = np.where(ok, rec[:-1], -1)
    # entre los puntos con el recall máximo, el de umbral más alto (thr es creciente)
    i = np.flatnonzero(r == r.max())[-1]
    return float(rec[i]), float(prec[i]), float(thr[i])


def binary_report(y, p, min_precision=0.80):
    r, pr, thr = recall_at_precision(y, p, min_precision)
    return {
        "PR-AUC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        f"recall@P≥{min_precision:.2f}": r,
        "precisión en ese punto": pr,
        "umbral": thr,
    }


def confusion(y, proba, labels):
    """Matriz de confusión multiclase con la clase de mayor probabilidad."""
    pred = np.asarray(proba).argmax(axis=1)
    return pd.DataFrame(confusion_matrix(y, pred, labels=range(len(labels))),
                        index=pd.Index(labels, name="real"), columns=pd.Index(labels, name="predicho"))


def f1_per_class(y, proba, labels):
    """F1 por clase (y macro) con la clase de mayor probabilidad. Devuelve dict clase -> F1, más 'macro'."""
    pred = np.asarray(proba).argmax(axis=1)
    f1 = f1_score(y, pred, labels=range(len(labels)), average=None, zero_division=0)
    out = {lab: float(v) for lab, v in zip(labels, f1)}
    out["macro"] = float(f1.mean())
    return out


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def lead_time(pred, failures, thr, min_run=3, lookback_h=168):
    """Tiempo de anticipación por falla (Entrega 2).

    pred: DataFrame con machineID, datetime, p (probabilidad binaria por hora).
    failures: PdM_failures. Se consideran las fallas que generan alguna hora positiva en pred
    (t < t_f <= t + H). Una alerta sostenida empieza en la primera hora t, dentro de las
    `lookback_h` horas previas a la falla y posterior a la falla anterior de la misma máquina,
    con p >= thr durante `min_run` horas seguidas (todas anteriores a la falla).

    Devuelve un DataFrame con una fila por falla: machineID, datetime, failure, lead_h
    (horas entre el inicio de la alerta y la falla; NaN si no hubo alerta sostenida).
    """
    pred = pred.sort_values(["machineID", "datetime"])
    lo, hi = pred["datetime"].min(), pred["datetime"].max() + pd.Timedelta(hours=24)
    f = failures[(failures["datetime"] > lo) & (failures["datetime"] <= hi)].sort_values(["machineID", "datetime"])
    rows = []
    for mid, g in f.groupby("machineID", observed=True):
        pm = pred[pred["machineID"] == mid]
        times, p = pm["datetime"].to_numpy(), pm["p"].to_numpy()
        hit = p >= thr
        # run[i] = True si hit[i], hit[i+1], ..., hit[i+min_run-1] son todos True
        run = hit.copy()
        for k in range(1, min_run):
            run[:-k] &= hit[k:]
            run[-k:] = False
        prev = pd.Timestamp.min
        for t_f, visit in g.groupby("datetime", sort=True):  # fallas simultáneas comparten la ventana de búsqueda
            tf = np.datetime64(t_f)
            start = max(np.datetime64(prev), tf - np.timedelta64(lookback_h, "h"))
            ok = (times > start) & (times + np.timedelta64(min_run - 1, "h") < tf) & run
            lead = (tf - times[ok][0]) / np.timedelta64(1, "h") if ok.any() else np.nan
            for r in visit.itertuples():
                rows.append({"machineID": mid, "datetime": r.datetime, "failure": r.failure, "lead_h": lead})
            prev = t_f
    return pd.DataFrame(rows)
