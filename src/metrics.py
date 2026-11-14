"""Métricas de evaluación (Entrega 1 y siguientes)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
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
