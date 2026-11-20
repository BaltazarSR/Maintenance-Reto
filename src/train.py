"""Entrenamiento de los modelos secuenciales (actividades 2.2-2.4).

- Pérdida combinada: BCE(binaria) + lam * CE(multiclase). Las dos tareas comparten la misma
  prior (la clase "ninguno" coincide con y_bin = 0) y están en escala de logits, así que con
  lam = 1 los gradientes de ambas cabezas son comparables; lam se incluye en la búsqueda de
  hiperparámetros.
- Desbalance (2.3): 'weighted' pondera la BCE con pos_weight = n_neg / n_pos y la CE con
  pesos balanceados N / (K n_c); 'undersample' usa pérdida sin ponderar pero cada época toma
  todas las horas positivas y una muestra aleatoria de horas sanas (NEG_RATIO por positiva);
  'none' = sin corrección.
- Entrenamiento disciplinado (2.4): AdamW (weight decay), dropout, early stopping sobre el
  PR-AUC de validación y restauración del mejor estado.
"""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .models import N_CLASSES

SEED = 14
NEG_RATIO = 10


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def class_weights(y_bin, y_cls):
    pos_weight = float((y_bin == 0).sum() / (y_bin == 1).sum())
    counts = np.bincount(y_cls, minlength=N_CLASSES).astype("float64")
    cw = len(y_cls) / (N_CLASSES * counts)
    return pos_weight, cw


def make_loss(strategy, y_bin, y_cls, lam, device):
    if strategy == "weighted":
        pw, cw = class_weights(y_bin, y_cls)
        pw = torch.tensor(pw, device=device)
        cw = torch.tensor(cw, dtype=torch.float32, device=device)
    else:
        pw = cw = None

    def loss_fn(bin_logit, cls_logits, yb, yc):
        lb = F.binary_cross_entropy_with_logits(bin_logit, yb, pos_weight=pw)
        lc = F.cross_entropy(cls_logits, yc, weight=cw)
        return lb + lam * lc

    return loss_fn


def epoch_order(strategy, y_bin, rng):
    """Índices (sobre el conjunto de entrenamiento) que recorre una época, ya barajados."""
    if strategy == "undersample":
        pos = np.flatnonzero(y_bin == 1)
        neg = np.flatnonzero(y_bin == 0)
        neg = rng.choice(neg, size=min(len(neg), NEG_RATIO * len(pos)), replace=False)
        idx = np.concatenate([pos, neg])
    else:
        idx = np.arange(len(y_bin))
    rng.shuffle(idx)
    return idx


@torch.no_grad()
def predict(model, wt, split, L, batch=2048):
    """Probabilidades en un conjunto: p_bin (n,) y p_cls (n, 5)."""
    model.eval()
    m = torch.from_numpy(split.m).to(wt.device)
    t = torch.from_numpy(split.t).to(wt.device)
    pb, pc = [], []
    for i in range(0, len(split), batch):
        x, mi, age = wt.batch(m[i:i + batch], t[i:i + batch], L)
        b, c = model(x, mi, age)
        pb.append(torch.sigmoid(b).cpu())
        pc.append(torch.softmax(c, dim=1).cpu())
    return torch.cat(pb).numpy().astype("float64"), torch.cat(pc).numpy().astype("float64")


def fit(model, wt, train, val, L, strategy="weighted", lam=1.0, lr=1e-3, weight_decay=1e-4,
        batch=512, max_epochs=12, patience=3, seed=SEED, trial=None, log=print):
    """Entrena con early stopping sobre el PR-AUC binario de validación.
    Devuelve (modelo con el mejor estado, historial por época)."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = wt.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = make_loss(strategy, train.y_bin, train.y_cls, lam, device)

    m_all = torch.from_numpy(train.m).to(device)
    t_all = torch.from_numpy(train.t).to(device)
    yb_all = torch.from_numpy(train.y_bin).to(device)
    yc_all = torch.from_numpy(train.y_cls).to(device)
    yb_val = torch.from_numpy(val.y_bin).to(device)
    yc_val = torch.from_numpy(val.y_cls).to(device)

    history, best, best_state, bad = [], -np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        order = torch.from_numpy(epoch_order(strategy, train.y_bin, rng)).to(device)
        tot, n = 0.0, 0
        for i in range(0, len(order), batch):
            sel = order[i:i + batch]
            x, mi, age = wt.batch(m_all[sel], t_all[sel], L)
            b, c = model(x, mi, age)
            loss = loss_fn(b, c, yb_all[sel], yc_all[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(sel)
            n += len(sel)

        # validación: pérdida (misma función) y métricas
        model.eval()
        with torch.no_grad():
            m_v = torch.from_numpy(val.m).to(device)
            t_v = torch.from_numpy(val.t).to(device)
            vl, pb = 0.0, []
            for i in range(0, len(val), 2048):
                x, mi, age = wt.batch(m_v[i:i + 2048], t_v[i:i + 2048], L)
                b, c = model(x, mi, age)
                vl += loss_fn(b, c, yb_val[i:i + 2048], yc_val[i:i + 2048]).item() * len(b)
                pb.append(torch.sigmoid(b).cpu())
        p = torch.cat(pb).numpy()
        pr_auc = average_precision_score(val.y_bin, p)
        roc = roc_auc_score(val.y_bin, p)
        rec = {"epoch": epoch, "train_loss": tot / n, "val_loss": vl / len(val), "val_pr_auc": pr_auc,
               "val_roc_auc": roc, "seconds": time.time() - t0}
        history.append(rec)
        log(f"época {epoch:2d}  pérdida train {rec['train_loss']:.4f}  val {rec['val_loss']:.4f}  "
            f"PR-AUC val {pr_auc:.4f}  ROC-AUC val {roc:.4f}  ({rec['seconds']:.0f} s)")

        if trial is not None:
            trial.report(pr_auc, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        if pr_auc > best + 1e-5:
            best, best_state, bad = pr_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log(f"early stopping (sin mejora en {patience} épocas)")
                break

    model.load_state_dict(best_state)
    return model, history
