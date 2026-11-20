"""Experimentos de la Entrega 2. Cada corrida escribe entrega2/results/runs/<name>/
{config.json, history.csv, metrics.json, val_preds.parquet}; los notebooks solo leen esos archivos.

Uso (desde la raíz del repositorio):
  python entrega2/run_experiments.py run  --name arch_gru --arch gru --L 72 --imbalance weighted
  python entrega2/run_experiments.py tune --arch gru --L 72 --imbalance weighted --trials 20
  python entrega2/run_experiments.py all          # las cuatro etapas en orden (≈ 2-3 h en un M3)

Etapas de `all` (presupuesto declarado): (1) tres arquitecturas con L = 72 y pérdida ponderada;
(2) la mejor arquitectura con L en {24, 72, 168}; (3) la mejor (arquitectura, L) con 'weighted'
vs 'undersample'; (4) Optuna, 20 ensayos de hasta 8 épocas con poda, y corrida final con los
mejores hiperparámetros. "Mejor" = mayor PR-AUC binario en validación.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.data import load_failures
from src.labels import CLASSES
from src.metrics import binary_report, brier, f1_per_class, lead_time
from src.models import build_model, n_params
from src.train import SEED, fit, get_device, predict, set_seed
from src.windows import prepare

RUNS = ROOT / "entrega2" / "results" / "runs"
RESULTS = ROOT / "entrega2" / "results"
DEFAULT_HP = {"gru": dict(hidden=64, layers=2, dropout=0.2),
              "tcn": dict(hidden=64, blocks=6, kernel=3, dropout=0.2),
              "cnnlstm": dict(hidden=64, filters=32, kernel=5, dropout=0.2)}
TRAIN_DEFAULTS = dict(lam=1.0, lr=1e-3, weight_decay=1e-4, batch=512, max_epochs=12, patience=3)

_DATA = {}


def data():
    if not _DATA:
        device = get_device()
        t0 = time.time()
        wt, norm, splits, machines, start = prepare(device)
        norm.save(RESULTS / "normalizer.json")
        print(f"tensor listo en {time.time() - t0:.0f} s  ({device}, X {tuple(wt.X.shape)}, "
              f"train {len(splits['train'])}, val {len(splits['val'])})")
        _DATA.update(wt=wt, splits=splits, failures=load_failures())
    return _DATA


def run(name, arch, L, imbalance, hp=None, train_kw=None, trial=None, save=True):
    d = data()
    hp = {**DEFAULT_HP[arch], **(hp or {})}
    train_kw = {**TRAIN_DEFAULTS, **(train_kw or {})}
    set_seed(SEED)
    model = build_model(arch, **hp)
    print(f"\n=== {name}: {arch} L={L} {imbalance} {hp} {train_kw}  ({n_params(model):,} parámetros)")
    model, history = fit(model, d["wt"], d["splits"]["train"], d["splits"]["val"], L, strategy=imbalance,
                         trial=trial, **train_kw)
    val = d["splits"]["val"]
    p_bin, p_cls = predict(model, d["wt"], val, L)
    rep = binary_report(val.y_bin, p_bin, 0.80)
    f1 = f1_per_class(val.y_cls, p_cls, CLASSES)
    pred = pd.DataFrame({"machineID": val.machineID, "datetime": val.datetime, "p": p_bin})
    lt = lead_time(pred, d["failures"], rep["umbral"]) if np.isfinite(rep["umbral"]) else pd.DataFrame(columns=["lead_h"])
    metrics = {"name": name, "arch": arch, "L": L, "imbalance": imbalance, "hp": hp, "train": train_kw,
               "n_params": n_params(model), "best_epoch": int(max(history, key=lambda r: r["val_pr_auc"])["epoch"]),
               "epochs": len(history), "seconds": float(sum(r["seconds"] for r in history)),
               **{k: float(v) for k, v in rep.items()}, "brier": brier(val.y_bin, p_bin),
               "f1": f1, "lead_median_h": float(np.nanmedian(lt["lead_h"])) if len(lt) else np.nan,
               "lead_mean_h": float(np.nanmean(lt["lead_h"])) if len(lt) else np.nan,
               "failures_anticipated": float(lt["lead_h"].notna().mean()) if len(lt) else np.nan,
               "n_failures": int(len(lt))}
    print(f"PR-AUC {rep['PR-AUC']:.4f}  ROC-AUC {rep['ROC-AUC']:.4f}  recall@P≥0.80 {rep['recall@P≥0.80']:.3f}  "
          f"F1 macro {f1['macro']:.3f}  anticipación mediana {metrics['lead_median_h']:.1f} h")
    if save:
        out = RUNS / name
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
        pd.DataFrame({"machineID": val.machineID, "datetime": val.datetime, "p_bin": p_bin,
                      **{f"p_{c}": p_cls[:, i] for i, c in enumerate(CLASSES)}}).to_parquet(out / "val_preds.parquet", index=False)
        lt.to_csv(out / "lead_time.csv", index=False)
        torch.save(model.state_dict(), out / "model.pt")
    return metrics


def tune(arch, L, imbalance, trials, max_epochs=8):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        hp = {"hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
              "dropout": trial.suggest_float("dropout", 0.0, 0.5)}
        if arch == "gru":
            hp["layers"] = trial.suggest_int("layers", 1, 2)
        elif arch == "tcn":
            hp["kernel"] = trial.suggest_categorical("kernel", [3, 5])
        else:
            hp["filters"] = trial.suggest_categorical("filters", [16, 32, 64])
        tk = {"lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
              "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
              "lam": trial.suggest_float("lam", 0.25, 2.0, log=True),
              "max_epochs": max_epochs, "patience": 3}
        m = run(f"trial_{trial.number:02d}", arch, L, imbalance, hp, tk, trial=trial, save=False)
        return m["PR-AUC"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    study.optimize(objective, n_trials=trials)
    df = study.trials_dataframe()
    df.to_csv(RESULTS / "optuna_trials.csv", index=False)
    best = {"arch": arch, "L": L, "imbalance": imbalance, "trials": trials, "max_epochs": max_epochs,
            "best_value": study.best_value, "best_params": study.best_params}
    (RESULTS / "optuna_best.json").write_text(json.dumps(best, indent=1))
    print(f"\nmejor ensayo: PR-AUC {study.best_value:.4f}  {study.best_params}")
    return best


def split_params(arch, params):
    hp = {k: v for k, v in params.items() if k in ("hidden", "dropout", "layers", "kernel", "filters")}
    tk = {k: v for k, v in params.items() if k in ("lr", "weight_decay", "lam")}
    return hp, tk


def run_all(trials):
    stages = {}
    # 1. arquitecturas
    archs = {a: run(f"arch_{a}", a, 72, "weighted") for a in ("gru", "tcn", "cnnlstm")}
    best_arch = max(archs, key=lambda a: archs[a]["PR-AUC"])
    stages["arch"] = best_arch
    # 2. longitud de ventana
    Ls = {72: archs[best_arch]}
    for L in (24, 168):
        Ls[L] = run(f"L_{best_arch}_{L}", best_arch, L, "weighted")
    best_L = max(Ls, key=lambda L: Ls[L]["PR-AUC"])
    stages["L"] = best_L
    # 3. desbalance
    imb = {"weighted": Ls[best_L], "undersample": run(f"imb_{best_arch}_{best_L}_undersample", best_arch, best_L, "undersample")}
    best_imb = max(imb, key=lambda s: imb[s]["PR-AUC"])
    stages["imbalance"] = best_imb
    # 4. búsqueda de hiperparámetros y corrida final
    best = tune(best_arch, best_L, best_imb, trials)
    hp, tk = split_params(best_arch, best["best_params"])
    run("final", best_arch, best_L, best_imb, hp, tk)
    (RESULTS / "stages.json").write_text(json.dumps(stages, indent=1))
    print("\nselección por etapa:", stages)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--arch", choices=list(DEFAULT_HP), required=True)
    r.add_argument("--L", type=int, default=72)
    r.add_argument("--imbalance", choices=["weighted", "undersample", "none"], default="weighted")
    r.add_argument("--max-epochs", type=int, default=TRAIN_DEFAULTS["max_epochs"])
    t = sub.add_parser("tune")
    t.add_argument("--arch", choices=list(DEFAULT_HP), required=True)
    t.add_argument("--L", type=int, default=72)
    t.add_argument("--imbalance", default="weighted")
    t.add_argument("--trials", type=int, default=20)
    a = sub.add_parser("all")
    a.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.name, args.arch, args.L, args.imbalance, train_kw={"max_epochs": args.max_epochs})
    elif args.cmd == "tune":
        tune(args.arch, args.L, args.imbalance, args.trials)
    else:
        run_all(args.trials)
