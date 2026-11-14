"""Estilo y guardado de figuras."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .data import ROOT

FIG_DIR = ROOT / "entrega1" / "figures"


def set_style():
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")
    plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight"})


def savefig(fig, name, fig_dir=FIG_DIR):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png", dpi=150)

