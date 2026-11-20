"""Arquitecturas secuenciales (actividad 2.2). Todas reciben la ventana (B, L, C) y las
entradas estáticas (modelo, antigüedad) y devuelven dos salidas: logit binario y logits
de 5 clases."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MODELS
from .labels import CLASSES
from .windows import CHANNELS

N_CHANNELS = len(CHANNELS)
N_CLASSES = len(CLASSES)


class StaticEncoder(nn.Module):
    """Embedding del modelo de máquina concatenado con la antigüedad estandarizada."""

    def __init__(self, emb_dim=4):
        super().__init__()
        self.emb = nn.Embedding(len(MODELS), emb_dim)
        self.out_dim = emb_dim + 1

    def forward(self, model_idx, age):
        return torch.cat([self.emb(model_idx), age[:, None]], dim=1)


class DualHead(nn.Module):
    """Capa oculta compartida y dos salidas: binaria (1 logit) y multiclase (5 logits)."""

    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.bin = nn.Linear(hidden, 1)
        self.cls = nn.Linear(hidden, N_CLASSES)

    def forward(self, h):
        z = self.fc(h)
        return self.bin(z).squeeze(1), self.cls(z)


class GRUNet(nn.Module):
    """GRU de 1 o 2 capas; el estado final se concatena con las entradas estáticas."""

    def __init__(self, hidden=64, layers=2, dropout=0.2, emb_dim=4):
        super().__init__()
        self.static = StaticEncoder(emb_dim)
        self.gru = nn.GRU(N_CHANNELS, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = DualHead(hidden + self.static.out_dim, hidden, dropout)

    def forward(self, x, model_idx, age):
        _, h = self.gru(x)
        return self.head(torch.cat([h[-1], self.static(model_idx, age)], dim=1))


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

    def forward(self, x):  # x: (B, C, L)
        return self.conv(F.pad(x, (self.pad, 0)))


class TemporalBlock(nn.Module):
    """Bloque residual del TCN (Bai et al., 2018): dos convoluciones causales dilatadas."""

    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(CausalConv(cin, cout, k, dilation), nn.ReLU(), nn.Dropout(dropout),
                                 CausalConv(cout, cout, k, dilation), nn.ReLU(), nn.Dropout(dropout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.relu(self.net(x) + self.skip(x))


class TCN(nn.Module):
    """Red convolucional temporal: bloques con dilatación 1, 2, 4, ... Campo receptivo
    1 + 2 (k - 1)(2^n - 1); con k = 3 y n = 6 cubre 253 h (>= 168)."""

    def __init__(self, hidden=64, blocks=6, kernel=3, dropout=0.2, emb_dim=4):
        super().__init__()
        self.static = StaticEncoder(emb_dim)
        layers, cin = [], N_CHANNELS
        for i in range(blocks):
            layers.append(TemporalBlock(cin, hidden, kernel, 2 ** i, dropout))
            cin = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = DualHead(hidden + self.static.out_dim, hidden, dropout)
        self.receptive_field = 1 + 2 * (kernel - 1) * (2 ** blocks - 1)

    def forward(self, x, model_idx, age):
        h = self.tcn(x.transpose(1, 2))[:, :, -1]
        return self.head(torch.cat([h, self.static(model_idx, age)], dim=1))


class CNNLSTM(nn.Module):
    """Híbrido: dos convoluciones causales extraen patrones locales y un LSTM resume la secuencia."""

    def __init__(self, hidden=64, filters=32, kernel=5, dropout=0.2, emb_dim=4):
        super().__init__()
        self.static = StaticEncoder(emb_dim)
        self.cnn = nn.Sequential(CausalConv(N_CHANNELS, filters, kernel, 1), nn.ReLU(),
                                 CausalConv(filters, filters, kernel, 1), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(filters, hidden, batch_first=True)
        self.head = DualHead(hidden + self.static.out_dim, hidden, dropout)

    def forward(self, x, model_idx, age):
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.lstm(z)
        return self.head(torch.cat([h[-1], self.static(model_idx, age)], dim=1))


ARCHS = {"gru": GRUNet, "tcn": TCN, "cnnlstm": CNNLSTM}


def build_model(arch, **hp):
    return ARCHS[arch](**hp)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
