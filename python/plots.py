# plots.py
# Taller Corte 3 - Laboratorio 3.2
# Autor: Alessandro De Bonis Rodriguez
#
# Lee los resultados de C++ y de Scikit-Learn (carpeta results/) y genera:
#   results/fig_cost.png        -> curvas de costo de los dos modelos
#   results/fig_confusion.png   -> matrices de confusion de los dos modelos
#   results/comparison.csv      -> tabla comparativa
# Primero hay que correr ./build/train y python3 python/sklearn_model.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT, "results")

CPP_COLOR = "#2a78d6"       # azul: C++
SKLEARN_COLOR = "#eb6834"   # naranja: Scikit-Learn
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})

# ------------------------------------------------------------
# 1. Curvas de costo
# ------------------------------------------------------------
cpp_cost = pd.read_csv(os.path.join(RESULTS, "cpp_cost.csv"))
sk_cost = pd.read_csv(os.path.join(RESULTS, "sklearn_cost.csv"))

# La iteracion 0 no se puede dibujar en escala logaritmica,
# por eso se grafica desde la iteracion 1.
cpp_plot = cpp_cost[cpp_cost["iteracion"] >= 1]
sk_plot = sk_cost[sk_cost["iteracion"] >= 1]

fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.plot(cpp_plot["iteracion"], cpp_plot["costo"], color=CPP_COLOR, lw=1.6,
        label="C++ (gradiente descendiente)")
ax.plot(sk_plot["iteracion"], sk_plot["costo"], color=SKLEARN_COLOR, lw=1.6,
        ls="--", marker="o", ms=2.5, label="Scikit-Learn (L-BFGS)")
ax.set_xscale("log")
ax.set_xlabel("Iteración (escala logarítmica)")
ax.set_ylabel("Costo (Cross-Entropy)")
ax.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# Etiquetas directas al final de cada curva
last_cpp = cpp_plot.iloc[-1]
last_sk = sk_plot.iloc[-1]
ax.annotate("%.4f" % last_cpp["costo"], (last_cpp["iteracion"], last_cpp["costo"]),
            xytext=(-2, 6), textcoords="offset points", ha="right", color=INK, fontsize=7)
ax.annotate("%.4f" % last_sk["costo"], (last_sk["iteracion"], last_sk["costo"]),
            xytext=(0, -10), textcoords="offset points", ha="center", color=INK, fontsize=7)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig_cost.png"), dpi=300)
plt.close(fig)

# ------------------------------------------------------------
# 2. Matrices de confusion
# ------------------------------------------------------------
blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#2a78d6", "#0d366b"])

fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.75))
for ax, name, title in zip(axes, ["cpp", "sklearn"], ["C++ (Eigen)", "Scikit-Learn"]):
    cm = pd.read_csv(os.path.join(RESULTS, name + "_confusion.csv"))
    values = cm[["dice_3", "dice_5"]].to_numpy()
    ax.imshow(values, cmap=blues, vmin=0, vmax=values.max())
    for i in range(2):
        for j in range(2):
            dark = values[i, j] > values.max() * 0.5
            ax.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=8,
                    color="#ffffff" if dark else INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["3", "5"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["3", "5"])
    ax.set_xlabel("Predicho")
    ax.set_title(title, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
axes[0].set_ylabel("Real")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig_confusion.png"), dpi=300)
plt.close(fig)

# ------------------------------------------------------------
# 3. Tabla comparativa
# ------------------------------------------------------------
table = pd.concat([pd.read_csv(os.path.join(RESULTS, "cpp_metrics.csv")),
                   pd.read_csv(os.path.join(RESULTS, "sklearn_metrics.csv"))])
table.to_csv(os.path.join(RESULTS, "comparison.csv"), index=False)
print(table.set_index("modelo").T.to_string())
print("\nGraficas guardadas en results/fig_cost.png y results/fig_confusion.png")
