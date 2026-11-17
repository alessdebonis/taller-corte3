# sklearn_model.py
# Taller Corte 3 - Laboratorio 3.2
# Autor: Alessandro De Bonis Rodriguez
#
# Entrena la regresion logistica de Scikit-Learn con EXACTAMENTE
# los mismos datos que el modelo de C++ (data/train_5vs3.csv y
# data/test_5vs3.csv) y guarda los mismos resultados en results/.

import os
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, log_loss)
from sklearn.exceptions import ConvergenceWarning

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT, "data")
RESULTS = os.path.join(PROJECT, "results")
os.makedirs(RESULTS, exist_ok=True)


# ------------------------------------------------------------
# 1. Cargar los mismos CSV que usa C++
# ------------------------------------------------------------
def load(name):
    df = pd.read_csv(os.path.join(DATA, name))
    X = df.drop(columns="label").to_numpy() / 255.0   # igual que en C++
    y = df["label"].to_numpy()
    return X, y


print("Cargando datos...")
X_train, y_train = load("train_5vs3.csv")
X_test, y_test = load("test_5vs3.csv")
print("Entrenamiento:", X_train.shape, " Prueba:", X_test.shape)


# ------------------------------------------------------------
# 2. El modelo
# El modelo de C++ no usa regularizacion (no castiga pesos grandes).
# Para comparar en igualdad de condiciones, aqui tampoco se usa.
# La forma de pedirlo cambio en Scikit-Learn 1.8, por eso se revisa la version.
# El optimizador por defecto de Scikit-Learn es L-BFGS, no gradiente
# descendiente simple como el de C++.
# ------------------------------------------------------------
major, minor = [int(v) for v in sklearn.__version__.split(".")[:2]]


def new_model(max_iter):
    if (major, minor) >= (1, 8):
        return LogisticRegression(C=np.inf, max_iter=max_iter)
    return LogisticRegression(penalty=None, max_iter=max_iter)


# ------------------------------------------------------------
# 3. Entrenar y medir tiempo real y tiempo de CPU
# ------------------------------------------------------------
print("\nEntrenando...")
wall_start = time.perf_counter()
cpu_start = time.process_time()
model = new_model(max_iter=1000)
model.fit(X_train, y_train)
wall_seconds = time.perf_counter() - wall_start
cpu_seconds = time.process_time() - cpu_start
cpu_percent = 100 * cpu_seconds / wall_seconds
iterations = int(model.n_iter_[0])

p_train = model.predict_proba(X_train)[:, 1]
final_cost = log_loss(y_train, p_train)
print("Convergio en", iterations, "iteraciones")
print("Costo final = %.4f" % final_cost)
print("Tiempo real: %.4f s   Tiempo de CPU: %.4f s   Uso de CPU: %.1f %%"
      % (wall_seconds, cpu_seconds, cpu_percent))


# ------------------------------------------------------------
# 4. Inferencia sobre las imagenes de prueba
# ------------------------------------------------------------
pred_test = model.predict(X_test)
p_test = model.predict_proba(X_test)[:, 1]
for i in range(5):
    print("Imagen de prueba %d: p(5) = %.4f -> dice %d | en realidad es %d"
          % (i + 1, p_test[i], 5 if pred_test[i] == 1 else 3, 5 if y_test[i] == 1 else 3))

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, pred_test)
precision = precision_score(y_test, pred_test)
recall = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
print("\nExactitud en entrenamiento: %.4f" % train_acc)
print("Exactitud en prueba:        %.4f" % test_acc)
print("Precision: %.4f   Recall: %.4f   F1: %.4f" % (precision, recall, f1))

# Matriz de confusion: filas = real (3, 5), columnas = predicho (3, 5)
cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
print("\nMatriz de confusion (prueba):")
print("               dice 3   dice 5")
print("  era 3     %8d %8d" % (cm[0, 0], cm[0, 1]))
print("  era 5     %8d %8d" % (cm[1, 0], cm[1, 1]))


# ------------------------------------------------------------
# 5. Curva de costo
# Scikit-Learn no guarda el costo de cada iteracion. Como L-BFGS
# siempre hace los mismos pasos con los mismos datos, se entrena de
# nuevo deteniendolo en 1, 2, 3, ... iteraciones y se mide el costo.
# ------------------------------------------------------------
print("\nCalculando la curva de costo...")
steps = sorted(set([1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 75,
                    90, 110, 130, 160, 200, 250, 300, iterations]))
steps = [s for s in steps if s <= iterations]
curve = [(0, log_loss(y_train, np.full(len(y_train), 0.5)))]   # costo inicial (pesos en cero)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    for s in steps:
        partial = new_model(max_iter=s).fit(X_train, y_train)
        curve.append((s, log_loss(y_train, partial.predict_proba(X_train)[:, 1])))


# ------------------------------------------------------------
# 6. Guardar resultados con el mismo formato que C++
# ------------------------------------------------------------
pd.DataFrame(curve, columns=["iteracion", "costo"]).to_csv(
    os.path.join(RESULTS, "sklearn_cost.csv"), index=False)

pd.DataFrame([{
    "modelo": "Scikit-Learn", "iteraciones": iterations,
    "tiempo_real_s": wall_seconds, "tiempo_cpu_s": cpu_seconds,
    "uso_cpu_pct": cpu_percent, "costo_final": final_cost,
    "exactitud_entrenamiento": train_acc, "exactitud_prueba": test_acc,
    "precision": precision, "recall": recall, "f1": f1,
}]).to_csv(os.path.join(RESULTS, "sklearn_metrics.csv"), index=False)

pd.DataFrame({"real": [3, 5], "dice_3": cm[:, 0], "dice_5": cm[:, 1]}).to_csv(
    os.path.join(RESULTS, "sklearn_confusion.csv"), index=False)

print("Resultados guardados en la carpeta results/")
print("Versiones: Python", ".".join(map(str, __import__("sys").version_info[:3])),
      "| scikit-learn", sklearn.__version__, "| numpy", np.__version__,
      "| pandas", pd.__version__)
