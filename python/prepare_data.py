# prepare_data.py
# Taller Corte 3 - Redes Neuronales
# Autor: Alessandro De Bonis Rodriguez
#
# Prepara los datos que usan los DOS modelos (C++ y Scikit-Learn).
# Asi los dos entrenan exactamente con el mismo dataset.
#
# Genera tres archivos en la carpeta data/:
#   train_5vs3.csv  -> imagenes de entrenamiento (solo 5 y 3)
#   test_5vs3.csv   -> imagenes de prueba (solo 5 y 3)
#   mini_5vs3.csv   -> 6 imagenes resumidas en un solo numero,
#                      para resolver la matematica a mano
#
# Formato de train y test: 784 columnas de pixeles (0 a 255)
# y una ultima columna "label": 1 si es un 5, 0 si es un 3.

import os
import pandas as pd

# Carpetas del proyecto (rutas relativas a este archivo)
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT, "data")
os.makedirs(DATA, exist_ok=True)


# ------------------------------------------------------------
# 1. Cargar MNIST
# Si mnist_784.csv ya existe (lo descargo el cuaderno del Corte 1),
# se usa ese. Si no, se descarga de OpenML y se guarda.
# ------------------------------------------------------------
def load_mnist():
    candidates = [
        os.path.join(DATA, "mnist_784.csv"),
        os.path.join(PROJECT, "mnist_784.csv"),
        os.path.join(os.path.dirname(PROJECT), "mnist_784.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print("Usando", path)
            return pd.read_csv(path)

    print("Descargando MNIST de OpenML (solo la primera vez, unos 15 MB)...")
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, as_frame=True, parser="auto")
    df = mnist.frame
    df["class"] = df["class"].astype(int)
    df.to_csv(candidates[0], index=False)
    return df


df = load_mnist()
df["class"] = df["class"].astype(int)
pixel_columns = [c for c in df.columns if c != "class"]

# ------------------------------------------------------------
# 2. Division oficial de MNIST: las primeras 60.000 imagenes son
#    de entrenamiento y las ultimas 10.000 de prueba.
# ------------------------------------------------------------
train = df.iloc[:60000]
test = df.iloc[60000:]


# ------------------------------------------------------------
# 3. Quedarse solo con los 5 y los 3
# ------------------------------------------------------------
def keep_5_and_3(part):
    part = part[part["class"].isin([5, 3])]
    out = part[pixel_columns].astype(int).copy()
    out["label"] = (part["class"] == 5).astype(int)  # 5 -> 1, 3 -> 0
    return out


train_53 = keep_5_and_3(train)
test_53 = keep_5_and_3(test)

train_53.to_csv(os.path.join(DATA, "train_5vs3.csv"), index=False)
test_53.to_csv(os.path.join(DATA, "test_5vs3.csv"), index=False)

print("Entrenamiento:", len(train_53), "imagenes |",
      int(train_53["label"].sum()), "cincos y",
      int((train_53["label"] == 0).sum()), "treses")
print("Prueba:       ", len(test_53), "imagenes |",
      int(test_53["label"].sum()), "cincos y",
      int((test_53["label"] == 0).sum()), "treses")


# ------------------------------------------------------------
# 4. Mini-dataset para resolver a mano
# Cada imagen se resume en UN solo numero (la caracteristica x):
#   En la franja del medio de la imagen (filas 9 a 13),
#   x = (pixeles con tinta a la izquierda - pixeles con tinta a la derecha) / 10
# Un pixel "tiene tinta" si vale mas de 127.
# En el 5 la tinta del medio se carga a la izquierda (x positivo).
# En el 3 se carga a la derecha (x negativo).
# ------------------------------------------------------------
def middle_feature(row):
    image = row[pixel_columns].to_numpy().reshape(28, 28)
    ink = image > 127
    left = int(ink[9:14, 0:14].sum())
    right = int(ink[9:14, 14:28].sum())
    return left, right, (left - right) / 10


# Posiciones (dentro del set de entrenamiento) de las 6 imagenes escogidas:
# tres 5 y tres 3.
chosen = [0, 47, 100, 7, 10, 12]
expected_digit = [5, 5, 5, 3, 3, 3]

rows = []
for position, digit in zip(chosen, expected_digit):
    row = train.iloc[position]
    if int(row["class"]) != digit:
        raise SystemExit("La imagen " + str(position) + " no es un " + str(digit)
                         + ". Revisa que mnist_784.csv tenga el orden original.")
    left, right, x = middle_feature(row)
    rows.append({"image": position, "digit": digit, "left": left,
                 "right": right, "x": x, "y": 1 if digit == 5 else 0})

mini = pd.DataFrame(rows)
mini.to_csv(os.path.join(DATA, "mini_5vs3.csv"), index=False)
print("\nMini-dataset (para resolver a mano):")
print(mini.to_string(index=False))
