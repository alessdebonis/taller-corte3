# Taller Corte 3: Regresión Logística en C++ con Eigen vs Scikit-Learn

**Universidad Sergio Arboleda** · Redes Neuronales · Docente: Oscar Andrés Arias
**Estudiante:** Alessandro De Bonis Rodríguez

Motor de regresión logística escrito desde cero en C++ con la biblioteca **Eigen**, comparado contra la regresión logística de **Scikit-Learn**. Los dos modelos aprenden a distinguir el dígito **5** del dígito **3** en imágenes de MNIST y entrenan con exactamente los mismos archivos de datos.

---

## Contenido del proyecto

```
taller-corte3/
├── CMakeLists.txt              # Archivo de construcción: enlaza Eigen y compila los programas
├── README.md                   # Este archivo
├── run_all.sh                  # Corre todo el taller con un solo comando
├── cpp/
│   ├── data.h / data.cpp       # Lectura de los CSV
│   ├── logistic_regression.h / .cpp   # El motor: sigmoid, costo, gradientes, entrenamiento
│   ├── metrics.h / .cpp        # Matriz de confusión, exactitud, precisión, recall, F1
│   ├── validation.cpp          # Lab 3.1: repite en C++ la matemática hecha a mano
│   ├── train.cpp               # Lab 3.1: entrena con las imágenes y predice imágenes nuevas
│   ├── eda.cpp                 # Lab 3.2: análisis exploratorio (EDA) en C++
│   └── linear_algebra.cpp      # Lab 2.1 del Corte 2: ejercicios de álgebra lineal
├── python/
│   ├── prepare_data.py         # Crea los CSV que usan los dos modelos
│   ├── sklearn_model.py        # Lab 3.2: el mismo experimento con Scikit-Learn
│   └── plots.py                # Lab 3.2: curvas de costo y matrices de confusión
├── data/                       # Datos (los CSV grandes se generan, no se suben)
│   └── mini_5vs3.csv           # Las 6 imágenes del ejercicio hecho a mano
├── results/                    # Aquí quedan los resultados de cada programa
├── docs/                       # Artículo IEEE y documento técnico
└── third_party/eigen/          # Copia de Eigen 3.4.0 (por si no está instalado)
```

---

## Dependencias

| Herramienta | Versión probada | Para qué |
|---|---|---|
| Compilador de C++ con C++17 | Apple Clang 21 (Xcode CLT), g++ 13 | Compilar el motor |
| CMake | 3.10 o más nueva | Construir el proyecto |
| Eigen | 3.4.0 (incluida en `third_party/`) y 5.0.1 (Homebrew) | Álgebra lineal en C++ |
| Python | 3.9.6 (funciona con 3.9 o más nueva) | Preparar datos, Scikit-Learn y gráficas |
| pandas, numpy, scikit-learn, matplotlib | pandas 2.3.3, numpy 2.0.2, scikit-learn 1.6.1 | Lab 3.2 |

**Eigen no es obligatorio instalarlo:** el `CMakeLists.txt` busca primero una versión instalada en el sistema y, si no la encuentra, usa la copia de `third_party/eigen`.

Instalar las librerías de Python:

```bash
python3 -m pip install --user pandas numpy scikit-learn matplotlib
```

---

## Atajo: todo con un solo comando

```bash
bash run_all.sh
```

Corre los pasos 1 a 4 en orden, hace la auditoría de memoria (en macOS), guarda lo que muestra cada programa en `results/` y crea `resultados.zip`.

---

## Paso a paso

Todos los comandos se escriben **desde la carpeta principal del proyecto** (`taller-corte3`), porque los programas buscan los archivos en `data/` y `results/`.

### 1. Preparar los datos

```bash
python3 python/prepare_data.py
```

Usa el archivo `mnist_784.csv` si ya existe (en `data/`, en la carpeta del proyecto o en la carpeta de arriba). Si no, lo descarga de OpenML (necesita internet la primera vez). Genera `data/train_5vs3.csv` (11.552 imágenes), `data/test_5vs3.csv` (1.902 imágenes) y `data/mini_5vs3.csv`.

### 2. Compilar

```bash
cmake -S . -B build
cmake --build build
```

### 3. Correr los programas de C++

```bash
./build/validation       # Lab 3.1: a mano vs C++ (debe decir "9 de 9 valores coinciden")
./build/train            # Lab 3.1: entrenamiento, tiempos e inferencia (unos segundos)
./build/eda              # Lab 3.2: análisis exploratorio en C++
./build/linear_algebra   # Lab 2.1: suma, producto punto, transpuesta, transformaciones
```

En **Windows** con Visual Studio los programas quedan en `build\Release\` (por ejemplo `build\Release\train.exe`) y se compila con `cmake --build build --config Release`.

### 4. Correr Scikit-Learn y hacer las gráficas

```bash
python3 python/sklearn_model.py
python3 python/plots.py
```

Quedan en `results/`: `fig_cost.png`, `fig_confusion.png` y `comparison.csv`.

---

## Auditoría de memoria

El código no usa `new` ni `delete`: la memoria la manejan las matrices de Eigen y los `std::vector`, que se liberan solos. Para comprobar que no hay fugas:

**macOS** (herramienta `leaks` de Apple):

```bash
leaks --atExit -- ./build/validation
leaks --atExit -- ./build/train
```

El resultado esperado es `0 leaks for 0 total leaked bytes`.

**Linux** (AddressSanitizer):

```bash
cmake -S . -B build-asan -DSANITIZE=ON
cmake --build build-asan
./build-asan/train
```

Si hubiera una fuga o un acceso fuera de un arreglo, el programa termina con un reporte `ERROR: AddressSanitizer` o `ERROR: LeakSanitizer`.

---

## Recomendaciones de despliegue

1. **Compilar en modo Release.** El `CMakeLists.txt` lo hace por defecto. En modo Debug el entrenamiento es varias veces más lento y los tiempos no son comparables.
2. **Correr desde la carpeta principal.** Las rutas `data/` y `results/` son relativas.
3. **No subir los CSV grandes al repositorio.** Se regeneran con `prepare_data.py`; el `.gitignore` ya los excluye. Solo se sube `mini_5vs3.csv`.
4. **Mismos datos para los dos modelos.** Si se cambia el dataset, hay que volver a correr `prepare_data.py` antes de `train` y de `sklearn_model.py`.
5. **Los tiempos dependen del computador.** Para comparar C++ y Scikit-Learn hay que correr los dos en la misma máquina y sin otros programas pesados abiertos.
6. **Hiperparámetros del motor de C++:** se cambian al inicio de `cpp/train.cpp` (`LEARNING_RATE`, `MAX_ITERATIONS`, `TOLERANCE`).

---

## Resultados incluidos

La carpeta `results/` trae los resultados reportados en el artículo, obtenidos en un Apple M5 Pro:

| Criterio | C++ (Eigen) | Scikit-Learn |
|---|---|---|
| Iteraciones | 1.029 | 266 |
| Tiempo real | 2,93 s | 0,43 s |
| Exactitud en prueba | 96,42 % | 96,32 % |
| F1 | 0,9616 | 0,9604 |

Validación a mano vs C++: 9 de 9 valores coinciden. Auditoría con `leaks`: 0 fugas.
