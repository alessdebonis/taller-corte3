#!/bin/bash
# run_all.sh
# Corre todo el taller en orden y guarda lo que muestra cada programa
# en la carpeta results/. Al final crea resultados.zip para enviarlo.
# Uso (desde la carpeta taller-corte3):   bash run_all.sh

set -e
cd "$(dirname "$0")"
mkdir -p results

echo "1/8 Preparando datos..."
python3 python/prepare_data.py | tee results/log_prepare_data.txt

echo "2/8 Compilando..."
cmake -S . -B build > results/log_build.txt
cmake --build build >> results/log_build.txt

echo "3/8 Auditoria de memoria (va antes para que no sobrescriba los resultados)..."
if command -v leaks > /dev/null; then
    leaks --atExit -- ./build/validation > results/log_leaks_validation.txt 2>&1 || true
    leaks --atExit -- ./build/train > results/log_leaks_train.txt 2>&1 || true
    grep -h "leaks for" results/log_leaks_*.txt || true
else
    echo "La herramienta leaks solo existe en macOS (en Linux usar -DSANITIZE=ON)."
fi

echo "4/8 Validacion (a mano vs C++)..."
./build/validation | tee results/log_validation.txt

echo "5/8 Entrenamiento en C++..."
./build/train | tee results/log_train.txt

echo "6/8 EDA en C++..."
./build/eda | tee results/log_eda.txt

echo "7/8 Algebra lineal (Lab 2.1)..."
./build/linear_algebra > results/log_linear_algebra.txt

echo "8/8 Scikit-Learn y graficas..."
python3 python/sklearn_model.py | tee results/log_sklearn.txt
python3 python/plots.py | tee results/log_plots.txt

# Datos del computador (para el articulo)
{
    echo "Procesador: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo)"
    echo "Sistema: $(uname -sr)"
    echo "Compilador: $(c++ --version | head -1)"
    echo "CMake: $(cmake --version | head -1)"
    echo "Python: $(python3 --version)"
} > results/log_machine.txt
cat results/log_machine.txt

rm -f resultados.zip
zip -rq resultados.zip results
echo
echo "Listo. Envia el archivo resultados.zip"
