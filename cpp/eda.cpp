// eda.cpp
// Laboratorio 3.2: rutinas de analisis exploratorio (EDA) en C++.
// Autor: Alessandro De Bonis Rodriguez
//
// Explora el set de entrenamiento antes de entrenar:
//   1. Cuantas imagenes hay de cada clase
//   2. Que tan "vacias" son las imagenes (pixeles en cero)
//   3. Pixeles que nunca cambian
//   4. Brillo promedio de cada clase
//   5. La caracteristica del mini-dataset (tinta izquierda - derecha)
//   6. La imagen promedio de cada clase, dibujada con caracteres

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <Eigen/Dense>

#include "data.h"

using namespace std;
using namespace Eigen;

// Dibuja una imagen de 28 x 28 con caracteres: mas tinta, caracter mas "lleno"
void drawImage(const VectorXd& image) {
    const string shades = " .:-=+*#%@";
    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            double v = image(row * 28 + col);          // entre 0 y 1
            int level = int(v * (shades.size() - 1) + 0.5);
            cout << shades[level] << shades[level];   // doble para que no se vea aplastada
        }
        cout << endl;
    }
}

// Caracteristica del mini-dataset para una imagen:
// en las filas 9 a 13, (pixeles con tinta a la izquierda - a la derecha) / 10
double middleFeature(const MatrixXd& X, int i) {
    int left = 0, right = 0;
    for (int row = 9; row <= 13; row++) {
        for (int col = 0; col < 28; col++) {
            bool ink = X(i, row * 28 + col) > 127.0 / 255.0;
            if (ink && col < 14) left++;
            if (ink && col >= 14) right++;
        }
    }
    return (left - right) / 10.0;
}

int main() {
    cout << fixed << setprecision(2);

    MatrixXd X;
    VectorXd y;
    cout << "Cargando datos de entrenamiento..." << endl;
    if (!loadImages("data/train_5vs3.csv", X, y)) return 1;
    int m = X.rows();
    int n = X.cols();

    // 1. Balance de clases
    int fives = int(y.sum());
    int threes = m - fives;
    cout << endl << "=== 1. Clases ===" << endl;
    cout << "Imagenes: " << m << "   pixeles por imagen: " << n << endl;
    cout << "Cincos: " << fives << " (" << 100.0 * fives / m << " %)" << endl;
    cout << "Treses: " << threes << " (" << 100.0 * threes / m << " %)" << endl;

    // 2. Pixeles en cero (fondo negro)
    int zeros = (X.array() == 0.0).count();
    cout << endl << "=== 2. Fondo ===" << endl;
    cout << "Pixeles que valen 0: " << 100.0 * zeros / X.size() << " %" << endl;

    // 3. Pixeles que nunca cambian (valen lo mismo en todas las imagenes)
    VectorXd colMax = X.colwise().maxCoeff();
    VectorXd colMin = X.colwise().minCoeff();
    int constant = ((colMax - colMin).array() == 0.0).count();
    cout << endl << "=== 3. Pixeles constantes ===" << endl;
    cout << constant << " de " << n << " pixeles valen lo mismo en todas las imagenes" << endl;

    // 4. Brillo promedio y 5. caracteristica del medio, por clase
    VectorXd mean5 = VectorXd::Zero(n), mean3 = VectorXd::Zero(n);
    double feat5 = 0, feat3 = 0;
    int ruleOk = 0;   // aciertos de la regla "si x > 0, es un 5"
    for (int i = 0; i < m; i++) {
        double f = middleFeature(X, i);
        if (y(i) == 1.0) { mean5 += X.row(i).transpose(); feat5 += f; }
        else             { mean3 += X.row(i).transpose(); feat3 += f; }
        bool says5 = f > 0;
        if (says5 == (y(i) == 1.0)) ruleOk++;
    }
    mean5 /= fives;
    mean3 /= threes;
    feat5 /= fives;
    feat3 /= threes;

    cout << endl << "=== 4. Brillo promedio (0 a 255) ===" << endl;
    cout << "Cincos: " << mean5.mean() * 255 << "   Treses: " << mean3.mean() * 255 << endl;

    cout << endl << "=== 5. Caracteristica del medio (izquierda - derecha) / 10 ===" << endl;
    cout << "Promedio en los cincos: " << feat5 << endl;
    cout << "Promedio en los treses: " << feat3 << endl;
    cout << "La regla 'si x > 0 es un 5' acierta el " << 100.0 * ruleOk / m << " % de las imagenes" << endl;

    // 6. Imagen promedio de cada clase
    cout << endl << "=== 6. Imagen promedio de los cincos ===" << endl;
    drawImage(mean5);
    cout << endl << "=== Imagen promedio de los treses ===" << endl;
    drawImage(mean3);

    ofstream out("results/cpp_eda.csv");
    out << "medida,valor" << endl;
    out << "imagenes," << m << endl;
    out << "cincos," << fives << endl;
    out << "treses," << threes << endl;
    out << "pct_pixeles_en_cero," << 100.0 * zeros / X.size() << endl;
    out << "pixeles_constantes," << constant << endl;
    out << "brillo_promedio_cincos," << mean5.mean() * 255 << endl;
    out << "brillo_promedio_treses," << mean3.mean() * 255 << endl;
    out << "caracteristica_promedio_cincos," << feat5 << endl;
    out << "caracteristica_promedio_treses," << feat3 << endl;
    out << "pct_aciertos_regla_x_mayor_0," << 100.0 * ruleOk / m << endl;
    return 0;
}
