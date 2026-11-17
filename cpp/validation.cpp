// validation.cpp
// Laboratorio 3.1 - Parte 1: validar la matematica hecha a mano.
// Autor: Alessandro De Bonis Rodriguez
//
// Usa el mini-dataset (6 imagenes resumidas en un numero x) y repite,
// con el motor de C++, los mismos calculos del documento tecnico:
//   1. Regresion lineal (formulas de minimos cuadrados y ecuacion normal)
//   2. Dos iteraciones de regresion logistica con gradiente descendiente
// Al final compara cada numero con el valor obtenido a mano.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <Eigen/Dense>

#include "data.h"
#include "logistic_regression.h"

using namespace std;
using namespace Eigen;

// Valores calculados a mano en el documento tecnico (4 decimales)
struct Check {
    string name;
    double byHand;
    double cpp;
};

int main() {
    cout << fixed << setprecision(4);

    VectorXd x, y;
    if (!loadMini("data/mini_5vs3.csv", x, y)) return 1;
    int m = x.size();

    cout << "=== Mini-dataset ===" << endl;
    for (int i = 0; i < m; i++) {
        cout << "Imagen " << i + 1 << ":  x = " << setw(7) << x(i)
             << "   y = " << y(i) << (y(i) == 1.0 ? "  (es un 5)" : "  (es un 3)") << endl;
    }

    // --------------------------------------------------------
    // 1. Regresion lineal: y = w*x + b
    // --------------------------------------------------------
    cout << endl << "=== 1. Regresion lineal ===" << endl;

    // Forma 1: formulas de minimos cuadrados (como a mano)
    double xMean = x.mean();
    double yMean = y.mean();
    VectorXd dx = x.array() - xMean;   // distancia de cada x a su promedio
    VectorXd dy = y.array() - yMean;
    double sxy = dx.dot(dy);           // suma de (x - xMedia)(y - yMedia)
    double sxx = dx.dot(dx);           // suma de (x - xMedia)^2
    double wLin = sxy / sxx;
    double bLin = yMean - wLin * xMean;

    cout << "Promedios: x = " << xMean << "   y = " << yMean << endl;
    cout << "Sxy = " << sxy << "   Sxx = " << sxx << endl;
    cout << "w = Sxy / Sxx = " << wLin << "   b = " << bLin << endl;

    // Forma 2: ecuacion normal con matrices de Eigen
    // A tiene una columna de unos (para b) y una columna con x (para w)
    MatrixXd A(m, 2);
    A.col(0) = VectorXd::Ones(m);
    A.col(1) = x;
    Matrix2d AtA = A.transpose() * A;
    Vector2d Aty = A.transpose() * y;
    Vector2d theta = AtA.ldlt().solve(Aty);  // resuelve (A^T A) theta = A^T y

    cout << endl << "Ecuacion normal (A^T A) theta = A^T y:" << endl;
    cout << "A^T A =" << endl << AtA << endl;
    cout << "A^T y = " << Aty.transpose() << endl;
    cout << "theta = [b, w] = " << theta.transpose() << endl;

    cout << endl << "Predicciones de la recta:" << endl;
    VectorXd yLin = (wLin * x).array() + bLin;
    for (int i = 0; i < m; i++) {
        cout << "x = " << setw(7) << x(i) << "  ->  y = " << setw(7) << yLin(i);
        if (yLin(i) > 1.0 || yLin(i) < 0.0) cout << "   <- fuera de [0, 1]: no es una probabilidad";
        cout << endl;
    }

    // --------------------------------------------------------
    // 2. Regresion logistica: dos iteraciones, paso a paso
    // --------------------------------------------------------
    cout << endl << "=== 2. Regresion logistica (alpha = 0.5) ===" << endl;

    MatrixXd X = x;                      // matriz de 6 filas y 1 columna
    LogisticRegression model(1);         // el MISMO motor que se usa con las imagenes
    double learningRate = 0.5;
    double costs[3];
    double ws[3], bs[3];
    cout << setprecision(6);  // en los pasos se muestran 6 decimales

    for (int it = 0; it <= 2; it++) {
        VectorXd p = model.predictProba(X);
        costs[it] = crossEntropy(y, p);
        ws[it] = model.weights(0);
        bs[it] = model.bias;

        cout << endl << "-- Iteracion " << it << ":  w = " << model.weights(0)
             << "   b = " << model.bias << endl;
        cout << "z = " << ((X * model.weights).array() + model.bias).transpose() << endl;
        cout << "p = " << p.transpose() << endl;
        cout << "Costo J = " << costs[it] << endl;

        if (it == 2) break;  // en la iteracion 2 solo se revisa el costo

        VectorXd dw;
        double db;
        model.computeGradients(X, y, dw, db);
        cout << "p - y = " << (p - y).transpose() << endl;
        cout << "dw = " << dw(0) << "   db = " << db << endl;
        model.update(dw, db, learningRate);
    }

    // --------------------------------------------------------
    // 3. Comparacion con los valores calculados a mano
    // --------------------------------------------------------
    Check checks[] = {
        {"Regresion lineal: w", 0.3604, wLin},
        {"Regresion lineal: b", 0.5541, bLin},
        {"Logistica: costo inicial J0", 0.6931, costs[0]},
        {"Logistica: w despues de 1 paso", 0.2792, ws[1]},
        {"Logistica: b despues de 1 paso", 0.0000, bs[1]},
        {"Logistica: costo J1", 0.5525, costs[1]},
        {"Logistica: w despues de 2 pasos", 0.5044, ws[2]},
        {"Logistica: b despues de 2 pasos", 0.0050, bs[2]},
        {"Logistica: costo J2", 0.4600, costs[2]},
    };

    cout << setprecision(4);
    cout << endl << "=== 3. A mano vs C++ (4 decimales) ===" << endl;
    ofstream out("results/validation.csv");
    out << "valor,a_mano,cpp,coincide" << endl;
    int ok = 0, total = 0;
    for (const Check& c : checks) {
        bool same = std::abs(c.byHand - c.cpp) < 0.00005 + 1e-9;
        cout << left << setw(34) << c.name << right
             << " a mano: " << setw(7) << c.byHand
             << "   C++: " << setw(7) << c.cpp
             << "   " << (same ? "coincide" : "NO coincide") << endl;
        out << c.name << "," << c.byHand << "," << c.cpp << "," << (same ? "si" : "no") << endl;
        if (same) ok++;
        total++;
    }
    cout << ok << " de " << total << " valores coinciden." << endl;
    return 0;
}
