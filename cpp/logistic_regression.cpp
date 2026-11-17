// logistic_regression.cpp
// Implementacion del motor de regresion logistica.
// Autor: Alessandro De Bonis Rodriguez

#include "logistic_regression.h"

#include <cmath>

// ------------------------------------------------------------
// Sigmoid: sigma(z) = 1 / (1 + e^(-z))
// .array() le dice a Eigen que opere elemento por elemento.
// ------------------------------------------------------------
VectorXd sigmoid(const VectorXd& z) {
    return (1.0 / (1.0 + (-z.array()).exp())).matrix();
}

// ------------------------------------------------------------
// Costo Cross-Entropy:
// J = -(1/m) * suma[ y*ln(p) + (1-y)*ln(1-p) ]
// ------------------------------------------------------------
double crossEntropy(const VectorXd& y, const VectorXd& p) {
    // ln(0) no existe (da -infinito). Para evitarlo, p se mantiene
    // un poquito por encima de 0 y un poquito por debajo de 1.
    const double eps = 1e-12;
    Eigen::ArrayXd pSafe = p.array().max(eps).min(1.0 - eps);
    Eigen::ArrayXd yArr = y.array();

    Eigen::ArrayXd loss = yArr * pSafe.log() + (1.0 - yArr) * (1.0 - pSafe).log();
    return -loss.sum() / y.size();
}

// ------------------------------------------------------------
// Constructor: pesos y sesgo arrancan en cero
// ------------------------------------------------------------
LogisticRegression::LogisticRegression(int numFeatures) {
    weights = VectorXd::Zero(numFeatures);
    bias = 0.0;
}

// p = sigmoid(X * w + b)
VectorXd LogisticRegression::predictProba(const MatrixXd& X) const {
    VectorXd z = (X * weights).array() + bias;
    return sigmoid(z);
}

// 1 si p >= 0.5, 0 si no
VectorXd LogisticRegression::predict(const MatrixXd& X) const {
    VectorXd p = predictProba(X);
    VectorXd labels(p.size());
    for (int i = 0; i < p.size(); i++) {
        labels(i) = (p(i) >= 0.5) ? 1.0 : 0.0;
    }
    return labels;
}

double LogisticRegression::cost(const MatrixXd& X, const VectorXd& y) const {
    return crossEntropy(y, predictProba(X));
}

// ------------------------------------------------------------
// Gradientes con Eigen (una sola linea hace todo el trabajo):
//   dw = (1/m) * X^T * (p - y)
//   db = (1/m) * suma(p - y)
// ------------------------------------------------------------
void LogisticRegression::computeGradients(const MatrixXd& X, const VectorXd& y,
                                          VectorXd& dw, double& db) const {
    int m = X.rows();
    VectorXd error = predictProba(X) - y;   // p - y
    dw = X.transpose() * error / m;
    db = error.sum() / m;
}

// ------------------------------------------------------------
// Los mismos gradientes, escritos con bucles for.
// Da el mismo resultado, pero es mas lento. Sirve para medir
// cuanto ayuda Eigen (seccion de optimizacion en train.cpp).
// ------------------------------------------------------------
void LogisticRegression::computeGradientsLoops(const MatrixXd& X, const VectorXd& y,
                                               VectorXd& dw, double& db) const {
    int m = X.rows();
    int n = X.cols();
    dw = VectorXd::Zero(n);
    db = 0.0;

    for (int i = 0; i < m; i++) {
        // z de la imagen i: suma de pixel * peso, mas el sesgo
        double z = bias;
        for (int j = 0; j < n; j++) {
            z += X(i, j) * weights(j);
        }
        double p = 1.0 / (1.0 + std::exp(-z));
        double error = p - y(i);

        // Cada pixel acumula su parte del gradiente
        for (int j = 0; j < n; j++) {
            dw(j) += error * X(i, j);
        }
        db += error;
    }
    dw = dw / m;
    db = db / m;
}

// w = w - alpha * dw ;  b = b - alpha * db
void LogisticRegression::update(const VectorXd& dw, double db, double learningRate) {
    weights = weights - learningRate * dw;
    bias = bias - learningRate * db;
}

// ------------------------------------------------------------
// Bucle de entrenamiento (gradiente descendiente)
// ------------------------------------------------------------
int LogisticRegression::fit(const MatrixXd& X, const VectorXd& y,
                            double learningRate, int maxIterations, double tolerance) {
    int m = X.rows();
    costHistory.clear();

    for (int it = 0; it < maxIterations; it++) {
        // 1. Probabilidades con los pesos actuales
        VectorXd p = predictProba(X);

        // 2. Costo actual (se guarda para la curva de costo)
        double J = crossEntropy(y, p);
        costHistory.push_back(J);

        // 3. Si el costo ya casi no cambia, el modelo convergio
        if (it > 0 && std::abs(costHistory[it - 1] - J) < tolerance) {
            return it;
        }

        // 4. Gradientes (se reutiliza p para no calcularla dos veces)
        VectorXd error = p - y;
        VectorXd dw = X.transpose() * error / m;
        double db = error.sum() / m;

        // 5. Actualizar pesos
        update(dw, db, learningRate);
    }
    costHistory.push_back(cost(X, y));  // costo despues de la ultima actualizacion
    return maxIterations;
}
