// logistic_regression.h
// Motor de regresion logistica con Eigen.
// Autor: Alessandro De Bonis Rodriguez
//
// Ecuaciones que implementa:
//   z = X * w + b                                 (puntaje lineal)
//   p = sigmoid(z) = 1 / (1 + e^(-z))             (probabilidad de ser un 5)
//   J = -(1/m) * suma[ y*ln(p) + (1-y)*ln(1-p) ]  (costo Cross-Entropy)
//   dw = (1/m) * X^T * (p - y)                    (gradiente de los pesos)
//   db = (1/m) * suma(p - y)                      (gradiente del sesgo)
//   w = w - alpha * dw ;  b = b - alpha * db      (gradiente descendiente)

#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Sigmoid aplicada a cada elemento de z
VectorXd sigmoid(const VectorXd& z);

// Costo Cross-Entropy entre etiquetas reales y y probabilidades p
double crossEntropy(const VectorXd& y, const VectorXd& p);

class LogisticRegression {
public:
    VectorXd weights;               // w: un peso por pixel (o por caracteristica)
    double bias;                    // b: el sesgo
    std::vector<double> costHistory;  // costo de cada iteracion (para la curva)

    // Crea el modelo con todos los pesos en cero
    LogisticRegression(int numFeatures);

    // Probabilidad de que cada fila de X sea un 5
    VectorXd predictProba(const MatrixXd& X) const;

    // Prediccion final: 1 si p >= 0.5 (es un 5), 0 si no (es un 3)
    VectorXd predict(const MatrixXd& X) const;

    // Costo del modelo actual sobre X, y
    double cost(const MatrixXd& X, const VectorXd& y) const;

    // Calcula los gradientes dw y db (version con Eigen, vectorizada)
    void computeGradients(const MatrixXd& X, const VectorXd& y,
                          VectorXd& dw, double& db) const;

    // Mismos gradientes, pero con bucles for (version lenta, para comparar)
    void computeGradientsLoops(const MatrixXd& X, const VectorXd& y,
                               VectorXd& dw, double& db) const;

    // Un paso de gradiente descendiente
    void update(const VectorXd& dw, double db, double learningRate);

    // Entrena hasta que el costo deje de bajar (cambio menor que tolerance)
    // o hasta maxIterations. Devuelve cuantas iteraciones hizo.
    int fit(const MatrixXd& X, const VectorXd& y,
            double learningRate, int maxIterations, double tolerance);
};

#endif
