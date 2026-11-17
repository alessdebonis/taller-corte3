// metrics.h
// Metricas para evaluar el clasificador.
// Autor: Alessandro De Bonis Rodriguez
//
// Clase positiva = 5 (etiqueta 1). Clase negativa = 3 (etiqueta 0).
//   VP: era 5 y dijo 5      FN: era 5 y dijo 3
//   FP: era 3 y dijo 5      VN: era 3 y dijo 3

#ifndef METRICS_H
#define METRICS_H

#include <Eigen/Dense>

using Eigen::VectorXd;

struct Metrics {
    int tp, tn, fp, fn;   // matriz de confusion
    double accuracy;      // (VP + VN) / total
    double precision;     // VP / (VP + FP)
    double recall;        // VP / (VP + FN)
    double f1;            // 2 * precision * recall / (precision + recall)
};

Metrics computeMetrics(const VectorXd& yTrue, const VectorXd& yPred);

#endif
