// metrics.cpp
// Autor: Alessandro De Bonis Rodriguez

#include "metrics.h"

Metrics computeMetrics(const VectorXd& yTrue, const VectorXd& yPred) {
    Metrics r = {0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0};

    // Se cuenta cada caso de la matriz de confusion
    for (int i = 0; i < yTrue.size(); i++) {
        bool real5 = (yTrue(i) == 1.0);
        bool said5 = (yPred(i) == 1.0);
        if (real5 && said5) r.tp++;
        else if (!real5 && !said5) r.tn++;
        else if (!real5 && said5) r.fp++;
        else r.fn++;
    }

    double total = yTrue.size();
    r.accuracy = (r.tp + r.tn) / total;
    r.precision = (r.tp + r.fp > 0) ? (double)r.tp / (r.tp + r.fp) : 0.0;
    r.recall = (r.tp + r.fn > 0) ? (double)r.tp / (r.tp + r.fn) : 0.0;
    r.f1 = (r.precision + r.recall > 0)
               ? 2.0 * r.precision * r.recall / (r.precision + r.recall)
               : 0.0;
    return r;
}
