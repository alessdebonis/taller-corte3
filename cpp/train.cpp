// train.cpp
// Laboratorio 3.1 - Parte 2: entrenar el motor con las imagenes de 5 y 3,
// medir tiempo y CPU, y hacer inferencias sobre imagenes nuevas (prueba).
// Autor: Alessandro De Bonis Rodriguez

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <ctime>
#include <Eigen/Dense>

#include "data.h"
#include "logistic_regression.h"
#include "metrics.h"

using namespace std;
using namespace Eigen;

// Hiperparametros (los mismos que se reportan en el articulo)
const double LEARNING_RATE = 0.5;
const int MAX_ITERATIONS = 3000;
const double TOLERANCE = 1e-5;   // si el costo cambia menos que esto, convergio

// Tiempo real transcurrido (como un cronometro)
double secondsSince(chrono::steady_clock::time_point start) {
    return chrono::duration<double>(chrono::steady_clock::now() - start).count();
}

// Tiempo que el procesador trabajo para este programa
double cpuSecondsSince(clock_t start) {
    return double(clock() - start) / CLOCKS_PER_SEC;
}

int main() {
    cout << fixed << setprecision(4);

    // --------------------------------------------------------
    // 1. Cargar los datos (los mismos CSV que usa Scikit-Learn)
    // --------------------------------------------------------
    MatrixXd Xtrain, Xtest;
    VectorXd ytrain, ytest;
    cout << "Cargando datos..." << endl;
    if (!loadImages("data/train_5vs3.csv", Xtrain, ytrain)) return 1;
    if (!loadImages("data/test_5vs3.csv", Xtest, ytest)) return 1;

    cout << "Entrenamiento: " << Xtrain.rows() << " imagenes x " << Xtrain.cols() << " pixeles" << endl;
    cout << "Prueba:        " << Xtest.rows() << " imagenes" << endl;
    cout << "Memoria de la matriz de entrenamiento: "
         << Xtrain.size() * sizeof(double) / (1024.0 * 1024.0) << " MB" << endl;

    // --------------------------------------------------------
    // 2. Optimizacion del gradiente: bucles for vs Eigen
    // Se calcula el mismo gradiente de las dos formas y se mide
    // cuanto tarda cada una en promedio.
    // --------------------------------------------------------
    cout << endl << "=== Optimizacion del gradiente ===" << endl;
    LogisticRegression probe(Xtrain.cols());
    VectorXd dwLoops, dwEigen;
    double dbLoops, dbEigen;

    const int REPS_LOOPS = 5;
    auto t0 = chrono::steady_clock::now();
    for (int r = 0; r < REPS_LOOPS; r++) probe.computeGradientsLoops(Xtrain, ytrain, dwLoops, dbLoops);
    double msLoops = secondsSince(t0) * 1000.0 / REPS_LOOPS;

    const int REPS_EIGEN = 20;
    t0 = chrono::steady_clock::now();
    for (int r = 0; r < REPS_EIGEN; r++) probe.computeGradients(Xtrain, ytrain, dwEigen, dbEigen);
    double msEigen = secondsSince(t0) * 1000.0 / REPS_EIGEN;

    double maxDiff = (dwLoops - dwEigen).cwiseAbs().maxCoeff();
    cout << "Bucles for: " << msLoops << " ms por gradiente" << endl;
    cout << "Eigen:      " << msEigen << " ms por gradiente" << endl;
    cout << "Eigen es " << msLoops / msEigen << " veces mas rapido" << endl;
    cout << "Diferencia maxima entre los dos resultados: " << scientific << maxDiff << fixed << endl;

    ofstream bench("results/cpp_gradient_benchmark.csv");
    bench << "metodo,ms_por_gradiente" << endl;
    bench << "bucles_for," << msLoops << endl;
    bench << "eigen," << msEigen << endl;

    // --------------------------------------------------------
    // 3. Entrenamiento con gradiente descendiente
    // --------------------------------------------------------
    cout << endl << "=== Entrenamiento ===" << endl;
    cout << "alpha = " << LEARNING_RATE << "   maximo de iteraciones = " << MAX_ITERATIONS
         << "   tolerancia = " << scientific << TOLERANCE << fixed << endl;

    LogisticRegression model(Xtrain.cols());
    auto wallStart = chrono::steady_clock::now();
    clock_t cpuStart = clock();

    int iterations = model.fit(Xtrain, ytrain, LEARNING_RATE, MAX_ITERATIONS, TOLERANCE);

    double wallSeconds = secondsSince(wallStart);
    double cpuSeconds = cpuSecondsSince(cpuStart);
    double cpuPercent = 100.0 * cpuSeconds / wallSeconds;

    cout << "Convergio en " << iterations << " iteraciones" << endl;
    cout << "Costo inicial = " << model.costHistory.front()
         << "   costo final = " << model.costHistory.back() << endl;
    cout << "Tiempo real: " << wallSeconds << " s   Tiempo de CPU: " << cpuSeconds
         << " s   Uso de CPU: " << setprecision(1) << cpuPercent << " %" << setprecision(4) << endl;

    // Curva de costo (una fila por iteracion)
    ofstream costFile("results/cpp_cost.csv");
    costFile << "iteracion,costo" << endl;
    for (size_t i = 0; i < model.costHistory.size(); i++) {
        costFile << i << "," << setprecision(8) << model.costHistory[i] << endl;
    }

    // --------------------------------------------------------
    // 4. Inferencia sobre imagenes nuevas (set de prueba)
    // --------------------------------------------------------
    cout << endl << "=== Inferencia sobre imagenes nuevas ===" << endl;
    VectorXd pTest = model.predictProba(Xtest);
    VectorXd predTest = model.predict(Xtest);

    for (int i = 0; i < 5; i++) {
        cout << "Imagen de prueba " << i + 1 << ": p(5) = " << pTest(i)
             << " -> dice " << (predTest(i) == 1.0 ? "5" : "3")
             << " | en realidad es " << (ytest(i) == 1.0 ? "5" : "3") << endl;
    }

    Metrics test = computeMetrics(ytest, predTest);
    Metrics train = computeMetrics(ytrain, model.predict(Xtrain));

    cout << endl << "Exactitud en entrenamiento: " << train.accuracy << endl;
    cout << "Exactitud en prueba:        " << test.accuracy << endl;
    cout << "Precision: " << test.precision << "   Recall: " << test.recall
         << "   F1: " << test.f1 << endl;
    cout << endl << "Matriz de confusion (prueba):" << endl;
    cout << "               dice 3   dice 5" << endl;
    cout << "  era 3     " << setw(8) << test.tn << " " << setw(8) << test.fp << endl;
    cout << "  era 5     " << setw(8) << test.fn << " " << setw(8) << test.tp << endl;

    // --------------------------------------------------------
    // 5. Guardar resultados para las graficas y el articulo
    // --------------------------------------------------------
    ofstream metricsFile("results/cpp_metrics.csv");
    metricsFile << setprecision(6);
    metricsFile << "modelo,iteraciones,tiempo_real_s,tiempo_cpu_s,uso_cpu_pct,costo_final,"
                   "exactitud_entrenamiento,exactitud_prueba,precision,recall,f1" << endl;
    metricsFile << "C++ (Eigen)," << iterations << "," << wallSeconds << "," << cpuSeconds << ","
                << cpuPercent << "," << model.costHistory.back() << "," << train.accuracy << ","
                << test.accuracy << "," << test.precision << "," << test.recall << "," << test.f1 << endl;

    ofstream confFile("results/cpp_confusion.csv");
    confFile << "real,dice_3,dice_5" << endl;
    confFile << "3," << test.tn << "," << test.fp << endl;
    confFile << "5," << test.fn << "," << test.tp << endl;

    // Pesos aprendidos: el modelo queda guardado para usarlo despues
    ofstream weightsFile("results/cpp_weights.csv");
    weightsFile << setprecision(8) << "parametro,valor" << endl;
    weightsFile << "bias," << model.bias << endl;
    for (int j = 0; j < model.weights.size(); j++) {
        weightsFile << "w" << j + 1 << "," << model.weights(j) << endl;
    }

    cout << endl << "Resultados guardados en la carpeta results/" << endl;
    return 0;
}
