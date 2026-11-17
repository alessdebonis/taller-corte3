// data.cpp
// Lectura de los archivos CSV.
// Autor: Alessandro De Bonis Rodriguez

#include "data.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

using namespace std;

// Separa una linea del CSV por comas y devuelve cada valor como texto.
static vector<string> splitLine(const string& line) {
    vector<string> cells;
    stringstream ss(line);
    string cell;
    while (getline(ss, cell, ',')) {
        cells.push_back(cell);
    }
    return cells;
}

bool loadImages(const string& path, MatrixXd& X, VectorXd& y) {
    ifstream file(path);
    if (!file.is_open()) {
        cout << "No se pudo abrir " << path << endl;
        cout << "Primero corre: python3 python/prepare_data.py" << endl;
        return false;
    }

    string line;
    getline(file, line);  // la primera linea son los nombres de las columnas

    // Primero se guardan las filas en vectores normales,
    // porque todavia no sabemos cuantas imagenes hay.
    vector<vector<double>> rows;
    vector<double> labels;
    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<string> cells = splitLine(line);
        vector<double> pixels;
        // Todas las columnas menos la ultima son pixeles
        for (size_t j = 0; j + 1 < cells.size(); j++) {
            pixels.push_back(stod(cells[j]) / 255.0);
        }
        rows.push_back(pixels);
        labels.push_back(stod(cells.back()));  // la ultima columna es la etiqueta
    }

    // Ahora que sabemos el tamano, se pasa todo a la matriz de Eigen
    int m = rows.size();
    int n = rows[0].size();
    X.resize(m, n);
    y.resize(m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            X(i, j) = rows[i][j];
        }
        y(i) = labels[i];
    }
    // No hace falta liberar memoria a mano: los vector y las matrices de Eigen
    // se liberan solos cuando terminan su uso. Por eso no hay new ni delete.
    return true;
}

bool loadMini(const string& path, VectorXd& x, VectorXd& y) {
    ifstream file(path);
    if (!file.is_open()) {
        cout << "No se pudo abrir " << path << endl;
        return false;
    }

    string line;
    getline(file, line);  // encabezado: image,digit,left,right,x,y

    vector<double> xs, ys;
    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<string> cells = splitLine(line);
        xs.push_back(stod(cells[4]));  // columna x
        ys.push_back(stod(cells[5]));  // columna y
    }

    int m = xs.size();
    x.resize(m);
    y.resize(m);
    for (int i = 0; i < m; i++) {
        x(i) = xs[i];
        y(i) = ys[i];
    }
    return true;
}
