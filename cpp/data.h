// data.h
// Funciones para leer los archivos CSV que genera python/prepare_data.py
// Autor: Alessandro De Bonis Rodriguez

#ifndef DATA_H
#define DATA_H

#include <string>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Lee un CSV de imagenes (784 columnas de pixeles + columna "label").
// X queda con una fila por imagen y una columna por pixel.
// Los pixeles se dividen entre 255 para que queden entre 0 y 1.
// y queda con la etiqueta de cada imagen: 1 si es un 5, 0 si es un 3.
// X y y se reciben por referencia (&): la funcion los llena sin copiarlos.
bool loadImages(const std::string& path, MatrixXd& X, VectorXd& y);

// Lee el mini-dataset (columnas image, digit, left, right, x, y).
// Devuelve la caracteristica x de cada imagen y su etiqueta y.
bool loadMini(const std::string& path, VectorXd& x, VectorXd& y);

#endif
