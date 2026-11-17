cmake_minimum_required(VERSION 3.10)
project(TallerCorte3 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Si no se indica el tipo de compilacion, se compila optimizado (-O2)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Genera build/compile_commands.json para que VS Code encuentre Eigen
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------
# Eigen
# 1. Si Eigen esta instalado en el sistema, se usa ese.
# 2. Si no, se usa la copia incluida en third_party/eigen.
# ------------------------------------------------------------
find_package(Eigen3 QUIET NO_MODULE)

if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen: usando la version instalada en el sistema (${Eigen3_VERSION})")
else()
    message(STATUS "Eigen: usando la copia incluida en third_party/eigen")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
endif()

# ------------------------------------------------------------
# Auditoria de memoria (opcional):
#   cmake -S . -B build-asan -DSANITIZE=ON
# Compila con AddressSanitizer, que avisa si hay fugas de memoria
# o accesos fuera de los limites de un arreglo.
# ------------------------------------------------------------
option(SANITIZE "Compilar con AddressSanitizer" OFF)
if(SANITIZE)
    message(STATUS "AddressSanitizer activado")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
endif()

# El motor (se comparte entre los programas)
add_library(engine cpp/data.cpp cpp/logistic_regression.cpp cpp/metrics.cpp)
target_link_libraries(engine PUBLIC Eigen3::Eigen)

# Laboratorio 3.1: validacion de la matematica hecha a mano
add_executable(validation cpp/validation.cpp)
target_link_libraries(validation engine)

# Laboratorio 3.1: entrenamiento e inferencia con las imagenes
add_executable(train cpp/train.cpp)
target_link_libraries(train engine)

# Laboratorio 3.2: rutinas de EDA en C++
add_executable(eda cpp/eda.cpp)
target_link_libraries(eda engine)

# Laboratorio 2.1 (Corte 2): ejercicios de algebra lineal
add_executable(linear_algebra cpp/linear_algebra.cpp)
target_link_libraries(linear_algebra Eigen3::Eigen)
