# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and settings
project(NeuralNetwork)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable target
add_executable(output main.cpp ../math/Matrix.cpp ../math/Random.cpp
        Layer.hh
        Layer.cpp
        ActivationFunction.hh
        ActivationFunctionType.hh
        ActivationFunction.cpp
        NeuralNetwork.cpp
        NeuralNetwork.hh
        Loss.cpp
        Loss.hh
        LossType.hh
)