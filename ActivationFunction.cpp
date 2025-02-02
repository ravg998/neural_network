//
// Created by Rayan Ould Maamar on 28/12/2024.
//

#ifndef NEURALNETWORK_ACTIVATIONFUNCTION_CPP
#define NEURALNETWORK_ACTIVATIONFUNCTION_CPP
#include "ActivationFunction.hh"
#include "ActivationFunctionType.hh"

// CONSTRUCTOR
ActivationFunction::ActivationFunction() {
}
ActivationFunction::ActivationFunction(const ActivationFunctionType& activation_function) {
    _activation_function_type = activation_function;
}



// METHOD
Matrix ActivationFunction::forward(const Matrix & z_matrix) {
    if (_activation_function_type==ActivationFunctionType::LINEAR)
        return z_matrix;
    else if (_activation_function_type==ActivationFunctionType::SIGMOID)
        return z_matrix.sigmoid();
    else if (_activation_function_type==ActivationFunctionType::RELU)
        return z_matrix.max(0);
    else if (_activation_function_type==ActivationFunctionType::TANH)
        return z_matrix.tanh();
    else if (_activation_function_type==ActivationFunctionType::SOFTMAX)
        return z_matrix.softmax();
    else
        return Matrix(MatrixType::MATRIX_NULL, 1, 1);
}


Matrix ActivationFunction::backward(const Matrix & z_matrix) {
    if (_activation_function_type==ActivationFunctionType::LINEAR)
        return Matrix(MatrixType::ID,z_matrix.get_n_row(),z_matrix.get_n_row());
    else if (_activation_function_type==ActivationFunctionType::SIGMOID)
    {
        Matrix m(MatrixType::MATRIX_NULL, z_matrix.get_n_row(), z_matrix.get_n_row());
        for (unsigned int i= 0; i<z_matrix.get_n_row(); i++)
                m[i][i] = std::exp(-z_matrix[i][0])/std::pow(1+std::exp(-z_matrix[i][0]),2);
        return m;
    }

    else if (_activation_function_type==ActivationFunctionType::RELU)
    {
        Matrix m(MatrixType::ID,z_matrix.get_n_row(),z_matrix.get_n_row());
        for (unsigned int i = 0; i<m.get_n_row();i++)
                if (z_matrix[i][0]<=0)
                    m[i][i] = 0;

        return m;
    }
    else if (_activation_function_type==ActivationFunctionType::TANH)
        return z_matrix.tanh();
    else if (_activation_function_type==ActivationFunctionType::SOFTMAX)
    {
        Matrix m(MatrixType::ID,z_matrix.get_n_row(),z_matrix.get_n_row());
        for (unsigned int i = 0; i<m.get_n_row();i++)
        {
            for (unsigned int j = 0; j<m.get_n_col(); j++)
            {
                if (i==j)
                    m[i][j] = m[i][j]*(1-m[i][j]);
                else
                    m[i][j] = -m[i][j]*m[i][i];
            }
        }
        return m;

    }
    else
        return Matrix(MatrixType::MATRIX_NULL, 1, 1);
}


// SET & GET

void ActivationFunction::set_activation_function(const ActivationFunctionType & activation_function_type) {
    _activation_function_type=activation_function_type;
}
#endif //NEURALNETWORK_ACTIVATIONFUNCTION_CPP
