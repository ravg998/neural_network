//
// Created by Rayan Ould Maamar on 28/12/2024.
//

#ifndef NEURALNETWORK_ACTIVATIONFUNCTION_CPP
#define NEURALNETWORK_ACTIVATIONFUNCTION_CPP
#include "Activation.hh"
#include "ActivationFunctionType.hh"

// CONSTRUCTOR
Activation::Activation() {
}

void Activation::set_activation_function(const ActivationFunctionType & activation_function_type) {
    _activation_function_type=activation_function_type;
}

// LINEAR
ActivationLinear::ActivationLinear(): Activation() {};

Matrix ActivationLinear::forward(const Matrix & z_matrix) {
    return z_matrix;
}
Matrix ActivationLinear::backward(const Matrix & z_matrix) {
    return Matrix(MatrixType::ID,z_matrix.get_n_row(),z_matrix.get_n_row());
}


// SIGMOID
ActivationSigmoid::ActivationSigmoid(): Activation() {};

Matrix ActivationSigmoid::forward(const Matrix& z_matrix)
{
    return z_matrix.sigmoid();
}

Matrix ActivationSigmoid::backward(const Matrix& z_matrix)
{
    Matrix m(MatrixType::MATRIX_NULL, z_matrix.get_n_row(), z_matrix.get_n_row());
    for (unsigned int i= 0; i<z_matrix.get_n_row(); i++)
        m[i][i] = std::exp(-z_matrix[i][0])/std::pow(1+std::exp(-z_matrix[i][0]),2);
    return m;
}

// RELU
ActivationRelu::ActivationRelu(): Activation() {};

Matrix ActivationRelu::forward(const Matrix& z_matrix)
{
    return z_matrix.max(0);
}

Matrix ActivationRelu::backward(const Matrix& z_matrix)
{
    Matrix m(MatrixType::ID,z_matrix.get_n_row(),z_matrix.get_n_row());
    for (unsigned int i = 0; i<m.get_n_row();i++)
        if (z_matrix[i][0]<=0)
            m[i][i] = 0;

    return m;
}

// TANH
ActivationTanh::ActivationTanh(): Activation() {};

Matrix ActivationTanh::forward(const Matrix& z_matrix)
{
    return z_matrix.tanh();
}

Matrix ActivationTanh::backward(const Matrix& z_matrix)
{
    return z_matrix.tanh();
}

// SOFTMAX
ActivationSoftmax::ActivationSoftmax(): Activation() {};

Matrix ActivationSoftmax::forward(const Matrix& z_matrix)
{
    return z_matrix.softmax();
}

Matrix ActivationSoftmax::backward(const Matrix& z_matrix)
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




#endif //NEURALNETWORK_ACTIVATIONFUNCTION_CPP
