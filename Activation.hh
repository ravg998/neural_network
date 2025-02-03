//
// Created by Rayan Ould Maamar on 28/12/2024.
//

#ifndef NEURALNETWORK_ACTIVATION_HH
#define NEURALNETWORK_ACTIVATION_HH
#include "../math/Matrix.h"
#include "ActivationFunctionType.hh"
class Activation
{
private:
    ActivationFunctionType _activation_function_type;
public:
    // CONSTRUCTOR
    Activation();

    // METHOD
    virtual Matrix forward(const Matrix&) = 0;
    virtual Matrix backward(const Matrix&)= 0;

    // SET & GET
    void set_activation_function(const ActivationFunctionType&);
};

// LINEAR
class ActivationLinear: public Activation
{
private:
    ActivationFunctionType _activation_type = ActivationFunctionType::LINEAR;

public:
    ActivationLinear();

    // METHOD
    Matrix forward(const Matrix&) override;
    Matrix backward(const Matrix&) override;
};


//SIGMOID
class ActivationSigmoid: public Activation
{
private:
    ActivationFunctionType _activation_type = ActivationFunctionType::SIGMOID;

public:
    ActivationSigmoid();

    // METHOD
    Matrix forward(const Matrix&) override;
    Matrix backward(const Matrix&) override;
};


//SIGMOID
class ActivationRelu: public Activation
{
private:
    ActivationFunctionType _activation_type = ActivationFunctionType::RELU;

public:
    ActivationRelu();

    // METHOD
    Matrix forward(const Matrix&) override;
    Matrix backward(const Matrix&) override;
};

//SIGMOID
class ActivationTanh: public Activation
{
private:
    ActivationFunctionType _activation_type = ActivationFunctionType::TANH;

public:
    ActivationTanh();

    // METHOD
    Matrix forward(const Matrix&) override;
    Matrix backward(const Matrix&) override;
};


//SIGMOID
class ActivationSoftmax: public Activation
{
private:
    ActivationFunctionType _activation_type = ActivationFunctionType::SOFTMAX;

public:
    ActivationSoftmax();

    // METHOD
    Matrix forward(const Matrix&) override;
    Matrix backward(const Matrix&) override;
};

#endif //NEURALNETWORK_ACTIVATION_HH
