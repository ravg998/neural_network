//
// Created by Rayan Ould Maamar on 28/12/2024.
//

#ifndef NEURALNETWORK_ACTIVATIONFUNCTION_HH
#define NEURALNETWORK_ACTIVATIONFUNCTION_HH
#include "../math/Matrix.h"
#include "ActivationFunctionType.hh"
class ActivationFunction
{
private:
    ActivationFunctionType _activation_function_type;
public:
    // CONSTRUCTOR
    ActivationFunction();
    ActivationFunction(const ActivationFunctionType&);

    // METHOD
    Matrix forward(const Matrix&);
    Matrix backward(const Matrix&);

    // SET & GET
    void set_activation_function(const ActivationFunctionType&);
};
#endif //NEURALNETWORK_ACTIVATIONFUNCTION_HH
