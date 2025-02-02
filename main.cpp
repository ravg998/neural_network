#include <iostream>
#include "/Users/rayanouldmaamar/Document/C++/math/Matrix.h"
#include "Layer.hh"
#include "ActivationFunctionType.hh"
#include "NeuralNetwork.hh"

int main() {
    unsigned int n_feature = 5;
    unsigned int n_observation = 10;
    unsigned int n_output = 5;
    unsigned int n_step = 10'000;
    ActivationFunctionType activation_function = ActivationFunctionType::SIGMOID;
    Matrix input(MatrixType::RANDOM, n_feature,n_observation);
    Matrix output(MatrixType::RANDOM, n_feature,n_observation);
    Layer l1(input,n_output, activation_function);
    Layer l2(input,n_output, activation_function);
    Layer l3(input,n_output, activation_function);
    output = output.max(0.1);
    output = output.min(1);
    Vect<Layer> l;
    l.append(l1);
    l.append(l2);
    l.append(l3);
    NeuralNetwork nn(l,
                     input,
                     output
                     );

    std::cout << "INPUT: " << std::endl;
    std::cout << input.transpose() << std::endl;
    std::cout << "OUTPUT: " << std::endl;

    std::cout << output.transpose() << std::endl;

    std::cout << "NAIVE PREDICTION:" << std::endl;
    std::cout << nn.predict().transpose() << std::endl;

    nn.train_model(n_step);

    std::cout << "TRAINED PREDICTION:" << std::endl;
    std::cout << nn.predict().transpose() << std::endl;

    std::cout << "LOSS:" << std::endl;
    nn.print_loss() ;

    return 0;
}
