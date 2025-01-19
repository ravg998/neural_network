//
// Created by Rayan Ould Maamar on 12/01/2025.
//

#ifndef NEURALNETWORK_NEURALNETWORK_HH
#define NEURALNETWORK_NEURALNETWORK_HH
#include "Layer.hh"
#include "Loss.hh"


class NeuralNetwork
{
private:
    // INPUTS
    Matrix _x;
    Matrix _y_hat;
    unsigned int _n_layer;
    Vect<Layer> _layer_vector;
    unsigned int _n_observation;

    //
    Loss _loss;

    // OUTPUT
    Matrix _y;


public:
    // CONSTRUCTOR
    NeuralNetwork(const Vect<Layer>&, const Matrix&, const Matrix&);

    // METHOD
    void forward();
    void compute_loss();
    void backward();
    void train_model(unsigned int);
    Matrix predict();

    // MISC
    void print_loss() const;
    void print_loss(unsigned int, unsigned int) const;

    // SET & GET
    Layer& get_final_layer() const;
    Loss get_loss() const;

};

#endif //NEURALNETWORK_NEURALNETWORK_HH
