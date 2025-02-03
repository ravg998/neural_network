//
// Created by Rayan Ould Maamar on 12/01/2025.
//

#include "NeuralNetwork.hh"
#include "LossType.hh"
// CONSTRUCTOR
NeuralNetwork::NeuralNetwork(const Vect<Layer>& layer_vector,
                             const Matrix & input,
                             const Matrix & output,
                             Loss& loss){
   _layer_vector = Vect<Layer>(layer_vector);
   _x = input;
    _y = output;
   _n_layer = _layer_vector.get_size();
   _loss = &loss;
   _loss->set_y(_y);
   _n_observation = _layer_vector[0].get_n_observation();

}

// METHOD
void NeuralNetwork::forward() {
    _layer_vector[0].forward();
    for (unsigned int i = 1; i<_n_layer; i++)
    {
        _layer_vector[i].forward(_layer_vector[i-1].get_y());

    }
    _y_hat = this->get_final_layer().get_y();

}

void NeuralNetwork::compute_loss()
{
    _loss->compute_loss(this->get_final_layer().get_y());
}


void NeuralNetwork::backward()
{
    _loss->compute_dl_dy();

    this->get_final_layer().backward(_loss->get_dl_dy());

    this->get_final_layer().update_weight();

    for ( int i = _n_layer-2; i>=0; i--)
    {
        _layer_vector[i].backward(_layer_vector[i+1].get_dl_dx());
        _layer_vector[i].update_weight();
        _layer_vector[i].update_bias();
    };
}

void NeuralNetwork::train_model(unsigned int n_step) {
    for (unsigned int i = 0; i<n_step;i++)
    {
        this->forward();
        this->compute_loss();
        this->backward();
        this->print_loss(i, 100);
    }
}

Matrix NeuralNetwork::predict()  {
    this->forward();
    return _y;
}

// MISC
void NeuralNetwork::print_loss() const {
    std::cout <<  this->get_loss()->get_loss().sum().sum()  /_n_observation << std::endl;
}

void NeuralNetwork::print_loss(unsigned int i, unsigned int div) const {
    if (i%div == 0)
    {
        std::cout << "EPOCH " << i <<": ";
        this->print_loss();

    }
}

// SET & GET
Layer& NeuralNetwork::get_final_layer() const
{
    return _layer_vector[_n_layer - 1];
}

Loss* NeuralNetwork::get_loss() const
{
    return _loss;
}



