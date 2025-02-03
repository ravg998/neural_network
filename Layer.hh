//
// Created by Rayan Ould Maamar on 28/12/2024.
//

#ifndef NEURALNETWORK_LAYER_HH
#define NEURALNETWORK_LAYER_HH
#include "../math/Matrix.h"
#include "Activation.hh"
/*
Here, we assume that:
    input  = _x
    w*_x --> _z
    f(_z) --> _y_hat


X = (n_features x n_observations)
W = (n_output x n_features)
*/

class Layer
{
private:
    // INPUTS
    Matrix _x;
    Activation* _activation_function;
    unsigned int _n_output;
    unsigned int _n_observation;

    // INTERMEDIATES
    Matrix _z;
    Matrix _bias;
    Matrix _weight;

    // DERIVATIVES
    Matrix _dy_dz;
    Matrix _dz_dw;
    Matrix _dz_dx;
    Matrix _dl_dx;
    Matrix _dz_db;
    Matrix _dl_dy;
    Matrix _dl_dw;
    Matrix _dl_db;

    // OUTPUTS
    Matrix _y_hat;

public:
    // CONSTRUCTOR
    Layer();
    Layer(const Matrix&, unsigned int,  Activation&);

    // METHOD
    /// INITIALIZATION
    void define_weight();
    void define_bias();

    void reset_dy_dz();
    void reset_dl_dw();
    void reset_dl_dx();
    void reset_dl_db();

    void reset_derivatives();

    /// FORWARD
    void update_input(const Matrix&);
    void compute_z();
    void compute_y();

    void forward();
    void forward(const Matrix&);

    /// BACKWARD
    void compute_dy_dz(unsigned int i = 0);
    void compute_dz_dx();
    void compute_dz_dw(unsigned int i = 0);
    void compute_dz_db();
    void compute_dl_dx(const Matrix&);
    void compute_dl_dw(const Matrix&, unsigned int);
    void compute_dl_db(const Matrix&);
    void update_weight();
    void update_bias();
    void backward(const Matrix&);


    // GET & SET
    unsigned int get_n_observation() const;
    Matrix get_x() const;
    Matrix get_x(unsigned int) const;
    Matrix get_z() const;
    Matrix get_z(unsigned int) const;
    Matrix get_weight() const;
    Matrix get_y() const;
    Matrix get_y(unsigned int) const;
    Matrix get_dy_dz() const;
    Matrix get_dy_dz(unsigned int) const;
    Matrix get_dz_db() const;
    Matrix get_dl_db() const;
    Matrix get_dl_db(unsigned int) const;
    Matrix get_dz_dx() const;
    Matrix get_dz_dx(unsigned int) const;
    Matrix get_dz_dw() const;
    Matrix get_dz_dw(unsigned int) const;
    Matrix get_dl_dx() const;
    Matrix get_dl_dw() const;

    // OPERATOR
    friend std::ostream& operator<<( std::ostream&,  Layer&);
};

#endif //NEURALNETWORK_LAYER_HH
