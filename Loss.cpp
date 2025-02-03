//
// Created by Rayan Ould Maamar on 14/01/2025.
//

#include "Loss.hh"
#include "LossType.hh"
Loss::Loss()
{

}
Loss::Loss(const Matrix& y)
{
    _y = y;
};



// GET & SET
void Loss::set_y(const Matrix & y) {
    _y = y;
}
Matrix Loss::get_dl_dy() const
{
    return _dl_dy;
}
Matrix Loss::get_loss() const
{
    return _loss;
}

LossType Loss::get_loss_type() const {
    return _loss_type;
}

Matrix Loss::get_y_hat() const
{
    return _y_hat;
}

// OPERATOR
std::ostream& operator<<(std::ostream& c, const Loss& l)
{
    c << l._loss;
    return c;
}

// LOSS ENTROPY
LossEntropy::LossEntropy(): Loss(){};
LossEntropy::LossEntropy(const Matrix &y): Loss(y) {};

Matrix LossEntropy::compute_loss(const Matrix & output) {
    _y_hat = output;
    _loss = _y.log().hadamard_product(_y_hat)*(-1);

    return _loss;
}

void LossEntropy::compute_dl_dy()
{
    _dl_dy = (_y_hat / _y).transpose();
}


// EUCLIDEAN
LossEuclidean::LossEuclidean(): Loss(){};
LossEuclidean::LossEuclidean(const Matrix& y): Loss(y){};
Matrix LossEuclidean::compute_loss(const Matrix & output) {
    _y_hat = output;
     _loss =  (_y_hat  - _y).square();
    return _loss;
}

void LossEuclidean::compute_dl_dy() {

   _dl_dy = -2*(_y - _y_hat).transpose() ;
}
