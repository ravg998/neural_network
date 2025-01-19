//
// Created by Rayan Ould Maamar on 14/01/2025.
//

#include "Loss.hh"
Loss::Loss()
{

}
Loss::Loss(const Matrix& y_hat)
{
    _y_hat = y_hat;
};

Matrix Loss::compute_loss(const Matrix& output) {
    _y = output;
    _loss =  (_y  - _y_hat).square();
    return _loss;
}

void Loss::compute_dl_dy() {
    _dl_dy = 2*(_y - _y_hat).transpose();


}


// GET & SET
Matrix Loss::get_dl_dy() const
{
    return _dl_dy;
}
Matrix Loss::get_loss() const
{
    return _loss;
}

// OPERATOR
std::ostream& operator<<(std::ostream& c, const Loss& l)
{
    c << l._loss;
    return c;
}
