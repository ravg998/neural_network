//
// Created by Rayan Ould Maamar on 14/01/2025.
//

#ifndef NEURALNETWORK_LOSS_HH
#define NEURALNETWORK_LOSS_HH
#include "../math/Matrix.h"
#include "LossType.hh"

class Loss {
protected:
    Matrix _y_hat;
    Matrix _y;
    Matrix _loss;
    Matrix _dl_dy;
    LossType _loss_type;
public:
    // CONSTRUCTOR
    Loss();
    Loss(const Matrix&);

    // COMPUTATIONS
    virtual Matrix compute_loss(const Matrix& )=0;
    virtual void compute_dl_dy()=0;

    // GET & SET
    void set_y(const Matrix&) ;
    Matrix get_dl_dy() const;
    Matrix get_loss() const;
    Matrix get_y_hat() const;
    LossType get_loss_type() const;

    // OPERATOR
    friend std::ostream& operator<<(std::ostream&, const Loss&);

};

class LossEntropy: public Loss{
private:
    LossType _loss_type = LossType::ENTROPY;
public:
    LossEntropy();
    LossEntropy(const Matrix& y);
    Matrix compute_loss(const Matrix&) override;
    void compute_dl_dy() override ;

};


class LossEuclidean: public Loss{
private:
    LossType _loss_type = LossType::EUCLIDEAN;

public:
    LossEuclidean();
    LossEuclidean(const Matrix&);
    Matrix compute_loss(const Matrix&) override;
    void compute_dl_dy() override;
};



#endif //NEURALNETWORK_LOSS_HH
