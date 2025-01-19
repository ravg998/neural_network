//
// Created by Rayan Ould Maamar on 14/01/2025.
//

#ifndef NEURALNETWORK_LOSS_HH
#define NEURALNETWORK_LOSS_HH
#include "../math/Matrix.h"

class Loss {
private:
    Matrix _y_hat;
    Matrix _y;
    Matrix _loss;
    Matrix _dl_dy;
public:
    // CONSTRUCTOR
    Loss();
    Loss(const Matrix&);

    // COMPUTATIONS
    Matrix compute_loss(const Matrix&);
    void compute_dl_dy();

    // GET & SET
    Matrix get_dl_dy() const;
    Matrix get_loss() const;

    // OPERATOR
    friend std::ostream& operator<<(std::ostream&, const Loss&);

};


#endif //NEURALNETWORK_LOSS_HH
