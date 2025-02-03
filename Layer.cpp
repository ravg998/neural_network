#include "Layer.hh"
#include "Activation.hh"

// CONSTRUCTOR
Layer::Layer()
{

}

Layer::Layer(const Matrix& input, unsigned int n_output,  Activation&  activation_function) {
    _activation_function =  &activation_function;
    _n_output = n_output;
    _x = input;
    _n_observation = input.get_n_col();

    // INTERMEDIATES
    this->define_weight();
    this->define_bias();
}


// METHOD
/// INITIALIZATION

void Layer::define_weight() {
    _weight = Matrix(MatrixType::GAUSSIAN, _n_output, _x.get_n_row()); // USE NORMAL DISTRIBUTION TO INITIALIZE WEIGHTS
}

void Layer::define_bias() {
    /*
     * Initialize Bias term with a Gaussian.
     * It will be a (_n_output x _n_observation) matrix. Each column is duplicated so all _n_observation have the same bias
     */
    _bias = Matrix(MatrixType::GAUSSIAN, 1, _n_output); // USE NORMAL DISTRIBUTION TO INITIALIZE WEIGHTS
    for (unsigned int j = 0; j<_n_observation-1; j++)
    {
        _bias.append(_bias[0]);
    }
    _bias = _bias.transpose();

}

void Layer::reset_dy_dz() {
    _dy_dz = Matrix(MatrixType::MATRIX_NULL, _n_output, _n_output);
}

void Layer::reset_dl_dw() {
    _dl_dw = Matrix(MatrixType::MATRIX_NULL, _weight.get_n_row(), _weight.get_n_col());
}

void Layer::reset_dl_dx() {
    _dl_dx = Matrix(MatrixType::MATRIX_NULL, 0,_x.get_n_row());
}

void Layer::reset_dl_db() {
    _dl_db = Matrix(MatrixType::MATRIX_NULL, 1,_n_output);
}

void Layer::reset_derivatives() {
    this->reset_dy_dz();
    this->reset_dl_dw();
    this->reset_dl_dx();
    this->reset_dl_db();

}

/// FORWARD
void Layer::update_input(const Matrix & input) {
    _x.CheckDimensionMatrixAddition(input); // IF THE TWO MATRICES DON'T HAVE SAME DIM, THEN GET AN ERROR
    _x = input;
}
void Layer::compute_z() {
    _z = _weight * _x + _bias;

}
void Layer::compute_y() {
    _y_hat = _activation_function->forward(_z);
}


void Layer::forward() {
    this->compute_z();
    this->compute_y();
}

void Layer::forward(const Matrix & input) {
    this->update_input(input);
    this->forward();
}

/// BACKWARD
void Layer::compute_dy_dz(unsigned int i)
{
    _dy_dz = _dy_dz + _activation_function->backward(this->get_z(i));
}

void Layer::compute_dz_dx() {

    _dz_dx = this->get_weight();
}

void Layer::compute_dz_db() {
    _dz_db = Matrix(MatrixType::ID, this->get_z().get_n_row(),this->get_z().get_n_row());
}

void Layer::compute_dz_dw(unsigned int i) {

    _dz_dw =this->get_x().transpose();
}

void Layer::compute_dl_dx(const Matrix& dl_dy)
{
    _dl_dx.append((dl_dy * this->get_dy_dz()*this->get_dz_dx())[0]);
}

void Layer::compute_dl_dw(const Matrix& dl_dy, unsigned int i)
{
    _dl_dw = _dl_dw + (dl_dy * this->get_dy_dz()).transpose()*this->get_dz_dw(i);
}

void Layer::compute_dl_db(const Matrix& dl_dy)
{
    _dl_db = _dl_db + dl_dy * this->get_dy_dz();
}

void Layer::backward(const Matrix & dl_dy) {

    this -> reset_derivatives();

    this->compute_dz_dx();
    this->compute_dz_db();
    this->compute_dz_dw();

    for (unsigned int i = 0; i<_n_observation; i++)
    {
        Matrix m(dl_dy[i]);
        this->compute_dy_dz(i);
        this->compute_dl_dw(m,i);
        this->compute_dl_dx(m);
        this->compute_dl_db(m);
    }

    _dl_dw = _dl_dw /_n_observation;
    _dl_db = _dl_db /_n_observation;

}

void Layer::update_weight() {
    _weight = _weight - 1e-4*this->get_dl_dw();
}

void Layer::update_bias() {
    Matrix one(MatrixType::ATTILA, 1, _n_observation);
   _bias = _bias - 1e-4*(this->get_dl_db()).transpose()*one;
}




// GET & SET
unsigned int Layer::get_n_observation() const
{
    return _n_observation;
}
Matrix Layer::get_x() const
{
    return _x;
}

Matrix Layer::get_x(unsigned int i) const
{
    return Matrix(_x.extract_col(i)).transpose();
}

Matrix Layer::get_z() const
{
    return _z;
}

Matrix Layer::get_z(unsigned int i) const
{
    return Matrix(_z.extract_col(i)).transpose();

}

Matrix Layer::get_weight() const
{
    return _weight;
}

Matrix Layer::get_y() const
{
    return _y_hat;
}

Matrix Layer::get_y(unsigned int i) const
{
    return Matrix(_y_hat.extract_col(i)).transpose();

}

Matrix Layer::get_dy_dz()  const
{
    return _dy_dz;
}

Matrix Layer::get_dy_dz(unsigned int i)  const
{
    return Matrix(_dy_dz.extract_col(i)).transpose();
}

Matrix Layer::get_dl_db()  const
{
    return _dl_db;
}

Matrix Layer::get_dl_db(unsigned int i)  const
{
    return Matrix(_dl_db.extract_col(i)).transpose();
}


Matrix Layer::get_dz_db() const
{
    return _dz_db;
}

Matrix Layer::get_dz_dw() const
{
    return _dz_dw;
}

Matrix Layer::get_dz_dw(unsigned int i) const
{
    return Matrix(_dz_dw[i]);
}


Matrix Layer::get_dz_dx() const
{
    return _dz_dx;
}

Matrix Layer::get_dz_dx(unsigned int i) const
{
    return Matrix(_dz_dx.extract_col(i)).transpose();
}

Matrix Layer::get_dl_dx() const
{
    return _dl_dx;
}
Matrix Layer::get_dl_dw() const
{
    return _dl_dw;
}
// OPERATOR
std::ostream& operator<<( std::ostream& c,  Layer& l)
{
    c << "X: "<<std::endl;
    c << l.get_x() << std::endl;
    c << "Z: "<<std::endl;
    c << l.get_z() << std::endl;
    c << "Weight: "<<std::endl;
    c << l.get_weight() << std::endl;
    c << "Y: "<<std::endl;
    c << l.get_y() << std::endl;
    c << "dY/dZ: "<<std::endl;
    c << l.get_dy_dz() << std::endl;
    c << "dZ/dX: "<<std::endl;
    c << l.get_dz_dx() << std::endl;
    c << "dZ/dW: "<<std::endl;
    c << l.get_dz_dw() << std::endl;
    return c;
}