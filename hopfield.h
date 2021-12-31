#include <utils.h>
#include <math.h>

double sigmoid(double x);
void shuffle_a(size_t a[], size_t n);
void print_a(size_t a[], size_t n);
void random_binary(Matrix *m);
void mat_col_shuffle(Matrix *m, size_t j, size_t numShuffles);
void sparse_binary(Matrix *m, double sparsity);
void hebb(Matrix *w, Matrix *x);
void hebb_gd(Matrix *w, Matrix *x, size_t N, double alpha);
void storkey(Matrix *w, Matrix *x);
Matrix* recall(Matrix *w, Matrix *xc);
void random_loss(Matrix *m, size_t numLoss);
Matrix* corrupt_column(Matrix *x, size_t n);
double accuracy(Matrix *xt, Matrix *xr);
double test_recall(Matrix *w, Matrix *x, size_t numCorrupt);
double hopfield(size_t numInput, size_t K, double sparseness, size_t numCorrupt, size_t numLoss, size_t numTrials, size_t numSteps, double alpha);
