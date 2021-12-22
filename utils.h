#ifndef utils_h
#define utils_h

#include <math.h>
#include <stdio.h>
#include <stdlib.h>


typedef struct {
	size_t nrows;
	size_t ncols;
	double data[];
} Matrix;


Matrix* mat_row(Matrix *Ref, const size_t row);
Matrix* mat_col(Matrix *Ref, const size_t col);
void mat_assign(Matrix *A, Matrix *B);
void mat_apply(Matrix *A, double (*function)(double), Matrix *B);
Matrix* mat_new(const size_t n1, const size_t n2, double input[]);
void randomize(Matrix *m, double min, double max);
void mat_shuffle(Matrix *m, size_t num_shuffles);
double mat_get(const Matrix *m, const size_t i, const size_t j);
void mat_set(Matrix *m, const size_t i, const size_t j, double v);
void mat_dot(Matrix *A, Matrix *B, Matrix *C);
void mat_add(Matrix *A, Matrix *B, Matrix *C);
void mat_sub(Matrix *A, Matrix *B, Matrix *C);
void mat_mult(Matrix *A, Matrix *B, Matrix *C);
Matrix* mat_transpose(Matrix *m);
void mat_free(Matrix *m);
void scalar_add(Matrix *A, const double s, Matrix *B);
void scalar_mult(const Matrix *A, const double s, Matrix *B);
void print_mat(const Matrix *m);

#endif
