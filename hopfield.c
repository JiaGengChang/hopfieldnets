#include <hopfield.h>

void zero_diagonal(Matrix* w)
{
	size_t i=0;
	while (i++ < w->nrows) mat_set(w, i, i, 0);
}

extern inline double activate(double x)
{
	return x > 0 ? 1 : -1;
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

void shuffle_a(size_t a[], size_t n) //fisher-yates shuffle an array
{
	//n = size of a[]
	size_t i = n-1, j;
	for (; 0<i; --i)
	{
		j = (size_t) rand() % i;
		size_t tmp = a[i];
		a[i] = a[j];
		a[j] = tmp;
	}
}

void print_a(size_t a[], size_t n)
{
	size_t i;
	for (i=0; i<n; ++i) printf("%lu\t", a[i]);
	printf("\n");

}

void random_binary(Matrix *m)
{
	size_t i,j;
	for (i=0; i<m->nrows-1; ++i)
	{
		for (j=0; j<m->ncols; ++j)
		{
			mat_set(m, i, j, (rand() % 2)*2 - 1); //-1 or 1
		}
	}
	//K+1 th bias term is set to -1
	for (j=0; j<m->ncols; ++j) mat_set(m, m->nrows-1, j, -1);
}

void mat_col_shuffle(Matrix *m, size_t j, size_t numShuffles)
/*shuffle column j of a matrix */
/*last row is left out*/
{
	size_t n, r1, r2;
	double tmp;
	for (n=0; n<numShuffles; ++n)
	{
		r1 = (size_t) rand() % (m->nrows-1);
		r2 = (size_t) rand() % (m->nrows-1);
		if (r1 != r2)
		{
			tmp = mat_get(m, r2, j);
			mat_set(m, r2, j, mat_get(m, r1, j));
			mat_set(m, r1, j, tmp);
		}
	}
}

void sparse_binary(Matrix *m, double sparsity)
/*sparsity is the fraction of +1 neurons */
{
	size_t numPos = (size_t) ( sparsity * (m->nrows-1) );
	size_t i,j;
	for (i=0; i<m->nrows; ++i)
	{
		for (j=0; j<m->ncols; ++j)
		{
			mat_set(m, i, j, i<numPos ? 1 : -1);
		}
	}
	//perform NROW shuffles on each column
	for (j=0; j<m->ncols; ++j)
	{
		mat_col_shuffle(m, j, m->nrows); //I assume shuffling a length K vector K times is enough to make it random
	}
}

void hebb(Matrix *w, Matrix *x)
/* w is a K by K weight matrix
 * x is a K by T binary column vector. T columns of training examples.
 * W is set to the outer product, xxT, averaging over the T training examples
 */
{
    Matrix *xT = mat_transpose(x);
    mat_dot(x, xT, w);

	//set self-connections to 0
	zero_diagonal(w);
	
	//scale by learning rate parameter 1/T
	scalar_mult(w, (double) (1.0/(double)x->ncols), w);
	
}

void hebb_gd(Matrix *w, Matrix *x, size_t N, double alpha)
/* w is a K by K weight matrix
 * x is a K by T binary column vector. T columns of training examples.
 * W is set to the outer product, xxT, averaging over the T training examples
 * alpha is the 0-1 parameter for preventing spiralling weights
 * N is the number of gradient descent steps
 */
{
	size_t i ,j;
	size_t _K = x->nrows;
	size_t T = x->ncols;
	size_t n; //number of training iterations

	hebb(w, x); //initialize weights using hebb rule

	//t is identical to x except that -1s are replaced by 0s
	//except bias row
	Matrix *t = mat_new(_K, T, x->data);
	for (i=0; i<_K-1; ++i)
		for (j=0; j<T; ++j)
			mat_set(t, i, j, mat_get(x, i, j) < 0 ? 0 : 1 );

	//start of improved version of hebb rule
	Matrix *xT = mat_transpose(x); //T by K
	Matrix *a = mat_new(_K, _K, NULL); //activation matrix
	Matrix *y = mat_new(_K, _K, NULL); //output matrix
	Matrix *e = mat_new(_K, _K, NULL); //error matrix
	Matrix *dw = mat_new(_K, _K, NULL); //delta matrix
	Matrix *gw = mat_new(_K, _K, NULL); //gradient matrix

	for (n=0; n<N; ++n)
	{
		//set self-weights to 0
		zero_diagonal(w);

		//compute all activations 
		mat_dot(x, w, a);

		//compute all outputs
		mat_apply(a, sigmoid, y);

		//compute all errors
		mat_sub(y, t, e);

		//compute the gradients
		mat_dot(xT, e, gw);

		//symmetrize gradients
		Matrix *gwT = mat_transpose(gw);
		mat_add(gw, gwT, gw);
		mat_free(gwT);
		
		//make step
		Matrix *aw = mat_new(_K, _K, NULL);
		scalar_mult(w, alpha, aw);
		mat_sub(dw, aw, dw);
		scalar_mult(dw, (double) (1.0/(double)T), dw); //multiply by lr=1/T

		mat_add(w, dw, w);

		//set bias weights to 0
		for (i=0; i<_K; ++i) mat_set(w, w->nrows-1, i, 0);

	}

	mat_free(xT);
	mat_free(a);
	mat_free(e);
	mat_free(dw);
	mat_free(gw);
	mat_free(t);
}

void storkey(Matrix *w, Matrix *x)
{
	size_t _K = x->nrows;
	size_t T = x->ncols;

	//one step hebb rule
    hebb(w,x);

	//h_ik = w_ij x_jk
	Matrix *h = mat_new(_K, T, NULL);
	mat_dot(w, x, h);

	//r_ij = s_ij - h_ij
	Matrix *r = mat_new(_K, T, NULL);
	mat_sub(x, h, r); 

	//dw_ij = r_ik * r_kj
	Matrix *dw = mat_new(_K, _K, NULL);
	Matrix *rT = mat_transpose(r);
	mat_dot(r, rT, dw);

	//perform storkey update on w
	mat_add(w, dw, w);

	//set self-connections to 0
	zero_diagonal(w);
	
	mat_free(rT); mat_free(r); mat_free(h); mat_free(dw);

}

Matrix* recall(Matrix *w, Matrix *xc)
{
	/* test the ability of hopfield net w to recall x given
	 * cx, a corrupted version of x
	 * NB. the x here is a _K by 1 column vector
	 */
	size_t j,k;
	size_t _K = xc->nrows;
	size_t K = xc->nrows-1; //dont count bias

	//initialize network state y
	Matrix *y = mat_new(_K, 1, NULL);
	mat_assign(y, xc);  //y:=xc

	int stable=0; //repeat until network is stable, aka y=x
	Matrix *uj = mat_new(1,1,NULL);

	//initialize a := {1...K}
	size_t a[K];
	for (j=0;j<K;++j) a[j]=j;

	while (stable==0)
	{
		stable=1;
		shuffle_a(a, K); 
		for (k=0; k<K; ++k)
		{
			j = a[k]; //random idx from 1 to K
			Matrix *w_col = mat_col(w, j);
			Matrix *yt = mat_transpose(y);
			mat_dot(yt, w_col, uj); //1 by 1 matrix. weighted sum of connections
			double y_new = activate(mat_get(uj, 0, 0)); //perceptron threshold function
			double y_last = mat_get(y, j, 0);
			mat_free(w_col);
			mat_free(yt);

			if (y_new - y_last > 1e-2 || y_last - y_new > 1e-2) //will it even converge?
			{
				mat_set(y, j, 0, y_new); //update network state
				stable=0; //repeat
			}
		}
	}
	mat_free(uj);

	return y;
}

/*simulate random loss of weights*/
void random_loss(Matrix *m, size_t numLoss)
{
	if (numLoss > 0)
	{
		size_t i,j,k,K = m->nrows, K2 = K*K;

		size_t idx[K2]; //linearized 2d idx
		for (i=0; i<numLoss; ++i) {idx[i] = i;}
		shuffle_a(idx, K2); 

		//select random coordinates in w to mutate
		//TODO: diagonals are already 0
		for (k=0; k<numLoss; ++k)
		{
			i = k % K;
			j = k / K;
			mat_set(m, i, j, 0);
		}
		
	}
}

Matrix* corrupt_column(Matrix *x, size_t n)
/* corrupt a column vector n times by flipping the sign
 * but don't change the bias
 * */
{
	Matrix *xc = mat_new(x->nrows, x->ncols, NULL);
	mat_assign(xc, x);

	size_t i;
	while (n-- > 0)
	{
		i = rand() % (x->nrows -1);
		mat_set(xc, i, 0, -mat_get(x, i, 0));
	}
	return xc;
}

double accuracy(Matrix *xt, Matrix *xr)
{
	size_t j;
	double ret = 0;
	for (j=0; j<xt->nrows-1; ++j) //bias not counted
	{
		if (mat_get(xt, j, 0) == mat_get(xr, j, 0))
		{
			++ret;
		}
	}
	ret /= (xt->nrows-1); //bias not counted
	return ret;
}

//test ability to recall all T patterns
double test_recall(Matrix *w, Matrix *x, size_t numCorrupt)
{
	size_t T = x->ncols;
	size_t t;
	double acc = 0;
	for (t=0; t<T; ++t)
	{	
		Matrix *xt = mat_col(x, t);
		Matrix *xc = corrupt_column(xt, numCorrupt); //corrupt 1 input
		Matrix *xr = recall(w, xc);

		double a = accuracy(xt, xr);
		acc += a;

		Matrix *xtt, *xct, *xrt;
		xtt = mat_transpose(xt);
		xct = mat_transpose(xc);
		xrt = mat_transpose(xr);

		// printf("Original\t"); print_mat(xtt);
		// printf("Corrupted\t"); print_mat(xct);
		// printf("Recall\t"); print_mat(xrt);
		// printf("Accuracy %.2f\n", a);
		// printf("=================\n");

		mat_free(xt);
		mat_free(xc);
		mat_free(xr);
		mat_free(xtt);
		mat_free(xct);
		mat_free(xrt);
	}
	acc /= T; //average accuracy
	return acc;
}

double hopfield(size_t numInput, size_t K, double sparseness, size_t numCorrupt, size_t numLoss, size_t numTrials, size_t numSteps, double alpha)
{

	const size_t _K = K+1; //+1 bias term

	Matrix *w = mat_new(_K, _K, NULL);
	Matrix *x = mat_new(_K, numInput, NULL);

	sparse_binary(x, sparseness); //random inputs to learn

	Matrix *xt = mat_transpose(x);
	
	storkey(w, x);
	//hebb_gd(w, x, numSteps, dup_alpha);
	
	random_loss(w, numLoss); //random loss of weights 

	double avg_acc = test_recall(w, x, numCorrupt); //corrupt random bits

	printf("%lu\t%lu\t%lu\t%lu\t%.4f\n", numTrials, numInput, K, numCorrupt, avg_acc);

	mat_free(w);
	mat_free(x);
	mat_free(xt);
	
	return avg_acc;
}
