#include <utils.h>
#include <time.h>

extern inline double activate(double x)
{
	return x > 0 ? 1 : -1;
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

void learn(Matrix *w, Matrix *x)
/* w is a K by K weight matrix
 * x is a K by T binary column vector. T columns of training examples.
 * W is set to the outer product, xxT, averaging over the T training examples
 */
{
	size_t i ,j, t;
	size_t _K = x->nrows;
	size_t T = x->ncols;

	Matrix *xxT = mat_new(_K, _K, NULL);

	for (t=0; t<T; ++t)
	{
		Matrix *x_tcol = mat_col(x, t);
		Matrix *x_trow = mat_transpose(x_tcol);
		mat_dot(x_tcol, x_trow, xxT);
		mat_add(w, xxT, w);
		mat_free(x_trow);
		mat_free(x_tcol);
	}

	//set self-connections to 0
	for (i=0; i<_K; ++i) mat_set(w, i, i, 0);
	
	//scale by learning rate parameter 1/T
	scalar_mult(w, (double) (1.0/(double)T), w); 
	
	mat_free(xxT);
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

			if (abs(y_new - y_last) > 1e-2) //will it even converge?
			{
				mat_set(y, j, 0, y_new); //update network state
				stable=0; //repeat
			}
		}
	}
	mat_free(uj);

	return y;
}

Matrix* corrupt_column(Matrix *x, size_t n)
/* corrupt a column vector n times 
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

double hopfield(size_t num_input, size_t K, size_t numCorrupt, size_t niter)
{

	const size_t _K = K+1; //+1 bias term

	Matrix *w = mat_new(_K, _K, NULL);
	Matrix *x = mat_new(_K, num_input, NULL);
	random_binary(x); //random inputs to learn
	
	learn(w,x);

	double avg_acc = test_recall(w, x, numCorrupt);
	//printf("num. inputs: %lu, input length: %lu, num. corrupt: %lu, avg. acc: %.4f\n", num_input, K, numCorrupt, avg_acc);
	printf("%lu\t%lu\t%lu\t%lu\t%.4f\n", niter, num_input, K, numCorrupt, avg_acc);
	
	return avg_acc;
}

int main()
{
	srand(time(0)); //initialize random seed

	size_t nReps=50, niter;
	double avg_acc=0;
	
	//header
	printf("trial\tnum patterns\tpattern size\tnum. corrupt units\tavg. acc\n");
	
	for (niter=0; niter<nReps; ++niter)
	{
		avg_acc += hopfield(90, 100, 0, niter);
	}
	avg_acc /= (double) nReps;

	printf("# overall average accuracy: %.4f\n", avg_acc);

	return 0;

}
