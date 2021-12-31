#include <hopfield.h>
#include <time.h>

int main()
{
	//srand(time(0)); //initialize random seed
	srand(42); //initialize random seed

	size_t numInput = 15;
	size_t K = 100;
	size_t numTrials = 100; //number of random binary input matrices

	//fine tune parameters
	double sparseness = 0.50; //0.50 gives best retrieval accuracy
	size_t numCorrupt = 0; //no. of input bits flipped
	size_t numLoss = 0; //total no. of weights is 101*101
	size_t numSteps = 1; //number of gradient descent steps
	double alpha=0; //prevent weights from becoming too large

	printf("# params:\n# num. input: %lu\n# K: %lu\n# sparseness: %.2f\n# num. corrupt: %lu\n# num. dropout: %lu\n# num. GD steps: %lu\n# alpha: %.2f\n# num. trials: %lu\n", numInput, K, sparseness, numCorrupt, numLoss, numSteps, alpha, numTrials);

	printf("trial\tnum patterns\tpattern size\tnum. corrupt units\tavg. acc\n");
	
	size_t ntrial=0;
	double avg_acc=0;
	for (; ntrial<numTrials; ++ntrial)
	{
		avg_acc += hopfield(numInput, K, sparseness, numCorrupt, numLoss, ntrial, numSteps, alpha);
	}
	avg_acc /= (double) numTrials;

	printf("# overall average accuracy: %.4f\n", avg_acc);

	return 0;
}
