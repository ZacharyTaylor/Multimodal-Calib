/*Small program to quickly get mi
/*Small program to quickly get mi
 * call using mi(A, B, normal, bins)
 *  A       input image, must be of type uint8
 *  B       second input image, must be of uint8 and same size as A
 *  normal  0 for standrad mi, 1 for NMI
 *  bins    number of bins to use (Note must be greater then the largest 
 *          value in A and B, there is no error checking)
 */ 

#include "mi.h"

/**
*	HistKernel-
*	Calculates the 2d histogram for the images assuming less then 64 bins (max bins possible with 16kb shared memory)
*
*	Inputs-
*	a - the first image (all values 0-1)
*   b - second image of same size (all values 0-1)
*	histAB - a n by n 2d histogram to be output with preallocated memory
*/
__global__ void HistKernel(float* a, float* b, unsigned int* histABI, size_t bins, size_t numElements){
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	extern __shared__ unsigned int blockMem[];



	//zero out shared mem
	for (size_t j = threadIdx.x; j < bins*bins; j += blockDim.x){ 
		blockMem[j] = 0;
	}
	__syncthreads();

	//for all elements
	for(size_t j = i; j < numElements; j += blockDim.x*gridDim.x){

		//get bin to put point in
		size_t locA = ((float)(bins-1))*a[j];
		size_t locB = ((float)(bins-1))*b[j];

		//atomicMax(&blockMem[(j%(bins*bins))],locB);
		atomicAdd(&blockMem[locA + bins*locB], 1);
	}
	
	__syncthreads();

	//write back to global memory
	for (size_t j = threadIdx.x; j < bins*bins; j += blockDim.x){ 

		if(blockMem[j] != 0){
			atomicAdd(&histABI[j], blockMem[j]);
		}
	}
}

/**
*	splitHistKernel-
*	performs 2 functions 1) removes black from histogram
*	2) splits joint histogram into 2 1d histograms for a and b
*
*	Inputs-
*	histA - histogram of input A to be calculated. Memory must be preallocated.
*   histB - histogram of input B to be calculated. Memory must be preallocated.
*	histAB - a 63x63 2d histogram that has been precalculated
*	histSize - give sum of all values in histAB and it will return new total after black elements removed
*/
__global__ void splitHistKernel(unsigned int* histABI, unsigned int* histAI, unsigned int* histBI, unsigned int numElements, unsigned int* histSize, size_t bins){
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numElements){
		return;
	}

	unsigned int locB = i / bins;
	unsigned int locA = i % bins;

	//ignore zeros
	if((locA == 0) || (locB == 0)){
		atomicSub(histSize, histABI[i]);
		histABI[i] = 0;
	}
	else{
		//split
		atomicAdd(&histAI[locA], histABI[i]);
		atomicAdd(&histBI[locB], histABI[i]);
	}
}

/**
*	entKernel-
*	Calculates p(x)*log2(p(x)) for each element from a histogram (note this is where transition from using ints to floats is made)
*
*	Inputs-
*	histIn - the input histogram of integers
*   histSize - the sum of all the elements in histIn
*	histOut - the output histogram of entropy floats
*/
__global__ void entKernel(const unsigned int* histIn, const unsigned int numElements, const unsigned int* histSize, float* histOut){
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numElements){
		return;
	}

	//explained, total, dist
	float val;
	if(histIn[i] != 0){
		val = ((float)(histIn[i]))/((float)(histSize[0]));
		val = -(val*log2(val));
	}
	else{
		val = 0;
	}

	histOut[i] = val;
}

/**
*	miRun-
*	runs the mutual information calculation
*
*	Inputs-
*	miData - struct which contains all histograms and other information needed in the calculation
*	imData - struct containing all the information on the images that mi will be performed on
*	imNum - which image in imData will have the mi calculated
*	Output - the normalized mutual information
*/
float miRun(float* A, float* B, size_t bins, size_t numElements, cudaStream_t* stream){

	//create histograms
	unsigned int* histAI;
	CudaSafeCall(cudaMalloc((void**)&histAI, sizeof(unsigned int)*bins));
	CudaSafeCall(cudaMemset(histAI, 0, sizeof(unsigned int)*bins));

	unsigned int* histBI;
	CudaSafeCall(cudaMalloc((void**)&histBI, sizeof(unsigned int)*bins));
	CudaSafeCall(cudaMemset(histBI, 0, sizeof(unsigned int)*bins));

	float* histAF;
	CudaSafeCall(cudaMalloc((void**)&histAF, sizeof(float)*bins));

	float* histBF;
	CudaSafeCall(cudaMalloc((void**)&histBF, sizeof(float)*bins));

	unsigned int* histABI;
	CudaSafeCall(cudaMalloc((void**)&histABI, sizeof(unsigned int)*bins*bins));
	CudaSafeCall(cudaMemset(histABI, 0, sizeof(unsigned int)*bins*bins));

	float* histABF;
	CudaSafeCall(cudaMalloc((void**)&histABF, sizeof(float)*bins*bins));

	//fill main histogram
	HistKernel<<<gridSize(1024), BLOCK_SIZE, bins*bins*sizeof(unsigned int), *stream>>>
		(A, B, histABI, bins, numElements);
	CudaCheckError();

	//split histogram
	unsigned int* histSize;
	CudaSafeCall(cudaMalloc((void**)&histSize, sizeof(unsigned int)));
	CudaSafeCall(cudaMemcpy(histSize, &numElements, sizeof(unsigned int),cudaMemcpyHostToDevice));
	splitHistKernel<<<gridSize(bins*bins), BLOCK_SIZE, 0, *stream>>>
		(histABI, histAI, histBI, (bins*bins), histSize, bins);
	CudaCheckError();

	//get entropy
	entKernel<<<gridSize(bins*bins), BLOCK_SIZE, 0, *stream>>>(histABI, (bins*bins), histSize, histABF);
	CudaCheckError();
	entKernel<<<gridSize(bins), BLOCK_SIZE, 0, *stream>>>(histAI, bins, histSize, histAF);
	CudaCheckError();
	entKernel<<<gridSize(bins), BLOCK_SIZE, 0, *stream>>>(histBI, bins, histSize, histBF);
	CudaCheckError();

	//reduce
	float eAB = reduceEasy(histABF, bins*bins);
	float eA = reduceEasy(histAF, bins);
	float eB = reduceEasy(histBF, bins);

	//finally get mi
	float mi = (eA + eB) / eAB;

	CudaSafeCall(cudaFree(histAI));
	CudaSafeCall(cudaFree(histBI));
	CudaSafeCall(cudaFree(histABI));
	CudaSafeCall(cudaFree(histAF));
	CudaSafeCall(cudaFree(histBF));
	CudaSafeCall(cudaFree(histABF));

	return mi;
}