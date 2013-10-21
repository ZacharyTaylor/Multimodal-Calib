#ifndef MI_H
#define MI_H

#include "common.h"
#include "reduction.h"

//! Runs the mutal information calculation
/*!
	\param A the first signal, range 0 to 1
	\param B the second signal to be compared, range 0 to 1
	\param bins number of bins to use, due to issue with gpu shared memory must be less then 64
	\param numElements number of points in signal 1 and 2 (signals must be same size)
*/
float miRun(float* A, float* B, size_t bins, size_t numElements, bool normalize, cudaStream_t* stream);

#endif
