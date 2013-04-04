#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MATLAB
#define CUDA_ERROR_CHECK

#ifdef MATLAB
	#include <mex.h>
	#define printf mexPrintf
#endif

#define BLOCK_SIZE 512

#define CudaSafeCall( err ) __cudaSafeCall( err, FILE, __LINE__ )
#define CudaCheckError()    __cudaCheckError( FILE, __LINE__ )

#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

__inline size_t gridSize(size_t num){
	return (size_t)ceil(((float)(num))/((float)(BLOCK_SIZE)));
};

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
	#ifdef CUDA_ERROR_CHECK
		if ( cudaSuccess != err )
		{
			printf("Cuda error at %s:%i : %s\n",
			file, line, cudaGetErrorString( err ) );
			cudaDeviceReset();
		}
	#endif
 
	return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err ){
		printf("Kernel error: %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
	}
 
	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err ){
		printf("Kernel error with sync failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
	}
	cudaDeviceReset();
#endif
 
return;
}

#endif //COMMON_H