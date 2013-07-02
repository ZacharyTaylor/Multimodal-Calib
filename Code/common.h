#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

// includes, graphics
#if defined (__APPLE__) || defined(MACOSX)
	#include <OpenGL/gl.h>
	#include <OpenGL/glu.h>
#else

	#ifdef _WIN32
		#  define WINDOWS_LEAN_AND_MEAN
		#  define NOMINMAX
		#  include <windows.h>
	#endif

	#include <GL/gl.h>
	#include <GL/glu.h>
#endif

#define MATLAB
#define CUDA_ERROR_CHECK

#ifdef MATLAB
	#include <mex.h>
	#define printf mexPrintf
#endif

#define BLOCK_SIZE 512

#define CudaSafeCall( err ) __cudaSafeCall( err, FILE, __LINE__ )
#define CudaCheckError()    __cudaCheckError( FILE, __LINE__ )
#define SdkCheckErrorGL()	__sdkCheckErrorGL( FILE, __LINE__)

#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

__inline size_t gridSize(size_t num){
	return (size_t)ceil(((float)(num))/((float)(BLOCK_SIZE)));
};

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
	#ifdef CUDA_ERROR_CHECK
		if ( cudaSuccess != err )
		{
			printf("CUDA Function Error at %s:%i : %s\n",
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
		printf("CUDA Kernel Error at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
		cudaDeviceReset();
	}
 
	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err ){
		printf("CUDA Kernel Error with sync failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
		cudaDeviceReset();
	}
#endif
 
return;
}

inline void __sdkCheckErrorGL(const char *file, const int line){

    // check for error
    GLenum gl_error = glGetError();

    if (gl_error != GL_NO_ERROR)
    {

		printf("GL Error at %s:%i : %s\n", file, line, gluErrorString(gl_error));
    }
}

#endif //COMMON_H