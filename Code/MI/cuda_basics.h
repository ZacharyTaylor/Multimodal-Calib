/*
 Copyright Ramtin Shams (hereafter referred to as 'the author'). All rights 
 reserved. **Citation required in derived works or publications** 
 
 NOTICE TO USER:   
 
 Users and possessors of this source code are hereby granted a nonexclusive, 
 royalty-free license to use this source code for non-commercial purposes only, 
 as long as the author is appropriately acknowledged by inclusion of this 
 notice in derived works and citation of appropriate publication(s) listed 
 at the end of this notice in any derived works or publications that use 
 or have benefited from this source code in its entirety or in part.
   
 
 THE AUTHOR MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 IMPLIED WARRANTY OF ANY KIND.  THE AUTHOR DISCLAIMS ALL WARRANTIES WITH 
 REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 OR PERFORMANCE OF THIS SOURCE CODE.  
 
 Relevant publication(s):
	@inproceedings{Shams_ICSPCS_2007,
		author        = "R. Shams and R. A. Kennedy",
		title         = "Efficient Histogram Algorithms for {NVIDIA} {CUDA} Compatible Devices",
		booktitle     = "Proc. Int. Conf. on Signal Processing and Communications Systems ({ICSPCS})",
		address       = "Gold Coast, Australia",
		month         = dec,
		year          = "2007",
		pages         = "418-422",
	}

	@inproceedings{Shams_DICTA_2007a,
		author        = "R. Shams and N. Barnes",
		title         = "Speeding up Mutual Information Computation Using {NVIDIA} {CUDA} Hardware",
		booktitle     = "Proc. Digital Image Computing: Techniques and Applications ({DICTA})",
		address       = "Adelaide, Australia",
		month         = dec,
		year          = "2007",
		pages         = "555-560",
		doi           = "10.1109/DICTA.2007.4426846",
	};
*/

#ifndef CUDA_BASICS_H_
#define CUDA_BASICS_H_

#include <driver_types.h>
// Turn on the VERBOSE flag in order to compile with timing information.
//#define VERBOSE

// Turn on MATLAB flag if the target is a MATLAB mex file.
// Instead of including the flag directly here add that to project options.
//#define MATLAB

#ifdef MATLAB
#include <mex.h>
// CUT_ macros use printf to output info, overwirte that with mexPrintf to see the output in Matlab.
#define printf mexPrintf
// CUT_ macros use fprintf to output error info, overwirte that with mexfPrintf to see the output in Matlab.
#define fprintf mexfPrintf
// CUT_ macros use exit on serious erros, overwrite with mexErrMsgTxt that exits the mex function gracefully.
#define exit(arg) mexErrMsgTxt("The mex program has terminated.")
extern "C" int mexfPrintf(FILE * _File, const char * fmt, ...);
#endif

struct cudaPoint3D
{
	int x, y, z;
};

struct cudaPoint2D
{
	int x, y;
};

#ifdef VERBOSE
#define TIMER_CREATE																\
	unsigned int _timer_hTimer;														\
    CUT_SAFE_CALL(cutCreateTimer(&_timer_hTimer));									\

#else
#define TIMER_CREATE
#endif

#ifdef VERBOSE
#define TIMER_DELETE																\
    CUT_SAFE_CALL(cutDeleteTimer(_timer_hTimer));									\

#else
#define TIMER_DELETE
#endif

#ifdef VERBOSE
#define TIMER_START																	\
    CUDA_SAFE_CALL(cudaThreadSynchronize());										\
    CUT_SAFE_CALL(cutResetTimer(_timer_hTimer));									\
    CUT_SAFE_CALL(cutStartTimer(_timer_hTimer));									\

#else
#define TIMER_START
#endif

#ifdef VERBOSE
#define TIMER_PRINT(message, length)												\
	do																				\
	{																				\
		CUDA_SAFE_CALL(cudaThreadSynchronize());									\
		CUT_SAFE_CALL(cutStopTimer(_timer_hTimer));									\
		double gpuTime = cutGetTimerValue(_timer_hTimer);							\
		if (length > 0)																\
			printf("%s: %f msec, %f Mpixels/sec\n",									\
				message, gpuTime, 1e-6 * length / (gpuTime * 0.001));				\
		else																		\
			printf("%s: %f msec\n",													\
				message, gpuTime);													\
	} while(0);																		\

#else
#define TIMER_PRINT(message, length)
#endif

#define CUDA_REDUCTION(fnSuffix)																\
extern "C" float cuda##fnSuffix(float *src, int length, bool device)							\
{																								\
	dim3 grid, block;																			\
	int size = length * sizeof(float);															\
	float *d_src, *d_dst1, *d_dst2, *d_dst;														\
	float res;																					\
	TIMER_CREATE;																				\
																								\
	const int max_threads = MAX_THREADS;														\
	int good_len = iRoundUp(length, WARP_SIZE);													\
																								\
	block.x = max_threads; block.y = 1; block.z = 1;											\
	grid.x = ceil(sqrtf(iDivUp(good_len, 2 * max_threads))); grid.y = grid.x; grid.z = 1;		\
																								\
	TIMER_START;																				\
	if (!device)																				\
	{																							\
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));										\
																								\
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));					\
	}																							\
	else																						\
		d_src = src;																			\
																								\
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_dst1, grid.x * grid.y * sizeof(float)));				\
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_dst2, grid.x * grid.y * sizeof(float)));				\
	TIMER_PRINT("Loading data", 0);																\
																								\
    TIMER_START;																				\
	float *d_tmp = d_src;																		\
	int count = 0;																				\
	int len = length;																			\
	do																							\
	{																							\
		d_dst = count % 2 ? d_dst = d_dst2 : d_dst = d_dst1;									\
																								\
		gpu##fnSuffix<<<grid, block>>>(d_tmp, d_dst, len);										\
		CUT_CHECK_ERROR("gpu" #fnSuffix "() execution failed\n");								\
																								\
		d_tmp = d_dst;																			\
		count++;																				\
		len = grid.x * grid.y;																	\
		good_len = iRoundUp(len, WARP_SIZE);													\
		grid.x = ceil(sqrtf(iDivUp(good_len, 2 * max_threads))); grid.y = grid.x; grid.z = 1;	\
	}while (len != 1);																			\
    TIMER_PRINT("gpu" #fnSuffix, length);														\
																								\
	TIMER_START;																				\
	if (!device)																				\
		CUDA_SAFE_CALL(cudaFree(d_src));														\
	CUDA_SAFE_CALL(cudaMemcpy(&res, d_dst, sizeof(float), cudaMemcpyDeviceToHost));				\
	CUDA_SAFE_CALL(cudaFree(d_dst1));															\
	CUDA_SAFE_CALL(cudaFree(d_dst2));															\
	TIMER_PRINT("Storing data", 0);																\
																								\
	TIMER_DELETE;																				\
	return res;																					\
}																								\

/*
	dst must be allocated by the caller
*/
#define CUDA_BINARY(fnSuffix)																	\
extern "C" void cuda##fnSuffix(float *src1, float *src2, float *dst, int length,				\
	bool device)																				\
{																								\
	dim3 grid, block;																			\
	float *d_src1, *d_src2, *d_dst;																\
	int size = length * sizeof(float);															\
	TIMER_CREATE;																				\
																								\
	TIMER_START;																				\
	if (!device)																				\
	{																							\
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src1, size));										\
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src2, size));										\
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_dst, size));										\
																								\
		CUDA_SAFE_CALL(cudaMemcpy(d_src1, src1, size, cudaMemcpyHostToDevice));					\
		CUDA_SAFE_CALL(cudaMemcpy(d_src2, src2, size, cudaMemcpyHostToDevice));					\
	}																							\
	else																						\
	{																							\
		d_src1 = src1; d_src2 = src2; d_dst = dst;												\
	}																							\
	TIMER_PRINT("Loading data", 0);																\
																								\
	const int max_threads = MAX_THREADS;														\
	int good_len = iRoundUp(length, WARP_SIZE);													\
																								\
	block.x = max_threads; block.y = 1; block.z = 1;											\
	grid.x = ceil(sqrtf(iDivUp(good_len, max_threads))); grid.y = grid.x; grid.z = 1;			\
																								\
    TIMER_START;																				\
	gpu##fnSuffix<<<grid, block>>>(d_src1, d_src2, d_dst, length);								\
	CUT_CHECK_ERROR("gpu" #fnSuffix "() execution failed\n");									\
    TIMER_PRINT("gpu" #fnSuffix, length);														\
																								\
	if (!device)																				\
	{																							\
		TIMER_START;																			\
		CUDA_SAFE_CALL(cudaMemcpy(dst, d_dst, size, cudaMemcpyHostToDevice));					\
		CUDA_SAFE_CALL(cudaFree(d_src1));														\
		CUDA_SAFE_CALL(cudaFree(d_src2));														\
		CUDA_SAFE_CALL(cudaFree(d_dst));														\
		TIMER_PRINT("Storing data", 0);															\
	}																							\
	TIMER_DELETE;																				\
}																								\

/*
	dst must be allocated by the caller
*/
#define CUDA_UNARY(fnSuffix)																	\
extern "C" void cuda##fnSuffix(float *src, float *dst, int length, bool device)					\
{																								\
	dim3 grid, block;																			\
	float *d_src, *d_dst;																		\
	int size = length * sizeof(float);															\
	TIMER_CREATE;																				\
																								\
	TIMER_START;																				\
	if (!device)																				\
	{																							\
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));										\
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_dst, size));										\
																								\
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));					\
	}																							\
	else																						\
	{																							\
		d_src = src; d_dst = dst;																\
	}																							\
	TIMER_PRINT("Loading data", 0);																\
																								\
	const int max_threads = MAX_THREADS;														\
	int good_len = iRoundUp(length, WARP_SIZE);													\
																								\
	block.x = max_threads; block.y = 1; block.z = 1;											\
	grid.x = ceil(sqrtf(iDivUp(good_len, max_threads))); grid.y = grid.x; grid.z = 1;			\
																								\
    TIMER_START;																				\
	gpu##fnSuffix<<<grid, block>>>(d_src, d_dst, length);										\
	CUT_CHECK_ERROR("gpu" #fnSuffix "() execution failed\n");									\
	TIMER_PRINT("gpu" #fnSuffix, length);														\
																								\
	if (!device)																				\
	{																							\
		TIMER_START;																			\
		CUDA_SAFE_CALL(cudaMemcpy(dst, d_dst, size, cudaMemcpyHostToDevice));					\
		CUDA_SAFE_CALL(cudaFree(d_src));														\
		CUDA_SAFE_CALL(cudaFree(d_dst));														\
		TIMER_PRINT("Storing data", 0);															\
	}																							\
	TIMER_DELETE;																				\
}																								\

extern "C" void cudaMallocWrapper(void** devPtr, size_t count);
extern "C" void cudaFreeWrapper(void *devPtr);
extern "C" void cudaMemcpyHostToDeviceWrapper(void *dst, const void *src,  size_t count);
extern "C" int cudaGetDeviceCountWrapper(void);
extern "C" void cudaGetDevicePropertiesWrapper(cudaDeviceProp *prop, int dev);
extern "C" void cudaMallocArrayWrapper(void** devPtr, size_t width, size_t height);
extern "C" void cudaMemcpyHostToArrayWrapper(void *dst, const void *src, size_t size);
extern "C" void cudaFreeArrayWrapper(void *devPtr);

extern "C" int iDivUp(int a, int b);
extern "C" int iRoundUp(int a, int b);
extern "C" void cudaZeroMem(float *d_mem, int length);
extern "C" void cudaSumAlongRows(float *src, float *dst, int xdim, int ydim, bool device = false);
extern "C" void cudaSumAlongCols(float *src, float *dst, int xdim, int ydim, bool device = false);
extern "C" float cudaSum(float *src, int length, bool device = false);
extern "C" float cudaMul(float *src, int length, bool device = false);
extern "C" float cudaMax(float *src, int length, bool device = false);
extern "C" float cudaMin(float *src, int length, bool device = false);
extern "C" float cudaReduction(float *src, int length, bool device = false);
extern "C" float cudaSSD(float *src1, float *src2, int length, bool device = false);
extern "C" float cudaSAD(float *src1, float *src2, int length, bool device = false);
extern "C" void cudaSSDBinary(float *src1, float *src2, float *dst, int length, bool device = false);
extern "C" void cudaSADBinary(float *src1, float *src2, float *dst, int length, bool device = false);

#endif
