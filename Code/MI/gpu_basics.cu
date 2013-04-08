#ifndef _GPU_BASICS_CU_
#define _GPU_BASICS_CU_

#include "gpu_basics.h"
/*
	Expects blockDim.y = 1.
*/
__global__ void gpuZeroMem(float *mem, int len)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int threads = blockDim.x;
	const unsigned int bid = blockIdx.x + IMUL(blockIdx.y, gridDim.x);

	int g_ofs = IMUL(bid, threads) + tid;
	if (g_ofs < len)
		mem[g_ofs] = 0.0f;
}

__global__ void gpuSumGlobalMem(float *src, float *dst, int num, int len)
{
	int data_per_block = ceil((float) len / gridDim.x);
	int data_per_thread = ceil((float) data_per_block / blockDim.x);
	int baseBlockIdx = IMUL(blockIdx.x, data_per_block);
	int baseIdx = IMUL(threadIdx.x, data_per_thread) + baseBlockIdx;
	int end = min(baseIdx + data_per_thread, baseBlockIdx + data_per_block);		//Clamp to block boundary
	end = min(end, len);					//Clamp to len

	for (int i = baseIdx; i < end; i++)
	{
		float sum = 0;
		for (int j = 0; j < num; j++)
			sum += src[IMUL(len, j) + i];
		dst[i] = sum;
	}
}

/*
	len: length of data processed by each block, must not exceed 2 * MAX_THREADS.

	Notes:
		blockDim.y must be 1.
*/
__global__ void gpuSumAlongRows(float *src, float *dst, int len, int src_pitch, int dst_pitch)
{
	__shared__ float shared[2 * MAX_THREADS];
	float *src_block = src + IMUL(src_pitch, blockIdx.y) + IMUL(len, blockIdx.x);			//Begining of data for current block
	int smem_len = 1 << (int)ceil(log2((float)len));										//Data length in shared memory is a power of two

	unsigned int tid = threadIdx.x;
	//read into shared memory
	if (tid < len)
		shared[tid] = src_block[tid];
	else
		shared[tid] = 0;

	//read the second half
	int tid2 = tid + (smem_len >> 1);
	if (tid2 < len)
		shared[tid2] = src_block[tid2];
	else
		shared[tid2] = 0;
	__syncthreads();

#ifdef __DEVICE_EMULATION__
	for (unsigned int d = smem_len >> 1; d > 0; d >>= 1) 
	{
		if (tid < d)
			shared[tid] += shared[tid + d];
		__syncthreads();
	}
#else
	for (unsigned int d = smem_len >> 1; d > 32; d >>= 1) 
	{
		if (tid < d)
			shared[tid] += shared[tid + d];
		__syncthreads();
	}

	//Shared read/writes are SIMD synchronous within a warp so skip syncthreads 
	//by unrolling the last 6 predicated steps 
	if (tid < 32) {  
		shared[tid] += shared[tid + 32];
		shared[tid] += shared[tid + 16];
		shared[tid] += shared[tid + 8];
		shared[tid] += shared[tid + 4];
		shared[tid] += shared[tid + 2];
		shared[tid] += shared[tid + 1];
	}
#endif
	__syncthreads();

	//Write the result 
	if (tid == 0)
		dst[dst_pitch * blockIdx.y + blockIdx.x] = shared[0];
}

/*
	len: length of data processed by each block, must not exceed 2 * MAX_THREADS.
*/
__global__ void gpuSumAlongCols(float *src, float *dst, int len, int src_pitch, int dst_pitch)
{
	__shared__ float shared[2 * MAX_THREADS];
	//float *src_block = src + IMUL(src_pitch, blockIdx.y) + IMUL(len, blockIdx.x);			//Begining of data for current block
	float *src_block = src + blockIdx.x + IMUL(IMUL(src_pitch, blockIdx.y), len);			//Begining of data for current block
	int smem_len = 1 << (int)ceil(log2((float)len));										//Data length in shared memory is a power of two

	unsigned int tid = threadIdx.y;
	//read into shared memory
	if (tid < len)
		shared[tid] = src_block[tid * src_pitch];
	else
		shared[tid] = 0;

	//read the second half
	int tid2 = tid + (smem_len >> 1);
	if (tid2 < len)
		shared[tid2] = src_block[tid2 * src_pitch];
	else
		shared[tid2] = 0;
	__syncthreads();

#ifdef __DEVICE_EMULATION__
	for (unsigned int d = smem_len >> 1; d > 0; d >>= 1) 
	{
		if (tid < d)
			shared[tid] += shared[tid + d];
		__syncthreads();
	}
#else
	for (unsigned int d = smem_len >> 1; d > 32; d >>= 1) 
	{
		if (tid < d)
			shared[tid] += shared[tid + d];
		__syncthreads();
	}

	//Shared read/writes are SIMD synchronous within a warp so skip syncthreads 
	//by unrolling the last 6 predicated steps 
	if (tid < 32) {  
		shared[tid] += shared[tid + 32];
		shared[tid] += shared[tid + 16];
		shared[tid] += shared[tid + 8];
		shared[tid] += shared[tid + 4];
		shared[tid] += shared[tid + 2];
		shared[tid] += shared[tid + 1];
	}
#endif
	__syncthreads();

	//Write the result 
	if (tid == 0)
		dst[dst_pitch * blockIdx.y + blockIdx.x] = shared[0];
}

__device__ float inlineMax(float a, float b)
{
	return a > b ? a : b;
}

__device__ float initMax()
{
	return -FLT_MAX;
}

__device__ float inlineMin(float a, float b)
{
	return a < b ? a : b;
}

__device__ float initMin()
{
	return FLT_MAX;
}

__device__ float inlineSum(float a, float b)
{
	return a + b;
}

__device__ float initSum()
{
	return 0.0f;
}

__device__ float inlineMul(float a, float b)
{
	return a * b;
}

__device__ float initMul()
{
	return 1.0f;
}

GPU_REDUCTION(Sum)
GPU_REDUCTION(Mul)
GPU_REDUCTION(Max)
GPU_REDUCTION(Min)

/*
	xyzReduction methods can be used for debugging and tesing of 
	reduction methods.
*/
__device__ float inlineReduction(float a, float b)
{
	return a > b ? a : b;
}

__device__ float initReduction()
{
	return -FLT_MAX;
}

__global__ void gpuReduction(float *src, float *dst, int len)
{
	__shared__ float shared[2 * MAX_THREADS];
	int bid = IMUL(blockIdx.y, gridDim.x) + blockIdx.x;									//Linear Block ID
	int g_ofs = IMUL(bid, blockDim.x) << 1;												//Each block processes two elements of data per thread
	float *src_block = src + g_ofs;														//Begining of data for current block
	int block_len = min(max(len - g_ofs, 0), 2 * MAX_THREADS);							//g_ofs can become more than len
	int smem_len = block_len != 0 ? 1 << (int) ceil(log2((float) block_len)) : 0;		//Data length in shared memory is a power of two

	unsigned int tid = threadIdx.x;
	//read into shared memory
	if (tid < block_len)
		shared[tid] = src_block[tid];
	else
		shared[tid] = initReduction();

	//read the second half
	int tid2 = tid + (smem_len >> 1);
	if (tid2 < block_len)
		shared[tid2] = src_block[tid2];
	else
		shared[tid2] = initReduction();
	__syncthreads();

#ifdef __DEVICE_EMULATION__
	for (unsigned int d = smem_len >> 1; d > 0; d >>= 1) 
	{
		if (tid < d)
			shared[tid] = inlineReduction( shared[tid], shared[tid + d]);
		__syncthreads();
	}
#else
	for (unsigned int d = smem_len >> 1; d > 32; d >>= 1) 
	{
		if (tid < d)
			shared[tid] = inlineReduction( shared[tid], shared[tid + d]);
		__syncthreads();
	}

	//Shared read/writes are SIMD synchronous within a warp so skip syncthreads 
	//by unrolling the last 6 predicated steps 
	//Note: This section does not work in Emulation mode
	if (tid < 32) {  
		shared[tid] = inlineReduction(shared[tid], shared[tid + 32]);
		shared[tid] = inlineReduction(shared[tid], shared[tid + 16]);
		shared[tid] = inlineReduction(shared[tid], shared[tid + 8]);
		shared[tid] = inlineReduction(shared[tid], shared[tid + 4]);
		shared[tid] = inlineReduction(shared[tid], shared[tid + 2]);
		shared[tid] = inlineReduction(shared[tid], shared[tid + 1]);
	}
#endif
	__syncthreads();

	//Write the result 
	if (tid == 0)
		dst[bid]= shared[0];
}

__device__ float inlineSSDBinary(float a, float b)
{
	float f = a - b;
	return f * f;
}

__device__ float inlineSADBinary(float a, float b)
{
	return fabsf(a -b);
}

//Sum of Squared Differences
GPU_BINARY(SSDBinary)
//Sum of Absolute Differences
GPU_BINARY(SADBinary)
#endif
