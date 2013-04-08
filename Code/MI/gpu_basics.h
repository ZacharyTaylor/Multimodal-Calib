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

#ifndef _GPU_BASICS_H_
#define _GPU_BASICS_H_

#define WARP_SIZE			32
#define LOG2_WARP_SIZE		5						//Must conform to the value of WARP_SIZE
#define HALFWARP_SIZE		16
#define SHARED_MEM_SIZE		16384
#define MAX_USABLE_SHARED	(SHARED_MEM_SIZE - 64)	//Typically 16-64 bytes of data is pre-allocated for compiler to pass execution configuration and the arguments
#define MAX_BLOCKS_PER_DIM	65536					//Maximum number of blocks for each dimension of the grid
#define PI					3.14159265358979f
#define FLT_MAX				3.402823466e+38F        /* max value */

#define SDATA(index)		CUT_BANK_CHECKER(sdata, index)
#define TO_DEGREE(radian)	radian * (180.0f / PI)

#ifdef __DEVICE_EMULATION__
	#define IMUL(a, b)	(((int)(a)) * ((int)(b)))
	#define UIMUL(a, b) (((unsigned int)(a)) * ((unsigned int)(b)))
	#define MAX_THREADS			32
	#define MAX_KERNEL_RADIUS	8
#else
	//on 8800 __mul24 and _umal24 take only 4 cycles to compute, in the future however this may change and
	//direct 32-bit operations (that take 15 cycles) can be faster. Use IMUL and UIMUL instead of __[u]mul24
	//so that the code can be easilty adapted in the future.
	//Note that __[u]mul24 works correctly as long as a and b with their possible sign can be represented with
	//24 bits. So the result can be more than 24 bits long and up to the expected 32-bits that can be stored.
	#define IMUL(a, b) __mul24(a, b)
	#define UIMUL(a, b) __umul24(a, b)
	//Maximum number of threads per block based on GPU specification
	#define MAX_THREADS			512
	#define MAX_KERNEL_RADIUS	64			//One requirement: MAX_RADIUS < 0.5 * ( MAX_THREADS - 1 ) rounded down to a a multipe of 32.
#endif

/*
	Note: 
		-The macro will not compile if there are any characters after '\'.
		-Comments cannot be used anywhere inside the macro.
*/
#ifdef __DEVICE_EMULATION__
#define GPU_REDUCTION(fnSuffix)														\
__global__ void gpu##fnSuffix(float *src, float *dst, int len)						\
{																					\
	__shared__ float shared[2 * MAX_THREADS];										\
	int bid = IMUL(blockIdx.y, gridDim.x) + blockIdx.x;								\
																					\
	int g_ofs = IMUL(bid, blockDim.x) << 1;											\
	float *src_block = src + g_ofs;													\
	int block_len = min(max(len - g_ofs, 0), 2 * MAX_THREADS);						\
	int smem_len = block_len != 0 ? 1 << (int) ceil(log2((float) block_len)) : 0;	\
																					\
	unsigned int tid = threadIdx.x;													\
																					\
	if (tid < block_len)															\
		shared[tid] = src_block[tid];												\
	else																			\
		shared[tid] = init##fnSuffix();												\
																					\
	int tid2 = tid + (smem_len >> 1);												\
	if (tid2 < block_len)															\
		shared[tid2] = src_block[tid2];												\
	else																			\
		shared[tid2] = init##fnSuffix();											\
	__syncthreads();																\
																					\
	for (unsigned int d = smem_len >> 1; d > 0; d >>= 1)							\
	{																				\
		if (tid < d)																\
			shared[tid] = inline##fnSuffix(shared[tid], shared[tid + d]);			\
		__syncthreads();															\
	}																				\
	__syncthreads();																\
																					\
	if (tid == 0)																	\
		dst[bid]= shared[0];														\
}																					

#else
#define GPU_REDUCTION(fnSuffix)														\
__global__ void gpu##fnSuffix(float *src, float *dst, int len)						\
{																					\
	__shared__ float shared[2 * MAX_THREADS];										\
	int bid = IMUL(blockIdx.y, gridDim.x) + blockIdx.x;								\
																					\
	int g_ofs = IMUL(bid, blockDim.x) << 1;											\
	float *src_block = src + g_ofs;													\
	int block_len = min(max(len - g_ofs, 0), 2 * MAX_THREADS);						\
	int smem_len = block_len != 0 ? 1 << (int) ceil(log2((float) block_len)) : 0;	\
																					\
	unsigned int tid = threadIdx.x;													\
																					\
	if (tid < block_len)															\
		shared[tid] = src_block[tid];												\
	else																			\
		shared[tid] = init##fnSuffix();												\
																					\
	int tid2 = tid + (smem_len >> 1);												\
	if (tid2 < block_len)															\
		shared[tid2] = src_block[tid2];												\
	else																			\
		shared[tid2] = init##fnSuffix();											\
	__syncthreads();																\
																					\
	for (unsigned int d = smem_len >> 1; d > 32; d >>= 1)							\
	{																				\
		if (tid < d)																\
			shared[tid] = inline##fnSuffix(shared[tid], shared[tid + d]);			\
		__syncthreads();															\
	}																				\
																					\
	if (tid < 32)																	\
	{																				\
		shared[tid] = inline##fnSuffix(shared[tid], shared[tid + 32]);				\
		shared[tid] = inline##fnSuffix(shared[tid], shared[tid + 16]);				\
		shared[tid] = inline##fnSuffix(shared[tid], shared[tid + 8]);				\
		shared[tid] = inline##fnSuffix(shared[tid], shared[tid + 4]);				\
		shared[tid] = inline##fnSuffix(shared[tid], shared[tid + 2]);				\
		shared[tid] = inline##fnSuffix(shared[tid], shared[tid + 1]);				\
	}																				\
	__syncthreads();																\
																					\
	if (tid == 0)																	\
		dst[bid]= shared[0];														\
}																					

#endif

#define GPU_UNARY(fnSuffix)																			\
__global__ void gpu##fnSuffix(float *src, float *dst, int length)									\
{																									\
	int g_ofs = IMUL(IMUL(gridDim.x, blockIdx.y) + blockIdx.x, blockDim.x) + threadIdx.x;			\
																									\
	if (g_ofs < length)																				\
	{																								\
		float f = src[g_ofs];																		\
		dst[g_ofs] = inline##fnSuffix(f);															\
	}																								\
}																									\

#define GPU_BINARY(fnSuffix)																		\
__global__ void gpu##fnSuffix(float *src1, float *src2, float *dst, int length)						\
{																									\
	int g_ofs = IMUL(IMUL(gridDim.x, blockIdx.y) + blockIdx.x, blockDim.x) + threadIdx.x;			\
																									\
	if (g_ofs < length)																				\
	{																								\
		float f1 = src1[g_ofs];																		\
		float f2 = src2[g_ofs];																		\
		dst[g_ofs] = inline##fnSuffix(f1, f2);														\
	}																								\
}																									\

__global__ void gpuSumGlobalMem(float *src, float *dst, int num, int len);
__global__ void gpuReduction(float *src, float *dst, int len);

#endif