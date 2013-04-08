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

#ifndef _GPU_HIST_H_
#define _GPU_HIST_H_

#include "gpu_basics.h"

/*
	Expects the cell_len and the number of threads to be a power of two
	Note:
		This method requires a minimum of 64 threads.
*/
__global__ void gpuReduceHist(float *interim, float *hist, int cell_len)
{
	extern __shared__ float shared[];							//Allocated by the caller, must be set to the number of threads
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;
	const int threads = blockDim.x;

	int g_ofs = IMUL(bid, cell_len);
	//Load data into shared memory
	shared[tid] = 0;
	for (int i = tid; i < cell_len; i += threads)
		shared[tid] += interim[g_ofs + i];
	__syncthreads();
	
#ifdef __DEVICE_EMULATION__
	for (unsigned int d = threads >> 1; d > 0; d >>= 1) 
	{
		if (tid < d)
			shared[tid] += shared[tid + d];
		__syncthreads();
	}
#else
	for (unsigned int d = threads >> 1; d > 32; d >>= 1) 
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

	if (tid == 0)
		hist[bid]= shared[0];
}

__device__ void gpuHist_DWordIndex(unsigned int bin, unsigned int bits_pbin, unsigned int unused_bits, unsigned int &dw_idx, unsigned int &bit_pos)
{
	unsigned int bin_by_bits_pbin = IMUL(bits_pbin, bin);
	unsigned int idx_in_bits = bin_by_bits_pbin + IMUL(bin_by_bits_pbin / (32 - unused_bits), unused_bits);	//idx in bits
	dw_idx = idx_in_bits >> 5;																				//idx in dwords
	bit_pos = idx_in_bits - (dw_idx <<5 );
}

__global__ void gpuHista(float *src, float *hist, int len, int bins, int bins_cur, int binOfs)
{
	extern __shared__ unsigned int s_hist[];

	const int tid = threadIdx.x + IMUL(threadIdx.y, blockDim.x);				//Linear thread ID
	const int wid = tid >> LOG2_WARP_SIZE;										//Warp ID
	const int bid = blockIdx.x;													//Linear block ID, it is assumed that the caller set up a 1D horizontal grid
	const int threads = IMUL(blockDim.x, blockDim.y);							//Total number of threads in the block
	const int blocks = gridDim.x;												//Total number of blocks in the grid, it is assumed that gridDim.y = 1
	const int warps = ceil((float)threads / WARP_SIZE);							//Total number of warps in the block
	const int hist_len = IMUL(bins_cur, warps); 

	//Initialize s_hist with zero
	for (int i = tid; i < hist_len; i += threads)
		s_hist[i] = 0;		
	__syncthreads();

	const int gid = tid + IMUL(bid, threads); 
	const int g_threads = IMUL(threads, blocks);
	volatile unsigned int *my_hist = s_hist + IMUL(wid, bins_cur);					//Histgoram pointer for group of horizontal threads
	for (int i = gid; i < len; i += g_threads)
	{	
		volatile unsigned int bin = (unsigned int) (src[i] * (bins - 1) + 0.5f);	
			
		if (bin < binOfs)
			continue;
		bin -= binOfs;
		if (bin >= bins_cur)
			continue;
		unsigned int tagged;
		do
		{
			unsigned int val = my_hist[bin] & 0x07FFFFFF;

			tagged = (tid << 27) | (val + 1);				//Only the least 5 bits of tid will be left after the shift so we don't need to do (tid & WARP_SZIE)
			my_hist[bin] = tagged;
		} while (my_hist[bin] != tagged);
	}

	__syncthreads();

	//Sum partial histograms up
	int hist_ofs = IMUL(bins, bid);
	for (int i = tid; i < bins_cur; i += threads)
	{
		unsigned int sum = s_hist[i];
		for (int j = i + bins_cur; j < hist_len; j += bins_cur)
			sum += s_hist[j];

		hist[hist_ofs + i] = sum & 0x07FFFFFF;
	}
}

#define GPUHIST_SHARED_LEN		4000
/*
	threads:
		Number of threads must be a power of 2.
	bits_pbin:		
		Must be 0, 1, 2, 4, 8, 16, or 32.
*/
__global__ void gpuHistb(float *src, float *hist, int len, int bins, int bits_pbin)
{
	__shared__ unsigned int shared[GPUHIST_SHARED_LEN];
	const int tid = threadIdx.x;				//Linear thread ID, caller must setup a 1D horizontal block
	const int bid = blockIdx.x;					//Linear block ID, it is assumed that the caller set up a 1D horizontal grid
	const int threads = blockDim.x;				//Total number of threads in the block, blockDim.y and blockDim.z must be 1
	const int blocks = gridDim.x;				//Total number of blocks in the grid, it is assumed that gridDim.y = 1
	const int log2threads = ceil(log2f(threads));
	const int bits = log2threads + ceil(log2f(blocks));
	int shared_len = GPUHIST_SHARED_LEN;
	const unsigned int max_count = 1 << bits_pbin;
	const unsigned int id = tid + (bid << log2threads);
	const int gid = tid + IMUL(bid, threads); 
	const int g_threads = IMUL(threads, blocks);

	if (bits_pbin > 1)
	{
		//Initialize 'shared' with zero
		for (int i = tid; i < shared_len; i += threads)
			shared[i] = 0;
		__syncthreads();

		for (int i = gid; i < len; i += g_threads)
		{
			unsigned int bin = (unsigned int) (src[i] * (bins - 1) + 0.5f);
			unsigned int bin_by_bits_pbin = IMUL(bin, bits_pbin);
			unsigned int ofs = bin_by_bits_pbin & (32 - 1);							//Mod 32
			unsigned int mask = ((1 << bits_pbin) - 1) << ofs;
			unsigned int smem_ofs = tid + IMUL(bin_by_bits_pbin >> 5, threads);
			unsigned int val = (shared[smem_ofs] & mask) >> ofs;
			val++;
			if(val == max_count)
			{
				hist[(bin << bits) + id] += val;
				val = 0;			
			}

			//Update the counter back in the shared memory
			shared[smem_ofs] = (shared[smem_ofs] & ~mask) | (val << ofs);
		}

		//Write what's left in the shared memory to our histogram
		for (unsigned int bin = 0; bin < bins; bin++)
		{
			unsigned int bin_by_bits_pbin = IMUL(bin, bits_pbin);
			unsigned int ofs = bin_by_bits_pbin & (32 - 1);							//Mod 32
			unsigned int mask = ((1 << bits_pbin) - 1) << ofs;
			unsigned int smem_ofs = tid + IMUL(bin_by_bits_pbin >> 5, threads);
			unsigned int val = (shared[smem_ofs] & mask) >> ofs;

			hist[(bin << bits) + id] += val;
		}
	}
	else
	{
		for (int i = gid; i < len; i += g_threads)
		{
			unsigned int bin = (unsigned int) (src[i] * (bins - 1) + 0.5f);
			hist[(bin << bits) + id]++;
		}
	}
}

__global__ void gpuHistc(float *src, float *hist, int len, int bins, int bits_pbin)
{
	__shared__ unsigned int shared[GPUHIST_SHARED_LEN];
	const int tid = threadIdx.x;												//Linear thread ID, caller must setup a 1D horizontal block
	const int wid = tid >> LOG2_WARP_SIZE;										//Warp ID
	const int bid = blockIdx.x;													//Linear block ID, it is assumed that the caller set up a 1D horizontal grid
	const int threads = blockDim.x;												//Total number of threads in the block, blockDim.y and blockDim.z must be 1
	const int blocks = gridDim.x;												//Total number of blocks in the grid, it is assumed that gridDim.y = 1
	const int log2threads = ceil(log2f(threads));
	const int bits = log2threads + ceil(log2f(blocks));
	int shared_len = GPUHIST_SHARED_LEN;
	const unsigned int max_count = 1 << bits_pbin;
	const unsigned int id = tid + (bid << log2threads);
	const int gid = tid + IMUL(bid, threads); 
	const int g_threads = IMUL(threads, blocks);
	const int unused_bits = 5 + (27 % bits_pbin);
	unsigned int dw_idx, bit_pos;

	//Initialize 'shared' with zero
	for (int i = tid; i < shared_len; i += threads)
		shared[i] = 0;
	__syncthreads();

	gpuHist_DWordIndex(bins, bits_pbin, unused_bits, dw_idx, bit_pos);
	unsigned int sharedlen_warp = IMUL(wid, dw_idx) + min(1, bit_pos);
	volatile unsigned int *warp_hist = shared + sharedlen_warp;					//Histgoram pointer for each warp
	for (int i = gid; i < len; i += g_threads)
	{
		unsigned int bin = (unsigned int) (src[i] * (bins - 1) + 0.5f);
		gpuHist_DWordIndex(bin, bits_pbin, unused_bits, dw_idx, bit_pos);
		unsigned int mask = ((1 << bits_pbin) - 1) << bit_pos;
		
		//Update the counter back in the shared memory
		unsigned int tagged, val, org;
		do
		{
			org = warp_hist[dw_idx];
			val = (org & mask) >> bit_pos;
			val++;
			if (val == max_count) val = 0;

			tagged = (tid << 27) | (org & ~(mask | 0xf8000000)) | (val << bit_pos);				//Only the least 5 bits of tid will be left after the shift so we don't need to do (tid & WARP_SZIE)
			warp_hist[dw_idx] = tagged;
		} while (warp_hist[dw_idx] != tagged);

		if (val == 0)				//update the per thread histogram
			hist[(bin << bits) + id] += val;
	}

	__syncthreads();

	//Write what's left in the shared memory to our histogram
	for (unsigned int bin = tid & (WARP_SIZE - 1); bin < bins; bin += WARP_SIZE)
	{
		gpuHist_DWordIndex(bin, bits_pbin, unused_bits, dw_idx, bit_pos);
		unsigned int mask = ((1 << bits_pbin) - 1) << bit_pos;
		unsigned int val = (warp_hist[dw_idx] & mask) >> bit_pos;

		hist[(bin << bits) + id] += val;
	}
}

/*
	threads:
		For best perforamnce, the number of threads should be a multiple
		of the warp size.
	bits_pbin:		
		Must be 0, 1, 2, 4, 8, 16, or 32.
*/
__global__ void gpuHist_Approx(float *src, float *hist, int len, int bins, int bits_pbin, float bins_pthread)
{
	__shared__ unsigned int shared[GPUHIST_SHARED_LEN];
	const int tid = threadIdx.x;				//Linear thread ID, caller must setup a 1D horizontal block
	const int bid = blockIdx.x;					//Linear block ID, it is assumed that the caller set up a 1D horizontal grid
	const int threads = blockDim.x;				//Total number of threads in the block, blockDim.y and blockDim.z must be 1
	const int blocks = gridDim.x;				//Total number of blocks in the grid, it is assumed that gridDim.y = 1
	const int bits = ceil(log2f(blocks));
	int shared_len = GPUHIST_SHARED_LEN;
	const unsigned int max_count = 1 << bits_pbin;
	const unsigned int id = bid;
	const int gid = tid + IMUL(bid, threads); 
	const int g_threads = IMUL(threads, blocks);
	const unsigned int first_bin = (unsigned int) (bins_pthread * tid);
	const unsigned int last_bin = (unsigned int) (bins_pthread * (tid + 1));

	//Initialize 'shared' with zero
	for (int i = tid; i < shared_len; i += threads)
		shared[i] = 0;
	__syncthreads();

	for (int i = gid; i < len; i += g_threads)
	{
		unsigned int bin = (unsigned int) (src[i] * (bins - 1) + 0.5f);
		if (bin >= first_bin && bin < last_bin)
		{
			unsigned int bin_by_bits_pbin = IMUL(bin, bits_pbin);
			unsigned int ofs = bin_by_bits_pbin & (32 - 1);							//Mod 32
			unsigned int mask = ((1 << bits_pbin) - 1) << ofs;
			unsigned int smem_ofs = bin_by_bits_pbin >> 5;
			unsigned int val = (shared[smem_ofs] & mask) >> ofs;
			val++;
			if(val == max_count)
			{
				hist[(bin << bits) + id] += val;
				val = 0;			
			}

			//Update the counter back in the shared memory
			shared[smem_ofs] = (shared[smem_ofs] & ~mask) | (val << ofs);
		}
	}

	__syncthreads();

	//Write what's left in the shared memory to our histogram
	for (int bin = tid; bin < bins; bin += threads)
	{
		unsigned int bin_by_bits_pbin = IMUL(bin, bits_pbin);
		unsigned int ofs = bin_by_bits_pbin & (32 - 1);							//Mod 32
		unsigned int mask = ((1 << bits_pbin) - 1) << ofs;
		unsigned int smem_ofs = bin_by_bits_pbin >> 5;
		unsigned int val = (shared[smem_ofs] & mask) >> ofs;

		hist[(bin << bits) + id] += val;
	}
}

__global__ void gpuCombineHist2DSrcData(float *src1, float *src2, float *dst, int length, int xbins, int ybins)
{
	int g_ofs = IMUL(IMUL(gridDim.x, blockIdx.y) + blockIdx.x, blockDim.x) + threadIdx.x;

	if (g_ofs < length)
	{
		float f_src1 = src1[g_ofs];
		float f_src2 = src2[g_ofs];
		int a1 = (int) (f_src1 * (xbins - 1) + 0.5f);
		int a2 = (int) (f_src2 * (ybins - 1) + 0.5f);
		float f_dst = a1 + IMUL(a2, xbins);

		dst[g_ofs] = f_dst / (IMUL(xbins, ybins) - 1);
	}	
}

#endif