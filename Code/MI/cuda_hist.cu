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

// includes, system
#include <stdlib.h>
#include <tchar.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cutil.h"
#include "cuda_basics.h"
#include "cuda_hist.h"

// includes, kernels
#include "gpu_hist.cu"

extern "C" double cudaHista(float *src, float *hist, int length, int bins, cudaHistOptions *p_options /*= NULL*/, bool device /*= false*/)
{
	dim3 grid, block;
	int size = length * sizeof(float);
	//Device memory pointers
	float *d_src, *d_hist;
	double time = 0;
	unsigned int hTimer;
	cudaHistOptions options;
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	TIMER_CREATE;

	if (!device)
	{
		TIMER_START;
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));
		TIMER_PRINT("Loading data", 0);
	}
	else
	{
		d_src = src;				//Do not copy hist!
	}

	if (p_options == NULL)
	{
		options.threads = 160;
		options.blocks = 64;
	}
	else options = *p_options;

	//Perform sanity checks
	if (options.threads > MAX_THREADS)
		printf("'threads' exceed the maximum."), exit(1);
	if (options.threads % WARP_SIZE != 0)
		printf("'threads' must be a multiple of the WARP_SIZE."), exit(1);
	if (options.blocks > MAX_BLOCKS_PER_DIM)
		printf("'blocks' exceed the maximum."), exit(1);

	//Prepare the execution configuration
	int warps = options.threads / WARP_SIZE;
	int max_bins = MAX_USABLE_SHARED / sizeof(unsigned int) / warps;
	block.x = WARP_SIZE; block.y = warps; block.z = 1;
	grid.x = options.blocks; grid.y = 1; grid.z = 1;

	int shared_mem_size = max_bins * warps * sizeof(unsigned int);
	if (shared_mem_size> MAX_USABLE_SHARED)
		printf("Maximum shared memory exceeded."), exit(1);

    CUDA_SAFE_CALL(cudaThreadSynchronize());								
    CUT_SAFE_CALL(cutStartTimer(hTimer));								

	TIMER_START;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, options.blocks * bins * sizeof(float)));
	//Initialize histogram memory
	CUDA_SAFE_CALL(cudaMemset(d_hist, 0, options.blocks * bins * sizeof(float)));
	TIMER_PRINT("Initializing data", 0);

	TIMER_START;
	int calls = iDivUp(bins, max_bins);
	for (int i = 0; i < calls; i++)
	{
		gpuHista<<<grid, block, shared_mem_size>>>(d_src, d_hist + max_bins * i, length, bins, min(max_bins, bins - max_bins * i), max_bins * i);
		CUT_CHECK_ERROR("gpuHista() execution failed\n");
	}
	TIMER_PRINT("gpuHista", length);

	//Sum up the histograms
	int numHists = grid.x;
	if (numHists > 1)
	{
		block.x = MAX_THREADS; block.y = 1; block.z = 1;
		grid.x = ceil((float) bins / block.x); grid.y = 1; grid.z = 1;

		TIMER_START;
		gpuSumGlobalMem<<<grid, block>>>(d_hist, d_hist, numHists, bins);
		CUT_CHECK_ERROR("gpuSumGlobalMem() execution failed\n");
		TIMER_PRINT("gpuSumGlobalMem", bins * numHists);
	}

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	time = cutGetTimerValue(hTimer);
	CUT_SAFE_CALL(cutDeleteTimer(hTimer));

	TIMER_START;
	if (!device)
	{
		CUDA_SAFE_CALL(cudaMemcpy(hist, d_hist, bins * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(d_src));
	}
	else
	{
		CUDA_SAFE_CALL(cudaMemcpy(hist, d_hist, bins * sizeof(float), cudaMemcpyDeviceToDevice));
	}
	CUDA_SAFE_CALL(cudaFree(d_hist));
	TIMER_PRINT("Storing data", 0);
	TIMER_DELETE;

	return time;
}

extern "C" double cudaHistb(float *src, float *hist, int length, int bins, cudaHistOptions *p_options /*= NULL*/, bool device /*= false*/)
{
	dim3 grid, block;
	int size = length * sizeof(float);
	//Device memory pointers
	float *d_src, *d_hist, *d_interim;
	double time = 0;
	unsigned int hTimer;
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	TIMER_CREATE;

	if (!device)
	{
		TIMER_START;
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, bins * sizeof(float)));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));
		TIMER_PRINT("Loading data", 0)
	}
	else
	{
		d_src = src; d_hist = hist;				
	}

	cudaHistOptions options;
	if (p_options)
		options = *p_options;
	else
	{
		options.threads = 128;
		options.blocks = 8;
	}

	//Prepare execution configuration
	block.x = options.threads; block.y = 1; block.z = 1;
	grid.x = options.blocks; grid.y = 1; grid.z = 1;

	int cell_len = powf(2.0f, ceilf(log2f(block.x)) + ceilf(log2f(grid.x))); 
	int hist_len = cell_len * bins;

    CUDA_SAFE_CALL(cudaThreadSynchronize());								
	CUT_SAFE_CALL(cutStartTimer(hTimer));								

	TIMER_START;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_interim, hist_len * sizeof(float)));
	TIMER_PRINT("Allocating memory", 0);

	cudaZeroMem(d_interim, hist_len);				//This much faster than cudaMemset

	TIMER_START;
	int shared_len_pt = GPUHIST_SHARED_LEN >> (int) ceil(log2f(options.threads));							//Length of shared memory available to each thread (in int32)
	int n = (shared_len_pt << 5) / bins;
	const int bits_pbin = n > 0 ? min((1 << (int)log2f(n)), 32) : 0;					 			//Number of bits per bin per thread 0, 1, 2, 4, 8, 16, 32	
#ifdef VERBOSE
	printf("bits per bin: %d\n", bits_pbin);
#endif
	gpuHistb<<<grid, block>>>(d_src, d_interim, length, bins, bits_pbin);
	CUT_CHECK_ERROR("gpuHistb() execution failed\n");
	TIMER_PRINT("gpuHistb", length);

	//Reduce the interim histogram 
	if (bins > MAX_BLOCKS_PER_DIM)						//We want the bins to fit in horizontal grid.x
		printf("Maximimum bins exceeded."), exit(1);

	block.x = max(min(cell_len, MAX_THREADS), 64); block.y = 1; block.z = 1;			//gpuReduceHist requires at least 64 threads
	grid.x = bins; grid.y = 1; grid.z = 1;

	TIMER_START;
	gpuReduceHist<<<grid, block, block.x * sizeof(float)>>>(d_interim, d_hist, cell_len);
	CUT_CHECK_ERROR("gpuReduceHist() execution failed\n");
	TIMER_PRINT("gpuReduceHist", hist_len);

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	time = cutGetTimerValue(hTimer);
	CUT_SAFE_CALL(cutDeleteTimer(hTimer));

	TIMER_START;
	CUDA_SAFE_CALL(cudaFree(d_interim));
	if (!device)
	{
		CUDA_SAFE_CALL(cudaMemcpy(hist, d_hist, bins * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(d_src));
		CUDA_SAFE_CALL(cudaFree(d_hist));
	}
	TIMER_PRINT("Storing data", 0);
	TIMER_DELETE;

	return time;
}

extern "C" double cudaHistc(float *src, float *hist, int length, int bins, cudaHistOptions *p_options /*= NULL*/, bool device /*= false*/)
{
	dim3 grid, block;
	int size = length * sizeof(float);
	//Device memory pointers
	float *d_src, *d_hist, *d_interim;
	double time = 0;
	unsigned int hTimer;
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	TIMER_CREATE;

	if (!device)
	{
		TIMER_START;
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, bins * sizeof(float)));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));
		TIMER_PRINT("Loading data", 0)
	}
	else
	{
		d_src = src; d_hist = hist;				
	}

	cudaHistOptions options;
	if (p_options)
		options = *p_options;
	else
	{
		options.threads = 128;
		options.blocks = 8;
	}

	//Prepare execution configuration
	block.x = options.threads; block.y = 1; block.z = 1;
	grid.x = options.blocks; grid.y = 1; grid.z = 1;

	int cell_len = powf(2.0f, ceilf(log2f(block.x)) + ceilf(log2f(grid.x))); 
	int hist_len = cell_len * bins;

    CUDA_SAFE_CALL(cudaThreadSynchronize());								
	CUT_SAFE_CALL(cutStartTimer(hTimer));								

	TIMER_START;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_interim, hist_len * sizeof(float)));
	TIMER_PRINT("Allocating memory", 0);

	cudaZeroMem(d_interim, hist_len);				//This much faster than cudaMemset

	TIMER_START;
	int shared_len_pw = GPUHIST_SHARED_LEN >> (int) ceil(log2f(options.threads >> LOG2_WARP_SIZE));	//Length of shared memory available to each warp (in int32)
	int bits_pbin = max(min((shared_len_pw * 27) / bins, 27), 0);
	for (int i = 1; i <= 28; i++)
	{
		if (bits_pbin >= 27 / i)
		{
			bits_pbin = 27 / i;
			break;
		}
	}
#ifdef VERBOSE
	printf("bits per bin: %d\n", bits_pbin);
#endif
	gpuHistc<<<grid, block>>>(d_src, d_interim, length, bins, bits_pbin);
	CUT_CHECK_ERROR("gpuHistc() execution failed\n");
	TIMER_PRINT("gpuHistc", length);

	//Reduce the interim histogram 
	/*if (cell_len > 1024)
		printf("Maximimum length exceeded."), exit(1);
	if (bins > MAX_BLOCKS_PER_DIM)						//We want the bins to fit in horizontal grid.x
		printf("Maximimum bins exceeded."), exit(1);

	block.x = cell_len >> 1; block.y = 1; block.z = 1;
	grid.x = bins; grid.y = 1; grid.z = 1;

	TIMER_START;
	gpuReduceHist<<<grid, block, cell_len * sizeof(float)>>>(d_interim, d_hist);
	CUT_CHECK_ERROR("gpuReduceHist() execution failed\n");
	TIMER_PRINT("gpuReduceHist", hist_len);*/

	if (bins > MAX_BLOCKS_PER_DIM)						//We want the bins to fit in horizontal grid.x
		printf("Maximimum bins exceeded."), exit(1);

	block.x = max(min(cell_len, MAX_THREADS), 64); block.y = 1; block.z = 1;			//gpuReduce requires at least 64 threads
	grid.x = bins; grid.y = 1; grid.z = 1;

	TIMER_START;
	gpuReduceHist<<<grid, block, block.x * sizeof(float)>>>(d_interim, d_hist, cell_len);
	CUT_CHECK_ERROR("gpuReduceHist() execution failed\n");
	TIMER_PRINT("gpuReduceHist", hist_len);

	/*if (bins > MAX_BLOCKS_PER_DIM)						//We want the bins to fit in horizontal grid.x
		printf("Maximimum bins exceeded."), exit(1);

	cudaSumAlongRows(d_interim, d_hist, cell_len, bins, true);*/

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	time = cutGetTimerValue(hTimer);
	CUT_SAFE_CALL(cutDeleteTimer(hTimer));

	TIMER_START;
	CUDA_SAFE_CALL(cudaFree(d_interim));
	if (!device)
	{
		CUDA_SAFE_CALL(cudaMemcpy(hist, d_hist, bins * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(d_src));
		CUDA_SAFE_CALL(cudaFree(d_hist));
	}
	TIMER_PRINT("Storing data", 0);
	TIMER_DELETE;

	return time;
}

extern "C" double cudaHist_Approx(float *src, float *hist, int length, int bins, cudaHistOptions *p_options /*= NULL*/, bool device /*= false*/)
{
	dim3 grid, block;
	int size = length * sizeof(float);
	//Device memory pointers
	float *d_src, *d_hist, *d_interim;
	double time = 0;
	unsigned int hTimer;
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	TIMER_CREATE;

	if (!device)
	{
		TIMER_START;
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, bins * sizeof(float)));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));
		TIMER_PRINT("Loading data", 0)
	}
	else
	{
		d_src = src; d_hist = hist;				
	}

	cudaHistOptions options;
	if (p_options)
		options = *p_options;
	else
	{
		options.threads = 256;
		options.blocks = 16;
	}

	//Prepare execution configuration
	block.x = options.threads; block.y = 1; block.z = 1;
	grid.x = options.blocks; grid.y = 1; grid.z = 1;

	int cell_len = powf(2.0f, ceilf(log2f(grid.x)));
	int hist_len = cell_len * bins;

    CUDA_SAFE_CALL(cudaThreadSynchronize());								
	CUT_SAFE_CALL(cutStartTimer(hTimer));								

	TIMER_START;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_interim, hist_len * sizeof(float)));
	TIMER_PRINT("Allocating memory", 0);

	cudaZeroMem(d_interim, hist_len);				//This is much faster than cudaMemset

	TIMER_START;
	int n = (GPUHIST_SHARED_LEN << 5) / bins;
	const int bits_pbin = n > 0 ? min((1 << (int)log2f(n)), 32) : 0;					 			//Number of bits per bin per thread 0, 1, 2, 4, 8, 16, 32	
#ifdef VERBOSE
	printf("bits per bin: %d\n", bits_pbin);
#endif
	gpuHist_Approx<<<grid, block>>>(d_src, d_interim, length, bins, bits_pbin, (float) bins / options.threads);
	CUT_CHECK_ERROR("gpuHist_Approx() execution failed\n");
	TIMER_PRINT("gpuHist_Approx", length);

	//Reduce the interim histogram 
	if (bins > MAX_BLOCKS_PER_DIM)						//We want the bins to fit in horizontal grid.x
		printf("Maximimum bins exceeded."), exit(1);

	block.x = max(min(cell_len, MAX_THREADS), 64); block.y = 1; block.z = 1;			//gpuReduceHist requires at least 64 threads
	grid.x = bins; grid.y = 1; grid.z = 1;

	TIMER_START;
	gpuReduceHist<<<grid, block, block.x * sizeof(float)>>>(d_interim, d_hist, cell_len);
	CUT_CHECK_ERROR("gpuReduceHist() execution failed\n");
	TIMER_PRINT("gpuReduceHist", hist_len);

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	time = cutGetTimerValue(hTimer);
	CUT_SAFE_CALL(cutDeleteTimer(hTimer));

	TIMER_START;
	CUDA_SAFE_CALL(cudaFree(d_interim));
	if (!device)
	{
		CUDA_SAFE_CALL(cudaMemcpy(hist, d_hist, bins * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(d_src));
		CUDA_SAFE_CALL(cudaFree(d_hist));
	}
	TIMER_PRINT("Storing data", 0);
	TIMER_DELETE;

	return time;
}

extern "C" void cudaHist2Da(float *src1, float *src2, float *hist, int length, int xbins, int ybins, cudaHistOptions *p_options /*= NULL*/, bool device /*= false*/)
{
	dim3 grid, block;
	int bins = xbins * ybins;
	float *d_src1, *d_src2, *d_hist, *d_src;				//Device memory pointers
	int size = length * sizeof(float);
	TIMER_CREATE;

	TIMER_START;
	if (!device)
	{
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src1, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src2, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, bins * sizeof(float)));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src1, src1, size, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_src2, src2, size, cudaMemcpyHostToDevice));
	}
	else
	{
		d_src1 = src1; d_src2 = src2; d_hist = hist;
	}
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));				//Buffer to hold the conbined source data that can be passed to cudaHist
	TIMER_PRINT("Loading data", 0);

	//Combine src1 and src2 into a single array for processing by cudaHist
	//Prepare execution configuration
	const int max_threads = MAX_THREADS;
	int good_len = iRoundUp(length, WARP_SIZE);

	block.x = max_threads; block.y = 1; block.z = 1;
	grid.x = ceil(sqrtf(iDivUp(good_len, max_threads))); grid.y = grid.x; grid.z = 1;				//CUDA throws an excution error if grid.z is not 1

	TIMER_START;
	gpuCombineHist2DSrcData<<<grid, block>>>(d_src1, d_src2, d_src, length, xbins, ybins);
	CUT_CHECK_ERROR("gpuCombineHist2DSrcData() execution failed\n");
	TIMER_PRINT("gpuCombineHist2DSrcData", length);

	cudaHista(d_src, d_hist, length, bins, p_options, true);			//No need to initialize d_hist, will be done by cudaHist
	
	if (!device)
	{
		TIMER_START;
		CUDA_SAFE_CALL(cudaMemcpy(hist, d_hist, bins * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(d_src1));
		CUDA_SAFE_CALL(cudaFree(d_src2));
		CUDA_SAFE_CALL(cudaFree(d_hist));
		TIMER_PRINT("Storing data", 0);
	}
	CUDA_SAFE_CALL(cudaFree(d_src));

	TIMER_DELETE;
}

extern "C" void cudaHist2Db(float *src1, float *src2, float *hist, int length, int xbins, int ybins, cudaHistOptions *p_options /*= NULL*/, bool device /*= false*/)
{
	dim3 grid, block;
	int bins = xbins * ybins;
	float *d_src1, *d_src2, *d_hist, *d_src;				//Device memory pointers
	int size = length * sizeof(float);
	TIMER_CREATE;

	TIMER_START;
	if (!device)
	{
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src1, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src2, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, bins * sizeof(float)));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src1, src1, size, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_src2, src2, size, cudaMemcpyHostToDevice));
	}
	else
	{
		d_src1 = src1; d_src2 = src2; d_hist = hist;
	}
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));				//Buffer to hold the conbined source data that can be passed to cudaHist
	TIMER_PRINT("Loading data", 0);

	//Combine src1 and src2 into a single array for processing by cudaHist
	//Prepare execution configuration
	const int max_threads = MAX_THREADS;
	int good_len = iRoundUp(length, WARP_SIZE);

	block.x = max_threads; block.y = 1; block.z = 1;
	grid.x = ceil(sqrtf(iDivUp(good_len, max_threads))); grid.y = grid.x; grid.z = 1;				//CUDA throws an excution error if grid.z is not 1

	TIMER_START;
	gpuCombineHist2DSrcData<<<grid, block>>>(d_src1, d_src2, d_src, length, xbins, ybins);
	CUT_CHECK_ERROR("gpuCombineHist2DSrcData() execution failed\n");
	TIMER_PRINT("gpuCombineHist2DSrcData", length);

	cudaHistb(d_src, d_hist, length, bins, p_options, true);			//No need to initialize d_hist, will be done by cudaHist
	
	if (!device)
	{
		TIMER_START;
		CUDA_SAFE_CALL(cudaMemcpy(hist, d_hist, bins * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(d_src1));
		CUDA_SAFE_CALL(cudaFree(d_src2));
		CUDA_SAFE_CALL(cudaFree(d_hist));
		TIMER_PRINT("Storing data", 0);
	}
	CUDA_SAFE_CALL(cudaFree(d_src));

	TIMER_DELETE;
}

extern "C" void cudaHist2D_Approx(float *src1, float *src2, float *hist, int length, int xbins, int ybins, cudaHistOptions *p_options /*= NULL*/, bool device /*= false*/)
{
	dim3 grid, block;
	int bins = xbins * ybins;
	float *d_src1, *d_src2, *d_hist, *d_src;				//Device memory pointers
	int size = length * sizeof(float);
	TIMER_CREATE;

	TIMER_START;
	if (!device)
	{
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src1, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src2, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, bins * sizeof(float)));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src1, src1, size, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_src2, src2, size, cudaMemcpyHostToDevice));
	}
	else
	{
		d_src1 = src1; d_src2 = src2; d_hist = hist;
	}
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));				//Buffer to hold the conbined source data that can be passed to cudaHist
	TIMER_PRINT("Loading data", 0);

	//Combine src1 and src2 into a single array for processing by cudaHist
	//Prepare execution configuration
	const int max_threads = MAX_THREADS;
	int good_len = iRoundUp(length, WARP_SIZE);

	block.x = max_threads; block.y = 1; block.z = 1;
	grid.x = ceil(sqrtf(iDivUp(good_len, max_threads))); grid.y = grid.x; grid.z = 1;				//CUDA throws an excution error if grid.z is not 1

	TIMER_START;
	gpuCombineHist2DSrcData<<<grid, block>>>(d_src1, d_src2, d_src, length, xbins, ybins);
	CUT_CHECK_ERROR("gpuCombineHist2DSrcData() execution failed\n");
	TIMER_PRINT("gpuCombineHist2DSrcData", length);

	cudaHist_Approx(d_src, d_hist, length, bins, p_options, true);			//No need to initialize d_hist, will be done by cudaHist
	
	if (!device)
	{
		TIMER_START;
		CUDA_SAFE_CALL(cudaMemcpy(hist, d_hist, bins * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(d_src1));
		CUDA_SAFE_CALL(cudaFree(d_src2));
		CUDA_SAFE_CALL(cudaFree(d_hist));
		TIMER_PRINT("Storing data", 0);
	}
	CUDA_SAFE_CALL(cudaFree(d_src));

	TIMER_DELETE;
}

