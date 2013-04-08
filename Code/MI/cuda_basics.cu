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

// includes, kernels
#include "gpu_basics.cu"

#ifdef MATLAB
extern "C" int mexfPrintf(FILE * _File, const char * fmt, ...)
{
	va_list arg;
	va_start(arg, fmt);

	char s[4096];				//I am hoping that the output of vsprintf is not going to exceed this limit
	vsprintf(s, fmt, arg);
	va_end(arg);

	return mexPrintf("%s", s);
}
#endif


//Round a / b to the nearest higher integer value
extern "C" int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a to the nearest multiple of b
extern "C" int iRoundUp(int a, int b)
{
	return iDivUp(a, b) * b;
}

extern "C" void cudaMallocWrapper(void** devPtr, size_t count)
{
	CUDA_SAFE_CALL(cudaMalloc(devPtr, count));
}

extern "C" void cudaFreeWrapper(void *devPtr)
{
	CUDA_SAFE_CALL(cudaFree(devPtr));
}

extern "C" void cudaMemcpyHostToDeviceWrapper(void *dst, const void *src, size_t size)
{
	CUDA_SAFE_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

extern "C" int cudaGetDeviceCountWrapper(void)
{
	int count;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&count));

	return count;
}

extern "C" void cudaGetDevicePropertiesWrapper(cudaDeviceProp *prop, int dev)
{
	CUDA_SAFE_CALL(cudaGetDeviceProperties(prop, dev));
}

/*
	'width' is limted to 2^16 and 'height' to 2^15. Exceeding these limits 
	throws a cuda error with the following message: 'invalid parameter'
*/
extern "C" void cudaMallocArrayWrapper(void** devPtr, size_t width, size_t height)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	CUDA_SAFE_CALL(cudaMallocArray((cudaArray **) devPtr, &channelDesc, width, height)); 
}

extern "C" void cudaMemcpyHostToArrayWrapper(void *dst, const void *src, size_t size)
{
	CUDA_SAFE_CALL(cudaMemcpyToArray((cudaArray *) dst, 0, 0, src, size, cudaMemcpyHostToDevice));
}

extern "C" void cudaFreeArrayWrapper(void *devPtr)
{
	CUDA_SAFE_CALL(cudaFreeArray((cudaArray *) devPtr));
}

/*
	d_mem must point to device memory
*/
extern "C" void cudaZeroMem(float *d_mem, int length)
{
	dim3 grid, block;
	TIMER_CREATE;

	const int max_threads = MAX_THREADS;
	int good_len = iRoundUp(length, WARP_SIZE);

	block.x = max_threads; block.y = 1; block.z = 1;
	int blocks = iDivUp(good_len, max_threads);
	if (blocks > MAX_BLOCKS_PER_DIM)
	{
		grid.x = ceil(sqrtf(blocks)); grid.y = grid.x; grid.z = 1;
	}
	else
	{
		grid.x = blocks; grid.y = 1; grid.z = 1;
	}

	TIMER_START;
	gpuZeroMem<<<grid, block>>>(d_mem, length);
	CUT_CHECK_ERROR("gpuZeroMem() execution failed\n");
	TIMER_PRINT("gpuZeroMem", length)
	TIMER_DELETE;
}

/*
	src:
		An MxN source matrix.
	dst:
		An Mx1 destination matrix; must be allocated by the caller.
	xdim:
		N
	ydim:
		M
	device:
		If set, the function assumes src and dst are given in device memory
*/
extern "C" void cudaSumAlongRows(float *src, float *dst, int xdim, int ydim, bool device /*= false*/)
{
	float *d_src, *d_dst;			//src and dst in device memory
	float *d_dst_tmp;
	dim3 grid, block;
	TIMER_CREATE;

	if (!device)
	{
		TIMER_START;
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, sizeof(float) * xdim * ydim));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_dst, sizeof(float) * ydim));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, sizeof(float) * xdim * ydim, cudaMemcpyHostToDevice));
		TIMER_PRINT("Loading data", 0);
	}
	else
	{
		d_src = src; 
		d_dst = dst;
	}

	const int cutoff = min(64, MAX_THREADS);			//For EmuDebug
	int dst_ofs = 0, src_ofs = 0;
	int remaining = xdim ;
	int data_per_block = 2 * MAX_THREADS ;
	int dst_width = 0;
	while (remaining > 0) 
	{
		dst_width += remaining / data_per_block;
		remaining = remaining - remaining / data_per_block * data_per_block;
		data_per_block >>= 1;
		if (remaining < cutoff)
			data_per_block = remaining;
	}
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_dst_tmp, sizeof(float) * ydim * dst_width));

	TIMER_START;
	remaining = xdim;
	data_per_block = 2 * MAX_THREADS;
	do
	{
		grid.x = remaining / data_per_block ; grid.y = ydim; grid.z = 1;
		block.x = MAX_THREADS ; block.y = 1; block.z = 1;

		if (grid.x > 0)
		{
			gpuSumAlongRows<<<grid, block>>>(d_src + src_ofs, d_dst_tmp + dst_ofs, data_per_block, xdim, dst_width);
			CUT_CHECK_ERROR("gpuSumAlongRows() execution failed\n");
		}

		src_ofs += data_per_block * grid.x;
		dst_ofs += grid.x;

		remaining = remaining - grid.x * data_per_block;
		data_per_block >>= 1;
		if (remaining < cutoff)
			data_per_block = remaining;
	} while (remaining > 0);
	TIMER_PRINT("gpuSumAlongRows", xdim * ydim);

	if (dst_ofs > 1)		//recursive call
		cudaSumAlongRows(d_dst_tmp, d_dst, dst_ofs, ydim, true);
	else
		CUDA_SAFE_CALL(cudaMemcpy2D(d_dst, 1 * sizeof(float), d_dst_tmp, dst_width * sizeof(float), dst_width * sizeof(float), ydim, cudaMemcpyDeviceToDevice));

	CUDA_SAFE_CALL(cudaFree(d_dst_tmp));
	if (!device)
	{
		TIMER_START;
		//Copy dst data from device memory
		CUDA_SAFE_CALL(cudaMemcpy(dst, d_dst, sizeof(float) * ydim, cudaMemcpyDeviceToHost));
		
		//Free memory
		CUDA_SAFE_CALL(cudaFree(d_src));
		CUDA_SAFE_CALL(cudaFree(d_dst));
		TIMER_PRINT("Storing data", 0);
	}

	TIMER_DELETE;
}

/*
	src:
		An MxN source matrix.
	dst:
		A 1xN destination matrix; must be allocated by the caller.
	xdim:
		N
	ydim:
		M
	device:
		If set, the function assumes src and dst are given in device memory
*/
extern "C" void cudaSumAlongCols(float *src, float *dst, int xdim, int ydim, bool device /*= false*/)
{
	float *d_src, *d_dst;			//src and dst in device memory
	float *d_dst_tmp;
	dim3 grid, block;
	TIMER_CREATE;

	if (!device)
	{
		TIMER_START;
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, sizeof(float) * xdim * ydim));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_dst, sizeof(float) * xdim));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, sizeof(float) * xdim * ydim, cudaMemcpyHostToDevice));
		TIMER_PRINT("Loading data", 0);
	}
	else
	{
		d_src = src; 
		d_dst = dst;
	}

	const int cutoff = min(64, MAX_THREADS);			//For EmuDebug
	int dst_ofs = 0, src_ofs = 0;
	int remaining = ydim ;
	int data_per_block = 2 * MAX_THREADS ;
	int dst_height = 0;
	while (remaining > 0) 
	{
		dst_height += remaining / data_per_block;
		remaining = remaining - remaining / data_per_block * data_per_block;
		data_per_block >>= 1;
		if (remaining < cutoff)
			data_per_block = remaining;
	}
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_dst_tmp, sizeof(float) * xdim * dst_height));

	TIMER_START;
	remaining = ydim;
	data_per_block = 2 * MAX_THREADS;
	int num_rows = 0;
	do
	{
		grid.x = xdim; grid.y = remaining / data_per_block; grid.z = 1;
		block.x = 1; block.y = MAX_THREADS; block.z = 1;

		if (grid.y > 0)
		{
			gpuSumAlongCols<<<grid, block>>>(d_src + src_ofs, d_dst_tmp + dst_ofs, data_per_block, xdim, xdim);
			CUT_CHECK_ERROR("gpuSumAlongCols() execution failed\n");
		}

		src_ofs += data_per_block * grid.y * xdim;
		dst_ofs += grid.y * xdim;
		num_rows += grid.y;

		remaining = remaining - grid.y * data_per_block;
		data_per_block >>= 1;
		if (remaining < cutoff)
			data_per_block = remaining;
	} while (remaining > 0);
	TIMER_PRINT("gpuSumAlongCols", xdim * ydim);

	if (num_rows > 1)		//recursive call
		cudaSumAlongCols(d_dst_tmp, d_dst, xdim, num_rows, true);
	else
		CUDA_SAFE_CALL(cudaMemcpy(d_dst, d_dst_tmp, xdim * sizeof(float), cudaMemcpyDeviceToDevice));

	CUDA_SAFE_CALL(cudaFree(d_dst_tmp));
	if (!device)
	{
		TIMER_START;
		//Copy dst data from device memory
		CUDA_SAFE_CALL(cudaMemcpy(dst, d_dst, sizeof(float) * xdim, cudaMemcpyDeviceToHost));
		
		//Free memory
		CUDA_SAFE_CALL(cudaFree(d_src));
		CUDA_SAFE_CALL(cudaFree(d_dst));
		TIMER_PRINT("Storing data", 0);
	}

	TIMER_DELETE;
}

//xyzReduction methods can be used for debugging and testing
extern "C" float cudaReduction(float *src, int length, bool device /*= false*/)
{
	dim3 grid, block;
	int size = length * sizeof(float);
	//Device memory pointers
	float *d_src, *d_dst1, *d_dst2, *d_dst;
	float res;
	TIMER_CREATE;

	const int max_threads = MAX_THREADS;
	int good_len = iRoundUp(length, WARP_SIZE);

	block.x = max_threads; block.y = 1; block.z = 1;
	//We can process up to 2 * max_threads in each round
	grid.x = ceil(sqrtf(iDivUp(good_len, 2 * max_threads))); grid.y = grid.x; grid.z = 1;

	TIMER_START;
	if (!device)
	{
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));
	}
	else
		d_src = src;

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_dst1, grid.x * grid.y * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_dst2, grid.x * grid.y * sizeof(float)));
	TIMER_PRINT("Loading data", 0);

	TIMER_START;
	float *d_tmp = d_src;
	int count = 0;
	int len = length;
	do
	{
		d_dst = count % 2 ? d_dst = d_dst2 : d_dst = d_dst1;

		gpuReduction<<<grid, block>>>(d_tmp, d_dst, len);
		CUT_CHECK_ERROR("gpuSum() execution failed\n");

		d_tmp = d_dst;
		count++;
		len = grid.x * grid.y;
		good_len = iRoundUp(len, WARP_SIZE);
		grid.x = ceil(sqrtf(iDivUp(good_len, 2 * max_threads))); grid.y = grid.x; grid.z = 1;
	}while (len != 1);
	TIMER_PRINT("gpuSum", length);

	TIMER_START;
	if (!device)
		CUDA_SAFE_CALL(cudaFree(d_src));
	CUDA_SAFE_CALL(cudaMemcpy(&res, d_dst, sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_dst1));
	CUDA_SAFE_CALL(cudaFree(d_dst2));
	TIMER_PRINT("Storing data", 0);
	TIMER_DELETE;

	return res;
}

CUDA_REDUCTION(Sum);
CUDA_REDUCTION(Mul);
CUDA_REDUCTION(Max);
CUDA_REDUCTION(Min);
CUDA_BINARY(SSDBinary)
CUDA_BINARY(SADBinary)
