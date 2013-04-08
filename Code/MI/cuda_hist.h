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
 
 Relevant publicationB(s):
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

/** 
	\file 
		cuda_hist.h
    \brief 
		Contains definition of C functions for 1D and 2D histogram calculation implemented 
		on the GPU.
    
	Contains definition of C functions for 1D and 2D histogram calculation implemented on 
	the GPU. The methods are based on the following two publications:<br/>
	R. Shams and R. A. Kennedy, "Efficient histogram algorithms for NVIDIA CUDA compatible devices," 
	Proc. Int. Conf. on Signal Processing and Communications Systems (ICSPCS), Gold Coast, Australia, 
	Dec. 2007, pp. 418-422.<br/>
	R. Shams and N. Barnes, "Speeding up mutual information computation using NVIDIA CUDA hardware," 
	Proc. Digital Image Computing: Techniques and Applications (DICTA), Adelaide, Australia, 
	Dec. 2007, pp. 555-560.
*/
#ifndef _CUDA_HIST_H_
#define _CUDA_HIST_H_

/**
	\brief
		The structure defines execution options and is used by histogram functions.
	
	\param threads
		Number of threads per block, must be a multiple of the WARP_SIZE.
	\param blocks
		Number of blocks over which the data is to be distributed, the method 
		creates as many partial histograms.
*/
struct cudaHistOptions
{
	int threads, blocks;
};

/**
	\brief
		Calculates a 1D histogram based on the first method described in:<br>
		R. Shams and R. A. Kennedy, "Efficient histogram algorithms for NVIDIA CUDA compatible devices," 
		Proc. Int. Conf. on Signal Processing and Communications Systems (ICSPCS), Gold Coast, Australia, 
		Dec. 2007, pp. 418-422. 

	\param src
		Pointer to the input array where the data is stored. The input values 
		must be between 0 and 1.
	\param hist
		Pointer to the output array where the computed histogram is to be stored.
		Output array must be allocated and freed by the caller.
	\param length
		Number of the input array's elements.
	\param bins
		Number of the histogram bins or output array's elements.
	\param p_options
		A structure which defines the execution configuration.
	\param device
		A flag which indicates whether input/output arrays are allocated on the
		host (CPU) memory or the device (GPU) memory.

	\remark
        When the function is being called inside a loop or multiple times, for
        best performance allocate the input and output arrays on device memory to
        avoid unnecessary allocation, memory transfers and deallocation by the
        function.
	\remark
		When \a device flag is set, the caller needs to allocate memory for \a 
		hist but does not need to initialize the array. The initialization will 
		be done by the function itself.
	\remark
		Input data must be normalized between 0 and 1. The behavior of the function
		for values outside this range is undefined and is most likely to cause 
		memory corruption.
	
	\return
		The execution time in milliseconds excluding any time spent in allocating
		input data from host to global memory and storing the results back to the 
		host memory. The time spent in creating and initializing any internal objects 
		is considered.

    C wrapper function that calculates a histogram with any number of bins using
    a method for synchronizing updates to histogram memory by simulating atomic
    operations in the GPU's shared memory. The function provides superior performance
    compared to \a cudaHistb, if the input data has a uniform or normal distribution
    and performs worse that \a cudaHistb if the distribution is degenerate (i.e. 
    all elements or most of them are the same). Refer to the paper for more information.
*/
extern "C" double cudaHista(float *src, float *hist, int length, int bins, cudaHistOptions *p_options = NULL, bool device = false);

/**
	\brief
		Calculates a 1D histogram based on the second method described in:<br>
		R. Shams and R. A. Kennedy, "Efficient histogram algorithms for NVIDIA CUDA compatible devices," 
		Proc. Int. Conf. on Signal Processing and Communications Systems (ICSPCS), Gold Coast, Australia, 
		Dec. 2007, pp. 418-422. 

	\param src
		Pointer to the input array where the data is stored. The input values 
		must be between 0 and 1.
	\param hist
		Pointer to the output array where the computed histogram is to be stored.
		Output array must be allocated and freed by the caller.
	\param length
		Number of the input array's elements.
	\param bins
		Number of the histogram bins or output array's elements.
	\param p_options
		A structure which defines the execution configuration.
	\param device
		A flag which indicates whether input/output arrays are allocated on the
		host (CPU) memory or the device (GPU) memory.

	\remark
        When the function is being called inside a loop or multiple times, for
        best performance allocate the input and output arrays on device memory to
        avoid unnecessary allocation, memory transfers and deallocation by the
        function.
	\remark
		When \a device flag is set, the caller needs to allocate memory for \a 
		hist but does not need to initialize the array. The initialization will 
		be done by the function itself.
	\remark
		Input data must be normalized between 0 and 1. The behavior of the function
		for values outside this range is undefined and is most likely to cause 
		memory corruption.
	
	\return
		The execution time in milliseconds excluding any time spent in allocating
		input data from host to global memory and storing the results back to the 
		host memory. The time spent in creating and initializing any internal objects 
		is considered.

    C wrapper function that calculates a histogram with any number of bins using
    a method for separating updates to histogram memory into multiple partial
    histograms. The function has a lower performance compared to \a cudaHista unless
    the input data distribution is degenerate or close to degenerate (i.e. many
    inputs are the same or fall within the same bin). Refer to the paper for more
    information. 
*/
extern "C" double cudaHistb(float *src, float *hist, int length, int bins, cudaHistOptions *p_options = NULL,  bool device = false);
extern "C" double cudaHistc(float *src, float *hist, int length, int bins, cudaHistOptions *p_options = NULL,  bool device = false);

/**
	\brief
		Calculates a 1D histogram based on method described in:<br>
		R. Shams and N. Barnes, "Speeding up mutual information computation using NVIDIA CUDA hardware," 
		Proc. Digital Image Computing: Techniques and Applications (DICTA), Adelaide, Australia, 
		Dec. 2007, pp. 555-560. 

	\param src
		Pointer to the input array where the data is stored. The input values 
		must be between 0 and 1.
	\param hist
		Pointer to the output array where the computed histogram is to be stored.
		Output array must be allocated and freed by the caller.
	\param length
		Number of the input array's elements.
	\param bins
		Number of the histogram bins or output array's elements.
	\param p_options
		A structure which defines the execution configuration.
	\param device
		A flag which indicates whether input/output arrays are allocated on the
		host (CPU) memory or the device (GPU) memory.

	\remark
        When the function is being called inside a loop or multiple times, for
        best performance allocate the input and output arrays on device memory to
        avoid unnecessary allocation, memory transfers and deallocation by the
        function.
	\remark
		When \a device flag is set, the caller needs to allocate memory for \a 
		hist but does not need to initialize the array. The initialization will 
		be done by the function itself.
	\remark
		Input data must be normalized between 0 and 1. The behavior of the function
		for values outside this range is undefined and is most likely to cause 
		memory corruption.
	
	\return
		The execution time in milliseconds excluding any time spent in allocating
		input data from host to global memory and storing the results back to the 
		host memory. The time spent in creating and initializing any internal objects 
		is considered.

	\see
		cudaHista, cudaHistb

    As the name suggestes an approximation to the histogram is calculated by this
	function. The ratio of the histogram bins are approximated by the method not
	the actual number of elements in each bin. As such, the method is an approximation
	of the probability mass function (pmf) of the input data. If you are interested
	in the exact histogram information you should use \a cudaHista or \a cudaHistb.
	Refer to the paper for more information.
*/
extern "C" double cudaHist_Approx(float *src, float *hist, int length, int bins, cudaHistOptions *p_options = NULL,  bool device = false);

/**
	\brief
		Calculates a 2D (joint) histogram based on the first method described in:<br>
		R. Shams and R. A. Kennedy, "Efficient histogram algorithms for NVIDIA CUDA compatible devices," 
		Proc. Int. Conf. on Signal Processing and Communications Systems (ICSPCS), Gold Coast, Australia, 
		Dec. 2007, pp. 418-422. 

	\param src1
		Pointer to the input array where the data is stored.
	\param src2
		Pointer to the input array where the data is stored.
	\param hist
		Pointer to the output array where the computed histogram is to be stored.
		Output array must be allocated and freed by the caller.
	\param length
		Number of the input array's elements. Both input arrays must have the 
		same length.
	\param xbins
		Number of the histogram bins along the x-axis of the 2D histogram.
	\param ybins
		Number of the histogram bins along the y-axis of the 2D histogram.
	\param p_options
		A structure which defines the execution configuration.
	\param device
		A flag which indicates whether input/output arrays are allocated on the
		host (CPU) memory or the device (GPU) memory.

	\remark
		Integrating the 2D histrogram along the x-axis gives the 1D histogram
		of \a src1 and along the y-axis gives the 1D histogram of \a src2.
	\remark
		Size of \a hist, allocated by the caller, must be sizeof(float)*xbins*ybins.
	\remark
        When the function is being called inside a loop or multiple times, for
        best performance allocate the input and output arrays on device memory to
        avoid unnecessary allocation, memory transfers and deallocation by the
        function.
	\remark
		When \a device flag is set, the caller needs to allocate memory for \a 
		hist but does not need to initialize the array. The initialization will 
		be done by the function itself.
	
    C wrapper function that calculates a 2D histogram with any number of bins based
	on \a cudaHista.

	\see
		cudaHista
*/
extern "C" void cudaHist2Da(float *src1, float *src2, float *hist, int length, int xbins, int ybins, cudaHistOptions *p_options = NULL, bool device = false);

/**
	\brief
		Calculates a 2D (joint) histogram based on the second method described in:<br>
		R. Shams and R. A. Kennedy, "Efficient histogram algorithms for NVIDIA CUDA compatible devices," 
		Proc. Int. Conf. on Signal Processing and Communications Systems (ICSPCS), Gold Coast, Australia, 
		Dec. 2007, pp. 418-422. 

	\param src1
		Pointer to the input array where the data is stored.
	\param src2
		Pointer to the input array where the data is stored.
	\param hist
		Pointer to the output array where the computed histogram is to be stored.
		Output array must be allocated and freed by the caller.
	\param length
		Number of the input array's elements. Both input arrays must have the 
		same length.
	\param xbins
		Number of the histogram bins along the x-axis of the 2D histogram.
	\param ybins
		Number of the histogram bins along the y-axis of the 2D histogram.
	\param p_options
		A structure which defines the execution configuration.
	\param device
		A flag which indicates whether input/output arrays are allocated on the
		host (CPU) memory or the device (GPU) memory.

	\remark
		Integrating the 2D histrogram along the x-axis gives the 1D histogram
		of \a src1 and along the y-axis gives the 1D histogram of \a src2.
	\remark
		Size of \a hist, allocated by the caller, must be sizeof(float)*xbins*ybins.
	\remark
        When the function is being called inside a loop or multiple times, for
        best performance allocate the input and output arrays on device memory to
        avoid unnecessary allocation, memory transfers and deallocation by the
        function.
	\remark
		When \a device flag is set, the caller needs to allocate memory for \a 
		hist but does not need to initialize the array. The initialization will 
		be done by the function itself.
	\remark
		Input data must be normalized between 0 and 1. The behavior of the function
		for values outside this range is undefined and is most likely to cause 
		memory corruption.
	
    C wrapper function that calculates a 2D histogram with any number of bins based
	on \a cudaHistb.

	\see
		cudaHistb
*/
extern "C" void cudaHist2Db(float *src1, float *src2, float *hist, int length, int xbins, int ybins, cudaHistOptions *p_options = NULL, bool device = false);

/**
	\brief
		Calculates a 2D (joint) histogram based on the second method described in:<br>
		R. Shams and N. Barnes, "Speeding up mutual information computation using NVIDIA CUDA hardware," 
		Proc. Digital Image Computing: Techniques and Applications (DICTA), Adelaide, Australia, 
		Dec. 2007, pp. 555-560. 

	\param src1
		Pointer to the input array where the data is stored.
	\param src2
		Pointer to the input array where the data is stored.
	\param hist
		Pointer to the output array where the computed histogram is to be stored.
		Output array must be allocated and freed by the caller.
	\param length
		Number of the input array's elements.
	\param bins
		Number of the histogram bins or output array's elements.
	\param p_options
		A structure which defines the execution configuration.
	\param device
		A flag which indicates whether input/output arrays are allocated on the
		host (CPU) memory or the device (GPU) memory.

	\remark
		Integrating the 2D histrogram along the x-axis gives the 1D histogram
		of \a src1 and along the y-axis gives the 1D histogram of \a src2.
	\remark
        When the function is being called inside a loop or multiple times, for
        best performance allocate the input and output arrays on device memory to
        avoid unnecessary allocation, memory transfers and deallocation by the
        function.
	\remark
		When \a device flag is set, the caller needs to allocate memory for \a 
		hist but does not need to initialize the array. The initialization will 
		be done by the function itself.
	\remark
		Input data must be normalized between 0 and 1. The behavior of the function
		for values outside this range is undefined and is most likely to cause 
		memory corruption.
	
	\return
		The execution time in milliseconds excluding any time spent in allocating
		input data from host to global memory and storing the results back to the 
		host memory. The time spent in creating and initializing any internal objects 
		is considered.

	\see
		cudaHist_Approx

    As the name suggestes an approximation to the histogram is calculated by this
	function. The ratio of the histogram bins are approximated by the method not
	the actual number of elements in each bin. As such, the method is an approximation
	of the probability mass function (pmf) of the input data. If you are interested
	in the exact histogram information you should use \a cudaHist2Da or \a cudaHist2Db.
	Refer to the paper for more information.
*/
extern "C" void cudaHist2D_Approx(float *src1, float *src2, float *hist, int length, int xbins, int ybins, cudaHistOptions *p_options = NULL, bool device = false);

#endif
