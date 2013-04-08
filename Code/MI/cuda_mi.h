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

/** 
	\file 
		cuda_mi.h
    \brief 
		Contains definition of C functions for mutual infomration (MI) computation
		implemented on the GPU.
    
	Contains definition of C functions for mutual information (MI) computation on  
	the GPU. The methods are based on the following two publications:<br/>
	R. Shams and R. A. Kennedy, "Efficient histogram algorithms for NVIDIA CUDA compatible devices," 
	Proc. Int. Conf. on Signal Processing and Communications Systems (ICSPCS), Gold Coast, Australia, 
	Dec. 2007, pp. 418-422.<br/>
	R. Shams and N. Barnes, "Speeding up mutual information computation using NVIDIA CUDA hardware," 
	Proc. Digital Image Computing: Techniques and Applications (DICTA), Adelaide, Australia, 
	Dec. 2007, pp. 555-560.
*/

#ifndef _CUDA_MI_H_
#define _CUDA_MI_H_

/**
	\brief
		Computes the mutual information between two equally sized input arrays 
		based on the first histogram method described in:<br>
		R. Shams and R. A. Kennedy, "Efficient histogram algorithms for NVIDIA CUDA compatible devices," 
		Proc. Int. Conf. on Signal Processing and Communications Systems (ICSPCS), Gold Coast, Australia, 
		Dec. 2007, pp. 418-422. 

	\param src1
		Pointer to the first input array where the data is stored. The input values 
		must be between 0 and 1.
	\param src2
		Pointer to the second input array where the data is stored. The input values 
		must be between 0 and 1.
	\param length
		Number of the input arrays' elements.
	\param xbins
		Number of bins used in computation of the histogram for \a src1.
	\param ybins
		Number of bins used in computation of the histogram for \a src2.
	\param time[out]
		The execution time in milliseconds excluding any time spent in allocating
		input data from host to global memory and storing the results back to the 
		host memory. The time spent in creating and initializing any internal objects 
		is considered.
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

	\see
		cudaHist2Da, cudaMIb, cudaMI_Approx
	
    C wrapper function that computes the mutual information (MI) of two similarly sized
    arrays. MI computation is based on approximating the joint and marginal probability
    mass function (pmf) of input data from a joint histogram with specified number of
    bins for each dimension. This routine uses \a cudaHist2Da histogram function for this
    purpose. Refer to the paper for more information.
*/
extern "C" float cudaMIa(float *src1, float *src2, int length, int xbins, int ybins, cudaHistOptions *p_options = NULL, bool device = false, bool incZeros = false);

/**
	\brief
		Computes the mutual information between two equally sized input arrays 
		based on the second histogram method described in:<br>
		R. Shams and R. A. Kennedy, "Efficient histogram algorithms for NVIDIA CUDA compatible devices," 
		Proc. Int. Conf. on Signal Processing and Communications Systems (ICSPCS), Gold Coast, Australia, 
		Dec. 2007, pp. 418-422. 

	\param src1
		Pointer to the first input array where the data is stored. The input values 
		must be between 0 and 1.
	\param src2
		Pointer to the second input array where the data is stored. The input values 
		must be between 0 and 1.
	\param length
		Number of the input arrays' elements.
	\param xbins
		Number of bins used in computation of the histogram for \a src1.
	\param ybins
		Number of bins used in computation of the histogram for \a src2.
	\param time[out]
		The execution time in milliseconds excluding any time spent in allocating
		input data from host to global memory and storing the results back to the 
		host memory. The time spent in creating and initializing any internal objects 
		is considered.
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

	\see
		cudaHist2Db, cudaMIa, cudaMI_Approx
	
    C wrapper function that computes the mutual information (MI) of two similarly sized
    arrays. MI computation is based on approximating the joint and marginal probability
    mass function (pmf) of input data from a joint histogram with specified number of
    bins for each dimension. This routine uses \a cudaHist2Db histogram function for this
    purpose. Refer to the paper for more information.
*/
extern "C" float cudaMIb(float *src1, float *src2, int length, int xbins, int ybins, cudaHistOptions *p_options = NULL, bool device = false);

/**
	\brief
		Computes the mutual information between two equally sized input arrays 
		based on the approximate histogram method described in:<br>
		R. Shams and N. Barnes, "Speeding up mutual information computation using NVIDIA CUDA hardware," 
		Proc. Digital Image Computing: Techniques and Applications (DICTA), Adelaide, Australia, 
		Dec. 2007, pp. 555-560. 

	\param src1
		Pointer to the first input array where the data is stored. The input values 
		must be between 0 and 1.
	\param src2
		Pointer to the second input array where the data is stored. The input values 
		must be between 0 and 1.
	\param length
		Number of the input arrays' elements.
	\param xbins
		Number of bins used in computation of the histogram for \a src1.
	\param ybins
		Number of bins used in computation of the histogram for \a src2.
	\param time[out]
		The execution time in milliseconds excluding any time spent in allocating
		input data from host to global memory and storing the results back to the 
		host memory. The time spent in creating and initializing any internal objects 
		is considered.
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

	\see
		cudaHist2Da, cudaMIb, cudaMI_Approx
	
    C wrapper function that computes the mutual information (MI) of two similarly sized
    arrays. MI computation is based on approximating the joint and marginal probability
    mass function (pmf) of input data from a joint histogram with specified number of
    bins for each dimension. This routine uses \a cudaHist2D_Approx histogram function 
	for this purpose. Refer to the paper for more information.
*/
extern "C" float cudaMI_Approx(float *src1, float *src2, int length, int xbins, int ybins, cudaHistOptions *p_options = NULL, bool device = false);

extern "C" float cudaEntropy(float *src, int length, bool device = false);
extern "C" void cudaEntropyUnary(float *src, float *dst, int length, bool device = false);

#endif
