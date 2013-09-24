#include <mex.h>

#include "Setup.h"

void checkForCUDA(void) {
	int nDevices;
	unsigned long maxMem = 0;
	unsigned long mem;
	int devIdx = 0;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&nDevices);

	if(nDevices == 0){
		mexErrMsgTxt("No CUDA capable device found\n");
		return;
	}
	else if(nDevices != 1){
		mexPrintf("  %d cuda Devices found, selecting device with largest total memory\n", nDevices);

			for (int i = 0; i < nDevices; i++) {
		
			cudaGetDeviceProperties(&prop, i);

			mem = (unsigned long)prop.totalGlobalMem;

			if(mem >= maxMem){
				devIdx = i;
				mem = maxMem;
			}
		}

		mexPrintf("  Device %d selected\n",devIdx);
	}
	else{
		mexPrintf("  One cuda Device found\n");
		devIdx = 0;
	}
	
	cudaGetDeviceProperties(&prop, devIdx);

	cudaSetDevice(devIdx);  	
	CudaCheckError();

	mexPrintf("  Device name: %s\n",prop.name);
	mexPrintf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
	mexPrintf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
	mexPrintf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/(1.0e6));
	mexPrintf("  Total Global Memory (MB): %f\n", prop.totalGlobalMem/(1.0e6));
}