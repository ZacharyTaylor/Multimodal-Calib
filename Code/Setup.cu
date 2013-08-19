#include "Setup.h"

void checkForCUDA(void) {
	int nDevices;
	unsigned long maxMem = 0;
	unsigned long mem;
	int devIdx = 0;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&nDevices);

	if(nDevices == 0){
		TRACE_ERROR("No CUDA capable device found");
		return;
	}
	else if(nDevices != 1){
		TRACE_INFO("  %i cuda Devices found, selecting device with largest total memory",nDevices);

			for (int i = 0; i < nDevices; i++) {
		
			cudaGetDeviceProperties(&prop, i);

			mem = (unsigned long)prop.totalGlobalMem;

			if(mem >= maxMem){
				devIdx = i;
				mem = maxMem;
			}
		}

		TRACE_INFO("  Device %i selected",devIdx);
	}
	else{
		TRACE_INFO("  %One cuda Device found",nDevices);
		devIdx = 0;
	}
	
	cudaGetDeviceProperties(&prop, devIdx);

	cudaSetDevice(devIdx);  	
	CudaCheckError();

	TRACE_INFO("  Device name: %s", prop.name);
	TRACE_INFO("  Memory Clock Rate (KHz): %d", prop.memoryClockRate);
	TRACE_INFO("  Memory Bus Width (bits): %d", prop.memoryBusWidth);
	TRACE_INFO("  Peak Memory Bandwidth (GB/s): %f", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	TRACE_INFO("  Total Global Memory (MB): %f\n", prop.totalGlobalMem/(1.0e6));
}