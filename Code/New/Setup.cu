#include "Setup.h"

void checkForCUDA(void) {
	int nDevices;
	unsigned long maxMem = 0;
	unsigned long mem;
	int devIdx = 0;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&nDevices);

	if(nDevices == 0){
		std::cerr << "No CUDA capable device found\n";
		return;
	}
	else if(nDevices != 1){
		std::cout << "  " << nDevices << " cuda Devices found, selecting device with largest total memory\n";

			for (int i = 0; i < nDevices; i++) {
		
			cudaGetDeviceProperties(&prop, i);

			mem = (unsigned long)prop.totalGlobalMem;

			if(mem >= maxMem){
				devIdx = i;
				mem = maxMem;
			}
		}

		std::cout << "  Device " << devIdx << "selected\n";
	}
	else{
		std::cout << "One cuda Device found\n";
		devIdx = 0;
	}
	
	cudaGetDeviceProperties(&prop, devIdx);

	cudaSetDevice(devIdx);  	
	CudaCheckError();

	std::cout << "  Device name: " << prop.name << "\n";
	std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << "\n";
	std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << "\n";
	std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6) << "\n";
	std::cout << "  Total Global Memory (MB): " << prop.totalGlobalMem/(1.0e6) << "\n";
}