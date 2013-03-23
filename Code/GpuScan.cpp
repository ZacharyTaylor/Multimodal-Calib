#include "Scan.cpp"
#include "common.h"

class GpuScan: public Scan {
public:
	float* d_points_;

	void pointsAllocateGpu(void){
		cudaMalloc((void**)&(d_points_), getNumPoints());
	}

	void pointsClearGpu(void){
		cudaFree(d_points_);
	}

	void pointsGpuToCpu(void){
		cudaMemcpy(d_points_, points_, getNumPoints(), cudaMemcpyHostToDevice);
	}

	void pointsCpuToGpu(void){
		cudaMemcpy(points_, d_points_, getNumPoints(), cudaMemcpyDeviceToHost);
	}
};

class GpuSparseScan: public SparseScan {
public:
	float* d_location_;

	void locationAllocateGpu(void){
		cudaMalloc((void**)&(d_location_), GpuScan.getNumPoints());
	}

	void locationClearGpu(void){
		cudaFree(d_location_);
	}

	void locationGpuToCpu(void){
		cudaMemcpy(d_location_, location_, GpuScan.getNumPoints(), cudaMemcpyHostToDevice);
	}

	void locationCpuToGpu(void){
		cudaMemcpy(location_, d_location_, GpuScan.getNumPoints(), cudaMemcpyDeviceToHost);
	}
};