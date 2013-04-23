#include "Metric.h"
#include "Reduce\reduction.h"

extern "C" float cudaMIa(float* src1, float* src2, int length, int xbins, int ybins, struct cudaHistOptions* p_options, int device, int incZeros);

float Metric::EvalMetric(SparseScan* A, SparseScan* B){
	return 0;
}

float MI::EvalMetric(SparseScan* A, SparseScan* B){
	
	//move scans to gpu if required
	/*if(A->getPoints()->IsOnGpu()){
		A->getPoints()->AllocateGpu();
		A->getPoints()->CpuToGpu();
	}

	if(B->getPoints()->IsOnGpu()){
		B->getPoints()->AllocateGpu();
		B->getPoints()->CpuToGpu();
	}*/

	size_t numElements;
	//check scans of same size
	if(A->getPoints()->GetNumEntries() != B->getPoints()->GetNumEntries()){
		numElements = (A->getPoints()->GetNumEntries() > B->getPoints()->GetNumEntries()) ? B->getPoints()->GetNumEntries() : A->getPoints()->GetNumEntries();
		TRACE_WARNING("Number of entries does not match, Scan A has %i, Scan B has %i, only using %i entries",A->getPoints()->GetNumEntries(),B->getPoints()->GetNumEntries(),numElements);
	}
	else{
		numElements = A->getPoints()->GetNumEntries();
	}

	struct cudaHistOptions *p_opt = 0;
	float miOut = cudaMIa((float*)A->getPoints()->GetGpuPointer(), (float*)B->getPoints()->GetGpuPointer(), numElements, MI_BINS, MI_BINS, p_opt, 1, true);

	return miOut;
}

float GOM::EvalMetric(SparseScan* A, SparseScan* B){
	
	//move scans to gpu if required
	/*if(A->getPoints()->IsOnGpu()){
		A->getPoints()->AllocateGpu();
		A->getPoints()->CpuToGpu();
	}

	if(B->getPoints()->IsOnGpu()){
		B->getPoints()->AllocateGpu();
		B->getPoints()->CpuToGpu();
	}*/

	if(A->getNumCh() != GOM_DEPTH){
		TRACE_ERROR("GOM requires two channels (mag, phase) to operate and Scan A has %i", A->getNumCh());
		return 0;
	}
	if(B->getNumCh() != GOM_DEPTH){
		TRACE_ERROR("GOM requires two channels (mag, phase) to operate and Scan B has %i", B->getNumCh());
		return 0;
	}

	size_t numElements;
	//check scans of same size
	if(A->getNumPoints() != B->getNumPoints()){
		numElements = (A->getNumPoints() > B->getNumPoints()) ? B->getNumPoints() : A->getNumPoints();
		TRACE_WARNING("Number of entries does not match, Scan A has %i, Scan B has %i, only using %i entries",A->getNumPoints(),B->getNumPoints(),numElements);
	}
	else{
		numElements = A->getNumPoints();
	}

	float* phaseOut;
	float* magOut;
	CudaSafeCall(cudaMalloc(&phaseOut, sizeof(float)*numElements));
	CudaSafeCall(cudaMalloc(&magOut, sizeof(float)*numElements));
    GOMKernel<<<gridSize(numElements), BLOCK_SIZE>>>
		((float*)A->getPoints()->GetGpuPointer(), (float*)B->getPoints()->GetGpuPointer(), numElements, phaseOut, magOut);
	CudaCheckError();

	//perform reduction

	int numThreads = 256;
	int numBlocks = 64;
	
	float* reduceOut;
	CudaSafeCall(cudaMalloc((void **) &reduceOut, numBlocks*sizeof(float)));
	
	reduce<float>(numElements, numThreads, numBlocks, 6, phaseOut, reduceOut);
	float phaseRes;
	CudaSafeCall(cudaMemcpy(&phaseRes,reduceOut,sizeof(float),cudaMemcpyDeviceToHost));

	reduce<float>(numElements, numThreads, numBlocks, 6, magOut, reduceOut);
	float magRes;
	CudaSafeCall(cudaMemcpy(&magRes,reduceOut,sizeof(float),cudaMemcpyDeviceToHost));
	
	/*float* reduceOut;
	float phaseRes = 0;
	float magRes = 0;
	reduceOut = new float[numElements];

	CudaSafeCall(cudaMemcpy(reduceOut,phaseOut,numElements*sizeof(float),cudaMemcpyDeviceToHost));
	for(size_t i = 0; i < numElements; i++){
		phaseRes += reduceOut[i];
	}

	CudaSafeCall(cudaMemcpy(reduceOut,magOut,numElements*sizeof(float),cudaMemcpyDeviceToHost));
	for(size_t i = 0; i < numElements; i++){
		magRes += reduceOut[i];
	}*/

	delete reduceOut;
	CudaSafeCall(cudaFree(phaseOut));
	CudaSafeCall(cudaFree(magOut));

	return (phaseRes / magRes);
}

float LIV::EvalMetric(SparseScan* A, SparseScan* B){
	TRACE_ERROR("Not yet implemented");
	return 0;
}
