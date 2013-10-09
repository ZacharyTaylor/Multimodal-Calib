#include "Metrics.h"
#include "Kernels.h"
#include "reduction.h"

float Metric::evalMetric(std::vector<float*>& gen, ScanList scan, size_t index, cudaStream_t stream){
	mexErrMsgTxt("No metric has been specified");
	return 0;
}

/*MI::MI(size_t bins):
	bins_(bins){
}

void MI::evalMetric(std::vector<float*> A, std::vector< thrust::device_vector<float>> B, cudaStream_t stream){
	
	//check scans exist
	if(A == NULL || B == NULL){
		TRACE_ERROR("Two scans are required for the metric to operate");
		*value = 0;
		return;
	}

	size_t numElements;
	//check scans of same size
	if(A->getPoints()->GetNumEntries() != B->getPoints()->GetNumEntries()){
		numElements = (A->getPoints()->GetNumEntries() > B->getPoints()->GetNumEntries()) ? B->getPoints()->GetNumEntries() : A->getPoints()->GetNumEntries();
		TRACE_WARNING("Number of entries does not match, Scan A has %i, Scan B has %i, only using %i entries",A->getPoints()->GetNumEntries(),B->getPoints()->GetNumEntries(),numElements);
	}
	else{
		numElements = A->getPoints()->GetNumEntries();
	}

	//float miOut = 0;
	float miOut = miRun((float*)A->getPoints()->GetGpuPointer(), (float*)B->getPoints()->GetGpuPointer(), bins_, numElements, stream);
	//struct cudaHistOptions *p_opt = 0;
	//float miOut = cudaMIa((float*)A->getPoints()->GetGpuPointer(), (float*)B->getPoints()->GetGpuPointer(), numElements, MI_BINS, MI_BINS, p_opt, 1, true);

	*value = miOut;
}*/

SSD::SSD(){};

float SSD::evalMetric(std::vector<float*>& gen, ScanList scan, size_t index, cudaStream_t stream){
	
	if((gen.size() != 1) || (scan.getNumCh(index) != 1)){
		mexErrMsgTxt("SSD metric can only accept a single intensity channel");
	}
	SSDKernel<<<gridSize(scan.getNumPoints(index)), BLOCK_SIZE, 0, stream>>>
		(gen[0], scan.getIP(index,0), scan.getNumPoints(index));
	CudaCheckError();

	//perform reduction
	float temp = reduceEasy(gen[0], scan.getNumPoints(index));
	temp = sqrt(temp);

	return temp;
}

GOM::GOM(){};

float GOM::evalMetric(std::vector<float*>& gen, ScanList scan, size_t index, cudaStream_t stream){
	
	if((gen.size() != 2) || (scan.getNumCh(index) != 2)){
		mexErrMsgTxt("GOM metric can only accept a two intensity channels (mag and angle)");
	}
   
	GOMKernel<<<gridSize(scan.getNumPoints(index)), BLOCK_SIZE, 0, stream>>>
		(gen[0],gen[1],scan.getIP(index,0),scan.getIP(index,1), scan.getNumPoints(index));
	CudaCheckError();
	
	float phase = reduceEasy(gen[0], scan.getNumPoints(index));
	float mag = reduceEasy(gen[1], scan.getNumPoints(index));
	
	float out = (phase / mag);
	
	return out;
}

/*LIV::LIV(float* avImg, size_t width, size_t height){
	avImg_ = new PointsList(avImg, (width*height), true);
}

LIV::~LIV(){
	delete avImg_;
}

void LIV::evalMetric(std::vector<float*> A, std::vector< thrust::device_vector<float>> B, cudaStream_t stream){

	//check scans exist
	if(A == NULL || B == NULL){
		TRACE_ERROR("Two scans are required for the metric to operate");
		*value = 0;
		return;
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

	float* out;
	CudaSafeCall(cudaMalloc(&out, sizeof(float)*numElements));
	
	livValKernel<<<gridSize(numElements), BLOCK_SIZE, 0, *stream>>>
		((float*)A->getPoints()->GetGpuPointer(), (float*)B->getPoints()->GetGpuPointer(), (float*)avImg_->GetGpuPointer(), numElements, out);
	CudaCheckError();

	//perform reduction
	float outVal = reduceEasy(out, numElements);
	CudaSafeCall(cudaFree(out));
	
	*value = outVal;
}*/
