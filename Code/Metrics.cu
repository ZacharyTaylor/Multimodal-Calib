#include "Metrics.h"
#include "Kernels.h"
#include "reduction.h"
#include "mi.h"

float Metric::evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx){
	mexErrMsgTxt("No metric has been specified");
	return 0;
}

MI::MI(size_t bins):
	bins_(bins){
}

float MI::evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx){
	
	if((scan->getNumCh(scanIdx) != 1)){
		mexErrMsgTxt("MI metric can only accept a single intensity channel");
	}

	float miOut = miRun(gen->getGIP(genIdx,0,scan->getNumPoints(scanIdx)), scan->getIP(scanIdx,0), bins_, scan->getNumPoints(scanIdx), false, gen->getStream(genIdx));
	CudaCheckError();
	cudaDeviceSynchronize();

	return -miOut;
}

NMI::NMI(size_t bins):
	bins_(bins){
}

float NMI::evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx){
	
	if((scan->getNumCh(scanIdx) != 1)){
		mexErrMsgTxt("NMI metric can only accept a single intensity channel");
	}

	float miOut = miRun(gen->getGIP(genIdx,0,scan->getNumPoints(scanIdx)), scan->getIP(scanIdx,0), bins_, scan->getNumPoints(scanIdx), true, gen->getStream(genIdx));
	CudaCheckError();
	cudaDeviceSynchronize();

	return -miOut;
}

SSD::SSD(){};

float SSD::evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx){
	
	if((scan->getNumCh(scanIdx) != 1)){
		mexErrMsgTxt("SSD metric can only accept a single intensity channel");
	}
	
	SSDKernel<<<gridSize(scan->getNumPoints(scanIdx)), BLOCK_SIZE, 0, gen->getStream(genIdx)>>>
		(gen->getGIP(genIdx,0,scan->getNumPoints(scanIdx)), scan->getIP(scanIdx,0), scan->getNumPoints(scanIdx));
	CudaCheckError();
	
	//perform reduction
	float temp = reduceEasy(gen->getGIP(genIdx,0,scan->getNumPoints(scanIdx)), scan->getNumPoints(scanIdx), gen->getStream(genIdx), gen->getTMPD(genIdx),gen->getTMPH(genIdx));
	temp = sqrt(temp);

	return temp;
}

GOM::GOM(){};

float GOM::evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx){
	
	if((scan->getNumCh(scanIdx) != 2)){
		mexErrMsgTxt("GOM metric can only accept a two intensity channels (mag and angle)");
	}
	
	GOMKernel<<<gridSize(scan->getNumPoints(scanIdx)), BLOCK_SIZE, 0, gen->getStream(genIdx)>>>
		(gen->getGIP(genIdx,0,scan->getNumPoints(scanIdx)),
		gen->getGIP(genIdx,1,scan->getNumPoints(scanIdx)),
		scan->getIP(scanIdx,0),
		scan->getIP(scanIdx,1),
		scan->getNumPoints(scanIdx));
	CudaCheckError();

	float phase = reduceEasy(gen->getGIP(genIdx,1,scan->getNumPoints(scanIdx)), scan->getNumPoints(scanIdx), gen->getStream(genIdx), gen->getTMPD(genIdx),gen->getTMPH(genIdx));
	float mag = reduceEasy(gen->getGIP(genIdx,0,scan->getNumPoints(scanIdx)), scan->getNumPoints(scanIdx), gen->getStream(genIdx), gen->getTMPD(genIdx),gen->getTMPH(genIdx));
	
	if(phase == 0){
		return 0;
	}

	float out = -phase / mag;

	return out;
}

GOMS::GOMS(){};

float GOMS::evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx){
	
	if((scan->getNumCh(scanIdx) != 2)){
		mexErrMsgTxt("GOMS metric can only accept a two intensity channels (mag and angle)");
	}
	
	GOMKernel<<<gridSize(scan->getNumPoints(scanIdx)), BLOCK_SIZE, 0, gen->getStream(genIdx)>>>
		(gen->getGIP(genIdx,0,scan->getNumPoints(scanIdx)),
		gen->getGIP(genIdx,1,scan->getNumPoints(scanIdx)),
		scan->getIP(scanIdx,0),
		scan->getIP(scanIdx,1),
		scan->getNumPoints(scanIdx));
	CudaCheckError();

	float phase = reduceEasy(gen->getGIP(genIdx,1,scan->getNumPoints(scanIdx)), scan->getNumPoints(scanIdx), gen->getStream(genIdx), gen->getTMPD(genIdx),gen->getTMPH(genIdx));
	float mag = reduceEasy(gen->getGIP(genIdx,0,scan->getNumPoints(scanIdx)), scan->getNumPoints(scanIdx), gen->getStream(genIdx), gen->getTMPD(genIdx),gen->getTMPH(genIdx));
	
	if(phase == 0){
		return 0;
	}
	float out = log(sqrt(6/(mag*PI))) - (6*((phase - (mag/2))*(phase - (mag/2)))/mag);
	((phase/mag) > 0.5) ? out = 1*out : out = -1*out;

	return out;
}

LEV::LEV(){};

float LEV::evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx){

	if((scan->getNumCh(scanIdx) != 1)){
		mexErrMsgTxt("LEV metric can only accept one intensity channel");
	}
	
	levValKernel<<<gridSize(scan->getNumPoints(scanIdx)), BLOCK_SIZE, 0, gen->getStream(genIdx)>>>
		(gen->getGIP(genIdx,0,scan->getNumPoints(scanIdx)),
		scan->getIP(scanIdx,0),
		scan->getNumPoints(scanIdx));
	CudaCheckError();

	//perform reduction
	float out = reduceEasy(gen->getGIP(genIdx,0,scan->getNumPoints(scanIdx)), scan->getNumPoints(scanIdx), gen->getStream(genIdx), gen->getTMPD(genIdx),gen->getTMPH(genIdx));

	return out;
}
