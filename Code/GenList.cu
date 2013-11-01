#include "GenList.h"
#include "Kernels.h"

void GenList::setupGenList(size_t numGenScans){
	genL.resize(numGenScans);
	genI.resize(numGenScans);
	streams.resize(numGenScans);
	tempMemD.resize(numGenScans);
	tempMemH.resize(numGenScans);

	for(size_t i = 0; i < numGenScans; i++){
		cudaStreamCreate (&(streams[i]));
		tempMemD[i].resize(TEMP_MEM_SIZE);
		cudaMallocHost((void**)&tempMemH[i], sizeof(float)*TEMP_MEM_SIZE);
	}
};

GenList::~GenList(void){
	for(size_t i = 0; i < streams.size(); i++){
		cudaStreamDestroy (streams[i]);
		cudaFreeHost(tempMemH[i]);
	}
};

size_t GenList::getNumGen(void){
	return streams.size();
}

cudaStream_t GenList::getStream(size_t idx){
	if(streams.size() <= idx){
		std::ostringstream err; err << "Error only " << streams.size() << " streams exist cannot get stream for scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}

	return streams[idx];
}

float* GenList::getGLP(size_t idx, size_t dim, size_t numPoints){
	if(genL.size() <= idx){
		std::ostringstream err; err << "Error only " << genL.size() << " scans exist cannot get pointer to scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}
	if(genL[idx].size() < (dim+1)){
		genL[idx].resize(dim+1);
	}
	if(genL[idx][dim].size() < (numPoints+1)){
		genL[idx][dim].resize(numPoints+1);
	}

	return thrust::raw_pointer_cast(&genL[idx][dim][0]);
}

float* GenList::getGIP(size_t idx, size_t ch, size_t numPoints){
	if(genI.size() <= idx){
		std::ostringstream err; err << "Error only " << genI.size() << " scans exist cannot get pointer to scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}
	if(genI[idx].size() < (ch+1)){
		genI[idx].resize(ch+1);
	}
	if(genI[idx][ch].size() < (numPoints+1)){
		genI[idx][ch].resize(numPoints+1);
	}

	return thrust::raw_pointer_cast(&genI[idx][ch][0]);
}

float* GenList::getTMPD(size_t idx){
	if(tempMemD.size() <= idx){
		std::ostringstream err; err << "Error only " << tempMemD.size() << " scans exist cannot get temp memory for scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}

	return thrust::raw_pointer_cast(&tempMemD[idx][0]);
}

float* GenList::getTMPH(size_t idx){
	if(tempMemD.size() <= idx){
		std::ostringstream err; err << "Error only " << tempMemD.size() << " scans exist cannot get temp memory for scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}

	return tempMemH[idx];
}