#include "ScanList.h"
#include "Kernels.h"

ScanList::ScanList(void){};

ScanList::~ScanList(void){};

size_t ScanList::getNumDim(size_t idx){
	if(scanL.size() <= idx){
		std::ostringstream err; err << "Cannot get dimensions of element " << idx << " as only " << scanL.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}

	return scanL[idx].size();
}
	
size_t ScanList::getNumCh(size_t idx){
	if(scanI.size() <= idx){
		std::ostringstream err; err << "Cannot get channels of element " << idx << " as only " << scanI.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}
	return scanI[idx].size();
}
	
size_t ScanList::getNumPoints(size_t idx){
	if(scanL.size() <= idx){
		std::ostringstream err; err << "Cannot get size of element " << idx << " as only " << scanL.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}

	size_t val = 0;
	if(scanL[idx].size()){
		val = scanL[idx][0].size();
	}
	return val;
}

size_t ScanList::getNumScans(void){
	return scanL.size();
}
	
float* ScanList::getLP(size_t idx, size_t dim){
	if(scanL.size() <= idx){
		std::ostringstream err; err << "Error only " << scanL.size() << " scans exist cannot get pointer to scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}
	if(scanL[idx].size() <= dim){
		std::ostringstream err; err << "Error only " << scanL[idx].size() << " dimensions exist cannot get pointer to dimension " << dim;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}

	return thrust::raw_pointer_cast(&scanL[idx][dim][0]);
}

float* ScanList::getIP(size_t idx, size_t ch){
	if(scanL.size() <= idx){
		std::ostringstream err; err << "Error only " << scanL.size() << " scans exist cannot get pointer to scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}
	if(scanL[idx].size() <= ch){
		std::ostringstream err; err << "Error only " << scanL[idx].size() << " channels exist cannot get pointer to channel " << ch;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}

	return thrust::raw_pointer_cast(&scanI[idx][ch][0]);
}

float* ScanList::getGLP(size_t idx, size_t dim){
	if(genL.size() <= idx){
		std::ostringstream err; err << "Error only " << scanL.size() << " scans exist cannot get pointer to scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}
	if(genL[idx].size() <= dim){
		std::ostringstream err; err << "Error only " << scanL[idx].size() << " dimensions exist cannot get pointer to dimension " << dim;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}

	return thrust::raw_pointer_cast(&genL[idx][dim][0]);
}

float* ScanList::getGIP(size_t idx, size_t ch){
	if(genI.size() <= idx){
		std::ostringstream err; err << "Error only " << scanL.size() << " scans exist cannot get pointer to scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}
	if(genI[idx].size() <= ch){
		std::ostringstream err; err << "Error only " << scanL[idx].size() << " channels exist cannot get pointer to channel " << ch;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}

	return thrust::raw_pointer_cast(&genI[idx][ch][0]);
}

void ScanList::setGenIDepth(size_t idx, size_t depth){
	if(genI.size() <= idx){
		std::ostringstream err; err << "Error only " << scanL.size() << " scans exist cannot get pointer to scan " << idx;
		mexErrMsgTxt(err.str().c_str());
	}
	while(genI[idx].size() < depth){
		genI[idx].push_back(genI[idx].back());
	}
}

float* ScanList::getTMP(size_t idx){
	if(tempMem.size() <= idx){
		std::ostringstream err; err << "Error only " << scanL.size() << " scans exist cannot get temp memory for scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}

	return thrust::raw_pointer_cast(&tempMem[idx][0]);
}

cudaStream_t ScanList::getStream(size_t idx){
	if(scanL.size() <= idx){
		std::ostringstream err; err << "Error only " << scanL.size() << " scans exist cannot get stream for scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}

	return streams[idx];
}

void ScanList::addScan(std::vector<thrust::device_vector<float>>& scanLIn, std::vector<thrust::device_vector<float>>& scanIIn){
	//check sizes all match
	if(scanLIn.size() == 0){
		std::ostringstream err; err << "Error cannot have empty location vector";
		mexErrMsgTxt(err.str().c_str());
		return;
	}
	for(size_t i = 1; i < scanLIn.size(); i++){
		if(scanLIn[0].size() != scanLIn[i].size()){
			std::ostringstream err; err << "Error all scan vectors must be of equal length";
			mexErrMsgTxt(err.str().c_str());
			return;
		}
	}
	for(size_t i = 0; i < scanIIn.size(); i++){
		if(scanLIn[0].size() != scanIIn[i].size()){
			std::ostringstream err; err << "Error all scan vectors must be of equal length";
			mexErrMsgTxt(err.str().c_str());
			return;
		}
	}

	scanL.push_back(scanLIn);
	scanI.push_back(scanIIn);

	genL.push_back(scanLIn);
	genI.push_back(scanIIn);

	streams.push_back(NULL);
	cudaStreamCreate (&(streams.back()));

	std::vector<float> temp;
	tempMem.push_back(temp);
	tempMem.back().resize(TEMP_MEM_SIZE);
}

void ScanList::addScan(std::vector<thrust::host_vector<float>>& scanLIn, std::vector<thrust::host_vector<float>>& scanIIn){
	//check sizes all match
	if(scanLIn.size() == 0){
		std::ostringstream err; err << "Error cannot have empty location vector";
		mexErrMsgTxt(err.str().c_str());
		return;
	}
	for(size_t i = 1; i < scanLIn.size(); i++){
		if(scanLIn[0].size() != scanLIn[i].size()){
			std::ostringstream err; err << "Error all scan vectors must be of equal length";
			mexErrMsgTxt(err.str().c_str());
			return;
		}
	}
	for(size_t i = 0; i < scanIIn.size(); i++){
		if(scanLIn[0].size() != scanIIn[i].size()){
			std::ostringstream err; err << "Error all scan vectors must be of equal length";
			mexErrMsgTxt(err.str().c_str());
			return;
		}
	}
	std::vector<thrust::device_vector<float>> tempLIn;
	std::vector<thrust::device_vector<float>> tempIIn;

	for(size_t i = 0; i < scanLIn.size(); i++){
		tempLIn.push_back(scanLIn[i]);
	}
	for(size_t i = 0; i < scanIIn.size(); i++){
		tempIIn.push_back(scanIIn[i]);
	}

	scanL.push_back(tempLIn);
	scanI.push_back(tempIIn);

	genL.push_back(tempLIn);
	genI.push_back(tempIIn);

	streams.push_back(NULL);
	cudaStreamCreate (&(streams.back()));

	std::vector<float> temp;
	tempMem.push_back(temp);
	tempMem.back().resize(TEMP_MEM_SIZE);
}

void ScanList::removeScan(size_t idx){
	if(scanL.size() < idx){
		std::ostringstream err; err << "Error only " << scanL.size() << " scans exist cannot erase scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return;
	}
	scanL.erase(scanL.begin() + idx);
	scanI.erase(scanI.begin() + idx);

	genL.erase(genL.begin() + idx);
	genI.erase(genI.begin() + idx);

	cudaStreamDestroy(streams[idx]);
	streams.erase(streams.begin() + idx);

	tempMem.erase(tempMem.begin() + idx);
}

void ScanList::removeLastScan(){
	scanL.pop_back();
	scanI.pop_back();

	genL.pop_back();
	genI.pop_back();

	cudaStreamDestroy(streams.back());
	streams.pop_back();

	tempMem.pop_back();
}

void ScanList::removeAllScans(){
	scanL.clear();
	scanI.clear();

	genL.clear();
	genI.clear();

	for(size_t i = 0; i < streams.size(); i++){
		cudaStreamDestroy(streams[i]);
	}
	streams.clear();

	tempMem.clear();
}