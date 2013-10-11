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

	return thrust::raw_pointer_cast(&scanL[idx][ch][0]);
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
}

void ScanList::removeScan(size_t idx){
	if(scanL.size() < idx){
		std::ostringstream err; err << "Error only " << scanL.size() << " scans exist cannot erase scan " << idx;
		mexErrMsgTxt(err.str().c_str());
		return;
	}
	scanL.erase(scanL.begin() + idx);
	scanI.erase(scanI.begin() + idx);
}

void ScanList::removeLastScan(){
	scanL.pop_back();
	scanI.pop_back();
}

void ScanList::removeAllScans(){
	scanL.clear();
	scanI.clear();
}

void ScanList::generateImage(thrust::device_vector<float>& out, size_t idx, size_t width, size_t height, size_t dilate){
		
	out.resize(width*height*this->getNumCh(idx));

	for(size_t i = 0; i < this->getNumCh(idx); i++){
		generateOutputKernel<<<gridSize(this->getNumPoints(idx)) ,BLOCK_SIZE>>>(
			this->getLP(idx,0),
			this->getLP(idx,1),
			this->getIP(idx,i),
			thrust::raw_pointer_cast(&out[width*height*i]),
			width,
			height,
			this->getNumPoints(idx),
			dilate);
	}

	CudaCheckError();
}