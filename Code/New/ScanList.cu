#include "ScanList.h"

ScanList::ScanList(void){};

ScanList::~ScanList(void){};

size_t ScanList::getNumDim(size_t idx){
	if(scanL.size() > idx){
		std::cerr << "Cannot get dimensions of element " << idx << " as only " << scanL.size() << " elements exist. Returning 0\n";
		return 0;
	}

	return scanL[idx].size();
}
	
size_t ScanList::getNumCh(size_t idx){
	if(scanI.size() > idx){
		std::cerr << "Cannot get channels of element " << idx << " as only " << scanI.size() << " elements exist. Returning 0\n";
		return 0;
	}
	return scanI[idx].size();
}
	
size_t ScanList::getNumPoints(size_t idx){
	if(scanL.size() > idx){
		std::cerr << "Cannot get size of element " << idx << " as only " << scanL.size() << " elements exist. Returning 0\n";
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
	if(scanL.size() < idx){
		std::cerr << "Error only " << scanL.size() << " scans exist cannot get pointer to scan " << idx << ". Returning NULL\n";
		return NULL;
	}
	if(scanL[idx].size() < dim){
		std::cerr << "Error only " << scanL[idx].size() << " dimensions exist cannot get pointer to dimension " << dim << ". Returning NULL\n";
		return NULL;
	}

	return thrust::raw_pointer_cast(&scanL[idx][dim][0]);
}

float* ScanList::getIP(size_t idx, size_t ch){
	if(scanL.size() < idx){
		std::cerr << "Error only " << scanL.size() << " scans exist cannot get pointer to scan " << idx << ". Returning NULL\n";
		return NULL;
	}
	if(scanL[idx].size() < ch){
		std::cerr << "Error only " << scanL[idx].size() << " channels exist cannot get pointer to channel " << ch << ". Returning NULL\n";
		return NULL;
	}

	return thrust::raw_pointer_cast(&scanL[idx][ch][0]);
}

void ScanList::addScan(std::vector<thrust::device_vector<float>> scanLIn, std::vector<thrust::device_vector<float>> scanIIn){
	//check sizes all match
	if(scanLIn.size() == 0){
		std::cerr << "Error cannot have empty location vector. Returning without setting \n";
		return;
	}
	for(size_t i = 1; i < scanLIn.size(); i++){
		if(scanLIn[0].size() != scanLIn[i].size()){
			std::cerr << "Error all scan vectors must be of equal length. Returning without setting \n";
			return;
		}
	}
	for(size_t i = 0; i < scanIIn.size(); i++){
		if(scanLIn[0].size() != scanIIn[i].size()){
			std::cerr << "Error all scan vectors must be of equal length. Returning without setting \n";
			return;
		}
	}

	scanL.push_back(scanLIn);
	scanI.push_back(scanIIn);
}

void ScanList::addScan(std::vector<thrust::host_vector<float>> scanLIn, std::vector<thrust::host_vector<float>> scanIIn){
	//check sizes all match
	if(scanLIn.size() == 0){
		std::cerr << "Error cannot have empty location vector. Returning without setting \n";
		return;
	}
	for(size_t i = 1; i < scanLIn.size(); i++){
		if(scanLIn[0].size() != scanLIn[i].size()){
			std::cerr << "Error all scan vectors must be of equal length. Returning without setting \n";
			return;
		}
	}
	for(size_t i = 0; i < scanIIn.size(); i++){
		if(scanLIn[0].size() != scanIIn[i].size()){
			std::cerr << "Error all scan vectors must be of equal length. Returning without setting \n";
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
		std::cerr << "Error only " << scanL.size() << " scans exist cannot erase scan " << idx << ". Returning\n";
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

size_t ScanList::allocateMemory(size_t dims, size_t ch, size_t length){
	
	if(scanL[0].size() <= length){
		return length;
	}

	for(size_t i = 0; i < scanL.size; i++){
		scanL[i].clear();
		scanI[i].clear();
		scanL[i].shrink_to_fit();
		scanI[i].shrink_to_fit();
	}
	size_t free;
	size_t total;
	cuMemGetInfo(&free, &total);
	CudaCheckError();

	free = free * MEM_LIMIT;
	ch = (ch != 0) ? ch : 1;
	dims = (dims != 0) ? dims : 1;
	free = free / (dims * ch * sizeof(float));

	length = (length > free) ? free : length;

	for(size_t i = 0; i < scanL.size; i++){
		scanL[i].resize(length);
		scanI[i].resize(length);
	}

	return length;
}