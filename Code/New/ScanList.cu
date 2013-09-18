#include "ScanList.h"

ScanList::ScanList(void){};

ScanList::~ScanList(void){};

size_t ScanList::getNumDim(void){
	return scanL.size();
}
	
size_t ScanList::getNumCh(void){
	return scanI.size();
}
	
size_t ScanList::getNumPoints(void){
	size_t val = 0;
	if(scanL.size()){
		val = scanL[0].size();
	}
	return val;
}
	
float* ScanList::getLP(size_t idx){
	if(scanL.size() < idx){
		std::cerr << "Error only " << scanL.size() << " dimensions exist cannot get pointer to dimension " << idx << "\n";
		return NULL;
	}
	return thrust::raw_pointer_cast(&scanL[idx][0]);
}

float* ScanList::getIP(size_t idx){
	if(scanI.size() < idx){
		std::cerr << "Error only " << scanL.size() << " dimensions exist cannot get pointer to dimension " << idx << "\n";
		return NULL;
	}
	return thrust::raw_pointer_cast(&scanI[idx][0]);
}

size_t* ScanList::getIdxP(void){
	return thrust::raw_pointer_cast(&scanIdx[0]);
}

void ScanList::addScan(std::vector<thrust::device_vector<float>> scanLIn, std::vector<thrust::device_vector<float>> scanIIn){
	for(size_t i = 0; i < scanL.size(); i++){
		scanL[i].insert(scanL[i].end(),scanLIn[i].begin(), scanLIn[i].end());
	}
	for(size_t i = 0; i < scanI.size(); i++){
		scanI[i].insert(scanI[i].end(),scanIIn[i].begin(), scanIIn[i].end());
	}
	
	if(scanIdx.size()){
		scanIdx.push_back(scanIdx.back() + scanLIn[0].size());
	}
	else{
		scanIdx.push_back(scanLIn[0].size());
	}
}

void ScanList::addScan(std::vector<thrust::host_vector<float>> scanLIn, std::vector<thrust::host_vector<float>> scanIIn){
	for(size_t i = 0; i < scanL.size(); i++){
		scanL[i].insert(scanL[i].end(),scanLIn[i].begin(), scanLIn[i].end());
	}
	for(size_t i = 0; i < scanI.size(); i++){
		scanI[i].insert(scanI[i].end(),scanIIn[i].begin(), scanIIn[i].end());
	}
		
	if(scanIdx.size()){
		scanIdx.push_back(scanIdx.back() + scanLIn[0].size());
	}
	else{
		scanIdx.push_back(scanLIn[0].size());
	}
}

void ScanList::removeScan(size_t idx){
	std::vector<thrust::device_vector<float>>::iterator startL, endL, startI, endI;
	
	if(idx >= scanIdx.size()){
		return;
	}
	else if((idx+1) != scanIdx.size()){
		endL = scanL.begin() + scanIdx[idx+1] - 1;
		endI = scanI.begin() + scanIdx[idx+1] - 1;
	}
	else{
		endL = scanL.end();
		endI = scanI.end();
	}

	startL = scanL.begin() + scanIdx[idx];
	startI = scanI.begin() + scanIdx[idx];
	
	scanL.erase(startL, endL);
	scanI.erase(startI, endI);

	size_t size = scanIdx[idx];
	if(idx != 0){
		size -= scanIdx[idx-1];
	}
	for(size_t i = idx+1; i < scanIdx.size(); i++){
		scanIdx[i] -= size;
	}
	scanIdx.pop_back();
}

void ScanList::removeLastScan(){
	std::vector<thrust::device_vector<float>>::iterator startL, startI;
		
	startL = scanL.end() - scanIdx.back();
	startI = scanI.end() - scanIdx.back();
	scanL.erase(startL, scanL.end());
	scanI.erase(startI, scanI.end());

	scanIdx.erase(scanIdx.end());
}

void ScanList::removeAllScans(){
	scanL.clear();
	scanI.clear();
	scanIdx.clear();
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