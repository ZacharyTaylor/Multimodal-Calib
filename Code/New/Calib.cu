#include "Calib.h"

Calib::Calib(size_t imHeight, size_t imWidth, size_t imDepth){
	checkForCUDA();

	moveStore = new ScanList();
	baseStore = new ImageList(imHeight, imWidth, imDepth);

	genStore = new ScanList();

	tformStore = new Tforms();
}


void Calib::clearScans(void){
	moveStore->removeAllScans();
}

void Calib::clearGenerated(void){
	genStore->removeAllScans();
}

void Calib::clearImages(void){
	baseStore->removeAllImages();
}

void Calib::clearTforms(void){
	tformStore->removeAllTforms();
}

void Calib::clearEverything(void){
	clearScans();
	clearImages();
	clearTforms();
}

void Calib::addScan(std::vector<thrust::host_vector<float>> scanLIn, std::vector<thrust::host_vector<float>> scanIIn){
	if((moveStore->getNumPoints() != 0) && ((moveStore->getNumDim() != scanLIn.size()) || moveStore->getNumCh() != scanIIn.size())){
		std::cerr << "Number of dimensions and channels must match scans already set, returning without setting\n";
		return;
	}
	moveStore->addScan(scanLIn, scanIIn);
}

void Calib::addImage(thrust::host_vector<float> imageIn){
	if(imageIn.size != (baseStore->getDepth() * baseStore->getHeight() * baseStore->getWidth())){
		std::cerr << "Image must match size specified at creation, returning without setting\n";
		return;
	}
	baseStore->addImage(imageIn);
}

void Calib::addTform(thrust::host_vector<float> tformIn){
	if(tformStore->getTformSize() != tformIn.size()){
		std::cerr << "Tform must match size of initilized tforms, returning without setting\n";
		return;
	}
	tformStore->addTforms(tformIn);
}

float Calib::evalMetric(void){
	size_t genLength = genStore->allocateMemory(IMAGE_DIM,baseStore->getDepth(),moveStore->getNumPoints());

	for(size_t i = 0; i < moveStore->getNumPoints(); i+= (genLength+1)){
		tformStore->transform(moveStore, genStore, baseStore, i);
	}
}