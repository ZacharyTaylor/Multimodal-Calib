#include "Calib.h"
#include <algorithm>
#include <string> 

Calib::Calib(std::string metricType){
	checkForCUDA();
	/*
	std::transform(metricType.begin(), metricType.end(), metricType.begin(), ::tolower);

	if(tformType == "affine"){
		tformStore = new AffineTforms;
	}
	else if(tformType == "camera"){
		tformStore = new CameraTforms;
	}
	else{
		std::cerr << "Error unrecognized tform " << tformType << ". Options are affine or camera, defaulting to camera";
		tformStore = new CameraTforms;
	}

	moveStore = new ScanList();
	baseStore = new ImageList();
	*/
}

void Calib::clearScans(void){
	moveStore->removeAllScans();
}

void Calib::clearImages(void){
	baseStore->removeAllImages();
}

void Calib::clearTforms(void){
	tformStore->removeAllTforms();
}

void Calib::clearExtras(void){
	return;
}

void Calib::clearEverything(void){
	clearScans();
	clearImages();
	clearTforms();
}

void Calib::addScan(std::vector<thrust::host_vector<float>> scanLIn, std::vector<thrust::host_vector<float>> scanIIn){
	moveStore->addScan(scanLIn, scanIIn);
}

void Calib::addImage(thrust::host_vector<float> imageIn, size_t height, size_t width, size_t depth, size_t tformIdxIn, size_t scanIdxIn){
	tformIdx.push_back(tformIdxIn);
	scanIdx.push_back(scanIdxIn);
	baseStore->addImage(imageIn, height, width, depth);
}

void Calib::addTform(thrust::host_vector<float> tformIn, size_t tformSizeX, size_t tformSizeY){
	tformStore->addTforms(tformIn, tformSizeX, tformSizeY);
}

void Calib::addCamera(thrust::host_vector<float> cameraIn, boolean panoramic){};

float Calib::evalMetric(void){
	return 0;
}

size_t Calib::allocateGenMem(ScanList* points, ImageList* images, std::vector<std::vector<float*>> genL, std::vector<std::vector<float*>> genI, size_t startIdx){
	
	cudaError_t err = cudaSuccess;
	size_t i;

	genL.resize(images->getNumImages());
	genI.resize(images->getNumImages());

	for(i = startIdx; i < images->getNumImages(); i++){

		genL[i].resize(IMAGE_DIM);
		for(size_t j = 0; j < IMAGE_DIM; j++){
			cudaError_t currentErr = cudaMalloc(&genL[i][j], sizeof(float)*points->getNumPoints(scanIdx[i]));
			if(currentErr != cudaSuccess){
				err = cudaErrorMemoryAllocation;
				break;
			}
		}
		genI[i].resize(images->getDepth(i));
		for(size_t j = 0; j < images->getDepth(i); j++){
			cudaError_t currentErr = cudaMalloc(&genI[i][j], sizeof(float)*points->getNumPoints(scanIdx[i]));
			if(currentErr != cudaSuccess){
				err = cudaErrorMemoryAllocation;
				break;
			}
		}

		if(err == cudaErrorMemoryAllocation){
			for(size_t j = 0; j < IMAGE_DIM; j++){
				cudaFree(&genL[i][j]);
			}
			for(size_t j = 0; j < images->getDepth(i); j++){
				cudaFree(&genI[i][j]);
			}
			break;
		}
	}

	return i;
}

CameraCalib::CameraCalib(std::string metricType) : Calib(metricType){}

void CameraCalib::addImage(thrust::host_vector<float> imageIn, size_t height, size_t width, size_t depth, size_t tformIdxIn, size_t scanIdxIn, size_t cameraIdxIn){
	tformIdx.push_back(tformIdxIn);
	scanIdx.push_back(scanIdxIn);
	cameraIdx.push_back(cameraIdxIn);
	baseStore->addImage(imageIn, height, width, depth);
}

void CameraCalib::addCamera(thrust::host_vector<float> cameraIn, boolean panoramic){
	cameraStore->addCams(cameraIn, panoramic);
}

float CameraCalib::evalMetric(void){

	std::vector<std::vector<float*>> genL;
	std::vector<std::vector<float*>> genI;

	std::vector<float> metricVal;

	std::vector<cudaStream_t> streams;

	size_t genLength = 0;
	for(size_t i = 0; i < moveStore->getNumScans(); i+= (genLength+1)){
		genLength = allocateGenMem(moveStore, baseStore, genL, genI, i);
		
		streams.resize(genLength-i);
		for(size_t j = 0; j < streams.size(); j++){
			cudaStreamCreate ( &streams[j]);
			tformStore->transform(moveStore, genL[j], cameraStore, tformIdx[i+j], cameraIdx[i+j], scanIdx[i+j], streams[j]);

		}
	}

	return 0;
}

