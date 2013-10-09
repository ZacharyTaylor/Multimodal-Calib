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
		std::ostringstream err; err << "Error unrecognized tform " << tformType << ". Options are affine or camera, defaulting to camera";
		tformStore = new CameraTforms;
	}

	moveStore = new ScanList();
	baseStore = new ImageList();
	*/
}

void Calib::clearScans(void){
	moveStore.removeAllScans();
}

void Calib::clearImages(void){
	baseStore.removeAllImages();
}

void Calib::clearTforms(void){
	tformStore.removeAllTforms();
}

void Calib::clearExtras(void){
	return;
}

void Calib::clearEverything(void){
	clearScans();
	clearImages();
	clearTforms();
}

void Calib::addScan(std::vector<thrust::host_vector<float>>& scanLIn, std::vector<thrust::host_vector<float>>& scanIIn){
	moveStore.addScan(scanLIn, scanIIn);
}

void Calib::addImage(thrust::host_vector<float>& imageIn, size_t height, size_t width, size_t depth){
	baseStore.addImage(imageIn, height, width, depth);
}

void Calib::addTform(thrust::host_vector<float>& tformIn, size_t tformSizeX, size_t tformSizeY){
	tformStore.addTforms(tformIn, tformSizeX, tformSizeY);
}

float Calib::evalMetric(void){
	return 0;
}

void Calib::addTformIndices(std::vector<size_t>& tformsIdxIn){
	tformIdx.insert(tformIdx.end(), tformsIdxIn.begin(), tformsIdxIn.end());
}

void Calib::addScanIndices(std::vector<size_t>& scansIdxIn){
	scanIdx.insert(scanIdx.end(),scansIdxIn.begin(), scansIdxIn.end());
}

size_t Calib::allocateGenMem(ScanList points, ImageList images, std::vector<std::vector<float*>>& genL, std::vector<std::vector<float*>>& genI, size_t startIdx){
	
	cudaError_t err = cudaSuccess;
	size_t i;

	genL.resize(images.getNumImages());
	genI.resize(images.getNumImages());

	for(i = startIdx; i < images.getNumImages(); i++){

		genL[i].resize(IMAGE_DIM);
		for(size_t j = 0; j < IMAGE_DIM; j++){
			cudaError_t currentErr = cudaMalloc(&genL[i][j], sizeof(float)*points.getNumPoints(scanIdx[i]));
			if(currentErr != cudaSuccess){
				err = cudaErrorMemoryAllocation;
				break;
			}
		}
		genI[i].resize(images.getDepth(i));
		for(size_t j = 0; j < images.getDepth(i); j++){
			cudaError_t currentErr = cudaMalloc(&genI[i][j], sizeof(float)*points.getNumPoints(scanIdx[i]));
			if(currentErr != cudaSuccess){
				err = cudaErrorMemoryAllocation;
				break;
			}
		}

		if(err == cudaErrorMemoryAllocation){
			for(size_t j = 0; j < IMAGE_DIM; j++){
				cudaFree(&genL[i][j]);
			}
			for(size_t j = 0; j < images.getDepth(i); j++){
				cudaFree(&genI[i][j]);
			}
			break;
		}
	}

	return i;
}

void Calib::setSSDMetric(void){
	metric = new SSD();
}

void Calib::setGOMMetric(void){
	metric = new GOM();
}

void Calib::addCameraIndices(std::vector<size_t>& cameraIdxIn){
	mexErrMsgTxt("Attempted to setup camera for use with non-camera calibration");
	return;
}

void Calib::addCamera(thrust::host_vector<float>& cameraIn, boolean panoramic){
	mexErrMsgTxt("Attempted to setup camera for use with non-camera calibration");
	return;
}

CameraCalib::CameraCalib(std::string metricType) : Calib(metricType){}

void CameraCalib::addTform(thrust::host_vector<float>& tformIn){
	tformStore.addTforms(tformIn);
}

void CameraCalib::addCameraIndices(std::vector<size_t>& cameraIdxIn){
	cameraIdx.insert(cameraIdx.end(),cameraIdxIn.begin(), cameraIdxIn.end());
}

void CameraCalib::addCamera(thrust::host_vector<float>& cameraIn, boolean panoramic){
	cameraStore.addCams(cameraIn, panoramic);
}

float CameraCalib::evalMetric(void){

	std::vector<std::vector<float*>> genL;
	std::vector<std::vector<float*>> genI;

	std::vector<float> metricVal;

	std::vector<cudaStream_t> streams;

	if(tformIdx.size() != baseStore.getNumImages()){
		std::ostringstream err; err << "Transform index has not been correctly set up";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}
	if(cameraIdx.size() != baseStore.getNumImages()){
		std::ostringstream err; err << "Camera index has not been correctly set up";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}
	if(scanIdx.size() != baseStore.getNumImages()){
		std::ostringstream err; err << "Scan index has not been correctly set up";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}

	size_t genLength = 0;
	float out = 0;
	for(size_t i = 0; i < moveStore.getNumScans(); i+= (genLength+1)){
		genLength = allocateGenMem(moveStore, baseStore, genL, genI, i);

		if(genLength == 0){
			mexErrMsgTxt("Memory allocation for generated scans failed\n");
		}
		
		streams.resize(genLength-i);
		for(size_t j = 0; j < streams.size(); j++){
			cudaStreamCreate ( &streams[j]);
			tformStore.transform(moveStore, genL[j], cameraStore, tformIdx[i+j], cameraIdx[i+j], scanIdx[i+j], streams[j]);
			cudaDeviceSynchronize();
			baseStore.interpolateImage(i+j, moveStore, scanIdx[i+j], genI[j], true, streams[j]);
			cudaDeviceSynchronize();
			out += metric->evalMetric(genI[j], moveStore, scanIdx[i+j], streams[j]);
			cudaDeviceSynchronize();
		}
	}

	return out;
}

