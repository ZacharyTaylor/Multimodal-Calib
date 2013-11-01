#include "Calib.h"
#include <algorithm>
#include <string> 

Calib::Calib(size_t numGenScans){
	checkForCUDA();
	genStore.setupGenList(numGenScans);
}

Calib::~Calib(void){

}

bool Calib::getIfPanoramic(size_t idx){
	return NULL;
}

size_t Calib::getNumPoints(size_t idx){
	return moveStore.getNumPoints(idx);
}

size_t Calib::getNumDim(size_t idx){
	return moveStore.getNumDim(idx);
}

size_t Calib::getImageDepth(size_t idx){
	return baseStore.getDepth(idx);
}

size_t Calib::getNumCh(size_t idx){
	return moveStore.getNumCh(scanIdx[idx]);
}

size_t Calib::getNumImages(void){
	return baseStore.getNumImages();
}

size_t Calib::getImageWidth(size_t idx){
	return baseStore.getWidth(idx);
}

size_t Calib::getImageHeight(size_t idx){
	return baseStore.getHeight(idx);
}

void Calib::clearScans(void){
	moveStore.removeAllScans();
}

void Calib::clearImages(void){
	baseStore.removeAllImages();
}

void Calib::clearTforms(void){
	return;
}

void Calib::clearExtras(void){
	return;
}

void Calib::clearIndices(void){
	tformIdx.clear();
	scanIdx.clear();
}

void Calib::addScan(std::vector<thrust::host_vector<float>>& scanLIn, std::vector<thrust::host_vector<float>>& scanIIn){
	moveStore.addScan(scanLIn, scanIIn);
}

void Calib::addImage(thrust::host_vector<float>& imageIn, size_t height, size_t width, size_t depth){
	baseStore.addImage(imageIn, height, width, depth);
}

/*void Calib::addTform(thrust::host_vector<float>& tformIn, size_t tformSizeX, size_t tformSizeY){
	tformStore.addTforms(tformIn, tformSizeX, tformSizeY);
}*/

void Calib::addTform(thrust::host_vector<float>& tformIn){
	return;
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

void Calib::setSSDMetric(void){
	metric = new SSD();
}

void Calib::setGOMMetric(void){
	metric = new GOM();
}

void Calib::setGOMSMetric(void){
	metric = new GOMS();
}

void Calib::setMIMetric(void){
	metric = new MI(50);
}

void Calib::setNMIMetric(void){
	metric = new NMI(50);
}

void Calib::setLEVMetric(void){
	metric = new LEV();
}


void Calib::addCameraIndices(std::vector<size_t>& cameraIdxIn){
	mexErrMsgTxt("Attempted to setup camera for use with non-camera calibration");
	return;
}

void Calib::addCamera(thrust::host_vector<float>& cameraIn, boolean panoramic){
	mexErrMsgTxt("Attempted to setup camera for use with non-camera calibration");
	return;
}

void Calib::getBaseImage(thrust::device_vector<float>& image, size_t idx){
	image = baseStore.getImage(idx);
}

void Calib::generateImage(thrust::device_vector<float>& image, size_t width, size_t height, size_t dilate, size_t idx, bool imageColour){
	return;
}

void Calib::colourScan(float* scan, size_t idx){
	return;
}

CameraCalib::CameraCalib(size_t numGen) : Calib(numGen){}

bool CameraCalib::getIfPanoramic(size_t idx){
	return cameraStore.getPanoramic(idx);
}

void CameraCalib::clearTforms(void){
	tformStore.removeAllTforms();
}

void CameraCalib::clearExtras(void){
	cameraStore.removeAllCameras();
	return;
}

void CameraCalib::clearIndices(void){
	tformIdx.clear();
	scanIdx.clear();
	cameraIdx.clear();
}

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

	std::vector<float> metricVal;

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

	float out = 0;

	for(size_t j = 0; j < baseStore.getNumImages(); j+= genStore.getNumGen()){
		for(size_t i = 0; i < genStore.getNumGen(); i++){
			if((i+j) >= baseStore.getNumImages()){
				break;
			}
			for(size_t k = 0; k < IMAGE_DIM; k++){
				CudaSafeCall(cudaMemsetAsync(genStore.getGLP(i,k,moveStore.getNumPoints(scanIdx[j+i])),0,moveStore.getNumPoints(scanIdx[j+i]),genStore.getStream(i)));
			}
			for(size_t k = 0; k < moveStore.getNumCh(scanIdx[j]); k++){
				CudaSafeCall(cudaMemsetAsync(genStore.getGIP(i,k,moveStore.getNumPoints(scanIdx[j+i])),0,moveStore.getNumPoints(scanIdx[j+i]),genStore.getStream(i)));
			}
 	
			tformStore.transform(&moveStore, &cameraStore, &genStore, tformIdx[j+i], cameraIdx[j+i], scanIdx[j+i],i);
		}
		for(size_t i = 0; i < genStore.getNumGen(); i++){
			if((i+j) >= baseStore.getNumImages()){
				break;
			}
			baseStore.interpolateImage(&moveStore, &genStore, j+i, scanIdx[j+i], i, true);
		}
		for(size_t i = 0; i < genStore.getNumGen(); i++){
			if((i+j) >= baseStore.getNumImages()){
				break;
			}
			out += metric->evalMetric(&moveStore, &genStore, scanIdx[j+i], i);
		}
	}

	return out;
}

void CameraCalib::generateImage(thrust::device_vector<float>& image, size_t width, size_t height, size_t dilate, size_t idx, bool imageColour){

	if(imageColour){
		image.resize(baseStore.getDepth(idx)*width*height);
	}
	else{
		image.resize(moveStore.getNumCh(scanIdx[idx])*width*height);
	}

	tformStore.transform(&moveStore, &cameraStore, &genStore, tformIdx[idx], cameraIdx[idx], scanIdx[idx], 0);
	cudaDeviceSynchronize();

	if(imageColour){
		baseStore.interpolateImage(&moveStore, &genStore, idx, scanIdx[idx], 0, true);
		cudaDeviceSynchronize();
		
		for(size_t i = 0; i < baseStore.getDepth(idx); i++){
			generateOutputKernel<<<gridSize(moveStore.getNumPoints(scanIdx[idx])) ,BLOCK_SIZE>>>(
				genStore.getGLP(0,0,moveStore.getNumPoints(scanIdx[idx])),
				genStore.getGLP(0,1,moveStore.getNumPoints(scanIdx[idx])),
				genStore.getGIP(0,i,moveStore.getNumPoints(scanIdx[idx])),
				thrust::raw_pointer_cast(&image[width*height*i]),
				width,
				height,
				moveStore.getNumPoints(scanIdx[idx]),
				dilate);
		}
	}
	else{
		for(size_t i = 0; i < moveStore.getNumCh(scanIdx[idx]); i++){
			generateOutputKernel<<<gridSize(moveStore.getNumPoints(scanIdx[idx])) ,BLOCK_SIZE>>>(
				genStore.getGLP(0,0,moveStore.getNumPoints(scanIdx[idx])),
				genStore.getGLP(0,1,moveStore.getNumPoints(scanIdx[idx])),
				moveStore.getIP(scanIdx[idx],i),
				thrust::raw_pointer_cast(&image[width*height*i]),
				width,
				height,
				moveStore.getNumPoints(scanIdx[idx]),
				dilate);
		}
	}

	CudaCheckError();
}

void CameraCalib::colourScan(float* scan, size_t idx){
	
	tformStore.transform(&moveStore, &cameraStore, &genStore, tformIdx[idx], cameraIdx[idx], scanIdx[idx], 0);

	baseStore.interpolateImage(&moveStore, &genStore, idx, scanIdx[idx], 0, true);

	cudaDeviceSynchronize();

	for(size_t j = 0; j < moveStore.getNumDim(scanIdx[idx]); j++){
		cudaMemcpy(&scan[j*moveStore.getNumPoints(scanIdx[idx])],moveStore.getLP(scanIdx[idx],j),moveStore.getNumPoints(scanIdx[idx])*sizeof(float),cudaMemcpyDeviceToHost);
	}
	for(size_t j = 0; j < baseStore.getDepth(idx); j++){
		cudaMemcpy(&scan[(j+moveStore.getNumDim(scanIdx[idx]))*moveStore.getNumPoints(scanIdx[idx])],genStore.getGIP(0,j,moveStore.getNumPoints(scanIdx[idx])),moveStore.getNumPoints(scanIdx[idx])*sizeof(float),cudaMemcpyDeviceToHost);
	}
	CudaCheckError();
}

ImageCalib::ImageCalib(size_t numGen) : Calib(numGen){}

bool ImageCalib::getIfPanoramic(size_t idx){
	return NULL;
}

void ImageCalib::clearTforms(void){
	tformStore.removeAllTforms();
}

void ImageCalib::clearExtras(void){
	return;
}

void ImageCalib::clearIndices(void){
	tformIdx.clear();
	scanIdx.clear();
}

void ImageCalib::addTform(thrust::host_vector<float>& tformIn){
	tformStore.addTforms(tformIn);
}

float ImageCalib::evalMetric(void){

	std::vector<float> metricVal;

	if(tformIdx.size() != baseStore.getNumImages()){
		std::ostringstream err; err << "Transform index has not been correctly set up";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}
	if(scanIdx.size() != baseStore.getNumImages()){
		std::ostringstream err; err << "Scan index has not been correctly set up";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}

	float out = 0;

	for(size_t j = 0; j < baseStore.getNumImages(); j+= genStore.getNumGen()){
		for(size_t i = 0; i < genStore.getNumGen(); i++){
			if((i+j) >= baseStore.getNumImages()){
				break;
			}
			for(size_t k = 0; k < IMAGE_DIM; k++){
				CudaSafeCall(cudaMemsetAsync(genStore.getGLP(i,k,moveStore.getNumPoints(scanIdx[j+i])),0,moveStore.getNumPoints(scanIdx[j+i]),genStore.getStream(i)));
			}
			for(size_t k = 0; k < moveStore.getNumCh(scanIdx[j]); k++){
				CudaSafeCall(cudaMemsetAsync(genStore.getGIP(i,k,moveStore.getNumPoints(scanIdx[j+i])),0,moveStore.getNumPoints(scanIdx[j+i]),genStore.getStream(i)));
			}
 	
			tformStore.transform(&moveStore, &noCamera, &genStore, tformIdx[j+i], NULL, scanIdx[j+i],i);
		}
		for(size_t i = 0; i < genStore.getNumGen(); i++){
			if((i+j) >= baseStore.getNumImages()){
				break;
			}
			baseStore.interpolateImage(&moveStore, &genStore, j+i, scanIdx[j+i], i, true);
		}
		for(size_t i = 0; i < genStore.getNumGen(); i++){
			if((i+j) >= baseStore.getNumImages()){
				break;
			}
			out += metric->evalMetric(&moveStore, &genStore, scanIdx[j+i], i);
		}
	}

	return out;
}

void ImageCalib::generateImage(thrust::device_vector<float>& image, size_t width, size_t height, size_t dilate, size_t idx, bool imageColour){

	if(imageColour){
		image.resize(baseStore.getDepth(idx)*width*height);
	}
	else{
		image.resize(moveStore.getNumCh(scanIdx[idx])*width*height);
	}

	tformStore.transform(&moveStore, &noCamera, &genStore, tformIdx[idx], NULL, scanIdx[idx], 0);
	cudaDeviceSynchronize();

	if(imageColour){
		baseStore.interpolateImage(&moveStore, &genStore, idx, scanIdx[idx], 0, true);
		cudaDeviceSynchronize();
		
		for(size_t i = 0; i < baseStore.getDepth(idx); i++){
			generateOutputKernel<<<gridSize(moveStore.getNumPoints(scanIdx[idx])) ,BLOCK_SIZE>>>(
				genStore.getGLP(0,0,moveStore.getNumPoints(scanIdx[idx])),
				genStore.getGLP(0,1,moveStore.getNumPoints(scanIdx[idx])),
				genStore.getGIP(0,i,moveStore.getNumPoints(scanIdx[idx])),
				thrust::raw_pointer_cast(&image[width*height*i]),
				width,
				height,
				moveStore.getNumPoints(scanIdx[idx]),
				dilate);
		}
	}
	else{
		for(size_t i = 0; i < moveStore.getNumCh(scanIdx[idx]); i++){
			generateOutputKernel<<<gridSize(moveStore.getNumPoints(scanIdx[idx])) ,BLOCK_SIZE>>>(
				genStore.getGLP(0,0,moveStore.getNumPoints(scanIdx[idx])),
				genStore.getGLP(0,1,moveStore.getNumPoints(scanIdx[idx])),
				moveStore.getIP(scanIdx[idx],i),
				thrust::raw_pointer_cast(&image[width*height*i]),
				width,
				height,
				moveStore.getNumPoints(scanIdx[idx]),
				dilate);
		}
	}

	CudaCheckError();
}

void ImageCalib::colourScan(float* scan, size_t idx){

	tformStore.transform(&moveStore, &noCamera, &genStore, tformIdx[idx], NULL, scanIdx[idx], 0);

	baseStore.interpolateImage(&moveStore, &genStore, idx, scanIdx[idx], 0, true);

	cudaDeviceSynchronize();

	for(size_t j = 0; j < moveStore.getNumCh(idx); j++){
		cudaMemcpy(&scan[j*moveStore.getNumPoints(idx)],moveStore.getIP(idx,j),moveStore.getNumPoints(idx)*sizeof(float),cudaMemcpyDeviceToHost);
	}
	for(size_t j = 0; j < baseStore.getDepth(idx); j++){
		cudaMemcpy(&scan[(j+moveStore.getNumCh(idx))*moveStore.getNumPoints(idx)],genStore.getGIP(0,j,moveStore.getNumPoints(scanIdx[idx])),moveStore.getNumPoints(idx)*sizeof(float),cudaMemcpyDeviceToHost);
	}
	CudaCheckError();
}


