#include "Calib.h"
#include <algorithm>
#include <string> 

Calib::Calib(std::string metricType){
	checkForCUDA();
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
	return moveStore.getNumCh(idx);
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

size_t Calib::allocateGenMem(ScanList points, ImageList images, std::vector<std::vector<float*>>& genL, std::vector<std::vector<float*>>& genI, size_t startIdx){
	
	cudaError_t err = cudaSuccess;
	size_t i;

	std::vector<float*> temp;

	for(i = startIdx; i < images.getNumImages(); i++){

		temp.resize(IMAGE_DIM);
		for(size_t j = 0; j < IMAGE_DIM; j++){
			cudaError_t currentErr = cudaMalloc(&temp[j], sizeof(float)*points.getNumPoints(scanIdx[i]));
			if(currentErr != cudaSuccess){
				for(size_t k = 0; k < j; k++){
					cudaFree(temp[k]);
				}
				break;
			}
		}
		genL.push_back(temp);

		temp.resize(images.getDepth(i));
		for(size_t j = 0; j < images.getDepth(i); j++){
			cudaError_t currentErr = cudaMalloc(&temp[j], sizeof(float)*points.getNumPoints(scanIdx[i]));
			if(currentErr != cudaSuccess){
				for(size_t k = 0; k < j; k++){
					cudaFree(temp[k]);
				}
				break;
			}
		}
		genI.push_back(temp);
	}


	return i;
}

void Calib::clearGenMem(std::vector<std::vector<float*>>& genL, std::vector<std::vector<float*>>& genI, size_t startIdx){
	
	size_t i;

	for(i = startIdx; i < genL.size(); i++){
		for(size_t j = 0; j < genL[i].size(); j++){
			cudaFree(genL[i][j]);
		}
	}
	for(i = startIdx; i < genI.size(); i++){
		for(size_t j = 0; j < genI[i].size(); j++){
			cudaFree(genI[i][j]);
		}
	}
	CudaCheckError();
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


void Calib::addCameraIndices(std::vector<size_t>& cameraIdxIn){
	mexErrMsgTxt("Attempted to setup camera for use with non-camera calibration");
	return;
}

void Calib::addCamera(thrust::host_vector<float>& cameraIn, boolean panoramic){
	mexErrMsgTxt("Attempted to setup camera for use with non-camera calibration");
	return;
}

void Calib::generateImage(thrust::device_vector<float>& image, size_t width, size_t height, size_t dilate, size_t idx, bool imageColour){
	return;
}

void Calib::colourScan(float* scan, size_t idx){
	return;
}

CameraCalib::CameraCalib(std::string metricType) : Calib(metricType){}

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

	//cudaEvent_t start, stop;
	//float time;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	
	for(size_t i = 0; i < moveStore.getNumScans(); i+= (genLength+1)){
		genLength = allocateGenMem(moveStore, baseStore, genL, genI, i);
		if(genLength == 0){
			mexErrMsgTxt("Memory allocation for generated scans failed\n");
		}
		
		streams.resize(genLength-i);
		for(size_t j = 0; j < streams.size(); j++){
				//cudaEventRecord(start, 0);
			cudaStreamCreate ( &streams[j]);
				//cudaEventRecord(stop, 0);cudaEventSynchronize(stop);cudaEventElapsedTime(&time, start, stop);mexPrintf ("Time for streams: %f ms\n", time);

				//cudaEventRecord(start, 0);
			tformStore.transform(moveStore, genL[j], cameraStore, tformIdx[i+j], cameraIdx[i+j], scanIdx[i+j], streams[j]);
			cudaDeviceSynchronize();
				//cudaEventRecord(stop, 0);cudaEventSynchronize(stop);cudaEventElapsedTime(&time, start, stop);mexPrintf ("Time for transform: %f ms\n", time);

				//cudaEventRecord(start, 0);
			baseStore.interpolateImage(i+j, scanIdx[i+j], genL[j], genI[j], moveStore.getNumPoints(scanIdx[i+j]), true, streams[j]);
			cudaDeviceSynchronize();
				//cudaEventRecord(stop, 0);cudaEventSynchronize(stop);cudaEventElapsedTime(&time, start, stop);mexPrintf ("Time for interpolation: %f ms\n", time);

				//cudaEventRecord(start, 0);
			out += metric->evalMetric(genI[j], moveStore, scanIdx[i+j], streams[j]);
			cudaDeviceSynchronize();
				//cudaEventRecord(stop, 0);cudaEventSynchronize(stop);cudaEventElapsedTime(&time, start, stop);mexPrintf ("Time for evaluation: %f ms\n", time);
		}
		
		for(size_t j = 0; j< streams.size(); j++){
			cudaStreamDestroy(streams[j]);
		}

		clearGenMem(genL, genI, i);
	}

	return out;
}

void CameraCalib::generateImage(thrust::device_vector<float>& image, size_t width, size_t height, size_t dilate, size_t idx, bool imageColour){

	std::vector<float*> genL;
	std::vector<float*> genI;

	if(imageColour){
		image.resize(baseStore.getDepth(idx)*width*height);
	}
	else{
		image.resize(moveStore.getNumCh(scanIdx[idx])*width*height);
	}

	genL.resize(IMAGE_DIM);
	for(size_t j = 0; j < IMAGE_DIM; j++){
		cudaError_t currentErr = cudaMalloc(&genL[j], sizeof(float)*moveStore.getNumPoints(scanIdx[idx]));
		if(currentErr != cudaSuccess){
			mexErrMsgTxt("Memory allocation error when generating image");
			break;
		}
	}
	if(imageColour){
		genI.resize(baseStore.getDepth(idx));
		for(size_t j = 0; j < baseStore.getDepth(idx); j++){
			cudaError_t currentErr = cudaMalloc(&genI[j], sizeof(float)*moveStore.getNumPoints(scanIdx[idx]));
			if(currentErr != cudaSuccess){
				mexErrMsgTxt("Memory allocation error when generating image");
				break;
			}
		}
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	tformStore.transform(moveStore, genL, cameraStore, tformIdx[idx], cameraIdx[idx], scanIdx[idx], stream);
	cudaDeviceSynchronize();

	if(imageColour){
		baseStore.interpolateImage(idx, scanIdx[idx], genL, genI, moveStore.getNumPoints(scanIdx[idx]), true, stream);
		cudaDeviceSynchronize();

		for(size_t i = 0; i < baseStore.getDepth(idx); i++){
			
			generateOutputKernel<<<gridSize(moveStore.getNumPoints(scanIdx[idx])) ,BLOCK_SIZE>>>(
				genL[0],
				genL[1],
				genI[i],
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
				genL[0],
				genL[1],
				moveStore.getIP(scanIdx[idx],i),
				thrust::raw_pointer_cast(&image[width*height*i]),
				width,
				height,
				moveStore.getNumPoints(scanIdx[idx]),
				dilate);
		}
	}

	cudaStreamDestroy(stream);

	for(size_t j = 0; j < genL.size(); j++){
		cudaFree(genL[j]);
	}
	if(imageColour){
		for(size_t j = 0; j < genI.size(); j++){
			cudaFree(genI[j]);
		}
	}
	CudaCheckError();
}

void CameraCalib::colourScan(float* scan, size_t idx){
	std::vector<float*> genL;
	std::vector<float*> genI;


	genL.resize(IMAGE_DIM);
	for(size_t j = 0; j < IMAGE_DIM; j++){
		cudaError_t currentErr = cudaMalloc(&genL[j], sizeof(float)*moveStore.getNumPoints(scanIdx[idx]));
		if(currentErr != cudaSuccess){
			mexErrMsgTxt("Memory allocation error when generating image");
			break;
		}
	}
	genI.resize(baseStore.getDepth(idx));
	for(size_t j = 0; j < baseStore.getDepth(idx); j++){
		cudaError_t currentErr = cudaMalloc(&genI[j], sizeof(float)*moveStore.getNumPoints(scanIdx[idx]));
		if(currentErr != cudaSuccess){
			mexErrMsgTxt("Memory allocation error when generating image");
			break;
		}
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	tformStore.transform(moveStore, genL, cameraStore, tformIdx[idx], cameraIdx[idx], scanIdx[idx], stream);
	cudaDeviceSynchronize();

	baseStore.interpolateImage(idx, scanIdx[idx], genL, genI, moveStore.getNumPoints(scanIdx[idx]), true, stream);
	cudaDeviceSynchronize();

	cudaStreamDestroy(stream);

	for(size_t j = 0; j < moveStore.getNumDim(idx); j++){
		cudaMemcpy(&scan[j*moveStore.getNumPoints(idx)],moveStore.getLP(idx,j),sizeof(float)*moveStore.getNumPoints(idx),cudaMemcpyDeviceToHost);
	}
	for(size_t j = 0; j < genI.size(); j++){
		cudaMemcpy(&scan[(j+moveStore.getNumDim(idx))*moveStore.getNumPoints(idx)],genI[j],sizeof(float)*moveStore.getNumPoints(idx),cudaMemcpyDeviceToHost);
	}

	for(size_t j = 0; j < genL.size(); j++){
		cudaFree(genL[j]);
	}
	for(size_t j = 0; j < genI.size(); j++){
		cudaFree(genI[j]);
	}
	CudaCheckError();
}

ImageCalib::ImageCalib(std::string metricType) : Calib(metricType){}

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

	std::vector<std::vector<float*>> genL;
	std::vector<std::vector<float*>> genI;

	std::vector<float> metricVal;

	std::vector<cudaStream_t> streams;

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

	size_t genLength = 0;
	float out = 0;

	//cudaEvent_t start, stop;
	//float time;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	
	for(size_t i = 0; i < moveStore.getNumScans(); i+= (genLength+1)){
		genLength = allocateGenMem(moveStore, baseStore, genL, genI, i);
		
		if(genLength == 0){
			mexErrMsgTxt("Memory allocation for generated scans failed\n");
		}
		
		streams.resize(genLength-i);
		for(size_t j = 0; j < streams.size(); j++){
				//cudaEventRecord(start, 0);
			cudaStreamCreate ( &streams[j]);
				//cudaEventRecord(stop, 0);cudaEventSynchronize(stop);cudaEventElapsedTime(&time, start, stop);mexPrintf ("Time for streams: %f ms\n", time);

				//cudaEventRecord(start, 0);
			tformStore.transform(moveStore, genL[j], noCamera, tformIdx[i+j], NULL, scanIdx[i+j], streams[j]);
			cudaDeviceSynchronize();
				//cudaEventRecord(stop, 0);cudaEventSynchronize(stop);cudaEventElapsedTime(&time, start, stop);mexPrintf ("Time for transform: %f ms\n", time);

				//cudaEventRecord(start, 0);
			baseStore.interpolateImage(i+j, scanIdx[i+j], genL[j], genI[j], moveStore.getNumPoints(scanIdx[i+j]), true, streams[j]);
			cudaDeviceSynchronize();
				//cudaEventRecord(stop, 0);cudaEventSynchronize(stop);cudaEventElapsedTime(&time, start, stop);mexPrintf ("Time for interpolation: %f ms\n", time);

				//cudaEventRecord(start, 0);
			out += metric->evalMetric(genI[j], moveStore, scanIdx[i+j], streams[j]);
			cudaDeviceSynchronize();
				//cudaEventRecord(stop, 0);cudaEventSynchronize(stop);cudaEventElapsedTime(&time, start, stop);mexPrintf ("Time for evaluation: %f ms\n", time);
		}
		
		for(size_t j = 0; j< streams.size(); j++){
			cudaStreamDestroy(streams[i]);
		}

		clearGenMem(genL, genI, i);
	}

	return out;
}

void ImageCalib::generateImage(thrust::device_vector<float>& image, size_t width, size_t height, size_t dilate, size_t idx, bool imageColour){

	std::vector<float*> genL;
	std::vector<float*> genI;

	if(imageColour){
		image.resize(baseStore.getDepth(idx)*width*height);
	}
	else{
		image.resize(moveStore.getNumCh(scanIdx[idx])*width*height);
	}

	genL.resize(IMAGE_DIM);
	for(size_t j = 0; j < IMAGE_DIM; j++){
		cudaError_t currentErr = cudaMalloc(&genL[j], sizeof(float)*moveStore.getNumPoints(scanIdx[idx]));
		if(currentErr != cudaSuccess){
			mexErrMsgTxt("Memory allocation error when generating image");
			break;
		}
	}
	if(imageColour){
		genI.resize(baseStore.getDepth(idx));
		for(size_t j = 0; j < baseStore.getDepth(idx); j++){
			cudaError_t currentErr = cudaMalloc(&genI[j], sizeof(float)*moveStore.getNumPoints(scanIdx[idx]));
			if(currentErr != cudaSuccess){
				mexErrMsgTxt("Memory allocation error when generating image");
				break;
			}
		}
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	tformStore.transform(moveStore, genL, noCamera, tformIdx[idx], NULL, scanIdx[idx], stream);
	cudaDeviceSynchronize();

	if(imageColour){
		baseStore.interpolateImage(idx, scanIdx[idx], genL, genI, moveStore.getNumPoints(scanIdx[idx]), true, stream);
		cudaDeviceSynchronize();

		for(size_t i = 0; i < baseStore.getDepth(idx); i++){
			
			generateOutputKernel<<<gridSize(moveStore.getNumPoints(scanIdx[idx])) ,BLOCK_SIZE>>>(
				genL[0],
				genL[1],
				genI[i],
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
				genL[0],
				genL[1],
				moveStore.getIP(scanIdx[idx],i),
				thrust::raw_pointer_cast(&image[width*height*i]),
				width,
				height,
				moveStore.getNumPoints(scanIdx[idx]),
				dilate);
		}
	}

	cudaStreamDestroy(stream);

	for(size_t j = 0; j < genL.size(); j++){
		cudaFree(genL[j]);
	}
	if(imageColour){
		for(size_t j = 0; j < genI.size(); j++){
			cudaFree(genI[j]);
		}
	}
	CudaCheckError();
}

void ImageCalib::colourScan(float* scan, size_t idx){
	std::vector<float*> genL;
	std::vector<float*> genI;


	genL.resize(IMAGE_DIM);
	for(size_t j = 0; j < IMAGE_DIM; j++){
		cudaError_t currentErr = cudaMalloc(&genL[j], sizeof(float)*moveStore.getNumPoints(scanIdx[idx]));
		if(currentErr != cudaSuccess){
			mexErrMsgTxt("Memory allocation error when generating image");
			break;
		}
	}
	genI.resize(baseStore.getDepth(idx));
	for(size_t j = 0; j < baseStore.getDepth(idx); j++){
		cudaError_t currentErr = cudaMalloc(&genI[j], sizeof(float)*moveStore.getNumPoints(scanIdx[idx]));
		if(currentErr != cudaSuccess){
			mexErrMsgTxt("Memory allocation error when generating image");
			break;
		}
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	tformStore.transform(moveStore, genL, noCamera, tformIdx[idx], NULL, scanIdx[idx], stream);
	cudaDeviceSynchronize();

	baseStore.interpolateImage(idx, scanIdx[idx], genL, genI, moveStore.getNumPoints(scanIdx[idx]), true, stream);
	cudaDeviceSynchronize();

	cudaStreamDestroy(stream);

	for(size_t j = 0; j < moveStore.getNumCh(idx); j++){
		cudaMemcpy(&scan[j*moveStore.getNumPoints(idx)],moveStore.getIP(idx,j),moveStore.getNumPoints(idx),cudaMemcpyDeviceToHost);
	}
	for(size_t j = 0; j < genI.size(); j++){
		cudaMemcpy(&scan[(j+moveStore.getNumCh(idx))*moveStore.getNumPoints(idx)],genI[j],moveStore.getNumPoints(idx),cudaMemcpyDeviceToHost);
	}

	for(size_t j = 0; j < genL.size(); j++){
		cudaFree(genL[j]);
	}
	for(size_t j = 0; j < genI.size(); j++){
		cudaFree(genI[j]);
	}
	CudaCheckError();
}


