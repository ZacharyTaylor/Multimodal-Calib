#include "MatCalls.h"
#include "Cameras.h"
#include "Calib.h"

//global so that matlab never has to see them
Calib* calibStore = NULL;

DllExport unsigned int getIfPanoramic(unsigned int idx){
	if(calibStore){
		return calibStore->getIfPanoramic(idx);
	}
	mexErrMsgTxt("Setup not run\n");
	return NULL;
}

DllExport unsigned int getNumPoints(unsigned int idx){
	if(calibStore){
		return calibStore->getNumPoints(idx);
	}
	mexErrMsgTxt("Setup not run\n");
	return NULL;
}

DllExport unsigned int getNumDim(unsigned int idx){
	if(calibStore){
		return calibStore->getNumDim(idx);
	}
	mexErrMsgTxt("Setup not run\n");
	return NULL;
}

DllExport unsigned int getImageDepth(unsigned int idx){
	if(calibStore){
		return calibStore->getImageDepth(idx);
	}
	mexErrMsgTxt("Setup not run\n");
	return NULL;
}

DllExport unsigned int getNumCh(unsigned int idx){
	if(calibStore){
		return calibStore->getNumCh(idx);
	}
	mexErrMsgTxt("Setup not run\n");
	return NULL;
}

DllExport unsigned int getNumImages(void){
	if(calibStore){
		return calibStore->getNumImages();
	}
	mexErrMsgTxt("Setup not run\n");
	return NULL;
}

DllExport unsigned int getImageWidth(unsigned int idx){
	if(calibStore){
		return calibStore->getImageWidth(idx);
	}
	mexErrMsgTxt("Setup not run\n");
	return NULL;
}

DllExport unsigned int getImageHeight(unsigned int idx){
	if(calibStore){
		return calibStore->getImageHeight(idx);
	}
	mexErrMsgTxt("Setup not run\n");
	return NULL;
}


DllExport void clearScans(void){
	if(calibStore){
		calibStore->clearScans();
	}
}

DllExport void clearImages(void){
	if(calibStore){
		calibStore->clearImages();
	}
}

DllExport void clearTforms(void){
	if(calibStore){
		calibStore->clearTforms();
	}
}

DllExport void clearExtras(void){
	if(calibStore){
		calibStore->clearExtras();
	}
}

DllExport void clearEverything(void){
	delete calibStore;
	calibStore = NULL;
}

DllExport void clearIndices(void){
	if(calibStore){
		calibStore->clearIndices();
	}
}

DllExport void initalizeCamera(void){
	if(calibStore){
		delete calibStore;
	}
	calibStore = new CameraCalib("test");
}

DllExport void initalizeImage(void){
	if(calibStore){
		delete calibStore;
	}
	calibStore = new ImageCalib("test");
}

DllExport void addMovingScan(float* moveLIn, float* moveIIn, unsigned int length, unsigned int numDim, unsigned int numCh){ 

	//copys data (slow and memory inefficient, but easy and memory safe)
	std::vector<thrust::host_vector<float>> scanLIn;
	scanLIn.resize(numDim);
	for( size_t i = 0; i < numDim; i++){
		scanLIn[i].assign(&moveLIn[i*length], &moveLIn[i*length] + length);
	}

	std::vector<thrust::host_vector<float>> scanIIn;
	scanIIn.resize(numCh);
	for( size_t i = 0; i < numCh; i++){
		scanIIn[i].assign(&moveIIn[i*length], &moveIIn[i*length] + length);
	}

	calibStore->addScan(scanLIn, scanIIn);

}

DllExport void addBaseImage(float* baseIn, unsigned int height, unsigned int width, unsigned int depth){ 

	//copys data (slow and memory inefficient, but easy and memory safe)
	thrust::host_vector<float> imageIn;
	imageIn.assign(baseIn, baseIn + (height*width*depth));

	calibStore->addImage(imageIn, height, width, depth);
}

DllExport void addTform(float* tformIn, unsigned int tformSizeX, unsigned int tformSizeY){ 

	//copys data (slow and memory inefficient, but easy and memory safe)
	thrust::host_vector<float> tIn;
	tIn.assign(tformIn, tformIn + (tformSizeX*tformSizeY));

	((CameraCalib*)calibStore)->addTform(tIn);
}

DllExport void addCamera(float* camIn, bool panoramic){ 

	//copys data (slow and memory inefficient, but easy and memory safe)
	thrust::host_vector<float> cIn;
	cIn.assign(camIn, camIn + CAM_WIDTH*CAM_HEIGHT);

	calibStore->addCamera(cIn, panoramic);
}

DllExport void addTformIndex(unsigned int* indexIn, unsigned int length){
	std::vector<size_t> index;
	index.assign(indexIn, indexIn + length);
	calibStore->addTformIndices(index);
}

DllExport void addScanIndex(unsigned int* indexIn, unsigned int length){
	std::vector<size_t> index;
	index.assign(indexIn, indexIn + length);
	calibStore->addScanIndices(index);
}

DllExport void addCameraIndex(unsigned int* indexIn, unsigned int length){
	std::vector<size_t> index;
	index.assign(indexIn, indexIn + length);
	calibStore->addCameraIndices(index);
}

DllExport void setupSSDMetric(void){
	calibStore->setSSDMetric();
}

DllExport void setupGOMMetric(void){
	calibStore->setGOMMetric();
}

DllExport void setupGOMSMetric(void){
	calibStore->setGOMSMetric();
}

DllExport void setupMIMetric(void){
	calibStore->setMIMetric();
}

DllExport void setupNMIMetric(void){
	calibStore->setNMIMetric();
}

DllExport float evalMetric(void){
	return calibStore->evalMetric();
}

DllExport void outputImage(float* image, unsigned int width, unsigned int height, unsigned int moveNum, unsigned int dilate, bool imageColour){
	if(calibStore){
		thrust::device_vector<float> devImage;
		calibStore->generateImage(devImage, width, height, dilate, moveNum, imageColour);
		thrust::copy(devImage.begin(), devImage.end(), image);
	}
}

DllExport void colourScan(float* scan, unsigned int moveNum){
	if(calibStore){
		calibStore->colourScan(scan, moveNum);
	}
}