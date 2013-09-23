#include "MatCalls.h"

#include "Calib.h"

//global so that matlab never has to see them
Calib* calibStore = NULL;

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

DllExport void initalizeCamera(void){
	checkForCUDA();
	if(calibStore){
		delete calibStore;
	}
	calibStore = new CameraCalib;
}

DllExport void addMovingScan(float* moveLIn, float* moveIIn, unsigned int length, unsigned int numDim, unsigned int numCh){ 

	//copys data (slow and memory inefficient, but easy and memory safe)
	std::vector<thrust::host_vector<float>> scanLIn;
	scanLIn.resize(numDim);
	for( size_t i = 0; i < numDim; i++){
		scanLIn[i].assign(&moveLIn[i*length], moveLIn + length);
	}

	std::vector<thrust::host_vector<float>> scanIIn;
	scanIIn.resize(numCh);
	for( size_t i = 0; i < numCh; i++){
		scanIIn[i].assign(&moveIIn[i*length], moveIIn + length);
	}

	calibStore->addScan(scanLIn, scanIIn);

}

DllExport void addBaseImage(float* baseIn, unsigned int height, unsigned int width, unsigned int depth, unsigned int tformIdx, unsigned int scanIdx){ 

	//copys data (slow and memory inefficient, but easy and memory safe)
	thrust::host_vector<float> imageIn;
	imageIn.assign(baseIn, baseIn + (height*width*depth));

	calibStore->addImage(imageIn, height, width, depth, tformIdx, scanIdx);
}

DllExport void addTform(float* tformIn, unsigned int tformSizeX, unsigned int tformSizeY){ 

	//copys data (slow and memory inefficient, but easy and memory safe)
	thrust::host_vector<float> tIn;
	tIn.assign(tformIn, tformIn + (tformSizeX*tformSizeY));

	calibStore->addTform(tIn, tformSizeX, tformSizeY);
}

DllExport void addCamera(float* cIn, bool panoramic){ 

	//copys data (slow and memory inefficient, but easy and memory safe)
	thrust::host_vector<float> cIn;
	cIn.assign(camIn, camIn + CAM_SIZE_X*CAM_SIZE_Y);

	calibStore->addCamera(cIn, panoramic);
}

DllExport float evalMetric(void){
	return calibStore->evalMetric();
}