#include "MatlabCalls.h"

#include "common.h"
#include "trace.h"
#include "Points.h"
#include "Scan.h"
#include "Tform.h"


//global so that matlab never has to see them
SparseScan** moveStore = NULL;
unsigned int numMove = 0;

DenseImage** baseStore = NULL;
unsigned int numBase = 0;

Camera* camera = NULL;

Tform* tform = NULL;

SparseScan* gen = NULL;

DllExport unsigned int getNumMove(void){
	return numMove;
}

DllExport unsigned int getNumBase(void){
	return numBase;
}

DllExport void clearScans(void){
	if(moveStore != NULL){
		for(unsigned int i = 0; i < numMove; i++){
			delete moveStore[i];
		}
	}	

	if(baseStore != NULL){
		for(unsigned int i = 0; i < numBase; i++){
			delete baseStore[i];
		}
	}	
}

DllExport void initalizeScans(unsigned int numBaseIn, unsigned int numMoveIn, unsigned int numPairsIn){
	
	if((moveStore != NULL) || (baseStore != NULL)){
		TRACE_INFO("Scans already initalized, clearing and writing new data");
		clearScans();
	}

	numMove = numMoveIn;
	numBase = numBaseIn;

	baseStore = new DenseImage*[numBase];
	
	//setting null so can tell if it is allocated
	for(unsigned int i = 0; i < numBase; i++){
		baseStore[i] = NULL;
	}

	moveStore = new SparseScan*[numMove];
	
	//setting null so can tell if it is allocated
	for(unsigned int i = 0; i < numMove; i++){
		moveStore[i] = NULL;
	}
}

DllExport void setBaseImage(unsigned int scanNum, unsigned int height, unsigned int width, unsigned int numCh, float* base){
	
	if(scanNum >= numBase){
		TRACE_ERROR("Cannot set scan %i as only %i scans exist",scanNum,numBase);
		return;
	}
	if(baseStore[scanNum] != NULL){
		TRACE_INFO("Base %i already allocated, clearing and overwritng with new data", scanNum);
		delete baseStore[scanNum];
		baseStore[scanNum] = NULL;
	}

	baseStore[scanNum] = new DenseImage(height, width, numCh, base);
}

DllExport void setMoveImage(unsigned int scanNum, unsigned int height, unsigned int width, unsigned int numCh, float* move){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot set scan %i as only %i scans exist",scanNum,numMove);
		return;
	}
	if(moveStore[scanNum] != NULL){
		TRACE_INFO("Move %i already allocated, clearing and overwritng with new data", scanNum);
		delete moveStore[scanNum];
		moveStore[scanNum] = NULL;
	}

	size_t dimSize[2] = {height, width};
	float* loc = SparseScan::GenLocation(IMAGE_DIM, dimSize);
	moveStore[scanNum] = new SparseScan(IMAGE_DIM,numCh,height*width, move, loc);
}

DllExport void setMoveScan(unsigned int scanNum, unsigned int numDim, unsigned int numCh, unsigned int numPoints, float* move){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot set scan %i as only %i scans exist",scanNum,numMove);
		return;
	}
	if(moveStore[scanNum] != NULL){
		TRACE_INFO("Move %i already allocated, clearing and overwritng with new data", scanNum);
		delete moveStore[scanNum];
		moveStore[scanNum] = NULL;
	}

	moveStore[scanNum] = new SparseScan(numDim,numCh,numPoints,&move[numDim*numPoints],move);
}

DllExport float* getMoveLocs(unsigned int scanNum){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot get scan %i as only %i scans exist",scanNum,numMove);
		return NULL;
	}
	if(moveStore[scanNum] == NULL){
		TRACE_ERROR("Move %i has not been allocated, returning", scanNum);
		return NULL;
	}

	//copy gpu info so that most up to date map is on cpu
	if(moveStore[scanNum]->GetLocation()->IsOnGpu()){
		moveStore[scanNum]->GetLocation()->GpuToCpu();
	}

	float* out = moveStore[scanNum]->GetLocation()->GetCpuPointer();
	return out;
}

DllExport float* getMovePoints(unsigned int scanNum){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot get scan %i as only %i scans exist",scanNum,numMove);
		return NULL;
	}
	if(moveStore[scanNum] == NULL){
		TRACE_ERROR("Move %i has not been allocated, returning", scanNum);
		return NULL;
	}

	//copy gpu info so that most up to date map is on cpu
	if(moveStore[scanNum]->getPoints()->IsOnGpu()){
		moveStore[scanNum]->getPoints()->GpuToCpu();
	}

	float* out = moveStore[scanNum]->getPoints()->GetCpuPointer();
	return out;
}

DllExport int getMoveNumCh(unsigned int scanNum){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot get scan %i as only %i scans exist",scanNum,numMove);
		return NULL;
	}
	if(moveStore[scanNum] == NULL){
		TRACE_ERROR("Move %i has not been allocated, returning", scanNum);
		return NULL;
	}

	int out = (int)moveStore[scanNum]->getNumCh();
	return out;
}

DllExport int getMoveNumDim(unsigned int scanNum){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot get scan %i as only %i scans exist",scanNum,numMove);
		return NULL;
	}
	if(moveStore[scanNum] == NULL){
		TRACE_ERROR("Move %i has not been allocated, returning", scanNum);
		return NULL;
	}

	int out = (int)moveStore[scanNum]->getNumDim();
	return out;
}

DllExport int getMoveNumPoints(unsigned int scanNum){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot get scan %i as only %i scans exist",scanNum,numMove);
		return NULL;
	}
	if(moveStore[scanNum] == NULL){
		TRACE_ERROR("Move %i has not been allocated, returning", scanNum);
		return NULL;
	}

	int out = (int)moveStore[scanNum]->getNumPoints();
	return out;
}

DllExport int getBaseDim(unsigned int scanNum, unsigned int dim){
	
	if(dim >= IMAGE_DIM){
		TRACE_ERROR("Cannot get dimension %i as image is only 2d",dim);
		return NULL;
	}
	if(scanNum >= numBase){
		TRACE_ERROR("Cannot get image %i as only %i images exist",scanNum,numBase);
		return NULL;
	}
	if(baseStore[scanNum] == NULL){
		TRACE_ERROR("Base %i has not been allocated, returning", scanNum);
		return NULL;
	}

	int out = (int)baseStore[scanNum]->getDimSize(dim);
	return out;
}

DllExport int getBaseNumCh(unsigned int scanNum){
	
	if(scanNum >= numBase){
		TRACE_ERROR("Cannot get image %i as only %i images exist",scanNum,numBase);
		return NULL;
	}
	if(baseStore[scanNum] == NULL){
		TRACE_ERROR("Base %i has not been allocated, returning", scanNum);
		return NULL;
	}

	int out = (int)baseStore[scanNum]->getNumCh();
	return out;
}

DllExport float* getBaseImage(unsigned int scanNum){
	
	if(scanNum >= numBase){
		TRACE_ERROR("Cannot get image %i as only %i images exist",scanNum,numBase);
		return NULL;
	}
	if(baseStore[scanNum] == NULL){
		TRACE_ERROR("Base %i has not been allocated, returning", scanNum);
		return NULL;
	}

	//copy gpu info so that most up to date map is on cpu
	if(baseStore[scanNum]->getPoints()->IsOnGpu()){
		baseStore[scanNum]->getPoints()->GpuToCpu();
	}

	float* out = baseStore[scanNum]->getPoints()->GetCpuPointer();
	return out;
}

DllExport void setupCamera(int panoramic){
	if(camera != NULL){
		TRACE_INFO("Camera already setup, clearing");
		delete camera;
		camera = NULL;
	}

	bool panBool = (panoramic != 0)?true:false;
	camera = new Camera(panBool);
}

DllExport void setupTformAffine(void){
	if(tform != NULL){
		TRACE_INFO("Tform already setup, clearing");
		delete tform;
		tform = NULL;
	}
	tform = new AffineTform();
}

DllExport void setupCameraTform(void){
	if(tform != NULL){
		TRACE_INFO("Tform already setup, clearing");
		delete tform;
		tform = NULL;
	}

	if(camera == NULL){
		TRACE_WARNING("Camera has not been set up, tform will be unable to run until this is performed");
	}

	tform = new CameraTform(camera);
}

DllExport void setCameraMatrix(float* camMat){
	if(camera == NULL){
		TRACE_ERROR("Camera not setup, returning");
		return;
	}
	camera->SetCam(camMat);
}

DllExport void setTformMatrix(float* tMat){
	if(tform == NULL){
		TRACE_ERROR("Tform not setup, returning");
		return;
	}
	tform->SetTform(tMat);
}

DllExport void transform(unsigned int imgNum){
	SparseScan* move = moveStore[imgNum];

	if(imgNum >= numMove){
		TRACE_ERROR("Cannot get image %i as only %i images exist",imgNum,numMove);
		return;
	}

	if(gen != NULL){
		TRACE_INFO("Clearing generated image ready for new transform");
		delete gen;
		gen = NULL;
	}
	
	if(move == NULL){
		TRACE_ERROR("A moving image is required to transform");
		return;
	}

	//setup generated image
	gen = new SparseScan(move->getNumDim(), 0, move->getNumPoints());
	gen->GetLocation()->AllocateGpu();

	//ensure move is setup
	if(!move->GetLocation()->IsOnGpu()){
		move->GetLocation()->AllocateGpu();
		move->GetLocation()->CpuToGpu();
	}

	tform->d_Transform(move, gen);
}


DllExport float* getGenLocs(void){
	
	if(gen == NULL){
		TRACE_ERROR("No image Generated, returning");
		return NULL;
	}

	//copy gpu info so that most up to date map is on cpu
	gen->GetLocation()->GpuToCpu();
	
	float* out = gen->GetLocation()->GetCpuPointer();
	return out;
}

DllExport void checkCudaErrors(void){
	CudaCheckError();
}
