#include "MatlabCalls.h"

#include "common.h"
#include "trace.h"
#include "Points.h"
#include "Scan.h"
#include "Pair.h"


//global so that matlab never has to see them
SparseScan** moveStore = NULL;
unsigned int numMove = 0;

DenseImage** baseStore = NULL;
unsigned int numBase = 0;

Pair** pairs = NULL;
unsigned int numPairs = 0;

DllExport unsigned int getNumMove(void){
	return numMove;
}

DllExport unsigned int getNumBase(void){
	return numBase;
}

DllExport unsigned int getNumPairs(void){
	return numPairs;
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

	if(pairs != NULL){
		for(unsigned int i = 0; i < numPairs; i++){
			delete pairs[i];
		}
	}	
}

DllExport void initalizeScans(unsigned int numBaseIn, unsigned int numMoveIn, unsigned int numPairsIn){
	
	if((moveStore != NULL) || (baseStore != NULL) || (pairs != NULL)){
		TRACE_INFO("Scans already initalized, clearing and writing new data");
		clearScans();
	}

	numMove = numMoveIn;
	numBase = numBaseIn;
	numPairs = numPairsIn;

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

	pairs = new Pair*[numPairs];
	//setting null so can tell if it is allocated
	for(unsigned int i = 0; i < numPairs; i++){
		pairs[i] = NULL;
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

DllExport const float* getMoveLocs(unsigned int scanNum){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot get scan %i as only %i scans exist",scanNum,numMove);
		return NULL;
	}
	if(moveStore[scanNum] == NULL){
		TRACE_ERROR("Move %i has not been allocated, returning", scanNum);
		return NULL;
	}

	//copy gpu info so that most up to date map is on cpu
	if(moveStore[scanNum]->GetLocation()->GetOnGpu()){
		moveStore[scanNum]->GetLocation()->CpuToGpu();
	}

	const float* out = moveStore[scanNum]->GetLocation()->GetCpuPointer();
	return out;
}

DllExport const float* getMovePoints(unsigned int scanNum){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot get scan %i as only %i scans exist",scanNum,numMove);
		return NULL;
	}
	if(moveStore[scanNum] == NULL){
		TRACE_ERROR("Move %i has not been allocated, returning", scanNum);
		return NULL;
	}

	//copy gpu info so that most up to date map is on cpu
	if(moveStore[scanNum]->getPoints()->GetOnGpu()){
		moveStore[scanNum]->getPoints()->CpuToGpu();
	}

	const float* out = moveStore[scanNum]->getPoints()->GetCpuPointer();
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

DllExport const float* getBaseImage(unsigned int scanNum){
	
	if(scanNum >= numBase){
		TRACE_ERROR("Cannot get image %i as only %i images exist",scanNum,numBase);
		return NULL;
	}
	if(baseStore[scanNum] == NULL){
		TRACE_ERROR("Base %i has not been allocated, returning", scanNum);
		return NULL;
	}

	//copy gpu info so that most up to date map is on cpu
	if(baseStore[scanNum]->getPoints()->GetOnGpu()){
		baseStore[scanNum]->getPoints()->CpuToGpu();
	}

	const float* out = baseStore[scanNum]->getPoints()->GetCpuPointer();
	return out;
}
