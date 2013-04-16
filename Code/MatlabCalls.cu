#include "MatlabCalls.h"

#include "common.h"
#include "trace.h"
#include "Points.h"
#include "Scan.h"
#include "Tform.h"
#include "Metric.h"
#include "Render.h"


//global so that matlab never has to see them
SparseScan** moveStore = NULL;
unsigned int numMove = 0;

DenseImage** baseStore = NULL;
unsigned int numBase = 0;

Camera* camera = NULL;

Tform* tform = NULL;

Metric* metric = NULL;

SparseScan* gen = NULL;

Render render;

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
		delete moveStore;
		moveStore = NULL;
	}	

	if(baseStore != NULL){
		for(unsigned int i = 0; i < numBase; i++){
			delete baseStore[i];
		}
		delete baseStore;
		baseStore = NULL;
	}
	if(gen != NULL){
		delete gen;
		gen = NULL;
	}
}

DllExport void clearTform(void){
	delete tform;
	tform = NULL;
}

DllExport void clearMetric(void){
	delete metric;
	metric = NULL;
}

DllExport void clearRender(void){
}

DllExport void initalizeScans(unsigned int numBaseIn, unsigned int numMoveIn){
	
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

DllExport void setBaseImage(unsigned int scanNum, unsigned int width, unsigned int height, unsigned int numCh, float* base){
	
	if(scanNum >= numBase){
		TRACE_ERROR("Cannot set scan %i as only %i scans exist",scanNum,numBase);
		return;
	}
	if(baseStore[scanNum] != NULL){
		TRACE_INFO("Base %i already allocated, clearing and overwritng with new data", scanNum);
		delete baseStore[scanNum];
		baseStore[scanNum] = NULL;
	}

	baseStore[scanNum] = new DenseImage(width, height, numCh, base);
}

DllExport void setMoveImage(unsigned int scanNum, unsigned int width, unsigned int height, unsigned int numCh, float* move){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot set scan %i as only %i scans exist",scanNum,numMove);
		return;
	}
	if(moveStore[scanNum] != NULL){
		TRACE_INFO("Move %i already allocated, clearing and overwritng with new data", scanNum);
		delete moveStore[scanNum];
		moveStore[scanNum] = NULL;
	}

	size_t* dimSize = new size_t[2];
	dimSize[0] = width;
	dimSize[1] = height;
	float* loc = SparseScan::GenLocation(IMAGE_DIM, dimSize);
	moveStore[scanNum] = new SparseScan(IMAGE_DIM,numCh,height*width, move, loc);
	delete[] loc;
	loc = NULL;
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

	if(imgNum >= numMove){
		TRACE_ERROR("Cannot get image %i as only %i images exist",imgNum,numMove);
		return;
	}

	SparseScan* move = moveStore[imgNum];

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
	
	if((gen == NULL) || (gen->GetLocation() == NULL)){
		TRACE_ERROR("Generated locations has not been allocated, returning");
		return NULL;
	}

	//copy gpu info so that most up to date map is on cpu
	gen->GetLocation()->GpuToCpu();

	float* out = gen->GetLocation()->GetCpuPointer();
	return out;
}

DllExport float* getGenPoints(void){
	
	if((gen == NULL) || (gen->getPoints() == NULL)){
		TRACE_ERROR("Generated points has not been allocated, returning");
		return NULL;
	}

	//copy gpu info so that most up to date map is on cpu
	gen->getPoints()->GpuToCpu();

	float* out = gen->getPoints()->GetCpuPointer();
	return out;
}

DllExport int getGenNumCh(void){
	
	if((gen == NULL)){
		TRACE_ERROR("Generated image has not been allocated, returning");
		return NULL;
	}

	int out = (int)gen->getNumCh();
	return out;
}

DllExport int getGenNumDim(void){
	
	if((gen == NULL)){
		TRACE_ERROR("Generated image has not been allocated, returning");
		return NULL;
	}

	int out = (int)gen->getNumDim();
	return out;
}

DllExport int getGenNumPoints(void){
	
	if((gen == NULL)){
		TRACE_ERROR("Generated image has not been allocated, returning");
		return NULL;
	}

	int out = (int)gen->getNumPoints();
	return out;
}

DllExport void checkCudaErrors(void){
	CudaCheckError();
}

DllExport void genBaseValues(unsigned int baseNum){
	if(baseNum >= numBase){
		TRACE_ERROR("Cannot get base image %i as only %i images exist",baseNum,numBase);
		return;
	}

	if((gen == NULL) || (gen->GetLocation() == NULL)){
		TRACE_ERROR("A generated location is required for interpolation");
		return;
	}

	PointsList* points = gen->getPoints();
	if(points != NULL){
		TRACE_INFO("Clearing generated image ready for new interpolation");
		delete points;
		points = NULL;
	}

	DenseImage* base = baseStore[baseNum];
	gen->changeNumCh(base->getNumCh());

	if(base == NULL){
		TRACE_ERROR("A base image is required to interpolate");
		return;
	}

	//setup generated image
	points = new PointsList(base->getNumCh()*gen->getNumPoints());
	points->AllocateGpu();

	//ensure gen is setup
	if(!gen->GetLocation()->IsOnGpu()){
		gen->GetLocation()->AllocateGpu();
		gen->GetLocation()->CpuToGpu();
	}

	//ensure base is setup
	if(!base->getPoints()->IsOnGpu()){
		base->getPoints()->AllocateGpu();
		base->getPoints()->CpuToGpu();
	}

	base->d_interpolate(gen);
}

DllExport void setupMIMetric(void){
	if(metric != NULL){
		TRACE_INFO("A metric already exists, overwriting it");
		delete metric;
		metric = NULL;
	}

	metric = new MI();
}
DllExport void setupGOMMetric(void){
	if(metric != NULL){
		TRACE_INFO("A metric already exists, overwriting it");
		delete metric;
		metric = NULL;
	}

	metric = new GOM();
}

DllExport void setupLivMetric(void){
	if(metric != NULL){
		TRACE_INFO("A metric already exists, overwriting it");
		delete metric;
		metric = NULL;
	}

	metric = new LIV();
}

DllExport float getMetricVal(unsigned int moveNum){
	if(metric == NULL){
		TRACE_ERROR("No metric setup, returning");
		return 0;
	}

	if(moveNum >= numMove){
		TRACE_ERROR("Cannot get move image %i as only %i images exist",moveNum,numMove);
		return 0;
	}

	SparseScan* move = moveStore[moveNum];
	
	if(move == NULL){
		TRACE_ERROR("A moving image is required");
		return 0;
	}
	if(gen == NULL){
		TRACE_ERROR("A generated image is required");
		return 0;
	}

	//ensure gen is setup
	if(!gen->getPoints()->IsOnGpu()){
		gen->getPoints()->AllocateGpu();
		gen->getPoints()->CpuToGpu();
	}

	//ensure move is setup
	if(!move->getPoints()->IsOnGpu()){
		move->getPoints()->AllocateGpu();
		move->getPoints()->CpuToGpu();
	}

	return metric->EvalMetric(move, gen);
}

DllExport float* outputImage(unsigned int width, unsigned int height){
	if((gen == NULL) || (gen->getPoints() == NULL)){
		TRACE_ERROR("A generated image is required");
		return 0;
	}

	//ensure gen is setup
	if(!gen->getPoints()->IsOnGpu()){
		gen->getPoints()->AllocateGpu();
		gen->getPoints()->CpuToGpu();
	}

	render.GetImage(gen, width, height);
	return render.out_;
}

