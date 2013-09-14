#include "MatlabCalls.h"

#include "common.h"
#include "trace.h"
#include "Points.h"
#include "Scan.h"
#include "Tform.h"
#include "Metric.h"
#include "Render.h"
#include "Setup.h"

#include <atomic>
#include <thread>


//global so that matlab never has to see them
std::vector<SparseScan*> moveStore;
unsigned int numMove = 0;

std::vector<DenseImage*> baseStore;
unsigned int numBase = 0;

Metric* metric = NULL;
Camera* camera = NULL;

std::vector<SparseScan*> genStore;
std::vector<Tform*> tformStore;
std::vector<cudaStream_t> streams;
unsigned int numTforms = 0;

Render render;

DllExport unsigned int getNumMove(void){
	return numMove;
}

DllExport unsigned int getNumBase(void){
	return numBase;
}

DllExport void clearScans(void){
	while(moveStore.size()){
		delete moveStore.back();
		moveStore.pop_back();
	}
	while(baseStore.size()){
		delete baseStore.back();
		baseStore.pop_back();
	}
	while(genStore.size()){
		delete genStore.back();
		genStore.pop_back();
	}
}

DllExport void clearTform(void){
	tformStore.clear();
	streams.clear();
}

DllExport void clearMetric(void){
	delete metric;
	metric = NULL;
}

DllExport void clearRender(void){
}

DllExport void initalizeScans(unsigned int numBaseIn, unsigned int numMoveIn, unsigned int numTformsIn){
	
	numMove = numMoveIn;
	numBase = numBaseIn;
	numTforms = numTformsIn;

	while(moveStore.size() < numMove){
		moveStore.push_back(NULL);
	}
	while(moveStore.size() > numMove){
		moveStore.pop_back();
	}

	while(baseStore.size() < numBase){
		baseStore.push_back(NULL);
	}
	while(baseStore.size() > numBase){
		baseStore.pop_back();
	}

	while(genStore.size() < numTforms){
		genStore.push_back(NULL);
	}
	while(genStore.size() > numTforms){
		genStore.pop_back();
	}
}

DllExport void setNumTforms(unsigned int numTformsIn){
	numTforms = numTformsIn;

	while(genStore.size() < numTforms){
		genStore.push_back(NULL);
	}
	while(genStore.size() > numTforms){
		genStore.pop_back();
	}

	while(streams.size() < numTforms){
		streams.push_back(NULL);
		cudaStreamCreate(&(streams.back()));
	}
	while(streams.size() > numTforms){
		streams.pop_back();
	}
}

DllExport void setBaseImage(unsigned int scanNum, unsigned int width, unsigned int height, unsigned int numCh, float* base){
	
	if(scanNum >= numBase){
		TRACE_ERROR("Cannot set scan %i as only %i scans exist",scanNum,numBase);
		return;
	}

	delete baseStore[scanNum];
	baseStore[scanNum] = new DenseImage(width, height, numCh, base);
}

DllExport void setMoveImage(unsigned int scanNum, unsigned int width, unsigned int height, unsigned int numCh, float* move){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot set scan %i as only %i scans exist",scanNum,numMove);
		return;
	}
	delete moveStore[scanNum];
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

	delete moveStore[scanNum];
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
	bool panBool = (panoramic != 0)?true:false;
	delete camera;
	camera = new Camera(panBool);
}

DllExport void setupTformAffine(void){
	tformStore.assign(numTforms,NULL);
	for(int i = 0; i < numTforms; i++){
		tformStore[i] = new AffineTform();
	}
}

DllExport void setupCameraTform(void){
	
	if(camera == NULL){
			TRACE_WARNING("Camera has not been set up, tform will be unable to run until this is performed");
	}

	while(tformStore.size() < numTforms){
		tformStore.push_back(new CameraTform(camera));
	}
	while(tformStore.size() > numTforms){
		delete tformStore.back();
		tformStore.pop_back();
	}
}

DllExport void setCameraMatrix(float* camMat){
	if(camera == NULL){
		TRACE_ERROR("Camera not setup, returning");
		return;
	}
	camera->SetCam(camMat);
}

DllExport void setTformMatrix(float* tMat, unsigned int tformIdx){
	if(tformStore[tformIdx] == NULL){
		TRACE_ERROR("Tform not setup, returning");
		return;
	}
	tformStore[tformIdx]->SetTform(tMat);
}

void threadTform(Tform* tform, SparseScan* move, SparseScan** gen, cudaStream_t* stream){
	tform->d_Transform(move,gen,stream);
}

DllExport void transform(unsigned int imgNum){

	if(imgNum >= numMove){
		TRACE_ERROR("Cannot get image %i as only %i images exist",imgNum,numMove);
		return;
	}

	SparseScan* move = moveStore[imgNum];

	if(move == NULL){
		TRACE_ERROR("A moving image is required to transform");
		return;
	}

	//ensure move is setup
	if(!move->GetLocation()->IsOnGpu()){
		move->GetLocation()->AllocateGpu();
		move->GetLocation()->CpuToGpu();
	}

	//spawn threads
	std::vector<std::thread> threads;
	for(int i = 0; i < numTforms; i++){
		threads.push_back(std::thread(threadTform,tformStore[i],move,&genStore[i],&streams[i]));
	}

	//collect threads
	for(auto& thread : threads){
        thread.join();
	}
}

DllExport float* getGenLocs(unsigned int idx){
	
	if(idx >= numTforms){
		TRACE_ERROR("Cannot get image %i as only %i images exist",idx,numTforms);
		return 0;
	}

	SparseScan* gen = genStore[idx];

	if((gen == NULL) || (gen->GetLocation() == NULL)){
		TRACE_ERROR("Generated locations has not been allocated, returning");
		return NULL;
	}

	//copy gpu info so that most up to date map is on cpu
	gen->GetLocation()->GpuToCpu();

	float* out = gen->GetLocation()->GetCpuPointer();
	return out;
}

DllExport float* getGenPoints(unsigned int idx){
	
	if(idx >= numTforms){
		TRACE_ERROR("Cannot get image %i as only %i images exist",idx,numTforms);
		return 0;
	}

	SparseScan* gen = genStore[idx];

	if((gen == NULL) || (gen->getPoints() == NULL)){
		TRACE_ERROR("Generated points has not been allocated, returning");
		return NULL;
	}

	//copy gpu info so that most up to date map is on cpu
	gen->getPoints()->GpuToCpu();

	float* out = gen->getPoints()->GetCpuPointer();
	return out;
}

DllExport int getGenNumCh(unsigned int idx){
	
	if(idx >= numTforms){
		TRACE_ERROR("Cannot get image %i as only %i images exist",idx,numTforms);
		return 0;
	}

	SparseScan* gen = genStore[idx];

	if((gen == NULL)){
		TRACE_ERROR("Generated image has not been allocated, returning");
		return NULL;
	}

	int out = (int)gen->getNumCh();
	return out;
}

DllExport int getGenNumDim(unsigned int idx){
	
	if(idx >= numTforms){
		TRACE_ERROR("Cannot get image %i as only %i images exist",idx,numTforms);
		return 0;
	}

	SparseScan* gen = genStore[idx];

	if((gen == NULL)){
		TRACE_ERROR("Generated image has not been allocated, returning");
		return NULL;
	}

	int out = (int)gen->getNumDim();
	return out;
}

DllExport int getGenNumPoints(unsigned int idx){
	
	if(idx >= numTforms){
		TRACE_ERROR("Cannot get image %i as only %i images exist",idx,numTforms);
		return 0;
	}

	SparseScan* gen = genStore[idx];

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

void threadInterpolate(DenseImage* base, SparseScan** gen, cudaStream_t* stream){
	base->d_interpolate(gen, stream);
}

DllExport void genBaseValues(unsigned int baseNum){
	if(baseNum >= numBase){
		TRACE_ERROR("Cannot get base image %i as only %i images exist",baseNum,numBase);
		return;
	}

	DenseImage* base = baseStore[baseNum];
	if(base == NULL){
		TRACE_ERROR("A base image is required to interpolate");
		return;
	}

	//ensure base is setup
	if(!base->getPoints()->IsOnGpu()){
		base->getPoints()->AllocateGpu();
		base->getPoints()->CpuToGpu();
	}

	//spawn threads
	std::vector<std::thread> threads;
	for(int i = 0; i < numTforms; i++){
		threads.push_back(std::thread(threadInterpolate,base,&genStore[i],&streams[i]));
	}

	//collect threads
	for(auto& thread : threads){
        thread.join();
	}
}

DllExport void replaceMovePoints(unsigned int scanNum, unsigned int genNum){
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot get move image %i as only %i images exist",scanNum,numMove);
		return;
	}
	if(genNum >= numTforms){
		TRACE_ERROR("Cannot get generated scan %i as only %i scans exist",genNum,numTforms);
		return;
	}

	SparseScan* move = moveStore[scanNum];
	SparseScan* gen = genStore[genNum];

	//ensure move is setup
	if(!move->getPoints()->IsOnGpu()){
		move->getPoints()->AllocateGpu();
		move->getPoints()->CpuToGpu();
	}
	
	move->swapPoints(gen);
}

DllExport void setupSSDMetric(void){
	delete metric;
	metric = new SSD();
}
DllExport void setupMIMetric(unsigned int numBins){
	delete metric;
	metric = new MI(numBins);
}
DllExport void setupGOMMetric(void){
	delete metric;
	metric = new GOM();
}

DllExport void setupLIVMetric(float* avImg, unsigned int width, unsigned int height){
	delete metric;
	metric = new LIV(avImg, width, height);
}

void threadEval(Metric* metric, SparseScan* move, SparseScan* gen, float* value, cudaStream_t* stream){
	metric->EvalMetric(move,gen,value,stream);
}
DllExport void getMetricVal(unsigned int moveNum, float* valuesOut){
	if(metric == NULL){
		TRACE_ERROR("No metric setup, returning");
		return;
	}

	if(moveNum >= numMove){
		TRACE_ERROR("Cannot get move image %i as only %i images exist",moveNum,numMove);
		return;
	}

	SparseScan* move = moveStore[moveNum];
	
	//ensure move is setup
	if(!move->getPoints()->IsOnGpu()){
		move->getPoints()->AllocateGpu();
		move->getPoints()->CpuToGpu();
	}

	//spawn threads
	std::vector<std::thread> threads;
	std::vector<float> value(numTforms,0);
	for(int i = 0; i < numTforms; i++){
		threads.push_back(std::thread(threadEval,metric,move,genStore[i],&value[i],&streams[i]));
	}

	//collect threads
	for(auto& thread : threads){
        thread.join();
	}

	//collect results
	for(int i = 0; i < numTforms; i++){
		valuesOut[i] = value[i];
	}

	return;
}

DllExport float* outputImage(unsigned int width, unsigned int height, unsigned int moveNum, unsigned int dilate){
	
	SparseScan* gen = genStore[0];

	if(gen == NULL){
		TRACE_ERROR("A generated image is required");
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

	render.GetImage(move->getPoints(), gen->GetLocation(), move->getNumPoints(), width, height, move->getNumCh(), dilate);
	return render.out_;
}

DllExport float* outputImageGen(unsigned int width, unsigned int height, unsigned int dilate){

	SparseScan* gen = genStore[0];

	if(gen == NULL){
		TRACE_ERROR("A generated image is required");
		return 0;
	}

	render.GetImage(gen->getPoints(), gen->GetLocation(), gen->getNumPoints(), width, height, gen->getNumCh(), dilate);
	return render.out_;
}

DllExport void setupCUDADevices(void){
	checkForCUDA();
}
