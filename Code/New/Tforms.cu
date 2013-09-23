#include "Tforms.h"
#include "ScanList.h"
#include "ImageList.h"
#include "Kernels.h"

void Tforms::addTforms(thrust::device_vector<float> tformDIn, size_t tformSizeX, size_t tformSizeY){
	if(tformDIn.size() != (tformSizeX*tformSizeY)){
		std::cerr << "Error input tform matricies must be same size as given dimensions in size. Returning without setting\n";
		return;
	}
	tform tformIn;
	tformD.push_back(tformIn);
	tformD.back().tform = tformDIn;
	tformD.back().tformSizeX = tformSizeX;
	tformD.back().tformSizeY = tformSizeY;
}

void Tforms::addTforms(thrust::host_vector<float> tformDIn, size_t tformSizeX, size_t tformSizeY){
	if(tformDIn.size() != (tformSizeX*tformSizeY)){
		std::cerr << "Error input tform matricies must be same size as given dimensions in size. Returning without setting\n";
		return;
	}
	tform tformIn;
	tformD.push_back(tformIn);
	tformD.back().tform = tformDIn;
	tformD.back().tformSizeX = tformSizeX;
	tformD.back().tformSizeY = tformSizeY;
}

void Tforms::removeAllTforms(void){
	tformD.clear();
}

float* Tforms::getTformP(size_t idx){
	if(tformD.size() > idx){
		std::cerr << "Cannot get pointer to element " << idx << " as only " << tformD.size() << " elements exist. Returning NULL\n";
		return NULL;
	}
	return thrust::raw_pointer_cast(&(tformD[idx].tform[0]));
}

size_t Tforms::getTformSize(size_t idx){
	if(tformD.size() > idx){
		std::cerr << "Cannot get element " << idx << " as only " << tformD.size() << " elements exist. Returning 0\n";
		return 0;
	}
	return (tformD[idx].tformSizeX * tformD[idx].tformSizeY);
}

void CameraTforms::addTforms(thrust::device_vector<float> tformDIn){
	if(tformDIn.size() != 16){
		std::cerr << "Error input tform matricies must be same size as given dimensions in size. Returning without setting\n";
		return;
	}
	tform tformIn;
	tformD.push_back(tformIn);
	tformD.back().tform = tformDIn;
	tformD.back().tformSizeX = 4;
	tformD.back().tformSizeY = 4;
}

void CameraTforms::addTforms(thrust::host_vector<float> tformDIn){
	if(tformDIn.size() != 16){
		std::cerr << "Error input tform matricies must be same size as given dimensions in size. Returning without setting\n";
		return;
	}
	tform tformIn;
	tformD.push_back(tformIn);
	tformD.back().tform = tformDIn;
	tformD.back().tformSizeX = 4;
	tformD.back().tformSizeY = 4;
}

void CameraTforms::transform(ScanList* scansIn, std::vector<float*> locOut, Cameras* cam, size_t tformIdx, size_t camIdx, size_t scanIdx, cudaStream_t stream){

	CameraTransformKernel<<<gridSize(scansIn->getNumPoints(scanIdx)), BLOCK_SIZE, 0, stream>>>(
		this->getTformP(tformIdx),
		cam->getCamP(camIdx),
		cam->getPanoramic(camIdx),
		scansIn->getLP(scanIdx,0),
		scansIn->getLP(scanIdx,1),
		scansIn->getLP(scanIdx,2),
		scansIn->getNumPoints(scanIdx),
		locOut[0],
		locOut[1]);

	CudaCheckError();
}

void AffineTforms::addTforms(thrust::host_vector<float> tformDIn){
	tform tformIn;
	tformD.push_back(tformIn);
	tformD.back().tform = tformDIn;
	tformD.back().tformSizeX = 3;
	tformD.back().tformSizeY = 3;
}

void AffineTforms::addTforms(thrust::device_vector<float> tformDIn){
	tform tformIn;
	tformD.push_back(tformIn);
	tformD.back().tform = tformDIn;
	tformD.back().tformSizeX = 3;
	tformD.back().tformSizeY = 3;
}

void AffineTforms::transform(ScanList* in, std::vector<float*> out, cudaStream_t* stream){

}
