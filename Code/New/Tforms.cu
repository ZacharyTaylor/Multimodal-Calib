#include "Tforms.h"
#include "ScanList.h"
#include "ImageList.h"

Camera::Camera(bool panoramic):
	panoramic_(panoramic){
	size_t camMemSize = CAM_WIDTH * CAM_HEIGHT * sizeof(float);
	CudaSafeCall(cudaMalloc((void**)&d_camera_, camMemSize));
	CudaSafeCall(cudaMemset(d_camera_, 0, camMemSize));
}

Camera::~Camera(void){
	 CudaSafeCall(cudaFree(d_camera_));
}

void Camera::SetCam(float* cam){
	size_t tformMemSize = CAM_WIDTH * CAM_HEIGHT * sizeof(float);
	 CudaSafeCall(cudaMemcpy(d_camera_, cam, tformMemSize, cudaMemcpyHostToDevice));
}

float* Camera::d_GetCam(void){
	return d_camera_;
}

bool Camera::IsPanoramic(void){
	return panoramic_;
}

Tforms::Tforms(size_t tformSizeX, size_t tformSizeY):
	tformSizeX_(tformSizeX),
	tformSizeY_(tformSizeY){};

void Tforms::addTforms(thrust::device_vector<float> tformDIn){
	tformD.insert(tformD.end(), tformDIn.begin(), tformDIn.end());
}

void Tforms::addTforms(thrust::host_vector<float> tformDIn){
	tformD.insert(tformD.end(), tformDIn.begin(), tformDIn.end());
}

void Tforms::removeAllTforms(void){
	tformD.clear();
}

float* Tforms::getTformP(void){
	thrust::raw_pointer_cast(&tformD[0]);
}

size_t Tforms::getTformSize(void){
	return tformSizeX_*tformSizeY_;
}

void CameraTforms::removeAllTforms(void){
	tformD.clear();
	camIdx.clear();
}

void CameraTforms::transform(ScanList* in, ScanList* out, ImageList* index, size_t start){

	CameraTransformKernel(
		this->getTformP(),
		index->getTformIdxP(),
		camStore->getCamP(),
		camStore->getPanP(),
		index->getCamIdxP(),
		in->getLP(0),
		in->getLP(1),
		in->getLP(2),
		in->getIdxP(),
		in->getNumScans(),
		out->getLP(0),
		out->getLP(1),
		out->getNumPoints(),
		start
	);


	CameraTransformKernel<<<gridSize(in->getDimSize(0)), BLOCK_SIZE, 0, *stream>>>
		(d_tform_, cam_->d_GetCam(), (float*)in->GetLocation()->GetGpuPointer(), (float*)(*out)->GetLocation()->GetGpuPointer(), in->getDimSize(0), cam_->IsPanoramic());
	CudaCheckError();
}


void AffineTform::transform(SparseScan* in, SparseScan** out, cudaStream_t* stream){

	delete *out;
	*out = new SparseScan(in->getNumDim(), 0, in->getNumPoints());
	(*out)->GetLocation()->AllocateGpu();

	if(in->getNumDim() != AFFINE_DIM){
		TRACE_ERROR("affine transform can only operate on a 2d input, returning untransformed points");
		CudaSafeCall(cudaMemcpy((*out)->GetLocation()->GetGpuPointer(), in->GetLocation()->GetGpuPointer(), in->getNumDim()*in->getNumPoints()*sizeof(float), cudaMemcpyDeviceToDevice));
		return;
	}

	AffineTransformKernel<<<gridSize(in->getDimSize(0)), BLOCK_SIZE, 0, *stream>>>(d_tform_, (float*)in->GetLocation()->GetGpuPointer(), (float*)(*out)->GetLocation()->GetGpuPointer(), in->getDimSize(0));
	 //CudaSafeCall(cudaMemcpy(out->GetLocation()->GetGpuPointer(), in->GetLocation()->GetGpuPointer(), in->getNumPoints()*sizeof(float), cudaMemcpyDeviceToDevice));
}
