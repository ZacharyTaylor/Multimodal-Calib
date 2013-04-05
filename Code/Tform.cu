#include "Tform.h"

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


Tform::Tform(size_t sizeTform):
	sizeTform_(sizeTform)
{
	size_t tformMemSize = sizeTform * sizeTform * sizeof(float);
	 CudaSafeCall(cudaMalloc((void**)&d_tform_, tformMemSize));
	 CudaSafeCall(cudaMemset(d_tform_, 0, tformMemSize));
}

Tform::~Tform(void){
	 CudaSafeCall(cudaFree(d_tform_));
}

void Tform::SetTform(float* tform){
	size_t tformMemSize = sizeTform_ * sizeTform_ * sizeof(float);

	 CudaSafeCall(cudaMemcpy(d_tform_, tform, tformMemSize, cudaMemcpyHostToDevice));
}

float* Tform::d_GetTform(void){
	return d_tform_;
}

CameraTform::CameraTform(Camera* cam):
	Tform(CAM_DIM + 1){
	cam_ = cam;
}


void CameraTform::d_Transform(SparseScan* in, SparseScan* out){

	if(out->getNumPoints() < in->getNumPoints()){
		TRACE_ERROR("output is too small to hold inputs points, returning");
		return;
	}
	if(in->getNumDim() != CAM_DIM){
		TRACE_ERROR("camera transform can only operate on a 3d input, returning untransformed points");
		CudaSafeCall(cudaMemcpy(out->GetLocation()->GetGpuPointer(), in->GetLocation()->GetGpuPointer(), in->getNumDim()*in->getNumPoints()*sizeof(float), cudaMemcpyDeviceToDevice));
		return;
	}
	if(cam_ == NULL){
		TRACE_ERROR("camera transform requires a setup camera, returning untransformed points");
		 CudaSafeCall(cudaMemcpy(out->GetLocation()->GetGpuPointer(), in->GetLocation()->GetGpuPointer(), in->getNumDim()*in->getNumPoints()*sizeof(float), cudaMemcpyDeviceToDevice));
		return;
	}

	CameraTransformKernel<<<gridSize(in->getDimSize(0)), BLOCK_SIZE>>>
		(d_tform_, cam_->d_GetCam(), (float*)in->GetLocation()->GetGpuPointer(), (float*)out->GetLocation()->GetGpuPointer(), in->getDimSize(0), cam_->IsPanoramic());
	CudaCheckError();
}

AffineTform::AffineTform(void):
	Tform(AFFINE_DIM + 1){}

void AffineTform::d_Transform(SparseScan* in, SparseScan* out){

	if(out->getNumPoints() < in->getNumPoints()){
		TRACE_ERROR("output is too small to hold inputs points, returning");
		return;
	}
	if(in->getNumDim() != AFFINE_DIM){
		TRACE_ERROR("affine transform can only operate on a 2d input, returning untransformed points");
		CudaSafeCall(cudaMemcpy(out->GetLocation()->GetGpuPointer(), in->GetLocation()->GetGpuPointer(), in->getNumDim()*in->getNumPoints()*sizeof(float), cudaMemcpyDeviceToDevice));
		return;
	}

	AffineTransformKernel<<<gridSize(in->getDimSize(0)), BLOCK_SIZE>>>(d_tform_, (float*)in->GetLocation()->GetGpuPointer(), (float*)out->GetLocation()->GetGpuPointer(), in->getDimSize(0));
	 //CudaSafeCall(cudaMemcpy(out->GetLocation()->GetGpuPointer(), in->GetLocation()->GetGpuPointer(), in->getNumPoints()*sizeof(float), cudaMemcpyDeviceToDevice));
}