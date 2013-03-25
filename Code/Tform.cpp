#include "Tform.h"

Camera::Camera(bool panoramic):
	panoramic_(panoramic){
	size_t camMemSize = CAM_WIDTH * CAM_HEIGHT * sizeof(float);
	cudaMalloc((void**)&d_camera_, camMemSize);
	cudaMemset(d_camera_, 0, camMemSize);
}

Camera::~Camera(void){
	cudaFree(d_camera_);
}

void Camera::SetCam(float* cam){
	size_t tformMemSize = CAM_WIDTH * CAM_HEIGHT * sizeof(float);
	cudaMemcpy(d_camera_, cam, tformMemSize, cudaMemcpyHostToDevice);
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
	size_t tformMemSize = sizeTform_ * sizeTform_ * sizeof(float);
	cudaMalloc((void**)&d_tform_, tformMemSize);
	cudaMemset(d_tform_, 0, tformMemSize);
}

Tform::~Tform(void){
	cudaFree(d_tform_);
}

void Tform::SetTform(float* tform){
	size_t tformMemSize = sizeTform_ * sizeTform_ * sizeof(float);
	cudaMemcpy(d_tform_, tform, tformMemSize, cudaMemcpyHostToDevice);
}

float* Tform::d_GetTform(void){
	return d_tform_;
}

void CameraTform::d_Transform(SparseScan* in, SparseScan* out){

	if(out->getNumPoints() < in->getNumPoints()){
		TRACE_ERROR("output is too small to hold inputs points, returning");
		return;
	}
	if(in->getNumDim() != CAM_DIM){
		TRACE_ERROR("affine transform can only operate on a 3d input, returning untransformed points\n");
		cudaMemcpy(out->GetLocationPointer(), in->GetLocationPointer(), in->getNumPoints()*sizeof(float), cudaMemcpyDeviceToDevice);
		return;
	}
	CameraTransformKernel<<<gridSize(in->getDimSize(0), BLOCK_SIZE>>>(d_tform_, d_cam_->d_getCam(), in->GetLocationPointer(), out->GetLocationPointer(), in->getDimSize(0), cam->isPanoramic());
}

__global__ void CameraTform::CameraTransformKernel(const float* tform, const float* cam, const float* pointsIn, float* pointsOut, const size_t numPoints, const bool panoramic){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numPoints){
		return;
	}

	const float xIn = pointsIn[i + 0*numPoints];
	const float yIn = pointsIn[i + 1*numPoints];
	const float zIn = pointsIn[i + 2*numPoints];

	//transform points
	float x = xIn*tform[0] + yIn*tform[4] + zIn*tform[8] + tform[12];
	float y = xIn*tform[1] + yIn*tform[5] + zIn*tform[9] + tform[13];
	float z = xIn*tform[2] + yIn*tform[6] + zIn*tform[10] + tform[14];

	if((z <= 0) && !panoramic){
		x = -1;
		y = -1;
	}
	else{

		//apply projective camera matrix
		x = cam[0]*x + cam[3]*y + cam[6]*z + cam[9];
		y = cam[1]*x + cam[4]*y + cam[7]*z + cam[10];
		z = cam[2]*x + cam[5]*y + cam[8]*z + cam[11];

		if(panoramic){
			//panoramic camera model
			y = (y/sqrt(z*z + x*x));
			x = atan2(x,z);

		}
		else{
			//pin point camera model
			y = y/z;
			x = x/z;
		}
	}

	//output points
	pointsOut[i + 0*numPoints] = x;
	pointsOut[i + 1*numPoints] = y;
}

AffineTform::AffineTform(void):
	Tform(AFFINE_DIM + 1){}

void AffineTform::d_Transform(SparseScan* in, SparseScan* out){

	if(out->getNumPoints() < in->getNumPoints()){
		TRACE_ERROR("output is too small to hold inputs points, returning");
		return;
	}
	if(in->getNumDim() != AFFINE_DIM){
		TRACE_ERROR("affine transform can only operate on a 2d input, returning untransformed points\n");
		cudaMemcpy(out->GetLocationPointer(), in->GetLocationPointer(), in->getNumPoints()*sizeof(float), cudaMemcpyDeviceToDevice);
		return;
	}

	AffineTransformKernel<<<gridSize(in->getDimSize(0), BLOCK_SIZE>>>(d_tform_, in->GetLocationPointer(), out->GetLocationPointer(), in->getDimSize(0));
}

__global__ void AffineTform::AffineTransformKernel(const float* tform, const float* pointsIn, float* pointsOut, const size_t numPoints){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numPoints){
		return;
	}

	//make it a bit clearer which are x and y points
	const float xIn = pointsIn[i];
	const float yIn = pointsIn[i + numPoints];

	//transform points
	float xOut = xIn*tform[0] + yIn*tform[3] + tform[6];
	float yOut = xIn*tform[1] + yIn*tform[4] + tform[7];

	pointsOut[i] = xOut;
	pointsOut[i + numPoints] = yOut;

}