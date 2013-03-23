#include "GpuScan.cpp"

#define CAM_SIZE 12

class Tform {
protected:
	float* d_tform_;
	size_t sizeTform_;

public:
	virtual SparseScan transform(Scan in) = 0;

	Tform(size_t sizeTform)

	void setTform(float* tform){
		size_t tformMemSize = sizeTform_ * sizeTform_ * sizeof(float);
	cudaMalloc((void**)&d_tform_, tformMemSize);
	cudaMemcpy(d_tform_, tform, tformMemSize, cudaMemcpyHostToDevice);

}

class AffineTform: public Tform {
public:
	SparseScan transform(DenseScan in, float* tformMat){

	}
}

class PinCameraTform: public Tform {
public:
	GpuSparseScan transform(GpuSparseScan in, float* tform, float* cam){

	//Allocate generated points memory
	float* location = new float[in.getNumPoints()];

	//get gpu copy of transform and cam
	float* d_tform;
	size_t tformSize = in.getDimSize() * in.getDimSize() * sizeof(float);
	cudaMalloc((void**)&d_tform, tformSize);
	cudaMemcpy(d_tform, tform, tformSize, cudaMemcpyHostToDevice);

	float* d_cam;
	size_t camSize = CAM_SIZE * sizeof(float);
	cudaMalloc((void**)&d_cam, camSize);
	cudaMemcpy(d_cam, cam, camSize, cudaMemcpyHostToDevice);

	unsigned int nBlocks = ((int)(ceil(float(in.getNumPoints()/BLOCK_SIZE))));
	pointsTransformKernel<<<gridSize(self->d_move.xsize) ,BLOCK_SIZE>>>
		(d_tform, d_cam, , self->genMove, panoramic);
	
	//release gpu copy
	cudaFree(d_tform);
	cudaFree(d_cam);

	}

private:
	__global__ void pointsTransformKernel(float* tform, float* cam, const float* pointsIn, float* pointsOut, size_t* dimSize, const bool panoramic){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= dimSize[0]){
		return;
	}

	const float xIn = pointsIn[i];
	const float yIn = pointsIn[i + 1*dimSize[0]];
	const float zIn = pointsIn[i + 2*dimSize[0]];

	float x, y, z;

	//transform points
	x = xIn*tform[0] + yIn*tform[4] + zIn*tform[8] + tform[12];
	y = xIn*tform[1] + yIn*tform[5] + zIn*tform[9] + tform[13];
	z = xIn*tform[2] + yIn*tform[6] + zIn*tform[10] + tform[14];

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

	pointsOut[i] = x;
	pointsOut[i + 1*dimSize[0]] = y;

	int j;
	for(j = 2; j < (dimSize[0] - 1); j++){
		pointsOut[i + j*dimSize[0]] = pointsIn[i + (j+1)*dimSize[0]];
	}
}
}