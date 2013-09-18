#include "Kernels.h"

__global__ void generateOutputKernel(float* locs, float* vals, float* out, size_t width, size_t height, size_t depth, size_t numPoints, size_t dilate){
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numPoints){
		return;
	}

	int2 loc;
	loc.x = floor(locs[i]+0.5f);
	loc.y = floor(locs[i + numPoints]+0.5f);

	for(int dx = (-((int)dilate)+1); dx < ((int)dilate); dx++){
		for(int dy = (-((int)dilate)+1); dy < ((int)dilate); dy++){
			bool inside =
				((0 <= (loc.x + dx)) && ((loc.x + dx) < width) &&
				(0 <= (loc.y + dy)) && ((loc.y + dy) < height));

			if (inside){
				for(size_t j = 0; j < depth; j++){
					out[(loc.x + dx) + width*(loc.y + dy) + j*width*height] = vals[i + j*numPoints];
				}
			}
		}
	}
}

__global__ void AffineTransformKernel(const float* tform, const float* pointsIn, float* pointsOut, const size_t numPoints){
	
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

__global__ void CameraTransformKernel(const float* const tform,
									  const size_t* const tformIdx,
									  const float* const cam,
									  const bool* const panoramic,
									  const size_t* const camIdx,
									  const float* const xIn,
									  const float* const yIn,
									  const float* const zIn,
									  float* const xOut,
									  float* const yOut,
									  const size_t numPoints,
									  const size_t offset){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int tIdx, cIdx;

	//get transform index (inefficient as hell, if things are running slow look at this line)
	for(unsigned int tIdx = 0; tformIdx[tIdx] < (i+offset); tIdx++); 
	//get camera index

	if(i >= numPoints){
		return;
	}

	//transform points
	float x = xIn*tform[0] + yIn*tform[4] + zIn*tform[8] + tform[12];
	float y = xIn*tform[1] + yIn*tform[5] + zIn*tform[9] + tform[13];
	float z = xIn*tform[2] + yIn*tform[6] + zIn*tform[10] + tform[14];

	if((z <= 0) && !panoramic){
		x = -1;
		y = -1;
	}
	else{

		

		if(panoramic){
			//panoramic camera model
			y = (y/sqrt(z*z + x*x));
			x = atan2(x,z);

			//apply projective camera matrix
			x = cam[0]*x + cam[6];
			y = cam[4]*y + cam[7];

		}
		else{
			//apply projective camera matrix
			x = cam[0]*x + cam[3]*y + cam[6]*z + cam[9];
			y = cam[1]*x + cam[4]*y + cam[7]*z + cam[10];
			z = cam[2]*x + cam[5]*y + cam[8]*z + cam[11];

			//pin point camera model
			y = y/z;
			x = x/z;
		}
	}

	//output points
	pointsOut[i + 0*numPoints] = x;
	pointsOut[i + 1*numPoints] = y;
}

__global__ void SSDKernel(const float* A, const float* B, const size_t length, float* out, float* zeroEl){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= length){
		return;
	}

	float temp;

	//ignore zeros
	if((A[i] == 0) || (B[i] == 0)){
		temp = 0;
		zeroEl[i] = 1;
	}
	else{
		temp = fabs(A[i] - B[i]);
		zeroEl[i] = 0;
	}

	out[i] = temp;
}

__global__ void GOMKernel(const float* A, const float* B, const size_t length, float* phaseOut, float* magOut){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= length){
		return;
	}

	const float* magA = &A[0];
	const float* magB = &B[0];
	const float* phaseA = &A[length];
	const float* phaseB = &B[length];
	
	//float phase = abs(magA[i]-magB[i]);
	float phase = PI*abs(phaseA[i] - phaseB[i])/180;

    phase = (cos(2*phase)+1)/2;
	float mag = magA[i]*magB[i];

	//ignore zeros
	if((phaseA[i] == 0) || (phaseB[i] == 0)){
		mag = 0;
	}

    phaseOut[i] =  mag*phase;
	magOut[i] = mag;
}

__global__ void livValKernel(const float* A, const float* B, const float* Bavg, const size_t length, float* out){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= length){
		return;
	}

	out[i] = A[i] * fabs(B[i] - Bavg[i]);
}

void RunBSplineKernel(float* volume, size_t width, size_t height){
	//CubicBSplinePrefilter2D(volume, sizeof(float), width,height);
}

