#include "Kernel.h"
#include <vector_types.h>
//#include "CI.h"

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

__global__ void CameraTransformKernel(const float* tform, const float* cam, const float* pointsIn, float* pointsOut, const size_t numPoints, const bool panoramic){
	
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

__global__ void livValKernel(const float* A, const float* B, const size_t length, float* out){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= length){
		return;
	}

	out[i] = A[i] * B[i];
}

__global__ void TestKernel(const float* locs, const float* A, const float* B, const size_t length, float* randNums, float* phaseOut, float* magOut){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= TEST_LOCS){
		return;
	}
	
	//get points
	size_t idx[TEST_POINTS];
	for(size_t j = 0; j < TEST_POINTS; j++){
		idx[j] = (size_t)(length*randNums[TEST_LOCS*j + i]);
	}

	float cx = 0; float dxA = 0; float dxB = 0;
	float cy = 0; float dyA = 0; float dyB = 0;
	float cA = 0; float dA = 0;
	float cB = 0; float dB = 0;
	
	//get centre point
	for(size_t j = 0; j < TEST_POINTS; j++){
		cx += locs[idx[j]];
		cy += locs[idx[j] + length];
		cA += A[idx[j]];
		cB += B[idx[j]];
	}
	cx /= TEST_POINTS;
	cy /= TEST_POINTS;
	cA /= TEST_POINTS;
	cB /= TEST_POINTS;

	//get differences
	for(size_t j = 0; j < TEST_POINTS; j++){
		float tx = cx - locs[idx[j]];
		float ty = cy - locs[idx[j] + length];
		float tA = cA - A[idx[j]];
		float tB = cB - B[idx[j]];

		dxA += tx*tA; dyA += ty*tA;
		dxB += tx*tB; dyB += ty*tB;

		dA += fabs(tA);
		dB += fabs(tB);
	}

	//float phase = abs(magA[i]-magB[i]);
	float phase = fabs(atan2(dyA,dxA) - atan2(dyB,dxB));

    phase = (cos(2*phase)+1)/2;
	float mag = dA*dB;

    phaseOut[i] =  mag*phase;
	magOut[i] = mag;
}

void RunBSplineKernel(float* volume, size_t width, size_t height){
	//CubicBSplinePrefilter2D(volume, sizeof(float), width,height);
}


