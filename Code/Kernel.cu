#include "Kernel.h"
#include <vector_types.h>
#include "CI.h"


texture<float> texRef;

__global__ void DenseImageInterpolateKernel(const size_t width, const size_t height, const float* locIn, const float layer, float* valsOut, const size_t numPoints){
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numPoints){
		valsOut[i] = 0.0f;
		return;
	}

	float3 loc;
	loc.x = locIn[i + numPoints]+0.5f;
	loc.y = locIn[i]+0.5f;
	loc.z = layer + 0.5f;

	bool inside =
		0 < loc.x && loc.x < width &&
		0 < loc.y && loc.y < height;

	if (!inside){
		valsOut[i] = 0.0f;
	}
	else{
		valsOut[i] = tex2D(tex, loc.x,loc.y);
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

void RunBSplineKernel(float* volume, size_t width, size_t height, size_t depth){
	CubicBSplinePrefilter3D(volume, sizeof(float), width,height,depth);
}