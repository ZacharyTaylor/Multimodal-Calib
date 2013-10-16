#include "Kernels.h"

#define CAM_TFORM_SIZE 16
#define CAM_MAT_SIZE 12

__global__ void generateOutputKernel(float* x, float* y, float* vals, float* out, size_t width, size_t height, size_t numPoints, size_t dilate){
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numPoints){
		return;
	}

	if((x[i] < 0) || (y[i] < 0)){
		return;
	}

	int2 loc;
	loc.x = floor(x[i]+0.5f);
	loc.y = floor(y[i]+0.5f);

	for(int dx = (-((int)dilate)+1); dx < ((int)dilate); dx++){
		for(int dy = (-((int)dilate)+1); dy < ((int)dilate); dy++){
			bool inside =
				((0 <= (loc.x + dx)) && ((loc.x + dx) < width) &&
				(0 <= (loc.y + dy)) && ((loc.y + dy) < height));

			if (inside){
				if(!out[(loc.y + dy) + height*(loc.x + dx)]){
					out[(loc.y + dy) + height*(loc.x + dx)] = vals[i];
				}
			}
		}
	}
}

__global__ void LinearInterpolateKernel(const float* const imageIn,
										float* const out,
										const size_t height,
										const size_t width,
										const float* const x,
										const float* const y,
										const size_t numPoints){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numPoints){
		return;
	}

	int xF = (int)x[i];
	int yF = (int)y[i];
	float xD = x[i] - (float)xF;
	float yD = y[i] - (float)yF;

	//check image boundries
	if((xF < 0) || (yF < 0) || (xF >= (width-1)) || (yF >= (height-1))){
		out[i] = 0;
		return;
	}

	//linear interpolate
	out[i] = (1-yD)*(1-xD)*imageIn[yF + xF*height] + 
		(1-yD)*xD*imageIn[yF + (xF+1)*height] + 
		yD*(1-xD)*imageIn[yF+1 + xF*height] +
		yD*xD*imageIn[yF+1 + (xF+1)*height];

	//keep numbers finite
	if(!isfinite(out[i])){
		out[i] = 0;
	}
}

__global__ void NearNeighKernel(const float* const imageIn,
										float* const out,
										const size_t height,
										const size_t width,
										const float* const x,
										const float* const y,
										const size_t numPoints){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numPoints){
		return;
	}

	int xF = (int)(x[i]+0.5);
	int yF = (int)(y[i]+0.5);

	//check image boundries
	if((xF < 0) || (yF < 0) || (xF >= width) || (yF >= height)){
		out[i] = 0;
		return;
	}

	//nearest neighbour interpolation
	out[i] = imageIn[yF + xF*height];

	//keep numbers finite
	if(!isfinite(out[i])){
		out[i] = 0;
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
									  const float* const cam,
									  const bool const pan,
									  const float* const xIn,
									  const float* const yIn,
									  const float* const zIn,
									  const size_t numPoints,
									  float* const xOut,
									  float* const yOut){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numPoints){
		return;
	}

	//transform points
	float x = xIn[i]*tform[0] + yIn[i]*tform[4] + zIn[i]*tform[8] + tform[12];
	float y = xIn[i]*tform[1] + yIn[i]*tform[5] + zIn[i]*tform[9] + tform[13];
	float z = xIn[i]*tform[2] + yIn[i]*tform[6] + zIn[i]*tform[10] + tform[14];

	if((z <= 0) && !pan){
		x = -1;
		y = -1;
	}
	else{
		if(pan){
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
	xOut[i] = x;
	yOut[i] = y;
}

__global__ void SSDKernel(float* const gen, const float* const scan, const size_t length){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= length){
		return;
	}

	//ignore zeros
	if((gen[i] != 0) && (scan[i] != 0)){
		gen[i] = (gen[i] - scan[i])*(gen[i] - scan[i]);
	}
	else{
		gen[i] = 0;
	}
}

__global__ void GOMKernel(float* const genMag, float* const genPhase, const float* const mag, const float* const phase, const size_t length){
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= length){
		return;
	}
	
	float phaseOut = PI*abs(genPhase[i] - phase[i])/180;

    phaseOut = cos(2*phaseOut)+1;
	float magOut = genMag[i]*mag[i];

	//ignore zeros
	if((phase[i] == 0) || (genPhase[i] == 0)){
		magOut = 0;
	}

    genPhase[i] =  magOut*phaseOut;
	genMag[i] = magOut;
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

