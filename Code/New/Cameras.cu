#include "Cameras.h"

void Cameras::addCams(thrust::device_vector<float> camDIn, boolean panoramic){
	if(camDIn.size() != (CAM_WIDTH * CAM_HEIGHT)){
		std::cerr << "Error input camera matricies must be " << (CAM_WIDTH * CAM_HEIGHT) << " in size. Returning without setting\n";
		return;
	}
	cam camIn;
	camD.push_back(camIn);
	camD.back().cam = camDIn;
	camD.back().panoramic = panoramic;
}

void Cameras::addCams(thrust::host_vector<float> camDIn, boolean panoramic){
	if(camDIn.size() != (CAM_WIDTH * CAM_HEIGHT)){
		std::cerr << "Error input camera matricies must be " << (CAM_WIDTH * CAM_HEIGHT) << " in size. Returning without setting\n";
		return;
	}
	cam camIn;
	camD.push_back(camIn);
	camD.back().cam = camDIn;
	camD.back().panoramic = panoramic;
}

void Cameras::removeAllCameras(void){
	camD.clear();
}

float* Cameras::getCamP(size_t idx){
	if(camD.size() > idx){
		std::cerr << "Cannot get pointer to element " << idx << " as only " << camD.size() << " elements exist. Returning NULL\n";
		return NULL;
	}
	return thrust::raw_pointer_cast(&(camD[idx].cam[0]));
}

bool Cameras::getPanoramic(size_t idx){
		if(camD.size() > idx){
		std::cerr << "Cannot get element " << idx << " as only " << camD.size() << " elements exist. Returning false\n";
		return false;
	}
		return camD[idx].panoramic;
}
