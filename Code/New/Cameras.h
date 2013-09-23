#ifndef CAMERAS_H
#define CAMERAS_H

#include "common.h"

//! width of camera projection matrix
#define CAM_WIDTH 4
//! height of camera projection matrix
#define CAM_HEIGHT 3

//! Holds the properties of the virtual camera, contains the camera matrix and properties for setting it and the type of camera
class Cameras {
private:

	//! structre holding infomation about each camera
	typedef struct {
		//! vector holding camera matrix
		thrust::device_vector<float> cam;

		//! flag for if camera is boolean
		boolean panoramic;
	} cam;

	//! Vector storing camera matrices
	std::vector<cam> camD;

public:
	//! Adds new camera matrices
	void addCams(thrust::device_vector<float> camDIn, boolean panoramic);
	//! Adds new camera matrices
	void addCams(thrust::host_vector<float> camDIn, boolean panoramic);
	//! Clears all the cameras from memory
	void removeAllCameras(void);
	//! Get a pointer to a camera matrix
	/*! /param index of matrix
	*/
	float* getCamP(size_t idx);
	//! Get if camera is panoramic
	/*! /param index of matrix
	*/
	bool getPanoramic(size_t idx);
};

#endif