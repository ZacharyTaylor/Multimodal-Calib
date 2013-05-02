#ifndef MI_H
#define MI_H

#include "common.h"
#include "trace.h"
#include "Reduce\reduction.h"

/**
*	miRun-
*	runs the mutual information calculation
*
*	Inputs-
*	miData - struct which contains all histograms and other information needed in the calculation
*	imData - struct containing all the information on the images that mi will be performed on
*	imNum - which image in imData will have the mi calculated
*	Output - the normalized mutual information
*/
float miRun(float* A, float* B, size_t bins, size_t numElements);

#endif