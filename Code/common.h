#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "trace.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 512

inline size_t gridSize(size_t num){
	return (size_t)ceil(((float)(num))/((float)(BLOCK_SIZE)));
};

#endif //COMMON_H