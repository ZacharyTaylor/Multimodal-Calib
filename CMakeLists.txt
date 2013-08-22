#First cmake file I have ever written so expect bugs

cmake_minimum_required(VERSION 2.8)
project (Multimodal-Calib) 

#packages

find_package(Matlab)
if (${MATLAB_FOUND})
   message(STATUS "Found Matlab.")
else (${MATLAB_FOUND})
   message(STATUS "Could not locate Matlab.")
   
   SET( MATLAB_ROOT_DIR "Must be set" CACHE STRING "" )
   SET( MATLAB_INCLUDE_DIR "${MATLAB_ROOT_DIR}\\extern\\include" CACHE STRING "" FORCE )
   
endif (${MATLAB_FOUND})

find_package(CUDA)
if (${CUDA_FOUND})
   message(STATUS "Found Cuda.")
else (${CUDA_FOUND})
   message(STATUS "Could not locate Cuda.")
   SET( CUDA_TOOLKIT_ROOT_DIR "Must be set" CACHE STRING "" )
endif (${CUDA_FOUND})

include_directories (${MATLAB_INCLUDE_DIR})
include_directories ("${CUDA_TOOLKIT_ROOT_DIR}\\include")


#nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -arch=compute_12 )

add_subdirectory(Code)
