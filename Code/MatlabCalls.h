#ifndef MATLAB_CALLS_H
#define MATLAB_CALLS_H

//must use "extern c" when creating dll so matlab can read it, but matlab can't see "extern c" as it is a c++ term.
#ifdef __cplusplus
	#define DllExport  extern "C" __declspec( dllexport )
#endif

#ifndef __cplusplus
	#define DllExport
#endif

DllExport unsigned int getNumMove(void);
DllExport unsigned int getNumBase(void);
DllExport unsigned int getNumPairs(void);
DllExport void clearScans(void);
DllExport void initalizeScans(unsigned int numBaseIn, unsigned int numMoveIn, unsigned int numPairsIn);
DllExport void setBaseImage(unsigned int scanNum, unsigned int height, unsigned int width, unsigned int numCh, float* base);
DllExport void setMoveImage(unsigned int scanNum, unsigned int height, unsigned int width, unsigned int numCh, float* move);
DllExport void setMoveScan(unsigned int scanNum, unsigned int numDim, unsigned int numCh, unsigned int numPoints, float* move);

#endif //MATLAB_CALLS_H