#ifndef MATLAB_CALLS_H
#define MATLAB_CALLS_H

unsigned int getNumMove(void);
unsigned int getNumBase(void);
unsigned int getNumPairs(void);
void clearScans(void);
void initalizeScans(unsigned int numBaseIn, unsigned int numMoveIn, unsigned int numPairsIn);
void setBaseImage(unsigned int scanNum, unsigned int height, unsigned int width, unsigned int numCh, float* base);
void setMoveImage(unsigned int scanNum, unsigned int height, unsigned int width, unsigned int numCh, float* move);
void setMoveScan(unsigned int scanNum, unsigned int numDim, unsigned int numCh, unsigned int numPoints, float* move);

#endif //MATLAB_CALLS_H