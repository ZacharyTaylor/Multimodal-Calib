#ifndef SETUP_H
#define SETUP_H

#include "common.h"
#include "trace.h"

//! Checks for CUDA capable cards, if multiple options exist selects best one
void checkForCUDA(void);

#endif //SETUP_H