/*
 * separableConvolution.h
 *
 *  Created on: Jul 24, 2014
 *      Author: preibisch
 */

#ifndef SEPARABLECONVOLUTION_H_
#define SEPARABLECONVOLUTION_H_

#include "standardCUDAfunctions.h"

// In-place convolution with a maximal kernel diameter of 31
extern "C" int convolve_31( float *image, float *kernelX, float *kernelY, float *kernelZ, int imageW, int imageH, int imageD, int convolveX, int convolveY, int convolveZ, int outofbounds, float outofboundsvalue, int devCUDA );
//extern "C" int convolve_15( float *image, float *kernelX, float *kernelY, float *kernelZ, int imageW, int imageH, int imageD, int convolveX, int convolveY, int convolveZ, int outofbounds, float outofboundsvalue, int devCUDA );

extern "C" void convolutionCPU( float *image, float *kernelX, float *kernelY, float *kernelZ, int kernelRX, int kernelRY, int kernelRZ, int imageW, int imageH, int imageD, int outofbounds, float outofboundsvalue );

#endif //SEPARABLECONVOLUTION_H_
