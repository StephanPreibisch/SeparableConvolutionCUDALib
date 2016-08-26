/*
 * separableConvolution.h
 *
 *  Created on: Jul 24, 2014
 *      Author: preibisch
 */

#ifndef SEPARABLECONVOLUTION_H_
#define SEPARABLECONVOLUTION_H_

#include "standardCUDAfunctions.h"

#ifdef _WIN32
#define FUNCTION_PREFIX extern "C" __declspec(dllexport)
#else
#define FUNCTION_PREFIX extern "C"
#endif

// In-place convolution with a maximal kernel diameter of 31
FUNCTION_PREFIX int convolve_255( float *image, float *kernelX, float *kernelY, float *kernelZ, int imageW, int imageH, int imageD, int convolveX, int convolveY, int convolveZ, int outofbounds, float outofboundsvalue, int devCUDA );
FUNCTION_PREFIX int convolve_127( float *image, float *kernelX, float *kernelY, float *kernelZ, int imageW, int imageH, int imageD, int convolveX, int convolveY, int convolveZ, int outofbounds, float outofboundsvalue, int devCUDA );
FUNCTION_PREFIX int convolve_63( float *image, float *kernelX, float *kernelY, float *kernelZ, int imageW, int imageH, int imageD, int convolveX, int convolveY, int convolveZ, int outofbounds, float outofboundsvalue, int devCUDA );
FUNCTION_PREFIX int convolve_31( float *image, float *kernelX, float *kernelY, float *kernelZ, int imageW, int imageH, int imageD, int convolveX, int convolveY, int convolveZ, int outofbounds, float outofboundsvalue, int devCUDA );
FUNCTION_PREFIX int convolve_15( float *image, float *kernelX, float *kernelY, float *kernelZ, int imageW, int imageH, int imageD, int convolveX, int convolveY, int convolveZ, int outofbounds, float outofboundsvalue, int devCUDA );
FUNCTION_PREFIX int convolve_7( float *image, float *kernelX, float *kernelY, float *kernelZ, int imageW, int imageH, int imageD, int convolveX, int convolveY, int convolveZ, int outofbounds, float outofboundsvalue, int devCUDA );

FUNCTION_PREFIX void convolutionCPU( float *image, float *kernelX, float *kernelY, float *kernelZ, int kernelRX, int kernelRY, int kernelRZ, int imageW, int imageH, int imageD, int outofbounds, float outofboundsvalue );

#endif //SEPARABLECONVOLUTION_H_
