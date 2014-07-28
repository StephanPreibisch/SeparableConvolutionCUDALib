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
extern "C" int convolve_31( float *image, float *kernelX, float *kernelY, float *kernelZ, int imageW, int imageH, int imageD, int convolveX, int convolveY, int convolveZ, int devCUDA );

extern "C" int multipleOfX_31();
extern "C" int multipleOfY_31();
extern "C" int multipleOfZ_31();

extern "C" void convolutionCPU( float *image, float *kernelX, float *kernelY, float *kernelZ, int kernelRX, int kernelRY, int kernelRZ, int imageW, int imageH, int imageD );

void setConvolutionKernel_31( float *h_Kernel );
void convolutionX_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD );
void convolutionY_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD );
void convolutionZ_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD );

#endif //SEPARABLECONVOLUTION_H_
