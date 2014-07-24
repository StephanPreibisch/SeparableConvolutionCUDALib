/*
 * separableConvolution.h
 *
 *  Created on: Jul 24, 2014
 *      Author: preibisch
 */

#ifndef SEPARABLECONVOLUTION_H_
#define SEPARABLECONVOLUTION_H_

#include "standardCUDAfunctions.h"

// Convolution with a maximal kernel diameter of 31
extern "C" float* convolve_31( float *d_Src, float *kernelX, float *kernelY, float *kernelZ, int imageW, int imageH, int imageD, bool convolveX, bool convolveY, bool convolveZ, int devCUDA );

void setConvolutionKernel_31( float *h_Kernel );
void convolutionX_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD );
void convolutionY_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD );
void convolutionZ_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD );

#endif //SEPARABLECONVOLUTION_H_
