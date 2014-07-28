#include "separableConvolution.h"
#include "cuda.h"
#include "book.h"
#include <math.h>

#define KERNEL_RADIUS 15
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

__constant__ float c_Kernel[ KERNEL_LENGTH ];

void setConvolutionKernel_31( float *h_Kernel )
{
    cudaMemcpyToSymbol( c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float) );
}

// how many threads per block in x (total num threads: x*y)
#define	ROWS_BLOCKDIM_X 16
// how many threads per block in y
#define	ROWS_BLOCKDIM_Y 4
// how many pixels in x are convolved by each thread
#define	ROWS_RESULT_STEPS 8
// these are the border pixels (loaded to support the kernel width for processing)
// the effective border width is ROWS_HALO_STEPS * ROWS_BLOCKDIM_X, which has to be
// larger or equal to the kernel radius to work
#define	ROWS_HALO_STEPS 1

#define	COLUMNS_BLOCKDIM_X 16
#define	COLUMNS_BLOCKDIM_Y 16
#define	COLUMNS_RESULT_STEPS 8
#define	COLUMNS_HALO_STEPS 1

#define	DEPTH_BLOCKDIM_X 16
#define	DEPTH_BLOCKDIM_Z 16
#define	DEPTH_RESULT_STEPS 8
#define	DEPTH_HALO_STEPS 1

extern "C" int multipleOfX_31()
{
	return imax( DEPTH_BLOCKDIM_X, imax( ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X, COLUMNS_BLOCKDIM_X) );
}
extern "C" int multipleOfY_31()
{
	return imax( ROWS_BLOCKDIM_Y, COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y );
}
extern "C" int multipleOfZ_31()
{
	return DEPTH_RESULT_STEPS * DEPTH_BLOCKDIM_Z;
}

extern "C" int convolve_31( float *image, float *kernelX, float *kernelY, float *kernelZ, int imageW, int imageH, int imageD, int convolveX, int convolveY, int convolveZ, int devCUDA )
{
	fprintf(stderr, "Cuda device: %i\n", devCUDA );

	// test dimensions
	if ( imageW % multipleOfX_31() != 0 ||
		 imageH % multipleOfY_31() != 0 ||
		 imageD % multipleOfZ_31() != 0 )
		return 0; //false

	fprintf(stderr, "Convolving X: %i\n", convolveX );
	fprintf(stderr, "Convolving Y: %i\n", convolveY );
	fprintf(stderr, "Convolving Z: %i\n", convolveZ );

	fprintf(stderr, "Image Size X: %i\n", imageW );
	fprintf(stderr, "Image Size Y: %i\n", imageH );
	fprintf(stderr, "Image Size Z: %i\n", imageD );

	float *d_Input, *d_Output;

	cudaSetDevice( devCUDA );

	// allocate memory for CUDA
	HANDLE_ERROR( cudaMalloc((void **)&d_Input,   imageW * imageH * imageD * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void **)&d_Output,  imageW * imageH * imageD * sizeof(float)) );

    // copy input to graphics card
	HANDLE_ERROR( cudaMemcpy(d_Input, image, imageW * imageH * imageD * sizeof(float), cudaMemcpyHostToDevice) );

	int in = 0;

    if ( convolveX != 0 )
    {
        HANDLE_ERROR( cudaDeviceSynchronize() );
		setConvolutionKernel_31( kernelX );
	    HANDLE_ERROR( cudaDeviceSynchronize() );
		convolutionX_31( d_Output, d_Input, imageW, imageH, imageD );
		in = 1;
    }

    if ( convolveY != 0 )
    {
        HANDLE_ERROR( cudaDeviceSynchronize() );
    	setConvolutionKernel_31( kernelY );
        HANDLE_ERROR( cudaDeviceSynchronize() );

    	if ( in == 0 )
    	{
    		convolutionY_31( d_Output, d_Input, imageW, imageH, imageD );
    		in = 1;
    	}
    	else
    	{
    		convolutionY_31( d_Input, d_Output, imageW, imageH, imageD );
    		in = 0;
    	}
    }

    if ( convolveZ != 0 )
    {
        HANDLE_ERROR( cudaDeviceSynchronize() );
		setConvolutionKernel_31( kernelZ );
	    HANDLE_ERROR( cudaDeviceSynchronize() );

		if ( in == 0 )
		{
			convolutionZ_31( d_Output, d_Input, imageW, imageH, imageD );
			in = 1;
		}
		else
		{
			convolutionZ_31( d_Input, d_Output, imageW, imageH, imageD );
			in = 0;
		}
    }

    HANDLE_ERROR( cudaDeviceSynchronize() );

    // copy back
    if ( in == 1 )
    	HANDLE_ERROR( cudaMemcpy(image, d_Output, imageW * imageH * imageD * sizeof(float), cudaMemcpyDeviceToHost) );
    else
    	HANDLE_ERROR( cudaMemcpy(image, d_Input, imageW * imageH * imageD * sizeof(float), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaFree(d_Output) );
    HANDLE_ERROR( cudaFree(d_Input) );

    cudaDeviceReset();

    return -1; // true
}

__global__ void convolutionX_31_Kernel( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD )
{
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z;

    // set the input and output arrays to the right offset (actually the output is not at the right offset, but this is corrected later)
    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    // Load main data
    // Start copying after the ROWS_HALO_STEPS, only the original data that will be convolved
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    // Load left halo
    // If the data fetched is outside of the image (note: baseX can be <0 for the first block) , use a zero-out of bounds strategy
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        float sum = 0;

#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

__global__ void convolutionY_31_Kernel( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD )
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z;

    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * imageW];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        float sum = 0;
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * imageW] = sum;
    }
}

__global__ void convolutionZ_31_Kernel( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD )
{
	// here it is [x][z], we leave out y as it has a size of 1
    __shared__ float s_Data[DEPTH_BLOCKDIM_X][(DEPTH_RESULT_STEPS + 2 * DEPTH_HALO_STEPS) * DEPTH_BLOCKDIM_Z + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * DEPTH_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y;
    const int baseZ = (blockIdx.z * DEPTH_RESULT_STEPS - DEPTH_HALO_STEPS) * DEPTH_BLOCKDIM_Z + threadIdx.z;

    d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
    d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

    //Main data
#pragma unroll

    for (int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < DEPTH_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = (baseZ >= -i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS + DEPTH_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z]= (imageD - baseZ > i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++)
    {
        float sum = 0;
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z + j];
        }

        d_Dst[i * DEPTH_BLOCKDIM_Z * imageW * imageH] = sum;
    }
}

void convolutionX_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD )
{
    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y, imageD);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1);

    convolutionX_31_Kernel<<<blocks, threads>>>( d_Dst, d_Src, imageW, imageH, imageD );
}

void convolutionY_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD )
{
    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y), imageD);
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1);

    convolutionY_31_Kernel<<<blocks, threads>>>( d_Dst, d_Src, imageW, imageH, imageD );
}

void convolutionZ_31( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD )
{
    dim3 blocks(imageW / DEPTH_BLOCKDIM_X, imageH, imageD/ (DEPTH_RESULT_STEPS * DEPTH_BLOCKDIM_Z) );
    dim3 threads(DEPTH_BLOCKDIM_X, 1, DEPTH_BLOCKDIM_Z);

    convolutionZ_31_Kernel<<<blocks, threads>>>( d_Dst, d_Src, imageW, imageH, imageD );
}
