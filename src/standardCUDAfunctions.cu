/*
 * standardCUDAfunctions.cu
 *
 *  Created on: Jul 24, 2014
 *      Author: preibisch
 */
#include "book.h"
#include "cuda.h"
#include "standardCUDAfunctions.h"

//==============================================
int getCUDAcomputeCapabilityMajorVersion(int devCUDA)
{
	//return 3;
	int major = 0, minor = 0;
	cuDeviceComputeCapability(&major, &minor,devCUDA);
	return major;
}
int getCUDAcomputeCapabilityMinorVersion(int devCUDA)
{
	//return 0;
	int major = 0, minor = 0;
	cuDeviceComputeCapability(&major, &minor,devCUDA);
	return minor;
}

int getNumDevicesCUDA()
{
	//return 2;
	int count = 0;
	if ( !HANDLE_ERROR_NOCRASH(cudaGetDeviceCount ( &count )) )
		return -1;
	return count;
}
void getNameDeviceCUDA(int devCUDA, char* name)
{
	//if ( devCUDA == 0 )
	//	strcpy(name, "CPU emulation #1");
	//else
	//	strcpy(name, "CPU emulation #2");
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties(&prop, devCUDA));
	memcpy(name,prop.name,sizeof(char)*256);
}
long long int getMemDeviceCUDA(int devCUDA)
{
	//return (long long int)3 * (long long int)1024 * (long long int)1024 * (long long int)1024;
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties(&prop, devCUDA));
	return ((long long int)prop.totalGlobalMem);
}
