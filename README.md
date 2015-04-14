SeparableConvolutionCUDALib
===========================

Implementation of 1/2/3d separable convolution using CUDA.

Windows Build Instructions
--------------------------

To compile it under Linux/Mac/Windows I suggest NSight. Clone this repository into your cuda-workspace directory. Then make a new shared library project with the same name as the directory. Under Project > Properties > Build > Settings > Tool Settings > NVCC Linker add -lcuda to the command line pattern so that it looks like this:

${COMMAND} ${FLAGS} -lcuda ${OUTPUT_FLAG} ${OUTPUT_PREFIX} ${OUTPUT} ${INPUTS}

Now build the .so/.dll library and put it into the Fiji directory.

NOTE: If you are compiling under Windows, you need to change all 'extern "C"' definitions to 'extern "C" __declspec(dllexport)' for all function calls in the separableConvolution.h and all separableConvolution_*.cu.

OSX/Linux Build Instructions
----------------------------

First, make sure that CUDA must be available through `PATH`, `LD_LIBRARY_PATH` or equivalent. The build system is based on cmake, so please install this at any version higher or equal than 2.8.

```bash
$ cd /path/to/repo
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=/directory/of/your/choice .. #default is /usr/bin/ or similar
$ make
$ make install
```
