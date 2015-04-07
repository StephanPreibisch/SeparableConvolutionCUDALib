# cmake compatibility issues
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# project name
PROJECT(SeparableConvolutionCUDALib CXX)
if(NOT(${CMAKE_VERSION} VERSION_LESS "3.0.0"))
cmake_policy(SET CMP0042 NEW)
endif()
 
# version number
SET (SEPARABLECONVOLUTIONCUDALIB_NAME "CUDA SEPARABLE CONVOLUTION LIBRARY")
SET (SEPARABLECONVOLUTIONCUDALIB_CODENAME "${PROJECT_NAME}")
SET (SEPARABLECONVOLUTIONCUDALIB_COPYRIGHT_YEARS "2013")
SET (SEPARABLECONVOLUTIONCUDALIB_VERSION_MAJOR 2)
SET (SEPARABLECONVOLUTIONCUDALIB_VERSION_MINOR 5)
SET (SEPARABLECONVOLUTIONCUDALIB_VERSION_PATCH 0)
SET (SEPARABLECONVOLUTIONCUDALIB_VERSION_TYPE SNAPSHOT)
SET (SEPARABLECONVOLUTIONCUDALIB_VERSION_STRING "${SEPARABLECONVOLUTIONCUDALIB_VERSION_MAJOR}.${SEPARABLECONVOLUTIONCUDALIB_VERSION_MINOR}.${SEPARABLECONVOLUTIONCUDALIB_VERSION_PATCH}-${CMAKE_BUILD_TYPE}")
SET (SEPARABLECONVOLUTIONCUDALIB_VERSION "${SEPARABLECONVOLUTIONCUDALIB_VERSION_MAJOR}.${SEPARABLECONVOLUTIONCUDALIB_VERSION_MINOR}.${SEPARABLECONVOLUTIONCUDALIB_VERSION_PATCH}")
SET (SEPARABLECONVOLUTIONCUDALIB_VENDOR_ID "mpi cbg")
SET (SEPARABLECONVOLUTIONCUDALIB_VENDOR_NAME "Max Planck Institute of Molecular Cell Biology and Genetics ")
SET (SEPARABLECONVOLUTIONCUDALIB_VENDOR_URL "www.mpi-cbg.de")
SET (SEPARABLECONVOLUTIONCUDALIB_ID "${SEPARABLECONVOLUTIONCUDALIB_VENDOR_ID}.${PROJECT_NAME}")

# trying to setup paths so this package can be picked up
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")

# shared path is architecture independent for now, TODO extend this to lib/bin/include
IF(UNIX)
  IF(APPLE)
    set(INSTALL_SHARE_DIR ${PROJECT_NAME}.app/Contents/Resources/ CACHE PATH "Installation directory for shared files")
    #the following was tested with OSX 10.8.5 and Xcode 5.0.2
    #seems to me that under apple the rpath is not stripped automatically when doing the install
    #under linux it is
    SET(CMAKE_SKIP_RPATH ON)
  ELSE(APPLE)
    set(INSTALL_SHARE_DIR lib/CMake/${PROJECT_NAME} CACHE PATH "Installation directory for shared files")
  ENDIF(APPLE)
ELSE(UNIX)
  IF(WIN32 AND NOT CYGWIN)
    set(INSTALL_SHARE_DIR CMake CACHE PATH "Installation directory for shared files")
  ELSE(WIN32)
    MESSAGE(FATAL_ERROR ">> UNKNOWN ARCHITECTURE .. unable to set share dir")
  ENDIF(WIN32)
ENDIF(UNIX)
 
# Make relative paths absolute (needed later on)
foreach(p LIB INCLUDE SHARE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

# set project build type
IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
MESSAGE(">> Setting up ${CMAKE_BUILD_TYPE} build")

#########################################################################################################
## CUDA related
# project options
OPTION(INCLUDE_CUDA "Set to OFF to not search for CUDA" ON)
# find project dependencies
# find cuda
IF(INCLUDE_CUDA)
    FIND_PACKAGE(CUDA)
    IF(CUDA_FOUND)
	SET(CUDA_VERBOSE_BUILD ON)
	
	set(CUDA_NVCC_FLAGS -gencode arch=compute_10,code=sm_10;-gencode arch=compute_20,code=sm_20)
	SET(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
	IF(APPLE)
	  IF(${CUDA_HOST_COMPILER} MATCHES "/usr/bin/.*cc" OR EXISTS "/usr/bin/llvm-g++")
	    MESSAGE(">> adapting CUDA_HOST_COMPILER (${CUDA_HOST_COMPILER}) to match a CUDA supported compiler (/usr/bin/llvm-g++-4.2)")
	    SET(CUDA_HOST_COMPILER "/usr/bin/llvm-g++")
	    SET(CMAKE_CXX_COMPILER ${CUDA_HOST_COMPILER})
	    SET(CMAKE_C_COMPILER "/usr/bin/llvm-gcc")
	    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")

	    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
	  ELSE()
	    MESSAGE(WARNING ">> unknown CUDA_HOST_COMPILER (${CUDA_HOST_COMPILER}) or /usr/bin/llvm-g++-4.2 does not exist, cuda host compiler remains set to default")
	  ENDIF()
	ENDIF(APPLE)
	IF("${CUDA_VERSION}" VERSION_GREATER "5" OR "${CUDA_VERSION}" VERSION_EQUAL "5")
	  MESSAGE(">> compiling for Compute Capability 2.x, 3.0 and 3.5 only ")
          set(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20;-gencode arch=compute_30,code=sm_30;-gencode arch=compute_35,code=sm_35")
	ELSE()
	  MESSAGE(">> CUDA less than version 5.0 detected, compiling for Compute Capability 2.x only ")
          set(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20;-gencode arch=compute_10,code=sm_10")
	ENDIF()
	set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};-O2;--use_fast_math)
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-g;-G)
    ELSE(CUDA_FOUND)
        MESSAGE(FATAL_ERROR ">> CUDA not found. Exiting ...")
    ENDIF(CUDA_FOUND)
ENDIF(INCLUDE_CUDA)
 

# add subdirectories
ADD_SUBDIRECTORY(src)

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
#              v = binary    v = library

export(PACKAGE ${PROJECT_NAME})

# Create the separableconvolutioncudalib-config.cmake and separableconvolutioncudalib-config-version files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_SHARE_DIR}"
   "${INSTALL_INCLUDE_DIR}")
# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(separableconvolutioncudalib-config.cmake.in
  "${PROJECT_BINARY_DIR}/separableconvolutioncudalib-config.cmake" @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "\${SEPARABLECONVOLUTIONCUDALIB}/${REL_INCLUDE_DIR}")
configure_file(separableconvolutioncudalib-config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/separableconvolutioncudalib-config.cmake" @ONLY)
# ... for both
configure_file(separableconvolutioncudalib-config-version.cmake.in
  "${PROJECT_BINARY_DIR}/separableconvolutioncudalib-config-version.cmake" @ONLY)

# Install the separableconvolutioncudalib-config.cmake and separableconvolutioncudalib-config-version.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/separableconvolutioncudalib-config.cmake"
  "${PROJECT_BINARY_DIR}/separableconvolutioncudalib-config-version.cmake"
  DESTINATION "${INSTALL_SHARE_DIR}" COMPONENT dev)

# Install the export set for use with the install-tree 
install(EXPORT separableconvolutioncudalib-targets 
  DESTINATION "${INSTALL_SHARE_DIR}" COMPONENT dev)



