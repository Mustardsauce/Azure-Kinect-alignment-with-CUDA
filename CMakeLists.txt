cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

project(Kinect_Alignment_Cuda VERSION 1.0.0)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH} )

find_package(k4a REQUIRED)
include_directories(${K4A_INCLUDE_DIRS})

find_package(OpenCV REQUIRED core)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
if(${CMAKE_VERSION} VERSION_EQUAL 3.7 OR ${CMAKE_VERSION} VERSION_GREATER 3.7)
  option(AUTO_DETECT_ARCH "Detect local GPU compute arch automatically" ON)
endif()

find_package(CUDA REQUIRED)

if(DEFINED AUTO_DETECT_ARCH AND "${AUTO_DETECT_ARCH}")
	CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
	set(CUDA_ARCH "${ARCH_FLAGS}" CACHE STRING "Value of the NVCC -arch option." FORCE)
endif()
message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

cuda_add_executable(cuda_alignment main.cpp cuda_k4a_align.cu)
target_link_libraries(cuda_alignment ${K4A_LIBRARIES})
target_link_libraries(cuda_alignment ${OpenCV_LIBS})
target_link_libraries(cuda_alignment ${CUDA_LIBRARIES})

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/Debug")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/Release")


add_custom_target(generate_dll
COMMAND ${CMAKE_COMMAND} -E copy ${K4A_BIN_DIRS}/k4a.dll ${CMAKE_BINARY_DIR}/Debug
COMMAND ${CMAKE_COMMAND} -E copy ${K4A_BIN_DIRS}/k4a.dll ${CMAKE_BINARY_DIR}/Release

COMMAND ${CMAKE_COMMAND} -E copy ${K4A_BIN_DIRS}/depthengine_2_0.dll ${CMAKE_BINARY_DIR}/Debug
COMMAND ${CMAKE_COMMAND} -E copy ${K4A_BIN_DIRS}/depthengine_2_0.dll ${CMAKE_BINARY_DIR}/Release

COMMAND ${CMAKE_COMMAND} -E copy ${K4A_BIN_DIRS}/k4arecord.dll ${CMAKE_BINARY_DIR}/Debug
COMMAND ${CMAKE_COMMAND} -E copy ${K4A_BIN_DIRS}/k4arecord.dll ${CMAKE_BINARY_DIR}/Release

COMMAND ${CMAKE_COMMAND} -E copy ${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}d.dll ${CMAKE_BINARY_DIR}/Debug
COMMAND ${CMAKE_COMMAND} -E copy ${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}.dll ${CMAKE_BINARY_DIR}/Release
)

add_dependencies(cuda_alignment generate_dll)

