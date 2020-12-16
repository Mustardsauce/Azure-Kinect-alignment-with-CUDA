cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(Kinect_Alignment_Cuda VERSION 1.0.0)

if(WIN32)
    # For Azure Kinect Sensor SDK Installer
    # find_package(k4a) is search library using cmake/Findk4a.cmake.
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
endif()

find_package(k4a REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

if(k4a_FOUND AND CUDA_FOUND AND OpenCV_FOUND)
    cuda_add_executable(cuda_alignment main.cpp cuda_k4a_align.cu)
    target_link_libraries(cuda_alignment k4a::k4a)
    target_link_libraries(cuda_alignment ${OpenCV_LIBS})
    target_link_libraries(cuda_alignment ${CUDA_LIBRARIES})

    cuda_add_executable(comparison comparison.cpp cuda_k4a_align.cu)
    target_link_libraries(comparison k4a::k4a)
    target_link_libraries(comparison ${OpenCV_LIBS})
    target_link_libraries(comparison ${CUDA_LIBRARIES})
endif()
