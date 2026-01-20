cmake_minimum_required(VERSION 3.18)
project(FaceReplacer CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(USE_CUDA "Enable CUDA support" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Find packages
find_package(OpenCV REQUIRED COMPONENTS 
    core imgproc imgcodecs highgui dnn
)

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    find_package(OpenCV REQUIRED COMPONENTS cudaarithm cudaimgproc cudafilters)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80;86")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

if(USE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Source files
set(SOURCES
    src/face_replacer.cpp
    src/face_detector.cpp
    src/segmentation.cpp
)

set(HEADERS
    include/face_replacer.hpp
    include/face_detector.hpp
    include/segmentation.hpp
)

# CUDA sources
if(USE_CUDA)
    set(CUDA_SOURCES
        src/cuda/gpu_blend.cu
    )
    set(CUDA_HEADERS
        include/cuda/gpu_blend.cuh
    )
endif()

# Library
if(USE_CUDA)
    add_library(facereplacer_lib
        ${SOURCES}
        ${HEADERS}
        ${CUDA_SOURCES}
        ${CUDA_HEADERS}
    )
    set_target_properties(facereplacer_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
else()
    add_library(facereplacer_lib
        ${SOURCES}
        ${HEADERS}
    )
endif()

target_link_libraries(facereplacer_lib
    ${OpenCV_LIBS}
)

if(USE_CUDA)
    target_link_libraries(facereplacer_lib
        ${CUDA_LIBRARIES}
    )
endif()

# Main executable
add_executable(face_replacer src/main.cpp)
target_link_libraries(face_replacer
    facereplacer_lib
    ${OpenCV_LIBS}
)

# Install targets
install(TARGETS face_replacer facereplacer_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

# Print configuration
message(STATUS "")
message(STATUS "=== Face Replacer Configuration ===")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA support: ${USE_CUDA}")
if(USE_CUDA)
    message(STATUS "CUDA version: ${CUDA_VERSION}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "===================================")
message(STATUS "")
