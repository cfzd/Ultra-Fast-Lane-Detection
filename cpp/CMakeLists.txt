cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(carimage_local)

set(OpenCV_DIR "/opt/opencv/lib/cmake/opencv4")
set(Torch_DIR "/data/bin/libtorch/share/cmake/Torch/")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SOURCE_FILES
"src/*.cpp")
add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})


target_link_libraries(${CMAKE_PROJECT_NAME} 
${TORCH_LIBRARIES} 
${OpenCV_LIBS})


set_property(TARGET ${CMAKE_PROJECT_NAME}  PROPERTY CXX_STANDARD 14)