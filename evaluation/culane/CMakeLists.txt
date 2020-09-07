# Thanks for the contribution of zchrissirhcz imzhuo@foxmail.com
cmake_minimum_required(VERSION 3.1)

project(culane_evaluator)

set(CMAKE_CXX_STANDARD 11)

add_definitions(
    -DCPU_ONLY
)

set(SRC_LST
    ${CMAKE_SOURCE_DIR}/src/counter.cpp
    ${CMAKE_SOURCE_DIR}/src/evaluate.cpp
    ${CMAKE_SOURCE_DIR}/src/lane_compare.cpp
    ${CMAKE_SOURCE_DIR}/src/spline.cpp
)

set(HDR_LST
    ${CMAKE_SOURCE_DIR}/include/counter.hpp
    ${CMAKE_SOURCE_DIR}/include/hungarianGraph.hpp
    ${CMAKE_SOURCE_DIR}/include/lane_compare.hpp
    ${CMAKE_SOURCE_DIR}/include/spline.hpp
)

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    list(APPEND SRC_LST ${CMAKE_SOURCE_DIR}/getopt/getopt.c)
    list(APPEND HDR_LST ${CMAKE_SOURCE_DIR}/getopt/getopt.h)
endif()

add_executable(${PROJECT_NAME}
    ${SRC_LST}
    ${HDR_LST}
)

set(dep_libs "")

#--- OpenCV
# You may switch different version of OpenCV like this:
# set(OpenCV_DIR "/usr/local/opencv-4.3.0" CACHE PATH "")
find_package(OpenCV REQUIRED 
    COMPONENTS core highgui imgproc imgcodecs
)
if(NOT OpenCV_FOUND) # if not OpenCV 4.x/3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

list(APPEND dep_libs 
    PUBLIC ${OpenCV_LIBS}
)

#--- OpenMP
find_package(OpenMP)
if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
    target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endif()

if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
    message(STATUS "Building with OpenMP")
    if(OpenMP_CXX_FOUND)
        list(APPEND dep_libs PUBLIC OpenMP::OpenMP_CXX)
    else()
        list(APPEND dep_libs PRIVATE "${OpenMP_CXX_FLAGS}")
    endif()
endif()

set(dep_incs ${CMAKE_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    list(APPEND dep_incs "${CMAKE_SOURCE_DIR}/getopt")
endif()

# --- target config with include dirs / libs
target_link_libraries(${PROJECT_NAME}
    ${dep_libs}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${dep_incs}
)
