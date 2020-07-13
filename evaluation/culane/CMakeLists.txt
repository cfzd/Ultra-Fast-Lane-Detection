cmake_minimum_required(VERSION 3.16)

project(seg_label_generate)

set(CMAKE_CXX_STANDARD 11)

add_definitions(
    -DCPU_ONLY
)

set(EXECUTE_COMMAND find src/ -name "*.cpp")

execute_process(
    COMMAND ${EXECUTE_COMMAND}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CXX_SRCS_RAW
)
string(REPLACE "\n" ";" CXX_SRCS ${CXX_SRCS_RAW}) # drop blank line

add_executable(${PROJECT_NAME}
    ${CXX_SRCS}
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

# --- target config with include dirs / libs
target_link_libraries(${PROJECT_NAME}
    ${dep_libs}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC include
)
