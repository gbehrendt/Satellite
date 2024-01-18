cmake_minimum_required(VERSION 3.26)
project(Satellite)

#set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

SET(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${CMAKE_SOURCE_DIR}/build")

# set the g++ compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_CXX_COMPILER_LOADED)
    message(STATUS "Compiler Path: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

list(APPEND CMAKE_PREFIX_PATH "/home/gbehrendt/eigen/cmake/")

# for using casadi
find_package(casadi REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(PythonLibs 3.8)

# set casadi include directory
set(CASADI_INCLUDE_DIR /usr/local/include/casadi)
set(EIGEN_INCLUDE_DIR /usr/local/include/eigen)
set(EIGEN_INCLUDE_DIR /usr/local/include/python3.8/Python.h)

# find casadi library
find_library(CASADI_LIBRARY
        NAMES casadi
        HINTS ${CASADI_INCLUDE_DIR}/../lib $ENV{CASADI_PREFIX}/lib)
if(CASADI_LIBRARY)
    set(CASADI_LIBRARIES ${CASADI_LIBRARIES} ${CASADI_LIBRARY})
endif()

# find eigen library
find_library(EIGEN_LIBRARY
        NAMES eigen
        HINTS ${EIGEN_INCLUDE_DIR}/../lib $ENV{EIGEN_PREFIX}/lib)
if(EIGEN_LIBRARY)
    set(EIGEN_LIBRARIES ${EIGEN_LIBRARIES} ${EIGEN_LIBRARY})
endif()

# find python library
find_library(PYTHON_LIBRARY
        NAMES python3.8
        HINTS ${PYTHON_INCLUDE_DIR}/../lib $ENV{PYTHON_PREFIX}/lib)
if(PYTHON_LIBRARY)
    set(PYTHON_LIBRARIES ${PYTHON_LIBRARIES} ${PYTHON_LIBRARY})
endif()

# include header directories
include_directories(
        ${CMAKE_SOURCE_DIR}/include
        SYSTEM ${IPOPT_INCLUDE_DIRS}
        SYSTEM ${CASADI_INCLUDE_DIR}
        SYSTEM ${EIGEN_INCLUDE_DIR}
        SYSTEM ${PYTHON_INCLUDE_DIR}
        ${CODE_GEN_INCLUDE_DIR})

# library directories
link_directories(${LIBRARY_DIRS})

# find all the header files
file(GLOB HEADER_FILES_HPP ${CMAKE_SOURCE_DIR}/include/*.hpp)
file(GLOB HEADER_FILES_H ${CMAKE_SOURCE_DIR}/include/*.h)

add_executable(Satellite main.cpp matplotlibcpp.h)
target_link_libraries(${PROJECT_NAME} ${CASADI_LIBRARIES} ipopt)
target_link_libraries (${PROJECT_NAME}  ${EIGEN_LIBRARIES})
target_include_directories(${PROJECT_NAME}  PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}  ${PYTHON_LIBRARIES})

add_executable(monteCarlo monteCarlo.cpp
        monteCarlo.cpp)
target_link_libraries(monteCarlo ${CASADI_LIBRARIES} ipopt)
