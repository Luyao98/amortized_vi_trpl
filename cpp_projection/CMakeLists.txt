cmake_minimum_required(VERSION 2.8.12)
project(IPP)

add_definitions(-std=c++11)

message("CPP = ${CMAKE_PREFIX_PATH}")
message("CIP = ${CMAKE_INSTALL_PREFIX}")
message("CLOD = ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lopenblas")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -llapack")
include_directories(include)

#Projection
add_library(projectionlib SHARED
        src/projection/MoreProjection.cpp
        src/projection/BatchedProjection.cpp
        src/projection/DiagCovOnlyMoreProjection.cpp
        src/projection/BatchedDiagCovOnlyProjection.cpp
        src/projection/SplitDiagMoreProjection.cpp
        src/projection/BatchedSplitDiagMoreProjection.cpp
        src/projection/CovOnlyMoreProjection.cpp
        src/projection/BatchedCovOnlyProjection.cpp)
target_link_libraries(projectionlib PUBLIC nlopt ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

pybind11_add_module(cpp_projection src/projection/PyProjection.cpp)
target_link_libraries(cpp_projection PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} projectionlib)