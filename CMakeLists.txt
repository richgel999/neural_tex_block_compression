cmake_minimum_required(VERSION 3.20)
project(ntc LANGUAGES CXX)

option(NTC_CUDA "Build the CUDA backend (--cuda); needs CUDA >= 12.8 for sm_120" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)

add_executable(ntc main.cpp)
target_include_directories(ntc PRIVATE ${CMAKE_SOURCE_DIR}/cuda)   # ntc_noise.h is plain C++, used by --rng hash
if(OpenMP_CXX_FOUND)
  target_link_libraries(ntc PRIVATE OpenMP::OpenMP_CXX)
endif()

# Host flags, scoped to C++ so they never reach nvcc.
if(MSVC)
  target_compile_options(ntc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2 /fp:fast /arch:AVX2>)
else()
  target_compile_options(ntc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -O3 -ffast-math -march=native>)
endif()

if(NTC_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(ntc PRIVATE ntc_cuda)
  target_compile_definitions(ntc PRIVATE NTC_CUDA=1)
endif()
