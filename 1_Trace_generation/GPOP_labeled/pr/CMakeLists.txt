cmake_minimum_required(VERSION 3.22)
project(pr)
set(CMAKE_CXX_STANDARD 11 -fopenmp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

add_executable(pr main.cpp)
