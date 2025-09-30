cmake_minimum_required(VERSION 4.0)
project(AnalyticDiffEEState)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(urdfdom_headers REQUIRED)

add_executable(AnalyticDiffEEState analytic_vs_numerical_diff.cpp)

target_include_directories(AnalyticDiffEEState PRIVATE
        include
        ${EIGEN3_INCLUDE_DIRS}
        ${PINOCCHIO_INCLUDE_DIRS}
        ${urdfdom_headers_INCLUDE_DIRS}
)

target_link_libraries(AnalyticDiffEEState PRIVATE
        pinocchio::pinocchio
        ${urdfdom_headers_LIBRARIES}
)