cmake_minimum_required(VERSION 3.14)
project(bdg)

set(CMAKE_CXX_STANDARD 17)
ADD_COMPILE_OPTIONS(-O3)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories(
  src
)

add_executable(
    correctness_test
    src/bdg.cpp
    test/correctness_test.cpp
)

target_link_libraries(
  correctness_test
  gtest
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(correctness_test)