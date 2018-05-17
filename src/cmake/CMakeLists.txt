cmake_minimum_required(VERSION 3.5)
set(PROJECT_ROOT ../../)
project(mavv)
set (CMAKE_CXX_STANDARD 11)
include_directories(	${PROJECT_ROOT}/lib/tensorflow/contrib/lite
			${PROJECT_ROOT}/lib/tensorflow/
			${PROJECT_ROOT}/lib/flatbuf/include)
link_directories(${PROJECT_ROOT}/lib/tensorflow/bazel-bin/tensorflow/contrib/lite)
add_executable(run_tflite ../cpp/main.cpp ../cpp/exp.cpp ../cpp/exp.h)
target_link_libraries(run_tflite libtflite.so)



