cmake_minimum_required(VERSION 3.5)
set(PROJECT_ROOT ../../)
project(run_tflite)
set (CMAKE_CXX_STANDARD 11)

## Include Tensorflow Lite
include_directories(	${PROJECT_ROOT}/lib/tensorflow/contrib/lite
			${PROJECT_ROOT}/lib/tensorflow/
			${PROJECT_ROOT}/lib/flatbuf/include)
link_directories(${PROJECT_ROOT}/lib/tensorflow/bazel-bin/tensorflow/contrib/lite)


## Include Opencv
set(OpenCV_DIR C:/Users/mail-/Documents/code/dronerace2018/lib/opencv/build)
find_package(OpenCV REQUIRED)

add_executable(run_tflite ../cpp/main.cpp ../cpp/exp.cpp ../cpp/exp.h)
target_link_libraries(run_tflite libtflite.so)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


