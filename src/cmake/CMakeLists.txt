project(mavv)

set(Eigen3_DIR ../../lib/eigen/cmake)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(Protobuf_DIR ../../lib/protobuf/cmake)

find_package (Protobuf REQUIRED)

add_executable(mavv ../cpp/inference_test.cpp)
target_link_libraries (mavv Eigen3::Eigen)
target_link_libraries (mavv ${Protobuf_LIBRARIES})

