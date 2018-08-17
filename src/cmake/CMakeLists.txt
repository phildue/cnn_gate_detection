cmake_minimum_required(VERSION 3.5)
set(PROJECT_ROOT ../../)
project(mavv)
set (CMAKE_CXX_STANDARD 11)


## Include Opencv
set(OpenCV_DIR C:/Users/mail-/Documents/code/dronerace2018/lib/opencv/build)
find_package(OpenCV REQUIRED)

## Include Tensorflow Lite
include_directories(	${PROJECT_ROOT}/lib/tensorflow/contrib/lite
			${PROJECT_ROOT}/lib/tensorflow/
			${PROJECT_ROOT}/lib/flatbuf/include)
link_directories(${PROJECT_ROOT}/lib/tensorflow/bazel-bin/tensorflow/contrib/lite)

add_executable(run_tflite ../cpp/run_tflite/main.cpp ../cpp/run_tflite/exp.cpp ../cpp/run_tflite/exp.h)
target_link_libraries(run_tflite libtflite.so)
target_link_libraries(run_tflite ${OpenCV_LIBS})

## Include Darknet
include_directories( ${PROJECT_ROOT}/lib/darknet/include/)
set(DARKNET_SOURCE ${PROJECT_ROOT}/lib/darknet/src/activation_kernels.cu
        ${PROJECT_ROOT}/lib/darknet/src/activation_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/activation_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/activations.c
        ${PROJECT_ROOT}/lib/darknet/src/activations.h
        ${PROJECT_ROOT}/lib/darknet/src/avgpool_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/avgpool_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/avgpool_layer_kernels.cu
        ${PROJECT_ROOT}/lib/darknet/src/batchnorm_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/batchnorm_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/blas.c
        ${PROJECT_ROOT}/lib/darknet/src/blas.h
        ${PROJECT_ROOT}/lib/darknet/src/blas_kernels.cu
        ${PROJECT_ROOT}/lib/darknet/src/box.c
        ${PROJECT_ROOT}/lib/darknet/src/box.h
        ${PROJECT_ROOT}/lib/darknet/src/classifier.h
        ${PROJECT_ROOT}/lib/darknet/src/col2im.h
        ${PROJECT_ROOT}/lib/darknet/src/col2im.c
        ${PROJECT_ROOT}/lib/darknet/src/col2im_kernels.cu
        ${PROJECT_ROOT}/lib/darknet/src/compare.c
        ${PROJECT_ROOT}/lib/darknet/src/connected_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/connected_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/convolutional_kernels.cu
        ${PROJECT_ROOT}/lib/darknet/src/convolutional_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/convolutional_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/cost_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/cost_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/crnn_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/crnn_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/crop_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/crop_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/crop_layer_kernels.cu
        ${PROJECT_ROOT}/lib/darknet/src/cuda.c
        ${PROJECT_ROOT}/lib/darknet/src/cuda.h
        ${PROJECT_ROOT}/lib/darknet/src/data.c
        ${PROJECT_ROOT}/lib/darknet/src/data.h
        ${PROJECT_ROOT}/lib/darknet/src/deconvolutional_kernels.cu
        ${PROJECT_ROOT}/lib/darknet/src/deconvolutional_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/deconvolutional_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/demo.c
        ${PROJECT_ROOT}/lib/darknet/src/demo.h
        ${PROJECT_ROOT}/lib/darknet/src/detection_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/detection_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/dropout_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/dropout_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/dropout_layer_kernels.cu
        ${PROJECT_ROOT}/lib/darknet/src/gemm.c
        ${PROJECT_ROOT}/lib/darknet/src/gemm.h
        ${PROJECT_ROOT}/lib/darknet/src/gru_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/gru_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/im2col.c
        ${PROJECT_ROOT}/lib/darknet/src/im2col.h
        ${PROJECT_ROOT}/lib/darknet/src/im2col_kernels.cu
        ${PROJECT_ROOT}/lib/darknet/src/image.c
        ${PROJECT_ROOT}/lib/darknet/src/image.h
        ${PROJECT_ROOT}/lib/darknet/src/l2norm_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/l2norm_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/layer.c
        ${PROJECT_ROOT}/lib/darknet/src/layer.h
        ${PROJECT_ROOT}/lib/darknet/src/list.c
        ${PROJECT_ROOT}/lib/darknet/src/list.h
        ${PROJECT_ROOT}/lib/darknet/src/local_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/local_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/logistic_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/logistic_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/lstm_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/lstm_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/matrix.c
        ${PROJECT_ROOT}/lib/darknet/src/matrix.h
        ${PROJECT_ROOT}/lib/darknet/src/maxpool_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/maxpool_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/maxpool_layer_kernels.cu
        ${PROJECT_ROOT}/lib/darknet/src/network.c
        ${PROJECT_ROOT}/lib/darknet/src/network.h
        ${PROJECT_ROOT}/lib/darknet/src/normalization_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/normalization_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/option_list.c
        ${PROJECT_ROOT}/lib/darknet/src/option_list.h
        ${PROJECT_ROOT}/lib/darknet/src/parser.c
        ${PROJECT_ROOT}/lib/darknet/src/parser.h
        ${PROJECT_ROOT}/lib/darknet/src/region_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/region_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/reorg_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/reorg_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/rnn_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/rnn_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/route_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/route_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/shortcut_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/shortcut_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/softmax_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/softmax_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/stb_image.h
        ${PROJECT_ROOT}/lib/darknet/src/stb_image_write.h
        ${PROJECT_ROOT}/lib/darknet/src/tree.h
        ${PROJECT_ROOT}/lib/darknet/src/tree.c
        ${PROJECT_ROOT}/lib/darknet/src/upsample_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/upsample_layer.h
        ${PROJECT_ROOT}/lib/darknet/src/utils.c
        ${PROJECT_ROOT}/lib/darknet/src/utils.h
        ${PROJECT_ROOT}/lib/darknet/src/yolo_layer.c
        ${PROJECT_ROOT}/lib/darknet/src/yolo_layer.h)



add_executable(darknet_runner ${DARKNET_SOURCE}
                              ${PROJECT_ROOT}/src/cpp/run_darknet/main.cpp)

target_link_libraries(darknet_runner ${OpenCV_LIBS})
target_link_libraries(darknet_runner pthread)
