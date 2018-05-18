//based on tensorflow/contrib/lite/examples/minimal/minimal.cc

#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include <cstdio>
#include <iostream>
#include "exp.h"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
// Usage: minimal <tflite model>

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x) \
  if(!(x)) {                                                    \
    fprintf(stderr, "Error at %s:%d\n",  __FILE__, __LINE__); \
    exit(1); \
  }


int main(int argc, char *argv[]) {
  if(argc != 3) {
    fprintf(stderr, "Usage: <model> <image>\n");
    return 1;
  }else{
      std::cout << "Reading model from: " << argv[1] << std::endl;
      std::cout << "Reading image from: " << argv[2] << std::endl;
  }
  const char* filename = argv[1];
    const char* imagefile = argv[2];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model
      = tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);



  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom("Exp", Register_EXP());

  InterpreterBuilder builder(*model.get(), resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);


    std::cout << "Tensors size: " << interpreter->tensors_size() << std::endl;
    std::cout << "Nodes size: " << interpreter->nodes_size()  << std::endl;
    std::cout << "inputs: " << interpreter->inputs().size()  << std::endl;
    std::cout << "input(0) name: " << interpreter->GetInputName(0)  << std::endl;

    int t_size = interpreter->tensors_size();
    std::cout << "Name       | Bytes | Type | Scale | Zero Point" << std::endl;
    for (int i = 0; i < t_size; ++i)
        if (interpreter->tensor(i)->name)
            std::cout << i << ": " << interpreter->tensor(i)->name << ", "
                    << "| " << interpreter->tensor(i)->bytes << ", "
                    << "| " << interpreter->tensor(i)->type << ", "
                    << "| " << interpreter->tensor(i)->params.scale << ", "
                    << "| " << interpreter->tensor(i)->params.zero_point << std::endl;


    std::cout << "input: " << interpreter->inputs()[0]  << std::endl;
    std::cout << "number of inputs: " << interpreter->inputs().size()  << std::endl;
    std::cout << "number of outputs: " << interpreter->outputs().size()  << std::endl;


    cv::Mat cvimg = cv::imread(imagefile);
    cv::imshow("Input",cvimg);
    cv::waitKey(0);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    // Fill input buffers

    int input = interpreter->inputs()[0];
    memcpy(interpreter->typed_tensor<float>(input), cvimg.data, cvimg.total() * cvimg.elemSize());

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // Read output buffers
  // TODO(user): Insert getting data out code.
    int output = interpreter->outputs()[0];
    std::cout << interpreter->typed_output_tensor<float>(0) << std::endl;
  return 0;
}