//based on tensorflow/contrib/lite/examples/minimal/minimal.cc

#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include <cstdio>
#include <iostream>
#include <tensorflow/contrib/lite/context.h>
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
  //resolver.AddCustom("Exp", Register_EXP());

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
    cvimg.convertTo(cvimg,CV_32FC3,1/255.0);
    cv::imshow("Input",cvimg);
    cv::waitKey(1);
    std::cout << "Input Dims:"
                 << interpreter->tensor(0)->dims->data[0]
              << "|" << interpreter->tensor(0)->dims->data[1]
              << "|" << interpreter->tensor(0)->dims->data[2]
              << "|" << interpreter->tensor(0)->dims->data[3] << std::endl;

    int output = interpreter->outputs()[0];
    std::cout << "Output Dims:"
              << interpreter->tensor(output)->dims->data[0]
              << "|" << interpreter->tensor(output)->dims->data[1]
              << "|" << interpreter->tensor(output)->dims->data[2]
              << "|" << interpreter->tensor(output)->dims->data[3] << std::endl;



    // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    // Fill input buffers

    int input = interpreter->inputs()[0];


    std::cout << cvimg.rows << "|" << cvimg.cols << std::endl;
    std::cout<< cvimg.total() << "|" << cvimg.elemSize() << std::endl;
    std::cout << 480*640*3*sizeof(float) << std::endl;
    float *in = interpreter->typed_tensor<float>(input);
    memcpy(in, cvimg.data, cvimg.total()*cvimg.elemSize());
    std::cout << "Running inference "<< std::endl;
  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // Read output buffers
  // TODO(user): Insert getting data out code.
    float *out = interpreter->typed_output_tensor<float>(0);
    if (out != NULL){
        int rows = interpreter->tensor(output)->dims->data[1];
        int columns = interpreter->tensor(output)->dims->data[2];
        int channels = interpreter->tensor(output)->dims->data[3];
        for (int r=0; r < rows; r++){
            std::cout << std::endl;
            for (int c=0; c < columns; c++){
                std::cout << "|";
                for (int ch=0; ch < channels; ch++){
                    std::cout << *out << ", ";
                    out++;
                }
            }
        }
    }else{
        std::cout << "Output node is NULL" << std::endl;
    }

//    for (int i = 0; i < 100; i++){
//        std::cout << *out << std::endl;
//        out ++;
//    }
    std::cout << *out << std::endl;
  return 0;
}
