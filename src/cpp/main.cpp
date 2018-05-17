//based on tensorflow/contrib/lite/examples/minimal/minimal.cc

#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include <cstdio>
#include "exp.h"
// Usage: minimal <tflite model>

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x) \
  if(!(x)) {                                                    \
    fprintf(stderr, "Error at %s:%d\n",  __FILE__, __LINE__); \
    exit(1); \
  }


int main(int argc, char *argv[]) {
  if(argc != 2) {
    fprintf(stderr, "Usage: %s <model>\n");
    return 1;
  }
  const char* filename = argv[1];

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

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // Read output buffers
  // TODO(user): Insert getting data out code.

  return 0;
}