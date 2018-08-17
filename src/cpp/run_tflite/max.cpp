//
// Created by phil on 17/05/18.
//

#include "max.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include <cstdio>
#include <cmath>


TfLiteStatus MaxPrepare(TfLiteContext *context, TfLiteNode *node) {
    using namespace tflite;
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    const TfLiteTensor* input = GetInput(context, node, 0);
    TfLiteTensor* output = GetOutput(context, node, 0);

    int num_dims = NumDimensions(input);

    TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
    for (int i=0; i<num_dims; ++i) {
        output_size->data[i] = input->dims->data[i];
    }

    return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus MaxEval(TfLiteContext *context, TfLiteNode *node) {
    using namespace tflite;
    const TfLiteTensor* input = GetInput(context, node,0);
    TfLiteTensor* output = GetOutput(context, node,0);

    float* input_data = input->data.f;
    float* output_data = output->data.f;

    size_t count = 1;
    int num_dims = NumDimensions(input);
    for (int i = 0; i < num_dims; ++i) {
        count *= input->dims->data[i];
    }

    for (size_t i=0; i<count; ++i) {
        output_data[i] = (float)exp(input_data[i]);
    }
    return kTfLiteOk;
}

TfLiteRegistration* Register_MAX() {
    static TfLiteRegistration r = {nullptr, nullptr, MaxPrepare, MaxEval};
    return &r;
}